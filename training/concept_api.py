"""OpenAPI tool server for the short-drama concept pipeline, for Open WebUI (or any OpenAPI tool client).

    ComfyUI\\.venv\\Scripts\\python.exe training\\concept_api.py        -> http://127.0.0.1:7863

Each endpoint is one tool: list stories and scenes, queue local Z-Image candidates, record the user's review,
queue MiniMax H3 clips for approved scenes, join the clips. Everything long is a queued job that one worker
thread runs, one at a time; the tools answer at once with a job id and `get_job` reports progress.

What this layer must not loosen:
- A chat model must not spend GPU time or Colab compute units on its own: the two spending tools answer with an
  estimate until they are called again with confirm=true.
- It must not pick images: `review_scene` records a decision the user made. The server cannot verify who decided,
  so the tool description says it plainly and the record is tagged decided_via=open-webui.
- H3 only gets APPROVED masters, and its prompt screen is run.py's (no bypass here).
- The chat model shares the 8 GB card with ComfyUI: gpu_guard unloads it before the batch and before each image.

User-visible strings are Traditional Chinese (the model relays them); tool descriptions are English because the
model reads those to decide which tool to call.
"""

import contextlib
import datetime
import hmac
import html as html_lib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

import comfyui_client as client
import concept_colab as cc
import concept_cut
import concept_stills as cs
import generate_character as gc
import gpu_guard
import resource_policy as rp

REPO_ROOT = cs.REPO_ROOT
OUTPUT_ROOT = os.path.join(REPO_ROOT, "output")
JOB_DIR = os.path.join(REPO_ROOT, "outputs", "concept_jobs")
H3_SCRIPTS = os.path.join(REPO_ROOT, ".claude", "skills", "minimax-h3-colab", "scripts")
HOST = os.environ.get("CONCEPT_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("CONCEPT_API_PORT", "7863"))
# The address the BROWSER uses for images and clips (the chat renders markdown links client-side).
PUBLIC_URL = os.environ.get("CONCEPT_API_PUBLIC_URL", f"http://localhost:{PORT}")
CORS_ORIGINS = [o for o in os.environ.get("CONCEPT_API_CORS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o]
FILE_EXTENSIONS = (".png", ".jpg", ".mp4", ".html")

IMAGE_STALL_SECONDS = 600           # a longer image is a hang (normal: 100-330 s)
MAX_COMFYUI_RESTARTS = 3
MAX_CLIPS_PER_SESSION = 4            # the conservative split from the story 1 report (4+3)
COOLDOWN_SECONDS_PER_IMAGE = 200     # an estimate until a gated batch has been measured: ~80 C back to 60 C
# Z-Image at 720x1280: 131-141 s on the old RTX 2070; on the RTX 5060 Ti (driver 617.42) 36.5 s warm, 56.7 s for the
# first image after ComfyUI starts (2026-10-17). COOLDOWN_SECONDS_PER_IMAGE above is still the 2070 figure: the new
# card peaks near 69 C, so the 60 C gate probably waits far less, but that has not been measured.
SECONDS_PER_IMAGE = 40
# Qwen-Image 2.1 on the RTX 5060 Ti: 62.5 s warm at cfg 1; the pipeline runs it at cfg 2.0 (two passes per step), so
# roughly double. An estimate until a chat-triggered batch has been measured.
QWEN_SECONDS_PER_IMAGE = 125
# Local first. A stand-in program holding 4.4, 10.1 and 14.3 GB of the 16 GB card never made a Z-Image render fail
# (2026-10-17, RTX 5060 Ti): ComfyUI moves part of the weights to system RAM and the render just slows down - bf16 at
# 720x1280 took 50 s, 63 s and 75 s from cold against 48 s on a free card, 576x1024 took 38 s at 6 GB free, 432x768 32 s
# (and int8 432x768 28 s at 2 GB free). So memory shortage is answered by a smaller picture first, which is faster, and
# only a practically full card (nothing below 2 GB free was measured) is sent to Colab. The tiers are the project's canvas,
# 80% of it and 60% of it (720x1280 -> 576x1024 -> 432x768) and the free-memory limits are read off those three runs.
RESOLUTION_TIERS = ((9500, 1.0, "全解析度"), (5000, 0.8, "中解析度"), (0, 0.6, "小解析度"))
LOCAL_MIN_FREE_MIB = 1500
REDUCED_SECONDS_PER_IMAGE = 45       # 576x1024 / 432x768 while the card is partly held: 32-38 s measured, with margin
# The Colab worker (.claude/skills/z-image-colab) renders Z-Image only, on an L4: 18 s per 720x1280 image; a session costs
# 9-15 minutes of fixed time (install or model copy, idle timeout) and measured 0.24 CU (8 jobs) and 0.40 CU (3 jobs).
COLAB_SESSION_MINUTES = 10
COLAB_SECONDS_PER_IMAGE = 18
COLAB_CU_PER_HOUR = 1.54
COLAB_WAIT_SECONDS = 45 * 60         # a session that has not delivered by then is stopped
COLAB_POLL_SECONDS = 10
ZIMAGE_SCRIPTS = os.path.join(REPO_ROOT, ".claude", "skills", "z-image-colab", "scripts")
ZIMAGE_TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "INVALID_OUTPUT")
SESSION_OVERHEAD_MINUTES = 6.5       # measured: fresh Colab session, models downloaded
INFERENCE_SECONDS_PER_VIDEO_SECOND = 39.5   # measured: 8 s clip ~315 s
CU_PER_HOUR = 5.2                    # measured average; batches ranged 3.1-6.1

QUEUED, RUNNING, WAITING, COMPLETED, FAILED, CANCELLED, INTERRUPTED = (
    "queued", "running", "waiting_for_user", "completed", "failed", "cancelled", "interrupted")
ACTIVE = (QUEUED, RUNNING, WAITING)


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def file_url(path):
    rel = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
    return f"{PUBLIC_URL}/files/{rel}"


def _scaled(width, height, factor):
    """The canvas scaled to a multiple of 16 on both sides (what the image models need): 720x1280 * 0.8 = 576x1024."""
    return max(16, round(width * factor / 16) * 16), max(16, round(height * factor / 16) * 16)


def local_capacity(image_model, canvas, guard=None):
    """What THIS machine's card can do once the chat model is unloaded, and the picture size to use. Video memory held by
    anything other than Ollama (a game, another render, the desktop) stays; what Ollama holds is released before the
    batch starts. An unreadable nvidia-smi is not a reason to refuse: the batch's own guard still decides."""
    guard = guard or gpu_guard
    state = guard.gpu_state()
    if state is None:
        return {"ok": True, "known": False, "plan": None}
    chat = guard.loaded_vram_mib()
    others = max(0.0, state["used_mib"] - chat)
    free = state["total_mib"] - others
    plan = None
    if image_model == "z_image":      # only this model was measured at smaller sizes
        for minimum, factor, label in RESOLUTION_TIERS:
            if free >= minimum:
                plan = {"label": label, "factor": factor, "size": _scaled(canvas[0], canvas[1], factor),
                        "reduced": factor < 1.0}
                break
    return {"ok": free >= LOCAL_MIN_FREE_MIB, "known": True, "total_mib": round(state["total_mib"]),
            "held_by_others_mib": round(others), "chat_model_mib": round(chat), "free_after_unload_mib": round(free),
            "plan": plan}


def choose_backend(requested, image_model, canvas=(720, 1280)):
    """('local' | 'colab', capacity, reason). `auto` is local; the size shrinks when the card is partly held, and only a
    practically full card is sent to Colab. Colab can only draw Z-Image, so a Qwen render on a full card is refused."""
    cap = local_capacity(image_model, canvas)
    if requested == "colab":
        chosen, reason = "colab", "使用者指定用 Colab"
    elif requested == "local" or cap["ok"]:
        return "local", cap, None
    else:
        chosen = "colab"
        reason = (f"本機顯示卡幾乎全滿：卸載聊天模型後只剩 {cap['free_after_unload_mib']} MiB 可用"
                  f"（其他程式佔了 {cap['held_by_others_mib']} MiB），低於 {LOCAL_MIN_FREE_MIB} MiB 就沒有實測過能不能生")
    if chosen == "colab" and image_model != "z_image":
        raise HTTPException(
            status_code=409 if requested == "auto" else 422,
            detail=("Qwen-Image 2.1 沒有雲端版本，" + (reason + "。請先關掉佔用顯示卡的程式，或改用 Z-Image"
                                                      if requested == "auto" else "不能用 Colab")))
    return chosen, cap, reason


# ---------------------------------------------------------------- jobs

class JobStore:
    """One JSON file per job. Survives a restart, but nothing is resumed: a job that was mid-run when the server
    stopped is marked interrupted (re-running could spend GPU time or compute units nobody asked for twice)."""

    def __init__(self, directory):
        self.dir = directory
        self.lock = threading.Lock()

    def _path(self, job_id):
        return os.path.join(self.dir, f"{job_id}.json")

    def save(self, job):
        with self.lock:
            cs._write_json(self._path(job["job_id"]), job)
        return job

    def get(self, job_id):
        path = self._path(job_id)
        if not (job_id.replace("-", "").isalnum() and os.path.isfile(path)):
            return None
        return cs._read_json(path)

    def all(self):
        if not os.path.isdir(self.dir):
            return []
        jobs = [cs._read_json(os.path.join(self.dir, n)) for n in os.listdir(self.dir) if n.endswith(".json")]
        return sorted(jobs, key=lambda j: j.get("order", 0), reverse=True)   # created_at only has seconds

    def mark_interrupted(self):
        changed = []
        for job in self.all():
            if job["status"] in ACTIVE:
                job.update(status=INTERRUPTED, finished_at=_now(), needs_user_action=None,
                           message="伺服器重新啟動時這個工作還沒做完；沒有自動重跑，需要的話請重新排入")
                self.save(job)
                changed.append(job["job_id"])
        return changed


class Runner:
    """Runs queued jobs one at a time. The side-effecting pieces are attributes so tests can replace them."""

    def __init__(self, store, root=cs.PROJECT_ROOT):
        self.store = store
        self.root = root
        self.queue = queue.Queue()
        self.cancel_requested = set()
        self.thread = None
        self.submit = None                      # None -> concept_stills' default (real ComfyUI)
        self.server_up = client.is_server_running
        self.start_comfyui = lambda: client.start_server(extra_args=["--disable-smart-memory"])
        self.popen = subprocess.Popen
        self.wait_exit = lambda proc: proc.wait(timeout=30) if hasattr(proc, "wait") else None
        self.run = subprocess.run
        self.guard = gpu_guard
        self.assemble = concept_cut.assemble
        self.thermal_log = gpu_guard.ThermalLog
        self.sleep = time.sleep
        self.clock = time.monotonic
        self.zimage_jobs_dir = cc.ZIMAGE_JOBS_DIR                 # where the z-image-colab queue keeps its job files
        self.zimage_sessions_log = os.path.join(REPO_ROOT, "output", "zimage", "zimage_sessions.jsonl")

    # -- lifecycle
    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while True:
            self.run_next(block=True)

    def enqueue(self, kind, story, params, total):
        job = {"job_id": f"{kind}-{datetime.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}", "kind": kind,
               "story_id": story.id, "story": story.number, "status": QUEUED, "created_at": _now(),
               "order": time.time_ns(),
               "started_at": None, "finished_at": None, "progress": {"done": 0, "total": total, "current": None},
               "needs_user_action": None, "message": "已排入佇列", "params": params, "result": None, "error": None}
        self.store.save(job)
        self.queue.put(job["job_id"])
        return job

    def run_next(self, block=False):
        try:
            job_id = self.queue.get(block=block)
        except queue.Empty:
            return None
        job = self.store.get(job_id)
        if job is None or job["status"] != QUEUED:
            return job
        job.update(status=RUNNING, started_at=_now(), message="執行中")
        self.store.save(job)
        try:
            result = getattr(self, f"_run_{job['kind']}")(job)
            job = self.store.get(job_id)
            job.update(status=CANCELLED if job_id in self.cancel_requested else COMPLETED, result=result,
                       message="已取消（已完成的部分保留）" if job_id in self.cancel_requested else "完成")
        except (gc.UsageError, gpu_guard.GpuBusy, RuntimeError, TimeoutError, OSError,
                subprocess.SubprocessError) as exc:
            job = self.store.get(job_id)
            job.update(status=FAILED, error=f"{type(exc).__name__}: {exc}", message="失敗")
        job.update(finished_at=_now(), needs_user_action=None)
        self.cancel_requested.discard(job_id)
        return self.store.save(job)

    def _update(self, job_id, **fields):
        job = self.store.get(job_id)
        progress = fields.pop("progress", None)
        if progress:
            job["progress"].update(progress)
        job.update(fields)
        return self.store.save(job)

    def _story(self, job):
        return cs.select_stories(cs.load_stories(self.root), job["story_id"])[0]

    # -- local Z-Image candidates
    def _run_images(self, job):
        if job["params"].get("backend") == "colab":
            return self._run_colab_images(job)
        job_id, params = job["job_id"], job["params"]
        story = self._story(job)
        bible = cs.load_bible(self.root)
        settings = rp.load_settings()
        self.guard.before_batch(max_start_temp_c=rp.start_temp_c(settings))
        process = None
        if not self.server_up():
            self._update(job_id, message="啟動 ComfyUI")
            process = self.start_comfyui()
        letters = tuple(params["letters"])
        scenes = set(params["scenes"]) if params["scenes"] else None

        def remaining():
            return len(cs.pending([story], letters, scene_numbers=scenes))

        to_make = remaining()
        restarts = 0
        previous_limit = client.POLL_TIMEOUT_SECONDS_ZIMAGE
        # An image normally takes 100-330 s. A longer one is the "Model Initializing" hang (seen three times on
        # 2026-10-04: sampling stuck at step 0/8 with 6 GB of VRAM spilled to shared memory, ComfyUI's HTTP
        # server no longer answering), which only a restart clears. The library default would wait 20 minutes.
        client.POLL_TIMEOUT_SECONDS_ZIMAGE = IMAGE_STALL_SECONDS

        thermal = self.thermal_log(os.path.join(REPO_ROOT, "outputs", "thermal", f"{job_id}.csv")).start()
        waited = {"s": 0.0}

        def before_each(_story, scene, letter):
            self.guard.before_image()
            self._update(job_id, message="等顯示卡降溫", progress={"current": f"scene_{scene:02d} 候選 {letter}"})
            waited["s"] += self.guard.wait_until_cool()["waited_seconds"]
            thermal.mark(f"start scene_{scene:02d} {letter}")
            self._update(job_id, message="生圖中",
                         progress={"done": to_make - remaining(), "current": f"scene_{scene:02d} 候選 {letter}"})

        try:
            while True:
                try:
                    cs.generate(bible, [story], letters=letters, scene_numbers=scenes, submit=self.submit,
                                server_up=self.server_up, before_each=before_each,
                                should_stop=lambda: job_id in self.cancel_requested,
                                image_model=params.get("image_model", cs.DEFAULT_IMAGE_MODEL),
                                canvas=tuple(params["canvas"]) if params.get("canvas") else None)
                    break
                except (TimeoutError, client.ComfyUIUnavailable) as exc:
                    # Only a ComfyUI this job started can be restarted; someone else's is not ours to kill.
                    if process is None or restarts >= MAX_COMFYUI_RESTARTS:
                        raise
                    restarts += 1
                    self._update(job_id, message=f"ComfyUI 卡住（{type(exc).__name__}），重新啟動第 {restarts} 次",
                                 progress={"done": to_make - remaining(), "current": None})
                    process.terminate()
                    self.wait_exit(process)
                    self.guard.before_batch(max_start_temp_c=rp.start_temp_c(settings))
                    process = self.start_comfyui()   # finished images are skipped, so this resumes
        finally:
            client.POLL_TIMEOUT_SECONDS_ZIMAGE = previous_limit
            temperature = dict(thermal.stop(), cooldown_wait_seconds=round(waited["s"]))
            if process is not None:      # only a ComfyUI this job started; never someone else's
                process.terminate()
            cs.build_index(cs.load_stories(self.root), self.root)
            cs.build_sheet(bible, cs.load_stories(self.root), self.root)
        made = to_make - remaining()
        self._update(job_id, progress={"done": made, "current": None})
        return {"images": made, "comfyui_restarts": restarts, "temperature": temperature,
                "review_sheet": file_url(os.path.join(self.root, "review_sheet.html")),
                "next": "請呼叫 show_images（newest_only=true）把新圖顯示在聊天裡，不要用文字描述圖片；"
                        "使用者看圖後再決定每個 Scene 核准哪一張，或判定重生"}

    # -- the same Z-Image candidates on the Colab worker, for when the local card cannot hold the render
    def _zimage_script(self, name, *args):
        env = dict(os.environ, ZIMG_MODEL_SOURCE="download", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        return self.run([sys.executable, os.path.join(ZIMAGE_SCRIPTS, name), *args], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", env=env, cwd=REPO_ROOT)

    @staticmethod
    def _json_tail(done):
        for line in reversed((done.stdout or "").strip().splitlines()):
            if line.strip().startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    break
        try:
            return json.loads(done.stdout)       # status.py prints one indented object
        except (ValueError, TypeError):
            return {}

    def _run_colab_images(self, job):
        """export -> submit (free) -> worker up (spends CU) -> wait -> import. Nothing here starts before the user said
        yes to the estimate, and a failed pre-flight stops before any compute unit is spent."""
        job_id, params = job["job_id"], job["params"]
        story = self._story(job)
        bible = cs.load_bible(self.root)
        letters = tuple(params["letters"])
        scenes = set(params["scenes"]) if params["scenes"] else None

        pre = self._zimage_script("setup_colab.py", "preflight")
        if pre.returncode != 0:
            raise RuntimeError("Colab 開始前的檢查沒過（沒有花 CU）：" + ((pre.stdout or pre.stderr).strip()[-400:]))
        out = cc.export(bible, story, letters=letters, scene_numbers=scenes, batch=job_id)
        sub = self._zimage_script("submit.py", "--manifest", out["manifest"])
        if sub.returncode != 0:
            raise RuntimeError("送進 Colab 佇列失敗（沒有花 CU）：" + ((sub.stdout or sub.stderr).strip()[-400:]))
        up = self._zimage_script("worker.py", "up")
        started = self._json_tail(up)
        if up.returncode != 0 or not started.get("ok", True):
            raise RuntimeError("Colab worker 沒有啟動：" + ((up.stdout or up.stderr).strip()[-400:]))
        total = out["images"]
        self._update(job_id, message="Colab 啟動中（約 4～14 分鐘，首張之後每張約 20 秒）", progress={"total": total})
        deadline = self.clock() + COLAB_WAIT_SECONDS
        waiting, dead_polls = False, 0
        while True:
            jobs = cc.find_jobs(job_id, self.zimage_jobs_dir)
            finished = [j for j in jobs if j["status"] in ZIMAGE_TERMINAL]
            self._update(job_id, progress={"done": sum(1 for j in jobs if j["status"] == "COMPLETED")})
            if len(finished) >= total:
                break
            if job_id in self.cancel_requested:
                self._zimage_script("worker.py", "stop")       # ends the session: it must not keep billing
                break
            if self.clock() >= deadline:
                self._zimage_script("worker.py", "stop")
                raise TimeoutError(f"Colab 超過 {COLAB_WAIT_SECONDS // 60} 分鐘沒有交出圖，已要求結束這個 session")
            status = self._json_tail(self._zimage_script("status.py", "--local"))
            controller = status.get("controller") or {}
            if controller.get("state") == "drive_consent":
                waiting = True
                self._update(job_id, status=WAITING, message="等使用者按 Google Drive 同意",
                             needs_user_action="請在瀏覽器按 Google Drive 的同意，按完後說一聲，再呼叫 consent_done")
            elif waiting:
                waiting = False
                self._update(job_id, status=RUNNING, needs_user_action=None, message="Colab 生圖中")
            dead_polls = 0 if status.get("controller_live", True) else dead_polls + 1
            if dead_polls >= 3:
                break                                          # the session ended with jobs left over; import what exists
            self.sleep(COLAB_POLL_SECONDS)
        report = cc.import_batch(bible, story, job_id, jobs_dir=self.zimage_jobs_dir)
        cs.build_index(cs.load_stories(self.root), self.root)
        cs.build_sheet(bible, cs.load_stories(self.root), self.root)
        if not report["imported"] and not report["skipped"]:
            raise RuntimeError("Colab 沒有交出任何圖：" + json.dumps(report, ensure_ascii=False)[:300])
        cu = None
        try:
            with open(self.zimage_sessions_log, encoding="utf-8") as f:
                last = [json.loads(line) for line in f if line.strip()][-1]
            cu = last.get("cu_used_measured")
        except (OSError, ValueError, IndexError):
            pass
        return {"backend": "colab", "images": len(report["imported"]), "imported": report["imported"],
                "failed": report["failed"] + report["missing"], "cu_used": cu,
                "next": "請呼叫 show_images（newest_only=true）把新圖顯示在聊天裡；Colab 用官方 bf16 權重，"
                        "同一個種子的畫面跟本機不會逐像素一樣"}

    # -- MiniMax H3 clips on Colab
    def _run_videos(self, job):
        job_id, params = job["job_id"], job["params"]
        story = self._story(job)
        batch_dir = os.path.join(story.dir, "videos", "_batches")   # the manifests sent to run.py
        os.makedirs(batch_dir, exist_ok=True)
        env = dict(os.environ, H3_MODEL_SOURCE="download", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        check = self.run([sys.executable, os.path.join(H3_SCRIPTS, "check.py")], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env, cwd=REPO_ROOT)
        if check.returncode != 0:
            raise RuntimeError("H3 開始前的檢查沒過（沒有花 CU）：" + (check.stdout or check.stderr).strip()[-400:])
        scenes = params["scenes"]
        batches = [scenes[i:i + MAX_CLIPS_PER_SESSION] for i in range(0, len(scenes), MAX_CLIPS_PER_SESSION)]
        records, cu = [], 0.0
        for index, batch in enumerate(batches, 1):
            if job_id in self.cancel_requested:
                break
            manifest = os.path.join(batch_dir, f"{job_id}_{index}.json")
            cs._write_json(manifest, video_manifest(story, batch, f"{job_id}-{index}"))
            self._update(job_id, status=RUNNING, message=f"H3 第 {index}/{len(batches)} 個 session",
                         progress={"current": f"session {index}"})
            summary = self._h3_session(job_id, manifest, env)
            cu += summary.get("cu_used_settled") or 0.0
            for rec in summary.get("records", []):
                records.append({"scene_id": rec.get("scene_id"), "status": rec.get("status"),
                                "error": rec.get("error"), "output": rec.get("output")})
                if rec.get("status") == "COMPLETED" and rec.get("output"):
                    record_clip(story, int(str(rec["scene_id"])[-2:]), rec["output"], job_id)
            self._update(job_id, progress={"done": sum(1 for r in records if r["status"] == "COMPLETED")})
        ok = [r for r in records if r["status"] == "COMPLETED"]
        return {"clips_completed": len(ok), "clips_failed": len(records) - len(ok), "cu_used": round(cu, 2),
                "clips": [dict(r, url=file_url(r["output"]) if r["status"] == "COMPLETED" else None) for r in records]}

    def _h3_session(self, job_id, manifest, env):
        """One run.py --manifest call = one Colab session. stderr carries the state lines; the last stdout line
        is the batch summary. The process is never killed from here: run.py stops its own Colab session on every
        exit path, and killing it could leave an A100 billing."""
        proc = self.popen([sys.executable, os.path.join(H3_SCRIPTS, "run.py"), "--manifest", manifest],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                          errors="replace", env=env, cwd=REPO_ROOT)
        out_lines = []
        reader = threading.Thread(target=lambda: out_lines.extend(proc.stdout), daemon=True)
        reader.start()
        for line in proc.stderr:
            line = line.strip()
            if "DRIVE_CONSENT_NEEDED" in line:
                self._update(job_id, status=WAITING, message="等使用者按 Google Drive 同意",
                             needs_user_action="請在瀏覽器按 Google Drive 的同意，按完後說一聲，再呼叫 consent_done")
            elif "Google Drive mounted" in line or "] CONNECTING_COLAB -> " in line:
                self._update(job_id, status=RUNNING, needs_user_action=None, message="H3 執行中")
            elif "] finished: " in line or " -> " in line:
                self._update(job_id, message=line[-160:])
        proc.wait()
        reader.join(timeout=5)
        last = next((x for x in reversed(out_lines) if x.strip().startswith("{")), None)
        if last is None:
            raise RuntimeError(f"H3 沒有回傳結果（exit code {proc.returncode}）")
        return json.loads(last)

    # -- the cut
    def _run_cut(self, job):
        story = self._story(job)
        clips = load_clips(story)
        ordered = [{"path": os.path.join(REPO_ROOT, clips[f"scene_{s['scene']:02d}"]["path"]),
                    "ambience": bool(s.get("video", {}).get("ambience"))} for s in story.data["scenes"]]
        out = next_version(story.cut_dir(), f"{story.id}_cut")
        result = self.assemble(ordered, out, subtitle=job["params"].get("subtitle"))
        return dict(result, url=file_url(out))


def clips_path(story):
    return os.path.join(story.dir, "clips.json")


def load_clips(story):
    return cs._read_json(clips_path(story)) if os.path.isfile(clips_path(story)) else {}


def record_clip(story, scene_number, output, job_id):
    """Remember which clip a scene's cut uses: the newest completed one."""
    clips = load_clips(story)
    clips[f"scene_{scene_number:02d}"] = {"path": os.path.relpath(output, REPO_ROOT).replace("\\", "/"),
                                          "job_id": job_id, "made_at": _now()}
    cs._write_json(clips_path(story), clips)


def next_version(directory, stem, extension=".mp4"):
    """directory/<stem>_vN.mp4 with N one above the highest that exists: clips and cuts are never overwritten."""
    os.makedirs(directory, exist_ok=True)
    taken = [int(m.group(1)) for name in os.listdir(directory)
             if (m := re.fullmatch(re.escape(stem) + r"_v(\d+)" + re.escape(extension), name))]
    return os.path.join(directory, f"{stem}_v{max(taken, default=0) + 1}{extension}")


def video_manifest(story, scene_numbers, batch_id):
    """The run.py manifest for these scenes: first frame = the scene's approved master.png."""
    jobs = []
    for n in scene_numbers:
        video = story.scene(n)["video"]
        jobs.append({
            "job_id": f"{batch_id}-sc{n:02d}", "scene_id": f"{story.id}/scene_{n:02d}",
            "input": {"image": os.path.join(story.scene_dir(n), "master.png")},
            "prompt": {"description": video["description"], "soundscape": video.get("soundscape", ""),
                       "constraints": list(video.get("constraints", []))},
            "settings": {"duration": video.get("duration", cs.VIDEO_DEFAULT_SECONDS)},
            "output": next_version(story.clip_dir(n), "clip"),
        })
    return {"batch_id": batch_id, "jobs": jobs}


def video_problems(story, scene_numbers):
    """Why these scenes cannot go to H3 yet, as a list of messages (empty = ready)."""
    problems = []
    for n in scene_numbers:
        review = cs.load_review(story, n)
        if review["status"] != cs.APPROVED or not os.path.isfile(os.path.join(story.scene_dir(n), "master.png")):
            problems.append(f"Scene {n} 還沒有核准的 master（目前 {review['status']}）")
        if "video" not in story.scene(n):
            problems.append(f"Scene {n} 還沒有影片描述（先用 set_video_prompt）")
    return problems


def video_estimate(story, scene_numbers):
    sessions = math.ceil(len(scene_numbers) / MAX_CLIPS_PER_SESSION)
    minutes = sessions * SESSION_OVERHEAD_MINUTES
    for n in scene_numbers:
        seconds = story.scene(n).get("video", {}).get("duration", cs.VIDEO_DEFAULT_SECONDS)
        minutes += (seconds * INFERENCE_SECONDS_PER_VIDEO_SECOND + 15) / 60
    return {"clips": len(scene_numbers), "sessions": sessions, "minutes": round(minutes),
            "compute_units": round(minutes / 60 * CU_PER_HOUR, 1)}


# ---------------------------------------------------------------- app

def _require_token(authorization: str | None = Header(default=None)):
    token = os.environ.get("CONCEPT_API_TOKEN")
    if not token:
        return
    given = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(given, token):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


store = JobStore(JOB_DIR)
runner = Runner(store)


@contextlib.asynccontextmanager
async def _lifespan(_app):
    interrupted = store.mark_interrupted()
    if interrupted:
        print(f"[concept_api] {len(interrupted)} 個未完成的工作標成 interrupted，沒有自動重跑", flush=True)
    runner.start()
    yield


app = FastAPI(
    title="Short-drama concept pipeline",
    version="1.0.0",
    description=("Tools for five short-drama stories (numbered 1-5), 7 scenes each: Z-Image candidate stills -> the "
                 "USER reviews -> MiniMax H3 clips for approved scenes -> one cut. Rules: never choose or approve an "
                 "image yourself; before generate_candidates or make_videos runs, show the estimate to the user and "
                 "call again with confirm=true only after they say yes; when a job has needs_user_action, tell the "
                 "user exactly that. Answer the user in Traditional Chinese."),
    lifespan=_lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
AUTH = [Depends(_require_token)]


def _story(number):
    stories = cs.load_stories(runner.root)
    try:
        story = cs.select_stories(stories, number)[0]
        cs.validate_story(story)
        return story
    except gc.UsageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


def _scene(story, number):
    try:
        return story.scene(number)
    except gc.UsageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


def _busy():
    return [j for j in store.all() if j["status"] in ACTIVE]


def _job_view(job):
    return {k: job[k] for k in ("job_id", "kind", "story", "status", "message", "progress", "needs_user_action",
                                "result", "error", "created_at", "started_at", "finished_at")}


@app.get("/stories", operation_id="list_stories", dependencies=AUTH,
         summary="List the five stories with how many scenes are in each review status and how many images exist.")
def list_stories():
    out = []
    for story in cs.load_stories(runner.root):
        counts = {}
        for s in story.data["scenes"]:
            status = cs.load_review(story, s["scene"])["status"]
            counts[status] = counts.get(status, 0) + 1
        out.append({"story": story.number, "title": story.data.get("title_zh", story.data["title"]),
                    "genre": story.data.get("genre"), "hook": story.data.get("hook"),
                    "images": sum(len(cs.existing_letters(story, s["scene"])) for s in story.data["scenes"]),
                    "clips": len(load_clips(story)), "scene_status": counts})
    return {"stories": out}


@app.get("/stories/{story}/scenes", operation_id="list_scenes", dependencies=AUTH,
         summary="Show one story's 7 scenes: plot, review status, candidate images (as markdown to show the user), "
                 "text that must be legible in the image, and whether a clip exists.")
def list_scenes(story: int):
    st = _story(story)
    clips = load_clips(st)
    scenes = []
    for s in st.data["scenes"]:
        n = s["scene"]
        review = cs.load_review(st, n)
        candidates = []
        for letter in cs.existing_letters(st, n):
            url = file_url(cs.candidate_paths(st, n, letter)["png"])
            number = cs.LETTERS.index(letter) + 1
            entry = {"candidate": number, "url": url, "markdown": f"![Scene {n} 候選 {number}]({url})"}
            try:
                meta = cs._read_json(cs.candidate_paths(st, n, letter)["meta"])
            except (OSError, ValueError):
                meta = {}
            if meta.get("image_model") and meta["image_model"] != cs.DEFAULT_IMAGE_MODEL:
                entry["image_model"] = cs.IMAGE_MODELS.get(meta["image_model"], {}).get("label", meta["image_model"])
            if meta.get("license"):
                entry["license_warning"] = f"授權 {meta['license']}：只能內部測試，不要放進商用流程"
            candidates.append(entry)
        clip = clips.get(f"scene_{n:02d}")
        scenes.append({"scene": n, "title": s["title"], "plot": s["beat_zh"], "status": review["status"],
                       "approved_candidate": review.get("approved_candidate"), "notes": review.get("notes") or None,
                       "text_to_verify": s["on_screen_text"] or None, "candidates": candidates,
                       "has_video_prompt": "video" in s,
                       "clip_url": file_url(os.path.join(REPO_ROOT, clip["path"])) if clip else None})
    return {"story": st.number, "title": st.data.get("title_zh"), "scenes": scenes,
            "review_checklist": cs.load_bible(runner.root)["review_checklist"]}


def _gallery_page(story, picked, note=""):
    """A self-contained HTML page of candidate pictures. Open WebUI embeds an external tool's text/html answer that
    carries Content-Disposition: inline in the chat as a frame, so the user SEES the pictures instead of a chat model's
    description of them. The page only references /files/ URLs (read-only, confined) and needs no script."""
    esc = html_lib.escape
    figures = []
    for entry in picked:
        label = f"Scene {entry['scene']} 候選 {entry['candidate']}"
        extra = f" · {esc(entry['image_model'])}" if entry.get("image_model") else ""
        if entry.get("reduced"):
            extra += f" · {esc(entry['reduced'])}（降低過解析度）"
        warn = f"<div class='w'>{esc(entry['license_warning'])}</div>" if entry.get("license_warning") else ""
        figures.append(
            f"<figure><a href='{esc(entry['url'])}' target='_blank' rel='noopener'><img src='{esc(entry['url'])}' "
            f"alt='{esc(label)}' loading='lazy'></a><figcaption><b>{esc(label)}</b>{extra}<br>"
            f"<span class='s'>{esc(entry['title'])} · {esc(entry['status'])}</span>{warn}</figcaption></figure>")
    body = "".join(figures) or "<p>這個範圍還沒有候選圖。</p>"
    return (
        "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content="
        "'width=device-width,initial-scale=1'><style>body{margin:0;padding:8px;font:13px/1.45 'Microsoft JhengHei',"
        "sans-serif;color:#222;background:transparent}.t{font-weight:600;margin:0 0 8px}.g{display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}figure{margin:0;border:1px solid #d0d5dd;"
        "border-radius:8px;overflow:hidden;background:#fff}img{display:block;width:100%;height:auto}"
        "figcaption{padding:6px 8px}.s{color:#667085}.w{margin-top:4px;color:#9a5b00;font-size:12px}</style></head>"
        f"<body><p class='t'>{esc(story.data.get('title_zh') or story.data['title'])}{esc(note)}</p>"
        f"<div class='g'>{body}</div></body></html>")


@app.get("/stories/{story}/gallery", operation_id="show_images", dependencies=AUTH, response_class=HTMLResponse,
         summary="SHOW the candidate pictures of a story to the user as an image gallery inside the chat (all scenes, or "
                 "one scene; newest_only=true shows just the latest picture of each scene, e.g. right after a "
                 "generation finished). Call this whenever the user wants to see pictures. The pictures appear by "
                 "themselves: do not describe them or paste links, just say that they are shown.")
def show_images(story: int, scene: int | None = None, newest_only: bool = False):
    st = _story(story)
    scenes = [_scene(st, scene)] if scene else st.data["scenes"]
    picked = []
    for s in scenes:
        n = s["scene"]
        review = cs.load_review(st, n)
        found = []
        for letter in cs.existing_letters(st, n):
            paths = cs.candidate_paths(st, n, letter)
            try:
                meta = cs._read_json(paths["meta"])
            except (OSError, ValueError):
                meta = {}
            entry = {"scene": n, "candidate": cs.LETTERS.index(letter) + 1, "url": file_url(paths["png"]),
                     "title": s["title"], "status": review["status"], "mtime": os.path.getmtime(paths["png"])}
            if meta.get("image_model") and meta["image_model"] != cs.DEFAULT_IMAGE_MODEL:
                entry["image_model"] = cs.IMAGE_MODELS.get(meta["image_model"], {}).get("label", meta["image_model"])
            if meta.get("license"):
                entry["license_warning"] = f"授權 {meta['license']}：只能內部測試，不要放進商用流程"
            if meta.get("resolution_reduced_from"):
                entry["reduced"] = f"{meta.get('width')}×{meta.get('height')}"
            found.append(entry)
        if newest_only and found:
            found = [max(found, key=lambda e: e["mtime"])]
        picked.extend(found)
    note = "（每個 Scene 最新的一張）" if newest_only else ""
    return HTMLResponse(_gallery_page(st, picked, note), headers={"Content-Disposition": "inline"})


class GenerateBody(BaseModel):
    scenes: list[int] | None = Field(default=None, description="Scene numbers 1-7; omit for all seven.")
    candidates: list[int] = Field(default=[1], description="Which candidate numbers to render: [1] first (the "
                                                       "default - one image per scene), [2] or [1, 2] when the user "
                                                       "wants a second option, [3, 4] only for scenes the user marked "
                                                       "for regeneration.")
    image_model: str = Field(default="z_image", pattern="^(z_image|qwen_image_2_1)$",
                             description="Which picture model draws the images: 'z_image' (the default and the "
                                         "pipeline's own model) or 'qwen_image_2_1' (slower, better lettering; "
                                         "INTERNAL TEST ONLY, non-commercial licence). Use qwen_image_2_1 only when "
                                         "the user asked for Qwen by name.")
    backend: str = Field(default="auto", pattern="^(auto|local|colab)$",
                         description="Where to draw. 'auto' (the default) uses this computer's graphics card and "
                                     "switches to the Google Colab cloud GPU only when the card cannot hold the render; "
                                     "the estimate then says so and states the compute-unit cost. Use 'colab' or 'local' "
                                     "only when the user asked for one by name. Colab draws Z-Image only.")
    confirm: bool = Field(default=False, description="false returns the estimate only. Set true only after the "
                                                     "user has seen the estimate and said yes.")


@app.post("/stories/{story}/candidates", operation_id="generate_candidates", dependencies=AUTH,
          summary="Queue generation of candidate stills for a story: on this computer's GPU first, on the Colab cloud "
                  "GPU when the local card is too full for the render. Without confirm=true it only returns the "
                  "estimate (images, minutes, model, where it will run and why, compute units for Colab, any licence "
                  "warning); nothing is generated and nothing is spent. Tell the user the estimate, including "
                  "when it says Colab.")
def generate_candidates(story: int, body: GenerateBody):
    st = _story(story)
    numbers = sorted(set(body.candidates))
    if not numbers or any(n < 1 or n > len(cs.LETTERS) for n in numbers):
        raise HTTPException(status_code=422, detail="candidates 要是 1-4 的編號")
    letters = tuple(cs.LETTERS[n - 1] for n in numbers)
    try:
        todo = cs.pending([st], letters, scene_numbers=set(body.scenes) if body.scenes else None)
    except gc.UsageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if not todo:
        return {"queued": False, "message": "這些候選圖都已經存在，沒有要生的"}
    model = cs.IMAGE_MODELS[body.image_model]
    bible_canvas = cs.load_bible(runner.root)["canvas"]
    full_size = (bible_canvas["width"], bible_canvas["height"])
    backend, capacity, reason = choose_backend(body.backend, body.image_model, full_size)
    plan = capacity.get("plan") if backend == "local" else None
    canvas = list(plan["size"]) if plan and plan["reduced"] else None
    if backend == "colab":
        minutes = round(COLAB_SESSION_MINUTES + len(todo) * COLAB_SECONDS_PER_IMAGE / 60)
        estimate = {"images": len(todo), "image_model": model["label"], "backend": "Google Colab 雲端 GPU（L4）",
                    "why_not_local": reason, "minutes": minutes,
                    "compute_units": round(minutes / 60 * COLAB_CU_PER_HOUR, 2),
                    "cost_note": "這會花 Colab 運算單元（CU）。實測一個 session 約 0.24～0.40 CU；估計值僅供參考",
                    "user_actions": ["Docker Desktop 要開著", "每個 session 要在瀏覽器按一次 Google Drive 同意"],
                    "note": "Colab 用官方 bf16 權重，同一個種子的畫面跟本機不會逐像素一樣；聊天模型不用卸載"}
    else:
        seconds = QWEN_SECONDS_PER_IMAGE if body.image_model == "qwen_image_2_1" else SECONDS_PER_IMAGE
        if canvas:
            seconds = REDUCED_SECONDS_PER_IMAGE
        estimate = {"images": len(todo), "image_model": model["label"], "backend": "本機顯示卡",
                    "resolution": f"{(canvas or full_size)[0]}×{(canvas or full_size)[1]}",
                    "minutes": round(len(todo) * (seconds + COOLDOWN_SECONDS_PER_IMAGE) / 60),
                    "gpu": "本機顯示卡；每張開始前會等顯示卡降到 60°C 以下，所以比連續跑慢；"
                           "生圖期間聊天模型會被卸載，聊天會讓生圖暫停等待"}
        if capacity.get("known"):
            estimate["local_memory"] = (f"卸載聊天模型後可用 {capacity['free_after_unload_mib']} MiB"
                                        f"（其他程式佔了 {capacity['held_by_others_mib']} MiB）")
        if canvas:
            estimate["resolution_reduced"] = (
                f"顯示卡被別的程式佔著，解析度從 {full_size[0]}×{full_size[1]} 降到 {canvas[0]}×{canvas[1]}"
                f"（{plan['label']}）：實測全解析度在這種情況下會慢到 60～75 秒一張，降解析度約 30～40 秒")
            estimate["resolution_warning"] = ("這批圖會比其他 Scene 的圖小，H3 用它當首幀時畫質可能較差；"
                                              "要全解析度請先關掉佔顯示卡的程式。請把這句話告訴使用者")
    if model["license"]:
        estimate["license_warning"] = (f"這個模型的授權是 {model['license']}：只能內部測試，產出的圖不要放進商用流程。"
                                       "請把這句話告訴使用者")
    if not body.confirm:
        return {"queued": False, "needs_confirmation": True, "estimate": estimate,
                "message": "還沒開始，也沒有花任何資源。請把預估告訴使用者（改用 Colab 時要特別講明會花 CU），"
                           "使用者同意後再用 confirm=true 呼叫一次"}
    if _busy():
        raise HTTPException(status_code=409, detail=f"已有工作在進行：{_busy()[0]['job_id']}，請等它結束")
    job = runner.enqueue("images", st, {"letters": list(letters), "scenes": sorted(set(body.scenes or [])),
                                        "image_model": body.image_model, "backend": backend, "canvas": canvas},
                         len(todo))
    return {"queued": True, "job_id": job["job_id"], "estimate": estimate, "message": "已排入佇列，用 get_job 查進度"}


# A 9B chat model approved scenes on "Scene 3 looks fine" and "the 2nd one seems nicer" and invented story names
# (measured 2026-10-14: 0/3 and 1/3 correct refusals), so the system prompt alone cannot keep the decision with the
# user. review_scene therefore needs the user's own words and checks that they are an explicit decision.
APPROVE_WORDS = ("核准", "批准")
REGENERATE_WORDS = ("重生", "重新生成", "重做", "重跑")
HEDGE_WORDS = ("好像", "也許", "或許", "可能", "看起來", "感覺", "應該", "要不要", "是不是", "比較", "如果", "嗎", "？", "?")
CHINESE_DIGITS = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4}
MAX_QUOTE_CHARS = 200


def _digit(text):
    return CHINESE_DIGITS.get(text) or int(text)


def quote_problem(decision, candidate, scene, quote):
    """Why `quote` is not an explicit decision by the user for this scene, or None. This checks the words, not who
    typed them: a model could still invent a matching quote, which the review record keeps so it can be audited."""
    q = (quote or "").strip()
    if not q:
        return "需要 user_quote：使用者這次說的原話，例如「Scene 3 核准第 2 張」"
    if len(q) > MAX_QUOTE_CHARS:
        return f"user_quote 太長（上限 {MAX_QUOTE_CHARS} 字）：只貼使用者下決定的那句話"
    hedge = next((w for w in HEDGE_WORDS if w in q), None)
    if hedge:
        return f"這句話含「{hedge}」，像是猶豫、比較或提問，不是明確決定。請問使用者要核准第幾張，或要不要重生"
    words = APPROVE_WORDS if decision == "approve" else REGENERATE_WORDS
    if not any(w in q for w in words):
        return f"這句話裡沒有明確的決定用語（{'、'.join(words)}）。使用者沒有明說就不要記錄"
    mentioned = {_digit(a or b) for a, b in re.findall(r"[Ss]cene\s*(\d+)|場景\s*(\d+)", q)}
    if mentioned and mentioned != {scene}:
        return f"這句話提到的是 Scene {'、'.join(map(str, sorted(mentioned)))}，不是 Scene {scene}"
    if decision == "approve":
        picked = {_digit(a or b) for a, b in re.findall(r"第\s*([1-4一二三四兩])\s*[張个]|候選\s*([1-4一二三四兩])", q)}
        if picked != {candidate}:
            return f"這句話沒有明確說出第 {candidate} 張（找到的編號：{sorted(picked) or '無'}）"
    return None


class ReviewBody(BaseModel):
    decision: str = Field(description="'approve' or 'regenerate'.", pattern="^(approve|regenerate)$")
    user_quote: str = Field(description="The user's own words from their latest message that state this decision, copied "
                                        "exactly, e.g. 'Scene 3 核准第 2 張' or 'Scene 3 重生'. Required; rejected if it "
                                        "is hesitant, a comparison or a question.")
    candidate: int | None = Field(default=None, ge=1, le=4, description="For approve: the candidate number the USER "
                                                                        "chose (1-4).")
    notes: str = Field(default="", description="The user's reason, in their words.")


@app.post("/stories/{story}/scenes/{scene}/review", operation_id="review_scene", dependencies=AUTH,
          summary="Record the USER's decision for one scene: approve a candidate (it becomes the master image) or "
                  "mark the scene for regeneration. Call this only when the user explicitly stated the decision and, "
                  "for approve, the candidate number; pass their exact words as user_quote. Remarks such as 'looks "
                  "fine', 'the second one seems nicer' or a question are NOT a decision: ask the user instead. "
                  "Never decide for the user.")
def review_scene(story: int, scene: int, body: ReviewBody):
    st = _story(story)
    _scene(st, scene)
    if body.decision == "approve" and body.candidate is None:
        raise HTTPException(status_code=422, detail="核准需要使用者指定的候選編號（1-4）")
    problem = quote_problem(body.decision, body.candidate, scene, body.user_quote)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    notes = f"{body.notes.strip()} ｜使用者原話：{body.user_quote.strip()}".strip(" ｜")
    try:
        result = cs.review(st, scene, approve=body.candidate if body.decision == "approve" else None,
                           regenerate=body.decision == "regenerate", notes=notes, decided_via="open-webui")
    except gc.UsageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    cs.build_index(cs.load_stories(runner.root), runner.root)
    return {"story": st.number, "scene": scene, "status": result["status"],
            "approved_candidate": result.get("approved_candidate")}


class VideoPromptBody(BaseModel):
    description: str = Field(description="English shot description: what is in the first frame, then the one action, "
                                         "then the camera. Positive statements only.")
    soundscape: str = Field(default="", description="1-2 English sentences of ambient and action sound.")
    duration: float = Field(default=8, ge=4, le=15)
    locked_camera: bool = Field(default=False, description="true only if the user asked for a fixed camera.")
    ambience_only: bool = Field(default=False, description="true for a scene with no speech (kept quieter in the cut).")


@app.put("/stories/{story}/scenes/{scene}/video-prompt", operation_id="set_video_prompt", dependencies=AUTH,
         summary="Save the H3 shot description for one scene. Needed before make_videos for scenes that have none.")
def set_video_prompt(story: int, scene: int, body: VideoPromptBody):
    st = _story(story)
    sc = _scene(st, scene)
    video = {"description": body.description.strip(), "soundscape": body.soundscape.strip(),
             "duration": body.duration, "constraints": ["locked-camera"] if body.locked_camera else [],
             "ambience": body.ambience_only}
    problem = cs.video_problem(video)
    if problem:
        raise HTTPException(status_code=422, detail=f"video {problem}")
    sc["video"] = video
    cs.save_story(st)
    return {"story": st.number, "scene": scene, "saved": True}


# What a chat user may change about a still. Characters, solo and the style stay fixed (they define the story and the
# safety path); only the scene's own picture description can be edited, and never after the scene has its master.
VISUAL_ENGLISH_FIELDS = ("action", "setting", "lighting")
VISUAL_TEXT_FIELDS = VISUAL_ENGLISH_FIELDS + ("on_screen_text", "negative_extra")
MAX_VISUAL_FIELD_CHARS = 700
MAX_VISUAL_QUOTE_CHARS = 300
VISUAL_EDIT_LOG = "visual_edits.jsonl"
_CJK = re.compile(r"[一-鿿]")
_h3 = None
# h3_colab.screen_prompt refuses "15 years old", "15 yo" and "15歲" but lets "aged 15" through; a free-text edit must not
# be able to write that, so this tool adds the phrase itself.
_AGE_PHRASE = re.compile(r"(?<![a-z])(?:aged?|age\s+of)\s*[:\-]?\s*(\d{1,2})(?!\d)", re.IGNORECASE)


def _screen(*texts):
    """The same positive-term screen the H3 prompts and the studio's picture prompts go through (no bypass)."""
    global _h3
    if _h3 is None:
        sys.path.insert(0, H3_SCRIPTS)
        try:
            import h3_colab
        finally:
            sys.path.remove(H3_SCRIPTS)
        _h3 = h3_colab
    for text in texts:
        young = [m.group(0) for m in _AGE_PHRASE.finditer(text or "") if int(m.group(1)) < 18]
        if young:
            raise HTTPException(status_code=422, detail=f"畫面描述不能寫出未滿 18 歲的年齡：{young}")
        try:
            _h3.screen_prompt(text or "")
        except _h3.H3Error as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None


class SceneVisualBody(BaseModel):
    user_quote: str = Field(description="The user's own words asking for this change, copied exactly. Required.")
    action: str | None = Field(default=None, description="English. Who does what in this one still moment: pose, "
                                                         "expression, hands. Omit to keep it.")
    setting: str | None = Field(default=None, description="English. The place and its props. Omit to keep it.")
    lighting: str | None = Field(default=None, description="English. Light sources and colour. Omit to keep it.")
    on_screen_text: str | None = Field(default=None, description="Exact words that must be legible in the image "
                                                                 "(any language), as short as possible; '' removes "
                                                                 "them. Omit to keep it.")
    negative_extra: str | None = Field(default=None, description="English. Extra things to keep out of this scene.")
    compositions: dict[str, str] | None = Field(
        default=None, description="English camera framing per candidate letter, e.g. {'C': 'low angle close-up ...'}. "
                                  "Only letters that have no image yet.")


@app.put("/stories/{story}/scenes/{scene}/visual", operation_id="set_scene_visual", dependencies=AUTH,
         summary="Change the picture description of one scene (action, setting, lighting, on-screen text, extra "
                 "negatives, camera framing) in the words the USER asked for. Pass their exact words as user_quote and "
                 "only the fields they asked to change. It does not draw anything: only images generated afterwards use "
                 "the new description, existing candidates stay as they are, and an approved scene cannot be edited. "
                 "Never invent changes the user did not ask for.")
def set_scene_visual(story: int, scene: int, body: SceneVisualBody):
    st = _story(story)
    sc = _scene(st, scene)
    quote = body.user_quote.strip()
    if not quote or len(quote) > MAX_VISUAL_QUOTE_CHARS:
        raise HTTPException(status_code=422, detail=f"需要 user_quote：使用者要求修改的原話（上限 {MAX_VISUAL_QUOTE_CHARS} 字）")
    changes = {f: getattr(body, f).strip() for f in VISUAL_TEXT_FIELDS if getattr(body, f) is not None}
    compositions = {k.strip().upper(): v.strip() for k, v in (body.compositions or {}).items()}
    if not changes and not compositions:
        raise HTTPException(status_code=422, detail="沒有要改的欄位：請問使用者想改畫面的哪一部分")
    for letter, text in compositions.items():
        if letter not in cs.LETTERS or not text:
            raise HTTPException(status_code=422, detail="compositions 的鍵要是 A、B、C、D，內容不能是空的")
    for field in VISUAL_ENGLISH_FIELDS:
        if field in changes and not changes[field]:
            raise HTTPException(status_code=422, detail=f"{field} 不能是空的")
    for field, text in {**changes, **compositions}.items():
        if len(text) > MAX_VISUAL_FIELD_CHARS:
            raise HTTPException(status_code=422, detail=f"{field} 太長（上限 {MAX_VISUAL_FIELD_CHARS} 字）")
    for field, text in {**{f: changes[f] for f in VISUAL_ENGLISH_FIELDS + ("negative_extra",) if f in changes},
                        **{f"composition {k}": v for k, v in compositions.items()}}.items():
        if _CJK.search(text):
            raise HTTPException(status_code=422, detail=f"{field} 要用英文寫（畫面文字才可以是中文，請放在 on_screen_text）")
    status = cs.load_review(st, scene)["status"]
    if status == cs.APPROVED:
        raise HTTPException(status_code=409, detail=f"Scene {scene} 已經核准，master 對得上它現在的描述，不能再改")
    running = [j for j in _busy() if j.get("story") == st.number]
    if running:
        raise HTTPException(status_code=409, detail=f"故事 {st.number} 有工作在進行（{running[0]['job_id']}），等它結束再改")
    taken = [x for x in compositions if x in cs.existing_letters(st, scene)]
    if taken:
        raise HTTPException(status_code=409, detail=f"候選 {', '.join(taken)} 已經生成，改它的構圖不會改變那張圖")
    _screen(*changes.values(), *compositions.values())
    before = {f: sc.get(f) for f in changes}
    before_compositions = {k: sc["compositions"].get(k) for k in compositions}
    sc.update(changes)
    sc["compositions"].update(compositions)
    cs.save_story(st)
    with open(os.path.join(st.dir, VISUAL_EDIT_LOG), "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _now(), "scene": scene, "user_quote": quote, "before": before,
                            "before_compositions": before_compositions, "after": changes,
                            "after_compositions": compositions, "decided_via": "open-webui"}, ensure_ascii=False) + "\n")
    if status == cs.REGENERATE:
        nxt = "新描述會用在接下來補的候選 3、4。可以直接問使用者要不要生"
    elif status == cs.NO_CANDIDATES:
        nxt = "還沒有候選圖，之後生的圖都會用新描述"
    else:
        nxt = ("已經有的候選圖不會改變。要用新描述得到新的圖，必須由使用者自己說「Scene "
               f"{scene} 重生」（你不能替他決定），之後再補候選 3、4")
    return {"story": st.number, "scene": scene, "saved": True, "changed": sorted({**changes, **{
        f"composition_{k}": v for k, v in compositions.items()}}), "status": status, "next": nxt}


class VideosBody(BaseModel):
    scenes: list[int] | None = Field(default=None, description="Scene numbers; omit for every approved scene.")
    confirm: bool = Field(default=False, description="false returns the estimate only. Set true only after the "
                                                     "user has seen the cost and said yes.")


@app.post("/stories/{story}/videos", operation_id="make_videos", dependencies=AUTH,
          summary="Queue MiniMax H3 clips on Google Colab for approved scenes. This spends the user's Colab compute "
                  "units and needs Docker Desktop running plus one Google Drive consent click per session. Without "
                  "confirm=true it only returns the estimate and what is missing.")
def make_videos(story: int, body: VideosBody):
    st = _story(story)
    wanted = sorted(set(body.scenes)) if body.scenes else [
        s["scene"] for s in st.data["scenes"] if cs.load_review(st, s["scene"])["status"] == cs.APPROVED]
    for n in wanted:
        _scene(st, n)
    if not wanted:
        raise HTTPException(status_code=409, detail="這個故事還沒有任何核准的 Scene，H3 只接受核准的 master")
    problems = video_problems(st, wanted)
    if problems:
        raise HTTPException(status_code=409, detail="；".join(problems))
    estimate = dict(video_estimate(st, wanted), note="預估值，實際可能差到正負三成；每個 session 最多 4 支",
                    user_actions=["Docker Desktop 要開著", "每個 session 要在瀏覽器按一次 Google Drive 同意"])
    if not body.confirm:
        return {"queued": False, "needs_confirmation": True, "estimate": estimate,
                "message": "還沒開始，也沒有花 CU。請把預估告訴使用者，使用者同意後再用 confirm=true 呼叫一次"}
    if _busy():
        raise HTTPException(status_code=409, detail=f"已有工作在進行：{_busy()[0]['job_id']}，請等它結束")
    job = runner.enqueue("videos", st, {"scenes": wanted}, len(wanted))
    return {"queued": True, "job_id": job["job_id"], "estimate": estimate, "message": "已排入佇列，用 get_job 查進度"}


class CutBody(BaseModel):
    subtitle: str | None = Field(default=None, description="End subtitle. Omit to use the script's own end subtitle "
                                                           "if it has one; pass an empty string for no subtitle.")


@app.post("/stories/{story}/cut", operation_id="make_cut", dependencies=AUTH,
          summary="Join the story's seven clips into one video with levelled audio (CPU only, about a minute). "
                  "Every scene needs a clip first.")
def make_cut(story: int, body: CutBody):
    st = _story(story)
    clips = load_clips(st)
    missing = [s["scene"] for s in st.data["scenes"] if f"scene_{s['scene']:02d}" not in clips]
    if missing:
        raise HTTPException(status_code=409, detail=f"這些 Scene 還沒有影片：{missing}")
    subtitle = body.subtitle
    if subtitle is None:
        subtitle = st.data["scenes"][-1].get("subtitle")
    if _busy():
        raise HTTPException(status_code=409, detail=f"已有工作在進行：{_busy()[0]['job_id']}，請等它結束")
    job = runner.enqueue("cut", st, {"subtitle": subtitle or None}, 1)
    return {"queued": True, "job_id": job["job_id"], "message": "已排入佇列，用 get_job 查進度"}


@app.get("/jobs", operation_id="list_jobs", dependencies=AUTH, summary="Recent jobs, newest first.")
def list_jobs(limit: int = 10):
    return {"jobs": [_job_view(j) for j in store.all()[:max(1, min(limit, 50))]]}


@app.get("/jobs/{job_id}", operation_id="get_job", dependencies=AUTH,
         summary="Progress and result of one job. If needs_user_action is set, tell the user exactly that.")
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="沒有這個 job_id")
    return _job_view(job)


@app.post("/jobs/{job_id}/cancel", operation_id="cancel_job", dependencies=AUTH,
          summary="Cancel a job. A queued job is dropped; a running image job stops after the current image; a "
                  "running H3 job only stops before its next session (a session in progress finishes its clips).")
def cancel_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="沒有這個 job_id")
    if job["status"] not in ACTIVE:
        return {"job_id": job_id, "status": job["status"], "message": "這個工作已經結束"}
    if job["status"] == QUEUED:
        job.update(status=CANCELLED, finished_at=_now(), message="已取消（還沒開始）")
        store.save(job)
    else:
        runner.cancel_requested.add(job_id)
    return {"job_id": job_id, "status": store.get(job_id)["status"], "message": "已要求取消"}


@app.post("/jobs/{job_id}/consent-done", operation_id="consent_done", dependencies=AUTH,
          summary="Tell a waiting H3 job that the USER approved Google Drive access in the browser. Call only after "
                  "the user says they clicked it; calling early makes the session re-download and store nothing.")
def consent_done(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="沒有這個 job_id")
    if job["status"] != WAITING:
        raise HTTPException(status_code=409, detail="這個工作目前沒有在等 Drive 同意")
    if job["kind"] == "images":     # a Colab image job: the z-image-colab worker waits for the same click
        done = runner.run([sys.executable, os.path.join(ZIMAGE_SCRIPTS, "worker.py"), "consent-done"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=REPO_ROOT)
        what = "Colab 生圖"
    else:
        done = runner.run([sys.executable, os.path.join(H3_SCRIPTS, "consent_done.py")], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", cwd=REPO_ROOT)
        what = "H3"
    if done.returncode != 0:
        raise HTTPException(status_code=500, detail=(done.stdout or done.stderr).strip()[-300:])
    return {"job_id": job_id, "message": f"已通知，{what}會繼續"}


def _allowed_file(path):
    """Images, clips and the review sheet under the project folder or output/story*/ - nothing else."""
    full = os.path.realpath(os.path.join(REPO_ROOT, path))
    if not full.lower().endswith(FILE_EXTENSIONS) or not os.path.isfile(full):
        return None
    project = os.path.realpath(runner.root)
    output = os.path.realpath(OUTPUT_ROOT)
    for base in (project, output):
        try:
            inside = os.path.commonpath([full, base]) == base
        except ValueError:
            inside = False
        if inside and (base == project or os.path.relpath(full, output).split(os.sep)[0].startswith("story")):
            return full
    return None


@app.get("/files/{path:path}", include_in_schema=False)
def files(path: str):
    # No token here: the browser's <img> cannot send one. Read-only, confined, loopback.
    full = _allowed_file(path)
    if full is None:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(full)


def check_bind(host, token):
    """A non-loopback bind without a token would hand GPU and compute-unit spending to the network."""
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise gc.UsageError("CONCEPT_API_HOST 不是 127.0.0.1 時一定要設定 CONCEPT_API_TOKEN")


class AccessLog:
    """ASGI wrapper that appends one line per request (time, client, method, path, status, ms) to a file. It never
    records request bodies, so a user_quote or a token cannot leak into it; /files/ is skipped (every thumbnail would
    add a line). It exists so a chat-side problem can be told apart: did Open WebUI call a tool at all?"""

    def __init__(self, app, path):
        self.app = app
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].startswith("/files/"):
            return await self.app(scope, receive, send)
        status = [0]
        started = time.time()

        async def _send(message):
            if message["type"] == "http.response.start":
                status[0] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            query = scope.get("query_string", b"").decode("latin-1")
            client = (scope.get("client") or ("-",))[0]
            target = scope["path"] + ("?" + query if query else "")
            line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {client} {scope['method']} {target} "
                    f"{status[0]} {(time.time() - started) * 1000:.0f}ms\n")
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                pass  # logging must never break a request


ACCESS_LOG_PATH = os.environ.get(
    "CONCEPT_API_ACCESS_LOG", os.path.join(REPO_ROOT, "outputs", "logs", "concept_api_access.log"))


if __name__ == "__main__":
    import uvicorn

    try:
        check_bind(HOST, os.environ.get("CONCEPT_API_TOKEN"))
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
    uvicorn.run(AccessLog(app, ACCESS_LOG_PATH), host=HOST, port=PORT)
