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
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import comfyui_client as client
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
SECONDS_PER_IMAGE = 140              # measured on the RTX 2070: 131-141 s mean per 720x1280 image
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

        def before_each(_story, scene, letter):
            self.guard.before_image()
            self._update(job_id, message="生圖中",
                         progress={"done": to_make - remaining(), "current": f"scene_{scene:02d} 候選 {letter}"})

        try:
            while True:
                try:
                    cs.generate(bible, [story], letters=letters, scene_numbers=scenes, submit=self.submit,
                                server_up=self.server_up, before_each=before_each,
                                should_stop=lambda: job_id in self.cancel_requested)
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
            if process is not None:      # only a ComfyUI this job started; never someone else's
                process.terminate()
            cs.build_index(cs.load_stories(self.root), self.root)
            cs.build_sheet(bible, cs.load_stories(self.root), self.root)
        made = to_make - remaining()
        self._update(job_id, progress={"done": made, "current": None})
        return {"images": made, "comfyui_restarts": restarts,
                "review_sheet": file_url(os.path.join(self.root, "review_sheet.html")),
                "next": "請使用者看圖後決定每個 Scene 核准哪一張，或判定重生"}

    # -- MiniMax H3 clips on Colab
    def _run_videos(self, job):
        job_id, params = job["job_id"], job["params"]
        story = self._story(job)
        out_dir = os.path.join(OUTPUT_ROOT, story.id)
        os.makedirs(out_dir, exist_ok=True)
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
            manifest = os.path.join(out_dir, f"h3_manifest_{job_id}_{index}.json")
            cs._write_json(manifest, video_manifest(story, batch, f"{job_id}-{index}", out_dir))
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
        out = os.path.join(OUTPUT_ROOT, story.id, f"{story.id}_cut_{job['job_id']}.mp4")
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


def video_manifest(story, scene_numbers, batch_id, out_dir):
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
            "output": os.path.join(out_dir, f"scene_{n:02d}_h3_{batch_id}.mp4"),
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
            candidates.append({"candidate": number, "url": url,
                               "markdown": f"![Scene {n} 候選 {number}]({url})"})
        clip = clips.get(f"scene_{n:02d}")
        scenes.append({"scene": n, "title": s["title"], "plot": s["beat_zh"], "status": review["status"],
                       "approved_candidate": review.get("approved_candidate"), "notes": review.get("notes") or None,
                       "text_to_verify": s["on_screen_text"] or None, "candidates": candidates,
                       "has_video_prompt": "video" in s,
                       "clip_url": file_url(os.path.join(REPO_ROOT, clip["path"])) if clip else None})
    return {"story": st.number, "title": st.data.get("title_zh"), "scenes": scenes,
            "review_checklist": cs.load_bible(runner.root)["review_checklist"]}


class GenerateBody(BaseModel):
    scenes: list[int] | None = Field(default=None, description="Scene numbers 1-7; omit for all seven.")
    extra_candidates: bool = Field(default=False, description="true = candidates 3 and 4, only for scenes the user "
                                                              "marked for regeneration; false = candidates 1 and 2.")
    confirm: bool = Field(default=False, description="false returns the estimate only. Set true only after the "
                                                     "user has seen the estimate and said yes.")


@app.post("/stories/{story}/candidates", operation_id="generate_candidates", dependencies=AUTH,
          summary="Queue local GPU generation of candidate stills for a story. Without confirm=true it only returns "
                  "the estimate (number of images, minutes); nothing is generated.")
def generate_candidates(story: int, body: GenerateBody):
    st = _story(story)
    letters = cs.EXTRA_ROUND if body.extra_candidates else cs.FIRST_ROUND
    try:
        todo = cs.pending([st], letters, scene_numbers=set(body.scenes) if body.scenes else None)
    except gc.UsageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if not todo:
        return {"queued": False, "message": "這些候選圖都已經存在，沒有要生的"}
    estimate = {"images": len(todo), "minutes": round(len(todo) * SECONDS_PER_IMAGE / 60),
                "gpu": "本機 RTX 2070；生圖期間聊天模型會被卸載，聊天會讓生圖暫停等待"}
    if not body.confirm:
        return {"queued": False, "needs_confirmation": True, "estimate": estimate,
                "message": "還沒開始。請把預估告訴使用者，使用者同意後再用 confirm=true 呼叫一次"}
    if _busy():
        raise HTTPException(status_code=409, detail=f"已有工作在進行：{_busy()[0]['job_id']}，請等它結束")
    job = runner.enqueue("images", st, {"letters": list(letters), "scenes": sorted(set(body.scenes or []))}, len(todo))
    return {"queued": True, "job_id": job["job_id"], "estimate": estimate, "message": "已排入佇列，用 get_job 查進度"}


class ReviewBody(BaseModel):
    decision: str = Field(description="'approve' or 'regenerate'.", pattern="^(approve|regenerate)$")
    candidate: int | None = Field(default=None, ge=1, le=4, description="For approve: the candidate number the USER "
                                                                        "chose (1-4).")
    notes: str = Field(default="", description="The user's reason, in their words.")


@app.post("/stories/{story}/scenes/{scene}/review", operation_id="review_scene", dependencies=AUTH,
          summary="Record the USER's decision for one scene: approve a candidate (it becomes the master image) or "
                  "mark the scene for regeneration. Call this only when the user explicitly stated the decision and, "
                  "for approve, the candidate number. Never decide for the user.")
def review_scene(story: int, scene: int, body: ReviewBody):
    st = _story(story)
    _scene(st, scene)
    if body.decision == "approve" and body.candidate is None:
        raise HTTPException(status_code=422, detail="核准需要使用者指定的候選編號（1-4）")
    try:
        result = cs.review(st, scene, approve=body.candidate if body.decision == "approve" else None,
                           regenerate=body.decision == "regenerate", notes=body.notes, decided_via="open-webui")
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
    done = runner.run([sys.executable, os.path.join(H3_SCRIPTS, "consent_done.py")], capture_output=True, text=True,
                      encoding="utf-8", errors="replace", cwd=REPO_ROOT)
    if done.returncode != 0:
        raise HTTPException(status_code=500, detail=(done.stdout or done.stderr).strip()[-300:])
    return {"job_id": job_id, "message": "已通知，H3 會繼續"}


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


if __name__ == "__main__":
    import uvicorn

    try:
        check_bind(HOST, os.environ.get("CONCEPT_API_TOKEN"))
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
    uvicorn.run(app, host=HOST, port=PORT)
