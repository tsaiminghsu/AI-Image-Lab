"""One-page studio: prompt -> candidate pictures -> "make video" on a picture -> MP4, all on one wall.

    ComfyUI\\.venv\\Scripts\\python.exe training\\studio_api.py        -> http://127.0.0.1:7864

The same pipeline the other entry points use, behind one page: pictures come from the local Z-Image (ComfyUI,
cooled to 60 C before every picture) or the Colab Z-Image worker; videos are MiniMax H3 on Colab. The job queue,
the H3 session driver and the cost constants are concept_api's, reused as they are.

What this layer must not loosen:
- Nothing that spends GPU time or Colab compute units is queued without confirm=true; without it the endpoints
  only return an estimate (and, for pictures, the exact English prompt that would be sent).
- H3's first frame is only ever a picture on this wall, referenced by id and checked against the sha256 recorded
  when it was generated. There is no upload endpoint: H3 has no negative prompt, so its safety rests on the first
  frame being a fictional adult (or nobody) made by this project, plus the prompt screen below.
- Picture prompts and H3 descriptions both go through h3_colab.screen_prompt, which has no bypass.
- One job at a time. A restart marks unfinished jobs interrupted and re-runs nothing.
"""

import contextlib
import datetime
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import comfyui_client as client
import concept_api as ca
import concept_colab
import generate_character as gc
import translate_prompt

sys.path.insert(0, ca.H3_SCRIPTS)
try:
    import h3_colab as h3
finally:
    sys.path.remove(ca.H3_SCRIPTS)

REPO_ROOT = ca.REPO_ROOT
STUDIO_ROOT = os.environ.get("STUDIO_ROOT") or os.path.join(REPO_ROOT, "outputs", "studio")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio", "index.html")
ZIMAGE_SKILL_SCRIPTS = os.path.join(REPO_ROOT, ".claude", "skills", "z-image-colab", "scripts")
HOST = os.environ.get("STUDIO_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("STUDIO_API_PORT", "7864"))

# Local sizes are what the 8 GB card was measured with (720x1280 ~ 131-141 s); Colab uses its ~1 MP presets.
LOCAL_SIZES = {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (1024, 1024)}
ASPECTS = tuple(LOCAL_SIZES)
BACKENDS = ("local", "colab")
DURATIONS = (5, 8, 10)
MAX_IMAGES = 4
MAX_CLIPS = 12
COLAB_IMAGE_CU = "約 0.2–0.4 CU（L4，實測 8 張一個 session 0.24 CU）"
IMAGE_ID_RE = re.compile(r"img-[A-Za-z0-9-]{1,60}")
FILE_EXTENSIONS = (".png", ".mp4")


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _new_id(prefix):
    return f"{prefix}-{datetime.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def images_dir():
    return os.path.join(STUDIO_ROOT, "images")


def videos_dir():
    return os.path.join(STUDIO_ROOT, "videos")


def file_url(path):
    rel = os.path.relpath(path, STUDIO_ROOT).replace("\\", "/")
    return f"/files/{rel}"


# ---------------------------------------------------------------- the wall

def library(kind=None):
    """Every picture and clip with a sidecar, newest first."""
    items = []
    for sub, label in (("images", "image"), ("videos", "video")):
        if kind and kind != label:
            continue
        directory = os.path.join(STUDIO_ROOT, sub)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            meta = _read_json(os.path.join(directory, name))
            media = os.path.join(directory, name[:-5] + (".png" if label == "image" else ".mp4"))
            if not os.path.isfile(media):
                continue
            items.append(dict(meta, kind=label, url=file_url(media)))
    return sorted(items, key=lambda m: m.get("created_at", ""), reverse=True)


def find_image(image_id):
    """(png path, sidecar) of a wall picture, or an HTTPException. The sha256 must still match: a file swapped in
    by hand is not a picture this project generated."""
    if not IMAGE_ID_RE.fullmatch(image_id or ""):
        raise HTTPException(status_code=422, detail="image_id 不是作品牆上的圖片編號")
    png = os.path.join(images_dir(), image_id + ".png")
    meta_path = os.path.join(images_dir(), image_id + ".json")
    if not (os.path.isfile(png) and os.path.isfile(meta_path)):
        raise HTTPException(status_code=404, detail=f"作品牆上沒有 {image_id}")
    meta = _read_json(meta_path)
    if meta.get("image_sha256") != _sha256(png):
        raise HTTPException(status_code=409, detail=f"{image_id} 的檔案跟生成時記錄的不一致，不能當首幀")
    return png, meta


# ---------------------------------------------------------------- prompts and estimates

def english_prompt(text):
    """The user's prompt as English text; non-ASCII goes through the same translator the GUI uses."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="prompt 是空的")
    if len(text) > gc.MAX_PROMPT_CHARS:
        raise HTTPException(status_code=422, detail=f"prompt 太長（上限 {gc.MAX_PROMPT_CHARS} 字）")
    if text.isascii():
        return text
    try:
        return translate_prompt.translate_to_english(text)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"翻譯失敗，請直接寫英文：{exc}") from None


def screen(*texts):
    for text in texts:
        try:
            h3.screen_prompt(text or "")
        except h3.H3Error as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None


def assemble_image_prompt(text, solo, allow_text):
    return gc._build_prompt_and_negative(
        text, None, "safe", None, "", gc.REALISTIC_NEGATIVE, "z_image_turbo", solo=solo, allow_text=allow_text)


def image_estimate(backend, count):
    if backend == "local":
        minutes = round(count * (ca.SECONDS_PER_IMAGE + ca.COOLDOWN_SECONDS_PER_IMAGE) / 60)
        return {"images": count, "minutes": minutes, "compute_units": 0,
                "where": "本機 RTX 2070；每張開始前會等顯示卡降到 60°C 以下，所以比連續跑慢"}
    return {"images": count, "minutes": round(ca.SESSION_OVERHEAD_MINUTES + count * 0.3), "compute_units": None,
            "where": f"Colab L4，{COLAB_IMAGE_CU}；每台新 VM 要在瀏覽器按一次 Google Drive 同意"}


def video_estimate(clips):
    sessions = -(-len(clips) // ca.MAX_CLIPS_PER_SESSION)
    minutes = sessions * ca.SESSION_OVERHEAD_MINUTES
    for c in clips:
        minutes += (c["duration"] * ca.INFERENCE_SECONDS_PER_VIDEO_SECOND + 15) / 60
    return {"clips": len(clips), "sessions": sessions, "minutes": round(minutes),
            "compute_units": round(minutes / 60 * ca.CU_PER_HOUR, 1),
            "note": "預估值，實際可能差到正負三成；每個 session 最多 4 支",
            "user_actions": ["Docker Desktop 要開著", "每個 session 要在瀏覽器按一次 Google Drive 同意"]}


# ---------------------------------------------------------------- jobs

class StudioRunner(ca.Runner):
    """concept_api's one-at-a-time runner with studio job kinds. The picture and clip jobs only touch STUDIO_ROOT."""

    def __init__(self, store):
        super().__init__(store)
        self.translate = translate_prompt.translate_to_english

    def enqueue(self, kind, params, total):
        job = {"job_id": f"{kind}-{datetime.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}", "kind": kind,
               "status": ca.QUEUED, "created_at": _now(), "order": time.time_ns(),
               "started_at": None, "finished_at": None, "progress": {"done": 0, "total": total, "current": None},
               "needs_user_action": None, "message": "已排入佇列", "params": params, "result": None, "error": None}
        self.store.save(job)
        self.queue.put(job["job_id"])
        return job

    # -- pictures
    def _save_image(self, raw, spec, backend, extra):
        image_id = spec["image_id"]
        png = os.path.join(images_dir(), image_id + ".png")
        os.makedirs(images_dir(), exist_ok=True)
        shutil.move(raw, png)
        meta = {"image_id": image_id, "prompt": spec["user_prompt"], "full_prompt": spec["prompt"],
                "negative_prompt": spec["negative"], "seed": spec["seed"], "aspect": spec["aspect"],
                "width": spec["width"], "height": spec["height"], "backend": backend, "created_at": _now(),
                "model": "z_image_turbo", "image_sha256": _sha256(png)}
        meta.update(extra)
        _write_json(os.path.join(images_dir(), image_id + ".json"), meta)
        return meta

    def _run_simg(self, job):
        params = job["params"]
        if params["backend"] == "colab":
            return self._images_colab(job)
        return self._images_local(job)

    def _images_local(self, job):
        job_id, specs = job["job_id"], job["params"]["specs"]
        self.guard.before_batch(max_start_temp_c=ca.rp.start_temp_c(ca.rp.load_settings()))
        process = None
        if not self.server_up():
            self._update(job_id, message="啟動 ComfyUI")
            process = self.start_comfyui()
        thermal = self.thermal_log(os.path.join(REPO_ROOT, "outputs", "thermal", f"{job_id}.csv")).start()
        waited = 0.0
        made = []
        submit = self.submit or client.submit_txt2img_generation_zimage
        try:
            for spec in specs:
                if job_id in self.cancel_requested:
                    break
                self.guard.before_image()
                self._update(job_id, message="等顯示卡降溫", progress={"current": spec["image_id"]})
                waited += self.guard.wait_until_cool()["waited_seconds"]
                thermal.mark(f"start {spec['image_id']}")
                self._update(job_id, message="生圖中")
                raw = submit(prompt=spec["prompt"], negative_prompt=spec["negative"], seed=spec["seed"],
                             filename_prefix=spec["image_id"], model="z_image_turbo", width=spec["width"],
                             height=spec["height"], steps=client.ZIMAGE_STEPS, cfg=client.ZIMAGE_CFG)
                self._save_image(raw, spec, "local", {"cfg": client.ZIMAGE_CFG, "steps": client.ZIMAGE_STEPS})
                made.append(spec["image_id"])
                self._update(job_id, progress={"done": len(made)})
        finally:
            temperature = dict(thermal.stop(), cooldown_wait_seconds=round(waited))
            if process is not None:   # only a ComfyUI this job started
                process.terminate()
        return {"images": made, "temperature": temperature}

    def _images_colab(self, job):
        """submit.py (free) then worker.py up --foreground (spends CU); the finished PNGs are copied to the wall."""
        job_id, specs = job["job_id"], job["params"]["specs"]
        batch = job_id
        manifest = os.path.join(STUDIO_ROOT, "jobs", f"{job_id}.colab.json")
        _write_json(manifest, {"batch": batch, "jobs": [
            {"prompt": s["user_prompt"], "aspect": s["aspect"], "seed": s["seed"], "solo": s["solo"],
             "allow_text": s["allow_text"]} for s in specs]})
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        for step, args in (("submit", ["--manifest", manifest]), ("worker", ["up", "--foreground"])):
            script = os.path.join(ZIMAGE_SKILL_SCRIPTS, "submit.py" if step == "submit" else "worker.py")
            self._update(job_id, message="排入 Colab 佇列" if step == "submit" else "Colab 執行中")
            proc = self.popen([sys.executable, script, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", env=env, cwd=REPO_ROOT)
            waiting = False
            for line in proc.stdout:
                if "consent" in line.lower() and "drive" in line.lower():
                    waiting = True
                    self._update(job_id, status=ca.WAITING, message="等使用者按 Google Drive 同意",
                                 needs_user_action="請在瀏覽器按 Google Drive 的同意，按完後在頁面按「我按好了」")
                elif waiting and "consent" not in line.lower():
                    waiting = False
                    self._update(job_id, status=ca.RUNNING, needs_user_action=None)
            if proc.wait() != 0:
                raise RuntimeError(f"Colab Z-Image 的 {step} 步驟失敗（exit {proc.returncode}）")
        done = {j["seed"]: j for j in concept_colab.find_jobs(batch)}
        made = []
        for spec in specs:
            found = done.get(spec["seed"])
            if found and found["status"] == "COMPLETED" and found.get("output") and os.path.isfile(found["output"]):
                self._save_image(found["output"], dict(spec, prompt=found["prompt"], negative=found["negative_prompt"]),
                                 "colab", {"colab_job_id": found["job_id"], "gpu": found.get("gpu"),
                                           "cfg": found.get("cfg")})
                made.append(spec["image_id"])
        self._update(job_id, progress={"done": len(made)})
        if not made:
            raise RuntimeError("Colab 沒有回傳任何圖片")
        return {"images": made, "missing": [s["image_id"] for s in specs if s["image_id"] not in made]}

    # -- clips
    def _run_svid(self, job):
        job_id, clips = job["job_id"], job["params"]["clips"]
        batch_dir = os.path.join(STUDIO_ROOT, "jobs")
        os.makedirs(videos_dir(), exist_ok=True)
        env = dict(os.environ, H3_MODEL_SOURCE="download", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        check = self.run([sys.executable, os.path.join(ca.H3_SCRIPTS, "check.py")], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env, cwd=REPO_ROOT)
        if check.returncode != 0:
            raise RuntimeError("H3 開始前的檢查沒過（沒有花 CU）：" + (check.stdout or check.stderr).strip()[-400:])
        groups = [clips[i:i + ca.MAX_CLIPS_PER_SESSION] for i in range(0, len(clips), ca.MAX_CLIPS_PER_SESSION)]
        results, cu = [], 0.0
        for index, group in enumerate(groups, 1):
            if job_id in self.cancel_requested:
                break
            manifest = os.path.join(batch_dir, f"{job_id}_{index}.json")
            _write_json(manifest, {"batch_id": f"{job_id}-{index}", "jobs": [self._h3_job(c) for c in group]})
            self._update(job_id, message=f"H3 第 {index}/{len(groups)} 個 session",
                         progress={"current": f"session {index}"})
            try:
                summary = self._h3_session(job_id, manifest, env)
            except (h3.H3Error, ValueError) as exc:
                raise RuntimeError(f"H3 失敗：{exc}") from None
            cu += summary.get("cu_used_settled") or 0.0
            by_scene = {c["video_id"]: c for c in group}
            for rec in summary.get("records", []):
                clip = by_scene.get(str(rec.get("scene_id")))
                ok = rec.get("status") == "COMPLETED" and rec.get("output") and os.path.isfile(rec["output"])
                if clip and ok:
                    _write_json(os.path.join(videos_dir(), clip["video_id"] + ".json"), {
                        "video_id": clip["video_id"], "image_id": clip["image_id"], "description": clip["description"],
                        "soundscape": clip["soundscape"], "duration": clip["duration"],
                        "constraints": clip["constraints"], "h3_job_id": rec.get("job_id"), "created_at": _now()})
                results.append({"video_id": str(rec.get("scene_id")), "status": rec.get("status"),
                                "error": rec.get("error")})
            self._update(job_id, progress={"done": sum(1 for r in results if r["status"] == "COMPLETED")})
        done = [r for r in results if r["status"] == "COMPLETED"]
        return {"clips_completed": len(done), "clips_failed": len(results) - len(done), "cu_used": round(cu, 2),
                "clips": results}

    @staticmethod
    def _h3_job(c):
        return {"job_id": c["video_id"], "scene_id": c["video_id"],
                "input": {"image": c["image_path"]},
                "prompt": {"description": c["description"], "soundscape": c["soundscape"],
                           "constraints": c["constraints"]},
                "settings": {"duration": c["duration"]},
                "output": os.path.join(videos_dir(), c["video_id"] + ".mp4")}


# ---------------------------------------------------------------- app

def _require_token(authorization: str | None = Header(default=None)):
    token = os.environ.get("STUDIO_API_TOKEN")
    if not token:
        return
    given = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(given, token):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


store = ca.JobStore(os.path.join(STUDIO_ROOT, "jobs"))
runner = StudioRunner(store)


@contextlib.asynccontextmanager
async def _lifespan(_app):
    interrupted = store.mark_interrupted()
    if interrupted:
        print(f"[studio_api] {len(interrupted)} 個未完成的工作標成 interrupted，沒有自動重跑", flush=True)
    runner.start()
    yield


app = FastAPI(title="Image-to-video studio", version="1.0.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[], allow_methods=["*"], allow_headers=["*"])
AUTH = [Depends(_require_token)]


def _busy():
    return [j for j in store.all() if j["status"] in ca.ACTIVE]


def _job_view(job):
    return {k: job.get(k) for k in ("job_id", "kind", "status", "message", "progress", "needs_user_action", "result",
                                    "error", "created_at", "started_at", "finished_at")}


def _refuse_if_busy():
    if _busy():
        raise HTTPException(status_code=409, detail=f"已有工作在進行：{_busy()[0]['job_id']}，請等它結束")


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE, media_type="text/html")


@app.get("/api/library", dependencies=AUTH)
def get_library(kind: str | None = None):
    if kind not in (None, "image", "video"):
        raise HTTPException(status_code=422, detail="kind 要是 image 或 video")
    return {"items": library(kind)}


class ImagesBody(BaseModel):
    prompt: str
    aspect: str = "9:16"
    count: int = Field(default=2, ge=1, le=MAX_IMAGES)
    backend: str = "local"
    seed: int | None = Field(default=None, description="First seed; the rest are seed+1... Random when omitted.")
    solo: bool = True
    allow_text: bool = False
    confirm: bool = False


@app.post("/api/images", dependencies=AUTH)
def make_images(body: ImagesBody):
    if body.aspect not in ASPECTS:
        raise HTTPException(status_code=422, detail=f"比例要是 {', '.join(ASPECTS)}")
    if body.backend not in BACKENDS:
        raise HTTPException(status_code=422, detail=f"引擎要是 {', '.join(BACKENDS)}")
    text = english_prompt(body.prompt)
    screen(text)
    prompt, negative = assemble_image_prompt(text, body.solo, body.allow_text)
    estimate = image_estimate(body.backend, body.count)
    if not body.confirm:
        return {"queued": False, "needs_confirmation": True, "estimate": estimate, "english_prompt": text,
                "message": "還沒開始。確認預估後用 confirm=true 再送一次"}
    _refuse_if_busy()
    seed = body.seed if body.seed is not None else int.from_bytes(os.urandom(3), "big")
    width, height = LOCAL_SIZES[body.aspect]
    specs = [{"image_id": _new_id("img") + f"-{i + 1}", "user_prompt": text, "prompt": prompt, "negative": negative,
              "seed": seed + i, "aspect": body.aspect, "width": width, "height": height, "solo": body.solo,
              "allow_text": body.allow_text} for i in range(body.count)]
    job = runner.enqueue("simg", {"backend": body.backend, "specs": specs}, len(specs))
    return {"queued": True, "job_id": job["job_id"], "estimate": estimate, "image_ids": [s["image_id"] for s in specs]}


class ClipBody(BaseModel):
    image_id: str
    description: str = Field(description="English: what is in the first frame, the one action, then the camera.")
    soundscape: str = ""
    duration: int = 5
    locked_camera: bool = False
    static_subject: bool = False


class VideosBody(BaseModel):
    clips: list[ClipBody] = Field(min_length=1, max_length=MAX_CLIPS)
    confirm: bool = False


@app.post("/api/videos", dependencies=AUTH)
def make_videos(body: VideosBody):
    clips = []
    for c in body.clips:
        if c.duration not in DURATIONS:
            raise HTTPException(status_code=422, detail=f"秒數要是 {', '.join(map(str, DURATIONS))}")
        description = english_prompt(c.description)
        soundscape = english_prompt(c.soundscape) if c.soundscape.strip() else ""
        screen(description, soundscape)
        png, _meta = find_image(c.image_id)
        constraints = [x for x, on in (("locked-camera", c.locked_camera), ("static-subject", c.static_subject)) if on]
        clips.append({"video_id": _new_id("vid") + f"-{len(clips) + 1}", "image_id": c.image_id, "image_path": png,
                      "description": description, "soundscape": soundscape, "duration": c.duration,
                      "constraints": constraints})
    estimate = video_estimate(clips)
    if not body.confirm:
        return {"queued": False, "needs_confirmation": True, "estimate": estimate,
                "message": "還沒開始，也沒有花 CU。確認預估後用 confirm=true 再送一次"}
    _refuse_if_busy()
    now = _now()
    for c in clips:   # the user picked this picture as the first frame by pressing the button on it
        meta_path = os.path.join(images_dir(), c["image_id"] + ".json")
        _write_json(meta_path, dict(_read_json(meta_path), approved_at=now))
    job = runner.enqueue("svid", {"clips": clips}, len(clips))
    return {"queued": True, "job_id": job["job_id"], "estimate": estimate, "video_ids": [c["video_id"] for c in clips]}


@app.get("/api/jobs", dependencies=AUTH)
def list_jobs(limit: int = 10):
    return {"jobs": [_job_view(j) for j in store.all()[:max(1, min(limit, 50))]]}


@app.get("/api/jobs/{job_id}", dependencies=AUTH)
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="沒有這個 job_id")
    return _job_view(job)


@app.post("/api/jobs/{job_id}/cancel", dependencies=AUTH)
def cancel_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="沒有這個 job_id")
    if job["status"] not in ca.ACTIVE:
        return {"job_id": job_id, "status": job["status"], "message": "這個工作已經結束"}
    if job["status"] == ca.QUEUED:
        job.update(status=ca.CANCELLED, finished_at=_now(), message="已取消（還沒開始）")
        store.save(job)
    else:
        runner.cancel_requested.add(job_id)
    return {"job_id": job_id, "status": store.get(job_id)["status"], "message": "已要求取消（進行中的 session 會先跑完）"}


@app.post("/api/jobs/{job_id}/consent-done", dependencies=AUTH)
def consent_done(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="沒有這個 job_id")
    if job["status"] != ca.WAITING:
        raise HTTPException(status_code=409, detail="這個工作目前沒有在等 Drive 同意")
    colab_images = job["kind"] == "simg"
    script = os.path.join(ZIMAGE_SKILL_SCRIPTS if colab_images else ca.H3_SCRIPTS,
                          "worker.py" if colab_images else "consent_done.py")
    argv = [sys.executable, script] + (["consent-done"] if colab_images else [])
    done = runner.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=REPO_ROOT)
    if done.returncode != 0:
        raise HTTPException(status_code=500, detail=(done.stdout or done.stderr).strip()[-300:])
    return {"job_id": job_id, "message": "已通知，會繼續"}


@app.get("/files/{path:path}", include_in_schema=False)
def files(path: str):
    # No token: <img>/<video> cannot send one. Read-only, confined to STUDIO_ROOT, loopback.
    full = os.path.realpath(os.path.join(STUDIO_ROOT, path))
    root = os.path.realpath(STUDIO_ROOT)
    try:
        inside = os.path.commonpath([full, root]) == root
    except ValueError:
        inside = False
    if not (inside and full.lower().endswith(FILE_EXTENSIONS) and os.path.isfile(full)):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(full)


def check_bind(host, token):
    """A non-loopback bind without a token would hand GPU and compute-unit spending to the network."""
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise gc.UsageError("STUDIO_API_HOST 不是 127.0.0.1 時一定要設定 STUDIO_API_TOKEN")


if __name__ == "__main__":
    import uvicorn

    try:
        check_bind(HOST, os.environ.get("STUDIO_API_TOKEN"))
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
    uvicorn.run(app, host=HOST, port=PORT)
