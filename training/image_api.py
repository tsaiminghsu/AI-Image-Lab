"""Thin FastAPI wrapper around generate_character.gen_custom() / CHARACTERS,
so an external project (the ai-companion backend at d:\\NSFW\\ai-companion)
can trigger character image generation over HTTP without sharing this
project's venv or importing across repos.

Does NOT reimplement any generation logic - every request here is just
marshalling into the exact same gen_custom() call the CLI and gui.py already
use, including the same safety-tier negative-prompt handling (AGE_SAFETY_NEGATIVE
is never touched by anything this file does).

Run (after ComfyUI is up on 127.0.0.1:8188):
    D:\\AI-Image-Lab\\ComfyUI\\.venv\\Scripts\\python.exe image_api.py
Listens on 127.0.0.1:7862 - distinct from gui.py's 7861 and ComfyUI's 8188,
so all three can run at once (VRAM budget permitting - see README's
ComfyUI-vs-kohya_ss troubleshooting note; running this alongside a resident
local LLM for the companion platform's chat has the same constraint).
"""

import glob
import hmac
import os
import random
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, field_validator

import generate_character as gc

# User-facing strings in this project are normally Traditional Chinese, but this file is the
# deliberate exception: every HTTP `detail` string below (validation errors, 401s, 404s) is
# consumed by the ai-companion backend's code, not read by a person, and the existing precedent
# ("Unknown job_id") is English - so all of them, old and new, stay English.


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Optional shared-secret gate, checked on every route via the app-level dependency below.

    IMAGE_API_TOKEN unset (the default) means no auth at all, and that has to stay the default:
    this service binds 127.0.0.1, so the only processes that can reach it already have
    filesystem access on this same machine - a token buys nothing against that threat model.
    It exists purely so that if this ever gets bound to a non-loopback address later, that
    change doesn't silently ship with zero auth. For the same "still a local single-user tool"
    reason, this deliberately does NOT add CORS middleware - that would be a step toward a
    public service this file is not meant to be.
    """
    token = os.environ.get("IMAGE_API_TOKEN")
    if not token:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, token):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


app = FastAPI(title="AI-Image-Lab image_api", version="0.1.0", dependencies=[Depends(_require_api_key)])

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "companion_chat")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=1)  # one at a time - matches BATCH_SIZE=1's
# "one RTX 2070, no concurrent generations" assumption already baked into comfyui_client.py

# OrderedDict (not dict) so "oldest" for cap eviction is just "iterate from the front" - insertion
# order is preserved and a job's position never needs to move after it's created. See _evict_jobs.
_jobs: "OrderedDict[str, dict]" = OrderedDict()
_jobs_lock = threading.Lock()
MAX_JOBS = 500
JOB_TTL_SECONDS = 6 * 3600

_ANCHOR_EXTENSIONS = (".png", ".jpg", ".jpeg")


class GenerateRequest(BaseModel):
    prompt: str
    trigger: str | None = None
    anchor_path: str | None = None
    tier: Literal["safe", "suggestive"] = "safe"
    # HQ two-pass path (base -> ESRGAN hires -> face/hand FaceDetailer + per-character
    # LoRA when available). Default on for quality; set False for the fast legacy
    # single-pass path. When hq is on, FaceDetailer runs by default (use_facedetailer
    # left None = auto), so callers usually don't need to set use_facedetailer.
    hq: bool = True
    use_facedetailer: bool | None = None

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, v: str) -> str:
        # No check existed here at all before; the RunPod worker has always enforced
        # MAX_PROMPT_CHARS, so this endpoint was the one entry point that didn't.
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be empty")
        if len(v) > gc.MAX_PROMPT_CHARS:
            raise ValueError(f"prompt must be at most {gc.MAX_PROMPT_CHARS} characters")
        return v


def _default_anchor_for(trigger: str) -> str | None:
    """Picks any existing anchor image for a known character so callers don't
    need to know exact filenames - mirrors what a human operator would do
    manually via gui.py's anchor dropdown.

    No containment check needed here (unlike _validate_anchor_path below): the glob pattern
    is hardcoded under reference_candidates/, so this can only ever return a path already
    inside gc.REFERENCE_CANDIDATES_DIR - there's no caller-supplied path component to escape
    with.
    """
    pattern = os.path.join(os.path.dirname(__file__), "reference_candidates", trigger, "anchor_*.png")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def _allowed_anchor_dirs() -> list[str]:
    """Base directories a caller-supplied anchor_path is allowed to resolve into.

    Defaults to gc.REFERENCE_CANDIDATES_DIR; IMAGE_API_ANCHOR_DIRS (os.pathsep-separated, so
    ';' on Windows) extends it for setups that keep anchors elsewhere.
    """
    dirs = [gc.REFERENCE_CANDIDATES_DIR]
    extra = os.environ.get("IMAGE_API_ANCHOR_DIRS")
    if extra:
        dirs.extend(p for p in extra.split(os.pathsep) if p)
    return [os.path.realpath(d) for d in dirs]


def _is_within(path: str, base: str) -> bool:
    """True if realpath `path` sits inside realpath `base`. normcase makes this correct on
    Windows' case-insensitive filesystem; commonpath raises ValueError instead of just
    returning a mismatch when the two paths are on different drives, so that's caught too."""
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(base)]) == os.path.normcase(base)
    except ValueError:
        return False


def _validate_anchor_path(raw_path: str) -> str:
    """Restrict a caller-supplied anchor_path to an allow-listed directory tree.

    Without this, anchor_path went straight into gen_custom, which uploads it into ComfyUI's
    input folder - from where it's fetchable over ComfyUI's own /view endpoint. That made any
    file on the machine this process can read exfiltratable through this endpoint (point
    anchor_path at an arbitrary file, get its bytes back as an "uploaded" image). realpath +
    commonpath on both sides (not a prefix check on the raw string) so a symlink or a `..`
    segment can't walk back out of the allow-listed tree.
    """
    real = os.path.realpath(raw_path)
    if not os.path.isfile(real):
        raise HTTPException(status_code=400, detail="anchor_path must be an existing file")
    if os.path.splitext(real)[1].lower() not in _ANCHOR_EXTENSIONS:
        raise HTTPException(status_code=400, detail="anchor_path must be a .png/.jpg/.jpeg file")
    if not any(_is_within(real, base) for base in _allowed_anchor_dirs()):
        raise HTTPException(status_code=400, detail="anchor_path is outside the allowed directories")
    return real


def _evict_jobs_locked(now: float) -> None:
    """Drop old finished jobs so a long-running process doesn't grow _jobs forever. Must be
    called with _jobs_lock already held. Pending/running jobs are never evicted - only "done"
    or "error" ones - so a slow generation is never forgotten out from under a polling client.

    Order: TTL first (drops anything finished and older than JOB_TTL_SECONDS, regardless of
    count), then oldest-finished-first while still over MAX_JOBS.
    """
    for job_id in list(_jobs):
        job = _jobs[job_id]
        if job["status"] in ("done", "error") and now - job["created_at"] > JOB_TTL_SECONDS:
            del _jobs[job_id]
    while len(_jobs) > MAX_JOBS:
        for job_id in _jobs:  # OrderedDict: iterates oldest-inserted first
            if _jobs[job_id]["status"] in ("done", "error"):
                del _jobs[job_id]
                break
        else:
            break  # everything left is pending/running - nothing safe to evict, stop even if over cap


def _run_job(job_id: str, req: GenerateRequest) -> None:
    try:
        anchor_path = req.anchor_path or (_default_anchor_for(req.trigger) if req.trigger else None)
        gc.gen_custom(
            prompt=req.prompt,
            extra_negative="",
            tier=req.tier,
            trigger=req.trigger,
            anchor_path=anchor_path,
            out_dir=OUTPUT_DIR,
            seed=random.randint(0, 2**31 - 1),
            filename=job_id,
            ip_adapter_weight=gc.client.IP_ADAPTER_WEIGHT,
            use_facedetailer=req.use_facedetailer,
            hq=req.hq,
        )
        image_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
        with _jobs_lock:
            # Update in place, don't replace the dict wholesale - it would drop created_at,
            # which _evict_jobs_locked needs and which GET /generate/{job_id} deliberately
            # doesn't expose. The job is guaranteed to still be present: pending/running jobs
            # (this one, until this line) are never evicted.
            _jobs[job_id].update(status="done", image_path=image_path, error=None)
    except BaseException as exc:  # noqa: BLE001 - surface any failure to the polling client, don't crash the worker thread
        # BaseException, not Exception: caller errors arrive as gc.UsageError, but the scripts
        # underneath (pose_skeletons, talking_head) can still raise SystemExit, which derives from
        # BaseException. `except Exception` let those escape, killing this worker thread while the
        # job stayed "pending" forever with no error ever reported to the polling client.
        # worker/handler.py catches BaseException for exactly this reason.
        with _jobs_lock:
            _jobs[job_id].update(status="error", image_path=None, error=str(exc))


@app.get("/characters")
def list_characters():
    return [
        {"trigger": trigger, **profile}
        for trigger, profile in gc.CHARACTERS.items()
    ]


@app.post("/generate")
def generate(req: GenerateRequest):
    # Reject an unknown trigger here, synchronously, instead of accepting the job and letting
    # it fail inside the worker thread several seconds later - the caller finds out immediately
    # instead of having to poll a job just to learn its trigger was invalid.
    if req.trigger is not None and req.trigger not in gc.CHARACTERS:
        raise HTTPException(status_code=400, detail=f"Unknown trigger '{req.trigger}'")
    if req.anchor_path is not None:
        # Mutate in place so _run_job (and gen_custom) see the resolved, validated realpath -
        # not the caller's original string - and so validation happens exactly once, here.
        req.anchor_path = _validate_anchor_path(req.anchor_path)

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _evict_jobs_locked(time.time())
        _jobs[job_id] = {"status": "pending", "image_path": None, "error": None, "created_at": time.time()}
    _executor.submit(_run_job, job_id, req)
    return {"job_id": job_id}


@app.get("/generate/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"status": job["status"], "image_path": job["image_path"], "error": job["error"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7862)
