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
import os
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import generate_character as gc

app = FastAPI(title="AI-Image-Lab image_api", version="0.1.0")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "companion_chat")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=1)  # one at a time - matches BATCH_SIZE=1's
# "one RTX 2070, no concurrent generations" assumption already baked into comfyui_client.py
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


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


def _default_anchor_for(trigger: str) -> str | None:
    """Picks any existing anchor image for a known character so callers don't
    need to know exact filenames - mirrors what a human operator would do
    manually via gui.py's anchor dropdown."""
    pattern = os.path.join(os.path.dirname(__file__), "reference_candidates", trigger, "anchor_*.png")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


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
            _jobs[job_id] = {"status": "done", "image_path": image_path, "error": None}
    except BaseException as exc:  # noqa: BLE001 - surface any failure to the polling client, don't crash the worker thread
        # BaseException, not Exception: generate_character signals invalid arguments with
        # SystemExit (get_character on an unknown trigger, gen_custom's flag checks), which is a
        # BaseException. `except Exception` let those escape, killing this worker thread while the
        # job stayed "pending" forever with no error ever reported to the polling client.
        # worker/handler.py:179 catches BaseException for exactly this reason.
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "image_path": None, "error": str(exc)}


@app.get("/characters")
def list_characters():
    return [
        {"trigger": trigger, **profile}
        for trigger, profile in gc.CHARACTERS.items()
    ]


@app.post("/generate")
def generate(req: GenerateRequest):
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "image_path": None, "error": None}
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
