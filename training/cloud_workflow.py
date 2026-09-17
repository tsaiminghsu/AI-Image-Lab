"""Run an already-built ComfyUI workflow on this repo's RunPod worker (jobType "workflow").

This is what comfyui_client's "runpod" backend calls. Every local generation - HQ stills with
FaceID and pose skeletons, Z-Image, SD1.5, GIF frames - ends in comfyui_client._submit_and_wait
with a finished workflow dict. Instead of posting that to the local ComfyUI, it is sent to the worker,
which runs it through its own copy of the same _submit_and_wait next to a bigger GPU. So nothing in
the worker has to know about any individual feature; it needs the nodes and the models, not the logic.

The video jobs in cloud_video.py (Wan 2.2 I2V, AnimateDiff) stay separate: those are high-level jobs
the worker composes itself. This module reuses their RunPod client, polling and cancellation.

Safety, in both places: before sending, the graph is checked here (workflow_safety) and the cfg floor
applied; the worker repeats both before touching the GPU, so a caller holding the endpoint key cannot
skip them by sending its own JSON.

Reference images: a serverless worker shares no disk with this machine, so upload_reference_image
stages bytes here (thread-local, under a content-hash name) and they travel inside the job that
references them. Content-hash names keep gen_gif's "upload once, submit N frames" pattern correct
across stateless workers, and make re-staging the same file a no-op.
"""

import base64
import collections
import hashlib
import json
import os
import re
import threading
import time

import cloud_video as cv
import comfyui_client as client
import generate_character as gc
import workflow_safety

JOB_TYPE = "workflow"
# A cold worker has to pull the image and load models from the network volume before the job's own
# timeout clock is meaningful, so the local wait gets this much on top of the workflow's timeout.
COLD_START_ALLOWANCE_SECONDS = int(os.environ.get("RUNPOD_COLD_START_SECONDS", "600"))
# Staged uploads are held in memory until the job that uses them is sent.
MAX_STAGED_BYTES = 64_000_000
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

_staged = threading.local()


def _staging():
    store = getattr(_staged, "store", None)
    if store is None:
        store = _staged.store = collections.OrderedDict()
    return store


def stage_upload(local_path):
    """Hold an image for the next cloud job and return the name its LoadImage node should use."""
    try:
        with open(local_path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise gc.UsageError(f"讀不到參考圖：{local_path}（{exc}）") from exc
    if raw.startswith(cv._PNG_MAGIC):
        ext = ".png"
    elif raw.startswith(cv._JPEG_MAGIC):
        ext = ".jpg"
    else:
        raise gc.UsageError(f"參考圖必須是 PNG 或 JPEG：{local_path}")
    stem = _SAFE_NAME.sub("_", os.path.splitext(os.path.basename(local_path))[0]).strip("._")[:60] or "image"
    name = f"{hashlib.sha256(raw).hexdigest()[:16]}_{stem}{ext}"
    store = _staging()
    store.pop(name, None)
    store[name] = raw
    while sum(len(v) for v in store.values()) > MAX_STAGED_BYTES and len(store) > 1:
        store.popitem(last=False)
    return name


def clear_staged():
    _staging().clear()


def collect_input_images(wf):
    """{name: base64} for exactly the staged images this workflow's LoadImage nodes read."""
    store = _staging()
    images = {}
    for name in workflow_safety.load_image_names(wf):
        if name not in store:
            raise gc.UsageError(f"工作流程要讀取圖片「{name}」，但雲端模式下沒有先暫存這張圖"
                                "（應該先呼叫 upload_reference_image）")
        images[name] = base64.b64encode(store[name]).decode("ascii")
    return images


def check_workflow(wf, output_node_id):
    """Local copy of the worker's checks, so a bad graph fails here for free."""
    try:
        workflow_safety.validate_graph(wf, output_node_id)
        workflow_safety.check_negative_safety(wf, required=gc.AGE_SAFETY_NEGATIVE)
    except workflow_safety.WorkflowRejected as exc:
        raise gc.UsageError(f"這個工作流程不能送到雲端：{exc}") from exc


def build_workflow_input(wf, output_node_id, timeout_seconds, images):
    inp = {
        "jobType": JOB_TYPE,
        "workflow": wf,
        "outputNodeId": str(output_node_id),
        "inputImages": images,
        "timeout": int(timeout_seconds),
    }
    size = len(json.dumps({"input": inp}).encode("utf-8"))
    if size > cv.RUNPOD_MAX_PAYLOAD_BYTES:
        raise gc.UsageError(f"送出的資料太大（{size} bytes，RunPod /run 上限約 10 MB）——參考圖太大，請換小一點的圖")
    return inp


def submit_and_wait(wf, output_node_id="9", timeout_seconds=client.POLL_TIMEOUT_SECONDS, *, progress=None,
                    session=None, settings=None, env=None, sleep=time.sleep, clock=time.monotonic):
    """Run `wf` on the worker and return a local path in raw_output_dir(), exactly like the local
    _submit_and_wait, so callers move the file the same way whichever backend ran it."""
    client.enforce_min_cfg(wf)
    check_workflow(wf, output_node_id)
    images = collect_input_images(wf)
    settings = settings or cv.load_cloud_settings()
    backend = cv.make_backend("runpod", session=session, settings=settings, env=env, sleep=sleep)
    inp = build_workflow_input(wf, output_node_id, timeout_seconds, images)

    # The worker's own timeout plus model loading has to fit in RunPod's execution limit; the local
    # wait additionally covers time spent queued and cold-starting.
    execution_timeout_s = max(settings["runpod"]["execution_timeout_s"], timeout_seconds + 300)
    max_wait_s = timeout_seconds + COLD_START_ALLOWANCE_SECONDS

    def on_status(label, elapsed, limit):
        if progress is not None:
            progress(f"{label}（已等 {elapsed} 秒）", elapsed, None)

    job_id = backend.submit(inp, execution_timeout_s)
    on_status(f"已送出雲端工作 {job_id}", 0, max_wait_s)
    data = cv.wait_for_job(backend, job_id, max_wait_s=max_wait_s, on_status=on_status, clock=clock, sleep=sleep)
    path, _queue_s, _execution_s = backend.result(data, client.raw_output_dir(), f"cloud_{str(job_id)[:12]}")
    return path
