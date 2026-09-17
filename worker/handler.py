"""RunPod Serverless handler for AI-Image-Lab.

Runs inside the worker container next to a ComfyUI server on localhost:8188. What to generate is
decided by jobs.py, which reuses the EXACT local generation code (training/comfyui_client.py +
generate_character.py), so the age/tier safety negatives and the cfg floor are the same hard
guarantee they are locally. That is why runpod.ts and training/cloud_video.py send the raw prompt +
tier and let this worker compose the negatives, rather than double-applying them.

This module only does the RunPod-facing parts: S3 upload, progress updates, the entry point. boto3
and runpod are imported lazily so tests/test_worker_handler.py can import it without either.

Input:  {"input": {...}} - see jobs.py's docstring for the per-jobType schema.
Output: {"outputKey", "outputUrl", "seed", "width", "height", "jobType", ...}
On any failure returns {"error": "..."} so RunPod marks the job FAILED.
"""

import os
import shutil
import tempfile
import time

import comfyui_client as client
import jobs

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
WORKER_CHECKPOINT = os.environ.get("WORKER_CHECKPOINT", "cyberrealistic_pony")
ANCHOR_DIR = os.environ.get("RUNPOD_ANCHOR_DIR", "/runpod-volume/anchors")
PRESIGN_SECONDS = int(os.environ.get("PRESIGN_SECONDS", "3600"))
# RunPod progress updates are HTTP calls - one per sampler step would be dozens a minute.
PROGRESS_MIN_INTERVAL_SECONDS = 5.0

_progress_update = None   # runpod.serverless.progress_update, set in __main__
_s3 = None


def make_s3():
    """S3 client. S3_ENDPOINT_URL makes it work against S3-compatible stores (Cloudflare R2 takes
    https://<account>.r2.cloudflarestorage.com with AWS_REGION=auto); unset means AWS S3."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
    )


def _progress_callback(job):
    """(stage, current, total) -> throttled runpod progress_update; never raises into the job."""
    state = {"stage": None, "at": 0.0}

    def callback(stage, current, total):
        if _progress_update is None:
            return
        now = time.monotonic()
        if stage == state["stage"] and now - state["at"] < PROGRESS_MIN_INTERVAL_SECONDS and current < total:
            return
        state.update(stage=stage, at=now)
        try:
            _progress_update(job, {"stage": stage, "current": current, "total": total})
        except Exception:
            pass

    return callback


def handler(job, *, s3_factory=None):
    global _s3
    tmp_files = []
    out_dir = None
    try:
        if not S3_BUCKET:
            return {"error": "S3_BUCKET env var not set on the worker"}
        job_id = job.get("id", "unknown")
        spec = jobs.parse_job(job.get("input") or {}, worker_checkpoint=WORKER_CHECKPOINT)

        out_dir = tempfile.mkdtemp(prefix=f"out_{job_id}_")
        started = time.monotonic()
        with client.progress_reporter(_progress_callback(job)):
            result = jobs.run_job(spec, job_id=job_id, anchor_dir=ANCHOR_DIR, out_dir=out_dir, tmp_files=tmp_files)

        if _s3 is None or s3_factory is not None:
            _s3 = (s3_factory or make_s3)()
        key = f"generated/{job_id}.{result['ext']}"
        _s3.upload_file(result["localPath"], S3_BUCKET, key, ExtraArgs={"ContentType": result["contentType"]})
        url = _s3.generate_presigned_url(
            "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=PRESIGN_SECONDS,
        )
        out = {k: v for k, v in result.items() if k not in ("localPath", "contentType", "ext")}
        out.update(outputKey=key, outputUrl=url, elapsedSeconds=round(time.monotonic() - started, 1))
        return out

    except jobs.JobInputError as exc:
        return {"error": str(exc)}
    except BaseException as exc:  # gc.UsageError included, and anything else a handler can hit
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    import runpod

    _progress_update = runpod.serverless.progress_update
    runpod.serverless.start({"handler": handler})
