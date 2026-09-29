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

Resource routing (resource_policy): every /generate is decided against the job's cost and the
machine's live state first. A cloud suggestion is a 409 carrying the reasons - this API never sends
work to a paid GPU on its own; the caller either retries with force_local=true or does nothing.
Outside the configured image window the job is deferred (queued with not_before) and job_scheduler,
started with the app, runs it when the window opens. GET /v2/route previews the decision.
"""

import contextlib
import glob
import hmac
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, field_validator

import capability_catalog as catalog
import generate_character as gc
import job_contracts as jc
import job_scheduler
import job_service
import job_store
import resource_policy as rp

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


_PROCESS_STARTED = time.time()


@contextlib.asynccontextmanager
async def _lifespan(app):
    """Start the deferred-job scheduler with the server (not at import, so tests and the restart
    test that re-imports this module never spawn a thread). Queued jobs older than this process
    lost their executor with the previous one; adopt_orphans hands them to the scheduler."""
    adopted = job_scheduler.adopt_orphans(_store, before=_PROCESS_STARTED)
    if adopted:
        print(f"[image_api] {len(adopted)} queued job(s) from before the restart handed to the scheduler", flush=True)
    scheduler = job_scheduler.Scheduler(_service)
    scheduler.start()
    try:
        yield
    finally:
        scheduler.stop()


app = FastAPI(title="AI-Image-Lab image_api", version="0.1.0", dependencies=[Depends(_require_api_key)],
              lifespan=_lifespan)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "companion_chat")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=1)  # one at a time - matches BATCH_SIZE=1's
# "one RTX 2070, no concurrent generations" assumption already baked into comfyui_client.py

MAX_JOBS = job_store.MAX_RECORDS
JOB_TTL_SECONDS = job_store.TTL_SECONDS

# Jobs used to live in a module-level OrderedDict, which meant a restart turned every outstanding
# job id into a 404 that a polling client could not tell apart from "that id never existed". They
# are now records on disk, so the answer survives the process. The eviction policy is unchanged -
# job_store.FileJobStore.purge is image_api._evict_jobs_locked's rules moved, not redesigned.
JOB_STORE_DIR = os.environ.get(
    "IMAGE_API_JOB_STORE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "outputs", "_jobs"),
)
_store = job_store.FileJobStore(JOB_STORE_DIR)
_service = job_service.JobService(_store, artifact_root=os.path.join(os.path.dirname(__file__), "..", "outputs"))

# One owner, "local", unless a caller names itself. The ai-companion backend is the only consumer
# and does not need partitioning today; the header exists so that adding a second consumer later
# is a header rather than a migration. It is never read from the request body - a caller must not
# be able to choose whose jobs it is looking at by editing a payload.
DEFAULT_OWNER = "local"

# Every internal status a client polling this API must keep waiting through, collapsed to the
# three words it has always seen. The richer record is available on /v2/jobs/{job_id} for a caller
# that wants it, but the ai-companion backend's contract does not change.
_LEGACY_STATUS = {
    jc.QUEUED: "pending",
    jc.SUBMITTING: "pending",
    jc.RUNNING: "pending",
    jc.SUBMISSION_UNKNOWN: "pending",
    jc.CHECKING: "pending",
    jc.COMPLETED: "done",
    jc.FAILED: "error",
    jc.CANCELLED: "error",
}

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
    # Overrides resource_policy's *advice*: a cloud suggestion (409 otherwise), the image time
    # window, RAM warnings. It does not skip waiting for a hot card to cool (bounded by
    # cooldown_timeout_s), which is protection rather than advice.
    force_local: bool = False

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


def _evict_jobs(now: float) -> int:
    """Drop old finished records so a long-running process doesn't accumulate them forever.

    Delegates to the store, whose purge is this function's former body: TTL first regardless of
    count, then oldest-finished-first while over the cap, and never a job that has not finished -
    so a slow generation is never forgotten out from under a polling client.
    """
    return _store.purge(now=now, ttl_seconds=JOB_TTL_SECONDS, max_records=MAX_JOBS)


def _owner(x_owner_id: str | None) -> str:
    return (x_owner_id or DEFAULT_OWNER).strip() or DEFAULT_OWNER


def _gen_custom_kwargs(job_id: str, req: GenerateRequest, seed: int) -> dict:
    """The exact gen_custom call for a request, as JSON-safe keyword arguments.

    Built once, at request time, and stored on the record: the executor runs it now, and
    job_scheduler can run the very same call after a deferral or a restart without this module.
    """
    return {
        "prompt": req.prompt,
        "extra_negative": "",
        "tier": req.tier,
        "trigger": req.trigger,
        "anchor_path": req.anchor_path or (_default_anchor_for(req.trigger) if req.trigger else None),
        "out_dir": OUTPUT_DIR,
        "seed": seed,
        "filename": job_id,
        "ip_adapter_weight": gc.client.IP_ADAPTER_WEIGHT,
        "use_facedetailer": req.use_facedetailer,
        "hq": req.hq,
    }


def _decide(req: GenerateRequest) -> rp.Decision:
    checkpoint = gc.DEFAULT_CUSTOM_CHECKPOINT
    kind = rp.image_kind(checkpoint, req.hq, default_checkpoint=checkpoint)
    return rp.decide(kind, rp.load_settings(), rp.probe(), force_local=req.force_local)


def _run_job(owner_id: str, job_id: str, req: GenerateRequest, decision: rp.Decision | None = None) -> None:
    try:
        record = _store.get(owner_id, job_id)
        kwargs = record["request"]["kwargs"]
        if decision is not None and decision.route == rp.WAIT:
            decision = rp.wait_until_ready(decision.kind, decision, force_local=req.force_local)
        kind = decision.kind if decision is not None else "txt2img"
        with rp.track(kind, decision.est_seconds if decision is not None else None):
            _service.submit_local(owner_id, job_id, lambda: gc.gen_custom(**kwargs),
                                  output_path=record["request"]["output_path"])
    except BaseException as exc:  # noqa: BLE001 - surface any failure to the polling client, don't crash the worker thread
        # BaseException, not Exception: caller errors arrive as gc.UsageError, but the scripts
        # underneath (pose_skeletons, talking_head) can still raise SystemExit, which derives from
        # BaseException. `except Exception` let those escape, killing this worker thread while the
        # job stayed "pending" forever with no error ever reported to the polling client.
        # worker/handler.py catches BaseException for exactly this reason.
        #
        # submit_local has usually already concluded the record by the time we get here, and its
        # classification is the better one - it knows whether the prompt may have reached ComfyUI.
        # This only has to cover a failure that happened before any of that, such as a bad anchor
        # resolved inside this function, and it must never overwrite a terminal record.
        current = _store.get(owner_id, job_id)
        if current is not None and current["status"] not in jc.TERMINAL_STATUSES:
            _service._conclude_failure(owner_id, job_id, str(exc), kind=type(exc).__name__)


@app.get("/characters")
def list_characters():
    return [
        {"trigger": trigger, **profile}
        for trigger, profile in gc.CHARACTERS.items()
    ]


@app.post("/generate")
def generate(req: GenerateRequest, x_owner_id: str | None = Header(default=None, alias="X-Owner-Id")):
    owner_id = _owner(x_owner_id)
    # Reject an unknown trigger here, synchronously, instead of accepting the job and letting
    # it fail inside the worker thread several seconds later - the caller finds out immediately
    # instead of having to poll a job just to learn its trigger was invalid.
    if req.trigger is not None and req.trigger not in gc.CHARACTERS:
        raise HTTPException(status_code=400, detail=f"Unknown trigger '{req.trigger}'")
    if req.anchor_path is not None:
        # Mutate in place so _run_job (and gen_custom) see the resolved, validated realpath -
        # not the caller's original string - and so validation happens exactly once, here.
        req.anchor_path = _validate_anchor_path(req.anchor_path)

    decision = _decide(req)
    if decision.route == rp.CLOUD:
        # Not created, not queued: a cloud send costs money and needs a human (or a caller that
        # has decided on its own) to say so. force_local=true is the "run it here anyway" retry.
        raise HTTPException(status_code=409, detail={"error": "cloud_suggested", **decision.to_dict()})

    _evict_jobs(time.time())
    checkpoint = gc.DEFAULT_CUSTOM_CHECKPOINT
    mode = catalog.MODE_TXT2IMG_HQ if req.hq else catalog.MODE_TXT2IMG
    # Chosen here rather than inside the worker thread, so the record says which seed produced the
    # image. It used to be picked at generation time and never written down anywhere, which made a
    # result impossible to reproduce.
    params = {
        "prompt": req.prompt,
        "seed": random.randint(0, 2**31 - 1),
        "tier": req.tier,
        "character": req.trigger,
        "checkpoint": checkpoint,
    }
    job_id = uuid.uuid4().hex
    spec = job_scheduler.call_spec(
        "gen_custom", _gen_custom_kwargs(job_id, req, params["seed"]),
        output_path=os.path.join(OUTPUT_DIR, f"{job_id}.png"), force_local=req.force_local,
    )
    deferred = decision.route == rp.DEFER
    record = _service.create(
        owner_id,
        mode=mode,
        catalog_id=f"{checkpoint}:{mode}",
        params=params,
        request={"hq": req.hq, "use_facedetailer": req.use_facedetailer, "anchor_path": req.anchor_path, **spec},
        job_id=job_id,
        not_before=decision.not_before if deferred else None,
        route=decision.to_dict(),
    )
    if deferred:
        # The legacy poll keeps answering "pending" until the scheduler has run it.
        return {"job_id": record["job_id"], "route": decision.route, "not_before": decision.not_before}
    _executor.submit(_run_job, owner_id, record["job_id"], req, decision)
    return {"job_id": record["job_id"], "route": decision.route}


@app.get("/v2/route")
def preview_route(hq: bool = True, force_local: bool = False):
    """What /generate would do with this request right now, without creating anything."""
    return _decide(GenerateRequest(prompt="preview", hq=hq, force_local=force_local)).to_dict()


@app.get("/generate/{job_id}")
def get_job(job_id: str, x_owner_id: str | None = Header(default=None, alias="X-Owner-Id")):
    record = _store.get(_owner(x_owner_id), job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    # Deliberately the legacy OUTPUT_DIR/<job_id>.png path, not the artifact's content-addressed
    # one. The ai-companion backend opens whatever this returns, and both names refer to the same
    # bytes (the service adds the second name, it does not move the file), so keeping this one
    # means the migration is invisible to that caller. The content-addressed artifact is on
    # /v2/jobs/{job_id} for anything that wants provenance instead of a file to read.
    completed = record["status"] == jc.COMPLETED
    return {
        "status": _LEGACY_STATUS[record["status"]],
        "image_path": os.path.join(OUTPUT_DIR, f"{job_id}.png") if completed else None,
        "error": record["last_error"],
    }


@app.get("/v2/jobs/{job_id}")
def get_job_v2(job_id: str, x_owner_id: str | None = Header(default=None, alias="X-Owner-Id")):
    """The whole record: the real status, the submit attempt, the output check report, the
    content-addressed artifact. Separate from /generate/{job_id} so the ai-companion backend's
    existing three-field contract keeps working untouched."""
    record = _store.get(_owner(x_owner_id), job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return record


@app.post("/v2/jobs/{job_id}/cancel")
def cancel_job_v2(job_id: str, x_owner_id: str | None = Header(default=None, alias="X-Owner-Id")):
    """Record the intent to cancel. A job still queued is concluded at once; one already running
    stops at its next poll, which is why this is a timestamp on the record rather than a status."""
    try:
        record = _service.request_cancel(_owner(x_owner_id), job_id)
    except job_store.JobNotFound:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None
    return {"status": record["status"], "cancel_requested_at": record["cancel_requested_at"]}


@app.get("/v2/catalog")
def get_catalog(checkpoint: str | None = None):
    """What each checkpoint can be asked for, disabled rows included with their reasons.

    Disabled rows are returned rather than filtered out: a caller that only ever sees what works
    cannot tell "this build has no ControlNet" from "not with this checkpoint, and here is why".
    """
    try:
        targets = [checkpoint] if checkpoint else list(catalog.checkpoints())
        return [
            {"catalog_id": row.catalog_id, "checkpoint": row.checkpoint, "family": row.family,
             "kind": row.kind, "name": row.name, "enabled": row.enabled, "reason": row.reason,
             "params": row.params}
            for name in targets for row in catalog.rows_for(name)
        ]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown checkpoint '{checkpoint}'") from None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7862)
