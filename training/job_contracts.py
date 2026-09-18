"""Contract layer for the job lifecycle: what a job record is, what an artifact is, and which
status transitions are legal.

Deliberately stdlib-only, and deliberately not pydantic. comfyui_client.py imports only
`requests` so it runs where there is no GPU, and worker/jobs.py imports neither boto3 nor
runpod so it tests offline; this module has to be importable from both. It also must not
import generate_character: comfyui_client is imported *by* generate_character, so the back
edge would be circular. That is why ContractError derives from Exception here instead of from
gc.UsageError - the callers that face a user translate it.

Two rules shape every validator below.

Loose outside, strict inside. An unknown *top-level* key on a job record survives a
load -> save round trip verbatim, so a record written by a newer version of this code is not
silently truncated by an older one reading it. An unknown key inside `params`, `artifacts[]`
or `output_check` is an error, because those sub-objects carry authority: they say what was
asked for, what actually came back, and whether anyone checked it. Forward compatibility is
worth having on the envelope and actively harmful on the payload.

A terminal success must carry evidence. A record with status "completed" and no artifact is
refused here, not by convention at the call sites.
"""

import json
import re

SCHEMA_VERSION = 1

# An artifact is a content-addressed pointer - (bucket, key, sha256) - never a URL. A URL is a
# short-lived grant derived from an artifact, so storing one would bake an expiry into a record
# that outlives it. Resolution is a separate, tagged step (see job_store.resolve_artifact).
MEDIA_KINDS = ("image", "video")

# `operation` is derived, never typed by a human. It exists because the borrowed design has it
# and it costs nothing, but here it collapses to a function of media_kind.
OPERATION_BY_MEDIA_KIND = {"image": "image_generation", "video": "video_generation"}

CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
}

# A relative posix key. The character class already rejects a leading "/", a backslash and a
# "C:" drive prefix; _validate_key additionally rejects ".." and empty segments, which a regex
# of this shape cannot express readably.
ARTIFACT_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,239}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

QUEUED = "queued"
SUBMITTING = "submitting"
RUNNING = "running"
SUBMISSION_UNKNOWN = "submission_unknown"
CHECKING = "checking"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

STATUSES = (QUEUED, SUBMITTING, RUNNING, SUBMISSION_UNKNOWN, CHECKING, COMPLETED, FAILED, CANCELLED)
TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, CANCELLED})

# The transition table is data so a test can assert reachability and that terminal states have
# no outgoing edges, rather than re-deriving the rules by reading branches.
#
# Two edges are absent on purpose:
#   * running -> completed. Completion must pass through `checking`, which is where the output
#     check re-reads the produced bytes. Allowing the shortcut would make the check skippable.
#   * submission_unknown -> submitting. An uncertain submission is resolved by reconciling with
#     the provider, never by submitting again: a re-submit can double-charge a GPU, or double-
#     queue an 8 GB card, against a job that was in fact accepted.
_ALLOWED = {
    QUEUED: frozenset({SUBMITTING, CANCELLED, FAILED}),
    SUBMITTING: frozenset({RUNNING, SUBMISSION_UNKNOWN, FAILED, CANCELLED}),
    RUNNING: frozenset({CHECKING, SUBMISSION_UNKNOWN, FAILED, CANCELLED}),
    SUBMISSION_UNKNOWN: frozenset({RUNNING, CHECKING, FAILED, CANCELLED}),
    CHECKING: frozenset({COMPLETED, FAILED, CANCELLED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}

BACKENDS = ("local", "runpod", "replicate")
PROVIDERS = ("comfyui", "runpod", "replicate")

# Strict key set for `params`. Phase 4's capability catalog narrows this further per row (a
# Z-Image row has no `lora_strength`); this is the outer bound shared by every mode, and it is
# here so a typo like "witdh" fails at the store's write path instead of silently doing nothing.
PARAM_KEYS = frozenset({
    "prompt", "negative_prompt", "width", "height", "seed", "steps", "cfg", "denoise",
    "checkpoint", "tier", "character", "lora_strength", "frames", "fps", "batch_size",
})

OUTPUT_CHECK_STATES = ("passed", "failed", "manual_review_required")
OUTPUT_CHECK_SCOPE = "technical_delivery"

# Cost and timing are a nullable tri-state: a real 0 means "the provider reported zero", None
# means "nobody told us". Consumers must render None as unknown and never infer a value from
# the other fields - once displayed, a derived number is indistinguishable from a measured one.
COST_FIELDS = (
    "provider_delay_ms",
    "provider_execution_ms",
    "gpu_seconds",
    "estimated_cost_microunits",
    "actual_cost_microunits",
)

# The camelCase spellings the Amplify Lambdas already use (web/amplify/functions/*). Kept in one
# place so a future DynamoDB engine translates here rather than guessing field by field.
TS_FIELD_MAP = {
    "job_id": "jobId",
    "owner_id": "ownerId",
    "provider_job_id": "providerJobId",
    "submit_attempt": "submitAttempt",
    "cancel_requested_at": "cancelRequestedAt",
    "created_at": "createdAt",
    "updated_at": "updatedAt",
    "output_check": "outputCheck",
    "last_error": "lastError",
}


class ContractError(Exception):
    """A record or artifact violates the contract. The message is safe to show a caller."""


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _validate_key(key):
    _require(isinstance(key, str), f"artifact key must be a string, got {key!r}")
    _require(bool(ARTIFACT_KEY_RE.match(key)), f"unsafe artifact key {key!r}")
    segments = key.split("/")
    _require(all(segments), f"unsafe artifact key {key!r} (empty path segment)")
    _require(".." not in segments, f"unsafe artifact key {key!r} (parent traversal)")
    # "." is harmless on its own but means the key was never normalised, and an unnormalised key
    # compares unequal to its own normalised form - which is how a content address stops being one.
    _require("." not in segments, f"unsafe artifact key {key!r} (unnormalised path segment)")
    return key


class Artifact:
    """One immutable output: (bucket, key, sha256) plus enough metadata to serve it.

    Strict - an unknown field is an error. This object is what a completed job points at, so a
    field nobody validated must never ride along inside it.
    """

    __slots__ = ("bucket", "key", "sha256", "media_kind", "content_type", "byte_length")

    def __init__(self, bucket, key, sha256, media_kind, content_type, byte_length):
        _require(isinstance(bucket, str) and 1 <= len(bucket) <= 128, f"bad artifact bucket {bucket!r}")
        self.bucket = bucket
        self.key = _validate_key(key)
        _require(isinstance(sha256, str) and bool(SHA256_RE.match(sha256)),
                 f"artifact sha256 must be 64 lowercase hex, got {sha256!r}")
        self.sha256 = sha256
        _require(media_kind in MEDIA_KINDS, f"artifact media_kind must be one of {MEDIA_KINDS}, got {media_kind!r}")
        self.media_kind = media_kind
        _require(isinstance(content_type, str) and 1 <= len(content_type) <= 128,
                 f"bad artifact content_type {content_type!r}")
        self.content_type = content_type
        _require(isinstance(byte_length, int) and not isinstance(byte_length, bool) and byte_length >= 0,
                 f"artifact byte_length must be a non-negative integer, got {byte_length!r}")
        self.byte_length = byte_length

    @classmethod
    def from_dict(cls, data):
        _require(isinstance(data, dict), f"artifact must be an object, got {type(data).__name__}")
        unknown = sorted(set(data) - set(cls.__slots__))
        _require(not unknown, f"unknown artifact field(s): {', '.join(unknown)}")
        missing = sorted(set(cls.__slots__) - set(data))
        _require(not missing, f"missing artifact field(s): {', '.join(missing)}")
        return cls(**{name: data[name] for name in cls.__slots__})

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __eq__(self, other):
        return isinstance(other, Artifact) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return f"Artifact(bucket={self.bucket!r}, key={self.key!r}, sha256={self.sha256[:12]}...)"


def can_transition(current, nxt):
    """True when `current -> nxt` is a legal edge. A same-status write is legal: a poll bumps
    poll_count without moving the status."""
    _require(current in _ALLOWED, f"unknown status {current!r}")
    _require(nxt in _ALLOWED, f"unknown status {nxt!r}")
    return current == nxt or nxt in _ALLOWED[current]


def require_transition(current, nxt):
    if not can_transition(current, nxt):
        raise ContractError(f"illegal status transition {current!r} -> {nxt!r}")
    return nxt


def _validate_params(params):
    _require(isinstance(params, dict), f"params must be an object, got {type(params).__name__}")
    unknown = sorted(set(params) - PARAM_KEYS)
    _require(not unknown, f"unknown param(s): {', '.join(unknown)}")
    for name, value in params.items():
        _require(isinstance(value, (str, int, float, bool)) or value is None,
                 f"param {name!r} must be a JSON scalar, got {type(value).__name__}")


def _validate_output_check(report):
    _require(isinstance(report, dict), f"output_check must be an object, got {type(report).__name__}")
    unknown = sorted(set(report) - {"policy", "state", "scope", "details"})
    _require(not unknown, f"unknown output_check field(s): {', '.join(unknown)}")
    policy = report.get("policy")
    _require(isinstance(policy, dict), "output_check.policy must be an object")
    policy_unknown = sorted(set(policy) - {"id", "version", "checks", "sha256"})
    _require(not policy_unknown, f"unknown output_check.policy field(s): {', '.join(policy_unknown)}")
    _require(isinstance(policy.get("id"), str) and policy["id"], "output_check.policy.id must be a non-empty string")
    version = policy.get("version")
    _require(isinstance(version, int) and not isinstance(version, bool) and version > 0,
             "output_check.policy.version must be a positive integer")
    checks = policy.get("checks")
    _require(isinstance(checks, list) and all(isinstance(c, str) for c in checks),
             "output_check.policy.checks must be a list of strings")
    _require(isinstance(policy.get("sha256"), str) and bool(SHA256_RE.match(policy.get("sha256") or "")),
             "output_check.policy.sha256 must be 64 lowercase hex")
    _require(report.get("state") in OUTPUT_CHECK_STATES,
             f"output_check.state must be one of {OUTPUT_CHECK_STATES}, got {report.get('state')!r}")
    # A literal, never computed: the report must not be able to widen the claim it makes.
    _require(report.get("scope") == OUTPUT_CHECK_SCOPE,
             f"output_check.scope must be the literal {OUTPUT_CHECK_SCOPE!r}")


def new_record(*, job_id, owner_id, mode, media_kind, catalog_id, backend, params, request=None, now):
    """Build a fresh record in `queued`. Every other field starts at its zero value, so the
    shape of a record never depends on which code path created it."""
    _require(isinstance(job_id, str) and job_id, "job_id must be a non-empty string")
    _require(isinstance(owner_id, str) and owner_id, "owner_id must be a non-empty string")
    _require(media_kind in MEDIA_KINDS, f"media_kind must be one of {MEDIA_KINDS}, got {media_kind!r}")
    _require(backend in BACKENDS, f"backend must be one of {BACKENDS}, got {backend!r}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "owner_id": owner_id,
        "mode": mode,
        "operation": OPERATION_BY_MEDIA_KIND[media_kind],
        "media_kind": media_kind,
        "catalog_id": catalog_id,
        "status": QUEUED,
        "cancel_requested_at": None,
        "submit_attempt": 0,
        "backend": backend,
        "provider": None,
        "provider_job_id": None,
        "params": dict(params),
        "request": dict(request or {}),
        "artifacts": [],
        "output_check": None,
        "created_at": now,
        "updated_at": now,
        "submitted_at": None,
        "completed_at": None,
        "poll_count": 0,
        "last_error": None,
        "last_error_kind": None,
    }
    record.update({name: None for name in COST_FIELDS})
    return validate_record(record)


def validate_record(record):
    """Validate in place and return the record. Unknown top-level keys are left untouched."""
    _require(isinstance(record, dict), f"record must be an object, got {type(record).__name__}")
    for name in ("schema_version", "job_id", "owner_id", "media_kind", "status", "submit_attempt", "params"):
        _require(name in record, f"record is missing {name!r}")
    _require(isinstance(record["job_id"], str) and record["job_id"], "job_id must be a non-empty string")
    _require(isinstance(record["owner_id"], str) and record["owner_id"], "owner_id must be a non-empty string")
    _require(record["media_kind"] in MEDIA_KINDS, f"bad media_kind {record['media_kind']!r}")
    _require(record.get("operation") == OPERATION_BY_MEDIA_KIND[record["media_kind"]],
             "operation is derived from media_kind and must match it")
    _require(record["status"] in STATUSES, f"unknown status {record['status']!r}")
    _require(record.get("backend") in BACKENDS, f"bad backend {record.get('backend')!r}")
    _require(record.get("provider") is None or record["provider"] in PROVIDERS,
             f"bad provider {record.get('provider')!r}")
    attempt = record["submit_attempt"]
    _require(isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0,
             f"submit_attempt must be a non-negative integer, got {attempt!r}")
    cancel_at = record.get("cancel_requested_at")
    _require(cancel_at is None or (isinstance(cancel_at, (int, float)) and not isinstance(cancel_at, bool)),
             "cancel_requested_at must be a timestamp or None")
    _validate_params(record["params"])

    artifacts = record.get("artifacts")
    _require(isinstance(artifacts, list), "artifacts must be a list")
    record["artifacts"] = [a.to_dict() if isinstance(a, Artifact) else Artifact.from_dict(a).to_dict()
                           for a in artifacts]

    if record.get("output_check") is not None:
        _validate_output_check(record["output_check"])

    for name in COST_FIELDS:
        value = record.get(name)
        _require(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0),
                 f"{name} must be a non-negative number or None, got {value!r}")

    # The one invariant worth stating twice: a completed job is a claim that something exists.
    if record["status"] == COMPLETED:
        _require(record["artifacts"], "a completed job must carry at least one output artifact")
    return record


def dumps(record):
    """Serialise for the store. sort_keys keeps a record byte-identical regardless of which path
    wrote it, which is what lets a test assert the poller and the webhook agree exactly."""
    return json.dumps(validate_record(record), ensure_ascii=False, sort_keys=True, indent=2)


def loads(raw):
    data = json.loads(raw)
    _require(isinstance(data, dict), "stored record must be a JSON object")
    return validate_record(data)
