"""Boundary behaviour of the FastAPI wrapper.

image_api hands every request to generate_character.gen_custom on a single worker thread and
reports the outcome only through the job dict a client polls. That makes the exception boundary
in _run_job the one place a failure can be lost: anything it does not catch kills the worker
thread silently and leaves the job "pending" forever.
"""

import os
import sys

import pytest
from test_output_check import make_png

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


@pytest.fixture
def api(monkeypatch, tmp_path):
    """A TestClient over a freshly imported image_api, with its executor made synchronous so a
    POST has already run the job by the time it returns.

    The job store directory is redirected into tmp_path and set before the import, because
    image_api builds its store at module scope - without that every test would share, and slowly
    fill, the repo's own outputs/_jobs.
    """
    sys.modules.pop("image_api", None)
    monkeypatch.setenv("IMAGE_API_JOB_STORE_DIR", str(tmp_path / "jobs"))
    import image_api

    class _Inline:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

    monkeypatch.setattr(image_api, "_executor", _Inline())
    monkeypatch.setattr(image_api, "OUTPUT_DIR", str(tmp_path / "out"))
    os.makedirs(str(tmp_path / "out"), exist_ok=True)
    return fastapi_testclient.TestClient(image_api.app), image_api


def writes_a_real_png(image_api, monkeypatch, width=64, height=64):
    """Replace gen_custom with something that actually produces a decodable file.

    A no-op mock is no longer enough: the job only reaches "done" if the output check can read the
    bytes back, which is the point of having the check. What the mock has to do is what the real
    generator does - leave a file at OUTPUT_DIR/<filename>.png.
    """

    def fake_gen_custom(*args, **kwargs):
        path = os.path.join(image_api.OUTPUT_DIR, f"{kwargs['filename']}.png")
        with open(path, "wb") as handle:
            handle.write(make_png(width, height))

    monkeypatch.setattr(image_api.gc, "gen_custom", fake_gen_custom)


def test_unknown_character_is_reported_as_an_error_not_left_pending(api, monkeypatch):
    """The regression this file exists for.

    generate_character signals invalid arguments with SystemExit, which derives from
    BaseException. While _run_job caught only Exception, a request naming an unknown character
    killed the worker thread and the job never left "pending" - the polling client waited
    forever with no error. Verified against the pre-fix code: this test hangs on "pending".

    No `trigger` in the request: an unknown trigger is now rejected synchronously with a 400
    before the job is even created (see test_unknown_trigger_is_400), so it can no longer reach
    gen_custom to exercise this path. gen_custom can still raise "unknown character" for other
    reasons (e.g. a stale LoRA registry entry), so the SystemExit-handling regression this test
    guards is simulated directly on the mock instead.
    """
    client, image_api = api

    def boom(*args, **kwargs):
        raise SystemExit("unknown character 'nobody'")

    monkeypatch.setattr(image_api.gc, "gen_custom", boom)

    job_id = client.post("/generate", json={"prompt": "portrait photo"}).json()["job_id"]
    body = client.get(f"/generate/{job_id}").json()

    assert body["status"] == "error", f"job should report the failure, got {body}"
    assert "unknown character" in body["error"]


def test_ordinary_exception_is_still_reported(api, monkeypatch):
    """Widening to BaseException must not have narrowed the ordinary path."""
    client, image_api = api
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("comfy down")))

    job_id = client.post("/generate", json={"prompt": "x"}).json()["job_id"]
    body = client.get(f"/generate/{job_id}").json()
    assert body["status"] == "error"
    assert "comfy down" in body["error"]


def test_successful_job_reports_done_with_a_path(api, monkeypatch):
    """image_path stays the OUTPUT_DIR/<job_id>.png the ai-companion backend has always opened,
    even though the job record now also carries a content-addressed artifact key."""
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)

    job_id = client.post("/generate", json={"prompt": "x"}).json()["job_id"]
    body = client.get(f"/generate/{job_id}").json()
    assert body["status"] == "done"
    assert body["image_path"].endswith(f"{job_id}.png")
    assert body["error"] is None


def test_a_generator_that_produces_no_file_is_an_error_not_a_success(api, monkeypatch):
    """This used to report "done" with a path to a file that did not exist, because nothing
    checked. The output check is what turns that into a failure the caller can see."""
    client, image_api = api
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)

    job_id = client.post("/generate", json={"prompt": "x"}).json()["job_id"]
    body = client.get(f"/generate/{job_id}").json()
    assert body["status"] == "error"
    assert body["image_path"] is None
    assert "does not exist" in body["error"]


def test_unknown_job_id_is_404(api):
    client, _ = api
    assert client.get("/generate/does-not-exist").status_code == 404


def test_characters_endpoint_lists_every_character(api):
    client, image_api = api
    rows = client.get("/characters").json()
    assert {r["trigger"] for r in rows} == set(image_api.gc.CHARACTERS)
    assert all(r["age"] >= image_api.gc.MINIMUM_AGE for r in rows)


# --- prompt validation -------------------------------------------------------------------


def test_empty_prompt_is_422(api):
    client, _ = api
    assert client.post("/generate", json={"prompt": ""}).status_code == 422


def test_whitespace_only_prompt_is_422(api):
    client, _ = api
    assert client.post("/generate", json={"prompt": "   \n\t  "}).status_code == 422


def test_prompt_over_cap_is_422(api, monkeypatch):
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    prompt = "a" * (image_api.gc.MAX_PROMPT_CHARS + 1)
    assert client.post("/generate", json={"prompt": prompt}).status_code == 422


def test_prompt_exactly_at_cap_is_accepted(api, monkeypatch):
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    prompt = "a" * image_api.gc.MAX_PROMPT_CHARS
    resp = client.post("/generate", json={"prompt": prompt})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert client.get(f"/generate/{job_id}").json()["status"] == "done"


# --- trigger validation ------------------------------------------------------------------


def test_unknown_trigger_is_400(api):
    client, _ = api
    resp = client.post("/generate", json={"prompt": "x", "trigger": "nobody"})
    assert resp.status_code == 400


def test_valid_trigger_is_accepted(api, monkeypatch):
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    known_trigger = next(iter(image_api.gc.CHARACTERS))
    resp = client.post("/generate", json={"prompt": "x", "trigger": known_trigger})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert client.get(f"/generate/{job_id}").json()["status"] == "done"


# --- anchor_path containment ---------------------------------------------------------------
# Allow-list is monkeypatched to a tmp_path subtree rather than writing into the real
# reference_candidates/ directory - cleaner than creating-and-cleaning-up a file in a
# directory that's part of the checkout, and doesn't depend on that directory's contents.


def test_anchor_path_outside_allowlist_is_400(api):
    client, _ = api
    # A real file that exists but is nowhere near reference_candidates/.
    resp = client.post("/generate", json={"prompt": "x", "anchor_path": r"C:\Windows\win.ini"})
    assert resp.status_code == 400


def test_anchor_path_traversal_outside_allowlist_is_400(api, monkeypatch, tmp_path):
    client, image_api = api
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(image_api.gc, "REFERENCE_CANDIDATES_DIR", str(allowed))

    escape_dir = tmp_path / "escape"
    escape_dir.mkdir()
    secret = escape_dir / "secret.png"
    secret.write_bytes(b"not really a png, contents don't matter for this check")

    traversal_path = str(allowed / ".." / "escape" / "secret.png")
    resp = client.post("/generate", json={"prompt": "x", "anchor_path": traversal_path})
    assert resp.status_code == 400


def test_anchor_path_inside_allowlist_is_accepted(api, monkeypatch, tmp_path):
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    monkeypatch.setattr(image_api.gc, "REFERENCE_CANDIDATES_DIR", str(tmp_path))

    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"fake png bytes")

    resp = client.post("/generate", json={"prompt": "x", "anchor_path": str(anchor)})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert client.get(f"/generate/{job_id}").json()["status"] == "done"


def test_anchor_path_non_image_extension_inside_allowlist_is_400(api, monkeypatch, tmp_path):
    client, image_api = api
    monkeypatch.setattr(image_api.gc, "REFERENCE_CANDIDATES_DIR", str(tmp_path))

    notes = tmp_path / "notes.txt"
    notes.write_text("not an image")

    resp = client.post("/generate", json={"prompt": "x", "anchor_path": str(notes)})
    assert resp.status_code == 400


def test_anchor_dirs_env_var_extends_allowlist(api, monkeypatch, tmp_path):
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)

    extra_dir = tmp_path / "extra_anchors"
    extra_dir.mkdir()
    anchor = extra_dir / "anchor.jpg"
    anchor.write_bytes(b"fake jpg bytes")
    monkeypatch.setenv("IMAGE_API_ANCHOR_DIRS", str(extra_dir))

    resp = client.post("/generate", json={"prompt": "x", "anchor_path": str(anchor)})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert client.get(f"/generate/{job_id}").json()["status"] == "done"


# --- bounded job store -------------------------------------------------------------------


def seed_record(image_api, job_id, status, created_at):
    """Put a record straight into the store, bypassing generation."""
    import job_contracts as jc

    record = jc.new_record(
        job_id=job_id,
        owner_id="local",
        mode="txt2img_hq",
        media_kind="image",
        catalog_id="c",
        backend="local",
        params={"prompt": "x"},
        now=created_at,
    )
    record["status"] = status
    if status == jc.COMPLETED:
        record["artifacts"] = [
            {
                "bucket": "local-outputs",
                "key": f"generated/{job_id}/{'a' * 64}.png",
                "sha256": "a" * 64,
                "media_kind": "image",
                "content_type": "image/png",
                "byte_length": 1,
            }
        ]
    image_api._store.create("local", record)


def test_eviction_never_drops_pending_or_running_jobs(api, monkeypatch):
    """Over MAX_JOBS, only finished records may be evicted - a job still in flight must survive
    no matter how far over the cap the store is, or a polling client loses its answer.

    The rules now live in job_store.FileJobStore.purge (moved verbatim from the OrderedDict
    eviction this module used to do), so this asserts image_api still delegates with the right
    arguments; the policy itself is pinned record-for-record in test_job_store_contract.py.
    """
    import job_contracts as jc

    client, image_api = api
    monkeypatch.setattr(image_api, "MAX_JOBS", 2)
    now = 1_000_000.0
    seed_record(image_api, "done1", jc.FAILED, now)
    seed_record(image_api, "pending1", jc.QUEUED, now)
    seed_record(image_api, "done2", jc.FAILED, now)
    seed_record(image_api, "running1", jc.RUNNING, now)

    image_api._evict_jobs(now)

    survivors = {r["job_id"] for r in image_api._store.list_by_owner("local")}
    assert survivors == {"pending1", "running1"}


def test_ttl_evicts_old_finished_jobs_but_keeps_recent_and_pending(api, monkeypatch):
    import job_contracts as jc

    client, image_api = api
    monkeypatch.setattr(image_api, "JOB_TTL_SECONDS", 100)
    seed_record(image_api, "old_done", jc.FAILED, 0.0)
    seed_record(image_api, "recent_done", jc.FAILED, 200.0)
    seed_record(image_api, "old_pending", jc.QUEUED, 0.0)

    image_api._evict_jobs(250.0)  # old_done is 250s old (>100 TTL); recent_done is 50s old

    survivors = {r["job_id"] for r in image_api._store.list_by_owner("local")}
    assert survivors == {"recent_done", "old_pending"}


def test_generate_evicts_before_inserting_new_job(api, monkeypatch):
    """The eviction pass runs on every POST /generate, not just as a background sweep."""
    import job_contracts as jc

    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    monkeypatch.setattr(image_api, "MAX_JOBS", 1)
    seed_record(image_api, "stale_done", jc.FAILED, 0.0)

    resp = client.post("/generate", json={"prompt": "x"})
    assert resp.status_code == 200
    assert image_api._store.get("local", "stale_done") is None


# --- the record survives a restart ---------------------------------------------------------


def test_a_job_is_still_answerable_after_the_process_restarts(api, monkeypatch, tmp_path):
    """The actual win of moving off the in-memory dict. Before, a restart turned every
    outstanding job id into a 404 that a client could not tell apart from "never existed"."""
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    job_id = client.post("/generate", json={"prompt": "x"}).json()["job_id"]

    sys.modules.pop("image_api", None)
    import image_api as reimported

    assert reimported is not image_api, "a genuinely fresh module object"
    assert reimported._store.get("local", job_id)["status"] == "completed"


# --- the owner boundary --------------------------------------------------------------------


def test_another_owner_cannot_read_the_job(api, monkeypatch):
    """owner_id comes from a header, never from the body, so a caller cannot reach another
    owner's job by editing a payload."""
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    job_id = client.post("/generate", json={"prompt": "x"}).json()["job_id"]
    assert client.get(f"/generate/{job_id}", headers={"X-Owner-Id": "someone-else"}).status_code == 404
    assert client.get(f"/generate/{job_id}").status_code == 200


# --- the v2 surface ------------------------------------------------------------------------


def test_v2_returns_the_whole_record_including_the_artifact(api, monkeypatch):
    client, image_api = api
    writes_a_real_png(image_api, monkeypatch)
    job_id = client.post("/generate", json={"prompt": "x"}).json()["job_id"]
    record = client.get(f"/v2/jobs/{job_id}").json()
    assert record["status"] == "completed"
    assert record["output_check"]["state"] == "passed"
    assert record["artifacts"][0]["key"].startswith(f"generated/{job_id}/")
    assert record["params"]["seed"] >= 0, "the seed is recorded, so a result is reproducible"


def test_v2_cancel_of_a_queued_job_concludes_it(api, monkeypatch):
    client, image_api = api
    import job_contracts as jc

    seed_record(image_api, "queued1", jc.QUEUED, 1.0)
    body = client.post("/v2/jobs/queued1/cancel").json()
    assert body["status"] == "cancelled"
    assert body["cancel_requested_at"] is not None


def test_v2_cancel_of_an_unknown_job_is_404(api):
    client, _ = api
    assert client.post("/v2/jobs/nope/cancel").status_code == 404


def test_v2_catalog_includes_disabled_rows_with_their_reasons(api):
    """A caller that only ever sees what works cannot tell "this build has no ControlNet" from
    "not with this checkpoint, and here is why"."""
    client, _ = api
    rows = client.get("/v2/catalog", params={"checkpoint": "z_image_turbo"}).json()
    disabled = [r for r in rows if not r["enabled"]]
    assert disabled, "Z-Image disables most of the feature rows"
    assert all(r["reason"] for r in disabled)


def test_v2_catalog_rejects_an_unknown_checkpoint(api):
    client, _ = api
    assert client.get("/v2/catalog", params={"checkpoint": "nope"}).status_code == 400


# --- optional auth -------------------------------------------------------------------------


def test_no_token_configured_means_no_auth_required(api, monkeypatch):
    client, _ = api
    monkeypatch.delenv("IMAGE_API_TOKEN", raising=False)
    assert client.get("/characters").status_code == 200


def test_token_configured_rejects_missing_or_wrong_key(api, monkeypatch):
    client, _ = api
    monkeypatch.setenv("IMAGE_API_TOKEN", "s3cret")
    assert client.get("/characters").status_code == 401
    assert client.get("/characters", headers={"X-API-Key": "wrong"}).status_code == 401


def test_token_configured_accepts_correct_key(api, monkeypatch):
    client, _ = api
    monkeypatch.setenv("IMAGE_API_TOKEN", "s3cret")
    resp = client.get("/characters", headers={"X-API-Key": "s3cret"})
    assert resp.status_code == 200
