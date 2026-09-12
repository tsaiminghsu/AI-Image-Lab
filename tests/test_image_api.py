"""Boundary behaviour of the FastAPI wrapper.

image_api hands every request to generate_character.gen_custom on a single worker thread and
reports the outcome only through the job dict a client polls. That makes the exception boundary
in _run_job the one place a failure can be lost: anything it does not catch kills the worker
thread silently and leaves the job "pending" forever.
"""

import sys

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


@pytest.fixture
def api(monkeypatch):
    """A TestClient over a freshly imported image_api, with its executor made synchronous so a
    POST has already run the job by the time it returns."""
    sys.modules.pop("image_api", None)
    import image_api

    class _Inline:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

    monkeypatch.setattr(image_api, "_executor", _Inline())
    image_api._jobs.clear()
    return fastapi_testclient.TestClient(image_api.app), image_api


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
    client, image_api = api
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)

    job_id = client.post("/generate", json={"prompt": "x"}).json()["job_id"]
    body = client.get(f"/generate/{job_id}").json()
    assert body["status"] == "done"
    assert body["image_path"].endswith(f"{job_id}.png")
    assert body["error"] is None


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
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)
    prompt = "a" * (image_api.gc.MAX_PROMPT_CHARS + 1)
    assert client.post("/generate", json={"prompt": prompt}).status_code == 422


def test_prompt_exactly_at_cap_is_accepted(api, monkeypatch):
    client, image_api = api
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)
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
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)
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
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)
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
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)

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


def test_eviction_never_drops_pending_or_running_jobs(api, monkeypatch):
    """Over MAX_JOBS, only finished ("done"/"error") jobs may be evicted - a pending or
    running job must survive no matter how far over the cap the store is."""
    client, image_api = api
    monkeypatch.setattr(image_api, "MAX_JOBS", 2)
    now = 1_000_000.0
    image_api._jobs["done1"] = {"status": "done", "image_path": "x", "error": None, "created_at": now}
    image_api._jobs["pending1"] = {"status": "pending", "image_path": None, "error": None, "created_at": now}
    image_api._jobs["done2"] = {"status": "done", "image_path": "x", "error": None, "created_at": now}
    image_api._jobs["running1"] = {"status": "pending", "image_path": None, "error": None, "created_at": now}

    with image_api._jobs_lock:
        image_api._evict_jobs_locked(now)

    assert set(image_api._jobs) == {"pending1", "running1"}


def test_ttl_evicts_old_finished_jobs_but_keeps_recent_and_pending(api, monkeypatch):
    client, image_api = api
    monkeypatch.setattr(image_api, "JOB_TTL_SECONDS", 100)
    image_api._jobs["old_done"] = {"status": "done", "image_path": "x", "error": None, "created_at": 0.0}
    image_api._jobs["recent_done"] = {"status": "done", "image_path": "x", "error": None, "created_at": 200.0}
    image_api._jobs["old_pending"] = {"status": "pending", "image_path": None, "error": None, "created_at": 0.0}

    with image_api._jobs_lock:
        image_api._evict_jobs_locked(250.0)  # old_done is 250s old (>100 TTL); recent_done is 50s old

    assert set(image_api._jobs) == {"recent_done", "old_pending"}


def test_generate_evicts_before_inserting_new_job(api, monkeypatch):
    """The eviction pass runs on every POST /generate, not just as a background sweep."""
    client, image_api = api
    monkeypatch.setattr(image_api.gc, "gen_custom", lambda *a, **k: None)
    monkeypatch.setattr(image_api, "MAX_JOBS", 1)
    image_api._jobs["stale_done"] = {
        "status": "done",
        "image_path": "x",
        "error": None,
        "created_at": 0.0,
    }

    resp = client.post("/generate", json={"prompt": "x"})
    assert resp.status_code == 200
    assert "stale_done" not in image_api._jobs


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
