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
    """
    client, image_api = api

    def boom(*args, **kwargs):
        raise SystemExit("unknown character 'nobody'")

    monkeypatch.setattr(image_api.gc, "gen_custom", boom)

    job_id = client.post("/generate", json={"prompt": "portrait photo", "trigger": "nobody"}).json()["job_id"]
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
