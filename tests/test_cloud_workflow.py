"""The "runpod" generation backend: comfyui_client routing + training/cloud_workflow.py.

Against a scripted fake HTTP session - no network, no account. What is pinned, roughly in order of
what would hurt most if it broke:
- a cloud generation never touches the local ComfyUI (no local server start, no local HTTP);
- the graph that leaves this machine carries AGE_SAFETY_NEGATIVE and a cfg at or above the floor,
  and an unsafe or incomplete graph is refused before anything is sent (and billed);
- an abandoned job is cancelled, and a worker failure keeps its original text (the AnimateDiff OOM
  retry matches on it);
- the result lands where the local path would have put it, with the right extension.
"""

import base64
import json

import pytest
import requests

import cloud_video as cv
import cloud_workflow as cw
import comfyui_client as client
import generate_character as gc
from test_cloud_video import FakeSession, Resp

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
ENV = {"RUNPOD_API_KEY": "rp-key"}
NEG = f"lowres, {gc.AGE_SAFETY_NEGATIVE}"


@pytest.fixture(autouse=True)
def clean_backend_state(monkeypatch):
    monkeypatch.delenv("GENERATION_BACKEND", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    client.set_backend_override(None)
    cw.clear_staged()
    yield
    client.set_backend_override(None)
    cw.clear_staged()


@pytest.fixture
def no_local_comfy(monkeypatch):
    """Any attempt to talk to the local ComfyUI fails the test."""

    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"local ComfyUI HTTP used in cloud mode: requests.{name}")

    monkeypatch.setattr(client, "requests", Forbidden())
    monkeypatch.setattr(client, "start_server", lambda *a, **k: pytest.fail("local ComfyUI started in cloud mode"))


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    """Route cloud_workflow through a FakeSession with a configured endpoint. Returns a function that
    installs the routes and gives back the session."""
    settings = json.loads(json.dumps(cv.DEFAULT_SETTINGS))
    settings["runpod"]["endpoint_id"] = "ep1"
    monkeypatch.setattr(cv, "load_cloud_settings", lambda path=None: settings)
    monkeypatch.setattr(cv, "POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setenv("COMFYUI_RAW_OUTPUT_DIR", str(tmp_path / "raw"))
    real_make_backend = cv.make_backend

    def install(routes):
        session = FakeSession(routes)
        monkeypatch.setattr(
            cv,
            "make_backend",
            lambda provider, **kw: real_make_backend(
                provider, session=session, settings=settings, env=ENV, sleep=lambda s: None
            ),
        )
        return session

    return install


def completed(output):
    return Resp(
        200, {"id": "job-abc", "status": "COMPLETED", "output": output, "delayTime": 1200, "executionTime": 30000}
    )


@pytest.fixture
def png_file(tmp_path):
    path = tmp_path / "anchor mei!.png"
    path.write_bytes(PNG_BYTES)
    return str(path)


# --- backend resolution -----------------------------------------------------------------------


def test_backend_defaults_to_local():
    assert client.backend() == "local"


def test_backend_precedence(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "runpod")
    assert client.backend() == "runpod"
    client.set_backend_override("local")
    assert client.backend() == "local"
    with client.backend_scope("runpod"):
        assert client.backend() == "runpod"
    assert client.backend() == "local"


def test_invalid_backend_names_are_refused(monkeypatch):
    with pytest.raises(ValueError):
        client.set_backend_override("aws")
    with pytest.raises(ValueError):
        with client.backend_scope("aws"):
            pass
    monkeypatch.setenv("GENERATION_BACKEND", "aws")
    with pytest.raises(ValueError):
        client.backend()


def test_runpod_backend_is_refused_inside_a_runpod_worker(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    with client.backend_scope("runpod"):
        with pytest.raises(RuntimeError, match="inside a RunPod worker"):
            client.backend()


def test_cli_backend_flag_sets_the_process_override(capsys):
    gc.main(["--backend", "runpod", "list-characters"])
    assert client.backend() == "runpod"


# --- staging ----------------------------------------------------------------------------------


def test_upload_is_staged_not_sent_in_cloud_mode(no_local_comfy, png_file, tmp_path):
    with client.backend_scope("runpod"):
        name = client.upload_reference_image(png_file)
    assert name.endswith("_anchor_mei.png")
    assert name == cw.stage_upload(png_file)  # same bytes, same name
    other = tmp_path / "sub"
    other.mkdir()
    (other / "anchor mei!.png").write_bytes(PNG_BYTES + b"extra")
    assert cw.stage_upload(str(other / "anchor mei!.png")) != name  # same basename, different content


def test_non_image_uploads_are_refused(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(b"not an image")
    with pytest.raises(gc.UsageError, match="PNG 或 JPEG"):
        cw.stage_upload(str(path))


# --- routing and the full round trip ----------------------------------------------------------


def test_submit_and_wait_routes_to_the_cloud_before_touching_local_comfy(monkeypatch, no_local_comfy):
    seen = []
    monkeypatch.setattr(
        cw, "submit_and_wait", lambda wf, node, timeout, progress=None: seen.append((node, timeout)) or "x"
    )
    with client.backend_scope("runpod"):
        assert client._submit_and_wait({"1": {}}, "7", 1234) == "x"
    assert seen == [("7", 1234)]


def test_hq_generation_round_trip(no_local_comfy, cloud, png_file, tmp_path):
    session = cloud(
        {
            ("POST", "/ep1/run"): [Resp(200, {"id": "job-abc"})],
            ("GET", "/ep1/status/job-abc"): [
                Resp(200, {"id": "job-abc", "status": "IN_QUEUE"}),
                Resp(
                    200,
                    {"id": "job-abc", "status": "IN_PROGRESS", "output": {"stage": "生成", "current": 5, "total": 24}},
                ),
                completed(
                    {"outputKey": "generated/job-abc.png", "outputUrl": "https://signed.example/generated/job-abc.png"}
                ),
            ],
            ("GET", "https://signed.example/"): [Resp(200, content=PNG_BYTES)],
        }
    )
    progress = []
    with client.backend_scope("runpod"), client.progress_reporter(lambda *a: progress.append(a)):
        anchor = client.upload_reference_image(png_file)
        path = client.submit_generation_hq(
            prompt="a portrait",
            negative_prompt=NEG,
            seed=1,
            filename_prefix="stem",
            ip_adapter_image_filename=anchor,
            cfg=1.0,
        )

    assert path.endswith(".png") and str(tmp_path / "raw") in path
    with open(path, "rb") as f:
        assert f.read() == PNG_BYTES

    body = next(kw["json"] for m, u, kw in session.calls if m == "POST" and u.endswith("/run"))
    inp = body["input"]
    assert inp["jobType"] == "workflow" and inp["outputNodeId"] == "9"
    assert list(inp["inputImages"]) == [anchor]
    assert base64.b64decode(inp["inputImages"][anchor]) == PNG_BYTES
    texts = [n["inputs"]["text"] for n in inp["workflow"].values() if n["class_type"] == "CLIPTextEncode"]
    assert any(gc.AGE_SAFETY_NEGATIVE in t for t in texts)
    cfgs = [n["inputs"]["cfg"] for n in inp["workflow"].values() if "cfg" in n["inputs"]]
    assert cfgs and all(c >= client.SAFETY_MIN_CFG for c in cfgs)
    assert body["policy"]["executionTimeout"] >= (client.POLL_TIMEOUT_SECONDS + 300) * 1000

    assert all(total is None for _stage, _elapsed, total in progress)
    assert any("生成 5/24" in stage for stage, _e, _t in progress)


def test_unsafe_graph_is_refused_before_anything_is_sent(cloud, captured_submit):
    session = cloud({})
    client.submit_generation_hq(prompt="p", negative_prompt="lowres only", seed=1, filename_prefix="stem")
    wf = captured_submit[-1]["wf"]
    with pytest.raises(gc.UsageError, match="不能送到雲端"):
        cw.submit_and_wait(wf, "9", 900, session=session)
    assert session.calls == []


def test_unstaged_image_is_refused_before_anything_is_sent(cloud, captured_submit):
    session = cloud({})
    client.submit_generation_hq(
        prompt="p", negative_prompt=NEG, seed=1, filename_prefix="stem", ip_adapter_image_filename="never_staged.png"
    )
    with pytest.raises(gc.UsageError, match="沒有先暫存"):
        cw.submit_and_wait(captured_submit[-1]["wf"], "9", 900)
    assert session.calls == []


def test_oversized_payload_is_refused_before_anything_is_sent(monkeypatch, cloud, captured_submit, png_file):
    session = cloud({})
    monkeypatch.setattr(cv, "RUNPOD_MAX_PAYLOAD_BYTES", 100)
    name = cw.stage_upload(png_file)
    client.submit_generation_hq(
        prompt="p", negative_prompt=NEG, seed=1, filename_prefix="stem", ip_adapter_image_filename=name
    )
    with pytest.raises(gc.UsageError, match="太大"):
        cw.submit_and_wait(captured_submit[-1]["wf"], "9", 900)
    assert session.calls == []


def test_timeout_cancels_the_remote_job(monkeypatch, cloud, captured_submit):
    session = cloud(
        {
            ("POST", "/ep1/run"): [Resp(200, {"id": "job-abc"})],
            ("GET", "/ep1/status/job-abc"): [Resp(200, {"id": "job-abc", "status": "IN_PROGRESS"})],
            ("POST", "/ep1/cancel/job-abc"): [Resp(200, {})],
        }
    )
    monkeypatch.setattr(cw, "COLD_START_ALLOWANCE_SECONDS", 0)
    ticks = iter(range(0, 10_000, 30))
    client.submit_generation_hq(prompt="p", negative_prompt=NEG, seed=1, filename_prefix="stem")
    with pytest.raises(cv.CloudJobFailed, match="已取消"):
        cw.submit_and_wait(captured_submit[-1]["wf"], "9", 60, clock=lambda: next(ticks))
    assert session.count("POST", "/cancel/job-abc") == 1


def test_worker_error_text_is_kept_for_callers_that_match_on_it(cloud, captured_submit):
    cloud(
        {
            ("POST", "/ep1/run"): [Resp(200, {"id": "job-abc"})],
            ("GET", "/ep1/status/job-abc"): [completed({"error": "RuntimeError: CUDA out of memory"})],
        }
    )
    client.submit_generation_hq(prompt="p", negative_prompt=NEG, seed=1, filename_prefix="stem")
    with pytest.raises(cv.CloudJobFailed) as err:
        cw.submit_and_wait(captured_submit[-1]["wf"], "9", 900)
    assert "out of memory" in str(err.value)
    assert isinstance(err.value, RuntimeError)


def test_job_creation_is_not_retried_when_the_response_is_lost(cloud, captured_submit):
    session = cloud({("POST", "/ep1/run"): [requests.exceptions.ReadTimeout("lost")]})
    client.submit_generation_hq(prompt="p", negative_prompt=NEG, seed=1, filename_prefix="stem")
    with pytest.raises(cv.CloudJobFailed, match="重複計費"):
        cw.submit_and_wait(captured_submit[-1]["wf"], "9", 900)
    assert session.count("POST", "/run") == 1


def test_missing_configuration_is_reported_not_crashed(monkeypatch, captured_submit):
    settings = json.loads(json.dumps(cv.DEFAULT_SETTINGS))
    monkeypatch.setattr(cv, "load_cloud_settings", lambda path=None: settings)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    client.submit_generation_hq(prompt="p", negative_prompt=NEG, seed=1, filename_prefix="stem")
    with pytest.raises(gc.UsageError, match="RUNPOD_API_KEY"):
        cw.submit_and_wait(captured_submit[-1]["wf"], "9", 900)


@pytest.mark.parametrize(
    ("key", "ext"),
    [
        ("generated/j.png", "png"),
        ("generated/j.webm", "webm"),
        ("generated/j.mp4", "mp4"),
        ("generated/j.exe", "mp4"),
        (None, "mp4"),
    ],
)
def test_runpod_result_extension_comes_from_the_output_key(tmp_path, key, ext):
    session = FakeSession({("GET", "https://signed/"): [Resp(200, content=b"data")]})
    backend = cv.RunPodBackend(session, "k", "ep1")
    path, _q, _e = backend.result(
        {"output": {"outputKey": key, "outputUrl": "https://signed/x"}}, str(tmp_path), "stem"
    )
    assert path.endswith(f"stem.{ext}")
