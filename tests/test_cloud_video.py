"""training/cloud_video.py against a scripted fake HTTP session - no network, no accounts.

What matters most here, in order: an abandoned cloud job is always cancelled (it keeps billing
otherwise); a hosted model that can't carry the safety negatives is refused; the payloads we build
always carry AGE_SAFETY_NEGATIVE and a cfg at or above the floor; secrets never reach the settings
file.
"""

import base64
import json
import os
import threading

import pytest
import requests

import cloud_video as cv
import comfyui_client as client
import generate_character as gc

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
ENV = {"RUNPOD_API_KEY": "rp-key", "REPLICATE_API_TOKEN": "r8-token"}


class Resp:
    def __init__(self, status=200, body=None, content=b""):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""
        self._content = content

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_content(self, chunk_size=1):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Routes by (method, url substring). Each route is a list of responses consumed in order (the
    last one repeats) or an exception instance to raise."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _respond(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        for (m, fragment), responses in self.routes.items():
            if m == method and fragment in url:
                item = responses.pop(0) if len(responses) > 1 else responses[0]
                if isinstance(item, BaseException):
                    raise item
                return item
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kwargs):
        return self._respond("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._respond("POST", url, **kwargs)

    def count(self, method, fragment):
        return sum(1 for m, u, _ in self.calls if m == method and fragment in u)


@pytest.fixture
def settings():
    s = json.loads(json.dumps(cv.DEFAULT_SETTINGS))
    s["runpod"]["endpoint_id"] = "ep1"
    s["replicate"]["model"] = "someone/wan-video"
    return s


@pytest.fixture
def frame(tmp_path):
    p = tmp_path / "frame.png"
    p.write_bytes(PNG_BYTES)
    return str(p)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def run(session, settings, tmp_path, provider="runpod", job_type="video_wan_i2v", frame=None, **kw):
    clock = Clock()
    defaults = dict(
        prompt="waves hello",
        tier="safe",
        image_path=frame,
        seed=5,
        out_dir=str(tmp_path / "out"),
        session=session,
        settings=settings,
        env=ENV,
        sleep=clock.sleep,
        clock=clock,
    )
    defaults.update(kw)
    return cv.run_cloud_video(provider, job_type, **defaults)


# --- RunPod -----------------------------------------------------------------------------------


def runpod_routes(statuses, output=None):
    done = {
        "status": "COMPLETED",
        "delayTime": 12000,
        "executionTime": 95000,
        "output": output or {"outputUrl": "https://signed/generated/j.mp4", "outputKey": "generated/j.mp4"},
    }
    return {
        ("POST", "/run"): [Resp(200, {"id": "job-abcdef123", "status": "IN_QUEUE"})],
        ("GET", "/status/"): [Resp(200, {"status": s}) for s in statuses] + [Resp(200, done)],
        ("GET", "https://signed/"): [Resp(200, content=b"MP4DATA")],
        ("POST", "/cancel/"): [Resp(200, {"status": "CANCELLED"})],
    }


def test_runpod_happy_path_downloads_and_reports_status(tmp_path, settings, frame):
    session = FakeSession(runpod_routes(["IN_QUEUE", "IN_PROGRESS"]))
    seen = []
    result = run(session, settings, tmp_path, frame=frame, on_status=lambda *a: seen.append(a))
    assert open(result.path, "rb").read() == b"MP4DATA"
    assert os.path.basename(result.path) == "cloud_runpod_wan_seed5_job-abcd.mp4"
    assert (result.queue_s, result.execution_s) == (12.0, 95.0)
    assert any("排隊" in label for label, _, _ in seen)
    assert session.count("POST", "/cancel/") == 0

    _, url, kwargs = next(c for c in session.calls if c[0] == "POST" and "/run" in c[1])
    assert url == "https://api.runpod.ai/v2/ep1/run"
    assert kwargs["headers"]["Authorization"] == "Bearer rp-key"
    body = kwargs["json"]
    assert body["policy"]["executionTimeout"] == settings["runpod"]["execution_timeout_s"] * 1000
    assert body["input"]["jobType"] == "video_wan_i2v"
    assert base64.b64decode(body["input"]["referenceImageBase64"]) == PNG_BYTES


def test_runpod_sends_the_raw_prompt_so_the_worker_composes_negatives(tmp_path, settings, frame):
    session = FakeSession(runpod_routes([]))
    run(session, settings, tmp_path, frame=frame, extra_negative="hats")
    inp = next(c for c in session.calls if "/run" in c[1])[2]["json"]["input"]
    assert inp["prompt"] == "waves hello"
    assert inp["negativePrompt"] == "hats"
    assert inp["tier"] == "safe"


def test_runpod_failure_raises_cloud_job_failed(tmp_path, settings, frame):
    routes = runpod_routes([])
    routes[("GET", "/status/")] = [Resp(200, {"status": "FAILED", "error": "CUDA error"})]
    with pytest.raises(cv.CloudJobFailed, match="failed"):
        run(FakeSession(routes), settings, tmp_path, frame=frame)


def test_worker_error_output_counts_as_failure(tmp_path, settings, frame):
    session = FakeSession(runpod_routes([], output={"error": "missing prompt"}))
    with pytest.raises(cv.CloudJobFailed, match="missing prompt"):
        run(session, settings, tmp_path, frame=frame)


def test_timeout_cancels_the_remote_job_exactly_once(tmp_path, settings, frame):
    routes = runpod_routes([])
    routes[("GET", "/status/")] = [Resp(200, {"status": "IN_PROGRESS"})]
    session = FakeSession(routes)
    with pytest.raises(cv.CloudJobFailed, match="已取消"):
        run(session, settings, tmp_path, frame=frame, max_wait_s=10)
    assert session.count("POST", "/cancel/job-abcdef123") == 1


def test_cancel_event_cancels_the_remote_job(tmp_path, settings, frame):
    routes = runpod_routes([])
    routes[("GET", "/status/")] = [Resp(200, {"status": "IN_PROGRESS"})]
    session = FakeSession(routes)
    event = threading.Event()
    event.set()
    with pytest.raises(cv.CloudJobCancelled):
        run(session, settings, tmp_path, frame=frame, cancel_event=event)
    assert session.count("POST", "/cancel/") == 1


def test_keyboard_interrupt_cancels_then_reraises(tmp_path, settings, frame):
    routes = runpod_routes([])
    routes[("GET", "/status/")] = [KeyboardInterrupt()]
    session = FakeSession(routes)
    with pytest.raises(KeyboardInterrupt):
        run(session, settings, tmp_path, frame=frame)
    assert session.count("POST", "/cancel/") == 1


def test_cancel_failure_does_not_mask_the_original_error(tmp_path, settings, frame):
    routes = runpod_routes([])
    routes[("GET", "/status/")] = [Resp(200, {"status": "IN_PROGRESS"})]
    routes[("POST", "/cancel/")] = [requests.exceptions.ConnectionError("down")]
    with pytest.raises(cv.CloudJobFailed, match="已取消"):
        run(FakeSession(routes), settings, tmp_path, frame=frame, max_wait_s=1)


@pytest.mark.parametrize("code,message", [(401, "金鑰"), (403, "金鑰"), (402, "餘額不足"), (429, "太頻繁")])
def test_http_errors_on_submit_have_distinct_messages(tmp_path, settings, frame, code, message):
    routes = runpod_routes([])
    routes[("POST", "/run")] = [Resp(code, {"error": "x"})]
    with pytest.raises(gc.UsageError, match=message):
        run(FakeSession(routes), settings, tmp_path, frame=frame)


def test_submit_is_never_retried_but_polls_are(tmp_path, settings, frame):
    routes = runpod_routes([])
    routes[("GET", "/status/")] = [
        requests.exceptions.ConnectionError("blip"),
        Resp(503, {}),
        Resp(200, {"status": "COMPLETED", "output": {"outputUrl": "https://signed/x.mp4"}}),
    ]
    session = FakeSession(routes)
    run(session, settings, tmp_path, frame=frame)
    assert session.count("POST", "/run") == 1
    assert session.count("GET", "/status/") == 3

    routes = runpod_routes([])
    routes[("POST", "/run")] = [requests.exceptions.ConnectionError("lost")]
    session = FakeSession(routes)
    with pytest.raises(cv.CloudJobFailed, match="重複計費"):
        run(session, settings, tmp_path, frame=frame)
    assert session.count("POST", "/run") == 1


def test_oversized_payload_is_refused_before_any_http(tmp_path, settings, monkeypatch):
    big = tmp_path / "big.png"
    big.write_bytes(PNG_BYTES + b"\x00" * 8_000_000)
    session = FakeSession({})
    with pytest.raises(gc.UsageError, match="太大"):
        run(session, settings, tmp_path, frame=str(big))
    assert session.calls == []


def test_non_image_first_frame_is_refused(tmp_path, settings):
    bad = tmp_path / "frame.gif"
    bad.write_bytes(b"GIF89a" + b"\x00" * 10)
    with pytest.raises(gc.UsageError, match="PNG 或 JPEG"):
        run(FakeSession({}), settings, tmp_path, frame=str(bad))


def test_missing_configuration_is_reported_before_any_http(tmp_path, frame):
    s = json.loads(json.dumps(cv.DEFAULT_SETTINGS))
    session = FakeSession({})
    with pytest.raises(gc.UsageError, match="endpoint ID"):
        run(session, s, tmp_path, frame=frame)
    with pytest.raises(gc.UsageError, match="RUNPOD_API_KEY"):
        run(session, s, tmp_path, frame=frame, env={})
    assert session.calls == []


# --- Replicate safety gate --------------------------------------------------------------------

GOOD_SCHEMA = {
    "prompt": {"type": "string"},
    "negative_prompt": {"type": "string"},
    "image": {"type": "string", "format": "uri"},
    "guidance_scale": {"type": "number", "minimum": 0, "maximum": 10},
    "num_frames": {"type": "integer"},
    "seed": {"type": "integer"},
}
FIELDS = dict(cv.DEFAULT_REPLICATE_FIELDS)


@pytest.mark.parametrize(
    "drop,message",
    [("negative_prompt", "負面詞"), ("guidance_scale", "cfg"), ("image", "圖片")],
)
def test_gate_refuses_models_missing_a_required_field(drop, message):
    schema = {k: v for k, v in GOOD_SCHEMA.items() if k != drop}
    with pytest.raises(gc.UsageError, match=message):
        cv.replicate_safety_gate(schema, FIELDS)


def test_gate_refuses_a_cfg_that_cannot_reach_the_floor():
    schema = dict(GOOD_SCHEMA, guidance_scale={"type": "number", "maximum": 1.0})
    with pytest.raises(gc.UsageError, match="安全下限"):
        cv.replicate_safety_gate(schema, FIELDS)


@pytest.mark.parametrize("tier", ["safe", "suggestive"])
def test_replicate_input_carries_safety_negatives_and_floors_cfg(frame, tier):
    inp = cv.build_replicate_input(
        GOOD_SCHEMA,
        FIELDS,
        prompt="waves",
        extra_negative="",
        tier=tier,
        trigger="mei",
        image_path=frame,
        seed=3,
        cfg=1.0,
        frames=81,
        fps=24,
    )
    assert gc.AGE_SAFETY_NEGATIVE in inp["negative_prompt"]
    assert inp["guidance_scale"] >= client.SAFETY_MIN_CFG
    assert inp["image"].startswith("data:image/png;base64,")
    assert inp["num_frames"] == 81 and inp["seed"] == 3
    assert "frames_per_second" not in inp  # not in this model's schema, so not sent


def test_large_first_frame_is_shrunk_under_the_data_url_limit(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(PNG_BYTES + b"\x00" * 300_000)
    calls = []

    def fake_shrink(raw, limit):
        calls.append(limit)
        return b"\xff\xd8\xff" + b"\x00" * 100

    inp = cv.build_replicate_input(
        GOOD_SCHEMA,
        FIELDS,
        prompt="p",
        extra_negative="",
        tier="safe",
        trigger=None,
        image_path=str(big),
        seed=1,
        cfg=5,
        shrink=fake_shrink,
    )
    assert calls == [cv.REPLICATE_MAX_DATA_URI_BYTES]
    assert inp["image"].startswith("data:image/jpeg;base64,")


def test_replicate_end_to_end_pins_the_checked_version(tmp_path, settings, frame):
    model = {
        "latest_version": {
            "id": "ver123",
            "openapi_schema": {"components": {"schemas": {"Input": {"properties": GOOD_SCHEMA}}}},
        }
    }
    routes = {
        ("GET", "/models/someone/wan-video"): [Resp(200, model)],
        ("POST", "/v1/predictions"): [Resp(201, {"id": "pred1", "status": "starting"})],
        ("GET", "/predictions/pred1"): [
            Resp(200, {"status": "processing"}),
            Resp(
                200,
                {
                    "status": "succeeded",
                    "output": ["https://replicate.delivery/a.png", "https://replicate.delivery/v.mp4"],
                    "metrics": {"predict_time": 88.5},
                },
            ),
        ],
        ("GET", "https://replicate.delivery/v.mp4"): [Resp(200, content=b"VID")],
    }
    session = FakeSession(routes)
    result = run(session, settings, tmp_path, provider="replicate", frame=frame, params={"cfg": 1.0})
    body = next(c for c in session.calls if c[0] == "POST")[2]["json"]
    assert body["version"] == "ver123"
    assert gc.AGE_SAFETY_NEGATIVE in body["input"]["negative_prompt"]
    assert body["input"]["guidance_scale"] >= client.SAFETY_MIN_CFG
    assert open(result.path, "rb").read() == b"VID"
    assert result.execution_s == 88.5


def test_replicate_refused_model_never_creates_a_prediction(tmp_path, settings, frame):
    schema = {k: v for k, v in GOOD_SCHEMA.items() if k != "negative_prompt"}
    model = {
        "latest_version": {"id": "v", "openapi_schema": {"components": {"schemas": {"Input": {"properties": schema}}}}}
    }
    session = FakeSession({("GET", "/models/"): [Resp(200, model)]})
    with pytest.raises(gc.UsageError, match="拒絕使用"):
        run(session, settings, tmp_path, provider="replicate", frame=frame)
    assert session.count("POST", "/predictions") == 0


def test_replicate_cannot_run_the_animatediff_pipeline(tmp_path, settings, frame):
    with pytest.raises(gc.UsageError, match="RunPod"):
        run(FakeSession({}), settings, tmp_path, provider="replicate", job_type="video_animatediff", frame=frame)


@pytest.mark.parametrize(
    "output,expected",
    [
        ("https://x/a.mp4", "https://x/a.mp4"),
        (["https://x/a.png", "https://x/b.mp4?sig=1"], "https://x/b.mp4?sig=1"),
        (["https://x/a.webm"], "https://x/a.webm"),
        ([], None),
        (None, None),
    ],
)
def test_pick_output_url(output, expected):
    assert cv.pick_output_url(output) == expected


# --- settings ---------------------------------------------------------------------------------


def test_missing_settings_file_gives_defaults(tmp_path):
    s = cv.load_cloud_settings(str(tmp_path / "nope.json"))
    assert s["runpod"]["endpoint_id"] == "" and s["version"] == cv.SETTINGS_VERSION


def test_save_merges_atomically_and_refuses_secrets(tmp_path):
    path = str(tmp_path / "settings" / "cloud.json")
    cv.save_cloud_settings({"runpod": {"endpoint_id": "ep9"}}, path)
    cv.save_cloud_settings({"replicate": {"model": "a/b"}}, path)
    s = cv.load_cloud_settings(path)
    assert s["runpod"]["endpoint_id"] == "ep9" and s["replicate"]["model"] == "a/b"
    assert not os.path.exists(path + ".tmp")
    with pytest.raises(gc.UsageError, match="環境變數"):
        cv.save_cloud_settings({"runpod": {"api_key": "rp-secret"}}, path)
    assert "rp-secret" not in open(path, encoding="utf-8").read()


def test_config_status_lists_what_is_missing():
    s = json.loads(json.dumps(cv.DEFAULT_SETTINGS))
    problems = cv.config_status(settings=s, env={})
    assert any("RUNPOD_API_KEY" in p for p in problems)
    assert any("REPLICATE_API_TOKEN" in p for p in problems)
    s["runpod"]["endpoint_id"] = "ep"
    assert cv.config_status("runpod", settings=s, env=ENV) == []


def test_cli_surface():
    parser = cv.build_parser()
    args = parser.parse_args(["run", "--provider", "runpod", "--job", "wan-i2v", "--prompt", "p"])
    assert args.tier == "safe" and args.image is None
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--provider", "modal", "--job", "wan-i2v", "--prompt", "p"])


# --- batches ----------------------------------------------------------------------------------


def batch_routes(outcomes):
    """outcomes: job id -> list of RunPod statuses; the last is COMPLETED/FAILED."""
    routes = {("POST", "/run"): [Resp(200, {"id": job_id, "status": "IN_QUEUE"}) for job_id in outcomes]}
    for job_id, statuses in outcomes.items():
        bodies = []
        for status in statuses:
            body = {"status": status}
            if status == "COMPLETED":
                body.update(
                    delayTime=1000,
                    executionTime=30000,
                    output={"outputUrl": f"https://signed/{job_id}.mp4", "outputKey": f"generated/{job_id}.mp4"},
                )
            bodies.append(Resp(200, body))
        routes[("GET", f"/status/{job_id}")] = bodies
        routes[("GET", f"https://signed/{job_id}.mp4")] = [Resp(200, content=job_id.encode())]
    routes[("POST", "/cancel/")] = [Resp(200, {"status": "CANCELLED"})]
    return routes


def specs_for(frame, n):
    return [
        cv.CloudJobSpec(stem=f"shot{i}", prompt="turns around", image_path=frame, seed=100 + i, params={"frames": 49})
        for i in range(n)
    ]


def run_batch(session, settings, tmp_path, specs, **kw):
    clock = Clock()
    return cv.run_cloud_batch(
        "runpod",
        "video_wan_i2v",
        specs,
        out_dir=str(tmp_path / "out"),
        session=session,
        settings=settings,
        env=ENV,
        sleep=clock.sleep,
        clock=clock,
        **kw,
    )


def test_batch_submits_everything_before_polling(tmp_path, settings, frame):
    session = FakeSession(
        batch_routes({"ja": ["IN_QUEUE", "COMPLETED"], "jb": ["IN_QUEUE", "IN_PROGRESS", "COMPLETED"]})
    )
    results = run_batch(session, settings, tmp_path, specs_for(frame, 2))
    methods = [(m, "/run" in u) for m, u, _ in session.calls]
    first_poll = next(i for i, (m, _) in enumerate(methods) if m == "GET")
    assert [m for m, is_run in methods[:first_poll]] == ["POST", "POST"]
    assert [os.path.basename(r.path) for _s, r in results] == ["shot0.mp4", "shot1.mp4"]
    assert [r.seed for _s, r in results] == [100, 101]
    assert open(results[1][1].path, "rb").read() == b"jb"
    assert session.count("POST", "/cancel/") == 0


def test_batch_keeps_going_when_one_job_fails(tmp_path, settings, frame):
    session = FakeSession(batch_routes({"ja": ["FAILED"], "jb": ["COMPLETED"]}))
    results = run_batch(session, settings, tmp_path, specs_for(frame, 2))
    assert isinstance(results[0][1], cv.CloudJobFailed) and results[0][1].job_id == "ja"
    assert isinstance(results[1][1], cv.CloudResult)


def test_batch_refuses_a_bad_spec_before_any_http(tmp_path, settings, frame):
    specs = specs_for(frame, 2)
    specs[1].prompt = "  "
    session = FakeSession(batch_routes({"ja": ["COMPLETED"]}))
    with pytest.raises(gc.UsageError):
        run_batch(session, settings, tmp_path, specs)
    assert session.calls == []
    specs = specs_for(frame, 2)
    specs[1].stem = specs[0].stem
    with pytest.raises(gc.UsageError, match="stem"):
        run_batch(session, settings, tmp_path, specs)


def test_batch_cancel_event_cancels_every_pending_job(tmp_path, settings, frame):
    session = FakeSession(batch_routes({"ja": ["IN_PROGRESS"], "jb": ["IN_QUEUE"]}))
    event = threading.Event()
    calls = []

    def on_status(label, elapsed, limit):
        calls.append(label)
        if len(calls) == 2:
            event.set()

    with pytest.raises(cv.CloudJobCancelled):
        run_batch(session, settings, tmp_path, specs_for(frame, 2), on_status=on_status, cancel_event=event)
    assert session.count("POST", "/cancel/ja") == 1 and session.count("POST", "/cancel/jb") == 1


def test_batch_deadline_scales_with_batch_size_and_cancels_the_rest(tmp_path, settings, frame):
    session = FakeSession(batch_routes({"ja": ["COMPLETED"], "jb": ["IN_QUEUE"]}))
    results = run_batch(session, settings, tmp_path, specs_for(frame, 2), max_wait_s=20)
    assert isinstance(results[0][1], cv.CloudResult)
    assert isinstance(results[1][1], cv.CloudJobFailed) and "40" in str(results[1][1])
    assert session.count("POST", "/cancel/jb") == 1 and session.count("POST", "/cancel/ja") == 0


def test_batch_submit_failure_cancels_what_was_already_submitted(tmp_path, settings, frame):
    routes = batch_routes({"ja": ["IN_QUEUE"]})
    routes[("POST", "/run")] = [Resp(200, {"id": "ja"}), Resp(402, {"error": "insufficient balance"})]
    session = FakeSession(routes)
    with pytest.raises(gc.UsageError, match="餘額不足"):
        run_batch(session, settings, tmp_path, specs_for(frame, 2))
    assert session.count("POST", "/cancel/ja") == 1


def test_empty_batch_does_nothing(tmp_path, settings):
    session = FakeSession({})
    assert run_batch(session, settings, tmp_path, []) == []


def test_cli_preview_fills_only_what_was_not_given():
    parser = cv.build_parser()
    args = parser.parse_args(
        ["run", "--provider", "runpod", "--job", "wan-i2v", "--prompt", "p", "--preview", "--steps", "15"]
    )
    params = cv.cli_params(args, "video_wan_i2v")
    assert (params["width"], params["height"], params["frames"], params["steps"]) == (480, 832, 49, 15)
    args = parser.parse_args(
        ["run", "--provider", "runpod", "--job", "wan-i2v", "--prompt", "p", "--preview", "--landscape"]
    )
    assert (cv.cli_params(args, "video_wan_i2v")["width"]) == 832
    with pytest.raises(gc.UsageError):
        cv.cli_params(args, "video_animatediff")


@pytest.mark.parametrize("preset", [cv.WAN_PREVIEW_PORTRAIT, cv.WAN_PREVIEW_LANDSCAPE])
def test_preview_presets_pass_the_wan_bounds(preset):
    gc.check_wan_params(
        preset["width"],
        preset["height"],
        preset["frames"],
        client.WAN_FPS,
        preset["steps"],
        client.WAN_CFG,
        client.WAN_DEFAULT_MODEL,
    )
