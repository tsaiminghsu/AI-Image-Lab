"""The MiniMax H3 job state machine, driven by a scripted fake `colab` - no Docker, Colab or GPU.

The properties that cost money when they break: bad inputs never reach `colab new`, every path that
created a session stops it, a timed-out `colab exec` is not retried, and success needs both the remote
H3_OUTPUT marker and a passing ffprobe check (colab exec exits 0 even when the remote code raised).
"""

import json
import sys
import time

import pytest

from minimax_h3_fakes import (
    FakeProber,
    FakeTransport,
    emit,
    good_probe,
    h3,
    import_skill,
    make_config,
    make_image,
    marker,
    raising,
)

HAPPY_CALLS = ["version", "usage", "new", "usage", "upload", "upload", "exec", "download", "stop", "usage"]


def run(tmp_path, transport=None, prober=None, config=None, **spec_kw):
    spec_kw.setdefault("description", "She walks slowly toward the camera.")
    spec_kw.setdefault("image", make_image(tmp_path))
    spec_kw.setdefault("job_id", "job1")
    transport = transport or FakeTransport()
    record = h3.run_job(
        h3.JobSpec(**spec_kw),
        config or make_config(tmp_path),
        transport=transport,
        prober=prober or FakeProber(),
        stream=None,
    )
    return record, transport


def transitions(record):
    with open(record["log"], encoding="utf-8") as f:
        return [line.split("] ", 1)[1].strip() for line in f if " -> " in line and "] |" not in line]


def test_happy_path_walks_every_state_and_writes_the_records(tmp_path):
    record, transport = run(tmp_path)
    assert record["status"] == h3.COMPLETED, record
    assert transitions(record) == [
        "PENDING -> PREPARING",
        "PREPARING -> CONNECTING_COLAB",
        "CONNECTING_COLAB -> UPLOADING",
        "UPLOADING -> LOADING_MODEL",
        "LOADING_MODEL -> INFERENCE",
        "INFERENCE -> DOWNLOADING",
        "DOWNLOADING -> VALIDATING",
        "VALIDATING -> COMPLETED",
    ]
    assert transport.subcommands == HAPPY_CALLS
    output = tmp_path / "out" / "first_h3_job1.mp4"
    assert record["output"] == str(output) and output.stat().st_size > 0
    assert record["resolution"] == "1376x768" and record["frames"] == 192 and record["actual_seconds"] == 8.0
    assert record["gpu"] == "NVIDIA A100-SXM4-40GB" and record["vram_peak_mib"] == 38000
    assert record["cu_balance_before"] == 100.0 and record["cu_balance_after"] == 98.5
    assert record["cu_used_measured"] == 1.5 and record["cu_rate_per_hour"] == 11.77
    assert record["cu_estimated"] is not None
    assert record["remote_seconds"] == {"setup": 150.0, "download": 300.0, "inference": 420.0}
    assert record["session_status"] == "stopped"
    assert set(h3.LIFECYCLE[1:]) <= set(record["stage_seconds"])
    saved = json.loads((tmp_path / "out" / "first_h3_job1.job.json").read_text(encoding="utf-8"))
    assert saved["status"] == h3.COMPLETED and saved["prompt"].startswith(h3.I2VA_ALIGNMENT)
    ledger = (tmp_path / "out" / "h3_jobs.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(ledger[-1])
    assert set(h3.LEDGER_FIELDS) == set(entry)
    assert entry["status"] == h3.COMPLETED and entry["gpu"] == "NVIDIA A100-SXM4-40GB"
    assert not (tmp_path / "out" / ".work" / "job1").exists()


def test_the_exec_call_carries_the_job_file_and_timeout(tmp_path):
    record, transport = run(tmp_path, timeout=1800)
    exec_args = next(c for c in transport.calls if c[0] == "exec")
    assert exec_args[:5] == ["exec", "--session", "h3-job1", "--timeout", "1800"]
    assert "H3_JOB_FILE=/content/h3_job1_job.json" in exec_args
    new_args = next(c for c in transport.calls if c[0] == "new")
    assert new_args == ["new", "--session", "h3-job1", "--gpu", "A100", "--high-mem"]


def test_staged_job_file_holds_what_the_remote_graph_needs(tmp_path):
    transport = FakeTransport(exec=raising(h3.ColabTimeout("stop here")))
    record, _ = run(tmp_path, transport=transport)
    job = json.loads((tmp_path / "out" / ".work" / "job1" / "job.json").read_text(encoding="utf-8"))
    assert job["deadline_seconds"] == 3600 - 120
    assert job["prompt"] == record["prompt"] and job["frames"] == 192
    assert job["remote_image"] == "/content/h3_job1_first_frame.png"


# --- nothing reaches Colab when the inputs are wrong ------------------------------------------------------


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"prober": FakeProber(image_error="Invalid data found when processing input")}, "not a readable image"),
        ({"prober": FakeProber(image=(1920, 1080, "h264"))}, "not a still image"),
        ({"description": ""}, "prompt is required"),
        ({"duration": 3}, "duration"),
        ({"duration": 16}, "duration"),
        ({"duration": float("nan")}, "duration"),
        ({"resolution": "1024x1024"}, "stretched"),
        ({"description": "A teenager waves."}, "never sends"),
    ],
)
def test_invalid_inputs_fail_before_any_colab_call(tmp_path, kw, match):
    record, transport = run(tmp_path, **kw)
    assert record["status"] == h3.FAILED
    assert record["error_code"] in ("INVALID_INPUT", "PROMPT_REJECTED")
    assert match in record["error"]
    assert record["failed_stage"] == h3.PREPARING
    assert transport.calls == []


def test_missing_image_fails_before_any_colab_call(tmp_path):
    record, transport = run(tmp_path, image=tmp_path / "missing.png")
    assert record["error_code"] == "INVALID_INPUT" and "not found" in record["error"]
    assert transport.calls == []


def test_existing_output_is_not_overwritten_without_the_flag(tmp_path):
    out = tmp_path / "exists.mp4"
    out.write_bytes(b"x")
    record, transport = run(tmp_path, output=out)
    assert "already exists" in record["error"] and transport.calls == []


def test_missing_ffprobe_fails_before_any_colab_call(tmp_path):
    config = make_config(tmp_path, H3_FFPROBE=str(tmp_path / "no" / "ffprobe.exe"))
    transport = FakeTransport()
    record = h3.run_job(
        h3.JobSpec(image=make_image(tmp_path), description="x", job_id="j"), config, transport=transport, stream=None
    )
    assert record["error_code"] == "FFPROBE_MISSING" and transport.calls == []


# --- Colab-side failures -----------------------------------------------------------------------------


def test_missing_authentication_is_reported_and_no_session_is_created(tmp_path):
    output = ["To authorize colab-cli, visit this URL in any browser:", "EOFError: EOF when reading a line"]
    transport = FakeTransport(usage=raising(h3.ColabCommandError("colab usage", 1, output)))
    record, _ = run(tmp_path, transport=transport)
    assert record["status"] == h3.FAILED and record["error_code"] == "AUTH_REQUIRED"
    assert record["error"] == h3.AUTH_REQUIRED_MESSAGE
    assert "--auth=oauth2 usage" in record["hint"]
    assert "new" not in transport.subcommands and "stop" not in transport.subcommands


def test_colab_connection_failure(tmp_path):
    transport = FakeTransport(usage=raising(h3.ColabCommandError("colab usage", 1, ["ConnectionError: timed out"])))
    record, _ = run(tmp_path, transport=transport)
    assert record["error_code"] == "COLAB_CONNECTION_FAILED" and "new" not in transport.subcommands


def test_refused_gpu_is_reported_plainly(tmp_path):
    refused = ["[colab] Backend rejected accelerator 'A100'. You may not have quota or entitlement ..."]
    not_found = h3.ColabCommandError("colab stop", 1, ["[colab] Session 'h3-job1' not found."])
    transport = FakeTransport(new=raising(h3.ColabCommandError("colab new", 1, refused)), stop=raising(not_found))
    record, _ = run(tmp_path, transport=transport)
    assert record["error_code"] == "GPU_UNAVAILABLE" and record["error"] == h3.GPU_UNAVAILABLE_MESSAGE
    assert record["failed_stage"] == h3.CONNECTING_COLAB
    assert record["session_status"] == "not_created"


def test_no_capacity_at_assignment_is_gpu_unavailable(tmp_path):
    """Recorded from a real run: `colab new --gpu A100 --high-mem` answered 503 by the assign endpoint."""
    output = [
        "ColabRequestError: Failed to issue request POST",
        "https://colab.research.google.com/tun/m/assign?nbh=f0561344_e173_4dea_ab80_7aa20",
        "bad4cd2........&variant=GPU&accelerator=A100&shape=hm: Service Unavailable",
    ]
    not_found = h3.ColabCommandError("colab stop", 1, ["[colab] Session 'h3-job1' not found."])
    transport = FakeTransport(new=raising(h3.ColabCommandError("colab new", 1, output)), stop=raising(not_found))
    record, _ = run(tmp_path, transport=transport)
    assert record["error_code"] == "GPU_UNAVAILABLE" and record["error"] == h3.GPU_UNAVAILABLE_MESSAGE
    assert record["session_status"] == "not_created" and "upload" not in transport.subcommands


def test_no_gpu_in_the_runtime_is_reported_plainly(tmp_path):
    exec_ = emit(marker("ERROR", code="GPU_UNAVAILABLE", message="No CUDA GPU in this Colab runtime."))
    record, transport = run(tmp_path, transport=FakeTransport(exec=exec_))
    assert record["error_code"] == "GPU_UNAVAILABLE" and record["error"] == h3.GPU_UNAVAILABLE_MESSAGE
    assert "stop" in transport.subcommands


def test_model_download_failure_is_recorded_and_the_session_stopped(tmp_path):
    exec_ = emit(
        marker("STAGE", name="LOADING_MODEL"),
        marker("ERROR", code="MODEL_DOWNLOAD_FAILED", message="HfHubHTTPError: 503"),
    )
    record, transport = run(tmp_path, transport=FakeTransport(exec=exec_))
    assert record["status"] == h3.FAILED and record["error_code"] == "MODEL_DOWNLOAD_FAILED"
    assert record["failed_stage"] == h3.LOADING_MODEL
    assert transport.subcommands[-2:] == ["stop", "usage"] and "download" not in transport.subcommands
    assert (tmp_path / "out" / ".work" / "job1").is_dir()  # kept for debugging


def test_exec_timeout_stops_the_session_and_is_not_retried(tmp_path):
    transport = FakeTransport(exec=raising(h3.ColabTimeout("colab exec exceeded 3720 seconds")))
    record, _ = run(tmp_path, transport=transport)
    assert record["status"] == h3.TIMEOUT and record["error_code"] == "TIMEOUT"
    assert record["failed_stage"] == h3.LOADING_MODEL
    assert transport.subcommands.count("exec") == 1
    assert "stop" in transport.subcommands and "download" not in transport.subcommands


def test_remote_deadline_is_a_timeout(tmp_path):
    exec_ = emit(marker("STAGE", name="INFERENCE"), marker("ERROR", code="TIMEOUT", message="deadline"))
    record, _ = run(tmp_path, transport=FakeTransport(exec=exec_))
    assert record["status"] == h3.TIMEOUT and record["failed_stage"] == h3.INFERENCE


def test_ctrl_c_cancels_and_stops_the_session(tmp_path):
    record, transport = run(tmp_path, transport=FakeTransport(exec=raising(KeyboardInterrupt())))
    assert record["status"] == h3.CANCELLED and "stop" in transport.subcommands


def test_exec_without_an_output_marker_is_output_not_found(tmp_path):
    record, transport = run(tmp_path, transport=FakeTransport(exec=emit(marker("STAGE", name="INFERENCE"))))
    assert record["error_code"] == "OUTPUT_NOT_FOUND" and "download" not in transport.subcommands


def test_failed_download_is_output_not_found(tmp_path):
    missing = h3.ColabCommandError("download", 1, ["[colab] Download failed: 404 Not Found"])
    record, _ = run(tmp_path, transport=FakeTransport(download=raising(missing)))
    assert record["error_code"] == "OUTPUT_NOT_FOUND" and record["failed_stage"] == h3.DOWNLOADING


def test_download_that_writes_nothing_is_output_not_found(tmp_path):
    record, _ = run(tmp_path, transport=FakeTransport(download=lambda args, on_line: "ok"))
    assert record["error_code"] == "OUTPUT_NOT_FOUND"


def test_exec_failure_without_a_marker_is_inference_failed(tmp_path):
    exec_ = raising(h3.ColabCommandError("colab exec", 1, ["[colab] Session 'h3-job1' appears to be lost"]))
    record, _ = run(tmp_path, transport=FakeTransport(exec=exec_))
    assert record["error_code"] == "INFERENCE_FAILED"


def test_marker_beats_the_exit_code_when_both_fail(tmp_path):
    exec_ = emit(
        marker("ERROR", code="INSUFFICIENT_DISK", message="40 GiB free"),
        then=h3.ColabCommandError("colab exec", 1, ["Traceback"]),
    )
    record, _ = run(tmp_path, transport=FakeTransport(exec=exec_))
    assert record["error_code"] == "INSUFFICIENT_DISK"


def test_upload_failure(tmp_path):
    failed = h3.ColabCommandError("upload", 1, ["[colab] Upload failed: 500"])
    record, transport = run(tmp_path, transport=FakeTransport(upload=raising(failed)))
    assert record["error_code"] == "UPLOAD_FAILED" and "stop" in transport.subcommands


def test_corrupt_mp4_fails_validation_and_is_kept(tmp_path):
    record, _ = run(tmp_path, prober=FakeProber(video_error="moov atom not found"))
    assert record["status"] == h3.FAILED_VALIDATION and record["error_code"] == "FAILED_VALIDATION"
    assert (tmp_path / "out" / "first_h3_job1.mp4").is_file()


def test_wrong_clip_fails_validation(tmp_path):
    record, _ = run(tmp_path, prober=FakeProber(video=good_probe(width=1344, audio=False)))
    assert record["status"] == h3.FAILED_VALIDATION
    assert "resolution" in record["error"] and "no audio" in record["error"]


def test_a_session_that_will_not_stop_is_flagged(tmp_path):
    stuck = h3.ColabCommandError("colab stop", 1, ["[colab] 503 Service Unavailable"])
    record, _ = run(tmp_path, transport=FakeTransport(stop=raising(stuck)))
    assert record["status"] == h3.COMPLETED and record["session_status"] == "stop_failed"
    assert any("stop --session h3-job1" in w for w in record["warnings"])


def test_dry_run_spends_nothing_and_cleans_up(tmp_path):
    transport = FakeTransport()
    record = h3.run_job(
        h3.JobSpec(image=make_image(tmp_path), description="She smiles.", job_id="dry"),
        make_config(tmp_path),
        transport=transport,
        prober=FakeProber(),
        dry_run=True,
        stream=None,
    )
    assert record["status"] == h3.DRY_RUN
    assert "new" not in transport.subcommands and "exec" not in transport.subcommands
    assert record["dry_run"]["prompt"].startswith(h3.I2VA_ALIGNMENT)
    assert any("new --session h3-dry --gpu A100" in c for c in record["dry_run"]["commands"])
    assert {c["name"] for c in record["dry_run"]["preflight"]} >= {"config", "colab_auth"}
    assert not (tmp_path / "out" / ".work" / "dry").exists()
    assert not (tmp_path / "out" / "h3_jobs.jsonl").exists()


def test_job_spec_round_trips_through_its_nested_form():
    spec = h3.JobSpec(image="a.png", description="d", constraints=("locked-camera",), duration=5, scene_id="s01")
    again = h3.JobSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec


def test_run_cli_prints_the_record_last_and_uses_input_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("H3_OUTPUT_DIR", str(tmp_path / "out"))
    cli = import_skill("run")
    code = cli.main(["--image", str(tmp_path / "missing.png"), "--prompt", "x", "--config", str(tmp_path / "c.json")])
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and record["error_code"] == "INVALID_INPUT"


# --- the subprocess layer ------------------------------------------------------------------------------


def test_run_streaming_kills_a_hung_process_on_timeout():
    aborted = []
    started = time.monotonic()
    with pytest.raises(h3.ColabTimeout):
        h3.run_streaming(
            [sys.executable, "-c", "import time; print('hi', flush=True); time.sleep(30)"],
            label="sleeper",
            timeout=1,
            on_abort=lambda: aborted.append(True),
        )
    assert time.monotonic() - started < 10
    assert aborted == [True]


def test_run_streaming_streams_lines_and_reports_failures():
    seen = []
    with pytest.raises(h3.ColabCommandError) as err:
        h3.run_streaming(
            [sys.executable, "-c", "print('one', flush=True); print('two', flush=True); raise SystemExit(3)"],
            label="failing",
            timeout=30,
            on_line=seen.append,
        )
    assert seen == ["one", "two"] and err.value.returncode == 3 and err.value.output == ["one", "two"]


def test_docker_command_mounts_the_config_volume_and_the_work_dir():
    t = h3.DockerTransport("h3-colab-cli:0.7.4", "h3-colab-config", "oauth2")
    assert t.command(["usage"], mount_dir="D:\\AI\\output\\.work\\j", name="h3-x") == [
        "docker", "run", "--rm", "--init", "--name", "h3-x", "-e", "PYTHONUNBUFFERED=1",
        "-v", "h3-colab-config:/root/.config/colab-cli", "-v", "D:\\AI\\output\\.work\\j:/work",
        "h3-colab-cli:0.7.4", "--auth=oauth2", "usage",
    ]  # fmt: skip
    assert "-it" in t.login_command() and t.login_command().endswith("--auth=oauth2 usage")


def test_docker_paths_map_into_the_work_mount(tmp_path):
    t = h3.DockerTransport("img", "vol", "oauth2")
    assert t.cli_path(tmp_path / "sub" / "job.json", tmp_path) == "/work/sub/job.json"
    with pytest.raises(ValueError):
        t.cli_path(tmp_path.parent / "elsewhere.png", tmp_path)


def test_docker_abort_kills_the_named_container(monkeypatch):
    seen = []
    monkeypatch.setattr(h3.subprocess, "run", lambda cmd, **kw: seen.append(cmd))
    h3.DockerTransport("img", "vol", "oauth2").abort("h3-abc")
    assert seen == [["docker", "kill", "h3-abc"]]


def test_transport_choice_follows_the_config(tmp_path):
    assert h3.make_transport(make_config(tmp_path, H3_TRANSPORT="docker")).kind == "docker"
    assert h3.make_transport(make_config(tmp_path, H3_TRANSPORT="native")).kind == "native"


def test_usage_output_is_parsed():
    text = "[colab] notice\nCurrent balance: 1,234.50 compute units\nUsage rate: 11.77/hr\nActive assignments: 1"
    assert h3.parse_usage(text) == {"balance": 1234.5, "rate_per_hour": 11.77, "active_assignments": 1.0}
