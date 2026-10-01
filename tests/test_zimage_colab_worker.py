"""The remote worker (remote/zimage_worker.py) run in-process against a fake ComfyUI, fake downloads and a
temp "Drive". The two things this skill exists for are pinned here: a restarted session reinstalls and
re-downloads nothing, and one failing job never takes the worker (or the jobs after it) down."""

import faulthandler
import json
import threading
import time
from pathlib import Path

import pytest
from zimage_colab_fakes import (
    L4,
    FakeComfy,
    FakeComfyProcess,
    Installers,
    make_worker,
    payload_for,
    put_inbox,
    remote,
)


@pytest.fixture(autouse=True)
def no_hangs():
    """These tests run real threads. A deadlock should end the run with every thread's stack, not hang it."""
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def comfy():
    server = FakeComfy(model_names=["unet.safetensors", "te.safetensors", "vae.safetensors"])
    FakeComfyProcess.instances.clear()
    yield server
    server.close()


def first_install(tmp_path, comfy):
    w, inst, clock = make_worker(tmp_path, comfy)
    w.boot()
    w.shutdown("test")
    return w, inst


def restarted(tmp_path, comfy, vm="vm2", **kw):
    """A new Colab VM: empty local disk, same Drive."""
    return make_worker(tmp_path, comfy, local_root=str(tmp_path / vm / "zimg"), **kw)


# --- bootstrap: install once, then never again ---------------------------------------------------


def test_first_install_installs_everything_once(tmp_path, comfy):
    w, inst = first_install(tmp_path, comfy)
    assert w.actions["comfyui"] == "installed"
    assert w.actions["deps"] == "installed"
    assert {w.actions[f"model:{n}"] for n in ("unet.safetensors", "te.safetensors", "vae.safetensors")} == {
        "downloaded"
    }
    assert len(inst.urls) == 1 and len(inst.hf) == 3 and len(inst.pip) == 1
    p = w.paths
    assert (p.cache / f"comfyui-{'c' * 40}.tar.gz").is_file()
    assert list(p.cache.glob("pydeps-*.tar"))
    manifest = json.loads(p.manifest.read_text())
    assert set(manifest) == {"unet.safetensors", "te.safetensors", "vae.safetensors"}
    for entry in manifest.values():
        assert {"model_name", "model_version", "download_source", "file_size", "sha256", "installed_at"} <= set(entry)
    env = json.loads(p.environment.read_text())
    for key in (
        "comfyui_version",
        "z_image_version",
        "custom_nodes",
        "python_version",
        "torch_version",
        "cuda_version",
        "installed_at",
        "deps_fingerprint",
    ):
        assert key in env
    assert inst.flushes == 1  # Drive is flushed before the controller releases the VM


def test_restart_reinstalls_and_redownloads_nothing(tmp_path, comfy):
    first_install(tmp_path, comfy)
    w2, inst2, _ = restarted(tmp_path, comfy)
    w2.boot()
    assert inst2.urls == [] and inst2.hf == [] and inst2.pip == [], "a restarted session must not hit the network"
    assert w2.actions["comfyui"] == "extracted"
    assert w2.actions["deps"] == "extracted"
    assert {w2.actions[f"model:{n}"] for n in ("unet.safetensors", "te.safetensors", "vae.safetensors")} == {"staged"}
    assert w2.state.data["worker_state"] == remote.READY
    history = json.loads(w2.paths.environment.read_text())["history"]
    assert len(history) == 2  # one entry per session, the record of what each one did


def test_comfyui_runs_with_the_archived_packages_and_offline(tmp_path, comfy):
    first_install(tmp_path, comfy)
    w2, _, _ = restarted(tmp_path, comfy)
    w2.boot()
    proc = FakeComfyProcess.instances[-1]
    assert Path(proc.pyuser) == w2.paths.pyuser
    env = remote.ComfyProcess(
        w2.paths.comfy, port=1, extra_paths_yaml="x", output_dir="o", log_path="l", pyuser=w2.paths.pyuser
    ).env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["PIP_NO_INDEX"] == "1"
    assert env["PYTHONUSERBASE"] == str(w2.paths.pyuser)
    assert (w2.paths.pyuser / "lib" / "site-packages" / "fakepkg.py").is_file()


def test_changed_runtime_migrates_only_the_packages(tmp_path, comfy):
    first_install(tmp_path, comfy)
    w2, inst2, _ = restarted(tmp_path, comfy)
    w2.base_freeze = "torch==2.12.0\n"  # Colab updated its image
    w2.boot()
    assert w2.actions["deps"] == "migrated"
    assert len(inst2.pip) == 1
    assert inst2.hf == [] and inst2.urls == []
    assert w2.actions["comfyui"] == "extracted"


def test_changed_comfyui_commit_downloads_the_new_source_once(tmp_path, comfy):
    first_install(tmp_path, comfy)
    w2, inst2, _ = restarted(tmp_path, comfy, comfyui={"commit": "d" * 40})
    w2.boot()
    assert w2.actions["comfyui"] == "installed"
    assert len(inst2.urls) == 1
    assert inst2.hf == []  # the models did not change


def test_a_missing_model_is_downloaded_alone(tmp_path, comfy):
    w, _ = first_install(tmp_path, comfy)
    (w.paths.models / "vae" / "vae.safetensors").unlink()
    w2, inst2, _ = restarted(tmp_path, comfy)
    w2.boot()
    assert inst2.hf == ["split_files/vae/vae.safetensors"]
    assert w2.actions["model:vae.safetensors"] == "downloaded"
    assert w2.actions["model:unet.safetensors"] == "staged"


def test_a_truncated_model_on_drive_is_downloaded_again(tmp_path, comfy):
    w, _ = first_install(tmp_path, comfy)
    (w.paths.models / "diffusion_models" / "unet.safetensors").write_bytes(b"U" * 10)
    w2, inst2, _ = restarted(tmp_path, comfy)
    w2.boot()
    assert inst2.hf == ["split_files/diffusion_models/unet.safetensors"]


def test_a_checksum_mismatch_fails_the_boot(tmp_path, comfy):
    inst = Installers()
    original = inst.hf_download

    def corrupt(repo, filename, revision, local_dir):
        path = Path(original(repo, filename, revision, local_dir))
        path.write_bytes(b"X" * path.stat().st_size)
        return str(path)

    inst.hf_download = corrupt
    w, _, _ = make_worker(tmp_path, comfy, installers=inst)
    with pytest.raises(remote.WorkerError) as exc:
        w.boot()
    assert exc.value.code == "MODEL_CHECKSUM"


def test_a_gpu_below_the_requirement_never_starts_the_queue(tmp_path, comfy):
    t4 = dict(L4, name="Tesla T4", vram_total_mib=15360, bf16=False)
    w, inst, _ = make_worker(tmp_path, comfy, gpu=t4)
    with pytest.raises(remote.WorkerError) as exc:
        w.boot()
    assert exc.value.code == "GPU_NOT_SUPPORTED"
    assert inst.hf == [] and inst.urls == [] and FakeComfyProcess.instances == []


def test_too_little_vram_is_refused(tmp_path, comfy):
    small = dict(L4, vram_total_mib=8192)
    w, _, _ = make_worker(tmp_path, comfy, gpu=small)
    with pytest.raises(remote.WorkerError) as exc:
        w.boot()
    assert "8192 MiB" in str(exc.value)


def test_drive_must_be_mounted_in_drive_mode(tmp_path, comfy):
    w, _, _ = make_worker(tmp_path, comfy, drive_root=str(tmp_path / "nowhere"))
    with pytest.raises(remote.WorkerError) as exc:
        w.boot()
    assert exc.value.code == "DRIVE_NOT_MOUNTED"


def test_ephemeral_mode_installs_locally_and_never_flushes(tmp_path, comfy):
    w, inst, _ = make_worker(
        tmp_path, comfy, persistence="ephemeral", persistent_root=str(tmp_path / "vm" / "zimg" / "persist")
    )
    w.boot()
    w.shutdown("test")
    assert w.actions["storage"] == "ephemeral"
    assert len(inst.hf) == 3 and len(inst.pip) == 1
    assert inst.flushes == 0
    assert not list(w.paths.cache.glob("pydeps-*.tar")), "nothing outlives an ephemeral session: no archive"
    assert not (tmp_path / "drive" / "MyDrive" / "AI").exists()


def test_node_check_fails_when_a_model_is_not_listed(tmp_path, comfy):
    comfy.model_names = ["unet.safetensors", "te.safetensors"]
    w, _, _ = make_worker(tmp_path, comfy)
    with pytest.raises(remote.WorkerError) as exc:
        w.boot()
    assert exc.value.code == "NODE_CHECK_FAILED"
    assert "vae.safetensors" in str(exc.value)


# --- the job loop --------------------------------------------------------------------------------


def booted(tmp_path, comfy, **kw):
    w, inst, clock = make_worker(tmp_path, comfy, **kw)
    w.boot()
    return w, inst, clock


def test_one_failing_job_does_not_stop_the_next(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    comfy.behaviors["job-2"] = "error"
    for jid in ("job-1", "job-2", "job-3"):
        put_inbox(w, payload_for(jid))
    reason = w.run_forever()
    assert reason == "idle_timeout"
    jobs = w.state.data["jobs"]
    assert jobs["job-1"]["status"] == remote.COMPLETED
    assert jobs["job-2"]["status"] == remote.FAILED
    assert jobs["job-2"]["error"]["code"] == "COMFY_EXECUTION_ERROR"
    assert "CUDA out of memory" in jobs["job-2"]["error"]["error_message"]
    assert jobs["job-3"]["status"] == remote.COMPLETED
    err = jobs["job-2"]["error"]
    assert {"error_type", "error_message", "stack_trace", "timestamp"} <= set(err)
    errors_log = (w.paths.logs / "errors.jsonl").read_text().splitlines()
    assert len(errors_log) == 1


def test_three_jobs_share_one_session_and_record_their_order(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    for jid in ("a", "b", "c"):
        put_inbox(w, payload_for(jid))
    w.run_forever()
    metas = [json.loads((w.paths.outbox / jid / "metadata.json").read_text()) for jid in ("a", "b", "c")]
    assert [m["session_job_index"] for m in metas] == [1, 2, 3]
    assert [m["session_reused"] for m in metas] == [False, True, True]
    assert {m["session"] for m in metas} == {"zimg-test"}
    assert len(FakeComfyProcess.instances) == 1 and FakeComfyProcess.instances[0].started == 1


def test_metadata_has_what_a_cost_analysis_needs(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    put_inbox(w, payload_for("m1", params={"steps": 8, "seed": 7, "cfg": 2.0}))
    w.run_forever()
    meta = json.loads((w.paths.outputs / "m1" / "metadata.json").read_text())  # persisted on Drive
    for key in (
        "job_id",
        "status",
        "gpu",
        "vram_total_mib",
        "model",
        "model_revision",
        "workflow",
        "width",
        "height",
        "started_at",
        "completed_at",
        "generation_seconds",
        "comfy_execution_seconds",
        "comfyui_commit",
        "params",
        "validation",
    ):
        assert key in meta, key
    assert meta["gpu"] == "NVIDIA L4"
    assert meta["comfy_execution_seconds"] == 2.5
    assert meta["validation"]["ok"] is True
    assert (w.paths.outputs / "m1" / "result.png").is_file()
    assert json.loads((w.paths.outputs / "m1" / "workflow.json").read_text())["9"]["class_type"] == "SaveImage"


def test_a_job_that_never_finishes_times_out_and_is_removed_from_comfyui(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    comfy.behaviors["slow"] = "hang"
    put_inbox(w, payload_for("slow", timeout_seconds=5))
    put_inbox(w, payload_for("after"))
    w.run_forever()
    assert w.state.data["jobs"]["slow"]["status"] == remote.TIMEOUT
    assert w.state.data["jobs"]["after"]["status"] == remote.COMPLETED
    posted = [c[1] for c in comfy.calls if c[0] == "POST"]
    assert "/queue" in posted and "/interrupt" in posted


@pytest.mark.parametrize("behavior,needle", [("wrong_size", "resolution 32x24"), ("bad_png", "")])
def test_a_bad_image_is_invalid_output(tmp_path, comfy, behavior, needle):
    w, _, _ = booted(tmp_path, comfy)
    comfy.behaviors["bad"] = behavior
    put_inbox(w, payload_for("bad"))
    w.run_forever()
    job = w.state.data["jobs"]["bad"]
    assert job["status"] == remote.INVALID_OUTPUT
    assert needle in job["error"]["error_message"]


def test_comfyui_rejecting_the_graph_fails_only_that_job(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    comfy.behaviors["r"] = "reject"
    put_inbox(w, payload_for("r"))
    put_inbox(w, payload_for("ok"))
    w.run_forever()
    assert w.state.data["jobs"]["r"]["error"]["code"] == "COMFY_REJECTED"
    assert w.state.data["jobs"]["ok"]["status"] == remote.COMPLETED


def test_an_unsafe_graph_is_refused_before_comfyui_sees_it(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    p = payload_for("low-cfg")
    p["graph"]["3"]["inputs"]["cfg"] = 1.0
    q = payload_for("no-age")
    q["graph"]["7"]["inputs"]["text"] = "lowres, blurry"
    put_inbox(w, p)
    put_inbox(w, q)
    w.run_forever()
    for jid in ("low-cfg", "no-age"):
        assert w.state.data["jobs"][jid]["error"]["code"] == "UNSAFE_JOB"
    assert not [c for c in comfy.calls if c[:2] == ("POST", "/prompt")]


def test_a_half_uploaded_inbox_file_is_retried_then_rejected(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    bad = w.paths.inbox / "partial.json"
    bad.write_text('{"job_id": "partial", "graph": {')
    for _ in range(remote.INBOX_PARSE_RETRIES - 1):
        w.ingest_inbox()
    assert bad.exists() and "partial" not in w.state.data["rejected"]
    bad.write_text(json.dumps(payload_for("partial")))  # the upload finished
    w.ingest_inbox()
    assert [p["job_id"] for p in w.queue] == ["partial"]


def test_a_never_valid_inbox_file_is_rejected(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    (w.paths.inbox / "junk.json").write_text("not json")
    for _ in range(remote.INBOX_PARSE_RETRIES):
        w.ingest_inbox()
    assert "junk" in w.state.data["rejected"]


def test_cancel_before_start(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    put_inbox(w, payload_for("x"))
    (w.paths.cancel / "x").write_text("{}")
    w.ingest_inbox()
    w.apply_cancels()
    assert w.queue == []
    assert w.state.data["jobs"]["x"]["status"] == remote.CANCELLED


def test_the_same_job_pushed_twice_in_a_session_renders_once(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    put_inbox(w, payload_for("dup"))
    w.ingest_inbox()
    put_inbox(w, payload_for("dup"))
    w.run_forever()
    assert len([c for c in comfy.calls if c[:2] == ("POST", "/prompt")]) == 1


def test_a_job_finished_by_a_lost_session_is_rereported_not_rerendered(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    put_inbox(w, payload_for("done-before"))
    w.run_forever()
    w.shutdown("lost")
    prompts_before = len([c for c in comfy.calls if c[:2] == ("POST", "/prompt")])
    w2, _, _ = restarted(tmp_path, comfy)
    w2.boot()
    put_inbox(w2, payload_for("done-before"))
    w2.run_forever()
    assert w2.state.data["jobs"]["done-before"]["status"] == remote.COMPLETED
    assert w2.state.data["jobs"]["done-before"]["reused_previous_output"] is True
    assert (w2.paths.outbox / "done-before" / "result.png").is_file()
    assert len([c for c in comfy.calls if c[:2] == ("POST", "/prompt")]) == prompts_before


def test_a_crashed_comfyui_is_restarted_for_the_next_job(tmp_path, comfy):
    w, _, _ = booted(tmp_path, comfy)
    FakeComfyProcess.instances[-1].alive = False
    put_inbox(w, payload_for("after-crash"))
    w.run_forever()
    assert w.state.data["jobs"]["after-crash"]["status"] == remote.COMPLETED
    assert FakeComfyProcess.instances[-1].started == 2


# --- idle shutdown -------------------------------------------------------------------------------


def test_idle_timeout_comes_from_the_config(tmp_path, comfy):
    w, _, clock = booted(tmp_path, comfy, worker={"idle_timeout_seconds": 45})
    t0 = clock()
    assert w.run_forever() == "idle_timeout"
    assert 45 <= clock() - t0 < 47


def test_a_job_arriving_at_the_deadline_is_not_stranded(tmp_path, comfy):
    w, _, clock = booted(tmp_path, comfy, worker={"idle_timeout_seconds": 10})
    real_sleep = w.sleep

    def sleep(seconds):
        real_sleep(seconds)
        if clock() >= 1000 + 10 and not (w.paths.inbox / "late.json").exists() and "late" not in w.seen:
            put_inbox(w, payload_for("late"))

    w.sleep = sleep
    w.run_forever()
    assert w.state.data["jobs"]["late"]["status"] == remote.COMPLETED


def test_shutdown_stops_comfyui_cleans_up_and_flushes(tmp_path, comfy):
    w, inst, _ = booted(tmp_path, comfy)
    (w.paths.comfy_out / "leftover.png").write_bytes(b"x")
    w.shutdown("idle_timeout")
    assert FakeComfyProcess.instances[-1].running() is False
    assert not w.paths.comfy_out.exists()
    assert inst.flushes == 1
    state = json.loads(w.paths.state.read_text())
    assert state["worker_state"] == remote.FLUSHED
    assert state["shutdown_reason"] == "idle_timeout"
    sessions = (w.paths.logs / "sessions.jsonl").read_text().splitlines()
    assert json.loads(sessions[-1])["reason"] == "idle_timeout"


def test_max_session_ends_the_loop_even_with_work_left(tmp_path, comfy):
    w, _, clock = booted(tmp_path, comfy, worker={"max_session_seconds": 100})
    clock.now += 200
    put_inbox(w, payload_for("too-late"))
    assert w.run_forever() == "max_session"


def test_run_worker_shuts_down_cleanly_on_interrupt(tmp_path, comfy):
    w, inst, _ = booted(tmp_path, comfy)
    w.boot = lambda: None

    def interrupted():
        raise KeyboardInterrupt

    w.run_forever = interrupted
    assert remote.run_worker(w, print_every_seconds=3600) == "interrupted"
    assert json.loads(w.paths.state.read_text())["worker_state"] == remote.FLUSHED
    assert inst.flushes == 1


def test_run_worker_reports_a_failed_boot_and_still_shuts_down(tmp_path, comfy):
    w, inst, _ = make_worker(tmp_path, comfy, gpu=dict(L4, available=False))
    assert remote.run_worker(w, print_every_seconds=3600) == "boot_failed:GPU_NOT_SUPPORTED"
    state = json.loads(w.paths.state.read_text())
    assert state["error"]["code"] == "GPU_NOT_SUPPORTED"
    assert state["worker_state"] == remote.FLUSHED


def test_the_heartbeat_advances_while_a_phase_is_silent(tmp_path):
    state = remote.State(tmp_path / "state.json", echo=lambda line: None)
    state.write()
    first = state.data["seq"]
    state.start_heartbeat(0.01, print_every_seconds=3600)
    time.sleep(0.2)  # nothing else touches the state, like a 20 GB copy
    state.stop_heartbeat()
    assert json.loads((tmp_path / "state.json").read_text())["seq"] > first + 3


def test_state_writes_are_atomic_under_concurrency(tmp_path):
    state = remote.State(tmp_path / "state.json", echo=lambda line: None)
    state.write()  # the file exists from the first line of run_worker on
    stop = threading.Event()
    writer_errors = []

    def writer():
        try:
            while not stop.is_set():
                state.job("j", status="RUNNING")
        except Exception as exc:  # noqa: BLE001 - reported by the assert below
            writer_errors.append(exc)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        reads = 0
        while reads < 200:
            try:
                text = (tmp_path / "state.json").read_text()
            except PermissionError:  # Windows: the rename is in progress; Linux (the VM) never raises here
                continue
            json.loads(text)  # never a half-written file
            reads += 1
    finally:
        stop.set()
        t.join(timeout=10)
    assert writer_errors == []


def test_the_remote_script_is_ascii_and_has_no_future_import():
    """`colab exec --env` prepends code to the file, so `from __future__` would be a SyntaxError there."""
    src = Path(remote.__file__).read_bytes()
    src.decode("ascii")
    assert b"from __future__" not in src
