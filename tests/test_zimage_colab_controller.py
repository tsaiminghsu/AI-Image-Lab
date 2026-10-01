"""The local controller against a fake `colab` (FakeTransport) and the real remote worker running
in-process (InProcessExec). This is the offline version of the live acceptance tests:
- 3 jobs -> exactly one `colab new`, one `colab stop`, all three results downloaded and validated;
- once the worker cell runs, the controller never calls `colab exec` or `colab drivemount` again;
- every way out (worker shutdown, stale heartbeat, max session, stop request, errors) stops the VM."""

import faulthandler
import json

import pytest
from zimage_colab_fakes import (
    FakeClock,
    FakeComfy,
    FakeComfyProcess,
    FakeTransport,
    FastSleep,
    InProcessExec,
    Installers,
    make_config,
    mounted,
    not_mounted,
    z,
)


@pytest.fixture(autouse=True)
def no_hangs():
    """A worker thread that dies leaves the controller waiting out its 300 s heartbeat timeout on the real
    clock. End the run with every thread's stack instead."""
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def comfy():
    server = FakeComfy(model_names=["unet.safetensors", "te.safetensors", "vae.safetensors"])
    FakeComfyProcess.instances.clear()
    yield server
    server.close()


def queue_jobs(config, n=3, **kw):
    store = z.JobStore(z.output_root(config))
    jobs = []
    for i in range(n):
        job = z.build_job(config, prompt=f"a ceramic mug on a white table, view {i}", aspect="1:1", **kw)
        store.add(job)
        jobs.append(job)
    return store, jobs


def controller(config, transport, exec_factory, *, mounter=mounted, clock=None, sleep=None, opened=None):
    opened = [] if opened is None else opened
    return z.Controller(
        config,
        transport=transport,
        exec_factory=exec_factory,
        drive_mounter=mounter,
        open_url=opened.append,
        awake=lambda on: None,
        echo=lambda line: None,
        clock=clock or __import__("time").monotonic,
        sleep=sleep or FastSleep(),
    )


def test_three_jobs_one_session_end_to_end(tmp_path, comfy):
    config = make_config(tmp_path)
    store, jobs = queue_jobs(config, 3)
    transport = FakeTransport(tmp_path / "vm")
    exec_factory = InProcessExec(transport, comfy, Installers(), tmp_path)
    opened = []
    summary = controller(config, transport, exec_factory, opened=opened).run()

    assert summary["reason"] == "worker_idle_timeout"
    assert transport.names().count("new") == 1
    assert transport.names().count("stop") == 1
    assert "exec" not in transport.names() and "drivemount" not in transport.names()
    assert summary["persistence"] == "drive"
    assert opened == ["https://accounts.google.com/o/oauth2/v2/auth?fake=1"]  # consent page opened for the user
    sessions = set()
    for job in jobs:
        done = store.get(job["job_id"])
        assert done["status"] == z.COMPLETED, done.get("error")
        out = z.output_root(config) / job["job_id"]
        assert z.validate_output(out / "result.png", 1024, 1024) == []
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert meta["validation"]["local"]["ok"] and meta["validation"]["remote"]["ok"]
        assert json.loads((out / "workflow.json").read_text())["6"]["inputs"]["text"] == job["prompt"]
        sessions.add(done["session"])
    assert len(sessions) == 1
    ledger = (z.output_root(config) / "zimage_jobs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 3
    first = json.loads(ledger[0])
    for key in (
        "job_id",
        "status",
        "gpu",
        "vram",
        "model",
        "workflow",
        "width",
        "height",
        "steps",
        "seed",
        "started_at",
        "completed_at",
        "generation_seconds",
        "output",
    ):
        assert key in first
    assert sorted(json.loads(line)["session_reused"] for line in ledger) == [False, True, True]
    session_ledger = (z.output_root(config) / "zimage_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(session_ledger) == 1
    s = json.loads(session_ledger[0])
    assert s["cu_balance_before"] == 100.0 and s["cu_rate_per_hour"] == 1.71
    assert s["setup_actions"]["comfyui"] == "installed"
    assert z.ControllerState(z.output_root(config)).read()["state"] == "stopped"


def test_the_second_session_reinstalls_nothing(tmp_path, comfy):
    config = make_config(tmp_path)
    transport = FakeTransport(tmp_path / "vm")
    installers = Installers()
    queue_jobs(config, 1)
    controller(config, transport, InProcessExec(transport, comfy, installers, tmp_path)).run()
    # A new VM: wipe the VM-local dir, keep "Drive".
    import shutil

    shutil.rmtree(tmp_path / "vm" / "content" / "zimg")
    installers2 = Installers()
    transport2 = FakeTransport(tmp_path / "vm")
    queue_jobs(config, 1)
    summary = controller(config, transport2, InProcessExec(transport2, comfy, installers2, tmp_path)).run()
    assert installers2.hf == [] and installers2.urls == [] and installers2.pip == []
    assert summary["setup_actions"]["comfyui"] == "extracted"
    assert summary["setup_actions"]["deps"] == "extracted"


def test_without_drive_consent_the_session_still_runs(tmp_path, comfy):
    config = make_config(tmp_path)
    store, jobs = queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")
    installers = Installers()
    summary = controller(
        config, transport, InProcessExec(transport, comfy, installers, tmp_path), mounter=not_mounted
    ).run()
    assert summary["persistence"] == "ephemeral"
    assert summary["drive"]["reason"] == "consent_not_given"
    assert store.get(jobs[0]["job_id"])["status"] == z.COMPLETED
    assert installers.flushes == 0
    assert not (tmp_path / "vm" / "content" / "drive").exists()


def test_drive_off_never_asks_for_consent(tmp_path, comfy):
    config = make_config(tmp_path)
    config["colab"]["drive"] = "off"
    queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")

    def mounter(*a, **k):
        raise AssertionError("must not mount")

    summary = controller(
        config, transport, InProcessExec(transport, comfy, Installers(), tmp_path), mounter=mounter
    ).run()
    assert summary["persistence"] == "ephemeral"


def test_another_runtime_on_the_account_is_colab_busy(tmp_path):
    config = make_config(tmp_path)
    queue_jobs(config, 1)
    transport = FakeTransport(
        tmp_path / "vm", sessions_text="[h3-20261001] m-s-abc | Hardware: A100 | Shape: High-RAM | Variant: GPU"
    )
    summary = controller(config, transport, None).run()
    assert summary["error"]["code"] == "COLAB_BUSY"
    assert "new" not in transport.names()


def test_an_orphaned_zimg_session_is_stopped_first(tmp_path, comfy):
    config = make_config(tmp_path)
    queue_jobs(config, 1)
    transport = FakeTransport(
        tmp_path / "vm", sessions_text="[zimg-20261001-200000] m-s-x | Hardware: L4 | Shape: Standard | Variant: GPU"
    )
    controller(config, transport, InProcessExec(transport, comfy, Installers(), tmp_path)).run()
    assert transport.calls[2] == ["stop", "--session", "zimg-20261001-200000"]
    assert transport.names().index("stop") < transport.names().index("new")


def test_auth_required_stops_before_any_session(tmp_path):
    config = make_config(tmp_path)
    queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")
    transport.fail["usage"] = z.ColabCommandError("colab usage", 1, ["google.auth.exceptions.RefreshError"])
    summary = controller(config, transport, None).run()
    assert summary["error"]["code"] == "AUTH_REQUIRED"
    assert "colab login" in summary["error"]["hint"]
    assert "new" not in transport.names()


def test_gpu_unavailable_maps_to_its_own_code_and_still_stops(tmp_path):
    config = make_config(tmp_path)
    queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")
    transport.fail["new"] = z.ColabCommandError(
        "colab new", 1, ["POST https://colab.research.google.com/tun/m/assign 503 Service Unavailable"]
    )
    summary = controller(config, transport, None).run()
    assert summary["error"]["code"] == "GPU_UNAVAILABLE"
    assert transport.names().count("stop") == 1  # a half-created session is never left behind


def test_no_jobs_means_no_session(tmp_path):
    config = make_config(tmp_path)
    transport = FakeTransport(tmp_path / "vm")
    assert controller(config, transport, None).run()["reason"] == "no_jobs"
    assert "new" not in transport.names()


class SilentExec:
    """A worker that never writes run/state.json (exec connection dead, VM reset)."""

    def __init__(self, done=False):
        import threading

        self.done = threading.Event()
        if done:
            self.done.set()
        self.error = None

    def __call__(self, *args, **kwargs):
        return self

    def stop_local(self):
        pass


def test_a_worker_that_never_reports_is_stopped_after_the_heartbeat_timeout(tmp_path):
    config = make_config(tmp_path)
    store, jobs = queue_jobs(config, 2)
    transport = FakeTransport(tmp_path / "vm")
    clock = FakeClock()
    summary = controller(config, transport, SilentExec(), clock=clock, sleep=clock.sleep).run()
    assert summary["reason"] == "heartbeat_stale"
    assert transport.names().count("stop") == 1
    assert clock() - 1000 <= config["controller"]["heartbeat_timeout_seconds"] + 2 * 10
    for job in jobs:
        rec = store.get(job["job_id"])
        assert rec["status"] == z.PENDING and rec["attempt"] == 0  # never handed over, nothing spent


def test_an_exec_that_died_before_the_worker_started_fails_fast(tmp_path):
    config = make_config(tmp_path)
    queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")
    clock = FakeClock()
    summary = controller(config, transport, SilentExec(done=True), clock=clock, sleep=clock.sleep).run()
    assert summary["reason"].startswith("exec_failed")
    assert clock() - 1000 < 60


def write_state(transport, **data):
    path = transport.vm("/content/zimg/run/state.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_a_frozen_seq_counts_as_a_lost_vm_and_requeues_its_jobs(tmp_path):
    config = make_config(tmp_path)
    store, jobs = queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")
    clock = FakeClock()
    write_state(transport, seq=7, worker_state="READY", jobs={})
    summary = controller(config, transport, SilentExec(), clock=clock, sleep=clock.sleep).run()
    assert summary["reason"] == "heartbeat_stale"
    rec = store.get(jobs[0]["job_id"])
    # It was uploaded but the worker never listed it: back to PENDING without using up an attempt.
    assert rec["status"] == z.PENDING and rec["attempt"] == 0
    assert transport.names().count("stop") == 1


def test_a_job_lost_twice_fails_instead_of_looping(tmp_path):
    config = make_config(tmp_path)
    store, jobs = queue_jobs(config, 1)
    store.update(jobs[0]["job_id"], attempt=z.MAX_ATTEMPTS)
    transport = FakeTransport(tmp_path / "vm")
    clock = FakeClock()
    write_state(transport, seq=1, worker_state="READY", jobs={})
    controller(config, transport, SilentExec(), clock=clock, sleep=clock.sleep).run()
    rec = store.get(jobs[0]["job_id"])
    assert rec["status"] == z.FAILED and rec["error"]["code"] == "SESSION_LOST"


class AdvancingExec(SilentExec):
    """The worker is alive (seq keeps moving) but never finishes - the max-session cap must still stop it."""

    def __init__(self, transport):
        super().__init__()
        self.transport = transport
        self.seq = 0

    def tick(self):
        self.seq += 1
        write_state(self.transport, seq=self.seq, worker_state="BUSY", jobs={})


def test_max_session_caps_a_worker_that_never_ends(tmp_path):
    config = make_config(tmp_path, ZIMG_MAX_SESSION_SECONDS="600")
    config["controller"]["poll_interval_seconds"] = 300  # 14 polls of fake time instead of 1440
    queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")
    clock = FakeClock()
    ex = AdvancingExec(transport)

    def sleep(seconds):
        clock.sleep(seconds)
        ex.tick()

    summary = controller(config, transport, ex, clock=clock, sleep=sleep).run()
    assert summary["reason"] == "max_session"
    assert transport.names().count("stop") == 1


def test_a_stop_request_ends_the_session(tmp_path):
    config = make_config(tmp_path)
    queue_jobs(config, 1)
    transport = FakeTransport(tmp_path / "vm")
    clock = FakeClock()
    ex = AdvancingExec(transport)
    root = z.output_root(config)

    def sleep(seconds):
        clock.sleep(seconds)
        ex.tick()
        (root / "stop.request").write_text("now")

    summary = controller(config, transport, ex, clock=clock, sleep=sleep).run()
    assert summary["reason"] == "stop_requested"
    assert transport.names().count("stop") == 1


def test_cancel_requests_are_forwarded_to_the_worker(tmp_path):
    config = make_config(tmp_path)
    store, jobs = queue_jobs(config, 1)
    jid = jobs[0]["job_id"]
    store.update(jid, status=z.QUEUED)
    assert store.request_cancel(jid) == "CANCEL_REQUESTED"
    transport = FakeTransport(tmp_path / "vm")
    c = controller(config, transport, None)
    c.session = "zimg-x"
    c.forward_cancels()
    assert transport.vm(f"/content/zimg/cancel/{jid}").is_file()
    assert not (z.output_root(config) / "cancel" / jid).exists()


def test_a_second_controller_refuses_to_start(tmp_path):
    config = make_config(tmp_path)
    root = z.output_root(config)
    z.ControllerState(root).write(state="running", session="zimg-live")  # this process, fresh heartbeat
    data = json.loads((root / "controller.json").read_text())
    data["pid"] = __import__("os").getppid()  # a live process that is not us
    (root / "controller.json").write_text(json.dumps(data))
    queue_jobs(config, 1)
    with pytest.raises(z.ZImageError) as exc:
        controller(config, FakeTransport(tmp_path / "vm"), None).run()
    assert exc.value.code == "CONTROLLER_RUNNING"


def test_a_dead_controllers_lock_does_not_block_the_next_one(tmp_path):
    config = make_config(tmp_path)
    root = z.output_root(config)
    root.mkdir(parents=True)
    (root / "controller.lock").write_text("99999999")
    (root / "controller.json").write_text(json.dumps({"pid": 99999999, "state": "running", "heartbeat_epoch": 0}))
    assert controller(config, FakeTransport(tmp_path / "vm"), None).run()["reason"] == "no_jobs"


def test_run_config_carries_no_secrets_and_the_pinned_versions(tmp_path):
    config = make_config(tmp_path)
    c = controller(config, FakeTransport(tmp_path / "vm"), None)
    c.session = "zimg-x"
    c.persistence = "drive"
    run_cfg = c.run_config()
    assert z.secret_like_keys(run_cfg) == []
    assert run_cfg["comfyui"]["commit"] == config["comfyui"]["commit"]
    assert run_cfg["comfyui"]["tarball_url"].endswith(config["comfyui"]["commit"])
    assert run_cfg["persistent_root"] == "/content/drive/MyDrive/AI/ZImage"
    assert {f["loader"] for f in run_cfg["model"]["files"]} == {"UNETLoader", "CLIPLoader", "VAELoader"}
    c.persistence = "ephemeral"
    assert c.run_config()["persistent_root"] == "/content/zimg/persist"


def test_parse_sessions_reads_the_cli_format():
    text = (
        "[zimg-20261001-231500] m-s-abc | Hardware: L4 | Shape: Standard | Variant: GPU\n"
        "[?] m-s-def | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
        "[colab] something else\n"
    )
    names = [s["name"] for s in z.parse_sessions(text)]
    assert names == ["zimg-20261001-231500", "?"]
    assert z.parse_sessions("[colab] No active sessions found on server.") == []


def test_parse_usage():
    u = z.parse_usage("Current balance: 1,060.25 compute units\nUsage rate: 1.71/hr\nActive assignments: 1")
    assert u == {"balance": 1060.25, "rate_per_hour": 1.71, "active_assignments": 1}


def test_interactive_command_runs_the_cli_under_a_pseudo_terminal():
    t = z.DockerTransport("img:1", "vol", "oauth2")
    cmd = t.interactive_command(["drivemount", "--session", "zimg-x", "/content/drive"], name="zimg-n")
    assert cmd[:3] == ["docker", "run", "-i"] and "-t" not in cmd
    assert cmd[cmd.index("--entrypoint") + 1] == "script"
    assert "colab --auth=oauth2 drivemount --session zimg-x /content/drive" in cmd
