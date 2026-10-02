"""The H3 skill's storage rule: model files and finished clips are kept in Google Drive, the Colab VM only
holds work in progress. No Docker, Colab, Drive or GPU here - the remote script runs in-process against temp
folders, and the runner against a scripted fake `colab`.

What must hold: a second session copies the models from the workspace instead of downloading them; a manifest
entry is only ever written for a complete copy; the layout and manifest are the platform's (ai_workflow/), so
the 43 GB exist once; and nothing about saving to Drive may keep a GPU running - the session is stopped
whether the save worked or not.
"""

import json
import sys
import types
from pathlib import Path

import pytest
from controller.environment import Environment
from controller.storage import DriveFolderStorage
from minimax_h3_fakes import (
    HAPPY_EXEC_LINES,
    SKILL_DIR,
    FakeProber,
    FakeTransport,
    blob,
    emit,
    fake_hub,
    h3,
    import_remote,
    import_skill,
    make_config,
    make_image,
    marker,
    raising,
    small_specs,
)

remote = import_remote()
REGISTRY = Path(__file__).resolve().parents[1] / "ai_workflow" / "workflows" / "registry.json"
PIN_KEYS = ("name", "folder", "store", "repo", "revision", "path_in_repo", "size", "sha256")
MOUNTED = {"mounted": True, "consent_asked": True, "confirmed_by_user": True, "seconds": 12.0, "reason": None}
NOT_MOUNTED = {"mounted": False, "consent_asked": True, "confirmed_by_user": False, "seconds": 95.0, "reason": "consent_not_given"}  # fmt: skip


# --- the pins and the layout are the platform's --------------------------------------------------------


def test_the_model_pins_are_the_platforms_and_cover_every_file_the_graph_loads():
    config = make_config(Path("unused"))
    ours = {m["name"]: m for m in config["models"]}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["workflows"]["minimax-h3-basic"]["models"]
    assert len(registry) == 5
    for theirs in registry:
        assert {k: ours[theirs["name"]][k] for k in PIN_KEYS} == {k: theirs[k] for k in PIN_KEYS}
    assert set(ours) == {r["name"] for r in registry} | {remote.DIFFUSION_INT8}
    assert {remote.DIFFUSION_FP8, remote.TEXT_ENCODER, remote.VIDEO_VAE, remote.AUDIO_VAE, remote.LORA} <= set(ours)
    assert h3.FINALIZE_ENV == remote.FINALIZE_ENV == "H3_FINALIZE"


def test_a_model_without_a_pin_is_refused():
    with pytest.raises(remote.StageError) as exc:
        remote.needed_models({"models": small_specs()[1:]}, remote.DIFFUSION_FP8)
    assert exc.value.code == "MODEL_DOWNLOAD_FAILED" and remote.DIFFUSION_FP8 in str(exc.value)


# --- the remote script ---------------------------------------------------------------------------------


@pytest.fixture
def vm(tmp_path, monkeypatch):
    """A fresh 'VM' (ComfyUI dir, stage dir, no copy threads) next to a persistent 'Drive' workspace."""
    state = types.SimpleNamespace(calls=[], workspace=tmp_path / "drive" / "AI-Workflow", n=0, monkeypatch=monkeypatch)

    def new_session(blobs=blob):
        state.n += 1
        state.calls = []
        monkeypatch.setattr(remote, "COMFY", tmp_path / ("vm%d" % state.n) / "ComfyUI")
        monkeypatch.setattr(remote, "STAGE_DIR", tmp_path / ("vm%d" % state.n) / "stage")
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(state.calls, blobs=blobs))
        monkeypatch.setattr(sys, remote.SYNC_ATTR, [], raising=False)
        return {"persistence": "drive", "workspace": str(state.workspace), "drive_output": "outputs/videos/clip.mp4"}

    state.new_session = new_session
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    return state


def wait_for_copies():
    for thread in remote.sync_threads():
        thread.join(30)
    return {name: result for thread in remote.sync_threads() for name, result in thread.report.items()}


def actions(capsys):
    lines = [h3.parse_marker(line) for line in capsys.readouterr().out.splitlines()]
    return {data["name"]: data["action"] for kind, data in filter(None, lines) if kind == "MODEL"}


def test_the_first_session_downloads_and_the_second_copies_from_drive(vm, capsys):
    specs = small_specs()
    storage = vm.new_session()
    remote.ensure_models(specs, storage)
    assert set(actions(capsys).values()) == {"downloaded"} and len(vm.calls) == 5
    report = wait_for_copies()
    assert len(report) == 5 and all(r.startswith("copied") for r in report.values())
    manifest = json.loads((vm.workspace / "models" / "manifest.json").read_text(encoding="utf-8"))
    for m in specs:
        assert (vm.workspace / m["store"] / m["name"]).read_bytes() == blob(m["name"])
        entry = manifest["%s/%s" % (m["store"], m["name"])]
        assert (entry["revision"], entry["sha256"], entry["size"]) == (m["revision"], m["sha256"], m["size"])
    assert (vm.workspace / "loras" / remote.LORA).is_file()  # the LoRA goes to loras/, like on the platform

    storage = vm.new_session()  # a new VM: empty ComfyUI/models, the same Drive
    remote.ensure_models(specs, storage)
    assert vm.calls == [] and set(actions(capsys).values()) == {"staged"}
    assert remote.sync_threads() == []  # nothing left to copy
    for m in specs:
        assert (remote.COMFY / "models" / m["folder"] / m["name"]).read_bytes() == blob(m["name"])


def test_the_platform_accepts_what_the_skill_stored_and_the_other_way_round(vm, tmp_path, capsys):
    """One copy of the models for both: the skill's manifest satisfies the platform's check, and a workspace
    filled by the platform is 'staged' by the skill."""
    specs = small_specs()
    remote.ensure_models(specs, vm.new_session())
    wait_for_copies()
    capsys.readouterr()
    env = Environment(DriveFolderStorage(vm.workspace), {"comfyui": {}}, local_root=tmp_path / "platform-vm")
    assert [ok for _, ok in env.model_status(specs)] == [True] * 5

    other = tmp_path / "drive2" / "AI-Workflow"
    (other / "cache").mkdir(parents=True)
    platform = Environment(DriveFolderStorage(other), {"comfyui": {}}, local_root=tmp_path / "platform-vm2")
    for m in specs:
        local = tmp_path / "src" / m["name"]
        local.parent.mkdir(exist_ok=True)
        local.write_bytes(blob(m["name"]))
        platform.save_to_workspace(m, local)
    storage = dict(vm.new_session(), workspace=str(other))
    remote.ensure_models(specs, storage)
    assert vm.calls == [] and set(actions(capsys).values()) == {"staged"}


def test_only_a_damaged_drive_file_is_downloaded_again(vm, capsys):
    specs = small_specs()
    remote.ensure_models(specs, vm.new_session())
    wait_for_copies()
    capsys.readouterr()
    (vm.workspace / "models" / "minimax-h3" / "vae" / remote.VIDEO_VAE).write_bytes(b"truncated")
    remote.ensure_models(specs, vm.new_session())
    got = actions(capsys)
    assert got.pop(remote.VIDEO_VAE) == "downloaded" and set(got.values()) == {"staged"}
    assert [c["file"] for c in vm.calls] == ["vae/" + remote.VIDEO_VAE]
    wait_for_copies()
    assert (vm.workspace / "models" / "minimax-h3" / "vae" / remote.VIDEO_VAE).read_bytes() == blob(remote.VIDEO_VAE)


def test_a_wrong_checksum_fails_the_job_and_reaches_neither_comfyui_nor_drive(vm):
    storage = vm.new_session(blobs=lambda name: b"tampered" if name == remote.LORA else blob(name))
    with pytest.raises(remote.StageError) as exc:
        remote.ensure_models(small_specs(), storage)
    assert exc.value.code == "MODEL_DOWNLOAD_FAILED" and remote.LORA in str(exc.value)
    assert not (remote.COMFY / "models" / "loras" / remote.LORA).exists()
    wait_for_copies()
    assert not (vm.workspace / "loras" / remote.LORA).exists()


def test_without_drive_nothing_is_stored(vm, capsys):
    storage = dict(vm.new_session(), persistence="ephemeral")
    remote.ensure_models(small_specs(), storage)
    assert set(actions(capsys).values()) == {"downloaded"}
    assert remote.sync_threads() == [] and not vm.workspace.exists()
    video = remote.COMFY / "out.mp4"
    video.write_bytes(b"mp4")
    remote.save_output(video, storage)
    assert not vm.workspace.exists() and "DRIVE_OUTPUT" not in capsys.readouterr().out


def test_a_copy_that_failed_is_sent_again_by_the_next_clip_without_a_download(vm, capsys):
    specs = small_specs()
    storage = vm.new_session()
    real_copy, calls = remote.copy_large, []

    def flaky(src, dst, block=0):
        calls.append(dst)
        if (
            str(dst).startswith(str(vm.workspace))
            and len([c for c in calls if str(c).startswith(str(vm.workspace))]) <= 5
        ):
            raise OSError("quota exceeded")
        real_copy(src, dst)

    vm.monkeypatch.setattr(remote, "copy_large", flaky)
    remote.ensure_models(specs, storage)
    assert all(r.startswith("failed") for r in wait_for_copies().values())
    assert not (vm.workspace / "models" / "manifest.json").exists()
    capsys.readouterr()
    vm.calls.clear()
    remote.ensure_models(specs, storage)  # the next clip of the same session
    assert vm.calls == [] and set(actions(capsys).values()) == {"cached"}
    assert all(r.startswith("copied") for r in wait_for_copies().values())
    assert len(json.loads((vm.workspace / "models" / "manifest.json").read_text(encoding="utf-8"))) == 5


def test_the_clip_is_copied_to_drive_and_a_failure_is_a_warning_not_an_error(vm, capsys):
    storage = vm.new_session()
    video = remote.COMFY / "h3_job_output.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"mp4 bytes")
    remote.save_output(video, storage)
    assert (vm.workspace / "outputs" / "videos" / "clip.mp4").read_bytes() == b"mp4 bytes"
    assert h3.parse_marker(capsys.readouterr().out.strip()) == (
        "DRIVE_OUTPUT",
        {"path": "outputs/videos/clip.mp4", "bytes": 9},
    )

    def broken(src, dst, block=0):
        raise OSError("drive went away")

    vm.monkeypatch.setattr(remote, "copy_large", broken)
    remote.save_output(video, storage)  # does not raise
    kind, data = h3.parse_marker(capsys.readouterr().out.strip())
    assert kind == "DRIVE_OUTPUT_FAILED" and "drive went away" in data["message"]


def fake_colab(monkeypatch, flush):
    drive = types.SimpleNamespace(flush_and_unmount=flush)
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = drive
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.drive", drive)


def test_finalize_waits_for_the_copies_then_flushes(vm, capsys):
    order = []
    remote.ensure_models(small_specs(), vm.new_session())
    fake_colab(vm.monkeypatch, lambda: order.append(("flush", len(wait_for_copies()))))
    capsys.readouterr()
    remote.finalize()
    markers = dict(filter(None, (h3.parse_marker(line) for line in capsys.readouterr().out.splitlines())))
    assert order == [("flush", 5)]  # every copy had ended before the flush
    assert len(markers["MODEL_SYNC"]["report"]) == 5 and "seconds" in markers["DRIVE_FLUSHED"]


def test_a_failed_flush_is_reported(vm, capsys):
    def broken():
        raise RuntimeError("mountpoint busy")

    vm.new_session()
    fake_colab(vm.monkeypatch, broken)
    with pytest.raises(RuntimeError):
        remote.finalize()
    markers = dict(filter(None, (h3.parse_marker(line) for line in capsys.readouterr().out.splitlines())))
    assert markers["ERROR"]["code"] == "DRIVE_FLUSH_FAILED" and "DRIVE_FLUSHED" not in markers


# --- the runner ----------------------------------------------------------------------------------------

DRIVE_EXEC_LINES = HAPPY_EXEC_LINES[:5] + [
    marker("MODEL", name=remote.LORA, gib=0.58, seconds=3.0, action="staged", cached=False),
    marker("MODEL", name=remote.VIDEO_VAE, gib=4.85, seconds=40.0, action="downloaded", cached=False),
] + HAPPY_EXEC_LINES[5:] + [marker("DRIVE_OUTPUT", path="outputs/videos/first_h3_job1.mp4", bytes=4194304)]  # fmt: skip
FINALIZE_LINES = [
    marker("MODEL_SYNC", report={remote.VIDEO_VAE: "copied in 60.0 s"}),
    marker("DRIVE_FLUSHED", seconds=8.5),
]


def drive_exec(finalize=emit(*FINALIZE_LINES), job=emit(*DRIVE_EXEC_LINES)):
    def handler(args, on_line):
        return (finalize if "H3_FINALIZE=1" in args else job)(args, on_line)

    return handler


def run(tmp_path, transport, **spec_kw):
    spec_kw.setdefault("description", "Steam rises slowly from the cup.")
    spec_kw.setdefault("image", make_image(tmp_path))
    spec_kw.setdefault("job_id", "job1")
    config = make_config(tmp_path, H3_DRIVE="consent")
    record = h3.run_job(h3.JobSpec(**spec_kw), config, transport=transport, prober=FakeProber(), stream=None)
    return record


def log_text(record):
    return Path(record["log"]).read_text(encoding="utf-8")


def test_with_drive_the_session_mounts_first_and_saves_before_it_stops(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(h3, "_open_in_browser", opened.append)
    seen = {}

    def upload(args, on_line):
        if args[-1].endswith("_job.json"):
            seen["job"] = json.loads(Path(args[-2]).read_text(encoding="utf-8"))
        return "[colab] Uploaded"

    t = FakeTransport(mount=MOUNTED, exec=drive_exec(), upload=upload)
    record = run(tmp_path, t)
    assert record["status"] == h3.COMPLETED, record
    assert t.subcommands == [
        "version", "usage", "new", "usage", "drivemount", "upload", "upload", "exec", "download", "download",
        "exec", "stop", "usage",
    ]  # fmt: skip
    assert t.mount_calls == [{"session": "h3-job1", "mount": "/content/drive", "wait_seconds": 90}]
    storage = seen["job"]["storage"]
    assert storage["persistence"] == "drive" and storage["workspace"] == "/content/drive/MyDrive/AI-Workflow"
    assert storage["drive_output"] == "outputs/videos/first_h3_job1.mp4" and len(storage["models"]) == 6
    finalize = t.calls[10]
    assert finalize[:3] == ["exec", "--session", "h3-job1"] and "H3_FINALIZE=1" in finalize
    assert record["persistence"] == "drive" and record["drive_output"] == "outputs/videos/first_h3_job1.mp4"
    assert record["model_actions"] == {remote.LORA: "staged", remote.VIDEO_VAE: "downloaded"}
    assert record["drive_saved"]["flushed"] is True and record["drive_saved"]["model_sync"] == {
        remote.VIDEO_VAE: "copied in 60.0 s"
    }
    assert not [w for w in record["warnings"] if "Drive" in w]
    assert (tmp_path / "out" / "first_h3_job1.mp4").is_file()  # the local copy is still downloaded and validated
    assert len(opened) == 1 and opened[0].startswith("https://accounts.google.com/")
    text = log_text(record)
    assert "DRIVE_CONSENT_NEEDED" in text and "consent_done.py" in text
    assert "accounts.google.com" not in text  # the consent URL is never written to the log
    entry = json.loads((tmp_path / "out" / "h3_jobs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert entry["persistence"] == "drive" and entry["drive_output"] and set(h3.LEDGER_FIELDS) == set(entry)


def test_without_consent_the_job_still_runs_and_says_nothing_was_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(h3, "_open_in_browser", lambda url: None)
    seen = {}

    def upload(args, on_line):
        if args[-1].endswith("_job.json"):
            seen["job"] = json.loads(Path(args[-2]).read_text(encoding="utf-8"))
        return ""

    t = FakeTransport(mount=NOT_MOUNTED, upload=upload)
    record = run(tmp_path, t)
    assert record["status"] == h3.COMPLETED and record["persistence"] == "ephemeral"
    assert seen["job"]["storage"]["persistence"] == "ephemeral"
    assert t.subcommands.count("exec") == 1  # no save step: there is nothing mounted to save to
    assert "This session is ephemeral" in log_text(record) and "consent_not_given" in log_text(record)


def test_a_mount_that_cannot_even_start_is_ephemeral_not_fatal(tmp_path):
    t = FakeTransport(mount=OSError("docker: no such container"))
    record = run(tmp_path, t)
    assert record["status"] == h3.COMPLETED and record["persistence"] == "ephemeral" and "stop" in t.subcommands


@pytest.mark.parametrize(
    "finalize,fragment",
    [
        (raising(h3.ColabCommandError("colab exec", 1, ["RuntimeError: Connection was lost."])), "Connection was lost"),
        (raising(h3.ColabTimeout("colab exec (save to Drive) printed nothing for 900 s")), "printed nothing"),
        (emit(marker("MODEL_SYNC", report={"a.safetensors": "failed: quota"}), marker("DRIVE_FLUSHED", seconds=1.0)), "a.safetensors"),
        (emit(marker("ERROR", code="DRIVE_FLUSH_FAILED", message="mountpoint busy")), "DRIVE_FLUSH_FAILED"),
        (emit(), "flushed: False"),
    ],
)  # fmt: skip
def test_a_save_that_fails_never_keeps_the_gpu_running(tmp_path, monkeypatch, finalize, fragment):
    monkeypatch.setattr(h3, "_open_in_browser", lambda url: None)
    t = FakeTransport(mount=MOUNTED, exec=drive_exec(finalize=finalize))
    record = run(tmp_path, t)
    assert record["status"] == h3.COMPLETED  # the clip itself is fine
    assert t.subcommands[-2:] == ["stop", "usage"] and record["session_status"] == "stopped"
    warning = next(w for w in record["warnings"] if "Google Drive may not hold everything" in w)
    assert fragment in warning


def test_a_session_whose_kernel_is_stuck_is_stopped_without_the_save_step(tmp_path, monkeypatch):
    monkeypatch.setattr(h3, "_open_in_browser", lambda url: None)
    t = FakeTransport(mount=MOUNTED, exec=raising(h3.ColabTimeout("colab exec printed nothing for 900 s")))
    record = run(tmp_path, t)
    assert record["status"] == h3.TIMEOUT
    assert t.subcommands.count("exec") == 1 and "stop" in t.subcommands  # no second exec queued behind a hung one
    assert any("save-to-Drive step was skipped" in w for w in record["warnings"])


def test_a_batch_mounts_once_and_saves_once(tmp_path, monkeypatch):
    monkeypatch.setattr(h3, "_open_in_browser", lambda url: None)
    t = FakeTransport(mount=MOUNTED, exec=drive_exec())
    specs = [
        h3.JobSpec(image=make_image(tmp_path, f"img{i}.png"), description="Steam rises.", job_id=f"j{i}")
        for i in (1, 2, 3)
    ]
    config = make_config(tmp_path, H3_DRIVE="consent")
    summary = h3.run_batch(specs, config, transport=t, prober=FakeProber(), stream=None, batch_id="b1")
    assert summary["completed"] == 3
    assert t.subcommands.count("drivemount") == 1 and t.subcommands.count("exec") == 4  # 3 clips + 1 save
    assert t.subcommands.index("stop") > max(i for i, c in enumerate(t.subcommands) if c == "exec")
    assert [r["persistence"] for r in summary["records"]] == ["drive"] * 3
    assert all(r["drive_saved"]["flushed"] for r in summary["records"])


def test_drive_off_never_calls_drivemount(tmp_path):
    t = FakeTransport()  # mount=None: mount_drive would assert
    record = h3.run_job(
        h3.JobSpec(image=make_image(tmp_path), description="Steam rises.", job_id="job1"),
        make_config(tmp_path),
        transport=t,
        prober=FakeProber(),
        stream=None,
    )
    assert record["status"] == h3.COMPLETED and record["persistence"] == "ephemeral"
    assert "drivemount" not in t.subcommands and "drive.mode is off" in log_text(record)


def test_the_dry_run_lists_the_mount_and_the_save_step(tmp_path):
    config = make_config(tmp_path, H3_DRIVE="consent")
    spec = h3.JobSpec(image=make_image(tmp_path), description="Steam rises.", job_id="job1")
    record = h3.run_job(spec, config, transport=FakeTransport(), prober=FakeProber(), stream=None, dry_run=True)
    commands = record["dry_run"]["commands"]
    assert any("drivemount --session h3-job1 /content/drive" in c for c in commands)
    assert "H3_FINALIZE=1" in commands[-1] and "H3_JOB_FILE" in commands[-2]


# --- consent signal, config, command lines -------------------------------------------------------------


def test_consent_done_writes_the_signal_the_runner_waits_for(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("H3_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(h3, "LOCAL_CONFIG", tmp_path / "no-config.json")
    assert import_skill("consent_done").main([]) == 0
    out = json.loads(capsys.readouterr().out)
    config = make_config(tmp_path)
    assert out["ok"] and Path(out["signal"]) == h3.consent_signal_path(config) and Path(out["signal"]).is_file()


def test_the_default_is_to_store_on_drive_in_the_platforms_workspace():
    config = json.loads((SKILL_DIR / "config" / "config.example.json").read_text(encoding="utf-8"))
    assert config["drive"]["mode"] == "consent" and config["drive"]["workspace"] == "MyDrive/AI-Workflow"
    assert config["drive"]["consent_wait_seconds"] < 120  # drive.mount gives up at 120 s


@pytest.mark.parametrize(
    "edit,fragment",
    [
        (lambda c: c["drive"].update(mode="always"), "drive.mode"),
        (lambda c: c["drive"].update(consent_wait_seconds=130), "consent_wait_seconds"),
        (lambda c: c["drive"].update(workspace="../elsewhere"), "drive.workspace"),
        (lambda c: c["drive"].update(mount="/mnt/drive"), "drive.mount"),
        (lambda c: c["drive"].update(finalize_timeout_seconds=5), "finalize_timeout_seconds"),
        (lambda c: c["models"][0].update(revision="main"), "pinned revision"),
        (lambda c: c["models"][0].update(sha256="abc"), "sha256"),
        (lambda c: c["models"][0].update(size=0), "size"),
        (lambda c: c.update(models=c["models"][:2]), "models must list"),
        (lambda c: c["models"].append(dict(c["models"][0])), "appears twice"),
    ],
)
def test_invalid_drive_and_model_settings_are_refused(tmp_path, edit, fragment):
    config = make_config(tmp_path)
    edit(config)
    with pytest.raises(h3.InputError) as exc:
        h3.validate_config(config)
    assert exc.value.code == "INVALID_CONFIG" and fragment in str(exc.value)


def test_the_interactive_docker_command_gives_the_cli_a_terminal():
    t = h3.DockerTransport("h3-colab-cli:0.7.4", "h3-colab-config", "oauth2")
    cmd = t.interactive_command(["drivemount", "--session", "s1", "/content/drive"], name="h3-x")
    assert (
        cmd[:3] == ["docker", "run", "-i"] and "--entrypoint" in cmd and cmd[cmd.index("--entrypoint") + 1] == "script"
    )
    assert cmd[-3:] == ["-qfec", "colab --auth=oauth2 drivemount --session s1 /content/drive", "/dev/null"]
    assert "h3-colab-config:/root/.config/colab-cli" in cmd
