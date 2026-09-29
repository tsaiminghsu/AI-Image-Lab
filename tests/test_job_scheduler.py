"""job_scheduler: deferred jobs on the real FileJobStore, a fake clock and a fake machine.

The generator is replaced with one that writes a real PNG (the output check has to be able to read
it back, same reason as test_image_api), so a job that "ran" has genuinely gone
queued -> submitting -> checking -> completed through job_service.
"""

import datetime
import os

import pytest
from test_output_check import make_mp4, make_png

import job_contracts as jc
import job_scheduler as sched
import job_service
import job_store
import resource_policy as rp

NOON = datetime.datetime(2026, 9, 27, 12, 0)
T0 = NOON.timestamp()


def calm(**overrides):
    base = dict(
        gpu_temp_c=50.0,
        gpu_throttling=False,
        vram_used_gb=2.0,
        vram_total_gb=8.0,
        ram_available_gb=20.0,
        comfy_running=True,
        comfy_rss_gb=4.0,
        comfy_queue=0,
        local_busy_seconds=0.0,
    )
    base.update(overrides)
    return rp.Snapshot(**base)


class Machine:
    def __init__(self):
        self.snap = calm()
        self.settings = dict(rp.DEFAULT_SETTINGS)
        self.t = T0

    def clock(self):
        return self.t

    def now(self):
        return datetime.datetime.fromtimestamp(self.t)


@pytest.fixture
def env(tmp_path):
    machine = Machine()
    store = job_store.FileJobStore(str(tmp_path / "jobs"), clock=machine.clock)
    service = job_service.JobService(store, clock=machine.clock, artifact_root=str(tmp_path / "outputs"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    calls = []

    def fake_call(name):
        def generate(**kwargs):
            calls.append((name, kwargs))
            video = name == "gen_video_animatediff"
            path = os.path.join(str(out_dir), f"{name}_{kwargs['seed']}.{'mp4' if video else 'png'}")
            with open(path, "wb") as handle:
                handle.write(make_mp4(32, 32) if video else make_png(32, 32))
            return path

        return generate

    scheduler = sched.Scheduler(
        service,
        settings_fn=lambda: machine.settings,
        probe_fn=lambda: machine.snap,
        clock=machine.clock,
        now_fn=machine.now,
        resolve_call=fake_call,
        log=lambda *_: None,
    )
    return machine, service, scheduler, calls, tmp_path


def defer(
    service,
    machine,
    *,
    kind="animatediff",
    call="gen_video_animatediff",
    not_before,
    force_local=False,
    kwargs=None,
    input_keys=(),
):
    decision = rp.Decision(
        rp.DEFER, kind, ["現在不在影片時段"], est_seconds=rp.JOB_COSTS[kind].seconds, not_before=not_before
    )
    return sched.enqueue(
        service,
        "gui",
        call=call,
        kwargs=kwargs or {"prompt": "walking", "seed": 7},
        mode="video_animatediff",
        media_kind="video",
        catalog_id="realistic_vision:video_animatediff",
        params={"prompt": "walking", "seed": 7},
        decision=decision,
        input_keys=input_keys,
        force_local=force_local,
    )


def status(service, job_id):
    return service.store.get("gui", job_id)


def test_enqueue_records_a_runnable_spec_and_copies_uploads(env):
    machine, service, _, _, tmp_path = env
    upload = tmp_path / "gradio_tmp_face.PNG"
    upload.write_bytes(make_png(8, 8))
    record = defer(
        service,
        machine,
        not_before=T0 + 3600,
        kwargs={"prompt": "p", "seed": 1, "face_ref_path": str(upload)},
        input_keys=("face_ref_path",),
    )
    assert record["status"] == jc.QUEUED
    assert record["not_before"] == T0 + 3600
    assert record["route"]["kind"] == "animatediff"
    copied = record["request"]["kwargs"]["face_ref_path"]
    assert copied != str(upload) and os.path.isfile(copied) and copied.endswith(".png")
    upload.unlink()  # the temp upload vanishing must not matter any more
    assert os.path.isfile(copied)


def test_enqueue_refuses_arguments_that_cannot_survive_a_restart(env):
    machine, service, _, _, _ = env
    with pytest.raises(ValueError):
        defer(service, machine, not_before=T0, kwargs={"prompt": "p", "seed": 1, "cb": object()})


def test_nothing_runs_before_its_time(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, not_before=T0 + 60)
    assert scheduler.tick() == []
    assert calls == []
    assert status(service, record["job_id"])["status"] == jc.QUEUED


def test_a_due_job_on_a_calm_machine_runs_to_completion(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, not_before=T0 + 60)
    machine.t += 61
    assert scheduler.tick() == [(record["job_id"], "ran")]
    assert calls[0][0] == "gen_video_animatediff"
    done = status(service, record["job_id"])
    assert done["status"] == jc.COMPLETED  # output path came from the generator's return value
    assert done["artifacts"][0]["media_kind"] == "video"


def test_at_most_one_job_per_tick_earliest_first(env):
    machine, service, scheduler, calls, _ = env
    later = defer(service, machine, not_before=T0 + 20)
    first = defer(service, machine, not_before=T0 + 10)
    machine.t += 30
    assert scheduler.tick() == [(first["job_id"], "ran")]
    assert status(service, later["job_id"])["status"] == jc.QUEUED
    assert scheduler.tick() == [(later["job_id"], "ran")]


def test_a_local_job_already_on_the_card_holds_the_scheduler(env):
    machine, service, scheduler, calls, _ = env
    defer(service, machine, not_before=T0)
    with rp.track("txt2img"):
        assert scheduler.tick() == [("*", "busy")]
    assert calls == []


def test_a_hot_card_waits_then_runs_anyway_after_the_cooldown_timeout(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, not_before=T0)
    machine.snap = calm(gpu_temp_c=83.0)
    assert scheduler.tick() == [(record["job_id"], rp.WAIT)]
    assert status(service, record["job_id"])["route"]["route"] == rp.WAIT
    machine.t += machine.settings["cooldown_timeout_s"]
    assert scheduler.tick() == [(record["job_id"], "ran")]


def test_a_cloud_suggestion_at_night_leaves_the_job_waiting_for_the_operator(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, kind="txt2img_hq_full", call="gen_custom", not_before=T0)
    machine.snap = calm(ram_available_gb=6.0, comfy_rss_gb=0.0)
    assert scheduler.tick() == [(record["job_id"], rp.CLOUD)]
    current = status(service, record["job_id"])
    assert current["status"] == jc.QUEUED
    assert current["route"]["route"] == rp.CLOUD
    assert calls == []


def test_a_changed_window_pushes_the_job_to_the_new_start(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, not_before=T0)
    machine.settings = {**machine.settings, "video_windows": ["23:00-08:00"]}
    assert scheduler.tick() == [(record["job_id"], rp.DEFER)]
    assert status(service, record["job_id"])["not_before"] == datetime.datetime(2026, 9, 27, 23, 0).timestamp()


def test_unchanged_waiting_does_not_rewrite_the_record_every_tick(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, not_before=T0)
    machine.snap = calm(gpu_temp_c=83.0)
    scheduler.tick()
    first = status(service, record["job_id"])["updated_at"]
    machine.t += 30
    machine.snap = calm(gpu_temp_c=82.0)  # same decision, different live numbers
    scheduler.tick()
    assert status(service, record["job_id"])["updated_at"] == first


def test_run_now_ignores_the_window(env):
    machine, service, scheduler, calls, _ = env
    machine.settings = {**machine.settings, "video_windows": ["23:00-08:00"]}
    record = defer(service, machine, not_before=T0 + 11 * 3600)
    scheduler.run_now("gui", record["job_id"])
    assert scheduler.tick() == [(record["job_id"], "ran")]


def test_a_job_another_process_already_took_is_skipped(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, not_before=T0)
    service._begin_submitting("gui", record["job_id"])  # the other scheduler won the swap
    assert scheduler.deferred() == []
    stale = dict(record)
    decision = rp.Decision(rp.LOCAL, "animatediff", [])
    assert scheduler._run("gui", stale, decision) == "taken"
    assert calls == []


def test_a_failing_generator_is_recorded_and_the_scheduler_survives(env):
    machine, service, scheduler, calls, _ = env
    record = defer(service, machine, not_before=T0)

    def boom(name):
        def generate(**kwargs):
            raise RuntimeError("ComfyUI fell over")

        return generate

    scheduler._resolve_call = boom
    assert scheduler.tick() == [(record["job_id"], "failed")]
    failed = status(service, record["job_id"])
    assert failed["status"] == jc.FAILED
    assert "fell over" in failed["last_error"]


def test_a_record_without_a_call_spec_is_failed_not_retried_forever(env):
    machine, service, scheduler, calls, _ = env
    record = service.create(
        "gui",
        mode="txt2img",
        catalog_id="x:txt2img",
        params={"seed": 1},
        not_before=T0,
        route={"route": rp.DEFER, "kind": "txt2img"},
    )
    assert scheduler.tick() == [(record["job_id"], "unschedulable")]
    assert status(service, record["job_id"])["status"] == jc.FAILED


def test_adopt_orphans_takes_only_old_runnable_never_deferred_jobs(env):
    machine, service, _, _, _ = env
    spec = sched.call_spec("gen_custom", {"prompt": "p", "seed": 1}, output_path="x.png")
    old = service.create("local", mode="txt2img", catalog_id="c", params={"seed": 1}, request=spec)
    no_spec = service.create("local", mode="txt2img", catalog_id="c", params={"seed": 2}, request={"hq": True})
    machine.t += 100
    started = machine.t
    machine.t += 1
    fresh = service.create("local", mode="txt2img", catalog_id="c", params={"seed": 3}, request=spec)
    adopted = sched.adopt_orphans(service.store, before=started, clock=machine.clock)
    assert adopted == [old["job_id"]]
    assert service.store.get("local", old["job_id"])["not_before"] == machine.t
    assert service.store.get("local", no_spec["job_id"])["not_before"] is None
    assert service.store.get("local", fresh["job_id"])["not_before"] is None


def test_submit_local_without_any_output_path_fails_honestly(env):
    machine, service, _, _, _ = env
    record = service.create("gui", mode="m", catalog_id="c", params={})
    result = service.submit_local("gui", record["job_id"], lambda: None, output_path=None)
    assert result["status"] == jc.FAILED
    assert result["last_error_kind"] == "output_missing"


def test_records_carry_not_before_and_route_and_validate_them():
    record = jc.new_record(
        job_id="j", owner_id="o", mode="m", media_kind="image", catalog_id="c", backend="local", params={}, now=1.0
    )
    assert record["not_before"] is None and record["route"] is None
    assert record["schema_version"] == 2
    with pytest.raises(jc.ContractError):
        jc.validate_record({**record, "not_before": "tonight"})
    with pytest.raises(jc.ContractError):
        jc.validate_record({**record, "route": {"reasons": []}})
    v1 = {k: v for k, v in record.items() if k not in ("not_before", "route")}
    v1["schema_version"] = 1
    assert jc.validate_record(v1)["job_id"] == "j"  # an old record still loads
