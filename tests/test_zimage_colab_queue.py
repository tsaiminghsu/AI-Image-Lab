"""The local job queue (output/zimage/jobs/*.json): creating jobs, the state machine, cancelling, and
the submit CLI that turns one request into N jobs for ONE worker session."""

import json

import pytest
from zimage_colab_fakes import import_skill, make_config, z

submit = import_skill("submit")


def store_with(tmp_path, n=1, **kw):
    config = make_config(tmp_path)
    store = z.JobStore(z.output_root(config))
    jobs = [store.add(z.build_job(config, prompt=f"a ceramic mug, view {i}", **kw)) for i in range(n)]
    return config, store, jobs


def test_a_new_job_has_the_documented_fields(tmp_path):
    _, store, (job,) = store_with(tmp_path)
    saved = store.get(job["job_id"])
    for key in (
        "job_id",
        "workflow",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "steps",
        "seed",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "output",
    ):
        assert key in saved
    assert saved["status"] == z.PENDING and saved["workflow"] == "z-image-txt2img"
    assert saved["history"] == [{"status": z.PENDING, "at": saved["history"][0]["at"]}]
    assert saved["attempt"] == 0 and saved["output"] is None


def test_jobs_are_listed_oldest_first_and_filtered_by_status(tmp_path):
    _, store, jobs = store_with(tmp_path, 3)
    for i, job in enumerate(jobs):  # same-second created_at falls back to the id: make the order explicit
        store.update(job["job_id"], created_at=f"2026-10-02T10:00:0{i}+0800")
    store.update(jobs[1]["job_id"], status=z.QUEUED)
    assert [j["job_id"] for j in store.list()] == [j["job_id"] for j in jobs]
    assert [j["job_id"] for j in store.list((z.PENDING,))] == [jobs[0]["job_id"], jobs[2]["job_id"]]
    assert store.list((z.COMPLETED,)) == []


def test_a_duplicate_job_id_is_refused(tmp_path):
    config, store, (job,) = store_with(tmp_path)
    with pytest.raises(z.UsageError, match="already exists"):
        store.add(z.build_job(config, prompt="another", job_id=job["job_id"]))


@pytest.mark.parametrize("job_id", ["../escape", "a/b", "", "x" * 81, "job id"])
def test_a_job_id_cannot_leave_the_jobs_directory(tmp_path, job_id):
    store = z.JobStore(tmp_path)
    with pytest.raises(z.UsageError):
        store.path(job_id)


def test_an_unknown_job_is_a_usage_error(tmp_path):
    with pytest.raises(z.UsageError, match="no job"):
        z.JobStore(tmp_path).get("zimg-nope")


def test_the_lifecycle_runs_forward_and_records_its_history(tmp_path):
    _, store, (job,) = store_with(tmp_path)
    for status in (z.QUEUED, z.RUNNING, z.GENERATED, z.VALIDATING, z.VALID, z.COMPLETED):
        store.update(job["job_id"], status=status)
    assert [h["status"] for h in store.get(job["job_id"])["history"]] == [
        z.PENDING, z.QUEUED, z.RUNNING, z.GENERATED, z.VALIDATING, z.VALID, z.COMPLETED
    ]  # fmt: skip


@pytest.mark.parametrize(
    "old,new,allowed",
    [
        (z.PENDING, z.QUEUED, True),
        (z.PENDING, z.RUNNING, True),  # the controller may miss a state between two polls
        (z.QUEUED, z.VALID, True),
        (z.RUNNING, z.QUEUED, False),  # never backwards...
        (z.VALID, z.VALIDATING, False),
        (z.RUNNING, z.PENDING, True),  # ...except back to the queue when its session was lost
        (z.QUEUED, z.PENDING, True),
        (z.PENDING, z.PENDING, False),
        (z.PENDING, z.CANCELLED, True),
        (z.RUNNING, z.FAILED, True),
        (z.RUNNING, z.TIMEOUT, True),
        (z.VALIDATING, z.INVALID_OUTPUT, True),
        (z.COMPLETED, z.PENDING, False),  # terminal states are final
        (z.FAILED, z.RUNNING, False),
        (z.CANCELLED, z.COMPLETED, False),
        (z.TIMEOUT, z.PENDING, False),
        (z.PENDING, "NONSENSE", False),
    ],
)
def test_transition_table(old, new, allowed):
    assert z.transition_allowed(old, new) is allowed


def test_a_forbidden_transition_changes_nothing(tmp_path):
    _, store, (job,) = store_with(tmp_path)
    store.update(job["job_id"], status=z.COMPLETED)
    with pytest.raises(z.ZImageError) as err:
        store.update(job["job_id"], status=z.RUNNING, output="x.png")
    assert err.value.code == "BAD_TRANSITION"
    saved = store.get(job["job_id"])
    assert saved["status"] == z.COMPLETED and saved["output"] is None


def test_cancelling_a_pending_job_is_immediate(tmp_path):
    _, store, (job,) = store_with(tmp_path)
    assert store.request_cancel(job["job_id"]) == z.CANCELLED
    assert store.get(job["job_id"])["status"] == z.CANCELLED
    assert not store.cancel_dir.exists()


def test_cancelling_a_job_a_session_has_leaves_a_request_for_the_controller(tmp_path):
    _, store, (job,) = store_with(tmp_path)
    store.update(job["job_id"], status=z.RUNNING)
    assert store.request_cancel(job["job_id"]) == "CANCEL_REQUESTED"
    assert (store.cancel_dir / job["job_id"]).is_file()
    assert store.get(job["job_id"])["status"] == z.RUNNING  # the worker decides; the controller reports back


def test_cancelling_a_finished_job_does_nothing(tmp_path):
    _, store, (job,) = store_with(tmp_path)
    store.update(job["job_id"], status=z.COMPLETED)
    assert store.request_cancel(job["job_id"]) == z.COMPLETED


def test_a_stale_lock_from_a_dead_process_is_broken(tmp_path):
    import os
    import time

    lock = tmp_path / "jobs" / ".lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("99999999")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    with z.FileLock(lock, timeout=1, stale=60):
        assert lock.exists()
    assert not lock.exists()


def test_a_held_lock_times_out_with_its_own_code(tmp_path):
    lock = tmp_path / ".lock"
    with z.FileLock(lock), pytest.raises(z.ZImageError) as err, z.FileLock(lock, timeout=0.2):
        pass
    assert err.value.code == "STORE_LOCKED"


# --- submit.py: one request, N jobs, no Colab call -----------------------------------------------


def run_submit(tmp_path, monkeypatch, capsys, *argv):
    monkeypatch.setenv("ZIMG_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(z, "LOCAL_CONFIG", tmp_path / "no-config.json")
    code = submit.main(list(argv))
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_ten_images_are_ten_jobs_in_one_queue(tmp_path, monkeypatch, capsys):
    code, out = run_submit(tmp_path, monkeypatch, capsys, "--prompt", "a product photo", "--count", "10")
    assert code == 0 and out["ok"] and len(out["jobs"]) == 10
    store = z.JobStore(tmp_path / "out")
    assert len(store.list((z.PENDING,))) == 10
    assert len({j["job_id"] for j in out["jobs"]}) == 10
    assert out["controller_running"] is False and "worker.py up" in out["next"]


def test_a_fixed_seed_with_count_gives_consecutive_seeds(tmp_path, monkeypatch, capsys):
    _, out = run_submit(tmp_path, monkeypatch, capsys, "--prompt", "a mug", "--count", "3", "--seed", "100")
    assert [j["seed"] for j in out["jobs"]] == [100, 101, 102]


def test_a_random_seed_is_recorded_per_job(tmp_path, monkeypatch, capsys):
    _, out = run_submit(tmp_path, monkeypatch, capsys, "--prompt", "a mug", "--count", "4")
    assert all(isinstance(j["seed"], int) and j["seed"] >= 0 for j in out["jobs"])
    saved = z.JobStore(tmp_path / "out").get(out["jobs"][0]["job_id"])
    assert saved["seed"] == out["jobs"][0]["seed"] and saved["seed_requested"] == -1


def test_dry_run_stores_nothing(tmp_path, monkeypatch, capsys):
    code, out = run_submit(tmp_path, monkeypatch, capsys, "--prompt", "a mug", "--dry-run")
    assert code == 0 and out["dry_run"] and len(out["jobs"]) == 1
    assert z.JobStore(tmp_path / "out").list() == []


def test_a_manifest_with_one_bad_entry_queues_nothing(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps({"batch": "mugs", "jobs": [{"prompt": "a mug", "aspect": "16:9"}, {"prompt": "a mug", "cfg": 1.0}]}),
        encoding="utf-8",
    )
    code, out = run_submit(tmp_path, monkeypatch, capsys, "--manifest", str(manifest))
    assert code == 2 and out["ok"] is False
    assert z.JobStore(tmp_path / "out").list() == []


def test_a_manifest_queues_every_job_under_its_batch_name(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps({"batch": "mugs", "jobs": [{"prompt": "a mug", "aspect": "16:9"}, {"prompt": "a cup", "count": 2}]}),
        encoding="utf-8",
    )
    code, out = run_submit(tmp_path, monkeypatch, capsys, "--manifest", str(manifest))
    assert code == 0 and len(out["jobs"]) == 3
    assert {j["batch"] for j in z.JobStore(tmp_path / "out").list()} == {"mugs"}


def test_a_manifest_with_an_unknown_key_is_refused(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps({"jobs": [{"prompt": "a mug", "sampler": "euler"}]}), encoding="utf-8")
    code, out = run_submit(tmp_path, monkeypatch, capsys, "--manifest", str(manifest))
    assert code == 2 and "unknown keys" in out["error"]["message"]
