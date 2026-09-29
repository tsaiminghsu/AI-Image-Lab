"""FileJobStore's own mechanics: the lockfile, atomic replace, the temp sweeper, and the two
orderings that decide whether a crash is recoverable.

Engine-agnostic behaviour lives in test_job_store_contract.py. What is here is everything that
would be different, or absent, in a DynamoDB engine - and everything whose failure mode is a
corrupted or unroutable record rather than a wrong answer.

Every test that involves waiting injects its clock and sleep, so the five-second lock timeout
and the thirty-second staleness window cost no wall time at all.
"""

import json
import os

import job_contracts as jc
import job_store as js
import pytest
from test_job_store_contract import make_record


class FakeClock:
    """Advances one second per reading. Enough to walk any timeout in a handful of calls."""

    def __init__(self, start=1000.0, step=1.0):
        self.now = start
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def make_store(tmp_path, clock=None):
    return js.FileJobStore(str(tmp_path / "jobs"), clock=clock or FakeClock(), sleep=lambda _seconds: None)


def write_lock(store, acquired_at):
    with open(store._lock_path, "w", encoding="utf-8") as handle:
        json.dump({"pid": 999999, "host": "elsewhere", "acquired_at": acquired_at}, handle)


# --- the lockfile ---------------------------------------------------------------------------------


def test_a_lock_held_by_a_live_process_is_respected(tmp_path):
    """acquired_at far in the future can never look stale, so this stands in for a holder that is
    genuinely still working."""
    store = make_store(tmp_path)
    write_lock(store, acquired_at=10**9)
    with pytest.raises(js.JobStoreLocked, match="could not acquire"):
        store.create("local", make_record())


def test_a_stale_lock_is_broken_and_logged(tmp_path):
    """A process that died holding the lock must not wedge the store forever. The break is logged
    because it is evidence something crashed, not routine housekeeping."""
    store = make_store(tmp_path)
    write_lock(store, acquired_at=0.0)
    store.create("local", make_record())
    assert store.get("local", "job1") is not None
    assert "broke stale lock" in open(store.reconciliation_log, encoding="utf-8").read()


def test_a_lock_with_an_unreadable_body_is_treated_as_stale(tmp_path):
    """A writer that died between creating the lockfile and describing itself would otherwise
    wedge the store permanently - exactly the case where being conservative is the wrong call."""
    store = make_store(tmp_path)
    with open(store._lock_path, "wb") as handle:
        handle.write(b"{ truncated")
    store.create("local", make_record())
    assert store.get("local", "job1") is not None


def test_the_lock_is_released_even_when_the_mutation_raises(tmp_path):
    store = make_store(tmp_path)
    store.create("local", make_record())

    def explode(_record):
        raise RuntimeError("mutation blew up")

    with pytest.raises(RuntimeError):
        store.update("local", "job1", explode)
    assert not os.path.exists(store._lock_path), "a raised mutation must not leave the store locked"
    store.update("local", "job1", lambda r: None)


# --- atomic replace -------------------------------------------------------------------------------


def test_the_temp_file_is_written_beside_its_target(tmp_path):
    """os.replace is only atomic within one volume, and same-directory is the only way to
    guarantee that on Windows without inspecting mount points."""
    store = make_store(tmp_path)
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append((os.path.dirname(src), os.path.dirname(dst)))
        return real_replace(src, dst)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", spy)
        store.create("local", make_record())
    assert seen and all(src == dst for src, dst in seen)


def test_a_sharing_violation_is_retried_rather_than_lost(tmp_path):
    """A concurrent reader on Windows makes os.replace raise on the *writer*. Dropping the write
    would silently lose a state transition, so it retries a bounded number of times."""
    store = make_store(tmp_path)
    real_replace = os.replace
    failures = {"left": 3}

    def flaky(src, dst):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError("simulated sharing violation")
        return real_replace(src, dst)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", flaky)
        store.create("local", make_record())
    assert store.get("local", "job1") is not None
    assert failures["left"] == 0


def test_a_permanent_sharing_violation_raises_and_cleans_up(tmp_path):
    """Loud failure, not a silent skip - and no orphaned temp left behind for the sweeper."""
    store = make_store(tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(PermissionError("always")))
        with pytest.raises(PermissionError):
            store.create("local", make_record())
    assert not [n for n in os.listdir(os.path.join(store.root, "jobs")) if n.endswith(".tmp")]


# --- a reader never sees a half-written record ----------------------------------------------------


def test_an_orphaned_temp_does_not_disturb_a_reader(tmp_path):
    """A process killed mid-write leaves a .tmp. Because the record itself is only ever swapped in
    whole, the reader keeps seeing the previous version rather than parsing wreckage."""
    store = make_store(tmp_path)
    store.create("local", make_record())
    orphan = os.path.join(store.root, "jobs", "job1.json.123.1.tmp")
    with open(orphan, "w", encoding="utf-8") as handle:
        handle.write('{"truncat')
    assert store.get("local", "job1")["params"]["seed"] == 9000


def test_old_orphaned_temps_are_swept(tmp_path):
    store = make_store(tmp_path)
    store.create("local", make_record("a"))
    orphan = os.path.join(store.root, "jobs", "a.json.123.1.tmp")
    with open(orphan, "w", encoding="utf-8") as handle:
        handle.write("x")
    ancient = os.path.getmtime(orphan) - js.ORPHAN_TEMP_SECONDS - 10
    os.utime(orphan, (ancient, ancient))
    store.create("local", make_record("b"))
    assert not os.path.exists(orphan)


def test_a_fresh_temp_is_left_alone(tmp_path):
    """It may belong to a writer that is still running; deleting it would be the bug, not the fix."""
    store = make_store(tmp_path)
    store.create("local", make_record("a"))
    orphan = os.path.join(store.root, "jobs", "a.json.999.1.tmp")
    with open(orphan, "w", encoding="utf-8") as handle:
        handle.write("x")
    store.create("local", make_record("b"))
    assert os.path.exists(orphan)


def test_a_corrupt_record_reads_as_absent_rather_than_crashing(tmp_path):
    """Whatever produced it, one unparseable file must not take down a reconciler sweeping every
    other job in the store."""
    store = make_store(tmp_path)
    store.create("local", make_record())
    with open(store._job_path("job1"), "w", encoding="utf-8") as handle:
        handle.write("{ not json")
    assert store.get("local", "job1") is None
    assert store.list_unterminal() == []


# --- the reverse index ----------------------------------------------------------------------------


def test_an_index_prefix_collision_is_detected_not_mis_routed(tmp_path):
    """The index filename is a truncated hash, so a collision is possible in principle. The literal
    provider id in the body is what turns it into a miss instead of delivering one provider's
    webhook to a different customer's job."""
    store = make_store(tmp_path)
    store.create("local", make_record(status=jc.SUBMITTING))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(js, "_digest", lambda value, length: "collision" * 4)
        store.bind_provider_job_id("local", "job1", "runpod", "rp-real")
        assert store.find_by_provider_job_id("runpod", "rp-real") == ("local", "job1")
        assert store.find_by_provider_job_id("runpod", "rp-different") is None
    assert "index prefix collision" in open(store.reconciliation_log, encoding="utf-8").read()


def test_a_crash_between_the_index_and_the_record_stays_routable(tmp_path):
    """The index is written first on purpose. This is what that ordering buys: the webhook can
    still find its job, and its own compare-and-swap fills in what the crash lost. The reverse
    order would produce a provider job nothing can ever be routed to."""
    store = make_store(tmp_path)
    store.create("local", make_record(status=jc.SUBMITTING))
    entry = {"owner_id": "local", "job_id": "job1", "provider_job_id": "rp-orphan"}
    index_path = store._index_path("runpod", "rp-orphan")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(entry, handle)

    assert store.find_by_provider_job_id("runpod", "rp-orphan") == ("local", "job1")
    assert store.get("local", "job1")["provider_job_id"] is None, "the record is the half that was lost"
    store.bind_provider_job_id("local", "job1", "runpod", "rp-orphan")
    assert store.get("local", "job1")["provider_job_id"] == "rp-orphan"


# --- artifact resolution is tagged ----------------------------------------------------------------


def test_a_local_artifact_resolves_to_a_path(tmp_path):
    """Not a fake signed URL. image_api's caller depends on image_path being a real path, and a
    URL shape with an expiry field would be a lie about something that never expires."""
    resolved = js.resolve_artifact(
        "local",
        {
            "bucket": js.LOCAL_BUCKET,
            "key": "generated/job1/abc.png",
            "sha256": "a" * 64,
            "media_kind": "image",
            "content_type": "image/png",
            "byte_length": 1,
        },
        local_root=str(tmp_path),
    )
    assert resolved["kind"] == "path"
    assert resolved["path"] == os.path.join(str(tmp_path), "generated", "job1", "abc.png")


def test_a_remote_artifact_resolves_to_a_grant_request(tmp_path):
    resolved = js.resolve_artifact(
        "local",
        {
            "bucket": "urmine-media",
            "key": "generated/job1/abc.png",
            "sha256": "a" * 64,
            "media_kind": "image",
            "content_type": "image/png",
            "byte_length": 1,
        },
    )
    assert resolved["kind"] == "remote"
    assert resolved["bucket"] == "urmine-media"


def test_resolution_revalidates_the_key_before_joining_it_to_a_path():
    """The key is about to be joined onto a real directory. Whatever validated it on the way in,
    this is the last moment before it becomes a filesystem operation."""
    with pytest.raises(jc.ContractError, match="unsafe artifact key"):
        js.resolve_artifact(
            "local",
            {
                "bucket": js.LOCAL_BUCKET,
                "key": "../../etc/passwd",
                "sha256": "a" * 64,
                "media_kind": "image",
                "content_type": "image/png",
                "byte_length": 1,
            },
        )
