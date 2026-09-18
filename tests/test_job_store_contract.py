"""The behaviour every JobStore engine must have, whatever it stores records in.

This file is parameterised over implementations and currently registers only FileJobStore. That
is the point: when a DynamoDB engine arrives it has to pass *this* file rather than shipping
with its own, softer definition of correctness. Engine-specific mechanics (locking, atomic
replace, the temp-file sweeper) live in test_job_store.py instead.
"""

import job_contracts as jc
import job_store as js
import pytest


def make_record(job_id="job1", owner_id="local", **overrides):
    record = jc.new_record(
        job_id=job_id,
        owner_id=owner_id,
        mode="txt2img_hq",
        media_kind="image",
        catalog_id="cyberrealistic_pony:txt2img_hq",
        backend="local",
        params={"prompt": "portrait photo", "seed": 9000},
        now=1000.0,
    )
    record.update(overrides)
    return record


ARTIFACT = {
    "bucket": js.LOCAL_BUCKET,
    "key": "generated/job1/" + "a" * 64 + ".png",
    "sha256": "a" * 64,
    "media_kind": "image",
    "content_type": "image/png",
    "byte_length": 67,
}


@pytest.fixture(params=["file"])
def store(request, tmp_path):
    if request.param == "file":
        return js.FileJobStore(str(tmp_path / "jobs"), sleep=lambda _seconds: None)
    raise AssertionError(f"unregistered engine {request.param}")


# --- create and read ----------------------------------------------------------------------------


def test_create_then_get_round_trips(store):
    store.create("local", make_record())
    assert store.get("local", "job1")["params"]["seed"] == 9000


def test_get_returns_none_for_another_owner(store):
    """None rather than an error: telling a caller apart "no such job" from "not yours" is an
    enumeration oracle, and the id space here is guessable by design (job ids appear in URLs)."""
    store.create("local", make_record())
    assert store.get("someone-else", "job1") is None


def test_get_returns_none_for_a_job_that_never_existed(store):
    assert store.get("local", "nope") is None


def test_creating_the_same_job_twice_is_refused(store):
    store.create("local", make_record())
    with pytest.raises(js.JobStoreError, match="already exists"):
        store.create("local", make_record())


def test_create_refuses_a_record_belonging_to_a_different_owner(store):
    with pytest.raises(js.JobStoreError, match="owner_id does not match"):
        store.create("local", make_record(owner_id="someone-else"))


# --- compare-and-swap ---------------------------------------------------------------------------


def test_update_applies_the_mutation(store):
    store.create("local", make_record())

    def advance(record):
        record["status"] = jc.SUBMITTING
        record["submit_attempt"] += 1

    assert store.update("local", "job1", advance)["status"] == jc.SUBMITTING
    assert store.get("local", "job1")["submit_attempt"] == 1


def test_update_loses_against_a_newer_submit_attempt(store):
    """The scenario this exists for: a provider response from attempt 1 arrives after a recovery
    pass already started attempt 2. It legitimately holds the lock and must still write nothing."""
    store.create("local", make_record(submit_attempt=2))
    with pytest.raises(js.StaleWriteError, match="submit_attempt"):
        store.update("local", "job1", lambda r: None, expect_submit_attempt=1)


def test_update_loses_against_an_unexpected_status(store):
    store.create("local", make_record(status=jc.QUEUED))
    with pytest.raises(js.StaleWriteError, match="expected one of"):
        store.update("local", "job1", lambda r: None, expect_status=jc.RUNNING)


def test_update_accepts_any_of_several_expected_statuses(store):
    store.create("local", make_record(status=jc.QUEUED))
    store.update("local", "job1", lambda r: None, expect_status=(jc.QUEUED, jc.RUNNING))


def test_update_refuses_an_illegal_transition(store):
    """The store is the last place that can stop a bad edge reaching disk, so it re-checks rather
    than trusting whichever service layer called it."""
    store.create("local", make_record(status=jc.QUEUED))

    def jump(record):
        record["status"] = jc.COMPLETED

    with pytest.raises(jc.ContractError, match="illegal status transition"):
        store.update("local", "job1", jump)


def test_update_refuses_a_job_belonging_to_another_owner(store):
    store.create("local", make_record())
    with pytest.raises(js.JobNotFound):
        store.update("someone-else", "job1", lambda r: None)


def test_update_stamps_updated_at(store):
    store.create("local", make_record())
    assert store.update("local", "job1", lambda r: None)["updated_at"] >= 1000.0


# --- the reverse index --------------------------------------------------------------------------


def test_bind_makes_a_provider_job_routable(store):
    store.create("local", make_record(status=jc.SUBMITTING, submit_attempt=1))
    store.bind_provider_job_id("local", "job1", "runpod", "rp-abc-123")
    assert store.find_by_provider_job_id("runpod", "rp-abc-123") == ("local", "job1")
    assert store.get("local", "job1")["provider_job_id"] == "rp-abc-123"


def test_find_by_provider_job_id_is_the_only_method_without_an_owner(store):
    """Structural, not an oversight: a provider webhook arrives with no authenticated owner. It
    hands back the owner it resolved, and everything after it goes through an owner-taking call."""
    store.create("local", make_record(status=jc.SUBMITTING))
    store.bind_provider_job_id("local", "job1", "runpod", "rp-1")
    owner_id, job_id = store.find_by_provider_job_id("runpod", "rp-1")
    assert store.get(owner_id, job_id)["job_id"] == "job1"


def test_find_by_unknown_provider_job_id_returns_none(store):
    assert store.find_by_provider_job_id("runpod", "never-seen") is None


def test_the_index_is_scoped_per_provider(store):
    store.create("local", make_record(status=jc.SUBMITTING))
    store.bind_provider_job_id("local", "job1", "runpod", "shared-id")
    assert store.find_by_provider_job_id("replicate", "shared-id") is None


def test_bind_loses_against_a_newer_submit_attempt(store):
    store.create("local", make_record(status=jc.SUBMITTING, submit_attempt=2))
    with pytest.raises(js.StaleWriteError):
        store.bind_provider_job_id("local", "job1", "runpod", "rp-1", expect_submit_attempt=1)


def test_bind_refuses_an_unknown_provider(store):
    store.create("local", make_record(status=jc.SUBMITTING))
    with pytest.raises(js.JobStoreError, match="unknown provider"):
        store.bind_provider_job_id("local", "job1", "not-a-provider", "x")


# --- cancellation is an intent flag, not a state ------------------------------------------------


def test_request_cancel_records_intent_without_touching_status(store):
    """A cancel that arrives while the job is moving submitting -> running must still stand; that
    is the whole reason it is a timestamp instead of a status."""
    store.create("local", make_record(status=jc.RUNNING))
    record = store.request_cancel("local", "job1", at=1234.0)
    assert record["cancel_requested_at"] == 1234.0
    assert record["status"] == jc.RUNNING


def test_request_cancel_is_idempotent_and_keeps_the_first_timestamp(store):
    store.create("local", make_record(status=jc.RUNNING))
    store.request_cancel("local", "job1", at=1.0)
    assert store.request_cancel("local", "job1", at=999.0)["cancel_requested_at"] == 1.0


def test_a_cancel_survives_a_later_status_advance(store):
    store.create("local", make_record(status=jc.RUNNING))
    store.request_cancel("local", "job1", at=5.0)
    store.update("local", "job1", lambda r: r.update(status=jc.CHECKING))
    assert store.get("local", "job1")["cancel_requested_at"] == 5.0


# --- listing ------------------------------------------------------------------------------------


def test_list_by_owner_sees_only_that_owner(store):
    store.create("local", make_record("a"))
    store.create("other", make_record("b", owner_id="other"))
    assert [r["job_id"] for r in store.list_by_owner("local")] == ["a"]


def test_list_by_owner_filters_by_status(store):
    store.create("local", make_record("a", status=jc.RUNNING))
    store.create("local", make_record("b", status=jc.QUEUED))
    assert [r["job_id"] for r in store.list_by_owner("local", status=jc.RUNNING)] == ["a"]


def test_list_unterminal_skips_finished_jobs(store):
    store.create("local", make_record("a", status=jc.RUNNING))
    store.create("local", make_record("b", status=jc.FAILED))
    store.create("local", make_record("c", status=jc.COMPLETED, artifacts=[ARTIFACT]))
    assert [job_id for _owner, job_id in store.list_unterminal()] == ["a"]


def test_list_unterminal_returns_owner_job_pairs(store):
    store.create("other", make_record("a", owner_id="other", status=jc.RUNNING))
    assert store.list_unterminal() == [("other", "a")]


# --- purge: the same policy image_api has always had --------------------------------------------


def test_purge_drops_finished_jobs_past_the_ttl(store):
    store.create("local", make_record("old", status=jc.FAILED))
    assert store.purge(now=1000.0 + js.TTL_SECONDS + 1) == 1
    assert store.get("local", "old") is None


def test_purge_never_evicts_an_unfinished_job(store):
    """A slow generation must never be forgotten out from under a polling client - the exact
    guarantee image_api._evict_jobs_locked:160 was written to give."""
    store.create("local", make_record("slow", status=jc.RUNNING))
    assert store.purge(now=1000.0 + js.TTL_SECONDS * 10) == 0
    assert store.get("local", "slow") is not None


def test_purge_over_the_cap_takes_the_oldest_finished_first(store):
    for index in range(4):
        store.create("local", make_record(f"j{index}", status=jc.FAILED, created_at=1000.0 + index))
    assert store.purge(now=1000.0, max_records=2) == 2
    assert [r["job_id"] for r in store.list_by_owner("local")] == ["j3", "j2"]


def test_purge_stops_when_nothing_left_is_safe_to_evict(store):
    """Over the cap but everything is still running: stop, do not evict. image_api's loop has the
    same `else: break`."""
    for index in range(4):
        store.create("local", make_record(f"j{index}", status=jc.RUNNING))
    assert store.purge(now=1000.0, max_records=1) == 0
    assert len(store.list_by_owner("local")) == 4


def test_purge_also_removes_the_reverse_index_entry(store):
    store.create("local", make_record("a", status=jc.SUBMITTING))
    store.bind_provider_job_id("local", "a", "runpod", "rp-9")
    store.update("local", "a", lambda r: r.update(status=jc.FAILED))
    store.purge(now=1000.0 + js.TTL_SECONDS + 1)
    assert store.find_by_provider_job_id("runpod", "rp-9") is None
