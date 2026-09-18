"""The job lifecycle against a fake ComfyUI: what each outcome means, and the two orderings that
decide whether a lost response is recoverable.

The assertion this file exists for is test_submitting_is_committed_before_the_first_post. Every
other guarantee here - never re-submitting, distinguishing "never left this machine" from "we do
not know", being able to look for an orphaned job in /queue - depends on the record reaching disk
before the prompt reaches the network. Afterwards both orders look identical in the final record,
which is why the ordering needs a witness rather than an inspection.
"""

import os

import job_contracts as jc
import job_service as svc
import job_store as js
import pytest
from helpers import FakeComfy, RecordingJobStore
from test_output_check import make_png


@pytest.fixture
def service(tmp_path, fake_comfy, no_sleep):
    """A service over a real FileJobStore, wired to the fake ComfyUI the fixture installed.

    no_sleep is not optional here: _post_prompt_with_retry's budget is bounded by
    max(wall clock, seconds slept), so the refused-connection test spent 6 real seconds walking
    the retry ladder before this fixture stubbed sleep out.
    """
    inner = js.FileJobStore(str(tmp_path / "jobs"), sleep=lambda _seconds: None)
    store = RecordingJobStore(inner, witness=lambda: len(fake_comfy.posted))
    return svc.JobService(store, clock=lambda: 1000.0, artifact_root=str(tmp_path / "out"))


def a_job(service, **overrides):
    params = {"prompt": "portrait photo", "width": 64, "height": 64, "seed": 9000}
    params.update(overrides.pop("params", {}))
    return service.create(
        "local",
        mode="txt2img_hq",
        catalog_id="cyberrealistic_pony:txt2img_hq",
        params=params,
        job_id=overrides.pop("job_id", "job1"),
        **overrides,
    )


def finishing_run(fake_comfy, tmp_path, width=64, height=64):
    """Point the fake at a real PNG and a completed history, and hand back the path the
    generation will have written to."""
    fake_comfy.view_bytes = make_png(width, height)
    fake_comfy.history_sequence = [FakeComfy.done()]
    return str(tmp_path / "raw" / "out.png")


def generate(client_module, wf=None):
    return lambda: client_module._submit_and_wait(wf if wf is not None else {})


# --- the ordering everything else rests on -------------------------------------------------------


def test_submitting_is_committed_before_the_first_post(service, fake_comfy, tmp_path):
    """If ComfyUI accepts the prompt and the response is then lost, the record is the only
    evidence a job exists on the card. Written afterwards, it would be evidence of nothing."""
    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    service.submit_local("local", "job1", generate(client), output_path=output)

    submitting = next(w for w in service.store.writes if w["status"] == jc.SUBMITTING)
    assert submitting["witness"] == 0, "the submitting write must land before any POST"
    assert "/prompt" in fake_comfy.paths("post")


def test_the_happy_path_walks_the_whole_machine(service, fake_comfy, tmp_path):
    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    record = service.submit_local("local", "job1", generate(client), output_path=output)

    assert record["status"] == jc.COMPLETED
    assert service.store.statuses() == [
        jc.QUEUED,
        jc.SUBMITTING,
        jc.SUBMITTING,
        jc.RUNNING,
        jc.CHECKING,
        jc.COMPLETED,
    ], "the second submitting write is bind_provider_job_id, which lands before the move to running"
    assert record["provider"] == "comfyui"
    assert record["provider_job_id"] == fake_comfy.prompt_id
    assert record["output_check"]["state"] == "passed"


def test_a_completed_job_is_routable_by_its_provider_job_id(service, fake_comfy, tmp_path):
    """The reverse index is what a webhook will arrive with; binding it during the run rather
    than at the end is what makes it useful while the job is still in flight."""
    import comfyui_client as client

    a_job(service)
    service.submit_local("local", "job1", generate(client), output_path=finishing_run(fake_comfy, tmp_path))
    assert service.store.find_by_provider_job_id("comfyui", fake_comfy.prompt_id) == ("local", "job1")


# --- artifacts are content-addressed -------------------------------------------------------------


def test_the_artifact_key_is_the_hash_not_the_seed(service, fake_comfy, tmp_path):
    """gui.py:1205 records a measured incident: two sessions both defaulting to seed 9000, one
    overwriting gui_seed9000.png before the other read it back. A key derived from the bytes
    cannot collide that way."""
    import comfyui_client as client

    a_job(service)
    record = service.submit_local("local", "job1", generate(client), output_path=finishing_run(fake_comfy, tmp_path))
    artifact = record["artifacts"][0]
    assert artifact["key"] == f"generated/job1/{artifact['sha256']}.png"
    assert "9000" not in artifact["key"]
    assert artifact["byte_length"] > 0


def test_the_recorded_hash_is_of_the_bytes_that_were_actually_written(service, fake_comfy, tmp_path):
    import hashlib

    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    record = service.submit_local("local", "job1", generate(client), output_path=output)
    on_disk = hashlib.sha256(open(output, "rb").read()).hexdigest()
    assert record["artifacts"][0]["sha256"] == on_disk


# --- submission failures are classified, never guessed -------------------------------------------


def test_a_lost_response_becomes_submission_unknown_and_never_resubmits(service, fake_comfy):
    """A read timeout may mean the prompt was accepted. Retrying would put a second copy of the
    same job on an 8 GB card; assuming failure would abandon one that is running."""
    import comfyui_client as client

    a_job(service)
    fake_comfy.post_failures = {"/prompt": [client.requests.exceptions.ReadTimeout("no answer")] * 10}
    with pytest.raises(Exception):
        service.submit_local("local", "job1", generate(client), output_path="unused.png")

    record = service.store.get("local", "job1")
    assert record["status"] == jc.SUBMISSION_UNKNOWN
    assert record["last_error_kind"] == "submission_unknown"
    assert record["submit_attempt"] == 1, "one attempt, whatever the transport retried internally"


def test_a_refused_connection_is_an_ordinary_failure(service, fake_comfy):
    """Refused means it never left this machine, so there is nothing on the card and nothing to
    reconcile. Calling that submission_unknown would turn the state into a catch-all and make the
    genuinely uncertain cases indistinguishable."""
    import comfyui_client as client

    a_job(service)
    fake_comfy.post_failures = {"/prompt": [FakeComfy.refused()] * 10}
    with pytest.raises(Exception):
        service.submit_local("local", "job1", generate(client), output_path="unused.png")

    record = service.store.get("local", "job1")
    assert record["status"] == jc.FAILED
    assert record["last_error_kind"] == "connection_refused"


def test_a_rejected_graph_is_a_clean_failure_not_an_uncertain_one(service, fake_comfy):
    """The POST succeeded and ComfyUI answered by refusing the graph, so nothing was queued."""
    import comfyui_client as client

    a_job(service)
    fake_comfy.node_errors = {"3": {"errors": [{"message": "bad"}]}}
    with pytest.raises(RuntimeError, match="workflow validation failed"):
        service.submit_local("local", "job1", generate(client), output_path="unused.png")

    record = service.store.get("local", "job1")
    assert record["status"] == jc.FAILED
    assert record["last_error_kind"] != "submission_unknown"


def test_an_uncertain_job_is_handed_to_a_person_rather_than_retried(service, fake_comfy):
    import comfyui_client as client

    a_job(service)
    fake_comfy.post_failures = {"/prompt": [client.requests.exceptions.ReadTimeout("x")] * 10}
    with pytest.raises(Exception):
        service.submit_local("local", "job1", generate(client), output_path="unused.png")

    record = service.give_up_uncertain("local", "job1")
    assert record["status"] == jc.FAILED
    assert record["last_error"] == svc.MANUAL_RECONCILIATION
    assert svc.MANUAL_RECONCILIATION in open(service.store.reconciliation_log, encoding="utf-8").read()


def test_giving_up_does_not_touch_a_job_that_is_not_uncertain(service, fake_comfy, tmp_path):
    import comfyui_client as client

    a_job(service)
    service.submit_local("local", "job1", generate(client), output_path=finishing_run(fake_comfy, tmp_path))
    assert service.give_up_uncertain("local", "job1")["status"] == jc.COMPLETED


def test_the_backoff_ramps_and_then_flattens():
    service_backoff = svc.JobService.reconcile_backoff
    assert service_backoff(None, 0) == 2
    assert service_backoff(None, 5) == 10
    assert service_backoff(None, 1000) == svc.RECONCILE_BACKOFF_CEILING_SECONDS


# --- terminal handling is one function, and it is idempotent -------------------------------------


def test_applying_the_same_terminal_twice_is_a_no_op(service, fake_comfy, tmp_path):
    """A webhook can be delivered more than once, and the poller may reach the same conclusion at
    the same moment. The second call has to return the record, not rewrite it."""
    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    first = service.submit_local("local", "job1", generate(client), output_path=output)
    second = service.apply_terminal("local", "job1", output)
    assert jc.dumps(first) == jc.dumps(second)


def test_a_terminal_from_a_superseded_attempt_writes_nothing(service, fake_comfy, tmp_path):
    """The compare-and-swap, from the outside: a response belonging to submit attempt 1 arriving
    after attempt 2 started must not conclude the job.

    The job is left mid-flight on purpose. Once it is terminal the idempotency short-circuit
    answers first and the attempt number never gets consulted - which is the right precedence
    (a duplicate delivery is not a conflict), but it means this guarantee has to be checked
    while the job is still running."""
    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    service._begin_submitting("local", "job1")
    service._set_status("local", "job1", jc.RUNNING)
    service.store.update("local", "job1", lambda r: r.update(submit_attempt=r["submit_attempt"] + 1))
    with pytest.raises(js.StaleWriteError):
        service.apply_terminal("local", "job1", output, expect_submit_attempt=1)
    assert service.store.get("local", "job1")["status"] == jc.RUNNING


def test_a_missing_output_file_fails_the_job_rather_than_crashing(service):
    a_job(service)
    service._begin_submitting("local", "job1")
    service._set_status("local", "job1", jc.RUNNING)
    record = service.apply_terminal("local", "job1", "no-such-file.png")
    assert record["status"] == jc.FAILED
    assert record["last_error_kind"] == "output_missing"


def test_an_output_that_fails_its_check_fails_the_job_and_keeps_the_report(service, fake_comfy, tmp_path):
    """The bytes arrived but are not the size the caller said to expect. Recording the report is
    what makes that distinguishable afterwards from a generation that never ran.

    The expectation is passed explicitly and deliberately not taken from params: those are the
    sampling dimensions, and the HQ path upscales past them on purpose."""
    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path, width=64, height=64)
    record = service.submit_local(
        "local", "job1", generate(client), output_path=output, expect_width=1024, expect_height=1024
    )
    assert record["status"] == jc.FAILED
    assert record["last_error_kind"] == "output_check_failed"
    assert record["output_check"]["state"] == "failed"
    assert "asked for 1024, got 64" in record["last_error"]


def test_the_sampling_dimensions_are_not_used_as_an_expectation(service, fake_comfy, tmp_path):
    """The bug a live run caught: the HQ two-pass path sampled 832x1216 and delivered 1056x1536,
    and defaulting the expectation from params failed a perfectly good image. Asserting the wrong
    thing is worse than asserting nothing."""
    import comfyui_client as client

    a_job(service, params={"width": 832, "height": 1216})
    output = finishing_run(fake_comfy, tmp_path, width=1056, height=1536)
    record = service.submit_local("local", "job1", generate(client), output_path=output)
    assert record["status"] == jc.COMPLETED
    assert record["output_check"]["details"]["width"] == 1056, "measured and recorded, just not asserted"
    assert "dimensions" not in record["output_check"]["policy"]["checks"], "and not claimed as a check"


# --- cancellation ---------------------------------------------------------------------------------


def test_cancelling_a_queued_job_concludes_it_immediately(service):
    a_job(service)
    assert service.request_cancel("local", "job1")["status"] == jc.CANCELLED


def test_cancelling_a_running_job_records_intent_without_moving_the_status(service):
    a_job(service)
    service._begin_submitting("local", "job1")
    service._set_status("local", "job1", jc.RUNNING)
    record = service.request_cancel("local", "job1")
    assert record["status"] == jc.RUNNING
    assert record["cancel_requested_at"] == 1000.0


# --- the owner boundary ----------------------------------------------------------------------------


def test_another_owner_cannot_see_or_conclude_the_job(service):
    a_job(service)
    assert service.store.get("someone-else", "job1") is None
    with pytest.raises(js.JobNotFound):
        service.apply_terminal("someone-else", "job1", "whatever.png")


# --- parameters are resolved through the catalog ----------------------------------------------------


def test_resolved_params_follow_the_checkpoint_the_job_is_for(service):
    """SD1.5's native resolution is 512x768, not the SDXL square - and the resolver clamps rather
    than refusing, so an absurd suggestion still produces a drawable job."""
    assert service.resolve_params("realistic_vision")["width"] == 512
    assert service.resolve_params("cyberrealistic_pony", width=4000)["width"] <= 1536
    assert service.resolve_params("cyberrealistic_pony", seed=-1)["seed"] == 0


# --- the artifact is actually there ----------------------------------------------------------------


def test_the_artifact_is_readable_at_its_content_addressed_key(service, fake_comfy, tmp_path):
    """A content-addressed key that no file sits at would make resolve() point at nothing - the
    record would describe an artifact that does not exist."""
    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    service.submit_local("local", "job1", generate(client), output_path=output)

    resolved = service.resolve("local", "job1")
    assert resolved["kind"] == "path"
    assert open(resolved["path"], "rb").read() == open(output, "rb").read()


def test_the_generators_own_output_file_is_left_alone(service, fake_comfy, tmp_path):
    """Existing callers and the CLI's --filename still expect their file where they asked for it,
    so the content-addressed name is a second name for the same bytes, not a move."""
    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    service.submit_local("local", "job1", generate(client), output_path=output)
    assert os.path.exists(output)


def test_materialising_twice_does_not_fail(service, fake_comfy, tmp_path):
    """apply_terminal is idempotent, and on a retry the destination may already exist."""
    import comfyui_client as client

    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    service.submit_local("local", "job1", generate(client), output_path=output)
    assert service.apply_terminal("local", "job1", output)["status"] == jc.COMPLETED


def test_a_copy_fallback_is_used_when_hardlinking_is_refused(service, fake_comfy, tmp_path, monkeypatch):
    """os.link fails across volumes and on some filesystems; losing the artifact there would be a
    silent data loss rather than a degraded one."""
    import comfyui_client as client

    monkeypatch.setattr(os, "link", lambda src, dst: (_ for _ in ()).throw(OSError("cross-device")))
    a_job(service)
    output = finishing_run(fake_comfy, tmp_path)
    service.submit_local("local", "job1", generate(client), output_path=output)
    resolved = service.resolve("local", "job1")
    assert open(resolved["path"], "rb").read() == open(output, "rb").read()


def test_resolve_returns_none_before_there_is_an_artifact(service):
    a_job(service)
    assert service.resolve("local", "job1") is None
