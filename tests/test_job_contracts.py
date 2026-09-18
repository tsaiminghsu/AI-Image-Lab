"""Pins the job contract: the artifact key rules, the loose-outside/strict-inside split, and the
status transition table.

The two properties worth stating up front, because they are the ones a later refactor is most
likely to erode:

  * An artifact key is attacker-shaped input by the time it reaches a filesystem or an S3 key, so
    traversal is rejected in the constructor rather than by whoever happens to join a path next.
  * "completed" is a claim that bytes exist. A completed record with no artifact is refused by the
    contract, so no call site can produce one by forgetting.
"""

import job_contracts as jc
import pytest

VALID_SHA = "a" * 64


def make_artifact(**overrides):
    fields = {
        "bucket": "local-outputs",
        "key": "generated/abc123/" + VALID_SHA + ".png",
        "sha256": VALID_SHA,
        "media_kind": "image",
        "content_type": "image/png",
        "byte_length": 67,
    }
    fields.update(overrides)
    return fields


def make_record(**overrides):
    record = jc.new_record(
        job_id="job1",
        owner_id="local",
        mode="txt2img_hq",
        media_kind="image",
        catalog_id="cyberrealistic_pony:txt2img_hq",
        backend="local",
        params={"prompt": "portrait photo", "width": 1024, "height": 1024, "seed": 9000},
        now=1000.0,
    )
    record.update(overrides)
    return record


# --- artifact keys: traversal is refused at the schema layer ------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        "../etc/passwd",
        "generated/../../secret.png",
        "..\\windows\\system32",
        "C:\\Users\\Attlie\\secret.png",
        "/absolute/path.png",
        "generated//double.png",
        "generated/./here.png",
        "",
        "-leading-dash.png",
    ],
)
def test_unsafe_artifact_keys_are_refused(bad_key):
    """Every one of these, joined naively onto an output directory, escapes it or resolves
    somewhere the caller did not mean. The constructor is the only place that can guarantee no
    caller forgot to check."""
    with pytest.raises(jc.ContractError, match="unsafe artifact key|artifact key must be"):
        jc.Artifact(**make_artifact(key=bad_key))


def test_a_plain_relative_key_is_accepted():
    artifact = jc.Artifact(**make_artifact())
    assert artifact.key.startswith("generated/")
    assert artifact.to_dict() == make_artifact()


def test_artifact_sha256_must_be_lowercase_hex_of_the_right_length():
    with pytest.raises(jc.ContractError, match="64 lowercase hex"):
        jc.Artifact(**make_artifact(sha256="A" * 64))
    with pytest.raises(jc.ContractError, match="64 lowercase hex"):
        jc.Artifact(**make_artifact(sha256="abc"))


def test_artifact_rejects_unknown_and_missing_fields():
    """Strict inside: an artifact is what a completed job points at, so a field nobody validated
    must not ride along inside it."""
    with pytest.raises(jc.ContractError, match="unknown artifact field"):
        jc.Artifact.from_dict(make_artifact(outputUrl="https://example.invalid/x.png"))
    incomplete = make_artifact()
    del incomplete["sha256"]
    with pytest.raises(jc.ContractError, match="missing artifact field"):
        jc.Artifact.from_dict(incomplete)


# --- loose outside, strict inside ---------------------------------------------------------------


def test_unknown_top_level_key_survives_a_round_trip():
    """Forward compatibility on the envelope: a record written by a newer version must not be
    silently truncated by an older one that happens to read and rewrite it."""
    record = make_record()
    record["quality_state"] = "approved"
    restored = jc.loads(jc.dumps(record))
    assert restored["quality_state"] == "approved"


def test_unknown_key_inside_params_is_refused():
    """A typo like 'witdh' would otherwise be accepted and then silently do nothing, which is the
    failure mode that wastes a GPU run before anyone notices."""
    record = make_record()
    record["params"]["witdh"] = 512
    with pytest.raises(jc.ContractError, match="unknown param"):
        jc.validate_record(record)


def test_unknown_key_inside_an_artifact_is_refused():
    record = make_record()
    record["artifacts"] = [make_artifact(providerJobId="runpod-123")]
    with pytest.raises(jc.ContractError, match="unknown artifact field"):
        jc.validate_record(record)


def test_params_values_must_be_json_scalars():
    record = make_record()
    record["params"]["prompt"] = {"text": "nested"}
    with pytest.raises(jc.ContractError, match="must be a JSON scalar"):
        jc.validate_record(record)


# --- a completed job must carry evidence --------------------------------------------------------


def test_completed_without_an_artifact_is_refused():
    record = make_record(status=jc.COMPLETED)
    with pytest.raises(jc.ContractError, match="at least one output artifact"):
        jc.validate_record(record)


def test_completed_with_an_artifact_is_accepted():
    record = make_record(status=jc.COMPLETED, artifacts=[make_artifact()])
    assert jc.validate_record(record)["status"] == jc.COMPLETED


# --- the transition table -----------------------------------------------------------------------


def test_terminal_states_have_no_outgoing_edges():
    for status in jc.TERMINAL_STATUSES:
        assert jc._ALLOWED[status] == frozenset(), f"{status} must be terminal"


def test_every_status_is_reachable_from_queued():
    """A state nothing can reach is dead code pretending to be a contract."""
    seen = {jc.QUEUED}
    frontier = [jc.QUEUED]
    while frontier:
        for nxt in jc._ALLOWED[frontier.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(jc.STATUSES)


def test_running_cannot_jump_straight_to_completed():
    """Completion must pass through `checking`, which is where the output check re-reads the
    produced bytes. Allowing the shortcut would make the check skippable."""
    assert not jc.can_transition(jc.RUNNING, jc.COMPLETED)
    assert jc.can_transition(jc.RUNNING, jc.CHECKING)
    assert jc.can_transition(jc.CHECKING, jc.COMPLETED)


def test_checking_is_reachable_without_passing_through_running():
    """A generator can return having never announced a submission - a backend that does not route
    through _submit_and_wait, or a result that arrived before the sink saw anything. The work is
    over either way, so there is an output to check."""
    assert jc.can_transition(jc.SUBMITTING, jc.CHECKING)


def test_completed_is_only_ever_reachable_through_checking():
    """The invariant the extra edge above must not weaken: every path to COMPLETED goes through
    the state where the produced bytes are re-read, so the output check cannot be skipped."""
    predecessors = [status for status, allowed in jc._ALLOWED.items() if jc.COMPLETED in allowed]
    assert predecessors == [jc.CHECKING]


def test_submission_unknown_never_goes_back_to_submitting():
    """An uncertain submission is resolved by reconciling with the provider, never by submitting
    again - a re-submit can double-charge a GPU against a job that was in fact accepted."""
    assert not jc.can_transition(jc.SUBMISSION_UNKNOWN, jc.SUBMITTING)
    assert jc.can_transition(jc.SUBMISSION_UNKNOWN, jc.FAILED)
    assert jc.can_transition(jc.SUBMISSION_UNKNOWN, jc.RUNNING)


def test_a_same_status_write_is_legal():
    """A poll bumps poll_count without moving the status; that must not need a special case."""
    assert jc.can_transition(jc.RUNNING, jc.RUNNING)


def test_require_transition_raises_on_an_illegal_edge():
    with pytest.raises(jc.ContractError, match="illegal status transition"):
        jc.require_transition(jc.COMPLETED, jc.RUNNING)


# --- cancellation is a timestamp, not a status --------------------------------------------------


def test_cancel_requested_at_is_not_a_status():
    """Cancellation is an orthogonal intent flag precisely so it cannot be lost when the status
    advances underneath it."""
    assert "cancel_requested_at" not in jc.STATUSES
    assert "cancel_requested" not in jc.STATUSES
    assert "cancel_requested_at" in make_record()


# --- cost and timing are a tri-state ------------------------------------------------------------


def test_a_reported_zero_is_not_the_same_as_unknown():
    """A provider reporting 0 ms of queue wait is information. Collapsing it into None is the bug
    this tri-state exists to prevent (cloud_video.py:418 had exactly that shape)."""
    record = make_record()
    record["provider_delay_ms"] = 0
    assert jc.validate_record(record)["provider_delay_ms"] == 0
    assert make_record()["provider_delay_ms"] is None


def test_cost_fields_reject_negative_values():
    record = make_record()
    record["gpu_seconds"] = -1
    with pytest.raises(jc.ContractError, match="non-negative"):
        jc.validate_record(record)


# --- derived fields -----------------------------------------------------------------------------


def test_operation_is_derived_from_media_kind_and_cannot_be_overridden():
    record = make_record()
    assert record["operation"] == "image_generation"
    record["operation"] = "video_generation"
    with pytest.raises(jc.ContractError, match="derived from media_kind"):
        jc.validate_record(record)


def test_dumps_is_stable_regardless_of_key_insertion_order():
    """The poller and the webhook build a record by different routes; a byte-identical dump is
    what lets a later test assert the two paths agree exactly."""
    first = make_record()
    second = {key: first[key] for key in reversed(list(first))}
    assert jc.dumps(first) == jc.dumps(second)


# --- output check report ------------------------------------------------------------------------


def make_report(**overrides):
    report = {
        "policy": {"id": "technical_delivery", "version": 1, "checks": ["decode"], "sha256": VALID_SHA},
        "state": "passed",
        "scope": "technical_delivery",
    }
    report.update(overrides)
    return report


def test_output_check_scope_must_be_the_literal():
    """scope is a literal so a report can never silently widen the claim it is making - a
    technical delivery check must not be able to call itself a quality approval."""
    record = make_record(output_check=make_report(scope="quality_approval"))
    with pytest.raises(jc.ContractError, match="must be the literal"):
        jc.validate_record(record)


def test_output_check_rejects_an_unknown_policy_field():
    record = make_record(
        output_check=make_report(
            policy={
                "id": "technical_delivery",
                "version": 1,
                "checks": ["decode"],
                "sha256": VALID_SHA,
                "reviewer": "nobody",
            }
        )
    )
    with pytest.raises(jc.ContractError, match="unknown output_check.policy field"):
        jc.validate_record(record)


def test_output_check_accepts_a_well_formed_report():
    record = make_record(output_check=make_report())
    assert jc.validate_record(record)["output_check"]["state"] == "passed"
