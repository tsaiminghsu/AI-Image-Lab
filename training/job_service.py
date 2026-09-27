"""Runs a generation as a recorded job: one row that survives a restart, one owner, one place
that decides what each outcome means.

Scope, stated plainly: this is the local ComfyUI backend only. The RunPod path binds a provider
job id and takes terminal state from a webhook, which needs the routing in job_webhook.py, so
submit_cloud is deliberately absent rather than half-present.

The one structural decision worth keeping when the cloud path arrives: apply_terminal is a
function, not an HTTP endpoint. The poller calls it and a webhook route will call it, so the two
can never grow into two state machines that disagree. It is also why every transition here is
testable offline today, with no provider and no GPU.

Owner boundary: owner_id is the first positional argument of every public method and is never
read out of a payload. The repo has no multi-tenancy and probably never will, but the parameter
costs one word and it is what would let image_api's caller - the ai-companion backend, the entire
reason that file exists - be partitioned later without revisiting every call site.
"""

import os
import shutil
import time
import uuid

import comfyui_client as client
import job_contracts as jc
import job_store as js
import output_check as oc
import param_resolver as pr

# min(60, 2 * poll_count). Linear ramp to a one-minute ceiling: a local generation that is going
# to finish finishes in seconds to minutes, and a job that has gone quiet is not helped by asking
# faster.
RECONCILE_BACKOFF_CEILING_SECONDS = 60
RECONCILE_BACKOFF_STEP_SECONDS = 2
RECONCILE_MAX_POLLS = 30

MANUAL_RECONCILIATION = "manual_reconciliation_required"


class _Sink:
    """Translates comfyui_client's three submission events into record transitions.

    Every write is a compare-and-swap on submit_attempt, so a late event from a superseded
    attempt writes nothing instead of resurrecting a job somebody already gave up on.
    """

    def __init__(self, service, owner_id, job_id, submit_attempt):
        self.service = service
        self.owner_id = owner_id
        self.job_id = job_id
        self.submit_attempt = submit_attempt
        self.provider_job_id = None
        self.submit_failed = False

    def on_submitting(self):
        pass  # already committed by submit_local before the sink was installed

    def on_submitted(self, provider, provider_job_id):
        self.provider_job_id = provider_job_id
        self.service.store.bind_provider_job_id(
            self.owner_id, self.job_id, provider, provider_job_id,
            expect_submit_attempt=self.submit_attempt)
        self.service._set_status(self.owner_id, self.job_id, jc.RUNNING,
                                 expect_submit_attempt=self.submit_attempt)

    def on_submit_failed(self, exc):
        self.submit_failed = True
        # A refused connection never left this machine, so there is nothing on the card and
        # nothing to reconcile - that is an ordinary failure. Anything else (a read timeout, a
        # dropped response) may have been accepted, and guessing either way is how you get a
        # double submission or an orphaned job holding 8 GB of VRAM.
        if client._is_connection_refused(exc):
            self.service._conclude_failure(self.owner_id, self.job_id, str(exc),
                                           kind="connection_refused",
                                           expect_submit_attempt=self.submit_attempt)
            return
        self.service._mark_uncertain(self.owner_id, self.job_id, str(exc),
                                     expect_submit_attempt=self.submit_attempt)


class JobService:
    def __init__(self, store, *, clock=time.time, artifact_root=None, bucket=js.LOCAL_BUCKET):
        self.store = store
        self._clock = clock
        self.artifact_root = artifact_root or os.path.join(os.getcwd(), "outputs")
        self.bucket = bucket

    # --- creation -------------------------------------------------------------------------

    def create(self, owner_id, *, mode, catalog_id, params, media_kind="image", backend="local",
               request=None, job_id=None, not_before=None, route=None):
        record = jc.new_record(
            job_id=job_id or uuid.uuid4().hex,
            owner_id=owner_id,
            mode=mode,
            media_kind=media_kind,
            catalog_id=catalog_id,
            backend=backend,
            params=params,
            request=request,
            now=self._clock(),
            not_before=not_before,
            route=route,
        )
        return self.store.create(owner_id, record)

    def resolve_params(self, checkpoint, **suggested):
        """Normalise a caller's suggestion against the catalog's defaults for that checkpoint.

        Imported here rather than at module scope: capability_catalog pulls in
        generate_character, and job_service has to stay importable by things that must not drag
        the whole CLI in behind them.
        """
        import capability_catalog as catalog

        defaults = catalog.defaults_for(checkpoint)
        return pr.resolve(
            prompt=suggested.get("prompt", ""),
            width=suggested.get("width"),
            height=suggested.get("height"),
            seed=suggested.get("seed"),
            default_width=defaults.get("width"),
            default_height=defaults.get("height"),
        )

    # --- the local run --------------------------------------------------------------------

    def submit_local(self, owner_id, job_id, generate, *, output_path, expect_width=None,
                     expect_height=None):
        """Run `generate` with this job recorded around it.

        `generate` is a zero-argument callable - normally a lambda over gc.gen_custom - so this
        module never grows an opinion about the generator's signature, which is the thing most
        likely to change underneath it.

        `output_path` is where the generator will leave its file. The caller knows it because it
        chose the filename; asking the generator would mean changing every one of them. None means
        "whatever path `generate` returns" - gen_video_animatediff names its own file.

        expect_width/expect_height are the *delivered* dimensions, if the caller knows them. They
        are not the sampling dimensions in params: the HQ path upscales, so a job sampled at
        832x1216 legitimately arrives 1056x1536. Left unset, the size is measured and recorded but
        not asserted.
        """
        row = self._begin_submitting(owner_id, job_id)
        sink = _Sink(self, owner_id, job_id, row["submit_attempt"])
        try:
            with client.job_sink_scope(sink):
                returned = generate()
        except BaseException as exc:
            # BaseException for the same reason image_api._run_job:190 and worker/handler.py do
            # it: pose_skeletons and talking_head can still raise SystemExit, which derives from
            # BaseException and would otherwise leave this job unfinished forever.
            if not sink.submit_failed:
                self._conclude_failure(owner_id, job_id, str(exc), kind=type(exc).__name__,
                                       expect_submit_attempt=row["submit_attempt"])
            raise
        if output_path is None:
            output_path = returned if isinstance(returned, (str, os.PathLike)) else None
        if output_path is None:
            return self._conclude_failure(owner_id, job_id, "the generator returned no output path",
                                          kind="output_missing", expect_submit_attempt=row["submit_attempt"])
        return self.apply_terminal(owner_id, job_id, output_path,
                                   expect_submit_attempt=row["submit_attempt"],
                                   expect_width=expect_width, expect_height=expect_height)

    def _begin_submitting(self, owner_id, job_id):
        """Commit `submitting` and claim an attempt number, in one write, before anything leaves
        this process. Everything after it compare-and-swaps on the number it returns."""

        def start(record):
            record["status"] = jc.SUBMITTING
            record["submit_attempt"] += 1
            record["submitted_at"] = self._clock()

        return self.store.update(owner_id, job_id, start, expect_status=jc.QUEUED)

    # --- outcomes -------------------------------------------------------------------------

    def apply_terminal(self, owner_id, job_id, output_path, *, expect_submit_attempt=None,
                       expect_width=None, expect_height=None, cost=None):
        """The single mutation for "the provider says it is done".

        The poller reaches it by returning from a generation; a webhook route will reach it by
        parsing a callback. Keeping it one function is what stops push and poll becoming two
        state machines that disagree - and it is idempotent, so a duplicate delivery is a no-op
        that returns the record already written.
        """
        current = self.store.get(owner_id, job_id)
        if current is None:
            raise js.JobNotFound(f"job {job_id} is not visible to owner {owner_id!r}")
        if current["status"] in jc.TERMINAL_STATUSES:
            return current

        record = self._set_status(owner_id, job_id, jc.CHECKING,
                                  expect_submit_attempt=expect_submit_attempt)
        media_kind = record["media_kind"]
        try:
            # Only what the caller explicitly expects, never record["params"]. Those are the
            # *sampling* dimensions, and the HQ two-pass path deliberately delivers something
            # larger - a live run at 832x1216 came back 1056x1536, and defaulting the expectation
            # from params failed a perfectly good image. Asserting the wrong thing is worse than
            # asserting nothing; the measured size is recorded in the report's details either way.
            # (Deriving the expected output size would mean the catalog declaring the hires scale,
            # which is worth doing and is not this change.)
            report, facts = oc.check_output(
                output_path, media_kind=media_kind,
                expect_width=expect_width, expect_height=expect_height,
            )
        except oc.OutputCheckError as exc:
            return self._conclude_failure(owner_id, job_id, str(exc), kind="output_missing")

        if report["state"] != oc.PASSED:
            return self._conclude_failure(owner_id, job_id, "; ".join(report["details"]["failures"]),
                                          kind="output_check_failed", report=report)

        extension = os.path.splitext(output_path)[1].lstrip(".").lower()
        artifact = self._materialise(job_id, output_path, facts, media_kind, extension)
        def finish(record):
            record["status"] = jc.COMPLETED
            record["artifacts"] = [artifact.to_dict()]
            record["output_check"] = report
            record["completed_at"] = self._clock()
            for field, value in (cost or {}).items():
                record[field] = value

        return self.store.update(owner_id, job_id, finish, expect_status=jc.CHECKING)

    def _materialise(self, job_id, output_path, facts, media_kind, extension):
        """Put the produced bytes at their content-addressed key and describe them as an artifact.

        The key is content-addressed with the full 64 hex, not cloud_workflow.stage_upload:54's
        truncated 16 - that is a transport cache key, this is the thing output_check asserts
        against. It is also what ends the collision gui.py:1205 measured on 2026-09-12, where two
        tabs both defaulting to seed 9000 overwrote each other's file.

        The generator's own output file is left where it is, so existing callers and the CLI's
        --filename keep working. The second name is made with os.link where the filesystem allows
        it, falling back to a copy: both names refer to immutable generated output that nothing
        edits in place, which is what makes a hardlink safe here and specifically not safe for the
        model weights CLAUDE.md warns about.
        """
        key = f"generated/{job_id}/{facts['sha256']}.{extension or 'png'}"
        destination = os.path.join(self.artifact_root, *key.split("/"))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if not os.path.exists(destination):
            try:
                os.link(output_path, destination)
            except (OSError, NotImplementedError):
                shutil.copy2(output_path, destination)
        return jc.Artifact(
            bucket=self.bucket,
            key=key,
            sha256=facts["sha256"],
            media_kind=media_kind,
            content_type=jc.CONTENT_TYPES.get(extension, "application/octet-stream"),
            byte_length=facts["byte_length"],
        )

    def resolve(self, owner_id, job_id):
        """Where a completed job's output can actually be read, or None while it has none."""
        record = self.store.get(owner_id, job_id)
        if not record or not record["artifacts"]:
            return None
        return js.resolve_artifact(owner_id, record["artifacts"][0], local_root=self.artifact_root)

    def request_cancel(self, owner_id, job_id):
        """Record the intent. The generation loop notices on its next pass; a job that has not
        left the queue is concluded immediately."""
        record = self.store.request_cancel(owner_id, job_id, at=self._clock())
        if record["status"] == jc.QUEUED:
            return self._set_status(owner_id, job_id, jc.CANCELLED)
        return record

    def _set_status(self, owner_id, job_id, status, *, expect_submit_attempt=None, expect_status=None):
        return self.store.update(owner_id, job_id, lambda record: record.update(status=status),
                                 expect_submit_attempt=expect_submit_attempt,
                                 expect_status=expect_status)

    def _conclude_failure(self, owner_id, job_id, message, *, kind, expect_submit_attempt=None,
                          report=None):
        def fail(record):
            record["status"] = jc.FAILED
            record["last_error"] = message
            record["last_error_kind"] = kind
            record["completed_at"] = self._clock()
            if report is not None:
                record["output_check"] = report

        try:
            return self.store.update(owner_id, job_id, fail, expect_submit_attempt=expect_submit_attempt)
        except js.StaleWriteError:
            return self.store.get(owner_id, job_id)

    def _mark_uncertain(self, owner_id, job_id, reason, *, expect_submit_attempt=None):
        """An outcome we cannot determine after the prompt may already have been accepted.

        Has reconciliation-only exits: nothing here ever re-submits. A re-submit can double-charge
        a cloud GPU, or put a second copy of the same job on an 8 GB card, against work that was
        in fact accepted - and "probably didn't go through" is not evidence.
        """

        def uncertain(record):
            record["status"] = jc.SUBMISSION_UNKNOWN
            record["poll_count"] += 1
            record["last_error"] = reason
            record["last_error_kind"] = "submission_unknown"

        return self.store.update(owner_id, job_id, uncertain, expect_submit_attempt=expect_submit_attempt)

    def reconcile_backoff(self, poll_count):
        return min(RECONCILE_BACKOFF_CEILING_SECONDS, RECONCILE_BACKOFF_STEP_SECONDS * max(1, poll_count))

    def give_up_uncertain(self, owner_id, job_id):
        """Stop polling an uncertain job and hand it to a person.

        It becomes `failed`, but the message says so honestly rather than claiming the work did
        not happen. In a one-person lab "escalate to a human" is a line in the store's
        reconciliation log plus whatever the GUI shows - not a pager, and not a silent drop.
        """
        record = self.store.get(owner_id, job_id)
        if record is None or record["status"] != jc.SUBMISSION_UNKNOWN:
            return record
        self.store._log_reconciliation(f"{MANUAL_RECONCILIATION} owner={owner_id} job={job_id}")
        return self._conclude_failure(owner_id, job_id, MANUAL_RECONCILIATION, kind=MANUAL_RECONCILIATION)
