"""Runs deferred jobs once their time window opens and the machine is ready for them.

A deferred job is an ordinary job-store record in `queued` with `not_before` set. Nothing new is
invented for the queue itself: the store already persists across restarts, already has a strict
state machine, and job_service._begin_submitting already moves `queued -> submitting` as a
compare-and-swap under the store-wide lock - which is what lets the GUI and image_api both run a
scheduler over the same directory without ever running one job twice.

What a record has to carry to be runnable later, in `request` (the free-form, unvalidated part of
the record; `params` keeps its strict whitelist):
    {"call": "gen_custom" | "gen_video_animatediff", "kwargs": {...}, "output_path": str | None,
     "force_local": bool}
and in `route`, the resource_policy decision it was deferred with (its `kind` is what the
scheduler re-decides with every tick).

Only records with `not_before` set are ever picked up. image_api's immediate jobs are queued too,
but they belong to its executor thread; touching them here would race it. The one exception is
adopt_orphans(): queued jobs left behind by a previous image_api process, whose executor died
with it, are given a not_before so this picks them up instead of leaving them queued forever.
"""

import datetime
import json
import os
import shutil
import threading
import time
import traceback
import uuid

import job_contracts as jc
import job_store as js
import resource_policy as rp

DEFAULT_STORE_DIR = os.environ.get(
    "IMAGE_API_JOB_STORE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "_jobs"),
)
TICK_SECONDS = 30
INPUTS_DIRNAME = "_inputs"
CALLS = ("gen_custom", "gen_video_animatediff")


def _resolve_call(name):
    # Imported late: generate_character drags in the whole CLI, and job_scheduler has to stay
    # importable by tests and by image_api's startup path without it.
    import generate_character as gc

    return {"gen_custom": gc.gen_custom, "gen_video_animatediff": gc.gen_video_animatediff}[name]


def call_spec(call, kwargs, *, output_path=None, force_local=False):
    if call not in CALLS:
        raise ValueError(f"unknown deferred call {call!r}; choices: {list(CALLS)}")
    try:
        json.dumps(kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"deferred job arguments must be JSON-serialisable: {exc}") from None
    return {"call": call, "kwargs": dict(kwargs), "output_path": output_path, "force_local": bool(force_local)}


def enqueue(service, owner_id, *, call, kwargs, mode, media_kind, catalog_id, params, decision,
            output_path=None, input_keys=(), force_local=False):
    """Record a deferred job. Files named by `input_keys` (an uploaded face reference, a custom
    anchor) are copied under the store first: a Gradio upload lives in a temp directory that may be
    gone by 23:00, and a job that fails at its start time because its input vanished is exactly the
    kind of surprise deferring is supposed to avoid."""
    job_id = uuid.uuid4().hex
    kwargs = dict(kwargs)
    for key in input_keys:
        source = kwargs.get(key)
        if source and os.path.isfile(source):
            folder = os.path.join(service.store.root, INPUTS_DIRNAME, job_id)
            os.makedirs(folder, exist_ok=True)
            target = os.path.join(folder, key + os.path.splitext(source)[1].lower())
            shutil.copy2(source, target)
            kwargs[key] = target
    return service.create(
        owner_id, mode=mode, catalog_id=catalog_id, params=params, media_kind=media_kind,
        request=call_spec(call, kwargs, output_path=output_path, force_local=force_local),
        job_id=job_id, not_before=decision.not_before, route=decision.to_dict(),
    )


def adopt_orphans(store, *, before, clock=time.time):
    """Give queued, never-deferred jobs created before `before` a not_before, so the scheduler
    runs them. Only jobs that carry a runnable call spec are adopted. Returns the adopted ids."""
    adopted = []
    for owner_id, job_id in store.list_unterminal(limit=10_000):
        record = store.get(owner_id, job_id)
        if (record is None or record["status"] != jc.QUEUED or record.get("not_before") is not None
                or record["created_at"] >= before or (record.get("request") or {}).get("call") not in CALLS):
            continue
        try:
            store.update(owner_id, job_id, lambda r: r.update(not_before=clock()), expect_status=jc.QUEUED)
            adopted.append(job_id)
        except js.StaleWriteError:
            continue
    return adopted


def _route_signature(route):
    """What counts as a change worth writing. Not the reasons or notes: those carry live readings
    ("GPU 83°C") and would rewrite every waiting record every tick. The stored reasons are
    therefore the ones from when the route last changed, which is what the GUI labels them as."""
    route = route or {}
    return (route.get("route"), route.get("not_before"), route.get("cloud_target"))


class Scheduler:
    def __init__(self, service, *, settings_fn=rp.load_settings, probe_fn=rp.probe, clock=time.time,
                 now_fn=datetime.datetime.now, ensure_backend=None, resolve_call=_resolve_call, log=print):
        self.service = service
        self.store = service.store
        self._settings_fn = settings_fn
        self._probe_fn = probe_fn
        self._clock = clock
        self._now_fn = now_fn
        self._ensure_backend = ensure_backend
        self._resolve_call = resolve_call
        self._log = log
        self._stop = threading.Event()
        self._thread = None

    # --- listing --------------------------------------------------------------------------------

    def deferred(self, owner_id=None):
        """Queued jobs with a not_before, earliest first: [(owner_id, record)]."""
        rows = []
        for owner, job_id in self.store.list_unterminal(limit=10_000):
            if owner_id is not None and owner != owner_id:
                continue
            record = self.store.get(owner, job_id)
            if record and record["status"] == jc.QUEUED and record.get("not_before") is not None:
                rows.append((owner, record))
        rows.sort(key=lambda row: row[1]["not_before"])
        return rows

    def run_now(self, owner_id, job_id):
        """Operator override from the GUI: start at the next tick, ignoring the window and advice."""
        def bring_forward(record):
            record["not_before"] = self._clock()
            record["request"] = {**(record.get("request") or {}), "force_local": True}

        return self.store.update(owner_id, job_id, bring_forward, expect_status=jc.QUEUED)

    # --- one pass -------------------------------------------------------------------------------

    def tick(self):
        """Look at every due job; run at most one. Returns [(job_id, what_happened)]."""
        if rp.local_jobs_running():
            return [("*", "busy")]   # a GUI/API job is on the card right now; next tick
        now = self._clock()
        due = [(owner, r) for owner, r in self.deferred() if r["not_before"] <= now]
        if not due:
            return []
        settings = self._settings_fn()
        snap = self._probe_fn()
        outcomes = []
        for owner, record in due:
            spec = record.get("request") or {}
            kind = (record.get("route") or {}).get("kind")
            if spec.get("call") not in CALLS or kind not in rp.JOB_COSTS:
                self.service._conclude_failure(owner, record["job_id"], "deferred job has no runnable call spec",
                                               kind="unschedulable")
                outcomes.append((record["job_id"], "unschedulable"))
                continue
            waited = now - record["not_before"]
            decision = rp.decide(kind, settings, snap, self._now_fn(),
                                 est_seconds=(record.get("route") or {}).get("est_seconds"),
                                 force_local=bool(spec.get("force_local")),
                                 skip_wait=waited >= settings["cooldown_timeout_s"])
            if decision.route == rp.LOCAL:
                outcomes.append((record["job_id"], self._run(owner, record, decision)))
                return outcomes
            self._note(owner, record, decision)
            outcomes.append((record["job_id"], decision.route))
        return outcomes

    def _note(self, owner, record, decision):
        """Keep the record's route current so the GUI can say why it is still waiting. A DEFER
        (the window setting changed) moves not_before; WAIT and CLOUD leave it, so the job stays
        due and is looked at again next tick - a cloud suggestion at 23:00 with nobody at the desk
        just waits for the RAM to free up or for the operator to act."""
        new = decision.to_dict()
        if _route_signature(new) == _route_signature(record.get("route")):
            return

        def mutate(r):
            r["route"] = new
            if decision.route == rp.DEFER and decision.not_before:
                r["not_before"] = decision.not_before

        try:
            self.store.update(owner, record["job_id"], mutate, expect_status=jc.QUEUED)
        except (js.StaleWriteError, js.JobNotFound):
            pass

    def _run(self, owner, record, decision):
        spec = record["request"]
        job_id = record["job_id"]
        fn = self._resolve_call(spec["call"])
        kwargs = spec["kwargs"]
        if self._ensure_backend and rp.JOB_COSTS[decision.kind].uses_comfyui:
            self._ensure_backend()
        self._log(f"[scheduler] starting deferred job {job_id} ({decision.kind})")
        try:
            with rp.track(decision.kind, decision.est_seconds):
                self.service.submit_local(owner, job_id, lambda: fn(**kwargs), output_path=spec.get("output_path"))
        except js.StaleWriteError:
            return "taken"   # another process's scheduler won the queued -> submitting swap
        except BaseException as exc:  # noqa: BLE001 - submit_local has recorded it; keep the thread alive
            self._log(f"[scheduler] deferred job {job_id} failed: {type(exc).__name__}: {exc}")
            if isinstance(exc, KeyboardInterrupt):
                raise
            return "failed"
        return "ran"

    # --- background thread ----------------------------------------------------------------------

    def start(self, interval=TICK_SECONDS):
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()

        def loop():
            while not self._stop.wait(interval):
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 - a bad tick must not end scheduling for the session
                    self._log("[scheduler] tick failed:\n" + traceback.format_exc())

        self._thread = threading.Thread(target=loop, name="job-scheduler", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop.set()
