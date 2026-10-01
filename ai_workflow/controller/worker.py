"""The worker: the one process that moves jobs through their states. It runs inside a notebook kernel on the
compute runtime, next to ComfyUI.

    boot()    environment, ComfyUI source and packages, ComfyUI started, stale jobs recovered
    drain()   run what is runnable now, then return             (a notebook's "generate" cell)
    serve()   keep taking jobs until the queue has been idle     (a notebook's "wait for jobs" cell)
    shutdown()

Both entry points run every job through run_job(), so a job made by a notebook form and a job made through the
tool interface take exactly the same path.

How a job runs: claim (pending -> running) -> validate the values again -> verify and copy its input files ->
stage the workflow's models -> bind the graph -> safety check -> node check against the running ComfyUI ->
submit -> wait -> fetch the output -> validate it -> store it in the workspace -> completed. Any failure ends
that job as failed with a code; the worker goes on to the next one.

A job this runtime cannot run (its dependency is not finished, or the GPU is smaller than the workflow needs)
stays pending and is listed under `waiting` in logs/worker_state.json: opening the right notebook later picks
it up. It is not failed for being asked of the wrong machine.

logs/worker_state.json is rewritten every few seconds with an increasing `seq`. It is how the tool interface
knows a worker is alive, and how a second notebook knows not to start a second one.
"""

import json
import shutil
import socket
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import WorkflowError, jobs, params, safety, validate
from .comfy import VramSampler, execution_error, execution_seconds
from .compute import WORKER_STATE_FILE
from .registry import OUTPUT_KINDS, REGISTRY_FILE
from .tools import Context

MAX_INPUT_BYTES = 50 * 2**20
JOB_LEDGER = "logs/jobs.jsonl"
SESSION_LEDGER = "logs/sessions.jsonl"


class WorkerState:
    def __init__(self, storage, session, *, clock=time.time):
        self.storage = storage
        self.clock = clock
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.data = {
            "session": session,
            "seq": 0,
            "worker_state": "booting",
            "started_at": jobs.now_iso(),
            "gpu": {},
            "current_job": None,
            "serving": False,
            "waiting": {},
            "setup_actions": {},
            "jobs_done": 0,
        }

    def update(self, **fields):
        with self.lock:
            self.data.update(fields)
            self.write()

    def write(self):
        with self.lock:
            self.data["seq"] += 1
            self.data["updated_epoch"] = self.clock()
            self.data["updated_at"] = jobs.now_iso()
            try:
                self.storage.write_json(WORKER_STATE_FILE, self.data)
            except OSError:
                pass  # a hiccup of the mounted folder must not stop a job; the next beat writes again

    def start_heartbeat(self, every_seconds):
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop = threading.Event()  # a worker that was shut down may be booted again in the same kernel

        def beat():
            while not self.stop.wait(every_seconds):
                self.write()

        self.thread = threading.Thread(target=beat, name="aiwf-heartbeat", daemon=True)
        self.thread.start()

    def stop_heartbeat(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


class Worker:
    def __init__(
        self,
        storage,
        env,
        *,
        session=None,
        clock=time.time,
        sleep=time.sleep,
        echo=print,
        prober=validate.probe_video,
        previewer=validate.make_preview,
        flush_fn=None,
        release_fn=None,
    ):
        self.storage = storage
        self.env = env
        self.clock = clock
        self.sleep = sleep
        self.echo = echo
        self.prober = prober
        self.previewer = previewer
        self.flush_fn = flush_fn
        self.release_fn = release_fn
        self.session = session or "%s-%s" % (socket.gethostname()[:20], uuid.uuid4().hex[:8])
        self.ctx = Context(storage)
        self.registry_stamp = self._registry_stamp()
        self.cfg = dict(self.ctx.config.get("worker", {}))
        self.state = WorkerState(storage, self.session, clock=clock)
        self.client_id = "aiwf-" + uuid.uuid4().hex[:12]
        self.t0 = clock()
        self.jobs_done = 0
        self.booted = False

    def log(self, message):
        self.echo("[%s] %s" % (time.strftime("%H:%M:%S"), message))

    # -- boot ---------------------------------------------------------------------------------------

    def boot(self):
        if self.booted:
            return
        other = self.ctx.provider.worker_state(self.storage, now=self.clock())
        if other.get("alive") and other.get("session") != self.session:
            raise WorkflowError(
                "WORKER_ALREADY_RUNNING",
                "Another runtime (%s) is already serving this workspace." % other.get("session"),
                hint="Use that notebook, or stop its runtime first. Two runtimes cost twice the compute units.",
            )
        self.state.write()
        self.state.start_heartbeat(self.cfg.get("heartbeat_seconds", 15))
        env = self.env
        try:
            env.check_workspace()
            env.report()
            self.state.update(gpu=env.gpu)
            env.timed("comfyui", env.ensure_comfyui, self.ctx.registry.custom_nodes())
            env.timed("deps", env.ensure_deps)
            env.timed("start_comfyui", env.ensure_comfy_running)
            env.save_environment()
            self.recover_stale()
        except Exception as exc:
            self.state.update(worker_state="failed", error=_error(exc))
            self.state.stop_heartbeat()
            raise
        self.booted = True
        self.state.update(worker_state="ready", setup_actions=dict(env.actions), phase_seconds=dict(env.phase_seconds))
        self.log("worker ready: %s" % json.dumps(env.actions, ensure_ascii=False))

    def recover_stale(self):
        """A job left in running/ by a runtime that is gone goes back to pending once; the second time it is
        failed, so a job that kills its runtime cannot loop forever."""
        for job in self.ctx.store.list(jobs.RUNNING):
            if job.get("session") == self.session:
                continue
            if job.get("attempt", 1) >= self.cfg.get("max_attempts", 2):
                error = {
                    "code": "SESSION_LOST",
                    "message": "the runtime was lost %d times while this job ran" % job["attempt"],
                }
                self.ctx.store.finish(job, jobs.FAILED, error=dict(error, timestamp=jobs.now_iso()))
            else:
                self.ctx.store.requeue(job)
                self.log("job %s was left running by a lost runtime; queued again" % job["job_id"])

    # -- choosing the next job ----------------------------------------------------------------------

    def _registry_stamp(self):
        try:
            return self.storage.mtime(REGISTRY_FILE)
        except OSError:
            return None

    def reload_registry(self):
        """A workflow added to the workspace while the worker runs is picked up without a restart."""
        stamp = self._registry_stamp()
        if stamp != self.registry_stamp:
            self.registry_stamp = stamp
            try:
                self.ctx = Context(self.storage)
            except WorkflowError as exc:
                self.log("registry not reloaded: %s" % exc)

    def pick(self):
        """The oldest pending job this runtime can run now, or None. Cancels and broken dependencies are
        settled on the way."""
        self.reload_registry()
        store = self.ctx.store
        waiting = {}
        chosen = None
        for job in store.list(jobs.PENDING):
            job_id = job["job_id"]
            if store.cancel_requested(job_id):
                store.finish(job, jobs.CANCELLED, error=_cancelled("cancelled before it started"))
                continue
            try:
                workflow = self.ctx.registry.get(job["workflow"])
            except WorkflowError as exc:
                store.finish(job, jobs.FAILED, error=_error(exc))
                continue
            blocked = self._blocked(job, workflow)
            if blocked is False:
                continue  # settled as failed
            if blocked:
                waiting[job_id] = blocked
                continue
            if chosen is None:
                chosen = job
        if waiting != self.state.data.get("waiting"):
            self.state.update(waiting=waiting)
        return chosen

    def _blocked(self, job, workflow):
        """None = runnable; a string = why it waits; False = it was failed here."""
        for dep_id in job.get("depends_on", []):
            dep = self.ctx.store.find(dep_id)
            if dep is None or dep["status"] in (jobs.FAILED, jobs.CANCELLED):
                error = {
                    "code": "DEPENDENCY_FAILED",
                    "message": "job %s, which this job needs, did not complete" % dep_id,
                }
                self.ctx.store.finish(job, jobs.FAILED, error=dict(error, timestamp=jobs.now_iso()))
                return False
            if dep["status"] != jobs.COMPLETED:
                return "waiting for job %s" % dep_id
        gpu = workflow.get("gpu", {})
        have = self.env.gpu.get("vram_gib") or 0
        if gpu.get("min_vram_gib", 0) > have:
            return "needs a GPU with at least %s GiB VRAM; this runtime has %s" % (
                gpu["min_vram_gib"],
                have or "no GPU",
            )
        if gpu.get("require_bf16") and not self.env.gpu.get("bf16"):
            return "needs a GPU with bf16 support"
        return None

    # -- one job --------------------------------------------------------------------------------------

    def run_job(self, job):
        store = self.ctx.store
        job = store.claim(job, self.session)
        job_id = job["job_id"]
        self.state.update(worker_state="busy", current_job=job_id)
        self.log("job %s (%s) started" % (job_id, job["workflow"]))
        t0 = self.clock()
        extra = {"gpu": self.env.gpu.get("name"), "session_job_index": self.jobs_done + 1}
        try:
            workflow = self.ctx.registry.get(job["workflow"])
            values = {k: v for k, v in job["input"].items() if k not in workflow.inputs}
            values.update(job["parameters"])
            normal = params.validate(workflow, values)
            input_files = self.resolve_inputs(workflow, job)
            before = dict(self.env.actions)
            self.env.timed("models", self.env.ensure_models, workflow.models)
            extra["model_actions"] = {
                k: v for k, v in self.env.actions.items() if k.startswith("model:") and before.get(k) != v
            }
            self.env.ensure_comfy_running(workflow.get("comfyui_args", []))
            graph = params.bind(workflow, normal, job_id, input_files)
            safety.check_graph(workflow.name, workflow.profile, graph, params.texts(workflow, normal))
            problems = self.env.node_check(graph, workflow.models)
            if problems:
                raise WorkflowError("NODE_CHECK_FAILED", "; ".join(problems))
            self.storage.write_json("logs/jobs/%s/workflow.json" % job_id, graph)
            item, timing, vram_peak = self.render(job, workflow, graph)
            extra.update(timing, vram_peak_mib=vram_peak)
            output = self.store_output(job_id, workflow, normal, graph, item)
            job = store.finish(dict(job, **extra), jobs.COMPLETED, output=output)
        except WorkflowError as exc:
            status = jobs.CANCELLED if exc.code == "CANCELLED" else jobs.FAILED
            job = store.finish(dict(job, **extra), status, error=_error(exc))
        except Exception as exc:  # noqa: BLE001 - one job must never stop the worker
            job = store.finish(dict(job, **extra), jobs.FAILED, error=_error(exc, "JOB_FAILED", traceback.format_exc()))
        self.jobs_done += 1
        seconds = round(self.clock() - t0, 1)
        self.log("job %s %s in %s s" % (job_id, job["status"], seconds))
        self.storage.append_line(
            JOB_LEDGER,
            json.dumps(
                {
                    "job_id": job_id,
                    "workflow": job["workflow"],
                    "status": job["status"],
                    "session": self.session,
                    "seconds": seconds,
                    "generation_seconds": job.get("generation_seconds"),
                    "gpu": job.get("gpu"),
                    "vram_peak_mib": job.get("vram_peak_mib"),
                    "output": (job.get("output") or {}).get("path"),
                    "error": (job.get("error") or {}).get("code"),
                    "completed_at": job.get("completed_at"),
                },
                ensure_ascii=False,
            ),
        )
        self.state.update(worker_state="ready", current_job=None, jobs_done=self.jobs_done)
        return job

    def resolve_inputs(self, workflow, job):
        """Verify each input file where it comes from and copy it into ComfyUI's input folder."""
        files = {}
        for name, spec in workflow.inputs.items():
            ref = job["input"].get(name)
            if not isinstance(ref, dict) or ref.get("source") not in spec["sources"]:
                raise WorkflowError("INVALID_JOB", "input %r has no valid source" % name)
            if ref["source"] in ("generate", "job"):
                dep = self.ctx.store.find(jobs.check_job_id(ref.get("job_id")))
                dep_flow = self.ctx.registry.workflows.get(dep["workflow"]) if dep else None
                if dep is None or dep["status"] != jobs.COMPLETED or dep_flow is None or dep_flow.type != spec["kind"]:
                    raise WorkflowError(
                        "INPUT_NOT_READY",
                        "input %r: job %s has no completed %s" % (name, ref.get("job_id"), spec["kind"]),
                    )
                if dep_flow.profile == safety.PROFILE_SCREENED_PROMPT:
                    raise WorkflowError(
                        "UNSAFE_JOB", "input %r: job %s was not made with a negative prompt" % (name, dep["job_id"])
                    )
                rel = dep["output"]["path"]
                path = self.storage.local_path(rel)
                if path is None or not path.is_file() or validate.sha256_file(path) != dep["output"].get("sha256"):
                    raise WorkflowError(
                        "INPUT_CHANGED",
                        "input %r: %s is missing or is not the file job %s wrote" % (name, rel, dep["job_id"]),
                    )
            else:
                rel = str(ref.get("path") or "")
                if not rel.startswith("inputs/") or ".." in rel.split("/"):
                    raise WorkflowError("INVALID_JOB", "input %r: %r is not under inputs/" % (name, rel))
                if "file" in spec.get("attestation_sources", []) and ref.get("attested") is not True:
                    raise WorkflowError("ATTESTATION_REQUIRED", "input %r: the file's origin was not confirmed" % name)
                path = self.storage.local_path(rel)
                if path is None or not path.is_file():
                    raise WorkflowError("INPUT_NOT_FOUND", "input %r: %s is not in the workspace" % (name, rel))
                if path.stat().st_size > MAX_INPUT_BYTES:
                    raise WorkflowError("INVALID_INPUT", "input %r: %s is larger than 50 MB" % (name, rel))
            width, height = validate.image_size(path)
            comfy_name = "%s_%s%s" % (job["job_id"], name, Path(rel).suffix.lower())
            target = self.env.comfy_dir / "input" / comfy_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(path), str(target))
            files[name] = {"comfy_name": comfy_name, "width": width, "height": height, "path": rel}
        return files

    def render(self, job, workflow, graph):
        job_id = job["job_id"]
        out = workflow["output"]
        node, key, ext = str(out["node"]), out.get("list_key", "images"), "." + out["ext"]
        timeout = float(workflow.get("timeout_seconds", 1800))
        prompt_id = str(uuid.uuid4())
        http = self.env.http
        with VramSampler() as vram:
            t = self.clock()
            http.submit(graph, prompt_id, self.client_id)
            while True:
                if self.ctx.store.cancel_requested(job_id):
                    http.cancel(prompt_id)
                    raise WorkflowError("CANCELLED", "cancelled while running")
                if self.clock() - t > timeout:
                    http.cancel(prompt_id)
                    raise WorkflowError("TIMEOUT", "no result after %.0f s" % timeout)
                if self.env.process is not None and not self.env.process.running():
                    raise WorkflowError("COMFY_CRASHED", "ComfyUI exited mid-job:\n" + self.env.process.log_tail())
                try:
                    entry = http.history(prompt_id)
                except (OSError, ValueError):
                    entry = None
                if entry:
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        raise WorkflowError("COMFY_EXECUTION_ERROR", execution_error(status))
                    items = (entry.get("outputs", {}).get(node) or {}).get(key) or []
                    items = [i for i in items if str(i.get("filename", "")).lower().endswith(ext)]
                    if items:
                        timing = {
                            "generation_seconds": round(self.clock() - t, 2),
                            "comfy_execution_seconds": execution_seconds(status),
                        }
                        return items[0], timing, vram.peak
                    if status.get("completed"):
                        raise WorkflowError("OUTPUT_NOT_FOUND", "the save node reported no %s file" % ext)
                self.sleep(self.cfg.get("job_poll_seconds", 2))

    def store_output(self, job_id, workflow, values, graph, item):
        """Fetch the file from ComfyUI, validate it on local disk, and only then put it in the workspace."""
        out = workflow["output"]
        kind = out["kind"]
        staged = self.env.stage / ("%s.%s" % (job_id, out["ext"]))
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(self.env.http.view(item))
        expect = params.expected_output(workflow, values, graph)
        output = {"kind": kind, "bytes": staged.stat().st_size}
        if kind == "image":
            problems = validate.check_image(staged, expect)
            output.update(width=expect.get("width"), height=expect.get("height"))
        else:
            probe = self.prober(staged)
            problems = validate.check_video(probe, expect)
            output.update({k: v for k, v in validate.summarize_probe(probe).items() if v is not None})
        if problems:
            raise WorkflowError("INVALID_OUTPUT", "; ".join(problems))
        output["sha256"] = validate.sha256_file(staged)
        output["path"] = "%s/%s.%s" % (OUTPUT_KINDS[kind], job_id, out["ext"])
        self.storage.copy_in(staged, output["path"])
        if kind == "video":
            preview = self.env.stage / ("%s.jpg" % job_id)
            if self.previewer(staged, preview):
                output["preview"] = "outputs/previews/%s.jpg" % job_id
                self.storage.copy_in(preview, output["preview"])
                preview.unlink()
        staged.unlink()
        return output

    # -- loops ----------------------------------------------------------------------------------------

    def drain(self, until=None):
        """Run every job that can run now. With `until`, stop as soon as that job has ended."""
        self.boot()
        done = []
        while True:
            job = self.pick()
            if job is None:
                return done
            done.append(self.run_job(job))
            if until is not None:
                target = self.ctx.store.find(until)
                if target is None or target["status"] in jobs.TERMINAL:
                    return done

    def serve(self, idle_timeout_seconds=None):
        """Take jobs until the queue has been idle for idle_timeout_seconds. Returns the reason it stopped."""
        self.boot()
        idle_timeout = (
            self.cfg.get("idle_timeout_seconds", 600) if idle_timeout_seconds is None else idle_timeout_seconds
        )
        idle_since = self.clock()
        self.state.update(serving=True)
        try:
            while True:
                if self.clock() - self.t0 >= self.cfg.get("max_session_seconds", 10800):
                    return "max_session"
                job = self.pick()
                if job is not None:
                    self.run_job(job)
                    idle_since = self.clock()
                    continue
                idle = self.clock() - idle_since
                self.state.update(worker_state="idle", idle_seconds=round(idle, 1))
                if idle >= idle_timeout:
                    return "idle_timeout"
                self.sleep(self.cfg.get("poll_interval_seconds", 5))
        except KeyboardInterrupt:
            return "interrupted"
        finally:
            self.state.update(serving=False)

    def shutdown(self, reason, release=False):
        """Stop ComfyUI, wait until downloaded models are in the workspace, write the session record, and -
        when asked - flush the mounted folder and give the runtime back so it stops costing compute units."""
        self.state.update(worker_state="stopping", shutdown_reason=reason)
        self.env.stop_comfy()
        sync = self.env.wait_for_sync()
        summary = {
            "session": self.session,
            "started_at": self.state.data.get("started_at"),
            "ended_at": jobs.now_iso(),
            "reason": reason,
            "jobs": self.jobs_done,
            "gpu": self.env.gpu.get("name"),
            "setup_actions": dict(self.env.actions),
            "phase_seconds": dict(self.env.phase_seconds),
            "model_seconds": dict(self.env.model_seconds),
            "model_sync": sync,
            "session_seconds": round(self.clock() - self.t0, 1),
        }
        try:
            self.storage.append_line(SESSION_LEDGER, json.dumps(summary, ensure_ascii=False, default=str))
        except OSError:
            pass
        self.state.update(worker_state="stopped", current_job=None, summary=summary)
        self.state.stop_heartbeat()
        self.booted = False
        if release:
            for fn in (self.flush_fn, self.release_fn):
                if fn is not None:
                    try:
                        fn()
                    except Exception as exc:  # noqa: BLE001 - report it; the person can still stop the runtime by hand
                        self.log("could not %s: %s" % (getattr(fn, "__name__", "release the runtime"), exc))
        return summary


def _error(exc, code=None, stack=None):
    record = {
        "code": code or getattr(exc, "code", None) or "JOB_FAILED",
        "message": str(exc)[-4000:],
        "error_type": type(exc).__name__,
        "timestamp": jobs.now_iso(),
    }
    if getattr(exc, "hint", None):
        record["hint"] = exc.hint
    if stack:
        record["stack_trace"] = stack[-6000:]
    return record


def _cancelled(message):
    return {"code": "CANCELLED", "message": message, "timestamp": jobs.now_iso()}
