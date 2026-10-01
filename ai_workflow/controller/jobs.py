"""Jobs: one JSON file per job, and the folder it sits in is its state.

    jobs/pending/   created, waiting for a worker
    jobs/running/   claimed by a worker
    jobs/completed/ finished, with an output
    jobs/failed/    finished without one (status "failed" or "cancelled")
    jobs/cancel/    cancel requests: <job_id>.request

Single-writer rule. The workspace is a folder synced between the local machine and the compute runtime, and a
sync client resolves two writers of one file by keeping both ("x (1).json"). So after a job file is created,
only the worker moves or rewrites it. Everyone else may add two kinds of NEW files and nothing more: a job in
jobs/pending/, and a cancel request in jobs/cancel/. JobStore's worker-only methods say so in their names'
docstrings; tools.py never calls them.

A job stores the workflow's name and the caller's values, not a node graph: the worker binds the graph from
the registry and writes what it actually submitted to logs/jobs/<job_id>/workflow.json.
"""

import re
import time
import uuid

from . import WorkflowError

SCHEMA = 1
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
STATUSES = (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
TERMINAL = (COMPLETED, FAILED, CANCELLED)
# Where each status lives. When a job is found in two folders (a move interrupted half way), the later
# folder in this order wins.
FOLDERS = ("pending", "running", "failed", "completed")
FOLDER_OF = {PENDING: "pending", RUNNING: "running", COMPLETED: "completed", FAILED: "failed", CANCELLED: "failed"}
JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")


def now_iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def new_job_id(now=None):
    return "job-%s-%s" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(now)), uuid.uuid4().hex[:6])


def check_job_id(job_id):
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise WorkflowError("INVALID_INPUT", "invalid job id %r" % (job_id,))
    return job_id


def new_job(job_id, *, workflow, task_type, job_input, parameters, source, depends_on=(), parent=None):
    return {
        "schema": SCHEMA,
        "job_id": job_id,
        "task_type": task_type,
        "workflow": workflow,
        "status": PENDING,
        "input": job_input,
        "parameters": parameters,
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "output": None,
        "error": None,
        "source": source,
        "depends_on": list(depends_on),
        "parent": parent,
        "attempt": 0,
        "session": None,
    }


class JobStore:
    def __init__(self, storage):
        self.storage = storage

    @staticmethod
    def _rel(folder, job_id):
        return "jobs/%s/%s.json" % (folder, check_job_id(job_id))

    # -- anyone ------------------------------------------------------------------------------------

    def create(self, job):
        rel = self._rel("pending", job["job_id"])
        if self.find(job["job_id"]) is not None:
            raise WorkflowError("INVALID_INPUT", "job %s already exists" % job["job_id"])
        self.storage.write_json(rel, job)
        return job

    def find(self, job_id):
        """The job, or None. The folder decides the state when two copies exist."""
        check_job_id(job_id)
        for folder in reversed(FOLDERS):
            job = self.storage.read_json(self._rel(folder, job_id))
            if isinstance(job, dict) and job.get("job_id") == job_id:
                return job
        return None

    def get(self, job_id):
        job = self.find(job_id)
        if job is None:
            raise WorkflowError("JOB_NOT_FOUND", "No job %s in this workspace." % job_id)
        return job

    def list(self, status=None):
        folders = FOLDERS if status is None else (FOLDER_OF[status],)
        seen = {}
        for folder in folders:  # later folders overwrite earlier ones: same precedence as find()
            for name in self.storage.list("jobs/%s" % folder, ".json"):
                job = self.storage.read_json("jobs/%s/%s" % (folder, name))
                if isinstance(job, dict) and job.get("job_id") == name[: -len(".json")]:
                    seen[job["job_id"]] = job
        jobs = [j for j in seen.values() if status is None or j.get("status") == status]
        return sorted(jobs, key=lambda j: (j.get("created_at") or "", j["job_id"]))

    def request_cancel(self, job_id):
        self.storage.write_text("jobs/cancel/%s.request" % check_job_id(job_id), now_iso())

    def cancel_requested(self, job_id):
        return self.storage.exists("jobs/cancel/%s.request" % check_job_id(job_id))

    # -- worker only (single-writer rule) ------------------------------------------------------------

    def claim(self, job, session):
        """Worker only. pending -> running."""
        job = dict(job, status=RUNNING, started_at=now_iso(), session=session, attempt=job.get("attempt", 0) + 1)
        self.storage.write_json(self._rel("running", job["job_id"]), job)
        self.storage.delete(self._rel("pending", job["job_id"]))
        return job

    def save_running(self, job):
        """Worker only. Rewrite the running copy (progress fields)."""
        self.storage.write_json(self._rel("running", job["job_id"]), job)

    def finish(self, job, status, *, output=None, error=None):
        """Worker only. -> completed / failed / cancelled. The terminal copy is written before the earlier
        copies are removed, so an interruption leaves two copies and find() picks the terminal one."""
        if status not in TERMINAL:
            raise WorkflowError("BAD_TRANSITION", "%s is not a terminal status" % status)
        if status == COMPLETED and not output:
            raise WorkflowError("BAD_TRANSITION", "a completed job must carry an output")
        job = dict(job, status=status, completed_at=now_iso(), output=output, error=error)
        self.storage.write_json(self._rel(FOLDER_OF[status], job["job_id"]), job)
        for folder in ("running", "pending"):
            self.storage.delete(self._rel(folder, job["job_id"]))
        self.storage.delete("jobs/cancel/%s.request" % job["job_id"])
        return job

    def requeue(self, job):
        """Worker only. running -> pending, for a job whose worker died. The attempt counter stays."""
        job = dict(job, status=PENDING, started_at=None, session=None)
        self.storage.write_json(self._rel("pending", job["job_id"]), job)
        self.storage.delete(self._rel("running", job["job_id"]))
        return job
