"""Where job records live, and the only place their invariants are enforced on write.

The interface is deliberately narrow and engine-agnostic. FileJobStore is the engine that ships
first, because it needs no account, no service and no daemon, and because the offline suite has
to run it in CI on two operating systems. A DynamoDB engine is the same interface with a
different body; tests/test_job_store_contract.py is the suite it would have to pass, so it
cannot arrive with its own softer definition of correctness.

The shape is borrowed from web/amplify/: a record keyed by job_id, a reverse index keyed by
provider_job_id (the Amplify table's byProviderJobId GSI), and terminal state arriving from a
provider rather than being polled for. What is *not* borrowed is the storage: see the locking
note on FileJobStore.

Two mechanisms carry almost all the weight:

  A lock makes read-modify-write atomic. Without it two threads both read submit_attempt 1 and
  both write 2.

  A compare-and-swap makes a *semantically late* writer lose even though it legitimately holds
  the lock. A provider response from submit attempt 1, arriving after a recovery pass already
  started attempt 2, must write nothing. Neither mechanism substitutes for the other.
"""

import hashlib
import json
import os
import socket
import time

import job_contracts as jc

LOCK_TIMEOUT_SECONDS = 5.0
LOCK_RETRY_SECONDS = 0.005
STALE_LOCK_SECONDS = 30.0

# os.replace onto a path another process has open raises on Windows: CPython's open() does not
# request FILE_SHARE_DELETE. Readers hold their handle for exactly one read(), so the window is
# sub-millisecond and a short bounded retry closes it. Failing loudly after that is the point -
# a silent skip here would lose a state transition.
REPLACE_RETRIES = 10
REPLACE_RETRY_SECONDS = 0.002

ORPHAN_TEMP_SECONDS = 300.0

# Same numbers image_api.py:71-72 has used all along, moved rather than redesigned.
MAX_RECORDS = 500
TTL_SECONDS = 6 * 3600

LOCAL_BUCKET = "local-outputs"


class JobStoreError(Exception):
    """Base for every store failure."""


class JobStoreLocked(JobStoreError):
    """The store lock could not be acquired, and the holder did not look stale."""


class StaleWriteError(JobStoreError):
    """A compare-and-swap lost. The caller is a late writer and must write nothing.

    Normal, not exceptional: it is what a provider response from a superseded submit attempt is
    supposed to do. Callers log it and carry on.
    """


class JobNotFound(JobStoreError):
    """No record with that id is visible to that owner."""


def _digest(value, length):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


class JobStore:
    """The engine-agnostic surface. Every method takes owner_id as its first positional argument
    and never reads it from a payload - that is what makes it impossible to forget and impossible
    for a browser or an HTTP body to choose.

    find_by_provider_job_id is the single exception, and the exception is structural rather than
    an oversight: a provider webhook arrives with no authenticated owner by construction. It
    returns the owner it resolved, and every call after it goes back through an owner-taking
    method.
    """

    def create(self, owner_id, record):
        raise NotImplementedError

    def get(self, owner_id, job_id):
        raise NotImplementedError

    def update(self, owner_id, job_id, mutate, *, expect_submit_attempt=None, expect_status=None):
        raise NotImplementedError

    def bind_provider_job_id(self, owner_id, job_id, provider, provider_job_id, *, expect_submit_attempt=None):
        raise NotImplementedError

    def find_by_provider_job_id(self, provider, provider_job_id):
        raise NotImplementedError

    def request_cancel(self, owner_id, job_id, *, at):
        raise NotImplementedError

    def list_by_owner(self, owner_id, *, status=None, limit=50):
        raise NotImplementedError

    def list_unterminal(self, *, updated_before=None, limit=100):
        raise NotImplementedError

    def purge(self, *, now, ttl_seconds=TTL_SECONDS, max_records=MAX_RECORDS):
        raise NotImplementedError


class FileJobStore(JobStore):
    """JSON files plus a single store-wide lockfile.

    Why a lockfile and os.replace rather than msvcrt.locking, which is the more obvious Windows
    answer:

      1. CI runs ubuntu-latest and windows-latest (.github/workflows/checks.yml:30). msvcrt is
         Windows-only, so it would need an fcntl twin and two tested code paths for one
         behaviour. O_CREAT|O_EXCL is identical on both.
      2. msvcrt.locking locks byte ranges of an open descriptor. Its advantage - the OS releases
         the lock when a process dies - does nothing about the worse failure, which is a process
         dying mid-write and leaving truncated JSON that every later reader chokes on. Writing a
         temp file and os.replace-ing it removes that entire class: a reader sees the old record
         or the new one, never half of either.
      3. At this scale - a Gradio thread, image_api's single worker thread, the odd CLI run, with
         comfyui_client._CLIENT_LOCK already serialising actual generation - a store-wide lock
         costs nothing, and per-row locking is multi-tenant ceremony this repo has no use for.

    clock and sleep are injectable so the stale-lock path can be tested without spending five
    real seconds, the same way cloud_video.wait_for_job already does it.
    """

    def __init__(self, root, *, clock=time.time, sleep=time.sleep):
        self.root = os.path.abspath(root)
        self._clock = clock
        self._sleep = sleep
        self._counter = 0
        for name in ("jobs", "index", "owners"):
            os.makedirs(os.path.join(self.root, name), exist_ok=True)

    # --- paths ----------------------------------------------------------------------------------

    def _job_path(self, job_id):
        return os.path.join(self.root, "jobs", f"{job_id}.json")

    def _index_path(self, provider, provider_job_id):
        # Provider ids are opaque and may contain characters NTFS will not accept in a filename,
        # so the name is a hash. The literal id is stored in the body, which is what turns a
        # prefix collision into a detected miss instead of a silent mis-route.
        return os.path.join(self.root, "index", provider, f"{_digest(provider_job_id, 32)}.json")

    def _owner_dir(self, owner_id):
        return os.path.join(self.root, "owners", _digest(owner_id, 16))

    @property
    def _lock_path(self):
        return os.path.join(self.root, ".lock")

    @property
    def reconciliation_log(self):
        return os.path.join(self.root, "manual_reconciliation.log")

    # --- locking --------------------------------------------------------------------------------

    def _acquire(self):
        deadline = self._clock() + LOCK_TIMEOUT_SECONDS
        broke_stale = False
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pass
            else:
                body = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "acquired_at": self._clock()})
                os.write(fd, body.encode("utf-8"))
                os.close(fd)
                return
            if self._clock() >= deadline:
                if broke_stale or not self._break_stale_lock():
                    raise JobStoreLocked(f"could not acquire {self._lock_path} within {LOCK_TIMEOUT_SECONDS}s")
                broke_stale = True
                deadline = self._clock() + LOCK_TIMEOUT_SECONDS
            self._sleep(LOCK_RETRY_SECONDS)

    def _break_stale_lock(self):
        """Remove a lock whose holder died. Returns True when it actually removed one.

        A lock with an unreadable or missing body is treated as stale: it means a writer died
        between creating the file and describing itself, which is precisely the case that would
        otherwise wedge the store forever.
        """
        try:
            with open(self._lock_path, "rb") as handle:
                held = json.loads(handle.read().decode("utf-8"))
            acquired_at = float(held["acquired_at"])
        except (OSError, ValueError, KeyError, TypeError):
            acquired_at = None
        if acquired_at is not None and self._clock() - acquired_at < STALE_LOCK_SECONDS:
            return False
        self._log_reconciliation(f"broke stale lock held since {acquired_at}")
        try:
            os.remove(self._lock_path)
        except FileNotFoundError:
            pass
        return True

    def _release(self):
        try:
            os.remove(self._lock_path)
        except FileNotFoundError:
            pass

    class _Lock:
        def __init__(self, store):
            self.store = store

        def __enter__(self):
            self.store._acquire()
            return self.store

        def __exit__(self, *exc_info):
            self.store._release()
            return False

    def _locked(self):
        return FileJobStore._Lock(self)

    def _log_reconciliation(self, message):
        try:
            with open(self.reconciliation_log, "a", encoding="utf-8") as handle:
                handle.write(f"{self._clock():.3f}\t{message}\n")
        except OSError:
            pass  # a store that cannot log must still be able to work

    # --- atomic file IO -------------------------------------------------------------------------

    def _write_atomic(self, path, text):
        self._counter += 1
        # Same directory as the target, so os.replace stays on one volume - which is the only
        # condition under which it is atomic.
        tmp = f"{path}.{os.getpid()}.{self._counter}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == REPLACE_RETRIES - 1:
                    os.remove(tmp)
                    raise
                self._sleep(REPLACE_RETRY_SECONDS)

    def _read_json(self, path):
        """Read and close immediately. Readers never take the lock and must never hold the handle
        across a parse - on Windows that is what turns a concurrent writer's os.replace into a
        sharing violation."""
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return None

    def _sweep_orphan_temps(self):
        # time.time(), not self._clock(): an mtime is a wall-clock fact the OS wrote, so it can
        # only be compared against wall clock. The injected clock exists for lock and record
        # timing, and mixing the two bases makes every temp look either ancient or brand new
        # depending on which epoch the fake clock started from.
        now = time.time()
        jobs_dir = os.path.join(self.root, "jobs")
        for name in os.listdir(jobs_dir):
            if not name.endswith(".tmp"):
                continue
            path = os.path.join(jobs_dir, name)
            try:
                if now - os.path.getmtime(path) > ORPHAN_TEMP_SECONDS:
                    os.remove(path)
            except OSError:
                pass

    # --- reads ----------------------------------------------------------------------------------

    def get(self, owner_id, job_id):
        """Return the record, or None when it does not exist *or* belongs to somebody else.

        A wrong owner is None rather than an error on purpose: distinguishing "no such job" from
        "not yours" tells a caller whether an id exists, which is the whole content of an
        enumeration oracle.
        """
        data = self._read_json(self._job_path(job_id))
        if data is None or data.get("owner_id") != owner_id:
            return None
        return jc.validate_record(data)

    def _read_any_owner(self, job_id):
        data = self._read_json(self._job_path(job_id))
        return jc.validate_record(data) if data is not None else None

    def find_by_provider_job_id(self, provider, provider_job_id):
        entry = self._read_json(self._index_path(provider, provider_job_id))
        if entry is None:
            return None
        # The name is a truncated hash; the body carries the literal id. Comparing it is what
        # makes a prefix collision a miss rather than a silent delivery to the wrong job.
        if entry.get("provider_job_id") != provider_job_id:
            self._log_reconciliation(f"index prefix collision for {provider}:{provider_job_id}")
            return None
        return entry["owner_id"], entry["job_id"]

    def list_by_owner(self, owner_id, *, status=None, limit=50):
        owner_dir = self._owner_dir(owner_id)
        if not os.path.isdir(owner_dir):
            return []
        records = []
        for job_id in os.listdir(owner_dir):
            record = self.get(owner_id, job_id)
            if record is None or (status is not None and record["status"] != status):
                continue
            records.append(record)
        records.sort(key=lambda r: r["created_at"], reverse=True)
        return records[:limit]

    def list_unterminal(self, *, updated_before=None, limit=100):
        """Everything a reconciler still owes an answer on. Returns (owner_id, job_id) pairs so
        the caller goes back through an owner-taking method to touch them."""
        pending = []
        jobs_dir = os.path.join(self.root, "jobs")
        for name in sorted(os.listdir(jobs_dir)):
            if not name.endswith(".json"):
                continue
            data = self._read_json(os.path.join(jobs_dir, name))
            if data is None or data.get("status") in jc.TERMINAL_STATUSES:
                continue
            if updated_before is not None and data.get("updated_at", 0) >= updated_before:
                continue
            pending.append((data["owner_id"], data["job_id"]))
            if len(pending) >= limit:
                break
        return pending

    # --- writes ---------------------------------------------------------------------------------

    def create(self, owner_id, record):
        jc.validate_record(record)
        if record["owner_id"] != owner_id:
            raise JobStoreError("record owner_id does not match the owner creating it")
        with self._locked():
            self._sweep_orphan_temps()
            path = self._job_path(record["job_id"])
            if os.path.exists(path):
                raise JobStoreError(f"job {record['job_id']} already exists")
            self._write_atomic(path, jc.dumps(record))
            owner_dir = self._owner_dir(owner_id)
            os.makedirs(owner_dir, exist_ok=True)
            with open(os.path.join(owner_dir, record["job_id"]), "wb"):
                pass
        return record

    def update(self, owner_id, job_id, mutate, *, expect_submit_attempt=None, expect_status=None):
        """Read-modify-write under the lock, guarded by a compare-and-swap.

        `mutate` receives the current record and mutates it in place. It runs inside the lock, so
        it must not do IO or block - anything slow belongs outside, with its result passed in.
        """
        with self._locked():
            record = self._read_any_owner(job_id)
            if record is None or record["owner_id"] != owner_id:
                raise JobNotFound(f"job {job_id} is not visible to owner {owner_id!r}")
            if expect_submit_attempt is not None and record["submit_attempt"] != expect_submit_attempt:
                raise StaleWriteError(
                    f"job {job_id} is on submit_attempt {record['submit_attempt']}, "
                    f"caller expected {expect_submit_attempt}"
                )
            if expect_status is not None:
                allowed = {expect_status} if isinstance(expect_status, str) else set(expect_status)
                if record["status"] not in allowed:
                    raise StaleWriteError(
                        f"job {job_id} is {record['status']!r}, caller expected one of {sorted(allowed)}"
                    )
            before = record["status"]
            mutate(record)
            jc.require_transition(before, record["status"])
            record["updated_at"] = self._clock()
            self._write_atomic(self._job_path(job_id), jc.dumps(record))
            return record

    def bind_provider_job_id(self, owner_id, job_id, provider, provider_job_id, *, expect_submit_attempt=None):
        """Attach a provider's id and make it routable.

        The index is written BEFORE the record gains the id, and the order is the whole point. A
        crash between the two writes leaves an index entry pointing at a record that does not yet
        name its provider job - recoverable, because a webhook still routes and its own
        compare-and-swap can fill in both. The reverse order would leave a provider job whose
        webhook can never be routed to anything, which nothing can recover.
        """
        if provider not in jc.PROVIDERS:
            raise JobStoreError(f"unknown provider {provider!r}")
        with self._locked():
            record = self._read_any_owner(job_id)
            if record is None or record["owner_id"] != owner_id:
                raise JobNotFound(f"job {job_id} is not visible to owner {owner_id!r}")
            if expect_submit_attempt is not None and record["submit_attempt"] != expect_submit_attempt:
                raise StaleWriteError(
                    f"job {job_id} is on submit_attempt {record['submit_attempt']}, "
                    f"caller expected {expect_submit_attempt}"
                )
            entry = {"owner_id": owner_id, "job_id": job_id, "provider_job_id": provider_job_id}
            self._write_atomic(self._index_path(provider, provider_job_id), json.dumps(entry, sort_keys=True))
            record["provider"] = provider
            record["provider_job_id"] = provider_job_id
            record["updated_at"] = self._clock()
            self._write_atomic(self._job_path(job_id), jc.dumps(record))
            return record

    def request_cancel(self, owner_id, job_id, *, at):
        """Record the intent to cancel without touching status.

        Cancellation is a timestamp rather than a state so it cannot be lost when the status
        advances underneath it: a cancel that arrives while a job is moving submitting -> running
        still stands, and the poll loop picks it up on its next pass.
        """
        with self._locked():
            record = self._read_any_owner(job_id)
            if record is None or record["owner_id"] != owner_id:
                raise JobNotFound(f"job {job_id} is not visible to owner {owner_id!r}")
            if record["cancel_requested_at"] is None:
                record["cancel_requested_at"] = at
                record["updated_at"] = self._clock()
                self._write_atomic(self._job_path(job_id), jc.dumps(record))
            return record

    def purge(self, *, now, ttl_seconds=TTL_SECONDS, max_records=MAX_RECORDS):
        """Drop old finished records. Same policy as image_api._evict_jobs_locked:160 - TTL first,
        regardless of count, then oldest-finished-first while still over the cap - and the same
        refusal to ever evict a job that has not finished, so a slow generation is never forgotten
        out from under a polling client."""
        removed = 0
        with self._locked():
            records = []
            jobs_dir = os.path.join(self.root, "jobs")
            for name in sorted(os.listdir(jobs_dir)):
                if not name.endswith(".json"):
                    continue
                data = self._read_json(os.path.join(jobs_dir, name))
                if data is not None:
                    records.append(data)
            terminal = [r for r in records if r["status"] in jc.TERMINAL_STATUSES]
            for record in terminal:
                if now - record["created_at"] > ttl_seconds:
                    self._delete_locked(record)
                    records.remove(record)
                    removed += 1
            survivors = [r for r in records if r["status"] in jc.TERMINAL_STATUSES]
            survivors.sort(key=lambda r: r["created_at"])
            while len(records) > max_records and survivors:
                victim = survivors.pop(0)
                self._delete_locked(victim)
                records.remove(victim)
                removed += 1
        return removed

    def _delete_locked(self, record):
        for path in (self._job_path(record["job_id"]), os.path.join(self._owner_dir(record["owner_id"]),
                                                                    record["job_id"])):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        if record.get("provider") and record.get("provider_job_id"):
            try:
                os.remove(self._index_path(record["provider"], record["provider_job_id"]))
            except FileNotFoundError:
                pass


def resolve_artifact(owner_id, artifact, *, local_root=None):
    """Turn an artifact into something a caller can actually open, tagged with what it is.

    Tagged rather than uniform on purpose. A local artifact is a filesystem path; wrapping it in
    a signed-URL shape would be a lie with an expiry field, and would break the image_path
    contract image_api's caller already depends on. A remote artifact is a short-lived grant that
    has to be minted, which is a different operation with different failure modes, and a caller
    that cannot tell them apart will eventually cache one as the other.
    """
    data = artifact.to_dict() if isinstance(artifact, jc.Artifact) else dict(artifact)
    jc.Artifact.from_dict(data)  # re-validate: the key is about to be joined onto a real path
    if data["bucket"] == LOCAL_BUCKET:
        root = local_root or os.path.join(os.getcwd(), "outputs")
        return {"kind": "path", "path": os.path.join(root, *data["key"].split("/"))}
    return {"kind": "remote", "bucket": data["bucket"], "key": data["key"], "owner_id": owner_id}
