"""Runs ON the Colab VM, started by the local controller with `colab exec -f`. Keep this file ASCII-only.

A persistent Z-Image worker: it boots once per Colab session, then takes txt2img jobs from an inbox until
the queue has been empty for worker.idle_timeout_seconds, then shuts ComfyUI down, flushes Google Drive
and returns. The local controller (scripts/zimage_colab.py) is what releases the VM (`colab stop`):
google.colab.runtime.unassign() goes through a notebook frontend that a CLI-driven runtime does not have.

Why it is shaped this way:

- Persistent storage is Google Drive, mounted by the controller (`colab drivemount`) before this runs.
  Everything expensive lives there once: the pinned ComfyUI source as a tarball, the pip packages the
  Colab image lacks as a tarball of a PYTHONUSERBASE tree (keyed by a fingerprint of the runtime and
  the requirements), and the model files with a manifest. A new session only extracts two archives and
  copies the models to local disk - no git clone, no pip install, no model download - unless a
  version check says something changed. The Colab FAQ recommends archives over many-file folders.
- Colab asks for Drive consent on every new VM (measured 2026-10-01: the grant is per runtime). When no
  one gave it, the controller passes persistence "ephemeral" and persistent_root is a VM-local dir: the
  same checks then find nothing and the session installs and downloads from the pinned public sources.
- The control plane is files under /content/zimg, read and written by the controller through the
  Jupyter Contents API (`colab upload/download`), which works while this cell keeps the kernel busy.
  The kernel stays busy on purpose: Colab keeps a VM alive while a kernel execution is active.
- run/state.json is rewritten every few seconds by a heartbeat thread with an increasing `seq`, from the
  first line of main(), so a long silent step (a 20 GB copy) never looks like a dead VM.
- One job never takes the worker down: every job runs inside its own try/except and the loop goes on.
- The checks on each job's graph (cfg floor, age-safety negative) are a trust boundary: they raise,
  never clamp. They duplicate the local ones on purpose; tests pin both lists to the project constants.

No __future__ import: `colab exec --env` prepends code to this file, and one would then be a SyntaxError.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

RUN_FILE_ENV = "ZIMG_RUN_FILE"
WORKER_VERSION = 1

# Pinned copies of generate_character.AGE_SAFETY_NEGATIVE and comfyui_client.SAFETY_MIN_CFG. This file
# runs on the VM without the repo, so it cannot import them; tests/test_zimage_colab_prompt.py asserts
# they are equal to the project constants.
AGE_TERMS = ("child", "children", "kid", "minor", "teen", "teenager", "underage", "young girl")
MIN_CFG = 1.5

QUEUED = "QUEUED"
RUNNING = "RUNNING"
GENERATED = "GENERATED"
VALIDATING = "VALIDATING"
VALID = "VALID"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
CANCELLED = "CANCELLED"
INVALID_OUTPUT = "INVALID_OUTPUT"
TERMINAL = (COMPLETED, FAILED, TIMEOUT, CANCELLED, INVALID_OUTPUT)

# Worker states written to run/state.json.
BOOTING = "BOOTING"
READY = "READY"
BUSY = "BUSY"
IDLE = "IDLE"
SHUTTING_DOWN = "SHUTTING_DOWN"
FLUSHED = "FLUSHED"
WORKER_FAILED = "FAILED"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
INBOX_PARSE_RETRIES = 5


class WorkerError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows only (the offline tests): a reader has the file open for a moment
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def sha256_file(path, block=8 * 2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def copy_large(src, dst, block=64 * 2**20):
    """Sequential large-block copy to a .partial name, then rename: a half-copied file never carries the
    final name, so a size check can trust it."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".partial")
    with open(src, "rb") as fi, open(partial, "wb") as fo:
        shutil.copyfileobj(fi, fo, block)
    os.replace(partial, dst)


# --- image validation (stdlib structure check + Pillow decode when available) -------------------


def png_structure(data):
    """(width, height, problems) from the raw bytes: signature, IHDR first, every chunk CRC, IDAT, IEND."""
    import struct
    import zlib

    problems = []
    if not data.startswith(PNG_SIGNATURE):
        return None, None, ["not a PNG (bad signature)"]
    pos = len(PNG_SIGNATURE)
    width = height = None
    seen_idat = seen_iend = False
    first = True
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
        end = pos + 8 + length + 4
        if end > len(data):
            problems.append("truncated chunk %r" % ctype)
            break
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : end])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            problems.append("bad CRC in chunk %r" % ctype)
        if first:
            if ctype != b"IHDR" or length != 13:
                problems.append("first chunk is not IHDR")
            else:
                width, height = struct.unpack(">II", body[:8])
            first = False
        if ctype == b"IDAT":
            seen_idat = True
        if ctype == b"IEND":
            seen_iend = True
            break
        pos = end
    if not seen_idat:
        problems.append("no IDAT chunk")
    if not seen_iend:
        problems.append("no IEND chunk (truncated)")
    return width, height, problems


def validate_image(path, width, height, fmt="png"):
    """Problems with an output image (empty list = valid). Pillow, when importable, also decodes it."""
    path = Path(path)
    if not path.is_file():
        return ["output file missing: %s" % path]
    size = path.stat().st_size
    if size == 0:
        return ["output file is empty"]
    if fmt != "png":
        return ["unsupported expected format %r" % fmt]
    data = path.read_bytes()
    w, h, problems = png_structure(data)
    if problems:
        return problems
    if (w, h) != (width, height):
        problems.append("resolution %sx%s, expected %sx%s" % (w, h, width, height))
    try:
        from PIL import Image
    except ImportError:
        return problems
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            if im.format != "PNG":
                problems.append("Pillow reads format %s, expected PNG" % im.format)
            im.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure means the file is not usable
        problems.append("Pillow cannot decode it: %s: %s" % (type(exc).__name__, exc))
    return problems


# --- safety and job checks (trust boundary: raise, never clamp) -----------------------------------


def check_graph(graph):
    """Raise WorkerError for a graph that would skip or weaken the safety negative."""
    if not isinstance(graph, dict) or not graph:
        raise WorkerError("INVALID_JOB", "graph is not a non-empty object")
    samplers = []
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
            raise WorkerError("INVALID_JOB", "node %s is not an API-format node" % node_id)
        cfg = node["inputs"].get("cfg")
        if cfg is not None:
            if not isinstance(cfg, (int, float)) or isinstance(cfg, bool) or cfg < MIN_CFG:
                raise WorkerError("UNSAFE_JOB", "node %s has cfg %r below the %s floor" % (node_id, cfg, MIN_CFG))
            samplers.append(node)
    if not samplers:
        raise WorkerError("INVALID_JOB", "graph has no sampler with a cfg input")
    for node in samplers:
        ref = node["inputs"].get("negative")
        if not (isinstance(ref, list) and ref and str(ref[0]) in graph):
            raise WorkerError("UNSAFE_JOB", "sampler has no negative conditioning node")
        text = graph[str(ref[0])].get("inputs", {}).get("text")
        if not isinstance(text, str):
            raise WorkerError("UNSAFE_JOB", "negative conditioning is not a text encoder")
        terms = {t.strip().lower() for t in text.split(",")}
        missing = [t for t in AGE_TERMS if t not in terms]
        if missing:
            raise WorkerError("UNSAFE_JOB", "negative prompt is missing the age-safety terms %s" % missing)


def check_payload(payload):
    for key in ("job_id", "graph", "width", "height", "save_node"):
        if key not in payload:
            raise WorkerError("INVALID_JOB", "job payload has no %r" % key)
    for key in ("width", "height"):
        if not isinstance(payload[key], int) or isinstance(payload[key], bool) or payload[key] <= 0:
            raise WorkerError("INVALID_JOB", "%s must be a positive integer" % key)
    if str(payload["save_node"]) not in payload["graph"]:
        raise WorkerError("INVALID_JOB", "save_node %r is not in the graph" % payload["save_node"])
    check_graph(payload["graph"])


def check_against_object_info(graph, object_info):
    """Every class_type installed, every input name declared (same rule as training/validate_workflow_nodes)."""
    problems = []
    for node_id, node in graph.items():
        cls = node.get("class_type")
        info = object_info.get(cls)
        if info is None:
            problems.append("node %s: class %s is not installed" % (node_id, cls))
            continue
        declared = set()
        for section in ("required", "optional", "hidden"):
            declared.update((info.get("input") or {}).get(section) or {})
        for name in node.get("inputs", {}):
            if name.split(".")[0] not in declared:
                problems.append("node %s (%s): input %r is not declared" % (node_id, cls, name))
    return problems


def combo_options(object_info, cls, field):
    try:
        spec = object_info[cls]["input"]["required"][field]
    except (KeyError, TypeError):
        return None
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return spec[0]
    if isinstance(spec, list) and len(spec) > 1 and spec[0] == "COMBO" and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return None


# --- state file + heartbeat ------------------------------------------------------------------------


class State:
    """run/state.json: the only thing the controller reads to follow the worker."""

    def __init__(self, path, *, clock=time.time, echo=print):
        self.path = Path(path)
        self.clock = clock
        self.echo = echo
        self.lock = threading.RLock()
        self.data = {
            "worker_version": WORKER_VERSION,
            "seq": 0,
            "worker_state": BOOTING,
            "phase": None,
            "phase_seconds": {},
            "setup_actions": {},
            "jobs": {},
            "rejected": {},
            "started_at": now_iso(),
        }
        self.stop = threading.Event()
        self.thread = None

    def update(self, **fields):
        with self.lock:
            self.data.update(fields)
            self.write()

    def job(self, job_id, **fields):
        with self.lock:
            self.data["jobs"].setdefault(job_id, {}).update(fields)
            self.write()

    def write(self):
        with self.lock:
            self.data["seq"] += 1
            self.data["updated_at"] = now_iso()
            write_json_atomic(self.path, self.data)

    def start_heartbeat(self, every_seconds, print_every_seconds=60):
        def beat():
            last_print = 0.0
            while not self.stop.wait(every_seconds):
                try:
                    self.write()
                except OSError:
                    pass
                if self.clock() - last_print >= print_every_seconds:
                    last_print = self.clock()
                    with self.lock:
                        brief = {k: self.data.get(k) for k in ("seq", "worker_state", "phase", "idle_seconds")}
                    self.echo("ZIMG_HEARTBEAT " + json.dumps(brief))

        self.thread = threading.Thread(target=beat, name="zimg-heartbeat", daemon=True)
        self.thread.start()

    def stop_heartbeat(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


# --- ComfyUI HTTP ----------------------------------------------------------------------------------


class Comfy:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def request(self, method, path, body=None, timeout=30, raw=False):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
        if raw:
            return payload
        return json.loads(payload.decode("utf-8")) if payload else {}

    def alive(self):
        try:
            self.request("GET", "/system_stats", timeout=3)
            return True
        except (OSError, ValueError):
            return False

    def submit(self, graph, prompt_id, client_id):
        try:
            return self.request("POST", "/prompt", {"prompt": graph, "client_id": client_id, "prompt_id": prompt_id})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[-2000:]
            raise WorkerError("COMFY_REJECTED", "ComfyUI refused the graph (%s): %s" % (exc.code, detail))

    def history(self, prompt_id):
        return self.request("GET", "/history/" + urllib.parse.quote(prompt_id)).get(prompt_id)

    def cancel(self, prompt_id):
        """Drop the prompt from the queue and interrupt it if it is the one running."""
        for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
            try:
                self.request("POST", path, body, timeout=10)
            except (OSError, ValueError):
                pass

    def view(self, image):
        query = urllib.parse.urlencode(
            {
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
        )
        return self.request("GET", "/view?" + query, timeout=120, raw=True)


class ComfyProcess:
    """The ComfyUI server subprocess, started from the local (extracted) checkout."""

    def __init__(self, comfy_dir, *, port, extra_paths_yaml, output_dir, log_path, pyuser, extra_args=()):
        self.comfy_dir = Path(comfy_dir)
        self.port = port
        self.extra_paths_yaml = extra_paths_yaml
        self.output_dir = output_dir
        self.log_path = Path(log_path)
        self.pyuser = pyuser
        self.extra_args = list(extra_args)
        self.proc = None

    def env(self):
        env = dict(os.environ)
        # PYTHONUSERBASE only here, not in the kernel: the archived packages are for ComfyUI alone.
        # HF_HUB_OFFLINE / PIP_NO_INDEX: a restart must not touch the network, and these make any
        # attempt fail loudly instead of quietly re-downloading.
        env.update(PYTHONUSERBASE=str(self.pyuser), HF_HUB_OFFLINE="1", PIP_NO_INDEX="1", PYTHONUNBUFFERED="1")
        return env

    def start(self):
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port)]
        cmd += ["--disable-auto-launch", "--extra-model-paths-config", str(self.extra_paths_yaml)]
        cmd += ["--output-directory", str(self.output_dir)] + self.extra_args
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            cmd, cwd=str(self.comfy_dir), env=self.env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        log.close()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, wait_seconds=20):
        if not self.running():
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def log_tail(self, chars=3000):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-chars:]
        except OSError:
            return ""


# --- GPU -------------------------------------------------------------------------------------------


def probe_gpu():
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_total_mib": int(props.total_memory // 2**20),
        "capability": "%d.%d" % (props.major, props.minor),
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def check_gpu(info, gpu_cfg):
    if not info.get("available"):
        raise WorkerError("GPU_NOT_SUPPORTED", "no CUDA GPU on this runtime")
    need = float(gpu_cfg["min_vram_gib"]) * 1024
    if info["vram_total_mib"] < need:
        raise WorkerError(
            "GPU_NOT_SUPPORTED",
            "%s has %d MiB VRAM, the configured minimum is %d MiB" % (info["name"], info["vram_total_mib"], need),
        )
    if gpu_cfg.get("require_bf16") and not info.get("bf16"):
        raise WorkerError("GPU_NOT_SUPPORTED", "%s has no bf16 support and gpu.require_bf16 is set" % info["name"])


class VramSampler:
    """Peak `nvidia-smi` memory.used while a job runs. ComfyUI is another process, so torch's own peak
    counters in this kernel would read zero."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak = None
        self.stop = threading.Event()
        self.thread = None

    def _read(self):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.split()[0])

    def __enter__(self):
        if not shutil.which("nvidia-smi"):
            return self

        def run():
            while not self.stop.is_set():
                try:
                    value = self._read()
                    self.peak = value if self.peak is None else max(self.peak, value)
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    pass
                self.stop.wait(self.interval)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


# --- the worker ------------------------------------------------------------------------------------


class Paths:
    def __init__(self, local_root, persistent_root):
        self.local = Path(local_root)
        self.persist = Path(persistent_root)
        self.state = self.local / "run" / "state.json"
        self.inbox = self.local / "inbox"
        self.cancel = self.local / "cancel"
        self.outbox = self.local / "outbox"
        self.comfy_out = self.local / "comfy_out"
        self.stage = self.local / "stage"
        self.comfy = self.local / "ComfyUI"
        self.pyuser = self.local / "pyuser"
        self.local_models = self.local / "models"
        self.comfy_log = self.local / "comfyui.log"
        self.extra_paths = self.local / "extra_model_paths.yaml"
        self.cache = self.persist / "cache"
        self.models = self.persist / "models"
        self.manifest = self.models / "manifest.json"
        self.environment = self.persist / "config" / "environment.json"
        self.outputs = self.persist / "outputs"
        self.logs = self.persist / "logs"

    def make(self):
        for d in (self.inbox, self.cancel, self.outbox, self.comfy_out, self.stage, self.local_models):
            d.mkdir(parents=True, exist_ok=True)
        for d in (self.cache, self.models, self.environment.parent, self.outputs, self.logs):
            d.mkdir(parents=True, exist_ok=True)


def hf_download(repo, filename, revision, local_dir):
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    # token=False: without it every call first waits on an HF_TOKEN lookup (measured 10 s each on H3).
    return hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(local_dir), token=False)


def url_download(url, dest, timeout=600):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(partial, "wb") as f:
        shutil.copyfileobj(resp, f, 8 * 2**20)
    os.replace(partial, dest)
    return dest


def drive_flush():
    from google.colab import drive

    drive.flush_and_unmount()


def run_cmd(cmd, env=None, timeout=3600):
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


class Worker:
    def __init__(
        self,
        cfg,
        *,
        clock=time.time,
        sleep=time.sleep,
        echo=print,
        gpu_probe=probe_gpu,
        hf_download_fn=hf_download,
        url_download_fn=url_download,
        runner=run_cmd,
        flush_fn=drive_flush,
        comfy_factory=None,
        base_freeze=None,
    ):
        self.cfg = cfg
        self.clock = clock
        self.sleep = sleep
        self.echo = echo
        self.gpu_probe = gpu_probe
        self.hf_download = hf_download_fn
        self.url_download = url_download_fn
        self.runner = runner
        self.flush_fn = flush_fn
        self.comfy_factory = comfy_factory or ComfyProcess
        self.base_freeze = base_freeze
        self.paths = Paths(cfg["local_root"], cfg["persistent_root"])
        # "drive": persistent_root is on Google Drive. "ephemeral": nobody gave this session's Drive consent
        # (Colab asks for it on every new VM), so persistent_root is a VM-local dir and the session downloads
        # from the pinned public sources instead - same checks, same code path, nothing archived or flushed.
        self.persistent = cfg.get("persistence", "drive") == "drive"
        self.state = State(self.paths.state, clock=clock, echo=echo)
        self.comfy_http = Comfy("http://127.0.0.1:%d" % cfg["comfyui"]["port"])
        self.comfy = None
        self.session = cfg.get("session")
        self.client_id = "zimg-" + uuid.uuid4().hex[:12]
        self.t0 = clock()
        self.gpu = {}
        self.env_record = {}
        self.actions = {}
        self.model_dir = None
        self.drive_sync = None
        self.queue = []  # payload dicts, oldest first
        self.seen = set()
        self.parse_failures = {}
        self.jobs_done = 0
        self.deps_installed_this_session = False

    # -- logging ----------------------------------------------------------------------------------

    def log(self, message):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), message)
        self.echo(line)
        try:
            self.paths.logs.mkdir(parents=True, exist_ok=True)
            with open(self.paths.logs / ("worker-%s.log" % (self.session or "local")), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def marker(self, kind, **fields):
        self.echo("ZIMG_%s %s" % (kind, json.dumps(fields, default=str)))

    def phase(self, name):
        """Context manager that times a bootstrap phase into state.json."""
        worker = self

        class _Phase:
            def __enter__(self_inner):
                self_inner.t = worker.clock()
                worker.state.update(phase=name)
                worker.log("phase %s" % name)
                return self_inner

            def __exit__(self_inner, *exc):
                seconds = round(worker.clock() - self_inner.t, 1)
                with worker.state.lock:
                    worker.state.data["phase_seconds"][name] = seconds
                worker.state.write()
                worker.marker("PHASE", name=name, seconds=seconds, ok=exc[0] is None)
                return False

        return _Phase()

    def action(self, component, what):
        self.actions[component] = what
        self.state.update(setup_actions=dict(self.actions))

    # -- bootstrap --------------------------------------------------------------------------------

    def boot(self):
        self.paths.local.mkdir(parents=True, exist_ok=True)
        with self.phase("GPU_CHECK"):
            self.gpu = self.gpu_probe()
            self.state.update(gpu=self.gpu)
            check_gpu(self.gpu, self.cfg["gpu"])
        with self.phase("DRIVE_CHECK"):
            if self.persistent:
                self.check_drive()
            else:
                self.paths.make()
            self.action("storage", "drive" if self.persistent else "ephemeral")
        with self.phase("ENV_CHECK"):
            self.env_record = read_json(self.paths.environment, {}) or {}
            self.state.update(previous_environment={k: self.env_record.get(k) for k in self.env_keys()})
        with self.phase("COMFYUI"):
            self.ensure_comfyui()
        with self.phase("DEPS"):
            self.ensure_deps()
        with self.phase("MODELS"):
            self.ensure_models()
        with self.phase("START_COMFYUI"):
            self.start_comfyui()
        with self.phase("NODE_CHECK"):
            self.node_check()
        self.save_environment()
        self.state.update(worker_state=READY, ready_at=now_iso())
        self.marker("READY", actions=self.actions, phase_seconds=self.state.data["phase_seconds"])

    @staticmethod
    def env_keys():
        return ("comfyui_commit", "z_image_revision", "deps_fingerprint", "python_version", "torch_version")

    def check_drive(self):
        drive_root = Path(self.cfg.get("drive_root", "/content/drive/MyDrive"))
        if not drive_root.is_dir():
            raise WorkerError("DRIVE_NOT_MOUNTED", "%s is not mounted (run `colab drivemount` first)" % drive_root)
        self.paths.make()
        probe = self.paths.persist / "config" / ".write_probe"
        probe.write_text(now_iso(), encoding="utf-8")
        if probe.read_text(encoding="utf-8") == "":
            raise WorkerError("DRIVE_NOT_WRITABLE", "could not read back a file written to %s" % probe.parent)
        # No Drive quota check: on the mount, disk_usage reports DriveFS's local cache disk, not the quota
        # (measured 2026-10-01: 225.8 GiB "total" = the VM's /content disk).
        try:
            local = shutil.disk_usage(str(self.paths.local))
            self.state.update(disk={"local_free_gib": round(local.free / 2**30, 1)})
        except OSError:
            pass

    def ensure_comfyui(self):
        c = self.cfg["comfyui"]
        commit = c["commit"]
        tarball = self.paths.cache / ("comfyui-%s.tar.gz" % commit)
        marker = self.paths.comfy / ".zimg_commit"
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == commit:
            self.action("comfyui", "present")
            return
        if tarball.is_file() and tarball.stat().st_size > 0:
            what = "extracted"
        else:
            self.log("ComfyUI %s (%s) is not on Drive yet: downloading the source tarball once" % (c["ref"], commit))
            staged = self.url_download(c["tarball_url"], self.paths.stage / tarball.name)
            copy_large(staged, tarball)
            what = "installed"
        if self.paths.comfy.exists():
            shutil.rmtree(self.paths.comfy)
        extract_tarball(tarball, self.paths.comfy)
        for node in self.cfg.get("custom_nodes", []):
            self.ensure_custom_node(node)
        marker.write_text(commit, encoding="utf-8")
        self.action("comfyui", what)

    def ensure_custom_node(self, node):
        name, commit = node["name"], node["commit"]
        tarball = self.paths.cache / ("node-%s-%s.tar.gz" % (name, commit))
        if not (tarball.is_file() and tarball.stat().st_size > 0):
            staged = self.url_download(node["tarball_url"], self.paths.stage / tarball.name)
            copy_large(staged, tarball)
            self.action("custom_node:" + name, "installed")
        else:
            self.action("custom_node:" + name, "extracted")
        extract_tarball(tarball, self.paths.comfy / "custom_nodes" / name)

    def requirement_files(self):
        files = [self.paths.comfy / "requirements.txt"]
        for node in self.cfg.get("custom_nodes", []):
            req = self.paths.comfy / "custom_nodes" / node["name"] / "requirements.txt"
            if req.is_file():
                files.append(req)
        return files

    def runtime_fingerprint(self):
        if self.base_freeze is None:
            code, out = self.runner([sys.executable, "-m", "pip", "freeze"], env=self.kernel_env())
            self.base_freeze = out if code == 0 else ""
        reqs = "\n".join(p.read_text(encoding="utf-8") for p in self.requirement_files() if p.is_file())
        parts = {
            "python": platform.python_version(),
            "torch": self.gpu.get("torch"),
            "cuda": self.gpu.get("cuda"),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
            "base_freeze": hashlib.sha256(self.base_freeze.encode("utf-8")).hexdigest(),
            "comfyui_commit": self.cfg["comfyui"]["commit"],
            "requirements": hashlib.sha256(reqs.encode("utf-8")).hexdigest(),
        }
        digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:16], parts

    @staticmethod
    def kernel_env():
        env = dict(os.environ)
        env.pop("PYTHONUSERBASE", None)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return env

    def ensure_deps(self, force=False):
        fingerprint, parts = self.runtime_fingerprint()
        self.deps_fingerprint, self.deps_parts = fingerprint, parts
        archive = self.paths.cache / ("pydeps-%s.tar" % fingerprint)
        done = self.paths.pyuser / ".zimg_fingerprint"
        if not force and done.is_file() and done.read_text(encoding="utf-8").strip() == fingerprint:
            self.action("deps", "present")
            return
        if not force and archive.is_file() and archive.stat().st_size > 0:
            if self.paths.pyuser.exists():
                shutil.rmtree(self.paths.pyuser)
            extract_tarball(archive, self.paths.pyuser, strip=False)
            done.write_text(fingerprint, encoding="utf-8")
            self.action("deps", "extracted")
            return
        previous = self.env_record.get("deps_fingerprint")
        self.log("installing ComfyUI's Python packages once for fingerprint %s (was %s)" % (fingerprint, previous))
        self.install_deps(archive, fingerprint)
        self.action("deps", "migrated" if previous and previous != fingerprint else "installed")

    def install_deps(self, archive, fingerprint):
        if self.paths.pyuser.exists():
            shutil.rmtree(self.paths.pyuser)
        self.paths.pyuser.mkdir(parents=True)
        constraints = self.paths.stage / "constraints.txt"
        constraints.write_text("\n".join(self.base_pins()) + "\n", encoding="utf-8")
        env = self.kernel_env()
        env["PYTHONUSERBASE"] = str(self.paths.pyuser)
        cmd = [sys.executable, "-m", "pip", "install", "--user", "--no-warn-script-location", "-c", str(constraints)]
        for req in self.requirement_files():
            cmd += ["-r", str(req)]
        code, out = self.runner(cmd, env=env)
        if code != 0:
            raise WorkerError("DEPS_INSTALL_FAILED", "pip install failed (exit %s): %s" % (code, out[-3000:]))
        (self.paths.pyuser / ".zimg_fingerprint").write_text(fingerprint, encoding="utf-8")
        self.deps_installed_this_session = True
        if not self.persistent:
            return  # nothing outlives an ephemeral session, so archiving would only cost time
        staged = self.paths.stage / archive.name
        make_tarball(self.paths.pyuser, staged)
        copy_large(staged, archive)

    @staticmethod
    def base_pins():
        """Pin the GPU stack to what the Colab image ships, so no requirement can archive a second torch."""
        from importlib import metadata

        pins = []
        for name in ("torch", "torchvision", "torchaudio", "numpy"):
            try:
                pins.append("%s==%s" % (name, metadata.version(name)))
            except metadata.PackageNotFoundError:
                pass
        return pins

    def model_specs(self):
        m = self.cfg["model"]
        return [dict(spec, repo=m["repo"], revision=m["revision"]) for spec in m["files"]]

    def ensure_models(self):
        manifest = read_json(self.paths.manifest, {}) or {}
        specs = self.model_specs()
        stage_local = bool(self.cfg["models"].get("stage_to_local", True))
        missing, present = [], []
        for spec in specs:
            drive_file = self.paths.models / spec["folder"] / spec["name"]
            entry = manifest.get(spec["name"]) or {}
            ok = (
                drive_file.is_file()
                and drive_file.stat().st_size == spec["size"]
                and entry.get("revision") == spec["revision"]
                and entry.get("sha256") == spec["sha256"]
            )
            (present if ok else missing).append(spec)
        if missing:
            need = sum(s["size"] for s in missing)
            self.check_space(need)
            self.download_models(missing, manifest)
            for spec in missing:
                self.action("model:" + spec["name"], "downloaded")
        # Downloaded files always run from the local copy (the Drive copy may still be uploading), so once
        # anything was downloaded the present files are staged next to them: ComfyUI gets one model dir.
        use_local = stage_local or bool(missing)
        staged_seconds = {}
        for spec in present:
            local = self.paths.local_models / spec["folder"] / spec["name"]
            if not use_local or (local.is_file() and local.stat().st_size == spec["size"]):
                self.action("model:" + spec["name"], "present")
                continue
            t = self.clock()
            copy_large(self.paths.models / spec["folder"] / spec["name"], local)
            staged_seconds[spec["name"]] = round(self.clock() - t, 1)
            self.log("staged %s from Drive in %.1f s" % (spec["name"], staged_seconds[spec["name"]]))
            self.action("model:" + spec["name"], "staged")
        if staged_seconds:
            self.state.update(model_stage_seconds=staged_seconds)
        self.model_dir = self.paths.local_models if use_local else self.paths.models

    def check_space(self, need_bytes):
        local = shutil.disk_usage(str(self.paths.local)).free
        # The local copy, plus (with Drive) DriveFS's own local cache of what is written to Drive, plus headroom.
        need = (2 if self.persistent else 1) * need_bytes + 10 * 2**30
        if local < need:
            raise WorkerError("DISK_FULL", "%.1f GiB free on /content, need %.1f GiB" % (local / 2**30, need / 2**30))

    def download_models(self, specs, manifest):
        import concurrent.futures

        def one(spec):
            t = self.clock()
            got = Path(self.hf_download(spec["repo"], spec["path_in_repo"], spec["revision"], self.paths.stage))
            seconds = self.clock() - t
            digest = sha256_file(got)
            if digest != spec["sha256"] or got.stat().st_size != spec["size"]:
                raise WorkerError(
                    "MODEL_CHECKSUM", "%s: sha256 %s, expected %s" % (spec["name"], digest, spec["sha256"])
                )
            local = self.paths.local_models / spec["folder"] / spec["name"]
            local.parent.mkdir(parents=True, exist_ok=True)
            os.replace(got, local)
            gib = spec["size"] / 2**30
            self.marker("MODEL", name=spec["name"], gib=round(gib, 2), seconds=round(seconds, 1))
            self.log(
                "downloaded %s: %.2f GiB in %.1f s (%.0f MB/s), sha256 ok"
                % (spec["name"], gib, seconds, spec["size"] / 1e6 / max(seconds, 0.001))
            )
            return spec, local

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
            results = list(pool.map(one, specs))
        if self.persistent:
            self.drive_sync = DriveSync(results, self.paths, manifest, self.log, self.clock)
            self.drive_sync.start()

    def start_comfyui(self):
        write_extra_model_paths(self.paths.extra_paths, self.model_dir)
        c = self.cfg["comfyui"]
        self.comfy = self.comfy_factory(
            self.paths.comfy,
            port=c["port"],
            extra_paths_yaml=self.paths.extra_paths,
            output_dir=self.paths.comfy_out,
            log_path=self.paths.comfy_log,
            pyuser=self.paths.pyuser,
            extra_args=c.get("extra_args", []),
        )
        try:
            self.wait_for_comfy()
        except WorkerError:
            tail = self.comfy.log_tail()
            if self.deps_installed_this_session or not any(s in tail for s in ("ModuleNotFoundError", "ImportError")):
                raise
            # An archive from the same fingerprint that does not import: rebuild it once.
            self.log("ComfyUI failed to import with the archived packages; reinstalling them once")
            self.ensure_deps(force=True)
            self.action("deps", "reinstalled")
            self.wait_for_comfy()

    def wait_for_comfy(self):
        self.comfy.start()
        deadline = self.clock() + self.cfg["comfyui"]["startup_timeout_seconds"]
        while self.clock() < deadline:
            if self.comfy_http.alive():
                return
            if not self.comfy.running():
                raise WorkerError("COMFYUI_START_FAILED", "ComfyUI exited during startup:\n" + self.comfy.log_tail())
            self.sleep(1)
        self.comfy.stop()
        raise WorkerError("COMFYUI_START_FAILED", "ComfyUI did not answer within the startup timeout")

    def node_check(self):
        info = self.comfy_http.request("GET", "/object_info", timeout=120)
        graph = self.cfg.get("reference_graph") or {}
        problems = check_against_object_info(graph, info)
        for spec in self.model_specs():
            options = combo_options(info, spec["loader"], spec["loader_field"])
            if options is not None and spec["name"] not in options:
                problems.append("%s does not list %s" % (spec["loader"], spec["name"]))
        if problems:
            raise WorkerError("NODE_CHECK_FAILED", "; ".join(problems))
        stats = self.comfy_http.request("GET", "/system_stats", timeout=10)
        self.state.update(comfyui_system=stats.get("system", {}))

    def save_environment(self):
        c = self.cfg["comfyui"]
        m = self.cfg["model"]
        rec = dict(self.env_record)
        rec.update(
            comfyui_version=c["ref"],
            comfyui_commit=c["commit"],
            z_image_version=m["revision"],
            z_image_revision=m["revision"],
            z_image_repo=m["repo"],
            z_image_files=[s["name"] for s in m["files"]],
            custom_nodes=[{"name": n["name"], "commit": n["commit"]} for n in self.cfg.get("custom_nodes", [])],
            python_version=platform.python_version(),
            torch_version=self.gpu.get("torch"),
            cuda_version=self.gpu.get("cuda"),
            colab_release_tag=os.environ.get("COLAB_RELEASE_TAG"),
            deps_fingerprint=self.deps_fingerprint,
            deps_parts=self.deps_parts,
            updated_at=now_iso(),
        )
        rec.setdefault("installed_at", now_iso())
        history = list(rec.get("history") or [])
        history.append({"at": now_iso(), "session": self.session, "actions": dict(self.actions)})
        rec["history"] = history[-30:]
        write_json_atomic(self.paths.environment, rec)
        self.env_record = rec
        self.state.update(environment={k: rec.get(k) for k in self.env_keys()})

    # -- queue ------------------------------------------------------------------------------------

    def ingest_inbox(self):
        for path in sorted(self.paths.inbox.glob("*.json"), key=lambda p: p.stat().st_mtime):
            job_id = path.stem
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or payload.get("job_id") != job_id:
                    raise ValueError("job_id does not match the file name")
            except (OSError, ValueError) as exc:
                # The Contents API may still be writing it: retry a few polls before rejecting.
                n = self.parse_failures.get(job_id, 0) + 1
                self.parse_failures[job_id] = n
                if n >= INBOX_PARSE_RETRIES:
                    path.replace(path.with_suffix(".rejected"))
                    with self.state.lock:
                        self.state.data["rejected"][job_id] = str(exc)
                    self.state.write()
                continue
            path.unlink()
            self.parse_failures.pop(job_id, None)
            if job_id in self.seen:
                self.log("job %s was pushed again; already handled this session" % job_id)
                continue
            self.seen.add(job_id)
            self.queue.append(payload)
            self.state.job(job_id, status=QUEUED, queued_at=now_iso())

    def apply_cancels(self):
        for path in list(self.paths.cancel.glob("*")):
            job_id = path.name.split(".")[0]
            path.unlink()
            for payload in list(self.queue):
                if payload["job_id"] == job_id:
                    self.queue.remove(payload)
                    self.state.job(job_id, status=CANCELLED, completed_at=now_iso())
                    self.log("job %s cancelled before it started" % job_id)

    def run_forever(self):
        idle_since = self.clock()
        cfg = self.cfg["worker"]
        loop_errors = 0
        while True:
            if self.clock() - self.t0 >= cfg["max_session_seconds"]:
                return "max_session"
            try:
                self.ingest_inbox()
                self.apply_cancels()
                loop_errors = 0
            except OSError as exc:
                loop_errors += 1
                self.log("inbox error %d: %s" % (loop_errors, exc))
                if loop_errors >= 10:
                    return "loop_errors"
            if self.queue:
                payload = self.queue.pop(0)
                self.state.update(worker_state=BUSY, idle_seconds=0)
                self.run_job(payload)
                self.jobs_done += 1
                idle_since = self.clock()
                continue
            idle = self.clock() - idle_since
            self.state.update(worker_state=IDLE, idle_seconds=round(idle, 1))
            if idle >= cfg["idle_timeout_seconds"]:
                self.ingest_inbox()  # one last look: a job pushed just now should not be stranded
                if self.queue:
                    continue
                return "idle_timeout"
            self.sleep(cfg["poll_interval_seconds"])

    # -- one job ----------------------------------------------------------------------------------

    def run_job(self, payload):
        job_id = payload["job_id"]
        rec = {
            "job_id": job_id,
            "session": self.session,
            "session_job_index": self.jobs_done + 1,
            "session_reused": self.jobs_done > 0,
            "started_at": now_iso(),
            "attempt": payload.get("attempt", 1),
            "gpu": self.gpu.get("name"),
            "vram_total_mib": self.gpu.get("vram_total_mib"),
            "model": self.cfg["model"]["name"],
            "model_revision": self.cfg["model"]["revision"],
            "comfyui_commit": self.cfg["comfyui"]["commit"],
            "workflow": payload.get("workflow"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "params": payload.get("params", {}),
        }
        t_job = self.clock()
        try:
            previous = read_json(self.paths.outputs / job_id / "metadata.json")
            if isinstance(previous, dict) and previous.get("status") == COMPLETED:
                # Pushed again after a lost session that had already finished it: report, don't re-render.
                out = self.paths.outbox / job_id
                out.mkdir(parents=True, exist_ok=True)
                for name in ("result.png", "metadata.json", "workflow.json"):
                    src = self.paths.outputs / job_id / name
                    if src.is_file():
                        shutil.copyfile(src, out / name)
                fields = {k: v for k, v in previous.items() if k != "job_id"}
                self.state.job(job_id, **dict(fields, reused_previous_output=True))
                self.log("job %s already completed in an earlier session; re-reported" % job_id)
                return
            self.state.job(job_id, status=RUNNING, started_at=rec["started_at"])
            check_payload(payload)
            if not self.comfy.running() or not self.comfy_http.alive():
                self.log("ComfyUI is not running; restarting it before job %s" % job_id)
                self.comfy.stop()
                self.wait_for_comfy()
            image, timing, vram_peak = self.render(job_id, payload)
            rec.update(timing)
            rec["vram_peak_mib"] = vram_peak
            out = self.paths.outbox / job_id
            out.mkdir(parents=True, exist_ok=True)
            result = out / "result.png"
            tmp = out / "result.png.tmp"
            tmp.write_bytes(self.comfy_http.view(image))
            os.replace(tmp, result)
            self.state.job(job_id, status=GENERATED)
            self.state.job(job_id, status=VALIDATING)
            problems = validate_image(result, payload["width"], payload["height"], payload.get("format", "png"))
            rec["validation"] = {"ok": not problems, "problems": problems, "bytes": result.stat().st_size}
            if problems:
                raise WorkerError("INVALID_OUTPUT", "; ".join(problems))
            self.state.job(job_id, status=VALID)
            rec["status"] = COMPLETED
        except WorkerError as exc:
            rec["status"] = {"TIMEOUT": TIMEOUT, "INVALID_OUTPUT": INVALID_OUTPUT, "CANCELLED": CANCELLED}.get(
                exc.code, FAILED
            )
            rec["error"] = self.error_record(exc, exc.code)
        except Exception as exc:  # noqa: BLE001 - one job must never stop the worker
            rec["status"] = FAILED
            rec["error"] = self.error_record(exc, "JOB_FAILED")
        rec["completed_at"] = now_iso()
        rec["job_seconds"] = round(self.clock() - t_job, 2)
        self.finish(job_id, payload, rec)

    def render(self, job_id, payload):
        prompt_id = str(uuid.uuid4())
        graph = payload["graph"]
        save_node = str(payload["save_node"])
        timeout = float(payload.get("timeout_seconds") or self.cfg["worker"]["job_timeout_seconds"])
        with VramSampler() as vram:
            t_submit = self.clock()
            resp = self.comfy_http.submit(graph, prompt_id, self.client_id)
            if resp.get("node_errors"):
                raise WorkerError("COMFY_REJECTED", "node errors: %s" % json.dumps(resp["node_errors"])[:2000])
            while True:
                if self.cancel_requested(job_id):
                    self.comfy_http.cancel(prompt_id)
                    raise WorkerError("CANCELLED", "cancelled while running")
                if self.clock() - t_submit > timeout:
                    self.comfy_http.cancel(prompt_id)
                    raise WorkerError("TIMEOUT", "no result after %.0f s" % timeout)
                if not self.comfy.running():
                    raise WorkerError("COMFY_CRASHED", "ComfyUI exited mid-job:\n" + self.comfy.log_tail())
                try:
                    entry = self.comfy_http.history(prompt_id)
                except (OSError, ValueError):
                    entry = None
                if entry:
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        raise WorkerError("COMFY_EXECUTION_ERROR", execution_error(status))
                    if status.get("completed"):
                        images = (entry.get("outputs", {}).get(save_node) or {}).get("images") or []
                        if not images:
                            raise WorkerError("OUTPUT_NOT_FOUND", "the save node reported no image")
                        timing = {
                            "generation_seconds": round(self.clock() - t_submit, 2),
                            "comfy_execution_seconds": execution_seconds(status),
                            "prompt_id": prompt_id,
                        }
                        return images[0], timing, vram.peak
                self.sleep(1)

    def cancel_requested(self, job_id):
        path = self.paths.cancel / job_id
        if path.exists():
            path.unlink()
            return True
        return False

    def error_record(self, exc, code):
        err = {
            "code": code,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[-4000:],
            "stack_trace": traceback.format_exc()[-8000:],
            "timestamp": now_iso(),
        }
        try:
            self.paths.logs.mkdir(parents=True, exist_ok=True)
            with open(self.paths.logs / "errors.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(err, session=self.session)) + "\n")
        except OSError:
            pass
        return err

    def finish(self, job_id, payload, rec):
        out = self.paths.outbox / job_id
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "workflow.json", payload.get("graph", {}))
        write_json_atomic(out / "metadata.json", rec)
        # Persist the result on Drive. The outbox copy is what the controller downloads, so a Drive
        # hiccup only costs the persistent copy, never the job.
        try:
            dest = self.paths.outputs / job_id
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("result.png", "workflow.json", "metadata.json"):
                if (out / name).is_file():
                    shutil.copyfile(out / name, dest / name)
        except OSError as exc:
            rec.setdefault("warnings", []).append("could not copy the output to Drive: %s" % exc)
            write_json_atomic(out / "metadata.json", rec)
        brief = {
            k: rec.get(k)
            for k in (
                "status",
                "error",
                "generation_seconds",
                "comfy_execution_seconds",
                "job_seconds",
                "vram_peak_mib",
                "session_job_index",
                "completed_at",
            )
        }
        self.state.job(job_id, **brief)
        self.marker("JOB", job_id=job_id, status=rec["status"], seconds=rec.get("generation_seconds"))
        try:
            for f in self.paths.comfy_out.glob("*"):
                if f.is_file():
                    f.unlink()
        except OSError:
            pass

    # -- shutdown ---------------------------------------------------------------------------------

    def shutdown(self, reason):
        self.state.update(worker_state=SHUTTING_DOWN, shutdown_reason=reason)
        self.log("shutting down: %s" % reason)
        if self.comfy is not None:
            self.comfy.stop()
        for d in (self.paths.comfy_out, self.paths.stage):
            shutil.rmtree(d, ignore_errors=True)
        if self.drive_sync is not None:
            self.state.update(phase="DRIVE_SYNC_WAIT")
            self.drive_sync.join()
            self.state.update(drive_sync=self.drive_sync.report)
        summary = {
            "session": self.session,
            "started_at": self.state.data.get("started_at"),
            "ended_at": now_iso(),
            "reason": reason,
            "jobs": self.jobs_done,
            "gpu": self.gpu.get("name"),
            "setup_actions": self.actions,
            "phase_seconds": self.state.data.get("phase_seconds"),
            "session_seconds": round(self.clock() - self.t0, 1),
        }
        try:
            with open(self.paths.logs / "sessions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, default=str) + "\n")
        except OSError:
            pass
        t = self.clock()
        if self.persistent:
            try:
                self.state.update(phase="DRIVE_FLUSH")
                self.flush_fn()
                self.state.update(drive_flush_seconds=round(self.clock() - t, 1))
            except Exception as exc:  # noqa: BLE001 - report and still finish the shutdown
                self.state.update(drive_flush_error="%s: %s" % (type(exc).__name__, exc))
        self.state.update(worker_state=FLUSHED, phase=None, ended_at=now_iso(), summary=summary)
        self.marker("SHUTDOWN", reason=reason, jobs=self.jobs_done)


class DriveSync(threading.Thread):
    """Copies freshly downloaded models to Drive in the background, then records each in the manifest.
    The manifest entry is written only after its file is fully copied, so the next session's size and
    revision check can never trust a half-uploaded file."""

    def __init__(self, results, paths, manifest, log, clock):
        super().__init__(name="zimg-drive-sync", daemon=True)
        self.results = results
        self.paths = paths
        self.manifest = dict(manifest)
        self.log = log
        self.clock = clock
        self.report = {}

    def run(self):
        for spec, local in self.results:
            t = self.clock()
            try:
                copy_large(local, self.paths.models / spec["folder"] / spec["name"])
            except OSError as exc:
                self.report[spec["name"]] = "failed: %s" % exc
                self.log("could not copy %s to Drive: %s" % (spec["name"], exc))
                continue
            self.manifest[spec["name"]] = {
                "model_name": spec["name"],
                "model_version": spec["revision"],
                "revision": spec["revision"],
                "download_source": "https://huggingface.co/%s/resolve/%s/%s"
                % (spec["repo"], spec["revision"], spec["path_in_repo"]),
                "file_size": spec["size"],
                "sha256": spec["sha256"],
                "folder": spec["folder"],
                "installed_at": now_iso(),
            }
            write_json_atomic(self.paths.manifest, self.manifest)
            seconds = round(self.clock() - t, 1)
            self.report[spec["name"]] = "copied in %s s" % seconds
            self.log("copied %s to Drive in %s s" % (spec["name"], seconds))


def execution_error(status):
    for kind, data in status.get("messages") or []:
        if kind == "execution_error":
            return "%s in node %s (%s): %s" % (
                data.get("exception_type"),
                data.get("node_id"),
                data.get("node_type"),
                data.get("exception_message"),
            )
    return "ComfyUI reported an execution error"


def execution_seconds(status):
    """ComfyUI's own start->success time from the history messages (ms timestamps), if present."""
    stamps = {}
    for kind, data in status.get("messages") or []:
        if isinstance(data, dict) and "timestamp" in data:
            stamps[kind] = data["timestamp"]
    if "execution_start" in stamps and "execution_success" in stamps:
        return round((stamps["execution_success"] - stamps["execution_start"]) / 1000.0, 2)
    return None


def extract_tarball(archive, dest, strip=True):
    """Extract into dest. With strip, drop the single top-level directory GitHub tarballs carry."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".extracting")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")
    src = tmp
    if strip:
        entries = [p for p in tmp.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            src = entries[0]
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)
    shutil.rmtree(tmp, ignore_errors=True)


def make_tarball(src_dir, archive):
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".partial")
    with tarfile.open(partial, "w") as tar:
        for child in sorted(Path(src_dir).iterdir()):
            tar.add(str(child), arcname=child.name)
    os.replace(partial, archive)


def write_extra_model_paths(path, model_dir):
    Path(path).write_text(
        "zimg:\n"
        "  base_path: %s\n"
        "  diffusion_models: diffusion_models\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n" % model_dir,
        encoding="utf-8",
    )


def run_worker(worker, print_every_seconds=60):
    """boot -> job loop -> shutdown. Never raises past here and never calls sys.exit: it runs inside the
    notebook kernel, and the kernel must stay usable for the controller's `colab stop`."""
    worker.state.write()
    worker.state.start_heartbeat(worker.cfg["worker"].get("heartbeat_seconds", 5), print_every_seconds)
    reason = "error"
    try:
        try:
            worker.boot()
        except WorkerError as exc:
            worker.state.update(worker_state=WORKER_FAILED, error=worker.error_record(exc, exc.code))
            worker.marker("ERROR", code=exc.code, message=str(exc)[-2000:])
            reason = "boot_failed:" + exc.code
            return reason
        except Exception as exc:  # noqa: BLE001
            worker.state.update(worker_state=WORKER_FAILED, error=worker.error_record(exc, "BOOT_FAILED"))
            worker.marker("ERROR", code="BOOT_FAILED", message=("%s: %s" % (type(exc).__name__, exc))[-2000:])
            reason = "boot_failed:BOOT_FAILED"
            return reason
        reason = worker.run_forever()
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        try:
            worker.shutdown(reason)
        finally:
            worker.state.stop_heartbeat()
    return reason


def main():
    cfg = json.loads(Path(os.environ[RUN_FILE_ENV]).read_text(encoding="utf-8"))
    run_worker(Worker(cfg))


if __name__ == "__main__":
    main()
