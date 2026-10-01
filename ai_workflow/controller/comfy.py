"""ComfyUI as the worker sees it: an HTTP API on 127.0.0.1 and the server process behind it.

The generation engine is reached only through /prompt, /history, /view, /object_info and /system_stats, so the
same code drives a real ComfyUI on the compute runtime and the fake one in the offline tests.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import WorkflowError


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

    def system_stats(self):
        return self.request("GET", "/system_stats", timeout=10)

    def object_info(self):
        return self.request("GET", "/object_info", timeout=120)

    def submit(self, graph, prompt_id, client_id):
        try:
            resp = self.request("POST", "/prompt", {"prompt": graph, "client_id": client_id, "prompt_id": prompt_id})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[-2000:]
            raise WorkflowError("COMFY_REJECTED", "ComfyUI refused the graph (%s): %s" % (exc.code, detail))
        if resp.get("node_errors"):
            raise WorkflowError("COMFY_REJECTED", "node errors: %s" % json.dumps(resp["node_errors"])[:2000])
        return resp

    def history(self, prompt_id):
        return self.request("GET", "/history/" + urllib.parse.quote(prompt_id)).get(prompt_id)

    def cancel(self, prompt_id):
        """Drop the prompt from the queue and interrupt it if it is the one running."""
        for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
            try:
                self.request("POST", path, body, timeout=10)
            except (OSError, ValueError):
                pass

    def view(self, item):
        query = urllib.parse.urlencode(
            {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
        )
        return self.request("GET", "/view?" + query, timeout=300, raw=True)


class ComfyProcess:
    """The ComfyUI server subprocess, started from the runtime-local checkout."""

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
        # HF_HUB_OFFLINE / PIP_NO_INDEX: ComfyUI itself must never fetch anything, and these make any attempt
        # fail loudly instead of quietly downloading on every new session.
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


class VramSampler:
    """Peak `nvidia-smi` memory.used while a job runs. ComfyUI is another process, so torch's own peak
    counters in this kernel would read zero."""

    def __init__(self, interval=1.0):
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


def check_against_object_info(graph, object_info):
    """Every class_type installed, every input name declared. Catches a missing custom node or an input renamed
    by a ComfyUI update before the job is queued, instead of as an opaque validation error."""
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
        if cls == "VHS_VideoCombine":
            # Its per-format widgets (pix_fmt, crf, ...) are declared inside the format list, not as inputs.
            spec = (info.get("input") or {}).get("required", {}).get("format")
            formats = spec[1].get("formats", {}) if isinstance(spec, list) and len(spec) > 1 else {}
            declared.update(widget[0] for widget in formats.get(node["inputs"].get("format"), []))
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
