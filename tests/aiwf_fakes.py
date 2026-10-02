"""Test doubles for the workflow platform tests (tests/test_aiwf_*.py). Nothing here touches a GPU, a
network or Google Drive:

- make_workspace() deploys ai_workflow/ into a temp folder with the real deploy script, then replaces the
  multi-GB model pins in that copy's registry with tiny blobs (same names, real sha256 of the blobs).
- FakeComfy is a stdlib HTTP server speaking the slice of ComfyUI's API the worker uses. It answers for any
  graph: PNG outputs take their size from the graph, video outputs are a small JSON blob that FakeProber turns
  back into an ffprobe-shaped report.
- Installers records every download and pip call, so the session-restart test can assert there were none.
"""

import hashlib
import importlib.util
import io
import json
import struct
import sys
import tarfile
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_DIR = REPO_ROOT / "ai_workflow"

from controller import notebook as nb  # noqa: E402
from controller.environment import Environment  # noqa: E402
from controller.storage import DriveFolderStorage  # noqa: E402
from controller.validate import PNG_SIGNATURE  # noqa: E402
from controller.worker import Worker  # noqa: E402

GPU_A100 = {"available": True, "name": "Fake A100", "vram_gib": 79.3, "bf16": True, "torch": "2.9.0", "cuda": "12.6"}
GPU_L4 = {"available": True, "name": "Fake L4", "vram_gib": 22.2, "bf16": True, "torch": "2.9.0", "cuda": "12.6"}
GPU_NONE = {"available": False, "torch": "2.9.0"}


def load_script(name):
    """Import ai_workflow/scripts/<name>.py as a module (the scripts folder is not a package)."""
    key = "aiwf_script_" + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, PLATFORM_DIR / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def png_bytes(width, height, color=(40, 90, 200)):
    row = b"\x00" + bytes(color) * width
    raw = row * height

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def make_workspace(tmp_path, name="ws"):
    """A deployed workspace with small model pins. Returns (storage, blobs) - blobs: file name -> bytes."""
    root = tmp_path / name
    load_script("deploy").deploy(root)
    storage = DriveFolderStorage(root)
    registry = storage.read_json("workflows/registry.json")
    blobs = {}
    for spec in registry["workflows"].values():
        for model in spec.get("models", []):
            blob = (model["name"].encode() + b"|") * 40
            blobs[model["name"]] = blob
            model["size"] = len(blob)
            model["sha256"] = hashlib.sha256(blob).hexdigest()
    storage.write_json("workflows/registry.json", registry)
    return storage, blobs


def tarball_bytes(top, files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("%s/%s" % (top, name))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Installers:
    def __init__(self, blobs):
        self.blobs = blobs
        self.hf = []
        self.urls = []
        self.pip = []

    def hf_download(self, repo, filename, revision, local_dir):
        self.hf.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.blobs[filename.rsplit("/", 1)[-1]])
        return str(path)

    def url_download(self, url, dest, timeout=600):
        self.urls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        if "ComfyUI/tar.gz" in url:
            data = tarball_bytes(
                "ComfyUI-abc", {"main.py": b"print('fake')\n", "requirements.txt": b"torch\naiohttp\n"}
            )
        else:
            data = tarball_bytes("node-abc", {"__init__.py": b"", "requirements.txt": b"opencv-python\n"})
        Path(dest).write_bytes(data)
        return Path(dest)

    def runner(self, cmd, env=None, timeout=3600):
        if "freeze" in cmd:
            return 0, "torch==2.9.0\nnumpy==2.1.0\n"
        self.pip.append(cmd)
        userbase = Path(env["PYTHONUSERBASE"])
        (userbase / "lib").mkdir(parents=True, exist_ok=True)
        (userbase / "lib" / "fakepkg.py").write_text("x = 1\n")
        return 0, "Successfully installed fakepkg"

    @property
    def network_calls(self):
        return len(self.hf) + len(self.urls) + len(self.pip)


class FakeProcess:
    """Stands in for comfy.ComfyProcess: the FakeComfy server is already listening; this tracks state."""

    instances = []

    def __init__(self, comfy_dir, *, port, extra_paths_yaml, output_dir, log_path, pyuser, extra_args=()):
        self.comfy_dir = Path(comfy_dir)
        self.extra_args = list(extra_args)
        self.started = 0
        self.alive = False
        FakeProcess.instances.append(self)

    def start(self):
        assert (self.comfy_dir / "main.py").is_file(), "ComfyUI was not extracted before start"
        self.started += 1
        self.alive = True

    def running(self):
        return self.alive

    def stop(self, wait_seconds=20):
        self.alive = False

    def log_tail(self, chars=3000):
        return ""


class FakeComfy:
    """behaviors: job_id -> ok | error | hang | bad_png | wrong_size | reject | no_output (default ok)."""

    def __init__(self, storage):
        self.storage = storage
        self.behaviors = {}
        self.prompts = {}  # prompt_id -> {"graph", "node", "prefix"}
        self.submitted = []  # graphs, in order
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/system_stats":
                    return self._send(200, {"system": {"comfyui_version": "fake", "python_version": "3.12.0"}})
                if url.path == "/object_info":
                    return self._send(200, fake.object_info())
                if url.path.startswith("/history/"):
                    return self._send(200, fake.history(url.path.split("/history/", 1)[1]))
                if url.path == "/view":
                    return self._send(200, fake.view(parse_qs(url.query)["filename"][0]), "application/octet-stream")
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                url = urlparse(self.path)
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if url.path == "/prompt":
                    return fake.submit(self, body)
                return self._send(200, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def object_info(self):
        """Every class and input used by any workflow graph in the workspace; loaders list every pinned model."""
        registry = self.storage.read_json("workflows/registry.json")["workflows"]
        info, models = {}, []
        for spec in registry.values():
            graph = self.storage.read_json("workflows/%s" % spec["workflow_file"]) or {}
            for node in graph.values():
                entry = info.setdefault(node["class_type"], {"input": {"required": {}}})
                for name in node["inputs"]:
                    entry["input"]["required"].setdefault(name, ["ANY"])
            models += spec.get("models", [])
        for model in models:
            cls, field = model["loader"]
            options = info[cls]["input"]["required"].get(field)
            if not (isinstance(options, list) and options and isinstance(options[0], list)):
                options = info[cls]["input"]["required"][field] = [[]]
            options[0].append(model["name"])
        return info

    def behavior(self, prefix):
        return next((b for job_id, b in self.behaviors.items() if job_id in prefix), "ok")

    def submit(self, handler, body):
        graph = body["prompt"]
        node, prefix = next(
            (n, v["inputs"]["filename_prefix"]) for n, v in graph.items() if "filename_prefix" in v["inputs"]
        )
        if self.behavior(prefix) == "reject":
            return handler._send(400, {"error": "invalid prompt", "node_errors": {"3": "bad"}})
        with self.lock:
            self.prompts[body["prompt_id"]] = {"graph": graph, "node": node, "prefix": prefix}
            self.submitted.append(graph)
        return handler._send(200, {"prompt_id": body["prompt_id"], "number": len(self.prompts), "node_errors": {}})

    def history(self, prompt_id):
        entry = self.prompts.get(prompt_id)
        if not entry:
            return {}
        b = self.behavior(entry["prefix"])
        if b == "hang":
            return {}
        if b == "error":
            msg = ["execution_error", {"node_id": "3", "node_type": "KSampler", "exception_type": "RuntimeError", "exception_message": "CUDA out of memory"}]  # fmt: skip
            return {prompt_id: {"status": {"status_str": "error", "completed": False, "messages": [msg]}}}
        messages = [["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 3500}]]
        status = {"status_str": "success", "completed": True, "messages": messages}
        if b == "no_output":
            return {prompt_id: {"status": status, "outputs": {}}}
        video = entry["graph"][entry["node"]]["class_type"] == "VHS_VideoCombine"
        if video:
            outputs = {"gifs": [{"filename": "%s_00001.mp4" % entry["prefix"], "subfolder": "", "type": "output"}]}
        else:
            outputs = {"images": [{"filename": "%s_00001_.png" % entry["prefix"], "subfolder": "", "type": "output"}]}
        return {prompt_id: {"status": status, "outputs": {entry["node"]: outputs}}}

    def view(self, filename):
        prefix = filename.rsplit("_00001", 1)[0]
        entry = next(e for e in self.prompts.values() if e["prefix"] == prefix)
        sized = next(n["inputs"] for n in entry["graph"].values() if "width" in n["inputs"] and "height" in n["inputs"])
        w, h = sized["width"], sized["height"]
        b = self.behavior(prefix)
        if filename.endswith(".mp4"):
            return json.dumps(
                {"width": w, "height": h, "frames": sized.get("length"), "broken": b == "bad_png"}
            ).encode()
        if b == "bad_png":
            return PNG_SIGNATURE + b"this is not a png"
        if b == "wrong_size":
            return png_bytes(w // 2, h // 2)
        return png_bytes(w, h)


def fake_prober(path):
    """ffprobe's report for a FakeComfy 'mp4' (a JSON blob): h264 / yuv420p / 24 fps with an audio stream."""
    info = json.loads(Path(path).read_text())
    video = {
        "codec_type": "video",
        "codec_name": "mpeg4" if info["broken"] else "h264",
        "width": info["width"],
        "height": info["height"],
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "24/1",
        "duration": str(info["frames"] / 24),
    }
    return {"streams": [video, {"codec_type": "audio", "codec_name": "aac"}], "format": {"duration": video["duration"]}}


def fake_previewer(video, target):
    Path(target).write_bytes(b"\xff\xd8fakejpeg")
    return True


class Rig:
    """One 'runtime' on a workspace: FakeComfy + Environment + Worker with fake installers and a fake clock."""

    def __init__(self, tmp_path, storage, blobs, *, gpu=GPU_A100, vm="vm", session=None, fake=None, copy_fn=None):
        self.storage = storage
        self.fake = fake or FakeComfy(storage)
        self.owns_fake = fake is None
        self.installers = Installers(blobs)
        self.now = [1_000_000.0]
        self.released = []
        self.env = Environment(
            storage,
            storage.read_json("configs/platform.json"),
            local_root=tmp_path / vm,
            clock=self.clock,
            sleep=self.sleep,
            echo=lambda *a: None,
            gpu_probe=lambda: dict(gpu),
            hf_download_fn=self.installers.hf_download,
            url_download_fn=self.installers.url_download,
            runner=self.installers.runner,
            comfy_factory=FakeProcess,
            comfy_url=self.fake.url,
            base_freeze="torch==2.9.0\n",
            **({"copy_fn": copy_fn} if copy_fn else {}),
        )
        self.worker = Worker(
            storage,
            self.env,
            session=session or vm,
            clock=self.clock,
            sleep=self.sleep,
            echo=lambda *a: None,
            prober=fake_prober,
            previewer=fake_previewer,
            flush_fn=lambda: self.released.append("flush"),
            release_fn=lambda: self.released.append("release"),
        )
        self.session = nb.Session(storage, self.env, self.worker)

    def clock(self):
        return self.now[0]

    def sleep(self, seconds):
        self.now[0] += seconds

    def close(self):
        self.worker.state.stop_heartbeat()
        if self.owns_fake:
            self.fake.close()
