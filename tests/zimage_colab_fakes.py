"""Shared imports and test doubles for the Z-Image Colab skill tests (tests/test_zimage_colab_*.py).

Nothing here touches a GPU, Docker or Colab:
- FakeComfy is a stdlib HTTP server speaking the slice of ComfyUI's API the worker uses
  (/system_stats, /object_info, /prompt, /history/{id}, /view, /queue, /interrupt).
- The remote worker runs in-process with fake GPU probe, downloads, pip and Drive flush.
- FakeTransport maps the VM's /content/... onto a temp dir, so the real controller's uploads and
  downloads land where the in-process worker reads and writes.
"""

import hashlib
import importlib
import io
import json
import struct
import sys
import tarfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SKILL_DIR = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "z-image-colab"
SCRIPTS_DIR = SKILL_DIR / "scripts"


def import_skill(name):
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


z = import_skill("zimage_colab")
remote = z.remote  # the library loads remote/zimage_worker.py once; tests share that module object


# --- images --------------------------------------------------------------------------------------


def png_bytes(width, height, color=(200, 30, 30)):
    row = b"\x00" + bytes(color) * width
    raw = row * height

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return remote.PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def write_png(path, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(width, height))
    return path


# --- config --------------------------------------------------------------------------------------


def make_config(tmp_path, **env):
    """Defaults from config.example.json, output under tmp_path, no local config.json, no CU settle wait."""
    env = {"ZIMG_OUTPUT_DIR": str(tmp_path / "out"), "ZIMG_CU_SETTLE_SECONDS": "0", **env}
    return z.load_config(tmp_path / "no-config.json", env=env)


# --- fake ComfyUI --------------------------------------------------------------------------------


class FakeComfy:
    """behaviors: filename_prefix (= job_id) -> ok | error | hang | bad_png | wrong_size | reject."""

    def __init__(self, model_names=()):
        self.behaviors = {}
        self.default = "ok"
        self.prompts = {}  # prompt_id -> {"graph", "prefix"}
        self.calls = []
        self.model_names = list(model_names)
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
                fake.calls.append(("GET", url.path))
                if url.path == "/system_stats":
                    return self._send(200, {"system": {"comfyui_version": "fake", "python_version": "3.13"}})
                if url.path == "/object_info":
                    return self._send(200, fake.object_info())
                if url.path.startswith("/history/"):
                    return self._send(200, fake.history(url.path.split("/history/", 1)[1]))
                if url.path == "/view":
                    q = parse_qs(url.query)
                    return self._send(200, fake.view(q["filename"][0]), "image/png")
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                url = urlparse(self.path)
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                fake.calls.append(("POST", url.path, body if url.path != "/prompt" else body.get("prompt_id")))
                if url.path == "/prompt":
                    return fake.submit(self, body)
                return self._send(200, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        # poll_interval: shutdown() waits out one poll, and the default 0.5 s is paid by every test's teardown.
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def object_info(self):
        info = {}
        graph = fake_reference_graph()
        for node in graph.values():
            inputs = {name: ["ANY"] for name in node["inputs"]}
            info[node["class_type"]] = {"input": {"required": inputs}}
        for cls, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"), ("VAELoader", "vae_name")):
            info[cls]["input"]["required"][field] = [list(self.model_names)]
        return info

    def behavior(self, prefix):
        return self.behaviors.get(prefix, self.default)

    def submit(self, handler, body):
        graph = body["prompt"]
        prefix = graph["9"]["inputs"]["filename_prefix"]
        if self.behavior(prefix) == "reject":
            return handler._send(400, {"error": "invalid prompt", "node_errors": {"3": "bad"}})
        with self.lock:
            self.prompts[body["prompt_id"]] = {"graph": graph, "prefix": prefix}
        return handler._send(200, {"prompt_id": body["prompt_id"], "number": len(self.prompts), "node_errors": {}})

    def history(self, prompt_id):
        entry = self.prompts.get(prompt_id)
        if not entry:
            return {}
        b = self.behavior(entry["prefix"])
        if b == "hang":
            return {}
        if b == "error":
            msg = [
                "execution_error",
                {
                    "node_id": "3",
                    "node_type": "KSampler",
                    "exception_type": "RuntimeError",
                    "exception_message": "CUDA out of memory",
                },
            ]
            return {prompt_id: {"status": {"status_str": "error", "completed": False, "messages": [msg]}}}
        messages = [["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 3500}]]
        return {
            prompt_id: {
                "status": {"status_str": "success", "completed": True, "messages": messages},
                "outputs": {
                    "9": {"images": [{"filename": f"{entry['prefix']}_00001_.png", "subfolder": "", "type": "output"}]}
                },
            }
        }

    def view(self, filename):
        prefix = filename.rsplit("_00001_", 1)[0]
        entry = next(e for e in self.prompts.values() if e["prefix"] == prefix)
        w = entry["graph"]["5"]["inputs"]["width"]
        h = entry["graph"]["5"]["inputs"]["height"]
        b = self.behavior(prefix)
        if b == "bad_png":
            return b"\x89PNG\r\n\x1a\nthis is not a png"
        if b == "wrong_size":
            return png_bytes(w // 2, h // 2)
        return png_bytes(w, h)


def fake_reference_graph():
    return z.client.build_zimage_txt2img_workflow(
        "ref",
        z.gc.SAFE_SAFETY_NEGATIVE,
        0,
        "ref",
        {"unet": "u.safetensors", "text_encoder": "t.safetensors", "vae": "v.safetensors"},
        512,
        512,
    )


class FakeComfyProcess:
    """Stands in for remote.ComfyProcess: the FakeComfy server is already up; this just tracks state."""

    instances = []

    def __init__(self, comfy_dir, *, port, extra_paths_yaml, output_dir, log_path, pyuser, extra_args=()):
        self.comfy_dir = Path(comfy_dir)
        self.port = port
        self.extra_paths_yaml = Path(extra_paths_yaml)
        self.pyuser = pyuser
        self.started = 0
        self.stopped = 0
        self.alive = False
        FakeComfyProcess.instances.append(self)

    def start(self):
        assert (self.comfy_dir / "main.py").is_file(), "ComfyUI was not extracted before start"
        self.started += 1
        self.alive = True

    def running(self):
        return self.alive

    def stop(self, wait_seconds=20):
        if self.alive:
            self.stopped += 1
        self.alive = False

    def log_tail(self, chars=3000):
        return ""


# --- fake installers -----------------------------------------------------------------------------


def comfy_tarball_bytes(requirements="torch\nsafetensors\n"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (
            ("ComfyUI-abc/main.py", b"print('fake comfy')\n"),
            ("ComfyUI-abc/requirements.txt", requirements.encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


MODEL_BLOBS = {
    "unet.safetensors": b"U" * 1000,
    "te.safetensors": b"T" * 700,
    "vae.safetensors": b"V" * 300,
}


def model_specs():
    roles = {
        "unet.safetensors": ("unet", "diffusion_models"),
        "te.safetensors": ("text_encoder", "text_encoders"),
        "vae.safetensors": ("vae", "vae"),
    }
    specs = []
    for name, blob in MODEL_BLOBS.items():
        role, folder = roles[name]
        loader, field = z.ROLE_LOADERS[role]
        specs.append(
            {
                "role": role,
                "name": name,
                "folder": folder,
                "path_in_repo": f"split_files/{folder}/{name}",
                "size": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "loader": loader,
                "loader_field": field,
            }
        )
    return specs


class Installers:
    """Records every network-ish call the worker makes: the restart test asserts there are none."""

    def __init__(self):
        self.hf = []
        self.urls = []
        self.pip = []
        self.flushes = 0

    def hf_download(self, repo, filename, revision, local_dir):
        self.hf.append(filename)
        name = filename.rsplit("/", 1)[1]
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(MODEL_BLOBS[name])
        return str(path)

    def url_download(self, url, dest, timeout=600):
        self.urls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(comfy_tarball_bytes())
        return Path(dest)

    def runner(self, cmd, env=None, timeout=3600):
        if "freeze" in cmd:
            return 0, "torch==2.11.0\nnumpy==2.1.0\n"
        self.pip.append(cmd)
        userbase = Path(env["PYTHONUSERBASE"])
        (userbase / "lib" / "site-packages").mkdir(parents=True, exist_ok=True)
        (userbase / "lib" / "site-packages" / "fakepkg.py").write_text("x = 1\n")
        return 0, "Successfully installed fakepkg"

    def flush(self):
        self.flushes += 1


def worker_config(tmp_path, port, **overrides):
    drive_root = tmp_path / "drive" / "MyDrive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "session": "zimg-test",
        "local_root": str(tmp_path / "vm" / "zimg"),
        "drive_root": str(drive_root),
        "persistence": "drive",
        "persistent_root": str(drive_root / "AI" / "ZImage"),
        "worker": {
            "idle_timeout_seconds": 30,
            "poll_interval_seconds": 1,
            "job_timeout_seconds": 60,
            "max_session_seconds": 3600,
            "heartbeat_seconds": 60,
        },
        "comfyui": {
            "host": "127.0.0.1",
            "port": port,
            "ref": "v0.38.0",
            "commit": "c" * 40,
            "tarball_url": "https://example.invalid/comfy.tar.gz",
            "startup_timeout_seconds": 30,
            "extra_args": [],
        },
        "model": {
            "name": "z_image_turbo",
            "repo": "Comfy-Org/z_image_turbo",
            "revision": "r" * 40,
            "files": model_specs(),
        },
        "models": {"stage_to_local": True},
        "gpu": {"min_vram_gib": 15, "require_bf16": True},
        "custom_nodes": [],
        "reference_graph": fake_reference_graph(),
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


L4 = {
    "available": True,
    "name": "NVIDIA L4",
    "vram_total_mib": 23034,
    "capability": "8.9",
    "bf16": True,
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
}


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        time.sleep(0.001)


def make_worker(tmp_path, comfy, installers=None, *, gpu=None, clock=None, **cfg_overrides):
    installers = installers or Installers()
    clock = clock or FakeClock()
    cfg = worker_config(tmp_path, comfy.port, **cfg_overrides)
    w = remote.Worker(
        cfg,
        clock=clock,
        sleep=clock.sleep,
        echo=lambda line: None,
        gpu_probe=lambda: dict(gpu or L4),
        hf_download_fn=installers.hf_download,
        url_download_fn=installers.url_download,
        runner=installers.runner,
        flush_fn=installers.flush,
        comfy_factory=FakeComfyProcess,
        base_freeze="torch==2.11.0\n",
    )
    w.base_pins = lambda: ["torch==2.11.0"]
    return w, installers, clock


def payload_for(job_id, width=64, height=48, **extra):
    graph = z.client.build_zimage_txt2img_workflow(
        "a red cube",
        z.gc.SAFE_SAFETY_NEGATIVE,
        7,
        job_id,
        {"unet": "unet.safetensors", "text_encoder": "te.safetensors", "vae": "vae.safetensors"},
        512,
        512,
    )
    graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"] = width, height
    return {
        "job_id": job_id,
        "workflow": "z-image-txt2img",
        "graph": graph,
        "save_node": "9",
        "width": width,
        "height": height,
        "format": "png",
        "attempt": 1,
        **extra,
    }


def put_inbox(worker, payload):
    path = worker.paths.inbox / f"{payload['job_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# --- controller side -----------------------------------------------------------------------------


class FakeTransport(z.Transport):
    """`colab` without Colab. /content/... maps to vm_root; exec is handled by the exec_factory."""

    kind = "fake"

    def __init__(self, vm_root, *, usage_seq=None, sessions_text="[colab] No active sessions found on server."):
        self.vm_root = Path(vm_root)
        self.calls = []
        self.usage_seq = list(usage_seq or [(100.0, 0.0, 0), (100.0, 1.71, 1), (99.5, 0.0, 0), (99.4, 0.0, 0)])
        self.sessions_text = sessions_text
        self.fail = {}  # first arg -> exception to raise

    def vm(self, remote_path):
        return self.vm_root / remote_path.lstrip("/")

    def display(self, args):
        return "colab " + " ".join(args)

    def login_command(self):
        return "colab login"

    def cli_path(self, local, mount_dir):
        return str(Path(local).resolve())

    def call(self, args, *, label, timeout, mount_dir=None, on_line=None, idle_timeout=None, stop_event=None):
        self.calls.append(list(args))
        if args[0] in self.fail:
            raise self.fail[args[0]]
        if args[0] == "usage":
            bal, rate, active = self.usage_seq.pop(0) if len(self.usage_seq) > 1 else self.usage_seq[0]
            return f"Current balance: {bal} compute units\nUsage rate: {rate}/hr\nActive assignments: {active}"
        if args[0] == "sessions":
            return self.sessions_text
        if args[0] in ("new", "stop"):
            return "[colab] ok"
        if args[0] == "upload":
            src, dst = Path(args[3]), self.vm(args[4])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            return "uploaded"
        if args[0] == "download":
            src, dst = self.vm(args[3]), Path(args[4])
            if not src.is_file():
                raise z.ColabCommandError(label, 1, [f"FileNotFoundError: {args[3]}"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            return "downloaded"
        raise AssertionError(f"unexpected colab call {args}")

    def names(self):
        return [c[0] for c in self.calls]


class InProcessExec:
    """exec_factory: runs the real remote worker in a thread against the fake VM dir and FakeComfy."""

    def __init__(self, transport, comfy, installers, tmp_path, *, gpu=None, idle=1.0):
        self.idle = idle
        self.transport = transport
        self.comfy = comfy
        self.installers = installers
        self.tmp_path = tmp_path
        self.gpu = gpu or L4
        self.workers = []

    def __call__(self, transport, args, *, mount_dir, timeout, log_path):
        run_file = args[args.index("--env") + 1].split("=", 1)[1]
        cfg = json.loads(self.transport.vm(run_file).read_text())
        cfg["local_root"] = str(self.transport.vm(cfg["local_root"]))
        cfg["persistent_root"] = str(self.transport.vm(cfg["persistent_root"]))
        cfg["drive_root"] = str(self.transport.vm(cfg["drive_root"]))
        if cfg["persistence"] == "drive":  # the controller mounted it; without consent there is no Drive dir
            Path(cfg["drive_root"]).mkdir(parents=True, exist_ok=True)
        cfg["comfyui"]["port"] = self.comfy.port
        cfg["model"]["files"] = model_specs()
        cfg["reference_graph"] = fake_reference_graph()
        cfg["worker"]["heartbeat_seconds"] = 0.05
        cfg["worker"]["poll_interval_seconds"] = 0.05
        cfg["worker"]["idle_timeout_seconds"] = self.idle
        worker = remote.Worker(
            cfg,
            echo=lambda line: None,
            gpu_probe=lambda: dict(self.gpu),
            hf_download_fn=self.installers.hf_download,
            url_download_fn=self.installers.url_download,
            runner=self.installers.runner,
            flush_fn=self.installers.flush,
            comfy_factory=FakeComfyProcess,
            base_freeze="torch==2.11.0\n",
        )
        worker.base_pins = lambda: ["torch==2.11.0"]
        self.workers.append(worker)
        return _ExecThread(worker)


class _ExecThread:
    def __init__(self, worker):
        self.worker = worker
        self.done = threading.Event()
        self.error = None

        def run():
            try:
                self.reason = remote.run_worker(worker, print_every_seconds=3600)
            except BaseException as exc:  # noqa: BLE001 - surfaced to the test through .error
                self.error = exc
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop_local(self):
        pass


def mounted(*args, **kwargs):
    kwargs["on_url"]("https://accounts.google.com/o/oauth2/v2/auth?fake=1")
    return {"mounted": True, "consent_asked": True, "seconds": 12.0, "reason": None}


def not_mounted(*args, **kwargs):
    return {"mounted": False, "consent_asked": True, "seconds": 95.0, "reason": "consent_not_given"}


class FastSleep:
    """Controller sleep: real but short, so the in-process worker thread gets to run."""

    def __init__(self, seconds=0.02):
        self.seconds = seconds
        self.calls = 0

    def __call__(self, _requested):
        self.calls += 1
        time.sleep(self.seconds)
