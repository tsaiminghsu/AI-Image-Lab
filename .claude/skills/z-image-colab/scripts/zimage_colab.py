"""Z-Image Turbo txt2img on a persistent Google Colab worker, driven through google-colab-cli.

Claude turns a request into jobs (submit.py). One controller (worker.py up) opens ONE Colab GPU session,
mounts Google Drive, starts remote/zimage_worker.py in it and feeds it every queued job until the worker
has been idle for worker.idle_timeout_seconds; then it releases the VM. Ten images are ten jobs in one
session, not ten sessions.

Why it is shaped this way:

- google-colab-cli has no Windows build, so every `colab` call runs in a short-lived Docker container
  (the same image and OAuth volume as the minimax-h3-colab skill). The transport is a copy of that
  skill's, not an import: skills are self-contained directories.
- The local job store (output/zimage/jobs/) is the only queue. The VM inbox is a delivery slot, and
  results are re-reported by job_id, so a job pushed twice after a lost session is not rendered twice.
- Once the worker cell runs, the controller never calls `colab exec` or `colab drivemount` again: the
  kernel runs one execution at a time, so a second one would queue behind the worker forever. Everything
  after that goes through `colab upload/download` (Jupyter Contents API), which works while it runs.
- A VM is declared lost when run/state.json's `seq` stops advancing on the local monotonic clock (no
  comparison of remote and local timestamps), and then `colab stop` runs: the 2026-10-01 H3 hang burned
  6.6 CU waiting on an exec that had lost its connection.
- Prompts and the graph come from the project's single assembly points:
  generate_character._build_prompt_and_negative (safety negatives) and
  comfyui_client.build_zimage_txt2img_workflow (the tested Z-Image template, cfg floor).

Python 3.11+. Imports training/ (requests only, no torch), so run it with ComfyUI\\.venv's python.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import json
import os
import queue
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

SKILL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SKILL_DIR.parents[2]
TRAINING_DIR = PROJECT_ROOT / "training"
REMOTE_SCRIPT = SKILL_DIR / "remote" / "zimage_worker.py"
EXAMPLE_CONFIG = SKILL_DIR / "config" / "config.example.json"
LOCAL_CONFIG = SKILL_DIR / "config" / "config.json"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import comfyui_client as client  # noqa: E402
import generate_character as gc  # noqa: E402
import workflow_contracts  # noqa: E402


def _load_remote():
    """remote/zimage_worker.py imports only the stdlib at module level, so the local side reuses its image
    validator and its graph safety check instead of keeping a second copy that could drift."""
    name = "zimage_worker"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REMOTE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


remote = _load_remote()

# --- constants -----------------------------------------------------------------------------------

WORKFLOWS = {
    "z-image-txt2img": {"template": "workflow_template_txt2img_zimage.json", "save_node": "9"},
}
DEFAULT_WORKFLOW = "z-image-txt2img"
CHECKPOINT_KEY = "z_image_turbo"  # for _build_prompt_and_negative: no Pony tags, no prompt adapter

# ~1 MP presets, sides multiples of 32 (param_resolver.STEP); the user's aspect ratio is what matters.
ASPECTS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
}
SIDE_MIN, SIDE_MAX = 512, 2048
SIZE_MULTIPLE = 16  # EmptySD3LatentImage
STEPS_MIN, STEPS_MAX = 1, 50  # capability_catalog's Z-Image range
CFG_MAX = 10.0
SEED_MAX = 2**32 - 1
GPUS = ("T4", "L4", "G4", "H100", "A100")
ROLE_LOADERS = {
    "unet": ("UNETLoader", "unet_name"),
    "text_encoder": ("CLIPLoader", "clip_name"),
    "vae": ("VAELoader", "vae_name"),
}
REMOTE_ROOT = "/content/zimg"
REMOTE_RUN_CONFIG = "/content/zimg_run_config.json"
SESSION_PREFIX = "zimg-"

PENDING = "PENDING"
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
LIFECYCLE = (PENDING, QUEUED, RUNNING, GENERATED, VALIDATING, VALID)
TERMINAL = (COMPLETED, FAILED, TIMEOUT, CANCELLED, INVALID_OUTPUT)
STATUSES = LIFECYCLE + TERMINAL
MAX_ATTEMPTS = 2

AUTH_REQUIRED_MESSAGE = "Google Colab authentication is required. Sign in once, then run again."


# --- errors --------------------------------------------------------------------------------------


class ZImageError(Exception):
    """An operational failure with a machine-readable code (Colab, Docker, Drive, the worker)."""

    def __init__(self, code: str, message: str, *, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.hint = hint

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "hint": self.hint}


UsageError = gc.UsageError


class ColabCommandError(RuntimeError):
    def __init__(self, label: str, returncode: int, output: list[str]):
        self.label = label
        self.returncode = returncode
        self.output = output
        tail = "\n".join(line for line in output[-15:] if line)
        super().__init__(f"{label} failed (exit {returncode})" + (f":\n{tail}" if tail else ""))

    @property
    def text(self) -> str:
        return "\n".join(self.output)


class ColabTimeout(TimeoutError):
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --- config --------------------------------------------------------------------------------------


def _int(value: str) -> int:
    return int(value)


ENV_OVERRIDES: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "ZIMG_GPU": ("colab", "gpu", str),
    "ZIMG_PERSISTENT_PATH": ("colab", "persistent_path", str),
    "ZIMG_CU_SETTLE_SECONDS": ("colab", "cu_settle_seconds", _int),
    "ZIMG_IDLE_TIMEOUT_SECONDS": ("worker", "idle_timeout_seconds", _int),
    "ZIMG_JOB_TIMEOUT_SECONDS": ("worker", "job_timeout_seconds", _int),
    "ZIMG_MAX_SESSION_SECONDS": ("worker", "max_session_seconds", _int),
    "ZIMG_OUTPUT_DIR": ("output", "directory", str),
}


def _merge(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def secret_like_keys(obj: Any, prefix: str = "") -> list[str]:
    """Same rule as training/cloud_video.py and the H3 skill: credentials never live in a settings file."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}{k}"
            if any(word in str(k).lower() for word in ("key", "token", "secret", "password", "credential")):
                found.append(name)
            found.extend(secret_like_keys(v, name + "."))
    return found


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ZImageError("INVALID_CONFIG", message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_in(section: dict, key: str, lo: int, hi: int, where: str) -> None:
    _check(_is_int(section.get(key)) and lo <= section[key] <= hi, f"{where}.{key} must be an integer {lo}-{hi}")


def validate_config(config: dict) -> dict:
    c = config["colab"]
    _check(c.get("auth") in ("oauth2", "adc"), "colab.auth must be oauth2 or adc")
    _check(str(c.get("gpu", "")).upper() in GPUS, f"colab.gpu must be one of {', '.join(GPUS)}")
    c["gpu"] = str(c["gpu"]).upper()
    _check(isinstance(c.get("high_mem"), bool), "colab.high_mem must be true or false")
    for key in ("docker_image", "config_volume"):
        _check(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-/:]*", str(c.get(key, "")))), f"colab.{key} is invalid")
    path = str(c.get("persistent_path", ""))
    _check(
        bool(path) and not path.startswith("/") and ".." not in path.split("/") and "\\" not in path,
        "colab.persistent_path must be a relative path under the Drive mount, e.g. MyDrive/AI/ZImage",
    )
    _check(str(c.get("drive_mount", "")).startswith("/content/"), "colab.drive_mount must be under /content/")
    _check(c.get("drive") in ("consent", "off"), "colab.drive must be consent or off")
    # drive.mount gives up after 120 s, so Enter must go in before that.
    _int_in(c, "drive_consent_wait_seconds", 10, 110, "colab")
    _check(isinstance(c.get("open_browser"), bool), "colab.open_browser must be true or false")
    for key in ("session_create_timeout_seconds", "transfer_timeout_seconds", "startup_timeout_seconds"):
        _int_in(c, key, 30, 7200, "colab")
    _int_in(c, "bootstrap_timeout_seconds", 300, 4 * 3600, "colab")
    _int_in(c, "cu_settle_seconds", 0, 3600, "colab")

    w = config["worker"]
    # The idle timeout is the cost dial: never hard-coded, only bounded.
    _int_in(w, "idle_timeout_seconds", 30, 4 * 3600, "worker")
    _int_in(w, "poll_interval_seconds", 1, 120, "worker")
    _int_in(w, "job_timeout_seconds", 60, 4 * 3600, "worker")
    _int_in(w, "max_session_seconds", 600, 24 * 3600, "worker")
    _int_in(w, "heartbeat_seconds", 1, 60, "worker")

    k = config["controller"]
    _int_in(k, "poll_interval_seconds", 2, 300, "controller")
    _int_in(k, "heartbeat_timeout_seconds", 60, 3600, "controller")
    _int_in(k, "max_missing_state_reads", 1, 100, "controller")
    _check(
        k["heartbeat_timeout_seconds"] > 3 * w["heartbeat_seconds"],
        "controller.heartbeat_timeout_seconds must be well above worker.heartbeat_seconds",
    )

    cu = config["comfyui"]
    # ComfyUI listens on the VM's loopback only: it is never exposed as a web service.
    _check(cu.get("host") in ("127.0.0.1", "localhost"), "comfyui.host must be 127.0.0.1 (loopback only)")
    _int_in(cu, "port", 1024, 65535, "comfyui")
    _check(bool(re.fullmatch(r"[0-9a-f]{40}", str(cu.get("commit", "")))), "comfyui.commit must be a full git SHA")
    _check(isinstance(cu.get("extra_args", []), list), "comfyui.extra_args must be a list")

    m = config["model"]
    _check(bool(re.fullmatch(r"[0-9a-f]{40}", str(m.get("revision", "")))), "model.revision must be a commit SHA")
    roles = set()
    for f in m.get("files", []):
        _check(f.get("role") in ROLE_LOADERS, f"model file role must be one of {sorted(ROLE_LOADERS)}")
        _check(_is_int(f.get("size")) and f["size"] > 0, f"model file {f.get('name')} needs its size in bytes")
        _check(bool(re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", "")))), f"{f.get('name')} needs a sha256")
        for key in ("name", "folder", "path_in_repo"):
            _check(isinstance(f.get(key), str) and f[key], f"model file needs {key}")
        roles.add(f["role"])
    _check(roles == set(ROLE_LOADERS), "model.files needs exactly one unet, text_encoder and vae")

    g = config["gpu"]
    _check(isinstance(g.get("min_vram_gib"), (int, float)) and g["min_vram_gib"] > 0, "gpu.min_vram_gib")
    _check(isinstance(g.get("require_bf16"), bool), "gpu.require_bf16 must be true or false")
    _check(isinstance(config["models"].get("stage_to_local"), bool), "models.stage_to_local must be true or false")
    _check(isinstance(config.get("custom_nodes", []), list), "custom_nodes must be a list")
    _check(
        _is_int(config["limits"].get("max_pixels")) and config["limits"]["max_pixels"] >= SIDE_MIN**2,
        "limits.max_pixels must be an integer",
    )
    _check(isinstance(config["output"].get("directory"), str) and config["output"]["directory"], "output.directory")
    return config


def load_config(path: Path | None = None, env: dict | None = None) -> dict:
    """config.example.json <- config/config.json (gitignored, optional) <- ZIMG_* environment variables."""
    env = os.environ if env is None else env
    config = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    local = LOCAL_CONFIG if path is None else Path(path)
    if local.is_file():
        try:
            data = json.loads(local.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ZImageError("INVALID_CONFIG", f"{local} is not valid JSON: {exc}") from exc
        _check(isinstance(data, dict), f"{local} must contain a JSON object")
        found = secret_like_keys(data)
        _check(
            not found,
            f"{local} has secret-looking keys {found}. Credentials never go in config files; "
            "google-colab-cli keeps its OAuth token in its own Docker volume.",
        )
        config = _merge(config, data)
    for var, (section, key, convert) in ENV_OVERRIDES.items():
        if env.get(var):
            try:
                config[section][key] = convert(env[var])
            except ValueError as exc:
                raise ZImageError("INVALID_CONFIG", f"{var}={env[var]!r} is not valid") from exc
    return validate_config(config)


def project_path(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def output_root(config: dict) -> Path:
    return project_path(config["output"]["directory"])


def model_files(config: dict) -> dict:
    """The ZIMAGE_MODELS-shaped dict comfyui_client.build_zimage_txt2img_workflow takes."""
    return {f["role"]: f["name"] for f in config["model"]["files"]}


# --- running the CLI (ported from the minimax-h3-colab skill) -------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_line(value: str) -> str:
    return _ANSI_RE.sub("", value).replace("\r", "").rstrip("\n").rstrip()


def _kill(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    try:
        if os.name == "nt":
            child.kill()
        else:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_streaming(
    cmd: list[str],
    *,
    label: str,
    timeout: float,
    on_line: Callable[[str], None] | None = None,
    on_abort: Callable[[], None] | None = None,
    env: dict | None = None,
    idle_timeout: float | None = None,
    stop_event: threading.Event | None = None,
) -> list[str]:
    """Run cmd, feed each output line to on_line, enforce a wall-clock limit and optionally a silence
    limit. stdin is /dev/null so an interactive OAuth or Drive-consent prompt fails fast instead of
    hanging. On any abort, on_abort runs first (Docker: `docker kill`, because killing the docker client
    does not stop the container), then the process tree is killed."""
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    child = subprocess.Popen(cmd, **kwargs)
    lines: queue.Queue[str | None] = queue.Queue()

    def collect() -> None:
        assert child.stdout is not None
        for raw in child.stdout:
            lines.put(raw)
        lines.put(None)

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    output: list[str] = []
    deadline = time.monotonic() + timeout
    last_output = time.monotonic()
    try:
        eof = False
        while not eof:
            now = time.monotonic()
            if stop_event is not None and stop_event.is_set():
                raise ColabTimeout(f"{label} was stopped")
            if now > deadline:
                raise ColabTimeout(f"{label} exceeded {timeout:g} seconds")
            if idle_timeout is not None and now - last_output > idle_timeout:
                raise ColabTimeout(f"{label} printed nothing for {idle_timeout:g} seconds")
            try:
                item = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                eof = True
                continue
            last_output = time.monotonic()
            line = clean_line(item)
            output.append(line)
            if on_line:
                on_line(line)
        try:
            returncode = child.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise ColabTimeout(f"{label} exceeded {timeout:g} seconds") from exc
    except BaseException:
        if on_abort:
            try:
                on_abort()
            except Exception:
                pass
        _kill(child)
        raise
    finally:
        reader.join(timeout=2)
        if child.stdout is not None:
            child.stdout.close()
    if returncode != 0:
        raise ColabCommandError(label, returncode, output)
    return output


class Transport:
    """Where the `colab` process runs. Tests replace call()."""

    kind = "base"

    def preflight(self) -> None:
        pass

    def login_command(self) -> str:
        raise NotImplementedError

    def interactive_command(self, args: list[str], *, name: str) -> list[str]:
        raise NotImplementedError

    def abort(self, name: str) -> None:
        pass

    def display(self, args: list[str]) -> str:
        raise NotImplementedError

    def cli_path(self, local: Path, mount_dir: Path) -> str:
        raise NotImplementedError

    def call(
        self,
        args: list[str],
        *,
        label: str,
        timeout: float,
        mount_dir: Path | None = None,
        on_line: Callable[[str], None] | None = None,
        idle_timeout: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        raise NotImplementedError


class DockerTransport(Transport):
    kind = "docker"
    CONFIG_MOUNT = "/root/.config/colab-cli"
    WORK_MOUNT = "/work"

    def __init__(self, image: str, volume: str, auth: str, docker: str = "docker"):
        self.image = image
        self.volume = volume
        self.auth = auth
        self.docker = docker

    def command(self, args: list[str], *, mount_dir: Path | None, name: str) -> list[str]:
        # PYTHONUNBUFFERED: without a TTY the CLI's stdout is block-buffered and the worker's lines
        # would arrive only when the whole `colab exec` ends.
        cmd = [self.docker, "run", "--rm", "--init", "--name", name, "-e", "PYTHONUNBUFFERED=1"]
        cmd += ["-v", f"{self.volume}:{self.CONFIG_MOUNT}"]
        if mount_dir is not None:
            cmd += ["-v", f"{mount_dir}:{self.WORK_MOUNT}"]
        return cmd + [self.image, f"--auth={self.auth}", *args]

    def cli_path(self, local: Path, mount_dir: Path) -> str:
        rel = Path(local).resolve().relative_to(Path(mount_dir).resolve())
        return f"{self.WORK_MOUNT}/{rel.as_posix()}"

    def abort(self, name: str) -> None:
        try:
            subprocess.run(
                [self.docker, "kill", name], stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def display(self, args: list[str]) -> str:
        return subprocess.list2cmdline(self.command(list(args), mount_dir=None, name="zimg-manual"))

    def login_command(self) -> str:
        return subprocess.list2cmdline(
            [self.docker, "run", "--rm", "-it", "-v", f"{self.volume}:{self.CONFIG_MOUNT}"]
            + [self.image, f"--auth={self.auth}", "usage"]
        )

    def interactive_command(self, args: list[str], *, name: str) -> list[str]:
        """`colab <args>` under `script`, so the CLI gets a /dev/tty that our piped stdin writes into."""
        inner = shlex.join(["colab", f"--auth={self.auth}", *args])
        return [
            self.docker,
            "run",
            "-i",
            "--rm",
            "--name",
            name,
            "-e",
            "PYTHONUNBUFFERED=1",
            "-v",
            f"{self.volume}:{self.CONFIG_MOUNT}",
            "--entrypoint",
            "script",
            self.image,
            "-qfec",
            inner,
            "/dev/null",
        ]

    def preflight(self) -> None:
        if not shutil.which(self.docker):
            raise ZImageError("DOCKER_UNAVAILABLE", "Docker is not installed or not on PATH.")
        try:
            info = subprocess.run(
                [self.docker, "info", "--format", "{{.ServerVersion}}"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ZImageError("DOCKER_UNAVAILABLE", "`docker info` did not answer within 60 s.") from exc
        if info.returncode != 0:
            raise ZImageError("DOCKER_UNAVAILABLE", "Docker Desktop is not running. Start it and run again.")
        image = subprocess.run(
            [self.docker, "image", "inspect", self.image],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if image.returncode != 0:
            raise ZImageError(
                "DOCKER_UNAVAILABLE",
                f"Docker image {self.image} is not built yet.",
                hint="Build it from .claude/skills/minimax-h3-colab/docker (see that skill's README).",
            )

    def call(
        self,
        args: list[str],
        *,
        label: str,
        timeout: float,
        mount_dir: Path | None = None,
        on_line: Callable[[str], None] | None = None,
        idle_timeout: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        name = f"zimg-{uuid.uuid4().hex[:12]}"
        cmd = self.command(list(args), mount_dir=mount_dir, name=name)
        lines = run_streaming(
            cmd,
            label=label,
            timeout=timeout,
            on_line=on_line,
            on_abort=lambda: self.abort(name),
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            idle_timeout=idle_timeout,
            stop_event=stop_event,
        )
        return "\n".join(line for line in lines if line)


def make_transport(config: dict) -> Transport:
    c = config["colab"]
    return DockerTransport(c["docker_image"], c["config_volume"], c["auth"])


# --- colab operations ----------------------------------------------------------------------------

_AUTH_RE = re.compile(
    r"To authorize colab-cli|authorization code|EOFError|invalid_grant|RefreshError|reauthenticat"
    r"|DefaultCredentialsError|token has been expired or revoked",
    re.IGNORECASE,
)
_GPU_RE = re.compile(
    r"Backend rejected accelerator|Allocation refused|quota or entitlement"
    r"|tun/m/assign[\s\S]{0,400}?(?:Service Unavailable|Too Many Requests)",
    re.IGNORECASE,
)
_SESSION_LINE_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)\s*\|\s*Hardware:\s*(?P<hardware>[^|]*)")


def auth_error(transport: Transport) -> ZImageError:
    return ZImageError(
        "AUTH_REQUIRED",
        AUTH_REQUIRED_MESSAGE,
        hint=f"Run this once in your own terminal and paste the code Google shows: {transport.login_command()}",
    )


def _connection_error(transport: Transport, exc: Exception, what: str) -> ZImageError:
    text = exc.text if isinstance(exc, ColabCommandError) else str(exc)
    if _AUTH_RE.search(text):
        return auth_error(transport)
    return ZImageError("COLAB_CONNECTION_FAILED", f"{what}: {exc}")


def parse_usage(text: str) -> dict:
    def number(pattern: str) -> float | None:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return float(m.group(1).replace(",", "")) if m else None

    return {
        "balance": number(r"^Current balance:\s*([\d,]+(?:\.\d+)?)\s+compute units"),
        "rate_per_hour": number(r"^Usage rate:\s*([\d,]+(?:\.\d+)?)\s*/\s*hr"),
        "active_assignments": number(r"^Active assignments:\s*(\d+)"),
    }


def parse_sessions(text: str) -> list[dict]:
    """`colab sessions` lines: `[name] endpoint | Hardware: X | Shape: Y | Variant: Z` ("?" = no local state)."""
    found = []
    for line in text.splitlines():
        m = _SESSION_LINE_RE.match(line.strip())
        if m:
            found.append({k: v.strip() for k, v in m.groupdict().items()})
    return found


def colab_usage(transport: Transport, *, timeout: float = 120) -> dict:
    try:
        return parse_usage(transport.call(["usage"], label="colab usage", timeout=timeout))
    except (ColabCommandError, ColabTimeout) as exc:
        raise _connection_error(transport, exc, "Could not reach Colab") from exc


def colab_sessions(transport: Transport, *, timeout: float = 120) -> list[dict]:
    try:
        return parse_sessions(transport.call(["sessions"], label="colab sessions", timeout=timeout))
    except (ColabCommandError, ColabTimeout) as exc:
        raise _connection_error(transport, exc, "Could not list Colab sessions") from exc


def colab_new(transport: Transport, session: str, *, gpu: str | None, high_mem: bool, timeout: float, on_line=None):
    args = ["new", "--session", session]
    if gpu:
        args += ["--gpu", gpu]
    if high_mem:
        args += ["--high-mem"]
    try:
        return transport.call(args, label="colab new", timeout=timeout, on_line=on_line)
    except ColabCommandError as exc:
        if _GPU_RE.search(exc.text):
            raise ZImageError("GPU_UNAVAILABLE", f"No {gpu} is available on Colab right now.", hint=exc.text[-600:])
        raise _connection_error(transport, exc, "Could not create the Colab session") from exc
    except ColabTimeout as exc:
        raise _connection_error(transport, exc, "Could not create the Colab session") from exc


CONSENT_URL_RE = re.compile(r"^https://accounts\.google\.com/\S+$")


def drive_consent_mount(
    transport: Transport,
    session: str,
    mount: str,
    *,
    wait_seconds: float,
    signal_path: Path,
    on_url: Callable[[str], None],
    on_line: Callable[[str], None] | None = None,
    timeout: float = 300,
) -> dict:
    """Mount Drive the way Colab allows it from a CLI: Google asks for consent on every new VM (measured
    2026-10-01) and drive.mount waits only 120 s for it. The CLI prints the consent URL and then reads
    Enter from /dev/tty, so it runs under util-linux `script` (a pseudo-terminal) with our stdin piped in.
    Enter goes in when the user confirmed (signal_path appears, see `worker.py consent-done`) or after
    wait_seconds, whichever is first - early enough to beat the 120 s limit if they approved silently."""
    name = f"zimg-{uuid.uuid4().hex[:12]}"
    cmd = transport.interactive_command(["drivemount", "--session", session, mount], name=name)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines: list[str] = []

    def read() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = clean_line(raw.decode("utf-8", "replace"))
            lines.append(line)
            if on_line and line and not CONSENT_URL_RE.match(line):
                on_line(line)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    t0 = time.monotonic()
    url = None
    t_url = None
    enter_sent = confirmed = timed_out = False
    try:
        while proc.poll() is None:
            if url is None:
                url = next((ln for ln in lines if CONSENT_URL_RE.match(ln)), None)
                if url:
                    t_url = time.monotonic()
                    on_url(url)
            if url and not enter_sent:
                confirmed = signal_path.exists()
                if confirmed or time.monotonic() - t_url >= wait_seconds:
                    assert proc.stdin is not None
                    proc.stdin.write(b"\n")
                    proc.stdin.flush()
                    enter_sent = True
            if time.monotonic() - t0 > timeout:
                timed_out = True
                break
            time.sleep(0.25)
    finally:
        if proc.poll() is None:
            transport.abort(name)
            _kill(proc)
        reader.join(timeout=5)
    text = "\n".join(lines)
    mounted = "Mounted at" in text or "already mounted" in text
    reason = None
    if not mounted:
        if timed_out:
            reason = "timeout"
        elif "Error propagating" in text:
            reason = "consent_not_given"
        elif "mount failed" in text:
            reason = "mount_failed"
        else:
            reason = "unknown"
    return {
        "mounted": mounted,
        "consent_asked": url is not None,
        "confirmed_by_user": confirmed,
        "enter_sent": enter_sent,
        "seconds": round(time.monotonic() - t0, 1),
        "reason": reason,
        "tail": text[-600:] if not mounted else None,
    }


def upload_file(transport: Transport, session: str, local: Path, remote_path: str, *, timeout: float) -> str:
    local = Path(local).resolve()
    return transport.call(
        ["upload", "--session", session, transport.cli_path(local, local.parent), remote_path],
        label=f"upload {local.name}",
        timeout=timeout,
        mount_dir=local.parent,
    )


def download_file(transport: Transport, session: str, remote_path: str, local: Path, *, timeout: float) -> str:
    local = Path(local).resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    return transport.call(
        ["download", "--session", session, remote_path, transport.cli_path(local, local.parent)],
        label=f"download {remote_path}",
        timeout=timeout,
        mount_dir=local.parent,
    )


def stop_session(transport: Transport, session: str, *, timeout: float = 300) -> str:
    return transport.call(["stop", "--session", session], label="colab stop", timeout=timeout)


class ExecHandle:
    """`colab exec` of the worker, in a background thread. Its output is a log only: the worker is followed
    through run/state.json, so a dropped exec connection does not end anything by itself."""

    def __init__(self, transport: Transport, args: list[str], *, mount_dir: Path, timeout: float, log_path: Path):
        self.log_path = log_path
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.last_line_at = time.monotonic()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def on_line(line: str) -> None:
            self.last_line_at = time.monotonic()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {line}\n")

        def run() -> None:
            try:
                transport.call(
                    args,
                    label="colab exec (worker)",
                    timeout=timeout,
                    mount_dir=mount_dir,
                    on_line=on_line,
                    stop_event=self.stop_event,
                )
            except BaseException as exc:  # noqa: BLE001 - recorded, the controller decides
                self.error = exc
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, name="zimg-exec", daemon=True)
        self.thread.start()

    def stop_local(self) -> None:
        """Kill only the local container (the remote cell is unaffected)."""
        self.stop_event.set()
        self.thread.join(timeout=30)


# --- jobs ----------------------------------------------------------------------------------------


def new_job_id(now: float | None = None) -> str:
    t = time.localtime(now if now is not None else time.time())
    return f"zimg-{time.strftime('%Y%m%d-%H%M%S', t)}-{uuid.uuid4().hex[:6]}"


def resolve_size(aspect: str | None, width: int | None, height: int | None, max_pixels: int) -> tuple[int, int]:
    if aspect is not None:
        if width is not None or height is not None:
            raise UsageError("give either --aspect or --width/--height, not both")
        if aspect not in ASPECTS:
            raise UsageError(f"unknown aspect {aspect!r}; one of {', '.join(ASPECTS)}")
        width, height = ASPECTS[aspect]
    elif width is None and height is None:
        width, height = ASPECTS["1:1"]
    elif width is None or height is None:
        raise UsageError("give both --width and --height")
    for name, value in (("width", width), ("height", height)):
        if not _is_int(value):
            raise UsageError(f"{name} must be an integer")
        if not SIDE_MIN <= value <= SIDE_MAX:
            raise UsageError(f"{name} {value} is outside {SIDE_MIN}-{SIDE_MAX}")
        if value % SIZE_MULTIPLE:
            raise UsageError(f"{name} {value} is not a multiple of {SIZE_MULTIPLE} (Z-Image's latent grid)")
    if width * height > max_pixels:
        raise UsageError(f"{width}x{height} is over the limit of {max_pixels} pixels (limits.max_pixels)")
    return width, height


def resolve_seed(seed: int | None, rng: random.Random | None = None) -> int:
    """-1 (or None) means "pick one". The pick is recorded in the job, so every image can be reproduced."""
    if seed is None or seed == -1:
        return (rng or random).randint(0, 2**31 - 1)
    if not _is_int(seed) or not 0 <= seed <= SEED_MAX:
        raise UsageError(f"seed must be -1 or an integer 0-{SEED_MAX}")
    return seed


def build_job(
    config: dict,
    *,
    prompt: str,
    negative: str | None = None,
    aspect: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    seed: int | None = -1,
    solo: bool = False,
    allow_text: bool = False,
    workflow: str = DEFAULT_WORKFLOW,
    batch: str | None = None,
    job_id: str | None = None,
    rng: random.Random | None = None,
) -> dict:
    """One job record: validated parameters plus the final API graph. The user's prompt is passed through
    as written (no style suffix, no rewriting): `_build_prompt_and_negative` with empty style strings only
    adds the safety negatives."""
    if workflow not in WORKFLOWS:
        raise UsageError(f"unknown workflow {workflow!r}; available: {', '.join(WORKFLOWS)}")
    text = (prompt or "").strip()
    if not text:
        raise UsageError("the prompt is empty")
    if len(text) > gc.MAX_PROMPT_CHARS:
        raise UsageError(f"the prompt is {len(text)} characters; the limit is {gc.MAX_PROMPT_CHARS}")
    width, height = resolve_size(aspect, width, height, config["limits"]["max_pixels"])
    steps = client.ZIMAGE_STEPS if steps is None else steps
    if not _is_int(steps) or not STEPS_MIN <= steps <= STEPS_MAX:
        raise UsageError(f"steps must be an integer {STEPS_MIN}-{STEPS_MAX}")
    cfg = client.ZIMAGE_CFG if cfg is None else cfg
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not client.SAFETY_MIN_CFG <= cfg <= CFG_MAX:
        raise UsageError(
            f"cfg must be {client.SAFETY_MIN_CFG}-{CFG_MAX}: below {client.SAFETY_MIN_CFG} ComfyUI would skip "
            "the negative prompt, and with it the safety terms"
        )
    seed_value = resolve_seed(seed, rng)
    user_negative = (negative or "").strip() or None
    positive, negative_prompt = gc._build_prompt_and_negative(
        text, user_negative, "safe", None, "", "", CHECKPOINT_KEY, solo=solo, allow_text=allow_text
    )
    job_id = job_id or new_job_id()
    spec = WORKFLOWS[workflow]
    graph = client.build_zimage_txt2img_workflow(
        positive, negative_prompt, seed_value, job_id, model_files(config), width, height, steps, float(cfg)
    )
    client.enforce_min_cfg(graph)
    problems = workflow_contracts.check(spec["template"], graph)
    if problems:
        raise ZImageError("INVALID_WORKFLOW", "; ".join(problems))
    remote.check_graph(graph)  # the worker's trust-boundary check, run early so a bad job never ships
    m = config["model"]
    return {
        "schema": 1,
        "job_id": job_id,
        "workflow": workflow,
        "status": PENDING,
        "batch": batch,
        "prompt": positive,
        "user_prompt": text,
        "user_negative": user_negative,
        "negative_prompt": negative_prompt,
        "solo": bool(solo),
        "allow_text": bool(allow_text),
        "aspect": aspect,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": float(cfg),
        "sampler": client.ZIMAGE_SAMPLER,
        "scheduler": client.ZIMAGE_SCHEDULER,
        "shift": client.ZIMAGE_SHIFT,
        "seed": seed_value,
        "seed_requested": seed,
        "model": m["name"],
        "model_repo": m["repo"],
        "model_revision": m["revision"],
        "model_files": model_files(config),
        "save_node": spec["save_node"],
        "graph": graph,
        "attempt": 0,
        "created_at": now_iso(),
        "queued_at": None,
        "started_at": None,
        "completed_at": None,
        "session": None,
        "session_reused": None,
        "gpu": None,
        "vram_total_mib": None,
        "vram_peak_mib": None,
        "generation_seconds": None,
        "output": None,
        "validation": None,
        "error": None,
        "history": [{"status": PENDING, "at": now_iso()}],
    }


def remote_payload(job: dict, config: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "workflow": job["workflow"],
        "graph": job["graph"],
        "save_node": job["save_node"],
        "width": job["width"],
        "height": job["height"],
        "format": "png",
        "attempt": job["attempt"],
        "timeout_seconds": config["worker"]["job_timeout_seconds"],
        "params": {k: job[k] for k in ("prompt", "negative_prompt", "steps", "cfg", "seed", "sampler", "scheduler")},
    }


def transition_allowed(old: str, new: str) -> bool:
    """Forward through the lifecycle (jumps allowed: the controller may miss intermediate states between two
    polls), any non-terminal state to a terminal one, and back to PENDING only from a state that was handed
    to a session (the session was lost). Terminal states are final."""
    if old not in STATUSES or new not in STATUSES or old in TERMINAL or old == new:
        return False
    if new in TERMINAL:
        return True
    if new == PENDING:
        return old in (QUEUED, RUNNING, GENERATED, VALIDATING, VALID)
    return LIFECYCLE.index(new) > LIFECYCLE.index(old)


# --- local store ---------------------------------------------------------------------------------


def write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader has the file open for a moment
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class FileLock:
    """O_CREAT|O_EXCL lockfile; a lock older than `stale` seconds is from a dead process and is broken."""

    def __init__(self, path: Path, *, timeout: float = 15.0, stale: float = 60.0):
        self.path = Path(path)
        self.timeout = timeout
        self.stale = stale

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise ZImageError("STORE_LOCKED", f"{self.path} is held by another process")
                time.sleep(0.05)

    def __exit__(self, *exc):
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        return False


class JobStore:
    """output/zimage/jobs/<job_id>.json - the only queue. Status changes go through transition()."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.cancel_dir = self.root / "cancel"
        self.lock_path = self.jobs_dir / ".lock"

    def path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_id):
            raise UsageError(f"invalid job id {job_id!r}")
        return self.jobs_dir / f"{job_id}.json"

    def add(self, job: dict) -> dict:
        with FileLock(self.lock_path):
            path = self.path(job["job_id"])
            if path.exists():
                raise UsageError(f"job {job['job_id']} already exists")
            write_json_atomic(path, job)
        return job

    def get(self, job_id: str) -> dict:
        path = self.path(job_id)
        if not path.is_file():
            raise UsageError(f"no job {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, statuses: tuple[str, ...] | None = None) -> list[dict]:
        jobs = []
        if not self.jobs_dir.is_dir():
            return jobs
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if statuses is None or job.get("status") in statuses:
                jobs.append(job)
        return sorted(jobs, key=lambda j: (j.get("created_at") or "", j["job_id"]))

    def update(self, job_id: str, *, status: str | None = None, **fields) -> dict:
        with FileLock(self.lock_path):
            job = self.get(job_id)
            if status is not None and status != job["status"]:
                if not transition_allowed(job["status"], status):
                    raise ZImageError("BAD_TRANSITION", f"{job_id}: {job['status']} -> {status} is not allowed")
                job["status"] = status
                job.setdefault("history", []).append({"status": status, "at": now_iso()})
            job.update(fields)
            write_json_atomic(self.path(job_id), job)
            return job

    def request_cancel(self, job_id: str) -> str:
        """PENDING is cancelled here; a job a session already has gets a request the controller forwards."""
        job = self.get(job_id)
        if job["status"] in TERMINAL:
            return job["status"]
        if job["status"] == PENDING:
            self.update(job_id, status=CANCELLED, completed_at=now_iso())
            return CANCELLED
        self.cancel_dir.mkdir(parents=True, exist_ok=True)
        (self.cancel_dir / job_id).write_text(now_iso(), encoding="utf-8")
        return "CANCEL_REQUESTED"


# --- output validation ---------------------------------------------------------------------------


def validate_output(path: Path, width: int, height: int) -> list[str]:
    """File exists, size > 0, PNG structure (signature, IHDR size, chunk CRCs, IEND) and, when Pillow is
    importable (it is in ComfyUI\\.venv), a full decode. Shared with the worker: remote.validate_image."""
    return remote.validate_image(path, width, height, "png")


# --- controller ----------------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) would TerminateProcess on Windows: ask the kernel instead.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _open_in_browser(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


def keep_awake(on: bool) -> None:
    """While a billing VM is assigned, ask Windows not to idle-sleep: a sleeping PC cannot stop it."""
    if os.name != "nt":
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))
    except (AttributeError, OSError):
        pass


class ControllerState:
    """output/zimage/controller.json: which session the live controller owns, plus its heartbeat."""

    def __init__(self, root: Path):
        self.path = Path(root) / "controller.json"
        self.lock_path = Path(root) / "controller.lock"

    def read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def live(self, max_age: float = 180.0) -> dict | None:
        data = self.read()
        if not data or data.get("state") == "stopped":
            return None
        if not pid_alive(int(data.get("pid") or 0)):
            return None
        if time.time() - float(data.get("heartbeat_epoch") or 0) > max_age:
            return None
        return data

    def write(self, **fields) -> None:
        data = self.read()
        data.update(fields, pid=os.getpid(), heartbeat_epoch=time.time(), heartbeat_at=now_iso())
        write_json_atomic(self.path, data)


class Controller:
    def __init__(
        self,
        config: dict,
        *,
        transport: Transport,
        store: JobStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        echo: Callable[[str], None] | None = None,
        exec_factory: Callable[..., Any] = ExecHandle,
        awake: Callable[[bool], None] = keep_awake,
        drive_mounter: Callable[..., dict] = drive_consent_mount,
        open_url: Callable[[str], Any] | None = None,
    ):
        self.config = config
        self.transport = transport
        self.root = output_root(config)
        self.store = store or JobStore(self.root)
        self.clock = clock
        self.sleep = sleep
        self.echo = echo or (lambda line: print(line, file=sys.stderr, flush=True))
        self.exec_factory = exec_factory
        self.awake = awake
        self.drive_mounter = drive_mounter
        self.open_url = open_url if open_url is not None else _open_in_browser
        self.persistence = "ephemeral"
        self.drive_outcome: dict | None = None
        self.state_file = ControllerState(self.root)
        self.work = self.root / ".work"
        self.session: str | None = None
        self.exec: Any = None
        self.opened = False
        self.t_open: float | None = None
        self.usage_before: dict = {}
        self.rate: float | None = None
        self.remote_state: dict = {}
        self.last_seq: int | None = None
        self.last_progress: float | None = None
        self.missing_reads = 0
        self.handled: set[str] = set()
        self.log_path = self.root / "logs" / f"controller-{time.strftime('%Y%m%d-%H%M%S')}.log"

    # -- logging ----------------------------------------------------------------------------------

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.echo(line)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -- remote paths -----------------------------------------------------------------------------

    @property
    def persistent_root(self) -> str:
        c = self.config["colab"]
        return f"{c['drive_mount'].rstrip('/')}/{c['persistent_path'].strip('/')}"

    def run_config(self) -> dict:
        """What the worker reads from ZIMG_RUN_FILE. Everything it needs, nothing secret."""
        c = self.config
        files = []
        for f in c["model"]["files"]:
            loader, field = ROLE_LOADERS[f["role"]]
            files.append(dict(f, loader=loader, loader_field=field))
        reference = client.build_zimage_txt2img_workflow(
            "reference", gc.SAFE_SAFETY_NEGATIVE, 0, "reference", model_files(c), 1024, 1024
        )
        comfy = dict(c["comfyui"])
        comfy["startup_timeout_seconds"] = c["colab"]["startup_timeout_seconds"]
        comfy["tarball_url"] = f"https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/{comfy['commit']}"
        return {
            "session": self.session,
            "local_root": REMOTE_ROOT,
            "drive_root": f"{c['colab']['drive_mount'].rstrip('/')}/MyDrive",
            "persistence": self.persistence,
            "persistent_root": self.persistent_root if self.persistence == "drive" else f"{REMOTE_ROOT}/persist",
            "worker": dict(c["worker"]),
            "comfyui": comfy,
            "model": {
                "name": c["model"]["name"],
                "repo": c["model"]["repo"],
                "revision": c["model"]["revision"],
                "files": files,
            },
            "models": dict(c["models"]),
            "gpu": dict(c["gpu"]),
            "custom_nodes": list(c.get("custom_nodes", [])),
            "reference_graph": reference,
        }

    # -- session lifecycle ------------------------------------------------------------------------

    def preflight(self) -> list[dict]:
        self.transport.preflight()
        self.usage_before = colab_usage(self.transport)
        sessions = colab_sessions(self.transport)
        self.log(
            f"compute units: {self.usage_before.get('balance')} "
            f"(rate {self.usage_before.get('rate_per_hour')}/hr, active {self.usage_before.get('active_assignments')})"
        )
        return sessions

    def check_busy(self, sessions: list[dict], ignore_busy: bool) -> None:
        others = [s for s in sessions if not s["name"].startswith(SESSION_PREFIX)]
        if others and not ignore_busy:
            names = ", ".join(s["name"] for s in others)
            raise ZImageError(
                "COLAB_BUSY",
                f"Another Colab runtime is active on this account ({names}), e.g. another Claude session's H3 job. "
                "Starting one now would compete for GPUs and mix the compute-unit readings.",
                hint="Ask the owner (ListAgents / SendMessage) and wait, or pass --ignore-busy.",
            )

    def sweep_orphans(self, sessions: list[dict]) -> None:
        """A zimg-* runtime with no live controller is billing for nothing: stop it."""
        for s in sessions:
            if s["name"].startswith(SESSION_PREFIX):
                self.log(f"stopping orphaned session {s['name']} (no controller is attached to it)")
                try:
                    stop_session(self.transport, s["name"])
                except (ColabCommandError, ColabTimeout) as exc:
                    self.log(f"WARNING: could not stop {s['name']}: {exc}")

    def open_session(self) -> None:
        c = self.config["colab"]
        self.session = f"{SESSION_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}"
        self.log(f"creating Colab session {self.session} ({c['gpu']}{', high-mem' if c['high_mem'] else ''})")
        self.opened = True
        self.t_open = self.clock()
        self.state_file.write(state="opening", session=self.session, opened_at=now_iso())
        self.awake(True)
        colab_new(
            self.transport,
            self.session,
            gpu=c["gpu"],
            high_mem=c["high_mem"],
            timeout=c["session_create_timeout_seconds"],
            on_line=lambda line: self.log(f"  {line}"),
        )
        try:
            during = colab_usage(self.transport)
            if during.get("rate_per_hour") is not None:
                self.rate = round(during["rate_per_hour"] - (self.usage_before.get("rate_per_hour") or 0.0), 3)
                self.log(f"session compute-unit rate: {self.rate}/hr")
        except ZImageError as exc:
            self.log(f"could not read the usage rate: {exc}")
        if c["drive"] == "consent":
            self.drive_outcome = self.mount_drive()
            if self.drive_outcome.get("mounted"):
                self.persistence = "drive"
                self.log(f"Google Drive mounted in {self.drive_outcome['seconds']} s: persistent storage in use")
            else:
                self.log(
                    f"Google Drive not mounted ({self.drive_outcome.get('reason')}): this session installs and "
                    "downloads from the pinned public sources instead (nothing is kept after it)"
                )
        else:
            self.log("colab.drive is off: this session installs and downloads from the pinned public sources")
        run_cfg = self.work / "run_config.json"
        write_json_atomic(run_cfg, self.run_config())
        upload_file(self.transport, self.session, run_cfg, REMOTE_RUN_CONFIG, timeout=c["transfer_timeout_seconds"])
        script = self.work / "zimage_worker.py"
        shutil.copyfile(REMOTE_SCRIPT, script)
        w = self.config["worker"]
        exec_timeout = w["max_session_seconds"] + c["bootstrap_timeout_seconds"] + 600
        args = ["exec", "--session", self.session, "--timeout", str(exec_timeout)]
        args += ["--env", f"ZIMG_RUN_FILE={REMOTE_RUN_CONFIG}", "--file", self.transport.cli_path(script, self.work)]
        self.exec = self.exec_factory(
            self.transport,
            args,
            mount_dir=self.work,
            timeout=exec_timeout + 300,
            log_path=self.root / "logs" / f"{self.session}.worker.log",
        )
        # From here on: no exec, no drivemount - the kernel is the worker's.
        self.state_file.write(state="running", session=self.session)
        self.last_progress = self.clock()

    def mount_drive(self) -> dict:
        """Ask for this VM's Drive consent (Colab wants it for every new runtime) and mount. The consent page
        opens in the default browser; `worker.py consent-done` (or the deadline) presses the Enter the CLI
        waits for. Never raises: no consent just means this session runs without persistence."""
        c = self.config["colab"]
        signal_path = self.root / "drive_consent.ok"
        with contextlib.suppress(FileNotFoundError):
            signal_path.unlink()

        def on_url(url: str) -> None:
            self.state_file.write(
                state="drive_consent",
                session=self.session,
                consent_url=url,
                consent_wait_seconds=c["drive_consent_wait_seconds"],
                consent_asked_at=now_iso(),
            )
            self.log(
                "DRIVE_CONSENT_NEEDED: approve Google Drive access in the browser page that just opened, "
                f"then run `worker.py consent-done` (Enter is sent anyway after {c['drive_consent_wait_seconds']} s)"
            )
            if c["open_browser"]:
                try:
                    self.open_url(url)
                except Exception as exc:  # noqa: BLE001 - the URL is in controller.json as well
                    self.log(f"could not open the browser: {exc}")

        self.log("mounting Google Drive (Colab asks for consent on every new runtime)")
        try:
            outcome = self.drive_mounter(
                self.transport,
                self.session,
                c["drive_mount"],
                wait_seconds=c["drive_consent_wait_seconds"],
                signal_path=signal_path,
                on_url=on_url,
                on_line=lambda line: self.log(f"  {line}"),
            )
        except (OSError, ColabCommandError, ColabTimeout, NotImplementedError) as exc:
            outcome = {"mounted": False, "reason": f"error: {exc}"}
        with contextlib.suppress(FileNotFoundError):
            signal_path.unlink()
        self.state_file.write(state="running", session=self.session, consent_url=None)
        return outcome

    def close_session(self, reason: str) -> dict:
        info: dict = {"reason": reason}
        if self.exec is not None:
            self.exec.stop_local()
        if self.opened and self.session:
            try:
                stop_session(self.transport, self.session)
                info["stop"] = "stopped"
                self.log(f"stopped Colab session {self.session}")
            except (ColabCommandError, ColabTimeout, KeyboardInterrupt) as exc:
                text = exc.text if isinstance(exc, ColabCommandError) else str(exc)
                info["stop"] = "not_found" if "not found" in text.lower() else "stop_failed"
                if info["stop"] == "stop_failed":
                    manual = self.transport.display(["stop", "--session", self.session])
                    info["warning"] = f"Colab session {self.session} may still be billing; stop it by hand: {manual}"
                    self.log("WARNING: " + info["warning"])
        self.awake(False)
        return info

    # -- polling ----------------------------------------------------------------------------------

    def read_remote_state(self) -> dict | None:
        local = self.work / "state.json"
        try:
            download_file(
                self.transport,
                self.session,
                f"{REMOTE_ROOT}/run/state.json",
                local,
                timeout=self.config["colab"]["transfer_timeout_seconds"],
            )
            data = json.loads(local.read_text(encoding="utf-8"))
        except (ColabCommandError, ColabTimeout, OSError, ValueError) as exc:
            self.missing_reads += 1
            if self.missing_reads in (1, 3) or self.missing_reads % 10 == 0:
                self.log(f"could not read the worker state ({self.missing_reads} in a row): {str(exc)[-200:]}")
            return None
        self.missing_reads = 0
        seq = data.get("seq")
        if seq != self.last_seq:
            self.last_seq = seq
            self.last_progress = self.clock()
        self.remote_state = data
        return data

    def push_pending(self) -> int:
        pushed = 0
        for job in self.store.list((PENDING,)):
            if job["attempt"] >= MAX_ATTEMPTS:
                self.store.update(
                    job["job_id"],
                    status=FAILED,
                    completed_at=now_iso(),
                    error={"code": "SESSION_LOST", "error_message": f"lost with its session {job['attempt']} times"},
                )
                continue
            job = self.store.update(job["job_id"], attempt=job["attempt"] + 1)
            payload_path = self.work / "inbox" / f"{job['job_id']}.json"
            write_json_atomic(payload_path, remote_payload(job, self.config))
            upload_file(
                self.transport,
                self.session,
                payload_path,
                f"{REMOTE_ROOT}/inbox/{job['job_id']}.json",
                timeout=self.config["colab"]["transfer_timeout_seconds"],
            )
            self.store.update(job["job_id"], status=QUEUED, queued_at=now_iso(), session=self.session)
            self.log(f"queued {job['job_id']} ({job['width']}x{job['height']}, seed {job['seed']})")
            pushed += 1
        return pushed

    def forward_cancels(self) -> None:
        if not self.store.cancel_dir.is_dir():
            return
        for path in self.store.cancel_dir.iterdir():
            job_id = path.name
            marker = self.work / "cancel" / job_id
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")
            try:
                upload_file(
                    self.transport,
                    self.session,
                    marker,
                    f"{REMOTE_ROOT}/cancel/{job_id}",
                    timeout=self.config["colab"]["transfer_timeout_seconds"],
                )
                path.unlink()
                self.log(f"cancel request for {job_id} sent to the worker")
            except (ColabCommandError, ColabTimeout) as exc:
                self.log(f"could not forward the cancel for {job_id}: {exc}")

    def sync_jobs(self, data: dict) -> None:
        remote_jobs = data.get("jobs") or {}
        for job_id, reason in (data.get("rejected") or {}).items():
            self._finish_failed(job_id, {"code": "INVALID_JOB", "error_message": f"worker could not read it: {reason}"})
        for job in self.store.list((QUEUED, RUNNING, GENERATED, VALIDATING, VALID)):
            job_id = job["job_id"]
            r = remote_jobs.get(job_id)
            if not r or job_id in self.handled:
                continue
            status = r.get("status")
            if status in (RUNNING, GENERATED, VALIDATING, VALID) and transition_allowed(job["status"], status):
                self.store.update(job_id, status=status, started_at=job.get("started_at") or now_iso())
            elif status == COMPLETED:
                self.collect(job, r)
            elif status in (FAILED, TIMEOUT, CANCELLED, INVALID_OUTPUT):
                self.collect_failure(job, r)

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _download_metadata(self, job_id: str) -> dict:
        local = self._job_dir(job_id) / "metadata.remote.json"
        try:
            download_file(
                self.transport,
                self.session,
                f"{REMOTE_ROOT}/outbox/{job_id}/metadata.json",
                local,
                timeout=self.config["colab"]["transfer_timeout_seconds"],
            )
            return json.loads(local.read_text(encoding="utf-8"))
        except (ColabCommandError, ColabTimeout, OSError, ValueError) as exc:
            self.log(f"could not download the metadata of {job_id}: {exc}")
            return {}

    def collect(self, job: dict, r: dict) -> None:
        job_id = job["job_id"]
        out_dir = self._job_dir(job_id)
        result = out_dir / "result.png"
        try:
            download_file(
                self.transport,
                self.session,
                f"{REMOTE_ROOT}/outbox/{job_id}/result.png",
                result,
                timeout=self.config["colab"]["transfer_timeout_seconds"],
            )
        except (ColabCommandError, ColabTimeout) as exc:
            self.log(f"download of {job_id} failed, will retry: {exc}")
            return
        self.handled.add(job_id)
        meta = self._download_metadata(job_id)
        write_json_atomic(out_dir / "workflow.json", job["graph"])
        self._advance(job_id, VALIDATING)
        problems = validate_output(result, job["width"], job["height"])
        fields = self._fields_from_remote(meta, r)
        fields["output"] = str(result)
        fields["validation"] = {
            "remote": meta.get("validation"),
            "local": {
                "ok": not problems,
                "problems": problems,
                "bytes": result.stat().st_size if result.exists() else 0,
            },
        }
        if problems:
            fields["error"] = {"code": "INVALID_OUTPUT", "error_message": "; ".join(problems), "timestamp": now_iso()}
            job = self.store.update(job_id, status=INVALID_OUTPUT, **fields)
            self.log(f"{job_id}: INVALID_OUTPUT after download: {problems}")
        else:
            self._advance(job_id, VALID)
            job = self.store.update(job_id, status=COMPLETED, **fields)
            self.log(
                f"{job_id}: COMPLETED in {fields.get('generation_seconds')} s "
                f"(VRAM peak {fields.get('vram_peak_mib')} MiB) -> {result}"
            )
        self._write_local_metadata(job, meta)

    def _advance(self, job_id: str, status: str) -> None:
        """Move forward to a lifecycle state unless the job is already there or past it: sync_jobs mirrors
        the worker's own VALIDATING/VALID, so by the time the image is downloaded either may be behind us."""
        if transition_allowed(self.store.get(job_id)["status"], status):
            self.store.update(job_id, status=status)

    def collect_failure(self, job: dict, r: dict) -> None:
        job_id = job["job_id"]
        self.handled.add(job_id)
        meta = self._download_metadata(job_id) if r.get("status") != CANCELLED else {}
        fields = self._fields_from_remote(meta, r)
        fields["error"] = meta.get("error") or r.get("error")
        job = self.store.update(job_id, status=r["status"], **fields)
        self.log(f"{job_id}: {r['status']} {((fields.get('error') or {}).get('error_message') or '')[:300]}")
        self._write_local_metadata(job, meta)

    def _finish_failed(self, job_id: str, error: dict) -> None:
        if job_id in self.handled:
            return
        try:
            job = self.store.get(job_id)
        except UsageError:
            return
        if job["status"] in TERMINAL:
            return
        self.handled.add(job_id)
        self.store.update(job_id, status=FAILED, completed_at=now_iso(), error=error)

    def _fields_from_remote(self, meta: dict, r: dict) -> dict:
        src = {**r, **meta}
        keys = (
            "started_at",
            "completed_at",
            "session_reused",
            "session_job_index",
            "gpu",
            "vram_total_mib",
            "vram_peak_mib",
            "generation_seconds",
            "comfy_execution_seconds",
            "job_seconds",
            "comfyui_commit",
        )
        fields = {k: src.get(k) for k in keys if src.get(k) is not None}
        fields["session"] = self.session
        return fields

    def _write_local_metadata(self, job: dict, remote_meta: dict) -> None:
        record = {k: v for k, v in job.items() if k != "graph"}
        record["remote_metadata"] = remote_meta
        write_json_atomic(self._job_dir(job["job_id"]) / "metadata.json", record)
        append_jsonl(self.root / "zimage_jobs.jsonl", ledger_line(job))

    def requeue_unfinished(self, reason: str) -> None:
        seen_remotely = (self.remote_state.get("jobs") or {}) if self.remote_state else {}
        for job in self.store.list((QUEUED, RUNNING, GENERATED, VALIDATING, VALID)):
            if job["job_id"] in self.handled:
                continue
            # A job the worker never listed (pushed as it was shutting down) did not really get an attempt.
            attempt = job["attempt"] if job["job_id"] in seen_remotely else max(0, job["attempt"] - 1)
            self.store.update(job["job_id"], status=PENDING, session=None, attempt=attempt, requeue_reason=reason)
            self.log(f"{job['job_id']} goes back to PENDING ({reason})")

    def loop(self) -> str:
        k = self.config["controller"]
        w = self.config["worker"]
        c = self.config["colab"]
        booted = False
        while True:
            data = self.read_remote_state()
            now = self.clock()
            if data is None and self.last_seq is None and self.exec is not None and self.exec.done.is_set():
                # The worker never wrote a state file and its exec is over: it never started.
                return f"exec_failed: {str(self.exec.error)[-300:]}" if self.exec.error else "exec_failed"
            if self.missing_reads >= k["max_missing_state_reads"] and (
                booted or now - self.t_open > c["bootstrap_timeout_seconds"]
            ):
                return "state_unreadable"
            if self.last_progress is not None and now - self.last_progress > k["heartbeat_timeout_seconds"]:
                return "heartbeat_stale"
            if now - self.t_open > w["max_session_seconds"] + c["bootstrap_timeout_seconds"]:
                return "max_session"
            stop_request = self.root / "stop.request"
            if stop_request.exists():
                stop_request.unlink()
                return "stop_requested"
            if data:
                ws = data.get("worker_state")
                if ws in ("READY", "BUSY", "IDLE"):
                    if not booted:
                        booted = True
                        self.log(
                            f"worker READY: setup {json.dumps(data.get('setup_actions'))} "
                            f"phases {json.dumps(data.get('phase_seconds'))}"
                        )
                    self.push_pending()
                    self.forward_cancels()
                elif ws == "BOOTING" and now - self.t_open > c["bootstrap_timeout_seconds"]:
                    return "bootstrap_timeout"
                self.sync_jobs(data)
                if ws == "FLUSHED":
                    return f"worker_{data.get('shutdown_reason') or 'shutdown'}"
            self.state_file.write(state="running", session=self.session, remote_seq=self.last_seq)
            self.sleep(k["poll_interval_seconds"])

    def run(self, *, ignore_busy: bool = False) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        live = self.state_file.live()
        if live and live.get("pid") != os.getpid():
            raise ZImageError(
                "CONTROLLER_RUNNING", f"a controller (pid {live['pid']}) already runs session {live.get('session')}"
            )
        started_at = now_iso()
        t0 = self.clock()
        reason = "error"
        close_info: dict = {}
        error: dict | None = None
        # No live controller (checked above by pid + heartbeat): a leftover lock is from one that crashed.
        with contextlib.suppress(FileNotFoundError):
            self.state_file.lock_path.unlink()
        with FileLock(self.state_file.lock_path, timeout=5, stale=24 * 3600):
            try:
                self.state_file.write(state="starting", session=None)
                sessions = self.preflight()
                self.check_busy(sessions, ignore_busy)
                self.sweep_orphans(sessions)
                if not self.store.list((PENDING, QUEUED, RUNNING)):
                    self.log("no queued jobs; nothing to start")
                    self.state_file.write(state="stopped")
                    return {"reason": "no_jobs"}
                self.requeue_unfinished("controller restarted")
                self.open_session()
                reason = self.loop()
            except ZImageError as exc:
                error = exc.as_dict()
                reason = exc.code
                self.log(f"ERROR {exc.code}: {exc} {exc.hint}")
            except KeyboardInterrupt:
                reason = "interrupted"
            finally:
                if self.opened:
                    if self.remote_state:
                        try:
                            self.sync_jobs(self.remote_state)
                        except (ZImageError, OSError) as exc:
                            self.log(f"final sync failed: {exc}")
                    close_info = self.close_session(reason)
                    self.requeue_unfinished(reason)
                summary = self.finish(reason, started_at, t0, close_info, error)
                self.state_file.write(state="stopped", session=self.session, reason=reason)
        return summary

    def finish(self, reason: str, started_at: str, t0: float, close_info: dict, error: dict | None) -> dict:
        summary: dict = {
            "session": self.session,
            "reason": reason,
            "error": error,
            "started_at": started_at,
            "ended_at": now_iso(),
            "controller_seconds": round(self.clock() - t0, 1),
            "session_seconds": round(self.clock() - self.t_open, 1) if self.t_open is not None else None,
            "gpu": self.config["colab"]["gpu"],
            "persistence": self.persistence if self.opened else None,
            "drive": self.drive_outcome,
            "setup_actions": self.remote_state.get("setup_actions"),
            "phase_seconds": self.remote_state.get("phase_seconds"),
            "model_stage_seconds": self.remote_state.get("model_stage_seconds"),
            "drive_flush_seconds": self.remote_state.get("drive_flush_seconds"),
            "worker_gpu": self.remote_state.get("gpu"),
            "jobs": {s: len([j for j in self.store.list((s,))]) for s in STATUSES},
            "handled_this_session": sorted(self.handled),
            "cu_balance_before": self.usage_before.get("balance"),
            "cu_rate_per_hour": self.rate,
            **close_info,
        }
        if self.opened:
            try:
                after = colab_usage(self.transport).get("balance")
                summary["cu_balance_after"] = after
                settle = self.config["colab"]["cu_settle_seconds"]
                if settle:
                    self.log(f"waiting {settle} s for Colab's late compute-unit deductions")
                    self.sleep(settle)
                    summary["cu_balance_settled"] = colab_usage(self.transport).get("balance")
                end = summary.get("cu_balance_settled", after)
                if summary["cu_balance_before"] is not None and end is not None:
                    summary["cu_used_measured"] = round(summary["cu_balance_before"] - end, 3)
            except ZImageError as exc:
                self.log(f"could not read the final balance: {exc}")
            if self.rate is not None and summary["session_seconds"] is not None:
                summary["cu_estimated"] = round(self.rate * summary["session_seconds"] / 3600, 3)
            append_jsonl(self.root / "zimage_sessions.jsonl", summary)
        self.log("session summary: " + json.dumps(summary, ensure_ascii=False, default=str))
        return summary


LEDGER_FIELDS = (
    "job_id",
    "status",
    "gpu",
    "vram_total_mib",
    "vram_peak_mib",
    "model",
    "model_revision",
    "workflow",
    "width",
    "height",
    "steps",
    "cfg",
    "sampler",
    "seed",
    "started_at",
    "completed_at",
    "generation_seconds",
    "comfy_execution_seconds",
    "job_seconds",
    "output",
    "session",
    "session_reused",
    "session_job_index",
    "attempt",
    "batch",
)


def ledger_line(job: dict) -> dict:
    """The spec's per-job measurement record (section 31) plus what is needed to compare sessions."""
    line = {k: job.get(k) for k in LEDGER_FIELDS}
    line["vram"] = job.get("vram_peak_mib")
    line["error_code"] = (job.get("error") or {}).get("code")
    return line


# --- detaching -----------------------------------------------------------------------------------


def spawn_detached(args: list[str], log_path: Path) -> int:
    """Start `python worker.py up --foreground` outside this process tree, so the controller survives the
    Claude session or terminal that started it (it must outlive them to stop the VM)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(args, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:  # the parent's job object forbids breakaway
            proc = subprocess.Popen(args, creationflags=flags, **kwargs)
    else:
        proc = subprocess.Popen(args, start_new_session=True, **kwargs)
    log.close()
    return proc.pid
