"""MiniMax H3 image-to-video on a Google Colab GPU, driven through google-colab-cli.

One job = one image + one prompt -> one MP4. Why it is shaped this way:

- google-colab-cli supports Linux/macOS only (its console module imports termios at startup), so on
  Windows every `colab` call runs in a short-lived Docker container. The OAuth token and session
  metadata live in a named Docker volume, never in this repo or its config files.
- The GPU work is remote/h3_colab_job.py, sent with `colab exec -f`. `colab exec` does not exit
  non-zero when the remote code raises, so success is decided by the H3_* marker lines the remote
  script prints, never by the exit code alone.
- Everything checkable locally (image decodes, prompt, duration, resolution, ffprobe, Docker, auth)
  is checked before `colab new`: every minute after that spends compute units.
- A timed-out `colab exec` is never retried - the remote kernel may still be working and a retry would
  pay twice. The session is always stopped in `finally`.

Stdlib only, Python 3.11+: the local machine never loads the model.
"""

from __future__ import annotations

import dataclasses
import json
import math
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
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

SKILL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SKILL_DIR.parents[2]
REMOTE_SCRIPT = SKILL_DIR / "remote" / "h3_colab_job.py"
EXAMPLE_CONFIG = SKILL_DIR / "config" / "config.example.json"
LOCAL_CONFIG = SKILL_DIR / "config" / "config.json"
DOCKERFILE_DIR = SKILL_DIR / "docker"

FPS = 24
MIN_DURATION = 4.0
MAX_DURATION = 15.0
# ComfyUI's H3 node: "trained range is ~124-362" frames; 124 frames is what a 5 s request becomes.
STABLE_MIN_DURATION = 5.0
MAX_IMAGE_BYTES = 50 * 2**20
MAX_PROMPT_CHARS = 4000
# Beyond this the first frame is visibly stretched onto the video canvas.
ASPECT_TOLERANCE = 0.02
GPUS = ("T4", "L4", "G4", "H100", "A100")
# "consent": mount Google Drive on the VM (the user approves it in the browser, once per VM) so models and
# clips are stored there. "off": nothing is stored on Drive - every session downloads the models again.
DRIVE_MODES = ("consent", "off")
PERSIST_DRIVE, PERSIST_EPHEMERAL = "drive", "ephemeral"
CONSENT_SIGNAL = "h3_drive_consent.signal"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_CODECS = ("png", "mjpeg", "webp", "bmp")
FINALIZE_ENV = "H3_FINALIZE"  # must match remote/h3_colab_job.py; tests assert it
JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Sampling settings of the turbo LoRA pairing (see remote/h3_colab_job.py).
STEPS = 8
SAMPLER = "euler"
SCHEDULER = "beta"
LORA_STRENGTH = 1.0
VIDEO_CRF = 20

PENDING = "PENDING"
PREPARING = "PREPARING"
CONNECTING_COLAB = "CONNECTING_COLAB"
UPLOADING = "UPLOADING"
LOADING_MODEL = "LOADING_MODEL"
INFERENCE = "INFERENCE"
DOWNLOADING = "DOWNLOADING"
VALIDATING = "VALIDATING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
CANCELLED = "CANCELLED"
FAILED_VALIDATION = "FAILED_VALIDATION"
DRY_RUN = "DRY_RUN"
QUEUED = "QUEUED"  # stage_seconds key only: a batch job waiting for the jobs before it; never a status
LIFECYCLE = (PENDING, PREPARING, CONNECTING_COLAB, UPLOADING, LOADING_MODEL, INFERENCE, DOWNLOADING, VALIDATING)
TERMINAL = (COMPLETED, FAILED, TIMEOUT, CANCELLED, FAILED_VALIDATION)

AUTH_REQUIRED_MESSAGE = "Google Colab authentication is required.\nPlease authenticate and run the job again."
GPU_UNAVAILABLE_MESSAGE = "No compatible GPU is currently available."


# --- errors -------------------------------------------------------------------------------------


class H3Error(Exception):
    """A job failure with a machine-readable code; status is the terminal state it maps to."""

    def __init__(self, code: str, message: str, *, status: str = FAILED, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.status = status
        self.hint = hint


class InputError(H3Error):
    """The caller's inputs are wrong. Raised while PREPARING, before any Colab call."""

    def __init__(self, message: str, *, code: str = "INVALID_INPUT", hint: str = ""):
        super().__init__(code, message, status=FAILED, hint=hint)


class SessionLost(H3Error):
    """`colab exec` failed without a remote marker, so the Colab session is in an unknown state. The job
    fails like any other; a batch also stops that session and gives the next job a fresh one."""


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


class ProbeError(Exception):
    pass


# --- config -------------------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(value)


ENV_OVERRIDES: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "H3_TRANSPORT": ("colab", "transport", str),
    "H3_DOCKER_IMAGE": ("colab", "docker_image", str),
    "H3_GPU": ("colab", "gpu", str),
    "H3_HIGH_MEM": ("colab", "high_mem", _parse_bool),
    "H3_TIMEOUT_SECONDS": ("colab", "timeout_seconds", int),
    "H3_OUTPUT_DIR": ("output", "directory", str),
    "H3_FFPROBE": ("tools", "ffprobe", str),
    "H3_CU_SETTLE_SECONDS": ("colab", "cu_settle_seconds", int),
    "H3_EXEC_IDLE_TIMEOUT_SECONDS": ("colab", "exec_idle_timeout_seconds", int),
    "H3_DRIVE": ("drive", "mode", str),
    "H3_DRIVE_WORKSPACE": ("drive", "workspace", str),
}


def _merge(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def secret_like_keys(obj: Any, prefix: str = "") -> list[str]:
    """Same rule as training/cloud_video.py: credentials never live in a settings file."""
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
        raise InputError(message, code="INVALID_CONFIG")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(config: dict) -> dict:
    c = config["colab"]
    _check(c.get("transport") in ("auto", "docker", "native"), "colab.transport must be auto, docker or native")
    _check(c.get("auth") in ("oauth2", "adc"), "colab.auth must be oauth2 or adc")
    _check(str(c.get("gpu", "")).upper() in GPUS, f"colab.gpu must be one of {', '.join(GPUS)}")
    c["gpu"] = str(c["gpu"]).upper()
    _check(isinstance(c.get("high_mem"), bool), "colab.high_mem must be true or false")
    for key in ("timeout_seconds", "session_create_timeout_seconds", "transfer_timeout_seconds"):
        _check(_is_int(c.get(key)) and 60 <= c[key] <= 6 * 3600, f"colab.{key} must be an integer 60-21600")
    settle = c.get("cu_settle_seconds")
    _check(_is_int(settle) and 0 <= settle <= 3600, "colab.cu_settle_seconds must be an integer 0-3600")
    idle = c.get("exec_idle_timeout_seconds")
    _check(_is_int(idle) and 300 <= idle <= 3600, "colab.exec_idle_timeout_seconds must be an integer 300-3600")
    for key in ("docker_image", "config_volume"):
        _check(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-/:]*", str(c.get(key, "")))), f"colab.{key} is invalid")
    v = config["video"]
    d = v.get("default_duration")
    _check(
        isinstance(d, (int, float)) and not isinstance(d, bool) and MIN_DURATION <= d <= MAX_DURATION,
        "video.default_duration must be 4-15",
    )
    s = v.get("short_side")
    _check(_is_int(s) and s % 32 == 0 and 256 <= s <= 2048, "video.short_side must be a multiple of 32, 256-2048")
    mp = v.get("max_megapixels")
    _check(isinstance(mp, (int, float)) and 0.1 <= mp <= 4, "video.max_megapixels must be 0.1-4")
    _check(isinstance(config["output"].get("directory"), str) and config["output"]["directory"], "output.directory")
    _check(isinstance(config["tools"].get("ffprobe", ""), str), "tools.ffprobe must be a path string")
    d = config["drive"]
    _check(d.get("mode") in DRIVE_MODES, f"drive.mode must be one of {', '.join(DRIVE_MODES)}")
    _check(
        bool(re.fullmatch(r"/content/[A-Za-z0-9_\-]+", str(d.get("mount", "")))), "drive.mount must be /content/<name>"
    )
    workspace = str(d.get("workspace", ""))
    _check(
        bool(re.fullmatch(r"[A-Za-z0-9_\- ]+(/[A-Za-z0-9_\- ]+)*", workspace)) and ".." not in workspace,
        "drive.workspace must be a relative path under the mount, e.g. MyDrive/AI-Workflow",
    )
    wait = d.get("consent_wait_seconds")
    # drive.mount gives up after 120 s; Enter has to be sent before that.
    _check(_is_int(wait) and 10 <= wait <= 110, "drive.consent_wait_seconds must be an integer 10-110")
    final = d.get("finalize_timeout_seconds")
    _check(_is_int(final) and 60 <= final <= 7200, "drive.finalize_timeout_seconds must be an integer 60-7200")
    _check(isinstance(d.get("open_browser"), bool), "drive.open_browser must be true or false")
    models = config.get("models")
    _check(isinstance(models, list) and len(models) >= 5, "models must list the pinned model files")
    for m in models:
        _check(isinstance(m, dict), "each models entry must be an object")
        for key in ("name", "folder", "store", "repo", "path_in_repo"):
            _check(
                isinstance(m.get(key), str) and m[key] and ".." not in m[key], f"models: {key} is missing or invalid"
            )
        _check(
            bool(re.fullmatch(r"[0-9a-f]{40}", str(m.get("revision", "")))),
            f"models: {m['name']} needs a pinned revision",
        )
        _check(bool(re.fullmatch(r"[0-9a-f]{64}", str(m.get("sha256", "")))), f"models: {m['name']} needs a sha256")
        _check(_is_int(m.get("size")) and m["size"] > 0, f"models: {m['name']} needs a size in bytes")
    _check(len({m["name"] for m in models}) == len(models), "models: a file name appears twice")
    return config


def load_config(path: Path | None = None, env: dict | None = None) -> dict:
    """config.example.json <- config/config.json (gitignored, optional) <- H3_* environment variables.
    Command-line flags override the result in run.py."""
    env = os.environ if env is None else env
    config = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    local = LOCAL_CONFIG if path is None else Path(path)
    if local.is_file():
        try:
            data = json.loads(local.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise InputError(f"{local} is not valid JSON: {exc}", code="INVALID_CONFIG") from exc
        _check(isinstance(data, dict), f"{local} must contain a JSON object")
        found = secret_like_keys(data)
        _check(
            not found,
            f"{local} has secret-looking keys {found}. Credentials never go in config files; "
            "google-colab-cli keeps its OAuth token in its own config volume.",
        )
        config = _merge(config, data)
    for var, (section, key, convert) in ENV_OVERRIDES.items():
        if env.get(var):
            try:
                config[section][key] = convert(env[var])
            except ValueError as exc:
                raise InputError(f"{var}={env[var]!r} is not valid", code="INVALID_CONFIG") from exc
    return validate_config(config)


def project_path(value: str | Path) -> Path:
    """Config paths are relative to the repo root, so the skill behaves the same from any cwd."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def output_dir(config: dict) -> Path:
    return project_path(config["output"]["directory"])


def consent_signal_path(config: dict) -> Path:
    """scripts/consent_done.py creates this file; the runner then answers the CLI's "press Enter" at once."""
    return output_dir(config) / CONSENT_SIGNAL


def find_ffprobe(config: dict) -> str | None:
    configured = config["tools"].get("ffprobe") or ""
    if configured:
        path = project_path(configured)
        return str(path) if path.is_file() else None
    return shutil.which("ffprobe")


# --- running the CLI ------------------------------------------------------------------------------

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
) -> list[str]:
    """Run cmd, feed each output line to on_line as it arrives, enforce a wall-clock timeout, and with
    idle_timeout also a limit on silence. The silence limit exists because `colab exec` can lose its
    connection and then hang without exiting (measured 2026-10-01: "RuntimeError: Connection was lost"
    and nothing for an hour while the A100 kept billing); the remote script prints at least every 30 s
    while it renders, so long silence means the call is dead.

    stdin is /dev/null on purpose: if the CLI falls into its interactive OAuth prompt it hits EOF
    and fails fast (-> AUTH_REQUIRED) instead of hanging the job. On timeout, Ctrl+C or a failing
    callback, on_abort runs first (for Docker: `docker kill`, because killing the docker client does
    not stop the container), then the process tree is killed.
    """
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
            if now > deadline:
                raise ColabTimeout(f"{label} exceeded {timeout:g} seconds")
            if idle_timeout is not None and now - last_output > idle_timeout:
                raise ColabTimeout(f"{label} printed nothing for {idle_timeout:g} seconds (connection lost?)")
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
    """Where the `colab` process runs. Subclasses build the command line and map local paths."""

    kind = "base"

    def __init__(self, auth: str):
        self.auth = auth

    def command(self, args: list[str], *, mount_dir: Path | None, name: str) -> list[str]:
        raise NotImplementedError

    def cli_path(self, local: Path, mount_dir: Path) -> str:
        raise NotImplementedError

    def abort(self, name: str) -> None:
        pass

    def preflight(self) -> None:
        pass

    def login_command(self) -> str:
        raise NotImplementedError

    def display(self, args: list[str], mount_dir: Path | None = None) -> str:
        return subprocess.list2cmdline(self.command(list(args), mount_dir=mount_dir, name="h3-manual"))

    def interactive_command(self, args: list[str], *, name: str) -> list[str]:
        """`colab <args>` with a pseudo-terminal whose input is our piped stdin (for `drivemount`)."""
        raise NotImplementedError

    def mount_drive(
        self,
        session: str,
        mount: str,
        *,
        wait_seconds: float,
        signal_path: Path,
        on_url: Callable[[str], None],
        on_line: Callable[[str], None] | None = None,
    ) -> dict:
        """Mount Google Drive on the VM. Tests replace this."""
        return drive_consent_mount(
            self, session, mount, wait_seconds=wait_seconds, signal_path=signal_path, on_url=on_url, on_line=on_line
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
    ) -> str:
        name = f"h3-{uuid.uuid4().hex[:12]}"
        cmd = self.command(list(args), mount_dir=mount_dir, name=name)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        lines = run_streaming(
            cmd,
            label=label,
            timeout=timeout,
            on_line=on_line,
            on_abort=lambda: self.abort(name),
            env=env,
            idle_timeout=idle_timeout,
        )
        return "\n".join(line for line in lines if line)


class DockerTransport(Transport):
    kind = "docker"
    CONFIG_MOUNT = "/root/.config/colab-cli"
    WORK_MOUNT = "/work"

    def __init__(self, image: str, volume: str, auth: str, docker: str = "docker"):
        super().__init__(auth)
        self.image = image
        self.volume = volume
        self.docker = docker

    def command(self, args: list[str], *, mount_dir: Path | None, name: str) -> list[str]:
        # PYTHONUNBUFFERED: without a TTY the CLI's stdout is block-buffered and the H3_* markers
        # would only arrive when the whole `colab exec` finishes.
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

    def interactive_command(self, args: list[str], *, name: str) -> list[str]:
        # util-linux `script` gives the CLI a /dev/tty; -i keeps our stdin attached to it.
        inner = shlex.join(["colab", f"--auth={self.auth}", *args])
        cmd = [self.docker, "run", "-i", "--rm", "--name", name, "-e", "PYTHONUNBUFFERED=1"]
        cmd += ["-v", f"{self.volume}:{self.CONFIG_MOUNT}", "--entrypoint", "script"]
        return cmd + [self.image, "-qfec", inner, "/dev/null"]

    def build_command(self) -> str:
        return subprocess.list2cmdline([self.docker, "build", "-t", self.image, str(DOCKERFILE_DIR)])

    def login_command(self) -> str:
        return subprocess.list2cmdline(
            [self.docker, "run", "--rm", "-it", "-v", f"{self.volume}:{self.CONFIG_MOUNT}"]
            + [self.image, f"--auth={self.auth}", "usage"]
        )

    def preflight(self) -> None:
        if not shutil.which(self.docker):
            raise H3Error("DOCKER_UNAVAILABLE", "Docker is not installed or not on PATH.")
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
            raise H3Error("DOCKER_UNAVAILABLE", "`docker info` did not answer within 60 s.") from exc
        if info.returncode != 0:
            raise H3Error(
                "DOCKER_UNAVAILABLE", "Docker Desktop is not running. Start Docker Desktop and run the job again."
            )
        image = subprocess.run(
            [self.docker, "image", "inspect", self.image],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if image.returncode != 0:
            raise H3Error(
                "DOCKER_UNAVAILABLE", f"Docker image {self.image} is not built yet.", hint=self.build_command()
            )


class NativeTransport(Transport):
    kind = "native"

    def __init__(self, auth: str, colab: str = "colab"):
        super().__init__(auth)
        self.colab = colab

    def command(self, args: list[str], *, mount_dir: Path | None, name: str) -> list[str]:
        return [self.colab, f"--auth={self.auth}", *args]

    def cli_path(self, local: Path, mount_dir: Path) -> str:
        return str(Path(local).resolve())

    def interactive_command(self, args: list[str], *, name: str) -> list[str]:
        return ["script", "-qfec", shlex.join([self.colab, f"--auth={self.auth}", *args]), "/dev/null"]

    def login_command(self) -> str:
        return f"{self.colab} --auth={self.auth} usage"

    def preflight(self) -> None:
        if not shutil.which(self.colab):
            raise H3Error(
                "COLAB_CLI_MISSING",
                "google-colab-cli is not installed.",
                hint="uv tool install --python 3.12 google-colab-cli==0.7.4",
            )


def make_transport(config: dict) -> Transport:
    c = config["colab"]
    kind = c["transport"]
    if kind == "auto":
        kind = "docker" if os.name == "nt" or not shutil.which("colab") else "native"
    if kind == "docker":
        return DockerTransport(c["docker_image"], c["config_volume"], c["auth"])
    return NativeTransport(c["auth"])


# --- colab operations ----------------------------------------------------------------------------

_AUTH_RE = re.compile(
    r"To authorize colab-cli|authorization code|EOFError|invalid_grant|RefreshError|reauthenticat"
    r"|DefaultCredentialsError|token has been expired or revoked",
    re.IGNORECASE,
)
# The last alternative is a 503 from the assignment endpoint: no capacity for that accelerator right now
# (measured 2026-09-30 on an A100 high-mem request, 1 minute after the previous A100 was released).
_GPU_RE = re.compile(
    r"Backend rejected accelerator|Allocation refused|quota or entitlement"
    r"|tun/m/assign[\s\S]{0,400}?(?:Service Unavailable|Too Many Requests)",
    re.IGNORECASE,
)


def auth_error(transport: Transport) -> H3Error:
    return H3Error(
        "AUTH_REQUIRED",
        AUTH_REQUIRED_MESSAGE,
        hint=f"Run this once in your own terminal and paste the code Google shows: {transport.login_command()}",
    )


def parse_usage(text: str) -> dict:
    """`colab usage` prints labelled lines (v0.7.4 has no JSON option)."""

    def number(pattern: str) -> float | None:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return float(m.group(1).replace(",", "")) if m else None

    return {
        "balance": number(r"^Current balance:\s*([\d,]+(?:\.\d+)?)\s+compute units"),
        "rate_per_hour": number(r"^Usage rate:\s*([\d,]+(?:\.\d+)?)\s*/\s*hr"),
        "active_assignments": number(r"^Active assignments:\s*(\d+)"),
    }


def colab_usage(transport: Transport, *, timeout: float = 90) -> dict:
    try:
        out = transport.call(["usage"], label="colab usage", timeout=timeout)
    except ColabCommandError as exc:
        if _AUTH_RE.search(exc.text):
            raise auth_error(transport) from exc
        raise H3Error("COLAB_CONNECTION_FAILED", f"Could not reach Colab: {exc}") from exc
    except ColabTimeout as exc:
        raise H3Error("COLAB_CONNECTION_FAILED", f"Could not reach Colab: {exc}") from exc
    return parse_usage(out)


def colab_new(
    transport: Transport,
    session: str,
    *,
    gpu: str,
    high_mem: bool,
    timeout: float,
    on_line: Callable[[str], None] | None = None,
) -> str:
    args = ["new", "--session", session, "--gpu", gpu] + (["--high-mem"] if high_mem else [])
    try:
        return transport.call(args, label="colab new", timeout=timeout, on_line=on_line)
    except ColabCommandError as exc:
        if _GPU_RE.search(exc.text):
            raise H3Error("GPU_UNAVAILABLE", GPU_UNAVAILABLE_MESSAGE, hint=exc.text[-800:]) from exc
        if _AUTH_RE.search(exc.text):
            raise auth_error(transport) from exc
        raise H3Error("COLAB_CONNECTION_FAILED", f"Could not create the Colab session: {exc}") from exc


def upload_file(transport: Transport, session: str, local: Path, remote: str, *, timeout: float, on_line=None) -> str:
    local = Path(local).resolve()
    return transport.call(
        ["upload", "--session", session, transport.cli_path(local, local.parent), remote],
        label=f"upload {local.name}",
        timeout=timeout,
        mount_dir=local.parent,
        on_line=on_line,
    )


def download_file(transport: Transport, session: str, remote: str, local: Path, *, timeout: float, on_line=None) -> str:
    local = Path(local).resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    return transport.call(
        ["download", "--session", session, remote, transport.cli_path(local, local.parent)],
        label=f"download {remote}",
        timeout=timeout,
        mount_dir=local.parent,
        on_line=on_line,
    )


CONSENT_URL_RE = re.compile(r"^https://accounts\.google\.com/\S+$")


def _open_in_browser(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


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
    """Mount Drive the way Colab allows it from a CLI (the same procedure as the z-image-colab skill):
    Google asks for consent on every new VM (measured 2026-10-01) and drive.mount waits only 120 s for it.
    The CLI prints the consent URL and then reads Enter from /dev/tty, so it runs under a pseudo-terminal
    with our stdin piped in. Enter goes in when the user confirmed (signal_path appears, see
    scripts/consent_done.py) or after wait_seconds, whichever is first. The URL is handed to on_url and is
    never written to the log."""
    name = f"h3-{uuid.uuid4().hex[:12]}"
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
    t_url = 0.0
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
        "seconds": round(time.monotonic() - t0, 1),
        "reason": reason,
        "tail": text[-600:] if not mounted else None,
    }


def stop_session(transport: Transport, session: str, *, timeout: float = 300, on_line=None) -> str:
    return transport.call(["stop", "--session", session], label="colab stop", timeout=timeout, on_line=on_line)


MARKER_RE = re.compile(r"^H3_([A-Z_]+)\s+(\{.*\})\s*$")


def parse_marker(line: str) -> tuple[str, dict] | None:
    m = MARKER_RE.match(line.strip())
    if not m:
        return None
    try:
        data = json.loads(m.group(2))
    except ValueError:
        return None
    return (m.group(1), data) if isinstance(data, dict) else None


# --- media probing and validation ------------------------------------------------------------------

VIDEO_ENTRIES = (
    "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,pix_fmt,"
    "r_frame_rate,avg_frame_rate,nb_frames,duration,sample_rate,channels"
)


class FFprobe:
    def __init__(self, path: str):
        self.path = path

    def _run(self, args: list[str], target: Path, timeout: float = 60) -> dict:
        try:
            r = subprocess.run(
                [self.path, "-v", "error", *args, "-of", "json", str(target)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProbeError(f"could not run ffprobe: {exc}") from exc
        if r.returncode != 0:
            raise ProbeError(r.stderr.strip()[-500:] or f"ffprobe exited {r.returncode}")
        try:
            return json.loads(r.stdout or "{}")
        except ValueError as exc:
            raise ProbeError(f"unreadable ffprobe output: {exc}") from exc

    def version(self) -> str:
        r = subprocess.run([self.path, "-version"], capture_output=True, text=True, timeout=30, check=False)
        if r.returncode != 0:
            raise ProbeError(f"ffprobe -version exited {r.returncode}")
        return (r.stdout.splitlines() or [""])[0]

    def image_info(self, path: Path) -> tuple[int, int, str]:
        data = self._run(["-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height"], path)
        streams = data.get("streams") or []
        if not streams or not streams[0].get("width") or not streams[0].get("height"):
            raise ProbeError("no image stream")
        s = streams[0]
        return int(s["width"]), int(s["height"]), str(s.get("codec_name", ""))

    def video(self, path: Path) -> dict:
        return self._run(["-show_entries", VIDEO_ENTRIES], path)


def _ratio(value: Any) -> float | None:
    try:
        num, _, den = str(value).partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def _float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def summarize_probe(probe: dict) -> dict:
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = probe.get("format", {})
    return {
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "pix_fmt": video.get("pix_fmt"),
        "fps": _ratio(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "nb_frames": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None,
        "video_duration": _float(video.get("duration")),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio["sample_rate"]) if str(audio.get("sample_rate", "")).isdigit() else None,
        "audio_channels": audio.get("channels"),
        "format_duration": _float(fmt.get("duration")),
        "size": int(fmt["size"]) if str(fmt.get("size", "")).isdigit() else None,
    }


def check_probe(probe: dict, *, width: int, height: int, frames: int) -> list[str]:
    """Problems with a downloaded clip; empty means it passes. Pure, so every rule is unit-tested."""
    s = summarize_probe(probe)
    if s["video_codec"] is None:
        return ["no video stream"]
    problems = []
    if s["size"] == 0:
        problems.append("file is empty")
    if s["video_codec"] != "h264":
        problems.append(f"video codec {s['video_codec']}, expected h264")
    if s["pix_fmt"] != "yuv420p":
        problems.append(f"pixel format {s['pix_fmt']}, expected yuv420p")
    if (s["width"], s["height"]) != (width, height):
        problems.append(f"resolution {s['width']}x{s['height']}, expected {width}x{height}")
    if s["fps"] is None or abs(s["fps"] - FPS) > 0.01:
        problems.append(f"frame rate {s['fps']}, expected {FPS}")
    expected = frames / FPS
    actual = s["video_duration"]
    if actual is None and s["nb_frames"] and s["fps"]:
        actual = s["nb_frames"] / s["fps"]
    if actual is None:
        actual = s["format_duration"]
    tolerance = max(0.1, 2 / FPS)
    if actual is None or abs(actual - expected) > tolerance:
        problems.append(f"duration {actual} s, expected {expected:.3f} s (+-{tolerance:.2f})")
    if s["audio_codec"] is None:
        problems.append("no audio stream (H3 always generates audio)")
    return problems


# --- durations and resolution ----------------------------------------------------------------------


def frames_for(duration: float) -> int:
    """Requested seconds -> frames on H3's 17k+5 grid at 24 fps (ComfyUI's H3 node snaps the same way):
    4 -> 107, 5 -> 124, 8 -> 192, 12 -> 294, 15 -> 362."""
    requested = max(5, round(duration * FPS))
    return requested + (5 - requested % 17) % 17


def round32(value: float) -> int:
    return max(32, int(round(value / 32)) * 32)


def auto_resolution(img_w: int, img_h: int, *, short_side: int, max_megapixels: float) -> tuple[int, int]:
    """Keep the image's aspect ratio (so the first frame is not stretched), short side 768 by default
    (H3's native 768p), shrinking the short side in steps of 32 until under the pixel budget."""
    side = short_side
    while True:
        if img_w >= img_h:
            w, h = round32(side * img_w / img_h), side
        else:
            w, h = side, round32(side * img_h / img_w)
        if w * h <= max_megapixels * 1_000_000 or side <= 256:
            return w, h
        side -= 32


def parse_resolution(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", text)
    if not m:
        raise InputError(f"resolution {text!r} must look like 1376x768 or be 'auto'")
    return int(m.group(1)), int(m.group(2))


def resolve_resolution(requested: str, img_w: int, img_h: int, video_cfg: dict) -> tuple[int, int]:
    short_side, max_mp = video_cfg["short_side"], video_cfg["max_megapixels"]
    if not requested or requested.strip().lower() == "auto":
        return auto_resolution(img_w, img_h, short_side=short_side, max_megapixels=max_mp)
    w, h = parse_resolution(requested)
    suggestion = "x".join(map(str, auto_resolution(img_w, img_h, short_side=short_side, max_megapixels=max_mp)))
    if w % 32 or h % 32:
        raise InputError(f"width and height must be multiples of 32 (got {w}x{h}); try {suggestion}")
    if not (256 <= w <= 2048 and 256 <= h <= 2048):
        raise InputError(f"each side must be 256-2048 px (got {w}x{h}); try {suggestion}")
    if w * h > max_mp * 1_000_000:
        raise InputError(f"{w}x{h} is over the {max_mp} MP budget (video.max_megapixels); try {suggestion}")
    mismatch = abs((w / h) / (img_w / img_h) - 1)
    if mismatch > ASPECT_TOLERANCE:
        raise InputError(
            f"{w}x{h} does not match the image's {img_w}x{img_h} aspect ratio ({mismatch:.1%} off), "
            f"so the first frame would be stretched; use {suggestion} or --resolution auto"
        )
    return w, h


# --- prompt ------------------------------------------------------------------------------------------

# Official MiniMax H3 prompt guide, single first frame (I2VA): this alignment line, a blank line, then
# the three fields. The runner never rewrites the user's shot description; it only wraps it.
I2VA_ALIGNMENT = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
)
DEFAULT_SOUNDSCAPE = "Soft ambient sound that matches the scene."
DEFAULT_MUSIC = "N/A"
CONSTRAINTS = ("locked-camera", "static-subject")
LOCKED_CAMERA_SENTENCE = "The camera remains static in a locked-off shot."
STATIC_SUBJECT_SENTENCE = "The subject stays in place, holding the same position with only subtle natural motion."

# Camera moves that mean a moving camera wherever they appear.
_CAMERA_MOVES = (
    "zoom", "zooms", "zooming", "dolly", "dollies", "dollying", "handheld", "hand-held", "tracking shot",
    "arc shot", "orbit", "orbits", "orbiting", "crane shot", "whip pan", "shaky cam",
)  # fmt: skip
# Verbs that are camera moves only in a sentence about the camera ("she tilts her head" is fine).
_CAMERA_VERBS = (
    "pan", "pans", "panning", "tilt", "tilts", "tilting", "truck", "trucks", "trucking", "push in",
    "pushes in", "pushing in", "pull out", "pulls out", "pull back", "pulls back", "pulling back", "move",
    "moves", "moving", "follow", "follows", "following", "track", "tracks", "tracking", "rise", "rises",
    "descend", "descends", "sweep", "sweeps", "circle", "circles", "shake", "shakes", "rotate", "rotates",
    "drift", "drifts", "glide", "glides", "approach", "approaches",
)  # fmt: skip
_LOCOMOTION = (
    "walk", "walks", "walking", "walked", "run", "runs", "running", "jog", "jogs", "jogging", "stride",
    "strides", "striding", "stroll", "strolls", "strolling", "step forward", "steps forward", "step toward",
    "steps toward", "steps towards", "step closer", "steps closer", "approach", "approaches", "approaching",
    "dance", "dances", "dancing", "jump", "jumps", "jumping", "turns around", "spins", "spinning",
    "moves toward", "moves towards", "moves forward", "moves closer", "leaves the frame", "sits down",
    "stands up",
)  # fmt: skip

# A model with no negative prompt gets a positive-side screen instead (CLAUDE.md, "年齡安全"): these
# terms are refused outright, with no override. The age list is a superset of
# generate_character.AGE_SAFETY_NEGATIVE; tests pin both that and this tuple's content. "minor" also
# blocks "minor adjustments" - write "small" instead.
BLOCKED_TERMS = (
    "child", "children", "kid", "kids", "minor", "minors", "teen", "teens", "teenage", "teenager",
    "teenagers", "underage", "preteen", "young girl", "young girls", "young boy", "young boys",
    "little girl", "little boy", "schoolgirl", "schoolboy", "infant", "toddler", "juvenile", "loli",
    "lolita", "shota", "nsfw", "nude", "nudity", "naked", "topless", "bottomless", "explicit", "sex",
    "sexual", "sexually", "porn", "porno", "pornographic", "erotic", "genitals", "undress", "undressing",
)  # fmt: skip
BLOCKED_TERMS_CJK = (
    "兒童", "儿童", "小孩", "孩童", "幼童", "幼兒", "幼儿", "嬰兒", "婴儿", "未成年", "少女", "國中生", "国中生",
    "初中生", "高中生", "小學生", "小学生", "蘿莉", "萝莉", "正太", "裸體", "裸体", "全裸", "色情", "性愛", "性爱",
    "做愛", "做爱",
)  # fmt: skip
_UNDER_18 = re.compile(
    r"(?<![0-9.])(?:1[0-7]|[1-9])\s*-?\s*(?:years?|yrs?)\s*-?\s*old(?![a-z])"
    r"|(?<![0-9.])(?:1[0-7]|[1-9])\s*(?:yo|y/o)(?![a-z])"
    r"|(?<![a-z])(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
    r"|fifteen|sixteen|seventeen)[\s-]*(?:years?|yrs?)[\s-]*old(?![a-z])"
    r"|(?<![0-9十百一二三四五六七八九])(?:1[0-7]|[1-9]|十[一二三四五六七]?|[一二三四五六七八九])\s*[歲岁]",
    re.IGNORECASE,
)


def _terms_regex(terms: tuple[str, ...]) -> re.Pattern:
    parts = sorted((re.escape(t).replace(r"\ ", r"[\s-]+") for t in terms), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])", re.IGNORECASE)


_BLOCKED_RE = _terms_regex(BLOCKED_TERMS)
_CAMERA_MOVES_RE = _terms_regex(_CAMERA_MOVES)
_CAMERA_VERBS_RE = _terms_regex(_CAMERA_VERBS)
_LOCOMOTION_RE = _terms_regex(_LOCOMOTION)


def screen_prompt(text: str) -> None:
    hits = {m.group(0).lower() for m in _BLOCKED_RE.finditer(text)}
    hits |= {t for t in BLOCKED_TERMS_CJK if t in text}
    hits |= {m.group(0) for m in _UNDER_18.finditer(text)}
    if hits:
        raise InputError(
            f"The prompt contains terms this project never sends to H3 (it has no negative prompt): {sorted(hits)}",
            code="PROMPT_REJECTED",
        )


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def constraint_conflicts(description: str, constraints: tuple[str, ...]) -> list[str]:
    conflicts = []
    if "locked-camera" in constraints:
        for sentence in _sentences(description):
            words = [m.group(0) for m in _CAMERA_MOVES_RE.finditer(sentence)]
            if re.search(r"\b(camera|shot|lens|frame|view)\b", sentence, re.IGNORECASE):
                words += [m.group(0) for m in _CAMERA_VERBS_RE.finditer(sentence)]
            conflicts += [f"locked-camera vs {w!r}" for w in words]
    if "static-subject" in constraints:
        conflicts += [f"static-subject vs {m.group(0)!r}" for m in _LOCOMOTION_RE.finditer(description)]
    return conflicts


def apply_constraints(description: str, constraints: tuple[str, ...]) -> str:
    """User intent over prompt optimization: a locked camera or a still subject must not be contradicted
    by the description, and is stated positively (the official guide says not to write negations)."""
    unknown = [c for c in constraints if c not in CONSTRAINTS]
    if unknown:
        raise InputError(f"unknown constraint {unknown}; use {', '.join(CONSTRAINTS)}")
    conflicts = constraint_conflicts(description, constraints)
    if conflicts:
        raise InputError(
            f"The shot description contradicts the requested constraints: {conflicts}",
            code="PROMPT_REJECTED",
            hint="Rewrite the description without those words; describe what does happen, not what doesn't.",
        )
    out = description.rstrip()
    if "locked-camera" in constraints and not re.search(r"\b(static|locked[- ]off)\b", out, re.IGNORECASE):
        out += " " + LOCKED_CAMERA_SENTENCE
    if "static-subject" in constraints and not re.search(r"\bstays? in place\b", out, re.IGNORECASE):
        out += " " + STATIC_SUBJECT_SENTENCE
    return out


def build_i2va_prompt(description: str, soundscape: str, music: str) -> str:
    return (
        f"{I2VA_ALIGNMENT}\n\n"
        f"integrated_multimodal_description: [Shot 1] {description}\n\n"
        f"overall_soundscape: {soundscape}\n\n"
        f"non_diegetic_music: {music}"
    )


def build_prompt(spec: JobSpec) -> str:
    constraints = tuple(spec.constraints)
    if spec.prompt_file:
        if spec.description.strip():
            raise InputError("give either --prompt or --prompt-file, not both")
        path = Path(spec.prompt_file).expanduser().resolve()
        if not path.is_file():
            raise InputError(f"prompt file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise InputError(f"prompt file must be UTF-8: {path}") from exc
        if not text:
            raise InputError(f"prompt file is empty: {path}")
        pictures = {int(n) for n in re.findall(r"<Picture\s+(\d+)>", text)}
        if pictures != {1}:
            raise InputError("a full prompt must reference <Picture 1> (the first frame) and no other picture")
        conflicts = constraint_conflicts(text, constraints)
        if conflicts:
            raise InputError(f"The prompt file contradicts the constraints: {conflicts}", code="PROMPT_REJECTED")
    else:
        description = spec.description.strip()
        if not description:
            raise InputError("a prompt is required: describe what happens in the shot (--prompt)")
        if re.search(r"<Picture\s+\d+>|\[Shot\s+\d+\]", description):
            raise InputError("--prompt is the shot description only; pass a complete H3 prompt with --prompt-file")
        text = build_i2va_prompt(
            apply_constraints(description, constraints),
            spec.soundscape.strip() or DEFAULT_SOUNDSCAPE,
            spec.music.strip() or DEFAULT_MUSIC,
        )
    if len(text) > MAX_PROMPT_CHARS:
        raise InputError(f"prompt is {len(text)} characters; the limit is {MAX_PROMPT_CHARS}")
    screen_prompt(text)
    return text


# --- job spec ----------------------------------------------------------------------------------------

MAX_BATCH_JOBS = 20


@dataclasses.dataclass
class JobSpec:
    """One clip. The nested dict form (to_dict/from_dict) is the unit a batch manifest (or a future
    storyboard runner) queues per scene: job_id, scene_id, input, prompt, settings, output.

    chain=True starts the clip from the previous job's last frame. That frame never leaves the Colab
    VM, so the job must run in the same batch (and session) as the one before it, and `image` stays
    empty."""

    image: Path | str | None = None
    description: str = ""
    soundscape: str = ""
    music: str = ""
    prompt_file: Path | str | None = None
    constraints: tuple[str, ...] = ()
    duration: float | None = None
    resolution: str = "auto"
    seed: int | None = None
    output: Path | str | None = None
    gpu: str | None = None
    high_mem: bool | None = None
    timeout: int | None = None
    overwrite: bool = False
    job_id: str | None = None
    scene_id: str | None = None
    mode: str = "first_frame"
    chain: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> JobSpec:
        prompt = data.get("prompt") or {}
        settings = data.get("settings") or {}
        if isinstance(prompt, str):
            prompt = {"description": prompt}
        chain = bool(data.get("chain", False))
        image = (data.get("input") or {}).get("image")
        if not image and not chain:
            raise InputError('job spec needs input.image (or "chain": true to continue the previous clip)')
        return cls(
            image=image or None,
            description=prompt.get("description", ""),
            soundscape=prompt.get("soundscape", ""),
            music=prompt.get("music", ""),
            prompt_file=prompt.get("prompt_file"),
            constraints=tuple(prompt.get("constraints") or ()),
            duration=settings.get("duration"),
            resolution=settings.get("resolution") or "auto",
            seed=settings.get("seed"),
            output=data.get("output"),
            gpu=settings.get("gpu"),
            high_mem=settings.get("high_mem"),
            timeout=settings.get("timeout"),
            overwrite=bool(settings.get("overwrite", False)),
            job_id=data.get("job_id"),
            scene_id=data.get("scene_id"),
            mode=settings.get("mode") or "first_frame",
            chain=chain,
        )

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "scene_id": self.scene_id,
            "chain": self.chain,
            "input": {"image": str(self.image) if self.image else None},
            "prompt": {
                "description": self.description,
                "soundscape": self.soundscape,
                "music": self.music,
                "prompt_file": str(self.prompt_file) if self.prompt_file else None,
                "constraints": list(self.constraints),
            },
            "settings": {
                "mode": self.mode,
                "duration": self.duration,
                "resolution": self.resolution,
                "seed": self.seed,
                "gpu": self.gpu,
                "high_mem": self.high_mem,
                "timeout": self.timeout,
                "overwrite": self.overwrite,
            },
            "output": str(self.output) if self.output else None,
        }


@dataclasses.dataclass
class Prepared:
    job_id: str
    session: str
    image: Path | None
    image_size: tuple[int, int]
    width: int
    height: int
    duration: float
    frames: int
    seed: int
    prompt: str
    output: Path
    record_path: Path
    work_dir: Path
    gpu: str
    high_mem: bool
    timeout: int
    warnings: list[str]
    chained_from: str | None = None
    remote_image_override: str | None = None

    @property
    def actual_seconds(self) -> float:
        return round(self.frames / FPS, 3)

    @property
    def remote_prefix(self) -> str:
        # Flat under /content: `colab upload` is not known to create missing remote directories.
        return f"/content/h3_{self.job_id}"

    def remote_job(self) -> dict:
        suffix = self.image.suffix.lower() if self.image is not None else ".png"
        return {
            "job_id": self.job_id,
            "remote_image": self.remote_image_override or f"{self.remote_prefix}_first_frame{suffix}",
            "remote_output": f"{self.remote_prefix}_output.mp4",
            "remote_last_frame": f"{self.remote_prefix}_last_frame.png",
            "comfy_image_name": f"h3_{self.job_id}_first_frame.png",
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "seed": self.seed,
            "steps": STEPS,
            "sampler": SAMPLER,
            "scheduler": SCHEDULER,
            "lora_strength": LORA_STRENGTH,
            "crf": VIDEO_CRF,
            # The remote script gives up this much earlier than the local `colab exec` timeout, so it
            # can report a clean TIMEOUT marker instead of being cut off mid-line.
            "deadline_seconds": max(60, self.timeout - 120),
        }


def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")[:48] or "clip"


def check_image_file(path: Path | str) -> Path:
    """The checks that need no ffprobe: exists, not empty, not huge, an image extension."""
    image = Path(path).expanduser().resolve()
    if not image.is_file():
        raise InputError(f"input image not found: {image}")
    size = image.stat().st_size
    if size == 0:
        raise InputError(f"input image is empty: {image}")
    if size > MAX_IMAGE_BYTES:
        raise InputError(f"input image is {size / 2**20:.1f} MiB; the limit is {MAX_IMAGE_BYTES // 2**20} MiB")
    if image.suffix.lower() not in IMAGE_SUFFIXES:
        raise InputError(f"input image must be one of {', '.join(IMAGE_SUFFIXES)}: {image.name}")
    return image


def prepare(spec: JobSpec, config: dict, prober: Any, *, job_id: str, chain_source: Prepared | None = None) -> Prepared:
    """Validate everything that can be validated without Colab. Raises InputError.

    A chained job takes its frame size from chain_source (the previous job) instead of reading an image."""
    warnings = []
    if spec.mode != "first_frame":
        raise InputError(f"mode {spec.mode!r} is not implemented; this phase does first_frame (I2VA) only")
    image: Path | None = None
    if spec.chain:
        if chain_source is None:
            raise InputError("a chained job needs the previous job of the same batch as its first frame")
        img_w, img_h = chain_source.width, chain_source.height
    else:
        image = check_image_file(spec.image)
        try:
            img_w, img_h, codec = prober.image_info(image)
        except ProbeError as exc:
            raise InputError(f"not a readable image: {image} ({exc})") from exc
        if codec not in IMAGE_CODECS:
            raise InputError(f"{image.name} decodes as {codec!r}, not a still image")

    duration = config["video"]["default_duration"] if spec.duration is None else spec.duration
    try:
        duration = float(duration)
    except (TypeError, ValueError) as exc:
        raise InputError(f"duration {spec.duration!r} is not a number") from exc
    if not math.isfinite(duration) or not MIN_DURATION <= duration <= MAX_DURATION:
        raise InputError(f"duration must be {MIN_DURATION:g}-{MAX_DURATION:g} seconds (got {spec.duration!r})")
    if duration < STABLE_MIN_DURATION:
        warnings.append(f"{duration:g} s is below H3's ~5 s trained range; motion and audio may be unstable")
    frames = frames_for(duration)

    if spec.chain:
        requested = (spec.resolution or "auto").strip().lower()
        if requested != "auto" and parse_resolution(requested) != (img_w, img_h):
            raise InputError(f"a chained job keeps the previous clip's {img_w}x{img_h}; leave --resolution as auto")
        width, height = img_w, img_h
    else:
        width, height = resolve_resolution(spec.resolution, img_w, img_h, config["video"])
    prompt = build_prompt(spec)

    if spec.seed is None:
        seed = random.SystemRandom().randrange(0, 2**48)
    else:
        try:
            seed = int(spec.seed)
        except (TypeError, ValueError) as exc:
            raise InputError(f"seed {spec.seed!r} is not an integer") from exc
        if not 0 <= seed < 2**64:
            raise InputError("seed must be 0 to 2^64-1")

    gpu = (spec.gpu or config["colab"]["gpu"]).upper()
    if gpu not in GPUS:
        raise InputError(f"gpu must be one of {', '.join(GPUS)}")
    high_mem = config["colab"]["high_mem"] if spec.high_mem is None else bool(spec.high_mem)
    timeout = int(spec.timeout or config["colab"]["timeout_seconds"])
    if not 300 <= timeout <= 6 * 3600:
        raise InputError("timeout must be 300-21600 seconds")

    out_dir = output_dir(config)
    if spec.output:
        output = Path(spec.output).expanduser().resolve()
    else:
        stem = _safe_stem(image.stem) if image is not None else "chain"
        output = out_dir / f"{stem}_h3_{job_id}.mp4"
    if output.suffix.lower() != ".mp4":
        raise InputError(f"output must be an .mp4 path: {output}")
    if output.exists() and not spec.overwrite:
        raise InputError(f"output already exists: {output} (pass --overwrite to replace it)")

    return Prepared(
        job_id=job_id,
        session=f"h3-{job_id}",
        image=image,
        image_size=(img_w, img_h),
        width=width,
        height=height,
        duration=duration,
        frames=frames,
        seed=seed,
        prompt=prompt,
        output=output,
        record_path=output.with_name(output.stem + ".job.json"),
        work_dir=out_dir / ".work" / job_id,
        gpu=gpu,
        high_mem=high_mem,
        timeout=timeout,
        warnings=warnings,
        chained_from=chain_source.job_id if spec.chain and chain_source is not None else None,
        remote_image_override=chain_source.remote_job()["remote_last_frame"] if spec.chain and chain_source else None,
    )


def stage_work_dir(p: Prepared) -> dict[str, Path | None]:
    """Everything a Colab call reads or writes sits in one directory: Docker mounts only that."""
    p.work_dir.mkdir(parents=True, exist_ok=True)
    image = None
    if p.image is not None:
        image = p.work_dir / f"first_frame{p.image.suffix.lower()}"
        shutil.copy2(p.image, image)
    job = p.work_dir / "job.json"
    job.write_text(json.dumps(p.remote_job(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    script = p.work_dir / REMOTE_SCRIPT.name
    shutil.copy2(REMOTE_SCRIPT, script)
    return {"image": image, "job": job, "script": script, "output": p.work_dir / "output.mp4"}


# --- the job state machine ---------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class JobRun:
    """State, timings, log file and the record written at the end."""

    def __init__(self, job_id: str, scene_id: str | None, log_path: Path, stream: Any):
        self.state = PENDING
        self.log_path = log_path
        self.stream = stream
        self._started = time.monotonic()
        self._entered = self._started
        self._held = False
        self.record: dict[str, Any] = {
            "job_id": job_id,
            "scene_id": scene_id,
            "status": PENDING,
            "started_at": now_iso(),
            "finished_at": None,
            "elapsed_seconds": None,
            "stage_seconds": {},
            "error_code": None,
            "error": None,
            "hint": None,
            "failed_stage": None,
            "warnings": [],
            "log": str(log_path),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{now_iso()} [{self.state}] {message}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.stream is not None:
            print(line, file=self.stream, flush=True)

    def log_remote(self, line: str) -> None:
        if line:
            self.log("| " + (line if len(line) <= 600 else line[:600] + " ..."))

    def _close_stage(self) -> None:
        now = time.monotonic()
        stages = self.record["stage_seconds"]
        key = QUEUED if self._held else self.state
        stages[key] = round(stages.get(key, 0.0) + now - self._entered, 1)
        self._entered, self._held = now, False

    def hold(self) -> None:
        """Wait for earlier jobs of the batch: the wait goes to QUEUED, not to the current stage or elapsed."""
        self._close_stage()
        self._held = True

    def advance(self, state: str) -> None:
        self._close_stage()
        previous, self.state = self.state, state
        self.record["status"] = state
        self.log(f"{previous} -> {state}")

    def fail(self, status: str, code: str, message: str, hint: str = "") -> None:
        self.record.update(
            {"status": status, "error_code": code, "error": message, "hint": hint or None, "failed_stage": self.state}
        )
        self.log(f"{status} ({code}) at {self.state}: {message}" + (f"\nhint: {hint}" if hint else ""))

    def finish(self) -> None:
        if self.record["finished_at"] is not None:
            return
        self._close_stage()
        queued = self.record["stage_seconds"].get(QUEUED, 0.0)
        self.record["finished_at"] = now_iso()
        self.record["elapsed_seconds"] = round(time.monotonic() - self._started - queued, 1)
        self.log(f"finished: {self.record['status']} in {self.record['elapsed_seconds']} s")


class RemoteProgress:
    """Turns the remote script's H3_* markers into state changes and record fields."""

    def __init__(self, run: JobRun):
        self.run = run
        self.error: tuple[str, str] | None = None
        self.output: dict | None = None
        self.last_frame: str | None = None

    def on_line(self, line: str) -> None:
        self.run.log_remote(line)
        parsed = parse_marker(line)
        if not parsed:
            return
        kind, data = parsed
        rec = self.run.record
        if kind == "STAGE" and data.get("name") == INFERENCE and self.run.state == LOADING_MODEL:
            self.run.advance(INFERENCE)
        elif kind == "GPU":
            rec["gpu"] = data.get("name")
            rec["gpu_info"] = data
        elif kind == "COMFYUI":
            rec["comfyui_commit"] = data.get("commit")
        elif kind == "CONFIG":
            rec["remote_config"] = data
        elif kind == "TIMING":
            rec.setdefault("remote_seconds", {})[data.get("stage")] = data.get("seconds")
        elif kind == "MODEL":
            rec.setdefault("remote_models", []).append(data)
            rec.setdefault("model_actions", {})[data.get("name")] = data.get("action")
        elif kind == "DRIVE_OUTPUT":
            rec["drive_output"] = data.get("path")
        elif kind == "DRIVE_OUTPUT_FAILED":
            warning = f"the clip was not copied to Google Drive ({data.get('path')}): {data.get('message')}"
            rec["warnings"].append(warning)
        elif kind == "VRAM_PEAK":
            rec["vram_peak_mib"] = data.get("mib")
        elif kind == "ERROR":
            self.error = (str(data.get("code") or "INFERENCE_FAILED"), str(data.get("message") or ""))
        elif kind == "OUTPUT":
            self.output = data
        elif kind == "LAST_FRAME":
            self.last_frame = data.get("path")

    def raise_if_error(self) -> None:
        if not self.error:
            return
        code, message = self.error
        if code == "GPU_UNAVAILABLE":
            raise H3Error(code, GPU_UNAVAILABLE_MESSAGE, hint=message)
        if code == "TIMEOUT":
            raise H3Error(code, f"Remote inference timed out: {message}", status=TIMEOUT)
        raise H3Error(code, message or code)


class ColabSession:
    """One `colab new` ... `colab stop`, kept for as many jobs of a batch as it can serve: the remote
    script reuses the ComfyUI install, the downloaded models and the running ComfyUI it finds there."""

    def __init__(self, transport: Transport, config: dict, name: str, *, gpu: str, high_mem: bool, ignore_busy: bool):
        self.transport = transport
        self.config = config
        self.name = name
        self.gpu = gpu
        self.high_mem = high_mem
        self.ignore_busy = ignore_busy
        self.attempted = False  # `colab new` was called, so `colab stop` must be too
        self.open_ok = False
        self.closed = False
        self.t0: float | None = None
        self.seconds: float | None = None
        self.balance_before: float | None = None
        self.rate_per_hour: float | None = None
        self.cli_version: str | None = None
        self.status: str | None = None
        self.warning: str | None = None
        self.drive_warning: str | None = None
        self.persistence = PERSIST_EPHEMERAL
        self.drive: dict | None = None  # how the mount went
        self.saved: dict | None = None  # what the finalize step reported
        self.kernel_busy = False  # an exec did not return cleanly: no further exec may be sent

    @property
    def alive(self) -> bool:
        return self.open_ok and not self.closed

    def storage(self, p: "Prepared") -> dict:
        """What the remote script needs to follow the storage rule (see remote/h3_colab_job.py)."""
        d = self.config["drive"]
        return {
            "persistence": self.persistence,
            "workspace": f"{d['mount']}/{d['workspace']}",
            "models": self.config["models"],
            "drive_output": f"outputs/videos/{p.output.name}",
        }

    def _mount_drive(self, run: JobRun) -> None:
        d = self.config["drive"]
        if d["mode"] != "consent":
            run.log("drive.mode is off: models are downloaded for this session only and nothing is stored on Drive")
            return
        signal_path = consent_signal_path(self.config)
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.unlink(missing_ok=True)

        def on_url(url: str) -> None:
            run.log(
                "DRIVE_CONSENT_NEEDED: approve Google Drive access for this VM in the browser, then run "
                f"scripts/consent_done.py (Enter is sent by itself after {d['consent_wait_seconds']} s)"
            )
            if d["open_browser"]:
                _open_in_browser(url)

        try:
            self.drive = self.transport.mount_drive(
                self.name,
                d["mount"],
                wait_seconds=d["consent_wait_seconds"],
                signal_path=signal_path,
                on_url=on_url,
                on_line=run.log_remote,
            )
        except (OSError, subprocess.SubprocessError, NotImplementedError) as exc:
            self.drive = {"mounted": False, "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            signal_path.unlink(missing_ok=True)
        if self.drive.get("mounted"):
            self.persistence = PERSIST_DRIVE
            run.log(f"Google Drive mounted in {self.drive.get('seconds')} s: models and clips are stored there")
        else:
            run.log(
                f"WARNING: Google Drive was not mounted ({self.drive.get('reason')}). This session is ephemeral: "
                "the models are downloaded again and nothing is stored on Drive."
            )

    def _finalize(self, run: JobRun) -> None:
        """Before the VM goes away: wait for the model copies and flush the mount. Never raises - a session
        that cannot be saved is still stopped, it must not keep an A100 running."""
        d = self.config["drive"]
        saved: dict = {"flushed": False, "model_sync": None, "error": None}
        self.saved = saved

        def on_line(line: str) -> None:
            run.log_remote(line)
            parsed = parse_marker(line)
            if not parsed:
                return
            kind, data = parsed
            if kind == "MODEL_SYNC":
                saved["model_sync"] = data.get("report")
            elif kind == "DRIVE_FLUSHED":
                saved["flushed"] = True
                saved["flush_seconds"] = data.get("seconds")
            elif kind == "ERROR":
                saved["error"] = f"{data.get('code')}: {data.get('message')}"

        t0 = time.monotonic()
        run.log("saving to Google Drive: waiting for the model copies, then flushing the mount")
        try:
            self.transport.call(
                ["exec", "--session", self.name, "--timeout", str(d["finalize_timeout_seconds"])]
                + ["--env", f"{FINALIZE_ENV}=1"]
                + ["--file", self.transport.cli_path(REMOTE_SCRIPT, REMOTE_SCRIPT.parent)],
                label="colab exec (save to Drive)",
                timeout=d["finalize_timeout_seconds"] + 120,
                mount_dir=REMOTE_SCRIPT.parent,
                on_line=on_line,
                idle_timeout=self.config["colab"]["exec_idle_timeout_seconds"],
            )
        except (ColabCommandError, ColabTimeout) as exc:
            saved["error"] = saved["error"] or str(exc)[-500:]
        except KeyboardInterrupt:
            saved["error"] = "cancelled by the user"
        saved["seconds"] = round(time.monotonic() - t0, 1)
        failed = sorted(k for k, v in (saved["model_sync"] or {}).items() if str(v).startswith("failed"))
        if saved["error"] or not saved["flushed"] or failed:
            self.drive_warning = (
                "Google Drive may not hold everything from this session "
                f"(flushed: {saved['flushed']}, failed copies: {failed or 'none'}, error: {saved['error']}). "
                "Files that did not arrive are downloaded again next time."
            )
            run.log("WARNING: " + self.drive_warning)

    def open(self, run: JobRun) -> None:
        t = self.transport
        t.preflight()
        try:
            self.cli_version = t.call(["version"], label="colab version", timeout=90).splitlines()[-1]
        except (ColabCommandError, ColabTimeout, IndexError):
            self.cli_version = None
        before = colab_usage(t)
        self.balance_before = before.get("balance")
        rate_before = before.get("rate_per_hour")
        active = int(before.get("active_assignments") or 0)
        run.log(f"compute units before: {before.get('balance')} (rate {rate_before}/hr, active runtimes {active})")
        if active and not self.ignore_busy:
            raise H3Error(
                "COLAB_BUSY",
                f"{active} Colab runtime(s) already running on this account, e.g. another Claude session's job. "
                "Starting another would compete for the A100 and mix the compute-unit readings.",
                hint="Wait until `colab sessions` shows none, or pass --ignore-busy.",
            )
        self.attempted = True
        self.t0 = time.monotonic()
        colab_new(
            t,
            self.name,
            gpu=self.gpu,
            high_mem=self.high_mem,
            timeout=self.config["colab"]["session_create_timeout_seconds"],
            on_line=run.log_remote,
        )
        self.open_ok = True
        try:
            during = colab_usage(t)
            if during.get("rate_per_hour") is not None:
                self.rate_per_hour = round(during["rate_per_hour"] - (rate_before or 0.0), 3)
                run.log(f"session compute-unit rate: {self.rate_per_hour}/hr")
        except H3Error as exc:
            run.log(f"could not read the usage rate after `colab new`: {exc}")
        self._mount_drive(run)

    def close(self, run: JobRun) -> None:
        if self.closed or not self.attempted:
            return
        self.closed = True
        if self.open_ok and self.persistence == PERSIST_DRIVE:
            if self.kernel_busy:
                self.drive_warning = (
                    "The session ended with its kernel busy, so the save-to-Drive step was skipped: model files "
                    "that were still being copied are not on Google Drive and are downloaded again next time."
                )
                run.log("WARNING: " + self.drive_warning)
            else:
                self._finalize(run)
        try:
            stop_session(self.transport, self.name, on_line=run.log_remote)
            self.status = "stopped"
        except (ColabCommandError, ColabTimeout, KeyboardInterrupt) as exc:
            text = exc.text if isinstance(exc, ColabCommandError) else str(exc)
            if "not found" in text.lower():
                self.status = "not_created"
            else:
                self.status = "stop_failed"
                manual = self.transport.display(["stop", "--session", self.name])
                self.warning = f"Colab session {self.name} may still be running and spending compute units: {manual}"
                run.log(f"WARNING: could not stop session {self.name}; stop it by hand: {manual}")
        if self.t0 is not None:
            self.seconds = round(time.monotonic() - self.t0, 1)


LEDGER_FIELDS = (
    "job_id", "scene_id", "batch_id", "started_at", "finished_at", "status", "error_code", "failed_stage",
    "duration", "frames", "actual_seconds", "resolution", "seed", "gpu", "vram_peak_mib", "session",
    "session_reused", "output", "bytes", "elapsed_seconds", "session_seconds", "stage_seconds", "remote_seconds",
    "cu_balance_before", "cu_balance_after", "cu_used_measured", "cu_rate_per_hour", "cu_estimated",
    "comfyui_commit", "cli_version", "transport", "persistence", "drive_output", "model_actions",
)  # fmt: skip


def _json_default(value: Any) -> Any:
    return str(value)


def _default_prober(config: dict) -> FFprobe:
    path = find_ffprobe(config)
    if not path:
        raise H3Error(
            "FFPROBE_MISSING",
            "ffprobe was not found; every clip is validated with it, so the job does not start without it.",
            hint="Set tools.ffprobe in config/config.json, or install it: winget install --id Gyan.FFmpeg",
        )
    return FFprobe(path)


def _record_prepared(run: JobRun, p: Prepared, spec: JobSpec) -> None:
    run.record.update(
        {
            "image": str(p.image) if p.image is not None else None,
            "chained_from": p.chained_from,
            "image_size": list(p.image_size),
            "duration": p.duration,
            "frames": p.frames,
            "actual_seconds": p.actual_seconds,
            "resolution": f"{p.width}x{p.height}",
            "seed": p.seed,
            "gpu_requested": p.gpu,
            "high_mem": p.high_mem,
            "timeout_seconds": p.timeout,
            "output": str(p.output),
            "prompt": p.prompt,
            "spec": spec.to_dict(),
        }
    )
    for warning in p.warnings:
        run.record["warnings"].append(warning)
        run.log("warning: " + warning)


def _read_balance(transport: Transport, run: JobRun) -> float | None:
    try:
        return colab_usage(transport).get("balance")
    except (H3Error, KeyboardInterrupt) as exc:
        run.log(f"could not read the compute-unit balance: {exc}")
        return None


def _account(rec: dict, before: float | None, after: float | None, t0: float | None, rate: float | None) -> None:
    """A job's share: the balance difference over its own window of the session (the first job of a
    session also carries the install and model download)."""
    rec["cu_balance_after"] = after
    if before is not None and after is not None:
        rec["cu_used_measured"] = round(before - after, 3)
    if t0 is not None:
        rec["session_seconds"] = round(time.monotonic() - t0, 1)
        if rate:
            # An estimate from the hourly rate; the balance difference is the real figure.
            rec["cu_estimated"] = round(rate * rec["session_seconds"] / 3600, 3)


def _run_in_session(
    run: JobRun, p: Prepared, f: dict, transport: Transport, session: ColabSession, config: dict, prober: Any
) -> None:
    rec = run.record
    transfer = config["colab"]["transfer_timeout_seconds"]
    remote = p.remote_job()

    run.advance(UPLOADING)
    if f["image"] is not None:
        upload_file(
            transport, session.name, f["image"], remote["remote_image"], timeout=transfer, on_line=run.log_remote
        )
    else:
        run.log(f"first frame: the last frame of {p.chained_from}, already on the VM at {remote['remote_image']}")
    remote_job_path = f"{p.remote_prefix}_job.json"
    # Known only now: whether this session has Drive. The staged job file gets the storage block added.
    f["job"].write_text(
        json.dumps(dict(remote, storage=session.storage(p)), ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    rec["persistence"] = session.persistence
    upload_file(transport, session.name, f["job"], remote_job_path, timeout=transfer, on_line=run.log_remote)

    run.advance(LOADING_MODEL)
    progress = RemoteProgress(run)
    session.kernel_busy = True  # until the exec returns: a timeout or Ctrl-C leaves the remote cell running
    try:
        transport.call(
            ["exec", "--session", session.name, "--timeout", str(p.timeout)]
            + ["--env", f"H3_JOB_FILE={remote_job_path}"]
            + ["--file", transport.cli_path(f["script"], p.work_dir)],
            label="colab exec",
            timeout=p.timeout + 120,
            mount_dir=p.work_dir,
            on_line=progress.on_line,
            idle_timeout=config["colab"]["exec_idle_timeout_seconds"],
        )
    except ColabCommandError as exc:
        progress.raise_if_error()
        raise SessionLost("INFERENCE_FAILED", f"`colab exec` failed: {exc}") from exc
    session.kernel_busy = False
    progress.raise_if_error()
    if not progress.output:
        raise H3Error("OUTPUT_NOT_FOUND", "Inference ended but the remote script reported no MP4 (no H3_OUTPUT).")
    rec["remote_last_frame"] = progress.last_frame

    run.advance(DOWNLOADING)
    local = f["output"]
    try:
        download_file(
            transport,
            session.name,
            progress.output.get("path") or remote["remote_output"],
            local,
            timeout=transfer,
            on_line=run.log_remote,
        )
    except ColabCommandError as exc:
        raise H3Error("OUTPUT_NOT_FOUND", f"Could not download the MP4: {exc}") from exc
    if not local.is_file() or local.stat().st_size == 0:
        raise H3Error("OUTPUT_NOT_FOUND", f"The downloaded MP4 is missing or empty: {local}")
    p.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(local, p.output)
    rec["bytes"] = p.output.stat().st_size
    if progress.last_frame:
        # Kept next to the clip for hand-made follow-ups; a failed download only costs that convenience.
        target = p.output.with_name(p.output.stem + ".last_frame.png")
        try:
            download_file(
                transport, session.name, progress.last_frame, target, timeout=transfer, on_line=run.log_remote
            )
            rec["last_frame"] = str(target)
        except (ColabCommandError, ColabTimeout) as exc:
            rec["warnings"].append(f"could not download the last frame: {exc}")
            run.log(f"could not download the last frame: {exc}")

    run.advance(VALIDATING)
    try:
        probe = prober.video(p.output)
    except ProbeError as exc:
        raise H3Error("FAILED_VALIDATION", f"ffprobe cannot read the MP4: {exc}", status=FAILED_VALIDATION) from exc
    rec["ffprobe"] = summarize_probe(probe)
    problems = check_probe(probe, width=p.width, height=p.height, frames=p.frames)
    if problems:
        raise H3Error("FAILED_VALIDATION", "; ".join(problems), status=FAILED_VALIDATION)


def _dry_run_commands(transport: Transport, session_name: str, p: Prepared, config: dict) -> list[str]:
    drive = config["drive"]["mode"] == "consent"
    commands = [
        transport.display(["usage"]),
        transport.display(["new", "--session", session_name, "--gpu", p.gpu] + (["--high-mem"] if p.high_mem else [])),
    ]
    if drive:
        commands.append(transport.display(["drivemount", "--session", session_name, config["drive"]["mount"]]))
    commands.append(
        transport.display(
            ["exec", "--session", session_name, "--timeout", str(p.timeout), "--env"]
            + [f"H3_JOB_FILE={p.remote_prefix}_job.json", "--file", "/work/" + REMOTE_SCRIPT.name],
            p.work_dir,
        )
    )
    if drive:
        commands.append(
            transport.display(
                [
                    "exec",
                    "--session",
                    session_name,
                    "--env",
                    f"{FINALIZE_ENV}=1",
                    "--file",
                    "/work/" + REMOTE_SCRIPT.name,
                ],
                REMOTE_SCRIPT.parent,
            )
        )
    return commands


def run_batch(
    specs: list[JobSpec],
    config: dict,
    *,
    transport: Transport | None = None,
    prober: Any = None,
    dry_run: bool = False,
    settle_seconds: int | None = None,
    ignore_busy: bool = False,
    batch_id: str | None = None,
    stream: Any = "stderr",
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Run several clips through ONE Colab session and return the batch summary (also written to disk).

    Every job is validated before any Colab call. A job that fails is recorded and the batch moves on
    (the user's rule); a job that leaves the session in an unknown state (exec timeout, exec failure
    with no marker) is not retried, and the next job gets a fresh session. A session that cannot be
    opened at all ends every job left with the same error."""
    stream = sys.stderr if stream == "stderr" else stream
    specs = list(specs)
    if not specs:
        raise InputError("a batch needs at least one job")
    if len(specs) > MAX_BATCH_JOBS:
        raise InputError(f"a batch may hold at most {MAX_BATCH_JOBS} jobs (got {len(specs)})")
    if batch_id is None:
        batch_id = specs[0].job_id if len(specs) == 1 and specs[0].job_id else new_job_id()
    if not JOB_ID_RE.fullmatch(batch_id):
        raise InputError(f"batch_id {batch_id!r} may only use letters, digits, _ and - (max 40)")
    job_ids: list[str] = []
    for i, spec in enumerate(specs):
        jid = spec.job_id or (batch_id if len(specs) == 1 else f"{batch_id}-{i + 1}")
        if not JOB_ID_RE.fullmatch(jid):
            raise InputError(f"job_id {jid!r} may only use letters, digits, _ and - (max 40)")
        if jid in job_ids:
            raise InputError(f"job_id {jid!r} appears twice in the batch")
        job_ids.append(jid)
    out_dir = output_dir(config)
    settle = config["colab"]["cu_settle_seconds"] if settle_seconds is None else int(settle_seconds)
    started, started_at = time.monotonic(), now_iso()
    runs = [JobRun(jid, spec.scene_id, out_dir / "logs" / f"{jid}.log", stream) for jid, spec in zip(job_ids, specs)]
    for run in runs:
        run.record["batch_id"] = batch_id
    prepared: list[Prepared | None] = [None] * len(specs)
    files: list[dict | None] = [None] * len(specs)

    # PREPARING: every job, before any Colab call, so a bad input never costs compute units.
    for i, (spec, run) in enumerate(zip(specs, runs)):
        run.advance(PREPARING)
        try:
            if spec.chain:
                if i == 0:
                    raise InputError("the first job of a batch cannot chain: there is no previous clip")
                if prepared[i - 1] is None:
                    raise H3Error(
                        "CHAIN_SOURCE_FAILED",
                        f"it continues {job_ids[i - 1]}, which did not pass its own checks",
                        status=CANCELLED,
                    )
            else:
                check_image_file(spec.image)
            if prober is None:
                prober = _default_prober(config)
            p = prepare(spec, config, prober, job_id=job_ids[i], chain_source=prepared[i - 1] if spec.chain else None)
            first = next((q for q in prepared if q is not None), None)
            if first is not None and (p.gpu, p.high_mem) != (first.gpu, first.high_mem):
                raise InputError(
                    "all jobs of a batch share one Colab runtime: gpu and high_mem must match the first job"
                )
            prepared[i] = p
            files[i] = stage_work_dir(p)
            _record_prepared(run, p, spec)
        except H3Error as exc:
            run.fail(exc.status, exc.code, str(exc), exc.hint)
    pending = [i for i in range(len(specs)) if prepared[i] is not None]
    session_base = f"h3-{batch_id}"

    if dry_run:
        transport = transport or make_transport(config)
        checks = preflight_checks(config, transport=transport, prober=prober)
        for i in pending:
            p, rec = prepared[i], runs[i].record
            rec["transport"] = transport.kind
            rec["dry_run"] = {
                "prompt": p.prompt,
                "remote_job": p.remote_job(),
                "commands": _dry_run_commands(transport, session_base, p, config),
                "preflight": checks,
            }
            rec["status"] = DRY_RUN
            shutil.rmtree(p.work_dir, ignore_errors=True)
        return {"batch_id": batch_id, "status": DRY_RUN, "records": [r.record for r in runs]}

    for i, run in enumerate(runs):
        if prepared[i] is None:
            run.finish()  # failed its own checks: done now, not when the batch ends
        elif pending and i != pending[0]:
            run.hold()
    if pending:
        transport = transport or make_transport(config)
    sessions: list[ColabSession] = []
    session: ColabSession | None = None
    balance: float | None = None  # balance at the start of the current accounting window
    window_t0: float | None = None
    window: list[dict] = []
    fatal: H3Error | None = None
    cancelled = False
    try:
        for n, i in enumerate(pending):
            run, p, f = runs[i], prepared[i], files[i]
            rec = run.record
            if fatal is not None:
                run.fail(fatal.status, fatal.code, str(fatal), fatal.hint)
                continue
            if cancelled:
                run.fail(CANCELLED, "CANCELLED", "Not started: the batch was cancelled.")
                continue
            if p.chained_from is not None:
                src = runs[i - 1].record
                if src.get("status") != COMPLETED or not src.get("remote_last_frame"):
                    run.fail(CANCELLED, "CHAIN_SOURCE_FAILED", f"{p.chained_from} produced no last frame to continue")
                    continue
                if session is None or not session.alive or src.get("session") != session.name:
                    run.fail(CANCELLED, "CHAIN_SOURCE_FAILED", "the session holding the previous clip's frame is gone")
                    continue
            rec["transport"] = transport.kind
            try:
                run.advance(CONNECTING_COLAB)
                if session is None or not session.alive:
                    name = session_base if not sessions else f"{session_base}-{len(sessions) + 1}"
                    session = ColabSession(
                        transport, config, name, gpu=p.gpu, high_mem=p.high_mem, ignore_busy=ignore_busy
                    )
                    sessions.append(session)
                    rec["session"] = session.name
                    try:
                        session.open(run)
                    except (H3Error, ColabTimeout) as exc:
                        if not session.open_ok and isinstance(exc, H3Error):
                            fatal = exc
                        raise
                    balance, window_t0 = session.balance_before, session.t0
                    rec["session_reused"] = False
                else:
                    run.log(f"reusing Colab session {session.name}: ComfyUI and the models are already there")
                    rec["session_reused"] = True
                rec.update(
                    {
                        "session": session.name,
                        "cli_version": session.cli_version,
                        "cu_rate_per_hour": session.rate_per_hour,
                        "cu_balance_before": balance,
                    }
                )
                _run_in_session(run, p, f, transport, session, config, prober)
                run.advance(COMPLETED)
            except SessionLost as exc:
                run.fail(exc.status, exc.code, str(exc), exc.hint)
                session.close(run)
            except H3Error as exc:
                run.fail(exc.status, exc.code, str(exc), exc.hint)
            except ColabTimeout as exc:
                run.fail(
                    TIMEOUT,
                    "TIMEOUT",
                    f"{exc}. The remote kernel may still be busy, so this session is stopped instead of retrying.",
                )
                if session is not None:
                    session.close(run)
            except ColabCommandError as exc:
                if run.state == UPLOADING:
                    run.fail(FAILED, "UPLOAD_FAILED", str(exc))
                else:
                    run.fail(FAILED, "COLAB_COMMAND_FAILED", str(exc))
                    if session is not None:
                        session.close(run)
            except KeyboardInterrupt:
                run.fail(CANCELLED, "CANCELLED", "Cancelled by the user.")
                cancelled = True
            except Exception as exc:
                # Still stop the session and write the record; the traceback goes to the log.
                run.fail(FAILED, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
                run.log(traceback.format_exc())
            finally:
                if session is not None and session.attempted and rec.get("session") == session.name:
                    window.append(rec)
                    more = n < len(pending) - 1 and not cancelled and fatal is None
                    if session.closed or (session.alive and more):
                        after = _read_balance(transport, run)
                        for r in window:
                            _account(r, balance, after, window_t0, session.rate_per_hour)
                        window, balance, window_t0 = [], after, time.monotonic()
                if n < len(pending) - 1:
                    run.finish()  # the last job also counts the session stop, as a single run always has
    except KeyboardInterrupt:
        cancelled = True
    finally:
        last_run = runs[pending[-1]] if pending else runs[-1]
        for s in sessions:
            if s.attempted and not s.closed:
                s.close(last_run)
        after = settled = None
        if any(s.attempted for s in sessions):
            after = _read_balance(transport, last_run)
            for r in window:
                _account(r, balance, after, window_t0, sessions[-1].rate_per_hour)
        if last_run.record["status"] in TERMINAL:
            last_run.finish()
        if any(s.attempted for s in sessions):
            if settle > 0:
                last_run.log(f"waiting {settle} s, then reading the balance again (Colab can deduct late)")
                try:
                    sleep(settle)
                    settled = _read_balance(transport, last_run)
                except KeyboardInterrupt:
                    last_run.log("settle wait skipped")
        for run in runs:
            rec = run.record
            s = next((x for x in sessions if x.name == rec.get("session")), None)
            if s is not None:
                rec["session_status"] = s.status
                if s.warning:
                    rec["warnings"].append(s.warning)
                rec["persistence"] = s.persistence
                rec["drive_saved"] = s.saved
                if s.drive_warning:
                    rec["warnings"].append(s.drive_warning)
            if rec["status"] not in TERMINAL:
                run.fail(CANCELLED, "CANCELLED", "Not started: the batch stopped early.")
        summary = _batch_summary(batch_id, runs, sessions, started, started_at, after, settled, settle)
        if len(runs) == 1 and summary.get("cu_used_settled") is not None:
            runs[0].record["cu_balance_settled"] = summary["cu_balance_settled"]
            runs[0].record["cu_used_settled"] = summary["cu_used_settled"]
        for i, run in enumerate(runs):
            run.finish()
            _write_records(run, prepared[i], out_dir)
            if run.record["status"] == COMPLETED and prepared[i] is not None:
                shutil.rmtree(prepared[i].work_dir, ignore_errors=True)
            elif prepared[i] is not None:
                run.log(f"work files kept for debugging: {prepared[i].work_dir}")
        if any(s.attempted for s in sessions) or len(runs) > 1:
            _write_batch(summary, out_dir)
    summary["records"] = [r.record for r in runs]
    return summary


def _batch_summary(batch_id, runs, sessions, started, started_at, after, settled, settle) -> dict:
    statuses = [r.record["status"] for r in runs]
    completed = statuses.count(COMPLETED)
    before = next((s.balance_before for s in sessions if s.balance_before is not None), None)
    summary = {
        "batch_id": batch_id,
        "status": COMPLETED if completed == len(runs) else FAILED if completed == 0 else "PARTIAL",
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "completed": completed,
        "failed": len(runs) - completed,
        "jobs": [
            {
                "job_id": r.record["job_id"],
                "status": r.record["status"],
                "error_code": r.record["error_code"],
                "output": r.record.get("output"),
                "session": r.record.get("session"),
                "session_reused": r.record.get("session_reused"),
                "cu_used_measured": r.record.get("cu_used_measured"),
            }
            for r in runs
        ],
        "sessions": [
            {"name": s.name, "status": s.status, "seconds": s.seconds, "rate_per_hour": s.rate_per_hour}
            for s in sessions
            if s.attempted
        ],
        "session_seconds": round(sum(s.seconds or 0 for s in sessions), 1),
        "cu_balance_before": before,
        "cu_balance_after": after,
        "cu_used_measured": round(before - after, 3) if before is not None and after is not None else None,
        "cu_settle_seconds": settle,
        "cu_balance_settled": settled,
        "cu_used_settled": round(before - settled, 3) if before is not None and settled is not None else None,
    }
    return summary


def _write_batch(summary: dict, out_dir: Path) -> None:
    path = out_dir / "batches" / f"{summary['batch_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    with open(out_dir / "h3_batches.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, default=_json_default) + "\n")


def run_job(
    spec: JobSpec,
    config: dict,
    *,
    transport: Transport | None = None,
    prober: Any = None,
    dry_run: bool = False,
    settle_seconds: int | None = None,
    ignore_busy: bool = False,
    stream: Any = "stderr",
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Run one clip (a batch of one) and return its record (also written to disk)."""
    summary = run_batch(
        [spec],
        config,
        transport=transport,
        prober=prober,
        dry_run=dry_run,
        settle_seconds=settle_seconds,
        ignore_busy=ignore_busy,
        stream=stream,
        sleep=sleep,
    )
    return summary["records"][0]


def _write_records(run: JobRun, prepared: Prepared | None, out_dir: Path) -> None:
    rec = run.record
    if prepared is not None:
        prepared.record_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.record_path.write_text(
            json.dumps(rec, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        rec["record"] = str(prepared.record_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {k: rec.get(k) for k in LEDGER_FIELDS}
    with open(out_dir / "h3_jobs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")


# --- preflight -------------------------------------------------------------------------------------


def preflight_checks(config: dict, *, transport: Transport | None = None, prober: Any = None) -> list[dict]:
    """What check.py prints and a dry run includes. `colab usage` spends no compute units."""
    checks: list[dict] = []

    def add(name: str, fn: Callable[[], Any]) -> bool:
        try:
            checks.append({"name": name, "ok": True, "detail": fn()})
            return True
        except H3Error as exc:
            checks.append({"name": name, "ok": False, "code": exc.code, "detail": str(exc), "hint": exc.hint or None})
        except (ColabCommandError, ColabTimeout, ProbeError, OSError, subprocess.SubprocessError) as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})
        return False

    def ffprobe() -> str:
        if prober is not None and not isinstance(prober, FFprobe):
            return "injected prober"
        path = prober.path if isinstance(prober, FFprobe) else find_ffprobe(config)
        if not path:
            raise H3Error(
                "FFPROBE_MISSING",
                "ffprobe not found (tools.ffprobe / PATH)",
                hint="Set tools.ffprobe in config/config.json, or: winget install --id Gyan.FFmpeg",
            )
        return f"{path}: {FFprobe(path).version()}"

    transport = transport or make_transport(config)
    add("config", lambda: str(LOCAL_CONFIG if LOCAL_CONFIG.is_file() else EXAMPLE_CONFIG))
    add("ffprobe", ffprobe)
    add("output_dir", lambda: str(output_dir(config)))
    if add(f"transport:{transport.kind}", lambda: transport.preflight() or "ready"):
        if add("colab_cli", lambda: transport.call(["version"], label="colab version", timeout=90)):

            def auth() -> str:
                u = colab_usage(transport)
                active = int(u.get("active_assignments") or 0)
                note = " - another job is running on this account" if active else ""
                return (
                    f"signed in; balance {u['balance']} compute units, rate {u['rate_per_hour']}/hr, "
                    f"active runtimes {active}{note}"
                )

            add("colab_auth", auth)
    return checks
