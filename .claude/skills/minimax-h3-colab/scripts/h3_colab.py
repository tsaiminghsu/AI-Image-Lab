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
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_CODECS = ("png", "mjpeg", "webp", "bmp")
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
) -> list[str]:
    """Run cmd, feed each output line to on_line as it arrives, enforce a wall-clock timeout.

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
    try:
        eof = False
        while not eof:
            if time.monotonic() > deadline:
                raise ColabTimeout(f"{label} exceeded {timeout:g} seconds")
            try:
                item = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                eof = True
                continue
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

    def call(
        self,
        args: list[str],
        *,
        label: str,
        timeout: float,
        mount_dir: Path | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> str:
        name = f"h3-{uuid.uuid4().hex[:12]}"
        cmd = self.command(list(args), mount_dir=mount_dir, name=name)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        lines = run_streaming(
            cmd, label=label, timeout=timeout, on_line=on_line, on_abort=lambda: self.abort(name), env=env
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


@dataclasses.dataclass
class JobSpec:
    """One clip. The nested dict form (to_dict/from_dict) is the unit a future storyboard runner will
    queue per scene: job_id, scene_id, input, prompt, settings, output."""

    image: Path | str
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

    @classmethod
    def from_dict(cls, data: dict) -> JobSpec:
        prompt = data.get("prompt") or {}
        settings = data.get("settings") or {}
        if isinstance(prompt, str):
            prompt = {"description": prompt}
        image = (data.get("input") or {}).get("image")
        if not image:
            raise InputError("job spec needs input.image")
        return cls(
            image=image,
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
        )

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "scene_id": self.scene_id,
            "input": {"image": str(self.image)},
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
    image: Path
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

    @property
    def actual_seconds(self) -> float:
        return round(self.frames / FPS, 3)

    @property
    def remote_prefix(self) -> str:
        # Flat under /content: `colab upload` is not known to create missing remote directories.
        return f"/content/h3_{self.job_id}"

    def remote_job(self) -> dict:
        return {
            "job_id": self.job_id,
            "remote_image": f"{self.remote_prefix}_first_frame{self.image.suffix.lower()}",
            "remote_output": f"{self.remote_prefix}_output.mp4",
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


def prepare(spec: JobSpec, config: dict, prober: Any, *, job_id: str) -> Prepared:
    """Validate everything that can be validated without Colab. Raises InputError."""
    warnings = []
    if spec.mode != "first_frame":
        raise InputError(f"mode {spec.mode!r} is not implemented; this phase does first_frame (I2VA) only")
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
        output = out_dir / f"{_safe_stem(image.stem)}_h3_{job_id}.mp4"
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
    )


def stage_work_dir(p: Prepared) -> dict[str, Path]:
    """Everything a Colab call reads or writes sits in one directory: Docker mounts only that."""
    p.work_dir.mkdir(parents=True, exist_ok=True)
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
        stages[self.state] = round(stages.get(self.state, 0.0) + now - self._entered, 1)
        self._entered = now

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
        self._close_stage()
        self.record["finished_at"] = now_iso()
        self.record["elapsed_seconds"] = round(time.monotonic() - self._started, 1)
        self.log(f"finished: {self.record['status']} in {self.record['elapsed_seconds']} s")


class RemoteProgress:
    """Turns the remote script's H3_* markers into state changes and record fields."""

    def __init__(self, run: JobRun):
        self.run = run
        self.error: tuple[str, str] | None = None
        self.output: dict | None = None

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
        elif kind == "VRAM_PEAK":
            rec["vram_peak_mib"] = data.get("mib")
        elif kind == "ERROR":
            self.error = (str(data.get("code") or "INFERENCE_FAILED"), str(data.get("message") or ""))
        elif kind == "OUTPUT":
            self.output = data

    def raise_if_error(self) -> None:
        if not self.error:
            return
        code, message = self.error
        if code == "GPU_UNAVAILABLE":
            raise H3Error(code, GPU_UNAVAILABLE_MESSAGE, hint=message)
        if code == "TIMEOUT":
            raise H3Error(code, f"Remote inference timed out: {message}", status=TIMEOUT)
        raise H3Error(code, message or code)


LEDGER_FIELDS = (
    "job_id", "scene_id", "started_at", "finished_at", "status", "error_code", "failed_stage", "duration",
    "frames", "actual_seconds", "resolution", "seed", "gpu", "vram_peak_mib", "session", "output", "bytes",
    "elapsed_seconds", "session_seconds", "stage_seconds", "remote_seconds", "cu_balance_before",
    "cu_balance_after", "cu_used_measured", "cu_rate_per_hour", "cu_estimated", "comfyui_commit",
    "cli_version", "transport",
)  # fmt: skip


def _json_default(value: Any) -> Any:
    return str(value)


def run_job(
    spec: JobSpec,
    config: dict,
    *,
    transport: Transport | None = None,
    prober: Any = None,
    dry_run: bool = False,
    stream: Any = "stderr",
) -> dict:
    """Run one clip through the whole lifecycle and return its record (also written to disk)."""
    stream = sys.stderr if stream == "stderr" else stream
    job_id = spec.job_id or new_job_id()
    if not JOB_ID_RE.fullmatch(job_id):
        raise InputError(f"job_id {job_id!r} may only use letters, digits, _ and - (max 40)")
    out_dir = output_dir(config)
    run = JobRun(job_id, spec.scene_id, out_dir / "logs" / f"{job_id}.log", stream)
    rec = run.record
    prepared: Prepared | None = None
    session_attempted = False
    session_t0 = None
    rate_before = None
    try:
        run.advance(PREPARING)
        check_image_file(spec.image)
        if prober is None:
            path = find_ffprobe(config)
            if not path:
                raise H3Error(
                    "FFPROBE_MISSING",
                    "ffprobe was not found; every clip is validated with it, so the job does not start without it.",
                    hint="Set tools.ffprobe in config/config.json, or install it: winget install --id Gyan.FFmpeg",
                )
            prober = FFprobe(path)
        prepared = prepare(spec, config, prober, job_id=job_id)
        rec.update(
            {
                "image": str(prepared.image),
                "image_size": list(prepared.image_size),
                "duration": prepared.duration,
                "frames": prepared.frames,
                "actual_seconds": prepared.actual_seconds,
                "resolution": f"{prepared.width}x{prepared.height}",
                "seed": prepared.seed,
                "gpu_requested": prepared.gpu,
                "high_mem": prepared.high_mem,
                "timeout_seconds": prepared.timeout,
                "session": prepared.session,
                "output": str(prepared.output),
                "prompt": prepared.prompt,
                "spec": spec.to_dict(),
            }
        )
        for warning in prepared.warnings:
            rec["warnings"].append(warning)
            run.log("warning: " + warning)
        files = stage_work_dir(prepared)
        remote = prepared.remote_job()
        transport = transport or make_transport(config)
        rec["transport"] = transport.kind
        cfg = config["colab"]

        if dry_run:
            rec["dry_run"] = {
                "prompt": prepared.prompt,
                "remote_job": remote,
                "commands": [
                    transport.display(["usage"]),
                    transport.display(
                        ["new", "--session", prepared.session, "--gpu", prepared.gpu]
                        + (["--high-mem"] if prepared.high_mem else [])
                    ),
                    transport.display(
                        ["exec", "--session", prepared.session, "--timeout", str(prepared.timeout), "--env"]
                        + [f"H3_JOB_FILE={prepared.remote_prefix}_job.json", "--file", "/work/" + REMOTE_SCRIPT.name],
                        prepared.work_dir,
                    ),
                ],
                "preflight": preflight_checks(config, transport=transport, prober=prober),
            }
            rec["status"] = DRY_RUN
            return rec

        run.advance(CONNECTING_COLAB)
        transport.preflight()
        try:
            rec["cli_version"] = transport.call(["version"], label="colab version", timeout=90).splitlines()[-1]
        except (ColabCommandError, ColabTimeout, IndexError):
            rec["cli_version"] = None
        before = colab_usage(transport)
        rec["cu_balance_before"] = before.get("balance")
        rate_before = before.get("rate_per_hour")
        run.log(f"compute units before: {before.get('balance')} (rate {rate_before}/hr)")
        session_attempted = True
        session_t0 = time.monotonic()
        colab_new(
            transport,
            prepared.session,
            gpu=prepared.gpu,
            high_mem=prepared.high_mem,
            timeout=cfg["session_create_timeout_seconds"],
            on_line=run.log_remote,
        )
        try:
            during = colab_usage(transport)
            if during.get("rate_per_hour") is not None:
                rec["cu_rate_per_hour"] = round(during["rate_per_hour"] - (rate_before or 0.0), 3)
                run.log(f"session compute-unit rate: {rec['cu_rate_per_hour']}/hr")
        except H3Error as exc:
            run.log(f"could not read the usage rate after `colab new`: {exc}")

        run.advance(UPLOADING)
        transfer = cfg["transfer_timeout_seconds"]
        upload_file(
            transport,
            prepared.session,
            files["image"],
            remote["remote_image"],
            timeout=transfer,
            on_line=run.log_remote,
        )
        remote_job_path = f"{prepared.remote_prefix}_job.json"
        upload_file(
            transport, prepared.session, files["job"], remote_job_path, timeout=transfer, on_line=run.log_remote
        )

        run.advance(LOADING_MODEL)
        progress = RemoteProgress(run)
        try:
            transport.call(
                ["exec", "--session", prepared.session, "--timeout", str(prepared.timeout)]
                + ["--env", f"H3_JOB_FILE={remote_job_path}"]
                + ["--file", transport.cli_path(files["script"], prepared.work_dir)],
                label="colab exec",
                timeout=prepared.timeout + 120,
                mount_dir=prepared.work_dir,
                on_line=progress.on_line,
            )
        except ColabCommandError as exc:
            progress.raise_if_error()
            raise H3Error("INFERENCE_FAILED", f"`colab exec` failed: {exc}") from exc
        progress.raise_if_error()
        if not progress.output:
            raise H3Error("OUTPUT_NOT_FOUND", "Inference ended but the remote script reported no MP4 (no H3_OUTPUT).")

        run.advance(DOWNLOADING)
        local = files["output"]
        try:
            download_file(
                transport,
                prepared.session,
                progress.output.get("path") or remote["remote_output"],
                local,
                timeout=transfer,
                on_line=run.log_remote,
            )
        except ColabCommandError as exc:
            raise H3Error("OUTPUT_NOT_FOUND", f"Could not download the MP4: {exc}") from exc
        if not local.is_file() or local.stat().st_size == 0:
            raise H3Error("OUTPUT_NOT_FOUND", f"The downloaded MP4 is missing or empty: {local}")
        prepared.output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(local, prepared.output)
        rec["bytes"] = prepared.output.stat().st_size

        run.advance(VALIDATING)
        try:
            probe = prober.video(prepared.output)
        except ProbeError as exc:
            raise H3Error("FAILED_VALIDATION", f"ffprobe cannot read the MP4: {exc}", status=FAILED_VALIDATION) from exc
        rec["ffprobe"] = summarize_probe(probe)
        problems = check_probe(probe, width=prepared.width, height=prepared.height, frames=prepared.frames)
        if problems:
            raise H3Error("FAILED_VALIDATION", "; ".join(problems), status=FAILED_VALIDATION)
        run.advance(COMPLETED)
    except H3Error as exc:
        run.fail(exc.status, exc.code, str(exc), exc.hint)
    except ColabTimeout as exc:
        run.fail(
            TIMEOUT,
            "TIMEOUT",
            f"{exc}. The remote kernel may still be busy; the session is stopped instead of retrying.",
        )
    except ColabCommandError as exc:
        code = "UPLOAD_FAILED" if run.state == UPLOADING else "COLAB_COMMAND_FAILED"
        run.fail(FAILED, code, str(exc))
    except KeyboardInterrupt:
        run.fail(CANCELLED, "CANCELLED", "Cancelled by the user.")
    except Exception as exc:
        # Still stop the session and write the record; the traceback goes to the log.
        run.fail(FAILED, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
        run.log(traceback.format_exc())
    finally:
        if session_attempted and transport is not None and prepared is not None:
            _stop_and_account(run, transport, prepared.session, session_t0)
        if rec["status"] == DRY_RUN and prepared is not None:
            shutil.rmtree(prepared.work_dir, ignore_errors=True)
        if rec["status"] != DRY_RUN:
            run.finish()
            _write_records(run, prepared, out_dir)
            if rec["status"] == COMPLETED and prepared is not None:
                shutil.rmtree(prepared.work_dir, ignore_errors=True)
            elif prepared is not None:
                run.log(f"work files kept for debugging: {prepared.work_dir}")
    return rec


def _stop_and_account(run: JobRun, transport: Transport, session: str, session_t0: float | None) -> None:
    rec = run.record
    try:
        stop_session(transport, session, on_line=run.log_remote)
        rec["session_status"] = "stopped"
    except (ColabCommandError, ColabTimeout, KeyboardInterrupt) as exc:
        text = exc.text if isinstance(exc, ColabCommandError) else str(exc)
        if "not found" in text.lower():
            rec["session_status"] = "not_created"
        else:
            rec["session_status"] = "stop_failed"
            manual = transport.display(["stop", "--session", session])
            rec["warnings"].append(f"Colab session {session} may still be running and spending compute units: {manual}")
            run.log(f"WARNING: could not stop session {session}; stop it by hand: {manual}")
    if session_t0 is not None:
        rec["session_seconds"] = round(time.monotonic() - session_t0, 1)
        rate = rec.get("cu_rate_per_hour")
        if rate:
            # An estimate from the hourly rate; cu_used_measured (balance difference) is the real figure.
            rec["cu_estimated"] = round(rate * rec["session_seconds"] / 3600, 3)
    try:
        after = colab_usage(transport)
        rec["cu_balance_after"] = after.get("balance")
        if rec.get("cu_balance_before") is not None and after.get("balance") is not None:
            rec["cu_used_measured"] = round(rec["cu_balance_before"] - after["balance"], 3)
    except (H3Error, KeyboardInterrupt) as exc:
        run.log(f"could not read the compute-unit balance after the job: {exc}")


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
                return f"signed in; balance {u['balance']} compute units, rate {u['rate_per_hour']}/hr"

            add("colab_auth", auth)
    return checks
