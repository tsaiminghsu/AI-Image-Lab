"""Shared imports and test doubles for the MiniMax H3 Colab skill tests (tests/test_minimax_h3_*.py).

The skill lives in .claude/skills/minimax-h3-colab/, outside the pytest pythonpath. Its scripts dir is
put on sys.path only for the import, like tests/test_worker_handler.py does for worker/.
"""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "minimax-h3-colab"
SCRIPTS_DIR = SKILL_DIR / "scripts"
REMOTE_PATH = SKILL_DIR / "remote" / "h3_colab_job.py"


def import_skill(name):
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


def import_remote():
    if "h3_colab_job" in sys.modules:
        return sys.modules["h3_colab_job"]
    spec = importlib.util.spec_from_file_location("h3_colab_job", REMOTE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["h3_colab_job"] = module
    spec.loader.exec_module(module)
    return module


h3 = import_skill("h3_colab")


def make_config(tmp_path, **env):
    """Defaults from config.example.json, output under tmp_path, and no local config.json. Google Drive is
    off unless a test asks for it (H3_DRIVE="consent"): most tests are about the job state machine."""
    env = {"H3_OUTPUT_DIR": str(tmp_path / "out"), "H3_CU_SETTLE_SECONDS": "0", "H3_DRIVE": "off", **env}
    return h3.load_config(tmp_path / "no-config.json", env=env)


def make_image(tmp_path, name="first.png"):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    return path


def good_probe(width=1376, height=768, frames=192, *, codec="h264", pix_fmt="yuv420p", fps="24/1", audio=True):
    seconds = frames / 24
    streams = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": codec,
            "width": width,
            "height": height,
            "pix_fmt": pix_fmt,
            "r_frame_rate": fps,
            "avg_frame_rate": fps,
            "nb_frames": str(frames),
            "duration": f"{seconds:.6f}",
        }
    ]
    if audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "32000",
                "channels": 2,
                "duration": f"{seconds:.6f}",
            }
        )
    return {"streams": streams, "format": {"duration": f"{seconds:.6f}", "size": "4194304"}}


class FakeProber:
    def __init__(self, image=(1920, 1080, "png"), video=None, image_error=None, video_error=None):
        self.image = image
        self._video = video
        self.image_error = image_error
        self.video_error = video_error

    def image_info(self, path):
        if self.image_error:
            raise h3.ProbeError(self.image_error)
        return self.image

    def video(self, path):
        if self.video_error:
            raise h3.ProbeError(self.video_error)
        return self._video if self._video is not None else good_probe()


def marker(kind, **fields):
    return "H3_%s %s" % (kind, json.dumps(fields))


HAPPY_EXEC_LINES = [
    marker("GPU", name="NVIDIA A100-SXM4-40GB", vram_gib=39.6, cuda="12.4"),
    marker("COMFYUI", commit="abc123"),
    marker("TIMING", stage="setup", seconds=150.0),
    marker("STAGE", name="LOADING_MODEL"),
    marker("TIMING", stage="download", seconds=300.0),
    marker("CONFIG", diffusion="minimax_h3_fl2va_pruned_fp8_scaled.safetensors", steps=8),
    marker("STAGE", name="INFERENCE"),
    marker("HEARTBEAT", queued=False),
    marker("TIMING", stage="inference", seconds=420.0),
    marker("VRAM_PEAK", mib=38000),
    marker("OUTPUT", path="/content/h3_job_output.mp4", bytes=4194304, streams=["audio", "video"]),
    marker("LAST_FRAME", path="/content/h3_job_last_frame.png"),
]


class FakeTransport(h3.Transport):
    """Scripted `colab`: handlers[subcommand](args, on_line) returns output text or raises."""

    kind = "fake"

    def __init__(self, mount=None, **handlers):
        super().__init__("oauth2")
        self.calls = []
        self.idle_timeouts = []
        self.mount = mount  # what `colab drivemount` answers; None = it must not be called
        self.mount_calls = []
        self._usage = iter(
            [
                "Current balance: 100.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0",
                "Current balance: 100.00 compute units\nUsage rate: 11.77/hr\nActive assignments: 1",
            ]
        )
        self.handlers = {
            "version": lambda args, on_line: "0.7.4",
            "usage": self._default_usage,
            "new": lambda args, on_line: "[colab] Session READY.",
            "upload": lambda args, on_line: "[colab] Uploaded",
            "exec": self._default_exec,
            "download": self._default_download,
            "stop": lambda args, on_line: "[colab] Session terminated.",
        }
        self.handlers.update(handlers)

    def _default_usage(self, args, on_line):
        return next(self._usage, "Current balance: 98.50 compute units\nUsage rate: 0.00/hr\nActive assignments: 0")

    @staticmethod
    def _default_exec(args, on_line):
        for line in HAPPY_EXEC_LINES:
            on_line(line)
        return ""

    @staticmethod
    def _default_download(args, on_line):
        Path(args[-1]).write_bytes(b"\0\0\0\x18ftypmp42" + b"\0" * 1024)
        return "[colab] Downloaded"

    def command(self, args, *, mount_dir, name):
        return ["colab", f"--auth={self.auth}", *args]

    def cli_path(self, local, mount_dir):
        return str(Path(local).resolve())

    def login_command(self):
        return "colab --auth=oauth2 usage"

    def mount_drive(self, session, mount, *, wait_seconds, signal_path, on_url, on_line=None):
        assert self.mount is not None, "drivemount was called although Drive is off"
        self.calls.append(["drivemount", "--session", session, mount])
        self.idle_timeouts.append(None)
        self.mount_calls.append({"session": session, "mount": mount, "wait_seconds": wait_seconds})
        if isinstance(self.mount, Exception):
            raise self.mount
        if self.mount.get("consent_asked"):
            on_url("https://accounts.google.com/o/oauth2/auth?client_id=fake")
        return dict(self.mount)

    def call(self, args, *, label, timeout, mount_dir=None, on_line=None, idle_timeout=None):
        self.calls.append(list(args))
        self.idle_timeouts.append(idle_timeout)
        return self.handlers[args[0]](list(args), on_line or (lambda line: None))

    @property
    def subcommands(self):
        return [c[0] for c in self.calls]


def emit(*lines, then=None):
    """An exec handler that streams lines, then returns normally (colab exec exits 0 even when the
    remote code raised) or raises `then`."""

    def handler(args, on_line):
        for line in lines:
            on_line(line)
        if then is not None:
            raise then
        return ""

    return handler


def raising(exc):
    def handler(args, on_line):
        raise exc

    return handler


# --- model files (the Drive storage tests) -----------------------------------------------------------


def blob(name):
    return (name.encode() + b"|") * 30


def small_specs(config=None, names=None):
    """The skill's pinned model entries with the multi-GB size and sha256 replaced by those of a tiny blob."""
    import hashlib

    remote = import_remote()
    names = names or [remote.DIFFUSION_FP8, remote.TEXT_ENCODER, remote.VIDEO_VAE, remote.AUDIO_VAE, remote.LORA]
    config = config or json.loads((SKILL_DIR / "config" / "config.example.json").read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in config["models"]}
    return [dict(by_name[n], size=len(blob(n)), sha256=hashlib.sha256(blob(n)).hexdigest()) for n in names]


def fake_hub(calls, fail=False, blobs=blob):
    """A stand-in for the huggingface_hub module: writes the blob for the requested file under local_dir."""
    import types

    def hf_hub_download(repo_id, filename, revision, local_dir, token):
        calls.append({"repo": repo_id, "file": filename, "revision": revision, "token": token})
        if fail:
            raise RuntimeError("HfHubHTTPError: 503 Service Unavailable")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blobs(filename.rsplit("/", 1)[-1]))
        return str(target)

    return types.SimpleNamespace(hf_hub_download=hf_hub_download)
