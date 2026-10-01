"""Is the output a usable file? A job is completed only after its output passed these checks.

Images: PNG structure from the raw bytes (signature, IHDR, every chunk CRC, IDAT, IEND), the expected size,
and a full decode when Pillow is importable. Videos: ffprobe (codec, pixel format, size, frame rate, duration,
audio stream) against what the workflow's registry entry expects.
"""

import hashlib
import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

from . import WorkflowError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def sha256_file(path, block=8 * 2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def png_structure(data):
    """(width, height, problems) from the raw bytes."""
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


def check_image(path, expect):
    """Problems with an output image (empty list = valid)."""
    path = Path(path)
    if not path.is_file():
        return ["output file missing: %s" % path]
    if path.stat().st_size == 0:
        return ["output file is empty"]
    w, h, problems = png_structure(path.read_bytes())
    if problems:
        return problems
    want = (expect.get("width"), expect.get("height"))
    if all(want) and (w, h) != want:
        problems.append("resolution %sx%s, expected %sx%s" % (w, h, want[0], want[1]))
    try:
        from PIL import Image
    except ImportError:
        return problems
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure means the file is not usable
        problems.append("Pillow cannot decode it: %s: %s" % (type(exc).__name__, exc))
    return problems


def image_size(path):
    """(width, height) of an input image: PNG from its header, anything else through Pillow."""
    path = Path(path)
    data = path.read_bytes()[:64]
    if data.startswith(PNG_SIGNATURE) and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    try:
        from PIL import Image
    except ImportError:
        raise WorkflowError("INVALID_INPUT", "%s is not a PNG and Pillow is not available to read it" % path.name)
    try:
        with Image.open(path) as im:
            return im.size
    except Exception as exc:  # noqa: BLE001
        raise WorkflowError("INVALID_INPUT", "%s is not a readable image: %s" % (path.name, exc))


VIDEO_ENTRIES = (
    "format=duration,size:stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
    "nb_frames,duration"
)


def probe_video(path):
    """ffprobe's JSON for a video file."""
    if not shutil.which("ffprobe"):
        raise WorkflowError("FFPROBE_MISSING", "ffprobe is not available to validate the video")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", VIDEO_ENTRIES, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if out.returncode != 0:
        raise WorkflowError("INVALID_OUTPUT", "ffprobe cannot read %s: %s" % (path, out.stderr[-500:]))
    return json.loads(out.stdout or "{}")


def _ratio(value):
    try:
        num, _, den = str(value).partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_probe(probe):
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = probe.get("format", {})
    fps = _ratio(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    duration = _float(video.get("duration"))
    if duration is None and str(video.get("nb_frames", "")).isdigit() and fps:
        duration = int(video["nb_frames"]) / fps
    if duration is None:
        duration = _float(fmt.get("duration"))
    return {
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "pix_fmt": video.get("pix_fmt"),
        "fps": fps,
        "duration": duration,
        "audio_codec": audio.get("codec_name"),
    }


def check_video(probe, expect):
    """Problems with a video (empty list = valid). Pure, so every rule is unit-tested."""
    s = summarize_probe(probe)
    if s["video_codec"] is None:
        return ["no video stream"]
    problems = []
    if expect.get("video_codec") and s["video_codec"] != expect["video_codec"]:
        problems.append("video codec %s, expected %s" % (s["video_codec"], expect["video_codec"]))
    if expect.get("pix_fmt") and s["pix_fmt"] != expect["pix_fmt"]:
        problems.append("pixel format %s, expected %s" % (s["pix_fmt"], expect["pix_fmt"]))
    if expect.get("width") and (s["width"], s["height"]) != (expect["width"], expect["height"]):
        problems.append(
            "resolution %sx%s, expected %sx%s" % (s["width"], s["height"], expect["width"], expect["height"])
        )
    fps = expect.get("fps")
    if fps and (s["fps"] is None or abs(s["fps"] - fps) > 0.01):
        problems.append("frame rate %s, expected %s" % (s["fps"], fps))
    if fps and expect.get("frames"):
        want = expect["frames"] / fps
        tolerance = max(0.1, 2 / fps)
        if s["duration"] is None or abs(s["duration"] - want) > tolerance:
            problems.append("duration %s s, expected %.3f s (+-%.2f)" % (s["duration"], want, tolerance))
    if expect.get("audio") and s["audio_codec"] is None:
        problems.append("no audio stream")
    return problems


def make_preview(video, target):
    """First frame of a video as a JPEG. Best effort: a missing preview never fails a job."""
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "3", "-update", "1", str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return out.returncode == 0 and Path(target).is_file()
