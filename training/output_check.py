"""Technical delivery check: re-read the bytes a generation actually produced and say, in a
report that cannot overstate itself, whether they are what was asked for.

This is one of three separate things, and conflating them is the mistake the shape here exists
to prevent:

  technical delivery  - the file exists, hashes to what we recorded, parses, and has the
                        dimensions that were requested. That is all this module claims.
  quality approval    - a human decided the picture is good. Nothing here decides that.
  delivered           - somebody actually received it.

`scope` is the literal "technical_delivery" rather than a computed string, so a report can
never widen the claim it is making.

The honest-by-construction bit. `checks` lists what actually ran, and `policy.sha256` is
computed over that list plus the version. So a report produced without Pillow (checks omit
"full_decode") hashes differently from one produced with it, and a report on a container whose
dimensions we could not derive hashes differently again. A weaker check therefore cannot
masquerade as a stronger one - the hash gives it away rather than a comment promising it won't
happen.

Pillow is not in requirements-dev.txt (the offline suite runs without it), so every parser here
is stdlib header reading. When Pillow *is* present - it is inside ComfyUI\\.venv - a full decode
is added on top, using the same lazy try/except ImportError pattern worker/jobs.py already uses
for decode_reference.
"""

import hashlib
import json
import os
import struct

POLICY_ID = "technical_delivery"
POLICY_VERSION = 1
SCOPE = "technical_delivery"

PASSED = "passed"
FAILED = "failed"
MANUAL_REVIEW_REQUIRED = "manual_review_required"

CHECK_SHA256 = "actual_bytes_sha256"
CHECK_CONTAINER = "container_parse"
CHECK_FORMAT = "format"
CHECK_DIMENSIONS = "dimensions"
CHECK_FULL_DECODE = "full_decode"

# Reading a whole 18 GB video into memory to hash it would defeat the point of checking it.
_HASH_CHUNK = 1 << 20

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_GIF_MAGICS = (b"GIF87a", b"GIF89a")

_IMAGE_FORMATS = ("png", "jpeg", "webp", "gif")
_VIDEO_FORMATS = ("mp4", "webm")


class OutputCheckError(Exception):
    """The file could not be read at all. A parse failure is a report, not an exception."""


def _pillow():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def policy_sha256(checks, version=POLICY_VERSION):
    """Hash the policy itself. Computed, never a constant: the whole mechanism depends on a
    different set of checks producing a different hash."""
    canonical = json.dumps({"id": POLICY_ID, "version": version, "checks": sorted(checks)},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- container parsers ---------------------------------------------------------------------------
# Each returns (format_name, width, height); width/height may be None when the container parsed
# but the dimensions are somewhere this reader does not go. That distinction matters: it is the
# difference between "the file is broken" and "we checked less than usual", and only the second
# is allowed to quietly drop a check from the policy.


def _parse_png(data):
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise ValueError("PNG header is truncated or has no IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    return "png", width, height


def _parse_jpeg(data):
    # Walk the marker chain. SOFn carries the dimensions; which SOF depends on the coding mode,
    # and DHT/DQT/APPn segments sit in front of it in an order nothing guarantees.
    pos = 2
    end = len(data)
    while pos + 4 <= end:
        if data[pos] != 0xFF:
            raise ValueError(f"JPEG marker expected at offset {pos}")
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if pos + 4 > end:
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if pos + 9 > end:
                raise ValueError("JPEG SOF segment is truncated")
            height, width = struct.unpack(">HH", data[pos + 5:pos + 9])
            return "jpeg", width, height
        if marker == 0xDA:  # start of scan - no SOF came first
            break
        pos += 2 + length
    raise ValueError("JPEG has no SOF segment")


def _parse_gif(data):
    if len(data) < 10:
        raise ValueError("GIF header is truncated")
    width, height = struct.unpack("<HH", data[6:10])
    return "gif", width, height


def _parse_webp(data):
    if len(data) < 16 or data[8:12] != b"WEBP":
        raise ValueError("not a RIFF/WEBP file")
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return "webp", width, height
    if chunk == b"VP8 " and len(data) >= 30:
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return "webp", width, height
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return "webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return "webp", None, None


def _iter_boxes(data, start, end):
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = pos + 8
        if size == 1:  # 64-bit extended size
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            body = pos + 16
        elif size == 0:
            size = end - pos
        if size < 8 or pos + size > end:
            return
        yield kind, body, pos + size
        pos += size


def _parse_mp4(data):
    if len(data) < 12 or data[4:8] != b"ftyp":
        raise ValueError("MP4 does not start with an ftyp box")
    for kind, body, box_end in _iter_boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for trak_kind, trak_body, trak_end in _iter_boxes(data, body, box_end):
            if trak_kind != b"trak":
                continue
            for tkhd_kind, tkhd_body, tkhd_end in _iter_boxes(data, trak_body, trak_end):
                if tkhd_kind != b"tkhd" or tkhd_end - tkhd_body < 8:
                    continue
                # width and height are the final two 16.16 fixed-point fields of tkhd, in both
                # version 0 and version 1 layouts - taking them from the end skips having to
                # branch on the version at all.
                width, height = struct.unpack(">II", data[tkhd_end - 8:tkhd_end])
                if width >> 16 and height >> 16:
                    return "mp4", width >> 16, height >> 16
    # A parseable MP4 whose track header we could not reach: real file, smaller claim.
    return "mp4", None, None


def _ebml_vint(data, pos, keep_marker):
    first = data[pos]
    if first == 0:
        raise ValueError("invalid EBML variable-length integer")
    length = 1
    mask = 0x80
    while not first & mask:
        mask >>= 1
        length += 1
    value = first if keep_marker else first & (mask - 1)
    for offset in range(1, length):
        value = (value << 8) | data[pos + offset]
    return value, pos + length


_EBML_MASTERS = {0x18538067, 0x1654AE6B, 0xAE, 0xE0}  # Segment, Tracks, TrackEntry, Video
_EBML_PIXEL_WIDTH = 0xB0
_EBML_PIXEL_HEIGHT = 0xBA


def _ebml_scan(data, pos, end, found, depth=0):
    while pos < end and depth < 8 and not (found.get("w") and found.get("h")):
        element_id, pos = _ebml_vint(data, pos, keep_marker=True)
        size, pos = _ebml_vint(data, pos, keep_marker=False)
        stop = min(end, pos + size)
        if element_id in _EBML_MASTERS:
            _ebml_scan(data, pos, stop, found, depth + 1)
        elif element_id == _EBML_PIXEL_WIDTH:
            found["w"] = int.from_bytes(data[pos:stop], "big")
        elif element_id == _EBML_PIXEL_HEIGHT:
            found["h"] = int.from_bytes(data[pos:stop], "big")
        pos = stop
    return found


def _parse_webm(data):
    if len(data) < 4 or data[:4] != b"\x1a\x45\xdf\xa3":
        raise ValueError("not an EBML/WebM file")
    try:
        found = _ebml_scan(data, 0, len(data), {})
    except (ValueError, IndexError):
        return "webm", None, None
    return "webm", found.get("w"), found.get("h")


_SNIFFERS = (
    (lambda d: d.startswith(_PNG_MAGIC), _parse_png),
    (lambda d: d.startswith(_JPEG_MAGIC), _parse_jpeg),
    (lambda d: any(d.startswith(m) for m in _GIF_MAGICS), _parse_gif),
    (lambda d: d[:4] == b"RIFF" and d[8:12] == b"WEBP", _parse_webp),
    (lambda d: d[:4] == b"\x1a\x45\xdf\xa3", _parse_webm),
    (lambda d: len(d) >= 8 and d[4:8] == b"ftyp", _parse_mp4),
)


def parse_container(data):
    """Sniff by magic bytes and return (format, width, height). Raises ValueError when the bytes
    are not a container we recognise or the header is malformed."""
    for matches, parser in _SNIFFERS:
        if matches(data):
            return parser(data)
    raise ValueError(f"unrecognised container (first bytes: {data[:8]!r})")


# --- the check ------------------------------------------------------------------------------------


def _report(state, checks, details):
    return {
        "policy": {
            "id": POLICY_ID,
            "version": POLICY_VERSION,
            "checks": sorted(checks),
            "sha256": policy_sha256(checks),
        },
        "state": state,
        "scope": SCOPE,
        "details": details,
    }


def check_output(path, *, media_kind, expect_sha256=None, expect_width=None, expect_height=None,
                 header_bytes=1 << 16):
    """Re-read `path` and report on it.

    Returns (report, facts) where facts carries sha256, byte_length, format, width and height -
    everything the caller needs to build an Artifact without reading the file a second time.

    A file that cannot be opened raises; a file that opens but does not parse is a `failed`
    report. The difference is deliberate: the first is our bug, the second is the provider's,
    and only the second belongs in a job record.
    """
    if not os.path.isfile(path):
        raise OutputCheckError(f"output file does not exist: {path}")

    checks = {CHECK_SHA256, CHECK_CONTAINER, CHECK_FORMAT}
    failures = []
    byte_length = os.path.getsize(path)
    actual_sha256 = sha256_file(path)
    if expect_sha256 is not None and actual_sha256 != expect_sha256:
        failures.append(f"sha256 mismatch: expected {expect_sha256}, read {actual_sha256}")

    with open(path, "rb") as handle:
        head = handle.read(header_bytes)

    fmt = width = height = None
    try:
        fmt, width, height = parse_container(head)
    except ValueError as exc:
        failures.append(f"{CHECK_CONTAINER}: {exc}")
    else:
        expected_formats = _IMAGE_FORMATS if media_kind == "image" else _VIDEO_FORMATS
        if fmt not in expected_formats:
            failures.append(f"format: {fmt} is not a {media_kind} container")
        if width and height:
            checks.add(CHECK_DIMENSIONS)
            if expect_width is not None and width != expect_width:
                failures.append(f"width: asked for {expect_width}, got {width}")
            if expect_height is not None and height != expect_height:
                failures.append(f"height: asked for {expect_height}, got {height}")

    image_module = _pillow() if media_kind == "image" and fmt else None
    if image_module is not None:
        checks.add(CHECK_FULL_DECODE)
        try:
            with image_module.open(path) as image:
                image.load()
        except Exception as exc:  # Pillow raises a wide family here, all meaning the same thing
            failures.append(f"{CHECK_FULL_DECODE}: {exc}")

    state = FAILED if failures else PASSED
    details = {"failures": failures, "format": fmt, "width": width, "height": height}
    facts = {
        "sha256": actual_sha256,
        "byte_length": byte_length,
        "format": fmt,
        "width": width,
        "height": height,
    }
    return _report(state, checks, details), facts
