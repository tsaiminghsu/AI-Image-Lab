"""Pins the technical delivery check, and above all the property that makes the report
trustworthy: a weaker check cannot masquerade as a stronger one.

`checks` lists what actually ran and `policy.sha256` is computed over that list, so a report
produced without Pillow, or on a container whose dimensions could not be derived, hashes
differently from a full one. test_policy_hash_is_computed_not_constant and
test_a_smaller_check_set_hashes_differently are what keep that mechanism real - if someone ever
replaces the computed hash with a literal, the whole guarantee silently evaporates and every
other test here would still pass.
"""

import struct
import zlib

import job_contracts as jc
import output_check as oc
import pytest

# --- fixture builders ---------------------------------------------------------------------------


def make_png(width, height):
    """A real, decodable PNG - not a synthetic header. The full_decode check runs Pillow when it
    is available (it is inside ComfyUI\\.venv, not in .venv-dev), so a fake header would make this
    file's result depend on which venv ran it."""

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    scanlines = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def make_mp4(width, height):
    def box(kind, payload):
        return struct.pack(">I", len(payload) + 8) + kind + payload

    tkhd = box(b"tkhd", b"\x00" * 76 + struct.pack(">II", width << 16, height << 16))
    return box(b"ftyp", b"isom" + b"\x00" * 4 + b"isom") + box(b"moov", box(b"trak", tkhd))


def make_webm(width, height):
    def element(element_id, payload):
        return element_id + b"\x01" + len(payload).to_bytes(7, "big") + payload

    video = element(b"\xb0", width.to_bytes(2, "big")) + element(b"\xba", height.to_bytes(2, "big"))
    track = element(b"\xae", element(b"\xe0", video))
    segment = element(b"\x18\x53\x80\x67", element(b"\x16\x54\xae\x6b", track))
    return element(b"\x1a\x45\xdf\xa3", b"\x00") + segment


def make_jpeg(width, height):
    """Header only. JPEG is exercised through parse_container rather than check_output, because
    hand-building entropy-coded scan data Pillow will decode is a lot of machinery for no extra
    coverage of this module."""
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9 + sof


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


# --- container parsing --------------------------------------------------------------------------


def test_png_dimensions_come_from_ihdr():
    assert oc.parse_container(make_png(320, 192)) == ("png", 320, 192)


def test_jpeg_dimensions_come_from_the_sof_segment():
    """The SOF marker sits behind APPn/DQT/DHT segments in an order nothing guarantees, so the
    parser has to walk the marker chain rather than read a fixed offset."""
    assert oc.parse_container(make_jpeg(640, 480)) == ("jpeg", 640, 480)


def test_gif_dimensions_are_little_endian():
    gif = b"GIF89a" + struct.pack("<HH", 48, 64) + b"\x00\x00"
    assert oc.parse_container(gif) == ("gif", 48, 64)


def test_mp4_dimensions_come_from_the_track_header():
    assert oc.parse_container(make_mp4(704, 1280)) == ("mp4", 704, 1280)


def test_webm_dimensions_come_from_an_ebml_walk():
    assert oc.parse_container(make_webm(512, 768)) == ("webm", 512, 768)


def test_unrecognised_bytes_raise_rather_than_guess():
    with pytest.raises(ValueError, match="unrecognised container"):
        oc.parse_container(b"not a picture at all")


def test_a_truncated_png_header_is_a_parse_error():
    with pytest.raises(ValueError, match="IHDR"):
        oc.parse_container(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)


# --- the policy hash is the honesty mechanism ---------------------------------------------------


def test_policy_hash_is_computed_not_constant():
    """If this ever becomes a literal, every other assertion in this file still passes while the
    guarantee it protects is gone."""
    assert oc.policy_sha256({oc.CHECK_SHA256}) != oc.policy_sha256({oc.CHECK_SHA256, oc.CHECK_DIMENSIONS})


def test_policy_hash_ignores_check_ordering():
    assert oc.policy_sha256([oc.CHECK_FORMAT, oc.CHECK_SHA256]) == oc.policy_sha256([oc.CHECK_SHA256, oc.CHECK_FORMAT])


def test_a_smaller_check_set_hashes_differently(tmp_path):
    """A video whose dimensions we could not derive must not produce a report that looks identical
    to one where we did check them. Both sides are asked for the same expectation, so the only
    difference is whether the container would give its size up."""
    expectation = {"media_kind": "video", "expect_width": 512, "expect_height": 768}
    full, _ = oc.check_output(write(tmp_path, "a.webm", make_webm(512, 768)), **expectation)
    headerless = make_webm(512, 768).replace(b"\xb0", b"\xb1", 1)
    partial, _ = oc.check_output(write(tmp_path, "b.webm", headerless), **expectation)
    assert oc.CHECK_DIMENSIONS in full["policy"]["checks"]
    assert oc.CHECK_DIMENSIONS not in partial["policy"]["checks"]
    assert full["policy"]["sha256"] != partial["policy"]["sha256"]


def test_dimensions_are_measured_but_not_claimed_when_nothing_was_expected(tmp_path):
    """A check that asserted nothing must not appear in the checks list, or policy.sha256 stops
    meaning what it says. The measured size still lands in details."""
    report, facts = oc.check_output(write(tmp_path, "x.png", make_png(48, 96)), media_kind="image")
    assert report["state"] == oc.PASSED
    assert oc.CHECK_DIMENSIONS not in report["policy"]["checks"]
    assert report["details"]["width"] == 48 and facts["height"] == 96


# --- check_output -------------------------------------------------------------------------------


def test_a_matching_file_passes_and_reports_its_facts(tmp_path):
    data = make_png(256, 320)
    path = write(tmp_path, "ok.png", data)
    report, facts = oc.check_output(path, media_kind="image", expect_width=256, expect_height=320)
    assert report["state"] == oc.PASSED
    assert facts["width"] == 256 and facts["height"] == 320
    assert facts["byte_length"] == len(data)
    assert facts["sha256"] == oc.sha256_file(path)


def test_sha256_mismatch_fails(tmp_path):
    path = write(tmp_path, "ok.png", make_png(64, 64))
    report, _ = oc.check_output(path, media_kind="image", expect_sha256="b" * 64)
    assert report["state"] == oc.FAILED
    assert any("sha256 mismatch" in failure for failure in report["details"]["failures"])


def test_a_truncated_file_fails_and_names_the_container_check(tmp_path):
    path = write(tmp_path, "bad.png", make_png(64, 64)[:12])
    report, _ = oc.check_output(path, media_kind="image")
    assert report["state"] == oc.FAILED
    assert any(oc.CHECK_CONTAINER in failure for failure in report["details"]["failures"])


def test_wrong_dimensions_fail(tmp_path):
    path = write(tmp_path, "small.png", make_png(64, 64))
    report, _ = oc.check_output(path, media_kind="image", expect_width=1024, expect_height=1024)
    assert report["state"] == oc.FAILED
    failures = " ".join(report["details"]["failures"])
    assert "asked for 1024, got 64" in failures


def test_a_video_delivered_as_an_image_fails(tmp_path):
    """The container parsed fine; it is simply not the kind of thing this job was supposed to
    produce. That has to be a failure, not a shrug."""
    path = write(tmp_path, "clip.mp4", make_mp4(704, 1280))
    report, _ = oc.check_output(path, media_kind="image")
    assert report["state"] == oc.FAILED
    assert any("is not a image container" in failure for failure in report["details"]["failures"])


def test_a_missing_file_raises_rather_than_reporting(tmp_path):
    """A file we cannot open is our bug; a file that will not parse is the provider's. Only the
    second belongs in a job record."""
    with pytest.raises(oc.OutputCheckError, match="does not exist"):
        oc.check_output(str(tmp_path / "nope.png"), media_kind="image")


# --- the report is honest about its own scope ---------------------------------------------------


def test_scope_is_the_literal(tmp_path):
    report, _ = oc.check_output(write(tmp_path, "ok.png", make_png(32, 32)), media_kind="image")
    assert report["scope"] == "technical_delivery"
    assert report["policy"]["id"] == "technical_delivery"


def test_the_report_is_accepted_by_the_job_contract(tmp_path):
    """The two modules have to agree on the report shape, or a checked job is unstorable."""
    report, facts = oc.check_output(write(tmp_path, "ok.png", make_png(32, 32)), media_kind="image")
    record = jc.new_record(
        job_id="j",
        owner_id="local",
        mode="txt2img_hq",
        media_kind="image",
        catalog_id="c",
        backend="local",
        params={"prompt": "x"},
        now=0.0,
    )
    record["output_check"] = report
    record["status"] = jc.COMPLETED
    record["artifacts"] = [
        {
            "bucket": "local-outputs",
            "key": f"generated/j/{facts['sha256']}.png",
            "sha256": facts["sha256"],
            "media_kind": "image",
            "content_type": "image/png",
            "byte_length": facts["byte_length"],
        }
    ]
    assert jc.validate_record(record)["output_check"]["state"] == oc.PASSED


def test_manual_review_required_exists_but_nothing_produces_it(tmp_path):
    """It stays in the vocabulary so `scope` remains an honest literal rather than implying a
    review that does not exist. No reviewer exists in this repo, so nothing should emit it."""
    assert oc.MANUAL_REVIEW_REQUIRED in jc.OUTPUT_CHECK_STATES
    for name, size in (("a.png", (32, 32)), ("b.png", (8, 8))):
        report, _ = oc.check_output(write(tmp_path, name, make_png(*size)), media_kind="image")
        assert report["state"] in (oc.PASSED, oc.FAILED)
