"""Output validation: a job is not COMPLETED because ComfyUI said so, but because the PNG that reached
this machine exists, is not empty, is structurally a PNG and has the requested size. The checker is
stdlib-only (Pillow adds a full decode where it is installed), so it runs the same on the Colab VM,
in ComfyUI's venv and in the torch-free dev venv."""

import json
import struct
import zlib

import pytest
from zimage_colab_fakes import (
    FakeTransport,
    import_skill,
    make_config,
    png_bytes,
    remote,
    write_png,
    z,
)

validate_cli = import_skill("validate")


def test_a_good_png_has_no_problems(tmp_path):
    assert z.validate_output(write_png(tmp_path / "ok.png", 64, 48), 64, 48) == []


def test_a_missing_file(tmp_path):
    problems = z.validate_output(tmp_path / "nope.png", 64, 48)
    assert len(problems) == 1 and "missing" in problems[0]


def test_an_empty_file(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    assert z.validate_output(path, 64, 48) == ["output file is empty"]


def test_the_wrong_resolution(tmp_path):
    problems = z.validate_output(write_png(tmp_path / "small.png", 32, 24), 64, 48)
    assert any("resolution 32x24, expected 64x48" in p for p in problems)


@pytest.mark.parametrize(
    "data,needle",
    [
        (b"\xff\xd8\xff\xe0" + b"\x00" * 64, "not a PNG"),  # a JPEG with a .png name
        (b"<html>502 Bad Gateway</html>", "not a PNG"),
        (png_bytes(64, 48)[:-20], "truncated"),  # the download was cut off
        (png_bytes(64, 48)[: len(remote.PNG_SIGNATURE) + 25], "no IDAT"),
    ],
)
def test_files_that_are_not_complete_pngs(tmp_path, data, needle):
    path = tmp_path / "bad.png"
    path.write_bytes(data)
    problems = z.validate_output(path, 64, 48)
    assert any(needle in p for p in problems), problems


def test_a_flipped_byte_fails_the_chunk_crc(tmp_path):
    data = bytearray(png_bytes(64, 48))
    data[len(data) // 2] ^= 0xFF
    path = tmp_path / "corrupt.png"
    path.write_bytes(bytes(data))
    assert any("bad CRC" in p for p in z.validate_output(path, 64, 48))


def test_a_png_that_does_not_start_with_ihdr(tmp_path):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    path = tmp_path / "odd.png"
    path.write_bytes(remote.PNG_SIGNATURE + chunk(b"IDAT", b"x") + chunk(b"IEND", b""))
    assert any("first chunk is not IHDR" in p for p in z.validate_output(path, 64, 48))


def test_only_png_is_an_accepted_format(tmp_path):
    assert remote.validate_image(write_png(tmp_path / "ok.png", 8, 8), 8, 8, "webp") == [
        "unsupported expected format 'webp'"
    ]


def test_the_validate_cli_exits_non_zero_on_a_bad_image(tmp_path, capsys):
    good = write_png(tmp_path / "ok.png", 64, 48)
    assert validate_cli.main([str(good), "--width", "64", "--height", "48"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert validate_cli.main([str(good), "--width", "128", "--height", "48"]) == 1
    assert json.loads(capsys.readouterr().out)["problems"]


# --- the controller's own check after the download ------------------------------------------------


def collect_with(tmp_path, image_bytes):
    """A worker reported COMPLETED; the file the controller then downloads is image_bytes."""
    config = make_config(tmp_path)
    store = z.JobStore(z.output_root(config))
    job = store.add(z.build_job(config, prompt="a ceramic mug", width=512, height=512, job_id="zimg-v1"))
    store.update(job["job_id"], status=z.QUEUED)
    transport = FakeTransport(tmp_path / "vm")
    outbox = transport.vm(f"/content/zimg/outbox/{job['job_id']}")
    outbox.mkdir(parents=True)
    (outbox / "result.png").write_bytes(image_bytes)
    (outbox / "metadata.json").write_text(json.dumps({"validation": {"ok": True, "problems": []}, "gpu": "NVIDIA L4"}))
    controller = z.Controller(config, transport=transport, store=store, echo=lambda line: None)
    controller.session = "zimg-test"
    controller.sync_jobs({"jobs": {job["job_id"]: {"status": z.COMPLETED, "generation_seconds": 3.2}}})
    return config, store.get(job["job_id"])


def test_a_downloaded_image_is_validated_again_locally(tmp_path):
    config, job = collect_with(tmp_path, png_bytes(512, 512))
    assert job["status"] == z.COMPLETED
    assert job["validation"]["local"] == {"ok": True, "problems": [], "bytes": len(png_bytes(512, 512))}
    assert [h["status"] for h in job["history"]][-3:] == [z.VALIDATING, z.VALID, z.COMPLETED]
    out = z.output_root(config) / job["job_id"]
    assert (out / "result.png").is_file() and (out / "workflow.json").is_file()
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    for key in (
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "steps",
        "model",
        "workflow",
        "generation_seconds",
    ):
        assert key in meta
    assert "graph" not in meta  # the graph is workflow.json; metadata stays readable


def test_a_download_that_arrived_broken_is_invalid_output_not_success(tmp_path):
    _, job = collect_with(tmp_path, png_bytes(512, 512)[:-30])
    assert job["status"] == z.INVALID_OUTPUT
    assert job["error"]["code"] == "INVALID_OUTPUT"
    assert job["validation"]["remote"]["ok"] is True and job["validation"]["local"]["ok"] is False


def test_a_download_of_the_wrong_size_is_invalid_output(tmp_path):
    _, job = collect_with(tmp_path, png_bytes(256, 256))
    assert job["status"] == z.INVALID_OUTPUT
    assert "resolution 256x256, expected 512x512" in job["error"]["error_message"]
