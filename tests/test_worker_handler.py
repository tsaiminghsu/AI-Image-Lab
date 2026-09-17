"""The RunPod worker's job parsing, dispatch and S3 hand-off - offline, without boto3 or runpod.

worker/ is not on the test path globally ("handler" and "jobs" are generic module names), so it is
added here only for the duration of the import.
"""

import base64
import importlib
import os
import sys

import pytest

import comfyui_client as client
import generate_character as gc

WORKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker")
# A real 1x1 PNG, so decode_reference's PIL verify passes too when the suite runs in a venv that has
# Pillow (.venv-dev doesn't; the magic-byte check covers that case).
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _import_worker(name):
    sys.path.insert(0, WORKER_DIR)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(WORKER_DIR)


jobs = _import_worker("jobs")
handler_mod = _import_worker("handler")


def test_worker_modules_do_not_import_boto3_or_runpod():
    assert "boto3" not in sys.modules
    assert "runpod" not in sys.modules


def parse(**inp):
    return jobs.parse_job(inp, worker_checkpoint="cyberrealistic_pony")


# --- parse_job --------------------------------------------------------------------------------


def test_missing_job_type_keeps_the_original_image_job():
    spec = parse(prompt="portrait", aspectRatio="1:1")
    assert spec["job_type"] == "image_hq"
    assert (spec["width"], spec["height"]) == (1024, 1024)


def test_unknown_job_type_is_rejected():
    with pytest.raises(jobs.JobInputError, match="unknown jobType"):
        parse(prompt="p", jobType="video_hunyuan")


@pytest.mark.parametrize("tier", [None, "", "explicit", "SAFE"])
def test_unknown_tier_falls_back_to_safe(tier):
    assert parse(prompt="p", tier=tier)["tier"] == "safe"


def test_prompt_and_negative_limits():
    with pytest.raises(jobs.JobInputError, match="missing prompt"):
        parse(prompt="   ")
    with pytest.raises(jobs.JobInputError, match="prompt too long"):
        parse(prompt="x" * (gc.MAX_PROMPT_CHARS + 1))
    with pytest.raises(jobs.JobInputError, match="negativePrompt too long"):
        parse(prompt="p", negativePrompt="x" * (jobs.MAX_EXTRA_NEGATIVE_CHARS + 1))


def test_unknown_character_is_rejected():
    with pytest.raises(gc.UsageError, match="unknown character"):
        parse(prompt="p", characterId="nobody")


def test_wan_defaults():
    spec = parse(prompt="p", jobType="video_wan_i2v")
    assert (spec["width"], spec["height"], spec["frames"], spec["fps"], spec["cfg"]) == (1280, 704, 121, 24, 5.0)
    assert spec["model"] == client.WAN_DEFAULT_MODEL


@pytest.mark.parametrize(
    "override,message",
    [
        ({"cfg": 1.0}, "cfg"),
        ({"frames": 100}, "4n\\+1"),
        ({"width": 1000}, "multiple of 32"),
        ({"width": "1280"}, "integer"),
        ({"cfg": True}, "number"),
        ({"model": "nope"}, "unknown wan model"),
    ],
)
def test_wan_bounds(override, message):
    with pytest.raises(jobs.JobInputError, match=message):
        parse(prompt="p", jobType="video_wan_i2v", **override)


def test_animatediff_defaults_and_bounds():
    spec = parse(prompt="p", jobType="video_animatediff")
    assert spec["checkpoint"] == "realistic_vision"
    assert spec["upscale_to"] == 1024 and spec["hires"] is True
    for override, message in [
        ({"frames": 24}, "frames"),
        ({"width": 1024}, "width"),
        ({"width": 500}, "multiples of 8"),
        ({"upscaleTo": 2048}, "upscaleTo"),
        ({"interp": 3}, "interp"),
        ({"hires": "yes"}, "true or false"),
        ({"checkpoint": "cyberrealistic_pony"}, "unknown checkpoint"),
    ]:
        with pytest.raises(jobs.JobInputError, match=message):
            parse(prompt="p", jobType="video_animatediff", **override)


# --- decode_reference -------------------------------------------------------------------------


def test_reference_must_be_png_or_jpeg(tmp_path):
    with pytest.raises(jobs.JobInputError, match="valid base64"):
        jobs.decode_reference("not base64!!", "j")
    with pytest.raises(jobs.JobInputError, match="PNG or JPEG"):
        jobs.decode_reference(base64.b64encode(b"GIF89a" + b"\x00" * 20).decode(), "j")
    with pytest.raises(jobs.JobInputError, match="too large"):
        jobs.decode_reference(base64.b64encode(PNG_BYTES).decode(), "j", max_bytes=10)


# --- run_job dispatch -------------------------------------------------------------------------


@pytest.fixture
def fake_gen(monkeypatch, tmp_path):
    calls = {}

    def fake_wan(prompt, extra_negative, tier, trigger, first_frame_path, out_dir, seed, **kw):
        calls["wan"] = dict(
            prompt=prompt, extra_negative=extra_negative, tier=tier, trigger=trigger, first_frame=first_frame_path, **kw
        )
        # The real function composes the negatives; prove the worker routes through it by
        # composing exactly as it does.
        calls["wan"]["negative"] = gc.build_wan_prompts(prompt, extra_negative, tier, trigger)[1]
        path = os.path.join(out_dir, "wan.mp4")
        open(path, "wb").close()
        return path

    def fake_ad(prompt, extra_negative, tier, trigger, face_ref_path, out_dir, seed, **kw):
        calls["ad"] = dict(face_ref=face_ref_path, **kw)
        path = os.path.join(out_dir, "ad.mp4")
        open(path, "wb").close()
        return path

    monkeypatch.setattr(gc, "gen_video_wan_i2v", fake_wan)
    monkeypatch.setattr(gc, "gen_video_animatediff", fake_ad)
    return calls


def test_wan_job_requires_a_first_frame(tmp_path, fake_gen):
    spec = parse(prompt="p", jobType="video_wan_i2v")
    with pytest.raises(jobs.JobInputError, match="no first frame"):
        jobs.run_job(spec, job_id="j", anchor_dir=str(tmp_path), out_dir=str(tmp_path), tmp_files=[])


def test_wan_job_uses_the_uploaded_frame_and_reports_mp4(tmp_path, fake_gen):
    spec = parse(
        prompt="waves",
        jobType="video_wan_i2v",
        referenceImageBase64=base64.b64encode(PNG_BYTES).decode(),
        tier="suggestive",
        negativePrompt="hats",
    )
    tmp_files = []
    result = jobs.run_job(spec, job_id="j", anchor_dir=str(tmp_path), out_dir=str(tmp_path), tmp_files=tmp_files)
    assert result["contentType"] == "video/mp4" and result["ext"] == "mp4"
    assert fake_gen["wan"]["first_frame"] in tmp_files
    assert fake_gen["wan"]["cfg"] == client.WAN_CFG
    assert gc.AGE_SAFETY_NEGATIVE in fake_gen["wan"]["negative"]
    assert "hats" in fake_gen["wan"]["negative"]
    for p in tmp_files:
        if os.path.exists(p):
            os.remove(p)


def test_animatediff_job_falls_back_to_the_volume_anchor_and_lifts_the_hires_cap(tmp_path, fake_gen):
    anchor = tmp_path / "mei" / "anchor_seed1.png"
    anchor.parent.mkdir()
    anchor.write_bytes(PNG_BYTES)
    spec = parse(prompt="p", jobType="video_animatediff", characterId="mei")
    jobs.run_job(spec, job_id="j", anchor_dir=str(tmp_path), out_dir=str(tmp_path), tmp_files=[])
    assert fake_gen["ad"]["face_ref"] == str(anchor)
    assert fake_gen["ad"]["hires_max_pixels"] == jobs.WORKER_AD_HIRES_MAX_PIXELS
    assert jobs.WORKER_AD_HIRES_MAX_PIXELS > client.ANIMATEDIFF_HIRES_MAX_PIXELS


# --- handler: S3 hand-off ---------------------------------------------------------------------


class FakeS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.uploads.append((os.path.basename(path), bucket, key, ExtraArgs))

    def generate_presigned_url(self, op, Params=None, ExpiresIn=None):
        return f"https://signed/{Params['Key']}?exp={ExpiresIn}"


def test_handler_uploads_video_with_mp4_content_type(monkeypatch, tmp_path, fake_gen):
    monkeypatch.setattr(handler_mod, "S3_BUCKET", "bucket")
    s3 = FakeS3()
    job = {
        "id": "job123",
        "input": {
            "jobType": "video_wan_i2v",
            "prompt": "p",
            "referenceImageBase64": base64.b64encode(PNG_BYTES).decode(),
        },
    }
    out = handler_mod.handler(job, s3_factory=lambda: s3)
    assert "error" not in out, out
    assert out["outputKey"] == "generated/job123.mp4"
    assert s3.uploads[0][3] == {"ContentType": "video/mp4"}
    assert out["outputUrl"].startswith("https://signed/generated/job123.mp4")


def test_handler_returns_input_errors_verbatim(monkeypatch):
    monkeypatch.setattr(handler_mod, "S3_BUCKET", "bucket")
    out = handler_mod.handler({"id": "j", "input": {"prompt": ""}}, s3_factory=FakeS3)
    assert out == {"error": "missing prompt"}


def test_handler_requires_a_bucket(monkeypatch):
    monkeypatch.setattr(handler_mod, "S3_BUCKET", None)
    assert "S3_BUCKET" in handler_mod.handler({"id": "j", "input": {"prompt": "p"}})["error"]


def test_make_s3_passes_the_endpoint_url(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, **kw):
            captured["config"] = kw

    fake_boto3 = type(sys)("boto3")
    fake_boto3.client = lambda *a, **kw: captured.update(args=a, kwargs=kw) or "client"
    fake_botocore = type(sys)("botocore")
    fake_botocore_config = type(sys)("botocore.config")
    fake_botocore_config.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    assert handler_mod.make_s3() == "client"
    assert captured["kwargs"]["endpoint_url"] == "https://acct.r2.cloudflarestorage.com"
    assert captured["config"] == {"signature_version": "s3v4"}
