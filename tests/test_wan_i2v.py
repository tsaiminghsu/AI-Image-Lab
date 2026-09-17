"""Wan 2.2 image-to-video: the workflow the client builds, the parameter bounds, and the prompt pair.

The Wan path only ever runs on a cloud GPU, so this suite is the main thing standing between a typo
and a billed job that fails minutes into a cold start. Offline: captured_submit replaces
_submit_and_wait, and the model-file preflight is stubbed.
"""

import os

import pytest

import comfyui_client as client
import generate_character as gc

PROMPT = "SENTINEL_WAN_PROMPT"
NEGATIVE = "SENTINEL_WAN_NEGATIVE"


@pytest.fixture
def no_preflight(monkeypatch):
    monkeypatch.setattr(client, "wan_missing_files", lambda model=client.WAN_DEFAULT_MODEL: [])


def _submit(**overrides):
    kwargs = dict(
        prompt=PROMPT, negative_prompt=NEGATIVE, seed=77, start_image_filename="start.png", filename_prefix="wan_test"
    )
    kwargs.update(overrides)
    return client.submit_generation_wan_i2v(**kwargs)


# --- workflow ---------------------------------------------------------------------------------


def test_defaults_match_the_official_ti2v_5b_template(captured_submit, no_preflight):
    _submit()
    wf = captured_submit[0]["wf"]
    files = client.WAN_MODELS["wan22_ti2v_5b"]
    assert wf["1"]["inputs"]["unet_name"] == files["unet"] == "wan2.2_ti2v_5B_fp16.safetensors"
    assert wf["2"]["inputs"]["clip_name"] == files["text_encoder"]
    assert wf["2"]["inputs"]["type"] == "wan"
    assert wf["3"]["inputs"]["vae_name"] == files["vae"]
    assert wf["4"]["inputs"]["shift"] == 8.0
    latent = wf["11"]["inputs"]
    assert (latent["width"], latent["height"], latent["length"]) == (1280, 704, 121)
    assert latent["start_image"] == ["10", 0]
    sampler = wf["12"]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (
        20,
        5.0,
        "uni_pc",
        "simple",
    )
    assert wf["90"]["inputs"]["fps"] == 24.0
    assert wf["9"]["inputs"]["format"] == "mp4"
    assert captured_submit[0]["timeout_seconds"] == client.POLL_TIMEOUT_SECONDS_WAN


def test_prompts_seed_and_start_image_reach_their_nodes(captured_submit, no_preflight):
    _submit()
    wf = captured_submit[0]["wf"]
    assert wf["6"]["inputs"]["text"] == PROMPT
    assert wf["7"]["inputs"]["text"] == NEGATIVE
    assert wf["12"]["inputs"]["negative"] == ["7", 0]
    assert wf["12"]["inputs"]["seed"] == 77
    assert wf["10"]["inputs"]["image"] == "start.png"


@pytest.mark.parametrize("frames,expected", [(121, 121), (120, 117), (81, 81), (82, 81), (3, 5)])
def test_length_is_snapped_to_4n_plus_1(captured_submit, no_preflight, frames, expected):
    _submit(frames=frames)
    assert captured_submit[0]["wf"]["11"]["inputs"]["length"] == expected


def test_size_is_snapped_to_multiples_of_32(captured_submit, no_preflight):
    _submit(width=1000, height=700)
    latent = captured_submit[0]["wf"]["11"]["inputs"]
    assert latent["width"] % 32 == 0 and latent["height"] % 32 == 0


def test_missing_model_files_fail_before_submitting(captured_submit, monkeypatch):
    monkeypatch.setattr(client, "wan_missing_files", lambda model=client.WAN_DEFAULT_MODEL: ["wan2.2_vae.safetensors"])
    with pytest.raises(RuntimeError, match="wan2.2_vae.safetensors"):
        _submit()
    assert captured_submit == []


# --- parameter bounds -------------------------------------------------------------------------

GOOD = dict(width=1280, height=704, frames=121, fps=24, steps=20, cfg=5.0, model="wan22_ti2v_5b")


def test_good_params_pass():
    gc.check_wan_params(**GOOD)


@pytest.mark.parametrize(
    "override,message",
    [
        ({"cfg": 1.0}, "cfg"),
        ({"cfg": 1.49}, "cfg"),
        ({"frames": 120}, "4n\\+1"),
        ({"frames": 125}, "4n\\+1"),
        ({"width": 1000}, "multiple of 32"),
        ({"width": 1312}, "multiple of 32"),
        ({"width": 1280, "height": 1280}, "1280x704"),
        ({"fps": 60}, "fps"),
        ({"steps": 4}, "steps"),
        ({"model": "wan21_nonexistent"}, "unknown wan model"),
    ],
)
def test_bad_params_raise_usage_error(override, message):
    with pytest.raises(gc.UsageError, match=message):
        gc.check_wan_params(**{**GOOD, **override})


def test_cfg_floor_is_the_shared_safety_constant():
    gc.check_wan_params(**{**GOOD, "cfg": client.SAFETY_MIN_CFG})
    with pytest.raises(gc.UsageError):
        gc.check_wan_params(**{**GOOD, "cfg": client.SAFETY_MIN_CFG - 0.01})


# --- prompts ----------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["safe", "suggestive"])
@pytest.mark.parametrize("trigger", [None, "minjun"])
def test_wan_prompts_keep_safety_negatives_and_drop_sd_conventions(tier, trigger):
    positive, negative = gc.build_wan_prompts("turns toward the camera", "", tier, trigger)
    assert gc.AGE_SAFETY_NEGATIVE in negative
    assert gc.WAN_VIDEO_NEGATIVE in negative
    # SD-era style tags and CLIP weight syntax are not for umt5.
    assert gc.REALISTIC_STYLE not in positive
    assert gc.REALISTIC_NEGATIVE not in negative
    assert ":1.3)" not in positive
    assert not positive.startswith(gc.PONY_QUALITY_TAGS)


def test_gen_video_wan_i2v_rejects_before_uploading(monkeypatch, tmp_path):
    uploads = []
    monkeypatch.setattr(client, "upload_reference_image", lambda p: uploads.append(p) or "x.png")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(gc.UsageError, match="cfg"):
        gc.gen_video_wan_i2v("p", "", "safe", None, str(frame), str(tmp_path), 1, cfg=1.0)
    with pytest.raises(gc.UsageError, match="first frame"):
        gc.gen_video_wan_i2v("p", "", "safe", None, str(tmp_path / "missing.png"), str(tmp_path), 1)
    assert uploads == []


def test_gen_video_wan_i2v_passes_the_safe_prompt_pair_to_the_client(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(client, "upload_reference_image", lambda p: "uploaded.png")
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"mp4")

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return str(raw)

    monkeypatch.setattr(client, "submit_generation_wan_i2v", fake_submit)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = gc.gen_video_wan_i2v("waves hello", "", "safe", "mei", str(frame), str(tmp_path / "out"), 42)
    assert os.path.basename(out) == "wan_i2v_seed42.mp4"
    assert gc.AGE_SAFETY_NEGATIVE in captured["negative_prompt"]
    assert "waves hello" in captured["prompt"]
    assert captured["start_image_filename"] == "uploaded.png"
    assert captured["cfg"] >= client.SAFETY_MIN_CFG
