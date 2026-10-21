"""Pony -> Z-Image two-stage picture: Pony decides who and what pose, Z-Image re-renders the material at a low denoise.
Only the graph and the glue are tested here; nothing talks to ComfyUI."""

import os

import pytest

import comfyui_client as client
import generate_character as gc

FILES = client.ZIMAGE_MODELS["z_image_turbo"]


def build(**kw):
    args = dict(
        prompt="p",
        negative_prompt="n",
        seed=7,
        filename_prefix="x",
        files=FILES,
        init_image="in.png",
        width=830,
        height=1210,
    )
    args.update(kw)
    return client.build_zimage_img2img_workflow(**args)


def test_the_graph_starts_from_the_picture_not_from_an_empty_latent():
    wf = build()
    assert wf["20"]["inputs"]["image"] == "in.png"
    assert wf["21"]["inputs"]["pixels"] == ["22", 0]
    assert wf["3"]["inputs"]["latent_image"] == ["21", 0]
    assert "5" not in wf  # no EmptySD3LatentImage: the latent is the encoded picture


def test_the_picture_is_scaled_to_multiples_of_16():
    scale = build(width=830, height=1210)["22"]["inputs"]
    assert (scale["width"], scale["height"]) == (832, 1216)


def test_denoise_is_clamped_and_cfg_keeps_its_safety_floor():
    assert build(denoise=0.0)["3"]["inputs"]["denoise"] == 0.05
    assert build(denoise=3)["3"]["inputs"]["denoise"] == 1.0
    assert build(cfg=1.0)["3"]["inputs"]["cfg"] == client.ZIMAGE_MIN_CFG  # cfg 1.0 would drop the negative prompt


def test_the_negative_prompt_reaches_the_sampler():
    wf = build(negative_prompt="no kids")
    assert wf["7"]["inputs"]["text"] == "no kids"
    assert wf["3"]["inputs"]["negative"] == ["7", 0]


@pytest.fixture
def stages(monkeypatch, tmp_path):
    calls = {}

    def fake_gen_custom(prompt, extra_negative, tier, trigger, anchor_path, out_dir, seed, filename, ip_weight, **kw):
        calls["pony"] = dict(filename=filename, checkpoint=kw.get("checkpoint"))
        (tmp_path / f"{filename}.png").write_bytes(b"png")

    def fake_refine(path, prompt, negative, seed, prefix, denoise, **kw):
        calls["refine"] = dict(path=path, prompt=prompt, negative=negative, denoise=denoise)
        raw = tmp_path / "raw.png"
        raw.write_bytes(b"png")
        return str(raw)

    monkeypatch.setattr(gc, "gen_custom", fake_gen_custom)
    monkeypatch.setattr(client, "zimage_missing_files", lambda model="z_image_turbo": [])
    monkeypatch.setattr(client, "submit_img2img_refine_zimage", fake_refine)
    return calls, tmp_path


def run(tmp_path, **kw):
    args = dict(
        prompt="woman reading",
        extra_negative="",
        tier="safe",
        trigger=None,
        anchor_path=None,
        out_dir=str(tmp_path),
        seed=1,
        filename="s",
        ip_adapter_weight=1.0,
    )
    args.update(kw)
    return gc.gen_pony_zimage(**args)


def test_both_pictures_are_written_and_the_refine_starts_from_the_pony_one(stages):
    calls, tmp_path = stages
    base, final = run(tmp_path)
    assert os.path.basename(base) == "s_pony.png" and os.path.isfile(base)
    assert os.path.basename(final) == "s.png" and os.path.isfile(final)
    assert calls["pony"]["checkpoint"] == gc.PONY_ZIMAGE_DEFAULT_PONY
    assert calls["refine"]["path"] == base
    assert calls["refine"]["denoise"] == client.ZIMAGE_REFINE_DENOISE


def test_the_refine_prompt_carries_the_age_safety_negative(stages):
    calls, tmp_path = stages
    run(tmp_path)
    assert gc.AGE_SAFETY_NEGATIVE.split(",")[0].strip() in calls["refine"]["negative"]


def test_skip_refine_stops_after_the_pony_picture(stages):
    calls, tmp_path = stages
    base, final = run(tmp_path, skip_refine=True)
    assert final is None and "refine" not in calls and os.path.isfile(base)


def test_a_missing_zimage_model_keeps_the_pony_picture_and_says_where_it_is(stages, monkeypatch):
    calls, tmp_path = stages
    monkeypatch.setattr(client, "zimage_missing_files", lambda model="z_image_turbo": ["vae.safetensors"])
    with pytest.raises(gc.UsageError, match="s_pony.png"):
        run(tmp_path)
    assert os.path.isfile(tmp_path / "s_pony.png")


def test_only_pony_checkpoints_and_sane_denoise_are_accepted(stages):
    _, tmp_path = stages
    with pytest.raises(gc.UsageError):
        run(tmp_path, pony_checkpoint="juggernaut")
    with pytest.raises(gc.UsageError):
        run(tmp_path, denoise=0.0)
