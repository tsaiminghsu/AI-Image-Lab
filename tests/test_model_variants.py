"""Coverage for the full/quant checkpoint-variant machinery in comfyui_client.py
(chosen_variant / load+save_variant_settings / apply_model_variants / effective_checkpoint /
variant_preflight / _notice_once). The important behaviour under test is the rule that keeps
ControlNet control-lora workflows from silently producing all-black images: ControlLora.pre_run
copies the base UNet state_dict into the control model, and for a per-layer-quantized Linear that
grabs the raw fp8 payload without its weight_scale, corrupting pose conditioning - so quantization
must be force-disabled whenever a control-lora is present, unless the caller opts out via
MODEL_VARIANT_ALLOW_CONTROLLORA=1.
"""

import json
import os

import pytest

import comfyui_client as client
from helpers import FakeComfy

FULL_JUGGERNAUT = client.CHECKPOINTS["juggernaut"]
QUANT_JUGGERNAUT = client.MODEL_VARIANTS["juggernaut"]["quant"]


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"fake-weights")


# -- chosen_variant precedence ---------------------------------------------------------------


def test_default_is_full_with_no_settings_env_or_override(isolated_variants):
    assert client.chosen_variant("juggernaut") == "full"


def test_settings_default_quant(isolated_variants):
    client.save_variant_settings(default="quant")
    assert client.chosen_variant("juggernaut") == "quant"


def test_per_model_setting_beats_settings_default(isolated_variants):
    client.save_variant_settings(default="quant", models={"juggernaut": "full"})
    assert client.chosen_variant("juggernaut") == "full"
    client.save_variant_settings(default="full", models={"juggernaut": "quant"})
    assert client.chosen_variant("juggernaut") == "quant"


def test_env_beats_settings(isolated_variants, monkeypatch):
    client.save_variant_settings(default="full")
    monkeypatch.setenv("MODEL_VARIANT", "quant")
    assert client.chosen_variant("juggernaut") == "quant"


def test_cli_override_beats_env(isolated_variants, monkeypatch):
    monkeypatch.setenv("MODEL_VARIANT", "quant")
    client.set_variant_override("full")
    try:
        assert client.chosen_variant("juggernaut") == "full"
    finally:
        client.set_variant_override(None)


def test_garbage_env_is_ignored_and_falls_through(isolated_variants, monkeypatch):
    monkeypatch.setenv("MODEL_VARIANT", "fp8")  # not a real VARIANT_CHOICES value
    assert client.chosen_variant("juggernaut") == "full"


def test_unknown_model_key_is_always_full(isolated_variants):
    client.save_variant_settings(default="quant")
    assert client.chosen_variant("not_a_real_model") == "full"


def test_quant_unsupported_key_is_always_full(isolated_variants, monkeypatch):
    monkeypatch.setattr(client, "QUANT_UNSUPPORTED", {"juggernaut": "NaN on this GPU"})
    client.save_variant_settings(default="quant")
    assert client.chosen_variant("juggernaut") == "full"


def test_set_variant_override_rejects_bad_value(isolated_variants):
    with pytest.raises(ValueError):
        client.set_variant_override("fp8")


def test_set_variant_override_none_clears_it(isolated_variants):
    client.set_variant_override("quant")
    client.set_variant_override(None)
    assert client._variant_override is None
    assert client.chosen_variant("juggernaut") == "full"


# -- settings file load/save -------------------------------------------------------------------


def test_load_missing_settings_file_returns_default(isolated_variants):
    assert client.load_variant_settings() == {"version": 1, "default": "full", "models": {}}


@pytest.mark.parametrize("bad_content", ["{not valid json", "[1, 2, 3]"])
def test_load_malformed_settings_returns_default_and_notices(isolated_variants, bad_content):
    _touch_text(client.MODEL_VARIANTS_FILE, bad_content)
    result = client.load_variant_settings()
    assert result == {"version": 1, "default": "full", "models": {}}
    assert len(client.LAST_VARIANT_NOTICES) == 1
    assert "讀不懂" in client.LAST_VARIANT_NOTICES[0]


def _touch_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_load_filters_unknown_keys_and_invalid_values(isolated_variants):
    _touch_text(
        client.MODEL_VARIANTS_FILE,
        json.dumps(
            {
                "version": 1,
                "default": "not_a_real_choice",
                "models": {"juggernaut": "quant", "not_a_real_model": "quant", "pony": "bogus_value"},
            }
        ),
    )
    result = client.load_variant_settings()
    assert result["default"] == "full"  # invalid default falls back
    assert result["models"] == {"juggernaut": "quant"}  # unknown key and bad value dropped


def test_save_variant_settings_round_trips(isolated_variants):
    client.save_variant_settings(models={"juggernaut": "quant"}, default="quant")
    assert client.load_variant_settings() == {
        "version": 1,
        "default": "quant",
        "models": {"juggernaut": "quant"},
    }
    assert not os.path.exists(client.MODEL_VARIANTS_FILE + ".tmp")


def test_save_variant_settings_writes_lf_terminated_json(isolated_variants):
    client.save_variant_settings(models={"juggernaut": "quant"})
    with open(client.MODEL_VARIANTS_FILE, "rb") as f:
        raw = f.read()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


def test_save_variant_settings_rejects_bad_model_value(isolated_variants):
    with pytest.raises(ValueError):
        client.save_variant_settings(models={"juggernaut": "fp8"})


def test_save_variant_settings_rejects_bad_default(isolated_variants):
    with pytest.raises(ValueError):
        client.save_variant_settings(default="fp8")


# -- _uses_control_lora ---------------------------------------------------------------------


def test_uses_control_lora_true_for_control_lora_filename():
    wf = {
        "22": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "control-lora-openposeXL2-rank256.safetensors"},
        }
    }
    assert client._uses_control_lora(wf) is True


def test_uses_control_lora_is_case_insensitive():
    wf = {
        "22": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "CONTROL-LORA-OpenPoseXL2.safetensors"}}
    }
    assert client._uses_control_lora(wf) is True


def test_uses_control_lora_false_for_plain_controlnet():
    wf = {
        "22": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "controlnet-openpose-sdxl.safetensors"}}
    }
    assert client._uses_control_lora(wf) is False


def test_uses_control_lora_true_for_diff_control_net_loader():
    wf = {"22": {"class_type": "DiffControlNetLoader", "inputs": {"control_net_name": "control-lora-foo.safetensors"}}}
    assert client._uses_control_lora(wf) is True


def test_uses_control_lora_false_with_no_loader_node():
    wf = {"3": {"class_type": "KSampler", "inputs": {}}}
    assert client._uses_control_lora(wf) is False


# -- end-to-end through the real _submit_and_wait / apply_model_variants ------------------------


def test_control_lora_forces_full_checkpoint_end_to_end(isolated_variants, fake_comfy):
    """The all-black-image bug this whole module exists to prevent: with quant selected and the
    quant file present, a control-lora workflow (submit_generation_with_pose's node 22) must still
    load the FULL checkpoint, and a notice must explain why."""
    _touch(os.path.join(str(isolated_variants), QUANT_JUGGERNAUT))
    client.save_variant_settings(models={"juggernaut": "quant"})
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    client.submit_generation_with_pose(
        prompt="p",
        negative_prompt="n",
        seed=1,
        ip_adapter_image_filename="face.png",
        pose_image_filename="pose.png",
        filename_prefix="test",
        checkpoint=FULL_JUGGERNAUT,
    )

    posted = fake_comfy.posted_prompt()
    assert posted["4"]["inputs"]["ckpt_name"] == FULL_JUGGERNAUT
    assert any("control-lora" in n for n in client.LAST_VARIANT_NOTICES)


def test_control_lora_allow_env_switches_to_quant(isolated_variants, fake_comfy, monkeypatch):
    _touch(os.path.join(str(isolated_variants), QUANT_JUGGERNAUT))
    client.save_variant_settings(models={"juggernaut": "quant"})
    monkeypatch.setenv("MODEL_VARIANT_ALLOW_CONTROLLORA", "1")
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    client.submit_generation_with_pose(
        prompt="p",
        negative_prompt="n",
        seed=1,
        ip_adapter_image_filename="face.png",
        pose_image_filename="pose.png",
        filename_prefix="test",
        checkpoint=FULL_JUGGERNAUT,
    )

    posted = fake_comfy.posted_prompt()
    assert posted["4"]["inputs"]["ckpt_name"] == QUANT_JUGGERNAUT


def test_hq_without_pose_uses_quant(isolated_variants, fake_comfy):
    """submit_generation_hq drops nodes 20-23 entirely when no pose is given, so no control-lora
    is present in the posted graph and quant should be used normally."""
    _touch(os.path.join(str(isolated_variants), QUANT_JUGGERNAUT))
    client.save_variant_settings(models={"juggernaut": "quant"})
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    client.submit_generation_hq(
        prompt="p",
        negative_prompt="n",
        seed=1,
        filename_prefix="test",
        checkpoint=FULL_JUGGERNAUT,
    )

    posted = fake_comfy.posted_prompt()
    assert posted["4"]["inputs"]["ckpt_name"] == QUANT_JUGGERNAUT


def test_hq_with_pose_forces_full(isolated_variants, fake_comfy):
    _touch(os.path.join(str(isolated_variants), QUANT_JUGGERNAUT))
    client.save_variant_settings(models={"juggernaut": "quant"})
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    client.submit_generation_hq(
        prompt="p",
        negative_prompt="n",
        seed=1,
        filename_prefix="test",
        checkpoint=FULL_JUGGERNAUT,
        pose_image_filename="pose.png",
    )

    posted = fake_comfy.posted_prompt()
    assert posted["4"]["inputs"]["ckpt_name"] == FULL_JUGGERNAUT
    assert any("control-lora" in n for n in client.LAST_VARIANT_NOTICES)


def test_quant_chosen_but_file_absent_falls_back_to_full(isolated_variants, fake_comfy):
    """isolated_variants leaves the fake checkpoints dir empty - the quant file is never touched
    into it, simulating a model that hasn't been converted yet."""
    client.save_variant_settings(models={"juggernaut": "quant"})
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    client.submit_generation_hq(
        prompt="p",
        negative_prompt="n",
        seed=1,
        filename_prefix="test",
        checkpoint=FULL_JUGGERNAUT,
    )

    posted = fake_comfy.posted_prompt()
    assert posted["4"]["inputs"]["ckpt_name"] == FULL_JUGGERNAUT
    assert any(client.convert_command("juggernaut") in n for n in client.LAST_VARIANT_NOTICES)


def test_loader_options_take_priority_over_filesystem(isolated_variants, fake_comfy, monkeypatch):
    """When the server can answer (object_info gives an explicit file list), that beats the
    filesystem check - here the quant file does NOT exist on disk, but the server claims it does."""
    monkeypatch.setattr(client, "_loader_options", lambda node, field: [QUANT_JUGGERNAUT, FULL_JUGGERNAUT])
    client.save_variant_settings(models={"juggernaut": "quant"})
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]

    client.submit_generation_hq(
        prompt="p",
        negative_prompt="n",
        seed=1,
        filename_prefix="test",
        checkpoint=FULL_JUGGERNAUT,
    )

    posted = fake_comfy.posted_prompt()
    assert posted["4"]["inputs"]["ckpt_name"] == QUANT_JUGGERNAUT


# -- effective_checkpoint / variant_preflight agree with apply_model_variants -------------------


def _apply_to_minimal_wf(full_name, uses_pose):
    wf = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": full_name}}}
    if uses_pose:
        wf["22"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "control-lora-openposeXL2-rank256.safetensors"},
        }
    client.apply_model_variants(wf)
    return wf["4"]["inputs"]["ckpt_name"]


@pytest.mark.parametrize("quant_setting", ["full", "quant"])
@pytest.mark.parametrize("uses_pose", [False, True])
@pytest.mark.parametrize("file_present", [False, True])
def test_effective_checkpoint_and_preflight_agree_with_apply(isolated_variants, quant_setting, uses_pose, file_present):
    client.save_variant_settings(models={"juggernaut": quant_setting})
    if file_present:
        _touch(os.path.join(str(isolated_variants), QUANT_JUGGERNAUT))

    actual = _apply_to_minimal_wf(FULL_JUGGERNAUT, uses_pose)
    effective = client.effective_checkpoint(FULL_JUGGERNAUT, uses_pose=uses_pose)
    preflight = client.variant_preflight("juggernaut", uses_pose=uses_pose)

    assert effective == actual

    if quant_setting == "full":
        assert effective == FULL_JUGGERNAUT
        assert preflight == []
    elif uses_pose:
        assert effective == FULL_JUGGERNAUT
        assert preflight and "ControlNet" in preflight[0]
    elif not file_present:
        assert effective == FULL_JUGGERNAUT
        assert preflight and client.convert_command("juggernaut") in preflight[0]
    else:
        assert effective == QUANT_JUGGERNAUT
        assert preflight and "量化版" in preflight[0]


def test_sd15_checkpoints_are_never_swapped(isolated_variants):
    sd15_full = client.CHECKPOINTS["realistic_vision"]
    client.save_variant_settings(default="quant")
    _touch(os.path.join(str(isolated_variants), sd15_full))  # irrelevant, but prove presence isn't the reason
    assert client.effective_checkpoint(sd15_full) == sd15_full
    assert client.effective_checkpoint(sd15_full, uses_pose=True) == sd15_full


# -- _notice_once dedup ---------------------------------------------------------------------


def test_notice_once_dedups_the_print_but_not_the_notices_list(isolated_variants, capsys):
    client._notice_once("hello there")
    capsys.readouterr()  # discard the first print
    client._notice_once("hello there")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert client.LAST_VARIANT_NOTICES == ["hello there", "hello there"]
