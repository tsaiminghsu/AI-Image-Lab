"""Pins the age/content-safety invariants generate_character.py and comfyui_client.py both
promise: MINIMUM_AGE, AGE_SAFETY_NEGATIVE never dropped from any assembled negative prompt, and
the cfg floors that exist purely so ComfyUI never silently skips the negative prompt (and with it
the safety negatives). These are the highest-value tests in the project: an upcoming refactor
de-duplicates gen_video_animatediff's copy-pasted prompt assembly (see test_prompt_assembly.py's
test_animatediff_inline_assembly_matches_shared_builder) into a single call to
_build_prompt_and_negative, and this file is what proves that refactor didn't quietly loosen a
safety guarantee.
"""

import itertools

import comfyui_client as client
import generate_character as gc
import pytest


def test_minimum_age_is_18():
    """MINIMUM_AGE lowering is precisely the change this test exists to catch - it must fail red
    the moment someone edits the literal, not just when a character profile violates it."""
    assert gc.MINIMUM_AGE == 18


def test_every_character_meets_minimum_age_and_has_a_lora_key():
    """CHARACTERS is validated at import time (a bad age raises before any entry point can run),
    and the trailing setdefault loop normalises every profile to carry a 'lora' key (None until a
    character LoRA is trained - see _run_hq). Both properties are re-checked here directly, so a
    change to that loop or the age floor is caught even if the import-time raise were ever removed."""
    assert gc.CHARACTERS, "CHARACTERS must not be empty"
    for name, profile in gc.CHARACTERS.items():
        assert profile["age"] >= gc.MINIMUM_AGE, f"{name} is below MINIMUM_AGE"
        assert "lora" in profile, f"{name} profile missing 'lora' key (setdefault loop didn't run?)"


# --- the safety constants must actually SAY something ------------------------------------------
# Every other test in this file asserts `AGE_SAFETY_NEGATIVE in negative`, i.e. that the constant
# PROPAGATES into the assembled prompt. That check is self-referential: it stays green if the
# constant itself is weakened, and it is vacuously true if the constant is emptied, because
# `"" in anything` is True. Measured: deleting "child, " from AGE_SAFETY_NEGATIVE, and even
# setting it to "", left all 628 tests in this suite passing. These two tests close that hole by
# pinning the content of the constants rather than their propagation.

# The terms ComfyUI must always receive as negative conditioning. Anything removed from this set
# is a deliberate weakening of the age-safety floor and has to be argued for here, in the diff.
REQUIRED_AGE_SAFETY_TERMS = frozenset(
    {
        "child",
        "children",
        "kid",
        "minor",
        "teen",
        "teenager",
        "underage",
        "young girl",
    }
)


def test_age_safety_negative_contains_every_required_term():
    """Pins the CONTENT of the constant, not just its propagation. Without this, emptying
    AGE_SAFETY_NEGATIVE passes the whole suite."""
    terms = {t.strip() for t in gc.AGE_SAFETY_NEGATIVE.split(",")}
    missing = REQUIRED_AGE_SAFETY_TERMS - terms
    assert not missing, f"AGE_SAFETY_NEGATIVE no longer blocks: {sorted(missing)}"


@pytest.mark.parametrize(
    "name",
    ["AGE_SAFETY_NEGATIVE", "QUALITY_NEGATIVE", "SAFE_SAFETY_NEGATIVE", "SUGGESTIVE_NEGATIVE", "NEGATIVE_PROMPT"],
)
def test_safety_constant_is_not_blank(name):
    """A blank safety constant makes every `in` assertion in this file vacuously true, so guard
    the emptiness case explicitly for each one."""
    value = getattr(gc, name)
    assert value.strip(), f"{name} is empty - every containment assertion against it is vacuous"
    assert len(value) > 20, f"{name} is suspiciously short ({value!r})"


def test_known_negative_prompt_constants_inventory():
    """Inventory guard for the classification below: every module-level string constant whose name
    contains NEGATIVE must be one of these two known groups. If someone adds a new one, this test
    fails until it's explicitly sorted into "safety-tier" (must carry AGE_SAFETY_NEGATIVE, checked
    below) or "style/quality" (a deliberate dial that must NOT be assumed to carry it)."""
    all_negative_constants = {
        name for name, value in vars(gc).items() if name.isupper() and "NEGATIVE" in name and isinstance(value, str)
    }
    safety_tier = {"AGE_SAFETY_NEGATIVE", "SAFE_SAFETY_NEGATIVE", "SUGGESTIVE_NEGATIVE", "NEGATIVE_PROMPT"}
    style_or_quality = {
        "REALISTIC_NEGATIVE",
        "VIDEO_REALISTIC_NEGATIVE",
        "QUALITY_NEGATIVE",
        "PONY_QUALITY_NEGATIVE_TAGS",
    }
    assert all_negative_constants == safety_tier | style_or_quality


@pytest.mark.parametrize("name", ["SAFE_SAFETY_NEGATIVE", "SUGGESTIVE_NEGATIVE", "NEGATIVE_PROMPT"])
def test_age_safety_negative_is_substring_of_every_safety_tier_constant(name):
    """AGE_SAFETY_NEGATIVE is "always included no matter the content tier" per its own comment
    (generate_character.py:67-69) - assert that promise against the actual composed strings rather
    than trusting the comment. REALISTIC_NEGATIVE/VIDEO_REALISTIC_NEGATIVE/QUALITY_NEGATIVE/
    PONY_QUALITY_NEGATIVE_TAGS are excluded on purpose: they are style/quality dials
    (_build_prompt_and_negative's docstring), never the safety negative itself, and never contain it."""
    value = getattr(gc, name)
    assert gc.AGE_SAFETY_NEGATIVE in value


@pytest.mark.parametrize("name", ["SAFE_SAFETY_NEGATIVE", "SUGGESTIVE_NEGATIVE", "NEGATIVE_PROMPT"])
def test_quality_negative_is_substring_of_every_safety_tier_constant(name):
    """Same guarantee as above for QUALITY_NEGATIVE (the always-on "no second person in frame"
    floor) - it is baked into both SAFE_SAFETY_NEGATIVE and SUGGESTIVE_NEGATIVE, and therefore into
    NEGATIVE_PROMPT transitively."""
    value = getattr(gc, name)
    assert gc.QUALITY_NEGATIVE in value


def test_video_realistic_negative_drops_only_the_symmetry_term():
    """generate_character.py:48 has a bare module-level `assert "symmetrical" not in
    VIDEO_REALISTIC_NEGATIVE` - `python -O` strips module-level asserts entirely, so today nothing
    actually guarantees this outside of manually running without -O. Turn it into a real test, and
    also pin that the *only* thing removed from REALISTIC_NEGATIVE is "symmetrical face, " (not,
    say, an unrelated safety term getting caught by a sloppier replace)."""
    assert "symmetrical" not in gc.VIDEO_REALISTIC_NEGATIVE
    assert gc.VIDEO_REALISTIC_NEGATIVE == gc.REALISTIC_NEGATIVE.replace("symmetrical face, ", "")


# --- AGE_SAFETY_NEGATIVE survives every _build_prompt_and_negative call shape --------------------

_TIERS = ["safe", "suggestive"]
_STYLE_NEGATIVES = [None, "", "custom neg"]
_EXTRA_NEGATIVES = [None, "", "extra"]
_CHECKPOINTS = [None, "juggernaut", "cyberrealistic_pony", "realistic_vision"]
_TRIGGERS = [None, "mei"]


@pytest.mark.parametrize(
    "tier,style_negative,extra_negative,checkpoint,trigger",
    list(itertools.product(_TIERS, _STYLE_NEGATIVES, _EXTRA_NEGATIVES, _CHECKPOINTS, _TRIGGERS)),
)
def test_age_safety_negative_survives_every_assembly_path(tier, style_negative, extra_negative, checkpoint, trigger):
    """The GUI exposes style_negative as an editable text field, so style_negative="" (user cleared
    the field) is a real, reachable input - and it takes a different branch than the style_negative
    is None default (`if style_negative` short-circuits on ""). Cover every tier x style_negative x
    extra_negative x checkpoint-family x trigger combination and assert the mandatory age negative
    is never lost in any of them."""
    _, negative = gc._build_prompt_and_negative(
        "a prompt",
        extra_negative,
        tier,
        trigger,
        None,
        style_negative,
        checkpoint,
    )
    assert gc.AGE_SAFETY_NEGATIVE in negative


# --- cfg floors: they exist ONLY because ComfyUI skips the negative prompt entirely at cfg == 1.0,
# which would silently disable every safety negative above. -------------------------------------


def test_cfg_floor_constants():
    assert client.ZIMAGE_MIN_CFG == client.LCM_MIN_CFG == 1.5
    assert client.ZIMAGE_MIN_CFG > 1.0
    assert client.LCM_MIN_CFG > 1.0
    assert client.ZIMAGE_CFG >= client.ZIMAGE_MIN_CFG


@pytest.mark.parametrize("name", sorted(client.ANIMATEDIFF_LCM_PRESETS))
def test_animatediff_lcm_preset_cfg_meets_the_floor(name):
    assert client.ANIMATEDIFF_LCM_PRESETS[name]["cfg"] >= client.LCM_MIN_CFG


def test_zimage_cfg_is_actually_floored_at_call_time(monkeypatch, captured_submit):
    """The floor constant being >= 1.0 (above) proves nothing on its own - this proves
    submit_txt2img_generation_zimage actually applies max(cfg, ZIMAGE_MIN_CFG) to the sampler node
    it sends, not just that the constant exists. zimage_missing_files is stubbed out so the test
    doesn't depend on a real ComfyUI server being reachable to answer /object_info."""
    monkeypatch.setattr(client, "zimage_missing_files", lambda model="z_image_turbo": [])
    client.submit_txt2img_generation_zimage(
        prompt="p",
        negative_prompt="n",
        seed=1,
        filename_prefix="stem",
        cfg=1.0,
    )
    wf = captured_submit[0]["wf"]
    assert wf["3"]["inputs"]["cfg"] == client.ZIMAGE_MIN_CFG


def test_animatediff_cfg_is_actually_floored_without_lcm_preset(captured_submit):
    """Every sampler in the AnimateDiff graph that carries a cfg - base (3), hires (34), and the
    video face detailer (52) - must all be floored, since any one of them left at cfg 1.0 would
    have ComfyUI silently drop the negative (safety) conditioning for that stage."""
    client.submit_generation_animatediff(
        prompt="p",
        negative_prompt="n",
        seed=1,
        face_ref_image_filename="face.png",
        filename_prefix="stem",
        cfg=1.0,
    )
    wf = captured_submit[0]["wf"]
    for node_id in ("3", "34", "52"):
        assert wf[node_id]["inputs"]["cfg"] >= client.LCM_MIN_CFG
        assert wf[node_id]["inputs"]["cfg"] != 1.0


@pytest.mark.parametrize("lcm_preset", sorted(client.ANIMATEDIFF_LCM_PRESETS))
def test_animatediff_cfg_is_actually_floored_with_lcm_preset(captured_submit, lcm_preset):
    """Same as above, but through the LCM branch, which overwrites cfg from the preset BEFORE the
    `cfg = max(cfg, LCM_MIN_CFG)` floor is applied (comfyui_client.py:837-852) - confirms the floor
    still holds even though the caller's cfg=1.0 is discarded before the floor line runs."""
    client.submit_generation_animatediff(
        prompt="p",
        negative_prompt="n",
        seed=1,
        face_ref_image_filename="face.png",
        filename_prefix="stem",
        cfg=1.0,
        lcm_preset=lcm_preset,
    )
    wf = captured_submit[0]["wf"]
    for node_id in ("3", "34", "52"):
        assert wf[node_id]["inputs"]["cfg"] >= client.LCM_MIN_CFG


# --- enforce_min_cfg: the choke-point guard -----------------------------------------------------
# The per-path max(cfg, FLOOR) calls in submit_txt2img_generation_zimage and
# submit_generation_animatediff only cover the paths that remember to make them. enforce_min_cfg
# runs inside _submit_and_wait, which every generation goes through, so a path added later cannot
# hand ComfyUI a cfg of 1.0 - which would make it drop the negative prompt, and the age-safety
# terms with it - just by forgetting the floor.


def test_all_templates_already_satisfy_the_floor():
    """No shipped template needs correcting today; the guard is a net, not a fix."""
    import glob
    import json
    import os

    templates = glob.glob(os.path.join(os.path.dirname(client.__file__), "workflow_template*.json"))
    assert templates
    for path in templates:
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        assert client.enforce_min_cfg(wf) == [], f"{os.path.basename(path)} ships a cfg below the floor"


def test_enforce_min_cfg_raises_and_reports_every_offending_node():
    wf = {
        "3": {"class_type": "KSampler", "inputs": {"cfg": 1.0}},
        "34": {"class_type": "KSampler", "inputs": {"cfg": 7.0}},
        "41": {"class_type": "FaceDetailer", "inputs": {"cfg": 0.5}},
    }
    assert sorted(client.enforce_min_cfg(wf)) == ["3", "41"]
    assert wf["3"]["inputs"]["cfg"] == client.SAFETY_MIN_CFG
    assert wf["41"]["inputs"]["cfg"] == client.SAFETY_MIN_CFG
    assert wf["34"]["inputs"]["cfg"] == 7.0, "a cfg already above the floor must be left alone"


def test_enforce_min_cfg_ignores_non_numeric_and_missing_cfg():
    """Node inputs can be edge references (["6", 0]) or absent; neither is a cfg to floor."""
    wf = {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"cfg": ["3", 0]}},
        "10": {"class_type": "Weird", "inputs": {"cfg": True}},
    }
    assert client.enforce_min_cfg(wf) == []


def test_submit_and_wait_floors_cfg_before_posting(fake_comfy):
    """The guard has to run on the way out, not just be available: a caller that sneaks a
    cfg of 1.0 into a workflow must still have it corrected in the prompt ComfyUI receives."""
    fake_comfy.history_sequence = [fake_comfy.done()]
    wf = {
        "3": {"class_type": "KSampler", "inputs": {"cfg": 1.0}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }
    client._submit_and_wait(wf, timeout_seconds=5)
    assert fake_comfy.posted_prompt()["3"]["inputs"]["cfg"] == client.SAFETY_MIN_CFG
