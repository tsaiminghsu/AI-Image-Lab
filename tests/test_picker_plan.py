"""plan_picker's per-checkpoint translation, which is the whole point of the picker tab.

The same three clicks - a character, a pose, a scene - have to mean different things depending
on the checkpoint: SDXL/Pony take the pose as a ControlNet skeleton and the character as a
FaceID anchor, while Z-Image and SD1.5 have neither adapter and can only be told in words. Get
that wrong in the quiet direction and the generation still succeeds, just without the pose or
the face the user picked - which is exactly the failure the tab exists to prevent, so it is
pinned here rather than left to a manual click-through.

Offline: reads the pose and scene libraries off disk, never talks to ComfyUI.
"""

import comfyui_client as client
import generate_character as gc
import pose_skeletons
import pytest
import scene_library

POSE = "hands_on_hips"
SCENE = "cozy_cafe"
CHARACTER = "mei"

SDXL_CHECKPOINTS = [None, "juggernaut", "cyberrealistic_pony"]
TEXT_ONLY_CHECKPOINTS = ["realistic_vision", "cyberrealistic", "z_image_turbo"]


def plan(**overrides):
    return gc.plan_picker(
        **{
            "character": CHARACTER,
            "pose_slug": POSE,
            "scene_slug": SCENE,
            "checkpoint": None,
            "tier": "safe",
            "extra_text": None,
            **overrides,
        }
    )


# --- SDXL/Pony: the picks are honoured structurally ---------------------------------------------


@pytest.mark.parametrize("checkpoint", SDXL_CHECKPOINTS)
def test_sdxl_gets_a_real_anchor_and_skeleton(checkpoint):
    p = plan(checkpoint=checkpoint)
    assert p.mode == "controlnet_faceid"
    assert p.pose_name == POSE, "the skeleton goes to ControlNet, not into the text"
    assert p.anchor_path and p.anchor_path.endswith(".png")
    assert not p.notices, "nothing was degraded, so there is nothing to warn about"


@pytest.mark.parametrize("checkpoint", SDXL_CHECKPOINTS)
def test_sdxl_does_not_duplicate_the_pose_hint_into_the_body(checkpoint):
    """_plan_custom already appends the hint when pose_name is set (generate_character.py:711).
    Adding it here too would send it twice - visible in the preview, and double-weighted to the
    sampler."""
    hint = pose_skeletons.load_meta(POSE)["prompt_hint"]
    p = plan(checkpoint=checkpoint)
    assert hint not in p.prompt_body
    assert p.preview_prompt.count(hint) == 1, "the hint must reach the model exactly once"


def test_the_anchor_is_an_identity_reference_not_a_previous_output():
    """reference_candidates/<char>/ also holds var_*.png and gui_seed*.png generations; feeding
    one of those back as the FaceID reference compounds whatever drift it already had."""
    assert "anchor_seed" in gc.picker_anchor_path(CHARACTER)


# --- Z-Image / SD1.5: the picks degrade to words, loudly ----------------------------------------


@pytest.mark.parametrize("checkpoint", TEXT_ONLY_CHECKPOINTS)
def test_text_only_families_get_no_adapter_inputs(checkpoint):
    p = plan(checkpoint=checkpoint)
    assert p.mode == "text_only"
    assert p.anchor_path is None and p.pose_name is None, (
        "_plan_custom rejects these outright for this family - passing them on would turn a "
        "silent degrade into a hard error at click time"
    )


@pytest.mark.parametrize("checkpoint", TEXT_ONLY_CHECKPOINTS)
def test_text_only_families_describe_the_pose_in_words(checkpoint):
    meta = pose_skeletons.load_meta(POSE)
    p = plan(checkpoint=checkpoint)
    assert meta["prompt_hint"] in p.prompt_body
    assert meta["camera"] in p.prompt_body, "a 2D pose described without a viewpoint is ambiguous"


@pytest.mark.parametrize("checkpoint", TEXT_ONLY_CHECKPOINTS)
def test_text_only_families_warn_about_both_degrades(checkpoint):
    p = plan(checkpoint=checkpoint)
    assert len(p.notices) == 2
    assert any("ControlNet" in n for n in p.notices)
    assert any("FaceID" in n for n in p.notices)
    assert all(checkpoint in n for n in p.notices), "the notice must name the checkpoint that caused it"


@pytest.mark.parametrize("checkpoint", TEXT_ONLY_CHECKPOINTS)
def test_text_only_families_still_describe_the_character(checkpoint):
    """Losing FaceID must not lose the character entirely - the trigger still drives
    character_base_prompt, which is the only identity signal left on these checkpoints."""
    p = plan(checkpoint=checkpoint)
    assert p.trigger == CHARACTER
    assert gc.CHARACTERS[CHARACTER]["appearance"] in p.preview_prompt


# --- the preview must be what actually gets sent -------------------------------------------------


@pytest.mark.parametrize("checkpoint", SDXL_CHECKPOINTS + TEXT_ONLY_CHECKPOINTS)
def test_safety_negatives_survive_every_family(checkpoint):
    assert gc.AGE_SAFETY_NEGATIVE in plan(checkpoint=checkpoint).preview_negative


@pytest.mark.parametrize("checkpoint", ["cyberrealistic_pony", "pony", "pony_realism"])
def test_pony_quality_tags_lead_the_preview(checkpoint):
    assert plan(checkpoint=checkpoint).preview_prompt.startswith(gc.PONY_QUALITY_TAGS)


@pytest.mark.parametrize("checkpoint", sorted(client.SD15_CHECKPOINTS))
def test_sd15_preview_shows_the_gender_weight(checkpoint):
    """SD1.5 renders soft-featured male characters as women without it (see SD15_GENDER_WEIGHT);
    the preview has to show the weight or it isn't showing what will be sent."""
    p = plan(checkpoint=checkpoint, character="minjun")
    assert f"(man:{gc.SD15_GENDER_WEIGHT})" in p.preview_prompt


def test_the_default_checkpoint_previews_as_the_one_that_will_run():
    """checkpoint=None means the GUI default, which _plan_custom substitutes to
    DEFAULT_CUSTOM_CHECKPOINT - a Pony model. A preview without its quality tags would be a
    preview of a generation nobody asked for."""
    assert plan(checkpoint=None).preview_prompt.startswith(gc.PONY_QUALITY_TAGS)


def test_the_scene_reaches_the_prompt():
    p = plan()
    assert scene_library.scene_text(SCENE) in p.prompt_body


def test_extra_text_goes_last_so_it_refines_rather_than_leads():
    p = plan(extra_text="holding a paper cup")
    assert p.prompt_body.endswith("holding a paper cup")


def test_blank_extra_text_adds_nothing():
    assert plan(extra_text="   ").prompt_body == plan(extra_text=None).prompt_body


# --- partial and empty selections ----------------------------------------------------------------


def test_a_character_alone_is_not_enough_to_generate():
    """No scene, no pose, no words: there is nothing to draw, and gen_custom would reject the
    empty prompt further down with a less obvious message."""
    p = plan(pose_slug=None, scene_slug=None)
    assert p.ready is False
    assert p.preview_prompt == ""


def test_nothing_selected_is_not_ready():
    assert gc.plan_picker().ready is False


def test_extra_text_alone_is_enough():
    p = plan(character=None, pose_slug=None, scene_slug=None, extra_text="a red umbrella")
    assert p.ready is True
    assert p.prompt_body == "a red umbrella"


def test_a_pose_alone_works_on_sdxl():
    p = plan(character=None, scene_slug=None, checkpoint="juggernaut")
    assert p.ready is False, "a skeleton with no scene and no words has no prompt to sample from"
    assert p.pose_name == POSE


def test_a_character_without_an_anchor_degrades_and_says_so(monkeypatch):
    """mylora has no anchor_seed*.png today. The generation should still run as a text-only
    description rather than failing, but silently dropping the face lock would look like the
    FaceID pass regressed."""
    monkeypatch.setattr(gc, "picker_anchor_path", lambda trigger: None)
    p = plan(checkpoint="juggernaut")
    assert p.anchor_path is None
    assert len(p.notices) == 1 and "anchor" in p.notices[0]


def test_an_unknown_scene_is_rejected_by_name():
    with pytest.raises(ValueError, match="not_a_scene"):
        plan(scene_slug="not_a_scene")


def test_an_unknown_pose_is_rejected_by_name():
    with pytest.raises(gc.UsageError, match="unknown pose"):
        plan(pose_slug="not_a_pose", checkpoint="juggernaut")


# --- suggestive tier ------------------------------------------------------------------------------


def test_the_suggestive_tier_still_blocks_explicit_content():
    p = plan(tier="suggestive", scene_slug="beach_sunset")
    assert gc.AGE_SAFETY_NEGATIVE in p.preview_negative
    assert "pornographic" in p.preview_negative
