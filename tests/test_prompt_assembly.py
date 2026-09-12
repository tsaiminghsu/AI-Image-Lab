"""Characterization tests that pin generate_character.py's CURRENT prompt/negative-prompt assembly
behaviour exactly - Pony quality-tag handling, prompt ordering, the SD1.5 gender-weight rescue, and
(most importantly) the three ways gen_video_animatediff's composition legitimately differs from the
still-image paths, now that it goes through the shared _build_prompt_and_negative() helper.

These are characterization tests, not correctness tests: they were written against the inline copy
gen_video_animatediff used to carry, and passing unchanged after that copy was replaced by a call
to the shared helper is what proved the de-duplication behaviour-preserving.
"""

import itertools

import comfyui_client as client
import generate_character as gc
import pose_pack
import pytest


# --- Pony quality-tag auto-prepend ---------------------------------------------------------------


@pytest.mark.parametrize("checkpoint", sorted(client.PONY_CHECKPOINTS))
def test_pony_checkpoint_gets_quality_tag_prefix(checkpoint):
    prompt, negative = gc._build_prompt_and_negative(
        "a photo of a scene",
        None,
        "safe",
        None,
        None,
        None,
        checkpoint,
    )
    assert prompt.startswith(gc.PONY_QUALITY_TAGS)
    assert negative.startswith(gc.PONY_QUALITY_NEGATIVE_TAGS)


@pytest.mark.parametrize("checkpoint", [None, "juggernaut", "realistic_vision"])
def test_non_pony_checkpoint_has_no_quality_tags(checkpoint):
    prompt, negative = gc._build_prompt_and_negative(
        "a photo of a scene",
        None,
        "safe",
        None,
        None,
        None,
        checkpoint,
    )
    assert gc.PONY_QUALITY_TAGS not in prompt
    assert gc.PONY_QUALITY_NEGATIVE_TAGS not in negative


# --- ordering: character prefix, user prompt, style suffix --------------------------------------


def test_ordering_with_trigger_is_base_prompt_then_user_prompt_then_default_style():
    profile = gc.CHARACTERS["mei"]
    prompt, _ = gc._build_prompt_and_negative(
        "wearing a red dress",
        None,
        "safe",
        "mei",
        None,
        None,
        None,
    )
    expected_base = gc.character_base_prompt("mei", profile, gender_weight=None)
    assert prompt == f"{expected_base}, wearing a red dress, {gc.REALISTIC_STYLE}"


def test_style_positive_empty_string_removes_the_style_block():
    """style_positive="" is falsy, so the `if style_positive:` branch is skipped entirely - distinct
    from the None default, which resolves to REALISTIC_STYLE (covered by the ordering test above)."""
    profile = gc.CHARACTERS["mei"]
    prompt, _ = gc._build_prompt_and_negative(
        "wearing a red dress",
        None,
        "safe",
        "mei",
        "",
        None,
        None,
    )
    expected_base = gc.character_base_prompt("mei", profile, gender_weight=None)
    assert prompt == f"{expected_base}, wearing a red dress"


def test_style_positive_none_defaults_to_realistic_style():
    prompt, _ = gc._build_prompt_and_negative(
        "a scene, no character",
        None,
        "safe",
        None,
        None,
        None,
        None,
    )
    assert prompt == f"a scene, no character, {gc.REALISTIC_STYLE}"


# --- SD1.5 gender weight rescue -------------------------------------------------------------------


@pytest.mark.parametrize("checkpoint", sorted(client.SD15_CHECKPOINTS))
def test_sd15_checkpoint_base_prompt_carries_the_gender_weight(checkpoint):
    profile = gc.CHARACTERS["mei"]
    prompt, _ = gc._build_prompt_and_negative(
        "wearing a red dress",
        None,
        "safe",
        "mei",
        None,
        None,
        checkpoint,
    )
    expected_base = gc.character_base_prompt("mei", profile, gender_weight=gc.SD15_GENDER_WEIGHT)
    assert expected_base in prompt
    assert f"({profile['gender']}:{gc.SD15_GENDER_WEIGHT})" in prompt


@pytest.mark.parametrize("checkpoint", [None, "juggernaut", "cyberrealistic_pony"])
def test_sdxl_checkpoint_base_prompt_has_no_gender_weight(checkpoint):
    profile = gc.CHARACTERS["mei"]
    prompt, _ = gc._build_prompt_and_negative(
        "wearing a red dress",
        None,
        "safe",
        "mei",
        None,
        None,
        checkpoint,
    )
    assert f"({profile['gender']}:" not in prompt


# --- get_character with an unknown trigger --------------------------------------------------------


def test_get_character_unknown_trigger_raises_usage_error_with_exact_message():
    """The message is user-facing text and is pinned byte for byte: the CLI prints it to
    stderr, the GUI shows it as a toast, image_api records it as the job error.

    The type changed from SystemExit to UsageError deliberately - SystemExit derives from
    BaseException, which is what let it escape `except Exception` at two boundaries. See
    tests/test_usage_error.py for that contract; the message itself is unchanged.
    """
    with pytest.raises(gc.UsageError) as excinfo:
        gc.get_character("definitely_not_a_real_character")
    assert str(excinfo.value) == (
        "unknown character 'definitely_not_a_real_character'. Known characters: " + ", ".join(sorted(gc.CHARACTERS))
    )


# --- pose_pack routes through the shared builder --------------------------------------------------


def test_pose_pack_build_prompt_routes_through_shared_builder():
    """pose_pack.py's own docstring claims it goes through _build_prompt_and_negative() so a tag
    that looks good in the reference pack composes identically in gen_custom() - confirm the safety
    negative actually comes along for the ride."""
    _, negative = pose_pack.build_prompt("standing straight", "poses", "mei", None)
    assert gc.AGE_SAFETY_NEGATIVE in negative


# --- the most important test in this file ---------------------------------------------------------

_ANIMATEDIFF_TIERS = ["safe", "suggestive"]
_ANIMATEDIFF_STYLE_NEGATIVES = [None, ""]
_ANIMATEDIFF_TRIGGERS = [None, "mei"]


@pytest.mark.parametrize(
    "tier,style_negative,trigger",
    list(itertools.product(_ANIMATEDIFF_TIERS, _ANIMATEDIFF_STYLE_NEGATIVES, _ANIMATEDIFF_TRIGGERS)),
)
def test_animatediff_inline_assembly_matches_shared_builder(monkeypatch, tmp_path, tier, style_negative, trigger):
    """gen_video_animatediff used to duplicate _build_prompt_and_negative() inline rather than call
    it. This test was written against that copy and still passes unchanged now that the copy is
    gone, which is what proves the de-duplication changed nothing observable. It keeps earning its
    place: it is the only thing that catches the shared helper being called with the wrong
    arguments for this path.

    Pinned here, both originally confirmed by reading the inline copy against
    _build_prompt_and_negative:

    - negative prompt: the inline code is a verbatim copy of the shared helper's negative-side
      logic (safety-tier selection by `tier`, then style_negative, then extra_negative), MINUS the
      Pony-tag prepend - irrelevant here since AnimateDiff checkpoints are never Pony. So calling
      the shared helper with checkpoint=None and the inline path's OWN default
      (VIDEO_REALISTIC_NEGATIVE, not the shared helper's REALISTIC_NEGATIVE default) reproduces it
      exactly.
    - positive prompt: the inline code applies SD15_GENDER_WEIGHT to the character base prompt
      UNCONDITIONALLY whenever a trigger is given - unlike the shared helper, which only applies it
      when checkpoint is in client.SD15_CHECKPOINTS (gen_video_animatediff's own checkpoint
      parameter, an ANIMATEDIFF_CHECKPOINTS key, never reaches the shared helper at all). There is
      today no way to force that weight through _build_prompt_and_negative, so the expected prompt
      is built directly from character_base_prompt rather than through the shared helper.
    """
    captured = []

    monkeypatch.setattr(client, "upload_reference_image", lambda path: "face_ref.png")
    monkeypatch.setattr(client, "has_node", lambda name: True)
    monkeypatch.setattr(client, "log_gpu_memory", lambda stage: None)
    monkeypatch.setattr(gc.shutil, "move", lambda src, dst: dst)

    def fake_submit(**kwargs):
        captured.append(kwargs)
        return str(tmp_path / "fake_out.mp4")

    monkeypatch.setattr(client, "submit_generation_animatediff", fake_submit)

    prompt_text = "a scene description"
    gc.gen_video_animatediff(
        prompt_text,
        None,
        tier,
        trigger,
        "unused_face_ref.png",
        str(tmp_path),
        seed=4242,
        style_negative=style_negative,
    )

    assert len(captured) == 1
    got_prompt = captured[0]["prompt"]
    got_negative = captured[0]["negative_prompt"]

    resolved_style_negative = gc.VIDEO_REALISTIC_NEGATIVE if style_negative is None else style_negative
    _, expected_negative = gc._build_prompt_and_negative(
        "irrelevant - negative side doesn't depend on this",
        None,
        tier,
        None,
        None,
        resolved_style_negative,
        None,
    )
    assert got_negative == expected_negative

    if trigger:
        profile = gc.get_character(trigger)
        expected_prompt = (
            f"{gc.character_base_prompt(trigger, profile, gender_weight=gc.SD15_GENDER_WEIGHT)}, {prompt_text}"
        )
    else:
        expected_prompt = prompt_text
    expected_prompt = f"{expected_prompt}, {gc.REALISTIC_STYLE}"
    assert got_prompt == expected_prompt
