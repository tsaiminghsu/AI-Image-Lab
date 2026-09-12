"""gen_custom's flag-combination resolution, exhaustively.

_plan_custom was lifted out of gen_custom precisely so this file could exist: the routing
between five workflow paths (HQ / pose / facedetailer / SD1.5 / Z-Image) used to be reachable
only by running a real generation, so a combination nobody happened to try could resolve wrong
- or be rejected for the wrong reason - and nothing would say so.

It is pure apart from reading the pose library off disk, so the whole matrix runs offline.
"""

import itertools

import comfyui_client as client
import generate_character as gc
import pose_skeletons
import pytest

A_POSE = pose_skeletons.list_names()[0]
DEFAULTS = dict(
    prompt="portrait photo",
    anchor_path=None,
    pose_reference_path=None,
    pose_name=None,
    use_facedetailer=None,
    checkpoint=None,
    hq=True,
    width=None,
    height=None,
    lora_strength=None,
)


def plan(**overrides):
    return gc._plan_custom(**{**DEFAULTS, **overrides})


# --- rejections: each must name its own reason, not fail somewhere downstream -----------------


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"checkpoint": "not-a-checkpoint"}, "unknown checkpoint"),
        ({"checkpoint": "z_image_turbo", "anchor_path": "a.png"}, "is Z-Image"),
        ({"checkpoint": "z_image_turbo", "pose_reference_path": "p.png"}, "is Z-Image"),
        ({"checkpoint": "z_image_turbo", "use_facedetailer": True}, "is Z-Image"),
        ({"pose_name": A_POSE, "pose_reference_path": "p.png"}, "not both"),
        ({"pose_name": "not-a-pose"}, "unknown pose"),
        ({"pose_name": A_POSE, "hq": False}, "needs the HQ path"),
        ({"hq": False, "pose_reference_path": "p.png"}, "requires anchor_path"),
        (
            {"hq": False, "use_facedetailer": True, "pose_reference_path": "p.png", "anchor_path": "a.png"},
            "can't be combined",
        ),
        ({"hq": False, "use_facedetailer": True}, "requires anchor_path"),
        ({"checkpoint": "realistic_vision", "anchor_path": "a.png"}, "is SD1.5"),
        ({"checkpoint": "realistic_vision", "pose_reference_path": "p.png", "anchor_path": "a.png"}, "is SD1.5"),
    ],
)
def test_rejected_combinations(overrides, fragment):
    with pytest.raises(gc.UsageError) as excinfo:
        plan(**overrides)
    assert fragment in str(excinfo.value)


# --- the resolution rules -----------------------------------------------------------------------


def test_sd15_with_a_pose_but_no_anchor_reports_the_anchor_rule_first():
    """Documented, not endorsed: SD1.5 + a pose reference and no anchor is rejected by the
    legacy-path "pose_reference requires anchor_path" rule, which is checked before the SD1.5
    one, so the message names the less fundamental problem. Both orderings reject the call, so
    this is a message-quality wart rather than a bug; pinned here so the ordering cannot change
    unnoticed, and so it is findable if anyone wonders why the message looks off."""
    with pytest.raises(gc.UsageError, match="requires anchor_path"):
        plan(checkpoint="realistic_vision", pose_reference_path="p.png")


def test_hq_is_the_default_for_sdxl_and_picks_the_pony_photoreal_checkpoint():
    p = plan()
    assert p.use_hq is True
    assert p.effective_checkpoint == gc.DEFAULT_CUSTOM_CHECKPOINT
    assert p.use_facedetailer is True, "HQ defaults FaceDetailer on"


def test_legacy_path_defaults_facedetailer_off_and_keeps_the_caller_checkpoint():
    p = plan(hq=False)
    assert p.use_hq is False
    assert p.use_facedetailer is False
    assert p.effective_checkpoint is None, "only the HQ path substitutes a default checkpoint"


@pytest.mark.parametrize("checkpoint", sorted(client.SD15_CHECKPOINTS))
def test_sd15_falls_back_off_hq_and_gets_its_own_resolution(checkpoint):
    """SD1.5 has no HQ wiring (no SDXL-family adapter/upscale files), so hq=True must fall
    back rather than build a workflow the checkpoint cannot run."""
    p = plan(checkpoint=checkpoint)
    assert p.is_sd15 and p.use_hq is False
    assert (p.width, p.height) == (client.SD15_WIDTH, client.SD15_HEIGHT)


def test_zimage_gets_its_own_resolution_and_no_hq():
    p = plan(checkpoint="z_image_turbo")
    assert p.is_zimage and p.use_hq is False
    assert (p.width, p.height) == (client.ZIMAGE_WIDTH, client.ZIMAGE_HEIGHT)


def test_a_library_skeleton_resolves_to_a_file_and_the_pose_canvas():
    """The canvas has to match the skeleton's own, or ControlNet's hint gets centre-cropped to
    the requested aspect - silently destructive, the head and feet just go."""
    p = plan(pose_name=A_POSE)
    assert p.pose_is_skeleton is True
    assert p.pose_reference_path and p.pose_reference_path.endswith(".png")
    assert (p.width, p.height) == pose_skeletons.canvas_for(A_POSE)


def test_a_library_skeleton_appends_its_camera_hint_to_the_prompt():
    hint = pose_skeletons.load_meta(A_POSE).get("prompt_hint")
    p = plan(pose_name=A_POSE)
    if hint:
        assert p.prompt == f"{DEFAULTS['prompt']}, {hint}"
    else:
        assert p.prompt == DEFAULTS["prompt"]


def test_a_mismatched_canvas_for_a_library_skeleton_is_rejected():
    w, h = pose_skeletons.canvas_for(A_POSE)
    with pytest.raises(gc.UsageError):
        plan(pose_name=A_POSE, width=h * 2, height=w)


def test_a_photo_pose_reference_gets_the_full_body_canvas():
    """A square canvas has no vertical room for a standing figure and just crops back to
    upper body, which defeats the point of supplying a pose at all."""
    assert (plan(pose_reference_path="p.png").width, plan(pose_reference_path="p.png").height) == (
        gc.FULL_BODY_RESOLUTION
    )


def test_explicit_size_always_wins_over_every_default():
    for overrides in ({}, {"pose_reference_path": "p.png"}, {"checkpoint": "realistic_vision"}):
        p = plan(width=640, height=960, **overrides)
        assert (p.width, p.height) == (640, 960)


def test_checkpoint_and_lora_strength_reach_the_workflow_kwargs():
    p = plan(checkpoint="cyberrealistic_pony", lora_strength=0.0)
    assert p.ckpt_kwargs["checkpoint"] == client.CHECKPOINTS["cyberrealistic_pony"]
    assert p.ckpt_kwargs["lora_strength"] == 0.0


def test_zimage_never_passes_a_checkpoint_filename():
    """Z-Image loads three separate files, so there is no CheckpointLoaderSimple name to set."""
    assert "checkpoint" not in plan(checkpoint="z_image_turbo").ckpt_kwargs


# --- the whole matrix: nothing may crash with anything other than UsageError --------------------

_CHECKPOINTS = [None, "juggernaut", "cyberrealistic_pony", "realistic_vision", "z_image_turbo"]
_POSES = [(None, None), ("p.png", None), (None, A_POSE)]


@pytest.mark.parametrize(
    "checkpoint,hq,anchor,pose,facedetailer",
    [
        (c, h, a, p, f)
        for c, h, a, p, f in itertools.product(
            _CHECKPOINTS, [True, False], [None, "a.png"], _POSES, [None, True, False]
        )
    ],
)
def test_every_combination_either_resolves_or_raises_usage_error(checkpoint, hq, anchor, pose, facedetailer):
    """A combination must never fall through to a TypeError/KeyError/AttributeError. Those are
    the failures that used to surface mid-generation as something unrelated."""
    pose_reference_path, pose_name = pose
    try:
        p = plan(
            checkpoint=checkpoint,
            hq=hq,
            anchor_path=anchor,
            pose_reference_path=pose_reference_path,
            pose_name=pose_name,
            use_facedetailer=facedetailer,
        )
    except gc.UsageError:
        return
    assert isinstance(p.use_hq, bool) and isinstance(p.use_facedetailer, bool)
    assert p.width > 0 and p.height > 0
    assert not (p.use_hq and (p.is_sd15 or p.is_zimage)), "HQ must never be selected for SD1.5/Z-Image"
    if p.pose_is_skeleton:
        assert p.use_hq, "a library skeleton only works on the HQ template"
