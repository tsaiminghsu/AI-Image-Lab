"""The anti-drift test, and the reason the catalog is worth having at all.

For every row the catalog declares, this re-runs the real generate_character._plan_custom with
that exact combination and checks the two agree. Without it the catalog is a table of claims that
nothing verifies, and it would start lying the first time somebody edits _plan_custom - which is
precisely the drift it exists to end.

_plan_custom is pure apart from reading the pose library off disk (its own docstring at
generate_character.py:692 says so), so this is offline and costs nothing.

One asymmetry worth naming, because it is a real difference the catalog surfaced rather than
invented: a disabled *feature* makes _plan_custom raise, but a disabled txt2img_hq row does not -
hq silently degrades to the single-pass path for SD1.5 and Z-Image. That is deliberate (hq
defaults to True, so raising would break every SD1.5 CLI call), so the two kinds of row are
checked differently and test_a_disabled_hq_row_degrades_silently_rather_than_raising pins the
distinction instead of papering over it.
"""

import capability_catalog as cc
import comfyui_client as client
import generate_character as gc
import pytest

A_REAL_POSE = "arms_crossed"

# How to ask _plan_custom for each capability: (kwargs, hq).
CAPABILITY_REQUESTS = {
    cc.MODE_TXT2IMG: ({}, False),
    cc.MODE_TXT2IMG_HQ: ({}, True),
    cc.FEATURE_FACEID_ANCHOR: ({"anchor_path": "anchor.png"}, True),
    cc.FEATURE_POSE_CONTROLNET: ({"pose_reference_path": "pose.png", "anchor_path": "anchor.png"}, True),
    cc.FEATURE_POSE_SKELETON: ({"pose_name": A_REAL_POSE}, True),
    cc.FEATURE_FACEDETAILER: ({"use_facedetailer": True, "anchor_path": "anchor.png"}, True),
}


def plan(checkpoint, name):
    kwargs, hq = CAPABILITY_REQUESTS[name]
    return gc._plan_custom(
        "portrait photo",
        kwargs.get("anchor_path"),
        kwargs.get("pose_reference_path"),
        kwargs.get("pose_name"),
        kwargs.get("use_facedetailer"),
        checkpoint,
        hq,
        None,
        None,
        None,
    )


def every_row():
    return [(row.checkpoint, row.name, row.enabled) for row in cc.all_rows()]


# --- the catalog and the code must agree ---------------------------------------------------------


@pytest.mark.parametrize("checkpoint,name,enabled", every_row())
def test_the_catalog_matches_what_plan_custom_actually_does(checkpoint, name, enabled):
    if enabled:
        assert plan(checkpoint, name) is not None, "an enabled row must be reachable"
        return
    if name in cc.MODES:
        # A disabled mode is a degrade, not a refusal - see the module docstring.
        assert plan(checkpoint, name).use_hq is False
        return
    with pytest.raises(gc.UsageError):
        plan(checkpoint, name)


def test_a_disabled_hq_row_degrades_silently_rather_than_raising():
    """hq defaults to True, so raising here would break every SD1.5 and Z-Image CLI call. The
    catalog therefore has to describe a degrade, and this is the line that says so out loud."""
    assert not cc.is_enabled("realistic_vision", cc.MODE_TXT2IMG_HQ)
    assert plan("realistic_vision", cc.MODE_TXT2IMG_HQ).use_hq is False
    assert not cc.is_enabled("z_image_turbo", cc.MODE_TXT2IMG_HQ)
    assert plan("z_image_turbo", cc.MODE_TXT2IMG_HQ).use_hq is False


def test_an_enabled_feature_really_reaches_the_hq_path():
    """_plan_custom hands back only what it rewrote - the caller keeps anchor_path itself - so
    the observable effects of an enabled feature are use_hq and the resolved pose reference."""
    assert plan("cyberrealistic_pony", cc.FEATURE_FACEID_ANCHOR).use_hq is True
    pose = plan("cyberrealistic_pony", cc.FEATURE_POSE_CONTROLNET)
    assert pose.pose_reference_path == "pose.png"
    assert pose.pose_is_skeleton is False
    skeleton = plan("cyberrealistic_pony", cc.FEATURE_POSE_SKELETON)
    assert skeleton.pose_is_skeleton is True
    assert skeleton.pose_reference_path.endswith(f"{A_REAL_POSE}.png")
    assert "arms crossed" in skeleton.prompt, "the skeleton's camera hint is appended to the prompt"


# --- the derivation is real, not a restatement ---------------------------------------------------


def test_capabilities_are_derived_from_the_workflow_contracts(monkeypatch):
    """If the HQ template lost its ControlNet nodes, the pose rows must go disabled on their own.
    Restating the answer in a constant would pass every other test in this file while quietly
    decoupling the catalog from the templates it describes."""
    assert cc.is_enabled("cyberrealistic_pony", cc.FEATURE_POSE_CONTROLNET)
    stripped = {
        template: {node: cls for node, cls in nodes.items() if cls != "ControlNetApplyAdvanced"}
        for template, nodes in cc.wc.CONTRACTS.items()
    }
    monkeypatch.setattr(cc.wc, "CONTRACTS", stripped)
    assert not cc.is_enabled("cyberrealistic_pony", cc.FEATURE_POSE_CONTROLNET)


def test_every_family_template_has_a_contract_entry():
    """The catalog rides on test_workflow_contract.py's file-to-contract bijection; naming a
    template here that has no contract would make the derivation silently return nothing."""
    for family, templates in cc.FAMILY_TEMPLATES.items():
        for template in templates:
            assert template in cc.wc.CONTRACTS, f"{family} names {template}, which has no contract"


def test_family_assignment_matches_the_client_constants():
    for checkpoint in client.ZIMAGE_MODELS:
        assert cc.family_of(checkpoint) == cc.FAMILY_ZIMAGE
    for checkpoint in client.SD15_CHECKPOINTS:
        assert cc.family_of(checkpoint) == cc.FAMILY_SD15
    for checkpoint in set(client.CHECKPOINTS) - client.SD15_CHECKPOINTS:
        assert cc.family_of(checkpoint) == cc.FAMILY_SDXL


def test_an_unknown_checkpoint_is_an_error_not_a_default():
    with pytest.raises(KeyError):
        cc.family_of("not_a_checkpoint")


# --- a disabled row must say something useful ----------------------------------------------------


def test_every_disabled_row_carries_both_reasons():
    for row in cc.all_rows():
        if row.enabled:
            continue
        assert row.reason, f"{row.catalog_id} has no Chinese reason"
        assert row.reason_en, f"{row.catalog_id} has no English reason"


def test_every_chinese_reason_is_actually_chinese():
    """User-facing strings are Traditional Chinese by project convention; an English string here
    would reach the GUI untranslated."""
    for row in cc.all_rows():
        if row.enabled:
            continue
        assert any(0x4E00 <= ord(ch) <= 0x9FFF for ch in row.reason), f"{row.catalog_id}: {row.reason}"


def test_a_reason_names_the_checkpoint_it_is_about():
    row = cc.row("z_image_turbo", cc.FEATURE_FACEID_ANCHOR)
    assert "z_image_turbo" in row.reason
    assert "z_image_turbo" in row.reason_en


def test_an_enabled_row_has_no_reason():
    row = cc.row("cyberrealistic_pony", cc.FEATURE_FACEID_ANCHOR)
    assert row.enabled and row.reason is None and row.reason_en is None


def test_the_english_reasons_still_match_what_plan_custom_raises():
    """The point of keeping reason_en is that turning the catalog into the source of these
    messages changes nothing a library caller or an existing script sees."""
    with pytest.raises(gc.UsageError) as raised:
        plan("z_image_turbo", cc.FEATURE_FACEID_ANCHOR)
    assert str(raised.value) == cc.reason_en_for("z_image_turbo", cc.FEATURE_FACEID_ANCHOR)

    with pytest.raises(gc.UsageError) as raised:
        plan("realistic_vision", cc.FEATURE_FACEID_ANCHOR)
    assert str(raised.value) == cc.reason_en_for("realistic_vision", cc.FEATURE_FACEID_ANCHOR)


# --- parameter bounds come from the catalog, not from a UI constant ------------------------------


def test_defaults_follow_the_per_family_native_resolution():
    """SD1.5 is 512x768 (comfyui_client.py:118), not the SDXL 1024 square. A GUI slider that
    hardcoded 1024 would quietly push SD1.5 off its training resolution."""
    assert cc.defaults_for("realistic_vision")["width"] == client.SD15_WIDTH
    assert cc.defaults_for("realistic_vision")["height"] == client.SD15_HEIGHT
    assert cc.defaults_for("cyberrealistic_pony")["width"] == client.WIDTH


def test_defaults_fall_back_to_txt2img_when_the_hq_row_is_disabled():
    assert not cc.is_enabled("z_image_turbo", cc.MODE_TXT2IMG_HQ)
    assert cc.defaults_for("z_image_turbo")["width"] == client.ZIMAGE_WIDTH


def test_the_zimage_cfg_floor_is_the_safety_floor_not_the_official_one():
    """At cfg 1.0 ComfyUI skips negative conditioning entirely, which would silently drop
    AGE_SAFETY_NEGATIVE. The catalog must never offer a minimum below the floor."""
    cfg = cc.row("z_image_turbo", cc.MODE_TXT2IMG).params["cfg"]
    assert cfg["minimum"] == client.SAFETY_MIN_CFG
    assert cfg["default"] >= client.SAFETY_MIN_CFG


def test_dimension_bounds_agree_with_the_resolver():
    """A catalog maximum the resolver would clamp anyway is a UI that lies about what it accepts."""
    params = cc.row("cyberrealistic_pony", cc.MODE_TXT2IMG_HQ).params
    assert params["width"]["maximum"] == cc.pr.MAX_SIDE
    assert params["width"]["minimum"] == cc.pr.MIN_SIDE
    assert params["width"]["step"] == cc.pr.STEP


# --- the frozen surface --------------------------------------------------------------------------

FROZEN_CATALOG = {
    "cyberrealistic:facedetailer": False,
    "cyberrealistic:faceid_anchor": False,
    "cyberrealistic:pose_controlnet": False,
    "cyberrealistic:pose_skeleton": False,
    "cyberrealistic:txt2img": True,
    "cyberrealistic:txt2img_hq": False,
    "cyberrealistic_pony:facedetailer": True,
    "cyberrealistic_pony:faceid_anchor": True,
    "cyberrealistic_pony:pose_controlnet": True,
    "cyberrealistic_pony:pose_skeleton": True,
    "cyberrealistic_pony:txt2img": True,
    "cyberrealistic_pony:txt2img_hq": True,
    "juggernaut:facedetailer": True,
    "juggernaut:faceid_anchor": True,
    "juggernaut:pose_controlnet": True,
    "juggernaut:pose_skeleton": True,
    "juggernaut:txt2img": True,
    "juggernaut:txt2img_hq": True,
    "pony:facedetailer": True,
    "pony:faceid_anchor": True,
    "pony:pose_controlnet": True,
    "pony:pose_skeleton": True,
    "pony:txt2img": True,
    "pony:txt2img_hq": True,
    "pony_realism:facedetailer": True,
    "pony_realism:faceid_anchor": True,
    "pony_realism:pose_controlnet": True,
    "pony_realism:pose_skeleton": True,
    "pony_realism:txt2img": True,
    "pony_realism:txt2img_hq": True,
    "realistic_vision:facedetailer": False,
    "realistic_vision:faceid_anchor": False,
    "realistic_vision:pose_controlnet": False,
    "realistic_vision:pose_skeleton": False,
    "realistic_vision:txt2img": True,
    "realistic_vision:txt2img_hq": False,
    "z_image_turbo:facedetailer": False,
    "z_image_turbo:faceid_anchor": False,
    "z_image_turbo:pose_controlnet": False,
    "z_image_turbo:pose_skeleton": False,
    "z_image_turbo:txt2img": True,
    "z_image_turbo:txt2img_hq": False,
}


def test_the_catalog_surface_is_frozen():
    """Editing FROZEN_CATALOG is the sign-off, in the same spirit as
    tests/test_cli_surface.py:FROZEN_CLI. A capability appearing or disappearing is a product
    decision, and it should not be possible to make one by accident."""
    assert {row.catalog_id: row.enabled for row in cc.all_rows()} == FROZEN_CATALOG
