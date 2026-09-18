"""The Pony tag library backs checkboxes a user ticks, so its failure modes are quiet ones: a
label that drifted away from the pool it mirrors shows one option fewer, a tag that survives a
tier downgrade reaches the generator invisibly, and a score tag emitted here would be prepended
twice without anything erroring. Each of those gets a test below.

The most important ones are the safety pair (rating_explicit is never offered; a suggestive pick
cannot leak into a safe generation) and the end-to-end check that compose()'s output still goes
through _build_prompt_and_negative with every safety negative intact.
"""

import copy

import pytest

import comfyui_client as client
import generate_character as gc
import pony_tags

PONY_CHECKPOINT = "cyberrealistic_pony"
NON_PONY_CHECKPOINTS = ["juggernaut", "realistic_vision", "z_image_turbo"]


def _cjk(text):
    """True if the string contains a CJK ideograph - "is this actually Chinese"."""
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)


def _all_tags(tier="suggestive", checkpoint=PONY_CHECKPOINT):
    """Every selectable tag, as compose() would be handed it if the user ticked everything."""
    return {
        cat["slug"]: [t["tag"] for t in cat["tags"]]
        for cat in pony_tags.categories(tier, checkpoint)
        if cat["selectable"]
    }


# --- structure ------------------------------------------------------------------------------


def test_the_library_validates_and_is_not_empty():
    pony_tags.validate()
    assert len(pony_tags.CATEGORIES) >= 10


def test_every_category_has_both_languages_in_the_right_place():
    for cat in pony_tags.CATEGORIES:
        assert _cjk(cat["name_zh"]), f"{cat['slug']}: name_zh must be Chinese"
        assert _cjk(cat["note_zh"]), f"{cat['slug']}: note_zh must be Chinese"
        for tag in cat["tags"]:
            assert tag["tag"].isascii(), f"{cat['slug']}: tag {tag['tag']!r} must be ASCII"
            assert _cjk(tag["name_zh"]), f"{cat['slug']}: {tag['tag']!r} needs a Chinese label"


def test_both_roles_are_actually_used():
    """The required/optional split is the whole point of the grouping - if one side ever
    emptied out, every test above would still pass while the GUI showed one flat list."""
    roles = {cat["role"] for cat in pony_tags.CATEGORIES}
    assert roles == set(pony_tags.ROLES)


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda c: c.update(slug="Not A Slug"), "slug must match"),
        (lambda c: c.update(role="mandatory"), "role must be one of"),
        (lambda c: c.update(name_zh="  "), "name_zh is empty"),
        (lambda c: c.update(multi="yes"), "multi must be a bool"),
        (lambda c: c.update(tags=[]), "no tags"),
        (lambda c: c["tags"].append({"tag": "1girl", "name_zh": "重複", "tier": "safe"}), "duplicate tag"),
        (lambda c: c["tags"].append({"tag": "測試", "name_zh": "非 ASCII", "tier": "safe"}), "must be ASCII"),
        (lambda c: c["tags"].append({"tag": "score_9", "name_zh": "重複前綴", "tier": "safe"}), "auto-prepended"),
        (
            lambda c: c["tags"].append({"tag": pony_tags.RATING_EXPLICIT, "name_zh": "露骨", "tier": "safe"}),
            "never offered",
        ),
        (lambda c: c["tags"].append({"tag": "nothing", "name_zh": "沒有分級", "tier": "all"}), "tier must be one of"),
    ],
)
def test_a_broken_category_is_rejected_by_name(mutate, fragment, monkeypatch):
    broken = copy.deepcopy(pony_tags.CATEGORIES)
    target = next(c for c in broken if c["slug"] == "subject")
    mutate(target)
    monkeypatch.setattr(pony_tags, "CATEGORIES", broken)
    with pytest.raises(ValueError) as exc:
        pony_tags.validate()
    assert fragment in str(exc.value)


# --- drift guard: the label maps mirror generate_character's pools ---------------------------


@pytest.mark.parametrize(
    "labels,pools",
    [
        (pony_tags.FRAMING_ZH, (gc.CLOSE_ANGLES, gc.FULL_BODY_ANGLES, gc.BACK_VIEW_ANGLES)),
        (pony_tags.POSE_ZH, (gc.POSES,)),
        (pony_tags.OUTFIT_ZH, (gc.OUTFITS, gc.MALE_OUTFITS)),
        (pony_tags.SUGGESTIVE_OUTFIT_ZH, (gc.SUGGESTIVE_OUTFITS, gc.MALE_SUGGESTIVE_OUTFITS)),
        (pony_tags.LIGHTING_ZH, (gc.LIGHTINGS,)),
        (pony_tags.SCENE_ZH, (gc.BACKGROUNDS,)),
        (pony_tags.SUGGESTIVE_SCENE_ZH, (gc.SUGGESTIVE_BACKGROUNDS,)),
    ],
)
def test_label_maps_cover_exactly_their_pools(labels, pools):
    expected = set()
    for pool in pools:
        expected |= set(pool)
    assert set(labels) == expected


def test_pool_helper_names_the_drifted_tag_in_both_directions():
    with pytest.raises(ValueError) as missing:
        pony_tags._pool({"a": "甲"}, ["a", "b"])
    assert "no label for ['b']" in str(missing.value)
    with pytest.raises(ValueError) as extra:
        pony_tags._pool({"a": "甲", "b": "乙"}, ["a"])
    assert "non-existent ['b']" in str(extra.value)


# --- the score prefix stays a single rewrite point -------------------------------------------


def test_quality_category_is_shown_but_not_selectable():
    quality = next(c for c in pony_tags.CATEGORIES if c["slug"] == "quality")
    assert quality["selectable"] is False
    assert [t["tag"] for t in quality["tags"]] == gc.PONY_QUALITY_TAGS.split(", ")


def test_compose_never_emits_a_score_tag():
    body, negative = pony_tags.compose(_all_tags(), "suggestive", PONY_CHECKPOINT)
    assert "score_" not in body
    assert "score_" not in negative


# --- safety ----------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", pony_tags.TIERS)
@pytest.mark.parametrize("checkpoint", [PONY_CHECKPOINT, *NON_PONY_CHECKPOINTS])
def test_rating_explicit_is_never_offered(tier, checkpoint):
    for cat in pony_tags.categories(tier, checkpoint):
        assert pony_tags.RATING_EXPLICIT not in [t["tag"] for t in cat["tags"]]


def test_safe_tier_hides_every_suggestive_entry():
    offered = {t["tag"] for cat in pony_tags.categories("safe", PONY_CHECKPOINT) for t in cat["tags"]}
    for tag in ("rating_questionable", "bikini", "swim trunks", "beach at sunset", "poolside"):
        assert tag not in offered
    assert "rating_safe" in offered


def test_suggestive_tier_is_a_superset_of_safe():
    safe = {t["tag"] for cat in pony_tags.categories("safe", PONY_CHECKPOINT) for t in cat["tags"]}
    suggestive = {t["tag"] for cat in pony_tags.categories("suggestive", PONY_CHECKPOINT) for t in cat["tags"]}
    assert safe < suggestive


def test_a_suggestive_pick_is_dropped_when_the_tier_goes_back_to_safe():
    """The GUI clears these on a tier change, but compose() must not depend on the UI having
    done so - this is the last line between a beach-in-swimwear pick and a safe generation."""
    picked = {"rating": ["rating_questionable"], "outfit": ["bikini"], "scene": ["beach at sunset"]}
    body, _ = pony_tags.compose(picked, "safe", PONY_CHECKPOINT)
    assert body == ""
    body, _ = pony_tags.compose(picked, "suggestive", PONY_CHECKPOINT)
    assert body == "rating_questionable, bikini, beach at sunset"


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        pony_tags.categories("explicit")


# --- per-checkpoint degrade -------------------------------------------------------------------


@pytest.mark.parametrize("checkpoint", NON_PONY_CHECKPOINTS)
def test_non_pony_checkpoints_lose_the_pony_only_categories(checkpoint):
    slugs = [cat["slug"] for cat in pony_tags.categories("suggestive", checkpoint)]
    assert "quality" not in slugs and "source" not in slugs and "rating" not in slugs
    assert "framing" in slugs and "scene" in slugs

    body, _ = pony_tags.compose(_all_tags(), "suggestive", checkpoint)
    assert "rating_" not in body and "source_" not in body
    assert "full body shot" in body  # the descriptive half survives


def test_checkpoint_none_means_the_pipeline_default():
    """gen_custom() falls back to DEFAULT_CUSTOM_CHECKPOINT, so a None here has to resolve the
    same way - otherwise the preview would hide tags the generation then applies."""
    assert gc.DEFAULT_CUSTOM_CHECKPOINT in client.PONY_CHECKPOINTS
    slugs = [cat["slug"] for cat in pony_tags.categories("safe")]
    assert "rating" in slugs


# --- composition ------------------------------------------------------------------------------


def test_output_follows_the_curated_order_not_the_click_order():
    picked = {"scene": ["cozy cafe interior"], "subject": ["adult", "1girl"], "rating": ["rating_safe"]}
    body, _ = pony_tags.compose(picked, "safe", PONY_CHECKPOINT)
    assert body == "rating_safe, 1girl, adult, cozy cafe interior"


def test_single_select_categories_keep_only_one():
    """source and rating are single-valued in Pony's convention; two of them is not "more
    guidance", it is two conflicting instructions. The one kept is the first in curated order."""
    body, _ = pony_tags.compose({"source": ["source_cartoon", "source_anime"]}, "safe", PONY_CHECKPOINT)
    assert body == "source_anime"


def test_unknown_tags_are_ignored():
    body, negative = pony_tags.compose({"subject": ["1girl", "not a real tag"]}, "safe", PONY_CHECKPOINT)
    assert body == "1girl"
    assert negative == ""


def test_extra_negative_category_only_feeds_the_negative_side():
    body, negative = pony_tags.compose(
        {"subject": ["1girl"], "extra_negative": ["bad hands", "jpeg artifacts"]}, "safe", PONY_CHECKPOINT
    )
    assert body == "1girl"
    assert negative == "bad hands, jpeg artifacts"


def test_extra_negative_tags_are_not_already_applied_automatically():
    """Repeating a term the pipeline always adds would not break anything, but it would make the
    category look like it does something it does not."""
    auto = set()
    for constant in (gc.SAFE_SAFETY_NEGATIVE, gc.SUGGESTIVE_NEGATIVE, gc.REALISTIC_NEGATIVE, gc.QUALITY_NEGATIVE):
        auto |= {t.strip() for t in constant.split(",")}
    cat = next(c for c in pony_tags.CATEGORIES if c["slug"] == "extra_negative")
    overlap = [t["tag"] for t in cat["tags"] if t["tag"] in auto]
    assert not overlap, f"already added automatically: {overlap}"


def test_style_tags_do_not_repeat_the_default_style_suffix():
    baked = {t.strip() for t in gc.REALISTIC_STYLE.split(",")}
    cat = next(c for c in pony_tags.CATEGORIES if c["slug"] == "style")
    overlap = [t["tag"] for t in cat["tags"] if t["tag"] in baked]
    assert not overlap, f"already in REALISTIC_STYLE: {overlap}"


def test_choices_show_the_tag_next_to_its_label():
    cat = next(c for c in pony_tags.categories("safe", PONY_CHECKPOINT) if c["slug"] == "subject")
    labels = dict((value, label) for label, value in pony_tags.choices(cat))
    assert "1girl" in labels["1girl"]


def test_every_selectable_category_is_reachable_at_the_widest_combination():
    """gui.py builds one checkbox group per entry of categories("suggestive") and never rebuilds
    the list, so anything missing from that call could never be shown at all."""
    widest = {cat["slug"] for cat in pony_tags.categories("suggestive") if cat["selectable"]}
    assert widest == {cat["slug"] for cat in pony_tags.CATEGORIES if cat["selectable"]}


# --- end to end: the library's output still goes through the shared builder --------------------


def test_composed_body_through_the_shared_builder_keeps_every_invariant():
    picked = {
        "rating": ["rating_safe"],
        "subject": ["1girl", "mature female", "adult"],
        "framing": ["full body shot"],
        "extra_negative": ["bad hands"],
    }
    body, extra_negative = pony_tags.compose(picked, "safe", PONY_CHECKPOINT)
    full, negative = gc._build_prompt_and_negative(body, extra_negative, "safe", None, None, None, PONY_CHECKPOINT)

    # the score prefix is applied exactly once, by the builder, with rating right behind it
    assert full.count("score_9") == 1
    assert full.startswith(f"{gc.PONY_QUALITY_TAGS}, rating_safe, 1girl, mature female, adult, full body shot")
    assert full.endswith(gc.REALISTIC_STYLE)

    # the safety floor is untouched by anything the picker contributed
    assert negative.startswith(gc.PONY_QUALITY_NEGATIVE_TAGS)
    for term in gc.AGE_SAFETY_NEGATIVE.split(", "):
        assert term in negative
    assert negative.endswith("bad hands")
