"""The scene library backs a gallery the user clicks, so its failure modes are quiet ones: a
typo'd slug shows as an empty tile, a wrong tier offers a swimwear scene in a `safe` session,
and a missing thumbnail is invisible until someone opens the tab. These run offline against the
committed JSON.
"""

import json
import os

import pytest
import scene_library


def test_the_library_loads_and_is_not_empty():
    scenes = scene_library.load_all()
    assert len(scenes) >= 10


def test_every_entry_is_well_formed():
    """load_all() validates on read; this asserts the committed file actually passes rather
    than trusting that whoever last edited it ran the GUI."""
    for scene in scene_library.load_all():
        assert scene_library.SLUG_RE.match(scene["slug"])
        assert scene["tier"] in scene_library.TIERS
        assert isinstance(scene["thumb_seed"], int)


def test_slugs_are_unique():
    slugs = [s["slug"] for s in scene_library.load_all()]
    assert len(slugs) == len(set(slugs))


def test_display_names_are_chinese_and_prompts_are_english():
    """The label is what the user reads; the prompt is what CLIP reads. Mixing them up (a
    Chinese prompt) is the exact failure the picker exists to prevent - CLIP largely ignores
    Chinese, so the scene would silently drop out of the image."""
    for scene in scene_library.load_all():
        assert any("一" <= c <= "鿿" for c in scene["name_zh"]), f"{scene['slug']}: name_zh isn't Chinese"
        for field in ("prompt", "lighting"):
            assert scene[field].isascii(), f"{scene['slug']}: {field} must be English for CLIP"


def test_safe_tier_hides_suggestive_scenes():
    safe = scene_library.list_scenes("safe")
    assert safe, "the safe tier must offer something"
    assert all(s["tier"] == "safe" for s in safe)


def test_suggestive_tier_is_a_superset_of_safe():
    safe = {s["slug"] for s in scene_library.list_scenes("safe")}
    suggestive = {s["slug"] for s in scene_library.list_scenes("suggestive")}
    assert safe < suggestive, "suggestive must add scenes, not replace the safe ones"


def test_list_scenes_preserves_file_order():
    """The gallery is built from this list and the click handler indexes back into it, so any
    reordering between the two would select the wrong scene."""
    order = [s["slug"] for s in scene_library.load_all() if s["tier"] == "safe"]
    assert [s["slug"] for s in scene_library.list_scenes("safe")] == order


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        scene_library.list_scenes("nsfw")


def test_scene_text_joins_setting_and_lighting():
    scene = scene_library.load_all()[0]
    assert scene_library.scene_text(scene["slug"]) == f"{scene['prompt']}, {scene['lighting']}"


def test_unknown_slug_names_itself_and_the_alternatives():
    with pytest.raises(ValueError) as excinfo:
        scene_library.get("not_a_scene")
    assert "not_a_scene" in str(excinfo.value)


@pytest.mark.parametrize(
    "broken, fragment",
    [
        ({"scenes": []}, "no 'scenes' list"),
        ({"scenes": [{"slug": "x"}]}, r"missing field\(s\)"),
        (
            {
                "scenes": [
                    {
                        "slug": "Bad Slug",
                        "name_zh": "壞",
                        "prompt": "p",
                        "lighting": "l",
                        "tier": "safe",
                        "thumb_seed": 1,
                    }
                ]
            },
            "slug must match",
        ),
        (
            {
                "scenes": [
                    {"slug": "x", "name_zh": "測", "prompt": "p", "lighting": "l", "tier": "nsfw", "thumb_seed": 1}
                ]
            },
            "tier must be one of",
        ),
        (
            {
                "scenes": [
                    {"slug": "x", "name_zh": "測", "prompt": "p", "lighting": "l", "tier": "safe", "thumb_seed": "1"}
                ]
            },
            "thumb_seed must be an int",
        ),
        (
            {
                "scenes": [
                    {"slug": "x", "name_zh": "測", "prompt": "  ", "lighting": "l", "tier": "safe", "thumb_seed": 1}
                ]
            },
            "prompt is empty",
        ),
    ],
)
def test_a_broken_entry_is_rejected_by_name(broken, fragment, tmp_path, monkeypatch):
    """Each rejection has to name the entry: the whole file is one JSON, so "invalid scene
    library" alone would mean hunting through 15 entries."""
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    monkeypatch.setattr(scene_library, "SCENES_FILE", str(path))
    with pytest.raises(ValueError, match=fragment):
        scene_library.load_all()


def test_a_missing_file_is_rejected_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_library, "SCENES_FILE", str(tmp_path / "nope.json"))
    with pytest.raises(ValueError, match="not found"):
        scene_library.load_all()


def test_duplicate_slugs_are_rejected(tmp_path, monkeypatch):
    entry = {"slug": "x", "name_zh": "測", "prompt": "p", "lighting": "l", "tier": "safe", "thumb_seed": 1}
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"scenes": [entry, dict(entry)]}), encoding="utf-8")
    monkeypatch.setattr(scene_library, "SCENES_FILE", str(path))
    with pytest.raises(ValueError, match="duplicate slug"):
        scene_library.load_all()


def test_thumb_seeds_are_unique():
    """Two scenes on the same seed with similar prompts render near-identical thumbnails, which
    reads as a bug in the gallery rather than a coincidence."""
    seeds = [s["thumb_seed"] for s in scene_library.load_all()]
    assert len(seeds) == len(set(seeds))


def test_every_scene_has_a_committed_thumbnail():
    """The gallery draws from committed files - no GPU, no ComfyUI. A scene added to the JSON
    without running `scene_library.py render-thumbs` would render as a blank tile."""
    missing = [s["slug"] for s in scene_library.load_all() if not os.path.exists(scene_library.thumb_path(s["slug"]))]
    assert not missing, (
        f"scene(s) with no thumbnail: {missing} - run "
        "ComfyUI\\.venv\\Scripts\\python.exe training\\scene_library.py render-thumbs"
    )


def test_thumbnails_are_small_enough_to_commit():
    for scene in scene_library.load_all():
        path = scene_library.thumb_path(scene["slug"])
        size = os.path.getsize(path)
        assert 1024 < size < 120 * 1024, f"{scene['slug']}: thumbnail is {size} bytes, expected a ~256px JPEG"


# --- extra scenes (the Pony x Z-Image tab's scene menu; no thumbnails) ---------------------------------------------


def test_the_extra_scenes_are_well_formed_and_do_not_reuse_a_library_slug():
    extras = scene_library.load_extras()
    assert len(extras) >= 20
    main = {s["slug"] for s in scene_library.load_all()}
    slugs = [s["slug"] for s in extras]
    assert len(slugs) == len(set(slugs)) and not main & set(slugs)
    for s in extras:
        assert any("一" <= c <= "鿿" for c in s["name_zh"]), f"{s['slug']}: name_zh isn't Chinese"
        assert s["prompt"].isascii() and s["lighting"].isascii(), f"{s['slug']}: prompt and lighting must be English"
        assert s["tier"] in scene_library.TIERS


def test_extra_scenes_describe_a_place_not_a_person():
    # the user's own prompt brings the person; a scene that names one would fight it
    banned = ("woman", "man ", "girl", "boy", "people", "person", "crowd", "child")
    for s in scene_library.load_extras():
        text = f"{s['prompt']} {s['lighting']} ".lower()
        assert not any(w in text for w in banned), f"{s['slug']}: names a person ({text})"


def test_list_all_adds_the_extras_and_still_hides_suggestive_scenes_at_the_safe_tier():
    safe = scene_library.list_all("safe")
    everything = scene_library.list_all("suggestive")
    assert {s["tier"] for s in safe} == {"safe"}
    assert len(everything) == len(scene_library.load_all()) + len(scene_library.load_extras())
    assert {s["slug"] for s in scene_library.load_extras()} <= {s["slug"] for s in safe}
    with pytest.raises(ValueError):
        scene_library.list_all("nsfw")


def test_the_picker_and_its_thumbnail_rule_do_not_see_the_extras():
    picker = {s["slug"] for s in scene_library.list_scenes("suggestive")}
    assert not picker & {s["slug"] for s in scene_library.load_extras()}


def test_get_any_finds_scenes_from_both_files():
    assert scene_library.get_any("cozy_cafe")["slug"] == "cozy_cafe"
    assert scene_library.get_any("night_market")["name_zh"] == "夜市"
    with pytest.raises(ValueError, match="unknown scene"):
        scene_library.get_any("nope")


def test_combine_scene_appends_english_and_names_the_scene_in_chinese_prompts():
    text = "night market street with food stalls, warm string lights at night"
    assert scene_library.combine_scene("a woman smiling", "night_market").startswith(
        "a woman smiling, busy night market"
    )
    zh = scene_library.combine_scene("一位女性在微笑", "night_market")
    assert zh == "一位女性在微笑，場景：夜市（busy " + text + "）"
    assert scene_library.combine_scene("", "night_market").startswith("busy night market")


@pytest.mark.parametrize(
    "scenes, fragment",
    [
        ({"scenes": [{"slug": "x"}]}, "missing field"),
        (
            {"scenes": [{"slug": "cozy_cafe", "name_zh": "測", "prompt": "p", "lighting": "l", "tier": "safe"}]},
            "duplicate",
        ),
        (
            {"scenes": [{"slug": "Bad Slug", "name_zh": "測", "prompt": "p", "lighting": "l", "tier": "safe"}]},
            "slug must match",
        ),
        (
            {"scenes": [{"slug": "new_one", "name_zh": "測", "prompt": "p", "lighting": "l", "tier": "nsfw"}]},
            "tier must be",
        ),
        (
            {"scenes": [{"slug": "new_one", "name_zh": "測", "prompt": " ", "lighting": "l", "tier": "safe"}]},
            "is empty",
        ),
        ({"scenes": []}, "no 'scenes' list"),
    ],
)
def test_a_broken_extra_scene_is_rejected_by_name(scenes, fragment, tmp_path, monkeypatch):
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(scenes, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(scene_library, "EXTRA_FILE", str(path))
    with pytest.raises(ValueError, match=fragment):
        scene_library.load_extras()
