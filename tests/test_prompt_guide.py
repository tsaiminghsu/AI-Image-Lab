"""training/PROMPT_GUIDE.md must never drift from the code it documents.

The guide is the one place that answers "what does each model actually get sent" - the per-model
table, the keyword pools, and eight copy-pasteable positive/negative pairs. That is exactly the kind
of document that rots: before this test existed it had been written once and never touched, so it
still listed 6 of the 13 models and quoted a SAFE_SAFETY_NEGATIVE from before QUALITY_NEGATIVE grew
its duplicate-person terms - the same stale string had also been copied into README.md and
model_prompt_test.py.

So nothing here is hand-maintained knowledge: every assertion re-derives the expected text from the
live constants, and the example blocks are re-built by calling the real assembly functions. Editing
a constant, adding a checkpoint, a character, a scene or a pose skeleton fails this test until the
guide is updated, and the example failures print the exact string to paste back.

Whitespace is normalized before comparing so the guide stays free to wrap long lines for reading.
"""

import json
import os
import re

import comfyui_client as client
import generate_character as gc
import pose_skeletons
import pytest
import scene_library

DOC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "PROMPT_GUIDE.md")

with open(DOC_PATH, encoding="utf-8", newline="") as _f:
    DOC_RAW = _f.read()


def norm(s):
    """Collapse every run of whitespace, so a constant the guide wrapped over two lines still
    matches the single-line value in the code."""
    return re.sub(r"\s+", " ", s).strip()


DOC = norm(DOC_RAW)


def test_doc_is_lf():
    """CLAUDE.md's line-ending rule: only README.md is CRLF. check.ps1 audits the index, this
    catches a working-tree rewrite (e.g. a regeneration script opened without newline="")."""
    assert "\r" not in DOC_RAW


# --- section 3: the constants, quoted verbatim ---------------------------------------------------

DOCUMENTED_CONSTANTS = [
    "REALISTIC_STYLE",
    "REALISTIC_NEGATIVE",
    "VIDEO_REALISTIC_NEGATIVE",
    "WAN_VIDEO_NEGATIVE",
    "PONY_QUALITY_TAGS",
    "PONY_QUALITY_NEGATIVE_TAGS",
    "AGE_SAFETY_NEGATIVE",
    "QUALITY_NEGATIVE",
    "SAFE_SAFETY_NEGATIVE",
    "SUGGESTIVE_NEGATIVE",
    "NEGATIVE_PROMPT",
]


_FENCED_BLOCKS = {norm(b) for b in re.findall(r"^```\n(.*?)\n```$", DOC_RAW, re.S | re.M)}


@pytest.mark.parametrize("name", DOCUMENTED_CONSTANTS)
def test_constant_is_quoted_verbatim(name):
    """The value must be a fenced block of its OWN, not merely a substring of the document. A
    substring check would pass on a truncated block as long as some example prompt happened to
    contain the full string."""
    value = getattr(gc, name)
    assert norm(value) in _FENCED_BLOCKS, f"PROMPT_GUIDE.md's {name} block is stale. Current value:\n\n{value}\n"


def test_every_prompt_constant_is_documented():
    """Guards the list above: a new module-level prompt string must be added to the guide (or
    explicitly skipped here), otherwise this file silently stops covering it."""
    not_prompt_text = {
        "AGE_SAFETY_TERMS",
        "DEFAULT_CUSTOM_CHECKPOINT",
        "REFERENCE_CANDIDATES_DIR",
    }
    suffixes = ("_NEGATIVE", "_TAGS", "_STYLE", "_PROMPT")
    found = {
        n
        for n in dir(gc)
        if n.isupper() and n.endswith(suffixes) and isinstance(getattr(gc, n), str) and n not in not_prompt_text
    }
    missing = sorted(found - set(DOCUMENTED_CONSTANTS))
    assert not missing, f"new prompt constants not documented in PROMPT_GUIDE.md section 3: {missing}"


# --- section 1: every model key and file ---------------------------------------------------------


def _model_rows():
    rows = [(k, v) for k, v in client.CHECKPOINTS.items()]
    rows += [(k, v) for k, v in client.ANIMATEDIFF_CHECKPOINTS.items()]
    rows += [(k, v) for k, v in client.ANIMATEDIFF_MOTION_LORAS.items()]
    for registry in (client.ZIMAGE_MODELS, client.WAN_MODELS):
        for key, files in registry.items():
            rows += [(key, f) for f in files.values()]
    for key, preset in client.ANIMATEDIFF_LCM_PRESETS.items():
        rows += [(key, preset["motion_module"]), (key, preset["lora"])]
    return rows


@pytest.mark.parametrize("key,filename", _model_rows(), ids=lambda v: str(v))
def test_model_key_and_filename_are_listed(key, filename):
    assert f"`{key}`" in DOC_RAW, f"model key {key!r} is missing from PROMPT_GUIDE.md's table"
    assert filename in DOC_RAW, f"{key}'s file {filename!r} is missing from PROMPT_GUIDE.md"


@pytest.mark.parametrize(
    "filename",
    [client.SVD_CHECKPOINT, client.CONTROLNET_MODEL, client.UPSCALE_MODEL, client.ANIMATEDIFF_MOTION_MODULE],
)
def test_supporting_model_file_is_listed(filename):
    assert filename in DOC_RAW


@pytest.mark.parametrize(
    "text",
    [
        f"{client.WIDTH}×{client.HEIGHT}",
        f"{client.SD15_WIDTH}×{client.SD15_HEIGHT}",
        f"{client.ZIMAGE_WIDTH}×{client.ZIMAGE_HEIGHT}",
        f"{client.WAN_WIDTH}×{client.WAN_HEIGHT}",
        f"{client.ANIMATEDIFF_WIDTH}×{client.ANIMATEDIFF_HEIGHT}",
        f"{gc.FULL_BODY_RESOLUTION[0]}×{gc.FULL_BODY_RESOLUTION[1]}",
        f"`{client.SAMPLER}` / `{client.SCHEDULER}`",
        f"`{client.ZIMAGE_SAMPLER}` / `{client.ZIMAGE_SCHEDULER}`",
        f"`{client.WAN_SAMPLER}` / `{client.WAN_SCHEDULER}`",
        f"(man:{gc.SD15_GENDER_WEIGHT})",
        f"`SAFETY_MIN_CFG` = {client.SAFETY_MIN_CFG}",
        f"`DEFAULT_CUSTOM_CHECKPOINT` = `{gc.DEFAULT_CUSTOM_CHECKPOINT}`",
        f"`MINIMUM_AGE` = {gc.MINIMUM_AGE}",
        str(gc.BACK_VIEW_IP_ADAPTER_WEIGHT),
    ],
)
def test_numeric_setting_is_listed(text):
    assert text in DOC_RAW, f"PROMPT_GUIDE.md no longer states {text!r}"


# --- section 4: the keyword pools ----------------------------------------------------------------

POOL_NAMES = [
    "CLOSE_ANGLES",
    "FULL_BODY_ANGLES",
    "BACK_VIEW_ANGLES",
    "POSES",
    "OUTFITS",
    "MALE_OUTFITS",
    "LIGHTINGS",
    "BACKGROUNDS",
    "SUGGESTIVE_OUTFITS",
    "MALE_SUGGESTIVE_OUTFITS",
    "SUGGESTIVE_BACKGROUNDS",
]


def _pool_items():
    return [(name, item) for name in POOL_NAMES for item in getattr(gc, name)]


@pytest.mark.parametrize("pool,item", _pool_items(), ids=lambda v: str(v))
def test_keyword_pool_item_is_listed(pool, item):
    assert f"**`{pool}`**" in DOC_RAW, f"PROMPT_GUIDE.md has no block for the {pool} pool"
    assert item in DOC_RAW, f"{pool} gained {item!r}; add it to PROMPT_GUIDE.md section 4"


@pytest.mark.parametrize("trigger", sorted(gc.CHARACTERS))
def test_character_identity_prefix_is_listed(trigger):
    expected = gc.character_base_prompt(trigger, gc.CHARACTERS[trigger])
    assert norm(expected) in DOC, f"PROMPT_GUIDE.md's identity prefix for {trigger} is stale. Current:\n\n{expected}\n"


@pytest.mark.parametrize("scene", scene_library.list_scenes("suggestive"), ids=lambda s: s["slug"])
def test_scene_is_listed(scene):
    for field in ("slug", "name_zh", "prompt", "lighting"):
        assert scene[field] in DOC_RAW, f"scene {scene['slug']}'s {field} is missing from PROMPT_GUIDE.md"


@pytest.mark.parametrize("slug", pose_skeletons.list_names())
def test_pose_skeleton_is_listed(slug):
    meta = pose_skeletons.load_meta(slug)
    assert f"`{slug}`" in DOC_RAW, f"pose skeleton {slug} is missing from PROMPT_GUIDE.md section 4f"
    for field in ("camera", "prompt_hint"):
        assert meta[field] in DOC_RAW, f"{slug}'s {field} text is stale in PROMPT_GUIDE.md"


# --- section 5: the examples are real assembly output --------------------------------------------

REQUIRED_EXAMPLE_IDS = {
    "sdxl_safe",
    "pony_safe",
    "pony_suggestive",
    "sd15_male",
    "zimage_en",
    "zimage_zh",
    "wan",
    "animatediff",
}

_EXAMPLE_RE = re.compile(
    r"<!-- example: (?P<meta>\{.*?\}) -->\n"
    r"\*\*Positive\*\*\n```\n(?P<positive>.*?)\n```\n"
    r"\*\*Negative\*\*\n```\n(?P<negative>.*?)\n```",
    re.S,
)


def _parse_examples():
    return [
        (json.loads(m.group("meta")), m.group("positive"), m.group("negative")) for m in _EXAMPLE_RE.finditer(DOC_RAW)
    ]


def _build(meta):
    """Re-run the real assembly for one example block. The three branches mirror the three ways the
    pipeline calls the shared builder - see gen_custom(), build_wan_prompts() and the inline call in
    gen_video_animatediff() (which has no build_* wrapper of its own; tests/test_prompt_assembly.py
    pins that argument set)."""
    fn = meta["fn"]
    prompt = meta["prompt"]
    tier = meta.get("tier", "safe")
    trigger = meta.get("trigger")
    extra_negative = meta.get("extra_negative")
    if fn == "build":
        return gc._build_prompt_and_negative(
            prompt,
            extra_negative,
            tier,
            trigger,
            meta.get("style_positive"),
            meta.get("style_negative"),
            meta.get("checkpoint"),
        )
    if fn == "wan":
        return gc.build_wan_prompts(prompt, extra_negative, tier, trigger)
    if fn == "animatediff":
        return gc._build_prompt_and_negative(
            prompt,
            extra_negative,
            tier,
            trigger,
            None,
            gc.VIDEO_REALISTIC_NEGATIVE,
            None,
            gender_weight=gc.SD15_GENDER_WEIGHT,
        )
    raise AssertionError(f"unknown example fn {fn!r} in PROMPT_GUIDE.md")


def test_all_required_examples_are_present():
    """Without this, deleting an example block would be a way to make the comparison below pass."""
    found = {meta["id"] for meta, _, _ in _parse_examples()}
    assert REQUIRED_EXAMPLE_IDS <= found, (
        f"PROMPT_GUIDE.md is missing example(s): {sorted(REQUIRED_EXAMPLE_IDS - found)}"
    )


@pytest.mark.parametrize(
    "meta,positive,negative",
    [pytest.param(*e, id=e[0]["id"]) for e in _parse_examples()],
)
def test_example_matches_real_assembly(meta, positive, negative):
    expected_positive, expected_negative = _build(meta)
    assert norm(positive) == norm(expected_positive), (
        f"example {meta['id']}'s Positive block is stale. Paste this instead:\n\n{expected_positive}\n"
    )
    assert norm(negative) == norm(expected_negative), (
        f"example {meta['id']}'s Negative block is stale. Paste this instead:\n\n{expected_negative}\n"
    )
