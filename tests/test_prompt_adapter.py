"""prompt_adapter keeps the user's natural language and appends the booru dialect a Pony checkpoint
prefers. These tests pin three things that would otherwise rot silently:

1. The lexicon actually covers the project's OWN vocabulary - every character appearance, keyword
   pool item, scene and pose hint yields at least one tag. Without this, someone adds a new pose or
   outfit and the adapter quietly contributes nothing for it.
2. No derived tag is an age or explicit term. Adaptation only ever ADDS to the positive prompt, so a
   bad lexicon entry would be a way to inject a forbidden term past the negative-prompt safety net.
3. The rules the assembly point relies on: Pony-only, off-switch honoured, natural language never
   rewritten, idempotent, and the LLM hook swappable.

Offline: imports only generate_character / comfyui_client / pose_skeletons / scene_library, all of
which import with just `requests`.
"""

import re

import comfyui_client as client
import generate_character as gc
import pose_skeletons
import prompt_adapter as pa
import pytest
import scene_library


@pytest.fixture(autouse=True)
def _restore_tagger_and_env(monkeypatch):
    """Every test starts with the built-in lexicon tagger and the adapter enabled, and any
    set_tagger() a test installs is undone afterwards."""
    monkeypatch.delenv(pa.ENV_VAR, raising=False)
    pa.set_tagger(None)
    yield
    pa.set_tagger(None)


# --- the lexicon is well-formed ------------------------------------------------------------------


def _load_raw():
    import json

    with open(pa.LEXICON_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_lexicon_entries_are_well_formed():
    data = _load_raw()
    assert data["entries"], "lexicon is empty"
    for e in data["entries"]:
        assert e["match"] and all(isinstance(p, str) and p == p.lower() and p.strip() for p in e["match"]), e
        assert e["tags"] and all(isinstance(t, str) and t == t.lower() and t.strip() for t in e["tags"]), e


def test_no_match_phrase_is_used_twice():
    """A phrase in two entries would make lexicon output depend on entry order; forbid it."""
    seen = {}
    for e in _load_raw()["entries"]:
        for phrase in e["match"]:
            assert phrase not in seen, f"match phrase {phrase!r} appears in two entries"
            seen[phrase] = True


# --- safety: derived tags never carry age/explicit terms -----------------------------------------

REQUIRED_FORBIDDEN_TERMS = {"child", "minor", "teen", "underage", "loli", "nude", "naked", "explicit", "sexual"}


def test_forbidden_terms_constant_is_not_weakened():
    """Pins the CONTENT of FORBIDDEN_TAG_TERMS, so emptying it can't make the next test vacuous."""
    missing = REQUIRED_FORBIDDEN_TERMS - set(pa.FORBIDDEN_TAG_TERMS)
    assert not missing, f"FORBIDDEN_TAG_TERMS no longer covers: {sorted(missing)}"


def test_no_lexicon_tag_contains_a_forbidden_term():
    for e in _load_raw()["entries"]:
        for tag in e["tags"]:
            for bad in pa.FORBIDDEN_TAG_TERMS:
                assert not re.search(r"(?<![a-z0-9])" + re.escape(bad) + r"(?![a-z0-9])", tag), (
                    f"tag {tag!r} contains forbidden term {bad!r} - safety terms belong only in the negative prompt"
                )


# --- coverage: the lexicon knows the project's own vocabulary -------------------------------------


def _project_phrases():
    phrases = []
    for trigger, p in gc.CHARACTERS.items():
        phrases.append((f"CHARACTERS[{trigger}].appearance", p["appearance"]))
        phrases.append((f"CHARACTERS[{trigger}].style", p["style"]))
    for pool in (
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
    ):
        for item in getattr(gc, pool):
            phrases.append((pool, item))
    for s in scene_library.list_scenes("suggestive"):
        phrases.append((f"scene:{s['slug']}.prompt", s["prompt"]))
        phrases.append((f"scene:{s['slug']}.lighting", s["lighting"]))
    for slug in pose_skeletons.list_names():
        meta = pose_skeletons.load_meta(slug)
        phrases.append((f"pose:{slug}.prompt_hint", meta["prompt_hint"]))
        phrases.append((f"pose:{slug}.camera", meta["camera"]))
    return phrases


@pytest.mark.parametrize("origin,text", _project_phrases(), ids=lambda v: v if isinstance(v, str) else v[:40])
def test_every_project_phrase_yields_at_least_one_tag(origin, text):
    assert pa.lexicon_tags(text), f"{origin} = {text!r} produces no booru tag; extend booru_lexicon.json"


# --- the matching rules --------------------------------------------------------------------------


def test_longest_phrase_wins_and_is_not_double_counted():
    tags = pa.lexicon_tags("long straight black hair")
    assert tags.count("black hair") == 1
    assert "long hair" in tags and "straight hair" in tags


def test_tags_follow_prompt_order():
    # cafe appears before the pose word here, and after it when reversed
    assert pa.lexicon_tags("in a cafe, a woman sitting").index("cafe") < pa.lexicon_tags(
        "in a cafe, a woman sitting"
    ).index("sitting")


def test_solo_dropped_when_two_people():
    tags = pa.lexicon_tags("a man and a woman walking")
    assert "1boy" in tags and "1girl" in tags and "solo" not in tags


def test_solo_kept_for_one_person():
    assert "solo" in pa.lexicon_tags("a woman standing")


def test_shirtless_is_allowed_and_shirt_does_not_match_inside_it():
    tags = pa.lexicon_tags("shirtless man")
    assert "shirtless" in tags and "shirt" not in tags


# --- adapt(): family gating, dedup, idempotence, switch ------------------------------------------


@pytest.mark.parametrize("checkpoint", sorted(client.PONY_CHECKPOINTS))
def test_pony_checkpoints_get_adapted(checkpoint):
    adapted, tags = pa.adapt("a woman standing in a park", checkpoint)
    assert tags and adapted != "a woman standing in a park"


@pytest.mark.parametrize("checkpoint", [None, "juggernaut", "realistic_vision", "cyberrealistic", "z_image_turbo"])
def test_non_pony_checkpoints_are_untouched(checkpoint):
    text = "a woman standing in a park"
    assert pa.adapt(text, checkpoint) == (text, [])


def test_pony_membership_matches_client():
    """The module keeps its own frozenset to stay torch-free; it must equal the real one."""
    assert pa._PONY_CHECKPOINTS == client.PONY_CHECKPOINTS


def test_adapt_does_not_repeat_a_tag_already_in_the_prompt():
    adapted, tags = pa.adapt("a woman, long hair, sitting", "pony")
    assert "long hair" not in tags  # already present as prose
    assert "1girl" in tags


def test_adapt_is_idempotent():
    once, _ = pa.adapt("a woman with long black hair in a cafe", "pony")
    twice, added = pa.adapt(once, "pony")
    assert twice == once and added == []


def test_off_switch_disables_adaptation(monkeypatch):
    monkeypatch.setenv(pa.ENV_VAR, "off")
    assert pa.adapt("a woman standing", "pony") == ("a woman standing", [])


def test_set_tagger_swaps_the_backend():
    pa.set_tagger(lambda text: ["custom_tag"])
    adapted, tags = pa.adapt("anything at all", "pony")
    assert tags == ["custom_tag"] and adapted.endswith(", custom_tag")


# --- integration through the real assembly point -------------------------------------------------


def test_assembly_appends_tags_after_prose_and_before_style_for_pony():
    prompt, negative = gc._build_prompt_and_negative(
        "a woman with long black hair sitting in a cafe", None, "safe", None, None, None, "cyberrealistic_pony"
    )
    # order: score tags, then user prose, then booru tags, then REALISTIC_STYLE
    assert prompt.startswith(gc.PONY_QUALITY_TAGS + ", a woman with long black hair sitting in a cafe, ")
    i_prose = prompt.index("sitting in a cafe")
    i_tags = prompt.index("1girl, solo")
    i_style = prompt.index(gc.REALISTIC_STYLE)
    assert i_prose < i_tags < i_style
    # negative side is unaffected by adaptation
    assert negative == f"{gc.PONY_QUALITY_NEGATIVE_TAGS}, {gc.SAFE_SAFETY_NEGATIVE}, {gc.REALISTIC_NEGATIVE}"


def test_assembly_tags_the_character_appearance_when_a_trigger_is_used():
    prompt, _ = gc._build_prompt_and_negative(
        "sitting at a cafe", None, "safe", "mei", None, None, "cyberrealistic_pony"
    )
    # mei's appearance is "long straight jet-black hair ..." -> straight hair is not in the prose
    assert "straight hair" in prompt


def test_assembly_untouched_for_non_pony():
    prompt, _ = gc._build_prompt_and_negative("a woman sitting at a cafe", None, "safe", None, None, None, "juggernaut")
    assert "1girl" not in prompt and "solo" not in prompt
