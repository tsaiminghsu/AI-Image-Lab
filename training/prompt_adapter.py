"""Per-model prompt adaptation: keep the user's natural language, append the dialect a model wants.

The user always types natural language. Most checkpoints are happy with that, but the Pony family
was trained on booru tags and follows them more reliably than prose (see PROMPT_GUIDE.md section 7
and README's pose-tag findings). Rather than force the user to learn two prompt styles, this module
leaves their sentence untouched and *appends* the booru tags it can infer from it, so the model sees
both and can lean on whichever it understands. Nothing is rewritten and nothing is dropped: a phrase
the lexicon does not know simply contributes no tag and still reaches the model as prose.

Where it runs: generate_character._build_prompt_and_negative() calls adapt() once, so every entry
point (CLI, GUI, image_api, the RunPod worker, the picker preview, GIF) gets the same behaviour and
it can never drift between them. Only the Pony checkpoints are adapted; every other family
(SDXL / SD1.5 / Z-Image / Wan / AnimateDiff, and checkpoint=None) is returned unchanged.

Offline and deterministic: the default tagger is a hand-maintained lexicon in booru_lexicon.json,
matched by whole phrase, longest first, in the order the phrases appear in the prompt. This module
imports only json / os / re so it runs in .venv-dev and inside the worker image with no torch. The
lexicon is a stand-in, not a ceiling: set_tagger() swaps in any callable (e.g. an LLM) later without
touching this file's callers or the assembly point.
"""

import json
import os
import re
from functools import lru_cache

LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "booru_lexicon.json")

# Env switch, same os.environ convention as MODEL_VARIANT (no new CLI flag, so FROZEN_CLI is
# untouched): "lexicon" (default) adapts Pony prompts, "off" disables adaptation everywhere.
ENV_VAR = "PROMPT_ADAPTER"

# A derived tag must never smuggle in an age or explicit term - those belong only in the negative
# prompt. Pinned by tests/test_prompt_adapter.py (both this tuple's content and that no lexicon tag
# hits it), the same way test_safety_invariants pins AGE_SAFETY_NEGATIVE. Note "shirtless" is a
# legitimate booru tag the project already uses (MALE_SUGGESTIVE_OUTFITS) and is deliberately absent
# here; "topless"/"nude" etc. are not. "petite" and "youthful" read as age cues in booru tagging;
# "petite build" mapped to "petite" until mei's and wanling's descriptions dropped it (2026-09-28).
FORBIDDEN_TAG_TERMS = (
    "child",
    "children",
    "kid",
    "minor",
    "teen",
    "teenager",
    "underage",
    "young",
    "youthful",
    "petite",
    "loli",
    "shota",
    "nsfw",
    "nude",
    "naked",
    "topless",
    "explicit",
    "sex",
    "sexual",
    "pornographic",
)

# Booru families the adapter targets. Kept as a small map rather than importing comfyui_client so
# this module stays torch-free and importable in isolation; the membership is asserted against
# client.PONY_CHECKPOINTS in tests/test_prompt_adapter.py so the two can't silently diverge.
_PONY_CHECKPOINTS = frozenset({"pony", "cyberrealistic_pony", "pony_realism"})

_tagger = None  # None => use the built-in lexicon tagger


def _word_pattern(phrase):
    # Boundaries that are "not an ascii letter/digit" rather than \b, so phrases with punctuation
    # match cleanly: "t-shirt", "3/4 view", "off-shoulder", "close-up". This also stops "shirt"
    # from matching inside "shirtless" (the char after is a letter) or "man" inside "woman".
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])")


@lru_cache(maxsize=1)
def _entries():
    """Load and compile the lexicon once. Longest phrase first so a specific phrase claims its span
    before a shorter one inside it can (see lexicon_tags)."""
    with open(LEXICON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for e in data["entries"]:
        for phrase in e["match"]:
            entries.append((phrase.lower(), tuple(e["tags"]), _word_pattern(phrase)))
    entries.sort(key=lambda t: len(t[0]), reverse=True)
    return entries


# Person-count tags; "solo" only makes sense when exactly one of these is present.
_PERSON_COUNT_TAGS = frozenset({"1girl", "1boy", "2girls", "2boys", "multiple girls", "multiple boys"})


def lexicon_tags(text):
    """Booru tags inferred from natural-language `text`, in the order the matched phrases appear.

    Longest phrases match first and consume their character span, so "long straight black hair"
    yields its own tags without "black hair" also firing inside it. Tags are de-duplicated keeping
    first appearance, and "solo" is dropped when more than one person-count tag is present."""
    low = text.lower()
    consumed = [False] * len(low)
    hits = []  # (position, tag)
    for phrase, tags, pattern in _entries():
        for m in pattern.finditer(low):
            s, e = m.span()
            if any(consumed[s:e]):
                continue
            for i in range(s, e):
                consumed[i] = True
            for tag in tags:
                hits.append((s, tag))
    hits.sort(key=lambda h: h[0])

    ordered = []
    seen = set()
    for _, tag in hits:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)

    if "solo" in seen and len(seen & _PERSON_COUNT_TAGS) > 1:
        ordered = [t for t in ordered if t != "solo"]
    return ordered


def set_tagger(fn):
    """Register a replacement tagger, fn(text:str) -> list[str], or None to restore the lexicon.
    The hook for a future LLM-based tagger; callers and the assembly point stay unchanged."""
    global _tagger
    _tagger = fn


def _current_tagger():
    return _tagger if _tagger is not None else lexicon_tags


def dialect_for(checkpoint):
    """The booru dialect this checkpoint wants, or None if it takes natural language as-is."""
    return "pony" if checkpoint in _PONY_CHECKPOINTS else None


def adapt(prompt, checkpoint):
    """Return (possibly_extended_prompt, appended_tags).

    For a Pony checkpoint (and only when PROMPT_ADAPTER is not "off") this appends the inferred booru
    tags that are not already spelled out in the prompt. For every other checkpoint it returns the
    prompt unchanged and an empty list. Idempotent: re-running over its own output appends nothing,
    because every tag it added is now present in the text."""
    if os.environ.get(ENV_VAR, "lexicon").lower() == "off":
        return prompt, []
    if dialect_for(checkpoint) is None:
        return prompt, []

    tags = _current_tagger()(prompt)
    low = prompt.lower()
    new_tags = [t for t in tags if not _word_pattern(t).search(low)]
    if not new_tags:
        return prompt, []
    return f"{prompt}, {', '.join(new_tags)}", new_tags


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Show how the prompt adapter extends a natural-language prompt for a checkpoint."
    )
    parser.add_argument("prompt", help="the natural-language prompt")
    parser.add_argument("--checkpoint", default="cyberrealistic_pony", help="checkpoint key")
    args = parser.parse_args(argv)
    adapted, tags = adapt(args.prompt, args.checkpoint)
    print("dialect:", dialect_for(args.checkpoint) or "(natural language, no adaptation)")
    print("appended tags:", ", ".join(tags) or "(none)")
    print("adapted prompt:")
    print(adapted)


if __name__ == "__main__":
    _main()
