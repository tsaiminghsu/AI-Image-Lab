"""Prompt translation for the GUI's "translate to English" button. Calls
Google Translate's public web-widget endpoint over HTTP (sl=auto -> tl=en) -
no API key, no local model, no model-load wait, and it auto-detects the
source language so it isn't limited to Chinese.

This is the same undocumented endpoint the free translate.google.com widget
and most "no-API-key" Python translate packages use - not an officially
supported Google Cloud API, so it can rate-limit or change without notice.
translate_to_english() raises RuntimeError on any failure rather than
silently returning the original text - a failed translation looks
identical to "nothing needed translating" if swallowed silently, and the
caller (gui.py) should tell the user rather than let them submit an
untranslated prompt without realizing it.

Chosen over a local MT model (Helsinki-NLP/opus-mt-zh-en, Meta's
NLLB-200-distilled-600M) after both produced bad results on this project's
comma-separated-phrase prompt style (not full sentences): opus-mt
mistranslated individual vocabulary (mangled "蓬鬆的棉被" - "fluffy quilt" -
into "loose tampons"), and NLLB hallucinated narrative framing that wasn't
in the source text at all (treated fragment input as if it needed to be
"completed" into a coherent sentence/story). Google Translate's production
system - trained on real-world short queries, not just clean sentence pairs
- handles short non-sentence phrases correctly without either failure mode.
"""

import requests

_ENDPOINT = "https://translate.googleapis.com/translate_a/single"


def translate_to_english(text: str) -> str:
    """Returns an English translation of text, suitable to paste into (or
    edit into) the GUI's Prompt field. Already-English text is returned
    unchanged (auto-detect resolves it as English and translates en->en as
    a no-op - verified, not assumed). Raises RuntimeError on any network or
    parsing failure."""
    try:
        r = requests.get(
            _ENDPOINT,
            params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text},
            timeout=10,
        )
        r.raise_for_status()
        segments = r.json()[0]
        return "".join(segment[0] for segment in segments)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
