"""Prompt translation for the GUI's "translate to English" button. Tries Google Translate's
public web-widget endpoint first (sl=auto -> tl=en), then falls back to MyMemory's free API if
Google fails - no API key, no local model, no model-load wait for either.

Both are the free/undocumented kind - not officially supported Cloud APIs - so either can
rate-limit or change without notice. Measured 2026-09-12: Google's endpoint returned a flat 429
for this machine's IP (not a one-off blip - three retries and a browser User-Agent both still
got 429, so retrying alone would not have helped), while MyMemory answered normally. Falling
back rather than only retrying covers that case. translate_to_english() raises RuntimeError only
if BOTH fail, with both underlying errors in the message - a failed translation looks identical
to "nothing needed translating" if swallowed silently, and the caller (gui.py) should tell the
user rather than let them submit an untranslated prompt without realizing it.

Chosen over a local MT model (Helsinki-NLP/opus-mt-zh-en, Meta's NLLB-200-distilled-600M) after
both produced bad results on this project's comma-separated-phrase prompt style (not full
sentences): opus-mt mistranslated individual vocabulary (mangled "蓬鬆的棉被" - "fluffy quilt" -
into "loose tampons"), and NLLB hallucinated narrative framing that wasn't in the source text at
all (treated fragment input as if it needed to be "completed" into a coherent sentence/story).
Google Translate's and MyMemory's production systems - trained on real-world short queries, not
just clean sentence pairs - both handle short non-sentence phrases correctly (verified against
the same "蓬鬆的棉被" case) without either failure mode.

MyMemory's API has no "auto-detect, no-op if already English" mode the way Google's sl=auto
does - passing langpair=autodetect|en for English text 403s ("PLEASE SELECT TWO DISTINCT
LANGUAGES") because it detects English and refuses to translate English to English. So the
MyMemory path special-cases plain-ASCII text as already-English and returns it unchanged without
calling the API at all, rather than relying on MyMemory to make that call.

MyMemory's autodetect is also just unreliable on short CJK phrases even when the text plainly
isn't English: "蓬鬆的棉被" alone flip-flopped between two calls minutes apart - one correctly
detected Chinese, the other mistook it for English and 403'd with the same "PLEASE SELECT TWO
DISTINCT LANGUAGES". Passing langpair=zh-CN|en explicitly for that same input worked every time
it was tried. Since this project's GUI is Traditional Chinese, any non-ASCII input is assumed
Chinese (checked for CJK Unified Ideographs) rather than trusting MyMemory's autodetect on it;
autodetect is only used as a last resort for non-ASCII, non-CJK text (Japanese, Korean, etc.).

MyMemory's responseStatus field's JSON type is also inconsistent between calls - a successful
response had it as the int 200, a failed one as the string "403" - so it's compared as a string
here rather than assumed to always be one or the other.
"""

import re

import requests

_GOOGLE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
_MYMEMORY_ENDPOINT = "https://api.mymemory.translated.net/get"
_CJK_RE = re.compile("[\u4e00-\u9fff\u3400-\u4dbf]")  # CJK Unified Ideographs + Extension A


def _translate_via_google(text: str) -> str:
    r = requests.get(
        _GOOGLE_ENDPOINT,
        params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text},
        timeout=10,
    )
    r.raise_for_status()
    segments = r.json()[0]
    return "".join(segment[0] for segment in segments)


def _translate_via_mymemory(text: str) -> str:
    if all(ord(c) < 128 for c in text):
        return text  # already ASCII/English - see the "no auto no-op" note above
    # See the module docstring's "autodetect is also just unreliable" note: assume Chinese for
    # CJK input rather than trusting MyMemory's own detector on it.
    source = "zh-CN" if _CJK_RE.search(text) else "autodetect"
    r = requests.get(_MYMEMORY_ENDPOINT, params={"q": text, "langpair": f"{source}|en"}, timeout=10)
    r.raise_for_status()
    body = r.json()
    if str(body.get("responseStatus")) != "200":
        raise RuntimeError(body.get("responseDetails") or f"MyMemory returned {body!r}")
    return body["responseData"]["translatedText"]


def translate_to_english(text: str) -> str:
    """Returns an English translation of text, suitable to paste into (or edit into) the GUI's
    Prompt field. Already-English text is returned unchanged by either backend (verified, not
    assumed - see the module docstring for how each one achieves that). Raises RuntimeError,
    naming both backends' errors, only if Google AND MyMemory both fail."""
    try:
        return _translate_via_google(text)
    except Exception as google_exc:
        try:
            return _translate_via_mymemory(text)
        except Exception as mymemory_exc:
            raise RuntimeError(
                f"Google Translate 失敗（{google_exc}），備援的 MyMemory 也失敗（{mymemory_exc}）"
            ) from mymemory_exc
