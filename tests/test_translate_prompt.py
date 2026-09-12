"""Coverage for translate_prompt.py's Google-then-MyMemory fallback chain.

Written after Google's free endpoint returned a flat 429 for this machine's IP (2026-09-12,
not a one-off - three retries and a browser User-Agent both still got 429). The fallback exists
so a rate-limited/down Google doesn't take the whole "translate to English" button down with it.
"""

import pytest
import requests

import translate_prompt as tp


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json_body = json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_body


def _google_ok(translated="translated text"):
    return _FakeResponse(200, [[[translated, "irrelevant source", None, None, 1]]])


def _mymemory_ok(translated="translated text"):
    return _FakeResponse(200, {"responseStatus": 200, "responseData": {"translatedText": translated}})


def test_google_success_is_returned_directly(monkeypatch):
    monkeypatch.setattr(tp.requests, "get", lambda url, **kw: _google_ok("hello world"))
    assert tp.translate_to_english("你好") == "hello world"


def test_google_failure_falls_back_to_mymemory(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url == tp._GOOGLE_ENDPOINT:
            return _FakeResponse(429)
        return _mymemory_ok("fallback translation")

    monkeypatch.setattr(tp.requests, "get", fake_get)
    assert tp.translate_to_english("你好") == "fallback translation"
    assert calls == [tp._GOOGLE_ENDPOINT, tp._MYMEMORY_ENDPOINT]


def test_both_backends_failing_raises_with_both_errors_named(monkeypatch):
    def fake_get(url, **kwargs):
        return _FakeResponse(429) if url == tp._GOOGLE_ENDPOINT else _FakeResponse(500)

    monkeypatch.setattr(tp.requests, "get", fake_get)
    try:
        tp.translate_to_english("你好")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Google Translate" in str(exc)
        assert "MyMemory" in str(exc)


def test_mymemory_skips_the_api_call_for_ascii_text(monkeypatch):
    """MyMemory 403s on langpair=autodetect|en for English input (it detects English and refuses
    to translate English to English) - the ASCII fast path must return the input unchanged
    without ever hitting the network, matching Google's sl=auto no-op behaviour."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(429)  # Google always fails in this test

    monkeypatch.setattr(tp.requests, "get", fake_get)
    assert tp.translate_to_english("a photo of a scene") == "a photo of a scene"
    assert calls == [tp._GOOGLE_ENDPOINT]  # MyMemory's endpoint never called


def test_mymemory_non_200_response_status_is_treated_as_failure(monkeypatch):
    """MyMemory can return HTTP 200 with an error payload (e.g. quota exceeded) - responseStatus
    inside the body, not the HTTP status code, is what says whether the translation is real."""

    def fake_get(url, **kwargs):
        if url == tp._GOOGLE_ENDPOINT:
            return _FakeResponse(429)
        return _FakeResponse(200, {"responseStatus": 403, "responseDetails": "QUOTA EXCEEDED"})

    monkeypatch.setattr(tp.requests, "get", fake_get)
    try:
        tp.translate_to_english("你好")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "QUOTA EXCEEDED" in str(exc)


def test_mymemory_response_status_as_a_string_is_still_detected_as_failure(monkeypatch):
    """Measured 2026-09-12: MyMemory's responseStatus came back as the string "403" on a failed
    call and the int 200 on a successful one - the check must not assume one JSON type or the
    other, or a string "403" would slip past a bare `!= 200` comparison."""

    def fake_get(url, **kwargs):
        if url == tp._GOOGLE_ENDPOINT:
            return _FakeResponse(429)
        return _FakeResponse(200, {"responseStatus": "403", "responseDetails": "PLEASE SELECT TWO DISTINCT LANGUAGES"})

    monkeypatch.setattr(tp.requests, "get", fake_get)
    try:
        tp.translate_to_english("你好")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "PLEASE SELECT TWO DISTINCT LANGUAGES" in str(exc)


def test_mymemory_uses_explicit_zh_cn_source_for_cjk_text():
    """Measured 2026-09-12: MyMemory's own langpair=autodetect|en detector is unreliable on short
    CJK phrases - the SAME input flip-flopped between correctly detecting Chinese and mistaking
    it for English (403 "PLEASE SELECT TWO DISTINCT LANGUAGES") across calls minutes apart. CJK
    input must route to the explicit zh-CN source instead of trusting autodetect on it."""
    captured = {}

    def fake_get(url, **kwargs):
        captured["langpair"] = kwargs["params"]["langpair"]
        return _mymemory_ok("translated")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tp, "_translate_via_google", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
        mp.setattr(tp.requests, "get", fake_get)
        tp.translate_to_english("你好世界")
    assert captured["langpair"] == "zh-CN|en"


def test_mymemory_uses_autodetect_for_non_cjk_non_ascii_text():
    """Japanese/Korean/etc. input isn't covered by the zh-CN special case - it still goes through
    MyMemory's autodetect, which this project has not observed to be unreliable on non-CJK text."""
    captured = {}

    def fake_get(url, **kwargs):
        captured["langpair"] = kwargs["params"]["langpair"]
        return _mymemory_ok("translated")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tp, "_translate_via_google", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
        mp.setattr(tp.requests, "get", fake_get)
        tp.translate_to_english("こんにちは")  # Japanese, no CJK Unified Ideographs match
    assert captured["langpair"] == "autodetect|en"


def test_google_path_parses_the_multi_segment_response_shape(monkeypatch):
    """Google's endpoint splits a long/punctuated translation into multiple [text, ...] segments
    that must be concatenated, not just segments[0] - a comma-separated prompt is exactly the
    input shape most likely to come back multi-segment."""
    body = [[["Sitting at the window, ", None, None, None, 1], ["smiling", None, None, None, 1]]]
    monkeypatch.setattr(tp.requests, "get", lambda url, **kw: _FakeResponse(200, body))
    assert tp.translate_to_english("坐在窗邊，微笑") == "Sitting at the window, smiling"
