"""prompt_rewrite: Chinese idea -> Pony tags + Z-Image sentences with the local Qwen. Nothing here starts Ollama."""

import json

import pytest

import content_gate
import generate_character as gc
import prompt_rewrite as pr


class Reply:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


class Recorder:
    def __init__(self, content):
        self.content = content
        self.bodies = []

    def __call__(self, url, json=None, timeout=None):
        self.bodies.append(json)
        return Reply(self.content)


def good(**over):
    data = {
        "pony": "1girl, early twenties, sitting, library, window light, half body",
        "zimage": "A woman in her early twenties sits in a library and reads. Warm window light.",
    }
    data.update(over)
    return json.dumps(data)


def get_with(*names):
    class R:
        status_code = 200

        def json(self):
            return {"models": [{"name": n} for n in names]}

    return lambda url, timeout=None: R()


def run(content, idea="坐在圖書館看書的女生", model="m"):
    post = Recorder(content)
    result = pr.rewrite(idea, model=model, post=post, get=get_with("m"), starter=None)
    return result, post


def test_both_prompts_come_back_and_the_model_is_unloaded_after_the_call():
    (pony, zimage), post = run(good())
    assert pony.startswith("1girl") and zimage.startswith("A woman")
    body = post.bodies[0]
    assert body["keep_alive"] == 0 and body["format"] == "json" and body["think"] is False
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 0  # same idea -> same English


def test_the_system_prompt_pins_adults_and_leaves_quality_tags_to_the_code():
    assert "early twenties" in pr.SYSTEM
    assert (
        "never write realistic texture, high detail" in pr.SYSTEM
    )  # measured 2026-10-21: those words roughen the skin
    assert "score_" in pr.SYSTEM and "Do NOT" in pr.SYSTEM


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"pony": "tags"}),
        good(pony="一個女生"),
        good(zimage='She says "hi"'),
        good(pony="score_9, 1girl"),
        good(pony="1girl, rating_safe"),
        good(zimage=""),
        good(zimage="x" * 700),
    ],
)
def test_a_bad_reply_is_refused(content):
    with pytest.raises(pr.RewriteError):
        run(content)


def test_a_blocked_idea_never_reaches_the_model(monkeypatch):
    def block(*texts):
        raise gc.UsageError("blocked")

    monkeypatch.setattr(content_gate, "check", block)
    post = Recorder(good())
    with pytest.raises(gc.UsageError):
        pr.rewrite("想法", model="m", post=post, get=get_with("m"), starter=None)
    assert post.bodies == []


def test_the_models_own_words_are_gated_too(monkeypatch):
    calls = []

    def check(*texts):
        calls.append(texts)
        if len(calls) == 2:
            raise gc.UsageError("blocked reply")

    monkeypatch.setattr(content_gate, "check", check)
    with pytest.raises(gc.UsageError, match="blocked reply"):
        run(good())
    assert len(calls) == 2  # the idea first, then the reply


def test_empty_and_overlong_ideas_are_refused():
    with pytest.raises(pr.RewriteError):
        pr.rewrite("  ", starter=None)
    with pytest.raises(pr.RewriteError):
        pr.rewrite("字" * (pr.MAX_IDEA_CHARS + 1), starter=None)


def test_a_missing_model_without_a_starter_names_what_is_there():
    with pytest.raises(pr.RewriteError, match="other"):
        pr.rewrite("想法", model="m", get=get_with("other"), starter=None)


def test_the_refine_prompt_replaces_the_prompt_only_for_zimage(monkeypatch, tmp_path):
    import comfyui_client as client

    seen = {}
    monkeypatch.setattr(gc, "gen_custom", lambda *a, **k: (tmp_path / f"{a[7]}.png").write_bytes(b"x"))
    monkeypatch.setattr(client, "zimage_missing_files", lambda model="z_image_turbo": [])

    def fake_refine(path, prompt, negative, seed, prefix, denoise, **kw):
        seen["prompt"], seen["negative"] = prompt, negative
        raw = tmp_path / "raw.png"
        raw.write_bytes(b"x")
        return str(raw)

    monkeypatch.setattr(client, "submit_img2img_refine_zimage", fake_refine)
    gc.gen_pony_zimage(
        "1girl, tags",
        "",
        "safe",
        None,
        None,
        str(tmp_path),
        1,
        "s",
        1.0,
        refine_prompt="A woman in her early twenties reads.",
    )
    assert seen["prompt"].startswith("A woman in her early twenties reads.")
    assert "1girl, tags" not in seen["prompt"]
    assert gc.AGE_SAFETY_NEGATIVE.split(",")[0].strip() in seen["negative"]
