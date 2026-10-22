"""prompt_chat: the GUI's Qwen chat about a picture or video idea. Nothing here starts Ollama."""

import pytest

import content_gate
import generate_character as gc
import prompt_chat as pc


class Reply:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


class Post:
    def __init__(self, content="好的"):
        self.content = content
        self.bodies = []

    def __call__(self, url, json=None, timeout=None):
        self.bodies.append(json)
        return Reply(self.content)


def get_with(*names):
    class R:
        status_code = 200

        def json(self):
            return {"models": [{"name": n} for n in names]}

    return lambda url, timeout=None: R()


def ask(message="幫我想一個咖啡廳的畫面", history=None, content="好的", **kw):
    post = Post(content)
    reply = pc.chat(history, message, model="m", post=post, get=get_with("m"), starter=None, **kw)
    return reply, post


def test_the_reply_comes_back_and_the_model_is_unloaded_after_the_call():
    reply, post = ask(content="想像一位女性坐在窗邊")
    assert reply == "想像一位女性坐在窗邊"
    body = post.bodies[0]
    assert body["keep_alive"] == 0 and body["stream"] is False and body["think"] is False


def test_the_system_prompt_comes_first_and_pins_adults_and_the_two_block_kinds():
    _, post = ask()
    messages = post.bodies[0]["messages"]
    assert messages[0]["role"] == "system" and messages[-1] == {"role": "user", "content": "幫我想一個咖啡廳的畫面"}
    for word in ("adult", "early twenties", "```prompt", "```motion", "no nudity"):
        assert word in pc.SYSTEM


def test_history_is_trimmed_flattened_and_cleaned():
    history = [{"role": "user", "content": f"q{i}"} for i in range(30)]
    history.insert(0, {"role": "system", "content": "ignored"})
    history.append(
        {"role": "assistant", "content": [{"type": "text", "text": "部分"}, {"type": "text", "text": "文字"}]}
    )
    history.append({"role": "assistant", "content": "   "})
    cleaned = pc.clean_history(history)
    assert len(cleaned) == pc.MAX_HISTORY_MESSAGES
    assert cleaned[-1] == {"role": "assistant", "content": "部分文字"}
    assert all(m["role"] in ("user", "assistant") for m in cleaned)
    long = pc.clean_history([{"role": "user", "content": "字" * 5000}])
    assert len(long[0]["content"]) == pc.MAX_MESSAGE_CHARS


def test_the_last_block_of_the_asked_kind_is_extracted_with_whitespace_collapsed():
    text = "先看這個\n```prompt\n一位女性\n在咖啡廳\n```\n再改一下\n```prompt\n一位女性在咖啡廳看書，\n窗邊的自然光\n```\n```motion\n她轉頭微笑\n```"
    assert pc.extract_block(text, "prompt") == "一位女性在咖啡廳看書， 窗邊的自然光"
    assert pc.extract_block(text, "motion") == "她轉頭微笑"
    assert pc.extract_block("沒有區塊", "prompt") is None
    assert pc.extract_block(None, "prompt") is None
    assert pc.extract_block("```python\nprint(1)\n```", "prompt") is None
    with pytest.raises(ValueError):
        pc.extract_block(text, "other")


def test_empty_and_overlong_messages_are_refused():
    with pytest.raises(pc.ChatError):
        pc.chat([], "   ", starter=None)
    with pytest.raises(pc.ChatError, match="最多"):
        pc.chat([], "字" * (pc.MAX_MESSAGE_CHARS + 1), starter=None)


def test_a_blocked_message_never_reaches_the_model(monkeypatch):
    def block(*texts):
        raise gc.UsageError("blocked")

    monkeypatch.setattr(content_gate, "check", block)
    post = Post()
    with pytest.raises(gc.UsageError):
        pc.chat([], "想法", model="m", post=post, get=get_with("m"), starter=None)
    assert post.bodies == []


def test_a_blocked_reply_is_replaced_not_shown(monkeypatch):
    calls = []

    def check(*texts):
        calls.append(texts)
        if len(calls) == 2:
            raise gc.UsageError("blocked reply")

    monkeypatch.setattr(content_gate, "check", check)
    reply, _ = ask(content="被擋的回覆")
    assert reply == pc.BLOCKED_REPLY and len(calls) == 2  # the message first, then the reply


def test_a_missing_model_without_a_starter_names_what_is_there():
    with pytest.raises(pc.ChatError, match="other"):
        pc.chat([], "想法", model="m", get=get_with("other"), starter=None)


def test_an_empty_or_failing_answer_is_a_readable_error():
    with pytest.raises(pc.ChatError, match="空白"):
        ask(content="   ")

    def boom(url, json=None, timeout=None):
        raise pc.requests.ConnectionError("down")

    with pytest.raises(pc.ChatError, match="沒有回應"):
        pc.chat([], "想法", model="m", post=boom, get=get_with("m"), starter=None)
