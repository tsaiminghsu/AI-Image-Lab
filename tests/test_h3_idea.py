"""training/h3_idea.py: a Chinese idea + the app's line -> the English H3 description, written by local qwen3.5 around the
line. The line is placed by code (never by the model), the model's text is validated, and Ollama is called with
keep_alive 0 so the GPU is free again. No network here: Ollama is faked."""

import json

import h3_idea as hi
import h3_local as hl
import pytest

import generate_character as gc


class Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def fake(reply, models=(hi.DEFAULT_MODEL,)):
    calls = []

    def get(url, timeout=None):
        return Resp({"models": [{"name": m} for m in models]})

    def post(url, json=None, timeout=None):
        calls.append(json)
        return Resp({"message": {"content": reply}})

    return get, post, calls


GOOD = json.dumps(
    {
        "before": "The woman in the first frame laughs and points toward the pot, then looks at the viewer.",
        "after": "Then she laughs again, still facing the viewer. The camera holds a steady shot.",
    }
)


def test_the_line_is_placed_by_the_code_inside_the_d_tags():
    get, post, calls = fake(GOOD)
    text = hi.write_description("她笑著指向冒煙的鍋子", "快！關小火！", get=get, post=post)
    assert "<d>[Chinese] 快！關小火！</d>" in text and text.index("points toward") < text.index("<d>") < text.index(
        "Then she"
    )
    assert "The woman (S1), in a clear, warm voice with a natural Taiwanese Mandarin accent, says:" in text
    body = calls[0]
    assert body["keep_alive"] == 0 and body["stream"] is False and body["format"] == "json" and body["think"] is False
    assert "快！關小火！" in body["messages"][1]["content"] and "do not repeat it" in body["messages"][1]["content"]


def test_the_result_passes_the_h3_prompt_screen():
    get, post, _ = fake(GOOD)
    text = hi.write_description("x", "好", get=get, post=post)
    assert "<Picture 1>" in hl.assemble_prompt("first.png", text)


def test_a_man_gets_the_man_wording():
    get, post, _ = fake(GOOD.replace("woman", "man"))
    assert "The man (S1)" in hi.write_description("他笑了", "好", "man", get=get, post=post)


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        json.dumps({"before": "x"}),
        json.dumps({"before": "她笑了", "after": "ok"}),
        json.dumps({"before": 'She says "hi".', "after": "ok"}),
        json.dumps({"before": "<d>[Chinese] 好</d>", "after": "ok"}),
        json.dumps({"before": "", "after": "ok"}),
        json.dumps({"before": "a" * 500, "after": "ok"}),
    ],
)
def test_a_bad_model_reply_is_refused(reply):
    get, post, _ = fake(reply)
    with pytest.raises(hi.IdeaError):
        hi.write_description("x", "好", get=get, post=post)


def test_line_length_tags_and_empty_input_are_checked_before_calling_the_model():
    get, post, calls = fake(GOOD)
    for line in ("", "字" * 21, "a<b"):
        with pytest.raises(hi.IdeaError):
            hi.write_description("x", line, get=get, post=post)
    with pytest.raises(hi.IdeaError):
        hi.write_description("  ", "好", get=get, post=post)
    with pytest.raises(hi.IdeaError):
        hi.write_description("x", "好", "other", get=get, post=post)
    assert calls == []


def test_a_missing_model_or_no_ollama_gives_a_pointer_not_a_crash():
    get, post, calls = fake(GOOD, models=())
    with pytest.raises(hi.IdeaError, match="沒有任何模型"):
        hi.write_description("x", "好", get=get, post=post, starter=None)
    assert calls == []

    def down(url, timeout=None):
        raise hi.requests.ConnectionError("down")

    with pytest.raises(hi.IdeaError, match="連不上 Ollama"):
        hi.find_ollama(down)


def test_an_error_is_a_usage_error_so_the_cli_exits_cleanly():
    assert issubclass(hi.IdeaError, gc.UsageError)


def test_h3_local_takes_either_an_idea_or_a_description(tmp_path, monkeypatch):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    with pytest.raises(gc.UsageError, match="擇一"):
        hl.main([str(img), "--idea", "x", "--line", "好", "--description", "y"])
    with pytest.raises(gc.UsageError, match="--line"):
        hl.main([str(img), "--idea", "x"])
    with pytest.raises(gc.UsageError, match="--description"):
        hl.main([str(img)])
    get, post, _ = fake(GOOD)
    monkeypatch.setattr(hi.requests, "get", get)
    monkeypatch.setattr(hi.requests, "post", post)
    monkeypatch.setattr(hi, "write_description", lambda *a, **k: "DESC")
    hl.main([str(img), "--idea", "x", "--line", "好", "--write-only"])  # prints and stops: no GPU, no ComfyUI


def test_a_model_missing_on_the_running_ollama_is_served_by_a_private_one_that_is_stopped_after(tmp_path):
    folder = tmp_path / "models"
    manifest = folder / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    manifest.mkdir(parents=True)
    (manifest / "9b").write_text("{}")
    assert hi.model_folder("qwen3.5:9b", [str(tmp_path / "empty"), str(folder)]) == str(folder)
    assert hi.model_folder("qwen3.5:9b", [str(tmp_path / "empty")]) is None

    get, post, calls = fake(GOOD, models=())
    log = []

    class Proc:
        def terminate(self):
            log.append("stopped")

    def starter(model):
        log.append(f"started {model}")
        return Proc()

    text = hi.write_description("x", "好", get=get, post=post, starter=starter)
    assert "<d>[Chinese] 好</d>" in text and log == [f"started {hi.DEFAULT_MODEL}", "stopped"]
    assert len(calls) == 1


def test_the_private_ollama_gets_the_right_folder_and_port(tmp_path, monkeypatch):
    folder = tmp_path / "m"
    manifest = folder / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5"
    manifest.mkdir(parents=True)
    (manifest / "9b").write_text("{}")
    monkeypatch.setattr(hi.shutil, "which", lambda name: "ollama.exe")
    seen = {}

    class Proc:
        def terminate(self):
            pass

    def popen(cmd, env=None, **kw):
        seen.update(cmd=cmd, env=env)
        return Proc()

    proc = hi.start_own_ollama("qwen3.5:9b", popen=popen, get=lambda u, timeout=None: Resp({}), folders=[str(folder)])
    assert isinstance(proc, Proc) and seen["cmd"] == ["ollama.exe", "serve"]
    assert seen["env"]["OLLAMA_HOST"] == f"127.0.0.1:{hi.OWN_PORT}" and seen["env"]["OLLAMA_MODELS"] == str(folder)
    with pytest.raises(hi.IdeaError, match="找不到"):
        hi.start_own_ollama("qwen3.5:9b", popen=popen, folders=[str(tmp_path / "nope")])


def test_the_default_model_is_the_hf_gguf_pulled_here_and_its_manifest_path_is_found(tmp_path):
    assert hi.DEFAULT_MODEL.startswith("hf.co/HauhauCS/Qwen3.5-9B") and hi.DEFAULT_MODEL.endswith(":Q4_K_M")
    repo = tmp_path / "manifests" / "hf.co" / "HauhauCS" / "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
    repo.mkdir(parents=True)
    (repo / "Q4_K_M").write_text("{}")
    assert hi.model_folder(hi.DEFAULT_MODEL, [str(tmp_path)]) == str(tmp_path)


def test_the_multi_people_writer_allows_other_adults_but_still_one_speaker():
    get, post, calls = fake(GOOD)
    hi.write_description("兩個朋友在咖啡廳聊天", "好", get=get, post=post, people="multi")
    system = calls[0]["messages"][0]["content"]
    assert "other people from the idea may appear" in system and "everyone is an adult" in system
    assert "only the person we follow speaks" in system and "nobody else appears" not in system
    get, post, calls = fake(GOOD)
    hi.write_description("她笑了", "好", get=get, post=post)
    assert "nobody else appears" in calls[0]["messages"][0]["content"]


def test_a_couple_puts_the_woman_as_s1_and_the_man_as_s2_with_both_lines_kept_by_the_code():
    get, post, calls = fake(GOOD.replace("The woman in the first frame", "The woman and the man in the first frame"))
    text = hi.write_description("兩個人在廚房裡", "快！", "couple", line2="好啦", get=get, post=post)
    assert "The woman (S1)" in text and "<d>[Chinese] 快！</d>" in text
    assert "The man (S2)" in text and "<d>[Chinese] 好啦</d>" in text
    assert text.index("S1") < text.index("S2")
    body = calls[0]
    assert body["messages"][0]["content"] == hi.SYSTEM_COUPLE and "exactly these two adults" in hi.SYSTEM_COUPLE
    assert "The man answers" in body["messages"][1]["content"] and "好啦" in body["messages"][1]["content"]
    assert "nobody else appears" not in hi.SYSTEM_COUPLE


def test_a_couple_without_a_second_line_only_has_the_womans_line_and_the_second_line_needs_a_couple():
    get, post, _ = fake(GOOD)
    text = hi.write_description("x", "快！", "couple", get=get, post=post)
    assert "(S2)" not in text and text.count("<d>") == 1
    with pytest.raises(hi.IdeaError, match="一男一女"):
        hi.write_description("x", "快！", "woman", line2="好", get=get, post=post)
    with pytest.raises(hi.IdeaError):
        hi.write_description("x", "快！", "couple", line2="字" * 21, get=get, post=post)
