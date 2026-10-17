"""training/content_gate.py: the prompt-side gate of the local picture / H3 tools. Explicit sexual terms and minor words are
refused in Chinese and English before translation, a model call or any GPU work, and again on the translator's English."""

import content_gate as cg
import h3_idea as hi
import h3_local as hl
import pytest

import generate_character as gc


@pytest.mark.parametrize(
    "text",
    [
        "a vagina being penetrated",
        "close-up of sexual intercourse",
        "A nude woman",
        "她的陰道被插入",
        "有一對男女在臥室，特寫插入的動作",
        "小學生在教室",
        "an underage schoolgirl",
        "a teen in the kitchen",
    ],
)
def test_explicit_and_minor_terms_are_refused(text):
    with pytest.raises(gc.UsageError, match="不會送出"):
        cg.check(text)


@pytest.mark.parametrize(
    "text",
    [
        "A young adult woman in her twenties laughing in a kitchen, steam rising from a pot",
        "一位二十多歲的台灣年輕女子，穿著米色圍裙，站在廚房裡笑著看向鏡頭",
        "她笑著指向冒煙的鍋子",
        "Sussex essex classic",  # words that merely contain a listed word are not hits
        "快！關小火！",
    ],
)
def test_ordinary_scenes_pass(text):
    cg.check(text)


def test_the_error_names_what_was_found_and_nothing_is_called(tmp_path):
    with pytest.raises(gc.UsageError, match="陰道"):
        cg.check("安全的句子", "她的陰道")
    called = []
    with pytest.raises(gc.UsageError):
        hl.generate_still(
            "有一對男女在臥室，插入的動作",
            "z_image",
            out_dir=str(tmp_path),
            translate=lambda t: called.append(t) or "x",
            guard=type(
                "G", (), {"before_image": lambda s: called.append("unload"), "wait_until_cool": lambda s: None}
            )(),
        )
    assert called == []  # no translation (nothing left the machine), no unloading, no GPU


def test_a_translation_that_comes_back_explicit_is_refused_too(tmp_path):
    called = []
    guard = type("G", (), {"before_image": lambda s: called.append("unload"), "wait_until_cool": lambda s: None})()
    with pytest.raises(gc.UsageError, match="不會送出"):
        hl.generate_still("她在廚房", "z_image", out_dir=str(tmp_path), translate=lambda t: "a nude woman", guard=guard)
    assert called == []


def test_the_video_description_and_the_spoken_line_are_gated(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    for text in ("She smiles. <d>[Chinese] 做愛</d>", "她笑了 <d>[Chinese] 好</d> 陰道"):
        with pytest.raises(gc.UsageError, match="不會送出"):
            hl.generate(str(img), text, translate=lambda t: t, server_up=lambda: True)


def test_the_uncensored_writer_never_sees_a_blocked_idea_or_line():
    def post(*a, **k):
        raise AssertionError("the model was called")

    with pytest.raises(gc.UsageError, match="不會送出"):
        hi.write_description("她被插入", "好", post=post, get=lambda *a, **k: None)
    with pytest.raises(gc.UsageError, match="不會送出"):
        hi.write_description("她笑了", "做愛", post=post, get=lambda *a, **k: None)
