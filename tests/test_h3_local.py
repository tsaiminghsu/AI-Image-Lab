"""training/h3_local.py: one local H3 clip per command. No GPU here: the graph, the prompt screen, the length rules, the
model-file check and the order of the steps (unload Ollama and cool first, free ComfyUI afterwards) are pinned."""

import json

import h3_local as hl
import pytest

import generate_character as gc


class FakeResp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class FakeGuard:
    def __init__(self, log):
        self.log = log

    def before_image(self):
        self.log.append("unload")

    def wait_until_cool(self):
        self.log.append("cool")


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "first.png"
    p.write_bytes(b"\x89PNG fake")
    return str(p)


def test_the_graph_is_the_gguf_turbo_i2v_graph_with_no_negative_or_cfg():
    g = hl.build_graph("P", "f.png", seed=7, width=576, height=1024, frames=124, megapixels=0.59, tag="t")
    classes = {n["class_type"] for n in g.values()}
    assert {"UnetLoaderGGUF", "MiniMaxH3TurboLoRA", "MiniMaxH3ImageToVideo", "BasicGuider", "SaveVideo"} <= classes
    assert "KSampler" not in classes and not any("negative" in str(n["inputs"]).lower() for n in g.values())
    assert (
        g["7"]["inputs"]["prompt"] == "P" and g["7"]["inputs"]["length"] == 124 and g["9"]["inputs"]["noise_seed"] == 7
    )
    assert g["11"]["inputs"]["steps"] == hl.h3.STEPS == 8 and g["17"]["inputs"]["filename_prefix"] == "video/h3local_t"
    assert g["15"]["inputs"]["fps"] == 24.0


def test_five_seconds_is_124_frames_and_longer_needs_allow_long():
    assert hl.check_seconds(5) == 124
    with pytest.raises(gc.UsageError, match="allow-long"):
        hl.check_seconds(8)
    assert hl.check_seconds(8, allow_long=True) == 192
    with pytest.raises(gc.UsageError):
        hl.check_seconds(4)  # below H3's trained range
    with pytest.raises(gc.UsageError):
        hl.check_seconds(20, allow_long=True)


def test_a_blocked_prompt_is_rejected_before_any_gpu_work(png):
    log = []
    with pytest.raises(
        gc.UsageError, match="不會送出"
    ):  # the content gate answers first; h3_colab.screen_prompt is the second line
        hl.generate(
            png,
            "a schoolgirl walks in",
            guard=FakeGuard(log),
            start=lambda: pytest.fail("started"),
            server_up=lambda: False,
        )
    assert log == []


def test_a_missing_first_frame_or_model_file_stops_before_unloading(png, monkeypatch, tmp_path):
    log = []
    with pytest.raises(gc.UsageError, match="首幀"):
        hl.generate(str(tmp_path / "nope.png"), "x", guard=FakeGuard(log))
    monkeypatch.setattr(hl, "missing_files", lambda: ["unet/x.gguf"])
    with pytest.raises(gc.UsageError, match="缺模型檔"):
        hl.generate(png, "She smiles at the viewer.", guard=FakeGuard(log), server_up=lambda: True)
    assert log == []


def test_missing_files_lists_what_is_not_on_disk(tmp_path):
    assert len(hl.missing_files(str(tmp_path))) == 5
    for folder, name in hl.required_files():
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(b"x")
    assert hl.missing_files(str(tmp_path)) == []


def test_generate_unloads_and_cools_first_runs_one_clip_and_frees_comfyui_it_started(png, tmp_path, monkeypatch):
    log = []
    out_dir = tmp_path / "out"
    video_dir = tmp_path / "comfy" / "output" / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "h3local_tag1_00001_.mp4").write_bytes(b"mp4")
    monkeypatch.setattr(hl.client, "COMFYUI_DIR", str(tmp_path / "comfy"))
    monkeypatch.setattr(hl, "missing_files", lambda: [])
    monkeypatch.setattr(hl, "free_vram_mib", lambda: 15000)
    monkeypatch.setattr(hl.client, "upload_reference_image", lambda p: "first.png")
    posted = []

    def post(url, json=None, timeout=None):
        posted.append(json)
        return FakeResp({"prompt_id": "abc"})

    def get(url, timeout=None):
        return FakeResp({"abc": {"status": {"completed": True, "status_str": "success", "messages": []}}})

    class Proc:
        def terminate(self):
            log.append("terminated")

    meta = hl.generate(
        png,
        "She smiles. says: <d>[Chinese] 好</d>",
        tag="tag1",
        seed=5,
        out_dir=str(out_dir),
        guard=FakeGuard(log),
        post=post,
        get=get,
        start=lambda: log.append("start") or Proc(),
        server_up=lambda: False,
    )
    assert log == ["unload", "cool", "start", "terminated"]
    assert len(posted) == 1 and meta["frames"] == 124 and meta["seed"] == 5
    assert (out_dir / "h3_local_tag1.mp4").is_file() and (out_dir / "graph_tag1.json").is_file()
    saved = json.loads((out_dir / "h3_local_tag1.json").read_text(encoding="utf-8"))
    assert saved["steps"] == 8 and "好" in saved["prompt"]


def test_an_already_running_comfyui_is_left_running_and_a_busy_card_is_refused(png, tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(hl, "missing_files", lambda: [])
    monkeypatch.setattr(hl, "free_vram_mib", lambda: 6000)
    with pytest.raises(gc.UsageError, match="MiB"):
        hl.generate(
            png,
            "She smiles.",
            guard=FakeGuard(log),
            server_up=lambda: True,
            start=lambda: pytest.fail("x"),
            out_dir=str(tmp_path),
        )
    assert log == ["unload", "cool"]


def _still_run(tmp_path, model, prompt="A young adult woman smiling in a kitchen"):
    raw = tmp_path / "raw.png"
    raw.write_bytes(b"\x89PNG fake")
    calls, log = [], []

    def submit(**kw):
        calls.append(kw)
        return str(raw)

    meta = hl.generate_still(
        prompt,
        model,
        seed=3,
        out_dir=str(tmp_path / "stills"),
        guard=FakeGuard(log),
        submit=submit,
        server_up=lambda: True,
    )
    return meta, calls, log


@pytest.mark.parametrize("model", ["z_image", "qwen_image_2_1"])
def test_a_still_keeps_the_safety_negatives_the_cfg_floor_and_cools_first(tmp_path, model):
    meta, calls, log = _still_run(tmp_path, model)
    assert log == ["unload", "cool"] and len(calls) == 1
    kw = calls[0]
    assert all(t in kw["negative_prompt"] for t in ("child", "minor", "underage", "teen"))
    assert kw["cfg"] >= 1.5 and (kw["width"], kw["height"]) == (720, 1280) and kw["seed"] == 3
    assert (tmp_path / "stills" / f"still_{model}_3.png").is_file() and (
        tmp_path / "stills" / f"still_{model}_3.json"
    ).is_file()
    assert (meta["license"] is not None) == (model == "qwen_image_2_1")


def test_a_blocked_or_empty_still_prompt_never_reaches_the_gpu(tmp_path):
    for prompt in ("a schoolgirl in a kitchen", "   "):
        log = []
        with pytest.raises(Exception):
            hl.generate_still(
                prompt,
                "z_image",
                guard=FakeGuard(log),
                submit=lambda **k: pytest.fail("ran"),
                server_up=lambda: True,
                out_dir=str(tmp_path),
            )
        assert log == []
    with pytest.raises(gc.UsageError, match="沒有這個圖片模型"):
        hl.generate_still("A woman", "nope", guard=FakeGuard([]), out_dir=str(tmp_path))


def fake_translate(text):
    return {"她笑著指向鍋子": "She smiles and points at the pot", "廚房": "a kitchen"}.get(text, "EN(" + text + ")")


def test_translate_outside_lines_leaves_the_spoken_line_alone():
    text = "她笑著指向鍋子 <d>[Chinese] 快！關小火！</d> 廚房"
    out = hl.translate_outside_lines(text, fake_translate)
    assert out == "She smiles and points at the pot <d>[Chinese] 快！關小火！</d> a kitchen"
    english = "She smiles. <d>[Chinese] 好</d> Done."
    assert hl.translate_outside_lines(english, lambda t: pytest.fail("translated English")) == english
    assert hl.has_cjk("好") and not hl.has_cjk("<d>x</d> ok")


def test_a_chinese_description_is_translated_before_the_prompt_is_built(png, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(hl, "missing_files", lambda: [])
    monkeypatch.setattr(hl, "free_vram_mib", lambda: 15000)
    monkeypatch.setattr(hl.client, "upload_reference_image", lambda p: "first.png")
    monkeypatch.setattr(hl, "assemble_prompt", lambda image, d, s, st: seen.setdefault("desc", d) or "P")
    log = []
    video_dir = tmp_path / "comfy" / "output" / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "h3local_t_00001_.mp4").write_bytes(b"mp4")
    monkeypatch.setattr(hl.client, "COMFYUI_DIR", str(tmp_path / "comfy"))

    def get(url, timeout=None):
        return FakeResp({"abc": {"status": {"completed": True, "status_str": "success", "messages": []}}})

    meta = hl.generate(
        png,
        "她笑著指向鍋子 <d>[Chinese] 快！</d>",
        tag="t",
        seed=1,
        out_dir=str(tmp_path / "o"),
        guard=FakeGuard(log),
        post=lambda *a, **k: FakeResp({"prompt_id": "abc"}),
        get=get,
        server_up=lambda: True,
        translate=fake_translate,
    )
    assert seen["desc"] == "She smiles and points at the pot <d>[Chinese] 快！</d>"
    assert meta["description_used"] == seen["desc"] and meta["description_original"].startswith("她笑著")


def test_chinese_without_a_translator_is_refused_before_any_gpu_work(png):
    log = []
    with pytest.raises(gc.UsageError, match="翻成英文"):
        hl.generate(png, "她笑了 <d>[Chinese] 好</d>", guard=FakeGuard(log), server_up=lambda: True)
    assert log == []


def test_a_chinese_blocked_term_is_caught_before_translation_and_a_chinese_still_prompt_is_translated(png, tmp_path):
    log = []
    called = []
    with pytest.raises(Exception):
        hl.generate(
            png,
            "一個小學生在廚房",
            guard=FakeGuard(log),
            server_up=lambda: True,
            translate=lambda t: called.append(t) or "x",
        )
    assert called == [] and log == []
    raw = tmp_path / "raw.png"
    raw.write_bytes(b"x")
    calls = []
    meta = hl.generate_still(
        "她笑著指向鍋子",
        "z_image",
        seed=2,
        out_dir=str(tmp_path / "s"),
        guard=FakeGuard(log),
        submit=lambda **kw: calls.append(kw) or str(raw),
        server_up=lambda: True,
        translate=fake_translate,
    )
    assert meta["prompt_original"] == "她笑著指向鍋子" and meta["prompt_english"] == "She smiles and points at the pot"
    assert "She smiles and points at the pot" in calls[0]["prompt"] and "她" not in calls[0]["prompt"]
    with pytest.raises(gc.UsageError, match="翻成英文"):
        hl.generate_still("她笑著", "z_image", guard=FakeGuard([]), out_dir=str(tmp_path))


def test_people_multi_drops_the_one_person_negatives_but_keeps_the_age_safety_ones(tmp_path):
    seen = {}

    def run(people):
        raw = tmp_path / f"raw_{people}.png"
        raw.write_bytes(b"x")
        calls = []
        hl.generate_still(
            "Two adult friends laughing at a cafe table",
            "z_image",
            seed=1,
            out_dir=str(tmp_path / people),
            guard=FakeGuard([]),
            submit=lambda **kw: calls.append(kw) or str(raw),
            server_up=lambda: True,
            people=people,
        )
        seen[people] = calls[0]["negative_prompt"]

    run("single")
    run("multi")
    assert "two people" in seen["single"] and "multiple people" in seen["single"]
    assert "two people" not in seen["multi"] and "multiple people" not in seen["multi"]
    for text in seen.values():
        assert all(t in text for t in ("child", "minor", "underage", "teen"))
    with pytest.raises(gc.UsageError, match="人數"):
        hl.generate_still("A woman", "z_image", people="crowd", guard=FakeGuard([]), out_dir=str(tmp_path))


def test_one_person_keeps_the_solo_terms_and_two_or_more_drop_them_but_stay_adult(tmp_path):
    def run(people):
        raw = tmp_path / f"r{people}.png"
        raw.write_bytes(b"x")
        calls = []
        hl.generate_still(
            "friends laughing at a picnic",
            "z_image",
            seed=people,
            out_dir=str(tmp_path / "s"),
            guard=FakeGuard([]),
            submit=lambda **kw: calls.append(kw) or str(raw),
            server_up=lambda: True,
            people=people,
        )
        return calls[0]

    one, two, group = run(1), run(2), run(3)
    assert "two people" in one["negative_prompt"] and "crowd" in one["negative_prompt"]
    for kw in (two, group):
        assert "two people" not in kw["negative_prompt"] and "extra person" not in kw["negative_prompt"]
        assert all(t in kw["negative_prompt"] for t in ("child", "minor", "underage", "teen"))
        assert "adult" in kw["prompt"] and kw["cfg"] >= 1.5
    assert two["prompt"].startswith("Exactly two adult people") and group["prompt"].startswith("A small group")
    assert not one["prompt"].startswith("Exactly")
    with pytest.raises(gc.UsageError, match="人數"):
        hl.generate_still("x", "z_image", people=9, guard=FakeGuard([]), out_dir=str(tmp_path))


def test_a_minor_word_is_still_refused_when_there_are_several_people(tmp_path):
    with pytest.raises(Exception):
        hl.generate_still(
            "two schoolgirls at a picnic",
            "z_image",
            people=2,
            guard=FakeGuard([]),
            submit=lambda **k: pytest.fail("ran"),
            server_up=lambda: True,
            out_dir=str(tmp_path),
        )


def test_the_translated_english_says_woman_and_man_not_girl_and_boy():
    assert hl.adultize("The girl and the boy smile. Girls, boys; girlfriend boyfriend") == (
        "The woman and the man smile. Women, men; girlfriend boyfriend"
    )
    out = hl.translate_outside_lines(
        "女生笑了 <d>[Chinese] 好</d> 男生點頭",
        lambda t: {"女生笑了": "The girl laughs", "男生點頭": "the boy nods"}[t],
    )
    assert out == "The woman laughs <d>[Chinese] 好</d> the man nods"


def test_the_users_negative_terms_are_added_after_the_fixed_safety_ones_and_not_gated(tmp_path):
    raw = tmp_path / "raw.png"
    raw.write_bytes(b"x")
    calls = []

    def run(negative, translate=None):
        raw.write_bytes(b"x")
        hl.generate_still(
            "A woman in a kitchen",
            "z_image",
            seed=4,
            out_dir=str(tmp_path / "s"),
            guard=FakeGuard([]),
            submit=lambda **kw: calls.append(kw) or str(raw),
            server_up=lambda: True,
            negative=negative,
            translate=translate,
        )
        return calls[-1]["negative_prompt"]

    neg = run("blurry, nude, child, watermark")  # words a negative list legitimately has: not refused
    assert neg.index("child") < neg.index("blurry, nude, child, watermark")  # the fixed ones come first and stay
    assert all(t in neg for t in ("underage", "teen", "minor"))
    assert "extra fingers" in run("多餘的手指", translate=lambda t: "extra fingers")
    with pytest.raises(gc.UsageError, match="負面詞"):
        run("模糊")
