"""training/drama.py and drama_compose.py - the short-drama pipeline, with every GPU, cloud and ffmpeg
call replaced by a recorder. What matters most: a bad shot list is refused before anything runs or
bills; keyframes go through the same plan_picker/gen_custom path as the picker tab (so the safety
negatives are the existing ones); cloud motion never leaves without confirmation and always carries
the episode's tier and a cfg at or above the floor; and the assembled commands put picture, voice
and subtitle where the timing says.
"""

import copy
import glob
import json
import os
import re
import subprocess
import unicodedata
import wave

import pytest

import cloud_video
import comfyui_client as client
import drama
import drama_compose as compose
import generate_character as gc

EXAMPLE = os.path.join(drama.EPISODES_DIR, "example_cafe_reunion.json")


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(drama, "OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(drama, "VOICES_DIR", str(tmp_path / "voices"))
    monkeypatch.setattr(gc, "picker_anchor_path", lambda trigger: f"/anchors/{trigger}.png")


CONSENT = {"speaker": "測試錄音者", "date": "2026-09-27", "scope": "單元測試"}


def make_voice(voice_id, transcript, consent=CONSENT):
    folder = os.path.join(drama.VOICES_DIR, voice_id)
    touch(os.path.join(folder, "prompt.wav"))
    config = {"prompt_wav": "prompt.wav", "prompt_text": transcript}
    if consent is not None:
        config["consent"] = consent
    with open(os.path.join(folder, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    return folder


def write_episode(tmp_path, data, name="ep01"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return drama.load_episode(str(path))


def example_data():
    with open(EXAMPLE, encoding="utf-8") as f:
        return json.load(f)


def touch(path, content=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def write_wav(path, seconds, rate=24000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))


# --- the episode file -------------------------------------------------------------------------


def test_example_episode_is_valid():
    errors, _warnings = drama.validate_episode(drama.load_episode(EXAMPLE))
    assert errors == []


def test_example_uses_adult_characters_and_the_safe_tier():
    data = example_data()
    assert data["tier"] == "safe"
    for shot in data["shots"]:
        if shot.get("character"):
            assert gc.CHARACTERS[shot["character"]]["age"] >= gc.MINIMUM_AGE


def _mutate(path, value):
    def apply(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return apply


_DELETE = object()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (_mutate(["version"], 2), "version"),
        (_mutate(["tier"], "explicit"), "tier"),
        (_mutate(["checkpoint"], "nope"), "checkpoint"),
        (_mutate(["shots"], []), "至少要有一個鏡頭"),
        (_mutate(["shots", 1, "id"], "s01"), "重複"),
        (_mutate(["shots", 1, "id"], "S 2"), "id 只能"),
        (_mutate(["shots", 1, "type"], "montage"), "type"),
        (_mutate(["shots", 1, "duration"], 6), "duration"),
        (_mutate(["shots", 1, "duration"], "3"), "duration"),
        (_mutate(["shots", 1, "framing"], "extreme"), "framing"),
        (_mutate(["shots", 0, "camera"], "spin"), "camera"),
        (_mutate(["shots", 1, "character"], "nobody"), "角色表"),
        (_mutate(["shots", 1, "scene"], "moon_base"), "場景庫"),
        (_mutate(["shots", 1, "scene"], "poolside"), "suggestive"),
        (_mutate(["shots", 1, "pose"], "flying"), "姿勢庫"),
        (_mutate(["shots", 3, "line"], ""), "台詞"),
        (_mutate(["shots", 3, "line"], "長" * 121), "拆成兩個鏡頭"),
        (_mutate(["shots", 0, "scene"], _DELETE), "prompt 或 scene"),
        (_mutate(["voices"], {"xinyi": "Bad Voice"}), "聲音 id"),
        (_mutate(["bgm"], "missing.mp3"), "配樂"),
    ],
)
def test_validation_refuses_mistakes(tmp_path, mutation, message):
    data = example_data()
    mutation(data)
    if message == "prompt 或 scene":
        data["shots"][0]["prompt"] = ""
    errors, _ = drama.validate_episode(write_episode(tmp_path, data))
    assert any(message in e for e in errors), errors


def test_episode_filename_becomes_a_folder_name_so_it_must_be_ascii(tmp_path):
    errors, _ = drama.validate_episode(write_episode(tmp_path, example_data(), name="第一集"))
    assert any("檔名" in e for e in errors)


def test_missing_anchor_is_a_warning_at_plan_time(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "picker_anchor_path", lambda trigger: None)
    errors, warnings = drama.validate_episode(write_episode(tmp_path, example_data()))
    assert errors == [] and any("anchor" in w for w in warnings)


def test_require_valid_raises_usage_error(tmp_path):
    data = example_data()
    data["shots"][1]["type"] = "montage"
    with pytest.raises(gc.UsageError, match="分鏡表有問題"):
        drama.require_valid(write_episode(tmp_path, data))


def test_select_shots_filters_and_refuses_unknown_ids():
    ep = drama.load_episode(EXAMPLE)
    assert [s["id"] for _i, s in drama.select_shots(ep, "s04,s02")] == ["s02", "s04"]
    assert all(s["type"] in drama.MOVING_TYPES for _i, s in drama.select_shots(ep, types=drama.MOVING_TYPES))
    with pytest.raises(gc.UsageError, match="s99"):
        drama.select_shots(ep, "s01,s99")


# --- frames and cost --------------------------------------------------------------------------


@pytest.mark.parametrize("seconds, frames", [(1.5, 37), (2.0, 49), (3.0, 73), (5.0, 121), (5.04, 121), (0.5, 17)])
def test_frames_for_snaps_to_wan_lengths(seconds, frames):
    assert drama.frames_for(seconds) == frames
    assert (frames - 1) % 4 == 0


def test_estimate_matches_the_cost_report_model():
    ep = drama.load_episode(EXAMPLE)
    shots = drama.select_shots(ep)
    n, preview = drama.estimate(ep, shots, "preview")
    n_final, final = drama.estimate(ep, shots, "final")
    cold = drama.RUNPOD_PER_HOUR / 60 * drama.COLD_START_MINUTES
    assert n == n_final == 6
    assert preview == pytest.approx(cold + 6 * drama.PREVIEW_COST)
    # a 3 s shot is 19 of the full render's 31 latent frames: 9 min * 19/31 at $1.10/h
    assert drama.final_cost(73) == pytest.approx(1.10 / 60 * 9 * 19 / 31)
    assert final == pytest.approx(cold + 6 * drama.final_cost(73))
    assert drama.estimate(ep, [(0, ep.shots[0])], "final") == (0, 0.0)
    _n, draft = drama.estimate(ep, shots, "draft")
    assert draft == pytest.approx(cold + drama.DRAFT_COST_RATIO * 6 * drama.final_cost(73))
    assert drama.DRAFT_COST_RATIO == pytest.approx(480 * 832 / (704 * 1280))


# --- keyframes --------------------------------------------------------------------------------


def test_keyframe_request_resolves_through_plan_picker():
    ep = drama.load_episode(EXAMPLE)
    req = drama.keyframe_request(ep, 1, ep.shots[1])  # xinyi, pose sitting_on_a_chair, wide
    assert req["trigger"] == "xinyi" and req["anchor_path"] == "/anchors/xinyi.png"
    assert req["pose_name"] == "sitting_on_a_chair"
    assert (req["width"], req["height"]) == (None, None)  # the skeleton's own canvas
    assert drama.FRAMINGS["wide"] in req["prompt"] and "window table" in req["prompt"]
    assert "cozy cafe interior" in req["prompt"]
    assert req["seed"] == 8110

    req = drama.keyframe_request(ep, 3, ep.shots[3])  # close-up, no pose
    assert (req["width"], req["height"]) == drama.KEYFRAME_SIZE
    assert drama.FRAMINGS["close"] in req["prompt"]


def test_chinese_keyframe_prompt_is_translated(tmp_path):
    data = example_data()
    data["shots"][1]["prompt"] = "坐在窗邊喝咖啡"
    ep = write_episode(tmp_path, data)
    req = drama.keyframe_request(ep, 1, ep.shots[1], translate=lambda t: "drinking coffee by the window")
    assert "drinking coffee by the window" in req["prompt"] and "坐" not in req["prompt"]

    def broken(_text):
        raise RuntimeError("offline")

    with pytest.raises(gc.UsageError, match="英文"):
        drama.keyframe_request(ep, 1, ep.shots[1], translate=broken)


def test_keyframe_request_refuses_a_character_without_anchor(monkeypatch):
    monkeypatch.setattr(gc, "picker_anchor_path", lambda trigger: None)
    ep = drama.load_episode(EXAMPLE)
    with pytest.raises(gc.UsageError, match="anchor"):
        drama.keyframe_request(ep, 1, ep.shots[1])


def test_run_keyframes_uses_gen_custom_with_the_episode_tier_and_skips_done_shots(tmp_path):
    ep = drama.load_episode(EXAMPLE)
    touch(ep.keyframe("s01"))
    calls = []

    def fake_gen(prompt, extra_negative, tier, trigger, anchor_path, out_dir, seed, filename, ip_weight, **kw):
        calls.append(
            dict(prompt=prompt, tier=tier, trigger=trigger, anchor=anchor_path, seed=seed, filename=filename, **kw)
        )
        touch(os.path.join(out_dir, f"{filename}.png"))

    run = Recorder()
    made = drama.run_keyframes(
        ep, drama.select_shots(ep, "s01,s02"), ffmpeg="ffmpeg", gen=fake_gen, run=run, server_up=lambda: True
    )
    assert made == 1 and len(calls) == 1
    call = calls[0]
    assert call["tier"] == "safe" and call["trigger"] == "xinyi" and call["filename"] == "s02_raw"
    assert call["checkpoint"] == "cyberrealistic_pony" and call["hq"] is True
    assert call["style_positive"] == gc.REALISTIC_STYLE and call["style_negative"] == gc.REALISTIC_NEGATIVE
    cmd, _ = run.calls[0]
    assert cmd[-1] == ep.keyframe("s02") and "crop=1080:1920" in " ".join(cmd)


def test_run_keyframes_needs_comfyui():
    ep = drama.load_episode(EXAMPLE)
    with pytest.raises(gc.UsageError, match="ComfyUI"):
        drama.run_keyframes(ep, drama.select_shots(ep), ffmpeg="ffmpeg", server_up=lambda: False)


# --- voice ------------------------------------------------------------------------------------


def test_voice_items_use_demo_voice_when_none_is_configured():
    ep = drama.load_episode(EXAMPLE)
    items, demo = drama.voice_items(ep, drama.select_shots(ep), allow_demo=True)
    assert demo is True
    assert [it["id"] for it in items] == ["s04", "s05", "s07", "s08"]
    first = items[0]
    assert first["text"] == "欣怡？真的是妳？"
    assert first["instruct"] and "驚訝" in first["instruct"] and first["instruct"].endswith("<|endofprompt|>")
    assert first["prompt_text"].startswith(drama.COSYVOICE_PROMPT_PREFIX)
    assert first["out"] == ep.voice("s04")


def test_configured_voice_is_used_and_needs_its_transcript(tmp_path):
    data = example_data()
    data["voices"] = {"taeoh": "taeoh_v1"}
    data["shots"][3].pop("emotion")
    ep = write_episode(tmp_path, data)
    vdir = os.path.join(drama.VOICES_DIR, "taeoh_v1")
    touch(os.path.join(vdir, "prompt.wav"))
    with open(os.path.join(vdir, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"prompt_wav": "prompt.wav", "prompt_text": "這是授權錄音的逐字稿。", "consent": CONSENT},
            f,
            ensure_ascii=False,
        )
    items, _ = drama.voice_items(ep, drama.select_shots(ep, "s04"))
    assert items[0]["prompt_wav"] == os.path.join(vdir, "prompt.wav")
    assert items[0]["prompt_text"].endswith("這是授權錄音的逐字稿。")
    assert items[0]["instruct"] is None  # no emotion -> zero-shot clone

    with open(os.path.join(vdir, "voice.json"), "w", encoding="utf-8") as f:
        json.dump({"prompt_wav": "prompt.wav"}, f)
    with pytest.raises(gc.UsageError, match="逐字稿"):
        drama.voice_items(ep, drama.select_shots(ep, "s04"))


def test_run_voice_refuses_while_comfyui_holds_the_gpu(monkeypatch):
    monkeypatch.setattr(drama, "cosyvoice_python", lambda: "/cosy/python")
    ep = drama.load_episode(EXAMPLE)
    with pytest.raises(gc.UsageError, match="stop_comfyui"):
        drama.run_voice(ep, drama.select_shots(ep), server_up=lambda: True, allow_demo=True)


def test_run_voice_writes_one_job_for_the_whole_episode(monkeypatch):
    monkeypatch.setattr(drama, "cosyvoice_python", lambda: "/cosy/python")
    ep = drama.load_episode(EXAMPLE)
    touch(ep.voice("s05"))

    def fake_run(cmd, **kwargs):
        with open(cmd[-1], encoding="utf-8") as f:
            job = json.load(f)
        for item in job["items"]:
            touch(item["out"])
        fake_run.job, fake_run.kwargs = job, kwargs

    done = drama.run_voice(ep, drama.select_shots(ep), run=fake_run, server_up=lambda: False, allow_demo=True)
    assert done == 3  # s05 already had audio
    assert [it["id"] for it in fake_run.job["items"]] == ["s04", "s07", "s08"]
    assert fake_run.kwargs["cwd"] == drama.COSYVOICE_DIR


def test_run_voice_needs_the_cosyvoice_venv(monkeypatch):
    monkeypatch.setattr(drama, "cosyvoice_python", lambda: None)
    ep = drama.load_episode(EXAMPLE)
    with pytest.raises(gc.UsageError, match="CosyVoice"):
        drama.run_voice(ep, drama.select_shots(ep), server_up=lambda: False, allow_demo=True)


# --- motion -----------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", drama.MOTION_MODES)
def test_motion_specs_are_valid_wan_jobs_carrying_the_tier(mode):
    ep = drama.load_episode(EXAMPLE)
    specs = drama.motion_specs(ep, drama.select_shots(ep), mode)
    assert [s.stem for s in specs] == [
        f"{sid}{drama.MOTION_SUFFIX[mode]}" for sid in ("s02", "s03", "s04", "s05", "s06", "s07")
    ]
    for spec in specs:
        p = spec.params
        gc.check_wan_params(
            p["width"], p["height"], p["frames"], p["fps"], p["steps"], p["cfg"], client.WAN_DEFAULT_MODEL
        )
        assert p["cfg"] >= client.SAFETY_MIN_CFG
        assert spec.tier == "safe" and spec.trigger in ("xinyi", "taeoh")
        assert spec.image_path.endswith("_wan.png")
        assert "safety" not in spec.prompt.lower()  # raw motion text; the worker adds negatives
    first = specs[0].params
    expected = {
        "preview": (480, 832, 12, 49),
        "draft": (480, 832, client.WAN_STEPS, 73),  # low-res, but the full length and steps
        "final": (704, 1280, client.WAN_STEPS, 73),
    }[mode]
    assert (first["width"], first["height"], first["steps"], first["frames"]) == expected


def test_motion_is_never_sent_without_confirmation(tmp_path):
    ep = drama.load_episode(EXAMPLE)
    for _i, shot in drama.select_shots(ep):
        touch(ep.keyframe(shot["id"]))
    sent = []
    with pytest.raises(gc.UsageError, match="--yes"):
        drama.run_motion(
            ep,
            drama.select_shots(ep),
            mode="preview",
            ffmpeg="ffmpeg",
            run=Recorder(),
            run_batch=lambda *a, **k: sent.append(a),
            stdin_isatty=False,
        )
    assert sent == []
    result = drama.run_motion(
        ep,
        drama.select_shots(ep),
        mode="preview",
        ffmpeg="ffmpeg",
        run=Recorder(),
        run_batch=lambda *a, **k: sent.append(a),
        confirm=lambda _p: "n",
        stdin_isatty=True,
    )
    assert result == [] and sent == []


def test_run_motion_batches_missing_shots_and_logs_cost(tmp_path):
    ep = drama.load_episode(EXAMPLE)
    for _i, shot in drama.select_shots(ep):
        touch(ep.keyframe(shot["id"]))
    touch(ep.motion("s02", "preview"))  # already drafted
    seen = {}

    def fake_batch(provider, job_type, specs, **kwargs):
        seen.update(provider=provider, job_type=job_type, specs=specs, **kwargs)
        out = []
        for i, spec in enumerate(specs):
            if i == 0:
                out.append((spec, cloud_video.CloudJobFailed("boom", job_id="j0")))
            else:
                out.append(
                    (
                        spec,
                        cloud_video.CloudResult(
                            path=f"/m/{spec.stem}.mp4",
                            provider=provider,
                            job_id=f"j{i}",
                            seed=spec.seed,
                            elapsed_s=1.0,
                            queue_s=3.0,
                            execution_s=36.0,
                        ),
                    )
                )
        return out

    run = Recorder()
    results = drama.run_motion(
        ep, drama.select_shots(ep), mode="preview", yes=True, ffmpeg="ffmpeg", run=run, run_batch=fake_batch
    )
    assert seen["provider"] == "runpod" and seen["job_type"] == "video_wan_i2v"
    assert [s.stem for s in seen["specs"]] == [
        "s03_preview",
        "s04_preview",
        "s05_preview",
        "s06_preview",
        "s07_preview",
    ]
    assert seen["out_dir"] == ep.sub("motion")
    assert len(run.calls) == 5 and all("crop=704:1280" in " ".join(c) for c, _ in run.calls)
    assert len(results) == 5
    with open(ep.sub("motion", "jobs.json"), encoding="utf-8") as f:
        log = json.load(f)
    assert log[0]["error"] and log[0]["job_id"] == "j0"
    assert log[1]["cost_floor"] == pytest.approx(36 / 3600 * drama.RUNPOD_PER_HOUR, abs=1e-4)


def test_run_motion_needs_keyframes_first():
    ep = drama.load_episode(EXAMPLE)
    with pytest.raises(gc.UsageError, match="keyframes"):
        drama.run_motion(
            ep,
            drama.select_shots(ep),
            mode="preview",
            yes=True,
            ffmpeg="ffmpeg",
            run=Recorder(),
            run_batch=lambda *a, **k: [],
        )


# --- assemble ---------------------------------------------------------------------------------


def test_shot_plan_prefers_final_clip_and_stretches_for_long_lines():
    ep = drama.load_episode(EXAMPLE)
    s04 = ep.shots[3]
    plan = drama.shot_plan(ep, s04)
    assert plan["video"] is None and plan["still"] == ep.keyframe("s04") and plan["duration"] == 3.0

    touch(ep.motion("s04", "preview"))
    assert drama.shot_plan(ep, s04)["video"] is None
    assert drama.shot_plan(ep, s04, allow_preview=True)["video"] == ep.motion("s04", "preview")
    touch(ep.motion("s04"))
    assert drama.shot_plan(ep, s04, allow_preview=True)["video"] == ep.motion("s04")

    write_wav(ep.voice("s04"), 4.0)
    plan = drama.shot_plan(ep, s04)
    assert plan["voice"] == ep.voice("s04")
    # 0.25 + 4.0 + 0.35 = 4.6 s is 110.4 frames; rounded up to 111 so picture and audio end together
    assert plan["duration"] == pytest.approx(111 / 24)


def test_run_assemble_cuts_every_shot_and_concatenates(tmp_path):
    ep = drama.load_episode(EXAMPLE)
    for shot in ep.shots:
        touch(ep.keyframe(shot["id"]))
    write_wav(ep.voice("s05"), 2.0)
    run = Recorder()
    out, total = drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=run)
    assert out == ep.output()
    assert len(run.calls) == len(ep.shots) + 4  # one per shot + s05's voice level + concat, measure, finish
    assert total == pytest.approx(sum(s["duration"] for s in ep.shots))
    with open(ep.text_file("s05", "line", 0), encoding="utf-8") as f:
        assert f.read() == "好久不見，你一點都沒變。"
    s05_cmd = " ".join(next(c for c, _ in run.calls if c[-1] == ep.segment("s05")))
    assert "drawtext" in s05_cmd and ep.voice("s05") in s05_cmd
    s01_cmd = " ".join(run.calls[0][0])
    assert "zoompan" in s01_cmd and "drawtext" not in s01_cmd and "anullsrc" in s01_cmd
    with open(ep.sub("segments", "concat.txt"), encoding="utf-8") as f:
        assert f.read().count("file '") == len(ep.shots)


def test_run_assemble_mixes_music_when_the_episode_has_it(tmp_path):
    data = example_data()
    bgm = tmp_path / "music.mp3"
    bgm.write_bytes(b"ID3")
    data["bgm"] = "music.mp3"
    data["bgm_volume"] = 0.1
    ep = write_episode(tmp_path, data)
    for shot in ep.shots:
        touch(ep.keyframe(shot["id"]))
    run = Recorder()
    drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=run)
    mix, measure, finish = (c for c, _ in run.calls[-3:])
    assert str(bgm) in mix and "volume=0.100" in " ".join(mix) and mix[-1] == ep.sub("segments", "mixed.mkv")
    assert ep.sub("segments", "mixed.mkv") in measure and "ebur128=peak=sample" in measure  # loudness of the mix
    assert finish[finish.index("-i") + 1] == ep.sub("segments", "mixed.mkv") and finish[-1] == ep.output()


def test_run_assemble_refuses_missing_pictures_and_missing_font():
    ep = drama.load_episode(EXAMPLE)
    with pytest.raises(gc.UsageError, match="關鍵幀"):
        drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=Recorder())
    for shot in ep.shots:
        touch(ep.keyframe(shot["id"]))
    with pytest.raises(gc.UsageError, match="字型"):
        drama.run_assemble(ep, ffmpeg="ffmpeg", font=None, run=Recorder())


def test_cli_surface():
    parser = drama.build_parser()
    args = parser.parse_args(["motion", EXAMPLE, "--preview", "--shots", "s02"])
    assert args.preview and not args.final and args.shots == "s02" and not args.yes
    with pytest.raises(SystemExit):
        parser.parse_args(["motion", EXAMPLE])  # --preview or --final is required
    with pytest.raises(SystemExit):
        parser.parse_args(["motion", EXAMPLE, "--preview", "--final"])


def test_plan_command_prints_estimate(capsys):
    assert drama.main(["plan", EXAMPLE]) == 0
    out = capsys.readouterr().out
    assert "s08" in out and "預覽 6 支" in out and "正式版 6 支" in out


# --- drama_compose ----------------------------------------------------------------------------


def test_wrap_subtitle_breaks_long_lines_at_punctuation():
    assert compose.wrap_subtitle("好久不見") == "好久不見"
    wrapped = compose.wrap_subtitle("這三年妳過得好嗎，我一直想問妳這句話，卻一直沒有機會")
    lines = wrapped.split("\n")
    assert all(len(line) <= compose.SUBTITLE_CHARS_PER_LINE for line in lines)
    assert lines[0].endswith("，")
    assert "".join(lines) == "這三年妳過得好嗎，我一直想問妳這句話，卻一直沒有機會"


@pytest.mark.parametrize("move", compose.CAMERA_MOVES)
def test_every_camera_move_has_expressions(move):
    z, x, y = compose.camera_expressions(move, 72)
    assert z and x and y
    if move != "static":
        assert "71" in (z + x)  # travels over the whole shot


def test_unknown_camera_move_is_rejected():
    with pytest.raises(ValueError):
        compose.camera_expressions("dolly_zoom", 72)


def test_segment_command_for_a_still_with_voice_and_subtitle():
    cmd = compose.segment_command(
        "ffmpeg",
        out="seg.mp4",
        duration=3.0,
        still="k.png",
        camera="pan_left",
        voice="v.wav",
        voice_delay=0.25,
        subtitle_lines=["C:\\subs\\s.txt"],
        font="C:\\Windows\\Fonts\\msjhbd.ttc",
    )
    joined = " ".join(cmd)
    assert cmd[cmd.index("-loop") + 1] == "1" and "k.png" in cmd
    assert "zoompan" in joined and "s=1080x1920" in joined
    assert "adelay=250|250" in joined and "atrim=0:3.000" in joined
    assert "textfile='C\\:/subs/s.txt'" in joined and "fontfile='C\\:/Windows/Fonts/msjhbd.ttc'" in joined
    assert cmd[-1] == "seg.mp4" and "libx264" in cmd and "pcm_s16le" in cmd and "aac" not in cmd


def test_segment_command_for_a_clip_holds_the_last_frame_and_adds_silence():
    cmd = compose.segment_command("ffmpeg", out="seg.mp4", duration=4.2, video="clip.mp4")
    joined = " ".join(cmd)
    assert "tpad=stop_mode=clone" in joined and "trim=duration=4.200" in joined
    assert "anullsrc=r=48000:cl=stereo" in joined and "drawtext" not in joined


def test_segment_command_argument_errors():
    with pytest.raises(ValueError):
        compose.segment_command("ffmpeg", out="o.mp4", duration=3, video="a.mp4", still="b.png")
    with pytest.raises(ValueError):
        compose.segment_command("ffmpeg", out="o.mp4", duration=3, still="b.png", subtitle_lines=["s.txt"])


def test_concat_list_quotes_paths():
    text = compose.concat_list(["C:\\a\\s01.mp4", "/tmp/it's.mp4"])
    assert text == "file 'C:/a/s01.mp4'\nfile '/tmp/it'\\''s.mp4'\n"


def test_concat_copies_video_and_encodes_aac_only_for_the_final_file():
    final = compose.concat_command("ffmpeg", "concat.txt", "out.mp4")
    assert final[final.index("-c:v") + 1] == "copy" and "aac" in final and "+faststart" in final
    body = compose.concat_command("ffmpeg", "concat.txt", "body.mkv", final=False)
    assert body[body.index("-c:a") + 1] == "copy" and "aac" not in body


@pytest.mark.parametrize("seconds, frames", [(2.5, 60), (3.0, 72), (4.6, 111), (4.8, 116)])
def test_frame_exact_rounds_up_to_whole_frames(seconds, frames):
    assert compose.frame_exact(seconds) * 24 == pytest.approx(frames)


def test_bgm_command_fades_out_and_keeps_video():
    cmd = compose.bgm_command("ffmpeg", "body.mp4", "m.mp3", "out.mp4", 20.0, 0.18)
    joined = " ".join(cmd)
    assert "-stream_loop -1" in joined and "volume=0.180" in joined and "afade=t=out:st=18.500" in joined
    assert cmd[cmd.index("-c:v") + 1] == "copy" and cmd[-1] == "out.mp4"


def test_find_ffmpeg_and_font_honour_the_environment(tmp_path):
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"")
    font = tmp_path / "font.ttc"
    font.write_bytes(b"")
    assert compose.find_ffmpeg({"DRAMA_FFMPEG": str(exe)}) == str(exe)
    assert compose.find_font({"DRAMA_FONT": str(font)}) == str(font)


def test_example_is_not_mutated_by_the_suite():
    before = example_data()
    ep = drama.load_episode(EXAMPLE)
    drama.validate_episode(ep)
    drama.motion_specs(ep, drama.select_shots(ep), "preview")
    assert example_data() == copy.deepcopy(before)


# --- language-learning episodes (JV Tutor Corner) ---------------------------------------------

JV = os.path.join(drama.EPISODES_DIR, "jv_en_ep01_cafe_order.json")


def jv_data():
    with open(JV, encoding="utf-8") as f:
        return json.load(f)


def test_jv_episode_is_valid_safe_adult_and_ends_on_the_platform_card():
    ep = drama.load_episode(JV)
    errors, _ = drama.validate_episode(ep)
    assert errors == []
    assert ep.tier == "safe"
    for shot in ep.shots:
        if shot.get("character"):
            assert gc.CHARACTERS[shot["character"]]["age"] >= gc.MINIMUM_AGE
    assert ep.shots[0]["type"] == "card"  # the hook
    assert ep.shots[-1]["type"] == "card" and "JV Tutor Corner" in ep.shots[-1]["label"]
    for shot in ep.shots:
        if shot["type"] == "dialogue":
            assert not compose.is_cjk_text(shot["line"]) and compose.is_cjk_text(shot["translation"])


def test_jv_plan_counts_only_moving_shots_for_the_cloud(capsys):
    assert drama.main(["plan", JV]) == 0
    out = capsys.readouterr().out
    assert "字卡" in out and "預覽 7 支" in out and "正式版 7 支" in out


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("phrase", "", "phrase"),
        ("phrase", "x" * 81, "phrase"),
        ("label", "很" * 21, "label"),
        ("background", "c08", "background"),  # another card has no keyframe
        ("background", "s99", "background"),
        ("speed", 2.0, "speed"),
        ("translation", "譯" * 121, "translation"),
    ],
)
def test_card_validation(tmp_path, key, value, message):
    data = jv_data()
    card = next(s for s in data["shots"] if s["id"] == "c05")
    card[key] = value
    errors, _ = drama.validate_episode(write_episode(tmp_path, data, name="jv_bad"))
    assert any(message in e for e in errors), errors


def test_cards_need_no_prompt_scene_or_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "picker_anchor_path", lambda trigger: None)
    errors, warnings = drama.validate_episode(drama.load_episode(JV))
    assert errors == []
    assert not any("c05" in w for w in warnings)  # a card's character is only its voice


def test_translation_without_a_line_is_flagged(tmp_path):
    data = jv_data()
    data["shots"][1]["translation"] = "沒有台詞的翻譯"
    _, warnings = drama.validate_episode(write_episode(tmp_path, data, name="jv_tr"))
    assert any("translation" in w for w in warnings)


@pytest.mark.parametrize(
    "line, transcript, emotion, mode",
    [
        ("Can I get a latte?", "希望你以后能够做的比我还好呦。", "", "cross_lingual"),
        ("Can I get a latte?", "希望你以后能够做的比我还好呦。", "開心", "cross_lingual"),
        ("可以給我拿鐵嗎？", "希望你以后能够做的比我还好呦。", "開心", "instruct"),
        ("可以給我拿鐵嗎？", "希望你以后能够做的比我还好呦。", "", "zero_shot"),
        ("Can I get a latte?", "This is my consented recording.", "", "zero_shot"),
    ],
)
def test_voice_mode(line, transcript, emotion, mode):
    assert drama.voice_mode(line, transcript, emotion) == mode


JV_COSYVOICE_VOICES = {"xinyi": "jv_customer_en", "taeoh": "jv_barista_en"}


def jv_cosyvoice_episode(tmp_path):
    """The JV episode as if it cloned consented recordings instead of using Kokoro's voices."""
    data = jv_data()
    data["voices"] = dict(JV_COSYVOICE_VOICES)
    return write_episode(tmp_path, data, name="jv_cosy")


def test_jv_voice_items_use_kokoro_voices_and_read_cards_slowly():
    ep = drama.load_episode(JV)
    items, demo = drama.voice_items(ep, drama.select_shots(ep))
    assert demo is False
    by_id = {it["id"]: it for it in items}
    assert set(by_id) == {"s03", "s04", "c05", "s06", "s07", "c08", "c09", "s11"}
    assert all(it["engine"] == drama.KOKORO for it in items)
    assert {by_id[i]["voice"] for i in ("s03", "s06", "c08")} == {"am_fenrir"}  # the barista
    assert {by_id[i]["voice"] for i in ("s04", "c05", "s07", "c09", "s11")} == {"af_heart"}  # the customer
    assert by_id["c05"]["speed"] == 0.85 and by_id["s03"]["speed"] == 1.0
    assert by_id["c05"]["text"] == "Can I get a medium latte, please?"


def test_jv_with_consented_cosyvoice_voices_clones_them_and_reads_cards_slowly(tmp_path):
    ep = jv_cosyvoice_episode(tmp_path)
    make_voice("jv_customer_en", "Hello, this is a sample of my voice for English lessons.")
    make_voice("jv_barista_en", "Good morning, this recording is for the cafe episode.")
    items, demo = drama.voice_items(ep, drama.select_shots(ep))
    assert demo is False
    by_id = {it["id"]: it for it in items}
    assert set(by_id) == {"s03", "s04", "c05", "s06", "s07", "c08", "c09", "s11"}
    assert all(it["engine"] == drama.COSYVOICE for it in items)
    assert all(it["mode"] == "zero_shot" and it["instruct"] is None for it in items)  # English voice, English line
    assert by_id["s03"]["prompt_wav"].endswith(os.path.join("jv_barista_en", "prompt.wav"))
    assert by_id["s04"]["prompt_wav"].endswith(os.path.join("jv_customer_en", "prompt.wav"))
    assert by_id["c05"]["speed"] == 0.85 and by_id["s03"]["speed"] == 1.0
    assert by_id["c05"]["text"] == "Can I get a medium latte, please?"


def test_keyframes_skip_cards():
    ep = drama.load_episode(JV)
    made = []

    def fake_gen(prompt, extra_negative, tier, trigger, anchor_path, out_dir, seed, filename, ip_weight, **kw):
        made.append(filename)
        touch(os.path.join(out_dir, f"{filename}.png"))

    drama.run_keyframes(
        ep, drama.select_shots(ep), ffmpeg="ffmpeg", gen=fake_gen, run=Recorder(), server_up=lambda: True
    )
    assert made == [f"{sid}_raw" for sid in ("s02", "s03", "s04", "s06", "s07", "s10", "s11")]


def test_jv_assemble_draws_cards_and_bilingual_subtitles():
    ep = drama.load_episode(JV)
    for shot in ep.shots:
        if shot["type"] != "card":
            touch(ep.keyframe(shot["id"]))
    run = Recorder()
    drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=run)
    assert len(run.calls) == len(ep.shots) + 3
    cmds = {os.path.basename(c[-1]): " ".join(c) for c, _ in run.calls}

    def text_files(sid):
        return [f for f in os.listdir(ep.sub("subs")) if f.startswith(f"{sid}_")]

    hook = cmds["c00.mkv"]
    assert "boxblur" in hook and ep.keyframe("s02") in hook
    assert hook.count("drawtext") == len(text_files("c00")) >= 3  # label, phrase line(s), translation
    assert not any("_note_" in f for f in text_files("c00"))
    card = cmds["c05.mkv"]
    assert card.count("drawtext") == len(text_files("c05")) == 4 and compose.TRANSLATION_COLOR in card
    dialogue = cmds["s04.mkv"]
    # "Can I get a medium latte, please?" is 33 characters: two subtitle lines, then the translation
    assert compose.TRANSLATION_COLOR in dialogue and dialogue.count("drawtext") == len(text_files("s04")) == 3
    with open(ep.text_file("s04", "tr", 0), encoding="utf-8") as f:
        assert f.read() == "可以給我一杯中杯拿鐵嗎？"
    with open(ep.text_file("c05", "phrase", 0), encoding="utf-8") as f:
        assert f.read() == "Can I get a ..., please?"


def test_card_background_keyframe_must_exist():
    ep = drama.load_episode(JV)
    for shot in ep.shots:
        if shot["type"] != "card" and shot["id"] != "s07":
            touch(ep.keyframe(shot["id"]))
    with pytest.raises(gc.UsageError) as info:
        drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=Recorder())
    assert "s07" in str(info.value) and "c09" in str(info.value)


def test_wrap_lines_wraps_english_between_words():
    lines = compose.wrap_lines("To go, please. Could I have it with oat milk?")
    assert lines == ["To go, please. Could I have it", "with oat milk?"]
    assert compose.wrap_lines("") == []
    assert compose.wrap_lines("好久不見") == ["好久不見"]


def test_subtitle_filters_put_the_translation_under_the_line():
    filters = compose.subtitle_filters(["a0.txt", "a1.txt"], "f.ttc", ["t0.txt"])
    assert len(filters) == 3
    ys = {f.split("textfile='")[1].split("'")[0]: int(f.rsplit(":y=", 1)[1]) for f in filters}
    assert ys["a0.txt"] < ys["a1.txt"] < ys["t0.txt"]
    assert ys["t0.txt"] + compose.TRANSLATION_FONT_SIZE == compose.SUBTITLE_BOTTOM
    assert all("x=(w-text_w)/2" in f for f in filters)  # every line centred on its own width


def test_card_filters_centre_the_stack_in_the_band():
    blocks = [("label", ["l.txt"]), ("phrase", ["p0.txt", "p1.txt"]), ("translation", []), ("note", ["n.txt"])]
    filters = compose.card_filters(blocks, "f.ttc")
    assert len(filters) == 4
    tops = [int(f.rsplit(":y=", 1)[1]) for f in filters]
    assert tops == sorted(tops)
    band_top, band_bottom = compose.CARD_BAND
    last_bottom = tops[-1] + compose.CARD_BLOCKS["note"][0]
    assert abs((tops[0] - band_top) - (band_bottom - last_bottom)) <= 1


def test_card_command_plain_background_and_font_required():
    cmd = compose.card_command("ffmpeg", out="c.mkv", duration=3.0, blocks=[("phrase", ["p.txt"])], font="f.ttc")
    joined = " ".join(cmd)
    assert "color=c=0x16202b" in joined and "boxblur" not in joined and "anullsrc" in joined
    assert "pcm_s16le" in cmd
    with pytest.raises(ValueError):
        compose.card_command("ffmpeg", out="c.mkv", duration=3.0, blocks=[], font=None)


# --- low resolution first, external enhancement later ----------------------------------------


def test_draft_keyframes_are_smaller_and_keep_the_skeleton_aspect():
    ep = drama.load_episode(EXAMPLE)
    req = drama.keyframe_request(ep, 3, ep.shots[3], draft=True)  # close-up, no pose
    assert (req["width"], req["height"]) == drama.KEYFRAME_DRAFT_SIZE
    req = drama.keyframe_request(ep, 1, ep.shots[1], draft=True)  # sitting_on_a_chair skeleton
    assert (req["width"], req["height"]) == (624, 912)
    import pose_skeletons

    pose_skeletons.check_canvas(req["pose_name"], req["width"], req["height"])  # no ControlNet crop


def test_draft_keyframes_skip_the_face_pass_and_are_redone_by_a_final_run():
    ep = drama.load_episode(EXAMPLE)
    calls = []

    def fake_gen(prompt, extra_negative, tier, trigger, anchor_path, out_dir, seed, filename, ip_weight, **kw):
        calls.append((filename, kw["use_facedetailer"], kw["width"]))
        touch(os.path.join(out_dir, f"{filename}.png"))

    shots = drama.select_shots(ep, "s02,s04")
    drama.run_keyframes(ep, shots, draft=True, ffmpeg="ffmpeg", gen=fake_gen, run=touch_last, server_up=lambda: True)
    assert calls == [("s02_raw", False, 624), ("s04_raw", False, 576)]
    assert os.path.isfile(ep.keyframe_draft_marker("s02"))

    calls.clear()
    drama.run_keyframes(ep, shots, draft=True, ffmpeg="ffmpeg", gen=fake_gen, run=touch_last, server_up=lambda: True)
    assert calls == []  # drafts exist; another draft run leaves them

    drama.run_keyframes(ep, shots, ffmpeg="ffmpeg", gen=fake_gen, run=touch_last, server_up=lambda: True)
    assert calls == [("s02_raw", None, None), ("s04_raw", None, 768)]  # the final run redoes drafts
    assert not os.path.isfile(ep.keyframe_draft_marker("s02"))

    calls.clear()
    drama.run_keyframes(ep, shots, draft=True, ffmpeg="ffmpeg", gen=fake_gen, run=touch_last, server_up=lambda: True)
    assert calls == []  # a draft run never overwrites a finished keyframe


def touch_last(cmd, **kwargs):
    touch(cmd[-1])


def test_assembly_prefers_enhanced_then_final_then_draft():
    ep = drama.load_episode(EXAMPLE)
    s04 = ep.shots[3]
    touch(ep.motion("s04", "draft"))
    plan = drama.shot_plan(ep, s04)
    assert plan["video"] == ep.motion("s04", "draft") and plan["source"] == "低解析度版"
    touch(ep.motion("s04"))
    assert drama.shot_plan(ep, s04)["video"] == ep.motion("s04")
    touch(ep.enhanced("s04", "mp4"))
    plan = drama.shot_plan(ep, s04)
    assert plan["video"] == ep.enhanced("s04", "mp4") and plan["source"] == "增強版"

    s01 = ep.shots[0]
    assert drama.shot_plan(ep, s01)["still"] == ep.keyframe("s01")
    touch(ep.enhanced("s01", "png"))
    assert drama.shot_plan(ep, s01)["still"] == ep.enhanced("s01", "png")


def test_assemble_at_48_fps_renders_every_segment_at_48():
    ep = drama.load_episode(EXAMPLE)
    for shot in ep.shots:
        touch(ep.keyframe(shot["id"]))
    run = Recorder()
    _out, total = drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=run, fps=48)
    for cmd, _ in run.calls[: len(ep.shots)]:
        joined = " ".join(cmd)
        assert cmd[cmd.index("-r") + 1] == "48" and "fps=48" in joined
    plan = drama.shot_plan(ep, ep.shots[0], fps=48)
    assert plan["duration"] * 48 == pytest.approx(round(plan["duration"] * 48))


def test_episode_fps_is_validated_and_used(tmp_path):
    data = example_data()
    data["fps"] = 60
    ep = write_episode(tmp_path, data)
    assert drama.validate_episode(ep)[0] == [] and ep.fps == 60
    data["fps"] = 29
    errors, _ = drama.validate_episode(write_episode(tmp_path, data, name="ep_bad_fps"))
    assert any("fps" in e for e in errors)


def test_export_copies_the_best_picture_per_shot_and_explains_the_round_trip():
    ep = drama.load_episode(JV)
    touch(ep.motion("s02", "draft"), b"draft")
    touch(ep.motion("s03"), b"final")
    touch(ep.motion("s03", "draft"), b"draft")
    touch(ep.keyframe("s04"), b"png")
    manifest = drama.run_export(ep)
    by_id = {m["id"]: m for m in manifest}
    assert set(by_id) == {"s02", "s03", "s04"}  # cards are drawn at assembly; the rest have nothing yet
    assert by_id["s02"]["from"] == "draft" and by_id["s03"]["from"] == "final"
    assert by_id["s04"]["from"] == "keyframe" and by_id["s04"]["file"] == "s04.png"
    with open(ep.sub("enhance", "in", "s03.mp4"), "rb") as f:
        assert f.read() == b"final"
    with open(ep.sub("enhance", "README.txt"), encoding="utf-8") as f:
        readme = f.read()
    assert "out/" in readme and "--fps 48" in readme
    assert os.path.isdir(ep.sub("enhance", "out"))


def test_cli_accepts_the_draft_modes_and_fps():
    parser = drama.build_parser()
    assert parser.parse_args(["motion", EXAMPLE, "--draft"]).draft
    assert parser.parse_args(["keyframes", EXAMPLE, "--draft"]).draft
    assert parser.parse_args(["assemble", EXAMPLE, "--fps", "48"]).fps == 48
    with pytest.raises(SystemExit):
        parser.parse_args(["assemble", EXAMPLE, "--fps", "29"])
    assert parser.parse_args(["export", EXAMPLE]).cmd == "export"


# --- licensed components only ----------------------------------------------------------------


def test_demo_voice_is_refused_unless_asked_for():
    ep = drama.load_episode(EXAMPLE)
    with pytest.raises(gc.UsageError, match="--allow-demo"):
        drama.voice_items(ep, drama.select_shots(ep, "s04"))


def test_commercial_episodes_never_use_the_demo_voice(tmp_path):
    data = jv_data()
    data["voices"] = {}
    ep = write_episode(tmp_path, data, name="jv_novoice")
    errors, _ = drama.validate_episode(ep)
    assert any("已授權的聲音" in e for e in errors)
    with pytest.raises(gc.UsageError, match="commercial"):
        drama.voice_items(ep, drama.select_shots(ep, "s03"), allow_demo=True)


def test_a_voice_without_a_consent_record_is_refused(tmp_path):
    ep = jv_cosyvoice_episode(tmp_path)
    make_voice("jv_barista_en", "Good morning.", consent=None)
    with pytest.raises(gc.UsageError, match="授權紀錄"):
        drama.voice_items(ep, drama.select_shots(ep, "s03"))
    make_voice("jv_barista_en", "Good morning.", consent={"speaker": "", "date": "2026-09-27"})
    with pytest.raises(gc.UsageError, match="授權紀錄"):
        drama.voice_items(ep, drama.select_shots(ep, "s03"))


@pytest.mark.parametrize("checkpoint", ["cyberrealistic_pony", "juggernaut", "pony"])
def test_commercial_episodes_refuse_checkpoints_that_forbid_paid_use(tmp_path, checkpoint):
    data = jv_data()
    data["checkpoint"] = checkpoint
    errors, _ = drama.validate_episode(write_episode(tmp_path, data, name="jv_ckpt"))
    assert any("不允許營利" in e for e in errors)


def test_commercial_flag_must_be_a_boolean(tmp_path):
    data = jv_data()
    data["commercial"] = "yes"
    errors, _ = drama.validate_episode(write_episode(tmp_path, data, name="jv_flag"))
    assert any("commercial" in e for e in errors)


def test_jv_episode_is_commercial_on_licensed_components():
    ep = drama.load_episode(JV)
    assert ep.data["commercial"] is True
    assert ep.checkpoint in drama.COMMERCIAL_CHECKPOINTS
    assert set(ep.data["voices"]) == {"xinyi", "taeoh"}
    assert all(drama.is_kokoro(spec) for spec in ep.data["voices"].values())


def test_voice_status_reports_what_is_missing(tmp_path):
    ep = jv_cosyvoice_episode(tmp_path)
    make_voice("jv_customer_en", "Hello there.")
    status = {voice_id: (who, state) for voice_id, who, state in drama.voice_status(ep)}
    assert status["jv_customer_en"] == (["xinyi"], "已設定、有授權紀錄")
    assert status["jv_barista_en"][0] == ["taeoh"] and "找不到聲音設定" in status["jv_barista_en"][1]


def test_interpolate_doubles_the_frame_rate_from_the_best_full_length_clip():
    ep = drama.load_episode(JV)
    touch(ep.motion("s02", "draft"), b"draft")
    touch(ep.motion("s03"), b"final")
    touch(ep.motion("s03", "draft"), b"draft")
    touch(ep.motion("s04", "preview"), b"preview")  # a preview is shorter than the shot: never used
    touch(ep.enhanced("s06", "mp4"), b"done")  # already enhanced: skipped
    uploads, submits = [], []

    def fake_upload(path):
        uploads.append(os.path.basename(path))
        return os.path.basename(path)

    def fake_submit(video_filename, filename_prefix, multiplier, fps):
        submits.append((video_filename, multiplier, fps))
        out = os.path.join(os.path.dirname(ep.sub("x")), f"{filename_prefix}.mp4")
        touch(out, b"rife")
        return out

    done = drama.run_interpolate(
        ep, drama.select_shots(ep), submit=fake_submit, upload=fake_upload, server_up=lambda: True
    )
    assert done == 2
    assert uploads == ["jv_en_ep01_cafe_order_s02_draft.mp4", "jv_en_ep01_cafe_order_s03.mp4"]
    assert all(m == 2 and f == 24 for _v, m, f in submits)
    with open(ep.enhanced("s03", "mp4"), "rb") as f:
        assert f.read() == b"rife"
    assert not os.path.isfile(ep.enhanced("s04", "mp4"))


def test_interpolate_needs_comfyui_only_when_there_is_work():
    ep = drama.load_episode(JV)
    assert drama.run_interpolate(ep, drama.select_shots(ep), server_up=lambda: False) == 0
    touch(ep.motion("s02", "draft"))
    with pytest.raises(gc.UsageError, match="ComfyUI"):
        drama.run_interpolate(ep, drama.select_shots(ep), server_up=lambda: False)


def test_cli_voice_demo_flag_and_interpolate():
    parser = drama.build_parser()
    assert parser.parse_args(["voice", EXAMPLE, "--allow-demo"]).allow_demo
    assert not parser.parse_args(["voice", EXAMPLE]).allow_demo
    assert parser.parse_args(["interpolate", JV, "--shots", "s02"]).cmd == "interpolate"


# --- per-episode cast overrides --------------------------------------------------------------


@pytest.mark.parametrize(
    "cast, message",
    [
        ({"nobody": {"style": "apron"}}, "角色表"),
        ({"taeoh": {"age": 30}}, "年齡和性別不能改"),
        ({"taeoh": {"style": ""}}, "文字"),
        ({"taeoh": {"style": "x" * 201}}, "文字"),
        ({"taeoh": {}}, "物件"),
        (["taeoh"], "物件"),
    ],
)
def test_cast_validation(tmp_path, cast, message):
    data = jv_data()
    data["cast"] = cast
    errors, _ = drama.validate_episode(write_episode(tmp_path, data, name="jv_cast"))
    assert any(message in e for e in errors), errors


def test_keyframes_carry_the_episode_wardrobe():
    ep = drama.load_episode(JV)
    seen = {}

    def fake_gen(prompt, extra_negative, tier, trigger, anchor_path, out_dir, seed, filename, ip_weight, **kw):
        seen[filename] = (trigger, kw["character_overrides"])
        touch(os.path.join(out_dir, f"{filename}.png"))

    drama.run_keyframes(
        ep,
        drama.select_shots(ep, "s03,s04"),
        draft=True,
        ffmpeg="ffmpeg",
        gen=fake_gen,
        run=touch_last,
        server_up=lambda: True,
    )
    assert seen["s03_raw"] == ("taeoh", {"style": "crisp white shirt under a dark brown barista apron"})
    assert seen["s04_raw"][1]["appearance"].startswith("shoulder-length wavy dark chestnut")


def test_motion_prompts_carry_the_wardrobe_instead_of_the_character_id():
    ep = drama.load_episode(JV)
    specs = {s.stem: s for s in drama.motion_specs(ep, drama.select_shots(ep), "draft")}
    barista = specs["s03_draft"]
    assert barista.trigger is None  # the worker would add the default outfit back
    assert barista.prompt.startswith("22 year old adult man") and "barista apron" in barista.prompt
    assert "taeoh" not in barista.prompt
    assert "dark chestnut brown hair" in specs["s04_draft"].prompt


# --- candidate seeds and pick ----------------------------------------------------------------


def test_candidates_render_the_next_seeds_without_touching_the_keyframe(tmp_path):
    ep = write_episode(tmp_path, jv_data(), name="jv_cand")
    touch(ep.keyframe("s03"), b"current")
    rendered = []

    def fake_gen(prompt, extra_negative, tier, trigger, anchor_path, out_dir, seed, filename, ip_weight, **kw):
        rendered.append((seed, filename))
        touch(os.path.join(out_dir, f"{filename}.png"))

    made = drama.run_keyframes(
        ep,
        drama.select_shots(ep, "s03"),
        draft=True,
        candidates=3,
        ffmpeg="ffmpeg",
        gen=fake_gen,
        run=touch_last,
        server_up=lambda: True,
    )
    base = ep.shot_seed(2, ep.shots[2])
    assert made == 3 and [seed for seed, _ in rendered] == [base + 1, base + 2, base + 3]
    for seed in (base + 1, base + 2, base + 3):
        assert os.path.isfile(ep.candidate("s03", seed))
        assert os.path.isfile(ep.candidate_draft_marker("s03", seed))
    with open(ep.keyframe("s03"), "rb") as f:
        assert f.read() == b"current"
    rendered.clear()
    drama.run_keyframes(
        ep,
        drama.select_shots(ep, "s03"),
        draft=True,
        candidates=3,
        ffmpeg="ffmpeg",
        gen=fake_gen,
        run=touch_last,
        server_up=lambda: True,
    )
    assert rendered == []  # existing candidates are kept


def test_candidates_count_is_bounded():
    ep = drama.load_episode(JV)
    with pytest.raises(gc.UsageError, match="candidates"):
        drama.run_keyframes(ep, drama.select_shots(ep, "s03"), candidates=13, ffmpeg="ffmpeg", server_up=lambda: True)


def test_pick_promotes_a_candidate_and_records_the_seed(tmp_path):
    ep = write_episode(tmp_path, jv_data(), name="jv_pick")
    touch(ep.candidate("s03", 9333), b"candidate")
    touch(ep.candidate_draft_marker("s03", 9333))
    touch(ep.motion("s03", "draft"))
    drama.run_pick(ep, "s03", 9333)
    with open(ep.keyframe("s03"), "rb") as f:
        assert f.read() == b"candidate"
    assert os.path.isfile(ep.keyframe_draft_marker("s03"))
    reloaded = drama.load_episode(ep.path)
    assert next(s for s in reloaded.shots if s["id"] == "s03")["seed"] == 9333
    assert drama.validate_episode(reloaded)[0] == []
    assert reloaded.shot_seed(2, reloaded.shots[2]) == 9333


def test_pick_refuses_unknown_shots_cards_and_missing_candidates(tmp_path):
    ep = write_episode(tmp_path, jv_data(), name="jv_pick_bad")
    with pytest.raises(gc.UsageError, match="c05"):
        drama.run_pick(ep, "c05", 1)
    with pytest.raises(gc.UsageError, match="candidates"):
        drama.run_pick(ep, "s03", 1)


def test_cli_candidates_and_pick():
    parser = drama.build_parser()
    assert parser.parse_args(["keyframes", JV, "--shots", "s03", "--candidates", "6"]).candidates == 6
    args = parser.parse_args(["pick", JV, "--shot", "s03", "--seed", "9323"])
    assert (args.cmd, args.shot, args.seed) == ("pick", "s03", 9323)


# --- Kokoro built-in voices ------------------------------------------------------------------


def test_kokoro_voice_list_is_english_only_and_has_the_jv_voices():
    assert {"af_heart", "am_michael"} <= drama.KOKORO_VOICES
    assert all(re.fullmatch(r"[ab][fm]_[a-z]+", v) for v in drama.KOKORO_VOICES)


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"engine": "kokoro", "voice": "zf_xiaobei"}, "沒有「zf_xiaobei」"),  # Mandarin G2P is not installed
        ({"engine": "kokoro", "voice": "af_heart", "speed": 0.9}, "engine"),
        ({"engine": "edge", "voice": "af_heart"}, "engine"),
        ({"voice": "af_heart"}, "engine"),
        (3, "聲音要是"),
        ("Bad Id", "小寫英數字"),
    ],
)
def test_voice_entries_are_a_cosyvoice_id_or_an_inline_kokoro_voice(tmp_path, spec, message):
    data = jv_data()
    data["voices"]["taeoh"] = spec
    errors, _ = drama.validate_episode(write_episode(tmp_path, data, name="jv_voice"))
    assert any(e.startswith("voices.taeoh") and message in e for e in errors), errors
    data = jv_data()
    data["shots"][2]["voice"] = spec
    errors, _ = drama.validate_episode(write_episode(tmp_path, data, name="jv_voice"))
    assert any("s03 的 voice" in e and message in e for e in errors), errors


def test_a_shot_can_switch_to_another_kokoro_voice(tmp_path):
    data = jv_data()
    data["shots"][2]["voice"] = {"engine": "kokoro", "voice": "bm_george"}
    ep = write_episode(tmp_path, data, name="jv_shot_voice")
    assert drama.validate_episode(ep)[0] == []
    items, _ = drama.voice_items(ep, drama.select_shots(ep, "s03,s06"))
    assert [it["voice"] for it in items] == ["bm_george", "am_fenrir"]


def test_kokoro_ignores_emotion_and_says_so(tmp_path, capsys):
    data = jv_data()
    data["shots"][2]["emotion"] = "開心"
    ep = write_episode(tmp_path, data, name="jv_emotion")
    items, _ = drama.voice_items(ep, drama.select_shots(ep, "s03"))
    assert "語氣「開心」不會套用" in capsys.readouterr().out
    assert set(items[0]) == {"id", "text", "out", "engine", "voice", "speed"}


def test_resolve_voice_refuses_a_kokoro_voice():
    ep = drama.load_episode(JV)
    with pytest.raises(gc.UsageError, match="Kokoro"):
        drama.resolve_voice(ep, ep.shots[2])


def test_run_voice_with_kokoro_runs_on_the_cpu_even_while_comfyui_is_up(monkeypatch):
    monkeypatch.setattr(drama, "kokoro_python", lambda: "/kokoro/python")
    monkeypatch.setattr(drama, "cosyvoice_python", lambda: None)  # not needed at all
    ep = drama.load_episode(JV)
    touch(ep.voice("s04"))

    def fake_run(cmd, **kwargs):
        with open(cmd[-1], encoding="utf-8") as f:
            job = json.load(f)
        for item in job["items"]:
            touch(item["out"])
        fake_run.cmd, fake_run.job, fake_run.kwargs = cmd, job, kwargs

    done = drama.run_voice(ep, drama.select_shots(ep), run=fake_run, server_up=lambda: True)
    assert done == 7  # s04 already had audio
    assert fake_run.cmd[:2] == ["/kokoro/python", drama.KOKORO_RUNNER]
    assert os.path.basename(fake_run.cmd[-1]) == "kokoro_job.json"
    assert fake_run.job["repo_id"] == drama.KOKORO_REPO
    assert "s04" not in [it["id"] for it in fake_run.job["items"]]
    assert fake_run.kwargs["cwd"] == drama.KOKORO_DIR
    assert fake_run.kwargs["env"]["HF_HOME"] == os.path.join(drama.KOKORO_DIR, "hf_home")
    assert fake_run.kwargs["env"]["PYTHONIOENCODING"] == "utf-8"


def test_run_voice_needs_the_kokoro_venv(monkeypatch):
    monkeypatch.setattr(drama, "kokoro_python", lambda: None)
    ep = drama.load_episode(JV)
    with pytest.raises(gc.UsageError, match="Kokoro"):
        drama.run_voice(ep, drama.select_shots(ep), run=Recorder(), server_up=lambda: False)


def test_mixed_voices_check_both_environments_before_speaking_anything(tmp_path, monkeypatch):
    data = jv_data()
    data["voices"]["taeoh"] = "jv_barista_en"
    ep = write_episode(tmp_path, data, name="jv_mixed")
    make_voice("jv_barista_en", "Good morning, this recording is for the cafe episode.")
    monkeypatch.setattr(drama, "kokoro_python", lambda: "/kokoro/python")
    monkeypatch.setattr(drama, "cosyvoice_python", lambda: None)
    run = Recorder()
    with pytest.raises(gc.UsageError, match="CosyVoice"):
        drama.run_voice(ep, drama.select_shots(ep), run=run, server_up=lambda: False)
    assert run.calls == []  # Kokoro did not run first and leave half the episode voiced

    monkeypatch.setattr(drama, "cosyvoice_python", lambda: "/cosy/python")
    calls = []

    def fake_run(cmd, **kwargs):
        with open(cmd[-1], encoding="utf-8") as f:
            job = json.load(f)
        calls.append((cmd[0], sorted(it["id"] for it in job["items"])))
        for item in job["items"]:
            touch(item["out"])

    assert drama.run_voice(ep, drama.select_shots(ep), run=fake_run, server_up=lambda: False) == 8
    assert calls == [("/kokoro/python", ["c05", "c09", "s04", "s07", "s11"]), ("/cosy/python", ["c08", "s03", "s06"])]


def test_voice_status_names_kokoro_voices(monkeypatch):
    monkeypatch.setattr(drama, "kokoro_python", lambda: "/kokoro/python")
    status = {label: (who, state) for label, who, state in drama.voice_status(drama.load_episode(JV))}
    assert status["kokoro:af_heart"] == (["xinyi"], "Kokoro 內建聲音（Apache-2.0，不需要錄音授權）")
    assert status["kokoro:am_fenrir"][0] == ["taeoh"]
    monkeypatch.setattr(drama, "kokoro_python", lambda: None)
    status = {label: state for label, _, state in drama.voice_status(drama.load_episode(JV))}
    assert "還沒安裝" in status["kokoro:af_heart"]


# --- loudness -------------------------------------------------------------------------------

EBUR128_LOG = """[Parsed_ebur128_0 @ 000001] t: 38.9 TARGET:-23 LUFS M: -70.0 S: -38.1 I: -25.9 LUFS LRA: 4.4 LU
[Parsed_ebur128_0 @ 000001] Summary:

  Integrated loudness:
    I:         -25.6 LUFS
    Threshold: -36.0 LUFS

  Loudness range:
    LRA:         4.5 LU
"""


def test_integrated_loudness_reads_the_summary_not_the_running_value():
    assert compose.integrated_loudness(EBUR128_LOG) == -25.6
    assert compose.integrated_loudness("no summary here") is None


@pytest.mark.parametrize("measured, gain", [(-25.6, "11.60"), (-10.0, "-4.00"), (-14.0, "0.00")])
def test_loudness_filter_is_one_fixed_gain_and_a_limiter(measured, gain):
    assert compose.loudness_filter(measured) == f"volume={gain}dB,alimiter=limit=0.84:level=false"


@pytest.mark.parametrize("measured", [None, -70.0])
def test_silence_or_no_measurement_is_left_alone(measured):
    assert compose.loudness_filter(measured) is None
    cmd = compose.finish_command("ffmpeg", "body.mkv", "out.mp4", None)
    assert "-af" not in cmd and cmd[cmd.index("-c:v") + 1] == "copy" and "aac" in cmd and cmd[-1] == "out.mp4"


def test_assemble_measures_the_mix_and_raises_it_to_the_target():
    ep = drama.load_episode(JV)
    for shot in ep.shots:
        if shot["type"] != "card":
            touch(ep.keyframe(shot["id"]))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "ebur128=peak=sample" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=EBUR128_LOG)
        return subprocess.CompletedProcess(cmd, 0)

    drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=fake_run)
    concat, measure, finish = (c for c, _ in calls[-3:])
    assert concat[-1] == ep.sub("segments", "body.mkv") and "-c:a" in concat and "aac" not in concat
    assert calls[-2][1]["capture_output"] is True and ep.sub("segments", "body.mkv") in measure
    assert finish[finish.index("-af") + 1] == "volume=11.60dB,alimiter=limit=0.84:level=false"
    assert finish[finish.index("-c:v") + 1] == "copy" and finish[-1] == ep.output()


# --- per-line voice levelling -----------------------------------------------------------------


def ebur128_log(integrated, peak):
    return (
        f"[Parsed_ebur128_0 @ 000001] Summary:\n\n  Integrated loudness:\n    I:         {integrated} LUFS\n"
        f"    Threshold: -33.0 LUFS\n\n  Sample peak:\n    Peak:       {peak} dBFS\n"
    )


def test_sample_peak_reads_the_summary():
    assert compose.sample_peak(ebur128_log(-21.0, -5.2)) == -5.2
    assert compose.sample_peak(ebur128_log(-70.0, "-inf")) == float("-inf")
    assert compose.sample_peak(EBUR128_LOG) is None  # measured without peak=sample


@pytest.mark.parametrize(
    "measured, peak, gain",
    [
        (-25.7, -7.0, 2.7),  # af_heart: raised to the reference
        (-21.0, -5.0, -2.0),  # am_fenrir: brought down to it
        (-35.0, -3.0, 2.0),  # a quiet line with a loud peak: raised only up to the -1 dBFS ceiling
        (-23.0, None, 0.0),
        (-70.0, "-inf", None),  # silence
        (None, None, None),
    ],
)
def test_line_gain_levels_to_the_reference_without_clipping(measured, peak, gain):
    peak = float(peak) if peak is not None else None
    assert compose.line_gain(measured, peak) == gain


def test_the_voice_gain_goes_in_front_of_the_voice_chain_only_when_given():
    levelled = compose.segment_command(
        "ffmpeg", out="o.mkv", duration=3, still="k.png", voice="v.wav", voice_gain_db=-2.04
    )
    graph = levelled[levelled.index("-filter_complex") + 1]
    assert "[1:a]volume=-2.04dB,aresample=48000" in graph
    plain = compose.segment_command("ffmpeg", out="o.mkv", duration=3, still="k.png", voice="v.wav")
    assert "volume=" not in plain[plain.index("-filter_complex") + 1]
    card = compose.card_command(
        "ffmpeg", out="c.mkv", duration=3, blocks=[], font="f.ttc", voice="v.wav", voice_gain_db=2.7
    )
    assert "[1:a]volume=2.70dB," in card[card.index("-filter_complex") + 1]


def test_assemble_levels_each_line_before_the_mix():
    ep = drama.load_episode(JV)
    for shot in ep.shots:
        if shot["type"] != "card":
            touch(ep.keyframe(shot["id"]))
    write_wav(ep.voice("s03"), 2.0)  # the barista, louder
    write_wav(ep.voice("s04"), 2.0)  # the customer, quieter
    levels = {ep.voice("s03"): ebur128_log(-21.0, -5.0), ep.voice("s04"): ebur128_log(-25.7, -8.0)}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=levels.get(cmd[cmd.index("-i") + 1], ""))

    drama.run_assemble(ep, ffmpeg="ffmpeg", font="C:/font.ttc", run=fake_run)
    segments = {os.path.basename(c[-1]): c[c.index("-filter_complex") + 1] for c in calls if "-filter_complex" in c}
    assert "[1:a]volume=-2.00dB," in segments["s03.mkv"]
    assert "[1:a]volume=2.70dB," in segments["s04.mkv"]
    assert "volume=" not in segments["s02.mkv"]  # no line, nothing to level


# --- the JV Tutor Corner series: one episode per course --------------------------------------

JV_EPISODES = sorted(glob.glob(os.path.join(drama.EPISODES_DIR, "jv_*.json")))


def test_the_series_covers_the_platforms_language_courses():
    names = {os.path.basename(p)[: -len(".json")] for p in JV_EPISODES}
    assert {
        "jv_en_ep01_cafe_order",
        "jv_en_ep02_gept_speaking",  # c1 英檢中級衝刺班
        "jv_en_ep03_business_meeting",  # c3 商用英語會議表達技巧
        "jv_ja_ep01_ramen_order",  # c4 旅遊日文
    } <= names


@pytest.mark.parametrize("path", JV_EPISODES, ids=os.path.basename)
def test_every_jv_episode_is_a_valid_commercial_lesson(path):
    ep = drama.load_episode(path)
    errors, _ = drama.validate_episode(ep)
    assert errors == []
    assert ep.data["commercial"] is True and ep.checkpoint in drama.COMMERCIAL_CHECKPOINTS and ep.tier == "safe"
    for shot in ep.shots:
        if shot.get("character"):
            assert gc.CHARACTERS[shot["character"]]["age"] >= gc.MINIMUM_AGE
    assert ep.shots[0]["type"] == "card" and not ep.shots[0].get("line")  # the hook
    assert ep.shots[-1]["type"] == "card" and ep.shots[-1]["label"] == "JV Tutor Corner"
    phrases = [s for s in ep.shots if s["type"] == "card" and s["label"].startswith("今日句型")]
    assert len(phrases) == 3 and all(card["line"] and card["speed"] < 1 for card in phrases)  # read slowly
    japanese = os.path.basename(path).startswith("jv_ja_")
    for shot in ep.shots:
        if shot["type"] == "dialogue":
            assert compose.is_cjk_text(shot["translation"])  # the Chinese subtitle under every line
            has_kana = any("\u3040" <= ch <= "\u30ff" for ch in shot["line"])
            assert has_kana if japanese else not compose.is_cjk_text(shot["line"])


def test_jv_characters_keep_one_voice_per_language_and_episodes_have_their_own_seeds():
    voices, seeds = {}, set()
    for path in JV_EPISODES:
        ep = drama.load_episode(path)
        language = os.path.basename(path).split("_")[1]
        for character, spec in ep.data["voices"].items():
            assert voices.setdefault((language, character), spec) == spec, f"{character} changes voice in {path}"
        assert ep.data["seed"] not in seeds
        seeds.add(ep.data["seed"])


@pytest.mark.parametrize("path", [p for p in JV_EPISODES if "jv_en_" in os.path.basename(p)], ids=os.path.basename)
def test_english_jv_episodes_speak_with_kokoro(path):
    ep = drama.load_episode(path)
    for shot in ep.shots:
        if (shot.get("line") or "").strip():
            assert drama.is_kokoro(drama.voice_spec(ep, shot)), shot["id"]


def test_kana_only_lines_wrap_by_character():
    assert compose.is_cjk_text("いらっしゃいませ！") and compose.is_cjk_text("ラーメン")
    assert not compose.is_cjk_text("Can I jump in here?")
    line = "すみません、おすすめは何ですか？ありがとうございました！またきます！"
    lines = compose.wrap_lines(line, cjk_width=12)
    assert len(lines) > 1 and "".join(lines) == line
    assert all(sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in part) <= 24 for part in lines)
