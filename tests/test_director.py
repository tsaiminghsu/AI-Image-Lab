"""training/director.py: the director's desk of the GUI's picture/video tab. Pinned with a fake ffmpeg: the clips are
joined in the given order, each transition lands on its own boundary (hard cut = one frame), the title and end cards
are built from the first and last clip, every clip is levelled to the voice reference, the music mix is 4.2-safe (no
amix normalize option), and the cut is finished at -14 LUFS. One smoke test runs the real bundled ffmpeg when found."""

import json
import os
import subprocess

import activity_promo as ap
import director as d
import drama_compose as dc
import pytest

import generate_character as gc

MEASURE = "Duration: 00:00:05.17, start: 0\n  Stream #0:1: Audio: aac\n  Summary:\n  Integrated loudness:\n    I:         -16.0 LUFS\n"


class FakeRun:
    """Records every command; answers loudness measurements with MEASURE and creates each output file."""

    def __init__(self):
        self.commands = []

    def __call__(self, cmd, **kw):
        self.commands.append(cmd)
        out = cmd[-1]
        if isinstance(out, str) and os.path.splitext(out)[1] in (".mkv", ".mp4", ".png", ".wav"):
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            open(out, "wb").close()
        for i, arg in enumerate(cmd):  # the dissolve writes two parts before the mux
            if isinstance(arg, str) and arg.endswith(("_picture.mkv", "_sound.wav")) and i == len(cmd) - 1:
                open(arg, "wb").close()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=MEASURE)

    def graphs(self):
        return [c[c.index("-filter_complex") + 1] for c in self.commands if "-filter_complex" in c]


@pytest.fixture
def clips(tmp_path):
    paths = []
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        p = tmp_path / name
        p.write_bytes(b"x")
        paths.append(str(p))
    return paths


def render(tmp_path, clips, **kw):
    run = FakeRun()
    lines = []
    report = d.render(clips, str(tmp_path / "cut" / "out.mp4"), ffmpeg="ffmpeg", font="font.ttc", run=run,
                      log=lines.append, **kw)  # fmt: skip
    return report, run, lines


def test_clips_are_joined_in_order_with_one_transition_per_boundary(tmp_path, clips):
    report, run, lines = render(tmp_path, clips, transitions=["硬切", "疊化 0.25 秒", "疊化 0.5 秒"])
    shots = [c for c in run.commands if any(str(a).endswith(".mkv") and "clip_" in str(a) for a in c[-1:])]
    assert [c[c.index("-i") + 1] for c in shots] == clips
    assert report["overlaps_seconds"] == [0.25, 0.5]  # the first clip's transition only matters after a title card
    picture = next(g for g in run.graphs() if "overlay" in g)
    assert "fade=t=in:st=0:d=0.250" in picture and "fade=t=in:st=0:d=0.500" in picture
    assert report["seconds"] == round(3 * dc.frame_exact(5.17) - 0.75, 3)
    assert any("完成" in line for line in lines)


def test_all_hard_cuts_use_the_concat_join(tmp_path, clips):
    report, run, _ = render(tmp_path, clips)
    assert any("concat" in c for c in run.commands) and not any("overlay" in g for g in run.graphs())
    assert report["overlaps_seconds"] == [0.0, 0.0]


def test_every_clip_is_levelled_to_the_voice_reference_and_the_cut_finishes_at_minus_14(tmp_path, clips):
    report, run, _ = render(tmp_path, clips)
    gains = [g for g in run.graphs() if g.startswith("[0:v]") and "volume=" in g]
    assert len(gains) == 3 and all("volume=-7.00dB" in g for g in gains)  # -16 measured -> -23
    finish = run.commands[-2]
    assert finish[-1].endswith("out.mp4") and any("volume=2.00dB" in str(a) for a in finish)  # -16 mix -> -14
    assert report["clip_loudness_lufs"] == [-16.0, -16.0, -16.0] and report["size"] == "1080x1920"
    saved = json.loads((tmp_path / "cut" / "out.json").read_text(encoding="utf-8"))
    assert saved["clips"] == [os.path.abspath(c) for c in clips]


def test_title_and_end_cards_use_the_first_and_last_clip_as_background(tmp_path, clips):
    report, run, _ = render(
        tmp_path, clips, transitions=["疊化 0.25 秒", "硬切", "硬切"], title=("一起去約會", "副標"), title_seconds=2,
        ending=("下載 App", ""), ending_seconds=2.5,
    )  # fmt: skip
    first = next(c for c in run.commands if c[-1].endswith("first_frame.png"))
    last = next(c for c in run.commands if c[-1].endswith("last_frame.png"))
    assert first[first.index("-i") + 1] == clips[0] and last[last.index("-i") + 1] == clips[-1] and "-sseof" in last
    assert report["overlaps_seconds"] == [0.25, round(ap.CUT_SECONDS, 3), round(ap.CUT_SECONDS, 3), d.END_TRANSITION]
    assert report["title"] == ["一起去約會", "副標"] and report["ending"] == ["下載 App", ""]
    cards = [g for g in run.graphs() if "boxblur" in g]
    assert len(cards) == 2 and all("drawtext" in g for g in cards)


def test_a_card_with_no_text_or_zero_seconds_is_left_out(tmp_path, clips):
    report, run, _ = render(tmp_path, clips, title=("", ""), title_seconds=3, ending=("x", ""), ending_seconds=0)
    assert report["title"] is None and report["ending"] is None
    assert not any("boxblur" in g for g in run.graphs())


def test_music_is_mixed_the_4_2_way_under_the_voice_and_faded_out(tmp_path, clips):
    music = tmp_path / "m.mp3"
    music.write_bytes(b"x")
    report, run, _ = render(tmp_path, clips, music=str(music), music_below_db=18, music_fade=1.5)
    mix = next(g for g in run.graphs() if "amix" in g and "afade" in g)
    assert "normalize" not in mix and "amix=inputs=2:duration=first,volume=2" in mix
    assert "volume=-25.00dB" in mix  # measured -16 -> -23 - 18 = -41
    assert report["music"]["gain_db"] == -25.0 and "-stream_loop" in next(c for c in run.commands if mix in c)
    assert d.music_gain_db(None, 18) == -18 and d.music_gain_db(-80, 18) == -18


def test_the_ai_label_and_the_ambience_bed_are_optional(tmp_path, clips):
    _, plain, _ = render(tmp_path, clips)
    assert not any("drawtext" in g for g in plain.graphs()) and not any("anoisesrc" in str(c) for c in plain.commands)
    report, tagged, _ = render(tmp_path, clips, ai_label=True, bed=True)
    assert report["ai_tag"] == ap.AI_TAG and sum("drawtext" in g for g in tagged.graphs()) == 3
    assert report["ambience_bed_lufs"] == ap.BED_LUFS and any("anoisesrc" in str(c) for c in tagged.commands)


def test_bad_input_is_refused_before_ffmpeg_runs(tmp_path, clips):
    for kw in (
        {"clips": []},
        {"clips": [str(tmp_path / "missing.mp4")]},
        {"clips": clips, "transitions": ["硬切"]},
        {"clips": clips, "transitions": ["溶接"] * 3},
        {"clips": clips, "music": str(tmp_path / "nope.mp3")},
    ):
        run = FakeRun()
        with pytest.raises(gc.UsageError):
            d.render(kw.pop("clips"), str(tmp_path / "o.mp4"), ffmpeg="ffmpeg", font="f", run=run, log=lambda m: None,
                     **kw)  # fmt: skip
        assert run.commands == []


def test_a_failing_ffmpeg_step_stops_with_its_name_and_message(tmp_path, clips):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom: no such filter")

    with pytest.raises(gc.UsageError, match="boom"):
        d.render(clips, str(tmp_path / "o.mp4"), ffmpeg="ffmpeg", font="f", run=run, log=lambda m: None)


def test_list_clips_reads_the_seconds_and_the_spoken_line_from_each_record(tmp_path):
    (tmp_path / "h3_local_x.mp4").write_bytes(b"x")
    (tmp_path / "h3_local_x.json").write_text(
        json.dumps({"frames": 124, "description_used": "She smiles. says: <d>[Chinese] 快！關小火！</d> Then."}),
        encoding="utf-8",
    )
    (tmp_path / "old.mp4").write_bytes(b"x")
    rows = {r["name"]: r for r in d.list_clips(str(tmp_path))}
    assert rows["h3_local_x.mp4"]["seconds"] == round(124 / 24, 2) and rows["h3_local_x.mp4"]["line"] == "快！關小火！"
    assert rows["old.mp4"]["seconds"] is None and rows["old.mp4"]["line"] == ""


@pytest.mark.skipif(not dc.find_ffmpeg(), reason="no ffmpeg on this machine")
def test_a_real_cut_from_two_generated_clips_with_cards_and_music(tmp_path):
    ff = dc.find_ffmpeg()
    font = dc.find_font()
    if not font:
        pytest.skip("no CJK font")
    clips = []
    for i, freq in enumerate((440, 660)):
        p = str(tmp_path / f"clip{i}.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=576x1024:r=24:d=2",
                        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=2", "-shortest", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", p], check=True)  # fmt: skip
        clips.append(p)
    music = str(tmp_path / "music.wav")
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=1", music], check=True
    )
    report = d.render(
        clips, str(tmp_path / "cut.mp4"), transitions=["硬切", "疊化 0.25 秒"], title=("片頭", ""), title_seconds=1,
        ending=("片尾", ""), ending_seconds=1, music=music, log=lambda m: None,
    )  # fmt: skip
    probe = subprocess.run([ff, "-hide_banner", "-i", report["output"]], capture_output=True, text=True).stderr
    assert "1080x1920" in probe and "Audio: aac" in probe
    assert abs(report["seconds"] - (1 + 2 + 2 + 1 - (ap.CUT_SECONDS + 0.25 + d.END_TRANSITION))) < 0.1
    assert report["final_loudness_lufs"] is not None and abs(report["final_loudness_lufs"] - dc.LOUDNESS_TARGET) < 1.5


def test_the_timeline_edits_return_a_new_list_and_the_index_to_select():
    t, i = d.timeline_add([], "a.mp4")
    t, i = d.timeline_add(t, "b.mp4", "疊化 0.25 秒")
    t, i = d.timeline_add(t, "c.mp4")
    assert [x["name"] for x in t] == ["a.mp4", "b.mp4", "c.mp4"] and i == 2
    moved, j = d.timeline_move(t, 2, -1)
    assert [x["name"] for x in moved] == ["a.mp4", "c.mp4", "b.mp4"] and j == 1 and t[2]["name"] == "c.mp4"
    assert d.timeline_move(t, 0, -1)[1] == 0 and d.timeline_move(t, None, 1)[1] is None
    left, k = d.timeline_remove(moved, 2)
    assert [x["name"] for x in left] == ["a.mp4", "c.mp4"] and k == 1
    assert d.timeline_remove([{"name": "a", "transition": "硬切"}], 0) == ([], None)
    changed = d.timeline_set_transition(left, 1, "疊化 0.5 秒")
    assert changed[1]["transition"] == "疊化 0.5 秒" and left[1]["transition"] == "硬切"
    with pytest.raises(gc.UsageError):
        d.timeline_set_transition(left, 1, "溶接")


def test_the_estimated_length_matches_what_render_reports(tmp_path, clips):
    names = {os.path.basename(c): 5.17 for c in clips}
    timeline = [
        {"name": os.path.basename(c), "transition": t} for c, t in zip(clips, ["硬切", "疊化 0.25 秒", "疊化 0.5 秒"])
    ]
    report, _, _ = render(
        tmp_path, clips, transitions=[t["transition"] for t in timeline], title=("a", ""), title_seconds=2,
        ending=("b", ""), ending_seconds=2.5,
    )  # fmt: skip
    estimate = d.timeline_seconds(timeline, names, 2, 2.5, has_title=True, has_ending=True)
    assert abs(estimate - report["seconds"]) < 0.02
    hard = [{"name": n, "transition": "硬切"} for n in names]
    assert d.timeline_seconds(hard, names) == round(3 * dc.frame_exact(5.17), 2)


@pytest.mark.skipif(not dc.find_ffmpeg(), reason="no ffmpeg on this machine")
def test_a_thumbnail_is_made_once_and_then_read_from_the_cache(tmp_path):
    ff = dc.find_ffmpeg()
    clip = str(tmp_path / "c.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=576x1024:r=24:d=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], check=True)  # fmt: skip
    first = d.thumbnail(clip, str(tmp_path / "thumbs"))
    assert first and os.path.isfile(first)
    calls = []
    assert d.thumbnail(clip, str(tmp_path / "thumbs"), run=lambda *a, **k: calls.append(1)) == first and calls == []


def test_import_copies_supported_files_with_unique_names_and_reads_a_folder(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("a.mp4", "b.png", "c.wav", "notes.txt"):
        (src / name).write_bytes(b"x")
    dest = tmp_path / "imports"
    first = d.import_media([str(src)], str(dest))
    assert sorted(os.path.basename(p) for p in first["imported"]) == ["a.mp4", "b.png", "c.wav"]
    assert first["skipped"] == ["notes.txt"]
    again = d.import_media([str(src / "a.mp4")], str(dest))
    assert os.path.basename(again["imported"][0]) == "a_2.mp4"
    with pytest.raises(gc.UsageError):
        d.import_media([str(src / "notes.txt")], str(dest))
    media = {r["name"]: r for r in d.list_media(str(tmp_path / "none"), str(dest), ffmpeg="ffmpeg", run=FakeRun())}
    assert media["b.png"]["kind"] == "image" and media["a.mp4"]["source"] == "匯入" and "c.wav" not in media
    assert [os.path.basename(p) for p in d.list_audio(str(dest))] == ["c.wav"]


def test_tracks_are_checked_with_readable_reasons(tmp_path):
    png, wav = tmp_path / "o.png", tmp_path / "s.wav"
    png.write_bytes(b"x")
    wav.write_bytes(b"x")
    texts, overlays, sfx = d.check_tracks(
        [{"text": " 嗨 ", "start": 0, "end": 2}],
        [{"path": str(png), "start": 1, "end": 3}],
        [{"path": str(wav), "start": 4}],
    )
    assert texts[0]["text"] == "嗨" and overlays[0]["position"] == "右上" and sfx[0]["volume_db"] == 0
    for bad in (
        {"texts": [{"text": "", "start": 0, "end": 1}]},
        {"texts": [{"text": "x", "start": 2, "end": 1}]},
        {"texts": [{"text": "x", "start": 0, "end": 1, "position": "旁邊"}]},
        {"overlays": [{"path": str(wav), "start": 0, "end": 1}]},
        {"overlays": [{"path": str(png), "start": 0, "end": 1, "scale": 500}]},
        {"sfx": [{"path": str(png), "start": 0}]},
        {"sfx": [{"path": str(wav), "start": 0, "volume_db": 40}]},
    ):
        with pytest.raises(gc.UsageError):
            d.check_tracks(**bad)


def test_text_overlays_and_sound_effects_go_into_their_own_4_2_safe_passes(tmp_path, clips):
    png, wav = tmp_path / "o.png", tmp_path / "s.wav"
    png.write_bytes(b"x")
    wav.write_bytes(b"x")
    report, run, _ = render(
        tmp_path, clips, texts=[{"text": "標題", "start": 0, "end": 2, "position": "上方", "size": "大"}],
        overlays=[{"path": str(png), "start": 1, "end": 4, "position": "左上", "scale": 30}],
        sfx=[{"path": str(wav), "start": 2.5, "volume_db": -6}],
    )  # fmt: skip
    layers = next(g for g in run.graphs() if "overlay=x=" in g)
    assert (
        "enable='between(t,1.000,4.000)'" in layers
        and "drawtext" in layers
        and "enable='between(t,0.000,2.000)'" in layers
    )
    assert "amix" not in layers  # picture only: overlay and audio filters never share a graph on this ffmpeg
    effects = next(g for g in run.graphs() if "adelay=2500|2500" in g)
    assert "amix=inputs=2:duration=first,volume=2" in effects and "normalize" not in effects
    assert report["tracks"]["sfx"][0]["start"] == 2.5


def test_an_image_clip_becomes_a_still_segment_and_a_silent_video_gets_silence(tmp_path, clips):
    png = tmp_path / "p.png"
    png.write_bytes(b"x")

    class SilentRun(FakeRun):
        def __call__(self, cmd, **kw):
            done = super().__call__(cmd, **kw)
            if cmd[-2:] == ["-i", clips[1]]:  # the probe of the second clip: no audio stream
                done.stderr = "Duration: 00:00:03.00, start: 0\n"
            return done

    run = SilentRun()
    report = d.render([clips[0], {"path": str(png), "seconds": 2}, clips[1]], str(tmp_path / "o.mp4"), ffmpeg="ffmpeg",
                      font="f", run=run, log=lambda m: None)  # fmt: skip
    still = next(c for c in run.commands if str(png) in c and "-loop" in c)
    assert "anullsrc" in " ".join(still) and report["clip_loudness_lufs"][1] is None
    silent = next(c for c in run.commands if clips[1] in c and "-filter_complex" in c)
    assert "anullsrc" in " ".join(silent) and report["clip_loudness_lufs"][2] is None


def test_the_layout_and_the_multi_track_view_match_the_cut(tmp_path, clips):
    timeline = [{"name": "a", "transition": "硬切", "kind": "video"},
                {"name": "b", "transition": "疊化 0.25 秒", "kind": "image", "seconds": 3}]  # fmt: skip
    segs, total = d.layout(timeline, {"a": 5.17}, 2, 2.5, True, True)
    assert [s["kind"] for s in segs] == ["card", "video", "image", "card"]
    assert total == round(
        dc.frame_exact(2) + dc.frame_exact(5.17) + 3 + 2.5 - ap.CUT_SECONDS - 0.25 - d.END_TRANSITION, 2
    )
    html = d.timeline_html(segs, total, {"texts": [{"text": "嗨<", "start": 0, "end": 1}], "overlays": [], "sfx": []},
                           "song.mp3", 0, lambda p: "/f/" + p)  # fmt: skip
    for label in ("V2", "V1", "A1", "A2", "A3", "片頭", "片尾", "♪ song.mp3", "嗨&lt;"):
        assert label in html, label


def test_every_template_is_valid():
    for name, tpl in d.TEMPLATES.items():
        assert tpl["transition"] in d.TRANSITIONS, name
        d.check_tracks(tpl["texts"], (), ())
        assert 6 <= tpl["music_below_db"] <= 30 and len(tpl["title"]) == len(tpl["ending"]) == 2


def test_an_imported_video_is_measured_once_so_the_estimate_matches_the_cut(tmp_path):
    imports = tmp_path / "imports"
    imports.mkdir()
    (imports / "short.mp4").write_bytes(b"x")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="  Duration: 00:00:03.00, start: 0.000000\n")

    d._LENGTHS.clear()
    first = d.list_media(str(tmp_path / "none"), str(imports), ffmpeg="ffmpeg", run=run)
    again = d.list_media(str(tmp_path / "none"), str(imports), ffmpeg="ffmpeg", run=run)
    assert first[0]["seconds"] == again[0]["seconds"] == 3.0
    assert len(calls) == 1  # cached by path and mtime
    segs, total = d.layout([{"name": "short.mp4", "transition": "硬切", "kind": "video"}], {"short.mp4": 3.0})
    assert total == dc.frame_exact(3.0)  # not the 5.17 s fallback
