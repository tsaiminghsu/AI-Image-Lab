"""projects/activity_promo + training/activity_promo.py, offline.

What is pinned: the promo specs are usable by concept_stills (4 scenes, safety negatives and cfg floor on every
candidate), the promo character is one of the project's adults of 23 or older and keeps its appearance text,
every H3 description passes the H3 skill's prompt screen, the AI tag is on every shot and on the end card, the
picked chip changes colour mid-shot, and the cut keeps the measure -> fixed gain recipe. No GPU, no ffmpeg."""

import copy
import os
import re
import subprocess

import activity_promo as ap
import concept_cut as cc
import concept_stills as cs
import pytest
from minimax_h3_fakes import h3

import comfyui_client as client
import drama_compose as dc
import generate_character as gc

PROMO_MIN_AGE = 23


@pytest.fixture(scope="module")
def bible():
    return cs.load_bible(ap.PROJECT_ROOT)


@pytest.fixture(scope="module")
def stories():
    return cs.load_stories(ap.PROJECT_ROOT)


@pytest.fixture
def story(stories):
    """The amusement-park promo (the first one made); tests about one promo's own content use it."""
    park = next(s for s in stories if s.id == "promo_01_amusement_park")
    return cs.Story(copy.deepcopy(park.data), park.dir)


# --- the specs -------------------------------------------------------------------------------------------


def test_every_promo_is_a_valid_six_shot_story_inside_the_planned_length(stories):
    assert [s.id for s in stories] == [
        "promo_01_amusement_park",
        "promo_02_cooking",
        "promo_03_coffee",
        "promo_04_claw_machine",
        "promo_05_cat_cafe",
        "promo_06_movie",
        "promo_07_night_view",
        "promo_08_travel",
        "promo_09_picnic",
    ]
    for s in stories:
        cs.validate_story(s)
        assert s.data["scene_count"] == len(s.data["scenes"]) == 6
        assert ap.promo_problem(s) is None
        assert ap.MIN_SECONDS <= ap.planned_seconds(s) <= ap.MAX_SECONDS


def test_the_short_drama_stories_still_need_seven_scenes():
    drama = cs.load_stories()[0]
    broken = cs.Story(dict(drama.data, scenes=drama.data["scenes"][:4]), drama.dir)
    with pytest.raises(gc.UsageError, match="scene 1..7"):
        cs.validate_story(broken)
    with pytest.raises(gc.UsageError, match="scene_count"):
        cs.validate_story(cs.Story(dict(drama.data, scene_count=0), drama.dir))


def test_the_promo_character_is_a_project_adult_and_keeps_its_appearance(stories):
    for s in stories:
        for c in s.data["characters"].values():
            profile = gc.CHARACTERS[c["character_key"]]
            assert profile["age"] >= PROMO_MIN_AGE
            assert profile["appearance"] in c["description"] and f"aged {profile['age']}" in c["description"]


def test_every_candidate_carries_the_safety_negatives_the_cfg_floor_and_one_person(bible, stories):
    reqs = cs.all_requests(bible, stories, cs.FIRST_ROUND)
    assert len(reqs) == 12 * len(stories)
    for r in reqs:
        assert gc.AGE_SAFETY_NEGATIVE in r["negative_prompt"]
        assert r["cfg"] >= client.SAFETY_MIN_CFG
        assert (r["width"], r["height"]) == (720, 1280)
    for s in stories:
        assert all(scene["solo"] and not scene["on_screen_text"] for scene in s.data["scenes"])


def test_every_h3_description_passes_the_h3_prompt_screen_and_holds_the_camera(stories):
    for s in stories:
        for scene in s.data["scenes"]:
            video = scene["video"]
            h3.screen_prompt(video["description"] + " " + video["soundscape"])
            assert "The camera holds a steady shot." in video["description"]
            assert video["description"].count("<d>") == 1 and video["description"].count("</d>") == 1
            assert f"<d>[Chinese] {scene['spoken']}</d>" in video["description"]  # she says it herself
            assert "footstep" not in (video["description"] + video["soundscape"]).lower()  # steps came out too heavy


def test_overlay_text_has_no_retention_pressure(stories):
    banned = ("不要走", "別走", "別離開", "不要離開", "你不在", "丟下")
    for s in stories:
        texts = [s.data["promo"]["hook"], *s.data["promo"]["end_card"].values()]
        for scene in s.data["scenes"]:
            overlay = scene["overlay"]
            texts += [overlay.get("caption", ""), overlay.get("question", ""), *overlay.get("choices", [])]
        assert not [t for t in texts for b in banned if b in t]
        assert "18" in s.data["promo"]["end_card"]["note"]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["promo"].pop("hook"), "promo.hook"),
        (lambda d: d["promo"]["end_card"].pop("note"), "end_card"),
        (lambda d: d["scenes"][0].pop("video"), "沒有 video"),
        (lambda d: d["scenes"][1]["overlay"].update(picked=3), "picked"),
        (lambda d: d["scenes"][1]["overlay"].update(choices=["a", "b", "c", "d"]), "choices"),
        (lambda d: d["scenes"][1]["overlay"].pop("question"), "question"),
        (lambda d: d["scenes"][1]["video"].update(duration=3.5), "太短"),
        (lambda d: [sc["video"].update(duration=15) for sc in d["scenes"]], "全長"),
    ],
)
def test_a_broken_promo_spec_is_named(story, change, message):
    change(story.data)
    assert message in ap.promo_problem(story)


def test_the_timeline_shows_the_picked_choice_and_the_end_card(story):
    lines = ap.timeline(story)
    assert len(lines) == 7 and "[雲霄飛車]／旋轉木馬／鬼屋" in lines[1] and "[棉花糖]／吉拿棒" in lines[2]
    # five 0.25 s dissolves and one cut (1 to 2) shorten the 32.5 s of footage to 31.2 s
    assert lines[-1].startswith("28.71-31.21s") and "結尾字卡" in lines[-1]


# --- filters and commands --------------------------------------------------------------------------------


def test_the_picked_chip_changes_colour_at_the_pick_and_the_others_stay():
    filters = ap.choice_filters(["q.txt"], ["a.txt", "b.txt", "c.txt"], 1, "C:\\f.ttc")
    chips = [f for f in filters if "box=1" in f]
    assert len(chips) == 4  # the picked chip is drawn twice: before and after
    b = [f for f in chips if "textfile='b.txt'" in f]
    assert f"boxcolor={ap.CHIP_COLOR}" in b[0] and b[0].endswith(f":enable='lt(t,{ap.PICK_AT})'")
    assert f"boxcolor={ap.CHIP_PICKED_COLOR}" in b[1] and b[1].endswith(f":enable='gte(t,{ap.PICK_AT})'")
    assert all("enable" not in f for f in chips if "b.txt" not in f)
    last_chip_bottom = ap.CHIP_TOP + (ap.MAX_CHOICES - 1) * ap.CHIP_PITCH + ap.CHIP_SIZE + ap.CHIP_PADDING
    assert last_chip_bottom < dc.SUBTITLE_BOTTOM - 2 * dc.SUBTITLE_FONT_SIZE - dc.LINE_GAP  # a two-line caption fits


def test_a_shot_has_the_tag_first_the_hook_only_when_asked_and_the_reply_after_the_pick(story, tmp_path):
    work = str(tmp_path)
    tag = ap.tag_filter("t.txt", "C:\\f.ttc")
    first = ap.shot_filters(work, story.data["scenes"][0], "C:\\f.ttc", tag, hook="約會去哪，你來選")
    assert first[0] == tag and sum(f"lt(t,{ap.HOOK_SECONDS})" in f for f in first) == 1
    assert open(tmp_path / "s1_hook_0.txt", encoding="utf-8").read() == "約會去哪，你來選"
    choice = ap.shot_filters(work, story.data["scenes"][1], "C:\\f.ttc", tag)
    pick = story.data["scenes"][1]["overlay"]["pick_at"]  # lit as she starts to answer
    assert choice[0] == tag and choice[-1].endswith(f":enable='gte(t,{pick})'")  # the reply waits
    assert "s2_caption_0.txt" in choice[-1] and not any("hook" in f for f in choice)
    plain = ap.shot_filters(work, story.data["scenes"][4], "C:\\f.ttc", tag)  # a one-line caption
    assert len(plain) == 2 and "enable" not in plain[-1]


def test_the_shot_command_keeps_the_clips_own_sound_and_the_end_card_is_tagged():
    cmd = ap.shot_command("ffmpeg", "a.mp4", "a.mkv", 5.0, ["TAG", "CAPTION"], -2.9)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert f"crop={dc.OUT_W}:{dc.OUT_H}" in graph and ",TAG,CAPTION,format=yuv420p[v]" in graph
    assert "[0:a]volume=-2.90dB," in graph and "atrim=0:5.000[a]" in graph and "anullsrc" not in " ".join(cmd)
    card = ap.end_card_command("ffmpeg", "e.mkv", [("phrase", ["p.txt"])], "C:\\f.ttc", "bg.png", "TAG")
    assert ",TAG,format=yuv420p[v]" in card[card.index("-filter_complex") + 1]
    assert card[card.index("-t") + 1] == f"{ap.END_CARD_SECONDS:.3f}"


# --- compose ---------------------------------------------------------------------------------------------


def test_compose_tags_every_segment_levels_each_clip_and_finishes_at_the_measured_gain(story, tmp_path):
    story.data["promo"].update(ai_label=True, on_screen_text=True, subtitles=True)  # this promo ships with them off
    clips = []
    for i in range(6):
        (tmp_path / f"c{i}.mp4").write_bytes(b"x")
        clips.append(str(tmp_path / f"c{i}.mp4"))
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd)
        text = " ".join(cmd)
        log = ""
        if "ebur128" in text:
            name = os.path.basename(cmd[cmd.index("-i") + 1])
            level = {"body.mkv": -24.0, "body_bed.mkv": -24.0, "out.mp4": -14.1}.get(name, -35.0)
            log = f"Duration: 00:00:05.04,\nSummary:\n I: {level} LUFS"
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=log)

    out = str(tmp_path / "cut" / "out.mp4")
    result = ap.compose(story, clips, out, ffmpeg="ffmpeg", font="C:\\f.ttc", run=run)
    graphs = [c[c.index("-filter_complex") + 1] for c in cmds if "-filter_complex" in c]
    assert len(graphs) == 10 and all(
        "ai_tag_0.txt" in g for g in graphs[:7]
    )  # 6 shots + the end card, then the dissolve passes
    assert all("volume=12.00dB" in g for g in graphs[:6])  # -35 -> the -23 reference
    assert sum("s1_hook_0.txt" in g for g in graphs) == 1 and "s1_hook_0.txt" in graphs[0]
    finish = cmds[-2]
    assert finish[-1] == os.path.abspath(out) and "volume=10.00dB,alimiter" in finish[finish.index("-af") + 1]
    assert result["final_loudness_lufs"] == -14.1 and result["shots"] == 6 and result["ai_tag"] == ap.AI_TAG
    assert result["seconds"] == round(
        6 * dc.frame_exact(5.04) + dc.frame_exact(ap.END_CARD_SECONDS) - sum(ap.shot_transitions(story)), 2
    )
    assert result["transition_seconds"] == ap.TRANSITION_SECONDS
    assert "overlay=" in graphs[7] and "acrossfade=" in graphs[8]  # picture and sound in separate runs
    assert (
        "amix=inputs=2" in graphs[9] and "normalize" not in graphs[9] and "anoisesrc" in " ".join(cmds[-4])
    )  # the ambience bed goes under it all
    assert result["ambience_bed_lufs"] == ap.BED_LUFS
    with pytest.raises(gc.UsageError, match="找不到片段"):
        ap.compose(story, [*clips[:5], str(tmp_path / "nope.mp4")], out, ffmpeg="ffmpeg", font="f", run=run)


def test_a_failed_ffmpeg_step_stops_the_cut(story, tmp_path):
    (tmp_path / "c.mp4").write_bytes(b"x")

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    with pytest.raises(subprocess.CalledProcessError):
        ap.compose(story, [str(tmp_path / "c.mp4")] * 6, str(tmp_path / "o.mp4"), ffmpeg="ffmpeg", font="f", run=run)


# --- clips and the H3 manifest ---------------------------------------------------------------------------


def test_clips_come_from_the_command_line_then_clips_json_then_the_newest_version(story, tmp_path):
    assert ap.clip_paths(story, ["a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4", "f.mp4"])[0] == os.path.abspath("a.mp4")
    with pytest.raises(gc.UsageError, match="剛好 6 支"):
        ap.clip_paths(story, ["a.mp4"])
    local = cs.Story(story.data, str(tmp_path))
    for n in (1, 2, 3, 4, 5):
        os.makedirs(local.clip_dir(n))
        for v in (1, 2, 10):
            open(os.path.join(local.clip_dir(n), f"clip_v{v}.mp4"), "wb").close()
    with pytest.raises(gc.UsageError, match=r"scene \[6\] 還沒有 H3 影片"):
        ap.clip_paths(local)
    os.makedirs(local.clip_dir(6))
    open(os.path.join(local.clip_dir(6), "clip_v1.mp4"), "wb").close()
    cs._write_json(str(tmp_path / "clips.json"), {"scene_02": {"path": "pinned/clip_v1.mp4"}})
    paths = ap.clip_paths(local)
    assert paths[0].endswith("clip_v10.mp4")  # numeric, not alphabetical
    assert paths[1] == os.path.join(cs.REPO_ROOT, "pinned/clip_v1.mp4") and paths[5].endswith("clip_v1.mp4")


def test_the_manifest_is_refused_until_every_scene_is_approved(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    with pytest.raises(gc.UsageError, match="還沒有核准的 master"):
        ap.write_manifest(local, "amusement-park-pilot")
    for scene in story.data["scenes"]:
        n = scene["scene"]
        paths = cs.candidate_paths(local, n, "A")
        os.makedirs(os.path.dirname(paths["png"]), exist_ok=True)
        open(paths["png"], "wb").write(b"png")
        cs._write_json(paths["meta"], {"seed": 1})
        cs.review(local, n, approve="A")
    path = ap.write_manifest(local, "amusement-park-pilot")
    assert path == os.path.join(str(tmp_path), "videos", "_batches", "amusement-park-pilot.json")
    manifest = cs._read_json(path)
    assert [j["output"] for j in manifest["jobs"]] == [
        os.path.join(local.clip_dir(n), "clip_v1.mp4") for n in (1, 2, 3, 4, 5, 6)
    ]
    assert [j["settings"]["duration"] for j in manifest["jobs"]] == [5, 5, 5, 5, 5, 5]
    assert all(j["input"]["image"].endswith("master.png") for j in manifest["jobs"])
    assert manifest["jobs"][1]["prompt"]["description"] == story.data["scenes"][1]["video"]["description"]


# --- continuity: the shots must hand over to each other ------------------------------------------------------


def test_neighbouring_shots_carry_the_same_woman_the_same_props_and_a_light_that_only_moves_forward(stories):
    for s in stories:
        scenes, check = s.data["scenes"], s.data["continuity_check"]
        assert len({tuple(sc["characters"]) for sc in scenes}) == 1  # one character, one description, every shot
        order = check["time_order"]
        times = [next(i for i, t in enumerate(order) if sc["time_of_day"].startswith(t)) for sc in scenes]
        assert times == sorted(times) and times[0] == 0 and times[-1] == len(order) - 1, s.id
        assert all(sc.get("bridge") for sc in scenes[1:])  # every cut after the first says how it joins the one before
        for key in ("prop", "prop2"):
            if key not in check:
                continue
            shots = check[key + "_shots"] if key == "prop2" else check["prop_shots"]
            assert shots == list(range(shots[0], shots[-1] + 1))  # the prop is never dropped and picked up again
            for sc in scenes:
                text = sc["action"] + " " + sc["video"]["description"]
                assert (check[key] in text) == (sc["scene"] in shots), (s.id, sc["title"], check[key])


def test_a_two_or_three_option_choice_shot_leaves_time_to_read_the_reply(stories):
    for s in stories:
        for sc in s.data["scenes"]:
            if "choices" in sc["overlay"]:
                assert sc["video"]["duration"] >= ap.PICK_AT + 2  # pick, then at least 2 s of the character's reply


# --- dissolves ---------------------------------------------------------------------------------------------


def test_the_dissolve_overlaps_every_boundary_by_the_same_amount_in_picture_and_sound():
    segs, durs = [f"s{i}.mkv" for i in range(3)], [4.0, 5.0, 2.5]
    picture, sound, mux = ap.dissolve_commands("ffmpeg", segs, durs, "o/body.mkv", 0.25)
    g = picture[picture.index("-filter_complex") + 1]
    assert "setpts=PTS-STARTPTS+3.750/TB" in g and "setpts=PTS-STARTPTS+8.500/TB" in g  # 4-.25, 4+5-.5
    assert g.count("fade=t=in:st=0:d=0.250:alpha=1") == 2 and g.endswith("[o2]format=yuv420p[v]")
    assert "acrossfade" not in g  # this ffmpeg build crashes with both in one graph
    h = sound[sound.index("-filter_complex") + 1]
    assert h.count("acrossfade=d=0.250") == 2 and "overlay" not in h
    total = "11.000"  # 4 + 5 + 2.5 - 2 x 0.25
    assert picture[picture.index("-t") + 1] == total == sound[sound.index("-t") + 1] == mux[mux.index("-t") + 1]
    assert picture[-1].endswith("body_picture.mkv") and sound[-1].endswith("body_sound.wav") and mux[-1] == "o/body.mkv"
    assert "copy" in mux and mux.count("-i") == 2


def test_a_dissolve_that_would_eat_a_shot_is_refused():
    with pytest.raises(ValueError, match="at least twice"):
        ap.dissolve_commands("ffmpeg", ["a", "b"], [0.4, 5.0], "o.mkv", 0.25)
    with pytest.raises(ValueError, match="0 < t"):
        ap.dissolve_commands("ffmpeg", ["a", "b"], [4.0, 5.0], "o.mkv", 0.0)
    with pytest.raises(ValueError, match="at least two"):
        ap.dissolve_commands("ffmpeg", ["a"], [4.0], "o.mkv", 0.25)


def test_hard_cuts_are_still_available_and_a_bad_transition_is_named(story, tmp_path):
    clips = []
    for i in range(6):
        (tmp_path / f"c{i}.mp4").write_bytes(b"x")
        clips.append(str(tmp_path / f"c{i}.mp4"))
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd)
        log = "Duration: 00:00:05.00,\nSummary:\n I: -20.0 LUFS" if "ebur128" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=log)

    out = str(tmp_path / "o" / "x.mp4")
    result = ap.compose(story, clips, out, ffmpeg="ffmpeg", font="f", run=run, transition=0)
    assert not any("overlay=" in " ".join(c) for c in cmds) and any("concat" in c for c in cmds)
    assert result["seconds"] == round(6 * 5.0 + dc.frame_exact(ap.END_CARD_SECONDS), 2)
    for bad in (-0.1, 0.6, 0.1):  # negative, longer than half a second, not a whole number of frames
        with pytest.raises(gc.UsageError, match="疊化秒數"):
            ap.compose(story, clips, out, ffmpeg="ffmpeg", font="f", run=run, transition=bad)


@pytest.mark.skipif(not dc.find_ffmpeg(), reason="needs ffmpeg")
def test_the_real_ffmpeg_crossfades_white_into_black_and_keeps_the_length(tmp_path):
    ffmpeg = dc.find_ffmpeg()
    segs = []
    for colour in ("white", "black"):
        seg = str(tmp_path / f"{colour}.mkv")
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=64x112:r=24:d=1",
             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "pcm_s16le", seg],
            check=True,
        )  # fmt: skip
        segs.append(seg)
    out = str(tmp_path / "wb.mkv")
    for cmd in ap.dissolve_commands(ffmpeg, segs, [1.0, 1.0], out, 0.25):
        subprocess.run(cmd, check=True, capture_output=True)

    def luma(at):
        log = subprocess.run(
            [ffmpeg, "-hide_banner", "-ss", str(at), "-i", out, "-frames:v", "1", "-vf", "signalstats,metadata=print", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stderr  # fmt: skip
        return float(re.search(r"YAVG=([\d.]+)", log).group(1))

    assert luma(0.5) > 230 and luma(1.1) < 20  # white, then black
    assert 40 < luma(0.875) < 200  # in the middle of the overlap it is neither
    length = cc.duration_seconds(subprocess.run([ffmpeg, "-hide_banner", "-i", out], capture_output=True, text=True,
                                                encoding="utf-8", errors="replace").stderr)  # fmt: skip
    assert abs(length - 1.75) < 0.05


# --- continuity across the cut: direction, action, landmark, framing, sound ----------------------------------


def test_every_cut_matches_direction_action_framing_landmark_and_sound(stories):
    for s in stories:
        scenes = s.data["scenes"]
        assert {sc["screen_direction"] for sc in scenes} == {"right"}
        word = s.data["continuity_check"]["landmark_word"]
        for sc in scenes:
            first_frame = sc["action"] + " " + " ".join(sc["compositions"].values())
            assert "right" in sc["action"] and "left of the frame" not in first_frame, sc["title"]
            assert word in sc["setting"] + sc["action"] and sc["landmark"], (s.id, sc["title"])
            words = {"medium": "medium shot", "close": "medium close-up"}[sc["framing"]]
            assert all(words in c for c in sc["compositions"].values()), sc["title"]  # A and B share the framing
            assert sc["sound_bridge"] in sc["video"]["soundscape"]
        for before, after in zip(scenes, scenes[1:]):
            assert after["enters_on"] == before["exits_on"]  # the next first frame is the motion this shot ends on
            assert before["framing"] != after["framing"]  # never two frames of the same size across a cut
            assert before["sound_bridge"].startswith("Near the end")  # the next place is heard before it is seen
        assert scenes[-1]["sound_bridge"] == ""


# --- the boundary sheet --------------------------------------------------------------------------------------


def test_light_numbers_come_from_signalstats_and_a_frame_without_them_is_unmeasured():
    log = "lavfi.signalstats.YAVG=120.52\nlavfi.signalstats.UAVG=118\nlavfi.signalstats.VAVG=140.26\n"
    assert ap.parse_stats(log) == {"brightness": 120.5, "warmth": 22.3}
    assert ap.parse_stats("no stats here") is None
    assert "signalstats,metadata=print" in ap.stats_command("ffmpeg", "a.png")


def test_blend_and_frame_commands():
    cmd = ap.blend_command("ffmpeg", "a.png", "b.png", "o.png")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count(f"crop={ap.BLEND_W}:{ap.BLEND_H}") == 2 and "blend=all_mode=average" in graph
    assert "-sseof" in ap.frame_command("ffmpeg", "c.mp4", "o.png", last=True)
    assert "-sseof" not in ap.frame_command("ffmpeg", "c.mp4", "o.png")


def _fake_do(calls):
    def do(cmd):
        calls.append(cmd)
        return "lavfi.signalstats.YAVG=100\nlavfi.signalstats.UAVG=120\nlavfi.signalstats.VAVG=130\n"

    return do


def test_stills_rows_pair_every_candidate_across_each_cut_and_an_approved_scene_shows_only_its_master(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    for n, letters in ((1, "AB"), (2, "AB"), (3, "A")):  # scenes 4-6 have nothing yet
        for letter in letters:
            paths = cs.candidate_paths(local, n, letter)
            os.makedirs(os.path.dirname(paths["png"]), exist_ok=True)
            open(paths["png"], "wb").write(b"png")
            cs._write_json(paths["meta"], {"seed": 1})
    cs.review(local, 2, approve="B")
    calls = []
    source, rows = ap.boundary_rows(local, _fake_do(calls), "ffmpeg")
    assert source == "stills" and len(rows) == 5
    assert [len(r["pairs"]) for r in rows] == [2, 1, 0, 0, 0]  # 2 candidates x master, master x 1 candidate
    assert {p["after"]["label"] for p in rows[0]["pairs"]} == {"master"}
    assert rows[0]["pairs"][0]["before"]["stats"] == {"brightness": 100.0, "warmth": 10.0}
    blends = [c for c in calls if "-filter_complex" in c]
    measured = [c for c in calls if "signalstats,metadata=print" in c]
    assert len(blends) == 3 and len(measured) == 4  # each distinct frame is measured once
    html = ap.boundary_html(cs.load_bible(ap.PROJECT_ROOT), local, source, rows, str(tmp_path))
    assert html.count("疊化中間") == 3 and html.count("這個交界有一邊還沒有圖") == 3
    assert "images/_boundaries/blend_01-1_02-1.png" in html and "亮度 100｜暖色 +10" in html
    assert "最好" not in html and "best" not in html.lower()  # the sheet lays out, it never ranks


def test_once_every_scene_has_a_clip_the_rows_use_last_frame_against_first_frame(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    for sc in story.data["scenes"]:
        os.makedirs(local.clip_dir(sc["scene"]))
        open(os.path.join(local.clip_dir(sc["scene"]), "clip_v1.mp4"), "wb").close()
    calls = []
    source, rows = ap.boundary_rows(local, _fake_do(calls), "ffmpeg")
    assert source == "clips" and [len(r["pairs"]) for r in rows] == [1] * 5
    pair = rows[0]["pairs"][0]
    assert pair["before"]["path"].endswith("scene_01_last.png") and pair["after"]["path"].endswith("scene_02_first.png")
    assert sum("-sseof" in c for c in calls) == 6


@pytest.mark.skipif(not dc.find_ffmpeg(), reason="needs ffmpeg")
def test_the_real_ffmpeg_writes_a_sheet_whose_blend_is_half_way_between_the_two_frames(story, tmp_path):
    ffmpeg = dc.find_ffmpeg()
    local = cs.Story(dict(story.data, scenes=story.data["scenes"][:2], scene_count=2), str(tmp_path / "promo"))
    for n, colour in ((1, "white"), (2, "black")):
        paths = cs.candidate_paths(local, n, "A")
        os.makedirs(os.path.dirname(paths["png"]), exist_ok=True)
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s=90x160",
             "-frames:v", "1", paths["png"]],
            check=True,
        )  # fmt: skip
    path = ap.build_boundary_sheet(cs.load_bible(ap.PROJECT_ROOT), [local], str(tmp_path), ffmpeg=ffmpeg)
    page = open(path, encoding="utf-8").read()
    assert "promo/images/_boundaries/blend_01-1_02-1.png" in page and "promo/images/scene_01/candidate_01.png" in page
    blend = os.path.join(local.dir, "images", ap.BOUNDARY_DIR, "blend_01-1_02-1.png")
    log = subprocess.run(
        ap.stats_command(ffmpeg, blend), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    white = ap.parse_stats(subprocess.run(ap.stats_command(ffmpeg, cs.candidate_paths(local, 1, "A")["png"]),
                                          capture_output=True, text=True, encoding="utf-8", errors="replace").stderr)  # fmt: skip
    mid = ap.parse_stats(log.stderr)
    assert white["brightness"] > 200 and 90 < mid["brightness"] < 160


# --- the ChatGPT prompt file --------------------------------------------------------------------------------


def test_each_chatgpt_prompt_file_is_exactly_what_its_story_json_produces(bible, stories):
    for story in stories:
        name = f"GPT_PROMPTS_{story.id}.md"
        with open(os.path.join(ap.PROJECT_ROOT, name), encoding="utf-8", newline="") as f:
            assert f.read() == ap.build_gpt_prompts_md(bible, story) + "\n", (
                f"{name} is stale: run activity_promo.py gpt-prompts"
            )


def test_every_shot_prompt_says_same_woman_keeps_its_framing_and_ends_with_one_person_and_no_text(stories):
    for story in stories:
        char = story.data["characters"]["ruoxi"]
        suffix = story.data["promo"]["gpt_suffix"]
        for scene in story.data["scenes"]:
            for letter in sorted(scene["compositions"]):
                text = ap.gpt_shot_prompt(story, scene, letter)
                assert text.startswith(char["gpt_same"]) and text.endswith(suffix)
                assert scene["compositions"][letter] in text and scene["action"].rstrip(".") in text
                assert scene["setting"] in text and scene["lighting"] in text
                if "gpt_outfit" in char:  # a promo in new clothes says so in every shot
                    assert char["gpt_outfit"] in text
        reference = ap.gpt_reference_prompt(story)
        assert "aged 23" in reference and "Only one person" in reference and "{" not in reference
        md = ap.build_gpt_prompts_md(cs.load_bible(ap.PROJECT_ROOT), story)
        takes = sum(len(sc["compositions"]) for sc in story.data["scenes"])  # A and B, plus C and D where written
        assert md.count("```text") == 1 + takes  # the reference, then every take of every shot


# --- stills made outside the repo ----------------------------------------------------------------------------


def _png(path, width=941, height=1672):
    import struct

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            + struct.pack(">II", width, height)
            + b"\x08\x02\x00\x00\x00fake"
            + os.path.abspath(path).encode()  # every file differs, as real pictures do
        )


def test_dropped_images_get_their_candidate_names_records_and_a_place_in_review_but_are_never_approved(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    _png(os.path.join(local.scene_dir(1), "candidate_01.png"))
    _png(os.path.join(local.scene_dir(2), "candidate_02.png"))  # the only file, named for the wrong slot
    _png(os.path.join(local.scene_dir(3), "a.png"))
    _png(os.path.join(local.scene_dir(3), "b.png"))
    os.makedirs(local.scene_dir(4))
    open(os.path.join(local.scene_dir(4), "from_gpt.webp"), "wb").write(b"webp")
    converted = []

    def run(cmd, **kw):
        converted.append(cmd)
        _png(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    adopted = ap.adopt_stills(local, ffmpeg="ffmpeg", run=run)
    assert [(a["scene"], a["letter"]) for a in adopted] == [(1, "A"), (2, "A"), (3, "A"), (3, "B"), (4, "A")]
    assert converted and converted[0][converted[0].index("-i") + 1].endswith("from_gpt.webp")
    for n in (1, 2, 3, 4):
        assert os.path.isfile(cs.candidate_paths(local, n, "A")["png"])
        assert cs.load_review(local, n)["status"] == cs.HUMAN_REVIEW  # handed to a person, not decided
    assert not os.path.exists(cs.candidate_paths(local, 2, "B")["png"])  # candidate_02.png did not stay a B
    assert sorted(os.listdir(local.scene_dir(3))) == [
        *(f"candidate_0{i}.{e}" for i in (1, 2) for e in ("json", "png")),
        "review.json",
    ]
    assert not os.path.exists(os.path.join(local.scene_dir(4), "from_gpt.webp"))
    meta = cs._read_json(cs.candidate_paths(local, 1, "A")["meta"])
    assert meta["source"] == ap.STILL_SOURCE and (meta["width"], meta["height"]) == (941, 1672)
    assert meta["prompt_suggested"] == ap.gpt_shot_prompt(local, local.scene(1), "A") and meta["image_sha256"]
    assert ap.adopt_stills(local, ffmpeg="ffmpeg", run=run) == []  # a second run finds nothing new
    assert all(cs.load_review(local, n)["status"] != cs.APPROVED for n in range(1, 7))


def test_a_third_image_for_one_shot_is_refused_and_a_non_png_needs_ffmpeg(story, tmp_path, monkeypatch):
    local = cs.Story(story.data, str(tmp_path))
    for name in ("a.png", "b.png", "c.png"):
        _png(os.path.join(local.scene_dir(1), name))
    with pytest.raises(gc.UsageError, match="只剩 2 個候選位置"):
        ap.adopt_stills(local)
    assert sorted(os.listdir(local.scene_dir(1))) == ["a.png", "b.png", "c.png"]  # refused before anything moved
    other = cs.Story(story.data, str(tmp_path / "other"))
    os.makedirs(other.scene_dir(1))
    open(os.path.join(other.scene_dir(1), "x.jpg"), "wb").write(b"jpg")
    monkeypatch.setattr(dc, "find_ffmpeg", lambda env=None: None)
    with pytest.raises(gc.UsageError, match="找不到 ffmpeg"):
        ap.adopt_stills(other)


def test_an_image_saved_over_a_recorded_candidate_refreshes_its_record_and_is_refused_once_approved(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    _png(os.path.join(local.scene_dir(1), "mine.png"), 941, 1672)
    ap.adopt_stills(local)
    paths = cs.candidate_paths(local, 1, "A")
    old = cs._read_json(paths["meta"])["image_sha256"]
    assert ap.refresh_replaced(local) == []  # nothing changed yet
    _png(paths["png"], 720, 1280)  # the user re-made the shot and saved it over the old file
    assert [(r["scene"], r["letter"], r["size"]) for r in ap.refresh_replaced(local)] == [(1, "A", (720, 1280))]
    meta = cs._read_json(paths["meta"])
    assert meta["image_sha256"] != old and (meta["width"], meta["height"]) == (720, 1280) and "replaced_at" in meta
    assert ap.refresh_replaced(local) == []
    cs.review(local, 1, approve="A")
    _png(paths["png"], 100, 200)
    with pytest.raises(gc.UsageError, match="已 APPROVED"):
        ap.refresh_replaced(local)


def test_a_redo_manifest_holds_only_the_named_scenes_and_the_next_clip_version(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    for scene in story.data["scenes"]:
        n = scene["scene"]
        paths = cs.candidate_paths(local, n, "A")
        os.makedirs(os.path.dirname(paths["png"]), exist_ok=True)
        open(paths["png"], "wb").write(b"png")
        cs._write_json(paths["meta"], {"seed": 1})
        cs.review(local, n, approve="A")
    os.makedirs(local.clip_dir(1))
    open(os.path.join(local.clip_dir(1), "clip_v1.mp4"), "wb").close()
    path = ap.write_manifest(local, "amusement-park-v2", {1, 3})
    manifest = cs._read_json(path)
    assert [j["scene_id"].split("/")[-1] for j in manifest["jobs"]] == ["scene_01", "scene_03"]
    assert manifest["jobs"][0]["output"].endswith(os.path.join("scene_01", "clip_v2.mp4"))  # v1 is kept
    assert manifest["jobs"][1]["output"].endswith(os.path.join("scene_03", "clip_v1.mp4"))
    with pytest.raises(gc.UsageError, match="沒有 scene"):
        ap.write_manifest(local, "amusement-park-v3", {9})


# --- per-boundary transitions ---------------------------------------------------------------------------------


def test_a_shot_can_ask_for_a_cut_into_it_and_the_others_keep_the_dissolve(story):
    ts = ap.shot_transitions(story)
    assert len(ts) == 6 and ts[0] == ap.CUT_SECONDS  # 1 -> 2 is a cut: its faces do not line up
    assert ts[1:] == [ap.TRANSITION_SECONDS] * 5  # including into the end card
    assert ap.shot_transitions(story, 0) == [0.0] * 6  # --transition 0 is hard cuts everywhere
    story.data["scenes"][2]["transition_in"] = 0.4
    assert ap.shot_transitions(story)[1] == 0.4
    assert ap.planned_seconds(story) == pytest.approx(30 + 2.5 - (ap.CUT_SECONDS + 0.4 + 4 * 0.25))


@pytest.mark.parametrize("bad", [0, -0.2, 0.9, True, "dissolve"])
def test_a_bad_transition_in_is_named(story, bad):
    story.data["scenes"][3]["transition_in"] = bad
    assert "transition_in" in ap.promo_problem(story)


def test_the_dissolve_takes_a_list_with_one_overlap_per_boundary():
    segs, durs = ["a.mkv", "b.mkv", "c.mkv"], [4.0, 5.0, 2.5]
    picture, sound, _ = ap.dissolve_commands("ffmpeg", segs, durs, "o/body.mkv", [1 / 24, 0.25])
    g = picture[picture.index("-filter_complex") + 1]
    assert "setpts=PTS-STARTPTS+3.958/TB" in g and "setpts=PTS-STARTPTS+8.708/TB" in g  # 4-1/24, 4+5-1/24-.25
    assert "fade=t=in:st=0:d=0.042:alpha=1" in g and "fade=t=in:st=0:d=0.250:alpha=1" in g
    assert picture[picture.index("-t") + 1] == f"{4 + 5 + 2.5 - 1 / 24 - 0.25:.3f}"
    with pytest.raises(ValueError, match="2 boundaries"):
        ap.dissolve_commands("ffmpeg", segs, durs, "o.mkv", [0.25])


# --- spoken lines instead of subtitles -----------------------------------------------------------------------


def test_the_promo_has_no_subtitles_but_keeps_the_hook_the_question_and_the_chips(story, tmp_path):
    assert story.data["promo"]["subtitles"] is False
    tag = ap.tag_filter("t.txt", "C:\\f.ttc")
    choice = ap.shot_filters(str(tmp_path), story.data["scenes"][1], "C:\\f.ttc", tag, subtitles=False)
    assert not any("caption" in f for f in choice) and sum("box=1" in f for f in choice) == 4  # the chips are drawn
    assert any("s2_question_0.txt" in f for f in choice)
    first = ap.shot_filters(str(tmp_path), story.data["scenes"][0], "C:\\f.ttc", tag, hook="hook", subtitles=False)
    assert [f for f in first if "s1_hook" in f] and not any("caption" in f for f in first)
    plain = ap.shot_filters(str(tmp_path), story.data["scenes"][4], "C:\\f.ttc", tag, subtitles=False)
    assert plain == [tag]  # nothing but the AI tag on a shot that only speaks


def test_a_choice_shot_can_light_its_chip_just_before_she_answers(story, tmp_path):
    scene = copy.deepcopy(story.data["scenes"][1])
    scene["overlay"]["pick_at"] = 1.4
    tag = ap.tag_filter("t.txt", "C:\\f.ttc")
    filters = ap.shot_filters(str(tmp_path), scene, "C:\\f.ttc", tag)
    assert any("lt(t,1.4)" in f for f in filters) and any("gte(t,1.4)" in f for f in filters)
    assert not any("gte(t,2.6)" in f for f in filters)


# --- the promo's own switches: AI label, on-screen text, ambience ---------------------------------------------


def test_with_the_promos_switches_off_a_shot_carries_no_text_at_all_and_the_end_card_no_label(story, tmp_path):
    assert story.data["promo"]["ai_label"] is False and story.data["promo"]["on_screen_text"] is False
    clips = []
    for i in range(6):
        (tmp_path / f"c{i}.mp4").write_bytes(b"x")
        clips.append(str(tmp_path / f"c{i}.mp4"))
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd)
        log = "Duration: 00:00:05.04,\nSummary:\n I: -20.0 LUFS" if "ebur128" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=log)

    result = ap.compose(story, clips, str(tmp_path / "o" / "x.mp4"), ffmpeg="ffmpeg", font="f.ttc", run=run)
    graphs = [c[c.index("-filter_complex") + 1] for c in cmds if "-filter_complex" in c]
    assert not any("drawtext" in g for g in graphs[:6])  # the six shots are bare picture
    assert "drawtext" in graphs[6] and "ai_tag" not in graphs[6]  # the end card keeps its own words, not the label
    assert result["ai_tag"] is None and result["on_screen_text"] is False and result["subtitles"] is False
    assert not any("ai_tag" in " ".join(c) for c in cmds)


def test_the_switches_default_to_on_so_leaving_them_out_never_drops_the_label(story, tmp_path):
    assert ap.shot_filters(str(tmp_path), story.data["scenes"][4], "f", "TAG", subtitles=True)[0] == "TAG"
    assert ap.shot_filters(str(tmp_path), story.data["scenes"][1], "f", None, on_screen_text=False) == []
    card = ap.end_card_command("ffmpeg", "e.mkv", [("phrase", ["p.txt"])], "C:\\f.ttc", "bg.png", None)
    assert "ai_tag" not in " ".join(card) and "drawtext" in card[card.index("-filter_complex") + 1]


def test_the_ambience_bed_is_pink_noise_mixed_far_under_the_cut_and_the_picture_is_copied():
    cmd = ap.bed_command("ffmpeg", "body.mkv", "bed.mkv", 31.0)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert f"volume={ap.BED_LUFS - ap.BED_RAW_LUFS:.2f}dB" in graph and "amix=inputs=2:duration=first,volume=2" in graph
    assert "anoisesrc=color=pink" in cmd[cmd.index("-f") + 3] and "seed=7" in " ".join(cmd)
    assert cmd[cmd.index("-c:v") + 1] == "copy" and cmd[cmd.index("-t") + 1] == "31.000"


@pytest.mark.skipif(not dc.find_ffmpeg(), reason="needs ffmpeg")
def test_the_bed_really_measures_what_the_constants_say_and_sits_under_a_quiet_clip(tmp_path):
    ffmpeg = dc.find_ffmpeg()

    def loudness(args):
        log = subprocess.run([ffmpeg, "-hide_banner", "-nostats", *args, "-af", "ebur128", "-f", "null", "-"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace").stderr  # fmt: skip
        return dc.integrated_loudness(log)

    raw = loudness(["-f", "lavfi", "-i", ap.BED_FILTER.format(seconds=10)])
    assert abs(raw - ap.BED_RAW_LUFS) < 0.6, raw  # BED_RAW_LUFS is what this chain measures
    silent = str(tmp_path / "silent.mkv")
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x112:r=24:d=4",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "pcm_s16le", silent], check=True)  # fmt: skip
    out = str(tmp_path / "bed.mkv")
    subprocess.run(ap.bed_command(ffmpeg, silent, out, 4.0), check=True, capture_output=True)
    assert abs(loudness(["-i", out]) - ap.BED_LUFS) < 1.0  # alone, the bed lands where BED_LUFS puts it


@pytest.mark.skipif(not dc.find_ffmpeg(), reason="needs ffmpeg")
def test_the_bed_leaves_her_own_sound_at_the_same_level(tmp_path):
    ffmpeg = dc.find_ffmpeg()
    tone = str(tmp_path / "tone.mkv")
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x112:r=24:d=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4,aformat=channel_layouts=stereo",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", tone], check=True)  # fmt: skip
    out = str(tmp_path / "bed.mkv")
    subprocess.run(ap.bed_command(ffmpeg, tone, out, 4.0), check=True, capture_output=True)

    def level(path):
        log = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace").stderr  # fmt: skip
        return dc.integrated_loudness(log)

    assert abs(level(out) - level(tone)) < 0.3  # a bed about 26 dB under her changes it by a few hundredths


def test_a_record_whose_picture_is_gone_does_not_block_the_next_image(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    _png(os.path.join(local.scene_dir(1), "one.png"))
    _png(os.path.join(local.scene_dir(1), "two.png"))
    ap.adopt_stills(local)
    os.remove(cs.candidate_paths(local, 1, "B")["png"])  # the user took the second picture away again
    _png(os.path.join(local.scene_dir(1), "again.png"))
    adopted = ap.adopt_stills(local)
    assert [(a["scene"], a["letter"]) for a in adopted] == [(1, "B")]  # the freed slot is used, no "no room" error
    assert os.path.isfile(cs.candidate_paths(local, 1, "B")["png"])


def test_the_same_picture_saved_into_another_shot_is_refused_not_registered(story, tmp_path):
    local = cs.Story(story.data, str(tmp_path))
    _png(os.path.join(local.scene_dir(3), "new.png"))
    ap.adopt_stills(local)
    twin = open(cs.candidate_paths(local, 3, "A")["png"], "rb").read()
    os.makedirs(local.scene_dir(5), exist_ok=True)
    with open(os.path.join(local.scene_dir(5), "new.png"), "wb") as f:
        f.write(twin)  # a mix-up: shot 3's picture saved as shot 5's
    with pytest.raises(gc.UsageError, match="scene 3 候選 A 是同一張圖"):
        ap.adopt_stills(local)
    assert os.path.isfile(os.path.join(local.scene_dir(5), "new.png"))  # left where it is, nothing recorded
    assert not os.path.exists(cs.candidate_paths(local, 5, "A")["meta"])


def test_a_batch_id_must_carry_its_story_so_two_promos_never_share_job_ids(stories, tmp_path):
    by_id = {st.id: st for st in stories}
    assert [ap.default_batch_id(by_id[i]) for i in by_id] == [
        "amusement-park-v1",
        "cooking-v1",
        "coffee-v1",
        "claw-machine-v1",
        "cat-cafe-v1",
        "movie-v1",
        "night-view-v1",
        "travel-v1",
        "picnic-v1",
    ]
    local = cs.Story(copy.deepcopy(by_id["promo_03_coffee"].data), str(tmp_path))
    for scene in local.data["scenes"]:
        n = scene["scene"]
        paths = cs.candidate_paths(local, n, "A")
        os.makedirs(os.path.dirname(paths["png"]), exist_ok=True)
        _png(paths["png"])
        cs._write_json(paths["meta"], {"seed": 1})
        cs.review(local, n, approve="A")
    with pytest.raises(gc.UsageError, match="要包含 'coffee'"):
        ap.write_manifest(local, "v1")  # the id both promos used when a dry run wiped the other's work folders
    manifest = cs._read_json(ap.write_manifest(local))
    assert manifest["batch_id"] == "coffee-v1" and [j["job_id"] for j in manifest["jobs"]][0] == "coffee-v1-sc01"


def test_a_story_with_extra_takes_takes_up_to_four_images_per_shot(stories, tmp_path):
    claw = next(s for s in stories if s.id == "promo_04_claw_machine")
    local = cs.Story(copy.deepcopy(claw.data), str(tmp_path))
    for name in ("a.png", "b.png", "c.png", "d.png"):
        _png(os.path.join(local.scene_dir(1), name))
    adopted = ap.adopt_stills(local)
    assert [a["letter"] for a in adopted] == ["A", "B", "C", "D"]
    assert os.path.basename(cs.candidate_paths(local, 1, "D")["png"]) == "candidate_04.png"
    _png(os.path.join(local.scene_dir(1), "e.png"))
    with pytest.raises(gc.UsageError, match="A、B、C、D"):
        ap.adopt_stills(local)
