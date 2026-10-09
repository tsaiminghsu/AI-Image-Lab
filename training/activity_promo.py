"""Social promo cut for one ai-companion outing (projects/activity_promo/).

A promo is a story.json in the concept_stills format with `scene_count`, a top-level `promo` block (hook, end
card) and a per-scene `overlay` (the character's line, or a question with choice chips). The stills come from
concept_stills.py --root projects/activity_promo and the clips from the H3 skill; this module only does the two
steps that are specific to a promo (clips and cuts live under the story's videos/ folder):

    manifest  the run.py --manifest file for the approved scenes (refuses an unapproved one)
    adopt-stills  take the images made outside the repo (ChatGPT) that were dropped into images/scene_NN/ - any
              name, png/jpg/webp - and give each its candidate_NN.png and record, ready for review
    gpt-prompts  GPT_PROMPTS.md: the character reference prompt and every shot's prompt, built from story.json,
              for making the stills in ChatGPT instead of on the local GPU
    boundaries  boundary_sheet.html: for every cut, the frames on both sides next to each other, their 50% blend
              (what the viewer sees mid-dissolve) and each frame's brightness and warmth - data for the person
              choosing, never a choice
    cut       H3 clips -> hook, captions, choice chips, AI tag -> end card, joined with short dissolves ->
              one 1080x1920 mp4 at -14 LUFS

The choice chips are the point of the format: one of them lights up mid-shot, which is what shows the viewer
that the app lets them decide. Every frame carries the AI tag (the H3 licence asks for it).

Command builders are pure; `compose` runs them. Loudness follows drama_compose: measure, fixed gain, limiter.
"""

import argparse
import itertools
import json
import os
import re
import struct
import subprocess

import concept_cut as cc
import concept_stills as cs
import drama_compose as dc
import generate_character as gc

PROJECT_ROOT = os.path.join(cs.REPO_ROOT, "projects", "activity_promo")
AI_TAG = "AI 生成影片｜MiniMax H3"
TAG_SIZE, TAG_TOP = 30, 196
MIN_SECONDS, MAX_SECONDS = 20, 32       # the length this format was planned for (six 5 s shots + card - dissolves = 31 s); outside it is a spec mistake
END_CARD_SECONDS = 2.5
TRANSITION_SECONDS = 0.25               # 6 frames at 24 fps; the bundled ffmpeg 4.2 has no xfade, see dissolve_command
MAX_TRANSITION_SECONDS = 0.5
CUT_SECONDS = 1 / dc.FPS                # a dissolve of one frame is a cut
# A very light ambience under the whole cut, so the stretches where H3 rendered only her voice (about -50 dB)
# are not dead air. Synthetic pink noise, band-limited like distant park air; BED_RAW_LUFS is what this exact
# chain measures at amplitude 0.5 (a test re-measures it), BED_LUFS is where it sits before the final gain:
# about 25 dB under her lines.
BED_FILTER = "anoisesrc=color=pink:sample_rate=48000:amplitude=0.5:seed=7:duration={seconds:.3f},"\
             "highpass=f=60,lowpass=f=2500,aformat=channel_layouts=stereo"
BED_RAW_LUFS, BED_LUFS = -24.5, -48.0
HOOK_SECONDS = 2.5
HOOK_SIZE, HOOK_TOP, HOOK_CHARS = 84, 300, 10
QUESTION_SIZE, QUESTION_CHARS = 50, 16
CHIP_SIZE, CHIP_TOP, CHIP_PITCH, CHIP_PADDING = 56, 940, 110, 22
CHIP_COLOR, CHIP_PICKED_COLOR = "black@0.55", "0xFF5C8A@0.92"
PICK_AT = 2.6                           # seconds into the shot: long enough to read three chips first
MIN_CHOICES, MAX_CHOICES = 2, 3         # a fourth chip would reach the subtitle band
CARD_FIELDS = ("label", "phrase", "translation", "note")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
STILL_SOURCE = "ChatGPT image made by the user outside this repo"
BOUNDARY_DIR = "_boundaries"            # under the story's images/: blends and frames pulled from clips
BLEND_W, BLEND_H = 360, 640


# ---------------------------------------------------------------- spec

def promo_problem(story):
    """Why this story cannot be cut as a promo, or None."""
    promo = story.data.get("promo")
    if not isinstance(promo, dict) or not str(promo.get("hook", "")).strip():
        return "缺少 promo.hook（開場的一句話）"
    card = promo.get("end_card")
    if not isinstance(card, dict) or not card.get("phrase") or not card.get("note"):
        return "promo.end_card 要有 phrase 和 note"
    if any(not isinstance(card.get(k, ""), str) for k in CARD_FIELDS):
        return "promo.end_card 的欄位都要是文字"
    for s in story.data["scenes"]:
        where = f"scene {s['scene']}"
        if "video" not in s:
            return f"{where} 沒有 video（H3 的鏡頭描述）"
        t_in = s.get("transition_in")
        if t_in is not None and t_in != "cut" and (
            isinstance(t_in, bool) or not isinstance(t_in, (int, float)) or not 0 < t_in <= MAX_TRANSITION_SECONDS
        ):
            return f"{where} 的 transition_in 要是 \"cut\" 或 0-{MAX_TRANSITION_SECONDS} 秒"
        overlay = s.get("overlay", {})
        if not isinstance(overlay, dict) or not isinstance(overlay.get("caption", ""), str):
            return f"{where} 的 overlay.caption 要是文字"
        if "choices" in overlay:
            choices, picked = overlay["choices"], overlay.get("picked")
            if (not isinstance(choices, list) or not MIN_CHOICES <= len(choices) <= MAX_CHOICES
                    or not all(isinstance(c, str) and c.strip() for c in choices)):
                return f"{where} 的 overlay.choices 要是 {MIN_CHOICES}-{MAX_CHOICES} 個選項"
            if isinstance(picked, bool) or not isinstance(picked, int) or not 0 <= picked < len(choices):
                return f"{where} 的 overlay.picked 要是被選中的選項位置（從 0 起算）"
            if not str(overlay.get("question", "")).strip():
                return f"{where} 有 choices 就要有 question"
            if s["video"].get("duration", cs.VIDEO_DEFAULT_SECONDS) <= PICK_AT + 1:
                return f"{where} 太短，選項亮起後沒有時間看"
    total = planned_seconds(story)
    if not MIN_SECONDS <= total <= MAX_SECONDS:
        return f"全長 {total:g} 秒，宣傳片要在 {MIN_SECONDS}-{MAX_SECONDS} 秒"
    return None


def shot_transitions(story, default=TRANSITION_SECONDS):
    """Seconds of overlap at each boundary after shot 1..n (the last one is into the end card). A shot may set
    `transition_in` to "cut" (one frame) or a number, for a cut where the two pictures do not line up - the
    faces differ in size or place - and a dissolve would only show two faces at once. default 0 means hard cuts
    everywhere."""
    if not default:
        return [0.0] * len(story.data["scenes"])
    out = []
    for shot in story.data["scenes"][1:]:
        t = shot.get("transition_in", default)
        out.append(CUT_SECONDS if t == "cut" else t)
    return [*out, default]


def planned_seconds(story, transition=TRANSITION_SECONDS):
    """Running time of the cut: every boundary (shot to shot, last shot to the end card) overlaps by its transition."""
    shots = story.data["scenes"]
    return sum(s.get("video", {}).get("duration", cs.VIDEO_DEFAULT_SECONDS) for s in shots) \
        + END_CARD_SECONDS - sum(shot_transitions(story, transition))


def timeline(story, transition=TRANSITION_SECONDS):
    """One line per shot, for reading the promo before any GPU or CU is spent."""
    lines, t = [], 0.0
    overlaps = shot_transitions(story, transition)
    for i, s in enumerate(story.data["scenes"]):
        d = s["video"].get("duration", cs.VIDEO_DEFAULT_SECONDS)
        overlay = s.get("overlay", {})
        text = overlay.get("caption", "")
        if "choices" in overlay:
            chips = "／".join(f"[{c}]" if i == overlay["picked"] else c for i, c in enumerate(overlay["choices"]))
            text = f"{overlay['question']} {chips} → {text}"
        lines.append(f"{t:5.2f}-{t + d:5.2f}s  {s['title']}  ｜ {text}")
        t += d - overlaps[i]
    lines.append(f"{t:5.2f}-{t + END_CARD_SECONDS:5.2f}s  結尾字卡 ｜ {story.data['promo']['end_card']['phrase']}")
    return lines


def newest_clip(story, scene_number):
    """videos/scene_NN/clip_vN.mp4 with the highest N, or None."""
    directory = story.clip_dir(scene_number)
    versions = [(int(m.group(1)), name) for name in (os.listdir(directory) if os.path.isdir(directory) else [])
                if (m := re.fullmatch(r"clip_v(\d+)\.mp4", name))]
    return os.path.join(directory, max(versions)[1]) if versions else None


def clip_paths(story, override=None):
    """The clip of every scene, in order: --clips, else the one clips.json names for the scene (concept_api
    records it, and it is where an older version is pinned by hand), else the scene's newest clip_vN.mp4."""
    scenes = story.data["scenes"]
    if override:
        if len(override) != len(scenes):
            raise gc.UsageError(f"--clips 要剛好 {len(scenes)} 支，依鏡頭順序")
        return [os.path.abspath(p) for p in override]
    path = os.path.join(story.dir, "clips.json")
    recorded = cs._read_json(path) if os.path.isfile(path) else {}
    paths, missing = [], []
    for s in scenes:
        key = f"scene_{s['scene']:02d}"
        clip = os.path.join(cs.REPO_ROOT, recorded[key]["path"]) if key in recorded else newest_clip(story, s["scene"])
        paths.append(clip)
        if clip is None:
            missing.append(s["scene"])
    if missing:
        raise gc.UsageError(f"scene {missing} 還沒有 H3 影片（{story.clip_dir(missing[0])}）；或用 --clips 指定")
    return paths


# ---------------------------------------------------------------- filters

def _when(filters, condition):
    return [f"{f}:enable='{condition}'" for f in filters]


def tag_filter(textfile, font):
    return (f"drawtext=fontfile='{dc.filter_path(font)}':textfile='{dc.filter_path(textfile)}':fontsize={TAG_SIZE}"
            f":fontcolor=white@0.85:borderw=3:bordercolor=black@0.7:x=(w-text_w)/2:y={TAG_TOP}")


def hook_filters(line_files, font):
    return _when(dc._block(line_files, font, HOOK_SIZE, "white", HOOK_TOP, border=6), f"lt(t,{HOOK_SECONDS})")


def _chip(textfile, font, y, color):
    return (f"drawtext=fontfile='{dc.filter_path(font)}':textfile='{dc.filter_path(textfile)}':fontsize={CHIP_SIZE}"
            f":fontcolor=white:box=1:boxcolor={color}:boxborderw={CHIP_PADDING}:x=(w-text_w)/2:y={y}")


def choice_filters(question_files, chip_files, picked, font, pick_at=PICK_AT):
    """The question above a column of chips; the picked chip changes colour at pick_at."""
    top = CHIP_TOP - CHIP_PADDING - 30 - dc._block_height(len(question_files), QUESTION_SIZE)
    filters = dc._block(question_files, font, QUESTION_SIZE, dc.TRANSLATION_COLOR, top, border=4)
    for i, f in enumerate(chip_files):
        y = CHIP_TOP + i * CHIP_PITCH
        if i == picked:
            filters += _when([_chip(f, font, y, CHIP_COLOR)], f"lt(t,{pick_at})")
            filters += _when([_chip(f, font, y, CHIP_PICKED_COLOR)], f"gte(t,{pick_at})")
        else:
            filters.append(_chip(f, font, y, CHIP_COLOR))
    return filters


def shot_command(ffmpeg, clip, out, duration, filters, gain_db):
    """One H3 clip filled to 1080x1920 with its overlays, keeping the clip's own (levelled) sound."""
    vchain = ",".join([dc.clip_filter(duration, dc.FPS), *filters])
    dur = f"{duration:.3f}"
    graph = (f"[0:v]{vchain},format=yuv420p[v];"
             f"[0:a]volume={gain_db:.2f}dB,aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{dur}[a]")
    return [ffmpeg, *dc.QUIET, "-i", clip, "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", dur,
            "-r", str(dc.FPS), *dc.VIDEO_CODEC, *dc.SEGMENT_AUDIO_CODEC, out]


def end_card_command(ffmpeg, out, blocks, font, background, tag):
    cmd = dc.card_command(ffmpeg, out=out, duration=dc.frame_exact(END_CARD_SECONDS), blocks=blocks, font=font,
                          background=background, fps=dc.FPS)
    if tag:
        i = cmd.index("-filter_complex") + 1
        tagged = cmd[i].replace(",format=yuv420p[v]", f",{tag},format=yuv420p[v]")
        assert tagged != cmd[i], "card_command's graph changed; the AI tag was not added"
        cmd[i] = tagged
    return cmd


def bed_command(ffmpeg, src, out, seconds):
    """`src` (picture + sound) with the ambience bed mixed under its sound; the picture is copied."""
    gain = BED_LUFS - BED_RAW_LUFS
    # The bundled ffmpeg 4.2 has no amix `normalize` option; amix scales each of its two inputs by 1/2 either way,
    # so volume=2 puts her sound back exactly where it was (a test measures it).
    graph = f"[1:a]volume={gain:.2f}dB[bed];[0:a][bed]amix=inputs=2:duration=first,volume=2,aresample=48000[a]"
    return [ffmpeg, *dc.QUIET, "-i", src, "-f", "lavfi", "-i", BED_FILTER.format(seconds=seconds + 1),
            "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy", *dc.SEGMENT_AUDIO_CODEC,
            "-t", f"{seconds:.3f}", out]


def dissolve_commands(ffmpeg, segments, durations, out, transition=TRANSITION_SECONDS):
    """Join finished segments (.mkv with PCM audio) with a short dissolve at every boundary: three ffmpeg runs
    (picture, sound, mux) whose last writes `out`.

    The bundled ffmpeg is 4.2, which has no xfade, so the picture is built the long-standing way: each later
    segment is faded in on its alpha channel, shifted to start `transition` before the previous one ends, and
    overlaid on everything so far. The sound uses acrossfade over the same overlap. They are separate runs because
    this ffmpeg build segfaults when the overlay chain and acrossfade share one filter graph (each alone is fine),
    and the copy mux keeps the two the same length. `durations` are the segments' frame-exact lengths; the result
    is sum(durations) minus the overlap at each boundary. `transition` is one number for every boundary or a
    list with one number per boundary."""
    n = len(segments)
    if n != len(durations) or n < 2:
        raise ValueError("need one duration per segment and at least two segments")
    overlaps = list(transition) if isinstance(transition, (list, tuple)) else [transition] * (n - 1)
    if len(overlaps) != n - 1:
        raise ValueError(f"{n} segments have {n - 1} boundaries, not {len(overlaps)}")
    if not all(0 < t <= MAX_TRANSITION_SECONDS for t in overlaps) or min(durations) < 2 * max(overlaps):
        raise ValueError(f"a dissolve needs 0 < t <= {MAX_TRANSITION_SECONDS} and shots of at least twice that")
    picture, sound, start = [], [], durations[0]
    video, audio = "0:v", "0:a"
    for i in range(1, n):
        transition = overlaps[i - 1]
        offset = start - transition
        picture.append(f"[{i}:v]format=yuva420p,fade=t=in:st=0:d={transition:.3f}:alpha=1,setpts=PTS-STARTPTS+{offset:.3f}/TB[f{i}]")
        picture.append(f"[{video}][f{i}]overlay=eof_action=repeat:format=yuv420[o{i}]")
        sound.append(f"[{audio}][{i}:a]acrossfade=d={transition:.3f}:c1=tri:c2=tri[x{i}]")
        video, audio, start = f"o{i}", f"x{i}", offset + durations[i]
    picture.append(f"[{video}]format=yuv420p[v]")
    inputs = [arg for seg in segments for arg in ("-i", seg)]
    base = os.path.splitext(out)[0]
    video_part, audio_part = base + "_picture.mkv", base + "_sound.wav"
    total = f"{start:.3f}"
    return [
        [ffmpeg, *dc.QUIET, *inputs, "-filter_complex", ";".join(picture), "-map", "[v]", "-t", total,
         "-r", str(dc.FPS), *dc.VIDEO_CODEC, video_part],
        [ffmpeg, *dc.QUIET, *inputs, "-filter_complex", ";".join(sound), "-map", f"[{audio}]", "-t", total,
         "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", audio_part],
        [ffmpeg, *dc.QUIET, "-i", video_part, "-i", audio_part, "-map", "0:v", "-map", "1:a", "-c", "copy",
         "-t", total, out],
    ]


# ---------------------------------------------------------------- compose

def _lines(work, name, text, cjk, latin):
    paths = []
    for i, line in enumerate(dc.wrap_lines(text, cjk, latin)):
        p = os.path.join(work, f"{name}_{i}.txt")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(line)
        paths.append(p)
    return paths


def shot_filters(work, scene, font, tag, hook=None, subtitles=True, on_screen_text=True):
    """Every overlay of one shot, AI tag first. On a choice shot the caption is the character's reply, so it
    waits for the pick. subtitles=False leaves the spoken lines to the character's voice: the hook, the question
    and the chips stay, the caption is not drawn. A choice shot may set overlay.pick_at (seconds) to light the
    chip just before she answers. tag=None draws no AI label; on_screen_text=False draws no text on the shot at
    all (no hook, question, chips or caption)."""
    n = scene["scene"]
    overlay = scene.get("overlay", {})
    filters = [tag] if tag else []
    if not on_screen_text:
        return filters
    if hook:
        filters += hook_filters(_lines(work, f"s{n}_hook", hook, HOOK_CHARS, 20), font)
    caption = dc.subtitle_filters(_lines(work, f"s{n}_caption", overlay.get("caption", ""),
                                         dc.SUBTITLE_CHARS_PER_LINE, dc.SUBTITLE_LATIN_CHARS_PER_LINE), font) if subtitles else []
    if "choices" in overlay:
        question = _lines(work, f"s{n}_question", overlay["question"], QUESTION_CHARS, 30)
        chips = [_lines(work, f"s{n}_chip{i}", c, 12, 24)[0] for i, c in enumerate(overlay["choices"])]
        pick_at = overlay.get("pick_at", PICK_AT)
        filters += choice_filters(question, chips, overlay["picked"], font, pick_at)
        caption = _when(caption, f"gte(t,{pick_at})")
    return filters + caption


def compose(story, clips, out, *, ffmpeg=None, font=None, run=subprocess.run, transition=TRANSITION_SECONDS, bed=True):
    """clips: one path per scene, in order. transition = seconds of dissolve at every boundary (0 = hard cuts).
    bed=False leaves out the ambience bed. The promo's own switches (all default on): `ai_label` draws the AI tag,
    `on_screen_text` draws the hook, question and chips, `subtitles` the caption lines. Returns the numbers worth
    reporting."""
    if not 0 <= transition <= MAX_TRANSITION_SECONDS or dc.frame_exact(transition) != transition:
        raise gc.UsageError(f"疊化秒數要在 0-{MAX_TRANSITION_SECONDS}，而且是整數個影格（{dc.FPS} fps）")
    problem = promo_problem(story)
    if problem:
        raise gc.UsageError(f"{story.id}: {problem}")
    ffmpeg = ffmpeg or dc.find_ffmpeg()
    if not ffmpeg:
        raise gc.UsageError("找不到 ffmpeg（設定 DRAMA_FFMPEG 或裝進 PATH）")
    font = font or dc.find_font()
    if not font:
        raise gc.UsageError("找不到中文字型（設定 DRAMA_FONT）")
    for p in clips:
        if not os.path.isfile(p):
            raise gc.UsageError(f"找不到片段 {p}")

    def do(cmd, cwd=None):
        done = run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
        if done.returncode != 0:
            raise subprocess.CalledProcessError(done.returncode, cmd, stderr=done.stderr)
        return done.stderr

    out = os.path.abspath(out)
    work = os.path.splitext(out)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    promo = story.data["promo"]
    tag = tag_filter(_lines(work, "ai_tag", AI_TAG, 30, 60)[0], font) if promo.get("ai_label", True) else None

    segments, measured, seconds = [], [], []
    for i, (scene, clip) in enumerate(zip(story.data["scenes"], clips)):
        log = do(dc.loudness_command(ffmpeg, clip))
        length = cc.duration_seconds(log)
        if length is None:
            raise gc.UsageError(f"讀不到片段長度：{clip}")
        loudness = dc.integrated_loudness(log)
        duration = dc.frame_exact(length)
        segment = os.path.join(work, f"shot_{scene['scene']:02d}.mkv")
        filters = shot_filters(work, scene, font, tag, hook=promo["hook"] if i == 0 else None,
                               subtitles=promo.get("subtitles", True), on_screen_text=promo.get("on_screen_text", True))
        do(shot_command(ffmpeg, clip, segment, duration, filters, cc.clip_gain_db(loudness)))
        segments.append(segment)
        measured.append(loudness)
        seconds.append(duration)

    background = os.path.join(work, "last_frame.png")
    do([ffmpeg, *dc.QUIET, "-sseof", "-0.2", "-i", clips[-1], "-frames:v", "1", background])
    card = promo["end_card"]
    blocks = [(k, _lines(work, f"end_{k}", card[k], *dc.CARD_BLOCKS[k][2:])) for k in CARD_FIELDS if card.get(k)]
    end = os.path.join(work, "end_card.mkv")
    do(end_card_command(ffmpeg, end, blocks, font, background, tag))
    segments.append(end)
    seconds.append(dc.frame_exact(END_CARD_SECONDS))

    body = os.path.join(work, "body.mkv")
    overlaps = shot_transitions(story, transition)
    if transition:
        for cmd in dissolve_commands(ffmpeg, segments, seconds, body, overlaps):
            do(cmd)
    else:
        with open(os.path.join(work, "concat.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(dc.concat_list(segments, base_dir=work))
        do(dc.concat_command(ffmpeg, "concat.txt", body, final=False), cwd=work)
    if bed:
        total = cc.duration_seconds(do(dc.loudness_command(ffmpeg, body))) or sum(seconds) - sum(overlaps)
        under = os.path.join(work, "body_bed.mkv")
        do(bed_command(ffmpeg, body, under, total))
        body = under
    mix = dc.integrated_loudness(do(dc.loudness_command(ffmpeg, body)))
    do(dc.finish_command(ffmpeg, body, out, dc.loudness_filter(mix)))
    final = dc.integrated_loudness(do(dc.loudness_command(ffmpeg, out)))
    return {"output": out, "shots": len(clips), "seconds": round(sum(seconds) - sum(overlaps), 2), "transition_seconds": transition,
            "transitions": [round(t, 3) for t in overlaps], "size": f"{dc.OUT_W}x{dc.OUT_H}",
            "clip_loudness_lufs": measured, "mix_loudness_lufs": mix, "final_loudness_lufs": final, "ai_tag": AI_TAG if tag else None,
            "on_screen_text": promo.get("on_screen_text", True), "subtitles": promo.get("subtitles", True),
            "ambience_bed_lufs": BED_LUFS if bed else None}


def story_slug(story):
    """'promo_03_coffee' -> 'coffee', 'promo_01_amusement_park' -> 'amusement-park': what makes a batch id this story's own."""
    return story.id.split("_", 2)[-1].replace("_", "-")


def default_batch_id(story, version=1):
    return f"{story_slug(story)}-v{version}"


def write_manifest(story, batch_id=None, scene_numbers=None):
    """The H3 batch for this promo's scenes (default all; or just the ones being redone), first frame = each
    scene's approved master.png; each clip lands in the scene's videos folder as the next clip_vN.mp4."""
    import concept_api   # FastAPI app module; only its pure helpers are used

    batch_id = batch_id or default_batch_id(story)
    if story_slug(story) not in batch_id:
        # run.py stages a job in output/.work/<job_id> and a --dry-run deletes that folder: two promos that both
        # used "v1-sc04" had a dry run of one wipe the first frames of the other's running batch (2026-10-05).
        raise gc.UsageError(f"batch id {batch_id!r} 要包含 {story_slug(story)!r}：不同短片的 job id 撞名，"
                            "output/.work/<job_id> 會被互相刪掉")
    numbers = sorted(scene_numbers) if scene_numbers else [s["scene"] for s in story.data["scenes"]]
    unknown = [n for n in numbers if n not in {s["scene"] for s in story.data["scenes"]}]
    if unknown:
        raise gc.UsageError(f"{story.id} 沒有 scene {unknown}")
    problems = concept_api.video_problems(story, numbers)
    if problems:
        raise gc.UsageError(f"{story.id}: " + "；".join(problems))
    manifest = concept_api.video_manifest(story, numbers, batch_id)
    path = os.path.join(story.dir, "videos", "_batches", f"{batch_id}.json")
    cs._write_json(path, manifest)
    return path


# ---------------------------------------------------------------- stills made outside the repo

def png_size(path):
    """(width, height) from a PNG's IHDR chunk, or None for anything else."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def unadopted_images(story, number):
    """Image files in a scene's folder that have no record yet, oldest first. master.png and anything with a
    sibling .json (a candidate this tool already knows) are left alone."""
    directory = story.scene_dir(number)
    if not os.path.isdir(directory):
        return []
    found = []
    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTENSIONS and stem != "master" and not os.path.isfile(os.path.join(directory, stem + ".json")):
            found.append(os.path.join(directory, name))
    return sorted(found, key=lambda path: (os.path.getmtime(path), path))


def refresh_replaced(story):
    """A candidate whose file was overwritten after it was recorded (the user re-made the shot and saved it over
    the old one) gets its record brought up to date: hash, size, time. The old picture is gone, so the record says
    so. An APPROVED scene is refused: its master.png would no longer be the candidate it was copied from."""
    refreshed = []
    for scene in story.data["scenes"]:
        n = scene["scene"]
        for letter in cs.LETTERS:
            paths = cs.candidate_paths(story, n, letter)
            if not (os.path.isfile(paths["png"]) and os.path.isfile(paths["meta"])):
                continue
            meta = cs._read_json(paths["meta"])
            digest = cs._sha256(paths["png"])
            if meta.get("image_sha256") == digest:
                continue
            if cs.load_review(story, n)["status"] == cs.APPROVED:
                raise gc.UsageError(f"scene {n}: 圖被換掉了，但這個鏡頭已 APPROVED；先 review --regenerate 再重新登記")
            size = png_size(paths["png"])
            meta.update({"image_sha256": digest, "width": size[0] if size else None, "height": size[1] if size else None,
                         "replaced_at": cs._now(), "note": "the image file was replaced after it was recorded"})
            cs._write_json(paths["meta"], meta)
            refreshed.append({"scene": n, "letter": letter, "path": paths["png"], "size": size})
    return refreshed


def recorded_digests(story):
    """{sha256: 'scene N candidate X'} of every still of this story that has a record."""
    out = {}
    for scene in story.data["scenes"]:
        for letter in cs.LETTERS:
            paths = cs.candidate_paths(story, scene["scene"], letter)
            if os.path.isfile(paths["png"]) and os.path.isfile(paths["meta"]):
                out.setdefault(cs._sha256(paths["png"]), f"scene {scene['scene']} 候選 {letter}")
    return out


def adopt_stills(story, *, ffmpeg=None, run=subprocess.run):
    """Give every unrecorded image its candidate_NN.png (converting webp/jpg with ffmpeg) and a record, then hand
    the scene to review. Nothing is approved here: choosing is the user's. Returns what was adopted."""
    adopted = []
    for scene in story.data["scenes"]:
        n = scene["scene"]
        files = unadopted_images(story, n)
        if not files:
            continue
        # a slot is taken only by a picture that has its record; a record whose picture is gone is just a leftover
        taken = {x for x in cs.LETTERS if all(os.path.isfile(v) for v in (cs.candidate_paths(story, n, x)["png"],
                                                                          cs.candidate_paths(story, n, x)["meta"]))}
        slots = [x for x in cs.LETTERS if x in scene["compositions"]]   # A, B, and C, D where the story wrote them
        free = [x for x in slots if x not in taken]
        if len(files) > len(free):
            raise gc.UsageError(f"scene {n}: {len(files)} 張新圖，但只剩 {len(free)} 個候選位置（{'、'.join(slots)}）；多的請先移走")
        known = recorded_digests(story)
        for path in files:   # the same picture saved into two shots is a mix-up, never a second candidate
            twin = known.get(cs._sha256(path))
            if twin:
                raise gc.UsageError(f"scene {n}: {os.path.basename(path)} 跟 {twin} 是同一張圖（檔案內容完全一樣），沒有登記；請換成這個鏡頭自己的圖")
        staged = []
        for i, path in enumerate(files):   # two steps, so candidate_02.png dropped in as the only file cannot clash
            temp = os.path.join(os.path.dirname(path), f"_adopt_{i}.png")
            if os.path.splitext(path)[1].lower() == ".png":
                os.replace(path, temp)
            else:
                ffmpeg = ffmpeg or dc.find_ffmpeg()
                if not ffmpeg:
                    raise gc.UsageError("找不到 ffmpeg，無法轉換非 PNG 的圖（設定 DRAMA_FFMPEG 或裝進 PATH）")
                done = run([ffmpeg, *dc.QUIET, "-i", path, "-frames:v", "1", temp], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
                if done.returncode != 0:
                    raise subprocess.CalledProcessError(done.returncode, done.args, stderr=done.stderr)
                os.remove(path)
            staged.append(temp)
        for temp, letter in zip(staged, free):
            paths = cs.candidate_paths(story, n, letter)
            os.replace(temp, paths["png"])
            size = png_size(paths["png"])
            cs._write_json(paths["meta"], {
                "story_id": story.id, "scene_id": f"scene_{n:02d}", "candidate_id": os.path.basename(paths["png"])[:12],
                "candidate_letter": letter, "source": STILL_SOURCE, "prompt_suggested": gpt_shot_prompt(story, scene, letter),
                "composition": scene["compositions"][letter], "width": size[0] if size else None,
                "height": size[1] if size else None, "image_sha256": cs._sha256(paths["png"]), "created_at": cs._now(),
            })
            adopted.append({"scene": n, "letter": letter, "path": paths["png"], "size": size})
    cs.mark_ready_scenes([story])
    return adopted


# ---------------------------------------------------------------- prompts for ChatGPT

def gpt_reference_prompt(story, character="ruoxi"):
    c = story.data["characters"][character]
    return c["gpt_reference"].format(description=c["description"])


def gpt_shot_prompt(story, scene, letter="A"):
    """One shot as a ChatGPT image prompt: the 'same woman' line, the framing, what she does, where, in what
    light, then the promo's fixed ending (one person, no text, 9:16). Built only from story.json, so the file
    cannot drift from the spec that the cut and the tests use."""
    char = story.data["characters"][scene["characters"][0]]
    same = f"{char['gpt_same']} She wears {char['gpt_outfit']}." if char.get("gpt_outfit") else char["gpt_same"]
    return (f"{same}\n{scene['compositions'][letter]}. {scene['action'].rstrip('.')}.\n"
            f"Setting: {scene['setting']}. Lighting: {scene['lighting']}.\n{story.data['promo']['gpt_suffix']}")


def build_gpt_prompts_md(bible, story):
    fence = "```"
    lines = [
        f"# ChatGPT 生圖提示詞：{story.data['title_zh']}（{story.id}）", "",
        "由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。", "",
        "## 做法", "",
        "1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。",
        "2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。",
        "3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。",
        f"4. 存檔位置：`{os.path.relpath(story.dir, cs.REPO_ROOT).replace(chr(92), '/')}/images/scene_NN/candidate_01.png`"
        "（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。"
        "放好告訴我，記錄檔和格式轉換我來處理。",
        "5. 生出來有任何一項就重生：", "",
    ]
    lines += [f"   - {c.removeprefix('KNOCK-OUT: ')}" for c in bible["review_checklist"] if c.startswith("KNOCK-OUT")]
    lines += ["", "## 第 0 步：角色定裝照", "",
              "已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；"
              "每個鏡頭的提示詞都寫了這支的衣服。", "", f"{fence}text", gpt_reference_prompt(story), fence, ""]
    for s in story.data["scenes"]:
        n = s["scene"]
        lines += [f"## 鏡頭 {n}：{s['title']}", "",
                  f"- 時段：{s['time_of_day']}｜手上：{s['props']}｜景別：{s['framing']}",
                  f"- 接上一鏡：{s.get('bridge', '（開場）')}", "",
                  "主要取景（存成 candidate_01.png）：", "", f"{fence}text", gpt_shot_prompt(story, s, "A"), fence, "",
                  "另一種取景（存成 candidate_02.png）：", "", f"{fence}text", gpt_shot_prompt(story, s, "B"), fence, ""]
        for letter in "CD":  # extra takes, only for a story that wrote them (camera lower / a little further left)
            if letter in s["compositions"]:
                lines += [f"更多取景 {letter}（存成 candidate_{'ABCD'.index(letter) + 1:02d}.png）：", "",
                          f"{fence}text", gpt_shot_prompt(story, s, letter), fence, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- boundaries

def stats_command(ffmpeg, image):
    return [ffmpeg, "-hide_banner", "-i", image, "-frames:v", "1", "-vf", "signalstats,metadata=print", "-f", "null", "-"]


def parse_stats(log):
    """Brightness (mean luma, 16-235) and warmth (mean V minus mean U: above 0 is orange, below is blue) of one
    frame, or None. Enough to see whether the light moves one step at a time from shot to shot."""
    found = {k: re.search(rf"signalstats\.{k}AVG=([\d.]+)", log) for k in "YUV"}
    if not all(found.values()):
        return None
    y, u, v = (float(found[k].group(1)) for k in "YUV")
    return {"brightness": round(y, 1), "warmth": round(v - u, 1)}


def blend_command(ffmpeg, before, after, out):
    """The frame half-way through the dissolve: both pictures filled to one size, averaged."""
    fit = f"scale={BLEND_W}:{BLEND_H}:force_original_aspect_ratio=increase,crop={BLEND_W}:{BLEND_H},setsar=1,format=gbrp"
    graph = f"[0:v]{fit}[a];[1:v]{fit}[b];[a][b]blend=all_mode=average,format=rgb24[v]"
    return [ffmpeg, *dc.QUIET, "-i", before, "-i", after, "-filter_complex", graph, "-map", "[v]", "-frames:v", "1", out]


def frame_command(ffmpeg, clip, out, last=False):
    seek = ["-sseof", "-0.2"] if last else []
    return [ffmpeg, *dc.QUIET, *seek, "-i", clip, "-frames:v", "1", out]


def boundary_sources(story):
    """'clips' once every scene has an H3 clip (the real boundary is last frame -> first frame), else 'stills'."""
    return "clips" if all(newest_clip(story, s["scene"]) for s in story.data["scenes"]) else "stills"


def scene_stills(story, number):
    """[(label, path)]: the approved master alone, else every candidate that exists."""
    master = os.path.join(story.scene_dir(number), "master.png")
    if cs.load_review(story, number)["status"] == cs.APPROVED and os.path.isfile(master):
        return [("master", master)]
    return [(f"候選 {x}", cs.candidate_paths(story, number, x)["png"]) for x in cs.existing_letters(story, number)]


def boundary_rows(story, do, ffmpeg, source=None):
    """One row per cut: every frame on the outgoing side x every frame on the incoming side, with the blend of
    each pair and the measured light of each frame. `do(cmd)` runs ffmpeg and returns its stderr."""
    source = source or boundary_sources(story)
    work = os.path.join(story.dir, "images", BOUNDARY_DIR)
    os.makedirs(work, exist_ok=True)
    scenes = story.data["scenes"]
    outgoing, incoming = {}, {}
    for s in scenes:
        n = s["scene"]
        if source == "clips":
            clip = clip_paths(story)[scenes.index(s)]
            first, last = os.path.join(work, f"scene_{n:02d}_first.png"), os.path.join(work, f"scene_{n:02d}_last.png")
            do(frame_command(ffmpeg, clip, first))
            do(frame_command(ffmpeg, clip, last, last=True))
            incoming[n], outgoing[n] = [("第一格", first)], [("最後一格", last)]
        else:
            incoming[n] = outgoing[n] = scene_stills(story, n)
    stats = {}

    def frame(label, path):
        if path not in stats:
            stats[path] = parse_stats(do(stats_command(ffmpeg, path)))
        return {"label": label, "path": path, "stats": stats[path]}

    rows = []
    for before, after in itertools.pairwise(scenes):
        a, b = before["scene"], after["scene"]
        pairs = []
        for (i, (la, pa)), (j, (lb, pb)) in itertools.product(enumerate(outgoing[a], 1), enumerate(incoming[b], 1)):
            blend = os.path.join(work, f"blend_{a:02d}-{i}_{b:02d}-{j}.png")
            do(blend_command(ffmpeg, pa, pb, blend))
            pairs.append({"before": frame(la, pa), "after": frame(lb, pb), "blend": blend})
        rows.append({"before": before, "after": after, "pairs": pairs})
    return source, rows


def boundary_html(bible, story, source, rows, root):
    """The sheet itself. It lays the pairs out and prints the numbers; it never ranks them."""
    def rel(path):
        return cs._esc(os.path.relpath(path, root).replace("\\", "/"))

    def cell(f):
        st = f["stats"]
        numbers = f"亮度 {st['brightness']:g}｜暖色 {st['warmth']:+g}" if st else "量不到"
        return (f'<div class="c"><img src="{rel(f["path"])}" loading="lazy">'
                f'<small>{cs._esc(f["label"])}<br>{numbers}</small></div>')

    checks = "".join(f"<li>☐ {cs._esc(c)}</li>" for c in bible.get("boundary_checklist", []))
    what = "H3 影片：前一支最後一格 → 後一支第一格" if source == "clips" else "候選圖（還沒有影片）"
    out = [f"<h2>{cs._esc(story.data['title_zh'])} · {what}</h2>", f"<ul>{checks}</ul>"]
    for row in rows:
        b, a = row["before"], row["after"]
        out.append(f'<div class="scene"><b>{b["scene"]} → {a["scene"]}</b>　{cs._esc(b["title"])} → {cs._esc(a["title"])}'
                   f'<div>接的動作：{cs._esc(a.get("enters_on", ""))}｜光線：{cs._esc(b.get("time_of_day", ""))} → '
                   f'{cs._esc(a.get("time_of_day", ""))}｜景別：{cs._esc(b.get("framing", ""))} → '
                   f'{cs._esc(a.get("framing", ""))}</div>')
        if not row["pairs"]:
            out.append("<div>這個交界有一邊還沒有圖</div>")
        for pair in row["pairs"]:
            out.append(f'<div class="row">{cell(pair["before"])}<div class="c"><img src="{rel(pair["blend"])}" '
                       f'loading="lazy"><small>疊化中間（各 50%）</small></div>{cell(pair["after"])}</div>')
        out.append("</div>")
    return "\n".join(out)


def build_boundary_sheet(bible, stories, root, *, ffmpeg=None, run=subprocess.run, source=None):
    ffmpeg = ffmpeg or dc.find_ffmpeg()
    if not ffmpeg:
        raise gc.UsageError("找不到 ffmpeg（設定 DRAMA_FFMPEG 或裝進 PATH）")

    def do(cmd):
        done = run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            raise subprocess.CalledProcessError(done.returncode, cmd, stderr=done.stderr)
        return done.stderr

    body = [boundary_html(bible, s, *boundary_rows(s, do, ffmpeg, source), root) for s in stories]
    page = ("<!doctype html><html lang=\"zh-Hant\"><head><meta charset=\"utf-8\"><title>Boundaries</title>"
            "<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}"
            ".scene{border-top:1px solid #444;padding:12px 0}.row{display:flex;gap:10px;margin:8px 0}"
            ".c{width:220px}.c img{width:220px;display:block}.c small{color:#aaa}ul{margin:.3em 0}</style></head><body>"
            "<h1>交界對照表</h1><p>每一列是一個轉場：左邊是前一鏡，右邊是後一鏡，中間是疊化到一半時觀眾看到的畫面。"
            "這張表只排資料，不替你選。</p>" + "\n".join(body) + "</body></html>\n")
    path = os.path.join(root, "boundary_sheet.html")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    return path


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description="Social promo cuts of ai-companion outings")
    p.add_argument("--root", default=PROJECT_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="validate every promo and print its timeline (no GPU, no CU)")
    m = sub.add_parser("manifest", help="write the H3 run.py --manifest file for the approved scenes")
    m.add_argument("--story", required=True)
    m.add_argument("--batch-id", help="must contain the story's slug (default <slug>-v1), so job ids never collide")
    m.add_argument("--scenes", help="only these scenes, e.g. 1,2,3 (a redo); default all")
    a = sub.add_parser("adopt-stills", help="record the images dropped into images/scene_NN/ and hand them to review")
    a.add_argument("--story")
    sub.add_parser("gpt-prompts", help="write GPT_PROMPTS.md: the reference and every shot as ChatGPT prompts")
    b = sub.add_parser("boundaries", help="write boundary_sheet.html: both sides of every cut, their blend, the light")
    b.add_argument("--story")
    b.add_argument("--source", choices=("stills", "clips"), help="default: clips once every scene has one")
    c = sub.add_parser("cut", help="join the H3 clips with overlays and the end card")
    c.add_argument("--story", required=True)
    c.add_argument("--clips", nargs="+", help="one clip per scene, in order (default: clips.json, else the newest)")
    c.add_argument("--transition", type=float, default=TRANSITION_SECONDS,
                   help=f"dissolve at every boundary, seconds (default {TRANSITION_SECONDS}; 0 = hard cuts)")
    c.add_argument("--out", help="default: the story's videos/cut/<story>_promo_vN.mp4")
    args = p.parse_args(argv)

    stories = cs.load_stories(args.root)
    for s in stories:
        cs.validate_story(s)
    if args.cmd == "check":
        for s in stories:
            problem = promo_problem(s)
            print(f"{s.id}  {planned_seconds(s):.1f} 秒  " + (f"✗ {problem}" if problem else "OK"))
            if not problem:
                print("\n".join("  " + line for line in timeline(s)))
        return
    if args.cmd == "adopt-stills":
        for story in cs.select_stories(stories, args.story):
            for item in refresh_replaced(story):
                print(f"scene {item['scene']:02d} 候選 {item['letter']}  圖被換過，記錄已更新  {item['size']}")
            for item in adopt_stills(story):
                print(f"scene {item['scene']:02d} 候選 {item['letter']}  {item['size']}  {item['path']}")
        cs.build_index(stories, args.root)
        return
    if args.cmd == "gpt-prompts":
        for story in stories:
            path = os.path.join(args.root, f"GPT_PROMPTS_{story.id}.md")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(build_gpt_prompts_md(cs.load_bible(args.root), story) + "\n")
            print(path)
        return
    if args.cmd == "boundaries":
        picked = cs.select_stories(stories, args.story)
        print(build_boundary_sheet(cs.load_bible(args.root), picked, args.root, source=args.source))
        return
    story = cs.select_stories(stories, args.story)[0]
    if args.cmd == "manifest":
        print(write_manifest(story, args.batch_id, cs._parse_scenes(args.scenes)))
    elif args.cmd == "cut":
        import concept_api

        out = args.out or concept_api.next_version(story.cut_dir(), f"{story.id}_promo")
        print(json.dumps(compose(story, clip_paths(story, args.clips), out, transition=args.transition), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
