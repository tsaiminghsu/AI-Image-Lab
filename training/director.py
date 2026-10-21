"""The director's desk of the GUI's picture/video tab: join local H3 clips into one 9:16 cut with ffmpeg.

Order, a transition into each clip (hard cut or a short dissolve), an optional title card and end card, optional
background music and ambience bed, every clip levelled to the same voice loudness and the whole cut finished at
-14 LUFS. It only puts together ffmpeg commands that the promo cut (activity_promo.compose) already uses and tests:
shot_command (fill to 1080x1920, PCM segment), dissolve_commands (the bundled ffmpeg 4.2 has no xfade), card_command
(drawtext on a blurred frame, Microsoft JhengHei), bed_command, loudness_command / loudness_filter / finish_command.

The one new piece is the music mix: drama_compose.bgm_command uses `amix ... normalize=0`, an option the bundled
ffmpeg 4.2.2 does not have, so music_command uses the same amix + volume=2 pattern as bed_command.
"""

import datetime
import glob
import json
import os
import re
import shutil
import subprocess

import activity_promo as ap
import concept_cut as cc
import drama_compose as dc
import generate_character as gc
import output_dir as od  # the output folder is a setting now, read when it is needed

CUT = "硬切"
TRANSITIONS = {CUT: ap.CUT_SECONDS, "疊化 0.25 秒": 0.25, "疊化 0.5 秒": 0.5}
DEFAULT_TRANSITION = CUT
END_TRANSITION = 0.25  # into the end card: the card's background is the last clip's own last frame
MUSIC_BELOW_DB = 18.0  # music sits this far under the voice reference (-23 LUFS)
MUSIC_FADE_SECONDS = 1.5
_LINE = re.compile(r"<d>\[Chinese\]\s*(.*?)</d>", re.S)


# ---------------------------------------------------------------- the clips on disk


def list_clips(directory=None):
    """Every local H3 clip, newest first: path, file name, seconds (from its record), the spoken line(s)."""
    directory = directory or od.clips_dir()
    rows = []
    for path in glob.glob(os.path.join(directory, "*.mp4")):
        meta = {}
        record = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(record):
            try:
                with open(record, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                meta = {}
        frames = meta.get("frames")
        text = meta.get("description_used") or meta.get("prompt") or ""
        rows.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "seconds": round(frames / dc.FPS, 2) if frames else None,
                "line": " / ".join(m.strip() for m in _LINE.findall(text)),
                "mtime": os.path.getmtime(path),
            }
        )
    return sorted(rows, key=lambda r: r["mtime"], reverse=True)


def transition_seconds(label):
    if label not in TRANSITIONS:
        raise gc.UsageError(f"沒有這種轉場：{label}（可用：{'、'.join(TRANSITIONS)}）")
    return TRANSITIONS[label]


# ---------------------------------------------------------------- commands


def music_command(ffmpeg, body, music, out, seconds, gain_db, fade=MUSIC_FADE_SECONDS):
    """The cut's sound with `music` under it at a fixed gain, looped to length and faded out at the end; the picture is
    copied. 4.2-safe: amix halves each of its two inputs and has no `normalize` option, so volume=2 restores the cut's
    own sound exactly (as in activity_promo.bed_command)."""
    fade_start = max(seconds - fade, 0.0)
    graph = (
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={gain_db:.2f}dB,atrim=0:{seconds:.3f},"
        f"afade=t=out:st={fade_start:.3f}:d={fade:.3f}[m];"
        f"[0:a][m]amix=inputs=2:duration=first,volume=2,aresample=48000[a]"
    )
    return [ffmpeg, *dc.QUIET, "-i", body, "-stream_loop", "-1", "-i", music, "-filter_complex", graph,
            "-map", "0:v", "-map", "[a]", "-c:v", "copy", *dc.SEGMENT_AUDIO_CODEC, "-t", f"{seconds:.3f}", out]  # fmt: skip


def music_gain_db(measured, below_db=MUSIC_BELOW_DB):
    """Fixed gain that puts the music `below_db` under the voice reference; 0 dB shift from -below_db when unmeasurable."""
    target = dc.VOICE_REFERENCE - below_db
    if measured is None or measured < dc.SILENT_BELOW:
        return -below_db
    return target - measured


def card_command(ffmpeg, out, seconds, blocks, font, background, tag=None):
    cmd = dc.card_command(ffmpeg, out=out, duration=seconds, blocks=blocks, font=font, background=background)
    if tag:
        i = cmd.index("-filter_complex") + 1
        tagged = cmd[i].replace(",format=yuv420p[v]", f",{tag},format=yuv420p[v]")
        if tagged == cmd[i]:
            raise RuntimeError("card_command's graph changed; the AI tag could not be added")
        cmd[i] = tagged
    return cmd


def _card_blocks(work, name, lines):
    """(block name, line files) for up to two lines: the first large, the second smaller."""
    blocks = []
    for key, text in zip(("phrase", "note"), lines):
        text = (text or "").strip()
        if text:
            size = dc.CARD_BLOCKS[key]
            blocks.append((key, ap._lines(work, f"{name}_{key}", text, size[2], size[3])))
    return blocks


# ---------------------------------------------------------------- media: generated clips, imported files

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")
IMAGE_SECONDS = 3.0


def media_kind(path):
    ext = os.path.splitext(str(path))[1].lower()
    return "video" if ext in VIDEO_EXT else "image" if ext in IMAGE_EXT else "audio" if ext in AUDIO_EXT else None


def as_item(clip):
    """A main-track entry: a path, or {"path", "kind"?, "seconds"?}. Images need a length (default IMAGE_SECONDS)."""
    item = {"path": clip} if isinstance(clip, str) else dict(clip)
    item.setdefault("kind", media_kind(item["path"]))
    if item["kind"] not in ("video", "image"):
        raise gc.UsageError(f"主影片軌只放影片或圖片：{os.path.basename(str(item['path']))}")
    if item["kind"] == "image":
        item["seconds"] = float(item.get("seconds") or IMAGE_SECONDS)
        if not 0.5 <= item["seconds"] <= 30:
            raise gc.UsageError("圖片片段的秒數要在 0.5～30 秒")
    return item


def _unique(directory, name):
    stem, ext = os.path.splitext(name)
    candidate, n = name, 2
    while os.path.exists(os.path.join(directory, candidate)):
        candidate, n = f"{stem}_{n}{ext}", n + 1
    return os.path.join(directory, candidate)


def import_media(paths, directory=None):
    """Copy video / image / audio files into the desk's import folder (so the GUI can show them). A folder imports the
    supported files directly inside it. Returns {"imported": [...], "skipped": [...]}."""
    directory = directory or od.imports_dir()
    files = []
    for p in paths or []:
        p = str(p).strip().strip('"')
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)))
        elif p:
            files.append(p)
    os.makedirs(directory, exist_ok=True)
    imported, skipped = [], []
    for f in files:
        if not os.path.isfile(f) or not media_kind(f):
            skipped.append(os.path.basename(f))
            continue
        dest = _unique(directory, os.path.basename(f))
        shutil.copy2(f, dest)
        imported.append(dest)
    if not imported:
        raise gc.UsageError("沒有可以匯入的檔案（支援影片 " + " ".join(VIDEO_EXT) + "、圖片 " + " ".join(IMAGE_EXT)
                            + "、音訊 " + " ".join(AUDIO_EXT) + "）")  # fmt: skip
    return {"imported": imported, "skipped": skipped}


_LENGTHS = {}  # (path, mtime) -> seconds; one ffmpeg call per file, not one per screen refresh


def video_seconds(path, ffmpeg=None, run=subprocess.run):
    """A video's length from ffmpeg's own description (None when it cannot be read). The desk's estimate used to
    count every imported clip as 5.17 s, which made a 3 s clip 2.2 s too long on screen."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _LENGTHS:
        ffmpeg = ffmpeg or dc.find_ffmpeg()
        _LENGTHS[key] = cc.duration_seconds(_probe(ffmpeg, path, run)) if ffmpeg else None
    return _LENGTHS[key]


def list_media(clips_dir=None, imports_dir=None, ffmpeg=None, run=subprocess.run):
    """Pictures for the main track: the generated H3 clips plus imported videos and images, newest first."""
    imports_dir = imports_dir or od.imports_dir()
    rows = [dict(r, kind="video", source="生成") for r in list_clips(clips_dir)]
    if os.path.isdir(imports_dir):
        for f in os.listdir(imports_dir):
            path = os.path.join(imports_dir, f)
            kind = media_kind(path)
            if kind in ("video", "image"):
                rows.append({"path": path, "name": f, "seconds": None, "line": "", "kind": kind, "source": "匯入",
                             "mtime": os.path.getmtime(path)})  # fmt: skip
    for r in rows:
        if r["kind"] == "video" and not r["seconds"]:
            r["seconds"] = video_seconds(r["path"], ffmpeg, run)
    return sorted(rows, key=lambda r: r["mtime"], reverse=True)


def list_audio(imports_dir=None):
    imports_dir = imports_dir or od.imports_dir()
    if not os.path.isdir(imports_dir):
        return []
    return sorted((os.path.join(imports_dir, f) for f in os.listdir(imports_dir) if media_kind(f) == "audio"),
                  key=os.path.getmtime, reverse=True)  # fmt: skip


# ---------------------------------------------------------------- the extra tracks: text, picture overlays, sound effects

TEXT_POSITIONS = {"上方": 230, "中間": 860, "下方": 1300}  # top of the block; kept above y=1440 (feed UI)
TEXT_SIZES = {"小": (48, 18), "中": (64, 14), "大": (88, 10)}  # font size, CJK characters per line
OVERLAY_POSITIONS = {
    "左上": ("40", "200"),
    "右上": ("W-w-40", "200"),
    "中間": ("(W-w)/2", "(H-h)/2"),
    "左下": ("40", "H-h-520"),
    "右下": ("W-w-40", "H-h-520"),
}


def _span(entry, what):
    try:
        start, end = float(entry.get("start", 0)), float(entry.get("end", 0))
    except (TypeError, ValueError):
        raise gc.UsageError(f"{what}的開始／結束秒數要是數字") from None
    if start < 0 or end <= start:
        raise gc.UsageError(f"{what}的結束秒數要大於開始秒數（{start:g}～{end:g}）")
    return start, end


def check_tracks(texts=(), overlays=(), sfx=()):
    """Validate and normalise the three extra tracks; raises gc.UsageError naming the bad entry."""
    out_t, out_o, out_s = [], [], []
    for n, t in enumerate(texts or [], 1):
        text = str(t.get("text", "")).strip()
        if not text:
            raise gc.UsageError(f"文字軌第 {n} 段是空的")
        start, end = _span(t, f"文字軌第 {n} 段")
        position, size = t.get("position", "上方"), t.get("size", "中")
        if position not in TEXT_POSITIONS or size not in TEXT_SIZES:
            raise gc.UsageError(f"文字軌第 {n} 段的位置或大小不對")
        out_t.append({"text": text, "start": start, "end": end, "position": position, "size": size})
    for n, o in enumerate(overlays or [], 1):
        path = str(o.get("path", ""))
        if media_kind(path) != "image" or not os.path.isfile(path):
            raise gc.UsageError(f"疊圖軌第 {n} 張要是存在的圖片檔：{os.path.basename(path)}")
        start, end = _span(o, f"疊圖軌第 {n} 張")
        position = o.get("position", "右上")
        scale = float(o.get("scale", 25))
        if position not in OVERLAY_POSITIONS or not 5 <= scale <= 100:
            raise gc.UsageError(f"疊圖軌第 {n} 張的位置或大小（5～100%）不對")
        out_o.append({"path": path, "start": start, "end": end, "position": position, "scale": scale})
    for n, a in enumerate(sfx or [], 1):
        path = str(a.get("path", ""))
        if media_kind(path) != "audio" or not os.path.isfile(path):
            raise gc.UsageError(f"音效軌第 {n} 段要是存在的音訊檔：{os.path.basename(path)}")
        start = float(a.get("start", 0))
        volume = float(a.get("volume_db", 0))
        if start < 0 or not -40 <= volume <= 12:
            raise gc.UsageError(f"音效軌第 {n} 段的開始秒數不能是負的、音量要在 -40～+12 dB")
        out_s.append({"path": path, "start": start, "volume_db": volume})
    return out_t, out_o, out_s


def layers_command(ffmpeg, body, out, texts, overlays, work, font):
    """The joined cut with the picture overlays and the text track drawn on top, each only between its start and end;
    the sound is copied. One pass, picture only (this ffmpeg build segfaults when overlay and audio filters share a
    graph, see activity_promo.dissolve_commands)."""
    inputs, graph, cur = ["-i", body], [], "0:v"
    for i, o in enumerate(overlays):
        inputs += ["-loop", "1", "-i", o["path"]]
        x, y = OVERLAY_POSITIONS[o["position"]]
        width = int(dc.OUT_W * o["scale"] / 100) // 2 * 2
        graph.append(f"[{i + 1}:v]scale={width}:-2,format=rgba[ov{i}]")
        graph.append(
            f"[{cur}][ov{i}]overlay=x={x}:y={y}:shortest=1:enable='between(t,{o['start']:.3f},{o['end']:.3f})'[v{i}]"
        )
        cur = f"v{i}"
    drawn = []
    for i, t in enumerate(texts):
        size, per_line = TEXT_SIZES[t["size"]]
        lines = ap._lines(work, f"text_{i}", t["text"], per_line, per_line * 2)
        block = dc._block(lines, font, size, "white", TEXT_POSITIONS[t["position"]])
        drawn += ap._when(block, f"between(t,{t['start']:.3f},{t['end']:.3f})")
    chain = ",".join([*drawn, "format=yuv420p"])
    graph.append(f"[{cur}]{chain}[v]")
    return [ffmpeg, *dc.QUIET, *inputs, "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "0:a",
            "-r", str(dc.FPS), *dc.VIDEO_CODEC, "-c:a", "copy", out]  # fmt: skip


def sfx_command(ffmpeg, body, out, sfx, seconds):
    """The cut's sound with each sound effect delayed to its start and set to its volume; the picture is copied.
    amix scales every input by 1/n and this ffmpeg has no `normalize`, so volume=n restores the cut's own sound."""
    inputs, graph, labels = ["-i", body], [], ["[0:a]"]
    for i, a in enumerate(sfx):
        inputs += ["-i", a["path"]]
        ms = int(round(a["start"] * 1000))
        graph.append(f"[{i + 1}:a]aresample=48000,aformat=channel_layouts=stereo,volume={a['volume_db']:.2f}dB,"
                     f"adelay={ms}|{ms}[s{i}]")  # fmt: skip
        labels.append(f"[s{i}]")
    n = len(labels)
    graph.append(f"{''.join(labels)}amix=inputs={n}:duration=first,volume={n},aresample=48000[a]")
    return [ffmpeg, *dc.QUIET, *inputs, "-filter_complex", ";".join(graph), "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", *dc.SEGMENT_AUDIO_CODEC, "-t", f"{seconds:.3f}", out]  # fmt: skip


def _probe(ffmpeg, path, run=subprocess.run):
    """ffmpeg's own description of the file (there is no ffprobe next to the bundled ffmpeg)."""
    done = run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return done.stderr or ""


def _tagged(cmd, tag):
    if not tag:
        return cmd
    i = cmd.index("-filter_complex") + 1
    cmd[i] = cmd[i].replace(",format=yuv420p[v]", f",{tag},format=yuv420p[v]", 1)
    return cmd


# ---------------------------------------------------------------- templates

TEMPLATES = {
    "社群短影音（快節奏、硬切）": {
        "description": "15～25 秒，全部硬切，開頭 2 秒大字標題，結尾導流字卡",
        "transition": "硬切", "title": ("", ""), "title_seconds": 0, "ending": ("追蹤看更多", "每天一個約會情境"),
        "ending_seconds": 2, "music_below_db": 14, "bed": False, "ai_label": False,
        "texts": [{"text": "今天跟她去哪？", "start": 0, "end": 2, "position": "上方", "size": "大"}],
    },
    "約會合輯（溫柔、疊化）": {
        "description": "片頭片尾字卡、0.5 秒疊化、配樂壓低、加環境底噪",
        "transition": "疊化 0.5 秒", "title": ("一起去約會", "三個地方，三種心情"), "title_seconds": 2,
        "ending": ("下載 App", "跟她一起出門"), "ending_seconds": 2.5, "music_below_db": 18, "bed": True,
        "ai_label": False, "texts": [],
    },
    "App 宣傳（片頭片尾＋AI 標示）": {
        "description": "短片頭、較長的下載字卡、下方標語、每格 AI 標示",
        "transition": "疊化 0.25 秒", "title": ("你的 AI 約會對象", "16 種約會情境"), "title_seconds": 1.5,
        "ending": ("立即下載", "限時免費體驗"), "ending_seconds": 3, "music_below_db": 16, "bed": True,
        "ai_label": True, "texts": [{"text": "AI 陪你約會", "start": 1.5, "end": 4.5, "position": "下方", "size": "中"}],
    },
    "Vlog 日常（底噪、日期字條）": {
        "description": "沒有片頭、短疊化、左上日期字條、配樂很小聲",
        "transition": "疊化 0.25 秒", "title": ("", ""), "title_seconds": 0, "ending": ("下次見", ""),
        "ending_seconds": 2, "music_below_db": 22, "bed": True, "ai_label": False,
        "texts": [{"text": "Day 1", "start": 0, "end": 2.5, "position": "上方", "size": "中"}],
    },
}  # fmt: skip


# ---------------------------------------------------------------- multi-track view (HTML for the GUI)

TRACK_COLORS = {"V2": "#3b82f6", "V1": "#e0663b", "T": "#eab308", "A1": "#22c55e", "A2": "#f97316", "A3": "#a855f7",
                "card": "#6366f1"}  # fmt: skip


def layout(timeline, clip_seconds, title_seconds=0.0, ending_seconds=0.0, has_title=False, has_ending=False):
    """Where each main-track segment starts and ends in the finished cut, as render() places them."""
    segs = []
    if has_title and title_seconds > 0:
        segs.append({"kind": "card", "label": "片頭", "seconds": dc.frame_exact(title_seconds), "into": 0.0})
    for n, item in enumerate(timeline or [], 1):
        secs = item.get("seconds") if item.get("kind") == "image" else clip_seconds.get(item.get("name"))
        segs.append({"kind": item.get("kind", "video"), "label": f"{n}", "name": item.get("name"), "item": item,
                     "seconds": dc.frame_exact(secs or 5.17), "into": transition_seconds(item["transition"])})  # fmt: skip
    if has_ending and ending_seconds > 0 and segs:
        segs.append(
            {"kind": "card", "label": "片尾", "seconds": dc.frame_exact(ending_seconds), "into": END_TRANSITION}
        )
    overlaps = [s["into"] for s in segs[1:]]
    hard = bool(overlaps) and all(t == ap.CUT_SECONDS for t in overlaps)
    t = 0.0
    for i, seg in enumerate(segs):
        start = 0.0 if i == 0 else t - (0.0 if hard else seg["into"])
        seg["start"], seg["end"] = round(start, 3), round(start + seg["seconds"], 3)
        t = seg["end"]
    return segs, round(t, 2)


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def timeline_html(segs, total, tracks=None, music_name=None, selected=None, file_url=None):
    """A dark editor-style multi-track view: time ruler, V2 overlays, V1 main pictures (thumbnails), T text, A1 clip
    sound, A2 sound effects, A3 music. Pure: file_url(path) turns a thumbnail path into a URL the page can load."""
    tracks = tracks or {}
    total = max(total, 1.0)
    pps = max(18.0, min(60.0, 1100.0 / total))  # pixels per second
    width = int(total * pps) + 20

    def block(start, end, color, label, extra=""):
        left, w = start * pps, max((end - start) * pps - 2, 6)
        return (f'<div title="{_esc(label)}" style="position:absolute;left:{left:.0f}px;width:{w:.0f}px;top:3px;bottom:3px;'
                f"background:{color};border-radius:6px;color:#fff;font-size:11px;padding:2px 5px;overflow:hidden;"
                f'white-space:nowrap;box-shadow:inset 0 0 0 1px rgba(255,255,255,.18);{extra}">{_esc(label)}</div>')  # fmt: skip

    def row(name, label, content, height=34):
        return (f'<div style="display:flex;align-items:stretch;border-bottom:1px solid #222a35">'
                f'<div style="flex:0 0 74px;color:#9aa4b2;font-size:11px;padding:6px 8px;background:#11161d;'
                f'border-right:1px solid #222a35"><b style="color:#e6edf3">{name}</b><br>{label}</div>'
                f'<div style="position:relative;height:{height}px;width:{width}px">{content}</div></div>')  # fmt: skip

    ticks = []
    step = 1 if total <= 30 else 2 if total <= 60 else 5
    s = 0
    while s <= total + 0.001:
        major = s % (5 if step < 5 else 10) == 0
        ticks.append(f'<div style="position:absolute;left:{s * pps:.0f}px;top:{0 if major else 10}px;bottom:0;'
                     f'border-left:1px solid {"#5b6573" if major else "#2d3642"}"></div>')  # fmt: skip
        if major:
            ticks.append(f'<div style="position:absolute;left:{s * pps + 3:.0f}px;top:0;font-size:10px;color:#9aa4b2">'
                         f"{int(s // 60)}:{int(s % 60):02d}</div>")  # fmt: skip
        s += step
    v1, a1 = [], []
    for seg in segs:
        if seg["kind"] == "card":
            v1.append(block(seg["start"], seg["end"], TRACK_COLORS["card"], seg["label"]))
            continue
        item = seg.get("item") or {}
        thumb = item.get("thumb")
        bg = (f"background-image:linear-gradient(rgba(0,0,0,.25),rgba(0,0,0,.55)),url('{file_url(thumb)}');"
              f"background-size:auto 100%;background-repeat:repeat-x") if thumb and file_url else ""  # fmt: skip
        sel = "outline:2px solid #fbbf24;" if selected is not None and seg["label"] == str(selected + 1) else ""
        name = item.get("name", "")
        v1.append(block(seg["start"], seg["end"], TRACK_COLORS["V1"], f"{seg['label']}  {name}", bg + sel))
        if seg["kind"] == "video":
            a1.append(block(seg["start"], seg["end"], TRACK_COLORS["A1"], "原聲"))
    v2 = [
        block(o["start"], o["end"], TRACK_COLORS["V2"], os.path.basename(o["path"])) for o in tracks.get("overlays", [])
    ]
    tt = [block(t["start"], t["end"], TRACK_COLORS["T"], t["text"]) for t in tracks.get("texts", [])]
    a2 = [block(a["start"], min(total, a["start"] + float(a.get("seconds") or 2.0)), TRACK_COLORS["A2"],
                os.path.basename(a["path"])) for a in tracks.get("sfx", [])]  # fmt: skip
    a3 = [block(0, total, TRACK_COLORS["A3"], f"♪ {music_name}")] if music_name else []
    body = "".join([
        row("", "", "".join(ticks), 20),
        row("V2", "疊圖", "".join(v2)),
        row("V1", "主影片", "".join(v1), 52),
        row("T", "文字", "".join(tt)),
        row("A1", "原聲", "".join(a1)),
        row("A2", "音效", "".join(a2)),
        row("A3", "配樂", "".join(a3)),
    ])  # fmt: skip
    return (f'<div style="background:#0d1117;border:1px solid #222a35;border-radius:12px;overflow-x:auto;'
            f'font-family:system-ui,sans-serif">{body}</div>'
            f'<div style="color:#9aa4b2;font-size:12px;margin-top:6px">總長 {total:.1f} 秒｜每秒 {pps:.0f} px</div>')  # fmt: skip


# ---------------------------------------------------------------- render


def render(
    clips,
    out=None,
    *,
    transitions=None,
    title=("", ""),
    title_seconds=0.0,
    ending=("", ""),
    ending_seconds=0.0,
    music=None,
    music_below_db=MUSIC_BELOW_DB,
    music_fade=MUSIC_FADE_SECONDS,
    bed=False,
    ai_label=False,
    texts=(),
    overlays=(),
    sfx=(),
    ffmpeg=None,
    font=None,
    run=subprocess.run,
    log=print,
):
    """Join `clips` (paths, in order) into one 1080x1920 mp4. `transitions`: one label of TRANSITIONS per clip, the
    transition INTO that clip (the first one only matters when there is a title card). title / ending: two lines of
    card text each; a card is drawn when its seconds > 0 and it has text. music: an audio file or None. Returns the
    numbers worth reporting; also writes them next to the cut as .json."""
    if not clips:
        raise gc.UsageError("時間軸是空的：至少勾一支片段")
    items = [as_item(c) for c in clips]
    clips = [it["path"] for it in items]
    for p in clips:
        if not os.path.isfile(p):
            raise gc.UsageError(f"找不到片段 {p}")
    texts, overlays, sfx = check_tracks(texts, overlays, sfx)
    labels = list(transitions or [DEFAULT_TRANSITION] * len(clips))
    if len(labels) != len(clips):
        raise gc.UsageError("每支片段要有一個轉場")
    into = [transition_seconds(t) for t in labels]
    if music and not os.path.isfile(music):
        raise gc.UsageError(f"找不到配樂檔 {music}")
    ffmpeg = ffmpeg or dc.find_ffmpeg()
    if not ffmpeg:
        raise gc.UsageError("找不到 ffmpeg（設定 DRAMA_FFMPEG 或裝進 PATH）")
    want_cards = (title_seconds > 0 and any((t or "").strip() for t in title)) or (
        ending_seconds > 0 and any((t or "").strip() for t in ending)
    )
    font = font or dc.find_font()
    if (want_cards or ai_label or texts) and not font:
        raise gc.UsageError("找不到中文字型（設定 DRAMA_FONT）")

    def do(cmd, what):
        log(f"▶ {what}")
        done = run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            log(f"✖ {what} 失敗：{(done.stderr or '').strip()[-400:]}")
            raise gc.UsageError(f"ffmpeg 在「{what}」失敗：{(done.stderr or '').strip()[-300:]}")
        return done.stderr or ""

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.abspath(out or os.path.join(od.cuts_dir(), f"cut_{stamp}.mp4"))
    work = os.path.splitext(out)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    tag = ap.tag_filter(ap._lines(work, "ai_tag", ap.AI_TAG, 30, 60)[0], font) if ai_label else None
    log(f"開始剪接：{len(clips)} 支片段 → {out}")

    segments, seconds, overlaps, measured = [], [], [], []
    title_blocks = _card_blocks(work, "title", title) if title_seconds > 0 else []
    if title_blocks:
        frame = os.path.join(work, "first_frame.png")
        do([ffmpeg, *dc.QUIET, "-i", clips[0], "-frames:v", "1", frame], "取第一支片段的首幀當片頭背景")
        frame = frame if items[0]["kind"] == "video" else clips[0]
        card = os.path.join(work, "title_card.mkv")
        length = dc.frame_exact(title_seconds)
        do(card_command(ffmpeg, card, length, title_blocks, font, frame, tag), f"片頭字卡（{length:.2f} 秒）")
        segments.append(card)
        seconds.append(length)

    for i, (item, clip) in enumerate(zip(items, clips)):
        segment = os.path.join(work, f"clip_{i + 1:02d}.mkv")
        if item["kind"] == "image":
            duration = dc.frame_exact(item["seconds"])
            loudness = None
            log(f"  圖片 {os.path.basename(clip)}：{duration:.2f} 秒，慢慢推近")
            cmd = dc.segment_command(ffmpeg, out=segment, duration=duration, still=clip, camera="push_in")
            do(_tagged(cmd, tag), f"片段 {i + 1}（圖片）做成 {duration:.2f} 秒的畫面")
        else:
            probe = _probe(ffmpeg, clip, run)
            length = cc.duration_seconds(probe)
            if length is None:
                raise gc.UsageError(f"讀不到片段長度：{clip}（ffmpeg：{probe.strip()[-200:]}）")
            duration = dc.frame_exact(length)
            if "Audio:" in probe:
                info = do(dc.loudness_command(ffmpeg, clip), f"量測片段 {i + 1}：{os.path.basename(clip)}")
                loudness = dc.integrated_loudness(info)
                gain = cc.clip_gain_db(loudness)
                log(
                    f"  長度 {duration:.2f} 秒、響度 {loudness} LUFS → 增益 {gain:+.1f} dB（對齊 {cc.CLIP_REFERENCE} LUFS）"
                )
                do(ap.shot_command(ffmpeg, clip, segment, duration, [tag] if tag else [], gain),
                   f"片段 {i + 1} 放大到 1080×1920")  # fmt: skip
            else:
                loudness = None
                log(f"  長度 {duration:.2f} 秒、沒有聲音軌（補上靜音）")
                cmd = dc.segment_command(ffmpeg, out=segment, duration=duration, video=clip)
                do(_tagged(cmd, tag), f"片段 {i + 1} 放大到 1080×1920（無聲）")
        if segments:
            overlaps.append(into[i])
        segments.append(segment)
        seconds.append(duration)
        measured.append(loudness)

    ending_blocks = _card_blocks(work, "ending", ending) if ending_seconds > 0 else []
    if ending_blocks:
        frame = os.path.join(work, "last_frame.png")
        do(
            [
                ffmpeg,
                *dc.QUIET,
                *(["-sseof", "-0.2"] if items[-1]["kind"] == "video" else []),
                "-i",
                clips[-1],
                "-frames:v",
                "1",
                frame,
            ],
            "取最後一支片段的末幀當片尾背景",
        )
        card = os.path.join(work, "end_card.mkv")
        length = dc.frame_exact(ending_seconds)
        do(card_command(ffmpeg, card, length, ending_blocks, font, frame, tag), f"片尾字卡（{length:.2f} 秒）")
        overlaps.append(END_TRANSITION)
        segments.append(card)
        seconds.append(length)

    body = os.path.join(work, "body.mkv")
    if len(segments) == 1:
        body = segments[0]
    elif all(t == ap.CUT_SECONDS for t in overlaps):
        with open(os.path.join(work, "concat.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(dc.concat_list(segments, base_dir=work))
        log("▶ 全部硬切：直接接起來")
        done = run(dc.concat_command(ffmpeg, "concat.txt", body, final=False), capture_output=True, text=True,
                   encoding="utf-8", errors="replace", cwd=work)  # fmt: skip
        if done.returncode != 0:
            raise gc.UsageError(f"ffmpeg 接片失敗：{(done.stderr or '').strip()[-300:]}")
        overlaps = [0.0] * len(overlaps)
    else:
        try:
            commands = ap.dissolve_commands(ffmpeg, segments, seconds, body, overlaps)
        except ValueError as exc:
            raise gc.UsageError(f"轉場設定不行：{exc}（疊化最長 0.5 秒，片段至少要是轉場的兩倍長）") from None
        for n, cmd in enumerate(commands, 1):
            do(cmd, ("接畫面（含疊化）", "接聲音（含交叉淡化）", "合併畫面與聲音")[n - 1])
    total = round(sum(seconds) - sum(overlaps), 3)
    log(f"  接好：約 {total:.2f} 秒")

    if texts or overlays:
        layered = os.path.join(work, "body_layers.mkv")
        do(layers_command(ffmpeg, body, layered, texts, overlays, work, font),
           f"疊上文字軌 {len(texts)} 段、疊圖軌 {len(overlays)} 張")  # fmt: skip
        body = layered
    if sfx:
        mixed = os.path.join(work, "body_sfx.mkv")
        do(sfx_command(ffmpeg, body, mixed, sfx, total), f"混入音效軌 {len(sfx)} 段")
        body = mixed

    if bed:
        under = os.path.join(work, "body_bed.mkv")
        do(ap.bed_command(ffmpeg, body, under, total), f"加環境底噪（{ap.BED_LUFS} LUFS）")
        body = under

    music_info = None
    if music:
        music_loudness = dc.integrated_loudness(do(dc.loudness_command(ffmpeg, music), "量測配樂響度"))
        gain = music_gain_db(music_loudness, music_below_db)
        mixed = os.path.join(work, "body_music.mkv")
        do(music_command(ffmpeg, body, music, mixed, total, gain, music_fade),
           f"配樂：比人聲低 {music_below_db:.0f} dB（增益 {gain:+.1f} dB），結尾淡出 {music_fade:.1f} 秒")  # fmt: skip
        body = mixed
        music_info = {"file": os.path.abspath(music), "measured_lufs": music_loudness, "gain_db": round(gain, 2),
                      "below_voice_db": music_below_db, "fade_seconds": music_fade}  # fmt: skip

    mix = dc.integrated_loudness(do(dc.loudness_command(ffmpeg, body), "量測混音響度"))
    do(dc.finish_command(ffmpeg, body, out, dc.loudness_filter(mix)), f"輸出成片（整支調到 {dc.LOUDNESS_TARGET} LUFS）")
    final = dc.integrated_loudness(do(dc.loudness_command(ffmpeg, out), "量測成片響度"))
    report = {
        "output": out,
        "clips": [os.path.abspath(c) for c in clips],
        "transitions": labels,
        "overlaps_seconds": [round(t, 3) for t in overlaps],
        "title": list(title) if title_blocks else None,
        "ending": list(ending) if ending_blocks else None,
        "seconds": total,
        "size": f"{dc.OUT_W}x{dc.OUT_H}",
        "clip_loudness_lufs": measured,
        "mix_loudness_lufs": mix,
        "final_loudness_lufs": final,
        "music": music_info,
        "ambience_bed_lufs": ap.BED_LUFS if bed else None,
        "ai_tag": ap.AI_TAG if tag else None,
        "tracks": {"texts": list(texts), "overlays": list(overlays), "sfx": list(sfx)},
        "created_at": stamp,
    }
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"✔ 完成：{out}（{total:.2f} 秒，成片響度 {final} LUFS）")
    return report


# ---------------------------------------------------------------- the desk's timeline (pure, for the GUI)
#
# The timeline is a list of {"name": clip file name, "transition": label of TRANSITIONS} in play order. Every edit
# returns a new list and the index to select afterwards; nothing here touches disk.


def timeline_add(timeline, name, transition=DEFAULT_TRANSITION, **extra):
    """extra: path, kind ("video" / "image"), seconds (images), thumb - whatever the desk wants to carry along."""
    items = [dict(t) for t in timeline or []]
    items.append({"name": name, "transition": transition, **extra})
    return items, len(items) - 1


def timeline_set_seconds(timeline, index, seconds):
    """An image clip's length on the main track (videos keep their own length)."""
    items = [dict(t) for t in timeline or []]
    if index is not None and 0 <= index < len(items) and items[index].get("kind") == "image":
        seconds = float(seconds)
        if not 0.5 <= seconds <= 30:
            raise gc.UsageError("圖片片段的秒數要在 0.5～30 秒")
        items[index]["seconds"] = seconds
    return items


def timeline_move(timeline, index, step):
    items = [dict(t) for t in timeline or []]
    if index is None or not 0 <= index < len(items):
        return items, index
    target = max(0, min(len(items) - 1, index + step))
    items.insert(target, items.pop(index))
    return items, target


def timeline_remove(timeline, index):
    items = [dict(t) for t in timeline or []]
    if index is None or not 0 <= index < len(items):
        return items, None
    items.pop(index)
    return items, (min(index, len(items) - 1) if items else None)


def timeline_set_transition(timeline, index, label):
    transition_seconds(label)  # refuses an unknown label
    items = [dict(t) for t in timeline or []]
    if index is not None and 0 <= index < len(items):
        items[index]["transition"] = label
    return items


def timeline_seconds(timeline, clip_seconds, title_seconds=0.0, ending_seconds=0.0, has_title=False, has_ending=False):
    """The cut's length as render() will make it: segments minus the overlap at every boundary. clip_seconds maps a
    clip name to its length (None/missing counts as 5.17 s, a 5 s H3 clip)."""
    parts, overlaps = [], []
    if has_title and title_seconds > 0:
        parts.append(dc.frame_exact(title_seconds))
    for item in timeline or []:
        if parts:
            overlaps.append(transition_seconds(item["transition"]))
        parts.append(dc.frame_exact(clip_seconds.get(item["name"]) or 5.17))
    if has_ending and ending_seconds > 0 and parts:
        overlaps.append(END_TRANSITION)
        parts.append(dc.frame_exact(ending_seconds))
    if overlaps and all(t == ap.CUT_SECONDS for t in overlaps):
        overlaps = []  # all hard cuts are joined end to end
    return round(sum(parts) - sum(overlaps), 2)



def thumbnail(path, directory=None, ffmpeg=None, run=subprocess.run):
    """A small jpg of the clip at 1 s (or its first frame), cached by file name and modification time."""
    directory = directory or od.thumbs_dir()
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(directory, f"{stem}_{int(os.path.getmtime(path))}.jpg")
    if os.path.isfile(out):
        return out
    ffmpeg = ffmpeg or dc.find_ffmpeg()
    if not ffmpeg:
        return None
    for seek in ("1", "0"):
        done = run([ffmpeg, *dc.QUIET, "-ss", seek, "-i", path, "-frames:v", "1", "-vf", "scale=270:-2", "-q:v", "4", out],
                   capture_output=True, text=True)  # fmt: skip
        if done.returncode == 0 and os.path.isfile(out):
            return out
    return None
