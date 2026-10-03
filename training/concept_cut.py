"""Join a story's H3 clips into one cut: hard cuts, levelled audio, optional end subtitle.

This is the recipe worked out by hand for 23:47 (projects/ai_short_drama/story_01_2347/rough_cut.md), made
repeatable. Loudness follows drama_compose: measure, then fixed gains - never single-pass loudnorm, which pumps.
Each clip is brought to CLIP_REFERENCE (an ambience-only clip AMBIENCE_OFFSET lower, so its room tone is not
amplified into noise), the joined audio is measured, and one gain plus a limiter brings the mix to
drama_compose.LOUDNESS_TARGET.

The command builders are pure; `assemble` runs them. H3 clips of one story share size, frame rate and audio
format, so nothing is rescaled.
"""

import os
import re
import subprocess

import drama_compose as dc
import generate_character as gc

CLIP_REFERENCE = dc.VOICE_REFERENCE     # -23 LUFS
AMBIENCE_OFFSET = -8.0
SUBTITLE_FONT_SIZE = 44                 # drama_compose's 60 px at 1080 wide, scaled to H3's 768
SUBTITLE_SECONDS = 1.5
VIDEO_CODEC = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
AUDIO_CODEC = ["-c:a", "aac", "-b:a", "192k"]


def clip_gain_db(measured, ambience=False):
    """Fixed gain that brings one clip to its reference, or 0.0 for a clip that cannot be measured or is silent
    (amplifying silence only raises noise)."""
    if measured is None or measured < dc.SILENT_BELOW:
        return 0.0
    reference = CLIP_REFERENCE + (AMBIENCE_OFFSET if ambience else 0.0)
    return round(reference - measured, 2)


def _inputs(paths):
    args = []
    for p in paths:
        args += ["-i", p]
    return args


def _gain_chain(gains):
    return [f"[{i}:a]volume={g:.2f}dB[a{i}]" for i, g in enumerate(gains)]


def mix_measure_command(ffmpeg, paths, gains):
    """Measure the integrated loudness of the levelled, joined audio (summary on stderr)."""
    n = len(paths)
    graph = _gain_chain(gains) + ["".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1,ebur128=peak=sample[am]"]
    return [ffmpeg, "-hide_banner", "-nostats", *_inputs(paths), "-filter_complex", ";".join(graph),
            "-map", "[am]", "-f", "null", "-"]


def final_command(ffmpeg, paths, gains, mix_loudness, out, *, fps=24, subtitle_file=None, subtitle_start=None,
                  font=None, frame_height=1376):
    """The cut itself. mix_loudness None (silent or unmeasurable) leaves the levelled mix as it is."""
    n = len(paths)
    graph = _gain_chain(gains)
    graph.append("".join(f"[{i}:v][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v0][a0mix]")
    video = "[v0]"
    if subtitle_file:
        y = int(frame_height * 0.75) - SUBTITLE_FONT_SIZE
        draw = dc._drawtext(subtitle_file, font, SUBTITLE_FONT_SIZE, "white", y, border=4)
        graph.append(f"[v0]{draw}:enable='gte(t,{subtitle_start:.3f})'[v]")
        video = "[v]"
    final = dc.loudness_filter(mix_loudness)
    audio = "[a0mix]"
    if final:
        graph.append(f"[a0mix]{final}[a]")
        audio = "[a]"
    return [ffmpeg, *dc.QUIET, *_inputs(paths), "-filter_complex", ";".join(graph), "-map", video, "-map", audio,
            *VIDEO_CODEC, "-r", str(fps), *AUDIO_CODEC, "-movflags", "+faststart", out]


def duration_seconds(log):
    """Seconds from the 'Duration: HH:MM:SS.xx' line ffmpeg prints for an input, or None."""
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", log)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def assemble(clips, out, *, subtitle=None, ffmpeg=None, font=None, run=subprocess.run):
    """clips: [{"path", "ambience": bool}] in story order. Returns the numbers worth reporting. Raises
    gc.UsageError for a missing clip, ffmpeg or font; an ffmpeg failure surfaces as CalledProcessError."""
    ffmpeg = ffmpeg or dc.find_ffmpeg()
    if not ffmpeg:
        raise gc.UsageError("找不到 ffmpeg（設定 DRAMA_FFMPEG 或裝進 PATH）")
    if not clips:
        raise gc.UsageError("沒有任何片段可以接")
    for c in clips:
        if not os.path.isfile(c["path"]):
            raise gc.UsageError(f"找不到片段 {c['path']}")
    paths = [c["path"] for c in clips]

    def text(cmd):
        return run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    measured, durations = [], []
    for p in paths:
        log = text(dc.loudness_command(ffmpeg, p)).stderr
        measured.append(dc.integrated_loudness(log))
        durations.append(duration_seconds(log))
    gains = [clip_gain_db(m, bool(c.get("ambience"))) for m, c in zip(measured, clips)]
    mix = dc.integrated_loudness(text(mix_measure_command(ffmpeg, paths, gains)).stderr)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subtitle_file = subtitle_start = None
    if subtitle:
        font = font or dc.find_font()
        if not font:
            raise gc.UsageError("找不到中文字型（設定 DRAMA_FONT）")
        if any(d is None for d in durations):
            raise gc.UsageError("讀不到片段長度，無法決定字幕出現的時間")
        subtitle_file = os.path.splitext(out)[0] + ".subtitle.txt"
        with open(subtitle_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(subtitle)
        subtitle_start = max(0.0, sum(durations) - SUBTITLE_SECONDS)
    cmd = final_command(ffmpeg, paths, gains, mix, out, subtitle_file=subtitle_file, subtitle_start=subtitle_start,
                        font=font)
    done = text(cmd)
    if done.returncode != 0:
        raise subprocess.CalledProcessError(done.returncode, cmd, stderr=done.stderr)
    final = dc.integrated_loudness(text(dc.loudness_command(ffmpeg, out)).stderr)
    total = sum(d for d in durations if d is not None)
    return {"output": out, "clips": len(paths), "seconds": round(total, 2),
            "clip_loudness_lufs": measured, "clip_gain_db": gains, "mix_loudness_lufs": mix,
            "final_loudness_lufs": final, "subtitle": subtitle or None,
            "subtitle_start": round(subtitle_start, 2) if subtitle_start is not None else None}
