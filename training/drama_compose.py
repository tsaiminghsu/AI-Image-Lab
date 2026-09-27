"""ffmpeg command builders for the drama pipeline (training/drama.py).

Everything here returns argument lists or filter strings and runs nothing, so the exact commands
are testable without ffmpeg installed. drama.py does the running.

Output is always 1080x1920 (9:16) at 24 fps. Segments are H.264 with 48 kHz PCM audio in .mkv and
last a whole number of frames: AAC pads every clip by a frame or so, and stream-copy concatenation
of AAC segments accumulated that padding into drifting timestamps (a 25 s cut came out at 23.85 fps
average). So the concat copies video, and AAC is encoded once, over the finished track. Subtitles
end above y=SUBTITLE_BOTTOM because the bottom ~450 px of a vertical feed (Reels, TikTok, Shorts)
is covered by captions and buttons.
"""

import math
import os
import shutil
import textwrap
import unicodedata

OUT_W, OUT_H = 1080, 1920
FPS = 24
CAMERA_MOVES = ("push_in", "pull_out", "pan_left", "pan_right", "static")
ZOOM = 1.10                   # how far a camera move travels; more reads as a zoom effect, not a camera
KEN_BURNS_WORK_HEIGHT = 3840  # zoompan on an upscaled frame, or the crop window's rounding makes it jitter
SUBTITLE_BOTTOM = 1440
SUBTITLE_FONT_SIZE = 60
SUBTITLE_CHARS_PER_LINE = 14  # CJK characters; about 900 px at the font size above
SUBTITLE_LATIN_CHARS_PER_LINE = 30
TRANSLATION_FONT_SIZE = 44
TRANSLATION_CHARS_PER_LINE = 18
TRANSLATION_COLOR = "0xFFE9A8"
LINE_GAP = 12
# Cards (phrase of the day, end card): text blocks stacked and centred in the band above the feed's
# bottom UI. The bundled ffmpeg is 4.2, whose drawtext has no text_align, so every line is its own
# drawtext centred on its own width - one multi-line drawtext would left-align lines inside the block.
CARD_BAND = (240, 1440)
CARD_BACKGROUND = "0x16202b"
CARD_BLOCKS = {   # name: (font size, colour, CJK chars per line, Latin chars per line)
    "label": (40, "0x7FD3C7", 16, 30),
    "phrase": (72, "white", 12, 24),
    "translation": (50, TRANSLATION_COLOR, 16, 30),
    "note": (38, "0xD7DEE6", 20, 38),
}
CARD_BLOCK_GAP = 34
VIDEO_CODEC = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
AUDIO_CODEC = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
SEGMENT_AUDIO_CODEC = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]
QUIET = ["-y", "-hide_banner", "-loglevel", "error"]

_WINDOWS_FONTS = (r"C:\Windows\Fonts\msjhbd.ttc", r"C:\Windows\Fonts\msjh.ttc")


def find_ffmpeg(env=None):
    """DRAMA_FFMPEG, then the ffmpeg.exe that talking_head already relies on next to SadTalker,
    then PATH. None when there is none - the caller turns that into a UsageError."""
    env = os.environ if env is None else env
    candidates = [env.get("DRAMA_FFMPEG")]
    try:
        import talking_head

        candidates.append(talking_head.SADTALKER_FFMPEG)
    except Exception:
        pass
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return shutil.which("ffmpeg")


def find_font(env=None):
    """A font with Traditional Chinese glyphs: DRAMA_FONT, else Microsoft JhengHei on Windows."""
    env = os.environ if env is None else env
    for path in (env.get("DRAMA_FONT"), *_WINDOWS_FONTS):
        if path and os.path.isfile(path):
            return path
    return None


def filter_path(path):
    """A path as an ffmpeg filter argument value: forward slashes, the drive colon escaped."""
    return path.replace("\\", "/").replace(":", r"\:")


def is_cjk_text(text):
    return any("㐀" <= ch <= "鿿" or "豈" <= ch <= "﫿" for ch in text or "")


def _units(ch):
    """Display width: 2 for full-width (CJK) characters, 1 for Latin letters, digits and spaces."""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _wrap_chars(text, width):
    """Wrap Chinese (possibly mixed with English) at `width` full-width characters, measured in
    display units so a Latin letter counts half as much as a Chinese character. Prefers a break after
    punctuation in the second half of a line."""
    limit = width * 2
    lines = []
    while sum(_units(ch) for ch in text) > limit:
        cut, used = 0, 0
        for i, ch in enumerate(text):
            used += _units(ch)
            if used > limit:
                break
            cut = i + 1
        for i in range(cut, cut // 2, -1):
            if text[i - 1] in "，。！？、；：,.!?;: ":
                cut = i
                break
        lines.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        lines.append(text)
    return lines


def wrap_lines(text, cjk_width=SUBTITLE_CHARS_PER_LINE, latin_width=SUBTITLE_LATIN_CHARS_PER_LINE):
    """Hard-wrap text into lines for drawtext, which never wraps by itself. Chinese wraps by
    character, preferring a break after punctuation in the second half of a line; English wraps
    between words."""
    text = " ".join((text or "").split())
    if not text:
        return []
    if is_cjk_text(text):
        return _wrap_chars(text, cjk_width)
    return textwrap.wrap(text, width=latin_width, break_long_words=True) or [text]


def wrap_subtitle(text, width=SUBTITLE_CHARS_PER_LINE):
    return "\n".join(wrap_lines(text, cjk_width=width))


def camera_expressions(move, frames):
    """zoompan z/x/y expressions for a camera move over `frames` output frames."""
    if move not in CAMERA_MOVES:
        raise ValueError(f"unknown camera move {move!r}; choices: {list(CAMERA_MOVES)}")
    last = max(frames - 1, 1)
    centre_x, centre_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    travel = f"{ZOOM - 1:.3f}"
    if move == "push_in":
        return f"1+{travel}*on/{last}", centre_x, centre_y
    if move == "pull_out":
        return f"{ZOOM:.3f}-{travel}*on/{last}", centre_x, centre_y
    if move == "pan_left":   # the camera turns left, so the picture slides right
        return f"{ZOOM:.3f}", f"(iw-iw/zoom)*(1-on/{last})", centre_y
    if move == "pan_right":
        return f"{ZOOM:.3f}", f"(iw-iw/zoom)*on/{last}", centre_y
    return "1", "0", "0"


def still_filter(move, duration, fps=FPS):
    """A looped still (1 input frame per output frame) turned into a camera move at output size."""
    frames = max(int(round(duration * fps)), 1)
    z, x, y = camera_expressions(move, frames)
    return (f"scale=-2:{KEN_BURNS_WORK_HEIGHT}:flags=lanczos,"
            f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={OUT_W}x{OUT_H}:fps={fps}")


def clip_filter(duration, fps=FPS):
    """A generated clip filled to 9:16 and held on its last frame if the line runs longer than it."""
    return (f"fps={fps},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={OUT_W}:{OUT_H},tpad=stop_mode=clone:stop_duration={duration:.3f},"
            f"trim=duration={duration:.3f},setpts=PTS-STARTPTS")


def _drawtext(textfile, font, size, color, y, border=5):
    return (f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(textfile)}'"
            f":fontsize={size}:fontcolor={color}:borderw={border}:bordercolor=black@0.85"
            f":x=(w-text_w)/2:y={y}")


def _block_height(n_lines, size):
    return n_lines * size + max(n_lines - 1, 0) * LINE_GAP


def _block(line_files, font, size, color, top, border=5):
    """One drawtext per line, each centred on its own width, starting at `top`."""
    return [_drawtext(f, font, size, color, top + i * (size + LINE_GAP), border) for i, f in enumerate(line_files)]


def subtitle_filters(line_files, font, translation_files=()):
    """The spoken line burned in above SUBTITLE_BOTTOM, with an optional translation under it in
    smaller, warmer text. Each argument is a list of one-line text files."""
    filters = []
    bottom = SUBTITLE_BOTTOM
    if translation_files:
        top = bottom - _block_height(len(translation_files), TRANSLATION_FONT_SIZE)
        filters += _block(translation_files, font, TRANSLATION_FONT_SIZE, TRANSLATION_COLOR, top, border=4)
        bottom = top - 18
    if line_files:
        top = bottom - _block_height(len(line_files), SUBTITLE_FONT_SIZE)
        filters += _block(line_files, font, SUBTITLE_FONT_SIZE, "white", top)
    return filters


def card_filters(blocks, font):
    """blocks: [(name, [line files])] in display order, names from CARD_BLOCKS. The stack is centred
    vertically in CARD_BAND."""
    blocks = [(name, files) for name, files in blocks if files]
    heights = [_block_height(len(files), CARD_BLOCKS[name][0]) for name, files in blocks]
    total = sum(heights) + CARD_BLOCK_GAP * max(len(blocks) - 1, 0)
    top = CARD_BAND[0] + max((CARD_BAND[1] - CARD_BAND[0] - total) // 2, 0)
    filters = []
    for (name, files), height in zip(blocks, heights):
        size, color = CARD_BLOCKS[name][:2]
        filters += _block(files, font, size, color, top, border=3)
        top += height + CARD_BLOCK_GAP
    return filters


def frame_exact(seconds, fps=FPS):
    """Round a duration UP to a whole number of frames, so picture and PCM audio end together."""
    return math.ceil(seconds * fps - 1e-6) / fps


def _audio(voice, voice_delay, duration):
    dur = f"{duration:.3f}"
    if voice:
        delay = int(round(voice_delay * 1000))
        return ["-i", voice], (f"aresample=48000,aformat=channel_layouts=stereo,adelay={delay}|{delay},"
                               f"apad,atrim=0:{dur}")
    return ["-f", "lavfi", "-t", dur, "-i", "anullsrc=r=48000:cl=stereo"], f"aresample=48000,atrim=0:{dur}"


def _segment(ffmpeg, picture_in, vchain, voice, voice_delay, duration, fps, out):
    audio_in, achain = _audio(voice, voice_delay, duration)
    dur = f"{duration:.3f}"
    graph = f"[0:v]{vchain},format=yuv420p[v];[1:a]{achain}[a]"
    return [ffmpeg, *QUIET, *picture_in, *audio_in, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            "-t", dur, "-r", str(fps), *VIDEO_CODEC, *SEGMENT_AUDIO_CODEC, out]


def segment_command(ffmpeg, *, out, duration, video=None, still=None, camera="push_in", voice=None,
                    voice_delay=0.25, subtitle_lines=(), translation_lines=(), font=None, fps=FPS):
    """One shot as a finished segment (.mkv): picture (clip or still with a camera move), optional
    burned-in subtitle and translation (lists of one-line text files), and the voice line delayed by
    voice_delay and padded with silence to the shot length. duration should already be frame_exact."""
    if (video is None) == (still is None):
        raise ValueError("pass exactly one of video / still")
    if (subtitle_lines or translation_lines) and not font:
        raise ValueError("a subtitle needs a font")
    if video:
        picture_in = ["-i", video]
        vchain = clip_filter(duration, fps)
    else:
        picture_in = ["-loop", "1", "-framerate", str(fps), "-t", f"{duration:.3f}", "-i", still]
        vchain = still_filter(camera, duration, fps)
    subs = subtitle_filters(list(subtitle_lines), font, list(translation_lines)) if font else []
    if subs:
        vchain += "," + ",".join(subs)
    return _segment(ffmpeg, picture_in, vchain, voice, voice_delay, duration, fps, out)


def card_command(ffmpeg, *, out, duration, blocks, font, background=None, voice=None, voice_delay=0.25, fps=FPS):
    """A text card (phrase of the day, end card) as a segment: text blocks over a blurred, darkened
    keyframe or a plain background, with an optional voice reading."""
    if not font:
        raise ValueError("a card needs a font")
    dur = f"{duration:.3f}"
    if background:
        picture_in = ["-loop", "1", "-framerate", str(fps), "-t", dur, "-i", background]
        vchain = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase:flags=lanczos,crop={OUT_W}:{OUT_H},"
                  f"boxblur=28:2,eq=brightness=-0.32")
    else:
        picture_in = ["-f", "lavfi", "-t", dur, "-i", f"color=c={CARD_BACKGROUND}:s={OUT_W}x{OUT_H}:r={fps}"]
        vchain = "setsar=1"
    texts = card_filters(blocks, font)
    if texts:
        vchain += "," + ",".join(texts)
    return _segment(ffmpeg, picture_in, vchain, voice, voice_delay, duration, fps, out)


def normalise_command(ffmpeg, src, dst, width=OUT_W, height=OUT_H):
    """Fill-and-centre-crop an image to width x height (one frame)."""
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={width}:{height}")
    return [ffmpeg, *QUIET, "-i", src, "-vf", vf, "-frames:v", "1", dst]


def concat_list(paths, base_dir=None):
    """Contents of a concat-demuxer list file. With base_dir (the list file's folder) each path is
    written relative to it, because the demuxer resolves entries against the list file and on
    Windows took an absolute "C:/..." entry for a relative one. Quotes inside a path are escaped."""
    lines = []
    for p in paths:
        if base_dir:
            p = os.path.relpath(p, base_dir)
        safe = p.replace("\\", "/").replace("'", r"'\''")
        lines.append(f"file '{safe}'")
    return "\n".join(lines) + "\n"


def concat_command(ffmpeg, list_file, out, final=True):
    """Join the segments, copying video. final=True encodes the audio to AAC for the mp4 that gets
    published; final=False keeps PCM for an intermediate that still gets music mixed in."""
    audio = [*AUDIO_CODEC, "-movflags", "+faststart"] if final else ["-c:a", "copy"]
    return [ffmpeg, *QUIET, "-f", "concat", "-safe", "0", "-i", list_file, "-c:v", "copy", *audio, out]


def bgm_command(ffmpeg, body, bgm, out, total_seconds, volume):
    """Lay a looped music bed under the finished cut at a fixed low volume, fading out at the end."""
    fade_start = max(total_seconds - 1.5, 0.0)
    graph = (f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={volume:.3f},"
             f"atrim=0:{total_seconds:.3f},afade=t=out:st={fade_start:.3f}:d=1.5[b];"
             f"[0:a][b]amix=inputs=2:duration=first:normalize=0[a]")
    return [ffmpeg, *QUIET, "-i", body, "-stream_loop", "-1", "-i", bgm, "-filter_complex", graph,
            "-map", "0:v", "-map", "[a]", "-c:v", "copy", *AUDIO_CODEC, "-movflags", "+faststart", out]
