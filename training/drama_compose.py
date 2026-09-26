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

OUT_W, OUT_H = 1080, 1920
FPS = 24
CAMERA_MOVES = ("push_in", "pull_out", "pan_left", "pan_right", "static")
ZOOM = 1.10                   # how far a camera move travels; more reads as a zoom effect, not a camera
KEN_BURNS_WORK_HEIGHT = 3840  # zoompan on an upscaled frame, or the crop window's rounding makes it jitter
SUBTITLE_BOTTOM = 1440
SUBTITLE_FONT_SIZE = 60
SUBTITLE_CHARS_PER_LINE = 14  # CJK characters; about 900 px at the font size above
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


def wrap_subtitle(text, width=SUBTITLE_CHARS_PER_LINE):
    """Hard-wrap a line for drawtext, which never wraps by itself. Breaks after punctuation when
    one falls in the second half of a line, so a phrase isn't split mid-word when it can be helped."""
    text = " ".join(text.split())
    lines = []
    while len(text) > width:
        cut = width
        for i in range(width, width // 2, -1):
            if text[i - 1] in "，。！？、；：,.!?;: ":
                cut = i
                break
        lines.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        lines.append(text)
    return "\n".join(lines)


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


def subtitle_filter(textfile, font):
    return (f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(textfile)}'"
            f":fontsize={SUBTITLE_FONT_SIZE}:fontcolor=white:borderw=5:bordercolor=black@0.85"
            f":line_spacing=12:x=(w-text_w)/2:y={SUBTITLE_BOTTOM}-text_h")


def frame_exact(seconds, fps=FPS):
    """Round a duration UP to a whole number of frames, so picture and PCM audio end together."""
    return math.ceil(seconds * fps - 1e-6) / fps


def segment_command(ffmpeg, *, out, duration, video=None, still=None, camera="push_in", voice=None,
                    voice_delay=0.25, subtitle_file=None, font=None, fps=FPS):
    """One shot as a finished segment (.mkv): picture (clip or still with a camera move), optional
    burned-in subtitle, and the voice line delayed by voice_delay and padded with silence to the
    shot length. duration should already be frame_exact."""
    if (video is None) == (still is None):
        raise ValueError("pass exactly one of video / still")
    if subtitle_file and not font:
        raise ValueError("a subtitle needs a font")
    dur = f"{duration:.3f}"
    if video:
        picture_in = ["-i", video]
        vchain = clip_filter(duration, fps)
    else:
        picture_in = ["-loop", "1", "-framerate", str(fps), "-t", dur, "-i", still]
        vchain = still_filter(camera, duration, fps)
    if subtitle_file:
        vchain += "," + subtitle_filter(subtitle_file, font)
    vchain += ",format=yuv420p"
    if voice:
        audio_in = ["-i", voice]
        delay = int(round(voice_delay * 1000))
        achain = (f"aresample=48000,aformat=channel_layouts=stereo,adelay={delay}|{delay},"
                  f"apad,atrim=0:{dur}")
    else:
        audio_in = ["-f", "lavfi", "-t", dur, "-i", "anullsrc=r=48000:cl=stereo"]
        achain = f"aresample=48000,atrim=0:{dur}"
    graph = f"[0:v]{vchain}[v];[1:a]{achain}[a]"
    return [ffmpeg, *QUIET, *picture_in, *audio_in, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            "-t", dur, "-r", str(fps), *VIDEO_CODEC, *SEGMENT_AUDIO_CODEC, out]


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
