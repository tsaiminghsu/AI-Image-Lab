"""Talking-head (lip-sync) video from one still portrait + a voice clip, via SadTalker.

SadTalker lives in its own vendored clone (D:\\AI-Image-Lab\\SadTalker, gitignored) with its own
Python 3.10 / torch 2.5.1+cu121 venv - a completely different dependency stack from ComfyUI's
venv, so this module never imports it: it only shells out to that venv's python running
SadTalker's own inference.py, same "don't import torch here" rule as comfyui_client.py. ComfyUI
does not need to be running for this route.

Content rule (same as every other route in this repo): the source image must be a fictional /
AI-generated face (e.g. an anchor produced by generate_character.py), never a real person's photo.
"""

import os
import re
import shutil
import subprocess
import uuid
from collections import deque

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SADTALKER_DIR = os.environ.get("SADTALKER_DIR", os.path.join(REPO_ROOT, "SadTalker"))
SADTALKER_PYTHON = os.path.join(SADTALKER_DIR, ".venv", "Scripts", "python.exe")
# Read by the local patch to SadTalker/src/utils/videoio.py; PATH is also prepended with the
# SadTalker dir below, so an unpatched clone still finds the ffmpeg.exe dropped next to it.
SADTALKER_FFMPEG = os.environ.get("SADTALKER_FFMPEG", os.path.join(SADTALKER_DIR, "ffmpeg.exe"))
# Job dirs must be ASCII-only: SadTalker builds its ffmpeg calls as os.system() command strings,
# which break on non-ASCII (e.g. Chinese) paths under the cp950 console code page.
RAW_DIR = os.path.join(REPO_ROOT, "outputs", "sadtalker", "_raw")
REQUIRED_CHECKPOINTS = (
    "SadTalker_V0.0.2_256.safetensors",
    "SadTalker_V0.0.2_512.safetensors",
    "mapping_00109-model.pth.tar",
    "mapping_00229-model.pth.tar",
)
SIZES = (256, 512)
PREPROCESS_MODES = ("crop", "extcrop", "resize", "full", "extfull")
ENHANCERS = ("gfpgan", "RestoreFormer")
TIMEOUT_SECONDS = 1800
LOG_TAIL_LINES = 40
_RESULT_RE = re.compile(r"The generated video is named:\s*(.+\.mp4)")


def is_available():
    """(ok, reason) - cheap filesystem check so the GUI can show an install hint instead of a
    subprocess traceback."""
    if not os.path.isfile(SADTALKER_PYTHON):
        return False, f"找不到 SadTalker 的 Python venv：{SADTALKER_PYTHON}"
    missing = [c for c in REQUIRED_CHECKPOINTS if not os.path.isfile(os.path.join(SADTALKER_DIR, "checkpoints", c))]
    if missing:
        return False, f"缺少 SadTalker checkpoint：{', '.join(missing)}"
    if not (os.path.isfile(SADTALKER_FFMPEG) or shutil.which("ffmpeg")):
        return False, f"找不到 ffmpeg（{SADTALKER_FFMPEG}，PATH 上也沒有）"
    return True, ""


def _ffmpeg():
    return SADTALKER_FFMPEG if os.path.isfile(SADTALKER_FFMPEG) else "ffmpeg"


def _prepare_inputs(image_path, audio_path, job_dir):
    """Copy inputs into the ASCII-only job dir; transcode non-wav audio to 16kHz mono wav
    (what SadTalker's audio2coeff front end expects)."""
    img_ext = os.path.splitext(image_path)[1].lower() or ".png"
    img = os.path.join(job_dir, f"source{img_ext}")
    shutil.copyfile(image_path, img)
    wav = os.path.join(job_dir, "voice.wav")
    if os.path.splitext(audio_path)[1].lower() == ".wav":
        shutil.copyfile(audio_path, wav)
    else:
        src = os.path.join(job_dir, "voice_src" + os.path.splitext(audio_path)[1].lower())
        shutil.copyfile(audio_path, src)
        r = subprocess.run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                            "-ac", "1", "-ar", "16000", wav],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.isfile(wav):
            raise RuntimeError(f"音檔轉 wav 失敗：{r.stderr.strip()[-500:]}")
    return img, wav


def generate_talking_head(image_path, audio_path, out_dir, size=512, preprocess="crop", still=False,
                          expression_scale=1.0, enhancer=None, pose_style=0, batch_size=2,
                          echo=False) -> str:
    """Run SadTalker on (portrait, voice) and return the path of the finished .mp4 (with audio).

    image_path must be a fictional/AI-generated face, never a real person's photo.
    size: 256 (faster, ~2-3GB VRAM) or 512 (sharper, ~4-6GB). preprocess: "crop" animates just
    the face crop; "full"/"extfull" pastes it back into the whole image (pair with still=True
    for half/full-body shots so the head doesn't swim). enhancer: None, "gfpgan" or
    "RestoreFormer" (GFPGAN weights ~348MB download on first use). echo=True streams SadTalker's
    log to stdout (CLI); otherwise only the tail is kept for the error message (GUI)."""
    ok, reason = is_available()
    if not ok:
        raise RuntimeError(reason)
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    if preprocess not in PREPROCESS_MODES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_MODES}")
    if enhancer and enhancer not in ENHANCERS:
        raise ValueError(f"enhancer must be one of {ENHANCERS} or None")

    job_dir = os.path.join(RAW_DIR, uuid.uuid4().hex[:12])
    os.makedirs(job_dir, exist_ok=True)
    try:
        img, wav = _prepare_inputs(image_path, audio_path, job_dir)
        cmd = [SADTALKER_PYTHON, "inference.py",
               "--driven_audio", wav, "--source_image", img, "--result_dir", job_dir,
               "--size", str(size), "--preprocess", preprocess,
               "--expression_scale", str(expression_scale), "--pose_style", str(int(pose_style)),
               "--batch_size", str(int(batch_size))]
        if still:
            cmd.append("--still")
        if enhancer:
            cmd += ["--enhancer", enhancer]

        env = dict(os.environ)
        env["SADTALKER_FFMPEG"] = _ffmpeg()
        env["PATH"] = SADTALKER_DIR + os.pathsep + env.get("PATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        # cwd must be the SadTalker dir: --checkpoint_dir defaults to ./checkpoints and the face
        # enhancer resolves gfpgan/weights relative to cwd.
        proc = subprocess.Popen(cmd, cwd=SADTALKER_DIR, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        tail = deque(maxlen=LOG_TAIL_LINES)
        result = None
        try:
            for line in proc.stdout:
                tail.append(line.rstrip())
                if echo:
                    print(line, end="", flush=True)
                m = _RESULT_RE.search(line)
                if m:
                    result = m.group(1).strip()
            proc.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"SadTalker 超過 {TIMEOUT_SECONDS} 秒沒有完成")
        if proc.returncode != 0:
            raise RuntimeError(f"SadTalker 失敗（exit {proc.returncode}）：\n" + "\n".join(tail))
        if not result or not os.path.isfile(result):
            mp4s = sorted((os.path.join(job_dir, f) for f in os.listdir(job_dir) if f.endswith(".mp4")),
                          key=os.path.getmtime)
            if not mp4s:
                raise RuntimeError("SadTalker 跑完了但找不到輸出的 .mp4：\n" + "\n".join(tail))
            result = mp4s[-1]

        os.makedirs(out_dir, exist_ok=True)
        stem = f"talk_{os.path.splitext(os.path.basename(image_path))[0]}_{os.path.splitext(os.path.basename(audio_path))[0]}"
        out_path = os.path.join(out_dir, f"{stem}.mp4")
        shutil.move(result, out_path)  # not os.replace: out_dir may be on another drive
        return out_path
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
