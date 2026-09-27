"""AI short drama: one episode file (a shot list) -> keyframes -> voice -> motion -> a 9:16 mp4.

An episode is a JSON file under training/episodes/ (see example_cafe_reunion.json). Each shot names
a character, scene, optional pose skeleton, framing, what the keyframe shows, what moves, and the
spoken line. Work files go to outputs/drama/<episode>/, so a stage can be re-run shot by shot:

    python training/drama.py plan       training/episodes/ep01.json   # validate + cost estimate
    python training/drama.py keyframes  training/episodes/ep01.json   # local ComfyUI (GPU)
    python training/drama.py voice      training/episodes/ep01.json   # CosyVoice3 (GPU, ComfyUI stopped)
    python training/drama.py motion     training/episodes/ep01.json --preview   # RunPod, cheap drafts
    python training/drama.py motion     training/episodes/ep01.json --final     # RunPod, 704x1280
    python training/drama.py assemble   training/episodes/ep01.json   # ffmpeg -> ep01.mp4
    python training/drama.py status     training/episodes/ep01.json

Low resolution first, professional enhancement later: `keyframes --draft` (smaller, no face pass,
redone by a plain `keyframes`), `motion --draft` (480x832 at full length and steps), then `export`
copies the clips and stills to enhance/in/ for an external upscaler / frame interpolator, and
whatever comes back in enhance/out/ under the same name wins at `assemble` (use --fps 48/60 when
the clips were interpolated).

Where each stage runs follows the cost report: keyframes and voice are free locally, only motion
goes to the cloud, and motion is drafted cheaply before the full render. The safety guarantees are
the existing ones, not new code: keyframes go through gc.plan_picker + gc.gen_custom, and motion
jobs go through cloud_video to the RunPod worker, which composes the tier + age safety negatives
server-side with the cfg floor. Lip-sync is not part of this stage; dialogue shots get the voice
line and a subtitle over Wan's motion.

For language-learning episodes a shot can carry a `translation` (a second, smaller subtitle line under
the spoken one) and a `speed` for the reading, and a `card` shot puts a phrase of the day or an end
card on screen - drawn by ffmpeg over a blurred keyframe of another shot, so it needs no GPU.
`interpolate` doubles the moving shots' frame rate locally with RIFE (MIT) into enhance/out/.

An episode marked "commercial": true may only use components licensed for commercial use: a
keyframe checkpoint from COMMERCIAL_CHECKPOINTS (Z-Image Turbo, Apache-2.0), Wan 2.2 (Apache-2.0),
RIFE (MIT), CosyVoice3 (Apache-2.0) with voices whose speakers consented - never the demo voice.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave

import cloud_video
import comfyui_client as client
import drama_compose as compose
import generate_character as gc
import pose_skeletons
import scene_library

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODES_DIR = os.path.join(TRAINING_DIR, "episodes")
OUTPUT_ROOT = os.environ.get("DRAMA_OUTPUT_ROOT") or os.path.join(gc.PROJECT_ROOT, "outputs", "drama")
VOICES_DIR = os.environ.get("DRAMA_VOICES_DIR") or os.path.join(TRAINING_DIR, "voices")
COSYVOICE_DIR = os.environ.get("COSYVOICE_DIR") or os.path.join(gc.PROJECT_ROOT, "CosyVoice")
COSYVOICE_MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
COSYVOICE_RUNNER = os.path.join(TRAINING_DIR, "cosyvoice_runner.py")
# CosyVoice's bundled prompt voice: a Mandarin female, fine for testing, not for anything published.
DEMO_VOICE_WAV = "asset/zero_shot_prompt.wav"
DEMO_VOICE_TEXT = "希望你以后能够做的比我还好呦。"
COSYVOICE_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
# Keyframe checkpoints whose licences allow commercial use (checked 2026-09-27: Z-Image Turbo's
# repository - transformer, text encoder and VAE - is Apache-2.0). The Pony-derived and SDXL
# checkpoints here, FaceID/InsightFace, the OpenPose ControlNet and UltraSharp all restrict it.
COMMERCIAL_CHECKPOINTS = frozenset({"z_image_turbo"})
INTERP_MULTIPLIER = 2

EPISODE_VERSION = 1
CARD = "card"
SHOT_TYPES = ("still", "motion", "dialogue", CARD)
MOVING_TYPES = ("motion", "dialogue")
FRAMINGS = {
    "close": "close-up shot, head and shoulders",
    "medium": "medium shot from the waist up",
    "wide": "full body wide shot",
}
DURATION_RANGE = (1.5, 5.0)     # Wan's 121-frame ceiling is 5.04 s at 24 fps
MAX_LINE_CHARS = 120
MAX_CARD_TEXT = 80
MAX_CARD_LABEL = 20
MAX_CAST_TEXT = 200
MAX_CANDIDATES = 12
SPEED_RANGE = (0.6, 1.4)
ID_RE = re.compile(r"^[a-z0-9_]{1,32}$")
EPISODE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FPS = compose.FPS
KEYFRAME_SIZE = (768, 1344)     # SDXL bucket closest to 9:16, used when no pose skeleton sets the canvas
KEYFRAME_DRAFT_SIZE = (576, 1024)     # exact 9:16, ~0.6 MP: a quick local draft, redone for the final
KEYFRAME_DRAFT_POSE_SCALE = 0.75      # posed drafts shrink the skeleton canvas, keeping its aspect
WAN_FINAL_SIZE = (704, 1280)
WAN_DRAFT_SIZE = (480, 832)
WAN_PREVIEW = cloud_video.WAN_PREVIEW_PORTRAIT
MOTION_MODES = ("preview", "draft", "final")
MOTION_SUFFIX = {"preview": "_preview", "draft": "_draft", "final": ""}
OUTPUT_FPS_CHOICES = (24, 25, 30, 48, 50, 60)
DEFAULT_OUTPUT_FPS = 24
VOICE_LEAD_S = 0.25             # silence before a line starts
VOICE_TAIL_S = 0.35             # and after it ends, before the cut
DEFAULT_BGM_VOLUME = 0.18

# Cost model for the estimate `plan` and `motion` print. From the 2026-09-26 cost report: RunPod
# Serverless 24GB PRO (4090) at $1.10/h, Wan 2.2 5B's official "5 s of 720p in under 9 minutes" on a
# 4090, a cold start guessed at 2 minutes. Estimates, not measurements - replace after a real run.
RUNPOD_PER_HOUR = 1.10
WAN_FULL_MINUTES = 9.0
WAN_FULL_FRAMES = 121
COLD_START_MINUTES = 2.0
PREVIEW_COST = 0.015
# A draft is the final's length and steps at 480x832: its share of the latent tokens, which
# overstates the saving slightly (attention grows faster than the token count) - an estimate.
DRAFT_COST_RATIO = (WAN_DRAFT_SIZE[0] * WAN_DRAFT_SIZE[1]) / (WAN_FINAL_SIZE[0] * WAN_FINAL_SIZE[1])


class Episode:
    """A loaded episode file plus the paths of its work files."""

    def __init__(self, data, path):
        self.data = data
        self.path = os.path.abspath(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.work_dir = os.path.normpath(os.path.join(OUTPUT_ROOT, self.name))

    @property
    def shots(self):
        return self.data.get("shots") or []

    @property
    def tier(self):
        return self.data.get("tier", "safe")

    @property
    def checkpoint(self):
        return self.data.get("checkpoint") or gc.DEFAULT_CUSTOM_CHECKPOINT

    @property
    def fps(self):
        return self.data.get("fps", DEFAULT_OUTPUT_FPS)

    def cast_for(self, character):
        """This episode's appearance/style overrides for a character (e.g. a barista apron), or None."""
        return (self.data.get("cast") or {}).get(character) or None

    def shot_seed(self, index, shot):
        if shot.get("seed") is not None:
            return int(shot["seed"])
        return int(self.data.get("seed", 8100)) + index * 10

    def sub(self, *parts):
        return os.path.join(self.work_dir, *parts)

    def keyframe(self, shot_id):
        return self.sub("keyframes", f"{shot_id}.png")

    def wan_input(self, shot_id):
        return self.sub("keyframes", f"{shot_id}_wan.png")

    def voice(self, shot_id):
        return self.sub("voice", f"{shot_id}.wav")

    def keyframe_draft_marker(self, shot_id):
        return self.sub("keyframes", f"{shot_id}.draft")

    def candidate(self, shot_id, seed):
        return self.sub("keyframes", "candidates", f"{shot_id}_seed{seed}.png")

    def candidate_draft_marker(self, shot_id, seed):
        return self.sub("keyframes", "candidates", f"{shot_id}_seed{seed}.draft")

    def motion(self, shot_id, mode="final"):
        return self.sub("motion", f"{shot_id}{MOTION_SUFFIX[mode]}.mp4")

    def enhanced(self, shot_id, ext):
        return self.sub("enhance", "out", f"{shot_id}.{ext}")

    def text_file(self, shot_id, kind, n):
        return self.sub("subs", f"{shot_id}_{kind}_{n}.txt")

    def segment(self, shot_id):
        return self.sub("segments", f"{shot_id}.mkv")

    def output(self):
        return self.sub(f"{self.name}.mp4")

    def resolve(self, rel):
        """A path in the episode file, relative to the episode file's own folder."""
        return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(os.path.dirname(self.path), rel))


def load_episode(path):
    if not os.path.isfile(path):
        raise gc.UsageError(f"找不到分鏡表：{path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise gc.UsageError(f"分鏡表不是合法的 JSON（第 {exc.lineno} 行）：{exc.msg}")
    if not isinstance(data, dict):
        raise gc.UsageError("分鏡表的最外層要是一個物件 {...}")
    return Episode(data, path)


def validate_episode(ep):
    """Return (errors, warnings), both lists of Chinese messages. Errors block every stage."""
    errors, warnings = [], []
    d = ep.data
    if not EPISODE_NAME_RE.match(ep.name):
        errors.append(f"分鏡表檔名只能用英數字、- 和 _（目前是「{ep.name}」），它會變成輸出資料夾的名字")
    if d.get("version") != EPISODE_VERSION:
        errors.append(f"version 要是 {EPISODE_VERSION}")
    if ep.tier not in scene_library.TIERS:
        errors.append(f"tier 只能是 {list(scene_library.TIERS)}")
    ckpt = ep.checkpoint
    if ckpt not in client.CHECKPOINTS and ckpt not in client.ZIMAGE_MODELS:
        errors.append(f"不認得的 checkpoint「{ckpt}」")
    elif ckpt in client.ZIMAGE_MODELS or ckpt in client.SD15_CHECKPOINTS:
        warnings.append(f"checkpoint「{ckpt}」沒有 FaceID／ControlNet，角色不會鎖臉、姿勢只用文字描述")
    voices = d.get("voices") or {}
    if not isinstance(voices, dict):
        errors.append("voices 要是 {角色: 聲音 id} 的物件")
        voices = {}
    for character, voice_id in voices.items():
        if not isinstance(voice_id, str) or not ID_RE.match(voice_id):
            errors.append(f"voices.{character} 的聲音 id 只能用小寫英數字和 _")
    bgm = d.get("bgm")
    if bgm and not os.path.isfile(ep.resolve(bgm)):
        errors.append(f"找不到配樂檔：{ep.resolve(bgm)}")
    if ep.fps not in OUTPUT_FPS_CHOICES:
        errors.append(f"fps（成片影格率）只能是 {list(OUTPUT_FPS_CHOICES)}")
    cast = d.get("cast") or {}
    if not isinstance(cast, dict):
        errors.append("cast 要是 {角色: {appearance／style 覆蓋}} 的物件")
    else:
        for character, overrides in cast.items():
            if character not in gc.CHARACTERS:
                errors.append(f"cast 裡的角色「{character}」不在角色表裡")
                continue
            if not isinstance(overrides, dict) or not overrides:
                errors.append(f"cast.{character} 要是 {{\"appearance\": ..., \"style\": ...}} 的物件")
                continue
            for key, value in overrides.items():
                if key not in gc.CHARACTER_OVERRIDE_KEYS:
                    errors.append(f"cast.{character} 只能覆蓋 {list(gc.CHARACTER_OVERRIDE_KEYS)}（年齡和性別不能改），"
                                  f"不能有「{key}」")
                elif not isinstance(value, str) or not value.strip() or len(value) > MAX_CAST_TEXT:
                    errors.append(f"cast.{character}.{key} 要是 {MAX_CAST_TEXT} 字以內的文字")
    commercial = d.get("commercial", False)
    if not isinstance(commercial, bool):
        errors.append("commercial 要是 true 或 false")
    elif commercial and ckpt not in COMMERCIAL_CHECKPOINTS:
        errors.append(f"這集標示 commercial，但 checkpoint「{ckpt}」的授權不允許營利；"
                      f"可營利的是 {sorted(COMMERCIAL_CHECKPOINTS)}")

    shots = d.get("shots")
    if not isinstance(shots, list) or not shots:
        errors.append("shots 至少要有一個鏡頭")
        return errors, warnings
    seen, picture_ids, card_backgrounds = set(), set(), []
    for index, shot in enumerate(shots):
        label = f"鏡頭 {index + 1}"
        if not isinstance(shot, dict):
            errors.append(f"{label} 要是一個物件")
            continue
        shot_id = shot.get("id")
        if not isinstance(shot_id, str) or not ID_RE.match(shot_id):
            errors.append(f"{label} 的 id 只能用小寫英數字和 _（1–32 字），會變成檔名")
            shot_id = None
        else:
            label = f"鏡頭 {shot_id}"
            if shot_id in seen:
                errors.append(f"{label} 的 id 重複")
            seen.add(shot_id)
        kind = shot.get("type")
        if kind not in SHOT_TYPES:
            errors.append(f"{label} 的 type 只能是 {list(SHOT_TYPES)}")
        duration = shot.get("duration")
        if not isinstance(duration, (int, float)) or not DURATION_RANGE[0] <= duration <= DURATION_RANGE[1]:
            errors.append(f"{label} 的 duration 要在 {DURATION_RANGE[0]}–{DURATION_RANGE[1]} 秒之間")
        character = shot.get("character")
        if character and character not in gc.CHARACTERS:
            errors.append(f"{label} 的角色「{character}」不在角色表裡")
            character = None
        _validate_speech(shot, label, kind, errors, warnings)
        if kind == CARD:
            _validate_card(shot, label, errors)
            if shot.get("background"):
                card_backgrounds.append((label, shot["background"]))
            continue
        if shot_id:
            picture_ids.add(shot_id)
        framing = shot.get("framing", "medium")
        if framing not in FRAMINGS:
            errors.append(f"{label} 的 framing 只能是 {list(FRAMINGS)}")
        camera = shot.get("camera")
        if camera is not None and camera not in compose.CAMERA_MOVES:
            errors.append(f"{label} 的 camera 只能是 {list(compose.CAMERA_MOVES)}")
        if character and ckpt in client.CHECKPOINTS and ckpt not in client.SD15_CHECKPOINTS \
                and not gc.picker_anchor_path(character):
            warnings.append(f"{label}：角色「{character}」還沒有 anchor 圖，生關鍵幀前要先產生一張")
        scene = shot.get("scene")
        if scene:
            try:
                scene_tier = scene_library.get(scene)["tier"]
            except ValueError:
                errors.append(f"{label} 的場景「{scene}」不在場景庫裡")
            else:
                too_high = (ep.tier in scene_library.TIERS
                            and scene_library.TIERS.index(scene_tier) > scene_library.TIERS.index(ep.tier))
                if too_high:
                    errors.append(f"{label} 的場景「{scene}」是 {scene_tier} 等級，這集是 {ep.tier}")
        pose = shot.get("pose")
        if pose:
            if not isinstance(pose, str) or pose_skeletons.resolve(pose) is None:
                errors.append(f"{label} 的姿勢「{pose}」不在姿勢庫裡")
            if framing == "close":
                warnings.append(f"{label}：姿勢骨架是全身的，跟特寫（close）搭配時可能不會照做")
        prompt = shot.get("prompt") or ""
        if not (prompt.strip() or scene):
            errors.append(f"{label} 要有 prompt 或 scene，關鍵幀才有畫面內容")
        if len(prompt) > gc.MAX_PROMPT_CHARS:
            errors.append(f"{label} 的 prompt 太長")
        if kind in MOVING_TYPES:
            motion = shot.get("motion") or prompt
            if not motion.strip():
                errors.append(f"{label} 要有 motion（描述要發生的動作）")
            elif len(motion) > gc.MAX_PROMPT_CHARS:
                errors.append(f"{label} 的 motion 太長")
        if kind == "dialogue" and not (shot.get("line") or "").strip():
            errors.append(f"{label} 是對話鏡頭，要有 line（台詞）")
    if d.get("commercial") is True:
        for shot in shots:
            if not isinstance(shot, dict) or not (shot.get("line") or "").strip():
                continue
            voice_id = shot.get("voice") or voices.get(shot.get("character") or "")
            if not voice_id:
                errors.append(f"鏡頭 {shot.get('id')}：這集標示 commercial，有台詞的鏡頭都要有已授權的聲音"
                              f"（voices 對應角色，或鏡頭自己的 voice），不能用示範聲音")
    for label, background in card_backgrounds:
        if background not in picture_ids:
            errors.append(f"{label} 的 background「{background}」要是另一個非字卡鏡頭的 id（用它的關鍵幀當背景）")
    return errors, warnings


def _validate_speech(shot, label, kind, errors, warnings):
    line = shot.get("line")
    if line is not None and not isinstance(line, str):
        errors.append(f"{label} 的 line 要是文字")
        line = None
    if line and len(line) > MAX_LINE_CHARS:
        errors.append(f"{label} 的台詞超過 {MAX_LINE_CHARS} 字，請拆成兩個鏡頭")
    translation = shot.get("translation")
    if translation is not None:
        if not isinstance(translation, str) or len(translation) > MAX_LINE_CHARS:
            errors.append(f"{label} 的 translation 要是 {MAX_LINE_CHARS} 字以內的文字")
        elif kind != CARD and not (line or "").strip():
            warnings.append(f"{label} 有 translation 但沒有 line，翻譯字幕不會出現")
    speed = shot.get("speed")
    if speed is not None and (not isinstance(speed, (int, float)) or not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]):
        errors.append(f"{label} 的 speed（語速）要在 {SPEED_RANGE[0]}–{SPEED_RANGE[1]} 之間")
    voice_id = shot.get("voice")
    if voice_id is not None and (not isinstance(voice_id, str) or not ID_RE.match(voice_id)):
        errors.append(f"{label} 的 voice 只能用小寫英數字和 _")


def _validate_card(shot, label, errors):
    phrase = shot.get("phrase")
    if not isinstance(phrase, str) or not phrase.strip():
        errors.append(f"{label} 是字卡，要有 phrase（主要文字）")
    elif len(phrase) > MAX_CARD_TEXT:
        errors.append(f"{label} 的 phrase 超過 {MAX_CARD_TEXT} 字")
    for key, limit in (("note", MAX_CARD_TEXT), ("label", MAX_CARD_LABEL)):
        value = shot.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            errors.append(f"{label} 的 {key} 要是 {limit} 字以內的文字")


def require_valid(ep):
    errors, warnings = validate_episode(ep)
    for w in warnings:
        print(f"[注意] {w}", flush=True)
    if errors:
        raise gc.UsageError("分鏡表有問題：\n  " + "\n  ".join(errors))


def select_shots(ep, ids=None, types=None):
    """Shots in episode order, filtered by --shots ids and/or type."""
    wanted = None
    if ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
        unknown = set(wanted) - {s["id"] for s in ep.shots}
        if unknown:
            raise gc.UsageError(f"分鏡表裡沒有這些鏡頭：{sorted(unknown)}")
    out = []
    for index, shot in enumerate(ep.shots):
        if wanted is not None and shot["id"] not in wanted:
            continue
        if types is not None and shot["type"] not in types:
            continue
        out.append((index, shot))
    return out


# --- frames and cost --------------------------------------------------------------------------


def frames_for(duration):
    """Wan frame count (4n+1, 17..121) closest to duration seconds at 24 fps."""
    n = round((duration * FPS - 1) / 4)
    return max(gc.WAN_FRAME_RANGE[0], min(gc.WAN_FRAME_RANGE[1], 4 * n + 1))


def _latent_frames(frames):
    return (frames - 1) // 4 + 1


def final_cost(frames):
    """Estimated RunPod cost of one 704x1280 render, excluding the cold start."""
    minutes = WAN_FULL_MINUTES * _latent_frames(frames) / _latent_frames(WAN_FULL_FRAMES)
    return RUNPOD_PER_HOUR / 60 * minutes


def estimate(ep, shots, mode):
    """(job count, estimated dollars) for sending these shots' motion in one batch in `mode`."""
    moving = [(i, s) for i, s in shots if s["type"] in MOVING_TYPES]
    if not moving:
        return 0, 0.0
    cold = RUNPOD_PER_HOUR / 60 * COLD_START_MINUTES
    if mode == "preview":
        return len(moving), cold + PREVIEW_COST * len(moving)
    ratio = DRAFT_COST_RATIO if mode == "draft" else 1.0
    return len(moving), cold + ratio * sum(final_cost(frames_for(s["duration"])) for _i, s in moving)


# --- keyframes --------------------------------------------------------------------------------


def keyframe_request(ep, index, shot, translate=None, draft=False):
    """Everything gen_custom needs for one shot's keyframe, resolved through plan_picker exactly like
    the picker tab: SDXL/Pony get the character as a FaceID anchor and the pose as a ControlNet
    skeleton, and the scene library supplies the setting."""
    prompt = (shot.get("prompt") or "").strip()
    if compose.is_cjk_text(prompt):
        if translate is None:
            import translate_prompt

            translate = translate_prompt.translate_to_english
        try:
            prompt = translate(prompt)
        except Exception as exc:
            raise gc.UsageError(f"鏡頭 {shot['id']} 的 prompt 翻譯失敗（{exc}），請改寫成英文")
    extra = ", ".join(x for x in (FRAMINGS[shot.get("framing", "medium")], prompt) if x)
    plan = gc.plan_picker(shot.get("character"), shot.get("pose"), shot.get("scene"), ep.checkpoint, ep.tier, extra)
    if plan.trigger and not plan.anchor_path and plan.mode == "controlnet_faceid":
        raise gc.UsageError(
            f"鏡頭 {shot['id']}：角色「{plan.trigger}」還沒有 anchor 圖。先跑 "
            f"`generate_character.py anchors --character {plan.trigger}`"
        )
    if plan.pose_name:
        width = height = None   # the skeleton's own canvas
        if draft:
            cw, ch = pose_skeletons.canvas_for(plan.pose_name)
            width, height = _round16(cw * KEYFRAME_DRAFT_POSE_SCALE), _round16(ch * KEYFRAME_DRAFT_POSE_SCALE)
    else:
        width, height = KEYFRAME_DRAFT_SIZE if draft else KEYFRAME_SIZE
    return {
        "prompt": plan.prompt_body,
        "trigger": plan.trigger,
        "anchor_path": plan.anchor_path,
        "pose_name": plan.pose_name,
        "width": width,
        "height": height,
        "seed": ep.shot_seed(index, shot),
        "notices": plan.notices,
    }


def _round16(value):
    return int(round(value / 16)) * 16


def candidate_seeds(ep, index, shot, count):
    """The seeds a --candidates run tries for a shot: the next `count` after its current seed."""
    base = ep.shot_seed(index, shot)
    return [base + k for k in range(1, count + 1)]


def _render_keyframe(ep, req, seed, stem, dst, *, draft, gen, run, ffmpeg):
    out_dir = os.path.dirname(dst)
    gen(req["prompt"], "", ep.tier, req["trigger"], req["anchor_path"], out_dir, seed, stem,
        client.IP_ADAPTER_WEIGHT, width=req["width"], height=req["height"],
        use_facedetailer=False if draft else None,
        style_positive=gc.REALISTIC_STYLE, style_negative=gc.REALISTIC_NEGATIVE,
        checkpoint=ep.checkpoint, lora_strength=0.0, hq=True, pose_name=req["pose_name"],
        character_overrides=ep.cast_for(req["trigger"]) if req["trigger"] else None)
    run(compose.normalise_command(ffmpeg, os.path.join(out_dir, f"{stem}.png"), dst), check=True)


def _set_marker(path, on, text=""):
    if on:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    elif os.path.isfile(path):
        os.remove(path)


def run_keyframes(ep, shots, *, force=False, draft=False, candidates=0, ffmpeg, translate=None, gen=None,
                  run=subprocess.run, server_up=client.is_server_running):
    """Generate keyframes. draft=True renders smaller with the face pass off and marks the result;
    a later non-draft run regenerates marked drafts without needing --force.

    candidates=N instead renders N alternatives per shot (the next N seeds) into
    keyframes/candidates/ and leaves the shot's keyframe alone; `pick` then promotes one."""
    gen = gen or gc.gen_custom
    shots = [(i, s) for i, s in shots if s["type"] != CARD]   # cards are drawn by ffmpeg, no keyframe
    if candidates and not 1 <= candidates <= MAX_CANDIDATES:
        raise gc.UsageError(f"--candidates 要在 1–{MAX_CANDIDATES} 之間")
    if not server_up():
        raise gc.UsageError("ComfyUI 沒有在跑。先啟動它（或開 GUI，會自動啟動），再生關鍵幀")
    os.makedirs(ep.sub("keyframes"), exist_ok=True)
    kind = "草稿關鍵幀" if draft else "關鍵幀"
    made = 0
    for index, shot in shots:
        sid = shot["id"]
        req = None
        if candidates:
            os.makedirs(ep.sub("keyframes", "candidates"), exist_ok=True)
            for seed in candidate_seeds(ep, index, shot, candidates):
                dst = ep.candidate(sid, seed)
                if os.path.isfile(dst) and not force:
                    print(f"{sid}: 候選 seed {seed} 已存在，略過", flush=True)
                    continue
                req = req or keyframe_request(ep, index, shot, translate=translate, draft=draft)
                print(f"{sid}: 生成候選{kind}（seed {seed}）", flush=True)
                _render_keyframe(ep, req, seed, f"{sid}_seed{seed}_raw", dst, draft=draft, gen=gen, run=run,
                                 ffmpeg=ffmpeg)
                _set_marker(ep.candidate_draft_marker(sid, seed), draft, f"{req['width']}x{req['height']}\n")
                made += 1
            continue
        final = ep.keyframe(sid)
        marker = ep.keyframe_draft_marker(sid)
        is_draft = os.path.isfile(marker)
        if os.path.isfile(final) and not force and (draft or not is_draft):
            print(f"{sid}: 已有{'草稿' if is_draft else ''}關鍵幀，略過（--force 重生）", flush=True)
            continue
        req = keyframe_request(ep, index, shot, translate=translate, draft=draft)
        for notice in req["notices"]:
            print(f"[注意] {sid}: {notice}", flush=True)
        print(f"{sid}: 生成{kind}（seed {req['seed']}）", flush=True)
        _render_keyframe(ep, req, req["seed"], f"{sid}_raw", final, draft=draft, gen=gen, run=run, ffmpeg=ffmpeg)
        _set_marker(marker, draft, f"{req['width']}x{req['height']}\n")
        made += 1
    if candidates:
        print("挑一張：drama.py pick <分鏡表> --shot <鏡頭> --seed <seed>", flush=True)
    return made


def run_pick(ep, shot_id, seed):
    """Promote a candidate to the shot's keyframe and write its seed into the episode file, so a
    later regeneration of this shot starts from the same seed. The seed reproduces the picture only
    at the same size - a final-size regeneration of a picked draft is a new picture."""
    shot = next((s for s in ep.shots if s["id"] == shot_id), None)
    if shot is None or shot["type"] == CARD:
        raise gc.UsageError(f"分鏡表裡沒有可以挑關鍵幀的鏡頭「{shot_id}」")
    src = ep.candidate(shot_id, seed)
    if not os.path.isfile(src):
        raise gc.UsageError(f"找不到候選 {src}；先跑 keyframes --shots {shot_id} --candidates N")
    shutil.copy2(src, ep.keyframe(shot_id))
    was_draft = os.path.isfile(ep.candidate_draft_marker(shot_id, seed))
    _set_marker(ep.keyframe_draft_marker(shot_id), was_draft, "picked\n")
    shot["seed"] = int(seed)
    with open(ep.path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(ep.data, ensure_ascii=False, indent=2) + "\n")
    print(f"{shot_id}: 改用 seed {seed}（{'草稿' if was_draft else '正式'}），已寫回分鏡表", flush=True)
    stale = [p for p in (ep.motion(shot_id, m) for m in MOTION_MODES) if os.path.isfile(p)]
    stale += [p for p in (ep.enhanced(shot_id, "mp4"),) if os.path.isfile(p)]
    if stale:
        print(f"[注意] {shot_id} 已有用舊關鍵幀做的影片，要重做：{[os.path.basename(p) for p in stale]}", flush=True)
    return ep.keyframe(shot_id)


# --- voice ------------------------------------------------------------------------------------


def resolve_voice(ep, shot, allow_demo=False):
    """(prompt_wav, prompt_transcript, is_demo) for a shot's speaker.

    A voice lives in training/voices/<id>/voice.json:
        {"prompt_wav": "prompt.wav", "prompt_text": "exact transcript of that recording",
         "consent": {"speaker": "who", "date": "YYYY-MM-DD", "scope": "what it may be used for"}}
    The consent block records that the speaker agreed to have the voice cloned; a voice without it
    is refused. The CosyVoice demo prompt is only used when allow_demo is set, and never for an
    episode marked commercial."""
    voice_id = shot.get("voice") or (ep.data.get("voices") or {}).get(shot.get("character") or "")
    if not voice_id:
        if ep.data.get("commercial") is True:
            raise gc.UsageError(f"鏡頭 {shot['id']}：這集標示 commercial，不能用示範聲音，請設定已授權的聲音")
        if not allow_demo:
            raise gc.UsageError(f"鏡頭 {shot['id']}：角色「{shot.get('character') or '（無）'}」沒有設定聲音。"
                                "請在 voices 指定已授權的聲音；只是內部測試的話加 --allow-demo")
        return os.path.join(COSYVOICE_DIR, DEMO_VOICE_WAV), DEMO_VOICE_TEXT, True
    folder = os.path.join(VOICES_DIR, voice_id)
    config = os.path.join(folder, "voice.json")
    if not os.path.isfile(config):
        raise gc.UsageError(f"鏡頭 {shot['id']}：找不到聲音設定 {config}")
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    wav = os.path.join(folder, cfg.get("prompt_wav", "prompt.wav"))
    text = (cfg.get("prompt_text") or "").strip()
    if not os.path.isfile(wav) or not text:
        raise gc.UsageError(f"聲音「{voice_id}」要有 prompt_wav 錄音檔和它的逐字稿 prompt_text")
    consent = cfg.get("consent") or {}
    if not (isinstance(consent, dict) and str(consent.get("speaker") or "").strip()
            and str(consent.get("date") or "").strip()):
        raise gc.UsageError(f"聲音「{voice_id}」缺少授權紀錄：voice.json 要有 consent.speaker（本人）和 consent.date")
    return wav, text, False


def voice_status(ep):
    """[(voice id or None, [character or shot ids], state)] for every voice the episode's lines use."""
    voices = ep.data.get("voices") or {}
    groups = {}
    for shot in ep.shots:
        if not (shot.get("line") or "").strip():
            continue
        voice_id = shot.get("voice") or voices.get(shot.get("character") or "")
        who = shot.get("character") or shot["id"]
        groups.setdefault(voice_id, [])
        if who not in groups[voice_id]:
            groups[voice_id].append(who)
    out = []
    for voice_id, who in groups.items():
        if not voice_id:
            state = "沒有設定（只能用 --allow-demo 的示範聲音）"
        else:
            probe = {"id": "-", "voice": voice_id}
            try:
                resolve_voice(ep, probe)
                state = "已設定、有授權紀錄"
            except gc.UsageError as exc:
                state = str(exc).replace("鏡頭 -：", "")
        out.append((voice_id, who, state))
    return out


def voice_mode(line, transcript, emotion):
    """How CosyVoice3 should speak a line. A line in a different language from the reference
    recording (English lines, a Mandarin reference) goes through cross_lingual, which keeps the timbre
    without trying to copy the recording's words; emotion instructions only apply otherwise."""
    if compose.is_cjk_text(line) != compose.is_cjk_text(transcript):
        return "cross_lingual"
    return "instruct" if emotion else "zero_shot"


def voice_items(ep, shots, allow_demo=False):
    """One CosyVoice job item per shot with a line (cards included - their line is the reading)."""
    items, demo_used = [], False
    for _index, shot in shots:
        line = (shot.get("line") or "").strip()
        if not line:
            continue
        wav, transcript, is_demo = resolve_voice(ep, shot, allow_demo)
        demo_used = demo_used or is_demo
        emotion = (shot.get("emotion") or "").strip()
        mode = voice_mode(line, transcript, emotion)
        if mode == "cross_lingual" and emotion:
            print(f"[注意] {shot['id']}: 台詞和參考錄音不同語言，走跨語言配音，語氣「{emotion}」不會套用", flush=True)
        items.append({
            "id": shot["id"],
            "text": line,
            "out": ep.voice(shot["id"]),
            "prompt_wav": wav,
            "prompt_text": COSYVOICE_PROMPT_PREFIX + transcript,
            "mode": mode,
            "instruct": f"You are a helpful assistant. 请用{emotion}的语气说这句话。<|endofprompt|>"
                        if mode == "instruct" else None,
            "speed": float(shot.get("speed", 1.0)),
        })
    return items, demo_used


def cosyvoice_python():
    for rel in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        path = os.path.join(COSYVOICE_DIR, *rel)
        if os.path.isfile(path):
            return path
    return None


def run_voice(ep, shots, *, force=False, ignore_comfyui=False, allow_demo=False, run=subprocess.run,
              server_up=client.is_server_running):
    python = cosyvoice_python()
    if not python:
        raise gc.UsageError(f"找不到 CosyVoice 的 venv（{COSYVOICE_DIR}\\.venv），配音需要它")
    if server_up() and not ignore_comfyui:
        raise gc.UsageError("ComfyUI 正在跑，會跟 CosyVoice 搶 8 GB 顯卡。先跑 training\\stop_comfyui.ps1，"
                            "或加 --ignore-comfyui")
    items, demo_used = voice_items(ep, shots, allow_demo)
    if not force:
        items = [it for it in items if not os.path.isfile(it["out"])]
    if not items:
        print("沒有需要配音的台詞（都已經有了，--force 重配）", flush=True)
        return 0
    if demo_used:
        print("[注意] 有角色沒有設定聲音，改用 CosyVoice 的示範聲音——只能內部測試，不能公開", flush=True)
    os.makedirs(ep.sub("voice"), exist_ok=True)
    job = ep.sub("voice", "job.json")
    with open(job, "w", encoding="utf-8") as f:
        json.dump({"model_dir": COSYVOICE_MODEL_DIR, "items": items}, f, ensure_ascii=False, indent=2)
    print(f"配音 {len(items)} 句（CosyVoice3）...", flush=True)
    # CosyVoice logs Simplified Chinese; on a cp950 console a strict stdout would crash on it.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    run([python, COSYVOICE_RUNNER, job], cwd=COSYVOICE_DIR, check=True, env=env)
    missing = [it["id"] for it in items if not os.path.isfile(it["out"])]
    if missing:
        raise gc.UsageError(f"CosyVoice 結束了但這些鏡頭沒有音檔：{missing}")
    return len(items)


def wav_seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


# --- motion -----------------------------------------------------------------------------------


def motion_specs(ep, shots, mode):
    """One CloudJobSpec per motion/dialogue shot. The first frame is the shot's keyframe at Wan's
    size; the prompt is the raw motion text plus the tier - the worker adds the character's
    description and the safety negatives."""
    specs = []
    for index, shot in shots:
        if shot["type"] not in MOVING_TYPES:
            continue
        frames = frames_for(shot["duration"])
        if mode == "preview":
            params = dict(WAN_PREVIEW)
            params["frames"] = min(frames, WAN_PREVIEW["frames"])
        else:
            width, height = WAN_DRAFT_SIZE if mode == "draft" else WAN_FINAL_SIZE
            params = {"width": width, "height": height, "frames": frames, "steps": client.WAN_STEPS}
        params.update(fps=FPS, cfg=client.WAN_CFG)
        gc.check_wan_params(params["width"], params["height"], params["frames"], params["fps"], params["steps"],
                            params["cfg"], client.WAN_DEFAULT_MODEL)
        stem = os.path.splitext(os.path.basename(ep.motion(shot["id"], mode)))[0]
        motion = (shot.get("motion") or shot.get("prompt") or "").strip()
        character = shot.get("character") or None
        overrides = ep.cast_for(character) if character else None
        if overrides:
            # The worker only knows the character table, so an episode's wardrobe travels in the prompt:
            # the overridden description (age and gender still from CHARACTERS) and no characterId, or
            # the worker would add the default outfit back and Wan would fight the first frame's.
            profile = gc.character_profile(character, overrides)
            motion = f"{gc.character_base_prompt(character, profile, include_trigger=False)}, {motion}"
            character = None
        specs.append(cloud_video.CloudJobSpec(
            stem=stem, prompt=motion, image_path=ep.wan_input(shot["id"]), trigger=character,
            seed=ep.shot_seed(index, shot), params=params, tier=ep.tier,
        ))
    return specs


def run_motion(ep, shots, *, mode, force=False, yes=False, provider="runpod", timeout=None, ffmpeg,
               run=subprocess.run, run_batch=None, confirm=input, stdin_isatty=None):
    run_batch = run_batch or cloud_video.run_cloud_batch
    moving = [(i, s) for i, s in shots if s["type"] in MOVING_TYPES]
    if not force:
        moving = [(i, s) for i, s in moving if not os.path.isfile(ep.motion(s["id"], mode))]
    if not moving:
        print("沒有需要送出的動態鏡頭（都已經有了，--force 重送）", flush=True)
        return []
    missing = [s["id"] for _i, s in moving if not os.path.isfile(ep.keyframe(s["id"]))]
    if missing:
        raise gc.UsageError(f"這些鏡頭還沒有關鍵幀，先跑 keyframes：{missing}")
    specs = motion_specs(ep, moving, mode)
    count, dollars = estimate(ep, moving, mode)
    label = {"preview": "預覽（480×832、12 步）", "draft": "低解析度完整版（480×832，之後交給外部工具加強）",
             "final": "正式版（704×1280）"}[mode]
    print(f"將送出 {count} 支{label}到 {provider}，估計約 ${dollars:.2f}（未實測）", flush=True)
    if not yes:
        isatty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
        if not isatty:
            raise gc.UsageError("雲端會計費：確認後加 --yes 再跑")
        if confirm("確定送出？[y/N] ").strip().lower() not in ("y", "yes"):
            print("已取消，沒有送出任何工作", flush=True)
            return []
    for _i, shot in moving:
        run(compose.normalise_command(ffmpeg, ep.keyframe(shot["id"]), ep.wan_input(shot["id"]),
                                      *WAN_FINAL_SIZE), check=True)
    results = run_batch(provider, "video_wan_i2v", specs, out_dir=ep.sub("motion"), max_wait_s=timeout,
                        on_status=cloud_video.print_status)
    log_motion(ep, results, mode)
    failed = 0
    for spec, outcome in results:
        if isinstance(outcome, cloud_video.CloudResult):
            print(f"{spec.stem}: {outcome.path}（執行 {outcome.execution_s}s，排隊 {outcome.queue_s}s）", flush=True)
        else:
            failed += 1
            print(f"{spec.stem}: 失敗 - {outcome}", flush=True)
    if failed:
        print(f"{failed} 支失敗；再跑一次同樣的指令只會重送沒完成的鏡頭", flush=True)
    return results


def log_motion(ep, results, mode):
    """Append each job to motion/jobs.json, with a billed-cost floor from the execution seconds
    (the cold start and idle seconds are billed too, so the real figure is higher)."""
    path = ep.sub("motion", "jobs.json")
    entries = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    for spec, outcome in results:
        entry = {"stem": spec.stem, "mode": mode, "seed": spec.seed, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if isinstance(outcome, cloud_video.CloudResult):
            entry.update(job_id=outcome.job_id, queue_s=outcome.queue_s, execution_s=outcome.execution_s,
                         cost_floor=round((outcome.execution_s or 0) / 3600 * RUNPOD_PER_HOUR, 4))
        else:
            entry.update(job_id=getattr(outcome, "job_id", None), error=str(outcome))
        entries.append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


# --- assemble ---------------------------------------------------------------------------------


def shot_plan(ep, shot, allow_preview=False, fps=None):
    """What the assembler will use for one shot: picture source, voice, duration. A clip or still
    that came back from an external enhancer (enhance/out/) beats everything generated here."""
    fps = fps or ep.fps
    sid = shot["id"]
    has_line = bool((shot.get("line") or "").strip())
    voice = ep.voice(sid) if has_line and os.path.isfile(ep.voice(sid)) else None
    duration = float(shot["duration"])
    if voice:
        duration = max(duration, VOICE_LEAD_S + wav_seconds(voice) + VOICE_TAIL_S)
    duration = round(compose.frame_exact(duration, fps), 6)
    if shot["type"] == CARD:
        background = ep.keyframe(shot["background"]) if shot.get("background") else None
        return {"card": True, "background": background, "voice": voice, "duration": duration}
    video = source = None
    if shot["type"] in MOVING_TYPES:
        candidates = [("增強版", ep.enhanced(sid, "mp4")), ("正式版", ep.motion(sid)), ("低解析度版", ep.motion(sid, "draft"))]
        if allow_preview:
            candidates.append(("預覽", ep.motion(sid, "preview")))
        for label, path in candidates:
            if os.path.isfile(path):
                video, source = path, label
                break
    still = None
    if not video:
        enhanced = ep.enhanced(sid, "png")
        still, source = (enhanced, "增強靜態圖") if os.path.isfile(enhanced) else (ep.keyframe(sid), "靜態圖")
    return {"card": False, "video": video, "still": still, "voice": voice, "duration": duration,
            "camera": shot.get("camera") or "push_in", "source": source}


def write_text_lines(ep, shot_id, kind, text, cjk_width, latin_width):
    """Wrap text and write one file per line (drawtext draws each centred on its own width)."""
    paths = []
    for n, line in enumerate(compose.wrap_lines(text, cjk_width, latin_width)):
        path = ep.text_file(shot_id, kind, n)
        with open(path, "w", encoding="utf-8") as f:
            f.write(line)
        paths.append(path)
    return paths


def card_blocks(ep, shot):
    blocks = []
    for name in ("label", "phrase", "translation", "note"):
        cjk_width, latin_width = compose.CARD_BLOCKS[name][2:]
        blocks.append((name, write_text_lines(ep, shot["id"], name, shot.get(name) or "", cjk_width, latin_width)))
    return blocks


def run_assemble(ep, *, allow_preview=False, ffmpeg, font, run=subprocess.run, fps=None):
    fps = fps or ep.fps
    plans, missing = [], []
    for shot in ep.shots:
        plan = shot_plan(ep, shot, allow_preview, fps)
        picture = plan["background"] if plan["card"] else plan["still"]
        if picture and not os.path.isfile(picture):
            missing.append(shot["id"])
        plans.append((shot, plan))
    if missing:
        raise gc.UsageError(f"這些鏡頭沒有影片也沒有關鍵幀（字卡是背景鏡頭的關鍵幀），無法組裝：{missing}")
    needs_text = any((s.get("line") or "").strip() or s["type"] == CARD for s in ep.shots)
    if needs_text and not font:
        raise gc.UsageError("找不到中文字型（微軟正黑體），設定 DRAMA_FONT 指向一個 .ttf/.ttc")
    for sub in ("segments", "subs"):
        os.makedirs(ep.sub(sub), exist_ok=True)
    segments, total = [], 0.0
    for shot, plan in plans:
        sid = shot["id"]
        spoken = "，有配音" if plan["voice"] else ""
        if plan["card"]:
            print(f"{sid}: {plan['duration']:.2f}s，字卡{spoken}", flush=True)
            cmd = compose.card_command(ffmpeg, out=ep.segment(sid), duration=plan["duration"],
                                       blocks=card_blocks(ep, shot), font=font, background=plan["background"],
                                       voice=plan["voice"], voice_delay=VOICE_LEAD_S, fps=fps)
        else:
            line = (shot.get("line") or "").strip()
            subtitle = write_text_lines(ep, sid, "line", line, compose.SUBTITLE_CHARS_PER_LINE,
                                        compose.SUBTITLE_LATIN_CHARS_PER_LINE) if line else []
            translation = write_text_lines(ep, sid, "tr", shot.get("translation") or "",
                                           compose.TRANSLATION_CHARS_PER_LINE,
                                           compose.SUBTITLE_LATIN_CHARS_PER_LINE) if line else []
            source = plan["source"] if plan["video"] else f"{plan['source']}＋{plan['camera']}"
            print(f"{sid}: {plan['duration']:.2f}s，{source}{spoken}", flush=True)
            cmd = compose.segment_command(
                ffmpeg, out=ep.segment(sid), duration=plan["duration"], video=plan["video"], still=plan["still"],
                camera=plan["camera"], voice=plan["voice"], voice_delay=VOICE_LEAD_S, subtitle_lines=subtitle,
                translation_lines=translation, font=font, fps=fps)
        run(cmd, check=True)
        segments.append(ep.segment(sid))
        total += plan["duration"]
    # The concat demuxer resolves entries against the list file's folder, and on Windows it got that
    # folder wrong for a path mixing both slash styles - so it runs inside segments/ with bare names.
    seg_dir = ep.sub("segments")
    with open(os.path.join(seg_dir, "concat.txt"), "w", encoding="utf-8") as f:
        f.write(compose.concat_list(segments, base_dir=seg_dir))
    out = ep.output()
    bgm = ep.data.get("bgm")
    if bgm:
        body = ep.sub("segments", "body.mkv")
        run(compose.concat_command(ffmpeg, "concat.txt", body, final=False), check=True, cwd=seg_dir)
        run(compose.bgm_command(ffmpeg, body, ep.resolve(bgm), out, total,
                                float(ep.data.get("bgm_volume", DEFAULT_BGM_VOLUME))), check=True)
    else:
        run(compose.concat_command(ffmpeg, "concat.txt", out), check=True, cwd=seg_dir)
    print(f"完成：{out}（{total:.1f} 秒，{len(segments)} 個鏡頭，{fps} fps）", flush=True)
    return out, total


def run_interpolate(ep, shots, *, force=False, submit=None, upload=None, server_up=client.is_server_running):
    """Double each moving shot's frame rate with RIFE on the local ComfyUI (MIT code and weights) and
    save the result as its enhanced clip, which assembly then prefers. The source is the final clip,
    else the low-res draft - never a preview, which is shorter than the shot."""
    submit = submit or client.submit_interpolation_rife
    upload = upload or client.upload_reference_image
    todo, missing = [], []
    for _index, shot in shots:
        if shot["type"] not in MOVING_TYPES:
            continue
        sid = shot["id"]
        if os.path.isfile(ep.enhanced(sid, "mp4")) and not force:
            print(f"{sid}: 已有加強版，略過（--force 重做）", flush=True)
            continue
        source = next((ep.motion(sid, m) for m in ("final", "draft") if os.path.isfile(ep.motion(sid, m))), None)
        if source:
            todo.append((sid, source))
        else:
            missing.append(sid)
    if missing:
        print(f"[注意] 這些鏡頭還沒有正式版或低解析度版影片，略過：{missing}", flush=True)
    if not todo:
        return 0
    if not server_up():
        raise gc.UsageError("ComfyUI 沒有在跑。RIFE 補幀在本機 ComfyUI 上執行，先啟動它")
    os.makedirs(ep.sub("enhance", "out"), exist_ok=True)
    staging = ep.sub("enhance", "staging")
    os.makedirs(staging, exist_ok=True)
    for sid, source in todo:
        # Uploaded under an episode-specific name: ComfyUI's input/ is shared, and the plain
        # "s02_draft.mp4" of two episodes would overwrite each other.
        staged = os.path.join(staging, f"{ep.name}_{os.path.basename(source)}")
        shutil.copy2(source, staged)
        print(f"{sid}: RIFE 補幀 {FPS}→{FPS * INTERP_MULTIPLIER} fps（來源 {os.path.basename(source)}）", flush=True)
        name = upload(staged)
        raw = submit(video_filename=name, filename_prefix=f"drama_{ep.name}_{sid}_rife",
                     multiplier=INTERP_MULTIPLIER, fps=FPS)
        shutil.move(raw, ep.enhanced(sid, "mp4"))
    print(f"完成 {len(todo)} 支；組裝時加 --fps {FPS * INTERP_MULTIPLIER} 才會保留補出來的影格", flush=True)
    return len(todo)


ENHANCE_README = """這些是 {name} 要交給外部工具加強的素材。

- in/ 裡的 .mp4：動態鏡頭，拿去補幀（例如 Topaz Video AI、RIFE、FILM）和放大（例如 Topaz、SeedVR2）
- in/ 裡的 .png：靜態鏡頭的關鍵幀，可以用圖片放大工具處理到 1080×1920 以上
- 處理完用「相同檔名」存到 out/（影片 .mp4、圖片 .png），影片長度不要改
- 補幀到 48 或 60 fps 的話，組裝時加 --fps 48（或 60），或在分鏡表寫 "fps": 48，否則會被降回 {fps} fps
- 再跑 drama.py assemble，out/ 裡有的檔案會優先使用
- 要營利時，確認加強工具的授權允許商業使用
"""


def run_export(ep, copy=shutil.copy2):
    """Copy each shot's best picture to enhance/in/<id>.mp4|png for an external enhancer and write a
    manifest. Clips: final, else low-res draft, else preview. Stills: the keyframe. Cards are drawn
    at assembly and have nothing to enhance."""
    in_dir, out_dir = ep.sub("enhance", "in"), ep.sub("enhance", "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    manifest, missing = [], []
    for shot in ep.shots:
        sid = shot["id"]
        if shot["type"] == CARD:
            continue
        if shot["type"] in MOVING_TYPES:
            options = [(ep.motion(sid, m), m) for m in ("final", "draft", "preview")] + [(ep.keyframe(sid), "keyframe")]
        else:
            options = [(ep.keyframe(sid), "keyframe")]
        found = next(((path, kind) for path, kind in options if os.path.isfile(path)), None)
        if not found:
            missing.append(sid)
            continue
        src, kind = found
        dst = os.path.join(in_dir, f"{sid}{os.path.splitext(src)[1]}")
        copy(src, dst)
        manifest.append({"id": sid, "from": kind, "file": os.path.basename(dst), "duration": shot["duration"]})
    with open(ep.sub("enhance", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"episode": ep.name, "fps": ep.fps, "items": manifest}, f, ensure_ascii=False, indent=2)
    with open(ep.sub("enhance", "README.txt"), "w", encoding="utf-8") as f:
        f.write(ENHANCE_README.format(name=ep.name, fps=ep.fps))
    for item in manifest:
        print(f"{item['id']}: {item['file']}（來源：{item['from']}）", flush=True)
    if missing:
        print(f"[注意] 這些鏡頭還沒有素材，沒有匯出：{missing}", flush=True)
    print(f"已匯出到 {in_dir}；加強後放到 {out_dir}（同檔名）", flush=True)
    return manifest


# --- plan / status ----------------------------------------------------------------------------


def print_plan(ep):
    kinds = {"still": "靜態", "motion": "動態", "dialogue": "對話", CARD: "字卡"}
    total = 0.0
    print(f"{ep.data.get('title', ep.name)}  |  checkpoint {ep.checkpoint}  |  tier {ep.tier}")
    for index, shot in enumerate(ep.shots):
        total += shot["duration"]
        who = shot.get("character") or "-"
        text = shot.get("phrase") if shot["type"] == CARD else shot.get("line")
        line = f"「{text}」" if text else ""
        print(f"  {shot['id']:<8} {kinds[shot['type']]}  {shot['duration']:>4.1f}s  {who:<8} "
              f"{shot.get('scene') or '-':<18} {line}")
    everything = select_shots(ep)
    n_prev, cost_prev = estimate(ep, everything, "preview")
    _n, cost_draft = estimate(ep, everything, "draft")
    n_final, cost_final = estimate(ep, everything, "final")
    print(f"共 {len(ep.shots)} 個鏡頭，約 {total:.1f} 秒（配音較長的鏡頭會自動延長），成片 {ep.fps} fps")
    print(f"雲端估計（未實測）：預覽 {n_prev} 支約 ${cost_prev:.2f}；低解析度完整版約 ${cost_draft:.2f}；"
          f"正式版 {n_final} 支約 ${cost_final:.2f}")
    status = voice_status(ep)
    if status:
        print("配音聲音：")
        for voice_id, who, state in status:
            print(f"  {voice_id or '（未設定）'}  ← {'、'.join(who)}：{state}")
    if ep.data.get("commercial") is True:
        print("這集標示 commercial：只允許可營利的元件（見 drama.py 開頭說明）")
    print(f"工作資料夾：{ep.work_dir}")


def print_status(ep):
    def mark(path):
        # Both symbols exist in cp950, the default Traditional Chinese console codepage; a check mark doesn't.
        return "●" if os.path.isfile(path) else "○"

    print(f"{'鏡頭':<8} 關鍵幀 配音 預覽 低解析 正式 增強")
    for shot in ep.shots:
        sid = shot["id"]
        voice = mark(ep.voice(sid)) if (shot.get("line") or "").strip() else " "
        moving = shot["type"] in MOVING_TYPES
        prev = mark(ep.motion(sid, "preview")) if moving else " "
        draft = mark(ep.motion(sid, "draft")) if moving else " "
        final = mark(ep.motion(sid)) if moving else " "
        if shot["type"] == CARD:
            keyframe, enhanced = "－", " "
        else:
            keyframe = "△" if os.path.isfile(ep.keyframe_draft_marker(sid)) else mark(ep.keyframe(sid))
            enhanced_path = ep.enhanced(sid, "mp4" if moving else "png")
            enhanced = mark(enhanced_path)
        print(f"{sid:<10} {keyframe:^5} {voice:^4} {prev:^4} {draft:^5} {final:^4} {enhanced:^4}")
    print("△＝草稿關鍵幀（跑 keyframes 不加 --draft 會重生）")
    print(f"成片：{ep.output() if os.path.isfile(ep.output()) else '（還沒組裝）'}")


# --- CLI --------------------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description="AI short drama: shot list -> keyframes -> voice -> motion -> 9:16 mp4")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_text):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("episode", help="episode file, e.g. training/episodes/ep01.json")
        return sp

    add("plan", "validate the shot list and print the shots and a cloud cost estimate")
    add("status", "which shots have a keyframe, voice, preview and final clip")
    kf = add("keyframes", "generate each shot's 9:16 keyframe with the local ComfyUI")
    kf.add_argument("--shots", help="comma-separated shot ids (default: all)")
    kf.add_argument("--force", action="store_true", help="regenerate shots that already have one")
    kf.add_argument("--draft", action="store_true",
                    help="quick low-res drafts without the face pass; a later run without --draft redoes them")
    kf.add_argument("--candidates", type=int, default=0,
                    help=f"render N alternative seeds per shot into keyframes/candidates/ (1-{MAX_CANDIDATES}); "
                         "the keyframe itself is untouched until `pick`")
    pk = add("pick", "promote a candidate keyframe and record its seed in the episode file")
    pk.add_argument("--shot", required=True)
    pk.add_argument("--seed", type=int, required=True)
    vo = add("voice", "speak every dialogue line with CosyVoice3 (stop ComfyUI first)")
    vo.add_argument("--shots")
    vo.add_argument("--force", action="store_true")
    vo.add_argument("--ignore-comfyui", action="store_true", help="run even though ComfyUI is up")
    vo.add_argument("--allow-demo", action="store_true",
                    help="internal tests only: use CosyVoice's demo voice for characters without a consented voice")
    mo = add("motion", "send motion/dialogue shots to Wan 2.2 on the cloud as one batch")
    mode = mo.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="cheap 480x832 drafts (about $0.015 each, estimated)")
    mode.add_argument("--draft", action="store_true",
                      help="480x832 at full length and steps, to hand to an external upscaler/interpolator")
    mode.add_argument("--final", action="store_true", help="704x1280 renders for the finished episode")
    mo.add_argument("--shots")
    mo.add_argument("--force", action="store_true")
    mo.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    mo.add_argument("--provider", choices=("runpod",), default="runpod")
    mo.add_argument("--timeout", type=int, help="max seconds to wait per job before cancelling")
    asm = add("assemble", "cut every shot together with subtitles, voice and music into one 1080x1920 mp4")
    asm.add_argument("--allow-preview", action="store_true",
                     help="use a preview clip where there is no final one (rough cut)")
    asm.add_argument("--fps", type=int, choices=OUTPUT_FPS_CHOICES,
                     help="output frame rate (default: the episode's fps, else 24); 48/60 keeps interpolated clips smooth")
    add("export", "copy clips and stills to enhance/in/ for an external upscaler or frame interpolator")
    it = add("interpolate", "double moving shots' frame rate with RIFE on the local ComfyUI (MIT) into enhance/out/")
    it.add_argument("--shots")
    it.add_argument("--force", action="store_true")
    return p


def _need_ffmpeg():
    ffmpeg = compose.find_ffmpeg()
    if not ffmpeg:
        raise gc.UsageError("找不到 ffmpeg：放一個 ffmpeg.exe 到 SadTalker 資料夾、加到 PATH，或設定 DRAMA_FFMPEG")
    return ffmpeg


def main(argv=None):
    args = build_parser().parse_args(argv)
    ep = load_episode(args.episode)
    require_valid(ep)
    if args.cmd == "plan":
        print_plan(ep)
        return 0
    if args.cmd == "status":
        print_status(ep)
        return 0
    if args.cmd == "keyframes":
        run_keyframes(ep, select_shots(ep, args.shots), force=args.force, draft=args.draft,
                      candidates=args.candidates, ffmpeg=_need_ffmpeg())
        return 0
    if args.cmd == "voice":
        run_voice(ep, select_shots(ep, args.shots), force=args.force, ignore_comfyui=args.ignore_comfyui,
                  allow_demo=args.allow_demo)
        return 0
    if args.cmd == "motion":
        mode = "preview" if args.preview else "draft" if args.draft else "final"
        results = run_motion(ep, select_shots(ep, args.shots, types=MOVING_TYPES), mode=mode,
                             force=args.force, yes=args.yes, provider=args.provider, timeout=args.timeout,
                             ffmpeg=_need_ffmpeg())
        return 1 if any(not isinstance(o, cloud_video.CloudResult) for _s, o in results) else 0
    if args.cmd == "assemble":
        run_assemble(ep, allow_preview=args.allow_preview, ffmpeg=_need_ffmpeg(), font=compose.find_font(),
                     fps=args.fps)
        return 0
    if args.cmd == "pick":
        run_pick(ep, args.shot, args.seed)
        return 0
    if args.cmd == "export":
        run_export(ep)
        return 0
    if args.cmd == "interpolate":
        run_interpolate(ep, select_shots(ep, args.shots), force=args.force)
        return 0
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except cloud_video.CloudJobFailed as exc:
        raise SystemExit(f"雲端工作失敗：{exc}")
    except (gc.UsageError, client.ComfyUIUnavailable) as exc:
        raise SystemExit(str(exc))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"外部程式失敗（exit {exc.returncode}）：{' '.join(map(str, exc.cmd[:3]))} ...")
