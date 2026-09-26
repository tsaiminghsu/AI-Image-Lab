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

Where each stage runs follows the cost report: keyframes and voice are free locally, only motion
goes to the cloud, and motion is drafted cheaply before the full render. The safety guarantees are
the existing ones, not new code: keyframes go through gc.plan_picker + gc.gen_custom, and motion
jobs go through cloud_video to the RunPod worker, which composes the tier + age safety negatives
server-side with the cfg floor. Lip-sync is not part of this stage; dialogue shots get the voice
line and a subtitle over Wan's motion.
"""

import argparse
import json
import os
import re
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

EPISODE_VERSION = 1
SHOT_TYPES = ("still", "motion", "dialogue")
MOVING_TYPES = ("motion", "dialogue")
FRAMINGS = {
    "close": "close-up shot, head and shoulders",
    "medium": "medium shot from the waist up",
    "wide": "full body wide shot",
}
DURATION_RANGE = (1.5, 5.0)     # Wan's 121-frame ceiling is 5.04 s at 24 fps
MAX_LINE_CHARS = 120
ID_RE = re.compile(r"^[a-z0-9_]{1,32}$")
EPISODE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FPS = compose.FPS
KEYFRAME_SIZE = (768, 1344)     # SDXL bucket closest to 9:16, used when no pose skeleton sets the canvas
WAN_FINAL_SIZE = (704, 1280)
WAN_PREVIEW = cloud_video.WAN_PREVIEW_PORTRAIT
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

    def motion(self, shot_id, preview=False):
        return self.sub("motion", f"{shot_id}_preview.mp4" if preview else f"{shot_id}.mp4")

    def subtitle(self, shot_id):
        return self.sub("subs", f"{shot_id}.txt")

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


def _has_cjk(text):
    return any("㐀" <= ch <= "鿿" or "豈" <= ch <= "﫿" for ch in text or "")


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

    shots = d.get("shots")
    if not isinstance(shots, list) or not shots:
        errors.append("shots 至少要有一個鏡頭")
        return errors, warnings
    seen = set()
    for index, shot in enumerate(shots):
        label = f"鏡頭 {index + 1}"
        if not isinstance(shot, dict):
            errors.append(f"{label} 要是一個物件")
            continue
        shot_id = shot.get("id")
        if not isinstance(shot_id, str) or not ID_RE.match(shot_id):
            errors.append(f"{label} 的 id 只能用小寫英數字和 _（1–32 字），會變成檔名")
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
        framing = shot.get("framing", "medium")
        if framing not in FRAMINGS:
            errors.append(f"{label} 的 framing 只能是 {list(FRAMINGS)}")
        camera = shot.get("camera")
        if camera is not None and camera not in compose.CAMERA_MOVES:
            errors.append(f"{label} 的 camera 只能是 {list(compose.CAMERA_MOVES)}")
        character = shot.get("character")
        if character:
            if character not in gc.CHARACTERS:
                errors.append(f"{label} 的角色「{character}」不在角色表裡")
            elif ckpt in client.CHECKPOINTS and ckpt not in client.SD15_CHECKPOINTS \
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
        line = shot.get("line")
        if kind == "dialogue" and not (line or "").strip():
            errors.append(f"{label} 是對話鏡頭，要有 line（台詞）")
        if line and len(line) > MAX_LINE_CHARS:
            errors.append(f"{label} 的台詞超過 {MAX_LINE_CHARS} 字，請拆成兩個鏡頭")
        voice_id = shot.get("voice")
        if voice_id is not None and (not isinstance(voice_id, str) or not ID_RE.match(voice_id)):
            errors.append(f"{label} 的 voice 只能用小寫英數字和 _")
    return errors, warnings


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


def estimate(ep, shots, preview):
    """(job count, estimated dollars) for sending these shots' motion in one batch."""
    moving = [(i, s) for i, s in shots if s["type"] in MOVING_TYPES]
    if not moving:
        return 0, 0.0
    cold = RUNPOD_PER_HOUR / 60 * COLD_START_MINUTES
    if preview:
        return len(moving), cold + PREVIEW_COST * len(moving)
    return len(moving), cold + sum(final_cost(frames_for(s["duration"])) for _i, s in moving)


# --- keyframes --------------------------------------------------------------------------------


def keyframe_request(ep, index, shot, translate=None):
    """Everything gen_custom needs for one shot's keyframe, resolved through plan_picker exactly like
    the picker tab: SDXL/Pony get the character as a FaceID anchor and the pose as a ControlNet
    skeleton, and the scene library supplies the setting."""
    prompt = (shot.get("prompt") or "").strip()
    if _has_cjk(prompt):
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
    width, height = (None, None) if plan.pose_name else KEYFRAME_SIZE
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


def run_keyframes(ep, shots, *, force=False, ffmpeg, translate=None, gen=None, run=subprocess.run,
                  server_up=client.is_server_running):
    gen = gen or gc.gen_custom
    if not server_up():
        raise gc.UsageError("ComfyUI 沒有在跑。先啟動它（或開 GUI，會自動啟動），再生關鍵幀")
    out_dir = ep.sub("keyframes")
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for index, shot in shots:
        final = ep.keyframe(shot["id"])
        if os.path.isfile(final) and not force:
            print(f"{shot['id']}: 已有關鍵幀，略過（--force 重生）", flush=True)
            continue
        req = keyframe_request(ep, index, shot, translate=translate)
        for notice in req["notices"]:
            print(f"[注意] {shot['id']}: {notice}", flush=True)
        print(f"{shot['id']}: 生成關鍵幀（seed {req['seed']}）", flush=True)
        stem = f"{shot['id']}_raw"
        gen(req["prompt"], "", ep.tier, req["trigger"], req["anchor_path"], out_dir, req["seed"], stem,
            client.IP_ADAPTER_WEIGHT, width=req["width"], height=req["height"],
            style_positive=gc.REALISTIC_STYLE, style_negative=gc.REALISTIC_NEGATIVE,
            checkpoint=ep.checkpoint, lora_strength=0.0, hq=True, pose_name=req["pose_name"])
        raw = os.path.join(out_dir, f"{stem}.png")
        run(compose.normalise_command(ffmpeg, raw, final), check=True)
        made += 1
    return made


# --- voice ------------------------------------------------------------------------------------


def resolve_voice(ep, shot):
    """(prompt_wav, prompt_transcript, is_demo) for a shot's speaker.

    A voice lives in training/voices/<id>/voice.json: {"prompt_wav": "prompt.wav", "prompt_text":
    "exact transcript of that recording"}. The recording must be one its speaker agreed to have
    cloned. Without a configured voice the CosyVoice demo prompt is used, which is only for testing."""
    voice_id = shot.get("voice") or (ep.data.get("voices") or {}).get(shot.get("character") or "")
    if not voice_id:
        return os.path.join(COSYVOICE_DIR, DEMO_VOICE_WAV), DEMO_VOICE_TEXT, True
    config = os.path.join(VOICES_DIR, voice_id, "voice.json")
    if not os.path.isfile(config):
        raise gc.UsageError(f"鏡頭 {shot['id']}：找不到聲音設定 {config}")
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    wav = os.path.join(VOICES_DIR, voice_id, cfg.get("prompt_wav", "prompt.wav"))
    text = (cfg.get("prompt_text") or "").strip()
    if not os.path.isfile(wav) or not text:
        raise gc.UsageError(f"聲音「{voice_id}」要有 prompt_wav 錄音檔和它的逐字稿 prompt_text")
    return wav, text, False


def voice_items(ep, shots):
    items, demo_used = [], False
    for _index, shot in shots:
        line = (shot.get("line") or "").strip()
        if not line:
            continue
        wav, transcript, is_demo = resolve_voice(ep, shot)
        demo_used = demo_used or is_demo
        emotion = (shot.get("emotion") or "").strip()
        items.append({
            "id": shot["id"],
            "text": line,
            "out": ep.voice(shot["id"]),
            "prompt_wav": wav,
            "prompt_text": COSYVOICE_PROMPT_PREFIX + transcript,
            "instruct": f"You are a helpful assistant. 请用{emotion}的语气说这句话。<|endofprompt|>" if emotion else None,
        })
    return items, demo_used


def cosyvoice_python():
    for rel in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        path = os.path.join(COSYVOICE_DIR, *rel)
        if os.path.isfile(path):
            return path
    return None


def run_voice(ep, shots, *, force=False, ignore_comfyui=False, run=subprocess.run,
              server_up=client.is_server_running):
    python = cosyvoice_python()
    if not python:
        raise gc.UsageError(f"找不到 CosyVoice 的 venv（{COSYVOICE_DIR}\\.venv），配音需要它")
    if server_up() and not ignore_comfyui:
        raise gc.UsageError("ComfyUI 正在跑，會跟 CosyVoice 搶 8 GB 顯卡。先跑 training\\stop_comfyui.ps1，"
                            "或加 --ignore-comfyui")
    items, demo_used = voice_items(ep, shots)
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


def motion_specs(ep, shots, preview):
    """One CloudJobSpec per motion/dialogue shot. The first frame is the shot's keyframe at Wan's
    size; the prompt is the raw motion text plus the tier - the worker adds the character's
    description and the safety negatives."""
    specs = []
    for index, shot in shots:
        if shot["type"] not in MOVING_TYPES:
            continue
        frames = frames_for(shot["duration"])
        if preview:
            params = dict(WAN_PREVIEW)
            params["frames"] = min(frames, WAN_PREVIEW["frames"])
        else:
            params = {"width": WAN_FINAL_SIZE[0], "height": WAN_FINAL_SIZE[1], "frames": frames,
                      "steps": client.WAN_STEPS}
        params.update(fps=FPS, cfg=client.WAN_CFG)
        gc.check_wan_params(params["width"], params["height"], params["frames"], params["fps"], params["steps"],
                            params["cfg"], client.WAN_DEFAULT_MODEL)
        stem = os.path.splitext(os.path.basename(ep.motion(shot["id"], preview)))[0]
        specs.append(cloud_video.CloudJobSpec(
            stem=stem, prompt=(shot.get("motion") or shot.get("prompt") or "").strip(),
            image_path=ep.wan_input(shot["id"]), trigger=shot.get("character") or None,
            seed=ep.shot_seed(index, shot), params=params, tier=ep.tier,
        ))
    return specs


def run_motion(ep, shots, *, preview, force=False, yes=False, provider="runpod", timeout=None, ffmpeg,
               run=subprocess.run, run_batch=None, confirm=input, stdin_isatty=None):
    run_batch = run_batch or cloud_video.run_cloud_batch
    moving = [(i, s) for i, s in shots if s["type"] in MOVING_TYPES]
    if not force:
        moving = [(i, s) for i, s in moving if not os.path.isfile(ep.motion(s["id"], preview))]
    if not moving:
        print("沒有需要送出的動態鏡頭（都已經有了，--force 重送）", flush=True)
        return []
    missing = [s["id"] for _i, s in moving if not os.path.isfile(ep.keyframe(s["id"]))]
    if missing:
        raise gc.UsageError(f"這些鏡頭還沒有關鍵幀，先跑 keyframes：{missing}")
    specs = motion_specs(ep, moving, preview)
    count, dollars = estimate(ep, moving, preview)
    mode = "預覽（480×832、12 步）" if preview else "正式版（704×1280）"
    print(f"將送出 {count} 支{mode}到 {provider}，估計約 ${dollars:.2f}（未實測）", flush=True)
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
    log_motion(ep, results, preview)
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


def log_motion(ep, results, preview):
    """Append each job to motion/jobs.json, with a billed-cost floor from the execution seconds
    (the cold start and idle seconds are billed too, so the real figure is higher)."""
    path = ep.sub("motion", "jobs.json")
    entries = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    for spec, outcome in results:
        entry = {"stem": spec.stem, "preview": preview, "seed": spec.seed, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
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


def shot_plan(ep, shot, allow_preview=False):
    """What the assembler will use for one shot: picture source, voice, duration."""
    has_line = bool((shot.get("line") or "").strip())
    voice = ep.voice(shot["id"]) if has_line and os.path.isfile(ep.voice(shot["id"])) else None
    duration = float(shot["duration"])
    if voice:
        duration = max(duration, VOICE_LEAD_S + wav_seconds(voice) + VOICE_TAIL_S)
    duration = compose.frame_exact(duration)
    video = None
    if shot["type"] in MOVING_TYPES:
        if os.path.isfile(ep.motion(shot["id"])):
            video = ep.motion(shot["id"])
        elif allow_preview and os.path.isfile(ep.motion(shot["id"], preview=True)):
            video = ep.motion(shot["id"], preview=True)
    still = None if video else ep.keyframe(shot["id"])
    return {"video": video, "still": still, "voice": voice, "duration": round(duration, 6),
            "camera": shot.get("camera") or "push_in"}


def run_assemble(ep, *, allow_preview=False, ffmpeg, font, run=subprocess.run):
    plans = []
    missing = []
    for shot in ep.shots:
        plan = shot_plan(ep, shot, allow_preview)
        if plan["still"] and not os.path.isfile(plan["still"]):
            missing.append(shot["id"])
        plans.append((shot, plan))
    if missing:
        raise gc.UsageError(f"這些鏡頭沒有影片也沒有關鍵幀，無法組裝：{missing}")
    if any((s.get("line") or "").strip() for s in ep.shots) and not font:
        raise gc.UsageError("找不到中文字型（微軟正黑體），設定 DRAMA_FONT 指向一個 .ttf/.ttc")
    for sub in ("segments", "subs"):
        os.makedirs(ep.sub(sub), exist_ok=True)
    segments, total = [], 0.0
    for shot, plan in plans:
        subtitle = None
        line = (shot.get("line") or "").strip()
        if line:
            subtitle = ep.subtitle(shot["id"])
            with open(subtitle, "w", encoding="utf-8") as f:
                f.write(compose.wrap_subtitle(line))
        source = "影片" if plan["video"] else f"靜態圖＋{plan['camera']}"
        print(f"{shot['id']}: {plan['duration']:.2f}s，{source}{'，有配音' if plan['voice'] else ''}", flush=True)
        run(compose.segment_command(
            ffmpeg, out=ep.segment(shot["id"]), duration=plan["duration"], video=plan["video"], still=plan["still"],
            camera=plan["camera"], voice=plan["voice"], voice_delay=VOICE_LEAD_S, subtitle_file=subtitle,
            font=font), check=True)
        segments.append(ep.segment(shot["id"]))
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
    print(f"完成：{out}（{total:.1f} 秒，{len(segments)} 個鏡頭）", flush=True)
    return out, total


# --- plan / status ----------------------------------------------------------------------------


def print_plan(ep):
    kinds = {"still": "靜態", "motion": "動態", "dialogue": "對話"}
    total = 0.0
    print(f"{ep.data.get('title', ep.name)}  |  checkpoint {ep.checkpoint}  |  tier {ep.tier}")
    for index, shot in enumerate(ep.shots):
        total += shot["duration"]
        who = shot.get("character") or "-"
        line = f"「{shot['line']}」" if shot.get("line") else ""
        print(f"  {shot['id']:<8} {kinds[shot['type']]}  {shot['duration']:>4.1f}s  {who:<8} "
              f"{shot.get('scene') or '-':<18} {line}")
    everything = select_shots(ep)
    n_prev, cost_prev = estimate(ep, everything, preview=True)
    n_final, cost_final = estimate(ep, everything, preview=False)
    print(f"共 {len(ep.shots)} 個鏡頭，約 {total:.1f} 秒（配音較長的鏡頭會自動延長）")
    print(f"雲端估計（未實測）：預覽 {n_prev} 支約 ${cost_prev:.2f}；正式版 {n_final} 支約 ${cost_final:.2f}")
    print(f"工作資料夾：{ep.work_dir}")


def print_status(ep):
    def mark(path):
        # Both symbols exist in cp950, the default Traditional Chinese console codepage; a check mark doesn't.
        return "●" if os.path.isfile(path) else "○"

    print(f"{'鏡頭':<8} 關鍵幀 配音 預覽 正式")
    for shot in ep.shots:
        sid = shot["id"]
        voice = mark(ep.voice(sid)) if (shot.get("line") or "").strip() else " "
        moving = shot["type"] in MOVING_TYPES
        prev = mark(ep.motion(sid, True)) if moving else " "
        final = mark(ep.motion(sid)) if moving else " "
        print(f"{sid:<10} {mark(ep.keyframe(sid)):^5} {voice:^4} {prev:^4} {final:^4}")
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
    vo = add("voice", "speak every dialogue line with CosyVoice3 (stop ComfyUI first)")
    vo.add_argument("--shots")
    vo.add_argument("--force", action="store_true")
    vo.add_argument("--ignore-comfyui", action="store_true", help="run even though ComfyUI is up")
    mo = add("motion", "send motion/dialogue shots to Wan 2.2 on the cloud as one batch")
    mode = mo.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="cheap 480x832 drafts (about $0.015 each, estimated)")
    mode.add_argument("--final", action="store_true", help="704x1280 renders for the finished episode")
    mo.add_argument("--shots")
    mo.add_argument("--force", action="store_true")
    mo.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    mo.add_argument("--provider", choices=("runpod",), default="runpod")
    mo.add_argument("--timeout", type=int, help="max seconds to wait per job before cancelling")
    asm = add("assemble", "cut every shot together with subtitles, voice and music into one 1080x1920 mp4")
    asm.add_argument("--allow-preview", action="store_true",
                     help="use a preview clip where there is no final one (rough cut)")
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
        run_keyframes(ep, select_shots(ep, args.shots), force=args.force, ffmpeg=_need_ffmpeg())
        return 0
    if args.cmd == "voice":
        run_voice(ep, select_shots(ep, args.shots), force=args.force, ignore_comfyui=args.ignore_comfyui)
        return 0
    if args.cmd == "motion":
        results = run_motion(ep, select_shots(ep, args.shots, types=MOVING_TYPES), preview=args.preview,
                             force=args.force, yes=args.yes, provider=args.provider, timeout=args.timeout,
                             ffmpeg=_need_ffmpeg())
        return 1 if any(not isinstance(o, cloud_video.CloudResult) for _s, o in results) else 0
    if args.cmd == "assemble":
        run_assemble(ep, allow_preview=args.allow_preview, ffmpeg=_need_ffmpeg(), font=compose.find_font())
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
