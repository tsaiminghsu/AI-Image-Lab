"""Local Gradio GUI for generate_character.py's custom-prompt mode.

Runs entirely on localhost (server_name="127.0.0.1", share=False) - this
handles suggestive-tier content and must never be exposed to the public
internet via Gradio's share tunnel.

ComfyUI no longer needs to be pre-started separately: _ensure_comfyui()
launches it on the first actual generate click if it isn't already up, and
_shutdown_comfyui() stops it again on exit (only if this GUI started it) -
so leaving the GUI open unused doesn't leave the ~2GB ComfyUI process
running for no reason.
"""

import atexit
import functools
import glob
import os
import threading

import gradio as gr

import caption_image
import cloud_video
import comfyui_client as client
import generate_character as gc
import pony_tags
import pose_skeletons
import scene_library
import talking_head
import translate_prompt

_comfyui_process = None  # only set when this GUI auto-started ComfyUI itself
_comfyui_start_lock = threading.Lock()  # see _ensure_comfyui


def _show_usage_errors(fn):
    """Turn generate_character's caller-error signal into a Gradio toast.

    generate_character raises gc.UsageError for invalid arguments (get_character on an unknown
    trigger, gen_custom's flag-combination checks, ...). The handlers below also pre-check most
    of those cases in Traditional Chinese, but nothing enforces that the pre-checks stay
    exhaustive and gen_custom keeps growing flags, so this catches whatever slips through and
    shows the message instead of failing the queue task.

    client.ComfyUIUnavailable is caught for the same reason: "ComfyUI stopped answering" is
    one readable line, not a requests/urllib3 traceback in the terminal.

    SystemExit is still caught alongside them: the CLI-only scripts this GUI reaches (SadTalker
    via talking_head, pose_skeletons) may still use it, and it derives from BaseException, so
    Gradio's Exception-only handling would let it escape and kill the worker task silently.

    cloud_video.CloudJobFailed covers a remote job that failed, timed out or was cancelled.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (gc.UsageError, client.ComfyUIUnavailable, cloud_video.CloudJobFailed, SystemExit) as exc:
            raise gr.Error(str(exc)) from exc

    return wrapper


def _step_reporter(progress):
    """Feed ComfyUI's real step counter into a Gradio progress tracker.

    Gradio's own bar is an ETA extrapolated from how long previous runs of the SAME event
    took, so it is badly wrong here: one click on this button is a ~30s plain txt2img and the
    next is a ~150s hires + FaceDetailer job, depending only on the toggles. Steps are the
    honest unit, so the description carries ComfyUI's actual counter and the bar is scoped to
    the current stage (whose total ComfyUI does tell us) rather than pretending to know how
    much of the whole job is left - the job's stages have different, unknowable step totals
    (FaceDetailer's depends on how many faces and hands it finds).
    """
    def report(stage, current, total):
        if total is None:
            # A cloud job reports status text (queued, cold start, remote step count) rather than a
            # local step counter, so there is no honest bar to draw - show the text only.
            progress(None, desc=stage)
            return
        progress((current, total), desc=f"{stage} {current}/{total} 步")
    return report


def _ensure_comfyui():
    """Start ComfyUI on first actual generation instead of requiring it
    pre-started just to open the GUI - keeps the ~2GB process off when the
    GUI is only sitting open unused.

    Serialised: the is-it-running check and the spawn are separate steps, and
    different tabs' buttons are different Gradio listeners running on their own
    worker threads (concurrency_limit is per listener). Two first-clicks landing
    together would both see "not running" and spawn a second ComfyUI that then
    fails to bind port 8188 - and whose ~2GB stays around unowned, since
    _shutdown_comfyui only tracks the one process in _comfyui_process. The lock
    is held across client.start_server(), which blocks until the server answers,
    so the second caller re-checks afterwards and finds it up."""
    global _comfyui_process
    with _comfyui_start_lock:
        if client.is_server_running():
            return
        if _comfyui_process is not None and _comfyui_process.poll() is None:
            return  # already starting from a previous click
        gr.Info("ComfyUI 尚未啟動，正在自動啟動（第一次可能要等幾秒）...")
        _comfyui_process = client.start_server()


@atexit.register
def _shutdown_comfyui():
    # Only kill the process this GUI started itself - never touch a ComfyUI
    # the user is running separately in its own terminal.
    if _comfyui_process is not None and _comfyui_process.poll() is None:
        _comfyui_process.terminate()


NO_CHARACTER = "(無 - 純文字生圖)"
CHARACTER_CHOICES = [NO_CHARACTER] + sorted(gc.CHARACTERS)
POSE_NONE = "(無 - 不用骨架庫)"


def _pose_preview(pose_library):
    """Show the selected library skeleton so the user sees what pose they picked."""
    if not pose_library or pose_library == POSE_NONE:
        return None
    return pose_skeletons.resolve(pose_library)


def list_anchors(character):
    if not character or character == NO_CHARACTER:
        return []
    pattern = os.path.join(os.path.dirname(__file__), "reference_candidates", character, "*.png")
    return sorted(glob.glob(pattern))


def refresh_anchors(character):
    choices = list_anchors(character)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value), value


def caption_uploaded_image(image_path):
    if not image_path:
        raise gr.Error("請先上傳圖片")
    caption = caption_image.caption_image(image_path)
    return caption, caption


def tag_uploaded_image(image_path):
    """Booru tags instead of prose - the format Pony-family checkpoints want.

    Needs ComfyUI running (the tagger is a custom node inside it), unlike the BLIP captioner
    which runs in this process, so start it the same way a generation would.
    """
    if not image_path:
        raise gr.Error("請先上傳圖片")
    _ensure_comfyui()
    tags = client.tag_image(image_path)
    return tags, tags


def translate_prompt_to_english(prompt_text):
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("請先輸入 prompt")
    try:
        return translate_prompt.translate_to_english(prompt_text)
    except RuntimeError as exc:
        raise gr.Error(f"翻譯失敗（{exc}）——可以稍後再試一次，或直接自己改打英文") from exc


def reset_resolution_for_sd15(checkpoint_choice):
    checkpoint = None if checkpoint_choice == CHECKPOINT_DEFAULT else checkpoint_choice
    if checkpoint in client.SD15_CHECKPOINTS:
        return gr.update(value=RESOLUTION_AUTO)
    return gr.update()


RESOLUTION_AUTO = "自動（有姿勢參考圖時用直式，否則正方形）"
RESOLUTION_SQUARE = "正方形 1024×1024"
RESOLUTION_PORTRAIT = "直式 832×1216（全身/背面照建議）"
RESOLUTION_LANDSCAPE = "橫式 1216×832（風景/多人/環境為主的畫面建議）"
RESOLUTION_CHOICES = [RESOLUTION_AUTO, RESOLUTION_SQUARE, RESOLUTION_PORTRAIT, RESOLUTION_LANDSCAPE]


FACEDETAILER_BACKEND_YOLO = "YOLO 矩形框（預設）"
FACEDETAILER_BACKEND_MEDIAPIPE = "MediaPipe 臉部網格（貼合臉型輪廓）"
FACEDETAILER_BACKEND_CHOICES = [FACEDETAILER_BACKEND_YOLO, FACEDETAILER_BACKEND_MEDIAPIPE]


CHECKPOINT_DEFAULT = "cyberrealistic_pony (預設 - Pony 系寫實)"
CHECKPOINT_CHOICES = ([CHECKPOINT_DEFAULT] + sorted(k for k in client.CHECKPOINTS if k != "cyberrealistic_pony")
                      + sorted(client.ZIMAGE_MODELS))

ANIMATEDIFF_CHECKPOINT_DEFAULT = "realistic_vision (預設)"
ANIMATEDIFF_CHECKPOINT_CHOICES = [ANIMATEDIFF_CHECKPOINT_DEFAULT] + sorted(
    k for k in client.ANIMATEDIFF_CHECKPOINTS if k != "realistic_vision"
)
UPSCALE_OFF = "不放大（維持採樣尺寸）"
UPSCALE_CHOICES = [UPSCALE_OFF] + [f"長邊 {n}px" for n in client.ANIMATEDIFF_UPSCALE_CHOICES if n]
TALK_ENHANCER_NONE = "(無)"

MOTION_LORA_NONE = "(無 - 純 prompt 描述動作)"
MOTION_LORA_CHOICES = [MOTION_LORA_NONE] + sorted(client.ANIMATEDIFF_MOTION_LORAS)


def picker_characters():
    """(trigger, anchor_path, caption) for every character that has an identity anchor.

    Characters with no anchor_seed*.png yet are left out of the gallery entirely rather than
    shown as a broken tile: on SDXL they are the one pick that cannot be honoured at all, and a
    tile you can click but that silently does less than the others is worse than no tile.
    """
    rows = []
    for trigger in sorted(gc.CHARACTERS):
        anchor = gc.picker_anchor_path(trigger)
        if not anchor:
            continue
        profile = gc.CHARACTERS[trigger]
        gender = "女性" if profile["gender"] == "woman" else "男性"
        rows.append((trigger, anchor, f"{trigger}｜{profile['age']} 歲{gender}"))
    return rows


def picker_pose_items():
    """(slug, skeleton_png, caption) for the pose gallery. The caption is the library's own tag
    with underscores relaxed - the 12 bootstrapped entries store it as their slug."""
    items = []
    for slug in pose_skeletons.list_names():
        meta = pose_skeletons.load_meta(slug)
        items.append((slug, pose_skeletons.resolve(slug), str(meta.get("tag") or slug).replace("_", " ")))
    return items


def picker_scene_items(tier):
    return [(s["slug"], scene_library.thumb_path(s["slug"]), s["name_zh"]) for s in scene_library.list_scenes(tier)]


def _gallery_value(items):
    return [(path, caption) for _slug, path, caption in items]


def _variant_label(row):
    full = f"完整版 {row['full_gb']} GB" if row["full_present"] else "完整版（找不到檔案）"
    quant = f"量化版 {row['quant_gb']} GB" if row["quant_present"] else "量化版（還沒轉出）"
    extra = f"｜不支援量化：{row['unsupported']}" if row["unsupported"] else ""
    return f"{row['key']}：{full}｜{quant}{extra}"


def _variant_table():
    lines = ["| 模型 | 目前選擇 | 完整版 | 量化版 fp8 |", "|---|---|---|---|"]
    for r in client.variant_status():
        full = f"{r['full_gb']} GB" if r["full_present"] else "找不到"
        quant = f"{r['quant_gb']} GB" if r["quant_present"] else "還沒轉出"
        lines.append(f"| {r['key']} | {'量化版' if r['chosen'] == 'quant' else '完整版'} | {full} | {quant} |")
    return "\n".join(lines)


def save_variant_choices(*choices):
    client.save_variant_settings(models=dict(zip(client.QUANT_MODEL_KEYS, choices)))
    gr.Info("已儲存模型版本設定，下一張開始生效")
    for r in client.variant_status():
        if r["chosen"] == "quant" and not r["quant_present"]:
            gr.Warning(f"{r['key']} 還沒有量化檔，會先用完整版。建立量化檔：{client.convert_command(r['key'])}")
    return _variant_table()


def refresh_variant_status():
    rows = client.variant_status()
    return [gr.update(label=_variant_label(r), value=r["chosen"]) for r in rows] + [_variant_table()]


# --- Pony keyword picker (自訂生圖 tab) -----------------------------------------------------
# One CheckboxGroup per selectable category, built by a comprehension, so their count is not
# statically visible and Gradio binds them positionally - hence the *args handlers below and
# this list, the single place that says which argument is which category. The groups are built
# for ALL categories once: Gradio cannot grow or shrink a component list afterwards, so a tier
# or checkpoint change hides/repopulates them instead (pony_tag_refresh). The widest combination
# (suggestive tier, Pony checkpoint) is what "all" means, and deriving the slug list from the
# same call that builds the components is what keeps the two in the same order.
PONY_TAG_INITIAL = [c for c in pony_tags.categories("suggestive") if c["selectable"]]
PONY_TAG_SLUGS = [c["slug"] for c in PONY_TAG_INITIAL]
PONY_TAG_PREVIEW_HINT = "勾選上面的關鍵字，這裡會顯示實際會被加進去的字（品質分數標籤不在這裡，它一律自動加）。"
# Read out of the library rather than restated here, so the tab can never claim something the
# quality category does not actually do.
PONY_TAG_QUALITY_NOTE = next(c for c in pony_tags.CATEGORIES if c["slug"] == "quality")["note_zh"]


def _pony_tag_checkpoint(checkpoint_choice):
    """The dropdown's display label -> a CHECKPOINTS key (or None for the pipeline default),
    the same conversion reset_resolution_for_sd15 and update_picker_preview do."""
    return None if checkpoint_choice == CHECKPOINT_DEFAULT else checkpoint_choice


def _pony_tag_split(args):
    """Splits a variadic handler's arguments into ({slug: picked}, trailing fields)."""
    n = len(PONY_TAG_SLUGS)
    return dict(zip(PONY_TAG_SLUGS, args[:n])), args[n:]


def _append_terms(existing, addition):
    """Appends comma-separated terms without repeating ones already there - the button is meant
    to survive being pressed twice, and to add to what the user typed rather than replace it."""
    terms = [t.strip() for t in (existing or "").split(",") if t.strip()]
    for term in [t.strip() for t in (addition or "").split(",") if t.strip()]:
        if term not in terms:
            terms.append(term)
    return ", ".join(terms)


def pony_tag_preview(*args):
    selected, (tier, checkpoint_choice) = _pony_tag_split(args)
    body, extra_negative = pony_tags.compose(
        selected, tier=tier, checkpoint=_pony_tag_checkpoint(checkpoint_choice))
    lines = []
    if body:
        lines.append(f"Prompt：{body}")
    if extra_negative:
        lines.append(f"額外負面詞：{extra_negative}")
    return "\n".join(lines) if lines else PONY_TAG_PREVIEW_HINT


def pony_tag_append(*args):
    selected, (tier, checkpoint_choice, prompt_text, negative_text) = _pony_tag_split(args)
    body, extra_negative = pony_tags.compose(
        selected, tier=tier, checkpoint=_pony_tag_checkpoint(checkpoint_choice))
    if not body and not extra_negative:
        gr.Warning("還沒有勾選任何關鍵字")
        return gr.update(), gr.update()
    return _append_terms(prompt_text, body), _append_terms(negative_text, extra_negative)


def pony_tag_refresh(*args):
    """Repopulates the groups after a tier or checkpoint change. Picks that the new combination
    no longer offers are dropped, for the reason refresh_picker_scenes drops its selection: a
    suggestive-only choice that stays ticked while invisible would still reach the generator."""
    selected, (tier, checkpoint_choice) = _pony_tag_split(args)
    available = {c["slug"]: c for c in pony_tags.categories(tier, _pony_tag_checkpoint(checkpoint_choice))}
    updates = []
    for slug in PONY_TAG_SLUGS:
        cat = available.get(slug)
        if cat is None:
            updates.append(gr.update(choices=[], value=[], visible=False))
            continue
        allowed = [t["tag"] for t in cat["tags"]]
        kept = [t for t in (selected.get(slug) or []) if t in allowed]
        updates.append(gr.update(choices=pony_tags.choices(cat), value=kept, visible=True))
    return updates


def pony_tag_clear():
    return [gr.update(value=[]) for _ in PONY_TAG_SLUGS] + [PONY_TAG_PREVIEW_HINT]


@_show_usage_errors
def generate(character, anchor, custom_anchor, prompt, tier, negative_prompt, seed, ip_adapter_weight,
             pose_reference, pose_library, controlnet_strength, resolution, use_hq, hires_denoise,
             character_lora_strength, use_facedetailer, face_denoise, hand_denoise, facedetailer_backend,
             style_positive, style_negative, checkpoint_choice, lora_strength, use_cloud=False,
             progress=gr.Progress()):
    # The scope has to wrap the whole handler, not just gen_custom: upload_reference_image and the
    # has_node / log_gpu_memory calls inside the generation path all consult the backend too.
    with client.backend_scope("runpod" if use_cloud else "local"):
        return _generate(character, anchor, custom_anchor, prompt, tier, negative_prompt, seed, ip_adapter_weight,
                         pose_reference, pose_library, controlnet_strength, resolution, use_hq, hires_denoise,
                         character_lora_strength, use_facedetailer, face_denoise, hand_denoise,
                         facedetailer_backend, style_positive, style_negative, checkpoint_choice, lora_strength,
                         use_cloud, progress)


def _generate(character, anchor, custom_anchor, prompt, tier, negative_prompt, seed, ip_adapter_weight,
              pose_reference, pose_library, controlnet_strength, resolution, use_hq, hires_denoise,
              character_lora_strength, use_facedetailer, face_denoise, hand_denoise, facedetailer_backend,
              style_positive, style_negative, checkpoint_choice, lora_strength, use_cloud, progress):
    if not prompt.strip():
        raise gr.Error("請輸入 prompt")
    if use_cloud:
        problems = cloud_video.config_status("runpod")
        if problems:
            raise gr.Error("雲端生成還沒設定好：" + "；".join(problems) + "（設定方式見「☁️ 雲端影片」分頁）")
        gr.Info("雲端生成：沒有暖機的 worker 時要先冷啟動，第一張可能要等好幾分鐘")
    else:
        _ensure_comfyui()
    trigger = None if character == NO_CHARACTER else character
    is_zimage = checkpoint_choice in client.ZIMAGE_MODELS
    if is_zimage:
        # Z-Image is plain txt2img here (no FaceID/ControlNet/FaceDetailer files exist for it): explicit
        # identity/pose inputs are refused, a picked character still works as a text description.
        if custom_anchor or pose_reference or (pose_library and pose_library != POSE_NONE):
            raise gr.Error("Z-Image 目前只支援純文字生圖——自行上傳的 anchor（FaceID 鎖臉）、骨架姿勢參考圖、"
                           "骨架庫都沒有 Z-Image 版的模型檔，請先清掉這些欄位")
        # The check asks the LOCAL server which model files it has; that says nothing about a worker.
        missing = [] if use_cloud else client.zimage_missing_files(checkpoint_choice)
        if missing:
            raise gr.Error(f"找不到 Z-Image 模型檔：{', '.join(missing)}——下載指令見 README「Z-Image Turbo（選用安裝）」")
        if trigger and anchor:
            gr.Info("Z-Image 沒有 FaceID：角色只會用文字描述，不會套用 anchor 圖")
        gr.Info("Z-Image：ComfyUI 啟動後的第一張要先載入約 11 GB 模型，會等比較久；之後每張約 1-3 分鐘")
        anchor, use_hq, use_facedetailer = None, False, False
    anchor_path = custom_anchor or anchor
    if trigger and not anchor_path and not is_zimage:
        raise gr.Error("選了角色就要選一張 anchor 圖，或自行上傳一張")
    # An uploaded photo (pose_reference) wins over a library pick; a library
    # skeleton goes through pose_name (fed to ControlNet directly, no preprocessor).
    pose_name = None if (pose_reference or pose_library == POSE_NONE) else pose_library
    # In HQ mode FaceDetailer + ControlNet + FaceID all run in one workflow, so
    # the old "can't combine" / "needs anchor" restrictions only apply to the
    # legacy (non-HQ) path.
    if not use_hq:
        if pose_reference and not anchor_path:
            raise gr.Error("非 HQ 模式下骨架姿勢參考圖需要搭配 anchor 圖（選一張角色 anchor 或自行上傳身分參考圖）；或打開 HQ 兩段式")
        if use_facedetailer and not anchor_path:
            raise gr.Error("臉部/手部精修需要 anchor 圖（選一張角色 anchor 或自行上傳身分參考圖）；或打開 HQ 兩段式")
        if use_facedetailer and pose_reference:
            raise gr.Error("非 HQ 模式下臉部/手部精修跟骨架姿勢控制不能同時使用（兩套獨立 workflow）；打開 HQ 兩段式即可合併")
        if pose_name:
            raise gr.Error("骨架庫需要 HQ 兩段式（直接餵 ControlNet）；請打開 HQ 兩段式")

    checkpoint = None if checkpoint_choice == CHECKPOINT_DEFAULT else checkpoint_choice
    is_sd15 = checkpoint in client.SD15_CHECKPOINTS
    variant_key = checkpoint or (gc.DEFAULT_CUSTOM_CHECKPOINT if use_hq else "juggernaut")
    if use_cloud:
        gr.Info("雲端生成用 worker 上的完整版權重，本機的「模型版本」（量化版）設定不影響雲端")
    else:
        for note in client.variant_preflight(variant_key, uses_pose=bool(pose_reference or pose_name)):
            gr.Info(note)
    if is_sd15 and (anchor_path or pose_reference):
        raise gr.Error("這個 checkpoint 是 SD1.5，目前只支援純文字生圖——anchor（IP-Adapter）、"
                        "骨架姿勢控制都還沒接（那些用的是 SDXL 專用的模型檔，跟 SD1.5 對不上）")
    if is_sd15 and resolution != RESOLUTION_AUTO:
        raise gr.Error("這個 checkpoint 是 SD1.5，畫布比例請選「自動」——其他選項的解析度是給 SDXL 用的，"
                        "套在 SD1.5 上解析度太高，容易出現肢體變形/重複身體部位")

    if resolution == RESOLUTION_SQUARE:
        width, height = client.WIDTH, client.HEIGHT
    elif resolution == RESOLUTION_PORTRAIT:
        width, height = gc.FULL_BODY_RESOLUTION
    elif resolution == RESOLUTION_LANDSCAPE:
        height, width = gc.FULL_BODY_RESOLUTION  # same pair, swapped - 1216x832
    else:
        width, height = None, None  # gen_custom's own auto default (SD15_WIDTH/HEIGHT for is_sd15)

    if pose_name and width:
        # ControlNet center-crops the skeleton to the canvas aspect, so a square
        # canvas quietly cuts off its head and feet. Stop rather than produce a
        # broken pose the user has no way to diagnose.
        loss = pose_skeletons.crop_fraction(pose_name, width, height)
        if loss > pose_skeletons.CANVAS_CROP_TOLERANCE:
            sw, sh = pose_skeletons.canvas_for(pose_name)
            raise gr.Error(
                f"骨架庫姿勢「{pose_name}」是 {sw}x{sh} 直式，這個畫布比例會把骨架"
                f"裁掉 {loss * 100:.0f}%（頭跟腳會被切掉），姿勢就散了。"
                "畫布比例請改成「自動」或「直向」。"
            )

    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates")
    stem = f"gui_seed{int(seed)}"
    backend = "mediapipe" if facedetailer_backend == FACEDETAILER_BACKEND_MEDIAPIPE else "yolo"
    with client.progress_reporter(_step_reporter(progress)):
        gc.gen_custom(prompt, negative_prompt, tier, trigger, anchor_path, out_dir, int(seed), stem, ip_adapter_weight,
                      pose_reference_path=pose_reference, pose_name=pose_name,
                      controlnet_strength=controlnet_strength if (pose_reference or pose_name) else None,
                      width=width, height=height,
                      use_facedetailer=use_facedetailer,
                      facedetailer_backend=backend,
                      style_positive=style_positive, style_negative=style_negative,
                      checkpoint=checkpoint, lora_strength=lora_strength,
                      hq=use_hq, character_lora_strength=character_lora_strength,
                      hires_denoise=hires_denoise if use_hq else None,
                      facedetailer_face_denoise=face_denoise if use_facedetailer else None,
                      facedetailer_hand_denoise=hand_denoise if use_facedetailer else None)
    return os.path.join(out_dir, f"{stem}.png")


def select_picker_character(evt: gr.SelectData):
    return picker_characters()[evt.index][0]


def select_picker_pose(evt: gr.SelectData):
    return picker_pose_items()[evt.index][0]


def select_picker_scene(tier, evt: gr.SelectData):
    return picker_scene_items(tier)[evt.index][0]


def refresh_picker_scenes(tier):
    """The suggestive-tier scenes only exist at that tier, so switching back to safe has to drop
    both the tiles and any selection made from them - otherwise a beach scene stays selected
    while invisible, and generates."""
    return gr.update(value=_gallery_value(picker_scene_items(tier)), selected_index=None), None


def _picker_extra_in_english(extra_text):
    """CLIP largely ignores Chinese, so a Chinese 補充描述 would quietly do nothing. Translate it
    here rather than making the user press the button first - the point of this tab is that
    typing correct model-facing text is not a prerequisite."""
    if not extra_text or not extra_text.strip() or extra_text.isascii():
        return extra_text
    try:
        return translate_prompt.translate_to_english(extra_text)
    except RuntimeError as exc:
        raise gr.Error(f"補充描述翻譯失敗（{exc}）——可以先自己改成英文，或清空這一欄") from exc


def update_picker_preview(character, pose_slug, scene_slug, checkpoint_choice, tier, extra_text):
    checkpoint = None if checkpoint_choice == CHECKPOINT_DEFAULT else checkpoint_choice
    try:
        plan = gc.plan_picker(character, pose_slug, scene_slug, checkpoint, tier, extra_text)
    except (gc.UsageError, ValueError) as exc:
        return "", f"⚠️ {exc}"
    if not plan.ready:
        return "", "選一個場景或姿勢（或填補充描述）就會在這裡看到實際送出的 prompt。"
    lines = [f"- ⚠️ {n}" for n in plan.notices]
    if plan.mode == "controlnet_faceid":
        locked = []
        if plan.anchor_path:
            locked.append("臉（FaceID anchor）")
        if plan.pose_name:
            locked.append("姿勢（ControlNet 骨架）")
        if locked:
            lines.insert(0, f"- ✅ 這個 checkpoint 會真的鎖住：{'、'.join(locked)}")
    if extra_text and not str(extra_text).isascii():
        lines.append("- ℹ️ 補充描述是中文，送出前會自動翻成英文（預覽顯示的是翻譯前的組合）")
    return plan.preview_prompt, "\n".join(lines)


@_show_usage_errors
def generate_from_picker(character, pose_slug, scene_slug, checkpoint_choice, tier, seed, extra_text,
                         progress=gr.Progress()):
    """Run a generation from three thumbnail picks.

    Delegates to generate() rather than calling gc.gen_custom itself: that handler carries ~90
    lines of gating (Z-Image model-file check, variant preflight, the HQ-only skeleton rule,
    canvas/crop guard, starting ComfyUI, where the output lands) which must not exist twice.
    plan_picker has already nulled the anchor and skeleton for checkpoints that can't take them,
    so none of generate()'s rejections can fire on input that got here.
    """
    checkpoint = None if checkpoint_choice == CHECKPOINT_DEFAULT else checkpoint_choice
    plan = gc.plan_picker(character, pose_slug, scene_slug, checkpoint, tier, _picker_extra_in_english(extra_text))
    if not plan.ready:
        raise gr.Error("至少要選一個場景或姿勢，或在「補充描述」填點東西——只選角色沒有畫面可以生成")
    if plan.trigger and not plan.anchor_path and plan.mode == "controlnet_faceid":
        # Unreachable from the gallery (picker_characters only lists characters that have an
        # anchor), but generate() refuses trigger-without-anchor on SDXL rather than quietly
        # producing a stranger under that name - so say the same thing in this tab's terms
        # instead of letting its less specific message through.
        raise gr.Error(
            f"角色「{plan.trigger}」還沒有 anchor 圖，這個 checkpoint 需要它才能鎖臉。"
            f"先用 `generate_character.py anchors --character {plan.trigger}` 產生一張，"
            "或改用 Z-Image / SD1.5（那些本來就只用文字描述）"
        )
    for notice in plan.notices:
        gr.Info(notice)
    return generate(
        plan.trigger or NO_CHARACTER, plan.anchor_path, None, plan.prompt_body, tier, "", seed,
        client.IP_ADAPTER_WEIGHT, None, plan.pose_name or POSE_NONE, client.CONTROLNET_STRENGTH,
        RESOLUTION_AUTO, True, client.HIRES_DENOISE, client.CHARACTER_LORA_STRENGTH,
        True, client.FACEDETAILER_FACE_DENOISE, client.FACEDETAILER_HAND_DENOISE, FACEDETAILER_BACKEND_YOLO,
        gc.REALISTIC_STYLE, gc.REALISTIC_NEGATIVE, checkpoint_choice, 0.0,
        use_cloud=False, progress=progress,
    )


@_show_usage_errors
def generate_video_animatediff(character, face_ref, prompt, tier, negative_prompt, seed, ip_adapter_weight,
                                facedetailer_denoise, frames, fps, width, height, style_positive, style_negative,
                                checkpoint_choice, motion_lora_choice, motion_lora_strength,
                                hires, upscale_choice, interp, use_facedetailer, lcm,
                                faceid_v2_weight, motion_scale):
    if not prompt.strip():
        raise gr.Error("請輸入 prompt")
    if not face_ref:
        raise gr.Error("請上傳臉部參考圖（用於 FaceID 鎖定長相）— 僅限虛構/AI生成的臉，禁止上傳真人照片")
    _ensure_comfyui()
    trigger = None if character == NO_CHARACTER else character
    checkpoint = None if checkpoint_choice == ANIMATEDIFF_CHECKPOINT_DEFAULT else checkpoint_choice
    motion_lora = None if motion_lora_choice == MOTION_LORA_NONE else motion_lora_choice
    upscale_to = 0 if upscale_choice == UPSCALE_OFF else int(upscale_choice.split()[1].rstrip("px"))
    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates", "videos")
    return gc.gen_video_animatediff(prompt, negative_prompt, tier, trigger, face_ref, out_dir, int(seed),
                                    ip_adapter_weight=ip_adapter_weight,
                                    facedetailer_denoise=facedetailer_denoise,
                                    frames=int(frames), fps=int(fps) or None,
                                    width=int(width), height=int(height),
                                    style_positive=style_positive, style_negative=style_negative,
                                    checkpoint=checkpoint,
                                    motion_lora=motion_lora, motion_lora_strength=motion_lora_strength,
                                    hires=hires, upscale_to=upscale_to, interp=int(interp),
                                    use_facedetailer=use_facedetailer, lcm=lcm,
                                    faceid_v2_weight=faceid_v2_weight, motion_scale=motion_scale)


def generate_talking_head_ui(image, audio, size, preprocess, still, expression_scale, enhancer, pose_style):
    if not image:
        raise gr.Error("請上傳來源人像（僅限虛構/AI生成的臉，禁止上傳真人照片）")
    if not audio:
        raise gr.Error("請上傳語音檔")
    ok, reason = talking_head.is_available()
    if not ok:
        raise gr.Error(f"SadTalker 無法使用：{reason}（安裝見 README「會講話的嘴型影片」章節）")
    # SadTalker is a separate process on the same 8GB card - make a warm ComfyUI drop its models
    # first. Deliberately NOT _ensure_comfyui(): this route doesn't need ComfyUI at all.
    if client.is_server_running():
        client.free_vram()
    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates", "videos")
    try:
        return talking_head.generate_talking_head(
            image, audio, out_dir, size=int(size), preprocess=preprocess, still=still,
            expression_scale=expression_scale,
            enhancer=None if enhancer == TALK_ENHANCER_NONE else enhancer,
            pose_style=int(pose_style))
    except RuntimeError as exc:
        print(exc, flush=True)
        raise gr.Error(f"SadTalker 生成失敗（完整 log 見終端機）：{str(exc)[-800:]}") from exc


@_show_usage_errors
def generate_video_svd(character, init_image, seed, frames, fps, motion_bucket_id):
    if not init_image:
        raise gr.Error("請上傳要配上動作的圖片")
    _ensure_comfyui()
    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates", character, "videos")
    gc.gen_video(character, init_image, out_dir, int(seed), int(frames), int(fps), int(motion_bucket_id))
    return os.path.join(out_dir, f"video_seed{int(seed)}.webm")


@_show_usage_errors
def generate_gif(character, anchor, prompt, tier, negative_prompt, seed, frame_count, duration_ms, denoise,
                  ip_adapter_weight, style_positive, style_negative, checkpoint_choice, lora_strength):
    if not prompt.strip():
        raise gr.Error("請輸入 prompt")
    _ensure_comfyui()
    trigger = None if character == NO_CHARACTER else character
    if trigger and not anchor:
        raise gr.Error("選了角色就要選一張 anchor 圖")
    checkpoint = None if checkpoint_choice == CHECKPOINT_DEFAULT else checkpoint_choice
    if checkpoint in client.SD15_CHECKPOINTS:
        raise gr.Error("這個 checkpoint 是 SD1.5——批次 GIF 的每一張都要靠 IP-Adapter 鎖同一張臉，SD1.5 沒接這個功能，換一個 SDXL 系列的 checkpoint")
    if checkpoint in client.ZIMAGE_MODELS:
        raise gr.Error("Z-Image 只接了單張純文字生圖——批次 GIF 的每一張都要靠 IP-Adapter 鎖同一張臉，Z-Image 沒有這個功能，換一個 SDXL 系列的 checkpoint")
    for note in client.variant_preflight(checkpoint or gc.DEFAULT_CUSTOM_CHECKPOINT):
        gr.Info(note)
    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates")
    return gc.gen_gif(prompt, negative_prompt, tier, trigger, anchor, out_dir, int(seed), int(frame_count),
                       ip_adapter_weight, duration_ms=int(duration_ms), denoise=denoise,
                       style_positive=style_positive, style_negative=style_negative,
                       checkpoint=checkpoint, lora_strength=lora_strength)


# --- ☁️ cloud video ---------------------------------------------------------------------------

CLOUD_PROVIDER_CHOICES = [
    ("RunPod（本專案自己的 worker，兩種工作都能跑）", "runpod"),
    ("Replicate（託管模型，只能跑 Wan 類圖生影片）", "replicate"),
]
CLOUD_JOB_WAN = "Wan 2.2 圖生影片（第一幀 → 最長 5 秒 720p）"
CLOUD_JOB_ANIMATEDIFF = "AnimateDiff 雲端高畫質（FaceID 鎖臉，解除 8GB 上限）"
CLOUD_PRESETS = {
    "直式 704×1280": dict(width=704, height=1280),
    "橫式 1280×704": dict(width=1280, height=704),
}
CLOUD_PRESET_DEFAULT = "直式 704×1280"
CLOUD_LENGTHS = {"短（約 2 秒）": 49, "中（約 3.4 秒）": 81, "長（約 5 秒）": 121}
CLOUD_LENGTH_DEFAULT = "中（約 3.4 秒）"
_CLOUD_CANCEL = {}   # Gradio session hash -> threading.Event for that session's running cloud job
_CLOUD_CANCEL_LOCK = threading.Lock()


def _cloud_reporter(progress):
    """Cloud jobs have no step counter we can see, only elapsed time against a limit - so report that,
    in the same (stage, current, total) shape _step_reporter uses."""
    def report(label, elapsed, limit):
        progress((elapsed, limit), desc=f"{label}（已 {elapsed} 秒／上限 {limit} 秒）")
    return report


def cloud_status_markdown():
    settings = cloud_video.load_cloud_settings()
    lines = [
        "**雲端工作按 GPU 秒數計費，關掉瀏覽器不會自動停止——要中止請按「取消」。**",
        f"- RunPod endpoint：`{settings['runpod']['endpoint_id'] or '未設定'}`",
        f"- Replicate 模型：`{settings['replicate']['model'] or '未設定'}`",
    ]
    problems = cloud_video.config_status()
    if problems:
        lines.append("- 尚未完成的設定：" + "；".join(problems) + "（步驟見 README「雲端影片」）")
    return "\n".join(lines)


def save_cloud_config(endpoint_id, replicate_model):
    cloud_video.save_cloud_settings(
        {"runpod": {"endpoint_id": (endpoint_id or "").strip()},
         "replicate": {"model": (replicate_model or "").strip()}}
    )
    return cloud_status_markdown()


def cloud_anchor_for(character):
    """First anchor of the picked character, as the cloud tab's first frame."""
    if not character or character == NO_CHARACTER:
        raise gr.Error("先選一個角色")
    path = gc.picker_anchor_path(character)
    if not path:
        raise gr.Error(f"角色「{character}」還沒有 anchor 圖")
    return path


@_show_usage_errors
def generate_cloud_video(provider, job_label, character, first_frame, prompt, tier, negative_prompt, seed,
                         preset, length, timeout_s, request: gr.Request, progress=gr.Progress()):
    """Submit one video job to RunPod or Replicate and wait for it.

    Never calls _ensure_comfyui(): nothing here touches the local ComfyUI or GPU, which is also why
    the button has its own concurrency id instead of "gpu" - a cloud job running for ten minutes
    must not block local generations, and vice versa."""
    job_type = "video_wan_i2v" if job_label == CLOUD_JOB_WAN else "video_animatediff"
    trigger = None if character == NO_CHARACTER else character
    if not first_frame and not (provider == "runpod" and trigger):
        raise gr.Error("需要第一幀／臉部參考圖（上傳，或按「用角色的 anchor」「用圖片選擇生圖的結果」）")
    prompt = (prompt or "").strip()
    if job_type == "video_animatediff" and any(ord(c) > 127 for c in prompt):
        # AnimateDiff's SD1.5 CLIP mostly ignores Chinese; Wan's umt5 reads it natively.
        try:
            prompt = translate_prompt.translate_to_english(prompt)
        except RuntimeError as exc:
            raise gr.Error(f"AnimateDiff 需要英文 prompt，自動翻譯失敗（{exc}）——請改打英文") from exc

    if job_type == "video_wan_i2v":
        params = dict(frames=CLOUD_LENGTHS.get(length, 81), fps=client.WAN_FPS, cfg=client.WAN_CFG,
                      **CLOUD_PRESETS.get(preset, CLOUD_PRESETS[CLOUD_PRESET_DEFAULT]))
    else:
        params = dict(upscaleTo=1024)

    event = threading.Event()
    key = getattr(request, "session_hash", None) or "default"
    with _CLOUD_CANCEL_LOCK:
        _CLOUD_CANCEL[key] = event
    try:
        result = cloud_video.run_cloud_video(
            provider, job_type, prompt=prompt, extra_negative=(negative_prompt or "").strip(), tier=tier,
            trigger=trigger, image_path=first_frame, seed=int(seed), params=params,
            out_dir=os.path.join(os.path.dirname(__file__), "reference_candidates", "videos", "cloud"),
            max_wait_s=int(timeout_s), on_status=_cloud_reporter(progress), cancel_event=event,
        )
    finally:
        with _CLOUD_CANCEL_LOCK:
            if _CLOUD_CANCEL.get(key) is event:
                del _CLOUD_CANCEL[key]
    info = (f"**完成** ｜ {result.provider} 工作 `{result.job_id}` ｜ 總耗時 {result.elapsed_s} 秒"
            + (f" ｜ 排隊 {result.queue_s:.0f} 秒" if result.queue_s is not None else "")
            + (f" ｜ GPU 執行 {result.execution_s:.0f} 秒（計費依據）" if result.execution_s is not None else "")
            + f"\n\n存到 `{result.path}`")
    return result.path, info


def cancel_cloud_video(request: gr.Request):
    key = getattr(request, "session_hash", None) or "default"
    with _CLOUD_CANCEL_LOCK:
        event = _CLOUD_CANCEL.get(key)
    if event is None:
        gr.Info("目前沒有正在跑的雲端工作")
        return
    event.set()
    gr.Info("已要求取消，下一次查詢狀態時會送出取消")


with gr.Blocks(title="AI Image Lab") as demo:
    gr.Markdown("# AI Image Lab\n先確定 ComfyUI server 已經在跑（127.0.0.1:8188）。")

    with gr.Tabs():
        with gr.Tab("🖼️ 自訂生圖"):
            with gr.Row():
                with gr.Column():
                    character = gr.Dropdown(CHARACTER_CHOICES, label="角色（選填，選了會用該角色身分）", value=NO_CHARACTER)
                    anchor = gr.Dropdown([], label="Anchor 圖（選了角色才需要）")
                    anchor_preview = gr.Image(label="Anchor 圖預覽", interactive=False, height=200)
                    custom_anchor = gr.Image(
                        label="或自行上傳身分參考圖（會覆蓋上面的 Anchor 選擇）— 僅限虛構/AI生成的角色照，禁止上傳真人照片",
                        type="filepath",
                    )
                    with gr.Accordion("依圖片產生 Prompt（上傳一張圖，自動轉成英文 prompt）", open=False):
                        caption_upload = gr.Image(label="上傳圖片", type="filepath")
                        with gr.Row():
                            caption_btn = gr.Button("產生描述（自然語言）")
                            tag_btn = gr.Button("產生標籤（WD14，Pony 系適用）")
                        caption_output = gr.Textbox(label="結果（僅供參考，可自行編輯後使用）", lines=2, interactive=False)
                        gr.Markdown(
                            "「描述」用 BLIP 出自然語言句子，juggernaut 和 SD1.5 吃這種。"
                            "「標籤」用 WD14 出 booru 標籤（`1girl, solo, long hair, ...`），"
                            "Pony 系 checkpoint 是用這種格式訓練的，通常不必再自己補太多字。"
                            "標籤模式需要 ComfyUI（第一次按會自動啟動），單張約 3 秒。"
                        )
                    with gr.Accordion("HQ 兩段式生成（預設開啟，5 分鐘內高品質）", open=True):
                        use_hq = gr.Checkbox(
                            value=True,
                            label="兩段式：先低解析度生成 → ESRGAN 放大 → 低 denoise 重採樣補細節 → 臉部/手部精修。"
                                  "修正失真與「3D render」感，並在有訓練好的角色 LoRA 時自動套用。關掉則走舊的單段式",
                        )
                        hires_denoise = gr.Slider(0.2, 0.6, value=client.HIRES_DENOISE, step=0.05,
                                                  label="Hires 第二段 denoise（0.4 建議；太高會讓臉飄移，太低補不到細節）")
                        character_lora_strength = gr.Slider(0.0, 1.2, value=client.CHARACTER_LORA_STRENGTH, step=0.05,
                                                             label="角色 LoRA 強度（僅在該角色有訓練好的 LoRA 時作用，否則忽略）")
                    with gr.Accordion("骨架姿勢控制（ControlNet，選填）", open=False):
                        pose_library = gr.Dropdown(
                            [POSE_NONE] + pose_skeletons.list_names(), value=POSE_NONE,
                            label="姿勢骨架庫（選一個內建姿勢；用來壓住 checkpoint 靠文字壓不住的姿勢，例如各種躺姿）。"
                                  "若同時上傳了下方的參考照片，以照片為準",
                        )
                        skeleton_preview = gr.Image(label="骨架預覽", interactive=False, height=200)
                        pose_reference = gr.Image(
                            label="或：自訂姿勢參考照片（自動抽取骨架）。此圖只抽取關節骨架，"
                                  "不會保留臉部/身分資訊，可以是任何照片",
                            type="filepath",
                        )
                        controlnet_strength = gr.Slider(0.0, 1.5, value=client.CONTROLNET_STRENGTH, step=0.05,
                                                         label="骨架控制強度（0=忽略骨架，1=嚴格鎖定姿勢）")
                        gr.Markdown("ℹ️ 內建骨架庫用預先畫好的骨架直接餵 ControlNet（跳過 preprocessor），"
                                    "不掛 FaceID 時單張約 60-90 秒。上傳照片則多一道骨架抽取。")
                        pose_library.change(_pose_preview, inputs=pose_library, outputs=skeleton_preview)
                    with gr.Accordion("臉部/手部精修（ADetailer，HQ 模式預設開啟）", open=True):
                        use_facedetailer = gr.Checkbox(
                            value=True,
                            label="生成後自動偵測臉部+手部，各自裁切放大重繪一次，修正常見的臉部/手指小瑕疵。"
                                  "HQ 模式可與骨架姿勢控制同時使用",
                        )
                        face_denoise = gr.Slider(0.0, 1.0, value=client.FACEDETAILER_FACE_DENOISE, step=0.05,
                                                 label="臉部精修強度（0=不變，1=該區域完全重畫）")
                        hand_denoise = gr.Slider(0.0, 1.0, value=client.FACEDETAILER_HAND_DENOISE, step=0.05,
                                                 label="手部精修強度（手部通常用比臉部略低的值，避免重畫出更糟的手）")
                        facedetailer_backend = gr.Radio(
                            FACEDETAILER_BACKEND_CHOICES, value=FACEDETAILER_BACKEND_YOLO,
                            label="臉部偵測方式（HQ 模式一律用 YOLO；MediaPipe 只在關掉 HQ 的舊單段式路徑有作用）",
                        )
                    resolution = gr.Radio(RESOLUTION_CHOICES, value=RESOLUTION_AUTO, label="畫布比例（HQ 最終尺寸：正方形→1248×1248、直式→1056×1536、橫式→1536×1056；正方形沒有足夠垂直空間放全身，會被裁成上半身）")
                    with gr.Accordion("Pony 關鍵字（點選代替打字：必備 4 類 ＋ 其他 9 類）", open=False):
                        gr.Markdown(
                            "「必備」是 Pony 系 checkpoint 的標配：品質分數、來源風格、內容分級、主體與人數。"
                            "少了它們畫面會偏插畫，人數和年齡也比較不受控。"
                            "「其他」是想到才加的：鏡頭、姿勢、表情、髮型體態、服裝、光線、場景、風格、額外負面詞。\n\n"
                            f"{PONY_TAG_QUALITY_NOTE}\n\n"
                            "非 Pony 系（juggernaut / SD1.5 / Z-Image）會自動收起 `score_*`、`source_*`、"
                            "`rating_*` 這幾類——它們沒有被這樣訓練過；描述性的字則所有 checkpoint 都能用。\n\n"
                            "打自然語言本來就會自動補上推導出的 booru 標籤（`prompt_adapter`），"
                            "這裡是想「明確指定」時用的——特別是 `rating_*` / `source_*`，那是句子推不出來的。"
                        )
                        pony_tag_groups = [
                            gr.CheckboxGroup(
                                pony_tags.choices(_cat), value=[],
                                label=f"{'必備' if _cat['role'] == 'required' else '其他'}｜{_cat['name_zh']}",
                                info=_cat["note_zh"],
                            )
                            for _cat in PONY_TAG_INITIAL
                        ]
                        pony_tag_preview_box = gr.Textbox(
                            label="會加進去的內容（唯讀預覽）", value=PONY_TAG_PREVIEW_HINT,
                            lines=3, interactive=False,
                        )
                        with gr.Row():
                            pony_tag_append_btn = gr.Button("加入 Prompt", variant="secondary")
                            pony_tag_clear_btn = gr.Button("清除勾選", size="sm", scale=0)
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="例如: sitting in a cozy library, reading a book, warm afternoon light")
                    translate_btn = gr.Button("翻譯成英文（CLIP 對中文支援不佳，建議先翻譯再送出；已經是英文按下去不會被改動）")
                    negative_prompt = gr.Textbox(label="額外負面詞（選填，一次性追加，不影響下面的風格詞預設值）", lines=1)
                    with gr.Accordion("風格正/負面詞（進階 - 可直接在這裡調整寫實感等風格用詞，不用改程式碼）", open=False):
                        gr.Markdown(
                            "下面兩欄會自動接在 prompt/negative prompt 後面，預設值就是目前程式碼裡用的寫實化詞彙。"
                            "**這裡改不到年齡保護詞、露骨內容封鎖詞**，那些永遠固定套用、不會因為這裡的設定被移除或減弱。"
                        )
                        style_positive = gr.Textbox(label="風格正面詞", value=gc.REALISTIC_STYLE, lines=2)
                        style_negative = gr.Textbox(label="風格負面詞", value=gc.REALISTIC_NEGATIVE, lines=2)
                    gr.Markdown(
                        "**目前預設（cyberrealistic_pony + HQ + 精修）就是實測最真實、失真最少的組合，不用改就能直接用。**"
                        "純文字生圖、不需要鎖臉/姿勢/精修時，`z_image_turbo` 寫實度和手部更好，可以切過去試試。\n\n"
                        "**什麼時候要切換 checkpoint：**\n"
                        "- 需要「Anchor 身分鎖定」「骨架姿勢控制」或「臉部/手部精修」任一項 → 只能選 SDXL 系列"
                        "（juggernaut / pony / cyberrealistic_pony / pony_realism），這三個功能都是接在 SDXL 專用模型檔上，"
                        "選 SD1.5 送出時會直接跳錯誤。\n"
                        "- 只要純文字生圖、想要更自然語言的 prompt、或想要 512×768 左右的原生解析度 → 可以切到 SD1.5 系列"
                        "（realistic_vision / cyberrealistic），但切過去後 anchor / 姿勢控制 / 精修都會被鎖住，畫布比例也只能選「自動」。\n"
                        "- pony / cyberrealistic_pony / pony_realism 需要的 `score_9, score_8_up, score_7_up` 品質 tag 前綴"
                        "會自動加上，Prompt 欄位一樣打自然語言就好，不用自己記得加。"
                        "下面的「寫實風格 LoRA 強度」建議切到 0（這顆 LoRA 是針對 juggernaut 調的，套在 pony 系會打架）。"
                        "\n- `z_image_turbo`（Z-Image Turbo）→ 純文字生圖，寫實度、手指跟畫面裡的文字（包括中文字）都比 SDXL 好，"
                        "Prompt 可以直接打中文不用翻譯；但沒有 anchor 鎖臉、姿勢控制、精修（選了角色只會用文字描述），"
                        "要先依 README 下載模型，第一張要載入約 11 GB 模型，會比較久。"
                    )
                    with gr.Accordion("模型版本（完整版 / 量化版）", open=False):
                        gr.Markdown(
                            "SDXL / Pony 四個模型可以各自選「完整版」或「量化版 fp8」。量化版把主模型權重存成 fp8，"
                            "VRAM、記憶體和硬碟用量比較少，換模型時搬得比較快；在這張 RTX 2070 上運算仍是 fp16，不會算得比較快。"
                            "這是全域設定：這裡、批次 GIF 和 CLI 都會套用，按「儲存」後下一張就生效。"
                            "選了量化版但還沒轉出量化檔時，會自動改用完整版並提示轉換指令；骨架姿勢（ControlNet）一律用完整版。"
                        )
                        variant_radios = [
                            gr.Radio([("完整版", "full"), ("量化版 fp8", "quant")], value=_row["chosen"],
                                     label=_variant_label(_row), interactive=not _row["unsupported"])
                            for _row in client.variant_status()
                        ]
                        variant_table = gr.Markdown(_variant_table())
                        with gr.Row():
                            variant_save_btn = gr.Button("儲存模型版本設定")
                            variant_refresh_btn = gr.Button("重新檢查檔案")
                    checkpoint_choice = gr.Dropdown(
                        CHECKPOINT_CHOICES, value="cyberrealistic_pony", label="Checkpoint 模型",
                    )
                    lora_strength = gr.Slider(
                        0.0, 3.0, value=0.0, step=0.1,
                        label="寫實風格 LoRA 強度（針對 Juggernaut 調的，換成 pony 等其他 checkpoint 時建議調到 0）",
                    )
                    use_cloud = gr.Checkbox(
                        value=False,
                        label="用 RunPod 雲端 GPU 生成（同一套 workflow 送到雲端跑，不佔本機 8GB 卡；"
                              "需先在「☁️ 雲端影片」分頁設定 endpoint 與環境變數 RUNPOD_API_KEY）",
                    )
                    tier = gr.Radio(["safe", "suggestive"], value="suggestive", label="內容分級（suggestive 上限跟 test-suggestive 一樣，露骨內容依然封鎖）")
                    seed = gr.Number(value=9000, label="Seed", precision=0)
                    ip_weight = gr.Slider(0.0, 3.0, value=client.IP_ADAPTER_WEIGHT, step=0.05, label="IP-Adapter 權重（FaceID 量表，有選角色才有作用）")
                    btn = gr.Button("生成", variant="primary")
                with gr.Column():
                    output = gr.Image(label="結果")

            character.change(refresh_anchors, inputs=character, outputs=[anchor, anchor_preview])
            anchor.change(lambda path: path, inputs=anchor, outputs=anchor_preview)
            # The Pony keyword groups follow both the tier and the checkpoint: the tier decides
            # whether the suggestive-only entries exist at all, the checkpoint whether the
            # score/source/rating categories apply. Chained onto the existing resolution reset
            # rather than replacing it - a second .change() on the same component would not
            # remove the first, but keeping them in one chain makes the order explicit.
            _pony_tag_inputs = pony_tag_groups + [tier, checkpoint_choice]
            checkpoint_choice.change(reset_resolution_for_sd15, inputs=checkpoint_choice, outputs=resolution).then(
                pony_tag_refresh, inputs=_pony_tag_inputs, outputs=pony_tag_groups).then(
                pony_tag_preview, inputs=_pony_tag_inputs, outputs=pony_tag_preview_box)
            tier.change(pony_tag_refresh, inputs=_pony_tag_inputs, outputs=pony_tag_groups).then(
                pony_tag_preview, inputs=_pony_tag_inputs, outputs=pony_tag_preview_box)
            for _group in pony_tag_groups:
                _group.change(pony_tag_preview, inputs=_pony_tag_inputs, outputs=pony_tag_preview_box)
            pony_tag_append_btn.click(
                pony_tag_append,
                inputs=pony_tag_groups + [tier, checkpoint_choice, prompt, negative_prompt],
                outputs=[prompt, negative_prompt],
            )
            pony_tag_clear_btn.click(
                pony_tag_clear, inputs=None, outputs=pony_tag_groups + [pony_tag_preview_box])
            variant_save_btn.click(save_variant_choices, inputs=variant_radios, outputs=variant_table)
            variant_refresh_btn.click(refresh_variant_status, inputs=None, outputs=variant_radios + [variant_table])
            caption_btn.click(caption_uploaded_image, inputs=caption_upload, outputs=[caption_output, prompt])
            tag_btn.click(tag_uploaded_image, inputs=caption_upload, outputs=[caption_output, prompt])
            translate_btn.click(translate_prompt_to_english, inputs=prompt, outputs=prompt)
            # concurrency_id="gpu" on every generation button in this file: Gradio's default
            # concurrency_limit=1 is PER LISTENER, so without a shared id one browser tab could
            # start "生成" while another starts "生成 GIF" - two jobs aimed at the same 8GB card.
            # comfyui_client's own lock would serialise them anyway, but the second user would
            # just see their button spin with no explanation; a shared concurrency_id makes
            # Gradio queue it visibly instead. Includes the SadTalker button, which does not go
            # through ComfyUI but competes for the same VRAM.
            btn.click(generate, inputs=[character, anchor, custom_anchor, prompt, tier, negative_prompt, seed, ip_weight, pose_reference, pose_library, controlnet_strength, resolution, use_hq, hires_denoise, character_lora_strength, use_facedetailer, face_denoise, hand_denoise, facedetailer_backend, style_positive, style_negative, checkpoint_choice, lora_strength, use_cloud], outputs=output, concurrency_id="gpu")

        with gr.Tab("🎬 AnimateDiff 動態影片"):
            gr.Markdown(
                "跟「自訂生圖」分頁是分開的 workflow，用 SD1.5 + AnimateDiff motion module 生成短動態影片，"
                "並用 IPAdapter-FaceID 鎖住整段影片的臉部身分，再把所有影格的臉部一起重新採樣精修（不會一幀一個樣）——"
                "解決純 SVD img2vid（見 README「幫已有的圖片配上動作」）常見的臉部變形/融化問題。"
                "**目前預設（realistic_vision、高清開、精修開、LCM 關）就是實測畫質最完整的組合，不用改就能直接用；"
                "只有女性角色、想省時間才建議勾 LCM 快速模式——男性角色開 LCM 常常會被畫成女生，見下方勾選框說明。**"
                "預設流程：採樣 512² → 二段式高清 768² → 臉部精修 → ESRGAN 放大到長邊 1024 → mp4。"
                "第一次執行會比較久（SD1.5 checkpoint、motion module、FaceID SD1.5 模型是分開載入的新模型組合）。"
            )
            with gr.Row():
                with gr.Column():
                    video_character = gr.Dropdown(CHARACTER_CHOICES, label="角色（選填，選了會用該角色身分敘述）", value=NO_CHARACTER)
                    video_face_ref = gr.Image(
                        label="臉部參考圖（FaceID 用，僅限虛構/AI生成，禁止上傳真人照片）",
                        type="filepath",
                    )
                    video_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="例如: sitting by a window, gentle breeze, turning head slightly")
                    video_translate_btn = gr.Button("翻譯成英文（已經是英文按下去不會被改動）")
                    video_negative_prompt = gr.Textbox(label="額外負面詞（選填）", lines=1)
                    video_tier = gr.Radio(["safe", "suggestive"], value="safe", label="內容分級（露骨內容依然封鎖）")
                    with gr.Row():
                        video_seed = gr.Number(value=6001, label="Seed", precision=0)
                        video_ip_weight = gr.Slider(0.0, 3.0, value=client.IP_ADAPTER_WEIGHT, step=0.05, label="IP-Adapter 權重")
                    with gr.Row():
                        video_frames = gr.Slider(8, 16, value=client.ANIMATEDIFF_FRAMES, step=1, label="採樣影格數（motion module 訓練上限 16）")
                        video_fps = gr.Slider(0, 32, value=0, step=1, label="輸出 FPS（0 = 自動：8 × 補幀倍數，片長不變只變順；調低可拉長成慢動作）")
                    with gr.Row():
                        video_width = gr.Number(value=client.ANIMATEDIFF_WIDTH, label="寬", precision=0)
                        video_height = gr.Number(value=client.ANIMATEDIFF_HEIGHT, label="高")
                    with gr.Group():
                        gr.Markdown("**畫質 / 流暢度 / 速度**")
                        with gr.Row():
                            video_hires = gr.Checkbox(value=True, label="二段式高清（1.5x，上限 768²；VRAM 不足會自動退回）")
                            video_upscale = gr.Dropdown(UPSCALE_CHOICES, value=f"長邊 {client.ANIMATEDIFF_UPSCALE_TO}px",
                                                        label="最終放大（ESRGAN 4x-UltraSharp，逐幀）")
                        with gr.Row():
                            video_interp = gr.Radio(list(client.ANIMATEDIFF_INTERP_CHOICES), value=1,
                                                    label="RIFE 補幀倍數（2/4 需先安裝 ComfyUI-Frame-Interpolation，見 README）")
                            video_use_facedetailer = gr.Checkbox(value=True, label="臉部精修（關掉比較快，但臉可能變糊/漂移）")
                        video_lcm = gr.Checkbox(value=False, label="LCM 快速模式（AnimateLCM，8 步取代 20 步；需先下載模型，見 README。男性角色實測常被畫成女生，不建議勾）")
                    video_facedetailer_denoise = gr.Slider(0.0, 1.0, value=client.ANIMATEDIFF_FACEDETAILER_DENOISE, step=0.05, label="臉部精修強度")
                    with gr.Row():
                        video_faceid_v2_weight = gr.Slider(0.0, 3.0, value=client.ANIMATEDIFF_FACEID_V2_WEIGHT, step=0.1,
                                                           label="FaceID 臉部結構權重（越高臉越固定但動作越少；實測對相似度幾乎沒幫助）")
                        video_motion_scale = gr.Slider(0.5, 1.2, value=client.ANIMATEDIFF_MOTION_SCALE, step=0.05,
                                                       label="動作幅度（1.0 = 完整動作；0.85 就幾乎靜止）")
                    video_checkpoint_choice = gr.Dropdown(
                        ANIMATEDIFF_CHECKPOINT_CHOICES, value=ANIMATEDIFF_CHECKPOINT_DEFAULT,
                        label="Checkpoint 模型（必須是 SD1.5，跟「自訂生圖」分頁的選項是分開的清單）",
                    )
                    with gr.Accordion("風格正/負面詞（進階，跟「自訂生圖」分頁獨立設定）", open=False):
                        video_style_positive = gr.Textbox(label="風格正面詞", value=gc.REALISTIC_STYLE, lines=2)
                        video_style_negative = gr.Textbox(label="風格負面詞（影片版不含 symmetrical face，避免臉被推向不對稱）",
                                                          value=gc.VIDEO_REALISTIC_NEGATIVE, lines=2)
                    with gr.Accordion("Motion LoRA（選用，鏡頭運動控制）", open=False):
                        gr.Markdown(
                            "官方 AnimateDiff Motion LoRA，控制的是**整個畫面的鏡頭運動**（縮放/平移/"
                            "傾斜/旋轉），**不是特定身體部位的物理晃動效果**（沒有「彈跳」「晃動」這類選項，"
                            "motion module 本身沒有對應的控制機制）。要先下載對應的 `.ckpt` 檔案放到"
                            "`ComfyUI/models/animatediff_motion_lora/`，見 README「AnimateDiff Motion LoRA」"
                            "安裝章節，沒下載的話選了會生成失敗。"
                        )
                        video_motion_lora = gr.Dropdown(MOTION_LORA_CHOICES, value=MOTION_LORA_NONE, label="鏡頭運動")
                        video_motion_lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="強度（超過 1 容易讓畫面明顯扭曲）")
                    video_btn = gr.Button("生成影片", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="結果")

            video_translate_btn.click(translate_prompt_to_english, inputs=video_prompt, outputs=video_prompt)
            video_btn.click(
                generate_video_animatediff,
                inputs=[video_character, video_face_ref, video_prompt, video_tier, video_negative_prompt, video_seed,
                        video_ip_weight, video_facedetailer_denoise, video_frames, video_fps, video_width, video_height,
                        video_style_positive, video_style_negative, video_checkpoint_choice,
                        video_motion_lora, video_motion_lora_strength,
                        video_hires, video_upscale, video_interp, video_use_facedetailer, video_lcm,
                        video_faceid_v2_weight, video_motion_scale],
                outputs=video_output,
                concurrency_id="gpu",
            )

        with gr.Tab("🗣️ SadTalker 對嘴影片"):
            gr.Markdown(
                "上傳一張人像 + 一段語音，產生嘴型跟著語音動的說話影片（mp4，含聲音）。"
                "**來源人像僅限虛構/AI生成的臉（例如上面生成的 anchor），禁止上傳真人照片。**"
                "跑在 SadTalker 自己的 Python 環境，不用開 ComfyUI；如果 ComfyUI 正開著，會先請它釋放顯存。"
                "256 比較快（約 2-3GB VRAM），512 比較清楚（約 4-6GB）。半身/全身圖建議選 full + 勾 still。"
            )
            with gr.Row():
                with gr.Column():
                    talk_image = gr.Image(label="來源人像（僅限虛構/AI生成，禁止上傳真人照片）", type="filepath")
                    talk_audio = gr.Audio(label="語音檔（wav/mp3 等，會自動轉成 16kHz wav）", type="filepath")
                    with gr.Row():
                        talk_size = gr.Radio(list(talking_head.SIZES), value=512, label="臉部渲染尺寸")
                        talk_preprocess = gr.Dropdown(list(talking_head.PREPROCESS_MODES), value="crop",
                                                      label="範圍（crop 只有臉；full/extfull 貼回整張圖）")
                    with gr.Row():
                        talk_still = gr.Checkbox(value=False, label="still（頭部幾乎不動，搭配 full 使用）")
                        talk_enhancer = gr.Dropdown([TALK_ENHANCER_NONE] + list(talking_head.ENHANCERS), value=TALK_ENHANCER_NONE,
                                                    label="臉部修復（gfpgan 第一次使用會下載 ~348MB）")
                    with gr.Row():
                        talk_expression = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="表情/嘴型幅度")
                        talk_pose_style = gr.Slider(0, 45, value=0, step=1, label="頭部動作風格")
                    talk_btn = gr.Button("生成說話影片", variant="primary")
                with gr.Column():
                    talk_output = gr.Video(label="結果")

            talk_btn.click(
                generate_talking_head_ui,
                inputs=[talk_image, talk_audio, talk_size, talk_preprocess, talk_still, talk_expression,
                        talk_enhancer, talk_pose_style],
                outputs=talk_output,
                concurrency_id="gpu",
            )

        with gr.Tab("📹 SVD 圖生影片"):
            gr.Markdown(
                "上傳一張既有的圖片（例如上面生成的 anchor 或 dataset 照片），用 Stable Video Diffusion "
                "直接讓那張圖動起來——跟「AnimateDiff 動態影片」分頁不同，這裡不是用 prompt 生成新內容，"
                "是直接讓那張圖本身動起來。**這條路線沒有 FaceID 鎖臉，動態幅度大時臉容易變形/融化**，"
                "低解析度/低影格數，只是「動起來測試」用途，非最終畫質。想要臉部穩定的動態影片，"
                "請用「AnimateDiff 動態影片」分頁。"
            )
            with gr.Row():
                with gr.Column():
                    svd_character = gr.Dropdown(sorted(gc.CHARACTERS), label="角色（決定輸出資料夾位置）")
                    svd_init_image = gr.Image(label="要配上動作的圖片", type="filepath")
                    svd_seed = gr.Number(value=6001, label="Seed", precision=0)
                    with gr.Row():
                        svd_frames = gr.Slider(6, 25, value=client.VIDEO_FRAMES, step=1, label="影格數")
                        svd_fps = gr.Slider(2, 12, value=client.VIDEO_FPS, step=1, label="FPS")
                    svd_motion = gr.Slider(1, 255, value=client.MOTION_BUCKET_ID, step=1,
                                            label="動態強度（越高動作越大，但越容易變形/融化，人像建議偏低）")
                    svd_btn = gr.Button("生成影片（SVD）", variant="primary")
                with gr.Column():
                    svd_output = gr.Video(label="結果")

            svd_btn.click(
                generate_video_svd,
                inputs=[svd_character, svd_init_image, svd_seed, svd_frames, svd_fps, svd_motion],
                outputs=svd_output,
                concurrency_id="gpu",
            )

        with gr.Tab("🎞️ 批次生成 GIF"):
            gr.Markdown(
                "先生成第一張（完整生成），後面每一張都是拿**同一張第一張的圖**用低 denoise 的 img2img 去微調"
                "（同一個 prompt，只有 seed 不同），組成一個會動的 GIF——同一個人、同一個姿勢、同一個場景，"
                "只有細節隨每張的 seed 有小幅變化，靠「動作幅度」滑桿控制變化大小。"
                "**這不是像 AnimateDiff 那樣有時序關聯的平滑動態**，是同一張圖反覆微調出來的效果，"
                "沒有動作連貫的「進行中」的感覺，比較像是原地小幅度的浮動/晃動；想要真正平滑、有動作進展的"
                "動態影片，請用「AnimateDiff 動態影片」分頁。生成速度比 AnimateDiff 快很多"
                "（每張都是普通靜態圖，沒有影格數的額外負擔）。"
            )
            with gr.Row():
                with gr.Column():
                    gif_character = gr.Dropdown(CHARACTER_CHOICES, label="角色（選填，選了會用該角色身分）", value=NO_CHARACTER)
                    gif_anchor = gr.Dropdown([], label="Anchor 圖（選了角色才需要）")
                    gif_anchor_preview = gr.Image(label="Anchor 圖預覽", interactive=False, height=200)
                    gif_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="例如: standing in a park, looking at the camera")
                    gif_translate_btn = gr.Button("翻譯成英文（已經是英文按下去不會被改動）")
                    gif_negative_prompt = gr.Textbox(label="額外負面詞（選填）", lines=1)
                    with gr.Accordion("風格正/負面詞（進階，跟「自訂生圖」分頁獨立設定）", open=False):
                        gif_style_positive = gr.Textbox(label="風格正面詞", value=gc.REALISTIC_STYLE, lines=2)
                        gif_style_negative = gr.Textbox(label="風格負面詞", value=gc.REALISTIC_NEGATIVE, lines=2)
                    gif_checkpoint_choice = gr.Dropdown(
                        CHECKPOINT_CHOICES, value="cyberrealistic_pony",
                        label="Checkpoint 模型（選了角色/anchor 就只能 SDXL 系列，跟「自訂生圖」分頁的規則一樣）",
                    )
                    gif_lora_strength = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="寫實風格 LoRA 強度（pony 系建議 0）")
                    gif_tier = gr.Radio(["safe", "suggestive"], value="suggestive", label="內容分級（露骨內容依然封鎖）")
                    gif_seed = gr.Number(value=9000, label="起始 Seed（每張圖依序 +1）", precision=0)
                    with gr.Row():
                        gif_frames = gr.Slider(2, 20, value=8, step=1, label="張數")
                        gif_duration = gr.Slider(100, 1000, value=300, step=50, label="每張顯示時間（毫秒）")
                    gif_denoise = gr.Slider(
                        0.0, 1.0, value=gc.GIF_WIGGLE_DENOISE, step=0.05,
                        label="動作幅度（實測：0.3 幾乎看不出變化，0.7 姿勢已經明顯跑掉；0.45 是還沒驗證過的中間值，"
                              "看起來太靜止就往上調，姿勢跑掉就往下調）",
                    )
                    gif_ip_weight = gr.Slider(0.0, 3.0, value=client.IP_ADAPTER_WEIGHT, step=0.05, label="IP-Adapter 權重（有選角色才有作用）")
                    gif_btn = gr.Button("生成 GIF", variant="primary")
                with gr.Column():
                    gif_output = gr.Image(label="結果（GIF，會自動播放）")

            gif_character.change(refresh_anchors, inputs=gif_character, outputs=[gif_anchor, gif_anchor_preview])
            gif_anchor.change(lambda path: path, inputs=gif_anchor, outputs=gif_anchor_preview)
            gif_translate_btn.click(translate_prompt_to_english, inputs=gif_prompt, outputs=gif_prompt)
            gif_btn.click(
                generate_gif,
                inputs=[gif_character, gif_anchor, gif_prompt, gif_tier, gif_negative_prompt, gif_seed, gif_frames,
                        gif_duration, gif_denoise, gif_ip_weight, gif_style_positive, gif_style_negative,
                        gif_checkpoint_choice, gif_lora_strength],
                outputs=gif_output,
                concurrency_id="gpu",
            )

        with gr.Tab("🎯 圖片選擇生圖"):
            gr.Markdown(
                "**點圖就好，不用自己想 prompt。**選人物、姿勢、場景各一張，系統會依照你選的 checkpoint "
                "組出它聽得懂的敘述——Pony 系自動加品質標籤、SD1.5 自動加性別權重，"
                "而且在 SDXL/Pony 上姿勢是走 ControlNet 骨架、臉是走 FaceID，不是只用文字講。\n\n"
                "Z-Image 跟 SD1.5 沒有這兩個功能，選到那些 checkpoint 時姿勢和臉會**自動改成文字描述**，"
                "下方會明講哪些被降級了。送出的完整 prompt 會即時顯示在預覽欄，不用猜。"
            )
            pick_char_state = gr.State(None)
            pick_pose_state = gr.State(None)
            pick_scene_state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        gr.Markdown("### 1. 人物")
                        pick_char_clear = gr.Button("清除人物", size="sm", scale=0)
                    pick_char_gallery = gr.Gallery(
                        value=_gallery_value(picker_characters()), columns=5, height=250,
                        allow_preview=False, object_fit="cover", show_label=False, buttons=[],
                    )
                    _picker_skipped = [t for t in sorted(gc.CHARACTERS) if not gc.picker_anchor_path(t)]
                    if _picker_skipped:
                        gr.Markdown(
                            f"（{'、'.join(_picker_skipped)} 還沒有 anchor 圖，先不列在這裡——"
                            "用 `generate_character.py anchors` 產生後就會自動出現）"
                        )
                    with gr.Row():
                        gr.Markdown("### 2. 姿勢")
                        pick_pose_clear = gr.Button("清除姿勢", size="sm", scale=0)
                    pick_pose_gallery = gr.Gallery(
                        value=_gallery_value(picker_pose_items()), columns=6, height=300,
                        allow_preview=False, object_fit="contain", show_label=False, buttons=[],
                    )
                    with gr.Row():
                        gr.Markdown("### 3. 場景")
                        pick_scene_clear = gr.Button("清除場景", size="sm", scale=0)
                    pick_scene_gallery = gr.Gallery(
                        value=_gallery_value(picker_scene_items("safe")), columns=5, height=250,
                        allow_preview=False, object_fit="cover", show_label=False, buttons=[],
                    )
                with gr.Column(scale=2):
                    pick_checkpoint = gr.Dropdown(
                        CHECKPOINT_CHOICES, value=CHECKPOINT_DEFAULT, label="Checkpoint 模型",
                    )
                    pick_tier = gr.Radio(
                        ["safe", "suggestive"], value="safe",
                        label="內容分級（suggestive 會多出海灘/泳池等場景；露骨內容依然封鎖）",
                    )
                    # Distinct from every other tab's default (9000: 自訂生圖/GIF; 6001: AnimateDiff/SVD) -
                    # measured 2026-09-12: two GUI sessions both defaulting to 9000 submitted around the
                    # same time and one clobbered gui_seed9000.png before the other's client read it back,
                    # so a tab's own client displayed a completely different generation as its result.
                    # Different defaults across tabs reduce how often that collision fires; it doesn't
                    # eliminate it (nothing stops someone typing 9000 here too, or two picker tabs racing
                    # each other), since output filenames are seed-keyed rather than job-id-keyed.
                    pick_seed = gr.Number(value=4001, label="Seed", precision=0)
                    pick_extra = gr.Textbox(
                        label="補充描述（選填，中文也可以，送出前會自動翻成英文）", lines=2,
                        placeholder="例如：手上拿著紙杯、戴著耳機",
                    )
                    pick_preview = gr.Textbox(
                        label="實際送出的 prompt（唯讀，含這個 checkpoint 需要的標籤）",
                        lines=6, interactive=False,
                    )
                    pick_notice = gr.Markdown("選一個場景或姿勢（或填補充描述）就會在這裡看到實際送出的 prompt。")
                    pick_btn = gr.Button("生成", variant="primary")
                    pick_output = gr.Image(label="結果")

            _preview_inputs = [pick_char_state, pick_pose_state, pick_scene_state,
                               pick_checkpoint, pick_tier, pick_extra]
            _preview_outputs = [pick_preview, pick_notice]
            pick_char_gallery.select(select_picker_character, inputs=None, outputs=pick_char_state).then(
                update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_pose_gallery.select(select_picker_pose, inputs=None, outputs=pick_pose_state).then(
                update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_scene_gallery.select(select_picker_scene, inputs=[pick_tier], outputs=pick_scene_state).then(
                update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_char_clear.click(
                lambda: (None, gr.update(selected_index=None)), inputs=None,
                outputs=[pick_char_state, pick_char_gallery]).then(
                update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_pose_clear.click(
                lambda: (None, gr.update(selected_index=None)), inputs=None,
                outputs=[pick_pose_state, pick_pose_gallery]).then(
                update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_scene_clear.click(
                lambda: (None, gr.update(selected_index=None)), inputs=None,
                outputs=[pick_scene_state, pick_scene_gallery]).then(
                update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_tier.change(
                refresh_picker_scenes, inputs=pick_tier, outputs=[pick_scene_gallery, pick_scene_state]).then(
                update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_checkpoint.change(update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_extra.change(update_picker_preview, inputs=_preview_inputs, outputs=_preview_outputs)
            pick_btn.click(
                generate_from_picker,
                inputs=[pick_char_state, pick_pose_state, pick_scene_state, pick_checkpoint,
                        pick_tier, pick_seed, pick_extra],
                outputs=pick_output,
                concurrency_id="gpu",
            )

        with gr.Tab("☁️ 雲端影片"):
            gr.Markdown(
                "在雲端 GPU 上生成影片，不佔用本機這張 8GB 卡。**Wan 2.2 圖生影片**是新一代模型："
                "拿一張角色圖當第一幀，生成動作自然的 720p 影片（沒有 FaceID，身分靠第一幀維持）。"
                "**AnimateDiff 雲端高畫質**是本機同一套 FaceID 鎖臉流程，只是解除 8GB 的高清上限，"
                "只能跑在 RunPod。所有 prompt 一樣會套用年齡保護與內容分級的負面詞。"
            )
            cloud_status = gr.Markdown(cloud_status_markdown())
            with gr.Accordion("雲端設定（endpoint ID／模型；API 金鑰請用環境變數）", open=False):
                _cloud_settings = cloud_video.load_cloud_settings()
                cloud_endpoint = gr.Textbox(
                    label="RunPod Serverless endpoint ID", value=_cloud_settings["runpod"]["endpoint_id"],
                )
                cloud_replicate_model = gr.Textbox(
                    label="Replicate 模型（owner/name 或 owner/name:version；不符合安全條件的模型會被拒絕）",
                    value=_cloud_settings["replicate"]["model"],
                )
                cloud_save_btn = gr.Button("儲存設定")
            with gr.Row():
                with gr.Column():
                    cloud_provider = gr.Radio(CLOUD_PROVIDER_CHOICES, value="runpod", label="雲端平台")
                    cloud_job = gr.Radio([CLOUD_JOB_WAN, CLOUD_JOB_ANIMATEDIFF], value=CLOUD_JOB_WAN,
                                         label="工作類型")
                    cloud_character = gr.Dropdown(CHARACTER_CHOICES, value=NO_CHARACTER,
                                                  label="角色（選填，會加上角色外貌描述）")
                    cloud_first_frame = gr.Image(
                        label="第一幀／臉部參考圖（僅限虛構/AI生成，禁止上傳真人照片）", type="filepath",
                    )
                    with gr.Row():
                        cloud_use_anchor_btn = gr.Button("用角色的 anchor", size="sm")
                        cloud_use_picker_btn = gr.Button("用「圖片選擇生圖」的結果", size="sm")
                    cloud_prompt = gr.Textbox(
                        label="Prompt：描述要發生的動作（Wan 看得懂中文；AnimateDiff 會自動翻成英文）", lines=3,
                        placeholder="例如：她轉頭看向鏡頭微笑，頭髮被微風吹動",
                    )
                    cloud_negative = gr.Textbox(label="額外負面詞（選填，安全負面詞一定會加上）", lines=1)
                    cloud_tier = gr.Radio(["safe", "suggestive"], value="safe", label="內容分級（露骨內容依然封鎖）")
                    with gr.Row():
                        cloud_seed = gr.Number(value=7001, label="Seed", precision=0)
                        cloud_timeout = gr.Slider(300, 2400, value=1800, step=60,
                                                  label="最長等待秒數（超過會自動取消雲端工作）")
                    with gr.Row():
                        cloud_preset = gr.Radio(list(CLOUD_PRESETS), value=CLOUD_PRESET_DEFAULT,
                                                label="畫面方向（只影響 Wan；AnimateDiff 固定 512² 放大到 1024）")
                        cloud_length = gr.Radio(list(CLOUD_LENGTHS), value=CLOUD_LENGTH_DEFAULT,
                                                label="長度（只影響 Wan；AnimateDiff 固定 16 幀）")
                    with gr.Row():
                        cloud_btn = gr.Button("在雲端生成", variant="primary")
                        cloud_cancel_btn = gr.Button("取消雲端工作", variant="stop")
                with gr.Column():
                    cloud_output = gr.Video(label="結果")
                    cloud_info = gr.Markdown()

            cloud_save_btn.click(save_cloud_config, inputs=[cloud_endpoint, cloud_replicate_model],
                                 outputs=cloud_status)
            cloud_use_anchor_btn.click(cloud_anchor_for, inputs=cloud_character, outputs=cloud_first_frame)
            cloud_use_picker_btn.click(lambda image: image, inputs=pick_output, outputs=cloud_first_frame)
            cloud_btn.click(
                generate_cloud_video,
                inputs=[cloud_provider, cloud_job, cloud_character, cloud_first_frame, cloud_prompt, cloud_tier,
                        cloud_negative, cloud_seed, cloud_preset, cloud_length, cloud_timeout],
                outputs=[cloud_output, cloud_info],
                # Not "gpu": a cloud job doesn't touch the local card, and a 10-minute remote run must
                # not queue local generations behind it. One cloud job at a time keeps spend bounded.
                concurrency_id="cloud",
                concurrency_limit=1,
            )
            cloud_cancel_btn.click(cancel_cloud_video, inputs=None, outputs=None)

if __name__ == "__main__":
    # 7861, not Gradio's default 7860 - kohya_ss's own training GUI (kohya_gui.py, this repo's parent
    # tool) also defaults to 7860 and can auto-start on this machine, silently stealing the port and
    # making this app unreachable at the URL people expect (looks like "the page is missing the anchor
    # upload" when it's actually a different app entirely serving that port).
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
