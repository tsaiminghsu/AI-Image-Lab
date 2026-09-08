"""Local Gradio GUI for generate_character.py's custom-prompt mode.

Runs entirely on localhost (server_name="127.0.0.1", share=False) - this
handles suggestive-tier content and must never be exposed to the public
internet via Gradio's share tunnel.

Requires the ComfyUI server to already be running (see comfyui_client.py's
module docstring for the start command) - this GUI is just a client, same
as generate_character.py's CLI.
"""

import glob
import os

import gradio as gr

import caption_image
import comfyui_client as client
import generate_character as gc
import pose_skeletons
import translate_prompt

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
CHECKPOINT_CHOICES = [CHECKPOINT_DEFAULT] + sorted(k for k in client.CHECKPOINTS if k != "cyberrealistic_pony")

ANIMATEDIFF_CHECKPOINT_DEFAULT = "sd15_base (預設)"
ANIMATEDIFF_CHECKPOINT_CHOICES = [ANIMATEDIFF_CHECKPOINT_DEFAULT] + sorted(
    k for k in client.ANIMATEDIFF_CHECKPOINTS if k != "sd15_base"
)

MOTION_LORA_NONE = "(無 - 純 prompt 描述動作)"
MOTION_LORA_CHOICES = [MOTION_LORA_NONE] + sorted(client.ANIMATEDIFF_MOTION_LORAS)


def generate(character, anchor, custom_anchor, prompt, tier, negative_prompt, seed, ip_adapter_weight,
             pose_reference, pose_library, controlnet_strength, resolution, use_hq, hires_denoise,
             character_lora_strength, use_facedetailer, face_denoise, hand_denoise, facedetailer_backend,
             style_positive, style_negative, checkpoint_choice, lora_strength):
    if not prompt.strip():
        raise gr.Error("請輸入 prompt")
    trigger = None if character == NO_CHARACTER else character
    anchor_path = custom_anchor or anchor
    if trigger and not anchor_path:
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

    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates")
    stem = f"gui_seed{int(seed)}"
    backend = "mediapipe" if facedetailer_backend == FACEDETAILER_BACKEND_MEDIAPIPE else "yolo"
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


def generate_video_animatediff(character, face_ref, prompt, tier, negative_prompt, seed, ip_adapter_weight,
                                facedetailer_denoise, frames, fps, width, height, style_positive, style_negative,
                                checkpoint_choice, motion_lora_choice, motion_lora_strength):
    if not prompt.strip():
        raise gr.Error("請輸入 prompt")
    if not face_ref:
        raise gr.Error("請上傳臉部參考圖（用於 FaceID 鎖定長相）— 僅限虛構/AI生成的臉，禁止上傳真人照片")
    trigger = None if character == NO_CHARACTER else character
    checkpoint = None if checkpoint_choice == ANIMATEDIFF_CHECKPOINT_DEFAULT else checkpoint_choice
    motion_lora = None if motion_lora_choice == MOTION_LORA_NONE else motion_lora_choice
    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates", "videos")
    gc.gen_video_animatediff(prompt, negative_prompt, tier, trigger, face_ref, out_dir, int(seed),
                              ip_adapter_weight=ip_adapter_weight,
                              facedetailer_denoise=facedetailer_denoise,
                              frames=int(frames), fps=int(fps), width=int(width), height=int(height),
                              style_positive=style_positive, style_negative=style_negative,
                              checkpoint=checkpoint,
                              motion_lora=motion_lora, motion_lora_strength=motion_lora_strength)
    return os.path.join(out_dir, f"animatediff_seed{int(seed)}.webm")


def generate_video_svd(character, init_image, seed, frames, fps, motion_bucket_id):
    if not init_image:
        raise gr.Error("請上傳要配上動作的圖片")
    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates", character, "videos")
    gc.gen_video(character, init_image, out_dir, int(seed), int(frames), int(fps), int(motion_bucket_id))
    return os.path.join(out_dir, f"video_seed{int(seed)}.webm")


def generate_gif(character, anchor, prompt, tier, negative_prompt, seed, frame_count, duration_ms, denoise,
                  ip_adapter_weight, style_positive, style_negative, checkpoint_choice, lora_strength):
    if not prompt.strip():
        raise gr.Error("請輸入 prompt")
    trigger = None if character == NO_CHARACTER else character
    if trigger and not anchor:
        raise gr.Error("選了角色就要選一張 anchor 圖")
    checkpoint = None if checkpoint_choice == CHECKPOINT_DEFAULT else checkpoint_choice
    if checkpoint in client.SD15_CHECKPOINTS:
        raise gr.Error("這個 checkpoint 是 SD1.5——批次 GIF 的每一張都要靠 IP-Adapter 鎖同一張臉，SD1.5 沒接這個功能，換一個 SDXL 系列的 checkpoint")
    out_dir = os.path.join(os.path.dirname(__file__), "reference_candidates")
    return gc.gen_gif(prompt, negative_prompt, tier, trigger, anchor, out_dir, int(seed), int(frame_count),
                       ip_adapter_weight, duration_ms=int(duration_ms), denoise=denoise,
                       style_positive=style_positive, style_negative=style_negative,
                       checkpoint=checkpoint, lora_strength=lora_strength)


with gr.Blocks(title="AI Image Lab") as demo:
    gr.Markdown("# AI Image Lab - 自訂生圖\n先確定 ComfyUI server 已經在跑（127.0.0.1:8188）。")
    with gr.Row():
        with gr.Column():
            character = gr.Dropdown(CHARACTER_CHOICES, label="角色（選填，選了會用該角色身分）", value=NO_CHARACTER)
            anchor = gr.Dropdown([], label="Anchor 圖（選了角色才需要）")
            anchor_preview = gr.Image(label="Anchor 圖預覽", interactive=False, height=200)
            custom_anchor = gr.Image(
                label="或自行上傳身分參考圖（會覆蓋上面的 Anchor 選擇）— 僅限虛構/AI生成的角色照，禁止上傳真人照片",
                type="filepath",
            )
            with gr.Accordion("依圖片產生 Prompt（上傳一張圖，自動描述成英文 prompt）", open=False):
                caption_upload = gr.Image(label="上傳圖片", type="filepath")
                caption_btn = gr.Button("產生 Prompt")
                caption_output = gr.Textbox(label="產生的描述（僅供參考，可自行編輯後使用）", lines=2, interactive=False)
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
                "**什麼時候要切換 checkpoint：**\n"
                "- 需要「Anchor 身分鎖定」「骨架姿勢控制」或「臉部/手部精修」任一項 → 只能選 SDXL 系列"
                "（juggernaut / pony / cyberrealistic_pony / pony_realism），這三個功能都是接在 SDXL 專用模型檔上，"
                "選 SD1.5 送出時會直接跳錯誤。\n"
                "- 只要純文字生圖、想要更自然語言的 prompt、或想要 512×768 左右的原生解析度 → 可以切到 SD1.5 系列"
                "（realistic_vision / cyberrealistic），但切過去後 anchor / 姿勢控制 / 精修都會被鎖住，畫布比例也只能選「自動」。\n"
                "- pony / cyberrealistic_pony / pony_realism 需要的 `score_9, score_8_up, score_7_up` 品質 tag 前綴"
                "會自動加上，Prompt 欄位一樣打自然語言就好，不用自己記得加。"
                "下面的「寫實風格 LoRA 強度」建議切到 0（這顆 LoRA 是針對 juggernaut 調的，套在 pony 系會打架）。"
            )
            checkpoint_choice = gr.Dropdown(
                CHECKPOINT_CHOICES, value="cyberrealistic_pony", label="Checkpoint 模型",
            )
            lora_strength = gr.Slider(
                0.0, 3.0, value=0.0, step=0.1,
                label="寫實風格 LoRA 強度（針對 Juggernaut 調的，換成 pony 等其他 checkpoint 時建議調到 0）",
            )
            tier = gr.Radio(["safe", "suggestive"], value="suggestive", label="內容分級（suggestive 上限跟 test-suggestive 一樣，露骨內容依然封鎖）")
            seed = gr.Number(value=9000, label="Seed", precision=0)
            ip_weight = gr.Slider(0.0, 3.0, value=client.IP_ADAPTER_WEIGHT, step=0.05, label="IP-Adapter 權重（FaceID 量表，有選角色才有作用）")
            btn = gr.Button("生成", variant="primary")
        with gr.Column():
            output = gr.Image(label="結果")

    character.change(refresh_anchors, inputs=character, outputs=[anchor, anchor_preview])
    anchor.change(lambda path: path, inputs=anchor, outputs=anchor_preview)
    checkpoint_choice.change(reset_resolution_for_sd15, inputs=checkpoint_choice, outputs=resolution)
    caption_btn.click(caption_uploaded_image, inputs=caption_upload, outputs=[caption_output, prompt])
    translate_btn.click(translate_prompt_to_english, inputs=prompt, outputs=prompt)
    btn.click(generate, inputs=[character, anchor, custom_anchor, prompt, tier, negative_prompt, seed, ip_weight, pose_reference, pose_library, controlnet_strength, resolution, use_hq, hires_denoise, character_lora_strength, use_facedetailer, face_denoise, hand_denoise, facedetailer_backend, style_positive, style_negative, checkpoint_choice, lora_strength], outputs=output)

    gr.Markdown("---\n## AnimateDiff 動態影片（SD1.5 + FaceID 鎖臉 + 逐幀臉部精修）")
    gr.Markdown(
        "跟上面的靜態圖是分開的 workflow，用 SD1.5 + AnimateDiff motion module 生成短動態影片，"
        "並用 IPAdapter-FaceID 鎖住整段影片的臉部身分、每一幀再跑一次 FaceDetailer 修臉——"
        "解決純 SVD img2vid（見 README「幫已有的圖片配上動作」）常見的臉部變形/融化問題。"
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
                video_frames = gr.Slider(8, 16, value=client.ANIMATEDIFF_FRAMES, step=1, label="影格數（motion module 訓練上限 16）")
                video_fps = gr.Slider(4, 16, value=client.ANIMATEDIFF_FPS, step=1, label="FPS")
            with gr.Row():
                video_width = gr.Number(value=client.ANIMATEDIFF_WIDTH, label="寬", precision=0)
                video_height = gr.Number(value=client.ANIMATEDIFF_HEIGHT, label="高")
            video_facedetailer_denoise = gr.Slider(0.0, 1.0, value=client.FACEDETAILER_DENOISE, step=0.05, label="逐幀臉部精修強度")
            video_checkpoint_choice = gr.Dropdown(
                ANIMATEDIFF_CHECKPOINT_CHOICES, value=ANIMATEDIFF_CHECKPOINT_DEFAULT,
                label="Checkpoint 模型（必須是 SD1.5，跟上面圖片區塊的選項是分開的清單）",
            )
            with gr.Accordion("風格正/負面詞（進階，跟上面圖片區塊獨立設定）", open=False):
                video_style_positive = gr.Textbox(label="風格正面詞", value=gc.REALISTIC_STYLE, lines=2)
                video_style_negative = gr.Textbox(label="風格負面詞", value=gc.REALISTIC_NEGATIVE, lines=2)
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
                video_motion_lora, video_motion_lora_strength],
        outputs=video_output,
    )

    gr.Markdown("---\n## SVD 圖生影片（幫已有的圖片配上動作）")
    gr.Markdown(
        "上傳一張既有的圖片（例如上面生成的 anchor 或 dataset 照片），用 Stable Video Diffusion "
        "直接讓那張圖動起來——跟上面的 AnimateDiff 不同，這裡不是用 prompt 生成新內容，"
        "是直接讓那張圖本身動起來。**這條路線沒有 FaceID 鎖臉，動態幅度大時臉容易變形/融化**，"
        "低解析度/低影格數，只是「動起來測試」用途，非最終畫質。想要臉部穩定的動態影片，"
        "請用上面的「AnimateDiff 動態影片」區塊。"
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
    )

    gr.Markdown("---\n## 批次生成 GIF（同一個場景微幅變化，組成動畫）")
    gr.Markdown(
        "先生成第一張（完整生成），後面每一張都是拿**同一張第一張的圖**用低 denoise 的 img2img 去微調"
        "（同一個 prompt，只有 seed 不同），組成一個會動的 GIF——同一個人、同一個姿勢、同一個場景，"
        "只有細節隨每張的 seed 有小幅變化，靠「動作幅度」滑桿控制變化大小。"
        "**這不是像 AnimateDiff 那樣有時序關聯的平滑動態**，是同一張圖反覆微調出來的效果，"
        "沒有動作連貫的「進行中」的感覺，比較像是原地小幅度的浮動/晃動；想要真正平滑、有動作進展的"
        "動態影片，請用上面的「AnimateDiff 動態影片」區塊。生成速度比 AnimateDiff 快很多"
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
            with gr.Accordion("風格正/負面詞（進階，跟上面「自訂生圖」區塊獨立設定）", open=False):
                gif_style_positive = gr.Textbox(label="風格正面詞", value=gc.REALISTIC_STYLE, lines=2)
                gif_style_negative = gr.Textbox(label="風格負面詞", value=gc.REALISTIC_NEGATIVE, lines=2)
            gif_checkpoint_choice = gr.Dropdown(
                CHECKPOINT_CHOICES, value="cyberrealistic_pony",
                label="Checkpoint 模型（選了角色/anchor 就只能 SDXL 系列，跟上面「自訂生圖」區塊的規則一樣）",
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
    )

if __name__ == "__main__":
    # 7861, not Gradio's default 7860 - kohya_ss's own training GUI (kohya_gui.py, this repo's parent
    # tool) also defaults to 7860 and can auto-start on this machine, silently stealing the port and
    # making this app unreachable at the URL people expect (looks like "the page is missing the anchor
    # upload" when it's actually a different app entirely serving that port).
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
