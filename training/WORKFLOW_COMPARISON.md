# Workflow 模板差異對照表

資料來源：直接讀 `training/workflow_template*.json`（2026-10-02 當下的內容）。
「預設值」是 JSON 裡寫的值；實際執行時 `comfyui_client` 會依 caller 參數改寫
（checkpoint、尺寸、steps、cfg、seed 都會被覆蓋，`REPLACE_*` 一定會被替換）。
這份只比較模板本身，沒有跑過任何生成，所以沒有速度或畫質數字。

## 1. 總覽

| 模板 | 用途 | 節點數 | 模型家族 | 輸出 |
|---|---|---|---|---|
| `workflow_template.json` | 文字生圖＋FaceID 身分 | 11 | SDXL | PNG |
| `_txt2img` | 純文字生圖（無身分） | 8 | SDXL | PNG |
| `_controlnet` | FaceID＋OpenPose 姿勢 | 15 | SDXL | PNG |
| `_facedetailer` | FaceID＋臉部精修（YOLO 偵測） | 15 | SDXL | PNG |
| `_mediapipe_facedetailer` | FaceID＋臉部精修（MediaPipe 偵測） | 15 | SDXL | PNG |
| `_hq` | 完整高清：FaceID＋姿勢＋角色 LoRA＋精修＋放大 | 26 | Pony / SDXL | PNG |
| `_img2img` | 圖生圖（低 denoise）＋FaceID | 12 | SDXL | PNG |
| `_txt2img_sd15` | SD1.5 純文字生圖 | 7 | SD1.5 | PNG |
| `_txt2img_zimage` | Z-Image Turbo 純文字生圖 | 10 | Z-Image（Lumina 系，AuraFlow 取樣） | PNG |
| `_img2vid` | SVD 圖生影片 | 7 | SVD | WEBM (vp9) |
| `_animatediff_facedetailer` | AnimateDiff 影片＋FaceID＋逐幀精修＋放大 | 24 | SD1.5 | MP4 (h264) |
| `_wan_i2v` | Wan 2.2 圖生影片 | 12 | Wan 2.2 | MP4 (h264) |
| `_rife_interp` | 影片補幀 | 5 | RIFE | MP4 (h264) |

## 2. 圖片流程：功能開關

✅ 有、— 沒有。

| 模板 | FaceID | 姿勢 ControlNet | 臉部精修 | 角色 LoRA | 照片風格 LoRA | 放大 | 輸入圖 |
|---|---|---|---|---|---|---|---|
| 基本（`workflow_template`） | ✅ | — | — | — | ✅ | — | 臉參考圖 |
| `_txt2img` | — | — | — | — | ✅ | — | — |
| `_controlnet` | ✅ | ✅ OpenPose | — | — | ✅ | — | 臉參考圖＋姿勢圖 |
| `_facedetailer` | ✅ | — | ✅ YOLO | — | ✅ | — | 臉參考圖 |
| `_mediapipe_facedetailer` | ✅ | — | ✅ MediaPipe | — | ✅ | — | 臉參考圖 |
| `_hq` | ✅ | ✅ OpenPose | ✅ | ✅（`REPLACE_CHARACTER_LORA`） | 權重 0.0 | ✅ 放大模型＋縮放＋VAEEncode 回精修 | 臉參考圖＋姿勢圖 |
| `_img2img` | ✅ | — | — | — | ✅ | — | 來源圖＋臉參考圖 |
| `_txt2img_sd15` | — | — | — | — | — | — | — |
| `_txt2img_zimage` | — | — | — | — | — | — | — |

「照片風格 LoRA」都是 `sdxl_photorealistic_slider_v1-0`。

## 3. 圖片流程：預設參數

| 模板 | Checkpoint / UNET | 尺寸 | steps | cfg | sampler / scheduler | denoise |
|---|---|---|---|---|---|---|
| 基本／`_txt2img`／`_controlnet`／`_facedetailer`／`_mediapipe_facedetailer` | juggernaut_xl_v9_photo | 1024×1024 | 30 | 6.0 | dpmpp_2m / karras | 1.0 |
| `_hq` | CyberRealisticPony_V18.0_F16 | 832×832 | 24（第一階段）、20（精修） | 6.0 | dpmpp_2m / karras | 1.0、精修 0.4 |
| `_img2img` | juggernaut_xl_v9_photo | 取自輸入圖 | 30 | 6.0 | dpmpp_2m / karras | 0.3 |
| `_txt2img_sd15` | `REPLACE_CHECKPOINT` | 512×768 | 30 | 7.0 | dpmpp_2m / karras | 1.0 |
| `_txt2img_zimage` | `REPLACE_UNET`（另有 CLIPLoader、VAELoader） | 1024×1024 | 8 | 2.0 | res_multistep / simple | 1.0 |

- 照片風格 LoRA 在 SDXL 基本流程的強度是 2.5（model、clip 都是），在 `_hq` 是 0.0。
- Z-Image 模板沒有負面詞以外的身分／姿勢節點，對應 CLAUDE.md「只支援純文字生圖」。
- `_hq` 是「一個模板、各階段可選」：`build_hq` 依參數把不用的階段連線改接並移除節點
  （角色 LoRA＝節點 14、FaceID＝10/11/12、ControlNet＝20-23、放大＝30-35、臉部精修＝40-43）。
  所以第 2 節 `_hq` 那一列的 ✅ 是「模板具備、實際可選」，不是每次都會跑。
  照片風格 LoRA（節點 13）在模板裡是權重 0.0 但節點還在，註解寫明是 Pony 的預設。

## 4. 影片／後製流程

| 模板 | 輸入 | 核心節點 | 預設取樣 | 輸出設定 |
|---|---|---|---|---|
| `_img2vid`（SVD） | 一張圖 | ImageOnlyCheckpointLoader、SVD_img2vid_Conditioning、VideoLinearCFGGuidance | 25 步、cfg 2.5、euler/karras | WEBM vp9、6 fps、crf 40 |
| `_animatediff_facedetailer` | 臉參考圖（文字驅動） | AnimateDiffLoaderGen1、16 幀 512×512、FaceID、DetailerForEachPipeForAnimateDiff、放大 | 20 步、cfg 7.5、dpmpp_2m/karras；精修階段 denoise 0.4 | MP4 h264、crf 20 |
| `_wan_i2v` | 一張圖 | UNETLoader、Wan22ImageToVideoLatent、ModelSamplingSD3、CreateVideo | 20 步、cfg 5.0、uni_pc/simple | MP4 h264、crf 20 |
| `_rife_interp` | 一支影片 | LoadVideo、GetVideoComponents、RIFE VFI、CreateVideo | — | MP4 h264、crf 18 |

## 5. 差異重點

1. **身分來源**：所有 SDXL 圖片模板（除 `_txt2img`）都靠 IPAdapter FaceID；SD1.5、Z-Image、SVD、Wan 沒有。
2. **姿勢**：只有 `_controlnet` 和 `_hq` 有 OpenPose ControlNet。
3. **兩個精修偵測器**：`_facedetailer` 用 Ultralytics YOLO；`_mediapipe_facedetailer` 用 MediaPipe 臉網格轉 SEGS，
   兩者其餘節點一樣，差別只在偵測方式。
4. **`_hq` 是唯一用 Pony checkpoint、唯一接角色 LoRA、唯一帶放大的圖片模板**，解析度也降到 832 以配合 Pony。
5. **取樣設定分三群**：SDXL（dpmpp_2m、cfg 6）、SD1.5 與 AnimateDiff（cfg 7～7.5）、新式 flow 模型
   Z-Image（8 步、cfg 2、res_multistep）與 Wan（uni_pc）。
6. **影片三條路線**：SVD 最輕、輸出最粗（6 fps、crf 40）；AnimateDiff 帶身分與逐幀精修；Wan 是最新的圖生影片；
   RIFE 不生成，只補幀。

## 6. 實際執行時的差異（`comfyui_client.py` 改寫後）

模板 JSON 只是起點。每個 `submit_*` 函式送出前會覆蓋參數、增減節點，所以實際跑的圖和第 3、4 節的預設值不一定相同。
以下依 2026-10-02 的程式碼整理，同樣沒有實際執行驗證。

### 6.1 各流程送出前的改寫

| 流程（函式） | 執行時覆蓋的值 | 增減節點 | 逾時 |
|---|---|---|---|
| 基本 FaceID（`submit_generation`） | checkpoint、LoRA 強度、FaceID 圖＋preset `FACEID PLUS V2`＋權重、尺寸、seed/steps/cfg；sampler 固定 dpmpp_2m / karras | 無 | 900 秒 |
| `_txt2img`（`submit_txt2img_generation`） | 同上，沒有 FaceID | 無 | 900 秒 |
| `_img2img`（`submit_img2img_generation`） | 另帶 denoise 和來源圖；尺寸跟著來源圖，不是參數 | 無 | 900 秒 |
| `_controlnet`（`submit_generation_with_pose`） | 另帶 ControlNet 模型 `control-lora-openposeXL2-rank256` 與強度（預設 0.8） | 無 | 900 秒 |
| `_facedetailer`／`_mediapipe_facedetailer` | 精修節點的 cfg、sampler、denoise（預設 0.5） | 無 | 900 秒 |
| `_hq`（`submit_generation_hq`） | 第一階段尺寸另算（見 6.2）、角色 LoRA 強度 0.8、放大模型 `4x-UltraSharp`、各階段 seed/cfg/sampler | **依參數移除** LoRA／FaceID／ControlNet／放大／精修（見 6.2） | 900 秒 |
| `_txt2img_sd15` | checkpoint 必填、尺寸預設 512×768、cfg 7.0 | 無 | 900 秒 |
| `_txt2img_zimage` | UNET／文字編碼器／VAE 三個檔名、shift 3.0、尺寸對齊 16、cfg 取 max(cfg, 1.5)；sampler 固定 res_multistep / simple | 無 | 1200 秒 |
| `_img2vid`（SVD） | 512×512、14 幀、6 fps、motion_bucket_id 15、30 步、cfg 2.5、euler / karras；`min_cfg` 1.0 | 無 | 1800 秒 |
| `_animatediff_facedetailer` | 見 6.3 | 見 6.3 | 2400 秒 |
| `_wan_i2v` | 三個檔名、shift 8.0、尺寸對齊 32（最小 256）、幀數對齊 4n+1、1280×704、121 幀、24 fps、20 步、cfg 5.0、uni_pc / simple | 無 | 2400 秒 |
| `_rife_interp` | 影片檔名、`rife47.pth`、倍率（至少 2）、輸出 fps＝來源 fps × 倍率 | 無；沒有 `RIFE VFI` 節點會先報錯 | 900 秒 |

多數 SDXL 流程的 sampler / scheduler 是由程式覆蓋成常數（dpmpp_2m / karras），改模板 JSON 裡的值不會生效。

### 6.2 `_hq` 的可選階段

`_hq` 是主要路徑（`generate_character.py` 的生圖、`pose_pack.py` 都走它）。參數決定哪些階段存在：

| 參數 | 關掉時 | 預設 |
|---|---|---|
| `character_lora` | 移除節點 14 | 沒有（11 個角色目前都沒有 LoRA） |
| `ip_adapter_image_filename` | 移除 FaceID（10/11/12） | 沒有圖就不走 FaceID |
| `pose_image_filename` | 移除 ControlNet（20–23） | 沒有姿勢圖就不走；是骨架圖時再移除預處理節點 21 |
| `hires` | 移除放大（30–35） | 開：放大模型→縮到 1.5 倍→resample，denoise 0.4、20 步 |
| `use_facedetailer` | 移除精修（40–43） | 開：臉 denoise 0.4／guide 768，手 denoise 0.35／guide 512 |

- **第一階段尺寸**：輸出 1024×1024 → 先跑 832×832；832×1216 → 704×1024；1216×832 → 1024×704；其他尺寸取 輸出÷1.5 再對齊 64。
  最後輸出是第一階段的 1.5 倍（例如 1248×1248）。
- 基礎步數 24。照片風格 LoRA 預設 0.0（Pony 用）。
- 有角色 LoRA 時，呼叫端會把 FaceID 權重降到約 0.7（LoRA 負責身分、FaceID 只修正偏移）。
- FaceID 的 InsightFace 預設跑在 CPU（`INSIGHTFACE_PROVIDER`），因為跑 GPU 會讓 ComfyUI 的 VRAM 計算失準而變慢。

### 6.3 AnimateDiff 實際圖形

模板只有 24 個節點，實際執行會加減：

- 預設 checkpoint 是 Realistic Vision（SD1.5），motion module `mm_sd_v15_v2`；16 幀、512×512、20 步、cfg 7.5、
  dpmpp_2m / karras；取樣 fps 8；輸出 h264 crf 20。
- **預設還會跑 hires 和放大**，模板本身沒顯示這點：hires 1.5 倍（像素上限 768×768）、denoise 0.4、10 步，
  再逐幀 ESRGAN 放大到長邊 1024。精修步數 12、denoise 0.45。
- 可選：RIFE 補幀（倍率 1/2/4）、相機運動 LoRA（zoom／pan／tilt／rolling）、motion scale（≠1.0 才加節點）。
- 關掉的階段（精修、hires、放大）會把節點移除。
- **LCM 快速模式**：`animatelcm` 換 motion module 並加 LoRA，或 `lcm_lora` 沿用 `mm_sd_v15_v2`；
  steps 8、hires 10、cfg 2.0、sampler lcm / sgm_uniform。三個 KSampler 要一起換，否則精修會出爛圖。
  `animatelcm` 的相機運動 LoRA 相容性沒有驗證過（程式標 `motion_lora_verified=False`）。

### 6.4 所有流程共通的改寫（`_submit_and_wait`）

1. **模型版本切換**（`apply_model_variants`）：4 個 SDXL／Pony checkpoint 若設為量化版且檔案存在，就換成 fp8 檔；
   **圖形裡有 ControlNet control-lora 時一律改回完整版**（量化版會出全黑圖）；找不到量化檔也退回完整版並提示。
   預設仍是完整版。
2. **cfg 下限**（`enforce_min_cfg`）：任何取樣節點的 cfg 低於 1.5 都會被拉到 1.5。
   原因是 cfg 1.0 時 ComfyUI 會跳過負面詞，年齡安全詞就失效。Z-Image 和 LCM 另有各自的 max() 下限。
3. **模式切換重置**：在 LCM 與非 LCM、或 Z-Image 換 prompt 之間切換時，先釋放 VRAM 再送（避免 8 GB 卡上換模型卡住）。
4. **雲端後端**：後端設為 runpod 時直接交給 `cloud_workflow`，本機的上述步驟改在 worker 端執行。
5. **單一佇列**：同一個行程內一次只送一個 job（`_CLIENT_LOCK`）。

### 6.5 能力限制（依流程）

| 流程 | FaceID | 姿勢 | 臉部精修 | 量化版 checkpoint |
|---|---|---|---|---|
| SDXL / Pony 各圖片流程 | ✅ | `_controlnet`、`_hq` | `_facedetailer`、`_mediapipe`、`_hq` | ✅（有姿勢骨架時強制完整版） |
| SD1.5 | — | — | — | — |
| Z-Image Turbo | — | — | — | 不適用（int8／fp8 是另外的權重檔） |
| SVD / Wan | — | — | — | — |
| AnimateDiff | ✅（SD1.5 版 FaceID） | — | ✅（逐幀） | — |

## 7. 這份對照表的限制

- 第 1～5 節只看模板 JSON 的靜態內容；第 6 節來自 `comfyui_client.py` 的程式碼閱讀，兩者都沒有實際執行驗證。
- 節點「連線」（誰接誰）沒有逐條比對，功能欄是由節點類型和程式註解推斷。
- 各預設值若之後改了常數（例如 `HIRES_SCALE`、`ANIMATEDIFF_*`），這份要跟著重寫。
- 節點契約（node id ↔ class_type）的單一真相是 `training/workflow_contracts.py`。
