# 變更記錄

每次疊代改了什麼、為什麼這樣改、實測數字如何。安裝步驟和用法在
[README.md](README.md)，這裡只放結論和對應章節連結。倒序排列，`xxxxxxx` 是 commit。

## 2026-09-12

### SDXL / Pony 模型可切換完整版 / 量化版 fp8（選用） `707cb23`

- **問題**：8 GB VRAM + PCIe gen3 x1，模型搬移是主要瓶頸；高清流程連系統 RAM 都會吃緊。
- **做法**：`training/quantize_models.py` 在本機把 4 個 SDXL / Pony checkpoint 轉成 ComfyUI
  的每層量化格式（只量化 transformer 的 Linear 層，約 86% UNet 參數），存成
  `<原檔名>.fp8q.safetensors`，原檔只讀不改。單純 cast 成 fp8 在 Turing 沒用，會被轉回
  fp16；每層量化格式才會被認成 mixed precision 而保持 fp8。改寫點只有
  `comfyui_client._submit_and_wait` 一處，優先序 `--variant` > `MODEL_VARIANT` > 設定檔 >
  完整版。GUI 有每個模型的切換面板，RunPod worker 固定用完整版。
- **實測**（cyberrealistic_pony，同 seed）：主模型上 GPU 4896 → 2791 MB；純文字生圖每張
  36-40 → 40-42 秒；GUI 預設高清每張 167 / 272 / 530 → 145 / 127 / 125 秒（完整版變慢是
  因為記憶體吃緊開始用分頁檔）；高清時 ComfyUI 記憶體 15.5-15.8 → 12.3-13.2 GB；身分
  相似度不變。juggernaut 兩版溫度相近時，取樣每步慢約 15%、整張時間一樣。
- **注意**：骨架姿勢（ControlNet control-lora）一律改用完整版，強制量化版會出全黑圖。
  同一個 seed 在兩版產出的圖不完全一樣（SSIM 0.76-0.95），要重現舊圖請用原本的版本。
- **文件**：README「3c. 量化版模型」、GUI「模型版本」、CLI「量化版模型」

## 2026-09-11

### Z-Image Turbo 純文字生圖 `950e7e7`

- **做法**：Z-Image Turbo（Tongyi-MAI，6B DiT，Apache 2.0）加進 CLI `--checkpoint
  z_image_turbo` 和 GUI 選單。刻意不放進 `CHECKPOINTS`：它要載三個檔案，而其他程式都把
  `CHECKPOINTS` 的值當成單一 checkpoint 檔。
- **實測**（1024²）：載入後 cfg 1 約 75 秒、cfg 2 約 155-195 秒；啟動後第一張要 5-7 分鐘，
  每次換 prompt 都要換 5.4 GB 的文字編碼器和 5.9 GB 的擴散模型。跟 juggernaut 對比，寫實度
  和手部較好、男性角色每張都是男生、看得懂中文也寫得出中文（杯子上的「早安」四張都對，
  SDXL 則畫成沒有人的茶杯）。
- **限制**：只有純文字生圖。沒有 IP-Adapter / FaceID、ControlNet、FaceDetailer 版本，所以
  anchor、姿勢、精修、HQ、GIF 都會拒絕並說明原因。用 CFG 2.0（下限 1.5）而不是官方範本的
  1.0，因為 cfg 1.0 時 ComfyUI 會跳過負面詞，年齡安全負面詞會靜默失效。
- **文件**：README「Z-Image Turbo」

### SD1.5 性別字加權重，男性角色不再被畫成女生 `09f346e`

- **問題**：SD1.5 Realistic Vision 在角色描述比較柔和時（「soft face」等），常把男性角色
  畫成女生；prompt 裡單一個「man」壓不過那些詞。
- **實測**（SD1.5 靜態圖、5 個男性角色 × 4 個 seed，用 InsightFace 判性別）：原本 10/20；
  負面詞加「woman, female」10/20（負面詞已有約 98 個 token，加了幾乎沒作用）；正向加
  「male, man, masculine」14/20；**`(man:1.3)` 19/20**。女性角色四種寫法都是 20/20。
- **做法**：SD1.5 的兩條路徑（AnimateDiff 影片、`gen_custom` 選 SD1.5）把性別字寫成
  `(man:1.3)` / `(woman:1.3)`（`SD15_GENDER_WEIGHT`）。SDXL 的 anchor / variations / 資料集
  prompt 不動，既有 seed 產出不變。影片驗證：jungi 男性影格 7/16 → 16/16、相似度
  0.581 → 0.691。
- **注意**：AnimateLCM（cfg 2）救不回來，jungi 加不加權重都還是女生；男性角色不建議用 LCM。
- **順帶修掉**：`generate_character.py` 還有 6 處用 `os.replace` 搬檔，輸出資料夾在別的
  磁碟時會噴 WinError 17，全部改用 `shutil.move`。
- **文件**：README「男性角色被畫成女生（SD1.5）」

### 切換 LCM / 一般取樣會產生雜訊的 bug `5bb12a3`

- **問題**：同一個 ComfyUI session 裡混用 LCM 和一般取樣，會生出純色雜訊。ComfyUI 啟動後
  先跑的那個模式正常，之後每次跑另一個模式都是雜訊。
- **實測**（taeoh、seed 6001、512 基底，動作量正常是 3-8）：LCM 先跑正常 4.48，接著 v2 是
  雜訊 114；`POST /free` 後 v2 正常 3.57，接著 LCM 雜訊 76。
- **做法**：偵測到模式切換就先 `/free` 重置（會丟掉快取的節點輸出）。離線和實機都驗過：
  LCM → v2 → LCM → SD1.5 靜態圖 → v2 全部正常，且剛好在三次切換時各重置一次。
- **文件**：README「2d. AnimateLCM 快速模式」

### AnimateDiff 臉部變形修正 `4021552`

- **原因一**：臉部精修從來沒有真的放大臉部。`crop_factor` 3.0 讓裁切區塊比整張影格還大，
  `max_size` 768 又把放大倍率壓回 1.0，等於拿整張影格重跑 20 步。改成裁臉框的 1.5 倍、
  `guide_size` 512、12 步、denoise 0.45，並補上漏傳的 `noise_mask_feather`。
- **原因二**：負面詞裡有 `symmetrical face`，等於把模型推向雙眼不對稱、下巴歪斜。影片路線
  改用拿掉這個詞的 `VIDEO_REALISTIC_NEGATIVE`，靜態圖不變。
- **實測**：修正前相似度反而最高（0.685），但逐幀拼圖看得出臉頰浮腫、嘟嘴——InsightFace
  只判「是不是同一個人」，對這種形變不敏感，所以分數只用來確認沒換臉。修正後臉頰和嘴型
  正常、動作量完全保留（5.08）。加強 FaceID 或調低 motion scale 會讓臉更定住，但主要是
  因為畫面不動了，所以維持預設。
- **新工具**：`training/face_similarity.py`，逐幀算臉部相似度並輸出標了分數的臉部拼圖，
  生成時加 `--face-report` 會自動跑。
- **文件**：README「臉部變形排查」

## 2026-09-10

### AnimateDiff 影片輸出優化 + SadTalker 對嘴影片 `ad1f664`

- **做法**：原本只有 512×512、16 幀的 vp9 webm。改成分階段輸出，每段關掉就從工作流程移除：
  1.5 倍高清第二段（面積上限 768²，仍經過 motion module）、影片臉部精修、逐幀 ESRGAN 放大
  到長邊 1024、選用 RIFE 補幀、h264 mp4 輸出。預設 checkpoint 從純 SD1.5 換成已安裝的
  Realistic Vision。
- **實測**（seed 6001、16 幀）：不高清不精修不放大 69 秒 @512²；預設高清 20 步 693 秒；
  高清 10 步 526 秒、峰值 6.5 GB VRAM（跟 20 步的平均影格差只有 1.41/255，所以 10 步成為
  預設）；LCM 8 步 480 秒（高清、VAE、ESRGAN 的時間不會縮，所以省得有限）；RIFE ×4 @16fps
  得到 61 幀 3.8 秒（RIFE 產出是 15×m+1 幀，不是 16×m）。
- **新增**：`training/talking_head.py` 把 SadTalker 包成 `talk` 指令和 GUI 區塊，用
  subprocess 呼叫 SadTalker 自己的 Python 3.10 venv（不 import，也不需要 ComfyUI 在跑）。
- **文件**：README「AnimateDiff 動態影片」「會講話的嘴型影片（SadTalker）」

## 更早（一行摘要）

| 日期 | commit | 內容 |
|---|---|---|
| 2026-09-09 | `b6fec49` | GUI 需要時自動啟動 ComfyUI，不必事先開好 |
| 2026-09-09 | `2358e16` | 骨架庫在各 checkpoint 的實測，並更正一個寫錯的說法 |
| 2026-09-09 | `db9cea0` | 擋掉負面詞裡重複的主體詞 |
| 2026-09-09 | `d6f3d33` | 會把骨架裁掉的畫布尺寸直接拒絕 |
| 2026-09-09 | `97cc69d` | 重建 6 張在膝蓋被裁掉的骨架的腿部 |
| 2026-09-09 | `853ba36` | 新增 14 個純骨架庫姿勢（跪、蹲、斜躺、騰空） |
| 2026-09-09 | `c8e5274` | ControlNet OpenPose 骨架庫，用來壓住 checkpoint 壓不住的姿勢 |
| 2026-09-09 | `5ccc794` | 姿勢／角度標籤參考包產生器與實測結果 |
| 2026-09-08 | `448151d` | 各 checkpoint 的 prompt 語法實測與 prompt 食譜 |
| 2026-09-08 | `99622cc` | HQ 兩段式流程、RunPod LoRA 訓練、RunPod serverless worker |
| 2026-08-23 | `d4a9e6a` | web：provider-types 與 replicate / runpod 產生器骨架（WIP） |
| 2026-08-23 | `09fcb23` | 把既有的 web/ Amplify Gen 2 骨架納入版控 |
| 2026-08-21 | `c2aa8a5` | 初始 commit：ComfyUI 生成流程、GUI、訓練設定 |
