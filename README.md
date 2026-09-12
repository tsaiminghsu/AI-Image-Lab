# AI Image Lab

用 **ComfyUI + SDXL + IP-Adapter** 生成虛構（非真人）角色的多角度訓練圖片資料集，
供之後用 [kohya_ss](https://github.com/bmaltais/kohya_ss) 訓練角色 LoRA。

每次改動的結論和實測數字整理在 [CHANGELOG.md](CHANGELOG.md)；給 Claude Code 看的
專案速覽在 [CLAUDE.md](CLAUDE.md)。

## 內容政策

- 所有角色皆為**虛構人物**，非真人、非公眾人物、非任何真實個體的肖像
- 年齡下限 **18 歲**，在程式碼層級強制（`generate_character.py` 的 `MINIMUM_AGE`，
  任何角色年齡低於這個值會在 import 時直接 `raise ValueError` 讓程式跑不起來）
- negative prompt 固定包含未成年防護詞（`AGE_SAFETY_NEGATIVE`：child/children/kid/
  minor/teen/teenager/underage/young girl），**不會**因為內容分級（一般/擦邊）而放寬
- 預設生成非露骨內容
- 僅供個人 LoRA 訓練研究用途，非商業使用

## 硬體 / 測試環境

| 項目 | 值 |
|---|---|
| GPU | NVIDIA RTX 2070, 8 GB VRAM（Turing sm_75：沒有 bf16、沒有 fp8 運算） |
| 系統 RAM | 32 GB |
| PCIe | 實測跑在 gen3 x1（不是 x16），模型載入／搬移特別慢，取樣本身不受影響 |
| OS | Windows 10 Pro 10.0.19045 |
| Python | 3.11.15（由 [uv](https://github.com/astral-sh/uv) 管理，隔離 venv） |

## 架構

```
generate_character.py (CLI, 角色定義 + prompt 組裝)
        │  HTTP
        ▼
comfyui_client.py (薄 client：建 workflow JSON、送出、輪詢、下載結果)
        │  POST /prompt, GET /history, GET /view, GET /system_stats
        ▼
ComfyUI server（常駐 process，port 8188）
        │  模型生命週期完全由 ComfyUI 管理：啟動時載入一次，
        │  之後每次 /prompt 呼叫不重新載入
        ▼
SDXL checkpoint + IP-Adapter + 風格 LoRA + KSampler → 輸出圖片
```

Python 端**完全不 import torch / 不碰 CUDA**——所有 VRAM 管理、模型快取都交給
ComfyUI 這個持續運行的 server 處理，避免每次生成重複載入模型或手動 GC 造成的
記憶體問題。診斷用的 `log_gpu_memory(stage)` 也是查詢 ComfyUI 的
`GET /system_stats`，不是本地 CUDA 呼叫。

## 安裝

### 1. Clone ComfyUI + IP-Adapter custom node

已驗證可用的版本（commit 已測試過整條 pipeline，不保證更新版一定相容）：

```powershell
git clone https://github.com/comfyanonymous/ComfyUI D:\AI-Image-Lab\ComfyUI
cd D:\AI-Image-Lab\ComfyUI
git checkout 62b3c94bd45154f6486c7abf1b9efcacee96ea69   # 2026-08-11

git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus custom_nodes\ComfyUI_IPAdapter_plus
cd custom_nodes\ComfyUI_IPAdapter_plus
git checkout a0f451a5113cf9becb0847b92884cb10cbdec0ef   # 2025-04-14
```

### 2. 建立 venv + 安裝套件

```powershell
cd D:\AI-Image-Lab\ComfyUI
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r ..\comfyui-requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

`comfyui-requirements.lock.txt`（repo 根目錄）是本專案實際測試過、跑得動的完整
凍結版本清單（`uv pip freeze` 產出），共 76 個套件。重點版本：

| 套件 | 版本 |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| transformers | 5.15.0 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |
| pillow | 12.3.0 |
| huggingface-hub | 1.27.0 |
| requests | 2.34.2 |
| psutil | 7.2.2 |

> `uv` 在這個環境下對大型 wheel（尤其 torch 系列）偶爾會靜默卡住。如果遇到，
> 改用 `curl` 直接下載 wheel 檔，再 `uv pip install <本機路徑>` 離線安裝，
> 這是本專案實際驗證過穩定的替代方案。

### 2b. 一鍵安裝網頁 GUI + 所有選用功能套件

上面的 `comfyui-requirements.lock.txt` 只涵蓋 ComfyUI 本體。網頁 GUI（`gui.py`）
在畫面上直接暴露了骨架姿勢控制（ControlNet）、臉部/手部精修（ADetailer，含
YOLO 跟 MediaPipe 兩種偵測後端）、依圖片產生 Prompt（BLIP）幾個選用功能，
外加 `image_api.py`（FastAPI 服務）——這些各自需要的 Python 套件現在整理進
一個檔案，一次裝完，不用再照著下面各選用功能章節一個一個手動 `pip install`：

```powershell
cd D:\AI-Image-Lab\ComfyUI
uv pip install --python .venv\Scripts\python.exe -r ..\comfyui-requirements-extra.lock.txt
```

`comfyui-requirements-extra.lock.txt`（repo 根目錄，同樣是 `uv pip freeze` 產出
的凍結版本清單，共 51 個套件）內部依 `uv pip tree` 實際跑出來的相依關係分成
六個區塊，每個區塊在檔案裡都有註解標題：

| 區塊 | 對應功能 | 代表套件 |
|---|---|---|
| 網頁 GUI | `gui.py` 本身 | `gradio`、`gradio-client`、`hf-gradio`、`pandas` |
| GUI + API 共用 | gradio 內建網頁伺服器，`image_api.py` 也直接用 | `fastapi`、`starlette`、`uvicorn` |
| ControlNet 骨架姿勢控制 | `comfyui_controlnet_aux` 前處理器 | `opencv-python`、`scikit-image`、`matplotlib`、`onnxruntime-gpu`、`importlib-metadata` |
| ADetailer 臉部/手部精修 | Impact-Pack / Impact-Subpack | `ultralytics`、`segment-anything`、`piexif`、`dill` |
| ADetailer MediaPipe 後端 | `facedetailer_backend="mediapipe"` | `mediapipe`、`opencv-contrib-python` |
| InsightFace/ONNX 執行期相依 | 見下方說明 | `ml-dtypes` |

`opencv-python`/`matplotlib` 這類被兩個以上功能同時用到的套件只在第一次出現的
區塊裝一次，檔案裡有註解標明是共用的，不會重複安裝。

> `insightface`、`onnx` 本身（FaceID 必要套件）刻意**沒有**放進這個檔案——
> 需要 `--no-deps` 單獨安裝，理由見下面「3b. 安裝 InsightFace」，混進同一個
> requirements 檔案裡沒辦法只對這兩個套件套用這個 flag。`ml-dtypes` 是 `onnx`
> 執行期實際會用到、但 `--no-deps` 不會自動帶進來的相依，所以額外列在這個
> 檔案的最後一個區塊。

### 3. 下載模型（不重新訓練、直接用官方釋出檔案）

| 模型 | 檔名 | 大小 | 來源 | 存放路徑 |
|---|---|---|---|---|
| Juggernaut XL v9（Photo）| `juggernaut_xl_v9_photo.safetensors` | 6.6 GB | [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)（下載 `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` 後改名） | `models/base/` |
| IP-Adapter FaceID Plus V2 | `ip-adapter-faceid-plusv2_sdxl.bin` | 1.49 GB | [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) | `models/ip_adapter/sdxl_models/` |
| IP-Adapter FaceID Plus V2 LoRA | `ip-adapter-faceid-plusv2_sdxl_lora.safetensors` | 372 MB | [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) | `models/ip_adapter/sdxl_models/` |
| CLIP Vision (ViT-H) | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | 2.53 GB | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)，`sdxl_models/` 資料夾 | `models/ip_adapter/sdxl_models/` |
| 寫實攝影風格 LoRA | `sdxl_photorealistic_slider_v1-0.safetensors` | 24 MB | [ostris/photorealistic-slider-sdxl-lora](https://huggingface.co/ostris/photorealistic-slider-sdxl-lora) | `ComfyUI/models/loras/` |

> 目前實際套用的 checkpoint 是 **Juggernaut XL v9（Photo 版本）**，不是原生
> SDXL Base 1.0——Juggernaut 是 SDXL 生態圈裡以寫實攝影見長的熱門 fine-tune，
> `comfyui_client.py` 的 `CHECKPOINT` 常數、`workflow_template.json` 的
> checkpoint node 都指向這個檔名。原生 `sd_xl_base_1.0.safetensors` 只在
> LoRA 訓練階段（kohya_ss）用得到，因為訓練出來的 LoRA 要能套用在乾淨的
> base model 上才有可攜性；生成訓練圖片本身不需要它。

> IP-Adapter 用的是 **FaceID Plus V2**，不是早期版本的 PLUS FACE——PLUS FACE
> 靠 CLIP-vision 影像相似度做臉部條件化，同一個角色不同 seed/pose 生成出來的
> 臉常常對不上（訓練 LoRA 需要的是同一個角色跨圖一致，這樣的資料集會讓模型學到
> 模糊的平均臉）。FaceID Plus V2 改用 InsightFace 的人臉辨識嵌入向量，同一張
> anchor 圖不管生成什麼姿勢/場景，臉孔辨識度都遠比 PLUS FACE 穩定（實測見下方
> 生成參數章節）。CLIP Vision (ViT-H) 仍然需要——FaceID Plus V2 是把 InsightFace
> 的身分嵌入跟 CLIP-vision 特徵一起用，不是完全取代它。

前四個放在 `models/` 底下（本 repo 的統一存放位置），再用 **NTFS hardlink**（不是
symlink——這台機器上 symlink 因權限問題不能用）接進 ComfyUI 自己的資料夾，
避免複製一份浪費空間：

```powershell
New-Item -ItemType HardLink -Path "ComfyUI\models\checkpoints\juggernaut_xl_v9_photo.safetensors" -Target "models\base\juggernaut_xl_v9_photo.safetensors"
New-Item -ItemType HardLink -Path "ComfyUI\models\ipadapter\ip-adapter-faceid-plusv2_sdxl.bin" -Target "models\ip_adapter\sdxl_models\ip-adapter-faceid-plusv2_sdxl.bin"
New-Item -ItemType HardLink -Path "ComfyUI\models\loras\ip-adapter-faceid-plusv2_sdxl_lora.safetensors" -Target "models\ip_adapter\sdxl_models\ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
New-Item -ItemType HardLink -Path "ComfyUI\models\clip_vision\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" -Target "models\ip_adapter\sdxl_models\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
```

風格 LoRA 檔案小，直接下載進 `ComfyUI/models/loras/` 即可，不需要 hardlink。

### 3b. 安裝 InsightFace（FaceID 必要套件）

```powershell
cd D:\AI-Image-Lab\ComfyUI
uv pip install --python .venv\Scripts\python.exe insightface onnx --no-deps
```

用 `--no-deps` 是因為 `insightface` 預設會拉一個獨立的 `onnxruntime`（CPU 版），
跟已經裝好的 `onnxruntime-gpu` 是同一個 Python 模組名稱（`onnxruntime`），兩個
一起裝會互相覆蓋檔案，還可能在 ComfyUI process 正在跑、DLL 檔案被鎖住時直接
安裝失敗（`os error 5`，存取被拒）。`onnxruntime-gpu` 本身就滿足 `insightface`
需要的 import，不需要額外裝 CPU 版。如果要重裝，記得先關掉 ComfyUI process
再跑這個指令。

`buffalo_l` 人臉辨識模型不需要手動下載——`IPAdapterUnifiedLoaderFaceID` 節點
第一次執行時會自動下載（約 280MB）到 `ComfyUI/models/insightface/`，下載完
會快取，之後不用重新下載。

### 3c. 量化版模型（選用，fp8，只限 juggernaut / pony / cyberrealistic_pony / pony_realism）

這四個 SDXL / Pony checkpoint 可以各自選「完整版」或「量化版 fp8」。量化版用
`training/quantize_models.py` 在本機從原始檔轉出，不用另外下載；原始檔（有些是
NTFS hardlink）只會被讀，不會被改。

**為什麼要用 ComfyUI 自己的每層量化格式，不是單純轉成 fp8**：RTX 2070 沒有 fp8 運算，
`CheckpointLoaderSimple` 載入單純 cast 成 fp8 的檔案時，只要模型放得下，就會把權重轉回 fp16
放進 VRAM，等於白做。每層量化格式（每個被量化的 Linear 層帶 `.comfy_quant` 標記和
`.weight_scale`）會被 ComfyUI 認成「mixed precision」，量化層在這張卡上以模擬方式保持 fp8
（存 fp8、每層前向時轉成 fp16 再算）。只有 transformer block 裡的 Linear 層和 proj_in/out
會被量化（約 86% 的 UNet 參數），卷積、norm、文字編碼器、VAE 原樣保留。

```powershell
ComfyUI\.venv\Scripts\python.exe training\quantize_models.py convert --model cyberrealistic_pony
ComfyUI\.venv\Scripts\python.exe training\quantize_models.py convert --all
ComfyUI\.venv\Scripts\python.exe training\quantize_models.py verify --model cyberrealistic_pony
ComfyUI\.venv\Scripts\python.exe training\quantize_models.py status
ComfyUI\.venv\Scripts\python.exe training\quantize_models.py set cyberrealistic_pony=quant juggernaut=full
```

每個量化檔寫在原始檔旁邊，檔名是 `<原檔名>.fp8q.safetensors`（例如
`CyberRealisticPony_V18.0_F16.fp8q.safetensors`），約 4.7-4.9 GB（原檔 6.9-7.1 GB），
四個全轉約多佔 19 GB。`convert` 轉完會自動跑 `verify`：用 ComfyUI 自己的函式重新讀檔，確認
偵測得到量化格式、架構仍是 SDXL，並抽樣比對反量化後的權重（cosine 應 > 0.999）。ComfyUI
更新之後建議重跑一次 `verify`。一個模型在這台機器上轉加驗證約 1.5-2 分鐘。

選哪一版存在 `training/settings/model_variants.json`（已加進 `.gitignore`；GUI 的「模型
版本」區塊或上面的 `set` 指令都會寫它）。優先順序：CLI 的 `--variant` > 環境變數
`MODEL_VARIANT=full|quant` > 設定檔裡該模型的選擇 > 設定檔的 `default` > 完整版。每次
送出都會重讀設定檔，改完下一張就生效，不用重啟 ComfyUI。

- **量化檔還沒轉出來**：自動改用完整版，CLI 印一行 `[variant]` 提示、GUI 跳提示，都附上
  轉換指令。
- **骨架姿勢（ControlNet control-lora）一律用完整版**：control-lora 會直接複製主模型的
  權重，量化層複製到的是沒乘 scale 的 fp8 原始值，姿勢會壞掉；工作流程裡有 ControlNet
  時會自動改用完整版並提示。
- **RunPod worker 一律用完整版**（`worker/Dockerfile` 設了 `MODEL_VARIANT=full`）。
- SD1.5 / AnimateDiff、SVD、Z-Image 不在這個機制裡。

**實測（RTX 2070 8 GB、PCIe x1、32 GB RAM；cyberrealistic_pony，同 seed 完整版 vs 量化版）**：

| 項目 | 完整版 | 量化版 fp8 |
|---|---|---|
| 主模型搬上 GPU 的大小（log 的 Staged） | 4896 MB | 2791 MB |
| 純文字生圖 1024² 每張（3 個 seed） | 36-40 秒 | 40-42 秒 |
| GUI 預設高清（FaceID + hires + 臉/手精修）每張 | 167 / 272 / 530 秒 | 145 / 127 / 125 秒 |
| 高清時 ComfyUI 記憶體峰值 | 15.5-15.8 GB | 12.3-13.2 GB |
| 身分相似度（InsightFace，對參考臉） | 0.498 / 0.560 / 0.553 | 0.487 / 0.557 / 0.560 |

- **速度**：純文字生圖單看取樣，量化版每步慢一點（每層多一次 fp8→fp16 轉換；這張卡沒有
  fp8 運算），而且量化版那幾張跑的時候 GPU 比較熱（74-78°C vs 53-70°C），數字不能直接比。
  高清流程完整版那幾張變慢，是因為記憶體吃緊（系統只剩約 4 GB 可用、開始用分頁檔），
  模型要反覆經過 x1 通道搬移；量化版少搬約 2 GB，所以反而快很多。
- **畫質**：同一個 seed 構圖、臉、場景一樣，放大看沒有雜點或色帶；但細節會小幅漂移
  （純文字生圖 SSIM 0.76-0.81，沒達到原本預設的 0.85），高清流程經過 hires 和精修會把差異
  放大，3 個 seed 裡有 2 個構圖和服裝不一樣（其中一張量化版的外套敞開、露出比較多皮膚）。
  身分相似度沒有變差。**同一個 seed 在兩個版本出來的圖不會一樣**，要重現舊圖請用原本的版本。
- **骨架姿勢**：強制讓量化版跑 control-lora，出來是全黑的圖，所以才一律改用完整版。

其他三個模型各跑一張純文字生圖（1024²，seed 7101，同 prompt）：

| 模型 | 完整版 | 量化版 | 取樣每步 完整 / 量化 | ComfyUI 記憶體 完整 / 量化 | SSIM |
|---|---|---|---|---|---|
| juggernaut | 36.4 秒 | 36.4 秒 | 0.60 / 0.69 秒 | 9.2 / 7.0 GB | 0.949 |
| pony | 40.3 秒 | 42.5 秒 | 0.7-0.8 / 1.05 秒 | 9.6 / 6.8 GB | 0.844 |
| pony_realism | 42.6 秒 | 46.5 秒 | 0.9 / 1.2 秒 | 9.2 / 7.0 GB | 0.827 |

三個都正常、肉眼看不出瑕疵。juggernaut 那組兩張跑的時候溫度差不多（57-67°C），最能看出
量化本身的代價：每步慢約 15%，整張時間一樣。

### 4. 啟動 ComfyUI server

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
```

`training/` 底下的 CLI 腳本（`generate_character.py` 等）都是透過 HTTP 呼叫
ComfyUI，不會自己啟動或關閉它，所以用 CLI 之前要先手動啟動、全程保持常駐。

> 網頁 GUI（`gui.py`）不在此限——它會在你按下「生成」時自動偵測 ComfyUI
> 有沒有在跑，沒有的話自動啟動，不用先手動執行這一步，見下面「網頁 GUI」章節。

### 快速測試安裝是否成功

```powershell
curl http://127.0.0.1:8188/system_stats
```

回傳 JSON（含 `devices` 底下的 GPU 資訊）代表 ComfyUI 正常運行。接著跑一次最小
的生圖測試，確認 checkpoint/IP-Adapter 都載得動、整條 pipeline 沒問題：

```powershell
cd D:\AI-Image-Lab\training
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe generate_character.py anchor --character mei --seeds 3001
```

跑完會在 `training/reference_candidates/mei/anchor_seed3001.png` 產生一張圖，
用檔案總管打開確認畫面正常，就代表整個環境裝好了。之後可以直接跳到下面的
「網頁 GUI」用瀏覽器操作，或繼續看「怎麼生成角色」用 CLI 批次生成 dataset。

## ControlNet 骨架姿勢控制（選用安裝）

基礎安裝完成、確認可以正常生圖之後才需要裝這個——用 OpenPose 骨架精確控制生成
姿勢，解決純文字描述姿勢常常不準的問題。

> **效能取捨**:8GB VRAM 卡上，checkpoint(6.6GB)+ FaceID(1.49GB+ 372MB LoRA)+
> CLIP vision(2.5GB)+ ControlNet(774MB)+ OpenPose 偵測模型全部疊在一起會超過顯存，
> ComfyUI 得把模型搬進搬出。**那個 440 秒/張是舊的 legacy 單段式路徑（`submit_generation_with_pose`）
> 且掛 FaceID 的手估值**——瓶頸是 FaceID 的 VRAM 帳外佔用，不是 ControlNet。實務上
> 不掛 FaceID 走骨架庫（見下方「ControlNet 姿勢骨架庫」）單張約 60-90 秒。裝這個
> custom node 是為了 `--pose-reference`（上傳照片抽骨架）；預先畫好的骨架庫
> （`--pose`）連 preprocessor 都不用跑。

### 1. Clone 前處理 custom node

```powershell
cd D:\AI-Image-Lab\ComfyUI\custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux
```

這個 repo 支援十幾種 ControlNet 前處理器（深度圖/線稿/分割等），但只用得到
OpenPose 相關的部分，所以**不要**照著它自己的 `requirements.txt` 整包裝
（裡面的 `mediapipe`/`trimesh`/`albumentations` 這些是給其他前處理器用的，
`albumentations` 依賴的 `albucore`/`stringzilla` 在 Windows 上還會因為缺
Visual C++ Build Tools 編譯失敗）。只裝 OpenPose 骨架偵測實際用到的套件：

```powershell
cd D:\AI-Image-Lab\ComfyUI
uv pip install --python .venv\Scripts\python.exe importlib_metadata huggingface_hub scipy opencv-python filelock pyyaml scikit-image python-dateutil matplotlib onnxruntime-gpu
```

> 如果已經在上面「2b. 一鍵安裝網頁 GUI + 所有選用功能套件」裝過
> `comfyui-requirements-extra.lock.txt`，這幾個套件已經包含在裡面，這一步可以跳過。

### 2. 下載 ControlNet 模型

```powershell
New-Item -ItemType Directory -Force -Path "D:\AI-Image-Lab\models\controlnet"
curl.exe -L -o "D:\AI-Image-Lab\models\controlnet\control-lora-openposeXL2-rank256.safetensors" https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/control-lora-openposeXL2-rank256.safetensors
New-Item -ItemType HardLink -Path "D:\AI-Image-Lab\ComfyUI\models\controlnet\control-lora-openposeXL2-rank256.safetensors" -Target "D:\AI-Image-Lab\models\controlnet\control-lora-openposeXL2-rank256.safetensors"
```

> 用的是 **Control-LoRA**（774MB，rank-decomposed 版本），不是同一個 repo 裡
> `OpenPoseXL2.safetensors` 那個完整版 ControlNet（5GB）——8GB VRAM 卡同時載
> Juggernaut(6.6GB) + IP-Adapter(848MB) + CLIP vision(2.5GB) 已經很緊，換成
> 完整版會直接爆顯存。ComfyUI 的 `ControlNetLoader`/`ControlNetApplyAdvanced`
> 這兩個核心節點原生支援 control-lora 格式，不需要額外的載入節點。

### 3. 重啟 ComfyUI

custom node 跟新模型都要重啟才會被 ComfyUI 掃描到：

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
```

啟動後可以確認節點有註冊成功：

```powershell
curl http://127.0.0.1:8188/object_info/ControlNetLoader
curl http://127.0.0.1:8188/object_info/OpenposePreprocessor
```

兩個都回傳 JSON（不是 404）就代表裝好了。

### 4. 第一次使用會下載額外模型（正常現象）

`OpenposePreprocessor` 節點第一次執行時，會自動從 `lllyasviel/Annotators`
下載身體/手部/臉部三個姿勢偵測模型（約 200MB 總計），存到
`ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/` 底下。這是在 ComfyUI
process 裡面下載，不是額外的手動步驟，但視網路速度可能要花好幾分鐘，第一次
生成請求可能因此超過平常的 timeout（`comfyui_client.py` 的
`POLL_TIMEOUT_SECONDS` 已經統一調高到 900 秒，涵蓋這種第一次執行會額外下載
模型的生成模式——FaceID 的 `buffalo_l` 模型也是同樣的情況）。下載一次之後
就會快取在本機，之後的生成速度就跟平常一樣。

用法見下面「自由輸入 Prompt」章節的 `--pose-reference` 參數，或網頁 GUI 的
「骨架姿勢控制」摺疊區塊。

## ADetailer 臉部/手部精修（選用安裝）

基礎安裝完成、確認可以正常生圖之後才需要裝這個——生成後自動偵測臉部/手部，
各自裁切放大重繪一次，修正 SDXL 常見的臉部細節模糊、手指變形等小瑕疵（原理
跟 A1111 的 ADetailer 擴充功能一樣，只是換成 ComfyUI 的節點實作）。

### 1. Clone Impact-Pack + Impact-Subpack

```powershell
cd D:\AI-Image-Lab\ComfyUI\custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack
```

`FaceDetailer` 節點（實際做裁切/重繪/貼回的邏輯）在 Impact-Pack，
`UltralyticsDetectorProvider`（載入 YOLO 偵測模型）在 Impact-Subpack，兩個都
要裝才會動。

```powershell
cd D:\AI-Image-Lab\ComfyUI
uv pip install --python .venv\Scripts\python.exe segment-anything piexif dill "ultralytics>=8.3.162"
```

> 如果已經裝過 `comfyui-requirements-extra.lock.txt`（見上面 2b），這幾個套件
> 已經包含在裡面，這一步可以跳過。

Impact-Pack 自己的 `requirements.txt` 還列了 `git+https://github.com/facebookresearch/sam2`，
但那個套件在這個 repo 裡沒有被直接 import 到（`segment_anything` 才是真的
會用到的），為了避免 git-based 套件安裝拖慢/搞複雜整個安裝流程，這裡跳過了；
如果之後真的遇到 `sam2` 相關的 ImportError 再另外裝。

### 2. 下載 YOLO 模型（自動檢查 + 一次性下載腳本）

```powershell
cd D:\AI-Image-Lab\training
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe check_adetailer_models.py
```

這個腳本是 idempotent 的——已經存在的模型不會重複下載，只補下載缺的，可以
放心重複執行來確認安裝完整。會下載三個模型到 `ComfyUI/models/ultralytics/`：

| 模型 | 用途 | 大小 | 來源 | 存放路徑 |
|---|---|---|---|---|
| `face_yolov8n.pt` | 臉部偵測 | ~6 MB | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) | `ultralytics/bbox/` |
| `hand_yolov8n.pt` | 手部偵測 | ~6 MB | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) | `ultralytics/bbox/` |
| `yolov8n-seg.pt` | 通用分割（COCO 80 類，非人物專用）| ~7 MB | [ultralytics/assets](https://github.com/ultralytics/assets/releases) | `ultralytics/segm/` |

> 都是 YOLOv8 的 "nano" 版本，檔案小、VRAM/運算負擔比 checkpoint/FaceID/
> ControlNet 這些主要模型輕很多，不是這個功能的效能瓶頸——真正花時間的是
> FaceDetailer 對偵測到的每個區域多跑一次採樣（見下方生成參數章節）。
> `yolov8n-seg.pt` 目前下載了但沒有接進預設的 workflow（現有的 FaceDetailer
> 用 bbox 偵測就夠用），留著是為了之後如果要做分割式遮罩會用到。

### 3. 重啟 ComfyUI

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
```

確認節點註冊成功：

```powershell
curl http://127.0.0.1:8188/object_info/FaceDetailer
curl http://127.0.0.1:8188/object_info/UltralyticsDetectorProvider
```

用法見下面「自由輸入 Prompt」章節的 `--use-facedetailer` 參數，或網頁 GUI 的
「臉部/手部精修」摺疊區塊。**目前跟 `--pose-reference`（ControlNet 骨架）不能
同時使用**——兩者是各自獨立的 workflow JSON 檔，還沒做合併版本。

### 4.（選用）MediaPipe 臉部網格 — 另一種臉部偵測方式

上面裝好的是 YOLO 矩形框偵測（`face_yolov8n.pt`），臉部偵測還有第二種選擇：
MediaPipe 的臉部網格（face mesh），遮罩會貼合實際臉型輪廓而不是矩形框，
下巴/髮際線這類非矩形邊界理論上比較不會把背景/頭髮一起裁進去重繪。手部
精修沒有對應的 MediaPipe 版本，一律用 YOLO。

```powershell
cd D:\AI-Image-Lab\ComfyUI
uv pip install --python .venv\Scripts\python.exe mediapipe
```

（跟其他套件一樣，如果 ComfyUI 正在跑，`cv2`/`opencv` 相關檔案會被鎖住導致
安裝失敗 `os error 5`——先關掉 ComfyUI process 再裝，裝完再重啟。）

> 如果已經裝過 `comfyui-requirements-extra.lock.txt`（見上面 2b），`mediapipe`
> 已經包含在裡面，這一步可以跳過。

用法：CLI 的 `custom` 指令加 `--facedetailer-backend mediapipe`（預設是
`yolo`），或網頁 GUI「臉部/手部精修」區塊裡的「臉部偵測方式」選單。

> 實測（同一張人物、同一個 prompt）：YOLO 矩形框版本跟 MediaPipe 版本兩次
> 生成結果都乾淨，沒有觀察到臉部失真、拼接痕跡——測試用的是正面、五官清楚
> 的構圖，還沒測過頭髮遮臉、側臉等比較容易讓矩形框誤裁的情境，MediaPipe
> 版本的優勢在那類情境下才比較看得出來，這點還沒驗證。

## AnimateDiff 動態影片（選用安裝）— 修正臉部變形

`video` 指令（SVD img2vid，見下面「幫已有的圖片配上動作」）沒辦法接
IP-Adapter：SVD 是完全不同的時序 U-Net 架構，ComfyUI 的 IPAdapter 節點沒辦法
patch 進去，所以生成過程中沒有任何機制鎖住臉部身分，動態幅度一大就容易變形/
融化（`comfyui_client.py` 的 `MOTION_BUCKET_ID` 註解本來就寫了這個限制）。

AnimateDiff 是把 motion module 插進一個**正常的 SD1.5 UNet**裡，而不是換一個
架構，所以現有的 `IPAdapterUnifiedLoaderFaceID`/`IPAdapterFaceID` 節點可以
原封不動用在它上面——同一組節點會自動偵測底模是 SD1.5（不是 SDXL），换用
對應的 `*_sd15.bin`/`*_sd15_lora.safetensors` 檔案，`CLIP-ViT-H` 跟
InsightFace `buffalo_l` 都直接沿用現有的，不用重下載。生成完再對每一幀跑一次
FaceDetailer（用同一個 AnimateDiff-patched 模型），臉部裁切區塊會被當成一小段
連續影格一起重新採樣，而不是逐幀獨立處理，這樣修出來的臉才不會一幀一個樣。

### 1. Clone AnimateDiff-Evolved custom node

```powershell
cd D:\AI-Image-Lab\ComfyUI
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git custom_nodes/ComfyUI-AnimateDiff-Evolved
```

沒有額外的 pip 套件需要裝。

### 2. 下載模型

AnimateDiff 成熟、社群支援完整的 motion module 是 SD1.5 系列（SDXL 版本還在
beta、VRAM 需求也重很多），所以這條路線用 SD1.5 checkpoint，不是目前生圖用的
Juggernaut SDXL。**預設 checkpoint 是 Realistic Vision V6**（SD1.5 靜態圖路線
本來就裝好的寫實 fine-tune，UNet 形狀跟官方 SD1.5 一模一樣，motion module 直接
套得上）——之前預設用官方 `v1-5-pruned-emaonly`、想靠 prompt 補寫實感，實際上
皮膚/臉部質感是整條影片 pipeline 最弱的一環，換 checkpoint 是零成本的畫質提升。

| 模型 | 用途 | 大小 | 來源 | 存放位置 |
|---|---|---|---|---|
| `Realistic_Vision_V6.0_NV_B1_fp16.safetensors` | SD1.5 底模（**預設**） | ~2.1 GB | 已跟 SD1.5 靜態圖路線共用 | `ComfyUI/models/checkpoints/` |
| `v1-5-pruned-emaonly.safetensors` | 官方 SD1.5 底模（選用，`--checkpoint sd15_base`） | ~4.3 GB | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `ComfyUI/models/checkpoints/` |
| `mm_sd_v15_v2.ckpt` | AnimateDiff motion module | ~1.8 GB | [guoyww/animatediff](https://huggingface.co/guoyww/animatediff) | `ComfyUI/models/animatediff_models/` |
| `ip-adapter-faceid-plusv2_sd15.bin` | FaceID（SD1.5 版） | ~150 MB | [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) | `ComfyUI/models/ipadapter/` |
| `ip-adapter-faceid-plusv2_sd15_lora.safetensors` | FaceID 隨附 LoRA | ~40 MB | 同上 | `ComfyUI/models/loras/` |
| `4x-UltraSharp.pth` | 二段式高清 + 最終放大 | ~67 MB | 跟「HQ 兩段式生成」共用，見該章節 | `ComfyUI/models/upscale_models/` |

補幀（RIFE）跟 LCM 快速模式需要的檔案見下面 2c、2d，都是選用。

### 2b. AnimateDiff Motion LoRA（選用，鏡頭運動控制）

> **先弄清楚這個功能實際能做什麼**：官方 Motion LoRA 控制的是**整個畫面的
> 鏡頭運動**（縮放、平移、傾斜、旋轉），**不是特定身體部位的物理晃動效果**
> ——motion module 本身沒有「彈跳」「晃動」這類局部物理控制的機制，這不是
> 這個專案接線沒接好，是 AnimateDiff 這個技術目前的能力邊界。想要局部動作
> 效果，目前唯一能試的路是直接在 prompt 裡描述動作（例如 `walking, hips
> swaying`），讓 motion module 憑通用的動作理解去逼近，沒有專門的 LoRA 能
> 精確控制，效果不保證。

官方 [guoyww/animatediff](https://huggingface.co/guoyww/animatediff) 釋出了
8 個鏡頭運動 LoRA，每個 ~77.5MB，**只相容 v2 motion module**（也就是這個
專案已經在用的 `mm_sd_v15_v2.ckpt`）：

| 檔名 | 效果 |
|---|---|
| `v2_lora_ZoomIn.ckpt` | 鏡頭放大 |
| `v2_lora_ZoomOut.ckpt` | 鏡頭縮小 |
| `v2_lora_PanLeft.ckpt` | 鏡頭向左平移 |
| `v2_lora_PanRight.ckpt` | 鏡頭向右平移 |
| `v2_lora_TiltUp.ckpt` | 鏡頭向上傾斜 |
| `v2_lora_TiltDown.ckpt` | 鏡頭向下傾斜 |
| `v2_lora_RollingClockwise.ckpt` | 鏡頭順時針旋轉 |
| `v2_lora_RollingAnticlockwise.ckpt` | 鏡頭逆時針旋轉 |

下載需要的幾個（不用全部下載，GUI/CLI 只有選到對應的才會用到）：

```powershell
New-Item -ItemType Directory -Force -Path "D:\AI-Image-Lab\ComfyUI\models\animatediff_motion_lora"
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\models\animatediff_motion_lora\v2_lora_ZoomIn.ckpt" https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt
```

其他 7 個把檔名換掉照樣下載即可。用法見下面「AnimateDiff 動態影片」GUI
說明的「Motion LoRA」摺疊區塊，或 CLI 的 `--motion-lora`/`--motion-lora-strength`
參數（`generate_character.py video-animatediff --help`）。沒下載就選的話，
ComfyUI 會在 `ADE_AnimateDiffLoRALoader` 節點報找不到檔案，生成直接失敗。

### 2c. RIFE 補幀節點（選用，讓影片更順/更長）

motion module 只能穩定生成 16 張影格（超過就要接 sliding context window，實測會在
第 16/17 幀出現服裝跳變、或整段構圖漂移，已放棄）。要更順或更長，改用
[ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
的 RIFE 在影格之間**插補**出中間幀：16 幀 ×2 = 31 幀、×4 = 61 幀（插在每兩張之間，所以是 15 × 倍數 + 1），不用重新採樣。
輸出 FPS 預設是 `8 × 倍數`（片長一樣約 2 秒、只是變順）；想拉長就把 FPS 調低，
例如 ×4 配 16fps = 約 3.8 秒的慢動作。這取代了以前手動用 ffmpeg `setpts` 放慢的做法
（那只會變長、不會變順）。

```powershell
cd D:\AI-Image-Lab\ComfyUI
git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git custom_nodes\ComfyUI-Frame-Interpolation
```

> 用 `--depth 1`（只抓最新版本）：這個 repo 的完整歷史很大，實測完整 clone 在這台
> 機器上十分鐘還抓不完；ComfyUI 只需要目前的檔案，不需要歷史。

> **只 clone，不要跑它的 `install.py`**。`install.py` 會嘗試安裝 `cupy`，那只有
> GMFSS/STMFNet 這幾個模型用得到（而且是用到時才 import），RIFE 是純 PyTorch；
> 它 `requirements-no-cupy.txt` 列的套件（kornia、einops、opencv-contrib、scipy
> 等）在 ComfyUI 的 `.venv` 裡已經全部有了。cupy 要系統裝 CUDA toolkit 才載得起來，
> 硬裝還可能重新解析 torch 版本，弄壞整個環境。

`rife47.pth`（~20 MB）第一次使用時會自動下載到
`custom_nodes\ComfyUI-Frame-Interpolation\ckpts\rife\`。如果自動下載失敗，手動下載：

```powershell
New-Item -ItemType Directory -Force -Path "D:\AI-Image-Lab\ComfyUI\custom_nodes\ComfyUI-Frame-Interpolation\ckpts\rife"
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\custom_nodes\ComfyUI-Frame-Interpolation\ckpts\rife\rife47.pth" https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife47.pth
```

沒裝這個節點就選補幀 ×2/×4 的話，CLI/GUI 會在送出前直接擋下並提示回來看這一節。

### 2d. AnimateLCM 快速模式（選用，8 步取代 20 步）

LCM（Latent Consistency Model）是蒸餾過的取樣方式，8 步就能出圖。預設 preset
`animatelcm` 用 [AnimateLCM](https://huggingface.co/wangfuyun/AnimateLCM) 自己的
motion module + UNet LoRA（兩者是一起蒸餾的，比拿通用 LCM LoRA 硬套 v2 motion module
閃爍少）；備援 preset `lcm_lora` 維持原本的 `mm_sd_v15_v2` + 通用 `lcm-lora-sdv1-5`。

```powershell
# animatelcm（預設 preset，1.81 GB + 135 MB）
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\models\animatediff_models\AnimateLCM_sd15_t2v.ckpt" https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\models\loras\AnimateLCM_sd15_t2v_lora.safetensors" https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors

# lcm_lora（備援 preset，135 MB，下載後改名）
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\models\loras\lcm-lora-sdv1-5.safetensors" https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
```

| 設定 | 一般模式 | LCM 模式 |
|---|---|---|
| 步數（基礎 / 高清第二段 / 臉部精修） | 20 / 10 / 20 | 8 / 10 / 8 |
| CFG | 7.5 | 2.0 |
| sampler / scheduler | `dpmpp_2m` / `karras` | `lcm` / `sgm_uniform` |
| motion module beta_schedule | `autoselect` | `lcm` |

- 整張圖裡**三個 KSampler 會一起切換**——臉部精修節點也是透過同一個 LCM 化的模型
  重新採樣，留在 karras/20 步會把臉修成一團糊。
- **CFG 最低鎖在 1.5**（`LCM_MIN_CFG`）：CFG = 1.0 時 ComfyUI 會直接跳過 negative
  prompt，年齡保護/露骨內容封鎖這些強制負面詞就會失效。這條不能調低。
- 鏡頭 Motion LoRA 是對 `mm_sd_v15_v2` 訓練的，套在 AnimateLCM 的 motion module
  上不會報錯，但效果**還沒實測**，CLI 會印警告；要確保鏡頭運動有效，改用
  `--lcm-preset lcm_lora`。
- LCM 在 CFG 2 下 FaceID 的身分鎖可能變弱一點，覺得不像可以把 IP-Adapter 權重
  從 1.0 調到 1.2 左右。
- **實測**（預設流程全開，RTX 2070）：LCM 只從 526 秒降到 480 秒。省下的是步數：
  基礎採樣 8 步只要 22 秒，臉部精修從 20 步約 3.8 分鐘降到 8 步約 1.6 分鐘；但高清
  第二段本來就是 10 步（每步約 11 秒），VAE 編解碼跟 ESRGAN 放大的時間也不受 LCM
  影響，所以整體省得不多。同一個 seed 在 LCM 模式下構圖會跟一般模式不一樣（換了
  motion module），畫質跟臉部一致性肉眼看起來沒有變差。
- ComfyUI 載入時把 AnimateLCM 標成 v2 motion module，所以鏡頭 Motion LoRA 有機會
  照樣生效，但還沒實測。
- **同一個 ComfyUI session 裡混用 LCM 跟一般模式會產生雜訊（已自動處理）**：ComfyUI
  開機（或重置）後先跑的那個模式正常，之後另一個模式的輸出會變成純色雜訊，連 SD1.5
  靜態圖也會受影響。狀態殘留在 ComfyUI 快取住的節點輸出裡（兩種模式共用同一個
  checkpoint 模型物件），`POST /free` 清空快取就會恢復。`comfyui_client` 現在每次送出
  前都會比對 ComfyUI 上一個執行的工作，LCM 與否不同就先清快取再送，log 會印出
  `[comfyui] switching to ... sampling`；代價是切換那一次要多花幾秒重新載入模型，同
  模式連續生成不受影響。實測 LCM → 一般 → LCM → SD1.5 靜態圖 → 一般，全部正常。

### 3. 重啟 ComfyUI

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
```

用法見下面「動態影片、臉部不變形」CLI 指令，或網頁 GUI 的「AnimateDiff 動態影片」
區塊。**跟靜態圖那些選項是分開的獨立 workflow**（不同 checkpoint、不同 IP-Adapter
模型檔），第一次執行會比較慢（新的模型組合，ComfyUI 還沒快取過）。

### 影片 pipeline 節點鏈（`workflow_template_animatediff_facedetailer.json`）

```
1 checkpoint ─(61 LCM LoRA)─ 2 AnimateDiff motion module ─(60 鏡頭 Motion LoRA)
  └─ 11/12 FaceID 鎖臉 ─ 3 KSampler（16 張影格一起採樣，512²）─ 8 VAEDecode
  ─ [30-35 二段式高清：ESRGAN 4x → 縮到 1.5x → VAEEncode → 34 KSampler denoise 0.4 → VAEDecodeTiled]
  ─ [40/50/51/52 影片臉部精修]
  ─ [36/37 逐幀 ESRGAN 放大到長邊 1024]
  ─ [70 RIFE 補幀 ×2/×4]
  ─ 90 CreateVideo(fps) ─ 9 SaveVideo（mp4 / h264, crf 20）
```

中括號裡的都是可以關掉的階段，關掉時程式會把節點從 graph 拿掉、把下游接回上游
（跟 HQ 靜態圖路線同一套 `_rewire`/`_drop_nodes` 做法）；`60`/`61`/`70` 三個節點
依賴選用下載/安裝，所以不寫在模板裡，只在用到時由程式注入，沒裝的環境模板照樣能跑。

- **二段式高清**：第二段 KSampler 仍然經過 motion module（節點 12），所以重新採樣的
  影格之間還是有時序關聯，不會變成逐張獨立重畫。16 張影格是一次一起採樣的，面積
  上限鎖在 768×768（`ANIMATEDIFF_HIRES_MAX_PIXELS`）——直式 512×768 的底圖會自動
  縮成約 1.22x。ComfyUI 回報 VRAM 不足（`out of memory` / `Allocation on device`）
  時，會先釋放模型快取、**自動關掉高清再重跑一次**，最終 ESRGAN 放大照樣套用。
- **最終放大**：逐幀 ESRGAN 是確定性的放大，不會引入新的閃爍；
  `ImageUpscaleWithModel` 遇到 VRAM 不足會自己切小塊處理。放在補幀之前，所以
  ESRGAN 只處理 16 張，不是 31/61 張。
- **臉部精修放在高清之後、放大之前**：精修是生成式的，要在 motion module 能處理的
  尺寸跑（臉部裁切 1.5 倍、guide 512 / max 768，臉以約 512px 重繪）；放大後再精修只是多花時間。
- **輸出 mp4/h264**：以前是 vp9 `.webm`（crf 32），改用 ComfyUI 核心的
  `CreateVideo` + `SaveVideo`，任何瀏覽器/播放器/手機都能直接播，編碼也比 vp9 快。

**實測時間（RTX 2070 8GB，16 幀，同一個 seed/prompt）**：

| 設定 | 輸出 | 耗時 | VRAM 峰值 |
|---|---|---|---|
| 不高清、不精修、不放大 | 512²，16 幀 @8fps | 69 s | — |
| 不高清、有精修、不放大 | 512² | 224 s | — |
| 預設（高清第二段 20 步 + 精修 + 放大 1024） | 1024² | 693 s | — |
| **預設（高清第二段 10 步 + 精修 + 放大 1024）** | 1024² | **526 s** | 6.5 GB |
| 預設 + LCM（AnimateLCM；含第一次載入 1.8 GB motion module） | 1024² | 480 s | 6.7 GB |
| 預設 + RIFE ×4、16fps（含 ComfyUI 重啟後第一次載入模型） | 1024²，61 幀、3.8 秒 | 683 s | 6.8 GB |

高清第二段從 20 步降到 10 步，畫面肉眼看不出差異，所以預設是 10 步。時間大宗是
兩個 768² 的整批採樣：高清第二段（10 步約 1.5 分鐘）跟臉部精修（修正前是 20 步約 4 分鐘，
而且那時的裁切其實涵蓋整張影格，見下面「臉部變形排查」）。只是要
快速看動作對不對的話，用 `--no-hires --no-facedetailer --upscale-to 0`（約 1 分鐘）。

> **臉部精修是怎麼接的**（之前版本的 README 這段寫錯了，描述的是更早已經拿掉的
> 做法）：節點 `50`（`ImpactSimpleDetectorSEGS_for_AD`）一次在**所有影格**上偵測
> 臉部區域，節點 `52`（Impact Pack 的 `DetailerForEachPipeForAnimateDiff`）再把
> 整批臉部裁切**當成一小段連續影片**，透過同一個 AnimateDiff + FaceID 模型（節點
> `12`，經 `51 ToBasicPipe` 傳入）一起重新採樣。更早的版本用一般的 `FaceDetailer`
> 接在另一組沒掛 motion module 的 FaceID 分支上（當時的節點 `41`/`11b`/`12b`），
> 雖然避開了「單張圖餵進 motion module 會產生雜訊碎片」的問題，但每一幀的臉都是
> 各自獨立重畫，影格之間明顯閃爍，所以改成現在這個影片專用的 detailer。
> 只做臉部，沒有另外接手部精修。

### 臉部變形排查

把測試影片逐幀拆開比對後，找到兩個讓臉變形的原因，都已經修正：

1. **臉部精修從來沒有真的放大臉部。** 節點 `50` 的 `crop_factor` 原本是 3.0，裁切
   區塊比整張影格還大，`max_size` 768 又把放大倍率壓回 1.0。ComfyUI log 寫的是
   `crop region (768, 768) x 1.0006 -> (768, 768)`：等於拿整張影格用 denoise 0.5
   再跑 20 步，臉沒有多任何解析度，反而多一次整幅漂移。現在裁切是臉框的 1.5 倍、
   `guide_size` 512，臉以約 512px 重繪；精修改成 12 步、denoise 0.45，並補上原本
   漏傳的 `noise_mask_feather`（沒傳時實際是 0，精修邊緣是硬邊）。
2. **負面詞裡有 `symmetrical face`。** 共用的 `REALISTIC_NEGATIVE` 把「對稱的臉」
   列為不要的東西，等於把模型推向雙眼不對稱、下巴歪斜。靜態圖有真正放大重繪的
   ADetailer 可以蓋掉，影片裡臉只有約 180px 寬，就直接露出來。影片路線改用
   `VIDEO_REALISTIC_NEGATIVE`（同一串，拿掉這個詞），靜態圖不變。

**實測**（RTX 2070，seed 6001，512 基底 + 影片臉部精修，同一個 prompt。相似度是影片裡的
臉跟參考臉的 InsightFace cosine，動作量是相鄰影格的平均像素差）：

| 設定 | 平均相似度 | 最低影格 | 動作量 |
|---|---|---|---|
| 修正前（整幅精修） | 0.685 | 0.623 | 4.67 |
| 精修裁切修正 | 0.626 | 0.597 | 4.60 |
| **＋拿掉 `symmetrical face`（目前預設）** | **0.644** | **0.621** | **5.08** |
| ＋FaceID LoRA 0.75 | 0.644 | 0.628 | 2.89 |
| ＋FaceID LoRA 0.75、臉部結構權重 1.5 | 0.654 | 0.635 | 2.22 |
| ＋FaceID LoRA 0.75、臉部結構權重 2.0 | 0.638 | 0.627 | 1.87 |
| ＋上一列再加 motion scale 0.85 | 0.675 | 0.660 | 0.73 |
| 預設再加 motion scale 0.85 | 0.591 | 0.579 | 0.86 |

修正前的相似度反而最高，但逐幀拼圖看得出臉頰浮腫、油光很重、嘴唇被擠成嘟嘴。
InsightFace 量的是「是不是同一個人」，對這種形變不敏感，所以修好了沒要看拼圖，
分數只用來確認沒有換臉。兩項修正之後嘴型跟臉頰都正常，動作量也完全保留，所以這就
是預設。

完整流程（高清第二段 + 精修 + 放大到 1024）用預設跑一次：16 幀 1024²，相似度平均 0.639、
最低 0.611，動作量 5.90（完整保留），逐幀拼圖沒有浮腫或嘟嘴。這次耗時 700 秒，但同樣是在
過熱降頻下量的，不能跟上面「實測時間」表修正前的 526 秒直接比。

加強 FaceID 或調低 motion scale 都會讓臉更「定住」，但主要是因為畫面不動了：LoRA
0.75 就讓動作少了約 40%，相似度卻沒變；motion scale 0.85 單獨用就讓畫面幾乎靜止，
相似度還下降。這幾個參數保留成選項（`--faceid-v2-weight`、`--faceid-lora-strength`、
`--motion-scale`，GUI 也有滑桿），預設維持原值 1.0 / 0.6 / 1.0。motion scale 不是 1.0
時才會注入節點 `62`（`ADE_MultivalDynamic`，接到節點 `2` 的 `scale_multival`）。

> 實測時這張 2070 到了 84°C、觸發硬體過熱降頻，第二輪之後每一輪都比正常慢約一倍，
> 所以上表不列時間。第一輪還沒過熱：512 基底 + 精修 181 秒，修正前是 224 秒；精修從
> 20 步重畫整張 768² 改成 12 步只重畫臉部，是變快的主因。

#### 男性角色被畫成女生（SD1.5）

SD1.5 寫實模型（Realistic Vision）在角色外貌描述比較柔和時（例如 minjun 的「soft face」、
taeoh 的「soft permed hair, refined face」），常常把男性角色畫成女生——prompt 裡只有一個
「man」，壓不過這些在訓練資料裡多半跟女性一起出現的詞。實測（SD1.5 靜態圖、不接 FaceID、
5 個男性角色 × 4 個 seed，性別用 InsightFace 判定，這個判定器對全部 23 張參考照都判對）：

| 寫法 | 男性角色畫成男生 | 女性角色畫成女生 |
|---|---|---|
| 原本的 prompt | 10 / 20 | 20 / 20 |
| 負面詞加「woman, female」 | 10 / 20 | — |
| 正向加「male, man, masculine」 | 14 / 20 | 20 / 20 |
| **性別字加權重 `(man:1.3)`** | **19 / 20** | 20 / 20 |

所以 SD1.5 的路線（AnimateDiff 影片、`custom` 選 SD1.5 checkpoint）現在會把角色 prompt 裡
的性別寫成 `(man:1.3)` / `(woman:1.3)`（`SD15_GENDER_WEIGHT`）。負面詞那條路沒用：負面詞
本來就有 98 個 token，加上去的詞落在第二段 CLIP 的最後面，幾乎沒有作用。SDXL 的 anchor /
variations / 資料集 prompt 刻意不動，既有的 seed 產出不會改變。

影片也一起驗證過（AnimateDiff 一般模式、seed 6001，加權重前 → 後）：jungi 男性影格 7/16 →
16/16、跟參考臉的相似度 0.58 → 0.69；taeoh 16/16 → 16/16、0.62 → 0.63；女性的 yuqing 維持
女生、0.68 → 0.70。`custom` 選 SD1.5 產 minjun、taeoh 的靜態圖也都是男生。

> **AnimateLCM 救不回來**：LCM 的 CFG 只有 2，文字條件本來就很弱，jungi 的 LCM 影片不管
> 有沒有加權重都還是女生（16 格裡最多 2 格判成男生）。男性角色建議不要用 `--lcm`；jungi
> 的參考照頭轉了約 47 度，FaceID 也抓不太到，換一張正面的參考照會更穩。

**客觀檢查臉有沒有跑掉**：`training/face_similarity.py` 用跟 FaceID 同一個
InsightFace 模型（`buffalo_l`），逐幀算影片裡的臉跟參考臉的相似度（cosine，同一
個人通常 0.5 以上），並輸出一張標了分數的逐幀臉部拼圖。生成時加 `--face-report`
會自動跑，拼圖存成 `<檔名>_faces.png`；也可以對任何影片單獨跑（要用 ComfyUI 的
venv，那裡才有 insightface）：

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe training\face_similarity.py --anchor <參考臉.png> --video <影片.mp4> --sheet faces.png
```

## 會講話的嘴型影片（SadTalker，選用安裝）

> **內容規則**：來源人像**僅限虛構/AI 生成的臉**（例如這個專案生成的 anchor），
> **禁止使用真人照片**——跟 FaceID 臉部參考圖的規則完全一樣。

上傳一張人像 + 一段語音，用 [SadTalker](https://github.com/OpenTalker/SadTalker)
產生嘴型跟著語音動的說話影片（mp4，含聲音）。這條路線**完全不經過 ComfyUI**：
SadTalker 有自己的 Python 3.10 / torch 2.5.1 環境，跟 ComfyUI 的 venv 版本衝突，
所以 `training/talking_head.py` 只是用 subprocess 呼叫 SadTalker 自己的
`inference.py`，不會 import 它。GUI 如果發現 ComfyUI 正開著，會先請它釋放顯存，
避免兩個 process 搶同一張 8GB 卡。

### 安裝

```powershell
cd D:\AI-Image-Lab
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
git checkout cd4c046                       # 本專案實測過的版本
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

模型放到 `SadTalker\checkpoints\`（從 SadTalker 的 GitHub Releases 下載）：
`SadTalker_V0.0.2_256.safetensors`、`SadTalker_V0.0.2_512.safetensors`、
`mapping_00109-model.pth.tar`、`mapping_00229-model.pth.tar`；臉部偵測用的
`alignment_WFLW_4HG.pth`、`detection_Resnet50_Final.pth` 放到 `SadTalker\gfpgan\weights\`。

**ffmpeg**：這台機器的 PATH 上沒有 ffmpeg，所以把 `ffmpeg.exe`（gyan.dev 或 BtbN
的 Windows build）直接放在 `SadTalker\ffmpeg.exe`，或用環境變數 `SADTALKER_FFMPEG`
指到別的位置。`talking_head.py` 會把 SadTalker 資料夾加到子程序的 PATH 最前面，
所以原版 SadTalker 寫死呼叫 `ffmpeg` 也找得到。本機另外對
`SadTalker/src/utils/videoio.py` 做了一個小修補，讓它讀 `SADTALKER_FFMPEG`（SadTalker
資料夾不進 repo，重新 clone 的話這個修補會消失，但有上面的 PATH 處理，不重套也能跑）：

```diff
+FFMPEG_BIN = os.environ.get("SADTALKER_FFMPEG", "ffmpeg")
+
 def save_video_with_watermark(video, audio, save_path, watermark=False):
     temp_file = str(uuid.uuid4())+'.mp4'
-    cmd = r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec copy "%s"' % (video, audio, temp_file)
+    cmd = r'%s -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec copy "%s"' % (FFMPEG_BIN, video, audio, temp_file)
```

選用：臉部修復 `GFPGANv1.4.pth`（~348 MB），不先下載的話第一次選 gfpgan 時會自動抓：

```powershell
curl.exe -L -o "D:\AI-Image-Lab\SadTalker\gfpgan\weights\GFPGANv1.4.pth" https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

### 參數怎麼選

| 參數 | 建議 |
|---|---|
| 尺寸 `--size` | 256 比較快、VRAM 約 2-3 GB；512 比較清楚、約 4-6 GB |
| 範圍 `--preprocess` | `crop`（預設）只輸出臉部裁切；`full`/`extfull` 把動起來的臉貼回整張原圖，半身/全身圖用這個 |
| `--still` | 頭部幾乎不動，搭配 `full` 使用，避免貼回原圖時頭部浮動錯位 |
| `--expression-scale` | 嘴型/表情幅度，1.0 預設，0.5-2.0 |
| `--enhancer` | `gfpgan` / `RestoreFormer` 每一幀做臉部修復，比較清楚但比較慢 |

SadTalker 用 `os.system` 組 ffmpeg 指令，中文路徑在 cp950 主控台下會壞掉，所以
`talking_head.py` 會先把圖跟音檔複製到純英文路徑的暫存資料夾
（`outputs\sadtalker\_raw\<隨機名>`）再跑；非 wav 的音檔會先轉成 16kHz 單聲道 wav。

> 另一個對嘴工具 [MuseTalk](https://github.com/TMElyralab/MuseTalk) 也 clone 在
> repo 根目錄、模型也下載了，但它的 `.venv` 目前沒有裝 torch、跑不起來，所以還
> 沒有接進 CLI/GUI。

## 網頁 GUI（本地生圖介面）

### 前置條件

- 已裝好 GUI 需要的套件（見上面「2b. 一鍵安裝網頁 GUI + 所有選用功能套件」）
- 開發者終端機或 PowerShell

不需要先手動啟動 ComfyUI server——`gui.py` 會在你第一次按下「生成」時自動
偵測、自動啟動，開著 GUI 但沒生圖不會多佔用 ComfyUI 那份記憶體（VRAM/RAM
各約 2GB）。CLI（`generate_character.py`）沒有這個自動啟動，仍需照上面
「啟動 ComfyUI server」章節手動先開。

### 啟動 GUI

#### 方式 1：終端機直接執行（推薦 — 可看實時日誌）

```powershell
cd D:\AI-Image-Lab\training
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe gui.py
```

執行後終端機會輸出類似：
```
* Running on http://127.0.0.1:7861
```

#### 方式 2：背景執行（不佔用終端機）

```powershell
cd D:\AI-Image-Lab\training
Start-Process powershell -ArgumentList '-NoExit','-Command','D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe gui.py'
```

### 使用網頁

GUI 啟動後，在瀏覽器打開：**http://127.0.0.1:7861**（不是 Gradio 預設的
7860——這個 port 被固定改掉是為了避開 kohya_ss 自己的訓練 GUI，見
`training/啟動GUI說明.md`）。

網頁會顯示生圖表單，填完後按「生成」。第一次按下去如果 ComfyUI 還沒啟動，
會先自動啟動它（多等幾秒），之後同一個 session 內都是熱的，通常 30 秒到
1 分鐘內圖片會出現在右側。

### 關閉 ComfyUI（用完釋放記憶體）

`gui.py` 退出時，如果 ComfyUI 是它自己自動啟動的，會一併關閉；如果 ComfyUI
是你自己另開終端機手動啟動的，`gui.py` 不會去動它。想在 GUI 還開著的情況下
單獨關掉 ComfyUI（例如要挪出 VRAM 給 kohya_ss 訓練），跑：

```powershell
powershell -ExecutionPolicy Bypass -File D:\AI-Image-Lab\training\stop_comfyui.ps1
```

下次按「生成」時 GUI 會自動重新啟動它。

### GUI 功能

#### 依角色生圖（以角色身分條件化）

1. **角色**（選填）：下拉選單選一個預定義角色，系統會套用該角色的年齡/外貌/風格描述
2. **Anchor 圖**（選了角色後需填）：該角色的身分基準照，下拉選單會顯示已存檔的圖片，
   選擇後右側會即時預覽
3. **Prompt**：自訂描述（例如 `walking through a cozy library, warm afternoon light`），
   系統會自動接在角色身分描述後面。**建議用英文** — 因為 CLIP tokenizer 對中文
   支援不佳，中文 prompt 容易被模型忽略。習慣打中文（或其他語言）的話，欄位
   下方的「翻譯成英文」按鈕會呼叫 Google Translate 把目前內容轉成英文並直接
   寫回 Prompt 欄位，送出前還可以再自行微調（見下面「自動翻譯」說明）
4. **內容分級**：`safe`（預設，禁止露骨內容）或 `suggestive`（允許泳裝/藝術尺度，
   但露骨性器官/性行為依然封鎖）
5. **Seed**：同樣 seed + prompt 會重現同樣結果
6. **IP-Adapter 權重**：1.0（預設，FaceID 量表）調高會更像角色臉孔、更像 anchor
   圖；調低會更自由發揮。背面鏡頭會自動降權到 0.3
7. **畫布比例**：`自動`（預設，有姿勢參考圖時用直式全身，否則正方形）/
   `正方形 1024×1024` / `直式 832×1216`（全身、背面照建議）/
   `橫式 1216×832`（風景、多人、環境為主的畫面建議）——**實測橫式明顯比直式慢**
   （直式 41s、正方形 62s、橫式 66s，同一組 prompt/seed 條件下測的）。直式跟
   橫式總像素完全一樣（只是長寬對調），差異不是運算量問題，是這張 RTX 2070 對
   特定長寬形狀的 cuDNN/xformers kernel 選擇效率不同，沒有辦法在程式層面修，
   純粹是這張卡的特性，需要橫式構圖時本來就得接受比較慢
8. **風格正/負面詞**（進階，摺疊區塊）：預設值就是程式碼裡 `REALISTIC_STYLE`/
   `REALISTIC_NEGATIVE` 的內容，直接在網頁上編輯即可調整寫實感等風格用詞，
   不需要改程式碼再重啟 GUI。**年齡保護詞跟露骨內容封鎖詞不在這裡、也改不到**
   ——那些永遠固定套用，跟這兩個欄位完全獨立
9. **生成**：按此按鈕，幾秒到 1 分鐘內圖片會出現在右側

#### 純文字生圖（無角色條件化）

- 角色選「(無 - 純文字生圖)」，下面不用選 anchor 圖，直接輸入 prompt 生成

#### 依圖片產生 Prompt（自動描述用上傳圖片）

- 點開「依圖片產生 Prompt」摺疊區塊，上傳一張圖片，按「產生 Prompt」
- 系統會用 BLIP（`Salesforce/blip-image-captioning-base`，透過 `transformers`
  lazy-load）自動生成英文描述，直接填入下方 Prompt 欄位，結果可再自行編輯
- **第一次使用**會自動從 Hugging Face 下載模型（約 1GB），需要網路連線，之後
  就會快取在本機（`~/.cache/huggingface`）不用重新下載
- 刻意跑在 CPU 上，不佔用 GPU VRAM——避免跟 ComfyUI 常駐的 SDXL checkpoint
  搶顯存，導致生成中的圖片 OOM

#### 自動翻譯（多語言輸入）

- Prompt 欄位下方的「翻譯成英文」按鈕，呼叫 Google Translate 的公開端點把
  欄位目前內容翻成英文，結果直接寫回 Prompt 欄位（不是另外開一個顯示框），
  送出生成前還可以再自行編輯
- 自動偵測來源語言，不限中文——已經是英文的話按下去內容不變，其他語言
  （日文、韓文等）也可以直接翻
- **不需要 API key**，但這是 Google Translate 網頁版翻譯小工具用的公開端點，
  不是官方 Cloud Translation API，可能會被 rate-limit 或未來變動——翻譯失敗
  時 GUI 會跳出明確的錯誤訊息（不會靜默失敗、也不會誤導你以為送出的是翻好的
  英文），可以稍後重試或自己手動改打英文
- 選這個方案是實測過的結果：本地小型翻譯模型（`opus-mt-zh-en`、
  `NLLB-200-distilled-600M`）在這種逗號分隔短語（不是完整句子）的 prompt
  風格上表現很差——`opus-mt` 會翻錯個別詞彙（「蓬鬆的棉被」被翻成
  "loose tampons"），`NLLB` 則會自己腦補出原文沒有的敘事內容，兩個都不能用；
  Google Translate 對這種短語輸入的處理正確得多
- 靜態圖片區塊、AnimateDiff 動態影片區塊的 Prompt 欄位都各自有一個翻譯按鈕，
  互相獨立；額外負面詞欄位目前沒有這個功能

#### 自行上傳身分參考圖

- 「Anchor 圖預覽」下方的「或自行上傳身分參考圖」欄位
- 上傳後會優先用上傳的圖作為臉部條件化，覆蓋上面的下拉選單選擇
- **重要**：僅限虛構/AI 生成的角色照，禁止上傳真人照片

#### 骨架姿勢控制（ControlNet）

- 點開「骨架姿勢控制」摺疊區塊，上傳一張姿勢參考照片，需要搭配上面的 anchor 圖
  一起使用
- 系統會用 OpenPose 自動從照片抽取骨架（頭/身體/手部關節位置），再用這個骨架
  精確控制生成結果的姿勢，比純文字描述姿勢準確很多
- **這張照片跟 anchor 圖不一樣**：只會抽取關節骨架座標，不會保留臉部/身分/
  外觀資訊，所以**可以是任何照片**（不限虛構角色），例如網路上找的姿勢參考圖、
  自己拍的照片都可以
- 骨架控制強度預設 0.8（0 = 完全忽略骨架，1 = 嚴格鎖定姿勢、可能犧牲畫質）
- **第一次使用**會自動下載 OpenPose 偵測模型（約 200MB，含身體/手部/臉部三個
  子模型），存在 `ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/` 底下，
  之後不用重新下載。下載+運算過程可能需要 5-10 分鐘（視網路速度），之後每次
  生成就跟平常一樣快

#### 臉部/手部精修（ADetailer）

- 點開「臉部/手部精修」摺疊區塊，勾選核取方塊即可，需要 anchor 圖
- 生成完成後會自動偵測臉部跟手部各自的區域，裁切出來放大重繪一次再貼回去，
  修正 SDXL 常見的臉部細節模糊、手指變形這類小瑕疵——偵測不到（例如手沒入鏡）
  就直接跳過那個步驟，不會報錯
- 精修強度預設 0.5（0 = 該區域完全不變，1 = 該區域從雜訊完全重新生成）
- 臉部偵測方式可選「YOLO 矩形框（預設）」或「MediaPipe 臉部網格」——後者遮罩
  貼合實際臉型輪廓，需要另外裝 `mediapipe` 套件（見上面 ADetailer 安裝章節）
- 會多跑兩次額外的採樣（臉部一次、手部一次），生成時間比平常長，模型本身
  （YOLOv8 nano / MediaPipe）很小，不是效能瓶頸，多花的時間主要來自額外的
  採樣步驟
- **目前不能跟「骨架姿勢控制」同時勾選**，兩者是各自獨立的 workflow，還沒有
  合併版本

#### 模型版本（完整版 / 量化版）

- 在「Checkpoint 模型」上方的摺疊區塊，四個 SDXL / Pony 模型各有一組「完整版 / 量化版
  fp8」選項，標籤會顯示兩個檔案的大小、量化檔轉出來了沒有；按「儲存模型版本設定」寫進
  設定檔，下一張開始生效，「重新檢查檔案」會重讀檔案狀態
- 這是全域設定：單張生圖、批次 GIF 跟 CLI 都套用
- 量化檔還沒轉出時儲存會跳警告並附上轉換指令；生成時如果會改用完整版（沒有量化檔、
  或用了骨架姿勢），也會跳提示
- 這裡不提供「立即轉換」按鈕：轉一個模型要讀約 7 GB、跑一分多鐘，請用 3c 的 CLI 指令

#### AnimateDiff 動態影片

- 頁面最下方獨立區塊，跟上面的靜態圖生成完全分開（不同 checkpoint：SD1.5，
  不是 SDXL），需要先完成「AnimateDiff 動態影片」安裝章節的模型下載
- 上傳一張臉部參考圖（FaceID 用，僅限虛構/AI生成，禁止上傳真人照片），輸入
  prompt，按「生成影片」——輸出是一段 **`.mp4`**（h264），會鎖住整段影片的
  臉部身分，並把所有影格的臉部一起重新採樣精修，修正純 SVD img2vid 常見的
  臉部變形問題
- 「採樣影格數」預設 16（motion module 訓練時的上限，不建議調更高）
- **「畫質 / 流暢度 / 速度」**：
  - 二段式高清（預設開）：512² → 768²，VRAM 不足會自動退回不做高清
  - 最終放大（預設長邊 1024）：逐幀 ESRGAN，不會增加閃爍
  - RIFE 補幀倍數 1/2/4：2/4 要先完成「2c. RIFE 補幀節點」安裝
  - 臉部精修（預設開）：關掉比較快，但臉可能變糊/漂移
  - LCM 快速模式（預設關）：8 步取代 20 步，要先完成「2d. AnimateLCM」下載
- **「FaceID 臉部結構權重」**（預設 1.0）跟 **「動作幅度」**（預設 1.0）：調高權重或
  調低動作幅度都會讓臉更固定，但動作明顯變少，實測見「臉部變形排查」
- 「輸出 FPS」0 = 自動（8 × 補幀倍數，片長一樣約 2 秒、只是變順）；手動調低就
  變成更長的慢動作，例如補幀 ×4 + 16fps = 61 幀、約 3.8 秒
- 第一次執行會比較久：SD1.5 checkpoint、motion module、FaceID SD1.5 模型都是
  全新的模型組合，ComfyUI 還沒快取過
- 「Motion LoRA」摺疊區塊（選用）：控制整個畫面的鏡頭運動（縮放/平移/傾斜/
  旋轉），**不是身體部位的物理晃動效果**——選了要先完成「AnimateDiff Motion
  LoRA」安裝章節的模型下載，沒下載會直接生成失敗

#### 會講話的嘴型影片

- AnimateDiff 區塊下面，上傳人像（**僅限虛構/AI 生成**）+ 語音檔（wav/mp3），
  按「生成說話影片」——輸出含聲音的 `.mp4`
- **不需要 ComfyUI**，跑在 SadTalker 自己的環境；ComfyUI 如果開著會先釋放顯存
- 需要先完成「會講話的嘴型影片（SadTalker）」安裝章節；缺檔案時按下去會直接
  提示缺什麼

#### 批次生成 GIF

- 頁面最下方獨立區塊，跟「自訂生圖」共用同一套 checkpoint/角色/anchor 概念，
  但欄位是各自獨立的（不會互相帶值）
- 輸入 prompt、設定「張數」跟「起始 Seed」，按「生成 GIF」——**第一張是完整
  生成**（建立人物/姿勢/場景），**後面每一張都是拿第一張的圖用低 denoise 的
  img2img 去微調**（同一個 prompt，只有 seed 不同），生成完自動組成一個動態
  GIF。同一個人、同一個姿勢、同一個場景，只有畫面細節隨每張的 seed 有小幅
  變化——這是這個功能設計上要達到的效果：「同一個場景微幅晃動」，不是「每張
  都重新生成、換人換場景」
- **「動作幅度」滑桿**（即 img2img 的 denoise 強度）控制變化大小，**實測數據**
  （對同一張基準圖分別跑 0.3/0.5/0.7，逐像素比對差異）：0.3 幾乎看不出變化
  （只有皮膚/髮絲/布料紋理等級的抖動，19% 像素有輕微差異、幾乎 0% 有明顯
  差異）；0.7 姿勢已經明顯跑掉（差異圖呈現手部/頭部雙重曝光般的殘影，代表
  是兩個不同姿勢疊在一起，不是同一姿勢的小幅變化）。預設 0.45 是這兩個實測
  點之間、還沒驗證過的中間值——覺得太靜止就往上調（0.5-0.6 一帶），姿勢跑掉
  就往下調（0.3-0.4 一帶）。拉到 1 = 完全重新生成，等同這個功能修正前的
  行為（換人換場景），如果就是想要那種效果，把滑桿拉到 1 即可
- **這不是像 AnimateDiff 那樣有時序關聯的平滑動態**——沒有動作模組在張與張
  之間傳遞資訊，是同一張圖反覆用不同 seed 微調出來的效果，看起來比較像原地
  小幅度的浮動/晃動，不是有連貫進展的動作（例如走路的動作不會真的往前走）。
  想要真正平滑、有動作進展的動態，用上面的「AnimateDiff 動態影片」區塊
- 生成速度比 AnimateDiff 快很多——每張都是普通靜態圖生成，沒有 AnimateDiff
  那種多影格疊在一起採樣的額外負擔
- **必須用 SDXL 系列 checkpoint**（不能選 SD1.5）——後面每一張的微調都要靠
  IP-Adapter 鎖住同一張臉，這是 SDXL 專用的模型檔，SD1.5 沒有接這個功能；
  pony 系的品質 tag 前綴一樣會自動加
- 沒選角色/anchor 的話，會拿第一張自己生成出來的圖當作後面幾張 IP-Adapter
  的臉部參考（自己參照自己）——這假設第一張圖裡真的有一張臉，適合拿來生成
  角色，如果 prompt 是純物件/場景（沒有人臉），IP-Adapter 會找不到臉可以
  鎖，建議這種情況改用「自訂生圖」單張生成，不要用這個功能
- 沒有骨架姿勢控制、臉部/手部精修選項——這兩個都沒有接進 img2img 這條新
  workflow，需要這兩項的話請用「自訂生圖」單張生成

## 怎麼生成角色（CLI 命令行模式）

### 列出目前定義的角色

```powershell
cd D:\AI-Image-Lab\training
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe generate_character.py list-characters
```

目前共 11 個虛構角色（`generate_character.py` 的 `CHARACTERS` dict），全部 18-25 歲（`mylora` 例外為 28）：

| trigger | 年齡 | 性別 | 風格 |
|---|---|---|---|
| `mylora` | 28 | 女 | 鄰家女孩、休閒穿搭 |
| `mei` | 19 | 女 | 台灣校園風 |
| `xinyi` | 21 | 女 | 現代極簡通勤 |
| `ruoxi` | 23 | 女 | 潮流街頭 |
| `yuqing` | 25 | 女 | 優雅辦公室穿搭 |
| `wanling` | 18 | 女 | Y2K 風格 |
| `minjun` | 18 | 男 | 台灣街頭風 |
| `junho` | 20 | 男 | 日本簡約風 |
| `taeoh` | 22 | 男 | 韓國都市冷淡風 |
| `hyunjun` | 24 | 男 | 日本極簡精英風 |
| `jungi` | 25 | 男 | 台灣潮牌街頭風 |

> `mylora` 是第一個測試角色，年齡設定較高（28）；其餘 5 女 + 5 男是隨後依需求
> 擴充、涵蓋 18-25 歲不同外觀/風格的獨立角色。

### Stage 1：Anchor（挑選身分基準照）

```powershell
python generate_character.py anchor --character mei --seeds 3001 3002 3003
```

用純 txt2img（無 IP-Adapter）生成幾張候選圖到 `reference_candidates/<character>/`，
人工挑一張最喜歡的長相/氣質當作該角色之後所有變化圖的身分基準（identity anchor）。

### Stage 2：Variations（IP-Adapter 生成完整 dataset）

```powershell
python generate_character.py variations --character mei --anchor "reference_candidates\mei\anchor_seed3001.png" --count 100
```

以 anchor 圖的臉部特徵透過 IP-Adapter 條件化，生成 `--count` 張不同角度/姿勢/
服裝/光線/背景的變化圖，輸出到 `datasets/<character>/`，每張圖同時輸出對應的
caption `.txt`（kohya_ss 訓練用）。已存在的檔案會自動跳過，可安全中斷後續傳。

### 加分測試：擦邊但非露骨內容（單張）

```powershell
python generate_character.py test-suggestive --character mei --anchor "reference_candidates\mei\anchor_seed3001.png"
```

用單獨的 `SUGGESTIVE_NEGATIVE`（只封鎖露骨性器官/性行為/色情字眼，允許泳裝/
若隱若現等藝術尺度），`AGE_SAFETY_NEGATIVE` 依然套用不放寬。單張快速測試用，
批次生成請用下面的 `variations-suggestive`。

### Stage 2b：擦邊內容批次（suggestive-tier dataset）

```powershell
python generate_character.py variations-suggestive --character mei --anchor "reference_candidates\mei\anchor_seed3001.png" --count 20
```

跟 `variations` 同一套續傳邏輯，但用泳裝/運動內衣等詞庫（`SUGGESTIVE_OUTFITS`/
`MALE_SUGGESTIVE_OUTFITS`）+ 海灘/泳池等背景（`SUGGESTIVE_BACKGROUNDS`），輸出
檔名前綴 `sugg_`（跟 `variations` 的 `var_` 前綴共用同一個 `datasets/<character>/`
資料夾不會衝突）。

### 自由輸入 Prompt（單張，CLI 版）

```powershell
python generate_character.py custom --prompt "sitting in a cozy library, reading a book" --character mei --anchor "reference_candidates\mei\anchor_seed3001.png" --tier safe
```

`--character`/`--anchor` 皆可省略（純文字生圖，不接 IP-Adapter）；兩者也可以
分開指定——`--anchor` 給臉部參考圖，`--character` 只是負責在 prompt 前面加上
該角色的身分描述文字，互相獨立。網頁 GUI 的「自訂生圖」用的就是這個函式
（`gen_custom`）。`--anchor` 圖片必須是虛構/AI 生成的角色照，不可以是真人照片。

加上 `--pose-reference` 可以額外用 ControlNet 骨架控制姿勢（需要 `--anchor`）：

```powershell
python generate_character.py custom --prompt "sitting on a park bench, reading a book, autumn leaves" --character xinyi --anchor "reference_candidates\xinyi\anchor_seed3101.png" --pose-reference "..\datasets\xinyi\var_0002_seed2002.png" --tier safe
```

`--pose-reference` 的圖片只會被拿去抽取關節骨架座標，不限定虛構角色（可以是
任何照片，包括既有 dataset 圖片、網路上的姿勢參考圖），跟 `--anchor` 的
「必須虛構角色」規則不同——兩者條件化的內容完全不一樣（骨架 vs 臉部特徵）。
可另外加 `--controlnet-strength`（預設 0.8）調整骨架控制的嚴格程度。

加上 `--use-facedetailer` 可以額外做 ADetailer 式的臉部+手部精修（需要
`--anchor`，不能跟 `--pose-reference` 同時用）：

```powershell
python generate_character.py custom --prompt "standing in a garden, warm afternoon light" --character xinyi --anchor "reference_candidates\xinyi\anchor_seed3101.png" --tier safe --use-facedetailer
```

可另外加 `--facedetailer-denoise`（預設 0.5）調整精修強度，`--facedetailer-backend`
選 `yolo`（預設，矩形框）或 `mediapipe`（臉部網格，需要另外裝 `mediapipe`
套件）。第一次用之前記得先跑過 `python check_adetailer_models.py` 確認三個
YOLO 模型都下載好了（見上面「ADetailer 臉部/手部精修」安裝章節）。

加上 `--style-positive`/`--style-negative` 可以覆蓋預設的 `REALISTIC_STYLE`/
`REALISTIC_NEGATIVE`（只影響這次呼叫，不改程式碼裡的預設值）：

```powershell
python generate_character.py custom --prompt "a cup of coffee on a wooden table" --tier safe --style-positive "shot on iPhone, amateur photo" --style-negative "professional photography, studio lighting"
```

年齡保護詞（`AGE_SAFETY_NEGATIVE`）跟露骨內容封鎖詞（`SAFE_SAFETY_NEGATIVE`/
`SUGGESTIVE_NEGATIVE`）不受這兩個參數影響，永遠固定套用——這兩個參數只調整
風格/寫實感相關的用詞，不是內容分級的開關。網頁 GUI 的「風格正/負面詞」摺疊
區塊做的就是同一件事，省去每次都要打 CLI 參數或改程式碼的麻煩。

### 量化版模型（CLI）

`--variant` 要放在子指令**前面**，只影響這一次執行：

```powershell
python generate_character.py --variant quant custom --checkpoint cyberrealistic_pony --prompt "portrait photo" --seed 9000
```

長期設定用 `quantize_models.py set`（見「3c. 量化版模型」），或設環境變數
`MODEL_VARIANT=full|quant`。實際載入哪個檔案會印在 `[variant]` 那一行。

### Z-Image Turbo（選用安裝，純文字生圖）

[Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)（通義 Tongyi-MAI，6B，Apache 2.0）
是另一種架構的文字生圖模型：寫實度、手指跟畫面裡的文字都比這個專案的 SDXL 好，看得懂
中文 prompt，也寫得出中文字。這個專案**只接了純文字生圖**——它沒有 IP-Adapter / FaceID
版本，所以不能用 anchor 鎖臉，也沒有骨架姿勢控制、HQ 兩段式、臉部/手部精修跟批次 GIF，
選了這些會直接跳錯誤。`--character` 可以用，但只會加上角色的文字描述。

安裝：三個檔案共約 11.3 GB，都在官方 [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)。

| 檔案 | 大小 | 放到 |
|---|---|---|
| `z_image_turbo_int8_convrot.safetensors` | 5.78 GB | `ComfyUI/models/diffusion_models/` |
| `qwen_3_4b_fp8_mixed.safetensors` | 5.24 GB | `ComfyUI/models/text_encoders/` |
| `ae.safetensors` | 0.31 GB | `ComfyUI/models/vae/` |

```powershell
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\models\diffusion_models\z_image_turbo_int8_convrot.safetensors" https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\models\text_encoders\qwen_3_4b_fp8_mixed.safetensors" https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors
curl.exe -L -o "D:\AI-Image-Lab\ComfyUI\models\vae\ae.safetensors" https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors
```

選 int8 / fp8 而不是 bf16 完整版（11.46 GB + 7.49 GB）是因為 8 GB 卡放不下 bf16。RTX 2070
沒有原生 bf16，ComfyUI 會自動改用 fp16 運算，不會退回很慢的 fp32；int8 權重在 Windows 上
走 comfy_kitchen 的 eager（純 PyTorch）路徑，比理想值慢，但可以用。

使用：

```powershell
python generate_character.py custom --checkpoint z_image_turbo --character ruoxi --prompt "sitting by a window in a cafe, soft natural smile, natural daylight" --tier safe
python generate_character.py custom --checkpoint z_image_turbo --prompt "一位25歲的台灣成年女性坐在咖啡店窗邊，手捧寫著「早安」的白色馬克杯，寫實攝影" --tier safe
```

GUI：「Checkpoint 模型」選 `z_image_turbo`。可以選角色（只用文字描述，自動選到的 anchor 圖
會被忽略），但不能自行上傳 anchor、不能選姿勢；四種畫布比例都能用，預設 1024×1024。

取樣照官方範本（8 步、`res_multistep`、`simple`、shift 3），**但 CFG 固定 2**（`ZIMAGE_CFG`，
下限 `ZIMAGE_MIN_CFG` 1.5）：官方範本是 CFG 1，而 CFG 1 時 ComfyUI 會直接略過負面 prompt，
這個專案一律要套的年齡保護負面詞就會失效。並排實測 CFG 2 的畫面跟 CFG 1 幾乎一樣，代價是
每步時間變兩倍。

實測（RTX 2070 8 GB，1024×1024）：

| 情況 | 耗時 | VRAM 峰值 |
|---|---|---|
| ComfyUI 啟動後第一張（載入約 11 GB 模型） | 約 5-7 分鐘 | 7.9 GB |
| 同一個 prompt 再生一張，CFG 1 | 約 75 秒（每步約 9 秒） | 7.7 GB |
| 同一個 prompt 再生一張，CFG 2（預設） | 約 155-195 秒 | 7.7 GB |
| 換 prompt（文字編碼器要重新載入） | 再多 2-4 分鐘 | — |

文字編碼器（5.4 GB）跟主模型（5.9 GB）一起放不進 8 GB，換 prompt 時要來回搬，時間大多花在
搬模型而不是取樣。同一個 prompt 換 seed 時構圖跟長相變化很小，要多樣性請改 prompt，不要只換
seed。同一個測試裡男性角色 taeoh 的 4 張都是男生，沒有 SD1.5 那種男變女的問題。

### 批次生成 GIF（快速預覽，CLI 版）

```powershell
python generate_character.py gif --prompt "standing in a park, looking at the camera" --character mei --anchor "reference_candidates\mei\anchor_seed3001.png" --frames 8 --seed 9000
```

**第一張是完整生成**（建立人物/姿勢/場景），**後面 `--frames`-1 張都是拿
第一張的圖用低 denoise 的 img2img 微調**（同一個 prompt，只有 seed 不同），
組成一個 `.gif`——同一個人、同一個姿勢、同一個場景，只有細節隨每張的 seed
有小幅變化。`--denoise`（預設 0.45，實測 0.3 幾乎靜止、0.7 姿勢已明顯跑掉，
見下面「批次生成 GIF」GUI 說明的實測數據）控制變化幅度：0=完全跟第一張一樣，
1=等於每張都獨立重新生成（換人換場景，這是這個功能修正前的行為）。跟
`custom` 一樣可以省略 `--character`/`--anchor`（這時第一張生出來的圖會自己
當自己的 IP-Adapter 臉部參考），`--style-positive`/`--style-negative`/
`--checkpoint`/`--lora-strength` 的用法也完全一樣，但 **`--checkpoint` 只能
選 SDXL 系列**（不支援 SD1.5，img2img 微調同樣要靠 IP-Adapter 鎖臉）。
**沒有** `--pose-reference`/`--use-facedetailer` 選項——沒有接進這條 img2img
workflow，需要的話用 `custom` 單張生成。

**這不是平滑動態影片**——沒有像 AnimateDiff 那樣在張與張之間傳遞資訊的
動作模組，是同一張圖反覆用不同 seed 微調出來的效果，看起來比較像原地小幅度
的浮動/晃動，不是有連貫進展的動作；想要真正平滑、有動作進展的動態，用下面
的 `video-animatediff`。`--duration-ms`（預設 300）調整 GIF 每張的顯示時間。

### 幫已有的圖片配上動作（SVD img2vid）

```powershell
python generate_character.py video --character mei --init-image "datasets\mei\var_0000_seed2000.png"
```

拿一張既有的 anchor 或 dataset 圖片，用 Stable Video Diffusion 生成短動態影片
（低解析度/低影格數，`svd.safetensors`，只是「動起來測試」用途，非最終畫質）。
**這條路線沒辦法接 FaceID，動態幅度大時臉容易變形/融化**——需要穩住臉部身分
的話用下面的 `video-animatediff`。

### 動態影片、臉部不變形（AnimateDiff + FaceID + 影片臉部精修）

```powershell
python generate_character.py video-animatediff --character mei --face-ref "reference_candidates\mei\anchor_0001.png" --prompt "sitting by a window, gentle breeze, turning head slightly"
```

見上面「AnimateDiff 動態影片」安裝章節。跟 `video` 不同，這是 txt2vid（不是
img2vid）：`--face-ref` 只用來鎖臉部身分（IP-Adapter FaceID），畫面內容完全
由 `--prompt` 決定。加 `--style-positive`/`--style-negative` 可以覆蓋預設的
寫實化用詞（影片預設的負面寫實詞不含 `symmetrical face`，見「臉部變形排查」），
用法跟 `custom` 指令一樣；`--tier`/`--negative-prompt` 的安全詞
保證（年齡保護、露骨內容封鎖）也完全一致，不受這兩個參數影響。輸出是
`animatediff_seed<seed>.mp4`。

畫質/流暢度/速度相關參數：

| 參數 | 預設 | 作用 |
|---|---|---|
| `--no-hires` | （高清開） | 關掉二段式高清 |
| `--hires-scale` / `--hires-denoise` | 1.5 / 0.4 | 高清倍率（面積上限 768²）/ 第二段重繪強度 |
| `--upscale-to {0,768,1024}` | 1024 | 最終逐幀 ESRGAN 放大的長邊，0 = 不放大 |
| `--interp {1,2,4}` | 1 | RIFE 補幀倍數（需要 2c 的節點） |
| `--fps` | 8 × `--interp` | **輸出** fps；調低就變長 |
| `--no-facedetailer` | （精修開） | 跳過影片臉部精修 |
| `--facedetailer-steps` / `--facedetailer-denoise` | 12 / 0.45 | 影片臉部精修的步數 / 重繪強度 |
| `--faceid-v2-weight` / `--faceid-lora-strength` | 1.0 / 0.6 | FaceID 臉部結構權重 / FaceID LoRA 強度；調高臉更固定但動作變少 |
| `--motion-scale` | 1.0 | motion module 強度，1.0 = 完整動作；0.85 就幾乎靜止 |
| `--face-report` | 關 | 生成後逐幀算臉部相似度，並存 `<檔名>_faces.png` 臉部拼圖 |
| `--lcm` / `--lcm-preset` | 關 / `animatelcm` | LCM 8 步快速模式（需要 2d 的模型） |
| `--checkpoint` | `realistic_vision` | 換 SD1.5 checkpoint（`sd15_base`、`cyberrealistic`） |

```powershell
# 最順：補幀 ×2（31 幀 @16fps），1024 輸出
python generate_character.py video-animatediff --face-ref "reference_candidates\mei\anchor_0001.png" --prompt "..." --interp 2
# 約 3.8 秒慢動作：補幀 ×4、16fps（61 幀）
python generate_character.py video-animatediff --face-ref "reference_candidates\mei\anchor_0001.png" --prompt "..." --interp 4 --fps 16
# 最快的預覽：LCM、不高清、不精修、不放大
python generate_character.py video-animatediff --face-ref "reference_candidates\mei\anchor_0001.png" --prompt "..." --lcm --no-hires --no-facedetailer --upscale-to 0
```

### 會講話的嘴型影片（SadTalker）

```powershell
python generate_character.py talk --image "reference_candidates\mei\anchor_seed3001.png" --audio "D:\voice\hello.wav"
```

來源人像**僅限虛構/AI 生成的臉**，禁止真人照片。輸出
`reference_candidates\videos\talk_<圖檔名>_<音檔名>.mp4`（含聲音）。不需要
ComfyUI。常用參數：`--size 256|512`、`--preprocess crop|full|extfull`、`--still`、
`--expression-scale`、`--enhancer gfpgan`，說明見上面「會講話的嘴型影片」安裝章節。

### 加入新角色

編輯 `generate_character.py` 的 `CHARACTERS` dict：

```python
"newname": {
    "age": 20,             # 硬性下限 18（MINIMUM_AGE），低於此值 import 時直接報錯
    "gender": "woman",     # 或 "man" —— 決定用 OUTFITS 還是 MALE_OUTFITS 詞庫、
                            # bikini/swim trunks 等性別相應措辭
    "appearance": "...",   # 髮型、眼睛、臉型、體型
    "style": "...",        # 服裝/整體風格（民族/地域風格描述放這裡）
},
```

dict 的 key（例如 `"newname"`）就是 LoRA 訓練用的 trigger word，會自動出現在
每張圖的 prompt 開頭。不需要改其他程式碼，`anchor` / `variations` /
`test-suggestive` / `variations-suggestive` / `custom` / `video` 這幾個 CLI
子指令、以及網頁 GUI 的角色下拉選單，都用 `--character newname`（或選
`newname`）直接可用。

## Prompt 結構

positive prompt 組成順序：

```
{trigger}, {age} year old adult {gender}, east asian, {appearance}, {style},
{angle}, {pose}, wearing {outfit}, {lighting}, {background}, {REALISTIC_STYLE}
```

- `REALISTIC_STYLE = "shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain"`
  —— 拉走 SDXL 預設偏「精緻插畫/3D 渲染」的觀感，往真實相機質感靠。原本
  「photorealistic, high detail」這類詞對 SDXL 來說常常只換來更光滑的
  「數位藝術感」，效果不如具體的相機/皮膚質感詞。`candid photograph`/
  `film grain` 是後來加的——光靠 DSLR/毛孔這幾個詞，輸出還是常常帶一點
  過度光滑、對稱、打光均勻的「渲染感」，這兩個詞加上下面 negative 的補強
  才更明顯往生活隨拍的方向拉
- `pose` / `outfit` / `lighting` / `background` 各自從
  `POSES` / `OUTFITS`（或 `MALE_OUTFITS`）/ `LIGHTINGS` / `BACKGROUNDS`
  詞庫隨機抽（`random.Random(base_seed)`，seed 固定則結果可重現）
- `POSES` 除了基本站/坐/走之外，也涵蓋躺姿（仰躺屈膝、側躺託腮、趴著撐肘看書、
  趴著托腮翹腳）、瑜珈（坐姿伸展、樹式）、運動（伸展、慢跑）等姿勢多樣性

`angle` 則是從三個獨立詞庫按機率抽（`pick_angle()`），不是單一扁平清單：

| 詞庫 | 內容 | 機率 | 解析度 | IP-Adapter 權重 |
|---|---|---|---|---|
| `BACK_VIEW_ANGLES` | 背面、回望鏡頭、後腦勺+肩膀 | 30% | 832×1216（直式）| 0.3（覆蓋預設值） |
| `FULL_BODY_ANGLES` | 全身照（站姿/走動/遠景/從頭到腳）| 35% | 832×1216（直式）| 沿用呼叫端設定 |
| `CLOSE_ANGLES` | 正面/3-4側/側面/回眸/俯視/仰視/特寫等 | 剩餘部分 | 1024×1024（方形）| 沿用呼叫端設定 |

> 背面鏡頭額外把 IP-Adapter 權重降低——因為背面看不到臉，權重太高的話
> IP-Adapter 為了「保持認得出臉」會硬拗回正面構圖，效果上跟之前 `VIT-G`
> preset 用整張圖（含構圖）做臉部條件化、蓋掉 prompt 想要的姿勢是同一種問題
> 的變形。這個值調過兩次：一開始降到 0.5（跟 PLUS FACE 時代的折半比例一樣），
> 實測發現背面照比例還是偏低、大部分還是正面——FaceID 的身分鎖定比舊版更強，
> 0.5 的降幅不夠壓過臉部條件化的拉力，所以再降到 0.3，機率也從 20% 提高到
> 30%。全身/背面照額外用 832×1216 直式畫布而不是原本的 1024×1024 正方形，
> 是因為正方形畫布沒有足夠垂直空間放下整個身體，會被裁回上半身。

negative prompt（`NEGATIVE_PROMPT`，anchor/variations 都用這個）：

```
nsfw, nude, naked, explicit, sexual content, {AGE_SAFETY_NEGATIVE},
{REALISTIC_NEGATIVE}, lowres, blurry, deformed, extra limbs, bad anatomy,
watermark, text
```

- `AGE_SAFETY_NEGATIVE = "child, children, kid, minor, teen, teenager, underage, young girl"`
  —— 任何內容分級都不會移除這段
- `REALISTIC_NEGATIVE = "3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine"`
  —— 跟 `REALISTIC_STYLE` 互補，一起把畫風往真實相機拉；後半段（smooth/perfect
  skin、symmetrical face、digital art/render/unreal engine）是後來補的，
  直接點名「3D 渲染感」實際上長什麼樣子，比只寫 `3d render, cgi` 更有效

`test-suggestive` 用的是 `SUGGESTIVE_NEGATIVE`（見上），只封鎖露骨部分，
`AGE_SAFETY_NEGATIVE` 一樣不變。

風格 LoRA 目前透過 `LoraLoader` 節點固定套用（所有 `workflow_template*.json`
的 node `"13"`），強度 `strength_model` / `strength_clip` 都是 2.5（該 LoRA
文件標示的建議範圍是 0-5.0；從 2.0 調到 2.5 是為了加強寫實感，見上方
`REALISTIC_STYLE` 的說明）。

### 各 checkpoint 的 prompt 語法差異（實測記錄）

`training/model_prompt_test.py` 用同一句自然語言 prompt，對六個已安裝 checkpoint
各跑一次（Pony 系額外測「有無 `score_9` 標籤」對照），純 txt2img（不掛 FaceID），
用來隔離「checkpoint 本身 + prompt 語法」的差異，排除 FaceID VRAM 換入換出的干擾。
測試圖存在 `training/reference_candidates/model_test/`，完整表格見同目錄
`REPORT.md`；對照合成圖見 `contact_sheet.png`。

測試 prompt（三種家族共用同一句，只有 Pony 多加/不加標籤）：

```
a 25 year old woman with long brown wavy hair, sitting at a wooden cafe table,
holding a ceramic coffee cup, soft window light, cozy interior background,
shot on DSLR, natural skin texture, candid photograph
```

實測結果（RTX 2070 8GB，各 checkpoint 當次是冷載入，含讀檔時間）：

| Checkpoint | 架構 | Prompt 寫法 | 解析度 | 耗時 |
|---|---|---|---|---|
| `juggernaut` | SDXL | 純自然語言 | 1024×1024 | 40.3s |
| `pony` | Pony | 無 `score_9` 標籤 | 1024×1024 | 40.3s |
| `pony` | Pony | 有 `score_9, score_8_up, score_7_up` 前綴 | 1024×1024 | 30.3s |
| `cyberrealistic_pony` | Pony | 無標籤 | 1024×1024 | 38.3s |
| `cyberrealistic_pony` | Pony | 有標籤 | 1024×1024 | 28.2s |
| `pony_realism` | Pony | 無標籤 | 1024×1024 | 42.3s |
| `pony_realism` | Pony | 有標籤 | 1024×1024 | 36.2s |
| `realistic_vision` | SD1.5 | 純自然語言 | 512×768 | 14.1s |
| `cyberrealistic` | SD1.5 | 純自然語言 | 512×768 | 14.2s |

觀察：

- **SDXL（`juggernaut`）與 SD1.5（`realistic_vision`/`cyberrealistic`）**：純自然
  語言直接可用，不需要任何特殊標籤語法。
- **Pony 系（`pony`/`cyberrealistic_pony`/`pony_realism`）**：沒加
  `score_9, score_8_up, score_7_up` 標籤時肉眼可見偏插畫/CG 感；加了之後寫實度
  明顯提升——這正是 `PONY_QUALITY_TAGS` 存在的理由（見上方「Pony-family
  checkpoints」說明），`gen_custom()` 已自動處理，一般呼叫端不用手動加。
- **加標籤反而更快**：三個 Pony checkpoint 加標籤後都省了 8-10 秒，推測是有
  品質標籤時取樣器更早收斂到清晰結構，不用繞路。這是附帶觀察，不是選擇加標籤
  的主要理由（主要理由是畫質）。
- **SD1.5 因為原生解析度只有 SDXL/Pony 的一半（512×768 vs 1024×1024）**，耗時
  只有前者的三分之一左右，跟前面 Benchmark 章節的既有認知一致。
- 這次每個 checkpoint 都是當次第一次載入（要從硬碟讀 6.5-7GB 檔案），
  `juggernaut`/`pony_realism` 沒加標籤那兩筆耗時偏長可能部分反映讀檔時間，
  不完全是純運算差異——同一 checkpoint 內「有無標籤」的相對差距（而非跨
  checkpoint 的絕對耗時）才是這次測試真正想驗證的東西。

### 姿勢/角度標籤實測（整合包）

`training/pose_pack.py` 生成了一份視覺參考包：39 個標籤（20 個 POSES + 19 個角度變體）的實際
產圖示例。所有圖用同一 seed/角色/checkpoint（xinyi × cyberrealistic_pony × seed 88001），
只變標籤，方便逐個對照「這個標籤標籤實際長什麼樣」。

有兩個版本：
- `pose_pack/`：純 txt2img 路徑（~22s/張），速度最快但在全身構圖下臉部無法精修，眼睛出現不對稱
- `pose_pack_facedetailer/`：HQ 路徑只開 FaceDetailer（hires 關掉，~72s/張），臉修乾淨；
  關鍵發現是 FaceDetailer 的實際成本約 **+50s**（比單張 A/B 測的 +14s 高一倍），因為有手入鏡
  時要多跑一次手部重繪

實測發現的限制：
- 5 個 `lying on ...` 標籤在**兩個版本都失敗**，全部塌成「坐著或半躺」，無法真正躺平。這不是
  臉部精修或解析度問題，而是 cyberrealistic_pony 對「躺」這個姿勢的先驗太弱——需要 ControlNet
  給骨架才能壓住，不是調參數能救的
- FULL_BODY_ANGLES 那 5 個標籤在純版幾乎全部失敗（都被裁到腰部），在 FaceDetailer 版大幅改善
- 解析度落差：請求的 832×1216（portrait）實際產出 704×1024（submit_generation_hq 的實際行為和
  docstring 不符；見源碼註解）
- 紅色愛心圖案會無故出現在白 T 上，不是 prompt 要求的——checkpoint 自身的傾向

完整對照表和 contact sheet 見各版本目錄下的 `INDEX.md` 與 `.png` 檔。

## HQ 兩段式生成（預設路徑）

自訂生圖（GUI / `image_api.py` / CLI `custom`）預設走 `submit_generation_hq`
（`workflow_template_hq.json`），一套 workflow 把 FaceID、角色 LoRA、ControlNet、
臉部/手部精修全部合併，並在中間插入 hires 放大。這是「5 分鐘內、不失真」的主路徑。

節點流程：

```
checkpoint → 風格 LoRA(node13, Pony 預設 0.0) → 角色 LoRA(node14, 選填)
  → FaceID(node10/11/12, 選填) → ControlNet(node20-23, 選填, 預設關)
  → KSampler 第一段(node3, 較低解析度, 24 步)
  → VAE decode(node8) → 4x-UltraSharp 放大(node31) → 縮到 1.5x(node32) → VAE encode(node33)
  → KSampler 第二段 hires(node34, denoise 0.4, 20 步) → VAE decode(node35)
  → FaceDetailer 臉(node41, denoise 0.4) → FaceDetailer 手(node43, denoise 0.35)
  → SaveImage(node9)
```

用不到的選填節點會被「改線＋刪節點」繞過（不是設強度 0——LoraLoader 找不到檔會
驗證失敗），所以同一套模板能服務純文字、FaceID、角色 LoRA、ControlNet 的任意組合。

### 為什麼選 ESRGAN 放大，而不是 latent upscale

latent upscale 要 denoise ≥0.55 才蓋得掉插值糊，那正是 FaceID 身分飄移、手被重畫
的區間；ESRGAN 提供真實像素細節，第二段 denoise 只要 0.4 就夠，臉不飄、手不壞。
`4x-UltraSharp` 選用而非 `RealESRGAN_x4plus`，因為後者會把皮膚過度平滑，正是我們要
消除的「塑膠感／3D render」來源。

### 下載 upscale 模型

`4x-UltraSharp.pth`（約 67 MB，[Kim2091/UltraSharp](https://huggingface.co/Kim2091/UltraSharp)），
放到 `models/upscale/`，再依本 repo 慣例用 NTFS hardlink 接進 ComfyUI：

```powershell
New-Item -ItemType HardLink -Path "ComfyUI\models\upscale_models\4x-UltraSharp.pth" -Target "models\upscale\4x-UltraSharp.pth"
```

### 時間預算（RTX 2070 8GB，直式 1056×1536）

| 階段 | 估時 |
|---|---|
| 第一段 base（0.72 MP × 24 步） | ~95 s |
| ESRGAN + VAE encode/decode | ~15 s |
| 第二段 hires（1.62 MP × 8 有效步） | ~70 s |
| FaceDetailer 臉 | ~30 s |
| FaceDetailer 手 | ~30 s |
| 存檔 | ~5 s |
| **合計** | **~245 s（<300 s）** |

> 以上用目前「換入換出」狀態的實測速率估算；`INSIGHTFACE_PROVIDER=CPU`（HQ 預設）
> 消除 FaceID 的 VRAM 帳外佔用後，預期整體降到 ~90 秒。實際數字用
> `python benchmark.py --hq --anchor <臉圖>` 量測（見下方 Benchmark）。

### ControlNet 姿勢骨架庫（修正 checkpoint 壓不住的姿勢）

`training/pose_skeletons.py` + `training/poses/` 提供一組預先畫好的 OpenPose 骨架，
對應 `POSES` 標籤，餵給 HQ 模板既有的 ControlNet（節點 20-23）來鎖定身體姿勢。動機：
上面「姿勢/角度標籤實測」證明 Pony 系 checkpoint 對 5 個 `lying on ...` 標籤完全不聽
文字（都塌成坐姿），只有骨架壓得住。

- 骨架庫 31 個姿勢：15 個從既有 `pose_pack_facedetailer` 產圖用 OpenposePreprocessor
  擷取，5 個躺姿手工重寫關鍵點（擷取出來的是坐姿失敗版），另外 14 個手工新增
  （跪、蹲、盤腿、斜倚、跳躍、揮手、背面回頭等）。`<slug>.json` 是關鍵點來源，
  `<slug>.png` 是 commit 進版控的骨架，產圖時不需要 torch。
- 後面 14 個是**只存在於骨架庫**、刻意不加進 `generate_character.POSES`：資料集
  variations 流程會從 POSES 隨機抽，而那條路徑不掛 ControlNet，把「非骨架不可」的
  姿勢丟進去只會增加失敗圖。它們照樣能用 `--pose`、GUI 下拉、`pose_pack --controlnet`。
- 骨架是**預先畫好的**，直接餵 ControlNet（`pose_is_skeleton=True` 跳過 preprocessor）；
  上傳照片走 `--pose-reference` 才會跑 preprocessor 抽骨架。
- 用法：CLI `--pose <名稱>`、GUI ControlNet accordion 的「姿勢骨架庫」下拉、
  `pose_pack.py --controlnet`（產全套對照）、`benchmark.py --hq --pose <名稱>`。
- **畫布比例必須跟骨架一致**。ControlNet 不會加黑邊，`common_upscale(..., "center")`
  是先把骨架 center-crop 成目標比例再縮放，所以直式骨架配正方形畫布會被裁掉 32% 的
  高度（頭跟腳直接消失，模型自己補），這是這條路徑最容易踩到的失真來源。不指定尺寸
  時會自動採用骨架自己的畫布；明確指定不相容的尺寸會被擋下來而不是靜默裁切
  （CLI/`gen_custom` 拋錯並附上裁切百分比、GUI 跳 `gr.Error`、`benchmark --pose`
  跳過不相容畫布、`pose_pack --controlnet` 退回純文字並印出原因）。容許值是
  `pose_skeletons.CANVAS_CROP_TOLERANCE`（5%）：實際的 832×1216→704×1024 只掉 0.5%，
  正方形掉 32%、橫式掉 53%。

**重複人物**：某些姿勢會生出第二個人。實測 `kneeling_sitting_back_on_heels` 在
cyberrealistic_pony 上三個 seed 中有一個變成兩個人，而且是**第一段**就產生的（把 hires
也接上 ControlNet 不能修，已驗證無效並回退）。負面提示原本完全沒有「只要一個人」的詞，
補上 `multiple people, two people, duplicate, twins, extra person, crowd`（見
`generate_character.QUALITY_NEGATIVE`，safe/suggestive 兩個 tier 都 always-on，
`style_negative` 覆寫不到）後，原本失敗的那個 seed 就正常了。

但**這組負面詞並沒有根除問題**：後續的 checkpoint 矩陣測試（負面詞已生效）裡，pony 與
pony_realism 在同一個跪姿的**純文字**條件下仍然各生出兩個人。同時要更正一個先前的錯誤
推論——我一度把重複歸因於「壓縮骨架留下大片空白」，矩陣結果不支持這個說法：12 個有骨架
的格子全是單人，2 次重複反而都發生在沒有骨架的純文字組。**骨架是抑制重複的，不是造成
重複的。** 重複比較像是特定 checkpoint 加特定姿勢描述的傾向，seed 相關。

註：加了那組負面詞之後產生的圖，跟先前 commit 的 checkpoint 對照表／姿勢整合包不再嚴格
可比，要對照請重跑。

### 骨架庫在各 checkpoint 的實測（4 × 3 × 2 矩陣）

4 個 SDXL 系 checkpoint × 3 個姿勢 × 純文字/骨架，seed 88001，**兩種條件 prompt 完全
相同**（骨架的 `prompt_hint` 在純文字組也照加），唯一變數是 ControlNet。兩個 SD1.5
checkpoint 不列入，control-lora 是 SDXL 形狀接不上。

| 姿勢 | juggernaut | pony | pony_realism | cyberrealistic_pony |
|---|---|---|---|---|
| lying on side（純文字） | 靠牆半坐 | 大致躺著 | 側坐 | 側坐 |
| lying on side（骨架） | 躺 | 躺 | 躺 | 躺 |
| kneeling（純文字） | 蹲非跪 | **兩個人** | **兩個人** | 跪 |
| kneeling（骨架） | 跪 | 跪 | 跪 | 跪 |
| squatting（純文字） | 坐在箱子上 | 蹲 | 蹲 | 蹲 |
| squatting（骨架） | 低蹲（仍最弱） | 蹲 | 蹲 | 蹲 |

結論：

- **沒有任何一個 checkpoint 靠純文字是可靠的**，只是各自壞在不同姿勢。12 個純文字格子
  有 6 個明顯失敗，12 個骨架格子全部正確且單人。骨架庫對四個 checkpoint 都有用，不是
  只有 cyberrealistic_pony 需要。
- **失敗模式因 checkpoint 而異**：juggernaut 傾向替換成「比較安全」的姿勢（把蹲畫成坐在
  箱子上），Pony 系則傾向多生一個人。
- **骨架的成本與 checkpoint 無關**，四個都一致增加約 20-30 秒。
- pony（base）即使給寫實 prompt 仍偏插畫風，這是它本來的特性，與骨架無關。

**耗時修正（實測完成）**：舊版這裡寫「單張約 390 秒、預設關閉」，那是掛 FaceID + 舊
`INSIGHTFACE_PROVIDER=CUDA` 換入換出時代的手估值，從未實測。`pose_pack --controlnet`
實測完整 20 pose 套件（HQ + FaceDetailer + 骨架庫，不掛 FaceID）：冷啟 84.5s（含
control-lora 載入）、中位數 102.5s（其餘 19 張）、平均 100.6s。遠低於 300 秒目標——
ControlNet 本身不是瓶頸。強度掃描（0.6/0.8/1.0）三個值都能讓躺姿正確躺下，預設沿用
`CONTROLNET_STRENGTH=0.8`。

## 生成參數

`comfyui_client.py` 頂部的集中設定：

| 參數 | 值 |
|---|---|
| CHECKPOINT（資料集流程） | `juggernaut_xl_v9_photo.safetensors`（`gen_anchors`/`gen_variations` 用） |
| 自訂生圖/GUI 預設 checkpoint | `CyberRealisticPony_V18.0_F16.safetensors`（`DEFAULT_CUSTOM_CHECKPOINT`，Pony 系寫實） |
| WIDTH × HEIGHT | 1024 × 1024（可調，見 `benchmark.py` 的 768 測試） |
| BATCH_SIZE | 1（8GB VRAM 卡死限制，不做多圖 batch） |
| STEPS | 30 |
| CFG | 6.0 |
| SAMPLER | `dpmpp_2m` |
| SCHEDULER | `karras` |
| IP_ADAPTER_PRESET | `FACEID PLUS V2` |
| IP_ADAPTER_WEIGHT | 1.0（FaceID 自己的權重量表，背面鏡頭自動降為 0.3） |
| 風格 LoRA 強度 | 2.5（`sdxl_photorealistic_slider_v1-0`，建議範圍 0-5.0） |
| FACEDETAILER_DENOISE | 0.5（ADetailer 臉部/手部精修，選用功能才會用到） |
| **HQ 路徑（`submit_generation_hq`，預設）** | 兩段式 base→ESRGAN hires→臉/手精修 |
| UPSCALE_MODEL | `4x-UltraSharp.pth`（4x ESRGAN，皮膚細節優於 RealESRGAN） |
| HIRES_SCALE / HIRES_DENOISE / HIRES_STEPS | 1.5 / 0.4 / 20 |
| HQ_BASE_STEPS | 24（第一段），第一段解析度＝最終÷1.5 |
| 最終解析度 | 正方 1248×1248、直式 1056×1536、橫式 1536×1056 |
| FACEDETAILER_FACE / HAND_DENOISE | 0.4 / 0.35 |
| CHARACTER_LORA_STRENGTH | 0.8（角色 LoRA，node 14，與 FaceID 並用） |
| INSIGHTFACE_PROVIDER | `CPU`（避開 onnxruntime 佔用 torch 帳外 VRAM 造成的換入換出，可用環境變數覆蓋） |
| ANIMATEDIFF_CHECKPOINT | `Realistic_Vision_V6.0_NV_B1_fp16.safetensors`（SD1.5，跟上面 SDXL 的 CHECKPOINT 分開） |
| ANIMATEDIFF_MOTION_MODULE | `mm_sd_v15_v2.ckpt` |
| ANIMATEDIFF_WIDTH × HEIGHT | 512 × 512（SD1.5 原生解析度，第一段採樣尺寸） |
| ANIMATEDIFF_FRAMES / FPS | 16（motion module 訓練上限）/ 8（採樣影格的 fps；補幀時輸出 fps 預設 8 × 倍數） |
| ANIMATEDIFF_STEPS / CFG | 20 / 7.5（SD1.5 慣用範圍，比 SDXL 的 30 步/6.0 少/高） |
| ANIMATEDIFF_HIRES_SCALE / DENOISE / STEPS | 1.5 / 0.4 / 10（面積上限 `ANIMATEDIFF_HIRES_MAX_PIXELS` = 768²） |
| ANIMATEDIFF_UPSCALE_TO | 1024（最終逐幀 ESRGAN 的長邊，0 = 不放大） |
| RIFE_NODE / RIFE_CKPT | `RIFE VFI` / `rife47.pth`（補幀倍數 1/2/4） |
| ANIMATEDIFF_VIDEO_CRF | 20（mp4/h264，數字越小畫質越好、檔案越大） |
| ANIMATEDIFF_LCM_PRESETS | `animatelcm`（預設）/ `lcm_lora`：8 步、CFG 2.0、`lcm`/`sgm_uniform` |
| LCM_MIN_CFG | 1.5（CFG = 1 時 ComfyUI 會跳過 negative，安全負面詞會失效，不能再低） |
| ANIMATEDIFF_FACEID_V2_WEIGHT / FACEID_LORA_STRENGTH | 1.0 / 0.6（調高會讓動作明顯變少，見「臉部變形排查」） |
| ANIMATEDIFF_MOTION_SCALE | 1.0（motion module 時序注意力強度；不是 1.0 時才注入節點 `62`） |
| SD15_GENDER_WEIGHT | 1.3（`generate_character.py`；SD1.5 路線的角色性別字寫成 `(man:1.3)`，見「男性角色被畫成女生」） |
| QUANT_MODEL_KEYS / QUANT_SUFFIX | juggernaut、pony、cyberrealistic_pony、pony_realism / `.fp8q.safetensors`（`comfyui_client.py`；設定檔 `training/settings/model_variants.json`，環境變數 `MODEL_VARIANT`，見「3c. 量化版模型」） |
| ANIMATEDIFF_FACE_CROP_FACTOR / FACE_GUIDE_SIZE | 1.5 / 512（影片臉部精修的裁切倍數 / 臉部重繪尺寸） |
| ANIMATEDIFF_FACEDETAILER_STEPS / DENOISE | 12 / 0.45（影片專用；靜態圖仍用 FACEDETAILER_DENOISE 0.5） |

> - `FACEID PLUS V2`：用 `IPAdapterUnifiedLoaderFaceID` + `IPAdapterFaceID` 這組
>   節點（不是舊版的 `IPAdapterUnifiedLoader` + `IPAdapterAdvanced`），身分條件化
>   靠 InsightFace 的人臉辨識嵌入，而不是 CLIP-vision 影像相似度——同一個角色
>   跨圖的臉孔一致性明顯更好（實測：兩張不同姿勢/背景/服裝的生成圖，臉孔可以
>   清楚辨識是同一個人；換成舊版 PLUS FACE 常常變成兩個不同人）
> - 權重量表跟舊版不一樣：FaceID 的 `weight` 範圍是 -1~3（預設 1.0），舊版
>   PLUS FACE 的量表大概是 0~1.5。背面鏡頭因為看不到臉，權重過高一樣會拉回
>   正面，本來降到 0.5（50% 折半）還是不夠，實測背面照比例還是偏低，後來
>   再降到 0.3——FaceID 的身分鎖定比舊版更強，需要更大的降幅才壓得住
> - `IPAdapterUnifiedLoaderFaceID` 額外需要 `lora_strength`（固定 0.6，寫死在
>   workflow JSON 裡，跟隨 LoRA 一起載入不需要另外用 `LoraLoader` 節點）跟
>   `provider`（固定 `CUDA`）兩個欄位，`IPAdapterFaceID` 節點也多一個
>   `weight_faceidv2` 參數（固定 1.0；AnimateDiff 影片路線可以用 `--faceid-v2-weight`、
>   `--faceid-lora-strength` 調整，見「臉部變形排查」）
> - 需要 `ip-adapter-faceid-plusv2_sdxl.bin` + 對應 LoRA + InsightFace 套件
>   （見上面安裝章節）；`CLIP Vision (ViT-H)` 檔名規則沒變，還是同一份
> - 風格 LoRA 強度從 2.0 調到 2.5，是為了讓輸出更往「相機拍出來的照片」的
>   方向拉，減少 SDXL 預設偏向的光滑「渲染感」——同時 `REALISTIC_STYLE`/
>   `REALISTIC_NEGATIVE`（見上面 Prompt 結構章節）也補了更明確的詞彙
>   （candid photograph、film grain、smooth/perfect skin、symmetrical face 等）

## Benchmark

```powershell
python benchmark.py
```

背景 thread 每 0.3 秒輪詢 `/system_stats` 抓 VRAM/RAM 峰值（單次前後快照會漏掉
生成過程中的瞬間峰值），比較第 1/5/10 張圖的 VRAM 判斷有無 leak。

已測結果（RTX 2070 8GB）：

| 測試 | 解析度 | VRAM | 平均耗時 | Leak |
|---|---|---|---|---|
| Test A | 768×768 ×10 張 | 5192 → 6280 MB，穩定 | 20.4 s/張 | 無（+0 MB） |
| Test B | 1024×1024 ×10 張 | 6280 → 5192 MB，穩定 | 36.5 s/張 | 無（+0 MB） |

HQ 兩段式（`benchmark.py --hq --anchor <臉圖>`，逐階段計時 + 300 秒 pass/fail）：

| 測試 | 最終解析度 | VRAM 峰值 | 平均耗時 | <300s |
|---|---|---|---|---|
| hq_square | 1248×1248 | 待測 | 待測 | 待測 |
| hq_portrait | 1056×1536 | 待測 | 待測 | 待測 |

> 加 `--insightface-provider CUDA` 與預設的 CPU 對照，可量出 FaceID 的 VRAM
> 換入換出對耗時的影響。ControlNet 骨架庫版本用 `--pose <名稱>` 量（不掛 FaceID
> 時實測約 60-90 秒，見上方「ControlNet 姿勢骨架庫」）。

## 本地顯卡不夠力時：雲端 GPU

文字生圖/圖生圖本地這張卡吃得消，影片生成（`video`/`video-animatediff`）和
kohya_ss LoRA 訓練還撐不住——什麼時候該搬去 Replicate、什麼時候該搬去
RunPod、怎麼接上現有的 `comfyui_client.py`，見 [CLOUD_GPU.md](CLOUD_GPU.md)。

要把網頁部署到 AWS Amplify、GPU 運算外包給 Replicate/RunPod 的架構規劃
（`gui.py`/`image_api.py` 哪些能重用、哪些要重寫），見
[WEB_DEPLOYMENT.md](WEB_DEPLOYMENT.md)——目前只是規劃文件，還沒有對應實作。

## 資料夾結構

```
AI-Image-Lab/
├── ComfyUI/                  # 第三方安裝，不進 repo（見安裝章節）
├── SadTalker/                # 第三方安裝（對嘴影片），不進 repo（見「會講話的嘴型影片」章節）
├── MuseTalk/                 # 第三方安裝（對嘴，尚未接進 CLI/GUI），不進 repo
├── models/                   # SDXL/IP-Adapter/CLIP 權重，不進 repo（見模型章節）
├── training/
│   ├── generate_character.py # CLI 入口：角色定義 + prompt 組裝
│   ├── comfyui_client.py     # ComfyUI HTTP client（不 import torch）
│   ├── gui.py                # Gradio 本地網頁 GUI（localhost:7861，會自動啟動/關閉 ComfyUI）
│   ├── stop_comfyui.ps1      # 手動關閉 ComfyUI server，釋放 VRAM/RAM
│   ├── caption_image.py      # BLIP 圖像描述模型（用於自動產生 prompt）
│   ├── translate_prompt.py   # Prompt 自動翻譯成英文（Google Translate 公開端點）
│   ├── check_adetailer_models.py # ADetailer YOLO 模型自動檢查/下載腳本
│   ├── benchmark.py          # VRAM/RAM 記憶體洩漏測試
│   ├── quantize_models.py    # 把 SDXL/Pony checkpoint 轉成 fp8 量化版 + 版本設定 CLI（用 ComfyUI 的 venv 跑）
│   ├── pose_skeletons.py     # OpenPose 骨架庫（poses/ 裡的 json/png）
│   ├── pose_pack.py          # 姿勢/角度標籤參考包產生器（每個標籤一張圖）
│   ├── model_prompt_test.py  # 各 checkpoint 的 prompt 語法探索腳本（手動跑，不在流程裡）
│   ├── image_api.py          # FastAPI 包裝 gen_custom()，給外部專案用 HTTP 呼叫
│   ├── runpod_bundle.py      # RunPod LoRA 訓練的 Windows 端打包/安裝工具
│   ├── runpod_train.sh       # 在 RunPod pod 上跑的訓練腳本
│   ├── workflow_template.json             # IP-Adapter workflow（variations 用）
│   ├── workflow_template_hq.json          # HQ 兩段式 workflow（含 ControlNet 節點 20-23）
│   ├── workflow_template_txt2img_sd15.json   # SD1.5 純 txt2img workflow
│   ├── workflow_template_txt2img_zimage.json # Z-Image Turbo workflow（三個模型檔）
│   ├── workflow_template_txt2img.json     # 純 txt2img workflow（anchor 用）
│   ├── workflow_template_img2img.json     # IP-Adapter + img2img workflow（gif 指令的 wiggle 幀用）
│   ├── workflow_template_controlnet.json  # IP-Adapter + ControlNet 骨架 workflow
│   ├── workflow_template_facedetailer.json # IP-Adapter + ADetailer 臉部/手部精修 workflow（YOLO）
│   ├── workflow_template_mediapipe_facedetailer.json # 同上，臉部偵測改用 MediaPipe 網格
│   ├── workflow_template_img2vid.json     # SVD img2vid workflow（video 指令用）
│   ├── workflow_template_animatediff_facedetailer.json # SD1.5 AnimateDiff + FaceID + 影片臉部精修 + 高清/放大/補幀/mp4（video-animatediff 用）
│   ├── talking_head.py       # SadTalker 對嘴影片包裝（subprocess 呼叫 SadTalker 自己的 venv，talk 指令/GUI 用）
│   ├── face_similarity.py    # 影片逐幀臉部相似度 + 臉部拼圖（InsightFace，用 ComfyUI 的 venv 跑）
│   ├── config/               # kohya_ss 訓練設定（mylora.toml 等，本地那次的產物）
│   ├── logs/                 # kohya_ss 訓練 log（train_mylora.*.log）
│   ├── poses/                # 骨架庫的 json + 預覽圖 + contact sheet
│   ├── settings/             # 本機設定（model_variants.json），不進 repo
│   └── reference_candidates/ # anchor 候選圖，不進 repo
├── datasets/<character>/     # 生成的訓練圖 + caption，不進 repo
├── outputs/_comfyui_raw/     # ComfyUI 原始輸出暫存，不進 repo
├── outputs/sadtalker/_raw/   # SadTalker 每次執行的暫存資料夾（跑完自動刪除），不進 repo
├── captions/ evaluation/ samples/  # 保留給 kohya_ss LoRA 訓練階段用，目前都是空的
├── web/                      # AWS Amplify Gen 2 前端骨架（見 WEB_DEPLOYMENT.md）
├── worker/                   # RunPod serverless worker（Dockerfile + handler）
├── comfyui-requirements.lock.txt   # ComfyUI 本體套件凍結版本清單
├── comfyui-requirements-extra.lock.txt   # 網頁 GUI + 所有選用功能（ControlNet/ADetailer/mediapipe）套件凍結版本清單
├── CHANGELOG.md               # 每次改動的結論與實測數字
├── CLAUDE.md                  # 給 Claude Code 的專案速覽（硬體限制、常用指令、慣例）
├── CLOUD_GPU.md               # 本地顯卡不夠力時的雲端 GPU 參考（Replicate/RunPod）
└── WEB_DEPLOYMENT.md          # 網頁部署規劃（AWS Amplify + Replicate/RunPod，尚無實作）
```

## 疑難排解

### ComfyUI 跟 kohya_ss LoRA 訓練不能同時執行

**症狀**：網頁 GUI 突然生不出圖片、連線錯誤，但 GUI process 本身還在跑
（`localhost:7861` 打得開），問題出在 ComfyUI（`localhost:8188`）連不上。

**原因**：這兩個都要用同一張 8GB VRAM 的 RTX 2070。ComfyUI 常駐時已經佔用
checkpoint(6.6GB)+ FaceID(1.49GB+372MB LoRA)+ CLIP vision(2.5GB) 等模型；
kohya_ss 的 `sdxl_train_network.py` 訓練 LoRA 時，載入 SDXL checkpoint + VAE +
optimizer state 也需要接近滿版的 VRAM。兩個 process 同時搶同一張卡，輕則其中一個
OOM 崩潰，重則兩個都跑不動或系統整個卡住。**這是本專案這次疊代時實際發生過的
狀況**：為了跑 kohya_ss 訓練測試，手動把 ComfyUI process 關掉讓訓練獨佔顯存，
這段時間網頁 GUI 自然連不上、生不出圖——不是 bug，是預期行為。

**處理方式**：兩邊只能擇一運行，需要切換時：

```powershell
# 停止 ComfyUI（訓練前）
powershell -ExecutionPolicy Bypass -File D:\AI-Image-Lab\training\stop_comfyui.ps1
```

訓練跑完/中斷後要恢復生圖：如果是用網頁 GUI，直接按「生成」就會自動重新
啟動 ComfyUI；如果是用 CLI，要手動重啟：

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
```

**如何快速判斷是不是這個問題**：`curl http://127.0.0.1:8188/system_stats` 打
不通（connection refused / timeout）就是 ComfyUI 沒在跑，不是生成邏輯出錯。

### kohya_ss 訓練：RTX 2070 不支援 bf16

`sdxl_train_network.py` 用 `--mixed_precision bf16` 會在 VAE encode 階段丟出
`xformers` 的 `NotImplementedError`（`bf16 is only supported on A100+ GPUs`）。
RTX 2070 是 Turing 架構（compute capability 7.5），bf16 tensor core 要 Ampere
（8.0）以上才有。**改用 `--mixed_precision fp16` + `--save_precision fp16`**。

> 换成 fp16 後又發現另一個問題：訓練 loss 從第一步開始就一直是 `NaN`，這是
> SDXL 在 fp16 下的已知問題（VAE 數值容易溢位），已知的標準解法是加
> `--no_half_vae`（VAE 維持 fp32，UNet/text encoder 照常用 fp16），但**這個
> 修正還沒有實際跑過驗證**，只是根據症狀對照到已知問題模式判斷出來的，下次
> 繼續訓練測試時要先確認這個 flag 真的解決了 NaN 的問題。
>
> **現在的做法**：不在本地用 fp16 硬撐，改到 RunPod 用 bf16 + `--no_half_vae`
> 訓練（見下方「在 RunPod 訓練角色 LoRA」），bf16 本身就避開了 fp16 VAE 溢位。

### 在 RunPod 訓練角色 LoRA（bf16 解決 NaN）

本地 RTX 2070 不支援 bf16、fp16 又會讓 loss 從第一步就變成 NaN，所以角色 LoRA
一律搬到 RunPod 用有 bf16 的卡（RTX 4090 / A100）訓練，訓好把 `.safetensors`
拉回本地，和 IP-Adapter FaceID **並用**（LoRA 扛身分、FaceID 只修飄移）。

**目前狀態**：還沒有訓練出可用的角色 LoRA。11 個角色的 `lora` 欄位全部是 `None`，
身分完全靠 IP-Adapter FaceID。`models/lora/` 現有兩個檔案，都沒有登記給任何角色、
生圖流程不會用到：

| 檔案 | 來源 | 內容 |
|---|---|---|
| `mylora-000001.safetensors` | 2026-08-15 本地那次失敗的訓練 | SDXL base 1.0、dim 32 / alpha 16、`datasets/character` 100 張、預定 10 epoch / 1000 步。只跑到 100 步（10 分 5 秒、6.05 秒/步）就中止，而且**整段 `avr_loss=nan`**，這個檔案是在 NaN 狀態下存的第 1 個 epoch 快照，不能用 |
| `dsfutaba_pd6.safetensors` | 外部下載 | Pony V6 底模的風格 LoRA（2024-02-15、dim 16 / alpha 8、trigger `dsfutaba`），放著參考用 |

那次的設定在 `training/config/`（`mylora.toml`、`dataset_mylora.toml`），log 在
`training/logs/train_mylora.{out,err}.log`——`out.log` 可以看到當時是 fp16 mixed
precision 再加 `enable fp8 training for U-Net / Text Encoder`。RunPod 上 bf16 +
`--no_half_vae` 的組合到目前為止還沒有實際跑過，`models/lora/` 裡也沒有任何來自 RunPod
的產出，所以下面的步驟是照 kohya 的已知解法寫的，第一次跑要照步驟 4 盯住前 20 步的
`avr_loss`。

底模用 **Pony V6 base**（`ponyDiffusionV6XL_v6StartWithThisOne.safetensors`），
因為推論用的是 cyberrealistic_pony / pony_realism 等 Pony 系；Pony V6 重訓過
text encoder，用 SDXL base 訓的 LoRA 套到 Pony 上會弱或變形。要給 Juggernaut 用
就以 `--base sdxl` 再訓一次（不到 $1）。

步驟：

1. **補資料集（若太少）**：目前 `wanling` 10 張、`xinyi` 9 張、`yuqing` 9 張、
   `ruoxi` 4 張（`datasets/character` 另有 100 張，是 `mylora` 那次用的），
   先在本地補到 40-60 張：
   ```powershell
   python generate_character.py variations --character xinyi --anchor <anchor.png> --count 60
   ```
2. **打包**（自動排除本地 fp16 產生的 `.npz` 快取，並依資料集張數算好 num_repeats）：
   ```powershell
   python runpod_bundle.py pack --character xinyi --base pony
   ```
3. **開 pod**：RunPod 控制台開一台 RTX 4090（Community，約 $0.34/hr），選官方
   PyTorch template，掛一個 30GB Network Volume（模型與輸出持久化）。先把 Pony V6
   base 下載到 `/workspace/models/`（一次即可）。
4. **上傳並訓練**：
   ```
   # Windows：runpodctl send 印出一次性代碼
   runpodctl send training\runpod_bundle\xinyi.zip
   # Pod：
   cd /workspace && runpodctl receive <代碼> && unzip -o xinyi.zip
   bash runpod_train.sh xinyi pony
   ```
   約 1000 步、4090 上 ~30 分鐘、不到 $1。**前 20 步先確認 `avr_loss` 是有限值
   （0.05-0.25）不是 NaN**；bf16 + `--no_half_vae` 正是修 NaN 的關鍵。
5. **挑 epoch**：看 `output/xinyi/` 每個 epoch 的 sample 圖，通常第 6-8 個 epoch
   在身分穩定與過擬合之間最平衡；`runpodctl send` 把該檔傳回。
6. **安裝並註冊**：
   ```powershell
   python runpod_bundle.py install --file xinyi_pony_v1-000007.safetensors --character xinyi --base pony
   ```
   會複製到 `models/lora/` 並 hardlink 進 `ComfyUI/models/loras/`，再把印出來的
   `lora` 欄位貼進 `generate_character.py` 的 `CHARACTERS["xinyi"]`。之後 HQ 生圖
   選這個角色就會自動套 LoRA，並把 FaceID 權重降到 0.7。
7. **用完務必停止 pod**（只留 Network Volume），否則按小時持續計費。

> 本地 `datasets/character` 裡的 `.npz`（latent / text-encoder 快取）是舊 fp16
> 那次 NaN 產生的，kohya 只驗 shape 不驗內容，且對 Pony 的 CLIP 也無效——打包時
> 已自動排除，pod 上會乾淨重算。

### 生成中途卡住 / VRAM、RAM 用量偏低導致崩潰

這台機器現在有 32GB 系統 RAM。早期只有 17GB，Chrome（多分頁）+ VS Code 背景就可能吃掉
8GB+，曾經兩次在批次生成到一半時因為系統記憶體被榨乾而讓 ComfyUI process 直接崩潰（不是
單一個生成請求逾時，是整個 server process 消失，`curl /system_stats` 連不上）。加到 32GB
之後沒再崩潰過，但還是會吃緊：跑 GUI 預設高清時完整版 checkpoint 就佔 15.5-15.8GB，系統
只剩約 4GB 可用、開始用分頁檔，同一批的第三張因此從 167 秒變成 530 秒（量化版可以省下約
3GB，見「3c. 量化版模型」）。
`gen_variations`/`gen_suggestive_variations` 都是續傳邏輯（依已存在檔案數判斷
從哪裡繼續），崩潰後不會遺失進度，重啟 ComfyUI 後重新執行同一條指令即可從中斷點
繼續，不會重跑已完成的部分。長時間批次生成前，建議先關閉不必要的背景程式騰出
系統 RAM。

### 影片生成：高清階段 VRAM 不足

終端機出現 `[warn] hires pass ran out of VRAM - ... retrying once without hires`
代表 768² 的第二段採樣在這張卡上塞不下（通常是別的程式也在用顯存，或底圖不是
正方形），程式已經自動改成不做高清、只做最終 ESRGAN 放大。想保留高清的話用
`--hires-scale 1.25`（約 640²），或先關掉其他佔顯存的程式。

### 影片生成：找不到 `RIFE VFI` 節點

選了補幀 ×2/×4 但 ComfyUI 沒有這個節點：照「2c. RIFE 補幀節點」clone 之後**要
重啟 ComfyUI** 才會載入。如果節點有了但生成時卡在下載 `rife47.pth`，用 2c 裡的
curl 指令手動下載。

### 對嘴影片：SadTalker 失敗

- `找不到 ffmpeg`：把 `ffmpeg.exe` 放到 `SadTalker\ffmpeg.exe`，或設定 `SADTALKER_FFMPEG`
- 選 gfpgan 後卡住很久：第一次會下載 `GFPGANv1.4.pth`（~348 MB），可以先照安裝
  章節手動下載
- 完整錯誤訊息會印在啟動 GUI/CLI 的終端機裡（GUI 只顯示最後一段）

## Roadmap（尚未完成）

- **kohya_ss LoRA 訓練階段**：本地 fp16 會 NaN，已改為在 RunPod 用 bf16 +
  `--no_half_vae` 訓練（見「在 RunPod 訓練角色 LoRA」章節，`runpod_bundle.py` +
  `runpod_train.sh`）。訓好的 LoRA 與 FaceID 並用，`CHARACTERS` 的 `lora` 欄位
  預設為 `None`，安裝後填入——目前 11 個角色全部還是 `None`，也還沒有成功的訓練產出
  （見「在 RunPod 訓練角色 LoRA」的目前狀態）。`captions/` / `evaluation/`（下有
  epoch01-10 空資料夾）/ `samples/` 仍是預留空資料夾
- 早期用 `diffusers` 直接載入模型時曾發生過一次無 traceback 的批次崩潰
  （生成到第 37 張左右 exit code 1）。改用本 repo 現在的 ComfyUI + API 架構後
  未再重現（100+ 張連續生成、benchmark 皆無異常），但當時沒有做根因診斷，
  記錄在此供之後參考
