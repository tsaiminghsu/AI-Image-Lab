# AI Image Lab

用 **ComfyUI + SDXL + IP-Adapter** 生成虛構（非真人）角色的多角度訓練圖片資料集，
供之後用 [kohya_ss](https://github.com/bmaltais/kohya_ss) 訓練角色 LoRA。

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
| GPU | NVIDIA RTX 2070, 8 GB VRAM |
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

### 4. 啟動 ComfyUI server（保持常駐）

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
```

這個 process 要一直開著——`training/` 底下的腳本都是透過 HTTP 呼叫它，
不會自己啟動或關閉 ComfyUI。

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
beta、VRAM 需求也重很多），所以這條路線用官方 SD1.5 base checkpoint，不是
目前生圖用的 Juggernaut SDXL：

| 模型 | 用途 | 大小 | 來源 | 存放位置 |
|---|---|---|---|---|
| `v1-5-pruned-emaonly.safetensors` | SD1.5 底模 | ~4.3 GB | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `ComfyUI/models/checkpoints/` |
| `mm_sd_v15_v2.ckpt` | AnimateDiff motion module | ~1.8 GB | [guoyww/animatediff](https://huggingface.co/guoyww/animatediff) | `ComfyUI/models/animatediff_models/` |
| `ip-adapter-faceid-plusv2_sd15.bin` | FaceID（SD1.5 版） | ~150 MB | [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) | `ComfyUI/models/ipadapter/` |
| `ip-adapter-faceid-plusv2_sd15_lora.safetensors` | FaceID 隨附 LoRA | ~40 MB | 同上 | `ComfyUI/models/loras/` |

寫實感靠 prompt/negative 調整（跟 SDXL 那條路線一樣，見「風格正/負面詞」），
不是靠底模——`v1-5-pruned-emaonly` 是官方原始權重，不需要登入/授權就能下載，
選它是為了穩定可重現，不是因為畫質最好。

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

### 3. 重啟 ComfyUI

```powershell
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
```

用法見下面「幫已有的圖片配上動作」章節旁邊的 `video-animatediff` CLI 指令，
或網頁 GUI 最下面的「AnimateDiff 動態影片」區塊。**跟靜態圖那些選項是分開的
獨立 workflow**（不同 checkpoint、不同 IP-Adapter 模型檔），第一次執行會比較
慢（新的模型組合，ComfyUI 還沒快取過）。

> 只做了臉部 FaceDetailer 這一道精修（沒有另外接手部），影片本身的採樣＋
> 16 張影格的逐幀臉部重繪在 8GB 卡上已經是不小的負擔，先把最主要的臉部變形
> 問題解決，手部精修有需要再加。
>
> **一個排坑細節**：workflow 裡的 FaceDetailer（節點 `41`）刻意接的是另一組
> **沒有**掛 AnimateDiff motion module 的 FaceID 分支（節點 `11b`/`12b`），
> 不是主生成用的那個 AnimateDiff+FaceID 模型（節點 `11`/`12`）。原因：
> FaceDetailer 是逐張處理（一次只裁一幀的臉，batch=1），如果餵給還掛著
> motion module 的模型，等於硬逼 AnimateDiff 用一張圖的「批次」跑時序運算
> ——motion module 需要真正的多幀序列才能正常運作，餵單張圖會直接生成
> 雜訊/馬賽克碎片，不是變形而是完全損毀（ComfyUI 自己也會印出警告：
> `FaceDetailer is not a node designed for video detailing`，並建議改用
> Impact Pack 另外提供的 `Detailer For AnimateDiff` 節點——這裡選擇維持用
> 一般的 `FaceDetailer` 但改接無 motion module 的模型，做法更簡單、跟現有
> 靜態圖那套 FaceDetailer workflow 完全一致，不需要再學一個新節點）。

## 網頁 GUI（本地生圖介面）

### 前置條件

- 已裝好 GUI 需要的套件（見上面「2b. 一鍵安裝網頁 GUI + 所有選用功能套件」）
- ComfyUI server 已經在執行（見上面「啟動 ComfyUI server」章節）
- 開發者終端機或 PowerShell

### 啟動 GUI

#### 方式 1：終端機直接執行（推薦 — 可看實時日誌）

```powershell
cd D:\AI-Image-Lab\training
D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe gui.py
```

執行後終端機會輸出類似：
```
* Running on http://127.0.0.1:7860
```

#### 方式 2：背景執行（不佔用終端機）

```powershell
cd D:\AI-Image-Lab\training
Start-Process powershell -ArgumentList '-NoExit','-Command','D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe gui.py'
```

### 使用網頁

GUI 啟動後，在瀏覽器打開：**http://127.0.0.1:7860**

網頁會顯示生圖表單，填完後按「生成」，通常 30 秒到 1 分鐘內圖片會出現在右側

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

#### AnimateDiff 動態影片

- 頁面最下方獨立區塊，跟上面的靜態圖生成完全分開（不同 checkpoint：SD1.5，
  不是 SDXL），需要先完成「AnimateDiff 動態影片」安裝章節的模型下載
- 上傳一張臉部參考圖（FaceID 用，僅限虛構/AI生成，禁止上傳真人照片），輸入
  prompt，按「生成影片」——輸出是一小段 `.webm` 動態影片，會鎖住整段影片的
  臉部身分，並對每一幀跑一次臉部精修，修正純 SVD img2vid 常見的臉部變形問題
- 影格數預設 16（motion module 訓練時的上限，不建議調更高，需要另外接
  sliding-context-window 節點才能超過）、FPS 預設 8（約 2 秒的影片）
- 第一次執行會比較久：SD1.5 checkpoint、motion module、FaceID SD1.5 模型都是
  全新的模型組合，ComfyUI 還沒快取過
- 「Motion LoRA」摺疊區塊（選用）：控制整個畫面的鏡頭運動（縮放/平移/傾斜/
  旋轉），**不是身體部位的物理晃動效果**——選了要先完成「AnimateDiff Motion
  LoRA」安裝章節的模型下載，沒下載會直接生成失敗

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

### 動態影片、臉部不變形（AnimateDiff + FaceID + FaceDetailer）

```powershell
python generate_character.py video-animatediff --character mei --face-ref "reference_candidates\mei\anchor_0001.png" --prompt "sitting by a window, gentle breeze, turning head slightly"
```

見上面「AnimateDiff 動態影片」安裝章節。跟 `video` 不同，這是 txt2vid（不是
img2vid）：`--face-ref` 只用來鎖臉部身分（IP-Adapter FaceID），畫面內容完全
由 `--prompt` 決定。加 `--style-positive`/`--style-negative` 可以覆蓋預設的
寫實化用詞，用法跟 `custom` 指令一樣；`--tier`/`--negative-prompt` 的安全詞
保證（年齡保護、露骨內容封鎖）也完全一致，不受這兩個參數影響。

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

- 骨架庫 17 個姿勢：15 個從既有 `pose_pack_facedetailer` 產圖用 OpenposePreprocessor
  擷取，5 個躺姿手工重寫關鍵點（擷取出來的是坐姿失敗版）。`<slug>.json` 是關鍵點來源，
  `<slug>.png` 是 commit 進版控的骨架，產圖時不需要 torch。
- 骨架是**預先畫好的**，直接餵 ControlNet（`pose_is_skeleton=True` 跳過 preprocessor）；
  上傳照片走 `--pose-reference` 才會跑 preprocessor 抽骨架。
- 用法：CLI `--pose <名稱>`、GUI ControlNet accordion 的「姿勢骨架庫」下拉、
  `pose_pack.py --controlnet`（產全套對照）、`benchmark.py --hq --pose <名稱>`。

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
| ANIMATEDIFF_CHECKPOINT | `v1-5-pruned-emaonly.safetensors`（SD1.5，跟上面 SDXL 的 CHECKPOINT 分開） |
| ANIMATEDIFF_MOTION_MODULE | `mm_sd_v15_v2.ckpt` |
| ANIMATEDIFF_WIDTH × HEIGHT | 512 × 512（SD1.5 原生解析度） |
| ANIMATEDIFF_FRAMES / FPS | 16（motion module 訓練上限）/ 8 |
| ANIMATEDIFF_STEPS / CFG | 20 / 7.5（SD1.5 慣用範圍，比 SDXL 的 30 步/6.0 少/高） |

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
>   `weight_faceidv2` 參數（固定 1.0）
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
├── models/                   # SDXL/IP-Adapter/CLIP 權重，不進 repo（見模型章節）
├── training/
│   ├── generate_character.py # CLI 入口：角色定義 + prompt 組裝
│   ├── comfyui_client.py     # ComfyUI HTTP client（不 import torch）
│   ├── gui.py                # Gradio 本地網頁 GUI（localhost:7860）
│   ├── caption_image.py      # BLIP 圖像描述模型（用於自動產生 prompt）
│   ├── translate_prompt.py   # Prompt 自動翻譯成英文（Google Translate 公開端點）
│   ├── check_adetailer_models.py # ADetailer YOLO 模型自動檢查/下載腳本
│   ├── benchmark.py          # VRAM/RAM 記憶體洩漏測試
│   ├── workflow_template.json             # IP-Adapter workflow（variations 用）
│   ├── workflow_template_txt2img.json     # 純 txt2img workflow（anchor 用）
│   ├── workflow_template_img2img.json     # IP-Adapter + img2img workflow（gif 指令的 wiggle 幀用）
│   ├── workflow_template_controlnet.json  # IP-Adapter + ControlNet 骨架 workflow
│   ├── workflow_template_facedetailer.json # IP-Adapter + ADetailer 臉部/手部精修 workflow（YOLO）
│   ├── workflow_template_mediapipe_facedetailer.json # 同上，臉部偵測改用 MediaPipe 網格
│   ├── workflow_template_img2vid.json     # SVD img2vid workflow（video 指令用）
│   ├── workflow_template_animatediff_facedetailer.json # SD1.5 AnimateDiff + FaceID + 逐幀臉部精修（video-animatediff 用）
│   └── reference_candidates/ # anchor 候選圖，不進 repo
├── datasets/<character>/     # 生成的訓練圖 + caption，不進 repo
├── outputs/_comfyui_raw/     # ComfyUI 原始輸出暫存，不進 repo
├── captions/ evaluation/ samples/  # 保留給未來 kohya_ss LoRA 訓練階段用，目前空
├── comfyui-requirements.lock.txt   # ComfyUI 本體套件凍結版本清單
├── comfyui-requirements-extra.lock.txt   # 網頁 GUI + 所有選用功能（ControlNet/ADetailer/mediapipe）套件凍結版本清單
├── CLOUD_GPU.md               # 本地顯卡不夠力時的雲端 GPU 參考（Replicate/RunPod）
└── WEB_DEPLOYMENT.md          # 網頁部署規劃（AWS Amplify + Replicate/RunPod，尚無實作）
```

## 疑難排解

### ComfyUI 跟 kohya_ss LoRA 訓練不能同時執行

**症狀**：網頁 GUI 突然生不出圖片、連線錯誤，但 GUI process 本身還在跑
（`localhost:7860` 打得開），問題出在 ComfyUI（`localhost:8188`）連不上。

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
Get-CimInstance Win32_Process -Filter "name like '%python%'" | Where-Object { $_.CommandLine -like '*ComfyUI*main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# 訓練跑完/中斷後，重啟 ComfyUI（恢復生圖）
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

底模用 **Pony V6 base**（`ponyDiffusionV6XL_v6StartWithThisOne.safetensors`），
因為推論用的是 cyberrealistic_pony / pony_realism 等 Pony 系；Pony V6 重訓過
text encoder，用 SDXL base 訓的 LoRA 套到 Pony 上會弱或變形。要給 Juggernaut 用
就以 `--base sdxl` 再訓一次（不到 $1）。

步驟：

1. **補資料集（若太少）**：`xinyi`/`yuqing`/`wanling`/`ruoxi` 目前只有個位數張，
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

這台機器系統 RAM 只有 17GB，Chrome(多分頁)+ VS Code 背景就可能吃掉 8GB+，曾經
兩次在批次生成到一半時因為系統記憶體被榨乾而讓 ComfyUI process 直接崩潰（不是
單一個生成請求逾時，是整個 server process 消失，`curl /system_stats` 連不上）。
`gen_variations`/`gen_suggestive_variations` 都是續傳邏輯（依已存在檔案數判斷
從哪裡繼續），崩潰後不會遺失進度，重啟 ComfyUI 後重新執行同一條指令即可從中斷點
繼續，不會重跑已完成的部分。長時間批次生成前，建議先關閉不必要的背景程式騰出
系統 RAM。

## Roadmap（尚未完成）

- **kohya_ss LoRA 訓練階段**：本地 fp16 會 NaN，已改為在 RunPod 用 bf16 +
  `--no_half_vae` 訓練（見「在 RunPod 訓練角色 LoRA」章節，`runpod_bundle.py` +
  `runpod_train.sh`）。訓好的 LoRA 與 FaceID 並用，`CHARACTERS` 的 `lora` 欄位
  預設為 `None`，安裝後填入。`captions/` / `evaluation/` / `samples/` 仍是預留空資料夾
- 早期用 `diffusers` 直接載入模型時曾發生過一次無 traceback 的批次崩潰
  （生成到第 37 張左右 exit code 1）。改用本 repo 現在的 ComfyUI + API 架構後
  未再重現（100+ 張連續生成、benchmark 皆無異常），但當時沒有做根因診斷，
  記錄在此供之後參考
