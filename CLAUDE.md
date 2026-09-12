# CLAUDE.md

給 Claude Code 的專案速覽。安裝與使用細節在 [README.md](README.md)（繁體中文、CRLF），
每次改動的結論與實測數字在 [CHANGELOG.md](CHANGELOG.md)。

## 這個專案在做什麼

本地用 ComfyUI + SDXL / Pony / SD1.5 生成虛構角色的圖片和影片。四個入口都會經過
`training/comfyui_client.py`，由它把 workflow JSON 送給 ComfyUI 的 HTTP API：

- `training/generate_character.py`：CLI，角色定義與 prompt 組裝也在這裡
- `training/gui.py`：Gradio 本地網頁（127.0.0.1:**7861**，需要時會自動啟動 ComfyUI）
- `training/image_api.py`：FastAPI 包裝，給外部專案用 HTTP 呼叫
- `worker/`：RunPod serverless worker（雲端 GPU）

## 硬體與環境（左右了大部分技術決定）

- **RTX 2070 8 GB、Turing sm_75**：沒有 bf16、沒有 fp8 運算（fp8 只能拿來存，算的時候
  會轉回 fp16）
- **PCIe gen3 x1**：模型載入／搬移才是瓶頸，不是取樣本身
- **32 GB 系統 RAM**：高清流程單張就可能吃到 15 GB+，會開始用分頁檔
- **過熱降頻**：GPU 到 84°C 會硬體降頻，同一個 session 裡後面的計時不能直接跟前面比；
  做效能比較一定要一起記溫度、註明是不是冷機
- **只有一個 venv**：`ComfyUI\.venv`。所有 python 指令都用
  `ComfyUI\.venv\Scripts\python.exe` 跑，torch / safetensors / insightface 只裝在那裡。
  `comfyui_client.py` 刻意只用 `requests`、不 import torch，這樣才能在沒有 GPU 環境的
  地方跑。

## 常用指令

```powershell
# GUI（會自動啟動 ComfyUI）
ComfyUI\.venv\Scripts\python.exe training\gui.py

# 單張生圖
ComfyUI\.venv\Scripts\python.exe training\generate_character.py custom --checkpoint cyberrealistic_pony --prompt "portrait photo" --seed 9000

# 關掉 ComfyUI 釋放 VRAM/RAM（訓練前一定要做）
powershell -ExecutionPolicy Bypass -File training\stop_comfyui.ps1

# 量化版模型狀態／切換
ComfyUI\.venv\Scripts\python.exe training\quantize_models.py status
```

## 程式碼慣例

- **換行**：`.py` 和 `CHANGELOG.md` / `CLAUDE.md` / `CLOUD_GPU.md` 是 LF，**`README.md`
  是 CRLF**。用腳本改檔案時 `open(..., newline="")` 讀寫，改完用 `git ls-files --eol`
  確認沒有把整個檔案換行改掉。
- **語言**：程式碼註解、commit 訊息用英文；使用者看得到的字串（GUI 標籤、CLI 提示、
  README）用繁體中文。`training/quantize_models.py` 全檔 ASCII。
- **單一改寫點**：所有流程都經過 `comfyui_client._submit_and_wait`。要對每條流程都生效
  的改寫（模型版本切換、LCM 模式切換重置）放在那裡，不要散在各個 caller。
- **workflow**：`training/workflow_template*.json`，程式用 node id 改參數。功能關掉時要
  把節點從 dict 移除，不是留著把權重設成 0。
- **年齡安全**：`MINIMUM_AGE` 在 import 時檢查，不符合會直接 `raise`；
  `AGE_SAFETY_NEGATIVE` 一定要留在負面詞裡。**cfg 1.0 時 ComfyUI 會跳過負面詞**，所以
  每個取樣器的 cfg 都設了下限（Z-Image 1.5、影片 LCM 1.5），不要為了省時間把它降到 1.0。

## 驗證慣例

- 先做離線檢查（改寫邏輯、拒絕條件、參數數量），再實機跑。GUI 改動要做 build 檢查：
  `generate()` 的參數數量必須等於按鈕的 inputs 數量。
- 臉／身分比較用 `training/face_similarity.py`（InsightFace `buffalo_l`，跟 FaceID 同一個
  模型）。它只判「是不是同一個人」，**對臉部形變不敏感**——形變問題要看逐幀拼圖，分數
  只能用來確認沒有換臉。
- 效能數字要附 GPU 溫度、VRAM 峰值和 ComfyUI 的 RSS；`nvidia-smi` + `psutil` 取樣。

## 不要做的事

- **不要改動 `models/` 和 `ComfyUI/models/` 裡的原始權重檔**：部分是 NTFS hardlink（改一
  邊會動到另一邊）。要產生衍生檔就寫成新檔名，並確認 `st_nlink == 1`。
- **不要自己下載模型**：動輒好幾 GB，先問使用者。
- **ComfyUI 跟 kohya_ss 訓練不能同時跑**（同一張 8 GB 卡），訓練前先跑 `stop_comfyui.ps1`。
- **有 ControlNet control-lora 的流程不能用量化版 checkpoint**（會出全黑圖），這個判斷已經
  在 client 裡，不要繞過。

## Git

- 改動先開分支 commit，再 fast-forward 回 `main`；**不要 push，除非使用者明確要求**。
- commit 訊息寫「為什麼這樣改」和實測數字，不只寫改了什麼。

## 現況（2026-09-12）

- 還沒有訓練出可用的角色 LoRA：11 個角色的 `lora` 欄位都是 `None`，身分完全靠 IP-Adapter
  FaceID。本地 fp16 訓練會 NaN，RunPod bf16 路線還沒實際跑過（見 README「在 RunPod 訓練
  角色 LoRA」）。
- 4 個 SDXL / Pony checkpoint 都有 fp8 量化版可選，預設仍是完整版。
- Z-Image Turbo 只支援純文字生圖，沒有 FaceID / ControlNet / 精修版本。
- `web/`（Amplify 骨架）和 `WEB_DEPLOYMENT.md` 目前只是規劃，沒有對應實作。
