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
- **執行環境只有一個**：`ComfyUI\.venv`（Python 3.11.15）。所有會碰到模型的 python 指令
  都用 `ComfyUI\.venv\Scripts\python.exe` 跑，torch / safetensors / insightface 只裝在
  那裡。`comfyui_client.py` 刻意只用 `requests`、不 import torch，這樣才能在沒有 GPU 環境
  的地方跑。
  其他 venv（別把 dev 工具裝進 ComfyUI\.venv）：
  - `.venv-dev`：只有 pytest / ruff / requests / fastapi（無 torch），給 `check.ps1` 和 CI 用。
    刻意分開是因為 `worker/Dockerfile` 從 `comfyui-requirements.lock.txt` 安裝，而那個 lock 是
    `uv pip freeze` 產生的——dev 工具裝進去遲早會被 freeze 進 worker image。
  - `SadTalker/.venv`、`MuseTalk/.venv`：第三方 clone 各自的 Python 3.10 環境（共約 13 GB），
    跟上面兩個互不相容。

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
  `AGE_SAFETY_NEGATIVE` 一定要留在負面詞裡。**cfg 1.0 時 ComfyUI 會跳過負面詞**，所以有
  `comfyui_client.SAFETY_MIN_CFG = 1.5` 這個下限（`ZIMAGE_MIN_CFG` / `LCM_MIN_CFG` 都是它的
  別名），並由 `enforce_min_cfg()` 在 `_submit_and_wait` 統一套用，不要為了省時間降到 1.0。
- **錯誤型別**：library 層用 `generate_character.UsageError` 表示呼叫端參數錯誤，**不要用
  `SystemExit`**（它是 `BaseException`，`except Exception` 接不到，這正是 image_api 和 GUI
  兩個邊界 bug 的成因）。只有 CLI 的 `__main__` 把它轉回 `SystemExit`。

## 驗證慣例

- **離線檢查就是一行**：`powershell -ExecutionPolicy Bypass -File check.ps1`
  （ruff → `ruff format --check tests` → pytest → 換行稽核，約 2 秒，不需要 GPU 也不需要
  ComfyUI）。改完先跑它再實機跑。這些以前要人眼複查的事現在都有測試守著：
  - GUI 每個 handler 的參數數量 vs 按鈕 `inputs` 長度（用 AST，不 import `gui.py`）
  - workflow JSON 的 node id ↔ class_type 契約（`training/workflow_contracts.py` 是單一真相，
    `_load_template` 也會檢查，模板被重新匯出會立刻報錯）
  - 年齡／內容安全負面詞在每條組裝路徑上都存活，**而且常數本身的內容也被釘住**
    （只檢查「有沒有傳下去」是自我指涉的：把常數清空，所有斷言都會變成恆真）
  - fp8 變體選擇優先序，以及 ControlNet control-lora 一律強制完整版的規則
  - CLI 子指令與旗標（`tests/test_cli_surface.py` 的 `FROZEN_CLI` 就是簽核點）
- CI（`.github/workflows/checks.yml`）在 ubuntu 與 windows 兩個 runner 上跑同一套，
  worker image 的 build 有 `needs: checks` 擋著。
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
- `web/amplify/` 有 **737 行實際的 TypeScript 後端**（DynamoDB + API Gateway + 3 個 Lambda +
  Replicate/RunPod provider + 兩個 webhook），不是骨架——但**從來沒有部署過、沒有整合測試過**
  （這個環境沒有 AWS 帳號），而且**完全沒有前端程式碼**。`web/README.md` 是最準確的說明，
  `WEB_DEPLOYMENT.md` 是更早的規劃文件。
