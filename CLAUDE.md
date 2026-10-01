# CLAUDE.md

給 Claude Code 的專案速覽。安裝與使用細節在 [README.md](README.md)（繁體中文、CRLF），
每次改動的結論與實測數字在 [CHANGELOG.md](CHANGELOG.md)。

## 這個專案在做什麼

本地用 ComfyUI + SDXL / Pony / SD1.5 生成虛構角色的圖片和影片。四個入口都會經過
`training/comfyui_client.py`，由它把 workflow JSON 送給 ComfyUI 的 HTTP API：

- `training/generate_character.py`：CLI，角色定義與 prompt 組裝也在這裡
  （各模型關鍵字總表與 Prompt 範例：[training/PROMPT_GUIDE.md](training/PROMPT_GUIDE.md)）
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
  - `Kokoro/.venv`：短劇配音用的 Kokoro-82M（CPU 版 torch，約 1 GB），由 `training/kokoro_runner.py` 使用。
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
- **單一改寫點**：所有流程都經過 `comfyui_client._submit_and_wait`（workflow 圖層級的改寫，
  例如模型版本切換、LCM 模式切換重置）與 `generate_character._build_prompt_and_negative`
  （prompt 字串層級的組裝：score 標籤、安全負面詞、依模型的自然語言適配 `prompt_adapter.adapt`）。
  要對每條流程都生效的改寫放在這兩個點，不要散在各個 caller。
- **workflow**：`training/workflow_template*.json`，程式用 node id 改參數。功能關掉時要
  把節點從 dict 移除，不是留著把權重設成 0。
- **年齡安全**：`MINIMUM_AGE` 在 import 時檢查，不符合會直接 `raise`；
  `AGE_SAFETY_NEGATIVE` 一定要留在負面詞裡。**cfg 1.0 時 ComfyUI 會跳過負面詞**，所以有
  `comfyui_client.SAFETY_MIN_CFG = 1.5` 這個下限（`ZIMAGE_MIN_CFG` / `LCM_MIN_CFG` 都是它的
  別名），並由 `enforce_min_cfg()` 在 `_submit_and_wait` 統一套用，不要為了省時間降到 1.0。
  **唯一的例外是 MiniMax H3**（`.claude/skills/minimax-h3-colab/`，跑在 Colab）：它的 turbo 節點圖是
  BasicGuider，沒有負面詞也沒有 cfg，照 `cloud_video.replicate_safety_gate` 的規則本來應該拒用。使用者
  同意改用兩道補償控管：首幀只能是本專案生成的虛構成人角色或沒有人物的圖（不能用真人照片），以及
  `h3_colab.screen_prompt` 對完整 prompt 做正向詞過濾、沒有繞過的參數（測試釘住詞表，而且要涵蓋
  `AGE_SAFETY_NEGATIVE` 的每個詞）。其他模型不要比照這個例外。
- **錯誤型別**：library 層用 `generate_character.UsageError` 表示呼叫端參數錯誤，**不要用
  `SystemExit`**（它是 `BaseException`，`except Exception` 接不到，這正是 image_api 和 GUI
  兩個邊界 bug 的成因）。只有 CLI 的 `__main__` 把它轉回 `SystemExit`。

## 驗證慣例

- **離線檢查就是一行**：`powershell -ExecutionPolicy Bypass -File check.ps1`
  （ruff → `ruff format --check tests` → pytest → 換行稽核，約 45 秒，不需要 GPU 也不需要
  ComfyUI）。改完先跑它再實機跑。這些以前要人眼複查的事現在都有測試守著：
  - GUI 每個 handler 的參數數量 vs 按鈕 `inputs` 長度（用 AST，不 import `gui.py`）
  - workflow JSON 的 node id ↔ class_type 契約（`training/workflow_contracts.py` 是單一真相，
    `_load_template` 也會檢查，模板被重新匯出會立刻報錯）
  - 年齡／內容安全負面詞在每條組裝路徑上都存活，**而且常數本身的內容也被釘住**
    （只檢查「有沒有傳下去」是自我指涉的：把常數清空，所有斷言都會變成恆真）
  - fp8 變體選擇優先序，以及 ControlNet control-lora 一律強制完整版的規則
  - CLI 子指令與旗標（`tests/test_cli_surface.py` 的 `FROZEN_CLI` 就是簽核點）
  - `training/PROMPT_GUIDE.md` 引用的常數、模型 key／檔名、詞庫、角色、場景、骨架與 8 組
    Prompt 範例跟程式一致（`tests/test_prompt_guide.py` 會重新呼叫組裝函式比對，失敗訊息
    直接印出可貼回的字串）
  - `training/booru_lexicon.json` 涵蓋專案自己的所有詞彙（角色外觀／詞庫／場景／骨架），
    而且推導出的標籤不含任何年齡／露骨詞（`tests/test_prompt_adapter.py`）
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
- 雲端影片（2026-09-17）：`worker/`（RunPod Serverless，`jobs.py` 支援 `image_hq`／`video_wan_i2v`／
  `video_animatediff`）＋ `training/cloud_video.py`（本機客戶端，RunPod 與 Replicate）＋ GUI「☁️ 雲端影片」
  分頁。Replicate 模型沒有負面詞＋cfg 欄位就拒用。**程式碼與離線測試完成，但從未在真的雲端 GPU 上跑過**
  （沒有帳號），worker image 也還沒 build 過。改 workflow 模板後可用
  `training/validate_workflow_nodes.py` 對本機 ComfyUI 做結構檢查。
- AI 短劇（2026-09-27）：`training/drama.py`（分鏡表 → 關鍵幀 → 配音 → 雲端動態 → 9:16 成片）＋
  `drama_compose.py`（ffmpeg 指令）＋`kokoro_runner.py`／`cosyvoice_runner.py`（各在自己的 venv 跑）。分鏡表在
  `training/episodes/`，工作檔在 `outputs/drama/`。JV 第一集的 Z-Image 草稿關鍵幀和 Kokoro 配音實跑過，
  組成 39 秒有聲粗剪；雲端動態還沒實跑，也還沒有對嘴。`assemble` 先把每句配音對齊到 −23 LUFS，
  成片再整體調到 −14 LUFS。
  「一次送多支」只有 `cloud_video.run_cloud_batch` 一個實作。
  語言學習短劇（JV Tutor Corner）用 `translation`／`speed` 欄位和 `card` 字卡鏡頭，平台每門語言課一集：
  `jv_en_ep01`～`ep03`、`jv_ja_ep01`（日文集的聲音還沒定）；英文台詞配中文參考錄音會走跨語言模式（帶口音）。
  低解析度優先：`keyframes --draft`、`motion --draft`、`export`（enhance/in → 外部工具 → enhance/out）、
  `assemble --fps 48`。
  `"commercial": true` 的集數只准用可營利元件（Z-Image 關鍵幀、Wan、RIFE、Kokoro 內建聲音或有 consent
  紀錄的 CosyVoice 聲音），
  示範聲音要 `--allow-demo` 才能用。
- GUI 有 8 個分頁，其中「🎯 圖片選擇生圖」是點縮圖（人物／姿勢／場景）取代打 prompt：
  `gc.plan_picker()` 依 checkpoint 決定姿勢和臉是走 ControlNet/FaceID（SDXL、Pony）還是
  降級成文字（Z-Image、SD1.5），降級時 GUI 會明講。場景庫在 `training/scenes/`（縮圖已
  commit，`scene_library.py render-thumbs` 重生）。
- 資源排程（2026-09-27）：GUI 本地分頁與 image_api `/generate` 送出前都經過 `resource_policy.decide()`
  （成本表＋nvidia-smi／psutil 即時狀態＋時段），結果是本地／等降溫／排時段／**建議**上雲——雲端一律要人確認，
  image_api 回 409。延後的工作是 job store 的 `queued`＋`not_before`，由 `job_scheduler` 執行（GUI 在
  `__main__`、image_api 在 lifespan 啟動）。CLI 刻意不經過這層。設定在 `training/settings/hardware.json`。
  測試裡 image_api 的 probe 固定成全 None 的 Snapshot，別讓真機器狀態漏進測試。
  開始門檻：批次／排程／API 55°C、GUI 按一次只生一張 70°C；實測 `hw_thermal_slowdown` 約 80°C 就 Active。
- `web/amplify/` 有 **737 行實際的 TypeScript 後端**（DynamoDB + API Gateway + 3 個 Lambda +
  Replicate/RunPod provider + 兩個 webhook），不是骨架——但**從來沒有部署過、沒有整合測試過**
  （這個環境沒有 AWS 帳號），而且**完全沒有前端程式碼**。`web/README.md` 是最準確的說明，
  `WEB_DEPLOYMENT.md` 是更早的規劃文件。
- MiniMax H3（2026-09-30）：Claude Code skill `.claude/skills/minimax-h3-colab/`，一張圖 → Colab A100 上的
  H3 首幀模式（影片＋音訊）→ `output/*.mp4`，ffprobe 驗證，成本記在 `output/h3_jobs.jsonl`。google-colab-cli
  不支援 Windows，所以跑在 Docker 映像檔 `h3-colab-cli:0.7.4` 裡，OAuth token 放在 volume `h3-colab-config`。
  多支影片用 `--manifest`／`--segments` 共用一個 session（約 40 GB 模型只下載一次，實測三支 5 秒 1.51 CU，分開跑約
  2.55 CU）。Colab 帳號是幾個 Claude 工作階段共用的：開跑前會檢查帳號上有沒有別的 runtime（`COLAB_BUSY`），成本只看
  整批 settle 後的餘額差。實測數字見 CHANGELOG。
- Z-Image Colab worker（2026-10-02）：Claude Code skill `.claude/skills/z-image-colab/`，文字 → Z-Image Turbo（官方
  bf16 權重）→ `output/zimage/<job_id>/result.png`。job 是本機檔案佇列（`submit.py`），`worker.py up` 開**一個**
  Colab L4 session 跑完整個佇列，閒置 `worker.idle_timeout_seconds` 後自動關。ComfyUI 原始碼、pip 套件、模型存在
  Google Drive，之後的 session 只解壓和複製、不重裝不重抓（`environment.json`＋模型 manifest 比對版本）；但 Colab
  每台新 VM 都要使用者按一次 Drive 同意，沒按就整個重新下載。與 H3 skill 共用 Docker 映像檔與登入。圖用的是
  `training/workflow_template_txt2img_zimage.json`（`comfyui_client.build_zimage_txt2img_workflow`），負面詞走
  `_build_prompt_and_negative`，所以年齡安全詞與 cfg 下限跟本機一樣，VM 上的 worker 會再檢查一次。
  **離線測試 210 個通過，但從未在真的 Colab GPU 上跑過**——首次安裝、重啟不重裝、連續 3 張、閒置關機四項實機
  驗收都還沒做，時間與 CU 都沒有數字。
