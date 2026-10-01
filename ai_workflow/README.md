# AI Workflow Platform（Google Drive ＋ Colab ＋ ComfyUI）

一套不綁任何 AI 供應商的生成工作流：

- **人工模式**：在 Google Drive 開 `AI-Workflow/notebooks/` 裡的 notebook → 用 Colab 開啟 → 全部執行。不需要任何 AI。
- **AI Agent 模式**：任何能執行指令的訂閱制 AI Agent，透過六個工具建立 job、查狀態、取結果。
- 兩種模式走**同一套 Core**（`controller/`）、同一份 workflow、同一個 job 佇列。系統不需要任何 AI API key。

```
AI Agent ──► scripts/aiwf.py（Tool Interface）─┐
                                              ├─► jobs/pending/*.json ──► Colab 上的 worker ──► ComfyUI ──► outputs/
人 ──► notebooks/*.ipynb（表單）───────────────┘        （Google Drive）
```

四個角色：訂閱制 AI＝控制器、Google Drive＝持久工作區、Google Colab＝運算、ComfyUI＝生成引擎。

> **驗證狀態（2026-10-02）**：離線測試全過（不需要 GPU、Colab 或 Google Drive，假的 ComfyUI 是本機 HTTP server）。
> **還沒有在真的 Colab 上跑過**——下面「實機驗收」每一項都還沒做，所有時間、VRAM、CU 都是未知數。

## 前置作業（一次）

1. 安裝 **Google Drive 電腦版**，同步方式選「**串流**」（不要「鏡像」：模型約 64 GB，鏡像會全部下載到本機）。
2. 告訴系統工作區在本機的路徑——建立 `ai_workflow/configs/local.json`（已 gitignore）：
   ```json
   { "workspace_root": "G:\\我的雲端硬碟\\AI-Workflow" }
   ```
3. 部署到 Drive（建目錄樹、複製 Core／notebooks／workflows；不會動 `models/ jobs/ outputs/`）：
   ```
   ComfyUI\.venv\Scripts\python.exe ai_workflow\scripts\deploy.py --dry-run
   ComfyUI\.venv\Scripts\python.exe ai_workflow\scripts\deploy.py
   ```
   改了程式或 registry 之後再跑一次即可。Drive 上的 `AI-Workflow/` 必須在「我的雲端硬碟」根目錄
   （notebook 預設路徑 `/content/drive/MyDrive/AI-Workflow`，可在第一格改）。
4. Colab 方案要有運算單元（CU）。Z-Image 需要 **L4** 以上，MiniMax H3 需要 **A100**。
5. Google Drive 空間：Z-Image 約 21 GB、H3 約 43 GB，加上 ComfyUI 與套件的壓縮檔。

## 人工模式

在 Google Drive 對 notebook 按右鍵 → 選擇開啟工具 → Google Colaboratory → **執行階段 > 全部執行**。

| Notebook | 做什麼 |
| --- | --- |
| `00_setup.ipynb` | 掛 Drive、環境檢查（Python／CUDA／GPU／VRAM／Drive／磁碟）、安裝或解壓 ComfyUI 與套件、列出模型狀態 |
| `01_comfyui.ipynb` | 啟動 ComfyUI、健康檢查、API 測試（不需要模型）、**等待 Agent 的 job** |
| `02_image_generation.ipynb` | Z-Image 文字生圖 → `outputs/images/<job_id>.png` |
| `03_video_generation.ipynb` | Prompt →（Z-Image 首幀）→ MiniMax H3 → `outputs/videos/<job_id>.mp4`，預覽圖在 `outputs/previews/` |
| `99_manual_debug.ipynb` | GPU／VRAM／模型／workflow 節點／API 測試／job 列表／ComfyUI log，每格一項 |

每本生成用的 notebook 都有 **Simple**（Prompt、比例、Duration）和 **Advanced**（打勾後生效：Seed、Steps、CFG、
Negative、LoRA 強度…，以及一個可填任何參數的 JSON 欄位）。Notebook 由 `scripts/build_notebooks.py` 從 registry
產生，裡面只有表單和對 `controller.notebook` 的呼叫——**不要手改 `.ipynb`**，改 registry 或 Core 後重新產生。

每台新的 Colab VM 掛 Drive 時都要按一次 Google 的同意畫面（Colab 的規定）。

## AI Agent 模式

Agent 讀 [`AGENTS.md`](AGENTS.md)（對所有 Agent 都一樣，沒有任何供應商專屬的內容），用這六個工具：

```
python scripts/aiwf.py list-workflows
python scripts/aiwf.py create-job z-image-basic --prompt "A white ceramic mug on a white background" --set aspect=16:9
python scripts/aiwf.py status JOB_ID
python scripts/aiwf.py list-jobs --status pending
python scripts/aiwf.py cancel JOB_ID
python scripts/aiwf.py result JOB_ID --wait 600
```

`aiwf.py` 只用標準函式庫，任何 Python 3.11+ 都能跑；輸出是 JSON。Agent **不會**自己開 Colab：`create-job` 只是把
job 寫進 `jobs/pending/`，要有人開著 `01_comfyui.ipynb`（或 02／03 最後一格打勾）才會被執行。沒有 worker 時，
工具的回應會帶 `hint` 告訴 Agent 該請使用者做什麼。佇列空了 `idle_minutes`（預設 10）分鐘後 worker 會停止並釋放
runtime。

換一個 Agent＝把同一份 `AGENTS.md` 給它，Core 一行都不用改。

## 工作區（Google Drive 上的 `AI-Workflow/`）

```
AI-Workflow/
├── notebooks/        入口（產生的，不要手改）
├── controller/       Core（本機與 Colab 跑同一份）
├── scripts/          aiwf.py（工具 CLI）、deploy.py、build_notebooks.py
├── workflows/        registry.json＋image/ video/ test/ 底下的 ComfyUI API 格式 JSON
├── configs/          platform.json（ComfyUI 釘選版本、worker 逾時）、environment.json（實際裝了什麼）
├── models/           z-image/ minimax-h3/ …＋manifest.json（來源、revision、大小、sha256）
├── loras/
├── inputs/           你自己放的輸入圖
├── outputs/          images/ videos/ previews/
├── jobs/             pending/ running/ completed/ failed/ cancel/
├── cache/            comfyui-<commit>.tar.gz、node-*.tar.gz、pydeps-<fingerprint>.tar
└── logs/             worker_state.json、jobs.jsonl、sessions.jsonl、jobs/<job_id>/workflow.json
```

Colab 的 `/content/aiwf` 只是這個 session 的暫存區，重要的東西都在 Drive。

### 第二個 session 不重裝、不重抓

| 項目 | 第一次 | 之後的 session | 什麼時候會重做 |
| --- | --- | --- | --- |
| ComfyUI／custom nodes | `installed`（下載釘選 commit 的 tarball，存進 `cache/`） | `extracted` | `configs/platform.json` 的 commit 改了 |
| pip 套件 | `installed`（裝一次並打包成 `pydeps-*.tar`） | `extracted` | Colab 映像檔、torch／CUDA、ComfyUI commit 或 requirements 變了 |
| 模型 | `downloaded`（HuggingFace 釘選 revision，驗 sha256，背景複製到 Drive） | `staged`（從 Drive 複製到 VM 本機碟） | 檔案缺少、大小或 sha256 不符——只處理那一個檔 |

每個 session 做了什麼記在 `logs/sessions.jsonl` 的 `setup_actions`。模型只在第一次執行用到它的 workflow 時才下載。

## Job

`jobs/<狀態>/<job_id>.json`，資料夾就是狀態：

```json
{
  "job_id": "job-20261002-101500-ab12cd",
  "task_type": "text_to_video",
  "workflow": "minimax-h3-basic",
  "status": "pending",
  "input": { "prompt": "A cinematic product introduction", "first_frame": { "source": "generate", "job_id": "…" } },
  "parameters": { "duration": 5, "aspect": "16:9", "steps": 8, "seed": 123456789 },
  "created_at": "…", "started_at": null, "completed_at": null, "output": null, "error": null
}
```

`pending → running → completed`，失敗是 `failed`，取消是 `cancelled`（放在 `jobs/failed/`）。

**單一寫入者規則**：Drive 同步遇到兩邊同時改同一個檔會留下「檔名 (1).json」。所以 job 檔建立之後只有 Colab 上的
worker 會搬動或改寫它；本機（工具或你自己）只能新增 `jobs/pending/<id>.json` 和 `jobs/cancel/<id>.request`。
不要手動搬 job 檔。

GPU 不夠大的 job（例如在 L4 上的 H3 job）會留在 `pending` 並在 `logs/worker_state.json` 的 `waiting` 寫明原因，
不會被標成失敗；換一台夠大的 runtime 開 notebook 就會被接走。

## 新增 workflow（不改 Core）

1. 把 ComfyUI 的 **API 格式** JSON 放到 `workflows/<image|video|test>/`。
2. 在 `workflows/registry.json` 加一筆：`type`、`workflow_file`、`safety`、`parameters`（每個參數綁到哪個
   node／input、型別、範圍、預設）、`output`（存檔節點）、`models`（檔名、HuggingFace repo、**釘選的** revision／
   size／sha256、放在 Drive 哪裡）、`gpu.min_vram_gib`。照抄 `z-image-basic` 改最快。
3. 直接加在 Drive 上的話 worker 不用重啟（它看到 `registry.json` 變了會重新載入）；加在 repo 的話跑
   `build_notebooks.py` 和 `deploy.py`。

`safety` 只有三種：`negative_cfg`（一般模型：有負面詞、cfg ≥ 1.5）、`static`（不含模型，只給測試用）、
`screened_prompt`（沒有負面詞的模型——**只有 `minimax-h3-basic` 可以用**，別的 workflow 宣告它會被拒絕；要再開放
一個這樣的模型必須改 `controller/safety.py`，這是刻意的）。

## 安全（沒有開關）

- 一般模型：安全負面詞由 Core 組，表單的 Negative Prompt 只會接在後面；cfg 低於 1.5 直接拒絕（低於這個值 ComfyUI
  會跳過負面詞）。建立 job 時檢查一次，worker 送出前再檢查一次——job 檔在共用資料夾裡，可能被手改。
- MiniMax H3 沒有負面詞也沒有 cfg，改用兩道控管：
  - Prompt、首幀 Prompt、Soundscape、Music 全部經過關鍵字過濾，命中就拒絕，沒有略過的參數。
  - 首幀只有三種來源：`generate`（Z-Image 依 prompt 生成，帶安全負面詞）、`job`（本工作區已完成的圖片 job，
    worker 會核對檔案的 sha256）、`file`（`inputs/` 裡的圖——程式無法驗證來源，所以必須勾選聲明：**本專案生成的
    虛構成人角色，或畫面中沒有人物；不是真人照片**）。
- 只有 safe 這一級；畫面中的人物一律是成人。

## 設定

`configs/platform.json`（會被 deploy 覆蓋，要改請改 repo 裡的）：

| 設定 | 預設 | 說明 |
| --- | --- | --- |
| `comfyui.commit` | `2d6b7328…` | 釘選的 ComfyUI commit（H3 skill 8 次成功執行用的那一版）。改了下一個 session 會重新安裝 |
| `custom_nodes.*.commit` | VideoHelperSuite `4d907bee…` | 每個 custom node 也釘 commit |
| `worker.idle_timeout_seconds` | 600 | 佇列空了多久停止（notebook 表單的 `idle_minutes` 會蓋過它） |
| `worker.max_session_seconds` | 10800 | worker 最長跑多久 |
| `worker.heartbeat_seconds` / `stale_worker_seconds` | 15 / 180 | 心跳間隔；多久沒心跳視為沒有 worker |
| `worker.max_attempts` | 2 | runtime 失聯時同一個 job 最多重領幾次 |
| `runtime.local_root` | `/content/aiwf` | VM 上的暫存區 |

工作區位置的優先序：`--root` → 環境變數 `AIWF_ROOT` → `aiwf.py` 所在的資料夾（從 Drive 資料夾執行時）→
`configs/local.json`。

## 實機驗收（尚未執行）

| 測試 | 做法 | 通過條件 |
| --- | --- | --- |
| Phase 3–6 | `00_setup` → `01_comfyui`（CPU 或 T4 即可） | 環境檢查全綠；ComfyUI 啟動；`test-generation` 的 PNG 出現在 `outputs/images/` |
| A 完全不用 AI | `03_video_generation` 全部執行 | MP4 在 `outputs/videos/`，通過 ffprobe 驗證 |
| B AI Agent | 開著 `01_comfyui`，Agent `create-job` → `result --wait` | Agent 回報輸出路徑；job 在 `jobs/completed/` |
| C 換 Agent | 另一個 Agent 讀同一份 `AGENTS.md` 做 B | `controller/` 沒有任何修改 |
| D 換 workflow | 把 B 的 workflow 換成 `z-image-basic` | 只改指令裡的 workflow 名稱 |
| Session Restart | 中斷 runtime，重開 notebook 再生成一次 | `setup_actions` 是 `extracted`／`staged`；舊的 jobs／outputs／notebooks 都還在 |

要量的數字：各階段秒數、GPU、VRAM 峰值、CU（關機後等 2 分鐘再讀餘額）、Drive 同步延遲，以及**從 Drive 複製 43 GB
到 VM 要多久**（H3 從 HuggingFace 直接下載實測約 5–6 分鐘；Drive 如果比較慢，就要重新考慮 H3 模型放 Drive 是否划算）。

## 測試

```
powershell -ExecutionPolicy Bypass -File check.ps1
```

`tests/test_aiwf_*.py`，不碰 GPU／Colab／Drive：安全常數釘在專案常數上（`AGE_SAFETY_NEGATIVE`、cfg 下限、H3 的禁詞表）；
兩個 workflow 綁定出來的節點圖等於專案既有的圖（`comfyui_client.build_zimage_txt2img_workflow`、H3 skill 的
`build_graph`）；工具只會新增檔案；worker 的失敗隔離、取消、首幀驗證；兩種模式產生相同的 job；第二個 session 零下載；
notebook 等於產生器的輸出而且不含邏輯；程式碼裡不出現任何 AI 供應商或 API key 的字樣。

## 第一階段沒做的事

GPU Router、RunPod／本機 GPU 的 Compute Provider（`controller/compute.py` 只有介面和 Colab notebook 一個實作）、
其他 Storage（`controller/storage.py` 同上，只有資料夾）、Agent 自動開 Colab runtime、成本最佳化、Web 介面、MCP server。
