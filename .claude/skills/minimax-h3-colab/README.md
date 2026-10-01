# MiniMax H3 × Google Colab：一張圖 → 一支 MP4

在 Claude Code 說「用這張圖生成一支 8 秒的 H3 影片，人物往鏡頭方向走」，Claude 會依 `SKILL.md`
寫好 prompt，在 Colab 的 A100 上跑 MiniMax H3（首幀模式，影片＋音訊），把 MP4 下載到 `output/`，
並用 ffprobe 驗證。本機 RTX 2070 不參與推論。

第一階段只做「1 張圖 + 1 段 prompt + 1 個 H3 job = 1 支 MP4」。

## 架構

```
Claude（SKILL.md、prompts/video_prompt.md）
  → scripts/run.py（本機，只用 Python 標準函式庫）
      → google-colab-cli 0.7.4（Windows 上跑在 Docker 容器裡）
          → Colab A100：colab exec -f remote/h3_colab_job.py
              → ComfyUI + MiniMax H3（pruned fp8 首幀模型 + turbo v4 LoRA，8 步）→ MP4
      ← colab download → ffprobe 驗證 → output/*.mp4 + 記錄
```

- **為什麼用 Docker**：google-colab-cli 官方只支援 Linux／macOS，它在啟動時就 `import termios`，
  Windows 上連 `colab version` 都跑不起來。每次呼叫都是一個用完即丟的容器，OAuth token 和 session
  資料放在 Docker volume `h3-colab-config`，不會出現在 repo 或設定檔裡。
- **為什麼遠端是 `.py` 不是 notebook**：`colab exec -f` 本來就能跑 `.py`；這樣 ruff／pytest 可以檢查
  節點圖與幀數計算。參考專案 killkli/minimax-h3-colab-skill 沒有授權條款，所以這裡**沒有複製它的
  程式碼**，只沿用可以查證的事實（CLI 指令、HF 檔名、ComfyUI 節點圖）重寫。
- **為什麼看 marker 不看 exit code**：`colab exec` 在遠端程式丟例外時也可能回 0。遠端腳本每個階段印
  `H3_<種類> {json}`，本機邊串流邊解析；沒有 `H3_OUTPUT` 就算失敗。

## 前置作業（一次）

1. **Docker Desktop** 要開著。
2. **建映像檔**（只裝 CLI，不含模型）：
   ```
   docker build -t h3-colab-cli:0.7.4 .claude\skills\minimax-h3-colab\docker
   ```
3. **Google 登入**（要你自己在終端機做，Claude 不能代填）：
   ```
   docker run --rm -it -v h3-colab-config:/root/.config/colab-cli h3-colab-cli:0.7.4 --auth=oauth2 usage
   ```
   它會印一個網址 → 用瀏覽器登入 Google → 把頁面顯示的授權碼貼回終端機。成功後會印出 CU 餘額。
   token 存在 volume 裡，之後不用再登入；要登出就 `docker volume rm h3-colab-config`。
4. **ffprobe**：放在 PATH，或寫進 `config/config.json`（已 gitignore）：
   ```json
   { "tools": { "ffprobe": "MuseTalk/ffmpeg_dl/ffmpeg-n7.1-latest-win64-gpl-shared-7.1/bin/ffprobe.exe" } }
   ```
   相對路徑以 repo 根目錄為準。沒有的話：`winget install --id Gyan.FFmpeg`。
5. **Colab 方案**：需要有 CU 而且能分配到 A100 的方案（Google AI Pro／Colab Pro 以上）。免費的 T4
   不夠（模型約 40 GB，參考 notebook 也這麼說）。

檢查全部就緒（不花 CU）：
```
ComfyUI\.venv\Scripts\python.exe .claude\skills\minimax-h3-colab\scripts\check.py
```

## 使用

透過 Claude：直接描述需求即可，Claude 會照 `SKILL.md` 走完整個流程。

直接用 CLI（在 repo 根目錄）：
```
ComfyUI\.venv\Scripts\python.exe .claude\skills\minimax-h3-colab\scripts\run.py ^
  --image input\test.png --duration 8 ^
  --prompt "The woman in the first frame ... walks slowly toward the camera ..." ^
  --soundscape "Light street ambience with her soft footsteps." ^
  --output output\test.mp4
```

| 參數 | 說明 |
| --- | --- |
| `--image` | 首幀圖（png／jpg／webp，≤ 50 MB）。**只能用本專案生成的虛構成人角色，或沒有人物的圖** |
| `--prompt` | 英文的鏡頭描述，原樣放進 H3 官方格式（寫法見 `prompts/video_prompt.md`） |
| `--prompt-file` | 完整的 H3 I2VA prompt，原樣送出（要有 `<Picture 1>`） |
| `--soundscape` / `--music` | H3 會同時生成音訊；music 預設 `N/A` |
| `--constraint` | `locked-camera`（鏡頭固定）、`static-subject`（人物不動），可重複 |
| `--duration` | 4–15 秒，預設 8。會對齊到 17k+5 幀：5→124 幀／5.17 秒，8→192 幀／8.00 秒 |
| `--resolution` | 預設 `auto`：依圖片比例、短邊 768（16:9→1376×768、9:16→768×1376）。指定 WxH 時要是 32 的倍數、≤ 1.2 MP、與圖片比例差 ≤ 2% |
| `--output` | 預設 `output/<圖名>_h3_<job_id>.mp4`；已存在時要加 `--overwrite` |
| `--gpu` / `--no-high-mem` / `--timeout` | 預設 A100、高 RAM、3600 秒 |
| `--job spec.json` | 用 JSON 描述 job（`job_id`、`scene_id`、`input`、`prompt`、`settings`、`output`），給之後的分鏡流程用 |
| `--dry-run` | 驗證輸入、組好 prompt、印出實際會跑的 docker／colab 指令，跑 preflight，**不花 CU** |

exit code：0 成功、2 輸入錯誤、3 逾時、130 取消、1 其他錯誤。stdout 最後一行是整份記錄（JSON）。

除錯用：`scripts/upload.py`、`scripts/download.py` 可以對指定 session 上傳／下載單一檔案（例如本機中斷後
把遠端已經生成的 MP4 救回來）。

## Job 狀態與錯誤

```
PENDING → PREPARING → CONNECTING_COLAB → UPLOADING → LOADING_MODEL → INFERENCE → DOWNLOADING → VALIDATING → COMPLETED
失敗：FAILED / TIMEOUT / CANCELLED / FAILED_VALIDATION（記錄裡有 error_code 和 failed_stage）
```

- **PREPARING 之前不花 CU**：圖片能不能解碼、prompt、長度、解析度、ffprobe、Docker、登入都先檢查。
- LOADING_MODEL 包含：在 Colab 裝 ComfyUI、從 Hugging Face 下載約 40 GB 模型、啟動 ComfyUI。
  模型真正載入 GPU 是在 INFERENCE 的前段。
- 每一步都有 timeout；**逾時的 `colab exec` 不會重試**（遠端可能還在跑，重試等於付兩次錢），一律
  `colab stop`。Ctrl+C 也會 stop。stop 失敗時記錄會標 `stop_failed` 並附手動 stop 指令。

| error_code | 意思 |
| --- | --- |
| `INVALID_INPUT` / `PROMPT_REJECTED` / `INVALID_CONFIG` | 輸入或設定有問題，沒有花 CU |
| `FFPROBE_MISSING` / `DOCKER_UNAVAILABLE` / `COLAB_CLI_MISSING` | 本機環境缺東西 |
| `AUTH_REQUIRED` | `Google Colab authentication is required. Please authenticate and run the job again.` |
| `GPU_UNAVAILABLE` | `No compatible GPU is currently available.`（沒有配額或暫時沒有 A100） |
| `COLAB_CONNECTION_FAILED` / `UPLOAD_FAILED` | 連不上 Colab 或上傳失敗 |
| `INSUFFICIENT_DISK` / `COMFYUI_SETUP_FAILED` / `MODEL_DOWNLOAD_FAILED` / `INFERENCE_FAILED` | 遠端各階段失敗 |
| `TIMEOUT` | 本機或遠端逾時 |
| `OUTPUT_NOT_FOUND` | 推論結束但沒有可下載的 MP4 |
| `FAILED_VALIDATION` | MP4 下載了但沒過 ffprobe 檢查（檔案保留） |

**ffprobe 檢查項目**：檔案存在且大小 > 0、有 video stream、h264、yuv420p、寬高符合、24 fps、長度 = 幀數／24
（誤差 ±max(0.1 秒, 2 幀)）、有音訊。

## 輸出與成本記錄

- `output/<名稱>.mp4`：影片
- `output/<名稱>.job.json`：完整記錄（prompt、設定、每個階段秒數、GPU、VRAM 峰值、ffprobe 摘要、CU）
- `output/h3_jobs.jsonl`：成本總帳，每個 job 一行
- `output/logs/<job_id>.log`：完整 log，每行都有 `[狀態]`
- 失敗時 `output/.work/<job_id>/` 會保留上傳的檔案給你除錯；成功就刪掉

CU 用兩種方式記：`cu_used_measured` 是 job 前後 `colab usage` 餘額的差（實際值，但 Colab 的餘額可能
延遲更新）；`cu_estimated` 是 session 開啟期間的每小時用量 × session 秒數（估算）。**不假設固定的
每支 CU**，實測表見下面。

## 設定

`config/config.example.json` 是預設值；要改就建立 `config/config.json`（已 gitignore，只寫要改的鍵）。
環境變數優先於設定檔：`H3_TRANSPORT`、`H3_DOCKER_IMAGE`、`H3_GPU`、`H3_HIGH_MEM`、`H3_TIMEOUT_SECONDS`、
`H3_OUTPUT_DIR`、`H3_FFPROBE`。命令列參數再優先於環境變數。

設定檔**不能放任何機密**：鍵名含 key／token／secret／password／credential 會直接被拒絕。唯一的憑證是
colab CLI 自己的 OAuth token，放在 Docker volume 裡。

## 安全

H3 的 turbo 節點圖沒有負面詞，也沒有 cfg（BasicGuider）。本專案其他流程靠負面詞裡的年齡安全詞把關，
H3 用不上，所以改用兩道補償控管（CLAUDE.md「年齡安全」有寫這個例外）：

1. **首幀只能是本專案生成的虛構成人角色，或沒有人物的圖**。不能用真人照片（SKILL.md 規定 Claude 遵守）。
2. **prompt 篩選**（`screen_prompt`）：組好的完整 prompt 裡只要出現年齡詞（`AGE_SAFETY_NEGATIVE`
   全部都在內）、18 歲以下的年齡寫法或露骨詞（中英文）就拒絕，**沒有繞過的參數**。已知 "minor" 會誤擋
   "minor movement"，改寫成 "small" 即可。

## 測試

| 層級 | 怎麼跑 | 需要 |
| --- | --- | --- |
| 單元（156 條） | `powershell -ExecutionPolicy Bypass -File check.ps1` | 什麼都不需要（CI 也跑） |
| 整合（CPU runtime 來回） | `$env:H3_LIVE_COLAB="1"; .venv-dev\Scripts\python.exe -m pytest -m live tests/test_minimax_h3_live.py` | Docker、已登入；佔用 CPU runtime 一兩分鐘 |
| 真實推論 | 上面的 `run.py` 指令 | A100 與 CU |

單元測試涵蓋：設定優先序與機密鍵、prompt 格式與原樣保留、constraint、安全詞（含釘住詞表）、幀數與解析度、
ffprobe 每一條規則、狀態機的每一條失敗路徑（壞圖／缺圖／缺 prompt／長度不合法時 transport 完全沒被呼叫、
未登入、GPU 被拒、模型下載失敗、exec 逾時不重試、Ctrl+C、沒有輸出、驗證失敗、stop 失敗）、真實子程序逾時、
Docker 指令組裝，以及遠端節點圖符合 ComfyUI H3 節點契約。

## 授權（使用前請看）

- **MiniMax H3**：MiniMax H3 Community License。
  - 允許商用。年營收超過 2,000 萬美元要另外取得 MiniMax 書面授權。商用產品介面上要明顯標示「MiniMax H3」。
  - **不能用 H3 的輸出去改進其他 AI 模型**，例如拿 H3 影片的幀去訓練 SDXL 角色 LoRA。
  - 公開發布時要清楚標示內容是機器生成的。
  - 授權範圍**不含美國、歐盟、英國、韓國**。Colab VM 可能位在這些地區，平台用戶也可能在這些地區，
    商用前請找法務確認。這一點我無法替你判斷。
- **drbaph/MiniMax-H3-Turbo-Lora-ComfyUI**（LoRA）：Apache-2.0。
- **google-colab-cli**：Apache-2.0。
- 參考：[kocpc 教學](https://www.kocpc.com.tw/archives/670501)、
  [killkli/minimax-h3-colab-skill](https://github.com/killkli/minimax-h3-colab-skill)（流程與模型組合的來源，
  未複製程式碼）、[MiniMax H3 模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)、
  [ComfyUI H3 節點](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)。

## 實測（2026-09-30，Google AI Pro）

兩支都是 Colab 分配的 NVIDIA A100-SXM4-80GB、高 RAM；session 每小時 6.77 CU。首幀是 9:16 的圖，解析度自動選 768×1376。

| | 5 秒（124 幀） | 8 秒（192 幀） |
| --- | ---: | ---: |
| 裝 ComfyUI＋下載 40 GB 模型＋啟動 | 422 秒 | 413 秒 |
| 推論 | 195 秒 | 365 秒 |
| VRAM 峰值 | 43,214 MiB | 45,598 MiB |
| 整個 job | 663 秒 | 818 秒 |
| **CU（實測，餘額差）** | **0.85** | **1.49** |

- 每支大約一半時間花在新 session 的安裝和下載（約 0.8 CU）。之後改成同一個 session 連跑多支，這段開銷只需要付一次。
- A100 偶爾會暫時沒有容量（assign 回 503 → `GPU_UNAVAILABLE`）。這種失敗不花 CU，過 1 分鐘重試就成功了。
- 5 秒那支從第 2 秒起鏡頭連續大幅繞到人物側面（不是硬切）。要保持原構圖，就寫明鏡頭不動並加上 "in one continuous shot"。第 2 支這樣寫之後就維持住了。
- **CU 餘額會延遲扣款**：實測有一次在 job 結束約 1.5 分鐘後，又多扣了 0.57 CU。`cu_used_measured` 是 job 結束當下的讀數，報成本前要隔幾分鐘再讀一次，或看整批的餘額差（目前 6 支平均 1.34 CU，約 5.9 CU/小時）。
- **同一個 Colab 帳號別的工作階段也會用**：另一個 Claude 工作階段可能同時開 job，餘額差會混在一起，而且 A100 也會被佔走。開始前先看 `colab sessions` 有沒有別人的 session；成本以整批的餘額差為準。
- 詳細數字與分析：repo 根目錄 `CHANGELOG.md`。

## 限制與下一階段

- 每個 job 都開新的 session，所以每支都要重新裝 ComfyUI、重新下載 40 GB 模型。下一步是讓同一個 session
  連跑多支（分鏡流程的前提），第二支之後可以省掉這段時間。
- 目前只有首幀模式。Ref2VA（1–9 張參照圖）和首尾幀模式，notebook 那邊已經有對應的節點，下一階段再開。
- 還沒做的：多鏡頭／分鏡 → FFmpeg 串接、R2 上傳、音訊替換、Lip Sync、放大、補幀。
  `JobSpec` 的 `job_id`／`scene_id`／`input`／`prompt`／`settings`／`output` 結構就是為這些預留的。
