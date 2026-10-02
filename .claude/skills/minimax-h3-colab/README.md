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
          → colab drivemount（掛 Google Drive；每台新 VM 要你按一次同意）
          → Colab A100：colab exec -f remote/h3_colab_job.py
              → 模型：Drive 上有就複製到 VM，沒有才下載（下載後背景存回 Drive）
              → ComfyUI + MiniMax H3（pruned fp8 首幀模型 + turbo v4 LoRA，8 步）→ MP4
              → MP4 複製到 Drive 的 AI-Workflow/outputs/videos/
          → colab exec（H3_FINALIZE）：等模型存完 → flush Drive
          → colab stop（一定會執行）
      ← colab download → ffprobe 驗證 → output/*.mp4 + 記錄
```

## 存放位置（2026-10-02 起）

規則：**下載的模型和生成的影片都存在 Google Drive，Colab 上只有運算過程中的暫存。**
跟 `ai_workflow/` 平台共用同一個 Drive 資料夾與 manifest 格式，所以 43 GB 的模型只有一份：

| Google Drive `AI-Workflow/` | Colab VM（session 結束就消失） |
| --- | --- |
| `models/minimax-h3/…`、`loras/`、`models/manifest.json` | ComfyUI、從 Drive 複製來的模型副本 |
| `outputs/videos/<檔名>.mp4` | ComfyUI 的輸出暫存、最後一格 |

- 每個模型檔三種結果，記在 job 記錄的 `model_actions`：`cached`（這台 VM 已經有）、`staged`（從 Drive 複製）、
  `downloaded`（從 HuggingFace 的**釘選 revision** 下載並驗 sha256，之後背景複製到 Drive）。manifest 只在複製完成後才寫，
  所以傳到一半的檔案下次不會被當成完整檔。複製失敗的檔，同一個 session 的下一支影片會再存一次，不會重抓。
- session 結束前會多一步「存檔」：等背景複製做完、flush Drive，結果記在 `drive_saved`。這一步失敗或逾時
  （`drive.finalize_timeout_seconds`，預設 3600 秒）**不會擋住關機**——只會在記錄的 `warnings` 寫明哪些可能沒存到。
  kernel 卡住（exec 逾時、Ctrl+C）時這一步直接跳過，同樣先關機。
- **每台新 VM 都要你在瀏覽器按一次 Drive 同意**（Colab 的限制，10-01 實測）。log 出現 `DRIVE_CONSENT_NEEDED` 時
  會自動開瀏覽器；按完後執行 `scripts\consent_done.py`（沒執行的話 `drive.consent_wait_seconds`，預設 90 秒後也會
  自動繼續）。**沒按同意**：這個 session 照樣出片，但模型重新下載、什麼都不存到 Drive，記錄的 `persistence` 是
  `ephemeral`。`drive.mode: "off"`（或 `H3_DRIVE=off`）則完全不掛 Drive。
- 本機 `output/` 仍然會下載一份 MP4：ffprobe 驗證和看畫面都靠它。Drive 那份是正本。
- **Drive 路線「第一次 session」已實機跑過一次**（2026-10-03，見下面「Drive 路線實測」）：模型下載並存進 Drive、
  影片存進 Drive、session 照常關掉。**「第二個 session 從 Drive 複製、不重抓」還沒測**，那才是這條路的主要賣點，
  所以從 Drive 複製回 VM 要多久仍然不知道。

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
| `--manifest batch.json` | 一批多支，**只開一個 session**（見下面「一批多支」） |
| `--segments N` | 同一張圖拆成 N 段接續的片段（第 2 段起從上一段最後一格開始），一個 session |
| `--settle 秒數` | 結束後等多久再讀一次餘額（預設 120，抓延遲扣款） |
| `--ignore-busy` | 帳號上已有別的 runtime 時照樣開始（預設會以 `COLAB_BUSY` 拒絕） |

exit code：0 成功、2 輸入錯誤、3 逾時、130 取消、1 其他錯誤（一批裡有任何一支失敗也是 1）。
stdout 最後一行是 JSON：單支是那一支的記錄，一批是整批摘要（每支的記錄放在 `records`）。

### 一批多支（同一個 session）

每開一個 Colab session 都要裝 ComfyUI、下載約 40 GB 模型，約 6–7 分鐘、0.7–0.8 CU。一批只付一次：
遠端腳本看到同一台機器上已經裝好、下載好、ComfyUI 還在跑，就直接沿用。

```json
{
  "batch_id": "realistic-3",
  "jobs": [
    {"job_id": "a", "input": {"image": "input/a.png"}, "prompt": {"description": "...", "constraints": ["locked-camera"]},
     "settings": {"duration": 5}, "output": "output/a.mp4"},
    {"job_id": "a2", "chain": true, "prompt": {"description": "..."}, "output": "output/a2.mp4"},
    {"job_id": "b", "input": {"image": "input/b.png"}, "prompt": {"description": "..."}, "output": "output/b.mp4"}
  ]
}
```

- `"chain": true`：首幀用上一支的最後一格。那一格一直留在 Colab 的機器上，不必來回傳。
- 每支都先在本機檢查完才開 session；輸入不合格的那支直接標失敗，其他照跑，不花 CU。
- **某一支失敗就跳過、繼續下一支**。接在失敗那支後面的接續片段標 `CHAIN_SOURCE_FAILED`。
- `colab exec` 逾時或沒有任何回報就失敗時，那一支不重試，這個 session 先關掉，下一支開新的 session。
- 每支都會另存最後一格 `<輸出檔名>.last_frame.png`。
- 一批只能用同一種 GPU 設定（`gpu`／`high_mem` 要一樣），最多 20 支。
- 整批的摘要在 `output/batches/<batch_id>.json`，總帳在 `output/h3_batches.jsonl`。

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
- **`colab exec` 15 分鐘沒有任何輸出也算逾時**（`exec_idle_timeout_seconds`）。2026-10-01 實測過
  `colab exec` 斷線後印出 `RuntimeError: Connection was lost.` 卻不結束，A100 空轉了一小時、白花 6.6 CU；
  遠端腳本推論時每 30 秒會印一行，最長的正常空檔是單一大檔下載（約 5–6 分鐘）。
- 開始前會看帳號上有沒有別的 runtime（`active assignments`）。這台機器上其他 Claude 工作階段也用同一個
  帳號，有的話以 `COLAB_BUSY` 拒絕，避免搶 A100、也避免餘額差混在一起。

| error_code | 意思 |
| --- | --- |
| `INVALID_INPUT` / `PROMPT_REJECTED` / `INVALID_CONFIG` | 輸入或設定有問題，沒有花 CU |
| `FFPROBE_MISSING` / `DOCKER_UNAVAILABLE` / `COLAB_CLI_MISSING` | 本機環境缺東西 |
| `AUTH_REQUIRED` | `Google Colab authentication is required. Please authenticate and run the job again.` |
| `GPU_UNAVAILABLE` | `No compatible GPU is currently available.`（沒有配額或暫時沒有 A100） |
| `COLAB_BUSY` | 帳號上已有別的 runtime 在跑（常是另一個 Claude 工作階段），等它結束 |
| `CHAIN_SOURCE_FAILED` | 接續片段的上一支失敗或沒有最後一格，這支沒有花 CU |
| `COLAB_CONNECTION_FAILED` / `UPLOAD_FAILED` | 連不上 Colab 或上傳失敗 |
| `INSUFFICIENT_DISK` / `COMFYUI_SETUP_FAILED` / `MODEL_DOWNLOAD_FAILED` / `INFERENCE_FAILED` | 遠端各階段失敗 |
| `TIMEOUT` | 本機或遠端逾時 |
| `OUTPUT_NOT_FOUND` | 推論結束但沒有可下載的 MP4 |
| `FAILED_VALIDATION` | MP4 下載了但沒過 ffprobe 檢查（檔案保留） |

**ffprobe 檢查項目**：檔案存在且大小 > 0、有 video stream、h264、yuv420p、寬高符合、24 fps、長度 = 幀數／24
（誤差 ±max(0.1 秒, 2 幀)）、有音訊。

## 輸出與成本記錄

- `output/<名稱>.mp4`：影片（本機副本；正本在 Drive 的 `AI-Workflow/outputs/videos/`，路徑記在 `drive_output`）
- `output/<名稱>.job.json`：完整記錄（prompt、設定、每個階段秒數、GPU、VRAM 峰值、ffprobe 摘要、CU）
- `output/h3_jobs.jsonl`：成本總帳，每個 job 一行
- `output/logs/<job_id>.log`：完整 log，每行都有 `[狀態]`
- `output/<名稱>.last_frame.png`：最後一格，手動接片用
- `output/batches/<batch_id>.json`、`output/h3_batches.jsonl`：整批摘要（session 數、總秒數、餘額前後、settle 後的讀數）
- 失敗時 `output/.work/<job_id>/` 會保留上傳的檔案給你除錯；成功就刪掉

CU 用三種方式記：
- `cu_used_measured`：每支各自那一段時間的餘額差。一批裡的第一支也包含安裝和下載。因為延遲扣款，
  批次裡單支的讀數常常落到下一支（實測 0.95／0.00／0.56），只能當參考。
- `cu_used_settled`（整批摘要，單支時也寫進記錄）：session 關掉後再等 `cu_settle_seconds`（預設 120 秒）
  重讀的餘額差。Colab 會延遲扣款，報成本以這個為準。
- `cu_estimated`：每小時用量 × 秒數，只是估算。

**不假設固定的每支 CU**，實測表見下面。

## 設定

`config/config.example.json` 是預設值；要改就建立 `config/config.json`（已 gitignore，只寫要改的鍵）。
環境變數優先於設定檔：`H3_TRANSPORT`、`H3_DOCKER_IMAGE`、`H3_GPU`、`H3_HIGH_MEM`、`H3_TIMEOUT_SECONDS`、
`H3_OUTPUT_DIR`、`H3_FFPROBE`、`H3_CU_SETTLE_SECONDS`、`H3_EXEC_IDLE_TIMEOUT_SECONDS`、`H3_DRIVE`（`consent`／`off`）、
`H3_DRIVE_WORKSPACE`。命令列參數再優先於環境變數。

`drive` 區塊：`mode`、`mount`（`/content/drive`）、`workspace`（`MyDrive/AI-Workflow`）、`consent_wait_seconds`、
`open_browser`、`finalize_timeout_seconds`。`models` 區塊是六個模型檔的釘選（repo、revision、size、sha256、Drive 上
的位置）；其中五個與 `ai_workflow/workflows/registry.json` 完全相同（測試會比對），第六個是 CUDA 13 用的 int8 版。

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
| 單元（225 條） | `powershell -ExecutionPolicy Bypass -File check.ps1` | 什麼都不需要（CI 也跑） |
| 整合（CPU runtime 來回） | `$env:H3_LIVE_COLAB="1"; .venv-dev\Scripts\python.exe -m pytest -m live tests/test_minimax_h3_live.py` | Docker、已登入；佔用 CPU runtime 一兩分鐘 |
| 真實推論 | 上面的 `run.py` 指令 | A100 與 CU |

單元測試涵蓋：設定優先序與機密鍵、prompt 格式與原樣保留、constraint、安全詞（含釘住詞表）、幀數與解析度、
ffprobe 每一條規則、狀態機的每一條失敗路徑（壞圖／缺圖／缺 prompt／長度不合法時 transport 完全沒被呼叫、
未登入、GPU 被拒、模型下載失敗、exec 逾時不重試、Ctrl+C、沒有輸出、驗證失敗、stop 失敗）、真實子程序逾時
（含 exec 沒有輸出的閒置逾時）、Docker 指令組裝，以及遠端節點圖符合 ComfyUI H3 節點契約。
批次另外涵蓋：整批只開關一次 session、一支失敗後下一支照跑、接續 job 用 VM 上的最後一格、來源失敗時接續 job
取消、逾時／session lost 後換新 session、帳號忙碌、settle 重讀，以及遠端平行下載與執行緒安全的標記輸出。

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

## Drive 路線實測（2026-10-03，第一個 session，Google AI Pro）

1 支 5 秒（`s01_scene01_B_h3.mp4`，首幀 720×1280 → 輸出 768×1376、124 幀），沒有別的 runtime，使用者在 session
開始約 2 分鐘內按了 Drive 同意。A100-SXM4-80GB、torch 2.11.0+cu130、`persistence: drive`，五個模型全是
`downloaded`（Drive 上原本沒有）。

| 階段 | 秒 |
| --- | ---: |
| 安裝 ComfyUI 等（setup） | 28.7 |
| 五個模型下載（平行；最大的 19.5 GiB 檔 171 秒） | 171.9 |
| ComfyUI 啟動 | 64.1 |
| 推論 | 170.6 |
| **VM 上的合計** | **440.1** |
| 存 Drive 的 finalize（`drive_saved.seconds`，其中 flush 238.2） | 253.2 |
| 整個 session | 857.5（job 861.3） |

- 背景複製到 Drive 各檔：0.56 GiB 2.5 秒、0.58 GiB 2.6 秒、4.85 GiB 29.0 秒、14.61 GiB 48.4 秒、19.53 GiB 61.3 秒，
  都跟下載重疊，所以複製本身沒有拉長 session；**多出來的是最後的 finalize 約 253 秒**（記錄是 `flushed: true`、
  `error: null`；影片在 `drive_output` = `outputs/videos/s01_scene01_B_h3.mp4`。我沒有在 Drive 網頁上逐檔核對）。
- 模型下載＋啟動合計 265 秒，比 09-30 的「安裝＋下載＋啟動 422 秒」短，推論 170.6 秒（09-30 是 195 秒）。VRAM 峰值 42,106 MiB。
- **CU：餘額 60.25 → 59.12，用掉 1.13**（settle 後再讀一次，一樣）。照 6.77 CU/小時乘 857.5 秒的預估是 1.61，實測是預估的
  70%；09-30 那支 5 秒（663 秒，預估 1.25，實測 0.85）也是約 68%，所以這個比例看起來穩定，原因不明。
- 跟 09-30 同樣是 5 秒單支的 0.85 CU 比，多出來的 0.28 CU 大致就是 finalize 這一步。**第一次存 Drive 要付這筆；**
  之後的 session 不用再下載和複製進 Drive，預期會省回來，但還沒量。
- 還沒量到的：第二個 session 從 Drive 複製（`staged`）要多久；沒按同意時（`ephemeral`）的實際行為；長時間 flush 失敗的情況。

## 實測（2026-09-30，Google AI Pro）

以下是 Drive 路線之前量的（每個 session 都重新下載、不存 Drive）。

兩支都是 Colab 分配的 NVIDIA A100-SXM4-80GB、高 RAM；session 每小時 6.77 CU。首幀是 9:16 的圖，解析度自動選 768×1376。

| | 5 秒（124 幀） | 8 秒（192 幀） |
| --- | ---: | ---: |
| 裝 ComfyUI＋下載 40 GB 模型＋啟動 | 422 秒 | 413 秒 |
| 推論 | 195 秒 | 365 秒 |
| VRAM 峰值 | 43,214 MiB | 45,598 MiB |
| 整個 job | 663 秒 | 818 秒 |
| **CU（實測，餘額差）** | **0.85** | **1.49** |

- 每支大約一半時間花在新 session 的安裝和下載（約 0.8 CU）。現在同一批的多支共用一個 session，這段開銷只付一次。
- A100 偶爾會暫時沒有容量（assign 回 503 → `GPU_UNAVAILABLE`）。這種失敗不花 CU，過 1 分鐘重試就成功了。
- 5 秒那支從第 2 秒起鏡頭連續大幅繞到人物側面（不是硬切）。要保持原構圖，就寫明鏡頭不動並加上 "in one continuous shot"。第 2 支這樣寫之後就維持住了。
- **CU 餘額會延遲扣款**：實測有一次在 job 結束約 1.5 分鐘後，又多扣了 0.57 CU。`cu_used_measured` 是 job 結束當下的讀數，報成本前要隔幾分鐘再讀一次，或看整批的餘額差（目前 6 支平均 1.34 CU，約 5.9 CU/小時）。
- **同一個 Colab 帳號別的工作階段也會用**：另一個 Claude 工作階段可能同時開 job，餘額差會混在一起，而且 A100 也會被佔走。開始前先看 `colab sessions` 有沒有別人的 session；成本以整批的餘額差為準。
- 詳細數字與分析：repo 根目錄 `CHANGELOG.md`。

### 一批三支、同一個 session（2026-10-01，batch `realistic-3`）

三張 Z-Image 生成的虛構成人首幀，各 5 秒（124 幀），鏡頭固定，彼此獨立、不接續。A100-SXM4-80GB、高 RAM。

| | 第 1 支（老師，1344×768） | 第 2 支（店員，768×1376） | 第 3 支（喝咖啡，768×1376） |
| --- | ---: | ---: | ---: |
| 開 session（含等 A100 分配） | 66 秒 | 沿用 | 沿用 |
| 裝 ComfyUI | 31 秒 | 0（沿用） | 0（沿用） |
| 下載模型 | 284 秒 | 0（5 個檔都已在 VM 上） | 0 |
| 啟動 ComfyUI | 24 秒 | 0（沿用） | 0（沿用） |
| 推論 | 190 秒 | 190 秒 | 190 秒 |
| 這支佔用的 session 時間 | 610 秒 | 204 秒 | 211 秒 |
| VRAM 峰值 | 42,710 MiB | 46,150 MiB | 46,142 MiB |

- **整批 1.51 CU**（63.46 → 61.95，關掉 session 後等 120 秒重讀仍是 61.95），平均每支 0.50 CU。
  分開跑三支，照之前 5 秒那支的實測 0.85 CU 算約 2.55 CU，**省了約 1.04 CU（41%）**。
  session 共 1,023 秒，換算約 5.3 CU/小時。從開始到 session 關閉 17 分鐘，加上等 120 秒重讀，總共 19 分鐘。
- **之後每多一支 5 秒影片，大約多 205 秒 session 時間**，照 5.3–6.8 CU/小時約 0.3–0.4 CU。第 1 支要多付約 400 秒的安裝和下載。
- **平行下載幾乎沒有幫助**：284 秒落在之前 254–367 秒的範圍內。19.5 GiB 的主模型單檔就要 283 秒（約 70 MB/s），
  其他 4 個檔在它下載完之前就都好了。省時間靠的是同一個 session 沿用，不是平行下載。
- **模型留在 GPU 上，推論也沒有變快**：三支都是 190 秒，之前單獨跑的 5 秒影片是 195 秒。時間都花在取樣本身。
- **批次裡每支的 CU 讀數不可靠**：三支分別讀到 0.95／0.00／0.56。第 2 支的扣款延遲到第 3 支才出現，
  所以成本只看整批的 `cu_used_settled`。
- 品質：三支都是鏡頭固定、背景和服裝沒變，動作照描述做（打招呼、點頭後低頭看櫃台、舉杯喝一口再微笑）。
  每支 124 格都偵測得到臉，對首幀的平均相似度 0.803／0.733／0.687。最低的兩格是 0.48，
  分別是店員低頭看櫃台和杯子擋住嘴，不是換臉。

## 限制與下一階段

- 一批只能用同一種 GPU，最多 20 支；不同 GPU 要分兩批。
- 存 Drive 之後不再是完全無人值守：每台新 VM 要按一次 Drive 同意。ComfyUI 本身仍然每個 session 重新 clone
  （那是中間過程，不在儲存規則內）。
- 目前只有首幀模式。Ref2VA（1–9 張參照圖）和首尾幀模式，notebook 那邊已經有對應的節點，下一階段再開。
- 還沒做的：多鏡頭／分鏡 → FFmpeg 串接、R2 上傳、音訊替換、Lip Sync、放大、補幀。
  `JobSpec` 的 `job_id`／`scene_id`／`input`／`prompt`／`settings`／`output` 結構就是為這些預留的。
