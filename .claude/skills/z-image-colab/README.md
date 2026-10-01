# Z-Image × ComfyUI × Google Colab：常駐 worker 文字生圖

在 Claude Code 說「用 Z Image 幫我生 10 張不同構圖的產品圖」，Claude 會依 `SKILL.md` 寫好 prompt、建立
10 個 job，開**一個** Colab GPU session 把它們依序跑完，PNG 下載回 `output/zimage/` 並驗證；佇列空了
之後閒置一段時間自動關機。本機 RTX 2070 不參與推論。

第一階段只做 Text → Image。

> **驗證狀態（2026-10-02）**：離線測試 210 個全過（不需要 GPU／Docker／Colab）。**還沒有在真的 Colab GPU
> 上跑過**——下面「實機驗收」四項都還沒做，所有時間與 CU 都是未知數，文件裡不寫估計值當成事實。

## 架構

```
Claude（SKILL.md、prompts/prompt_template.md）
  → scripts/submit.py          建立 job（本機檔案佇列，不花 CU）
  → scripts/worker.py up       controller（本機、背景常駐）
      → google-colab-cli 0.7.4（Docker，與 minimax-h3-colab 共用映像檔與登入）
          → colab new          一個 L4 session
          → colab drivemount   掛 Google Drive（每台新 VM 要你按一次同意）
          → colab exec -f remote/zimage_worker.py
                GPU 檢查 → 版本比對 → ComfyUI／套件／模型（有就用、沒有才裝）
                → 啟動 ComfyUI（只聽 127.0.0.1）→ 節點檢查 → READY
                → 迴圈：收 job → POST /prompt → GET /history → GET /view → 驗證
                → 佇列空了超過 idle_timeout → 關 ComfyUI、清暫存、flush Drive
      ← colab upload / download（job 進、PNG 出、每幾秒讀一次 state.json）
      → 本機再驗證一次 PNG → output/zimage/<job_id>/
      → colab stop             一定會執行
```

為什麼長這樣：

- **佇列在本機**。VM 上的 inbox 只是投遞口；結果用 `job_id` 回報，所以 session 斷掉後重推的 job 不會重畫。
- **controller 只在開頭用一次 `colab exec`**。Colab 的 kernel 一次只跑一個 cell，worker 佔著它（這也讓 VM
  不會被當成閒置回收），之後的 job 與狀態全部走 `colab upload/download`（Jupyter Contents API）。
- **判斷 VM 還活著看的是 `state.json` 的 `seq` 有沒有前進**，不是 exec 的輸出。2026-10-01 的 H3 事故就是
  `colab exec` 斷線後掛著不結束，A100 空轉一小時。`seq` 停 `controller.heartbeat_timeout_seconds`（預設 300 秒）
  就 `colab stop`，沒做完的 job 退回 `PENDING`。
- **workflow 不在這個資料夾裡**：用的是專案已實測的 `training/workflow_template_txt2img_zimage.json`
  （UNETLoader → ModelSamplingAuraFlow shift 3 → KSampler res_multistep／simple、CLIPLoader `lumina2`、
  EmptySD3LatentImage），由 `comfyui_client.build_zimage_txt2img_workflow()` 填參數。本機生圖和 Colab 生圖填的
  是同一張圖，`workflow_contracts` 的 node id ↔ class_type 契約也同時守著兩邊。job 只能改
  prompt／負面詞／seed／寬高／steps／cfg／檔名前綴。
- **安全路徑跟本機一樣**：負面詞由 `generate_character._build_prompt_and_negative` 組（一定含
  `AGE_SAFETY_NEGATIVE`），cfg 下限 1.5。本機 `submit.py` 檢查一次，VM 上的 worker 送進 ComfyUI 前再檢查一次，
  不合格是拒絕、不是偷偷修正。

## 前置作業（一次）

1. **Docker Desktop** 開著，映像檔 `h3-colab-cli:0.7.4` 已建好（與 H3 skill 共用）：
   ```
   docker build -t h3-colab-cli:0.7.4 .claude\skills\minimax-h3-colab\docker
   ```
2. **Google 登入**（你自己在終端機做，Claude 不能代填）：
   ```
   docker run --rm -it -v h3-colab-config:/root/.config/colab-cli h3-colab-cli:0.7.4 --auth=oauth2 usage
   ```
   H3 skill 登入過就不用再做。token 在 Docker volume `h3-colab-config`，不在 repo 裡。
3. **Colab 方案**：要有 CU 而且分得到 L4（24 GB）。預設用官方 bf16 權重（UNet 12.3 GB＋文字編碼器 8.0 GB），
   worker 開機先檢查 GPU：VRAM 低於 `gpu.min_vram_gib`（15）或不支援 bf16 就回 `GPU_NOT_SUPPORTED`、不開始佇列。
   這個門檻是照檔案大小訂的，T4 會不會被擋、L4 實際吃多少 VRAM 都還沒實測。
4. **Google Drive 空間**：約 21 GB（模型 20.7 GB＋ComfyUI 原始碼與套件的壓縮檔）。

檢查全部就緒（不花 CU）：
```
ComfyUI\.venv\Scripts\python.exe .claude\skills\z-image-colab\scripts\setup_colab.py preflight
```

## 使用

透過 Claude：直接描述需求，Claude 會照 `SKILL.md` 走完。直接用 CLI（在 repo 根目錄）：

```
set PY=ComfyUI\.venv\Scripts\python.exe
set S=.claude\skills\z-image-colab\scripts

%PY% %S%\submit.py --prompt "A white ceramic mug on a plain white background, centered." --aspect 1:1 --count 3
%PY% %S%\worker.py up              :: 花 CU。開一個 session，跑完佇列，閒置逾時後自動關
%PY% %S%\worker.py consent-done    :: 瀏覽器按完 Drive 同意之後
%PY% %S%\status.py                 :: 佇列、controller、帳號上的 session
%PY% %S%\cancel.py JOB_ID
%PY% %S%\worker.py stop            :: 提早結束 session
%PY% %S%\validate.py output\zimage\JOB_ID\result.png --width 1024 --height 1024
```

| `submit.py` 參數 | 說明 |
| --- | --- |
| `--prompt` | 原樣送給模型，程式不加任何風格詞。中英文都可以 |
| `--negative` | 額外的負面詞，接在安全負面詞**後面**，不會取代它 |
| `--aspect` | `1:1` 1024×1024、`16:9` 1344×768、`9:16` 768×1344、`4:3` 1152×896、`3:4`、`3:2` 1216×832、`2:3` |
| `--width` / `--height` | 自訂尺寸：16 的倍數、每邊 512–2048、總像素 ≤ `limits.max_pixels` |
| `--steps` / `--cfg` | 預設 8／2.0（turbo 的設定）。cfg 低於 1.5 會被拒絕 |
| `--seed` | 預設 -1＝隨機，實際用的 seed 會記在 job 裡。固定 seed 配 `--count N` 會是 seed…seed+N-1 |
| `--count` | 同一個 prompt 生 N 張（N 個 job，同一個 session） |
| `--solo` | 畫面裡剛好一個人時加上，補「不要第二個人」的負面詞 |
| `--allow-text` | 使用者要求畫面上有字時才加（預設負面詞含 `text`） |
| `--manifest batch.json` | `{"batch": "名稱", "jobs": [{"prompt": "...", "aspect": "16:9", "count": 2}, ...]}`；全部先驗證，有一筆不合格就整批不入列 |
| `--dry-run` | 只驗證並印出 job，不寫入佇列 |

`worker.py up` 預設是背景常駐（它必須活得比啟動它的終端機或 Claude 工作階段久，因為只有它會關 VM）；
`--foreground` 留在終端機。已經有 controller 在跑的時候，`submit.py` 新增的 job 會被同一個 session 接走。

## 持久化：第一次裝好，之後不重裝

Google Drive 上（`colab.persistent_path`，預設 `MyDrive/AI/ZImage`，相對於掛載點）：

```
AI/ZImage/
├── cache/
│   ├── comfyui-<commit>.tar.gz     釘選版本的 ComfyUI 原始碼
│   └── pydeps-<fingerprint>.tar    Colab 映像檔沒有的 pip 套件（PYTHONUSERBASE 整棵樹）
├── models/
│   ├── diffusion_models/z_image_turbo_bf16.safetensors
│   ├── text_encoders/qwen_3_4b.safetensors
│   ├── vae/ae.safetensors
│   └── manifest.json               每個檔：model_name、model_version、download_source、file_size、sha256、installed_at
├── config/environment.json         comfyui_version、z_image_version、custom_nodes、python／torch／cuda 版本、installed_at、history
├── outputs/<job_id>/               result.png、metadata.json、workflow.json（Drive 上的備份）
└── logs/                           worker-<session>.log、errors.jsonl、sessions.jsonl
```

存壓縮檔而不是整個資料夾，是因為 Drive 掛載對大量小檔很慢（Colab FAQ 也這樣建議）。

每個 session 開機時逐項檢查，結果記在 `setup_actions`：

| 項目 | 判斷依據 | 首次 | 之後 | 什麼時候會重做 |
| --- | --- | --- | --- | --- |
| ComfyUI | `cache/comfyui-<commit>.tar.gz` 存在 | `installed`（下載 tarball 一次） | `extracted`（解壓到 VM 本機碟） | `comfyui.commit` 改了 |
| pip 套件 | `cache/pydeps-<fingerprint>.tar`；fingerprint＝python／torch／CUDA 版本＋Colab 映像檔的 `pip freeze`＋ComfyUI commit＋requirements 內容 | `installed`（pip install 一次並打包） | `extracted` | fingerprint 變了 → `migrated`；解壓後 ComfyUI import 失敗 → `reinstalled`（只重做一次） |
| 模型 | Drive 上檔案大小＝設定值，而且 manifest 的 revision／sha256＝設定值 | `downloaded`（HF 下載、驗 sha256、背景複製到 Drive） | `staged`（從 Drive 複製到 VM 本機碟） | 缺檔、大小不符、revision 或 sha256 改了——只處理那一個檔 |
| custom nodes | `cache/node-<name>-<commit>.tar.gz` | 同 ComfyUI | `extracted` | commit 改了（目前 `custom_nodes` 是空的，Z-Image 只用內建節點） |

ComfyUI 啟動時帶 `HF_HUB_OFFLINE=1`、`PIP_NO_INDEX=1`：重啟的 session 如果偷偷想連外下載，會直接失敗而不是默默重抓。
模型檔的 manifest 是「複製完成後」才寫的，所以半途中斷的上傳下次不會被當成完整檔。

**升級**：改 `config/config.json` 的 `comfyui.ref`＋`comfyui.commit`（完整 SHA），或 `model.revision`＋各檔的
`size`／`sha256`。下一個 session 只會重做有變的那一項。版本是釘死的，不會自己跟著 master 跑。

### 限制：每台新 VM 都要按一次 Drive 同意

2026-10-01 實測：`colab drivemount` 的授權是**每個 runtime 一次**，不是每個帳號一次，而且 `drive.mount` 只等
120 秒。所以「新 session 掛載後直接可用」做得到，但開頭需要你在瀏覽器點一下。

- `colab.drive: "consent"`（預設）：controller 開瀏覽器到同意頁，log 印 `DRIVE_CONSENT_NEEDED`。你按完後執行
  `worker.py consent-done`；沒執行的話 `colab.drive_consent_wait_seconds`（預設 90）秒後也會自動送出。
- 沒有同意：這個 session 是 `ephemeral`——照樣出圖，但 ComfyUI、套件、模型全部重新下載（約 20 GB），結束後什麼都不留。
  log 與 session 摘要的 `persistence` 會寫明。
- `colab.drive: "off"`：永遠不問，永遠 ephemeral。

## 設定

`config/config.example.json` 是預設值；要改就建 `config/config.json`（已 gitignore，只寫要覆蓋的 key），
或用環境變數。**設定檔裡出現 key／token／secret／password／credential 字樣的欄位會被拒絕載入。**

| 設定 | 預設 | 環境變數 | 說明 |
| --- | --- | --- | --- |
| `worker.idle_timeout_seconds` | 600 | `ZIMG_IDLE_TIMEOUT_SECONDS` | 佇列空了多久關機。成本的主要旋鈕，範圍 30–14400 |
| `worker.job_timeout_seconds` | 1800 | `ZIMG_JOB_TIMEOUT_SECONDS` | 單一 job 上限，逾時會在 ComfyUI 裡中斷它 |
| `worker.max_session_seconds` | 10800 | `ZIMG_MAX_SESSION_SECONDS` | session 總長上限，到了就關，不管還有沒有 job |
| `worker.poll_interval_seconds` | 5 | | worker 多久看一次 inbox |
| `colab.gpu` | `L4` | `ZIMG_GPU` | `T4`／`L4`／`G4`／`A100`／`H100`（要過下面的 GPU 檢查） |
| `colab.persistent_path` | `MyDrive/AI/ZImage` | `ZIMG_PERSISTENT_PATH` | Drive 掛載點底下的相對路徑 |
| `colab.drive` | `consent` | | 見上一節 |
| `colab.cu_settle_seconds` | 120 | `ZIMG_CU_SETTLE_SECONDS` | 關機後等多久再讀餘額（Colab 會延遲扣款） |
| `controller.heartbeat_timeout_seconds` | 300 | | `seq` 多久沒前進就視為 VM 失聯 |
| `gpu.min_vram_gib` / `gpu.require_bf16` | 15 / true | | 不符合就 `GPU_NOT_SUPPORTED`，不開始佇列 |
| `limits.max_pixels` | 4194304 | | 單張像素上限 |
| `output.directory` | `./output/zimage` | `ZIMG_OUTPUT_DIR` | 相對於 repo 根目錄 |

## Job 狀態與錯誤

```
PENDING → QUEUED → RUNNING → GENERATED → VALIDATING → VALID → COMPLETED
失敗：FAILED / TIMEOUT / CANCELLED / INVALID_OUTPUT
```

- `PENDING`：在本機佇列。`QUEUED`：已上傳到某個 session。之後的狀態由 worker 回報。
- **一個 job 失敗不會讓 worker 停下**：每個 job 包在自己的 try/except 裡；ComfyUI 掛掉會在下一個 job 前重啟。
  錯誤記 `code`、`error_type`、`error_message`、`stack_trace`、`timestamp`（job 記錄與 Drive 的 `logs/errors.jsonl`）。
- session 失聯時，還沒做完的 job 退回 `PENDING`；同一個 job 因為 session 失聯失敗兩次就標 `SESSION_LOST`，不會無限重試。
- 驗證做兩次：worker 在 VM 上驗一次，controller 下載回本機再驗一次（下載壞掉也抓得到）。檢查項目：
  檔案存在、大小 > 0、PNG 簽章、IHDR 在最前面、每個 chunk 的 CRC、有 IDAT 與 IEND、寬高符合；裝了 Pillow 的環境
  （`ComfyUI\.venv`、Colab）再完整解碼一次。

錯誤碼對照表在 `SKILL.md`。

## 輸出與成本記錄（`output/zimage/`，已 gitignore）

- `jobs/<job_id>.json`：佇列本體，含完整 workflow 與狀態歷程
- `<job_id>/result.png`、`metadata.json`（prompt、負面詞、seed、解析度、steps、模型與 revision、GPU、VRAM 峰值、
  生成秒數、兩次驗證結果）、`workflow.json`（實際送出的 API 圖，可直接重現）
- `zimage_jobs.jsonl`：每個 job 一行（`job_id`、`status`、`gpu`、`vram`、`model`、`workflow`、`width`、`height`、
  `steps`、`seed`、`started_at`、`completed_at`、`generation_seconds`、`output`、`session`、`session_reused`…）
- `zimage_sessions.jsonl`：每個 session 一行（結束原因、`setup_actions`、各階段秒數、模型複製秒數、
  `cu_balance_before`／`after`／`settled`、`cu_used_measured`、`cu_estimated`）
- `logs/controller-*.log`、`logs/<session>.worker.log`

成本模型是「啟動＋載入模型＋推論＋最短閒置」只付一次，而不是每張圖都付。實際數字要等實機跑過才有。

## 實機驗收（尚未執行）

| 測試 | 做法 | 通過條件 |
| --- | --- | --- |
| A 首次安裝 | 1 個 job → `worker.py up` → 按 Drive 同意 | `setup_actions` 全是 `installed`／`downloaded`；PNG 下載並通過本機驗證；Drive 上有 tarball、模型、`manifest.json`、`environment.json` |
| B 重啟不重裝 | 再開一個 session | comfyui／deps `extracted`、模型 `staged`；沒有 HF 下載、沒有 pip install |
| C 連續 3 張 | 同一個 session 3 個 job | 3 個 job 的 `session` 相同；`session_reused` 為 false／true／true |
| D 閒置關機 | 佇列空了之後等 `idle_timeout` | 結束原因 `worker_idle_timeout`；`status.py` 看不到 `zimg-*` session |

## 測試

```
powershell -ExecutionPolicy Bypass -File check.ps1
```

`tests/test_zimage_colab_*.py` 共 210 個，約 22 秒，不碰 GPU／Docker／Colab：假的 ComfyUI 是一個本機 HTTP server，
遠端 worker 直接在測試行程裡跑，`colab` 指令由 `FakeTransport` 把 `/content/...` 對到暫存資料夾。涵蓋：
設定與上下限、workflow 只改參數、佇列狀態機、圖片驗證、prompt 不被改寫與安全負面詞、首次安裝／重啟不重裝／
版本變更只重做那一項、單一 job 失敗不影響下一個、job 逾時、閒置逾時、3 個 job 同一個 session、VM 失聯會關機。

## 之後可以擴充的地方（第一階段沒做）

- 新 workflow：在 `zimage_colab.WORKFLOWS` 加一筆（模板檔＋存檔節點），job 的 `workflow` 欄位已經在。
  img2img／inpaint 另外需要把輸入圖上傳到 VM 的 `inputs/`。
- LoRA／custom nodes：`config.custom_nodes` 與模型 manifest 的機制已經支援多個檔案。
- 接到 MiniMax H3：這裡的輸出是本專案管線生成的圖，符合 H3 skill 對首幀來源的要求。
