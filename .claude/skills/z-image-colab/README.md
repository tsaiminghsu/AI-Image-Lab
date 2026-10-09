# Z-Image × ComfyUI × Google Colab：常駐 worker 文字生圖

在 Claude Code 說「用 Z Image 幫我生 10 張不同構圖的產品圖」，Claude 會依 `SKILL.md` 寫好 prompt、建立
10 個 job，開**一個** Colab GPU session 把它們依序跑完，PNG 下載回 `output/zimage/` 並驗證；佇列空了
之後閒置一段時間自動關機。本機 RTX 2070 不參與推論。

第一階段只做 Text → Image。

> **驗證狀態（2026-10-06）**：離線測試全過（不需要 GPU／Docker／Colab）。**實機驗收 A、B、C、D 四項都在 2026-10-06 跑過**，
> 數字見下面「實機驗收」。**B 的結論是：不重裝、不重抓是真的，但從 Drive 取模型比直接下載慢約 6.6 倍**（606 秒對 92 秒），
> 所以這個「不重抓」的賣點在時間上是負的。
>
> **2026-10-08／09 加了「一個 job 一次出 N 張」、`bench.py` 和素材庫，在 A100 與 L4 上實測**：`models.source: download` 實機驗證（下載
> 92～97 秒）、A100 的速度與成本、一次幾張最省、人物年齡與一致性、素材庫 212 張，數字見下面「批次與 GPU 實測」和「素材庫」。

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
| `--batch-size` | **一個 job 出 N 張**（同一個提示詞，ComfyUI 一次推論出 N 張；1～`limits.max_batch_size`）。N 張存成 `result.png`、`result_02.png`…，逐張驗證；要有 VRAM（見下面實測）。manifest 的 job 也可帶 `batch_size` |
| `tag`（只在 manifest） | 記在 job 和 `zimage_jobs.jsonl` 裡的標籤，`bench.py` 和 `training/asset_library.py` 靠它把完成的 job 對回哪一張圖 |
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
| `models.source` | `drive` | `ZIMG_MODEL_SOURCE` | `drive`：Drive 上已有的模型就從 Drive 複製到 VM；`download`：一律從釘選 revision 下載並驗 sha256，Drive 已有的檔不會再存一次（Drive 沒有的照樣存）。實測複製約 20 GB 要 606 秒、下載只要 92 秒，所以想要快、省 CU 就用 `download` |
| `colab.drive` | `consent` | | 見上一節 |
| `colab.cu_settle_seconds` | 120 | `ZIMG_CU_SETTLE_SECONDS` | 關機後等多久再讀餘額（Colab 會延遲扣款） |
| `controller.heartbeat_timeout_seconds` | 300 | | `seq` 多久沒前進就視為 VM 失聯 |
| `gpu.min_vram_gib` / `gpu.require_bf16` | 15 / true | | 不符合就 `GPU_NOT_SUPPORTED`，不開始佇列 |
| `limits.max_pixels` | 4194304 | | 單張像素上限 |
| `limits.max_batch_size` | 8 | | 一個 job 最多出幾張（1–16）；worker 自己的上限是 16，graph 的 batch_size 和 job 不一致會拒絕 |
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

## 實機驗收

**2026-10-06 第一個 session（A、C、D）**：L4（22.5 GB，bf16，torch 2.11.0+cu130）、8 個 job（故事 5 概念片的 7 張
720×1280，加上 10-02 排的 1024×1024 馬克杯測試）、Drive 同意由使用者按了一次（掛載 77.8 秒）。

| 項目 | 實測 |
| --- | --- |
| `setup_actions` | 全部是 `installed`／`downloaded`，存進 Drive（`persistence: drive`） |
| 階段秒數 | DEPS 33.4、MODELS 92.4、START_COMFYUI 39.1、COMFYUI 2.9；從 `worker up` 到 READY 約 4 分 20 秒 |
| 每張生成 | 720×1280、8 步：**18.03–18.22 秒**（7 張）；1024×1024 的馬克杯 53.5 秒（含暖機） |
| VRAM 峰值 | 20,404–22,260 MiB（L4 共 22,563 MiB，幾乎滿；更大的尺寸可能放不下） |
| Drive flush | 10.2 秒 |
| 整個 session | 648.8 秒（其中閒置等 180 秒才關機）；`stop: stopped` |
| **CU** | **34.41 → 34.17，用掉 0.24**（結算後同一個值；費率 1.54 CU／小時，預估 0.278） |
| 收尾 | 結束原因 `worker_idle_timeout`；之後 `zimg-*` 和其他 runtime 都是空的 |

- **A 首次安裝**：通過（8 個 job 都 COMPLETED，PNG 通過本機驗證）。
- **C 同一個 session 連續跑**：通過（8 個 job 同一個 session）。
- **D 閒置關機**：通過。
- **B 重啟不重裝**：通過，也是負面的發現（見下表）。
- 同一個 seed 在 Colab（官方 bf16）和本機（量化版）出的是不同的圖。

**2026-10-06 第二個 session（B）**：同一個 L4、3 個 job（故事 4 的補圖），Drive 掛載成功（105.6 秒，同意頁開了但程式在 90 秒時自己送了
Enter）。

| 項目 | 第一個 session（下載） | 第二個 session（從 Drive） |
| --- | ---: | ---: |
| `setup_actions` | comfyui／deps `installed`，模型 `downloaded` | comfyui／deps **`extracted`**，模型 **`staged`**，沒有 HF 下載、沒有 pip install |
| DEPS | 33.4 秒 | 29.9 秒 |
| **MODELS** | **92.4 秒** | **606.6 秒（慢 6.6 倍）** |
| START_COMFYUI | 39.1 秒 | 29.1 秒 |
| 到 READY | 約 4 分 20 秒 | 約 14 分 |
| 每張 720×1280 | 18.03–18.22 秒 | 18.03–18.04 秒（第一張含暖機 66 秒） |
| 整個 session | 648.8 秒 | 1,098.8 秒 |
| **CU** | **0.24** | **0.40**（預估 0.47） |

- 結論：第二個 session 確實不重裝、不重抓，但**從 Drive 複製三個模型（約 20 GB）花了 10 分鐘，比第一次直接從 HuggingFace 下載慢**，
  整個 session 多花約 67% 的 CU。這和 H3 skill 的量測一致（從 Drive 複製 371–536 秒，下載 172 秒）。
- **選項 `models.source: download`（環境變數 `ZIMG_MODEL_SOURCE=download`）**：第二個 session 起模型改成直接下載，
  ComfyUI 與套件仍然從 Drive 解壓、不重裝。Drive 上已有的模型不會再上傳。2026-10-06 加入，離線測試過，**還沒有實機跑過**，
  所以「下載比較快」目前只有第一個 session 的 92 秒當依據。預設仍是 `drive`。

## 實機驗收項目（A–D 的定義）

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

## 批次與 GPU 實測（2026-10-08，A100／L4）

用 `scripts/bench.py` 在真實的宣傳片分鏡提示詞上量（`plan` 寫測試 manifest、`report` 算表、`faces` 用 InsightFace 量年齡與身分）。
全部 `ZIMG_MODEL_SOURCE=download`、720×1280、8 步、cfg 2.0。

**A100（一個 session：36 個 job、51 張，692 秒，0.98 CU）**

| 一次幾張 | 每張秒數 | VRAM |
| --- | ---: | ---: |
| 1 | 5.4（同一提示詞連跑 5.8） | 20.7 GB |
| 2 | 4.7 | 21.3 GB |
| 4 | 4.5 | 29.6 GB |
| 8 | 4.3 | **39.0 GB**（A100 共 40 GB，幾乎滿） |
| 864×1536，1／2 張 | 7.2／6.6 | 20.9／27.0 GB |

- 費率 **5.3 CU/小時**（不是 H3 的 6.77）；固定成本（安裝、下載模型、啟動、閒置關機）約 0.54 CU，每張邊際約 0.0087 CU。
- 一次多張只快 16～21%（批次 4 是合理的最佳點，批次 8 太貼近顯存上限）；要省 CU，做法是把很多張放進**同一個 session**，不是加大批次。
- **A100 對 L4**：每張的 CU 差不多（L4 約 0.0077：18 秒 × 1.54 CU/小時；A100 批次 1 約 0.0079、批次 8 約 0.0063），L4 固定成本約 0.2。
  所以 **200 張以內 L4 比較省，A100 只是快 3～4 倍**；批次 8 的 A100 要超過約 240 張才反過來。實際：196 張在 A100 用了 2.19 CU
  （1549 秒，臨時儲存），差不多張數在 L4 估約 1.7 CU、要 1 小時。L4 的 22.5 GB 一次 1 張已用到 22 GB，批次 2 以上沒有測。
- **模型直接下載（`models.source: download`）實機通過**：MODELS 階段 91.9～96.7 秒（Drive 複製是 606 秒），A100 與 L4 都是。

**真人感：年齡與一致性（貓咖六個鏡頭，A100，第二輪 1.15 CU）**

估計年齡是 InsightFace buffalo_l 的推測（有誤差，系統性偏高，只用來比較版本）；身分相似度是六個鏡頭兩兩的特徵餘弦相似度（0.5 以上通常同一人）。

| 版本 | 人物寫法 | 風格 | 估計年齡（平均，範圍） | 身分相似度（平均，最低） |
| --- | --- | --- | --- | --- |
| v0 | 「adult woman aged 23」（原本） | bible | 30.5（26–37） | 0.47（0.37） |
| v0 | 同上 | documentary | 32.8（27–44） | 0.45（0.25） |
| **v1** | 「young adult in her early twenties, youthful fresh face, smooth clear skin」 | bible | **24.2（23–25）** | **0.51（0.46）** |
| v1 | 同上 | documentary | 26.0（24–29） | 0.40（0.26） |
| v2 | v1＋固定五官（臉型、眼睛、鼻子、嘴唇、痣、瀏海） | bible | 26.7（24–35） | 0.52（0.32） |
| v3 | v2＋六個鏡頭同一個 seed | bible | 25.5（23–28） | 0.48（0.31） |

- 寫「aged 23」會生出 30～45 歲的臉；改成「early twenties」措辭才到約 24 歲。固定五官與固定 seed 沒有明顯加分；documentary 風格毛孔更真但更老、
  相似度更低。選用 **v1＋bible**（`visual_bible.json` 的 `identity_variants.v1`，`concept_colab.py export --identity v1`，並自動加
  `outstretched arm, selfie` 負面詞擋掉鏡頭 1 常見的自拍手臂）。
- 同一組提示詞換故事不保證同樣效果：四支新故事（L4，48 張，0.46 CU）估計年齡 27～32、相似度 0.39～0.49，日本旅行最差（最低 0.21）。
- Z-Image 沒有參考圖，同一人的六個鏡頭只是「同一類型的人」；和 ChatGPT 生的臉不是同一批，不能混用。

## 素材庫（`training/asset_library.py`，2026-10-09）

用這個 skill 一次生 212 張 720×1280：人物定裝照 64、同樣 64 人在場景裡的第一人稱中景 64、空景 84（4 個年代 × 21 個場所）。
人物是 2 性別 × 4 個成年年齡層 × 4 個年代（現代、近代復古、古代、未來）× 2，分到七個地區；全部成年，prompt 不含未成年字眼（測試釘住），
年齡安全負面詞與 cfg 下限照舊。圖與索引在 `projects/asset_library/`（`CATALOG.md` 追蹤，圖不追蹤），`index.html` 可篩選。

| 階段 | GPU | 張數 | 時間 | CU |
| --- | --- | ---: | ---: | ---: |
| 試跑 | L4 | 16 | 643 秒 | 0.18 |
| 年齡驗證 | L4 | 8 | 482 秒 | 0.19 |
| 全量 | A100（臨時儲存） | 196 | 1549 秒 | 2.19 |
| 補生 1（20 張） | L4 | 20 | 675 秒 | 0.21 |
| 補生 2（7 張） | L4 | 7 | 447 秒 | 0.16 |
| 補生 3（5 張，換場所） | L4 | 5 | 396 秒 | 0.11 |
| 合計 | | | | **3.04** |

- **年齡層**：女 26／38／58／76、男 30／46／68／75（估計值，由小到大拉開）。「late thirties／mature adult face」會被畫成 55～65 歲，要寫
  「mid-thirties, fresh smooth complexion」；五六十歲寫「early fifties」。人像定裝照與場景內同一人的相似度平均 **0.55**（64 對，最低 0.21）。
- **空景要用眼睛複查，臉部偵測不夠**：偵測只抓到 4 張有臉，我看圖總覽實際有約 12 張有人、手、腳或半裸的人（生物發光森林一群只穿內褲的人），
  還有可讀文字（NOEW、NSSW、TEVY、TELEPHONE）。負面詞（nude, naked, legs, feet, hand, letters…）補生後有一半修好；
  **地鐵、網咖、電影院、電話亭、洞穴蘑菇林這類本身暗示「有人」或「有標誌」的場所，加負面詞也擋不住，要換場所**。
- 端莊問題：肯特布洋裝畫成無肩帶、披肩露半邊胸，改成有袖高領／蓋住兩肩；1970 年代西裝領口開太低，改成扣好襯衫加領帶。
- 最後 1 張（未來洞穴蘑菇林）仍有兩個人彎腰在看，待換場所。

## 之後可以擴充的地方（第一階段沒做）

- 新 workflow：在 `zimage_colab.WORKFLOWS` 加一筆（模板檔＋存檔節點），job 的 `workflow` 欄位已經在。
  img2img／inpaint 另外需要把輸入圖上傳到 VM 的 `inputs/`。
- LoRA／custom nodes：`config.custom_nodes` 與模型 manifest 的機制已經支援多個檔案。
- 接到 MiniMax H3：這裡的輸出是本專案管線生成的圖，符合 H3 skill 對首幀來源的要求。
