# 變更記錄

每次疊代改了什麼、為什麼這樣改、實測數字如何。安裝步驟和用法在
[README.md](README.md)，這裡只放結論和對應章節連結。倒序排列，`xxxxxxx` 是 commit。

## 2026-09-18

### 各模型關鍵字總表與 Prompt 範例，並用測試把文件釘在程式上

- **問題**：`training/PROMPT_GUIDE.md` 是 2026-09-08 寫的 prompt 食譜，之後一次都沒更新，
  而且**沒有任何檔案連到它**（`grep PROMPT_GUIDE` 全 repo 零命中），所以沒人會發現它爛掉。
  實際爛的地方有三處：只列了 6 個單檔 checkpoint（現在總共 13 個模型，少了 Z-Image、Wan 2.2、
  AnimateDiff 的 3 個 checkpoint 加 8 個動作 LoRA、SVD）；引用的 `SAFE_SAFETY_NEGATIVE` /
  `SUGGESTIVE_NEGATIVE` 還是 `QUALITY_NEGATIVE` 長出重複人物詞之前的版本；完全沒提
  `SD15_GENDER_WEIGHT`、`VIDEO_REALISTIC_NEGATIVE`、`WAN_VIDEO_NEGATIVE`、角色身分前綴、
  詞庫、場景庫、骨架庫。
- **同一份舊字串在另外兩個地方**：`README.md` 的「Prompt 結構」負面詞區塊（而且順序也寫錯，
  `REALISTIC_NEGATIVE` 實際上是接在最後而不是中間），以及 `training/model_prompt_test.py`
  ——它自己複製了一份 `SAFETY_NEGATIVE`/`REALISTIC_NEGATIVE`/score 標籤，所以那個「實測各
  checkpoint prompt 語法」的腳本量到的已經不是管線真正送出的東西。
- **解法**：重寫 PROMPT_GUIDE.md（614 行）成一份總表——13 個模型的 key/檔名/家族/方言/解析度/
  取樣設定/自動加什麼/功能可用性、組裝規則與各入口差異、13 個常數逐字、11 個詞庫、11 個角色的
  展開前綴、15 個場景、31 組骨架、以及 8 組**管線實際送出的**完整 positive/negative。
  `model_prompt_test.py` 改成 `import generate_character as gc` 直接用常數，不再自己抄一份。
- **文件不會再爛掉**：新增 `tests/test_prompt_guide.py`（213 個案例），每一條斷言都從活的常數
  重新推導，8 組範例是**重新呼叫 `_build_prompt_and_negative()`／`build_wan_prompts()` 比對**，
  失敗訊息直接印出可以貼回文件的正確字串。常數那節比對的是「整個 fenced block 相等」而不是
  子字串——子字串檢查會在把 `score_9, score_8_up, score_7_up` 截短成 `score_9, score_8_up`
  時通過（因為完整字串還出現在下面的範例裡），實測確認過這個破綻並改掉。
  另外有一條 meta 測試：`generate_character.py` 新增任何 `*_NEGATIVE`/`*_TAGS`/`*_STYLE`/
  `*_PROMPT` 常數而沒寫進文件就會紅，否則這個檔案會靜悄悄地漏掉新東西。
- **順手修正**：`tests/TEST_PROMPTS.md` 說 Z-Image「極快（~3-4 秒）、原生解析度約 512×768」，
  兩個都錯——`ZIMAGE_WIDTH/HEIGHT` 是 1024×1024，README 的實測是 cfg 1 約 75 秒、本專案用的
  cfg 2 約 155-195 秒；裡面的「第 114-181 行」這類行號引用也改成符號名稱。
- **驗證**：`check.ps1` 全綠，1482 passed（原 1269）。三次刻意改壞（改範例一個字、截短常數
  區塊、改動作 LoRA 的 key）都被抓到並印出正確字串。換行稽核通過（README.md 仍是 CRLF，
  新的 .md/.py 是 LF）。

## 2026-09-17

### 雲端生圖：把本機組好的完整 workflow 送到 RunPod 執行

- **問題**：雲端影片已經可用，但「自訂生圖」與 CLI 的靜態圖**沒有任何上雲的入口**。worker 雖然有
  `image_hq`，但它必須有臉圖、沒有姿勢骨架、checkpoint 綁死在 endpoint，而且 Z-Image、SD1.5、GIF
  都沒有。照高階參數的做法，每個功能都要在 worker 再寫一份，跟本機逐漸漂移。
- **解法**：在既有架構上加第四種工作 `workflow`，**不動**另外三種。本機照常把完整 workflow 組好
  （prompt、安全負面詞、FaceID、骨架、精修全部在本機），整包送去執行。worker 用它自己的
  `_submit_and_wait` 跑，所以量化版選擇、cfg 下限、`/free` 重置都在 GPU 旁邊發生。只要 volume 上有
  模型，骨架、Z-Image、SD1.5、GIF 就自動能上雲，不需要逐功能寫 worker 程式。
- **切換點只有兩個**：`comfyui_client._submit_and_wait` 與 `upload_reference_image`，也就是所有流程
  必經之處。優先序：GUI 勾選（執行緒範圍）> CLI `--backend`（行程範圍）> `GENERATION_BACKEND` >
  預設本機。偵測到 `RUNPOD_POD_ID` 時拒絕開雲端模式，避免 worker 自己再把工作送回 RunPod。
- **worker 不能直接相信收到的圖**，所以新增 `training/workflow_safety.py`，本機送出前與 worker 執行前
  各跑一次：節點類別只能是 repo 模板用到的（擋掉 `SamplerCustom*` 這類可能跳過負面詞的取樣器）；每個
  `negative` 輸入**沿連線往回追**，穿過 ControlNet 等傳遞節點，源頭必須是含年齡保護詞的文字
  （SVD 的圖像條件是唯一例外，它本來就沒有文字編碼器）。先盤點過全部 12 個模板的負面條件來源，
  只有三種，規則剛好涵蓋。
- **參考圖**：雲端沒有共用硬碟，`upload_reference_image` 在雲端模式改成暫存 bytes、以內容雜湊命名，
  跟著引用它的工作一起送。雜湊命名讓 GIF「上傳一次、送出 N 幀」在無狀態的 worker 之間也正確。
- **重用既有程式**：RunPod 的送出、輪詢、取消、逾時、設定檔、送出不重試等全部沿用 `cloud_video.py`。
  唯一的修改是 `RunPodBackend.result` 的副檔名改由 worker 回傳的 `outputKey` 決定（原本寫死 mp4），
  只接受 png／mp4／webm。
- **CLI 隱藏問題**：`generate_character.py` 直接執行時是 `__main__`，雲端模組延遲匯入的會是**另一份**
  `generate_character`，兩份的 `UsageError` 不是同一個類別，錯誤會變成完整 traceback。進入點先把自己
  註冊成 `generate_character`，並補上 `CloudJobFailed` 轉一行錯誤。實測：沒設定時 exit 1、只印一行。
- **驗證**：
  - `check.ps1` 全綠，1269 passed（原 1205）。`test_workflow_safety.py` 用**真實建構器**產出 15 種圖
    （含骨架走 ControlNet、AnimateDiff 加 RIFE、Wan、SVD）全部通過，並逐一證明各種繞過方式會被擋；
    `test_cloud_workflow.py` 涵蓋雲端模式完全不碰本機 ComfyUI、送出的圖帶年齡詞且 cfg ≥ 1.5、不安全
    或缺圖或過大的請求在送出前就擋下、逾時會取消、worker 錯誤原文保留（AnimateDiff OOM 重試靠比對它）。
  - **模擬雲端往返**：本機組出真正要送的請求（HQ + FaceID + 姿勢骨架，2.15 MB），經 JSON 序列化後
    原封不動交給 worker 的程式碼，在本機 ComfyUI 產出 704×1024 的圖，圖片改名、LoadImage 重寫、
    骨架與輸出回收都正確。
  - GUI 實機點過：勾選雲端、沒設定時在啟動本機 ComfyUI 之前就提示缺 RUNPOD_API_KEY 與 endpoint ID。
  - **尚未在真正的雲端 GPU 上跑過**，冷啟動、每張耗時與成本都沒有實測數字。
- **限制**：雲端生圖仍佔 `concurrency_id="gpu"` 這條排隊；進度只有雲端狀態文字；本機 8GB 的限制常數
  （高清 1.5 倍等）照樣套用，大卡沒有自動放寬。
- **附帶發現，這次沒有修**：
  - 「自訂生圖」與「批次生成 GIF」的 Checkpoint 下拉選單預設值 `cyberrealistic_pony` 不在選項清單裡
    （清單裡是 `cyberrealistic_pony (預設 - Pony 系寫實)`），gradio 6.24 下不動下拉選單直接按生成會被擋。
    這是先前就存在的問題，「圖片選擇生圖」分頁寫法正確。
  - **`safe` 分級會漏出泳裝**：模擬往返那張圖分級是 `safe`、prompt 沒提服裝，cyberrealistic_pony 畫出
    高衩泳裝。`SAFE_SAFETY_NEGATIVE` 只擋 `nsfw, nude, naked, explicit, sexual content`，沒有擋泳裝、
    內衣、暴露服裝，而這些照定義屬於 `suggestive`。改全域負面詞會影響所有生成，留待決定。

### 雲端影片：RunPod 跑 Wan 2.2 圖生影片／AnimateDiff，Replicate 跑託管模型 `28de7e0` `9929671`

- **問題**：本機 8GB 卡把影片品質卡死在兩處——AnimateDiff 二段高清上限 768²、只能用 SD1.5 motion
  module；新一代圖生影片模型完全跑不動。只把同一套 AnimateDiff 搬到大卡上，畫質天花板仍是 SD1.5，
  所以**先做新模型**：Wan 2.2 TI2V-5B，拿角色 anchor 或「🎯 圖片選擇生圖」的結果當第一幀。
- **兩個平台刻意不對等**：
  - **RunPod Serverless** 跑本專案自己的 worker。`worker/handler.py` 拆成薄的 S3／進度外殼 +
    新的 `worker/jobs.py`（不 import boto3／runpod，可離線測），支援 `image_hq`（預設，行為不變）、
    `video_wan_i2v`、`video_animatediff`，全部走跟本機同一套 `generate_character`，安全負面詞與
    cfg 下限在伺服器端保證。
  - **Replicate** 跑別人的模型，所以 `training/cloud_video.py` 使用前讀模型輸入格式，**缺負面詞、
    圖片或 cfg（上限 ≥ 1.5）欄位一律拒絕**。cfg 欄位是必要條件：沒有 cfg 控制的模型常是 cfg 1 蒸餾
    模型，負面詞形同虛設。預期多數託管「快速版」模型會被拒。
- **外部細節全部查證，不憑記憶**（2026-09-17）：Wan 設定照 Comfy-Org 官方 `video_wan2_2_5B_ti2v`
  範本（shift 8、20 步、cfg 5、uni_pc／simple、1280×704、121 幀、24 fps），模型網址逐一 HEAD 檢查
  （5B 10.0 GB＋umt5 6.7 GB＋VAE 1.4 GB）；RunPod `/run` 上限 10 MB、`policy.executionTimeout`
  單位毫秒、結果保留 30 分鐘；Replicate data URL 上限 256 KB。Replicate 的 Files API 在官方參考
  文件查不到，所以**沒有使用**，過大的第一幀改成壓成 256 KB 內的 JPEG。刻意不用 cfg 1 的
  lightning／蒸餾 LoRA。
- **順手修掉從沒 build 過的 worker image 裡的潛在錯誤**：基底映像是 cuda12.4.1／torch 2.4，但 lock
  鎖的是 `torch==2.11.0+cu128`（而且那個 wheel 不在 PyPI，要加 PyTorch cu128 索引）；ComfyUI 與節點包
  都追分支名稱，現在全部鎖到本機實際跑的 commit，並補上 worker 從來沒有的 AnimateDiff-Evolved 與
  Frame-Interpolation；`extra_model_paths.yaml` 缺 Wan／AnimateDiff 的模型資料夾；S3 client 沒有
  `endpoint_url`，R2 不能用。
- **成本與可靠性**：逾時、GUI 取消、Ctrl-C 都會先送遠端取消；建立工作的 POST 不重試（回應遺失時重送
  會變成兩個計費工作）；輸出檔名用雲端工作 ID，不再用會跨 session 撞名的 `gui_seed<seed>`；密鑰只讀
  環境變數，設定檔拒存 key／token。GUI 新分頁「☁️ 雲端影片」用自己的 `concurrency_id="cloud"`，
  雲端工作不會卡住本機生成。
- **驗證**：
  - `check.ps1` 全綠，1205 passed（原 1104）。新增 `test_cloud_video.py`（假 HTTP session：狀態轉換、
    每種中止路徑都剛好取消一次、401／402／429 訊息、送出不重試、安全閘門、payload 的安全欄位）、
    `test_worker_handler.py`（斷言 boto3／runpod 沒被 import）、`test_wan_i2v.py`、
    `test_validate_workflow_nodes.py`。
  - 新工具 `training/validate_workflow_nodes.py` 對本機 ComfyUI（62b3c94）的 `/object_info` 檢查，
    不載入模型：12 個模板（含 Wan）0 個未知類別或輸入。用同一份真實 object_info 做突變測試，拼錯的
    輸入、不存在的類別、拼錯的 codec 子選項都有抓到；uni_pc／simple、mp4／h264 確認是合法選項值。
  - GUI 在沒有帳號的狀態下實機點過：狀態列正確列出 4 項缺漏設定、「用角色的 anchor」帶入圖片、
    生成被擋下並說明缺 RunPod key 與 endpoint、無工作時取消給提示、儲存 endpoint 後狀態列更新。
  - **尚未在真正的雲端 GPU 上跑過**（還沒有 RunPod 帳號／Replicate token），冷啟動、每支耗時與成本
    都還沒有實測數字。
- **附帶發現**：Comfy-Org 文件寫 Wan 2.2 5B「用 ComfyUI 原生 offload 可以塞進 8GB VRAM」。這張卡是
  PCIe x1，offload 會非常慢，沒有實測，但代表 Wan 也許能在本機做小規模的畫面驗證。

## 2026-09-13

### 生成進度改成顯示真實步數，不再顯示推估的百分比

- **問題**：Gradio 的進度條是用「同一個 event 前幾次跑了多久」外推出來的 ETA。這在這裡沒有
  參考價值——同一顆按鈕可能是 30 秒的純文字生圖，也可能是 150 秒以上的 hires + FaceDetailer，
  差別只在幾個開關，所以那個百分比在大半個過程裡都是錯的。
- **解法**：ComfyUI 只在 `/ws` 上送真實步數（`{"type":"progress","data":{"value","max","node"}}`），
  `GET /history` 在完成前什麼都沒有，所以顯示步數非得走那個 socket。新增
  `client.progress_reporter()` context manager，listener 掛在 `_submit_and_wait`——CLAUDE.md
  指定的單一改寫點，所以每條流程都自動吃到。callback 是 **thread-local** 而不是模組全域，
  因為 Gradio 每個請求各跑一條 thread，兩個瀏覽器分頁不能互相蓋掉。
- **依賴用 aiohttp**：它已經是 ComfyUI 的依賴、**而且已經在 `comfyui-requirements.lock.txt` 裡**，
  所以 worker image 不受影響（CLAUDE.md 警告過往 `ComfyUI\.venv` 加套件會被 `uv pip freeze`
  進 worker）。**順帶發現**：`websockets` 套件其實沒安裝，所以 `benchmark.py` 的 `WsStageTimer`
  一直靜默失效、從來沒產出過 per-stage 計時。這次沒有修它。
- aiohttp 是**在 listener thread 裡 lazy import**，任何失敗都只是沒有進度、不會讓生成失敗：
  這個模組刻意維持 import 時只依賴 `requests`，而測試用的 `.venv-dev` 根本沒有 aiohttp。
- **階段名稱從 workflow 自己的 `class_type` 解析**，不維護節點 id 表——節點 id 在模板被重新
  匯出時就會全部重編號（`workflow_contracts.py` 就是為這件事存在的，而且它已經點名
  `benchmark.py` 的 `HQ_STAGE_NAMES` 是同一份知識的複製品）。同一種 sampler 出現多次時，
  按**實際執行順序**編號，而不是按節點 id（後者跟執行順序無關）。
- **實測一次完整 HQ 路徑**：`生成 24 步` → `放大 6 步` → `第 2 段生成 20 步` →
  `臉部/手部精修 20 步`，共 70 個進度事件。這四段的性質和解析度都不同，FaceDetailer 的總步數
  還取決於它找到幾張臉和幾隻手——所以進度條只綁在**當前階段**（ComfyUI 有告訴我們該段的總數），
  步數則直接顯示在說明文字上，不假裝知道整個工作還剩多少。
- `tests/test_gui_arity.py` 補上 Gradio 的真實規則：`gr.Progress` 是靠**預設值**偵測的
  （`helpers.py` 的 `isinstance(param.default, Progress)`），不是靠註解，所以只看註解會把
  `progress=gr.Progress()` 誤算成一個少接的 input。

### WD14 Tagger：上傳圖片直接轉成 booru 標籤

- **問題**：GUI 既有的「依圖片產生 Prompt」用 BLIP 出自然語言句子，但預設 checkpoint 是
  `cyberrealistic_pony`，而 Pony 系是**用 booru 標籤訓練的**（CHECKPOINTS 的註解就寫著這件事）。
  想少打字的人拿到的是最不對格式的那一種。
- **解法**：裝 `ComfyUI-WD14-Tagger`，新增 `client.tag_image()`，GUI 在同一個 accordion 加第二顆
  按鈕。**BLIP 保留**——juggernaut 和 SD1.5 要的就是自然語言，兩者是互補不是取代。
- **沒有新增 Python 依賴**：節點只要 `onnxruntime`，而它跟 `huggingface_hub`、`pandas` 都已經
  裝好了，所以 worker lock 同樣不受影響。
- 模型 `wd-swinv2-tagger-v3`（446 MB，v3 中公認最準的中型模型）下載到節點自己的 `models/`。
- `pysssss.user.json`（優先於 `pysssss.json`，所以 `git pull` 節點不會蓋掉）把 `ortProviders`
  **釘死在 CPU**：節點預設 CUDA 優先，但這台只裝了 CPU 版 `onnxruntime`；而且
  `caption_image.py` 當初就刻意跑 CPU，理由是不要讓第二個模型跟正在跑的生成搶 8 GB。
- 走節點自己的 HTTP 路由 `/pysssss/wd14tagger/tag`，不必組 workflow，也就完全不經過
  `_submit_and_wait` 的佇列／變體／cfg 機制。那個路由只讀 ComfyUI 自己的 input/output/temp
  目錄，所以先用既有的 `upload_reference_image()` 上傳。回傳的是 JSON 字串，要用 `.json()`
  取值，`.text` 會連引號一起拿到。
- **實測**：CPU 上單張 3.3–3.5 秒。範例輸出
  `1girl, solo, long hair, looking at viewer, standing, jacket, white shirt, full body, jeans, crossed arms`
  ——連姿勢（`crossed arms`）和單人（`solo`）都抓得到。

## 2026-09-12

### 「圖片選擇生圖」分頁：點縮圖組 prompt，每個模型自動換成它聽得懂的敘述

- **問題**：每個模型的敘述慣例都不一樣——Pony 要 `score_9` 系列品質標籤、SD1.5 男性角色要把
  性別加權重（否則被畫成女生）、Z-Image 吃自然語言。使用者得自己記得哪個模型要加什麼，
  打錯就是白生一張。而且「姿勢」和「臉」在 SDXL 上明明可以用 ControlNet 骨架和 FaceID
  **結構性地鎖住**，用文字描述反而是最弱的做法。
- **解法**：新增分頁，人物／姿勢／場景三個縮圖庫各點一張，`gc.plan_picker()` 依 checkpoint
  翻譯成該模型能接受的輸入：
  - **SDXL / Pony** → anchor 圖走 FaceID、骨架走 ControlNet（真的鎖住，不是文字）
  - **Z-Image / SD1.5** → 這兩個沒有這些 adapter（`_plan_custom` 會直接拒絕），所以自動降級成
    文字：骨架換成它的 `prompt_hint` + `camera`、角色換成外貌描述，並在 GUI 明講哪些被降級了
  - 最終 prompt 即時預覽。預覽是**直接跑真正的管線函式**（`_plan_custom` →
    `_build_prompt_and_negative`）產生的，不是另外拼一份字串，所以不可能跟實際送出的內容不一致
- **順帶修掉的洞**：姿勢庫 31 個裡有 12 個是自動抽取的，`prompt_hint` 是空字串。在有 ControlNet
  的路徑上沒差（骨架自己會講話），但在 Z-Image/SD1.5 上等於**選了姿勢卻什麼都沒要求**。
  已全部補上文字描述，並加測試擋住未來再出現空的。
- **場景庫**（`training/scenes/`，15 個）：縮圖預先生成後 commit，開分頁不用 GPU。
  `tier` 是能見度規則——`safe` 分級下海灘/泳池那幾個場景根本不會出現在圖庫裡，而不是出現了
  再讓負面詞去對抗它。
- **實測（縮圖批次在 8 GB 卡上的兩次卡死）**：原本用 SDXL 原生 1024²，跑到**第 7 張**卡死在
  `VAEDecode`——7.9/8.2 GB、GPU 100%、**沒有拋 OOM**，是靜默換頁到共享記憶體，在 PCIe x1 上
  等於停住。改 768² 後撐到**第 10 張**才發生同樣的事，而且那次 ComfyUI 有 132 秒沒回應 HTTP，
  連客戶端的重試預算都用光。所以壓力不是單張太大，是**同一個 ComfyUI session 連續生成累積的
  記憶體碎片**。最後兩件一起做：768²（解碼 activation 縮到 56%，反正輸出都是 256px）+ 每張
  render 前先 `client.free_vram()`（每張多約 20 秒重載 checkpoint，換來整批能無人看顧跑完）。
  之後每張 12-16 秒、VRAM 穩定在 5.7 GB、溫度 65-73°C。
- **測試**：`test_picker_plan.py`（每個模型家族的翻譯結果、hint 不重複、安全負面詞在所有家族
  都存在）、`test_scene_library.py`（含「JSON 有但縮圖沒 commit 就紅」）、
  `test_pose_library_meta.py`。`test_gui_arity.py` 教會兩件事：`evt: gr.SelectData` 這種由
  Gradio 依型別注入的參數不算 `inputs=`，以及指派給名字的字面清單長度仍可靜態檢查（新分頁
  九個事件共用同一組六個輸入元件，不必把清單複製九次）。

### 翻譯按鈕的 Google Translate 被限流，加 MyMemory 當備援 `bc16bf7`

- **問題**：實測發現 `translate_prompt.py` 唯一用的 Google Translate 免費端點對這台機器的
  IP 直接回 429（連續重試 3 次、換瀏覽器 User-Agent 都一樣），不是瞬間性的限流——「翻譯成
  英文」按鈕整個掛掉，除了自己手動打英文沒有別的辦法。
- **解法**：Google 失敗時自動改用 MyMemory 免費翻譯 API 當備援，兩邊都失敗才報錯（訊息附
  兩邊各自的失敗原因）。套用 MyMemory 時發現兩個要處理的差異：
  - 它沒有 Google `sl=auto` 那種「來源=目標語言就直接不改動」的行為——對已經是英文的輸入
    傳 `autodetect|en` 會被 403（偵測出英文，拒絕英文翻英文），所以純 ASCII 文字直接原樣
    回傳，不呼叫 API。
  - **它自己的語言偵測對短中文片語不穩定**：同一句「蓬鬆的棉被」在不同次呼叫之間一下正確
    判斷成中文、一下又誤判成英文而 403，同樣的輸入結果不一致。這個專案 GUI 是繁體中文，
    含 CJK 表意文字的輸入現在明確指定 `zh-CN` 來源，不信任它的自動偵測；日文、韓文等非
    CJK 的非 ASCII 輸入才繼續用 `autodetect`（沒觀察到這類輸入有同樣的不穩定）。
  - 它回傳的 `responseStatus` 欄位型別不一致（成功時是整數 200，失敗時是字串 `"403"`，都是
    實測到的），改成用字串比較。
- **驗證**：離線加了 10 個 mock 測試涵蓋備援鏈、ASCII 快速路徑、CJK 對 autodetect 的路由、
  `responseStatus` 型別不一致，`check.ps1` 全過（925 passed）。對著真實服務連跑兩輪，確認
  之前不穩定的「蓬鬆的棉被」案例、中文、已是英文、逗號分隔片語、日文全部穩定翻譯正確。

### Z-Image 換 prompt 會爆 VRAM，補上跟 LCM 切換一樣的自動 `/free` `b6043ca`

- **問題**：Z-Image 的主模型（5.9 GB）跟文字編碼器（5.24 GB）加起來超過這張卡的 8 GB
  VRAM，兩個沒辦法同時常駐。同一個 prompt 換 seed 只會用到已經常駐的主模型沒事，但一換
  **不同的** prompt，文字編碼器就要搬回 VRAM 跟主模型擠——沒有先騰出空間的話，這個瞬間疊加
  可能超過 8 GB，直接爆掉或掉到走 PCIe x1 的龜速共享記憶體，而不是 README「Z-Image Turbo」
  章節原本記錄的乾淨換模型。
- **解法**：仿照既有的 LCM ↔ 一般取樣切換保護（`_reset_if_mode_switch`），加了
  `_reset_if_zimage_prompt_changed()`：比對這次送出的 workflow 跟上次執行的
  `CLIPTextEncode` 文字內容，兩者都是 Z-Image、而且 prompt 真的變了才觸發 `/free`；佇列上
  還有別人的工作在跑就跳過（避免把別人正在取樣用的模型卸載掉）。掛在 `_submit_and_wait`
  這個所有流程共用的單一改寫點，跟 `apply_model_variants`、`_reset_if_mode_switch` 同一個
  位置。
- **驗證**：離線加了 9 個測試（觸發／不觸發／同 prompt 換 seed／佇列忙碌時跳過／不會跟
  非 Z-Image workflow 搞混），`check.ps1` 全過（916 passed）。實機冷開 ComfyUI，第一張
  212 秒（完成後 69°C、3595 MB 常駐）；換成不同 prompt 的第二張，log 印出
  `[comfyui] Z-Image 換了 prompt...`（確認真的觸發了），124 秒完成（75°C、3704 MB），
  兩張都沒有 OOM。

### 補上測試、CI 與例外邊界；安全關鍵邏輯去重 `67cda27`..`f11e915`

- **問題**：功能面相當完整，工程面幾乎是裸的。**零自動化測試**，唯一的 CI 只 build worker
  image。好幾個「壞掉才會知道」的隱性契約全靠人眼複查（CLAUDE.md 的「驗證慣例」就是在描述
  這件事）：GUI handler 的參數數量 vs 按鈕 `inputs` 長度、workflow JSON 的 node id ↔
  class_type、fp8 變體規則、年齡／內容安全負面詞。
- **一個真 bug**：`image_api._run_job` 用 `except Exception` 接 `gen_custom()`，但
  `generate_character` 有 18 處用 `raise SystemExit` 表示參數錯誤，而 `SystemExit` 是
  `BaseException`——接不到。驗證失敗會靜靜殺掉 worker thread，job 永遠停在 `pending`，
  輪詢端永遠等不到錯誤。`gui.py` 有同樣的洞（只有 AnimateDiff 那條有 inline try）。
  `worker/handler.py` 早就用 `except BaseException`，正是因為踩過。
  現在 library 層改用 `UsageError`（一般 `Exception`），只有 CLI 的 `__main__` 轉回
  `SystemExit`——實測 stderr 與 exit code 逐位元組不變。
- **安全關鍵邏輯去重**：`gen_video_animatediff` 自己重抄了一份負面詞組裝，而
  `_build_prompt_and_negative` 的 docstring 明講它存在就是為了避免這件事。改成呼叫共用
  builder，但**不能天真地直接換**：AnimateDiff 的 checkpoint key（`sd15_base`）不在
  `SD15_CHECKPOINTS` 裡，直接轉會默默失去 SD1.5 gender weight，所以加了 keyword-only 的
  `gender_weight`。cfg 下限也從兩份副本收斂成 `SAFETY_MIN_CFG`，由 `enforce_min_cfg()` 在
  `_submit_and_wait` 統一套用（11 個模板現有 cfg 全部 ≥ 2.0，今天不改變任何輸出）。
- **測試自己的漏洞**：原本所有年齡安全斷言都是 `AGE_SAFETY_NEGATIVE in negative`，這是
  自我指涉的。**實測：把那個常數清空，628 個測試全部照樣通過**（`"" in 任何字串` 恆真）。
  現在常數的內容本身也被釘住。
- **韌性**：client 原本沒有任何重試，一次暫時性的 `ConnectionError` 就會殺掉一個可能已經跑
  了好幾分鐘、而且伺服器端還在正常算的工作。現在依「重送會不會多花一次生成」分流：冪等的
  讀取（`GET /history`、`/view`、`overwrite=true` 上傳）退避重試 2/4/8/16 秒、連續失敗
  120 秒才放棄；**`POST /prompt` 不冪等**，只在連線被拒／ConnectTimeout（請求證實沒送出）
  時重送，`ReadTimeout` 直接報錯並要使用者先看佇列。逾時與 Ctrl-C 會把孤兒 prompt 從佇列
  移除，只有在 `GET /queue` 證實是自己的 prompt 在跑時才 `/interrupt`（API 形狀是讀
  ComfyUI 原始碼確認的，不是猜的）。
- **併發**：`_submit_and_wait` 全段在 module RLock 內（那段正是非可重入的 module globals
  會變動、以及 LCM 模式切換檢查會 race 的區間）；GUI 五個生成按鈕共用
  `concurrency_id="gpu"`（Gradio 的 `concurrency_limit=1` 是**每個 listener** 各算的，
  兩個分頁本來會同時打同一張卡）。
- **數字**：654 → 903 個離線測試，整套 2.3 秒，不需要 GPU、不需要 ComfyUI、不需要模型權重、
  不需要 torch。CI 在 ubuntu + windows 兩個 runner 上跑同一套，worker image 的 build 有
  `needs: checks` 擋著。graph surgery 那組做過 mutation check：把 `_rewire` 改成 no-op，
  278 個組合中 232 個變紅。
- **注意**：dev 工具（pytest/ruff）裝在**獨立的 `.venv-dev`**，不要裝進 `ComfyUI\.venv`——
  `worker/Dockerfile` 是從 `comfyui-requirements.lock.txt` 安裝的，而那個 lock 是
  `uv pip freeze` 產生的。
- **文件**：README「改程式之前：跑一次檢查」、CLAUDE.md「驗證慣例」


### SDXL / Pony 模型可切換完整版 / 量化版 fp8（選用） `707cb23`

- **問題**：8 GB VRAM + PCIe gen3 x1，模型搬移是主要瓶頸；高清流程連系統 RAM 都會吃緊。
- **做法**：`training/quantize_models.py` 在本機把 4 個 SDXL / Pony checkpoint 轉成 ComfyUI
  的每層量化格式（只量化 transformer 的 Linear 層，約 86% UNet 參數），存成
  `<原檔名>.fp8q.safetensors`，原檔只讀不改。單純 cast 成 fp8 在 Turing 沒用，會被轉回
  fp16；每層量化格式才會被認成 mixed precision 而保持 fp8。改寫點只有
  `comfyui_client._submit_and_wait` 一處，優先序 `--variant` > `MODEL_VARIANT` > 設定檔 >
  完整版。GUI 有每個模型的切換面板，RunPod worker 固定用完整版。
- **實測**（cyberrealistic_pony，同 seed）：主模型上 GPU 4896 → 2791 MB；純文字生圖每張
  36-40 → 40-42 秒；GUI 預設高清每張 167 / 272 / 530 → 145 / 127 / 125 秒（完整版變慢是
  因為記憶體吃緊開始用分頁檔）；高清時 ComfyUI 記憶體 15.5-15.8 → 12.3-13.2 GB；身分
  相似度不變。juggernaut 兩版溫度相近時，取樣每步慢約 15%、整張時間一樣。
- **注意**：骨架姿勢（ControlNet control-lora）一律改用完整版，強制量化版會出全黑圖。
  同一個 seed 在兩版產出的圖不完全一樣（SSIM 0.76-0.95），要重現舊圖請用原本的版本。
- **文件**：README「3c. 量化版模型」、GUI「模型版本」、CLI「量化版模型」

## 2026-09-11

### Z-Image Turbo 純文字生圖 `950e7e7`

- **做法**：Z-Image Turbo（Tongyi-MAI，6B DiT，Apache 2.0）加進 CLI `--checkpoint
  z_image_turbo` 和 GUI 選單。刻意不放進 `CHECKPOINTS`：它要載三個檔案，而其他程式都把
  `CHECKPOINTS` 的值當成單一 checkpoint 檔。
- **實測**（1024²）：載入後 cfg 1 約 75 秒、cfg 2 約 155-195 秒；啟動後第一張要 5-7 分鐘，
  每次換 prompt 都要換 5.4 GB 的文字編碼器和 5.9 GB 的擴散模型。跟 juggernaut 對比，寫實度
  和手部較好、男性角色每張都是男生、看得懂中文也寫得出中文（杯子上的「早安」四張都對，
  SDXL 則畫成沒有人的茶杯）。
- **限制**：只有純文字生圖。沒有 IP-Adapter / FaceID、ControlNet、FaceDetailer 版本，所以
  anchor、姿勢、精修、HQ、GIF 都會拒絕並說明原因。用 CFG 2.0（下限 1.5）而不是官方範本的
  1.0，因為 cfg 1.0 時 ComfyUI 會跳過負面詞，年齡安全負面詞會靜默失效。
- **文件**：README「Z-Image Turbo」

### SD1.5 性別字加權重，男性角色不再被畫成女生 `09f346e`

- **問題**：SD1.5 Realistic Vision 在角色描述比較柔和時（「soft face」等），常把男性角色
  畫成女生；prompt 裡單一個「man」壓不過那些詞。
- **實測**（SD1.5 靜態圖、5 個男性角色 × 4 個 seed，用 InsightFace 判性別）：原本 10/20；
  負面詞加「woman, female」10/20（負面詞已有約 98 個 token，加了幾乎沒作用）；正向加
  「male, man, masculine」14/20；**`(man:1.3)` 19/20**。女性角色四種寫法都是 20/20。
- **做法**：SD1.5 的兩條路徑（AnimateDiff 影片、`gen_custom` 選 SD1.5）把性別字寫成
  `(man:1.3)` / `(woman:1.3)`（`SD15_GENDER_WEIGHT`）。SDXL 的 anchor / variations / 資料集
  prompt 不動，既有 seed 產出不變。影片驗證：jungi 男性影格 7/16 → 16/16、相似度
  0.581 → 0.691。
- **注意**：AnimateLCM（cfg 2）救不回來，jungi 加不加權重都還是女生；男性角色不建議用 LCM。
- **順帶修掉**：`generate_character.py` 還有 6 處用 `os.replace` 搬檔，輸出資料夾在別的
  磁碟時會噴 WinError 17，全部改用 `shutil.move`。
- **文件**：README「男性角色被畫成女生（SD1.5）」

### 切換 LCM / 一般取樣會產生雜訊的 bug `5bb12a3`

- **問題**：同一個 ComfyUI session 裡混用 LCM 和一般取樣，會生出純色雜訊。ComfyUI 啟動後
  先跑的那個模式正常，之後每次跑另一個模式都是雜訊。
- **實測**（taeoh、seed 6001、512 基底，動作量正常是 3-8）：LCM 先跑正常 4.48，接著 v2 是
  雜訊 114；`POST /free` 後 v2 正常 3.57，接著 LCM 雜訊 76。
- **做法**：偵測到模式切換就先 `/free` 重置（會丟掉快取的節點輸出）。離線和實機都驗過：
  LCM → v2 → LCM → SD1.5 靜態圖 → v2 全部正常，且剛好在三次切換時各重置一次。
- **文件**：README「2d. AnimateLCM 快速模式」

### AnimateDiff 臉部變形修正 `4021552`

- **原因一**：臉部精修從來沒有真的放大臉部。`crop_factor` 3.0 讓裁切區塊比整張影格還大，
  `max_size` 768 又把放大倍率壓回 1.0，等於拿整張影格重跑 20 步。改成裁臉框的 1.5 倍、
  `guide_size` 512、12 步、denoise 0.45，並補上漏傳的 `noise_mask_feather`。
- **原因二**：負面詞裡有 `symmetrical face`，等於把模型推向雙眼不對稱、下巴歪斜。影片路線
  改用拿掉這個詞的 `VIDEO_REALISTIC_NEGATIVE`，靜態圖不變。
- **實測**：修正前相似度反而最高（0.685），但逐幀拼圖看得出臉頰浮腫、嘟嘴——InsightFace
  只判「是不是同一個人」，對這種形變不敏感，所以分數只用來確認沒換臉。修正後臉頰和嘴型
  正常、動作量完全保留（5.08）。加強 FaceID 或調低 motion scale 會讓臉更定住，但主要是
  因為畫面不動了，所以維持預設。
- **新工具**：`training/face_similarity.py`，逐幀算臉部相似度並輸出標了分數的臉部拼圖，
  生成時加 `--face-report` 會自動跑。
- **文件**：README「臉部變形排查」

## 2026-09-10

### AnimateDiff 影片輸出優化 + SadTalker 對嘴影片 `ad1f664`

- **做法**：原本只有 512×512、16 幀的 vp9 webm。改成分階段輸出，每段關掉就從工作流程移除：
  1.5 倍高清第二段（面積上限 768²，仍經過 motion module）、影片臉部精修、逐幀 ESRGAN 放大
  到長邊 1024、選用 RIFE 補幀、h264 mp4 輸出。預設 checkpoint 從純 SD1.5 換成已安裝的
  Realistic Vision。
- **實測**（seed 6001、16 幀）：不高清不精修不放大 69 秒 @512²；預設高清 20 步 693 秒；
  高清 10 步 526 秒、峰值 6.5 GB VRAM（跟 20 步的平均影格差只有 1.41/255，所以 10 步成為
  預設）；LCM 8 步 480 秒（高清、VAE、ESRGAN 的時間不會縮，所以省得有限）；RIFE ×4 @16fps
  得到 61 幀 3.8 秒（RIFE 產出是 15×m+1 幀，不是 16×m）。
- **新增**：`training/talking_head.py` 把 SadTalker 包成 `talk` 指令和 GUI 區塊，用
  subprocess 呼叫 SadTalker 自己的 Python 3.10 venv（不 import，也不需要 ComfyUI 在跑）。
- **文件**：README「AnimateDiff 動態影片」「會講話的嘴型影片（SadTalker）」

## 更早（一行摘要）

| 日期 | commit | 內容 |
|---|---|---|
| 2026-09-09 | `b6fec49` | GUI 需要時自動啟動 ComfyUI，不必事先開好 |
| 2026-09-09 | `2358e16` | 骨架庫在各 checkpoint 的實測，並更正一個寫錯的說法 |
| 2026-09-09 | `db9cea0` | 擋掉負面詞裡重複的主體詞 |
| 2026-09-09 | `d6f3d33` | 會把骨架裁掉的畫布尺寸直接拒絕 |
| 2026-09-09 | `97cc69d` | 重建 6 張在膝蓋被裁掉的骨架的腿部 |
| 2026-09-09 | `853ba36` | 新增 14 個純骨架庫姿勢（跪、蹲、斜躺、騰空） |
| 2026-09-09 | `c8e5274` | ControlNet OpenPose 骨架庫，用來壓住 checkpoint 壓不住的姿勢 |
| 2026-09-09 | `5ccc794` | 姿勢／角度標籤參考包產生器與實測結果 |
| 2026-09-08 | `448151d` | 各 checkpoint 的 prompt 語法實測與 prompt 食譜 |
| 2026-09-08 | `99622cc` | HQ 兩段式流程、RunPod LoRA 訓練、RunPod serverless worker |
| 2026-08-23 | `d4a9e6a` | web：provider-types 與 replicate / runpod 產生器骨架（WIP） |
| 2026-08-23 | `09fcb23` | 把既有的 web/ Amplify Gen 2 骨架納入版控 |
| 2026-08-21 | `c2aa8a5` | 初始 commit：ComfyUI 生成流程、GUI、訓練設定 |
