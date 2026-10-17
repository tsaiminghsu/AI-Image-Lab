# 用 Open WebUI 操作短劇概念片流程

Open WebUI 的聊天模型透過一個本機的工具伺服器（`training/concept_api.py`）操作整條流程：
看故事與 Scene、生候選圖、記錄你的審核、送 H3 影片、接成片。

```
Open WebUI（Docker，localhost:3000）──工具呼叫──▶ concept_api（主機，127.0.0.1:7863）
                                                    ├─ 本機 ComfyUI ＋ Z-Image（候選圖）
                                                    ├─ MiniMax H3 skill（Colab，影片）
                                                    └─ ffmpeg（成片）
```

## 零、一鍵啟動

雙擊專案根目錄的 `start_concept_webui.bat`（或跑 `training\start_concept_webui.ps1`）。它依序：開 Docker Desktop（沒開才開）→
啟動 `open-webui` 容器 → 啟動工具伺服器（背景執行、不佔視窗、記錄在 `outputs\logs\concept_api.log`）→ 檢查容器連得到
工具伺服器 → 確認 Ollama 服務在跑（沒開就 `ollama serve`，只開服務、不載入模型）→ 開瀏覽器到 `http://localhost:3000`。
已經在跑的部分會沿用。**不會啟動 ComfyUI、不會動用顯示卡**。

Open WebUI 裡的工具伺服器註冊存在容器的 volume，**只要註冊一次**，之後只需要這個啟動指令。
關閉工具伺服器：`training\stop_concept_webui.ps1`（有工作沒做完會拒絕，`-Force` 才硬關）。

**埠與模型資料夾**：Windows 會動態保留一段 TCP 埠（`netsh interface ipv4 show excludedportrange protocol=tcp`），重開機後會移動，
2026-10-07 保留範圍剛好蓋住 Ollama 的 11434（`ollama serve` 報 `bind: ... forbidden by its access permissions`），
ComfyUI 的 8188 也是。所以 ComfyUI 改用 9188（`COMFYUI_URL`），Ollama 由腳本固定開在 **21434**（`CONCEPT_OLLAMA_PORT` 可改），
工具伺服器的 `OLLAMA_URL` 也指向它。**Open WebUI 的 Ollama 連線要改一次**：管理員面板 → 設定 → 連線 → Ollama API，
URL 填 `http://host.docker.internal:21434`。另外環境變數 `OLLAMA_MODELS` 指向的資料夾可能是空的（這台是 `D:\AI\Models\Ollama`，
`qwen3.5:9b` 實際在 `D:\Ollama\Models`），腳本會自動挑第一個真的有模型的資料夾。

**上下文長度**：Ollama 在 8 GB 的卡上預設只開 4096 個 token，但 Open WebUI 每則訊息都會附上 13 個工具的說明和系統提示
（約 6,200 token），會報 `exceed_context_size_error`（`request (6198 tokens) exceeds the available context size (4096 tokens)`）。
腳本因此用 `OLLAMA_CONTEXT_LENGTH=32768` 啟動 Ollama（`CONCEPT_OLLAMA_CTX` 可改）。16384 在長對話裡會被塞滿（工具定義約 6,000 token，加上一次 `list_scenes` 約 3,800 token），塞滿時模型寫工具參數寫到一半被截斷，Open WebUI 會顯示
`llama-server returned invalid tool call arguments ... unexpected end of JSON input`。實測 32768 只比 16384 多佔約 0.6 GB
（Q4_K_M 6.15 → 6.72 GB、Q6_K 7.62 → 8.19 GB，16 GB 顯示卡）。**Open WebUI 的模型進階參數 `num_ctx (Ollama)` 也要改成 32768**，
它每次都會明確送出自己的值，只改 Ollama 的預設不會生效。對話還是要一件事一個對話，太長就開新的。

## 一、啟動工具伺服器（手動）

```
$env:COMFYUI_URL="http://127.0.0.1:9188"; ComfyUI\.venv\Scripts\python.exe training\concept_api.py
```

- 只綁 `127.0.0.1:7863`。Open WebUI 的容器用 `http://host.docker.internal:7863` 連得到（2026-10-03 實測回 200）。
- 圖片與影片的網址是 `http://localhost:7863/files/...`，由你的瀏覽器直接開。
- 環境變數（都可以不設）：`CONCEPT_API_PORT`、`CONCEPT_API_PUBLIC_URL`、`CONCEPT_API_CORS`、
  `CONCEPT_API_TOKEN`（設了就要 Bearer token）、`CONCEPT_API_HOST`（不是 127.0.0.1 時一定要設 token，否則拒絕啟動）、
  `OLLAMA_URL`（預設 `http://127.0.0.1:11434`）。
- 伺服器重啟時，沒做完的工作會標成 `interrupted`，**不會自動重跑**。

## 二、在 Open WebUI 註冊（要用管理員帳號，這一步由你操作）

1. 管理員面板 → 設定 → Integrations（整合）→ Tool Servers → 新增連線。
   （Open WebUI 0.11.4 的後端設定是 `tool_server.connections`；選單的中文名稱我沒有在畫面上核對過，找不到時搜尋
   「OpenAPI」或「Tool Server」。）
2. 類型 OpenAPI；URL `http://host.docker.internal:7863`；路徑 `openapi.json`；驗證選「無」
   （有設 `CONCEPT_API_TOKEN` 就選 Bearer 並填入）。儲存後應該看到 13 個工具。
3. 模型 `qwen3.5:9b` 的進階參數：Function Calling 設成 Native（原生）。
4. 建議在「工作區 → 模型」建一個「短劇概念片助理」：基礎模型 `qwen3.5:9b`、勾選上面的工具伺服器，系統提示：

   ```
   你是短劇概念片的製作助理，用繁體中文回答，透過工具操作流程。
   規則：
   1. 不替使用者選圖或核准。只有使用者明確說出「Scene N 核准第 M 張」或「Scene N 重生」時才呼叫 review_scene，
      並把使用者那句話原文放進 user_quote。「看起來不錯」「第 2 張好像比較好看」這類話不是決定，要反問使用者。
   2. generate_candidates 與 make_videos 要先不帶 confirm 呼叫一次，把預估（張數、分鐘、CU、需要的人工動作）告訴使用者，
      使用者說好之後才帶 confirm=true 再呼叫。
   3. 工作排入後用 get_job 查進度；needs_user_action 有內容時，原文告訴使用者要做什麼。
   4. 使用者想看圖片時（剛生完圖、或說「給我看」），呼叫 show_images：圖會直接顯示在聊天裡。看新生的圖用 newest_only=true。不要用文字描述圖片，也不要自己貼圖片連結。
   5. 生候選圖時先只生每個 Scene 的第 1 張（預設值）；使用者看過、指名要第二張的 Scene 才用 candidates=[2] 補。
   6. 工具回傳錯誤時，照實轉述，不要自己猜原因。
   7. 使用者想改某個 Scene 的畫面（動作、場景、光線、畫面文字、構圖）時，用 set_scene_visual，只改他說到的欄位，
      user_quote 放他的原話；動作、場景、光線、構圖要用英文寫。改完要說清楚：已有的候選圖不會變，
      要用新描述得到新圖，必須由使用者自己說「Scene N 重生」，不要替他決定。不要自己加使用者沒要求的改動。
   8. generate_candidates 預設用 z_image。只有使用者明說要用 Qwen（Qwen-Image 2.1）時才帶 image_model=qwen_image_2_1；
      這時預估裡有 license_warning，要把那句話原文告訴使用者（只能內部測試，不能商用）。
   9. generate_candidates 的預估裡如果寫「Google Colab 雲端 GPU」，表示本機顯示卡記憶體不夠；要把原因、預估時間、
      CU 費用與要使用者做的事（Docker 開著、按 Drive 同意）原文告訴使用者，使用者同意才帶 confirm=true。預估裡如果有 resolution_reduced，表示本機顯示卡被別的程式占著、解析度被降低，也要把 resolution_warning 原文告訴使用者（圖會比較小，H3 首幀畫質可能較差）。
      不要自己把 backend 改成 colab。
   ```

## 三、工具

| 工具 | 作用 | 會不會花資源 |
| --- | --- | --- |
| `list_stories` | 五個故事的狀態、圖片數、影片數 | 不會 |
| `list_scenes` | 一個故事的 7 個 Scene：劇情、狀態、候選圖、要核對的畫面文字 | 不會 |
| `show_images` | **把候選圖直接顯示在聊天裡**（圖片畫廊，標註 Scene、候選編號、模型、授權警告）。可指定 scene，或 newest_only=true 只看每個 Scene 最新的一張 | 不會 |
| `generate_candidates` | 排入生圖，**本機優先、記憶體不夠才改 Colab**（見六之二）。預設只生每個 Scene 的候選 1（7 張）；`candidates: [2]` 補第二張，`[3, 4]` 只給判定重生的 Scene；`image_model` 選模型，`backend` 可指定 local／colab | **本機 GPU**（Z-Image bf16 每張約 30 秒、Qwen 約 2 分鐘）；改 Colab 時花 **Colab CU** |
| `review_scene` | 記錄你的決定：核准第幾張（變成 `master.png`）或判定重生。**必須附上你的原話**（`user_quote`） | 不會 |
| `set_video_prompt` | 存一個 Scene 的 H3 鏡頭描述（英文） | 不會 |
| `set_scene_visual` | 用你的話改一個 Scene 的**圖片描述**：動作、場景、光線、畫面文字、額外負面詞、構圖。不會畫圖，只影響之後生的圖 | 不會 |
| `make_videos` | 把**已核准**的 Scene 送 H3，每個 session 最多 4 支 | **Colab CU**，要 Docker、要按 Drive 同意 |
| `make_cut` | 7 支片段接成一支，音量調平到約 −14 LUFS，可加結尾字幕 | CPU，約 1 分鐘 |
| `get_job`／`list_jobs` | 進度、結果、是否在等你 | 不會 |
| `cancel_job` | 取消（見下） | 不會 |
| `consent_done` | 你按完 Drive 同意後，通知 H3 或 Colab 生圖繼續 | 不會 |

## 四、要知道的行為

- **一批先生每個 Scene 一張**：顯示卡大約第 3 張之後就升到 83–84°C、每張慢約 30 秒，批次也可能中途卡死，所以預設一批
  只生候選 1（7 張，約 15 分鐘）。看過之後再對需要的 Scene 補候選 2；已核准的 Scene 不會再補。只有一張候選的 Scene 就可以審核。
  生圖順序是「所有 Scene 的候選 1，再所有 Scene 的候選 2」，批次中途停掉時每個 Scene 至少有一張。
- **圖片和短片分開**：圖在 `story_*/images/scene_NN/`，短片在 `story_*/videos/scene_NN/clip_vN.mp4`，成片在
  `story_*/videos/cut/`。重做的短片與成片是新的版本號，不覆蓋舊的；成片用每個 Scene 最新的那支。
- **兩段式確認**：`generate_candidates`、`make_videos` 不帶 `confirm=true` 只回預估，什麼都不會開始。
- **一次只跑一個工作**。有工作在跑時再排新的會被拒絕（409）。
- **聊天模型和生圖共用同一張 8 GB 顯示卡**。生圖開始前與每一張之前，伺服器會請 Ollama 卸載模型；你在生圖期間聊天會把模型
  載回來，下一張就會等它卸載（最多 2 分鐘，等不到這個工作就失敗，已生的圖保留）。**生圖期間盡量不要聊天**，
  查進度可以，但每查一次都會讓下一張多等一次卸載。
- 生圖開始前顯示卡要低於 55°C，而且卸載聊天模型後**至少還剩 1,500 MiB 可用**（這是跟預估同一個數字，`gpu_guard.START_MIN_FREE_MIB`；
  以前是「已用不超過 2,500 MiB」，那是 8 GB 舊卡的數字，會誤擋 16 GB 卡），等 5 分鐘還不行就失敗並說明原因，不會硬跑。
  等待期間聊天模型又被載回來會再卸載一次。
- **為什麼生完圖 VRAM 還是高**：生圖用的 ComfyUI 結束就會把記憶體還掉；之後高，是因為你接著聊天，聊天模型被載回（6～8 GB）。Ollama 預設閒置 5 分鐘才卸載，
  啟動腳本現在把它縮成 **2 分鐘**（`OLLAMA_KEEP_ALIVE`，`CONCEPT_OLLAMA_KEEP_ALIVE` 可改；代價是久沒聊天後再開口多等約 8～10 秒載入）。
- **孤兒模型程式**：Ollama 被強制結束或用不乾淨的方式重啟時，它載入模型用的子程式 `llama-server` 不會跟著結束，會繼續佔著聊天模型（實測 7.4 GB），
  而新的 Ollama 看不到它（`ollama ps` 說沒有載入任何模型），所以 VRAM 看起來「沒人用卻很高」。開工前的檢查（`gpu_guard.before_batch`）與啟動腳本都會自動清掉
  「父程式已經不在」的 `llama-server`；`job_status.bat` 的顯示卡區塊會列出誰佔著顯示卡，孤兒會標紅。重啟 Ollama 請用啟動腳本，不要直接強制結束。
- **每張圖開始前都會等顯示卡降到 60°C 以下**（工作狀態會顯示「等顯示卡降溫」）。這是為了讓顯示卡不要長時間待在
  降頻點，所以一批會比連續跑慢很多。工作結果裡的 `temperature` 有最高溫、在 83°C 以上的秒數與等待的總秒數。
- 伺服器自己用 `--disable-smart-memory` 啟動 ComfyUI，整批結束後關掉；如果 ComfyUI 本來就開著（別人開的），用完不會關它。
- **H3**：只送 `APPROVED` 的 Scene，首幀是 `master.png`；模型一律用下載（`H3_MODEL_SOURCE=download`）；
  開始前先檢查 Docker 與登入，沒過就直接失敗、不花 CU。每個 session 開始時工作會變成「等使用者」：
  請在瀏覽器按 Google Drive 同意，然後跟聊天模型說一聲，它會呼叫 `consent_done`。
- **取消**：排隊中的直接取消；生圖中的做完當前這張就停；H3 已經開始的 session 會把這個 session 的片段做完才停
  （中途殺掉可能讓 Colab 繼續計費）。
- 缺鏡頭描述的 Scene（故事 2–5 目前都沒有）送 H3 會被拒絕，先用 `set_video_prompt` 補上。故事 1 已經放進這次實際用過的描述。

## 五、這層保證不了的事

- **「是你決定的」只能部分驗證。** 實測 9B 模型（Q4_K_M 與 Q8_0）會把「Scene 3 看起來還不錯」「第 2 張好像比較好看」
  當成核准，還會自己編故事名稱，所以光靠系統提示不夠。`review_scene` 現在必須附 `user_quote`（你的原話），伺服器檢查：
  有明確的決定用語（核准／批准、重生／重新生成／重做／重跑）、沒有猶豫或比較或提問的字眼（好像、看起來、要不要、嗎、比較…）、
  提到的 Scene 與候選編號和呼叫的一致；不合就回 422，不會記錄。原話會寫進 `review.json` 的 `notes`，`decided_via: open-webui`。
  **這驗的是話，不是人**：模型仍可能編出一句符合格式的原話。核准之後請自己看一眼 `review_sheet.html` 或 `list_scenes` 的結果。
- 9B 的模型可能選錯工具或漏參數；工具回傳的錯誤訊息是中文，會說明缺什麼。
- H3 仍然需要你在場：Docker Desktop 要開著、每個 session 要按一次 Drive 同意。

## 六之一、選圖片模型（`generate_candidates` 的 `image_model`）

| 說法 | 用的模型 | 預估 | 說明 |
| --- | --- | --- | --- |
| 沒特別說（預設） | Z-Image Turbo | 每張約 40 秒（新顯卡實測暖機 36.5 秒） | 專案原本的模型 |
| 「用 Qwen 生」 | Qwen-Image 2.1（社群 GGUF Q6_K，25 步） | 每張約 2 分鐘（實測暖機 119 秒、首張 185 秒） | 中文畫面文字更準；**授權是 Qwen Research，只能內部測試，不能商用** |

- **安全規則不破例**：Qwen 官方設定是 cfg 1（負面詞不生效），這裡一律用基礎版、真 cfg 2.0，並加跟 Z-Image 同一套安全負面詞；
  Turbo 版是為 cfg 1 訓練的，沒有開放。測試釘住：每個故事每個 Scene 的 Qwen 請求都帶全部年齡安全詞、cfg 不低於 1.5，
  工作流過 `workflow_safety` 的檢查。
- **授權標記**：Qwen 產的每張圖，記錄檔有 `license: qwen-research (non-commercial)`、`image_model`、模型檔名；生圖預估和
  `list_scenes` 的每張候選圖都會帶 `license_warning`。程式擋不了你之後把圖拿去商用，那要你自己注意。
- 兩個模型共用同一個種子規則與提示詞組裝，只有畫圖的模型不同，所以同一個 Scene 可以直接比較。
- 一次只能選一個模型，同一批圖用同一個模型；先有 Z-Image 的候選 1、2，要補 Qwen 的就要走「Scene N 重生」再補候選 3、4。

## 六、在聊天裡改圖片細節（`set_scene_visual`）

圖片的提示詞不是你寫的，而是伺服器用每個 Scene 的規格組出來的。`set_scene_visual` 讓你在聊天裡改這些規格，例如：
「故事 4 Scene 2，男主要抓著被子，手機在床邊亮著藍光。」

- **可以改**：`action`（誰在做什麼）、`setting`（場景）、`lighting`（光線）、`on_screen_text`（畫面上的字，可以是中文）、
  `negative_extra`（這一場額外不要的東西）、`compositions`（C、D 的構圖）。**不能改**：角色設定、`solo`、整體風格。
- **要附你的原話**（`user_quote`），原話連同改前改後記在該故事資料夾的 `visual_edits.jsonl`，可以查。
- **只影響之後生的圖**：已經有的候選圖不會變。要用新描述得到新圖，要由你說「Scene N 重生」，再補候選 3、4。
- **拒絕的情況**：Scene 已核准、這個故事有工作在跑、要改的構圖那一張已經生出來、英文欄位寫了中文、欄位空白或太長（700 字）。
- **過濾**：所有文字都要通過 `h3_colab.screen_prompt`（兒童、未成年、露骨等詞，沒有繞過的參數），另外本工具自己擋「aged 15」
  「age 16」這類未滿 18 歲的年齡寫法（共用的過濾器擋不到這種寫法）。
## 六之二、本機優先：顯示卡被占著就降解析度，幾乎全滿才建議 Colab（`generate_candidates` 的 `backend`）

預設 `backend=auto`：一律先用這台電腦的顯示卡。生圖前會讀 `nvidia-smi`，扣掉聊天模型（Ollama，生圖前會卸載）佔的，算**其他程式佔了多少、還剩多少**。

**為什麼不是「不夠就上雲端」**：2026-10-17 用一支只佔顯示卡記憶體的小程式模擬「別的程式在用」，分別佔 4.4、10.1、14.3 GB，Z-Image 本機生圖**一次都沒失敗**，ComfyUI 會自動把一部分權重換到系統記憶體，只是變慢。

| 實測（Z-Image，從冷啟動算起） | 被占 4.4 GB | 被占 10.1 GB | 被占 14.3 GB |
| --- | --- | --- | --- |
| bf16 720×1280（全解析度） | 50.5 秒 | 62.7 秒 | 74.7 秒 |
| bf16 576×1024 | 22.3 秒 | 38.4 秒 | — |
| bf16 432×768 | 18.2 秒 | 32.3 秒 | — |
| int8 720×1280 | 40.5 秒 | 46.5 秒 | 52.6 秒 |
| int8 576×1024 | — | 28.3 秒 | 36.4 秒 |
| int8 432×768 | — | — | 28.4 秒 |
| 對照：沒人占的全解析度 | 48.5 秒（冷）／26.3 秒（暖機） | | |

所以記憶體少時，**先降解析度**（比較快），不是先上雲端：

| 卸載聊天模型後可用 | 本機用的尺寸 | 說明 |
| --- | --- | --- |
| ≥ 9,500 MiB | 專案畫布（720×1280） | 全解析度，沒有變慢 |
| 5,000～9,500 MiB | 80%（576×1024） | 預估會寫明降低解析度，附警告 |
| 1,500～5,000 MiB | 60%（432×768） | 同上 |
| < 1,500 MiB | 建議 Colab | 這個範圍**沒有實測過**，建議雲端，仍要你同意 |

- **降解析度的代價**：這批圖會比其他 Scene 小，H3 用它當首幀時畫質可能較差；預估裡有 `resolution_warning`，助理要原文告訴你。想要全解析度，就先關掉佔顯示卡的程式。每張圖的記錄檔有 `resolution_reduced_from: [720, 1280]` 與實際的 `width`／`height`，`show_images` 的畫廊也標「降低過解析度」。
- **只有 Z-Image 會降解析度**：Qwen-Image 2.1 沒有在被占用的情況下量過小尺寸，所以一律用全解析度，記憶體少也只是比較慢。
- **改 Colab 的條件**：可用少於 1,500 MiB（幾乎全滿），或你明說「用 Colab」。預估改寫「Google Colab 雲端 GPU（L4）」，附原因、預估分鐘、**CU 費用**（實測一個 session 約 0.24～0.40 CU）與要你做的事（Docker 開著、按一次 Drive 同意）。**花 CU 一律要你說好才開始**，助理不能自己改 backend。Qwen 沒有雲端版本：全滿時回 409。
- **Colab 開始之後**：先做免費檢查（`setup_colab.py preflight`）→ 匯出清單 → 送進 Colab 佇列 → 才啟動（這步才花 CU）→ 等圖 → 存回 `images/scene_NN/`（記錄 `backend: colab`）。Drive 同意時狀態變「等使用者」，你按完說一聲，助理呼叫 `consent_done`。`cancel_job` 會要求結束 Colab session，避免繼續計費。
- **本機途中失敗**：開始後若顯示卡被占住（`GpuBusy`）只會失敗並說明，**不會自動改走 Colab**，因為那會在你不知情時花 CU。

要強制本機：`backend=local`；要強制雲端：`backend=colab`（只對 Z-Image）。Colab 用官方 bf16 權重，同一個種子的畫面跟本機不會逐像素一樣。

## 六之三、H3 首幀太小的警告（`make_videos` 的預估）

H3 會先把首幀放大到約 1 MP（最近鄰放大，放大越多越軟），再以短邊 768（9:16 → 768×1376）生影片。專案的 720×1280 母圖只要放大約 1.04 倍；降解析度後的圖要放大更多：

| 母圖尺寸 | 放大倍數 | 預估裡的警告 |
| --- | --- | --- |
| 720×1280 | 約 1.04 倍 | 沒有 |
| 576×1024 | 約 1.3 倍 | 輕度：畫質可能差一點，在意就先用全解析度重生 |
| 432×768 | 約 1.74 倍 | **強烈**：臉和畫面文字很可能糊掉，建議先全解析度重生 |

警告線是 1.15 倍與 1.5 倍（`H3_UPSCALE_WARN`、`H3_UPSCALE_STRONG`），只警告、不擋；`make_videos` 的預估多一個 `first_frame_warnings`，助理要原文告訴你，你決定要先重生還是照樣送（會花 CU）。這兩個警告線是依放大倍數推的，實測（`outputs/h3_res_test`）出來後會依結果調整。