# 用 Open WebUI 操作短劇概念片流程

Open WebUI 的聊天模型透過一個本機的工具伺服器（`training/concept_api.py`）操作整條流程：
看故事與 Scene、生候選圖、記錄你的審核、送 H3 影片、接成片。

```
Open WebUI（Docker，localhost:3000）──工具呼叫──▶ concept_api（主機，127.0.0.1:7863）
                                                    ├─ 本機 ComfyUI ＋ Z-Image（候選圖）
                                                    ├─ MiniMax H3 skill（Colab，影片）
                                                    └─ ffmpeg（成片）
```

## 一、啟動工具伺服器

```
ComfyUI\.venv\Scripts\python.exe training\concept_api.py
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
   （有設 `CONCEPT_API_TOKEN` 就選 Bearer 並填入）。儲存後應該看到 11 個工具。
3. 模型 `qwen3.5:9b` 的進階參數：Function Calling 設成 Native（原生）。
4. 建議在「工作區 → 模型」建一個「短劇概念片助理」：基礎模型 `qwen3.5:9b`、勾選上面的工具伺服器，系統提示：

   ```
   你是短劇概念片的製作助理，用繁體中文回答，透過工具操作流程。
   規則：
   1. 不替使用者選圖或核准。只有使用者明確說出「Scene N 核准第 M 張」或「Scene N 重生」時才呼叫 review_scene。
   2. generate_candidates 與 make_videos 要先不帶 confirm 呼叫一次，把預估（張數、分鐘、CU、需要的人工動作）告訴使用者，
      使用者說好之後才帶 confirm=true 再呼叫。
   3. 工作排入後用 get_job 查進度；needs_user_action 有內容時，原文告訴使用者要做什麼。
   4. 給使用者看候選圖時，直接貼出 list_scenes 回傳的 markdown。
   5. 生候選圖時先只生每個 Scene 的第 1 張（預設值）；使用者看過、指名要第二張的 Scene 才用 candidates=[2] 補。
   6. 工具回傳錯誤時，照實轉述，不要自己猜原因。
   ```

## 三、工具

| 工具 | 作用 | 會不會花資源 |
| --- | --- | --- |
| `list_stories` | 五個故事的狀態、圖片數、影片數 | 不會 |
| `list_scenes` | 一個故事的 7 個 Scene：劇情、狀態、候選圖、要核對的畫面文字 | 不會 |
| `generate_candidates` | 排入本機生圖。預設只生每個 Scene 的候選 1（7 張）；`candidates: [2]` 補第二張，`[3, 4]` 只給判定重生的 Scene | **本機 GPU**，每張約 2.3 分鐘 |
| `review_scene` | 記錄你的決定：核准第幾張（變成 `master.png`）或判定重生 | 不會 |
| `set_video_prompt` | 存一個 Scene 的 H3 鏡頭描述（英文） | 不會 |
| `make_videos` | 把**已核准**的 Scene 送 H3，每個 session 最多 4 支 | **Colab CU**，要 Docker、要按 Drive 同意 |
| `make_cut` | 7 支片段接成一支，音量調平到約 −14 LUFS，可加結尾字幕 | CPU，約 1 分鐘 |
| `get_job`／`list_jobs` | 進度、結果、是否在等你 | 不會 |
| `cancel_job` | 取消（見下） | 不會 |
| `consent_done` | 你按完 Drive 同意後，通知 H3 繼續 | 不會 |

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
- 生圖開始前顯示卡要低於 55°C、VRAM 用量低於 2.5 GB，等 5 分鐘還不行就失敗並說明原因，不會硬跑。
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

- **「是你決定的」驗不了。** 伺服器看不到是誰下的決定，只能靠工具說明與系統提示約束模型，並在 `review.json` 記
  `decided_via: open-webui`。核准之後請自己看一眼 `review_sheet.html` 或 `list_scenes` 的結果。
- 9B 的模型可能選錯工具或漏參數；工具回傳的錯誤訊息是中文，會說明缺什麼。
- H3 仍然需要你在場：Docker Desktop 要開著、每個 session 要按一次 Drive 同意。
