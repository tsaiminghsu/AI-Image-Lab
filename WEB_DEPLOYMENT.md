# 網頁部署規劃：AWS Amplify + Replicate/RunPod

規劃文件，還沒有對應的程式碼實作。目標架構：前端放 AWS Amplify，實際 GPU
運算外包給 Replicate 或 RunPod（見 [CLOUD_GPU.md](CLOUD_GPU.md) 了解這兩個
服務本身的計費/使用方式）。**現階段只給自己/內部用**，先不做會員系統。

## 目標架構

```
瀏覽器
  │
  ▼
AWS Amplify（前端：靜態網站或 React/Next.js，取代現在的 gui.py 網頁）
  │  呼叫後端 API（帶一組共用密鑰驗證身分，見下面「存取控制」）
  ▼
API Gateway + Lambda（後端：取代 image_api.py，邏輯大致相同，但要改成
無伺服器可用的形狀，見下面「為什麼要走非同步」）
  │
  ├──▶ Replicate API（webhook 回呼結果）
  │
  └──▶ RunPod（Pod 常駐 ComfyUI，或 RunPod Serverless）
```

## 為什麼不能把 `gui.py` 直接搬上去

`gui.py` 是 Gradio 寫的——Gradio 本身是一個常駐的 Python process，自己監聽
port、自己 serve 網頁，設計上就是「本機跑一個服務、瀏覽器連過去」，不是
「靜態檔案 + API」這種形狀。AWS Amplify 是給靜態網站/SPA（React、
Next.js 等）+ Lambda 後端用的平台，沒有地方可以放一個常駐的 Gradio
process。真的要上 Amplify，前端得重寫成一般網頁（HTML/JS 或 React），透過
API 呼叫後端——這是這份文件裡最大的一塊工作，不是小改動。

`image_api.py` 的方向是對的（純 HTTP API、不管前端長什麼樣），但它現在的
實作方式（`ThreadPoolExecutor` + 存在記憶體裡的 `_jobs` dict，見
[training/image_api.py:38-41](training/image_api.py#L38-L41)）假設自己是
一個常駐 process。Lambda 每次呼叫都是全新、無狀態的執行環境，呼叫跟呼叫
之間不會共用記憶體，所以這個 `_jobs` dict 的做法在 Lambda 上完全不能用，
job 狀態要改存到外部（DynamoDB 之類），下面會展開。

## 為什麼一定要走非同步 job-queue（不是我加的限制，是平台的硬限制）

這點很重要，決定了整個後端要怎麼設計：

- **API Gateway 的 integration timeout 預設 29 秒**（REST API 可以申請提高，
  但不是預設值；HTTP API 固定 30 秒上限）
- **Lambda 單次執行最長 15 分鐘**，超過會被平台強制砍斷，沒有例外

而這個專案實際的生成時間，從 `comfyui_client.py` 現有的 timeout 常數就看得
出來跨度多大：一般圖片 20-60 秒沒問題，但 ControlNet/FaceDetailer 第一次
用要下載模型可以到 900 秒，AnimateDiff 影片第一次執行 `POLL_TIMEOUT_SECONDS_ANIMATEDIFF`
設到 1800 秒（30 分鐘）——這已經超過 Lambda 15 分鐘的硬上限，**不可能**
用「前端發一個請求、後端同步算完才回應」這種寫法。

現有 `image_api.py` 的 `/generate` 立刻回 `job_id`、`/generate/{job_id}`
另外輪詢結果的設計，剛好就是對的模式——搬到 Lambda 上時這個形狀要保留，
只是內部實作要換掉（記憶體 dict → DynamoDB，背景 thread → 另一個 Lambda
或 Replicate/RunPod 自己的 webhook）。

## 後端：Amplify Gen 2 Functions + API Gateway

Amplify Gen 2 支援用 `amplify/functions/` 定義 Lambda function，並用 CDK
在 `amplify/backend.ts` 裡把它們掛到 API Gateway 的路由上，前端透過
Amplify 的 REST/HTTP API client 呼叫，這條路跟現在 Vercel/Next.js 常見的
"API routes" 概念類似。Job 狀態的持久化可以直接用 Amplify Data（背後是
AppSync + DynamoDB），取代 `image_api.py` 現在那個 `_jobs` dict——順便還能
用 AppSync 的即時訂閱讓前端不用真的輪詢，等 job 狀態變更會自動推送，但這是
之後的優化，第一版直接照搬現在 `/generate/{job_id}` 的輪詢方式最省事。

## Replicate 這條路：用 webhook，不要自己輪詢

Replicate 的 prediction API 支援建立時帶一個 `webhook` URL，指定
`webhook_events_filter: ["completed"]`，跑完後 Replicate 會自己 POST 通知
你——比自己寫輪詢邏輯簡單很多，也更省 Lambda 呼叫次數（不用每幾秒醒來檢查
一次）。對應到這個架構：

1. 前端呼叫「送出生成」的 Lambda A → Lambda A 呼叫 Replicate 建立 prediction，
   `webhook` 指向 Lambda B 的 URL，立刻把 `job_id`（用我們自己存 DynamoDB 的
   id，關聯到 Replicate 回傳的 prediction id）回給前端
2. Replicate 跑完，POST 到 Lambda B → Lambda B 把 DynamoDB 裡對應的 job
   狀態更新成完成、記下輸出網址
3. 前端輪詢（或訂閱）自己後端的 job 狀態，不直接碰 Replicate

## RunPod 這條路：Pod 常駐 vs RunPod Serverless

`CLOUD_GPU.md` 記錄的是「Pod 常駐 ComfyUI」模式（`comfyui_client.py` 直接
換個網址就能打）——但那個模式是**用小時計費，不管有沒有在跑都要付錢**，跟
「內部零星使用」這個場景（現在的使用情境）不太搭。RunPod 另外有
**Serverless Endpoint** 模式：把 ComfyUI 包成 Docker image 部署成
serverless worker，沒人呼叫就縮到 0、按呼叫計費，且 RunPod 自己的 API 也
支援 webhook/輪詢兩種取結果方式，跟上面 Replicate 那條路的架構形狀幾乎一樣。

**這個部署情境下建議用 RunPod Serverless，不是 Pod 常駐**——Pod 常駐比較
適合像本專案現在這種「一次跑一大批 dataset 生成、跑完就關」的批次工作，跟
「網頁上零星生成請求」的用量模式不一樣。真的要接 Pod 常駐模式，Lambda A
一樣只做「送出 ComfyUI `/prompt` 請求、立刻拿到 `prompt_id`」這個快速動作
（`comfyui_client.py` 的 `_submit_and_wait` 現在是同步輪詢到完成才回傳，這
部分要拆開，提交跟等待分成兩個獨立步驟），完成的檢查另外交給排程 Lambda
（EventBridge 定時觸發）或前端輪詢驅動。

## 存取控制（現階段：內部使用）

先不做 Cognito/會員系統。最簡單的做法：API Gateway 的 **API Key** 功能，或
自己在後端檢查一個固定的 secret header（例如 `x-internal-key`），前端把這
組密鑰內嵌在呼叫裡。這組密鑰**不要**硬編碼進前端原始碼裡明著提交到 git——
放在 Amplify 的環境變數，build 時注入。之後如果真的要開放給別人用，這一步
換成 AWS Cognito（Amplify 官方支援度最好，跟 Amplify Data/API 的權限系統
原生整合），到時候再處理，不用現在就做。

## 現有程式碼：能重用 vs 要重寫

| 部分 | 這個架構下 |
|---|---|
| `generate_character.py` 的角色定義、prompt 組裝邏輯 | **可重用**——跟後端跑在哪裡無關，是純邏輯 |
| `comfyui_client.py` 打 RunPod Pod 的 HTTP 呼叫方式 | **可重用，但要拆開**——`_submit_and_wait` 現在是「送出+同步等到完成」綁在一起，要拆成「送出」跟「查詢狀態」兩個獨立函式 |
| `comfyui_client.py` 的 workflow JSON 組裝（IP-Adapter/ControlNet/FaceDetailer 節點） | **只對 RunPod Pod 這條路有用**——Replicate 是別人包好的固定模型，不吃 ComfyUI 的 workflow JSON 格式，這塊邏輯用不上 |
| `image_api.py` 的 request/response 形狀（`GenerateRequest`、job_id、輪詢） | **設計可重用，實作要重寫**——介面照抄，內部從「記憶體 dict + thread」換成「DynamoDB + Lambda」 |
| `gui.py`（Gradio） | **不能重用**，前端要整個重寫 |
| `caption_image.py`（BLIP 圖片轉 prompt） | 目前跑在 CPU、模型約 1GB——搬進 Lambda 要考慮 Lambda 的 container image 大小限制跟冷啟動時間，值不值得搬，還是先只保留在本機工具鏈裡，之後再評估 |

## 建議分階段順序

1. **Phase 1**：後端 API 骨架（Amplify Functions + API Gateway + DynamoDB
   存 job 狀態），先只接 Replicate（webhook 模式），因為不用管 ComfyUI/
   RunPod 那條路的「提交/查詢拆開」重構，最快能端到端跑通
2. **Phase 2**：接 RunPod Serverless（形狀跟 Replicate 那條路相似，複用
   Phase 1 的 job 狀態/webhook 骨架）
3. **Phase 3**：前端搬上 Amplify，接 Phase 1/2 做好的 API
4. **Phase 4**（要開放給別人用才需要）：存取控制從共用密鑰換成 Cognito
