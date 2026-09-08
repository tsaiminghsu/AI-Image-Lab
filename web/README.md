# web/ — Amplify 後端（RunPod Serverless + Replicate 雙 provider）

對應 [WEB_DEPLOYMENT.md](../WEB_DEPLOYMENT.md)：非同步 job-queue、webhook 模式。
現在同時接兩條 GPU 路：

- **RunPod Serverless（預設）**：跑我們自建的 ComfyUI worker（見 repo 根目錄
  `worker/`），沿用本地那套 FaceID / 角色 LoRA / HQ 兩段式 workflow，安全負面詞
  是硬保證（worker 匯入 `generate_character.py`）。結果存進私有 S3 桶。
- **Replicate**：跑別人包好的模型，安全負面詞只能 best-effort（`shared/safety.ts`）。

`provider` 由請求 body 的 `provider` 欄位決定，沒帶就用 `DEFAULT_PROVIDER`（預設
`runpod`）。

**驗證狀態**：`npm install` + `npx tsc --noEmit` 對全部原始檔通過型別檢查。**沒有
實際 `npx ampx sandbox` 部署過**（這個環境沒有 AWS 帳號），CDK 接線型別正確不代表
部署一定順利。

## 結構

```
web/amplify/
├── backend.ts                    # CDK：DynamoDB + API Gateway + 3 個 Lambda + S3 輸出桶
└── functions/
    ├── generate/                 # POST /generate（依 provider 分派）、GET /generate/{jobId}（查狀態 + presign）
    │   └── providers/            # replicate.ts / runpod.ts / errors.ts
    ├── replicate-webhook/        # POST /replicate-webhook（Replicate 完成通知，HMAC 驗證）
    ├── runpod-webhook/           # POST /runpod-webhook（RunPod 完成通知，?token= 驗證）
    └── shared/{safety.ts, provider-types.ts}
```

DynamoDB job 欄位：`jobId`（PK）、`provider`、`providerJobId`（GSI `byProviderJobId`
反查用，兩條路共用）、`status`、`tier`、`characterId`、`outputKey`（RunPod 的 S3
key）/`outputUrl`（Replicate 的暫存網址）、`seed`、`errorMessage`、時間戳。

## 資料流

1. `POST /generate` → 依 provider 呼叫 `createRunpodJob` / `createReplicateJob` →
   寫 `status: pending` + `providerJobId` → 立刻回 `{ jobId }`。
2. provider 完成後打對應 webhook：Replicate 走 HMAC-SHA256 簽章；RunPod 走
   `?token=<RUNPOD_WEBHOOK_TOKEN>`（timing-safe 比對）。webhook 用 GSI 反查 jobId，
   更新 `status` 與 `outputKey`/`outputUrl`。
3. `GET /generate/{jobId}`：若有 `outputKey` 就對私有 S3 桶產 1 小時 presigned URL
   回給前端；Replicate 則直接回它的 `outputUrl`。

RunPod 的圖存進 **不公開**（BLOCK_ALL）的 S3 桶，30 天後自動刪，只能透過
presigned URL 存取。

## 還沒做的事

- **沒有實際部署測試過**（沒有 AWS 帳號）。
- **完全沒有前端**（Phase 3）。
- **RunPod worker 的 Docker image 要另外建**：見 `worker/` 與
  `.github/workflows/worker-image.yml`（在 GitHub Actions 上 build 推到 GHCR）。
- **Cognito 還沒接**：目前對外驗證只有 `x-internal-key`（已改成 timing-safe 比對），
  Phase 4 才換成正式登入。

## 設定步驟（實際部署時）

1. 裝依賴：
   ```powershell
   cd web
   npm install
   ```

2. 建 worker image 並開 RunPod endpoint（見 repo 根目錄 `worker/` 說明與
   [../CLOUD_GPU.md](../CLOUD_GPU.md)）：push 觸發 GitHub Actions build → RunPod
   建立 serverless endpoint（掛好 Network Volume，記下 endpoint id）→ 建一個只有
   `s3:PutObject on generated/*` 權限的 IAM user，把 key 貼進 RunPod endpoint 的
   secrets（`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION`/`S3_BUCKET`）。

3. 啟動 sandbox（需先 `aws configure`）：
   ```powershell
   npx ampx sandbox
   ```
   設定 secret：
   ```powershell
   npx ampx sandbox secret set INTERNAL_API_KEY
   npx ampx sandbox secret set RUNPOD_API_KEY
   npx ampx sandbox secret set RUNPOD_WEBHOOK_TOKEN
   # 只有要用 Replicate 那條路才需要：
   npx ampx sandbox secret set REPLICATE_API_TOKEN
   npx ampx sandbox secret set REPLICATE_WEBHOOK_SECRET
   ```
   - `INTERNAL_API_KEY`：自己產一組長字串（`openssl rand -hex 32`），前端帶在
     `x-internal-key` header
   - `RUNPOD_API_KEY`：RunPod 帳號的 API key
   - `RUNPOD_WEBHOOK_TOKEN`：自己產一組長字串，同一個值 generate 用來簽 webhook
     URL、runpod-webhook 用來驗證

4. 部署後在 Lambda 主控台把 `generate` 的環境變數填上：
   - `RUNPOD_ENDPOINT_ID`：第 2 步的 endpoint id
   - `REPLICATE_MODEL_VERSION`：只有要用 Replicate 才需要（見 `providers/replicate.ts`）
   - `S3_BUCKET`（給 RunPod worker，不是 Lambda）：`backend.ts` 的 `addOutput`
     會印出 `generatedBucketName`，那個值填進 RunPod endpoint 的 secrets

5. 測試（RunPod 路徑）：
   ```powershell
   curl -X POST "<generationApiUrl>generate" `
     -H "x-internal-key: <INTERNAL_API_KEY>" `
     -H "Content-Type: application/json" `
     -d '{\"prompt\": \"sitting in a cafe\", \"characterId\": \"mei\", \"tier\": \"safe\", \"aspectRatio\": \"3:4\"}'
   ```
   拿到 `jobId` 後輪詢，`status` 由 `pending` 轉 `completed`，`outputUrl` 是
   presigned S3 網址。錯的 `?token=` 應該回 401；`tier` 非法值會被 worker 當成
   `safe` 處理。
