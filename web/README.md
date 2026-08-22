# web/ — Amplify + Replicate 後端骨架

對應 [WEB_DEPLOYMENT.md](../WEB_DEPLOYMENT.md) 的 Phase 1：接 Replicate、
webhook 模式、非同步 job-queue。

**驗證狀態**：`npm install` 實際裝過依賴（`package.json` 版本號是裝起來後
從 `node_modules` 讀回來的真實版本，不是猜的），`npx tsc --noEmit` 對全部
6 個原始檔通過型別檢查——過程中抓到並修掉兩個真的會編譯失敗的問題（
`defineFunction` 沒有獨立的 `secrets` 選項，要跟 `environment` 合併用；
`backend.<function>.resources.lambda` 是唯讀的 `IFunction` 介面，沒有
`addEnvironment`，要呼叫 `backend.<function>.addEnvironment` 本身）。**但
沒有實際 `npx ampx sandbox` 部署測試過**——這個環境沒有 AWS 帳號存取，CDK
接線在型別層面正確不代表部署也一定順利（IAM 權限、資源命名限制這類問題
只有真的部署才會浮現）。

## 這裡有什麼

```
web/
├── amplify/
│   ├── backend.ts                        # CDK 接線：DynamoDB + API Gateway + 兩個 Lambda
│   └── functions/
│       ├── generate/                     # POST /generate（送出）、GET /generate/{jobId}（查狀態）
│       ├── replicate-webhook/            # POST /replicate-webhook（Replicate 完成通知）
│       └── shared/safety.ts              # port 自 generate_character.py 的安全機制
├── package.json
└── tsconfig.json
```

## 還沒做的事（誠實列出，不要以為這裡就是完整實作）

- **沒有實際部署測試過**——`npm install` + `npx tsc --noEmit` 都通過了，
  但 `npx ampx sandbox` 需要 AWS 帳號憑證，這個環境沒有，沒辦法驗證 IAM
  權限、資源命名這類只有真的部署才會浮現的問題
- **完全沒有前端**——只有後端 API，Phase 3 才會處理
- **只支援自由輸入 prompt**，`generate_character.py` 的 `CHARACTERS` 角色
  設定檔（11 個角色的年齡/外貌/風格）還沒 port 過來，`generate/handler.ts`
  現在只有 `prompt`/`tier` 兩個輸入欄位
- **negative prompt 的安全保證是 best-effort**，不是本地 ComfyUI workflow
  那種硬保證——見 `shared/safety.ts` 開頭註解，選用的 Replicate 模型如果
  input schema 沒有 `negative_prompt` 欄位，這層防護實際上不會生效
- **本地沒辦法測試**，因為 Lambda 需要 AWS 環境跑，`replicate-webhook` 也
  需要一個外網可打到的網址（Replicate 才能真的送 webhook 過來）

## 設定步驟（要實際部署時）

1. **裝依賴**（`package.json` 的版本號已經實測 `npm install` 裝得起來、
   `npx tsc --noEmit` 也通過，正常情況這步不會有版本衝突）：
   ```powershell
   cd web
   npm install
   ```

2. **選 Replicate 模型**：上 <https://replicate.com/explore> 找一個支援
   image/video 生成的模型，進它的 API 分頁複製 **version id**（一長串
   hash，不是 `owner/model-name` 這種名稱），本地先記下來，部署後設定成
   `REPLICATE_MODEL_VERSION`

3. **啟動 sandbox 開發環境**（需要先 `aws configure` 設好你自己的 AWS 帳號
   憑證）：
   ```powershell
   npx ampx sandbox
   ```
   跑起來後用下面指令設定三個 secret：
   ```powershell
   npx ampx sandbox secret set REPLICATE_API_TOKEN
   npx ampx sandbox secret set REPLICATE_WEBHOOK_SECRET
   npx ampx sandbox secret set INTERNAL_API_KEY
   ```
   - `REPLICATE_API_TOKEN`：<https://replicate.com/account/api-tokens>
   - `REPLICATE_WEBHOOK_SECRET`：Replicate 帳號的 webhook 簽章密鑰設定頁面
     （見 [replicate.com/docs/topics/webhooks/verify-webhook](https://replicate.com/docs/topics/webhooks/verify-webhook)）
   - `INTERNAL_API_KEY`：自己隨便產生一組長字串（例如
     `openssl rand -hex 32`），前端呼叫 API 時要帶在 `x-internal-key` header

4. **部署完成後**，去 Lambda 主控台把 `generate` function 的環境變數
   `REPLICATE_MODEL_VERSION` 填上第 2 步拿到的 version id（`backend.ts`
   目前先留空字串，避免寫死進原始碼跟著 git commit）

5. **測試**：
   ```powershell
   curl -X POST "<sandbox 輸出的 generationApiUrl>generate" `
     -H "x-internal-key: <INTERNAL_API_KEY 的值>" `
     -H "Content-Type: application/json" `
     -d '{\"prompt\": \"a photo of a cat\", \"tier\": \"safe\"}'
   ```
   拿到 `jobId` 後過幾秒到幾分鐘（依模型而定）：
   ```powershell
   curl "<generationApiUrl>generate/<jobId>" -H "x-internal-key: <...>"
   ```
   `status` 應該從 `pending` 變成 `completed`（或 `failed`），`outputUrl`
   會是生成結果的網址。
