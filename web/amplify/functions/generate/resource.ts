import { defineFunction, secret } from "@aws-amplify/backend";

/**
 * 接收前端的生成請求（POST /generate）跟狀態查詢（GET /generate/{jobId}）。
 * 只負責「送出 Replicate prediction + 記錄 job」，不等生成完成——完成通知
 * 靠 Replicate 的 webhook 打去 replicate-webhook function，見同層目錄。
 * 詳細架構原因見 repo 根目錄 WEB_DEPLOYMENT.md「為什麼一定要走非同步」。
 *
 * timeoutSeconds 只需要覆蓋「呼叫 Replicate API 建立 prediction」這個快
 * 動作（通常 <5 秒），不是整個生成時間。
 */
export const generate = defineFunction({
  name: "generate",
  entry: "./handler.ts",
  timeoutSeconds: 30,
  // 目前這個版本的 @aws-amplify/backend-function 沒有獨立的 `secrets` 選項
  // ——secret() 回傳的值直接放進 environment 就好，跟一般字串環境變數共用
  // 同一個欄位（型別是 Record<string, string | BackendSecret>）
  environment: {
    REPLICATE_API_TOKEN: secret("REPLICATE_API_TOKEN"),
    INTERNAL_API_KEY: secret("INTERNAL_API_KEY"),
    // RunPod Serverless path (see providers/runpod.ts). API key authenticates
    // the /run call; the webhook token is appended to the callback URL so
    // runpod-webhook can verify the callback (RunPod has no HMAC signing).
    RUNPOD_API_KEY: secret("RUNPOD_API_KEY"),
    RUNPOD_WEBHOOK_TOKEN: secret("RUNPOD_WEBHOOK_TOKEN"),
  },
});
