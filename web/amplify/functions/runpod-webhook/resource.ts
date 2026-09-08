import { defineFunction, secret } from "@aws-amplify/backend";

/**
 * 接收 RunPod Serverless 的 webhook 回呼（POST /runpod-webhook）。RunPod 沒有
 * Replicate 那種 Standard Webhooks HMAC 簽章機制，只能用 query string 帶一個
 * 共用密鑰（?token=<RUNPOD_WEBHOOK_TOKEN>）驗證回呼真的是我們送出的 job 觸發
 * 的——比 HMAC 弱，但 RunPod 目前只提供這個。generate function 送 job 時會把
 * 同一個 token 接在 webhook URL 後面（見 backend.ts 的 RUNPOD_WEBHOOK_URL
 * 回填 + providers/runpod.ts）。
 */
export const runpodWebhook = defineFunction({
  name: "runpod-webhook",
  entry: "./handler.ts",
  timeoutSeconds: 10,
  // 見 generate/resource.ts 同樣的註解：secret() 放進 environment
  environment: {
    RUNPOD_WEBHOOK_TOKEN: secret("RUNPOD_WEBHOOK_TOKEN"),
  },
});
