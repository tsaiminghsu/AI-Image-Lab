import { defineFunction, secret } from "@aws-amplify/backend";

/**
 * 接收 Replicate 的 webhook 回呼（POST /replicate-webhook），Replicate
 * 生成完成時會自己打這個網址過來，不需要我們自己輪詢。
 * 見 https://replicate.com/docs/topics/webhooks
 *
 * REPLICATE_WEBHOOK_SECRET 用來驗證這個 request 真的是 Replicate 送的，不是
 * 有人猜到/找到這個網址後偽造完成通知、把 outputUrl 換成惡意內容——這個網址
 * 一定會是公開可存取的（Replicate 才打得到），沒有簽章驗證的話任何人都能
 * 偽造。見 handler.ts 的 verifySignature，密鑰在 Replicate 帳號的 webhook
 * 設定頁面可以拿到。
 */
export const replicateWebhook = defineFunction({
  name: "replicate-webhook",
  entry: "./handler.ts",
  timeoutSeconds: 10,
  // 見 generate/resource.ts 同樣的註解：secret() 放進 environment，沒有獨立的 secrets 選項
  environment: {
    REPLICATE_WEBHOOK_SECRET: secret("REPLICATE_WEBHOOK_SECRET"),
  },
});
