import { ProviderError } from "./errors.js";

/**
 * 從原本內嵌在 generate/handler.ts 裡的邏輯抽出來——這個檔案只負責「呼叫
 * Replicate REST API 建立一個 prediction」，不知道 DynamoDB、不知道
 * request 怎麼被驗證，這些留在 handler.ts。
 *
 * REPLICATE_MODEL_VERSION 這個環境變數的值仍然沒有解決——這個環境裡沒有
 * Replicate 帳號，沒辦法挑一個真的存在、驗證過的 model version id。
 * 交棒給部署的人時的候選方向：Replicate 上 InstantID 家族（例如
 * `zsxkib/instant-id` 這類 image-to-image + IP-Adapter face 的模型）或
 * PhotoMaker 家族（`tencentarc/photomaker`），兩者都是「給一張參考臉、
 * 生出保留該人臉特徵的新圖」這個需求方向對得上的模型類型。**這只是選型
 * 方向的建議，不是驗證過的具體 slug/version**——Replicate 的模型目錄會
 * 汰換、model 頁面的 version id 會隨作者更新模型而變動，部署前一定要自己
 * 上 replicate.com 確認：(1) 這個 model 還在、還維護中，(2) API 分頁列出
 * 的最新 version id，(3) 它的 input schema 有沒有 `negative_prompt` 欄位
 * （見 ../../shared/safety.ts 開頭的說明——沒有的話那層防護不會生效），
 * (4) 目前的計價（Replicate 的模型計價會變動，跟 CLOUD_GPU.md 記錄的當時
 * 查證結果不一定還準）。
 */
const REPLICATE_MODEL_VERSION = process.env.REPLICATE_MODEL_VERSION ?? "";

export interface CreateReplicateJobInput {
  positivePrompt: string;
  negativePrompt: string;
}

export interface CreateProviderJobResult {
  providerJobId: string;
}

export async function createReplicateJob(
  input: CreateReplicateJobInput,
): Promise<CreateProviderJobResult> {
  if (!REPLICATE_MODEL_VERSION) {
    // 還沒設定要用哪個 Replicate 模型 — 見 web/README.md「設定 Replicate 模型」
    throw new ProviderError(500, "REPLICATE_MODEL_VERSION 環境變數還沒設定");
  }

  // 用原生 fetch 直接打 Replicate 的 REST API，不引入 replicate npm SDK——
  // 這裡只需要「建立 prediction」這一個呼叫，SDK 的其他功能用不到。
  const response = await fetch("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.REPLICATE_API_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      version: REPLICATE_MODEL_VERSION,
      // negative_prompt 不是每個模型的 input schema 都有，模型不支援時
      // Replicate 通常會回 422 驗證錯誤——選模型時先去它的 API 文件分頁確認
      input: { prompt: input.positivePrompt, negative_prompt: input.negativePrompt },
      webhook: process.env.REPLICATE_WEBHOOK_URL,
      webhook_events_filter: ["completed"],
    }),
  });

  const prediction = (await response.json()) as { id?: string; detail?: string };
  if (!response.ok || !prediction.id) {
    throw new ProviderError(502, prediction.detail ?? "Replicate API 呼叫失敗");
  }

  return { providerJobId: prediction.id };
}
