import { ProviderError } from "./errors.js";
import type { ContentTier } from "../../shared/safety.js";
import type { AspectRatio, ReferenceImageContentType } from "../../shared/provider-types.js";
import type { CreateProviderJobResult } from "./replicate.js";

/**
 * RunPod Serverless 這條路——對應 CLOUD_GPU.md / WEB_DEPLOYMENT.md 裡的
 * Phase 2。這裡只負責「打 RunPod 的 /run 端點送出一個 job」，實際跑
 * ComfyUI（IP-Adapter FaceID、LoRA、workflow 組裝……）是部署在 RunPod
 * Serverless 上的 worker container 的事——那個 container 不是這次的工作
 * 範圍（Docker image build + push 是明確列出的手動步驟，見 README）。
 *
 * 注意這裡刻意送的是**原始 `prompt` + `tier`**，不是 Replicate 那條路用
 * 的 buildSafePrompt() 產物（positivePrompt/negativePrompt）——見
 * ../../shared/safety.ts 開頭的說明：本地 ComfyUI workflow（未來 RunPod
 * worker 會原封不動搬過去的那套）本身就有寫死的 negative prompt 節點，
 * 是硬保證，會自己套用 generate_character.py 那一套年齡/安全機制。這裡
 * 再疊一層 buildSafePrompt() 算出來的 positivePrompt/negativePrompt 送過
 * 去，只會跟 worker 自己的安全邏輯打架或重複，不會更安全，所以不做。
 * Replicate 那條路才需要 buildSafePrompt()，因為它跑的是別人包好的模型，
 * 沒有這層硬保證，只能在呼叫端 best-effort 加一層。
 *
 * `_submit_and_wait()`（training/comfyui_client.py）不用拆成「送出/查詢」
 * 兩步——RunPod Serverless worker 是長駐 process，不像 Lambda 有硬性
 * 執行時間上限，可以在 worker handler 裡直接同步等到 ComfyUI 跑完，這裡
 * 只是把整個生成請求丟給 RunPod 排隊，跟 Replicate 一樣是 fire-and-forget
 * + webhook 拿結果，不是我們自己在 Lambda 裡等。
 */
export interface CreateRunpodJobInput {
  prompt: string;
  tier: ContentTier;
  characterId?: string;
  referenceImageBase64?: string;
  referenceImageContentType?: ReferenceImageContentType;
  loraId?: string;
  aspectRatio?: AspectRatio;
}

export async function createRunpodJob(
  input: CreateRunpodJobInput,
): Promise<CreateProviderJobResult> {
  const endpointId = process.env.RUNPOD_ENDPOINT_ID;
  const apiKey = process.env.RUNPOD_API_KEY;
  if (!endpointId || !apiKey) {
    throw new ProviderError(
      500,
      "RUNPOD_ENDPOINT_ID / RUNPOD_API_KEY 環境變數還沒設定",
    );
  }

  // RUNPOD_WEBHOOK_URL 是 runpod-webhook function 的 API Gateway 網址（見
  // amplify/backend.ts，跟 REPLICATE_WEBHOOK_URL 同樣的回填模式），
  // RUNPOD_WEBHOOK_TOKEN 是 runpod-webhook/handler.ts 用來驗證回呼真的是
  // 這次送出的請求觸發的共用密鑰——RunPod 沒有 Replicate 那種 HMAC 簽章
  // 機制，只能用 query string 帶密鑰這種較弱的防護，見 runpod-webhook
  // handler.ts 開頭的說明。
  const webhookBaseUrl = process.env.RUNPOD_WEBHOOK_URL;
  const webhookToken = process.env.RUNPOD_WEBHOOK_TOKEN ?? "";
  const webhook = webhookBaseUrl
    ? `${webhookBaseUrl}?token=${encodeURIComponent(webhookToken)}`
    : undefined;

  const response = await fetch(`https://api.runpod.ai/v2/${endpointId}/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: {
        prompt: input.prompt,
        tier: input.tier,
        characterId: input.characterId,
        referenceImageBase64: input.referenceImageBase64,
        referenceImageContentType: input.referenceImageContentType,
        loraId: input.loraId,
        aspectRatio: input.aspectRatio,
      },
      webhook,
    }),
  });

  const result = (await response.json()) as { id?: string; error?: string };
  if (!response.ok || !result.id) {
    throw new ProviderError(502, result.error ?? "RunPod API 呼叫失敗");
  }

  return { providerJobId: result.id };
}
