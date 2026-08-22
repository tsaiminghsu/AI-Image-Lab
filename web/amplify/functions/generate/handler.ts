import type { APIGatewayProxyHandler } from "aws-lambda";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand, PutCommand } from "@aws-sdk/lib-dynamodb";
import { randomUUID } from "node:crypto";
import { buildSafePrompt, type ContentTier } from "../shared/safety.js";

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.JOBS_TABLE_NAME!;

// 上 replicate.com/explore 選定模型後填進 Amplify 的環境變數/secret，不寫死
// 在程式碼裡——跟 CLOUD_GPU.md 的建議一致，Replicate 上的影片模型汰換快。
// 值是模型的 version id（在模型頁面的 API 分頁可以找到），不是 owner/name。
const REPLICATE_MODEL_VERSION = process.env.REPLICATE_MODEL_VERSION ?? "";

interface GenerateRequestBody {
  prompt?: string;
  tier?: ContentTier;
}

export const handler: APIGatewayProxyHandler = async (event) => {
  const providedKey = event.headers["x-internal-key"] ?? event.headers["X-Internal-Key"];
  if (providedKey !== process.env.INTERNAL_API_KEY) {
    return { statusCode: 401, body: JSON.stringify({ error: "unauthorized" }) };
  }

  if (event.httpMethod === "GET") {
    return handleStatusCheck(event.pathParameters?.jobId);
  }

  return handleCreateJob(event.body);
};

async function handleStatusCheck(jobId: string | undefined) {
  if (!jobId) {
    return { statusCode: 400, body: JSON.stringify({ error: "missing jobId" }) };
  }
  const { Item } = await ddb.send(new GetCommand({ TableName: TABLE, Key: { jobId } }));
  if (!Item) {
    return { statusCode: 404, body: JSON.stringify({ error: "job not found" }) };
  }
  return { statusCode: 200, body: JSON.stringify(Item) };
}

async function handleCreateJob(rawBody: string | null) {
  if (!REPLICATE_MODEL_VERSION) {
    // 還沒設定要用哪個 Replicate 模型 — 見 web/README.md「設定 Replicate 模型」
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "REPLICATE_MODEL_VERSION 環境變數還沒設定" }),
    };
  }

  let body: GenerateRequestBody;
  try {
    body = JSON.parse(rawBody ?? "{}");
  } catch {
    return { statusCode: 400, body: JSON.stringify({ error: "invalid JSON body" }) };
  }

  const { prompt, tier = "safe" } = body;
  if (!prompt) {
    return { statusCode: 400, body: JSON.stringify({ error: "missing prompt" }) };
  }

  const jobId = randomUUID();
  const { positivePrompt, negativePrompt } = buildSafePrompt(prompt, tier);

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
      input: { prompt: positivePrompt, negative_prompt: negativePrompt },
      webhook: process.env.REPLICATE_WEBHOOK_URL,
      webhook_events_filter: ["completed"],
    }),
  });

  const prediction = (await response.json()) as { id?: string; detail?: string };
  if (!response.ok || !prediction.id) {
    return {
      statusCode: 502,
      body: JSON.stringify({ error: prediction.detail ?? "Replicate API 呼叫失敗" }),
    };
  }

  await ddb.send(
    new PutCommand({
      TableName: TABLE,
      Item: {
        jobId,
        status: "pending",
        prompt: positivePrompt,
        tier,
        replicatePredictionId: prediction.id,
        createdAt: new Date().toISOString(),
      },
    }),
  );

  return { statusCode: 200, body: JSON.stringify({ jobId }) };
}
