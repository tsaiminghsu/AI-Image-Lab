import type { APIGatewayProxyHandler } from "aws-lambda";
import { createHmac, timingSafeEqual } from "node:crypto";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, QueryCommand, UpdateCommand } from "@aws-sdk/lib-dynamodb";

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.JOBS_TABLE_NAME!;

interface ReplicatePrediction {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  output?: unknown;
  error?: string | null;
}

export const handler: APIGatewayProxyHandler = async (event) => {
  const rawBody = event.body ?? "";
  if (!verifySignature(event.headers, rawBody, process.env.REPLICATE_WEBHOOK_SECRET!)) {
    return { statusCode: 401, body: "invalid signature" };
  }

  const prediction = JSON.parse(rawBody) as ReplicatePrediction;
  if (!prediction.id) {
    return { statusCode: 400, body: "missing prediction id" };
  }

  // jobId 不是 predictionId，要反查——用 provider 中立的 byProviderJobId GSI
  // （partition key: providerJobId，見 web/amplify/backend.ts），Replicate 跟
  // RunPod 兩條路共用同一個欄位與索引。量小的時候這樣就夠。
  const { Items } = await ddb.send(
    new QueryCommand({
      TableName: TABLE,
      IndexName: "byProviderJobId",
      KeyConditionExpression: "providerJobId = :pid",
      ExpressionAttributeValues: { ":pid": prediction.id },
    }),
  );

  const job = Items?.[0];
  if (!job) {
    return { statusCode: 404, body: "job not found for this prediction" };
  }

  const output = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output;

  await ddb.send(
    new UpdateCommand({
      TableName: TABLE,
      Key: { jobId: job.jobId },
      UpdateExpression: "SET #status = :status, outputUrl = :output, updatedAt = :now, errorMessage = :error",
      ExpressionAttributeNames: { "#status": "status" },
      ExpressionAttributeValues: {
        ":status": prediction.status === "succeeded" ? "completed" : "failed",
        ":output": output ?? null,
        ":now": new Date().toISOString(),
        ":error": prediction.error ?? null,
      },
    }),
  );

  return { statusCode: 200, body: "ok" };
};

/**
 * Replicate 用 Standard Webhooks 格式簽章：webhook-id / webhook-timestamp /
 * webhook-signature 三個 header，簽章內容是 `${id}.${timestamp}.${rawBody}`
 * 的 HMAC-SHA256（密鑰是 whsec_ 開頭、去掉前綴後 base64 解碼）。
 * 見 https://replicate.com/docs/topics/webhooks/verify-webhook
 *
 * 沒驗證這一步，任何人打這個公開網址過來都能偽造「生成完成」、塞任意
 * outputUrl 進資料庫——這條不是可以跳過的步驟。
 */
function verifySignature(
  headers: Record<string, string | undefined>,
  rawBody: string,
  secret: string,
): boolean {
  const id = getHeader(headers, "webhook-id");
  const timestamp = getHeader(headers, "webhook-timestamp");
  const signatureHeader = getHeader(headers, "webhook-signature");
  if (!id || !timestamp || !signatureHeader || !secret) return false;

  const signedContent = `${id}.${timestamp}.${rawBody}`;
  const secretBytes = Buffer.from(secret.replace(/^whsec_/, ""), "base64");
  const expected = createHmac("sha256", secretBytes).update(signedContent).digest();

  // header 格式是空白分隔的多組 "v1,<base64簽章>"，符合任何一組即通過
  return signatureHeader.split(" ").some((entry) => {
    const sig = entry.split(",")[1];
    if (!sig) return false;
    try {
      const provided = Buffer.from(sig, "base64");
      return provided.length === expected.length && timingSafeEqual(provided, expected);
    } catch {
      return false;
    }
  });
}

function getHeader(headers: Record<string, string | undefined>, name: string): string | undefined {
  const key = Object.keys(headers).find((k) => k.toLowerCase() === name.toLowerCase());
  return key ? headers[key] : undefined;
}
