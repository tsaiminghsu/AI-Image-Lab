import type { APIGatewayProxyHandler } from "aws-lambda";
import { timingSafeEqual } from "node:crypto";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, QueryCommand, UpdateCommand } from "@aws-sdk/lib-dynamodb";

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.JOBS_TABLE_NAME!;

/**
 * RunPod serverless webhook body. RunPod POSTs the whole job object; the fields
 * we use are id (== the providerJobId we stored), status, and output (our
 * handler.py returns {outputKey, outputUrl, seed, width, height}) or error.
 * See https://docs.runpod.io/serverless/endpoints/send-requests#webhooks
 */
interface RunpodWebhook {
  id: string;
  status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED" | "TIMED_OUT";
  output?: {
    outputKey?: string;
    outputUrl?: string;
    seed?: number;
    width?: number;
    height?: number;
    error?: string;
  } | null;
  error?: string | null;
}

const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);

export const handler: APIGatewayProxyHandler = async (event) => {
  // RunPod has no HMAC signature; we gate on a shared secret in the query string.
  const provided = event.queryStringParameters?.token ?? "";
  const expected = process.env.RUNPOD_WEBHOOK_TOKEN ?? "";
  if (!tokensMatch(provided, expected)) {
    return { statusCode: 401, body: "invalid token" };
  }

  let body: RunpodWebhook;
  try {
    body = JSON.parse(event.body ?? "{}") as RunpodWebhook;
  } catch {
    return { statusCode: 400, body: "invalid JSON body" };
  }
  if (!body.id) {
    return { statusCode: 400, body: "missing job id" };
  }

  // Non-terminal statuses (queued/in-progress) are acknowledged but ignored, so
  // RunPod doesn't retry them.
  if (!TERMINAL.has(body.status)) {
    return { statusCode: 200, body: "ignored non-terminal status" };
  }

  // Map the provider job id back to our jobId via the GSI (see backend.ts).
  const { Items } = await ddb.send(
    new QueryCommand({
      TableName: TABLE,
      IndexName: "byProviderJobId",
      KeyConditionExpression: "providerJobId = :pid",
      ExpressionAttributeValues: { ":pid": body.id },
    }),
  );
  const job = Items?.[0];
  if (!job) {
    // 200 so RunPod stops retrying; nothing we can do about an unknown job.
    return { statusCode: 200, body: "job not found for this provider job id" };
  }

  const completed = body.status === "COMPLETED" && !!body.output?.outputKey;
  const errorMessage = body.error ?? body.output?.error ?? (completed ? null : `job ${body.status}`);

  await ddb.send(
    new UpdateCommand({
      TableName: TABLE,
      Key: { jobId: job.jobId },
      UpdateExpression:
        "SET #status = :status, outputKey = :key, seed = :seed, errorMessage = :error, updatedAt = :now",
      ExpressionAttributeNames: { "#status": "status" },
      ExpressionAttributeValues: {
        ":status": completed ? "completed" : "failed",
        ":key": body.output?.outputKey ?? null,
        ":seed": body.output?.seed ?? null,
        ":error": errorMessage,
        ":now": new Date().toISOString(),
      },
    }),
  );

  return { statusCode: 200, body: "ok" };
};

function tokensMatch(provided: string, expected: string): boolean {
  if (!expected || provided.length !== expected.length) return false;
  try {
    return timingSafeEqual(Buffer.from(provided), Buffer.from(expected));
  } catch {
    return false;
  }
}
