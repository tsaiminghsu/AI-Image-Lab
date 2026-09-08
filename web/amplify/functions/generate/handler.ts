import type { APIGatewayProxyHandler } from "aws-lambda";
import { timingSafeEqual } from "node:crypto";
import { randomUUID } from "node:crypto";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand, PutCommand } from "@aws-sdk/lib-dynamodb";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { buildSafePrompt, type ContentTier } from "../shared/safety.js";
import type { Provider, AspectRatio, ReferenceImageContentType } from "../shared/provider-types.js";
import { ProviderError } from "./providers/errors.js";
import { createReplicateJob } from "./providers/replicate.js";
import { createRunpodJob } from "./providers/runpod.js";

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const s3 = new S3Client({});
const TABLE = process.env.JOBS_TABLE_NAME!;
const OUTPUT_BUCKET = process.env.OUTPUT_BUCKET;
const DEFAULT_PROVIDER = (process.env.DEFAULT_PROVIDER as Provider) ?? "runpod";

// ~2MB of binary is ~2.8M base64 chars. Reject bigger reference uploads before
// they hit the provider (and DynamoDB, though we don't store the image).
const MAX_REFERENCE_BASE64 = 2_800_000;

interface GenerateRequestBody {
  prompt?: string;
  tier?: ContentTier;
  provider?: Provider;
  characterId?: string;
  referenceImageBase64?: string;
  referenceImageContentType?: ReferenceImageContentType;
  loraId?: string;
  aspectRatio?: AspectRatio;
}

export const handler: APIGatewayProxyHandler = async (event) => {
  const providedKey = event.headers["x-internal-key"] ?? event.headers["X-Internal-Key"];
  if (!internalKeyMatches(providedKey, process.env.INTERNAL_API_KEY)) {
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

  // RunPod stores an S3 key (private bucket) - presign a short-lived GET so the
  // caller can fetch the image. Replicate stores its own temporary URL directly
  // in outputUrl (no S3 copy yet), so fall through to whatever is there.
  let outputUrl = Item.outputUrl ?? null;
  if (Item.outputKey && OUTPUT_BUCKET) {
    outputUrl = await getSignedUrl(
      s3,
      new GetObjectCommand({ Bucket: OUTPUT_BUCKET, Key: Item.outputKey }),
      { expiresIn: 3600 },
    );
  }
  return { statusCode: 200, body: JSON.stringify({ ...Item, outputUrl }) };
}

async function handleCreateJob(rawBody: string | null) {
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
  if (body.referenceImageBase64) {
    if (body.referenceImageBase64.length > MAX_REFERENCE_BASE64) {
      return { statusCode: 413, body: JSON.stringify({ error: "reference image too large (max ~2MB)" }) };
    }
    if (!body.referenceImageContentType) {
      return { statusCode: 400, body: JSON.stringify({ error: "referenceImageContentType required with referenceImageBase64" }) };
    }
  }

  const provider: Provider = body.provider ?? DEFAULT_PROVIDER;
  const jobId = randomUUID();

  try {
    let providerJobId: string;
    if (provider === "replicate") {
      // Replicate runs someone else's packaged model with no built-in safety
      // negatives, so we apply buildSafePrompt here (best effort).
      const { positivePrompt, negativePrompt } = buildSafePrompt(prompt, tier);
      ({ providerJobId } = await createReplicateJob({ positivePrompt, negativePrompt }));
    } else if (provider === "runpod") {
      // RunPod runs OUR ComfyUI worker, which applies the age/tier negatives
      // itself (generate_character._build_prompt_and_negative), so we send the
      // raw prompt + tier - double-applying would only conflict. See runpod.ts.
      ({ providerJobId } = await createRunpodJob({
        prompt,
        tier,
        characterId: body.characterId,
        referenceImageBase64: body.referenceImageBase64,
        referenceImageContentType: body.referenceImageContentType,
        loraId: body.loraId,
        aspectRatio: body.aspectRatio,
      }));
    } else {
      return { statusCode: 400, body: JSON.stringify({ error: `unknown provider ${provider}` }) };
    }

    await ddb.send(
      new PutCommand({
        TableName: TABLE,
        Item: {
          jobId,
          status: "pending",
          provider,
          providerJobId,
          tier,
          characterId: body.characterId ?? null,
          createdAt: new Date().toISOString(),
        },
      }),
    );

    return { statusCode: 200, body: JSON.stringify({ jobId }) };
  } catch (err) {
    if (err instanceof ProviderError) {
      return { statusCode: err.statusCode, body: JSON.stringify({ error: err.message }) };
    }
    return { statusCode: 502, body: JSON.stringify({ error: (err as Error).message ?? "provider call failed" }) };
  }
}

function internalKeyMatches(provided: string | undefined, expected: string | undefined): boolean {
  if (!provided || !expected || provided.length !== expected.length) return false;
  try {
    return timingSafeEqual(Buffer.from(provided), Buffer.from(expected));
  } catch {
    return false;
  }
}
