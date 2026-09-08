import { defineBackend } from "@aws-amplify/backend";
import { Stack, Duration } from "aws-cdk-lib";
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";
import { LambdaIntegration, RestApi } from "aws-cdk-lib/aws-apigateway";
import { Bucket, BlockPublicAccess, BucketEncryption } from "aws-cdk-lib/aws-s3";
import { generate } from "./functions/generate/resource.js";
import { replicateWebhook } from "./functions/replicate-webhook/resource.js";
import { runpodWebhook } from "./functions/runpod-webhook/resource.js";

/**
 * Amplify Gen 2 backend CDK 接線層：DynamoDB（job 狀態）+ 三個 Lambda（generate /
 * replicateWebhook / runpodWebhook）+ API Gateway（REST 端點）+ S3（生成圖輸出）。
 *
 * 沒有在這個沙盒環境跑過 `npx ampx sandbox` 實際部署（沒有 AWS 帳號），是照
 * Amplify Gen 2 文件手刻的——CDK escape hatch 的確切 API 版本間偶爾會變，真的
 * 跑起來如果某幾行報錯，大機率在這裡，不代表架構有問題。
 */

const backend = defineBackend({
  generate,
  replicateWebhook,
  runpodWebhook,
});

const apiStack = backend.createStack("generation-api-stack");
const region = Stack.of(apiStack).region;

const jobsTable = new Table(apiStack, "GenerationJobsTable", {
  partitionKey: { name: "jobId", type: AttributeType.STRING },
  billingMode: BillingMode.PAY_PER_REQUEST, // 內部零星用量，按量計費
});

// Provider 中立的反查索引：兩條路（Replicate / RunPod）webhook 回來時都只有
// provider 自己的 job id（providerJobId），靠這個 GSI 反查我們的 jobId。
// 見 replicate-webhook/handler.ts 與 runpod-webhook/handler.ts。
jobsTable.addGlobalSecondaryIndex({
  indexName: "byProviderJobId",
  partitionKey: { name: "providerJobId", type: AttributeType.STRING },
});

jobsTable.grantReadWriteData(backend.generate.resources.lambda);
jobsTable.grantReadWriteData(backend.replicateWebhook.resources.lambda);
jobsTable.grantReadWriteData(backend.runpodWebhook.resources.lambda);

// 生成圖輸出桶——RunPod worker 用 s3:PutObject 寫入 generated/<jobId>.png，
// generate 的狀態查詢用 s3:GetObject 產 presigned URL 回給前端。**不對外公開**
// （BLOCK_ALL），只能透過 presigned URL 存取，30 天後自動刪除。
const outputBucket = new Bucket(apiStack, "GeneratedImagesBucket", {
  blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
  encryption: BucketEncryption.S3_MANAGED,
  enforceSSL: true,
  lifecycleRules: [{ expiration: Duration.days(30) }],
});
outputBucket.grantRead(backend.generate.resources.lambda);

backend.generate.addEnvironment("JOBS_TABLE_NAME", jobsTable.tableName);
backend.replicateWebhook.addEnvironment("JOBS_TABLE_NAME", jobsTable.tableName);
backend.runpodWebhook.addEnvironment("JOBS_TABLE_NAME", jobsTable.tableName);

backend.generate.addEnvironment("OUTPUT_BUCKET", outputBucket.bucketName);

// 預設用哪個 provider（runpod / replicate），可在 Lambda 主控台改。
backend.generate.addEnvironment("DEFAULT_PROVIDER", "runpod");

// 這些留空/待填，部署後在 Lambda 主控台或 `npx ampx sandbox` 補上，不寫死進 git：
//  - REPLICATE_MODEL_VERSION：Replicate 模型 version id（見 generate/handler.ts）
//  - RUNPOD_ENDPOINT_ID：RunPod serverless endpoint id（見 providers/runpod.ts）
backend.generate.addEnvironment("REPLICATE_MODEL_VERSION", "");
backend.generate.addEnvironment("RUNPOD_ENDPOINT_ID", "");

const api = new RestApi(apiStack, "GenerationApi", {
  restApiName: "ai-image-lab-generation-api",
});

const generateResource = api.root.addResource("generate");
generateResource.addMethod("POST", new LambdaIntegration(backend.generate.resources.lambda));
generateResource
  .addResource("{jobId}")
  .addMethod("GET", new LambdaIntegration(backend.generate.resources.lambda));

const replicateWebhookResource = api.root.addResource("replicate-webhook");
replicateWebhookResource.addMethod("POST", new LambdaIntegration(backend.replicateWebhook.resources.lambda));

const runpodWebhookResource = api.root.addResource("runpod-webhook");
runpodWebhookResource.addMethod("POST", new LambdaIntegration(backend.runpodWebhook.resources.lambda));

// webhook 網址要回填給 generate function，讓它送 provider job 時能帶上正確的
// 回呼網址。**不要用 `api.url` 或 `api.deploymentStage`**——那會建立
// Function -> Stage -> Deployment -> Method -> Function 的 CloudFormation
// 循環相依（原本 Replicate 版就是這樣，從沒成功部署過）。改用 restApiId +
// 寫死的預設 stage 名 "prod"，restApiId 只參照 RestApi resource 本身，打斷循環。
const apiBaseUrl = `https://${api.restApiId}.execute-api.${region}.amazonaws.com/prod`;
backend.generate.addEnvironment("REPLICATE_WEBHOOK_URL", `${apiBaseUrl}/replicate-webhook`);
backend.generate.addEnvironment("RUNPOD_WEBHOOK_URL", `${apiBaseUrl}/runpod-webhook`);

backend.addOutput({
  custom: {
    generationApiUrl: api.url,
    // 給部署者建立「只有 s3:PutObject on generated/*」的 IAM user 時用（那把
    // key 貼進 RunPod endpoint 的 secrets，worker 才能上傳結果）。
    generatedBucketName: outputBucket.bucketName,
  },
});
