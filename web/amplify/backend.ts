import { defineBackend } from "@aws-amplify/backend";
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";
import { LambdaIntegration, RestApi } from "aws-cdk-lib/aws-apigateway";
import { generate } from "./functions/generate/resource.js";
import { replicateWebhook } from "./functions/replicate-webhook/resource.js";

/**
 * 這個檔案是 Amplify Gen 2 backend 的 CDK 接線層——串起 DynamoDB（job 狀態）
 * + 兩個 Lambda（generate / replicateWebhook）+ API Gateway（對外 REST 端點）。
 *
 * 沒有在這個沙盒環境跑過 `npx ampx sandbox` 實際部署驗證過（沒有 AWS 帳號
 * 存取），是照 Amplify Gen 2 文件的既定寫法手刻的第一版——`@aws-amplify/backend`
 * 版本間 CDK escape hatch 的確切 API 偶爾會變，真的跑起來如果某幾行報錯，
 * 大機率就是這裡，不代表整體架構有問題。跑得動之後把這段註解拿掉即可。
 */

const backend = defineBackend({
  generate,
  replicateWebhook,
});

const apiStack = backend.createStack("generation-api-stack");

const jobsTable = new Table(apiStack, "GenerationJobsTable", {
  partitionKey: { name: "jobId", type: AttributeType.STRING },
  billingMode: BillingMode.PAY_PER_REQUEST, // 內部零星用量，按量計費比預先配置容量便宜
});

// replicate-webhook 收到回呼時只有 Replicate 的 predictionId，要靠這個 GSI
// 反查是哪個 jobId——見 replicate-webhook/handler.ts
jobsTable.addGlobalSecondaryIndex({
  indexName: "byReplicatePredictionId",
  partitionKey: { name: "replicatePredictionId", type: AttributeType.STRING },
});

jobsTable.grantReadWriteData(backend.generate.resources.lambda);
jobsTable.grantReadWriteData(backend.replicateWebhook.resources.lambda);

backend.generate.addEnvironment("JOBS_TABLE_NAME", jobsTable.tableName);
backend.replicateWebhook.addEnvironment("JOBS_TABLE_NAME", jobsTable.tableName);

// Replicate 上選定的模型 version id（見 REPLICATE_MODEL_VERSION 在
// generate/handler.ts 的說明）——這裡先留空字串，部署後在 Lambda 主控台或
// `npx ampx sandbox secret set` 補上，避免寫死在原始碼裡跟著 git commit
backend.generate.addEnvironment("REPLICATE_MODEL_VERSION", "");

const api = new RestApi(apiStack, "GenerationApi", {
  restApiName: "ai-image-lab-generation-api",
});

const generateResource = api.root.addResource("generate");
generateResource.addMethod("POST", new LambdaIntegration(backend.generate.resources.lambda));
generateResource
  .addResource("{jobId}")
  .addMethod("GET", new LambdaIntegration(backend.generate.resources.lambda));

const webhookResource = api.root.addResource("replicate-webhook");
webhookResource.addMethod("POST", new LambdaIntegration(backend.replicateWebhook.resources.lambda));

// webhook 網址要等 API Gateway 建立完成才知道實際網域，這裡回填給 generate
// function，讓它建立 Replicate prediction 時能帶上正確的 webhook URL
backend.generate.addEnvironment(
  "REPLICATE_WEBHOOK_URL",
  `${api.url}replicate-webhook`,
);

backend.addOutput({
  custom: {
    generationApiUrl: api.url,
  },
});
