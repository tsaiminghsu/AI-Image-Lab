/**
 * 共用型別：generate/handler.ts、providers/replicate.ts、providers/runpod.ts、
 * 兩個 webhook 都會用到，集中定義一次避免 union literal 在多個檔案各自重複、
 * 之後改一個忘了改另一個。
 */

export type Provider = "replicate" | "runpod";

export type AspectRatio = "1:1" | "3:4" | "4:3" | "9:16" | "16:9";

export type ReferenceImageContentType = "image/png" | "image/jpeg";
