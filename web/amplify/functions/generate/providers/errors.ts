/**
 * Provider 呼叫失敗時用這個攜帶「應該回給呼叫端的 HTTP 狀態碼」，讓
 * generate/handler.ts 可以直接把 provider 模組丟出的錯誤轉成 API 回應，
 * 不用在 handler 裡對每個 provider 分別寫一次錯誤處理邏輯。
 */
export class ProviderError extends Error {
  readonly statusCode: number;

  constructor(statusCode: number, message: string) {
    super(message);
    this.name = "ProviderError";
    this.statusCode = statusCode;
  }
}
