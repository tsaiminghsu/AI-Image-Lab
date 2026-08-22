/**
 * 對應 training/generate_character.py 的安全機制（AGE_SAFETY_NEGATIVE /
 * SAFE_SAFETY_NEGATIVE / SUGGESTIVE_NEGATIVE / MINIMUM_AGE），port 過來確保
 * 這條 Replicate 路徑有一樣的年齡保護跟內容分級下限。
 *
 * 重要差異：本地 ComfyUI workflow 是固定的 graph，negative prompt 節點
 * 保證每次生成都會套用；Replicate 上的模型是別人包好的，input schema 各不
 * 相同，有些模型（尤其純文字轉影片的）根本沒有 negative_prompt 欄位，這裡
 * 傳了也不會生效。這只是 best-effort 的第二層防線，不是像本地那樣的硬保證
 * ——選用哪個 Replicate 模型時要自己確認它的 input schema 有沒有吃
 * negative_prompt，不要假設這個函式就能把關住。
 */

export const AGE_SAFETY_NEGATIVE =
  "child, children, kid, minor, teen, teenager, underage, young girl";

const SAFE_SAFETY_NEGATIVE =
  `nsfw, nude, naked, explicit, sexual content, ${AGE_SAFETY_NEGATIVE}, ` +
  "lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text";

const SUGGESTIVE_NEGATIVE =
  "exposed genitalia, exposed vulva, exposed penis, exposed nipples, " +
  "sexual intercourse, penetration, pornographic, explicit sexual act, " +
  `${AGE_SAFETY_NEGATIVE}, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text`;

export type ContentTier = "safe" | "suggestive";

export function buildSafePrompt(userPrompt: string, tier: ContentTier) {
  // 跟 generate_character.py 的 MINIMUM_AGE=18 邏輯一致方向：在正面 prompt
  // 開頭就明講是成年角色，不是只靠 negative prompt 單方面把關。
  const positivePrompt = `adult, 18+ years old, ${userPrompt}`;
  const negativePrompt = tier === "suggestive" ? SUGGESTIVE_NEGATIVE : SAFE_SAFETY_NEGATIVE;
  return { positivePrompt, negativePrompt };
}
