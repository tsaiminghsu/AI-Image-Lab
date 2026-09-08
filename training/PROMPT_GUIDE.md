# Prompt cookbook：各 checkpoint 的可直接複製範例

這份文件把散在 `generate_character.py` 常數、`model_prompt_test.py` 實測結果、
`README.md` 觀察筆記裡的東西整理成一份「照著貼就能用」的參考。分兩層：

- **管線呼叫端**（GUI / `image_api.py` / `generate_character.py custom`）：永遠只打
  自然語言 prompt，下面列的 score 標籤、安全負面詞都由 `_build_prompt_and_negative()`
  自動處理，不用手動加。
- **直接在 ComfyUI 網頁介面測試**（例如驗證某個 checkpoint 的行為）：要自己把
  標籤/負面詞拼進去，下面每個 checkpoint 都給了完整可貼的字串。

## 6 個已安裝 checkpoint 一覽

| key | 檔名 | 架構 | Prompt 語法 |
|---|---|---|---|
| `juggernaut` | `juggernaut_xl_v9_photo.safetensors` | SDXL | 純自然語言 |
| `pony` | `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | Pony (SDXL 相容) | 需要 `score_9` 標籤 |
| `cyberrealistic_pony` | `CyberRealisticPony_V18.0_F16.safetensors` | Pony (SDXL 相容) | 需要 `score_9` 標籤 |
| `pony_realism` | `ponyRealism_V22.safetensors` | Pony (SDXL 相容) | 需要 `score_9` 標籤 |
| `realistic_vision` | `Realistic_Vision_V6.0_NV_B1_fp16.safetensors` | SD1.5 | 純自然語言 |
| `cyberrealistic` | `CyberRealistic_FINAL_FP16.safetensors` | SD1.5 | 純自然語言 |

GUI / `generate_character.py` 自訂生圖預設用 `cyberrealistic_pony`；資料集流程
（anchor/variations）固定用 `juggernaut`（見 `client.CHECKPOINT`）。

## 共用元件（`generate_character.py` 常數）

```
REALISTIC_STYLE  = "shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain"
REALISTIC_NEGATIVE = "3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine"

PONY_QUALITY_TAGS          = "score_9, score_8_up, score_7_up"   # 只有 Pony 系會自動加
PONY_QUALITY_NEGATIVE_TAGS = "score_6, score_5, score_4"          # 只有 Pony 系會自動加

AGE_SAFETY_NEGATIVE = "child, children, kid, minor, teen, teenager, underage, young girl"  # 任何分級都不會移除

SAFE_SAFETY_NEGATIVE = "nsfw, nude, naked, explicit, sexual content, " + AGE_SAFETY_NEGATIVE +
                        ", lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text"

SUGGESTIVE_NEGATIVE = "exposed genitalia, exposed vulva, exposed penis, exposed nipples, " +
                       "sexual intercourse, penetration, pornographic, explicit sexual act, " +
                       AGE_SAFETY_NEGATIVE + ", lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text"
```

管線內拼接順序（`_build_prompt_and_negative`）：

- **positive** = `[PONY_QUALITY_TAGS, ]` + 使用者 prompt + `, ` + `REALISTIC_STYLE`
  （Pony 系才有第一段）
- **negative** = `[PONY_QUALITY_NEGATIVE_TAGS, ]` + 安全負面詞（safe→`SAFE_SAFETY_NEGATIVE`
  / suggestive→`SUGGESTIVE_NEGATIVE`）+ `, ` + `REALISTIC_NEGATIVE`

---

## SDXL / SD1.5（`juggernaut`、`realistic_vision`、`cyberrealistic`）

純自然語言，不需要任何特殊標籤。範例（safe tier，等同管線實際送出的完整字串）：

**Positive**
```
a 25 year old woman with long brown wavy hair, sitting at a wooden cafe table,
holding a ceramic coffee cup, soft window light, cozy interior background,
shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```

**Negative**
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager,
underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text,
3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin,
symmetrical face, digital art, render, unreal engine
```

- `juggernaut` 解析度用 1024×1024（SDXL 原生）。
- `realistic_vision` / `cyberrealistic` 用 `client.SD15_WIDTH/HEIGHT`（512×768 附近，
  SD1.5 原生解析度只有 SDXL 的一半，實測耗時也是 SDXL/Pony 的約 1/3）。
- 風格 LoRA（`sdxl_photorealistic_slider`）只對這三個檔案調過，SD1.5 完全不掛
  （`SD15_CHECKPOINTS` 走的是沒有 LoRA 節點的獨立 workflow）。

---

## Pony 系（`pony`、`cyberrealistic_pony`、`pony_realism`）

Booru 標籤訓練出身，**一定要**在最前面加 `score_9, score_8_up, score_7_up`，
否則畫面明顯偏插畫/CG（實測見下方「有無標籤對照」）。範例（safe tier）：

**Positive**
```
score_9, score_8_up, score_7_up, a 25 year old woman with long brown wavy hair,
sitting at a wooden cafe table, holding a ceramic coffee cup, soft window light,
cozy interior background, shot on DSLR, natural skin texture, visible pores,
film photo, candid photograph, slight film grain
```

**Negative**
```
score_6, score_5, score_4, nsfw, nude, naked, explicit, sexual content, child, children,
kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs,
bad anatomy, watermark, text, 3d render, cgi, illustration, airbrushed, plastic skin,
doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

- 風格 LoRA 對 Pony 系要傳 `lora_strength=0.0`（它是照真實 SDXL checkpoint 調的，
  硬套在 Pony 上會跟它本身的畫風打架，見 `comfyui_client.py` 註解）。
- 解析度用 1024×1024（跟 juggernaut 一樣，Pony 是 SDXL 架構）。

### 有無 `score_9` 標籤對照（實測，RTX 2070，同 seed）

| checkpoint | 無標籤 | 有標籤 |
|---|---|---|
| `pony` | 40.3s，明顯插畫/CG 感 | 30.3s，寫實度明顯提升 |
| `cyberrealistic_pony` | 38.3s | 28.2s |
| `pony_realism` | 42.3s | 36.2s |

三個 Pony checkpoint 加標籤後反而都快了 8-10 秒（推測取樣器更早收斂到清晰結構），
但選擇加標籤的理由是畫質、不是速度。完整測試圖見
`training/reference_candidates/model_test/`（`contact_sheet.png` 是對照合成圖，
`REPORT.md` 是完整表格）。

---

## Suggestive tier 範例（僅供內部管線參考，不外流）

管線的 `tier="suggestive"` 只換安全負面詞（`SUGGESTIVE_NEGATIVE` 取代
`SAFE_SAFETY_NEGATIVE`），positive 端寫法不變，`AGE_SAFETY_NEGATIVE` 永遠保留：

**Negative（cyberrealistic_pony 例，含 Pony 標籤）**
```
score_6, score_5, score_4, exposed genitalia, exposed vulva, exposed penis, exposed nipples,
sexual intercourse, penetration, pornographic, explicit sexual act, child, children, kid,
minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs,
bad anatomy, watermark, text, 3d render, cgi, illustration, airbrushed, plastic skin,
doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

---

## 重現這份測試

```bash
python training/model_prompt_test.py
```

跑完在 `training/reference_candidates/model_test/REPORT.md` 產生表格、
每個 checkpoint 各一張 PNG。腳本用同一句 base prompt 對六個 checkpoint 各跑一次
（Pony 系額外測有/無標籤），純 txt2img（不掛 FaceID）以隔離「checkpoint + prompt
語法」本身的差異。
