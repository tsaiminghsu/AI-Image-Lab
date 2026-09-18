# 各模型關鍵字總表與 Prompt 範例

這份文件把 `generate_character.py` 的常數與詞庫、`comfyui_client.py` 的模型登錄表、
以及各條流程實際送給 ComfyUI 的字串整理成一份總表。兩種讀者：

- **管線呼叫端**（GUI／`image_api.py`／`generate_character.py custom`／worker）：只要打自然語言，
  score 標籤、身分前綴、風格詞、安全負面詞都由 `_build_prompt_and_negative()` 自動接上，不用手動加。
- **直接在 ComfyUI 網頁介面測試**：要自己把標籤和負面詞拼進去，第 5 節每個家族都有可以整段複製的字串。

分工：本檔是「常數 + 總表 + 管線實際送出的字串」；[`tests/TEST_PROMPTS.md`](../tests/TEST_PROMPTS.md)
是「手動測試要打什麼」的題庫。

> 這份文件由 `tests/test_prompt_guide.py` 釘住：常數、詞庫、角色、場景、骨架、範例只要跟程式不一致，
> `check.ps1` 就會紅，並印出可以直接貼回來的正確字串。所以請不要憑印象改這裡的字串。

## 1. 模型總表

「自動加」欄指管線會替你接上的東西；「中文」欄指這個模型的文字編碼器本身讀不讀得懂中文。

| key | 檔名 | 家族 | Prompt 方言 | 解析度 | steps / cfg | sampler / scheduler | 自動加 | 中文 |
|---|---|---|---|---|---|---|---|---|
| `juggernaut` | `juggernaut_xl_v9_photo.safetensors` | SDXL | 純自然語言 | 1024×1024 | 30 / 6.0 | `dpmpp_2m` / `karras` | 風格詞 | 否 |
| `pony` | `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | Pony（SDXL 架構） | booru 標籤 + score 標籤 | 1024×1024 | 30 / 6.0 | `dpmpp_2m` / `karras` | score 標籤 + 風格詞 | 否 |
| `cyberrealistic_pony` | `CyberRealisticPony_V18.0_F16.safetensors` | Pony（SDXL 架構） | booru 標籤 + score 標籤 | 1024×1024 | 30 / 6.0 | `dpmpp_2m` / `karras` | score 標籤 + 風格詞 | 否 |
| `pony_realism` | `ponyRealism_V22.safetensors` | Pony（SDXL 架構） | booru 標籤 + score 標籤 | 1024×1024 | 30 / 6.0 | `dpmpp_2m` / `karras` | score 標籤 + 風格詞 | 否 |
| `realistic_vision` | `Realistic_Vision_V6.0_NV_B1_fp16.safetensors` | SD1.5 | 純自然語言 | 512×768 | 30 / 7.0 | `dpmpp_2m` / `karras` | 風格詞 + `(man:1.3)` | 否 |
| `cyberrealistic` | `CyberRealistic_FINAL_FP16.safetensors` | SD1.5 | 純自然語言 | 512×768 | 30 / 7.0 | `dpmpp_2m` / `karras` | 風格詞 + `(man:1.3)` | 否 |
| `z_image_turbo` | 三檔（見下） | Z-Image Turbo | 自然語言，中英皆可 | 1024×1024 | 8 / 2.0 | `res_multistep` / `simple`（shift 3.0） | 風格詞 | **是** |
| `wan22_ti2v_5b` | 三檔（見下） | Wan 2.2 TI2V-5B（影片） | 自然語言句子，描述動作 | 1280×704 | 20 / 5.0 | `uni_pc` / `simple`（shift 8.0） | 不加風格詞 | **是** |
| `sd15_base` | `v1-5-pruned-emaonly.safetensors` | AnimateDiff（SD1.5） | 純自然語言 | 512×512 | 20 / 7.5 | `dpmpp_2m` / `karras` | 影片風格詞 + `(man:1.3)` | 否 |
| realistic_vision（AnimateDiff） | `Realistic_Vision_V6.0_NV_B1_fp16.safetensors` | AnimateDiff（SD1.5） | 純自然語言 | 512×512 | 20 / 7.5 | `dpmpp_2m` / `karras` | 影片風格詞 + `(man:1.3)` | 否 |
| cyberrealistic（AnimateDiff） | `CyberRealistic_FINAL_FP16.safetensors` | AnimateDiff（SD1.5） | 純自然語言 | 512×512 | 20 / 7.5 | `dpmpp_2m` / `karras` | 影片風格詞 + `(man:1.3)` | 否 |
| SVD | `svd.safetensors` | SVD 圖生影片 | **沒有 prompt 欄位** | 512×512 | 30 / 2.5 | `euler` / `karras` | 不適用 | 不適用 |

功能可用性（決定同一句 prompt 在哪些模型上會被降級成純文字）：

| 家族 | FaceID 鎖臉 | ControlNet 骨架 | HQ 兩段式 | 風格 LoRA | 角色 LoRA |
|---|---|---|---|---|---|
| SDXL `juggernaut` | 有 | 有 | 有 | 有，強度 2.5 | 有 |
| Pony 三檔 | 有 | 有 | 有 | 要傳 `lora_strength=0.0` 關掉 | 有 |
| SD1.5 兩檔 | 無 | 無 | 無 | 無（走沒有 LoRA 節點的獨立 workflow） | 無 |
| Z-Image Turbo | 無 | 無 | 無 | 無 | 無 |
| Wan 2.2（影片） | 無（長相由首幀圖決定） | 無 | 不適用 | 無 | 無 |
| AnimateDiff（影片） | 有 | 無 | 逐幀 hires + 臉部精修 | 無 | 無 |
| SVD（影片） | 不適用 | 不適用 | 不適用 | 無 | 無 |

多檔模型的實際檔名（`ZIMAGE_MODELS`／`WAN_MODELS`。它們是三個檔案不是一個 checkpoint，所以刻意不放進 `CHECKPOINTS`）：

| key | unet | text_encoder | vae |
|---|---|---|---|
| `z_image_turbo` | `z_image_turbo_int8_convrot.safetensors` | `qwen_3_4b_fp8_mixed.safetensors` | `ae.safetensors` |
| `wan22_ti2v_5b` | `wan2.2_ti2v_5B_fp16.safetensors` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `wan2.2_vae.safetensors` |

ControlNet 模型 `control-lora-openposeXL2-rank256.safetensors`，強度 0.8。
FaceID 用 `FACEID PLUS V2` preset，權重 1.0（背面鏡頭自動降到 0.3）。
HQ 路徑：第一段 24 步，`4x-UltraSharp.pth` 放大 1.5 倍，hires 20 步（denoise 0.4）。

### 1a. AnimateDiff 的動作 LoRA 與 LCM 預設

動作模組 `mm_sd_v15_v2.ckpt`，16 影格／8 fps。
動作 LoRA **只控制鏡頭運動**（整個畫面的推拉搖移），沒有身體部位層級的效果：

| key | 檔名 | 效果 |
|---|---|---|
| `zoom_in` | `v2_lora_ZoomIn.ckpt` | 鏡頭推近 |
| `zoom_out` | `v2_lora_ZoomOut.ckpt` | 鏡頭拉遠 |
| `pan_left` | `v2_lora_PanLeft.ckpt` | 向左橫搖 |
| `pan_right` | `v2_lora_PanRight.ckpt` | 向右橫搖 |
| `tilt_up` | `v2_lora_TiltUp.ckpt` | 向上仰搖 |
| `tilt_down` | `v2_lora_TiltDown.ckpt` | 向下俯搖 |
| `rolling_clockwise` | `v2_lora_RollingClockwise.ckpt` | 順時針旋轉 |
| `rolling_anticlockwise` | `v2_lora_RollingAnticlockwise.ckpt` | 逆時針旋轉 |

LCM 加速預設（`ANIMATEDIFF_LCM_PRESETS`，兩個都是 8 步／cfg 2.0／`lcm`／`sgm_uniform`）：

| preset | 動作模組 | LoRA | 動作 LoRA 相容性 |
|---|---|---|---|
| `animatelcm` | `AnimateLCM_sd15_t2v.ckpt` | `AnimateLCM_sd15_t2v_lora.safetensors` | **未驗證** |
| `lcm_lora` | `mm_sd_v15_v2.ckpt` | `lcm-lora-sdv1-5.safetensors` | 已驗證 |

### 1b. cfg 下限

**`SAFETY_MIN_CFG` = 1.5**（`comfyui_client.py`）。cfg 等於 1.0 時 ComfyUI 會整段跳過負面條件，
年齡與內容安全負面詞會無聲失效，所以由 `enforce_min_cfg()` 在 `_submit_and_wait` 統一把 cfg 夾到這個下限。
`ZIMAGE_MIN_CFG` 和 `LCM_MIN_CFG` 都只是它的別名。Z-Image 官方範本用 cfg 1.0，本專案刻意用
2.0；Wan 的官方 cfg 5.0 本來就在下限之上。**不要為了省時間降到 1.0。**

## 2. 組裝規則

所有流程都經過 `generate_character._build_prompt_and_negative()`，順序固定：

```
positive = [PONY_QUALITY_TAGS, ] [character_base_prompt(trigger), ] <你打的 prompt> [, style_positive]
negative = [PONY_QUALITY_NEGATIVE_TAGS, ] <tier 安全負面詞> [, style_negative] [, extra_negative]
```

- `PONY_QUALITY_TAGS`／`PONY_QUALITY_NEGATIVE_TAGS` 只有 checkpoint 屬於 `PONY_CHECKPOINTS` 時才加。
- `style_positive`／`style_negative` 傳 `None` 時分別預設成 `REALISTIC_STYLE`／`REALISTIC_NEGATIVE`；
  傳空字串則整段不加（Wan 就是這樣關掉風格詞的）。
- tier `safe` 用 `SAFE_SAFETY_NEGATIVE`，`suggestive` 用 `SUGGESTIVE_NEGATIVE`。兩者都含 `AGE_SAFETY_NEGATIVE`，
  沒有任何參數能拿掉。
- `gender_weight="auto"`（預設）只有 SD1.5 checkpoint 會套用 `SD15_GENDER_WEIGHT`，把性別寫成 `(man:1.3)`。

各入口傳進去的參數不同，這是家族差異的真正來源：

| 入口 | checkpoint | style_positive | style_negative | gender_weight |
|---|---|---|---|---|
| `gen_custom()`（CLI／GUI／API 靜態圖） | 使用者選的 | 預設 `REALISTIC_STYLE` | 預設 `REALISTIC_NEGATIVE` | `auto` |
| `gen_gif()` | 使用者選的 | 同上 | 同上 | `auto` |
| `plan_picker()` 預覽 | 使用者選的 | `REALISTIC_STYLE` | `REALISTIC_NEGATIVE` | `auto` |
| `gen_video_animatediff()` | **`None`** | 預設 | **`VIDEO_REALISTIC_NEGATIVE`** | **固定 1.3** |
| `build_wan_prompts()` | **`None`** | **空字串，不加** | **`WAN_VIDEO_NEGATIVE`** | **`None`，不加** |

兩條影片路線傳 `checkpoint=None` 是刻意的：AnimateDiff 的 checkpoint key（例如 `sd15_base`）不在
`SD15_CHECKPOINTS` 裡，傳進去會讓 `auto` 判斷失準；而兩條路線都不該拿到 Pony 的 score 標籤。

姿勢骨架的文字提示（`prompt_hint`）由 `_plan_custom()` 追加在 prompt 尾端，不在這個函式裡；
picker 分頁在沒有 ControlNet 的模型上會額外把 `camera` 也接上去（見 4f）。

## 3. 常數一覽

全部在 `training/generate_character.py`。下面是逐字內容，改了程式就要改這一節（測試會抓）。

**`REALISTIC_STYLE`** — 預設風格正面詞，靜態圖與 AnimateDiff 都用：
```
shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```

**`REALISTIC_NEGATIVE`** — 預設風格負面詞：
```
3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

**`VIDEO_REALISTIC_NEGATIVE`** — AnimateDiff 用，比上面少了 `symmetrical face`：
```
3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, digital art, render, unreal engine
```

**`WAN_VIDEO_NEGATIVE`** — Wan 用，改成針對影片假影（靜止、閃爍、手指、臉部形變）：
```
static image, frozen frame, flickering, jitter, blurry, overexposed, low quality, jpeg artifacts, deformed hands, extra fingers, distorted face, morphing face, melting limbs, subtitles, watermark
```

**`PONY_QUALITY_TAGS`**／**`PONY_QUALITY_NEGATIVE_TAGS`** — 只有 Pony 系自動加：
```
score_9, score_8_up, score_7_up
```
```
score_6, score_5, score_4
```

**`AGE_SAFETY_NEGATIVE`** — 任何 tier 都不會移除，內容本身也被 `tests/test_safety_invariants.py` 釘住：
```
child, children, kid, minor, teen, teenager, underage, young girl
```

**`QUALITY_NEGATIVE`** — 畫質加重複人物。後半段的多人負面詞是後來補的：ControlNet 骨架在畫布留白處
常常補出第二個人（跪姿骨架最明顯），只靠 `bad anatomy` 擋不掉：
```
lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd
```

**`SAFE_SAFETY_NEGATIVE`**（tier `safe`）＝ 露骨詞加 `AGE_SAFETY_NEGATIVE` 加 `QUALITY_NEGATIVE`：
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd
```

**`SUGGESTIVE_NEGATIVE`**（tier `suggestive`）＝ 只封鎖露骨部分，年齡與畫質段完全一樣：
```
exposed genitalia, exposed vulva, exposed penis, exposed nipples, sexual intercourse, penetration, pornographic, explicit sexual act, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd
```

**`NEGATIVE_PROMPT`** — 資料集流程（`gen_anchors`／`gen_variations`）直接用這個完整字串：
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

**`SD15_GENDER_WEIGHT` = 1.3** — SD1.5 與 AnimateDiff 把性別寫成 `(man:1.3)`。男性角色的外觀描述偏柔和，SD1.5 常把他們畫成女生；實測加權後從 10/20 變成 19/20 正確。

**`DEFAULT_CUSTOM_CHECKPOINT` = `cyberrealistic_pony`** — GUI／API／CLI `custom` 的預設；
資料集流程另外固定用 `juggernaut_xl_v9_photo.safetensors`（`client.CHECKPOINT`）。

## 4. 關鍵字詞庫

`gen_variations` 用 `random.Random(base_seed)` 從這些詞庫抽，seed 固定就可重現。
自訂生圖不會自動抽，但這些就是已經在本專案模型上驗證過的措辭，可以直接抄進 prompt。

### 4a. 鏡頭角度

`pick_angle()` 從三個獨立詞庫按機率抽，不是一個扁平清單：

| 詞庫 | 機率 | 解析度 | FaceID 權重 |
|---|---|---|---|
| `BACK_VIEW_ANGLES` | 30% | 832×1216 | 0.3（覆蓋呼叫端） |
| `FULL_BODY_ANGLES` | 35% | 832×1216 | 沿用呼叫端 |
| `CLOSE_ANGLES` | 其餘 | 1024×1024 | 沿用呼叫端 |

背面鏡頭把 FaceID 權重壓到 0.3，是因為背面看不到臉，權重高的話 FaceID 會硬把構圖拗回正面。
全身與背面用直式畫布，是因為正方形沒有足夠垂直空間放下整個身體，會被裁回上半身。

**`CLOSE_ANGLES`**
```
front view portrait
3/4 view portrait
side profile portrait
looking over shoulder
slightly from above
slightly from below
close-up headshot
medium shot upper body
overhead view looking down at camera
high angle view from directly above
```

**`FULL_BODY_ANGLES`**
```
full body shot
full body shot from a distance
full length shot standing
full body shot walking
head-to-toe full body shot
```

**`BACK_VIEW_ANGLES`**
```
back view, from behind
rear view, walking away, seen from behind
3/4 back view, looking back over shoulder from behind
back of head and shoulders, from behind
```

### 4b. 姿勢、服裝、光線、背景

**`POSES`**
```
standing straight
sitting on a chair
leaning against a wall
walking
hands in pockets
arms crossed
hand touching hair
looking directly at camera
looking away from camera
slight smile
sitting on the floor, knees bent
lying on back, relaxed
lying on back, one knee bent, arms above head
lying on side, relaxed, head resting on hand
lying on stomach, propped up on elbows, reading a book
lying on stomach, chin resting on hands, ankles crossed in the air
seated yoga stretch pose
standing yoga tree pose
stretching arms overhead after exercise
jogging pose, mid-stride
```

**`OUTFITS`**
```
white t-shirt and jeans
beige knit sweater
denim jacket over t-shirt
floral summer dress
beige trench coat
cream cardigan
white blouse
casual hoodie
light sweater and skirt
```

**`MALE_OUTFITS`**
```
white t-shirt and jeans
black bomber jacket over t-shirt
denim jacket over t-shirt
casual button-up shirt
beige trench coat
knit sweater
graphic hoodie
casual blazer over t-shirt
cargo pants and hoodie
```

**`LIGHTINGS`**
```
soft natural window light
golden hour sunlight
studio softbox lighting
overcast daylight
warm indoor lighting
soft rim light
```

**`BACKGROUNDS`**
```
plain white studio background
cozy cafe interior
quiet city street
park with trees
bright bedroom interior
outdoor garden
minimalist indoor setting
bookstore interior
```

### 4c. suggestive tier 的詞庫

只有 `gen_variations --tier suggestive` 和 `test-suggestive` 會抽這些。tier 本身只換負面詞，
正面詞要不要用這些字是呼叫端自己決定的。

**`SUGGESTIVE_OUTFITS`**
```
bikini
one-piece swimsuit
sports bra and yoga shorts
off-shoulder top and shorts
tank top and short shorts
camisole and shorts
```

**`MALE_SUGGESTIVE_OUTFITS`**
```
swim trunks
athletic shorts, shirtless
board shorts
tank top and shorts
shirtless, joggers
swim shorts
```

**`SUGGESTIVE_BACKGROUNDS`**
```
beach at sunset
poolside
beach during the day
outdoor shower area
tropical resort
lakeside dock
```

### 4d. 角色身分前綴

`character_base_prompt()` 產生的字串會接在使用者 prompt 前面：

```
{trigger}, {age} year old adult {gender}, east asian, {appearance}, {style}
```

11 個角色（全部東亞，全部不小於 `MINIMUM_AGE` = 18，不符合會在 import 時直接 raise）：

| trigger | 年齡 | 性別 | 外觀 | 風格 |
|---|---|---|---|---|
| `mylora` | 28 | woman | long straight black hair, dark brown eyes, oval face, mature adult features, natural healthy build | girl-next-door style, casual outfit |
| `mei` | 19 | woman | long straight jet-black hair with blunt bangs, bright round eyes, soft round face, petite build | casual campus style, oversized hoodie and pleated skirt, sneakers |
| `xinyi` | 21 | woman | shoulder-length wavy chestnut brown hair, almond eyes, heart-shaped face, slim build | modern minimalist chic, tailored blazer and trousers |
| `ruoxi` | 23 | woman | long layered hair with soft curls, dark brown eyes, oval face, athletic build | trendy streetwear, cropped jacket and jeans |
| `yuqing` | 25 | woman | sleek short bob haircut, sharp eyes, angular face, tall slender build | elegant office chic, fitted blouse and skirt |
| `wanling` | 18 | woman | high ponytail, big expressive eyes, youthful round face, petite build | y2k fashion, colorful crop top and cargo pants |
| `minjun` | 18 | man | tousled black hair, warm friendly eyes, soft face, slim build | Taiwanese street style, oversized hoodie and cargo pants |
| `junho` | 20 | man | undercut hairstyle, calm reserved eyes, angular face, lean build | Japanese minimalist style, monochrome knit sweater and slim trousers |
| `taeoh` | 22 | man | soft permed hair, cool detached eyes, refined face, toned build | Korean urban style, oversized shirt and black trousers |
| `hyunjun` | 24 | man | tweed cut hairstyle, deep-set eyes, sharp jawline, athletic build | Japanese minimalist elegance, cream sweater and dark grey trousers |
| `jungi` | 25 | man | slicked side part hair, confident sharp eyes, angular jaw, fit build | Taiwanese flashy streetwear, bold surf-brand jacket and fitted pants |

展開後的完整前綴（沒有性別加權，也就是 SDXL／Pony／Z-Image／Wan 看到的樣子）：

```
mylora, 28 year old adult woman, east asian, long straight black hair, dark brown eyes, oval face, mature adult features, natural healthy build, girl-next-door style, casual outfit
mei, 19 year old adult woman, east asian, long straight jet-black hair with blunt bangs, bright round eyes, soft round face, petite build, casual campus style, oversized hoodie and pleated skirt, sneakers
xinyi, 21 year old adult woman, east asian, shoulder-length wavy chestnut brown hair, almond eyes, heart-shaped face, slim build, modern minimalist chic, tailored blazer and trousers
ruoxi, 23 year old adult woman, east asian, long layered hair with soft curls, dark brown eyes, oval face, athletic build, trendy streetwear, cropped jacket and jeans
yuqing, 25 year old adult woman, east asian, sleek short bob haircut, sharp eyes, angular face, tall slender build, elegant office chic, fitted blouse and skirt
wanling, 18 year old adult woman, east asian, high ponytail, big expressive eyes, youthful round face, petite build, y2k fashion, colorful crop top and cargo pants
minjun, 18 year old adult man, east asian, tousled black hair, warm friendly eyes, soft face, slim build, Taiwanese street style, oversized hoodie and cargo pants
junho, 20 year old adult man, east asian, undercut hairstyle, calm reserved eyes, angular face, lean build, Japanese minimalist style, monochrome knit sweater and slim trousers
taeoh, 22 year old adult man, east asian, soft permed hair, cool detached eyes, refined face, toned build, Korean urban style, oversized shirt and black trousers
hyunjun, 24 year old adult man, east asian, tweed cut hairstyle, deep-set eyes, sharp jawline, athletic build, Japanese minimalist elegance, cream sweater and dark grey trousers
jungi, 25 year old adult man, east asian, slicked side part hair, confident sharp eyes, angular jaw, fit build, Taiwanese flashy streetwear, bold surf-brand jacket and fitted pants
```

> `build_variation_prompt()`（資料集流程）**不含 `style` 欄位**，只有 `character_base_prompt()`
> 這條（`custom`／picker／影片）是完整的。兩邊的身分文字因此不完全相同。

### 4e. 場景庫

`training/scenes/scenes.json`，picker 分頁的縮圖就是這些。`scene_text()` 送出的是 prompt 加 lighting：

| slug | 中文名 | prompt | lighting | tier |
|---|---|---|---|---|
| `studio_white` | 攝影棚白背景 | plain white studio background | studio softbox lighting | safe |
| `cozy_cafe` | 溫馨咖啡廳 | cozy cafe interior | warm indoor lighting | safe |
| `city_street` | 城市街道 | quiet city street | overcast daylight | safe |
| `park_trees` | 公園綠地 | park with trees | golden hour sunlight | safe |
| `bright_bedroom` | 明亮臥室 | bright bedroom interior | soft natural window light | safe |
| `outdoor_garden` | 戶外花園 | outdoor garden | overcast daylight | safe |
| `minimalist_room` | 極簡室內 | minimalist indoor setting | soft rim light | safe |
| `bookstore` | 書店 | bookstore interior | warm indoor lighting | safe |
| `night_neon_street` | 夜晚霓虹街 | city street at night, neon signs | soft rim light | safe |
| `rooftop_sunset` | 天台夕陽 | rooftop terrace overlooking the city | golden hour sunlight | safe |
| `beach_sunset` | 夕陽海灘 | beach at sunset | golden hour sunlight | suggestive |
| `poolside` | 泳池畔 | poolside | golden hour sunlight | suggestive |
| `beach_day` | 白天海灘 | beach during the day | overcast daylight | suggestive |
| `tropical_resort` | 熱帶度假村 | tropical resort | golden hour sunlight | suggestive |
| `lakeside_dock` | 湖畔木棧道 | lakeside dock | overcast daylight | suggestive |

### 4f. 姿勢骨架庫

`training/poses/*.json`，共 31 組。有 ControlNet 的模型（SDXL／Pony）吃骨架 PNG，
`prompt_hint` 只是附帶；**沒有 ControlNet 的模型（Z-Image／SD1.5）只剩文字**，這時 picker 會把
`prompt_hint` 和 `camera` 都接進 prompt，並在介面上明講已經降級。

| slug | camera | prompt_hint |
|---|---|---|
| `arms_crossed` | front view, eye level | standing, arms crossed over the chest |
| `bending_forward_from_the_waist` | side view, eye level | side view, bending forward from the waist with the arms hanging down |
| `hand_touching_hair` | front view, eye level | standing, one hand raised to touch the hair |
| `hands_in_pockets` | front view, eye level | standing, both hands in the trouser pockets |
| `hands_on_hips` | front view, eye level | standing with both hands on the hips, elbows out |
| `jogging_pose_mid_stride` | 3/4 view, eye level | jogging mid-stride, one knee lifted, elbows bent |
| `jumping_both_feet_off_the_ground` | front view, eye level | jumping in mid-air with both feet off the ground, arms raised |
| `kneeling_on_one_knee` | front view, eye level | kneeling on one knee, the other knee raised in front |
| `kneeling_sitting_back_on_heels` | front view, eye level | kneeling on the floor, sitting back on the heels, upright torso |
| `leaning_against_a_wall` | 3/4 view, eye level | leaning against a wall, one arm raised overhead against it |
| `leaning_forward_hands_on_a_table` | three-quarter view, eye level | leaning forward with both hands resting on a table top |
| `looking_over_shoulder_back_to_camera` | back view, eye level | seen from behind, looking back over one shoulder toward the camera |
| `lying_on_back_knees_up_feet_flat_on_the_floor` | high angle, from above | high angle shot from above, lying on the back with the knees drawn up and feet flat |
| `lying_on_back_one_knee_bent_arms_above_head` | high angle, from above | high angle shot from above, lying flat on the back, arms stretched above the head, one knee raised |
| `lying_on_back_relaxed` | high angle, from above | high angle shot from above, lying flat on the floor, seen from directly overhead |
| `lying_on_side_relaxed_head_resting_on_hand` | side view, camera at floor level | side view, camera at floor level, lying on one side, head propped on one hand |
| `lying_on_stomach_chin_resting_on_hands_ankles_crossed_in_the` | front view, camera at floor level | front view, camera at floor level, lying on the stomach, chin resting on hands, feet raised in the air behind |
| `lying_on_stomach_propped_up_on_elbows_reading_a_book` | front view, camera at floor level | front view, camera at floor level, lying on the stomach propped up on the elbows |
| `one_arm_raised_waving` | front view, eye level | standing and waving with one arm raised above the head |
| `reclining_on_one_elbow_legs_extended` | side view, camera at floor level | side view at floor level, reclining on one elbow with the legs stretched out |
| `seated_yoga_stretch_pose` | side view, eye level | seated on the floor, legs extended forward, leaning forward reaching toward the toes |
| `sitting_cross_legged_on_the_floor` | front view, eye level | sitting cross-legged on the floor, legs folded, upright torso |
| `sitting_on_a_chair` | side view, eye level | sitting on a chair, knees bent, hands resting on the lap |
| `sitting_on_steps_elbows_on_knees` | front view, eye level | sitting on a low step with the elbows resting on the knees |
| `sitting_on_the_floor_knees_bent` | 3/4 view, eye level | sitting on the floor, knees bent, one hand on the floor behind for support |
| `sitting_on_the_floor_legs_extended_leaning_back_on_hands` | three-quarter view, low angle | sitting on the floor with the legs stretched out, leaning back on both hands |
| `squatting_crouching_on_heels` | front view, low angle | squatting down on the heels, knees drawn up, full body |
| `standing_straight` | front view, eye level | standing upright, arms relaxed at the sides |
| `standing_yoga_tree_pose` | front view, eye level | tree pose, standing on one leg, the other foot resting against the inner thigh |
| `stretching_arms_overhead_after_exercise` | front view, eye level | standing, both arms stretched overhead |
| `walking` | front view, eye level | walking toward the camera, mid-step |

其中 14 組（跪姿、蹲姿、盤腿、側臥、跳躍、揮手等）**只在骨架庫裡、不在 `POSES` 詞庫**：
資料集流程是不掛 ControlNet 隨機抽 `POSES` 的，把需要骨架才畫得對的姿勢放進去只會增加失敗率。

### 4g. 從圖片反推關鍵字

| 工具 | 產出 | 適合 |
|---|---|---|
| `comfyui_client.tag_image()`（WD14 `wd-swinv2-tagger-v3`） | booru 標籤，例如 `1girl, solo, long hair` | Pony 系 |
| `caption_image.caption_image()`（BLIP） | 自然語言句子 | SDXL／SD1.5／Z-Image |

GUI 每個 prompt 欄位下方的「依圖片產生 Prompt」摺疊區就是這兩個按鈕，旁邊還有「翻譯成英文」。

## 5. 各家族 Prompt 範例

下面每組都是**管線實際送給 ComfyUI 的完整字串**，由 `tests/test_prompt_guide.py` 重新呼叫組裝函式驗證。
在 GUI／CLI 裡你只要打 prompt 那一段；要在 ComfyUI 網頁手動重現才需要整段複製。

### 5a. SDXL（`juggernaut`）、safe、無角色

純自然語言，沒有 score 標籤，尾端接 `REALISTIC_STYLE`。

<!-- example: {"checkpoint": "juggernaut", "fn": "build", "id": "sdxl_safe", "prompt": "sitting at a wooden cafe table, holding a ceramic coffee cup, soft window light", "tier": "safe"} -->
**Positive**
```
sitting at a wooden cafe table, holding a ceramic coffee cup, soft window light, shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```
**Negative**
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

### 5b. Pony（`cyberrealistic_pony`）、safe、角色 `mei`

比 5a 多了 score 標籤和身分前綴，負面詞前面多了 `score_6, score_5, score_4`。

<!-- example: {"checkpoint": "cyberrealistic_pony", "fn": "build", "id": "pony_safe", "prompt": "sitting at a wooden cafe table, holding a ceramic coffee cup, soft window light", "tier": "safe", "trigger": "mei"} -->
**Positive**
```
score_9, score_8_up, score_7_up, mei, 19 year old adult woman, east asian, long straight jet-black hair with blunt bangs, bright round eyes, soft round face, petite build, casual campus style, oversized hoodie and pleated skirt, sneakers, sitting at a wooden cafe table, holding a ceramic coffee cup, soft window light, shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```
**Negative**
```
score_6, score_5, score_4, nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

### 5c. Pony（`pony_realism`）、suggestive、角色 `xinyi`

跟 5b 唯一的差別是負面詞從 `SAFE_SAFETY_NEGATIVE` 換成 `SUGGESTIVE_NEGATIVE`。正面詞寫法不變，年齡保護詞照樣在。

<!-- example: {"checkpoint": "pony_realism", "fn": "build", "id": "pony_suggestive", "prompt": "standing on the beach at sunset, wearing a one-piece swimsuit", "tier": "suggestive", "trigger": "xinyi"} -->
**Positive**
```
score_9, score_8_up, score_7_up, xinyi, 21 year old adult woman, east asian, shoulder-length wavy chestnut brown hair, almond eyes, heart-shaped face, slim build, modern minimalist chic, tailored blazer and trousers, standing on the beach at sunset, wearing a one-piece swimsuit, shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```
**Negative**
```
score_6, score_5, score_4, exposed genitalia, exposed vulva, exposed penis, exposed nipples, sexual intercourse, penetration, pornographic, explicit sexual act, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

### 5d. SD1.5（`realistic_vision`）、safe、男性角色 `minjun`

注意 `adult (man:1.3)`：這是 SD1.5 專屬的性別加權，其他家族看不到。

<!-- example: {"checkpoint": "realistic_vision", "fn": "build", "id": "sd15_male", "prompt": "walking on a quiet city street, overcast daylight", "tier": "safe", "trigger": "minjun"} -->
**Positive**
```
minjun, 18 year old adult (man:1.3), east asian, tousled black hair, warm friendly eyes, soft face, slim build, Taiwanese street style, oversized hoodie and cargo pants, walking on a quiet city street, overcast daylight, shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```
**Negative**
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

### 5e. Z-Image Turbo、safe、角色 `ruoxi`（英文）

沒有 score 標籤也沒有性別加權，組裝方式跟 SDXL 一樣。但 Z-Image 沒有 FaceID，身分只靠這段文字描述，臉不會被鎖住。

<!-- example: {"checkpoint": "z_image_turbo", "fn": "build", "id": "zimage_en", "prompt": "standing in a bookstore, reading a book, warm indoor lighting", "tier": "safe", "trigger": "ruoxi"} -->
**Positive**
```
ruoxi, 23 year old adult woman, east asian, long layered hair with soft curls, dark brown eyes, oval face, athletic build, trendy streetwear, cropped jacket and jeans, standing in a bookstore, reading a book, warm indoor lighting, shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```
**Negative**
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

### 5f. Z-Image Turbo、safe、中文 prompt

Z-Image 的文字編碼器讀得懂中文，不用先翻譯。但**風格詞和負面詞仍然是英文接在後面**，因為那是管線的預設值，不是 Z-Image 專屬設定。這就是實際送出的樣子：

<!-- example: {"checkpoint": "z_image_turbo", "fn": "build", "id": "zimage_zh", "prompt": "站在書店裡看書，溫暖的室內燈光", "tier": "safe"} -->
**Positive**
```
站在書店裡看書，溫暖的室內燈光, shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```
**Negative**
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine
```

### 5g. Wan 2.2 影片、safe、角色 `taeoh`

正面詞**沒有** `REALISTIC_STYLE`（那些是 CLIP 時代的標籤，對 umt5 沒有意義），性別也不加權（`(man:1.3)` 是 SD 的 CLIP 語法）。負面詞換成 `WAN_VIDEO_NEGATIVE`。

<!-- example: {"fn": "wan", "id": "wan", "prompt": "slowly turning toward the camera and smiling, gentle breeze moving the hair", "tier": "safe", "trigger": "taeoh"} -->
**Positive**
```
taeoh, 22 year old adult man, east asian, soft permed hair, cool detached eyes, refined face, toned build, Korean urban style, oversized shirt and black trousers, slowly turning toward the camera and smiling, gentle breeze moving the hair
```
**Negative**
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, static image, frozen frame, flickering, jitter, blurry, overexposed, low quality, jpeg artifacts, deformed hands, extra fingers, distorted face, morphing face, melting limbs, subtitles, watermark
```

### 5h. AnimateDiff 影片、safe、男性角色 `jungi`

沒有 score 標籤（`checkpoint=None`），但**有**性別加權，負面詞用 `VIDEO_REALISTIC_NEGATIVE`。

<!-- example: {"fn": "animatediff", "id": "animatediff", "prompt": "walking toward the camera, hands in pockets", "tier": "safe", "trigger": "jungi"} -->
**Positive**
```
jungi, 25 year old adult (man:1.3), east asian, slicked side part hair, confident sharp eyes, angular jaw, fit build, Taiwanese flashy streetwear, bold surf-brand jacket and fitted pants, walking toward the camera, hands in pockets, shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain
```
**Negative**
```
nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text, multiple people, two people, duplicate, twins, extra person, crowd, 3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, perfect skin, digital art, render, unreal engine
```

## 6. 在 ComfyUI 網頁手動測試時

- **不要重複加 score 標籤**：管線已經加過了，上面的字串裡就有。
- **Pony 系要把風格 LoRA 關掉**：`sdxl_photorealistic_slider_v1-0.safetensors` 在每個模板裡固定 2.5，
  是照寫實 SDXL 調的，套在 Pony 上會跟它自己的畫風打架。程式路徑傳 `lora_strength=0.0`。
- **只有 Z-Image 和 Wan 讀得懂中文**，其他全部先翻成英文（GUI 有「翻譯成英文」按鈕，
  `training/translate_prompt.py`）。CLIP tokenizer 對中文的支援差到中文詞多半直接被忽略。
- **SVD 沒有 prompt 欄位**，動作大小靠 `MOTION_BUCKET_ID`（本專案用 15，
  官方預設 127 在人像上會讓臉融掉）。
- **有 ControlNet control-lora 的流程不能用量化版 checkpoint**（會出全黑圖），這個判斷已經在 client 裡。
- **cfg 不要低於 1.5**，理由見 1b。

## 7. 有無 score 標籤的實測對照

RTX 2070、同 seed、純 txt2img（不掛 FaceID，隔離 checkpoint 與 prompt 語法本身的差異）：

| checkpoint | 無標籤 | 有標籤 |
|---|---|---|
| `pony` | 40.3s，明顯插畫／CG 感 | 30.3s，寫實度明顯提升 |
| `cyberrealistic_pony` | 38.3s | 28.2s |
| `pony_realism` | 42.3s | 36.2s |

三個 Pony checkpoint 加標籤後反而都快了 8 到 10 秒（推測取樣器更早收斂到清晰結構），
但選擇加標籤的理由是畫質、不是速度。重現：

```
ComfyUI\.venv\Scripts\python.exe training\model_prompt_test.py
```

輸出在 `training/reference_candidates/model_test/`（`REPORT.md` 是表格，`contact_sheet.png` 是對照合成圖）。

## 8. 維護

改了常數、詞庫、角色、場景、骨架 metadata 之後：

```
powershell -ExecutionPolicy Bypass -File check.ps1
```

`tests/test_prompt_guide.py` 會指出這份文件哪一段跟程式對不上，範例的失敗訊息會直接印出可以貼回來的字串。
