# 測試提示詞參考

測試和手動驗證用的提示詞集，按 checkpoint 類型組織。每種類型的提示詞都設計用來驗證對應模型的特性。

> 這裡是「手動測試要打什麼」。各模型的關鍵字總表、全部詞庫常數、以及管線**實際送給 ComfyUI 的完整字串**
> 在 [`training/PROMPT_GUIDE.md`](../training/PROMPT_GUIDE.md)（由 `tests/test_prompt_guide.py` 釘住，
> 跟程式不一致就會讓 `check.ps1` 紅）。

## 中文提示詞支援總覽

| Checkpoint 類型 | 中文支援 | 說明 |
|---|---|---|
| Z-Image Turbo | ✅ 原生支援，**不需翻譯** | 底層模型本身理解中文，Prompt 可以直接打中文，連畫面裡的中文字都寫得出來 |
| Pony 系列 / SDXL / SD1.5 / AnimateDiff / GIF | ⚠️ 技術上能輸入，但**建議先翻譯** | CLIP tokenizer 對中文支援不佳，中文 prompt 容易被模型忽略；GUI 每個 prompt 欄位下方都有「翻譯成英文」按鈕（呼叫 Google Translate 公開端點，`training/translate_prompt.py`），自動偵測來源語言，已經是英文的話按下去內容不變 |
| SVD 圖生影片 | 不適用 | 沒有 prompt 欄位，靠已有圖片 + 動作強度滑桿控制，中文/英文都無影響 |

**實測建議**：測試「中文 prompt 直接可用」時只需要測 Z-Image；測其他路徑的中文輸入時，重點是驗證「翻譯成英文」按鈕本身有沒有正常運作，而不是驗證中文被模型正確理解（CLIP 本來就理解不了）。

## Checkpoint 類型

### 1. Pony 系列 (SDXL-shaped)

**Checkpoints**: `pony`, `cyberrealistic_pony`, `pony_realism`

**自動前綴**: `score_9, score_8_up, score_7_up`（質量等級，不用手動加）

**特性**:
- 用 booru 標籤訓練。自動前綴只是**品質標籤**，跟「能不能打自然語言」無關，兩件事要分開看：
  - **場景、服裝、光線、畫風用自然語言可以**——實測用同一句自然語言 prompt 跑六個 checkpoint，
    Pony 系加了 score 標籤後寫實度明顯提升（見 `training/PROMPT_GUIDE.md` 第 7 節）
  - **姿勢用自然語言不可靠**——實測 5 個 `lying on ...` 標籤在 Pony 系上全部塌成坐姿；
    4×3×2 矩陣裡 12 個純文字格子有 6 個失敗，`pony` 和 `pony_realism` 在跪姿還會各生出
    兩個人。姿勢要用 ControlNet 骨架庫（`--pose`、GUI 下拉），見 README「ControlNet 姿勢骨架庫」
  - 想要原生的 booru 標籤，用 GUI 的「產生標籤（WD14，Pony 系適用）」從參考圖反推
- SDXL 級別的 VRAM 需求 (5+ GB)
- 支援 FaceID (IP-Adapter) 和 HQ 兩段式

**推薦測試提示詞**:

#### 簡單場景
```
a photo of a scene
standing in a park on a sunny day
sitting by a window with coffee
```

#### 有角色的提示詞
```
portrait of a woman, professional headshot
a woman wearing a red dress
a man in casual streetwear
woman sitting on a couch, indoor lighting
```

#### 複雜場景
```
full body portrait, woman in elegant office attire, sitting at desk, morning light through window, professional photography
a man standing in urban street at dusk, casual oversized hoodie, film noir lighting, candid photograph
woman in cozy coffee shop, warm afternoon light, reading book, intimate portrait, depth of field
```

#### 無角色場景
```
a scene, no character
landscape photography, mountain range at sunset
modern minimalist interior design
```

#### 中文 prompt（⚠️ 建議先按「翻譯成英文」再送出）
```
一個女人穿著紅色洋裝的肖像
坐在咖啡廳裡看書的女人
公園裡陽光明媚的一天
```

---

### 2. SDXL (非 Pony 系列)

**Checkpoints**: `juggernaut`

**特性**:
- 標準 SDXL，自然語言 prompt（不需要 booru tag 前綴）
- 不需要 quality tag
- VRAM 需求同 Pony

**推薦測試提示詞**:

#### 簡單場景
```
a photo of a scene
portrait photo
woman in professional clothing
```

#### 有角色
```
portrait of a person in casual outfit
a person standing outdoors
woman sitting at a cafe table
```

#### 複雜場景
```
professional portrait photography, woman in elegant black dress, studio lighting, sharp focus, high detail
person in modern minimalist interior, natural window light, cinematic composition
```

#### 中文 prompt（⚠️ 建議先按「翻譯成英文」再送出）
```
穿著休閒服裝站在戶外的人
坐在咖啡廳桌前的女人
```

---

### 3. SD1.5 真實感

**Checkpoints**: `realistic_vision`, `cyberrealistic`

**特性**:
- SD1.5 架構，與 SDXL workflow 完全不同
- 原生解析度約 512×768（不是 1024×1024）
- **男性角色容易被渲染成女性**（軟性描述會強行推向女性）
  - 需要 `(man:1.3)` 或明確的 `masculine` 術語
  - 自動加上 `SD15_GENDER_WEIGHT = 1.3` 來穩定性別
- 沒有 ControlNet 骨架控制支援
- 沒有 HQ 兩段式、沒有手部精修

**推薦測試提示詞**:

#### 簡單場景
```
portrait photo
headshot of a person
photograph of a scene
```

#### 有角色（女性）
```
portrait of a young woman, casual outdoor setting
woman in summer dress, natural light
professional headshot of a woman
```

#### 有角色（男性）- 需要性別權重
```
portrait of a man, professional photography
man in casual hoodie, warm lighting
young man, friendly expression, outdoor setting
```

#### 複雜場景
```
professional headshot, natural lighting, soft focus background, documentary photography style
person in cozy home setting, warm ambient light, intimate portrait
```

#### 中文 prompt（⚠️ 建議先按「翻譯成英文」再送出，男性角色翻譯後仍要留意性別穩定性）
```
一個年輕女人的肖像，戶外自然光
穿著連帽衫的年輕男人，友善的表情
```

---

### 4. Z-Image Turbo

**特性**:
- 純文字生圖，無 FaceID / ControlNet / 精修
- 解析度 1024×1024（`ZIMAGE_WIDTH`/`ZIMAGE_HEIGHT`）、8 步、`res_multistep`/`simple`
- **在這張 2070 上不快**：cfg 1 約 75 秒，本專案用的 cfg 2 約 155-195 秒（見 README 的
  Z-Image 實測表）。步數少不代表快，每步約 9 秒
- **唯一原生理解中文的路徑，Prompt 可以直接打中文，不用翻譯**（其他所有 checkpoint 都建議先翻譯，見文檔開頭「中文提示詞支援總覽」）
- 手部和文字生成品質比 SDXL 好，中文字也寫得出來

**推薦測試提示詞**:

#### 簡單場景
```
a photo of a person
portrait photography
person in casual outfit
```

#### 有角色
```
portrait of a young woman with long black hair
man with undercut hairstyle, friendly smile
woman in minimalist office chic outfit
```

#### 含文字的場景（Z-Image 的強項）
```
a poster with large bold text "Hello World" and a person
person holding a sign with text
text-heavy scene with person in foreground
```

#### 中文 prompt（✅ 直接打中文送出，不用按翻譯）
```
穿著紅色裙子的女人的肖像
穿著休閒服裝的男人
舒適的咖啡館內景
```

#### 中文字渲染測試（Z-Image 專屬強項，其他 checkpoint 畫面裡的文字幾乎都是亂碼）
```
一張海報，上面用大字寫著「歡迎光臨」，背景有一個人
一個人拿著寫有「生日快樂」的牌子
街道上的招牌寫著「早安咖啡」，霓虹燈風格
```

---

### 5. 影片生成 (AnimateDiff)

**Checkpoints**: `sd15_base`, `realistic_vision`

**特性**:
- SD1.5 base (動作模組訓練上限 16 幀)
- 同樣的男性性別倒退問題（會自動加 `SD15_GENDER_WEIGHT`）
- 影片不用負面詞 "symmetrical face"（會導致臉部幀間晃動）
- 支援 FaceID 鎖臉
- LCM 模式會加快但男性角色有性別倒退風險

**推薦測試提示詞**:

#### 簡單動態
```
a scene description
person standing still
turning head slowly
```

#### 有角色的動態
```
woman sitting by a window, turning to look at camera
man standing in a room, looking around
woman walking through a park
```

#### 複雜動作序列
```
woman sitting on a couch, looking at camera, slight smile, then turning head to the side, soft natural light
man in office setting, reaching for coffee cup, then taking a sip, window light in background
```

#### 無角色場景
```
camera pans across a room
sunset light moving across a wall
water flowing in a stream
```

#### 中文 prompt（⚠️ 建議先按「翻譯成英文」再送出）
```
女人坐在窗邊，轉頭看向鏡頭
男人站在房間裡，環顧四周
```

---

### 6. SVD 圖生影片

**特性**:
- 直接讓一張既有圖片動起來
- 沒有 FaceID 鎖臉，動作幅度大時臉容易變形
- 低解析度、低幀數，主要用來驗證構圖
- 提示詞影響較小（主要是改動作）
- **沒有 prompt 欄位，不涉及中文/英文問題**

**推薦測試提示詞**:

#### 簡單動作
```
slight motion
gentle movement
subtle animation
```

#### 有對象的動作
```
person standing, slight head turn
woman smiling, gentle sway
man looking around the room
```

#### 動作強度調整（`motion_bucket_id`）
```
低強度 (15-30): gentle sway, subtle movement
中強度 (40-60): person walking, turning around
高強度 (80-127): dramatic motion, spinning (⚠️ 人像容易變形)
```

---

### 7. 圖片選擇生圖分頁（點縮圖，不打字）

**分頁**：GUI 的「🎯 圖片選擇生圖」，跟前面 6 種不一樣——這裡**不是測 prompt 文字**，是測「點選組合」在不同
checkpoint 家族下會不會被正確翻譯。核心邏輯在 `gc.plan_picker()`（`generate_character.py`），對應的自動化測試在
`tests/test_picker_plan.py`（42 個案例，離線）。手動測試時真正要核對的是**預覽欄的文字**跟**降級提示有沒有出現**，
而不是最後生成的畫面像不像——像不像是另一件事，這裡先確認「送對東西」。

**測試矩陣**：人物 × 姿勢 × 場景，各家族至少跑一組

| Checkpoint 家族 | 人物 | 姿勢 | 場景 | 預期行為 |
|---|---|---|---|---|
| Pony（`cyberrealistic_pony`） | xinyi | hands on hips | 溫馨咖啡廳 | 預覽以 `score_9, score_8_up, score_7_up` 開頭；提示區顯示「✅ 這個 checkpoint 會真的鎖住：臉（FaceID anchor）、姿勢（ControlNet 骨架）」；姿勢的 `prompt_hint` 只出現在預覽（`_plan_custom` 加的），不在點選組出的 `prompt_body` 裡 |
| SDXL（`juggernaut`） | ruoxi | walking | 城市街道 | 同上但沒有 `score_9` 標籤 |
| SD1.5（`realistic_vision`） | minjun（男性） | arms crossed | 公園綠地 | 預覽含 `(man:1.3)`；提示區出現兩則「⚠️ realistic_vision 沒有 ControlNet／FaceID」；anchor/pose_name 都是 None |
| Z-Image（`z_image_turbo`） | mei | standing straight | 書店 | 預覽含姿勢的 `prompt_hint` + `camera` 文字（例如「standing upright, arms relaxed at the sides, front view, eye level」）；同樣兩則降級警告 |

**邊界案例**（對應 `test_picker_plan.py` 的具體測試名）：

```
只選人物，不選姿勢/場景 → 生成按鈕應該擋下（"至少要選一個場景或姿勢，或在「補充描述」填點東西"）
角色沒有 anchor 圖（例如 mylora）+ SDXL → 提示「還沒有 anchor 圖」，不能送出
只填補充描述，不選任何縮圖 → 可以生成（純文字路徑）
suggestive 分級下選海灘場景 → 安全負面詞（AGE_SAFETY_NEGATIVE、pornographic 等）仍要出現在預覽的負面欄
切回 safe 分級 → 場景選取要被清空（不能讓已選的海灘場景「隱形地」留著送出去）
```

**中文補充描述**：「補充描述」欄位可以直接打中文，送出前會自動翻成英文（複用
`translate_prompt.translate_to_english`，見「中文提示詞支援總覽」的 Google→MyMemory 備援鏈）。手動測試時可以打
「手上拿著紙杯」，確認生成按鈕按下去後有跑翻譯（不用先按翻譯鍵）。

**已知的資料完整性檢查**（避免縮圖庫本身跟測試脫節）：

```bash
ComfyUI\.venv\Scripts\python.exe training\scene_library.py list      # 場景庫 + 縮圖有沒有缺
python -m pytest tests/test_scene_library.py tests/test_picker_plan.py tests/test_pose_library_meta.py -q
```

---

## 使用指南

### 測試場景的選擇

1. **快速冒煙測試**: 用簡單提示詞 (1 個 seed)
   ```bash
   ComfyUI\.venv\Scripts\python.exe training\generate_character.py custom \
     --checkpoint cyberrealistic_pony \
     --prompt "a photo of a scene" \
     --seed 9000
   ```

2. **模型對比**: 用相同提示詞跑所有 checkpoint
   ```bash
   for checkpoint in juggernaut cyberrealistic_pony realistic_vision; do
     # ... run with same prompt, 3-5 seeds
   done
   ```

3. **角色驗證**: 用每個角色生成，確認身分穩定
   ```bash
   ComfyUI\.venv\Scripts\python.exe training\generate_character.py custom \
     --character mei \
     --prompt "portrait, professional headshot" \
     --seed 9000
   ```

4. **性別穩定性（SD1.5 專用）**: 男女角色各 3-5 seed
   ```bash
   # 男性角色測試 - SD1.5 應該通過自動的 SD15_GENDER_WEIGHT
   ComfyUI\.venv\Scripts\python.exe training\generate_character.py custom \
     --character minjun \
     --checkpoint realistic_vision \
     --prompt "portrait photo" \
     --seed 9000
   ```

### 提示詞調整時的注意事項

- **Pony**: 不要手動加 `score_9` 前綴（會被重複加）。場景/服裝/畫風寫自然語言沒問題，但**姿勢別指望文字**，
  要用 ControlNet 骨架庫（見上面第 1 節）
- **SD1.5 男性**: 確保用的是 `realistic_vision` 或 `cyberrealistic` checkpoint（自動應用 gender weight）
- **影片**: 避免在負面詞裡用 "symmetrical face"（已自動移除）
- **中文 prompt**: 只有 Z-Image Turbo 原生理解中文，不用翻譯；其他所有 checkpoint（Pony/SDXL/SD1.5/AnimateDiff/GIF）都要先用「翻譯成英文」按鈕轉換，直接打中文送出會被 CLIP 忽略掉大半內容
- **SVD**: 提示詞影響較小，動作強度靠 `motion_bucket_id` 滑桿控制，沒有語言問題
- **圖片選擇生圖**: 不用打字，但選了角色卻沒 anchor 圖、或只選人物沒選姿勢/場景時會被擋下——這些是設計行為，
  不是 bug；驗證重點是預覽欄文字有沒有跟著 checkpoint 換，不是打字技巧

---

## 檔案位置

- **角色定義**: `training/generate_character.py` 的 `CHARACTERS` 字典
- **風格常數**: `training/generate_character.py` 的 `REALISTIC_STYLE`／`REALISTIC_NEGATIVE`／
  `SAFE_SAFETY_NEGATIVE` 等（逐字內容見 `training/PROMPT_GUIDE.md` 第 3 節）
- **測試案例**: `tests/test_prompt_assembly.py` - 自動化測試用例
- **Checkpoint 映射**: `training/comfyui_client.py` 中的 `CHECKPOINTS`、`PONY_CHECKPOINTS`、`SD15_CHECKPOINTS` 等
- **翻譯功能**: `training/translate_prompt.py`（Google Translate 公開端點，GUI 各 prompt 欄位下方的「翻譯成英文」按鈕呼叫這裡）
- **圖片選擇生圖**: `training/generate_character.py` 中的 `plan_picker()`／`picker_anchor_path()`；場景庫在
  `training/scene_library.py` + `training/scenes/scenes.json`；姿勢庫在 `training/pose_skeletons.py` +
  `training/poses/*.json`；測試在 `tests/test_picker_plan.py`、`tests/test_scene_library.py`、
  `tests/test_pose_library_meta.py`

---

## 更新記錄

- **2026-09-12**: 初版，包含 6 種 checkpoint 類型的提示詞集
- **2026-09-12**: 加入「中文提示詞支援總覽」，並在每個 checkpoint 章節補上中文範例——修正了 Z-Image 段落原本「支援中文 prompt（會自動翻譯）」的錯誤描述（Z-Image 其實是唯一**不需要**翻譯的路徑，其他 checkpoint 才需要翻譯）
- **2026-09-18**: 連到新的 `training/PROMPT_GUIDE.md`；修正 Z-Image 段落的解析度（1024×1024，不是
  512×768）和速度（cfg 2 約 155-195 秒，不是 3-4 秒）；行號引用改成符號名稱；修正 Pony 段落把
  「自動加 score 前綴」跟「可以打自然語言」當成因果的說法——前綴只是品質標籤，而且姿勢用文字在
  Pony 系上實測不可靠，要走骨架庫
- **2026-09-12**: 加入「圖片選擇生圖分頁」章節——這個分頁測的是「點選組合」而非文字，附測試矩陣（人物×姿勢×場景 ×
  4 個 checkpoint 家族）跟邊界案例清單，對應到 `test_picker_plan.py` 的離線測試
