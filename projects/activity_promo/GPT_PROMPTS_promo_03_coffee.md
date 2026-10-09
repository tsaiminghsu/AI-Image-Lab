# ChatGPT 生圖提示詞：一起去喝咖啡（promo_03_coffee）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_03_coffee/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, cream knit cardigan over a white t-shirt, blue jeans, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：午後的街上，她回頭

- 時段：afternoon, golden｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third facing the viewer, the café window on the right. The woman stands on a quiet side street in the afternoon, turned toward the right where a small café glows, looking back at the viewer with a relaxed smile.
Setting: a quiet side street in the afternoon, a small café with a large glass window on the right, glowing warmly, a wooden door beside it, clean pavement with nobody else on it. Lighting: soft golden afternoon sunlight from the left, the café window glowing warmly, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred facing the viewer, the café window on the right. The woman stands on a quiet side street in the afternoon, turned toward the right where a small café glows, looking back at the viewer with a relaxed smile.
Setting: a quiet side street in the afternoon, a small café with a large glass window on the right, glowing warmly, a wooden door beside it, clean pavement with nobody else on it. Lighting: soft golden afternoon sunlight from the left, the café window glowing warmly, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：咖啡廳門口，她驚喜

- 時段：afternoon, golden｜手上：none｜景別：close
- 接上一鏡：from the street to the café door: the same light, she has turned to face you

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, her hand on the door handle, the café window on the right. The woman stands at the wooden door of a small café with its glass window on the right of the frame, turning to the viewer with a delighted, surprised smile and one hand on the door handle.
Setting: the wooden door of a small café with a large glass window on the right, afternoon light, clean pavement with nobody else on it. Lighting: soft golden afternoon sunlight, the café window glowing warmly, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, her hand on the door handle, the café window on the right. The woman stands at the wooden door of a small café with its glass window on the right of the frame, turning to the viewer with a delighted, surprised smile and one hand on the door handle.
Setting: the wooden door of a small café with a large glass window on the right, afternoon light, clean pavement with nobody else on it. Lighting: soft golden afternoon sunlight, the café window glowing warmly, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：櫃台前，她等你點

- 時段：late afternoon｜手上：none｜景別：medium
- 接上一鏡：from the door to the counter: the same afternoon, the light one step later, now indoors

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third at the counter, the large window on the right. The woman stands at the café counter with her hands clasped in front of her, waiting with an amused smile, a chalkboard menu with no readable text on the left and the large window on the right.
Setting: the counter of a small café in the late afternoon, a chalkboard menu with no readable text on the left, the large glass window on the right, a warm wooden interior. Lighting: warm café lighting and soft late-afternoon light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred at the counter, the large window on the right. The woman stands at the café counter with her hands clasped in front of her, waiting with an amused smile, a chalkboard menu with no readable text on the left and the large window on the right.
Setting: the counter of a small café in the late afternoon, a chalkboard menu with no readable text on the left, the large glass window on the right, a warm wooden interior. Lighting: warm café lighting and soft late-afternoon light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：窗邊座位，她捧著杯子

- 時段：late afternoon｜手上：a coffee cup (shots 4-6)｜景別：close
- 接上一鏡：from the counter to the window seat: the same café, the light one step later

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, the coffee cup held in both hands, the window on the right. The woman sits at a window seat holding a coffee cup in both hands, looking out at the street through the large window on the right, then turning to the viewer with a warm smile.
Setting: a window seat in a small café in the late afternoon, the large glass window on the right showing the street, a small wooden table with a saucer. Lighting: soft late-afternoon light from the window on the right on her face, warm café lighting behind.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, the coffee cup held in both hands, the window on the right. The woman sits at a window seat holding a coffee cup in both hands, looking out at the street through the large window on the right, then turning to the viewer with a warm smile.
Setting: a window seat in a small café in the late afternoon, the large glass window on the right showing the street, a small wooden table with a saucer. Lighting: soft late-afternoon light from the window on the right on her face, warm café lighting behind.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：黃昏的窗邊，她認真看你

- 時段：dusk｜手上：a coffee cup (shots 4-6)｜景別：medium
- 接上一鏡：from the afternoon window to dusk: the same seat, the street now blue with the first lights

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third, the window with the dusk street on the right. The woman sets her coffee cup down on its saucer and looks at the viewer sincerely, the large window on the right showing the street at dusk with the first lights switching on.
Setting: a window seat in a small café at dusk, the large glass window on the right showing the street with the first lights switching on, a small wooden table with a coffee cup on a saucer. Lighting: blue dusk light from the window on the right and warm café lighting on her face, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred, the window with the dusk street on the right. The woman sets her coffee cup down on its saucer and looks at the viewer sincerely, the large window on the right showing the street at dusk with the first lights switching on.
Setting: a window seat in a small café at dusk, the large glass window on the right showing the street with the first lights switching on, a small wooden table with a coffee cup on a saucer. Lighting: blue dusk light from the window on the right and warm café lighting on her face, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：夜色的窗光裡，她笑著

- 時段：evening｜手上：a coffee cup (shots 4-6)｜景別：close
- 接上一鏡：from the dusk to the evening: the same seat, the window now blue

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, the coffee cup near her chin, the evening window on the right. The woman holds her coffee cup near her chin and smiles warmly at the viewer, the large window on the right glowing with the evening blue and the first street lights.
Setting: a window seat in a small café in the evening, the large glass window on the right glowing blue with the street lights, a small wooden table. Lighting: soft evening blue from the window on the right and warm café light on her face, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream knit cardigan over a white t-shirt and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, the coffee cup near her chin, the evening window on the right. The woman holds her coffee cup near her chin and smiles warmly at the viewer, the large window on the right glowing with the evening blue and the first street lights.
Setting: a window seat in a small café in the evening, the large glass window on the right glowing blue with the street lights, a small wooden table. Lighting: soft evening blue from the window on the right and warm café light on her face, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

