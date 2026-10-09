# ChatGPT 生圖提示詞：一起下廚（promo_02_cooking）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_02_cooking/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, cream high-neck knit top with the sleeves pushed up, a beige linen apron, blue jeans, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：廚房，她綁上圍裙

- 時段：early evening, warm window light｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third facing the viewer, the stove with a pot on the right. The woman stands at the kitchen counter tying the strings of a beige linen apron behind her back and looks up at the viewer with a bright smile, the stove on the right of the frame.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, a window on the left with warm early-evening light. Lighting: warm golden window light from the left, a warm ceiling light, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred facing the viewer, the stove with a pot on the right. The woman stands at the kitchen counter tying the strings of a beige linen apron behind her back and looks up at the viewer with a bright smile, the stove on the right of the frame.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, a window on the left with warm early-evening light. Lighting: warm golden window light from the left, a warm ceiling light, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：冰箱前，她舉起馬鈴薯

- 時段：early evening, warm window light｜手上：a potato (shots 2-3)｜景別：close
- 接上一鏡：from the fridge to the counter: the same kitchen and light, the potato now in her hand

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, the potato held up near her face, the open fridge and the stove on the right. The woman holds up a potato toward the viewer with one hand, an open fridge glowing on the right of the frame and the stove beyond it, her face lit with an excited smile.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, an open fridge glowing on the right beside the stove, early-evening window light on the left. Lighting: warm window light from the left and the cool glow of the open fridge on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, the potato held up beside her cheek, the open fridge and the stove on the right. The woman holds up a potato toward the viewer with one hand, an open fridge glowing on the right of the frame and the stove beyond it, her face lit with an excited smile.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, an open fridge glowing on the right beside the stove, early-evening window light on the left. Lighting: warm window light from the left and the cool glow of the open fridge on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：流理台邊，她陪你煮

- 時段：evening, the sky outside deepening｜手上：a potato (shots 2-3)｜景別：medium
- 接上一鏡：from fetching the potato to waiting beside the counter: the same kitchen, the light one step later

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third leaning on the counter, the stove with a pot on the right. The woman leans on the counter beside a cutting board with a potato resting near her hand, her chin on her other hand, watching the viewer cook with a gentle smile, the stove with a pot on the right of the frame.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, a cutting board with a potato on the counter, the window on the left showing a dusky blue-and-orange sky just after sunset. Lighting: warm ceiling light and the soft dusk light through the window on the left, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred leaning on the counter, the stove with a pot on the right. The woman leans on the counter beside a cutting board with a potato resting near her hand, her chin on her other hand, watching the viewer cook with a gentle smile, the stove with a pot on the right of the frame.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, a cutting board with a potato on the counter, the window on the left showing a dusky blue-and-orange sky just after sunset. Lighting: warm ceiling light and the soft dusk light through the window on the left, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：鍋子冒煙，她笑著喊

- 時段：evening, the window deep blue｜手上：none｜景別：close
- 接上一鏡：from watching the pot to the steam: the same kitchen, the same light, a little steam now rising

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, her hand pointing toward the pot, a curl of steam from the stove on the right. A thin curl of steam rises from the pot on the stove on the right of the frame, and the woman laughs, pointing at the pot with one hand while looking at the viewer.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, a thin curl of steam rising from the pot, no flames visible, the window on the left deep blue. Lighting: warm ceiling light and the soft glow from the stove, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, her hand pointing toward the pot, a curl of steam from the stove on the right. A thin curl of steam rises from the pot on the stove on the right of the frame, and the woman laughs, pointing at the pot with one hand while looking at the viewer.
Setting: a bright home kitchen with a gas stove and a pot on the right side of the frame, a wooden counter, clean uncluttered surfaces, a thin curl of steam rising from the pot, no flames visible, the window on the left deep blue. Lighting: warm ceiling light and the soft glow from the stove, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：餐桌前，她吃第一口

- 時段：night, the window dark blue｜手上：a spoon (shots 5-6)｜景別：medium
- 接上一鏡：from the stove to the table: night now, the pot still visible on the right

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third leaning in with the spoon at her lips, the bowl in front, the stove on the right in the background. The woman sits at the dining table with a steaming bowl of curry rice in front of her, a spoonful of curry rice held at her own lips as she leans in to take the first bite, her eyes on the viewer with a smile, the stove visible beyond the table on the right.
Setting: a dining table in a home kitchen at night with a steaming bowl of curry rice, the stove with a pot visible beyond the table on the right, the window on the left dark blue. Lighting: warm pendant light over the table, her face clearly lit, the window dark blue behind.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred leaning in with the spoon at her lips, the bowl in front, the stove on the right in the background. The woman sits at the dining table with a steaming bowl of curry rice in front of her, a spoonful of curry rice held at her own lips as she leans in to take the first bite, her eyes on the viewer with a smile, the stove visible beyond the table on the right.
Setting: a dining table in a home kitchen at night with a steaming bowl of curry rice, the stove with a pot visible beyond the table on the right, the window on the left dark blue. Lighting: warm pendant light over the table, her face clearly lit, the window dark blue behind.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：餐桌前，她對你笑

- 時段：night, the window dark blue｜手上：a spoon (shots 5-6)｜景別：close
- 接上一鏡：from the first bite to the smile: the same table and light, her mouth now empty

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, the stove softly out of focus on the right. The woman lowers the spoon into the bowl and smiles warmly at the viewer, the stove with its pot softly out of focus on the right of the frame.
Setting: a dining table in a home kitchen at night, the stove with a pot softly out of focus on the right, the window on the left dark blue. Lighting: warm pendant light over the table, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a cream high-neck knit top with a round neckline that covers the collarbone and the sleeves pushed up, a beige linen apron and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, the stove softly out of focus on the right. The woman lowers the spoon into the bowl and smiles warmly at the viewer, the stove with its pot softly out of focus on the right of the frame.
Setting: a dining table in a home kitchen at night, the stove with a pot softly out of focus on the right, the window on the left dark blue. Lighting: warm pendant light over the table, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

