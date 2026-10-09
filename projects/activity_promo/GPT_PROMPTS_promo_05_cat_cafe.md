# ChatGPT 生圖提示詞：一起去貓咪咖啡廳（promo_05_cat_cafe）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_05_cat_cafe/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, a sage-green knit sweater with a round neckline, blue jeans, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：門口，牠們都在看

- 時段：afternoon, golden｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third facing the viewer with wide amused eyes, the sunlit window on the right. The woman stands just inside the door of a cat café with the sunlit window and cushions on the right of the frame, looking at the viewer with wide, amused eyes.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred facing the viewer with wide amused eyes, the sunlit window on the right. The woman stands just inside the door of a cat café with the sunlit window and cushions on the right of the frame, looking at the viewer with wide, amused eyes.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred facing the viewer with wide amused eyes, the sunlit window on the right. The woman stands just inside the door of a cat café with the sunlit window and cushions on the right of the frame, looking at the viewer with wide, amused eyes.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre facing the viewer with wide amused eyes, the sunlit window on the right. The woman stands just inside the door of a cat café with the sunlit window and cushions on the right of the frame, looking at the viewer with wide, amused eyes.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：窗邊軟墊，陽光照進來

- 時段：afternoon, golden｜手上：none｜景別：close
- 接上一鏡：from the door to the window seat: the same cat café and light, she is now seated on the cushion

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third seated on the cushion by the window, the sunlit window on the right. The woman sits on the soft cushion by the sunlit window on the right of the frame, sunlight on her face, smiling at the viewer.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred seated on the cushion by the window, the sunlit window on the right. The woman sits on the soft cushion by the sunlit window on the right of the frame, sunlight on her face, smiling at the viewer.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred seated on the cushion by the window, the sunlit window on the right. The woman sits on the soft cushion by the sunlit window on the right of the frame, sunlight on her face, smiling at the viewer.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre seated on the cushion by the window, the sunlit window on the right. The woman sits on the soft cushion by the sunlit window on the right of the frame, sunlight on her face, smiling at the viewer.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：一人一半，她遞零食

- 時段：afternoon, golden｜手上：treats, an orange cat (shots 3-6)｜景別：medium
- 接上一鏡：from sitting to the treat: the same cat café and light, she now holds a small cup of treats

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third holding out half of the treats toward the viewer, the sunlit window on the right. The woman sits on the cushion by the sunlit window on the right of the frame, holding out half of a small cup of treats toward the viewer with a bright smile, a fluffy orange cat curled up asleep on the table in the lower right of the frame.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred holding out half of the treats toward the viewer, the sunlit window on the right. The woman sits on the cushion by the sunlit window on the right of the frame, holding out half of a small cup of treats toward the viewer with a bright smile, a fluffy orange cat curled up asleep on the table in the lower right of the frame.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred holding out half of the treats toward the viewer, the sunlit window on the right. The woman sits on the cushion by the sunlit window on the right of the frame, holding out half of a small cup of treats toward the viewer with a bright smile, a fluffy orange cat curled up asleep on the table in the lower right of the frame.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre holding out half of the treats toward the viewer, the sunlit window on the right. The woman sits on the cushion by the sunlit window on the right of the frame, holding out half of a small cup of treats toward the viewer with a bright smile, a fluffy orange cat curled up asleep on the table in the lower right of the frame.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a few tabby and white cats resting far behind her on the cushions, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: warm golden afternoon sunlight from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：橘貓趴上桌，她輕輕摸

- 時段：late afternoon, deeper gold｜手上：an orange cat (shots 4-6)｜景別：close
- 接上一鏡：from the treat to the cat: the same cat café, the light a step later, she turns to the orange cat that was already asleep on the table

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third gently stroking the cat on the table, the sunlit window on the right. The woman gently strokes a fluffy orange cat curled up on the table in the lower right of the frame, looking softly at the viewer, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred gently stroking the cat on the table, the sunlit window on the right. The woman gently strokes a fluffy orange cat curled up on the table in the lower right of the frame, looking softly at the viewer, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred gently stroking the cat on the table, the sunlit window on the right. The woman gently strokes a fluffy orange cat curled up on the table in the lower right of the frame, looking softly at the viewer, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre gently stroking the cat on the table, the sunlit window on the right. The woman gently strokes a fluffy orange cat curled up on the table in the lower right of the frame, looking softly at the viewer, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：貓睡著，她食指放唇邊

- 時段：late afternoon, deeper gold｜手上：an orange cat (shots 4-6)｜景別：medium
- 接上一鏡：from stroking to sleeping: the same table and light, the orange cat now asleep

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third with one finger to her lips, the sunlit window on the right. The woman holds one finger to her lips and smiles at the viewer, a fluffy orange cat fast asleep on the table in the lower right of the frame, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred with one finger to her lips, the sunlit window on the right. The woman holds one finger to her lips and smiles at the viewer, a fluffy orange cat fast asleep on the table in the lower right of the frame, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred with one finger to her lips, the sunlit window on the right. The woman holds one finger to her lips and smiles at the viewer, a fluffy orange cat fast asleep on the table in the lower right of the frame, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre with one finger to her lips, the sunlit window on the right. The woman holds one finger to her lips and smiles at the viewer, a fluffy orange cat fast asleep on the table in the lower right of the frame, the sunlit window on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat curled up asleep on the wooden table in the lower right. Lighting: deeper golden light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：夕陽裡，她想幫牠取名

- 時段：sunset, warm orange-pink｜手上：an orange cat (shots 4-6)｜景別：close
- 接上一鏡：from the whisper to the name: the same table, the light now a sunset glow, the cat still asleep

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third resting her chin on her folded arms, the sunlit window on the right. The woman rests her chin on her folded arms on the table beside the sleeping orange cat in the lower right of the frame and smiles warmly at the viewer, the sunset window glowing on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat asleep on the wooden table in the lower right, the window glowing orange-pink. Lighting: warm orange-pink sunset light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred resting her chin on her folded arms, the sunlit window on the right. The woman rests her chin on her folded arms on the table beside the sleeping orange cat in the lower right of the frame and smiles warmly at the viewer, the sunset window glowing on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat asleep on the wooden table in the lower right, the window glowing orange-pink. Lighting: warm orange-pink sunset light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred resting her chin on her folded arms, the sunlit window on the right. The woman rests her chin on her folded arms on the table beside the sleeping orange cat in the lower right of the frame and smiles warmly at the viewer, the sunset window glowing on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat asleep on the wooden table in the lower right, the window glowing orange-pink. Lighting: warm orange-pink sunset light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a sage-green knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre resting her chin on her folded arms, the sunlit window on the right. The woman rests her chin on her folded arms on the table beside the sleeping orange cat in the lower right of the frame and smiles warmly at the viewer, the sunset window glowing on the right.
Setting: a cozy cat café with a big sunlit window and soft floor cushions on the right side of the frame, a wooden table, clean uncluttered surfaces, no other people, a fluffy orange cat asleep on the wooden table in the lower right, the window glowing orange-pink. Lighting: warm orange-pink sunset light from the window on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

