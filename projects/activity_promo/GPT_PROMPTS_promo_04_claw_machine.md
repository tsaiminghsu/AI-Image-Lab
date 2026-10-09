# ChatGPT 生圖提示詞：一起玩娃娃機（promo_04_claw_machine）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_04_claw_machine/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, a lavender crew-neck sweatshirt, blue jeans, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：機台前，她看著裡面

- 時段：evening, neon arcade glow｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third facing the viewer with one hand on the glass, the claw machine on the right. The woman stands in front of the claw machine on the right of the frame, one hand resting on its glass, looking at the plush toys inside and then at the viewer with wide, delighted eyes.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred facing the viewer with one hand on the glass, the claw machine on the right. The woman stands in front of the claw machine on the right of the frame, one hand resting on its glass, looking at the plush toys inside and then at the viewer with wide, delighted eyes.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred facing the viewer with one hand on the glass, the claw machine on the right. The woman stands in front of the claw machine on the right of the frame, one hand resting on its glass, looking at the plush toys inside and then at the viewer with wide, delighted eyes.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre facing the viewer with one hand on the glass, the claw machine on the right. The woman stands in front of the claw machine on the right of the frame, one hand resting on its glass, looking at the plush toys inside and then at the viewer with wide, delighted eyes.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：她握住搖桿，瞄準那隻

- 時段：evening, neon arcade glow｜手上：none｜景別：close
- 接上一鏡：from the machine front to the controls: the same arcade and light, her hand now on the joystick and the claw moving

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third with one hand on the joystick at the lower right of the frame, the claw machine on the right. The woman holds the joystick of the claw machine with one hand at the lower right of the frame, steering it while she looks at the white-and-lilac plush on the top shelf inside the machine with a playful grin.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred with one hand on the joystick at the lower right of the frame, the claw machine on the right. The woman holds the joystick of the claw machine with one hand at the lower right of the frame, steering it while she looks at the white-and-lilac plush on the top shelf inside the machine with a playful grin.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred with one hand on the joystick at the lower right of the frame, the claw machine on the right. The woman holds the joystick of the claw machine with one hand at the lower right of the frame, steering it while she looks at the white-and-lilac plush on the top shelf inside the machine with a playful grin.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre with one hand on the joystick at the lower right of the frame, the claw machine on the right. The woman holds the joystick of the claw machine with one hand at the lower right of the frame, steering it while she looks at the white-and-lilac plush on the top shelf inside the machine with a playful grin.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：她按下按鈕，爪子下降

- 時段：evening, neon arcade glow｜手上：none｜景別：medium
- 接上一鏡：from steering to pressing: the same arcade and light, the claw now stopped above the plush and her finger on the button

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third with one hand on the joystick and the other on the big round button, the claw machine on the right. The woman presses the big round button on the control panel at the lower right of the frame with one hand, her other hand on the joystick, her eyes on the claw hanging above the white-and-lilac plush inside the machine.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred with one hand on the joystick and the other on the big round button, the claw machine on the right. The woman presses the big round button on the control panel at the lower right of the frame with one hand, her other hand on the joystick, her eyes on the claw hanging above the white-and-lilac plush inside the machine.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred with one hand on the joystick and the other on the big round button, the claw machine on the right. The woman presses the big round button on the control panel at the lower right of the frame with one hand, her other hand on the joystick, her eyes on the claw hanging above the white-and-lilac plush inside the machine.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre with one hand on the joystick and the other on the big round button, the claw machine on the right. The woman presses the big round button on the control panel at the lower right of the frame with one hand, her other hand on the joystick, her eyes on the claw hanging above the white-and-lilac plush inside the machine.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：夾起來，又滑掉了

- 時段：evening, neon arcade glow｜手上：none｜景別：close
- 接上一鏡：from the descent to the grab: the same arcade and light, the claw now holds the plush and loses it

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third with both hands at her cheeks, the claw machine on the right. The woman has her hands at her cheeks in a half-laugh, half-groan, her face turned toward the viewer, while inside the claw machine on the right the claw lifts the white-and-lilac plush a little and the plush slips back down onto the pile.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred with both hands at her cheeks, the claw machine on the right. The woman has her hands at her cheeks in a half-laugh, half-groan, her face turned toward the viewer, while inside the claw machine on the right the claw lifts the white-and-lilac plush a little and the plush slips back down onto the pile.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred with both hands at her cheeks, the claw machine on the right. The woman has her hands at her cheeks in a half-laugh, half-groan, her face turned toward the viewer, while inside the claw machine on the right the claw lifts the white-and-lilac plush a little and the plush slips back down onto the pile.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre with both hands at her cheeks, the claw machine on the right. The woman has her hands at her cheeks in a half-laugh, half-groan, her face turned toward the viewer, while inside the claw machine on the right the claw lifts the white-and-lilac plush a little and the plush slips back down onto the pile.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：再夾一次，掉進出口

- 時段：evening, neon arcade glow｜手上：none｜景別：medium
- 接上一鏡：from the miss to the second try: the same arcade and light, the claw now carries the plush to the chute

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third with both hands raised, the claw machine on the right. The woman has both hands raised in hope and then in cheering, her face turned toward the viewer, while inside the claw machine on the right the claw carries the white-and-lilac plush over the prize chute and lets it drop.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred with both hands raised, the claw machine on the right. The woman has both hands raised in hope and then in cheering, her face turned toward the viewer, while inside the claw machine on the right the claw carries the white-and-lilac plush over the prize chute and lets it drop.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred with both hands raised, the claw machine on the right. The woman has both hands raised in hope and then in cheering, her face turned toward the viewer, while inside the claw machine on the right the claw carries the white-and-lilac plush over the prize chute and lets it drop.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre with both hands raised, the claw machine on the right. The woman has both hands raised in hope and then in cheering, her face turned toward the viewer, while inside the claw machine on the right the claw carries the white-and-lilac plush over the prize chute and lets it drop.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：她抱著娃娃對你笑

- 時段：evening, neon arcade glow｜手上：a white unicorn plush (shot 6)｜景別：close
- 接上一鏡：from the cheer to the prize: the same arcade and light, the plush from the chute now in her arms

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third hugging the plush against her chest, the claw machine on the right. The woman hugs the white unicorn plush against her chest and smiles warmly at the viewer, the claw machine softly out of focus on the right of the frame.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred hugging the plush against her chest, the claw machine on the right. The woman hugs the white unicorn plush against her chest and smiles warmly at the viewer, the claw machine softly out of focus on the right of the frame.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred hugging the plush against her chest, the claw machine on the right. The woman hugs the white unicorn plush against her chest and smiles warmly at the viewer, the claw machine softly out of focus on the right of the frame.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a lavender crew-neck sweatshirt with a round neckline and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre hugging the plush against her chest, the claw machine on the right. The woman hugs the white unicorn plush against her chest and smiles warmly at the viewer, the claw machine softly out of focus on the right of the frame.
Setting: a bright arcade in the evening with a glass claw machine full of pastel plush toys on the right side of the frame, the claw hanging inside the cabinet above the plush toys, a control panel with a joystick and a big round button at the lower right of the frame, a soft pink-and-blue neon glow, a clean uncluttered floor, nobody else in sight. Lighting: soft pink and blue neon glow from the machine on the right and warm overhead light, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

