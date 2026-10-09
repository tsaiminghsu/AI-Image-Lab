# ChatGPT 生圖提示詞：一起去看夜景（promo_07_night_view）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_07_night_view/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, an ivory turtleneck knit, a camel wool skirt, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：黃昏的坡道，她回頭

- 時段：dusk, first lights｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third facing the viewer with a soft smile, the hill with its tower on the right. The woman stands on a quiet hillside street in Seoul at dusk and looks at the viewer with a soft smile, the hill with its observation tower on the right of the frame.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium shot from the waist up, the woman centred facing the viewer with a soft smile, the hill with its tower on the right. The woman stands on a quiet hillside street in Seoul at dusk and looks at the viewer with a soft smile, the hill with its observation tower on the right of the frame.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred facing the viewer with a soft smile, the hill with its tower on the right. The woman stands on a quiet hillside street in Seoul at dusk and looks at the viewer with a soft smile, the hill with its observation tower on the right of the frame.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre facing the viewer with a soft smile, the hill with its tower on the right. The woman stands on a quiet hillside street in Seoul at dusk and looks at the viewer with a soft smile, the hill with its observation tower on the right of the frame.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：她指向山上的觀景台

- 時段：dusk, first lights｜手上：none｜景別：close
- 接上一鏡：from the street to the choice: the same Seoul street and dusk light, she points at the tower

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third pointing toward the hill with bright eyes, the hill with its tower on the right. The woman points toward the observation tower on the hill on the right of the frame in Seoul at dusk, her eyes bright, looking at the viewer.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred pointing toward the hill with bright eyes, the hill with its tower on the right. The woman points toward the observation tower on the hill on the right of the frame in Seoul at dusk, her eyes bright, looking at the viewer.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred pointing toward the hill with bright eyes, the hill with its tower on the right. The woman points toward the observation tower on the hill on the right of the frame in Seoul at dusk, her eyes bright, looking at the viewer.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre pointing toward the hill with bright eyes, the hill with its tower on the right. The woman points toward the observation tower on the hill on the right of the frame in Seoul at dusk, her eyes bright, looking at the viewer.
Setting: a quiet hillside street in a Seoul neighbourhood at dusk, low tiled-roof houses, a hill with an observation tower on the right side of the frame, the first city lights coming on, nobody else in sight, no readable signs. Lighting: soft dusk light and a warm street lamp from the left, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：城牆邊的步道，她停下來

- 時段：blue hour, lamps on｜手上：none｜景別：medium
- 接上一鏡：from the street to the path: Seoul one step later, the sky now deep blue and the path lamps on

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third standing by the fortress wall with one hand on it, the city lights between the pines on the right. The woman stands on a stone path beside an old Seoul fortress wall in the blue hour, one hand resting on the wall, smiling at the viewer, the city lights showing between the pines on the right of the frame.
Setting: a stone path beside an old Seoul fortress wall in the blue hour, small warm path lamps, the city lights showing between pine trees on the right side of the frame, nobody else in sight, no readable signs. Lighting: blue-hour sky and a warm path lamp beside her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium shot from the waist up, the woman centred standing by the fortress wall with one hand on it, the city lights between the pines on the right. The woman stands on a stone path beside an old Seoul fortress wall in the blue hour, one hand resting on the wall, smiling at the viewer, the city lights showing between the pines on the right of the frame.
Setting: a stone path beside an old Seoul fortress wall in the blue hour, small warm path lamps, the city lights showing between pine trees on the right side of the frame, nobody else in sight, no readable signs. Lighting: blue-hour sky and a warm path lamp beside her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred standing by the fortress wall with one hand on it, the city lights between the pines on the right. The woman stands on a stone path beside an old Seoul fortress wall in the blue hour, one hand resting on the wall, smiling at the viewer, the city lights showing between the pines on the right of the frame.
Setting: a stone path beside an old Seoul fortress wall in the blue hour, small warm path lamps, the city lights showing between pine trees on the right side of the frame, nobody else in sight, no readable signs. Lighting: blue-hour sky and a warm path lamp beside her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre standing by the fortress wall with one hand on it, the city lights between the pines on the right. The woman stands on a stone path beside an old Seoul fortress wall in the blue hour, one hand resting on the wall, smiling at the viewer, the city lights showing between the pines on the right of the frame.
Setting: a stone path beside an old Seoul fortress wall in the blue hour, small warm path lamps, the city lights showing between pine trees on the right side of the frame, nobody else in sight, no readable signs. Lighting: blue-hour sky and a warm path lamp beside her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：欄杆邊，整片燈海

- 時段：night, city lights｜手上：none｜景別：close
- 接上一鏡：from the path to the deck: Seoul at night now, the whole city lit behind her

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third counting on her fingers with a playful look, the sea of city lights on the right. The woman stands at the railing of an observation deck above Seoul at night, counting on her fingers with a playful look at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred counting on her fingers with a playful look, the sea of city lights on the right. The woman stands at the railing of an observation deck above Seoul at night, counting on her fingers with a playful look at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred counting on her fingers with a playful look, the sea of city lights on the right. The woman stands at the railing of an observation deck above Seoul at night, counting on her fingers with a playful look at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre counting on her fingers with a playful look, the sea of city lights on the right. The woman stands at the railing of an observation deck above Seoul at night, counting on her fingers with a playful look at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：靠著欄杆，她說心裡話

- 時段：night, city lights｜手上：none｜景別：medium
- 接上一鏡：from counting to talking: the same deck and lights, she has turned quiet

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third leaning on the railing with a gentle smile, the sea of city lights on the right. The woman leans on the railing of the observation deck above Seoul at night, her hair moving in the breeze, looking at the viewer with a gentle, honest smile, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium shot from the waist up, the woman centred leaning on the railing with a gentle smile, the sea of city lights on the right. The woman leans on the railing of the observation deck above Seoul at night, her hair moving in the breeze, looking at the viewer with a gentle, honest smile, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred leaning on the railing with a gentle smile, the sea of city lights on the right. The woman leans on the railing of the observation deck above Seoul at night, her hair moving in the breeze, looking at the viewer with a gentle, honest smile, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre leaning on the railing with a gentle smile, the sea of city lights on the right. The woman leans on the railing of the observation deck above Seoul at night, her hair moving in the breeze, looking at the viewer with a gentle, honest smile, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：她披著你的外套

- 時段：night, city lights｜手上：jacket｜景別：close
- 接上一鏡：from the talk to the jacket: the same deck and lights, the viewer's jacket now on her shoulders

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third with a dark jacket draped over her shoulders, the sea of city lights on the right. The woman has the viewer's dark jacket draped over her shoulders on the observation deck above Seoul at night, holding its collar closed with both hands and smiling at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred with a dark jacket draped over her shoulders, the sea of city lights on the right. The woman has the viewer's dark jacket draped over her shoulders on the observation deck above Seoul at night, holding its collar closed with both hands and smiling at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred with a dark jacket draped over her shoulders, the sea of city lights on the right. The woman has the viewer's dark jacket draped over her shoulders on the observation deck above Seoul at night, holding its collar closed with both hands and smiling at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears an ivory turtleneck knit and a camel wool skirt.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre with a dark jacket draped over her shoulders, the sea of city lights on the right. The woman has the viewer's dark jacket draped over her shoulders on the observation deck above Seoul at night, holding its collar closed with both hands and smiling at the viewer, the sea of city lights on the right of the frame.
Setting: a hilltop observation deck above Seoul at night with a wooden railing, a wide sea of city lights and a river with lit bridges on the right side of the frame, a warm deck lamp nearby, nobody else in sight, no readable signs. Lighting: a warm deck lamp on her face, the city lights glowing behind her on the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

