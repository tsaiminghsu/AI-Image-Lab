# ChatGPT 生圖提示詞：一起去旅行（promo_08_travel）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_08_travel/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, a white linen blouse with short sleeves and a round neckline, a light-blue midi skirt, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：早晨的和室，地圖攤開

- 時段：morning, soft window light｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third kneeling at the low table with a map, the bright window on the right. The woman kneels at a low wooden table in a Japanese-style inn room with a paper map spread on it, looking up at the viewer with an excited smile, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium shot from the waist up, the woman centred kneeling at the low table with a map, the bright window on the right. The woman kneels at a low wooden table in a Japanese-style inn room with a paper map spread on it, looking up at the viewer with an excited smile, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred kneeling at the low table with a map, the bright window on the right. The woman kneels at a low wooden table in a Japanese-style inn room with a paper map spread on it, looking up at the viewer with an excited smile, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre kneeling at the low table with a map, the bright window on the right. The woman kneels at a low wooden table in a Japanese-style inn room with a paper map spread on it, looking up at the viewer with an excited smile, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：她指著地圖上的海岸

- 時段：morning, soft window light｜手上：straw hat｜景別：close
- 接上一鏡：from the map to the choice: the same Japanese-style room and morning light, the straw hat now on her head

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third wearing a straw hat and grinning, the bright window on the right. The woman wears a straw hat and taps the coastline on the paper map with one finger in the Japanese-style inn room, grinning at the viewer, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred wearing a straw hat and grinning, the bright window on the right. The woman wears a straw hat and taps the coastline on the paper map with one finger in the Japanese-style inn room, grinning at the viewer, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred wearing a straw hat and grinning, the bright window on the right. The woman wears a straw hat and taps the coastline on the paper map with one finger in the Japanese-style inn room, grinning at the viewer, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre wearing a straw hat and grinning, the bright window on the right. The woman wears a straw hat and taps the coastline on the paper map with one finger in the Japanese-style inn room, grinning at the viewer, the bright window on the right of the frame.
Setting: a Japanese-style inn room in the morning with tatami mats, paper shoji screens and a low wooden table, a window with soft morning light on the right side of the frame, nobody else in sight, no readable signs. Lighting: soft morning light from the window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：火車窗邊，海出現了

- 時段：afternoon, bright sea light｜手上：straw hat｜景別：medium
- 接上一鏡：from the room to the train: the Japanese coast in the afternoon, the sea now outside the window

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third seated by the train window with the straw hat on her lap, the sea outside the train window on the right. The woman sits by the window of a small Japanese local train with the straw hat on her lap, turning from the sea outside the window on the right of the frame to smile at the viewer.
Setting: a window seat on a small Japanese local train in the afternoon, the sea bright outside the window on the right side of the frame, empty seats, nobody else in sight, no readable signs. Lighting: bright afternoon light from the train window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium shot from the waist up, the woman centred seated by the train window with the straw hat on her lap, the sea outside the train window on the right. The woman sits by the window of a small Japanese local train with the straw hat on her lap, turning from the sea outside the window on the right of the frame to smile at the viewer.
Setting: a window seat on a small Japanese local train in the afternoon, the sea bright outside the window on the right side of the frame, empty seats, nobody else in sight, no readable signs. Lighting: bright afternoon light from the train window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred seated by the train window with the straw hat on her lap, the sea outside the train window on the right. The woman sits by the window of a small Japanese local train with the straw hat on her lap, turning from the sea outside the window on the right of the frame to smile at the viewer.
Setting: a window seat on a small Japanese local train in the afternoon, the sea bright outside the window on the right side of the frame, empty seats, nobody else in sight, no readable signs. Lighting: bright afternoon light from the train window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre seated by the train window with the straw hat on her lap, the sea outside the train window on the right. The woman sits by the window of a small Japanese local train with the straw hat on her lap, turning from the sea outside the window on the right of the frame to smile at the viewer.
Setting: a window seat on a small Japanese local train in the afternoon, the sea bright outside the window on the right side of the frame, empty seats, nobody else in sight, no readable signs. Lighting: bright afternoon light from the train window on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：海邊步道，她拿著糰子

- 時段：afternoon, bright sea light｜手上：straw hat, dango｜景別：close
- 接上一鏡：from the train to the shore: the same afternoon on the Japanese coast, the sea now right beside her

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third holding her straw hat with one hand and a skewer of dango in the other, the blue sea on the right. The woman holds her straw hat on her head with one hand against the sea breeze and a skewer of dango in the other on a Japanese seaside promenade, smiling at the viewer, the blue sea on the right of the frame.
Setting: a Japanese seaside promenade in the afternoon with a low sea wall, the blue sea on the right side of the frame, a clean uncluttered path, nobody else in sight, no readable signs. Lighting: bright afternoon sunlight, the sea sparkling on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred holding her straw hat with one hand and a skewer of dango in the other, the blue sea on the right. The woman holds her straw hat on her head with one hand against the sea breeze and a skewer of dango in the other on a Japanese seaside promenade, smiling at the viewer, the blue sea on the right of the frame.
Setting: a Japanese seaside promenade in the afternoon with a low sea wall, the blue sea on the right side of the frame, a clean uncluttered path, nobody else in sight, no readable signs. Lighting: bright afternoon sunlight, the sea sparkling on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred holding her straw hat with one hand and a skewer of dango in the other, the blue sea on the right. The woman holds her straw hat on her head with one hand against the sea breeze and a skewer of dango in the other on a Japanese seaside promenade, smiling at the viewer, the blue sea on the right of the frame.
Setting: a Japanese seaside promenade in the afternoon with a low sea wall, the blue sea on the right side of the frame, a clean uncluttered path, nobody else in sight, no readable signs. Lighting: bright afternoon sunlight, the sea sparkling on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre holding her straw hat with one hand and a skewer of dango in the other, the blue sea on the right. The woman holds her straw hat on her head with one hand against the sea breeze and a skewer of dango in the other on a Japanese seaside promenade, smiling at the viewer, the blue sea on the right of the frame.
Setting: a Japanese seaside promenade in the afternoon with a low sea wall, the blue sea on the right side of the frame, a clean uncluttered path, nobody else in sight, no readable signs. Lighting: bright afternoon sunlight, the sea sparkling on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：夕陽的堤防，她回頭

- 時段：sunset, golden｜手上：straw hat｜景別：medium
- 接上一鏡：from the afternoon to the sunset: the same Japanese coast, the sun now low over the sea

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third sitting on the sea wall, looking back at the viewer, the sunset over the sea on the right. The woman sits on a low sea wall on a Japanese coast at sunset with the straw hat on her head, looking back at the viewer with a calm smile, the sun low over the sea on the right of the frame.
Setting: a low sea wall on a Japanese coast at sunset, the sun low over the sea on the right side of the frame, calm water, nobody else in sight, no readable signs. Lighting: warm golden sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium shot from the waist up, the woman centred sitting on the sea wall, looking back at the viewer, the sunset over the sea on the right. The woman sits on a low sea wall on a Japanese coast at sunset with the straw hat on her head, looking back at the viewer with a calm smile, the sun low over the sea on the right of the frame.
Setting: a low sea wall on a Japanese coast at sunset, the sun low over the sea on the right side of the frame, calm water, nobody else in sight, no readable signs. Lighting: warm golden sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred sitting on the sea wall, looking back at the viewer, the sunset over the sea on the right. The woman sits on a low sea wall on a Japanese coast at sunset with the straw hat on her head, looking back at the viewer with a calm smile, the sun low over the sea on the right of the frame.
Setting: a low sea wall on a Japanese coast at sunset, the sun low over the sea on the right side of the frame, calm water, nobody else in sight, no readable signs. Lighting: warm golden sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre sitting on the sea wall, looking back at the viewer, the sunset over the sea on the right. The woman sits on a low sea wall on a Japanese coast at sunset with the straw hat on her head, looking back at the viewer with a calm smile, the sun low over the sea on the right of the frame.
Setting: a low sea wall on a Japanese coast at sunset, the sun low over the sea on the right side of the frame, calm water, nobody else in sight, no readable signs. Lighting: warm golden sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：旅館陽台，滿天星星

- 時段：night, starry｜手上：straw hat｜景別：close
- 接上一鏡：from the sunset to the stars: the Japanese coast at night, a lantern on her face

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third holding the straw hat against her chest, the starry sky over the sea on the right. The woman stands on the wooden balcony of a Japanese seaside inn at night with the straw hat held against her chest, looking at the viewer with a soft smile, the starry sky over the sea on the right of the frame.
Setting: a wooden balcony of a Japanese seaside inn at night, a paper lantern glowing warmly, a sky full of stars over the dark sea on the right side of the frame, nobody else in sight, no readable signs. Lighting: a warm paper-lantern glow on her face, a starry sky behind her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred holding the straw hat against her chest, the starry sky over the sea on the right. The woman stands on the wooden balcony of a Japanese seaside inn at night with the straw hat held against her chest, looking at the viewer with a soft smile, the starry sky over the sea on the right of the frame.
Setting: a wooden balcony of a Japanese seaside inn at night, a paper lantern glowing warmly, a sky full of stars over the dark sea on the right side of the frame, nobody else in sight, no readable signs. Lighting: a warm paper-lantern glow on her face, a starry sky behind her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred holding the straw hat against her chest, the starry sky over the sea on the right. The woman stands on the wooden balcony of a Japanese seaside inn at night with the straw hat held against her chest, looking at the viewer with a soft smile, the starry sky over the sea on the right of the frame.
Setting: a wooden balcony of a Japanese seaside inn at night, a paper lantern glowing warmly, a sky full of stars over the dark sea on the right side of the frame, nobody else in sight, no readable signs. Lighting: a warm paper-lantern glow on her face, a starry sky behind her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a white linen blouse with short sleeves and a round neckline, and a light-blue midi skirt.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre holding the straw hat against her chest, the starry sky over the sea on the right. The woman stands on the wooden balcony of a Japanese seaside inn at night with the straw hat held against her chest, looking at the viewer with a soft smile, the starry sky over the sea on the right of the frame.
Setting: a wooden balcony of a Japanese seaside inn at night, a paper lantern glowing warmly, a sky full of stars over the dark sea on the right side of the frame, nobody else in sight, no readable signs. Lighting: a warm paper-lantern glow on her face, a starry sky behind her, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

