# ChatGPT 生圖提示詞：一起看電影（promo_06_movie）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_06_movie/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, a dusty-rose knit sweater with a round neckline, blue jeans, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：戲院大廳，她聞著爆米花

- 時段：bright lobby, warm evening｜手上：a popcorn bucket (shots 1-5)｜景別：close
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third holding the popcorn bucket at the bottom of the frame, the glowing popcorn stand on the right. The woman holds a bucket of popcorn at the bottom of the frame and smiles at the viewer, the glowing popcorn stand on the right of the frame in a cinema lobby.
Setting: a warm cinema lobby in the evening with a glowing popcorn stand on the right side of the frame, soft golden lights, a clean uncluttered floor, nobody else in sight. Lighting: warm golden lobby light with the popcorn stand glowing on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred holding the popcorn bucket at the bottom of the frame, the glowing popcorn stand on the right. The woman holds a bucket of popcorn at the bottom of the frame and smiles at the viewer, the glowing popcorn stand on the right of the frame in a cinema lobby.
Setting: a warm cinema lobby in the evening with a glowing popcorn stand on the right side of the frame, soft golden lights, a clean uncluttered floor, nobody else in sight. Lighting: warm golden lobby light with the popcorn stand glowing on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred holding the popcorn bucket at the bottom of the frame, the glowing popcorn stand on the right. The woman holds a bucket of popcorn at the bottom of the frame and smiles at the viewer, the glowing popcorn stand on the right of the frame in a cinema lobby.
Setting: a warm cinema lobby in the evening with a glowing popcorn stand on the right side of the frame, soft golden lights, a clean uncluttered floor, nobody else in sight. Lighting: warm golden lobby light with the popcorn stand glowing on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre holding the popcorn bucket at the bottom of the frame, the glowing popcorn stand on the right. The woman holds a bucket of popcorn at the bottom of the frame and smiles at the viewer, the glowing popcorn stand on the right of the frame in a cinema lobby.
Setting: a warm cinema lobby in the evening with a glowing popcorn stand on the right side of the frame, soft golden lights, a clean uncluttered floor, nobody else in sight. Lighting: warm golden lobby light with the popcorn stand glowing on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：雙人座，她拍拍旁邊

- 時段：house lights on, before the film｜手上：a popcorn bucket (shots 1-5)｜景別：medium
- 接上一鏡：from the lobby to the seat: the same evening, now inside the cinema auditorium with the lights still on

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third seated, patting the seat beside her, the soft screen glow on the right. The woman sits in a red velvet double seat in a cinema auditorium with the house lights still on, patting the seat beside her with one hand, the popcorn bucket in her lap, the soft screen glow on the right of the frame.
Setting: a cinema auditorium with the house lights still on, rows of red velvet seats, the screen out of frame on the right, no other people. Lighting: warm house lights and a soft glow from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred seated, patting the seat beside her, the soft screen glow on the right. The woman sits in a red velvet double seat in a cinema auditorium with the house lights still on, patting the seat beside her with one hand, the popcorn bucket in her lap, the soft screen glow on the right of the frame.
Setting: a cinema auditorium with the house lights still on, rows of red velvet seats, the screen out of frame on the right, no other people. Lighting: warm house lights and a soft glow from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred seated, patting the seat beside her, the soft screen glow on the right. The woman sits in a red velvet double seat in a cinema auditorium with the house lights still on, patting the seat beside her with one hand, the popcorn bucket in her lap, the soft screen glow on the right of the frame.
Setting: a cinema auditorium with the house lights still on, rows of red velvet seats, the screen out of frame on the right, no other people. Lighting: warm house lights and a soft glow from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre seated, patting the seat beside her, the soft screen glow on the right. The woman sits in a red velvet double seat in a cinema auditorium with the house lights still on, patting the seat beside her with one hand, the popcorn bucket in her lap, the soft screen glow on the right of the frame.
Setting: a cinema auditorium with the house lights still on, rows of red velvet seats, the screen out of frame on the right, no other people. Lighting: warm house lights and a soft glow from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：燈暗，她專心看電影

- 時段：dim, screen glow｜手上：a popcorn bucket (shots 1-5)｜景別：close
- 接上一鏡：from the lit hall to the film: the same seats, the lights have dimmed and the screen glow lights her face

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third watching the screen to the right with wide eyes, the soft screen glow on the right. The woman watches the film in the dim cinema auditorium, her eyes fixed on the screen out of frame on the right, the soft blue-white glow lighting her face, the popcorn bucket in her lap at the bottom of the frame.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred watching the screen to the right with wide eyes, the soft screen glow on the right. The woman watches the film in the dim cinema auditorium, her eyes fixed on the screen out of frame on the right, the soft blue-white glow lighting her face, the popcorn bucket in her lap at the bottom of the frame.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred watching the screen to the right with wide eyes, the soft screen glow on the right. The woman watches the film in the dim cinema auditorium, her eyes fixed on the screen out of frame on the right, the soft blue-white glow lighting her face, the popcorn bucket in her lap at the bottom of the frame.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre watching the screen to the right with wide eyes, the soft screen glow on the right. The woman watches the film in the dim cinema auditorium, her eyes fixed on the screen out of frame on the right, the soft blue-white glow lighting her face, the popcorn bucket in her lap at the bottom of the frame.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：劇情緊張，她遞爆米花

- 時段：dim, screen glow｜手上：a popcorn bucket (shots 1-5)｜景別：medium
- 接上一鏡：from the film to the popcorn: the same dim hall and glow, she turns from the screen to the viewer

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third holding the popcorn bucket out toward the viewer, the soft screen glow on the right. The woman gasps softly at the film with the popcorn bucket in her lap, then turns and holds the popcorn out toward the viewer, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred holding the popcorn bucket out toward the viewer, the soft screen glow on the right. The woman gasps softly at the film with the popcorn bucket in her lap, then turns and holds the popcorn out toward the viewer, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred holding the popcorn bucket out toward the viewer, the soft screen glow on the right. The woman gasps softly at the film with the popcorn bucket in her lap, then turns and holds the popcorn out toward the viewer, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre holding the popcorn bucket out toward the viewer, the soft screen glow on the right. The woman gasps softly at the film with the popcorn bucket in her lap, then turns and holds the popcorn out toward the viewer, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：她轉頭，笑出來

- 時段：dim, screen glow｜手上：a popcorn bucket (shots 1-5)｜景別：close
- 接上一鏡：from the popcorn to the laugh: the same dim hall and glow, she has noticed the viewer watching her

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third laughing quietly, a hand on the popcorn bucket, the soft screen glow on the right. The woman turns her head toward the viewer and laughs quietly, her hand resting on the popcorn bucket at the bottom of the frame, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred laughing quietly, a hand on the popcorn bucket, the soft screen glow on the right. The woman turns her head toward the viewer and laughs quietly, her hand resting on the popcorn bucket at the bottom of the frame, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred laughing quietly, a hand on the popcorn bucket, the soft screen glow on the right. The woman turns her head toward the viewer and laughs quietly, her hand resting on the popcorn bucket at the bottom of the frame, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre laughing quietly, a hand on the popcorn bucket, the soft screen glow on the right. The woman turns her head toward the viewer and laughs quietly, her hand resting on the popcorn bucket at the bottom of the frame, the soft blue-white glow from the screen on the right lighting her face.
Setting: a dim cinema auditorium with rows of red velvet seats softly dark behind her, a steady soft blue-white glow from the screen out of frame on the right, no other people. Lighting: a steady soft blue-white glow from the screen out of frame on the right lighting her face, the seats behind softly dark, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：散場燈亮，她伸懶腰

- 時段：lights up, after the film｜手上：none｜景別：medium
- 接上一鏡：from the dark to the lights: the same hall, the house lights are up and her hands are empty

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third stretching with a content smile, the soft glow on the right. The woman stretches her arms up in the lit cinema auditorium after the film and smiles at the viewer, the seats behind her softly lit, a soft glow on the right of the frame.
Setting: a cinema auditorium after the film with the house lights up, rows of red velvet seats, a soft glow from the right out of frame, no other people. Lighting: warm house lights, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level, medium shot from the waist up, the woman centred stretching with a content smile, the soft glow on the right. The woman stretches her arms up in the lit cinema auditorium after the film and smiles at the viewer, the seats behind her softly lit, a soft glow on the right of the frame.
Setting: a cinema auditorium after the film with the house lights up, rows of red velvet seats, a soft glow from the right out of frame, no other people. Lighting: warm house lights, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred stretching with a content smile, the soft glow on the right. The woman stretches her arms up in the lit cinema auditorium after the film and smiles at the viewer, the seats behind her softly lit, a soft glow on the right of the frame.
Setting: a cinema auditorium after the film with the house lights up, rows of red velvet seats, a soft glow from the right out of frame, no other people. Lighting: warm house lights, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a dusty-rose knit sweater with a round neckline that covers the collarbone, and blue jeans.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre stretching with a content smile, the soft glow on the right. The woman stretches her arms up in the lit cinema auditorium after the film and smiles at the viewer, the seats behind her softly lit, a soft glow on the right of the frame.
Setting: a cinema auditorium after the film with the house lights up, rows of red velvet seats, a soft glow from the right out of frame, no other people. Lighting: warm house lights, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

