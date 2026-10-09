# ChatGPT 生圖提示詞：一起去遊樂園（promo_01_amusement_park）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_01_amusement_park/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, dark brown leather jacket with a cream shearling collar and cuffs over a cream knit top, blue jeans, pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：園區入口，她回頭招手

- 時段：late afternoon, golden｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium shot framed from the chest up with nothing below her waist in frame, the woman on the left third facing right and seen from the front, the arch of lights framing her, the Ferris wheel far away on the right. The woman stands just inside the amusement park entrance with her body turned toward the right of the frame, where the park lies, looking back over her shoulder at the viewer with a bright open smile, one hand raised in a small wave.
Setting: amusement park entrance plaza in the late afternoon, an arch of warm light bulbs overhead, a carousel and a small distant Ferris wheel on the right horizon, clean open pavement. Lighting: golden late-afternoon sunlight from the side, the warm bulbs glowing, bright and clear, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium shot framed from the chest up with nothing below her waist in frame, the woman centred and seen from the front with her body angled toward the right, the arch of lights overhead, the Ferris wheel far away on the right. The woman stands just inside the amusement park entrance with her body turned toward the right of the frame, where the park lies, looking back over her shoulder at the viewer with a bright open smile, one hand raised in a small wave.
Setting: amusement park entrance plaza in the late afternoon, an arch of warm light bulbs overhead, a carousel and a small distant Ferris wheel on the right horizon, clean open pavement. Lighting: golden late-afternoon sunlight from the side, the warm bulbs glowing, bright and clear, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：她指著設施，等你選

- 時段：late afternoon, golden｜手上：none｜景別：close
- 接上一鏡：from the entrance to the midway: the same afternoon light, she has walked in and turned around to face the viewer

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, her pointing arm leading to the right, the Ferris wheel above the rides behind it. The woman has just walked in toward the right and is turning back to the viewer in mid-step, pointing with one outstretched arm toward the rides on the right of the frame, eyebrows raised in a playful questioning smile.
Setting: amusement park midway in the late afternoon, a roller coaster track and a carousel on the right, the Ferris wheel clearly visible above them, clean open pavement. Lighting: golden late-afternoon sunlight, soft shadows, bright and clear, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, her pointing arm leading to the right, the roller coaster track soft behind her. The woman has just walked in toward the right and is turning back to the viewer in mid-step, pointing with one outstretched arm toward the rides on the right of the frame, eyebrows raised in a playful questioning smile.
Setting: amusement park midway in the late afternoon, a roller coaster track and a carousel on the right, the Ferris wheel clearly visible above them, clean open pavement. Lighting: golden late-afternoon sunlight, soft shadows, bright and clear, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：點心攤，她把棉花糖遞給你

- 時段：sunset, low orange sun｜手上：pink cotton candy on a stick (shots 3-5)｜景別：medium
- 接上一鏡：from the midway to the stalls: still afternoon, the light turns orange

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third walking right, the cotton candy held out toward the viewer, the stalls and the Ferris wheel behind her on the right. The woman walks toward the right beside the viewer along a quiet lane of food stalls, half turned to the viewer, holding a pink cotton candy on a stick out toward the viewer as if offering a bite, laughing.
Setting: amusement park food lane at sunset, a small cotton candy cart with no readable text, strings of lights just switching on, the Ferris wheel rising behind the stalls on the right, clean open pavement with nobody else on it. Lighting: low orange sunset sun from the side, warm string lights starting to glow, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium shot from the waist up, the woman centred and half turned to the viewer, the cotton candy held out in the foreground, the Ferris wheel rising on the right. The woman walks toward the right beside the viewer along a quiet lane of food stalls, half turned to the viewer, holding a pink cotton candy on a stick out toward the viewer as if offering a bite, laughing.
Setting: amusement park food lane at sunset, a small cotton candy cart with no readable text, strings of lights just switching on, the Ferris wheel rising behind the stalls on the right, clean open pavement with nobody else on it. Lighting: low orange sunset sun from the side, warm string lights starting to glow, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：天快黑了，她拉你去摩天輪

- 時段：dusk, sky turning blue and pink｜手上：pink cotton candy on a stick (shots 3-5)｜景別：close
- 接上一鏡：from the stalls to the wheel: she is still carrying the cotton candy, the sky goes from orange to blue

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium close-up of her face turned back over her shoulder on the left third, her free hand reaching toward the viewer at the lower edge of the frame, small and in sharp focus, the glowing Ferris wheel filling the right side. The woman walks toward the right, toward a large Ferris wheel whose lights are switching on ring by ring, a pink cotton candy on a stick in one hand, looking back over her shoulder at the viewer with a smile and reaching her free hand toward the viewer, the hand held at a natural arm's length at the lower edge of the frame, small in the picture and in sharp focus with five natural fingers.
Setting: open walkway at dusk at the foot of a large Ferris wheel that fills the right side of the frame, its lights switching on, a few lit lamp posts, clean open pavement with nobody else on it. Lighting: dusk sky fading from pink to blue, the Ferris wheel lights glowing warm, soft light on her face from the front.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium close-up of her face turned back over her shoulder in the centre, her free hand reaching toward the viewer at the lower edge of the frame, small and in sharp focus, the glowing Ferris wheel soft behind her on the right. The woman walks toward the right, toward a large Ferris wheel whose lights are switching on ring by ring, a pink cotton candy on a stick in one hand, looking back over her shoulder at the viewer with a smile and reaching her free hand toward the viewer, the hand held at a natural arm's length at the lower edge of the frame, small in the picture and in sharp focus with five natural fingers.
Setting: open walkway at dusk at the foot of a large Ferris wheel that fills the right side of the frame, its lights switching on, a few lit lamp posts, clean open pavement with nobody else on it. Lighting: dusk sky fading from pink to blue, the Ferris wheel lights glowing warm, soft light on her face from the front.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：摩天輪車廂，她轉回來看你

- 時段：blue dusk｜手上：pink cotton candy on a stick (shots 3-5)｜景別：medium
- 接上一鏡：from the walkway into the cabin: the same dusk, the cotton candy still in her hand

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view from the opposite bench, medium shot from the waist up, the woman on the left third, the window and the lights below on the right. The woman has just sat down on the bench of a Ferris wheel cabin opposite the viewer, her free hand just lowered from reaching toward the viewer and now resting on the bench beside her, the cotton candy in her other hand, smiling at the viewer, the window on the right of the frame.
Setting: inside a small glass Ferris wheel cabin at dusk, the window on the right looking out over the park lights far below, a carousel, a lake and the city lights, with no other Ferris wheel visible outside the window. Lighting: blue dusk sky outside, warm cabin light on her face, the park lights as soft bokeh, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view from the opposite bench, medium shot from the waist up, the woman centred, the lit window behind her on the right. The woman has just sat down on the bench of a Ferris wheel cabin opposite the viewer, her free hand just lowered from reaching toward the viewer and now resting on the bench beside her, the cotton candy in her other hand, smiling at the viewer, the window on the right of the frame.
Setting: inside a small glass Ferris wheel cabin at dusk, the window on the right looking out over the park lights far below, a carousel, a lake and the city lights, with no other Ferris wheel visible outside the window. Lighting: blue dusk sky outside, warm cabin light on her face, the park lights as soft bokeh, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：煙火下，她轉向你

- 時段：night｜手上：none｜景別：close
- 接上一鏡：from the cabin to the ground: the sky is now dark, her hands are free

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium close-up of her face and shoulders on the left third, her face turned toward the viewer, soft bursts of colour and the lit Ferris wheel behind her on the right. The woman stands in the open plaza at night with her face turned toward the viewer in a happy relaxed smile, fireworks bursting in the sky behind her on the right of the frame, her hands empty.
Setting: amusement park plaza at night, large colourful fireworks in the sky on the right, the lit Ferris wheel glowing in the distance on the right, clean open pavement. Lighting: colourful firework light on her hair and shoulders, warm lamp light on her face from the front, her face clearly lit, the background readable.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair, jacket, jeans and earrings.
first-person point of view at eye level, medium close-up of her face and shoulders in the centre, her face turned toward the viewer, fireworks as soft bursts of colour behind her on the right. The woman stands in the open plaza at night with her face turned toward the viewer in a happy relaxed smile, fireworks bursting in the sky behind her on the right of the frame, her hands empty.
Setting: amusement park plaza at night, large colourful fireworks in the sky on the right, the lit Ferris wheel glowing in the distance on the right, clean open pavement. Lighting: colourful firework light on her hair and shoulders, warm lamp light on her face from the front, her face clearly lit, the background readable.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

