# ChatGPT 生圖提示詞：一起去野餐（promo_09_picnic）

由 `training/activity_promo.py gpt-prompts` 從 `story.json` 產生，不要手改；要改請改 story.json 再重跑。

## 做法

1. 開**一個**新對話，從「第 0 步」開始，**一次只貼一段**，滿意了再貼下一段。同一個對話裡臉才會一路連著。
2. 第 0 步滿意的那張存成 `projects/activity_promo/reference/ruoxi_reference.png`。
3. 每個鏡頭先用「主要取景」；想要另一張再貼「另一種取景」。每個鏡頭放 1–2 張就夠。
4. 存檔位置：`projects/activity_promo/promo_09_picnic/images/scene_NN/candidate_01.png`（主要取景）、`candidate_02.png`（另一種取景）；有「更多取景」的鏡頭再用 `candidate_03.png`、`candidate_04.png`。放好告訴我，記錄檔和格式轉換我來處理。
5. 生出來有任何一項就重生：

   - not the same face as the reference shot
   - a second person in frame
   - wrong outfit, or the story's prop missing / present in the wrong shots (story.json props)
   - wrong time of day for this shot
   - broken hands or earrings, painted text or signage

## 第 0 步：角色定裝照

已經有若曦的定裝照（遊樂園那支用的，或任何一張臉最清楚的圖）的話，把它上傳到新對話，跳過這一步，直接從鏡頭 1 開始；每個鏡頭的提示詞都寫了這支的衣服。

```text
Create a photorealistic portrait photo of an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, a navy-and-white striped long-sleeved top with a round neckline, beige trousers, small pearl drop earrings, for use as a consistent character reference. Natural skin texture, soft daylight, neutral background, waist-up, looking at the camera with a relaxed smile. Vertical 9:16 frame. Only one person, no text, no logos.
```

## 鏡頭 1：公園草地，她抱著野餐墊

- 時段：midday, bright｜手上：none｜景別：medium
- 接上一鏡：（開場）

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third holding a rolled picnic blanket against her chest, the pond on the right. The woman stands on the lawn of a Parisian park holding a rolled checked picnic blanket against her chest, smiling at the viewer, the pond on the right of the frame.
Setting: a wide lawn in a Parisian park at midday with pale gravel paths and clipped trees, a pond on the right side of the frame, cream stone buildings far behind the trees, nobody else in sight, no readable signs. Lighting: bright midday sunlight with soft shade from the trees, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium shot from the waist up, the woman centred holding a rolled picnic blanket against her chest, the pond on the right. The woman stands on the lawn of a Parisian park holding a rolled checked picnic blanket against her chest, smiling at the viewer, the pond on the right of the frame.
Setting: a wide lawn in a Parisian park at midday with pale gravel paths and clipped trees, a pond on the right side of the frame, cream stone buildings far behind the trees, nobody else in sight, no readable signs. Lighting: bright midday sunlight with soft shade from the trees, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred holding a rolled picnic blanket against her chest, the pond on the right. The woman stands on the lawn of a Parisian park holding a rolled checked picnic blanket against her chest, smiling at the viewer, the pond on the right of the frame.
Setting: a wide lawn in a Parisian park at midday with pale gravel paths and clipped trees, a pond on the right side of the frame, cream stone buildings far behind the trees, nobody else in sight, no readable signs. Lighting: bright midday sunlight with soft shade from the trees, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre holding a rolled picnic blanket against her chest, the pond on the right. The woman stands on the lawn of a Parisian park holding a rolled checked picnic blanket against her chest, smiling at the viewer, the pond on the right of the frame.
Setting: a wide lawn in a Parisian park at midday with pale gravel paths and clipped trees, a pond on the right side of the frame, cream stone buildings far behind the trees, nobody else in sight, no readable signs. Lighting: bright midday sunlight with soft shade from the trees, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 2：池邊，她鋪好野餐墊

- 時段：midday, bright｜手上：basket｜景別：close
- 接上一鏡：from the lawn to the water: the same Parisian park and midday light, the blanket now spread by the pond

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third seated on the picnic blanket with a wicker basket beside her, the pond on the right. The woman sits on a checked picnic blanket at the edge of the pond in a Parisian park with a wicker picnic basket beside her, looking up at the viewer with a relaxed smile, the water on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: bright midday sunlight with a soft sparkle off the water on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred seated on the picnic blanket with a wicker basket beside her, the pond on the right. The woman sits on a checked picnic blanket at the edge of the pond in a Parisian park with a wicker picnic basket beside her, looking up at the viewer with a relaxed smile, the water on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: bright midday sunlight with a soft sparkle off the water on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred seated on the picnic blanket with a wicker basket beside her, the pond on the right. The woman sits on a checked picnic blanket at the edge of the pond in a Parisian park with a wicker picnic basket beside her, looking up at the viewer with a relaxed smile, the water on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: bright midday sunlight with a soft sparkle off the water on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre seated on the picnic blanket with a wicker basket beside her, the pond on the right. The woman sits on a checked picnic blanket at the edge of the pond in a Parisian park with a wicker picnic basket beside her, looking up at the viewer with a relaxed smile, the water on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: bright midday sunlight with a soft sparkle off the water on the right, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 3：她打開野餐籃

- 時段：afternoon, warm｜手上：basket｜景別：medium
- 接上一鏡：from sitting down to the food: the same pond in the Parisian park, the light one step later

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third holding the open picnic basket on her lap, the pond on the right. The woman holds the wicker picnic basket open on her lap on the picnic blanket in a Parisian park, baguette sandwiches inside, looking at the viewer with delighted surprise, the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight with soft shade, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium shot from the waist up, the woman centred holding the open picnic basket on her lap, the pond on the right. The woman holds the wicker picnic basket open on her lap on the picnic blanket in a Parisian park, baguette sandwiches inside, looking at the viewer with delighted surprise, the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight with soft shade, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred holding the open picnic basket on her lap, the pond on the right. The woman holds the wicker picnic basket open on her lap on the picnic blanket in a Parisian park, baguette sandwiches inside, looking at the viewer with delighted surprise, the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight with soft shade, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre holding the open picnic basket on her lap, the pond on the right. The woman holds the wicker picnic basket open on her lap on the picnic blanket in a Parisian park, baguette sandwiches inside, looking at the viewer with delighted surprise, the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight with soft shade, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 4：躺著看雲

- 時段：afternoon, warm｜手上：basket｜景別：close
- 接上一鏡：from the food to the clouds: the same blanket by the pond, she is lying back now

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third lying on the blanket, pointing up at the sky, the pond on the right. The woman lies on the picnic blanket in a Parisian park, framed from the shoulders up, one arm raised to point at the sky, her face turned toward the viewer with a playful smile, the wicker picnic basket at the edge of the blanket and the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight from above, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred lying on the blanket, pointing up at the sky, the pond on the right. The woman lies on the picnic blanket in a Parisian park, framed from the shoulders up, one arm raised to point at the sky, her face turned toward the viewer with a playful smile, the wicker picnic basket at the edge of the blanket and the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight from above, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred lying on the blanket, pointing up at the sky, the pond on the right. The woman lies on the picnic blanket in a Parisian park, framed from the shoulders up, one arm raised to point at the sky, her face turned toward the viewer with a playful smile, the wicker picnic basket at the edge of the blanket and the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight from above, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre lying on the blanket, pointing up at the sky, the pond on the right. The woman lies on the picnic blanket in a Parisian park, framed from the shoulders up, one arm raised to point at the sky, her face turned toward the viewer with a playful smile, the wicker picnic basket at the edge of the blanket and the pond on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm afternoon sunlight from above, her face fully lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 5：金色的草地，她抱著膝蓋

- 時段：golden hour, low sun｜手上：none｜景別：medium
- 接上一鏡：from the clouds to the gold: the same blanket by the pond, the sun now low

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium shot from the waist up, the woman on the left third sitting with her arms around her knees, the pond on the right. The woman sits on the picnic blanket in a Parisian park hugging her knees in the golden light, smiling at the viewer, the pond glowing on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: low golden-hour sunlight from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium shot from the waist up, the woman centred sitting with her arms around her knees, the pond on the right. The woman sits on the picnic blanket in a Parisian park hugging her knees in the golden light, smiling at the viewer, the pond glowing on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: low golden-hour sunlight from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view, slightly low camera angle, medium shot from the waist up, the woman centred sitting with her arms around her knees, the pond on the right. The woman sits on the picnic blanket in a Parisian park hugging her knees in the golden light, smiling at the viewer, the pond glowing on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: low golden-hour sunlight from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level from a little further to the left, medium shot from the waist up, the woman just left of centre sitting with her arms around her knees, the pond on the right. The woman sits on the picnic blanket in a Parisian park hugging her knees in the golden light, smiling at the viewer, the pond glowing on the right of the frame.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: low golden-hour sunlight from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

## 鏡頭 6：夕陽在水面上

- 時段：sunset, orange｜手上：none｜景別：close
- 接上一鏡：from the gold to the sunset: the same pond in the Parisian park, the sky now orange

主要取景（存成 candidate_01.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman on the left third turned from the water to smile at the viewer, the sunset on the pond on the right. The woman sits by the pond in a Parisian park at sunset, turned from the glowing water on the right of the frame to smile softly at the viewer.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm orange sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

另一種取景（存成 candidate_02.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level, medium close-up of her face and shoulders, the woman centred turned from the water to smile at the viewer, the sunset on the pond on the right. The woman sits by the pond in a Parisian park at sunset, turned from the glowing water on the right of the frame to smile softly at the viewer.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm orange sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 C（存成 candidate_03.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view, slightly low camera angle, medium close-up of her face and shoulders, the woman centred turned from the water to smile at the viewer, the sunset on the pond on the right. The woman sits by the pond in a Parisian park at sunset, turned from the glowing water on the right of the frame to smile softly at the viewer.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm orange sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

更多取景 D（存成 candidate_04.png）：

```text
Same woman as the reference image above — same face, hair and pearl drop earrings. She wears a navy-and-white striped long-sleeved top with a round neckline that covers the collarbone, and beige trousers.
first-person point of view at eye level from a little further to the left, medium close-up of her face and shoulders, the woman just left of centre turned from the water to smile at the viewer, the sunset on the pond on the right. The woman sits by the pond in a Parisian park at sunset, turned from the glowing water on the right of the frame to smile softly at the viewer.
Setting: the grassy edge of a pond in a Parisian park, the calm water on the right side of the frame, clipped trees and cream stone buildings far behind, a checked picnic blanket on the grass, nobody else in sight, no readable signs. Lighting: warm orange sunset light from the right, her face clearly lit.
Photorealistic, natural skin texture, shallow depth of field. Her face is clearly lit. Only one person in the frame, no crowd, no text or signs, no logos. Vertical 9:16.
```

