# Writing the H3 shot description

`run.py --prompt` takes only the **shot description**. The runner wraps it, unchanged, in the official
MiniMax H3 single-first-frame (I2VA) format:

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] <your description>

overall_soundscape: <--soundscape>

non_diegetic_music: <--music, default N/A>
```

Source: the official guide, `MiniMaxAI/MiniMax-H3` → `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`.

## Priority: user intent > prompt optimization

Optimizing means making the user's request clearer for the model, never changing it.

- Keep every requested action, and add **no** new action, prop or person.
- If the user said nothing about the camera, keep it steady ("The camera holds a steady eye-level shot.");
  a camera move is only added when the user asks for one.
- 「人物不要移動」 → the subject stays in place. Do not write walking or turning. Pass `--constraint static-subject`.
- 「鏡頭固定」/"camera locked" → no camera move at all. Pass `--constraint locked-camera`.
- If the request is vague (「讓她動起來」), pick the smallest natural motion (breathing, blinking, a small
  head turn) rather than inventing a storyline, and say what you chose.
- The constraints make the runner reject a description that contradicts them (`PROMPT_REJECTED`) and
  append one positive sentence ("The camera remains static in a locked-off shot.").

## How to write it

1. **Anchor to the first frame.** Start with what is already visible: who/what, clothing, place,
   framing. This keeps identity and scene stable.
2. **Then the motion**, as plain action in the present tense, one clear action for a 5–8 s clip.
3. **Camera**: type + amplitude + speed, woven into the sentence — Push In/Out, Pan, Truck, Tilt, Pedestal,
   Arc, Tracking, Static. E.g. "The camera pushes in with small amplitude at slow speed."
4. **Positive statements only.** The guide says not to write what is absent. Write "the camera stays
   still", not "no camera movement"; "smooth, steady motion", not "no sudden movement".
5. **English**, 2–5 sentences. Dialogue only if asked: `The woman (S1) says: <d>[Chinese] 你好。</d>`.
6. **Soundscape** (`--soundscape`): 1–2 sentences of ambient and action sound that fit the scene
   (footsteps, room tone, wind, cups clinking). No dialogue there. `N/A` only if the user wants silence.
7. **Music** (`--music`): N/A unless requested; then instruments, tempo and dynamics, not moods.
8. Words the safety screen blocks (age and explicit terms) are refused. "minor" is one of them — write
   "small" or "slight".

## Examples

**User**: 使用這張圖片生成一支 8 秒的影片，人物往鏡頭方向走，保持原本人物與場景。

```
--prompt "The woman in the first frame stands on the same tree-lined street, keeping her cream coat, long dark hair and the soft afternoon light. She walks slowly and naturally toward the camera, arms relaxed, with a calm expression. The camera holds a steady eye-level shot as she approaches."
--soundscape "Light street ambience with distant traffic and her soft footsteps on the pavement."
--duration 8
```

**User**: 一個女生在咖啡廳喝咖啡（the image already shows her at a cafe table）

```
--prompt "The woman sits at the same wooden table in the cozy modern cafe, keeping her outfit and hairstyle. She slowly lifts the white coffee cup, takes a small sip and lowers it with a gentle smile, moving naturally. Soft window light stays constant. The camera holds a steady medium shot."
--soundscape "Quiet cafe ambience with low chatter, a cup touching the saucer and an espresso machine in the distance."
```

Compared with a generic "cinematic" rewrite this keeps the user's one action (a sip), adds no camera move
the user did not ask for, and states the steady parts positively.

**User**: 人物不要移動，只有頭髮被風吹動，鏡頭固定。

```
--prompt "The woman stands on the same cliffside path, keeping her pose, white dress and the overcast sky. A light breeze lifts strands of her hair and the hem of her dress while she keeps her gaze on the horizon."
--soundscape "Steady sea wind and distant waves below the cliff."
--constraint static-subject --constraint locked-camera
```

**User**: 產品慢慢旋轉，鏡頭往前推（no person）

```
--prompt "The matte black wireless speaker sits on the same white pedestal against the soft grey backdrop. It rotates slowly clockwise on the spot, its fabric texture catching the studio light. The camera pushes in with small amplitude at slow speed."
--soundscape "Quiet studio room tone."
```
