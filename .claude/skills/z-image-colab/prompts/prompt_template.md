# From a request to Z-Image jobs

Z-Image Turbo follows plain descriptive sentences, in English or Chinese. It is not a tag model: no
`score_9`, no `masterpiece, best quality`, no weights like `(word:1.3)`. The scripts send your prompt
unchanged and add only the safety negative prompt, so what you write here is exactly what the model reads.

## Order of a prompt

1. **Subject** - what the image is of, with the user's own nouns.
2. **Placement and framing** - centered, close-up, full view, from above, eye level.
3. **Background and setting** - exactly as the user said it.
4. **Light and material** - only as far as it serves what they asked for.
5. **Style** - photo, illustration, 3D render... only if they named one, or it is implied by the use
   (an "ad photo" is a photo).

One to three sentences is enough. Stop when the request is covered.

## What you may and may not change

| The user said | Keep | Do not turn it into |
| --- | --- | --- |
| 白色背景 / white background | "on a plain white background" | a dark, gradient, cinematic or "studio" backdrop |
| 黑色背景 | "against a solid black background" | a moody scene with props |
| 產品放中央 | "centered in the frame" | an off-center "dynamic composition" |
| 固定鏡頭 / 正面 | "straight-on view, eye level" | a tilted, low or dramatic angle |
| 簡單 / 乾淨 | few elements, empty space | added props, reflections, particles |
| 一個人 | one person, and pass `--solo` | a group |
| a named colour, count or object | that colour, count, object | a "better" one |

You may: fix grammar, translate, order the sentences, name the obvious (a mug is ceramic only if they said
so; "a mug" stays "a mug"), and state the aspect ratio through parameters instead of words.

You may not: add a mood, a time of day, a camera move, extra objects, a style or a brand they did not ask
for. If the request is too thin to draw ("a nice picture"), ask one question instead of inventing.

For "N different compositions" the differences are yours to choose, but only along the axis they asked
for (composition), with everything else held: same product, same background, same style.

## Parameters

| Request | Parameter |
| --- | --- |
| 16:9, 橫式, banner, YouTube thumbnail | `--aspect 16:9` (1344x768) |
| 9:16, 直式, Reels / Shorts / story | `--aspect 9:16` (768x1344) |
| 1:1, square, avatar, IG post | `--aspect 1:1` (1024x1024, the default) |
| 4:3 / 3:4, 3:2 / 2:3 | `--aspect 4:3` (1152x896), `3:4`, `3:2` (1216x832), `2:3` |
| an exact size | `--width W --height H` - multiples of 16, each side 512-2048 |
| N images of one prompt | `--count N` (random seeds), or `--seed S --count N` for S..S+N-1 |
| N different prompts | a manifest with N entries (one `submit.py --manifest` call) |
| "same as before" | the recorded `seed` plus the same prompt and size |
| words that must appear in the image | put them in quotes in the prompt and pass `--allow-text` |
| exactly one person | `--solo` |
| something to keep out | `--negative "..."` (appended to the safety negatives, never replacing them) |

Leave `--steps` at 8 and `--cfg` at 2.0: they are the model's turbo settings. cfg below 1.5 is refused,
because ComfyUI would then skip the negative prompt.

## People

Any person in a prompt is an adult; say so when age could be read either way ("a woman in her thirties").
No real, named people. Nothing explicit - this skill only has the `safe` tier.

## Example

Request: 「做一張高級科技產品廣告圖，黑色背景，產品放中央，16:9。」

The user did not say which product, so ask - or, if they named it earlier in the conversation, use it:

```
submit.py --aspect 16:9 --prompt "A premium advertising photo of a matte black wireless earbud case, centered in the frame against a solid black background, soft rim light outlining its edges."
```

Kept: ad photo, black background, centered, 16:9. Added: only the light that makes a black product
visible on black. Not added: reflections on a table, smoke, a tagline, a camera angle.
