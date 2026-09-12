# Scene library

Curated backgrounds for the GUI's 「🎯 圖片選擇生圖」 tab, so a scene can be picked by
clicking a picture instead of typing a prompt. See `training/scene_library.py`.

## What's here

- `scenes.json` - every scene, **in gallery display order**. One file rather than one per
  scene (unlike `training/poses/`): each entry is five short fields with no per-entry payload,
  the order is part of the data, and curating the set should be one reviewable diff.
- `thumbs/<slug>.jpg` - 256px preview, committed. The gallery draws from these, so opening the
  tab needs neither a GPU nor a running ComfyUI - same reasoning as the committed skeleton PNGs.

Each entry:

```json
{
  "slug": "cozy_cafe",            // filesystem/id, [a-z0-9_]+, unique
  "name_zh": "溫馨咖啡廳",         // what the user reads under the tile
  "prompt": "cozy cafe interior", // what CLIP reads - English, or it is ignored
  "lighting": "warm indoor lighting",
  "tier": "safe",                 // "suggestive" entries are hidden at the safe tier
  "thumb_seed": 7102              // pinned so a re-render reproduces the same thumbnail
}
```

`scene_text(slug)` is what reaches the prompt: `"{prompt}, {lighting}"`. The texts are curated
pairings drawn from `generate_character.py`'s `BACKGROUNDS`/`SUGGESTIVE_BACKGROUNDS`/`LIGHTINGS`
pools - one lighting chosen per scene, not their 48-way product, because a picker needs a fixed
labelled set with a picture each, while those pools exist for the random batch generators.

`tier` is a visibility rule, not a content filter: the safe tier simply does not offer
suggestive scenes, so the picker can never hand the sampler a beach-in-swimwear setting that the
tier's own negatives would then be fighting. The safety negatives themselves are unchanged and
never come from here.

## Adding a scene

1. Add the entry to `scenes.json` (append, or insert where you want it in the gallery). Give it
   a `thumb_seed` no other scene uses.
2. Render its thumbnail - needs ComfyUI running:

```bash
ComfyUI\.venv\Scripts\python.exe training\scene_library.py render-thumbs
ComfyUI\.venv\Scripts\python.exe training\scene_library.py list
```

`render-thumbs` skips scenes that already have a thumbnail (`--force` re-renders, `--only <slug>`
narrows). It renders at 768x768 with juggernaut - chosen over the GUI's Pony default because
these are empty rooms and streets and Pony's prior is character-centric - then downscales to a
256px JPEG. A "no people" negative is applied: an "empty cafe" prompt otherwise still tends to
seat someone in it, and a thumbnail with a stranger in it reads as the character you picked.

Two things about this batch are shaped by the 8GB card, both measured 2026-09-12 rather than
assumed, and both are why it renders at 768 and frees VRAM between every scene:

- at SDXL's native 1024 the 7th consecutive scene wedged in `VAEDecode` - 7.9/8.2 GB, GPU pinned
  at 100%, no OOM raised, just silent spilling to shared memory over the PCIe x1 link
- at 768 it got to the 10th before doing the same, and that time ComfyUI stopped answering HTTP
  for long enough (132s) that the client's retry budget gave up

So the pressure is cumulative fragmentation across generations in one ComfyUI session, not any
single scene. `client.free_vram()` before each render costs a ~20s checkpoint reload per scene
and makes the batch finish unattended, which is the right trade for something run once.

3. Commit the JSON and the JPEG together. `tests/test_scene_library.py` fails on a scene with no
   committed thumbnail, since it would otherwise show as a blank tile only when someone opens
   the tab.
