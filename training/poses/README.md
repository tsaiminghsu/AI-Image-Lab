# Pose skeleton library

OpenPose skeleton references for the `POSES` tag pool in `generate_character.py`,
fed to ControlNet OpenPose to constrain body position. Exists because Pony-family
checkpoints ignore some pose tags (notably every `lying on ...` variant) from text
alone - see the "姿勢/角度標籤實測" section in the top-level README and
`pose_skeletons.py`.

## What's here

Each library entry is a pair, named by the tag's slug (`pose_skeletons.slugify`):

- `<slug>.json` - canonical source: 18 body keypoints in OpenPose JSON layout
  (`pose_keypoints_2d` = `[x, y, c]` * 18, x/y normalized 0..1, c=1 present / c=0
  hidden) plus metadata (`camera`, `prompt_hint`, `source`).
- `<slug>.png` - 832x1216 skeleton rendered from the JSON, committed so generation
  never needs torch/controlnet_aux. ControlNet resizes it to the latent anyway.

`contact_sheet.png` is a browsing overview of every skeleton.

The 18-point order (0-based) is the COCO/OpenPose body convention the ControlNet
was trained on:

```
0 nose  1 neck  2 Rsho 3 Relb 4 Rwri  5 Lsho 6 Lelb 7 Lwri
8 Rhip  9 Rknee 10 Rankle  11 Lhip 12 Lknee 13 Lankle
14 Reye 15 Leye 16 Rear 17 Lear      (R = the subject's right)
```

## How these were built

Three sources (see `pose_skeletons.py`):

1. **Extracted** (15 poses the checkpoint renders correctly - standing, sitting,
   yoga, walking): `pose_skeletons.py extract --from-pack` ran OpenposePreprocessor
   over the existing `reference_candidates/pose_pack_facedetailer/poses/` renders.
2. **Hand-authored, in the POSES pool** (the 5 `lying on ...` poses): their renders
   were the seated failure, so extracting them would just capture that. The keypoint
   JSON was written by hand using an extracted standing skeleton for limb proportions.
3. **Hand-authored, library-only** (14 more: kneeling, squatting, cross-legged,
   reclining, jumping, waving, back-to-camera ...). See below.

A 2D skeleton is ambiguous about camera height - a body "lying, seen from above"
and one "standing, seen head-on" differ only in proportions - so entries carry a
`prompt_hint` (e.g. "high angle shot from above, lying flat") that the pose paths
append to the prompt.

Gaze/expression tags (`looking at camera`, `slight smile`, ...) are intentionally
not in the library - they aren't body poses.

## Library-only poses (not in `generate_character.POSES`)

The 14 poses added in the second batch exist **only here**, on purpose. The dataset
variations flow (`build_variation_prompt`) picks from `POSES` at random and runs
**without** ControlNet, so putting skeleton-dependent poses into that pool would
just add failures to generated datasets. They are reachable everywhere a skeleton
is actually used:

- `generate_character.py custom --pose <slug>`
- the GUI's 姿勢骨架庫 dropdown
- `pose_pack.py --controlnet` (which folds them in via `pose_pack.library_only_tags`,
  taking their prompt text from each JSON's `tag` field)

If one of them ever proves reliable from text alone, it can be promoted into
`POSES`; nothing else has to change.

## Using a pose

```bash
# one-off generation with a library pose (HQ path)
python generate_character.py custom --prompt "..." --character xinyi \
    --pose lying_on_side_relaxed_head_resting_on_hand --checkpoint cyberrealistic_pony --lora-strength 0

# reference pack for all pose tags via ControlNet
python pose_pack.py --controlnet

# benchmark one pose
python benchmark.py --hq --pose walking
```

In the GUI, the ControlNet accordion has a "姿勢骨架庫" dropdown; an uploaded
photo overrides the dropdown.

## Adding or fixing a pose

Edit the `<slug>.json` keypoints, then re-render:

```bash
ComfyUI\.venv\Scripts\python.exe training\pose_skeletons.py render <slug>
ComfyUI\.venv\Scripts\python.exe training\pose_skeletons.py sheet
```

Rendering needs ComfyUI's interpreter (it imports controlnet_aux's `draw_bodypose`
so colors/limb order match what the ControlNet was trained on). The lookup helpers
(`list_names`/`resolve`/`load_meta`) import nothing heavy.
