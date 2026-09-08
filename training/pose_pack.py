"""Generates a visual reference pack for the pose/angle tag pools in
generate_character.py - one image per tag, everything else held constant, so
the tag itself is the only variable and you can see what each one actually
renders as before picking it for a dataset or a prompt.

Covers POSES + CLOSE_ANGLES + FULL_BODY_ANGLES + BACK_VIEW_ANGLES. Prompts go
through _build_prompt_and_negative() so the pack uses exactly the same syntax
the real pipeline does (character identity prefix, REALISTIC_STYLE, safety
negatives, Pony score-tag auto-prepend) - a tag that looks good here composes
the same way in gen_custom().

Two generation paths, each writing to its own directory so both can coexist:
  (default)        plain txt2img            ~22s/tag - fastest, but at full-body
                                            framing the face gets no refinement
                                            pass and eyes come out asymmetric
  --facedetailer   HQ path, hires off       ~36s/tag - adds the FaceDetailer
                                            face/hand pass, which crops the face
                                            out and re-renders it at guide_size,
                                            fixing the eyes for +14s. Recommended.
Full hires is deliberately not wired up here: measured against --facedetailer it
costs ~85s more per tag and buys resolution, not face quality (the face pass
already runs at its own guide size either way).

Resolution follows _pick_angle()'s own logic: close-up angles on the square
canvas, everything full-body/back-view/pose on FULL_BODY_RESOLUTION, since a
square canvas crops "lying on back" and "full body shot" down to upper body.
NOTE: submit_generation_hq does not honour the requested width/height (see its
docstring vs actual behaviour - with hires off you get HQ_FIRST_PASS[...] , not
what you asked for), so the index records each image's ACTUAL size read back
off the PNG rather than the size we requested.

Re-runnable: existing files are skipped, so an interrupted run resumes.

Usage:
    python pose_pack.py --facedetailer                   # all 39 tags, refined faces
    python pose_pack.py --character mei --seed 12345
    python pose_pack.py --categories poses
    python pose_pack.py --facedetailer --contact-sheet-only
"""

import argparse
import os
import re
import time

import comfyui_client as client
import generate_character as gc

REF_DIR = os.path.join(os.path.dirname(__file__), "reference_candidates")

# (category, tag list, requested (width, height)). Resolutions mirror
# _pick_angle(): CLOSE_ANGLES render fine on the square SDXL canvas, everything
# else needs the portrait canvas or the subject gets cropped at the knees.
CATEGORIES = {
    "poses": (gc.POSES, gc.FULL_BODY_RESOLUTION),
    "close_angles": (gc.CLOSE_ANGLES, (client.WIDTH, client.HEIGHT)),
    "full_body_angles": (gc.FULL_BODY_ANGLES, gc.FULL_BODY_RESOLUTION),
    "back_view_angles": (gc.BACK_VIEW_ANGLES, gc.FULL_BODY_RESOLUTION),
}

# Held the same across every generation so the tag under test is the only
# variable - picked to be as neutral as possible (no competing pose/lighting
# language in the fixed part of the prompt).
FIXED_OUTFIT = "white t-shirt and jeans"
FIXED_LIGHTING = "soft natural window light"
FIXED_BACKGROUND = "plain white studio background"
# Pose tags describe the body; angle tags describe the camera. Each pool gets
# the other axis pinned to a neutral value so the tag isn't fighting a second
# unrelated instruction.
NEUTRAL_ANGLE_FOR_POSES = "full body shot"
NEUTRAL_POSE_FOR_ANGLES = "standing straight"


def out_dir_for(facedetailer):
    return os.path.join(REF_DIR, "pose_pack_facedetailer" if facedetailer else "pose_pack")


def slugify(tag):
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")[:60]


def image_size(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return None


def build_prompt(tag, category, trigger, checkpoint_key):
    """Assembles the scene description the same way the variations flow does
    (angle + pose + outfit + lighting + background), then hands it to the
    shared prompt builder for identity/style/safety/Pony handling."""
    if category == "poses":
        pose, angle = tag, NEUTRAL_ANGLE_FOR_POSES
    else:
        pose, angle = NEUTRAL_POSE_FOR_ANGLES, tag
    scene = f"{angle}, {pose}, {FIXED_OUTFIT}, {FIXED_LIGHTING}, {FIXED_BACKGROUND}"
    return gc._build_prompt_and_negative(
        scene, None, "safe", trigger, None, None, checkpoint_key,
    )


def generate(args, out_dir):
    checkpoint_file = client.CHECKPOINTS[args.checkpoint]
    gc.get_character(args.character)  # fails loudly on an unknown trigger
    is_pony = args.checkpoint in client.PONY_CHECKPOINTS

    results = []
    todo = [(cat, tag) for cat in args.categories for tag in CATEGORIES[cat][0]]
    path_label = "HQ facedetailer (hires off)" if args.facedetailer else "plain txt2img"
    print(f"{len(todo)} tags | character={args.character} checkpoint={args.checkpoint} "
          f"seed={args.seed} | {path_label}\n", flush=True)

    for i, (category, tag) in enumerate(todo, 1):
        width, height = CATEGORIES[category][1]
        cat_dir = os.path.join(out_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        out_path = os.path.join(cat_dir, f"{slugify(tag)}.png")

        if os.path.exists(out_path) and not args.force:
            print(f"[{i}/{len(todo)}] skip (exists) {category}/{tag}", flush=True)
            results.append({"category": category, "tag": tag, "path": out_path,
                            "size": image_size(out_path), "elapsed": None})
            continue

        prompt, negative = build_prompt(tag, category, args.character, args.checkpoint)
        # Style slider LoRA was tuned for photoreal SDXL, not Pony - see
        # comfyui_client.CHECKPOINTS notes.
        lora_strength = 0.0 if is_pony else None
        t0 = time.time()
        try:
            if args.facedetailer:
                raw = client.submit_generation_hq(
                    prompt=prompt, negative_prompt=negative, seed=args.seed,
                    filename_prefix=f"posepack_{category}_{slugify(tag)}",
                    width=width, height=height, checkpoint=checkpoint_file,
                    lora_strength=lora_strength,
                    hires=False, use_facedetailer=True,
                )
            else:
                raw = client.submit_txt2img_generation(
                    prompt=prompt, negative_prompt=negative, seed=args.seed,
                    filename_prefix=f"posepack_{category}_{slugify(tag)}",
                    width=width, height=height, checkpoint=checkpoint_file,
                    lora_strength=lora_strength,
                )
            elapsed = time.time() - t0
            os.replace(raw, out_path)
            size = image_size(out_path)
            print(f"[{i}/{len(todo)}] OK {elapsed:5.1f}s {size[0]}x{size[1]} {category}/{tag}", flush=True)
            results.append({"category": category, "tag": tag, "path": out_path,
                            "size": size, "elapsed": elapsed})
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[{i}/{len(todo)}] FAIL {elapsed:5.1f}s {category}/{tag}: {exc}", flush=True)
            results.append({"category": category, "tag": tag, "path": None,
                            "size": None, "elapsed": elapsed})
    return results


def build_contact_sheets(categories, out_dir):
    """One labelled contact sheet per category - the actual browsing artifact;
    the per-tag PNGs are for looking at a specific tag up close."""
    from PIL import Image, ImageDraw, ImageFont

    THUMB_W, COLS, LABEL_H, PAD = 320, 5, 26, 8
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    made = []
    for category in categories:
        cat_dir = os.path.join(out_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        tags = [t for t in CATEGORIES[category][0]
                if os.path.exists(os.path.join(cat_dir, f"{slugify(t)}.png"))]
        if not tags:
            continue

        # Thumb aspect comes from what was actually produced, not what we asked
        # for - those differ on the HQ path (see module docstring).
        first = image_size(os.path.join(cat_dir, f"{slugify(tags[0])}.png"))
        thumb_h = round(THUMB_W * first[1] / first[0]) if first else THUMB_W
        cell_h = thumb_h + LABEL_H
        rows = (len(tags) + COLS - 1) // COLS
        sheet = Image.new("RGB",
                          (COLS * THUMB_W + (COLS + 1) * PAD, rows * cell_h + (rows + 1) * PAD),
                          "white")
        draw = ImageDraw.Draw(sheet)

        for idx, tag in enumerate(tags):
            r, c = divmod(idx, COLS)
            x = PAD + c * (THUMB_W + PAD)
            y = PAD + r * (cell_h + PAD)
            with Image.open(os.path.join(cat_dir, f"{slugify(tag)}.png")) as im:
                sheet.paste(im.convert("RGB").resize((THUMB_W, thumb_h), Image.LANCZOS), (x, y))
            label = tag if len(tag) <= 46 else tag[:43] + "..."
            draw.text((x + 2, y + thumb_h + 6), label, fill="black", font=font)

        path = os.path.join(out_dir, f"contact_sheet_{category}.png")
        sheet.save(path)
        made.append((category, path, len(tags)))
        print(f"contact sheet: {path} ({len(tags)} tags)", flush=True)
    return made


def write_index(results, args, sheets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "INDEX.md")
    timed = [r for r in results if r["elapsed"] is not None]
    path_label = ("HQ path with FaceDetailer, hires off"
                  if args.facedetailer else "plain txt2img (no FaceID / hires / FaceDetailer)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Pose / angle tag reference pack\n\n")
        f.write(f"- Character: `{args.character}` | Checkpoint: `{args.checkpoint}` | "
                f"Seed: `{args.seed}` (fixed across every tag)\n")
        f.write(f"- Path: {path_label}\n")
        f.write(f"- Same for every tag: `{FIXED_OUTFIT}` / `{FIXED_LIGHTING}` / `{FIXED_BACKGROUND}`; "
                f"pose tags use angle `{NEUTRAL_ANGLE_FOR_POSES}`, angle tags use pose "
                f"`{NEUTRAL_POSE_FOR_ANGLES}`\n")
        f.write("- Note: the character profile's own `style`/`appearance` terms are in the prompt too "
                "(via `character_base_prompt`) and in practice win over the outfit/background terms "
                "above - same as the real pipeline. They're identical across every tag, so tag-to-tag "
                "comparison is still clean.\n")
        f.write("- Resolution columns are the ACTUAL produced size; `submit_generation_hq` does not "
                "honour the requested width/height (see pose_pack.py module docstring).\n")
        if timed:
            total = sum(r["elapsed"] for r in timed)
            f.write(f"- Generated {len(timed)} images in {total/60:.1f} min "
                    f"(avg {total/len(timed):.1f}s)\n")
        f.write("\nTag pools live in `generate_character.py` - regenerate with "
                "`python training/pose_pack.py"
                f"{' --facedetailer' if args.facedetailer else ''}`.\n")
        for category, sheet_path, n in sheets:
            f.write(f"\n## {category} ({n})\n\n")
            f.write(f"![{category}]({os.path.basename(sheet_path)})\n\n")
            f.write("| Tag | File | Resolution | Time |\n|---|---|---|---|\n")
            for r in [x for x in results if x["category"] == category]:
                t = f"{r['elapsed']:.1f}s" if r["elapsed"] is not None else "cached"
                status = f"`{category}/{slugify(r['tag'])}.png`" if r["path"] else "**FAILED**"
                res = f"{r['size'][0]}x{r['size'][1]}" if r["size"] else "-"
                f.write(f"| {r['tag']} | {status} | {res} | {t} |\n")
    print(f"index: {path}", flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description="Generate the pose/angle tag reference pack")
    p.add_argument("--character", default="xinyi", help="CHARACTERS trigger to render the tags with")
    p.add_argument("--checkpoint", default=gc.DEFAULT_CUSTOM_CHECKPOINT, choices=sorted(client.CHECKPOINTS))
    p.add_argument("--seed", type=int, default=88001, help="fixed across all tags so the tag is the only variable")
    p.add_argument("--categories", nargs="+", default=sorted(CATEGORIES), choices=sorted(CATEGORIES))
    p.add_argument("--facedetailer", action="store_true",
                   help="run the HQ path with the FaceDetailer face/hand pass (hires off) - "
                        "+14s per tag, fixes the asymmetric eyes of the plain path")
    p.add_argument("--force", action="store_true", help="regenerate images that already exist")
    p.add_argument("--contact-sheet-only", action="store_true", help="rebuild sheets/index from existing PNGs")
    args = p.parse_args()

    if args.checkpoint in client.SD15_CHECKPOINTS:
        raise SystemExit(f"{args.checkpoint} is SD1.5; this pack targets the SDXL/Pony txt2img path")

    out_dir = out_dir_for(args.facedetailer)
    if args.contact_sheet_only:
        results = []
        for c in args.categories:
            for t in CATEGORIES[c][0]:
                fp = os.path.join(out_dir, c, f"{slugify(t)}.png")
                results.append({"category": c, "tag": t,
                                "path": fp if os.path.exists(fp) else None,
                                "size": image_size(fp), "elapsed": None})
    else:
        results = generate(args, out_dir)

    sheets = build_contact_sheets(args.categories, out_dir)
    write_index(results, args, sheets, out_dir)
    ok = sum(1 for r in results if r["path"] and os.path.exists(r["path"]))
    print(f"\n{ok}/{len(results)} tags rendered -> {out_dir}")


if __name__ == "__main__":
    main()
