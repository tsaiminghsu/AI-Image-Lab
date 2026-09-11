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
import sys
import time

import comfyui_client as client
import generate_character as gc
import pose_skeletons

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


def out_dir_for(facedetailer, controlnet=False, strength=None):
    if controlnet:
        name = "pose_pack_controlnet"
        # keep strength sweeps in separate dirs so they coexist for comparison
        if strength is not None and abs(strength - client.CONTROLNET_STRENGTH) > 1e-6:
            name += f"_s{strength:g}"
        return os.path.join(REF_DIR, name)
    return os.path.join(REF_DIR, "pose_pack_facedetailer" if facedetailer else "pose_pack")


# Kept as a module-level alias so callers/tests using pose_pack.slugify keep
# working; the single source of truth lives in pose_skeletons.
slugify = pose_skeletons.slugify


def image_size(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return None


def library_only_tags(categories, controlnet):
    """Skeletons in training/poses/ that aren't in the POSES pool.

    Those poses (kneeling, squatting, reclining, jumping ...) are deliberately
    kept out of gc.POSES: the dataset variations flow picks POSES at random
    WITHOUT ControlNet, so seeding it with poses that need a skeleton would just
    add failures there. They still deserve a reference render, so the ControlNet
    pack picks them up here. Their prompt text comes from the JSON's `tag`."""
    if not controlnet or "poses" not in categories:
        return []
    known = {slugify(t) for t in CATEGORIES["poses"][0]}
    extra = []
    for slug in pose_skeletons.list_names():
        if slug in known:
            continue
        tag = pose_skeletons.load_meta(slug).get("tag") or slug.replace("_", " ")
        extra.append(("poses", tag))
    return extra


def build_prompt(tag, category, trigger, checkpoint_key, extra=None):
    """Assembles the scene description the same way the variations flow does
    (angle + pose + outfit + lighting + background), then hands it to the
    shared prompt builder for identity/style/safety/Pony handling. `extra` is
    an optional camera/pose hint appended to the scene - used by the ControlNet
    path to pair a skeleton with its intended camera (see pose_skeletons)."""
    if category == "poses":
        pose, angle = tag, NEUTRAL_ANGLE_FOR_POSES
    else:
        pose, angle = NEUTRAL_POSE_FOR_ANGLES, tag
    scene = f"{angle}, {pose}, {FIXED_OUTFIT}, {FIXED_LIGHTING}, {FIXED_BACKGROUND}"
    if extra:
        scene = f"{scene}, {extra}"
    return gc._build_prompt_and_negative(
        scene, None, "safe", trigger, None, None, checkpoint_key,
    )


def generate(args, out_dir):
    checkpoint_file = client.CHECKPOINTS[args.checkpoint]
    gc.get_character(args.character)  # fails loudly on an unknown trigger
    is_pony = args.checkpoint in client.PONY_CHECKPOINTS

    results = []
    todo = [(cat, tag) for cat in args.categories for tag in CATEGORIES[cat][0]]
    todo += library_only_tags(args.categories, args.controlnet)
    if args.only:
        todo = [(c, t) for (c, t) in todo if any(s.lower() in t.lower() for s in args.only)]
    if args.controlnet:
        path_label = f"HQ facedetailer + ControlNet OpenPose (strength {args.controlnet_strength:g})"
    elif args.facedetailer:
        path_label = "HQ facedetailer (hires off)"
    else:
        path_label = "plain txt2img"
    print(f"{len(todo)} tags | character={args.character} checkpoint={args.checkpoint} "
          f"seed={args.seed} | {path_label}\n", flush=True)

    for i, (category, tag) in enumerate(todo, 1):
        width, height = CATEGORIES[category][1]
        cat_dir = os.path.join(out_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        out_path = os.path.join(cat_dir, f"{slugify(tag)}.png")

        # ControlNet mode: use the library skeleton if this tag has one; tags
        # without a skeleton (gaze/expression) fall back to the plain HQ path
        # and are marked text-only in the index rather than skipped.
        skel_path = pose_skeletons.resolve(tag) if args.controlnet else None
        if skel_path and pose_skeletons.crop_fraction(tag, width, height) > pose_skeletons.CANVAS_CROP_TOLERANCE:
            # Can't happen for the poses category (FULL_BODY_RESOLUTION matches the
            # skeletons), but a cropped hint is silently destructive, so never ship
            # one - drop to text-only and say so.
            print(f"[{i}/{len(todo)}] {category}/{tag}: skeleton dropped, "
                  f"{width}x{height} would crop it", flush=True)
            skel_path = None

        if os.path.exists(out_path) and not args.force:
            print(f"[{i}/{len(todo)}] skip (exists) {category}/{tag}", flush=True)
            results.append({"category": category, "tag": tag, "path": out_path,
                            "size": image_size(out_path), "elapsed": None, "skeleton": skel_path})
            continue

        extra = pose_skeletons.load_meta(tag).get("prompt_hint") if skel_path else None
        prompt, negative = build_prompt(tag, category, args.character, args.checkpoint, extra=extra)
        # Style slider LoRA was tuned for photoreal SDXL, not Pony - see
        # comfyui_client.CHECKPOINTS notes.
        lora_strength = 0.0 if is_pony else None
        t0 = time.time()
        try:
            if args.controlnet and skel_path:
                pose_name = client.upload_reference_image(skel_path)
                raw = client.submit_generation_hq(
                    prompt=prompt, negative_prompt=negative, seed=args.seed,
                    filename_prefix=f"posepack_{category}_{slugify(tag)}",
                    width=width, height=height, checkpoint=checkpoint_file,
                    lora_strength=lora_strength,
                    pose_image_filename=pose_name, pose_is_skeleton=True,
                    controlnet_strength=args.controlnet_strength,
                    hires=False, use_facedetailer=True,
                )
            elif args.facedetailer or args.controlnet:
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
            marker = " [skel]" if (args.controlnet and skel_path) else (" [text-only]" if args.controlnet else "")
            print(f"[{i}/{len(todo)}] OK {elapsed:5.1f}s {size[0]}x{size[1]} {category}/{tag}{marker}", flush=True)
            results.append({"category": category, "tag": tag, "path": out_path,
                            "size": size, "elapsed": elapsed, "skeleton": skel_path})
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[{i}/{len(todo)}] FAIL {elapsed:5.1f}s {category}/{tag}: {exc}", flush=True)
            results.append({"category": category, "tag": tag, "path": None,
                            "size": None, "elapsed": elapsed, "skeleton": skel_path})
    return results


def build_contact_sheets(categories, out_dir, skeleton_lookup=None, tags_by_category=None):
    """One labelled contact sheet per category - the actual browsing artifact;
    the per-tag PNGs are for looking at a specific tag up close.

    When skeleton_lookup ({tag: skeleton_path|None}) is given (ControlNet mode),
    each cell shows the input skeleton next to the render so you can check
    limb-for-limb that the pose was followed; tags with no skeleton get a grey
    'text only' placeholder in the skeleton slot."""
    from PIL import Image, ImageDraw, ImageFont

    THUMB_W, LABEL_H, PAD = 320, 26, 8
    COLS = 3 if skeleton_lookup else 5
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    made = []
    for category in categories:
        cat_dir = os.path.join(out_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        # Prefer the tags actually generated (that's what carries library-only
        # poses); fall back to the category pool.
        pool = (tags_by_category or {}).get(category) or CATEGORIES[category][0]
        tags = [t for t in pool
                if os.path.exists(os.path.join(cat_dir, f"{slugify(t)}.png"))]
        if not tags:
            continue

        # Thumb aspect comes from what was actually produced, not what we asked
        # for - those differ on the HQ path (see module docstring).
        first = image_size(os.path.join(cat_dir, f"{slugify(tags[0])}.png"))
        thumb_h = round(THUMB_W * first[1] / first[0]) if first else THUMB_W
        cell_w = (2 * THUMB_W + PAD) if skeleton_lookup else THUMB_W
        cell_h = thumb_h + LABEL_H
        rows = (len(tags) + COLS - 1) // COLS
        sheet = Image.new("RGB",
                          (COLS * cell_w + (COLS + 1) * PAD, rows * cell_h + (rows + 1) * PAD),
                          "white")
        draw = ImageDraw.Draw(sheet)

        for idx, tag in enumerate(tags):
            r, c = divmod(idx, COLS)
            x = PAD + c * (cell_w + PAD)
            y = PAD + r * (cell_h + PAD)
            if skeleton_lookup is not None:
                skel = skeleton_lookup.get(tag)
                if skel and os.path.exists(skel):
                    with Image.open(skel) as sk:
                        sheet.paste(sk.convert("RGB").resize((THUMB_W, thumb_h), Image.LANCZOS), (x, y))
                else:
                    ph = Image.new("RGB", (THUMB_W, thumb_h), (40, 40, 40))
                    ImageDraw.Draw(ph).text((8, thumb_h // 2 - 8), "text only", fill="white", font=font)
                    sheet.paste(ph, (x, y))
                x += THUMB_W + PAD
            with Image.open(os.path.join(cat_dir, f"{slugify(tag)}.png")) as im:
                sheet.paste(im.convert("RGB").resize((THUMB_W, thumb_h), Image.LANCZOS), (x, y))
            label = tag if len(tag) <= 46 else tag[:43] + "..."
            draw.text((PAD + c * (cell_w + PAD) + 2, y + thumb_h + 6), label, fill="black", font=font)

        path = os.path.join(out_dir, f"contact_sheet_{category}.png")
        sheet.save(path)
        made.append((category, path, len(tags)))
        print(f"contact sheet: {path} ({len(tags)} tags)", flush=True)
    return made


def write_index(results, args, sheets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "INDEX.md")
    timed = [r for r in results if r["elapsed"] is not None]
    if args.controlnet:
        path_label = (f"HQ path with FaceDetailer + ControlNet OpenPose "
                      f"(skeleton library, strength {args.controlnet_strength:g}, hires off)")
    elif args.facedetailer:
        path_label = "HQ path with FaceDetailer, hires off"
    else:
        path_label = "plain txt2img (no FaceID / hires / FaceDetailer)"
    regen_flag = " --controlnet" if args.controlnet else (" --facedetailer" if args.facedetailer else "")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Pose / angle tag reference pack\n\n")
        loaded = client.effective_checkpoint(client.CHECKPOINTS[args.checkpoint], uses_pose=bool(args.controlnet))
        f.write(f"- Character: `{args.character}` | Checkpoint: `{args.checkpoint}` (file `{loaded}`) | "
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
        if args.controlnet:
            f.write("- Skeletons come from the `training/poses/` library (fed to ControlNet directly, "
                    "no preprocessor). Tags without a skeleton are generated text-only for comparison.\n")
        if timed:
            total = sum(r["elapsed"] for r in timed)
            f.write(f"- Generated {len(timed)} images in {total/60:.1f} min "
                    f"(avg {total/len(timed):.1f}s)\n")
            if args.controlnet and len(timed) >= 2:
                # first generation is cold (loads the 774MB control-lora); the
                # rest are the number that matters for the per-image budget
                cold = timed[0]["elapsed"]
                rest = sorted(r["elapsed"] for r in timed[1:])
                median = rest[len(rest) // 2]
                f.write(f"- ControlNet timing: first tag {cold:.1f}s (cold, includes control-lora load), "
                        f"median of the rest {median:.1f}s\n")
        f.write("\nTag pools live in `generate_character.py` - regenerate with "
                f"`python training/pose_pack.py{regen_flag}`.\n")
        for category, sheet_path, n in sheets:
            f.write(f"\n## {category} ({n})\n\n")
            f.write(f"![{category}]({os.path.basename(sheet_path)})\n\n")
            if args.controlnet:
                f.write("| Tag | File | Skeleton | Resolution | Time |\n|---|---|---|---|---|\n")
            else:
                f.write("| Tag | File | Resolution | Time |\n|---|---|---|---|\n")
            for r in [x for x in results if x["category"] == category]:
                t = f"{r['elapsed']:.1f}s" if r["elapsed"] is not None else "cached"
                status = f"`{category}/{slugify(r['tag'])}.png`" if r["path"] else "**FAILED**"
                res = f"{r['size'][0]}x{r['size'][1]}" if r["size"] else "-"
                if args.controlnet:
                    skel = r.get("skeleton")
                    skel_cell = f"`../../poses/{slugify(r['tag'])}.png`" if skel else "none (text only)"
                    f.write(f"| {r['tag']} | {status} | {skel_cell} | {res} | {t} |\n")
                else:
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
    p.add_argument("--controlnet", action="store_true",
                   help="feed each pose tag its training/poses/ skeleton through ControlNet OpenPose "
                        "(implies the FaceDetailer HQ path); tags without a skeleton run text-only. "
                        "Defaults --categories to just 'poses'.")
    p.add_argument("--controlnet-strength", type=float, default=client.CONTROLNET_STRENGTH,
                   help="ControlNet conditioning strength (0=ignored, 1=rigid); sweep to pick a default")
    p.add_argument("--only", nargs="+", help="restrict to tags containing any of these substrings")
    p.add_argument("--force", action="store_true", help="regenerate images that already exist")
    p.add_argument("--contact-sheet-only", action="store_true", help="rebuild sheets/index from existing PNGs")
    args = p.parse_args()

    if args.checkpoint in client.SD15_CHECKPOINTS:
        raise SystemExit(f"{args.checkpoint} is SD1.5; this pack targets the SDXL/Pony txt2img path")
    # ControlNet skeletons only exist for the POSES pool; default there unless the
    # user explicitly asked for other categories.
    if args.controlnet and not _categories_explicit(sys.argv):
        args.categories = ["poses"]

    out_dir = out_dir_for(args.facedetailer, args.controlnet, args.controlnet_strength)
    if args.contact_sheet_only:
        pairs = [(c, t) for c in args.categories for t in CATEGORIES[c][0]]
        pairs += library_only_tags(args.categories, args.controlnet)
        results = []
        for c, t in pairs:
            if args.only and not any(s.lower() in t.lower() for s in args.only):
                continue
            fp = os.path.join(out_dir, c, f"{slugify(t)}.png")
            results.append({"category": c, "tag": t,
                            "path": fp if os.path.exists(fp) else None,
                            "size": image_size(fp), "elapsed": None,
                            "skeleton": pose_skeletons.resolve(t) if args.controlnet else None})
    else:
        results = generate(args, out_dir)

    skeleton_lookup = ({r["tag"]: r.get("skeleton") for r in results} if args.controlnet else None)
    tags_by_category = {}
    for r in results:
        tags_by_category.setdefault(r["category"], []).append(r["tag"])
    sheets = build_contact_sheets(args.categories, out_dir, skeleton_lookup, tags_by_category)
    write_index(results, args, sheets, out_dir)
    ok = sum(1 for r in results if r["path"] and os.path.exists(r["path"]))
    print(f"\n{ok}/{len(results)} tags rendered -> {out_dir}")


def _categories_explicit(argv):
    return any(a == "--categories" for a in argv)


if __name__ == "__main__":
    main()
