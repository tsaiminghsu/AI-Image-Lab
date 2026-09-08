"""OpenPose skeleton library for the POSES tag pool in generate_character.py.

Motivation: pose_pack.py's empirical run showed cyberrealistic_pony (and Pony
checkpoints generally) ignore "lying on ..." pose tags - text alone can't push
the body off its standing/sitting prior. ControlNet OpenPose is already wired
into workflow_template_hq.json (nodes 20-23); the missing piece was a reusable
set of skeleton reference images to feed it. This module builds and serves that
set (training/poses/), so a pose tag maps to a skeleton PNG that constrains the
first-pass composition.

Each library entry is a pair:
  training/poses/<slug>.json   canonical source: 18 normalized body keypoints in
                               OpenPose JSON layout + metadata (camera, prompt_hint)
  training/poses/<slug>.png    832x1216 skeleton rendered from the JSON, committed
                               so generation never needs torch/controlnet_aux

<slug> = slugify(<POSES tag>). The 18-point order (0-based) is the COCO/OpenPose
body convention the ControlNet was trained on:
  0 nose 1 neck 2 Rsho 3 Relb 4 Rwri 5 Lsho 6 Lelb 7 Lwri
  8 Rhip 9 Rknee 10 Rankle 11 Lhip 12 Lknee 13 Lankle
  14 Reye 15 Leye 16 Rear 17 Lear   (R = subject's right)
In pose_keypoints_2d the triplets are [x, y, c] with x/y normalized 0..1 and
c=1 present / c=0 hidden (a hidden joint is drawn as absent - e.g. the far ear
in a side view).

The draw/extract helpers lazily import comfyui_controlnet_aux, which pulls in
torch, so run the CLI with ComfyUI's interpreter:
    ComfyUI\.venv\Scripts\python.exe training\pose_skeletons.py ...
The lookup helpers (list_names/resolve/load_meta/slugify) import nothing heavy
and are safe to call from generate_character.py / gui.py / pose_pack.py.

CLI:
    pose_skeletons.py extract --from-pack <dir> [--only <slug> ...]  # bootstrap from renders
    pose_skeletons.py render --all | <slug>                          # JSON -> PNG
    pose_skeletons.py sheet                                          # contact sheet of all skeletons
"""

import argparse
import json
import os
import re
import sys

POSES_DIR = os.path.join(os.path.dirname(__file__), "poses")
CANVAS_W, CANVAS_H = 832, 1216  # matches FULL_BODY_RESOLUTION; ControlNet resizes to the latent anyway
NUM_BODY_KEYPOINTS = 18

_CONTROLNET_AUX_SRC = os.path.join(
    os.path.dirname(__file__), "..", "ComfyUI", "custom_nodes",
    "comfyui_controlnet_aux", "src",
)


def slugify(tag):
    """POSES tag -> filesystem slug. Kept identical to pose_pack.slugify (which
    imports this) so the two never drift."""
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")[:60]


def _json_path(slug):
    return os.path.join(POSES_DIR, f"{slug}.json")


def _png_path(slug):
    return os.path.join(POSES_DIR, f"{slug}.png")


def list_names():
    """Slugs that have a rendered .png in the library."""
    if not os.path.isdir(POSES_DIR):
        return []
    return sorted(
        fn[:-4] for fn in os.listdir(POSES_DIR)
        if fn.endswith(".png") and os.path.isfile(_json_path(fn[:-4]))
    )


def resolve(name_or_tag):
    """Accept a slug or a raw POSES tag; return the skeleton PNG path or None."""
    if not name_or_tag:
        return None
    slug = name_or_tag if os.path.exists(_png_path(name_or_tag)) else slugify(name_or_tag)
    png = _png_path(slug)
    return png if os.path.exists(png) else None


def load_meta(name_or_tag):
    """Return the JSON metadata dict for a library entry (or {} if absent)."""
    slug = name_or_tag if os.path.exists(_json_path(name_or_tag)) else slugify(name_or_tag)
    path = _json_path(slug)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _keypoints_from_flat(flat):
    """[x,y,c]*18 -> list of Keypoint|None (None where c<=0), for draw_bodypose."""
    from custom_controlnet_aux.open_pose.body import Keypoint
    pts = []
    for i in range(NUM_BODY_KEYPOINTS):
        x, y, c = flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]
        pts.append(Keypoint(x=float(x), y=float(y)) if c > 0 else None)
    return pts


def render(json_path, png_path, width=CANVAS_W, height=CANVAS_H):
    """Render a skeleton PNG from a keypoint JSON using controlnet_aux's own
    draw_bodypose (so colors/limb order match what the ControlNet was trained
    on). Saved via PIL to keep RGB channel order."""
    if _CONTROLNET_AUX_SRC not in sys.path:
        sys.path.insert(0, _CONTROLNET_AUX_SRC)
    import numpy as np
    from PIL import Image
    from custom_controlnet_aux.open_pose.util import draw_bodypose

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    people = data.get("people", [])
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for person in people:
        flat = person.get("pose_keypoints_2d")
        if flat:
            canvas = draw_bodypose(canvas, _keypoints_from_flat(flat))
    Image.fromarray(canvas).save(png_path)
    return png_path


def extract(image_path, json_path, png_path, detector=None):
    """Bootstrap a skeleton from an existing rendered photo: run OpenPose body
    detection, keep the highest-scoring person, write JSON (+ metadata) and a
    re-rendered PNG at the library canvas size. Returns True if a body was found.

    Useless for poses the checkpoint renders wrong (the 5 lying tags) - those
    get hand-authored JSON instead - but near-free for the ones it renders
    correctly, and gives limb-length templates to re-pose the lying ones from."""
    if _CONTROLNET_AUX_SRC not in sys.path:
        sys.path.insert(0, _CONTROLNET_AUX_SRC)
    import numpy as np
    from PIL import Image
    from custom_controlnet_aux.open_pose import OpenposeDetector

    if detector is None:
        detector = OpenposeDetector.from_pretrained()
    img = np.array(Image.open(image_path).convert("RGB"))
    _, pose_json = detector(
        img, detect_resolution=512, include_body=True,
        include_hand=False, include_face=False,
        output_type="np", image_and_json=True,
    )
    people = pose_json.get("people", [])
    people = [p for p in people if p.get("pose_keypoints_2d")]
    if not people:
        return False

    # highest-scoring person = the one with the most present joints
    def present(p):
        flat = p["pose_keypoints_2d"]
        return sum(1 for i in range(NUM_BODY_KEYPOINTS) if flat[3 * i + 2] > 0)

    best = max(people, key=present)
    data = {
        "tag": os.path.basename(json_path)[:-5],
        "camera": "extracted from render",
        "prompt_hint": "",
        "source": f"extracted from {os.path.basename(image_path)}",
        "canvas_width": CANVAS_W,
        "canvas_height": CANVAS_H,
        "people": [{"pose_keypoints_2d": best["pose_keypoints_2d"]}],
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    render(json_path, png_path)
    return True


def build_sheet(out_path=None):
    """Contact sheet of every library skeleton, labelled by slug."""
    from PIL import Image, ImageDraw, ImageFont
    names = list_names()
    if not names:
        print("no skeletons in library yet", flush=True)
        return None
    THUMB_W, COLS, LABEL_H, PAD = 240, 5, 22, 6
    thumb_h = round(THUMB_W * CANVAS_H / CANVAS_W)
    cell_h = thumb_h + LABEL_H
    rows = (len(names) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * THUMB_W + (COLS + 1) * PAD,
                              rows * cell_h + (rows + 1) * PAD), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for idx, slug in enumerate(names):
        r, c = divmod(idx, COLS)
        x = PAD + c * (THUMB_W + PAD)
        y = PAD + r * (cell_h + PAD)
        with Image.open(_png_path(slug)) as im:
            sheet.paste(im.convert("RGB").resize((THUMB_W, thumb_h), Image.LANCZOS), (x, y))
        label = slug if len(slug) <= 34 else slug[:31] + "..."
        draw.text((x + 2, y + thumb_h + 5), label, fill="black", font=font)
    out_path = out_path or os.path.join(POSES_DIR, "contact_sheet.png")
    sheet.save(out_path)
    print(f"sheet: {out_path} ({len(names)} skeletons)", flush=True)
    return out_path


def _cmd_extract(args):
    if _CONTROLNET_AUX_SRC not in sys.path:
        sys.path.insert(0, _CONTROLNET_AUX_SRC)
    from custom_controlnet_aux.open_pose import OpenposeDetector
    os.makedirs(POSES_DIR, exist_ok=True)
    detector = OpenposeDetector.from_pretrained()
    src_dir = args.from_pack
    pngs = sorted(fn for fn in os.listdir(src_dir) if fn.endswith(".png"))
    if args.only:
        wanted = set(args.only)
        pngs = [fn for fn in pngs if fn[:-4] in wanted]
    ok = 0
    for fn in pngs:
        slug = fn[:-4]
        found = extract(os.path.join(src_dir, fn), _json_path(slug), _png_path(slug), detector)
        print(f"{'OK ' if found else 'no body'} {slug}", flush=True)
        ok += found
    print(f"\nextracted {ok}/{len(pngs)} -> {POSES_DIR}", flush=True)


def _cmd_render(args):
    if args.target == "--all" or args.target == "all":
        slugs = sorted(fn[:-5] for fn in os.listdir(POSES_DIR) if fn.endswith(".json"))
    else:
        slugs = [args.target if os.path.exists(_json_path(args.target)) else slugify(args.target)]
    for slug in slugs:
        jp = _json_path(slug)
        if not os.path.exists(jp):
            print(f"missing JSON: {slug}", flush=True)
            continue
        render(jp, _png_path(slug))
        print(f"rendered {slug}", flush=True)


def main():
    p = argparse.ArgumentParser(description="OpenPose skeleton library builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="bootstrap skeletons from a folder of rendered pose images")
    pe.add_argument("--from-pack", required=True, help="dir of <slug>.png renders (e.g. pose_pack_facedetailer/poses)")
    pe.add_argument("--only", nargs="+", help="restrict to these slugs")
    pe.set_defaults(func=_cmd_extract)

    pr = sub.add_parser("render", help="render <slug>.json -> <slug>.png (or --all)")
    pr.add_argument("target", help="a slug/tag, or --all")
    pr.set_defaults(func=_cmd_render)

    ps = sub.add_parser("sheet", help="build contact_sheet.png of all skeletons")
    ps.set_defaults(func=lambda a: build_sheet())

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
