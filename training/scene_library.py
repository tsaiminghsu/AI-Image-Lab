"""Scene library for the GUI's picker tab: a curated set of backgrounds with committed
thumbnails, so a scene can be chosen by clicking a picture instead of typing a prompt.

Why a library rather than the BACKGROUNDS/LIGHTINGS pools directly: those are text pools for
the random batch generators (build_variation_prompt pairs them arbitrarily). A picker needs a
fixed, ordered, human-labelled set with a picture per entry - and needs to know which entries
are suggestive-tier so a `safe` session never shows a beach in swimwear territory. The 15
entries here are curated pairings drawn from those same pools, not their 48-way product.

Layout, deliberately unlike training/poses/ (one JSON per pose):

    training/scenes/scenes.json        every scene, in gallery display order
    training/scenes/thumbs/<slug>.jpg  256px preview, committed

One file because each entry is five short fields with no per-entry payload (a pose carries 18
keypoints, which is what justifies a file each), display order is part of the data, and
curating the set should be one reviewable diff rather than fifteen.

Thumbnails are generated once by this module's `render-thumbs` CLI and committed, so the GUI
never needs a GPU or a running ComfyUI to draw the gallery - same reasoning as the committed
skeleton PNGs. Module-level imports stay light for that reason: gui.py imports this, and PIL
(thumbnail scaling) plus generate_character (prompt assembly) are imported lazily inside the
CLI path only, which also breaks the import cycle with generate_character.

CLI:
    scene_library.py list
    scene_library.py render-thumbs [--only <slug> ...] [--force] [--checkpoint juggernaut]
"""

import argparse
import json
import os
import re
import sys

SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")
SCENES_FILE = os.path.join(SCENES_DIR, "scenes.json")
THUMBS_DIR = os.path.join(SCENES_DIR, "thumbs")

TIERS = ("safe", "suggestive")
SLUG_RE = re.compile(r"^[a-z0-9_]+$")
REQUIRED_FIELDS = ("slug", "name_zh", "prompt", "lighting", "tier", "thumb_seed")

THUMB_SIZE = 256
THUMB_QUALITY = 85
# Rendered larger and downscaled: a 256px direct render would be off distribution for the
# checkpoint and look nothing like what the scene actually generates as. 768 rather than SDXL's
# native 1024 because 1024 does not survive a batch on this 8GB card - measured 2026-09-12, the
# 7th consecutive scene wedged in VAEDecode at 7.9/8.2 GB with the GPU pinned at 100%, no OOM
# raised, just silent spilling to shared memory over the PCIe x1 link. The decode's activations
# scale with pixels, and 768 is 56% of them. The output is a 256px thumbnail either way.
THUMB_RENDER_SIZE = 768
THUMB_STEPS = 20
# Juggernaut rather than the GUI's Pony default: these are empty rooms and streets, and Pony's
# training prior is character-centric enough that its scene-only renders come out weaker.
THUMB_CHECKPOINT = "juggernaut"
# The thumbnail is a picture of the place, not of a person - the negative has to say so, or an
# "empty cafe" prompt still tends to seat someone in it.
THUMB_EXTRA_PROMPT = "empty scene, no people, wide establishing shot"
THUMB_EXTRA_NEGATIVE = "person, people, man, woman, face, portrait, crowd"


def load_all():
    """Every scene in file order. Raises ValueError naming the offending entry rather than
    letting a typo reach the GUI as an empty gallery or a KeyError mid-click."""
    try:
        with open(SCENES_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"scene library not found: {SCENES_FILE}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{SCENES_FILE} is not valid JSON: {exc}")

    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"{SCENES_FILE} has no 'scenes' list")

    seen = set()
    for i, scene in enumerate(scenes):
        where = scene.get("slug") or f"entry #{i}"
        missing = [k for k in REQUIRED_FIELDS if k not in scene]
        if missing:
            raise ValueError(f"scene {where}: missing field(s) {missing}")
        slug = scene["slug"]
        if not SLUG_RE.match(slug):
            raise ValueError(f"scene {where}: slug must match {SLUG_RE.pattern}")
        if slug in seen:
            raise ValueError(f"scene {slug}: duplicate slug")
        seen.add(slug)
        if scene["tier"] not in TIERS:
            raise ValueError(f"scene {slug}: tier must be one of {TIERS}, got {scene['tier']!r}")
        if not isinstance(scene["thumb_seed"], int):
            raise ValueError(f"scene {slug}: thumb_seed must be an int")
        for field in ("name_zh", "prompt", "lighting"):
            if not str(scene[field]).strip():
                raise ValueError(f"scene {slug}: {field} is empty")
    return scenes


def list_scenes(tier="safe"):
    """Scenes selectable at this content tier, in display order. `safe` hides suggestive-tier
    scenes entirely rather than letting the picker offer one the tier would then fight."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r} - choices: {TIERS}")
    scenes = load_all()
    if tier == "suggestive":
        return scenes
    return [s for s in scenes if s["tier"] == "safe"]


def get(slug):
    for scene in load_all():
        if scene["slug"] == slug:
            return scene
    raise ValueError(f"unknown scene {slug!r} - choices: {[s['slug'] for s in load_all()]}")


def scene_text(slug):
    """The prompt fragment for a scene: its setting plus the lighting curated with it."""
    scene = get(slug)
    return f"{scene['prompt']}, {scene['lighting']}"


def thumb_path(slug):
    return os.path.join(THUMBS_DIR, f"{slug}.jpg")


def _render_one(scene, checkpoint_key, client, gc):
    from PIL import Image

    # Free first, every time. Measured 2026-09-12 on the 8GB card: rendering the batch back to
    # back in one ComfyUI session fragments VRAM until a VAEDecode spills into shared memory and
    # effectively stops - at 1024 the 7th scene wedged, at 768 the 10th did, and the second time
    # the server stopped answering HTTP for long enough that the client's retry budget gave up.
    # Reloading the checkpoint each time costs ~20s over the PCIe x1 link, which is a fair price
    # for a one-off batch that otherwise has to be babysat and restarted.
    client.free_vram()

    slug = scene["slug"]
    prompt, negative = gc._build_prompt_and_negative(
        f"{scene_text(slug)}, {THUMB_EXTRA_PROMPT}",
        THUMB_EXTRA_NEGATIVE,
        "safe",
        None,
        None,
        None,
        checkpoint_key,
    )
    raw = client.submit_txt2img_generation(
        prompt,
        negative,
        scene["thumb_seed"],
        f"scene_thumb_{slug}",
        width=THUMB_RENDER_SIZE,
        height=THUMB_RENDER_SIZE,
        steps=THUMB_STEPS,
        checkpoint=client.CHECKPOINTS[checkpoint_key],
        lora_strength=0.0,
    )
    with Image.open(raw) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        im.save(thumb_path(slug), "JPEG", quality=THUMB_QUALITY)
    os.remove(raw)
    return thumb_path(slug)


def _cmd_render_thumbs(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import comfyui_client as client
    import generate_character as gc

    if args.checkpoint not in client.CHECKPOINTS:
        raise SystemExit(f"unknown checkpoint {args.checkpoint!r} - choices: {sorted(client.CHECKPOINTS)}")
    if not client.is_server_running():
        raise SystemExit(
            "ComfyUI is not running - start it first (training/gui.py starts it on demand, or run "
            "ComfyUI/main.py --listen 127.0.0.1 --port 8188)"
        )

    os.makedirs(THUMBS_DIR, exist_ok=True)
    scenes = load_all()
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s["slug"] for s in scenes}
        if unknown:
            raise SystemExit(f"unknown scene(s): {sorted(unknown)}")
        scenes = [s for s in scenes if s["slug"] in wanted]

    pending = [s for s in scenes if args.force or not os.path.exists(thumb_path(s["slug"]))]
    if not pending:
        print("all thumbnails already rendered, nothing to do (use --force to rebuild)", flush=True)
        return
    print(f"rendering {len(pending)} thumbnail(s) with {args.checkpoint}", flush=True)

    client.log_gpu_memory("before_scene_thumbs")
    for scene in pending:
        path = _render_one(scene, args.checkpoint, client, gc)
        print(f"{scene['slug']}: {path} ({os.path.getsize(path) / 1024:.0f} KB)", flush=True)
    client.log_gpu_memory("after_scene_thumbs")


def _cmd_list(args):
    for scene in load_all():
        mark = " " if os.path.exists(thumb_path(scene["slug"])) else "!"
        print(f"{mark} {scene['slug']:20s} {scene['tier']:11s} {scene['name_zh']}  |  {scene_text(scene['slug'])}")
    print("\n(! = thumbnail missing, run render-thumbs)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show every scene and whether its thumbnail exists")
    r = sub.add_parser("render-thumbs", help="generate the committed gallery thumbnails (needs ComfyUI running)")
    r.add_argument("--only", nargs="*", default=None, help="render just these slugs")
    r.add_argument("--force", action="store_true", help="re-render thumbnails that already exist")
    r.add_argument("--checkpoint", default=THUMB_CHECKPOINT, help=f"checkpoint key (default {THUMB_CHECKPOINT})")
    args = p.parse_args()

    if args.cmd == "list":
        _cmd_list(args)
    elif args.cmd == "render-thumbs":
        _cmd_render_thumbs(args)


if __name__ == "__main__":
    main()
