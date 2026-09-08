"""Generate reference/dataset images for original fictional characters using
SDXL via a persistently-running ComfyUI server (see comfyui_client.py).
Model lifecycle, VRAM management, and caching between prompts is entirely
ComfyUI's responsibility - this script never imports torch.

Supports multiple independent characters (own trigger word, own identity,
own anchor + dataset folder) - see CHARACTERS below.

Start ComfyUI once before running this script:
    D:\\AI-Image-Lab\\ComfyUI\\.venv\\Scripts\\python.exe D:\\AI-Image-Lab\\ComfyUI\\main.py --listen 127.0.0.1 --port 8188

Stage 1 (mode="anchor"): plain txt2img, generate a handful of candidate
identity anchor images from different seeds. Pick the best one manually.

Stage 2 (mode="variations"): IP-Adapter conditioned on the chosen anchor's
face, varying pose/angle/outfit/lighting/background prompts while keeping
the character recognizable across the dataset.
"""

import argparse
import os
import random

import comfyui_client as client

# "photorealistic, high detail" reads to SDXL as "polished digital art" as
# much as "photo" - it's a big part of why outputs look like a 3D render.
# Short, specific camera/skin-texture terms push it toward an actual photo
# instead; kept brief so it survives the CLIP 77-token budget alongside the
# per-image angle/pose/outfit/lighting/background terms. "candid photograph"
# and "film grain" push further against the smooth, symmetric, overlit look
# that reads as "render" even after the first pass of these terms - that look
# kept showing up in practice, so the negative list also got more explicit
# about naming it (smooth/perfect skin, symmetrical face, digital art/render)
# rather than relying on "3d render, cgi" alone to cover it.
REALISTIC_STYLE = "shot on DSLR, natural skin texture, visible pores, film photo, candid photograph, slight film grain"
REALISTIC_NEGATIVE = (
    "3d render, cgi, illustration, airbrushed, plastic skin, doll-like, "
    "smooth skin, perfect skin, symmetrical face, digital art, render, unreal engine"
)

# Pony-family checkpoints (client.PONY_CHECKPOINTS) were trained on a
# "score_9, score_8_up, score_7_up"-style quality-tag prefix convention, not
# plain natural language - this is a technical quality switch baked into
# Pony's training data, not content description, so gen_custom() auto-
# prepends it whenever a Pony checkpoint is selected (see below). Callers
# (GUI/API/CLI) never need to know or type this themselves - prompt input
# stays plain natural language no matter which checkpoint is picked.
PONY_QUALITY_TAGS = "score_9, score_8_up, score_7_up"

# Negative-side complement to PONY_QUALITY_TAGS - excludes the quality tiers
# just below the positive prefix's floor (score_7_up means "7 or better", so
# these push away from the 4-6 range). Same auto-prepend treatment, same
# reasoning: a Pony-specific technical convention, not something callers
# should need to know about or type into extra_negative/style_negative
# themselves.
PONY_QUALITY_NEGATIVE_TAGS = "score_6, score_5, score_4"

# Always included no matter the content tier below - not a dial that gets
# loosened for the suggestive/nsfw modes.
AGE_SAFETY_NEGATIVE = "child, children, kid, minor, teen, teenager, underage, young girl"

# Safety-only, no style/quality terms baked in - kept separate from
# REALISTIC_NEGATIVE specifically so gen_custom's style_negative param (see
# below, exposed as an editable GUI field) can be freely overridden without
# ever touching this. Never expose this constant itself as user-editable.
SAFE_SAFETY_NEGATIVE = (
    f"nsfw, nude, naked, explicit, sexual content, {AGE_SAFETY_NEGATIVE}, "
    "lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text"
)

NEGATIVE_PROMPT = f"{SAFE_SAFETY_NEGATIVE}, {REALISTIC_NEGATIVE}"

# Blocks fully explicit content specifically (genitals, sex acts, porn) while
# allowing suggestive/artistic content (swimwear, lingerie, implied nudity,
# bare shoulders/back) through. AGE_SAFETY_NEGATIVE is never dropped.
SUGGESTIVE_NEGATIVE = (
    f"exposed genitalia, exposed vulva, exposed penis, exposed nipples, "
    f"sexual intercourse, penetration, pornographic, explicit sexual act, "
    f"{AGE_SAFETY_NEGATIVE}, "
    "lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text"
)

# Hard floor - 16/17 are minors in most jurisdictions and are never
# generated here, regardless of per-character age choices below.
MINIMUM_AGE = 18

# Each entry is an independent fictional adult character: own trigger word,
# own identity, own anchor + dataset folder. Ages vary 18-25, each with a
# distinct face/build and a distinct modern style so they stay visually
# separable from one another and from the original "mylora" character.
CHARACTERS = {
    "mylora": {
        "age": 28,
        "gender": "woman",
        "appearance": "long straight black hair, dark brown eyes, oval face, mature adult features, natural healthy build",
        "style": "girl-next-door style, casual outfit",
    },
    "mei": {
        "age": 19,
        "gender": "woman",
        "appearance": "long straight jet-black hair with blunt bangs, bright round eyes, soft round face, petite build",
        "style": "casual campus style, oversized hoodie and pleated skirt, sneakers",
    },
    "xinyi": {
        "age": 21,
        "gender": "woman",
        "appearance": "shoulder-length wavy chestnut brown hair, almond eyes, heart-shaped face, slim build",
        "style": "modern minimalist chic, tailored blazer and trousers",
    },
    "ruoxi": {
        "age": 23,
        "gender": "woman",
        "appearance": "long layered hair with soft curls, dark brown eyes, oval face, athletic build",
        "style": "trendy streetwear, cropped jacket and jeans",
    },
    "yuqing": {
        "age": 25,
        "gender": "woman",
        "appearance": "sleek short bob haircut, sharp eyes, angular face, tall slender build",
        "style": "elegant office chic, fitted blouse and skirt",
    },
    "wanling": {
        "age": 18,
        "gender": "woman",
        "appearance": "high ponytail, big expressive eyes, youthful round face, petite build",
        "style": "y2k fashion, colorful crop top and cargo pants",
    },
    "minjun": {
        "age": 18,
        "gender": "man",
        "appearance": "tousled black hair, warm friendly eyes, soft face, slim build",
        "style": "Taiwanese street style, oversized hoodie and cargo pants",
    },
    "junho": {
        "age": 20,
        "gender": "man",
        "appearance": "undercut hairstyle, calm reserved eyes, angular face, lean build",
        "style": "Japanese minimalist style, monochrome knit sweater and slim trousers",
    },
    "taeoh": {
        "age": 22,
        "gender": "man",
        "appearance": "soft permed hair, cool detached eyes, refined face, toned build",
        "style": "Korean urban style, oversized shirt and black trousers",
    },
    "hyunjun": {
        "age": 24,
        "gender": "man",
        "appearance": "tweed cut hairstyle, deep-set eyes, sharp jawline, athletic build",
        "style": "Japanese minimalist elegance, cream sweater and dark grey trousers",
    },
    "jungi": {
        "age": 25,
        "gender": "man",
        "appearance": "slicked side part hair, confident sharp eyes, angular jaw, fit build",
        "style": "Taiwanese flashy streetwear, bold surf-brand jacket and fitted pants",
    },
}

# Per-character LoRA registry. None until a LoRA is trained on RunPod and its
# .safetensors is dropped into ComfyUI/models/loras/ (see runpod_bundle.py
# install). When set, it's a dict:
#   {"file": "xinyi_pony_v1.safetensors", "trigger": "xinyi", "base_family": "pony",
#    "strength_model": 0.8, "strength_clip": 0.8, "faceid_weight": 0.7}
# The trigger equals the CHARACTERS key (already the first caption token). Used
# TOGETHER with IP-Adapter FaceID: the LoRA carries identity, FaceID (dropped to
# faceid_weight) only corrects drift. base_family lets gen_custom warn when the
# selected checkpoint family doesn't match what the LoRA was trained on.
for _name, _profile in CHARACTERS.items():
    if _profile["age"] < MINIMUM_AGE:
        raise ValueError(f"character '{_name}' age {_profile['age']} is below MINIMUM_AGE={MINIMUM_AGE}")
    _profile.setdefault("lora", None)

# Where character LoRAs live locally (hardlinked from models/lora/, see README).
# gen_custom checks a LoRA file exists here before wiring it into the workflow.
COMFYUI_LORAS_DIR = os.path.join(os.path.dirname(__file__), "..", "ComfyUI", "models", "loras")

# Default checkpoint for the HQ custom-generation path (GUI/API/CLI) when the
# caller doesn't pick one. Pony-family photoreal merge - matches the GUI's
# shipped default. client.CHECKPOINT stays juggernaut for the anchor/variations
# dataset stages (gen_anchors / gen_variations), which are a separate concern.
DEFAULT_CUSTOM_CHECKPOINT = "cyberrealistic_pony"


def get_character(trigger):
    try:
        return CHARACTERS[trigger]
    except KeyError:
        raise SystemExit(
            f"unknown character '{trigger}'. Known characters: {', '.join(sorted(CHARACTERS))}"
        )


def character_base_prompt(trigger, profile):
    return (
        f"{trigger}, {profile['age']} year old adult {profile['gender']}, east asian, "
        f"{profile['appearance']}, {profile['style']}"
    )


def gen_anchors(trigger, out_dir, seeds):
    profile = get_character(trigger)
    os.makedirs(out_dir, exist_ok=True)
    pending = []
    for seed in seeds:
        path = os.path.join(out_dir, f"anchor_seed{seed}.png")
        if os.path.exists(path):
            print(f"skip (already exists): {path}", flush=True)
        else:
            pending.append(seed)
    if not pending:
        print("all anchors already generated, nothing to do", flush=True)
        return

    prompt = f"{character_base_prompt(trigger, profile)}, soft studio lighting, {REALISTIC_STYLE}"

    client.log_gpu_memory("before_anchor_batch")
    for seed in pending:
        raw_path = client.submit_txt2img_generation(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            seed=seed,
            filename_prefix=f"{trigger}_anchor_seed{seed}",
        )
        path = os.path.join(out_dir, f"anchor_seed{seed}.png")
        os.replace(raw_path, path)
        print(f"saved {path}", flush=True)
    client.log_gpu_memory("after_anchor_batch")


CLOSE_ANGLES = [
    "front view portrait", "3/4 view portrait", "side profile portrait",
    "looking over shoulder", "slightly from above", "slightly from below",
    "close-up headshot", "medium shot upper body",
    "overhead view looking down at camera", "high angle view from directly above",
]
# Separate pool + explicit odds below, instead of one flat ANGLES list - at
# 1/9 odds a square 1024x1024 canvas kept cropping "full body shot" down to
# upper-body anyway (no vertical room for full height). FULL_BODY_RESOLUTION
# gives it the room; the ~45% pick rate actually gets a visible fraction of
# the dataset to render as full body instead of a few stragglers.
FULL_BODY_ANGLES = [
    "full body shot", "full body shot from a distance", "full length shot standing",
    "full body shot walking", "head-to-toe full body shot",
]
FULL_BODY_PICK_RATE = 0.35
FULL_BODY_RESOLUTION = (832, 1216)  # SDXL-native portrait aspect - room for head to feet

# No angle pool had an actual back-facing option before (closest was "looking
# over shoulder", which is still a front-of-face shot) - that's the entire
# reason nothing ever came out from behind, not just low odds. Separate pool
# + its own lowered weight below: the reference face isn't visible from
# behind at all, so conditioning at the normal ip_adapter_weight fights the
# pose the same way the old "VIT-G" preset used to fight prompted poses -
# high identity-matching pressure with no face to match just pulls the
# composition back toward frontal.
BACK_VIEW_ANGLES = [
    "back view, from behind", "rear view, walking away, seen from behind",
    "3/4 back view, looking back over shoulder from behind",
    "back of head and shoulders, from behind",
]
BACK_VIEW_PICK_RATE = 0.30  # bumped from 0.20 - even at a 50% weight cut, generations still
# came back mostly frontal (observed on FaceID Plus V2's default anchors). FaceID's InsightFace
# lock turned out to fight a faceless pose harder than PLUS FACE ever did (see weight cut below).
BACK_VIEW_IP_ADAPTER_WEIGHT = 0.3  # cut from 0.5 to 0.3 for the same reason - 0.5 still wasn't
# enough headroom for "back view" in the prompt to actually win out over FaceID's identity pull.
POSES = [
    "standing straight", "sitting on a chair", "leaning against a wall",
    "walking", "hands in pockets", "arms crossed", "hand touching hair",
    "looking directly at camera", "looking away from camera", "slight smile",
    "sitting on the floor, knees bent",
    "lying on back, relaxed", "lying on back, one knee bent, arms above head",
    "lying on side, relaxed, head resting on hand",
    "lying on stomach, propped up on elbows, reading a book",
    "lying on stomach, chin resting on hands, ankles crossed in the air",
    "seated yoga stretch pose", "standing yoga tree pose",
    "stretching arms overhead after exercise", "jogging pose, mid-stride",
]
OUTFITS = [
    "white t-shirt and jeans", "beige knit sweater", "denim jacket over t-shirt",
    "floral summer dress", "beige trench coat", "cream cardigan",
    "white blouse", "casual hoodie", "light sweater and skirt",
]
MALE_OUTFITS = [
    "white t-shirt and jeans", "black bomber jacket over t-shirt", "denim jacket over t-shirt",
    "casual button-up shirt", "beige trench coat", "knit sweater",
    "graphic hoodie", "casual blazer over t-shirt", "cargo pants and hoodie",
]
LIGHTINGS = [
    "soft natural window light", "golden hour sunlight", "studio softbox lighting",
    "overcast daylight", "warm indoor lighting", "soft rim light",
]
BACKGROUNDS = [
    "plain white studio background", "cozy cafe interior", "quiet city street",
    "park with trees", "bright bedroom interior", "outdoor garden",
    "minimalist indoor setting", "bookstore interior",
]

# Swimwear/athletic-wear level only - same ceiling as SUGGESTIVE_NEGATIVE
# enforces (no exposed genitalia/nipples, no sexual acts). Separate pools
# from OUTFITS/MALE_OUTFITS since regular clothing doesn't make sense here.
SUGGESTIVE_OUTFITS = [
    "bikini", "one-piece swimsuit", "sports bra and yoga shorts",
    "off-shoulder top and shorts", "tank top and short shorts", "camisole and shorts",
]
MALE_SUGGESTIVE_OUTFITS = [
    "swim trunks", "athletic shorts, shirtless", "board shorts",
    "tank top and shorts", "shirtless, joggers", "swim shorts",
]
SUGGESTIVE_BACKGROUNDS = [
    "beach at sunset", "poolside", "beach during the day",
    "outdoor shower area", "tropical resort", "lakeside dock",
]


def pick_angle(rng):
    """Returns (angle, width, height, ip_adapter_weight_override). Full-body
    prompts get a taller canvas so there's actually room to render feet-to-
    head instead of getting cropped back down to upper-body. Back-view
    prompts additionally get a lowered IP-Adapter weight override (see
    BACK_VIEW_IP_ADAPTER_WEIGHT) - None means "use the caller's default"."""
    roll = rng.random()
    if roll < BACK_VIEW_PICK_RATE:
        return rng.choice(BACK_VIEW_ANGLES), *FULL_BODY_RESOLUTION, BACK_VIEW_IP_ADAPTER_WEIGHT
    if roll < BACK_VIEW_PICK_RATE + FULL_BODY_PICK_RATE:
        return rng.choice(FULL_BODY_ANGLES), *FULL_BODY_RESOLUTION, None
    return rng.choice(CLOSE_ANGLES), client.WIDTH, client.HEIGHT, None


def build_variation_prompt(trigger, profile, rng):
    angle, width, height, weight_override = pick_angle(rng)
    pose = rng.choice(POSES)
    outfit_pool = OUTFITS if profile["gender"] == "woman" else MALE_OUTFITS
    outfit = rng.choice(outfit_pool)
    lighting = rng.choice(LIGHTINGS)
    background = rng.choice(BACKGROUNDS)
    prompt = (
        f"{trigger}, {profile['age']} year old adult {profile['gender']}, east asian, "
        f"{profile['appearance']}, "
        f"{angle}, {pose}, wearing {outfit}, {lighting}, {background}, "
        f"{REALISTIC_STYLE}"
    )
    caption = (
        f"{trigger}, adult {profile['gender']}, portrait, {angle}, {pose}, {outfit}, "
        f"{lighting}, {background}"
    )
    return prompt, caption, width, height, weight_override


def gen_variations(trigger, anchor_path, out_dir, count, ip_adapter_weight, base_seed):
    profile = get_character(trigger)
    os.makedirs(out_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(out_dir)
        if f.startswith("var_") and f.endswith(".png")
    ]
    start_idx = len(existing)
    if start_idx >= count:
        print(f"already have {start_idx} variations >= requested {count}, nothing to do", flush=True)
        return

    ref_filename = client.upload_reference_image(anchor_path)
    rng = random.Random(base_seed)

    client.log_gpu_memory("before_variation_batch")
    for i in range(start_idx, count):
        prompt, caption, width, height, weight_override = build_variation_prompt(trigger, profile, rng)
        seed = base_seed + i
        stem = f"var_{i:04d}_seed{seed}"
        raw_path = client.submit_generation(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            seed=seed,
            ip_adapter_image_filename=ref_filename,
            filename_prefix=stem,
            ip_adapter_weight=weight_override if weight_override is not None else ip_adapter_weight,
            width=width,
            height=height,
        )
        img_path = os.path.join(out_dir, f"{stem}.png")
        txt_path = os.path.join(out_dir, f"{stem}.txt")
        os.replace(raw_path, img_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"saved {img_path}", flush=True)
        if i == start_idx or (i + 1) % 10 == 0:
            client.log_gpu_memory(f"after_image_{i}")
    client.log_gpu_memory("after_variation_batch")


def gen_test_suggestive(trigger, anchor_path, out_dir, seed, ip_adapter_weight):
    profile = get_character(trigger)
    os.makedirs(out_dir, exist_ok=True)
    ref_filename = client.upload_reference_image(anchor_path)

    swimwear = "bikini" if profile["gender"] == "woman" else "swim trunks"
    prompt = (
        f"{trigger}, {profile['age']} year old adult {profile['gender']}, east asian, "
        f"{profile['appearance']}, "
        f"wearing {swimwear}, standing on a beach, sunset lighting, alluring pose, "
        f"looking at camera, {REALISTIC_STYLE}"
    )
    caption = f"{trigger}, adult {profile['gender']}, portrait, {swimwear}, beach, sunset lighting, alluring pose"
    stem = f"test_suggestive_seed{seed}"

    raw_path = client.submit_generation(
        prompt=prompt,
        negative_prompt=f"{SUGGESTIVE_NEGATIVE}, {REALISTIC_NEGATIVE}",
        seed=seed,
        ip_adapter_image_filename=ref_filename,
        filename_prefix=stem,
        ip_adapter_weight=ip_adapter_weight,
    )
    img_path = os.path.join(out_dir, f"{stem}.png")
    txt_path = os.path.join(out_dir, f"{stem}.txt")
    os.replace(raw_path, img_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(caption)
    print(f"saved {img_path}", flush=True)


def build_suggestive_variation_prompt(trigger, profile, rng):
    """Same shape as build_variation_prompt, but swimwear/athletic-wear
    outfits and beach/pool backgrounds, and SUGGESTIVE_NEGATIVE at the call
    site instead of NEGATIVE_PROMPT. Still capped at swimwear-level - never
    exposes what SUGGESTIVE_NEGATIVE blocks, and AGE_SAFETY_NEGATIVE is
    unchanged."""
    angle, width, height, weight_override = pick_angle(rng)
    pose = rng.choice(POSES)
    outfit_pool = SUGGESTIVE_OUTFITS if profile["gender"] == "woman" else MALE_SUGGESTIVE_OUTFITS
    outfit = rng.choice(outfit_pool)
    lighting = rng.choice(LIGHTINGS)
    background = rng.choice(SUGGESTIVE_BACKGROUNDS)
    prompt = (
        f"{trigger}, {profile['age']} year old adult {profile['gender']}, east asian, "
        f"{profile['appearance']}, "
        f"{angle}, {pose}, wearing {outfit}, {lighting}, {background}, "
        f"{REALISTIC_STYLE}"
    )
    caption = (
        f"{trigger}, adult {profile['gender']}, portrait, {angle}, {pose}, {outfit}, "
        f"{lighting}, {background}"
    )
    return prompt, caption, width, height, weight_override


def gen_suggestive_variations(trigger, anchor_path, out_dir, count, ip_adapter_weight, base_seed):
    """Batch suggestive-tier variations, same resumable-by-file-count shape
    as gen_variations. Filenames are prefixed "sugg_" so they can share a
    dataset folder with gen_variations' "var_" files without colliding."""
    profile = get_character(trigger)
    os.makedirs(out_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(out_dir)
        if f.startswith("sugg_") and f.endswith(".png")
    ]
    start_idx = len(existing)
    if start_idx >= count:
        print(f"already have {start_idx} suggestive variations >= requested {count}, nothing to do", flush=True)
        return

    ref_filename = client.upload_reference_image(anchor_path)
    rng = random.Random(base_seed)
    negative_prompt = f"{SUGGESTIVE_NEGATIVE}, {REALISTIC_NEGATIVE}"

    client.log_gpu_memory("before_suggestive_variation_batch")
    for i in range(start_idx, count):
        prompt, caption, width, height, weight_override = build_suggestive_variation_prompt(trigger, profile, rng)
        seed = base_seed + i
        stem = f"sugg_{i:04d}_seed{seed}"
        raw_path = client.submit_generation(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            ip_adapter_image_filename=ref_filename,
            filename_prefix=stem,
            ip_adapter_weight=weight_override if weight_override is not None else ip_adapter_weight,
            width=width,
            height=height,
        )
        img_path = os.path.join(out_dir, f"{stem}.png")
        txt_path = os.path.join(out_dir, f"{stem}.txt")
        os.replace(raw_path, img_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"saved {img_path}", flush=True)
        if i == start_idx or (i + 1) % 10 == 0:
            client.log_gpu_memory(f"after_suggestive_image_{i}")
    client.log_gpu_memory("after_suggestive_variation_batch")


def _build_prompt_and_negative(prompt, extra_negative, tier, trigger, style_positive, style_negative, checkpoint):
    """Shared by gen_custom() and gen_gif() so both compose the final
    positive/negative prompt strings identically - character identity
    prefix, style terms, safety negatives, and the Pony quality-tag
    auto-prepend all live in exactly one place instead of two copies that
    could drift out of sync."""
    style_positive = REALISTIC_STYLE if style_positive is None else style_positive
    style_negative = REALISTIC_NEGATIVE if style_negative is None else style_negative

    safety_negative = SAFE_SAFETY_NEGATIVE if tier == "safe" else SUGGESTIVE_NEGATIVE
    base_negative = f"{safety_negative}, {style_negative}" if style_negative else safety_negative
    negative_prompt = f"{base_negative}, {extra_negative}" if extra_negative else base_negative
    if checkpoint in client.PONY_CHECKPOINTS:
        negative_prompt = f"{PONY_QUALITY_NEGATIVE_TAGS}, {negative_prompt}"

    if trigger:
        profile = get_character(trigger)
        full_prompt = f"{character_base_prompt(trigger, profile)}, {prompt}"
    else:
        full_prompt = prompt
    if style_positive:
        full_prompt = f"{full_prompt}, {style_positive}"
    if checkpoint in client.PONY_CHECKPOINTS:
        full_prompt = f"{PONY_QUALITY_TAGS}, {full_prompt}"

    return full_prompt, negative_prompt


def _checkpoint_family(checkpoint_key):
    """Maps a CHECKPOINTS key to a family label for LoRA compatibility checks."""
    if checkpoint_key in client.PONY_CHECKPOINTS:
        return "pony"
    if checkpoint_key in client.SD15_CHECKPOINTS:
        return "sd15"
    return "sdxl"


def _run_hq(full_prompt, negative_prompt, seed, stem, trigger, anchor_path, pose_reference_path,
            width, height, ip_adapter_weight, effective_checkpoint, lora_strength,
            character_lora_strength, controlnet_strength, hires_denoise, use_facedetailer,
            face_denoise, hand_denoise):
    """Resolves the per-character LoRA (if any), lowers FaceID weight when a
    LoRA carries identity, uploads anchor/pose, and calls submit_generation_hq.
    Shared by gen_custom's HQ branch. anchor_path (IP-Adapter) and
    pose_reference_path (ControlNet) are both optional here - the HQ template
    bypasses whichever nodes aren't needed."""
    character_lora = None
    resolved_lora_strength = character_lora_strength
    if trigger:
        loradef = get_character(trigger).get("lora")
        if loradef:
            fname = loradef["file"]
            if os.path.isfile(os.path.join(COMFYUI_LORAS_DIR, fname)):
                character_lora = fname
                if resolved_lora_strength is None:
                    resolved_lora_strength = loradef.get("strength_model", client.CHARACTER_LORA_STRENGTH)
                fam = loradef.get("base_family")
                if fam and effective_checkpoint and _checkpoint_family(effective_checkpoint) != fam:
                    print(f"warning: character LoRA {fname} was trained on '{fam}' but checkpoint "
                          f"{effective_checkpoint!r} is '{_checkpoint_family(effective_checkpoint)}' - "
                          "identity may transfer poorly", flush=True)
                # LoRA carries identity, so FaceID only corrects drift - drop its
                # weight unless the caller explicitly changed it from the default.
                if anchor_path and ip_adapter_weight == client.IP_ADAPTER_WEIGHT:
                    ip_adapter_weight = loradef.get("faceid_weight", 0.7)
            else:
                print(f"warning: character LoRA {fname} not found in {COMFYUI_LORAS_DIR} - "
                      "generating with FaceID only (train it on RunPod, see runpod_bundle.py install)", flush=True)

    ip_filename = client.upload_reference_image(anchor_path) if anchor_path else None
    pose_filename = client.upload_reference_image(pose_reference_path) if pose_reference_path else None

    hq_kwargs = {}
    if effective_checkpoint:
        hq_kwargs["checkpoint"] = client.CHECKPOINTS[effective_checkpoint]
    if lora_strength is not None:
        hq_kwargs["lora_strength"] = lora_strength
    elif effective_checkpoint in client.PONY_CHECKPOINTS:
        hq_kwargs["lora_strength"] = 0.0  # style slider was tuned for juggernaut, off for Pony
    if controlnet_strength is not None:
        hq_kwargs["controlnet_strength"] = controlnet_strength

    return client.submit_generation_hq(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        filename_prefix=stem,
        ip_adapter_image_filename=ip_filename,
        width=width,
        height=height,
        ip_adapter_weight=ip_adapter_weight,
        character_lora=character_lora,
        character_lora_strength=(resolved_lora_strength if resolved_lora_strength is not None
                                 else client.CHARACTER_LORA_STRENGTH),
        pose_image_filename=pose_filename,
        hires=True,
        hires_denoise=(hires_denoise if hires_denoise is not None else client.HIRES_DENOISE),
        use_facedetailer=use_facedetailer,
        face_denoise=(face_denoise if face_denoise is not None else client.FACEDETAILER_FACE_DENOISE),
        hand_denoise=(hand_denoise if hand_denoise is not None else client.FACEDETAILER_HAND_DENOISE),
        **hq_kwargs,
    )


def gen_custom(prompt, extra_negative, tier, trigger, anchor_path, out_dir, seed, filename, ip_adapter_weight,
                pose_reference_path=None, controlnet_strength=None, width=None, height=None,
                use_facedetailer=None, facedetailer_denoise=None, facedetailer_backend="yolo",
                style_positive=None, style_negative=None, checkpoint=None, lora_strength=None,
                hq=True, character_lora_strength=None, hires_denoise=None,
                facedetailer_face_denoise=None, facedetailer_hand_denoise=None):
    """Free-form prompt generation for one-off tests. The descriptive part of
    the prompt is fully up to the caller, but the safety negatives are not a
    dial that gets turned off here: AGE_SAFETY_NEGATIVE is always included,
    and even the "suggestive" tier still blocks explicit content via
    SUGGESTIVE_NEGATIVE - same ceiling as test-suggestive.

    trigger and anchor_path are independent: trigger (if given) prepends a
    known CHARACTERS profile's identity description; anchor_path (if given)
    face-conditions via IP-Adapter. Either can be used without the other -
    e.g. a custom-uploaded anchor with no trigger, for a one-off identity
    that isn't one of the predefined characters. anchor_path must itself be
    a fictional/AI-generated face, same as every other anchor in this
    project - never a real person's photo.

    pose_reference_path (optional) additionally skeleton-conditions the pose
    via ControlNet (see comfyui_client.submit_generation_with_pose) - the
    OpenposePreprocessor node extracts a skeleton from this photo, no
    per-character setup needed. Requires anchor_path (the workflow still
    face-conditions via IP-Adapter alongside the pose control); pose_reference
    is about body position, not identity, so it doesn't need to be the same
    photo as anchor_path and doesn't carry the same fictional-only
    requirement - a real person's photo is fine here since only the
    stick-figure pose is extracted, no face/identity data crosses over.

    facedetailer_backend picks which face detector the face pass uses when
    use_facedetailer is set: "yolo" (default, bounding-box detection via
    face_yolov8n.pt) or "mediapipe" (face-mesh contour via
    MediaPipeFaceMeshToSEGS - follows the actual face shape instead of a
    rectangle, so less hair/background gets pulled into the re-sampled
    region). Hand pass is YOLO either way - no mediapipe hand-mesh node.

    style_positive/style_negative override REALISTIC_STYLE/REALISTIC_NEGATIVE
    (the photorealism-tuning terms) for this call only - meant to be exposed
    as editable GUI fields so prompt/negative-prompt tuning (e.g. fighting
    the "3D render" look, or SDXL's usual face/hand artifacts) can happen
    from the browser instead of editing this file each time. This is
    strictly the style dial: SAFE_SAFETY_NEGATIVE/SUGGESTIVE_NEGATIVE (age
    protection + explicit-content blocking) are never touched by these and
    have no override path - they're the same regardless of what's passed
    here.

    checkpoint (optional) picks a name from comfyui_client.CHECKPOINTS
    (e.g. "pony") instead of the pipeline default (Juggernaut); omit for the
    default. lora_strength (optional) overrides the sdxl_photorealistic_slider
    LoRA baked into every template at 2.5 - pass 0.0 when using a non-Juggernaut
    checkpoint like pony, since that LoRA was tuned against Juggernaut's
    photoreal style, not Pony's more illustration-leaning training data.

    checkpoints in comfyui_client.PONY_CHECKPOINTS (pony, cyberrealistic_pony,
    pony_realism) expect a "score_9, score_8_up, score_7_up"-style quality-tag
    prefix per Pony's training convention, plus its negative-side complement
    ("score_6, score_5, score_4") - PONY_QUALITY_TAGS/PONY_QUALITY_NEGATIVE_TAGS
    are auto-prepended to prompt/negative_prompt whenever one of these is
    selected, so both stay plain natural language regardless of checkpoint;
    callers never need to type either tag set themselves.

    checkpoints in comfyui_client.SD15_CHECKPOINTS (e.g. "realistic_vision",
    "cyberrealistic") are SD1.5, not SDXL/Pony - only plain txt2img works with
    these so far (no trigger/anchor_path, pose_reference_path, or
    use_facedetailer - those need the SDXL-family IP-Adapter/ControlNet files,
    which don't match SD1.5's UNet/CLIP shape). width/height default to
    comfyui_client.SD15_WIDTH/SD15_HEIGHT (512x768) instead of the SDXL 1024x1024
    default when omitted.

    width/height default to a portrait canvas (FULL_BODY_RESOLUTION) when
    pose_reference_path is given, since a square 1024x1024 canvas has no
    vertical room for a full body and just gets cropped back to upper-body -
    same issue that build_variation_prompt's pick_angle() works around for
    the batch generators. Explicit width/height always wins over that
    default. Plain 1024x1024 stays the default with no pose reference."""
    if checkpoint and checkpoint not in client.CHECKPOINTS:
        raise SystemExit(f"unknown checkpoint {checkpoint!r} - choices: {sorted(client.CHECKPOINTS)}")
    is_sd15 = checkpoint in client.SD15_CHECKPOINTS

    # HQ two-pass path is the default for SDXL/Pony. It combines the FaceID,
    # ControlNet and FaceDetailer chains into one workflow, so the old "can't
    # combine" restrictions only apply to the legacy (hq=False) path. SD1.5
    # has no HQ path (no SDXL-family adapter/upscale wiring), so it falls back.
    use_hq = hq and not is_sd15

    if use_facedetailer is None:
        use_facedetailer = use_hq  # HQ defaults FaceDetailer on; legacy defaults off

    if not use_hq:
        if pose_reference_path and not anchor_path:
            raise SystemExit("pose_reference requires anchor_path (ControlNet workflow still needs a face anchor for IP-Adapter)")
        if use_facedetailer and pose_reference_path:
            raise SystemExit("use_facedetailer and pose_reference are on separate workflow templates and can't be combined yet (turn on hq to combine them)")
        if use_facedetailer and not anchor_path:
            raise SystemExit("use_facedetailer requires anchor_path (the FaceDetailer workflow still needs a face anchor for IP-Adapter)")
    if is_sd15 and (anchor_path or pose_reference_path):
        raise SystemExit(f"checkpoint {checkpoint!r} is SD1.5 - only plain txt2img is wired up "
                          "(anchor/IP-Adapter and pose_reference/ControlNet need the SDXL-family "
                          "adapter files, which don't match SD1.5's UNet/CLIP shape)")

    # HQ custom generation defaults to the Pony-family photoreal checkpoint when
    # the caller didn't pick one (matches the GUI default). Dataset stages
    # (gen_anchors/gen_variations) keep client.CHECKPOINT (juggernaut) - separate.
    effective_checkpoint = checkpoint
    if effective_checkpoint is None and use_hq:
        effective_checkpoint = DEFAULT_CUSTOM_CHECKPOINT

    if width is None or height is None:
        if is_sd15:
            width, height = client.SD15_WIDTH, client.SD15_HEIGHT
        else:
            width, height = FULL_BODY_RESOLUTION if pose_reference_path else (client.WIDTH, client.HEIGHT)

    ckpt_kwargs = {}
    if checkpoint:
        ckpt_kwargs["checkpoint"] = client.CHECKPOINTS[checkpoint]
    if lora_strength is not None:
        ckpt_kwargs["lora_strength"] = lora_strength

    os.makedirs(out_dir, exist_ok=True)
    stem = filename or f"custom_seed{seed}"

    full_prompt, negative_prompt = _build_prompt_and_negative(
        prompt, extra_negative, tier, trigger, style_positive, style_negative, effective_checkpoint,
    )

    if use_hq:
        raw_path = _run_hq(
            full_prompt, negative_prompt, seed, stem, trigger, anchor_path, pose_reference_path,
            width, height, ip_adapter_weight, effective_checkpoint, lora_strength,
            character_lora_strength, controlnet_strength,
            hires_denoise, use_facedetailer,
            facedetailer_face_denoise if facedetailer_face_denoise is not None else facedetailer_denoise,
            facedetailer_hand_denoise,
        )
    elif pose_reference_path:
        ref_filename = client.upload_reference_image(anchor_path)
        pose_filename = client.upload_reference_image(pose_reference_path)
        kwargs = dict(ckpt_kwargs)
        if controlnet_strength is not None:
            kwargs["controlnet_strength"] = controlnet_strength
        raw_path = client.submit_generation_with_pose(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            ip_adapter_image_filename=ref_filename,
            pose_image_filename=pose_filename,
            filename_prefix=stem,
            ip_adapter_weight=ip_adapter_weight,
            width=width,
            height=height,
            **kwargs,
        )
    elif use_facedetailer:
        ref_filename = client.upload_reference_image(anchor_path)
        kwargs = dict(ckpt_kwargs)
        if facedetailer_denoise is not None:
            kwargs["facedetailer_denoise"] = facedetailer_denoise
        submit_fn = (client.submit_generation_with_facedetailer_mediapipe if facedetailer_backend == "mediapipe"
                     else client.submit_generation_with_facedetailer)
        raw_path = submit_fn(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            ip_adapter_image_filename=ref_filename,
            filename_prefix=stem,
            ip_adapter_weight=ip_adapter_weight,
            width=width,
            height=height,
            **kwargs,
        )
    elif anchor_path:
        ref_filename = client.upload_reference_image(anchor_path)
        raw_path = client.submit_generation(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            ip_adapter_image_filename=ref_filename,
            filename_prefix=stem,
            ip_adapter_weight=ip_adapter_weight,
            width=width,
            height=height,
            **ckpt_kwargs,
        )
    elif is_sd15:
        raw_path = client.submit_txt2img_generation_sd15(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            filename_prefix=stem,
            checkpoint=client.CHECKPOINTS[checkpoint],
            width=width,
            height=height,
        )
    else:
        raw_path = client.submit_txt2img_generation(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            filename_prefix=stem,
            width=width,
            height=height,
            **ckpt_kwargs,
        )

    img_path = os.path.join(out_dir, f"{stem}.png")
    os.replace(raw_path, img_path)
    print(f"saved {img_path}", flush=True)


GIF_WIGGLE_DENOISE = 0.45  # measured, not guessed: diffed actual output pixels at 0.3/0.5/0.7 against
# the same base frame (juggernaut, 1024x1024). 0.3 -> ~19% of pixels changed >10/255, 0% changed
# >50/255 - reads as static because it's texture/edge shimmer only (skin/hair/fabric grain jitter),
# no actual pose movement. 0.7 -> 93%/36% changed, and the diff shows visible double-exposure
# ghosting (hand and head clearly in two different positions) - past the "same pose" point, this is
# a different pose the sampler resampled its way into, not the base frame wiggling. 0.45 sits inside
# the still-unmeasured gap between those two data points, picked as a starting point that should read
# as visible motion without yet crossing into pose drift - not itself verified at this exact value,
# only bracketed between two that were.


def gen_gif(prompt, extra_negative, tier, trigger, anchor_path, out_dir, base_seed, frame_count,
            ip_adapter_weight, duration_ms=300, denoise=GIF_WIGGLE_DENOISE, width=None, height=None,
            style_positive=None, style_negative=None, checkpoint=None, lora_strength=None):
    """Generates ONE base frame via gen_custom() (full txt2img, establishes
    the person/pose/scene), then frame_count-1 more frames via low-denoise
    img2img on that SAME base frame image (same prompt, different seed each
    time), then stitches all frames into an animated GIF with Pillow.

    This is a fix for an earlier version of this function that called
    gen_custom() from scratch for every frame (same prompt, different seed,
    nothing else shared) - independent txt2img samples don't share any
    structure with each other even under an identical prompt, so that
    produced a "different person/outfit/scene each frame" slideshow instead
    of the same scene wiggling. img2img at a low denoise barely perturbs the
    base frame's composition, so every frame stays the same person, same
    pose, same scene - the different seed only introduces small per-frame
    variation (the "wiggle"), because the sampler starts from the base
    frame's own (lightly re-noised) latent instead of pure noise.

    denoise controls how far each wiggle frame is allowed to drift from the
    base frame: 0 = identical to the base (no motion at all), 1 = fully
    independent resample (back to the old from-scratch behavior). Measured
    against real output (see GIF_WIGGLE_DENOISE's comment): 0.3 reads as
    static (only texture/edge-level jitter, no pose movement), 0.7 already
    shows the pose visibly drifting (double-exposure-style ghosting in a
    pixel diff against the base frame) - default 0.45 sits in the
    unmeasured gap between those two, turn it down toward 0.3 if frames
    still drift too far, up toward 0.6-0.7 if they still look static.

    Still not real motion like gen_video_animatediff - there's no temporal
    model, just the same base image repeatedly perturbed with fresh noise -
    but every frame now actually shares the base frame's composition, which
    independent per-frame txt2img never guaranteed.

    A face reference is required for the wiggle frames' IP-Adapter FaceID
    conditioning: anchor_path if given, otherwise the base frame's own
    generated image is used as its own face reference (self-conditioning) -
    this assumes the base frame actually contains a recognizable face, which
    holds for this project's normal character-generation use but won't for
    an anchor-less, trigger-less object/scene prompt (e.g. "a cup of coffee
    on a wooden table" from gen_custom's own docstring example) - IP-Adapter
    FaceID has no face to lock onto in that case. gen_gif is meant for
    character generation, not that edge case; use gen_custom directly for a
    single object/scene image instead.

    Deliberately skips pose_reference_path/use_facedetailer (gen_custom's
    per-frame extras) - not wired into the img2img workflow template."""
    if checkpoint in client.SD15_CHECKPOINTS:
        raise SystemExit(f"checkpoint {checkpoint!r} is SD1.5 - gen_gif's img2img wiggle frames need "
                          "IP-Adapter, which needs the SDXL-family model files SD1.5 doesn't have")

    os.makedirs(out_dir, exist_ok=True)

    base_stem = f"gif_frame_000_seed{base_seed}"
    gen_custom(prompt, extra_negative, tier, trigger, anchor_path, out_dir, base_seed, base_stem, ip_adapter_weight,
               width=width, height=height,
               style_positive=style_positive, style_negative=style_negative,
               checkpoint=checkpoint, lora_strength=lora_strength,
               hq=False, use_facedetailer=False)  # gif base frame stays single-pass to match the img2img wiggle frames' size/speed
    base_path = os.path.join(out_dir, f"{base_stem}.png")
    frame_paths = [base_path]

    full_prompt, negative_prompt = _build_prompt_and_negative(
        prompt, extra_negative, tier, trigger, style_positive, style_negative, checkpoint,
    )
    ckpt_kwargs = {}
    if checkpoint:
        ckpt_kwargs["checkpoint"] = client.CHECKPOINTS[checkpoint]
    if lora_strength is not None:
        ckpt_kwargs["lora_strength"] = lora_strength

    face_ref_path = anchor_path or base_path
    ip_ref_filename = client.upload_reference_image(face_ref_path)
    base_image_filename = client.upload_reference_image(base_path)

    for i in range(1, frame_count):
        seed = base_seed + i
        stem = f"gif_frame_{i:03d}_seed{seed}"
        raw_path = client.submit_img2img_generation(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            ip_adapter_image_filename=ip_ref_filename,
            init_image_filename=base_image_filename,
            denoise=denoise,
            filename_prefix=stem,
            ip_adapter_weight=ip_adapter_weight,
            **ckpt_kwargs,
        )
        frame_path = os.path.join(out_dir, f"{stem}.png")
        os.replace(raw_path, frame_path)
        print(f"saved {frame_path}", flush=True)
        frame_paths.append(frame_path)

    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    gif_path = os.path.join(out_dir, f"gif_seed{base_seed}.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"saved {gif_path}", flush=True)
    return gif_path


def gen_video(trigger, init_image_path, out_dir, seed, video_frames, fps, motion_bucket_id):
    """Animate an existing image (an anchor or a dataset variation) with SVD
    img2vid. Low-res/low-frame by default - see comfyui_client's VIDEO_* -
    this is a "does it move" pass, not a final-quality render."""
    get_character(trigger)  # validates trigger, keeps the same fail-fast shape as the other modes
    os.makedirs(out_dir, exist_ok=True)
    ref_filename = client.upload_reference_image(init_image_path)
    stem = f"video_seed{seed}"

    client.log_gpu_memory("before_video")
    raw_path = client.submit_img2vid_generation(
        init_image_filename=ref_filename,
        seed=seed,
        filename_prefix=stem,
        video_frames=video_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
    )
    out_path = os.path.join(out_dir, f"{stem}.webm")
    os.replace(raw_path, out_path)
    print(f"saved {out_path}", flush=True)


def gen_video_animatediff(prompt, extra_negative, tier, trigger, face_ref_path, out_dir, seed,
                           ip_adapter_weight=client.IP_ADAPTER_WEIGHT, facedetailer_denoise=None,
                           frames=None, fps=None, width=None, height=None,
                           style_positive=None, style_negative=None, checkpoint=None,
                           motion_lora=None, motion_lora_strength=1.0):
    """AnimateDiff (SD1.5) txt2vid with IPAdapter-FaceID identity locking and
    a per-frame FaceDetailer face-fix pass - fixes the face warping that
    gen_video's plain SVD img2vid produces (SVD's temporal U-Net can't be
    IPAdapter-patched, so it has no way to hold identity/geometry steady
    across frames; AnimateDiff runs the motion module inside a normal SD1.5
    UNet, so FaceID and FaceDetailer both work on it exactly like the still-
    image paths in gen_custom).

    face_ref_path plays the same role anchor_path does in gen_custom - a
    fictional/AI-generated face image, never a real person's photo. Safety-
    negative and style-positive/negative composition mirrors gen_custom
    exactly (same AGE_SAFETY_NEGATIVE/SUGGESTIVE_NEGATIVE guarantee, same
    REALISTIC_STYLE/REALISTIC_NEGATIVE override mechanism).

    checkpoint (optional) picks a name from comfyui_client.ANIMATEDIFF_CHECKPOINTS
    (e.g. "realistic_vision", "cyberrealistic") instead of the default plain
    SD1.5 base - must be SD1.5 (same UNet shape the motion module patches into),
    NOT one of the SDXL/Pony names from comfyui_client.CHECKPOINTS.

    motion_lora (optional) picks a name from comfyui_client.ANIMATEDIFF_MOTION_LORAS
    (e.g. "zoom_in") - conditions the motion module for that specific CAMERA
    movement (zoom/pan/tilt/roll of the whole frame). This is NOT a control for
    body-part-specific physical effects (no "bounce"/"jiggle" knob exists -
    the motion module has no dedicated mechanism for that). motion_lora_strength
    scales its effect, typical useful range is 0-2."""
    if checkpoint and checkpoint not in client.ANIMATEDIFF_CHECKPOINTS:
        raise SystemExit(f"unknown animatediff checkpoint {checkpoint!r} - choices: {sorted(client.ANIMATEDIFF_CHECKPOINTS)}")
    if motion_lora and motion_lora not in client.ANIMATEDIFF_MOTION_LORAS:
        raise SystemExit(f"unknown motion lora {motion_lora!r} - choices: {sorted(client.ANIMATEDIFF_MOTION_LORAS)}")
    style_positive = REALISTIC_STYLE if style_positive is None else style_positive
    style_negative = REALISTIC_NEGATIVE if style_negative is None else style_negative

    safety_negative = SAFE_SAFETY_NEGATIVE if tier == "safe" else SUGGESTIVE_NEGATIVE
    base_negative = f"{safety_negative}, {style_negative}" if style_negative else safety_negative
    negative_prompt = f"{base_negative}, {extra_negative}" if extra_negative else base_negative

    if trigger:
        profile = get_character(trigger)
        full_prompt = f"{character_base_prompt(trigger, profile)}, {prompt}"
    else:
        full_prompt = prompt
    if style_positive:
        full_prompt = f"{full_prompt}, {style_positive}"

    os.makedirs(out_dir, exist_ok=True)
    stem = f"animatediff_seed{seed}"
    ref_filename = client.upload_reference_image(face_ref_path)

    kwargs = {}
    if facedetailer_denoise is not None:
        kwargs["facedetailer_denoise"] = facedetailer_denoise
    if frames is not None:
        kwargs["frames"] = frames
    if fps is not None:
        kwargs["fps"] = fps
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    if checkpoint:
        kwargs["checkpoint"] = client.ANIMATEDIFF_CHECKPOINTS[checkpoint]
    if motion_lora:
        kwargs["motion_lora_name"] = client.ANIMATEDIFF_MOTION_LORAS[motion_lora]
        kwargs["motion_lora_strength"] = motion_lora_strength

    client.log_gpu_memory("before_animatediff")
    raw_path = client.submit_generation_animatediff(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        face_ref_image_filename=ref_filename,
        filename_prefix=stem,
        ip_adapter_weight=ip_adapter_weight,
        **kwargs,
    )
    out_path = os.path.join(out_dir, f"{stem}.webm")
    os.replace(raw_path, out_path)
    print(f"saved {out_path}", flush=True)
    client.log_gpu_memory("after_animatediff")
    client.log_gpu_memory("after_video")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("list-characters")

    p_anchor = sub.add_parser("anchor")
    p_anchor.add_argument("--character", required=True, choices=sorted(CHARACTERS))
    p_anchor.add_argument("--out", default=None)
    p_anchor.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])

    p_var = sub.add_parser("variations")
    p_var.add_argument("--character", required=True, choices=sorted(CHARACTERS))
    p_var.add_argument("--anchor", required=True)
    p_var.add_argument("--out", default=None)
    p_var.add_argument("--count", type=int, default=100)
    p_var.add_argument("--ip-adapter-weight", type=float, default=client.IP_ADAPTER_WEIGHT)
    p_var.add_argument("--seed", type=int, default=2000)

    p_test_nsfw = sub.add_parser("test-suggestive")
    p_test_nsfw.add_argument("--character", required=True, choices=sorted(CHARACTERS))
    p_test_nsfw.add_argument("--anchor", required=True)
    p_test_nsfw.add_argument("--out", default=r"D:\AI-Image-Lab\training\reference_candidates")
    p_test_nsfw.add_argument("--seed", type=int, default=5000)
    p_test_nsfw.add_argument("--ip-adapter-weight", type=float, default=client.IP_ADAPTER_WEIGHT)

    p_var_sugg = sub.add_parser("variations-suggestive")
    p_var_sugg.add_argument("--character", required=True, choices=sorted(CHARACTERS))
    p_var_sugg.add_argument("--anchor", required=True)
    p_var_sugg.add_argument("--out", default=None)
    p_var_sugg.add_argument("--count", type=int, default=20)
    p_var_sugg.add_argument("--ip-adapter-weight", type=float, default=client.IP_ADAPTER_WEIGHT)
    p_var_sugg.add_argument("--seed", type=int, default=7000)

    p_custom = sub.add_parser("custom")
    p_custom.add_argument("--prompt", required=True, help="free-form positive prompt; combined with the character's identity prompt if --character is given")
    p_custom.add_argument("--negative-prompt", default="", help="extra negative terms, appended on top of the mandatory safety negatives (never replaces them)")
    p_custom.add_argument("--tier", choices=["safe", "suggestive"], default="safe", help="'suggestive' allows swimwear/lingerie-level content, same ceiling as test-suggestive - explicit content stays blocked either way")
    p_custom.add_argument("--character", default=None, choices=sorted(CHARACTERS), help="optional - prepends this character's identity description")
    p_custom.add_argument("--anchor", default=None, help="optional - face-conditions via IP-Adapter on this image, independent of --character. Must be a fictional/AI-generated face, never a real person's photo")
    p_custom.add_argument("--out", default=r"D:\AI-Image-Lab\training\reference_candidates")
    p_custom.add_argument("--seed", type=int, default=9000)
    p_custom.add_argument("--filename", default=None)
    p_custom.add_argument("--ip-adapter-weight", type=float, default=client.IP_ADAPTER_WEIGHT)
    p_custom.add_argument("--pose-reference", default=None, help="optional - skeleton-conditions the pose via ControlNet (OpenPose), extracted from this photo. Requires --anchor. Unlike --anchor, this can be any photo (only stick-figure joint positions are extracted, no face/identity data carries over)")
    p_custom.add_argument("--controlnet-strength", type=float, default=None, help="0=ignored, 1=rigidly locked to the skeleton; defaults to comfyui_client.CONTROLNET_STRENGTH (0.8) if omitted")
    p_custom.add_argument("--width", type=int, default=None, help="defaults to 832 (portrait) if --pose-reference is given, else 1024 (square)")
    p_custom.add_argument("--height", type=int, default=None, help="defaults to 1216 (portrait) if --pose-reference is given, else 1024 (square)")
    p_custom.add_argument("--use-facedetailer", action="store_true", help="ADetailer-style face+hand fix-up pass after the base generation. Requires --anchor. Can't be combined with --pose-reference yet (separate workflow templates)")
    p_custom.add_argument("--facedetailer-denoise", type=float, default=None, help="0=no change, 1=fully re-generate the detected region; defaults to comfyui_client.FACEDETAILER_DENOISE (0.5) if omitted")
    p_custom.add_argument("--facedetailer-backend", choices=["yolo", "mediapipe"], default="yolo", help="face detector for --use-facedetailer's face pass: yolo (bbox, default) or mediapipe (face-mesh contour, needs the mediapipe package)")
    p_custom.add_argument("--style-positive", default=None, help="overrides REALISTIC_STYLE for this call only; omit to use the default")
    p_custom.add_argument("--style-negative", default=None, help="overrides REALISTIC_NEGATIVE for this call only; omit to use the default. Never touches the age-safety/explicit-content negative terms, those aren't overridable")
    p_custom.add_argument("--checkpoint", default=None, choices=sorted(client.CHECKPOINTS), help="swap the SDXL checkpoint for this call only; omit for the pipeline default (juggernaut)")
    p_custom.add_argument("--lora-strength", type=float, default=None, help="overrides the sdxl_photorealistic_slider LoRA strength (baked into every template at 2.5); pass 0.0 when using --checkpoint pony, since that LoRA was tuned for juggernaut's photoreal style")
    p_custom.add_argument("--no-hq", action="store_true", help="disable the default HQ two-pass path (base -> ESRGAN hires -> face/hand FaceDetailer) and use the legacy single-pass workflow instead")
    p_custom.add_argument("--no-facedetailer", action="store_true", help="turn OFF the HQ path's face/hand FaceDetailer passes (on by default in HQ mode)")
    p_custom.add_argument("--character-lora-strength", type=float, default=None, help="strength for the per-character LoRA (HQ path); defaults to the CHARACTERS profile's value or 0.8. Only used when the character has a trained LoRA")
    p_custom.add_argument("--hires-denoise", type=float, default=None, help="HQ hires second-pass denoise; defaults to comfyui_client.HIRES_DENOISE (0.4). Higher restores more detail but risks identity drift")

    p_gif = sub.add_parser("gif", help="batch-generate frame_count independent stills (same prompt, incrementing seed) and stitch them into an animated GIF - a fast flipbook-style preview, NOT smooth motion like video-animatediff")
    p_gif.add_argument("--prompt", required=True, help="free-form positive prompt; combined with the character's identity prompt if --character is given")
    p_gif.add_argument("--negative-prompt", default="", help="extra negative terms, appended on top of the mandatory safety negatives (never replaces them)")
    p_gif.add_argument("--tier", choices=["safe", "suggestive"], default="safe")
    p_gif.add_argument("--character", default=None, choices=sorted(CHARACTERS), help="optional - prepends this character's identity description")
    p_gif.add_argument("--anchor", default=None, help="optional - face-conditions via IP-Adapter on this image. Must be a fictional/AI-generated face, never a real person's photo")
    p_gif.add_argument("--out", default=r"D:\AI-Image-Lab\training\reference_candidates")
    p_gif.add_argument("--seed", type=int, default=9000, help="base seed - frame i uses seed + i")
    p_gif.add_argument("--frames", type=int, default=8, help="total frames in the GIF - frame 1 is a full txt2img generation, frames 2+ are low-denoise img2img 'wiggle' variations of frame 1 (see gen_gif's docstring)")
    p_gif.add_argument("--duration-ms", type=int, default=300, help="milliseconds each frame is shown for")
    p_gif.add_argument("--denoise", type=float, default=GIF_WIGGLE_DENOISE, help="how far each wiggle frame can drift from frame 1's composition; 0=identical/no motion, 1=fully independent resample. Measured: 0.3 reads as static (texture jitter only), 0.7 already shows the pose visibly drifting - default 0.45 is the untested midpoint")
    p_gif.add_argument("--ip-adapter-weight", type=float, default=client.IP_ADAPTER_WEIGHT)
    p_gif.add_argument("--width", type=int, default=None)
    p_gif.add_argument("--height", type=int, default=None)
    p_gif.add_argument("--style-positive", default=None, help="overrides REALISTIC_STYLE for this call only; omit to use the default")
    p_gif.add_argument("--style-negative", default=None, help="overrides REALISTIC_NEGATIVE for this call only; omit to use the default")
    p_gif.add_argument("--checkpoint", default=None, choices=sorted(client.CHECKPOINTS), help="swap the SDXL checkpoint for this call only; omit for the pipeline default (juggernaut)")
    p_gif.add_argument("--lora-strength", type=float, default=None, help="overrides the sdxl_photorealistic_slider LoRA strength; pass 0.0 when using a pony-family --checkpoint")

    p_video = sub.add_parser("video")
    p_video.add_argument("--character", required=True, choices=sorted(CHARACTERS))
    p_video.add_argument("--init-image", required=True, help="existing anchor or dataset image to animate")
    p_video.add_argument("--out", default=None)
    p_video.add_argument("--seed", type=int, default=6001)
    p_video.add_argument("--frames", type=int, default=client.VIDEO_FRAMES)
    p_video.add_argument("--fps", type=int, default=client.VIDEO_FPS)
    p_video.add_argument("--motion", type=int, default=client.MOTION_BUCKET_ID, help="motion_bucket_id - higher = more motion, less coherent")

    p_video_ad = sub.add_parser("video-animatediff", help="AnimateDiff (SD1.5) txt2vid with FaceID + FaceDetailer - fixes the face warping plain 'video' (SVD) produces")
    p_video_ad.add_argument("--prompt", required=True, help="free-form positive prompt; combined with the character's identity prompt if --character is given")
    p_video_ad.add_argument("--negative-prompt", default="", help="extra negative terms, appended on top of the mandatory safety negatives (never replaces them)")
    p_video_ad.add_argument("--tier", choices=["safe", "suggestive"], default="safe")
    p_video_ad.add_argument("--character", default=None, choices=sorted(CHARACTERS), help="optional - prepends this character's identity description")
    p_video_ad.add_argument("--face-ref", required=True, help="face image for IPAdapter-FaceID identity locking - must be a fictional/AI-generated face, never a real person's photo")
    p_video_ad.add_argument("--out", default=r"D:\AI-Image-Lab\training\reference_candidates\videos")
    p_video_ad.add_argument("--seed", type=int, default=6001)
    p_video_ad.add_argument("--ip-adapter-weight", type=float, default=client.IP_ADAPTER_WEIGHT)
    p_video_ad.add_argument("--facedetailer-denoise", type=float, default=None, help="0=no change, 1=fully re-generate the detected face region; defaults to comfyui_client.FACEDETAILER_DENOISE (0.5) if omitted")
    p_video_ad.add_argument("--frames", type=int, default=None, help="defaults to comfyui_client.ANIMATEDIFF_FRAMES (16, the motion module's trained context length)")
    p_video_ad.add_argument("--fps", type=int, default=None, help="defaults to comfyui_client.ANIMATEDIFF_FPS (8) if omitted")
    p_video_ad.add_argument("--width", type=int, default=None, help="defaults to comfyui_client.ANIMATEDIFF_WIDTH (512) if omitted")
    p_video_ad.add_argument("--height", type=int, default=None, help="defaults to comfyui_client.ANIMATEDIFF_HEIGHT (512) if omitted")
    p_video_ad.add_argument("--style-positive", default=None, help="overrides REALISTIC_STYLE for this call only; omit to use the default")
    p_video_ad.add_argument("--style-negative", default=None, help="overrides REALISTIC_NEGATIVE for this call only; omit to use the default. Never touches the age-safety/explicit-content negative terms, those aren't overridable")
    p_video_ad.add_argument("--checkpoint", default=None, choices=sorted(client.ANIMATEDIFF_CHECKPOINTS), help="swap the SD1.5 checkpoint AnimateDiff patches its motion module into; omit for the pipeline default (plain SD1.5 base)")
    p_video_ad.add_argument("--motion-lora", default=None, choices=sorted(client.ANIMATEDIFF_MOTION_LORAS), help="camera-motion LoRA (zoom/pan/tilt/roll of the whole frame) - NOT a body-part physics control, no bounce/jiggle option exists. Requires the .ckpt downloaded to ComfyUI/models/animatediff_motion_lora/, see README")
    p_video_ad.add_argument("--motion-lora-strength", type=float, default=1.0, help="scales the motion LoRA's effect; typical useful range is 0-2, values much above 1 tend to distort the frame")

    args = parser.parse_args()
    if args.mode == "list-characters":
        for name, profile in sorted(CHARACTERS.items()):
            print(f"{name}: age {profile['age']}, {profile['appearance']}, {profile['style']}")
    elif args.mode == "anchor":
        out = args.out or rf"D:\AI-Image-Lab\training\reference_candidates\{args.character}"
        gen_anchors(args.character, out, args.seeds)
    elif args.mode == "variations":
        out = args.out or rf"D:\AI-Image-Lab\datasets\{args.character}"
        gen_variations(args.character, args.anchor, out, args.count, args.ip_adapter_weight, args.seed)
    elif args.mode == "variations-suggestive":
        out = args.out or rf"D:\AI-Image-Lab\datasets\{args.character}"
        gen_suggestive_variations(args.character, args.anchor, out, args.count, args.ip_adapter_weight, args.seed)
    elif args.mode == "video":
        out = args.out or rf"D:\AI-Image-Lab\training\reference_candidates\{args.character}\videos"
        gen_video(args.character, args.init_image, out, args.seed, args.frames, args.fps, args.motion)
    elif args.mode == "video-animatediff":
        gen_video_animatediff(args.prompt, args.negative_prompt, args.tier, args.character, args.face_ref, args.out, args.seed,
                               ip_adapter_weight=args.ip_adapter_weight, facedetailer_denoise=args.facedetailer_denoise,
                               frames=args.frames, fps=args.fps, width=args.width, height=args.height,
                               style_positive=args.style_positive, style_negative=args.style_negative,
                               checkpoint=args.checkpoint,
                               motion_lora=args.motion_lora, motion_lora_strength=args.motion_lora_strength)
    elif args.mode == "custom":
        # use_facedetailer sentinel: None = auto (on for HQ, off for legacy);
        # explicit flags override. --use-facedetailer forces on, --no-facedetailer off.
        uf = None
        if args.use_facedetailer:
            uf = True
        if args.no_facedetailer:
            uf = False
        gen_custom(args.prompt, args.negative_prompt, args.tier, args.character, args.anchor, args.out, args.seed, args.filename, args.ip_adapter_weight,
                   pose_reference_path=args.pose_reference, controlnet_strength=args.controlnet_strength,
                   width=args.width, height=args.height,
                   use_facedetailer=uf, facedetailer_denoise=args.facedetailer_denoise,
                   facedetailer_backend=args.facedetailer_backend,
                   style_positive=args.style_positive, style_negative=args.style_negative,
                   checkpoint=args.checkpoint, lora_strength=args.lora_strength,
                   hq=not args.no_hq, character_lora_strength=args.character_lora_strength,
                   hires_denoise=args.hires_denoise)
    elif args.mode == "gif":
        gen_gif(args.prompt, args.negative_prompt, args.tier, args.character, args.anchor, args.out, args.seed, args.frames,
                args.ip_adapter_weight, duration_ms=args.duration_ms, denoise=args.denoise,
                width=args.width, height=args.height,
                style_positive=args.style_positive, style_negative=args.style_negative,
                checkpoint=args.checkpoint, lora_strength=args.lora_strength)
    else:
        gen_test_suggestive(args.character, args.anchor, args.out, args.seed, args.ip_adapter_weight)
