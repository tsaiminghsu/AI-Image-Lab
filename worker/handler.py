"""RunPod Serverless handler for AI-Image-Lab.

Runs inside the worker container next to a ComfyUI server on localhost:8188.
Reuses the EXACT local generation code (training/comfyui_client.py +
generate_character.py) so the age/tier safety negatives are the same hard
guarantee they are locally - that is why runpod.ts deliberately sends the raw
prompt + tier and lets this worker apply buildSafePrompt's equivalent
(_build_prompt_and_negative) itself, rather than double-applying it.

Input (matches web/amplify/functions/generate/providers/runpod.ts):
  {"input": {"prompt", "tier", "characterId", "referenceImageBase64",
             "referenceImageContentType", "loraId", "aspectRatio", "seed"}}
Output on success:
  {"outputKey", "outputUrl", "seed", "width", "height"}
On any failure returns {"error": "..."} so RunPod marks the job FAILED.
"""

import base64
import binascii
import io
import os
import random
import tempfile

import boto3
import runpod

import comfyui_client as client
import generate_character as gc

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
WORKER_CHECKPOINT = os.environ.get("WORKER_CHECKPOINT", "cyberrealistic_pony")
ANCHOR_DIR = os.environ.get("RUNPOD_ANCHOR_DIR", "/runpod-volume/anchors")

# aspectRatio -> (width, height). Values are SDXL-friendly and match the set
# provider-types.ts declares. These are FINAL sizes; submit_generation_hq
# derives the lower first-pass size itself.
ASPECT_SIZES = {
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (832, 1216),
    "16:9": (1216, 832),
}
MAX_PROMPT_CHARS = 2000
_s3 = boto3.client("s3", region_name=AWS_REGION)


def _decode_reference(b64, job_id):
    """Decode a base64 reference image, verify it really is a PNG/JPEG (never
    trust the declared content type), and write it to a tempfile. Returns the
    path, or raises ValueError."""
    from PIL import Image

    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"referenceImageBase64 is not valid base64: {exc}")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()  # structural check; raises on anything that isn't a real image
        fmt = (img.format or "").upper()
    except Exception as exc:
        raise ValueError(f"referenceImageBase64 is not a decodable image: {exc}")
    if fmt not in ("PNG", "JPEG"):
        raise ValueError(f"reference image must be PNG or JPEG, got {fmt or 'unknown'}")
    fd, path = tempfile.mkstemp(prefix=f"ref_{job_id}_", suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def _anchor_for_character(character_id):
    """Fallback face reference from the Network Volume when the caller didn't
    upload one: anchors/<characterId>/anchor_*.png (mirrors the local
    reference_candidates layout)."""
    import glob

    if not character_id:
        return None
    matches = sorted(glob.glob(os.path.join(ANCHOR_DIR, character_id, "anchor_*.png")))
    if not matches:
        matches = sorted(glob.glob(os.path.join(ANCHOR_DIR, character_id, "*.png")))
    return matches[0] if matches else None


def handler(job):
    tmp_files = []
    try:
        if not S3_BUCKET:
            return {"error": "S3_BUCKET env var not set on the worker"}

        job_id = job.get("id", "unknown")
        inp = job.get("input") or {}

        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "missing prompt"}
        if len(prompt) > MAX_PROMPT_CHARS:
            return {"error": f"prompt too long (>{MAX_PROMPT_CHARS} chars)"}

        # Never trust an arbitrary tier - anything but the two known values
        # falls back to the strictest ("safe"), same ceiling logic as locally.
        tier = inp.get("tier")
        if tier not in ("safe", "suggestive"):
            tier = "safe"

        aspect = inp.get("aspectRatio") or "3:4"
        if aspect not in ASPECT_SIZES:
            return {"error": f"unknown aspectRatio {aspect!r}; choices: {sorted(ASPECT_SIZES)}"}
        width, height = ASPECT_SIZES[aspect]

        character_id = inp.get("characterId") or None
        trigger = None
        if character_id:
            gc.get_character(character_id)  # raises SystemExit on unknown -> caught below
            trigger = character_id

        # Reference face: uploaded image wins, else the character's anchor.
        ref_b64 = inp.get("referenceImageBase64")
        if ref_b64:
            ref_path = _decode_reference(ref_b64, job_id)
            tmp_files.append(ref_path)
        else:
            ref_path = _anchor_for_character(character_id)
        if not ref_path:
            return {"error": "no reference face: provide referenceImageBase64 or a characterId with an anchor on the volume"}

        seed = inp.get("seed")
        if not isinstance(seed, int):
            seed = random.randint(0, 2**31 - 1)

        checkpoint_key = WORKER_CHECKPOINT
        if checkpoint_key not in client.CHECKPOINTS:
            return {"error": f"WORKER_CHECKPOINT {checkpoint_key!r} not in CHECKPOINTS"}

        # Same composition point as CLI/GUI: applies AGE_SAFETY_NEGATIVE and the
        # tier ceiling, and the Pony quality-tag prefix when relevant.
        full_prompt, negative_prompt = gc._build_prompt_and_negative(
            prompt, "", tier, trigger, None, None, checkpoint_key,
        )

        lora_strength = 0.0 if checkpoint_key in client.PONY_CHECKPOINTS else None

        lora_id = inp.get("loraId") or None
        character_lora = f"characters/{lora_id}.safetensors" if lora_id else None
        ip_weight = client.IP_ADAPTER_WEIGHT
        if character_lora:
            ip_weight = 0.7  # LoRA carries identity; FaceID only corrects drift

        ip_name = client.upload_reference_image(ref_path)

        out_path = client.submit_generation_hq(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            filename_prefix=f"rp_{job_id}",
            ip_adapter_image_filename=ip_name,
            width=width,
            height=height,
            checkpoint=client.CHECKPOINTS[checkpoint_key],
            lora_strength=lora_strength,
            character_lora=character_lora,
            ip_adapter_weight=ip_weight,
            hires=True,
            use_facedetailer=True,
        )
        tmp_files.append(out_path)

        key = f"generated/{job_id}.png"
        _s3.upload_file(out_path, S3_BUCKET, key, ExtraArgs={"ContentType": "image/png"})
        # Debug-only presigned URL; the status Lambda re-presigns from outputKey.
        url = _s3.generate_presigned_url(
            "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600,
        )
        return {"outputKey": key, "outputUrl": url, "seed": seed, "width": width, "height": height}

    except BaseException as exc:  # SystemExit from get_character included
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
