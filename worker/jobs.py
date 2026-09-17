"""Job parsing and dispatch for the RunPod worker, kept free of boto3/runpod so it tests offline.

handler.py is the thin RunPod-facing wrapper (S3 upload, progress_update, runpod.serverless.start);
everything that decides WHAT to generate lives here and goes through the same generate_character /
comfyui_client code as the local CLI and GUI. That is the point of running our own worker instead of
a hosted model: the tier + age safety negatives and the cfg floor are applied by the one shared
builder, not re-implemented per provider.

Input schema (camelCase, matching web/amplify/functions/generate/providers/runpod.ts):
  common:            jobType ("image_hq" default), prompt, negativePrompt, tier, characterId, seed,
                     referenceImageBase64
  image_hq:          aspectRatio, loraId
  video_wan_i2v:     model, width, height, frames, fps, steps, cfg
  video_animatediff: frames, fps, width, height, hires, hiresScale, upscaleTo, interp,
                     useFacedetailer, checkpoint, motionScale, faceidV2Weight
  workflow:          workflow, outputNodeId, inputImages {name: base64}, timeout
                     (none of the common fields: the graph was composed by the local
                     generate_character code and arrives finished - see training/cloud_workflow.py)

The workflow job is the odd one out on purpose. The other three compose their prompt here; a workflow
job's prompt is already inside the graph, so instead of composing it the worker VERIFIES it with
training/workflow_safety.py (node allowlist, every negative traced to the safety text) and lets its
own _submit_and_wait re-apply the cfg floor.
"""

import base64
import binascii
import copy
import glob
import io
import os
import random
import re
import tempfile

import comfyui_client as client
import generate_character as gc
import workflow_safety

JOB_TYPES = ("image_hq", "video_wan_i2v", "video_animatediff", "workflow")
WORKER_MAX_TIMEOUT_SECONDS = int(os.environ.get("WORKER_MAX_TIMEOUT", "1800"))
WORKFLOW_TIMEOUT_RANGE = (60, WORKER_MAX_TIMEOUT_SECONDS)
# cloud_workflow.stage_upload names: <16 hex>_<safe stem>.png|.jpg. Anything else is not ours.
_INPUT_IMAGE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}\.(png|jpe?g)$")

# aspectRatio -> final (width, height); submit_generation_hq derives the lower first-pass size.
ASPECT_SIZES = {
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (832, 1216),
    "16:9": (1216, 832),
}
MAX_EXTRA_NEGATIVE_CHARS = 2000
MAX_REFERENCE_BYTES = 8_000_000   # /run caps the whole payload at 10 MB; base64 inflates by 4/3
AD_UPSCALE_CHOICES = (0, 768, 1024, 1536)
AD_FRAME_RANGE = (8, 16)          # 16 is the motion module's trained context, not a VRAM limit
AD_BASE_SIDE_RANGE = (256, 768)   # SD1.5 base resolution; the hires pass and upscale go above it
# The local ANIMATEDIFF_HIRES_MAX_PIXELS (768^2) is an 8GB-card ceiling. A 24GB+ worker can sample
# the 16-frame hires batch at 1024^2; overridable per endpoint without rebuilding the image.
WORKER_AD_HIRES_MAX_PIXELS = int(os.environ.get("WORKER_AD_HIRES_MAX_PIXELS", str(1024 * 1024)))

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class JobInputError(gc.UsageError):
    """The caller's job input is invalid. Its message goes back to the caller verbatim."""


def _get_int(inp, name, default, lo, hi):
    value = inp.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise JobInputError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if not lo <= value <= hi:
        raise JobInputError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _get_float(inp, name, default, lo, hi):
    value = inp.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JobInputError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not lo <= value <= hi:
        raise JobInputError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _get_bool(inp, name, default):
    value = inp.get(name, default)
    if not isinstance(value, bool):
        raise JobInputError(f"{name} must be true or false, got {value!r}")
    return value


def parse_job(inp, *, worker_checkpoint):
    """Validate a job's input dict and return a normalised spec. Pure: decodes nothing, writes
    nothing, so a bad job is rejected before any temp file or GPU work exists."""
    if not isinstance(inp, dict):
        raise JobInputError("job input must be a JSON object")
    job_type = inp.get("jobType") or "image_hq"
    if job_type not in JOB_TYPES:
        raise JobInputError(f"unknown jobType {job_type!r}; choices: {list(JOB_TYPES)}")
    if job_type == "workflow":
        return _parse_workflow_job(inp)

    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        raise JobInputError("missing prompt")
    if len(prompt) > gc.MAX_PROMPT_CHARS:
        raise JobInputError(f"prompt too long (>{gc.MAX_PROMPT_CHARS} chars)")
    extra_negative = (inp.get("negativePrompt") or "").strip()
    if len(extra_negative) > MAX_EXTRA_NEGATIVE_CHARS:
        raise JobInputError(f"negativePrompt too long (>{MAX_EXTRA_NEGATIVE_CHARS} chars)")

    # Never trust an arbitrary tier - anything but the two known values falls back to the strictest.
    tier = inp.get("tier")
    if tier not in ("safe", "suggestive"):
        tier = "safe"

    character_id = inp.get("characterId") or None
    if character_id:
        gc.get_character(character_id)  # UsageError on unknown

    seed = inp.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        seed = random.randint(0, 2**31 - 1)

    spec = {
        "job_type": job_type, "prompt": prompt, "extra_negative": extra_negative, "tier": tier,
        "trigger": character_id, "character_id": character_id, "seed": seed,
        "reference_b64": inp.get("referenceImageBase64") or None,
    }

    if job_type == "image_hq":
        aspect = inp.get("aspectRatio") or "3:4"
        if aspect not in ASPECT_SIZES:
            raise JobInputError(f"unknown aspectRatio {aspect!r}; choices: {sorted(ASPECT_SIZES)}")
        if worker_checkpoint not in client.CHECKPOINTS:
            raise JobInputError(f"WORKER_CHECKPOINT {worker_checkpoint!r} not in CHECKPOINTS")
        spec.update(width=ASPECT_SIZES[aspect][0], height=ASPECT_SIZES[aspect][1],
                    checkpoint=worker_checkpoint, lora_id=inp.get("loraId") or None)

    elif job_type == "video_wan_i2v":
        model = inp.get("model") or client.WAN_DEFAULT_MODEL
        params = dict(
            model=model,
            width=_get_int(inp, "width", client.WAN_WIDTH, 1, 100000),
            height=_get_int(inp, "height", client.WAN_HEIGHT, 1, 100000),
            frames=_get_int(inp, "frames", client.WAN_FRAMES, 1, 100000),
            fps=_get_int(inp, "fps", client.WAN_FPS, 1, 1000),
            steps=_get_int(inp, "steps", client.WAN_STEPS, 1, 1000),
            cfg=_get_float(inp, "cfg", client.WAN_CFG, 0.0, 100.0),
        )
        try:
            gc.check_wan_params(**params)
        except gc.UsageError as exc:
            raise JobInputError(str(exc)) from exc
        spec.update(params)

    else:  # video_animatediff
        checkpoint = inp.get("checkpoint") or "realistic_vision"
        if checkpoint not in client.ANIMATEDIFF_CHECKPOINTS:
            raise JobInputError(f"unknown checkpoint {checkpoint!r}; choices: {sorted(client.ANIMATEDIFF_CHECKPOINTS)}")
        width = _get_int(inp, "width", client.ANIMATEDIFF_WIDTH, *AD_BASE_SIDE_RANGE)
        height = _get_int(inp, "height", client.ANIMATEDIFF_HEIGHT, *AD_BASE_SIDE_RANGE)
        if width % 8 or height % 8:
            raise JobInputError(f"width and height must be multiples of 8, got {width}x{height}")
        upscale_to = _get_int(inp, "upscaleTo", 1024, 0, 100000)
        if upscale_to not in AD_UPSCALE_CHOICES:
            raise JobInputError(f"upscaleTo must be one of {list(AD_UPSCALE_CHOICES)}, got {upscale_to}")
        interp = _get_int(inp, "interp", 1, 1, 4)
        if interp not in client.ANIMATEDIFF_INTERP_CHOICES:
            raise JobInputError(f"interp must be one of {list(client.ANIMATEDIFF_INTERP_CHOICES)}, got {interp}")
        spec.update(
            checkpoint=checkpoint, width=width, height=height,
            frames=_get_int(inp, "frames", client.ANIMATEDIFF_FRAMES, *AD_FRAME_RANGE),
            fps=_get_int(inp, "fps", 0, 0, 32),
            hires=_get_bool(inp, "hires", True),
            hires_scale=_get_float(inp, "hiresScale", client.ANIMATEDIFF_HIRES_SCALE, 1.0, 2.0),
            upscale_to=upscale_to, interp=interp,
            use_facedetailer=_get_bool(inp, "useFacedetailer", True),
            motion_scale=_get_float(inp, "motionScale", client.ANIMATEDIFF_MOTION_SCALE, 0.5, 1.2),
            faceid_v2_weight=_get_float(inp, "faceidV2Weight", client.ANIMATEDIFF_FACEID_V2_WEIGHT, 0.0, 3.0),
        )
    return spec


def _parse_workflow_job(inp):
    wf = inp.get("workflow")
    output_node_id = str(inp.get("outputNodeId") or "9")
    try:
        workflow_safety.validate_graph(wf, output_node_id)
        workflow_safety.check_negative_safety(wf, required=gc.AGE_SAFETY_NEGATIVE)
    except workflow_safety.WorkflowRejected as exc:
        raise JobInputError(f"workflow rejected: {exc}") from exc

    images = inp.get("inputImages") or {}
    if not isinstance(images, dict):
        raise JobInputError("inputImages must be an object of {filename: base64}")
    total_b64 = 0
    for name, b64 in images.items():
        if not isinstance(name, str) or not _INPUT_IMAGE_NAME.match(name):
            raise JobInputError(f"invalid input image name {name!r}")
        if not isinstance(b64, str):
            raise JobInputError(f"input image {name} must be a base64 string")
        total_b64 += len(b64)
    # Checked on the encoded length so an oversized job is refused before anything is decoded.
    if total_b64 * 3 // 4 > MAX_REFERENCE_BYTES:
        raise JobInputError(f"input images too large (>{MAX_REFERENCE_BYTES} bytes decoded)")
    # The worker has no pose library or anchors for these - every image the graph loads must arrive.
    missing = [n for n in workflow_safety.load_image_names(wf) if n not in images]
    if missing:
        raise JobInputError(f"workflow loads images that were not sent: {missing}")

    timeout = _get_int(inp, "timeout", client.POLL_TIMEOUT_SECONDS, 1, 10**7)
    timeout = max(WORKFLOW_TIMEOUT_RANGE[0], min(timeout, WORKFLOW_TIMEOUT_RANGE[1]))
    ext, content_type = workflow_safety.output_kind(wf, output_node_id)
    return {
        "job_type": "workflow", "workflow": wf, "output_node_id": output_node_id, "input_images": images,
        "timeout": timeout, "ext": ext, "content_type": content_type,
    }


def _run_workflow_job(spec, *, job_id, tmp_files):
    wf = copy.deepcopy(spec["workflow"])
    renamed = {}
    for name, b64 in spec["input_images"].items():
        path = decode_reference(b64, job_id, field=f"inputImages[{name}]")
        tmp_files.append(path)
        renamed[name] = client.upload_reference_image(path)
    for node in wf.values():
        if node["class_type"] == "LoadImage" and node["inputs"].get("image") in renamed:
            node["inputs"]["image"] = renamed[node["inputs"]["image"]]
    # This _submit_and_wait is the worker's own: model variants, the cfg floor and the /free resets
    # all run here, next to the GPU they are about.
    path = client._submit_and_wait(wf, spec["output_node_id"], spec["timeout"])
    tmp_files.append(path)
    return {"jobType": "workflow", "localPath": path, "contentType": spec["content_type"], "ext": spec["ext"]}


def decode_reference(b64, job_id, *, max_bytes=MAX_REFERENCE_BYTES, field="referenceImageBase64"):
    """Decode a base64 reference image, verify it really is a PNG/JPEG (never trust a declared content
    type), and write it to a tempfile. Returns the path, or raises JobInputError.

    Magic bytes are checked first so the rejection works without PIL; PIL's structural verify runs
    on top of that when available (it always is inside the worker image)."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise JobInputError(f"{field} is not valid base64: {exc}") from exc
    if len(raw) > max_bytes:
        raise JobInputError(f"reference image too large ({len(raw)} bytes > {max_bytes})")
    if raw.startswith(_PNG_MAGIC):
        suffix = ".png"
    elif raw.startswith(_JPEG_MAGIC):
        suffix = ".jpg"
    else:
        raise JobInputError("reference image must be PNG or JPEG")
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        try:
            Image.open(io.BytesIO(raw)).verify()
        except Exception as exc:
            raise JobInputError(f"{field} is not a decodable image: {exc}") from exc
    fd, path = tempfile.mkstemp(prefix=f"ref_{job_id}_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def anchor_for_character(character_id, anchor_dir):
    """Fallback reference from the Network Volume: anchors/<characterId>/anchor_*.png (mirrors the
    local reference_candidates layout)."""
    if not character_id:
        return None
    matches = sorted(glob.glob(os.path.join(anchor_dir, character_id, "anchor_*.png")))
    if not matches:
        matches = sorted(glob.glob(os.path.join(anchor_dir, character_id, "*.png")))
    return matches[0] if matches else None


def run_job(spec, *, job_id, anchor_dir, out_dir, tmp_files):
    """Generate what `spec` describes. Temp files it creates are appended to tmp_files (the caller
    deletes them). Returns {"localPath", "contentType", "ext", "seed", "width", "height", ...}."""
    if spec["job_type"] == "workflow":
        return _run_workflow_job(spec, job_id=job_id, tmp_files=tmp_files)
    if spec["reference_b64"]:
        ref_path = decode_reference(spec["reference_b64"], job_id)
        tmp_files.append(ref_path)
    else:
        ref_path = anchor_for_character(spec["character_id"], anchor_dir)
    if not ref_path:
        kind = "first frame" if spec["job_type"] == "video_wan_i2v" else "reference face"
        raise JobInputError(f"no {kind}: provide referenceImageBase64 or a characterId with an anchor on the volume")

    seed = spec["seed"]
    result = {"seed": seed, "jobType": spec["job_type"]}

    if spec["job_type"] == "image_hq":
        checkpoint_key = spec["checkpoint"]
        # Same composition point as CLI/GUI: tier ceiling, AGE_SAFETY_NEGATIVE, Pony quality tags.
        full_prompt, negative_prompt = gc._build_prompt_and_negative(
            spec["prompt"], spec["extra_negative"], spec["tier"], spec["trigger"], None, None, checkpoint_key)
        lora_id = spec["lora_id"]
        character_lora = f"characters/{lora_id}.safetensors" if lora_id else None
        # A trained LoRA carries identity, so FaceID only corrects drift.
        ip_weight = 0.7 if character_lora else client.IP_ADAPTER_WEIGHT
        path = client.submit_generation_hq(
            prompt=full_prompt, negative_prompt=negative_prompt, seed=seed, filename_prefix=f"rp_{job_id}",
            ip_adapter_image_filename=client.upload_reference_image(ref_path),
            width=spec["width"], height=spec["height"],
            checkpoint=client.CHECKPOINTS[checkpoint_key],
            lora_strength=0.0 if checkpoint_key in client.PONY_CHECKPOINTS else None,
            character_lora=character_lora, ip_adapter_weight=ip_weight, hires=True, use_facedetailer=True,
        )
        tmp_files.append(path)
        result.update(localPath=path, contentType="image/png", ext="png",
                      width=spec["width"], height=spec["height"])

    elif spec["job_type"] == "video_wan_i2v":
        path = gc.gen_video_wan_i2v(
            spec["prompt"], spec["extra_negative"], spec["tier"], spec["trigger"], ref_path, out_dir, seed,
            model=spec["model"], width=spec["width"], height=spec["height"], frames=spec["frames"],
            fps=spec["fps"], steps=spec["steps"], cfg=spec["cfg"],
        )
        tmp_files.append(path)
        result.update(localPath=path, contentType="video/mp4", ext="mp4", width=spec["width"],
                      height=spec["height"], frames=spec["frames"], fps=spec["fps"])

    else:  # video_animatediff
        path = gc.gen_video_animatediff(
            spec["prompt"], spec["extra_negative"], spec["tier"], spec["trigger"], ref_path, out_dir, seed,
            frames=spec["frames"], fps=spec["fps"] or None, width=spec["width"], height=spec["height"],
            checkpoint=spec["checkpoint"], hires=spec["hires"], hires_scale=spec["hires_scale"],
            upscale_to=spec["upscale_to"], interp=spec["interp"], use_facedetailer=spec["use_facedetailer"],
            motion_scale=spec["motion_scale"], faceid_v2_weight=spec["faceid_v2_weight"],
            hires_max_pixels=WORKER_AD_HIRES_MAX_PIXELS,
        )
        tmp_files.append(path)
        result.update(localPath=path, contentType="video/mp4", ext="mp4", width=spec["width"],
                      height=spec["height"], frames=spec["frames"], upscaleTo=spec["upscale_to"])
    return result
