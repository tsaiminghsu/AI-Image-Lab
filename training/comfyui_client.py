"""Thin HTTP client for a persistently-running ComfyUI server.

Model lifecycle (checkpoint/CLIP/VAE/IP-Adapter loading, VRAM management,
caching between prompts) is entirely ComfyUI's responsibility. This module
never imports torch or touches CUDA directly - it only builds workflow JSON,
submits it over HTTP, polls for completion, and downloads the result.

Start the server once, separately, before using this client:
    D:\\AI-Image-Lab\\ComfyUI\\.venv\\Scripts\\python.exe D:\\AI-Image-Lab\\ComfyUI\\main.py --listen 127.0.0.1 --port 8188
"""

import copy
import json
import os
import subprocess
import time

import requests

# Overridable so the exact same client code runs unchanged inside a RunPod
# Serverless worker (ComfyUI on localhost there too) or against a RunPod Pod
# (https://<POD_ID>-8188.proxy.runpod.net) - see CLOUD_GPU.md. _submit_and_wait/
# _download_output read this module global at call time, so nothing else needs
# to change.
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_DIR = os.path.join(os.path.dirname(__file__), "..", "ComfyUI")
WORKFLOW_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "workflow_template.json")
WORKFLOW_TEMPLATE_HQ_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_hq.json")
WORKFLOW_TEMPLATE_TXT2IMG_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_txt2img.json")
WORKFLOW_TEMPLATE_TXT2IMG_SD15_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_txt2img_sd15.json")
WORKFLOW_TEMPLATE_IMG2VID_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_img2vid.json")
WORKFLOW_TEMPLATE_CONTROLNET_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_controlnet.json")
WORKFLOW_TEMPLATE_FACEDETAILER_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_facedetailer.json")
WORKFLOW_TEMPLATE_MEDIAPIPE_FACEDETAILER_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_mediapipe_facedetailer.json")
WORKFLOW_TEMPLATE_ANIMATEDIFF_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_animatediff_facedetailer.json")
WORKFLOW_TEMPLATE_IMG2IMG_PATH = os.path.join(os.path.dirname(__file__), "workflow_template_img2img.json")

CHECKPOINTS = {
    "juggernaut": "juggernaut_xl_v9_photo.safetensors",
    "pony": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",  # same SDXL UNet/CLIP shape as
    # juggernaut, so it drops into the exact same workflow templates (IP-Adapter FaceID, ControlNet,
    # FaceDetailer all reuse the SDXL-family files already installed). Trained on booru tags, not
    # natural language - benefits from a "score_9, score_8_up, score_7_up" quality-tag prefix, which
    # generate_character.py's gen_custom() auto-prepends when this checkpoint is picked (see
    # PONY_CHECKPOINTS below) - callers never need to type the tag themselves, prompt input stays
    # plain natural language regardless of checkpoint choice.
    # sdxl_photorealistic_slider_v1-0.safetensors (baked into every template at strength 2.5) was
    # tuned against photoreal SDXL checkpoints, not Pony's more illustration-leaning training data -
    # pass lora_strength=0.0 when using pony to leave it out rather than fighting its style.
    "cyberrealistic_pony": "CyberRealisticPony_V18.0_F16.safetensors",  # also Pony/SDXL-architecture
    # (same drop-in compatibility as "pony" above) - Cyberdelia's photoreal-tuned merge onto the Pony
    # base, meant to keep Pony's prompt/tag conventions while looking less illustration/anime-leaning
    # than vanilla Pony Diffusion. Same score_9/score_8_up prompt convention applies (auto-prepended,
    # see PONY_CHECKPOINTS).
    "realistic_vision": "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",  # SD1.5, not SDXL - different
    # UNet/CLIP shape than every checkpoint above, so it does NOT drop into the SDXL workflow templates
    # (the baked-in sdxl_photorealistic_slider LoRA and IP-Adapter/ControlNet SDXL model files don't
    # match its architecture). Only plain txt2img is wired up so far (WORKFLOW_TEMPLATE_TXT2IMG_SD15_PATH,
    # no LoRA node) - see SD15_CHECKPOINTS and submit_txt2img_generation_sd15. Natural-language prompts,
    # not Pony's booru tags. Native training resolution is ~512x512-768 range, not SDXL's 1024.
    "cyberrealistic": "CyberRealistic_FINAL_FP16.safetensors",  # SD1.5, same caveats as realistic_vision above
    "pony_realism": "ponyRealism_V22.safetensors",  # also Pony/SDXL-architecture (same drop-in
    # compatibility as "pony"/"cyberrealistic_pony") - ZyloO's photoreal-tuned Pony checkpoint, VAE
    # baked in. Same score_9/score_8_up prompt convention applies (auto-prepended, see PONY_CHECKPOINTS).
}
CHECKPOINT = CHECKPOINTS["juggernaut"]
SD15_CHECKPOINTS = {"realistic_vision", "cyberrealistic"}  # keys into CHECKPOINTS that are SD1.5,
# not SDXL/Pony - callers use this to route to submit_txt2img_generation_sd15 and pick SD15-appropriate
# resolution instead of the SDXL-family submit_* functions (which all assume an SDXL-shaped checkpoint)
PONY_CHECKPOINTS = {"pony", "cyberrealistic_pony", "pony_realism"}  # keys into CHECKPOINTS that are
# Pony-Diffusion-architecture (SDXL-shaped, drop into the same workflow templates as juggernaut, but
# trained on a "score_9, score_8_up, score_7_up"-style quality-tag prefix convention rather than plain
# natural language) - generate_character.py's gen_custom() uses this set to auto-prepend that prefix,
# so every caller (GUI/API/CLI) can keep writing plain natural-language prompts no matter which
# checkpoint is selected, instead of needing to know/type the Pony-specific tag convention themselves.
SD15_WIDTH, SD15_HEIGHT = 512, 768
WIDTH, HEIGHT = 1024, 1024   # switch to 768/832/896 as needed - not hardcoded elsewhere
BATCH_SIZE = 1                # RTX 2070 8GB - always 1, no multi-image batches
STEPS = 30
CFG = 6.0
SAMPLER = "dpmpp_2m"
SCHEDULER = "karras"
IP_ADAPTER_PRESET = "FACEID PLUS V2"  # InsightFace face-recognition embedding, not CLIP-vision
# similarity - "PLUS FACE" conditions on general visual resemblance and drifted across generations
# (different seeds could look like different people despite the same anchor). FaceID locks onto the
# actual face identity, which is what a LoRA training dataset needs: one consistent character across
# every image. Reuses the same ViT-H clip vision encoder as PLUS FACE did (still needed - FaceID Plus
# V2 combines the InsightFace ID embedding with CLIP-vision features, it doesn't replace it). Needs
# ip-adapter-faceid-plusv2_sdxl.bin + ip-adapter-faceid-plusv2_sdxl_lora.safetensors + the
# `insightface` package + its buffalo_l model (auto-downloads to ComfyUI/models/insightface/ on
# first use, same lazy-download pattern as the ControlNet OpenPose detector weights).
IP_ADAPTER_WEIGHT = 1.0  # FaceID's own weight scale (-1 to 3) is different from PLUS FACE's (0-1.5ish)

# Control-LoRA (not a full ControlNet checkpoint) - the full OpenPoseXL2
# ControlNet is ~5GB, which alongside the 6.6GB Juggernaut checkpoint + 848MB
# IP-Adapter + 2.5GB CLIP vision would blow the 8GB VRAM budget. The
# control-lora variant is a rank-decomposed adapter at ~774MB instead, chosen
# specifically to fit this card. Needs the comfyui_controlnet_aux custom node
# for the OpenposePreprocessor that turns a reference photo into a skeleton.
CONTROLNET_MODEL = "control-lora-openposeXL2-rank256.safetensors"
CONTROLNET_STRENGTH = 0.8  # 0=ignored, 1=rigidly locked to the skeleton; 0.8 leaves some room for the prompt

# ADetailer-equivalent post-process (ComfyUI-Impact-Pack's FaceDetailer node +
# ComfyUI-Impact-Subpack's UltralyticsDetectorProvider) - detects the face
# (and separately the hands), crops just that region, and re-samples it at
# guide_size/max_size with a low denoise as an inpaint pass. Fixes the small
# SDXL face/hand artifacts (warped fingers, slightly mismatched eyes) that
# survive at the full 1024-ish canvas resolution because that region is only
# a small fraction of the total pixels the base KSampler pass spent its steps
# on. Models are the small YOLOv8 "nano" variants (~6-7MB each, negligible
# VRAM/time cost compared to the checkpoint/FaceID/ControlNet stack) - see
# check_adetailer_models.py for the one-time download.
FACEDETAILER_DENOISE = 0.5  # 0=no change, 1=fully re-generate the cropped region from noise

# --- HQ two-pass path (workflow_template_hq.json, submit_generation_hq) ---
# The single unified high-quality path: style LoRA -> optional character LoRA ->
# FaceID -> optional ControlNet -> base KSampler at a LOWER first-pass resolution
# -> ESRGAN model upscale + downscale to the target -> second KSampler at low
# denoise (hires fix) -> FaceDetailer face pass -> FaceDetailer hand pass. Replaces
# the three separate FaceID / FaceDetailer / ControlNet templates for the main
# path by bypassing (rewiring + dropping) whichever optional nodes aren't needed.
UPSCALE_MODEL = "4x-UltraSharp.pth"  # 4x ESRGAN, ~67MB - real pixel detail so the
# hires pass only needs denoise ~0.4 (vs latent upscale's >=0.55 where FaceID
# identity drifts and hands get re-invented). Chosen over RealESRGAN_x4plus, which
# over-smooths skin - the "plastic"/3D-render complaint. Put the .pth in
# models/upscale/ and NTFS-hardlink into ComfyUI/models/upscale_models/ (README).
UPSCALE_MODEL_SCALE = 4  # the model's native scale factor (4x-UltraSharp is 4x)
HIRES_SCALE = 1.5        # final size / first-pass size. 1.5x keeps the 8GB card
# out of the 1024->1536^2 (2.4MP) danger zone while still restoring real detail.
HIRES_DENOISE = 0.4
HIRES_STEPS = 20
HQ_BASE_STEPS = 24
# Explicit first-pass sizes for the three canonical output sizes (all /64 for the
# UNet and /8 for the VAE). Anything else falls back to round64(dim / HIRES_SCALE).
HQ_FIRST_PASS = {
    (1024, 1024): (832, 832),    # square   -> 1248x1248
    (832, 1216): (704, 1024),    # portrait -> 1056x1536 (full-body / back view)
    (1216, 832): (1024, 704),    # landscape-> 1536x1056
}
FACEDETAILER_FACE_DENOISE = 0.4
FACEDETAILER_HAND_DENOISE = 0.35
FACEDETAILER_FACE_GUIDE_SIZE = 768
FACEDETAILER_HAND_GUIDE_SIZE = 512
CHARACTER_LORA_STRENGTH = 0.8  # per-character LoRA (node 14), trained on RunPod;
# used TOGETHER with FaceID (which drops to faceid_weight ~0.7 when a LoRA is present)
INSIGHTFACE_PROVIDER = os.environ.get("INSIGHTFACE_PROVIDER", "CPU")  # "CPU" or "CUDA".
# FaceID's insightface runs an onnxruntime session whose ~1GB VRAM lives OUTSIDE
# torch's allocator, so ComfyUI's VRAM accounting is wrong and it thrashes (the
# reason FaceID cost 150-180s vs 36s plain). Detection runs once per anchor (~1-2s)
# and its output is cached across seeds, so CPU barely costs anything and stops the
# thrash. Override to CUDA to A/B the difference (see benchmark.py).

# SVD (Stable Video Diffusion) img2vid - low-res/low-frame defaults, 8GB VRAM
# can't comfortably run the model's native 1024x576 training resolution
# alongside everything else, and "low quality is fine" was the explicit ask.
SVD_CHECKPOINT = "svd.safetensors"
# Square, not 16:9 - anchor/dataset images are 1024x1024 portraits, and SVD's
# center-crop would otherwise chop off the top/bottom of the face to fit a
# wide frame. motion_bucket_id defaults to SVD's own 127, which is tuned for
# landscape/camera-motion shots; on a static portrait it drives the classic
# face-melting artifact, so keep it low for subtle, coherent motion instead.
# Half the imported image's native 1024x1024 - this was left at full 1024
# despite the "low-res is fine" intent above, which made SVD_img2vid_Conditioning
# resize/encode at full resolution and made an already-slow ~9.5GB checkpoint
# swap even slower for what's meant to be a quick "does it move" test, not a
# final-quality render.
VIDEO_WIDTH, VIDEO_HEIGHT = 512, 512
VIDEO_FRAMES = 14        # svd.safetensors (base) is trained for 14 frames; svd_xt for 25
VIDEO_FPS = 6
MOTION_BUCKET_ID = 15    # higher = more motion but more warping/distortion on portraits
VIDEO_STEPS = 30
VIDEO_CFG = 2.5
VIDEO_MIN_CFG = 1.0
VIDEO_SAMPLER = "euler"
VIDEO_SCHEDULER = "karras"
VIDEO_CRF = 40           # vp9 crf - higher = lower quality/smaller file

# AnimateDiff (SD1.5) + IPAdapter-FaceID + per-frame FaceDetailer - fixes the
# face warping/melting that plain SVD img2vid produces (see MOTION_BUCKET_ID's
# comment above). SVD is a separate temporal-U-Net architecture that ComfyUI's
# IPAdapter nodes can't patch into, so identity-locking isn't possible there;
# AnimateDiff instead injects a motion module into a normal SD1.5 UNet, so the
# existing IPAdapterUnifiedLoaderFaceID/IPAdapterFaceID nodes work on it
# unmodified (they auto-detect SD1.5 vs SDXL from the model and pick the
# matching *_sd15.bin/*_sd15_lora.safetensors files - same CLIP-ViT-H vision
# encoder and buffalo_l InsightFace model as the SDXL FaceID path already
# uses, no separate copies needed). The FaceDetailer pass afterward re-samples
# just the face crop from every frame using this same AnimateDiff-patched
# model, so the crops are denoised together as a short temporal sequence
# (not independently per frame) - that's what keeps the touched-up face
# consistent frame to frame instead of flickering.
ANIMATEDIFF_CHECKPOINT = "v1-5-pruned-emaonly.safetensors"  # official SD1.5 base - AnimateDiff's
# mature/well-supported motion modules are SD1.5-only (the SDXL ones are beta and much heavier
# on VRAM); realism comes from prompt/negative tuning same as the SDXL path, not the checkpoint
ANIMATEDIFF_CHECKPOINTS = {
    "sd15_base": ANIMATEDIFF_CHECKPOINT,
    "realistic_vision": CHECKPOINTS["realistic_vision"],
    "cyberrealistic": CHECKPOINTS["cyberrealistic"],
}  # any SD1.5 checkpoint works here (same UNet shape the motion module patches into) - deliberately
# NOT sharing CHECKPOINTS/SD15_CHECKPOINTS directly since those also list SDXL/Pony entries that would
# crash AnimateDiff (motion module tensor shapes don't match an SDXL UNet)
ANIMATEDIFF_MOTION_MODULE = "mm_sd_v15_v2.ckpt"

# Motion LoRAs (official guoyww/animatediff release) - these condition the
# motion module for a specific CAMERA movement (zoom/pan/tilt/roll of the
# whole frame), not a body-part-specific physical effect (e.g. no "bounce"/
# "jiggle" control - the motion module has no dedicated mechanism for that,
# camera-motion LoRAs are the only officially-supported motion LoRA category
# for this v2 motion module). Only compatible with v2-based motion modules
# (mm_sd_v15_v2 - what ANIMATEDIFF_MOTION_MODULE already is - and mm-p_0.5/
# mm-p_0.75), per the ComfyUI-AnimateDiff-Evolved project's own compatibility
# notes. Download to ComfyUI/models/animatediff_motion_lora/, see README's
# "AnimateDiff Motion LoRA" install section. Optional - submit_generation_
# animatediff() only wires the loader node in when motion_lora_name is given.
ANIMATEDIFF_MOTION_LORAS = {
    "zoom_in": "v2_lora_ZoomIn.ckpt",
    "zoom_out": "v2_lora_ZoomOut.ckpt",
    "pan_left": "v2_lora_PanLeft.ckpt",
    "pan_right": "v2_lora_PanRight.ckpt",
    "tilt_up": "v2_lora_TiltUp.ckpt",
    "tilt_down": "v2_lora_TiltDown.ckpt",
    "rolling_clockwise": "v2_lora_RollingClockwise.ckpt",
    "rolling_anticlockwise": "v2_lora_RollingAnticlockwise.ckpt",
}
ANIMATEDIFF_WIDTH, ANIMATEDIFF_HEIGHT = 512, 512
ANIMATEDIFF_FRAMES = 16   # mm_sd_v15_v2's trained context length. Tried going over this via a
# sliding-context-window node (ADE_StandardUniformContextOptions) to get more time for an action
# to visibly progress - both context_overlap=4 and =8 produced worse artifacts than staying at 16
# (overlap=4: a visible clothing/appearance jump at the window boundary around frame 16-17;
# overlap=8: much worse - progressive zoom/composition drift and background ghosting across the
# clip). Reverted; if more perceived motion is needed, retime the finished 16-frame clip to a
# lower fps instead (e.g. `ffmpeg -filter:v "setpts=2.0*PTS" -r 4` for a free 4s clip from the
# same 16 frames) rather than generating more frames.
ANIMATEDIFF_FPS = 8
ANIMATEDIFF_STEPS = 20
ANIMATEDIFF_CFG = 7.5      # SD1.5's usual cfg range, higher than SDXL's 6.0
ANIMATEDIFF_SAMPLER = "dpmpp_2m"
ANIMATEDIFF_SCHEDULER = "karras"
ANIMATEDIFF_CRF = 32
POLL_TIMEOUT_SECONDS_ANIMATEDIFF = 1800  # 16 frames x (base KSampler + the batched video-detailer
# pass) on an 8GB card is slower than a single still image; first run also lazy-loads a new
# checkpoint/motion-module/FaceID-SD1.5 stack ComfyUI hasn't cached yet

POLL_INTERVAL_SECONDS = 2
# 900s, not the more typical 300s - FaceID (now the default IP_ADAPTER_PRESET
# for every generation mode) lazy-downloads the buffalo_l InsightFace model
# (~280MB) on its very first use ever, inside the ComfyUI process; the
# OpenposePreprocessor does the same for its own ~200MB of pose weights on
# its first use. Both only matter once - the weights get cached locally
# (ComfyUI/models/insightface/ and .../comfyui_controlnet_aux/ckpts/) and
# every call after that is as fast as a normal generation - but a client-side
# timeout that's too tight abandons a job that's still succeeding server-side
# (measured: OpenPose's first run alone took ~345s on a ~300KB/s link).
POLL_TIMEOUT_SECONDS = 900
POLL_TIMEOUT_SECONDS_VIDEO = 1800  # svd.safetensors is ~9.5GB - first load / swap from another
# checkpoint (SDXL or, worse, the AnimateDiff SD1.5+motion-module stack) already loaded in VRAM
# is slow, and can be much slower still if VRAM is fragmented from a prior session's models -
# measured one run taking >900s this way even with VIDEO_WIDTH/HEIGHT halved to 512


def _load_template(path=WORKFLOW_TEMPLATE_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_server_running(timeout: float = 2) -> bool:
    """Cheap reachability check - lets gui.py start ComfyUI on demand instead
    of requiring it pre-started for every session (idle ComfyUI otherwise
    sits at ~2GB RAM/VRAM doing nothing)."""
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False


def start_server(startup_timeout: float = 120) -> subprocess.Popen:
    """Launch the ComfyUI server as a background process and block until it
    responds. Assumes this repo's local layout (COMFYUI_DIR) - not used by
    the cloud worker, which manages its own already-running ComfyUI process
    via COMFYUI_URL instead."""
    python_exe = os.path.join(COMFYUI_DIR, ".venv", "Scripts", "python.exe")
    process = subprocess.Popen(
        [python_exe, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        cwd=COMFYUI_DIR,
    )
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if is_server_running():
            return process
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI process exited early (code {process.returncode})")
        time.sleep(1)
    raise TimeoutError(f"ComfyUI server didn't come up within {startup_timeout}s")


def log_gpu_memory(stage: str):
    r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=10)
    r.raise_for_status()
    device = r.json()["devices"][0]
    total = device["vram_total"] / (1024 ** 2)
    free = device["vram_free"] / (1024 ** 2)
    used = total - free
    print(f"[MEMORY] {stage}: VRAM used {used:.0f} MB / {total:.0f} MB (free {free:.0f} MB)", flush=True)
    return used, total


def upload_reference_image(local_path: str) -> str:
    """Upload a face-reference image into ComfyUI's input/ dir. Returns the
    filename to use as the LoadImage node's `image` input."""
    filename = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        r = requests.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (filename, f)},
            data={"overwrite": "true"},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()["name"]


def submit_generation(
    prompt: str,
    negative_prompt: str,
    seed: int,
    ip_adapter_image_filename: str,
    filename_prefix: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = STEPS,
    cfg: float = CFG,
    ip_adapter_weight: float = IP_ADAPTER_WEIGHT,
    checkpoint: str = CHECKPOINT,
    lora_strength: float = None,
) -> str:
    """Submit one generation job, block until done, return the local path of
    the saved output image. Does not import torch / touch CUDA - ComfyUI
    owns the model and VRAM."""
    wf = copy.deepcopy(_load_template())
    wf["4"]["inputs"]["ckpt_name"] = checkpoint
    if lora_strength is not None:
        wf["13"]["inputs"]["strength_model"] = lora_strength
        wf["13"]["inputs"]["strength_clip"] = lora_strength
    wf["10"]["inputs"]["image"] = ip_adapter_image_filename
    wf["11"]["inputs"]["preset"] = IP_ADAPTER_PRESET
    wf["12"]["inputs"]["weight"] = ip_adapter_weight
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["batch_size"] = BATCH_SIZE
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf)


def submit_img2img_generation(
    prompt: str,
    negative_prompt: str,
    seed: int,
    ip_adapter_image_filename: str,
    init_image_filename: str,
    denoise: float,
    filename_prefix: str,
    steps: int = STEPS,
    cfg: float = CFG,
    ip_adapter_weight: float = IP_ADAPTER_WEIGHT,
    checkpoint: str = CHECKPOINT,
    lora_strength: float = None,
) -> str:
    """img2img on an existing image (init_image_filename), with IP-Adapter
    FaceID conditioning same as submit_generation. Used by generate_character.
    gen_gif()'s "wiggle" frames: starting from the same init image every call
    with a low denoise keeps composition/pose locked to that image while a
    different seed each call still introduces small per-frame variation -
    unlike a from-scratch txt2img call (submit_generation), which resamples
    the whole composition independently every time with no shared structure
    between calls at all. width/height aren't parameters here - VAEEncode
    just encodes whatever size init_image_filename already is, output stays
    that size. Both image filenames must already be uploaded (see
    upload_reference_image) - can be the same file (the init image's own
    face doubling as the FaceID reference) or different ones."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_IMG2IMG_PATH))
    wf["4"]["inputs"]["ckpt_name"] = checkpoint
    if lora_strength is not None:
        wf["13"]["inputs"]["strength_model"] = lora_strength
        wf["13"]["inputs"]["strength_clip"] = lora_strength
    wf["10"]["inputs"]["image"] = ip_adapter_image_filename
    wf["11"]["inputs"]["preset"] = IP_ADAPTER_PRESET
    wf["12"]["inputs"]["weight"] = ip_adapter_weight
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["20"]["inputs"]["image"] = init_image_filename
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER
    wf["3"]["inputs"]["denoise"] = denoise
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf)


def submit_generation_with_pose(
    prompt: str,
    negative_prompt: str,
    seed: int,
    ip_adapter_image_filename: str,
    pose_image_filename: str,
    filename_prefix: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = STEPS,
    cfg: float = CFG,
    ip_adapter_weight: float = IP_ADAPTER_WEIGHT,
    controlnet_strength: float = CONTROLNET_STRENGTH,
    checkpoint: str = CHECKPOINT,
    lora_strength: float = None,
) -> str:
    """Same as submit_generation, but additionally skeleton-conditions the
    pose via ControlNet: pose_image_filename is auto-converted to an OpenPose
    skeleton by the OpenposePreprocessor node, then that skeleton constrains
    KSampler's positive/negative conditioning alongside the IP-Adapter face
    conditioning. Both pose_image_filename and ip_adapter_image_filename must
    already be uploaded (see upload_reference_image) - they can be the same
    file or two different ones (e.g. a fictional anchor for the face, a
    separate reference photo purely for its pose)."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_CONTROLNET_PATH))
    wf["4"]["inputs"]["ckpt_name"] = checkpoint
    if lora_strength is not None:
        wf["13"]["inputs"]["strength_model"] = lora_strength
        wf["13"]["inputs"]["strength_clip"] = lora_strength
    wf["10"]["inputs"]["image"] = ip_adapter_image_filename
    wf["11"]["inputs"]["preset"] = IP_ADAPTER_PRESET
    wf["12"]["inputs"]["weight"] = ip_adapter_weight
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["batch_size"] = BATCH_SIZE
    wf["20"]["inputs"]["image"] = pose_image_filename
    wf["22"]["inputs"]["control_net_name"] = CONTROLNET_MODEL
    wf["23"]["inputs"]["strength"] = controlnet_strength
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf)


def submit_generation_with_facedetailer(
    prompt: str,
    negative_prompt: str,
    seed: int,
    ip_adapter_image_filename: str,
    filename_prefix: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = STEPS,
    cfg: float = CFG,
    ip_adapter_weight: float = IP_ADAPTER_WEIGHT,
    facedetailer_denoise: float = FACEDETAILER_DENOISE,
    checkpoint: str = CHECKPOINT,
    lora_strength: float = None,
) -> str:
    """Same as submit_generation, but runs two ADetailer-style refinement
    passes on the output afterward: detect the face (YOLOv8 bbox) -> crop ->
    re-sample at higher effective resolution -> paste back, then the same
    again for hands. Fixes small face/hand artifacts without needing a
    ControlNet pose reference or a bigger base resolution. See
    check_adetailer_models.py for the one-time model download this depends
    on (face_yolov8n.pt, hand_yolov8n.pt - yolov8n-seg.pt is downloaded
    alongside for segmentation-based masking but isn't wired into this
    default bbox-based chain)."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_FACEDETAILER_PATH))
    wf["4"]["inputs"]["ckpt_name"] = checkpoint
    if lora_strength is not None:
        wf["13"]["inputs"]["strength_model"] = lora_strength
        wf["13"]["inputs"]["strength_clip"] = lora_strength
    wf["10"]["inputs"]["image"] = ip_adapter_image_filename
    wf["11"]["inputs"]["preset"] = IP_ADAPTER_PRESET
    wf["12"]["inputs"]["weight"] = ip_adapter_weight
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["batch_size"] = BATCH_SIZE
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER
    for node_id in ("41", "43"):
        wf[node_id]["inputs"]["seed"] = seed
        wf[node_id]["inputs"]["cfg"] = cfg
        wf[node_id]["inputs"]["sampler_name"] = SAMPLER
        wf[node_id]["inputs"]["scheduler"] = SCHEDULER
        wf[node_id]["inputs"]["denoise"] = facedetailer_denoise
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf)


def submit_generation_with_facedetailer_mediapipe(
    prompt: str,
    negative_prompt: str,
    seed: int,
    ip_adapter_image_filename: str,
    filename_prefix: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = STEPS,
    cfg: float = CFG,
    ip_adapter_weight: float = IP_ADAPTER_WEIGHT,
    facedetailer_denoise: float = FACEDETAILER_DENOISE,
    checkpoint: str = CHECKPOINT,
    lora_strength: float = None,
) -> str:
    """Same as submit_generation_with_facedetailer, but the face pass uses
    MediaPipeFaceMeshToSEGS + DetailerForEach instead of the YOLOv8 bbox
    detector: the inpaint mask follows the actual face-mesh contour (a
    rounded face-shaped region) rather than a rectangular bounding box, so
    less non-face background/hair gets pulled into the re-sampled region -
    fewer blend-boundary artifacts around the jawline/hairline than the bbox
    version can produce. Hand pass is unchanged (still YOLOv8 bbox - there's
    no equivalent mediapipe hand-mesh node installed). Needs the `mediapipe`
    package (see README's ADetailer install section)."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_MEDIAPIPE_FACEDETAILER_PATH))
    wf["4"]["inputs"]["ckpt_name"] = checkpoint
    if lora_strength is not None:
        wf["13"]["inputs"]["strength_model"] = lora_strength
        wf["13"]["inputs"]["strength_clip"] = lora_strength
    wf["10"]["inputs"]["image"] = ip_adapter_image_filename
    wf["11"]["inputs"]["preset"] = IP_ADAPTER_PRESET
    wf["12"]["inputs"]["weight"] = ip_adapter_weight
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["batch_size"] = BATCH_SIZE
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER
    for node_id in ("41", "43"):
        wf[node_id]["inputs"]["seed"] = seed
        wf[node_id]["inputs"]["cfg"] = cfg
        wf[node_id]["inputs"]["sampler_name"] = SAMPLER
        wf[node_id]["inputs"]["scheduler"] = SCHEDULER
        wf[node_id]["inputs"]["denoise"] = facedetailer_denoise
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf)


def submit_generation_animatediff(
    prompt: str,
    negative_prompt: str,
    seed: int,
    face_ref_image_filename: str,
    filename_prefix: str,
    width: int = ANIMATEDIFF_WIDTH,
    height: int = ANIMATEDIFF_HEIGHT,
    frames: int = ANIMATEDIFF_FRAMES,
    fps: int = ANIMATEDIFF_FPS,
    steps: int = ANIMATEDIFF_STEPS,
    cfg: float = ANIMATEDIFF_CFG,
    ip_adapter_weight: float = IP_ADAPTER_WEIGHT,
    facedetailer_denoise: float = FACEDETAILER_DENOISE,
    checkpoint: str = ANIMATEDIFF_CHECKPOINT,
    motion_lora_name: str = None,
    motion_lora_strength: float = 1.0,
) -> str:
    """AnimateDiff (SD1.5) txt2vid with IPAdapter-FaceID identity locking and
    a video-native FaceDetailer face-fix pass (Impact Pack's "Detailer For
    Video" - detects the face across all frames at once via
    ImpactSimpleDetectorSEGS_for_AD, then re-samples that whole face-crop
    batch together through the same AnimateDiff-patched model, node 12). This
    keeps the face-fix pass temporally coherent (no per-frame flicker) - the
    earlier version ran a plain per-frame FaceDetailer on a *separate*
    non-AnimateDiff FaceID branch (worked around a batch=1-into-motion-module
    corruption bug, see git history) but each frame's face was redrawn
    independently, causing visible flicker between frames. face_ref_image_filename
    must already be uploaded (see upload_reference_image) - same anchor image
    used for still-image FaceID generation works here. Returns the local path
    of the saved .webm.

    motion_lora_name (optional) is a filename from ANIMATEDIFF_MOTION_LORAS
    (e.g. "v2_lora_ZoomIn.ckpt") - conditions the motion module for a specific
    CAMERA movement, not a body-part physical effect. The loader node
    (ADE_AnimateDiffLoRALoader) is only added to the workflow when this is
    given - the template has no baked-in placeholder for it, since it's an
    optional input on node "2" (ADE_AnimateDiffLoaderGen1), not a required
    one. motion_lora_strength scales its effect (node default/typical range
    is 0-2, values much above 1 tend to distort the frame)."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_ANIMATEDIFF_PATH))
    wf["1"]["inputs"]["ckpt_name"] = checkpoint
    wf["2"]["inputs"]["model_name"] = ANIMATEDIFF_MOTION_MODULE
    if motion_lora_name:
        wf["60"] = {
            "class_type": "ADE_AnimateDiffLoRALoader",
            "inputs": {"name": motion_lora_name, "strength": motion_lora_strength},
        }
        wf["2"]["inputs"]["motion_lora"] = ["60", 0]
    wf["10"]["inputs"]["image"] = face_ref_image_filename
    wf["11"]["inputs"]["preset"] = IP_ADAPTER_PRESET
    wf["12"]["inputs"]["weight"] = ip_adapter_weight
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["batch_size"] = frames
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = ANIMATEDIFF_SAMPLER
    wf["3"]["inputs"]["scheduler"] = ANIMATEDIFF_SCHEDULER
    wf["52"]["inputs"]["seed"] = seed
    wf["52"]["inputs"]["cfg"] = cfg
    wf["52"]["inputs"]["sampler_name"] = ANIMATEDIFF_SAMPLER
    wf["52"]["inputs"]["scheduler"] = ANIMATEDIFF_SCHEDULER
    wf["52"]["inputs"]["denoise"] = facedetailer_denoise
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix
    wf["9"]["inputs"]["fps"] = fps

    return _submit_and_wait(wf, timeout_seconds=POLL_TIMEOUT_SECONDS_ANIMATEDIFF)


def submit_txt2img_generation(
    prompt: str,
    negative_prompt: str,
    seed: int,
    filename_prefix: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = STEPS,
    cfg: float = CFG,
    checkpoint: str = CHECKPOINT,
    lora_strength: float = None,
) -> str:
    """Plain txt2img, no IP-Adapter - used for the anchor stage where there's
    no reference face yet to condition on."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_TXT2IMG_PATH))
    wf["4"]["inputs"]["ckpt_name"] = checkpoint
    if lora_strength is not None:
        wf["13"]["inputs"]["strength_model"] = lora_strength
        wf["13"]["inputs"]["strength_clip"] = lora_strength
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["batch_size"] = BATCH_SIZE
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf)


def submit_txt2img_generation_sd15(
    prompt: str,
    negative_prompt: str,
    seed: int,
    filename_prefix: str,
    checkpoint: str,
    width: int = SD15_WIDTH,
    height: int = SD15_HEIGHT,
    steps: int = STEPS,
    cfg: float = 7.0,
) -> str:
    """Plain txt2img for an SD1.5 checkpoint (e.g. CHECKPOINTS["realistic_vision"] or
    ["cyberrealistic"]) - no LoRA node, no IP-Adapter/ControlNet, since the SDXL-family
    LoRA/adapter files baked into the other templates don't match SD1.5's UNet/CLIP shape.
    checkpoint is required (not defaulted to CHECKPOINT) since this function only makes
    sense for an SD1.5 file - see SD15_CHECKPOINTS."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_TXT2IMG_SD15_PATH))
    wf["4"]["inputs"]["ckpt_name"] = checkpoint
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt
    wf["5"]["inputs"]["width"] = width
    wf["5"]["inputs"]["height"] = height
    wf["5"]["inputs"]["batch_size"] = BATCH_SIZE
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf)


def submit_img2vid_generation(
    init_image_filename: str,
    seed: int,
    filename_prefix: str,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    video_frames: int = VIDEO_FRAMES,
    fps: int = VIDEO_FPS,
    motion_bucket_id: int = MOTION_BUCKET_ID,
    steps: int = VIDEO_STEPS,
    cfg: float = VIDEO_CFG,
    crf: float = VIDEO_CRF,
) -> str:
    """Submit one SVD img2vid job, block until done, return the local path of
    the saved .webm. init_image_filename must already be uploaded (see
    upload_reference_image). Same non-torch, ComfyUI-owns-the-model model as
    submit_generation - this just builds a different workflow graph."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_IMG2VID_PATH))
    wf["1"]["inputs"]["ckpt_name"] = SVD_CHECKPOINT
    wf["2"]["inputs"]["image"] = init_image_filename
    wf["3"]["inputs"]["width"] = width
    wf["3"]["inputs"]["height"] = height
    wf["3"]["inputs"]["video_frames"] = video_frames
    wf["3"]["inputs"]["fps"] = fps
    wf["3"]["inputs"]["motion_bucket_id"] = motion_bucket_id
    wf["4"]["inputs"]["min_cfg"] = VIDEO_MIN_CFG
    wf["5"]["inputs"]["seed"] = seed
    wf["5"]["inputs"]["steps"] = steps
    wf["5"]["inputs"]["cfg"] = cfg
    wf["5"]["inputs"]["sampler_name"] = VIDEO_SAMPLER
    wf["5"]["inputs"]["scheduler"] = VIDEO_SCHEDULER
    wf["7"]["inputs"]["filename_prefix"] = filename_prefix
    wf["7"]["inputs"]["fps"] = fps
    wf["7"]["inputs"]["crf"] = crf

    return _submit_and_wait(wf, output_node_id="7", timeout_seconds=POLL_TIMEOUT_SECONDS_VIDEO)


def _round64(x: int) -> int:
    """Round to the nearest multiple of 64 (SDXL UNet + VAE both want /64)."""
    return max(64, int(round(x / 64.0)) * 64)


def _rewire(wf: dict, old_ref: list, new_ref: list) -> None:
    """Replace every node input that links to old_ref (e.g. ["14", 0]) with
    new_ref, so a node can be dropped and its consumers re-pointed at its
    upstream. Used to bypass optional HQ nodes (character LoRA / ControlNet /
    IP-Adapter / hires / FaceDetailer) instead of keeping separate templates -
    a LoraLoader with a missing file fails validation, so it must be removed
    from the graph entirely, not just set to strength 0."""
    for node in wf.values():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and val == old_ref:
                node["inputs"][key] = list(new_ref)


def _drop_nodes(wf: dict, ids) -> None:
    for node_id in ids:
        wf.pop(node_id, None)


def submit_generation_hq(
    prompt: str,
    negative_prompt: str,
    seed: int,
    filename_prefix: str,
    ip_adapter_image_filename: str = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    steps: int = HQ_BASE_STEPS,
    cfg: float = CFG,
    ip_adapter_weight: float = IP_ADAPTER_WEIGHT,
    checkpoint: str = CHECKPOINT,
    lora_strength: float = None,
    character_lora: str = None,
    character_lora_strength: float = CHARACTER_LORA_STRENGTH,
    pose_image_filename: str = None,
    controlnet_strength: float = CONTROLNET_STRENGTH,
    pose_is_skeleton: bool = False,
    hires: bool = True,
    hires_denoise: float = HIRES_DENOISE,
    hires_steps: int = HIRES_STEPS,
    use_facedetailer: bool = True,
    face_denoise: float = FACEDETAILER_FACE_DENOISE,
    hand_denoise: float = FACEDETAILER_HAND_DENOISE,
    client_id: str = None,
) -> str:
    """Unified high-quality generation (see workflow_template_hq.json and the
    HQ constants above). width/height are the FINAL output size; the first
    pass runs at HQ_FIRST_PASS[(w,h)] (or round64(dim/HIRES_SCALE)) and the
    hires pass brings it up to width/height via ESRGAN + a low-denoise resample.

    Optional stages are bypassed by rewiring + dropping their nodes so one
    template serves every combination:
      - character_lora=None  -> drop node 14 (second LoraLoader)
      - ip_adapter_image_filename=None -> drop the IP-Adapter chain (10/11/12)
      - pose_image_filename=None -> drop the ControlNet chain (20-23)
      - hires=False -> drop the ESRGAN/hires chain (30-35)
      - use_facedetailer=False -> drop the FaceDetailer chain (40-43)

    character_lora is a filename (or "characters/<id>.safetensors" subpath)
    relative to ComfyUI's loras dir; used TOGETHER with FaceID (callers lower
    ip_adapter_weight to ~0.7 when a LoRA is present - the LoRA carries
    identity, FaceID only corrects drift). ControlNet is honoured when
    pose_image_filename is given but is the slowest stage on 8GB - callers
    keep it off by default. pose_is_skeleton=True means pose_image_filename is
    already an OpenPose skeleton render (from training/poses/) rather than a
    photo, so the OpenposePreprocessor (node 21) is skipped and the skeleton
    feeds ControlNet directly."""
    wf = copy.deepcopy(_load_template(WORKFLOW_TEMPLATE_HQ_PATH))
    wf["4"]["inputs"]["ckpt_name"] = checkpoint

    # node 13: style slider LoRA (default baked at 0.0 for Pony)
    if lora_strength is not None:
        wf["13"]["inputs"]["strength_model"] = lora_strength
        wf["13"]["inputs"]["strength_clip"] = lora_strength

    # node 14: per-character LoRA
    if character_lora:
        wf["14"]["inputs"]["lora_name"] = character_lora
        wf["14"]["inputs"]["strength_model"] = character_lora_strength
        wf["14"]["inputs"]["strength_clip"] = character_lora_strength
    else:
        _rewire(wf, ["14", 0], ["13", 0])
        _rewire(wf, ["14", 1], ["13", 1])
        _drop_nodes(wf, ["14"])

    # nodes 10/11/12: IP-Adapter FaceID
    if ip_adapter_image_filename:
        wf["10"]["inputs"]["image"] = ip_adapter_image_filename
        wf["11"]["inputs"]["preset"] = IP_ADAPTER_PRESET
        wf["11"]["inputs"]["provider"] = INSIGHTFACE_PROVIDER
        wf["12"]["inputs"]["weight"] = ip_adapter_weight
        wf["12"]["inputs"]["weight_faceidv2"] = ip_adapter_weight
        model_source = ["12", 0]
    else:
        # KSamplers/FaceDetailer take the model straight from node 14 (or 13 if
        # 14 was dropped - _rewire above already collapsed 14->13 in that case).
        model_source = ["14", 0] if character_lora else ["13", 0]
        _rewire(wf, ["12", 0], model_source)
        _drop_nodes(wf, ["10", "11", "12"])

    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt

    # nodes 20-23: ControlNet OpenPose (optional, slowest on 8GB)
    if pose_image_filename:
        wf["20"]["inputs"]["image"] = pose_image_filename
        wf["22"]["inputs"]["control_net_name"] = CONTROLNET_MODEL
        wf["23"]["inputs"]["strength"] = controlnet_strength
        if pose_is_skeleton:
            # The image IS already an OpenPose skeleton render (from the pose
            # library), not a photo - feed node 23 straight from LoadImage and
            # drop the preprocessor, which would otherwise try to detect a body
            # in a stick figure and produce garbage.
            _rewire(wf, ["21", 0], ["20", 0])
            _drop_nodes(wf, ["21"])
    else:
        # pass 1 KSampler positive/negative go straight to the text encoders
        _rewire(wf, ["23", 0], ["6", 0])
        _rewire(wf, ["23", 1], ["7", 0])
        _drop_nodes(wf, ["20", "21", "22", "23"])

    # first-pass resolution
    fp_w, fp_h = HQ_FIRST_PASS.get((width, height), (_round64(width / HIRES_SCALE), _round64(height / HIRES_SCALE)))
    wf["5"]["inputs"]["width"] = fp_w
    wf["5"]["inputs"]["height"] = fp_h
    wf["5"]["inputs"]["batch_size"] = BATCH_SIZE

    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = steps
    wf["3"]["inputs"]["cfg"] = cfg
    wf["3"]["inputs"]["sampler_name"] = SAMPLER
    wf["3"]["inputs"]["scheduler"] = SCHEDULER

    # nodes 30-35: ESRGAN upscale + hires resample. scale_by brings the 4x
    # ESRGAN output down to the requested HIRES_SCALE of the first pass.
    if hires:
        wf["30"]["inputs"]["model_name"] = UPSCALE_MODEL
        wf["32"]["inputs"]["scale_by"] = HIRES_SCALE / UPSCALE_MODEL_SCALE
        wf["34"]["inputs"]["model"] = list(model_source)
        wf["34"]["inputs"]["seed"] = seed
        wf["34"]["inputs"]["steps"] = hires_steps
        wf["34"]["inputs"]["cfg"] = cfg
        wf["34"]["inputs"]["sampler_name"] = SAMPLER
        wf["34"]["inputs"]["scheduler"] = SCHEDULER
        wf["34"]["inputs"]["denoise"] = hires_denoise
    else:
        _rewire(wf, ["35", 0], ["8", 0])
        _drop_nodes(wf, ["30", "31", "32", "33", "34", "35"])

    # nodes 40-43: FaceDetailer face pass then hand pass
    if use_facedetailer:
        wf["41"]["inputs"]["model"] = list(model_source)
        wf["41"]["inputs"]["seed"] = seed
        wf["41"]["inputs"]["cfg"] = cfg
        wf["41"]["inputs"]["sampler_name"] = SAMPLER
        wf["41"]["inputs"]["scheduler"] = SCHEDULER
        wf["41"]["inputs"]["denoise"] = face_denoise
        wf["41"]["inputs"]["guide_size"] = FACEDETAILER_FACE_GUIDE_SIZE
        wf["43"]["inputs"]["model"] = list(model_source)
        wf["43"]["inputs"]["seed"] = seed
        wf["43"]["inputs"]["cfg"] = cfg
        wf["43"]["inputs"]["sampler_name"] = SAMPLER
        wf["43"]["inputs"]["scheduler"] = SCHEDULER
        wf["43"]["inputs"]["denoise"] = hand_denoise
        wf["43"]["inputs"]["guide_size"] = FACEDETAILER_HAND_GUIDE_SIZE
    else:
        # SaveImage takes the hires (or base) decode directly
        _rewire(wf, ["43", 0], ["35", 0] if hires else ["8", 0])
        _drop_nodes(wf, ["40", "41", "42", "43"])

    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return _submit_and_wait(wf, client_id=client_id)


def _submit_and_wait(wf: dict, output_node_id: str = "9", timeout_seconds: int = POLL_TIMEOUT_SECONDS,
                     client_id: str = None) -> str:
    payload = {"prompt": wf}
    if client_id:
        payload["client_id"] = client_id
    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("node_errors"):
        raise RuntimeError(f"workflow validation failed: {body['node_errors']}")
    prompt_id = body["prompt_id"]

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10).json()
        entry = hist.get(prompt_id)
        if entry:
            status = entry["status"]
            # status["completed"] never becomes True on a server-side node error - it just
            # stays False forever, so a naive "wait for completed" loop spins for the full
            # timeout_seconds (up to 1800s) on a job that actually failed in under a second.
            # Check status_str explicitly so failures surface immediately with the real cause.
            if status["status_str"] == "error":
                raise RuntimeError(f"generation failed: {_format_execution_error(status)}")
            if status["completed"]:
                image_info = entry["outputs"][output_node_id]["images"][0]
                return _download_output(image_info)
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"prompt {prompt_id} did not complete within {timeout_seconds}s")


def _format_execution_error(status: dict) -> str:
    for kind, payload in status.get("messages", []):
        if kind == "execution_error":
            return f"{payload.get('node_type')} (node {payload.get('node_id')}): {payload.get('exception_message')}"
    return str(status)


def _download_output(image_info: dict) -> str:
    params = {
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    }
    r = requests.get(f"{COMFYUI_URL}/view", params=params, timeout=60)
    r.raise_for_status()
    # Overridable so the RunPod worker can write to a tempdir instead of the
    # repo's outputs/ tree (which doesn't exist in the container).
    out_dir = os.environ.get(
        "COMFYUI_RAW_OUTPUT_DIR",
        os.path.join(os.path.dirname(__file__), "..", "outputs", "_comfyui_raw"),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, image_info["filename"])
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path
