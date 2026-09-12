"""Node-id contracts for the ComfyUI workflow templates.

`training/comfyui_client.py` builds every generation request by mutating a loaded
`training/workflow_template*.json` file BY MAGIC STRING NODE ID -
`wf["4"]["inputs"]["ckpt_name"]`, `wf["13"]["inputs"]["strength_model"]`,
`wf["41"]["inputs"]["seed"]`, and so on for ~15 `submit_*` functions across 11 template
files. Which node id means "the checkpoint loader" or "the negative prompt" is nowhere
declared - it is tribal knowledge duplicated across the JSON files, the Python that pokes
them, and a third copy in `training/benchmark.py`'s `HQ_STAGE_NAMES`.

That convention is fragile in one specific, expensive way: re-exporting a template from the
ComfyUI UI (to add a node, or after a ComfyUI upgrade changes a node's default widgets)
renumbers every node. Nothing catches this at load time - `_load_template()` just returns
whatever dict is in the file. The break only surfaces minutes into a real run, after models
have already loaded onto the 8GB card, as either a bare `KeyError` (a node id that used to
exist is gone) or an opaque ComfyUI `node_errors` response (a node id now points at a
different, incompatible class - e.g. what used to be node "7"'s `CLIPTextEncode` is now a
`LoraLoader`, and setting `wf["7"]["inputs"]["text"]` either KeyErrors or silently no-ops
into the wrong field).

This module makes the convention an explicit, checkable data structure: for each template
file, which node id is expected to hold which ComfyUI `class_type`. `check()` compares that
expectation against a loaded workflow dict and returns human-readable mismatches - it does
no I/O itself, so callers (or a test) decide when and what to load.

The CONTRACTS values below were built by reading every `training/workflow_template*.json`
file's actual node ids, not guessed from the Python. Shared sub-dicts (_SDXL_CORE, _FACEID,
_CONTROLNET, _FACEDETAILER, _HIRES_CORE, ...) exist only where multiple templates genuinely
reuse the same node-id convention - composing them with `{**a, **b}` makes that sharing
visible instead of restating the same id/class pairs eleven times.
"""

# --- Shared sub-dicts -------------------------------------------------------------------
# Every SDXL-family still-image template (juggernaut/pony/cyberrealistic_pony/pony_realism,
# all sharing one UNet/CLIP shape - see CHECKPOINTS in comfyui_client.py) that was built from
# the same original export keeps this same core id layout.
_CHECKPOINT = {"4": "CheckpointLoaderSimple"}
_STYLE_LORA = {"13": "LoraLoader"}  # the baked-in sdxl_photorealistic_slider_v1-0 LoRA
_PROMPTS = {"6": "CLIPTextEncode", "7": "CLIPTextEncode"}  # 6=positive, 7=negative - see
# POSITIVE_NODE/NEGATIVE_NODE below
_LATENT = {"5": "EmptyLatentImage"}
_SAMPLE_SAVE = {"3": "KSampler", "8": "VAEDecode", "9": "SaveImage"}
_SDXL_CORE = {**_CHECKPOINT, **_STYLE_LORA, **_PROMPTS, **_LATENT, **_SAMPLE_SAVE}

# IP-Adapter FaceID identity-locking chain (see IP_ADAPTER_PRESET in comfyui_client.py) -
# reused by every template that conditions on a reference face.
_FACEID = {
    "10": "LoadImage",
    "11": "IPAdapterUnifiedLoaderFaceID",
    "12": "IPAdapterFaceID",
}

# ControlNet OpenPose (control-lora) chain - photo or skeleton reference -> preprocessor ->
# loader -> apply. Reused by the standalone ControlNet template and the HQ unified template.
_CONTROLNET = {
    "20": "LoadImage",
    "21": "OpenposePreprocessor",
    "22": "ControlNetLoader",
    "23": "ControlNetApplyAdvanced",
}

# ADetailer-equivalent face+hand refinement, YOLOv8-bbox-based (see FACEDETAILER_DENOISE) -
# reused by the plain FaceDetailer template and the HQ template's own detailer stage.
_FACEDETAILER = {
    "40": "UltralyticsDetectorProvider",
    "41": "FaceDetailer",
    "42": "UltralyticsDetectorProvider",
    "43": "FaceDetailer",
}
# Mediapipe variant: only the face pass's detector/sampler differ (face-mesh SEGS instead of
# a bbox), the hand pass stays YOLOv8 - see submit_generation_with_facedetailer_mediapipe.
_MEDIAPIPE_FACEDETAILER = {
    "40": "MediaPipeFaceMeshToSEGS",
    "41": "DetailerForEach",
    "42": "UltralyticsDetectorProvider",
    "43": "FaceDetailer",
}

# ESRGAN-seeded hires second pass (see UPSCALE_MODEL/HIRES_* in comfyui_client.py): upscale
# model -> image upscale -> lanczos scale-down -> VAEEncode -> second KSampler. The final
# decode (node "35") is the one place the HQ still-image path and the AnimateDiff video path
# diverge (VAEDecode vs VAEDecodeTiled, since video decodes a whole frame batch) - so it's
# deliberately left out of this shared core and added per-template below.
_HIRES_CORE = {
    "30": "UpscaleModelLoader",
    "31": "ImageUpscaleWithModel",
    "32": "ImageScaleBy",
    "33": "VAEEncode",
    "34": "KSampler",
}

# Z-Image Turbo loads three separate files instead of one CheckpointLoaderSimple checkpoint
# (see ZIMAGE_MODELS) - a genuinely different id convention, kept as its own sub-dict rather
# than forced into _SDXL_CORE.
_ZIMAGE_LOADERS = {
    "10": "UNETLoader",
    "11": "CLIPLoader",
    "12": "VAELoader",
    "13": "ModelSamplingAuraFlow",
}

# SVD img2vid: a small, self-contained id scheme (1-7) that shares nothing with the SDXL
# templates - no LoRA, no FaceID, and conditioning comes from SVD_img2vid_Conditioning
# rather than a pair of CLIPTextEncode nodes, so there is no positive/negative prompt node
# here at all (see the NEGATIVE_NODE docstring note and check group 5 in the test file).
_IMG2VID = {
    "1": "ImageOnlyCheckpointLoader",
    "2": "LoadImage",
    "3": "SVD_img2vid_Conditioning",
    "4": "VideoLinearCFGGuidance",
    "5": "KSampler",
    "6": "VAEDecode",
    "7": "SaveWEBM",
}

# AnimateDiff (SD1.5 + motion module) reuses _FACEID, _PROMPTS, _LATENT and _HIRES_CORE
# verbatim (same node ids, same classes) but has its own checkpoint/motion-module loader
# pair and its own per-frame video detailer + final-export chain.
_ANIMATEDIFF_CHECKPOINT = {"1": "CheckpointLoaderSimple", "2": "ADE_AnimateDiffLoaderGen1"}
# Video face detailer: face detected across the whole frame batch at once (40/50), bundled
# into a basic_pipe (51) so DetailerForEachPipeForAnimateDiff (52) re-samples all frames
# together through the same AnimateDiff-patched model instead of independently per frame.
_ANIMATEDIFF_DETAILER = {
    "40": "UltralyticsDetectorProvider",
    "50": "ImpactSimpleDetectorSEGS_for_AD",
    "51": "ToBasicPipe",
    "52": "DetailerForEachPipeForAnimateDiff",
}
_ANIMATEDIFF_FINAL = {
    "36": "ImageUpscaleWithModel",
    "37": "ImageScale",
    "90": "CreateVideo",
    "9": "SaveVideo",  # not SaveImage - the one id that collides with _SAMPLE_SAVE's "9"
    # but means something different, which is exactly why AnimateDiff doesn't compose
    # _SAMPLE_SAVE at all and spells its own "3"/"8"/"9" out below.
}

# --- The contract map --------------------------------------------------------------------
# template filename -> {node_id: expected class_type}. Every training/workflow_template*.json
# file must have exactly one entry here (see tests/test_workflow_contract.py group 1).
CONTRACTS: dict[str, dict[str, str]] = {
    "workflow_template.json": {**_SDXL_CORE, **_FACEID},
    "workflow_template_txt2img.json": dict(_SDXL_CORE),
    "workflow_template_txt2img_sd15.json": {**_CHECKPOINT, **_PROMPTS, **_LATENT, **_SAMPLE_SAVE},
    "workflow_template_txt2img_zimage.json": {
        **_ZIMAGE_LOADERS, **_PROMPTS, "5": "EmptySD3LatentImage", **_SAMPLE_SAVE,
    },
    "workflow_template_img2img.json": {
        **_CHECKPOINT, **_STYLE_LORA, **_FACEID, **_PROMPTS, "20": "LoadImage", "21": "VAEEncode", **_SAMPLE_SAVE,
    },
    "workflow_template_controlnet.json": {**_SDXL_CORE, **_FACEID, **_CONTROLNET},
    "workflow_template_facedetailer.json": {**_SDXL_CORE, **_FACEID, **_FACEDETAILER},
    "workflow_template_mediapipe_facedetailer.json": {**_SDXL_CORE, **_FACEID, **_MEDIAPIPE_FACEDETAILER},
    "workflow_template_hq.json": {
        **_CHECKPOINT, **_STYLE_LORA, "14": "LoraLoader",  # per-character LoRA, on top of the style LoRA
        **_FACEID, **_PROMPTS, **_CONTROLNET, **_LATENT,
        **_HIRES_CORE, "35": "VAEDecode",
        **_FACEDETAILER, **_SAMPLE_SAVE,
    },
    "workflow_template_img2vid.json": dict(_IMG2VID),
    "workflow_template_animatediff_facedetailer.json": {
        **_ANIMATEDIFF_CHECKPOINT, **_FACEID, **_PROMPTS, **_LATENT,
        "3": "KSampler", "8": "VAEDecode",
        **_HIRES_CORE, "35": "VAEDecodeTiled",
        **_ANIMATEDIFF_DETAILER, **_ANIMATEDIFF_FINAL,
    },
}

# Semantic roles shared by every still-image template that has them (all but img2vid, whose
# conditioning comes from SVD_img2vid_Conditioning instead of a text encoder pair - see
# _IMG2VID above). cfg is floored above 1.0 everywhere (see comfyui_client.py's ZIMAGE_MIN_CFG/
# LCM_MIN_CFG) specifically so NEGATIVE_NODE's text - which carries AGE_SAFETY_NEGATIVE - is
# never skipped by ComfyUI's cfg==1.0 shortcut.
POSITIVE_NODE = "6"
NEGATIVE_NODE = "7"


def check(template_name: str, wf: dict) -> list:
    """Compare a loaded workflow dict against CONTRACTS[template_name].

    Returns a list of human-readable mismatch strings (empty = OK). Each names the node id,
    what class was expected there and what was actually found (or that the node is simply
    missing), plus a hint pointing at the most likely cause: the template was re-exported
    from the ComfyUI UI and every node got renumbered.
    """
    contract = CONTRACTS.get(template_name)
    if contract is None:
        return [f"no contract registered for template {template_name!r} - add one to workflow_contracts.CONTRACTS"]

    issues = []
    for node_id, expected_class in contract.items():
        node = wf.get(node_id)
        if not isinstance(node, dict):
            issues.append(
                f"node {node_id!r}: expected class_type {expected_class!r}, but the node is missing - "
                "the template may have been re-exported from the ComfyUI UI and renumbered"
            )
            continue
        actual_class = node.get("class_type")
        if actual_class != expected_class:
            issues.append(
                f"node {node_id!r}: expected class_type {expected_class!r}, found {actual_class!r} - "
                "the template may have been re-exported from the ComfyUI UI and renumbered"
            )
    return issues
