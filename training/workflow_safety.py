"""Structural and safety checks for a ComfyUI workflow that is about to run somewhere else.

The RunPod worker's "workflow" job executes a graph built on the local machine. That is what lets
every local feature (pose skeletons, Z-Image, SD1.5, GIF frames) run in the cloud without a copy of
each one in the worker - but it also means the worker must not simply trust the graph it receives.
The same two checks run on both sides: locally before a job is sent (so a bad graph fails fast and
costs nothing), and in the worker before it touches the GPU (so a caller holding the endpoint key
cannot bypass them by sending its own JSON).

What is enforced:
- every node's class comes from this repo's own templates, plus the few nodes the code injects;
  that keeps samplers that could skip the negative prompt (SamplerCustom*, BasicGuider) out;
- the output node is one of the save nodes the local code reads back;
- every "negative" conditioning input traces back, through conditioning pass-through nodes, to a
  literal CLIPTextEncode containing the required safety text. Links are followed rather than node
  ids assumed, so a re-exported (renumbered) template still passes and a rewired one still fails.

The cfg floor is deliberately not re-checked here: comfyui_client.enforce_min_cfg already runs
inside _submit_and_wait on both sides and raises any sampler below SAFETY_MIN_CFG.

Stdlib only, and the caller passes the required text in: comfyui_client imports this module, and
generate_character (which owns AGE_SAFETY_NEGATIVE) imports comfyui_client.
"""

import glob
import json
import os

TEMPLATE_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_template*.json")

# Nodes the code adds to a template at build time rather than shipping in the JSON. Kept in step
# with the source by tests/test_workflow_safety.py, which scans comfyui_client/generate_character.
INJECTED_CLASSES = frozenset({
    "ADE_AnimateDiffLoRALoader",
    "ADE_MultivalDynamic",
    "LoraLoaderModelOnly",
    "RIFE VFI",
})

# output class -> (file extension, MIME type) of what the local code downloads from it
OUTPUT_CLASSES = {
    "SaveImage": ("png", "image/png"),
    "SaveVideo": ("mp4", "video/mp4"),
    "SaveWEBM": ("webm", "video/webm"),
}

MAX_NODES = 500

# Pass-through conditioning nodes: (class, output index) -> the input whose link it forwards.
_CONDITIONING_PASSTHROUGH = {
    ("ControlNetApplyAdvanced", 0): "positive",
    ("ControlNetApplyAdvanced", 1): "negative",
}
# Image-only sources that legitimately produce negative conditioning with no text at all.
# SVD has no text encoder; its negative is derived from the conditioning image.
_IMAGE_ONLY_NEGATIVE_SOURCES = {("SVD_img2vid_Conditioning", 1)}


class WorkflowRejected(ValueError):
    """The workflow failed a structural or safety check. The message is safe to show the caller."""


def allowed_classes(template_glob=TEMPLATE_GLOB):
    """Every class_type used by this repo's templates, plus the injected ones."""
    classes = set(INJECTED_CLASSES)
    for path in glob.glob(template_glob):
        with open(path, encoding="utf-8") as f:
            for node in json.load(f).values():
                classes.add(node["class_type"])
    return frozenset(classes)


def _is_link(value):
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def validate_graph(wf, output_node_id, allowed=None):
    """Shape, node-class allowlist, dangling links and output node. Raises WorkflowRejected."""
    if not isinstance(wf, dict) or not wf:
        raise WorkflowRejected("workflow must be a non-empty JSON object of nodes")
    if len(wf) > MAX_NODES:
        raise WorkflowRejected(f"workflow has {len(wf)} nodes (limit {MAX_NODES})")
    allowed = allowed_classes() if allowed is None else allowed
    for node_id, node in wf.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            raise WorkflowRejected(f"node {node_id!r} is not an object")
        cls = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(cls, str) or not isinstance(inputs, dict):
            raise WorkflowRejected(f"node {node_id} needs a string class_type and an inputs object")
        if cls not in allowed:
            raise WorkflowRejected(f"node {node_id} uses class {cls!r}, which no template in this repo uses")
        for name, value in inputs.items():
            if _is_link(value) and value[0] not in wf:
                raise WorkflowRejected(f"node {node_id} input {name!r} links to missing node {value[0]}")
    out = wf.get(str(output_node_id))
    if out is None:
        raise WorkflowRejected(f"output node {output_node_id!r} is not in the workflow")
    if out["class_type"] not in OUTPUT_CLASSES:
        raise WorkflowRejected(f"output node {output_node_id} is {out['class_type']}, not one of "
                               f"{sorted(OUTPUT_CLASSES)}")


def output_kind(wf, output_node_id):
    """(extension, MIME type) for the workflow's output node."""
    return OUTPUT_CLASSES[wf[str(output_node_id)]["class_type"]]


def _trace_negative(wf, link, required, where, seen):
    node_id, index = link
    if (node_id, index) in seen:
        raise WorkflowRejected(f"{where}: conditioning links form a cycle")
    seen.add((node_id, index))
    node = wf[node_id]
    cls = node["class_type"]
    if cls == "CLIPTextEncode":
        text = node["inputs"].get("text")
        if not isinstance(text, str):
            raise WorkflowRejected(f"{where}: negative text in node {node_id} is not literal text")
        if required not in text:
            raise WorkflowRejected(f"{where}: negative prompt in node {node_id} is missing the required "
                                   "safety terms")
        return
    if (cls, index) in _IMAGE_ONLY_NEGATIVE_SOURCES:
        return
    forwarded = _CONDITIONING_PASSTHROUGH.get((cls, index))
    if forwarded is None:
        raise WorkflowRejected(f"{where}: negative conditioning comes from {cls} output {index} "
                               f"(node {node_id}), which is not an allowed source")
    upstream = node["inputs"].get(forwarded)
    if not _is_link(upstream):
        raise WorkflowRejected(f"{where}: node {node_id} input {forwarded!r} is not connected")
    _trace_negative(wf, upstream, required, where, seen)


def check_negative_safety(wf, required):
    """Every `negative` input must resolve to text containing `required`. Raises WorkflowRejected.

    Run validate_graph first: this assumes links point at existing nodes."""
    if not required:
        raise WorkflowRejected("no required safety text was given")
    checked = 0
    for node_id, node in wf.items():
        link = node["inputs"].get("negative")
        if link is None:
            continue
        where = f"node {node_id} ({node['class_type']})"
        if not _is_link(link):
            raise WorkflowRejected(f"{where}: negative input is not connected to a node")
        _trace_negative(wf, link, required, where, set())
        checked += 1
    return checked


def load_image_names(wf):
    """Names every LoadImage node reads from ComfyUI's input directory."""
    names = []
    for node in wf.values():
        if node.get("class_type") == "LoadImage":
            name = node.get("inputs", {}).get("image")
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names
