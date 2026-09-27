"""training/workflow_safety.py: the checks both sides run on a workflow sent to the RunPod worker.

Two failure directions matter and both are pinned here:
- a false rejection makes a working local feature unusable in the cloud, so EVERY graph the real
  submit_* builders produce must pass (including ControlNet pass-through and SVD, which has no text);
- a false acceptance lets a caller holding the endpoint key skip the age-safety negatives, so each
  way of doing that is shown to be refused.
"""

import copy
import os
import re

import pytest

import comfyui_client as client
import generate_character as gc
import workflow_safety as ws

NEG = f"lowres, {gc.AGE_SAFETY_NEGATIVE}"
TRAINING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")

BUILDERS = {
    "hq_plain": ("submit_generation_hq", {}),
    "hq_faceid_pose_skeleton": (
        "submit_generation_hq",
        {"ip_adapter_image_filename": "a.png", "pose_image_filename": "p.png", "pose_is_skeleton": True},
    ),
    "hq_pose_photo": ("submit_generation_hq", {"pose_image_filename": "p.png"}),
    "hq_no_hires_no_detailer": ("submit_generation_hq", {"hires": False, "use_facedetailer": False}),
    "faceid": ("submit_generation", {"ip_adapter_image_filename": "a.png"}),
    "legacy_pose": (
        "submit_generation_with_pose",
        {"ip_adapter_image_filename": "a.png", "pose_image_filename": "p.png"},
    ),
    "facedetailer": ("submit_generation_with_facedetailer", {"ip_adapter_image_filename": "a.png"}),
    "facedetailer_mediapipe": ("submit_generation_with_facedetailer_mediapipe", {"ip_adapter_image_filename": "a.png"}),
    "img2img": (
        "submit_img2img_generation",
        {"ip_adapter_image_filename": "a.png", "init_image_filename": "i.png", "denoise": 0.45},
    ),
    "txt2img": ("submit_txt2img_generation", {}),
    "txt2img_sd15": ("submit_txt2img_generation_sd15", {"checkpoint": client.CHECKPOINTS["realistic_vision"]}),
    "zimage": ("submit_txt2img_generation_zimage", {}),
    "wan_i2v": ("submit_generation_wan_i2v", {"start_image_filename": "f.png"}),
    "animatediff": ("submit_generation_animatediff", {"face_ref_image_filename": "a.png"}),
    "animatediff_rife": ("submit_generation_animatediff", {"face_ref_image_filename": "a.png", "interp_multiplier": 2}),
}


def _build(monkeypatch, captured_submit, key):
    monkeypatch.setattr(client, "zimage_missing_files", lambda model="z_image_turbo": [])
    monkeypatch.setattr(client, "has_node", lambda name: True)
    fn_name, extra = BUILDERS[key]
    kwargs = {"seed": 1, "filename_prefix": "stem", **extra}
    if fn_name != "submit_img2vid_generation":
        kwargs.update(prompt="a portrait", negative_prompt=NEG)
    getattr(client, fn_name)(**kwargs)
    call = captured_submit[-1]
    return call["wf"], call["output_node_id"]


@pytest.mark.parametrize("key", sorted(BUILDERS))
def test_every_real_builder_graph_passes(monkeypatch, captured_submit, key):
    wf, out = _build(monkeypatch, captured_submit, key)
    ws.validate_graph(wf, out)
    assert ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE) >= 1


def test_rife_interpolation_graph_is_valid_and_generates_nothing_from_text(monkeypatch, captured_submit):
    """Not in BUILDERS on purpose: every builder there generates from a prompt and must carry the
    safety negative; RIFE only adds in-between frames to a clip that already exists."""
    monkeypatch.setattr(client, "has_node", lambda name: True)
    client.submit_interpolation_rife(video_filename="clip.mp4", filename_prefix="stem", multiplier=2, fps=24)
    call = captured_submit[-1]
    wf = call["wf"]
    ws.validate_graph(wf, call["output_node_id"])
    assert ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE) == 0
    assert not any(n["class_type"] in ("KSampler", "CLIPTextEncode") for n in wf.values())
    assert wf["1"]["inputs"]["file"] == "clip.mp4"
    assert wf["3"]["inputs"]["multiplier"] == 2 and wf["3"]["inputs"]["ckpt_name"] == client.RIFE_CKPT
    assert wf["4"]["inputs"]["fps"] == 48.0  # source rate x multiplier: same speed, same length
    assert ws.output_kind(wf, call["output_node_id"]) == ("mp4", "video/mp4")


def test_rife_interpolation_needs_the_node_pack_and_a_real_multiplier(monkeypatch):
    monkeypatch.setattr(client, "has_node", lambda name: False)
    with pytest.raises(RuntimeError, match="ComfyUI-Frame-Interpolation"):
        client.submit_interpolation_rife(video_filename="clip.mp4", filename_prefix="stem")
    monkeypatch.setattr(client, "has_node", lambda name: True)
    with pytest.raises(ValueError):
        client.submit_interpolation_rife(video_filename="clip.mp4", filename_prefix="stem", multiplier=1)


def test_svd_graph_passes_without_any_text_encoder(monkeypatch, captured_submit):
    client.submit_img2vid_generation(init_image_filename="i.png", seed=1, filename_prefix="stem")
    call = captured_submit[-1]
    ws.validate_graph(call["wf"], call["output_node_id"])
    assert ws.check_negative_safety(call["wf"], gc.AGE_SAFETY_NEGATIVE) >= 1
    assert ws.output_kind(call["wf"], call["output_node_id"]) == ("webm", "video/webm")


def test_output_kinds_match_what_the_client_downloads(monkeypatch, captured_submit):
    wf, out = _build(monkeypatch, captured_submit, "hq_plain")
    assert ws.output_kind(wf, out) == ("png", "image/png")
    wf, out = _build(monkeypatch, captured_submit, "animatediff")
    assert ws.output_kind(wf, out) == ("mp4", "video/mp4")


def test_injected_classes_cover_every_node_the_code_adds():
    """INJECTED_CLASSES must list every class the builders add at runtime, or the worker would
    refuse a graph the local code legitimately built."""
    found = set()
    for fname in ("comfyui_client.py", "generate_character.py"):
        with open(os.path.join(TRAINING, fname), encoding="utf-8") as f:
            src = f.read()
        found.update(re.findall(r'"class_type":\s*"([^"]+)"', src))
        if re.search(r'"class_type":\s*RIFE_NODE', src):
            found.add(client.RIFE_NODE)
    assert found <= ws.allowed_classes()


# --- refusals ---------------------------------------------------------------------------------


@pytest.fixture
def hq(monkeypatch, captured_submit):
    return _build(monkeypatch, captured_submit, "hq_faceid_pose_skeleton")


def _negative_text_nodes(wf):
    return [
        nid
        for nid, n in wf.items()
        if n["class_type"] == "CLIPTextEncode" and gc.AGE_SAFETY_NEGATIVE in (n["inputs"].get("text") or "")
    ]


def test_missing_age_terms_are_refused(hq):
    wf = copy.deepcopy(hq[0])
    for nid in _negative_text_nodes(wf):
        wf[nid]["inputs"]["text"] = "lowres, blurry"
    with pytest.raises(ws.WorkflowRejected, match="missing the required safety terms"):
        ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE)


def test_negative_wired_to_the_positive_encoder_is_refused(hq):
    wf = copy.deepcopy(hq[0])
    positive_id = next(
        nid for nid, n in wf.items() if n["class_type"] == "CLIPTextEncode" and nid not in _negative_text_nodes(wf)
    )
    for node in wf.values():
        if "negative" in node["inputs"] and node["class_type"] != "ControlNetApplyAdvanced":
            node["inputs"]["negative"] = [positive_id, 0]
    with pytest.raises(ws.WorkflowRejected):
        ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE)


def test_controlnet_positive_output_used_as_negative_is_refused(hq):
    wf = copy.deepcopy(hq[0])
    cn_id = next(nid for nid, n in wf.items() if n["class_type"] == "ControlNetApplyAdvanced")
    # The positive encoder carries no safety text, so routing the sampler's negative to the
    # ControlNet's POSITIVE output (index 0) must fail even though the node itself is allowed.
    wf[wf[cn_id]["inputs"]["positive"][0]]["inputs"]["text"] = "a portrait"
    sampler = next(n for n in wf.values() if n["inputs"].get("negative") == [cn_id, 1])
    sampler["inputs"]["negative"] = [cn_id, 0]
    with pytest.raises(ws.WorkflowRejected, match="missing the required safety terms"):
        ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE)


def test_unknown_sampler_class_is_refused(hq):
    wf, out = copy.deepcopy(hq[0]), hq[1]
    wf["999"] = {"class_type": "SamplerCustomAdvanced", "inputs": {}}
    with pytest.raises(ws.WorkflowRejected, match="SamplerCustomAdvanced"):
        ws.validate_graph(wf, out)


def test_non_literal_negative_text_is_refused(hq):
    wf = copy.deepcopy(hq[0])
    nid = _negative_text_nodes(wf)[0]
    wf[nid]["inputs"]["text"] = [nid, 0]
    with pytest.raises(ws.WorkflowRejected, match="not literal text"):
        ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE)


def test_unconnected_negative_is_refused(hq):
    wf = copy.deepcopy(hq[0])
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    sampler["inputs"]["negative"] = "some text"
    with pytest.raises(ws.WorkflowRejected, match="not connected"):
        ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE)


def test_dangling_link_and_bad_output_node_are_refused(hq):
    wf, out = copy.deepcopy(hq[0]), hq[1]
    broken = copy.deepcopy(wf)
    next(n for n in broken.values() if n["class_type"] == "KSampler")["inputs"]["model"] = ["nope", 0]
    with pytest.raises(ws.WorkflowRejected, match="missing node"):
        ws.validate_graph(broken, out)
    with pytest.raises(ws.WorkflowRejected, match="not in the workflow"):
        ws.validate_graph(wf, "12345")
    loader = next(nid for nid, n in wf.items() if n["class_type"] == "CheckpointLoaderSimple")
    with pytest.raises(ws.WorkflowRejected, match="not one of"):
        ws.validate_graph(wf, loader)


@pytest.mark.parametrize("bad", [None, {}, [], "x"])
def test_non_object_workflow_is_refused(bad):
    with pytest.raises(ws.WorkflowRejected):
        ws.validate_graph(bad, "9")


def test_oversized_workflow_is_refused(hq):
    wf, out = copy.deepcopy(hq[0]), hq[1]
    for i in range(ws.MAX_NODES):
        wf[f"x{i}"] = {"class_type": "EmptyLatentImage", "inputs": {}}
    with pytest.raises(ws.WorkflowRejected, match="nodes"):
        ws.validate_graph(wf, out)


def test_empty_required_text_is_refused(hq):
    with pytest.raises(ws.WorkflowRejected):
        ws.check_negative_safety(hq[0], "")


def test_load_image_names_lists_each_once(hq):
    names = ws.load_image_names(hq[0])
    assert sorted(names) == ["a.png", "p.png"]
