"""The script that runs on the Colab VM, checked offline: it must import without torch, build a graph
that matches ComfyUI's MiniMax H3 node contract, agree with the local runner on frame counts and the
job file, and print markers the runner can parse."""

import ast
import json

import pytest

from minimax_h3_fakes import REMOTE_PATH, FakeProber, h3, import_remote, make_config, make_image

remote = import_remote()

# comfy_extras/nodes_minimax_h3.py, MiniMaxH3ImageToVideo.define_schema (2026-09).
H3_I2V_INPUTS = {"clip", "vae", "prompt", "width", "height", "length", "first_frame", "last_frame"}
H3_I2V_REQUIRED = {"clip", "vae", "prompt", "width", "height", "length"}


def remote_job(tmp_path, **kw):
    spec = h3.JobSpec(image=make_image(tmp_path), description="She smiles.", **kw)
    prepared = h3.prepare(spec, make_config(tmp_path), FakeProber(), job_id="j1")
    return prepared.remote_job()


def test_script_is_ascii_and_has_no_future_import():
    """colab exec reads the file with the platform default encoding and prepends its --env lines,
    which a `from __future__` import could not follow."""
    tree = ast.parse(REMOTE_PATH.read_bytes().decode("ascii"))
    assert not any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in ast.walk(tree))


def test_torch_and_huggingface_are_imported_only_inside_functions():
    """So this file imports in .venv-dev and on the CI runners, which have neither."""
    tree = ast.parse(REMOTE_PATH.read_text(encoding="ascii"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert not top_level & {"torch", "huggingface_hub", "requests"}


@pytest.mark.parametrize("seconds", [4, 4.2, 5, 6.5, 8, 10, 12, 14.375, 15])
def test_frame_rule_matches_the_local_runner(seconds):
    assert remote.frames_for(seconds) == h3.frames_for(seconds)


def test_graph_uses_the_h3_first_frame_node_contract(tmp_path):
    job = remote_job(tmp_path)
    graph = remote.build_graph(job, remote.DIFFUSION_FP8)
    node = graph["7"]
    assert node["class_type"] == "MiniMaxH3ImageToVideo"
    assert H3_I2V_REQUIRED <= set(node["inputs"]) <= H3_I2V_INPUTS
    assert node["inputs"]["length"] == 192 and (node["inputs"]["width"], node["inputs"]["height"]) == (1376, 768)
    assert node["inputs"]["prompt"] == job["prompt"]
    assert {n["class_type"] for n in graph.values()} == set(remote.REQUIRED_NODES)
    for node_id, n in graph.items():
        for value in n["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph, (node_id, value)


def test_graph_carries_the_job_settings(tmp_path):
    job = remote_job(tmp_path, seed=123)
    graph = remote.build_graph(job, remote.DIFFUSION_INT8)
    assert graph["1"]["inputs"]["unet_name"] == remote.DIFFUSION_INT8
    assert graph["9"]["inputs"]["noise_seed"] == 123
    assert graph["11"]["inputs"]["steps"] == h3.STEPS == 8
    out = graph[remote.OUTPUT_NODE]["inputs"]
    assert (out["format"], out["pix_fmt"], out["frame_rate"]) == ("video/h264-mp4", "yuv420p", 24)
    assert out["audio"] == ["14", 0]


def test_pruned_base_is_paired_with_the_pruned_lora():
    assert "pruned" in remote.DIFFUSION_FP8 and "pruned" in remote.DIFFUSION_INT8 and "pruned" in remote.LORA
    assert remote.choose_diffusion("13.0") == remote.DIFFUSION_INT8
    assert remote.choose_diffusion("12.4") == remote.DIFFUSION_FP8
    assert remote.choose_diffusion(None) == remote.DIFFUSION_FP8
    files = [f for _, f, _ in remote.model_files(remote.DIFFUSION_FP8)]
    assert files == [
        "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
    ]


def object_info_for(graph, drop=None):
    info = {}
    for node in graph.values():
        spec = info.setdefault(node["class_type"], {"input": {"required": {}, "optional": {}}})
        for name in node["inputs"]:
            if (node["class_type"], name) != drop:
                spec["input"]["required"][name] = ("ANY",)
    vhs = info["VHS_VideoCombine"]["input"]["required"]
    for widget in ("pix_fmt", "crf", "save_metadata", "trim_to_audio"):
        vhs.pop(widget)
    vhs["format"] = (
        ["video/h264-mp4"],
        {"formats": {"video/h264-mp4": [["pix_fmt"], ["crf"], ["save_metadata"], ["trim_to_audio"]]}},
    )
    return info


def test_changed_upstream_inputs_are_caught_before_queueing(tmp_path):
    graph = remote.build_graph(remote_job(tmp_path), remote.DIFFUSION_FP8)
    assert remote.unexpected_inputs(graph, object_info_for(graph)) == {}
    changed = remote.unexpected_inputs(
        graph, object_info_for(graph, drop=("ImageScaleToTotalPixels", "resolution_steps"))
    )
    assert changed == {"16": ["resolution_steps"]}


def test_job_file_leaves_room_for_a_clean_remote_timeout(tmp_path):
    job = remote_job(tmp_path, timeout=1800)
    assert job["deadline_seconds"] == 1680
    assert job["remote_output"] == "/content/h3_j1_output.mp4"
    json.dumps(job, ensure_ascii=True)


def test_markers_round_trip(capsys):
    remote.marker("GPU", name="NVIDIA L4", vram_gib=22.5)
    remote.marker("ERROR", code="MODEL_DOWNLOAD_FAILED", message='quote " and newline\n')
    lines = capsys.readouterr().out.splitlines()
    assert h3.parse_marker(lines[0]) == ("GPU", {"name": "NVIDIA L4", "vram_gib": 22.5})
    assert h3.parse_marker(lines[1])[1]["message"] == 'quote " and newline\n'
    assert h3.parse_marker("[colab] Session READY.") is None
    assert h3.parse_marker("H3_GPU {not json}") is None
