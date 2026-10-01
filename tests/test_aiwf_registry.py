"""The registry, parameter validation and graph binding - and that the two shipped workflows bind to exactly
the graphs the project already runs: the Z-Image graph comfyui_client builds, and the H3 graph the H3 skill
builds. If either drifts, the platform would be running something nobody tested.
"""

import json
import random

import comfyui_client as client
import pytest
from aiwf_fakes import PLATFORM_DIR, make_workspace
from controller import WorkflowError, params, safety
from controller.registry import Registry
from minimax_h3_fakes import h3, import_remote

ZIMAGE_FILES = {
    "unet": "z_image_turbo_bf16.safetensors",
    "text_encoder": "qwen_3_4b.safetensors",
    "vae": "ae.safetensors",
}


@pytest.fixture
def registry(tmp_path):
    storage, _ = make_workspace(tmp_path)
    return Registry.load(storage)


def test_the_shipped_registry_loads_without_problems(registry):
    assert registry.names() == ["z-image-basic", "minimax-h3-basic", "test-generation"]
    assert registry.problems == {}
    assert registry.custom_nodes() == ["ComfyUI-VideoHelperSuite"]
    assert {registry.get(n).profile for n in registry.names()} == set(safety.PROFILES)


def test_the_z_image_workflow_file_is_the_projects_template():
    ours = json.loads((PLATFORM_DIR / "workflows" / "image" / "z-image-basic.json").read_text(encoding="utf-8"))
    theirs = json.loads(
        (PLATFORM_DIR.parent / "training" / "workflow_template_txt2img_zimage.json").read_text(encoding="utf-8")
    )
    assert ours == theirs


@pytest.mark.parametrize("aspect,size", [("1:1", (1024, 1024)), ("16:9", (1344, 768)), ("2:3", (832, 1216))])
def test_z_image_binds_to_the_graph_the_project_builds(registry, aspect, size):
    wf = registry.get("z-image-basic")
    values = params.validate(wf, {"prompt": "A white mug.", "aspect": aspect, "seed": 1234, "negative": "extra mug"})
    assert (values["width"], values["height"]) == size
    graph = params.bind(wf, values, "job-x")
    expected = client.build_zimage_txt2img_workflow(
        "A white mug.", safety.build_negative("extra mug"), 1234, "job-x", ZIMAGE_FILES, size[0], size[1]
    )
    assert graph == expected
    safety.check_graph(wf.name, wf.profile, graph)


def test_the_h3_workflow_binds_to_the_graph_the_h3_skill_builds(registry):
    remote = import_remote()
    wf = registry.get("minimax-h3-basic")
    values = params.validate(
        wf, {"prompt": "A cup of coffee, steam rises.", "duration": 8, "seed": 77, "soundscape": "Soft cafe ambience."}
    )
    frame = {"comfy_name": "job-x_first_frame.png", "width": 768, "height": 1344}
    graph = params.bind(wf, values, "job-x", {"first_frame": frame})
    prompt = h3.build_i2va_prompt("A cup of coffee, steam rises.", "Soft cafe ambience.", "N/A")
    width, height = h3.auto_resolution(768, 1344, short_side=768, max_megapixels=1.2)
    job = {
        "lora_strength": 1.0, "comfy_image_name": "job-x_first_frame.png", "prompt": prompt, "width": width,
        "height": height, "frames": h3.frames_for(8), "seed": 77, "sampler": h3.SAMPLER, "scheduler": h3.SCHEDULER,
        "steps": h3.STEPS, "crf": h3.VIDEO_CRF, "job_id": "job-x",
    }  # fmt: skip
    assert graph == remote.build_graph(job, remote.DIFFUSION_FP8)
    assert {n["class_type"] for n in graph.values()} == set(remote.REQUIRED_NODES)
    assert params.expected_output(wf, values, graph) == {
        "video_codec": "h264", "pix_fmt": "yuv420p", "fps": 24, "audio": True, "width": width, "height": height, "frames": 192,
    }  # fmt: skip


@pytest.mark.parametrize("seconds", [4, 5, 5.5, 8, 12, 15])
def test_the_frame_grid_is_the_h3_skills(seconds):
    assert params.seconds_to_frames(seconds, 24, 17, 5) == h3.frames_for(seconds)


@pytest.mark.parametrize("size", [(768, 1344), (1344, 768), (1024, 1024), (2000, 900), (640, 1920)])
def test_auto_resolution_is_the_h3_skills(size):
    ours = params.auto_resolution(*size, short_side=768, max_megapixels=1.2, multiple=32)
    assert ours == h3.auto_resolution(*size, short_side=768, max_megapixels=1.2)


def test_models_are_pinned_with_revision_size_and_sha256():
    registry = json.loads((PLATFORM_DIR / "workflows" / "registry.json").read_text(encoding="utf-8"))
    models = [m for spec in registry["workflows"].values() for m in spec["models"]]
    assert len(models) == 8
    for m in models:
        assert len(m["revision"]) == 40 and len(m["sha256"]) == 64 and m["size"] > 100_000_000
        assert m["store"].startswith(("models/", "loras"))


def test_a_random_seed_is_picked_and_recorded(registry):
    wf = registry.get("z-image-basic")
    a = params.validate(wf, {"prompt": "x"}, rng=random.Random(1))
    b = params.validate(wf, {"prompt": "x", "seed": -1}, rng=random.Random(1))
    assert a["seed"] == b["seed"] and 0 <= a["seed"] < 2**31
    assert params.validate(wf, {"prompt": "x", "seed": 5})["seed"] == 5


@pytest.mark.parametrize(
    "values,fragment",
    [
        ({}, "prompt is required"),
        ({"prompt": "   "}, "prompt is required"),
        ({"prompt": "x", "cfg": 1.0}, "at least 1.5"),
        ({"prompt": "x", "cfg": 11}, "outside"),
        ({"prompt": "x", "steps": 0}, "outside"),
        ({"prompt": "x", "steps": 8.5}, "integer"),
        ({"prompt": "x", "steps": True}, "integer"),
        ({"prompt": "x", "width": 1000, "height": 1024}, "multiple of 16"),
        ({"prompt": "x", "width": 1024}, "both width and height"),
        ({"prompt": "x", "aspect": "16:9", "width": 1024, "height": 1024}, "not both"),
        ({"prompt": "x", "aspect": "5:4"}, "must be one of"),
        ({"prompt": "x", "width": 2048, "height": 2048, "seed": 1, "nonsense": 1}, "unknown parameter"),
        ({"prompt": "x" * 2001}, "limit is 2000"),
        ({"prompt": "x", "seed": -2}, "seed must be"),
        ({"prompt": "x", "solo": "yes"}, "true or false"),
    ],
)
def test_bad_values_are_named_and_refused(registry, values, fragment):
    with pytest.raises(WorkflowError) as exc:
        params.validate(registry.get("z-image-basic"), values)
    assert exc.value.code == "INVALID_INPUT" and fragment in str(exc.value)


def test_the_pixel_limit_applies_to_custom_sizes(registry):
    wf = registry.get("z-image-basic")
    assert params.validate(wf, {"prompt": "x", "width": 2048, "height": 2048})["aspect"] is None
    wf.spec["limits"]["max_pixels"] = 1_000_000
    with pytest.raises(WorkflowError) as exc:
        params.validate(wf, {"prompt": "x", "width": 1024, "height": 1024})
    assert "over the limit" in str(exc.value)


def test_a_stored_job_revalidates_to_the_same_values(registry):
    """The worker validates the stored values again; the round trip must be stable (aspect + its size)."""
    wf = registry.get("z-image-basic")
    first = params.validate(wf, {"prompt": "x", "aspect": "16:9"})
    job_input, job_params = params.split(wf, first)
    assert job_input == {"prompt": "x"} and job_params["width"] == 1344
    assert params.validate(wf, dict(job_input, **job_params)) == first


def test_the_cfg_floor_holds_even_if_a_registry_entry_lowers_its_minimum(registry):
    wf = registry.get("z-image-basic")
    wf.spec["parameters"]["cfg"]["min"] = 0.0
    with pytest.raises(WorkflowError) as exc:
        params.validate(wf, {"prompt": "x", "cfg": 1.0})
    assert "safety terms" in str(exc.value)


def test_h3_markup_in_the_description_is_refused(registry):
    wf = registry.get("minimax-h3-basic")
    for text in ("<Picture 2> walks in", "[Shot 2] a new scene"):
        with pytest.raises(WorkflowError):
            params.validate(wf, {"prompt": text})


def test_an_unbound_placeholder_is_an_error_not_a_silent_default(registry):
    wf = registry.get("z-image-basic")
    wf.spec["constants"] = [c for c in wf.spec["constants"] if c["input"] != "vae_name"]
    with pytest.raises(WorkflowError) as exc:
        params.bind(wf, params.validate(wf, {"prompt": "x"}), "job-x")
    assert "12.vae_name" in str(exc.value)


def test_describe_exposes_no_node_ids_or_file_names(registry):
    for name in registry.names():
        text = json.dumps(registry.get(name).describe())
        assert '"node"' not in text and "bind" not in text and ".safetensors" not in text and ".json" not in text


def _add(storage, name, spec, graph_file=None, graph=None):
    reg = storage.read_json("workflows/registry.json")
    reg["workflows"][name] = spec
    storage.write_json("workflows/registry.json", reg)
    if graph_file:
        storage.write_json("workflows/%s" % graph_file, graph)


def test_one_broken_entry_does_not_take_the_others_down(tmp_path):
    storage, _ = make_workspace(tmp_path)
    _add(
        storage,
        "broken",
        {"type": "image", "workflow_file": "image/missing.json", "safety": "negative_cfg", "output": {}},
    )
    reg = Registry.load(storage)
    assert "z-image-basic" in reg.names() and "broken" not in reg.names()
    assert reg.problems["broken"].code == "INVALID_REGISTRY"
    with pytest.raises(WorkflowError):
        reg.get("broken")


def test_a_registry_entry_cannot_claim_the_no_negative_profile(tmp_path):
    storage, _ = make_workspace(tmp_path)
    h3_spec = storage.read_json("workflows/registry.json")["workflows"]["minimax-h3-basic"]
    _add(storage, "another-video-model", dict(h3_spec))
    reg = Registry.load(storage)
    assert "another-video-model" not in reg.names()
    assert "screened_prompt" in str(reg.problems["another-video-model"])


def test_a_disabled_workflow_is_not_offered(tmp_path):
    storage, _ = make_workspace(tmp_path)
    reg = storage.read_json("workflows/registry.json")
    reg["workflows"]["test-generation"]["enabled"] = False
    storage.write_json("workflows/registry.json", reg)
    loaded = Registry.load(storage)
    assert "test-generation" not in loaded.names()
    with pytest.raises(WorkflowError) as exc:
        loaded.get("test-generation")
    assert exc.value.code == "UNKNOWN_WORKFLOW"
