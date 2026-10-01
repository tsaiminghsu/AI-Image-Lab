"""The workflow is fixed; a job only fills parameters into it. These tests pin that: the graph a job
carries is the project's Z-Image template with nothing but the documented fields changed."""

import copy
import json
from pathlib import Path

import pytest
from zimage_colab_fakes import make_config, remote, z

TEMPLATE_DIR = Path(z.TRAINING_DIR)

# (node id, input name) pairs a job may set. Everything else must equal the template byte for byte.
PARAMETER_FIELDS = {
    ("6", "text"),
    ("7", "text"),
    ("5", "width"),
    ("5", "height"),
    ("5", "batch_size"),
    ("3", "seed"),
    ("3", "steps"),
    ("3", "cfg"),
    ("3", "sampler_name"),
    ("3", "scheduler"),
    ("13", "shift"),
    ("9", "filename_prefix"),
    ("10", "unet_name"),
    ("11", "clip_name"),
    ("12", "vae_name"),
}


def template(name="z-image-txt2img"):
    return json.loads((TEMPLATE_DIR / z.WORKFLOWS[name]["template"]).read_text(encoding="utf-8"))


def changed_fields(graph, base):
    return {
        (node_id, key)
        for node_id, node in graph.items()
        for key, value in node["inputs"].items()
        if base[node_id]["inputs"].get(key) != value
    }


def test_every_registered_workflow_has_a_template_and_a_save_node():
    for name, spec in z.WORKFLOWS.items():
        graph = template(name)
        assert graph[spec["save_node"]]["class_type"] == "SaveImage", name
    assert z.DEFAULT_WORKFLOW in z.WORKFLOWS


def test_a_job_changes_only_parameters_never_the_graph(tmp_path):
    config = make_config(tmp_path)
    job = z.build_job(
        config, prompt="a ceramic mug on a white table", aspect="16:9", steps=10, cfg=2.5, seed=1234, negative="logo"
    )
    base = template()
    graph = job["graph"]
    assert set(graph) == set(base)
    for node_id, node in graph.items():
        assert node["class_type"] == base[node_id]["class_type"]
        assert set(node["inputs"]) == set(base[node_id]["inputs"])
        for key, value in node["inputs"].items():
            if isinstance(base[node_id]["inputs"][key], list):  # a link to another node
                assert value == base[node_id]["inputs"][key], (node_id, key)
    assert changed_fields(graph, base) <= PARAMETER_FIELDS
    assert z.workflow_contracts.check(z.WORKFLOWS["z-image-txt2img"]["template"], graph) == []


def test_the_job_parameters_land_in_the_graph(tmp_path):
    config = make_config(tmp_path)
    job = z.build_job(config, prompt="a ceramic mug", aspect="16:9", steps=10, cfg=2.5, seed=1234, job_id="zimg-t1")
    g = job["graph"]
    assert g["6"]["inputs"]["text"] == job["prompt"] == "a ceramic mug"
    assert g["7"]["inputs"]["text"] == job["negative_prompt"]
    assert (g["5"]["inputs"]["width"], g["5"]["inputs"]["height"]) == (1344, 768)
    assert g["5"]["inputs"]["batch_size"] == 1  # more images = more jobs, so each has its own seed and record
    assert (g["3"]["inputs"]["seed"], g["3"]["inputs"]["steps"], g["3"]["inputs"]["cfg"]) == (1234, 10, 2.5)
    assert g["9"]["inputs"]["filename_prefix"] == "zimg-t1"


def test_the_graph_names_the_models_the_worker_installs(tmp_path):
    config = make_config(tmp_path)
    g = z.build_job(config, prompt="a ceramic mug")["graph"]
    by_role = {f["role"]: f["name"] for f in config["model"]["files"]}
    assert g["10"]["inputs"]["unet_name"] == by_role["unet"]
    assert g["11"]["inputs"]["clip_name"] == by_role["text_encoder"]
    assert g["12"]["inputs"]["vae_name"] == by_role["vae"]


def test_defaults_are_the_tested_z_image_settings(tmp_path):
    job = z.build_job(make_config(tmp_path), prompt="a ceramic mug")
    assert (job["width"], job["height"]) == (1024, 1024)
    assert job["steps"] == z.client.ZIMAGE_STEPS and job["cfg"] == z.client.ZIMAGE_CFG
    assert (job["sampler"], job["scheduler"], job["shift"]) == ("res_multistep", "simple", 3.0)


def test_an_unknown_workflow_is_refused(tmp_path):
    with pytest.raises(z.UsageError, match="unknown workflow"):
        z.build_job(make_config(tmp_path), prompt="a ceramic mug", workflow="z-image-img2img")


def test_the_remote_payload_is_what_the_worker_checks(tmp_path):
    config = make_config(tmp_path)
    job = z.build_job(config, prompt="a ceramic mug", aspect="3:4")
    payload = z.remote_payload(job, config)
    remote.check_payload(payload)  # raises on a payload the worker would reject
    assert payload["graph"] == job["graph"] and payload["save_node"] == "9"
    assert (payload["width"], payload["height"], payload["format"]) == (896, 1152, "png")


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda g: g["3"]["inputs"].update(cfg=1.0), "UNSAFE_JOB"),  # ComfyUI skips the negative at cfg 1.0
        (lambda g: g["7"]["inputs"].update(text="lowres, blurry"), "UNSAFE_JOB"),
        (lambda g: g["3"]["inputs"].update(negative=["99", 0]), "UNSAFE_JOB"),
        (lambda g: g.pop("3"), "INVALID_JOB"),
        (lambda g: g.update({"99": "not a node"}), "INVALID_JOB"),
    ],
)
def test_a_tampered_graph_is_refused_at_the_trust_boundary(tmp_path, mutate, code):
    graph = copy.deepcopy(z.build_job(make_config(tmp_path), prompt="a ceramic mug")["graph"])
    mutate(graph)
    with pytest.raises(remote.WorkerError) as err:
        remote.check_graph(graph)
    assert err.value.code == code


def test_node_check_reports_a_missing_class_and_an_undeclared_input(tmp_path):
    graph = z.build_job(make_config(tmp_path), prompt="a ceramic mug")["graph"]
    info = {
        node["class_type"]: {"input": {"required": {name: ["ANY"] for name in node["inputs"]}}}
        for node in graph.values()
    }
    assert remote.check_against_object_info(graph, info) == []
    del info["ModelSamplingAuraFlow"]
    del info["KSampler"]["input"]["required"]["scheduler"]
    problems = remote.check_against_object_info(graph, info)
    assert any("ModelSamplingAuraFlow is not installed" in p for p in problems)
    assert any("'scheduler' is not declared" in p for p in problems)
