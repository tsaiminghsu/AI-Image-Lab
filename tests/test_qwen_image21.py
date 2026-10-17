"""Qwen-Image 2.1 as a second still-image model for the concept stills, offline.

It is INTERNAL TEST ONLY (Qwen Research License, non-commercial). What is pinned: the graph keeps the project's
safety rule even though the official recipe runs at cfg 1 (cfg floor, safety negatives reach the sampler through the
combined encoder node, the graph passes the shared workflow checks), the default Z-Image requests are unchanged, and
every image the new model makes carries its licence in its record."""

import copy
import shutil

import concept_stills as cs
import pytest

import comfyui_client as client
import generate_character as gc
import workflow_safety as ws


@pytest.fixture(scope="module")
def bible():
    return cs.load_bible()


@pytest.fixture(scope="module")
def stories():
    return cs.load_stories()


@pytest.fixture
def work(tmp_path):
    root = tmp_path / "ai_short_drama"
    shutil.copytree(
        cs.PROJECT_ROOT, root, ignore=shutil.ignore_patterns("*.png", "*.mp4", "videos", "review.json", "master.json")
    )
    return str(root), cs.load_bible(str(root)), cs.load_stories(str(root))


def fake_submit(tmp_path):
    calls = []

    def submit(**kw):
        path = tmp_path / f"raw_{len(calls)}.png"
        path.write_bytes(b"\x89PNG fake " + str(len(calls)).encode())
        calls.append(kw)
        return str(path)

    submit.calls = calls
    return submit


def graph(**kw):
    args = dict(
        prompt="a coffee shop", negative_prompt=f"{gc.AGE_SAFETY_NEGATIVE}, blurry", seed=7, filename_prefix="x"
    )
    args.update(kw)
    return client.build_qwen_image21_txt2img_workflow(
        args["prompt"],
        args["negative_prompt"],
        args["seed"],
        args["filename_prefix"],
        client.QWEN_IMAGE21_MODELS["qwen_image_2_1"],
        **{k: v for k, v in args.items() if k in ("width", "height", "steps", "cfg")},
    )


# --- the graph ----------------------------------------------------------------------------------------------


def test_the_graph_uses_the_gguf_loader_and_the_combined_encoder_for_both_conditionings():
    wf = graph(width=720, height=1280)
    assert wf["10"]["class_type"] == "UnetLoaderGGUF" and wf["10"]["inputs"]["unet_name"].endswith("Q6_K.gguf")
    assert wf["11"]["inputs"]["type"] == "qwen_image" and wf["12"]["class_type"] == "VAELoader"
    sampler = wf["3"]["inputs"]
    assert sampler["positive"] == ["6", 0] and sampler["negative"] == ["6", 1]  # one node, two outputs
    assert (
        wf["6"]["inputs"]["prompt"] == "a coffee shop"
        and gc.AGE_SAFETY_NEGATIVE in wf["6"]["inputs"]["negative_prompt"]
    )
    assert (wf["5"]["inputs"]["width"], wf["5"]["inputs"]["height"]) == (720, 1280)
    assert wf["9"]["class_type"] == "SaveImage"


def test_cfg_never_goes_below_the_safety_floor_not_even_at_the_official_cfg_1():
    for cfg in (1.0, 0.5, 1.4):
        assert graph(cfg=cfg)["3"]["inputs"]["cfg"] == client.SAFETY_MIN_CFG
    assert graph(cfg=3.0)["3"]["inputs"]["cfg"] == 3.0
    assert client.QWEN_IMAGE21_CFG >= client.SAFETY_MIN_CFG
    assert client.enforce_min_cfg({"3": {"inputs": {"cfg": 1.0}}}) == ["3"]  # the shared guard covers it as well


def test_the_graph_passes_the_shared_workflow_checks():
    wf = graph()
    ws.validate_graph(wf, "9")
    assert ws.check_negative_safety(wf, gc.AGE_SAFETY_NEGATIVE) == 1


def test_a_graph_without_the_safety_terms_or_with_the_positive_output_as_negative_is_refused():
    with pytest.raises(ws.WorkflowRejected, match="missing the required safety terms"):
        ws.check_negative_safety(graph(negative_prompt="blurry"), gc.AGE_SAFETY_NEGATIVE)
    rewired = copy.deepcopy(graph())
    rewired["3"]["inputs"]["negative"] = ["6", 0]
    with pytest.raises(ws.WorkflowRejected, match="not an allowed source"):
        ws.check_negative_safety(rewired, gc.AGE_SAFETY_NEGATIVE)


def test_missing_model_files_are_named_before_anything_is_submitted(monkeypatch):
    monkeypatch.setattr(
        client, "_loader_options", lambda node, field: ["something_else.gguf"] if node == "UnetLoaderGGUF" else None
    )
    assert client.qwen_image21_missing_files() == ["qwen_image_2.1-Q6_K.gguf"]
    with pytest.raises(RuntimeError, match="qwen_image_2.1-Q6_K.gguf"):
        client.submit_txt2img_generation_qwen_image21("p", "n", 1, "x")


# --- the concept stills ---------------------------------------------------------------------------------------


def test_every_qwen_candidate_keeps_the_age_safety_negatives_and_cfg_floor(bible, stories):
    for story in stories:
        for scene in story.data["scenes"]:
            for letter in cs.LETTERS:
                req = cs.build_request(bible, story, scene["scene"], letter, "qwen_image_2_1")
                for term in gc.AGE_SAFETY_NEGATIVE.split(", "):
                    assert term in req["negative_prompt"], (story.id, scene["scene"], term)
                assert req["cfg"] >= client.SAFETY_MIN_CFG and req["steps"] == client.QWEN_IMAGE21_STEPS
                assert req["image_model"] == "qwen_image_2_1" and req["model"] == "qwen_image_2_1"


def test_the_default_requests_are_still_z_image_and_identical_in_text(bible, stories):
    default = cs.build_request(bible, stories[3], 2, "A")
    qwen = cs.build_request(bible, stories[3], 2, "A", "qwen_image_2_1")
    assert default["image_model"] == "z_image" and default["model"] == cs.MODEL_KEY and default["steps"] == 8
    assert (default["prompt"], default["negative_prompt"], default["seed"]) == (
        qwen["prompt"],
        qwen["negative_prompt"],
        qwen["seed"],
    )  # same assembly, same seeds: only the model differs


def test_an_unknown_image_model_is_refused(bible, stories):
    with pytest.raises(gc.UsageError, match="沒有這個圖片模型"):
        cs.build_request(bible, stories[0], 1, "A", "flux")
    with pytest.raises(gc.UsageError):
        cs.generate(bible, stories[:1], image_model="flux", server_up=lambda: True)


def test_a_qwen_image_carries_its_licence_and_the_qwen_graph_in_its_record(work, tmp_path):
    root, bible, stories = work
    submit = fake_submit(tmp_path)
    story = stories[4]
    assert (
        cs.generate(
            bible, [story], scene_numbers={2}, submit=submit, server_up=lambda: True, image_model="qwen_image_2_1"
        )
        == 2
    )
    assert all(call["model"] == "qwen_image_2_1" and call["cfg"] >= client.SAFETY_MIN_CFG for call in submit.calls)
    paths = cs.candidate_paths(story, 2, "A")
    meta = cs._read_json(paths["meta"])
    assert meta["image_model"] == "qwen_image_2_1" and meta["license"] == client.QWEN_IMAGE21_LICENSE
    assert meta["model_files"]["unet"].endswith(".gguf") and meta["steps"] == client.QWEN_IMAGE21_STEPS
    saved = cs._read_json(paths["workflow"])
    assert saved["3"]["inputs"]["negative"] == ["6", 1] and saved["6"]["inputs"]["prompt"] == meta["prompt"]
    assert ws.check_negative_safety(saved, gc.AGE_SAFETY_NEGATIVE) == 1


def test_a_z_image_record_has_no_licence_note(work, tmp_path):
    root, bible, stories = work
    submit = fake_submit(tmp_path)
    cs.generate(bible, [stories[4]], scene_numbers={3}, submit=submit, server_up=lambda: True)
    meta = cs._read_json(cs.candidate_paths(stories[4], 3, "A")["meta"])
    assert meta["image_model"] == "z_image" and "license" not in meta and meta["model"] == cs.MODEL_KEY
