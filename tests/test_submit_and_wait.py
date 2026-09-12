"""Coverage for _submit_and_wait's HTTP/polling contract (comfyui_client.py:1472-1540) and the
LCM<->non-LCM mode-switch reset logic (comfyui_client.py:418-478) that it also drives.

The regression this file exists to pin down: status["completed"] never flips True on a
server-side node error, so a naive "poll until completed" loop would spin for the full
timeout_seconds (up to 2400s for the AnimateDiff path) on a job that failed in under a second.
_submit_and_wait checks status_str == "error" explicitly to surface that failure immediately.
"""

import os

import pytest

import comfyui_client as client
from helpers import FakeComfy

# Grabbed before any test monkeypatches client._submit_and_wait (see the LCM tests below), so
# these tests always exercise the real function regardless of what other fixtures replace it with.
_real_submit_and_wait = client._submit_and_wait


def test_node_errors_raise_before_any_history_get(fake_comfy):
    fake_comfy.node_errors = {"3": ["required input missing"]}
    with pytest.raises(RuntimeError, match="workflow validation"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=5)
    # "/history" (no trailing slash, no id) is _reset_if_mode_switch's own last-executed-prompt
    # check and fires before every POST /prompt regardless of node_errors; "/history/<id>" is the
    # completion-polling GET this test guards against ever happening.
    assert not any(path.startswith("/history/") for path, _ in fake_comfy.gets)


def test_execution_error_surfaces_immediately(fake_comfy):
    fake_comfy.history_sequence = [{}, FakeComfy.failed("KSampler", "3", "boom")]
    with pytest.raises(RuntimeError, match=r"KSampler \(node 3\): boom"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=5)


def test_completed_downloads_output_and_queries_view_correctly(fake_comfy, monkeypatch):
    raw_dir = os.environ["COMFYUI_RAW_OUTPUT_DIR"]
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]
    fake_comfy.view_bytes = b"PNGDATA"

    path = _real_submit_and_wait({"3": {}}, timeout_seconds=5)

    assert os.path.dirname(path) == raw_dir
    with open(path, "rb") as f:
        assert f.read() == b"PNGDATA"
    view_calls = [params for p, params in fake_comfy.gets if p == "/view"]
    assert view_calls == [{"filename": "out.png", "subfolder": "", "type": "output"}]


def test_output_node_id_is_respected(fake_comfy):
    fake_comfy.history_sequence = [FakeComfy.done("7", "vid.mp4")]
    path = _real_submit_and_wait({"3": {}}, output_node_id="7", timeout_seconds=5)
    assert os.path.basename(path) == "vid.mp4"


def test_empty_history_forever_times_out_naming_the_prompt_id(fake_comfy):
    fake_comfy.prompt_id = "p42"
    # history_sequence left empty: GET /history/<id> always comes back with nothing for this id.
    with pytest.raises(TimeoutError, match="p42"):
        _real_submit_and_wait({"3": {}}, timeout_seconds=0)


def test_client_id_included_when_supplied(fake_comfy):
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]
    _real_submit_and_wait({"3": {}}, timeout_seconds=5, client_id="abc123")
    (path, payload) = next(p for p in fake_comfy.posted if p[0] == "/prompt")
    assert payload["client_id"] == "abc123"


def test_client_id_omitted_when_not_supplied(fake_comfy):
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.png")]
    _real_submit_and_wait({"3": {}}, timeout_seconds=5)
    (path, payload) = next(p for p in fake_comfy.posted if p[0] == "/prompt")
    assert "client_id" not in payload


# -- LCM <-> non-LCM mode-switch reset ------------------------------------------------------


def _build_workflow_via_animatediff(lcm_preset=None):
    """Get a realistic AnimateDiff workflow dict out of the real submit_generation_animatediff,
    without it ever touching the network: swap _submit_and_wait for a recorder for the duration
    of the call only, then hand back the exact wf dict it built."""
    calls = []

    def fake(wf, output_node_id="9", timeout_seconds=0, client_id=None):
        calls.append(wf)
        return os.path.join("fake", "out.mp4")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(client, "_submit_and_wait", fake)
        client.submit_generation_animatediff(
            prompt="p",
            negative_prompt="n",
            seed=1,
            face_ref_image_filename="face.png",
            filename_prefix="test",
            lcm_preset=lcm_preset,
        )
    return calls[0]


def _history_with_last_prompt(wf):
    """Shape _last_executed_workflow() expects from GET /history?max_items=1: prompt is a
    [num, prompt_id, workflow_dict, extra_data, outputs] tuple/list, workflow_dict at index 2."""
    return {"some-id": {"prompt": [1, "some-id", wf, {}, []]}}


def test_lcm_mode_switch_frees_before_prompt_when_mode_differs(fake_comfy):
    non_lcm_wf = _build_workflow_via_animatediff(lcm_preset=None)
    lcm_wf = _build_workflow_via_animatediff(lcm_preset="lcm_lora")
    assert client._is_lcm_workflow(non_lcm_wf) is False
    assert client._is_lcm_workflow(lcm_wf) is True

    fake_comfy.last_history = _history_with_last_prompt(non_lcm_wf)
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/free", "/prompt"]


def test_same_mode_does_not_free(fake_comfy):
    non_lcm_wf = _build_workflow_via_animatediff(lcm_preset=None)
    fake_comfy.last_history = _history_with_last_prompt(non_lcm_wf)
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(non_lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/prompt"]


def test_fresh_server_does_not_free(fake_comfy):
    lcm_wf = _build_workflow_via_animatediff(lcm_preset="lcm_lora")
    fake_comfy.last_history = {}  # no prior prompt at all
    fake_comfy.history_sequence = [FakeComfy.done("9", "out.mp4")]

    _real_submit_and_wait(lcm_wf, timeout_seconds=5)

    assert fake_comfy.paths("post") == ["/prompt"]


# -- _is_lcm_workflow detectors ---------------------------------------------------------------


def test_is_lcm_workflow_detects_beta_schedule():
    wf = {"2": {"class_type": "ADE_AnimateDiffLoaderGen1", "inputs": {"beta_schedule": "lcm[sqrt_linear]"}}}
    assert client._is_lcm_workflow(wf) is True


def test_is_lcm_workflow_detects_lcm_lora():
    lora_file = next(iter(client.LCM_LORA_FILES))
    for class_type in ("LoraLoaderModelOnly", "LoraLoader"):
        wf = {"61": {"class_type": class_type, "inputs": {"lora_name": lora_file}}}
        assert client._is_lcm_workflow(wf) is True


def test_is_lcm_workflow_detects_lcm_sampler():
    wf = {"3": {"class_type": "KSampler", "inputs": {"sampler_name": "lcm"}}}
    assert client._is_lcm_workflow(wf) is True


def test_is_lcm_workflow_false_otherwise():
    wf = {"3": {"class_type": "KSampler", "inputs": {"sampler_name": "dpmpp_2m"}}}
    assert client._is_lcm_workflow(wf) is False
    assert client._is_lcm_workflow({}) is False


# -- is_server_running / has_node swallow connection errors -------------------------------------


def test_is_server_running_false_when_down(fake_comfy):
    fake_comfy.down = True
    assert client.is_server_running() is False


def test_has_node_false_when_down(fake_comfy):
    fake_comfy.down = True
    assert client.has_node("KSampler") is False


# -- _loader_options parses both combo shapes ----------------------------------------------


def test_loader_options_classic_list_shape(fake_comfy):
    fake_comfy.object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}}}
    }
    assert client._loader_options("CheckpointLoaderSimple", "ckpt_name") == ["a.safetensors", "b.safetensors"]


def test_loader_options_combo_dict_shape(fake_comfy):
    fake_comfy.object_info = {
        "UNETLoader": {"input": {"required": {"unet_name": ["COMBO", {"options": ["x.safetensors"]}]}}}
    }
    assert client._loader_options("UNETLoader", "unet_name") == ["x.safetensors"]


def test_loader_options_malformed_spec_returns_none(fake_comfy):
    fake_comfy.object_info = {"Foo": {"input": {"required": {"bar": "not-a-list-or-combo"}}}}
    assert client._loader_options("Foo", "bar") is None


def test_loader_options_missing_node_returns_none(fake_comfy):
    assert client._loader_options("DoesNotExist", "field") is None
