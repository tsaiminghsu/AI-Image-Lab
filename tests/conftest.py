"""Shared pytest fixtures.

Everything under tests/ is offline by design: no GPU, no ComfyUI process, no torch, no
gradio. The modules under test import only `requests` at module scope (PIL is imported
lazily inside gen_gif), which is what lets the same suite run in CI on a plain Python.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(REPO_ROOT, "training")

# pyproject.toml already sets pythonpath, but do it defensively so `python -m pytest
# tests/test_x.py` from any cwd and IDE runners behave identically. Import the siblings as
# TOP-LEVEL modules (not training.comfyui_client) to match how every script and the RunPod
# worker import them - otherwise monkeypatching `client` would not be seen by
# generate_character's own `import comfyui_client as client`.
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

import comfyui_client as client  # noqa: E402


@pytest.fixture
def isolated_variants(monkeypatch, tmp_path):
    """Fresh full/quant selection state: no CLI override, no env, settings file and
    checkpoints dir in tmp, and no server round-trips (so _quant_present falls back to the
    filesystem). Yields the fake checkpoints dir so a test can touch quant files into it."""
    monkeypatch.setattr(client, "_variant_override", None)
    monkeypatch.delenv("MODEL_VARIANT", raising=False)
    monkeypatch.delenv("MODEL_VARIANT_ALLOW_CONTROLLORA", raising=False)
    monkeypatch.setattr(client, "MODEL_VARIANTS_FILE", str(tmp_path / "settings" / "model_variants.json"))
    monkeypatch.setattr(client, "QUANT_UNSUPPORTED", {})
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    monkeypatch.setattr(client, "CHECKPOINTS_DIR", str(ckpt_dir))
    monkeypatch.setattr(client, "_loader_options", lambda node, field: None)
    client._variant_notices_shown.clear()
    client.LAST_VARIANT_NOTICES[:] = []
    yield ckpt_dir
    client._variant_notices_shown.clear()
    client.LAST_VARIANT_NOTICES[:] = []


@pytest.fixture
def captured_submit(monkeypatch):
    """Replace _submit_and_wait so the submit_* functions hand back the workflow they built
    instead of talking to ComfyUI. Returns the list of recorded calls."""
    calls = []

    def fake(wf, output_node_id="9", timeout_seconds=0, client_id=None):
        calls.append(
            {"wf": wf, "output_node_id": output_node_id, "timeout_seconds": timeout_seconds, "client_id": client_id}
        )
        return os.path.join("fake", "out.png")

    monkeypatch.setattr(client, "_submit_and_wait", fake)
    return calls


@pytest.fixture
def no_sleep(monkeypatch):
    """Make comfyui_client's retry backoff instant and recorded instead of real.

    The retry budget is bounded by max(wall clock, seconds slept), so with sleep stubbed out
    the "seconds slept" half is what decides when a test's retry loop gives up - the returned
    list is therefore both the assertion surface and the thing that keeps the suite at ~2s.
    """
    slept = []
    monkeypatch.setattr(client.time, "sleep", slept.append)
    return slept


@pytest.fixture
def fake_comfy(monkeypatch, tmp_path):
    """Swap comfyui_client's `requests` for a fake ComfyUI server, and make polling instant."""
    from helpers import FakeComfy

    fc = FakeComfy()
    monkeypatch.setattr(client, "requests", fc)
    monkeypatch.setattr(client, "POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(client, "MODE_SWITCH_RESET_WAIT_SECONDS", 0)
    monkeypatch.setenv("COMFYUI_RAW_OUTPUT_DIR", str(tmp_path / "raw"))
    return fc
