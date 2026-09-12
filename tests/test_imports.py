"""Smoke test: the modules the rest of the suite exercises must import with nothing but
`requests` available - no torch, no gradio, no GPU. If this breaks, CI breaks."""

import importlib

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "comfyui_client",
        "generate_character",
        "pose_skeletons",
        "pose_pack",
        "benchmark",
    ],
)
def test_module_imports_offline(name):
    assert importlib.import_module(name) is not None


def test_no_heavy_dependency_pulled_in():
    """These must stay out of the import graph: the suite has to run on a plain CI python."""
    import sys

    importlib.import_module("generate_character")
    for heavy in ("torch", "gradio", "insightface", "cv2"):
        assert heavy not in sys.modules, f"{heavy} was imported at module scope"
