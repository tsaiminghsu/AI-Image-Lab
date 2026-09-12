"""Every pose library entry must carry the text a caller can fall back to when ControlNet
isn't available.

The library's whole reason to exist is that Pony checkpoints ignore pose words from text alone
(see pose_skeletons.py) - but the reverse case is just as real: Z-Image and SD1.5 have no
ControlNet at all, so for them the *only* way to ask for a pose is the words in `prompt_hint`
and `camera`. 12 of the 31 entries were bootstrapped by running OpenPose over existing renders
and shipped with `prompt_hint: ""` and `camera: "extracted from render"`, which silently
degraded to "no pose requested" on those checkpoints. These tests keep the metadata a required
part of adding a skeleton rather than an optional extra.
"""

import glob
import os

import pose_skeletons
import pytest

_JSON_PATHS = sorted(glob.glob(os.path.join(pose_skeletons.POSES_DIR, "*.json")))
_SLUGS = [os.path.splitext(os.path.basename(p))[0] for p in _JSON_PATHS]


def test_the_library_is_not_empty():
    """Guards the harness: a bad glob would make every parametrized test below vacuous."""
    assert len(_JSON_PATHS) >= 31


@pytest.mark.parametrize("slug", _SLUGS)
def test_every_pose_has_a_usable_prompt_hint(slug):
    meta = pose_skeletons.load_meta(slug)
    hint = meta.get("prompt_hint", "")
    assert hint.strip(), (
        f"{slug}.json has no prompt_hint - on Z-Image/SD1.5 the skeleton PNG is unusable and the "
        "hint is the only thing describing this pose, so the pose would be silently dropped"
    )


@pytest.mark.parametrize("slug", _SLUGS)
def test_every_pose_names_a_real_camera(slug):
    meta = pose_skeletons.load_meta(slug)
    camera = meta.get("camera", "")
    assert camera.strip(), f"{slug}.json has no camera"
    assert camera != "extracted from render", (
        f"{slug}.json still has the bootstrap placeholder as its camera - a 2D skeleton is "
        "ambiguous about viewpoint, which is exactly what this field disambiguates"
    )


@pytest.mark.parametrize("path", _JSON_PATHS, ids=_SLUGS)
def test_pose_json_files_stay_crlf(path):
    """training/poses/*.json is `-text` in .gitattributes (pose_skeletons.py writes them with
    Windows newlines). check.ps1 audits this repo-wide, but an editing script that reads/writes
    without newline="" would rewrite every line here, so pin it next to the metadata tests that
    invite exactly that kind of script."""
    with open(path, "rb") as f:
        data = f.read()
    assert b"\r\n" in data, f"{path} lost its CRLF line endings"
    assert data.count(b"\n") == data.count(b"\r\n"), f"{path} has mixed line endings"
