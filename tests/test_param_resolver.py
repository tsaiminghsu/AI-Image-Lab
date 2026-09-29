"""Pins clamp-don't-trust: a bad suggestion from our own UI, LLM or HTTP caller is normalised into
something drawable, never turned into an error.

The companion property, and the one that matters for safety, is that this is the *opposite* of
what worker/jobs.py does. The worker is a trust boundary receiving untrusted JSON off the network
and it raises (`_get_int` at worker/jobs.py:71); clamping there would silently accept a hostile
payload and run it on a GPU somebody is paying for.
test_the_worker_still_raises_rather_than_clamping is what stops someone "unifying" the two
behaviours in the wrong direction.
"""

import importlib
import os
import sys

import generate_character as gc
import param_resolver as pr
import pytest

WORKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker")


def _import_worker_jobs():
    sys.path.insert(0, WORKER_DIR)
    try:
        return importlib.import_module("jobs")
    finally:
        sys.path.remove(WORKER_DIR)


# --- dimensions: snap to 32, clamp, then fit the pixel budget -----------------------------------


@pytest.mark.parametrize(
    "given,expected",
    [
        (1000, 992),  # nearest multiple of 32, rounding down
        (1008, 1024),  # nearest multiple of 32, rounding up
        (1024, 1024),  # already aligned
        (16, 256),  # below the floor
        (200, 256),  # below the floor after snapping
        (4000, 1536),  # above the ceiling
        (-512, 256),  # negative clamps to the floor rather than raising
    ],
)
def test_sides_snap_to_32_and_clamp(given, expected):
    assert pr.snap_side(given) == expected


@pytest.mark.parametrize("junk", [None, "", "wide", float("nan"), float("inf"), [1024]])
def test_unusable_sides_fall_back_instead_of_raising(junk):
    """A chat model that answers "wide" instead of a number still gets a picture."""
    assert pr.snap_side(junk, 768) == 768


def test_a_nonsense_fallback_still_produces_a_legal_side():
    assert pr.snap_side("wide", "also wide") == pr.MIN_SIDE


def test_pixel_budget_shrinks_the_longer_side():
    """Shrinking the longer side moves the aspect ratio toward square, which distorts a framing
    less than squashing the short side would."""
    width, height = pr.fit_pixel_budget(1536, 1536)
    assert width == 1536, "the short side is left alone"
    assert height <= 1024
    assert width * height <= pr.PIXEL_BUDGET


def test_pixel_budget_leaves_a_request_that_already_fits():
    assert pr.fit_pixel_budget(1024, 1024) == (1024, 1024)


def test_pixel_budget_result_is_still_a_multiple_of_32():
    for width, height in [(1536, 1536), (1504, 1312), (1280, 1280)]:
        fitted_w, fitted_h = pr.fit_pixel_budget(width, height)
        assert fitted_w % 32 == 0 and fitted_h % 32 == 0
        assert fitted_w * fitted_h <= pr.PIXEL_BUDGET


def test_a_square_breaks_the_tie_deterministically():
    """Same input, same output - otherwise two identical requests produce two different seeds of
    truth about what was asked for."""
    assert pr.fit_pixel_budget(1536, 1536) == pr.fit_pixel_budget(1536, 1536)


# --- seed ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given,expected",
    [
        (0, 0),
        (9000, 9000),
        (-1, 0),
        (2**33, pr.SEED_MAX),
        (pr.SEED_MAX, pr.SEED_MAX),
        ("not a seed", 0),
        (None, 0),
        (12.0, 12),
    ],
)
def test_seed_clamps_into_uint32(given, expected):
    assert pr.resolve_seed(given) == expected


def test_minus_one_is_not_a_random_sentinel():
    """This repo randomises by *absence* (image_api.py:191, worker/jobs.py:129), never by -1.
    Inventing a sentinel here would make an explicitly passed seed non-reproducible."""
    assert pr.resolve_seed(-1) == 0
    assert pr.resolve_seed(-1) == pr.resolve_seed(-1)


# --- prompt -------------------------------------------------------------------------------------


def test_prompt_is_stripped_and_left_alone_when_short():
    assert pr.resolve_prompt("  portrait photo  ") == "portrait photo"


def test_prompt_truncates_to_the_project_limit():
    long_prompt = "word " * 1000
    assert len(pr.resolve_prompt(long_prompt)) <= gc.MAX_PROMPT_CHARS


def test_prompt_truncation_prefers_a_word_boundary():
    """Cutting mid-token turns "photorealistic" into "photoreal", which CLIP encodes as something
    else entirely."""
    prompt = ("tag, " * 100) + "photorealistic"
    cut = pr.resolve_prompt(prompt, limit=len(prompt) - 5)
    assert not cut.endswith("photoreal")
    assert cut.endswith("tag")


def test_prompt_falls_back_to_a_hard_cut_for_one_enormous_token():
    """No whitespace anywhere near the boundary means backing up would throw away the whole
    prompt; a hard cut is the honest answer."""
    assert len(pr.resolve_prompt("x" * 500, limit=100)) == 100


def test_prompt_handles_none():
    assert pr.resolve_prompt(None) == ""


# --- the whole request --------------------------------------------------------------------------


def test_resolve_normalises_a_hostile_looking_request_without_raising():
    resolved = pr.resolve(prompt="  a portrait  ", width=4000, height=3000, seed=-7)
    assert resolved["prompt"] == "a portrait"
    assert resolved["width"] % 32 == 0 and resolved["height"] % 32 == 0
    assert resolved["width"] * resolved["height"] <= pr.PIXEL_BUDGET
    assert resolved["seed"] == 0


def test_resolve_uses_the_per_family_defaults_when_sides_are_absent():
    """SD1.5 is 512x768 (comfyui_client.py:118), not the SDXL 1024 square."""
    resolved = pr.resolve(prompt="x", default_width=512, default_height=768)
    assert (resolved["width"], resolved["height"]) == (512, 768)


def test_resolve_returns_only_the_keys_it_resolved():
    """A caller splats this into params; carrying Nones for things it never asked about would put
    unowned keys into a record that job_contracts validates strictly."""
    assert set(pr.resolve(prompt="x")) == {"prompt", "width", "height", "seed"}


def test_resolved_params_are_accepted_by_the_contract():
    """The two modules have to agree on key names or the resolver's output is unstorable."""
    import job_contracts as jc

    record = jc.new_record(
        job_id="j",
        owner_id="local",
        mode="txt2img_hq",
        media_kind="image",
        catalog_id="c",
        backend="local",
        params=pr.resolve(prompt="portrait", width=1000, height=1000, seed=3),
        now=0.0,
    )
    assert record["params"]["width"] == 992


# --- the direction that must never be unified ---------------------------------------------------


def test_the_worker_still_raises_rather_than_clamping():
    """worker/jobs.py sits on a network trust boundary. If a future refactor made it clamp like
    this module does, a hostile payload would be silently accepted instead of rejected."""
    jobs = _import_worker_jobs()
    with pytest.raises(jobs.JobInputError):
        jobs._get_int({"frames": 10**9}, "frames", 16, 8, 16)
    with pytest.raises(jobs.JobInputError):
        jobs._get_int({"frames": "sixteen"}, "frames", 16, 8, 16)
