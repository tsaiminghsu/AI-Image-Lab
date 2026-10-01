"""Prompt and parameter handling for the Z-Image Colab skill.

Two promises are pinned here. The user's words go to the model as written (user intent > prompt
optimization: nothing is appended, restyled or reworded). And the project's safety path is intact on a
remote GPU too: the age-safety negatives are in every job and cfg can never reach the value at which
ComfyUI skips the negative prompt."""

import random

import pytest
from zimage_colab_fakes import make_config, remote, z


@pytest.fixture
def config(tmp_path):
    return make_config(tmp_path)


# --- user intent ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "a premium tech product ad, black background, product centered",
        "white background, a single ceramic mug, soft studio light",
        "fixed camera, a street at dusk",
        "桌上的一杯咖啡，白色背景",
    ],
)
def test_the_prompt_reaches_the_model_exactly_as_written(config, prompt):
    job = z.build_job(config, prompt=prompt)
    assert job["prompt"] == job["user_prompt"] == prompt
    assert job["graph"]["6"]["inputs"]["text"] == prompt


def test_no_style_or_quality_words_are_added(config):
    job = z.build_job(config, prompt="white background, a ceramic mug")
    for word in ("cinematic", "dark", "photorealistic", "masterpiece", "score_9", "dynamic"):
        assert word not in job["prompt"]
    assert z.gc.REALISTIC_NEGATIVE not in job["negative_prompt"]  # no style negative either


def test_surrounding_whitespace_is_the_only_thing_trimmed(config):
    assert z.build_job(config, prompt="  a mug \n")["prompt"] == "a mug"


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t", None])
def test_an_empty_prompt_is_refused(config, prompt):
    with pytest.raises(z.UsageError, match="empty"):
        z.build_job(config, prompt=prompt)


def test_an_overlong_prompt_is_refused(config):
    with pytest.raises(z.UsageError, match="characters"):
        z.build_job(config, prompt="a" * (z.gc.MAX_PROMPT_CHARS + 1))


# --- safety: the negatives and the cfg floor -------------------------------------------------------


@pytest.mark.parametrize("solo,allow_text", [(False, False), (True, False), (False, True), (True, True)])
def test_the_age_safety_negative_is_in_every_job(config, solo, allow_text):
    job = z.build_job(config, prompt="two people at a table", solo=solo, allow_text=allow_text, negative="logo")
    negative = job["graph"]["7"]["inputs"]["text"]
    assert z.gc.AGE_SAFETY_NEGATIVE in negative
    assert z.gc.SAFE_CONTENT_NEGATIVE in negative
    assert negative.endswith("logo")  # the user's own negative is appended, never a replacement
    terms = {t.strip() for t in negative.split(",")}
    assert ("text" in terms) is (not allow_text)
    assert (z.gc.SOLO_NEGATIVE in negative) is solo


def test_the_default_job_allows_several_people_but_not_lettering(config):
    job = z.build_job(config, prompt="a team around a whiteboard")
    assert job["solo"] is False and job["allow_text"] is False
    assert "multiple people" not in job["negative_prompt"]


def test_the_remote_copies_of_the_safety_constants_match_the_project(config):
    """remote/zimage_worker.py runs on the VM without the repo, so it carries its own copies. If the
    project's constants change, this fails until the worker's copies follow."""
    assert ", ".join(remote.AGE_TERMS) == z.gc.AGE_SAFETY_NEGATIVE
    assert remote.MIN_CFG == z.client.SAFETY_MIN_CFG == 1.5


def test_the_age_terms_themselves_are_pinned():
    assert remote.AGE_TERMS == ("child", "children", "kid", "minor", "teen", "teenager", "underage", "young girl")


@pytest.mark.parametrize("cfg", [1.0, 1.49, 0, -1, 10.5, True, "2.0", float("nan")])
def test_cfg_outside_the_safe_range_is_refused(config, cfg):
    with pytest.raises(z.UsageError, match="cfg"):
        z.build_job(config, prompt="a mug", cfg=cfg)


def test_cfg_at_the_floor_is_accepted_and_kept(config):
    job = z.build_job(config, prompt="a mug", cfg=1.5)
    assert job["cfg"] == job["graph"]["3"]["inputs"]["cfg"] == 1.5


# --- parameters ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aspect,size",
    [("1:1", (1024, 1024)), ("16:9", (1344, 768)), ("9:16", (768, 1344)), ("4:3", (1152, 896)), ("2:3", (832, 1216))],
)
def test_aspect_presets(config, aspect, size):
    job = z.build_job(config, prompt="a mug", aspect=aspect)
    assert (job["width"], job["height"]) == size


def test_every_preset_fits_the_latent_grid_and_the_pixel_limit(config):
    for width, height in z.ASPECTS.values():
        assert width % z.SIZE_MULTIPLE == 0 and height % z.SIZE_MULTIPLE == 0
        assert width * height <= config["limits"]["max_pixels"]


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"aspect": "21:9"}, "unknown aspect"),
        ({"aspect": "16:9", "width": 1024, "height": 1024}, "either --aspect or --width/--height"),
        ({"width": 1024}, "both --width and --height"),
        ({"width": 1000, "height": 1024}, "multiple of 16"),
        ({"width": 256, "height": 1024}, "outside"),
        ({"width": 4096, "height": 1024}, "outside"),
        ({"width": 1024.0, "height": 1024}, "integer"),
        ({"width": 2048, "height": 2048 + 16}, "outside"),
    ],
)
def test_invalid_sizes(config, kwargs, needle):
    with pytest.raises(z.UsageError, match=needle):
        z.build_job(config, prompt="a mug", **kwargs)


def test_the_pixel_limit_comes_from_the_config(config):
    config["limits"]["max_pixels"] = 1024 * 1024
    with pytest.raises(z.UsageError, match="limits.max_pixels"):
        z.build_job(config, prompt="a mug", width=1536, height=1024)
    assert z.build_job(config, prompt="a mug", width=1024, height=1024)["width"] == 1024


@pytest.mark.parametrize("steps", [0, 51, 8.0, "8", True])
def test_invalid_steps(config, steps):
    with pytest.raises(z.UsageError, match="steps"):
        z.build_job(config, prompt="a mug", steps=steps)


@pytest.mark.parametrize("seed", [-2, 2**32, 1.5, "7"])
def test_invalid_seeds(config, seed):
    with pytest.raises(z.UsageError, match="seed"):
        z.build_job(config, prompt="a mug", seed=seed)


def test_seed_minus_one_picks_a_seed_and_records_it(config):
    job = z.build_job(config, prompt="a mug", seed=-1, rng=random.Random(7))
    again = z.build_job(config, prompt="a mug", seed=-1, rng=random.Random(7))
    assert job["seed"] == again["seed"] == job["graph"]["3"]["inputs"]["seed"]
    assert 0 <= job["seed"] <= z.SEED_MAX and job["seed_requested"] == -1


def test_a_given_seed_is_used_as_is(config):
    assert z.build_job(config, prompt="a mug", seed=0)["seed"] == 0
    assert z.build_job(config, prompt="a mug", seed=z.SEED_MAX)["graph"]["3"]["inputs"]["seed"] == z.SEED_MAX
