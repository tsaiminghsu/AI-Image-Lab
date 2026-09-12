"""Runtime graph surgery: the _rewire / _drop_nodes passes in submit_generation_hq and
submit_generation_animatediff.

These two functions serve every feature combination from a single template by re-pointing
consumers at an optional stage's upstream and then deleting the stage. Two things can go
wrong and neither is visible without a GPU:

  * _rewire matches on an exact ["node_id", output_index] pair. Re-saving a template from
    the ComfyUI UI can renumber a node or move an output index, which silently turns the
    rewire into a no-op - the graph still validates locally but samples through the wrong
    branch (e.g. the hires pass gets bypassed, or FaceID stops being applied).
  * _drop_nodes deletes unconditionally. A single consumer left pointing at a dropped node
    comes back from ComfyUI as an opaque `node_errors` payload naming a node id, with
    nothing pointing at the Python-side toggle that actually caused it.

So every case here asserts the *shape* of the resulting graph: no dangling references, the
save node still reachable from the base sampler, dropped nodes genuinely gone, and the exact
edges the code's own comments promise. All offline - captured_submit replaces
_submit_and_wait, so no ComfyUI, no network, no GPU.
"""

import itertools

import pytest

import comfyui_client as client
from helpers import assert_no_dangling, upstream

PROMPT = "SENTINEL_POSITIVE_PROMPT"
NEGATIVE = "SENTINEL_NEGATIVE_PROMPT"
SEED = 1234567
PREFIX = "sentinel_prefix"
FACE_REF = "sentinel_face.png"
POSE_REF = "sentinel_pose.png"
CHAR_LORA = "characters/sentinel.safetensors"


def _yn(flag):
    return "y" if flag else "n"


# ---------------------------------------------------------------------------------------
# 1. submit_generation_hq
# ---------------------------------------------------------------------------------------

HQ_CASES = [
    pytest.param(
        lora,
        ipref,
        pose,
        hires,
        fd,
        id=f"lora={_yn(lora)}-ip={_yn(ipref)}-pose={pose}-hires={_yn(hires)}-fd={_yn(fd)}",
    )
    for lora, ipref, pose, hires, fd in itertools.product(
        [None, CHAR_LORA],
        [None, FACE_REF],
        ["none", "photo", "skeleton"],
        [True, False],
        [True, False],
    )
]


@pytest.mark.parametrize("character_lora, ip_ref, pose, hires, use_facedetailer", HQ_CASES)
def test_hq_graph_surgery(captured_submit, character_lora, ip_ref, pose, hires, use_facedetailer):
    """All 48 HQ toggle combinations produce a connected, fully-wired graph.

    The expected edges below are read off workflow_template_hq.json and the bypass branches
    in submit_generation_hq; if a template edit or a refactor breaks one of them, the failure
    names the exact toggle combination instead of surfacing as a bad image hours later.
    """
    client.submit_generation_hq(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        seed=SEED,
        filename_prefix=PREFIX,
        character_lora=character_lora,
        ip_adapter_image_filename=ip_ref,
        pose_image_filename=POSE_REF if pose != "none" else None,
        pose_is_skeleton=(pose == "skeleton"),
        hires=hires,
        use_facedetailer=use_facedetailer,
    )
    wf = captured_submit[-1]["wf"]

    # The whole point of the bypass dance: nothing may still reference a dropped node.
    assert_no_dangling(wf)

    # SaveImage must survive every combination and must still be fed, transitively, by the
    # base KSampler - a rewire that accidentally short-circuits to a LoadImage would still
    # "work" but would save the reference photo instead of a generated image.
    assert "9" in wf, "SaveImage node 9 was dropped"
    assert "3" in upstream(wf, "9"), "base KSampler 3 no longer reaches SaveImage"

    # -- optional stages are removed, not merely neutered (a LoraLoader with a missing file
    # fails ComfyUI validation even at strength 0, which is why the code drops nodes).
    if character_lora:
        assert wf["14"]["inputs"]["lora_name"] == character_lora
        assert wf["14"]["inputs"]["model"] == ["13", 0], "character LoRA must stack on the style LoRA"
        assert wf["13"]["inputs"]["model"] == ["4", 0]
    else:
        assert "14" not in wf

    if ip_ref:
        assert wf["10"]["inputs"]["image"] == ip_ref
        # The FaceID loader hangs off the last LoRA in the chain, whichever that is.
        assert wf["11"]["inputs"]["model"] == (["14", 0] if character_lora else ["13", 0])
        assert wf["12"]["inputs"]["model"] == ["11", 0]
    else:
        for nid in ("10", "11", "12"):
            assert nid not in wf

    # -- model chain. Every sampler/detailer must read the SAME model source, and that source
    # must be the LoRA-loaded model when a character LoRA is in play - reading ["4", 0] (the
    # raw checkpoint) instead would silently drop the character's identity.
    expected_model = ["12", 0] if ip_ref else (["14", 0] if character_lora else ["13", 0])
    assert wf["3"]["inputs"]["model"] == expected_model
    if hires:
        assert wf["34"]["inputs"]["model"] == expected_model
    if use_facedetailer:
        assert wf["41"]["inputs"]["model"] == expected_model
        assert wf["43"]["inputs"]["model"] == expected_model

    # The CLIP half of the same chain: encoders and detailers must use the LoRA's CLIP, so
    # the LoRA's trigger tokens are actually in the text encoder's vocabulary.
    expected_clip = ["14", 1] if character_lora else ["13", 1]
    for nid in ("6", "7"):
        assert wf[nid]["inputs"]["clip"] == expected_clip
    if use_facedetailer:
        for nid in ("41", "43"):
            assert wf[nid]["inputs"]["clip"] == expected_clip

    # -- ControlNet pose chain.
    if pose == "none":
        # Without a pose the sampler's conditioning must come straight from the encoders;
        # a leftover ["23", *] here is the classic dangling-reference failure.
        assert wf["3"]["inputs"]["positive"] == ["6", 0]
        assert wf["3"]["inputs"]["negative"] == ["7", 0]
        for nid in ("20", "21", "22", "23"):
            assert nid not in wf
    else:
        assert wf["3"]["inputs"]["positive"] == ["23", 0]
        assert wf["3"]["inputs"]["negative"] == ["23", 1]
        assert wf["20"]["inputs"]["image"] == POSE_REF
        if pose == "skeleton":
            # A library skeleton is already an OpenPose render; running the preprocessor over
            # a stick figure produces garbage, so 23 must read LoadImage directly.
            assert "21" not in wf
            assert wf["23"]["inputs"]["image"] == ["20", 0]
        else:
            assert "21" in wf, "a photo pose still needs the OpenposePreprocessor"
            assert wf["23"]["inputs"]["image"] == ["21", 0]

    # The hires pass deliberately skips ControlNet (it only refines an existing composition).
    if hires:
        assert wf["34"]["inputs"]["positive"] == ["6", 0]
        assert wf["34"]["inputs"]["negative"] == ["7", 0]

    # -- hires chain.
    if hires:
        assert wf["31"]["inputs"]["image"] == ["8", 0], "ESRGAN must upscale the base VAEDecode"
        assert wf["35"]["inputs"]["samples"] == ["34", 0]
    else:
        for nid in ("30", "31", "32", "33", "34", "35"):
            assert nid not in wf

    # -- FaceDetailer chain. Note the ordering dependency inside submit_generation_hq: the
    # hires bypass runs first, so with hires off the detailer must already read the base
    # decode. If the two blocks were ever swapped, node 41 would point at a dropped node 35.
    if use_facedetailer:
        assert wf["41"]["inputs"]["image"] == (["35", 0] if hires else ["8", 0])
        assert wf["43"]["inputs"]["image"] == ["41", 0]
        assert wf["9"]["inputs"]["images"] == ["43", 0]
    else:
        for nid in ("40", "41", "42", "43"):
            assert nid not in wf
        assert wf["9"]["inputs"]["images"] == (["35", 0] if hires else ["8", 0])

    # The case the task description calls out explicitly: both refinement stages off means
    # SaveImage hangs directly off the base VAEDecode.
    if not hires and not use_facedetailer:
        assert wf["9"]["inputs"]["images"] == ["8", 0]


# ---------------------------------------------------------------------------------------
# 2. submit_generation_animatediff
# ---------------------------------------------------------------------------------------

AD_CASES = [
    pytest.param(
        lcm,
        motion_lora,
        motion_scale,
        hires,
        upscale_to,
        interp,
        fd,
        id=f"lcm={lcm or 'none'}-mlora={_yn(motion_lora)}-mscale={motion_scale}"
        f"-hires={_yn(hires)}-up={upscale_to}-interp={interp}-fd={_yn(fd)}",
    )
    for lcm, motion_lora, motion_scale, hires, upscale_to, interp, fd in itertools.product(
        [None, "animatelcm", "lcm_lora"],
        [None, client.ANIMATEDIFF_MOTION_LORAS["zoom_in"]],
        [1.0, 0.85],
        [True, False],
        [0, 1024],
        [1, 2],
        [True, False],
    )
]


@pytest.fixture
def stub_has_node(monkeypatch):
    """has_node() hits ComfyUI's /object_info. The RIFE interpolation path checks it before
    injecting node 70 (today from generate_character, historically closer to the client), so
    stub it True: the graph shape under test must not depend on a live server."""
    monkeypatch.setattr(client, "has_node", lambda class_name: True)


@pytest.mark.parametrize("lcm, motion_lora, motion_scale, hires, upscale_to, interp, use_fd", AD_CASES)
def test_animatediff_graph_surgery(
    captured_submit, stub_has_node, lcm, motion_lora, motion_scale, hires, upscale_to, interp, use_fd
):
    """All 192 AnimateDiff toggle combinations produce a connected, fully-wired graph.

    The video path has more moving parts than the still path (three samplers, two upscalers
    sharing one loader, code-injected nodes) and every one of them is only exercised by a
    ~10-minute GPU render, so structural coverage here is the only fast feedback there is.
    """
    client.submit_generation_animatediff(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        seed=SEED,
        face_ref_image_filename=FACE_REF,
        filename_prefix=PREFIX,
        lcm_preset=lcm,
        motion_lora_name=motion_lora,
        motion_scale=motion_scale,
        hires=hires,
        upscale_to=upscale_to,
        interp_multiplier=interp,
        use_facedetailer=use_fd,
    )
    wf = captured_submit[-1]["wf"]

    assert_no_dangling(wf)
    assert "9" in wf, "SaveVideo node 9 was dropped"
    assert "3" in upstream(wf, "9"), "base KSampler 3 no longer reaches SaveVideo"

    # -- LCM: the ORDER of the rewire matters, and the code says so in a comment ("Rewire
    # first, THEN add node 61 - otherwise 61's own model input would point at itself").
    # Assert exactly that: the injected LoraLoaderModelOnly reads the base CheckpointLoader,
    # and the AnimateDiff loader now reads the LCM node instead of the base loader. Building
    # node 61 before the rewire would give 61.model == ["61", 0], a self-loop that ComfyUI
    # reports as an unrelated recursion/validation error.
    if lcm:
        preset = client.ANIMATEDIFF_LCM_PRESETS[lcm]
        assert "61" in wf
        assert wf["61"]["inputs"]["model"] == ["1", 0], "LCM LoRA must take the base checkpoint"
        assert wf["61"]["inputs"]["model"] != ["61", 0]
        assert wf["2"]["inputs"]["model"] == ["61", 0], "AnimateDiff loader must read through the LCM LoRA"
        assert wf["61"]["inputs"]["lora_name"] == preset["lora"]
        assert wf["2"]["inputs"]["model_name"] == preset["motion_module"]
        assert wf["2"]["inputs"]["beta_schedule"] == preset["beta_schedule"]
        expected_sampler, expected_scheduler = preset["sampler"], preset["scheduler"]
        expected_steps = preset["steps"]
    else:
        assert "61" not in wf
        assert wf["2"]["inputs"]["model"] == ["1", 0]
        assert wf["2"]["inputs"]["model_name"] == client.ANIMATEDIFF_MOTION_MODULE
        expected_sampler, expected_scheduler = client.ANIMATEDIFF_SAMPLER, client.ANIMATEDIFF_SCHEDULER
        expected_steps = client.ANIMATEDIFF_STEPS

    # Every sampler in the graph must switch together - the detailer re-samples through the
    # same LCM-patched model and produces mush at karras/20 steps.
    for nid in ("3", "34", "52"):
        if nid in wf:
            assert wf[nid]["inputs"]["sampler_name"] == expected_sampler
            assert wf[nid]["inputs"]["scheduler"] == expected_scheduler
            assert wf[nid]["inputs"]["seed"] == SEED
            # Content-safety floor: at cfg 1.0 ComfyUI skips the negative conditioning, which
            # would silently disable the mandatory age-safety negatives.
            assert wf[nid]["inputs"]["cfg"] >= client.LCM_MIN_CFG
    assert wf["3"]["inputs"]["steps"] == expected_steps

    # -- optional code-injected loaders hang off node 2's optional inputs.
    if motion_lora:
        assert wf["60"]["inputs"]["name"] == motion_lora
        assert wf["2"]["inputs"]["motion_lora"] == ["60", 0]
    else:
        assert "60" not in wf
        assert "motion_lora" not in wf["2"]["inputs"]

    if motion_scale != 1.0:
        assert wf["62"]["inputs"]["float_val"] == pytest.approx(motion_scale)
        assert wf["2"]["inputs"]["scale_multival"] == ["62", 0]
    else:
        # At 1.0 the node is a no-op, so it is not injected at all rather than injected inert.
        assert "62" not in wf
        assert "scale_multival" not in wf["2"]["inputs"]

    # -- FaceID chain is never optional in this path.
    assert wf["11"]["inputs"]["model"] == ["2", 0]
    assert wf["12"]["inputs"]["model"] == ["11", 0]
    for nid in ("3", "34", "51"):
        if nid in wf:
            assert wf[nid]["inputs"]["model"] == ["12", 0], "samplers must run through AnimateDiff+FaceID"

    # -- hires chain.
    if hires:
        assert wf["31"]["inputs"]["image"] == ["8", 0]
        assert wf["35"]["inputs"]["samples"] == ["34", 0]
    else:
        for nid in ("31", "32", "33", "34", "35"):
            assert nid not in wf

    # -- video face detailer. Same ordering dependency as the HQ path: the detailer bypass
    # runs BEFORE the hires bypass, so with both off node 36 must end up on the base decode.
    frames_after_detail = ["52", 0] if use_fd else (["35", 0] if hires else ["8", 0])
    if use_fd:
        detail_input = ["35", 0] if hires else ["8", 0]
        assert wf["50"]["inputs"]["image_frames"] == detail_input
        assert wf["52"]["inputs"]["image_frames"] == detail_input
        assert wf["52"]["inputs"]["segs"] == ["50", 0]
    else:
        for nid in ("40", "50", "51", "52"):
            assert nid not in wf

    # -- final per-frame ESRGAN upscale. With the default 512x512 base, the sampled frames are
    # 768x768 with hires (eff_scale capped at ANIMATEDIFF_HIRES_MAX_PIXELS) and 512x512
    # without, so upscale_to=1024 always applies and upscale_to=0 always drops the stage.
    if upscale_to:
        assert wf["36"]["inputs"]["image"] == frames_after_detail
        assert wf["37"]["inputs"]["image"] == ["36", 0]
        assert wf["37"]["inputs"]["width"] == 1024 and wf["37"]["inputs"]["height"] == 1024
        video_source = ["37", 0]
    else:
        for nid in ("36", "37"):
            assert nid not in wf
        video_source = frames_after_detail

    # The shared upscale-model loader must be present exactly while something still reads it.
    if hires or upscale_to:
        assert "30" in wf
        users = [nid for nid in ("31", "36") if nid in wf]
        assert users, "node 30 kept but no consumer"
        for nid in users:
            assert wf[nid]["inputs"]["upscale_model"] == ["30", 0]
    else:
        assert "30" not in wf, "UpscaleModelLoader left in the graph with nothing downstream using it"

    # -- RIFE interpolation is spliced in immediately before CreateVideo, taking over whatever
    # CreateVideo used to read.
    if interp > 1:
        assert wf["70"]["class_type"] == client.RIFE_NODE
        assert wf["70"]["inputs"]["frames"] == video_source
        assert wf["70"]["inputs"]["multiplier"] == interp
        assert wf["90"]["inputs"]["images"] == ["70", 0]
    else:
        assert "70" not in wf
        assert wf["90"]["inputs"]["images"] == video_source

    # Default fps keeps the clip duration constant when interpolating.
    assert wf["90"]["inputs"]["fps"] == pytest.approx(float(client.ANIMATEDIFF_FPS * interp))
    assert wf["9"]["inputs"]["video"] == ["90", 0]
    assert wf["9"]["inputs"]["filename_prefix"] == PREFIX


# ---------------------------------------------------------------------------------------
# 3. every simple submit path
# ---------------------------------------------------------------------------------------

# (function name, extra required kwargs, does the template have an EmptyLatentImage-style
# node "5"?). The three positional-ish basics - prompt, negative_prompt, seed - are common
# to all of them and are filled in by the test.
SIMPLE_PATHS = [
    ("submit_generation", {"ip_adapter_image_filename": FACE_REF}, True),
    (
        "submit_img2img_generation",
        {"ip_adapter_image_filename": FACE_REF, "init_image_filename": "init.png", "denoise": 0.35},
        # img2img has no latent node at all: VAEEncode (21) encodes the init image, so the
        # output size is whatever that image already is. Nothing to assert BATCH_SIZE on.
        False,
    ),
    ("submit_generation_with_pose", {"ip_adapter_image_filename": FACE_REF, "pose_image_filename": POSE_REF}, True),
    ("submit_generation_with_facedetailer", {"ip_adapter_image_filename": FACE_REF}, True),
    ("submit_generation_with_facedetailer_mediapipe", {"ip_adapter_image_filename": FACE_REF}, True),
    ("submit_txt2img_generation", {}, True),
    ("submit_txt2img_generation_sd15", {"checkpoint": "realisticVisionV60B1_v51HyperVAE.safetensors"}, True),
    ("submit_txt2img_generation_zimage", {}, True),
    ("submit_generation_hq", {}, True),
]


@pytest.fixture
def offline_loader_options(monkeypatch):
    """_loader_options asks a running ComfyUI which model files it can see. Offline it would
    spend three connection timeouts per Z-Image call before giving up; returning None is the
    same 'server can't say' answer it produces on a refused connection."""
    monkeypatch.setattr(client, "_loader_options", lambda node, field: None)


@pytest.mark.parametrize(
    "func_name, extra, has_latent_node",
    SIMPLE_PATHS,
    ids=[name for name, _, _ in SIMPLE_PATHS],
)
def test_simple_paths_set_prompt_negative_and_seed(
    captured_submit, offline_loader_options, func_name, extra, has_latent_node
):
    """Each submit_* path writes the caller's prompt/negative/seed into the template.

    Sentinel strings catch the mistakes that a smoke test cannot: writing the prompt into the
    negative encoder (a real risk - the templates only differ by which node id is which), or
    leaving the REPLACE_* placeholder in place because a template renumbered its encoders.
    """
    func = getattr(client, func_name)
    func(prompt=PROMPT, negative_prompt=NEGATIVE, seed=SEED, filename_prefix=PREFIX, **extra)
    wf = captured_submit[-1]["wf"]

    assert_no_dangling(wf)
    assert wf["6"]["inputs"]["text"] == PROMPT, "node 6 must be the POSITIVE encoder"
    assert wf["7"]["inputs"]["text"] == NEGATIVE, "node 7 must be the NEGATIVE encoder"
    assert wf["3"]["inputs"]["seed"] == SEED, "seed must reach the base sampler for reproducibility"

    if has_latent_node:
        # One RTX 2070 with 8GB: always one image per job, never a multi-image batch.
        assert wf["5"]["inputs"]["batch_size"] == client.BATCH_SIZE
    else:
        assert "5" not in wf, "template gained a latent node - drop the skip and assert BATCH_SIZE"


# ---------------------------------------------------------------------------------------
# 4. rounding helpers
# ---------------------------------------------------------------------------------------
#
# These decide the actual pixel dimensions handed to the samplers and the video encoder. Each
# has a documented floor and rounds to the NEAREST multiple (not down), so an off-by-one-step
# regression would only show up as a ComfyUI shape error mid-render.


@pytest.mark.parametrize(
    "value, expected",
    [
        (1024, 1024),  # already a multiple of 16
        (1020, 1024),  # 63.75 -> 64: rounds UP to nearest, not down
        (1028, 1024),  # 64.25 -> 64: rounds DOWN to nearest
        (1, 256),  # below the documented floor
        (0, 256),
        (255, 256),  # 15.94 -> 16, which also happens to be the floor
        (300, 304),
    ],
)
def test_round16(value, expected):
    """EmptySD3LatentImage wants multiples of 16; the helper also floors at 256."""
    assert client._round16(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1024, 1024),
        (1000, 1024),  # 15.6 -> 16
        (1040, 1024),  # 16.25 -> 16
        (832 / 1.5, 576),  # the real HIRES_SCALE first-pass computation: 554.7 -> 8.67 -> 9
        (31, 64),  # below the floor
        (0, 64),
    ],
)
def test_round64(value, expected):
    """SDXL UNet + VAE both want /64, floored at 64."""
    assert client._round64(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (768, 768),
        (512 * 1.5, 768),  # the AnimateDiff hires size
        (767, 768),  # 95.875 -> 96
        (769, 768),  # 96.125 -> 96
        (3, 8),  # below the floor
        (0, 8),
    ],
)
def test_round8(value, expected):
    """SD1.5 VAE latent grid wants /8, floored at 8."""
    assert client._round8(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1024, 1024),
        (1023.9, 1024),
        (1020.2, 1020),
        (1, 2),  # 0.5 -> 0 -> clamped to the floor
        (0, 2),
        (2, 2),
    ],
)
def test_round_even(value, expected):
    """h264/yuv420p needs even frame dimensions, floored at 2."""
    assert client._round_even(value) == expected


@pytest.mark.parametrize(
    "func, value, expected",
    [
        (client._round64, 96, 128),  # 1.5 -> 2
        (client._round64, 160, 128),  # 2.5 -> 2
        (client._round8, 20, 16),  # 2.5 -> 2
        (client._round8, 28, 32),  # 3.5 -> 4
    ],
)
def test_rounding_helpers_use_bankers_rounding_on_exact_ties(func, value, expected):
    """All four helpers are built on Python's round(), which breaks exact .5 ties towards the
    EVEN multiple rather than always up. Pinned here so that a well-meaning switch to
    math.floor(x + 0.5) - which would change these four results - is caught deliberately
    rather than showing up as a one-step size difference in a rendered frame.
    """
    assert func(value) == expected
