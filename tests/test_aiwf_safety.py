"""The platform's safety constants are pinned copies (the core runs on the compute runtime without this
repository). These tests are what keeps the copies honest: each one is compared with the project's constant
AND with a literal written out here - comparing only "is it passed along" would turn every assertion into a
tautology the day someone empties the constant.
"""

import itertools

import comfyui_client as client
import generate_character as gc
import pytest
from controller import WorkflowError, safety
from minimax_h3_fakes import h3

AGE_TERMS = ("child", "children", "kid", "minor", "teen", "teenager", "underage", "young girl")


def test_the_age_terms_are_the_projects_and_are_spelled_out():
    assert safety.AGE_TERMS == AGE_TERMS
    assert safety.AGE_SAFETY_NEGATIVE == gc.AGE_SAFETY_NEGATIVE == ", ".join(AGE_TERMS)
    assert safety.SAFE_CONTENT_NEGATIVE == gc.SAFE_CONTENT_NEGATIVE
    assert safety.SAFE_CONTENT_NEGATIVE.startswith("nsfw, nude, naked, explicit, sexual content, child,")
    assert safety.QUALITY_CORE_NEGATIVE == gc.QUALITY_CORE_NEGATIVE
    assert safety.TEXT_NEGATIVE == gc.TEXT_NEGATIVE
    assert safety.SOLO_NEGATIVE == gc.SOLO_NEGATIVE


def test_the_cfg_floor_is_the_projects():
    assert safety.MIN_CFG == client.SAFETY_MIN_CFG == 1.5


@pytest.mark.parametrize(
    "solo,allow_text,extra", list(itertools.product((False, True), (False, True), (None, "blurry hands, extra mug")))
)
def test_the_negative_prompt_equals_the_projects_assembly(solo, allow_text, extra):
    _, expected = gc._build_prompt_and_negative(
        "a mug", extra, "safe", None, "", "", "z_image_turbo", solo=solo, allow_text=allow_text
    )
    built = safety.build_negative(extra, solo=solo, allow_text=allow_text)
    assert built == expected
    for term in AGE_TERMS:
        assert term in [t.strip() for t in built.split(",")]


def test_a_users_negative_is_appended_never_substituted():
    built = safety.build_negative("child")  # even a hostile "negative" cannot displace the safety half
    assert built.startswith(safety.SAFE_CONTENT_NEGATIVE)
    assert built.endswith(", child")


def test_the_prompt_screen_is_the_h3_skills_and_covers_every_age_term():
    assert safety.BLOCKED_TERMS == h3.BLOCKED_TERMS
    assert safety.BLOCKED_TERMS_CJK == h3.BLOCKED_TERMS_CJK
    assert safety._UNDER_18.pattern == h3._UNDER_18.pattern
    assert set(AGE_TERMS) <= set(safety.BLOCKED_TERMS)
    for term in AGE_TERMS + ("nude", "未成年", "少女"):
        with pytest.raises(WorkflowError) as exc:
            safety.screen_prompt("A portrait of a %s by a window." % term)
        assert exc.value.code == "PROMPT_REJECTED"


@pytest.mark.parametrize("text", ["a 16 year old", "she is 17yo", "fifteen years old", "十六歲的學生", "9 歲"])
def test_the_prompt_screen_rejects_ages_under_18(text):
    with pytest.raises(WorkflowError):
        safety.screen_prompt(text)


@pytest.mark.parametrize(
    "text", ["A 25 year old woman pours coffee.", "A cup of coffee, steam rises.", "二十五歲的老師"]
)
def test_the_prompt_screen_passes_ordinary_text(text):
    safety.screen_prompt(text)


def _graph(cfg=2.0, negative=None):
    negative = safety.build_negative() if negative is None else negative
    return {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a mug"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative}},
        "3": {"class_type": "KSampler", "inputs": {"cfg": cfg, "positive": ["6", 0], "negative": ["7", 0]}},
    }


def test_a_safe_graph_passes():
    safety.check_graph("anything", "negative_cfg", _graph())


@pytest.mark.parametrize("cfg", [1.0, 1.49, 0, True, "2.0", None])
def test_cfg_below_the_floor_is_refused_not_clamped(cfg):
    graph = _graph(cfg=cfg)
    with pytest.raises(WorkflowError) as exc:
        safety.check_graph("anything", "negative_cfg", graph)
    assert exc.value.code == "UNSAFE_JOB"
    assert graph["3"]["inputs"]["cfg"] is cfg or graph["3"]["inputs"]["cfg"] == cfg  # untouched


@pytest.mark.parametrize("term", AGE_TERMS)
def test_a_negative_prompt_missing_any_age_term_is_refused(term):
    terms = [t.strip() for t in safety.build_negative().split(",") if t.strip() != term]
    with pytest.raises(WorkflowError) as exc:
        safety.check_graph("anything", "negative_cfg", _graph(negative=", ".join(terms)))
    assert exc.value.code == "UNSAFE_JOB" and term in str(exc.value)


def test_a_sampler_without_a_negative_link_is_refused():
    graph = _graph()
    del graph["3"]["inputs"]["negative"]
    with pytest.raises(WorkflowError):
        safety.check_graph("anything", "negative_cfg", graph)


def test_a_guider_graph_cannot_pass_as_negative_cfg():
    graph = _graph()
    graph["8"] = {"class_type": "BasicGuider", "inputs": {}}
    with pytest.raises(WorkflowError) as exc:
        safety.check_graph("anything", "negative_cfg", graph)
    assert exc.value.code == "UNSAFE_JOB"


def test_only_the_named_workflow_may_run_without_a_negative_prompt():
    assert safety.SCREENED_PROMPT_WORKFLOWS == ("minimax-h3-basic",)
    assert set(safety.SCREENED_PROMPT_NODE) == set(safety.SCREENED_PROMPT_WORKFLOWS)
    h3_graph = {"7": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {}}}
    safety.check_graph("minimax-h3-basic", "screened_prompt", h3_graph, ["A cup of coffee."])
    with pytest.raises(WorkflowError) as exc:
        safety.check_graph("some-other-model", "screened_prompt", h3_graph, ["A cup of coffee."])
    assert exc.value.code == "INVALID_REGISTRY"
    with pytest.raises(WorkflowError):  # the name cannot be reused for a different guider graph
        safety.check_graph(
            "minimax-h3-basic", "screened_prompt", {"1": {"class_type": "BasicGuider", "inputs": {}}}, []
        )
    with pytest.raises(WorkflowError) as exc:
        safety.check_graph("minimax-h3-basic", "screened_prompt", h3_graph, ["A teen walks by."])
    assert exc.value.code == "PROMPT_REJECTED"


def test_the_static_profile_admits_no_model_nodes():
    ok = {"1": {"class_type": "EmptyImage", "inputs": {}}, "2": {"class_type": "SaveImage", "inputs": {}}}
    safety.check_graph("test", "static", ok)
    with pytest.raises(WorkflowError):
        safety.check_graph("test", "static", dict(ok, **{"3": {"class_type": "KSampler", "inputs": {}}}))


def test_an_unknown_profile_is_refused():
    with pytest.raises(WorkflowError):
        safety.check_graph("x", "none", _graph())
