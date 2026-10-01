"""The safety path. It has no switch, no tier parameter and no bypass argument.

Three profiles exist, and a workflow's registry entry names one of them:

  negative_cfg     The normal case. Every sampler has a negative prompt containing the age-safety terms and a
                   cfg of at least MIN_CFG (below that ComfyUI skips the negative conditioning entirely, and the
                   safety terms with it). The core writes the negative prompt; a user's own negative text is
                   only ever appended after it.
  screened_prompt  For a model whose graph has no negative prompt and no cfg. Allowed ONLY for the workflow
                   names in SCREENED_PROMPT_WORKFLOWS - adding such a model is a deliberate change to this file,
                   not a registry entry. Every text the job carries is screened against BLOCKED_TERMS, and the
                   first frame must come from a source this platform can account for (see inputs.py).
  static           Graphs made only of STATIC_CLASSES (no model at all): the API self-test.

The checks raise; they never clamp or repair. They run when a job is created and again on the worker right
before the graph is submitted, because a job file sits in a shared folder where it can be edited by hand.

The constants are pinned copies: this package runs on the compute runtime without the repository, so it cannot
import generate_character or comfyui_client. tests/test_aiwf_safety.py asserts every one of them equals the
project's constant, and that the term tables are a superset of the age-safety list.
"""

import re

from . import WorkflowError

MIN_CFG = 1.5
AGE_TERMS = ("child", "children", "kid", "minor", "teen", "teenager", "underage", "young girl")
AGE_SAFETY_NEGATIVE = ", ".join(AGE_TERMS)
SAFE_CONTENT_NEGATIVE = "nsfw, nude, naked, explicit, sexual content, " + AGE_SAFETY_NEGATIVE
QUALITY_CORE_NEGATIVE = "lowres, blurry, deformed, extra limbs, bad anatomy, watermark"
TEXT_NEGATIVE = "text"
SOLO_NEGATIVE = "multiple people, two people, duplicate, twins, extra person, crowd"

PROFILE_NEGATIVE_CFG = "negative_cfg"
PROFILE_SCREENED_PROMPT = "screened_prompt"
PROFILE_STATIC = "static"
PROFILES = (PROFILE_NEGATIVE_CFG, PROFILE_SCREENED_PROMPT, PROFILE_STATIC)

# The one model allowed to run without a negative prompt. Do not extend this from a registry file.
SCREENED_PROMPT_WORKFLOWS = ("minimax-h3-basic",)
# ...and the node its graph must contain, so the name cannot be reused for some other guider graph.
SCREENED_PROMPT_NODE = {"minimax-h3-basic": "MiniMaxH3ImageToVideo"}
STATIC_CLASSES = frozenset({"EmptyImage", "SaveImage"})
# Samplers that take a guider instead of positive/negative/cfg. Under negative_cfg they would run with no
# negative prompt at all, so their presence is refused outright rather than discovered by a missing cfg.
GUIDER_CLASSES = frozenset({"BasicGuider", "SamplerCustomAdvanced", "SamplerCustom", "CFGGuider", "DualCFGGuider"})

# Positive-side screen for the screened_prompt profile: refused outright. The age list is a superset of
# AGE_TERMS. "minor" also blocks "minor adjustments" - write "small" instead.
BLOCKED_TERMS = (
    "child", "children", "kid", "kids", "minor", "minors", "teen", "teens", "teenage", "teenager",
    "teenagers", "underage", "preteen", "young girl", "young girls", "young boy", "young boys",
    "little girl", "little boy", "schoolgirl", "schoolboy", "infant", "toddler", "juvenile", "loli",
    "lolita", "shota", "nsfw", "nude", "nudity", "naked", "topless", "bottomless", "explicit", "sex",
    "sexual", "sexually", "porn", "porno", "pornographic", "erotic", "genitals", "undress", "undressing",
)  # fmt: skip
BLOCKED_TERMS_CJK = (
    "兒童", "儿童", "小孩", "孩童", "幼童", "幼兒", "幼儿", "嬰兒", "婴儿", "未成年", "少女", "國中生", "国中生",
    "初中生", "高中生", "小學生", "小学生", "蘿莉", "萝莉", "正太", "裸體", "裸体", "全裸", "色情", "性愛", "性爱",
    "做愛", "做爱",
)  # fmt: skip
_UNDER_18 = re.compile(
    r"(?<![0-9.])(?:1[0-7]|[1-9])\s*-?\s*(?:years?|yrs?)\s*-?\s*old(?![a-z])"
    r"|(?<![0-9.])(?:1[0-7]|[1-9])\s*(?:yo|y/o)(?![a-z])"
    r"|(?<![a-z])(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
    r"|fifteen|sixteen|seventeen)[\s-]*(?:years?|yrs?)[\s-]*old(?![a-z])"
    r"|(?<![0-9十百一二三四五六七八九])(?:1[0-7]|[1-9]|十[一二三四五六七]?|[一二三四五六七八九])\s*[歲岁]",
    re.IGNORECASE,
)


def _terms_regex(terms):
    parts = sorted((re.escape(t).replace(r"\ ", r"[\s-]+") for t in terms), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])", re.IGNORECASE)


_BLOCKED_RE = _terms_regex(BLOCKED_TERMS)


def screen_prompt(text):
    """Raise PROMPT_REJECTED when the text contains a blocked term. There is no override."""
    hits = {m.group(0).lower() for m in _BLOCKED_RE.finditer(text)}
    hits |= {t for t in BLOCKED_TERMS_CJK if t in text}
    hits |= {m.group(0) for m in _UNDER_18.finditer(text)}
    if hits:
        raise WorkflowError(
            "PROMPT_REJECTED",
            "The prompt contains terms that are never sent to a model without a negative prompt: %s" % sorted(hits),
        )


def build_negative(extra=None, *, solo=False, allow_text=False):
    """The negative prompt for the negative_cfg profile: content safety first, then the quality floor, then
    whatever the user added. `solo` and `allow_text` only trim the quality floor; the content half (with the
    age-safety terms) is not affected by anything."""
    quality = [QUALITY_CORE_NEGATIVE]
    if not allow_text:
        quality.append(TEXT_NEGATIVE)
    if solo:
        quality.append(SOLO_NEGATIVE)
    negative = "%s, %s" % (SAFE_CONTENT_NEGATIVE, ", ".join(quality))
    extra = (extra or "").strip()
    return "%s, %s" % (negative, extra) if extra else negative


def _nodes(graph):
    if not isinstance(graph, dict) or not graph:
        raise WorkflowError("INVALID_WORKFLOW", "the workflow graph is not a non-empty object")
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
            raise WorkflowError("INVALID_WORKFLOW", "node %s is not an API-format node" % node_id)
        yield str(node_id), node


def check_negative_cfg(graph):
    samplers = []
    for node_id, node in _nodes(graph):
        if node["class_type"] in GUIDER_CLASSES:
            raise WorkflowError(
                "UNSAFE_JOB", "node %s (%s) samples without a negative prompt" % (node_id, node["class_type"])
            )
        cfg = node["inputs"].get("cfg")
        if cfg is not None:
            if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or cfg < MIN_CFG:
                raise WorkflowError("UNSAFE_JOB", "node %s has cfg %r below the %s floor" % (node_id, cfg, MIN_CFG))
            samplers.append(node)
    if not samplers:
        raise WorkflowError("UNSAFE_JOB", "the graph has no sampler with a cfg input")
    for node in samplers:
        ref = node["inputs"].get("negative")
        if not (isinstance(ref, list) and ref and str(ref[0]) in graph):
            raise WorkflowError("UNSAFE_JOB", "a sampler has no negative conditioning node")
        text = graph[str(ref[0])].get("inputs", {}).get("text")
        if not isinstance(text, str):
            raise WorkflowError("UNSAFE_JOB", "the negative conditioning is not a literal text encoder")
        terms = {t.strip().lower() for t in text.split(",")}
        missing = [t for t in AGE_TERMS if t not in terms]
        if missing:
            raise WorkflowError("UNSAFE_JOB", "the negative prompt is missing the age-safety terms %s" % missing)


def check_static(graph):
    extra = sorted({node["class_type"] for _, node in _nodes(graph)} - STATIC_CLASSES)
    if extra:
        raise WorkflowError(
            "UNSAFE_JOB", "a static workflow may only contain %s, found %s" % (sorted(STATIC_CLASSES), extra)
        )


def check_screened(workflow_name, graph, texts):
    if workflow_name not in SCREENED_PROMPT_WORKFLOWS:
        raise WorkflowError(
            "UNSAFE_JOB",
            "workflow %r asks for the screened_prompt profile, which is reserved for %s"
            % (workflow_name, list(SCREENED_PROMPT_WORKFLOWS)),
        )
    classes = {node["class_type"] for _, node in _nodes(graph)}
    if SCREENED_PROMPT_NODE[workflow_name] not in classes:
        raise WorkflowError(
            "UNSAFE_JOB", "workflow %r must contain a %s node" % (workflow_name, SCREENED_PROMPT_NODE[workflow_name])
        )
    for text in texts:
        screen_prompt(text)


def check_profile_name(workflow_name, profile):
    """Registry-time check: the profile exists, and screened_prompt is only claimed by an allowed workflow."""
    if profile not in PROFILES:
        raise WorkflowError("INVALID_REGISTRY", "workflow %r has unknown safety profile %r" % (workflow_name, profile))
    if profile == PROFILE_SCREENED_PROMPT and workflow_name not in SCREENED_PROMPT_WORKFLOWS:
        raise WorkflowError(
            "INVALID_REGISTRY",
            "workflow %r may not use the screened_prompt profile (reserved for %s)"
            % (workflow_name, list(SCREENED_PROMPT_WORKFLOWS)),
        )


def check_graph(workflow_name, profile, graph, texts=()):
    """The trust-boundary check on a bound graph. `texts` are every string the job contributes."""
    check_profile_name(workflow_name, profile)
    if profile == PROFILE_NEGATIVE_CFG:
        check_negative_cfg(graph)
    elif profile == PROFILE_STATIC:
        check_static(graph)
    else:
        check_screened(workflow_name, graph, texts)
