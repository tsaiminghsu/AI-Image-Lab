"""Job parameters: validate what a caller sent against the workflow's registry entry, then write the values
into a copy of the workflow graph.

validate() runs when a job is created and again on the worker (the job file may have been edited in between).
bind() runs only on the worker, because some values exist only there: the first frame's file name inside
ComfyUI, and the video size derived from that image.

The transforms are a small fixed set, selected by the registry: seconds_to_frames (a model's frame grid) and
auto_resolution (keep an input image's aspect ratio inside a pixel budget).
"""

import random
import re

from . import WorkflowError, safety

RANDOM_SEED_MAX = 2**31 - 1
PLACEHOLDER = "REPLACE_"


def _bad(message):
    raise WorkflowError("INVALID_INPUT", message)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def seconds_to_frames(seconds, fps, step, offset):
    """Requested seconds -> the nearest frame count not below it on a `step*k + offset` grid."""
    requested = max(offset, round(seconds * fps))
    return requested + (offset - requested % step) % step


def round_to_multiple(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


def auto_resolution(img_w, img_h, *, short_side, max_megapixels, multiple):
    """Keep the image's aspect ratio (so it is not stretched) with the given short side, shrinking the short
    side one grid step at a time until the frame fits the pixel budget."""
    side = short_side
    while True:
        if img_w >= img_h:
            w, h = round_to_multiple(side * img_w / img_h, multiple), side
        else:
            w, h = side, round_to_multiple(side * img_h / img_w, multiple)
        if w * h <= max_megapixels * 1_000_000 or side <= 256:
            return w, h
        side -= multiple


def _number(name, spec, value, integer):
    if integer:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not _is_int(value):
            _bad("%s must be an integer" % name)
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _bad("%s must be a number" % name)
        value = float(value)
    lo, hi = spec.get("min"), spec.get("max")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        _bad("%s %s is outside %s-%s" % (name, value, lo, hi))
    multiple = spec.get("multiple_of")
    if multiple and value % multiple:
        _bad("%s %s is not a multiple of %s" % (name, value, multiple))
    return value


def _string(name, spec, value):
    if not isinstance(value, str):
        _bad("%s must be text" % name)
    value = value.strip()
    limit = spec.get("max_chars")
    if limit and len(value) > limit:
        _bad("%s is %d characters; the limit is %d" % (name, len(value), limit))
    for pattern in spec.get("forbid_patterns", []):
        if re.search(pattern, value):
            _bad("%s contains markup the workflow adds by itself (%s)" % (name, pattern))
    return value


def _cfg_floor(workflow, name, spec, value):
    """Under negative_cfg, anything bound to a `cfg` input obeys the safety floor whatever the registry says."""
    if workflow.profile != safety.PROFILE_NEGATIVE_CFG:
        return
    if any(b["input"] == "cfg" for b in spec.get("bind", [])) and value < safety.MIN_CFG:
        raise WorkflowError(
            "INVALID_INPUT",
            "%s must be at least %s: below that ComfyUI skips the negative prompt, and with it the safety terms"
            % (name, safety.MIN_CFG),
        )


def validate(workflow, values, *, rng=None):
    """Normalised values for every declared parameter (None = not given and no default). Raises
    WorkflowError(INVALID_INPUT) with the parameter's name; never repairs a bad value."""
    specs = workflow.parameters
    unknown = sorted(set(values) - set(specs))
    if unknown:
        _bad("unknown parameter(s) %s for workflow %s; it takes: %s" % (unknown, workflow.name, ", ".join(specs)))
    out = {}
    for name, spec in specs.items():
        value = values.get(name)
        kind = spec["type"]
        if kind == "seed":
            if value is None or value == -1:
                value = (rng or random).randint(0, min(spec.get("max", RANDOM_SEED_MAX), RANDOM_SEED_MAX))
            elif not _is_int(value) or not 0 <= value <= spec.get("max", RANDOM_SEED_MAX):
                _bad("%s must be -1 or an integer 0-%s" % (name, spec.get("max", RANDOM_SEED_MAX)))
            out[name] = value
            continue
        if kind == "aspect":
            out[name] = value
            continue
        if value is None or (isinstance(value, str) and not value.strip() and kind in ("string", "negative")):
            value = spec.get("default")
            if value is None:
                if spec.get("required"):
                    _bad("%s is required" % name)
                out[name] = None
                continue
        if kind in ("string", "negative"):
            value = _string(name, spec, value)
        elif kind == "int":
            value = _number(name, spec, value, True)
        elif kind == "float":
            # The floor first: its message says why, where the range check would only say "outside".
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _cfg_floor(workflow, name, spec, value)
            value = _number(name, spec, value, False)
        elif kind == "bool":
            if not isinstance(value, bool):
                _bad("%s must be true or false" % name)
        elif kind == "choice":
            if value not in spec["choices"]:
                _bad("%s must be one of %s" % (name, spec["choices"]))
        out[name] = value
    _resolve_aspect(workflow, out)
    return out


def _resolve_aspect(workflow, out):
    """`aspect` is a shortcut for width and height: give it, or both sizes, never a mix."""
    for name, spec in workflow.parameters.items():
        if spec["type"] != "aspect":
            continue
        width, height, aspect = out.get("width"), out.get("height"), out.get(name)
        if aspect is None and width is None and height is None:
            aspect = spec["default"]
        if aspect is not None:
            if aspect not in spec["sizes"]:
                _bad("%s must be one of %s" % (name, list(spec["sizes"])))
            size = tuple(spec["sizes"][aspect])
            # A stored job carries both (the aspect and the size it resolved to); that is the same request.
            if (width, height) not in ((None, None), size):
                _bad("give either %s or width and height, not both" % name)
            out["width"], out["height"] = size
            out[name] = aspect
        elif width is None or height is None:
            _bad("give both width and height")
        limit = workflow.get("limits", {}).get("max_pixels")
        if limit and out["width"] * out["height"] > limit:
            _bad("%sx%s is over the limit of %s pixels" % (out["width"], out["height"], limit))


def composed(workflow, values):
    """The text of every compose entry, with {name} replaced by the string parameters' values."""
    texts = []
    for entry in workflow.get("compose", []):
        text = entry["template"]
        for name, spec in workflow.parameters.items():
            if spec["type"] == "string":
                text = text.replace("{%s}" % name, values.get(name) or "")
        limit = entry.get("max_chars")
        if limit and len(text) > limit:
            _bad("the assembled prompt is %d characters; the limit is %d" % (len(text), limit))
        texts.append(text)
    return texts


def texts(workflow, values):
    """Every string a job contributes to the graph: what the screened_prompt profile screens."""
    found = [v for n, v in values.items() if workflow.parameters[n]["type"] in ("string", "negative") and v]
    return found + composed(workflow, values)


def split(workflow, values):
    """(input, parameters) the way the job file stores them."""
    job_input, job_params = {}, {}
    for name, value in values.items():
        if value is None:
            continue
        target = job_input if workflow.parameters[name].get("in", "parameters") == "input" else job_params
        target[name] = value
    return job_input, job_params


def _set(graph, binds, value):
    for b in binds:
        graph[str(b["node"])]["inputs"][b["input"]] = value


def bind(workflow, values, job_id, input_files=None):
    """The graph to submit. input_files: name -> {"comfy_name", "width", "height"} for each declared input."""
    graph = workflow.template()
    input_files = input_files or {}
    for c in workflow.get("constants", []):
        _set(graph, [c], c["value"])
    for name, spec in workflow.parameters.items():
        if "bind" not in spec:
            continue
        value = values.get(name)
        if spec["type"] == "negative":
            value = safety.build_negative(
                value, solo=bool(values.get("solo")), allow_text=bool(values.get("allow_text"))
            )
        if value is None:
            continue
        transform = spec.get("transform")
        if transform:
            if transform["kind"] != "seconds_to_frames":
                raise WorkflowError("INVALID_REGISTRY", "unknown transform %r" % transform["kind"])
            value = seconds_to_frames(value, transform["fps"], transform["step"], transform["offset"])
        _set(graph, spec["bind"], value)
    for entry, text in zip(workflow.get("compose", []), composed(workflow, values)):
        _set(graph, entry["bind"], text)
    for name, spec in workflow.inputs.items():
        if name not in input_files:
            raise WorkflowError("INPUT_NOT_READY", "input %r has not been resolved to a file" % name)
        _set(graph, spec["bind"], input_files[name]["comfy_name"])
    for d in workflow.get("derived", []):
        info = input_files[d["from_input"]]
        w, h = auto_resolution(
            info["width"], info["height"],
            short_side=d["short_side"], max_megapixels=d["max_megapixels"], multiple=d["multiple"],
        )  # fmt: skip
        _set(graph, d["bind_width"], w)
        _set(graph, d["bind_height"], h)
    out = workflow["output"]
    if out.get("prefix_input"):
        prefix = out.get("prefix_template", "{job_id}").replace("{job_id}", job_id)
        graph[str(out["node"])]["inputs"][out["prefix_input"]] = prefix
    leftover = [
        "%s.%s" % (node_id, key)
        for node_id, node in graph.items()
        for key, value in node["inputs"].items()
        if isinstance(value, str) and value.startswith(PLACEHOLDER)
    ]
    if leftover:
        raise WorkflowError(
            "INVALID_REGISTRY", "workflow %s leaves placeholders unbound: %s" % (workflow.name, leftover)
        )
    return graph


def expected_output(workflow, values, graph):
    """What a valid output of this job looks like: the registry's `expect`, plus the size and frame count this
    job asked for (the width/height parameters, or the derived size read back from the bound graph)."""
    expect = dict(workflow["output"].get("expect", {}))
    if values.get("width") and values.get("height"):
        expect["width"], expect["height"] = values["width"], values["height"]
    for d in workflow.get("derived", []):
        b = d["bind_width"][0]
        expect["width"] = graph[str(b["node"])]["inputs"][b["input"]]
        b = d["bind_height"][0]
        expect["height"] = graph[str(b["node"])]["inputs"][b["input"]]
    for spec in workflow.parameters.values():
        if spec.get("transform", {}).get("kind") == "seconds_to_frames":
            b = spec["bind"][0]
            expect["frames"] = graph[str(b["node"])]["inputs"][b["input"]]
            expect.setdefault("fps", spec["transform"]["fps"])
    return expect
