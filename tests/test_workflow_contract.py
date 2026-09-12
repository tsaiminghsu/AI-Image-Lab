"""Guards the tribal knowledge workflow_contracts.py makes explicit (see its docstring):
which node id in each training/workflow_template*.json means which ComfyUI class_type, and
that comfyui_client.py's ~15 submit_* functions only ever poke ids their own template
actually has under the class they expect.

Six groups:
  1. every workflow_template*.json on disk has exactly one CONTRACTS entry, and vice versa.
  2. every template file on disk matches its own contract (check() == []).
  3. an AST cross-check that each submit_* function only references node ids present in the
     contract of the template it loads - so a future `wf["44"]` with no contract entry (e.g.
     because the template was re-exported and renumbered, or because of a typo) fails here
     instead of as a KeyError minutes into a real generation.
  4. every `_rewire(wf, old_ref, ...)` call's old_ref actually appears as some node's input
     in the corresponding template - otherwise the rewire is a silent no-op (the exact
     "template re-exported, output order changed" failure mode this module exists to catch).
  5. the wiring half of the age-safety invariant: every sampler/detailer's "negative" input
     resolves back to NEGATIVE_NODE, so a correct AGE_SAFETY_NEGATIVE string in node 7 is
     not wasted on a graph that doesn't actually feed it to the sampler.
  6. benchmark.HQ_STAGE_NAMES (a third, independent copy of the HQ node-id convention) agrees
     with the HQ contract.
"""

import ast
import glob
import json
import os
import re

import pytest

import benchmark
import comfyui_client as client
import workflow_contracts

TRAINING_DIR = os.path.dirname(os.path.abspath(client.__file__))

with open(client.__file__, encoding="utf-8") as _f:
    _CLIENT_SOURCE = _f.read()
_CLIENT_AST = ast.parse(_CLIENT_SOURCE, filename=client.__file__)
_MODULE_FUNCTIONS = [n for n in _CLIENT_AST.body if isinstance(n, ast.FunctionDef)]
_FUNCTIONS_BY_NAME = {f.name: f for f in _MODULE_FUNCTIONS}


def _load_json(name):
    with open(os.path.join(TRAINING_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------
# Group 1/2: the contract set and per-template correctness.
# ---------------------------------------------------------------------------------------


def test_every_template_on_disk_has_exactly_one_contract():
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(TRAINING_DIR, "workflow_template*.json"))}
    assert on_disk == set(workflow_contracts.CONTRACTS.keys())


@pytest.mark.parametrize("template_name", sorted(workflow_contracts.CONTRACTS.keys()))
def test_template_matches_its_contract(template_name):
    wf = _load_json(template_name)
    assert workflow_contracts.check(template_name, wf) == []


# ---------------------------------------------------------------------------------------
# Group 3: AST cross-check - comfyui_client.py only pokes ids its template actually has.
# ---------------------------------------------------------------------------------------
#
# Approach: for each top-level function that calls _load_template(<CONST>), resolve <CONST>
# to a template filename via the module-level WORKFLOW_TEMPLATE_*_PATH assignments. Then
# collect every short all-digit string constant ANYWHERE in that function's body (not just
# `wf["N"]` subscripts) minus the ids the function creates itself. This is a deliberate
# over-approximation - narrower "just wf[...] subscripts" collection misses ids referenced
# through a loop tuple, e.g.
#   for node_id, node_steps in (("3", steps), ("34", hires_steps), ("52", detailer_steps)):
#       wf[node_id]["inputs"]["steps"] = node_steps
# in submit_generation_animatediff, where "3"/"34"/"52" never appear as a literal wf["..."]
# subscript at all.
#
# This was verified NOT to produce false positives for the current file: the only all-digit
# string literals in comfyui_client.py that are not real node-id references are
# `"1"` (an env-var value comparison in `_control_lora_forces_full`) and the `"9"` default
# for `_submit_and_wait`'s own `output_node_id` parameter - both live in helper functions
# that never call `_load_template`, so restricting the walk to only the callers of
# `_load_template` (rather than the whole module) already excludes them. No further
# narrowing was needed.

_DIGIT_ID = re.compile(r"[0-9]{1,3}")


def _template_path_constants():
    """name -> filename for every `WORKFLOW_TEMPLATE_*_PATH = os.path.join(..., "x.json")`
    module-level assignment."""
    consts = {}
    for node in _CLIENT_AST.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if not (name.startswith("WORKFLOW_TEMPLATE") and name.endswith("_PATH")):
            continue
        call = node.value
        if isinstance(call, ast.Call) and call.args and isinstance(call.args[-1], ast.Constant):
            last_arg = call.args[-1].value
            if isinstance(last_arg, str):
                consts[name] = last_arg
    return consts


def _load_template_call(func_node):
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_load_template":
            return node
    return None


def _function_templates():
    """func_name -> template filename, for every top-level function that calls
    _load_template() (bare, meaning the default WORKFLOW_TEMPLATE_PATH) or
    _load_template(SOME_CONST)."""
    consts = _template_path_constants()
    mapping = {}
    for func in _MODULE_FUNCTIONS:
        call = _load_template_call(func)
        if call is None:
            continue
        if not call.args:
            mapping[func.name] = consts.get("WORKFLOW_TEMPLATE_PATH")
        elif isinstance(call.args[0], ast.Name):
            mapping[func.name] = consts.get(call.args[0].id)
    return mapping


FUNCTION_TEMPLATES = _function_templates()


def test_function_templates_were_actually_resolved():
    # Sanity check on the AST-walking machinery above, so a typo/refactor in comfyui_client.py
    # that breaks the WORKFLOW_TEMPLATE_*_PATH <-> _load_template(...) resolution fails here
    # with a clear message instead of group-3/4 silently checking zero functions.
    assert len(FUNCTION_TEMPLATES) >= 11, FUNCTION_TEMPLATES
    assert all(template is not None for template in FUNCTION_TEMPLATES.values()), FUNCTION_TEMPLATES
    assert set(FUNCTION_TEMPLATES.values()) == set(workflow_contracts.CONTRACTS.keys())


def _created_node_ids(func_node):
    """Ids the function builds itself via `wf["N"] = {...}` (a brand new node dict) - these
    are legitimately not in the template's contract, since they don't exist until this
    function adds them (e.g. the animatediff LCM/motion-lora/RIFE nodes)."""
    created = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "wf"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
            and isinstance(node.value, ast.Dict)
        ):
            created.add(target.slice.value)
    return created


def _referenced_node_ids(func_node):
    """Every short all-digit string constant in the function body, minus ids it creates
    itself. See the module-level comment above for why this over-approximates rather than
    only matching literal `wf["N"]` subscripts."""
    referenced = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _DIGIT_ID.fullmatch(node.value):
            referenced.add(node.value)
    return referenced - _created_node_ids(func_node)


@pytest.mark.parametrize("func_name", sorted(FUNCTION_TEMPLATES.keys()))
def test_function_only_references_contracted_node_ids(func_name):
    template = FUNCTION_TEMPLATES[func_name]
    contract_ids = set(workflow_contracts.CONTRACTS[template].keys())
    referenced = _referenced_node_ids(_FUNCTIONS_BY_NAME[func_name])
    extra = referenced - contract_ids
    assert not extra, (
        f"{func_name} references node id(s) {sorted(extra)} that have no entry in "
        f"{template}'s contract - either the template was re-exported and renumbered, or "
        f"workflow_contracts.CONTRACTS[{template!r}] needs a new entry"
    )


# ---------------------------------------------------------------------------------------
# Group 4: every _rewire(...) old_ref literal actually exists as a node input in the
# corresponding template - otherwise the rewire is a silent no-op.
# ---------------------------------------------------------------------------------------


def _rewire_old_refs(func_node):
    """[node_id, output_index] pairs passed as the old_ref (2nd positional arg) of every
    _rewire(wf, old_ref, new_ref) call in this function."""
    refs = []
    for node in ast.walk(func_node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_rewire"):
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.List) or len(node.args[1].elts) != 2:
            continue
        id_node, idx_node = node.args[1].elts
        if (
            isinstance(id_node, ast.Constant)
            and isinstance(id_node.value, str)
            and isinstance(idx_node, ast.Constant)
            and isinstance(idx_node.value, int)
        ):
            refs.append([id_node.value, idx_node.value])
    return refs


def _ref_exists_as_some_input(wf, ref):
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        for val in node.get("inputs", {}).values():
            if isinstance(val, list) and list(val) == ref:
                return True
    return False


def _rewire_cases():
    cases = []
    for func_name, template in FUNCTION_TEMPLATES.items():
        for ref in _rewire_old_refs(_FUNCTIONS_BY_NAME[func_name]):
            cases.append((func_name, template, ref))
    return cases


@pytest.mark.parametrize("func_name,template,ref", _rewire_cases(), ids=lambda v: str(v) if isinstance(v, list) else v)
def test_rewire_old_refs_exist_in_their_template(func_name, template, ref):
    wf = _load_json(template)
    assert _ref_exists_as_some_input(wf, ref), (
        f"{func_name}'s _rewire(wf, {ref}, ...) has no matching input anywhere in {template} - "
        "this rewire is a silent no-op (the template was likely re-exported with different "
        "node ids or output wiring)"
    )


# ---------------------------------------------------------------------------------------
# Group 5: the negative CLIPTextEncode (node 7) actually feeds every sampler/detailer -
# the wiring half of the age-safety invariant (AGE_SAFETY_NEGATIVE lives in node 7's text,
# but that's worthless if a sampler doesn't read node 7).
# ---------------------------------------------------------------------------------------

_SAMPLER_LIKE_CLASSES = {"KSampler", "KSamplerAdvanced", "FaceDetailer", "ToBasicPipe"}


def _is_sampler_like(class_type):
    return class_type in _SAMPLER_LIKE_CLASSES or class_type.startswith("DetailerForEach")


def _resolve_negative_source(wf, ref, _seen=None):
    """Walk a `negative` input upstream to the CLIPTextEncode node id that ultimately feeds
    it, following ControlNetApplyAdvanced's output index 1 (its own negative branch)."""
    _seen = _seen or set()
    node_id, out_idx = ref[0], ref[1]
    assert node_id not in _seen, f"cycle walking negative wiring back through node {node_id}"
    node = wf[node_id]
    class_type = node["class_type"]
    if class_type == "CLIPTextEncode":
        return node_id
    if class_type == "ControlNetApplyAdvanced" and out_idx == 1:
        return _resolve_negative_source(wf, node["inputs"]["negative"], _seen | {node_id})
    raise AssertionError(f"don't know how to walk a negative input through {class_type} (node {node_id})")


# img2vid's conditioning comes from SVD_img2vid_Conditioning (an image-conditioning node,
# not a text encoder pair) - it has no node 7 / no CLIPTextEncode at all, so the id scheme
# genuinely doesn't apply here. Skipped deliberately, not silently: see workflow_contracts's
# _IMG2VID comment for the same point made at the contract-authoring end.
_SKIP_NEGATIVE_WIRING_CHECK = {"workflow_template_img2vid.json"}


def _negative_wiring_cases():
    cases = []
    for template_name in sorted(workflow_contracts.CONTRACTS.keys()):
        if template_name in _SKIP_NEGATIVE_WIRING_CHECK:
            continue
        wf = _load_json(template_name)
        for node_id, node in wf.items():
            class_type = node.get("class_type")
            if not _is_sampler_like(class_type):
                continue
            if "negative" not in node.get("inputs", {}):
                continue
            cases.append((template_name, node_id, class_type))
    return cases


_NEGATIVE_WIRING_CASES = _negative_wiring_cases()


def test_negative_wiring_cases_cover_every_template_with_a_sampler():
    # Sanity check: every non-skipped template must contribute at least one case, otherwise
    # this test group would be silently vacuous for that template (e.g. a class-name typo in
    # _is_sampler_like).
    covered = {template for template, _, _ in _NEGATIVE_WIRING_CASES}
    expected = set(workflow_contracts.CONTRACTS.keys()) - _SKIP_NEGATIVE_WIRING_CHECK
    assert covered == expected


@pytest.mark.parametrize("template_name,node_id,class_type", _NEGATIVE_WIRING_CASES)
def test_negative_input_resolves_to_negative_node(template_name, node_id, class_type):
    wf = _load_json(template_name)
    source = _resolve_negative_source(wf, wf[node_id]["inputs"]["negative"])
    assert source == workflow_contracts.NEGATIVE_NODE, (
        f"{template_name} node {node_id} ({class_type}): negative input resolves to node "
        f"{source}, not NEGATIVE_NODE ({workflow_contracts.NEGATIVE_NODE!r}) - the safety "
        "negative text would not reach this sampler"
    )


# ---------------------------------------------------------------------------------------
# Group 6: benchmark.HQ_STAGE_NAMES (a third copy of the HQ node-id convention, used only
# for per-stage timing labels) agrees with the HQ contract.
# ---------------------------------------------------------------------------------------

# Independently-judged expected class per label (not derived from HQ_STAGE_NAMES or the
# contract - this is what each label OUGHT to mean), cross-checked against the real contract
# below so a label/id drift (e.g. HQ_STAGE_NAMES kept "34": "pass2_hires" after node 34 was
# repurposed) is caught rather than asserting the contract agrees with itself.
_HQ_STAGE_EXPECTED_CLASS = {
    "3": "KSampler",  # pass1_base
    "21": "OpenposePreprocessor",  # openpose
    "22": "ControlNetLoader",  # controlnet_load
    "31": "ImageUpscaleWithModel",  # esrgan_upscale
    "34": "KSampler",  # pass2_hires
    "41": "FaceDetailer",  # facedetailer_face
    "43": "FaceDetailer",  # facedetailer_hand
    "9": "SaveImage",  # save
}


def test_hq_stage_names_matches_hq_stage_expectations():
    assert set(benchmark.HQ_STAGE_NAMES.keys()) == set(_HQ_STAGE_EXPECTED_CLASS.keys())


@pytest.mark.parametrize("node_id,label", sorted(benchmark.HQ_STAGE_NAMES.items()))
def test_hq_stage_name_node_is_in_hq_contract_with_plausible_class(node_id, label):
    hq_contract = workflow_contracts.CONTRACTS["workflow_template_hq.json"]
    assert node_id in hq_contract, f"HQ_STAGE_NAMES[{node_id!r}] ({label!r}) has no entry in the HQ contract"
    assert hq_contract[node_id] == _HQ_STAGE_EXPECTED_CLASS[node_id], (
        f"HQ_STAGE_NAMES[{node_id!r}] = {label!r} but the HQ contract says node {node_id} is "
        f"{hq_contract[node_id]!r}, not the expected {_HQ_STAGE_EXPECTED_CLASS[node_id]!r}"
    )
