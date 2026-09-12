"""Guards training/gui.py's Gradio event wiring statically, without importing it.

gui.py can't be imported here: it builds a `gr.Blocks` at module scope (gui.py:336) and
registers an `atexit` hook (gui.py:44), and gradio itself isn't installed in this dev venv
or CI. Gradio also binds each event handler's arguments *positionally* from its
`inputs=[...]` list - a handler with the wrong parameter count passes import cleanly and
only blows up when a user actually clicks the button. CLAUDE.md's "GUI 改動要做 build
檢查" (arity must equal the button's inputs count) was previously a manual step; this
file makes it automatic.

A second check below (the keyword-argument contract) catches the sibling failure mode:
gui.py calling `gc.<fn>(...)`/`client.<fn>(...)` with a keyword that fn no longer accepts
(e.g. a rename on the generate_character/comfyui_client side that gui.py wasn't updated
for) - also otherwise invisible until someone clicks the button.
"""

import ast
import inspect
import os

import pytest

GUI_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "gui.py")

EVENT_METHODS = {"click", "change", "submit", "upload", "select", "input", "release"}

# The five generation entry points every tab's primary button is wired to. If gui.py stops
# calling one of these from an event binding, the arity test below would silently stop
# covering it - test_generation_handlers_are_covered catches that.
GENERATION_HANDLERS = {
    "generate",
    "generate_video_animatediff",
    "generate_talking_head_ui",
    "generate_video_svd",
    "generate_gif",
}


def _load_tree():
    with open(GUI_PATH, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=GUI_PATH)


def _find_functions(tree):
    """Map name -> def node for every function in the file (module-level and nested), so a
    handler referenced by name resolves even if gui.py later nests one inside another."""
    return {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _component_names(tree):
    """Names bound to a bare `x = gr.Something(...)` call anywhere in the Blocks tree
    (module scope, or inside a `with gr.Row():` / `with gr.Tab():` / ... block) - each one
    is a single Gradio component, so `inputs=x` sends exactly one value. Deliberately does
    NOT recurse into function bodies (a handler's local variables are never components) and
    deliberately does NOT match list-comprehension variables like
    `variant_radios = [gr.Radio(...) for r in ...]` (gui.py:429-433), which build a *list*
    of components and must instead be paired with a `*args` handler.
    """
    names = set()

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Attribute)
                and isinstance(child.value.func.value, ast.Name)
                and child.value.func.value.id == "gr"
            ):
                names.add(child.targets[0].id)
            visit(child)

    visit(tree)
    return names


def _inputs_count(inputs_node, component_names):
    """Expected positional-arg count implied by an `inputs=` value, or None when it's a
    bare Name that isn't a single component (i.e. a list variable) - callers must then
    require the handler to take `*args`, since the true count isn't visible statically."""
    if inputs_node is None:
        return 0
    if isinstance(inputs_node, ast.Constant) and inputs_node.value is None:
        return 0
    if isinstance(inputs_node, (ast.List, ast.Tuple)):
        return len(inputs_node.elts)
    if isinstance(inputs_node, ast.Name):
        return 1 if inputs_node.id in component_names else None
    return None  # e.g. a BinOp like `outputs=x + [y]` - not used for inputs= today


def _handler_info(call, functions):
    """Returns (label, arity, is_variadic) for the call's handler (first positional arg, or
    the `fn=` keyword), or None if it isn't a Name/Lambda this file knows how to resolve."""
    handler_node = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == "fn"), None)
    if handler_node is None:
        return None
    if isinstance(handler_node, ast.Lambda):
        node_args = handler_node.args
        label = "<lambda>"
    elif isinstance(handler_node, ast.Name) and handler_node.id in functions:
        node_args = functions[handler_node.id].args
        label = handler_node.id
    else:
        return None
    arity = len(node_args.posonlyargs) + len(node_args.args)
    return label, arity, node_args.vararg is not None


def _collect_bindings():
    """Every `<component>.<event>(handler, inputs=..., ...)` call in gui.py, resolved to
    (test id, handler label, arity, inputs_count, is_variadic, source line)."""
    tree = _load_tree()
    functions = _find_functions(tree)
    component_names = _component_names(tree)
    bindings = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in EVENT_METHODS
        ):
            continue
        info = _handler_info(node, functions)
        if info is None:
            continue
        label, arity, is_variadic = info
        inputs_kw = next((kw.value for kw in node.keywords if kw.arg == "inputs"), None)
        inputs_count = _inputs_count(inputs_kw, component_names)
        test_id = f"gui.py:{node.lineno}:{label}"
        bindings.append((test_id, label, arity, inputs_count, is_variadic, node.lineno))
    return bindings


_BINDINGS = _collect_bindings()


@pytest.mark.parametrize(
    "test_id,label,arity,inputs_count,is_variadic,lineno", _BINDINGS, ids=[b[0] for b in _BINDINGS]
)
def test_handler_arity_matches_inputs(test_id, label, arity, inputs_count, is_variadic, lineno):
    if is_variadic:
        pytest.skip(f"gui.py:{lineno}: {label} takes *args, arity is not fixed (e.g. save_variant_choices)")
    if inputs_count is None:
        pytest.fail(
            f"gui.py:{lineno}: {label} is bound to a list-variable `inputs=` whose length can't be checked "
            "statically - either give it a literal inputs=[...] list, or make the handler take *args"
        )
    assert arity == inputs_count, (
        f"gui.py:{lineno}: {label} takes {arity} positional args but is wired to inputs= with "
        f"{inputs_count} value(s) - Gradio binds these positionally, so this will TypeError at click time"
    )


def test_generation_handlers_are_covered():
    """Confirms the five generate_* handlers driving each tab's main button actually went
    through an event binding this file parsed - if gui.py's wiring style changes enough that
    _collect_bindings stops seeing one, this fails loudly instead of the arity test above
    just quietly covering fewer handlers than intended."""
    covered = {b[1] for b in _BINDINGS}
    missing = GENERATION_HANDLERS - covered
    assert not missing, f"generation handler(s) not found wired to any event: {missing}"


# --- Keyword-argument contract: gui.py's gc.<fn>(...)/client.<fn>(...) calls must only use
# keywords that generate_character.py/comfyui_client.py's functions still accept. Both
# modules import only `requests` (and each other) at module scope - no torch, no gradio -
# so they're safe to import for real here and use `inspect.signature` on the target side,
# while the call side (gui.py) still has to stay AST-only.

import comfyui_client as client_module  # noqa: E402
import generate_character as gc_module  # noqa: E402

_ALIAS_MODULES = {"gc": gc_module, "client": client_module}


def _collect_kwarg_calls():
    """Every `gc.<fn>(...)`/`client.<fn>(...)` call in gui.py that passes keyword args,
    as (test id, fn name, sorted kwarg names, target module), skipping calls that splat
    `**kwargs` (name set not visible statically) and skipping targets that themselves
    declare `**kwargs` (any keyword is legal for them)."""
    tree = _load_tree()
    calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _ALIAS_MODULES
        ):
            continue
        if any(kw.arg is None for kw in node.keywords):
            continue  # **kwargs splat - names not visible statically
        kwarg_names = [kw.arg for kw in node.keywords]
        if not kwarg_names:
            continue
        module = _ALIAS_MODULES[node.func.value.id]
        fn = getattr(module, node.func.attr, None)
        if fn is None or not inspect.isfunction(fn):
            continue
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            continue  # target itself accepts **kwargs - any keyword is legal
        test_id = f"gui.py:{node.lineno}:{node.func.value.id}.{node.func.attr}"
        calls.append((test_id, node.func.attr, tuple(sorted(kwarg_names)), module))
    return calls


_KWARG_CALLS = _collect_kwarg_calls()


@pytest.mark.parametrize("test_id,fn_name,kwarg_names,module", _KWARG_CALLS, ids=[c[0] for c in _KWARG_CALLS])
def test_call_keywords_are_accepted_by_target(test_id, fn_name, kwarg_names, module):
    fn = getattr(module, fn_name)
    valid = set(inspect.signature(fn).parameters)
    unknown = set(kwarg_names) - valid
    assert not unknown, (
        f"{test_id} passes keyword(s) {sorted(unknown)} that {module.__name__}.{fn_name} no longer "
        f"accepts (current params: {sorted(valid)}) - a rename on the {module.__name__} side wasn't "
        "carried over to gui.py, this would only fail at button-click time otherwise"
    )


def test_kwarg_contract_actually_checked_something():
    """Sanity check on the harness itself: if gui.py's calls ever stopped matching the
    `gc.<fn>(...)`/`client.<fn>(...)` shape this scan looks for, the parametrized test above
    would silently collect zero cases and always "pass"."""
    assert len(_KWARG_CALLS) >= 5
