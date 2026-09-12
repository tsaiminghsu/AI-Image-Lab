"""Freezes generate_character.py's CLI surface (subcommands + flags) so a refactor can't
silently rename/drop/add a user-facing flag. This is the project's hard constraint from
the refactor plan: "do not change the existing CLI subcommands/flags" without deliberate
sign-off - changing FROZEN_CLI below IS that sign-off, so only do it after confirming the
CLI change is intentional, never just to make a failing test pass.

The parser is built inside `if __name__ == "__main__":` (training/generate_character.py,
around line 1208), so it can't be imported without also faking CHARACTERS/pose_skeletons/
client wiring just to construct the object. AST reads the add_argument() calls straight out
of the source instead.
"""

import ast
import os

import pytest

GENERATE_CHARACTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "generate_character.py"
)

# Frozen 2026-09-12 from training/generate_character.py by actually parsing it (see
# _parse_cli_surface below) - not hand-typed. "__top__" is the top-level parser's own
# flags, i.e. what has to come BEFORE the subcommand name (currently just --variant).
# Changing any set here means changing the real CLI: update only after confirming that's
# the intended, signed-off change.
FROZEN_CLI = {
    "__top__": {"--variant"},
    "list-characters": set(),
    "anchor": {"--character", "--out", "--seeds"},
    "variations": {"--character", "--anchor", "--out", "--count", "--ip-adapter-weight", "--seed"},
    "test-suggestive": {"--character", "--anchor", "--out", "--seed", "--ip-adapter-weight"},
    "variations-suggestive": {"--character", "--anchor", "--out", "--count", "--ip-adapter-weight", "--seed"},
    "custom": {
        "--prompt",
        "--negative-prompt",
        "--tier",
        "--character",
        "--anchor",
        "--out",
        "--seed",
        "--filename",
        "--ip-adapter-weight",
        "--pose",
        "--pose-reference",
        "--controlnet-strength",
        "--width",
        "--height",
        "--use-facedetailer",
        "--facedetailer-denoise",
        "--facedetailer-backend",
        "--style-positive",
        "--style-negative",
        "--checkpoint",
        "--lora-strength",
        "--no-hq",
        "--no-facedetailer",
        "--character-lora-strength",
        "--hires-denoise",
    },
    "gif": {
        "--prompt",
        "--negative-prompt",
        "--tier",
        "--character",
        "--anchor",
        "--out",
        "--seed",
        "--frames",
        "--duration-ms",
        "--denoise",
        "--ip-adapter-weight",
        "--width",
        "--height",
        "--style-positive",
        "--style-negative",
        "--checkpoint",
        "--lora-strength",
    },
    "video": {"--character", "--init-image", "--out", "--seed", "--frames", "--fps", "--motion"},
    "video-animatediff": {
        "--prompt",
        "--negative-prompt",
        "--tier",
        "--character",
        "--face-ref",
        "--out",
        "--seed",
        "--ip-adapter-weight",
        "--facedetailer-denoise",
        "--facedetailer-steps",
        "--faceid-v2-weight",
        "--faceid-lora-strength",
        "--motion-scale",
        "--face-report",
        "--no-facedetailer",
        "--frames",
        "--fps",
        "--width",
        "--height",
        "--style-positive",
        "--style-negative",
        "--checkpoint",
        "--motion-lora",
        "--motion-lora-strength",
        "--no-hires",
        "--hires-scale",
        "--hires-denoise",
        "--upscale-to",
        "--interp",
        "--lcm",
        "--lcm-preset",
    },
    "talk": {
        "--image",
        "--audio",
        "--out",
        "--size",
        "--preprocess",
        "--still",
        "--expression-scale",
        "--enhancer",
        "--pose-style",
    },
}


def _is_argparser_call(call):
    func = call.func
    return (isinstance(func, ast.Name) and func.id == "ArgumentParser") or (
        isinstance(func, ast.Attribute) and func.attr == "ArgumentParser"
    )


def _parse_cli_surface():
    """Returns {subcommand_name: {flag, ...}} plus "__top__" for the top-level parser, read
    straight from the AST: find the `ArgumentParser()` and `add_subparsers()` assignments,
    map each subparser variable (`p_x = sub.add_parser("name", ...)`, or a bare
    `sub.add_parser("name")` with no follow-up args, e.g. list-characters) to its
    subcommand name, then collect every add_argument() call's first positional string
    against whichever variable it was called on.
    """
    with open(GENERATE_CHARACTER_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=GENERATE_CHARACTER_PATH)

    parser_var = None
    sub_var = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.targets[0], ast.Name):
            if _is_argparser_call(node.value):
                parser_var = node.targets[0].id
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "add_subparsers":
                sub_var = node.targets[0].id
    assert parser_var and sub_var, (
        "couldn't find the top-level ArgumentParser()/add_subparsers() assignments in "
        f"{GENERATE_CHARACTER_PATH} - did the CLI setup move or get restructured?"
    )

    # subparser variable name -> subcommand name
    var_to_cmd = {}
    for node in ast.walk(tree):
        call = None
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        if call is None:
            continue
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_parser" and isinstance(func.value, ast.Name)):
            continue
        if func.value.id != sub_var:
            continue
        name = call.args[0].value
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_to_cmd[node.targets[0].id] = name
        else:
            var_to_cmd[f"__bare__{name}"] = name  # e.g. sub.add_parser("list-characters"), never assigned

    surface = {cmd: set() for cmd in var_to_cmd.values()}
    surface["__top__"] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument" and isinstance(func.value, ast.Name)):
            continue
        owner = func.value.id
        if not call.args:
            continue
        flag = call.args[0].value
        if owner == parser_var:
            surface["__top__"].add(flag)
        elif owner in var_to_cmd:
            surface[var_to_cmd[owner]].add(flag)

    return surface


CLI_SURFACE = _parse_cli_surface()


def test_cli_subcommands_match_frozen_snapshot():
    found = set(CLI_SURFACE) - {"__top__"}
    frozen = set(FROZEN_CLI) - {"__top__"}
    added = found - frozen
    removed = frozen - found
    assert found == frozen, (
        f"generate_character.py's subcommand set changed - added {added or '{}'}, removed {removed or '{}'}. "
        "If intentional, update FROZEN_CLI in this file to match (deliberate sign-off)."
    )


@pytest.mark.parametrize("subcommand", sorted(FROZEN_CLI))
def test_cli_flags_match_frozen_snapshot(subcommand):
    if subcommand not in CLI_SURFACE:
        pytest.fail(f"subcommand {subcommand!r} is missing from generate_character.py's parser")
    found = CLI_SURFACE[subcommand]
    frozen = FROZEN_CLI[subcommand]
    added = found - frozen
    removed = frozen - found
    assert found == frozen, (
        f"generate_character.py subcommand {subcommand!r} flags changed - "
        f"added {added or '{}'}, removed {removed or '{}'}. "
        "If intentional, update FROZEN_CLI in this file to match (deliberate sign-off)."
    )
