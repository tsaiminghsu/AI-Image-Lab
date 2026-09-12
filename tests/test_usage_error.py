"""UsageError is the library's caller-error signal; SystemExit is the CLI's exit mechanism.

Keeping those separate is what fixed the two boundary bugs (image_api's worker thread and
gui.py's queue task both swallowed SystemExit incorrectly, because it derives from
BaseException rather than Exception). These tests hold the separation in place and pin the
CLI behaviour it must not change.
"""

import ast
import io
import os
import subprocess
import sys

import generate_character as gc
import pytest

SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "generate_character.py")


def test_usage_error_is_an_ordinary_exception():
    """The whole point: `except Exception` at any boundary must catch it."""
    assert issubclass(gc.UsageError, Exception)
    assert not issubclass(gc.UsageError, SystemExit)


def test_library_code_raises_no_systemexit():
    """Every `raise SystemExit` below the CLI layer is a boundary bug waiting to happen. Only
    main() and the __main__ guard - the CLI's own exit path - may use it."""
    tree = ast.parse(io.open(SOURCE, encoding="utf-8").read(), SOURCE)
    cli_only = {"main"}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in cli_only:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Raise)
                and isinstance(inner.exc, ast.Call)
                and isinstance(inner.exc.func, ast.Name)
                and inner.exc.func.id == "SystemExit"
            ):
                offenders.append(f"{node.name} (line {inner.lineno})")
    assert not offenders, f"library code must raise UsageError, not SystemExit: {offenders}"


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"trigger": "definitely-not-a-character"}, "unknown character"),
        ({"checkpoint": "definitely-not-a-checkpoint"}, "unknown checkpoint"),
        ({"checkpoint": "z_image_turbo", "anchor_path": "foo.png"}, "is Z-Image"),
        ({"pose_name": "not-a-pose", "hq": True}, "unknown pose"),
    ],
)
def test_gen_custom_rejections_raise_usage_error(kwargs, fragment, tmp_path):
    call = {
        "prompt": "x",
        "extra_negative": "",
        "tier": "safe",
        "trigger": None,
        "anchor_path": None,
        "out_dir": str(tmp_path),
        "seed": 1,
        "filename": "f",
        "ip_adapter_weight": 0.8,
    }
    call.update(kwargs)
    with pytest.raises(gc.UsageError) as excinfo:
        gc.gen_custom(**call)
    assert fragment in str(excinfo.value)


def test_get_character_rejects_unknown_trigger():
    with pytest.raises(gc.UsageError, match="unknown character"):
        gc.get_character("definitely-not-a-character")


def test_cli_still_exits_1_with_the_message_on_stderr(tmp_path):
    """The contract the migration must not change: `raise SystemExit(msg)` printed msg to
    stderr and exited 1, and main()'s UsageError handler has to reproduce that exactly."""
    proc = subprocess.run(
        [
            sys.executable,
            SOURCE,
            "custom",
            "--prompt",
            "x",
            "--checkpoint",
            "z_image_turbo",
            "--anchor",
            "foo.png",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(SOURCE),
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr.strip().startswith("checkpoint 'z_image_turbo' is Z-Image")
