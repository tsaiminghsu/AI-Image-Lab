"""Check every workflow template against a running ComfyUI's /object_info - without executing anything.

workflow_contracts.py pins which node id holds which class_type, but only a real server knows whether
that class exists in the installed ComfyUI and node packs, and what its inputs are called. This asks
the server (GET /object_info, no model loads, no GPU work) and reports any template node whose class
is unknown or that sets an input the class doesn't declare.

It is the cheap way to validate a template built for a GPU this machine doesn't have - the Wan 2.2
image-to-video template targets a RunPod worker pinned to the same ComfyUI commit as the local install,
so a clean report here means the worker will accept the graph.

    ComfyUI\\.venv\\Scripts\\python.exe training\\validate_workflow_nodes.py [template.json ...]
"""

import glob
import json
import os
import sys

import requests

import comfyui_client as client

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))


def _declared_inputs(spec):
    inputs = (spec or {}).get("input") or {}
    names = set()
    for group in ("required", "optional", "hidden"):
        names.update((inputs.get(group) or {}).keys())
    return names


def check_against_object_info(wf, object_info):
    """Problems (human-readable strings) for one workflow dict; empty = every class and input is known.

    Inputs like "codec.encoding.crf" are sub-options of a dynamic combo input ("codec"), so a dotted
    name is accepted when its first segment is a declared input."""
    problems = []
    for node_id, node in sorted(wf.items(), key=lambda kv: kv[0]):
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type")
        spec = object_info.get(cls)
        if spec is None:
            problems.append(f"node {node_id}: class_type {cls!r} is not installed on this ComfyUI")
            continue
        declared = _declared_inputs(spec)
        for name in (node.get("inputs") or {}):
            if name in declared or name.split(".", 1)[0] in declared:
                continue
            problems.append(f"node {node_id} ({cls}): input {name!r} is not declared "
                            f"(known: {', '.join(sorted(declared)) or 'none'})")
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = argv or sorted(glob.glob(os.path.join(TRAINING_DIR, "workflow_template*.json")))
    try:
        object_info = requests.get(f"{client.COMFYUI_URL}/object_info", timeout=60).json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        print(f"cannot read {client.COMFYUI_URL}/object_info - is ComfyUI running? ({exc})")
        return 2
    total = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        problems = check_against_object_info(wf, object_info)
        total += len(problems)
        print(f"{'OK  ' if not problems else 'FAIL'} {os.path.basename(path)}")
        for p in problems:
            print(f"     - {p}")
    print(f"\n{len(paths)} templates, {total} problems")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
