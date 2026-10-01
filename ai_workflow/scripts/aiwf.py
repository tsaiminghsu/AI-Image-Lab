"""Command line for the tool interface. One subcommand per tool, JSON on stdout.

    python scripts/aiwf.py list-workflows
    python scripts/aiwf.py create-job z-image-basic --prompt "A white ceramic mug on a white background" --set aspect=16:9
    python scripts/aiwf.py status JOB_ID
    python scripts/aiwf.py list-jobs --status pending
    python scripts/aiwf.py cancel JOB_ID
    python scripts/aiwf.py result JOB_ID --wait 600

Standard library only: any Python 3.11+ runs it, and it reads no credentials of any kind. The workspace is
found from --root, the AIWF_ROOT variable, the folder this script sits in (when run from the workspace), or
configs/local.json (when run from the repository).

Exit code 0 = the tool ran (read "status" in the output for the job's own state), 1 = it reported an error
(`{"ok": false, "error": {"code", "message", "hint"}}`), 2 = bad command line.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller import WorkflowError, jobs, tools  # noqa: E402


def parse_value(text):
    """`--set key=value`: the value is JSON when it parses as JSON (8, 2.5, true, {"a": 1}), text otherwise."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def build_parser():
    parser = argparse.ArgumentParser(prog="aiwf", description="Workflow tool interface (JSON output).")
    parser.add_argument("--root", help="workspace folder (default: AIWF_ROOT, this folder, or configs/local.json)")
    sub = parser.add_subparsers(dest="tool", required=True)

    sub.add_parser("list-workflows", help="workflows and the parameters each one takes")

    create = sub.add_parser("create-job", help="validate and queue a job")
    create.add_argument("workflow")
    create.add_argument("--prompt", help="shortcut for --set prompt=...")
    create.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a parameter or an input")
    create.add_argument("--values-json", help="all values as one JSON object (instead of, or besides, --set)")
    create.add_argument(
        "--attest", action="append", default=[], metavar="INPUT", help="confirm the origin of a file input"
    )
    create.add_argument("--task-type")

    status = sub.add_parser("status", help="state of one job")
    status.add_argument("job_id")

    listing = sub.add_parser("list-jobs", help="jobs in the workspace")
    listing.add_argument("--status", choices=jobs.STATUSES)
    listing.add_argument("--limit", type=int, default=50)

    cancel = sub.add_parser("cancel", help="ask for a job to be cancelled")
    cancel.add_argument("job_id")

    result = sub.add_parser("result", help="output of a finished job")
    result.add_argument("job_id")
    result.add_argument(
        "--wait", type=float, default=0, metavar="SECONDS", help="poll until the job ends, at most this long"
    )
    return parser


def collect_values(args):
    values = {}
    if args.values_json:
        loaded = json.loads(args.values_json)
        if not isinstance(loaded, dict):
            raise WorkflowError("INVALID_INPUT", "--values-json must be a JSON object")
        values.update(loaded)
    for item in args.set:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise WorkflowError("INVALID_INPUT", "--set takes NAME=VALUE, got %r" % item)
        values[name] = parse_value(value)
    if args.prompt is not None:
        values["prompt"] = args.prompt
    for name in args.attest:
        ref = values.get(name)
        if isinstance(ref, str):
            ref = {"source": "file", "path": ref}
        if not isinstance(ref, dict):
            raise WorkflowError(
                "INVALID_INPUT", "--attest %s: give that input with --set %s=inputs/<file>" % (name, name)
            )
        values[name] = dict(ref, attested=True)
    return values


def run(args):
    if args.tool == "list-workflows":
        return tools.list_workflows(args.root)
    if args.tool == "create-job":
        return tools.create_job(args.workflow, collect_values(args), task_type=args.task_type, root=args.root)
    if args.tool == "status":
        return tools.get_job_status(args.job_id, args.root)
    if args.tool == "list-jobs":
        return tools.list_jobs(args.status, args.limit, args.root)
    if args.tool == "cancel":
        return tools.cancel_job(args.job_id, args.root)
    deadline = time.monotonic() + args.wait
    while True:
        out = tools.get_result(args.job_id, args.root)
        if out["status"] in jobs.TERMINAL or time.monotonic() >= deadline:
            return out
        time.sleep(5)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        out = dict(run(args), ok=True)
        code = 0
    except WorkflowError as exc:
        out, code = {"ok": False, "error": exc.as_dict()}, 1
    except ValueError as exc:
        out, code = {"ok": False, "error": {"code": "INVALID_INPUT", "message": str(exc)}}, 1
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
