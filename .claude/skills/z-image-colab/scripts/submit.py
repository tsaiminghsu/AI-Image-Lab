"""Add Z-Image txt2img jobs to the local queue (output/zimage/jobs/). Spends nothing: no Colab call.

    python submit.py --prompt "..." [--aspect 16:9 | --width W --height H] [--count 4] [--seed -1]
    python submit.py --manifest batch.json      # {"batch": "name", "jobs": [{"prompt": ..., ...}, ...]}

Every job of one call is validated before any is stored, so a bad entry leaves the queue unchanged.
The last stdout line is JSON. Then run `worker.py up` (or it picks the jobs up if it is already running).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import zimage_colab as z

SPEC_KEYS = ("prompt", "negative", "aspect", "width", "height", "steps", "cfg", "seed", "solo", "allow_text", "count")


def specs_from_args(args) -> list[dict]:
    if args.manifest:
        data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), list) or not data["jobs"]:
            raise z.UsageError('the manifest must be {"batch": "...", "jobs": [{...}, ...]}')
        specs = []
        for i, entry in enumerate(data["jobs"]):
            unknown = set(entry) - set(SPEC_KEYS)
            if unknown:
                raise z.UsageError(f"manifest job {i}: unknown keys {sorted(unknown)}")
            specs.append(dict(entry, batch=data.get("batch") or args.batch))
        return specs
    if not args.prompt:
        raise z.UsageError("--prompt or --manifest is required")
    return [
        {
            "prompt": args.prompt,
            "negative": args.negative,
            "aspect": args.aspect,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "solo": args.solo,
            "allow_text": args.allow_text,
            "count": args.count,
            "batch": args.batch,
        }
    ]


def expand(spec: dict) -> list[dict]:
    """--count N: N jobs from one spec. A fixed seed becomes seed, seed+1, ...; -1 stays random per job."""
    count = spec.pop("count", None) or 1
    if not isinstance(count, int) or not 1 <= count <= 50:
        raise z.UsageError("count must be 1-50")
    seed = spec.get("seed", -1)
    out = []
    for i in range(count):
        one = dict(spec)
        if seed not in (None, -1) and count > 1:
            one["seed"] = seed + i
        out.append(one)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Queue Z-Image txt2img jobs for the Colab worker.")
    p.add_argument("--prompt")
    p.add_argument("--negative", help="extra negative terms; the safety negatives are always added")
    p.add_argument("--aspect", choices=sorted(z.ASPECTS))
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--seed", type=int, default=-1, help="-1 = random (the chosen seed is recorded)")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--solo", action="store_true", help="exactly one person: add the no-second-person negatives")
    p.add_argument("--allow-text", action="store_true", help="the user asked for lettering in the image")
    p.add_argument("--batch", help="a label stored on every job")
    p.add_argument("--manifest")
    p.add_argument("--dry-run", action="store_true", help="validate and print, store nothing")
    args = p.parse_args(argv)
    try:
        config = z.load_config()
        jobs = []
        for spec in specs_from_args(args):
            for one in expand(dict(spec)):
                jobs.append(z.build_job(config, **{k: v for k, v in one.items() if v is not None}))
        store = z.JobStore(z.output_root(config))
        if not args.dry_run:
            for job in jobs:
                store.add(job)
    except (z.UsageError, z.ZImageError, ValueError, OSError) as exc:
        code = getattr(exc, "code", "INVALID_INPUT")
        print(json.dumps({"ok": False, "error": {"code": code, "message": str(exc)}}, ensure_ascii=False))
        return 2
    live = z.ControllerState(z.output_root(config)).live()
    brief = [
        {k: j[k] for k in ("job_id", "width", "height", "steps", "cfg", "seed", "solo", "allow_text", "prompt")}
        for j in jobs
    ]
    result = {
        "ok": True,
        "dry_run": args.dry_run,
        "jobs": brief,
        "negative_prompt": jobs[0]["negative_prompt"] if jobs else None,
        "controller_running": bool(live),
        "next": "the running controller will pick these up"
        if live
        else "start the worker: python worker.py up (spends Colab compute units)",
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
