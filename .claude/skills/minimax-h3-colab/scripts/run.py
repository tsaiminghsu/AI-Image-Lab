"""Generate one MiniMax H3 clip from one image on a Google Colab GPU.

    python run.py --image input/test.png --prompt "The woman walks slowly toward the camera ..." \
        --duration 8 --output output/test.mp4

The last stdout line is the job record as JSON (for Claude to parse); progress goes to stderr and to
output/logs/<job_id>.log. Exit codes: 0 completed, 2 invalid input, 3 timeout, 130 cancelled, 1 other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h3_colab as h3

EXIT_CODES = {h3.COMPLETED: 0, h3.DRY_RUN: 0, h3.TIMEOUT: 3, h3.CANCELLED: 130}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniMax H3 image-to-video on a Google Colab GPU")
    p.add_argument("--job", type=Path, help="JSON job spec (input/prompt/settings/output); flags override it")
    p.add_argument("--image", "-i", help="first-frame image (png/jpg/webp)")
    p.add_argument("--prompt", "-p", help="English shot description, kept verbatim (see prompts/video_prompt.md)")
    p.add_argument("--prompt-file", type=Path, help="complete H3 I2VA prompt, sent verbatim instead of --prompt")
    p.add_argument("--soundscape", help="overall_soundscape field (ambient and action sounds)")
    p.add_argument("--music", help="non_diegetic_music field (default N/A)")
    p.add_argument("--constraint", action="append", choices=h3.CONSTRAINTS, help="repeatable")
    p.add_argument("--duration", type=float, help="seconds, 4-15 (default from config: 8)")
    p.add_argument("--resolution", help="WxH (multiples of 32) or auto (default: image aspect, short side 768)")
    p.add_argument("--seed", type=int)
    p.add_argument("--output", "-o", help="output .mp4 (default output/<image>_h3_<job_id>.mp4)")
    p.add_argument("--gpu", choices=h3.GPUS)
    p.add_argument("--no-high-mem", action="store_true", help="do not request a high-RAM runtime")
    p.add_argument("--timeout", type=int, help="seconds for the remote run (default from config: 3600)")
    p.add_argument("--job-id")
    p.add_argument("--scene-id")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="validate, stage and print the commands; spends no CU")
    p.add_argument("--config", type=Path, help="config file (default config/config.json if present)")
    return p


def spec_from_args(args: argparse.Namespace) -> h3.JobSpec:
    if args.job:
        spec = h3.JobSpec.from_dict(json.loads(args.job.read_text(encoding="utf-8")))
    else:
        if not args.image:
            raise h3.InputError("--image is required")
        spec = h3.JobSpec(image=args.image)
    overrides = {
        "image": args.image,
        "description": args.prompt,
        "prompt_file": args.prompt_file,
        "soundscape": args.soundscape,
        "music": args.music,
        "constraints": tuple(args.constraint) if args.constraint else None,
        "duration": args.duration,
        "resolution": args.resolution,
        "seed": args.seed,
        "output": args.output,
        "gpu": args.gpu,
        "high_mem": False if args.no_high_mem else None,
        "timeout": args.timeout,
        "job_id": args.job_id,
        "scene_id": args.scene_id,
        "overwrite": True if args.overwrite else None,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(spec, key, value)
    return spec


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = h3.load_config(args.config)
        spec = spec_from_args(args)
        record = h3.run_job(spec, config, dry_run=args.dry_run)
    except h3.H3Error as exc:
        record = {"status": exc.status, "error_code": exc.code, "error": str(exc), "hint": exc.hint or None}
    except (OSError, ValueError) as exc:
        record = {"status": h3.FAILED, "error_code": "INVALID_INPUT", "error": str(exc)}

    status = record.get("status")
    if status == h3.DRY_RUN:
        print(record["dry_run"]["prompt"], file=sys.stderr)
        for command in record["dry_run"]["commands"]:
            print("  " + command, file=sys.stderr)
        for check in record["dry_run"]["preflight"]:
            mark = "ok  " if check["ok"] else "FAIL"
            print(f"[{mark}] {check['name']}: {check['detail']}", file=sys.stderr)
            if check.get("hint"):
                print(f"       -> {check['hint']}", file=sys.stderr)
    elif status == h3.COMPLETED:
        probe = record.get("ffprobe") or {}
        print(
            f"COMPLETED {record['output']} ({probe.get('width')}x{probe.get('height')}, "
            f"{probe.get('video_duration')} s, {probe.get('fps')} fps, GPU {record.get('gpu')}, "
            f"{record.get('elapsed_seconds')} s, CU used {record.get('cu_used_measured')})",
            file=sys.stderr,
        )
    else:
        print(f"{status} ({record.get('error_code')}): {record.get('error')}", file=sys.stderr)
        if record.get("hint"):
            print(f"-> {record['hint']}", file=sys.stderr)
    print(json.dumps(record, ensure_ascii=False, default=str))
    if status == h3.DRY_RUN and not all(c["ok"] for c in record["dry_run"]["preflight"]):
        return 1
    if status in EXIT_CODES:
        return EXIT_CODES[status]
    return 2 if record.get("error_code") in ("INVALID_INPUT", "PROMPT_REJECTED", "INVALID_CONFIG") else 1


if __name__ == "__main__":
    sys.exit(main())
