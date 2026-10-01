"""Generate MiniMax H3 clips from images on a Google Colab GPU.

One clip:
    python run.py --image input/test.png --prompt "The woman walks slowly toward the camera ..." \
        --duration 8 --output output/test.mp4

Several clips in ONE Colab session (the ComfyUI install and the 40 GB model download are paid once):
    python run.py --manifest batch.json                  # {"batch_id": ..., "jobs": [job spec, ...]}
    python run.py --image a.png --prompt "..." --segments 2 --output output/a.mp4
        # 2 chained clips (a_part1.mp4, a_part2.mp4); part 2 starts from part 1's last frame

The last stdout line is JSON for Claude to parse: the job record for one clip, the batch summary
(with every record under "records") for a batch. Progress goes to stderr and output/logs/.
Exit codes: 0 completed, 2 invalid input, 3 timeout, 130 cancelled, 1 other (or any failed clip in a batch).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import h3_colab as h3

EXIT_CODES = {h3.COMPLETED: 0, h3.DRY_RUN: 0, h3.TIMEOUT: 3, h3.CANCELLED: 130}
INPUT_CODES = ("INVALID_INPUT", "PROMPT_REJECTED", "INVALID_CONFIG")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniMax H3 image-to-video on a Google Colab GPU")
    p.add_argument("--job", type=Path, help="JSON job spec (input/prompt/settings/output); flags override it")
    p.add_argument("--manifest", type=Path, help='batch file {"batch_id"?, "jobs": [job spec, ...]}; one session')
    p.add_argument(
        "--segments", type=int, help="split into N chained clips in one session (each starts from the last frame)"
    )
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
    p.add_argument("--settle", type=int, help="seconds to wait after the run before re-reading the CU balance")
    p.add_argument("--ignore-busy", action="store_true", help="start even if another Colab runtime is active")
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


def specs_from_manifest(args: argparse.Namespace) -> tuple[list[h3.JobSpec], str | None]:
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise h3.InputError(f'{args.manifest} needs a non-empty "jobs" list')
    specs = [h3.JobSpec.from_dict(job) for job in jobs]
    for spec in specs:  # session-wide flags apply to every job
        if args.gpu:
            spec.gpu = args.gpu
        if args.no_high_mem:
            spec.high_mem = False
        if args.timeout:
            spec.timeout = args.timeout
        if args.overwrite:
            spec.overwrite = True
    return specs, data.get("batch_id") or args.job_id


def segment_specs(base: h3.JobSpec, n: int) -> list[h3.JobSpec]:
    if not 2 <= n <= h3.MAX_BATCH_JOBS:
        raise h3.InputError(f"--segments must be 2-{h3.MAX_BATCH_JOBS}")
    if base.chain:
        raise h3.InputError("--segments starts from an image; the first segment cannot chain")
    specs = []
    for k in range(1, n + 1):
        output = None
        if base.output:
            out = Path(base.output)
            output = str(out.with_name(f"{out.stem}_part{k}{out.suffix}"))
        specs.append(
            dataclasses.replace(
                base,
                image=base.image if k == 1 else None,
                chain=k > 1,
                output=output,
                job_id=f"{base.job_id}-p{k}" if base.job_id else None,
                seed=base.seed if k == 1 else None,
            )
        )
    return specs


def report(record: dict) -> None:
    status = record.get("status")
    name = record.get("job_id") or ""
    if status == h3.DRY_RUN:
        print(f"[{name}] " + record["dry_run"]["prompt"], file=sys.stderr)
        for command in record["dry_run"]["commands"]:
            print("  " + command, file=sys.stderr)
    elif status == h3.COMPLETED:
        probe = record.get("ffprobe") or {}
        print(
            f"COMPLETED {record['output']} ({probe.get('width')}x{probe.get('height')}, "
            f"{probe.get('video_duration')} s, {probe.get('fps')} fps, GPU {record.get('gpu')}, "
            f"{record.get('elapsed_seconds')} s, CU used {record.get('cu_used_measured')})",
            file=sys.stderr,
        )
    else:
        print(f"{status} {name} ({record.get('error_code')}): {record.get('error')}", file=sys.stderr)
        if record.get("hint"):
            print(f"-> {record['hint']}", file=sys.stderr)


def report_preflight(record: dict) -> bool:
    checks = record["dry_run"]["preflight"]
    for check in checks:
        mark = "ok  " if check["ok"] else "FAIL"
        print(f"[{mark}] {check['name']}: {check['detail']}", file=sys.stderr)
        if check.get("hint"):
            print(f"       -> {check['hint']}", file=sys.stderr)
    return all(c["ok"] for c in checks)


def single(args: argparse.Namespace, config: dict) -> int:
    try:
        record = h3.run_job(
            spec_from_args(args), config, dry_run=args.dry_run, settle_seconds=args.settle, ignore_busy=args.ignore_busy
        )
    except h3.H3Error as exc:
        record = {"status": exc.status, "error_code": exc.code, "error": str(exc), "hint": exc.hint or None}
    except (OSError, ValueError) as exc:
        record = {"status": h3.FAILED, "error_code": "INVALID_INPUT", "error": str(exc)}
    report(record)
    preflight_ok = report_preflight(record) if record.get("status") == h3.DRY_RUN else True
    print(json.dumps(record, ensure_ascii=False, default=str))
    status = record.get("status")
    if status == h3.DRY_RUN and not preflight_ok:
        return 1
    if status in EXIT_CODES:
        return EXIT_CODES[status]
    return 2 if record.get("error_code") in INPUT_CODES else 1


def batch(args: argparse.Namespace, config: dict) -> int:
    try:
        if args.manifest:
            specs, batch_id = specs_from_manifest(args)
        else:
            specs, batch_id = segment_specs(spec_from_args(args), args.segments), args.job_id
        summary = h3.run_batch(
            specs,
            config,
            dry_run=args.dry_run,
            settle_seconds=args.settle,
            ignore_busy=args.ignore_busy,
            batch_id=batch_id,
        )
    except h3.H3Error as exc:
        summary = {"status": exc.status, "error_code": exc.code, "error": str(exc), "hint": exc.hint or None}
    except (OSError, ValueError) as exc:
        summary = {"status": h3.FAILED, "error_code": "INVALID_INPUT", "error": str(exc)}
    records = summary.get("records") or []
    for record in records:
        report(record)
    preflight_ok = report_preflight(records[0]) if records and records[0].get("status") == h3.DRY_RUN else True
    if "records" not in summary:
        print(f"{summary['status']} ({summary.get('error_code')}): {summary.get('error')}", file=sys.stderr)
    elif summary.get("status") == h3.DRY_RUN:
        prepared = sum(r.get("status") == h3.DRY_RUN for r in records)
        print(f"dry run {summary['batch_id']}: {prepared} of {len(records)} jobs ready, nothing spent", file=sys.stderr)
    else:
        print(
            f"batch {summary['batch_id']}: {summary.get('completed')} of {len(records)} completed, "
            f"{len(summary.get('sessions') or [])} Colab session(s), CU used {summary.get('cu_used_measured')}"
            f" (after settle {summary.get('cu_used_settled')})",
            file=sys.stderr,
        )
    print(json.dumps(summary, ensure_ascii=False, default=str))
    status = summary.get("status")
    if status == h3.DRY_RUN:
        return 0 if preflight_ok else 1
    if status == h3.COMPLETED:
        return 0
    codes = [r.get("error_code") for r in records if r.get("status") != h3.COMPLETED] or [summary.get("error_code")]
    return 2 if all(c in INPUT_CODES for c in codes) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = h3.load_config(args.config)
    except h3.H3Error as exc:
        print(json.dumps({"status": exc.status, "error_code": exc.code, "error": str(exc)}, ensure_ascii=False))
        return 2
    if args.manifest or (args.segments or 0) > 1:
        return batch(args, config)
    return single(args, config)


if __name__ == "__main__":
    sys.exit(main())
