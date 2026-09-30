"""Download one file from a running Colab session and ffprobe it if it is a video (debugging/recovery,
e.g. saving a finished clip after the local runner was interrupted before DOWNLOADING).

    python download.py --session h3-20260930-120000-abc123 /content/h3_<job_id>_output.mp4 output/rescued.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h3_colab as h3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", required=True)
    p.add_argument("remote")
    p.add_argument("local", type=Path)
    args = p.parse_args(argv)
    if not h3.SESSION_RE.fullmatch(args.session):
        print("invalid session name", file=sys.stderr)
        return 2
    try:
        config = h3.load_config()
        transport = h3.make_transport(config)
        transport.preflight()
        print(
            h3.download_file(
                transport, args.session, args.remote, args.local, timeout=config["colab"]["transfer_timeout_seconds"]
            )
        )
    except (h3.H3Error, h3.ColabCommandError, h3.ColabTimeout) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    ffprobe = h3.find_ffprobe(config)
    if ffprobe and args.local.suffix.lower() == ".mp4":
        try:
            print(json.dumps(h3.summarize_probe(h3.FFprobe(ffprobe).video(args.local)), indent=2))
        except h3.ProbeError as exc:
            print(f"ffprobe cannot read {args.local}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
