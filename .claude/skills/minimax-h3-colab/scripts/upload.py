"""Upload one local file to a running Colab session (debugging and recovery; run.py does its own).

python upload.py --session h3-20260930-120000-abc123 input/test.png /content/test.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h3_colab as h3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", required=True)
    p.add_argument("local", type=Path)
    p.add_argument("remote")
    args = p.parse_args(argv)
    if not h3.SESSION_RE.fullmatch(args.session):
        print("invalid session name", file=sys.stderr)
        return 2
    if not args.local.is_file():
        print(f"not a file: {args.local}", file=sys.stderr)
        return 2
    try:
        config = h3.load_config()
        transport = h3.make_transport(config)
        transport.preflight()
        print(
            h3.upload_file(
                transport, args.session, args.local, args.remote, timeout=config["colab"]["transfer_timeout_seconds"]
            )
        )
    except (h3.H3Error, h3.ColabCommandError, h3.ColabTimeout) as exc:
        print(f"upload failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
