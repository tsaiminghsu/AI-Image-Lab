"""Preflight for the MiniMax H3 skill: config, ffprobe, Docker/colab CLI, Colab sign-in and CU balance.

    python check.py            # human-readable, exit 0 when everything is ready
    python check.py --json     # the same as JSON

Spends no compute units: `colab usage` only reads the account balance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h3_colab as h3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true")
    p.add_argument("--config", type=Path)
    args = p.parse_args(argv)
    try:
        config = h3.load_config(args.config)
    except h3.H3Error as exc:
        checks = [{"name": "config", "ok": False, "code": exc.code, "detail": str(exc)}]
    else:
        checks = h3.preflight_checks(config)
    ok = all(c["ok"] for c in checks)
    if args.json:
        print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, default=str))
    else:
        for c in checks:
            print(f"[{'ok  ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
            if c.get("hint"):
                print(f"       -> {c['hint']}")
        print("ready" if ok else "not ready")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
