"""Check an output image the way the worker and controller do.

    python validate.py output/zimage/<job_id>/result.png --width 1344 --height 768

Checks: file exists, size > 0, PNG structure (signature, IHDR size, chunk CRCs, IEND) and, with Pillow
installed (ComfyUI\\.venv has it), a full decode. Exit 0 = valid, 1 = problems (listed in the JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import zimage_colab as z


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate a generated PNG.")
    p.add_argument("path")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    args = p.parse_args(argv)
    problems = z.validate_output(Path(args.path), args.width, args.height)
    print(json.dumps({"ok": not problems, "path": args.path, "problems": problems}))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
