"""Tell a running H3 job that the user approved Google Drive access in the browser.

    ComfyUI\\.venv\\Scripts\\python.exe .claude\\skills\\minimax-h3-colab\\scripts\\consent_done.py

run.py mounts Google Drive on every new Colab VM, and Google asks for consent each time. The runner waits
for this signal (or `drive.consent_wait_seconds`, whichever comes first) before it lets the CLI go on.
Run it only after the user said they clicked through the consent page: sending the go-ahead early makes the
mount fail, and the session then downloads the models again and stores nothing on Drive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h3_colab as h3  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        config = h3.load_config()
    except h3.H3Error as exc:
        print(json.dumps({"ok": False, "error_code": exc.code, "error": str(exc)}))
        return 2
    path = h3.consent_signal_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(h3.now_iso(), encoding="utf-8")
    print(json.dumps({"ok": True, "signal": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
