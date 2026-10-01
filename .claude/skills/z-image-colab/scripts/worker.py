"""Start or stop the Colab worker for the queued Z-Image jobs. `up` SPENDS COLAB COMPUTE UNITS.

    python worker.py up                 # detached controller: opens one GPU session, runs the queue,
                                        # stops the VM after worker.idle_timeout_seconds of no jobs
    python worker.py up --foreground    # same, attached to this terminal (debugging, tests)
    python worker.py stop               # ask the live controller to shut down now (it stops the VM)
    python worker.py consent-done       # the user approved Google Drive access in the browser

Google asks for Drive consent on every new Colab runtime. The controller opens the consent page in the
default browser and waits up to colab.drive_consent_wait_seconds; with consent the session uses the
persistent storage on Drive, without it the session downloads from the pinned public sources instead.

The controller is detached by default because it must outlive the terminal or Claude session that
started it: it is the only thing that releases the VM. Progress: output/zimage/logs/controller-*.log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import zimage_colab as z

WORKER_ENDED = ("worker_idle_timeout", "worker_max_session")


def run_foreground(args, config) -> dict:
    """One controller run per session. A job that arrived just as the worker shut down is left PENDING;
    start another session for it (bounded by --max-sessions)."""
    store = z.JobStore(z.output_root(config))
    summaries = []
    for _ in range(args.max_sessions):
        controller = z.Controller(config, transport=z.make_transport(config), store=store)
        summary = controller.run(ignore_busy=args.ignore_busy)
        summaries.append(summary)
        leftover = [j for j in store.list((z.PENDING,)) if j["attempt"] < z.MAX_ATTEMPTS]
        if summary.get("reason") not in WORKER_ENDED or not leftover:
            break
        print(f"{len(leftover)} job(s) arrived as the worker shut down; opening another session", file=sys.stderr)
    return {"sessions": summaries}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Z-Image Colab worker for the queued jobs.")
    sub = p.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up")
    up.add_argument("--foreground", action="store_true")
    up.add_argument("--ignore-busy", action="store_true", help="start even if another runtime is active")
    up.add_argument("--max-sessions", type=int, default=3)
    sub.add_parser("stop")
    sub.add_parser("consent-done", help="the user approved the Google Drive consent page: press Enter for them")
    args = p.parse_args(argv)
    try:
        config = z.load_config()
    except z.ZImageError as exc:
        print(json.dumps({"ok": False, "error": exc.as_dict()}))
        return 2
    root = z.output_root(config)
    state = z.ControllerState(root)

    if args.command == "consent-done":
        live = state.live()
        if not live or live.get("state") != "drive_consent":
            print(json.dumps({"ok": False, "status": "no_consent_pending", "controller": live}))
            return 1
        (root / "drive_consent.ok").write_text(z.now_iso(), encoding="utf-8")
        print(json.dumps({"ok": True, "status": "consent_signalled", "session": live.get("session")}))
        return 0

    if args.command == "stop":
        live = state.live()
        if not live:
            print(
                json.dumps(
                    {"ok": True, "status": "no_controller", "hint": "check `status.py` for orphaned zimg-* sessions"}
                )
            )
            return 0
        root.mkdir(parents=True, exist_ok=True)
        (root / "stop.request").write_text(z.now_iso(), encoding="utf-8")
        print(
            json.dumps({"ok": True, "status": "stop_requested", "session": live.get("session"), "pid": live.get("pid")})
        )
        return 0

    if not args.foreground:
        live = state.live()
        if live:
            print(
                json.dumps(
                    {"ok": True, "status": "already_running", "session": live.get("session"), "pid": live.get("pid")}
                )
            )
            return 0
        log = root / "logs" / f"controller-{time.strftime('%Y%m%d-%H%M%S')}.out"
        cmd = [sys.executable, __file__, "up", "--foreground", "--max-sessions", str(args.max_sessions)]
        if args.ignore_busy:
            cmd.append("--ignore-busy")
        pid = z.spawn_detached(cmd, log)
        print(json.dumps({"ok": True, "status": "started", "pid": pid, "log": str(log)}))
        return 0

    result = run_foreground(args, config)
    last = result["sessions"][-1] if result["sessions"] else {}
    result["ok"] = not last.get("error") and last.get("stop") != "stop_failed"
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
