"""Preflight for the Z-Image Colab worker. Spends no compute units: it never creates a runtime.

    python setup_colab.py preflight

Checks Docker Desktop, the google-colab-cli image (shared with the minimax-h3-colab skill), the Colab
sign-in (`colab usage`), what is running on the account (`colab sessions`) and the config. On
AUTH_REQUIRED the JSON carries the login command for the user to run in their own terminal.
"""

from __future__ import annotations

import argparse
import json
import sys

import zimage_colab as z


def preflight() -> dict:
    report: dict = {"ok": True, "checks": []}

    def add(name: str, ok: bool, detail=None, hint: str = "") -> None:
        report["checks"].append({"check": name, "ok": ok, "detail": detail, "hint": hint})
        report["ok"] = report["ok"] and ok

    try:
        config = z.load_config()
        add(
            "config",
            True,
            {
                "gpu": config["colab"]["gpu"],
                "drive": config["colab"]["drive"],
                "persistent_path": config["colab"]["persistent_path"],
                "idle_timeout_seconds": config["worker"]["idle_timeout_seconds"],
                "comfyui": f"{config['comfyui']['ref']} ({config['comfyui']['commit'][:10]})",
                "model": f"{config['model']['repo']}@{config['model']['revision'][:10]}",
                "output": str(z.output_root(config)),
            },
        )
    except z.ZImageError as exc:
        add("config", False, exc.as_dict())
        return report
    transport = z.make_transport(config)
    try:
        transport.preflight()
        add("docker", True, config["colab"]["docker_image"])
    except z.ZImageError as exc:
        add("docker", False, str(exc), exc.hint)
        return report
    try:
        usage = z.colab_usage(transport)
        add("colab_auth", True, usage)
    except z.ZImageError as exc:
        add("colab_auth", False, str(exc), exc.hint)
        return report
    sessions = z.colab_sessions(transport)
    others = [s["name"] for s in sessions if not s["name"].startswith(z.SESSION_PREFIX)]
    ours = [s["name"] for s in sessions if s["name"].startswith(z.SESSION_PREFIX)]
    add(
        "colab_sessions",
        not others,
        {"other_runtimes": others, "zimg_runtimes": ours},
        "another runtime is active (e.g. an H3 job): ask its owner before `worker.py up`" if others else "",
    )
    live = z.ControllerState(z.output_root(config)).live()
    report["controller"] = live
    store = z.JobStore(z.output_root(config))
    report["queued_jobs"] = len(store.list((z.PENDING, z.QUEUED, z.RUNNING)))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Z-Image Colab worker setup checks (no compute units).")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    p.parse_args(argv)
    report = preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
