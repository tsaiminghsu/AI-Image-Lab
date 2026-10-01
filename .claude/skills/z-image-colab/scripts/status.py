"""Show the Z-Image queue, the controller and (unless --local) the Colab sessions on the account.

    python status.py            # local queue + controller + `colab sessions` (no compute units)
    python status.py --local    # local only, no Docker call
    python status.py --job ID   # one job's full record (without the graph)

A zimg-* session with no live controller is billing for nothing: stop it with `worker.py up` (which
sweeps orphans) or the printed `colab stop` command.
"""

from __future__ import annotations

import argparse
import json
import sys

import zimage_colab as z


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Z-Image Colab worker status.")
    p.add_argument("--local", action="store_true")
    p.add_argument("--job")
    p.add_argument("--recent", type=int, default=10)
    args = p.parse_args(argv)
    config = z.load_config()
    root = z.output_root(config)
    store = z.JobStore(root)
    if args.job:
        job = store.get(args.job)
        job.pop("graph", None)
        print(json.dumps(job, ensure_ascii=False, indent=2))
        return 0
    jobs = store.list()
    counts = {s: sum(1 for j in jobs if j["status"] == s) for s in z.STATUSES}
    active = [
        {k: j.get(k) for k in ("job_id", "status", "attempt", "session", "width", "height", "seed")}
        for j in jobs
        if j["status"] not in z.TERMINAL
    ]
    recent = [
        {k: j.get(k) for k in ("job_id", "status", "generation_seconds", "gpu", "output", "completed_at")}
        for j in sorted((j for j in jobs if j["status"] in z.TERMINAL), key=lambda j: j.get("completed_at") or "")[
            -args.recent :
        ]
    ]
    controller = z.ControllerState(root)
    result = {
        "counts": {k: v for k, v in counts.items() if v},
        "active": active,
        "recent": recent,
        "controller": controller.read(),
        "controller_live": bool(controller.live()),
    }
    if not args.local:
        transport = z.make_transport(config)
        try:
            sessions = z.colab_sessions(transport)
            result["colab_sessions"] = sessions
            if not result["controller_live"]:
                result["orphaned"] = [
                    {"session": s["name"], "stop": transport.display(["stop", "--session", s["name"]])}
                    for s in sessions
                    if s["name"].startswith(z.SESSION_PREFIX)
                ]
        except z.ZImageError as exc:
            result["colab_sessions_error"] = exc.as_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
