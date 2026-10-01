"""Cancel queued Z-Image jobs.

    python cancel.py JOB_ID [JOB_ID ...]

PENDING jobs are cancelled at once. A job already handed to a session gets a cancel request that the
running controller forwards to the worker: a job that has not started is dropped, a running one is
interrupted in ComfyUI. Finished jobs are left as they are.
"""

from __future__ import annotations

import argparse
import json
import sys

import zimage_colab as z


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cancel Z-Image jobs.")
    p.add_argument("job_ids", nargs="+")
    args = p.parse_args(argv)
    store = z.JobStore(z.output_root(z.load_config()))
    results = {}
    for job_id in args.job_ids:
        try:
            results[job_id] = store.request_cancel(job_id)
        except z.UsageError as exc:
            results[job_id] = f"error: {exc}"
    print(json.dumps({"ok": True, "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
