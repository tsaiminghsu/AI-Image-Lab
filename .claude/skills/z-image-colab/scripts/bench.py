"""Measure how many pictures per job (and which GPU) is cheapest, and compare looks. Spends nothing itself.

    python bench.py plan --gpu L4 [--looks]      write output/zimage/bench/<name>.manifest.json (for submit.py)
    python bench.py report [--gpu L4] [--html F]  read the finished jobs and sessions, print the tables

`plan` fills the manifest from the real promo storyboards (projects/activity_promo), so the numbers answer the
question that matters: what do the stills of a promo cost on this GPU. Groups (the job's `tag`):

  single      six different prompts, one picture each - the production baseline
  repeat      one prompt four times - how much of a job is the prompt change (text encoder) and how much sampling
  batch       one prompt as one job of 2, 4, 8 pictures - where the VRAM runs out and what a picture then costs
  hires       864x1536 as one job of 1, 2, 4 pictures
  look        (--looks) three shots x three looks x two sizes, plus steps 8 vs 12: the realism comparison
  r2          (plan --round2) one story's six shots written four ways (baseline, "early twenties", plus fixed
              features, plus one seed for all six) x two looks, to see whether the face reads as 23 and stays
              the same woman; `faces` then measures it with InsightFace (age estimate, identity similarity)

Order is small to large, so a job that runs out of memory comes last and costs the ones before it nothing (one
failing job never stops the worker). Then:  submit.py --manifest <file>   and   worker.py up   (that spends CU).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import zimage_colab as z

SEED_BASE = 9_100_000
SIZE = (720, 1280)
HIRES = (864, 1536)
GPUS = ("L4", "A100")


def _promo():
    """(concept_colab, concept_stills, activity_promo): imported late so `report` works without the training folder."""
    import activity_promo as ap
    import concept_colab as cc
    import concept_stills as cs

    return cc, cs, ap


def _entry(spec: dict, tag: str, *, width=None, height=None, batch_size=1, seed=None, steps=None) -> dict:
    keep = ("prompt", "negative", "cfg", "solo", "allow_text")
    out = {k: spec[k] for k in keep if k in spec}
    out.update(
        width=width or spec["width"],
        height=height or spec["height"],
        steps=steps or spec["steps"],
        seed=seed if seed is not None else spec["seed"],
        batch_size=batch_size,
        tag=tag,
    )
    return out


def build_matrix(*, looks: bool = False) -> list[dict]:
    cc, cs, ap = _promo()
    bible = cs.load_bible(ap.PROJECT_ROOT)
    stories = {s.id: s for s in cs.load_stories(ap.PROJECT_ROOT)}
    cafe = stories["promo_05_cat_cafe"]
    spec = lambda story, n, letter="A", style=None: cc.job_spec(bible, story, n, letter, style)  # noqa: E731
    jobs: list[dict] = []
    for n in range(1, 7):
        jobs.append(_entry(spec(cafe, n), "single"))
    base = spec(cafe, 1)
    for i in range(4):
        jobs.append(_entry(base, "repeat", seed=SEED_BASE + i))
    batch_prompt = spec(cafe, 2)
    for size in (2, 4, 8):
        jobs.append(_entry(batch_prompt, f"batch-{size}", batch_size=size, seed=SEED_BASE + 100))
    for size in (1, 2, 4):
        jobs.append(
            _entry(
                batch_prompt, f"hires-{size}", width=HIRES[0], height=HIRES[1], batch_size=size, seed=SEED_BASE + 200
            )
        )
    if looks:
        shots = [
            ("day-close", stories["promo_05_cat_cafe"], 2),
            ("night", stories["promo_07_night_view"], 4),
            ("medium", stories["promo_08_travel"], 1),
        ]
        for label, story, n in shots:
            seed = cs.candidate_seed(story, n, "A")
            for style in ("bible", "documentary", "phone"):
                s = spec(story, n, "A", style)
                for w, h in (SIZE, HIRES):
                    jobs.append(_entry(s, f"look-{label}-{style}-{w}x{h}", width=w, height=h, seed=seed))
        night = spec(stories["promo_07_night_view"], 4, "A", "documentary")
        for steps in (8, 12):
            jobs.append(
                _entry(
                    night,
                    f"look-steps-{steps}",
                    steps=steps,
                    seed=cs.candidate_seed(stories["promo_07_night_view"], 4, "A"),
                )
            )
    return jobs


YOUNG = "a young adult Taiwanese woman in her early twenties with a youthful fresh face and smooth clear skin"
FEATURES = (
    "a soft round-oval face with a gently rounded jawline, large dark brown eyes with double eyelids, a small straight nose, "
    "full natural lips, a tiny mole under her left eye, long layered dark brown hair with soft curls and side-swept bangs"
)
VARIANTS = (
    "v0",
    "v1",
    "v2",
    "v3",
)  # baseline / young wording / young + fixed features / the same, one seed for all shots
R2_LOOKS = ("bible", "documentary")
R2_SEED = SEED_BASE + 500


def describe(variant: str, outfit: str) -> str:
    if variant == "v0":
        return f"an adult Taiwanese woman aged 23, long layered hair with soft curls, dark brown eyes, oval face, athletic build, {outfit}, small pearl drop earrings"
    if variant == "v1":
        return f"{YOUNG}, long layered dark brown hair with soft curls, dark brown eyes, oval face, athletic build, {outfit}, small pearl drop earrings"
    return f"{YOUNG}, {FEATURES}, {outfit}, small pearl drop earrings"


def build_round2() -> list[dict]:
    """Four ways of describing the same woman across the six cat-cafe shots, in two looks, plus one job of four."""
    import copy
    import re

    cc, cs, ap = _promo()
    bible = cs.load_bible(ap.PROJECT_ROOT)
    cafe = next(s for s in cs.load_stories(ap.PROJECT_ROOT) if s.id == "promo_05_cat_cafe")
    outfit = re.search(r"athletic build, (.*?), small pearl", cafe.data["characters"]["ruoxi"]["description"]).group(1)
    jobs: list[dict] = []
    for variant in VARIANTS:
        story = cs.Story(copy.deepcopy(cafe.data), cafe.dir)
        story.data["characters"]["ruoxi"]["description"] = describe(variant, outfit)
        for look in R2_LOOKS:
            for n in range(1, 7):
                spec = cc.job_spec(bible, story, n, "A", look)
                seed = R2_SEED if variant == "v3" else None
                jobs.append(_entry(spec, f"r2-{variant}-{look}-s{n}", seed=seed))
    story = cs.Story(copy.deepcopy(cafe.data), cafe.dir)
    story.data["characters"]["ruoxi"]["description"] = describe("v3", outfit)
    jobs.append(
        _entry(cc.job_spec(bible, story, 2, "A", "documentary"), "r2-batch4-v3-documentary", batch_size=4, seed=R2_SEED)
    )
    return jobs


def cmd_plan(args) -> int:
    jobs = build_round2() if args.round2 else build_matrix(looks=args.looks)
    name = args.name or f"bench-{args.gpu}{'-round2' if args.round2 else ''}{'-looks' if args.looks else ''}"
    config = z.load_config()
    directory = z.output_root(config) / "bench"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.manifest.json"
    path.write_text(json.dumps({"batch": name, "jobs": jobs}, ensure_ascii=False, indent=1), encoding="utf-8")
    pictures = sum(j["batch_size"] for j in jobs)
    print(json.dumps({"ok": True, "manifest": str(path), "jobs": len(jobs), "pictures": pictures, "gpu": args.gpu}))
    print(
        f"next (submit spends nothing, worker.py up spends CU):\n"
        f"  set ZIMG_GPU={args.gpu}\n  python submit.py --manifest {path}\n  python worker.py up"
    )
    return 0


# --- report ----------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def group_of(tag: str | None) -> str:
    tag = tag or ""
    for head in ("single", "repeat", "batch", "hires", "look", "r2"):
        if tag == head or tag.startswith(head + "-"):
            return head
    return "other"


def per_picture(job: dict) -> float | None:
    seconds = job.get("generation_seconds")
    if not seconds:
        return None
    return round(seconds / max(1, int(job.get("batch_size") or 1)), 2)


def summarize(jobs: list[dict], sessions: list[dict], *, gpu: str | None = None) -> dict:
    """Rows per job, a per-(group, size, batch) cost table, and for each session what a picture costs there."""
    rows = []
    for j in jobs:
        if not j.get("tag"):  # only jobs this tool planned carry a tag
            continue
        if gpu and gpu.lower() not in str(j.get("gpu") or "").lower():
            continue
        batch = int(j.get("batch_size") or 1)
        rows.append(
            {
                "job_id": j["job_id"],
                "tag": j.get("tag"),
                "group": group_of(j.get("tag")),
                "gpu": j.get("gpu"),
                "session": j.get("session"),
                "status": j.get("status"),
                "error": j.get("error_code"),
                "width": j.get("width"),
                "height": j.get("height"),
                "steps": j.get("steps"),
                "batch_size": batch,
                "seconds": j.get("generation_seconds"),
                "per_picture": per_picture(j),
                "vram": j.get("vram"),
                "warmup": bool(j.get("session_reused") is False),
            }
        )
    ok = [r for r in rows if r["status"] == "COMPLETED"]
    by_session: dict[str, dict] = {}
    for s in sessions:
        name = s.get("session")
        if not name:
            continue
        mine = [r for r in ok if r["session"] == name]
        if not mine:
            continue
        rate = s.get("cu_rate_per_hour")
        used = s.get("cu_used_measured")
        if used is None:
            used = s.get("cu_estimated")
        pictures = sum(r["batch_size"] for r in mine)
        generation = sum(r["seconds"] or 0 for r in mine)
        marginal = round((rate or 0) * generation / 3600, 4)
        by_session[name] = {
            "gpu": s.get("gpu"),
            "pictures": pictures,
            "jobs": len(mine),
            "session_seconds": s.get("session_seconds"),
            "generation_seconds": round(generation, 1),
            "cu_used": used,
            "cu_rate_per_hour": rate,
            "cu_marginal_per_picture": round(marginal / pictures, 5) if pictures else None,
            "cu_fixed": round(used - marginal, 4) if used is not None else None,
            "cu_per_picture_all_in": round(used / pictures, 4) if used is not None and pictures else None,
        }
    table: dict[tuple, list[float]] = {}
    for r in ok:
        if r["group"] in ("look", "r2", "other") or r["warmup"]:
            continue
        key = (r["gpu"], r["group"], f"{r['width']}x{r['height']}", r["batch_size"])
        table.setdefault(key, []).append(r["per_picture"])
    cells = [
        {
            "gpu": k[0],
            "group": k[1],
            "size": k[2],
            "batch_size": k[3],
            "n": len(v),
            "per_picture_seconds": round(sum(v) / len(v), 2),
        }
        for k, v in sorted(table.items(), key=lambda kv: (str(kv[0][0]), kv[0][2], kv[0][3]))
    ]
    failed = [r for r in rows if r["status"] != "COMPLETED"]
    return {"jobs": rows, "cells": cells, "sessions": by_session, "failed": failed}


def break_even(a: dict, b: dict) -> float | None:
    """How many pictures in one session before session `b` (higher fixed cost, cheaper pictures) beats `a`."""
    try:
        fixed_a, fixed_b = a["cu_fixed"], b["cu_fixed"]
        pa, pb = a["cu_marginal_per_picture"], b["cu_marginal_per_picture"]
    except KeyError:
        return None
    if None in (fixed_a, fixed_b, pa, pb) or pa <= pb:
        return None
    return round((fixed_b - fixed_a) / (pa - pb), 1)


def render_html(summary: dict, root: Path) -> str:
    e = html.escape
    out = [
        "<!doctype html><meta charset=utf-8><title>Z-Image bench</title>",
        "<style>body{font:14px sans-serif;margin:20px}table{border-collapse:collapse;margin:8px 0}"
        "td,th{border:1px solid #ccc;padding:3px 8px}th{background:#eee}.bad{background:#fdd}"
        ".grid{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;width:220px}img{width:220px}</style>",
    ]
    out.append(
        "<h2>Sessions</h2><table><tr><th>session</th><th>GPU</th><th>pictures</th><th>seconds</th><th>CU</th>"
        "<th>fixed CU</th><th>CU/picture (marginal)</th><th>CU/picture (all in)</th></tr>"
    )
    for name, s in summary["sessions"].items():
        out.append(
            "<tr>"
            + "".join(
                f"<td>{e(str(v))}</td>"
                for v in (
                    name,
                    s["gpu"],
                    s["pictures"],
                    s["session_seconds"],
                    s["cu_used"],
                    s["cu_fixed"],
                    s["cu_marginal_per_picture"],
                    s["cu_per_picture_all_in"],
                )
            )
            + "</tr>"
        )
    out.append(
        "</table><h2>Seconds per picture</h2><table><tr><th>GPU</th><th>group</th><th>size</th><th>batch</th>"
        "<th>jobs</th><th>s / picture</th></tr>"
    )
    for c in summary["cells"]:
        out.append(
            "<tr>"
            + "".join(
                f"<td>{e(str(c[k]))}</td>" for k in ("gpu", "group", "size", "batch_size", "n", "per_picture_seconds")
            )
            + "</tr>"
        )
    out.append("</table>")
    if summary["failed"]:
        out.append(
            "<h2>Jobs that did not finish</h2><table><tr><th>tag</th><th>size</th><th>batch</th><th>status</th><th>why</th></tr>"
        )
        for r in summary["failed"]:
            out.append(
                f'<tr class="bad"><td>{e(str(r["tag"]))}</td><td>{r["width"]}x{r["height"]}</td><td>{r["batch_size"]}</td>'
                f"<td>{e(str(r['status']))}</td><td>{e(str(r['error']))}</td></tr>"
            )
        out.append("</table>")
    looks = [r for r in summary["jobs"] if r["group"] == "look" and r["status"] == "COMPLETED"]
    if looks:
        out.append("<h2>Looks</h2><div class=grid>")
        for r in sorted(looks, key=lambda r: r["tag"]):
            png = root / r["job_id"] / "result.png"
            out.append(
                f"<figure><img src='{e(png.as_uri())}'><figcaption>{e(r['tag'])}<br>{r['seconds']} s, VRAM {r['vram']}</figcaption></figure>"
            )
        out.append("</div>")
    return "\n".join(out)


def identity_stats(embeddings: list) -> dict:
    """Mean and lowest cosine similarity between every pair of (already normalised) face embeddings."""
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            pairs.append(float(sum(a * b for a, b in zip(embeddings[i], embeddings[j]))))
    if not pairs:
        return {"pairs": 0, "mean": None, "min": None}
    return {"pairs": len(pairs), "mean": round(sum(pairs) / len(pairs), 3), "min": round(min(pairs), 3)}


def cmd_faces(args) -> int:
    """Run with ComfyUI's python (it has insightface); reads the finished jobs, spends nothing."""
    from insightface.app import FaceAnalysis

    import face_similarity as fs

    config = z.load_config()
    root = z.output_root(config)
    app = FaceAnalysis(
        name="buffalo_l",
        root=fs.INSIGHTFACE_ROOT,
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition", "genderage"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    groups: dict[str, list] = {}
    for j in read_jsonl(root / "zimage_jobs.jsonl"):
        tag = j.get("tag") or ""
        if not tag.startswith(args.prefix) or j.get("status") != "COMPLETED":
            continue
        for png in sorted((root / j["job_id"]).glob("result*.png")):
            faces = app.get(fs._read_image_bgr(str(png)))
            if not faces:
                groups.setdefault(tag.rsplit("-s", 1)[0], []).append(
                    {"tag": tag, "png": png.name, "age": None, "emb": None}
                )
                continue
            f = max(faces, key=lambda x: x.det_score)
            groups.setdefault(tag.rsplit("-s", 1)[0], []).append(
                {"tag": tag, "png": png.name, "age": int(f.age), "emb": [float(v) for v in f.normed_embedding]}
            )
    table = []
    for name, items in sorted(groups.items()):
        ages = [i["age"] for i in items if i["age"] is not None]
        stats = identity_stats([i["emb"] for i in items if i["emb"] is not None])
        table.append(
            {
                "group": name,
                "pictures": len(items),
                "faces_found": len(ages),
                "age_mean": round(sum(ages) / len(ages), 1) if ages else None,
                "age_min": min(ages, default=None),
                "age_max": max(ages, default=None),
                "identity_mean": stats["mean"],
                "identity_min": stats["min"],
            }
        )
    print(
        json.dumps(
            {"groups": table, "note": "same person usually scores 0.5 or more; 0.3-0.4 is borderline"},
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


def cmd_report(args) -> int:
    config = z.load_config()
    root = z.output_root(config)
    summary = summarize(
        read_jsonl(root / "zimage_jobs.jsonl"), read_jsonl(root / "zimage_sessions.jsonl"), gpu=args.gpu
    )
    names = list(summary["sessions"])
    if len(names) >= 2:
        a, b = summary["sessions"][names[0]], summary["sessions"][names[-1]]
        summary["break_even_pictures"] = {"from": a["gpu"], "to": b["gpu"], "pictures": break_even(a, b)}
    if args.html:
        Path(args.html).write_text(render_html(summary, root), encoding="utf-8")
        summary["html"] = args.html
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Z-Image on Colab: batch size and GPU benchmark.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan", help="write the benchmark manifest (spends nothing)")
    a.add_argument("--gpu", choices=GPUS, required=True, help="only used to name the manifest; the GPU is ZIMG_GPU")
    a.add_argument("--looks", action="store_true", help="add the realism comparison (21 more pictures)")
    a.add_argument("--round2", action="store_true", help="the age / identity test instead of the speed matrix")
    a.add_argument("--name")
    f = sub.add_parser(
        "faces", help="age estimate and identity similarity of the round-2 pictures (CPU, needs insightface)"
    )
    f.add_argument("--prefix", default="r2-")
    r = sub.add_parser("report", help="tables from the finished jobs")
    r.add_argument("--gpu", help="only jobs run on a GPU whose name contains this")
    r.add_argument("--html")
    args = p.parse_args(argv)
    try:
        return {"plan": cmd_plan, "faces": cmd_faces}.get(args.cmd, cmd_report)(args)
    except (z.UsageError, z.ZImageError, OSError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
