"""Render the concept stills on the Colab Z-Image worker instead of the local card.

    concept_colab.py export --story 5            write a manifest for .claude/skills/z-image-colab (spends nothing)
    ... submit.py --manifest <manifest>   then   worker.py up   (the skill's own steps; `worker.py up` spends CU)
    concept_colab.py import --story 5 --batch <batch>    file the finished PNGs into the story's images/ folders

The local card heats to 83-84 C and the owner prefers its lifespan to speed, so the batches can run on a Colab
GPU. The prompts, seeds, sizes and safety terms are the same as the local path: `export` hands the skill the
scene text plus the visual-bible style, and the skill's own assembly (generate_character._build_prompt_and_negative)
adds the safety negatives exactly as concept_stills does - a test pins that the two give the same strings.

What is NOT the same: the Colab worker runs the official bf16 Z-Image weights and the local ComfyUI the quantised
ones, so one seed gives two different pictures. Each candidate's metadata records which backend made it, and a
scene's candidates should not be mixed up by seed across backends.
"""

import argparse
import glob
import json
import os
import shutil

import concept_stills as cs
import generate_character as gc

ZIMAGE_JOBS_DIR = os.path.join(cs.REPO_ROOT, "output", "zimage", "jobs")
BACKEND = "colab"
COLAB_MODEL = "z_image_turbo (official bf16, Colab)"


def job_spec(bible, story, scene_number, letter):
    """One manifest entry. The skill adds the safety negatives itself, so this passes the scene text plus style and
    only the extra negative terms - never the already-assembled negative (it would be assembled twice)."""
    scene = story.scene(scene_number)
    composition = cs.composition_for(bible, scene, letter)
    text = cs.scene_prompt_text(story, scene, composition)
    extra = cs.style_negative(bible)
    if scene.get("negative_extra"):
        extra = f"{extra}, {scene['negative_extra']}"
    return {
        "prompt": f"{text}, {bible['style_positive']}",
        "negative": extra,
        "width": bible["canvas"]["width"],
        "height": bible["canvas"]["height"],
        "steps": bible["sampler"]["steps"],
        "cfg": max(bible["sampler"]["cfg"], cs.client.SAFETY_MIN_CFG),
        "seed": cs.candidate_seed(story, scene_number, letter),
        "solo": scene["solo"],
        "allow_text": bool(scene["on_screen_text"]),
    }


def export(bible, story, *, letters=cs.FIRST_ROUND[:1], scene_numbers=None, batch=None, directory=None):
    """Write the skill's manifest and a map from each seed back to (scene, candidate). Only images that would be
    rendered locally are included (finished ones, and approved scenes, are skipped)."""
    todo = cs.pending([story], letters, scene_numbers=scene_numbers)
    if not todo:
        raise gc.UsageError("沒有要生的圖：這些候選都已經存在，或 Scene 已核准")
    import datetime

    batch = batch or f"{story.id}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    directory = directory or os.path.join(story.dir, "colab")
    entries = [(n, letter, job_spec(bible, story, n, letter)) for _, n, letter in todo]
    manifest = os.path.join(directory, f"{batch}.manifest.json")
    cs._write_json(manifest, {"batch": batch, "jobs": [spec for _, _, spec in entries]})
    cs._write_json(os.path.join(directory, f"{batch}.map.json"), {
        "batch": batch, "story_id": story.id,
        "images": [{"scene": n, "letter": letter, "seed": spec["seed"]} for n, letter, spec in entries]})
    return {"batch": batch, "manifest": manifest, "images": len(entries)}


def find_jobs(batch, jobs_dir=ZIMAGE_JOBS_DIR):
    jobs = []
    for path in glob.glob(os.path.join(jobs_dir, "*.json")):
        try:
            job = cs._read_json(path)
        except (OSError, ValueError):
            continue
        if job.get("batch") == batch:
            jobs.append(job)
    return jobs


def import_batch(bible, story, batch, *, jobs_dir=ZIMAGE_JOBS_DIR, directory=None, force=False):
    """Copy every COMPLETED job of the batch into images/scene_NN/candidate_NN.png with the metadata (and the filled
    graph) the local path writes, then hand the scenes to review. Returns what happened per image."""
    directory = directory or os.path.join(story.dir, "colab")
    mapping = cs._read_json(os.path.join(directory, f"{batch}.map.json"))
    if mapping["story_id"] != story.id:
        raise gc.UsageError(f"這個批次是 {mapping['story_id']} 的，不是 {story.id}")
    by_seed = {m["seed"]: m for m in mapping["images"]}
    jobs = {j["seed"]: j for j in find_jobs(batch, jobs_dir)}
    report = {"imported": [], "skipped": [], "missing": [], "failed": []}
    for seed, m in by_seed.items():
        n, letter = m["scene"], m["letter"]
        paths = cs.candidate_paths(story, n, letter)
        label = f"scene_{n:02d} {letter}"
        job = jobs.get(seed)
        if job is None:
            report["missing"].append(label)
            continue
        if job["status"] != "COMPLETED" or not job.get("output") or not os.path.isfile(job["output"]):
            report["failed"].append(f"{label} ({job['status']})")
            continue
        if os.path.isfile(paths["png"]) and not force:
            report["skipped"].append(label)
            continue
        os.makedirs(os.path.dirname(paths["png"]), exist_ok=True)
        shutil.copy2(job["output"], paths["png"])
        req = cs.build_request(bible, story, n, letter)
        meta = {k: req[k] for k in ("story_id", "scene_id", "candidate_id", "seed", "width", "height", "steps",
                                    "workflow")}
        meta.update({
            "prompt": job["prompt"], "negative_prompt": job["negative_prompt"], "model": COLAB_MODEL,
            "backend": BACKEND, "cfg": job["cfg"], "composition": req["composition"], "candidate_letter": letter,
            "sampler": job.get("sampler"), "scheduler": job.get("scheduler"), "shift": job.get("shift"),
            "workflow_graph": os.path.basename(paths["workflow"]), "image_sha256": cs._sha256(paths["png"]),
            "generation_seconds": job.get("generation_seconds"), "gpu": job.get("gpu"),
            "vram_peak_mib": job.get("vram_peak_mib"), "colab_job_id": job["job_id"], "colab_batch": batch,
            "created_at": job.get("completed_at") or cs._now(),
        })
        cs._write_json(paths["meta"], meta)
        if job.get("graph"):
            cs._write_json(paths["workflow"], job["graph"])
        report["imported"].append(label)
    cs.mark_ready_scenes([story])
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Concept stills on the Colab Z-Image worker")
    p.add_argument("--root", default=cs.PROJECT_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the manifest for the z-image-colab skill (spends nothing)")
    e.add_argument("--story", required=True)
    e.add_argument("--scene")
    e.add_argument("--candidates", default="A")
    e.add_argument("--batch")
    i = sub.add_parser("import", help="file the finished jobs of a batch into the story's images/ folders")
    i.add_argument("--story", required=True)
    i.add_argument("--batch", required=True)
    i.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    bible = cs.load_bible(args.root)
    story = cs.select_stories(cs.load_stories(args.root), args.story)[0]
    cs.validate_story(story)
    if args.cmd == "export":
        letters = cs._parse_letters(args.candidates)
        out = export(bible, story, letters=letters, scene_numbers=cs._parse_scenes(args.scene), batch=args.batch)
        print(json.dumps(out, ensure_ascii=False))
        print("下一步（submit 不花錢，worker.py up 會花 Colab CU）：\n"
              f"  ComfyUI\\.venv\\Scripts\\python.exe .claude\\skills\\z-image-colab\\scripts\\submit.py --manifest {out['manifest']}\n"
              "  ComfyUI\\.venv\\Scripts\\python.exe .claude\\skills\\z-image-colab\\scripts\\worker.py up")
    else:
        report = import_batch(bible, story, args.batch, force=args.force)
        cs.build_index(cs.load_stories(args.root), args.root)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
