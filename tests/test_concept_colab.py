"""training/concept_colab.py: the bridge from the concept stills to the Z-Image Colab worker, offline.

Pinned: the prompt and negative the Colab skill will assemble are exactly the ones the local path assembles (so the
two backends differ only by weights), the map from seed back to scene, and the import writing the same records."""

import json
import os
import shutil

import concept_colab as cc
import concept_stills as cs
import pytest

import generate_character as gc


@pytest.fixture
def work(tmp_path):
    root = tmp_path / "ai_short_drama"
    shutil.copytree(
        cs.PROJECT_ROOT,
        root,
        ignore=shutil.ignore_patterns(
            "*.png", "*.mp4", "videos", "images", "colab", "clips.json", "candidate_index.json", "*.html"
        ),
    )
    return cs.load_bible(str(root)), cs.load_stories(str(root)), tmp_path


def skill_assembly(spec):
    """What .claude/skills/z-image-colab/scripts/zimage_colab.build_job does with a manifest entry."""
    return gc._build_prompt_and_negative(
        spec["prompt"],
        spec["negative"],
        "safe",
        None,
        "",
        "",
        "z_image_turbo",
        solo=spec["solo"],
        allow_text=spec["allow_text"],
    )


def test_the_colab_skill_assembles_the_same_prompt_and_negative_as_the_local_path(work):
    bible, stories, _ = work
    for story in stories:
        for scene in story.data["scenes"]:
            for letter in "AB":
                local = cs.build_request(bible, story, scene["scene"], letter)
                prompt, negative = skill_assembly(cc.job_spec(bible, story, scene["scene"], letter))
                assert prompt == local["prompt"], (story.id, scene["scene"], letter)
                assert sorted(negative.split(", ")) == sorted(local["negative_prompt"].split(", "))


def test_the_spec_carries_the_same_seed_size_and_flags_and_never_the_assembled_negative(work):
    bible, stories, _ = work
    story = stories[3]
    spec = cc.job_spec(bible, story, 7, "A")
    local = cs.build_request(bible, story, 7, "A")
    assert (spec["seed"], spec["width"], spec["height"], spec["steps"]) == (
        local["seed"],
        local["width"],
        local["height"],
        local["steps"],
    )
    assert spec["cfg"] >= cs.client.SAFETY_MIN_CFG
    assert "nsfw" not in spec["negative"] and "child" not in spec["negative"]  # the skill adds the safety terms
    empty = cc.job_spec(bible, stories[0], 7, "A")
    assert "people, person, crowd" in empty["negative"] and empty["allow_text"] is True


def test_export_lists_only_what_would_be_rendered_and_maps_each_seed_back(work):
    bible, stories, tmp = work
    story = stories[4]
    out = cc.export(bible, story, batch="b1")
    manifest = json.load(open(out["manifest"], encoding="utf-8"))
    assert manifest["batch"] == "b1" and len(manifest["jobs"]) == out["images"] == 7
    assert [j["seed"] for j in manifest["jobs"]] == [cs.candidate_seed(story, n, "A") for n in range(1, 8)]
    mapping = json.load(open(os.path.join(story.dir, "colab", "b1.map.json"), encoding="utf-8"))
    assert [(m["scene"], m["letter"]) for m in mapping["images"]] == [(n, "A") for n in range(1, 8)]
    with pytest.raises(gc.UsageError, match="REGENERATE"):
        cc.export(bible, story, letters=("C",))


def make_job(tmp, batch, seed, status="COMPLETED", name=None):
    png = tmp / f"{name or seed}.png"
    png.write_bytes(b"\x89PNG colab " + str(seed).encode())
    return {
        "job_id": f"zimg-{seed}",
        "batch": batch,
        "seed": seed,
        "status": status,
        "output": str(png),
        "prompt": "the prompt",
        "negative_prompt": "the negative",
        "cfg": 2.0,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "shift": 3.0,
        "gpu": "NVIDIA L4",
        "vram_peak_mib": 15000,
        "generation_seconds": 21.5,
        "completed_at": "2026-10-06T20:00:00+08:00",
        "graph": {"3": {"inputs": {"seed": seed}}},
    }


def test_import_files_each_finished_job_where_the_local_path_would_and_hands_the_scenes_to_review(work):
    bible, stories, tmp = work
    story = stories[4]
    out = cc.export(bible, story, batch="b2", scene_numbers={1, 2, 3})
    jobs_dir = tmp / "jobs"
    jobs_dir.mkdir()
    seeds = [cs.candidate_seed(story, n, "A") for n in (1, 2, 3)]
    for seed, status in zip(seeds, ("COMPLETED", "FAILED", "COMPLETED")):
        job = make_job(tmp, "b2", seed, status)
        (jobs_dir / f"{job['job_id']}.json").write_text(json.dumps(job), encoding="utf-8")
    other = make_job(tmp, "another-batch", seeds[1])
    (jobs_dir / "other.json").write_text(json.dumps(other), encoding="utf-8")
    report = cc.import_batch(bible, story, out["batch"], jobs_dir=str(jobs_dir))
    assert report["imported"] == ["scene_01 A", "scene_03 A"] and report["failed"] == ["scene_02 A (FAILED)"]
    paths = cs.candidate_paths(story, 1, "A")
    assert open(paths["png"], "rb").read().startswith(b"\x89PNG colab")
    meta = json.load(open(paths["meta"], encoding="utf-8"))
    for key in (
        "story_id",
        "scene_id",
        "candidate_id",
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "steps",
    ):
        assert key in meta
    assert meta["backend"] == "colab" and meta["model"] == cc.COLAB_MODEL and meta["gpu"] == "NVIDIA L4"
    assert meta["colab_job_id"] == f"zimg-{seeds[0]}" and meta["seed"] == seeds[0]
    assert json.load(open(paths["workflow"], encoding="utf-8"))["3"]["inputs"]["seed"] == seeds[0]
    assert [cs.load_review(story, n)["status"] for n in (1, 2, 3)] == [
        cs.HUMAN_REVIEW,
        cs.NO_CANDIDATES,
        cs.HUMAN_REVIEW,
    ]
    again = cc.import_batch(bible, story, out["batch"], jobs_dir=str(jobs_dir))
    assert again["imported"] == [] and again["skipped"] == ["scene_01 A", "scene_03 A"]  # never overwrites


def test_import_refuses_a_batch_from_another_story_and_reports_jobs_that_are_missing(work):
    bible, stories, tmp = work
    out = cc.export(bible, stories[4], batch="b3", scene_numbers={1})
    with pytest.raises(gc.UsageError, match="不是"):
        cc.import_batch(bible, stories[3], out["batch"], directory=os.path.join(stories[4].dir, "colab"))
    empty = tmp / "nojobs"
    empty.mkdir()
    assert cc.import_batch(bible, stories[4], "b3", jobs_dir=str(empty))["missing"] == ["scene_01 A"]
