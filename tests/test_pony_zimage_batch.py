"""pony_zimage_batch: several Pony x Z-Image pictures, each with its own random seed, run one after another.
Nothing here touches ComfyUI, Ollama or the GPU."""

import json
import os
import random

import pytest

import generate_character as gc
import pony_zimage_batch as pb


class FakeGuard:
    def __init__(self, calls):
        self.calls = calls

    def before_batch(self, *, max_start_temp_c):
        self.calls.append(("before_batch", max_start_temp_c))

    def before_image(self):
        self.calls.append(("before_image",))

    def wait_until_cool(self):
        self.calls.append(("wait_until_cool",))
        return {"waited_seconds": 3, "temp_c": 58}


def batch(tmp_path, jobs, *, stop_after=None, fail_seeds=(), fail_prompts=()):
    calls = []

    def prepare(prompt):
        calls.append(("prepare", prompt))
        if prompt in fail_prompts:
            raise RuntimeError("Qwen 改寫失敗")
        return {"pony": f"tags {prompt}", "zimage": f"sentence {prompt}"}

    def generate(job, prepared, stem):
        calls.append(("generate", job.index, job.seed, stem))
        if job.seed in fail_seeds:
            raise gc.UsageError("boom")
        return str(tmp_path / f"{stem}_pony.png"), str(tmp_path / f"{stem}.png")

    done = []

    def should_stop():
        return stop_after is not None and len(done) >= stop_after

    out = []
    for rec in pb.run_batch(
        jobs,
        prepare=prepare,
        generate=generate,
        guard=FakeGuard(calls),
        start_temp_c=55,
        should_stop=should_stop,
        batch_dir=str(tmp_path),
        meta={"qwen": True},
        log=lambda *_: None,
        clock=lambda: 100.0,
    ):
        done.append(rec)
        out.append(rec)
    return out, calls


def three(rng=None):
    return pb.plan_jobs(["a", "b"], 2, rng=rng or random.Random(1))


def test_prompts_are_one_per_line_and_blank_lines_are_dropped():
    assert pb.parse_prompts("  一隻貓 \n\n  a dog\n") == ["一隻貓", "a dog"]
    with pytest.raises(gc.UsageError):
        pb.parse_prompts("  \n \n")
    with pytest.raises(gc.UsageError, match="最多"):
        pb.parse_prompts("\n".join(f"p{i}" for i in range(pb.MAX_PROMPTS + 1)))


def test_every_job_gets_its_own_seed_in_range_even_for_one_prompt():
    jobs = pb.plan_jobs(["same prompt"], 8, rng=random.Random(7))
    seeds = [j.seed for j in jobs]
    assert len(set(seeds)) == 8 and all(0 <= s <= pb.SEED_MAX for s in seeds)
    assert [j.index for j in jobs] == list(range(1, 9)) and {j.prompt_no for j in jobs} == {1}


def test_seeds_do_not_repeat_even_when_the_generator_does():
    class Stuck:
        def __init__(self):
            self.values = iter([5, 5, 5, 6, 6, 7])

        def randint(self, a, b):
            return next(self.values)

    assert [j.seed for j in pb.plan_jobs(["a"], 3, rng=Stuck())] == [5, 6, 7]


def test_seeds_differ_between_runs_with_the_default_generator():
    first = [j.seed for j in pb.plan_jobs(["a"], 4)]
    second = [j.seed for j in pb.plan_jobs(["a"], 4)]
    assert first != second


def test_limits_are_enforced_with_a_readable_message():
    with pytest.raises(gc.UsageError, match="1～"):
        pb.plan_jobs(["a"], 0)
    with pytest.raises(gc.UsageError, match="最多"):
        pb.plan_jobs([f"p{i}" for i in range(6)], 5)  # 30 > MAX_TOTAL


def test_jobs_are_ordered_by_prompt_then_repeat():
    jobs = three()
    assert [(j.prompt_no, j.prompt) for j in jobs] == [(1, "a"), (1, "a"), (2, "b"), (2, "b")]


def test_each_distinct_prompt_is_prepared_once_and_before_the_first_picture(tmp_path):
    _, calls = batch(tmp_path, three())
    names = [c[0] for c in calls]
    assert names.count("prepare") == 2
    assert max(i for i, n in enumerate(names) if n == "prepare") < names.index("generate")


def test_gpu_guard_runs_once_per_batch_and_before_every_picture(tmp_path):
    _, calls = batch(tmp_path, three())
    names = [c[0] for c in calls]
    assert names.count("before_batch") == 1 and ("before_batch", 55) in calls
    assert names.index("before_batch") < names.index("generate")
    gens = [i for i, n in enumerate(names) if n == "generate"]
    for i in gens:
        assert names[i - 2 : i] == ["before_image", "wait_until_cool"]


def test_one_failing_picture_is_recorded_and_the_rest_still_run(tmp_path):
    jobs = three()
    out, _ = batch(tmp_path, jobs, fail_seeds={jobs[1].seed})
    assert [r["status"] for r in out] == ["ok", "error", "ok", "ok"]
    assert out[1]["error"] == "boom" and out[1]["final"] is None


def test_a_failing_prepare_fails_only_that_prompts_jobs(tmp_path):
    out, calls = batch(tmp_path, three(), fail_prompts={"a"})
    assert [r["status"] for r in out] == ["error", "error", "ok", "ok"]
    assert "Qwen" in out[0]["error"]
    assert [c for c in calls if c[0] == "generate"][0][1] == 3  # the first picture is job 3


def test_nothing_runs_on_the_card_when_every_prompt_fails_to_prepare(tmp_path):
    out, calls = batch(tmp_path, pb.plan_jobs(["a"], 2, rng=random.Random(2)), fail_prompts={"a"})
    assert [r["status"] for r in out] == ["error", "error"]
    assert not [c for c in calls if c[0] in ("before_batch", "generate")]


def test_stop_lets_the_running_picture_finish_and_skips_the_rest(tmp_path):
    out, calls = batch(tmp_path, three(), stop_after=2)
    assert [r["status"] for r in out] == ["ok", "ok", "stopped", "stopped"]
    assert len([c for c in calls if c[0] == "generate"]) == 2


def test_the_manifest_has_every_seed_prompt_and_the_english_that_was_sent(tmp_path):
    jobs = three()
    batch(tmp_path, jobs, stop_after=3)
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["qwen"] is True and data["stopped"] is True
    assert [j["seed"] for j in data["jobs"]] == [j.seed for j in jobs]
    assert data["jobs"][0]["english"] == {"pony": "tags a", "zimage": "sentence a"}
    assert data["jobs"][0]["final"].endswith(f"p1_s{jobs[0].seed}.png") and data["jobs"][0]["seconds"] == 0
    assert data["jobs"][0]["waited_seconds"] == 3
    assert not os.path.exists(tmp_path / "manifest.json.tmp")


def test_each_batch_gets_its_own_folder_and_files_are_named_by_prompt_and_seed(tmp_path):
    first = pb.new_batch_dir(str(tmp_path), now=1_800_000_000)
    second = pb.new_batch_dir(str(tmp_path), now=1_800_000_060)
    assert first != second and os.path.isdir(first) and os.path.basename(first).startswith("ponyz_batch_")
    assert pb.stem_for(pb.Job(3, 2, "b", 4242)) == "p2_s4242"


def test_the_real_generator_gets_the_jobs_own_seed_and_the_zimage_sentence(monkeypatch, tmp_path):
    import comfyui_client as client

    seen = []
    monkeypatch.setattr(gc, "gen_custom", lambda *a, **k: (tmp_path / f"{a[7]}.png").write_bytes(b"x"))
    monkeypatch.setattr(client, "zimage_missing_files", lambda model="z_image_turbo": [])

    def fake_refine(path, prompt, negative, seed, prefix, denoise, **kw):
        seen.append((seed, prompt, negative))
        raw = tmp_path / f"raw{seed}.png"
        raw.write_bytes(b"x")
        return str(raw)

    monkeypatch.setattr(client, "submit_img2img_refine_zimage", fake_refine)
    for job in pb.plan_jobs(["x"], 2, rng=random.Random(3)):
        gc.gen_pony_zimage(
            "tags",
            "",
            "safe",
            None,
            None,
            str(tmp_path),
            job.seed,
            pb.stem_for(job),
            1.0,
            refine_prompt="A woman in her early twenties reads.",
        )
    assert len({s for s, _, _ in seen}) == 2
    for _, prompt, negative in seen:
        assert prompt.startswith("A woman in her early twenties reads.")
        assert gc.AGE_SAFETY_NEGATIVE.split(",")[0].strip() in negative
