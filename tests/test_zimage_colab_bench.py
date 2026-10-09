"""bench.py: the benchmark manifest is a valid queue of real promo prompts, and the report arithmetic is right."""

import json
import sys
from pathlib import Path

import pytest
from zimage_colab_fakes import make_config, z

SCRIPTS = Path(z.__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import bench  # noqa: E402
import submit  # noqa: E402


def test_the_matrix_is_small_to_large_and_every_job_is_a_valid_job(tmp_path):
    config = make_config(tmp_path)
    jobs = bench.build_matrix(looks=True)
    assert len(jobs) == 36 and sum(j["batch_size"] for j in jobs) == 51
    groups = [bench.group_of(j["tag"]) for j in jobs]
    assert groups[:6] == ["single"] * 6 and groups[6:10] == ["repeat"] * 4
    assert [j["batch_size"] for j in jobs if bench.group_of(j["tag"]) == "batch"] == [2, 4, 8]
    for j in jobs:
        built = z.build_job(config, **{k: v for k, v in j.items()})  # same checks as submit.py
        assert built["tag"] == j["tag"] and built["batch_size"] == j["batch_size"]
        assert all(t in built["negative_prompt"] for t in ("child", "minor", "underage"))


def test_the_looks_only_change_the_style_text():
    jobs = {j["tag"]: j for j in bench.build_matrix(looks=True)}
    plain = jobs["look-night-bible-720x1280"]["prompt"]
    doc = jobs["look-night-documentary-720x1280"]["prompt"]
    assert plain != doc and "film grain" in doc and "film grain" not in plain
    assert jobs["look-steps-8"]["steps"] == 8 and jobs["look-steps-12"]["steps"] == 12


def test_plan_writes_a_manifest_that_submit_accepts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZIMG_OUTPUT_DIR", str(tmp_path / "out"))
    assert bench.main(["plan", "--gpu", "L4", "--looks"]) == 0
    manifest = tmp_path / "out" / "bench" / "bench-L4-looks.manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["batch"] == "bench-L4-looks"
    capsys.readouterr()
    assert submit.main(["--manifest", str(manifest), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"] is True


def job(tag, session, gpu, seconds, batch=1, status="COMPLETED", reused=True, w=720, h=1280, vram=20000, error=None):
    return {
        "job_id": f"{tag}-{seconds}-{session}",
        "tag": tag,
        "session": session,
        "gpu": gpu,
        "status": status,
        "generation_seconds": seconds,
        "batch_size": batch,
        "width": w,
        "height": h,
        "steps": 8,
        "vram": vram,
        "session_reused": reused,
        "error_code": error,
    }


def test_the_report_prices_a_picture_marginally_and_all_in():
    jobs = [
        job("single", "s1", "NVIDIA L4", 60, reused=False),
        job("single", "s1", "NVIDIA L4", 18),
        job("batch-4", "s1", "NVIDIA L4", 60, batch=4),
        job("batch-8", "s1", "NVIDIA L4", 0, batch=8, status="FAILED", error="COMFY_EXECUTION_ERROR"),
    ]
    sessions = [
        {"session": "s1", "gpu": "NVIDIA L4", "cu_rate_per_hour": 1.8, "cu_used_measured": 0.4, "session_seconds": 900}
    ]
    s = bench.summarize(jobs, sessions)
    one = s["sessions"]["s1"]
    assert one["pictures"] == 6 and one["generation_seconds"] == 138
    assert one["cu_marginal_per_picture"] == pytest.approx(1.8 * 138 / 3600 / 6, abs=1e-5)
    assert one["cu_fixed"] == pytest.approx(0.4 - 1.8 * 138 / 3600, abs=1e-4)
    assert one["cu_per_picture_all_in"] == pytest.approx(0.4 / 6, abs=1e-4)
    cells = {(c["group"], c["batch_size"]): c["per_picture_seconds"] for c in s["cells"]}
    assert cells[("single", 1)] == 18  # the first job of a session is warm-up and not counted
    assert cells[("batch", 4)] == 15
    assert [f["tag"] for f in s["failed"]] == ["batch-8"] and s["failed"][0]["error"] == "COMFY_EXECUTION_ERROR"


def test_break_even_between_two_sessions():
    cheap_fixed = {"cu_fixed": 0.20, "cu_marginal_per_picture": 0.010}
    fast = {"cu_fixed": 0.60, "cu_marginal_per_picture": 0.004}
    assert bench.break_even(cheap_fixed, fast) == pytest.approx(66.7, abs=0.1)
    assert bench.break_even(fast, cheap_fixed) is None  # the second one is not cheaper per picture


def test_round_two_writes_the_same_six_shots_four_ways_in_two_looks(tmp_path):
    config = make_config(tmp_path)
    jobs = bench.build_round2()
    assert len(jobs) == 49 and sum(j["batch_size"] for j in jobs) == 52
    by_tag = {j["tag"]: j for j in jobs}
    assert "aged 23" in by_tag["r2-v0-bible-s1"]["prompt"]
    assert (
        "early twenties" in by_tag["r2-v1-bible-s1"]["prompt"] and "tiny mole" not in by_tag["r2-v1-bible-s1"]["prompt"]
    )
    assert "tiny mole" in by_tag["r2-v2-documentary-s3"]["prompt"]
    assert len({by_tag[f"r2-v2-bible-s{n}"]["seed"] for n in range(1, 7)}) == 6  # v2: a seed per shot
    assert {by_tag[f"r2-v3-bible-s{n}"]["seed"] for n in range(1, 7)} == {bench.R2_SEED}  # v3: one seed for all
    assert by_tag["r2-batch4-v3-documentary"]["batch_size"] == 4
    for j in jobs:  # every one is a valid job with the age-safety negatives, and no minor-sounding wording
        built = z.build_job(config, **j)
        assert all(t in built["negative_prompt"] for t in ("child", "minor", "underage"))
        assert not any(w in j["prompt"].lower() for w in ("girl", "teen", "child", "kid", "minor"))


def test_identity_stats_are_cosine_similarities_over_all_pairs():
    same, other = [1.0, 0.0], [0.0, 1.0]
    assert bench.identity_stats([same, same, same]) == {"pairs": 3, "mean": 1.0, "min": 1.0}
    assert bench.identity_stats([same, other]) == {"pairs": 1, "mean": 0.0, "min": 0.0}
    assert bench.identity_stats([same]) == {"pairs": 0, "mean": None, "min": None}
