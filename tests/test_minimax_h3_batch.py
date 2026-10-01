"""Batch runs of the MiniMax H3 skill: several clips through ONE Colab session, so the ComfyUI install and
the ~40 GB model download (6-7 minutes, ~0.75 CU measured per session) are paid once.

The user's rule for failures: skip the clip and carry on. A clip that leaves the session in an unknown
state (exec timeout, exec failure without a marker) is not retried, and the next clip gets a fresh
session. A session that cannot be opened at all ends every clip left with the same error.
"""

import json

from minimax_h3_fakes import (
    HAPPY_EXEC_LINES,
    FakeProber,
    FakeTransport,
    emit,
    h3,
    import_skill,
    make_config,
    make_image,
    marker,
    raising,
)


def usage_seq(*balances, active_first=0):
    """`colab usage` answers: the first read carries active_first runtimes, the rest a session's rate."""
    answers = iter(balances)

    def handler(args, on_line):
        b = next(answers, balances[-1])
        first = handler.calls == 0
        handler.calls += 1
        rate, active = ("0.00", active_first) if first else ("11.77", 1)
        return f"Current balance: {b:.2f} compute units\nUsage rate: {rate}/hr\nActive assignments: {active}"

    handler.calls = 0
    return handler


def per_call(*handlers):
    """An exec handler that behaves differently on each call (the last one repeats)."""
    state = {"n": 0}

    def handler(args, on_line):
        h = handlers[min(state["n"], len(handlers) - 1)]
        state["n"] += 1
        return h(args, on_line)

    return handler


HAPPY = emit(*HAPPY_EXEC_LINES)
NO_LAST_FRAME = emit(*[line for line in HAPPY_EXEC_LINES if "LAST_FRAME" not in line])


def specs(tmp_path, n=3, **kw):
    return [
        h3.JobSpec(image=make_image(tmp_path, f"img{i}.png"), description="She smiles.", job_id=f"j{i}", **kw)
        for i in range(1, n + 1)
    ]


def batch(tmp_path, job_specs, transport=None, **kw):
    transport = transport or FakeTransport(usage=usage_seq(100.0, 100.0, 99.0, 98.4, 97.8, 97.8))
    summary = h3.run_batch(
        job_specs, make_config(tmp_path), transport=transport, prober=FakeProber(), stream=None, batch_id="b1", **kw
    )
    return summary, transport


def statuses(summary):
    return [(r["job_id"], r["status"], r["error_code"]) for r in summary["records"]]


def test_one_session_serves_every_job(tmp_path):
    summary, t = batch(tmp_path, specs(tmp_path))
    calls = t.subcommands
    assert (calls.count("new"), calls.count("stop"), calls.count("exec")) == (1, 1, 3)
    assert calls.count("upload") == 6 and calls.count("download") == 6  # image + job.json, mp4 + last frame
    assert summary["status"] == h3.COMPLETED and summary["completed"] == 3
    assert [r["session"] for r in summary["records"]] == ["h3-b1"] * 3
    assert [r["session_reused"] for r in summary["records"]] == [False, True, True]
    assert all((tmp_path / "out" / f"img{i}_h3_j{i}.mp4").is_file() for i in (1, 2, 3))
    for r in summary["records"]:
        assert r["status"] == h3.COMPLETED and r["session_status"] == "stopped"


def test_each_job_gets_its_own_share_of_the_balance(tmp_path):
    summary, _ = batch(tmp_path, specs(tmp_path))
    used = [r["cu_used_measured"] for r in summary["records"]]
    assert used == [1.0, 0.6, 0.6]  # the first job carries the install and the model download
    assert summary["cu_balance_before"] == 100.0 and summary["cu_balance_after"] == 97.8
    assert summary["cu_used_measured"] == 2.2
    assert len(summary["sessions"]) == 1 and summary["sessions"][0]["status"] == "stopped"


def test_each_job_times_only_its_own_turn(tmp_path, monkeypatch):
    # Measured live: every record said 1150.7 s (the whole batch) and job 2's PREPARING held its 613 s wait.
    clock = {"t": 1000.0}
    monkeypatch.setattr(h3.time, "monotonic", lambda: clock["t"])

    def exec_(args, on_line):
        clock["t"] += 100.0  # every clip renders for 100 s
        return HAPPY(args, on_line)

    summary, _ = batch(tmp_path, specs(tmp_path), transport=FakeTransport(exec=exec_))
    records = summary["records"]
    assert [r["elapsed_seconds"] for r in records] == [100.0, 100.0, 100.0]
    assert [r["stage_seconds"].get("QUEUED") for r in records] == [None, 100.0, 200.0]
    assert [r["stage_seconds"]["PREPARING"] for r in records] == [0.0, 0.0, 0.0]
    assert summary["elapsed_seconds"] == 300.0


def test_a_failed_job_is_skipped_and_the_batch_goes_on(tmp_path):
    exec_ = per_call(HAPPY, emit(marker("ERROR", code="INFERENCE_FAILED", message="CUDA out of memory")), HAPPY)
    summary, t = batch(tmp_path, specs(tmp_path), transport=FakeTransport(exec=exec_))
    assert statuses(summary) == [
        ("j1", h3.COMPLETED, None),
        ("j2", h3.FAILED, "INFERENCE_FAILED"),
        ("j3", h3.COMPLETED, None),
    ]
    assert summary["status"] == "PARTIAL"
    assert t.subcommands.count("new") == 1 and t.subcommands.count("stop") == 1


def test_chained_job_starts_from_the_previous_last_frame_on_the_vm(tmp_path):
    uploaded = {}

    def upload(args, on_line):
        if args[3].endswith("job.json"):
            job = json.loads(open(args[3], encoding="utf-8").read())
            uploaded[job["job_id"]] = job
        return "[colab] Uploaded"

    job_specs = specs(tmp_path, n=1) + [h3.JobSpec(chain=True, description="She keeps smiling.", job_id="j2")]
    summary, t = batch(tmp_path, job_specs, transport=FakeTransport(upload=upload))
    assert statuses(summary) == [("j1", h3.COMPLETED, None), ("j2", h3.COMPLETED, None)]
    assert t.subcommands.count("upload") == 3  # j2 sends only its job.json
    assert uploaded["j2"]["remote_image"] == uploaded["j1"]["remote_last_frame"] == "/content/h3_j1_last_frame.png"
    assert summary["records"][1]["chained_from"] == "j1"
    assert summary["records"][1]["resolution"] == summary["records"][0]["resolution"]


def test_a_chain_after_a_failed_clip_is_cancelled(tmp_path):
    exec_ = per_call(emit(marker("ERROR", code="INFERENCE_FAILED", message="boom")), HAPPY)
    job_specs = specs(tmp_path, n=1) + [h3.JobSpec(chain=True, description="Then she waves.", job_id="j2")]
    summary, t = batch(tmp_path, job_specs, transport=FakeTransport(exec=exec_))
    assert statuses(summary)[1] == ("j2", h3.CANCELLED, "CHAIN_SOURCE_FAILED")
    assert t.subcommands.count("exec") == 1


def test_a_chain_needs_the_last_frame_marker(tmp_path):
    job_specs = specs(tmp_path, n=1) + [h3.JobSpec(chain=True, description="Then she waves.", job_id="j2")]
    summary, _ = batch(tmp_path, job_specs, transport=FakeTransport(exec=per_call(NO_LAST_FRAME, HAPPY)))
    assert statuses(summary) == [("j1", h3.COMPLETED, None), ("j2", h3.CANCELLED, "CHAIN_SOURCE_FAILED")]


def test_exec_timeout_recycles_the_session_and_is_not_retried(tmp_path):
    exec_ = per_call(HAPPY, raising(h3.ColabTimeout("colab exec exceeded 3720 seconds")), HAPPY)
    summary, t = batch(tmp_path, specs(tmp_path), transport=FakeTransport(exec=exec_))
    assert statuses(summary)[1] == ("j2", h3.TIMEOUT, "TIMEOUT")
    assert statuses(summary)[2] == ("j3", h3.COMPLETED, None)
    assert t.subcommands.count("exec") == 3
    assert t.subcommands.count("new") == 2 and t.subcommands.count("stop") == 2
    third = summary["records"][2]
    assert third["session"] == "h3-b1-2" and third["session_reused"] is False


def test_exec_failure_without_a_marker_recycles_the_session(tmp_path):
    lost = h3.ColabCommandError("colab exec", 1, ["[colab] Session 'h3-b1' appears to be lost (404/401)."])
    summary, t = batch(tmp_path, specs(tmp_path), transport=FakeTransport(exec=per_call(HAPPY, raising(lost), HAPPY)))
    assert statuses(summary)[1] == ("j2", h3.FAILED, "INFERENCE_FAILED")
    assert t.subcommands.count("new") == 2 and summary["records"][2]["status"] == h3.COMPLETED


def test_ctrl_c_cancels_the_rest_and_stops_the_session(tmp_path):
    summary, t = batch(tmp_path, specs(tmp_path), transport=FakeTransport(exec=raising(KeyboardInterrupt())))
    assert [s for _, s, _ in statuses(summary)] == [h3.CANCELLED] * 3
    assert t.subcommands.count("exec") == 1 and t.subcommands.count("stop") == 1


def test_a_busy_account_starts_nothing(tmp_path):
    transport = FakeTransport(usage=usage_seq(100.0, active_first=1))
    summary, t = batch(tmp_path, specs(tmp_path), transport=transport)
    assert [c for _, _, c in statuses(summary)] == ["COLAB_BUSY"] * 3
    assert "new" not in t.subcommands
    assert "--ignore-busy" in summary["records"][0]["hint"]


def test_ignore_busy_starts_anyway(tmp_path):
    transport = FakeTransport(usage=usage_seq(100.0, 100.0, 99.0, active_first=1))
    summary, _ = batch(tmp_path, specs(tmp_path, n=1), transport=transport, ignore_busy=True)
    assert summary["status"] == h3.COMPLETED


def test_a_session_that_cannot_open_ends_every_job(tmp_path):
    refused = h3.ColabCommandError("colab new", 1, ["[colab] Backend rejected accelerator 'A100'."])
    not_found = h3.ColabCommandError("colab stop", 1, ["[colab] Session 'h3-b1' not found."])
    transport = FakeTransport(new=raising(refused), stop=raising(not_found))
    summary, t = batch(tmp_path, specs(tmp_path), transport=transport)
    assert [c for _, _, c in statuses(summary)] == ["GPU_UNAVAILABLE"] * 3
    assert t.subcommands.count("new") == 1
    assert summary["records"][0]["session_status"] == "not_created"


def test_settle_rereads_the_balance_after_the_wait(tmp_path):
    waited = []
    transport = FakeTransport(usage=usage_seq(100.0, 100.0, 99.0, 98.7))
    summary, _ = batch(tmp_path, specs(tmp_path, n=1), transport=transport, settle_seconds=90, sleep=waited.append)
    assert waited == [90]
    assert summary["cu_used_measured"] == 1.0 and summary["cu_used_settled"] == 1.3
    assert summary["records"][0]["cu_used_settled"] == 1.3  # a one-clip batch copies it into the record
    line = json.loads((tmp_path / "out" / "h3_batches.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert line["cu_used_settled"] == 1.3 and "records" not in line


def test_a_bad_input_is_dropped_before_any_colab_call(tmp_path):
    job_specs = specs(tmp_path)
    job_specs[1].image = tmp_path / "missing.png"
    summary, t = batch(tmp_path, job_specs)
    assert statuses(summary)[1] == ("j2", h3.FAILED, "INVALID_INPUT")
    assert summary["records"][1]["failed_stage"] == h3.PREPARING
    assert t.subcommands.count("exec") == 2


def test_all_bad_inputs_cost_nothing(tmp_path):
    job_specs = specs(tmp_path, n=2)
    for spec in job_specs:
        spec.duration = 30
    summary, t = batch(tmp_path, job_specs)
    assert t.calls == [] and summary["status"] == h3.FAILED


def test_the_first_job_cannot_chain(tmp_path):
    summary, t = batch(tmp_path, [h3.JobSpec(chain=True, description="x", job_id="j1")])
    assert statuses(summary) == [("j1", h3.FAILED, "INVALID_INPUT")] and t.calls == []


def test_one_runtime_means_one_gpu(tmp_path):
    job_specs = specs(tmp_path, n=2)
    job_specs[1].gpu = "L4"
    summary, _ = batch(tmp_path, job_specs)
    assert statuses(summary)[1][2] == "INVALID_INPUT" and "share one Colab runtime" in summary["records"][1]["error"]


def test_batch_records_are_written(tmp_path):
    summary, _ = batch(tmp_path, specs(tmp_path))
    saved = json.loads((tmp_path / "out" / "batches" / "b1.json").read_text(encoding="utf-8"))
    assert [j["job_id"] for j in saved["jobs"]] == ["j1", "j2", "j3"]
    ledger = [json.loads(x) for x in (tmp_path / "out" / "h3_jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["batch_id"] for e in ledger] == ["b1"] * 3
    assert set(ledger[0]) == set(h3.LEDGER_FIELDS)


def test_job_spec_chain_round_trips():
    spec = h3.JobSpec(chain=True, description="d", job_id="x")
    assert h3.JobSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_a_spec_without_image_must_chain():
    try:
        h3.JobSpec.from_dict({"prompt": {"description": "d"}})
    except h3.InputError as exc:
        assert "chain" in str(exc)
    else:
        raise AssertionError("expected InputError")


# --- the CLI ------------------------------------------------------------------------------------------


def test_segments_become_chained_jobs():
    run = import_skill("run")
    base = h3.JobSpec(image="a.png", description="d", output="output/a.mp4", job_id="t", seed=7)
    parts = run.segment_specs(base, 3)
    assert [p.chain for p in parts] == [False, True, True]
    assert [p.image for p in parts] == ["a.png", None, None]
    assert [p.output.replace("\\", "/") for p in parts] == [
        "output/a_part1.mp4",
        "output/a_part2.mp4",
        "output/a_part3.mp4",
    ]
    assert [p.job_id for p in parts] == ["t-p1", "t-p2", "t-p3"] and [p.seed for p in parts] == [7, None, None]


def test_manifest_flags_apply_to_every_job(tmp_path):
    run = import_skill("run")
    path = tmp_path / "batch.json"
    jobs = [{"input": {"image": "a.png"}, "prompt": "x"}, {"chain": True, "prompt": "y"}]
    path.write_text(json.dumps({"batch_id": "m1", "jobs": jobs}), encoding="utf-8")
    args = run.build_parser().parse_args(["--manifest", str(path), "--gpu", "L4", "--no-high-mem"])
    job_specs, batch_id = run.specs_from_manifest(args)
    assert batch_id == "m1" and [s.chain for s in job_specs] == [False, True]
    assert {(s.gpu, s.high_mem) for s in job_specs} == {("L4", False)}


def test_batch_cli_exit_code_for_bad_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("H3_OUTPUT_DIR", str(tmp_path / "out"))
    run = import_skill("run")
    path = tmp_path / "batch.json"
    jobs = [{"input": {"image": str(tmp_path / "nope.png")}, "prompt": "x"}]
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    code = run.main(["--manifest", str(path), "--config", str(tmp_path / "c.json")])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and summary["records"][0]["error_code"] == "INVALID_INPUT"
