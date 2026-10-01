"""The job store and the six tools, without a worker: what a caller on the local machine can do, and - just
as important for a folder that two machines sync - what it never does.
"""

import json
import os
import time

import pytest
from aiwf_fakes import load_script, make_workspace
from controller import WorkflowError, jobs, tools
from controller.compute import WORKER_STATE_FILE
from controller.storage import LAYOUT, DriveFolderStorage, resolve_root


@pytest.fixture
def ctx(tmp_path):
    storage, _ = make_workspace(tmp_path)
    return tools.open_context(storage=storage)


def tree(storage):
    """Every file in the workspace with its content: the before/after picture of a tool call."""
    root = storage.root
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_the_workspace_has_the_documented_tree(ctx):
    for rel in LAYOUT:
        assert (ctx.storage.root / rel).is_dir(), rel
    for name in ("00_setup", "01_comfyui", "02_image_generation", "03_video_generation", "99_manual_debug"):
        assert (ctx.storage.root / "notebooks" / (name + ".ipynb")).is_file()


def test_list_workflows_describes_what_a_caller_fills_in(ctx):
    out = tools.list_workflows(context=ctx)
    by_name = {w["name"]: w for w in out["workflows"]}
    assert set(by_name) == {"z-image-basic", "minimax-h3-basic", "test-generation"}
    assert by_name["minimax-h3-basic"]["type"] == "video"
    assert by_name["minimax-h3-basic"]["inputs"][0]["sources"] == ["generate", "job", "file"]
    prompt = next(p for p in by_name["z-image-basic"]["parameters"] if p["name"] == "prompt")
    assert prompt["required"] and prompt["in"] == "input"
    assert "problems" not in out


def test_create_job_writes_the_documented_schema(ctx):
    out = tools.create_job("z-image-basic", {"prompt": "A white mug."}, {"aspect": "16:9", "seed": 7}, context=ctx)
    assert out["status"] == "pending" and out["jobs"] == [out["job_id"]]
    assert out["worker"]["alive"] is False and "01_comfyui.ipynb" in out["hint"]
    job = ctx.storage.read_json("jobs/pending/%s.json" % out["job_id"])
    assert job["task_type"] == "text_to_image" and job["workflow"] == "z-image-basic"
    assert job["input"] == {"prompt": "A white mug."}
    assert job["parameters"] == {"aspect": "16:9", "width": 1344, "height": 768, "steps": 8, "cfg": 2.0, "seed": 7, "solo": False, "allow_text": False}  # fmt: skip
    for key in ("created_at", "started_at", "completed_at", "output", "error"):
        assert key in job
    assert job["started_at"] is None and job["output"] is None and "graph" not in job


def test_values_may_arrive_in_either_dict(ctx):
    a = tools.create_job("z-image-basic", {"prompt": "x", "steps": 9, "seed": 1}, context=ctx)
    b = tools.create_job("z-image-basic", None, {"prompt": "x", "steps": 9, "seed": 1}, context=ctx)
    ja, jb = ctx.store.get(a["job_id"]), ctx.store.get(b["job_id"])
    assert (ja["input"], ja["parameters"]) == (jb["input"], jb["parameters"])


@pytest.mark.parametrize(
    "workflow,values,code",
    [
        ("nope", {"prompt": "x"}, "UNKNOWN_WORKFLOW"),
        ("z-image-basic", {}, "INVALID_INPUT"),
        ("z-image-basic", {"prompt": "x", "cfg": 1.0}, "INVALID_INPUT"),
        ("minimax-h3-basic", {"prompt": "A teen dances."}, "PROMPT_REJECTED"),
        ("minimax-h3-basic", {"prompt": "A cup.", "first_frame_prompt": "a child"}, "PROMPT_REJECTED"),
        ("minimax-h3-basic", {"prompt": "A cup.", "soundscape": "nsfw noises"}, "PROMPT_REJECTED"),
        ("minimax-h3-basic", {"prompt": "A cup.", "first_frame": {"source": "upload"}}, "INVALID_INPUT"),
        ("minimax-h3-basic", {"prompt": "A cup.", "first_frame": "inputs/a.png"}, "ATTESTATION_REQUIRED"),
        ("minimax-h3-basic", {"prompt": "A cup.", "first_frame": {"source": "file", "path": "outputs/images/a.png", "attested": True}}, "INVALID_INPUT"),
        ("minimax-h3-basic", {"prompt": "A cup.", "first_frame": {"source": "file", "path": "inputs/../configs/a.png", "attested": True}}, "INVALID_INPUT"),
        ("minimax-h3-basic", {"prompt": "A cup.", "first_frame": {"source": "file", "path": "inputs/a.txt", "attested": True}}, "INVALID_INPUT"),
        ("minimax-h3-basic", {"prompt": "A cup.", "first_frame": "job-does-not-exist"}, "JOB_NOT_FOUND"),
    ],
)  # fmt: skip
def test_a_refused_job_writes_nothing(ctx, workflow, values, code):
    before = tree(ctx.storage)
    with pytest.raises(WorkflowError) as exc:
        tools.create_job(workflow, values, context=ctx)
    assert exc.value.code == code
    assert tree(ctx.storage) == before


def test_a_video_job_queues_its_first_frame_job_first(ctx):
    out = tools.create_job(
        "minimax-h3-basic", {"prompt": "A cup of coffee, steam rises.", "aspect": "16:9"}, context=ctx
    )
    assert len(out["jobs"]) == 2
    video = ctx.store.get(out["job_id"])
    child = ctx.store.get(video["depends_on"][0])
    assert video["input"]["first_frame"] == {"source": "generate", "job_id": child["job_id"]}
    assert child["workflow"] == "z-image-basic" and child["source"] == "chain" and child["parent"] == video["job_id"]
    assert child["input"]["prompt"] == "A cup of coffee, steam rises." and child["parameters"]["aspect"] == "16:9"
    assert video["task_type"] == "text_to_video"


def test_a_separate_first_frame_prompt_is_used_for_the_still(ctx):
    out = tools.create_job(
        "minimax-h3-basic", {"prompt": "Steam rises.", "first_frame_prompt": "A cup on a table."}, context=ctx
    )
    child = ctx.store.get(ctx.store.get(out["job_id"])["depends_on"][0])
    assert child["input"]["prompt"] == "A cup on a table."


def test_an_attested_file_records_the_statement(ctx):
    out = tools.create_job(
        "minimax-h3-basic",
        {"prompt": "A cup.", "first_frame": {"source": "file", "path": "inputs/a.png", "attested": True}},
        context=ctx,
    )
    ref = ctx.store.get(out["job_id"])["input"]["first_frame"]
    assert ref["attested"] is True and "not a photo of a real person" in ref["attestation"]
    assert out["jobs"] == [out["job_id"]]


def test_a_video_job_may_point_at_an_image_job_but_not_at_a_video_job(ctx):
    image = tools.create_job("z-image-basic", {"prompt": "A cup."}, context=ctx)["job_id"]
    video = tools.create_job("minimax-h3-basic", {"prompt": "Steam rises.", "first_frame": image}, context=ctx)
    assert ctx.store.get(video["job_id"])["depends_on"] == [image]
    with pytest.raises(WorkflowError) as exc:
        tools.create_job("minimax-h3-basic", {"prompt": "Steam rises.", "first_frame": video["job_id"]}, context=ctx)
    assert "not an image job" in str(exc.value)


def test_the_tools_only_ever_add_pending_jobs_and_cancel_requests(ctx):
    """The single-writer rule: no tool modifies or moves a file that already exists."""
    job_id = tools.create_job("z-image-basic", {"prompt": "x"}, context=ctx)["job_id"]
    before = tree(ctx.storage)
    tools.list_workflows(context=ctx)
    tools.get_job_status(job_id, context=ctx)
    tools.list_jobs(context=ctx)
    tools.get_result(job_id, context=ctx)
    assert tree(ctx.storage) == before
    tools.cancel_job(job_id, context=ctx)
    tools.create_job("test-generation", {}, context=ctx)
    after = tree(ctx.storage)
    assert {k: v for k, v in after.items() if k in before} == before
    added = sorted(set(after) - set(before))
    assert len(added) == 2
    assert added[0] == "jobs/cancel/%s.request" % job_id and added[1].startswith("jobs/pending/")


def test_status_list_cancel_and_result_on_a_pending_job(ctx):
    job_id = tools.create_job("z-image-basic", {"prompt": "x"}, context=ctx)["job_id"]
    status = tools.get_job_status(job_id, context=ctx)
    assert status["status"] == "pending" and status["worker"]["alive"] is False and "hint" in status
    assert tools.get_result(job_id, context=ctx)["output"] is None
    listing = tools.list_jobs(context=ctx)
    assert listing["counts"] == {"pending": 1} and listing["jobs"][0]["job_id"] == job_id
    assert tools.list_jobs("completed", context=ctx)["jobs"] == []
    out = tools.cancel_job(job_id, context=ctx)
    assert out["cancel_requested"] is True and out["status"] == "pending"
    assert tools.get_job_status(job_id, context=ctx)["cancel_requested"] is True
    with pytest.raises(WorkflowError) as exc:
        tools.get_job_status("job-unknown", context=ctx)
    assert exc.value.code == "JOB_NOT_FOUND"
    with pytest.raises(WorkflowError):
        tools.list_jobs("done", context=ctx)
    with pytest.raises(WorkflowError):
        tools.get_job_status("../../etc/passwd", context=ctx)


def test_a_live_worker_changes_the_hint(ctx):
    job_id = tools.create_job("z-image-basic", {"prompt": "x"}, context=ctx)["job_id"]
    state = {
        "session": "s1",
        "worker_state": "ready",
        "serving": False,
        "updated_epoch": time.time(),
        "gpu": {"name": "L4", "vram_gib": 22.2},
    }
    ctx.storage.write_json(WORKER_STATE_FILE, state)
    status = tools.get_job_status(job_id, context=ctx)
    assert status["worker"]["alive"] and not status["worker"]["serving"] and "not taking jobs" in status["hint"]
    ctx.storage.write_json(
        WORKER_STATE_FILE, dict(state, serving=True, worker_state="idle", waiting={job_id: "needs a GPU"})
    )
    status = tools.get_job_status(job_id, context=ctx)
    assert status["worker"]["serving"] and "hint" not in status and status["waiting"] == "needs a GPU"
    ctx.storage.write_json(WORKER_STATE_FILE, dict(state, serving=True, updated_epoch=time.time() - 600))
    assert tools.get_job_status(job_id, context=ctx)["worker"]["alive"] is False
    ctx.storage.write_json(WORKER_STATE_FILE, dict(state, serving=True, worker_state="stopped"))
    assert tools.get_job_status(job_id, context=ctx)["worker"]["alive"] is False


# --- the store's worker-only half ------------------------------------------------------------------


def test_the_folder_is_the_state_and_the_later_folder_wins(ctx):
    store = ctx.store
    job = store.get(tools.create_job("test-generation", {}, context=ctx)["job_id"])
    running = store.claim(job, "s1")
    assert running["status"] == "running" and running["attempt"] == 1 and running["session"] == "s1"
    assert not ctx.storage.exists("jobs/pending/%s.json" % job["job_id"])
    # An interrupted move leaves two copies; the terminal one is the truth.
    ctx.storage.write_json("jobs/pending/%s.json" % job["job_id"], job)
    assert store.find(job["job_id"])["status"] == "running"
    done = store.finish(running, jobs.COMPLETED, output={"path": "outputs/images/x.png"})
    assert done["completed_at"] and store.find(job["job_id"])["status"] == "completed"
    assert ctx.storage.list("jobs/running") == [] and ctx.storage.list("jobs/pending") == []
    assert [j["status"] for j in store.list()] == ["completed"]


def test_completed_needs_an_output_and_cancelled_lives_in_failed(ctx):
    store = ctx.store
    job = store.claim(store.get(tools.create_job("test-generation", {}, context=ctx)["job_id"]), "s1")
    with pytest.raises(WorkflowError):
        store.finish(job, jobs.COMPLETED)
    with pytest.raises(WorkflowError):
        store.finish(job, jobs.RUNNING)
    store.request_cancel(job["job_id"])
    store.finish(job, jobs.CANCELLED, error={"code": "CANCELLED"})
    assert ctx.storage.exists("jobs/failed/%s.json" % job["job_id"])
    assert not store.cancel_requested(job["job_id"])  # the request is cleared with the job
    assert (
        tools.list_jobs("cancelled", context=ctx)["total"] == 1 and tools.list_jobs("failed", context=ctx)["total"] == 0
    )


def test_a_half_written_file_is_never_listed(ctx):
    (ctx.storage.root / "jobs" / "pending" / "job-x.json.tmp-123").write_text("{")
    (ctx.storage.root / "jobs" / "pending" / "job-y.json").write_text("{not json")
    (ctx.storage.root / "jobs" / "pending" / "job-z (1).json").write_text(json.dumps({"job_id": "job-z"}))
    assert ctx.store.list() == []


# --- storage and the command line -------------------------------------------------------------------


@pytest.mark.parametrize("rel", ["/etc/passwd", "../x", "jobs/../../x", "", "C:\\x\\..\\..\\y"])
def test_storage_refuses_paths_outside_the_workspace(tmp_path, rel):
    storage = DriveFolderStorage(tmp_path)
    for call in (storage.read_json, storage.exists, lambda r: storage.write_text(r, "x")):
        with pytest.raises(WorkflowError) as exc:
            call(rel)
        assert exc.value.code == "INVALID_PATH"


def test_the_workspace_is_found_from_the_argument_then_the_environment(tmp_path):
    assert resolve_root(tmp_path, env={}) == tmp_path
    assert resolve_root(None, env={"AIWF_ROOT": str(tmp_path)}) == tmp_path


def cli(capsys, *argv):
    code = load_script("aiwf").main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_the_command_line_is_the_same_six_tools(ctx, capsys, monkeypatch):
    for key in [k for k in os.environ if k.upper().endswith("_API_KEY")]:
        monkeypatch.delenv(key)  # the tool interface needs no credential of any kind
    root = str(ctx.storage.root)
    code, out = cli(capsys, "--root", root, "list-workflows")
    assert code == 0 and out["ok"] and len(out["workflows"]) == 3
    code, out = cli(
        capsys,
        "--root",
        root,
        "create-job",
        "z-image-basic",
        "--prompt",
        "白色馬克杯",
        "--set",
        "aspect=16:9",
        "--set",
        "steps=9",
    )
    assert code == 0 and out["status"] == "pending"
    job_id = out["job_id"]
    job = ctx.store.get(job_id)
    assert job["input"]["prompt"] == "白色馬克杯" and job["parameters"]["steps"] == 9 and job["source"] == "tool"
    code, out = cli(capsys, "--root", root, "status", job_id)
    assert code == 0 and out["status"] == "pending"
    code, out = cli(capsys, "--root", root, "list-jobs", "--status", "pending")
    assert code == 0 and out["total"] == 1
    code, out = cli(capsys, "--root", root, "result", job_id)
    assert code == 0 and out["output"] is None
    code, out = cli(capsys, "--root", root, "cancel", job_id)
    assert code == 0 and out["cancel_requested"] is True
    code, out = cli(capsys, "--root", root, "create-job", "z-image-basic", "--prompt", "x", "--set", "cfg=1.0")
    assert code == 1 and out == {"ok": False, "error": out["error"]} and out["error"]["code"] == "INVALID_INPUT"
    code, out = cli(capsys, "--root", root, "status", "job-nope")
    assert code == 1 and out["error"]["code"] == "JOB_NOT_FOUND"


def test_the_command_line_attests_a_file_input(ctx, capsys):
    root = str(ctx.storage.root)
    args = ["--root", root, "create-job", "minimax-h3-basic", "--prompt", "A cup.", "--set", "first_frame=inputs/a.png"]
    code, out = cli(capsys, *args)
    assert code == 1 and out["error"]["code"] == "ATTESTATION_REQUIRED" and "real person" in out["error"]["hint"]
    code, out = cli(capsys, *args, "--attest", "first_frame")
    assert code == 0
    assert ctx.store.get(out["job_id"])["input"]["first_frame"]["attested"] is True
