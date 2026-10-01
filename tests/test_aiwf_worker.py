"""The worker end to end, against a fake ComfyUI: both ways in (the tool interface and a notebook form), the
input checks, failure isolation, the queue rules, and the session-restart guarantee - a second runtime on the
same workspace installs and downloads nothing.
"""

import json

import pytest
from aiwf_fakes import GPU_L4, GPU_NONE, FakeComfy, Rig, make_workspace, nb, png_bytes
from controller import WorkflowError, safety, tools
from controller.compute import WORKER_STATE_FILE

COFFEE = "A cup of coffee on a table, thin steam rises slowly."


@pytest.fixture
def workspace(tmp_path):
    storage, blobs = make_workspace(tmp_path)
    return tmp_path, storage, blobs


@pytest.fixture
def make_rig(workspace):
    """Runtimes on one workspace. They share one fake ComfyUI, which outlives each of them - a second session
    in these tests is a new Rig (a new runtime disk) on the same workspace."""
    tmp_path, storage, blobs = workspace
    fake = FakeComfy(storage)
    rigs = []

    def make(vm="vm", **kwargs):
        rigs.append(Rig(tmp_path, storage, kwargs.pop("blobs", blobs), vm=vm, fake=fake, **kwargs))
        return rigs[-1]

    yield make
    for r in rigs:
        r.close()
    fake.close()


@pytest.fixture
def rig(make_rig):
    return make_rig()


def create(rig, workflow, values):
    return tools.create_job(workflow, values, context=rig.worker.ctx)["job_id"]


def job_of(rig, job_id):
    return rig.worker.ctx.store.get(job_id)


# --- the two ways in ---------------------------------------------------------------------------------


def test_a_tool_job_is_rendered_validated_and_stored(rig):
    job_id = create(rig, "z-image-basic", {"prompt": "A white mug.", "aspect": "16:9", "seed": 5})
    done = rig.worker.drain()
    assert [j["job_id"] for j in done] == [job_id]
    job = job_of(rig, job_id)
    assert job["status"] == "completed" and job["started_at"] and job["completed_at"] and job["error"] is None
    out = job["output"]
    assert out["path"] == "outputs/images/%s.png" % job_id and (out["width"], out["height"]) == (1344, 768)
    assert rig.storage.read_bytes(out["path"]) == png_bytes(1344, 768)
    assert rig.storage.list("jobs/pending") == [] and rig.storage.list("jobs/running") == []
    submitted = rig.fake.submitted[-1]
    assert submitted == rig.storage.read_json("logs/jobs/%s/workflow.json" % job_id)
    assert submitted["6"]["inputs"]["text"] == "A white mug." and submitted["3"]["inputs"]["seed"] == 5
    assert submitted["7"]["inputs"]["text"] == safety.build_negative()
    result = tools.get_result(job_id, context=rig.worker.ctx)
    assert result["output"]["available_locally"] and result["gpu"] == "Fake A100"
    ledger = [json.loads(line) for line in rig.storage.read_text("logs/jobs.jsonl").splitlines()]
    assert ledger[-1]["job_id"] == job_id and ledger[-1]["status"] == "completed"


def test_the_notebook_form_and_the_tool_take_the_same_path(rig, capsys):
    values = {"prompt": "A white mug.", "aspect": "3:2", "seed": 11}
    tool_id = create(rig, "z-image-basic", values)
    rig.worker.drain()
    form = nb.generate(
        rig.session,
        "z-image-basic",
        {"prompt": "A white mug.", "aspect": "3:2"},
        {"seed": 11, "width": 0, "negative": ""},
    )
    printed = capsys.readouterr().out
    assert "Job ID" in printed and "Output Path" in printed and "Generation Time" in printed and "completed" in printed
    a, b = job_of(rig, tool_id), job_of(rig, form["job_id"])
    assert (a["source"], b["source"]) == ("tool", "notebook")
    for key in ("workflow", "task_type", "input", "parameters", "status"):
        assert a[key] == b[key], key
    assert set(a) == set(b)
    assert a["output"]["sha256"] == b["output"]["sha256"]
    assert rig.fake.submitted[-1]["3"] == rig.fake.submitted[-2]["3"]  # the same sampler settings were submitted


def test_a_prompt_becomes_a_video_through_a_generated_first_frame(rig):
    """Test A/B in miniature: text in, MP4 in outputs/videos, the first frame made with the safety negatives."""
    out = tools.create_job(
        "minimax-h3-basic", {"prompt": COFFEE, "duration": 5, "aspect": "16:9"}, context=rig.worker.ctx
    )
    rig.worker.drain()
    video = job_of(rig, out["job_id"])
    child = job_of(rig, video["depends_on"][0])
    assert child["status"] == "completed" and video["status"] == "completed"
    still, clip = rig.fake.submitted[-2], rig.fake.submitted[-1]
    assert still["7"]["inputs"]["text"] == safety.build_negative() and still["3"]["inputs"]["cfg"] >= safety.MIN_CFG
    assert clip["6"]["inputs"]["image"] == "%s_first_frame.png" % video["job_id"]
    assert (rig.env.comfy_dir / "input" / clip["6"]["inputs"]["image"]).read_bytes() == png_bytes(1344, 768)
    assert (
        clip["7"]["inputs"]["width"] == 1344
        and clip["7"]["inputs"]["height"] == 768
        and clip["7"]["inputs"]["length"] == 124
    )
    assert "[Shot 1] " + COFFEE in clip["7"]["inputs"]["prompt"]
    o = video["output"]
    assert (
        o["path"] == "outputs/videos/%s.mp4" % video["job_id"]
        and o["preview"] == "outputs/previews/%s.jpg" % video["job_id"]
    )
    assert (
        o["video_codec"] == "h264"
        and o["audio_codec"] == "aac"
        and o["fps"] == 24.0
        and (o["width"], o["height"]) == (1344, 768)
    )
    assert rig.storage.exists(o["path"]) and rig.storage.exists(o["preview"])
    assert rig.env.actions["model:minimax_h3_fl2va_pruned_fp8_scaled.safetensors"] == "downloaded"


def test_a_new_workflow_is_a_json_file_and_a_registry_entry(rig):
    """Test D: add a graph and an entry to the workspace while the worker runs - no core change, no restart."""
    create(rig, "test-generation", {})
    rig.worker.drain()
    graph = rig.storage.read_json("workflows/image/z-image-basic.json")
    graph["3"]["inputs"]["steps"] = 20
    rig.storage.write_json("workflows/image/my-model.json", graph)
    registry = rig.storage.read_json("workflows/registry.json")
    entry = json.loads(json.dumps(registry["workflows"]["z-image-basic"]))
    entry.update(workflow_file="image/my-model.json", title="My model")
    entry["parameters"]["steps"]["default"] = 20
    registry["workflows"]["my-model"] = entry
    rig.storage.write_json("workflows/registry.json", registry)
    job_id = tools.create_job("my-model", {"prompt": "A lamp."}, root=rig.storage.root)["job_id"]
    rig.worker.drain()
    assert job_of(rig, job_id)["status"] == "completed"
    assert rig.fake.submitted[-1]["3"]["inputs"]["steps"] == 20


# --- first frame sources -------------------------------------------------------------------------------


def test_a_completed_image_job_can_be_the_first_frame(rig):
    image = create(rig, "z-image-basic", {"prompt": "A cup on a table.", "aspect": "9:16"})
    rig.worker.drain()
    video = create(rig, "minimax-h3-basic", {"prompt": COFFEE, "first_frame": image})
    rig.worker.drain()
    assert job_of(rig, video)["status"] == "completed"
    assert job_of(rig, video)["output"]["height"] == 1344


def test_a_first_frame_that_was_replaced_after_its_job_is_refused(rig):
    image = create(rig, "z-image-basic", {"prompt": "A cup on a table."})
    rig.worker.drain()
    rig.storage.write_bytes(job_of(rig, image)["output"]["path"], png_bytes(1024, 1024, color=(1, 2, 3)))
    video = create(rig, "minimax-h3-basic", {"prompt": COFFEE, "first_frame": image})
    rig.worker.drain()
    assert job_of(rig, video)["error"]["code"] == "INPUT_CHANGED"


def test_an_attested_file_is_used_and_an_unattested_one_never_is(rig):
    rig.storage.write_bytes("inputs/frame.png", png_bytes(1024, 576))
    ref = {"source": "file", "path": "inputs/frame.png", "attested": True}
    ok = create(rig, "minimax-h3-basic", {"prompt": COFFEE, "first_frame": ref})
    rig.worker.drain()
    done = job_of(rig, ok)
    assert done["status"] == "completed" and (done["output"]["width"], done["output"]["height"]) == (1376, 768)
    # A job file edited by hand to drop the confirmation is refused by the worker, not just by create_job.
    edited = create(rig, "minimax-h3-basic", {"prompt": COFFEE, "first_frame": ref})
    job = rig.storage.read_json("jobs/pending/%s.json" % edited)
    job["input"]["first_frame"]["attested"] = False
    rig.storage.write_json("jobs/pending/%s.json" % edited, job)
    missing = create(rig, "minimax-h3-basic", {"prompt": COFFEE, "first_frame": dict(ref, path="inputs/nope.png")})
    rig.worker.drain()
    assert job_of(rig, edited)["error"]["code"] == "ATTESTATION_REQUIRED"
    assert job_of(rig, missing)["error"]["code"] == "INPUT_NOT_FOUND"


def test_a_failed_first_frame_fails_the_video_without_running_it(rig):
    out = tools.create_job("minimax-h3-basic", {"prompt": COFFEE}, context=rig.worker.ctx)
    child = job_of(rig, out["job_id"])["depends_on"][0]
    rig.fake.behaviors[child] = "error"
    rig.worker.drain()
    assert job_of(rig, child)["error"]["code"] == "COMFY_EXECUTION_ERROR"
    assert job_of(rig, out["job_id"])["error"]["code"] == "DEPENDENCY_FAILED"
    assert len(rig.fake.submitted) == 1


# --- the worker rechecks what it is given ---------------------------------------------------------------


@pytest.mark.parametrize(
    "edit,code",
    [
        (lambda job: job["parameters"].update(cfg=1.0), "INVALID_INPUT"),
        (lambda job: job["parameters"].update(steps=500), "INVALID_INPUT"),
        (lambda job: job["parameters"].update(sampler="custom"), "INVALID_INPUT"),
        (lambda job: job.update(workflow="deleted-workflow"), "UNKNOWN_WORKFLOW"),
    ],
)
def test_a_job_file_edited_by_hand_is_validated_again(rig, edit, code):
    job_id = create(rig, "z-image-basic", {"prompt": "A mug."})
    job = rig.storage.read_json("jobs/pending/%s.json" % job_id)
    edit(job)
    rig.storage.write_json("jobs/pending/%s.json" % job_id, job)
    rig.worker.drain()
    assert job_of(rig, job_id)["status"] == "failed" and job_of(rig, job_id)["error"]["code"] == code
    assert rig.fake.submitted == []


def test_an_edited_h3_prompt_is_screened_again_on_the_worker(rig):
    rig.storage.write_bytes("inputs/frame.png", png_bytes(768, 1344))
    ref = {"source": "file", "path": "inputs/frame.png", "attested": True}
    job_id = create(rig, "minimax-h3-basic", {"prompt": COFFEE, "first_frame": ref})
    job = rig.storage.read_json("jobs/pending/%s.json" % job_id)
    job["input"]["prompt"] = "A teenager dances."
    rig.storage.write_json("jobs/pending/%s.json" % job_id, job)
    rig.worker.drain()
    assert job_of(rig, job_id)["error"]["code"] == "PROMPT_REJECTED" and rig.fake.submitted == []


# --- failures stay with their job -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "behavior,code",
    [
        ("error", "COMFY_EXECUTION_ERROR"),
        ("reject", "COMFY_REJECTED"),
        ("bad_png", "INVALID_OUTPUT"),
        ("wrong_size", "INVALID_OUTPUT"),
        ("no_output", "OUTPUT_NOT_FOUND"),
        ("hang", "TIMEOUT"),
    ],
)
def test_one_failed_job_does_not_stop_the_next(rig, behavior, code):
    # A 30 s limit instead of 1800: the clock is fake, but every poll is a real HTTP round trip.
    registry = rig.storage.read_json("workflows/registry.json")
    registry["workflows"]["z-image-basic"]["timeout_seconds"] = 30
    rig.storage.write_json("workflows/registry.json", registry)
    bad = create(rig, "z-image-basic", {"prompt": "A mug."})
    good = create(rig, "z-image-basic", {"prompt": "A lamp."})
    rig.fake.behaviors[bad] = behavior
    rig.worker.drain()
    failed = job_of(rig, bad)
    assert failed["status"] == "failed" and failed["error"]["code"] == code and failed["output"] is None
    assert not rig.storage.exists("outputs/images/%s.png" % bad)  # nothing invalid reaches outputs/
    assert job_of(rig, good)["status"] == "completed"
    assert rig.storage.list("jobs/failed") == ["%s.json" % bad]


def test_an_invalid_video_is_not_stored(rig):
    rig.storage.write_bytes("inputs/frame.png", png_bytes(768, 1344))
    job_id = create(
        rig,
        "minimax-h3-basic",
        {"prompt": COFFEE, "first_frame": {"source": "file", "path": "inputs/frame.png", "attested": True}},
    )
    rig.fake.behaviors[job_id] = "bad_png"  # the fake prober reports the wrong codec for it
    rig.worker.drain()
    error = job_of(rig, job_id)["error"]
    assert error["code"] == "INVALID_OUTPUT" and "codec mpeg4" in error["message"]
    assert rig.storage.list("outputs/videos") == []


# --- queue rules ------------------------------------------------------------------------------------------


def test_cancel_before_and_during_a_job(rig):
    queued = create(rig, "z-image-basic", {"prompt": "A mug."})
    running = create(rig, "z-image-basic", {"prompt": "A lamp."})
    tools.cancel_job(queued, context=rig.worker.ctx)
    rig.fake.behaviors[running] = "hang"
    original_sleep = rig.worker.sleep

    def sleep_and_cancel(seconds):
        tools.cancel_job(running, context=rig.worker.ctx)
        original_sleep(seconds)

    rig.worker.sleep = sleep_and_cancel
    rig.worker.drain()
    for job_id in (queued, running):
        job = job_of(rig, job_id)
        assert job["status"] == "cancelled" and job["error"]["code"] == "CANCELLED"
        assert rig.storage.exists("jobs/failed/%s.json" % job_id)
    assert rig.storage.list("jobs/cancel") == []
    assert len(rig.fake.submitted) == 1  # the queued one never reached ComfyUI


def test_a_job_too_big_for_this_runtime_waits_instead_of_failing(make_rig):
    small = make_rig(gpu=GPU_L4)
    video = tools.create_job("minimax-h3-basic", {"prompt": COFFEE}, context=small.worker.ctx)
    small.worker.drain()
    child = job_of(small, video["job_id"])["depends_on"][0]
    assert job_of(small, child)["status"] == "completed"  # the L4 can make the first frame
    assert job_of(small, video["job_id"])["status"] == "pending"
    waiting = small.storage.read_json(WORKER_STATE_FILE)["waiting"][video["job_id"]]
    assert "at least 38 GiB" in waiting and "22.2" in waiting
    small.worker.shutdown("test")
    big = make_rig("vm2")
    big.worker.drain()
    assert job_of(big, video["job_id"])["status"] == "completed"


def test_a_cpu_runtime_runs_the_model_free_test_only(make_rig):
    cpu = make_rig(gpu=GPU_NONE)
    test_id = create(cpu, "test-generation", {})
    image_id = create(cpu, "z-image-basic", {"prompt": "A mug."})
    cpu.worker.drain()
    assert job_of(cpu, test_id)["status"] == "completed" and job_of(cpu, image_id)["status"] == "pending"
    assert cpu.env.process.extra_args == ["--cpu"]


def test_h3_restarts_comfyui_with_lowvram_on_a_40_gib_card(make_rig):
    gpu = {"available": True, "name": "Fake A100-40", "vram_gib": 39.6, "bf16": True, "torch": "2.9.0", "cuda": "12.6"}
    rig = make_rig(gpu=gpu)
    tools.create_job("minimax-h3-basic", {"prompt": COFFEE}, context=rig.worker.ctx)
    rig.worker.drain()
    assert rig.env.process.extra_args == ["--lowvram"] and rig.env.process.started == 1
    assert len([j for j in rig.worker.ctx.store.list() if j["status"] == "completed"]) == 2


def test_serve_takes_new_jobs_then_stops_after_the_idle_timeout(rig):
    first = create(rig, "test-generation", {})
    late = []
    original_sleep = rig.worker.sleep

    def sleep(seconds):
        if not late:
            late.append(tools.create_job("test-generation", {"width": 128}, root=rig.storage.root)["job_id"])
        original_sleep(seconds)

    rig.worker.sleep = sleep
    reason = rig.worker.serve(idle_timeout_seconds=30)
    assert reason == "idle_timeout"
    assert job_of(rig, first)["status"] == "completed" and job_of(rig, late[0])["status"] == "completed"
    summary = rig.worker.shutdown(reason, release=True)
    assert summary["jobs"] == 2 and rig.released == ["flush", "release"]
    state = rig.storage.read_json(WORKER_STATE_FILE)
    assert state["worker_state"] == "stopped" and state["serving"] is False
    assert json.loads(rig.storage.read_text("logs/sessions.jsonl").splitlines()[-1])["reason"] == "idle_timeout"


def test_the_notebook_serve_cell_releases_only_when_asked(rig, capsys):
    rig.worker.cfg["poll_interval_seconds"] = 60
    nb.serve(rig.session, idle_minutes=1, release_runtime=False)
    assert rig.released == [] and "runtime 還開著" in capsys.readouterr().out


# --- one runtime at a time, and what happens when one is lost -------------------------------------------


def test_a_second_runtime_is_refused_while_the_first_is_alive(make_rig):
    first, second = make_rig(), make_rig("vm2")
    first.worker.boot()
    with pytest.raises(WorkflowError) as exc:
        second.worker.boot()
    assert exc.value.code == "WORKER_ALREADY_RUNNING"
    first.worker.shutdown("done")
    second.worker.boot()  # a stopped worker no longer blocks


def test_a_job_left_running_by_a_lost_runtime_is_retried_once(make_rig):
    lost = make_rig()
    job_id = create(lost, "z-image-basic", {"prompt": "A mug."})
    lost.worker.boot()
    lost.worker.ctx.store.claim(job_of(lost, job_id), lost.worker.session)  # ...and the runtime vanishes
    lost.close()
    lost.storage.delete(WORKER_STATE_FILE)

    second = make_rig("vm2")
    second.worker.boot()
    assert job_of(second, job_id)["status"] == "pending" and job_of(second, job_id)["attempt"] == 1
    second.worker.ctx.store.claim(job_of(second, job_id), second.worker.session)  # lost a second time
    second.close()
    second.storage.delete(WORKER_STATE_FILE)

    third = make_rig("vm3")
    third.worker.boot()
    job = job_of(third, job_id)
    assert job["status"] == "failed" and job["error"]["code"] == "SESSION_LOST" and job["attempt"] == 2


# --- session restart: the headline guarantee --------------------------------------------------------------


def test_a_second_session_installs_and_downloads_nothing(make_rig):
    first = make_rig()
    video = tools.create_job("minimax-h3-basic", {"prompt": COFFEE}, context=first.worker.ctx)["job_id"]
    first.worker.drain()
    assert first.env.actions["comfyui"] == "installed" and first.env.actions["deps"] == "installed"
    assert first.env.actions["custom_node:ComfyUI-VideoHelperSuite"] == "installed"
    assert [v for k, v in first.env.actions.items() if k.startswith("model:")] == ["downloaded"] * 8
    assert len(first.installers.hf) == 8 and len(first.installers.urls) == 2 and len(first.installers.pip) == 1
    first.worker.shutdown("session over")
    storage = first.storage
    manifest = storage.read_json("models/manifest.json")
    assert len(manifest) == 8 and "loras/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors" in manifest
    assert storage.exists("models/z-image/vae/ae.safetensors")
    assert storage.exists("models/minimax-h3/vae/minimax_h3_video_vae_fp16.safetensors")
    assert sorted(n.split("-")[0] for n in storage.list("cache")) == ["comfyui", "node", "pydeps"]
    assert storage.read_json("configs/environment.json")["comfyui_commit"] == "2d6b73283af2447bdd065ece4090b8c6b1784544"

    second = make_rig("vm2")  # a brand-new runtime disk, the same workspace
    again = tools.create_job("minimax-h3-basic", {"prompt": COFFEE}, context=second.worker.ctx)["job_id"]
    second.worker.drain()
    assert second.installers.network_calls == 0
    assert second.env.actions["comfyui"] == "extracted" and second.env.actions["deps"] == "extracted"
    assert second.env.actions["custom_node:ComfyUI-VideoHelperSuite"] == "extracted"
    assert [v for k, v in second.env.actions.items() if k.startswith("model:")] == ["staged"] * 8
    store = second.worker.ctx.store
    assert store.get(video)["status"] == "completed" and store.get(again)["status"] == "completed"  # jobs survive
    assert storage.exists(store.get(video)["output"]["path"])  # and so do outputs
    assert len(storage.list("notebooks", ".ipynb")) == 5 and len(storage.list("outputs/videos")) == 2


def test_only_a_changed_model_is_fetched_again(make_rig, workspace):
    _, storage, blobs = workspace
    first = make_rig()
    create(first, "z-image-basic", {"prompt": "A mug."})
    first.worker.drain()
    first.worker.shutdown("done")
    (storage.root / "models" / "z-image" / "vae" / "ae.safetensors").write_bytes(b"truncated")
    second = make_rig("vm2")
    create(second, "z-image-basic", {"prompt": "A lamp."})
    second.worker.drain()
    assert second.installers.hf == ["split_files/vae/ae.safetensors"]
    assert second.env.actions["model:ae.safetensors"] == "downloaded"
    assert second.env.actions["model:qwen_3_4b.safetensors"] == "staged"
    second.worker.shutdown("done")
    assert storage.read_bytes("models/z-image/vae/ae.safetensors") == blobs["ae.safetensors"]


def test_a_download_with_the_wrong_checksum_fails_the_job_and_reaches_nothing(make_rig, workspace):
    _, storage, blobs = workspace
    rig = make_rig(blobs=dict(blobs, **{"ae.safetensors": b"tampered"}))
    job_id = create(rig, "z-image-basic", {"prompt": "A mug."})
    rig.worker.drain()
    assert job_of(rig, job_id)["error"]["code"] == "MODEL_CHECKSUM"
    rig.worker.shutdown("done")
    assert not storage.exists("models/z-image/vae/ae.safetensors")


# --- the notebook helpers ---------------------------------------------------------------------------------


def test_the_setup_and_debug_cells_report(rig, capsys):
    assert nb.setup(rig.session)["comfyui"] == "installed"
    assert nb.check_models(rig.session) is False  # nothing downloaded yet
    nb.start_comfyui(rig.session)
    assert nb.check_workflows(rig.session) is True
    assert nb.api_test(rig.session)["status"] == "completed"
    nb.list_jobs(rig.session)
    nb.describe(rig.session, "z-image-basic")
    nb.comfy_log(rig.session)
    out = capsys.readouterr().out
    for text in ("環境檢查", "首次安裝", "尚未下載", "ComfyUI 已啟動", "節點齊全", "ComfyUI API 測試", "Jobs："):
        assert text in out, text
    assert rig.env.actions["comfyui"] == "installed"  # a later "present" did not overwrite it


def test_the_form_reports_a_refused_job_instead_of_raising(rig, capsys):
    assert nb.generate(rig.session, "z-image-basic", {"prompt": ""}) is None
    assert nb.generate(rig.session, "z-image-basic", {"prompt": "x"}, {"cfg": 1.0}) is None
    assert nb.generate(rig.session, "z-image-basic", {"prompt": "x"}, extra="{not json") is None
    assert (
        nb.generate(
            rig.session,
            "minimax-h3-basic",
            {"prompt": "A cup."},
            inputs={"first_frame": nb.input_ref("file", path="inputs/a.png")},
        )
        is None
    )
    out = capsys.readouterr().out
    assert out.count("沒有建立 job") == 3 and "ATTESTATION_REQUIRED" in out and "不是合法的 JSON" in out
    assert rig.worker.ctx.store.list() == []


def test_the_form_ignores_fields_of_another_workflow_and_takes_extra_json(rig):
    result = nb.generate(
        rig.session, "test-generation", {"prompt": "ignored", "aspect": "1:1"}, {"steps": 8}, extra='{"width": 128}'
    )
    assert result["status"] == "completed" and result["output"]["width"] == 128
    custom = nb.generate(
        rig.session, "z-image-basic", {"prompt": "A mug.", "aspect": "1:1"}, {"width": 1536, "height": 640}
    )
    assert (custom["output"]["width"], custom["output"]["height"]) == (1536, 640)


def test_connect_explains_a_workspace_that_was_never_deployed(tmp_path):
    with pytest.raises(WorkflowError) as exc:
        nb.connect(tmp_path / "empty", mount=False)
    assert exc.value.code == "WORKSPACE_NOT_FOUND" and "deploy.py" in exc.value.hint


def test_a_worker_that_was_shut_down_can_be_used_again_in_the_same_kernel(rig):
    """serve() ends with a shutdown; running the generate cell afterwards must boot again, heartbeat included."""
    rig.worker.cfg["poll_interval_seconds"] = 60
    nb.serve(rig.session, idle_minutes=1, release_runtime=False)
    assert rig.storage.read_json(WORKER_STATE_FILE)["worker_state"] == "stopped"
    result = nb.generate(rig.session, "test-generation", {})
    assert result["status"] == "completed"
    assert rig.worker.state.thread.is_alive()
    assert rig.storage.read_json(WORKER_STATE_FILE)["worker_state"] == "ready"


def test_alternating_workflows_do_not_restart_comfyui_every_job(make_rig):
    gpu = {"available": True, "name": "Fake A100-40", "vram_gib": 39.6, "bf16": True, "torch": "2.9.0", "cuda": "12.6"}
    rig = make_rig(gpu=gpu)
    for _ in range(2):
        tools.create_job("minimax-h3-basic", {"prompt": COFFEE}, context=rig.worker.ctx)
    rig.worker.drain()  # image, video, image, video
    processes = [p for p in type(rig.env.process).instances if p.comfy_dir == rig.env.comfy_dir]
    assert [p.extra_args for p in processes] == [[], ["--lowvram"]]  # one restart, when H3 first needed it
    assert len(rig.fake.submitted) == 4
