"""training/concept_api.py - the OpenAPI tool server Open WebUI drives - plus gpu_guard and concept_cut, offline.

Pinned here: a chat model cannot spend GPU time or compute units without confirm=true; it cannot send an
unapproved scene to H3; H3 always runs with H3_MODEL_SOURCE=download and at most four clips per session; the
chat model is unloaded before the batch and before every image; the file route cannot leave its two folders;
a restart resumes nothing."""

import json
import os
import shutil
import subprocess

import concept_api as api
import concept_cut as cc
import concept_stills as cs
import gpu_guard
import pytest
import requests
from fastapi.testclient import TestClient

TOOLS = {
    "list_stories",
    "list_scenes",
    "generate_candidates",
    "review_scene",
    "set_video_prompt",
    "make_videos",
    "make_cut",
    "list_jobs",
    "get_job",
    "cancel_job",
    "consent_done",
}


class FakeGuard:
    GpuBusy = gpu_guard.GpuBusy

    def __init__(self, busy=None):
        self.calls, self.busy = [], busy

    def before_batch(self, **kw):
        self.calls.append(("batch", kw))
        if self.busy:
            raise gpu_guard.GpuBusy(self.busy)

    def before_image(self, **kw):
        self.calls.append(("image", kw))

    def wait_until_cool(self, max_c=None, **kw):
        self.calls.append(("cool", max_c))
        return {"waited_seconds": 30.0, "temp_c": 58.0}


class FakeThermalLog:
    def __init__(self, path):
        self.path, self.marks = path, []

    def start(self):
        return self

    def mark(self, event):
        self.marks.append(event)

    def stop(self):
        return {"samples": 3, "peak_c": 79.0, "mean_c": 66.0, "seconds_at_throttle": 0, "log": self.path}


class FakeComfy:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A scratch repo root: projects/ai_short_drama (specs only), output/, a job store, and a runner whose GPU,
    ComfyUI and subprocess pieces are fakes."""
    root = tmp_path / "projects" / "ai_short_drama"
    shutil.copytree(
        cs.PROJECT_ROOT,
        root,
        ignore=shutil.ignore_patterns(
            "*.png",
            "*.mp4",
            "videos",
            "review.json",
            "master.json",
            "candidate_*.json",
            "clips.json",
            "candidate_index.json",
            "*.html",
        ),
    )
    monkeypatch.setattr(api, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(api, "OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.delenv("CONCEPT_API_TOKEN", raising=False)
    store = api.JobStore(str(tmp_path / "jobs"))
    runner = api.Runner(store, root=str(root))
    made = []

    def submit(**kw):
        path = tmp_path / f"raw_{len(made)}.png"
        path.write_bytes(b"\x89PNG fake")
        made.append(kw)
        return str(path)

    runner.submit = submit
    runner.submitted = made
    runner.guard = FakeGuard()
    runner.thermal_log = FakeThermalLog
    runner.comfy = FakeComfy()
    runner.server_state = {"up": False}
    runner.server_up = lambda: runner.server_state["up"]

    def start():
        runner.server_state["up"] = True
        return runner.comfy

    runner.start_comfyui = start
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "runner", runner)
    return runner


@pytest.fixture
def web(env):
    return TestClient(api.app)


def approve_all(web, runner, story=1):
    assert web.post(f"/stories/{story}/candidates", json={"candidates": [1, 2], "confirm": True}).json()["queued"]
    assert runner.run_next()["status"] == api.COMPLETED
    for scene in range(1, 8):
        r = web.post(f"/stories/{story}/scenes/{scene}/review", json={"decision": "approve", "candidate": 2})
        assert r.status_code == 200, r.text


# --- surface ---------------------------------------------------------------------------------------------


def test_the_openapi_document_offers_exactly_these_tools(web):
    spec = web.get("/openapi.json").json()
    ids = {op["operationId"] for path in spec["paths"].values() for op in path.values()}
    assert ids == TOOLS  # the file route is not a tool
    assert "never choose or approve" in spec["info"]["description"]


def test_stories_and_scenes_are_listed_with_images_as_markdown(web, env):
    stories = web.get("/stories").json()["stories"]
    assert [s["story"] for s in stories] == [1, 2, 3, 4, 5] and stories[0]["images"] == 0
    web.post("/stories/1/candidates", json={"candidates": [1, 2], "scenes": [1], "confirm": True})
    env.run_next()
    scene = web.get("/stories/1/scenes").json()["scenes"][0]
    assert scene["status"] == cs.HUMAN_REVIEW and [c["candidate"] for c in scene["candidates"]] == [1, 2]
    assert scene["candidates"][0]["markdown"].startswith("![Scene 1 候選 1](http://localhost:7863/files/projects/")
    assert "23:46" in scene["text_to_verify"]
    assert web.get("/stories/9/scenes").status_code == 404


# --- spending needs a yes ---------------------------------------------------------------------------------


def test_generation_without_confirm_only_quotes_an_estimate(web, env):
    r = web.post("/stories/3/candidates", json={}).json()
    assert r["queued"] is False and r["needs_confirmation"] is True
    # the default is one image per scene: look first, then decide which scenes need a second
    assert r["estimate"]["images"] == 7
    assert web.post("/stories/3/candidates", json={"candidates": [1, 2]}).json()["estimate"]["images"] == 14
    assert web.post("/stories/3/candidates", json={"candidates": [5]}).status_code == 422
    assert api.store.all() == [] and env.run_next() is None and env.submitted == []


def test_confirmed_generation_unloads_the_chat_model_before_the_batch_and_every_image(web, env):
    job_id = web.post("/stories/2/candidates", json={"candidates": [1, 2], "scenes": [3], "confirm": True}).json()[
        "job_id"
    ]
    job = env.run_next()
    assert job["status"] == api.COMPLETED and job["result"]["images"] == 2 and len(env.submitted) == 2
    # per image: unload the chat model, then wait for the card to cool, then render
    assert [c[0] for c in env.guard.calls] == ["batch", "image", "cool", "image", "cool"]
    assert job["result"]["temperature"]["peak_c"] == 79.0
    assert job["result"]["temperature"]["cooldown_wait_seconds"] == 60
    assert env.guard.calls[0][1]["max_start_temp_c"] > 0
    assert env.comfy.terminated  # the ComfyUI this job started is stopped again
    assert web.get(f"/jobs/{job_id}").json()["progress"] == {"done": 2, "total": 2, "current": None}
    assert os.path.isfile(os.path.join(env.root, "review_sheet.html"))


def test_a_comfyui_someone_else_started_is_left_running(web, env):
    env.server_state["up"] = True
    web.post("/stories/2/candidates", json={"candidates": [1, 2], "scenes": [1], "confirm": True})
    assert env.run_next()["status"] == api.COMPLETED
    assert env.comfy.terminated is False


def test_a_busy_card_fails_the_job_without_starting_comfyui(web, env):
    env.guard = FakeGuard(busy="VRAM 已用 7000 MiB")
    web.post("/stories/2/candidates", json={"candidates": [1, 2], "scenes": [1], "confirm": True})
    job = env.run_next()
    assert job["status"] == api.FAILED and "7000" in job["error"]
    assert env.submitted == [] and env.server_state["up"] is False


def test_extra_candidates_need_the_users_regenerate_decision(web, env):
    web.post("/stories/1/candidates", json={"candidates": [1, 2], "scenes": [4], "confirm": True})
    env.run_next()
    r = web.post("/stories/1/candidates", json={"candidates": [3, 4], "scenes": [4], "confirm": True})
    assert r.status_code == 409 and "REGENERATE" in r.json()["detail"]
    web.post("/stories/1/scenes/4/review", json={"decision": "regenerate", "notes": "手機看不清楚"})
    assert (
        web.post("/stories/1/candidates", json={"candidates": [3, 4], "scenes": [4]}).json()["estimate"]["images"] == 2
    )


def test_only_one_job_at_a_time(web, env):
    assert web.post("/stories/1/candidates", json={"candidates": [1, 2], "confirm": True}).json()["queued"]
    assert web.post("/stories/2/candidates", json={"candidates": [1, 2], "confirm": True}).status_code == 409


# --- review -------------------------------------------------------------------------------------------------


def test_approval_needs_the_candidate_number_and_is_tagged_as_coming_from_the_chat(web, env):
    web.post("/stories/1/candidates", json={"candidates": [1, 2], "scenes": [2], "confirm": True})
    env.run_next()
    assert web.post("/stories/1/scenes/2/review", json={"decision": "approve"}).status_code == 422
    assert web.post("/stories/1/scenes/2/review", json={"decision": "pick the best"}).status_code == 422
    r = web.post("/stories/1/scenes/2/review", json={"decision": "approve", "candidate": 1, "notes": "表情對"})
    assert r.json() == {"story": 1, "scene": 2, "status": cs.APPROVED, "approved_candidate": "candidate_01.png"}
    story = cs.load_stories(env.root)[0]
    assert cs.load_review(story, 2)["decided_via"] == "open-webui"
    assert os.path.isfile(os.path.join(story.scene_dir(2), "master.png"))
    assert web.post("/stories/1/scenes/3/review", json={"decision": "approve", "candidate": 1}).status_code == 409


# --- H3 -----------------------------------------------------------------------------------------------------


class FakeH3:
    """Stands in for `run.py --manifest`: records the call, emits the consent line, reports every clip done."""

    def __init__(self, runner, tmp_output):
        self.runner, self.calls, self.statuses, self.out = runner, [], [], tmp_output

    def __call__(self, cmd, **kw):
        manifest = json.load(open(cmd[-1], encoding="utf-8"))
        self.calls.append({"cmd": cmd, "env": kw["env"], "manifest": manifest})
        fake = self

        class Proc:
            returncode = 0

            def wait(self):
                return 0

            @property
            def stderr(self):
                yield "t [CONNECTING_COLAB] DRIVE_CONSENT_NEEDED: approve Google Drive access\n"
                fake.statuses.append(next(j for j in api.store.all() if j["kind"] == "videos")["status"])
                yield "t [CONNECTING_COLAB] Google Drive mounted in 100 s\n"
                fake.statuses.append(next(j for j in api.store.all() if j["kind"] == "videos")["status"])

            @property
            def stdout(self):
                records = []
                for job in manifest["jobs"]:
                    os.makedirs(os.path.dirname(job["output"]), exist_ok=True)
                    open(job["output"], "wb").write(b"mp4")
                    records.append({"scene_id": job["scene_id"], "status": "COMPLETED", "output": job["output"]})
                yield json.dumps({"batch_id": manifest["batch_id"], "records": records, "cu_used_settled": 1.25}) + "\n"

        return Proc()


def ok_run(calls):
    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ready", stderr="")

    return run


def test_unapproved_scenes_and_scenes_without_a_shot_description_cannot_go_to_h3(web, env):
    assert web.post("/stories/1/videos", json={"confirm": True}).status_code == 409  # nothing approved yet
    web.post("/stories/1/candidates", json={"candidates": [1, 2], "scenes": [1, 2], "confirm": True})
    env.run_next()
    web.post("/stories/1/scenes/1/review", json={"decision": "approve", "candidate": 2})
    r = web.post("/stories/1/videos", json={"scenes": [1, 2], "confirm": True})
    assert r.status_code == 409 and "Scene 2 還沒有核准的 master" in r.json()["detail"]
    approve_all(web, env, story=2)  # story 2 has no video blocks
    r = web.post("/stories/2/videos", json={"confirm": True})
    assert r.status_code == 409 and "set_video_prompt" in r.json()["detail"]
    assert api.store.all()[0]["kind"] == "images"  # no video job was ever queued


def test_videos_quote_cost_first_then_run_in_sessions_of_four_with_downloaded_models(web, env, tmp_path):
    approve_all(web, env)
    quote = web.post("/stories/1/videos", json={}).json()
    assert quote["queued"] is False and quote["estimate"]["clips"] == 7 and quote["estimate"]["sessions"] == 2
    assert quote["estimate"]["compute_units"] > 0 and "Docker Desktop 要開著" in quote["estimate"]["user_actions"]
    checks = []
    env.run = ok_run(checks)
    env.popen = FakeH3(env, tmp_path)
    job_id = web.post("/stories/1/videos", json={"confirm": True}).json()["job_id"]
    job = env.run_next()
    assert job["status"] == api.COMPLETED, job["error"]
    assert checks[0][-1].endswith("check.py")  # preflight before any session
    calls = env.popen.calls
    assert [len(c["manifest"]["jobs"]) for c in calls] == [4, 3]
    assert all(c["env"]["H3_MODEL_SOURCE"] == "download" for c in calls)
    first = calls[0]["manifest"]["jobs"][0]
    assert first["input"]["image"].endswith(os.path.join("images", "scene_01", "master.png"))
    assert first["settings"]["duration"] == 5 and first["prompt"]["constraints"] == ["locked-camera"]
    assert env.popen.statuses[:2] == [api.WAITING, api.RUNNING]  # consent asked, then cleared by the mount line
    assert job["result"]["clips_completed"] == 7 and job["result"]["cu_used"] == 2.5
    story = cs.load_stories(env.root)[0]
    assert sorted(api.load_clips(story)) == [f"scene_0{i}" for i in range(1, 8)]
    assert web.get(f"/jobs/{job_id}").json()["needs_user_action"] is None


def test_a_failed_preflight_spends_nothing(web, env, tmp_path):
    approve_all(web, env)
    env.run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="[FAIL] transport:docker", stderr="")
    env.popen = FakeH3(env, tmp_path)
    web.post("/stories/1/videos", json={"confirm": True})
    job = env.run_next()
    assert job["status"] == api.FAILED and "transport:docker" in job["error"] and env.popen.calls == []


def test_consent_done_is_refused_unless_a_job_is_waiting_for_it(web, env):
    web.post("/stories/1/candidates", json={"candidates": [1, 2], "scenes": [1], "confirm": True})
    job_id = api.store.all()[0]["job_id"]
    assert web.post(f"/jobs/{job_id}/consent-done").status_code == 409
    calls = []
    env.run = ok_run(calls)
    api.store.save(dict(api.store.get(job_id), status=api.WAITING))
    assert web.post(f"/jobs/{job_id}/consent-done").status_code == 200 and calls[0][-1].endswith("consent_done.py")


def test_a_shot_description_can_be_saved_from_the_chat_and_is_validated(web, env):
    body = {"description": "The engineer in the first frame leans toward the monitor.", "duration": 6}
    assert web.put("/stories/2/scenes/1/video-prompt", json=body).json()["saved"] is True
    video = cs.load_stories(env.root)[1].scene(1)["video"]
    assert video["duration"] == 6 and video["constraints"] == [] and video["ambience"] is False
    assert web.put("/stories/2/scenes/1/video-prompt", json={"description": " ", "duration": 6}).status_code == 422
    assert web.put("/stories/2/scenes/1/video-prompt", json={"description": "x", "duration": 60}).status_code == 422


# --- cut ----------------------------------------------------------------------------------------------------


def test_the_cut_needs_every_scene_and_keeps_story_order_ambience_and_the_scripts_subtitle(web, env, tmp_path):
    assert web.post("/stories/1/cut", json={}).status_code == 409
    approve_all(web, env)
    env.run = ok_run([])
    env.popen = FakeH3(env, tmp_path)
    web.post("/stories/1/videos", json={"confirm": True})
    env.run_next()
    seen = {}

    def assemble(clips, out, *, subtitle=None):
        seen.update(clips=clips, out=out, subtitle=subtitle)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(b"mp4")
        return {"output": out, "final_loudness_lufs": -14.2}

    env.assemble = assemble
    web.post("/stories/1/cut", json={})
    job = env.run_next()
    assert job["status"] == api.COMPLETED and job["result"]["url"].startswith(
        "http://localhost:7863/files/projects/ai_short_drama/story_01_2347/videos/cut/"
    )
    assert [c["path"].replace("\\", "/").split("/")[-3:] for c in seen["clips"]] == [
        ["videos", f"scene_0{i}", "clip_v1.mp4"] for i in range(1, 8)
    ]
    assert os.path.basename(seen["out"]) == "story_01_2347_cut_v1.mp4"
    assert [c["ambience"] for c in seen["clips"]] == [False] * 6 + [True]
    assert seen["subtitle"] == "第十八次，開始。"
    web.post("/stories/1/cut", json={"subtitle": ""})
    env.run_next()
    assert seen["subtitle"] is None


# --- files, jobs, auth ----------------------------------------------------------------------------------


def test_the_file_route_serves_project_images_and_story_output_only(web, env, tmp_path):
    web.post("/stories/1/candidates", json={"candidates": [1, 2], "scenes": [1], "confirm": True})
    env.run_next()
    ok = "/files/projects/ai_short_drama/story_01_2347/images/scene_01/candidate_01.png"
    assert web.get(ok).status_code == 200
    (tmp_path / "output" / "story_x").mkdir(parents=True)
    (tmp_path / "output" / "story_x" / "a.mp4").write_bytes(b"x")
    (tmp_path / "output" / "logs").mkdir()
    (tmp_path / "output" / "logs" / "a.png").write_bytes(b"x")
    (tmp_path / "secret.png").write_bytes(b"x")
    assert web.get("/files/output/story_x/a.mp4").status_code == 200
    for bad in (
        "/files/output/logs/a.png",
        "/files/secret.png",
        "/files/projects/ai_short_drama/story_01_2347/story.json",
        "/files/projects/ai_short_drama/%2e%2e/%2e%2e/secret.png",
    ):
        assert web.get(bad).status_code == 404, bad


def test_a_restart_marks_unfinished_jobs_interrupted_and_resumes_nothing(web, env):
    job_id = web.post("/stories/1/candidates", json={"candidates": [1, 2], "confirm": True}).json()["job_id"]
    assert api.store.mark_interrupted() == [job_id]
    assert web.get(f"/jobs/{job_id}").json()["status"] == api.INTERRUPTED
    assert env.run_next()["status"] == api.INTERRUPTED and env.submitted == []  # the queued id is not run
    assert web.post("/stories/2/candidates", json={"candidates": [1, 2], "confirm": True}).json()[
        "queued"
    ]  # and nothing is blocked


def test_a_queued_job_can_be_cancelled_and_a_running_image_job_stops_after_the_current_image(web, env):
    job_id = web.post("/stories/1/candidates", json={"candidates": [1, 2], "confirm": True}).json()["job_id"]
    assert web.post(f"/jobs/{job_id}/cancel").json()["status"] == api.CANCELLED
    assert env.run_next()["status"] == api.CANCELLED and env.submitted == []
    job_id = web.post("/stories/1/candidates", json={"candidates": [1, 2], "confirm": True}).json()["job_id"]
    real = env.submit

    def submit_then_cancel(**kw):
        api.runner.cancel_requested.add(job_id)
        return real(**kw)

    env.submit = submit_then_cancel
    job = env.run_next()
    assert job["status"] == api.CANCELLED and job["result"]["images"] == 1
    assert web.get("/jobs/nope").status_code == 404 and web.get("/jobs").json()["jobs"][0]["job_id"] == job_id


def test_a_token_guards_the_tools_but_not_the_image_route(web, env, monkeypatch):
    monkeypatch.setenv("CONCEPT_API_TOKEN", "s3cret")
    assert web.get("/stories").status_code == 401
    assert web.get("/stories", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert web.get("/stories", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert web.get("/files/nope.png").status_code == 404  # reaches the route, not a 401


def test_a_network_bind_without_a_token_is_refused():
    api.check_bind("127.0.0.1", None)
    api.check_bind("0.0.0.0", "token")
    with pytest.raises(api.gc.UsageError):
        api.check_bind("0.0.0.0", None)


# --- gpu_guard ----------------------------------------------------------------------------------------------


class FakeOllama:
    def __init__(self, loaded, sticky=False, down=False):
        self.loaded, self.sticky, self.down, self.posts = list(loaded), sticky, down, []

    def get(self, url, timeout=0):
        if self.down:
            raise requests.exceptions.ConnectionError("refused")
        models = [{"name": n} for n in self.loaded]

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"models": models}

        return R()

    def post(self, url, json=None, timeout=0):
        self.posts.append(json)
        if not self.sticky:
            self.loaded.remove(json["model"])


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_loaded_chat_models_are_unloaded_with_keep_alive_zero():
    ollama = FakeOllama(["qwen3.5:9b"])
    assert gpu_guard.unload_models(ollama, sleep=lambda s: None) == ["qwen3.5:9b"]
    assert ollama.posts == [{"model": "qwen3.5:9b", "keep_alive": 0}] and gpu_guard.loaded_models(ollama) == []


def test_no_ollama_or_nothing_loaded_is_not_an_error():
    assert gpu_guard.unload_models(FakeOllama([], down=True)) == []
    assert gpu_guard.before_image(http=FakeOllama([]), log=lambda m: None) == []


def test_a_model_that_stays_loaded_stops_the_step_instead_of_rendering_into_a_full_card():
    clock = Clock()
    with pytest.raises(gpu_guard.GpuBusy, match="qwen3.5:9b"):
        gpu_guard.unload_models(FakeOllama(["qwen3.5:9b"], sticky=True), timeout=30, sleep=clock.sleep, clock=clock)
    assert clock.t >= 30


def smi(rows):
    rows = list(rows)

    def run(cmd, **kw):
        row = rows.pop(0) if len(rows) > 1 else rows[0]
        if row is None:
            raise OSError("no nvidia-smi")
        return subprocess.CompletedProcess(cmd, 0, stdout=row + "\n", stderr="")

    return run


def test_the_batch_waits_for_a_free_cool_card_and_gives_up_with_the_reason():
    clock = Clock()
    got = gpu_guard.before_batch(
        max_start_temp_c=55,
        http=FakeOllama([]),
        run=smi(["70, 7000, 8192", "60, 900, 8192", "50, 900, 8192"]),
        sleep=clock.sleep,
        clock=clock,
        log=lambda m: None,
    )
    assert got["gpu"] == {"temp_c": 50.0, "used_mib": 900.0, "total_mib": 8192.0} and clock.t > 0
    clock = Clock()
    with pytest.raises(gpu_guard.GpuBusy, match="7000 MiB"):
        gpu_guard.before_batch(
            max_start_temp_c=55,
            http=FakeOllama([]),
            run=smi(["50, 7000, 8192"]),
            sleep=clock.sleep,
            clock=clock,
            wait_seconds=20,
            log=lambda m: None,
        )
    assert (
        gpu_guard.before_batch(max_start_temp_c=55, http=FakeOllama([]), run=smi([None]), log=lambda m: None)["gpu"]
        is None
    )


# --- concept_cut --------------------------------------------------------------------------------------------


def test_clip_gains_level_speech_to_the_reference_and_keep_ambience_lower():
    assert cc.clip_gain_db(-14.6) == -8.4
    assert cc.clip_gain_db(-35.1, ambience=True) == 4.1  # to -31, not -23
    assert cc.clip_gain_db(None) == 0.0 and cc.clip_gain_db(-70.0) == 0.0  # silence is not amplified


def test_the_cut_command_joins_in_order_levels_and_shows_the_subtitle_at_the_end():
    cmd = cc.final_command(
        "ffmpeg",
        ["a.mp4", "b.mp4"],
        [-2.9, 4.1],
        -24.1,
        "out.mp4",
        subtitle_file="s.txt",
        subtitle_start=6.5,
        font="C:\\f.ttc",
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:a]volume=-2.90dB[a0];[1:a]volume=4.10dB[a1]" in graph
    assert "[0:v][a0][1:v][a1]concat=n=2:v=1:a=1" in graph
    assert "enable='gte(t,6.500)'" in graph and "textfile='s.txt'" in graph
    assert "volume=10.10dB,alimiter" in graph and "loudnorm" not in graph
    assert cmd[cmd.index("-map") + 1] == "[v]" and cmd[-1] == "out.mp4"
    plain = cc.final_command("ffmpeg", ["a.mp4"], [0.0], None, "out.mp4")
    assert "drawtext" not in " ".join(plain) and plain[plain.index("-map") + 1] == "[v0]"


def test_assemble_measures_each_clip_then_the_mix_and_refuses_a_missing_clip(tmp_path):
    clips = []
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(b"x")
        clips.append({"path": str(tmp_path / name), "ambience": name == "b.mp4"})
    logs = iter(
        [
            "Duration: 00:00:05.17,\nSummary:\n I: -20.1 LUFS",
            "Duration: 00:00:08.00,\nSummary:\n I: -35.1 LUFS",
            "Summary:\n I: -24.0 LUFS",
            "",
            "Summary:\n I: -14.2 LUFS",
        ]
    )
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=next(logs))

    out = str(tmp_path / "cut" / "x.mp4")
    result = cc.assemble(clips, out, subtitle="完", ffmpeg="ffmpeg", font="f.ttc", run=run)
    assert result["clip_gain_db"] == [-2.9, 4.1] and result["mix_loudness_lufs"] == -24.0
    assert result["final_loudness_lufs"] == -14.2 and result["seconds"] == 13.17
    assert result["subtitle_start"] == round(13.17 - cc.SUBTITLE_SECONDS, 2) and len(cmds) == 5
    assert open(os.path.splitext(out)[0] + ".subtitle.txt", encoding="utf-8").read() == "完"
    with pytest.raises(api.gc.UsageError, match="找不到片段"):
        cc.assemble([{"path": str(tmp_path / "nope.mp4")}], out, ffmpeg="ffmpeg", run=run)


def test_start_server_listens_on_the_port_of_comfyui_url(monkeypatch):
    """Windows can reserve 8188 after a restart (2026-10-04); COMFYUI_URL then moves both the client and the
    server it starts."""
    import comfyui_client as client

    seen = []
    monkeypatch.setattr(
        client.subprocess, "Popen", lambda cmd, **kw: seen.append(cmd) or type("P", (), {"poll": lambda s: None})()
    )
    monkeypatch.setattr(client, "is_server_running", lambda *a, **k: True)
    for url, port in (
        ("http://127.0.0.1:9188", "9188"),
        ("http://127.0.0.1:8188", "8188"),
        ("http://localhost", "8188"),
    ):
        monkeypatch.setattr(client, "COMFYUI_URL", url)
        client.start_server(extra_args=["--disable-smart-memory"])
        cmd = seen[-1]
        assert cmd[cmd.index("--port") + 1] == port and cmd[-1] == "--disable-smart-memory"


def test_a_hung_comfyui_is_restarted_and_the_batch_resumes_where_it_stopped(web, env):
    """2026-10-04: sampling stuck at step 0/8 three times (Model Initializing, ComfyUI's HTTP server silent).
    The job's own ComfyUI is restarted and finished images are not redone; the 20-minute library timeout is
    cut for the job and restored afterwards."""
    import comfyui_client as client

    real, calls, starts, limits = env.submit, {"n": 0}, {"n": 0}, []

    def flaky(**kw):
        calls["n"] += 1
        limits.append(client.POLL_TIMEOUT_SECONDS_ZIMAGE)
        if calls["n"] == 3:
            raise client.ComfyUIUnavailable("ComfyUI 伺服器連續 7 次沒有回應")
        return real(**kw)

    def start():
        starts["n"] += 1
        env.server_state["up"] = True
        return env.comfy

    env.submit, env.start_comfyui = flaky, start
    before = client.POLL_TIMEOUT_SECONDS_ZIMAGE
    web.post("/stories/2/candidates", json={"candidates": [1, 2], "scenes": [1, 2], "confirm": True})
    job = env.run_next()
    assert job["status"] == api.COMPLETED, job["error"]
    assert job["result"]["images"] == 4 and job["result"]["comfyui_restarts"] == 1 and starts["n"] == 2
    assert len({c["seed"] for c in env.submitted}) == len(env.submitted) == 4  # nothing rendered twice
    assert set(limits) == {api.IMAGE_STALL_SECONDS} and client.POLL_TIMEOUT_SECONDS_ZIMAGE == before
    assert [c[0] for c in env.guard.calls].count("batch") == 2  # the card is checked again before the restart


def test_a_comfyui_the_job_did_not_start_is_never_restarted_and_restarts_are_capped(web, env):
    import comfyui_client as client

    def hang(**kw):
        raise client.ComfyUIUnavailable("no answer")

    env.submit = hang
    env.server_state["up"] = True  # someone else's ComfyUI
    web.post("/stories/2/candidates", json={"candidates": [1, 2], "scenes": [1], "confirm": True})
    job = env.run_next()
    assert job["status"] == api.FAILED and "no answer" in job["error"] and env.comfy.terminated is False
    env.server_state["up"] = False  # now it is the job's own, but it never recovers
    web.post("/stories/2/candidates", json={"candidates": [1, 2], "scenes": [1], "confirm": True})
    job = env.run_next()
    assert job["status"] == api.FAILED and env.comfy.terminated is True


def test_the_first_batch_is_one_image_per_scene_and_that_is_enough_to_review(web, env):
    """The card throttles after about three images and a batch can die part-way, so the default batch covers
    every scene once. A scene with one candidate can be approved, or given a second one on request."""
    web.post("/stories/4/candidates", json={"confirm": True})
    job = env.run_next()
    assert job["status"] == api.COMPLETED and job["result"]["images"] == 7
    scenes = web.get("/stories/4/scenes").json()["scenes"]
    assert [len(s["candidates"]) for s in scenes] == [1] * 7 and {s["status"] for s in scenes} == {cs.HUMAN_REVIEW}
    assert web.post("/stories/4/scenes/1/review", json={"decision": "approve", "candidate": 1}).status_code == 200
    # the second batch skips the approved scene: it already has its master
    r = web.post("/stories/4/candidates", json={"candidates": [2], "scenes": [1, 2, 3]}).json()
    assert r["estimate"]["images"] == 2
    web.post("/stories/4/candidates", json={"candidates": [2], "scenes": [1, 2, 3], "confirm": True})
    env.run_next()
    scenes = web.get("/stories/4/scenes").json()["scenes"]
    assert [len(s["candidates"]) for s in scenes[:3]] == [1, 2, 2] and scenes[0]["status"] == cs.APPROVED


def test_stills_and_clips_live_apart_and_new_clips_never_overwrite_old_ones(web, env, tmp_path):
    """One story folder, two sides: images/scene_NN for stills and their records, videos/scene_NN for clips,
    videos/cut for cuts. A redone clip becomes clip_v2 next to clip_v1."""
    approve_all(web, env)
    env.run = ok_run([])
    env.popen = FakeH3(env, tmp_path)
    for _ in range(2):
        web.post("/stories/1/videos", json={"scenes": [5], "confirm": True})
        assert env.run_next()["status"] == api.COMPLETED
    story = cs.load_stories(env.root)[0]
    assert sorted(os.listdir(story.clip_dir(5))) == ["clip_v1.mp4", "clip_v2.mp4"]
    assert api.load_clips(story)["scene_05"]["path"].endswith("videos/scene_05/clip_v2.mp4")  # the cut uses the newest
    assert not [f for f in os.listdir(story.scene_dir(5)) if f.endswith(".mp4")]
    assert os.path.isfile(os.path.join(story.scene_dir(5), "master.png"))
    assert os.listdir(os.path.join(story.dir, "videos", "_batches"))  # the manifests that were sent


# --- keeping the card cool ----------------------------------------------------------------------------------


def test_an_image_waits_until_the_card_has_cooled_and_gives_up_if_it_never_does():
    """The owner would rather wait than run the card at its throttle point: every image starts at or below the
    limit. A card that never cools means something else is heating it, and that stops the batch."""
    clock = Clock()
    got = gpu_guard.wait_until_cool(
        60,
        run=smi(["81, 900, 8192", "70, 900, 8192", "60, 900, 8192"]),
        sleep=clock.sleep,
        clock=clock,
        log=lambda m: None,
    )
    assert got == {"waited_seconds": 2 * gpu_guard.SAMPLE_SECONDS, "temp_c": 60.0}
    clock = Clock()
    with pytest.raises(gpu_guard.GpuBusy, match="沒有降到 60"):
        gpu_guard.wait_until_cool(
            60, run=smi(["75, 900, 8192"]), sleep=clock.sleep, clock=clock, timeout=40, log=lambda m: None
        )
    # an unreadable nvidia-smi or a gate of 0 never blocks
    assert gpu_guard.wait_until_cool(60, run=smi([None]))["temp_c"] is None
    assert gpu_guard.wait_until_cool(0, run=smi(["90, 900, 8192"]))["waited_seconds"] == 0.0


def test_the_thermal_summary_counts_time_at_the_throttle_point():
    samples = [(0, 55.0), (5, 70.0), (10, 83.0), (15, 84.0), (20, None), (25, 62.0)]
    assert gpu_guard.summarize(samples) == {"samples": 5, "peak_c": 84.0, "mean_c": 70.8, "seconds_at_throttle": 10}
    assert gpu_guard.summarize([])["peak_c"] is None


def test_the_thermal_log_writes_a_curve_with_the_power_limit_and_image_marks(tmp_path):
    def run(cmd, **kw):
        out = "130.00" if "power.limit" in cmd[1] else "64, 1200, 8192"
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")

    log = gpu_guard.ThermalLog(str(tmp_path / "t" / "log.csv"), run=run, interval=0.05).start()
    log.mark("start scene_01 A")
    import time

    time.sleep(0.2)
    summary = log.stop()
    lines = open(summary["log"], encoding="utf-8").read().splitlines()
    assert lines[0].startswith("# power_limit_w=130.0") and lines[1] == "elapsed_s,temp_c,used_mib,event"
    assert any(line.endswith(",start scene_01 A") for line in lines)
    assert summary["peak_c"] == 64.0 and summary["samples"] >= 1 and summary["seconds_at_throttle"] == 0


def test_the_estimate_includes_the_cooldown_wait(web, env):
    r = web.post("/stories/4/candidates", json={}).json()
    assert r["estimate"]["minutes"] == round(7 * (api.SECONDS_PER_IMAGE + api.COOLDOWN_SECONDS_PER_IMAGE) / 60)
    assert "60°C" in r["estimate"]["gpu"]
