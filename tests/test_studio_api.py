"""training/studio_api.py - the one-page image-to-video studio - offline.

Pinned here: nothing spends GPU time or compute units without confirm=true; H3's first frame can only be a wall
picture whose sha256 still matches (no upload route); prompts are screened; pictures are cooled before each render;
the file route cannot leave STUDIO_ROOT; a restart resumes nothing."""

import json
import os
import subprocess

import concept_api as ca
import pytest
import studio_api as st
from fastapi.testclient import TestClient
from test_concept_api import FakeComfy, FakeGuard, FakeThermalLog


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "STUDIO_ROOT", str(tmp_path / "studio"))
    monkeypatch.setattr(st, "REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    store = ca.JobStore(str(tmp_path / "studio" / "jobs"))
    runner = st.StudioRunner(store)
    made = []

    def submit(**kw):
        path = tmp_path / f"raw_{len(made)}.png"
        path.write_bytes(b"\x89PNG fake" + bytes([len(made)]))
        made.append(kw)
        return str(path)

    runner.submit, runner.submitted = submit, made
    runner.guard, runner.thermal_log, runner.comfy = FakeGuard(), FakeThermalLog, FakeComfy()
    runner.server_up = lambda: False
    runner.start_comfyui = lambda: runner.comfy
    monkeypatch.setattr(st, "store", store)
    monkeypatch.setattr(st, "runner", runner)
    monkeypatch.setattr(st.translate_prompt, "translate_to_english", lambda t: "translated: " + t)
    return runner


@pytest.fixture
def web(env):
    return TestClient(st.app)


def make_wall_image(web, env, prompt="a woman in a cafe"):
    r = web.post("/api/images", json={"prompt": prompt, "count": 1, "confirm": True}).json()
    assert env.run_next()["status"] == ca.COMPLETED
    return r["image_ids"][0]


def test_pictures_without_confirm_only_quote_and_show_the_exact_prompt(web, env):
    r = web.post("/api/images", json={"prompt": "咖啡廳裡的女生", "count": 2}).json()
    assert r["queued"] is False and r["needs_confirmation"] is True
    assert r["english_prompt"] == "translated: 咖啡廳裡的女生" and r["estimate"]["images"] == 2
    assert st.store.all() == [] and env.run_next() is None and env.submitted == []


def test_a_confirmed_local_picture_is_cooled_first_and_lands_on_the_wall(web, env):
    r = web.post("/api/images", json={"prompt": "a woman in a cafe", "count": 2, "confirm": True}).json()
    job = env.run_next()
    assert job["status"] == ca.COMPLETED and len(env.submitted) == 2
    assert [c[0] for c in env.guard.calls] == ["batch", "image", "cool", "image", "cool"]
    assert env.submitted[1]["seed"] == env.submitted[0]["seed"] + 1
    assert "child" in env.submitted[0]["negative_prompt"] or env.submitted[0]["negative_prompt"]  # safety negatives
    assert env.comfy.terminated
    items = web.get("/api/library?kind=image").json()["items"]
    assert {i["image_id"] for i in items} == set(r["image_ids"])
    assert web.get(items[0]["url"]).status_code == 200


def test_prompts_are_screened_for_pictures_and_clips(web, env):
    assert web.post("/api/images", json={"prompt": "a young girl, 15 years old"}).status_code == 422
    image_id = make_wall_image(web, env)
    clip = {"image_id": image_id, "description": "a schoolgirl smiles"}
    assert web.post("/api/videos", json={"clips": [clip]}).status_code == 422


def test_videos_only_take_wall_pictures_whose_hash_still_matches(web, env, tmp_path):
    clip = {"image_id": "img-nope", "description": "she smiles at the camera"}
    assert web.post("/api/videos", json={"clips": [clip]}).status_code == 404
    assert web.post("/api/videos", json={"clips": [dict(clip, image_id="../../etc/passwd")]}).status_code == 422
    image_id = make_wall_image(web, env)
    clip["image_id"] = image_id
    with open(os.path.join(st.images_dir(), image_id + ".png"), "ab") as f:
        f.write(b"swapped in by hand")
    assert web.post("/api/videos", json={"clips": [clip]}).status_code == 409
    assert not [r for r in web.app.routes if "upload" in getattr(r, "path", "")]


def test_videos_without_confirm_quote_the_cost_and_queue_nothing(web, env):
    image_id = make_wall_image(web, env)
    clip = {"image_id": image_id, "description": "she smiles at the camera", "duration": 8}
    r = web.post("/api/videos", json={"clips": [clip]}).json()
    assert r["needs_confirmation"] and r["estimate"]["compute_units"] > 0 and r["estimate"]["sessions"] == 1
    assert st.store.all()[0]["kind"] == "simg" and len(st.store.all()) == 1
    assert web.post("/api/videos", json={"clips": [dict(clip, duration=7)]}).status_code == 422


def test_confirmed_videos_run_h3_with_the_wall_picture_and_record_the_approval(web, env, tmp_path):
    image_id = make_wall_image(web, env)
    clip = {"image_id": image_id, "description": "she smiles", "duration": 5, "locked_camera": True}
    r = web.post("/api/videos", json={"clips": [clip], "confirm": True}).json()
    assert r["queued"]
    video_id = r["video_ids"][0]
    seen = {}

    def fake_session(job_id, manifest, e):
        seen["manifest"] = json.load(open(manifest, encoding="utf-8"))
        assert e["H3_MODEL_SOURCE"] == "download"
        job = seen["manifest"]["jobs"][0]
        open(job["output"], "wb").write(b"mp4")
        return {
            "cu_used_settled": 0.9,
            "records": [{"scene_id": video_id, "status": "COMPLETED", "output": job["output"], "job_id": "h3-1"}],
        }

    env._h3_session = fake_session
    env.run = lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
    job = env.run_next()
    assert job["status"] == ca.COMPLETED and job["result"]["clips_completed"] == 1 and job["result"]["cu_used"] == 0.9
    h3_job = seen["manifest"]["jobs"][0]
    assert h3_job["input"]["image"].endswith(image_id + ".png") and h3_job["prompt"]["constraints"] == ["locked-camera"]
    meta = json.load(open(os.path.join(st.images_dir(), image_id + ".json"), encoding="utf-8"))
    assert meta["approved_at"]
    video = web.get("/api/library?kind=video").json()["items"][0]
    assert video["image_id"] == image_id and web.get(video["url"]).status_code == 200


def test_only_one_job_at_a_time(web, env):
    assert web.post("/api/images", json={"prompt": "a cafe", "confirm": True}).status_code == 200
    assert web.post("/api/images", json={"prompt": "a park", "confirm": True}).status_code == 409


def test_the_file_route_stays_inside_the_studio_folder(web, env, tmp_path):
    (tmp_path / "secret.png").write_bytes(b"x")
    os.makedirs(st.STUDIO_ROOT, exist_ok=True)
    assert web.get("/files/../secret.png").status_code == 404
    assert web.get("/files/jobs/anything.json").status_code == 404


def test_a_restart_marks_unfinished_jobs_interrupted(web, env):
    web.post("/api/images", json={"prompt": "a cafe", "confirm": True})
    assert st.store.mark_interrupted()
    assert st.store.all()[0]["status"] == ca.INTERRUPTED


def test_a_public_bind_needs_a_token():
    with pytest.raises(st.gc.UsageError):
        st.check_bind("0.0.0.0", None)
    st.check_bind("127.0.0.1", None)
    st.check_bind("0.0.0.0", "t")


def test_the_page_is_served(web):
    r = web.get("/")
    assert r.status_code == 200 and "圖片轉影片工作室" in r.text
