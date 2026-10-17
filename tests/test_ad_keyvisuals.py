"""The ad key visuals catalog (training/ad_keyvisuals.py): 308 pictures of the app's seven characters aged 21 and over in
the 16 date outings. Pinned: the counts and spread, only adults of 21+ (and the four younger characters left out), the
names match ai-companion's, no minor-sounding words, no readable lettering, every entry is a job the Colab skill accepts
(age-safety negatives and the cfg floor applied), and import / measure / picks behave."""

import json
import re
import sys
from pathlib import Path

import ad_keyvisuals as kv
import asset_library as al
import pytest

import generate_character as gc

SCRIPTS = Path(kv.REPO_ROOT) / ".claude" / "skills" / "z-image-colab" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import zimage_colab as z  # noqa: E402

SEED_FILE = Path("D:/NSFW/ai-companion/backend/scripts/seed_characters.py")


@pytest.fixture(scope="module")
def catalog():
    return kv.build_catalog()


def test_308_pictures_seven_characters_sixteen_outings(catalog):
    assert len(catalog) == 308
    assert len(kv.CHARACTERS) == 7 and len(kv.OUTINGS) == 16
    assert len({e["id"] for e in catalog}) == 308 and len({e["seed"] for e in catalog}) == 308
    d = kv.distribution(catalog)
    assert d["per_composition"] == {"a": 154, "b": 154}
    assert set(d["per_character"].values()) == {44}


def test_every_character_has_every_outing_in_both_compositions(catalog):
    for key in kv.CHARACTERS:
        for outing in kv.OUTINGS:
            letters = {e["letter"] for e in catalog if e["character"] == key and e["outing"] == outing}
            want = {"a", "b", "c", "d"} if outing in kv.PRIORITY else {"a", "b"}
            assert letters == want, (key, outing)


def test_the_six_priority_outings_are_the_ones_in_the_plan():
    assert set(kv.PRIORITY) == {"cooking", "coffee", "night_view", "picnic", "travel", "cat_cafe"}
    assert set(kv.PRIORITY) <= set(kv.OUTINGS)


def test_only_characters_aged_21_and_over_are_used(catalog):
    for key in kv.CHARACTERS:
        assert gc.CHARACTERS[key]["age"] >= 21, key
    assert all(e["age"] >= 21 for e in catalog)
    for key, age in kv.EXCLUDED.items():
        assert key not in kv.CHARACTERS and gc.CHARACTERS[key]["age"] == age and age <= 20


def test_four_women_three_men_and_the_age_words_match_the_age(catalog):
    people = {e["character"]: e for e in catalog}
    assert sorted(p["gender"] for p in people.values()).count("woman") == 4
    assert sorted(p["gender"] for p in people.values()).count("man") == 3
    for key in kv.CHARACTERS:
        age = gc.CHARACTERS[key]["age"]
        text = kv.look_for(key)
        word = "early twenties" if age <= 23 else "mid-twenties" if age <= 26 else "late twenties"
        assert word in text and not re.search(r"\d", text), (key, text)
        assert "youthful fresh face" in text and "mature" not in text


def test_a_character_under_21_is_refused(monkeypatch):
    monkeypatch.setitem(kv.CHARACTERS, "mei", "小美")
    with pytest.raises(gc.UsageError):
        kv.build_catalog()
    with pytest.raises(gc.UsageError):
        kv.look_for("wanling")


@pytest.mark.skipif(not SEED_FILE.is_file(), reason="ai-companion is not checked out here")
def test_the_chinese_names_are_the_ones_in_ai_companion():
    seed = SEED_FILE.read_text(encoding="utf-8")
    for name in kv.CHARACTERS.values():
        assert f'"display_name": "{name}"' in seed, name


def test_no_prompt_has_a_minor_word_or_lettering(catalog):
    for e in catalog:
        low = e["text"].lower()
        for word in al.FORBIDDEN_WORDS:
            assert not re.search(rf"\b{re.escape(word)}\b", low), (e["id"], word)
        assert not any(w in low for w in ("logo", "brand", "slogan", "signboard", "billboard")), e["id"]
        assert "readable text" in e["negative"] and "outstretched arm" in e["negative"]
        assert e["solo"] is True and "two people" in e["negative"]


def test_clothes_cover_the_shoulders_and_chest():
    for key, outing in kv.OUTINGS.items():
        for gender in ("women", "men"):
            outfit = outing[gender].lower()
            assert "{c}" in outfit and not any(w in outfit for w in ("strapless", "tank", "bikini", "swim", "crop")), (
                key
            )


def test_the_outings_that_failed_before_are_written_so_they_cannot_repeat():
    assert "screen out of frame" in kv.OUTINGS["movie"]["moment"]
    assert "movie screen" in kv.OUTINGS["movie"]["negative"]
    assert "television out of frame" in kv.OUTINGS["games"]["moment"]
    assert "crowd" in kv.OUTINGS["concert"]["negative"] and "crowd" in kv.OUTINGS["night_market"]["negative"]
    assert "no writing" in kv.OUTINGS["night_market"]["moment"]
    assert "plain paper popcorn bucket" in kv.OUTINGS["movie"]["moment"]
    for gender, pronoun in (("woman", "her"), ("man", "his")):
        assert f"close to {pronoun} chest" in kv.moment_text(kv.OUTINGS["amusement_park"], gender)


def test_the_pilot_is_28_pictures_one_per_character_and_pilot_outing(catalog):
    ids = kv.pilot_ids(catalog)
    assert len(ids) == 28 and len(set(ids)) == 28
    by_id = {e["id"]: e for e in catalog}
    picked = [by_id[i] for i in ids]
    assert {e["character"] for e in picked} == set(kv.CHARACTERS)
    assert {e["outing"] for e in picked} == set(kv.PILOT_OUTINGS)
    assert {e["letter"] for e in picked} == {"a"}


def test_every_entry_is_a_job_the_skill_accepts_with_the_safety_terms(tmp_path, catalog):
    bible = kv.cs.load_bible(kv.BIBLE_ROOT)
    config = z.load_config(tmp_path / "none.json", env={"ZIMG_OUTPUT_DIR": str(tmp_path / "out")})
    for entry in catalog:
        job = z.build_job(config, **kv.job_spec(bible, entry))
        assert job["width"] == 720 and job["height"] == 1280 and job["tag"] == entry["id"]
        assert all(t in job["negative_prompt"] for t in ("child", "minor", "underage", "teen"))
        assert job["cfg"] >= 1.5


def test_export_writes_the_manifest_in_the_skills_format(tmp_path):
    out = kv.export("pilot", directory=str(tmp_path))
    data = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
    assert out["images"] == 28 and data["batch"] == "keyvisuals-pilot" and len(data["jobs"]) == 28
    assert {"prompt", "negative", "width", "height", "steps", "cfg", "seed", "solo", "tag"} <= set(data["jobs"][0])
    with pytest.raises(gc.UsageError):
        kv.export("all", ids=["nope"], directory=str(tmp_path))
    assert kv.export("all", directory=str(tmp_path))["images"] == 308


def test_a_redone_picture_gets_a_new_seed(monkeypatch):
    before = {e["id"]: e["seed"] for e in kv.build_catalog()}
    monkeypatch.setitem(kv.REVISIONS, "kv-ruoxi-cooking-a", 1)
    after = {e["id"]: e["seed"] for e in kv.build_catalog()}
    assert after["kv-ruoxi-cooking-a"] == before["kv-ruoxi-cooking-a"] + kv.REVISION_SEED_OFFSET
    assert all(after[i] == before[i] for i in before if i != "kv-ruoxi-cooking-a")


def _fake_jobs(tmp_path, tags):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG fake")
    for n, tag in enumerate(tags):
        record = {
            "job_id": f"j{n}",
            "tag": tag,
            "status": "COMPLETED",
            "output": str(png),
            "prompt": "p",
            "completed_at": f"2026-10-10T0{n}",
            "gpu": "L4",
        }
        (jobs / f"{n}.json").write_text(json.dumps(record))
    return jobs


def test_import_files_the_pictures_by_outing_and_builds_the_index(tmp_path):
    jobs = _fake_jobs(tmp_path, ["kv-ruoxi-cooking-a", "kv-taeoh-night_view-a"])
    images = tmp_path / "lib" / "images"
    report = kv.import_jobs(jobs_dir=str(jobs), images_dir=str(images))
    assert sorted(report["imported"]) == ["kv-ruoxi-cooking-a", "kv-taeoh-night_view-a"]
    assert (images / "cooking" / "kv-ruoxi-cooking-a.png").is_file()
    index = json.loads((tmp_path / "lib" / "index.json").read_text(encoding="utf-8"))
    assert index["count"] == 2
    html = (tmp_path / "lib" / "index.html").read_text(encoding="utf-8")
    assert "kv-picks" in html and "複製選取清單" in html and "kv-ruoxi-cooking-a" in html
    again = kv.import_jobs(jobs_dir=str(jobs), images_dir=str(images))
    assert again["imported"] == [] and len(again["skipped"]) == 2


def test_picks_copies_the_chosen_pictures_and_refuses_typos(tmp_path):
    jobs = _fake_jobs(tmp_path, ["kv-ruoxi-cooking-a", "kv-xinyi-cooking-a"])
    images, picked = tmp_path / "lib" / "images", tmp_path / "picked"
    kv.import_jobs(jobs_dir=str(jobs), images_dir=str(images))
    out = kv.picks(["kv-ruoxi-cooking-a", "kv-ruoxi-cooking-a"], images_dir=str(images), picked_dir=str(picked))
    assert out["added"] == ["kv-ruoxi-cooking-a"] and (picked / "kv-ruoxi-cooking-a.png").is_file()
    again = kv.picks(["kv-ruoxi-cooking-a", "kv-xinyi-cooking-a"], images_dir=str(images), picked_dir=str(picked))
    assert again["added"] == ["kv-xinyi-cooking-a"] and again["picked"] == 2
    with pytest.raises(gc.UsageError):
        kv.picks(["kv-nobody-cooking-a"], images_dir=str(images), picked_dir=str(picked))
    with pytest.raises(gc.UsageError):
        kv.picks(["kv-jungi-cooking-a"], images_dir=str(images), picked_dir=str(picked))  # not imported


def test_measure_flags_young_estimates_second_faces_and_missing_faces(tmp_path):
    images = tmp_path / "images"
    for e in kv.build_catalog():
        if e["id"] in ("kv-ruoxi-cooking-a", "kv-ruoxi-coffee-a", "kv-ruoxi-picnic-a", "kv-ruoxi-travel-a"):
            path = images / e["outing"] / f"{e['id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def describe(path):
        if "cooking" in path:
            return {"age": 19, "embedding": [1.0, 0.0], "faces": 1}
        if "coffee" in path:
            return {"age": 25, "embedding": [0.6, 0.8], "faces": 2}
        if "picnic" in path:
            return None
        return {"age": 27, "embedding": [0.0, 1.0], "faces": 1}

    report = kv.measure(describe, images_dir=str(images))
    assert report["age_by_character"]["ruoxi"] == {"n": 3, "mean": 23.7, "min": 19, "max": 27}
    assert report["estimated_age_under_20"] == ["kv-ruoxi-cooking-a"]
    assert report["second_face"] == ["kv-ruoxi-coffee-a"] and report["no_face"] == ["kv-ruoxi-picnic-a"]
    assert report["same_character_similarity"]["ruoxi"]["pairs"] == 3


def test_catalog_markdown_lists_every_character_and_outing(catalog):
    text = kv.catalog_markdown(catalog)
    assert "308 張" in text
    for zh in kv.CHARACTERS.values():
        assert zh in text
    for outing in kv.OUTINGS.values():
        assert outing["zh"] in text
    assert "小美" not in text.split("不用的角色")[0]


def test_the_fixes_from_the_pilot_are_in_every_prompt(catalog):
    # 2026-10-10 pilot: the 28-year-old read 40-50 because of "mature adult features"; picnic skirts rode up when
    # hugging the knees; most faces were blank.
    for e in catalog:
        assert "mature" not in e["text"], e["id"]
        assert "soft warm smile" in e["text"], e["id"]
    picnic_women = [e for e in catalog if e["outing"] == "picnic" and e["gender"] == "woman"]
    assert picnic_women and all("trousers" in e["outfit"] and "skirt" not in e["outfit"] for e in picnic_women)


def test_twelve_new_faces_give_192_more_pictures_without_moving_the_308_seeds(catalog):
    extra = kv.build_extra_catalog()
    assert len(kv.EXTRA_FACES) == 12 and len(extra) == 192 and len(kv.build_all()) == 500
    assert len({e["id"] for e in kv.build_all()}) == 500 and len({e["seed"] for e in kv.build_all()}) == 500
    assert min(e["seed"] for e in extra) > max(e["seed"] for e in catalog)
    assert sum(1 for f in kv.EXTRA_FACES.values() if f[1] == "woman") == 6


def test_the_new_faces_are_adults_of_21_and_over_and_all_look_different():
    forbidden = r"(?:" + "|".join(al.FORBIDDEN_WORDS) + r")"
    for key, face in kv.EXTRA_FACES.items():
        assert face[2] >= 21, key
        assert not re.search(forbidden, face[4].lower()), key
    looks = [f[4] for f in kv.EXTRA_FACES.values()]
    assert len(set(looks)) == 12
    for c in gc.CHARACTERS.values():
        assert c["appearance"] not in looks
    with pytest.raises(gc.UsageError):
        kv.describe_person(20, "woman", "Taiwanese", "x")


def test_every_new_face_entry_is_a_job_the_skill_accepts(tmp_path):
    bible = kv.cs.load_bible(kv.BIBLE_ROOT)
    config = z.load_config(tmp_path / "none.json", env={"ZIMG_OUTPUT_DIR": str(tmp_path / "out")})
    for entry in kv.build_extra_catalog():
        job = z.build_job(config, **kv.job_spec(bible, entry))
        assert job["width"] == 720 and job["height"] == 1280 and job["tag"] == entry["id"]
        assert all(t in job["negative_prompt"] for t in ("child", "minor", "underage", "teen"))
        assert "soft warm smile" in entry["text"] and "mature" not in entry["text"]


def test_export_faces_lists_only_the_new_faces(tmp_path):
    out = kv.export("faces", directory=str(tmp_path))
    data = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
    assert out["images"] == 192 and data["batch"] == "keyvisuals-faces"
    assert {j["tag"].split("-")[1] for j in data["jobs"]} == set(kv.EXTRA_FACES)


def test_every_outing_has_a_six_shot_script_and_a_visual_shot():
    boards = kv.parse_storyboards()
    assert set(boards) == set(kv.OUTINGS) and set(kv.VISUAL_SHOT) == set(kv.OUTINGS)
    for outing_id, b in boards.items():
        assert [sh["n"] for sh in b["shots"]] == [1, 2, 3, 4, 5, 6], outing_id
        assert kv.VISUAL_SHOT[outing_id] in range(1, 7) and all(sh["line"] for sh in b["shots"])
    assert "貓咪咖啡廳" in boards["cat_cafe"]["title"] and "夜景" in boards["night_view"]["title"]


def test_select_page_has_a_section_per_outing_with_its_candidates_and_the_script(tmp_path):
    boards = kv.parse_storyboards()
    index = [
        {"id": "kv-ruoxi-cooking-a", "outing": "cooking", "image": "cooking/kv-ruoxi-cooking-a.png", "summary_zh": "x",
         "name_zh": "若曦", "age": 23, "letter": "a"},
    ]  # fmt: skip
    page = kv.select_html(index, boards)
    assert page.count("<section") == 16 and "data-id='kv-ruoxi-cooking-a'" in page
    assert "快！關小火！" in page and "kv-select" in page and "複製選取清單" in page
    text = kv.scripts_markdown(boards, {"cooking": 1})
    assert "◀ 主視覺" in text and text.count("候選") >= 16


def test_picks_accepts_the_outing_grouped_choice_from_the_select_page(tmp_path):
    jobs = _fake_jobs(tmp_path, ["kv-ruoxi-cooking-a", "kv-xinyi-coffee-a"])
    images, picked = tmp_path / "lib" / "images", tmp_path / "picked"
    kv.import_jobs(jobs_dir=str(jobs), images_dir=str(images))
    out = kv.picks(
        {"cooking": ["kv-ruoxi-cooking-a"], "coffee": ["kv-xinyi-coffee-a"]},
        images_dir=str(images),
        picked_dir=str(picked),
    )
    assert out["picked"] == 2 and (picked / "kv-xinyi-coffee-a.png").is_file()


def test_qwen_request_keeps_the_safety_terms_and_the_cfg_floor():
    bible = kv.cs.load_bible(kv.BIBLE_ROOT)
    entry = kv.build_catalog()[0]
    req = kv.qwen_request(bible, entry)
    assert req["cfg"] >= 1.5 and req["seed"] == entry["seed"] and (req["width"], req["height"]) == (720, 1280)
    assert all(t in req["negative_prompt"] for t in ("child", "minor", "underage", "teen"))
    assert "outstretched arm" in req["negative_prompt"] and req["model"] == "qwen_image_2_1"


def test_qwen_generate_files_pictures_apart_marks_the_licence_and_skips_done(tmp_path):
    png = tmp_path / "raw.png"
    calls = []

    def submit(**kw):
        calls.append(kw)
        out = tmp_path / f"{len(calls)}.png"
        out.write_bytes(b"\x89PNG fake")
        return str(out)

    seen = []
    ids = kv.pilot_ids()[:2]
    out = kv.qwen_generate(
        ids, directory=str(tmp_path / "q"), submit=submit, server_up=lambda: True, before_each=seen.append
    )
    assert out == {"made": ids, "skipped": 0} and seen == ids and png.exists() is False
    first = next(e for e in kv.build_all() if e["id"] == ids[0])
    meta = json.loads((tmp_path / "q" / "images" / first["outing"] / f"{ids[0]}.json").read_text(encoding="utf-8"))
    assert "non-commercial" in meta["license"] and "internal" in meta["note"] and meta["cfg"] >= 1.5
    assert kv.QWEN_DIR != kv.IMAGES_DIR and not kv.QWEN_DIR.startswith(kv.PICKED_DIR)
    again = kv.qwen_generate(ids, directory=str(tmp_path / "q"), submit=submit, server_up=lambda: True)
    assert again == {"made": [], "skipped": 2} and len(calls) == 2
    with pytest.raises(gc.UsageError):
        kv.qwen_generate(ids, directory=str(tmp_path / "q2"), submit=submit, server_up=lambda: False)
    with pytest.raises(gc.UsageError):
        kv.qwen_generate(["nope"], directory=str(tmp_path / "q3"), submit=submit, server_up=lambda: True)
