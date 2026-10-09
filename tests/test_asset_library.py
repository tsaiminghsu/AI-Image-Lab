"""The asset library catalog (training/asset_library.py): 212 pictures of adults of every age band, both genders, seven
regions and four eras, plus empty places. Pinned: the counts and spread, only adults and no minor-sounding words, no
readable lettering or real names, every entry is a job the Colab skill accepts (so the age-safety negatives and the cfg
floor are applied), and importing files the finished pictures."""

import json
import re
import sys
from pathlib import Path

import asset_library as al
import pytest

import generate_character as gc

SCRIPTS = Path(al.REPO_ROOT) / ".claude" / "skills" / "z-image-colab" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import zimage_colab as z  # noqa: E402


@pytest.fixture(scope="module")
def catalog():
    return al.build_catalog()


def test_212_pictures_in_three_kinds(catalog):
    assert len(catalog) == 212
    kinds = {k: sum(1 for e in catalog if e["kind"] == k) for k in ("portrait", "scene", "place")}
    assert kinds == {"portrait": 64, "scene": 64, "place": 84}
    assert len({e["id"] for e in catalog}) == 212 and len({e["seed"] for e in catalog}) == 212


def test_every_era_gender_and_age_band_has_two_people_from_different_regions(catalog):
    people = [e for e in catalog if e["kind"] == "portrait"]
    for era in al.ERAS:
        for gender in al.GENDERS:
            for age in al.AGES:
                group = [p for p in people if (p["era"], p["gender"], p["age"]) == (era, gender, age)]
                assert len(group) == 2 and group[0]["region"] != group[1]["region"], (era, gender, age)


def test_regions_are_spread_evenly_and_every_one_appears_in_every_era(catalog):
    people = [e for e in catalog if e["kind"] == "portrait"]
    counts = [sum(1 for p in people if p["region"] == r) for r in al.REGIONS]
    assert max(counts) - min(counts) <= 2 and sum(counts) == 64
    for era in al.ERAS:
        assert {p["region"] for p in people if p["era"] == era} == set(al.REGIONS)


def test_the_same_person_has_a_portrait_and_a_scene_in_the_same_clothes(catalog):
    by_id = {e["id"]: e for e in catalog}
    for e in catalog:
        if e["kind"] == "portrait":
            scene = by_id["scene-" + e["person"]]
            assert scene["outfit"] == e["outfit"] and scene["look"] == e["look"]
            assert e["outfit"] in e["text"] and scene["venue"] in scene["text"]


def test_clothes_are_varied_and_follow_the_era(catalog):
    people = [e for e in catalog if e["kind"] == "portrait"]
    assert len({p["outfit"] for p in people}) >= 45
    for p in people:
        if p["era"] == "future":
            assert "near-future" in p["outfit"]
        if p["era"] == "retro":
            assert re.search(r"19[2-9]0s", p["outfit"]), p["outfit"]


def test_everyone_is_an_adult_and_no_prompt_has_a_minor_word(catalog):
    for e in catalog:
        for word in al.FORBIDDEN_WORDS:
            assert not re.search(rf"\b{re.escape(word)}\b", e["text"].lower()), (e["id"], word)
        if e["kind"] != "place":
            assert re.search(r"mid-twenties|mid-thirties|early fifties|seventies", e["look"]), e["id"]


def test_no_readable_lettering_or_brands(catalog):
    for e in catalog:
        assert not any(w in e["text"].lower() for w in ("logo", "brand", "slogan", "signboard", "billboard")), e["id"]
    assert all("text" in e["negative"] for e in catalog if e["kind"] == "place")


def test_places_have_nobody_and_people_are_solo(catalog):
    for e in catalog:
        if e["kind"] == "place":
            assert "no people" in e["text"] and e["solo"] is False and "person" in e["negative"]
        else:
            assert e["solo"] is True and "outstretched arm" in e["negative"]


def test_each_era_has_21_distinct_places_and_the_modern_ones_include_the_scene_library():
    for era in al.ERAS:
        places = al.venues(era)
        assert len(places) == 21 and len(set(places)) == 21, era
    import scene_library

    assert al.venues("modern")[0] == scene_library.scene_text("cozy_cafe")


def test_every_entry_is_a_job_the_skill_accepts_with_the_safety_terms(tmp_path):
    bible = al.cs.load_bible(al.BIBLE_ROOT)
    config = z.load_config(tmp_path / "none.json", env={"ZIMG_OUTPUT_DIR": str(tmp_path / "out")})
    for entry in al.build_catalog():
        job = z.build_job(config, **al.job_spec(bible, entry))
        assert job["width"] == 720 and job["height"] == 1280 and job["tag"] == entry["id"]
        assert all(t in job["negative_prompt"] for t in ("child", "minor", "underage", "teen"))
        assert job["cfg"] >= 1.5


def test_the_pilot_covers_every_era_age_band_gender_and_kind(catalog):
    ids = al.pilot_ids(catalog)
    assert len(ids) == 16 and len(set(ids)) == 16
    by_id = {e["id"]: e for e in catalog}
    picked = [by_id[i] for i in ids]
    assert {e["era"] for e in picked} == set(al.ERAS)
    assert {e["kind"] for e in picked} == {"portrait", "scene", "place"}
    assert {e["gender"] for e in picked if e["gender"]} == set(al.GENDERS)
    assert {e["age"] for e in picked if e["age"]} == set(al.AGES)


def test_export_writes_the_manifest_in_the_skills_format(tmp_path):
    out = al.export("pilot", directory=str(tmp_path))
    data = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
    assert out["images"] == 16 and data["batch"] == "assets-pilot" and len(data["jobs"]) == 16
    assert {"prompt", "negative", "width", "height", "steps", "cfg", "seed", "solo", "tag"} <= set(data["jobs"][0])
    with pytest.raises(gc.UsageError):
        al.export("all", ids=["nope"], directory=str(tmp_path))


def test_import_files_the_finished_pictures_and_builds_the_index(tmp_path):
    jobs, images = tmp_path / "jobs", tmp_path / "lib" / "images"
    jobs.mkdir()
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG fake")
    for n, tag in enumerate(("portrait-modern-w-20s-1", "place-future-03", "place-future-03")):
        status = "COMPLETED" if n < 2 else "FAILED"
        record = {
            "job_id": f"j{n}",
            "tag": tag,
            "status": status,
            "output": str(png),
            "prompt": "p",
            "completed_at": f"2026-10-09T0{n}",
            "gpu": "L4",
        }
        (jobs / f"{n}.json").write_text(json.dumps(record))
    report = al.import_jobs(jobs_dir=str(jobs), images_dir=str(images))
    assert sorted(report["imported"]) == ["place-future-03", "portrait-modern-w-20s-1"]
    assert (images / "portrait" / "modern" / "portrait-modern-w-20s-1.png").is_file()
    assert (images / "place" / "future" / "place-future-03.png").is_file()
    index = json.loads((tmp_path / "lib" / "index.json").read_text(encoding="utf-8"))
    assert index["count"] == 2 and (tmp_path / "lib" / "index.html").is_file()
    again = al.import_jobs(jobs_dir=str(jobs), images_dir=str(images))
    assert again["imported"] == [] and len(again["skipped"]) == 2


def test_measure_reports_age_by_band_similarity_and_faces_in_empty_places(tmp_path):
    images = tmp_path / "images"
    wanted = ("portrait-modern-w-20s-1", "scene-modern-w-20s-1", "place-modern-01", "place-modern-02")
    for e in al.build_catalog():
        if e["id"] in wanted:
            path = images / e["kind"] / e["era"] / f"{e['id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def describe(path):
        if "place-modern-02" in path:
            return {"age": 30, "embedding": [1.0, 0.0]}  # a face where nobody should be
        if "place" in path:
            return None
        return {"age": 24, "embedding": [1.0, 0.0] if "portrait" in path else [0.6, 0.8]}

    report = al.measure(describe, images_dir=str(images))
    assert report["age_by_band"]["woman-20s"]["mean"] == 24
    assert report["portrait_vs_scene_similarity"] == {"pairs": 1, "mean": 0.6, "min": 0.6}
    assert report["faces_in_empty_places"] == ["place-modern-02"]


def test_the_look_texts_use_the_right_article_and_name_the_age_in_words(catalog):
    for e in catalog:
        if e["kind"] != "place":
            assert " a a " not in e["look"] and not re.search(r"\ba [AEIOU]", e["look"]), e["look"]
    assert al.with_article("Ethiopian") == "an Ethiopian" and al.with_article("Thai") == "a Thai"


def test_empty_places_also_forbid_hands_and_arms(catalog):
    assert all("hand" in e["negative"] and "arm" in e["negative"] for e in catalog if e["kind"] == "place")


def test_the_agecheck_set_is_eight_portraits_of_the_two_bands_drawn_too_old(catalog):
    ids = al.agecheck_ids(catalog)
    by_id = {e["id"]: e for e in catalog}
    assert len(ids) == 8 and {by_id[i]["age"] for i in ids} == {"30-40s", "50-60s"}
    assert {by_id[i]["gender"] for i in ids} == set(al.GENDERS) and {by_id[i]["kind"] for i in ids} == {"portrait"}


def test_the_redo_set_is_twenty_pictures_with_new_seeds_and_stronger_negatives(catalog):
    ids = al.redo_ids(catalog)
    by_id = {e["id"]: e for e in catalog}
    assert len(ids) == 20 and len(set(ids)) == 20
    for i in ids:
        e = by_id[i]
        assert e["seed"] >= al.SEED_BASE + al.REDO_SEED_OFFSET  # a new seed, so it is a new picture
        if e["kind"] == "place":
            assert all(w in e["negative"] for w in ("nude", "legs", "feet", "letters", "shadow of a person"))
        else:
            assert all(w in e["negative"] for w in ("strapless", "shirtless", "plunging neckline"))
    others = [e for e in catalog if e["id"] not in ids]
    assert max(e["seed"] for e in others) < al.SEED_BASE + len(catalog)


def test_the_clothes_that_were_redone_cover_the_shoulders_and_chest(catalog):
    by_id = {e["id"]: e for e in catalog}
    assert "covered neckline" in by_id["portrait-ancient-w-20s-1"]["outfit"]
    assert "both shoulders" in by_id["portrait-ancient-m-70plus-1"]["outfit"]
    assert "buttoned shirt" in by_id["portrait-retro-m-20s-1"]["outfit"]


def test_the_second_redo_kept_two_pictures_with_rewritten_places_and_the_other_five_went_to_the_third(catalog):
    ids = al.redo2_ids(catalog)
    by_id = {e["id"]: e for e in catalog}
    assert sorted(ids) == ["place-future-09", "place-modern-08"] and all(i in al.REDO_IDS for i in ids)
    for i in ids:
        assert by_id[i]["seed"] >= al.SEED_BASE + 2 * al.REDO_SEED_OFFSET
    assert "deserted" in by_id["place-modern-08"]["venue"] and "no furniture" in by_id["place-future-09"]["venue"]


def test_the_third_redo_swaps_five_places_for_new_ones_with_their_own_seeds(catalog):
    ids = al.redo3_ids(catalog)
    by_id = {e["id"]: e for e in catalog}
    assert sorted(ids) == ["place-future-17", "place-modern-11", "place-retro-08", "place-retro-16", "place-retro-18"]
    for i in ids:
        assert by_id[i]["seed"] >= al.SEED_BASE + 3 * al.REDO_SEED_OFFSET
    venues = " ".join(by_id[i]["venue"] for i in ids)
    for word in ("subway", "telephone", "internet cafe", "drive-in", "forest"):
        assert word not in venues  # the places that kept failing are gone
    assert "hotel corridor" in by_id["place-modern-11"]["venue"] and "mushrooms" in by_id["place-future-17"]["venue"]
