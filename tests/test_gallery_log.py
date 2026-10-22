"""gallery_log: the record of Pony x Z-Image pictures with their prompts, and the kohya-style export for training."""

import json
import os

import pytest

import gallery_log as gl


def make_png(path):
    path.write_bytes(b"\x89PNG fake")
    return str(path)


def add(store, tmp_path, name, **over):
    args = dict(
        base=make_png(tmp_path / f"{name}_pony.png"),
        final=make_png(tmp_path / f"{name}.png"),
        prompt_original="一位女性在咖啡廳",
        prompt_pony="asian woman, early twenties, cafe",
        prompt_zimage="An Asian woman in her early twenties sits in a cafe.",
        negative="lowres",
        seed=7,
        denoise=0.2,
        pony_checkpoint="cyberrealistic_pony",
        tier="safe",
        store=str(store),
        now=1_800_000_000,
    )
    args.update(over)
    return gl.record(**args)


@pytest.fixture
def store(tmp_path):
    return tmp_path / "gallery" / "records.json"


def test_a_record_keeps_every_prompt_and_setting(store, tmp_path):
    rec = add(store, tmp_path, "a")
    saved = json.loads(store.read_text(encoding="utf-8"))
    assert saved == [rec]
    assert rec["prompt_original"] == "一位女性在咖啡廳" and rec["seed"] == 7 and rec["denoise"] == 0.2
    assert rec["marked"] is False and rec["source"] == "single"
    assert not os.path.exists(str(store) + ".tmp")


def test_writing_the_same_picture_again_replaces_the_record_but_keeps_its_mark(store, tmp_path):
    first = add(store, tmp_path, "a")
    gl.set_marked(first["id"], True, store=str(store))
    again = add(store, tmp_path, "a", prompt_pony="new tags")
    records = gl.load(str(store))
    assert len(records) == 1 and records[0]["id"] == first["id"] == again["id"]
    assert records[0]["prompt_pony"] == "new tags" and records[0]["marked"] is True


def test_a_record_needs_a_picture():
    with pytest.raises(ValueError):
        gl.record(
            base=None,
            final=None,
            prompt_original="",
            prompt_pony="x",
            seed=1,
            denoise=0.2,
            pony_checkpoint="p",
            tier="safe",
            store="unused",
        )


def test_load_is_newest_first_and_drops_records_whose_pictures_are_gone(store, tmp_path):
    old = add(store, tmp_path, "old", now=1_700_000_000)
    new = add(store, tmp_path, "new", now=1_800_000_000)
    gone = add(store, tmp_path, "gone", now=1_900_000_000)
    os.remove(gone["final"])
    os.remove(gone["base"])
    assert [r["id"] for r in gl.load(str(store))] == [new["id"], old["id"]]
    assert len(gl.load(str(store), only_existing=False)) == 3


def test_search_matches_every_word_in_any_prompt_and_can_keep_only_the_marked(store, tmp_path):
    a = add(store, tmp_path, "a")
    b = add(
        store, tmp_path, "b", prompt_original="廚房切菜", prompt_pony="asian man, kitchen", prompt_zimage=None, seed=99
    )
    gl.set_marked(b["id"], True, store=str(store))
    records = gl.load(str(store))
    assert [r["id"] for r in gl.search(records, "cafe woman")] == [a["id"]]
    assert [r["id"] for r in gl.search(records, "廚房")] == [b["id"]]
    assert [r["id"] for r in gl.search(records, "99")] == [b["id"]]
    assert [r["id"] for r in gl.search(records, "", only_marked=True)] == [b["id"]]
    assert gl.search(records, "nonexistent") == []


def test_set_marked_says_whether_the_record_exists(store, tmp_path):
    rec = add(store, tmp_path, "a")
    assert gl.set_marked(rec["id"], True, store=str(store)) is True
    assert gl.find(rec["id"], str(store))["marked"] is True
    assert gl.set_marked("nope", True, store=str(store)) is False


def test_the_caption_follows_the_chosen_kind_and_falls_back_for_the_plain_translation(store, tmp_path):
    rec = add(store, tmp_path, "a")
    assert gl.caption_for(rec, "pony") == "asian woman, early twenties, cafe"
    assert gl.caption_for(rec, "zimage").startswith("An Asian woman")
    assert gl.caption_for(rec, "original") == "一位女性在咖啡廳"
    assert gl.caption_for(rec, "pony", trigger="mychar") == "mychar, asian woman, early twenties, cafe"
    plain = add(store, tmp_path, "b", prompt_zimage=None)
    assert gl.caption_for(plain, "zimage") == plain["prompt_pony"]
    with pytest.raises(ValueError):
        gl.caption_for(rec, "other")


def test_picture_falls_back_to_the_other_file_when_the_asked_one_is_missing(store, tmp_path):
    rec = add(store, tmp_path, "a")
    assert gl.picture(rec, "final") == rec["final"] and gl.picture(rec, "base") == rec["base"]
    os.remove(rec["final"])
    assert gl.picture(rec, "final") == rec["base"]
    os.remove(rec["base"])
    assert gl.picture(rec, "final") is None


def test_export_writes_kohya_layout_with_a_caption_beside_every_picture(store, tmp_path):
    a = add(store, tmp_path, "a", seed=1)
    b = add(store, tmp_path, "b", seed=2)
    add(store, tmp_path, "c", seed=3)  # not asked for
    out = gl.export_dataset(
        [a["id"], b["id"]],
        name="My Char!",
        repeats=5,
        caption_kind="pony",
        trigger="mychar",
        dest_root=str(tmp_path / "out"),
        store=str(store),
        now=1_800_000_000,
    )
    assert out["written"] == 2 and out["skipped"] == []
    assert os.path.basename(out["dataset_dir"]) == "5_My_Char"
    files = sorted(os.listdir(out["dataset_dir"]))
    assert files == sorted([f"{a['id']}.png", f"{a['id']}.txt", f"{b['id']}.png", f"{b['id']}.txt"])
    text = (
        (tmp_path / "out")
        .joinpath(os.path.basename(out["folder"]), "5_My_Char", f"{a['id']}.txt")
        .read_text(encoding="utf-8")
    )
    assert text == "mychar, asian woman, early twenties, cafe\n"
    manifest = json.loads(open(os.path.join(out["folder"], "export.json"), encoding="utf-8").read())
    assert manifest["repeats"] == 5 and manifest["trigger"] == "mychar" and len(manifest["written"]) == 2


def test_export_can_use_the_pony_base_instead_of_the_final_picture(store, tmp_path):
    rec = add(store, tmp_path, "a")
    out = gl.export_dataset([rec["id"]], name="x", image_kind="base", dest_root=str(tmp_path / "out"), store=str(store))
    copied = os.path.join(out["dataset_dir"], f"{rec['id']}.png")
    assert open(copied, "rb").read() == open(rec["base"], "rb").read()


def test_export_skips_what_it_cannot_write_and_refuses_an_empty_export(store, tmp_path):
    ok = add(store, tmp_path, "ok")
    lost = add(store, tmp_path, "lost")
    os.remove(lost["final"])
    os.remove(lost["base"])
    blank = add(store, tmp_path, "blank", prompt_pony="", prompt_zimage=None)
    out = gl.export_dataset(
        [ok["id"], lost["id"], blank["id"]], name="x", dest_root=str(tmp_path / "out"), store=str(store)
    )
    assert out["written"] == 1 and {s["id"] for s in out["skipped"]} == {lost["id"], blank["id"]}
    with pytest.raises(ValueError, match="沒有可匯出"):
        gl.export_dataset([lost["id"]], name="y", dest_root=str(tmp_path / "out2"), store=str(store))
    assert not (tmp_path / "out2").exists()  # no empty folder left behind


def test_export_rejects_bad_options(store, tmp_path):
    rec = add(store, tmp_path, "a")
    for kwargs in ({"caption_kind": "x"}, {"image_kind": "x"}, {"repeats": 0}):
        with pytest.raises(ValueError):
            gl.export_dataset([rec["id"]], name="x", dest_root=str(tmp_path / "o"), store=str(store), **kwargs)


def test_import_batches_adds_earlier_batch_pictures_once(store, tmp_path):
    root = tmp_path / "reference_candidates"
    batch = root / "ponyz_batch_20261022_193348"
    batch.mkdir(parents=True)
    img = make_png(batch / "p1_s5.png")
    base = make_png(batch / "p1_s5_pony.png")
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "denoise": 0.2,
                "pony_checkpoint": "cyberrealistic_pony",
                "tier": "suggestive",
                "qwen": True,
                "jobs": [
                    {
                        "status": "ok",
                        "seed": 5,
                        "prompt": "一隻貓",
                        "base": base,
                        "final": img,
                        "english": {"pony": "cat, sitting", "zimage": "A cat sits."},
                    },
                    {"status": "error", "seed": 6, "prompt": "x", "base": None, "final": None, "english": None},
                    {"status": "stopped", "seed": 7, "prompt": "y", "base": None, "final": None, "english": None},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert gl.import_batches(str(root), store=str(store)) == 1
    assert gl.import_batches(str(root), store=str(store)) == 0  # idempotent
    rec = gl.load(str(store))[0]
    assert rec["source"] == "batch" and rec["prompt_original"] == "一隻貓" and rec["prompt_zimage"] == "A cat sits."
    assert gl.import_batches(str(tmp_path / "missing"), store=str(store)) == 0
