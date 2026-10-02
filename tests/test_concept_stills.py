"""projects/ai_short_drama + training/concept_stills.py, offline.

What is pinned: the brief's shape (5 stories x 7 scenes, folder names, seeds, A/B first round), that every
candidate prompt of every story carries the project's safety negatives and cfg floor, that the character
bible reaches every scene prompt unchanged, and the review state machine (no C/D without a REGENERATE
decision, no silent overwrite of an approved scene). No GPU: submit is a fake that hands back a file."""

import os
import re
import shutil

import concept_stills as cs
import pytest

import comfyui_client as client
import generate_character as gc

FOLDERS = [
    "story_01_2347",
    "story_02_do_not_answer",
    "story_03_seventh_person",
    "story_04_i_wont_save_you",
    "story_05_another_me",
]


@pytest.fixture(scope="module")
def bible():
    return cs.load_bible()


@pytest.fixture(scope="module")
def stories():
    return cs.load_stories()


@pytest.fixture
def work(tmp_path):
    """A scratch copy of the project (specs only) so generate/review can write freely."""
    root = tmp_path / "ai_short_drama"
    shutil.copytree(cs.PROJECT_ROOT, root, ignore=shutil.ignore_patterns("*.png", "review.json", "master.json"))
    return str(root), cs.load_bible(str(root)), cs.load_stories(str(root))


def fake_submit(tmp_path):
    calls = []

    def submit(**kw):
        path = tmp_path / f"raw_{len(calls)}.png"
        path.write_bytes(b"\x89PNG fake " + str(len(calls)).encode())
        calls.append(kw)
        return str(path)

    submit.calls = calls
    return submit


# --- the brief's shape -------------------------------------------------------------------------------


def test_five_stories_seven_scenes_in_the_agreed_folders(stories):
    assert [os.path.basename(s.dir) for s in stories] == FOLDERS
    for story in stories:
        cs.validate_story(story)
        assert [s["scene"] for s in story.data["scenes"]] == list(range(1, 8))


def test_first_round_is_two_per_scene_seventy_in_total(bible, stories):
    assert len(cs.all_requests(bible, stories, cs.FIRST_ROUND)) == 5 * 7 * 2


def test_seeds_are_unique_and_follow_the_documented_rule(bible, stories):
    reqs = cs.all_requests(bible, stories, cs.LETTERS)
    assert len({r["seed"] for r in reqs}) == len(reqs) == 5 * 7 * 4
    story = stories[0]
    assert cs.candidate_seed(story, 3, "B") == story.data["seed_base"] + 3 * 10 + 2


def test_a_and_b_differ_only_in_framing(bible, stories):
    for story in stories:
        for scene in story.data["scenes"]:
            a = cs.build_request(bible, story, scene["scene"], "A")
            b = cs.build_request(bible, story, scene["scene"], "B")
            assert a["composition"] != b["composition"]
            # Same character, location, lighting and story information: strip each framing and compare.
            strip = lambda r: r["prompt"].replace(r["composition"].rstrip("."), "")  # noqa: E731
            assert strip(a) == strip(b)
            assert a["seed"] != b["seed"]


# --- safety path ---------------------------------------------------------------------------------------


def test_every_candidate_keeps_the_age_safety_negatives_and_cfg_floor(bible, stories):
    for r in cs.all_requests(bible, stories, cs.LETTERS):
        for term in gc.AGE_SAFETY_NEGATIVE.split(", "):
            assert term in r["negative_prompt"], (r["story_id"], r["scene_id"], term)
        assert "nsfw" in r["negative_prompt"]
        assert r["cfg"] >= client.SAFETY_MIN_CFG


def test_a_low_cfg_in_the_bible_is_floored(bible, stories):
    low = {**bible, "sampler": {"steps": 8, "cfg": 1.0}}
    assert cs.build_request(low, stories[0], 1, "A")["cfg"] == client.SAFETY_MIN_CFG


ADULT_AGE = re.compile(r"\b(?:(?:late|early|mid)\s+)?([2-9])0s\b|aged\s+(\d\d)")


def test_every_character_in_the_bible_is_an_adult(stories):
    for story in stories:
        for key, character in story.data["characters"].items():
            m = ADULT_AGE.search(character["description"])
            assert m, f"{story.id}/{key} has no adult age marker"
            first = int(m.group(1) + "0") if m.group(1) else int(m.group(2))
            assert first >= 20


def test_prompts_are_within_the_shared_length_limit(bible, stories):
    assert max(len(r["prompt"]) for r in cs.all_requests(bible, stories, cs.LETTERS)) <= gc.MAX_PROMPT_CHARS


# --- prompt content ------------------------------------------------------------------------------------


def test_style_exclusions_live_in_the_negative_not_the_positive(bible, stories):
    for r in cs.all_requests(bible, stories, "AB"):
        positive = r["prompt"].lower()
        for word in ("anime", "cartoon", "no illustration", "extremely dark"):
            assert word not in positive
    assert "anime" in cs.build_request(bible, stories[0], 1, "A")["negative_prompt"]


def test_the_character_bible_reaches_every_scene_unchanged(bible, stories):
    for story in stories:
        for scene in story.data["scenes"]:
            r = cs.build_request(bible, story, scene["scene"], "A")
            for key in scene["characters"]:
                assert story.data["characters"][key]["description"] in r["prompt"]


def test_story_titles_are_never_painted_into_the_image(bible, stories):
    for story in stories:
        for scene in story.data["scenes"]:
            r = cs.build_request(bible, story, scene["scene"], "A")
            assert story.data["title"].lower() not in r["scene_prompt"].lower()


def test_another_me_double_is_the_same_man(stories):
    story = next(s for s in stories if s.id == "story_05_another_me")
    lead, double = story.data["characters"]["lead"], story.data["characters"]["double"]
    assert "same face" in double["description"] and "same age" in double["description"]
    assert "same short black hair" in double["description"]
    # the clothing sentence is shared word for word, so the two cannot drift apart
    assert "dark green utility jacket over a white T-shirt" in lead["description"]
    assert "dark green utility jacket over a white T-shirt" in double["description"]


def test_text_in_image_scenes_drop_the_text_negative_and_group_scenes_drop_solo(bible, stories):
    tokens = lambda r: set(r["negative_prompt"].split(", "))  # noqa: E731
    seen_text = seen_group = False
    for story in stories:
        for scene in story.data["scenes"]:
            r = cs.build_request(bible, story, scene["scene"], "A")
            if scene["on_screen_text"]:
                seen_text = True
                assert "text" not in tokens(r)
            else:
                assert "text" in tokens(r)
            if scene["solo"]:
                assert "crowd" in tokens(r)
            elif not scene.get("negative_extra"):
                seen_group = True
                assert "crowd" not in tokens(r)
    assert seen_text and seen_group


def test_the_empty_platform_scene_asks_for_no_people(bible, stories):
    r = cs.build_request(bible, stories[0], 7, "A")
    assert "crowd" in r["negative_prompt"] and "people" in r["negative_prompt"]


def test_validation_rejects_a_scene_with_an_unknown_character(stories):
    story = cs.Story(
        {**stories[0].data, "scenes": [{**s} for s in stories[0].data["scenes"]]},
        stories[0].dir,
    )
    story.data["scenes"][2]["characters"] = ["nobody"]
    with pytest.raises(gc.UsageError, match="nobody"):
        cs.validate_story(story)


# --- generation, metadata, review ---------------------------------------------------------------------


def test_generate_writes_image_metadata_workflow_and_hands_off_to_review(work, tmp_path):
    root, bible, stories = work
    submit = fake_submit(tmp_path)
    story = stories[1]
    made = cs.generate(bible, [story], scene_numbers={2}, submit=submit, server_up=lambda: True)
    assert made == 2 and len(submit.calls) == 2
    for letter, num in (("A", "01"), ("B", "02")):
        paths = cs.candidate_paths(story, 2, letter)
        assert paths["png"].endswith(f"scene_02{os.sep}candidate_{num}.png") and os.path.isfile(paths["png"])
        meta = cs._read_json(paths["meta"])
        for key in (
            "story_id",
            "scene_id",
            "candidate_id",
            "prompt",
            "negative_prompt",
            "seed",
            "width",
            "height",
            "steps",
            "model",
            "workflow",
            "created_at",
        ):
            assert key in meta
        # the record is enough to rebuild the exact request, and the saved graph carries the same seed/prompt
        req = cs.build_request(bible, story, 2, letter)
        assert (meta["prompt"], meta["negative_prompt"], meta["seed"]) == (
            req["prompt"],
            req["negative_prompt"],
            req["seed"],
        )
        graph = cs._read_json(paths["workflow"])
        assert graph["3"]["inputs"]["seed"] == meta["seed"]
        assert graph["6"]["inputs"]["text"] == meta["prompt"]
        assert graph["3"]["inputs"]["cfg"] >= client.SAFETY_MIN_CFG
    review = cs.load_review(story, 2)
    assert review["status"] == cs.HUMAN_REVIEW
    assert [h["status"] for h in review["history"]] == [cs.CANDIDATES_READY, cs.HUMAN_REVIEW]


def test_generate_skips_existing_and_a_dry_run_touches_nothing(work, tmp_path, capsys):
    root, bible, stories = work
    submit = fake_submit(tmp_path)
    cs.generate(bible, [stories[0]], scene_numbers={1}, dry_run=True, submit=submit, server_up=lambda: False)
    assert submit.calls == [] and "2 張待生成" in capsys.readouterr().out
    cs.generate(bible, [stories[0]], scene_numbers={1}, submit=submit, server_up=lambda: True)
    assert cs.generate(bible, [stories[0]], scene_numbers={1}, submit=submit, server_up=lambda: True) == 0
    assert len(submit.calls) == 2


def test_generate_refuses_when_comfyui_is_not_running(work, tmp_path):
    _, bible, stories = work
    with pytest.raises(gc.UsageError, match="ComfyUI"):
        cs.generate(bible, [stories[0]], submit=fake_submit(tmp_path), server_up=lambda: False)


def test_extra_candidates_need_a_regenerate_decision(work, tmp_path):
    _, bible, stories = work
    submit = fake_submit(tmp_path)
    story = stories[2]
    cs.generate(bible, [story], scene_numbers={4}, submit=submit, server_up=lambda: True)
    with pytest.raises(gc.UsageError, match="REGENERATE"):
        cs.generate(bible, [story], scene_numbers={4}, letters=("C", "D"), submit=submit, server_up=lambda: True)
    cs.review(story, 4, regenerate=True, notes="knife not readable")
    assert (
        cs.generate(bible, [story], scene_numbers={4}, letters=("C", "D"), submit=submit, server_up=lambda: True) == 2
    )
    assert cs.load_review(story, 4)["status"] == cs.HUMAN_REVIEW
    assert cs.load_review(story, 4)["notes"] == "knife not readable"
    assert sorted(os.path.basename(cs.candidate_paths(story, 4, x)["png"]) for x in cs.existing_letters(story, 4)) == [
        f"candidate_0{i}.png" for i in (1, 2, 3, 4)
    ]


def test_approve_copies_the_master_and_protects_it_from_force(work, tmp_path):
    _, bible, stories = work
    submit = fake_submit(tmp_path)
    story = stories[3]
    cs.generate(bible, [story], scene_numbers={5}, submit=submit, server_up=lambda: True)
    result = cs.review(story, 5, approve="2")
    assert result["status"] == cs.APPROVED and result["approved_candidate"] == "candidate_02.png"
    master = os.path.join(story.scene_dir(5), "master.png")
    assert open(master, "rb").read() == open(cs.candidate_paths(story, 5, "B")["png"], "rb").read()
    assert cs._read_json(os.path.join(story.scene_dir(5), "master.json"))["seed"] == cs.candidate_seed(story, 5, "B")
    with pytest.raises(gc.UsageError, match="APPROVED"):
        cs.generate(bible, [story], scene_numbers={5}, force=True, submit=submit, server_up=lambda: True)


def test_review_needs_exactly_one_decision_and_a_real_candidate(work, tmp_path):
    _, bible, stories = work
    story = stories[0]
    with pytest.raises(gc.UsageError):
        cs.review(story, 1)
    with pytest.raises(gc.UsageError):
        cs.review(story, 1, approve="A", regenerate=True)
    with pytest.raises(gc.UsageError, match="還沒有候選圖"):
        cs.review(story, 1, regenerate=True)
    with pytest.raises(gc.UsageError, match="沒有這張候選圖"):
        cs.review(story, 1, approve="A")
    with pytest.raises(gc.UsageError, match="1-4"):
        cs.review(story, 1, approve="9")


def test_index_sheet_and_status_reflect_what_exists(work, tmp_path):
    root, bible, stories = work
    cs.generate(bible, [stories[0]], scene_numbers={1}, submit=fake_submit(tmp_path), server_up=lambda: True)
    index = cs.build_index(stories, root)
    row = next(r for r in index["scenes"] if (r["story_id"], r["scene_id"]) == ("story_01_2347", "scene_01"))
    assert row["status"] == cs.HUMAN_REVIEW and len(row["candidates"]) == 2
    assert row["candidates"][0]["image"] == "story_01_2347/scene_01/candidate_01.png"
    assert len(index["scenes"]) == 35
    sheet = cs.build_sheet(bible, stories, root)
    html = open(sheet, encoding="utf-8").read()
    assert "candidate_02.png" in html and "23:46" in html and "HUMAN_REVIEW" in html
    assert "images  2" in cs.status_lines(stories)[0]


def test_init_creates_scene_folders_and_prompt_files(work):
    root, _, _ = work
    cs.main(["--root", root, "init"])
    for folder in FOLDERS:
        assert os.path.isfile(os.path.join(root, folder, "prompts.json"))
        assert len([d for d in os.listdir(os.path.join(root, folder)) if d.startswith("scene_")]) == 7
    assert len(cs._read_json(os.path.join(root, FOLDERS[0], "prompts.json"))["candidates"]) == 14
