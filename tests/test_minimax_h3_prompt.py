"""Prompt assembly for MiniMax H3 first-frame (I2VA) jobs: the official format, the user's words kept
verbatim, constraints that stop the prompt contradicting the user, and the positive-side safety screen
that stands in for the negative prompt H3's turbo graph does not have (CLAUDE.md, 年齡安全)."""

import pytest

import generate_character as gc
from minimax_h3_fakes import h3, import_skill

WALK = (
    "The woman in the first frame stands on the same sunlit street, keeping her outfit and hairstyle. "
    "She walks slowly toward the camera with a relaxed expression."
)


def spec(description="", **kw):
    return h3.JobSpec(image="x.png", description=description, **kw)


def test_i2va_prompt_matches_the_official_layout():
    text = h3.build_prompt(spec("She smiles.", soundscape="Birds chirp.", music="Soft piano."))
    assert text == (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        "\n\nintegrated_multimodal_description: [Shot 1] She smiles."
        "\n\noverall_soundscape: Birds chirp."
        "\n\nnon_diegetic_music: Soft piano."
    )


def test_the_users_description_is_kept_verbatim_and_nothing_is_added():
    text = h3.build_prompt(spec(WALK))
    assert f"[Shot 1] {WALK}\n\n" in text
    assert h3.LOCKED_CAMERA_SENTENCE not in text and h3.STATIC_SUBJECT_SENTENCE not in text


def test_defaults_for_soundscape_and_music():
    text = h3.build_prompt(spec("She smiles."))
    assert f"overall_soundscape: {h3.DEFAULT_SOUNDSCAPE}" in text
    assert text.endswith("non_diegetic_music: N/A")


@pytest.mark.parametrize("description", ["", "   "])
def test_missing_prompt_is_rejected(description):
    with pytest.raises(h3.InputError) as err:
        h3.build_prompt(spec(description))
    assert err.value.code == "INVALID_INPUT"


def test_a_full_prompt_in_the_description_flag_is_rejected():
    with pytest.raises(h3.InputError, match="--prompt-file"):
        h3.build_prompt(spec("[Shot 1] <Picture 1> walks"))


def test_prompt_file_is_sent_verbatim(tmp_path):
    full = h3.build_i2va_prompt("She waves.", "Wind.", "N/A")
    path = tmp_path / "p.txt"
    path.write_text(full + "\n", encoding="utf-8")
    assert h3.build_prompt(spec(prompt_file=path)) == full


@pytest.mark.parametrize(
    "content,match",
    [
        ("", "empty"),
        ("integrated_multimodal_description: [Shot 1] She waves.", "Picture 1"),
        (h3.I2VA_ALIGNMENT + " <Picture 2> too", "Picture 1"),
    ],
)
def test_bad_prompt_files_are_rejected(tmp_path, content, match):
    path = tmp_path / "p.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(h3.InputError, match=match):
        h3.build_prompt(spec(prompt_file=path))


def test_prompt_and_prompt_file_together_are_rejected(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(h3.build_i2va_prompt("a", "b", "c"), encoding="utf-8")
    with pytest.raises(h3.InputError, match="not both"):
        h3.build_prompt(spec("She waves.", prompt_file=path))


def test_overlong_prompt_is_rejected():
    with pytest.raises(h3.InputError, match="limit"):
        h3.build_prompt(spec("She waves. " * 500))


# --- constraints: user intent over prompt optimization ----------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "The camera slowly pushes in toward her face.",
        "A slow zoom onto the cup.",
        "The camera pans left across the cafe.",
        "Handheld footage of her laughing.",
        "The shot follows her across the room.",
    ],
)
def test_locked_camera_rejects_camera_movement(description):
    with pytest.raises(h3.InputError) as err:
        h3.build_prompt(spec(description, constraints=("locked-camera",)))
    assert err.value.code == "PROMPT_REJECTED"


def test_locked_camera_allows_subject_motion_and_states_it_positively():
    text = h3.build_prompt(spec("She tilts her head and smiles.", constraints=("locked-camera",)))
    assert "She tilts her head and smiles. " + h3.LOCKED_CAMERA_SENTENCE in text


def test_locked_camera_sentence_is_not_added_twice():
    text = h3.build_prompt(spec("The camera remains static. She smiles.", constraints=("locked-camera",)))
    assert h3.LOCKED_CAMERA_SENTENCE not in text


@pytest.mark.parametrize("description", [WALK, "She runs out of the frame.", "He steps forward and waves."])
def test_static_subject_rejects_locomotion(description):
    with pytest.raises(h3.InputError) as err:
        h3.build_prompt(spec(description, constraints=("static-subject",)))
    assert err.value.code == "PROMPT_REJECTED"


def test_static_subject_allows_camera_motion_and_small_gestures():
    text = h3.build_prompt(
        spec("The camera pushes in with small amplitude at slow speed. She blinks.", constraints=("static-subject",))
    )
    assert text.count(h3.STATIC_SUBJECT_SENTENCE) == 1


def test_unknown_constraint_is_rejected():
    with pytest.raises(h3.InputError, match="unknown constraint"):
        h3.build_prompt(spec("She smiles.", constraints=("slow-motion",)))


def test_constraints_also_check_a_prompt_file(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(h3.build_i2va_prompt("The camera orbits her.", "Wind.", "N/A"), encoding="utf-8")
    with pytest.raises(h3.InputError):
        h3.build_prompt(spec(prompt_file=path, constraints=("locked-camera",)))


# --- the safety screen ---------------------------------------------------------------------------------


def test_blocked_terms_are_pinned():
    """Pinned like AGE_SAFETY_NEGATIVE in test_safety_invariants.py: removing a term must fail here,
    not just stop matching."""
    assert h3.BLOCKED_TERMS == (
        "child", "children", "kid", "kids", "minor", "minors", "teen", "teens", "teenage", "teenager",
        "teenagers", "underage", "preteen", "young girl", "young girls", "young boy", "young boys",
        "little girl", "little boy", "schoolgirl", "schoolboy", "infant", "toddler", "juvenile", "loli",
        "lolita", "shota", "nsfw", "nude", "nudity", "naked", "topless", "bottomless", "explicit", "sex",
        "sexual", "sexually", "porn", "porno", "pornographic", "erotic", "genitals", "undress", "undressing",
    )  # fmt: skip
    assert len(h3.BLOCKED_TERMS_CJK) == 28


def test_every_age_safety_negative_term_is_blocked():
    terms = [t.strip() for t in gc.AGE_SAFETY_NEGATIVE.split(",")]
    assert terms  # an emptied constant must not make this vacuous
    for term in terms:
        assert term in h3.BLOCKED_TERMS
        with pytest.raises(h3.InputError) as err:
            h3.build_prompt(spec(f"A {term} waves at the camera."))
        assert err.value.code == "PROMPT_REJECTED"


@pytest.mark.parametrize(
    "text",
    [
        "She is naked.",
        "An NSFW scene.",
        "A young-girl smiles.",
        "a 16-year-old walks",
        "a 17 yo smiles",
        "sixteen years old",
        "十六歲的女孩",
        "16歲",
        "一個蘿莉",
        "穿制服的高中生",
    ],
)
def test_blocked_content_is_refused(text):
    with pytest.raises(h3.InputError) as err:
        h3.screen_prompt(text)
    assert err.value.code == "PROMPT_REJECTED"


@pytest.mark.parametrize(
    "text",
    [
        "He is kidding around.",
        "An 18-year-old woman.",
        "She is 25 years old.",
        "十八歲的女人",
        "二十歲",
        "It takes fifteen minutes.",
        "Sussex countryside.",
    ],
)
def test_adult_and_benign_text_passes(text):
    h3.screen_prompt(text)


def test_screen_covers_soundscape_and_prompt_files(tmp_path):
    with pytest.raises(h3.InputError):
        h3.build_prompt(spec("She smiles.", soundscape="Children laughing nearby."))
    path = tmp_path / "p.txt"
    path.write_text(h3.build_i2va_prompt("A teenager waves.", "Wind.", "N/A"), encoding="utf-8")
    with pytest.raises(h3.InputError):
        h3.build_prompt(spec(prompt_file=path))


def test_the_screen_has_no_bypass():
    run = import_skill("run")
    flags = {a for action in run.build_parser()._actions for a in action.option_strings}
    assert not any(word in flag for flag in flags for word in ("unsafe", "skip", "allow", "force", "no-screen"))
