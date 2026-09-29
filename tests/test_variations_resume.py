"""gen_variations / gen_suggestive_variations pick up where a batch stopped.

Both are resumable by what is already in the dataset folder. Two things used to go wrong:
- a resumed run restarted random.Random(base_seed), so the image at the resume point got image 0's
  angle, pose and outfit (and the same caption) again;
- the resume point was the number of files, so deleting a bad image while curating the set made the
  next run start one index too early and overwrite the newest image.

The ComfyUI round-trip is replaced by a fake that writes the prompt into the "image", so every file
records the prompt it was generated from. Offline, no GPU.
"""

import comfyui_client as client
import generate_character as gc
import pytest

GENERATORS = [
    pytest.param(gc.gen_variations, "var_", id="variations"),
    pytest.param(gc.gen_suggestive_variations, "sugg_", id="suggestive"),
]
TRIGGER = "wanling"
BASE_SEED = 2000


@pytest.fixture
def submitted(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    calls = []

    def fake_submit(**kwargs):
        calls.append(kwargs)
        path = raw_dir / f"{kwargs['filename_prefix']}.png"
        path.write_text(kwargs["prompt"], encoding="utf-8")
        return str(path)

    monkeypatch.setattr(client, "upload_reference_image", lambda path: "face_ref.png")
    monkeypatch.setattr(client, "log_gpu_memory", lambda stage: None)
    monkeypatch.setattr(client, "submit_generation", fake_submit)
    return calls


def run(generate, out_dir, count):
    generate(TRIGGER, "anchor.png", str(out_dir), count, 1.0, BASE_SEED)


def contents(out_dir):
    """{filename: text} for every image (which holds its prompt) and caption in out_dir."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(out_dir.iterdir())}


@pytest.mark.parametrize("generate, prefix", GENERATORS)
def test_a_resumed_run_produces_what_one_uninterrupted_run_would(generate, prefix, submitted, tmp_path):
    whole, split = tmp_path / "whole", tmp_path / "split"
    run(generate, whole, 6)
    run(generate, split, 3)
    submitted.clear()
    run(generate, split, 6)

    assert [c["filename_prefix"] for c in submitted] == [f"{prefix}{i:04d}_seed{BASE_SEED + i}" for i in (3, 4, 5)]
    assert contents(split) == contents(whole)


@pytest.mark.parametrize("generate, prefix", GENERATORS)
def test_a_deleted_image_is_neither_regenerated_nor_overwritten(generate, prefix, submitted, tmp_path):
    reference, curated = tmp_path / "reference", tmp_path / "curated"
    run(generate, reference, 6)
    run(generate, curated, 5)
    for suffix in (".png", ".txt"):
        (curated / f"{prefix}0001_seed{BASE_SEED + 1}{suffix}").unlink()
    kept = contents(curated)
    submitted.clear()

    run(generate, curated, 5)

    assert [c["filename_prefix"] for c in submitted] == [f"{prefix}0005_seed{BASE_SEED + 5}"]
    after = contents(curated)
    assert {name: after[name] for name in kept} == kept, "an existing image or caption was rewritten"
    new = {name: text for name, text in after.items() if name not in kept}
    assert new == {name: text for name, text in contents(reference).items() if name.startswith(f"{prefix}0005_")}


@pytest.mark.parametrize("generate, prefix", GENERATORS)
def test_nothing_is_generated_once_the_folder_holds_count_images(generate, prefix, submitted, tmp_path):
    run(generate, tmp_path / "set", 3)
    submitted.clear()
    run(generate, tmp_path / "set", 3)
    run(generate, tmp_path / "set", 2)
    assert submitted == []


def test_the_two_prefixes_resume_independently_in_a_shared_folder(submitted, tmp_path):
    shared = tmp_path / "shared"
    run(gc.gen_suggestive_variations, shared, 2)
    run(gc.gen_variations, shared, 2)
    assert sorted(p.name for p in shared.glob("*.png")) == [
        f"sugg_0000_seed{BASE_SEED}.png",
        f"sugg_0001_seed{BASE_SEED + 1}.png",
        f"var_0000_seed{BASE_SEED}.png",
        f"var_0001_seed{BASE_SEED + 1}.png",
    ]
