"""training/output_dir.py: where the picture/video tab writes is a setting that can be changed and reset, read when a
job needs it (never fixed at import time). The settings file is redirected to a temp folder, so these tests never touch
the real one."""

import json
import os
import subprocess
import tempfile

import director as d
import output_dir as od
import pytest

import generate_character as gc


@pytest.fixture
def settings(tmp_path, monkeypatch):
    path = tmp_path / "settings" / "output_dir.json"
    monkeypatch.setattr(od, "SETTINGS_PATH", str(path))
    return path


def test_without_a_setting_the_default_folder_is_used(settings):
    assert od.current() == os.path.abspath(od.DEFAULT_DIR)
    assert od.is_default()
    assert od.DEFAULT_DIR.endswith(os.path.join("outputs", "h3_local"))


def test_a_folder_can_be_set_remembered_and_reset(settings, tmp_path):
    target = tmp_path / "my videos"
    assert od.set_dir(str(target)) == str(target)
    assert target.is_dir()  # created, because that is the only way to know it can be
    assert json.load(open(settings, encoding="utf-8"))["path"] == str(target)
    assert od.current() == str(target) and not od.is_default()
    assert od.stills_dir() == str(target / "stills") and od.cuts_dir() == str(target / "cuts")
    assert od.imports_dir() == str(target / "imports") and od.thumbs_dir() == str(target / "thumbs")
    assert od.reset() == os.path.abspath(od.DEFAULT_DIR)
    assert not settings.exists() and od.is_default()
    assert od.reset() == os.path.abspath(od.DEFAULT_DIR)  # resetting twice is fine


def test_bad_folders_are_refused_with_a_reason_and_the_old_setting_stays(settings, tmp_path):
    keep = od.set_dir(str(tmp_path / "keep"))
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    for bad, why in (
        ("", "請輸入"),
        ("   ", "請輸入"),
        ("relative/folder", "完整路徑"),
        (str(a_file), "檔案"),
        (os.path.join(od.REPO_ROOT, "models", "out"), "模型"),
        (os.path.join(od.REPO_ROOT, "ComfyUI", "models"), "模型"),
    ):
        with pytest.raises(gc.UsageError, match=why):
            od.set_dir(bad)
    assert od.current() == keep


def test_a_folder_that_cannot_be_written_is_refused(settings, tmp_path, monkeypatch):
    def denied(*a, **k):
        raise PermissionError(13, "拒絕存取")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", denied)
    with pytest.raises(gc.UsageError, match="不能寫入"):
        od.set_dir(str(tmp_path / "locked"))
    assert not settings.exists()  # nothing was remembered


def test_a_damaged_settings_file_falls_back_to_the_default(settings):
    settings.parent.mkdir(parents=True)
    for text in ("not json", "[]", '{"path": ""}', '{"path": 5}'):
        settings.write_text(text, encoding="utf-8")
        assert od.current() == os.path.abspath(od.DEFAULT_DIR)


def test_the_quotes_and_spaces_people_paste_are_understood(settings, tmp_path):
    quoted = '"' + str(tmp_path / "q") + '"'
    assert od.set_dir(quoted) == str(tmp_path / "q")
    assert od.validate(" " + str(tmp_path / "plain") + " ") == str(tmp_path / "plain")


def test_the_director_reads_the_setting_when_it_is_used_not_when_it_is_imported(settings, tmp_path):
    folder = tmp_path / "elsewhere"
    od.set_dir(str(folder))
    (folder / "h3_local_demo.mp4").write_bytes(b"x")
    assert [r["name"] for r in d.list_clips()] == ["h3_local_demo.mp4"]
    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    report = d.import_media([str(source)])
    assert report["imported"] == [str(folder / "imports" / "source.png")]
    names = {
        r["name"]
        for r in d.list_media(ffmpeg="ffmpeg", run=lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", ""))
    }
    assert "source.png" in names and "h3_local_demo.mp4" in names
    od.reset()
    assert all(os.path.dirname(r["path"]) != str(folder) for r in d.list_clips())


def test_no_module_fixes_the_output_folder_in_its_own_constants():
    """The regression that made the folder impossible to change: a module-level path built at import time."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("director.py", "h3_local.py"):
        text = open(os.path.join(here, "training", name), encoding="utf-8").read()
        assert '"outputs", "h3_local"' not in text, name
        for constant in ("CLIPS_DIR =", "CUTS_DIR =", "IMPORTS_DIR =", "THUMBS_DIR =", "OUT_DIR =", "STILLS_DIR ="):
            assert constant not in text, (name, constant)


def test_applying_the_folder_already_in_use_writes_nothing(settings, tmp_path, monkeypatch):
    folder = od.set_dir(str(tmp_path / "usb"))
    probes = []
    real = tempfile.NamedTemporaryFile
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda *a, **k: probes.append(1) or real(*a, **k))
    assert od.set_dir(folder) == folder  # the same folder again: no probe file (a sleeping USB disk costs seconds)
    assert probes == []
    od.set_dir(str(tmp_path / "other"))  # a different folder is still checked
    assert probes == [1]


def test_the_page_load_handler_does_not_read_the_output_drive():
    """output_dir_show runs on every F5: it may only read the setting, never list the output folder."""
    import ast

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(here, "training", "gui.py"), encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "output_dir_show")
    called = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert not called & {"desk_library_refresh", "_output_dir_view", "desk_library", "_metrics_choices"}


def test_the_gui_starts_on_the_default_folder_before_the_page_is_built():
    """Run as a script, gui.py resets the saved folder before `with gr.Blocks`; imported, it must not touch it."""
    import ast

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(here, "training", "gui.py"), encoding="utf-8").read())
    blocks = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.With) and "Blocks" in ast.dump(n.items[0]))
    resets = [
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.If)
        and "__main__" in ast.dump(n.test)
        and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "reset" for c in ast.walk(n))
    ]
    assert resets and resets[0] < blocks  # reset first, under the __main__ guard
