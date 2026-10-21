"""Where the 圖片→影片（本機） tab writes: pictures, clips, imported media, thumbnails and finished cuts.

The folder used to be fixed at <repo>/outputs/h3_local. It is now a setting the GUI can change and reset; every writer
asks current() at the moment it needs the folder (never at import time), so a change takes effect on the next job
without restarting the GUI. The choice lives in training/settings/output_dir.json (git-ignored: it is per machine).
Pure stdlib on purpose, like the other small settings helpers.
"""

import json
import os
import tempfile

import generate_character as gc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO_ROOT, "outputs", "h3_local")
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings", "output_dir.json")

# Model weights are shared by hard link with other tools (CLAUDE.md): never let outputs land among them.
_PROTECTED = (os.path.join(REPO_ROOT, "models"), os.path.join(REPO_ROOT, "ComfyUI", "models"))


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _read():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            value = json.load(f).get("path")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value.strip() else None


def current():
    """The configured folder, or the default when nothing valid is configured."""
    return os.path.abspath(_read() or DEFAULT_DIR)


def is_default():
    return _norm(current()) == _norm(DEFAULT_DIR)


def _clean(path):
    """The absolute form of path after the checks that write nothing; raises gc.UsageError saying why not."""
    text = os.path.expandvars(os.path.expanduser((path or "").strip().strip('"')))
    if not text:
        raise gc.UsageError("請輸入資料夾路徑，或按「恢復預設」")
    if not os.path.isabs(text):
        raise gc.UsageError(f"請輸入完整路徑（例如 D:\\Videos\\輸出），不要用相對路徑：{text}")
    full = os.path.abspath(text)
    if os.path.isfile(full):
        raise gc.UsageError(f"這是一個檔案，不是資料夾：{full}")
    for protected in _PROTECTED:
        root = _norm(protected)
        if _norm(full) == root or _norm(full).startswith(root + os.sep):
            raise gc.UsageError(f"不能放在模型資料夾裡（會跟模型檔混在一起）：{full}")
    return full


def validate(path):
    """The absolute form of path when a job could write there; raises gc.UsageError saying why not otherwise.
    Creates the folder (that is the only way to know it can be created) and removes the probe file again."""
    full = _clean(path)
    try:
        os.makedirs(full, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=full, prefix=".write_test_", delete=True):
            pass
    except OSError as exc:
        raise gc.UsageError(f"這個資料夾不能寫入：{full}（{exc.strerror or exc}）") from None
    return full


def set_dir(path):
    """Validate, remember and return the absolute folder. Files already written elsewhere are not moved.
    Pressing 套用 again on the folder already in use writes nothing: on an external hard disk that has spun
    down, even the write probe costs seconds."""
    full = _clean(path)
    if _read() is not None and _norm(full) == _norm(current()) and os.path.isdir(full):
        return full
    full = validate(path)
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"path": full}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return full


def reset():
    """Back to the default folder; returns it."""
    try:
        os.remove(SETTINGS_PATH)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise gc.UsageError(f"沒辦法清掉設定檔：{SETTINGS_PATH}（{exc.strerror or exc}）") from None
    return current()


# The sub-folders, derived from the current base each time they are asked for.
def clips_dir():
    return current()


def stills_dir():
    return os.path.join(current(), "stills")


def cuts_dir():
    return os.path.join(current(), "cuts")


def imports_dir():
    return os.path.join(current(), "imports")


def thumbs_dir():
    return os.path.join(current(), "thumbs")
