"""Where the workspace lives. Everything else addresses files by workspace-relative POSIX paths
("jobs/pending/x.json") through this interface, so a second backend is one new class.

Phase 1 has exactly one backend: a folder. On the compute runtime it is the mounted Google Drive
(/content/drive/MyDrive/AI-Workflow); on the local machine it is the same folder as synced by Google Drive
for desktop. Both sides therefore run the same code against the same tree.

Because two machines write into one synced tree, files are never edited in place by both: see the
single-writer rule in jobs.py. Writes here go to a temporary name and are renamed, so a reader (or the
sync client) never sees half a file under its final name.
"""

import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath

from . import WorkflowError

DEFAULT_RUNTIME_ROOT = "/content/drive/MyDrive/AI-Workflow"
ROOT_ENV = "AIWF_ROOT"

# The persistent workspace tree. deploy.py creates it; the worker creates whatever is missing.
LAYOUT = (
    "notebooks",
    "models",
    "loras",
    "workflows/image",
    "workflows/video",
    "workflows/test",
    "inputs",
    "outputs/images",
    "outputs/videos",
    "outputs/previews",
    "jobs/pending",
    "jobs/running",
    "jobs/completed",
    "jobs/failed",
    "jobs/cancel",
    "cache",
    "configs",
    "logs",
)
# Code and definitions: the only directories deploy.py writes into.
CODE_DIRS = ("controller", "scripts", "notebooks", "workflows", "configs")
TMP_MARK = ".tmp-"


def _clean(rel):
    path = PurePosixPath(str(rel).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise WorkflowError("INVALID_PATH", "not a workspace-relative path: %r" % (rel,))
    return path


class Storage:
    """The interface. Paths are workspace-relative, '/'-separated."""

    name = "base"

    def exists(self, rel):
        raise NotImplementedError

    def read_bytes(self, rel):
        raise NotImplementedError

    def write_bytes(self, rel, data):
        raise NotImplementedError

    def list(self, rel_dir, suffix=""):
        raise NotImplementedError

    def move(self, src, dst):
        raise NotImplementedError

    def delete(self, rel):
        raise NotImplementedError

    def makedirs(self, rel_dir):
        raise NotImplementedError

    def size(self, rel):
        raise NotImplementedError

    def mtime(self, rel):
        raise NotImplementedError

    def local_path(self, rel):
        """A real filesystem path for tools that need one (ComfyUI, ffprobe), or None if the backend has none."""
        return None

    # -- helpers built on the primitives ----------------------------------------------------------

    def read_text(self, rel):
        return self.read_bytes(rel).decode("utf-8")

    def write_text(self, rel, text):
        self.write_bytes(rel, text.encode("utf-8"))

    def read_json(self, rel, default=None):
        try:
            return json.loads(self.read_text(rel))
        except (OSError, ValueError):
            return default

    def write_json(self, rel, data):
        self.write_text(rel, json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")

    def copy_in(self, source, rel):
        self.write_bytes(rel, Path(source).read_bytes())

    def append_line(self, rel, line):
        try:
            text = self.read_text(rel)
        except OSError:
            text = ""
        self.write_text(rel, text + line + "\n")

    def ensure_layout(self):
        for rel in LAYOUT:
            self.makedirs(rel)


class DriveFolderStorage(Storage):
    name = "drive-folder"

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, rel):
        return self.root.joinpath(*_clean(rel).parts)

    def local_path(self, rel):
        return self._path(rel)

    def exists(self, rel):
        return self._path(rel).exists()

    def read_bytes(self, rel):
        return self._path(rel).read_bytes()

    def write_bytes(self, rel, data):
        path = self._path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("%s%s%d" % (path.name, TMP_MARK, os.getpid()))
        tmp.write_bytes(data)
        _replace(tmp, path)

    def list(self, rel_dir, suffix=""):
        """File names (not paths) directly inside rel_dir, sorted. In-flight temporary files are skipped."""
        path = self._path(rel_dir)
        if not path.is_dir():
            return []
        return sorted(
            p.name for p in path.iterdir() if p.is_file() and p.name.endswith(suffix) and TMP_MARK not in p.name
        )

    def move(self, src, dst):
        target = self._path(dst)
        target.parent.mkdir(parents=True, exist_ok=True)
        _replace(self._path(src), target)

    def delete(self, rel):
        try:
            self._path(rel).unlink()
        except FileNotFoundError:
            pass

    def makedirs(self, rel_dir):
        self._path(rel_dir).mkdir(parents=True, exist_ok=True)

    def size(self, rel):
        return self._path(rel).stat().st_size

    def mtime(self, rel):
        return self._path(rel).stat().st_mtime

    def append_line(self, rel, line):
        path = self._path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def copy_in(self, source, rel):
        """Copy a local file into the workspace under a temporary name, then rename."""
        path = self._path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("%s%s%d" % (path.name, TMP_MARK, os.getpid()))
        shutil.copyfile(str(source), str(tmp))
        _replace(tmp, path)


def _replace(src, dst):
    # Windows: os.replace fails while a reader (or the sync client) holds the target open for a moment.
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(src, dst)


def resolve_root(root=None, env=None):
    """The workspace folder: explicit argument, then AIWF_ROOT, then the folder this package was loaded from
    (the deployed copy sits inside the workspace), then configs/local.json next to the package (the
    repository copy, which is not a workspace itself), then the runtime's default mount path."""
    env = os.environ if env is None else env
    if root:
        return Path(root)
    if env.get(ROOT_ENV):
        return Path(env[ROOT_ENV])
    package_parent = Path(__file__).resolve().parents[1]
    if (package_parent / "jobs").is_dir() and (package_parent / "workflows").is_dir():
        return package_parent
    local = package_parent / "configs" / "local.json"
    if local.is_file():
        try:
            value = json.loads(local.read_text(encoding="utf-8")).get("workspace_root")
        except (OSError, ValueError):
            value = None
        if value:
            return Path(value)
    if Path(DEFAULT_RUNTIME_ROOT).is_dir():
        return Path(DEFAULT_RUNTIME_ROOT)
    raise WorkflowError(
        "WORKSPACE_NOT_FOUND",
        "The workspace folder is not configured.",
        hint='Pass --root, set %s, or write {"workspace_root": "..."} to configs/local.json.' % ROOT_ENV,
    )


def open_storage(root=None, env=None):
    path = resolve_root(root, env)
    if not path.is_dir():
        raise WorkflowError("WORKSPACE_NOT_FOUND", "The workspace folder %s does not exist." % path)
    return DriveFolderStorage(path)
