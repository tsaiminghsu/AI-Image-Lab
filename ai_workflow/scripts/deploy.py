"""Put the platform into the workspace folder (the Google Drive folder, as synced to this machine).

    python ai_workflow/scripts/deploy.py --root "G:\\My Drive\\AI-Workflow"      (add --dry-run to only list)

It creates the workspace tree and copies the code and definitions: controller/, scripts/, notebooks/,
workflows/ and configs/platform.json. It never writes to models/, loras/, inputs/, outputs/, jobs/, cache/ or
logs/, and never deletes anything outside controller/ and scripts/ (which mirror the repository exactly, so a
removed module does not linger).

workflows/registry.json is merged, not replaced: entries from the repository win, and entries that exist only
in the workspace - workflows someone added there - are kept, together with their graph files.

Standard library only.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from controller.storage import LAYOUT  # noqa: E402

MIRRORED = ("controller", "scripts")  # exact copies: stale files are removed
COPIED = ("notebooks", "workflows")  # files are added or replaced, nothing is removed
SINGLE_FILES = ("configs/platform.json", "AGENTS.md", "README.md")
REGISTRY = "workflows/registry.json"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def source_files(folder):
    base = SOURCE / folder
    return sorted(p for p in base.rglob("*") if p.is_file() and not SKIP_DIRS.intersection(p.parts))


def merged_registry(target):
    ours = json.loads((SOURCE / REGISTRY).read_text(encoding="utf-8"))
    path = target / REGISTRY
    if path.is_file():
        try:
            theirs = json.loads(path.read_text(encoding="utf-8")).get("workflows", {})
        except ValueError:
            theirs = {}
        for name, spec in theirs.items():
            ours["workflows"].setdefault(name, spec)
    return json.dumps(ours, indent=2, ensure_ascii=False) + "\n"


def plan(target):
    """[(action, relative path)] - what a deploy would do, without doing it."""
    actions = [("mkdir", rel) for rel in LAYOUT if not (target / rel).is_dir()]
    for folder in MIRRORED + COPIED:
        for src in source_files(folder):
            rel = src.relative_to(SOURCE).as_posix()
            if rel == REGISTRY:
                continue
            dst = target / rel
            if not dst.is_file() or dst.read_bytes() != src.read_bytes():
                actions.append(("copy", rel))
    for folder in MIRRORED:
        base = target / folder
        if base.is_dir():
            keep = {p.relative_to(SOURCE).as_posix() for p in source_files(folder)}
            for dst in sorted(p for p in base.rglob("*") if p.is_file() and not SKIP_DIRS.intersection(p.parts)):
                rel = dst.relative_to(target).as_posix()
                if rel not in keep:
                    actions.append(("remove", rel))
    for rel in SINGLE_FILES:
        src, dst = SOURCE / rel, target / rel
        if src.is_file() and (not dst.is_file() or dst.read_bytes() != src.read_bytes()):
            actions.append(("copy", rel))
    registry = target / REGISTRY
    if not registry.is_file() or registry.read_text(encoding="utf-8") != merged_registry(target):
        actions.append(("merge", REGISTRY))
    return actions


def deploy(target, dry_run=False):
    target = Path(target)
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    actions = plan(target)
    if dry_run:
        return actions
    for action, rel in actions:
        dst = target / rel
        if action == "mkdir":
            dst.mkdir(parents=True, exist_ok=True)
        elif action == "copy":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / rel, dst)
        elif action == "remove":
            dst.unlink()
        elif action == "merge":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(merged_registry(target), encoding="utf-8", newline="\n")
    return actions


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deploy the workflow platform into the workspace folder.")
    parser.add_argument("--root", help="workspace folder; default: workspace_root in configs/local.json")
    parser.add_argument("--dry-run", action="store_true", help="list what would change, change nothing")
    args = parser.parse_args(argv)
    root = args.root
    if not root:
        local = SOURCE / "configs" / "local.json"
        if local.is_file():
            root = json.loads(local.read_text(encoding="utf-8")).get("workspace_root")
    if not root:
        parser.error("give --root, or set workspace_root in ai_workflow/configs/local.json")
    target = Path(root)
    if target.resolve() == SOURCE.resolve():
        parser.error("the workspace cannot be the repository's ai_workflow folder")
    if not target.parent.is_dir():
        parser.error("%s does not exist - is Google Drive for desktop running?" % target.parent)
    actions = deploy(target, args.dry_run)
    for action, rel in actions:
        print("%-6s %s" % (action, rel))
    print("%s%d change(s) in %s" % ("(dry run) " if args.dry_run else "", len(actions), target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
