"""A record of the pictures the Pony x Z-Image tab made, each with the prompts and settings that produced it.

The GUI's 📚 圖庫 tab shows them in a gallery; picking one shows its prompts, loads them back into the generator, and
marks it for a training set. export_dataset copies the marked pictures into a kohya-style folder (`<repeats>_<name>/`
with `<id>.png` and a `<id>.txt` caption next to it) for fine-tuning or LoRA training. Nothing here trains anything,
touches ComfyUI or imports gradio.

Pictures stay where they were written (training/reference_candidates); the record only points at them, so a picture
that was moved or deleted is simply left out of the gallery and of an export.
"""

import hashlib
import json
import os
import re
import shutil
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(REPO_ROOT, "outputs", "gallery", "records.json")
DATASETS_DIR = os.path.join(REPO_ROOT, "outputs", "lora_datasets")
CAPTION_KINDS = ("pony", "zimage", "original")  # booru tags / plain sentences / what the user typed (Chinese)
IMAGE_KINDS = ("final", "base")  # the Z-Image result / the Pony picture it started from
_LOCK = threading.Lock()


def _read(store):
    try:
        with open(store, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _write(store, records):
    os.makedirs(os.path.dirname(store), exist_ok=True)
    tmp = store + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    os.replace(tmp, store)


def record_id(base, final):
    return hashlib.sha1(os.path.abspath(final or base).encode("utf-8")).hexdigest()[:10]


def record(
    *,
    base,
    final,
    prompt_original,
    prompt_pony,
    prompt_zimage=None,
    negative="",
    seed,
    denoise,
    pony_checkpoint,
    tier,
    character=None,
    qwen=False,
    source="single",
    batch_dir=None,
    store=None,
    now=None,
):
    """Add (or replace: the same picture file written again) one record and return it. prompt_zimage is None when the
    same English text went to both models (the plain translation)."""
    if not (final or base):
        raise ValueError("a record needs at least one picture")
    rec = {
        "id": record_id(base, final),
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "source": source,
        "base": base,
        "final": final,
        "prompt_original": prompt_original,
        "prompt_pony": prompt_pony,
        "prompt_zimage": prompt_zimage,
        "negative": negative or "",
        "seed": int(seed),
        "denoise": denoise,
        "pony_checkpoint": pony_checkpoint,
        "tier": tier,
        "character": character,
        "qwen": bool(qwen),
        "batch_dir": batch_dir,
        "marked": False,
    }
    store = store or STORE
    with _LOCK:
        records = _read(store)
        for i, old in enumerate(records):
            if old.get("id") == rec["id"]:
                rec["marked"] = bool(old.get("marked"))  # a regenerated picture keeps its training mark
                records[i] = rec
                break
        else:
            records.append(rec)
        _write(store, records)
    return rec


def load(store=None, *, only_existing=True):
    """All records, newest first. With only_existing a record whose picture files are all gone is dropped."""
    records = _read(store or STORE)
    if only_existing:
        records = [r for r in records if any(p and os.path.isfile(p) for p in (r.get("final"), r.get("base")))]
    return sorted(records, key=lambda r: r.get("created", ""), reverse=True)


def find(rec_id, store=None):
    return next((r for r in _read(store or STORE) if r.get("id") == rec_id), None)


def set_marked(rec_id, flag, store=None):
    """Mark or unmark a record for the training set. Returns False when there is no such record."""
    store = store or STORE
    with _LOCK:
        records = _read(store)
        for r in records:
            if r.get("id") == rec_id:
                r["marked"] = bool(flag)
                _write(store, records)
                return True
    return False


def search(records, text="", only_marked=False):
    """Records whose prompts (any language) or checkpoint contain every word of `text`; optionally only the marked."""
    words = [w for w in (text or "").lower().split() if w]
    out = []
    for r in records:
        if only_marked and not r.get("marked"):
            continue
        hay = " ".join(
            str(r.get(k) or "")
            for k in ("prompt_original", "prompt_pony", "prompt_zimage", "pony_checkpoint", "character", "seed")
        ).lower()
        if all(w in hay for w in words):
            out.append(r)
    return out


def picture(rec, kind="final"):
    """The path to show or export for this record; falls back to the other picture when the asked one is missing."""
    order = ("final", "base") if kind == "final" else ("base", "final")
    for k in order:
        p = rec.get(k)
        if p and os.path.isfile(p):
            return p
    return None


def caption_for(rec, kind="pony", trigger=""):
    """The training caption: booru tags for a Pony LoRA, sentences for a Z-Image LoRA, or the Chinese original.
    A record from the plain translation has one English text only, which serves for both."""
    if kind not in CAPTION_KINDS:
        raise ValueError(f"caption kind must be one of {CAPTION_KINDS}")
    if kind == "original":
        text = rec.get("prompt_original") or ""
    elif kind == "zimage":
        text = rec.get("prompt_zimage") or rec.get("prompt_pony") or ""
    else:
        text = rec.get("prompt_pony") or ""
    text = " ".join(text.split())
    trigger = (trigger or "").strip()
    return f"{trigger}, {text}" if trigger and text else (trigger or text)


def _safe_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return cleaned or "dataset"


def export_dataset(
    ids, *, name, repeats=10, caption_kind="pony", image_kind="final", trigger="", dest_root=None, store=None, now=None
):
    """Copy the pictures of `ids` into <dest_root>/<name>_<time>/<repeats>_<name>/ as <id>.png + <id>.txt (the layout
    kohya's trainers read) and write export.json beside it. Returns {"folder", "dataset_dir", "written", "skipped"}.
    Raises ValueError when nothing could be written, so an empty folder is never left behind."""
    if caption_kind not in CAPTION_KINDS:
        raise ValueError(f"caption kind must be one of {CAPTION_KINDS}")
    if image_kind not in IMAGE_KINDS:
        raise ValueError(f"image kind must be one of {IMAGE_KINDS}")
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    name = _safe_name(name)
    wanted = {i for i in ids}
    records = [r for r in _read(store or STORE) if r.get("id") in wanted]
    jobs, skipped = [], []
    for rec in records:
        path = picture(rec, image_kind)
        caption = caption_for(rec, caption_kind, trigger)
        if not path:
            skipped.append({"id": rec["id"], "why": "圖檔不見了"})
        elif not caption:
            skipped.append({"id": rec["id"], "why": "沒有可用的 prompt 當說明文字"})
        else:
            jobs.append((rec, path, caption))
    if not jobs:
        raise ValueError("沒有可匯出的圖片" + (f"（略過 {len(skipped)} 張）" if skipped else ""))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    folder = os.path.join(dest_root or DATASETS_DIR, f"{name}_{stamp}")
    dataset_dir = os.path.join(folder, f"{repeats}_{name}")
    os.makedirs(dataset_dir, exist_ok=True)
    written = []
    for rec, path, caption in jobs:
        ext = os.path.splitext(path)[1] or ".png"
        shutil.copyfile(path, os.path.join(dataset_dir, f"{rec['id']}{ext}"))
        with open(os.path.join(dataset_dir, f"{rec['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(caption + "\n")
        written.append({"id": rec["id"], "seed": rec["seed"], "source_file": path, "caption": caption})
    manifest = {
        "name": name,
        "repeats": repeats,
        "caption_kind": caption_kind,
        "image_kind": image_kind,
        "trigger": (trigger or "").strip(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "written": written,
        "skipped": skipped,
    }
    with open(os.path.join(folder, "export.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return {"folder": folder, "dataset_dir": dataset_dir, "written": len(written), "skipped": skipped}


def import_batches(root, store=None):
    """Add the pictures of earlier batches (ponyz_batch_*/manifest.json under root) that are not recorded yet.
    Returns how many were added. Single pictures made before the record existed have no saved prompt to import."""
    store = store or STORE
    added = 0
    if not os.path.isdir(root):
        return 0
    known = {r.get("id") for r in _read(store)}
    for entry in sorted(os.listdir(root)):
        manifest = os.path.join(root, entry, "manifest.json")
        if not entry.startswith("ponyz_batch_") or not os.path.isfile(manifest):
            continue
        try:
            with open(manifest, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        for job in meta.get("jobs", []):
            english = job.get("english") or {}
            base, final = job.get("base"), job.get("final")
            if job.get("status") != "ok" or not (base or final) or not english.get("pony"):
                continue
            if record_id(base, final) in known:
                continue
            record(
                base=base,
                final=final,
                prompt_original=job.get("prompt") or "",
                prompt_pony=english["pony"],
                prompt_zimage=english.get("zimage"),
                seed=job["seed"],
                denoise=meta.get("denoise"),
                pony_checkpoint=meta.get("pony_checkpoint", ""),
                tier=meta.get("tier", ""),
                qwen=bool(meta.get("qwen")),
                source="batch",
                batch_dir=os.path.join(root, entry),
                store=store,
            )
            added += 1
    return added
