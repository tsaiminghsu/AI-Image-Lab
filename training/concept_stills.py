"""Concept stills for the AI short-drama pitches (projects/ai_short_drama/).

Script -> scene -> Z-Image candidates -> HUMAN REVIEW -> approved master image. This module stops at the
master image: it never touches H3, video or FFmpeg, and it never decides which candidate is "best" - a
person does that with `review`.

Source of truth is data, not code: projects/ai_short_drama/visual_bible.json (style, canvas, sampler) and
story_*/story.json (character bible + 7 scenes each). Prompts are assembled here, in one place, and go
through generate_character._build_prompt_and_negative like every other path, so the age-safety negatives
and the cfg floor cannot be dropped. The graph is the fixed Z-Image template (comfyui_client.
build_zimage_txt2img_workflow); only prompt, negative, seed, size, steps and cfg change.

Layout per scene (matches the brief):
    story_01_2347/scene_01/candidate_01.png   the image
                           candidate_01.json  metadata (prompt, negative, seed, size, steps, model, ...)
                           candidate_01.workflow.json   the filled graph, for exact re-generation
                           review.json        status history + decision
                           master.png         only after `review --approve`

Scene status: NO_CANDIDATES -> (CANDIDATES_READY ->) HUMAN_REVIEW -> APPROVED | REGENERATE. Candidates C/D
are generated only for a scene whose decision is REGENERATE.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import time

import comfyui_client as client
import generate_character as gc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.join(REPO_ROOT, "projects", "ai_short_drama")
MODEL_KEY = "z_image_turbo"
SCENES_PER_STORY = 7
LETTERS = "ABCD"
FIRST_ROUND = ("A", "B")
EXTRA_ROUND = ("C", "D")

NO_CANDIDATES = "NO_CANDIDATES"
CANDIDATES_READY = "CANDIDATES_READY"
HUMAN_REVIEW = "HUMAN_REVIEW"
APPROVED = "APPROVED"
REGENERATE = "REGENERATE"


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- loading and validation

class Story:
    def __init__(self, data, directory):
        self.data = data
        self.dir = directory
        self.id = data["story_id"]
        self.number = data["number"]

    def scene(self, number):
        for s in self.data["scenes"]:
            if s["scene"] == number:
                return s
        raise gc.UsageError(f"{self.id} 沒有 scene {number}")

    def scene_dir(self, number):
        return os.path.join(self.dir, f"scene_{number:02d}")


def load_bible(root=PROJECT_ROOT):
    return _read_json(os.path.join(root, "visual_bible.json"))


def load_stories(root=PROJECT_ROOT):
    stories = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "story.json")
        if os.path.isfile(path):
            stories.append(Story(_read_json(path), os.path.join(root, name)))
    return stories


def select_stories(stories, story_arg):
    """story_arg: None for all, else a story number (1-5) or story_id."""
    if story_arg is None:
        return stories
    picked = [s for s in stories if str(s.number) == str(story_arg) or s.id == story_arg]
    if not picked:
        raise gc.UsageError(f"找不到故事 {story_arg!r}；有 {', '.join(f'{s.number}={s.id}' for s in stories)}")
    return picked


def validate_story(story):
    """Raises gc.UsageError naming the first structural problem; returns None when the spec is usable."""
    d = story.data
    for key in ("story_id", "number", "title", "seed_base", "characters", "scenes"):
        if key not in d:
            raise gc.UsageError(f"{story.id}: story.json 缺少 {key}")
    scenes = d["scenes"]
    if [s.get("scene") for s in scenes] != list(range(1, SCENES_PER_STORY + 1)):
        raise gc.UsageError(f"{story.id}: 需要 scene 1..{SCENES_PER_STORY}，依序")
    for s in scenes:
        where = f"{story.id} scene {s.get('scene')}"
        for key in ("title", "beat_zh", "characters", "action", "setting", "lighting", "on_screen_text", "solo",
                    "compositions"):
            if key not in s:
                raise gc.UsageError(f"{where}: 缺少 {key}")
        for c in s["characters"]:
            if c not in d["characters"]:
                raise gc.UsageError(f"{where}: 角色 {c!r} 不在 character bible")
        if not isinstance(s["solo"], bool):
            raise gc.UsageError(f"{where}: solo 要是 true/false")
        for letter in FIRST_ROUND:
            if not s["compositions"].get(letter):
                raise gc.UsageError(f"{where}: compositions 缺少 {letter}")


# ---------------------------------------------------------------- prompts

def composition_for(bible, scene, letter):
    if letter in scene["compositions"]:
        return scene["compositions"][letter]
    return bible["extra_compositions"][letter]


def candidate_seed(story, scene_number, letter):
    return story.data["seed_base"] + scene_number * 10 + LETTERS.index(letter) + 1


def style_negative(bible):
    return f"{gc.REALISTIC_NEGATIVE}, {bible['style_negative_extra']}"


def scene_prompt_text(story, scene, composition):
    """The scene-specific part: no style suffix and no safety terms (those are added by
    _build_prompt_and_negative). The story's title is deliberately never included - Z-Image paints words."""
    chars = story.data["characters"]
    parts = [composition.rstrip(".")]
    if scene["characters"]:
        parts.append("Characters: " + "; ".join(chars[c]["description"] for c in scene["characters"]))
    parts.append(scene["action"].rstrip("."))
    parts.append("Setting: " + scene["setting"])
    parts.append("Lighting: " + scene["lighting"])
    if scene["on_screen_text"]:
        parts.append("Legible text in the image: " + scene["on_screen_text"])
    return ". ".join(parts)


def build_request(bible, story, scene_number, letter):
    """Everything one candidate needs, with no I/O. The final prompt/negative come from the shared
    assembly so the safety path is identical to every other generator."""
    scene = story.scene(scene_number)
    composition = composition_for(bible, scene, letter)
    scene_text = scene_prompt_text(story, scene, composition)
    prompt, negative = gc._build_prompt_and_negative(
        scene_text, scene.get("negative_extra"), "safe", None, bible["style_positive"], style_negative(bible),
        MODEL_KEY, solo=scene["solo"], allow_text=bool(scene["on_screen_text"]),
    )
    return {
        "story_id": story.id,
        "scene_id": f"scene_{scene_number:02d}",
        "candidate_id": f"candidate_{LETTERS.index(letter) + 1:02d}",
        "candidate_letter": letter,
        "composition": composition,
        "scene_prompt": scene_text,
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": candidate_seed(story, scene_number, letter),
        "width": bible["canvas"]["width"],
        "height": bible["canvas"]["height"],
        "steps": bible["sampler"]["steps"],
        "cfg": max(bible["sampler"]["cfg"], client.SAFETY_MIN_CFG),
        "model": MODEL_KEY,
        "workflow": bible["workflow"],
    }


def all_requests(bible, stories, letters):
    out = []
    for story in stories:
        for s in story.data["scenes"]:
            for letter in letters:
                out.append(build_request(bible, story, s["scene"], letter))
    return out


def write_prompts(bible, stories):
    """story_xx/prompts.json: the final prompt + negative + seed of every first-round candidate, so the
    prompts can be read and edited in the story file BEFORE any GPU time is spent."""
    for story in stories:
        reqs = all_requests(bible, [story], FIRST_ROUND)
        _write_json(os.path.join(story.dir, "prompts.json"), {
            "story_id": story.id, "visual_bible": "../visual_bible.json", "candidates": reqs,
        })
    return sum(len(all_requests(bible, [s], FIRST_ROUND)) for s in stories)


# ---------------------------------------------------------------- paths and state

def candidate_paths(story, scene_number, letter):
    stem = os.path.join(story.scene_dir(scene_number), f"candidate_{LETTERS.index(letter) + 1:02d}")
    return {"png": stem + ".png", "meta": stem + ".json", "workflow": stem + ".workflow.json"}


def review_path(story, scene_number):
    return os.path.join(story.scene_dir(scene_number), "review.json")


def load_review(story, scene_number):
    path = review_path(story, scene_number)
    if os.path.isfile(path):
        return _read_json(path)
    return {"story_id": story.id, "scene_id": f"scene_{scene_number:02d}", "status": NO_CANDIDATES,
            "approved_candidate": None, "notes": "", "history": []}


def _set_status(story, scene_number, status, **fields):
    review = load_review(story, scene_number)
    review["status"] = status
    review.update(fields)
    review["history"].append({"status": status, "at": _now()})
    _write_json(review_path(story, scene_number), review)
    return review


def existing_letters(story, scene_number):
    return [x for x in LETTERS if os.path.isfile(candidate_paths(story, scene_number, x)["png"])]


# ---------------------------------------------------------------- generation

def generate(bible, stories, *, letters=FIRST_ROUND, scene_numbers=None, force=False, dry_run=False,
             submit=None, server_up=client.is_server_running):
    """Render candidates. Skips existing images (unless force). Letters C/D are refused for a scene whose
    review is not REGENERATE, so the first round can never silently grow to four per scene."""
    submit = submit or client.submit_txt2img_generation_zimage
    letters = tuple(letters)
    todo = []
    for story in stories:
        validate_story(story)
        for s in story.data["scenes"]:
            n = s["scene"]
            if scene_numbers and n not in scene_numbers:
                continue
            for letter in letters:
                if letter in EXTRA_ROUND and load_review(story, n)["status"] != REGENERATE:
                    raise gc.UsageError(
                        f"{story.id} scene {n}: 候選 {letter} 只在人工審核判定 REGENERATE 之後才生成"
                        f"（目前 {load_review(story, n)['status']}）。先跑 review --regenerate")
                if os.path.isfile(candidate_paths(story, n, letter)["png"]):
                    if not force:
                        continue
                    if load_review(story, n)["status"] == APPROVED:
                        raise gc.UsageError(f"{story.id} scene {n} 已 APPROVED，--force 會讓 master.png 對不上候選；"
                                            "先 review --regenerate")
                todo.append((story, n, letter))
    if dry_run:
        for story, n, letter in todo:
            print(f"[dry-run] {story.id} scene_{n:02d} candidate {letter} seed {candidate_seed(story, n, letter)}")
        print(f"[dry-run] {len(todo)} 張待生成（未動用 GPU）")
        return 0
    if todo and not server_up():
        raise gc.UsageError("ComfyUI 沒有在跑。先啟動它（見 CLAUDE.md），再 generate")
    made = 0
    touched = []
    for story, n, letter in todo:
        req = build_request(bible, story, n, letter)
        paths = candidate_paths(story, n, letter)
        os.makedirs(os.path.dirname(paths["png"]), exist_ok=True)
        prefix = f"{story.id}_s{n:02d}_{letter}"
        print(f"{story.id} scene_{n:02d} {letter}: seed {req['seed']} {req['width']}x{req['height']}", flush=True)
        graph = client.build_zimage_txt2img_workflow(
            req["prompt"], req["negative_prompt"], req["seed"], prefix, client.ZIMAGE_MODELS[MODEL_KEY],
            req["width"], req["height"], req["steps"], req["cfg"])
        started = time.time()
        raw = submit(prompt=req["prompt"], negative_prompt=req["negative_prompt"], seed=req["seed"],
                     filename_prefix=prefix, model=MODEL_KEY, width=req["width"], height=req["height"],
                     steps=req["steps"], cfg=req["cfg"])
        elapsed = round(time.time() - started, 1)
        shutil.move(raw, paths["png"])
        meta = {k: req[k] for k in ("story_id", "scene_id", "candidate_id", "prompt", "negative_prompt", "seed",
                                    "width", "height", "steps", "model", "workflow")}
        meta.update({
            "cfg": req["cfg"], "composition": req["composition"], "candidate_letter": letter,
            "sampler": client.ZIMAGE_SAMPLER, "scheduler": client.ZIMAGE_SCHEDULER, "shift": client.ZIMAGE_SHIFT,
            "workflow_graph": os.path.basename(paths["workflow"]),
            "workflow_template_sha256": _sha256(client.WORKFLOW_TEMPLATE_TXT2IMG_ZIMAGE_PATH),
            "image_sha256": _sha256(paths["png"]), "generation_seconds": elapsed, "created_at": _now(),
        })
        _write_json(paths["meta"], meta)
        _write_json(paths["workflow"], graph)
        print(f"  saved {paths['png']} ({elapsed}s)", flush=True)
        made += 1
        if (story.id, n) not in touched:
            touched.append((story.id, n))
    by_id = {s.id: s for s in stories}
    for story_id, n in touched:
        _mark_ready(by_id[story_id], n)
    return made


def _mark_ready(story, scene_number):
    """A scene holding its first-round candidates goes CANDIDATES_READY -> HUMAN_REVIEW: a person decides."""
    if all(x in existing_letters(story, scene_number) for x in FIRST_ROUND):
        _set_status(story, scene_number, CANDIDATES_READY)
        _set_status(story, scene_number, HUMAN_REVIEW)


# ---------------------------------------------------------------- review

def review(story, scene_number, *, approve=None, regenerate=False, notes=""):
    """Record a human decision. approve = candidate number or letter; copies it to master.png."""
    if bool(approve) == bool(regenerate):
        raise gc.UsageError("review 要二選一：--approve <候選> 或 --regenerate")
    story.scene(scene_number)
    if regenerate:
        if not existing_letters(story, scene_number):
            raise gc.UsageError("這個 scene 還沒有候選圖，沒有東西可以判定 REGENERATE")
        return _set_status(story, scene_number, REGENERATE, approved_candidate=None, notes=notes)
    letter = str(approve).upper()
    if letter.isdigit():
        letter = LETTERS[int(letter) - 1] if 1 <= int(letter) <= len(LETTERS) else ""
    if len(letter) != 1 or letter not in LETTERS:
        raise gc.UsageError(f"候選要是 1-4 或 A-D，不是 {approve!r}")
    paths = candidate_paths(story, scene_number, letter)
    if not os.path.isfile(paths["png"]):
        raise gc.UsageError(f"沒有這張候選圖：{paths['png']}")
    master = os.path.join(story.scene_dir(scene_number), "master.png")
    shutil.copy2(paths["png"], master)
    shutil.copy2(paths["meta"], os.path.join(story.scene_dir(scene_number), "master.json"))
    return _set_status(story, scene_number, APPROVED, approved_candidate=os.path.basename(paths["png"]),
                       notes=notes)


# ---------------------------------------------------------------- index, sheet, status

def build_index(stories, root=PROJECT_ROOT):
    rows = []
    for story in stories:
        for s in story.data["scenes"]:
            n = s["scene"]
            r = load_review(story, n)
            cands = []
            for letter in existing_letters(story, n):
                p = candidate_paths(story, n, letter)
                meta = _read_json(p["meta"]) if os.path.isfile(p["meta"]) else {}
                cands.append({"candidate_id": f"candidate_{LETTERS.index(letter) + 1:02d}", "image": os.path.relpath(
                    p["png"], root).replace("\\", "/"), "seed": meta.get("seed"), "metadata": os.path.relpath(
                    p["meta"], root).replace("\\", "/")})
            rows.append({"story_id": story.id, "scene_id": f"scene_{n:02d}", "title": s["title"],
                         "status": r["status"], "approved_candidate": r.get("approved_candidate"),
                         "candidates": cands})
    index = {"generated_at": _now(), "scenes": rows}
    _write_json(os.path.join(root, "candidate_index.json"), index)
    return index


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def build_sheet(bible, stories, root=PROJECT_ROOT):
    """review_sheet.html: every scene with its candidates side by side, the beat, the on-screen text to
    verify and the review checklist. Static, relative image links - open it in a browser."""
    checks = "".join(f"<li>☐ {_esc(c)}</li>" for c in bible["review_checklist"])
    video = "".join(f"<li>{_esc(q)}</li>" for q in bible["video_potential_questions"])
    out = ["<!doctype html><html lang=\"zh-Hant\"><head><meta charset=\"utf-8\"><title>Concept stills review</title>",
           "<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}h1,h2{margin:.6em 0 .2em}"
           ".scene{border-top:1px solid #444;padding:10px 0}.row{display:flex;gap:10px;flex-wrap:wrap}"
           ".c{width:260px}.c img{width:260px;display:block}.c small{color:#aaa;word-break:break-all}"
           ".st{display:inline-block;padding:1px 8px;border-radius:8px;background:#444}"
           ".APPROVED{background:#2a7a3a}.REGENERATE{background:#a33}.HUMAN_REVIEW{background:#a77d12}"
           "code{color:#9cf}ul{margin:.2em 0}</style></head><body><h1>Concept stills — human review</h1>",
           f"<p>Checklist per scene:</p><ul>{checks}</ul><p>Image→video risk:</p><ul>{video}</ul>",
           "<p>Decide with <code>concept_stills.py review --story N --scene M --approve 1|2|3|4</code> or "
           "<code>--regenerate</code>. The tool never picks a best image.</p>"]
    for story in stories:
        out.append(f"<h2>{_esc(story.data['title_zh'])} · {_esc(story.data['genre'])}</h2>")
        out.append(f"<p>{_esc(story.data['hook'])}</p>")
        for s in story.data["scenes"]:
            n = s["scene"]
            r = load_review(story, n)
            out.append(f"<div class=\"scene\"><b>{story.number}-{n:02d} {_esc(s['title'])}</b> "
                       f"<span class=\"st {r['status']}\">{r['status']}</span>")
            out.append(f"<div>{_esc(s['beat_zh'])}</div>")
            if s["on_screen_text"]:
                out.append(f"<div>畫面文字要核對：{_esc(s['on_screen_text'])}</div>")
            if s.get("video_note"):
                out.append(f"<div>影片備註：{_esc(s['video_note'])}</div>")
            out.append("<div class=\"row\">")
            for letter in existing_letters(story, n):
                p = candidate_paths(story, n, letter)
                rel = os.path.relpath(p["png"], root).replace("\\", "/")
                meta = _read_json(p["meta"]) if os.path.isfile(p["meta"]) else {}
                out.append(f"<div class=\"c\"><a href=\"{_esc(rel)}\"><img src=\"{_esc(rel)}\" loading=\"lazy\">"
                           f"</a><small>{letter} · seed {_esc(meta.get('seed'))}</small></div>")
            out.append("</div></div>")
    out.append("</body></html>")
    path = os.path.join(root, "review_sheet.html")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    return path


def status_lines(stories):
    lines = []
    for story in stories:
        counts = {}
        for s in story.data["scenes"]:
            st = load_review(story, s["scene"])["status"]
            counts[st] = counts.get(st, 0) + 1
        images = sum(len(existing_letters(story, s["scene"])) for s in story.data["scenes"])
        lines.append(f"{story.id:<28} images {images:>2}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return lines


# ---------------------------------------------------------------- CLI

def _parse_letters(text):
    letters = tuple(x.strip().upper() for x in text.split(",") if x.strip())
    bad = [x for x in letters if x not in LETTERS]
    if bad or not letters:
        raise gc.UsageError(f"--candidates 要是 A,B,C,D 的子集，不是 {text!r}")
    return letters


def _parse_scenes(text):
    if not text:
        return None
    try:
        return {int(x) for x in text.split(",")}
    except ValueError:
        raise gc.UsageError(f"--scene 要是數字或逗號分隔，不是 {text!r}") from None


def main(argv=None):
    p = argparse.ArgumentParser(description="Concept stills for the AI short-drama pitches")
    p.add_argument("--root", default=PROJECT_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="validate the specs, create scene folders, write prompts.json and the index")
    sub.add_parser("prompts", help="rewrite story_*/prompts.json (no GPU)")
    g = sub.add_parser("generate", help="render candidates through ComfyUI (GPU) - ComfyUI must already be running")
    g.add_argument("--story")
    g.add_argument("--scene")
    g.add_argument("--candidates", default="A,B")
    g.add_argument("--force", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    sub.add_parser("index", help="rebuild candidate_index.json")
    sub.add_parser("sheet", help="write review_sheet.html")
    sub.add_parser("status", help="per-story image and review counts")
    r = sub.add_parser("review", help="record the human decision for one scene")
    r.add_argument("--story", required=True)
    r.add_argument("--scene", type=int, required=True)
    r.add_argument("--approve", help="candidate number 1-4 or letter A-D -> copied to master.png")
    r.add_argument("--regenerate", action="store_true")
    r.add_argument("--notes", default="")
    args = p.parse_args(argv)

    root = args.root
    bible = load_bible(root)
    stories = load_stories(root)
    for s in stories:
        validate_story(s)
    if args.cmd == "init":
        for s in stories:
            for scene in s.data["scenes"]:
                os.makedirs(s.scene_dir(scene["scene"]), exist_ok=True)
        n = write_prompts(bible, stories)
        build_index(stories, root)
        print(f"{len(stories)} 個故事、{sum(len(s.data['scenes']) for s in stories)} 個 scene、{n} 條第一輪 prompt 已寫入")
    elif args.cmd == "prompts":
        print(f"{write_prompts(bible, stories)} 條 prompt 已寫入")
    elif args.cmd == "generate":
        picked = select_stories(stories, args.story)
        made = generate(bible, picked, letters=_parse_letters(args.candidates), scene_numbers=_parse_scenes(args.scene),
                        force=args.force, dry_run=args.dry_run)
        if not args.dry_run:
            build_index(stories, root)
            print(f"生成 {made} 張；下一步：打開 review_sheet.html 人工審核（先跑 sheet）")
    elif args.cmd == "index":
        build_index(stories, root)
        print("candidate_index.json 已更新")
    elif args.cmd == "sheet":
        print(build_sheet(bible, stories, root))
    elif args.cmd == "status":
        print("\n".join(status_lines(stories)))
    elif args.cmd == "review":
        story = select_stories(stories, args.story)[0]
        result = review(story, args.scene, approve=args.approve, regenerate=args.regenerate, notes=args.notes)
        build_index(stories, root)
        print(f"{story.id} scene_{args.scene:02d}: {result['status']}")


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
