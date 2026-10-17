"""Ad key visuals: the app's adult characters (21 and over) in the 16 date outings, as 9:16 stills to pick from.

For the promo plan in projects/activity_promo/AUDIENCE.md. 7 characters x 16 outings x 2 compositions = 224 pictures, plus
2 more seeds of each character in the 6 priority outings = 84, so 308 in all. Every entry is a job for the z-image-colab
skill (which adds the age-safety negatives and keeps the cfg floor); this module only holds the data, writes the manifest,
files the finished pictures, measures them and archives the ones the owner picks. The tool never chooses a picture.

    check                      write projects/ad_keyvisuals/CATALOG.md and print the counts
    export --set pilot|all     write a manifest for the skill (spends nothing); --skip-done leaves out filed pictures
    import                     file the finished jobs under projects/ad_keyvisuals/images/<outing>/ and build index.html
    measure                    estimated age, same-character similarity across outings, second faces (InsightFace)
    picks --file picks.json    copy the pictures the owner chose (a JSON list of ids) to picked/

Z-Image here has no reference image: one character in 16 outings is the same *kind* of person, not the same face
(measured 0.39-0.55 similarity), and it is not the face of the app's avatar.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import asset_library as al
import concept_stills as cs
import generate_character as gc

REPO_ROOT = cs.REPO_ROOT
ROOT = os.path.join(REPO_ROOT, "projects", "ad_keyvisuals")
IMAGES_DIR = os.path.join(ROOT, "images")
PICKED_DIR = os.path.join(ROOT, "picked")
CATALOG_MD = os.path.join(ROOT, "CATALOG.md")
BIBLE_ROOT = al.BIBLE_ROOT
ZIMAGE_JOBS_DIR = al.ZIMAGE_JOBS_DIR

SEED_BASE = 4_000_000
REVISION_SEED_OFFSET = 100_000  # a redone picture's seed is the original plus this times its revision
REVISIONS = {}  # id -> revision number, filled when a picture is drawn again with a new seed
MIN_ESTIMATED_AGE = 20  # InsightFace reads high, so a lower estimate is a strong sign the face looks too young
SECOND_FACE = 2

# the seven adult characters the ads use (generate_character.CHARACTERS keys); the ages come from there
CHARACTERS = {
    "xinyi": "欣怡",
    "ruoxi": "若曦",
    "yuqing": "雨晴",
    "mylora": "思穎",
    "taeoh": "泰宇",
    "hyunjun": "賢哲",
    "jungi": "駿祺",
}
# a colour per character so they do not all wear the same thing (the outfits below take a {c})
COLOUR = {
    "xinyi": "soft pink",
    "ruoxi": "cream",
    "yuqing": "sage green",
    "mylora": "dusty blue",
    "taeoh": "charcoal",
    "hyunjun": "navy",
    "jungi": "olive",
}
# not used in ads: 19 and 18 are too young for an ad, 20 is under the owner's "21 and over"
EXCLUDED = {"mei": 19, "wanling": 18, "minjun": 18, "junho": 20}

COMPOSITIONS = {
    "a": "straight-on medium shot from the waist up, the person centred in the frame and looking at the camera",
    "b": "three-quarter view from a slightly different angle, a medium-wide shot that shows more of the place, "
    "the person glancing toward the camera",
}
LETTERS = ("a", "b", "c", "d")  # c and d repeat a and b with other seeds
COMPOSITION_OF = {"a": "a", "b": "b", "c": "a", "d": "b"}

PRIORITY = ("cooking", "coffee", "night_view", "picnic", "travel", "cat_cafe")
PILOT_OUTINGS = ("cooking", "night_view", "cat_cafe", "picnic")  # daytime, night, indoor with a prop, an animal

# 16 outings, in the order of ai-companion's outings.py. `moment` is the storyboard's high-point shot.
# Outfits cover the shoulders and chest; places have nobody else in them and no readable lettering.
OUTINGS = {
    "amusement_park": {
        "zh": "遊樂園",
        "moment": "holding a cotton candy close to her or his chest and smiling, a softly lit Ferris wheel behind, "
        "the park quiet and empty of people",
        "light": "dusk, warm golden light on the face with the first lights of the wheel glowing behind",
        "women": "a {c} knit cardigan over a white tee and a denim skirt",
        "men": "a {c} zip-up jacket over a white tee and jeans",
        "negative": "crowd, other people, text on signs",
    },
    "night_view": {
        "zh": "看夜景",
        "moment": "leaning on a railing at a hillside lookout, wind in the hair, a wide sea of city lights behind",
        "light": "blue hour, a soft lamp on the railing lighting the face clearly, the city lights glittering behind",
        "women": "a {c} coat over a plain knit top, a light scarf",
        "men": "a {c} jacket over a plain knit sweater",
        "negative": "other people, crowd, text on signs",
    },
    "cat_cafe": {
        "zh": "貓咪咖啡廳",
        "moment": "sitting on a window cushion at a small cafe table, one finger raised to the lips to ask for quiet, "
        "a sleeping orange tabby cat curled on the table",
        "light": "warm late-afternoon window light, a cosy interior",
        "women": "a {c} oversized sweater and a long skirt",
        "men": "a {c} sweater and light trousers",
        "negative": "other people, two cats, text on signs, extra fingers",
    },
    "travel": {
        "zh": "旅行",
        "moment": "sitting on a seawall at sunset wearing a straw hat, looking back over the shoulder, the sea behind",
        "light": "low golden sunset light, a warm glow on the face, the sea glittering",
        "women": "a {c} linen shirt and a light skirt",
        "men": "a {c} linen shirt and light trousers",
        "negative": "other people, crowd, text on signs",
    },
    "night_market": {
        "zh": "逛夜市",
        "moment": "holding a soft plush toy beside the face and smiling, a quiet corner at the end of a street of plain red "
        "paper lanterns with no writing",
        "light": "warm lantern light on the face, a deep blue night sky",
        "women": "a {c} hoodie and jeans",
        "men": "a {c} jacket over a tee and jeans",
        "negative": "crowd, other people, text, signboards, lettering on lanterns",
    },
    "cooking": {
        "zh": "下廚",
        "moment": "in a home kitchen wearing an apron, laughing and pointing at a steaming pot on the stove",
        "light": "warm early-evening kitchen light, a little steam in the air",
        "women": "a {c} apron over a plain long-sleeved top",
        "men": "a {c} apron over a plain long-sleeved shirt",
        "negative": "other people, text on packaging, extra fingers",
    },
    "coffee": {
        "zh": "喝咖啡",
        "moment": "sitting by a cafe window holding a coffee cup in both hands, looking up warmly",
        "light": "soft dusk window light on one side of the face, a warm interior",
        "women": "a {c} knit top and a long cardigan",
        "men": "a {c} shirt and a light jacket",
        "negative": "other people, text on the cup, text on signs",
    },
    "aquarium": {
        "zh": "水族館",
        "moment": "standing before a huge tank that glows blue, a few small fish far away in the water, pointing sideways "
        "at the tank with a delighted look",
        "light": "soft blue light from the tank on the face, the face still clearly lit",
        "women": "a {c} sweater and jeans",
        "men": "a {c} sweater and jeans",
        "negative": "other people, crowd, text on signs, fish close to the face",
    },
    "picnic": {
        "zh": "野餐",
        "moment": "sitting on a picnic blanket beside a wicker basket, hugging the knees, a calm lake behind, "
        "looking at the camera",
        "light": "golden afternoon sun, a warm glow, gentle shade",
        "women": "a {c} cotton blouse and a long skirt",
        "men": "a {c} cotton shirt and light trousers",
        "negative": "other people, crowd, text on the basket",
    },
    "photo": {
        "zh": "拍照",
        "moment": "holding a small vintage camera at the chest, wearing a canvas cap, in front of a plain pastel-coloured wall",
        "light": "good soft afternoon light, the face evenly lit",
        "women": "a {c} cap, a light shirt and a long skirt",
        "men": "a {c} cap, a light shirt and chinos",
        "negative": "other people, text on the wall, text on the camera, extra fingers",
    },
    "concert": {
        "zh": "演唱會",
        "moment": "holding a glow stick near the face and smiling, only blurred coloured stage lights behind, "
        "nobody else in the frame",
        "light": "soft coloured stage light on the face, the face clearly lit",
        "women": "a {c} t-shirt and a light jacket",
        "men": "a {c} t-shirt and a light jacket",
        "negative": "crowd, performers, other people, text, banners, screens",
    },
    "karaoke": {
        "zh": "唱歌",
        "moment": "sitting on a sofa in a private karaoke room holding a microphone, laughing",
        "light": "dim room with soft pink and blue ambient light, the face clearly lit",
        "women": "a {c} knit top and jeans",
        "men": "a {c} shirt and jeans",
        "negative": "other people, screen content, lyrics, text, extra fingers",
    },
    "movie": {
        "zh": "看電影",
        "moment": "sitting in a cinema seat with a plain paper popcorn bucket on the lap, turning with a laugh, "
        "the screen out of frame",
        "light": "dim theatre, soft flickering light from the side on the face",
        "women": "a {c} sweater and jeans",
        "men": "a {c} sweater and jeans",
        "negative": "movie screen, screen content, other people, text on the bucket, lettering",
    },
    "games": {
        "zh": "玩遊戲",
        "moment": "on a living-room sofa holding a plain black game controller, grinning as if about to win, "
        "the television out of frame",
        "light": "evening, a soft lamp and a cool glow from the television on the side of the face",
        "women": "a {c} hoodie and sweatpants",
        "men": "a {c} hoodie and sweatpants",
        "negative": "television, screen content, other people, text",
    },
    "shopping": {
        "zh": "逛街",
        "moment": "walking on a quiet shopping street holding a plain paper shopping bag against the chest, wearing a warm scarf",
        "light": "late-afternoon sun, soft warm light on the face",
        "women": "a {c} scarf, a light coat and a skirt",
        "men": "a {c} scarf, a light coat and trousers",
        "negative": "crowd, other people, lettering on shop windows, text on the bag, brand logos",
    },
    "exhibition": {
        "zh": "看展",
        "moment": "in a gallery room with abstract coloured light projected over the clothes, holding a blank postcard, "
        "a quiet look",
        "light": "soft gallery light with patches of violet and teal light from an installation, the face clearly lit",
        "women": "a {c} blouse and wide trousers",
        "men": "a {c} shirt and trousers",
        "negative": "paintings, artworks with figures, other people, text on walls, labels",
    },
}

PERSON_NEGATIVE = f"{al.PERSON_NEGATIVE}, two people, a second person, group photo, watermark, logo, readable text"


def look_for(key):
    """Describes the character in words, always an adult; the age comes from generate_character.CHARACTERS."""
    c = gc.CHARACTERS[key]
    age, woman = c["age"], c["gender"] == "woman"
    if age < 21:
        raise gc.UsageError(f"{key} 只有 {age} 歲，廣告只用 21 歲以上的角色")
    word = "early twenties" if age <= 23 else "mid-twenties" if age <= 26 else "late twenties"
    who = "woman" if woman else "man"
    pron = "her" if woman else "his"
    base = f"a young adult Taiwanese {who} in {pron} {word}" if age <= 26 else f"a Taiwanese {who} in {pron} {word}"
    skin = "with a youthful fresh face and smooth clear skin" if age <= 26 else "with a fresh smooth complexion"
    return f"{base} {skin}, {c['appearance']}"


def moment_text(outing, gender):
    """The storyboard's high-point text with the pronoun of this character."""
    text = outing["moment"]
    if gender == "woman":
        return text.replace("her or his", "her").replace("the person", "she")
    return text.replace("her or his", "his").replace("the person", "he")


def build_catalog():
    entries = []
    for key in CHARACTERS:
        if gc.CHARACTERS[key]["age"] < 21:
            raise gc.UsageError(f"{key} 不到 21 歲")
    index = 0
    for outing_id, outing in OUTINGS.items():
        for key, zh in CHARACTERS.items():
            c = gc.CHARACTERS[key]
            gender = c["gender"]
            outfit = outing["women" if gender == "woman" else "men"].format(c=COLOUR[key])
            letters = LETTERS if outing_id in PRIORITY else LETTERS[:2]
            for letter in letters:
                ident = f"kv-{key}-{outing_id}-{letter}"
                text = (
                    f"first-person point of view, {COMPOSITIONS[COMPOSITION_OF[letter]]}. "
                    f"{look_for(key)}, wearing {outfit}. {moment_text(outing, gender)}. "
                    f"Lighting: {outing['light']}"
                )
                entries.append(
                    {
                        "id": ident,
                        "character": key,
                        "name_zh": zh,
                        "age": c["age"],
                        "gender": gender,
                        "outing": outing_id,
                        "outing_zh": outing["zh"],
                        "letter": letter,
                        "composition": COMPOSITION_OF[letter],
                        "outfit": outfit,
                        "priority": outing_id in PRIORITY,
                        "text": text,
                        "negative": f"{PERSON_NEGATIVE}, {outing['negative']}",
                        "solo": True,
                        "seed": SEED_BASE + index + REVISION_SEED_OFFSET * REVISIONS.get(ident, 0),
                        "summary_zh": f"{zh} {c['age']} 歲｜{outing['zh']}｜{letter.upper()}",
                    }
                )
                index += 1
    return entries


def pilot_ids(entries=None):
    """28 pictures: every character in four outings (day, night, indoor with a prop, an animal), composition A."""
    entries = entries or build_catalog()
    ids = [e["id"] for e in entries if e["outing"] in PILOT_OUTINGS and e["letter"] == "a"]
    if len(ids) != len(CHARACTERS) * len(PILOT_OUTINGS):
        raise gc.UsageError("pilot 的張數不對")
    return ids


def job_spec(bible, entry):
    """One manifest entry for the skill; the style text comes from the shared bible, the age-safety negatives are the
    skill's job, as in asset_library.job_spec."""
    return al.job_spec(bible, entry)


def export(which="all", *, ids=None, batch=None, directory=None):
    bible = cs.load_bible(BIBLE_ROOT)
    entries = build_catalog()
    if ids is None:
        ids = pilot_ids(entries) if which == "pilot" else [e["id"] for e in entries]
    by_id = {e["id"]: e for e in entries}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise gc.UsageError(f"沒有這些主視覺：{unknown[:5]}")
    directory = directory or os.path.join(ROOT, "colab")
    batch = batch or f"keyvisuals-{which}"
    path = os.path.join(directory, f"{batch}.manifest.json")
    cs._write_json(path, {"batch": batch, "jobs": [job_spec(bible, by_id[i]) for i in ids]})
    return {"batch": batch, "manifest": path, "images": len(ids)}


def distribution(entries):
    return {
        "total": len(entries),
        "per_character": {k: sum(1 for e in entries if e["character"] == k) for k in CHARACTERS},
        "per_outing": {o: sum(1 for e in entries if e["outing"] == o) for o in OUTINGS},
        "per_composition": {c: sum(1 for e in entries if e["composition"] == c) for c in COMPOSITIONS},
    }


def catalog_markdown(entries):
    d = distribution(entries)
    lines = [
        "# 廣告主視覺目錄",
        "",
        "由 `training/ad_keyvisuals.py check` 產生，不要手改；要改請改程式裡的資料表再重跑。",
        "",
        f"共 **{d['total']} 張**：7 個角色 × 16 個約會情境 × 2 個取景 = {7 * 16 * 2}，"
        f"再加 6 個優先情境每個角色多 2 張不同 seed = {7 * 6 * 2}。全部 720×1280 直式，21 歲以上的角色。",
        "",
        "Z-Image 沒有參考圖：同一角色在不同情境是「同類型的人」，不保證同一張臉，也不是 app 角色頭像那張臉。",
        "",
        "## 角色",
        "",
        "| 角色 | 年齡 | 性別 | 外觀（讀自 `generate_character.CHARACTERS`） | 服裝主色 | 張數 |",
        "|---|---|---|---|---|---|",
    ]
    for key, zh in CHARACTERS.items():
        c = gc.CHARACTERS[key]
        lines.append(
            f"| {zh}（{key}） | {c['age']} | {'女' if c['gender'] == 'woman' else '男'} | {c['appearance']} | "
            f"{COLOUR[key]} | {d['per_character'][key]} |"
        )
    lines += [
        "",
        "不用的角色：" + "、".join(f"{k} {a} 歲" for k, a in EXCLUDED.items()) + "（廣告只用 21 歲以上）。",
        "",
        "## 16 個情境",
        "",
        "| 情境 | 主視覺（storyboard 的高光鏡頭） | 光線 | 女角服裝 | 男角服裝 | 優先 | 張數 |",
        "|---|---|---|---|---|---|---|",
    ]
    for oid, o in OUTINGS.items():
        lines.append(
            f"| {o['zh']} | {o['moment']} | {o['light']} | {o['women']} | {o['men']} | "
            f"{'✓' if oid in PRIORITY else ''} | {d['per_outing'][oid]} |"
        )
    lines += [
        "",
        "取景：A＝平視置中的半身；B＝三分之四側面、中遠景多拍一點場所；C、D＝A、B 換 seed（只有優先情境有）。",
        "",
        "## 試跑（28 張）",
        "",
        "7 個角色 × 4 個情境（" + "、".join(OUTINGS[o]["zh"] for o in PILOT_OUTINGS) + "），取景 A。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- importing finished jobs


def import_jobs(*, jobs_dir=ZIMAGE_JOBS_DIR, images_dir=IMAGES_DIR, force=False):
    entries = build_catalog()
    done = al.finished_jobs(jobs_dir)
    report = {"imported": [], "skipped": [], "missing": []}
    index = []
    for e in entries:
        job = done.get(e["id"])
        rel = os.path.join(e["outing"], f"{e['id']}.png")
        dest = os.path.join(images_dir, rel)
        if job is None and not os.path.isfile(dest):
            report["missing"].append(e["id"])
            continue
        if job is not None and (force or not os.path.isfile(dest)):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(job["output"], dest)
            report["imported"].append(e["id"])
        else:
            report["skipped"].append(e["id"])
        record = {
            k: e[k]
            for k in (
                "id",
                "character",
                "name_zh",
                "age",
                "gender",
                "outing",
                "outing_zh",
                "letter",
                "composition",
                "outfit",
                "seed",
                "summary_zh",
            )
        }
        record.update(
            {
                "image": rel.replace(os.sep, "/"),
                "prompt": job["prompt"] if job else None,
                "colab_job_id": job["job_id"] if job else None,
                "gpu": job.get("gpu") if job else None,
            }
        )
        index.append(record)
    root = os.path.dirname(images_dir)
    cs._write_json(os.path.join(root, "index.json"), {"count": len(index), "images": index})
    with open(os.path.join(root, "index.html"), "w", encoding="utf-8", newline="") as f:
        f.write(index_html(index))
    return report


def index_html(index):
    """A page to filter by outing, character and gender, click pictures to select them and copy the list as JSON.
    Choosing is the owner's; the page only records the clicks."""
    data = json.dumps(index, ensure_ascii=False)
    return f"""<!doctype html><meta charset="utf-8"><title>廣告主視覺</title>
<style>body{{font:14px sans-serif;margin:16px}}select,button{{margin-right:8px}}.g{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}}
figure{{margin:0;width:180px;border:3px solid transparent;border-radius:6px;cursor:pointer}}figure.on{{border-color:#d33}}
img{{width:100%;border-radius:4px;display:block}}figcaption{{font-size:11px;color:#555}}textarea{{width:100%;height:80px;margin-top:8px}}</style>
<h2>廣告主視覺 <span id=n></span> ／ 已選 <span id=s>0</span></h2>
<div id=f></div><button id=c>複製選取清單（JSON）</button><button id=x>清除選取</button>
<textarea id=t readonly placeholder="選好後按複製，貼回給 Claude"></textarea><div class=g id=g></div>
<script>
const D={data};
const F={{outing_zh:"情境",name_zh:"角色",gender:"性別",composition:"取景"}};
const S=new Set();
try{{JSON.parse(localStorage.getItem("kv-picks")||"[]").forEach(i=>S.add(i))}}catch(e){{}}
const f=document.getElementById("f");
for(const k in F){{const s=document.createElement("select");s.id=k;s.innerHTML="<option value=''>"+F[k]+"（全部）</option>"+
[...new Set(D.map(d=>d[k]).filter(Boolean))].map(v=>"<option>"+v+"</option>").join("");s.onchange=r;f.appendChild(s)}}
function save(){{document.getElementById("s").textContent=S.size;try{{localStorage.setItem("kv-picks",JSON.stringify([...S]))}}catch(e){{}}}}
function r(){{const w={{}};for(const k in F)w[k]=document.getElementById(k).value;
const L=D.filter(d=>Object.keys(w).every(k=>!w[k]||d[k]===w[k]));document.getElementById("n").textContent="（"+L.length+" 張）";
const g=document.getElementById("g");g.innerHTML="";
for(const d of L){{const e=document.createElement("figure");if(S.has(d.id))e.className="on";
e.innerHTML="<img loading=lazy src='images/"+d.image+"'><figcaption>"+d.summary_zh+"</figcaption>";
e.onclick=()=>{{S.has(d.id)?S.delete(d.id):S.add(d.id);e.classList.toggle("on");save()}};g.appendChild(e)}}save()}}
document.getElementById("c").onclick=()=>{{const t=document.getElementById("t");t.value=JSON.stringify([...S]);t.select();
try{{navigator.clipboard.writeText(t.value)}}catch(e){{document.execCommand("copy")}}}};
document.getElementById("x").onclick=()=>{{S.clear();r()}};
r();
</script>
"""


# ---------------------------------------------------------------- picks (the owner's choice, filed)


def picks(ids, *, images_dir=IMAGES_DIR, picked_dir=PICKED_DIR):
    """Copies the pictures the owner chose to picked/ and appends them to picked/PICKS.json. Unknown ids or pictures
    that are not filed yet are an error, so a typo does not silently drop a choice."""
    known = {e["id"]: e for e in build_catalog()}
    bad = [i for i in ids if i not in known]
    if bad:
        raise gc.UsageError(f"沒有這些主視覺：{bad[:5]}")
    missing = [i for i in ids if not os.path.isfile(os.path.join(images_dir, known[i]["outing"], f"{i}.png"))]
    if missing:
        raise gc.UsageError(f"這些還沒匯入圖：{missing[:5]}")
    os.makedirs(picked_dir, exist_ok=True)
    log_path = os.path.join(picked_dir, "PICKS.json")
    log = cs._read_json(log_path) if os.path.isfile(log_path) else {"picks": []}
    have = {p["id"] for p in log["picks"]}
    added = []
    for i in dict.fromkeys(ids):
        shutil.copy2(os.path.join(images_dir, known[i]["outing"], f"{i}.png"), os.path.join(picked_dir, f"{i}.png"))
        if i not in have:
            log["picks"].append({"id": i, "summary_zh": known[i]["summary_zh"], "seed": known[i]["seed"]})
            added.append(i)
    cs._write_json(log_path, log)
    return {"picked": len(log["picks"]), "added": added}


# ---------------------------------------------------------------- measuring (CPU, InsightFace)


def measure(describe, *, images_dir=IMAGES_DIR):
    """`describe(path) -> {"age", "embedding", "faces"} or None`. Reports the estimated age per character, the mean
    similarity between one character's pictures in different outings, and the pictures to look at first: no face found,
    a second face, or an estimated age under MIN_ESTIMATED_AGE."""
    ages, embeddings = {}, {}
    no_face, second_face, too_young = [], [], []
    for e in build_catalog():
        path = os.path.join(images_dir, e["outing"], f"{e['id']}.png")
        if not os.path.isfile(path):
            continue
        info = describe(path)
        if info is None:
            no_face.append(e["id"])
            continue
        ages.setdefault(e["character"], []).append(info["age"])
        embeddings.setdefault(e["character"], []).append(info["embedding"])
        if info.get("faces", 1) >= SECOND_FACE:
            second_face.append(e["id"])
        if info["age"] < MIN_ESTIMATED_AGE:
            too_young.append(e["id"])
    same = {}
    for key, vecs in embeddings.items():
        sims = [
            sum(x * y for x, y in zip(vecs[i], vecs[j], strict=True))
            for i in range(len(vecs))
            for j in range(i + 1, len(vecs))
        ]
        same[key] = {"pairs": len(sims), "mean": round(sum(sims) / len(sims), 3) if sims else None}
    return {
        "age_by_character": {
            k: {"n": len(v), "mean": round(sum(v) / len(v), 1), "min": min(v), "max": max(v)}
            for k, v in sorted(ages.items())
        },
        "same_character_similarity": same,
        "no_face": no_face,
        "second_face": second_face,
        "estimated_age_under_20": too_young,
    }


def _insightface_describe():
    sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
    import face_similarity as fs
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root=fs.INSIGHTFACE_ROOT,
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition", "genderage"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def describe(path):
        faces = app.get(fs._read_image_bgr(path))
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        return {
            "age": int(f.age),
            "embedding": [float(v) for v in f.normed_embedding],
            "faces": sum(1 for x in faces if x.det_score >= 0.6),
        }

    return describe


def main(argv=None):
    p = argparse.ArgumentParser(description="Ad key visuals: adult characters in the 16 date outings")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="print the counts and write CATALOG.md")
    e = sub.add_parser("export", help="write the manifest for the z-image-colab skill (spends nothing)")
    e.add_argument("--set", choices=("pilot", "all"), default="all", dest="which")
    e.add_argument("--batch")
    e.add_argument("--skip-done", action="store_true", help="leave out pictures already under images/")
    i = sub.add_parser("import", help="file the finished jobs under projects/ad_keyvisuals/images/")
    i.add_argument("--force", action="store_true")
    sub.add_parser("measure", help="age / similarity / second faces (needs insightface: ComfyUI's python)")
    k = sub.add_parser("picks", help="archive the pictures the owner chose")
    k.add_argument("--file", required=True, help="a JSON list of ids (what the index page copies)")
    args = p.parse_args(argv)
    if args.cmd == "check":
        entries = build_catalog()
        os.makedirs(ROOT, exist_ok=True)
        with open(CATALOG_MD, "w", encoding="utf-8", newline="") as f:
            f.write(catalog_markdown(entries) + "\n")
        print(json.dumps(dict(distribution(entries), catalog=CATALOG_MD), ensure_ascii=False))
    elif args.cmd == "export":
        ids = None
        if args.skip_done:
            pool = pilot_ids() if args.which == "pilot" else [x["id"] for x in build_catalog()]
            have = {os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(IMAGES_DIR, "*", "*.png"))}
            ids = [x for x in pool if x not in have]
        out = export(args.which, ids=ids, batch=args.batch)
        print(json.dumps(out, ensure_ascii=False))
        print(
            "下一步（submit 不花錢，worker.py up 會花 Colab CU）：\n"
            f"  ComfyUI\\.venv\\Scripts\\python.exe .claude\\skills\\z-image-colab\\scripts\\submit.py --manifest {out['manifest']}\n"
            "  ComfyUI\\.venv\\Scripts\\python.exe .claude\\skills\\z-image-colab\\scripts\\worker.py up"
        )
    elif args.cmd == "import":
        report = import_jobs(force=args.force)
        print(
            json.dumps(
                {k_: len(v) for k_, v in report.items()} | {"missing_ids": report["missing"][:10]}, ensure_ascii=False
            )
        )
    elif args.cmd == "picks":
        print(json.dumps(picks(cs._read_json(args.file)), ensure_ascii=False))
    else:
        print(json.dumps(measure(_insightface_describe()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
