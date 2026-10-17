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
import time

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

# Faces beyond the app's seven characters (2026-10-11, the owner found the same few faces everywhere). Fictional adults
# written for the ads only, not app characters: a different hair, face, build and a glasses / mole / freckles each, so
# the text encoder has something else to draw. Same age wording rules as the characters (21 and over, youthful).
EXTRA_FACES = {
    "w1": (
        "新女1",
        "woman",
        22,
        "Taiwanese",
        "short curly dark brown hair with curtain bangs, round glasses, soft round face, petite build",
        "lavender",
    ),
    "w2": (
        "新女2",
        "woman",
        24,
        "Taiwanese",
        "long straight black hair with a side part, a small mole beside the lip, narrow face, slender build",
        "mustard",
    ),
    "w3": (
        "新女3",
        "woman",
        25,
        "Japanese",
        "messy half-up bun with loose strands, light freckles across the nose, warm brown eyes, average build",
        "terracotta",
    ),
    "w4": (
        "新女4",
        "woman",
        23,
        "Korean",
        "shoulder-length hair dyed soft brown with see-through bangs, strong brows, square jaw, tall build",
        "sky blue",
    ),
    "w5": (
        "新女5",
        "woman",
        26,
        "Hong Kong Chinese",
        "low ponytail over one shoulder, thin-rimmed glasses, high cheekbones, athletic build",
        "burgundy",
    ),
    "w6": (
        "新女6",
        "woman",
        21,
        "Taiwanese",
        "twin low braids, wide-set eyes, full cheeks, tanned skin, sturdy build",
        "coral",
    ),
    "m1": (
        "新男1",
        "man",
        22,
        "Taiwanese",
        "short wavy hair with a centre part, round glasses, soft jaw, lean build",
        "beige",
    ),
    "m2": (
        "新男2",
        "man",
        24,
        "Taiwanese",
        "buzzed sides and a textured top, light stubble, strong brows, broad shoulders",
        "forest green",
    ),
    "m3": (
        "新男3",
        "man",
        25,
        "Japanese",
        "medium-length black hair tucked behind the ears, long narrow face, gentle eyes, slim build",
        "rust",
    ),
    "m4": (
        "新男4",
        "man",
        23,
        "Korean",
        "neat side-swept dark brown hair, a mole under one eye, defined jaw, tall build",
        "slate grey",
    ),
    "m5": (
        "新男5",
        "man",
        26,
        "Hong Kong Chinese",
        "short crop with a faint fade, thick eyebrows, square face, stocky build",
        "denim blue",
    ),
    "m6": (
        "新男6",
        "man",
        21,
        "Taiwanese",
        "curly dark hair, round face, dimples, tanned skin, medium build",
        "mustard",
    ),
}
EXTRA_SEED_BASE = SEED_BASE + 1000

COMPOSITIONS = {
    "a": "straight-on medium shot from the waist up, the person centred in the frame and looking at the camera with a soft warm smile",
    "b": "three-quarter view from a slightly different angle, a medium-wide shot that shows more of the place, "
    "the person glancing toward the camera with a soft warm smile",
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
        "women": "a {c} cotton blouse and long wide trousers",
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


def describe_person(age, gender, nationality, appearance):
    """The words for an adult of 21 or over. Measured 2026-10-10: with "mature adult features" in the appearance, the
    28-year-old read 40-50 (InsightFace mean 41.8) while the others read 23-26, so that phrase is dropped and everyone
    gets the same youthful wording."""
    if age < 21:
        raise gc.UsageError(f"廣告只用 21 歲以上的角色，這個只有 {age} 歲")
    word = "early twenties" if age <= 23 else "mid-twenties" if age <= 26 else "late twenties"
    who, pron = ("woman", "her") if gender == "woman" else ("man", "his")
    appearance = appearance.replace("mature adult features, ", "")
    return (
        f"a young adult {nationality} {who} in {pron} {word} with a youthful fresh face and smooth clear skin, "
        f"{appearance}"
    )


def look_for(key):
    """Describes an app character in words; the age and appearance come from generate_character.CHARACTERS."""
    c = gc.CHARACTERS[key]
    if c["age"] < 21:
        raise gc.UsageError(f"{key} 只有 {c['age']} 歲，廣告只用 21 歲以上的角色")
    return describe_person(c["age"], c["gender"], "Taiwanese", c["appearance"])


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


def build_extra_catalog():
    """The new faces: one picture (composition A) per face and outing, seeds after the 308 so those never move."""
    entries = []
    index = 0
    for outing_id, outing in OUTINGS.items():
        for key, (zh, gender, age, nationality, appearance, colour) in EXTRA_FACES.items():
            outfit = outing["women" if gender == "woman" else "men"].format(c=colour)
            ident = f"kv-{key}-{outing_id}-a"
            text = (
                f"first-person point of view, {COMPOSITIONS['a']}. "
                f"{describe_person(age, gender, nationality, appearance)}, wearing {outfit}. "
                f"{moment_text(outing, gender)}. Lighting: {outing['light']}"
            )
            entries.append(
                {
                    "id": ident,
                    "character": key,
                    "name_zh": zh,
                    "age": age,
                    "gender": gender,
                    "outing": outing_id,
                    "outing_zh": outing["zh"],
                    "letter": "a",
                    "composition": "a",
                    "outfit": outfit,
                    "priority": False,
                    "extra": True,
                    "text": text,
                    "negative": f"{PERSON_NEGATIVE}, {outing['negative']}",
                    "solo": True,
                    "seed": EXTRA_SEED_BASE + index + REVISION_SEED_OFFSET * REVISIONS.get(ident, 0),
                    "summary_zh": f"{zh} {age} 歲｜{outing['zh']}｜A",
                }
            )
            index += 1
    return entries


def build_all():
    return build_catalog() + build_extra_catalog()


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
    entries = build_all()
    if ids is None:
        base = build_catalog()
        ids = (
            pilot_ids(base)
            if which == "pilot"
            else [e["id"] for e in build_extra_catalog()]
            if which == "faces"
            else [e["id"] for e in base]
        )
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
        f"## 新面孔（{len(EXTRA_FACES)} 位 × 16 情境 = {len(EXTRA_FACES) * len(OUTINGS)} 張，取景 A）",
        "",
        "不是 app 的角色，是為廣告另外寫的虛構成人（21 歲以上），髮型、臉型、眼鏡或痣各不相同；"
        "`export --set faces` 產生清單。",
        "",
        "| 代號 | 年齡 | 性別 | 國籍措辭 | 外觀 | 服裝主色 |",
        "|---|---|---|---|---|---|",
    ]
    for key, (zh, gender, age, nat, look, colour) in EXTRA_FACES.items():
        lines.append(f"| {zh}（{key}） | {age} | {'女' if gender == 'woman' else '男'} | {nat} | {look} | {colour} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- importing finished jobs


def import_jobs(*, jobs_dir=ZIMAGE_JOBS_DIR, images_dir=IMAGES_DIR, force=False):
    entries = build_all()
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


# ---------------------------------------------------------------- scripts per outing + the page to choose candidates

STORYBOARDS_MD = os.path.join(REPO_ROOT, "projects", "activity_promo", "STORYBOARDS.md")
SCRIPTS_MD = os.path.join(ROOT, "SCRIPTS.md")
SELECT_HTML = os.path.join(ROOT, "select.html")
# the storyboard heading of each outing ("## 3. 一起去貓咪咖啡廳"): the outing's Chinese name appears in it
STORYBOARD_KEY = {
    "amusement_park": "遊樂園",
    "night_view": "夜景",
    "cat_cafe": "貓咪咖啡廳",
    "travel": "旅行",
    "night_market": "夜市",
    "cooking": "下廚",
    "coffee": "喝咖啡",
    "aquarium": "水族館",
    "picnic": "野餐",
    "photo": "拍照",
    "concert": "演唱會",
    "karaoke": "唱歌",
    "movie": "電影",
    "games": "遊戲",
    "shopping": "逛街",
    "exhibition": "看展",
}
# the storyboard shot each outing's key visual is closest to (not always the "高" shot: the cotton candy is shot 3, the
# picnic blanket shot 2, ...); the first frame of that shot can be taken from the chosen picture
VISUAL_SHOT = {
    "amusement_park": 3, "night_view": 5, "cat_cafe": 5, "travel": 5, "night_market": 5, "cooking": 4,
    "coffee": 4, "aquarium": 4, "picnic": 2, "photo": 3, "concert": 2, "karaoke": 1,
    "movie": 4, "games": 1, "shopping": 6, "exhibition": 5,
}  # fmt: skip
PURPOSE_ZH = {"開": "開場", "選": "你的選擇", "過": "過場", "高": "高光", "收": "收尾"}


def parse_storyboards(path=STORYBOARDS_MD):
    """outing id -> {"title", "meta", "shots": [{n, purpose, scene, line, choice}], "cuts"} from STORYBOARDS.md."""
    import re

    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"\n(?=## \d+\. )", text)
    out = {}
    for block in blocks:
        head = re.match(r"## \d+\. (.+)", block)
        if not head:
            continue
        for outing_id, key in STORYBOARD_KEY.items():
            if key in head.group(1) and outing_id not in out:
                shots = []
                for line in block.splitlines():
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    if len(cells) == 5 and cells[0].isdigit():
                        shots.append(
                            {
                                "n": int(cells[0]),
                                "purpose": cells[1],
                                "scene": cells[2],
                                "line": cells[3],
                                "choice": cells[4],
                            }
                        )
                meta = next((ln for ln in block.splitlines() if ln.startswith("地標")), "")
                cuts = next((ln for ln in block.splitlines() if ln.startswith("短版")), "")
                out[outing_id] = {
                    "title": head.group(1).replace("（已完成）", ""),
                    "meta": meta,
                    "shots": shots,
                    "cuts": cuts,
                }
    missing = [o for o in OUTINGS if o not in out]
    if missing:
        raise gc.UsageError(f"STORYBOARDS.md 找不到這些情境的分鏡：{missing}")
    return out


def scripts_markdown(boards, counts):
    lines = [
        "# 各情境的劇本與候選圖",
        "",
        "由 `training/ad_keyvisuals.py scripts` 產生，不要手改。劇本取自 `projects/activity_promo/STORYBOARDS.md`"
        "（6 個鏡頭 × 5 秒，台詞是 app 的原句）；候選圖就是各情境的主視覺，對應劇本裡標「◀ 主視覺」的那個鏡頭。",
        "挑圖請開 `select.html`（每個情境一段：劇本加候選圖，點圖勾選，複製 JSON）。",
        "",
        "注意：劇本台詞寫的是「她」；挑男角時，台詞照用，只是代名詞換成他。主視覺只有「高」鏡頭那一格，"
        "其餘 5 個鏡頭的首幀還沒生，選定人物後再用同一個人物描述去生。",
        "",
    ]
    for outing_id, outing in OUTINGS.items():
        b = boards[outing_id]
        lines += [f"## {outing['zh']}（{counts.get(outing_id, 0)} 張候選）", "", b["meta"], ""]
        lines += ["| # | 用途 | 畫面與動作 | 她說 | 你的選擇 |", "|---|---|---|---|---|"]
        for sh in b["shots"]:
            mark = " ◀ 主視覺" if sh["n"] == VISUAL_SHOT[outing_id] else ""
            lines.append(
                f"| {sh['n']} | {PURPOSE_ZH.get(sh['purpose'], sh['purpose'])}{mark} | {sh['scene']} | {sh['line']} | {sh['choice']} |"
            )
        lines += ["", f"主視覺的畫面（◀ 的鏡頭）：{outing['moment']}", "", b["cuts"], ""]
    return "\n".join(lines)


def select_html(index, boards):
    """One page, one section per outing: the script above a grid of that outing's candidate pictures. Click to select,
    copy the choice as JSON ({outing: [ids]}); `picks --file` files them. The page records clicks, nothing else."""
    import html as _html

    by_outing = {}
    for rec in index:
        by_outing.setdefault(rec["outing"], []).append(rec)
    nav, sections = [], []
    for outing_id, outing in OUTINGS.items():
        recs = by_outing.get(outing_id, [])
        b = boards[outing_id]
        rows = "".join(
            f"<tr{' class=hi' if sh['n'] == VISUAL_SHOT[outing_id] else ''}><td>{sh['n']}</td><td>{_html.escape(PURPOSE_ZH.get(sh['purpose'], sh['purpose']))}</td>"
            f"<td>{_html.escape(sh['scene'])}</td><td>{_html.escape(sh['line'])}</td></tr>"
            for sh in b["shots"]
        )
        figs = "".join(
            f"<figure data-id='{r['id']}' title='{_html.escape(r['summary_zh'])}'><img loading=lazy src='images/{r['image']}'>"
            f"<figcaption>{_html.escape(r['name_zh'])} {r['age']}｜{r['letter'].upper()}</figcaption></figure>"
            for r in recs
        )
        nav.append(f"<a href='#{outing_id}'>{outing['zh']}</a>")
        sections.append(
            f"<section id='{outing_id}'><h2>{outing['zh']} <small>{len(recs)} 張候選，已選 <b class=c>0</b></small></h2>"
            f"<p class=meta>{_html.escape(b['meta'])}</p><table><tr><th>#</th><th>用途</th><th>畫面與動作</th><th>她說</th></tr>{rows}</table>"
            f"<p class=meta>主視覺（對應黃底鏡頭）：{_html.escape(outing['moment'])}</p><div class=g>{figs}</div></section>"
        )
    return f"""<!doctype html><meta charset="utf-8"><title>各情境劇本與候選圖</title>
<style>body{{font:14px sans-serif;margin:0 16px 80px}}nav{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc;z-index:2}}
nav a{{margin-right:10px}}table{{border-collapse:collapse;margin:6px 0}}td,th{{border:1px solid #ccc;padding:3px 8px;text-align:left}}tr.hi{{background:#fff3cd}}
.meta{{color:#555;margin:4px 0}}.g{{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}}figure{{margin:0;width:170px;border:3px solid transparent;border-radius:6px;cursor:pointer}}
figure.on{{border-color:#d33}}img{{width:100%;border-radius:4px;display:block}}figcaption{{font-size:11px;color:#555}}small{{font-weight:normal;color:#777}}
#bar{{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #ccc;padding:8px 16px}}textarea{{width:60%;height:40px;vertical-align:middle}}</style>
<nav>{"".join(nav)}</nav>{"".join(sections)}
<div id=bar>已選 <b id=n>0</b> 張 <button id=cp>複製選取清單（JSON）</button><button id=cl>清除</button><textarea id=t readonly placeholder="選好後按複製，貼回給 Claude"></textarea></div>
<script>
const S=new Set();try{{JSON.parse(localStorage.getItem("kv-select")||"[]").forEach(i=>S.add(i))}}catch(e){{}}
function upd(){{document.getElementById("n").textContent=S.size;
document.querySelectorAll("section").forEach(sec=>{{sec.querySelector(".c").textContent=sec.querySelectorAll("figure.on").length}});
try{{localStorage.setItem("kv-select",JSON.stringify([...S]))}}catch(e){{}}}}
document.querySelectorAll("figure").forEach(f=>{{if(S.has(f.dataset.id))f.classList.add("on");
f.onclick=()=>{{S.has(f.dataset.id)?S.delete(f.dataset.id):S.add(f.dataset.id);f.classList.toggle("on");upd()}}}});
document.getElementById("cp").onclick=()=>{{const o={{}};document.querySelectorAll("section").forEach(sec=>{{
const ids=[...sec.querySelectorAll("figure.on")].map(f=>f.dataset.id);if(ids.length)o[sec.id]=ids}});
const t=document.getElementById("t");t.value=JSON.stringify(o);t.select();try{{navigator.clipboard.writeText(t.value)}}catch(e){{document.execCommand("copy")}}}};
document.getElementById("cl").onclick=()=>{{S.clear();document.querySelectorAll("figure.on").forEach(f=>f.classList.remove("on"));upd()}};
upd();
</script>
"""


def write_scripts():
    """SCRIPTS.md and select.html from STORYBOARDS.md and the imported pictures (index.json)."""
    boards = parse_storyboards()
    index_path = os.path.join(ROOT, "index.json")
    index = cs._read_json(index_path)["images"] if os.path.isfile(index_path) else []
    counts = {}
    for rec in index:
        counts[rec["outing"]] = counts.get(rec["outing"], 0) + 1
    with open(SCRIPTS_MD, "w", encoding="utf-8", newline="") as f:
        f.write(scripts_markdown(boards, counts) + "\n")
    with open(SELECT_HTML, "w", encoding="utf-8", newline="") as f:
        f.write(select_html(index, boards))
    return {"outings": len(boards), "pictures": len(index), "scripts": SCRIPTS_MD, "select": SELECT_HTML}


# ---------------------------------------------------------------- picks (the owner's choice, filed)


def picks(ids, *, images_dir=IMAGES_DIR, picked_dir=PICKED_DIR):
    """Copies the pictures the owner chose to picked/ and appends them to picked/PICKS.json. Unknown ids or pictures
    that are not filed yet are an error, so a typo does not silently drop a choice."""
    if isinstance(ids, dict):  # what select.html copies: {outing: [ids]}
        ids = [i for group in ids.values() for i in group]
    known = {e["id"]: e for e in build_all()}
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


# ---------------------------------------------------------------- Qwen-Image 2.1 on the local card (internal test only)

QWEN_DIR = os.path.join(ROOT, "qwen_internal")  # outside images/ and picked/: never imported, never sent to H3
QWEN_NOTE = "Qwen Research License (non-commercial): internal comparison only, not for ads or H3"


def qwen_request(bible, entry):
    """Prompt, negative, seed and sampler for one picture, no I/O. The negative comes from the shared assembly (age-safety
    terms included) and cfg is floored at SAFETY_MIN_CFG: Qwen's own cfg 1 would drop the negative prompt."""
    prompt, negative = gc._build_prompt_and_negative(
        entry["text"],
        entry["negative"],
        "safe",
        None,
        bible["style_positive"],
        cs.style_negative(bible),
        cs.IMAGE_MODELS["qwen_image_2_1"]["key"],
        solo=entry["solo"],
        allow_text=False,
    )
    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": entry["seed"],
        "width": bible["canvas"]["width"],
        "height": bible["canvas"]["height"],
        "steps": cs.client.QWEN_IMAGE21_STEPS,
        "cfg": max(cs.client.QWEN_IMAGE21_CFG, cs.client.SAFETY_MIN_CFG),
        "model": cs.IMAGE_MODELS["qwen_image_2_1"]["key"],
    }


def qwen_generate(ids, *, directory=QWEN_DIR, force=False, submit=None, server_up=None, before_each=None):
    """Render the given ids with Qwen-Image 2.1 on the local ComfyUI, one at a time. Skips pictures already there."""
    submit = submit or cs.client.submit_txt2img_generation_qwen_image21
    server_up = server_up or cs.client.is_server_running
    bible = cs.load_bible(BIBLE_ROOT)
    by_id = {e["id"]: e for e in build_all()}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise gc.UsageError(f"沒有這些主視覺：{unknown[:5]}")
    todo = [
        i for i in ids if force or not os.path.isfile(os.path.join(directory, "images", by_id[i]["outing"], f"{i}.png"))
    ]
    if todo and not server_up():
        raise gc.UsageError("ComfyUI 沒有在跑。先啟動它（見 CLAUDE.md），再 qwen")
    made = []
    for ident in todo:
        entry = by_id[ident]
        if before_each:
            before_each(ident)  # e.g. gpu_guard: unload Ollama, wait until the card is cool; may raise to stop
        req = qwen_request(bible, entry)
        out_dir = os.path.join(directory, "images", entry["outing"])
        os.makedirs(out_dir, exist_ok=True)
        started = time.time()
        raw = submit(
            prompt=req["prompt"],
            negative_prompt=req["negative_prompt"],
            seed=req["seed"],
            filename_prefix=f"kvq_{ident}",
            model=req["model"],
            width=req["width"],
            height=req["height"],
            steps=req["steps"],
            cfg=req["cfg"],
        )
        dest = os.path.join(out_dir, f"{ident}.png")
        shutil.move(raw, dest)
        cs._write_json(
            os.path.join(out_dir, f"{ident}.json"),
            dict(
                req,
                id=ident,
                image_model="qwen_image_2_1",
                license=cs.client.QWEN_IMAGE21_LICENSE,
                note=QWEN_NOTE,
                generation_seconds=round(time.time() - started, 1),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            ),
        )
        made.append(ident)
        print(f"  saved {dest}", flush=True)
    return {"made": made, "skipped": len(ids) - len(todo)}


# ---------------------------------------------------------------- measuring (CPU, InsightFace)


def measure(describe, *, images_dir=IMAGES_DIR):
    """`describe(path) -> {"age", "embedding", "faces"} or None`. Reports the estimated age per character, the mean
    similarity between one character's pictures in different outings, and the pictures to look at first: no face found,
    a second face, or an estimated age under MIN_ESTIMATED_AGE."""
    ages, embeddings = {}, {}
    no_face, second_face, too_young = [], [], []
    for e in build_all():
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
    e.add_argument("--set", choices=("pilot", "all", "faces"), default="all", dest="which")
    e.add_argument("--batch")
    e.add_argument("--skip-done", action="store_true", help="leave out pictures already under images/")
    i = sub.add_parser("import", help="file the finished jobs under projects/ad_keyvisuals/images/")
    i.add_argument("--force", action="store_true")
    sub.add_parser("measure", help="age / similarity / second faces (needs insightface: ComfyUI's python)")
    q = sub.add_parser("qwen", help="render with Qwen-Image 2.1 on the local card (internal test only, uses the GPU)")
    q.add_argument("--set", choices=("pilot",), default=None, dest="which")
    q.add_argument("--ids", help="comma separated ids instead of a set")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    k = sub.add_parser("picks", help="archive the pictures the owner chose")
    k.add_argument("--file", required=True, help="a JSON list of ids, or {outing: [ids]} (what the pages copy)")
    sub.add_parser("scripts", help="write SCRIPTS.md and select.html: each outing's script with its candidate pictures")
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
            pool = (
                pilot_ids()
                if args.which == "pilot"
                else [x["id"] for x in build_extra_catalog()]
                if args.which == "faces"
                else [x["id"] for x in build_catalog()]
            )
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
    elif args.cmd == "scripts":
        print(json.dumps(write_scripts(), ensure_ascii=False))
    elif args.cmd == "qwen":
        ids = args.ids.split(",") if args.ids else pilot_ids() if args.which == "pilot" else []
        if not ids:
            raise gc.UsageError("給 --set pilot 或 --ids")
        if args.dry_run:
            print(f"[dry-run] {len(ids)} 張，Qwen-Image 2.1，本機 GPU（未動用）：{', '.join(ids[:4])} ...")
            return
        import gpu_guard

        def cool_first(_ident):
            gpu_guard.before_image()
            gpu_guard.wait_until_cool()

        print(json.dumps(qwen_generate(ids, force=args.force, before_each=cool_first), ensure_ascii=False))
    elif args.cmd == "picks":
        print(json.dumps(picks(cs._read_json(args.file)), ensure_ascii=False))
    else:
        print(json.dumps(measure(_insightface_describe()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
