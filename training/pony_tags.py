"""Pony prompt vocabulary, grouped into the categories a Pony-family checkpoint expects.

Pony Diffusion V6 XL and its merges (client.PONY_CHECKPOINTS) were trained on booru tags, and
that convention has two layers:

  required  score_* quality prefix, source_*, rating_*, subject/count (1girl, solo)
  optional  framing, pose, expression, hair/build, outfit, lighting, scene, style, negatives

Of the required half, only the score prefix existed in code (generate_character.PONY_QUALITY_TAGS)
- source_* and rating_* appeared nowhere at all, not even in booru_lexicon.json. The optional half
existed as text pools written for the random batch generators, with no Chinese labels and no way to
pick from them in the GUI. This module is the single machine-readable source for both halves, and
gui.py turns it into checkboxes.

Not a replacement for prompt_adapter, the other half of the same problem: that one INFERS booru
tags from whatever prose the user typed and appends them automatically, so natural language keeps
working. This one lets the user PICK tags explicitly - which is the only way to reach the ones no
sentence implies (rating_*, source_*) and the only way to see the vocabulary at all. They compose:
compose()'s output is ordinary prompt text, so adapt() runs over it like any other prompt and its
own dedupe keeps a picked tag from being appended twice.

Three rules this module deliberately follows:

1. It never emits a score_* tag. _build_prompt_and_negative() already prepends
   PONY_QUALITY_TAGS for every Pony checkpoint at the one rewrite point CLAUDE.md requires;
   emitting it here too would put "score_9" in the prompt twice. The quality category is
   carried anyway, with selectable=False, so the GUI can SHOW what is already being added.
2. It composes a prompt BODY, not a prompt. What compose() returns is exactly the text a user
   would otherwise have typed into the prompt box - it still goes through gen_custom() and
   _build_prompt_and_negative(), so the identity prefix, style terms and safety negatives are
   applied to it the same way as to anything hand-typed.
3. rating_explicit is documented (RATING_EXPLICIT below) but never offered. SUGGESTIVE_NEGATIVE
   blocks explicit content outright, so a selectable rating_explicit would fight the negative
   prompt on every generation - a broken image and a false promise, not a feature.

Pools that already exist in generate_character are referenced, never copied: this module only
adds the Chinese label for each entry. If a pool gains or loses an entry without its label
being updated, _pool() raises at import naming the tag, the same way _load_template() fails on
a re-exported workflow rather than letting the GUI quietly show one option fewer.
"""

import re

import comfyui_client as client
import generate_character as gc

TIERS = ("safe", "suggestive")
ROLES = ("required", "optional")
SLUG_RE = re.compile(r"^[a-z0-9_]+$")
REQUIRED_FIELDS = ("slug", "name_zh", "role", "pony_only", "multi", "selectable", "negative",
                   "note_zh", "tags")

# Pony's third rating tier. Kept as a named constant so PROMPT_GUIDE.md can document the full
# convention and the tests can assert it is absent from every category - see rule 3 above.
RATING_EXPLICIT = "rating_explicit"


def _pool(labels, *pools, tier="safe"):
    """Tag entries for English phrases that already live in generate_character, paired with the
    Chinese labels curated here. Raises if the two have drifted apart in either direction."""
    tags = []
    for pool in pools:
        for tag in pool:
            if tag not in tags:
                tags.append(tag)
    missing = [t for t in tags if t not in labels]
    extra = [t for t in labels if t not in tags]
    if missing or extra:
        raise ValueError(
            "pony_tags label map is out of sync with generate_character: "
            f"no label for {missing}, label for non-existent {extra}"
        )
    return [{"tag": t, "name_zh": labels[t], "tier": tier} for t in tags]


def _own(pairs, tier="safe"):
    """Tag entries for vocabulary this module owns (no generate_character pool to mirror)."""
    return [{"tag": t, "name_zh": zh, "tier": tier} for t, zh in pairs]


FRAMING_ZH = {
    "front view portrait": "正面半身",
    "3/4 view portrait": "四分之三側面",
    "side profile portrait": "正側臉",
    "looking over shoulder": "越肩回眸",
    "slightly from above": "微俯角",
    "slightly from below": "微仰角",
    "close-up headshot": "大頭特寫",
    "medium shot upper body": "上半身中景",
    "overhead view looking down at camera": "正上方俯拍看鏡頭",
    "high angle view from directly above": "正上方高角度",
    "full body shot": "全身",
    "full body shot from a distance": "遠景全身",
    "full length shot standing": "站姿全身",
    "full body shot walking": "行走全身",
    "head-to-toe full body shot": "頭到腳全身",
    "back view, from behind": "背面",
    "rear view, walking away, seen from behind": "背影走遠",
    "3/4 back view, looking back over shoulder from behind": "背面回頭",
    "back of head and shoulders, from behind": "後腦與肩背",
}

POSE_ZH = {
    "standing straight": "站直",
    "sitting on a chair": "坐在椅子上",
    "leaning against a wall": "靠牆",
    "walking": "走路",
    "hands in pockets": "手插口袋",
    "arms crossed": "雙手抱胸",
    "hand touching hair": "手撥頭髮",
    "looking directly at camera": "直視鏡頭",
    "looking away from camera": "視線看向他處",
    "slight smile": "淺笑",
    "sitting on the floor, knees bent": "坐在地上屈膝",
    "lying on back, relaxed": "仰躺放鬆",
    "lying on back, one knee bent, arms above head": "仰躺、單膝彎、手舉過頭",
    "lying on side, relaxed, head resting on hand": "側躺撐頭",
    "lying on stomach, propped up on elbows, reading a book": "趴著撐肘看書",
    "lying on stomach, chin resting on hands, ankles crossed in the air": "趴著托腮、小腿交叉上抬",
    "seated yoga stretch pose": "坐姿瑜伽伸展",
    "standing yoga tree pose": "瑜伽樹式",
    "stretching arms overhead after exercise": "運動後伸展",
    "jogging pose, mid-stride": "慢跑中",
}

OUTFIT_ZH = {
    "white t-shirt and jeans": "白T配牛仔褲",
    "beige knit sweater": "米色針織衫",
    "denim jacket over t-shirt": "牛仔外套配T恤",
    "floral summer dress": "碎花洋裝",
    "beige trench coat": "米色風衣",
    "cream cardigan": "奶油色開襟衫",
    "white blouse": "白襯衫",
    "casual hoodie": "休閒帽T",
    "light sweater and skirt": "薄毛衣配裙",
    "black bomber jacket over t-shirt": "黑色飛行外套配T恤",
    "casual button-up shirt": "休閒襯衫",
    "knit sweater": "針織毛衣",
    "graphic hoodie": "印花帽T",
    "casual blazer over t-shirt": "休閒西裝外套配T恤",
    "cargo pants and hoodie": "工裝褲配帽T",
}

SUGGESTIVE_OUTFIT_ZH = {
    "bikini": "比基尼",
    "one-piece swimsuit": "連身泳衣",
    "sports bra and yoga shorts": "運動內衣配瑜伽短褲",
    "off-shoulder top and shorts": "露肩上衣配短褲",
    "tank top and short shorts": "背心配熱褲",
    "camisole and shorts": "細肩帶配短褲",
    "swim trunks": "泳褲",
    "athletic shorts, shirtless": "運動短褲、赤膊",
    "board shorts": "衝浪褲",
    "tank top and shorts": "背心配短褲",
    "shirtless, joggers": "赤膊配運動長褲",
    "swim shorts": "海灘泳褲",
}

LIGHTING_ZH = {
    "soft natural window light": "柔和窗光",
    "golden hour sunlight": "黃金時刻陽光",
    "studio softbox lighting": "棚拍柔光箱",
    "overcast daylight": "陰天日光",
    "warm indoor lighting": "溫暖室內光",
    "soft rim light": "柔和輪廓光",
}

SCENE_ZH = {
    "plain white studio background": "純白棚景",
    "cozy cafe interior": "溫馨咖啡廳",
    "quiet city street": "安靜街道",
    "park with trees": "有樹的公園",
    "bright bedroom interior": "明亮臥室",
    "outdoor garden": "戶外花園",
    "minimalist indoor setting": "極簡室內",
    "bookstore interior": "書店內",
}

SUGGESTIVE_SCENE_ZH = {
    "beach at sunset": "夕陽海灘",
    "poolside": "泳池畔",
    "beach during the day": "白天海灘",
    "outdoor shower area": "戶外淋浴區",
    "tropical resort": "熱帶度假村",
    "lakeside dock": "湖畔木棧道",
}

# Order matters: this is the order compose() emits in, and it follows Pony's own convention -
# the score prefix (added upstream) is followed by source, rating, then the subject, then
# everything describing the picture. Putting rating this high, rather than next to the other
# content categories, is what lands it right behind the auto-prepended prefix.
CATEGORIES = [
    {
        "slug": "quality",
        "name_zh": "品質分數（已自動加上，不用勾）",
        "role": "required",
        "pony_only": True,
        "multi": True,
        "selectable": False,
        "negative": False,
        "note_zh": (
            "Pony 系最重要的必備前綴。選了 Pony 系 checkpoint 時，"
            f"「{gc.PONY_QUALITY_TAGS}」會自動加在 prompt 最前面，"
            f"負面側也會自動加「{gc.PONY_QUALITY_NEGATIVE_TAGS}」——"
            "這裡只是顯示出來讓你知道它已經在了，不用自己打，打了反而會重複。"
        ),
        "tags": _own([
            ("score_9", "最高品質"),
            ("score_8_up", "8 分以上"),
            ("score_7_up", "7 分以上"),
        ]),
    },
    {
        "slug": "source",
        "name_zh": "來源風格 source_*（寫實 merge 通常不用勾）",
        "role": "required",
        "pony_only": True,
        "multi": False,
        "selectable": True,
        "negative": False,
        "note_zh": (
            "Pony V6 官方的四個來源標籤，用來切換畫風大方向。"
            "cyberrealistic_pony / pony_realism 這類寫實 merge 通常不要加——"
            "它們已經被調成照片風，再加 source_anime / source_cartoon 會把畫面拉回插畫。"
            "原版 pony 想要動漫或卡通風時才選一個。"
        ),
        "tags": _own([
            ("source_pony", "Pony 原生風"),
            ("source_anime", "日系動畫風"),
            ("source_cartoon", "西方卡通風"),
            ("source_furry", "獸人風"),
        ]),
    },
    {
        "slug": "rating",
        "name_zh": "內容分級 rating_*",
        "role": "required",
        "pony_only": True,
        "multi": False,
        "selectable": True,
        "negative": False,
        "note_zh": (
            "Pony 的分級標籤，跟上方「內容分級」選項配合使用："
            "safe 只有 rating_safe，suggestive 才會多出 rating_questionable（泳裝／曖昧構圖上限）。"
            f"第三級 {RATING_EXPLICIT} 本專案不提供：露骨內容本來就被負面詞擋著，"
            "勾了只會跟負面詞打架、生出壞圖。"
        ),
        "tags": _own([("rating_safe", "全年齡")]) + _own([
            ("rating_questionable", "微擦邊（泳裝／曖昧構圖）"),
        ], tier="suggestive"),
    },
    {
        "slug": "subject",
        "name_zh": "主體與人數",
        "role": "required",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": (
            "booru 的 1girl / 1boy 是「畫面裡幾個人、什麼性別」，solo 則是「只有一個人」。"
            "注意 1girl 本身的年齡偏誤：booru 資料裡它常對應到偏年輕的畫風，"
            "所以寫實成人角色請一起勾 mature female / mature male 和 adult。"
            f"負面詞那邊的年齡保護（{gc.AGE_SAFETY_NEGATIVE}）永遠都在，這裡是把正面側也講清楚。"
        ),
        "tags": _own([
            ("1girl", "一位女性"),
            ("1boy", "一位男性"),
            ("solo", "畫面只有一人"),
            ("mature female", "成熟女性"),
            ("mature male", "成熟男性"),
            ("adult", "成人"),
        ]),
    },
    {
        "slug": "framing",
        "name_zh": "鏡頭與構圖",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": (
            "全身類的取景需要直式畫布才放得下（上方「畫布比例」選直式），"
            "正方形沒有足夠垂直空間，會被裁成上半身。"
        ),
        "tags": _pool(FRAMING_ZH, gc.CLOSE_ANGLES, gc.FULL_BODY_ANGLES, gc.BACK_VIEW_ANGLES),
    },
    {
        "slug": "pose",
        "name_zh": "姿勢",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": (
            "這些是用文字描述姿勢。躺姿、跪姿、蹲姿這類光靠文字壓不住的，"
            "請改用上方「骨架姿勢控制（ControlNet）」選內建骨架，那才是真的鎖姿勢。"
        ),
        "tags": _pool(POSE_ZH, gc.POSES),
    },
    {
        "slug": "expression",
        "name_zh": "表情與視線",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": "looking at viewer 是 booru 的標準寫法，比 looking at camera 更容易被 Pony 認得。",
        "tags": _own([
            ("smiling", "微笑"),
            ("gentle smile", "溫柔笑容"),
            ("laughing", "大笑"),
            ("serious expression", "嚴肅"),
            ("calm expression", "平靜"),
            ("pensive expression", "若有所思"),
            ("looking at viewer", "看向鏡頭"),
            ("eyes closed", "閉眼"),
        ]),
    },
    {
        "slug": "body",
        "name_zh": "髮型與體態",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": (
            "選了角色（Anchor／FaceID）時這些會跟角色本身的外觀描述疊加，"
            "衝突的話以角色設定為準——想換髮型時建議只勾髮型、不要整組勾。"
        ),
        "tags": _own([
            ("long hair", "長髮"),
            ("short hair", "短髮"),
            ("wavy hair", "波浪捲"),
            ("straight hair", "直髮"),
            ("ponytail", "馬尾"),
            ("black hair", "黑髮"),
            ("brown hair", "棕髮"),
            ("east asian", "東亞臉孔"),
            ("slim build", "纖細身型"),
            ("athletic build", "運動型身型"),
            ("freckles", "雀斑"),
        ]),
    },
    {
        "slug": "outfit",
        "name_zh": "服裝",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": "泳裝／運動服類只在「內容分級」選 suggestive 時才出現。",
        "tags": (_pool(OUTFIT_ZH, gc.OUTFITS, gc.MALE_OUTFITS)
                 + _pool(SUGGESTIVE_OUTFIT_ZH, gc.SUGGESTIVE_OUTFITS, gc.MALE_SUGGESTIVE_OUTFITS,
                         tier="suggestive")),
    },
    {
        "slug": "lighting",
        "name_zh": "光線",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": "一張圖挑一個就好，兩種光線互相打架時通常兩個都不像。",
        "tags": _pool(LIGHTING_ZH, gc.LIGHTINGS),
    },
    {
        "slug": "scene",
        "name_zh": "場景",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": (
            "想用縮圖挑場景（含策展好的光線搭配）的話，用「🎯 圖片選擇生圖」分頁；"
            "這裡是同一批場景的純文字版。海灘／泳池類只在 suggestive 分級出現。"
        ),
        "tags": (_pool(SCENE_ZH, gc.BACKGROUNDS)
                 + _pool(SUGGESTIVE_SCENE_ZH, gc.SUGGESTIVE_BACKGROUNDS, tier="suggestive")),
    },
    {
        "slug": "style",
        "name_zh": "攝影風格（額外的）",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": False,
        "note_zh": (
            f"寫實基底詞（{gc.REALISTIC_STYLE}）"
            "已經由上方「風格正/負面詞」欄位自動附加在每張圖上，這裡列的是額外的，不重複。"
        ),
        "tags": _own([
            ("photorealistic", "照片寫實"),
            ("raw photo", "未修圖原始照片感"),
            ("cinematic photo", "電影感"),
            ("analog film", "底片感"),
            ("shot on 85mm lens", "85mm 人像鏡"),
            ("shallow depth of field", "淺景深"),
            ("bokeh background", "背景散景"),
            ("high detail", "高細節"),
        ]),
    },
    {
        "slug": "extra_negative",
        "name_zh": "額外負面詞",
        "role": "optional",
        "pony_only": False,
        "multi": True,
        "selectable": True,
        "negative": True,
        "note_zh": (
            "這一類會加到「額外負面詞」欄位，不是 prompt。"
            "年齡保護、露骨內容封鎖、以及 lowres/blurry/bad anatomy 這些基本負面詞"
            "本來就永遠自動套用，不用也不能在這裡重複。"
        ),
        "tags": _own([
            ("bad hands", "壞掉的手"),
            ("mutated hands", "變形的手"),
            ("extra fingers", "多指"),
            ("fused fingers", "手指黏連"),
            ("missing fingers", "缺指"),
            ("jpeg artifacts", "JPEG 壓縮痕跡"),
            ("oversaturated", "過飽和"),
            ("harsh shadows", "生硬陰影"),
            ("motion blur", "動態模糊"),
        ]),
    },
]


def validate():
    """Self-check on CATEGORIES, run at import. Same reasoning as MINIMUM_AGE's import-time
    check: a typo here reaches the user as an empty checkbox group or a KeyError mid-click,
    which is exactly the kind of quiet failure this repo's checks exist to make loud."""
    seen_slugs = set()
    for i, cat in enumerate(CATEGORIES):
        where = cat.get("slug") or f"entry #{i}"
        missing = [k for k in REQUIRED_FIELDS if k not in cat]
        if missing:
            raise ValueError(f"category {where}: missing field(s) {missing}")
        slug = cat["slug"]
        if not SLUG_RE.match(slug):
            raise ValueError(f"category {where}: slug must match {SLUG_RE.pattern}")
        if slug in seen_slugs:
            raise ValueError(f"category {slug}: duplicate slug")
        seen_slugs.add(slug)
        if cat["role"] not in ROLES:
            raise ValueError(f"category {slug}: role must be one of {ROLES}, got {cat['role']!r}")
        for field in ("name_zh", "note_zh"):
            if not str(cat[field]).strip():
                raise ValueError(f"category {slug}: {field} is empty")
        for field in ("pony_only", "multi", "selectable", "negative"):
            if not isinstance(cat[field], bool):
                raise ValueError(f"category {slug}: {field} must be a bool")
        if not cat["tags"]:
            raise ValueError(f"category {slug}: no tags")
        seen_tags = set()
        for tag in cat["tags"]:
            name = tag.get("tag")
            if not name or not str(name).strip():
                raise ValueError(f"category {slug}: a tag has no text")
            if not name.isascii():
                raise ValueError(f"category {slug}: tag {name!r} must be ASCII - the tag is what "
                                 "the text encoder sees, the Chinese belongs in name_zh")
            if name in seen_tags:
                raise ValueError(f"category {slug}: duplicate tag {name!r}")
            seen_tags.add(name)
            if name == RATING_EXPLICIT:
                raise ValueError(f"category {slug}: {RATING_EXPLICIT} is never offered as a "
                                 "choice - it fights SUGGESTIVE_NEGATIVE on every generation")
            if name.startswith("score_") and slug != "quality":
                raise ValueError(f"category {slug}: {name!r} - score tags are auto-prepended by "
                                 "_build_prompt_and_negative, emitting them here duplicates them")
            if not str(tag.get("name_zh", "")).strip():
                raise ValueError(f"category {slug}: tag {name!r} has no name_zh")
            if tag.get("tier") not in TIERS:
                raise ValueError(f"category {slug}: tag {name!r} tier must be one of {TIERS}")


validate()


def categories(tier="safe", checkpoint=None):
    """The categories selectable at this content tier on this checkpoint, in display order.

    checkpoint=None means the pipeline default (gc.DEFAULT_CUSTOM_CHECKPOINT), which is what
    gen_custom() itself falls back to - the preview must not disagree with what gets sent.

    Non-Pony checkpoints lose the pony_only categories entirely rather than being handed tags
    they were never trained on: same principle as plan_picker(), every family gets asked in the
    dialect it understands. The descriptive categories are plain English and stay for everyone.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r} - choices: {TIERS}")
    key = checkpoint or gc.DEFAULT_CUSTOM_CHECKPOINT
    is_pony = key in client.PONY_CHECKPOINTS
    out = []
    for cat in CATEGORIES:
        if cat["pony_only"] and not is_pony:
            continue
        tags = [t for t in cat["tags"] if t["tier"] == "safe" or tier == "suggestive"]
        if tags:
            out.append({**cat, "tags": tags})
    return out


def choices(cat):
    """(label, value) pairs for a gr.CheckboxGroup - the label shows both languages so the tag
    that actually reaches the model is never hidden from the person picking it."""
    return [(f"{t['name_zh']}（{t['tag']}）", t["tag"]) for t in cat["tags"]]


def compose(selected, tier="safe", checkpoint=None):
    """Turn {category slug: [picked tags]} into (prompt_body, extra_negative).

    Both strings are ordinary prompt text: they go on to gen_custom() and from there to
    _build_prompt_and_negative(), which adds the score prefix, the identity prefix, the style
    terms and the safety negatives. Nothing here bypasses that.

    Anything the tier or the checkpoint no longer allows is dropped silently rather than
    raising - the tier radio can be turned back down to `safe` after the boxes were ticked,
    and an error at that moment would be about the UI's history, not the user's intent.
    """
    positive, negative = [], []
    for cat in categories(tier, checkpoint):
        if not cat["selectable"]:
            continue
        picked = selected.get(cat["slug"]) or []
        if isinstance(picked, str):
            picked = [picked]
        # Iterate the curated order, not the click order, so the output reads the same way
        # whichever order the boxes happened to be ticked in.
        kept = [t["tag"] for t in cat["tags"] if t["tag"] in picked]
        if not cat["multi"]:
            kept = kept[:1]
        bucket = negative if cat["negative"] else positive
        for tag in kept:
            if tag not in bucket:
                bucket.append(tag)
    return ", ".join(positive), ", ".join(negative)
