"""A library of still images for the short films: people of every age and region in clothes of four eras, and empty
places of the same eras. Rendered on the Colab Z-Image worker (.claude/skills/z-image-colab), 720x1280.

    asset_library.py check                       the distribution table; writes projects/asset_library/CATALOG.md
    asset_library.py export --set pilot|all      write a manifest for the skill's submit.py (spends nothing)
    ... submit.py --manifest <file>   then   worker.py up      (the skill's own steps; `worker.py up` spends CU)
    asset_library.py import                      file the finished PNGs under projects/asset_library/images/
    asset_library.py measure                     InsightFace estimate of age / identity / faces in empty places (CPU)

Three kinds of picture, 212 in all:
    portrait   64 people, waist-up in front of a plain backdrop (a reference of the person and the clothes)
    scene      the same 64 people, first-person medium shot in a place of their era
    place      4 eras x 21 places with nobody in them

The people: 2 genders x 4 adult age bands x 4 eras x 2 = 64, spread round-robin over seven regions. Every person is an
adult (the youngest band is "mid-twenties"; the safety path adds the age-safety negatives and the cfg floor exactly as
for every other generator, and a test forbids minor-sounding words in any prompt). No real person, brand or readable
lettering; clothes are modest; regions are named as countries or cultures.

Z-Image has no reference picture, so a portrait and the scene of the same person are two people of the same type in the
same clothes, not the same face. `measure` reports how close they come.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import concept_stills as cs
import generate_character as gc

REPO_ROOT = cs.REPO_ROOT
LIBRARY_ROOT = os.path.join(REPO_ROOT, "projects", "asset_library")
IMAGES_DIR = os.path.join(LIBRARY_ROOT, "images")
CATALOG_MD = os.path.join(LIBRARY_ROOT, "CATALOG.md")
INDEX_JSON = os.path.join(LIBRARY_ROOT, "index.json")
INDEX_HTML = os.path.join(LIBRARY_ROOT, "index.html")
BIBLE_ROOT = os.path.join(REPO_ROOT, "projects", "activity_promo")  # the photographic style text and size
ZIMAGE_JOBS_DIR = os.path.join(REPO_ROOT, "output", "zimage", "jobs")

SEED_BASE = 3_000_000
WIDTH, HEIGHT = 720, 1280

ERAS = ("modern", "retro", "ancient", "future")
ERA_ZH = {"modern": "現代", "retro": "近代復古", "ancient": "古代", "future": "未來"}
GENDERS = ("woman", "man")
GENDER_ZH = {"woman": "女", "man": "男"}
AGES = ("20s", "30-40s", "50-60s", "70plus")
AGE_ZH = {"20s": "二十多歲", "30-40s": "三四十歲", "50-60s": "五六十歲", "70plus": "七十歲以上"}
REGIONS = ("east_asia", "southeast_asia", "south_asia", "mena", "sub_saharan_africa", "europe", "latin_america")
REGION_ZH = {
    "east_asia": "東亞",
    "southeast_asia": "東南亞",
    "south_asia": "南亞",
    "mena": "中東北非",
    "sub_saharan_africa": "撒哈拉以南非洲",
    "europe": "歐洲",
    "latin_america": "拉丁美洲",
}

# how an adult of each band and gender looks (always an adult; the youngest band is a young adult)
AGE_LOOK = {
    ("woman", "20s"): "a young adult {nat} woman in her mid-twenties with a youthful fresh face and smooth clear skin",
    ("man", "20s"): "a young adult {nat} man in his mid-twenties with a youthful fresh face and smooth clear skin",
    ("woman", "30-40s"): "{a_nat} woman in her mid-thirties with a fresh smooth complexion and glossy hair",
    ("man", "30-40s"): "{a_nat} man in his mid-thirties with a smooth healthy face and thick dark hair",
    (
        "woman",
        "50-60s",
    ): "{a_nat} woman in her early fifties with mostly dark hair, a few grey strands and smooth skin",
    ("man", "50-60s"): "{a_nat} man in his early fifties with short dark hair streaked with grey and healthy skin",
    ("woman", "70plus"): "{a_nat} woman in her seventies with white hair and a deeply lined, kind face",
    ("man", "70plus"): "{a_nat} man in his seventies with white hair and a deeply lined, kind face",
}

# region -> nationalities (for modern / retro / future) ; ancient dress is tied to a culture, so it has its own list
REGION_DATA = {
    "east_asia": {
        "nationalities": ["Taiwanese", "Japanese", "Korean", "Chinese"],
        "modern": (
            ["a cream knit sweater and blue jeans", "a light-blue shirt dress and white sneakers"],
            ["a navy crew-neck sweater and chinos", "a grey hooded sweatshirt and jeans"],
        ),
        "retro": (
            ["a 1930s Shanghai qipao with a high collar", "a 1960s cotton blouse with a pleated skirt"],
            ["a 1930s long changshan gown in grey silk", "a 1960s white shirt and high-waisted trousers"],
        ),
        "ancient": [
            (
                "Chinese",
                "a Tang-dynasty ruqun with wide sleeves and a silk sash",
                "a Tang-dynasty round-collar robe with a belt",
            ),
            ("Japanese", "an Edo-period kimono with an obi", "an Edo-period kimono with a hakama and haori"),
            (
                "Korean",
                "a Joseon-era hanbok with a short jacket and a full skirt",
                "a Joseon-era hanbok with a long coat",
            ),
        ],
    },
    "southeast_asia": {
        "nationalities": ["Thai", "Vietnamese", "Indonesian", "Filipino"],
        "modern": (
            ["a light linen blouse and wide-leg trousers", "a floral cotton dress with a light cardigan"],
            ["a short-sleeved linen shirt and trousers", "a polo shirt and long trousers"],
        ),
        "retro": (
            ["a 1960s Vietnamese ao dai in white silk", "a 1960s batik sarong with a kebaya top"],
            ["a 1960s white short-sleeved shirt and dark trousers", "a 1970s batik shirt and trousers"],
        ),
        "ancient": [
            (
                "Indonesian",
                "a Majapahit-era Javanese kebaya with a batik sarong",
                "a batik sarong, a high-collared jacket and a headcloth",
            ),
            (
                "Vietnamese",
                "an ao tu than four-panel dress with a conical hat",
                "a brown cotton ao ba ba tunic with a headscarf",
            ),
        ],
    },
    "south_asia": {
        "nationalities": ["Indian", "Pakistani", "Bangladeshi", "Sri Lankan"],
        "modern": (
            ["a cotton kurta and trousers with a light scarf", "a long cotton dress with a scarf"],
            ["a cotton kurta and trousers", "a plain shirt and jeans"],
        ),
        "retro": (
            ["a 1970s silk saree with a printed border", "a 1960s cotton salwar kameez"],
            ["a 1970s kurta with a Nehru jacket", "a 1960s shirt, trousers and a woollen vest"],
        ),
        "ancient": [
            (
                "Indian",
                "a Mughal-era silk anarkali gown with an embroidered dupatta",
                "a Mughal-era long jama coat with a turban",
            ),
            (
                "Indian",
                "a Rajput-era ghagra skirt, a full-sleeved choli and a veil",
                "a Rajput-era angarkha coat with a turban",
            ),
        ],
    },
    "mena": {
        "nationalities": ["Egyptian", "Iranian", "Turkish", "Moroccan"],
        "modern": (
            [
                "a long cotton tunic, trousers and a soft headscarf",
                "a tailored blazer over a long dress with a soft scarf",
            ],
            ["a linen shirt and trousers", "a polo shirt and chinos"],
        ),
        "retro": (
            ["a 1960s tailored knee-length dress with a silk headscarf", "a 1970s flared-sleeve blouse and long skirt"],
            [
                "a 1950s two-piece suit with a red fez",
                "a 1970s brown suit with wide lapels over a buttoned shirt and a tie",
            ],
        ),
        "ancient": [
            (
                "Turkish",
                "an Ottoman-era long embroidered caftan with a silk headscarf",
                "an Ottoman-era caftan with a turban",
            ),
            ("Iranian", "a Safavid-era long embroidered robe with a veil", "a Safavid-era long coat with a turban"),
        ],
    },
    "sub_saharan_africa": {
        "nationalities": ["Nigerian", "Ghanaian", "Kenyan", "Ethiopian", "South African"],
        "modern": (
            ["a printed cotton dress with a denim jacket", "an Ankara-print top and trousers"],
            ["a cotton shirt and chinos", "a patterned button-up shirt and jeans"],
        ),
        "retro": (
            ["a 1970s printed wax-cloth dress with a matching head wrap", "a 1960s shift dress in bold prints"],
            ["a 1970s safari-style suit", "a 1960s cotton shirt and flared trousers"],
        ),
        "ancient": [
            (
                "Ghanaian",
                "a Kente-cloth wrapped dress with short sleeves and a high covered neckline, gold bead jewellery",
                "a Kente-cloth robe covering both shoulders over a plain white tunic",
            ),
            (
                "Ethiopian",
                "a white cotton habesha kemis with an embroidered hem",
                "a white cotton shamma wrap with a plain tunic",
            ),
        ],
    },
    "europe": {
        "nationalities": ["French", "Italian", "German", "Swedish", "Spanish"],
        "modern": (
            ["a wool coat over a plain top and jeans", "a striped long-sleeved top and a long skirt"],
            ["a wool jacket over a shirt and jeans", "a knitted sweater and corduroy trousers"],
        ),
        "retro": (
            ["a 1950s swing dress with a cardigan", "a 1970s patterned long dress"],
            ["a 1950s three-piece suit with a fedora", "a 1970s turtleneck and flared trousers"],
        ),
        "ancient": [
            (
                "French",
                "a medieval woollen long gown with a linen veil",
                "a medieval woollen tunic with a leather belt and hose",
            ),
            (
                "Italian",
                "a Renaissance velvet gown with slashed sleeves",
                "a Renaissance velvet doublet with a soft cap",
            ),
        ],
    },
    "latin_america": {
        "nationalities": ["Mexican", "Brazilian", "Peruvian", "Colombian"],
        "modern": (
            ["a bright cotton blouse and jeans", "a denim shirt dress"],
            ["a short-sleeved cotton shirt and jeans", "a plain t-shirt and a light jacket with jeans"],
        ),
        "retro": (
            ["a 1950s polka-dot dress with a belt", "a 1970s printed blouse and wide-leg trousers"],
            ["a 1950s guayabera shirt and a straw hat", "a 1960s short-sleeved shirt and trousers"],
        ),
        "ancient": [
            (
                "Mexican",
                "a 19th-century long cotton dress with a rebozo shawl",
                "a 19th-century linen shirt, waistcoat and wide-brimmed hat",
            ),
            (
                "Peruvian",
                "a colonial-era long dress with a lace mantilla shawl",
                "a colonial-era linen shirt, waistcoat and felt hat",
            ),
        ],
    },
}
FUTURE_WEAR = {
    "woman": [
        "a sleek near-future utility jacket with soft light piping and technical trousers",
        "a long pale near-future coat with a high collar and fitted trousers",
        "a layered near-future jacket in matte fabric with a thin glowing collar line",
        "a close-fitting near-future jumpsuit in grey with a short cape",
    ],
    "man": [
        "a near-future utility jacket with soft light piping and technical trousers",
        "a long dark near-future coat with a high collar and straight trousers",
        "a layered near-future jacket in matte fabric with a thin glowing collar line",
        "a close-fitting near-future jumpsuit in grey with a short cape",
    ],
}
BACKDROP = {
    "modern": "a plain light-grey studio backdrop",
    "retro": "a plain warm-beige studio backdrop",
    "ancient": "a plain soft-grey studio backdrop",
    "future": "a plain pale-blue studio backdrop",
}
ERA_LIGHT = {
    "modern": "soft natural daylight, her or his face clearly lit",
    "retro": "warm period light with a slightly faded colour palette, the face clearly lit",
    "ancient": "soft natural light through the room, the face clearly lit",
    "future": "cool white light with a soft neon accent, the face clearly lit",
}

# 21 places per era. The modern list starts with nine of training/scenes/scenes.json's own safe scenes.
SCENES_FILE_SLUGS = (
    "cozy_cafe",
    "city_street",
    "park_trees",
    "bright_bedroom",
    "outdoor_garden",
    "minimalist_room",
    "bookstore",
    "night_neon_street",
    "rooftop_sunset",
)
MODERN_NEW = [
    "a quiet train station platform in the morning, a train waiting",
    "an empty hotel corridor with a patterned carpet runner and a row of closed doors",
    "a supermarket aisle with fresh vegetables",
    "a university library reading room with long wooden tables",
    "a bright open-plan office with plants",
    "a tidy family kitchen with morning light",
    "a living room with a sofa, a bookshelf and a window",
    "a riverside promenade at dusk with benches",
    "a mountain trail with pine trees and a view",
    "a museum gallery with white walls and plain frames",
    "a laundromat with rows of washing machines",
    "a rainy bus stop at night with a glowing shelter",
]
RETRO_SCENES = [
    "a 1930s Shanghai lane house alley with laundry poles and wooden shutters",
    "a 1920s jazz-age ballroom with a polished floor and a chandelier",
    "a 1940s tailor shop with fabric rolls and a sewing table",
    "a 1950s American diner with a long counter and red stools",
    "a 1950s kitchen with pastel appliances",
    "a 1960s railway compartment with wooden seats and a window",
    "a 1960s town market square with awnings",
    "a 1960s tram stop shelter at dusk with a wooden bench and a plain roof",
    "a 1970s record shop with wooden crates of albums",
    "a 1970s living room with wood panelling and a floral sofa",
    "a 1970s city park with a concrete fountain",
    "a 1980s arcade room with glowing cabinets",
    "a 1980s video rental shop with shelves of cassettes",
    "a 1980s Taipei night street with neon lights and scooters",
    "a 1980s office with typewriters and filing cabinets",
    "a 1990s living room with a CRT television and a sofa, a window with net curtains",
    "a 1930s grand hotel lobby with a marble floor",
    "a 1950s suburban street at dusk with pastel houses, trimmed lawns and parked cars",
    "a 1940s postwar train station hall with a wooden bench",
    "a 1960s seaside promenade with a striped kiosk",
    "a 1970s disco room with a mirror ball",
]
ANCIENT_SCENES = [
    "a Tang-style palace courtyard with red pillars and tiled roofs",
    "an Edo-period wooden townhouse street at dusk",
    "a Joseon-era hanok courtyard with a stone well",
    "a Mughal garden pavilion with a marble fountain",
    "a medieval stone castle hall with a long table and torches",
    "a medieval market square with timber houses and no one in it",
    "a Renaissance study with books, a globe and a candle",
    "a bamboo forest path leading to a small shrine",
    "an Ottoman covered bazaar corridor with hanging lamps and no one in it",
    "a Persian courtyard garden with a tiled fountain",
    "a walled clay-and-thatch compound of a West African kingdom at sunset",
    "a highland stone church exterior in Ethiopia at dawn",
    "a colonial plaza with a fountain and arcades",
    "stone terraces of an Andean mountain village in the mist",
    "an ink-wash style river landscape with a wooden boat",
    "a Japanese tea house with a garden pond",
    "a medieval monastery cloister with arches",
    "a Roman villa garden with columns and a pool",
    "a Greek hillside with white temple ruins",
    "an Egyptian temple hall with tall carved columns",
    "a Viking-age longhouse interior with a central fire",
]
FUTURE_SCENES = [
    "a rainy cyberpunk alley with soft neon reflections and no signs with lettering",
    "an orbital space station corridor with curved white walls",
    "a vertical farm tower with rows of glowing plants",
    "a hyperloop station platform with a sleek capsule",
    "a floating city skyline at dusk with glass towers",
    "a Mars colony dome interior with red hills outside",
    "a robotics laboratory with quiet machines and soft light",
    "a holographic cafe with floating lights and empty tables",
    "a rooftop garden above the clouds at sunrise with planters and a low wall, no furniture",
    "an autonomous taxi lane at night with glowing road lines",
    "an underwater habitat with a large round window to the sea",
    "a glass sky bridge between two towers",
    "a solar farm in the desert at dawn",
    "a lunar base surface with the Earth rising",
    "a high-tech apartment living room with a city view",
    "a data centre corridor with blue light",
    "a cave grove of glowing blue mushrooms around a still pond, wide view",
    "a space elevator station concourse",
    "the inside of a maglev train carriage",
    "an orbital greenhouse with a view of the planet",
    "a terraformed canyon city with green terraces",
]

PERSON_NEGATIVE = (
    "outstretched arm, extended arm, selfie, arm reaching toward the camera, hand reaching toward the camera, "
    "strapless, bare shoulders, shirtless, open shirt, plunging neckline, bare chest, revealing clothes"
)
PLACE_NEGATIVE = (
    "person, people, man, woman, face, portrait, crowd, hand, hands, arm, legs, feet, shadow of a person, "
    "human body part, nude, naked, shirtless, bare skin, text, sign, lettering, letters, words, numbers, writing"
)
# Pictures the first full run got wrong (checked by eye 2026-10-09): people, hands, feet or a half-naked crowd in an
# "empty" place, lettering on a sign or screen, strapless or open clothes. Each is drawn again with a new seed and the
# stronger negative terms above; `export --set redo` lists them.
REDO_REVISIONS = {
    **dict.fromkeys(
        (
            "place-future-06", "place-future-09", "place-future-14", "place-future-17",
            "place-modern-08", "place-modern-10", "place-modern-11", "place-modern-19",
            "place-ancient-07",
            "place-retro-08", "place-retro-12", "place-retro-16", "place-retro-18", "place-retro-21",
            "portrait-ancient-w-20s-1", "scene-ancient-w-20s-1", "portrait-ancient-m-70plus-1",
            "scene-ancient-m-70plus-1", "portrait-retro-m-20s-1", "scene-retro-m-20s-1",
        ),
        1,
    ),
    # still wrong after the first redo (legs, a woman, passengers, a hand, a sign): venue text rewritten, new seed
    **dict.fromkeys(
        (
            "place-future-09", "place-future-17", "place-modern-08", "place-modern-11",
            "place-retro-16", "place-retro-18", "place-retro-08",
        ),
        2,
    ),
    # places swapped for new ones (the old ones kept drawing passengers, bare feet, people at computers, lettering)
    **dict.fromkeys(
        ("place-future-17", "place-modern-11", "place-retro-08", "place-retro-16", "place-retro-18"),
        3,
    ),
}  # fmt: skip
REDO_IDS = tuple(REDO_REVISIONS)
REDO_SEED_OFFSET = 100_000  # a redo's seed is the original plus this times its revision
FORBIDDEN_WORDS = (
    "girl",
    "boy",
    "teen",
    "teenager",
    "child",
    "children",
    "kid",
    "minor",
    "underage",
    "baby",
    "toddler",
    "schoolgirl",
    "schoolboy",
    "student uniform",
)


MODERN_SCENE_SUFFIX = {"night_neon_street": ", the street deserted and empty, nobody walking"}


def modern_places():
    import scene_library

    out = []
    for slug in SCENES_FILE_SLUGS:
        out.append(f"{scene_library.scene_text(slug)}{MODERN_SCENE_SUFFIX.get(slug, '')}")
    return out + MODERN_NEW


def venues(era):
    return {
        "modern": modern_places,
        "retro": lambda: RETRO_SCENES,
        "ancient": lambda: ANCIENT_SCENES,
        "future": lambda: FUTURE_SCENES,
    }[era]()


def outfit_for(era, gender, region, counters):
    """(nationality, outfit) of the next person of this region in this era."""
    data = REGION_DATA[region]
    g = 0 if gender == "woman" else 1
    n = counters.setdefault((era, region), 0)
    counters[(era, region)] = n + 1
    if era == "ancient":
        nat, w, m = data["ancient"][n % len(data["ancient"])]
        return nat, (w if gender == "woman" else m)
    nat = data["nationalities"][n % len(data["nationalities"])]
    if era == "future":
        return nat, FUTURE_WEAR[gender][n % len(FUTURE_WEAR[gender])]
    options = data[era][g]
    return nat, options[n % len(options)]


def with_article(nationality):
    """'a Thai' / 'an Indian': the look texts begin with the article."""
    return ("an " if nationality[0] in "AEIOU" else "a ") + nationality


def build_people():
    """64 people: every (era, gender, age) twice, with the two from different regions, regions round-robin."""
    people, counters, i = [], {}, 0
    for era in ERAS:
        for gender in GENDERS:
            for age in AGES:
                for k in (0, 1):
                    region = REGIONS[i % len(REGIONS)]
                    nat, outfit = outfit_for(era, gender, region, counters)
                    i += 1
                    people.append(
                        {
                            "person": f"{era}-{gender[0]}-{age}-{k + 1}",
                            "era": era,
                            "gender": gender,
                            "age": age,
                            "region": region,
                            "nationality": nat,
                            "outfit": outfit,
                            "look": AGE_LOOK[(gender, age)].format(nat=nat, a_nat=with_article(nat)),
                        }
                    )
    return people


def _zh_summary(e):
    if e["kind"] == "place":
        return f"{ERA_ZH[e['era']]}｜空景｜{e['venue']}"
    kind = "定裝照" if e["kind"] == "portrait" else "場景內"
    return (
        f"{ERA_ZH[e['era']]}｜{kind}｜{GENDER_ZH[e['gender']]}、{AGE_ZH[e['age']]}｜{REGION_ZH[e['region']]}"
        f"（{e['nationality']}）｜{e['outfit']}" + (f"｜{e['venue']}" if e["kind"] == "scene" else "")
    )


def build_catalog():
    """All 212 pictures, with the prompt text, the extra negative terms and a fixed seed."""
    entries = []
    people = build_people()
    for p in people:
        entries.append(
            dict(
                p,
                id=f"portrait-{p['person']}",
                kind="portrait",
                venue=None,
                text=(
                    f"straight-on waist-up portrait, looking at the camera with a relaxed expression. "
                    f"{p['look']}, wearing {p['outfit']}. Setting: {BACKDROP[p['era']]}. "
                    f"Lighting: {ERA_LIGHT[p['era']]}"
                ),
                negative=PERSON_NEGATIVE,
                solo=True,
            )
        )
    pool = {era: venues(era) for era in ERAS}
    for i, p in enumerate(people):
        place = pool[p["era"]][(i * 5) % len(pool[p["era"]])]
        entries.append(
            dict(
                p,
                id=f"scene-{p['person']}",
                kind="scene",
                venue=place,
                text=(
                    f"first-person point of view at eye level, medium shot from the waist up. "
                    f"{p['look']}, wearing {p['outfit']}, standing in {place}. "
                    f"Lighting: {ERA_LIGHT[p['era']]}"
                ),
                negative=PERSON_NEGATIVE,
                solo=True,
            )
        )
    for era in ERAS:
        for n, place in enumerate(pool[era], 1):
            entries.append(
                {
                    "id": f"place-{era}-{n:02d}",
                    "kind": "place",
                    "era": era,
                    "venue": place,
                    "person": None,
                    "gender": None,
                    "age": None,
                    "region": None,
                    "nationality": None,
                    "outfit": None,
                    "text": f"{place}, empty scene, no people, wide establishing shot, vertical frame",
                    "negative": PLACE_NEGATIVE,
                    "solo": False,
                }
            )
    for i, e in enumerate(entries):
        e["seed"] = SEED_BASE + i + REDO_SEED_OFFSET * REDO_REVISIONS.get(e["id"], 0)
        e["summary_zh"] = _zh_summary(e)
    return entries


def pilot_ids(entries=None):
    """16 pictures: per era two portraits (a woman and a man of different age bands), one scene and one place,
    so the four age bands, both genders and all three kinds are seen before the full run."""
    entries = entries or build_catalog()
    ids = []
    for e, era in enumerate(ERAS):
        woman = f"portrait-{era}-w-{AGES[e % 4]}-1"
        man = f"portrait-{era}-m-{AGES[(e + 2) % 4]}-2"
        scene = f"scene-{era}-w-{AGES[(e + 1) % 4]}-2"
        place = f"place-{era}-{e * 3 + 1:02d}"
        ids += [woman, man, scene, place]
    known = {x["id"] for x in entries}
    missing = [i for i in ids if i not in known]
    if missing:
        raise gc.UsageError(f"pilot ids not in the catalog: {missing}")
    return ids


def redo_ids(entries=None):
    """The 20 pictures the first full run got wrong (see REDO_IDS)."""
    entries = entries or build_catalog()
    known = {x["id"] for x in entries}
    missing = [i for i in REDO_IDS if i not in known]
    if missing:
        raise gc.UsageError(f"redo ids not in the catalog: {missing}")
    return list(REDO_IDS)


def redo2_ids(entries=None):
    """The 7 pictures that were still wrong after the first redo."""
    entries = entries or build_catalog()
    known = {x["id"] for x in entries}
    ids = [i for i, rev in REDO_REVISIONS.items() if rev == 2]
    missing = [i for i in ids if i not in known]
    if missing:
        raise gc.UsageError(f"redo2 ids not in the catalog: {missing}")
    return ids


def redo3_ids(entries=None):
    """The 5 places that were replaced by new ones after the second redo."""
    entries = entries or build_catalog()
    known = {x["id"] for x in entries}
    ids = [i for i, rev in REDO_REVISIONS.items() if rev == 3]
    missing = [i for i in ids if i not in known]
    if missing:
        raise gc.UsageError(f"redo3 ids not in the catalog: {missing}")
    return ids


def agecheck_ids(entries=None):
    """8 portraits for the two bands the pilot drew too old (30-40s and 50-60s), a woman and a man each, in two eras."""
    entries = entries or build_catalog()
    ids = [
        f"portrait-{era}-{g}-{age}-1"
        for era in ("retro", "ancient")
        for g in ("w", "m")
        for age in ("30-40s", "50-60s")
    ]
    known = {x["id"] for x in entries}
    missing = [i for i in ids if i not in known]
    if missing:
        raise gc.UsageError(f"agecheck ids not in the catalog: {missing}")
    return ids


def job_spec(bible, entry):
    """One manifest entry for the skill. The skill adds the age-safety negatives itself; the style text and the
    realistic-photo negatives come from the shared bible, as in concept_colab.job_spec."""
    extra = f"{cs.style_negative(bible)}, {entry['negative']}"
    return {
        "prompt": f"{entry['text']}, {bible['style_positive']}",
        "negative": extra,
        "width": bible["canvas"]["width"],
        "height": bible["canvas"]["height"],
        "steps": bible["sampler"]["steps"],
        "cfg": max(bible["sampler"]["cfg"], cs.client.SAFETY_MIN_CFG),
        "seed": entry["seed"],
        "solo": entry["solo"],
        "allow_text": False,
        "tag": entry["id"],
    }


def export(which="all", *, ids=None, batch=None, directory=None):
    bible = cs.load_bible(BIBLE_ROOT)
    entries = build_catalog()
    if ids is None:
        ids = (
            {"pilot": pilot_ids, "agecheck": agecheck_ids, "redo": redo_ids, "redo2": redo2_ids, "redo3": redo3_ids}[
                which
            ](entries)
            if which != "all"
            else [e["id"] for e in entries]
        )
    by_id = {e["id"]: e for e in entries}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise gc.UsageError(f"沒有這些素材：{unknown[:5]}")
    directory = directory or os.path.join(LIBRARY_ROOT, "colab")
    batch = batch or f"assets-{which}"
    path = os.path.join(directory, f"{batch}.manifest.json")
    cs._write_json(path, {"batch": batch, "jobs": [job_spec(bible, by_id[i]) for i in ids]})
    return {"batch": batch, "manifest": path, "images": len(ids)}


def distribution(entries):
    """Counts the table in CATALOG.md shows: people per (era, gender, age) and per region, places per era."""
    people = [e for e in entries if e["kind"] == "portrait"]
    cells = {}
    for p in people:
        cells[(p["era"], p["gender"], p["age"])] = cells.get((p["era"], p["gender"], p["age"]), 0) + 1
    regions = {r: sum(1 for p in people if p["region"] == r) for r in REGIONS}
    places = {era: sum(1 for e in entries if e["kind"] == "place" and e["era"] == era) for era in ERAS}
    return {
        "cells": cells,
        "regions": regions,
        "places": places,
        "total": len(entries),
        "kinds": {k: sum(1 for e in entries if e["kind"] == k) for k in ("portrait", "scene", "place")},
    }


def catalog_markdown(entries):
    d = distribution(entries)
    lines = [
        "# 素材庫目錄",
        "",
        "由 `training/asset_library.py check` 產生，不要手改；要改請改程式裡的資料表再重跑。",
        "",
        f"共 **{d['total']} 張**：定裝照 {d['kinds']['portrait']}、場景內 {d['kinds']['scene']}、空景 {d['kinds']['place']}。"
        "全部是成年人，720×1280 直式。",
        "",
        "## 分布",
        "",
        "| 年代 | 性別 | " + " | ".join(AGE_ZH[a] for a in AGES) + " |",
        "|---|---|" + "---|" * len(AGES),
    ]
    for era in ERAS:
        for g in GENDERS:
            lines.append(
                f"| {ERA_ZH[era]} | {GENDER_ZH[g]} | "
                + " | ".join(str(d["cells"].get((era, g, a), 0)) for a in AGES)
                + " |"
            )
    lines += [
        "",
        "每格是人數（每人一張定裝照加一張場景內）。",
        "",
        "| 地區 | " + " | ".join(REGION_ZH[r] for r in REGIONS) + " |",
        "|---|" + "---|" * len(REGIONS),
        "| 人數 | " + " | ".join(str(d["regions"][r]) for r in REGIONS) + " |",
        "",
        "空景：" + "、".join(f"{ERA_ZH[e]} {d['places'][e]}" for e in ERAS),
        "",
    ]
    for kind, title in (("portrait", "定裝照"), ("scene", "場景內"), ("place", "空景")):
        lines += [f"## {title}", ""]
        lines += [f"- `{e['id']}`　{e['summary_zh']}" for e in entries if e["kind"] == kind]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- importing finished jobs


def finished_jobs(jobs_dir=ZIMAGE_JOBS_DIR):
    """tag -> the newest COMPLETED job of the skill's queue that carries it."""
    found = {}
    for path in glob.glob(os.path.join(jobs_dir, "*.json")):
        try:
            job = cs._read_json(path)
        except (OSError, ValueError):
            continue
        tag = job.get("tag")
        if not tag or job.get("status") != "COMPLETED" or not job.get("output") or not os.path.isfile(job["output"]):
            continue
        if tag not in found or (job.get("completed_at") or "") > (found[tag].get("completed_at") or ""):
            found[tag] = job
    return found


def import_jobs(*, jobs_dir=ZIMAGE_JOBS_DIR, images_dir=IMAGES_DIR, force=False):
    entries = build_catalog()
    done = finished_jobs(jobs_dir)
    report = {"imported": [], "skipped": [], "missing": []}
    index = []
    for e in entries:
        job = done.get(e["id"])
        rel = os.path.join(e["kind"], e["era"], f"{e['id']}.png")
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
                "kind",
                "era",
                "gender",
                "age",
                "region",
                "nationality",
                "outfit",
                "venue",
                "seed",
                "summary_zh",
            )
        }
        record.update(
            {
                "image": rel.replace(os.sep, "/"),
                "width": WIDTH,
                "height": HEIGHT,
                "backend": "colab",
                "model": "z_image_turbo (official bf16, Colab)",
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
    data = json.dumps(index, ensure_ascii=False)
    return f"""<!doctype html><meta charset="utf-8"><title>素材庫</title>
<style>body{{font:14px sans-serif;margin:16px}}select{{margin-right:8px}}.g{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}}
figure{{margin:0;width:180px}}img{{width:180px;border-radius:4px}}figcaption{{font-size:11px;color:#555}}</style>
<h2>素材庫 <span id=n></span></h2>
<div id=f></div><div class=g id=g></div>
<script>
const D={data};
const F={{kind:"類別",era:"年代",gender:"性別",age:"年齡層",region:"地區"}};
const f=document.getElementById("f");
for(const k in F){{const s=document.createElement("select");s.id=k;s.innerHTML="<option value=''>"+F[k]+"（全部）</option>"+
[...new Set(D.map(d=>d[k]).filter(Boolean))].map(v=>"<option>"+v+"</option>").join("");s.onchange=r;f.appendChild(s)}}
function r(){{const w={{}};for(const k in F)w[k]=document.getElementById(k).value;
const L=D.filter(d=>Object.keys(w).every(k=>!w[k]||d[k]===w[k]));document.getElementById("n").textContent="（"+L.length+" 張）";
document.getElementById("g").innerHTML=L.map(d=>"<figure><img loading=lazy src='images/"+d.image+"'><figcaption>"+d.summary_zh+"</figcaption></figure>").join("")}}
r();
</script>
"""


# ---------------------------------------------------------------- measuring (CPU, InsightFace)


def measure(describe, *, images_dir=IMAGES_DIR):
    """`describe(path) -> {"age": int, "embedding": [...]} or None` (no face). Returns the report the plan asks for:
    estimated age per age band, similarity of each person's portrait and scene, faces found in empty places."""
    entries = build_catalog()
    by_person = {}
    ages = {}
    faces_in_places = []
    for e in entries:
        path = os.path.join(images_dir, e["kind"], e["era"], f"{e['id']}.png")
        if not os.path.isfile(path):
            continue
        info = describe(path)
        if e["kind"] == "place":
            if info is not None:
                faces_in_places.append(e["id"])
            continue
        if info is None:
            by_person.setdefault(e["person"], {})[e["kind"]] = None
            continue
        by_person.setdefault(e["person"], {})[e["kind"]] = info["embedding"]
        ages.setdefault((e["gender"], e["age"]), []).append(info["age"])
    sims = []
    for person, got in by_person.items():
        a, b = got.get("portrait"), got.get("scene")
        if a is not None and b is not None:
            sims.append(sum(x * y for x, y in zip(a, b)))
    return {
        "age_by_band": {
            f"{g}-{a}": {"n": len(v), "mean": round(sum(v) / len(v), 1), "min": min(v), "max": max(v)}
            for (g, a), v in sorted(ages.items())
        },
        "portrait_vs_scene_similarity": {
            "pairs": len(sims),
            "mean": round(sum(sims) / len(sims), 3) if sims else None,
            "min": round(min(sims), 3) if sims else None,
        },
        "faces_in_empty_places": faces_in_places,
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
        return {"age": int(f.age), "embedding": [float(v) for v in f.normed_embedding]}

    return describe


def main(argv=None):
    p = argparse.ArgumentParser(description="Asset library: people and places for the short films")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="print the distribution and write CATALOG.md")
    e = sub.add_parser("export", help="write the manifest for the z-image-colab skill (spends nothing)")
    e.add_argument("--set", choices=("pilot", "agecheck", "redo", "redo2", "redo3", "all"), default="all", dest="which")
    e.add_argument("--batch")
    e.add_argument("--skip-done", action="store_true", help="leave out pictures already under images/")
    i = sub.add_parser("import", help="file the finished jobs under projects/asset_library/images/")
    i.add_argument("--force", action="store_true")
    sub.add_parser("measure", help="age / identity / faces in empty places (needs insightface: ComfyUI's python)")
    args = p.parse_args(argv)
    if args.cmd == "check":
        entries = build_catalog()
        os.makedirs(LIBRARY_ROOT, exist_ok=True)
        with open(CATALOG_MD, "w", encoding="utf-8", newline="") as f:
            f.write(catalog_markdown(entries) + "\n")
        d = distribution(entries)
        print(
            json.dumps(
                {
                    "total": d["total"],
                    "kinds": d["kinds"],
                    "regions": d["regions"],
                    "places": d["places"],
                    "catalog": CATALOG_MD,
                },
                ensure_ascii=False,
            )
        )
    elif args.cmd == "export":
        ids = None
        if args.skip_done:
            everything = [e["id"] for e in build_catalog()]
            pool = (
                {
                    "pilot": pilot_ids,
                    "agecheck": agecheck_ids,
                    "redo": redo_ids,
                    "redo2": redo2_ids,
                    "redo3": redo3_ids,
                }[args.which]()
                if args.which != "all"
                else everything
            )
            have = {
                os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(IMAGES_DIR, "*", "*", "*.png"))
            }
            ids = [i_ for i_ in pool if i_ not in have]
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
                {k: len(v) for k, v in report.items()} | {"missing_ids": report["missing"][:10]}, ensure_ascii=False
            )
        )
    else:
        print(json.dumps(measure(_insightface_describe()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
