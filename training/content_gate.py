"""A prompt-side content gate for the local picture / H3 tools (training/h3_local.py, training/h3_idea.py).

The project's rule (CLAUDE.md, generate_character) is that explicit sexual content and anything that reads as a minor are
never made. For pictures that rule was carried only by negative prompt terms, and a negative term does not stop a positive
prompt that asks for the act (2026-10-17: a Chinese prompt describing a sex act went through the new picture tab and
produced an image). So these tools also refuse such text BEFORE any translation, model call or GPU work, in Chinese and
in English, and again on the English the translator returned. There is no bypass argument. This is a term list, not a
classifier: it will not catch every phrasing, and the negative terms stay in place underneath it.
"""

import re

import generate_character as gc

ENGLISH = (
    r"vagina|vaginal|vulva|clitoris|penis|cock|dick|genital\w*|nipples?|anus|anal sex|butt ?hole|penetrat\w*|"
    r"intercourse|sex|sexual\w*|porn\w*|orgasm\w*|cum|cumshot|ejaculat\w*|fellatio|blow ?job|hand ?job|masturbat\w*|"
    r"erotic\w*|hentai|nude|naked|topless|bottomless|creampie|gangbang|bdsm|bondage|fetish|"
    r"child|children|kid|kids|minor|underage|teen|teens|teenager|schoolgirl|schoolboy|loli\w*|shota|toddler|baby|"
    r"preteen|pre-teen|jailbait"
)

CHINESE = (
    "陰道|陰莖|陰蒂|陰唇|陰部|私處|下體|生殖器|乳頭|乳暈|插入|抽插|性交|做愛|性愛|口交|肛交|手淫|自慰|高潮|射精|內射|顏射|"
    "色情|情色|淫|裸體|全裸|裸露|露點|無碼|A片|AV女優|強姦|性侵|捆綁|"
    "蘿莉|幼女|女童|男童|小孩|兒童|孩童|未成年|小學生|國小|國中生|高中生|學生妹|少女|正太|幼齒"
)
_EN = re.compile(rf"(?<![A-Za-z])(?:{ENGLISH})(?![A-Za-z])", re.IGNORECASE)
_ZH = re.compile(CHINESE)  # Chinese terms only: no case folding, so no Latin letters can match inside English words

MESSAGE = (
    "這段文字含有露骨性內容或未成年相關的字眼（{found}），這個工具不會送出。"
    "專案規定不生成露骨內容、也不生成看起來像未成年的人物；要改成一般的生活或旅遊情境。"
)


def found_terms(text):
    text = text or ""
    return sorted({m.group(0).lower() for m in _EN.finditer(text)} | {m.group(0) for m in _ZH.finditer(text)})


def check(*texts):
    """Raise gc.UsageError when any text contains a blocked term; otherwise return None."""
    for text in texts:
        hits = found_terms(text)
        if hits:
            raise gc.UsageError(MESSAGE.format(found="、".join(hits[:5])))
