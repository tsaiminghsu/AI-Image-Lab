"""Turn a Chinese picture idea into the two English prompts the Pony x Z-Image tab needs, with the local Ollama model
(qwen3.5:9b, the same one h3_idea uses): booru-style tags for the Pony base picture and plain sentences for Z-Image.

The model only writes the scene. Quality tags (score_9...), the age-safety negatives, the character prefix and the style
words are still added by generate_character._build_prompt_and_negative afterwards, so nothing here can switch them off.
The model is the uncensored one: the idea is gated before it is sent and both replies are gated after (content_gate),
and a person must be written as an adult. keep_alive 0 unloads it the moment it answers, so the card is free for ComfyUI.

    ComfyUI\\.venv\\Scripts\\python.exe training\\prompt_rewrite.py --idea "坐在圖書館看書的女生"
"""

import argparse
import json
import re

import requests

import content_gate
import generate_character as gc
import gpu_topology
import h3_idea

MAX_IDEA_CHARS = 300
MAX_PART_CHARS = 600
CJK = h3_idea.CJK

SYSTEM = (
    "You write prompts for text-to-image models from a short Chinese idea. Answer with one JSON object "
    '{"pony": ..., "zimage": ...} and nothing else. "pony": one line of comma-separated English tags for an '
    "SDXL Pony model (subject, clothing, pose, place, light, camera), at most 40 words. Do NOT write score_, source_ or "
    'rating_ tags and no quality words. "zimage": two or three plain English sentences describing the same picture '
    "for a natural-language model: who, what they do, the place, the light, the framing. Rules: any person is an adult "
    'woman or man; unless the idea gives an age, write "in her early twenties" or "in his early twenties" for the main '
    "person; never write a child, a teenager, a school uniform or anything youthful; no nudity or sexual content; do "
    "not invent objects, places or clothes the idea does not mention beyond a plain default; no Chinese characters, no "
    "quotation marks, no text or lettering in the picture, no commentary, and no words about skin, texture, detail, sharpness or photo quality (never write realistic texture, high detail, sharp focus, pores, grain): the later steps add those themselves. Both fields must describe the same scene."
)


class RewriteError(gc.UsageError):
    pass


def shorten(text, limit=MAX_PART_CHARS):
    """`text` cut back to at most `limit` characters at the last sentence end or comma that fits (a hard cut at a space when
    there is none in the second half). A shorter prompt is always safe to send, so a long reply is trimmed, not refused:
    the 2026-10-23 batch failed all four pictures on a 762-character answer."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    best = max(cut.rfind(mark) for mark in (". ", "! ", "? ", "; ", ", ", "。"))
    if best < limit // 2:
        best = cut.rfind(" ")
    return (cut[: best + 1] if best >= limit // 2 else cut).rstrip(" ,;")


def parse_reply(text):
    """The two prompts, validated: English only, no quotes. A reply over MAX_PART_CHARS is cut back (see shorten)."""
    try:
        data = json.loads(text)
        pony, zimage = str(data["pony"]).strip(), str(data["zimage"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise RewriteError(f"模型沒有回有效的 JSON（pony / zimage）：{text[:200]!r}") from exc
    pony, zimage = shorten(pony), shorten(zimage)
    for name, part in (("pony", pony), ("zimage", zimage)):
        if not part:
            raise RewriteError(f"{name} 是空的")
        if CJK.search(part):
            raise RewriteError(f"{name} 含中文字，只能是英文：{part[:80]!r}")
        if re.search(r"[\"“”]|(?:^|[\s,])(?:score_\d|source_|rating_)", part):
            raise RewriteError(f"{name} 含引號或品質／來源／分級標籤（那些由程式加）：{part[:80]!r}")
    return pony, zimage


def _ask(idea, model, base, post):
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": gpu_topology.llm_keep_alive(),
        # temperature 0 + a fixed seed: the same Chinese idea gives the same English, so a seed means the same picture and
        # a strength or denoise can be compared without the rewrite changing under it (at 0.4 two runs of one idea differed)
        "options": {"temperature": 0, "seed": 0, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Idea (Chinese): {idea}"},
        ],
    }
    r = post(f"{base}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    pony, zimage = parse_reply(r.json()["message"]["content"])
    content_gate.check(pony, zimage)  # the uncensored model's own words are gated too
    return pony, zimage


def rewrite(
    idea,
    *,
    model=h3_idea.DEFAULT_MODEL,
    url=None,
    post=requests.post,
    get=requests.get,
    starter=h3_idea.start_own_ollama,
):
    """(pony_prompt, zimage_prompt) for a Chinese idea. Raises RewriteError / gc.UsageError, never returns a prompt
    that failed the content gate."""
    idea = (idea or "").strip()
    if not idea:
        raise RewriteError("情境不能是空的")
    if len(idea) > MAX_IDEA_CHARS:
        raise RewriteError(f"情境 {len(idea)} 個字，超過 {MAX_IDEA_CHARS}")
    content_gate.check(idea)  # before the model is started: blocked wording is never sent to it
    own = None
    try:
        base, models = h3_idea.find_ollama(get, url)
    except h3_idea.IdeaError:
        base, models = None, []
    try:
        if model not in models:
            if starter is None:
                raise RewriteError(f"{base or 'Ollama'} 沒有 {model}（現有：{', '.join(models) or '沒有任何模型'}）")
            try:
                own = starter(model)
            except h3_idea.IdeaError as exc:
                raise RewriteError(str(exc)) from exc
            base = h3_idea.own_url()
        return _ask(idea, model, base, post)
    finally:
        if own is not None:
            own.terminate()


def main(argv=None):
    p = argparse.ArgumentParser(description="Chinese idea -> Pony tags + Z-Image sentences (local qwen3.5)")
    p.add_argument("--idea", required=True)
    p.add_argument("--model", default=h3_idea.DEFAULT_MODEL)
    a = p.parse_args(argv)
    pony, zimage = rewrite(a.idea, model=a.model)
    print("PONY:", pony)
    print("ZIMAGE:", zimage)


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
