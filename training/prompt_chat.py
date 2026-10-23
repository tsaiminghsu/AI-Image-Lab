"""A chat with the local Qwen (qwen3.5:9b through Ollama, as h3_idea and prompt_rewrite use it) about a picture or a
video idea, inside the GUI, so the wording can be settled by talking before it is typed into the generator.

The assistant talks Chinese and, when the idea is clear, puts the result in one fenced block: ```prompt for the
picture prompt, ```motion for what moves in the video. The GUI's buttons copy the last such block into the generator's
boxes; the usual pipeline (translation or Qwen rewrite, the safety negatives, content_gate) still runs after that, so
nothing said here switches anything off. The model is the uncensored one: the user's message is gated before it is sent
and the reply after, and a blocked reply never reaches the chat. Each call unloads the model again (keep_alive 0) so the
card is free for ComfyUI; a reply costs a model load of about 7-12 s.

No gradio here; the GUI keeps the history.
"""

import re

import requests

import content_gate
import generate_character as gc
import gpu_topology
import h3_idea

MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARS = 1500
BLOCK_KINDS = ("prompt", "motion")
THINKING_MARK = "💭"  # the animated "Qwen is thinking" bubble the GUI shows while waiting; never sent to the model
BLOCKED_MARK = "⛔"  # a chat bubble that starts with this is a notice from the tool, never sent back to the model

SYSTEM = (
    "You are the prompt assistant inside a local tool that makes pictures and short videos of fictional adult "
    "characters. Talk with the user in Traditional Chinese. Help them make an idea precise: ask one or two short "
    "questions when something important is unclear (who, what they do, where, the light, the framing, the mood), "
    "and suggest concrete improvements. Rules: every person is an adult, early twenties or older, never a child or "
    "a teenager and no school uniforms; no nudity or sexual content; no real people. When the idea is clear, give the "
    "final wording in ONE fenced block. For a picture: a block that starts with ```prompt and holds one to three "
    "Chinese sentences (who, what they do, where, light, framing; no English tags, the tool translates it). For a "
    "video: a block that starts with ```motion and holds a short Chinese description of what moves, the person's "
    "gesture and look before and after a spoken line, with the camera holding steady. Keep everything outside the "
    "block under 120 words, and only write a block when the user wants one or the idea is ready."
)


class ChatError(gc.UsageError):
    pass


class ChatBlocked(ChatError):
    """The content gate stopped the user's message or the model's reply; str(exc) says which and names the terms."""


def _text(content):
    """Gradio can hand a message's content as a string or as a list of parts ({"type": "text", "text": ...})."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def clean_history(history):
    """The last MAX_HISTORY_MESSAGES user/assistant messages as plain {role, content} dicts, each cut to
    MAX_MESSAGE_CHARS."""
    out = []
    for m in history or []:
        role = m.get("role") if isinstance(m, dict) else None
        text = _text(m.get("content")).strip() if isinstance(m, dict) else ""
        if role == "assistant" and text.startswith(THINKING_MARK):
            continue
        if role == "assistant" and text.startswith(BLOCKED_MARK):
            # a notice that the previous user message was stopped: neither the notice nor that message goes to the model
            if out and out[-1]["role"] == "user":
                out.pop()
            continue
        if role in ("user", "assistant") and text:
            out.append({"role": role, "content": text[:MAX_MESSAGE_CHARS]})
    return out[-MAX_HISTORY_MESSAGES:]


def extract_block(text, kind="prompt"):
    """The text of the last ```kind block in `text`, or None. Whitespace is collapsed to single spaces."""
    if kind not in BLOCK_KINDS:
        raise ValueError(f"kind must be one of {BLOCK_KINDS}")
    found = re.findall(rf"```{kind}[ \t]*\r?\n(.*?)```", text or "", flags=re.S)
    return " ".join(found[-1].split()) if found else None


def chat(
    history,
    message,
    *,
    model=h3_idea.DEFAULT_MODEL,
    url=None,
    post=requests.post,
    get=requests.get,
    starter=h3_idea.start_own_ollama,
):
    """The assistant's reply to `message` after `history` (a list of {role, content}). Raises ChatError for an empty
    or when Ollama or the model cannot be reached; ChatBlocked (a ChatError) when the content gate stopped the message or the reply."""
    message = (message or "").strip()
    if not message:
        raise ChatError("請先輸入要問的內容")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ChatError(f"一則訊息最多 {MAX_MESSAGE_CHARS} 字，現在是 {len(message)} 字")
    try:
        content_gate.check(message)  # before the model is started: blocked wording is never sent to it
    except gc.UsageError as exc:
        raise ChatBlocked(f"你的訊息沒有送出：{exc}") from exc
    own = None
    try:
        base, models = h3_idea.find_ollama(get, url)
    except h3_idea.IdeaError:
        base, models = None, []
    try:
        if model not in models:
            if starter is None:
                raise ChatError(f"{base or 'Ollama'} 沒有 {model}（現有：{', '.join(models) or '沒有任何模型'}）")
            try:
                own = starter(model)
            except h3_idea.IdeaError as exc:
                raise ChatError(str(exc)) from exc
            base = h3_idea.own_url()
        body = {
            "model": model,
            "stream": False,
            "think": False,
            "keep_alive": gpu_topology.llm_keep_alive(),
            "options": {"temperature": 0.6, "num_ctx": 4096},
            "messages": [
                {"role": "system", "content": SYSTEM},
                *clean_history(history),
                {"role": "user", "content": message},
            ],
        }
        try:
            r = post(f"{base}/api/chat", json=body, timeout=300)
            r.raise_for_status()
            reply = (r.json()["message"]["content"] or "").strip()
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise ChatError(f"Qwen 沒有回應（{exc}）") from exc
    finally:
        if own is not None:
            own.terminate()
    if not reply:
        raise ChatError("Qwen 回了空白，請再問一次")
    try:
        content_gate.check(reply)
    except gc.UsageError as exc:
        terms = "、".join(content_gate.found_terms(reply)[:5])
        raise ChatBlocked(
            f"Qwen 的回覆含有被內容過濾擋下的字（{terms}），這則回覆已丟掉、沒有放進對話；請換個說法再問一次。"
        ) from exc
    return reply
