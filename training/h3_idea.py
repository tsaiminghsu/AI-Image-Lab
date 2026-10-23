"""Turn a Chinese idea ("她笑著指向冒煙的鍋子") plus her spoken line into the English H3 shot description, with the local
Ollama model qwen3.5:9b. The model only writes the two action sentences around the line; the line itself is put into
`<d>[Chinese] ...</d>` by this code, so the app's wording is never rewritten. The result still goes through
h3_colab.build_prompt / screen_prompt in h3_local.generate, so a blocked term is refused there as for any other prompt.

Needs nothing but Ollama's HTTP API (no ComfyUI node, nothing installed). The call uses keep_alive 0, so the model is
unloaded as soon as it answers and the GPU is free again for H3 (the two do not fit together: ~6.7 GB + ~15.7 GB on a
16 GB card).

    ComfyUI\\.venv\\Scripts\\python.exe training\\h3_idea.py --idea "她笑著指向冒煙的鍋子" --line "快！關小火！"
"""

import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import time

import requests

import content_gate
import generate_character as gc
import gpu_topology

# the Qwen3.5 9B that is pulled on this machine (HauhauCS GGUF, Q4_K_M = 6.1 GB, ~65 tok/s measured 2026-10-17)
DEFAULT_MODEL = "hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
URL_CANDIDATES = (
    "http://127.0.0.1:21434",
    "http://127.0.0.1:11434",
)  # this machine's fixed port, then Ollama's default
MAX_LINE_CHARS = 20  # H3 says about 14 Chinese characters in 5 s; longer is cut off (STORYBOARDS.md)
MAX_PART_CHARS = 420
WHO = {"woman": "woman", "man": "man", "couple": "woman"}  # "couple": a woman (S1) and a man (S2) on screen
VOICE = "in a clear, warm voice with a natural Taiwanese Mandarin accent"
CJK = re.compile(r"[㐀-鿿＀-￯　-〿]")

SYSTEM = (
    "You write shot descriptions for an image-to-video model. The first frame is already fixed: you only say what "
    'moves. Answer with one JSON object {"before": ..., "after": ...} and nothing else. "before": one or two '
    "English sentences about what the person visibly does at the start, up to the moment before she or he speaks "
    '(gesture, look, small movement). "after": one or two English sentences about what happens right after the '
    'line (expression, small movement), ending with "The camera holds a steady shot." Rules: refer to the person '
    'as "The woman in the first frame" or "The man in the first frame" in "before", never invent clothes, a '
    "place or objects that the idea does not mention, no dialogue and no quotation marks, no Chinese characters, no "
    "sound or music, nobody else appears, keep it calm and realistic, at most 45 words each."
)
SYSTEM_MULTI = SYSTEM.replace(
    "nobody else appears",
    "other people from the idea may appear but everyone is an adult and only the person we follow speaks",
)
SYSTEM_COUPLE = (
    SYSTEM.replace(
        'refer to the person as "The woman in the first frame" or "The man in the first frame" in "before"',
        'refer to them as "The woman and the man in the first frame" in "before"',
    )
    .replace("what the person visibly does", "what the two people visibly do")
    .replace("before she or he speaks", "before the woman speaks")
    .replace("nobody else appears", "exactly these two adults appear and nobody else")
)


class IdeaError(gc.UsageError):
    pass


def pinned_wanted(get=requests.get):
    """True when Qwen should run on its own card: a second card is planned for it and it is either already serving or has
    room to start (gpu_topology.effective_split). Then the Ollama server of that card is used, not a shared one that may
    have put the model on the picture card."""
    return gpu_topology.effective_split(get)


def pinned_url():
    return f"http://127.0.0.1:{gpu_topology.PINNED_PORT}"


def own_url():
    """Where start_own_ollama's server answers: the Qwen card's pinned server, or the shared one on OWN_PORT."""
    return pinned_url() if pinned_wanted() else f"http://127.0.0.1:{OWN_PORT}"


def find_ollama(get=requests.get, url=None):
    """The first Ollama URL that answers (OLLAMA_URL wins when set). With a second card for Qwen only its pinned server
    counts: a shared server could load the model onto the picture card."""
    if url:
        urls = [url]
    elif os.environ.get("OLLAMA_URL"):
        urls = [os.environ["OLLAMA_URL"]]
    elif pinned_wanted(get):
        urls = [pinned_url()]
    else:
        urls = list(URL_CANDIDATES)
    for u in urls:
        try:
            r = get(f"{u}/api/tags", timeout=4)
            if r.status_code == 200:
                return u, [m.get("name", "") for m in r.json().get("models", [])]
        except (requests.RequestException, ValueError):
            continue
    raise IdeaError(
        "連不上 Ollama。先啟動它（見 projects/ai_short_drama/OPEN_WEBUI.md 的「一鍵啟動」，或 ollama serve）"
    )


OWN_PORT = 21434  # the fixed port the concept flow uses (Windows reserves ranges that swallow the default 11434)
MODEL_FOLDERS = (os.environ.get("OLLAMA_MODELS"), r"D:\Ollama\Models", r"D:\AI\Models\Ollama")


def model_folder(model, folders=MODEL_FOLDERS):
    """The first models folder that really holds `model` (OLLAMA_MODELS can point at an empty one: seen on this
    machine, see projects/ai_short_drama/OPEN_WEBUI.md)."""
    name, _, tag = model.partition(":")
    parts = ("hf.co", *name.split("/")[1:]) if name.startswith("hf.co/") else ("registry.ollama.ai", "library", name)
    for folder in folders:
        if folder and os.path.isfile(os.path.join(folder, "manifests", *parts, tag or "latest")):
            return folder
    return None


class _Persistent:
    """The pinned server outlives a single call (Qwen stays loaded on its own card between chat turns), so the callers'
    terminate() does nothing; it is stopped when the GUI exits (stop_pinned)."""

    def __init__(self, proc):
        self.proc = proc

    def terminate(self):
        pass

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()


_PINNED = {"handle": None}


def stop_pinned():
    handle = _PINNED["handle"]
    if handle is not None:
        handle.stop()
        _PINNED["handle"] = None


atexit.register(stop_pinned)


def start_own_ollama(model, *, popen=subprocess.Popen, get=requests.get, folders=MODEL_FOLDERS, wait=45):
    """Start a private `ollama serve` with the folder that holds the model. It loads no model by itself. With a second
    card for Qwen it is pinned to that card on its own port and kept running (see _Persistent); otherwise it is on OWN_PORT
    and the caller stops it. Raises IdeaError when the model is in no known folder or the server does not come up."""
    pinned = pinned_wanted(get)
    if pinned:
        handle = _PINNED["handle"]
        if handle is not None and handle.proc.poll() is None and pinned_server_up_now(get):
            return handle
    folder = model_folder(model, folders)
    exe = shutil.which("ollama")
    if not folder or not exe:
        raise IdeaError(
            f"找不到 {model}（檢查過的資料夾：{', '.join(f for f in folders if f)}）或 ollama 指令。先 ollama pull {model}"
        )
    port = gpu_topology.PINNED_PORT if pinned else OWN_PORT
    env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=folder, OLLAMA_CONTEXT_LENGTH="4096")
    if pinned:
        env.update(gpu_topology.llm_env())  # only the Qwen card is visible to this server
    proc = popen([exe, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            if get(f"http://127.0.0.1:{port}/api/tags", timeout=3).status_code == 200:
                if pinned:
                    _PINNED["handle"] = _Persistent(proc)
                    return _PINNED["handle"]
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.terminate()
    raise IdeaError(f"自己啟動的 ollama serve 在 {wait} 秒內沒有回應（埠 {port}）")


def pinned_server_up_now(get=requests.get):
    return gpu_topology.pinned_server_up(get)


def check_line(line):
    line = line.strip()
    if not line:
        raise IdeaError("台詞不能是空的")
    if len(line) > MAX_LINE_CHARS:
        raise IdeaError(f"台詞 {len(line)} 個字，超過 {MAX_LINE_CHARS}（5 秒說不完，會被切掉；先縮短）")
    if re.search(r"[<>\n]", line):
        raise IdeaError("台詞不能含 < > 或換行")
    return line


def parse_reply(text):
    """The two sentences, validated: English only, no tags or markers, no quotes, a sensible length."""
    try:
        data = json.loads(text)
        before, after = str(data["before"]).strip(), str(data["after"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise IdeaError(f"模型沒有回有效的 JSON（before / after）：{text[:200]!r}") from exc
    for name, part in (("before", before), ("after", after)):
        if not part or len(part) > MAX_PART_CHARS:
            raise IdeaError(f"{name} 是空的或太長（{len(part)} 字元）")
        if CJK.search(part):
            raise IdeaError(f"{name} 含中文字，只能是英文（台詞由程式放進去）：{part[:80]!r}")
        if re.search(r"[<>\"“”]|<Picture|\[Shot|\[Chinese\]", part):
            raise IdeaError(f"{name} 含標記或引號：{part[:80]!r}")
    return before, after


def compose(before, after, line, gender="woman", line2=None):
    """The description H3 gets: the model's two parts around the app's own line(s). "couple" puts a woman (S1) and a
    man (S2) on screen; the second line (his answer) is optional."""
    if gender == "couple":
        text = f"{before} The woman (S1), {VOICE}, says: <d>[Chinese] {line}</d>"
        if line2:
            text += f" The man (S2), {VOICE}, answers: <d>[Chinese] {line2}</d>"
        return f"{text} {after}"
    return f"{before} The {WHO[gender]} (S1), {VOICE}, says: <d>[Chinese] {line}</d> {after}"


def write_description(
    idea,
    line,
    gender="woman",
    *,
    line2=None,
    people="single",
    model=DEFAULT_MODEL,
    url=None,
    post=requests.post,
    get=requests.get,
    starter=start_own_ollama,
):
    if gender not in WHO:
        raise IdeaError(f"gender 要是 {', '.join(WHO)}")
    if people not in PEOPLE:
        raise IdeaError(f"people 要是 {', '.join(PEOPLE)}")
    line = check_line(line)
    line2 = check_line(line2) if (line2 or "").strip() else None
    if line2 and gender != "couple":
        raise IdeaError("第二個人的台詞只有「一男一女」才用得到")
    idea = idea.strip()
    if not idea:
        raise IdeaError("情境不能是空的")
    content_gate.check(idea, line, line2)  # before the model is started: explicit / minor wording is never sent to it
    own = None
    try:
        base, models = find_ollama(get, url)
    except IdeaError:
        base, models = None, []
    try:
        if model not in models:
            if starter is None:
                raise IdeaError(f"{base or 'Ollama'} 沒有 {model}（現有：{', '.join(models) or '沒有任何模型'}）")
            own = starter(model)  # a private ollama serve on the folder that really holds the model
            base = own_url()
        return _ask(idea, line, gender, model, base, post, line2, people)
    finally:
        if own is not None:
            own.terminate()


PEOPLE = ("single", "multi")  # multi: other adults from the idea may be in the shot, only the main person speaks


def _system_for(gender, people):
    return SYSTEM_COUPLE if gender == "couple" else SYSTEM_MULTI if people == "multi" else SYSTEM


def _user_message(idea, line, gender, line2=None):
    if gender == "couple":
        text = f"People: a woman and a man. Idea (Chinese): {idea}\nThe woman says (do not repeat it): {line}"
        if line2:
            text += f"\nThe man answers (do not repeat it): {line2}"
        return text
    return f"Person: {gender}. Idea (Chinese): {idea}\nShe or he will say (do not repeat it): {line}"


def _ask(idea, line, gender, model, base, post, line2=None, people="single"):
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": gpu_topology.llm_keep_alive(),
        "options": {"temperature": 0.4, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": _system_for(gender, people)},
            {
                "role": "user",
                "content": _user_message(idea, line, gender, line2),
            },
        ],
    }
    r = post(f"{base}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    before, after = parse_reply(r.json()["message"]["content"])
    content_gate.check(before, after)  # the model here is the uncensored one: its own sentences are gated too
    return compose(before, after, line, gender, line2)


def main(argv=None):
    p = argparse.ArgumentParser(description="Chinese idea + spoken line -> English H3 description (local qwen3.5)")
    p.add_argument("--idea", required=True)
    p.add_argument("--line", required=True)
    p.add_argument("--line2", help="the man's answer (only with --gender couple)")
    p.add_argument("--gender", choices=tuple(WHO), default="woman")
    p.add_argument("--model", default=DEFAULT_MODEL)
    a = p.parse_args(argv)
    print(write_description(a.idea, a.line, a.gender, line2=a.line2, model=a.model))


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
