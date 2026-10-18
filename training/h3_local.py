"""MiniMax H3 image-to-video on the local card (RTX 5060 Ti 16 GB): one command, one image -> one MP4 with sound.

    ComfyUI\\.venv\\Scripts\\python.exe training\\h3_local.py first_frame.png --description "..." [--seconds 5]

Same safeguards as the Colab skill: the prompt is assembled and screened by h3_colab.build_prompt (it raises on a
blocked term, there is no bypass), and the first frame must be a still this project made (a fictional adult; the
caller attests that by passing a file, as with the skill's --image). The model is the community GGUF Q4_K plus the
turbo LoRA at 8 steps, measured 2026-10-17 on this card: 5 s at 576x1024 takes 504-597 s, VRAM peaks at ~15.7 GB
(of 16), ComfyUI's Python commits ~43 GB against 32 GB of RAM so the page file is busy and the SSD sits near 100 %.

So: run one clip at a time, nothing else on the GPU (Ollama is unloaded first), the card is allowed to cool first,
and ComfyUI is shut down afterwards if this command started it. Longer than 5 s has not been measured locally and
may run out of VRAM; it is refused unless --allow-long is given.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import requests

import comfyui_client as client
import content_gate
import generate_character as gc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, ".claude", "skills", "minimax-h3-colab", "scripts"))
import h3_colab as h3  # noqa: E402

CJK = re.compile(r"[㐀-鿿]")
LINE_SPAN = re.compile(r"<d>.*?</d>", re.S)  # the spoken line stays Chinese: never translated


def has_cjk(text):
    return bool(CJK.search(text or ""))


_ADULT_WORDS = {"girls": "women", "girl": "woman", "boys": "men", "boy": "man"}
_ADULT_RE = re.compile(r"\b(girls|girl|boys|boy)\b", re.IGNORECASE)


def adultize(text):
    """The translator turns 女生/男生 into "girl"/"boy", and those words pull the picture model towards young faces.
    Everyone in these tools is an adult, so the translated English says woman/man instead."""

    def swap(m):
        word = _ADULT_WORDS[m.group(1).lower()]
        return word.capitalize() if m.group(1)[0].isupper() else word

    return _ADULT_RE.sub(swap, text)


def translate_outside_lines(text, translate):
    """Translate the Chinese parts of a description to English with `translate`, but leave every <d>...</d> (the line the
    person speaks) exactly as it is. Text without Chinese outside the lines is returned unchanged."""

    def part(segment):
        if not has_cjk(segment):
            return segment
        lead = segment[: len(segment) - len(segment.lstrip())]
        trail = segment[len(segment.rstrip()) :]
        return lead + adultize(translate(segment.strip())) + trail

    out, pos = [], 0
    for m in LINE_SPAN.finditer(text):
        out.append(part(text[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(part(text[pos:]))
    return "".join(out)


OUT_DIR = os.path.join(REPO_ROOT, "outputs", "h3_local")
UNET = "minimax_h3_fl2va_pruned-Q4_K.gguf"
LORA = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
TEXT_ENCODER = "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-Q4_K_M.gguf"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
SAFE_SECONDS = 5.0  # the only length measured on this card
DEFAULT_SIZE = (576, 1024)
DEFAULT_MEGAPIXELS = 0.59
MIN_FREE_VRAM_MIB = 14000  # the run peaks at ~15.7 GB of 16.3 GB; anything else holding the card makes it fail
POLL_SECONDS = 3
TIMEOUT_SECONDS = 2400


def required_files():
    """(folder, file) pairs the graph loads, relative to ComfyUI/models."""
    return [
        ("unet", UNET),
        ("loras", LORA),
        ("text_encoders", TEXT_ENCODER),
        ("vae", VIDEO_VAE),
        ("vae", AUDIO_VAE),
    ]


def missing_files(models_dir=None):
    models_dir = models_dir or os.path.join(client.COMFYUI_DIR, "models")
    return [f"{d}/{f}" for d, f in required_files() if not os.path.isfile(os.path.join(models_dir, d, f))]


def build_graph(
    prompt,
    image_name,
    *,
    seed,
    width,
    height,
    frames,
    megapixels,
    tag,
    steps=h3.STEPS,
    sampler=h3.SAMPLER,
    scheduler=h3.SCHEDULER,
):
    """The ComfyUI graph (API format), no I/O: GGUF H3 + turbo LoRA, first-frame image-to-video, audio and video
    decoded and muxed by CreateVideo/SaveVideo. BasicGuider only, so there is no negative prompt and no cfg here -
    the screen on the prompt is the compensating control (see CLAUDE.md)."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "2": {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {"model": ["1", 0], "lora_name": LORA, "strength": 1.0, "low_vram": False},
        },
        "3": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "16": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["6", 0],
                "upscale_method": "nearest-exact",
                "megapixels": megapixels,
                "resolution_steps": 32,
            },
        },
        "7": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "first_frame": ["16", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": frames,
            },
        },
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["7", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["2", 0], "scheduler": scheduler, "steps": steps, "denoise": 1.0},
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["8", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["7", 1],
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "fps": float(h3.FPS), "audio": ["14", 0]}},
        "17": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["15", 0],
                "filename_prefix": f"video/h3local_{tag}",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def screen(text):
    """h3_colab.screen_prompt plus the plural of each blocked English term: the skill's list has "schoolgirl" but a text
    that says "two schoolgirls" passed it (found 2026-10-17 while adding multi-person pictures, where plurals are
    normal). The skill's own list is left alone; this only adds the plurals for the texts typed into this tool.
    A refusal is raised as gc.UsageError so the GUI shows the reason: h3_colab.InputError is not one, and the GUI's
    handler wrapper only turns UsageError into a readable message (2026-10-18: the picture section showed a bare
    "Error" for "全裸" / "nude")."""
    try:
        h3.screen_prompt(text)
    except h3.InputError as exc:
        raise gc.UsageError(SCREEN_MESSAGE.format(found=_terms_in(str(exc)))) from None
    low = (text or "").lower()
    hit = sorted(
        {t for t in h3.BLOCKED_TERMS if t.isascii() and re.search(rf"(?<![a-z]){re.escape(t)}s(?![a-z])", low)}
    )
    if hit:
        raise gc.UsageError(SCREEN_MESSAGE.format(found="、".join(t + "s" for t in hit)))


SCREEN_MESSAGE = (
    "這段文字含有 H3／本專案不送出的字眼（{found}），沒有送出。"
    "這是第二層過濾（h3_colab.screen_prompt：未成年與露骨字眼），跟 content_gate 是分開的。"
)


def _terms_in(message):
    m = re.search(r"\[(.*)\]", message)
    return m.group(1).replace("'", "") if m else message


def assemble_prompt(image, description, soundscape="", static=False):
    """The full H3 prompt through the same assembly and screen as the Colab skill. Raises on a blocked term."""
    spec = h3.JobSpec(
        image=image,
        description=description,
        soundscape=soundscape,
        constraints=("static-subject",) if static else (),
    )
    try:
        return h3.build_prompt(spec)
    except h3.InputError as exc:
        raise gc.UsageError(f"H3 不收這個描述：{exc}") from None


def check_seconds(seconds, allow_long=False):
    if not h3.MIN_DURATION <= seconds <= h3.MAX_DURATION:
        raise gc.UsageError(f"秒數要在 {h3.MIN_DURATION:g}-{h3.MAX_DURATION:g} 之間（H3 的範圍）")
    if seconds < h3.STABLE_MIN_DURATION:
        raise gc.UsageError(f"低於 {h3.STABLE_MIN_DURATION:g} 秒超出 H3 的訓練範圍，本機只做 {SAFE_SECONDS:g} 秒")
    if seconds > SAFE_SECONDS and not allow_long:
        raise gc.UsageError(
            f"本機只量過 {SAFE_SECONDS:g} 秒（5 秒 ≈ 500-600 秒、VRAM 峰值 15.7/16 GB）；更長沒量過，可能顯存不足。"
            "要試就加 --allow-long（建議同時降解析度，例如 --width 512 --height 896 --megapixels 0.46）"
        )
    return h3.frames_for(seconds)


STATUS_EVERY_SECONDS = 10


def machine_status(run=subprocess.run):
    """One line for the GUI's log: GPU temperature and memory, and the system RAM still free."""
    parts = []
    try:
        out = run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=15).stdout  # fmt: skip
        temp, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        parts.append(f"GPU {temp:.0f}°C｜顯存 {used / 1024:.1f}/{total / 1024:.1f} GB")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        parts.append("GPU 讀不到")
    try:
        import psutil

        parts.append(f"RAM 可用 {psutil.virtual_memory().available / 2**30:.1f} GB")
    except ImportError:
        pass
    return "｜".join(parts)


def _guard_call(fn, log):
    """gpu_guard's helpers take log=; a test double may not."""
    try:
        return fn(log=log)
    except TypeError:
        return fn()


class _Heartbeat:
    """Logs machine_status() every STATUS_EVERY_SECONDS while a blocking call (a picture render) runs."""

    def __init__(self, log, label, every=STATUS_EVERY_SECONDS):
        import threading

        self.log, self.label, self.every = log, label, every
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0 = time.time()
        while not self.stop.wait(self.every):
            self.log(f"{self.label} {round(time.time() - t0)} 秒｜{machine_status()}")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()


def free_vram_mib(run=subprocess.run):
    try:
        out = run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        return int(float(out.strip().splitlines()[0]))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def generate(
    image,
    description,
    *,
    soundscape="",
    static=False,
    seconds=SAFE_SECONDS,
    seed=None,
    width=DEFAULT_SIZE[0],
    height=DEFAULT_SIZE[1],
    megapixels=DEFAULT_MEGAPIXELS,
    tag=None,
    out_dir=OUT_DIR,
    allow_long=False,
    keep_comfyui=False,
    guard=None,
    post=requests.post,
    get=requests.get,
    start=None,
    server_up=None,
    translate=None,
    free_vram=None,
    unload_wait=10,
    log=print,
):
    """Make one clip. `translate` (Chinese -> English) is applied to the description and soundscape when they contain
    Chinese outside the <d>...</d> line; without it, Chinese there is refused instead of being sent to H3 as is. Everything that can be checked without the GPU is checked first (image, prompt screen, length,
    model files); then Ollama is unloaded, the card is cooled and ComfyUI is started if it is not already running."""
    start = start or client.start_server
    server_up = server_up or client.is_server_running
    free_vram = free_vram or client.free_vram
    if not os.path.isfile(image):
        raise gc.UsageError(f"找不到首幀：{image}")
    frames = check_seconds(seconds, allow_long)
    log(f"首幀：{os.path.basename(image)}｜{seconds:g} 秒 = {frames} 幀｜{width}×{height}")
    content_gate.check(description, soundscape)  # explicit / minor wording: refused before anything else, no bypass
    original_description = description
    for text in (description, soundscape):
        screen(LINE_SPAN.sub("", text or ""))  # also catches the Chinese blocked terms before translating
    log("內容檢查：通過")
    if has_cjk(LINE_SPAN.sub("", description)) or has_cjk(soundscape):
        if translate is None:
            raise gc.UsageError("描述含中文（<d>…</d> 裡的台詞除外）：要先翻成英文，H3 的描述只收英文")
        log("描述含中文：翻譯成英文中（台詞不翻）…")
        description = translate_outside_lines(description, translate)
        soundscape = translate_outside_lines(soundscape, translate) if soundscape else soundscape
        content_gate.check(description, soundscape)  # and again on what the translator returned
        log(f"英文描述：{description}")
    prompt = assemble_prompt(image, description, soundscape, static)  # raises PROMPT_REJECTED on a blocked term
    missing = missing_files()
    if missing:
        raise gc.UsageError("缺模型檔（ComfyUI/models 下）：" + ", ".join(missing))
    seed = int(seed) if seed is not None else int(time.time()) % 1_000_000
    tag = tag or f"{os.path.splitext(os.path.basename(image))[0]}_{seed}"
    if guard is None:
        import gpu_guard as guard
    log("卸載聊天模型（Ollama）、等 GPU 降溫…")
    _guard_call(guard.before_image, log)
    _guard_call(guard.wait_until_cool, log)
    log(f"目前狀態：{machine_status()}")
    started = None
    if not server_up():
        log("ComfyUI 沒在跑：啟動中（第一次約 30 秒）…")
        started = start()
    try:
        free = free_vram_mib()
        if free is not None and free < MIN_FREE_VRAM_MIB:
            # The picture section leaves its model (Z-Image, ~13 GB) in ComfyUI's memory: measured 3.2 GB free after
            # one picture, 15 GB after ComfyUI was asked to unload. So ask it to unload first, then look again.
            log(f"顯存只剩 {free} MiB（要 {MIN_FREE_VRAM_MIB}）：請 ComfyUI 卸載上一個模型…")
            free_vram()
            deadline = time.time() + unload_wait
            while True:
                free = free_vram_mib()
                if free is None or free >= MIN_FREE_VRAM_MIB or time.time() >= deadline:
                    break
                time.sleep(1)
            log(f"卸載後顯存可用 {free} MiB")
        if free is not None and free < MIN_FREE_VRAM_MIB:
            raise gc.UsageError(
                f"顯示卡只剩 {free} MiB 可用（要 {MIN_FREE_VRAM_MIB}）。先關掉佔顯存的程式（聊天模型、別的 ComfyUI 工作）"
            )
        name = client.upload_reference_image(image)
        graph = build_graph(
            prompt, name, seed=seed, width=width, height=height, frames=frames, megapixels=megapixels, tag=tag
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"graph_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=1)
        base = client.COMFYUI_URL
        r = post(base + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=60)
        r.raise_for_status()
        pid = r.json()["prompt_id"]
        log(f"已送出給 ComfyUI（seed {seed}）。載入模型時 RAM 會先吃滿再降下來，5 秒影片本機約 8～10 分鐘")
        t0 = last = time.time()
        while True:
            h = get(f"{base}/history/{pid}", timeout=30).json()
            if pid in h and h[pid].get("status", {}).get("completed") is not None:
                break
            if time.time() - t0 > TIMEOUT_SECONDS:
                raise gc.UsageError(f"H3 沒有在 {TIMEOUT_SECONDS} 秒內完成（ComfyUI 的終端機會有原因）")
            if time.time() - last >= STATUS_EVERY_SECONDS:
                last = time.time()
                log(f"生成中 {round(last - t0)} 秒｜{machine_status()}")
            time.sleep(POLL_SECONDS)
        status = h[pid]["status"]
        if status.get("status_str") != "success":
            raise gc.UsageError(f"H3 失敗：{json.dumps(status.get('messages', [])[-2:], ensure_ascii=False)[:800]}")
        seconds_taken = round(time.time() - t0)
        found = sorted(
            glob.glob(os.path.join(client.COMFYUI_DIR, "output", "video", f"h3local_{tag}_*.mp4")), key=os.path.getmtime
        )
        if not found:
            raise gc.UsageError("ComfyUI 說完成了，但找不到輸出的 mp4")
        dest = os.path.join(out_dir, f"h3_local_{tag}.mp4")
        shutil.copy2(found[-1], dest)
        log(f"完成：{seconds_taken} 秒 → {dest}")
        meta = {
            "tag": tag,
            "image": os.path.abspath(image),
            "seed": seed,
            "width": width,
            "height": height,
            "frames": frames,
            "seconds_requested": seconds,
            "steps": h3.STEPS,
            "megapixels": megapixels,
            "prompt": prompt,
            "description_used": description,
            "description_original": original_description if original_description != description else None,
            "seconds_taken": seconds_taken,
            "output": dest,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": UNET,
            "lora": LORA,
        }
        with open(os.path.join(out_dir, f"h3_local_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return meta
    finally:
        if started is not None and not keep_comfyui:
            started.terminate()  # we started it, so we free the ~13 GB of VRAM and RAM again


# ---------------------------------------------------------------- stills for the first frame (the GUI's picture section)

STILL_MODELS = {
    "z_image": {"label": "Z-Image Turbo（可商用，約 30 秒）", "license": None},
    "qwen_image_2_1": {"label": "Qwen-Image 2.1（非商用，只做內部測試，約 2 分鐘）", "license": None},
}
STILLS_DIR = os.path.join(OUT_DIR, "stills")
# How many people are in the picture. 1 keeps the project's solo terms (the promos are "the character alone"); 2 and 3
# drop them, state the count and keep everyone an adult. The age-safety negatives and the prompt screen apply to all.
PEOPLE_ALIASES = {"single": 1, "multi": 2}  # the words the description writer uses
PEOPLE_CHOICES = {
    1: ("1 人", ""),
    2: ("2 人", "Exactly two adult people in the frame, both in their twenties. "),
    3: ("3～4 人", "A small group of three or four adult friends in their twenties, all clearly visible. "),
}


def generate_still(
    prompt,
    model="z_image",
    *,
    seed=None,
    out_dir=STILLS_DIR,
    guard=None,
    submit=None,
    server_up=None,
    start=None,
    translate=None,
    people=1,
    negative="",
    log=print,
):
    """One 720x1280 picture for the first frame, from text, on the local card. Same assembly as every other still of
    this project (age-safety negatives and the cfg floor come from _build_prompt_and_negative / the client), and the
    text goes through h3_colab.screen_prompt first, because this picture is meant to become an H3 first frame.
    A Qwen picture is written with its non-commercial licence in the record."""
    import asset_library as al
    import concept_stills as cs

    if model not in STILL_MODELS:
        raise gc.UsageError(f"沒有這個圖片模型：{model}（可用：{', '.join(STILL_MODELS)}）")
    people = PEOPLE_ALIASES.get(people, people)
    if people not in PEOPLE_CHOICES:
        raise gc.UsageError(f"人數要是 {', '.join(str(k) for k in PEOPLE_CHOICES)}（3 = 三到四人）")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gc.UsageError("請輸入圖片的描述（英文最準；人物要是成年人）")
    log(f"模型：{STILL_MODELS[model]['label']}｜人數：{PEOPLE_CHOICES[people][0]}")
    content_gate.check(prompt)  # refused before any translation, unloading or GPU work, in Chinese and in English
    screen(prompt)  # the Chinese blocked terms are caught here, before any translation
    log("內容檢查：通過")
    original_prompt = prompt
    if has_cjk(prompt):
        if translate is None:
            raise gc.UsageError("描述含中文：要先翻成英文（GUI 會自動翻譯）")
        log("描述含中文：翻譯成英文中…")
        prompt = adultize(translate(prompt).strip())
        log(f"英文描述：{prompt}")
        content_gate.check(prompt)  # the translator's English is checked again
        screen(prompt)  # and the English ones after it
    # The user's own negative terms (Chinese or English). They are appended AFTER the fixed safety negatives, which
    # cannot be removed or overridden, and they are not run through the content gate: a negative list legitimately
    # contains words like "nude" or "child".
    negative = (negative or "").strip()
    if negative and has_cjk(negative):
        if translate is None:
            raise gc.UsageError("負面詞含中文：要先翻成英文（GUI 會自動翻譯）")
        negative = translate(negative).strip()
        log(f"負面詞（英文）：{negative}")
    extra_negative = negative
    qwen = model == "qwen_image_2_1"
    server_up = server_up or client.is_server_running
    start = start or client.start_server
    submit = submit or (
        client.submit_txt2img_generation_qwen_image21 if qwen else client.submit_txt2img_generation_zimage
    )
    bible = cs.load_bible(os.path.join(REPO_ROOT, "projects", "activity_promo"))
    model_key = cs.IMAGE_MODELS[model]["key"]
    full, negative = gc._build_prompt_and_negative(
        PEOPLE_CHOICES[people][1] + prompt,
        f"{al.PERSON_NEGATIVE}, {negative}" if negative else al.PERSON_NEGATIVE,
        "safe",
        None,
        bible["style_positive"],
        cs.style_negative(bible),
        model_key,
        solo=people == 1,
        allow_text=False,
    )
    steps, cfg = (
        (client.QWEN_IMAGE21_STEPS, client.QWEN_IMAGE21_CFG)
        if qwen
        else (bible["sampler"]["steps"], bible["sampler"]["cfg"])
    )
    cfg = max(cfg, client.SAFETY_MIN_CFG)
    seed = int(seed) if seed is not None else int(time.time()) % 1_000_000
    if guard is None:
        import gpu_guard as guard
    log("卸載聊天模型（Ollama）、等 GPU 降溫…")
    _guard_call(guard.before_image, log)
    _guard_call(guard.wait_until_cool, log)
    log(f"目前狀態：{machine_status()}")
    started = None
    if not server_up():
        log("ComfyUI 沒在跑：啟動中（第一次約 30 秒）…")
        started = start()
    try:
        tag = f"{model}_{seed}"
        log(f"生成中（seed {seed}，{steps} 步，cfg {cfg}）。Z-Image 約 30 秒、Qwen 約 2 分鐘；第一次要載入模型會更久")
        t0 = time.time()
        with _Heartbeat(log, "生成中"):
            raw = submit(
                prompt=full,
                negative_prompt=negative,
                seed=seed,
                filename_prefix=f"h3still_{tag}",
                model=model_key,
                width=bible["canvas"]["width"],
                height=bible["canvas"]["height"],
                steps=steps,
                cfg=cfg,
            )
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, f"still_{tag}.png")
        shutil.move(raw, dest)
        log(f"完成：{round(time.time() - t0)} 秒 → {dest}")
        meta = {
            "output": dest,
            "model": model,
            "people": people,
            "negative_extra": extra_negative or None,
            "seed": seed,
            "prompt": full,
            "prompt_original": original_prompt if original_prompt != prompt else None,
            "prompt_english": prompt,
            "negative_prompt": negative,
            "steps": steps,
            "cfg": cfg,
            "license": client.QWEN_IMAGE21_LICENSE if qwen else None,
            "note": "internal test only, not for ads" if qwen else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(os.path.join(out_dir, f"still_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return meta
    finally:
        if started is not None:
            started.terminate()


def _translator():
    """The project's own Chinese -> English translation (the one the GUI's other tabs use)."""
    import translate_prompt

    def translate(text):
        try:
            return translate_prompt.translate_to_english(text)
        except RuntimeError as exc:
            raise gc.UsageError(f"翻譯失敗，沒有送出：{exc}") from None

    return translate


def main(argv=None):
    p = argparse.ArgumentParser(description="MiniMax H3 image-to-video on the local GPU (one clip, one command)")
    p.add_argument("image", help="first frame (a still this project made: a fictional adult or an empty place)")
    p.add_argument("--description", help="what happens in the clip, including the line in <d>[Chinese] ...</d>")
    p.add_argument("--description-file", help="read the description from a text file instead")
    p.add_argument("--idea", help="a Chinese idea; local qwen3.5:9b writes the English description around --line")
    p.add_argument("--line", help="what she or he says (Chinese, up to 20 characters); used with --idea")
    p.add_argument(
        "--gender",
        choices=("woman", "man", "couple"),
        default="woman",
        help="used with --idea; couple = a woman (S1) and a man (S2)",
    )
    p.add_argument("--line2", help="the man's answer, only with --gender couple")
    p.add_argument("--write-only", action="store_true", help="with --idea: print the description and stop (no GPU)")
    p.add_argument("--soundscape", default="")
    p.add_argument("--static", action="store_true", help="add the static-subject constraint (cat, steam, background)")
    p.add_argument("--seconds", type=float, default=SAFE_SECONDS)
    p.add_argument("--allow-long", action="store_true", help="allow more than 5 s (not measured on this card)")
    p.add_argument("--seed", type=int)
    p.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    p.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    p.add_argument("--megapixels", type=float, default=DEFAULT_MEGAPIXELS)
    p.add_argument("--tag")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--keep-comfyui", action="store_true", help="leave ComfyUI running afterwards")
    a = p.parse_args(argv)
    text = a.description
    if a.description_file:
        with open(a.description_file, encoding="utf-8") as f:
            text = f.read().strip()
    if a.idea:
        if text:
            raise gc.UsageError("--idea 和 --description 擇一")
        if not a.line:
            raise gc.UsageError("--idea 要搭配 --line（台詞）")
        import h3_idea

        text = h3_idea.write_description(a.idea, a.line, a.gender, line2=a.line2)
        print("描述：" + text, flush=True)
        if a.write_only:
            return
    if not text:
        raise gc.UsageError("要給 --description、--description-file 或 --idea")
    meta = generate(
        a.image,
        text,
        soundscape=a.soundscape,
        static=a.static,
        seconds=a.seconds,
        seed=a.seed,
        width=a.width,
        height=a.height,
        megapixels=a.megapixels,
        tag=a.tag,
        out_dir=a.out_dir,
        allow_long=a.allow_long,
        keep_comfyui=a.keep_comfyui,
        translate=_translator(),
    )
    print(json.dumps({k: meta[k] for k in ("output", "seconds_taken", "seed", "frames")}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except gc.UsageError as exc:
        raise SystemExit(str(exc)) from None
