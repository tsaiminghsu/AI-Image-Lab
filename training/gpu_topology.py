"""Which graphics card does what, when the machine has more than one.

The GUI runs two kinds of GPU work: ComfyUI (pictures and video, 16 GB class) and the local Qwen chat model (qwen3.5:9b
through Ollama, about 6.7 GB). On a single card they take turns: Qwen is unloaded before every picture and ComfyUI's
models are freed before every Qwen call. With a second card the two can be kept apart, so Qwen can stay loaded and chat
and rendering can run at the same time.

Default (no settings): ComfyUI goes to the card with the most memory, Qwen to the other card with the most memory, as
long as that card can hold it (LLM_MIN_TOTAL_MIB) and, when its server is not running yet, has room right now
(LLM_MIN_FREE_MIB). Anything else - one card, a second card that is too small or occupied, GPU_SPLIT=off - leaves the old
single-card behaviour untouched. Overrides, read from the environment: IMAGE_GPU and LLM_GPU (an index, a UUID or a part
of the name), GPU_SPLIT=off to switch the whole thing off (nothing is pinned then).

Cards are pinned with CUDA_VISIBLE_DEVICES set to a UUID (and CUDA_DEVICE_ORDER=PCI_BUS_ID), so the choice does not depend
on how the driver orders them. nvidia-smi's own index is used for the telemetry (temperature, memory) of the picture card.

No gradio and no torch here; the nvidia-smi call is injectable.
"""

import os
import subprocess
import time
from dataclasses import dataclass

LLM_MIN_TOTAL_MIB = 7500  # a 9B Q4 model plus its context needs about 6.7 GB; a 6 GB card cannot hold it
LLM_MIN_FREE_MIB = 7000  # needed free on the card when the pinned server has to be started
PINNED_PORT = 21435  # the Ollama server this tool starts for the Qwen card (21434 is the shared one)
KEEP_ALIVE_SPLIT = "10m"  # Qwen stays on its own card between chat turns; on a shared card it is unloaded at once
CACHE_SECONDS = 15
QUERY = "index,name,uuid,memory.total,memory.used"


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    uuid: str
    total_mib: float
    used_mib: float

    @property
    def free_mib(self):
        return self.total_mib - self.used_mib

    @property
    def label(self):
        return self.name.replace("NVIDIA GeForce ", "")


@dataclass(frozen=True)
class Plan:
    image: Gpu  # ComfyUI (pictures, video)
    llm: Gpu  # Qwen
    split: bool  # the two are on different cards
    reason: str


def detect(run=subprocess.run):
    """Every card nvidia-smi lists; an empty list when it cannot be read."""
    try:
        out = run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"], capture_output=True,
                  text=True, timeout=15).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(Gpu(int(parts[0]), parts[1], parts[2], float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return gpus


def _find(gpus, spec):
    spec = (spec or "").strip()
    if not spec:
        return None
    for g in gpus:
        if spec.isdigit() and g.index == int(spec) or spec == g.uuid:
            return g
    low = spec.lower()
    return next((g for g in gpus if low in g.name.lower()), None)


def _biggest(gpus):
    return max(gpus, key=lambda g: (g.total_mib, -g.index))


def _split_off(env=None):
    env = os.environ if env is None else env
    return (env.get("GPU_SPLIT") or "").strip().lower() in ("0", "off", "false", "no")


def make_plan(gpus, env=None):
    """The plan for these cards, or None when nothing should be pinned (no nvidia-smi or GPU_SPLIT=off). One card gives
    a plan with split False."""
    env = os.environ if env is None else env
    if not gpus or _split_off(env):
        return None
    image = _find(gpus, env.get("IMAGE_GPU")) or _biggest(gpus)
    others = [g for g in gpus if g is not image]
    if not others:
        return Plan(image, image, False, f"只有一張卡（{image.label}）：圖片、影片和 Qwen 輪流用")
    llm = _find(others, env.get("LLM_GPU")) or _biggest(others)
    if llm.total_mib < LLM_MIN_TOTAL_MIB:
        return Plan(image, image, False,
                    f"{llm.label} 只有 {llm.total_mib / 1024:.0f} GB，放不下 Qwen：和圖片共用 {image.label}")  # fmt: skip
    return Plan(image, llm, True, f"圖片／影片 → {image.label}；Qwen → {llm.label}")


_cache = {"at": None, "plan": None, "gpus": []}


def reset():
    _cache.update(at=None, plan=None, gpus=[])


def _refresh(run=subprocess.run, now=time.monotonic):
    if _split_off():
        _cache.update(at=None, gpus=[], plan=None)  # switched off: nvidia-smi is not even asked
        return
    if _cache["at"] is not None and now() - _cache["at"] < CACHE_SECONDS:
        return
    gpus = detect(run)
    _cache.update(at=now(), gpus=gpus, plan=make_plan(gpus))


def plan(run=subprocess.run, now=time.monotonic):
    """The current Plan (cached for CACHE_SECONDS), or None when nothing is pinned."""
    _refresh(run, now)
    return _cache["plan"]


def gpus(run=subprocess.run, now=time.monotonic):
    _refresh(run, now)
    return list(_cache["gpus"])


def planned_split():
    p = plan()
    return bool(p and p.split)


def pinned_server_up(get=None):
    """True when the Ollama server this tool started for the Qwen card answers."""
    import requests

    get = get or requests.get
    try:
        return get(f"http://127.0.0.1:{PINNED_PORT}/api/tags", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def effective_split(get=None):
    """True when Qwen really is on its own card right now: a split is planned and either the pinned server is already
    running (it holds the card) or the card has room to start it. A second card that something else filled falls back to
    the single-card behaviour instead of failing."""
    p = plan()
    if not (p and p.split):
        return False
    if pinned_server_up(get):
        return True
    fresh = next((g for g in gpus() if g.index == p.llm.index), None)
    return bool(fresh and fresh.free_mib >= LLM_MIN_FREE_MIB)


def _pin(gpu):
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu.uuid}


def image_env():
    """Environment for the ComfyUI process: only the picture card visible. Empty when nothing is pinned."""
    p = plan()
    return _pin(p.image) if p else {}


def llm_env():
    """Environment for the Ollama server of the Qwen card. Empty when nothing is pinned.

    CUDA_VISIBLE_DEVICES alone is not enough: Ollama 0.40 also enumerates cards through Vulkan (OLLAMA_VULKAN is on), and
    the Vulkan backend ignores CUDA_VISIBLE_DEVICES, so the picture card still showed up as a second compute device and,
    having more free memory, won (found 2026-10-23: Qwen loaded on the RTX 5060 Ti while the RTX 2070 stayed at 0 MiB).
    Vulkan is switched off for this server so the CUDA card named above is the only one it can use."""
    p = plan()
    return {**_pin(p.llm), "OLLAMA_VULKAN": "0"} if p and p.split else {}


def llm_keep_alive(get=None):
    """Ollama's keep_alive for a Qwen call: 0 (unload at once) on a shared card, a few minutes on its own card."""
    return KEEP_ALIVE_SPLIT if effective_split(get) else 0


def smi_command():
    """['nvidia-smi'] or ['nvidia-smi', '-i', N]: the telemetry of the picture card, not of whichever the driver lists first."""
    p = plan()
    return ["nvidia-smi", "-i", str(p.image.index)] if p else ["nvidia-smi"]


def describe():
    """One line for the GUI: which card does what."""
    p = plan()
    if p is None:
        return "GPU：沒有指定（單卡或已用 GPU_SPLIT=off 關閉）"
    return f"GPU：{p.reason}"
