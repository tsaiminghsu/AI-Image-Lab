"""Keep the chat model and image generation from sharing the 8 GB card.

Open WebUI's chat model runs in Ollama on the same RTX 2070 that ComfyUI renders on. With both in VRAM a
Z-Image job overflows into shared memory and sits at step 0/8 until it times out (2026-10-03: story 3's first
image, 1200 s, nothing rendered). So before every image the caller asks this module to unload whatever Ollama
holds and to wait until it is gone; a chat message during a batch reloads the model, and the next image waits
again instead of stalling.

Total VRAM is only checked before ComfyUI starts: once it is running, nvidia-smi on Windows cannot tell its
memory from anyone else's (per-process usage reads N/A).

Only `requests` and the standard library, like comfyui_client.
"""

import os
import subprocess
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
UNLOAD_TIMEOUT_SECONDS = 120
START_MAX_USED_MIB = 2500   # desktop apps alone were measured at 0.6-1.9 GB; above this something else holds the card
START_WAIT_SECONDS = 300
POLL_SECONDS = 3


class GpuBusy(RuntimeError):
    """The card did not become free in time. The caller fails that step with this message; it never renders anyway."""


def loaded_models(http=requests, url=None):
    """Names of the models Ollama holds in memory. An Ollama that is not running holds nothing."""
    try:
        r = http.get(f"{url or OLLAMA_URL}/api/ps", timeout=5)
        r.raise_for_status()
        return [m.get("name") or m.get("model") for m in r.json().get("models", []) if m.get("name") or m.get("model")]
    except (requests.exceptions.RequestException, ValueError):
        return []


def unload_models(http=requests, url=None, *, timeout=UNLOAD_TIMEOUT_SECONDS, sleep=time.sleep, clock=time.monotonic):
    """Ask Ollama to drop every loaded model (keep_alive 0) and wait until /api/ps is empty. Returns the names
    that were unloaded; raises GpuBusy when a model is still loaded after `timeout` (someone keeps chatting)."""
    base = url or OLLAMA_URL
    names = loaded_models(http, base)
    if not names:
        return []
    for name in names:
        try:
            http.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
        except requests.exceptions.RequestException:
            pass   # judged by /api/ps below, not by this call's answer
    deadline = clock() + timeout
    while loaded_models(http, base):
        if clock() >= deadline:
            raise GpuBusy(f"Ollama 還佔著顯示卡（{', '.join(loaded_models(http, base))}），等了 {timeout} 秒沒有釋放；"
                          "生圖期間請先不要聊天")
        sleep(POLL_SECONDS)
    return names


def gpu_state(run=subprocess.run):
    """{'temp_c', 'used_mib', 'total_mib'} from nvidia-smi, or None when it cannot be read."""
    try:
        out = run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        temp, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        return {"temp_c": temp, "used_mib": used, "total_mib": total}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def before_batch(*, max_start_temp_c, http=requests, run=subprocess.run, sleep=time.sleep, clock=time.monotonic,
                 wait_seconds=START_WAIT_SECONDS, log=print):
    """Run once, before ComfyUI is started: unload the chat model, then wait for the card to be free and cool
    enough. Raises GpuBusy when it is not within wait_seconds. An unreadable nvidia-smi is not a reason to
    refuse - the per-image unload still applies."""
    unloaded = unload_models(http, sleep=sleep, clock=clock)
    if unloaded:
        log(f"[gpu_guard] 已請 Ollama 卸載：{', '.join(unloaded)}")
    deadline = clock() + wait_seconds
    while True:
        state = gpu_state(run)
        if state is None:
            return {"unloaded": unloaded, "gpu": None}
        hot = state["temp_c"] >= max_start_temp_c
        full = state["used_mib"] > START_MAX_USED_MIB
        if not hot and not full:
            return {"unloaded": unloaded, "gpu": state}
        if clock() >= deadline:
            why = []
            if full:
                why.append(f"VRAM 已用 {state['used_mib']:.0f} MiB（上限 {START_MAX_USED_MIB}），有別的程式佔著顯示卡")
            if hot:
                why.append(f"GPU {state['temp_c']:.0f}°C，還沒降到 {max_start_temp_c}°C")
            raise GpuBusy("；".join(why))
        sleep(POLL_SECONDS)


def before_image(*, http=requests, sleep=time.sleep, clock=time.monotonic, log=print):
    """Run before every image: a chat message since the last one has reloaded the model."""
    unloaded = unload_models(http, sleep=sleep, clock=clock)
    if unloaded:
        log(f"[gpu_guard] 生圖前卸載 Ollama：{', '.join(unloaded)}")
    return unloaded
