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
import threading
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


# ---------------------------------------------------------------- keeping the card cool between images
#
# One Z-Image render takes this card from about 53 C to about 80 C in 100 s (measured 2026-10-04), and a
# batch run back to back sits at 83-84 C, the hardware throttle point, from the third image on. The owner
# would rather wait than run it that hot ("時間長沒關係，要讓顯示卡可以用更久"). So every image starts only
# once the card is back under IMAGE_START_MAX_C. This bounds the time spent near the throttle point; it does
# not lower the peak inside one render - only a lower power limit (nvidia-smi -pl, the owner's setting) does.

IMAGE_START_MAX_C = float(os.environ.get("GPU_IMAGE_START_MAX_C", "60"))
COOL_WAIT_SECONDS = 1200
THROTTLE_C = 83.0
SAMPLE_SECONDS = 5


def wait_until_cool(max_c=None, *, run=subprocess.run, sleep=time.sleep, clock=time.monotonic,
                    timeout=COOL_WAIT_SECONDS, log=print):
    """Block until the GPU is at or below max_c. Returns {'waited_seconds', 'temp_c'} (temp_c None when
    nvidia-smi cannot be read - then there is nothing to wait on). max_c 0 or below turns the gate off.
    Raises GpuBusy when the card has not cooled within `timeout`: something else is heating it."""
    max_c = IMAGE_START_MAX_C if max_c is None else max_c
    started = clock()
    if max_c <= 0:
        return {"waited_seconds": 0.0, "temp_c": None}
    announced = False
    while True:
        state = gpu_state(run)
        if state is None or state["temp_c"] <= max_c:
            return {"waited_seconds": round(clock() - started, 1), "temp_c": state and state["temp_c"]}
        if not announced:
            log(f"[gpu_guard] GPU {state['temp_c']:.0f}°C，等降到 {max_c:.0f}°C 再生下一張")
            announced = True
        if clock() - started >= timeout:
            raise GpuBusy(f"等了 {timeout} 秒 GPU 還在 {state['temp_c']:.0f}°C，沒有降到 {max_c:.0f}°C；"
                          "有別的程式在用顯示卡，或散熱有問題")
        sleep(SAMPLE_SECONDS)


def power_limit_w(run=subprocess.run):
    """The card's current power limit in watts, or None. Recorded with every thermal log: it is the one
    setting that changes the peak temperature, and it resets on reboot."""
    try:
        out = run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=15).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def summarize(samples, interval=SAMPLE_SECONDS):
    """Peak temperature and the seconds spent at or above the throttle point, from [(elapsed, temp_c), ...]."""
    temps = [t for _, t in samples if t is not None]
    if not temps:
        return {"samples": 0, "peak_c": None, "mean_c": None, "seconds_at_throttle": 0}
    return {"samples": len(temps), "peak_c": max(temps), "mean_c": round(sum(temps) / len(temps), 1),
            "seconds_at_throttle": sum(interval for t in temps if t >= THROTTLE_C)}


class ThermalLog:
    """Samples the GPU every SAMPLE_SECONDS on a thread and writes elapsed,temp_c,used_mib,event to a CSV, so a
    batch leaves its own heating and cooling curve behind. `mark` adds an event row (image started)."""

    def __init__(self, path, run=subprocess.run, interval=SAMPLE_SECONDS):
        self.path, self.run, self.interval = path, run, interval
        self.samples, self.started = [], None
        self._lock, self._stop, self._thread = threading.Lock(), threading.Event(), None

    def _write(self, state, event=""):
        elapsed = time.monotonic() - self.started
        temp = state["temp_c"] if state else ""
        used = state["used_mib"] if state else ""
        with self._lock, open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{elapsed:.0f},{temp},{used},{event}\n")
        if not event:
            self.samples.append((elapsed, state["temp_c"] if state else None))

    def start(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.started = time.monotonic()
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# power_limit_w={power_limit_w(self.run)} start_max_c={IMAGE_START_MAX_C}\n")
            f.write("elapsed_s,temp_c,used_mib,event\n")

        def loop():
            while not self._stop.wait(self.interval):
                self._write(gpu_state(self.run))

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def mark(self, event):
        if self.started is not None:
            self._write(gpu_state(self.run), event.replace(",", " "))

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval + 2)
        return dict(summarize(self.samples, self.interval), log=self.path)
