"""Memory/speed benchmark for the ComfyUI-backed generation pipeline.

Runs two tests (768x768 and 1024x1024, batch=1, 10 images each), recording
VRAM/RAM before/peak/after and per-image generation time via ComfyUI's
/system_stats endpoint - no local torch/CUDA access needed.
"""

import argparse
import threading
import time

import comfyui_client as client
import requests

PROMPT = (
    "mylora, 28 year old adult woman, long straight black hair, "
    "dark brown eyes, east asian, oval face, mature adult features, natural healthy build, "
    "front view portrait, standing straight, wearing white t-shirt, soft natural window light, "
    "plain white studio background, photorealistic, high detail"
)
NEGATIVE = (
    "nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, teen, teenager, "
    "underage, young girl, lowres, blurry, deformed, extra limbs, bad anatomy, watermark, text"
)


class VramSampler:
    """Polls /system_stats in the background while a generation runs, to
    capture peak usage (a single before/after snapshot would miss it)."""

    def __init__(self, interval=0.3):
        self.interval = interval
        self.peak_vram_mb = 0
        self.peak_ram_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                r = requests.get(f"{client.COMFYUI_URL}/system_stats", timeout=5)
                data = r.json()
                device = data["devices"][0]
                used_vram = (device["vram_total"] - device["vram_free"]) / (1024 ** 2)
                used_ram = (data["system"]["ram_total"] - data["system"]["ram_free"]) / (1024 ** 2)
                self.peak_vram_mb = max(self.peak_vram_mb, used_vram)
                self.peak_ram_mb = max(self.peak_ram_mb, used_ram)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def run_test(name, width, height, count):
    print(f"\n=== {name}: {width}x{height}, batch=1, {count} images ===", flush=True)
    before_vram, total_vram = client.log_gpu_memory(f"{name}_before")

    per_image_times = []
    per_image_vram_after = []

    for i in range(count):
        t0 = time.time()
        with VramSampler() as sampler:
            client.submit_txt2img_generation(
                prompt=PROMPT,
                negative_prompt=NEGATIVE,
                seed=42000 + i,
                filename_prefix=f"benchmark_{name}_{i}",
                width=width,
                height=height,
            )
        elapsed = time.time() - t0
        per_image_times.append(elapsed)
        used_after, _ = client.log_gpu_memory(f"{name}_image_{i}")
        per_image_vram_after.append(used_after)
        print(f"  image {i+1}/{count}: {elapsed:.1f}s, VRAM peak during gen: {sampler.peak_vram_mb:.0f} MB, "
              f"VRAM after: {used_after:.0f} MB", flush=True)

    after_vram, _ = client.log_gpu_memory(f"{name}_after")

    avg_time = sum(per_image_times) / len(per_image_times)
    leak_signal = per_image_vram_after[-1] - per_image_vram_after[0] if len(per_image_vram_after) > 1 else 0

    print(f"\n--- {name} summary ---")
    print(f"VRAM before: {before_vram:.0f} MB")
    print(f"VRAM after:  {after_vram:.0f} MB")
    print(f"VRAM at image 1: {per_image_vram_after[0]:.0f} MB")
    if len(per_image_vram_after) >= 5:
        print(f"VRAM at image 5: {per_image_vram_after[4]:.0f} MB")
    print(f"VRAM at image {len(per_image_vram_after)}: {per_image_vram_after[-1]:.0f} MB")
    print(f"Average generation time: {avg_time:.1f}s")
    print(f"Possible memory leak (image1 -> imageN delta): {leak_signal:+.0f} MB")
    print(f"Memory leak: {'YES' if leak_signal > 500 else 'NO'}")

    return {
        "name": name,
        "before_vram": before_vram,
        "after_vram": after_vram,
        "avg_time": avg_time,
        "per_image_vram_after": per_image_vram_after,
        "leak_signal": leak_signal,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    results = [
        run_test("test_a_768", 768, 768, args.count),
        run_test("test_b_1024", 1024, 1024, args.count),
    ]

    print("\n\n========== FINAL REPORT ==========")
    for r in results:
        print(f"{r['name']}: avg {r['avg_time']:.1f}s/image, "
              f"VRAM {r['before_vram']:.0f}->{r['after_vram']:.0f} MB, "
              f"leak signal {r['leak_signal']:+.0f} MB "
              f"({'YES' if r['leak_signal'] > 500 else 'NO'})")
