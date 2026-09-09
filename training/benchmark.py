"""Memory/speed benchmark for the ComfyUI-backed generation pipeline.

Runs two tests (768x768 and 1024x1024, batch=1, 10 images each), recording
VRAM/RAM before/peak/after and per-image generation time via ComfyUI's
/system_stats endpoint - no local torch/CUDA access needed.
"""

import argparse
import json
import threading
import time
import uuid

import comfyui_client as client
import pose_skeletons
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


# node id -> HQ pipeline stage label, for per-stage timing (see run_hq_test).
HQ_STAGE_NAMES = {
    "3": "pass1_base",
    "21": "openpose",           # photo pose path only; a library skeleton skips this node
    "22": "controlnet_load",    # skeleton path: the ControlLora build lands mostly in pass1_base
    "31": "esrgan_upscale",
    "34": "pass2_hires",
    "41": "facedetailer_face",
    "43": "facedetailer_hand",
    "9": "save",
}
HQ_TIME_BUDGET_SECONDS = 300  # the "5 minute" target this whole path is built to hit


class WsStageTimer:
    """Listens on ComfyUI's /ws for this client_id and records the wall-clock
    time each node STARTS executing (the "executing" message). Diffing
    consecutive starts gives each node's duration - ComfyUI's /history doesn't
    expose per-node timing. Degrades gracefully to no per-stage breakdown if
    the `websockets` package isn't importable."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.node_start = []  # list of (node_id, timestamp) in execution order
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = False

    def _run(self):
        try:
            from websockets.sync.client import connect
        except Exception:
            return
        self.available = True
        ws_url = (client.COMFYUI_URL.replace("https://", "wss://").replace("http://", "ws://")
                  + f"/ws?clientId={self.client_id}")
        try:
            with connect(ws_url, open_timeout=10) as ws:
                while not self._stop.is_set():
                    try:
                        raw = ws.recv(timeout=1)
                    except TimeoutError:
                        continue
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    if msg.get("type") == "executing":
                        data = msg.get("data", {})
                        self.node_start.append((data.get("node"), time.time()))
        except Exception:
            pass

    def __enter__(self):
        self._thread.start()
        time.sleep(0.3)  # let the socket connect before the prompt is submitted
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3)

    def stage_durations(self, total_end):
        """Turn recorded node starts into {stage_label: seconds}. A node's
        duration is until the next node starts (or total_end for the last)."""
        out = {}
        starts = [s for s in self.node_start if s[0] is not None]
        for idx, (node_id, ts) in enumerate(starts):
            end = starts[idx + 1][1] if idx + 1 < len(starts) else total_end
            label = HQ_STAGE_NAMES.get(node_id, f"node_{node_id}")
            out[label] = out.get(label, 0.0) + (end - ts)
        return out


def run_hq_test(name, width, height, count, anchor=None, character_lora=None, controlnet=False, pose=None):
    print(f"\n=== {name}: HQ two-pass, final {width}x{height}, {count} images "
          f"(insightface provider={client.INSIGHTFACE_PROVIDER}, controlnet={controlnet}, pose={pose}) ===", flush=True)
    before_vram, total_vram = client.log_gpu_memory(f"{name}_before")

    ip_name = client.upload_reference_image(anchor) if anchor else None
    # A --pose picks a real library skeleton (fed to ControlNet directly, no
    # preprocessor); plain --controlnet without --pose falls back to reusing the
    # anchor photo as a stand-in (goes through OpenposePreprocessor).
    pose_is_skeleton = False
    if pose:
        skel = pose_skeletons.resolve(pose)
        if not skel:
            raise SystemExit(f"unknown pose {pose!r} - choices: {pose_skeletons.list_names()}")
        pose_name = client.upload_reference_image(skel)
        pose_is_skeleton = True
    elif controlnet:
        pose_name = ip_name  # reuse the anchor's own pose as a stand-in skeleton (needs --anchor)
    else:
        pose_name = None

    per_image_times = []
    passed = 0
    for i in range(count):
        client_id = uuid.uuid4().hex
        timer = WsStageTimer(client_id)
        t0 = time.time()
        with VramSampler() as sampler, timer:
            client.submit_generation_hq(
                prompt=PROMPT,
                negative_prompt=NEGATIVE,
                seed=42000 + i,
                filename_prefix=f"bench_hq_{name}_{i}",
                ip_adapter_image_filename=ip_name,
                width=width,
                height=height,
                character_lora=character_lora,
                pose_image_filename=pose_name,
                pose_is_skeleton=pose_is_skeleton,
                hires=True,
                use_facedetailer=True,
                client_id=client_id,
            )
        elapsed = time.time() - t0
        per_image_times.append(elapsed)
        used_after, _ = client.log_gpu_memory(f"{name}_image_{i}")
        ok = elapsed < HQ_TIME_BUDGET_SECONDS
        passed += ok
        stages = timer.stage_durations(time.time())
        stage_str = ", ".join(f"{k} {v:.0f}s" for k, v in stages.items()) if stages else "(no per-stage timing)"
        print(f"  image {i+1}/{count}: {elapsed:.1f}s [{'PASS' if ok else 'FAIL'} <{HQ_TIME_BUDGET_SECONDS}s], "
              f"VRAM peak {sampler.peak_vram_mb:.0f} MB / after {used_after:.0f} MB\n    stages: {stage_str}", flush=True)

    after_vram, _ = client.log_gpu_memory(f"{name}_after")
    avg_time = sum(per_image_times) / len(per_image_times)
    print(f"\n--- {name} summary ---")
    print(f"Average total time: {avg_time:.1f}s/image")
    print(f"Under {HQ_TIME_BUDGET_SECONDS}s budget: {passed}/{count} images")
    print(f"VRAM {before_vram:.0f} -> {after_vram:.0f} MB")
    return {"name": name, "avg_time": avg_time, "passed": passed, "count": count,
            "before_vram": before_vram, "after_vram": after_vram, "leak_signal": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--hq", action="store_true", help="run the HQ two-pass benchmark (square + portrait) instead of the plain txt2img tests")
    parser.add_argument("--anchor", default=None, help="face anchor image for the HQ test's IP-Adapter FaceID (a fictional/AI face)")
    parser.add_argument("--character-lora", default=None, help="optional character LoRA filename (relative to ComfyUI/models/loras) to include in the HQ test")
    parser.add_argument("--controlnet", action="store_true", help="add the ControlNet OpenPose stage to the HQ test. Without --pose it reuses the --anchor photo as a stand-in skeleton (needs --anchor, runs the preprocessor); with --pose it feeds a library skeleton directly")
    parser.add_argument("--pose", default=None, help="use a training/poses/ library skeleton for the ControlNet stage (implies --controlnet, no --anchor needed for the pose)")
    parser.add_argument("--insightface-provider", choices=["CPU", "CUDA"], default=None,
                        help="override comfyui_client.INSIGHTFACE_PROVIDER for the HQ test to A/B the VRAM-thrash hypothesis")
    args = parser.parse_args()

    if args.insightface_provider:
        client.INSIGHTFACE_PROVIDER = args.insightface_provider

    if args.hq:
        canvases = [("hq_square", 1024, 1024), ("hq_portrait", 832, 1216)]
        if args.pose:
            # ControlNet center-crops the hint to the canvas aspect, so timing a
            # portrait skeleton on the square canvas would measure a broken pose.
            keep = []
            for name, w, h in canvases:
                loss = pose_skeletons.crop_fraction(args.pose, w, h)
                if loss > pose_skeletons.CANVAS_CROP_TOLERANCE:
                    print(f"skipping {name}: {w}x{h} would crop {loss * 100:.0f}% off "
                          f"the '{args.pose}' skeleton", flush=True)
                else:
                    keep.append((name, w, h))
            canvases = keep
        results = [
            run_hq_test(name, w, h, args.count, anchor=args.anchor,
                        character_lora=args.character_lora,
                        controlnet=args.controlnet or bool(args.pose), pose=args.pose)
            for name, w, h in canvases
        ]
        print("\n\n========== HQ FINAL REPORT ==========")
        for r in results:
            print(f"{r['name']}: avg {r['avg_time']:.1f}s/image, "
                  f"under-budget {r['passed']}/{r['count']}, "
                  f"VRAM {r['before_vram']:.0f}->{r['after_vram']:.0f} MB")
    else:
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
