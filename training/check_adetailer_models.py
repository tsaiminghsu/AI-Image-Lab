"""One-time check-and-download for the three ADetailer YOLO models this
project uses (face/hand bbox detection + generic segmentation). Idempotent -
safe to run every time before a FaceDetailer generation; already-present
files are left alone, only missing ones are downloaded.

Run standalone to verify/repair the install:
    python check_adetailer_models.py
"""

import os
import urllib.request

COMFYUI_MODELS = os.path.join(os.path.dirname(__file__), "..", "ComfyUI", "models")

# (relative path under ComfyUI/models/ultralytics/, source URL)
MODELS = [
    ("bbox/face_yolov8n.pt", "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"),
    ("bbox/hand_yolov8n.pt", "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"),
    ("segm/yolov8n-seg.pt", "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt"),
]


def check_and_download():
    ultralytics_dir = os.path.join(COMFYUI_MODELS, "ultralytics")
    results = []
    for rel_path, url in MODELS:
        full_path = os.path.join(ultralytics_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"OK (already present): {rel_path} ({size_mb:.1f} MB)", flush=True)
            results.append((rel_path, True))
            continue

        print(f"downloading {rel_path} from {url} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, full_path)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"OK (downloaded): {rel_path} ({size_mb:.1f} MB)", flush=True)
            results.append((rel_path, True))
        except Exception as e:
            print(f"FAILED: {rel_path} - {e}", flush=True)
            if os.path.exists(full_path):
                os.remove(full_path)  # don't leave a partial/corrupt file behind
            results.append((rel_path, False))

    return results


if __name__ == "__main__":
    results = check_and_download()
    ok = all(success for _, success in results)
    print("\nall models present and usable" if ok else "\nsome models are missing - see FAILED lines above", flush=True)
    raise SystemExit(0 if ok else 1)
