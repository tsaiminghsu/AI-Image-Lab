"""Identity check for generated videos.

Scores every frame of a video against a reference face with InsightFace buffalo_l, the same
recognition model IPAdapter FaceID conditions on, and can save a contact sheet of the face in
each frame so warping between frames is easy to see at a glance.

The score is the cosine similarity of the two face embeddings. Compare runs made with the same
seed and prompt against each other; as a rough guide, the same person usually scores 0.5 or
more and 0.3-0.4 is borderline.

Needs insightface, onnxruntime, cv2 and av, which ComfyUI's venv already has, so run it with
that interpreter:

    D:\\AI-Image-Lab\\ComfyUI\\.venv\\Scripts\\python.exe face_similarity.py --anchor face.png --video clip.mp4 --sheet clip_faces.png

generate_character.py video-animatediff --face-report runs this automatically after generating.
"""

import argparse
import json
import os
import sys

# numpy/cv2/av/insightface are imported inside the functions that need them, so
# generate_character.py (system python) can import this module just for
# run_report_subprocess/format_summary without those packages installed.

COMFYUI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ComfyUI"))
INSIGHTFACE_ROOT = os.path.join(COMFYUI_DIR, "models", "insightface")
COMFYUI_PYTHON = os.path.join(COMFYUI_DIR, ".venv", "Scripts", "python.exe")

_app = None


def _get_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        # CPU only: runs while ComfyUI may still hold the GPU, and 16 frames take seconds anyway.
        app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_ROOT, providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _app = app
    return _app


def _read_image_bgr(path):
    import cv2
    import numpy as np

    # imdecode instead of imread: cv2.imread can't open non-ASCII paths on Windows.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"can't read image {path}")
    return img


def _best_face(img_bgr):
    faces = _get_app().get(img_bgr)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score)


def _video_frames_bgr(path):
    import av

    with av.open(path) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


def score_video(anchor_path, video_path, sheet_path=None, sheet_frames=16):
    """Returns {"frames": [{"index", "score" or None, "bbox"}], "mean", "min", "min_frame",
    "missing", "count"}. With sheet_path, also saves a contact sheet of up to sheet_frames evenly
    spaced frames, cropped to the detected face and labeled with each frame's score."""
    import numpy as np

    ref = _best_face(_read_image_bgr(anchor_path))
    if ref is None:
        raise ValueError(f"no face found in the reference image {anchor_path}")
    ref_emb = ref.normed_embedding

    results, images = [], []
    for index, img in enumerate(_video_frames_bgr(video_path)):
        face = _best_face(img)
        if face is None:
            results.append({"index": index, "score": None, "bbox": None})
        else:
            results.append({"index": index, "score": float(np.dot(ref_emb, face.normed_embedding)),
                            "bbox": [float(v) for v in face.bbox]})
        if sheet_path:
            images.append(img)

    scored = [r for r in results if r["score"] is not None]
    summary = {
        "frames": results,
        "count": len(results),
        "missing": [r["index"] for r in results if r["score"] is None],
        "mean": float(np.mean([r["score"] for r in scored])) if scored else None,
        "min": min((r["score"] for r in scored), default=None),
        "min_frame": min(scored, key=lambda r: r["score"])["index"] if scored else None,
    }
    if sheet_path and images:
        _save_sheet(images, results, sheet_path, sheet_frames)
    return summary


def _face_crop_box(bbox, width, height, margin=1.8):
    """Square crop around the face box, margin x its larger side, clamped to the frame."""
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1) * margin
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = min(side, width, height)
    left = min(max(cx - side / 2.0, 0), width - side)
    top = min(max(cy - side / 2.0, 0), height - side)
    return int(left), int(top), int(left + side), int(top + side)


def _save_sheet(images, results, sheet_path, sheet_frames, tile=256, cols=4):
    from PIL import Image, ImageDraw

    count = len(images)
    picks = sorted({round(i * (count - 1) / max(1, sheet_frames - 1)) for i in range(min(sheet_frames, count))})
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), "black")
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(picks):
        frame = Image.fromarray(images[index][:, :, ::-1])  # BGR -> RGB
        result = results[index]
        if result["bbox"]:
            crop = frame.crop(_face_crop_box(result["bbox"], frame.width, frame.height))
        else:
            crop = frame  # no face detected: show the whole frame so the failure is visible
        x, y = (slot % cols) * tile, (slot // cols) * tile
        sheet.paste(crop.resize((tile, tile)), (x, y))
        label = f"#{index} " + (f"{result['score']:.2f}" if result["score"] is not None else "no face")
        draw.rectangle([x, y, x + 92, y + 16], fill="black")
        draw.text((x + 4, y + 2), label, fill="white")
    os.makedirs(os.path.dirname(os.path.abspath(sheet_path)), exist_ok=True)
    sheet.save(sheet_path)


def format_summary(summary):
    if summary["mean"] is None:
        return f"no face detected in any of {summary['count']} frames"
    text = (f"identity vs reference: mean {summary['mean']:.3f}, min {summary['min']:.3f} "
            f"(frame {summary['min_frame']}), {summary['count']} frames")
    if summary["missing"]:
        text += f", no face in frames {summary['missing']}"
    return text


def run_report_subprocess(anchor_path, video_path, sheet_path):
    """For callers running outside ComfyUI's venv (generate_character.py runs under the system
    python, which may not have av/insightface): run this file with ComfyUI's interpreter and return
    the parsed summary."""
    import subprocess

    proc = subprocess.run(
        [COMFYUI_PYTHON, os.path.abspath(__file__), "--anchor", anchor_path, "--video", video_path,
         "--sheet", sheet_path, "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"face report failed (exit {proc.returncode}):\n{tail}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anchor", required=True, help="reference face image (the FaceID face ref)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--sheet", default=None, help="optional contact sheet output path (.png)")
    parser.add_argument("--json", action="store_true", help="print the summary as one JSON line (for scripts)")
    args = parser.parse_args()

    summary = score_video(args.anchor, args.video, sheet_path=args.sheet)
    if args.json:
        print(json.dumps(summary))
        return
    for r in summary["frames"]:
        print(f"frame {r['index']:3d}: " + (f"{r['score']:.3f}" if r["score"] is not None else "no face"))
    print(format_summary(summary))
    if args.sheet:
        print(f"saved {args.sheet}")


if __name__ == "__main__":
    sys.exit(main())
