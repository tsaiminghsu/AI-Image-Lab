"""Batch TTS with Fun-CosyVoice3 - runs INSIDE the CosyVoice clone's own venv, never ComfyUI's.

drama.py writes a job file and runs this as
    <CosyVoice>/.venv/Scripts/python.exe training/cosyvoice_runner.py job.json
with the CosyVoice clone as the working directory (its imports are relative to it). The model is
loaded once for the whole episode.

Job file: {"model_dir": "...", "items": [{"id", "text", "out", "prompt_wav", "prompt_text",
"instruct"}]}. With "instruct" set, inference_instruct2 carries the delivery (emotion, pace) and
takes timbre from prompt_wav; otherwise inference_zero_shot clones prompt_wav using its exact
transcript in prompt_text. One JSON line per item goes to stdout: {"id", "out", "seconds"}.

This file imports torch and cosyvoice, so nothing in the main repo may import it.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "third_party", "Matcha-TTS"))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def synth(model, item):
    if item.get("instruct"):
        chunks = model.inference_instruct2(item["text"], item["instruct"], item["prompt_wav"], stream=False)
    else:
        chunks = model.inference_zero_shot(item["text"], item["prompt_text"], item["prompt_wav"], stream=False)
    speech = torch.cat([c["tts_speech"] for c in chunks], dim=1)
    os.makedirs(os.path.dirname(os.path.abspath(item["out"])), exist_ok=True)
    torchaudio.save(item["out"], speech, model.sample_rate)
    return speech.shape[1] / model.sample_rate


def main(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    t0 = time.time()
    model = AutoModel(model_dir=job["model_dir"], fp16=False)
    print(f"# model loaded in {time.time() - t0:.1f}s, sample rate {model.sample_rate}", flush=True)
    for item in job["items"]:
        t = time.time()
        seconds = synth(model, item)
        print(json.dumps({"id": item["id"], "out": item["out"], "seconds": round(seconds, 3),
                          "wall": round(time.time() - t, 1)}, ensure_ascii=False), flush=True)
    if torch.cuda.is_available():
        print(f"# peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
