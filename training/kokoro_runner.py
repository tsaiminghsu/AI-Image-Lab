"""Batch TTS with Kokoro-82M's built-in voices - runs INSIDE Kokoro\\.venv, never ComfyUI's.

drama.py writes a job file and runs this as
    Kokoro/.venv/Scripts/python.exe training/kokoro_runner.py job.json
with HF_HOME pointing into the Kokoro folder, where the model and voice packs download on first use.
Everything runs on the CPU: an 82M-parameter model needs no GPU, so ComfyUI can keep the card.

Job file: {"repo_id": "hexgrad/Kokoro-82M", "items": [{"id", "text", "out", "voice", "speed"}]}.
The voice name's first letter is Kokoro's language code (a = American English, b = British).
speed scales the delivery (1.0 normal; learners' phrase readings use ~0.85). Each item becomes a
24 kHz mono 16-bit WAV, written to a temporary name first so an interrupted run leaves no partial
file that drama.py would take as done. One JSON line per item goes to stdout: {"id", "out",
"seconds", "wall"}.

This file imports torch and kokoro, so nothing in the main repo may import it.
"""

import json
import os
import sys
import time
import warnings

# Library deprecation noise (torch weight_norm / jit, an LSTM dropout note) that says nothing about the audio.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="dropout option adds dropout")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from kokoro import KPipeline  # noqa: E402

SAMPLE_RATE = 24000


def synth(pipeline, item):
    chunks = [np.asarray(result.audio, dtype=np.float32)
              for result in pipeline(item["text"], voice=item["voice"], speed=float(item.get("speed") or 1.0))
              if result.audio is not None]
    if not chunks:
        raise RuntimeError(f"{item['id']}: Kokoro produced no audio for {item['text']!r}")
    audio = np.concatenate(chunks)
    out = os.path.abspath(item["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part.wav"
    sf.write(tmp, audio, SAMPLE_RATE, subtype="PCM_16")
    os.replace(tmp, out)
    return len(audio) / SAMPLE_RATE


def main(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    pipelines = {}
    for item in job["items"]:
        lang = item["voice"][0]
        if lang not in pipelines:
            t0 = time.time()
            pipelines[lang] = KPipeline(lang_code=lang, repo_id=job["repo_id"], device="cpu")
            print(f"# pipeline '{lang}' loaded in {time.time() - t0:.1f}s", flush=True)
        t = time.time()
        seconds = synth(pipelines[lang], item)
        print(json.dumps({"id": item["id"], "out": item["out"], "seconds": round(seconds, 3),
                          "wall": round(time.time() - t, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
