"""One-off exploratory script: random test generations across every installed
checkpoint, recording generation time and prompt-syntax behavior per model
family (SDXL natural language vs Pony score-tag convention vs SD1.5). Not part
of the regular pipeline - this is a benchmarking/notes tool, run manually.

Plain txt2img only (no FaceID/HQ) so results isolate checkpoint + prompt
behavior without the FaceID VRAM-thrash confound - see comfyui_client.py's
IP_ADAPTER_PRESET comment for why FaceID alone is much slower.

Usage:
    python model_prompt_test.py
Writes images to reference_candidates/model_test/ and a report to
reference_candidates/model_test/REPORT.md.
"""

import os
import random
import time

import comfyui_client as client

OUT_DIR = os.path.join(os.path.dirname(__file__), "reference_candidates", "model_test")
os.makedirs(OUT_DIR, exist_ok=True)

SAFETY_NEGATIVE = (
    "nsfw, nude, naked, explicit, sexual content, child, children, kid, minor, "
    "teen, teenager, underage, young girl, lowres, blurry, deformed, extra limbs, "
    "bad anatomy, watermark, text"
)
REALISTIC_NEGATIVE = (
    "3d render, cgi, illustration, airbrushed, plastic skin, doll-like, smooth skin, "
    "perfect skin, symmetrical face, digital art, render, unreal engine"
)
PONY_NEG_TAGS = "score_6, score_5, score_4"
PONY_POS_TAGS = "score_9, score_8_up, score_7_up"

# One natural-language descriptive prompt reused across every checkpoint, so
# the ONLY variable is the checkpoint (and, for Pony, whether the score tags
# are prepended) - isolates what each model actually needs syntactically.
BASE_PROMPT = (
    "a 25 year old woman with long brown wavy hair, sitting at a wooden cafe table, "
    "holding a ceramic coffee cup, soft window light, cozy interior background, "
    "shot on DSLR, natural skin texture, candid photograph"
)
NEGATIVE_PLAIN = f"{SAFETY_NEGATIVE}, {REALISTIC_NEGATIVE}"

# Checkpoint registry: (key, architecture family, native resolution)
CHECKPOINTS = [
    ("juggernaut", "sdxl", (1024, 1024)),
    ("pony", "pony", (1024, 1024)),
    ("cyberrealistic_pony", "pony", (1024, 1024)),
    ("pony_realism", "pony", (1024, 1024)),
    ("realistic_vision", "sd15", (client.SD15_WIDTH, client.SD15_HEIGHT)),
    ("cyberrealistic", "sd15", (client.SD15_WIDTH, client.SD15_HEIGHT)),
]

random.seed(20260908)


def timed_txt2img(checkpoint_key, prompt, negative, width, height, seed, label):
    ckpt_file = client.CHECKPOINTS[checkpoint_key]
    stem = f"{checkpoint_key}_{label}_seed{seed}"
    t0 = time.time()
    try:
        if checkpoint_key in client.SD15_CHECKPOINTS:
            raw = client.submit_txt2img_generation_sd15(
                prompt=prompt, negative_prompt=negative, seed=seed,
                filename_prefix=stem, checkpoint=ckpt_file, width=width, height=height,
            )
        else:
            raw = client.submit_txt2img_generation(
                prompt=prompt, negative_prompt=negative, seed=seed,
                filename_prefix=stem, width=width, height=height, checkpoint=ckpt_file,
                lora_strength=0.0,  # style slider only tuned for juggernaut; off elsewhere
            )
        elapsed = time.time() - t0
        out_path = os.path.join(OUT_DIR, f"{stem}.png")
        os.replace(raw, out_path)
        return {"ok": True, "elapsed": elapsed, "path": out_path, "error": None}
    except Exception as exc:
        elapsed = time.time() - t0
        return {"ok": False, "elapsed": elapsed, "path": None, "error": str(exc)}


def main():
    results = []
    seed_base = 77000

    for i, (ckpt_key, family, (w, h)) in enumerate(CHECKPOINTS):
        seed = seed_base + i
        print(f"\n=== {ckpt_key} ({family}, {w}x{h}) ===", flush=True)
        client.log_gpu_memory(f"before_{ckpt_key}")

        if family == "pony":
            # Test WITHOUT score tags first (demonstrates the Pony-specific
            # need), then WITH tags (the actually-recommended usage).
            prompt_no_tags = BASE_PROMPT
            neg_no_tags = NEGATIVE_PLAIN
            r1 = timed_txt2img(ckpt_key, prompt_no_tags, neg_no_tags, w, h, seed, "no_score_tags")
            print(f"  no-score-tags: {'OK' if r1['ok'] else 'FAIL'} {r1['elapsed']:.1f}s {r1['error'] or ''}", flush=True)
            results.append({
                "checkpoint": ckpt_key, "family": family, "label": "no_score_tags",
                "prompt_style": "plain natural language (no score_9 tags)",
                **r1,
            })

            prompt_tags = f"{PONY_POS_TAGS}, {BASE_PROMPT}"
            neg_tags = f"{PONY_NEG_TAGS}, {NEGATIVE_PLAIN}"
            r2 = timed_txt2img(ckpt_key, prompt_tags, neg_tags, w, h, seed, "with_score_tags")
            print(f"  with-score-tags: {'OK' if r2['ok'] else 'FAIL'} {r2['elapsed']:.1f}s {r2['error'] or ''}", flush=True)
            results.append({
                "checkpoint": ckpt_key, "family": family, "label": "with_score_tags",
                "prompt_style": f"'{PONY_POS_TAGS}' prefix + natural language (Pony convention)",
                **r2,
            })
        else:
            r = timed_txt2img(ckpt_key, BASE_PROMPT, NEGATIVE_PLAIN, w, h, seed, "plain")
            print(f"  plain: {'OK' if r['ok'] else 'FAIL'} {r['elapsed']:.1f}s {r['error'] or ''}", flush=True)
            results.append({
                "checkpoint": ckpt_key, "family": family, "label": "plain",
                "prompt_style": "plain natural language",
                **r,
            })

        client.log_gpu_memory(f"after_{ckpt_key}")

    # --- write report ---
    report_path = os.path.join(OUT_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Local model / prompt-syntax test report\n\n")
        f.write(f"Base prompt (identical across all checkpoints):\n\n> {BASE_PROMPT}\n\n")
        f.write("| Checkpoint | Family | Prompt style | Resolution | Time | Result |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            res = "OK" if r["ok"] else f"FAIL: {r['error']}"
            ckpt_key = r["checkpoint"]
            w, h = dict((c[0], c[2]) for c in CHECKPOINTS)[ckpt_key]
            f.write(f"| {ckpt_key} | {r['family']} | {r['prompt_style']} | {w}x{h} | {r['elapsed']:.1f}s | {res} |\n")
        f.write("\n## Notes\n\n")
        f.write("- SDXL (`juggernaut`) and SD1.5 (`realistic_vision`, `cyberrealistic`) checkpoints "
                "take plain natural-language prompts directly.\n")
        f.write("- Pony-family checkpoints (`pony`, `cyberrealistic_pony`, `pony_realism`) were trained "
                "on a `score_9, score_8_up, score_7_up` quality-tag convention; compare the "
                "`no_score_tags` vs `with_score_tags` rows/images for the same seed to see the effect.\n")
        f.write("- `generate_character.py`'s `gen_custom()` auto-prepends the score tags for Pony "
                "checkpoints, so normal pipeline callers never need to type them manually.\n")

    print(f"\n\nReport written to {report_path}")
    ok_count = sum(1 for r in results if r["ok"])
    print(f"{ok_count}/{len(results)} generations succeeded")


if __name__ == "__main__":
    main()
