"""Several Pony x Z-Image pictures in one go, each with its own random seed, to compare prompts or seeds side by side.

The card runs one picture at a time (comfyui_client holds a lock for the whole submit-and-wait, and Pony and Z-Image have
to be loaded in turn), so a "worker" here is a queued job: its own seed, run in order, shown as soon as it is done. The
GPU rules are the ones the other local batches use (studio_api, concept_api): one gpu_guard.before_batch, then before
every picture an Ollama unload and a wait for the card to cool.

This module has no gradio and no ComfyUI import: the GUI hands in `prepare` (Chinese -> English, once per distinct
prompt, before the first picture so Qwen and ComfyUI never fight over the card) and `generate` (one gen_pony_zimage call).
"""

import json
import os
import random
import time
from dataclasses import dataclass

import generate_character as gc

MAX_PROMPTS = 10
MAX_PER_PROMPT = 8
MAX_TOTAL = 24
SEED_MAX = 2**31 - 1  # the range image_api and the Colab skill use


@dataclass(frozen=True)
class Job:
    index: int  # 1-based position in the batch
    prompt_no: int  # 1-based line of the prompt box
    prompt: str
    seed: int


def parse_prompts(text):
    """One prompt per line; blank lines are dropped. Raises UsageError when there is none or too many."""
    prompts = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not prompts:
        raise gc.UsageError("請輸入 prompt（批次一行一個）")
    if len(prompts) > MAX_PROMPTS:
        raise gc.UsageError(f"批次最多 {MAX_PROMPTS} 行 prompt，現在有 {len(prompts)} 行")
    return prompts


def plan_jobs(prompts, per_prompt, rng=None):
    """prompts x per_prompt jobs, every one with its own random seed (no seed repeats inside a batch)."""
    per_prompt = int(per_prompt)
    if not 1 <= per_prompt <= MAX_PER_PROMPT:
        raise gc.UsageError(f"每個 prompt 生 1～{MAX_PER_PROMPT} 張，現在是 {per_prompt}")
    total = len(prompts) * per_prompt
    if total > MAX_TOTAL:
        raise gc.UsageError(f"一批最多 {MAX_TOTAL} 張，現在是 {len(prompts)} 行 × {per_prompt} 張 = {total} 張")
    rng = rng or random.SystemRandom()
    seeds = set()
    jobs = []
    for prompt_no, prompt in enumerate(prompts, 1):
        for _ in range(per_prompt):
            seed = rng.randint(0, SEED_MAX)
            while seed in seeds:
                seed = rng.randint(0, SEED_MAX)
            seeds.add(seed)
            jobs.append(Job(len(jobs) + 1, prompt_no, prompt, seed))
    return jobs


def new_batch_dir(root, now=None):
    """A fresh folder per batch, so nothing from another batch (or a single picture) is overwritten."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    path = os.path.join(root, f"ponyz_batch_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def stem_for(job):
    return f"p{job.prompt_no}_s{job.seed}"


def _write_manifest(batch_dir, meta, records, stopped):
    data = {**meta, "stopped": stopped, "jobs": records}
    tmp = os.path.join(batch_dir, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(batch_dir, "manifest.json"))


def run_batch(
    jobs, *, prepare, generate, guard, start_temp_c, should_stop, batch_dir, meta=None, log=print, clock=time.time
):
    """Run the jobs in order and yield one record per job as it finishes (status ok / error / stopped).

    prepare(prompt) -> dict with the English prompts (any keys; it is stored and handed to generate). It is called once
    per distinct prompt, for all of them before the first picture. A prepare that raises marks only that prompt's jobs
    as failed. generate(job, prepared, stem) -> (base_path, final_path_or_None). One failing picture is recorded and
    the batch goes on; should_stop() is checked before each picture, so the one running always finishes.
    manifest.json in batch_dir is rewritten after every job."""
    meta = dict(meta or {})
    records = []
    prepared, prepare_errors = {}, {}
    for job in jobs:
        if job.prompt in prepared or job.prompt in prepare_errors:
            continue
        try:
            prepared[job.prompt] = prepare(job.prompt)
        except Exception as exc:  # noqa: BLE001 - a bad prompt must not take the other prompts down
            prepare_errors[job.prompt] = str(exc) or type(exc).__name__
            log(f"[batch] prompt #{job.prompt_no} 準備失敗：{prepare_errors[job.prompt]}")
    if len(prepare_errors) < len({j.prompt for j in jobs}):
        guard.before_batch(max_start_temp_c=start_temp_c)
    stopped = False
    for job in jobs:
        record = {
            "index": job.index,
            "prompt_no": job.prompt_no,
            "prompt": job.prompt,
            "seed": job.seed,
            "english": prepared.get(job.prompt),
            "base": None,
            "final": None,
            "error": None,
            "seconds": None,
            "waited_seconds": 0,
            "status": "ok",
        }
        if job.prompt in prepare_errors:
            record.update(status="error", error=prepare_errors[job.prompt])
        elif stopped or should_stop():
            stopped = True
            record["status"] = "stopped"
        else:
            started = clock()
            try:
                guard.before_image()
                waited = guard.wait_until_cool()
                record["waited_seconds"] = (waited or {}).get("waited_seconds", 0)
                started = clock()
                record["base"], record["final"] = generate(job, prepared[job.prompt], stem_for(job))
            except Exception as exc:  # noqa: BLE001 - one failed picture is recorded, the rest still run
                record.update(status="error", error=str(exc) or type(exc).__name__)
                log(f"[batch] #{job.index} (seed {job.seed}) 失敗：{record['error']}")
            record["seconds"] = round(clock() - started)
        records.append(record)
        _write_manifest(batch_dir, meta, records, stopped)
        yield record
