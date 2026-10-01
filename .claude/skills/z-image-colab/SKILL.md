---
name: z-image-colab
description: Generate images from text with Z-Image Turbo on a Google Colab GPU through a persistent ComfyUI worker - queue one or many jobs, one Colab session renders them all, then the PNGs are downloaded and validated. Use when the user asks for Z-Image / Colab / cloud-GPU image generation, for several images or variations at once, or says things like 「用 Z Image 生成」「幫我生 10 張產品圖」「用 Colab 生圖」「不要用本機顯卡生圖」. Not for local-GPU generation (training/generate_character.py, the GUI), character identity (FaceID / LoRA), img2img, inpainting, ControlNet or video.
---

# Z-Image Turbo txt2img on a persistent Colab worker

Jobs are JSON files in a local queue. One controller opens ONE Colab GPU session, a worker on that VM
keeps ComfyUI and the model loaded and renders every queued job, and the session is released after
`worker.idle_timeout_seconds` with an empty queue. The local machine never loads the model. Design,
storage layout and measured numbers: `README.md`.

Python: `ComfyUI\.venv\Scripts\python.exe`, run from the repo root. Paths below are relative to
`.claude/skills/z-image-colab/`.

## Rules that are not negotiable

1. **User intent > prompt optimization.** The prompt goes to the model as written; the scripts add
   nothing to it. When you turn a request into a prompt, improve grammar and add detail the user would
   agree with, but never change what they asked for: 「白色背景」 stays a white background, 「固定鏡頭」
   gets no camera movement, a product they named stays the subject. See `prompts/prompt_template.md`.
2. **Many images = many jobs in ONE session.** Ten images are `--count 10` or a manifest, then one
   `worker.py up`. Never start a session per image, and never start a second controller.
3. **`worker.py up` spends Colab compute units; nothing else does.** `submit.py`, `status.py`,
   `cancel.py`, `validate.py` and `setup_colab.py preflight` are free. Start the worker for the jobs the
   user asked for. Ask before re-running after a failed session; never loop on failures.
4. **One runtime on the account at a time, across Claude sessions.** The controller refuses with
   `COLAB_BUSY` when another runtime (an H3 job, usually) is active. Find its owner (ListAgents /
   SendMessage) and wait; do not pass `--ignore-busy` on your own.
5. **Sign-in and Drive consent are the user's.** On `AUTH_REQUIRED`, give them the login command from the
   `hint`. Google asks for Drive consent on every new Colab VM: the controller opens the page in their
   browser and they click through; you only run `worker.py consent-done` after they say they did.
6. **The safety path has no switch.** Every job carries the project's age-safety negatives and cfg >= 1.5
   (below that ComfyUI skips the negative prompt). `submit.py` refuses anything else and so does the
   worker; do not edit a job file or the graph to get around either. This skill is the `safe` tier only:
   any person in a prompt is an adult, and explicit content is out of scope.
7. **A job is done when its PNG passed validation on this machine**, not when ComfyUI said so. Report
   `INVALID_OUTPUT`, `FAILED` and `TIMEOUT` as failures, with the job's `error`.
8. **Always confirm the runtime is gone.** After a session ends, `status.py` must show no `zimg-*`
   session. If the summary has `"stop": "stop_failed"`, tell the user at once and show the manual stop
   command from `warning` - that VM is still billing.

## Workflow

1. **Understand the request**: subject, background, composition, aspect ratio, how many images, whether
   lettering must appear in the image, whether exactly one person is meant.
2. **Write the prompt** (English or Chinese - Z-Image reads both) following `prompts/prompt_template.md`,
   and show it to the user in your reply.
3. **Pick parameters**: `--aspect` (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`) or
   `--width/--height` (multiples of 16, 512-2048). Leave `--steps` (8) and `--cfg` (2.0) alone unless the
   user asks. `--seed -1` picks a seed and records it; a fixed seed with `--count N` gives seed..seed+N-1.
   `--solo` when exactly one person is wanted; `--allow-text` only when the user asked for lettering.
4. **Preflight** (free): `python scripts/setup_colab.py preflight`. Fix what it reports - Docker Desktop
   not running: ask the user to start it; image not built: build it from the H3 skill's `docker/`
   directory; not signed in: rule 5; another runtime active: rule 4.
5. **Queue** (free):

   ```
   ComfyUI\.venv\Scripts\python.exe .claude\skills\z-image-colab\scripts\submit.py ^
     --prompt "A matte black wireless earbud case centered on a black background, ..." ^
     --aspect 16:9 --count 4
   ```

   Different prompts in one go: `--manifest batch.json` =
   `{"batch": "name", "jobs": [{"prompt": "...", "aspect": "16:9", "count": 2}, ...]}`. Every entry is
   validated before any is stored. `--dry-run` validates and prints without storing.
6. **Start the worker** (spends CU): `python scripts/worker.py up`. It returns at once; the controller
   runs detached and logs to `output/zimage/logs/controller-*.log`. If a controller is already running it
   picks the new jobs up - submitting is enough.
   - Within about a minute the log says `DRIVE_CONSENT_NEEDED` and a Google page opens in the user's
     browser. Tell them to approve it; when they confirm, run `python scripts/worker.py consent-done`.
     Without consent inside `colab.drive_consent_wait_seconds` the session still works, but installs and
     downloads everything again (about 20 GB) and keeps nothing - say so when it happens.
   - First session ever: ComfyUI install + model download + copy to Drive. Later sessions: unpack two
     archives and copy the models from Drive. The log line `worker READY: setup {...}` shows which
     (`installed`/`downloaded` vs `extracted`/`staged`).
7. **Follow**: `python scripts/status.py` (add `--local` to skip the Colab call, `--job ID` for one job).
   While it runs, the controller log must keep growing; a `heartbeat_stale` ending means the VM was lost
   and was stopped.
8. **Report** per job: `status`, the path `output/zimage/<job_id>/result.png`, size, seed,
   `generation_seconds`, `gpu`, `vram_peak_mib`; per session (from `output/zimage/zimage_sessions.jsonl`):
   `reason`, `setup_actions`, `phase_seconds`, `cu_used_measured` (already re-read
   `colab.cu_settle_seconds` after the stop - Colab deducts late). Look at the images (Read tool) before
   calling them good: validation checks the file, not the picture.
9. **More images while the worker is idle**: just `submit.py` again - the same session takes them until
   the idle timeout. `python scripts/worker.py stop` ends the session early.

## Job states

`PENDING → QUEUED → RUNNING → GENERATED → VALIDATING → VALID → COMPLETED`; terminal failures `FAILED`,
`TIMEOUT`, `CANCELLED`, `INVALID_OUTPUT`. A job whose session was lost goes back to `PENDING` once and
fails with `SESSION_LOST` the second time.

## Error codes

| Where | Code | What to tell the user / do |
| --- | --- | --- |
| submit | `INVALID_INPUT` | Nothing was queued. The message says which parameter (size, steps, cfg, seed, empty prompt). |
| submit, any | `INVALID_CONFIG` | `config/config.json` or a `ZIMG_*` variable is out of range; the message names the key. |
| preflight / up | `DOCKER_UNAVAILABLE` | Start Docker Desktop, or build the `h3-colab-cli` image (`hint`). |
| preflight / up | `AUTH_REQUIRED` | The user runs the `hint` login command once (rule 5). |
| up | `COLAB_BUSY` | Another runtime is active on the account. Wait for its owner (rule 4). |
| up | `GPU_UNAVAILABLE` | No such GPU right now; nothing was spent. Retry later, or `ZIMG_GPU=A100` if they agree. |
| up | `CONTROLLER_RUNNING` | A controller already runs; submit to it instead. |
| up | `COLAB_CONNECTION_FAILED` | Network or Colab problem. Check `status.py` for a leftover session. |
| session | `worker_boot_failed:GPU_NOT_SUPPORTED` | The VM's GPU is below `gpu.min_vram_gib` or has no bf16 (a T4). No job ran. |
| session | `worker_boot_failed:DISK_FULL` / `MODEL_CHECKSUM` / `DEPS_INSTALL_FAILED` / `COMFYUI_START_FAILED` / `NODE_CHECK_FAILED` | Setup failed before any job; report the message and the worker log. |
| session | `heartbeat_stale`, `state_unreadable`, `bootstrap_timeout` | The VM stopped answering; it was stopped and its jobs went back to `PENDING`. Ask before starting again. |
| job | `COMFY_REJECTED`, `COMFY_EXECUTION_ERROR`, `OUTPUT_NOT_FOUND`, `JOB_FAILED` | That job failed; the others carried on. Report `error.error_message`. |
| job | `TIMEOUT` | No image within `worker.job_timeout_seconds`; it was interrupted in ComfyUI. Do not resubmit without asking. |
| job | `INVALID_OUTPUT` | The PNG failed validation (listed in `validation`). Never call it a success. |
| job | `UNSAFE_JOB`, `INVALID_JOB` | The graph was changed after submit. Rule 6. |
