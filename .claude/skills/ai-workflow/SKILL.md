---
name: ai-workflow
description: Queue image or video generation jobs on the Drive + Colab + ComfyUI workflow platform (ai_workflow/) through its six-tool interface - list workflows, create a job, check status, list, cancel, get the result. Use when the user asks to generate through "the workflow platform" / 「工作流平台」/ AI-Workflow, to queue a job for the Colab notebook worker, or to check or fetch a platform job. Not for local-GPU generation (training/), and not for the older single-purpose skills z-image-colab and minimax-h3-colab, which start their own Colab session.
---

# AI Workflow platform (tool interface)

This skill is a pointer, on purpose: the platform is assistant-neutral, and everything an assistant needs is in
`ai_workflow/AGENTS.md`. **Read that file and follow it.** Nothing below overrides it.

What is specific to this repository:

- Run the tools from the repo root with `ComfyUI\.venv\Scripts\python.exe ai_workflow\scripts\aiwf.py ...`
  (any Python 3.11+ works; it is standard library only). The workspace location comes from
  `ai_workflow/configs/local.json`; on `WORKSPACE_NOT_FOUND`, ask the user where Google Drive for desktop put
  `AI-Workflow` instead of guessing.
- You cannot start the Colab runtime. When a tool's answer carries a `hint`, tell the user to open the named
  notebook in Colab and press Run all. That spends their compute units, so it is their decision
  (memory: ask before GPU work).
- After changing anything under `ai_workflow/` (core, registry, workflows): run
  `ai_workflow\scripts\build_notebooks.py`, then `check.ps1`, then `ai_workflow\scripts\deploy.py` to copy it
  to the workspace. Never edit the `.ipynb` files or the deployed copy by hand.
- The safety rules of CLAUDE.md (年齡安全) apply unchanged: no bypass of `PROMPT_REJECTED` / `UNSAFE_JOB`, and
  `--attest first_frame` only after the user confirmed the image's origin.
