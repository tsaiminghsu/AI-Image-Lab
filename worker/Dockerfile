# RunPod Serverless worker: ComfyUI + this repo's custom nodes + our handler.
#
# Built by .github/workflows/worker-image.yml (no Docker on the Windows dev
# box). Models are NOT baked in - they live on a RunPod Network Volume mounted
# at /runpod-volume (see worker/extra_model_paths.yaml, worker/download_models.sh).
# Expected image size ~9-11GB (torch + CUDA dominate).
#
# Base image: the lock file pins torch==2.11.0+cu128, which the old cuda12.4.1 base could not
# have run (this image had never actually been built, so nothing caught it). cuda12.8.1 matches
# the wheels; RunPod endpoints must be limited to hosts whose driver supports CUDA 12.8.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Every ref is the commit the LOCAL install runs (read 2026-09-17 with
# `git -C ComfyUI[/custom_nodes/<pack>] rev-parse HEAD`), so the worker executes the same node
# code the workflow templates were built and tested against. Bump them together with the local
# install, never to a branch name.
ARG COMFYUI_REF=62b3c94bd45154f6486c7abf1b9efcacee96ea69
ARG IPADAPTER_REF=a0f451a5113cf9becb0847b92884cb10cbdec0ef
ARG CONTROLNET_AUX_REF=e8b689a513c3e6b63edc44066560ca5919c0576e
ARG IMPACT_PACK_REF=429d0159ad429e64d2b3916e6e7be9c22d025c3c
ARG IMPACT_SUBPACK_REF=50c7b71a6a224734cc9b21963c6d1926816a97f1
ARG ANIMATEDIFF_REF=9257651221002dcba0a12f9cff37e1944e58fb60
ARG FRAME_INTERP_REF=26545cc2dd95bc3d27f056016300673bdeee78f5

# MODEL_VARIANT=full: the fp8 SDXL/Pony variants (training/quantize_models.py) exist for the 8 GB
# local card; RunPod GPUs don't need them and the network volume has no *.fp8q.safetensors files.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFYUI_URL=http://127.0.0.1:8188 \
    MODEL_VARIANT=full \
    PYTHONPATH=/app/training

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- ComfyUI core ---
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI && \
    cd /app/ComfyUI && git checkout ${COMFYUI_REF} && \
    pip install -r requirements.txt

# --- custom nodes (the six this repo uses) ---
RUN cd /app/ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git && \
    cd ComfyUI_IPAdapter_plus && git checkout ${IPADAPTER_REF} && cd .. && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && git checkout ${CONTROLNET_AUX_REF} && \
    pip install -r requirements.txt && cd .. && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && git checkout ${IMPACT_PACK_REF} && \
    pip install -r requirements.txt && cd .. && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git && \
    cd ComfyUI-Impact-Subpack && git checkout ${IMPACT_SUBPACK_REF} && \
    pip install -r requirements.txt && cd .. && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git && \
    cd ComfyUI-AnimateDiff-Evolved && git checkout ${ANIMATEDIFF_REF} && cd .. && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && git checkout ${FRAME_INTERP_REF} && \
    pip install -r requirements-no-cupy.txt
# RIFE weights (rife47.pth, tens of MB) are fetched by the node into its own ckpts/ folder on first
# use - once per fresh worker, not per job. Not pre-fetched here because the download URL is the
# node pack's internal detail.

# --- pinned extra deps (insightface / onnxruntime-gpu / ultralytics etc.) ---
# These lock files are the source of truth so the container matches local, and they install LAST
# so the unpinned `torch` in the node packs' requirements is replaced by the pinned build. The
# +cu128 torch wheels are not on PyPI, hence the PyTorch index.
COPY comfyui-requirements.lock.txt comfyui-requirements-extra.lock.txt /app/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /app/comfyui-requirements.lock.txt && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /app/comfyui-requirements-extra.lock.txt

# --- worker deps ---
COPY worker/requirements.txt /app/worker/requirements.txt
RUN pip install -r /app/worker/requirements.txt

# --- pre-download the OpenPose annotator weights (controlnet_aux stores them
# inside its own dir, NOT under models/, so they can't live on the volume) ---
RUN python - <<'PY' || echo "annotator pre-download skipped (will lazy-download at runtime)"
import os
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
try:
    from huggingface_hub import hf_hub_download
    for f in ["body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"]:
        hf_hub_download("lllyasviel/Annotators", f,
                        local_dir="/app/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators")
except Exception as e:
    print("skip:", e)
    raise SystemExit(1)
PY

# --- our code + handler ---
COPY training/*.py training/*.json /app/training/
COPY worker/ /app/worker/

RUN chmod +x /app/worker/start.sh

CMD ["/app/worker/start.sh"]
