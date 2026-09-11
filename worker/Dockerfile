# RunPod Serverless worker: ComfyUI + this repo's custom nodes + our handler.
#
# Built by .github/workflows/worker-image.yml (no Docker on the Windows dev
# box). Models are NOT baked in - they live on a RunPod Network Volume mounted
# at /runpod-volume (see worker/extra_model_paths.yaml, worker/download_models.sh).
# Expected image size ~9-11GB (torch + CUDA dominate).
#
# NOTE: the *_REF build args below are MUTABLE branch names. For reproducible
# builds, pin each to a known-good commit SHA (verify against your working local
# install) and pass them via --build-arg / the workflow.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG COMFYUI_REF=master
ARG IPADAPTER_REF=main
ARG CONTROLNET_AUX_REF=main
ARG IMPACT_PACK_REF=Main
ARG IMPACT_SUBPACK_REF=main

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

# --- custom nodes (the four this repo uses) ---
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
    pip install -r requirements.txt

# --- pinned extra deps (insightface / onnxruntime-gpu / ultralytics etc.) ---
# These lock files are the source of truth so the container matches local.
COPY comfyui-requirements.lock.txt comfyui-requirements-extra.lock.txt /app/
RUN pip install -r /app/comfyui-requirements.lock.txt && \
    pip install -r /app/comfyui-requirements-extra.lock.txt

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
