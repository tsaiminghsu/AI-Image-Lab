#!/usr/bin/env bash
# ONE-TIME: populate the RunPod Network Volume with the model weights the worker
# needs. Run this ON a pod that has the volume mounted at /runpod-volume (a cheap
# CPU pod is fine - this only downloads). The serverless endpoint then mounts the
# same volume read-mostly, so cold starts don't re-download anything.
#
# You must supply the checkpoints / IP-Adapter / CLIP-vision / ControlNet files
# yourself (same files the local README lists). This script creates the layout
# and fetches the freely-downloadable helpers (YOLO detectors, upscaler,
# insightface buffalo_l). Fill in the CIVITAI/HF sources for the gated ones.
set -euo pipefail
VOL="${VOL:-/runpod-volume}"
M="$VOL/models"

mkdir -p "$M"/{checkpoints,loras/characters,ipadapter,clip_vision,controlnet,upscale_models,ultralytics/bbox,ultralytics/segm,insightface/models}
mkdir -p "$VOL/anchors"

echo "=== created layout under $M ==="
echo "You must copy these in manually (same files as local README):"
echo "  checkpoints/  <- CyberRealisticPony_V18.0_F16.safetensors (and/or juggernaut/pony)"
echo "  loras/        <- sdxl_photorealistic_slider_v1-0.safetensors, ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
echo "  loras/characters/ <- your trained <loraId>.safetensors from runpod_train.sh"
echo "  ipadapter/    <- ip-adapter-faceid-plusv2_sdxl.bin"
echo "  clip_vision/  <- CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
echo "  controlnet/   <- control-lora-openposeXL2-rank256.safetensors"
echo "  anchors/<characterId>/anchor_*.png  <- one anchor per character"
echo

fetch() { # url dest
  if [ -f "$2" ]; then echo "skip (exists): $2"; else echo "download: $2"; curl -fL "$1" -o "$2"; fi
}

# YOLO nano detectors (ADetailer / FaceDetailer)
fetch "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" "$M/ultralytics/bbox/face_yolov8n.pt"
fetch "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt" "$M/ultralytics/bbox/hand_yolov8n.pt"

# 4x ESRGAN upscaler for the HQ hires pass
fetch "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" "$M/upscale_models/4x-UltraSharp.pth"

# InsightFace buffalo_l (FaceID). Pre-seed so cold starts don't lazy-download.
if [ ! -d "$M/insightface/models/buffalo_l" ]; then
  echo "download: buffalo_l"
  curl -fL "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -o /tmp/buffalo_l.zip
  mkdir -p "$M/insightface/models/buffalo_l"
  (cd "$M/insightface/models/buffalo_l" && unzip -o /tmp/buffalo_l.zip)
else
  echo "skip (exists): buffalo_l"
fi

echo "=== helper downloads done. Copy in the gated checkpoints/adapters listed above. ==="
