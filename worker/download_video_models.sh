#!/usr/bin/env bash
# ONE-TIME: add the VIDEO model weights to the RunPod Network Volume, on top of what
# download_models.sh set up. Run it ON a pod that has the volume mounted at /runpod-volume
# (a cheap CPU pod is fine - it only downloads; nothing here needs a GPU).
#
# Sizes below were read from each URL's Content-Length on 2026-09-17:
#   Wan 2.2 TI2V-5B image-to-video   ~18.1 GB  (diffusion 10.0 + umt5 text encoder 6.7 + vae 1.4)
#   AnimateDiff cloud tier            ~2.0 GB  (motion module 1.8 + FaceID SD1.5 0.2)
# Leave ~25 GB free on the volume for these plus the image-path weights.
#
# Re-running is safe: existing files are skipped. Partial downloads go to <file>.part and are only
# renamed into place when curl finishes, so an interrupted run never leaves a truncated model
# that ComfyUI would try (and fail) to load.
set -euo pipefail
VOL="${VOL:-/runpod-volume}"
M="$VOL/models"

mkdir -p "$M"/{diffusion_models,text_encoders,vae,animatediff_models,animatediff_motion_lora,ipadapter,loras,checkpoints}

fetch() { # url dest
  if [ -f "$2" ]; then
    echo "skip (exists): $2"
    return
  fi
  echo "download: $2"
  curl -fL --retry 3 --retry-delay 5 "$1" -o "$2.part"
  mv "$2.part" "$2"
}

HF=https://huggingface.co

echo "=== Wan 2.2 TI2V-5B (image-to-video) ==="
fetch "$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors" \
      "$M/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
fetch "$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
      "$M/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
fetch "$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors" \
      "$M/vae/wan2.2_vae.safetensors"

echo "=== AnimateDiff cloud tier ==="
fetch "$HF/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt" \
      "$M/animatediff_models/mm_sd_v15_v2.ckpt"
fetch "$HF/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin" \
      "$M/ipadapter/ip-adapter-faceid-plusv2_sd15.bin"
fetch "$HF/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors" \
      "$M/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors"

echo
echo "=== Copy in manually (same file as the local install, not freely downloadable here) ==="
echo "  checkpoints/Realistic_Vision_V6.0_NV_B1_fp16.safetensors   <- AnimateDiff's default SD1.5 base"
echo "  anchors/<characterId>/anchor_*.png                         <- first frame / face reference fallback"
echo "Already covered by download_models.sh: 4x-UltraSharp.pth, YOLO face/hand detectors, buffalo_l."
echo "RIFE (rife47.pth) is fetched by the Frame-Interpolation node on each fresh worker's first use."
