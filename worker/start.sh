#!/usr/bin/env bash
# Container entrypoint: launch ComfyUI in the background, wait for it to accept
# requests, then hand control to the RunPod serverless handler. The handler
# talks to ComfyUI over localhost:8188 - exactly like the local pipeline, so
# comfyui_client.py runs unchanged (COMFYUI_URL defaults to that address).
set -euo pipefail

export COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"
export COMFYUI_RAW_OUTPUT_DIR="${COMFYUI_RAW_OUTPUT_DIR:-/tmp/comfyui_out}"
export PYTHONPATH="/app/training:${PYTHONPATH:-}"
mkdir -p "$COMFYUI_RAW_OUTPUT_DIR"

echo "[start] launching ComfyUI"
python /app/ComfyUI/main.py \
  --listen 127.0.0.1 --port 8188 \
  --extra-model-paths-config /app/worker/extra_model_paths.yaml \
  --disable-auto-launch \
  > /tmp/comfyui.log 2>&1 &
COMFY_PID=$!

echo "[start] waiting for ComfyUI /system_stats"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:8188/system_stats" >/dev/null 2>&1; then
    echo "[start] ComfyUI up after ${i}s"
    break
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "[start] ComfyUI process died during startup; log:" >&2
    tail -n 50 /tmp/comfyui.log >&2
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:8188/system_stats" >/dev/null 2>&1; then
  echo "[start] ComfyUI did not become ready in time; log:" >&2
  tail -n 50 /tmp/comfyui.log >&2
  exit 1
fi

echo "[start] starting RunPod handler"
exec python -u /app/worker/handler.py
