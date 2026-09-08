#!/usr/bin/env bash
# Per-character LoRA training on a RunPod Pod (NOT the local RTX 2070).
#
# Why cloud: the RTX 2070 (Turing) has no bf16 tensor cores, and kohya's SDXL
# LoRA training in fp16 produces NaN loss from step 1 (SDXL fp16 VAE overflow).
# A 4090/A100 has bf16, so we train with --mixed_precision bf16 --no_half_vae
# and the NaN goes away. Run this ON the pod after receiving the bundle.
#
# Usage (on the pod):
#   bash runpod_train.sh <trigger> <base>
#     <trigger>  character key, e.g. xinyi   (dataset + configs named after it)
#     <base>     pony | sdxl   (which base checkpoint to train the LoRA on)
#
# Expects this layout on the Network Volume (mounted at /workspace):
#   /workspace/models/<base checkpoint>.safetensors
#   /workspace/datasets/<trigger>/*.png + *.txt         (NO *.npz - see below)
#   /workspace/config/{lora_<trigger>.toml, dataset_<trigger>.toml, <trigger>_samples.txt}
#   /workspace/runpod_train.sh (this file)
#
# runpod_bundle.py (Windows side) produces the dataset+config bundle and the
# `runpodctl send` command; `runpodctl receive` it here into /workspace.
set -euo pipefail

TRIGGER="${1:?usage: runpod_train.sh <trigger> <base:pony|sdxl>}"
BASE="${2:?usage: runpod_train.sh <trigger> <base:pony|sdxl>}"

WORKSPACE="${WORKSPACE:-/workspace}"
SD_SCRIPTS="$WORKSPACE/sd-scripts"
DATA_DIR="$WORKSPACE/datasets/$TRIGGER"
CFG_DIR="$WORKSPACE/config"
OUT_DIR="$WORKSPACE/output/$TRIGGER"

case "$BASE" in
  pony) BASE_FILE="ponyDiffusionV6XL_v6StartWithThisOne.safetensors" ;;
  sdxl) BASE_FILE="sd_xl_base_1.0.safetensors" ;;
  *) echo "base must be 'pony' or 'sdxl', got '$BASE'" >&2; exit 1 ;;
esac
BASE_PATH="$WORKSPACE/models/$BASE_FILE"

echo "=== LoRA training: trigger=$TRIGGER base=$BASE ($BASE_FILE) ==="

# --- one-time setup: sd-scripts + deps -------------------------------------
if [ ! -d "$SD_SCRIPTS" ]; then
  echo "[setup] cloning kohya-ss/sd-scripts"
  git clone https://github.com/kohya-ss/sd-scripts "$SD_SCRIPTS"
  pushd "$SD_SCRIPTS" >/dev/null
  # sd3 branch has the current SDXL trainer; main also works. Pin if you need
  # reproducibility (verify a known-good commit on the RunPod console).
  pip install -r requirements.txt
  pip install bitsandbytes  # AdamW8bit optimizer
  popd >/dev/null
else
  echo "[setup] sd-scripts already present, skipping clone"
fi

# --- preflight assertions ---------------------------------------------------
[ -f "$BASE_PATH" ]  || { echo "missing base checkpoint: $BASE_PATH" >&2; exit 1; }
[ -d "$DATA_DIR" ]   || { echo "missing dataset dir: $DATA_DIR" >&2; exit 1; }
[ -f "$CFG_DIR/lora_$TRIGGER.toml" ]    || { echo "missing $CFG_DIR/lora_$TRIGGER.toml" >&2; exit 1; }
[ -f "$CFG_DIR/dataset_$TRIGGER.toml" ] || { echo "missing $CFG_DIR/dataset_$TRIGGER.toml" >&2; exit 1; }

# --- delete poisoned local caches ------------------------------------------
# The *.npz latent/text-encoder caches in datasets/character were produced by
# the local fp16 run that NaN'd; kohya reuses them by shape without validating
# content, and the text-encoder cache is invalid for Pony's re-trained CLIP.
# Remove so they get recomputed cleanly on this GPU.
NPZ_COUNT=$(find "$DATA_DIR" -name '*.npz' | wc -l)
if [ "$NPZ_COUNT" -gt 0 ]; then
  echo "[clean] removing $NPZ_COUNT stale .npz cache files from $DATA_DIR"
  find "$DATA_DIR" -name '*.npz' -delete
fi

IMG_COUNT=$(find "$DATA_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "[info] $IMG_COUNT training images in $DATA_DIR"

mkdir -p "$OUT_DIR/logs"

# --- train ------------------------------------------------------------------
# CLI flags override TOML, so the TOMLs carry no absolute paths (portable).
# bf16 + --no_half_vae is the NaN fix; --sdpa avoids depending on an xformers
# wheel matching the torch build (swap to --xformers if you have a matching one).
cd "$SD_SCRIPTS"
echo "[train] starting - watch the first ~20 steps: avr_loss must be finite (0.05-0.25), not nan"
accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py \
  --config_file "$CFG_DIR/lora_$TRIGGER.toml" \
  --dataset_config "$CFG_DIR/dataset_$TRIGGER.toml" \
  --pretrained_model_name_or_path "$BASE_PATH" \
  --output_dir "$OUT_DIR" \
  --output_name "${TRIGGER}_${BASE}_v1" \
  --logging_dir "$OUT_DIR/logs" \
  --sample_prompts "$CFG_DIR/${TRIGGER}_samples.txt" \
  --mixed_precision bf16 \
  --no_half_vae \
  --sdpa \
  2>&1 | tee "$OUT_DIR/train.log"

echo
echo "=== done. epochs in $OUT_DIR: ==="
ls -1 "$OUT_DIR"/*.safetensors || true
echo
echo "NaN check: $(grep -ci nan "$OUT_DIR/train.log" || true) lines mention nan (want 0)"
echo
echo "Pick the earliest epoch that holds identity at strength 0.8 with FaceID off"
echo "(usually epoch 6-8), then send it back to Windows, e.g.:"
echo "  runpodctl send $OUT_DIR/${TRIGGER}_${BASE}_v1-000007.safetensors"
echo
echo "Remember to STOP this pod afterwards (keep only the Network Volume)."
