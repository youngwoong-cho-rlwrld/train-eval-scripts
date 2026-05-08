#!/usr/bin/env bash
# Run by sbatch via the top-level submit wrapper.
# Reads $VARIANT and $CLUSTER from the environment (set by submit --export).
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

# REPO_ROOT, CLUSTER, VARIANT come from sbatch --export (see submit wrapper).
: "${REPO_ROOT:?REPO_ROOT must be set by submit wrapper}"
: "${CLUSTER:?CLUSTER must be set by submit wrapper}"
: "${VARIANT:?VARIANT must be set by submit wrapper}"
source "$REPO_ROOT/clusters/${CLUSTER}.env"
source "$REPO_ROOT/lib/_common.sh"

EXP_DIR="$REPO_ROOT/experiments/$VARIANT"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR"; exit 1; }
source "$EXP_DIR/config.sh"

GPU_INSTANCE="$(detect_gpu_instance)"
# EXP_NAME mirrors the slurm job name when launched via submit; fallback for ad-hoc runs.
EXP_NAME="${SLURM_JOB_NAME:-${VARIANT}_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)}"

CKPT_DIR="$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/logs" "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="$EXP_DIR/logs/train.log"

log "========================================="
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=$PARTITION  gpu=$GPU_INSTANCE"
log "  variant note: $TRAIN_NOTE"
log "========================================="

# Render data_config.yaml from cluster + variant config
DATA_PATH="$DATA_DIR/$DATASET_NAME"
DATA_CONFIG_YAML="$EXP_DIR/data_config.yaml"
cat > "$DATA_CONFIG_YAML" <<EOF
train:
  datasets:
    - path: $DATA_PATH
      embodiment_tag: new_embodiment
      data_config: $DATA_CONFIG
      weight: 1.0
EOF
log "Dataset:        $DATA_PATH"
log "Data config:    $DATA_CONFIG"
log "Output:         $CKPT_DIR"
log "Max steps:      $MAX_STEPS"

if [ -d "$CKPT_DIR/checkpoint-${MAX_STEPS}" ]; then
    log "Final checkpoint already exists at $CKPT_DIR/checkpoint-${MAX_STEPS} — skipping training."
    exit 0
fi

cd "$GROOT_DIR"
source "$GROOT_DIR/.venv/bin/activate"
export WANDB_PROJECT=gr00t
export WANDB_DIR="$EXP_DIR"

python scripts/gr00t_finetune.py \
    --num-gpus "$TRAIN_NUM_GPUS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --learning_rate 1e-4 \
    --output-dir "$CKPT_DIR" \
    --data-config "$DATA_CONFIG_YAML" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --dataloader_num_workers 16 \
    --dataloader-prefetch-factor 10 \
    --video-backend torchcodec \
    --report-to wandb \
    --pin_memory \
    --run_name "$EXP_NAME" \
    --seed 42 \
    "${TRAIN_EXTRA_ARGS[@]}"

log "Training completed."
