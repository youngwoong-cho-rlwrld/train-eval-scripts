#!/bin/bash
#SBATCH --job-name=baseline_scratch
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --output=/rlwrld2/home/youngwoong_cho/logs/baseline_scratch_%j.out
#SBATCH --error=/rlwrld2/home/youngwoong_cho/logs/baseline_scratch_%j.err
#SBATCH --comment="baseline_scratch: no torque, with random-diffusion (action head from scratch)"

###############################################################################
# baseline_scratch: Cube_Box_left baseline_scratch without torque & random-diffusion
#
# 학습 → 평가 → 결과 집계까지 한번에 실행
# 모든 결과는 CLAUDE_EXPERIMENTS/baseline_scratch/ 에 저장
#
# 사용법:
#   cd ~/scripts
#   sbatch baseline_scratch/run.sh
###############################################################################

set -e

export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

# ─── 경로 설정 (resolve from Slurm env when sbatch'd, realpath when run directly) ───
# Under Slurm, $0 points to a temp copy at /var/spool/slurmd/<jobid>/..., so we
# anchor off SLURM_JOB_NAME (= the #SBATCH --job-name = parent dir) and
# SLURM_SUBMIT_DIR (where sbatch was invoked) instead.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PARENT_DIR="${SLURM_JOB_NAME:-baseline_scratch}"
    if [[ "$SLURM_SUBMIT_DIR" == */"$PARENT_DIR" ]]; then
        EXP_DIR="$SLURM_SUBMIT_DIR"
    elif [ -d "$SLURM_SUBMIT_DIR/$PARENT_DIR" ]; then
        EXP_DIR="$SLURM_SUBMIT_DIR/$PARENT_DIR"
    elif [ -d "$SLURM_SUBMIT_DIR/kakao/$PARENT_DIR" ]; then
        EXP_DIR="$SLURM_SUBMIT_DIR/kakao/$PARENT_DIR"
    else
        EXP_DIR="$SLURM_SUBMIT_DIR"
    fi
else
    EXP_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
    PARENT_DIR="$(basename "$EXP_DIR")"
fi
GROOT_DIR="/rlwrld2/home/youngwoong_cho/workspace/gr00t"
ISAAC_DIR="/rlwrld2/home/youngwoong_cho/workspace/rlwrld_isaac"

# ─── Detect GPU instance + compose EXP_NAME ───
# Format: {parent_dir}_{gpu_instance}_{yyyymmddHHMMSS}
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "$gpu_name" in
    *A100*) GPU_INSTANCE="a100" ;;
    *H100*) GPU_INSTANCE="h100" ;;
    *H200*) GPU_INSTANCE="h200" ;;
    *L40S*) GPU_INSTANCE="l40s" ;;
    *V100*) GPU_INSTANCE="v100" ;;
    *)      GPU_INSTANCE="unknown" ;;
esac
EXP_NAME="${PARENT_DIR}_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)"

# Rename Slurm job so squeue/sacct show the dynamic name
[ -n "$SLURM_JOB_ID" ] && scontrol update job="$SLURM_JOB_ID" JobName="$EXP_NAME" 2>/dev/null || true

# ─── 실험 설정 ───
DATA_PATH="/rlwrld2/home/seungcheol/80_datasets/v4/v4_cube_box_5cm_left_100_100"
DATA_CONFIG="allex_thetwo_ck40_egostereo"

# 학습 설정
MAX_STEPS=30000
SAVE_STEPS=10000

# 평가 설정
TASK_NAME="task-Cube_Box-5cmLeft"
INSTRUCTION="Pick up the cube with your left hand and place it in the box"
N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3
EVAL_SETS=(0cm 3cm)

# ─── 결과 저장 경로 ───
CKPT_DIR="${EXP_DIR}/checkpoints"
EVAL_DIR="${EXP_DIR}/eval_results"

mkdir -p "${EXP_DIR}/logs" "/rlwrld2/home/youngwoong_cho/logs" "${EVAL_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${EXP_DIR}/logs/run.log"
}

###############################################################################
# Phase 1: 학습
###############################################################################

log "=========================================="
log "${EXP_NAME} Phase 1: Training"
log "=========================================="

source "${GROOT_DIR}/.venv/bin/activate"

# 런타임 YAML 생성
mkdir -p "${EXP_DIR}"
CONFIG_YAML="${EXP_DIR}/data_config.yaml"
cat > "${CONFIG_YAML}" << EOFCFG
train:
  datasets:
    - path: ${DATA_PATH}
      embodiment_tag: new_embodiment
      data_config: ${DATA_CONFIG}
      weight: 1.0
EOFCFG

cd "${GROOT_DIR}"

log "Dataset: ${DATA_PATH}"
log "Data config: ${DATA_CONFIG}"
log "Output: ${CKPT_DIR}"
log "Max steps: ${MAX_STEPS}"
log "NOTE: WITH --random-diffusion (action head from scratch), No torque"

export WANDB_PROJECT=gr00t
export WANDB_DIR="${EXP_DIR}"

if [ -d "${CKPT_DIR}/checkpoint-${MAX_STEPS}" ]; then
    log "Final checkpoint already exists at ${CKPT_DIR}/checkpoint-${MAX_STEPS} — skipping Phase 1"
else
    python scripts/gr00t_finetune.py \
        --num-gpus 2 \
        --batch-size 64 \
        --learning_rate 1e-4 \
        --output-dir "${CKPT_DIR}" \
        --data-config "${CONFIG_YAML}" \
        --max-steps ${MAX_STEPS} \
        --save-steps ${SAVE_STEPS} \
        --dataloader_num_workers 16 \
        --video-backend torchcodec \
        --dataloader-prefetch-factor 10 \
        --report-to wandb \
        --pin_memory \
        --random-diffusion \
        --run_name "${EXP_NAME}" \
        --seed 42

    log "Training completed!"
fi

# 마지막 체크포인트 찾기
LAST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$LAST_CKPT" ]; then
    log "ERROR: No checkpoint found after training"
    exit 1
fi
log "Last checkpoint: ${LAST_CKPT}"

