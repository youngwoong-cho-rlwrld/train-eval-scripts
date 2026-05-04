#!/bin/bash
#SBATCH --job-name=allex-cube-box-5cmleft-youngwoong-pretrained
#SBATCH --partition=l40s-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/rlwrld/youngwoong_cho/logs/%x_%j.out
#SBATCH --error=/fsx/rlwrld/youngwoong_cho/logs/%x_%j.err

set -e
source /fsx/rlwrld/youngwoong_cho/workspace/gr00t/.venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=gr00t

python /fsx/rlwrld/youngwoong_cho/workspace/gr00t/scripts/gr00t_finetune.py \
    --num-gpus 2 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --output-dir /fsx/rlwrld/youngwoong_cho/checkpoints/onboard_cube_box_left_pretrained \
    --data-config /fsx/rlwrld/youngwoong_cho/configs/onboarding.yaml \
    --max-steps 30000 \
    --save-steps 10000 \
    --dataloader-num-workers 16 \
    --video-backend torchcodec \
    --dataloader-prefetch-factor 10 \
    --pin-memory \
    --report-to wandb \
    --run-name onboard_cube_box_left_pretrained \
    --seed 42
