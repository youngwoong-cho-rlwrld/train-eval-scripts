#!/bin/bash
#SBATCH --job-name=allex-cube-box-5cmleft-youngwoong-scratch
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --output=/rlwrld2/home/youngwoong_cho/logs/%x_%j.out
#SBATCH --error=/rlwrld2/home/youngwoong_cho/logs/%x_%j.err

set -e
source /rlwrld2/home/youngwoong_cho/workspace/gr00t/.venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=gr00t

python /rlwrld2/home/youngwoong_cho/workspace/gr00t/scripts/gr00t_finetune.py \
    --num-gpus 2 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --output-dir /rlwrld2/home/youngwoong_cho/checkpoints/onboard_cube_box_left_scratch \
    --data-config /rlwrld2/home/youngwoong_cho/configs/onboarding.yaml \
    --max-steps 30000 \
    --save-steps 10000 \
    --dataloader-num-workers 16 \
    --video-backend torchcodec \
    --dataloader-prefetch-factor 10 \
    --pin-memory \
    --report-to wandb \
    --random-diffusion \
    --run-name onboard_cube_box_left_scratch \
    --seed 42
