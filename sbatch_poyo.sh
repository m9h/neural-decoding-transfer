#!/usr/bin/env bash
# Non-interactive POYO training/eval on the DGX Spark via Slurm + the
# galaxy-brain:26.05 container.
#
# Usage:
#   sbatch sbatch_poyo.sh <config-name> [hydra overrides...]
#
# Examples:
#   sbatch sbatch_poyo.sh train_poyo_mp                       # POYO-mp on perich_miller (ready now)
#   sbatch sbatch_poyo.sh train_poyo_mp batch_size=256 epochs=200
#   sbatch sbatch_poyo.sh train_poyo_1                        # POYO-1 (needs churchland too)
#   sbatch sbatch_poyo.sh train_poyo_mp wandb.enable=true     # opt into (offline) wandb logging
#
# config-name is one of the files in poyo_harness/configs/ (without .yaml):
#   train_poyo_mp | train_poyo_1 | train_mc_maze_small
#
#SBATCH --job-name=poyo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=112G   # node RealMemory=120000MB; stay under it (118G=120832MB fails)
#SBATCH --output=/home/mhough/dev/neural-decoding-transfer/slurm-logs/poyo-%j.out
#SBATCH --error=/home/mhough/dev/neural-decoding-transfer/slurm-logs/poyo-%j.out
set -euo pipefail

IMAGE="galaxy-brain:26.05"
PROJECT="/home/mhough/dev/neural-decoding-transfer"

CONFIG="${1:-train_poyo_mp}"
shift || true
OVERRIDES=("$@")

echo "[$(date -Is)] node=$(hostname) job=${SLURM_JOB_ID:-none} config=${CONFIG}"
echo "overrides: ${OVERRIDES[*]:-<none>}"

# GB10 unified-memory caveat: CPU+GPU share one 120GB pool and `docker --memory`
# does NOT bound CUDA. The real guard is batch sizing + bf16 (default). Watch
# `nvidia-smi` on the first run of any new config.
exec docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e WANDB_MODE="${WANDB_MODE:-offline}" \
    -e HOME=/workspace \
    -v /data:/data \
    -v "${PROJECT}":/workspace \
    -w /workspace/poyo_harness \
    "${IMAGE}" \
    python train.py --config-name "${CONFIG}" "${OVERRIDES[@]}"
