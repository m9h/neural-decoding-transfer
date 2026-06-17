#!/usr/bin/env bash
# Launch the galaxy-brain / POYO container on the DGX Spark.
#   ./run.sh                      -> interactive shell
#   ./run.sh brainsets --help     -> run a command in the container
#
# GB10 caveat: CPU and GPU share one 120 GB pool. `docker --memory` does NOT
# bound CUDA allocations on unified memory, so a runaway allocation can freeze
# the whole box. Keep batch sizes sane and watch `nvidia-smi`. We cap host RAM
# as a soft backstop and enable PyTorch's expandable allocator to reduce
# fragmentation, but the real guard is workload sizing.
set -euo pipefail

IMAGE="galaxy-brain:26.05"

exec docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e WANDB_MODE="${WANDB_MODE:-offline}" \
    -v /data:/data \
    -v "$(pwd)":/workspace \
    "$IMAGE" "$@"
