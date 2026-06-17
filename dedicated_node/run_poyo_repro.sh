#!/usr/bin/env bash
# Faithful POYO-MP reproduction on a DEDICATED (non-preemptible) 8xGPU node.
#
# WHY a dedicated node: every dollar wasted so far came from running on PREEMPTIBLE
# Modal spot nodes -- container deaths forced a self-healing resume loop, and resuming
# corrupted the OneCycle LR schedule (peak relocated -> R2 flat-lined). A non-preemptible
# node runs the UNMODIFIED repo in ONE uninterrupted pass: no resume logic, no schedule
# hacks, faithful by construction. ~97 steps/epoch x 1000 ~= 9h, target R2 ~= 0.9.
#
# TARGET: RunPod / Lambda 8x H100-80GB (or 8x A100-80GB; slower). ~$18-24/hr -> ~$180-220.
# Run as:  bash run_poyo_repro.sh 2>&1 | tee repro.log
set -euo pipefail

# ---- knobs ----
GPUS="${GPUS:-8}"
EPOCHS="${EPOCHS:-1000}"
WORKDIR="${WORKDIR:-/workspace}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data/processed}"   # must contain perich_miller_population_2018/*.h5
OUT_DIR="${OUT_DIR:-$WORKDIR/poyo_out}"
REPO_DIR="$WORKDIR/poyo"

mkdir -p "$WORKDIR" "$OUT_DIR"
cd "$WORKDIR"

echo "[1/4] clone UNMODIFIED nerdslab/poyo (Apache-2.0)"
[ -d "$REPO_DIR" ] || git clone --depth 1 https://github.com/nerdslab/poyo.git "$REPO_DIR"

echo "[2/4] pinned env (their requirements.txt: torch~2.10, torch_brain@ca3cfb, brainsets@4ccee58, temporaldata==0.1.4)"
python -m venv "$WORKDIR/.venv"
source "$WORKDIR/.venv/bin/activate"
pip install -U pip
pip install -r "$REPO_DIR/requirements.txt"

echo "[3/4] verify data present (transfer it BEFORE running -- see README; re-prepping from DANDI adds hours)"
if [ ! -d "$DATA_DIR/perich_miller_population_2018" ]; then
  echo "!! No data at $DATA_DIR/perich_miller_population_2018"
  echo "   Either transfer the 111 pre-prepped .h5 (README step) or re-prep on-node:"
  echo "     brainsets prepare perich_miller_population_2018 --raw-dir $WORKDIR/data/raw --processed-dir $DATA_DIR --cores 8"
  exit 1
fi
N_H5=$(find "$DATA_DIR/perich_miller_population_2018" -name '*.h5' | wc -l)
echo "   found $N_H5 .h5 (expect 111)"

echo "[4/4] SINGLE uninterrupted run -- EXACT recipe, NO overrides except gpus/epochs/log/data/wandb"
cd "$REPO_DIR"
python -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count(),torch.cuda.get_device_name(0))"
exec python train.py --config-name train_poyo_mp.yaml \
    data_root="$DATA_DIR" \
    log_dir="$OUT_DIR" \
    gpus="$GPUS" \
    epochs="$EPOCHS" \
    wandb.enable=false
# Final checkpoint: $OUT_DIR/lightning_logs/version_0/checkpoints/  (last.ckpt + epoch=999-*.ckpt)
# scp it back, then run alpha_report.py / bayes_report.py on it (Track A tooling).
