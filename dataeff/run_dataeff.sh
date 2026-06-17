#!/bin/bash
# Data-efficiency curve, on-pod. POYO-finetune (freeze core, from converged base) vs
# from-scratch, at several train-data fractions, on a single held-out area2_bump session.
# Emits /workspace/dataeff_results.jsonl (one line per run).
#
# Expects already uploaded to /workspace:
#   poyo_mp_converged.ckpt        the reproduced base
#   poyo_transfer_train.patch     finetune wiring (train.py + defaults.yaml)
#   poyo_area2_dataeff.py         dataset (-> src/datasets/)
#   finetune_area2_dataeff.yaml   config   (-> configs/)
#   area2_bump_pipeline/          local brainset pipeline (pipeline.py)
set -euo pipefail
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }

# --- pinned env + repo ---
[ -d poyo ] || git clone --depth 1 https://github.com/nerdslab/poyo.git poyo
cd poyo
git checkout -- . 2>/dev/null || true
if ! grep -q "freeze_core" train.py; then
  git apply /workspace/poyo_transfer_train.patch
fi
grep -q "freeze_core" train.py || { echo "FATAL: finetune patch not applied to train.py"; exit 1; }
[ -d .venv ] || uv venv
source .venv/bin/activate
uv pip install -q -r requirements.txt
cp /workspace/poyo_area2_dataeff.py src/datasets/poyo_area2_dataeff.py
cp /workspace/finetune_area2_dataeff.yaml configs/finetune_area2_dataeff.yaml
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# --- data: prep area2_bump locally via brainsets (one small NLB session) ---
DATA=/workspace/data/processed
if [ ! -f "$DATA/area2_bump/han_area2_bump_train.h5" ]; then
  python -c "import dandi" 2>/dev/null || uv pip install -q "dandi==0.74.0"
  brainsets prepare /workspace/area2_bump_pipeline --local \
    --raw-dir /workspace/data/raw --processed-dir "$DATA"
fi
ls -la "$DATA/area2_bump/"

CKPT=/workspace/poyo_mp_converged.ckpt
RES=/workspace/dataeff_results.jsonl
: > "$RES"
FRACS="0.1 0.25 0.5 1.0"

run_one () {  # frac  cond(ft|scratch)
  local frac="$1" cond="$2" out r
  out="/workspace/out_${cond}_${frac}"
  rm -rf "$out"
  if [ "$cond" = "ft" ]; then
    PRE="$CKPT"; FREEZE=true
  else
    PRE="null"; FREEZE=false
  fi
  echo "=== $(date -u +%H:%MZ) cond=$cond frac=$frac ==="
  # DECOUPLE LR FROM BATCH: repo sets max_lr = base_lr*batch_size. We use batch_size=8 (small
  # session), so bump base_lr to 1e-3 -> max_lr=8e-3, the value the batch-256 finetune used to
  # reach R2=0.834. Same LR for ft AND scratch -> data amount is the only varying factor.
  POYO_TRAIN_FRACTION="$frac" python train.py --config-name finetune_area2_dataeff.yaml \
      data_root="$DATA" log_dir="$out" gpus=1 wandb.enable=false optim.base_lr=1e-3 \
      finetune.pretrained_ckpt="$PRE" finetune.freeze_core="$FREEZE" \
      2>&1 | tee "$out.log" | grep -Ei "epoch|val_metric|test_metric" || true
  # grab the final test R2 printed by Lightning's test()
  r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "$out.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  echo "{\"cond\":\"$cond\",\"frac\":$frac,\"test_r2\":$r}" | tee -a "$RES"
}

for f in $FRACS; do
  run_one "$f" ft
  run_one "$f" scratch
done

echo "[DONE] dataeff complete"
cat "$RES"
