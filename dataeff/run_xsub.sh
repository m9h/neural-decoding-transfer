#!/bin/bash
# Cross-SUBJECT transfer demo: §2.5 finetune (freeze core) vs from-scratch on a HELD-OUT monkey-t
# perich center-out session (never seen by the base). Measures the session's train-window count to
# pick a safe batch (avoid OneCycle total_steps=0), bumps base_lr to keep max_lr~8e-3, and runs
# both conditions. Emits /workspace/xsub_results.jsonl.
#
# Expects uploaded to /workspace: poyo_mp_converged.ckpt, poyo_transfer_train.patch,
#   poyo_perich_session.py, finetune_perich_xsub.yaml, and the t-session .h5 already in
#   /workspace/data/processed/perich_miller_population_2018/.
set -uo pipefail
cd /workspace/poyo
git checkout -- . 2>/dev/null || true
if ! grep -q "freeze_core" train.py; then git apply /workspace/poyo_transfer_train.patch; fi
grep -q "freeze_core" train.py || { echo "FATAL: patch not applied"; exit 1; }
source .venv/bin/activate
cp /workspace/poyo_perich_session.py src/datasets/poyo_perich_session.py
cp /workspace/finetune_perich_xsub.yaml configs/finetune_perich_xsub.yaml

SESSION="${SESSION:-t_20130819_center_out_reaching}"
DATA=/workspace/data/processed
H5="$DATA/perich_miller_population_2018/$SESSION.h5"
[ -f "$H5" ] || { echo "FATAL: missing $H5"; exit 1; }

# measure train-window count to pick batch (window_length=1.0s, sampler floors per interval)
NW=$(python - "$H5" <<'PY'
import sys, h5py, numpy as np
from temporaldata import Data
d = Data.from_hdf5(h5py.File(sys.argv[1], "r"), lazy=True)
iv = d.train_domain
dur = np.asarray(iv.end, float) - np.asarray(iv.start, float)
print(int(np.floor(dur).sum()))
PY
)
echo "train windows ~ $NW"
# pick batch so we get >= ~4 batches/epoch; keep max_lr ~8e-3 via base_lr = 8e-3/batch
BATCH=16; [ "$NW" -ge 256 ] && BATCH=32; [ "$NW" -ge 1024 ] && BATCH=64
BLR=$(python -c "print(8e-3/$BATCH)")
echo "batch=$BATCH base_lr=$BLR (max_lr~8e-3)"

CKPT=/workspace/poyo_mp_converged.ckpt
RES=/workspace/xsub_results.jsonl
: > "$RES"

run_one () {  # cond
  local cond="$1" out r PRE FREEZE
  out="/workspace/xsub_${cond}"; rm -rf "$out"
  if [ "$cond" = "ft" ]; then PRE="$CKPT"; FREEZE=true; else PRE="null"; FREEZE=false; fi
  echo "=== $(date -u +%H:%MZ) session=$SESSION cond=$cond ==="
  POYO_SESSION="$SESSION" python train.py --config-name finetune_perich_xsub.yaml \
      data_root="$DATA" log_dir="$out" gpus=1 wandb.enable=false \
      batch_size="$BATCH" optim.base_lr="$BLR" \
      finetune.pretrained_ckpt="$PRE" finetune.freeze_core="$FREEZE" \
      2>&1 | tee "$out.log" | grep -Ei "epoch|val_metric|test_metric|Testing on|Error|Traceback" || true
  r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "$out.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  echo "{\"session\":\"$SESSION\",\"cond\":\"$cond\",\"test_r2\":$r}" | tee -a "$RES"
}

run_one ft
run_one scratch
echo "[DONE] xsub complete"
cat "$RES"
