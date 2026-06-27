#!/bin/bash
# CROSS-SPECIES spatial transfer (farthest domain): §2.5 finetune (freeze core) vs from-scratch on
# HELD-OUT rat (Vollan/Moser 2025) MEC/hippocampus open-field sessions, decoding 2D position via a
# NEW position_2d head. Monkey-motor base -> rat spatial coding. Per session/cond: test R²(position).
# Emits /workspace/vollan_results.jsonl.
#
# Expects uploaded to /workspace: poyo_mp_converged.ckpt, poyo_transfer_train.patch,
#   poyo_vollan_session.py, register_position_modality.py, finetune_vollan.yaml, and the chosen
#   of_*.h5 already under /workspace/data/processed/vollan_moser_alternating_2025/.
set -uo pipefail
# --- self-bootstrap on a fresh pod (clone poyo + pinned env + apply transfer patch) ---
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d poyo ] || git clone --depth 1 https://github.com/nerdslab/poyo.git poyo
cd /workspace/poyo
git checkout -- . 2>/dev/null || true
if ! grep -q "freeze_core" train.py; then git apply /workspace/poyo_transfer_train.patch; fi
grep -q "freeze_core" train.py || { echo "FATAL: patch not applied"; exit 1; }
[ -d .venv ] || uv venv
source .venv/bin/activate
uv pip install -q -r requirements.txt
cp /workspace/poyo_vollan_session.py        src/datasets/poyo_vollan_session.py
cp /workspace/register_position_modality.py src/datasets/register_position_modality.py
cp /workspace/finetune_vollan.yaml          configs/finetune_vollan.yaml
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

DATA=/workspace/data/processed
DIR="$DATA/vollan_moser_alternating_2025"
CKPT=/workspace/poyo_mp_converged.ckpt
RES=/workspace/vollan_results.jsonl
: > "$RES"

# default: the open-field sessions validated locally (clean x/y, healthy window counts)
SESSIONS="${SESSIONS:-of_24365_2 of_24666_1 of_25127_1}"
echo "sessions: $SESSIONS"

nwin () {  # h5 -> floored 1s train-window count (window_length=1.0s; sampler floors per interval)
  python - "$1" <<'PY'
import sys, h5py, numpy as np
from temporaldata import Data
d = Data.from_hdf5(h5py.File(sys.argv[1], "r"), lazy=True)
iv = d.train_domain
print(int(np.floor(np.asarray(iv.end, float) - np.asarray(iv.start, float)).sum()))
PY
}

run_one () {  # session cond batch blr
  local session="$1" cond="$2" batch="$3" blr="$4" out r PRE FREEZE
  out="/workspace/vollan_${session}_${cond}"; rm -rf "$out"
  if [ "$cond" = "ft" ]; then PRE="$CKPT"; FREEZE=true; else PRE="null"; FREEZE=false; fi
  echo "=== $(date -u +%H:%MZ) session=$session cond=$cond batch=$batch base_lr=$blr ==="
  POYO_SESSION="$session" python train.py --config-name finetune_vollan.yaml \
      data_root="$DATA" log_dir="$out" gpus=1 wandb.enable=false \
      batch_size="$batch" optim.base_lr="$blr" \
      finetune.pretrained_ckpt="$PRE" finetune.freeze_core="$FREEZE" \
      2>&1 | tee "$out.log" | grep -Ei "epoch|val_metric|test_metric|Testing on|Error|Traceback|total_steps" || true
  r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "$out.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  echo "{\"session\":\"$session\",\"cond\":\"$cond\",\"test_r2\":$r}" | tee -a "$RES"
}

for session in $SESSIONS; do
  H5="$DIR/$session.h5"
  [ -f "$H5" ] || { echo "SKIP (missing) $H5"; continue; }
  NW=$(nwin "$H5"); echo "$session train windows ~ $NW"
  BATCH=16; [ "$NW" -ge 256 ] && BATCH=32; [ "$NW" -ge 1024 ] && BATCH=64
  BLR=$(python -c "print(8e-3/$BATCH)")   # keep max_lr ~8e-3 (max_lr = base_lr * batch)
  run_one "$session" ft      "$BATCH" "$BLR"
  run_one "$session" scratch "$BATCH" "$BLR"
done
echo "[DONE] vollan cross-species transfer complete"
cat "$RES"
