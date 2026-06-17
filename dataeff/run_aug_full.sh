#!/bin/bash
# FINETUNE-ALL (unfrozen) UnitDropout ablation — the regime where the JMLR augmentation=spectral-
# regularization effect can appear (plastic feature extractor). Same as run_aug.sh but
# freeze_core=false. Held-out monkey-t session, 3 UnitDropout levels. Per level: test R2, then
# readout-input feature-covariance spectrum. Saves /workspace/aug_full_results.jsonl + per-level ckpt.
set -uo pipefail
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
[ -d poyo ] || git clone --depth 1 https://github.com/nerdslab/poyo.git poyo
cd /workspace/poyo
git checkout -- . 2>/dev/null || true
if ! grep -q "freeze_core" train.py; then git apply /workspace/poyo_transfer_train.patch; fi
grep -q "freeze_core" train.py || { echo "FATAL: patch not applied"; exit 1; }
[ -d .venv ] || uv venv
source .venv/bin/activate
uv pip install -q -r requirements.txt
cp /workspace/poyo_perich_session.py     src/datasets/poyo_perich_session.py
cp /workspace/finetune_perich_xsub.yaml  configs/finetune_perich_xsub.yaml
cp /workspace/extract_featcov.py         extract_featcov.py
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

SESSION="${SESSION:-t_20130819_center_out_reaching}"
DATA=/workspace/data/processed
CKPT=/workspace/poyo_mp_converged.ckpt
RES=/workspace/aug_full_results.jsonl
: > "$RES"

NW=$(python - "$DATA/perich_miller_population_2018/$SESSION.h5" <<'PY'
import sys, h5py, numpy as np
from temporaldata import Data
d = Data.from_hdf5(h5py.File(sys.argv[1], "r"), lazy=True); iv = d.train_domain
print(int(np.floor(np.asarray(iv.end, float) - np.asarray(iv.start, float)).sum()))
PY
)
BATCH=16; [ "$NW" -ge 256 ] && BATCH=32; [ "$NW" -ge 1024 ] && BATCH=64
BLR=$(python -c "print(8e-3/$BATCH)")
echo "session=$SESSION windows~$NW batch=$BATCH base_lr=$BLR FREEZE=false"

run_level () {  # name  [hydra transform override ...]
  local name="$1"; shift
  local out=/workspace/augf_${name}; rm -rf "$out"
  echo "=== $(date -u +%H:%MZ) FULL UnitDropout=$name ==="
  POYO_SESSION="$SESSION" python train.py --config-name finetune_perich_xsub.yaml \
     data_root="$DATA" log_dir="$out" gpus=1 wandb.enable=false \
     batch_size=$BATCH optim.base_lr=$BLR \
     finetune.pretrained_ckpt="$CKPT" finetune.freeze_core=false \
     "$@" 2>&1 | tee "$out.train.log" | grep -Ei "test_metric|Error|Traceback|total_steps" || true
  local r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "$out.train.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  local best=$(find "$out" -name "*.ckpt" ! -name last.ckpt | head -1)
  echo "R2=$r best=$best"
  cp "$best" "/workspace/augf_${name}.ckpt" 2>/dev/null || true
  POYO_SESSION="$SESSION" python extract_featcov.py --config-name finetune_perich_xsub.yaml \
     data_root="$DATA" log_dir="$out" gpus=1 \
     finetune.pretrained_ckpt="$CKPT" finetune.freeze_core=false \
     ++ft_ckpt="/workspace/augf_${name}.ckpt" ++featcov_out="$out/featcov.json" \
     2>&1 | tee "$out.featcov.log" | grep -Ei "\[featcov\]|Error|Traceback" || true
  R2="$r" NAME="$name" FC="$out/featcov.json" RES="$RES" python - <<'PY'
import os, json
name, r, fc, res = os.environ["NAME"], os.environ["R2"], os.environ["FC"], os.environ["RES"]
d = json.load(open(fc)) if os.path.exists(fc) else {}
row = {"level": name, "test_r2": (float(r) if r != "nan" else None),
       "eff_rank": d.get("eff_rank_entropy"), "participation_ratio": d.get("participation_ratio"),
       "trace": d.get("trace"), "top1_frac": d.get("top1_frac"), "cond": d.get("cond_number")}
open(res, "a").write(json.dumps(row) + "\n"); print("ROW", json.dumps(row))
PY
}

run_level off        'train_transforms=[]'
run_level standard
run_level aggressive 'train_transforms.0.mode_units=50'
echo "[DONE] aug-full complete"
cat "$RES"
