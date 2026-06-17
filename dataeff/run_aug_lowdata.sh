#!/bin/bash
# LOW-DATA generalization test: does UnitDropout's spectral regularization BUY generalization when
# the decode isn't saturated? Finetune-ALL (plastic) on held-out monkey-t, grid of
# {train fraction} x {UnitDropout level}. Fixed batch=16 (no cross-fraction batch confound),
# max_lr~8e-3. valid/test never truncated. Saves /workspace/aug_lowdata_results.jsonl.
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
RES=/workspace/aug_lowdata_results.jsonl
: > "$RES"
BATCH=16; BLR=$(python -c "print(8e-3/$BATCH)")
echo "session=$SESSION batch=$BATCH base_lr=$BLR FREEZE=false"

run_cell () {  # frac  level  [transform override...]
  local frac="$1" level="$2"; shift 2
  local tag="${frac}_${level}"
  local out=/workspace/ld_${tag}; rm -rf "$out"
  echo "=== $(date -u +%H:%MZ) frac=$frac UnitDropout=$level ==="
  POYO_TRAIN_FRACTION="$frac" POYO_SESSION="$SESSION" python train.py --config-name finetune_perich_xsub.yaml \
     data_root="$DATA" log_dir="$out" gpus=1 wandb.enable=false \
     batch_size=$BATCH optim.base_lr=$BLR \
     finetune.pretrained_ckpt="$CKPT" finetune.freeze_core=false \
     "$@" 2>&1 | tee "$out.train.log" | grep -Ei "test_metric|Error|Traceback|total_steps" || true
  local r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "$out.train.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  local best=$(find "$out" -name "*.ckpt" ! -name last.ckpt | head -1)
  echo "frac=$frac level=$level R2=$r"
  cp "$best" "/workspace/ld_${tag}.ckpt" 2>/dev/null || true
  POYO_SESSION="$SESSION" python extract_featcov.py --config-name finetune_perich_xsub.yaml \
     data_root="$DATA" log_dir="$out" gpus=1 \
     finetune.pretrained_ckpt="$CKPT" finetune.freeze_core=false \
     ++ft_ckpt="/workspace/ld_${tag}.ckpt" ++featcov_out="$out/featcov.json" \
     2>&1 | tee "$out.featcov.log" | grep -Ei "\[featcov\]|Error|Traceback" || true
  FRAC="$frac" LEVEL="$level" R2="$r" FC="$out/featcov.json" RES="$RES" python - <<'PY'
import os, json
frac, level, r, fc, res = os.environ["FRAC"], os.environ["LEVEL"], os.environ["R2"], os.environ["FC"], os.environ["RES"]
d = json.load(open(fc)) if os.path.exists(fc) else {}
row = {"frac": float(frac), "level": level, "test_r2": (float(r) if r != "nan" else None),
       "eff_rank": d.get("eff_rank_entropy"), "top1_frac": d.get("top1_frac"),
       "trace": d.get("trace"), "cond": d.get("cond_number")}
open(res, "a").write(json.dumps(row) + "\n"); print("ROW", json.dumps(row))
PY
}

for frac in 0.1 0.25 1.0; do
  run_cell "$frac" off        'train_transforms=[]'
  run_cell "$frac" standard
  run_cell "$frac" aggressive 'train_transforms.0.mode_units=50'
done
echo "[DONE] aug-lowdata complete"
cat "$RES"
