#!/bin/bash
# Re-run ONLY the feature-covariance extraction on the clean-named aug ckpts (the in-run featcov
# failed because Lightning's best-ckpt filename contains '=', which breaks Hydra overrides).
# No re-training. Rebuilds /workspace/aug_results.jsonl with the spectral metrics + R2.
set -uo pipefail
cd /workspace/poyo
source .venv/bin/activate
cp /workspace/extract_featcov.py extract_featcov.py

SESSION="${SESSION:-t_20130819_center_out_reaching}"
DATA=/workspace/data/processed
CKPT=/workspace/poyo_mp_converged.ckpt
RES=/workspace/aug_results.jsonl
: > "$RES"

for name in off standard aggressive; do
  FT="/workspace/aug_${name}.ckpt"
  [ -f "$FT" ] || { echo "skip $name (no ckpt)"; continue; }
  r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "/workspace/aug_${name}.train.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  echo "=== $(date -u +%H:%MZ) featcov $name (R2=$r) ==="
  POYO_SESSION="$SESSION" python extract_featcov.py --config-name finetune_perich_xsub.yaml \
     data_root="$DATA" log_dir="/workspace/fc_${name}" gpus=1 \
     finetune.pretrained_ckpt="$CKPT" finetune.freeze_core=true \
     ++ft_ckpt="$FT" ++featcov_out="/workspace/fc_${name}.json" \
     2>&1 | tee "/workspace/fc_${name}.log" | grep -Ei "\[featcov\]|Error|Traceback|mismatch" || true
  R2="$r" NAME="$name" FC="/workspace/fc_${name}.json" RES="$RES" python - <<'PY'
import os, json
name, r, fc, res = os.environ["NAME"], os.environ["R2"], os.environ["FC"], os.environ["RES"]
d = json.load(open(fc)) if os.path.exists(fc) else {}
row = {"level": name, "test_r2": (float(r) if r != "nan" else None),
       "eff_rank": d.get("eff_rank_entropy"), "participation_ratio": d.get("participation_ratio"),
       "trace": d.get("trace"), "top1_frac": d.get("top1_frac"), "cond": d.get("cond_number")}
open(res, "a").write(json.dumps(row) + "\n"); print("ROW", json.dumps(row))
PY
done
echo "[DONE] featcov-fix complete"
cat "$RES"
