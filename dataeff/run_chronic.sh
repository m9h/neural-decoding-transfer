#!/bin/bash
# Chronic-stability eval: frozen POYO-MP base evaluated per-session across ~3 years of Chewie
# center-out recordings. Emits /workspace/chronic_results.jsonl (session, date, R2).
set -uo pipefail
cd /workspace/poyo
source .venv/bin/activate
cp /workspace/poyo_perich_session.py src/datasets/poyo_perich_session.py
cp /workspace/eval_session.py eval_session.py
cp /workspace/eval_perich.yaml configs/eval_perich.yaml

CKPT=/workspace/poyo_mp_converged.ckpt
DATA=/workspace/data/processed
RES=/workspace/chronic_results.jsonl
: > "$RES"

SESSIONS="c_20131003_center_out_reaching c_20131220_center_out_reaching \
c_20150309_center_out_reaching c_20150629_center_out_reaching c_20150716_center_out_reaching \
c_20151103_center_out_reaching c_20151201_center_out_reaching c_20160909_center_out_reaching \
c_20161021_center_out_reaching"

for s in $SESSIONS; do
  out=/workspace/eval_$s
  rm -rf "$out"
  echo "=== $(date -u +%H:%MZ) eval $s ==="
  POYO_SESSION="$s" python eval_session.py --config-name eval_perich.yaml \
      data_root="$DATA" log_dir="$out" gpus=1 \
      finetune.pretrained_ckpt="$CKPT" \
      2>&1 | tee "$out.log" | grep -Ei "test_metric|Testing on|Error|Traceback" || true
  r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "$out.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  d=$(echo "$s" | grep -oE "201[0-9]{5}")
  echo "{\"session\":\"$s\",\"date\":\"$d\",\"r2\":$r}" | tee -a "$RES"
done

echo "[DONE] chronic complete"
cat "$RES"
