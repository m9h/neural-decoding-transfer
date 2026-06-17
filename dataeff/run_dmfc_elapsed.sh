#!/bin/bash
# dmfc_rsg ELAPSED-TIME (time-to-go ramp) transfer: POYO-1 finetune (freeze core, skip_readout)
# vs from-scratch, on the haydn dmfc_rsg session. Emits /workspace/dmfc_elapsed_results.jsonl.
#
# Expects uploaded to /workspace:
#   poyo_mp_converged.ckpt, poyo_transfer_train.patch,
#   register_timing_modality.py, poyo_dmfc_elapsed.py, finetune_dmfc_elapsed.yaml,
#   dmfc_rsg_pipeline/  (pipeline.py edited to emit the time-to-go ramp)
set -uo pipefail
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }

[ -d poyo ] || git clone --depth 1 https://github.com/nerdslab/poyo.git poyo
cd poyo
git checkout -- . 2>/dev/null || true
if ! grep -q "freeze_core" train.py; then git apply /workspace/poyo_transfer_train.patch; fi
grep -q "freeze_core" train.py || { echo "FATAL: patch not applied"; exit 1; }
[ -d .venv ] || uv venv
source .venv/bin/activate
uv pip install -q -r requirements.txt
cp /workspace/register_timing_modality.py src/datasets/register_timing_modality.py
cp /workspace/poyo_dmfc_elapsed.py        src/datasets/poyo_dmfc_elapsed.py
cp /workspace/finetune_dmfc_elapsed.yaml  configs/finetune_dmfc_elapsed.yaml
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

DATA=/workspace/data/processed
if [ ! -f "$DATA/dmfc_rsg/haydn_dmfc_rsg_train.h5" ]; then
  python -c "import dandi" 2>/dev/null || uv pip install -q "dandi==0.74.0"
  brainsets prepare /workspace/dmfc_rsg_pipeline --local --raw-dir /workspace/data/raw --processed-dir "$DATA"
fi
ls -la "$DATA/dmfc_rsg/"

CKPT=/workspace/poyo_mp_converged.ckpt
RES=/workspace/dmfc_elapsed_results.jsonl
: > "$RES"

run_one () {  # cond(ft|scratch)
  local cond="$1" out r PRE FREEZE
  out="/workspace/dmfce_${cond}"; rm -rf "$out"
  if [ "$cond" = "ft" ]; then PRE="$CKPT"; FREEZE=true; else PRE="null"; FREEZE=false; fi
  echo "=== $(date -u +%H:%MZ) cond=$cond ==="
  python train.py --config-name finetune_dmfc_elapsed.yaml \
      data_root="$DATA" log_dir="$out" gpus=1 wandb.enable=false \
      finetune.pretrained_ckpt="$PRE" finetune.freeze_core="$FREEZE" \
      2>&1 | tee "$out.log" | grep -Ei "epoch|val_metric|test_metric|Testing on|Error|Traceback" || true
  r=$(grep -Eo "average_test_metric[^0-9-]*[-0-9.]+" "$out.log" | tail -1 | grep -Eo "[-0-9.]+$" || echo nan)
  echo "{\"cond\":\"$cond\",\"target\":\"time_to_go\",\"test_r2\":$r}" | tee -a "$RES"
}

run_one ft
run_one scratch
echo "[DONE] dmfc elapsed complete"
cat "$RES"
