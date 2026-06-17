# Dedicated-node POYO-MP reproduction (do-it-right, uninterrupted)

The faithful full run that the preemptible Modal 8×H100 kept thrashing on. On a
**non-preemptible** node it's a single ~9h pass of the **unmodified** repo — none of the
resume/schedule/durability machinery that caused the wasted spend is needed.

**Cost:** RunPod/Lambda 8×H100-80GB on-demand ≈ $18–24/hr × ~9h ≈ **$180–220**, fixed.
(8×A100-80GB works too, ~1.5–2× slower.)

## Steps

1. **Provision** an on-demand (NOT spot/interruptible) 8×H100-80GB pod on RunPod or Lambda.
   Image: any recent CUDA + Python 3.11 (e.g. RunPod `pytorch:2.4` template, or Lambda's
   PyTorch stack). Add your SSH key (`~/.ssh/id_ed25519.pub`).

2. **Get the data onto the node** (skip the multi-hour DANDI re-prep — the 111 prepped `.h5`
   already exist on the Modal volume). From this machine:
   ```bash
   # pull prepped perich .h5 from Modal -> local -> node
   modal volume get poyo-repo-data processed/perich_miller_population_2018 /tmp/perich_h5
   tar czf /tmp/perich.tgz -C /tmp/perich_h5 .
   scp /tmp/perich.tgz root@<NODE_IP>:/workspace/
   ssh root@<NODE_IP> 'mkdir -p /workspace/data/processed/perich_miller_population_2018 && \
       tar xzf /workspace/perich.tgz -C /workspace/data/processed/perich_miller_population_2018'
   ```
   (Alternative, no transfer: let the script re-prep on-node via `brainsets prepare` — adds
   ~1–3h of DANDI download.)

3. **Run** (single uninterrupted pass):
   ```bash
   scp run_poyo_repro.sh root@<NODE_IP>:/workspace/
   ssh root@<NODE_IP> 'cd /workspace && bash run_poyo_repro.sh 2>&1 | tee repro.log'
   ```
   Watch `repro.log` for `average_val_metric` — R² stays ~0 until the ~epoch-500 LR peak,
   then climbs toward ~0.9 by epoch 1000. (No resume = no schedule corruption, so the climb
   is the real one.)

4. **Retrieve + analyze.** `scp` back `…/version_0/checkpoints/last.ckpt`, then run the Track A
   tooling on it: `bayes_report.py` / `alpha_report.py` (wwj venv) for the spectral axis,
   and use it as the pretrained base for the Merchant `dmfc_rsg` transfer.

## What's deliberately NOT here
No `ckpt_path` resume, no periodic commits, no `onecycle_total_steps.txt`, no `retries` — all
of that was scaffolding to survive preemption. A dedicated node makes it unnecessary, which is
exactly why it's faithful: it's the authors' exact `train.py` run once, start to finish.

## Sanity targets
- `Training on 99644 samples / … / 99 sessions`, ~97 steps/epoch.
- `epoch=999-step=~97000.ckpt` at the end.
- final val R² climbing into the high tenths (de-risk reached 0.16 in just 100 epochs on the
  correct schedule; 1000 epochs targets ~0.9).
