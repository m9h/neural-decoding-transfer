# POYO → Merchant-task transfer scaffolding (§2.5 unit identification)

Everything needed to finetune the **converged** POYO-MP base onto a new dataset by freezing the
core and learning new unit/session embeddings (+ a new readout for dmfc_rsg). Built against the
verified local APIs (torch_brain `load_pretrained`/`register_modality`/`extend_vocab`, the repo's
`train.py`/`nlb.py`). **Untested end-to-end** — it needs the converged checkpoint + the brainsets
env + the prepped data, which don't exist yet. Three integration SEAMS are flagged below; finalize
them in-env. This is staged so each stage changes ONE thing (the project's hard-won lesson).

## Apply to a fresh poyo clone
```
git clone --depth 1 https://github.com/nerdslab/poyo.git poyo && cd poyo
git apply  <…>/transfer/poyo_transfer_train.patch          # train.py + defaults.yaml (§2.5 + resume-fix)
cp <…>/transfer/register_timing_modality.py src/datasets/
cp <…>/transfer/poyo_area2.py               src/datasets/
cp <…>/transfer/poyo_dmfc.py                src/datasets/
cp <…>/transfer/finetune_area2.yaml configs/
cp <…>/transfer/finetune_dmfc.yaml  configs/
```
The patch adds a `finetune:` config block (`pretrained_ckpt`/`skip_readout`/`freeze_core`) and wires:
- `main()` → `POYO.load_pretrained(ckpt, readout_spec, skip_readout=…)` when finetuning;
- `_init_model_vocab` → `extend_vocab` (add new units/sessions) instead of `initialize_vocab`;
- `configure_optimizers` → freeze the core (everything except unit/session emb + readout) when `freeze_core`.

## Run (one command per stage)
```
# Stage A — area2_bump: validates the transfer machinery (reuses cursor_velocity_2d, no new readout)
python train.py --config-name finetune_area2.yaml data_root=<area2 processed> \
    finetune.pretrained_ckpt=<converged last.ckpt> log_dir=<out> gpus=<n> wandb.enable=false
# Stage B — dmfc_rsg: the Merchant timing transfer (new interval_timing_1d readout)
python train.py --config-name finetune_dmfc.yaml  data_root=<dmfc processed> \
    finetune.pretrained_ckpt=<converged last.ckpt> log_dir=<out> gpus=<n> wandb.enable=false
```
For each, run the **scratch baseline** too (`finetune.pretrained_ckpt=null freeze_core=false`) →
report R² finetune-vs-scratch. Then `bayes_report.py` (wwj) on the transferred vs base weights.

## SEAMS to finalize in-env (don't guess — verify)
1. **brainsets base class** (`poyo_area2.py`/`poyo_dmfc.py`): `nlb.py` extends
   `brainsets.datasets.PeiPandarinathNLB2021`, but area2_bump/dmfc_rsg are LOCAL pipelines with no
   registered class. Confirm the loader for a local brainset (generic `brainsets.datasets.Dataset`
   keyed by `brainset_id`, or register a class) and fix the one import line. The `READOUT_CONFIG`
   (the part that matters) is fully specified.
2. **vocab persistence** (`_init_model_vocab`): `load_pretrained` restores embedding *weights*, but
   `InfiniteVocabEmbedding.vocab` is a plain dict (not in `state_dict`). Verify whether the vocab map
   survives `load_pretrained`; if NOT, `initialize_vocab(<pretrained_unit_ids>)` before `extend_vocab`
   so the pretrained rows keep their identities. (Check `torch_brain/models/poyo.py:load_pretrained`.)
3. **timing target representation** (dmfc only): `tp` is one scalar/trial, but the readout reads a
   timestamped channel `data.timing`. The dmfc_rsg pipeline must emit it — pick:
   (A) broadcast `tp` constant over each `[set_time, go_time]` window (richer; stitching averages to
   one tp/trial), or (B) a single `(go_time, tp)` IrregularTimeSeries point/trial (simplest, sparse).
   Add ~10 lines to `dmfc_rsg_pipeline/pipeline.py` and re-prep. `tp` is in **ms** (~500–1000) → the
   config `normalize_mean/std` (750/150) center it; tune from the data.

## Why staged
Stage A changes only the transfer *machinery* (same readout) — if R²(hand.vel) finetune ≥ scratch,
the freeze/extend_vocab path works. Stage B then adds the *new readout* on top of a proven machine,
so a failure is localizable to one axis. Same discipline that (eventually) fixed the reproduction.
