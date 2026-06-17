# neural-decoding-transfer

Reproducing the [neuro-galaxy](https://github.com/neuro-galaxy) **POYO** spiking-foundation
model and characterizing what its representation transfers — across sessions, time, subjects,
and tasks — plus a study of UnitDropout as implicit spectral regularization. Reproduction and
the showcase/transfer experiments run on the DGX Spark (GB10, sm_121, aarch64) and on cloud
8×H100 / 1×H100 for the heavy training.

**Headline results** (see `paper/paper.pdf`): faithful POYO-MP reproduction (held-out test
R²=0.904); foundation-model payoff in data efficiency, 3-year chronic stability, and
cross-subject transfer to an unseen animal; an honest motor→timing negative-transfer limit;
and a bridge from the Lin/Dyer/Muthukumar (JMLR'24) linear augmentation theory to a deep model.

## Layout

| path | what |
|------|------|
| `paper/` | reproducible-document manuscript (Python/Pweave pattern): `generate.py` tangles `dataeff/*.jsonl` → figures + macros + tables, `paper.tex` weaves with `tectonic`. `bash paper/build.sh` rebuilds end-to-end. |
| `dataeff/` | transfer + augmentation experiment harness and raw result pulls (`*_results.jsonl`, `*.png`): data efficiency, chronic stability, cross-subject, dmfc timing, the UnitDropout/spectral grids. |
| `transfer/` | §2.5 unit-identification transfer machinery (load_pretrained + extend_vocab + freeze_core patch, dataset/config scaffolding). |
| `Dockerfile`, `constraints.txt` | `galaxy-brain:26.05` image: NGC PyTorch 26.05 + torch_brain 0.1.2 / brainsets / temporaldata + training stack. `constraints.txt` pins the sm_121 torch so pip can't swap it. |
| `run.sh` | interactive container shell with GB10 flags + `/data` bind-mount. |
| `sbatch_poyo.sh` | non-interactive POYO training/eval as a Slurm job. |
| `poyo_harness/` | vendored `torch_brain` v0.1.2 `examples/poyo` harness, wired to our data. |
| `alpha_report.py`, `bayes_report.py` | data-free HTSR-α model-quality scoring via [wwj](../wwj) (JAX WeightWatcher). |
| `logs/`, `slurm-logs/`, `ckpts/` | training outputs / Slurm stdout / checkpoints (git-ignored; large). |

## Data

NAS, prepared by the brainsets pipeline at `/data/datasets/brainsets/processed/<brainset>/`
(per-session `.h5`, temporaldata format with train/valid/test split intervals baked in).
`data_root` in `poyo_harness/configs/defaults.yaml` points there.

Ready: `flint_slutzky_accurate_2012`, `odoherty_sabes_nonhuman_2017`,
`pei_pandarinath_nlb_2021`, `perich_miller_population_2018`,
`vollan_moser_alternating_2025`. Still preparing: `churchland_shenoy_neural_2012`.

- **POYO-mp** (`train_poyo_mp`) trains on `perich_miller_population_2018` only — **runnable now**.
- **POYO-1** (`train_poyo_1`) combines perich_miller + flint + odoherty + churchland —
  needs churchland to finish first.

## Train / eval (Slurm)

```bash
cd ~/dev/neural-decoding-transfer
sbatch sbatch_poyo.sh train_poyo_mp                  # POYO-mp, ready now
sbatch sbatch_poyo.sh train_poyo_mp batch_size=256 epochs=200   # smaller/faster
sbatch sbatch_poyo.sh train_poyo_1                   # once churchland is ready
squeue -u $USER
```

The harness trains, then runs `trainer.test(...)` reporting **R² decoding** per session
(stitched). Checkpoints + `R²` land in `logs/`. wandb is off by default
(`wandb.enable=true` to turn on; runs offline).

GB10 defaults in `configs/defaults.yaml`: `precision: bf16-mixed`, `num_workers: 8`.
Override `precision=32` to reproduce the paper exactly. **Unified-memory caveat:**
CPU+GPU share 120 GB and `docker --memory` does not bound CUDA — watch `nvidia-smi`
on the first run of any new config; drop `batch_size` if memory climbs.

## Quality score (HTSR α, data-free)

Complements R² with the weight-spectrum power-law exponent (well-trained ≈ 2). Runs
in the wwj venv (JAX + CPU torch), reading a Lightning checkpoint directly:

```bash
/home/mhough/dev/wwj/.venv/bin/python alpha_report.py \
    logs/<run>/checkpoints/last.ckpt --out logs/<run>/alpha.json
```

Embedding tables (unit/session vocab) are skipped; only transformer weights are scored.
On an untrained model every layer reports `PL-valid=False` (random matrices aren't
power-laws) — that flipping to True with α→2 after training is the signal.

## Throughput & cloud projection

Measured on the GB10 (`poyo_harness/bench_throughput.py`, real train steps @batch128,
11.8M model): **POYO-mp 128.3 samp/s** (99,644 samp/epoch), **POYO-1 57.2 samp/s**
(185,082 samp/epoch). At the configs' default `epochs: 1000` that's ~9 days (mp) /
~37 days (POYO-1) on the single Spark — epochs is the dominant lever, times scale
linearly with it.

```bash
# remeasure locally
docker run --rm --gpus all --ipc=host -v /data:/data -v "$PWD":/workspace \
  -w /workspace/poyo_harness galaxy-brain:26.05 python bench_throughput.py train_poyo_mp
```

For multi-GPU/cloud, `modal_probe.py` measures per-GPU samples/sec across GPU types
(A100/H100/L40S) and projects 8-GPU full-job time + cost, anchored to the GB10
baselines above. POYO is small (12–17 GB at batch 128) and can't saturate an H100, so
the probe sweeps types to find best $/result — A100/L40S often beat H100 here.

```bash
modal volume create poyo-probe-data
modal volume put poyo-probe-data <a few perich + odoherty .h5> /<dirname>/<file>.h5
modal run modal_probe.py --gpus "A100-80GB,H100,L40S"
```

Modal uses standard CUDA GPUs (x86_64, sm_80/89/90) → plain PyPI torch, none of the
sm_121/NGC handling the Spark needs. Verify `COST_PER_GPU_HR` against current pricing.
