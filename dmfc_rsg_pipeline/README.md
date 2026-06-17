# dmfc_rsg — local brainsets pipeline (the Merchant-aligned POYO target)

Onboards **DANDI:000130** (`DMFC_RSG`: macaque **dorsomedial frontal cortex** spiking
during a **Ready-Set-Go time-interval reproduction** task; Neural Latents Benchmark '21,
Sohn–Narain–Meirhaeghe–Jazayeri) into the POYO/brainsets format.

## Why this is the Merchant target
Hugo Merchant (UNAM/INB) studies **interval timing in the medial premotor system**
(pre-SMA/SMA) — "population clocks" for rhythmic tapping. `dmfc_rsg` is the closest
**public, raw-spike, benchmarked** analogue: same scientific question (a cognitive
timing computation in medial/dorsomedial frontal cortex), on a dataset POYO can ingest.

Merchant's *own* rhythmic-tapping recordings (Betancourt et al., Cell Reports 2023) are
on Zenodo (records 8361030 code / **8352554 data**) but the deposit is **processed MATLAB
firing-rate matrices, not raw spike times** — so it is **not POYO-ingestible** without
going back to the lab for the raw spikes. `dmfc_rsg` lets us tell the same story now.

## ⚠ This is NOT the area2_bump pipeline with a swapped DANDISET
| | area2_bump (000127) | dmfc_rsg (000130) |
|---|---|---|
| Behavioral output | **continuous 2D hand velocity** @1kHz | **one scalar `tp` (produced interval) per trial** |
| POYO readout | `cursor_velocity_2d` (drop-in) | **NEW timing-regression head** (must be built) |
| Effector | hand reaching | eye **or** hand report |
| Area | somatosensory area 2 | dorsomedial frontal cortex |

So `dmfc_rsg` is a **harder, more pointed** demo: can POYO's pretrained motor-cortex
representation transfer — via unit identification (freeze core, learn new unit/session
embeddings) — to a **cognitive-timing readout in a different area**? That transfer design
is being planned separately (see the plan on POYO→Merchant-task transfer).

## Run
```bash
brainsets prepare ./dmfc_rsg_pipeline --local \
    --raw-dir ./data/raw --processed-dir ./data/processed --cores 4
# -> ./data/processed/dmfc_rsg/<subject>_dmfc_rsg_{train,test}.h5
```
(Same pinned env as the POYO reproduction: torch_brain@ca3cfb / brainsets@4ccee58.)
The deposit is tiny (~15.7 MB, 2 assets), so a full `dandi download DANDI:000130` for
validation is trivial.

## Validate against one NWB before a real run ([V1]–[V4] in pipeline.py)
- **[V1]** `recording_tech` — DMFC_RSG is linear/V-probe, **not** Utah array.
- **[V2]** trial columns — confirm `tp` / `ts`(`theta`) / `is_eye` / `is_short` /
  `set_time` / `go_time` names against `nwbfile.trials.colnames`.
- **[V3]** eval window — Set→Go production epoch (not move-onset).
- **[V4]** `split` column present for the NLB train/val masks.

## Then (POYO side, separate)
Add a `PoyoTimingDataset` with a scalar **timing-regression** readout on `trials.tp`
(read within the Set→Go window), run unit-identification finetune from the reproduced
POYO checkpoint, and report timing-decode skill vs a from-scratch baseline + wwj/wwjd on
the weights.
