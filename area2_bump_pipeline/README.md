# area2_bump — local brainsets pipeline (POYO new-dataset demo)

Onboards **DANDI:000127** (`Area2_Bump`: macaque somatosensory **area 2** spiking
during center-out reaching with mechanical bump perturbations; Neural Latents
Benchmark '21, Chowdhury–Glaser–Miller) into the POYO/brainsets format.

Chosen as the post-reproduction demo because it's **domain-proximal but new**:
- Decoding target = 2D **hand velocity** → maps directly onto POYO's `cursor_velocity_2d` readout.
- Units are **new** (area 2, not M1) and the animal/session are new → exercises POYO **unit identification** (freeze core, learn fresh unit + session embeddings).
- **Not** in POYO-1's training set (POYO trained on Perich/Churchland/Makin/Flint/NLB-Maze/NLB-RTT).

Adapted from the `pei_pandarinath_nlb_2021` (mc_maze, DANDI:000140) pipeline — same NLB NWB schema.

## Run
```bash
brainsets prepare ./area2_bump_pipeline --local \
    --raw-dir ./data/raw --processed-dir ./data/processed --cores 8
# -> ./data/processed/area2_bump/han_area2_bump_{train,test}.h5
```
(Use the same pinned env as the POYO reproduction: torch_brain@ca3cfb / brainsets@4ccee58.)

## Validate against one NWB before a real run (markers `[V1]`/`[V2]`/`[V3]` in pipeline.py)
- **[V1]** eye tracking may be absent in area2_bump — handled (eye is optional).
- **[V2]** confirm the official NLB area2_bump eval window (placeholder = mc_maze's move-onset [-0.05, +0.65]).
- **[V3]** active vs passive (bump) trials are distinguished by a trial column (e.g. `ctr_hold_bump`); it carries over automatically and can later weight the readout.

Quick check: `dandi download DANDI:000127` (or stream one asset) and inspect
`nwbfile.processing["behavior"]` keys + `nwbfile.trials.colnames`.

## Then (POYO side, separate)
Add a `PoyoArea2Dataset` (mirror `poyo_datasets/poyo_mp.py`) with a `cursor_velocity_2d`
readout on `hand.vel`, run unit-identification + finetune from the reproduced POYO
checkpoint, and report R² vs a from-scratch baseline + wwj/wwjd on the weights.
