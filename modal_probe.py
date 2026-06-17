"""Modal throughput probe for POYO — measures per-GPU samples/sec across GPU
types, then projects full multi-GPU training time + cost.

WHY this exists: my local estimate of "GB10 -> cloud" speed carried one big
unknown — the per-GPU throughput ratio of a datacenter GPU vs the GB10. This
probe MEASURES it on the same workload, on each GPU type, so the projection
stops being a guess.

WHY it sweeps types (not just H100): POYO is tiny (~13-15M params, 12-17GB at
batch 128). It cannot saturate an H100, so the H100's compute/HBM edge is largely
wasted while you pay ~2x the hourly rate. A100 (Ampere, older) or L40S (Ada) often
win on $/result for small models — this measures whether that's true here.

Note: Modal nodes are x86_64 with standard CUDA GPUs (A100=sm80, L40S=sm89,
H100=sm90), so this uses ordinary PyPI torch — NONE of the sm_121/NGC handling the
Spark's GB10 needs. Different machine, different rules.

--- setup (once) ---
  modal volume create poyo-probe-data
  # a few light (perich) + a couple heavy (odoherty) sessions are enough for
  # throughput; samples/sec doesn't need the full epoch.
  modal volume put poyo-probe-data \
    /data/datasets/brainsets/processed/perich_miller_population_2018/c_20131003_center_out_reaching.h5 \
    /perich_miller_population_2018/c_20131003_center_out_reaching.h5
  # ...repeat for ~3 perich + ~2 odoherty (odoherty files are ~hundreds of MB each)

--- run ---
  modal run modal_probe.py                                  # A100-80GB, H100, L40S
  modal run modal_probe.py --gpus "H100,H200,A100-80GB"
"""
import time

import modal

# ---- measured on the GB10 (batch 128), from bench_throughput.py. These anchor
# the projection: cloud ratio = cloud_samples_per_sec / gb10 value below. ----
GB10 = {
    "light": {"samp_s": 128.3, "samples_per_epoch": 99_644},   # POYO-mp baseline
    "heavy": {"samp_s": 57.2, "samples_per_epoch": 185_082},   # POYO-1 baseline (odoherty = heaviest component; conservative)
}
EPOCHS = 1000          # configs' default; the dominant lever — scales linearly
DDP_EFF = 0.85         # assumed strong-scaling efficiency at 8 GPUs (sparse-embed grads may lower it)
N_GPUS = 8

# Approximate Modal on-demand $/GPU-hr — VERIFY against current modal.com pricing.
COST_PER_GPU_HR = {
    "A100-40GB": 2.10, "A100-80GB": 2.50, "L40S": 1.95,
    "H100": 3.95, "H200": 4.54, "A10G": 1.10, "L4": 0.80,
}

# 11.8M POYO (configs/model/poyo_11.8M.yaml), inlined so the probe is self-contained.
POYO_HP = dict(
    sequence_length=1.0, latent_step=0.125, num_latents_per_step=32,
    dim=128, dim_head=64, depth=24, cross_heads=4, self_heads=8,
    ffn_dropout=0.2, lin_dropout=0.4, atn_dropout=0.2,
)

PROFILE_DIR = {
    "light": "perich_miller_population_2018",
    "heavy": "odoherty_sabes_nonhuman_2017",
}

app = modal.App("poyo-throughput-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",  # default PyPI cuda wheels — fine on A100/H100/L40S
        "torch_brain==0.1.2", "brainsets==0.2.2", "temporaldata==0.1.6",  # match the .h5 writer version
        "lightning", "hydra-core", "omegaconf", "torchmetrics", "einops",
    )
)

vol = modal.Volume.from_name("poyo-probe-data", create_if_missing=True)
DATA_ROOT = "/data/datasets/brainsets/processed"


@app.function(image=image, volumes={DATA_ROOT: vol}, timeout=1800)
def probe(profile: str, batch: int = 128, steps: int = 20, warmup: int = 5, max_recordings: int = 3):
    """Time `steps` real training steps (data->tokenize->bf16 fwd/bwd->SparseLamb)
    for one profile on whatever GPU this function is scheduled on. Returns samp/s."""
    import os
    from copy import deepcopy

    import torch
    import torchmetrics
    from torch.utils.data import DataLoader
    from torch_brain.models.poyo import POYO
    from torch_brain.registry import MODALITY_REGISTRY
    from torch_brain.optim import SparseLamb
    from torch_brain.data import collate
    from torch_brain.data.sampler import RandomFixedWindowSampler
    from torch_brain.transforms import Compose
    from temporaldata import Data

    torch.set_float32_matmul_precision("medium")
    dev_name = torch.cuda.get_device_name(0)

    ddir = os.path.join(DATA_ROOT, PROFILE_DIR[profile])
    rec_ids = sorted(f[:-3] for f in os.listdir(ddir) if f.endswith(".h5"))[:max_recordings]
    if not rec_ids:
        raise RuntimeError(f"No .h5 files in {ddir} — upload some to the poyo-probe-data volume.")

    # Pick the right brainset class for the profile; attach a cursor_velocity_2d
    # readout (compute is identical regardless of the normalization constants).
    from brainsets.datasets import PerichMillerPopulation2018, OdohertySabesNonhuman2017
    BASE = {"light": PerichMillerPopulation2018, "heavy": OdohertySabesNonhuman2017}[profile]
    READOUT = {"readout": {"readout_id": "cursor_velocity_2d", "normalize_mean": 0.0,
                           "normalize_std": 200.0, "metrics": [{"metric": torchmetrics.R2Score()}]}}

    class ProbeDS(BASE):
        def get_recording_hook(self, data: Data):
            data.config = deepcopy(READOUT)
            return super().get_recording_hook(data)

    ds = ProbeDS(DATA_ROOT, recording_ids=rec_ids)
    spec = MODALITY_REGISTRY["cursor_velocity_2d"]
    model = POYO(readout_spec=spec, **POYO_HP).cuda()
    model.unit_emb.initialize_vocab(ds.get_unit_ids())
    model.session_emb.initialize_vocab(ds.recording_ids)
    ds.transform = Compose([model.tokenize])

    sampler = RandomFixedWindowSampler(
        sampling_intervals=ds.get_sampling_intervals("train"),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(1),
    )
    special = list(model.unit_emb.parameters()) + list(model.session_emb.parameters())
    rest = [p for n, p in model.named_parameters() if "unit_emb" not in n and "session_emb" not in n]
    opt = SparseLamb([{"params": special, "sparse": True}, {"params": rest}], lr=1e-4, weight_decay=1e-4)

    loader = DataLoader(ds, sampler=sampler, collate_fn=collate, batch_size=batch,
                        num_workers=6, drop_last=True, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)

    def cycle(dl):  # infinite stream — probe times compute, so reusing windows is fine
        while True:
            for b in dl:
                yield b
    it = cycle(loader)
    mv = lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x
    torch.cuda.reset_peak_memory_stats()
    t = None
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(); t = time.time()
        batch_data = next(it)
        mi = {k: mv(v) for k, v in batch_data["model_inputs"].items()}
        tgt, w = mv(batch_data["target_values"]), mv(batch_data["target_weights"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**mi)
        m = mi["output_mask"]
        loss = spec.loss_fn(out[m], tgt[m], w[m])
        loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    dt = time.time() - t
    return {
        "profile": profile, "device": dev_name, "batch": batch,
        "samples_per_sec": steps * batch / dt,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "recordings_used": len(rec_ids),
    }


@app.local_entrypoint()
def main(gpus: str = "A100-80GB,H100,L40S", batch: int = 128, steps: int = 20):
    gpu_list = [g.strip() for g in gpus.split(",")]
    profiles = ["light", "heavy"]

    # fan out: one remote call per (gpu, profile)
    handles = {}
    for g in gpu_list:
        for p in profiles:
            handles[(g, p)] = probe.with_options(gpu=g).spawn(p, batch, steps)
    results = {k: h.get() for k, h in handles.items()}

    label = {"light": "POYO-mp", "heavy": "POYO-1 (odoherty proxy)"}
    for p in profiles:
        base = GB10[p]
        spe = base["samples_per_epoch"]
        print(f"\n=== {label[p]}  (global batch {N_GPUS*batch}, {N_GPUS}x GPUs, {EPOCHS} epochs) ===")
        print(f"  GB10 baseline: {base['samp_s']:.1f} samp/s @batch{batch}  "
              f"->  single-GB10 full job {spe*EPOCHS/base['samp_s']/86400:.1f} days")
        print(f"  {'GPU':<12} {'samp/s':>8} {'vsGB10':>7} {'peakGB':>7} {'job(8x)':>9} {'cost(8x)':>10}")
        for g in gpu_list:
            r = results[(g, p)]
            sps = r["samples_per_sec"]
            agg = sps * N_GPUS * DDP_EFF
            job_h = spe * EPOCHS / agg / 3600
            cost = job_h * N_GPUS * COST_PER_GPU_HR.get(g, float("nan"))
            print(f"  {g:<12} {sps:>8.1f} {sps/base['samp_s']:>6.2f}x {r['peak_gb']:>7.1f} "
                  f"{job_h:>8.1f}h ${cost:>8.0f}")
    print("\nratios are MEASURED; cost uses approximate $/GPU-hr in COST_PER_GPU_HR (verify).")
    print("scale times linearly for fewer epochs (1000 is just the config default).")
