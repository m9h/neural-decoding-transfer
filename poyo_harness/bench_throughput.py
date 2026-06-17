#!/usr/bin/env python
"""Measure real POYO training throughput on the GB10.

Uses the actual harness pieces (dataset + RandomFixedWindowSampler + workers +
model.tokenize + collate + 11.8M model + SparseLamb + bf16 fwd/bwd/step) so the
samples/sec it reports is the realistic per-GPU training rate, not a synthetic
matmul. That rate is the basis for projecting to Modal H100s.

    python bench_throughput.py <config-name> [batch_sizes=32,64,128] [steps=20]
"""
import sys, time, json
import torch, hydra
from hydra import compose, initialize
from torch.utils.data import DataLoader

from train import DataModule  # faithful dataset/readout construction
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler

torch.set_float32_matmul_precision("medium")

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "train_poyo_mp"
BATCHES = [int(x) for x in (sys.argv[2].split("=")[1] if len(sys.argv) > 2 else "32,64,128").split(",")]
STEPS = int(sys.argv[3].split("=")[1]) if len(sys.argv) > 3 else 20
WARMUP = 5
NW = 6


def move(x):
    return x.cuda(non_blocking=True) if torch.is_tensor(x) else x


def main():
    t0 = time.time()
    with initialize(version_base="1.3", config_path="configs"):
        cfg = compose(config_name=CONFIG, overrides=["wandb.enable=false", "num_workers=%d" % NW])
    dm = DataModule(cfg)
    spec = dm.readout_spec
    init_t = time.time() - t0

    model = hydra.utils.instantiate(cfg.model, readout_spec=spec)
    dm.link_model(model)  # appends model.tokenize to dataset transform + inits vocab
    model = model.cuda()

    sampler = RandomFixedWindowSampler(
        sampling_intervals=dm.train_dataset.get_sampling_intervals("train"),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(1),
    )
    samples_per_epoch = len(sampler)

    # SparseLamb exactly as train.py.configure_optimizers
    from torch_brain.optim import SparseLamb
    special = list(model.unit_emb.parameters()) + list(model.session_emb.parameters())
    rest = [p for n, p in model.named_parameters() if "unit_emb" not in n and "session_emb" not in n]
    opt = SparseLamb([{"params": special, "sparse": True}, {"params": rest}], lr=1e-4, weight_decay=1e-4)

    print(f"config={CONFIG} model_params={sum(p.numel() for p in model.parameters()):,} "
          f"recordings={len(dm.train_dataset.recording_ids)} units={len(dm.train_dataset.get_unit_ids()):,}")
    print(f"dataset_init={init_t:.1f}s  samples/epoch={samples_per_epoch:,}  seq_len={model.sequence_length}s")
    print(f"{'batch':>6} {'samp/s':>9} {'steps/s':>8} {'s/epoch':>9} {'peakGB':>7}")

    results = {"config": CONFIG, "samples_per_epoch": samples_per_epoch, "by_batch": {}}
    for B in BATCHES:
        loader = DataLoader(dm.train_dataset, sampler=sampler, collate_fn=collate, batch_size=B,
                            num_workers=NW, drop_last=True, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)
        it = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(WARMUP + STEPS):
                if i == WARMUP:
                    torch.cuda.synchronize(); t = time.time()
                batch = next(it)
                mi = {k: move(v) for k, v in batch["model_inputs"].items()}
                tgt = move(batch["target_values"]); w = move(batch["target_weights"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**mi)
                m = mi["output_mask"]
                loss = spec.loss_fn(out[m], tgt[m], w[m])
                loss.backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            dt = time.time() - t
            sps = STEPS * B / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{B:>6} {sps:>9.1f} {STEPS/dt:>8.2f} {samples_per_epoch/sps:>9.1f} {peak:>7.1f}")
            results["by_batch"][B] = {"samples_per_sec": sps, "sec_per_epoch": samples_per_epoch/sps, "peak_gb": peak}
        except RuntimeError as e:
            print(f"{B:>6}  OOM/err: {str(e)[:60]}")
            results["by_batch"][B] = {"error": str(e)[:120]}
            break
        finally:
            del loader, it
            torch.cuda.empty_cache()

    with open(f"/workspace/logs/bench_{CONFIG}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
