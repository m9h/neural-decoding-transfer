"""Real POYO training on Modal — 8x L40S DDP, running the vendored harness unchanged.

Beam's multi-GPU is waitlisted, so Modal (confirmed 8-GPU) is the path. This runs
poyo_harness/train.py via Lightning's native DDP: we invoke `python train.py` in a
subprocess inside an 8-GPU container, and Lightning's SubprocessScriptLauncher fans
out the 8 ranks exactly as it would from a normal CLI (the tested path for
lightning+hydra — avoids the torchrun/elastic-env edge cases).

The thing this must prove on the FIRST (shakedown) run: SparseLamb's *sparse*
embedding gradients surviving DDP all-reduce. That's the flagged correctness/eff
risk; a 2-epoch run surfaces it for ~$1 before the ~$182 full run.

  # shakedown (cheap, proves 8-GPU DDP + sparse-embed path):
  modal run modal_train.py --epochs 2 --batch-size 256
  # full run:
  modal run modal_train.py --epochs 1000 --batch-size 1024

Checkpoints land on the poyo-mp-ckpts volume under /<run_name>/ (committed at end;
for the long run, mid-flight crashes lose checkpoints — acceptable for run #1).
"""
import subprocess

import modal

GPU = "L40S:8"          # 8x L40S in one container; Lightning DDP across them
CPU = 32                # dataloader headroom across 8 ranks
DATA_MOUNT = "/data/datasets/brainsets/processed"
CKPT_MOUNT = "/ckpts"

app = modal.App("poyo-mp-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torch_brain==0.1.2", "brainsets==0.2.2", "temporaldata==0.1.6",
        "lightning", "hydra-core", "omegaconf", "torchmetrics", "einops",
    )
    # bake the harness in (copy=True) so the DDP subprocess can exec it
    .add_local_dir("poyo_harness", "/root/poyo_harness", copy=True)
)

data_vol = modal.Volume.from_name("poyo-mp-train", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("poyo-mp-ckpts", create_if_missing=True)


@app.function(image=image, gpu=GPU, cpu=CPU,
              volumes={DATA_MOUNT: data_vol, CKPT_MOUNT: ckpt_vol},
              timeout=86400)  # 24h cap covers the ~12h full run
def train(run_name: str, epochs: int, batch_size: int, num_workers: int = 4, extra: str = ""):
    import torch

    n_gpu = torch.cuda.device_count()
    print(f"=== {run_name}: {n_gpu}x {torch.cuda.get_device_name(0)} | "
          f"epochs={epochs} batch_size={batch_size} ===", flush=True)

    cmd = [
        "python", "train.py", "--config-name", "train_poyo_mp",
        f"data_root={DATA_MOUNT}",
        f"log_dir={CKPT_MOUNT}/{run_name}",
        f"gpus={n_gpu}",
        f"epochs={epochs}",
        f"batch_size={batch_size}",
        f"num_workers={num_workers}",
        "wandb.enable=false",
    ] + [o for o in extra.split() if o]   # extra hydra overrides, e.g. "precision=bf16-mixed"
    print("launch:", " ".join(cmd), flush=True)
    # stream child output; raise on failure
    r = subprocess.run(cmd, cwd="/root/poyo_harness")
    ckpt_vol.commit()  # persist whatever checkpoints exist
    if r.returncode != 0:
        raise RuntimeError(f"training exited {r.returncode}")
    return {"run_name": run_name, "n_gpu": n_gpu, "epochs": epochs, "status": "ok"}


@app.local_entrypoint()
def main(epochs: int = 2, batch_size: int = 256, num_workers: int = 4, extra: str = "",
         tag: str = "", gpu: str = GPU, cpu: float = CPU):
    run_name = f"poyo_mp_e{epochs}_b{batch_size}" + (f"_{tag}" if tag else "")
    res = train.with_options(gpu=gpu, cpu=cpu).remote(run_name, epochs, batch_size, num_workers, extra)
    print("RESULT:", res)
