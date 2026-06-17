"""Run the UNMODIFIED nerdslab/poyo repo with its EXACT recipe — faithful POYO-MP reproduction.

Disciplined reset: every prior run deviated (batch 256 not their 1024 -> wrong max_lr;
bf16; multi-GPU; NGC torch 2.12). This runs their repo + pinned env + their freshly-
re-prepped data (poyo-repo-data volume) with ZERO config overrides except epochs:
their train_poyo_mp.yaml (batch 1024) + defaults.yaml (precision 32, gpus 1, SparseLamb,
max_lr 0.032). Single H200 (141GB) so batch 1024 fits with no deviation from gpus=1.

De-risk confirm = 100 epochs. Verdict: do the weights stay ALIVE (not collapse to 0 like
our NGC-torch-2.12 runs) and does R2 trend up? If yes -> recipe is healthy -> green-light
the full 1000-epoch run.

    modal run --detach modal_poyo_repo.py --epochs 100
"""
import os
import subprocess

import modal

app = modal.App("poyo-repo-unmod")

# Their pinned env from their requirements.txt (read at build time).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("/tmp/poyo_src/requirements.txt")
    .add_local_dir("/tmp/poyo_src", "/root/poyo", copy=True)   # repo stays UNMODIFIED
)

data_vol = modal.Volume.from_name("poyo-repo-data")          # fresh brainsets@4ccee58 prep
ckpt_vol = modal.Volume.from_name("poyo-repo-ckpts", create_if_missing=True)


# confirm_full2: clean dir for the resume-FIXED rerun. The old confirm_full holds the
# schedule-corrupted run (peak relocated by the estimated_stepping_batches resume bug);
# a fresh dir ensures _latest_last_ckpt() starts from scratch (writing a correct
# onecycle_total_steps.txt) instead of resuming the corrupted checkpoints.
LOG_DIR = "/ckpts/confirm_full2"


def _latest_last_ckpt():
    """Newest last.ckpt across Lightning version_* dirs (by mtime), or None.
    Each resumed fit() writes a new version_N, so pick the most recently written."""
    import glob, os
    cks = glob.glob(f"{LOG_DIR}/lightning_logs/version_*/checkpoints/last.ckpt")
    return max(cks, key=os.path.getmtime) if cks else None


@app.function(image=image, gpu="H100:8", cpu=32,
              volumes={"/data": data_vol, "/ckpts": ckpt_vol},
              timeout=86400,   # Modal's max (24h); covers the measured ~9h full run
              retries=3)       # whole-container preemption -> reschedule; resumes from last.ckpt
def train(epochs: int):
    import torch
    n_gpu = torch.cuda.device_count()
    print(f"torch={torch.__version__} n_gpu={n_gpu} dev={torch.cuda.get_device_name(0)}", flush=True)
    import torch_brain, temporaldata
    print(f"torch_brain={torch_brain.__version__} temporaldata={temporaldata.__version__}", flush=True)

    # EXACT recipe: batch_size (GLOBAL 1024), precision (32), max_lr (0.032), optimizer,
    # LR schedule all from their committed configs. We only set gpus to the actual device
    # count (batch 1024 fp32 needs >140GB on ONE GPU -> the authors trained multi-GPU,
    # code does batch//world_size -> per-GPU 128). NO change to batch/precision/LR.
    base = [
        "python", "train.py", "--config-name", "train_poyo_mp.yaml",
        "data_root=/data/processed",
        f"log_dir={LOG_DIR}",
        f"gpus={n_gpu}",
        f"epochs={epochs}",
        "wandb.enable=false",
    ]
    # NCCL watchdog hardening: the 8xH100 deaths are HeartbeatMonitor aborts (a transiently
    # slow/stuck rank trips a short timeout and aborts the whole group). Raise the timeouts so
    # a blip recovers instead of killing the run. Infra-only -- does NOT touch the recipe.
    env = dict(
        __import__("os").environ,
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="600",   # watchdog heartbeat (default ~minutes)
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        NCCL_TIMEOUT="3600",
    )

    # SELF-HEALING resume loop. The 8xH100 runs die intermittently from a distributed
    # (NCCL/TCPStore heartbeat) fault -- one rank drops and the group aborts; nothing to do
    # with the recipe. On any non-zero exit, resume from the newest last.ckpt (Lightning
    # restores model+optimizer+epoch; total_steps is pinned via onecycle_total_steps.txt so
    # the LR schedule stays faithful). Bounded so a broken config can't spin forever.
    #
    # PERIODIC COMMIT (background thread, every 4 min): Lightning writes last.ckpt to the
    # mounted /ckpts every epoch, but those writes only PERSIST to the Modal volume on commit.
    # Without frequent commits a container-level death (Modal retries reschedules) rewinds to
    # the last infrequent background commit -- we observed a 47-epoch rewind. Lightning writes
    # checkpoints atomically (temp+rename), so committing between/over writes is safe.
    import threading, time
    MAX_RESTARTS = 20
    r = None
    for attempt in range(MAX_RESTARTS):
        ckpt = _latest_last_ckpt()
        cmd = base + ([f"ckpt_path={ckpt}"] if ckpt else [])
        print(f"[attempt {attempt}] resume_from={ckpt or 'SCRATCH'} :: {' '.join(cmd)}", flush=True)
        stop = threading.Event()
        def _committer():
            while not stop.wait(240):
                try:
                    ckpt_vol.commit(); print("[commit] periodic", flush=True)
                except Exception as e:
                    print(f"[commit] periodic failed: {e}", flush=True)
        ct = threading.Thread(target=_committer, daemon=True); ct.start()
        r = subprocess.run(cmd, cwd="/root/poyo", env=env)
        stop.set(); ct.join(timeout=10)
        ckpt_vol.commit()
        if r.returncode == 0:
            print(f"[done] clean exit after {attempt + 1} attempt(s)", flush=True)
            return {"rc": 0, "epochs": epochs, "attempts": attempt + 1}
        print(f"[attempt {attempt}] train.py exited rc={r.returncode}; resuming from last.ckpt", flush=True)
    return {"rc": r.returncode if r else -1, "epochs": epochs, "attempts": MAX_RESTARTS}


@app.local_entrypoint()
def main(epochs: int = 100):
    # spawn() (not remote()) so `modal run --detach` submits the job and the CLI EXITS
    # immediately — nothing stays tethered to interrupt. (remote() keeps the CLI streaming
    # for the whole run; killing that CLI takes the remote down with it, even under --detach.
    # That's how the first full-run attempt died at epoch 17.)
    call = train.spawn(epochs)
    print("SPAWNED call_id:", call.object_id, "epochs:", epochs)
