#!/usr/bin/env python
"""Provision a DEDICATED 8x H100 RunPod pod and run the faithful POYO-MP reproduction.

Mirrors the user's brain-fwi/run_runpod.py pattern (runpod SDK + create_pod), scaled to 8 GPUs
and pointed at the unmodified nerdslab/poyo recipe. A non-preemptible pod = ONE uninterrupted
~9h run, no resume logic, faithful by construction.

    export RUNPOD_API_KEY=...                 # from https://runpod.io/console/user/settings
    python run_poyo_runpod.py --dry-run       # show config + scripts, create nothing
    python run_poyo_runpod.py                  # CREATE the pod (this is the ~$200 spend)
    python run_poyo_runpod.py --gpu "NVIDIA A100 80GB PCIe"   # cheaper/slower fallback

After it boots, SSH in and run the printed command (or it auto-runs if --autostart).
Cost: 8x H100 on-demand ~= $18-24/hr x ~9h ~= $180-220.
"""
import argparse, json, os, sys, time

# Self-contained on-pod setup: pinned env + data + train. Data is RE-PREPPED on-pod via
# brainsets (self-contained, no cross-cloud creds) -- note this adds ~1-3h of CPU download
# while the H100s idle (~$40-70). FASTER option: pre-stage the 111 prepped .h5 to S3/HTTP and
# set DATA_URL below to pull them in minutes (see README "Get the data onto the node").
SETUP_AND_RUN = r"""#!/bin/bash
set -euo pipefail
cd /workspace
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
[ -d poyo ] || git clone --depth 1 https://github.com/nerdslab/poyo.git poyo
cd poyo && uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count())"

DATA=/workspace/data/processed
if [ -n "${DATA_URL:-}" ]; then
    mkdir -p "$DATA/perich_miller_population_2018"
    curl -L "$DATA_URL" | tar xz -C "$DATA/perich_miller_population_2018"
elif [ ! -d "$DATA/perich_miller_population_2018" ]; then
    brainsets prepare perich_miller_population_2018 --raw-dir /workspace/data/raw \
        --processed-dir "$DATA" --cores 8
fi

# SINGLE uninterrupted run -- EXACT recipe, no resume hacks needed on a dedicated node.
exec python train.py --config-name train_poyo_mp.yaml \
    data_root="$DATA" log_dir=/workspace/poyo_out gpus=8 epochs=1000 wandb.enable=false
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--gpu-count", type=int, default=8)
    ap.add_argument("--disk", type=int, default=200, help="persistent volume GB (data+ckpts)")
    ap.add_argument("--data-url", default="", help="optional tar.gz of prepped perich .h5 (skips re-prep)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = {
        "name": "poyo-mp-repro",
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeId": args.gpu,
        "gpuCount": args.gpu_count,
        "volumeInGb": args.disk,
        "containerDiskInGb": 60,
        "minVcpuCount": 32,
        "minMemoryInGb": 128,
        "env": {"DATA_URL": args.data_url},
    }

    if args.dry_run:
        print("RunPod config:\n" + json.dumps(config, indent=2))
        print("\nOn-pod setup+run script:\n" + SETUP_AND_RUN)
        print("\n[dry-run] nothing created. Re-run without --dry-run to CREATE the pod (~$200).")
        return

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("Set RUNPOD_API_KEY (https://runpod.io/console/user/settings). Not creating a pod.")

    try:
        import runpod
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q runpod"); import runpod
    runpod.api_key = api_key

    # inject our public key so SSH works (RunPod pytorch image adds PUBLIC_KEY to authorized_keys)
    try:
        config["env"]["PUBLIC_KEY"] = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()
    except OSError:
        print("WARN: ~/.ssh/id_ed25519.pub not found -> SSH may be unavailable")

    print(f"Creating {args.gpu_count}x {args.gpu} pod (SECURE cloud = non-preemptible)...")
    pod = runpod.create_pod(
        name=config["name"], image_name=config["imageName"],
        gpu_type_id=config["gpuTypeId"], gpu_count=config["gpuCount"],
        cloud_type="SECURE", start_ssh=True,             # non-preemptible + ssh access
        volume_in_gb=config["volumeInGb"], container_disk_in_gb=config["containerDiskInGb"],
        min_vcpu_count=config["minVcpuCount"], min_memory_in_gb=config["minMemoryInGb"],
        env=config["env"], volume_mount_path="/workspace",
    )
    pod_id = pod["id"]
    print(f"Pod: {pod_id}  dashboard: https://runpod.io/console/pods/{pod_id}")
    for i in range(90):
        rt = (runpod.get_pod(pod_id) or {}).get("runtime") or {}
        if rt.get("uptimeInSeconds", 0) > 0:
            print(f"Running (uptime {rt['uptimeInSeconds']}s)"); break
        print(f"  booting... ({i*10}s)"); time.sleep(10)

    # write the setup script to a local file the user scps, and print the run command
    with open("poyo_pod_setup.sh", "w") as f:
        f.write(SETUP_AND_RUN)
    print("\nWrote poyo_pod_setup.sh. To launch training:")
    print(f"  scp poyo_pod_setup.sh root@{pod_id}-ssh.runpod.io:/workspace/")
    print(f"  ssh root@{pod_id}-ssh.runpod.io 'cd /workspace && bash poyo_pod_setup.sh 2>&1 | tee repro.log'")
    print("Watch repro.log for average_val_metric (climbs in the back half toward ~0.9).")
    print(f"!! Pod bills until stopped: runpod.io/console/pods or `runpod stop pod {pod_id}`")


if __name__ == "__main__":
    main()
