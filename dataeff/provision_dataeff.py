#!/usr/bin/env python
"""Provision ONE cheap 1x H100 SECURE RunPod pod for the area2 data-efficiency curve,
upload the harness, and kick off run_dataeff.sh under nohup. Then monitor via SSH from the host.

    RUNPOD_API_KEY=$(cat ~/.runpod_key) python provision_dataeff.py            # create + upload + launch
    RUNPOD_API_KEY=$(cat ~/.runpod_key) python provision_dataeff.py --stop POD_ID   # terminate

Cost: 1x H100 SECURE ~= $2.5-4/hr; ~8 short finetunes ~= 1.5-2h => ~$5-8.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # galaxy-brain-bench


def _runpod():
    try:
        import runpod
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "runpod"], check=True)
        import runpod
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    return runpod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--stop", default="", help="terminate this pod id and exit")
    ap.add_argument("--no-launch", action="store_true", help="create+upload only, don't start training")
    args = ap.parse_args()
    runpod = _runpod()

    if args.stop:
        runpod.terminate_pod(args.stop)
        print(f"terminated {args.stop}")
        return

    pubkey = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()
    print(f"Creating 1x {args.gpu} SECURE pod...")
    pod = runpod.create_pod(
        name="poyo-dataeff", image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id=args.gpu, gpu_count=1, cloud_type="SECURE", start_ssh=True,
        ports="22/tcp", volume_in_gb=60, container_disk_in_gb=40,
        min_vcpu_count=16, min_memory_in_gb=64,
        env={"PUBLIC_KEY": pubkey}, volume_mount_path="/workspace",
    )
    pid = pod["id"]
    print(f"Pod: {pid}  dashboard: https://runpod.io/console/pods/{pid}")

    ip = port = None
    for i in range(120):
        info = runpod.get_pod(pid) or {}
        rt = info.get("runtime") or {}
        for p in (rt.get("ports") or []):
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                ip, port = p["ip"], p["publicPort"]
        if ip:
            print(f"SSH ready: root@{ip}:{port}")
            break
        print(f"  booting... ({i*10}s)")
        time.sleep(10)
    if not ip:
        sys.exit("SSH never came up; check dashboard. Pod still billing -> --stop it.")

    print(f"{pid} {ip} {port}")  # machine-readable line for the host-side driver
    with open(os.path.join(HERE, "pod.txt"), "w") as f:
        f.write(f"{pid} {ip} {port}\n")
    print("wrote pod.txt")


if __name__ == "__main__":
    main()
