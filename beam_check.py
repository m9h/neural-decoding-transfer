"""Beam connectivity + multi-GPU availability check.

Run with the beam venv:  .venv-beam/bin/python beam_check.py
"""
from beam import function, Image

img = Image(python_version="python3.11").add_python_packages(["torch"])


@function(gpu="L40S", image=img)
def single():
    import torch
    return {"cuda": torch.cuda.is_available(),
            "n_gpu": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0)}


@function(gpu="L40S", gpu_count=2, image=img)
def multi():
    import torch
    return {"n_gpu": torch.cuda.device_count(),
            "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


if __name__ == "__main__":
    print("== single-GPU (connectivity) ==")
    print(single.remote())

    print("== multi-GPU (gpu_count=2 — is it enabled on this account?) ==")
    try:
        print(multi.remote())
    except Exception as e:
        print(f"MULTI-GPU UNAVAILABLE: {type(e).__name__}: {e}")
