"""Clean Beam check — minimal sync. Tests connectivity + multi-GPU gating on A10G
(more available than L40S right now), plus a single L40S to gauge capacity."""
from beam import function, Image

img = Image(python_version="python3.11").add_python_packages(["torch"])

@function(gpu="A10G", image=img)
def a10g_single():
    import torch
    return {"n_gpu": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0)}

@function(gpu="A10G", gpu_count=2, image=img)
def a10g_multi():
    import torch
    return {"n_gpu": torch.cuda.device_count(),
            "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

@function(gpu="L40S", image=img)
def l40s_single():
    import torch
    return {"n_gpu": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0)}

if __name__ == "__main__":
    for label, fn in [("A10G single", a10g_single), ("A10G gpu_count=2 (GATING TEST)", a10g_multi), ("L40S single (CAPACITY)", l40s_single)]:
        print(f"== {label} ==")
        try:
            print("  ->", fn.remote())
        except Exception as e:
            print(f"  -> FAIL: {type(e).__name__}: {str(e)[:160]}")
