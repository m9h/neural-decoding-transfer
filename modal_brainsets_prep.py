"""Re-prepare perich_miller_population_2018 with the AUTHORS' pinned brainsets@4ccee58,
on Modal, into a fresh volume — eliminates the data-processing confound for the faithful
POYO-MP reproduction (their env builds from /tmp/poyo_src/requirements.txt).

    modal run --detach modal_brainsets_prep.py
"""
import os
import subprocess

import modal

app = modal.App("poyo-brainsets-prep")

# Their exact pinned env (same requirements as the training image -> layer is cached).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("/tmp/poyo_src/requirements.txt")
)
vol = modal.Volume.from_name("poyo-repo-data", create_if_missing=True)


@app.function(image=image, cpu=8.0, memory=32768, volumes={"/data": vol}, timeout=86400)
def prep():
    import brainsets, temporaldata
    print(f"brainsets={brainsets.__version__} temporaldata={temporaldata.__version__}", flush=True)
    cmd = [
        "brainsets", "prepare", "perich_miller_population_2018",
        "--raw-dir", "/data/raw",
        "--processed-dir", "/data/processed",
        "--cores", "8",
    ]
    print("run:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    vol.commit()
    pdir = "/data/processed/perich_miller_population_2018"
    n = len([f for f in os.listdir(pdir) if f.endswith(".h5")]) if os.path.isdir(pdir) else 0
    print(f"DONE rc={r.returncode} processed_h5={n}", flush=True)
    return {"rc": r.returncode, "n_h5": n}


@app.local_entrypoint()
def main():
    print("RESULT:", prep.remote())
