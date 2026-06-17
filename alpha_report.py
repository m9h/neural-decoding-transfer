#!/usr/bin/env python
"""HTSR alpha (heavy-tailed self-regularization) report for a trained POYO model.

Complements the harness's R2 decoding metric with a *data-free* quality score:
per-layer power-law exponent alpha of the weight-matrix eigenvalue spectrum.
Well-trained layers sit near alpha~2 (Martin & Mahoney HTSR); alpha>>2 or a
failed power-law fit flags under-trained / undertuned layers.

Runs in the wwj venv (JAX + a CPU torch), NOT the galaxy-brain container:
    /home/mhough/dev/wwj/.venv/bin/python alpha_report.py <ckpt> [--out report.json]

<ckpt> is a Lightning checkpoint from the POYO harness (logs/.../*.ckpt or last.ckpt).
"""
import argparse
import json
import sys

import torch
import jax.numpy as jnp
import wwj
from wwj.core import _eigvals


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Lightning wraps weights under "state_dict"; raw torch.save may be flat or "model".
    for key in ("state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def weight_matrices(sd, min_dim=50):
    """2D weight matrices large enough for a meaningful spectrum.

    Strips a leading 'model.' (Lightning TrainWrapper.model) for readable names.
    Embedding tables (unit_emb/session_emb) are data-dependent vocab, not learned
    transforms — skip them so the score reflects the transformer weights.
    """
    out = []
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.ndim != 2 or min(v.shape) < min_dim:
            continue
        if "unit_emb" in k or "session_emb" in k:
            continue
        name = k[len("model."):] if k.startswith("model.") else k
        out.append((name, jnp.asarray(v.float().numpy())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--min-dim", type=int, default=50)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    sd = load_state_dict(args.ckpt)
    mats = weight_matrices(sd, args.min_dim)
    if not mats:
        sys.exit(f"No 2D weight matrices with min dim >= {args.min_dim} found in {args.ckpt}")

    rows = []
    print(f"{'layer':<48} {'alpha':>6} {'95% CI':>16} {'PL-valid':>9}")
    print("-" * 82)
    for name, W in mats:
        eigs = _eigvals(W)
        boot = wwj.bootstrap_alpha_ci(eigs, n_bootstrap=args.n_bootstrap, ci=0.95)
        fit = wwj.fit_distributions(eigs)
        pl_valid = bool(fit["pl_vs_exp_lrt"] > 0 and fit["pl_vs_ln_lrt"] > 0)
        rows.append({
            "layer": name,
            "shape": list(W.shape),
            "alpha": float(boot["alpha"]),
            "ci_low": float(boot["ci_low"]),
            "ci_high": float(boot["ci_high"]),
            "pl_valid": pl_valid,
        })
        ci = f"[{boot['ci_low']:.2f},{boot['ci_high']:.2f}]"
        print(f"{name:<48} {boot['alpha']:>6.2f} {ci:>16} {str(pl_valid):>9}")

    import math
    # Degenerate (near-dead / low-rank) layers come back as nan alpha — exclude
    # them from the spectral stats and report them as their own count, rather
    # than letting one dead layer nan the whole mean.
    fin = [r["alpha"] for r in rows if math.isfinite(r["alpha"])]
    n_dead = len(rows) - len(fin)
    summary = {
        "n_layers": len(rows),
        "n_degenerate": n_dead,
        "n_fit": len(fin),
        "mean_alpha": (sum(fin) / len(fin)) if fin else float("nan"),
        # fractions over the FITTABLE layers (the degenerate ones have no alpha)
        "frac_pl_valid": (sum(r["pl_valid"] for r in rows) / len(fin)) if fin else 0.0,
        # HTSR rule of thumb: well-trained layers in [2, 6]; >6 = under-trained.
        "frac_well_trained": (sum(2.0 <= a <= 6.0 for a in fin) / len(fin)) if fin else 0.0,
    }
    print("-" * 82)
    print(f"mean_alpha={summary['mean_alpha']:.3f}  "
          f"PL-valid={summary['frac_pl_valid']:.0%}  "
          f"well-trained(2-6)={summary['frac_well_trained']:.0%}  "
          f"over {summary['n_fit']} fittable layers "
          f"({n_dead} degenerate/dead excluded, {summary['n_layers']} total)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "summary": summary, "layers": rows}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
