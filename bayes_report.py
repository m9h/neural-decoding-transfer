#!/usr/bin/env python
"""Bayesian (wwjd) HT-SR alpha report for a trained POYO model — the calibrated
sibling of alpha_report.py. Where alpha_report gives a frequentist point alpha +
bootstrap CI + Vuong LRT, this reports the wwjd posterior: BMA posterior mean of
alpha with a credible interval, posterior model probability (is the tail really a
power law?), P(alpha in ROPE around 2), and an optional posterior-predictive p-value.

Implements the `wwj-spectral` skill contract: vetted methods only (everything comes
from wwj.bayes), correctness over efficiency (Bayesian path), and — the headline —
NO SILENT DATA REDUCTION: every walked matrix is accounted for as fit XOR degenerate,
degenerate (near-dead/low-rank) layers are retained and flagged, and the population
mean is over FIT layers only so one dead matrix can't nan-poison the aggregate.

Runs in the wwj venv (JAX + CPU torch), NOT the galaxy-brain container:
    /home/mhough/dev/wwj/.venv/bin/python bayes_report.py <ckpt> [--out report.json]
"""
import argparse
import json
import math
import sys

import jax.numpy as jnp
import wwj
from wwj.core import _eigvals, _n_meaningful
from wwj.bayes import bayes_analyze_matrix, prob_in_rope

# Reuse alpha_report's verified loader + embedding policy so the two reports are
# strictly comparable (same min_dim filter, same unit_emb/session_emb skip, same
# 'model.' strip). Imported lazily in main() so build_report stays torch-free/testable.


def build_report(mats, min_tail_size: int = 50, ppc: bool = False,
                 rope_delta: float = 0.1, key=None) -> dict:
    """Pure core: a list of (name, jnp 2D matrix) -> the bounded report dict.

    A layer is DEGENERATE (no fittable heavy tail) iff it has fewer than min_tail_size
    MEANINGFUL eigenvalues (core's own `_n_meaningful` gate — the Bayesian path does NOT
    apply this guard itself and will otherwise fit a spurious finite alpha to near-zero
    eigenvalues of a dead/low-rank matrix), or bayes_analyze_matrix raises, or returns a
    non-finite alpha_mean. Degenerate layers are kept in `layers` (flagged) and listed in
    `degenerate_layers`; `summary` aggregates over FIT layers only.
    """
    layers, degenerate_layers, fit = [], [], []
    for name, W in mats:
        rec = {"layer": name, "shape": list(W.shape)}
        eigs = _eigvals(W)
        n_meaningful = int(_n_meaningful(eigs))
        rec["n_meaningful"] = n_meaningful
        bs = None
        if n_meaningful < min_tail_size:            # core's vetted degeneracy gate
            is_degen = True
        else:
            try:
                bs = bayes_analyze_matrix(W, name=name, min_tail_size=min_tail_size,
                                          ppc=ppc, key=key)
                is_degen = not math.isfinite(bs.alpha_mean)
            except Exception as e:                  # pathological matrix -> degenerate, not dropped
                is_degen, rec["error"] = True, repr(e)

        if is_degen or bs is None:
            rec.update({"degenerate": True, "alpha_mean": float("nan"),
                        "ci_low": float("nan"), "ci_high": float("nan"),
                        "best_model": (bs.best_model if bs else None),
                        "tail_size": (bs.tail_size if bs else 0)})
            degenerate_layers.append(name)
        else:
            rope = prob_in_rope(eigs, center=2.0, delta=rope_delta)["prob_in_rope"]
            rec.update({
                "degenerate": False,
                "alpha_mean": float(bs.alpha_mean), "alpha_std": float(bs.alpha_std),
                "ci_low": float(bs.ci_low), "ci_high": float(bs.ci_high),
                "p_alpha_lt_2": float(bs.p_alpha_lt_2),
                "best_model": bs.best_model, "prob_powerlaw": float(bs.prob_powerlaw),
                "prob_in_rope": float(rope), "tail_size": int(bs.tail_size),
                "ppc_pvalue": (None if bs.ppc_pvalue is None else float(bs.ppc_pvalue)),
                # PL-supported = power-law is the best model AND (if computed) PPC doesn't reject
                "pl_supported": bool(bs.best_model == "powerlaw"
                                     and (bs.ppc_pvalue is None or bs.ppc_pvalue > 0.05)),
            })
            fit.append(rec)
        layers.append(rec)

    n = len(mats)
    acc = {"walked": n, "degenerate": len(degenerate_layers), "fit": len(fit)}
    if fit:
        a = jnp.array([r["alpha_mean"] for r in fit])
        summary = {
            "n_layers": n, "n_fit": len(fit), "n_degenerate": len(degenerate_layers),
            "mean_alpha": float(jnp.mean(a)),
            "median_alpha": float(jnp.median(a)),
            "alpha_dist_mean": float(jnp.mean(jnp.abs(a - 2.0))),     # mean |alpha - 2|
            "mean_prob_powerlaw": float(sum(r["prob_powerlaw"] for r in fit) / len(fit)),
            "frac_pl_supported": float(sum(r["pl_supported"] for r in fit) / len(fit)),
            "frac_in_rope": float(sum(r["prob_in_rope"] for r in fit) / len(fit)),
        }
    else:
        summary = {"n_layers": n, "n_fit": 0, "n_degenerate": len(degenerate_layers),
                   "mean_alpha": float("nan")}
    return {"accounting": acc, "summary": summary, "layers": layers,
            "degenerate_layers": degenerate_layers,
            "params": {"min_tail_size": min_tail_size, "ppc": ppc, "rope_delta": rope_delta}}


def _print_report(rep, ckpt):
    p = rep["params"]; acc = rep["accounting"]; s = rep["summary"]
    print(f"wwj method=bayes(wwjd)  min_tail_size={p['min_tail_size']}  ppc={p['ppc']}  "
          f"rope=2±{p['rope_delta']}  ckpt={ckpt}")
    print(f"accounting: walked={acc['walked']}  degenerate={acc['degenerate']}  fit={acc['fit']}")
    print(f"{'layer':<46}{'alpha':>7}{'95% CrI':>16}{'P(PL)':>7}{'PPC p':>7}{'P(ROPE)':>8}")
    print("-" * 91)
    for r in rep["layers"]:
        if r["degenerate"]:
            print(f"{r['layer']:<46}{'DEGEN':>7}{'(no fittable tail)':>16}{'':>7}{'':>7}{'':>8}")
        else:
            ci = f"[{r['ci_low']:.2f},{r['ci_high']:.2f}]"
            ppcv = "-" if r["ppc_pvalue"] is None else f"{r['ppc_pvalue']:.2f}"
            print(f"{r['layer']:<46}{r['alpha_mean']:>7.2f}{ci:>16}"
                  f"{r['prob_powerlaw']:>7.2f}{ppcv:>7}{r['prob_in_rope']:>8.2f}")
    print("-" * 91)
    if s["n_fit"]:
        print(f"mean_alpha={s['mean_alpha']:.3f}  |alpha-2|={s['alpha_dist_mean']:.3f}  "
              f"PL-supported={s['frac_pl_supported']:.0%}  in-ROPE(2±{p['rope_delta']})="
              f"{s['frac_in_rope']:.0%}  over {s['n_fit']} fit layers "
              f"({s['n_degenerate']} degenerate retained, {s['n_layers']} walked)")
    else:
        print(f"no fittable layers ({s['n_degenerate']} degenerate, {s['n_layers']} walked)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--min-dim", type=int, default=50)
    ap.add_argument("--min-tail-size", type=int, default=50)
    ap.add_argument("--ppc", action="store_true", help="posterior-predictive p-value (samples; slower)")
    ap.add_argument("--rope-delta", type=float, default=0.1)
    args = ap.parse_args()

    from alpha_report import load_state_dict, weight_matrices   # verified loader + embedding policy
    sd = load_state_dict(args.ckpt)
    mats = weight_matrices(sd, args.min_dim)
    if not mats:
        sys.exit(f"No 2D weight matrices with min dim >= {args.min_dim} in {args.ckpt}")

    rep = build_report(mats, min_tail_size=args.min_tail_size, ppc=args.ppc,
                       rope_delta=args.rope_delta)
    _print_report(rep, args.ckpt)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, **rep}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
