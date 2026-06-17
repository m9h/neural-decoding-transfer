#!/usr/bin/env python
"""UnitDropout (augmentation) ablation figure: generalization (R²), feature-covariance spectrum
metrics, and (optional) weight-spectrum alpha, across dropout levels.
    python plot_aug.py aug_results.jsonl [alpha.json]
alpha.json (optional): {"off": <mean_alpha>, "standard": ..., "aggressive": ...}
"""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
order = ["off", "standard", "aggressive"]
rows.sort(key=lambda r: order.index(r["level"]) if r["level"] in order else 99)
levels = [r["level"] for r in rows]
r2 = [r.get("test_r2") for r in rows]
eff = [r.get("eff_rank") for r in rows]
top1 = [r.get("top1_frac") for r in rows]
alpha = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None

panels = 3 if alpha else 2
fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 4))
x = range(len(levels))

axes[0].plot(x, r2, "o-", color="#1f77b4", lw=2, ms=8)
axes[0].set_title("Generalization"); axes[0].set_ylabel("test R²"); axes[0].set_ylim(0, 1)

ax2 = axes[1]; ax2.plot(x, eff, "s-", color="#2ca02c", lw=2, ms=8, label="effective rank")
ax2.set_ylabel("feature-cov effective rank", color="#2ca02c")
ax2b = ax2.twinx(); ax2b.plot(x, top1, "^--", color="#d62728", lw=2, ms=7, label="top-1 eigval frac")
ax2b.set_ylabel("top-1 eigenvalue fraction", color="#d62728")
ax2.set_title("Feature-covariance spectrum")

if alpha:
    a = [alpha.get(l) for l in levels]
    axes[2].plot(x, a, "D-", color="#9467bd", lw=2, ms=8)
    axes[2].axhline(2.0, color="k", ls=":", lw=1, label="HT-SR α=2")
    axes[2].set_title("Weight-spectrum α (wwj)"); axes[2].set_ylabel("mean α"); axes[2].legend()

for ax in axes:
    ax.set_xticks(list(x)); ax.set_xticklabels(levels); ax.grid(alpha=0.3)
fig.suptitle("UnitDropout as implicit spectral regularization (POYO, held-out monkey t)")
fig.tight_layout()
fig.savefig("aug_curve.png", dpi=140)
print("wrote aug_curve.png")
for r in rows:
    print(r["level"], "R2", round(r.get("test_r2") or 0, 3),
          "eff_rank", round(r.get("eff_rank") or 0, 2),
          "top1", round(r.get("top1_frac") or 0, 3),
          "trace", round(r.get("trace") or 0, 2))
