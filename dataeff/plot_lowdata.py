#!/usr/bin/env python
"""Low-data generalization grid: test R² vs UnitDropout level, one line per train fraction
(does augmentation open a generalization gap when data is scarce?), plus the feature-cov
condition number as the spectral-regularization correlate.
    python plot_lowdata.py aug_lowdata_results.jsonl  ->  lowdata_curve.png
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
levels = ["off", "standard", "aggressive"]
fracs = sorted({r["frac"] for r in rows})
def get(frac, level, key):
    return next((r.get(key) for r in rows if r["frac"] == frac and r["level"] == level), None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
x = range(len(levels))
colors = {0.1: "#d62728", 0.25: "#ff7f0e", 1.0: "#1f77b4"}
for fr in fracs:
    r2 = [get(fr, l, "test_r2") for l in levels]
    ax1.plot(x, r2, "o-", lw=2, ms=8, color=colors.get(fr, None), label=f"{int(fr*100)}% data")
    cond = [get(fr, l, "cond") for l in levels]
    ax2.plot(x, cond, "s--", lw=2, ms=7, color=colors.get(fr, None), label=f"{int(fr*100)}% data")
ax1.set_title("Generalization vs augmentation"); ax1.set_ylabel("test R²")
ax2.set_title("Feature-cov conditioning vs augmentation"); ax2.set_ylabel("condition number"); ax2.set_yscale("log")
for ax in (ax1, ax2):
    ax.set_xticks(list(x)); ax.set_xticklabels(levels); ax.set_xlabel("UnitDropout"); ax.grid(alpha=0.3); ax.legend()
fig.suptitle("Does UnitDropout's spectral regularization buy generalization at low data? (POYO, finetune-all, monkey t)")
fig.tight_layout()
fig.savefig("lowdata_curve.png", dpi=140)
print("wrote lowdata_curve.png")
for fr in fracs:
    for l in levels:
        print(f"frac={fr} {l}: R2={get(fr,l,'test_r2')}  cond={get(fr,l,'cond')}  eff_rank={get(fr,l,'eff_rank')}")
