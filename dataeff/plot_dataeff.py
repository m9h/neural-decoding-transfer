#!/usr/bin/env python
"""Plot the data-efficiency curve from dataeff_results.jsonl pulled back from the pod.
    python plot_dataeff.py dataeff_results.jsonl  ->  dataeff_curve.png
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "dataeff_results.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
by = {"ft": {}, "scratch": {}}
for r in rows:
    by[r["cond"]][r["frac"]] = r["test_r2"]

fig, ax = plt.subplots(figsize=(6, 4.2))
for cond, label, style in [("ft", "POYO-1 finetune (frozen core)", "o-"),
                           ("scratch", "from scratch", "s--")]:
    fr = sorted(by[cond])
    ax.plot([f * 100 for f in fr], [by[cond][f] for f in fr], style, label=label, lw=2, ms=7)
ax.set_xlabel("new-session training data (% of session)")
ax.set_ylabel("test R²  (hand velocity)")
ax.set_title("Data efficiency: pretrained POYO-1 vs from-scratch (area2_bump)")
ax.axhline(0, color="k", lw=0.5)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("dataeff_curve.png", dpi=140)
print("wrote dataeff_curve.png")
for cond in ("ft", "scratch"):
    print(cond, {f: round(by[cond][f], 3) for f in sorted(by[cond])})
