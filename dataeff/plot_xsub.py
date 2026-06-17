#!/usr/bin/env python
"""Cross-subject transfer figure: ft (frozen core) vs scratch R² on held-out monkey-t sessions.
    python plot_xsub.py xsub_all.jsonl  ->  xsub_curve.png
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "xsub_all.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
sessions = sorted({r["session"] for r in rows})
ft = [next((r["test_r2"] for r in rows if r["session"] == s and r["cond"] == "ft"), float("nan")) for s in sessions]
sc = [next((r["test_r2"] for r in rows if r["session"] == s and r["cond"] == "scratch"), float("nan")) for s in sessions]
labels = [s.split("_")[1] for s in sessions]  # date

x = np.arange(len(sessions)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - w/2, ft, w, label="POYO-1 finetune (frozen core)", color="#1f77b4")
ax.bar(x + w/2, sc, w, label="from scratch", color="#d62728")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("test R²  (reach-period hand velocity)")
ax.set_xlabel("held-out monkey-t session (never seen in pretraining)")
ax.set_title("Cross-subject transfer to a NEW animal (perich monkey t)")
ax.set_ylim(min(-0.1, min(sc) - 0.05), 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("xsub_curve.png", dpi=140)
print("wrote xsub_curve.png")
for s, f, c in zip(sessions, ft, sc):
    print(s, "ft", round(f, 3), "scratch", round(c, 3), "gap", round(f - c, 3))
print(f"mean ft {np.nanmean(ft):.3f}  mean scratch {np.nanmean(sc):.3f}  mean gap {np.nanmean(np.array(ft)-np.array(sc)):.3f}")
