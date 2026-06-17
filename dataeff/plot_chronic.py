#!/usr/bin/env python
"""Plot frozen POYO-1 decode R² vs calendar date (chronic stability).
    python plot_chronic.py chronic_results.jsonl  ->  chronic_curve.png
"""
import json, sys, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "chronic_results.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
rows = [r for r in rows if isinstance(r.get("r2"), (int, float))]
rows.sort(key=lambda r: r["date"])
dates = [datetime.datetime.strptime(r["date"], "%Y%m%d") for r in rows]
r2 = [r["r2"] for r in rows]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(dates, r2, "o-", lw=2, ms=8, color="#1f77b4")
ax.set_ylabel("test R²  (reach-period hand velocity)")
ax.set_xlabel("recording date")
span = (dates[-1] - dates[0]).days
ax.set_title(f"Chronic stability: one frozen POYO-1 across {span} days of Chewie recordings")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("chronic_curve.png", dpi=140)
print("wrote chronic_curve.png")
print("span days:", span, " mean R2:", round(sum(r2) / len(r2), 3),
      " min:", round(min(r2), 3), " max:", round(max(r2), 3))
for r in rows:
    print(r["date"], round(r["r2"], 3))
