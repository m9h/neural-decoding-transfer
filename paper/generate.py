r"""Reproducible-document 'tangle' step (the Python/Sweave pattern). Reads the POYO
reproduction + transfer + augmentation result files and emits, FROM DATA:
  - paper/figures/fig_*.png         (matplotlib)
  - paper/generated_values.tex      (\newcommand macros for every headline number)
  - paper/generated_tables.tex      (the transfer + spectral + low-data LaTeX tables)
The manuscript \input's the two .tex files and \includegraphics the figures, so re-running
`bash paper/build.sh` reweaves all numbers, tables, and figures from the raw pod pulls --
nothing in paper.tex is hand-transcribed.

Sources: ../dataeff/*.jsonl + ../dataeff/xsub_all.jsonl (raw per-condition pulls),
and data/facts.json (curated scalars from the project log, each traceable to a logged run).
Run via build.sh.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATAEFF = HERE.parent / "dataeff"
FIGS = HERE / "figures"
LEVELS = ["off", "standard", "aggressive"]
FACTS = json.loads((HERE / "data" / "facts.json").read_text())


def jl(name):
    return [json.loads(l) for l in (DATAEFF / name).read_text().splitlines() if l.strip()]


# ----------------------------------------------------------------------------- figures
def fig_dataeff():
    rows = jl("dataeff_results.jsonl")
    by = {"ft": {}, "scratch": {}}
    for r in rows:
        by[r["cond"]][r["frac"]] = r["test_r2"]
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for cond, label, style, c in [("ft", "POYO-MP finetune (frozen core)", "o-", "#1f77b4"),
                                  ("scratch", "from scratch", "s--", "#d62728")]:
        fr = sorted(by[cond])
        ax.plot([f * 100 for f in fr], [by[cond][f] for f in fr], style, label=label, lw=2, ms=7, color=c)
    ax.set_xlabel("new-session training data (\\% of session)")
    ax.set_ylabel("test $R^2$ (hand velocity)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.1, 1)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_dataeff.png", dpi=150)
    plt.close(fig)


def fig_chronic():
    import datetime
    rows = [r for r in jl("chronic_results.jsonl") if isinstance(r.get("r2"), (int, float))]
    rows.sort(key=lambda r: r["date"])
    dates = [datetime.datetime.strptime(r["date"], "%Y%m%d") for r in rows]
    r2 = [r["r2"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.plot(dates, r2, "o-", lw=2, ms=8, color="#1f77b4")
    ax.axhline(np.mean(r2), color="#888", ls=":", lw=1.2, label=f"mean {np.mean(r2):.3f}")
    ax.set_ylabel("test $R^2$ (reach-period hand velocity)")
    ax.set_xlabel("recording date")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_chronic.png", dpi=150)
    plt.close(fig)


def fig_xsub():
    rows = jl("xsub_all.jsonl")
    sessions = sorted({r["session"] for r in rows})
    ft = [next((r["test_r2"] for r in rows if r["session"] == s and r["cond"] == "ft"), np.nan) for s in sessions]
    sc = [next((r["test_r2"] for r in rows if r["session"] == s and r["cond"] == "scratch"), np.nan) for s in sessions]
    labels = [s.split("_")[1] for s in sessions]
    x = np.arange(len(sessions)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.bar(x - w / 2, ft, w, label="POYO-MP finetune (frozen core)", color="#1f77b4")
    ax.bar(x + w / 2, sc, w, label="from scratch", color="#d62728")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("test $R^2$ (reach-period hand velocity)")
    ax.set_xlabel("held-out monkey-t session (never seen in pretraining)")
    ax.set_ylim(min(-0.12, min(sc) - 0.05), 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_xsub.png", dpi=150)
    plt.close(fig)


def fig_rodent():
    """Farthest-domain transfer: monkey-motor base -> rat MEC/hippocampus, decoding 2D position
    via a NEW head. ft vs scratch over held-out sessions, ordered by from-scratch baseline so the
    transfer gain (annotated delta) reads monotone against the scratch deficit."""
    rows = jl("vollan_results.jsonl")
    win = FACTS["rodent"]["windows"]

    def get(s, c):
        return next(r["test_r2"] for r in rows if r["session"] == s and r["cond"] == c)
    sessions = sorted({r["session"] for r in rows}, key=lambda s: get(s, "scratch"))
    ft = [get(s, "ft") for s in sessions]
    sc = [get(s, "scratch") for s in sessions]
    labels = [f"{s.split('_')[1]}\n({win[s]} win)" for s in sessions]
    x = np.arange(len(sessions)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.bar(x - w / 2, ft, w, label="POYO-MP finetune (frozen core)", color="#1f77b4")
    ax.bar(x + w / 2, sc, w, label="from scratch", color="#d62728")
    for i, (f, s) in enumerate(zip(ft, sc)):
        ax.annotate(f"{f - s:+.2f}", (i, max(f, s) + 0.02), ha="center", fontsize=9,
                    color=("#1f77b4" if f >= s else "#d62728"))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("test $R^2$ (2D position)")
    ax.set_xlabel("held-out rat session (Vollan/Moser), by from-scratch baseline")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_rodent.png", dpi=150)
    plt.close(fig)


def _series(rows, key):
    by = {r["level"]: r.get(key) for r in rows}
    return [by[l] for l in LEVELS]


def fig_spectral():
    """The JMLR bridge: finetune-all (plastic) UnitDropout reshapes the feature covariance
    (eff-rank up, top-1 down, condition number down); freeze-core (frozen extractor) is the
    flat NULL control. Three panels, full=solid, freeze=dashed."""
    full = jl("aug_full_results.jsonl")
    frz = jl("aug_results.jsonl")
    x = range(len(LEVELS))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    specs = [("eff_rank", "feature-cov effective rank", "#2ca02c"),
             ("top1_frac", "top-1 eigenvalue fraction", "#d62728"),
             ("cond", "condition number", "#9467bd")]
    for ax, (key, ylab, c) in zip(axes, specs):
        ax.plot(x, _series(full, key), "o-", lw=2, ms=8, color=c, label="finetune-all (plastic)")
        ax.plot(x, _series(frz, key), "s--", lw=1.8, ms=6, color="#888", label="freeze-core (null)")
        ax.set_ylabel(ylab)
        ax.set_xticks(list(x)); ax.set_xticklabels(LEVELS)
        ax.set_xlabel("UnitDropout")
        ax.grid(alpha=0.3)
        if key == "cond":
            ax.set_yscale("log")
    axes[0].legend(fontsize=8.5, loc="best")
    fig.suptitle("UnitDropout reshapes the feature covariance only when the extractor is plastic "
                 "(POYO-MP readout input, held-out monkey t)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_spectral.png", dpi=150)
    plt.close(fig)


def fig_lowdata():
    rows = jl("aug_lowdata_results.jsonl")
    fracs = sorted({r["frac"] for r in rows})

    def get(fr, lv, key):
        return next((r.get(key) for r in rows if r["frac"] == fr and r["level"] == lv), None)
    colors = {0.1: "#d62728", 0.25: "#ff7f0e", 1.0: "#1f77b4"}
    x = range(len(LEVELS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for fr in fracs:
        ax1.plot(x, [get(fr, l, "test_r2") for l in LEVELS], "o-", lw=2, ms=8,
                 color=colors.get(fr), label=f"{int(fr*100)}\\% data")
        ax2.plot(x, [get(fr, l, "cond") for l in LEVELS], "s--", lw=2, ms=7,
                 color=colors.get(fr), label=f"{int(fr*100)}\\% data")
    ax1.set_title("Generalization vs augmentation"); ax1.set_ylabel("test $R^2$")
    ax2.set_title("Feature-cov conditioning vs augmentation")
    ax2.set_ylabel("condition number"); ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xticks(list(x)); ax.set_xticklabels(LEVELS); ax.set_xlabel("UnitDropout")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lowdata.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- macros
def _m(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def _sci(x, sig=1):
    r"""LaTeX \ensuremath scientific-notation string, e.g. 3.06e12 -> \ensuremath{3.1\times10^{12}}."""
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"\\ensuremath{{{m:.{sig}f}\\times10^{{{e}}}}}"


def values_tex():
    out = ["% AUTO-GENERATED by paper/generate.py -- do not edit by hand.\n"]
    R = FACTS["repro"]
    out += [_m("poyoTestRtwo", f"{R['test_r2']:.3f}"), _m("poyoMeanAlpha", f"{R['mean_alpha']:.2f}"),
            _m("poyoRope", f"{R['in_rope_pct']}"), _m("poyoDegen", f"{R['degen']}"),
            _m("poyoNmat", f"{R['n_matrices']}"), _m("poyoParams", f"{R['params_m']}"),
            _m("poyoEpochs", f"{R['epochs']}"), _m("poyoGpus", f"{R['gpus']}"),
            _m("poyoBatch", f"{R['global_batch']}"), _m("poyoMaxLR", f"{R['max_lr']}"),
            _m("poyoDecKv", f"{R['dec_atn_to_kv_alpha']:.2f}")]

    # data efficiency
    de = jl("dataeff_results.jsonl")
    ft = {r["frac"]: r["test_r2"] for r in de if r["cond"] == "ft"}
    sc = [r["test_r2"] for r in de if r["cond"] == "scratch"]
    out += [_m("deFtFracLo", f"{int(min(ft)*100)}"), _m("deFtLo", f"{ft[min(ft)]:.2f}"),
            _m("deFtMid", f"{ft[0.5]:.2f}"), _m("deFtHi", f"{ft[max(ft)]:.2f}"),
            _m("deScratchBound", f"{max(abs(v) for v in sc):.3f}")]

    # chronic
    import datetime
    ch = [r for r in jl("chronic_results.jsonl") if isinstance(r.get("r2"), (int, float))]
    ch.sort(key=lambda r: r["date"])
    r2 = [r["r2"] for r in ch]
    d0 = datetime.datetime.strptime(ch[0]["date"], "%Y%m%d")
    d1 = datetime.datetime.strptime(ch[-1]["date"], "%Y%m%d")
    out += [_m("chronicMean", f"{np.mean(r2):.3f}"), _m("chronicLo", f"{min(r2):.3f}"),
            _m("chronicHi", f"{max(r2):.3f}"), _m("chronicDays", f"{(d1-d0).days}"),
            _m("chronicNsess", f"{len(ch)}"), _m("chronicYrLo", f"{d0.year}"),
            _m("chronicYrHi", f"{d1.year}")]

    # cross-subject
    xs = jl("xsub_all.jsonl")
    sess = sorted({r["session"] for r in xs})
    xft = [next(r["test_r2"] for r in xs if r["session"] == s and r["cond"] == "ft") for s in sess]
    xsc = [next(r["test_r2"] for r in xs if r["session"] == s and r["cond"] == "scratch") for s in sess]
    out += [_m("xsubNsess", f"{len(sess)}"), _m("xsubFt", f"{np.mean(xft):.3f}"),
            _m("xsubScratch", f"{np.mean(xsc):.2f}"), _m("xsubGap", f"{np.mean(np.array(xft)-np.array(xsc)):.2f}"),
            _m("xsubFtLo", f"{min(xft):.3f}"), _m("xsubFtHi", f"{max(xft):.3f}")]

    # dmfc timing
    dm = {r["cond"]: r["test_r2"] for r in jl("dmfc_elapsed_results.jsonl")}
    D = FACTS["dmfc"]
    out += [_m("dmfcRampFt", f"{dm['ft']:.3f}"), _m("dmfcRampScratch", f"{dm['scratch']:.3f}"),
            _m("dmfcRampGap", f"{dm['ft']-dm['scratch']:+.3f}"),
            _m("dmfcConstFt", f"{D['const_ft']:.3f}"), _m("dmfcConstScratch", f"{D['const_scratch']:.3f}"),
            _m("dmfcAlphaFt", f"{D['alpha_ft']:.2f}"), _m("dmfcAlphaScratch", f"{D['alpha_scratch']:.2f}")]

    # rodent cross-species spatial transfer (farthest domain): gain tracks the scratch deficit
    vo = jl("vollan_results.jsonl")
    win = FACTS["rodent"]["windows"]
    vses = sorted({r["session"] for r in vo})
    vft = {s: next(r["test_r2"] for r in vo if r["session"] == s and r["cond"] == "ft") for s in vses}
    vsc = {s: next(r["test_r2"] for r in vo if r["session"] == s and r["cond"] == "scratch") for s in vses}
    starved = min(vses, key=lambda s: win[s])   # fewest train windows -> most data-starved
    rich = max(vses, key=lambda s: vsc[s])       # highest from-scratch baseline (saturated)
    fmean = float(np.mean([vft[s] for s in vses])); smean = float(np.mean([vsc[s] for s in vses]))
    out += [_m("rodentNsess", f"{len(vses)}"),
            _m("rodentFtMean", f"{fmean:.3f}"), _m("rodentScratchMean", f"{smean:.3f}"),
            _m("rodentDeltaMean", f"{fmean-smean:+.3f}"),
            _m("rodentFtLo", f"{min(vft.values()):.3f}"), _m("rodentFtHi", f"{max(vft.values()):.3f}"),
            _m("rodentStarvedWin", f"{win[starved]}"),
            _m("rodentStarvedScratch", f"{vsc[starved]:.3f}"), _m("rodentStarvedFt", f"{vft[starved]:.3f}"),
            _m("rodentStarvedDelta", f"{vft[starved]-vsc[starved]:+.3f}"),
            _m("rodentRichWin", f"{win[rich]}"),
            _m("rodentRichScratch", f"{vsc[rich]:.3f}"), _m("rodentRichDelta", f"{vft[rich]-vsc[rich]:+.3f}")]

    # augmentation, finetune-all (positive) + freeze-core (null)
    full = {r["level"]: r for r in jl("aug_full_results.jsonl")}
    frz = {r["level"]: r for r in jl("aug_results.jsonl")}
    out += [_m("augEffOff", f"{full['off']['eff_rank']:.2f}"), _m("augEffAgg", f"{full['aggressive']['eff_rank']:.2f}"),
            _m("augTopOff", f"{full['off']['top1_frac']:.2f}"), _m("augTopAgg", f"{full['aggressive']['top1_frac']:.2f}"),
            _m("augCondOff", _sci(full['off']['cond'])), _m("augCondAgg", _sci(full['aggressive']['cond'])),
            _m("augCondDrop", f"{full['off']['cond']/full['aggressive']['cond']:.1f}"),
            _m("augFreezeEffSpread", f"{max(frz[l]['eff_rank'] for l in LEVELS)-min(frz[l]['eff_rank'] for l in LEVELS):.3f}"),
            _m("augFreezeAlpha", f"{FACTS['aug_alpha']['freeze']['off']:.2f}"),
            _m("augFullAlphaOff", f"{FACTS['aug_alpha']['full']['off']:.3f}"),
            _m("augFullAlphaAgg", f"{FACTS['aug_alpha']['full']['aggressive']:.3f}")]

    # low-data interaction
    ld = jl("aug_lowdata_results.jsonl")

    def g(fr, lv, key):
        return next(r.get(key) for r in ld if r["frac"] == fr and r["level"] == lv)
    out += [_m("ldLoOff", f"{g(0.1,'off','test_r2'):.3f}"), _m("ldLoStd", f"{g(0.1,'standard','test_r2'):.3f}"),
            _m("ldLoAgg", f"{g(0.1,'aggressive','test_r2'):.3f}"),
            _m("ldLoGain", f"{g(0.1,'aggressive','test_r2')-g(0.1,'off','test_r2'):.2f}"),
            _m("ldLoCondOff", _sci(g(0.1, 'off', 'cond'))), _m("ldLoCondStd", _sci(g(0.1, 'standard', 'cond'))),
            _m("ldLoCondDrop", f"{g(0.1,'off','cond')/g(0.1,'standard','cond'):.1f}"),
            _m("ldHiOff", f"{g(1.0,'off','test_r2'):.3f}"), _m("ldHiAgg", f"{g(1.0,'aggressive','test_r2'):.3f}")]
    return "".join(out)


# ----------------------------------------------------------------------------- tables
def _booktab(header, rows, colfmt):
    s = ["\\begin{tabular}{" + colfmt + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    s += [" & ".join(r) + " \\\\" for r in rows]
    s += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(s)


def tables_tex():
    # transfer two-axis table (decode R^2 + wwj alpha, same checkpoints)
    head = ["Model", "decode $R^2$", "$\\bar\\alpha$", "in-ROPE", "degen/" + str(FACTS["repro"]["n_matrices"])]
    rows = [[t["model"], f"{t['decode_r2']:.3f}", f"{t['mean_alpha']:.2f}",
             f"{t['in_rope_pct']}\\%", f"{t['degen']}"] for t in FACTS["transfer"]]
    transfer = _booktab(head, rows, "lcccc")

    # spectral table: finetune-all feature-covariance metrics + weight alpha, per UnitDropout level
    full = {r["level"]: r for r in jl("aug_full_results.jsonl")}
    a = FACTS["aug_alpha"]["full"]
    head2 = ["UnitDropout", "test $R^2$", "eff.\\ rank", "top-1 frac", "cond.\\ number", "wwj $\\bar\\alpha$"]
    rows2 = [[l, f"{full[l]['test_r2']:.3f}", f"{full[l]['eff_rank']:.2f}",
              f"{full[l]['top1_frac']:.2f}", _sci(full[l]['cond']), f"{a[l]:.3f}"] for l in LEVELS]
    spectral = _booktab(head2, rows2, "lccccc")

    # low-data interaction table (test R^2 grid)
    ld = jl("aug_lowdata_results.jsonl")
    fracs = sorted({r["frac"] for r in ld})

    def g(fr, lv):
        return next(r["test_r2"] for r in ld if r["frac"] == fr and r["level"] == lv)
    head3 = ["train data"] + LEVELS + ["$\\Delta$(agg$-$off)"]
    rows3 = [[f"{int(fr*100)}\\%"] + [f"{g(fr,l):.3f}" for l in LEVELS]
             + [f"{g(fr,'aggressive')-g(fr,'off'):+.3f}"] for fr in fracs]
    lowdata = _booktab(head3, rows3, "lcccc")

    # rodent cross-species transfer table (sorted by from-scratch baseline; gain tracks the deficit)
    vo = jl("vollan_results.jsonl")
    win = FACTS["rodent"]["windows"]

    def vget(s, c):
        return next(r["test_r2"] for r in vo if r["session"] == s and r["cond"] == c)
    vses = sorted({r["session"] for r in vo}, key=lambda s: vget(s, "scratch"))
    head4 = ["Rat session", "train windows", "scratch $R^2$", "finetune $R^2$", "$\\Delta$"]
    rows4 = [[s.replace("_", "\\_"), f"{win[s]}", f"{vget(s,'scratch'):.3f}",
              f"{vget(s,'ft'):.3f}", f"{vget(s,'ft')-vget(s,'scratch'):+.3f}"] for s in vses]
    rodent = _booktab(head4, rows4, "lcccc")

    return ("% AUTO-GENERATED by paper/generate.py.\n"
            "\\newcommand{\\poyoTransferTable}{" + transfer + "}\n"
            "\\newcommand{\\poyoSpectralTable}{" + spectral + "}\n"
            "\\newcommand{\\poyoLowdataTable}{" + lowdata + "}\n"
            "\\newcommand{\\poyoRodentTable}{" + rodent + "}\n")


def main():
    FIGS.mkdir(exist_ok=True)
    fig_dataeff(); fig_chronic(); fig_xsub(); fig_rodent(); fig_spectral(); fig_lowdata()
    (HERE / "generated_values.tex").write_text(values_tex())
    (HERE / "generated_tables.tex").write_text(tables_tex())
    print("generated: figures/{dataeff,chronic,xsub,rodent,spectral,lowdata}.png, "
          "generated_values.tex, generated_tables.tex")


if __name__ == "__main__":
    main()
