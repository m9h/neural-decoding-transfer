"""TDD for bayes_report.build_report — the Bayesian (wwjd) sibling of alpha_report.

Run (wwj venv, no pytest needed):
    /home/mhough/dev/wwj/.venv/bin/python test_bayes_report.py
or, where pytest is available:
    pytest test_bayes_report.py

The single behaviour under test is the bounded-autonomy CONTRACT, not the alpha
math (that's wwj's, already oracle-tested): no silent data reduction — every walked
matrix is accounted for as fit XOR degenerate, degenerate layers are RETAINED and
flagged (never dropped), and the population mean is computed over FIT layers only so
one dead matrix can't nan-poison or silently shrink the aggregate.
"""
import math

import numpy as np
import jax.numpy as jnp

from bayes_report import build_report


def _alive(seed=0, n=512):
    """Full-rank matrix: a comfortably-sized heavy tail -> finite alpha, large tail."""
    return jnp.asarray(np.random.default_rng(seed).standard_normal((n, n)).astype("float32"))


def _dead(n=256, block=10):
    """Genuinely degenerate: a zero matrix with a small nonzero block, so there are
    STRUCTURAL zero eigenvalues (only `block` meaningful, < min_tail_size 50).
    NB: a merely low-rank float32 product is NOT degenerate by wwj's relative
    _n_meaningful floor (eigvalsh noise ~1e-7·max sits above the 1e-9 floor), and a
    uniformly-tiny matrix keeps its full spectral shape — only structural zeros count."""
    W = np.zeros((n, n), dtype="float32")
    W[:block, :block] = np.random.default_rng(1).standard_normal((block, block))
    return jnp.asarray(W)


def test_accounting_separates_dead_from_fit():
    rep = build_report([("alive", _alive()), ("dead", _dead())], min_tail_size=50)
    acc = rep["accounting"]
    assert acc["walked"] == 2, acc
    assert acc["degenerate"] == 1, acc
    assert acc["fit"] == 1, acc
    assert acc["walked"] == acc["degenerate"] + acc["fit"]      # nothing lost
    assert "dead" in rep["degenerate_layers"]
    # both layers are RETAINED in the per-layer records, each flagged
    recs = {r["layer"]: r for r in rep["layers"]}
    assert recs["dead"]["degenerate"] is True
    assert recs["alive"]["degenerate"] is False
    # population mean is over FIT only and is finite (not nan-poisoned by the dead layer)
    assert math.isfinite(rep["summary"]["mean_alpha"])
    assert rep["summary"]["n_fit"] == 1


def test_empty_input_is_explicit_not_silent():
    rep = build_report([], min_tail_size=50)
    assert rep["accounting"] == {"walked": 0, "degenerate": 0, "fit": 0}
    assert math.isnan(rep["summary"]["mean_alpha"])


if __name__ == "__main__":
    test_accounting_separates_dead_from_fit()
    test_empty_input_is_explicit_not_silent()
    print("ALL PASS")
