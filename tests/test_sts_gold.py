"""Offline gold-fixture parity for topica STS vs R `stm`-on-slim (issue #271, final wave).

Loads the committed gold (``parity/sts_gold.npz`` + ``.json``), fits topica STS AND
topica STM on the SAME fixed-seed 2,000-doc poliblog subsample that R was fit on
(frozen in the gold), reads STS's recognizable topic at the mean sentiment, aligns it
to R's Spectral beta, and asserts (a) STS sits as close to R as topica's already
validated STM does — within the Spectral basin slack — and (b) STS agrees with
topica's own STM (it structurally extends it).

REFERENCE PATH. The cross-implementation reference here is **R `stm` on the slim
subsample**, not the authors' published STS RDS: that RDS is a 12 MB full-corpus
pre-fit (too big to commit, and not re-fittable on a subsample). STS reduces to STM
structurally, so R-stm-on-slim is the faithful slim-corpus reference; the
authors'-RDS full-corpus comparison stays live in ``parity/sts_r_compare.py``.

This runs in CI WITHOUT Rscript: the reference fit and the exact corpus are frozen in
the committed gold, so no R toolchain is touched at test time (``run()`` imports only
topica + numpy/scipy). The topica STS+STM refit is ~60s, so the heavy refit assertion
is marked ``@pytest.mark.slow``; the gold-present / shape / non-vacuous checks run by
default.
"""

import sys
from pathlib import Path

import pytest

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import sts_gold  # noqa: E402


def test_sts_gold_present():
    npz, js = harness.gold_paths("sts")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/sts_gold.py --regenerate` "
        "(needs R + the stm package)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_sts_gold_shape():
    """Fast default check: the frozen arrays round-trip to the documented sizes."""
    arrays, meta = harness.load_gold("sts")
    assert arrays["r_spectral"].shape == (meta["K"], meta["vocab_size"])
    assert arrays["rating"].shape == (meta["num_docs"],)
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    assert len(docs) == meta["num_docs"]
    assert len(arrays["vocab"]) == meta["vocab_size"]


def test_sts_gold_is_non_vacuous():
    """A shuffled topic-word matrix must FAIL the cross-impl cosine bar — proving the
    gate discriminates a correct fit from a wrong one. Runs by default (no fit)."""
    import numpy as np

    arrays, meta = harness.load_gold("sts")
    r_spectral = np.asarray(arrays["r_spectral"], dtype=np.float64)
    # The same bar the offline compare applies, computed from the frozen provenance.
    bar = max(
        float(meta["topica_stm_vs_r_cosine"]),
        float(meta["r_spectral_vs_random"]),
    ) - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = r_spectral[:, rng.permutation(r_spectral.shape[1])]
    cos, _ = harness.align_cosine(r_spectral, shuffled)
    assert cos < bar, (
        f"shuffled beta cosine {cos:.4f} should be below the cross-impl bar "
        f"{bar:.4f}; the gate is vacuous"
    )


@pytest.mark.slow
def test_sts_matches_committed_gold():
    """Refit topica STS + STM on the frozen slim corpus and check the parity bars.

    Marked ``slow`` (the STS lasso + STM refit is ~60s). Run with ``-m slow``."""
    r = sts_gold.run(verbose=False)
    assert r["passes"], (
        f"topica STS failed the parity bars: STS-vs-R cosine {r['sts_vs_r_cosine']:.4f} "
        f"(cross-impl bar {r['cross_impl_bar']:.4f}, STM-vs-R {r['stm_vs_r_cosine']:.4f}), "
        f"STS-vs-STM {r['sts_vs_stm_cosine']:.4f} (>= {r['sts_vs_stm_min']}); details: {r}"
    )
