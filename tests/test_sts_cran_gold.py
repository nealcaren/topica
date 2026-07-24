"""Offline R-`sts` gold test for topica STS's reference-compatible profile (#454).

topica's STS ships a `reference="cran"` profile that matches the CRAN `sts`
package's public default (`kappaEstimation="adjusted"`, reference `diag(1/20)`
sentiment prior, anchor init, half-step kappa damping). This loads the committed
gold (``parity/sts_cran_gold.npz`` + ``.json``) — R `sts`'s topic-word
distribution at neutral latent sentiment (α^(s)=0) on a fixed 300-doc poliblog
corpus, plus the exact tokenized corpus/sentiment/rating R fit on — refits topica
STS with ``reference="cran"`` on that identical data, and asserts topica's
topic-word distribution clears a bar of R's two-seed cosine floor minus a margin.
R's fit is near-identical across seeds (self cosine ~0.998), so the bar lands at
~0.80 — an externally calibrated cross-implementation regression threshold. The
shuffle check proves the gate is non-vacuous.

Runs in CI WITHOUT Rscript: the reference fit and corpus are frozen in the
committed gold. The topica refit is ~30s, so the parity assertion is marked
``@pytest.mark.slow``; the present / shape / non-vacuous checks run by default.
Complements ``parity/sts_cran_gold.py`` (the live regenerate path) and the
glmnet-kernel check in ``parity/sts_kappa_glmnet.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import sts_cran_gold  # noqa: E402

NAME = "sts_cran"


def test_sts_cran_gold_present():
    npz, js = harness.gold_paths(NAME)
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/sts_cran_gold.py --regenerate` "
        "(needs Rscript + the sts + stm packages)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_sts_cran_gold_shape():
    """Fast default check: the frozen arrays round-trip to the documented sizes."""
    arrays, meta = harness.load_gold(NAME)
    assert arrays["beta1"].shape == (meta["num_topics"], meta["vocab_size"])
    assert len(arrays["vocab"]) == meta["vocab_size"]
    assert len(arrays["docs"]) == meta["num_docs"]
    assert arrays["sent"].shape == (meta["num_docs"],)
    assert arrays["rating"].shape == (meta["num_docs"],)


def test_sts_cran_gold_is_non_vacuous():
    """A shuffled topic-word matrix must fall below the cosine bar — proving the
    gate discriminates a correct adjusted-profile fit from a wrong one. No fit."""
    arrays, meta = harness.load_gold(NAME)
    beta1 = np.asarray(arrays["beta1"], dtype=np.float64)
    bar = float(meta["r_self_cosine"]) - float(meta["margin"])

    rng = np.random.default_rng(0)
    shuffled = beta1[:, rng.permutation(beta1.shape[1])]
    cos, _ = harness.align_cosine(beta1, shuffled)
    jacc = harness.top_word_jaccard(beta1, shuffled, n=10)
    assert cos < bar, (
        f"shuffled STS beta cosine {cos:.4f} should be below the bar {bar:.4f}; "
        f"the gate is vacuous"
    )
    assert jacc < 0.5, f"shuffled STS top-word jaccard {jacc:.4f} too high; gate is weak"


@pytest.mark.slow
def test_sts_cran_matches_committed_gold():
    """Refit topica STS with reference="cran" on the frozen corpus and check that
    its topic-word distribution at mean sentiment clears R's two-seed cosine floor
    minus the margin. Marked ``slow`` (the adjusted-profile EM refit is ~30s)."""
    r = sts_cran_gold.run(verbose=False)
    assert r["passes"], (
        f"topica STS(reference='cran') topic-word cosine {r['cosine']:.4f} below bar "
        f"{r['bar']:.4f} (R self {r['r_self_cosine']:.4f}); the adjusted-profile fit "
        f"has drifted from the CRAN sts package. details: {r}"
    )
