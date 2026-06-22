"""Offline gold-fixture parity for topica DTM vs gensim ``LdaSeqModel`` (#271, Wave 1).

Loads the committed gold (``parity/dtm_gold.npz`` + ``.json``), fits topica DTM on
the SAME smooth-drift corpus + time slices frozen in the gold, aligns its topics
to gensim's per-slice topic-word distributions, and asserts the minimum per-slice/
per-topic cosine clears the bar. On the smooth Gaussian-random-walk drift design
both engines agree at ~0.999, so the bar is cleared by a wide margin.

This runs in CI WITHOUT gensim: the reference fit and the exact corpus are frozen
in the committed gold, so no gensim is imported at test time. The topica refit
(1,200 short docs over 4 slices, 40 iterations) is fast (~0.4s), so it stays in
the default suite alongside the gold-present / shape / non-vacuous checks.
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1] / "parity"
sys.path.insert(0, str(PARITY))

import harness  # noqa: E402
import dtm_gold  # noqa: E402


def test_dtm_gold_present():
    npz, js = harness.gold_paths("dtm")
    assert npz.exists(), (
        f"missing {npz}; regenerate with `python parity/dtm_gold.py --regenerate` "
        "(needs gensim)"
    )
    assert js.exists(), f"missing provenance log {js}"


def test_dtm_gold_shape():
    """Fast default check: the frozen gensim topic-word array has the documented
    (K, T, V) shape and the corpus/times round-trip."""
    arrays, meta = harness.load_gold("dtm")
    ktv = arrays["topic_word_ktv"]
    assert ktv.shape == (meta["num_topics"], meta["num_times"], meta["vocab_size"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    assert len(docs) == meta["num_docs"]
    assert len(arrays["times"]) == meta["num_docs"]


def test_dtm_gold_is_non_vacuous():
    """A shuffled topic-word matrix must FALL BELOW the cosine bar — proving the
    gate discriminates a correct fit from a wrong one. Runs by default (no fit)."""
    import numpy as np

    arrays, meta = harness.load_gold("dtm")
    ktv = arrays["topic_word_ktv"].astype(np.float64)
    bar = float(meta["cosine_bar"])

    # Shuffle the vocab axis of the first-slice topics; aligned cosine must drop.
    g0 = ktv[:, 0, :]
    rng = np.random.default_rng(0)
    shuffled = g0[:, rng.permutation(g0.shape[1])]
    cos, _ = harness.align_cosine(g0, shuffled)
    assert cos < bar, (
        f"shuffled DTM topic-word cosine {cos:.4f} should be below the bar "
        f"{bar:.2f}; the gate is vacuous"
    )


def test_dtm_matches_committed_gold():
    """Refit topica DTM and compare to the frozen gensim gold."""
    r = dtm_gold.run(verbose=False)
    assert r["passes"], (
        f"topica DTM minimum per-slice cosine {r['min_cosine']:.4f} below bar "
        f"{r['bar']:.2f} (mean {r['mean_cosine']:.4f}); details: {r}"
    )
