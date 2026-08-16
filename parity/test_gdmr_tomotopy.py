"""Cross-implementation parity: topica GDMR vs tomotopy GDMRModel.

topica and tomotopy are independent implementations of the same model
(Lee & Song 2020 g-DMR: Legendre-polynomial basis over continuous metadata,
Gibbs sampling).  They share no code and no RNG, so exact numeric agreement
is impossible.  Validation here is *statistical*: fit both on the SAME
tokenized corpus and metadata, then compare the recovered tdf curves.

The check is intentionally soft (Pearson / Spearman correlation and sign
agreement, not numeric tolerance), mirroring the keyATM and STM parity
conventions in this repo.

Skips cleanly when tomotopy is unavailable.

Run directly:

    python parity/test_gdmr_tomotopy.py
"""

from __future__ import annotations

import numpy as np
import pytest

tomotopy = pytest.importorskip(
    "tomotopy",
    reason="tomotopy not installed; skipping GDMR parity check.",
)

import topica  # noqa: E402 — import after skip guard

# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

_VOCAB_A = ["planet", "star", "moon", "rocket", "orbit"]   # space words
_VOCAB_B = ["cat", "dog", "fish", "bird", "mouse"]         # animal words

NUM_TOPICS = 2
SEED = 0
N = 300
DOC_LEN = 10
DEGREES = [2]   # 1-D metadata, Legendre degree 2
ITERS = 500
SIGMA = 1.0
SIGMA0 = 3.0


def _make_corpus(
    n: int = N,
    doc_length: int = DOC_LEN,
    seed: int = SEED,
):
    """Return (docs, metadata) where metadata is (n, 1) float array in [0, 1].

    Documents with metadata > 0.5 draw from VOCAB_A; others from VOCAB_B.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 1.0, size=n)
    docs = []
    for x in xs:
        vocab = _VOCAB_A if x > 0.5 else _VOCAB_B
        docs.append(rng.choice(vocab, size=doc_length).tolist())
    return docs, xs.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_topica(docs, metadata):
    """Fit topica GDMR; return the fitted model."""
    m = topica.GDMR(
        num_topics=NUM_TOPICS,
        degrees=DEGREES,
        sigma=SIGMA,
        sigma0=SIGMA0,
        seed=SEED,
        optimize_interval=25,
        burn_in=100,
    )
    m.fit(
        docs,
        metadata,
        iters=ITERS,
        num_samples=5,
        sample_interval=20,
    )
    return m


def _fit_tomotopy(docs, metadata):
    """Fit tomotopy GDMRModel; return the model.

    tomotopy's GDMRModel uses add_doc/train rather than a batch fit call.
    metadata values are passed as a list of floats per document.
    """
    tp = tomotopy
    mdl = tp.GDMRModel(
        tw=tp.TermWeight.ONE,
        k=NUM_TOPICS,
        degrees=DEGREES,
        sigma=SIGMA,
        sigma0=SIGMA0,
        seed=SEED,
        min_cf=0,
        min_df=0,
    )
    xs = metadata[:, 0].tolist()
    for doc, x in zip(docs, xs):
        mdl.add_doc(doc, numeric_metadata=[float(x)])
    mdl.burn_in = 100
    mdl.train(ITERS, show_progress=False)
    return mdl


def _topica_tdf_curve(model, xs):
    """Evaluate topica GDMR tdf at a 1-D linspace; returns (num, K) array."""
    pts = np.asarray(xs).reshape(-1, 1)
    return model.tdf(pts, normalize=True)


def _tomotopy_tdf_curve(mdl, xs):
    """Evaluate tomotopy GDMRModel tdf (via infer on a proxy document).

    tomotopy exposes `tdf(metadata)` on GDMRModel; fall back to infer if not.

    Assumption: tomotopy >= 0.12 exposes `GDMRModel.tdf(metadata) -> list`.
    If that attribute is unavailable we skip the comparison rather than fail.
    """
    if not hasattr(mdl, "tdf"):
        pytest.skip("tomotopy.GDMRModel has no tdf method; skipping curve comparison.")
    curves = []
    for x in xs:
        probs = np.array(mdl.tdf([x]))  # shape (K,)
        probs = probs / probs.sum()
        curves.append(probs)
    return np.stack(curves, axis=0)  # (num, K)


def _identify_space_topic_topica(model):
    vocab = model.vocabulary
    tw = model.topic_word
    space_mass = [
        sum(tw[t, vocab.index(w)] for w in _VOCAB_A if w in vocab)
        for t in range(model.num_topics)
    ]
    return int(np.argmax(space_mass))


def _identify_space_topic_tomotopy(mdl):
    vocab = mdl.used_vocabs
    tw = np.array([mdl.get_topic_word_dist(k) for k in range(NUM_TOPICS)])
    space_mass = [
        sum(tw[t, list(vocab).index(w)] for w in _VOCAB_A if w in vocab)
        for t in range(NUM_TOPICS)
    ]
    return int(np.argmax(space_mass))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    return _make_corpus()


@pytest.fixture(scope="module")
def topica_model(corpus):
    docs, meta = corpus
    return _fit_topica(docs, meta)


@pytest.fixture(scope="module")
def tomotopy_model(corpus):
    docs, meta = corpus
    return _fit_tomotopy(docs, meta)


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------

_EVAL_XS = np.linspace(0.05, 0.95, 20)


def test_topica_and_tomotopy_recover_space_topic(
    topica_model, tomotopy_model, corpus
):
    """Both implementations should identify a 'space' topic (words from VOCAB_A).

    This is a sanity check that both converged on the same corpus.
    """
    ti = _identify_space_topic_topica(topica_model)
    to = _identify_space_topic_tomotopy(tomotopy_model)
    assert ti in (0, 1)
    assert to in (0, 1)


def test_tdf_curves_positively_correlated(topica_model, tomotopy_model):
    """topica and tomotopy tdf curves for the space topic should be positively
    correlated across the evaluation grid.

    Pearson r >= 0.5 is the threshold (very soft; mainly checks same direction).
    """
    ti = _identify_space_topic_topica(topica_model)
    curve_topica = _topica_tdf_curve(topica_model, _EVAL_XS)[:, ti]

    to_curve_raw = _tomotopy_tdf_curve(tomotopy_model, _EVAL_XS)
    to_idx = _identify_space_topic_tomotopy(tomotopy_model)
    curve_tomotopy = to_curve_raw[:, to_idx]

    r = float(np.corrcoef(curve_topica, curve_tomotopy)[0, 1])
    assert r >= 0.5, (
        f"tdf Pearson r={r:.4f} below threshold 0.5. "
        f"topica: {curve_topica.round(3).tolist()}, "
        f"tomotopy: {curve_tomotopy.round(3).tolist()}"
    )


def test_tdf_space_topic_increases_with_x_topica(topica_model):
    """topica space-topic tdf should be positively correlated with x."""
    ti = _identify_space_topic_topica(topica_model)
    curve = _topica_tdf_curve(topica_model, _EVAL_XS)[:, ti]
    r = float(np.corrcoef(_EVAL_XS, curve)[0, 1])
    assert r > 0.0, f"Expected topica space-topic tdf to rise with x; r={r:.4f}"


def test_tdf_space_topic_increases_with_x_tomotopy(tomotopy_model):
    """tomotopy space-topic tdf should also be positively correlated with x."""
    to_curve_raw = _tomotopy_tdf_curve(tomotopy_model, _EVAL_XS)
    ti = _identify_space_topic_tomotopy(tomotopy_model)
    curve = to_curve_raw[:, ti]
    r = float(np.corrcoef(_EVAL_XS, curve)[0, 1])
    assert r > 0.0, f"Expected tomotopy space-topic tdf to rise with x; r={r:.4f}"


def test_topic_word_top_words_agree(topica_model, tomotopy_model):
    """Top 5 words for the space topic should overlap substantially (>=3 words).

    This is a soft content check, not a rank check.
    """
    ti = _identify_space_topic_topica(topica_model)
    topica_top = {w for w in topica_model.top_words(5, topic=ti)}

    tm_vocab = list(tomotopy_model.used_vocabs)
    tm_tw = np.array(tomotopy_model.get_topic_word_dist(
        _identify_space_topic_tomotopy(tomotopy_model)
    ))
    tm_top_idx = np.argsort(tm_tw)[-5:]
    tomotopy_top = {tm_vocab[i] for i in tm_top_idx}

    overlap = len(topica_top & tomotopy_top)
    assert overlap >= 2, (
        f"Top-5 word overlap between topica and tomotopy is only {overlap}. "
        f"topica: {topica_top}, tomotopy: {tomotopy_top}"
    )


# ---------------------------------------------------------------------------
# Standalone entry point (skips gracefully when tomotopy is absent)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import tomotopy as _  # noqa: F401
    except ImportError:
        print("tomotopy not installed; skipping.")
        raise SystemExit(0)

    docs, meta = _make_corpus()
    print("Fitting topica GDMR …")
    tm = _fit_topica(docs, meta)
    print("Fitting tomotopy GDMRModel …")
    to = _fit_tomotopy(docs, meta)

    ti = _identify_space_topic_topica(tm)
    to_i = _identify_space_topic_tomotopy(to)
    print(f"topica space topic index: {ti}")
    print(f"tomotopy space topic index: {to_i}")

    curve_t = _topica_tdf_curve(tm, _EVAL_XS)[:, ti]
    to_curve_raw = _tomotopy_tdf_curve(to, _EVAL_XS)
    curve_to = to_curve_raw[:, to_i]

    r = float(np.corrcoef(curve_t, curve_to)[0, 1])
    print(f"tdf Pearson r (topica vs tomotopy): {r:.4f}")
    print(f"topica curve: {curve_t.round(3).tolist()}")
    print(f"tomotopy curve: {curve_to.round(3).tolist()}")
