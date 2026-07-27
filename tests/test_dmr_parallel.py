"""DMR multi-threaded (AD-LDA) training mode — the num_threads knob (#566).

DMR reuses LDA's sparse collapsed-Gibbs machinery, so it inherits the same
MALLET-style approximate-parallel path: ``num_threads=1`` is the exact serial
sampler; ``num_threads>1`` partitions documents across workers that sample
against private count tables and reconciles once per sweep.

Contract under test (identical to LDA's):
- ``num_threads=1`` reproduces byte-for-byte across runs (exact serial path).
- a fixed ``num_threads>1`` is deterministic across runs (same threads+seed →
  identical ``topic_word``/``feature_effects``).
- ``num_threads=0`` is clamped to 1 and matches the serial result.
- ``fit(num_threads=)`` overrides the constructor value.
- parallel training preserves the doc_topic simplex + shapes and recovers the
  planted covariate-steered structure.
- GDMR (a pure-Python wrapper over DMR) forwards the knob.
"""

import numpy as np
import pytest

import topica
from topica import DMR


def _make_corpus(seed=0, D=400, V=120, tokens=30):
    """Two document groups with group-biased vocabulary, plus a group covariate
    so the DMR prior actually depends on the feature."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(V)]
    docs, feats = [], []
    for d in range(D):
        grp = d % 2
        # each group draws mostly from its own half of the vocabulary
        lo, hi = (0, V // 2) if grp == 0 else (V // 2, V)
        idx = rng.integers(lo, hi, size=tokens)
        docs.append([vocab[w] for w in idx])
        feats.append([float(grp)])
    return docs, np.asarray(feats)


def _fit(docs, X, num_threads, seed=123, ctor_threads=None):
    ctor = ctor_threads if ctor_threads is not None else num_threads
    m = DMR(6, seed=seed, burn_in=20, num_threads=ctor)
    kw = {} if ctor_threads is None else {"num_threads": num_threads}
    m.fit(docs, X, iters=60, num_samples=3, progress=False, **kw)
    return m


def test_serial_is_reproducible():
    docs, X = _make_corpus()
    a = _fit(docs, X, 1)
    b = _fit(docs, X, 1)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.feature_effects, b.feature_effects)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_fixed_thread_count_is_deterministic(nt):
    docs, X = _make_corpus()
    a = _fit(docs, X, nt)
    b = _fit(docs, X, nt)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.feature_effects, b.feature_effects)


def test_zero_threads_clamped_to_serial():
    docs, X = _make_corpus()
    serial = _fit(docs, X, 1)
    clamped = _fit(docs, X, 0)
    assert np.array_equal(serial.topic_word, clamped.topic_word)


def test_fit_num_threads_overrides_constructor():
    docs, X = _make_corpus()
    # constructor says 1, fit says 4 -> must match a pure-4 run
    override = _fit(docs, X, 4, ctor_threads=1)
    pure4 = _fit(docs, X, 4)
    assert np.array_equal(override.topic_word, pure4.topic_word)


def test_parallel_preserves_invariants_and_structure():
    docs, X = _make_corpus()
    m = _fit(docs, X, 4)
    dt = m.doc_topic
    assert dt.shape == (len(docs), 6)
    npt = np.testing
    npt.assert_allclose(dt.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    assert m.topic_word.shape[0] == 6
    # covariate steers topic prevalence: the two groups should have different
    # mean topic distributions (structure recovered, not collapsed).
    grp = X[:, 0].astype(bool)
    assert not np.allclose(dt[grp].mean(0), dt[~grp].mean(0), atol=1e-3)


def test_settings_report_num_threads():
    m = DMR(6, num_threads=4)
    assert m.settings["num_threads"] == 4


def test_gdmr_forwards_num_threads():
    docs, X = _make_corpus()
    # 1-D continuous metadata for GDMR's Legendre basis.
    meta = X  # single continuous column in [0,1]
    a = topica.GDMR(6, degrees=[2], seed=5, burn_in=20, num_threads=4)
    a.fit(docs, meta, iters=40, num_samples=2)
    b = topica.GDMR(6, degrees=[2], seed=5, burn_in=20, num_threads=4)
    b.fit(docs, meta, iters=40, num_samples=2)
    assert a.settings["num_threads"] == 4
    assert np.array_equal(a.topic_word, b.topic_word)
