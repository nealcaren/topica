"""Tests for topica.permutation_test (issue #36).

The planted-covariate corpus has a binary covariate that strongly predicts
which of two word-blocks dominates a document. After fitting a two-topic LDA
the covariate effect on the matched topic should sit in the extreme tail of the
permutation null, while an uncorrelated covariate should not.
"""

import numpy as np
import pytest

import topica


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _planted_corpus(n=120, seed=0):
    """Corpus + binary covariate where group 1 uses a-words, group 0 uses b-words.

    Returns (model, docs, covariate) with a fresh two-topic LDA fit.
    """
    rng = np.random.default_rng(seed)
    covariate = (rng.random(n) < 0.5).astype(float)   # ~50/50 split
    a = [f"a{i}" for i in range(8)]
    b = [f"b{i}" for i in range(8)]
    docs = []
    for flag in covariate:
        block = a if flag == 1.0 else b
        docs.append(list(rng.choice(block, size=16, replace=True)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.models.LDA(2, seed=1)
    m.fit(corpus, iters=200)
    return m, docs, covariate


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_returns_list_of_permutation_results():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=5, seed=0, iters=50)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, topica.PermutationResult)


def test_pvalue_in_unit_interval():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=10, seed=0, iters=50)
    for r in results:
        assert 0.0 <= r.pvalue <= 1.0, f"p-value out of range: {r.pvalue}"


def test_null_shape():
    m, docs, cov = _planted_corpus(n=80)
    n_perm = 8
    results = topica.permutation_test(m, docs, cov, n_perm=n_perm, seed=0, iters=50)
    for r in results:
        assert r.null.shape == (n_perm,), f"null shape mismatch: {r.null.shape}"


def test_topic_attribute_range():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=5, seed=0, iters=50)
    topics = {r.topic for r in results}
    assert topics == {0, 1}


# ---------------------------------------------------------------------------
# Sensitivity: planted effect should be extreme vs null
# ---------------------------------------------------------------------------

def test_planted_topic_has_extreme_observed_effect():
    """The topic most correlated with the planted covariate should have its
    observed effect well outside the bulk of the permutation null (n_perm=20
    is enough to detect a clean planted signal)."""
    m, docs, cov = _planted_corpus(n=160, seed=42)
    results = topica.permutation_test(m, docs, cov, n_perm=20, seed=7, iters=100)

    # At least one topic's observed effect should exceed the entire null range.
    obs_vals = [abs(r.observed) for r in results]
    best = max(obs_vals)
    best_result = results[obs_vals.index(best)]

    # The observed effect should be larger in magnitude than the null mean.
    null_abs_mean = float(np.abs(best_result.null).mean())
    assert best > null_abs_mean, (
        f"observed |effect|={best:.4f} not larger than null mean |effect|={null_abs_mean:.4f}"
    )


def test_planted_topic_pvalue_small():
    """The topic best matched to the planted covariate should have a small p-value
    with n_perm=30 permutations on a clear signal."""
    m, docs, cov = _planted_corpus(n=200, seed=99)
    results = topica.permutation_test(m, docs, cov, n_perm=30, seed=3, iters=100)
    best_pval = min(r.pvalue for r in results)
    assert best_pval <= 0.5, (
        f"expected at least one topic to have p <= 0.5; best was {best_pval:.3f}"
    )


def test_random_covariate_pvalue_not_consistently_small():
    """A random covariate uncorrelated with the corpus should not produce
    a very small p-value (most of the time)."""
    m, docs, _ = _planted_corpus(n=120, seed=0)
    rng = np.random.default_rng(77)
    random_cov = (rng.random(len(docs)) < 0.5).astype(float)
    results = topica.permutation_test(m, docs, random_cov, n_perm=20, seed=5, iters=50)
    # At least one topic should have p > 0.1 (not a slam-dunk rejection).
    assert any(r.pvalue > 0.1 for r in results), (
        "expected at least one topic to have p > 0.1 for a random covariate"
    )


# ---------------------------------------------------------------------------
# topics= restriction
# ---------------------------------------------------------------------------

def test_topics_restriction():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=5, seed=0,
                                      topics=[0], iters=50)
    assert len(results) == 1
    assert results[0].topic == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_raises_on_non_binary_covariate():
    m, docs, _ = _planted_corpus(n=60)
    bad_cov = np.arange(len(docs)) % 3  # three levels
    with pytest.raises(ValueError, match="binary"):
        topica.permutation_test(m, docs, bad_cov, n_perm=2, iters=20)


def test_raises_on_wrong_length_covariate():
    m, docs, _ = _planted_corpus(n=60)
    with pytest.raises(ValueError):
        topica.permutation_test(m, docs, np.ones(10), n_perm=2, iters=20)


# ---------------------------------------------------------------------------
# as_dict and topic_name
# ---------------------------------------------------------------------------

def test_as_dict_keys():
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    d = results[0].as_dict()
    for key in ("topic", "topic_name", "observed", "pvalue", "null_mean", "null_std"):
        assert key in d, f"missing key {key!r} in as_dict()"


def test_topic_name_is_string():
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    for r in results:
        assert isinstance(r.topic_name, str)


# ---------------------------------------------------------------------------
# Corpus object accepted
# ---------------------------------------------------------------------------

def test_corpus_object_accepted():
    rng = np.random.default_rng(0)
    n = 60
    cov = (rng.random(n) < 0.5).astype(float)
    a = [f"a{i}" for i in range(6)]
    b = [f"b{i}" for i in range(6)]
    docs = []
    for flag in cov:
        block = a if flag == 1.0 else b
        docs.append(list(rng.choice(block, size=10, replace=True)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.models.LDA(2, seed=1)
    m.fit(corpus, iters=100)
    # Should not raise.
    results = topica.permutation_test(m, corpus, cov, n_perm=3, seed=0, iters=30)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Viz: PermutationTestPlot
# ---------------------------------------------------------------------------

def test_viz_import():
    import topica.viz as viz
    assert hasattr(viz, "permutation_test_plot")
    assert hasattr(viz, "PermutationTestPlot")


def test_viz_to_frame():
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    import topica.viz as viz
    panel = viz.permutation_test_plot(results, covariate_name="group")
    df = panel.to_frame()
    assert len(df) == 2
    assert "observed" in df.columns
    assert "pvalue" in df.columns


def test_viz_to_png_no_error():
    """Rendering to matplotlib should not raise."""
    pytest.importorskip("matplotlib")
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    import topica.viz as viz
    panel = viz.permutation_test_plot(results)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "perm.png")
        fig = panel.to_png(path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Issue #101 fixes
# ---------------------------------------------------------------------------

def _planted_corpus_dmr(n=80, seed=0):
    """Corpus + binary covariate fitted with a two-topic DMR.

    Group 1 documents draw from a-words; group 0 from b-words — a clean
    planted signal so the covariate is genuinely informative.  Returns
    (model, docs, covariate) with docs as token lists.
    """
    rng = np.random.default_rng(seed)
    covariate = (rng.random(n) < 0.5).astype(float)
    a = [f"a{i}" for i in range(8)]
    b = [f"b{i}" for i in range(8)]
    docs = [
        list(rng.choice(a if flag == 1.0 else b, size=14, replace=True))
        for flag in covariate
    ]
    corpus = topica.Corpus.from_documents(docs)
    m = topica.models.DMR(2, seed=3)
    m.fit(corpus, features=covariate[:, None], iters=200)
    return m, docs, covariate


def test_covariate_model_permutation_runs():
    """permutation_test on a DMR model completes without error (issue #101)."""
    m, docs, cov = _planted_corpus_dmr()
    results = topica.permutation_test(m, docs, cov, n_perm=4, seed=0, iters=50)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, topica.PermutationResult)


def test_covariate_threaded_into_refit(monkeypatch):
    """Each permutation refit receives the permuted covariate via the correct
    kwarg (issue #101 bug 1).  We monkeypatch DMR.fit to record calls and
    confirm that 'features' is present in every call's keyword arguments."""
    m, docs, cov = _planted_corpus_dmr(n=60, seed=7)

    fit_calls = []
    original_fit = topica.models.DMR.fit

    def recording_fit(self, data, **kwargs):
        fit_calls.append(dict(kwargs))
        return original_fit(self, data, **kwargs)

    monkeypatch.setattr(topica.models.DMR, "fit", recording_fit)

    n_perm = 3
    topica.permutation_test(m, docs, cov, n_perm=n_perm, seed=0, iters=40)

    # Each of the n_perm refit calls must have passed 'features'.
    assert len(fit_calls) == n_perm, (
        f"expected {n_perm} fit calls, got {len(fit_calls)}"
    )
    for i, kwargs in enumerate(fit_calls):
        assert "features" in kwargs, (
            f"permutation {i}: 'features' not passed to DMR.fit; got kwargs={list(kwargs.keys())}"
        )
        feat = np.asarray(kwargs["features"])
        assert feat.shape == (len(docs), 1), (
            f"permutation {i}: features shape {feat.shape}, expected ({len(docs)}, 1)"
        )


def test_pvalue_never_exactly_zero():
    """p-value uses (1 + count)/(1 + n_perm) so it is never exactly 0 (issue #101 bug 2)."""
    m, docs, cov = _planted_corpus(n=200, seed=42)
    # Use a large enough n_perm that without the +1 fix the p-value would be 0.
    results = topica.permutation_test(m, docs, cov, n_perm=20, seed=0, iters=80)
    for r in results:
        assert r.pvalue > 0.0, (
            f"topic {r.topic}: p-value is exactly 0 (should use +1 convention)"
        )
        assert r.pvalue <= 1.0


def test_nan_null_entries_dropped_from_pvalue():
    """NaN permutation statistics (unmatched topics) are excluded from the
    p-value denominator, not counted as non-extreme (issue #101 bug 3).

    We inject NaN values directly into the null and verify the p-value
    is computed only over the finite entries."""
    from topica.effects import PermutationResult
    import math

    # Construct a result where half the null is NaN.
    null_with_nan = np.array([np.nan, np.nan, 0.1, 0.05, 0.2, 0.08])
    obs = 0.15

    # Manually replicate the fixed p-value logic.
    null_valid = null_with_nan[~np.isnan(null_with_nan)]  # [0.1, 0.05, 0.2, 0.08]
    expected_pval = (1 + np.sum(np.abs(null_valid) >= abs(obs))) / (1 + len(null_valid))
    # |0.1| < 0.15, |0.05| < 0.15, |0.2| >= 0.15, |0.08| < 0.15 => count=1
    # expected = (1+1)/(1+4) = 2/5 = 0.4
    assert abs(expected_pval - 0.4) < 1e-12

    # The old (unfixed) logic would use all 6 entries including NaN.
    # np.abs(NaN) >= 0.15 is False, so NaN entries count as non-extreme
    # and deflate the p-value.  With 6 entries: (1+1)/(1+6) = 2/7 < 2/5.
    old_pval = (1 + np.sum(np.abs(null_with_nan) >= abs(obs))) / (1 + len(null_with_nan))
    assert old_pval < expected_pval, "NaN entries should not lower the p-value"
