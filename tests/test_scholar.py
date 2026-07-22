"""Tests for SCHOLAR (Card, Tan & Smith 2018) — prior (prevalence) covariates."""

import numpy as np
import pytest

import topica

BLOCK0 = ["a", "b", "c", "d"]
BLOCK1 = ["e", "f", "g", "h"]


def _corpus(n_per_group=40, dlen=30, seed=0, block_frac=0.85):
    """Two covariate groups. Group A (covariate [1,0]) draws mostly from block 0,
    group B (covariate [0,1]) mostly from block 1 — a covariate that shifts topic
    prevalence."""
    rng = np.random.default_rng(seed)
    docs, X = [], []
    for g in range(2):
        block = BLOCK0 if g == 0 else BLOCK1
        other = BLOCK1 if g == 0 else BLOCK0
        for _ in range(n_per_group):
            doc = [
                (block if rng.random() < block_frac else other)[int(rng.integers(4))]
                for _ in range(dlen)
            ]
            docs.append(doc)
            X.append([1.0, 0.0] if g == 0 else [0.0, 1.0])
    return docs, np.array(X)


def _fit(iters=120, seed=1, **kw):
    docs, X = _corpus()
    m = topica.Scholar(2, covariate_names=["groupA", "groupB"], seed=seed, **kw)
    m.fit(docs, covariates=X, iters=iters)
    return m, docs, X


def test_shapes_and_simplices():
    m, docs, X = _fit()
    assert m.num_topics == 2
    assert m.topic_word.shape == (2, 8)
    assert m.doc_topic.shape == (len(docs), 2)
    # rows are valid distributions
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)


def test_covariate_effects_shape_and_names():
    m, _, _ = _fit()
    assert m.covariate_effects.shape == (2, 2)  # (n_covars, K)
    assert m.covariate_names == ["groupA", "groupB"]


def test_recovers_covariate_prevalence():
    """The covariate should raise the prevalence of the topic its group loads."""
    m, _, _ = _fit()
    tw = m.topic_word
    # Which topic is block-0 dominated (highest mass on words a..d = cols 0..3)?
    block0_topic = 0 if tw[0, :4].sum() > tw[1, :4].sum() else 1
    block1_topic = 1 - block0_topic
    eff = m.covariate_effects
    # Covariate 0 (group A) contrast toward the block-0 topic should exceed
    # covariate 1's contrast toward it.
    c0 = eff[0, block0_topic] - eff[0, block1_topic]
    c1 = eff[1, block0_topic] - eff[1, block1_topic]
    assert c0 > c1


def _corpus_k(k=4, n_per_group=60, dlen=40, seed=0, block_frac=0.9):
    """K covariate groups; group g draws mostly from word block g (one-hot covariate).
    A larger, harder recovery task than the K=2 fixture."""
    rng = np.random.default_rng(seed)
    blocks = [[f"w{g}_{i}" for i in range(5)] for g in range(k)]
    docs, X = [], []
    for g in range(k):
        for _ in range(n_per_group):
            doc = []
            for _ in range(dlen):
                if rng.random() < block_frac:
                    src = blocks[g]
                else:
                    src = blocks[int(rng.integers(k))]
                doc.append(src[int(rng.integers(5))])
            docs.append(doc)
            row = [0.0] * k
            row[g] = 1.0
            X.append(row)
    return docs, np.array(X)


def test_recovers_k4_prevalence_needs_more_epochs():
    """With K=4 the two-layer AVITM encoder needs more epochs than the reference to
    fully separate the blocks; at 400 iters each covariate should raise its own
    topic's prevalence (argmax of covariate_effects row == that covariate's topic)."""
    docs, X = _corpus_k(k=4, seed=0)
    m = topica.Scholar(4, seed=1)
    m.fit(docs, covariates=X, iters=400)
    eff = m.covariate_effects  # (4, 4)
    tw = m.topic_word
    # Map each covariate/block to the topic dominating its 5 planted words.
    vocab = list(m.vocabulary)
    hits = 0
    for g in range(4):
        block_cols = [vocab.index(f"w{g}_{i}") for i in range(5) if f"w{g}_{i}" in vocab]
        block_mass = tw[:, block_cols].sum(axis=1)
        block_topic = int(block_mass.argmax())
        # covariate g should give topic block_topic its highest effect among topics
        if int(eff[g].argmax()) == block_topic:
            hits += 1
    assert hits >= 3, f"only {hits}/4 covariates raised their own topic's prevalence"


def test_covariates_at_construction():
    """Covariates given at construction are used when fit() omits them."""
    docs, X = _corpus()
    m = topica.Scholar(2, covariates=X, seed=1)
    m.fit(docs, iters=40)
    assert m.covariate_effects.shape == (2, 2)
    # default names when none supplied
    assert m.covariate_names == ["covariate_0", "covariate_1"]


def test_transform_requires_matching_covariates():
    m, docs, X = _fit()
    th = m.transform(docs[:5], X[:5])
    assert th.shape == (5, 2)
    np.testing.assert_allclose(th.sum(axis=1), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        m.transform(docs[:5], np.zeros((5, 3)))  # wrong number of covariate columns


def test_determinism():
    a, _, _ = _fit(seed=7)
    b, _, _ = _fit(seed=7)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.covariate_effects, b.covariate_effects)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_seed_changes_fit():
    a, _, _ = _fit(seed=1)
    b, _, _ = _fit(seed=2)
    assert not np.array_equal(a.topic_word, b.topic_word)


def test_save_load_roundtrip(tmp_path):
    m, docs, X = _fit()
    p = tmp_path / "scholar.topica"
    m.save(str(p))
    m2 = topica.Scholar.load(str(p))
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert np.array_equal(m.covariate_effects, m2.covariate_effects)
    assert np.array_equal(m.doc_topic, m2.doc_topic)
    assert m2.covariate_names == m.covariate_names
    # transform still works after load
    th = m2.transform(docs[:3], X[:3])
    assert th.shape == (3, 2)


def test_l2_prior_reg_shrinks_effects():
    """A large L2 penalty on the covariate weights shrinks the effects toward zero."""
    a, _, _ = _fit(l2_prior_reg=0.0)
    b, _, _ = _fit(l2_prior_reg=50.0)
    assert np.abs(b.covariate_effects).max() < np.abs(a.covariate_effects).max()


def test_missing_covariates_raises():
    docs, _ = _corpus()
    m = topica.Scholar(2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, iters=10)  # no covariates anywhere


def test_covariate_row_mismatch_raises():
    docs, X = _corpus()
    m = topica.Scholar(2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, covariates=X[:-1], iters=10)  # one fewer row than documents


def test_fit_history_and_bound():
    m, _, _ = _fit(iters=20)
    hist = m.fit_history
    assert len(hist) == m.epochs_run
    assert hist[0][0] == 1
    assert np.isfinite(m.bound)
    assert len(m.bound_history) == m.epochs_run
