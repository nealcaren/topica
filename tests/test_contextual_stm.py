"""Tests for ContextualSTM (experimental): a contextual sentence-embedding topic
model with STM/SCHOLAR-style prevalence covariates."""

import numpy as np
import pytest

import topica

topica.enable_experimental()

BLOCK0 = ["a", "b", "c", "d"]
BLOCK1 = ["e", "f", "g", "h"]


def _corpus(n_per_group=40, dlen=30, seed=0, block_frac=0.85):
    """Two groups. Group A (covariate 0) draws mostly from word block 0, group B
    (covariate 1) mostly from block 1 — a covariate that shifts topic prevalence.
    Returns (docs, doc_embeddings, covariates) with a single non-redundant covariate
    column and a 2-d block-signature embedding per document."""
    rng = np.random.default_rng(seed)
    docs, embs, X = [], [], []
    for g in range(2):
        block = BLOCK0 if g == 0 else BLOCK1
        other = BLOCK1 if g == 0 else BLOCK0
        for _ in range(n_per_group):
            doc, n0, n1 = [], 0, 0
            for _ in range(dlen):
                src = block if rng.random() < block_frac else other
                w = src[int(rng.integers(4))]
                doc.append(w)
                if w in BLOCK0:
                    n0 += 1
                else:
                    n1 += 1
            s = n0 + n1
            docs.append(doc)
            embs.append([n0 / s, n1 / s])
            X.append([float(g)])  # 0 for group A, 1 for group B
    return docs, np.array(embs), np.array(X)


def _fit(iters=150, seed=1, encoder="combined", covariate_mode="encoder_prior", **kw):
    docs, embs, X = _corpus()
    m = topica.ContextualSTM(
        2, encoder=encoder, covariate_mode=covariate_mode,
        covariate_names=["group"], seed=seed, **kw,
    )
    m.fit(docs, embs, covariates=X, iters=iters)
    return m, docs, embs, X


def _block1_topic(tw):
    """Index of the topic that loads word block 1 (cols 4..7)."""
    return 0 if tw[0, 4:].sum() > tw[1, 4:].sum() else 1


# --- shapes / surface -------------------------------------------------------

def test_shapes_and_simplices():
    m, docs, _, _ = _fit()
    assert m.num_topics == 2
    assert m.topic_word.shape == (2, 8)
    assert m.doc_topic.shape == (len(docs), 2)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)


def test_covariate_effects_shape_and_names():
    m, _, _, _ = _fit()
    assert m.covariate_effects.shape == (1, 2)  # (n_covars, K)
    assert m.covariate_names == ["group"]


def test_settings_keys():
    m = topica.ContextualSTM(2, encoder="zeroshot", covariate_mode="prior_only")
    s = m.settings
    assert s["num_topics"] == 2
    assert s["encoder"] == "zeroshot"
    assert s["covariate_mode"] == "prior_only"
    assert "covariates" not in s  # data arg excluded


# --- recovery (both modes) --------------------------------------------------

def test_recovers_covariate_prevalence_encoder_prior():
    """Covariate=1 (group B) loads block 1, so its prevalence effect on the block-1
    topic should exceed its effect on the block-0 topic."""
    m, _, _, _ = _fit(encoder="combined", covariate_mode="encoder_prior")
    tw = m.topic_word
    b1 = _block1_topic(tw)
    b0 = 1 - b1
    eff = m.covariate_effects
    assert eff[0, b1] > eff[0, b0]


def test_recovers_covariate_prevalence_prior_only():
    m, _, _, _ = _fit(encoder="zeroshot", covariate_mode="prior_only")
    tw = m.topic_word
    b1 = _block1_topic(tw)
    b0 = 1 - b1
    eff = m.covariate_effects
    assert eff[0, b1] > eff[0, b0]


def test_effect_direction_matches_empirical_prevalence_shift():
    """B1 consistency check: the built-in covariate effect agrees in direction with
    the empirically observed prevalence shift (group B has more block-1 topic mass)
    and with the post-hoc estimate_effect coefficient sign."""
    m, docs, _, X = _fit()
    tw = m.topic_word
    b1 = _block1_topic(tw)
    theta = m.doc_topic
    groupA = X[:, 0] == 0.0
    groupB = X[:, 0] == 1.0
    empirical_gap = theta[groupB, b1].mean() - theta[groupA, b1].mean()
    assert empirical_gap > 0  # group B really does load the block-1 topic more
    # Built-in effect agrees in sign.
    assert m.covariate_effects[0, b1] > m.covariate_effects[0, 1 - b1]
    # Post-hoc proportion-scale effect (the STM number) agrees in sign.
    res = topica.estimate_effect(theta, X=X)
    te = next(e for e in res if e.topic == b1)
    assert te.as_dict()["feature_0"]["coef"] > 0


# --- determinism ------------------------------------------------------------

def test_determinism():
    a, _, _, _ = _fit(seed=7)
    b, _, _, _ = _fit(seed=7)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    assert np.array_equal(a.covariate_effects, b.covariate_effects)


# --- covariate hygiene guards (B4/B5) ---------------------------------------

def test_constant_covariate_rejected():
    docs, embs, _ = _corpus()
    X = np.ones((len(docs), 1))  # constant across documents
    m = topica.ContextualSTM(2, seed=1)
    with pytest.raises(ValueError, match="constant"):
        m.fit(docs, embs, covariates=X, iters=10)


def test_collinear_covariates_rejected_without_ridge():
    """Full dummy coding (one-hot both levels) is collinear; reject with l2=0."""
    docs, embs, X1 = _corpus()
    g = X1[:, 0]
    X = np.column_stack([1.0 - g, g])  # [1,0]/[0,1] one-hot: rank-deficient
    m = topica.ContextualSTM(2, seed=1, l2_prior_reg=0.0)
    with pytest.raises(ValueError, match="collinear"):
        m.fit(docs, embs, covariates=X, iters=10)


def test_collinear_covariates_allowed_with_ridge():
    docs, embs, X1 = _corpus()
    g = X1[:, 0]
    X = np.column_stack([1.0 - g, g])
    m = topica.ContextualSTM(2, seed=1, l2_prior_reg=0.1)
    m.fit(docs, embs, covariates=X, iters=20)  # ridge identifies it; no error
    assert m.covariate_effects.shape == (2, 2)


# --- constructor guards -----------------------------------------------------

def test_invalid_encoder_rejected():
    with pytest.raises(ValueError, match="encoder"):
        topica.ContextualSTM(2, encoder="bogus")


def test_invalid_covariate_mode_rejected():
    with pytest.raises(ValueError, match="covariate_mode"):
        topica.ContextualSTM(2, covariate_mode="bogus")


def test_num_topics_at_least_two():
    with pytest.raises(ValueError):
        topica.ContextualSTM(1)


# --- experimental gate ------------------------------------------------------

def test_experimental_gate():
    was = topica.experimental_enabled()
    topica.enable_experimental(False)
    try:
        with pytest.raises(RuntimeError):
            topica.ContextualSTM(2)
    finally:
        topica.enable_experimental(was)


# --- persistence ------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    m, docs, embs, X = _fit()
    p = tmp_path / "ctxstm.topica"
    m.save(str(p))
    m2 = topica.ContextualSTM.load(str(p))
    np.testing.assert_array_equal(m.topic_word, m2.topic_word)
    np.testing.assert_array_equal(m.covariate_effects, m2.covariate_effects)
    # transform on the reloaded model reproduces the training doc_topic.
    theta = m2.transform(docs, embs, covariates=X)
    np.testing.assert_allclose(theta, m.doc_topic, atol=1e-6)
