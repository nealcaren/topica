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


def _label_corpus(n_per_class=50, k=3, dlen=30, seed=0, block_frac=0.9):
    """K classes, each drawing mostly from its own word block; the class is
    predictable from the words (so predictable from the topics)."""
    rng = np.random.default_rng(seed)
    blocks = [[f"w{g}_{i}" for i in range(4)] for g in range(k)]
    docs, y = [], []
    for g in range(k):
        for _ in range(n_per_class):
            doc = [
                (blocks[g] if rng.random() < block_frac else blocks[int(rng.integers(k))])[
                    int(rng.integers(4))
                ]
                for _ in range(dlen)
            ]
            docs.append(doc)
            y.append(f"class{g}")
    return docs, y


def test_labels_only_predicts():
    docs, y = _label_corpus(k=3, seed=0)
    m = topica.Scholar(3, seed=1)
    m.fit(docs, labels=y, iters=400)
    assert m.classes == ["class0", "class1", "class2"]
    proba = m.predict_proba(docs)
    assert proba.shape == (len(docs), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    pred = m.predict(docs)
    acc = np.mean([p == t for p, t in zip(pred, y)])
    assert acc > 0.8, f"label accuracy {acc}"


def test_covariates_and_labels_compose():
    docs, y = _label_corpus(k=3, seed=1)
    X = np.eye(3)[[int(t[-1]) for t in y]]
    m = topica.Scholar(3, seed=2)
    m.fit(docs, covariates=X, labels=y, iters=400)
    assert m.covariate_effects.shape == (3, 3)
    assert m.classes == ["class0", "class1", "class2"]
    # predict needs the same covariates
    acc = np.mean([p == t for p, t in zip(m.predict(docs, X), y)])
    assert acc > 0.8


def test_predict_requires_labels():
    docs, X = _corpus()
    m = topica.Scholar(2, seed=1)
    m.fit(docs, covariates=X, iters=40)
    with pytest.raises(ValueError):
        m.predict(docs, X)  # fit without labels
    with pytest.raises(ValueError):
        m.predict_proba(docs, X)


def test_labels_save_load_roundtrip(tmp_path):
    docs, y = _label_corpus(k=3, seed=2)
    m = topica.Scholar(3, seed=1)
    m.fit(docs, labels=y, iters=120)
    p = tmp_path / "scholar_lab.topica"
    m.save(str(p))
    m2 = topica.Scholar.load(str(p))
    assert m2.classes == m.classes
    assert np.array_equal(m.predict_proba(docs), m2.predict_proba(docs))


def test_single_class_labels_raises():
    docs, _ = _label_corpus(k=3, seed=0)
    y = ["only"] * len(docs)
    m = topica.Scholar(3, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, labels=y, iters=10)


def test_missing_covariates_and_labels_raises():
    docs, _ = _corpus()
    m = topica.Scholar(2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, iters=10)  # neither covariates, labels, nor content


def _content_corpus(n_per_group=60, dlen=25, seed=0, marker="wMARK", marker_frac=0.35):
    """Two content groups; group 1 heavily over-uses a marker word regardless of topic.
    beta_c for group 1 should place its largest deviation on the marker."""
    rng = np.random.default_rng(seed)
    docs, TC = [], []
    for g in range(2):
        for _ in range(n_per_group):
            doc = []
            for _ in range(dlen):
                if g == 1 and rng.random() < marker_frac:
                    doc.append(marker)
                elif rng.random() < 0.5:
                    doc.append(f"a{int(rng.integers(4))}")
                else:
                    doc.append(f"b{int(rng.integers(4))}")
            docs.append(doc)
            TC.append([1.0, 0.0] if g == 0 else [0.0, 1.0])
    return docs, np.array(TC), marker


def test_content_recovers_word_deviation():
    docs, TC, marker = _content_corpus(seed=0)
    m = topica.Scholar(2, content_names=["g0", "g1"], seed=1)
    m.fit(docs, content=TC, iters=200)
    assert m.content_names == ["g0", "g1"]
    eff = m.content_effects
    assert eff.shape[0] == 2
    mi = list(m.vocabulary).index(marker)
    # Group-1 marker deviation exceeds group 0, and is the top group-1 deviation.
    assert eff[1][mi] > eff[0][mi]
    assert int(eff[1].argmax()) == mi


def test_content_transform_and_interactions():
    docs, TC, _ = _content_corpus(seed=1)
    m = topica.Scholar(2, interactions=True, seed=2)
    m.fit(docs, content=TC, iters=60)
    th = m.transform(docs[:5], content=TC[:5])
    assert th.shape == (5, 2)
    np.testing.assert_allclose(th.sum(axis=1), 1.0, atol=1e-6)


def test_all_three_roles_compose():
    docs, TC, _ = _content_corpus(seed=2)
    y = ["c" + str(int(t[1])) for t in TC]
    m = topica.Scholar(2, seed=3)
    m.fit(docs, covariates=TC, labels=y, content=TC, iters=200)
    assert m.covariate_effects.shape == (2, 2)
    assert m.content_effects.shape[0] == 2
    assert m.classes == ["c0", "c1"]
    acc = np.mean([a == b for a, b in zip(m.predict(docs, covariates=TC, content=TC), y)])
    assert acc > 0.8


def test_content_save_load_roundtrip(tmp_path):
    docs, TC, _ = _content_corpus(seed=3)
    m = topica.Scholar(2, interactions=True, seed=1)
    m.fit(docs, content=TC, iters=80)
    p = tmp_path / "scholar_content.topica"
    m.save(str(p))
    m2 = topica.Scholar.load(str(p))
    assert np.array_equal(m.content_effects, m2.content_effects)
    assert np.array_equal(m.transform(docs[:4], content=TC[:4]),
                          m2.transform(docs[:4], content=TC[:4]))


def test_content_determinism():
    docs, TC, _ = _content_corpus(seed=0)
    a = topica.Scholar(2, seed=7)
    a.fit(docs, content=TC, iters=60)
    b = topica.Scholar(2, seed=7)
    b.fit(docs, content=TC, iters=60)
    assert np.array_equal(a.content_effects, b.content_effects)
    assert np.array_equal(a.topic_word, b.topic_word)


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
