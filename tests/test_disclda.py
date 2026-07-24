"""Tests for DiscLDA (Lacoste-Julien, Sha & Jordan 2008)."""

import numpy as np
import pytest

import topica

NC, CBW, SBW = 3, 6, 6            # classes, class-block words, shared-block words
V = NC * CBW + SBW
SHARED_START = NC * CBW
LABELS = ["A", "B", "C"]


def _corpus(n=300, dlen=14, seed=0, class_frac=0.65):
    """Class c documents draw mostly from class c's word block, some from shared."""
    rng = np.random.default_rng(seed)
    docs, y = [], []
    for i in range(n):
        c = i % NC
        doc = []
        for _ in range(dlen):
            if rng.random() < class_frac:
                doc.append(f"w{c * CBW + int(rng.integers(CBW))}")
            else:
                doc.append(f"w{SHARED_START + int(rng.integers(SBW))}")
        docs.append(doc)
        y.append(LABELS[c])
    return docs, y


def _fit(k_class=2, k_shared=2, iters=300, seed=42, **kw):
    docs, y = _corpus()
    m = topica.DiscLDA(k_class, k_shared, iters=iters, seed=seed, **kw)
    m.fit(docs, y)
    return m, docs, y


def test_shapes_and_simplices():
    m, docs, y = _fit()
    assert m.classes == LABELS
    assert m.num_topics == NC * 2 + 2
    assert m.topic_word.shape == (m.num_topics, V)
    assert np.allclose(m.topic_word.sum(1), 1.0)
    assert m.doc_topic.shape == (len(docs), m.num_topics)
    assert np.allclose(m.doc_topic.sum(1), 1.0)


def test_doc_topic_restricted_to_class_and_shared_block():
    m, docs, y = _fit(k_class=1, k_shared=2)
    dt = m.doc_topic
    shared = set(m.shared_topic_ids())
    for d, lab in enumerate(y):
        allowed = set(m.class_topic_ids(lab)) | shared
        # all mass on allowed topics
        mass = sum(dt[d, t] for t in range(m.num_topics) if t not in allowed)
        assert mass < 1e-9, f"doc {d} put mass outside its class+shared block"


def test_class_topics_recover_class_blocks():
    m, docs, y = _fit(k_class=1, k_shared=2)
    vocab = m.vocabulary
    for c, lab in enumerate(LABELS):
        # the class's single specific topic should be dominated by its own block
        tid = m.class_topic_ids(lab)[0]
        top_word = vocab[int(np.argmax(m.topic_word[tid]))]
        block = int(top_word[1:]) // CBW
        assert block == c and int(top_word[1:]) < SHARED_START, (
            f"class {lab} topic peaked on {top_word}, expected class-{c} block"
        )


def test_shared_topics_recover_shared_block():
    m, docs, y = _fit(k_class=1, k_shared=2)
    vocab = m.vocabulary
    for tid in m.shared_topic_ids():
        top_word = vocab[int(np.argmax(m.topic_word[tid]))]
        assert int(top_word[1:]) >= SHARED_START, (
            f"shared topic {tid} peaked on {top_word}, expected shared block"
        )


def test_predict_and_proba():
    m, docs, y = _fit(k_class=2, k_shared=2)
    test_docs, test_y = _corpus(n=60, seed=999)
    preds = m.predict(test_docs)
    acc = np.mean([p == t for p, t in zip(preds, test_y)])
    assert acc > 0.7, f"held-out accuracy {acc}"
    proba = m.predict_proba(test_docs)
    assert proba.shape == (len(test_docs), NC)
    assert np.allclose(proba.sum(1), 1.0)


def test_transform_shape():
    m, docs, y = _fit()
    rep = m.transform(docs[:10])
    assert rep.shape == (10, m.num_topics)
    assert np.allclose(rep.sum(1), 1.0, atol=1e-6)


def test_determinism():
    m1, docs, y = _fit()
    m2 = topica.DiscLDA(2, 2, iters=300, seed=42)
    m2.fit(docs, y)
    assert np.array_equal(m1.topic_word, m2.topic_word)
    assert np.array_equal(m1.doc_topic, m2.doc_topic)


def test_save_load_roundtrip(tmp_path):
    m, docs, y = _fit()
    p = tmp_path / "disclda.topica"
    m.save(str(p))
    ml = topica.DiscLDA.load(str(p))
    assert ml.classes == m.classes
    assert np.array_equal(ml.topic_word, m.topic_word)
    assert np.array_equal(ml.doc_topic, m.doc_topic)


def test_int_labels_accepted():
    docs, _ = _corpus()
    y = [i % NC for i in range(len(docs))]  # int labels
    m = topica.DiscLDA(1, 2, iters=50)
    m.fit(docs, y)
    assert m.classes == ["0", "1", "2"]


def test_mismatched_labels_raise():
    docs, y = _corpus()
    m = topica.DiscLDA(1, 2, iters=10)
    with pytest.raises(ValueError):
        m.fit(docs, y[:-5])


def test_single_class_raises():
    docs, _ = _corpus()
    m = topica.DiscLDA(1, 2, iters=10)
    with pytest.raises(ValueError):
        m.fit(docs, ["A"] * len(docs))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_hyperparams_rejected(bad):
    # #460: `<= 0.0` is false for NaN/+inf, so they slipped past the guard and
    # produced NaN topic-word / doc-topic / predict_proba. Now rejected.
    with pytest.raises(ValueError, match="finite"):
        topica.DiscLDA(2, 2, alpha=bad)
    with pytest.raises(ValueError, match="finite"):
        topica.DiscLDA(2, 2, beta=bad)


def test_zero_infer_sweeps_rejected():
    with pytest.raises(ValueError, match="infer_sweeps"):
        topica.DiscLDA(2, 2, infer_sweeps=0)


# --- #460: direct-classifier class-prior semantics --------------------------

def _imbalanced_identical(n=200, dlen=8, seed=0, maj_frac=0.9):
    """A deliberately imbalanced corpus whose two classes are language-identical:
    every document draws from the SAME vocabulary regardless of label, so the
    likelihood carries no class signal and the classifier output is driven purely
    by the class prior. The review's calibration test case."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(12)]
    docs = [[vocab[int(rng.integers(len(vocab)))] for _ in range(dlen)] for _ in range(n)]
    n_maj = int(round(n * maj_frac))
    y = ["maj"] * n_maj + ["min"] * (n - n_maj)
    return docs, y


def test_empirical_prior_calibrates_to_class_prevalence():
    # Language-identical 90:10 corpus: with the empirical prior (default),
    # predict_proba tracks prevalence rather than collapsing to ~0.5/0.5.
    docs, y = _imbalanced_identical(maj_frac=0.9)
    m = topica.DiscLDA(2, 2, iters=200, infer_sweeps=40, seed=1)
    m.fit(docs, y)
    assert m.classes == ["maj", "min"]
    assert m.class_counts == [180, 20]
    np.testing.assert_allclose(m.class_prior, [0.9, 0.1], atol=1e-9)
    # A held-out language-identical doc leans to the majority class near the prior.
    proba = np.asarray(m.predict_proba([["w1", "w2", "w3"]]))[0]
    assert proba[0] > proba[1] and proba[0] > 0.75
    # An empty / all-OOV document returns exactly the prior (majority prediction),
    # not a sorted-order tie break.
    oov = np.asarray(m.predict_proba([["not_in_vocab"]]))[0]
    np.testing.assert_allclose(oov, [0.9, 0.1], atol=1e-9)
    assert m.predict([[]]) == ["maj"]


def test_uniform_prior_is_uninformative_on_identical_classes():
    docs, y = _imbalanced_identical(maj_frac=0.9)
    m = topica.DiscLDA(2, 2, iters=200, infer_sweeps=40, seed=1, class_prior="uniform")
    m.fit(docs, y)
    np.testing.assert_allclose(m.class_prior, [0.5, 0.5], atol=1e-9)
    oov = np.asarray(m.predict_proba([["not_in_vocab"]]))[0]
    np.testing.assert_allclose(oov, [0.5, 0.5], atol=1e-9)


def test_custom_prior_is_normalized_and_ordered():
    docs, y = _imbalanced_identical()
    # Weights are in sorted-class order: classes == ["maj", "min"].
    m = topica.DiscLDA(2, 2, iters=100, infer_sweeps=20, seed=1, class_prior=[3.0, 1.0])
    m.fit(docs, y)
    np.testing.assert_allclose(m.class_prior, [0.75, 0.25], atol=1e-9)
    assert m.settings["class_prior"] == [3.0, 1.0]


def test_bad_class_prior_string_rejected_at_construction():
    with pytest.raises(ValueError, match="empirical"):
        topica.DiscLDA(2, 2, class_prior="bogus")


def test_wrong_length_custom_prior_rejected_at_fit():
    # Length is only checkable once num_classes is known (at fit): 3 weights, 2 classes.
    docs, y = _imbalanced_identical()
    m = topica.DiscLDA(2, 2, iters=10, class_prior=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="classes"):
        m.fit(docs, y)


@pytest.mark.parametrize("bad", [[1.0, -1.0], [1.0, float("nan")], []])
def test_bad_custom_prior_weights_rejected_at_construction(bad):
    with pytest.raises(ValueError, match="class_prior"):
        topica.DiscLDA(2, 2, class_prior=bad)


def test_class_prior_survives_save_load(tmp_path):
    docs, y = _imbalanced_identical()
    m = topica.DiscLDA(2, 2, iters=100, infer_sweeps=20, seed=1)  # empirical
    m.fit(docs, y)
    p = tmp_path / "disclda_prior.topica"
    m.save(str(p))
    ml = topica.DiscLDA.load(str(p))
    np.testing.assert_allclose(ml.class_prior, m.class_prior, atol=1e-12)
    assert ml.class_counts == m.class_counts
    # predictions reproduce after reload
    np.testing.assert_allclose(
        np.asarray(ml.predict_proba([["not_in_vocab"]])),
        np.asarray(m.predict_proba([["not_in_vocab"]])),
        atol=1e-12,
    )


def test_fit_iters_zero_rejected():
    # The constructor rejects iters == 0; a per-fit override must too (#460).
    docs, y = _corpus()
    m = topica.DiscLDA(2, 2, iters=10)
    with pytest.raises(ValueError, match="iters"):
        m.fit(docs, y, iters=0)
