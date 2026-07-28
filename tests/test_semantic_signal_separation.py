"""Tests for SemanticSignalSeparation (S³), topica's ICA-over-embeddings model.

S³ is embedding-native: the caller brings a document-embedding matrix and an
aligned vocabulary-embedding matrix. These tests build a planted two-axis
structure (two independent latent signals over disjoint embedding dimensions,
with vocabulary words loading on one axis or the other) so the "right" answer is
known and robust to axis permutation and sign.
"""

import numpy as np
import pytest

import topica


# Two independent latent axes: dims 0-1 carry signal A, dims 2-3 carry signal B,
# dims 4-5 are near-zero noise. Words 0-2 load on axis A, words 3-5 on axis B.
_VOCAB = ["cat", "dog", "pet", "star", "moon", "sky"]
_BLOCK_A = {0, 1, 2}
_BLOCK_B = {3, 4, 5}
_VOCAB_EMB = np.array(
    [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.8, 0.0, 0.0, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.9, 1.0, 0.0, 0.0],
    ]
)


def _planted(d=160, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, d)
    b = rng.uniform(-1.0, 1.0, d)
    doc_emb = np.stack(
        [a, 0.9 * a, b, 0.9 * b, 0.05 * rng.standard_normal(d), 0.05 * rng.standard_normal(d)],
        axis=1,
    )
    docs = [[_VOCAB[i] for i in rng.integers(0, len(_VOCAB), 8)] for _ in range(d)]
    return docs, doc_emb


def _fit(k=2, seed=42, **kw):
    docs, doc_emb = _planted()
    m = topica.SemanticSignalSeparation(k, seed=seed, **kw)
    return m.fit(docs, doc_emb, _VOCAB_EMB, vocabulary=_VOCAB)


def test_shapes_and_normalization():
    m = _fit()
    v = len(m.vocabulary)
    assert m.topic_word.shape == (2, v)
    assert m.doc_topic.shape == (160, 2)
    assert m.components.shape == (2, v)
    assert m.axial_components.shape == (2, v)
    assert m.source_scores.shape == (160, 2)
    # phi / theta are nonnegative distributions.
    assert (m.topic_word >= 0).all()
    assert (m.doc_topic >= 0).all()
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)


def test_recovers_planted_axes():
    """Each ICA axis should own one of the two word blocks at its stronger pole."""
    m = _fit()
    owned = []
    for t in range(2):
        pos = m.top_words(3, topic=t)
        neg = m.top_words(3, topic=t, pole="negative")
        strong = pos if abs(pos[0][1]) >= abs(neg[0][1]) else neg
        ids = {_VOCAB.index(w) for w, _ in strong}
        owned.append(0 if len(ids & _BLOCK_A) >= len(ids & _BLOCK_B) else 1)
    assert sorted(owned) == [0, 1], "the two axes must recover the two blocks"


def test_signed_poles_are_opposites():
    """A signed ICA axis: its two poles draw from opposite word blocks."""
    m = _fit()
    for t in range(2):
        pos = {_VOCAB.index(w) for w, _ in m.top_words(3, topic=t)}
        neg = {_VOCAB.index(w) for w, _ in m.top_words(3, topic=t, pole="negative")}
        assert pos.isdisjoint(neg), "positive and negative poles must not overlap"
        # The positive pole's leading importance is >= the negative pole's.
        assert m.top_words(1, topic=t)[0][1] >= m.top_words(1, topic=t, pole="negative")[0][1]


def test_feature_importance_modes():
    for mode in ("combined", "axial", "angular"):
        m = _fit(feature_importance=mode)
        assert m.settings["feature_importance"] == mode
        assert m.components.shape == (2, len(m.vocabulary))


def test_determinism():
    a = _fit(seed=3)
    b = _fit(seed=3)
    assert np.array_equal(a.components, b.components)
    assert np.array_equal(a.source_scores, b.source_scores)
    assert np.array_equal(a.topic_word, b.topic_word)
    # A different seed gives a different fit (so the test cannot pass trivially).
    c = _fit(seed=999)
    assert not np.array_equal(a.source_scores, c.source_scores)


def test_vocabulary_ordered_without_alignment():
    """Omitting `vocabulary=` accepts rows already in corpus order."""
    docs, doc_emb = _planted()
    corpus = topica.Corpus.from_documents(docs)
    order = [_VOCAB.index(w) for w in corpus.vocabulary]
    m = topica.SemanticSignalSeparation(2, seed=1).fit(corpus, doc_emb, _VOCAB_EMB[order])
    assert m.topic_word.shape == (2, len(corpus.vocabulary))


def test_analysis_surface():
    m = _fit()
    assert len(topica.topic_table(m)) == 2
    _ = topica.summary(m)
    assert m.coherence(5).shape == (2,)


def test_save_load_roundtrip(tmp_path):
    m = _fit()
    p = str(tmp_path / "s3.tt")
    m.save(p)
    loaded = topica.SemanticSignalSeparation.load(p)
    assert np.array_equal(m.components, loaded.components)
    assert np.array_equal(m.source_scores, loaded.source_scores)
    assert np.array_equal(m.topic_word, loaded.topic_word)
    assert loaded.vocabulary == m.vocabulary


def test_bad_params():
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(0)
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2, feature_importance="nope")
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2, iters=0)
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2, convergence_tol=0.0)


def test_fit_input_validation():
    docs, doc_emb = _planted()
    # doc_embeddings row count must match the corpus.
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2).fit(docs, doc_emb[:10], _VOCAB_EMB, vocabulary=_VOCAB)
    # num_topics beyond min(num_docs, dim) is rejected by FastICA.
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(50).fit(docs, doc_emb, _VOCAB_EMB, vocabulary=_VOCAB)
    # unfitted access raises.
    with pytest.raises(RuntimeError):
        _ = topica.SemanticSignalSeparation(2).topic_word


def test_bad_pole_argument():
    m = _fit()
    with pytest.raises(ValueError):
        m.top_words(3, topic=0, pole="sideways")


def test_non_finite_convergence_tol_rejected():
    # +inf and NaN tolerances are rejected (an inf tol would "converge" instantly).
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2, convergence_tol=float("inf"))
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2, convergence_tol=float("nan"))


def test_duplicate_vocabulary_rejected():
    docs, doc_emb = _planted()
    dup = list(_VOCAB)
    dup[1] = dup[0]  # duplicate "cat"
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2).fit(docs, doc_emb, _VOCAB_EMB, vocabulary=dup)


def test_missing_vocab_words_get_zero_importance():
    # A corpus term with no supplied embedding must contribute exactly zero to every
    # axis (not the spurious score a placeholder vector would project to), so it never
    # appears among a topic's top words.
    docs, doc_emb = _planted()
    partial_vocab = _VOCAB[:4]  # drop "moon", "sky"
    partial_emb = _VOCAB_EMB[:4]
    m = topica.SemanticSignalSeparation(2, seed=1).fit(
        docs, doc_emb, partial_emb, vocabulary=partial_vocab
    )
    missing_ids = [i for i, w in enumerate(m.vocabulary) if w in {"moon", "sky"}]
    assert missing_ids, "the dropped words should still be in the corpus vocabulary"
    comps = np.asarray(m.components)
    for i in missing_ids:
        assert np.all(comps[:, i] == 0.0), "a word with no embedding must score zero"
    # A zero-scored missing word cannot outrank a genuinely positive word, so it
    # never leads a topic's positive pole.
    leads = {m.top_words(1, topic=t)[0][0] for t in range(2)}
    assert leads.isdisjoint({"moon", "sky"})


def test_all_missing_vocab_rejected():
    docs, doc_emb = _planted()
    with pytest.raises(ValueError):
        topica.SemanticSignalSeparation(2).fit(
            docs, doc_emb, np.zeros((1, doc_emb.shape[1])), vocabulary=["not_a_corpus_word"]
        )
