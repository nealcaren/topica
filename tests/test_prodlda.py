"""ProdLDA (AVITM) on topica's hand-coded VAE core: it recovers planted word
blocks, exposes the standard fitted surface, transforms new documents with one
encoder pass, is deterministic under a seed, and validates its inputs."""

import numpy as np
import pytest

import topica


def _planted(k=3, block=8, n=240, length=15, seed=0):
    """K word-blocks; each document draws its tokens from one block. Returns
    (docs, vocab, truth)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs, truth = [], []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(length)])
        truth.append(b)
    return docs, vocab, np.array(truth)


def _model(num_topics=3, **kw):
    return topica.ProdLDA(
        num_topics=num_topics, batch_size=60, lr=0.01, dropout=0.0, **kw
    )


def test_prodlda_recovers_planted_blocks():
    docs, vocab, _ = _planted()
    m = _model(seed=1)
    theta = m.fit_transform(docs)

    assert m.num_topics == 3
    assert m.topic_word.shape == (3, len(vocab))
    assert theta.shape == (len(docs), 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)

    # Each topic's top words come from one block, covering all blocks.
    covered = set()
    for t in range(3):
        blocks = {w.split("w")[0] for w, _ in m.top_words(4, topic=t)}
        assert len(blocks) == 1, f"topic {t} mixes blocks: {blocks}"
        covered |= blocks
    assert len(covered) == 3


def test_prodlda_top_words_all_topics_and_names():
    docs, vocab, _ = _planted()
    m = _model(seed=1)
    m.fit(docs, iters=150)
    allw = m.top_words(5)  # topic=None -> list per topic
    assert len(allw) == 3 and all(len(t) == 5 for t in allw)
    assert m.topic_names == ["topic_0", "topic_1", "topic_2"]
    assert set(m.vocabulary) == set(vocab)


def test_prodlda_transform_is_encoder_pass():
    docs, _, _ = _planted()
    m = _model(seed=2)
    m.fit(docs, iters=150)
    new = [["b0w1", "b0w2"], ["b1w0", "b1w3"], ["b2w5", "b2w4"]]
    theta = m.transform(new)
    assert theta.shape == (3, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert len(set(theta.argmax(axis=1))) == 3  # each lands on a distinct topic


def test_prodlda_bound_and_trace():
    docs, _, _ = _planted()
    m = _model(seed=3)
    m.fit(docs, iters=150)
    assert m.epochs_run == 150
    assert len(m.bound_history) == 150
    # The ELBO improves over training (final beats the first epoch).
    assert m.bound_history[-1] > m.bound_history[0]
    assert m.bound == pytest.approx(m.bound_history[-1])


def test_prodlda_determinism():
    docs, _, _ = _planted()
    a = _model(seed=7)
    a.fit(docs, iters=150)
    b = _model(seed=7)
    b.fit(docs, iters=150)
    assert np.allclose(a.topic_word, b.topic_word)
    assert np.allclose(a.doc_topic, b.doc_topic)


def test_prodlda_validation():
    with pytest.raises(ValueError):
        topica.ProdLDA(num_topics=1)  # need >= 2 topics
    with pytest.raises(ValueError):
        topica.ProdLDA(num_topics=3, alpha=0.0)  # alpha must be > 0
    with pytest.raises(ValueError):
        topica.ProdLDA(num_topics=3, dropout=1.0)  # dropout in [0, 1)
    with pytest.raises(RuntimeError):
        topica.ProdLDA(num_topics=3).topic_word  # not fitted


# --- VAE objective/prior flags (#174 contrastive, #176 prior) ----------------

def test_prodlda_flags_accepted_and_exposed():
    m = topica.ProdLDA(
        num_topics=3, prior="dirichlet", contrastive=True,
        contrastive_weight=0.3, contrastive_temp=0.2,
    )
    assert m.prior == "dirichlet"
    assert m.contrastive is True
    # Laplace, no contrastive is the default.
    d = topica.ProdLDA(num_topics=3)
    assert d.prior == "laplace" and d.contrastive is False


def test_prodlda_flag_validation():
    with pytest.raises(ValueError):
        topica.ProdLDA(num_topics=3, prior="gaussian")  # unknown prior
    with pytest.raises(ValueError):
        topica.ProdLDA(num_topics=3, contrastive=True, contrastive_weight=-1.0)
    with pytest.raises(ValueError):
        topica.ProdLDA(num_topics=3, contrastive=True, contrastive_temp=0.0)


def test_prodlda_flags_change_results_but_stay_valid():
    docs, vocab, _ = _planted(n=120)
    base = _model(seed=1); base.fit(docs, iters=60)
    dir_m = _model(seed=1, prior="dirichlet"); dir_m.fit(docs, iters=60)
    con_m = _model(seed=1, contrastive=True); con_m.fit(docs, iters=60)
    # Each flag changes the fitted topics.
    assert not np.allclose(base.topic_word, dir_m.topic_word)
    assert not np.allclose(base.topic_word, con_m.topic_word)
    # Outputs remain valid distributions.
    for m in (dir_m, con_m):
        assert np.allclose(m.topic_word.sum(axis=1), 1.0)
        assert np.allclose(m.doc_topic.sum(axis=1), 1.0)


def test_prodlda_flags_deterministic():
    docs, _, _ = _planted(n=120)
    for kw in ({"prior": "dirichlet"}, {"contrastive": True},
               {"prior": "dirichlet", "contrastive": True}):
        a = _model(seed=5, **kw); a.fit(docs, iters=60)
        b = _model(seed=5, **kw); b.fit(docs, iters=60)
        assert np.array_equal(a.topic_word, b.topic_word)
        assert np.array_equal(a.doc_topic, b.doc_topic)


def test_prodlda_flags_save_load_roundtrip(tmp_path):
    docs, _, _ = _planted(n=120)
    m = _model(seed=2, prior="dirichlet", contrastive=True, contrastive_weight=0.4)
    m.fit(docs, iters=60)
    path = str(tmp_path / "prodlda_flags.bin")
    m.save(path)
    loaded = topica.ProdLDA.load(path)
    assert loaded.prior == "dirichlet"
    assert loaded.contrastive is True
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)
