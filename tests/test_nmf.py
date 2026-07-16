"""NMF on topica's multiplicative-update core: it recovers planted word blocks,
exposes the standard fitted surface, round-trips through save/load, is
deterministic under a seed, supports both divergences and both initializations,
and validates its inputs."""

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


def test_construction_defaults():
    m = topica.models.NMF(3)
    assert m.num_topics == 3
    assert "NMF(num_topics=3" in repr(m)


def test_fit_recovers_planted_blocks():
    docs, vocab, _ = _planted()
    m = topica.models.NMF(3)
    m.fit(docs)

    assert m.num_topics == 3
    assert m.topic_word.shape == (3, len(vocab))
    assert m.doc_topic.shape == (len(docs), 3)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)

    # Each topic's top words come from a single planted block; all blocks covered.
    # Word ids are assigned by the corpus builder, so map a column back to its
    # planted block through the model's vocabulary labels ("b{b}w{i}").
    vocab_arr = list(m.vocabulary)
    covered = set()
    tw = m.topic_word
    for t in range(3):
        order = np.argsort(tw[t])[::-1][:4]
        blocks = {int(vocab_arr[w].split("w")[0][1:]) for w in order}
        assert len(blocks) == 1, f"topic {t} top words mix blocks"
        covered.add(next(iter(blocks)))
    assert covered == {0, 1, 2}


def test_four_method_surface():
    docs, vocab, _ = _planted()
    m = topica.models.NMF(3)
    m.fit(docs)

    # topic_word, doc_topic, top_words, save/load are the contract; plus the
    # uniform fitted surface.
    assert isinstance(m.topic_word, np.ndarray)
    assert isinstance(m.doc_topic, np.ndarray)
    assert list(m.topic_names) == ["topic_0", "topic_1", "topic_2"]
    assert sorted(m.vocabulary) == sorted(set(vocab))
    assert len(m.doc_names) == len(docs)
    assert isinstance(m.reconstruction_error, float)
    assert len(m.error_history) >= 2
    assert m.error_history[-1] <= m.error_history[0]  # MU never increases the loss
    assert isinstance(m.converged, bool)
    fh = m.fit_history
    assert fh and fh[0][0] == 1
    assert m.iters_run >= 1
    coh = m.coherence(5)
    assert coh.shape == (3,)


def test_top_words():
    docs, _, _ = _planted()
    m = topica.models.NMF(3)
    m.fit(docs)
    allw = m.top_words(5)
    assert len(allw) == 3
    assert all(len(row) == 5 for row in allw)
    one = m.top_words(5, topic=0)
    assert len(one) == 5
    assert all(isinstance(w, str) and isinstance(p, float) for w, p in one)
    with pytest.raises(Exception):
        m.top_words(5, topic=99)


def test_save_load_roundtrip(tmp_path):
    docs, _, _ = _planted()
    m = topica.models.NMF(3, beta_loss="kullback-leibler", init="random", seed=7)
    m.fit(docs)
    path = str(tmp_path / "nmf.bin")
    m.save(path)

    loaded = topica.models.NMF.load(path)
    assert loaded.num_topics == 3
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)
    assert loaded.reconstruction_error == m.reconstruction_error
    assert list(loaded.topic_names) == list(m.topic_names)


@pytest.mark.parametrize("beta_loss", ["frobenius", "kullback-leibler", "kl"])
def test_beta_loss_values(beta_loss):
    docs, vocab, _ = _planted()
    m = topica.models.NMF(3, beta_loss=beta_loss)
    m.fit(docs)
    assert m.topic_word.shape == (3, len(set(vocab)))
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)


@pytest.mark.parametrize("init", ["nndsvd", "random"])
def test_init_values(init):
    docs, _, _ = _planted()
    m = topica.models.NMF(3, init=init, seed=3)
    m.fit(docs)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)


def test_determinism_same_seed():
    docs, _, _ = _planted()
    # NNDSVD init is seed-independent and deterministic.
    a = topica.models.NMF(3, init="nndsvd")
    a.fit(docs)
    b = topica.models.NMF(3, init="nndsvd")
    b.fit(docs)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    # Random init is reproducible from the seed.
    c = topica.models.NMF(3, init="random", seed=11)
    c.fit(docs)
    d = topica.models.NMF(3, init="random", seed=11)
    d.fit(docs)
    assert np.array_equal(c.topic_word, d.topic_word)
    assert np.array_equal(c.doc_topic, d.doc_topic)


def test_weighting_tfidf():
    docs, vocab, _ = _planted()
    m = topica.models.NMF(3, weighting="tfidf")
    m.fit(docs)
    assert m.topic_word.shape == (3, len(set(vocab)))
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)


def test_input_validation():
    with pytest.raises(Exception):
        topica.models.NMF(1)  # K < 2
    with pytest.raises(Exception):
        topica.models.NMF(3, beta_loss="nonsense")
    with pytest.raises(Exception):
        topica.models.NMF(3, init="nonsense")
    with pytest.raises(Exception):
        topica.models.NMF(3, weighting="nonsense")

    docs, _, _ = _planted(k=3, block=2)  # vocabulary size 6
    m = topica.models.NMF(20)  # K > V
    with pytest.raises(Exception):
        m.fit(docs)

    empty = topica.models.NMF(3)
    with pytest.raises(Exception):
        empty.fit([])


def test_iters_override():
    docs, _, _ = _planted()
    m = topica.models.NMF(3, convergence_tol=0.0)
    m.fit(docs, iters=5)
    assert m.iters_run == 5
    assert len(m.error_history) == 6  # initial error + one per iteration
