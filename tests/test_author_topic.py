"""Tests for AuthorTopic: the Author-Topic Model (Rosen-Zvi, Griffiths, Steyvers &
Smyth, UAI 2004), validated against gensim's AuthorTopicModel in parity/.

ATM conditions topics on authors: each author has a topic distribution, and a
document mixes its authors. doc_topic is the per-document empirical topic simplex
(content-based, like LDA); author_topic (A×K) is the model-defining output.
"""

import numpy as np
import pytest

import topica


def _planted():
    """Three disjoint word blocks; three authors, each writing one block. Plus a few
    co-authored documents that mix two authors' topics."""
    blocks = [["a", "b", "c", "d"], ["m", "n", "o", "p"], ["x", "y", "z", "q"]]
    docs, authors = [], []
    for ai, blk in enumerate(blocks):
        for _ in range(30):
            docs.append(list(blk) * 3)
            authors.append([f"auth{ai}"])
    # co-authored: authors 0 and 1 together
    for _ in range(10):
        docs.append(["a", "b", "m", "n"])
        authors.append(["auth0", "auth1"])
    return docs, authors


def test_shapes_and_normalization():
    docs, authors = _planted()
    m = topica.AuthorTopic(3, seed=0).fit(docs, authors, iters=200)
    V = len(m.vocabulary)
    assert m.topic_word.shape == (3, V)
    assert m.doc_topic.shape == (len(docs), 3)
    assert m.author_topic.shape == (3, 3)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert np.allclose(m.author_topic.sum(axis=1), 1.0)
    assert list(m.authors) == ["auth0", "auth1", "auth2"]


def test_recovers_author_structure():
    docs, authors = _planted()
    m = topica.AuthorTopic(3, seed=0).fit(docs, authors, iters=300)
    # Each single-topic author has a distinct dominant topic.
    dom = m.author_topic.argmax(axis=1)
    assert len(set(dom.tolist())) == 3, f"authors not on distinct topics: {dom}"
    # Each recovered topic concentrates on its 4-word block: the top 4 words carry
    # almost all the mass.
    top4 = np.sort(m.topic_word, axis=1)[:, -4:].sum(axis=1)
    assert np.all(top4 > 0.9), f"topics not block-concentrated: {top4}"


def test_determinism():
    docs, authors = _planted()
    a = topica.AuthorTopic(3, seed=7).fit(docs, authors, iters=120)
    b = topica.AuthorTopic(3, seed=7).fit(docs, authors, iters=120)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.author_topic, b.author_topic)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_top_authors():
    docs, authors = _planted()
    m = topica.AuthorTopic(3, seed=0).fit(docs, authors, iters=300)
    # The top author for each author's own topic should be that author.
    for ai in range(3):
        t = int(m.author_topic[ai].argmax())
        top = m.top_authors(t, n=1)
        assert top[0][0] == f"auth{ai}"
        assert 0.0 <= top[0][1] <= 1.0


def test_top_words_and_coherence():
    docs, authors = _planted()
    m = topica.AuthorTopic(3, seed=0).fit(docs, authors, iters=100)
    tw = m.top_words(3, topic=0)
    assert len(tw) == 3 and isinstance(tw[0], tuple)
    assert m.coherence(4).shape == (3,)


def test_fit_history_and_converged():
    docs, authors = _planted()
    m = topica.AuthorTopic(3, seed=0).fit(docs, authors, iters=100)
    assert len(m.fit_history) > 0
    it, ll = m.fit_history[0]
    assert isinstance(it, int) and it > 0 and isinstance(ll, float)
    # Held-in log-likelihood should improve from first to last recorded point.
    assert m.fit_history[-1][1] >= m.fit_history[0][1]
    assert m.converged is False


def test_lda_degeneracy_matches_topica_lda():
    """One unique author per document reduces ATM to LDA. Topic recovery should
    match topica.LDA on the same corpus (topic-aligned cosine near 1); not bit-exact
    because topica LDA uses the SparseLDA sampler, a different RNG stream."""
    docs, _ = _planted()
    authors = [[f"doc{d}"] for d in range(len(docs))]  # unique author per doc
    at = topica.AuthorTopic(3, seed=1).fit(docs, authors, iters=300)
    lda = topica.LDA(3, seed=1).fit(docs, iters=300)

    def aligned_cosine(A, B):
        # greedy Hungarian-lite: best one-to-one match by cosine
        A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
        B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
        sim = A @ B.T
        used, total = set(), 0.0
        for i in np.argsort(-sim.max(axis=1)):
            j = int(np.argmax([sim[i, c] if c not in used else -1 for c in range(sim.shape[1])]))
            used.add(j)
            total += sim[i, j]
        return total / A.shape[0]

    assert aligned_cosine(at.topic_word, lda.topic_word) > 0.9


def test_settings_and_save_load(tmp_path):
    docs, authors = _planted()
    m = topica.AuthorTopic(3, alpha=1.0, beta=0.02, seed=5).fit(docs, authors, iters=80)
    assert m.settings == {"num_topics": 3, "alpha": 1.0, "beta": 0.02, "seed": 5}
    p = str(tmp_path / "at.topica")
    m.save(p)
    L = topica.AuthorTopic.load(p)
    assert np.array_equal(L.topic_word, m.topic_word)
    assert np.array_equal(L.author_topic, m.author_topic)
    assert list(L.authors) == list(m.authors)
    assert L.settings == m.settings


def test_default_alpha_is_paper_value():
    m = topica.AuthorTopic(5)
    assert m.settings["alpha"] == pytest.approx(50.0 / 5)
    assert m.settings["beta"] == 0.01


def test_rejects_mismatched_authors_length():
    docs, authors = _planted()
    with pytest.raises((ValueError, RuntimeError)):
        topica.AuthorTopic(3).fit(docs, authors[:-1], iters=5)


def test_rejects_empty_author_set():
    docs, authors = _planted()
    authors = [list(a) for a in authors]
    authors[0] = []
    with pytest.raises((ValueError, RuntimeError)):
        topica.AuthorTopic(3).fit(docs, authors, iters=5)


def test_handles_lopsided_authors_and_topics():
    # Many topics, few authors, and vice versa — must not panic.
    docs = [["a", "b", "c", "d"], ["a", "b", "b", "c"], ["c", "d", "d", "a"]] * 5
    authors = [["a0"], ["a1"], ["a0"]] * 5
    m = topica.AuthorTopic(12, seed=0).fit(docs, authors, iters=20)
    assert m.author_topic.shape == (2, 12)
