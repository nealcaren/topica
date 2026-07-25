"""Top2Vec search API (#489): the reference package's ``search_*`` /
``similar_words`` / ``get_topic_sizes`` surface, plus the word/doc
embedding-dimension guard the separate-matrix design requires.

The geometry is deterministic under a fixed seed and PCA, so we assert on the
*content* of the results (which block a word/topic/document belongs to), not just
their shapes.
"""

import numpy as np
import pytest

import topica

K, BLOCK, E = 3, 6, 8


def _planted(seed=0):
    """K well-separated blocks in one embedding space. Block ``b`` owns words
    ``b{b}w*`` near axis ``b``; documents are drawn from a single block. Returns
    (docs, vocab, word_emb, doc_emb)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(K) for i in range(BLOCK)]
    v = len(vocab)
    word_emb = np.array([[3.0 if d == w // BLOCK else 0.0 for d in range(E)] for w in range(v)])
    word_emb += rng.normal(0, 0.15, (v, E))
    docs, doc_emb = [], []
    for d in range(150):
        b = d % K
        docs.append([f"b{b}w{int(rng.integers(BLOCK))}" for _ in range(8)])
        doc_emb.append(word_emb[vocab.index(docs[-1][0])] + rng.normal(0, 0.35, E))
    return docs, vocab, word_emb, np.array(doc_emb)


def _block_of(word):
    return word[1]  # "b{block}w{i}"


def _fit(seed=1):
    docs, vocab, word_emb, doc_emb = _planted(seed)
    m = topica.Top2Vec(n_components=5, min_cluster_size=6, seed=seed)
    m.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    return m, docs, vocab, word_emb, doc_emb


# ---------------------------------------------------------------------------
# topic_sizes
# ---------------------------------------------------------------------------

def test_topic_sizes_counts_nonnoise_documents():
    m, docs, *_ = _fit()
    sizes = m.topic_sizes
    assert len(sizes) == m.num_topics
    # Every count is a non-noise cluster's document count.
    expected = [sum(1 for l in m.labels if l == t) for t in range(m.num_topics)]
    assert sizes == expected
    assert sum(sizes) == sum(1 for l in m.labels if l >= 0)


# ---------------------------------------------------------------------------
# word searches
# ---------------------------------------------------------------------------

def test_similar_words_returns_same_block():
    m, _, vocab, *_ = _fit()
    hits = m.similar_words(["b0w0", "b0w1"], n=4)
    assert all(isinstance(w, str) for w, _ in hits)
    assert all(_block_of(w) == "0" for w, _ in hits)


def test_search_words_by_vector_matches_block_axis():
    m, _, vocab, word_emb, _ = _fit()
    # A query on block 1's axis should surface block-1 words.
    q = np.zeros(E)
    q[1] = 3.0
    hits = m.search_words_by_vector(q.tolist(), n=4)
    assert all(_block_of(w) == "1" for w, _ in hits)
    # Scores are descending cosines.
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_words_by_vector_dim_checked():
    m, *_ = _fit()
    with pytest.raises(ValueError):
        m.search_words_by_vector([1.0, 2.0], n=3)


# ---------------------------------------------------------------------------
# topic search
# ---------------------------------------------------------------------------

def test_search_topics_ranks_the_matching_topic_first():
    m, _, vocab, *_ = _fit()
    ranked = m.search_topics(["b2w0", "b2w1"])
    assert len(ranked) == m.num_topics
    best_topic = ranked[0][0]
    # The best topic's centroid words come from block 2.
    words = m.top_words(4, topic=best_topic)
    assert all(_block_of(w) == "2" for w, _ in words)
    # Descending cosine, limitable by n.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(m.search_topics(["b2w0"], n=1)) == 1


# ---------------------------------------------------------------------------
# document searches
# ---------------------------------------------------------------------------

def test_search_documents_by_topic_returns_only_members():
    m, docs, *_ = _fit()
    for t in range(m.num_topics):
        hits = m.search_documents_by_topic(t, num_docs=5)
        assert len(hits) <= 5
        # Only that topic's own documents, never noise.
        assert all(m.labels[i] == t for i, _ in hits)
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)


def test_search_documents_by_keywords_finds_that_block():
    m, docs, *_ = _fit()
    hits = m.search_documents_by_keywords(["b0w0", "b0w1"], num_docs=6)
    assert len(hits) == 6
    # The returned documents should be block-0 documents (index % K == 0).
    assert all(i % K == 0 for i, _ in hits)


def test_search_documents_out_of_range_topic_raises():
    m, *_ = _fit()
    with pytest.raises(ValueError):
        m.search_documents_by_topic(m.num_topics, num_docs=3)


# ---------------------------------------------------------------------------
# keyword resolution + word-vector requirement
# ---------------------------------------------------------------------------

def test_out_of_vocabulary_keywords_raise():
    m, *_ = _fit()
    with pytest.raises(ValueError):
        m.search_topics(["not_a_real_word"])


def test_searches_need_word_embeddings():
    docs, vocab, _, doc_emb = _planted()
    m = topica.Top2Vec(n_components=5, min_cluster_size=6, seed=1)
    m.fit(docs, doc_emb)  # no word_embeddings
    with pytest.raises(RuntimeError):
        m.similar_words(["b0w0"])
    with pytest.raises(RuntimeError):
        m.search_topics(["b0w0"])
    # document-by-topic needs only doc vectors, so it still works
    assert len(m.search_documents_by_topic(0, num_docs=2)) <= 2


# ---------------------------------------------------------------------------
# item 4: word/doc embedding-dimension guard
# ---------------------------------------------------------------------------

def test_word_doc_dim_mismatch_rejected():
    docs, vocab, _, doc_emb = _planted()  # doc_emb is (N, 8)
    bad_word = np.zeros((len(vocab), 6))  # wrong width
    m = topica.Top2Vec(n_components=5, min_cluster_size=6, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, doc_emb, word_embeddings=bad_word, vocabulary=vocab)


def test_ragged_word_rows_rejected():
    docs, vocab, _, doc_emb = _planted()
    ragged = [[0.0] * E for _ in vocab]
    ragged[0] = [0.0] * (E - 1)  # one short row
    m = topica.Top2Vec(n_components=5, min_cluster_size=6, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, doc_emb, word_embeddings=ragged, vocabulary=vocab)


# ---------------------------------------------------------------------------
# search survives save/load
# ---------------------------------------------------------------------------

def test_search_survives_round_trip(tmp_path):
    m, docs, *_ = _fit()
    path = str(tmp_path / "t2v.bin")
    m.save(path)
    ld = topica.Top2Vec.load(path)
    assert ld.topic_sizes == m.topic_sizes
    assert ld.search_documents_by_topic(0, num_docs=4) == m.search_documents_by_topic(0, num_docs=4)
    assert ld.similar_words(["b1w0"], n=3) == m.similar_words(["b1w0"], n=3)
    assert ld.search_documents_by_keywords(["b0w0"], num_docs=3) == m.search_documents_by_keywords(
        ["b0w0"], num_docs=3
    )
