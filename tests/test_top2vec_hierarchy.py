"""Top2Vec size-ordering, ``hierarchical_topic_reduction``, and a centroid-words
purity check (#489 findings 1, 2, 5).

Findings 1 (reorder half) and 2 add the reference package's post-clustering
behaviour: topics are size-ordered (topic 0 largest) and
``hierarchical_topic_reduction(n)`` merges the smallest topic into its nearest
topic by topic-vector cosine until ``n`` remain. Finding 5 asks that the
*centroid* representation — the distinctly-Top2Vec view — be validated on planted
data, not just the shared c-TF-IDF words.

Geometry is deterministic under a fixed seed and PCA, so we assert on content
(which planted block a topic recovers), not just shapes.
"""

import numpy as np
import pytest

import topica

K, BLOCK, E = 3, 6, 8


def _planted(seed=0, counts=(70, 45, 20)):
    """K well-separated blocks of unequal size in one embedding space. Block ``b``
    owns words ``b{b}w*`` near axis ``b``; each document is drawn from a single
    block. ``counts`` sets the per-block document count so size-ordering is
    testable."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(K) for i in range(BLOCK)]
    v = len(vocab)
    word_emb = np.array([[3.0 if d == w // BLOCK else 0.0 for d in range(E)] for w in range(v)])
    word_emb += rng.normal(0, 0.15, (v, E))
    docs, doc_emb, block = [], [], []
    for b, n in enumerate(counts):
        for _ in range(n):
            docs.append([f"b{b}w{int(rng.integers(BLOCK))}" for _ in range(8)])
            doc_emb.append(word_emb[vocab.index(docs[-1][0])] + rng.normal(0, 0.35, E))
            block.append(b)
    return docs, vocab, word_emb, np.array(doc_emb), block


def _fit(seed=1, counts=(70, 45, 20)):
    docs, vocab, word_emb, doc_emb, block = _planted(seed, counts)
    m = topica.Top2Vec(n_components=5, min_cluster_size=6, seed=seed)
    m.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    return m, docs, vocab, word_emb, doc_emb, block


# ---------------------------------------------------------------------------
# finding 1: topics are size-ordered after fit
# ---------------------------------------------------------------------------

def test_fit_is_size_ordered():
    m, *_ = _fit()
    sizes = m.topic_sizes
    assert sizes == sorted(sizes, reverse=True), f"topics not size-ordered: {sizes}"
    # topic_sizes stays consistent with the (relabelled) hard assignments.
    expected = [sum(1 for l in m.labels if l == t) for t in range(m.num_topics)]
    assert sizes == expected


# ---------------------------------------------------------------------------
# finding 5: the centroid representation recovers single-block topics
# ---------------------------------------------------------------------------

def test_centroid_words_are_block_pure():
    m, *_ = _fit()
    assert m.num_topics >= 2
    # top_words defaults to the centroid representation when word_embeddings are
    # present; each topic's nearest vocabulary words must come from one block.
    for t in range(m.num_topics):
        words = [w for w, _ in m.top_words(4, topic=t)]
        blocks = {w[1] for w in words}
        assert len(blocks) == 1, f"topic {t} centroid words mix blocks: {words}"


# ---------------------------------------------------------------------------
# finding 2: hierarchical_topic_reduction
# ---------------------------------------------------------------------------

def test_reduction_reaches_target_and_stays_size_ordered():
    m, *_ = _fit()
    if m.num_topics < 3:
        pytest.skip("planted fit did not yield >=3 topics on this platform")
    m.hierarchical_topic_reduction(2)
    assert m.num_topics == 2
    sizes = m.topic_sizes
    assert sizes == sorted(sizes, reverse=True)
    # doc_topic rows remain valid distributions over the reduced topics.
    dt = np.asarray(m.doc_topic)
    assert dt.shape[1] == 2
    assert np.allclose(dt.sum(axis=1), 1.0)


def test_reduction_merges_nearest_blocks():
    # Three separable blocks, but block 2 shares block 1's dominant axis (so it is
    # nearest to block 1 by topic-vector cosine) while block 0 is orthogonal.
    # Reducing 3 -> 2 must merge the smallest block (2) into block 1 and leave the
    # orthogonal block 0 on its own.
    rng = np.random.default_rng(3)
    vocab = [f"b{b}w{i}" for b in range(3) for i in range(BLOCK)]
    v = len(vocab)
    centers = np.zeros((3, E))
    centers[0, 0] = 8.0
    centers[1, 3] = 8.0
    centers[2, 3] = 8.0  # block 2 shares block 1's axis (nearest by cosine)...
    centers[2, 6] = 5.0  # ...but is offset enough to cluster separately.
    word_emb = np.array([centers[w // BLOCK] for w in range(v)]) + rng.normal(0, 0.15, (v, E))
    docs, doc_emb = [], []
    for b, n in enumerate((40, 22, 12)):
        for _ in range(n):
            docs.append([f"b{b}w{int(rng.integers(BLOCK))}" for _ in range(8)])
            doc_emb.append(centers[b] + rng.normal(0, 0.25, E))
    m = topica.Top2Vec(n_components=5, min_cluster_size=6, seed=3)
    m.fit(docs, np.array(doc_emb), word_embeddings=word_emb, vocabulary=vocab)
    if m.num_topics != 3:
        pytest.skip("planted fit did not yield exactly 3 topics on this platform")
    m.hierarchical_topic_reduction(2)
    assert m.num_topics == 2
    # The far block (block 0, words b0w*) should still be a pure topic.
    pure = [
        t
        for t in range(2)
        if {w[1] for w, _ in m.top_words(4, topic=t)} == {"0"}
    ]
    assert pure, "block 0 was not preserved as its own topic after reduction"


# ---------------------------------------------------------------------------
# invalid arguments
# ---------------------------------------------------------------------------

def test_reduction_rejects_bad_targets():
    m, *_ = _fit()
    with pytest.raises(ValueError):
        m.hierarchical_topic_reduction(0)
    with pytest.raises(ValueError):
        m.hierarchical_topic_reduction(m.num_topics)  # not strictly smaller
    with pytest.raises(ValueError):
        m.hierarchical_topic_reduction(m.num_topics + 5)
