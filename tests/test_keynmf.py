"""KeyNMF (Kristensen-McLachlan et al. 2024): embedding-keyword NMF.

Checks planted-topic recovery from embeddings, that the keyword extraction exactly
matches a correct numpy cosine/top-N/positive oracle, the top-N and positive-filter
behaviour, the K <= min(D, V) guard, determinism, and save/load. Uses cheap planted
embeddings (no sentence-transformers needed).
"""
import numpy as np
import pytest

import topica


def _planted(k=3, words_per=8, docs_per=20, dim=5, seed=0):
    """k topics; each has its own word cluster pointing along an axis. Each document
    draws its words from one topic and its embedding sits on that topic's axis."""
    rng = np.random.default_rng(seed)
    vocab = [f"t{t}w{j}" for t in range(k) for j in range(words_per)]
    word_topic = np.array([t for t in range(k) for _ in range(words_per)])
    word_emb = rng.normal(0, 0.15, (len(vocab), dim))
    for w, t in enumerate(word_topic):
        word_emb[w, t] += 3.0
    docs, doc_emb = [], []
    for _ in range(k * docs_per):
        t = rng.integers(k)
        words = [vocab[i] for i in np.where(word_topic == t)[0]]
        docs.append(list(rng.choice(words, 5, replace=False)))
        e = rng.normal(0, 0.15, dim)
        e[t] += 3.0
        doc_emb.append(e)
    return docs, np.array(doc_emb), vocab, np.asarray(word_emb), word_topic


def test_recovers_planted_topics():
    docs, doc_emb, vocab, word_emb, word_topic = _planted(seed=1)
    m = topica.KeyNMF(num_topics=3, top_n=8, seed=13).fit(
        docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
    )
    assert m.topic_word.shape == (3, len(vocab))
    assert m.doc_topic.shape == (len(docs), 3)
    # each recovered topic's top words come from a single planted cluster
    pure = 0
    for t in range(3):
        top = np.argsort(-m.topic_word[t])[:4]
        if len({word_topic[w] for w in top}) == 1:
            pure += 1
    assert pure >= 2, f"only {pure}/3 topics are cluster-pure"
    # topic_word rows are a simplex
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)


def test_keyword_extraction_matches_numpy_oracle():
    # topica's stage-1 keyword extraction must equal a correct cosine/top-N/positive
    # computation (turftopic's buggy zip is deliberately NOT replicated).
    docs, doc_emb, vocab, word_emb, _ = _planted(seed=2)
    top_n = 6
    m = topica.KeyNMF(num_topics=3, top_n=top_n, seed=13).fit(
        docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
    )
    vidx = {w: i for i, w in enumerate(vocab)}
    wn = word_emb / np.linalg.norm(word_emb, axis=1, keepdims=True)
    for d in range(0, len(docs), 7):
        present = sorted({vidx[w] for w in docs[d] if w in vidx})
        dn = doc_emb[d] / np.linalg.norm(doc_emb[d])
        sims = {i: float(dn @ wn[i]) for i in present}
        oracle = sorted([i for i in present if sims[i] > 0], key=lambda i: -sims[i])[:top_n]
        got_pairs = m.keywords(d)
        got = [vidx[w] for w, _ in got_pairs]
        assert set(got) == set(oracle), f"doc {d}: {got} != {oracle}"
        # the stored importance is the word's own cosine (the NMF input), not just a
        # correct selection — guard against a wrong-value/wrong-metric regression.
        for w, imp in got_pairs:
            assert abs(imp - sims[vidx[w]]) < 1e-6, f"doc {d} word {w}: {imp} != {sims[vidx[w]]}"


def test_top_n_and_positive_filter():
    # A doc whose candidate words have mixed-sign similarity: only positive kept,
    # at most top_n.
    vocab = ["a", "b", "c", "d", "e", "f"]
    dim = 2
    word_emb = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [-1, 0], [-0.9, 0.1], [-0.8, 0.2]], float)
    # 20 docs all pointing +x, each containing all words (so vocab words all present)
    docs = [vocab[:] for _ in range(20)]
    doc_emb = np.tile([1.0, 0.0], (20, 1)) + np.random.default_rng(0).normal(0, 0.01, (20, dim))
    m = topica.KeyNMF(num_topics=2, top_n=2, seed=0).fit(
        docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
    )
    kw = [w for w, _ in m.keywords(0)]
    assert len(kw) <= 2
    assert all(w in {"a", "b", "c"} for w in kw)  # only the +x words are positive


def test_k_exceeds_rank_errors():
    docs, doc_emb, vocab, word_emb, _ = _planted(k=2, docs_per=2, seed=3)  # small D
    # K larger than min(D, V) must error (NNDSVD requirement)
    big_k = len(docs) + len(vocab)
    with pytest.raises(ValueError, match="must be <="):
        topica.KeyNMF(num_topics=big_k, seed=13).fit(
            docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
        )


def test_deterministic():
    docs, doc_emb, vocab, word_emb, _ = _planted(seed=4)
    a = topica.KeyNMF(num_topics=3, seed=13).fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    b = topica.KeyNMF(num_topics=3, seed=13).fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    assert a.keywords(0) == b.keywords(0)


def test_dot_metric():
    docs, doc_emb, vocab, word_emb, word_topic = _planted(seed=5)
    m = topica.KeyNMF(num_topics=3, metric="dot", seed=13).fit(
        docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
    )
    assert m.topic_word.shape == (3, len(vocab))
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)


def test_save_load_round_trip(tmp_path):
    docs, doc_emb, vocab, word_emb, _ = _planted(seed=6)
    m = topica.KeyNMF(num_topics=3, seed=13).fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    p = tmp_path / "keynmf.model"
    m.save(str(p))
    loaded = topica.KeyNMF.load(str(p))
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert loaded.keywords(0) == m.keywords(0)
    assert loaded.vocabulary == m.vocabulary
    assert loaded.settings == m.settings
    assert loaded.coherence(10).shape == m.coherence(10).shape


def test_settings_keys():
    m = topica.KeyNMF(5, top_n=15, metric="dot")
    assert set(m.settings) == {"num_topics", "top_n", "metric", "seed"}
    assert m.settings["seed"] == 13
    assert m.settings["top_n"] == 15
    assert m.settings["metric"] == "dot"


def test_input_validation():
    docs, doc_emb, vocab, word_emb, _ = _planted(seed=7)
    with pytest.raises(ValueError):  # wrong metric
        topica.KeyNMF(3, metric="euclid")
    with pytest.raises(ValueError, match="doc_embeddings has"):
        topica.KeyNMF(3).fit(docs, doc_emb[:-1], word_embeddings=word_emb, vocabulary=vocab)
    with pytest.raises(ValueError, match="word_embeddings has"):
        topica.KeyNMF(3).fit(docs, doc_emb, word_embeddings=word_emb[:-1], vocabulary=vocab)


def test_empty_candidate_doc_is_graceful():
    """A document whose words all have non-positive similarity (or none in vocab)
    yields an empty keyword row and an all-zero X row. The fit must not produce NaN:
    that document's doc_topic falls back to a uniform (finite) distribution."""
    docs, doc_emb, vocab, word_emb, _ = _planted(seed=8)
    # Append a document pointing opposite every planted axis (all cosines < 0), plus a
    # document made entirely of out-of-vocabulary tokens.
    docs = docs + [docs[0], ["zzz_oov_a", "zzz_oov_b"]]
    doc_emb = np.vstack([doc_emb, -np.ones((1, doc_emb.shape[1])), doc_emb[:1]])
    m = topica.KeyNMF(num_topics=3, seed=13).fit(
        docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
    )
    assert np.isfinite(m.topic_word).all()
    assert np.isfinite(m.doc_topic).all()
    # the out-of-vocabulary document has no extractable keywords
    assert m.keywords(len(docs) - 1) == []
    # every doc_topic row is still a valid simplex
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)


def test_zero_norm_word_embedding_is_dropped_not_nan():
    """A word with a zero-norm embedding has an undefined cosine; the model defines it
    as 0 similarity (dropped by the positive filter) rather than NaN."""
    docs, doc_emb, vocab, word_emb, _ = _planted(seed=9)
    word_emb = word_emb.copy()
    word_emb[0] = 0.0  # zero out one word's embedding
    m = topica.KeyNMF(num_topics=3, seed=13).fit(
        docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab
    )
    assert np.isfinite(m.topic_word).all()
    # the zero-norm word never appears as an extracted keyword for any document
    for d in range(len(docs)):
        assert vocab[0] not in {w for w, _ in m.keywords(d)}
