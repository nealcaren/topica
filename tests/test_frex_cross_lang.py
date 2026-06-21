"""Single source of truth for FREX / score / lift: topica's Python ``validation``
scorers route through topica-core's stm-faithful Rust ``inspect`` module (the same
one faSTM and the Stata plugin use), so the definitions cannot drift across
languages (issue #260).

These tests verify the Python wrappers expose the Rust core faithfully:

- **FREX**: ``topica.frex`` returns the Rust ``inspect_frex_scores`` exactly, and
  passing ``word_counts`` engages stm's James-Stein exclusivity shrinkage.
- **score**: ``label_topics`` score is the Rust ``inspect_score_scores``.
- **lift**: ``label_topics`` lift is stm's log-lift from ``inspect_lift_scores``
  (exact with ``word_counts``; estimated from the column marginal otherwise, which
  preserves the ranking).

No R or external data needed.
"""

import numpy as np
import pytest

import topica
from topica import _topica
from topica import validation as val


def _continuous_beta(seed=0, K=6, V=200):
    """A K x V topic-word matrix with no ties (so ranks are unambiguous)."""
    rng = np.random.default_rng(seed)
    beta = rng.gamma(0.3, size=(K, V))
    beta /= beta.sum(axis=1, keepdims=True)
    vocab = [f"w{i}" for i in range(V)]
    return beta, vocab


def _matrix_from_labels(pairs, K, V):
    """Rebuild a (K, V) score matrix from per-topic (word 'wIDX', score) lists."""
    mat = np.full((K, V), -np.inf)
    for t in range(K):
        for word, score in pairs[t]:
            mat[t, int(word[1:])] = score
    return mat


def test_frex_routes_through_core_no_shrink():
    beta, vocab = _continuous_beta()
    rust = np.array(_topica.inspect_frex_scores(beta.tolist(), [], 0.5))
    py = _matrix_from_labels(val.frex(beta, vocab, w=0.5, n=beta.shape[1]), *beta.shape)
    np.testing.assert_allclose(py, rust, atol=1e-12, rtol=0)


def test_frex_word_counts_enable_stm_shrinkage():
    beta, vocab = _continuous_beta(seed=2)
    rng = np.random.default_rng(2)
    wc = rng.integers(1, 500, beta.shape[1])
    # With word_counts, topica.frex matches the shrinkage path of the core...
    rust_shrink = np.array(_topica.inspect_frex_scores(beta.tolist(), wc.tolist(), 0.5))
    py_shrink = _matrix_from_labels(
        val.frex(beta, vocab, w=0.5, n=beta.shape[1], word_counts=wc), *beta.shape)
    np.testing.assert_allclose(py_shrink, rust_shrink, atol=1e-12, rtol=0)
    # ...and shrinkage actually changes the scores vs the no-shrink default.
    no_shrink = np.array(_topica.inspect_frex_scores(beta.tolist(), [], 0.5))
    assert not np.allclose(rust_shrink, no_shrink)


@pytest.mark.parametrize("w", [0.3, 0.5, 0.7])
def test_frex_top_words_match_core(w):
    beta, vocab = _continuous_beta(seed=1)
    rust = np.array(_topica.inspect_frex_scores(beta.tolist(), [], w))
    py = val.frex(beta, vocab, w=w, n=10)
    for t in range(beta.shape[0]):
        got = [int(word[1:]) for word, _ in py[t]]
        assert got == np.argsort(rust[t])[::-1][:10].tolist()


def test_label_topics_score_is_core_score():
    beta, vocab = _continuous_beta(seed=3)
    rust = np.array(_topica.inspect_score_scores(beta.tolist()))
    labels = val.label_topics(beta, vocab, n=beta.shape[1])
    py = _matrix_from_labels([d["score"] for d in labels], *beta.shape)
    # Selected ordering matches exactly; values match where defined.
    finite = np.isfinite(py)
    np.testing.assert_allclose(py[finite], rust[finite], atol=1e-9, rtol=1e-6)


def test_label_topics_lift_is_stm_log_lift_with_word_counts():
    # The headline fix: lift is now stm's log-lift from the core, exact when
    # word_counts are supplied (previously this was a beta/mean ratio that did not
    # match stm at all — the old strict xfail).
    beta, vocab = _continuous_beta(seed=4)
    rng = np.random.default_rng(4)
    wc = rng.integers(1, 500, beta.shape[1])
    rust = np.array(_topica.inspect_lift_scores(beta.tolist(), wc.tolist()))
    labels = val.label_topics(beta, vocab, n=beta.shape[1], word_counts=wc)
    py = _matrix_from_labels([d["lift"] for d in labels], *beta.shape)
    finite = np.isfinite(py)
    np.testing.assert_allclose(py[finite], rust[finite], atol=1e-9, rtol=1e-6)


def test_corpus_supplies_word_counts():
    # corpus= reads the corpus word frequencies (aligned to the vocabulary), so it
    # is equivalent to passing those counts explicitly — the zero-effort
    # stm-faithful path.
    docs = [["a", "b", "b", "c"], ["a", "a", "c", "c", "c"], ["b", "c", "d"],
            ["d", "d", "a"]] * 6
    corpus = topica.Corpus.from_documents(docs, min_doc_freq=1)
    vocab = corpus.vocabulary
    wc = np.array(corpus.word_counts)
    assert len(wc) == corpus.num_words and wc.sum() == corpus.total_tokens
    rng = np.random.default_rng(0)
    beta = rng.gamma(0.4, size=(4, corpus.num_words))
    beta /= beta.sum(axis=1, keepdims=True)
    assert val.label_topics(beta, vocab, corpus=corpus) == \
        val.label_topics(beta, vocab, word_counts=wc)
    assert val.frex(beta, vocab, corpus=corpus) == val.frex(beta, vocab, word_counts=wc)
    with pytest.raises(ValueError):
        val.frex(beta, vocab, word_counts=wc, corpus=corpus)


def test_exclusivity_routes_through_core():
    beta, _ = _continuous_beta(seed=6)
    rust = np.array(_topica.inspect_exclusivity(beta.tolist(), 10, 0.7))
    py = topica.exclusivity(beta, n=10, w=0.7)
    np.testing.assert_allclose(py, rust, atol=1e-12, rtol=0)
    # stm's exclusivity is a FREX summary (~[0, n]), not a [0, 1] mean.
    assert np.all(py >= 0.0) and py.max() > 1.0


def test_semantic_coherence_routes_through_core():
    beta, vocab = _continuous_beta(seed=7, K=4, V=60)
    rng = np.random.default_rng(7)
    docs = [[f"w{i}" for i in rng.integers(0, 60, rng.integers(5, 20))]
            for _ in range(200)]
    py = topica.semantic_coherence(beta, docs, vocab, n=10)
    vmap = {w: i for i, w in enumerate(vocab)}
    docs_ids = [[vmap[w] for w in d if w in vmap] for d in docs]
    rust = np.array(_topica.inspect_semantic_coherence(beta.tolist(), docs_ids, 10))
    np.testing.assert_allclose(py, rust, atol=1e-12, rtol=0)
    assert py.shape == (4,)


def test_semantic_coherence_reads_corpus_vocabulary():
    docs = [["a", "b", "c"], ["a", "a", "b"], ["b", "c", "c"], ["a", "c"]] * 8
    corpus = topica.Corpus.from_documents(docs, min_doc_freq=1)
    rng = np.random.default_rng(0)
    beta = rng.gamma(0.4, size=(3, corpus.num_words))
    beta /= beta.sum(axis=1, keepdims=True)
    # Passing the Corpus supplies both the documents and the vocabulary.
    from_corpus = topica.semantic_coherence(beta, corpus, n=5)
    from_lists = topica.semantic_coherence(beta, corpus.documents(),
                                           corpus.vocabulary, n=5)
    np.testing.assert_allclose(from_corpus, from_lists, atol=1e-12)


def test_lift_without_counts_ranks_like_marginal_lift():
    # No word_counts: P(w) is estimated from the column marginal, so lift is
    # log(beta) - log(marginal) — the same ranking as the historical beta/marginal
    # lift, now on the correct (log) scale.
    beta, vocab = _continuous_beta(seed=5)
    labels = val.label_topics(beta, vocab, n=10)
    marginal = beta.mean(axis=0)
    old_lift = beta / np.where(marginal > 0, marginal, 1e-12)
    for t in range(beta.shape[0]):
        got = [int(word[1:]) for word, _ in labels[t]["lift"]]
        assert got == np.argsort(old_lift[t])[::-1][:10].tolist()
