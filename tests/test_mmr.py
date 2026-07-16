"""MMR (maximal marginal relevance) top words: diversity-aware reranking."""

import numpy as np
import pytest

import topica


def _setup():
    # three near-synonym clusters; embeddings encode the clusters
    vocab = ["car", "auto", "automobile", "dog", "puppy", "hound", "sky", "cloud", "sun"]
    groups = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    emb = np.zeros((9, 4))
    for g, idxs in groups.items():
        for i in idxs:
            emb[i] = np.eye(4)[g] + np.random.default_rng(i).normal(0, 0.02, 4)
    # relevance order: car-cluster > dog-cluster > sky-cluster
    phi = np.array([[0.30, 0.27, 0.25, 0.07, 0.05, 0.03, 0.012, 0.011, 0.007]])
    return phi, emb, vocab, groups


def test_diversity_zero_is_plain_top_words():
    phi, emb, vocab, _ = _setup()
    plain = [vocab[i] for i in np.argsort(phi[0])[::-1][:5]]
    out = [w for w, _ in topica.mmr(phi, emb, vocab, n=5, diversity=0.0)[0]]
    assert out == plain


def test_diversity_spreads_across_clusters():
    phi, emb, vocab, groups = _setup()
    which = {w: g for g, idxs in groups.items() for w in (vocab[i] for i in idxs)}
    plain_clusters = {which[vocab[i]] for i in np.argsort(phi[0])[::-1][:3]}
    div_words = [w for w, _ in topica.mmr(phi, emb, vocab, n=3, diversity=0.7)[0]]
    div_clusters = {which[w] for w in div_words}
    # the plain top-3 are all one cluster; MMR pulls in the others
    assert len(div_clusters) > len(plain_clusters)
    assert len(div_clusters) == 3


def test_accepts_model_and_validates():
    rng = np.random.default_rng(0)
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    m = topica.models.LDA(2, seed=1)
    m.fit(docs, iters=100)
    word_emb = rng.normal(0, 1, (len(m.vocabulary), 8))
    out = topica.mmr(m, word_emb, n=3, diversity=0.3)  # model-first, vocab from model
    assert len(out) == 2 and len(out[0]) == 3

    with pytest.raises(ValueError, match="word_embeddings has"):
        topica.mmr(m, word_emb[:2])
    with pytest.raises(ValueError, match="diversity"):
        topica.mmr(m, word_emb, diversity=1.5)
