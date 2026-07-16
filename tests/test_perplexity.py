"""Model-agnostic held-out perplexity: topica.diagnostics.perplexity(model, held_out)."""

import numpy as np
import pytest

import topica


def _planted(n, seed):
    rng = np.random.default_rng(seed)
    blocks = [[f"{c}{i}" for i in range(6)] for c in "abcd"]
    return [list(rng.choice(blocks[rng.integers(0, 4)], size=14)) for _ in range(n)]


@pytest.fixture(scope="module")
def split():
    return _planted(240, 0), _planted(60, 1)


def test_perplexity_is_positive_and_finite(split):
    train, held = split
    m = topica.models.LDA(4, seed=1)
    m.fit(train, iters=300)
    pp = topica.diagnostics.perplexity(m, held)
    assert np.isfinite(pp) and pp > 1.0


def test_perplexity_discriminates_k(split):
    # The true structure has 4 blocks; held-out perplexity should prefer K=4
    # over a too-small K=2 (it does not trivially fall with K because the scored
    # tokens are held out from the mixture estimate).
    train, held = split
    pp = {}
    for k in (2, 4):
        m = topica.models.LDA(k, seed=1)
        m.fit(train, iters=300)
        pp[k] = topica.diagnostics.perplexity(m, held)
    assert pp[4] < pp[2]


def test_perplexity_accepts_corpus_and_variational(split):
    train, held = split
    m = topica.models.CTM(4, seed=1)
    m.fit(train, iters=20)
    corpus = topica.Corpus.from_documents(held)
    assert np.isfinite(topica.diagnostics.perplexity(m, corpus))


def test_perplexity_rejects_clustering_models(split):
    train, _ = split
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(train), 8))
    for cls in (topica.models.BERTopic, topica.models.Top2Vec):
        m = cls(min_cluster_size=5, seed=1)
        m.fit(train, emb)
        with pytest.raises(ValueError, match="no held-out perplexity|generative"):
            topica.diagnostics.perplexity(m, train)


def test_perplexity_needs_scorable_documents(split):
    train, _ = split
    m = topica.models.LDA(4, seed=1)
    m.fit(train, iters=150)
    with pytest.raises(ValueError, match="at least 2 tokens"):
        topica.diagnostics.perplexity(m, [["a0"], ["b1"]])  # single-token docs cannot be split
