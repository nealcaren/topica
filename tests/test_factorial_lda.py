"""Tests for FactorialLDA (fLDA, Paul & Dredze NIPS 2012; issue #606).

The reference Java is non-reproducible, so the main-roster confidence comes from
the Rust finite-difference gradient + factor-tying tests (see
``src/factorial_lda.rs``) plus the tuple-level recovery here: on a planted
topic x sentiment corpus the six tuple word-distributions recover the six planted
(topic, sentiment) profiles after Hungarian alignment (invariant to which factor
becomes which axis, since the factors are role-exchangeable).
"""

import random

import numpy as np
import pytest

import topica

TOPICS = {
    0: ["game", "team", "score", "play", "ball", "coach", "season", "league"],
    1: ["market", "stock", "price", "trade", "bank", "profit", "invest", "shares"],
    2: ["film", "movie", "actor", "scene", "director", "plot", "cast", "screen"],
}
SENTS = {
    0: ["great", "love", "excellent", "wonderful", "best", "brilliant", "superb"],
    1: ["terrible", "hate", "awful", "worst", "poor", "boring", "disappointing"],
}
BG = ["the", "a", "and", "of", "to", "is", "was", "very", "really", "this"]


def planted_corpus(n_docs=300, doc_len=40, seed=12345):
    rng = random.Random(seed)
    docs, labels = [], []
    for _ in range(n_docs):
        t = rng.randrange(len(TOPICS))
        s = rng.randrange(len(SENTS))
        words = []
        for _ in range(doc_len):
            r = rng.random()
            if r < 0.45:
                words.append(rng.choice(TOPICS[t]))
            elif r < 0.80:
                words.append(rng.choice(SENTS[s]))
            else:
                words.append(rng.choice(BG))
        docs.append(words)
        labels.append((t, s))
    return docs, labels


def _planted_profiles(vocab):
    vi = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)

    def prof(words, mass):
        out = np.zeros(v)
        for w in words:
            if w in vi:
                out[vi[w]] += mass / len(words)
        return out

    profs = []
    for t in range(len(TOPICS)):
        for s in range(len(SENTS)):
            p = prof(TOPICS[t], 0.45) + prof(SENTS[s], 0.35) + prof(BG, 0.20)
            profs.append(p / p.sum())
    return np.array(profs)


def _mean_aligned_cosine(phi, planted):
    from scipy.optimize import linear_sum_assignment

    def norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    sim = norm(phi) @ norm(planted).T
    r, c = linear_sum_assignment(-sim)
    return float(np.mean([sim[i, j] for i, j in zip(r, c)]))


def test_shapes_and_distributions():
    docs, _ = planted_corpus(n_docs=120, doc_len=30)
    m = topica.FactorialLDA(factor_sizes=[3, 2], seed=1)
    m.fit(docs, iters=150, samples=40)
    assert m.num_topics == 6
    assert m.num_tuples == 6
    assert m.factor_sizes == [3, 2]
    assert m.topic_word.shape == (6, len(m.vocabulary))
    assert m.doc_topic.shape == (len(docs), 6)
    assert np.allclose(m.doc_topic.sum(1), 1.0, atol=1e-6)
    assert np.allclose(m.topic_word.sum(1), 1.0, atol=1e-6)
    assert np.isfinite(m.topic_word).all()
    # tuples enumerate the Cartesian product in mixed-radix order
    assert m.tuples == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]


def test_determinism():
    docs, _ = planted_corpus(n_docs=100, doc_len=25)
    a = topica.FactorialLDA(factor_sizes=[2, 2], seed=7)
    a.fit(docs, iters=120, samples=30)
    b = topica.FactorialLDA(factor_sizes=[2, 2], seed=7)
    b.fit(docs, iters=120, samples=30)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_tuple_recovery_on_planted_corpus():
    docs, _ = planted_corpus()
    m = topica.FactorialLDA(factor_sizes=[3, 2], seed=1)
    m.fit(docs, iters=800, samples=150)
    planted = _planted_profiles(list(m.vocabulary))
    cos = _mean_aligned_cosine(m.topic_word, planted)
    # The six tuples recover the six planted topic x sentiment profiles.
    assert cos > 0.9, f"mean aligned cosine {cos:.3f} too low"


def test_word_priors_help_recovery():
    """Ablation direction (data-dependent, not universal): on a corpus built with
    shared-component structure, structured word priors recover the tuples at least
    as well as the base model."""
    docs, _ = planted_corpus()
    full = topica.FactorialLDA(factor_sizes=[3, 2], seed=2, word_priors=True)
    full.fit(docs, iters=600, samples=120)
    base = topica.FactorialLDA(factor_sizes=[3, 2], seed=2, word_priors=False)
    base.fit(docs, iters=600, samples=120)
    profs = _planted_profiles(list(full.vocabulary))
    cos_full = _mean_aligned_cosine(full.topic_word, profs)
    cos_base = _mean_aligned_cosine(base.topic_word, profs)
    assert cos_full >= cos_base - 0.02, f"word priors hurt: full={cos_full:.3f} base={cos_base:.3f}"


def test_sparsity_flag_controls_activity():
    docs, _ = planted_corpus(n_docs=150, doc_len=30)
    on = topica.FactorialLDA(factor_sizes=[3, 2], seed=3, sparsity=True)
    on.fit(docs, iters=300, samples=60)
    off = topica.FactorialLDA(factor_sizes=[3, 2], seed=3, sparsity=False)
    off.fit(docs, iters=300, samples=60)
    # With sparsity off, every tuple stays fully active (b = 1).
    assert np.allclose(off.tuple_activity, 1.0)
    # With sparsity on, activity is a learned value in (0, 1).
    assert on.tuple_activity.min() >= 0.0 and on.tuple_activity.max() <= 1.0
    assert not np.allclose(on.tuple_activity, 1.0)


def test_omega_priors_seed_axis():
    """Informed omega priors (NAACL-2013 feature): seeding a factor's components
    with sentiment words steers those components toward the sentiment axis."""
    docs, _ = planted_corpus()
    priors = {
        "components": {
            (1, 0): {w: 2.0 for w in SENTS[0]},  # factor 1 comp 0 <- positive
            (1, 1): {w: 2.0 for w in SENTS[1]},  # factor 1 comp 1 <- negative
        }
    }
    m = topica.FactorialLDA(factor_sizes=[3, 2], seed=4)
    m.fit(docs, iters=500, samples=100, omega_priors=priors)
    pos = dict(m.factor_top_words(1, 0, 8))
    neg = dict(m.factor_top_words(1, 1, 8))
    # The seeded components put their sentiment words at the top.
    assert sum(w in pos for w in SENTS[0]) >= 3
    assert sum(w in neg for w in SENTS[1]) >= 3


def test_save_load_roundtrip(tmp_path):
    docs, _ = planted_corpus(n_docs=100, doc_len=25)
    m = topica.FactorialLDA(factor_sizes=[2, 2], seed=5)
    m.fit(docs, iters=120, samples=30)
    p = tmp_path / "flda.bin"
    m.save(str(p))
    loaded = topica.FactorialLDA.load(str(p))
    assert loaded.num_topics == m.num_topics
    assert loaded.factor_sizes == m.factor_sizes
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)
    assert loaded.settings["factor_sizes"] == [2, 2]


def test_settings_roundtrip():
    m = topica.FactorialLDA(factor_sizes=[4, 3], seed=9, delta0=0.2, sparsity=False)
    s = m.settings
    assert s["factor_sizes"] == [4, 3]
    assert s["delta0"] == 0.2
    assert s["sparsity"] is False
    assert s["seed"] == 9


def test_validation_errors():
    with pytest.raises(ValueError):
        topica.FactorialLDA(factor_sizes=[])
    with pytest.raises(ValueError):
        topica.FactorialLDA(factor_sizes=[3, 0])
    with pytest.raises(ValueError):
        topica.FactorialLDA(factor_sizes=[3, 2], sigma_omega=0.0)
    with pytest.raises((ValueError, RuntimeError)):
        topica.FactorialLDA(factor_sizes=[2, 2]).fit([])
