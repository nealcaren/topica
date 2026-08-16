"""keyATM multi-threaded fitting. The token sweep uses approximate distributed
Gibbs (AD-LDA): documents are partitioned across `num_threads` workers and the
topic-word counts are reconciled each sweep. These check that threading recovers
the same topic structure and is deterministic for a fixed seed and worker count."""

import numpy as np
import pytest

import topica

A = ["tax", "market", "trade", "fiscal", "budget", "deficit"]
B = ["abortion", "gay", "church", "family", "prayer", "faith"]
SEEDS = {"econ": A[:3], "soc": B[:3]}


def _corpus(seed=0, n=600):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        heavy, light = (A, B) if i % 2 else (B, A)
        docs.append(rng.choice(heavy, 12).tolist() + rng.choice(light, 3).tolist())
    return docs


def _econ_topic(model):
    """Index of the topic whose top words are economic."""
    for k in range(2):
        tops = {w for w, _ in model.top_words(4, topic=k, weights=True)}
        if len(tops & set(A)) >= 2:
            return k
    return None


def test_parallel_recovers_topics():
    docs = _corpus()
    for nt in (2, 4):
        m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
        m.fit(docs, iters=300, num_threads=nt)
        assert _econ_topic(m) is not None, f"econ topic not recovered with {nt} threads"


def test_parallel_deterministic():
    docs = _corpus()
    m1 = topica.KeyATM(SEEDS, num_topics=2, seed=7)
    m1.fit(docs, iters=200, num_threads=4)
    m2 = topica.KeyATM(SEEDS, num_topics=2, seed=7)
    m2.fit(docs, iters=200, num_threads=4)
    assert np.allclose(m1.topic_word, m2.topic_word)
    assert np.allclose(m1.doc_topic, m2.doc_topic)


def test_parallel_counts_stay_nonnegative():
    # The AD-LDA reconcile must not drive any topic-word probability negative.
    docs = _corpus(seed=2)
    m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=250, num_threads=8)
    phi = m.topic_word
    assert (phi >= 0).all()
    assert np.allclose(phi.sum(axis=1), 1.0)


def test_parallel_merge_higher_k_deterministic_and_valid():
    # The sparse topic-word reconcile is parallel over topic rows, so exercise it
    # with K well above the worker count to cover the multi-row merge path. It
    # must stay deterministic for a fixed (num_threads, seed) and produce a valid
    # (non-negative, row-normalized) topic-word matrix.
    docs = _corpus(seed=3, n=800)
    for nt in (4, 8):
        a = topica.KeyATM(SEEDS, num_topics=24, seed=11)
        a.fit(docs, iters=120, num_threads=nt)
        b = topica.KeyATM(SEEDS, num_topics=24, seed=11)
        b.fit(docs, iters=120, num_threads=nt)
        assert np.array_equal(a.topic_word, b.topic_word), (
            f"non-deterministic topic_word at num_threads={nt}"
        )
        assert (a.topic_word >= 0).all()
        assert np.allclose(a.topic_word.sum(axis=1), 1.0)


def test_parallel_dynamic_recovers_change_point():
    # Threading composes with the dynamic model.
    rng = np.random.default_rng(0)
    docs, years = [], []
    for t in range(12):
        soc_share = 0.15 if t < 6 else 0.85
        for _ in range(60):
            heavy = B if rng.random() < soc_share else A
            light = A if heavy is B else B
            docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
            years.append(2000 + t)
    m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, timestamps=years, num_states=2, iters=300, num_threads=4)
    si = m.topic_names.index("soc")
    tp = m.time_prevalence[:, si]
    assert tp[6:].mean() - tp[:6].mean() > 0.25
