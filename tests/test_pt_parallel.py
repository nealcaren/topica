"""Pseudo-document Topic Model multi-threaded (AD-LDA) training — num_threads (#566).

PTM's collapsed Gibbs sweep has two phases (resample token topics, then reassign
each document to a pseudo-document). Three count tables are shared across documents
— the topic-word counts (nkw/nk) and the pseudo-doc tables (npk/np/mp, the CRP
customer counts) — while the per-document pseudo-doc assignment l[d] and per-token
topics z[d] are disjoint by document. So the knob follows the dense AD-LDA pattern:
num_threads=1 is the exact serial sweep; num_threads>1 partitions documents across
workers sampling against private clones of ALL three global tables, then reconciles
(nkw/npk/mp additively, nk/np recomputed from the merged rows) and writes each
worker's l/z slice back. num_pseudo (P) is a fixed parameter, so nothing discrete is
thread-dependent.

Contract under test:
- num_threads=1 reproduces byte-for-byte across runs.
- a fixed num_threads>1 is deterministic across runs (topic_word + doc_topic).
- num_threads=0 clamps to serial; fit(num_threads=) overrides the constructor.
- doc_topic rows sum to 1; token/document counts are conserved through the merge.
- settings and save/load carry num_threads.

Recovery quality is not asserted under threading: PTM's (m_p + lambda)
rich-get-richer pseudo-doc aggregation is init/seed-sensitive (#491), and under
AD-LDA workers reassign against a stale mp snapshot, so like every AD-LDA model
num_threads>1 is an exact-merge approximation that trades some mixing accuracy for
speed. Recovery is validated serially in the Rust unit tests and test_pt_gold.py;
these tests pin down the merge correctness, not the model's recovery power.
"""

import numpy as np
import pytest

import topica

K = 3            # topics
P = 8            # pseudo-documents (fixed; P << D)
Vp = 8           # vocabulary words per topic block


def _make_corpus(seed=0, n=160):
    # Short docs (3 tokens) drawn from K disjoint vocabulary blocks — PTM's regime.
    rng = np.random.default_rng(seed)
    docs = []
    for d in range(n):
        b = d % K
        docs.append([f"b{b}w{rng.integers(Vp)}" for _ in range(3)])
    return docs


def _fit(docs, num_threads, seed=42, ctor_threads=None, iters=100):
    ctor = ctor_threads if ctor_threads is not None else num_threads
    m = topica.PT(K, num_pseudo=P, seed=seed, num_threads=ctor)
    kw = {} if ctor_threads is None else {"num_threads": num_threads}
    m.fit(docs, iters=iters, keep_theta_draws=False, **kw)
    return m


def test_serial_is_reproducible():
    docs = _make_corpus()
    a = _fit(docs, 1)
    b = _fit(docs, 1)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_fixed_thread_count_is_deterministic(nt):
    docs = _make_corpus()
    a = _fit(docs, nt)
    b = _fit(docs, nt)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_zero_threads_clamped_to_serial():
    docs = _make_corpus()
    serial = _fit(docs, 1)
    clamped = _fit(docs, 0)
    assert np.array_equal(serial.topic_word, clamped.topic_word)


def test_fit_num_threads_overrides_constructor():
    docs = _make_corpus()
    override = _fit(docs, 4, ctor_threads=1)
    pure4 = _fit(docs, 4)
    assert np.array_equal(override.topic_word, pure4.topic_word)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_parallel_preserves_simplex_and_token_counts(nt):
    docs = _make_corpus()
    m = _fit(docs, nt)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    assert sum(m.doc_lengths) == sum(len(d) for d in docs)


def test_settings_report_num_threads():
    m = topica.PT(K, num_pseudo=P, num_threads=4)
    assert m.settings["num_threads"] == 4


def test_save_load_round_trips_num_threads(tmp_path):
    docs = _make_corpus()
    m = _fit(docs, 4)
    path = str(tmp_path / "pt_threaded.topica")
    m.save(path)
    loaded = topica.PT.load(path)
    assert loaded.settings["num_threads"] == 4
    assert np.array_equal(m.topic_word, loaded.topic_word)
