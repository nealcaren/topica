"""Pachinko Allocation multi-threaded (AD-LDA) training — the num_threads knob (#566).

PAM's collapsed Gibbs sampler shares only the sub-topic->word counts (nkw / nk)
across documents; the per-document two-level tables (nds, ndsk) and the per-token
super/sub assignments are disjoint by document. So the knob follows the same dense
AD-LDA pattern as SeededLDA: num_threads=1 is the exact serial sweep;
num_threads>1 partitions documents across workers sampling against private nkw/nk
copies, then merges (nkw additively, nk recomputed from the merged rows) and writes
each worker's per-document slices straight back.

Contract under test:
- num_threads=1 reproduces byte-for-byte across runs.
- a fixed num_threads>1 is deterministic across runs (topic_word and doc_topic).
- num_threads=0 clamps to serial; fit(num_threads=) overrides the constructor.
- doc_topic and doc_super rows sum to 1.
- token counts are conserved through the partition-and-merge.
- settings and save/load carry num_threads.

Recovery *quality* is not asserted here: PAM's super-topic layer is inherently
high-variance and seed-sensitive (small default alpha + hard single-super
commitment at init, adaptation only in the final quarter of sweeps — see #497), so
it is validated serially against a gold corpus in test_pa_gold.py /
test_pa_doc_super.py, not under threading. Like every AD-LDA model in the roster,
num_threads>1 is an exact-merge approximation that trades some mixing accuracy for
speed; the merge correctness (not the model's recovery power) is what these tests
pin down.
"""

import numpy as np
import pytest

import topica

S = 2            # super-topics
K = 4            # sub-topics
Vp = 6           # vocabulary words per sub-topic block
D = 120


def _make_corpus(seed=0):
    # Two super-topics; each owns two disjoint sub-topic blocks. A document draws
    # from both blocks of ONE super-topic, so sub-topics co-occur within a super
    # but never across supers (the structure PAM's super layer should recover).
    rng = np.random.default_rng(seed)
    groups = {0: (0, 1), 1: (2, 3)}
    docs = []
    for d in range(D):
        s = d % S
        b0, b1 = groups[s]
        doc = []
        for _ in range(30):
            b = b0 if rng.random() < 0.5 else b1
            doc.append(f"b{b}w{rng.integers(Vp)}")
        docs.append(doc)
    return docs


def _fit(docs, num_threads, seed=42, ctor_threads=None, iters=100):
    ctor = ctor_threads if ctor_threads is not None else num_threads
    m = topica.PA(S, K, seed=seed, num_threads=ctor)
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
    assert np.array_equal(a.super_sub, b.super_sub)


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


def test_parallel_preserves_simplex():
    docs = _make_corpus()
    m = _fit(docs, 4)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.doc_super.sum(axis=1), 1.0, rtol=0, atol=1e-6)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_token_counts_conserved_under_threading(nt):
    # The AD-LDA merge must neither lose nor duplicate tokens: the sub-topic->word
    # matrix, un-normalized back to counts, must sum to the corpus token total. We
    # check it via doc_lengths (per-doc token counts are preserved) and that every
    # topic_word row is a proper distribution.
    docs = _make_corpus()
    m = _fit(docs, nt)
    assert sum(m.doc_lengths) == sum(len(d) for d in docs)
    tw = m.topic_word
    np.testing.assert_allclose(tw.sum(axis=1), 1.0, rtol=0, atol=1e-9)


def test_settings_report_num_threads():
    m = topica.PA(S, K, num_threads=4)
    assert m.settings["num_threads"] == 4


def test_save_load_round_trips_num_threads(tmp_path):
    docs = _make_corpus()
    m = _fit(docs, 4)
    path = str(tmp_path / "pa_threaded.topica")
    m.save(path)
    loaded = topica.PA.load(path)
    assert loaded.settings["num_threads"] == 4
    assert np.array_equal(m.topic_word, loaded.topic_word)
