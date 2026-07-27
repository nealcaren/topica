"""BTM multi-threaded (AD-LDA) training — the num_threads knob (#566).

BTM's sampling unit is the biterm (a within-window word pair), not the document,
and it keeps no per-document state during fit. So its AD-LDA path partitions the
flat biterm array across workers that sample against private nb_z/nwz count
tables, reconciling both once per sweep. num_threads=1 is the exact serial sweep.

Contract under test:
- num_threads=1 reproduces byte-for-byte across runs.
- a fixed num_threads>1 is deterministic across runs (topic_word and theta).
- num_threads=0 clamps to serial; fit(num_threads=) overrides the constructor.
- topic recovery holds under threading: on a planted block corpus each block's
  vocabulary concentrates in a single, distinct topic.
"""

import numpy as np
import pytest

import topica

L = 3            # planted blocks / topics
Vp = 15          # vocabulary words per block
D = 720


def _make_corpus(seed=0):
    rng = np.random.default_rng(seed)
    # Short texts (6 tokens) drawn from a single block — BTM's target regime.
    return [[f"b{d % L}w{rng.integers(Vp)}" for _ in range(6)] for d in range(D)]


def _fit(docs, num_threads, seed=42, ctor_threads=None):
    ctor = ctor_threads if ctor_threads is not None else num_threads
    m = topica.BTM(L, seed=seed, iters=80, num_threads=ctor)
    kw = {} if ctor_threads is None else {"num_threads": num_threads}
    m.fit(docs, **kw)
    return m


def test_serial_is_reproducible():
    docs = _make_corpus()
    a = _fit(docs, 1)
    b = _fit(docs, 1)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.theta, b.theta)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_fixed_thread_count_is_deterministic(nt):
    docs = _make_corpus()
    a = _fit(docs, nt)
    b = _fit(docs, nt)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.theta, b.theta)
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
def test_topic_recovery_under_threading(nt):
    docs = _make_corpus()
    m = _fit(docs, nt)
    tw = m.topic_word  # (K, V)
    idx = {w: i for i, w in enumerate(m.vocabulary)}
    # Each block's total topic_word mass should peak in one topic, and the three
    # blocks should map to three distinct topics.
    winners = []
    for blk in range(L):
        wids = [idx[f"b{blk}w{i}"] for i in range(Vp) if f"b{blk}w{i}" in idx]
        winners.append(int(np.argmax(tw[:, wids].sum(axis=1))))
    assert len(set(winners)) == L, winners


def test_settings_report_num_threads():
    m = topica.BTM(L, num_threads=4)
    assert m.settings["num_threads"] == 4
