"""SeededLDA multi-threaded (AD-LDA) training — the num_threads knob (#566).

SeededLDA uses dense count tables (nkw / nk / ndk), so unlike DMR/LabeledLDA it
cannot reuse the packed reconcile helper — it has its own dense AD-LDA merge. The
knob behaves the same: num_threads=1 is the exact serial seeded-Gibbs sweep;
num_threads>1 partitions documents across workers that sample against private
count tables and reconciles once per sweep. The seeded asymmetric prior beta_{k,w}
is a fixed pseudocount that never enters nkw, so it does not affect the merge.

Contract under test:
- num_threads=1 reproduces byte-for-byte across runs.
- a fixed num_threads>1 is deterministic across runs (phi and theta).
- num_threads=0 clamps to serial; fit(num_threads=) overrides the constructor.
- the seeding still works under threading: each seed word is top-weighted in its
  own topic.
- doc_topic rows sum to 1.
"""

import numpy as np
import pytest

import topica

L = 3            # seeded topics
Vp = 20          # vocabulary words per block
D = 480


def _make_corpus(seed=0):
    rng = np.random.default_rng(seed)
    docs = [[f"b{d % L}w{rng.integers(Vp)}" for _ in range(30)] for d in range(D)]
    seed_words = {f"topic{b}": [f"b{b}w0"] for b in range(L)}
    return docs, seed_words


def _fit(docs, seed_words, num_threads, seed=42, ctor_threads=None):
    ctor = ctor_threads if ctor_threads is not None else num_threads
    m = topica.SeededLDA(seed_words, seed=seed, num_threads=ctor)
    kw = {} if ctor_threads is None else {"num_threads": num_threads}
    m.fit(docs, iters=80, keep_theta_draws=False, **kw)
    return m


def test_serial_is_reproducible():
    docs, sw = _make_corpus()
    a = _fit(docs, sw, 1)
    b = _fit(docs, sw, 1)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_fixed_thread_count_is_deterministic(nt):
    docs, sw = _make_corpus()
    a = _fit(docs, sw, nt)
    b = _fit(docs, sw, nt)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_zero_threads_clamped_to_serial():
    docs, sw = _make_corpus()
    serial = _fit(docs, sw, 1)
    clamped = _fit(docs, sw, 0)
    assert np.array_equal(serial.topic_word, clamped.topic_word)


def test_fit_num_threads_overrides_constructor():
    docs, sw = _make_corpus()
    override = _fit(docs, sw, 4, ctor_threads=1)
    pure4 = _fit(docs, sw, 4)
    assert np.array_equal(override.topic_word, pure4.topic_word)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_seeding_holds_under_threading(nt):
    # The point of SeededLDA: each seed word steers its topic. Threading must not
    # break that — the seed word bXw0 must have its highest probability in its own
    # seeded topic X (topics are created in seed_words insertion order). This is
    # the same "seeds steer topics" invariant the Rust unit test checks.
    docs, sw = _make_corpus()
    m = _fit(docs, sw, nt)
    tw = m.topic_word  # (K, V)
    idx = {w: i for i, w in enumerate(m.vocabulary)}
    for b in range(L):
        wi = idx[f"b{b}w0"]
        assert int(np.argmax(tw[:, wi])) == b, (b, tw[:, wi].tolist())


def test_parallel_preserves_simplex():
    docs, sw = _make_corpus()
    m = _fit(docs, sw, 4)
    dt = m.doc_topic
    np.testing.assert_allclose(dt.sum(axis=1), 1.0, rtol=0, atol=1e-6)


def test_settings_report_num_threads():
    _, sw = _make_corpus()
    m = topica.SeededLDA(sw, num_threads=4)
    assert m.settings["num_threads"] == 4
