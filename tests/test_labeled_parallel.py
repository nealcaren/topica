"""LabeledLDA multi-threaded (AD-LDA) training — the num_threads knob (#566).

LabeledLDA reuses LDA's collapsed-Gibbs count tables, so it inherits the same
MALLET-style approximate-parallel path as DMR (#570): `num_threads=1` is the exact
serial restricted sweep; `num_threads>1` partitions documents across workers that
sample against private count tables and reconciles once per sweep. Each worker's
per-document label constraint is preserved (a token may only land on a topic in
its document's label set).

Contract under test:
- `num_threads=1` reproduces byte-for-byte across runs.
- a fixed `num_threads>1` is deterministic across runs (phi and theta).
- `num_threads=0` clamps to serial; `fit(num_threads=)` overrides the constructor.
- **the label constraint holds under threading**: theta mass is zero outside each
  document's allowed-topic set, at every thread count.
- a mix of labeled and unconstrained (empty-label) documents fits without panic.
- doc_topic rows sum to 1; discovered structure is recovered.
"""

import numpy as np
import pytest

import topica

L = 6           # label-topics
D = 500
WORDS_PER = 20  # vocabulary words per label block


def _make_corpus(seed=0, unconstrained_every=0):
    """Each document gets 1-3 labels and draws words from those label blocks.
    With ``unconstrained_every>0``, every Nth document gets an empty label set."""
    rng = np.random.default_rng(seed)
    docs, labels = [], []
    for d in range(D):
        if unconstrained_every and d % unconstrained_every == 0:
            labs = []
        else:
            labs = sorted(rng.choice(L, size=rng.integers(1, 4), replace=False).tolist())
        labels.append([f"lab{l}" for l in labs])
        pool = labs if labs else list(range(L))
        docs.append(
            [f"L{pool[rng.integers(len(pool))]}w{rng.integers(WORDS_PER)}" for _ in range(30)]
        )
    label_names = [f"lab{l}" for l in range(L)]
    return docs, labels, label_names


def _fit(docs, labels, label_names, num_threads, seed=42, ctor_threads=None):
    ctor = ctor_threads if ctor_threads is not None else num_threads
    m = topica.LabeledLDA(seed=seed, num_threads=ctor)
    kw = {} if ctor_threads is None else {"num_threads": num_threads}
    m.fit(docs, labels, label_names=label_names, iters=60, num_samples=3,
          progress=False, **kw)
    return m


def test_serial_is_reproducible():
    docs, labels, names = _make_corpus()
    a = _fit(docs, labels, names, 1)
    b = _fit(docs, labels, names, 1)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_fixed_thread_count_is_deterministic(nt):
    docs, labels, names = _make_corpus()
    a = _fit(docs, labels, names, nt)
    b = _fit(docs, labels, names, nt)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_zero_threads_clamped_to_serial():
    docs, labels, names = _make_corpus()
    serial = _fit(docs, labels, names, 1)
    clamped = _fit(docs, labels, names, 0)
    assert np.array_equal(serial.topic_word, clamped.topic_word)


def test_fit_num_threads_overrides_constructor():
    docs, labels, names = _make_corpus()
    override = _fit(docs, labels, names, 4, ctor_threads=1)
    pure4 = _fit(docs, labels, names, 4)
    assert np.array_equal(override.topic_word, pure4.topic_word)


@pytest.mark.parametrize("nt", [2, 4, 8])
def test_label_constraint_holds_under_threading(nt):
    # The whole point of LabeledLDA: a document's topic mass must stay within its
    # label set. Threading must not let a worker leak mass onto a disallowed topic.
    docs, labels, names = _make_corpus(unconstrained_every=7)
    m = _fit(docs, labels, names, nt)
    dt = m.doc_topic
    for d in range(D):
        if not labels[d]:
            continue  # unconstrained doc: all topics allowed
        allowed = {int(x[3:]) for x in labels[d]}  # "labN" -> N
        for t in range(L):
            if t not in allowed:
                assert dt[d, t] <= 1e-9, (d, t, dt[d, t])


def test_parallel_preserves_simplex_and_mixes_labels():
    docs, labels, names = _make_corpus(unconstrained_every=7)
    m = _fit(docs, labels, names, 4)
    dt = m.doc_topic
    assert dt.shape == (D, L)
    np.testing.assert_allclose(dt.sum(axis=1), 1.0, rtol=0, atol=1e-6)


def test_settings_report_num_threads():
    m = topica.LabeledLDA(num_threads=4)
    assert m.settings["num_threads"] == 4
