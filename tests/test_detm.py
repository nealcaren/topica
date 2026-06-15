"""Tests for the Dynamic Embedded Topic Model (DETM)."""

import numpy as np
import pytest

import topica


def _planted_corpus(seed=0, k=3, block=6, t=4, d_per_t=20, length=30):
    """A small time-stamped corpus over K disjoint word blocks, with topic 0
    rising and topic K-1 falling across the slices. Returns (docs, vocab,
    embeddings, times)."""
    rng = np.random.default_rng(seed)
    v = k * block
    vocab = [f"w{i}" for i in range(v)]
    # word embeddings: word in block b points along axis b.
    el = k + 2
    emb = (rng.standard_normal((v, el)) * 0.1)
    for w in range(v):
        emb[w, w // block] += 3.0
    docs, times = [], []
    for tt in range(t):
        frac = tt / max(t - 1, 1)
        base = np.full(k, 0.25)
        base[0] = 0.4 * frac + 0.1
        base[k - 1] = 0.4 * (1 - frac) + 0.1
        base /= base.sum()
        for _ in range(d_per_t):
            toks = []
            for _ in range(length):
                kk = rng.choice(k, p=base)
                w = kk * block + rng.integers(block)
                toks.append(vocab[w])
            docs.append(toks)
            times.append(tt)
    return docs, vocab, emb, np.array(times)


def test_fit_shapes_and_distributions():
    docs, vocab, emb, times = _planted_corpus()
    k, t, v = 3, 4, len(vocab)
    m = topica.DETM(k, delta=0.005, hidden_size=32, lr=0.02, seed=42)
    m.fit(docs, emb, vocab, times=times, iters=30)

    tw = np.asarray(m.topic_word)
    assert tw.shape == (k, v)
    np.testing.assert_allclose(tw.sum(1), 1.0, atol=1e-6)

    bot = np.asarray(m.beta_over_time)
    assert bot.shape == (t, k, v)
    # every (t, k) row of beta is a distribution.
    np.testing.assert_allclose(bot.sum(2), 1.0, atol=1e-6)
    # topic_word_over_time is the same tensor.
    np.testing.assert_array_equal(bot, np.asarray(m.topic_word_over_time))

    dt = np.asarray(m.doc_topic)
    assert dt.shape == (len(docs), k)
    np.testing.assert_allclose(dt.sum(1), 1.0, atol=1e-6)

    assert np.asarray(m.eta).shape == (t, k)
    assert np.asarray(m.alpha).shape[:2] == (t, k)
    assert m.num_times == t


def test_topic_word_at_and_top_words_at():
    docs, vocab, emb, times = _planted_corpus()
    k = 3
    m = topica.DETM(k, hidden_size=32, lr=0.02, seed=42)
    m.fit(docs, emb, vocab, times=times, iters=20)

    tw0 = np.asarray(m.topic_word_at(0))
    assert tw0.shape == (k, len(vocab))
    np.testing.assert_allclose(tw0.sum(1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(tw0, np.asarray(m.beta_over_time)[0])

    tops = m.top_words_at(0, 5)
    assert len(tops) == k
    assert all(len(words) == 5 for words in tops)
    one = m.top_words_at(0, 5, topic=1)
    assert len(one) == 5

    with pytest.raises(Exception):
        m.topic_word_at(99)


def test_determinism_bit_for_bit():
    docs, vocab, emb, times = _planted_corpus(seed=1)
    k = 3
    a = topica.DETM(k, hidden_size=16, lr=0.02, seed=7)
    a.fit(docs, emb, vocab, times=times, iters=25)
    b = topica.DETM(k, hidden_size=16, lr=0.02, seed=7)
    b.fit(docs, emb, vocab, times=times, iters=25)
    np.testing.assert_array_equal(np.asarray(a.topic_word), np.asarray(b.topic_word))
    np.testing.assert_array_equal(np.asarray(a.doc_topic), np.asarray(b.doc_topic))
    np.testing.assert_array_equal(np.asarray(a.beta_over_time), np.asarray(b.beta_over_time))
    np.testing.assert_array_equal(np.asarray(a.eta), np.asarray(b.eta))


def test_times_alias_timestamps():
    docs, vocab, emb, times = _planted_corpus()
    k = 3
    a = topica.DETM(k, hidden_size=16, seed=42)
    a.fit(docs, emb, vocab, times=times, iters=10)
    b = topica.DETM(k, hidden_size=16, seed=42)
    b.fit(docs, emb, vocab, timestamps=times, iters=10)
    np.testing.assert_array_equal(np.asarray(a.topic_word), np.asarray(b.topic_word))
    # passing both is an error
    c = topica.DETM(k, seed=42)
    with pytest.raises(Exception):
        c.fit(docs, emb, vocab, times=times, timestamps=times, iters=5)
    # passing neither is an error
    d = topica.DETM(k, seed=42)
    with pytest.raises(Exception):
        d.fit(docs, emb, vocab, iters=5)


def test_temporal_drift_recovered():
    # topic 0 prevalence rises, topic K-1 falls; the fitted eta prior should move
    # across the slices, recovering the planted drift.
    docs, vocab, emb, times = _planted_corpus(seed=3, k=3, block=8, t=5, d_per_t=40)
    k, t = 3, 5
    m = topica.DETM(k, delta=0.005, hidden_size=32, lr=0.02, seed=42)
    m.fit(docs, emb, vocab, times=times, iters=120)
    eta = np.asarray(m.eta)
    prior = np.exp(eta - eta.max(1, keepdims=True))
    prior = prior / prior.sum(1, keepdims=True)
    drift = np.abs(prior[-1] - prior[0]).max()
    assert drift > 0.03, f"eta prior did not drift across time (max {drift})"


def test_save_load_roundtrip(tmp_path):
    docs, vocab, emb, times = _planted_corpus()
    k = 3
    m = topica.DETM(k, hidden_size=16, seed=42)
    m.fit(docs, emb, vocab, times=times, iters=15)
    p = tmp_path / "detm.bin"
    m.save(str(p))
    loaded = topica.DETM.load(str(p))
    np.testing.assert_array_equal(np.asarray(m.topic_word), np.asarray(loaded.topic_word))
    np.testing.assert_array_equal(np.asarray(m.beta_over_time), np.asarray(loaded.beta_over_time))
    np.testing.assert_array_equal(np.asarray(m.doc_topic), np.asarray(loaded.doc_topic))
    np.testing.assert_array_equal(np.asarray(m.eta), np.asarray(loaded.eta))
    assert loaded.num_times == m.num_times


@pytest.mark.skipif(
    not __import__("os").path.exists("/private/tmp/detm_gold.npz"),
    reason="gold standard npz not present",
)
def test_gold_standard_smoke_parity():
    """Smoke parity against the reference gold standard: fit on the identical
    inputs and require the Hungarian-aligned topic-word cosine to clear a floor
    calibrated to the reference's own seed-to-seed variation (~0.81)."""
    d = np.load("/private/tmp/detm_gold.npz", allow_pickle=True)
    doc_tokens, doc_counts = d["doc_tokens"], d["doc_counts"]
    times = d["times"].astype(int)
    emb = d["embeddings"].astype(float)
    vocab = [str(w) for w in d["vocab"]]
    k = int(d["K"])
    docs = []
    for tok, cnt in zip(doc_tokens, doc_counts):
        toks = []
        for w, c in zip(np.atleast_1d(tok), np.atleast_1d(cnt)):
            toks += [vocab[int(w)]] * int(c)
        docs.append(toks)

    m = topica.DETM(k, delta=float(d["delta"]), hidden_size=64, lr=0.01, seed=42)
    m.fit(docs, emb, vocab, times=times, iters=400)

    a_tw = np.asarray(m.topic_word)
    b_tw = d["topic_word"]
    an = a_tw / (np.linalg.norm(a_tw, axis=1, keepdims=True) + 1e-12)
    bn = b_tw / (np.linalg.norm(b_tw, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-sim)
    mean_cos = sim[rows, cols].mean()
    # The reference seed-to-seed cosine floor on this corpus is ~0.81; require we
    # land in that regime (allowing slack for the stochastic VAE).
    assert mean_cos > 0.7, f"aligned topic-word cosine {mean_cos:.3f} below floor"
