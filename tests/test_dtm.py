"""Dynamic Topic Model (Blei & Lafferty 2006) — topics that evolve over time.

These tests check that a planted drifting topic is tracked across time slices,
that outputs are well-formed and normalized, that fitting is deterministic, and
that the input contract (per-document time slices) is validated.
"""

import numpy as np
import pytest

from topica import DTM, Corpus


def _drift_corpus():
    """Topic A drifts: words {0,1,2} -> {2,3,4} -> {4,5,6} across 3 slices.
    Topic B is stable on {10,11,12}. Returns (docs, times, vocab)."""
    vocab = [f"w{i}" for i in range(20)]
    a = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    b = [10, 11, 12]
    docs, times = [], []
    for s in range(3):
        for _ in range(60):
            docs.append([vocab[i] for i in (a[s] + a[s])])
            times.append(s)
            docs.append([vocab[i] for i in (b + b)])
            times.append(s)
    return docs, times, vocab


@pytest.fixture(scope="module")
def fitted():
    docs, times, _ = _drift_corpus()
    # Looser chain_variance: the planted drift is abrupt, so the topic chain
    # needs room to move between slices.
    m = DTM(num_topics=2, chain_variance=0.5, seed=1)
    m.fit(docs, times, iters=20)
    return m


class TestDrift:
    def _drift_and_stable(self, m):
        # The drifting topic's top word changes between the first and last slice.
        def top(k, t):
            row = m.topic_word(t)[k]
            return int(np.argmax(row))

        drift = next(k for k in range(2) if top(k, 0) != top(k, 2))
        return drift, 1 - drift

    def test_topic_tracks_drift(self, fitted):
        drift, _ = self._drift_and_stable(fitted)
        d0 = fitted.topic_word(0)[drift]
        d2 = fitted.topic_word(2)[drift]
        early = lambda d: d[[0, 1, 2]].sum()  # noqa: E731
        late = lambda d: d[[4, 5, 6]].sum()  # noqa: E731
        # Mass moves from the early block toward the late block over time.
        assert late(d2) > late(d0)
        assert early(d0) > early(d2)

    def test_stable_topic_anchored(self, fitted):
        _, stable = self._drift_and_stable(fitted)
        vocab = fitted.vocabulary
        ids = [vocab.index(w) for w in ("w10", "w11", "w12")]
        for t in range(fitted.num_times):
            top = int(np.argmax(fitted.topic_word(t)[stable]))
            assert top in ids

    def test_word_evolution_monotone_ish(self, fitted):
        drift, _ = self._drift_and_stable(fitted)
        # w4 enters the drifting topic late, so its probability should rise.
        traj = fitted.word_evolution(drift, "w4")
        assert traj.shape == (3,)
        assert traj[2] > traj[0]


class TestOutputs:
    def test_shapes_and_normalization(self, fitted):
        v = len(fitted.vocabulary)
        for t in range(fitted.num_times):
            tw = fitted.topic_word(t)
            assert tw.shape == (2, v)
            np.testing.assert_allclose(tw.sum(axis=1), 1.0, atol=1e-9)

    def test_top_words(self, fitted):
        tw = fitted.top_words(0, 0, n=4)
        assert len(tw) == 4
        assert all(isinstance(w, str) for w in tw)
        tw_w = fitted.top_words(0, 0, n=4, weights=True)
        assert all(isinstance(w, str) and isinstance(p, float) for w, p in tw_w)

    def test_word_evolution_by_id(self, fitted):
        # Accepts an integer word id as well as a string.
        wid = fitted.vocabulary.index("w10")
        traj = fitted.word_evolution(1, wid)
        assert traj.shape == (fitted.num_times,)

    def test_word_drift(self, fitted):
        drift, _ = TestDrift()._drift_and_stable(fitted)
        d = fitted.word_drift(drift, n=5)
        assert set(d) == {"rising", "falling"}
        # "rising" words gained probability, "falling" words lost it.
        assert all(delta > 0 for _, delta in d["rising"])
        assert all(delta < 0 for _, delta in d["falling"])
        # w4 enters the drifting topic late -> it should be among the risers,
        # and match the first-vs-last delta from word_evolution.
        risers = {w: delta for w, delta in d["rising"]}
        assert "w4" in risers
        traj = fitted.word_evolution(drift, "w4")
        assert risers["w4"] == pytest.approx(traj[-1] - traj[0], abs=1e-9)

    def test_word_drift_time_range_and_errors(self, fitted):
        d = fitted.word_drift(0, n=3, from_time=0, to_time=1)
        assert len(d["rising"]) <= 3
        with pytest.raises(ValueError):
            fitted.word_drift(99)            # topic out of range
        with pytest.raises(ValueError):
            fitted.word_drift(0, to_time=99)  # time out of range

    def test_bound_finite(self, fitted):
        assert np.isfinite(fitted.bound)

    def test_doc_topic_shape_and_normalized(self, fitted):
        # Per-document topic proportions (#494): D x K, rows sum to 1.
        docs, _, _ = _drift_corpus()
        dt = fitted.doc_topic
        assert dt.shape == (len(docs), fitted.num_topics)
        np.testing.assert_allclose(dt.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(dt >= 0.0)


class TestApi:
    def test_deterministic(self):
        docs, times, _ = _drift_corpus()
        a = DTM(num_topics=2, chain_variance=0.5, seed=2)
        a.fit(docs, times, iters=12)
        b = DTM(num_topics=2, chain_variance=0.5, seed=2)
        b.fit(docs, times, iters=12)
        for t in range(a.num_times):
            assert np.array_equal(a.topic_word(t), b.topic_word(t))

    def test_spectral_init_is_seed_independent(self):
        # init="spectral" uses the deterministic anchor-word seed, so the fit is
        # reproducible across seeds (no random static-LDA scatter).
        docs, times, _ = _drift_corpus()
        a = DTM(num_topics=2, chain_variance=0.5, seed=2, init="spectral")
        a.fit(docs, times, iters=12)
        b = DTM(num_topics=2, chain_variance=0.5, seed=7, init="spectral")
        b.fit(docs, times, iters=12)
        for t in range(a.num_times):
            assert np.array_equal(a.topic_word(t), b.topic_word(t))

    def test_random_init_default_is_seeded_not_fixed(self):
        # The default (random, gensim-style) seed reproduces for a fixed seed and
        # differs across seeds.
        docs, times, _ = _drift_corpus()
        a = DTM(num_topics=2, chain_variance=0.5, seed=2)
        a.fit(docs, times, iters=12)
        b = DTM(num_topics=2, chain_variance=0.5, seed=2)
        b.fit(docs, times, iters=12)
        c = DTM(num_topics=2, chain_variance=0.5, seed=7)
        c.fit(docs, times, iters=12)
        assert np.array_equal(a.topic_word(0), b.topic_word(0))
        assert not all(
            np.array_equal(a.topic_word(t), c.topic_word(t)) for t in range(a.num_times)
        )

    def test_accepts_corpus_object(self):
        docs, times, _ = _drift_corpus()
        c = Corpus.from_documents(docs)
        m = DTM(num_topics=2, seed=1)
        m.fit(c, times, iters=5)
        assert m.num_times == 3

    def test_times_length_mismatch_raises(self):
        docs, times, _ = _drift_corpus()
        with pytest.raises(ValueError):
            DTM(num_topics=2).fit(docs, times[:-1], iters=2)

    def test_noncontiguous_slices_raise(self):
        docs, times, _ = _drift_corpus()
        bad = [t if t != 1 else 3 for t in times]  # slice 1 empty, slice 3 used
        with pytest.raises(ValueError):
            DTM(num_topics=2).fit(docs, bad, iters=2)

    def test_unfitted_raises(self):
        m = DTM(num_topics=2)
        with pytest.raises(RuntimeError):
            m.topic_word(0)

    def test_single_time_slice(self):
        # num_times == 1 collapses to a static model but must still fit and
        # produce valid outputs (#494).
        docs, _, _ = _drift_corpus()
        times = [0] * len(docs)
        m = DTM(num_topics=2, seed=1)
        m.fit(docs, times, iters=6)
        assert m.num_times == 1
        tw = m.topic_word(0)
        np.testing.assert_allclose(tw.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)

    def test_save_load_roundtrip(self, tmp_path):
        # doc_topic survives a save/load round-trip (#494).
        docs, times, _ = _drift_corpus()
        m = DTM(num_topics=2, chain_variance=0.5, seed=1)
        m.fit(docs, times, iters=8)
        path = str(tmp_path / "dtm.bin")
        m.save(path)
        loaded = DTM.load(path)
        np.testing.assert_array_equal(loaded.doc_topic, m.doc_topic)
        for t in range(m.num_times):
            np.testing.assert_array_equal(loaded.topic_word(t), m.topic_word(t))

    def test_bad_hyperparams_raise(self):
        with pytest.raises(ValueError):
            DTM(num_topics=1)
        with pytest.raises(ValueError):
            DTM(num_topics=2, chain_variance=0.0)
