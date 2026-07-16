"""Ensemble topic modeling: combine independent fits into a consensus.

The central claim (Hoyle et al. 2022, §6) is that pooling topics across runs and
clustering them yields a consensus more reliable than any single run — it beats the
median run and rarely loses to the best. The default ``method="cluster"``
reproduces that procedure; ``method="align"`` is the lighter reference-matching
alternative. Most tests work on hand-built topic-word arrays so the ground truth is
known exactly; a couple exercise the real fitted-model path end to end.
"""

import numpy as np
import pytest

import topica


def _sharp_topics(blocks, V, rng, peak=0.7):
    """K topic-word rows, each concentrating ``peak`` mass on a disjoint block of
    terms and spreading the rest uniformly. ``blocks`` is a list of index lists."""
    K = len(blocks)
    beta = np.full((K, V), (1.0 - peak) / V)
    for k, idx in enumerate(blocks):
        beta[k, idx] += peak / len(idx)
    beta /= beta.sum(axis=1, keepdims=True)
    return beta


def _noisy_run(beta_true, rng, noise, permute=True):
    """A noisy, possibly topic-reordered copy of ``beta_true``."""
    b = beta_true + rng.normal(0, noise, size=beta_true.shape)
    b = np.clip(b, 1e-6, None)
    b /= b.sum(axis=1, keepdims=True)
    if permute:
        b = b[rng.permutation(beta_true.shape[0])]
    return b


def _aligned_error(beta_true, mat):
    """Mean per-topic L1 distance after Hungarian alignment to the truth."""
    pairs = topica.align_topics(beta_true, mat, metric="cosine")
    return float(np.mean([np.abs(beta_true[i] - mat[j]).sum() for i, j, _ in pairs]))


class TestClusterMethod:
    """The default method: Hoyle et al. §6 — pool, cluster, average per cluster."""

    def test_identical_runs_recover_the_topics(self):
        rng = np.random.default_rng(0)
        beta = _sharp_topics([[0, 1], [2, 3], [4, 5]], 8, rng)
        res = topica.ensemble([beta, beta.copy(), beta.copy()], topn=2, lambda_=1.0)
        assert res.method == "cluster"
        assert res.topic_word.shape == beta.shape
        # Each consensus topic equals one input topic (order may differ).
        for _, _, dist in topica.align_topics(beta, res.topic_word):
            assert dist < 1e-9
        np.testing.assert_allclose(res.stability, 1.0)
        np.testing.assert_allclose(res.support, 1.0)
        assert res.reliable.all()
        np.testing.assert_array_equal(res.cluster_sizes, [3, 3, 3])

    def test_ensemble_beats_median_run(self):
        # The Hoyle claim in miniature: the clustered average is closer to the
        # truth than the median individual run.
        rng = np.random.default_rng(7)
        V = 24
        blocks = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18)), list(range(18, 24))]
        beta_true = _sharp_topics(blocks, V, rng, peak=0.6)
        runs = [_noisy_run(beta_true, rng, noise=0.04) for _ in range(15)]

        res = topica.ensemble(runs, topn=6, lambda_=1.0)
        ens_err = _aligned_error(beta_true, res.topic_word)
        run_errs = sorted(_aligned_error(beta_true, r) for r in runs)
        median_err = run_errs[len(run_errs) // 2]
        assert ens_err < median_err

    def test_unstable_topic_is_flagged(self):
        # Two topics are identical across runs; a third shares only its top word and
        # is otherwise random each run. Its cluster is high-support (every run
        # contributes) but low-stability, so it is marked unreliable.
        rng = np.random.default_rng(3)
        V = 18
        A = _sharp_topics([[0, 1, 2]], V, rng, peak=0.8)[0]
        B = _sharp_topics([[3, 4, 5]], V, rng, peak=0.8)[0]
        runs = []
        for _ in range(6):
            r1, r2 = rng.choice(range(7, V), size=2, replace=False)
            wild = np.full(V, 0.01)
            wild[6], wild[r1], wild[r2] = 0.5, 0.2, 0.2
            wild /= wild.sum()
            runs.append(np.vstack([A, B, wild]))
        res = topica.ensemble(runs, topn=3, lambda_=1.0)
        assert int(res.reliable.sum()) == 2
        assert res.stability.min() < 0.5
        # The unstable topic is still high-support: every run fed it a topic.
        assert res.support.min() == pytest.approx(1.0)

    def test_jaccard_distance_also_works(self):
        rng = np.random.default_rng(1)
        beta = _sharp_topics([[0, 1], [2, 3], [4, 5]], 8, rng)
        res = topica.ensemble([beta, beta.copy()], distance="jaccard", topn=2, lambda_=1.0)
        assert res.reliable.all()


class TestAlignMethod:
    """The retained reference-matching alternative: deterministic, exact."""

    def test_identical_runs_reproduce_input(self):
        rng = np.random.default_rng(0)
        beta = _sharp_topics([[0, 1], [2, 3], [4, 5]], 8, rng)
        res = topica.ensemble([beta, beta.copy()], method="align", topn=2)
        assert res.method == "align"
        np.testing.assert_allclose(res.topic_word, beta, atol=1e-9)
        np.testing.assert_allclose(res.stability, 1.0)
        assert res.reliable.all()
        assert res.reference in (0, 1)

    def test_recovers_topic_permutation(self):
        rng = np.random.default_rng(1)
        beta = _sharp_topics([[0, 1], [2, 3], [4, 5]], 8, rng)
        shuffled = beta[[2, 0, 1]]  # same topics, different order
        res = topica.ensemble([beta, shuffled], method="align", reference="first", topn=2)
        np.testing.assert_allclose(res.topic_word, beta, atol=1e-9)

    def test_weights_average_is_weighted(self):
        a = np.array([[0.9, 0.1], [0.1, 0.9]])
        b = np.array([[0.5, 0.5], [0.5, 0.5]])
        res = topica.ensemble([a, b], method="align", reference="first", weights=[0.75, 0.25])
        expected = 0.75 * a + 0.25 * b
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(res.topic_word, expected, atol=1e-9)


class TestStableMethod:
    """gensim EnsembleLda port: discover stable topics, discard noise (no K)."""

    def _clean_runs(self, m=5, K=4, V=30, noise=0.01, seed=0):
        rng = np.random.default_rng(seed)
        protos = np.zeros((K, V))
        for k in range(K):
            protos[k, k * (V // K):(k + 1) * (V // K)] = 1.0
        protos /= protos.sum(1, keepdims=True)
        runs = []
        for _ in range(m):
            b = protos + rng.normal(0, noise, protos.shape)
            b = np.clip(b, 1e-6, None)
            b /= b.sum(1, keepdims=True)
            runs.append(b)
        return runs, protos

    def test_discovers_the_stable_topics(self):
        runs, protos = self._clean_runs()
        res = topica.ensemble(runs, method="stable")
        assert res.method == "stable"
        # Four reproducible prototypes -> four stable topics, each recovered.
        assert res.topic_word.shape[0] == 4
        for _, _, dist in topica.align_topics(protos, res.topic_word):
            assert dist < 5e-3
        assert res.reliable.all()
        np.testing.assert_allclose(res.support, 1.0)

    def test_unstable_topics_are_dropped(self):
        # Three stable prototypes plus, in each run, one purely random topic. The
        # random topics do not recur, so they form no core and are discarded — the
        # ensemble keeps only the three stable topics, not 4.
        rng = np.random.default_rng(2)
        m, V = 6, 30
        protos = np.zeros((3, V))
        for k in range(3):
            protos[k, k * 8:(k + 1) * 8] = 1.0
        protos /= protos.sum(1, keepdims=True)
        runs = []
        for _ in range(m):
            junk = rng.random(V)
            junk /= junk.sum()
            b = np.vstack([protos + rng.normal(0, 0.01, protos.shape), junk])
            b = np.clip(b, 1e-6, None)
            b /= b.sum(1, keepdims=True)
            runs.append(b)
        res = topica.ensemble(runs, method="stable")
        assert res.topic_word.shape[0] == 3

    def test_no_stable_topic_warns_and_returns_empty(self):
        # Every topic in every run is random noise: nothing recurs, so no stable
        # topic exists.
        rng = np.random.default_rng(5)
        runs = []
        for _ in range(4):
            b = rng.random((3, 20))
            b /= b.sum(1, keepdims=True)
            runs.append(b)
        with pytest.warns(UserWarning, match="no stable topic"):
            res = topica.ensemble(runs, method="stable", eps=0.05)
        assert res.topic_word.shape == (0, 20)
        assert np.isnan(res.agreement)

    def test_bad_masking_rejected(self):
        runs, _ = self._clean_runs()
        with pytest.raises(ValueError, match="masking must be"):
            topica.ensemble(runs, method="stable", masking="soft")


class TestApiSurface:
    def test_accepts_select_model_result(self):
        runs = [np.eye(3)[[0, 1, 2]] + 0.01, np.eye(3)[[1, 0, 2]] + 0.01]
        runs = [r / r.sum(axis=1, keepdims=True) for r in runs]

        class _FakeSelect:
            models = runs

        res = topica.ensemble(_FakeSelect(), lambda_=1.0)
        assert res.n_runs == 2

    def test_top_words_pairs_use_indices_without_vocab(self):
        rng = np.random.default_rng(0)
        beta = _sharp_topics([[0, 1], [5, 6]], 8, rng)
        res = topica.ensemble([beta, beta.copy()], topn=2, lambda_=1.0)
        tw = res.top_words(2)  # list of [(term, prob), ...] per topic; terms are ints
        terms = {t for t, _ in tw[0]}
        assert terms == {0, 1} or terms == {5, 6}
        assert all(isinstance(p, float) for _, p in tw[0])

    def test_repr_reports_method_and_reliability(self):
        rng = np.random.default_rng(0)
        beta = _sharp_topics([[0, 1], [2, 3]], 6, rng)
        res = topica.ensemble([beta, beta.copy()], topn=2, lambda_=1.0)
        r = repr(res)
        assert "method='cluster'" in r
        assert "reliable=2/2" in r

    def test_missing_doc_topic_warns_and_falls_back(self):
        rng = np.random.default_rng(0)
        beta = _sharp_topics([[0, 1], [2, 3]], 6, rng)
        with pytest.warns(UserWarning, match="document-topic distance is"):
            topica.ensemble([beta, beta.copy()], topn=2, lambda_=0.5)


class TestErrors:
    def test_single_run_rejected(self):
        with pytest.raises(ValueError, match="at least two runs"):
            topica.ensemble([np.eye(3)])

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            topica.ensemble([np.ones((3, 5)), np.ones((3, 6))], lambda_=1.0)

    def test_bad_method_rejected(self):
        b = np.ones((2, 4)) / 4
        with pytest.raises(ValueError, match="method must be"):
            topica.ensemble([b, b], method="magic", lambda_=1.0)

    def test_bad_distance_rejected(self):
        b = np.ones((2, 4)) / 4
        with pytest.raises(ValueError, match="distance must be"):
            topica.ensemble([b, b], distance="euclidean", lambda_=1.0)

    def test_bad_lambda_rejected(self):
        b = np.ones((2, 4)) / 4
        with pytest.raises(ValueError, match="lambda_"):
            topica.ensemble([b, b], lambda_=2.0)

    def test_bad_reference_rejected(self):
        b = np.ones((2, 4)) / 4
        with pytest.raises(ValueError, match="reference"):
            topica.ensemble([b, b], method="align", reference="best")
        with pytest.raises(ValueError, match="out of range"):
            topica.ensemble([b, b], method="align", reference=5)

    def test_bad_weights_rejected(self):
        b = np.ones((2, 4)) / 4
        with pytest.raises(ValueError, match="length"):
            topica.ensemble([b, b], method="align", weights=[1.0])
        with pytest.raises(ValueError, match="non-negative"):
            topica.ensemble([b, b], method="align", weights=[-1.0, 2.0])


class TestFittedModels:
    """End-to-end on real LDA fits: doc-topic averaging and the analysis surface."""

    def _runs(self, n=4):
        rng = np.random.default_rng(0)
        A = ["cat", "dog", "pet", "kitten", "puppy", "vet"]
        B = ["star", "moon", "sky", "sun", "comet", "orbit"]
        docs = []
        for _ in range(80):
            v = A if rng.random() < 0.5 else B
            docs.append([v[int(rng.integers(len(v)))] for _ in range(10)])
        runs = []
        for s in range(n):
            m = topica.models.LDA(num_topics=2, seed=s + 1)
            m.fit(docs, iters=300)
            runs.append(m)
        return runs, docs

    def test_doc_topic_averaged_for_same_docs(self):
        runs, docs = self._runs()
        res = topica.ensemble(runs)  # default cluster, lambda_=0.5 uses theta
        assert res.doc_topic is not None
        assert res.doc_topic.shape == (len(docs), 2)
        np.testing.assert_allclose(res.doc_topic.sum(axis=1), 1.0, atol=1e-6)
        assert res.vocabulary is not None

    def test_result_flows_into_coherence(self):
        runs, docs = self._runs()
        res = topica.ensemble(runs)
        # The ensemble duck-types as a model: the model-neutral coherence surface
        # accepts it directly.
        cv = topica.coherence(res, docs, coherence_type="c_v", topn=5)
        assert np.asarray(cv).shape == (2,)
        assert np.all(np.isfinite(cv))
