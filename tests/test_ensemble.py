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

    def test_rank_masking_keeps_a_term_on_small_vocab(self):
        # Deliberate divergence from gensim (#626): gensim's rank_masking keeps
        # int(V * threshold) terms, which is 0 for any V < 10 at the default 0.11,
        # collapsing every distance to 1.0. topica keeps >= 1 term, so rank masking
        # still discriminates topics on a small vocabulary.
        from topica.ensemble import _rank_mask

        a = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])  # V = 8
        assert int(_rank_mask(a, None).sum()) >= 1
        assert int(_rank_mask(a, 1.0).sum()) >= 1  # threshold 1.0 must not crash

        # ...but keep gensim's strict-greater rule where it is non-empty: a sparse
        # row (5 nonzero terms in a 100-word vocab) must select the 5 real terms,
        # not explode to the whole vocabulary on the tie at 0.0.
        sparse = np.zeros(100)
        sparse[:5] = 0.2
        assert int(_rank_mask(sparse, 0.11).sum()) == 5

        # End-to-end: four reproducible prototypes over a 9-term vocab still cluster
        # under rank masking, where gensim's empty mask would find nothing.
        rng = np.random.default_rng(0)
        V = 9
        protos = np.zeros((4, V))
        for k in range(4):
            protos[k, k * 2:k * 2 + 2] = 1.0
        protos /= protos.sum(1, keepdims=True)
        runs = []
        for _ in range(5):
            b = np.clip(protos + rng.normal(0, 0.01, protos.shape), 1e-6, None)
            b /= b.sum(1, keepdims=True)
            runs.append(b)
        res = topica.ensemble(runs, method="stable", masking="rank")
        assert res.topic_word.shape[0] >= 1

    def test_isolated_core_is_validated_regardless_of_scan_order(self):
        # Deliberate divergence from gensim (#628): a core whose neighbors are all
        # non-core keeps its own label in neighboring_labels, so it is counted as
        # an isolated core and its cluster is validated -- gensim leaves that set
        # empty and drops the cluster order-dependently.
        from topica.ensemble import _cbdbscan, _stable_labels

        # Chain 0-1-2: 0<->1 and 1<->2 within eps, 0<->2 outside. Only topic 1 has
        # two neighbors, so it is the sole core; both its neighbors are non-core.
        base = np.array([[0.0, 0.05, 0.20], [0.05, 0.0, 0.05], [0.20, 0.05, 0.0]])
        # Same graph under every relabeling of the topics: the validated cluster
        # must not depend on which index seeds the scan.
        for perm in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [2, 0, 1]):
            D = base[np.ix_(perm, perm)]
            results = _cbdbscan(D, eps=0.1, min_samples=2)
            core = next(i for i, r in enumerate(results) if r.is_core)
            assert results[core].neighboring_labels == {results[core].label}
            # The lone core survives cluster validation at min_cores=1.
            stable = _stable_labels(results, num_models=3, min_cores=1)
            assert results[core].label in stable


class TestBootstrapCI:
    """#627: bootstrap CIs for agreement / stability (SEs-everywhere)."""

    def _noisy_runs(self, m=6, K=4, V=12, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        protos = np.zeros((K, V))
        for k in range(K):
            protos[k, 3 * k:3 * k + 3] = 1.0
        protos /= protos.sum(1, keepdims=True)
        runs = []
        for _ in range(m):
            b = np.clip(protos + rng.normal(0, noise, protos.shape), 1e-6, None)
            runs.append(b / b.sum(1, keepdims=True))
        return runs

    def test_off_by_default(self):
        res = topica.ensemble(self._noisy_runs())
        assert res.agreement_ci is None and res.agreement_se is None
        assert res.stability_ci is None

    @pytest.mark.parametrize("method", ["cluster", "align", "stable"])
    def test_agreement_ci_is_sane(self, method):
        runs = self._noisy_runs()
        res = topica.ensemble(runs, method=method, n_boot=150, boot_seed=1)
        lo, hi = res.agreement_ci
        assert 0.0 <= lo <= res.agreement <= hi <= 1.0   # centered, clipped, contains
        assert res.agreement_se >= 0.0

    def test_per_topic_ci_only_for_fixed_k(self):
        runs = self._noisy_runs()
        for method in ("cluster", "align"):
            res = topica.ensemble(runs, method=method, n_boot=100, boot_seed=2)
            assert res.stability_ci.shape == (res.topic_word.shape[0], 2)
            for i in range(len(res.stability)):
                lo, hi = res.stability_ci[i]
                if not np.isnan(lo):
                    assert lo <= res.stability[i] <= hi
        # stable has a variable topic count, so only the scalar agreement CI
        assert topica.ensemble(runs, method="stable", n_boot=100).stability_ci is None

    def test_bootstrap_is_deterministic(self):
        runs = self._noisy_runs()
        a = topica.ensemble(runs, n_boot=100, boot_seed=7)
        b = topica.ensemble(runs, n_boot=100, boot_seed=7)
        assert a.agreement_ci == b.agreement_ci
        assert np.array_equal(a.stability_ci, b.stability_ci, equal_nan=True)
        # a different seed generally gives a (slightly) different interval
        c = topica.ensemble(runs, n_boot=100, boot_seed=8)
        assert a.agreement_se >= 0 and c.agreement_se >= 0

    def test_negative_n_boot_raises(self):
        with pytest.raises(ValueError, match="n_boot must be"):
            topica.ensemble(self._noisy_runs(), n_boot=-1)


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
            m = topica.LDA(num_topics=2, seed=s + 1)
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
