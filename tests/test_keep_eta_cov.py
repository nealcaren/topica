"""Tests for CTM/STM keep_eta_cov=False memory-saving option.

Issue #160: add keep_eta_cov: bool = True to STM and CTM fit.  When False,
skip storing the per-document variational covariance (nu) to save O(N·K²)
memory, but keep the fit bit-identical. _recompute_eta_cov() regenerates nu
on demand; posterior_theta_samples falls back to it automatically.
"""

import numpy as np
import pytest
import topica


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _tiny_docs():
    """Small reproducible corpus of 60 docs over a 15-word vocabulary."""
    rng = np.random.default_rng(7)
    vocab = [[f"w{i}" for i in range(15)] for _ in range(60)]
    # Each doc draws 8-12 words uniformly from the vocabulary.
    docs = []
    for _ in range(60):
        n = rng.integers(8, 13)
        tokens = rng.choice(vocab[0], size=n, replace=True).tolist()
        docs.append(tokens)
    return docs


DOCS = _tiny_docs()


# ---------------------------------------------------------------------------
# (a) fit bit-identity: keep_eta_cov=True vs False produce equal doc_topic/bound
# ---------------------------------------------------------------------------

class TestFitBitIdentical:
    def test_ctm_doc_topic_equal(self):
        """CTM fit with keep_eta_cov=False is bit-for-bit identical to True."""
        m_keep = topica.models.CTM(3, init="random", seed=0)
        m_keep.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.CTM(3, init="random", seed=0)
        m_drop.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=False)

        np.testing.assert_array_equal(
            np.asarray(m_keep.doc_topic),
            np.asarray(m_drop.doc_topic),
            err_msg="doc_topic differs between keep_eta_cov=True and False",
        )

    def test_ctm_bound_equal(self):
        m_keep = topica.models.CTM(3, init="random", seed=0)
        m_keep.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.CTM(3, init="random", seed=0)
        m_drop.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=False)

        assert m_keep.bound == m_drop.bound, (
            f"bound differs: {m_keep.bound} vs {m_drop.bound}"
        )


# ---------------------------------------------------------------------------
# (b) eta_cov getter raises RuntimeError when keep_eta_cov=False
# ---------------------------------------------------------------------------

class TestEtaCovGetterRaises:
    def test_ctm_eta_cov_raises(self):
        m = topica.models.CTM(3, init="random", seed=1)
        m.fit(DOCS, iters=3, convergence_tol=0, keep_eta_cov=False)
        with pytest.raises(RuntimeError, match="keep_eta_cov"):
            _ = m.eta_cov

    def test_stm_eta_cov_raises(self):
        X = np.ones((len(DOCS), 1))
        m = topica.models.STM(3, init="random", seed=1)
        m.fit(DOCS, prevalence=X, iters=3, convergence_tol=0, keep_eta_cov=False)
        with pytest.raises(RuntimeError, match="keep_eta_cov"):
            _ = m.eta_cov


# ---------------------------------------------------------------------------
# (c) _recompute_eta_cov returns same result as stored eta_cov (for CTM)
# ---------------------------------------------------------------------------

class TestRecomputeEtaCov:
    def test_ctm_recompute_equals_stored(self):
        """_recompute_eta_cov output must equal what keep_eta_cov=True stores."""
        m_keep = topica.models.CTM(3, init="random", seed=2)
        m_keep.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.CTM(3, init="random", seed=2)
        m_drop.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=False)

        stored = np.asarray(m_keep.eta_cov, dtype=np.float32)
        recomputed = np.asarray(m_drop._recompute_eta_cov(), dtype=np.float32)

        np.testing.assert_array_equal(
            stored,
            recomputed,
            err_msg="_recompute_eta_cov does not match stored eta_cov",
        )

    def test_ctm_recompute_shape(self):
        m = topica.models.CTM(4, init="random", seed=3)
        m.fit(DOCS, iters=3, convergence_tol=0, keep_eta_cov=False)
        cov = m._recompute_eta_cov()
        d = len(DOCS)
        km1 = 3
        assert cov.shape == (d, km1, km1), (
            f"_recompute_eta_cov shape {cov.shape} != ({d}, {km1}, {km1})"
        )

    def test_stm_recompute_shape(self):
        X = np.ones((len(DOCS), 1))
        m = topica.models.STM(3, init="random", seed=4)
        m.fit(DOCS, prevalence=X, iters=3, convergence_tol=0, keep_eta_cov=False)
        cov = m._recompute_eta_cov()
        d = len(DOCS)
        km1 = 2
        assert cov.shape == (d, km1, km1), (
            f"STM _recompute_eta_cov shape {cov.shape} != ({d}, {km1}, {km1})"
        )


# ---------------------------------------------------------------------------
# (d) posterior_theta_samples falls back to _recompute_eta_cov automatically
# ---------------------------------------------------------------------------

class TestPosteriorThetaSamplesFallback:
    def test_ctm_fallback_runs(self):
        """posterior_theta_samples should not raise when keep_eta_cov=False."""
        from topica.stm import posterior_theta_samples

        m = topica.models.CTM(3, init="random", seed=5)
        m.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=False)

        draws = posterior_theta_samples(m, nsims=10, seed=0)
        d = len(DOCS)
        assert draws.shape == (10, d, 3), (
            f"posterior_theta_samples shape {draws.shape} != (10, {d}, 3)"
        )
        # All theta rows should sum to 1.
        np.testing.assert_allclose(
            draws.sum(axis=2),
            np.ones((10, d)),
            atol=1e-5,
            err_msg="theta draws do not sum to 1",
        )

    def test_ctm_fallback_matches_stored(self):
        """Same seed should give equal draws whether eta_cov was kept or recomputed."""
        from topica.stm import posterior_theta_samples

        m_keep = topica.models.CTM(3, init="random", seed=6)
        m_keep.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.CTM(3, init="random", seed=6)
        m_drop.fit(DOCS, iters=5, convergence_tol=0, keep_eta_cov=False)

        draws_keep = posterior_theta_samples(m_keep, nsims=20, seed=99)
        draws_drop = posterior_theta_samples(m_drop, nsims=20, seed=99)

        np.testing.assert_allclose(
            draws_keep,
            draws_drop,
            atol=1e-5,
            err_msg="posterior_theta_samples draws differ between keep and recompute paths",
        )


# ---------------------------------------------------------------------------
# #164: num_threads controls the rayon pool without changing results
# ---------------------------------------------------------------------------

def _toy_prevalence_corpus(n=120, seed=1):
    rng = np.random.default_rng(seed)
    A = [f"a{i}" for i in range(12)]; B = [f"b{i}" for i in range(12)]
    docs, xs = [], []
    for _ in range(n):
        x = rng.random(); xs.append(x)
        docs.append(list(rng.choice(A if x > 0.5 else B, size=10)))
    return docs, np.array(xs).reshape(-1, 1)


def test_num_threads_does_not_change_stm_results():
    docs, X = _toy_prevalence_corpus()
    ref = topica.models.STM(num_topics=4, seed=7); ref.fit(docs, prevalence=X, iters=60)
    for nt in (1, 2, 3):
        m = topica.models.STM(num_topics=4, seed=7)
        m.fit(docs, prevalence=X, iters=60, num_threads=nt)
        np.testing.assert_array_equal(m.topic_word, ref.topic_word)
        np.testing.assert_array_equal(np.asarray(m.eta_mean), np.asarray(ref.eta_mean))


def test_num_threads_does_not_change_ctm_results():
    docs, _ = _toy_prevalence_corpus()
    ref = topica.models.CTM(num_topics=4, seed=7); ref.fit(docs, iters=60)
    m = topica.models.CTM(num_topics=4, seed=7); m.fit(docs, iters=60, num_threads=2)
    np.testing.assert_array_equal(m.topic_word, ref.topic_word)
