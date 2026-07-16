"""Tests for STS keep_eta_cov=False memory-saving option.

Issue #162: extend keep_eta_cov to STS (eta_dim = 2K-1).  When False,
skip storing the per-document variational covariance (nu) to save O(N*(2K-1)^2)
memory, but keep the fit bit-identical. _recompute_eta_cov() regenerates nu
on demand.
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
    words = [f"w{i}" for i in range(15)]
    docs = []
    for _ in range(60):
        n = rng.integers(8, 13)
        tokens = rng.choice(words, size=n, replace=True).tolist()
        docs.append(tokens)
    return docs


DOCS = _tiny_docs()
# Two-group sentiment seed (values 0 or 1 alternating).
SEED = [float(i % 2) for i in range(len(DOCS))]


# ---------------------------------------------------------------------------
# (a) fit bit-identity: keep_eta_cov=True vs False produce equal results
# ---------------------------------------------------------------------------

class TestFitBitIdentical:
    def test_sts_doc_topic_equal(self):
        """STS fit with keep_eta_cov=False is bit-for-bit identical to True."""
        m_keep = topica.models.STS(3, init="random", seed=0)
        m_keep.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.STS(3, init="random", seed=0)
        m_drop.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)

        np.testing.assert_array_equal(
            np.asarray(m_keep.doc_topic),
            np.asarray(m_drop.doc_topic),
            err_msg="STS doc_topic differs between keep_eta_cov=True and False",
        )

    def test_sts_topic_word_equal(self):
        """STS topic_word is bit-identical with and without keep_eta_cov."""
        m_keep = topica.models.STS(3, init="random", seed=0)
        m_keep.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.STS(3, init="random", seed=0)
        m_drop.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)

        np.testing.assert_array_equal(
            np.asarray(m_keep.topic_word),
            np.asarray(m_drop.topic_word),
            err_msg="STS topic_word differs between keep_eta_cov=True and False",
        )

    def test_sts_eta_mean_equal(self):
        """STS eta_mean is bit-identical with and without keep_eta_cov."""
        m_keep = topica.models.STS(3, init="random", seed=0)
        m_keep.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.STS(3, init="random", seed=0)
        m_drop.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)

        np.testing.assert_array_equal(
            np.asarray(m_keep.eta_mean),
            np.asarray(m_drop.eta_mean),
            err_msg="STS eta_mean differs between keep_eta_cov=True and False",
        )

    def test_sts_bound_equal(self):
        """STS bound is identical with and without keep_eta_cov."""
        m_keep = topica.models.STS(3, init="random", seed=0)
        m_keep.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.STS(3, init="random", seed=0)
        m_drop.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)

        assert m_keep.bound == m_drop.bound, (
            f"STS bound differs: {m_keep.bound} vs {m_drop.bound}"
        )


# ---------------------------------------------------------------------------
# (b) eta_cov getter raises RuntimeError when keep_eta_cov=False
# ---------------------------------------------------------------------------

class TestEtaCovGetterRaises:
    def test_sts_eta_cov_raises(self):
        """STS.eta_cov raises RuntimeError with useful message when not kept."""
        m = topica.models.STS(3, init="random", seed=1)
        m.fit(DOCS, SEED, iters=3, convergence_tol=0, keep_eta_cov=False)
        with pytest.raises(RuntimeError, match="keep_eta_cov"):
            _ = m.eta_cov

    def test_sts_eta_cov_available_when_kept(self):
        """STS.eta_cov does NOT raise when keep_eta_cov=True (default)."""
        m = topica.models.STS(3, init="random", seed=1)
        m.fit(DOCS, SEED, iters=3, convergence_tol=0, keep_eta_cov=True)
        cov = m.eta_cov  # should not raise
        assert cov is not None


# ---------------------------------------------------------------------------
# (c) _recompute_eta_cov returns same result as stored eta_cov (bit-exact)
# ---------------------------------------------------------------------------

class TestRecomputeEtaCov:
    def test_sts_recompute_equals_stored(self):
        """_recompute_eta_cov output must equal what keep_eta_cov=True stores."""
        m_keep = topica.models.STS(3, init="random", seed=2)
        m_keep.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.STS(3, init="random", seed=2)
        m_drop.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)

        stored = np.asarray(m_keep.eta_cov, dtype=np.float32)
        recomputed = np.asarray(m_drop._recompute_eta_cov(), dtype=np.float32)

        np.testing.assert_array_equal(
            stored,
            recomputed,
            err_msg="STS _recompute_eta_cov does not match stored eta_cov",
        )

    def test_sts_recompute_shape(self):
        """_recompute_eta_cov returns the correct (D, 2K-1, 2K-1) shape."""
        k = 3
        m = topica.models.STS(k, init="random", seed=3)
        m.fit(DOCS, SEED, iters=3, convergence_tol=0, keep_eta_cov=False)
        cov = m._recompute_eta_cov()
        d = len(DOCS)
        n = 2 * k - 1
        assert cov.shape == (d, n, n), (
            f"STS _recompute_eta_cov shape {cov.shape} != ({d}, {n}, {n})"
        )

    def test_sts_recompute_is_positive_definite(self):
        """Each recomputed covariance should be (approximately) positive definite."""
        m = topica.models.STS(3, init="random", seed=4)
        m.fit(DOCS, SEED, iters=3, convergence_tol=0, keep_eta_cov=False)
        cov = np.asarray(m._recompute_eta_cov(), dtype=np.float64)
        # Check eigenvalues are all positive for a sample of documents.
        for di in range(min(5, cov.shape[0])):
            eigs = np.linalg.eigvalsh(cov[di])
            assert np.all(eigs > -1e-6), (
                f"doc {di}: non-positive eigenvalues {eigs}"
            )

    def test_sts_recompute_allclose_to_stored(self):
        """Recomputed eta_cov is within float32 rounding of stored (allclose)."""
        m_keep = topica.models.STS(3, init="random", seed=5)
        m_keep.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=True)

        m_drop = topica.models.STS(3, init="random", seed=5)
        m_drop.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)

        stored = np.asarray(m_keep.eta_cov, dtype=np.float64)
        recomputed = np.asarray(m_drop._recompute_eta_cov(), dtype=np.float64)

        np.testing.assert_allclose(
            stored, recomputed, atol=1e-5,
            err_msg="STS recomputed eta_cov not close to stored",
        )


# ---------------------------------------------------------------------------
# (d) keep_eta_cov=False does not break the STS fit / SE paths
# ---------------------------------------------------------------------------

class TestStsWorkflowWithKeepFalse:
    def test_sts_doc_topic_valid(self):
        """STS with keep_eta_cov=False still produces valid topic proportions."""
        m = topica.models.STS(3, init="random", seed=6)
        m.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)
        theta = np.asarray(m.doc_topic)
        np.testing.assert_allclose(
            theta.sum(axis=1), np.ones(len(DOCS)), atol=1e-6,
            err_msg="doc_topic rows do not sum to 1",
        )

    def test_sts_sentiment_shape(self):
        """STS sentiment has shape (D, K) with keep_eta_cov=False."""
        k = 3
        m = topica.models.STS(k, init="random", seed=7)
        m.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)
        sent = np.asarray(m.sentiment)
        assert sent.shape == (len(DOCS), k), (
            f"sentiment shape {sent.shape} != ({len(DOCS)}, {k})"
        )

    def test_sts_recompute_after_keep_false_eta_mean_unchanged(self):
        """_recompute_eta_cov must not alter eta_mean."""
        m = topica.models.STS(3, init="random", seed=8)
        m.fit(DOCS, SEED, iters=5, convergence_tol=0, keep_eta_cov=False)
        mean_before = np.asarray(m.eta_mean).copy()
        _ = m._recompute_eta_cov()
        mean_after = np.asarray(m.eta_mean)
        np.testing.assert_array_equal(
            mean_before, mean_after,
            err_msg="eta_mean changed after _recompute_eta_cov call",
        )

    def test_sts_keep_false_default_true_gives_eta_cov(self):
        """Default keep_eta_cov=True still stores eta_cov (no regression)."""
        m = topica.models.STS(3, init="random", seed=9)
        m.fit(DOCS, SEED, iters=3, convergence_tol=0)  # default keep_eta_cov=True
        # Should not raise
        cov = m.eta_cov
        assert cov.shape[0] == len(DOCS)
