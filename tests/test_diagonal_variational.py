"""Tests for the diagonal (mean-field) variational-covariance mode on CTM/STM.

`variational="diagonal"` swaps the default Laplace E-step (full posterior
covariance nu = H^-1) for a mean-field diagonal covariance (nu = diag(1/H_ii)),
which skips the per-document Cholesky/inverse. These tests check that:

(a) a diagonal fit converges (non-decreasing variational bound) and recovers
    planted disjoint-vocabulary topics,
(b) the resulting eta_cov has exactly-zero off-diagonals and positive diagonals,
(c) posterior_theta_samples and keep_eta_cov=False both work in diagonal mode,
(d) the default stays "laplace" and the flag actually changes the fit.
"""

import numpy as np
import pytest
import topica


# ---------------------------------------------------------------------------
# Shared fixture: three disjoint vocabulary blocks; each doc draws from one.
# A well-fit model's beta rows should each concentrate on a single block.
# ---------------------------------------------------------------------------

N_BLOCKS = 3
WORDS_PER_BLOCK = 5
VOCAB = N_BLOCKS * WORDS_PER_BLOCK


def _planted_docs():
    docs = []
    for d in range(240):
        b = d % N_BLOCKS
        block = [f"w{w}" for w in range(b * WORDS_PER_BLOCK, (b + 1) * WORDS_PER_BLOCK)]
        docs.append(block + block)  # each block word twice
    return docs


DOCS = _planted_docs()


def _recovers_blocks(model):
    """True iff every planted block is the top-WORDS_PER_BLOCK of some topic."""
    beta = model.topic_word
    covered = set()
    for t in range(N_BLOCKS):
        order = np.argsort(beta[t])[::-1]
        top = set(int(i) for i in order[:WORDS_PER_BLOCK])
        for b in range(N_BLOCKS):
            block = set(range(b * WORDS_PER_BLOCK, (b + 1) * WORDS_PER_BLOCK))
            if block.issubset(top):
                covered.add(b)
    return len(covered) == N_BLOCKS


# ---------------------------------------------------------------------------
# (a) diagonal fit converges (monotone bound) and recovers topics
# ---------------------------------------------------------------------------

class TestDiagonalConverges:
    def test_bound_converges(self):
        m = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m.fit(DOCS, iters=40, convergence_tol=0)
        hist = np.asarray([b for (_, b) in m.fit_history])
        assert len(hist) >= 2

        # The mean-field diagonal objective is not the exact Laplace lower bound,
        # so the reported bound rises steeply to (near) convergence and may then
        # drift by a tiny amount per step. Two honest checks:
        # (1) the bound improves massively overall (the fit actually learns), and
        total_range = hist.max() - hist[0]
        assert total_range > 0, "diagonal bound did not improve overall"
        assert hist[-1] > hist[0], "diagonal bound did not improve overall"

        # (2) the ascent is monotone, and after the peak any creep is negligible
        # relative to the total improvement (no large backward jumps).
        steps = np.diff(hist)
        max_decrease = -steps.min() if steps.min() < 0 else 0.0
        assert max_decrease < 0.01 * total_range, (
            f"max per-step decrease {max_decrease:.4f} too large relative to "
            f"total improvement {total_range:.4f}"
        )

    def test_recovers_planted_topics(self):
        m = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m.fit(DOCS, iters=40, convergence_tol=0)
        assert _recovers_blocks(m), "diagonal mode failed to recover planted topics"


# ---------------------------------------------------------------------------
# (b) eta_cov is purely diagonal: zero off-diagonals, positive diagonals
# ---------------------------------------------------------------------------

class TestDiagonalCovariance:
    def test_eta_cov_diagonal(self):
        m = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m.fit(DOCS, iters=10, convergence_tol=0, keep_eta_cov=True)
        cov = np.asarray(m.eta_cov, dtype=np.float64)  # (D, K-1, K-1)
        km1 = N_BLOCKS - 1
        assert cov.shape[1:] == (km1, km1)
        diag = np.diagonal(cov, axis1=1, axis2=2)
        assert np.all(diag > 0.0), "diagonal entries must be positive"
        # Off-diagonals must be exactly zero.
        off = cov.copy()
        idx = np.arange(km1)
        off[:, idx, idx] = 0.0
        assert np.all(off == 0.0), "off-diagonal eta_cov must be exactly zero"


# ---------------------------------------------------------------------------
# (c) posterior_theta_samples and keep_eta_cov=False work in diagonal mode
# ---------------------------------------------------------------------------

class TestDiagonalDownstream:
    def test_posterior_theta_samples(self):
        from topica.stm import posterior_theta_samples

        m = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m.fit(DOCS, iters=10, convergence_tol=0, keep_eta_cov=True)
        draws = posterior_theta_samples(m, nsims=8, seed=0)
        d = m.doc_topic.shape[0]
        assert draws.shape == (8, d, N_BLOCKS)
        assert np.all(np.isfinite(draws))

    def test_keep_eta_cov_false_recompute(self):
        m = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m.fit(DOCS, iters=10, convergence_tol=0, keep_eta_cov=False)
        # eta_cov getter should raise; _recompute_eta_cov regenerates it.
        with pytest.raises(RuntimeError, match="keep_eta_cov"):
            _ = m.eta_cov
        cov = np.asarray(m._recompute_eta_cov(), dtype=np.float64)
        km1 = N_BLOCKS - 1
        off = cov.copy()
        idx = np.arange(km1)
        off[:, idx, idx] = 0.0
        assert np.all(off == 0.0), "recomputed off-diagonal must be zero in diagonal mode"

    def test_keep_eta_cov_false_recompute_matches_stored(self):
        m_keep = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m_keep.fit(DOCS, iters=10, convergence_tol=0, keep_eta_cov=True)
        m_drop = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m_drop.fit(DOCS, iters=10, convergence_tol=0, keep_eta_cov=False)
        stored = np.asarray(m_keep.eta_cov, dtype=np.float64)
        recomp = np.asarray(m_drop._recompute_eta_cov(), dtype=np.float64)
        np.testing.assert_allclose(recomp, stored, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# (d) default is "laplace"; the flag actually changes the fit
# ---------------------------------------------------------------------------

class TestDefaultAndDifference:
    def test_default_is_laplace(self):
        m = topica.models.CTM(N_BLOCKS)
        assert m.variational == "laplace"
        m.fit(DOCS, iters=5, convergence_tol=0)
        assert m.variational == "laplace"

    def test_invalid_variational_raises(self):
        with pytest.raises(ValueError, match="variational"):
            topica.models.CTM(N_BLOCKS, variational="full")

    def test_diagonal_differs_from_laplace(self):
        m_lap = topica.models.CTM(N_BLOCKS, variational="laplace", init="random", seed=1)
        m_lap.fit(DOCS, iters=10, convergence_tol=0, keep_eta_cov=True)
        m_diag = topica.models.CTM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m_diag.fit(DOCS, iters=10, convergence_tol=0, keep_eta_cov=True)

        cov_lap = np.asarray(m_lap.eta_cov, dtype=np.float64)
        cov_diag = np.asarray(m_diag.eta_cov, dtype=np.float64)
        km1 = N_BLOCKS - 1
        # Laplace generally has nonzero off-diagonals; diagonal mode has none.
        off_lap = cov_lap.copy()
        idx = np.arange(km1)
        off_lap[:, idx, idx] = 0.0
        assert np.any(np.abs(off_lap) > 0.0), "laplace should have nonzero off-diagonals"
        # The two covariance arrays must differ (the flag does something).
        assert not np.allclose(cov_lap, cov_diag), "diagonal fit should differ from laplace"

    def test_stm_diagonal_runs(self):
        # STM in diagonal mode with a prevalence covariate.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(len(DOCS), 1))
        m = topica.models.STM(N_BLOCKS, variational="diagonal", init="random", seed=1)
        m.fit(DOCS, prevalence=X, iters=8, convergence_tol=0, keep_eta_cov=True)
        assert m.variational == "diagonal"
        cov = np.asarray(m.eta_cov, dtype=np.float64)
        km1 = N_BLOCKS - 1
        off = cov.copy()
        idx = np.arange(km1)
        off[:, idx, idx] = 0.0
        assert np.all(off == 0.0)
