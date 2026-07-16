"""Tests for STM gamma_prior="pooled" / "l1" (elastic-net prevalence regression).

Covers:
- Default ("pooled") is the existing ridge path — bit-for-bit unchanged.
- "l1" on a high-dimensional one-hot design produces a sparser gamma than "pooled".
- Invalid gamma_prior raises ValueError.
- gamma_enet out of range raises ValueError.
- Correct shapes and basic invariants hold for both priors.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import topica


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOCAB_A = ["alpha", "bravo", "charlie", "delta", "echo"]
_VOCAB_B = ["foxtrot", "golf", "hotel", "india", "juliet"]


def _make_binary_corpus(n_per_class: int = 60, seed: int = 1):
    """Two-class corpus with a binary prevalence covariate."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_per_class):
        docs.append(
            list(rng.choice(_VOCAB_A, size=8, replace=True))
            + list(rng.choice(_VOCAB_B, size=2, replace=True))
        )
    for _ in range(n_per_class):
        docs.append(
            list(rng.choice(_VOCAB_B, size=8, replace=True))
            + list(rng.choice(_VOCAB_A, size=2, replace=True))
        )
    x = np.array([1.0] * n_per_class + [0.0] * n_per_class).reshape(-1, 1)
    return docs, x


def _make_high_dim_corpus(n_docs: int = 300, n_covariates: int = 30, seed: int = 7):
    """High-dimensional corpus: n_covariates continuous predictors.

    The first two columns have true signal (correlated with the topic split);
    the remaining columns are pure random noise. This is the design where the
    L1 prior outperforms pooled ridge by zeroing the noise coefficients.
    """
    rng = np.random.default_rng(seed)
    half = n_docs // 2
    docs = []
    for _ in range(half):
        docs.append(
            list(rng.choice(_VOCAB_A, size=8, replace=True))
            + list(rng.choice(_VOCAB_B, size=2, replace=True))
        )
    for _ in range(half):
        docs.append(
            list(rng.choice(_VOCAB_B, size=8, replace=True))
            + list(rng.choice(_VOCAB_A, size=2, replace=True))
        )

    # binary signal for the topic split
    signal = np.array([1.0] * half + [0.0] * half)
    x = np.zeros((n_docs, n_covariates), dtype=np.float64)
    x[:, 0] = signal
    x[:, 1] = signal * 0.8 + rng.normal(0, 0.1, n_docs)
    x[:, 2:] = rng.standard_normal((n_docs, n_covariates - 2))

    return docs, x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pooled_model():
    """Default-prior STM on the binary corpus."""
    docs, x = _make_binary_corpus()
    m = topica.models.STM(num_topics=2, seed=1)
    m.fit(docs, x, prevalence_names=["x"], iters=30, convergence_tol=0.0)
    return m


@pytest.fixture(scope="module")
def pooled_model_explicit():
    """Explicit gamma_prior='pooled' — must give bit-for-bit identical result."""
    docs, x = _make_binary_corpus()
    m = topica.models.STM(num_topics=2, seed=1)
    m.fit(docs, x, prevalence_names=["x"], iters=30, convergence_tol=0.0, gamma_prior="pooled")
    return m


@pytest.fixture(scope="module")
def l1_model_high_dim():
    """L1 prior STM on the high-dimensional one-hot design."""
    docs, x = _make_high_dim_corpus()
    m = topica.models.STM(num_topics=2, seed=1)
    m.fit(docs, x, iters=30, convergence_tol=0.0, gamma_prior="l1")
    return m


@pytest.fixture(scope="module")
def pooled_model_high_dim():
    """Pooled (ridge) STM on the same high-dimensional design (for sparsity comparison)."""
    docs, x = _make_high_dim_corpus()
    m = topica.models.STM(num_topics=2, seed=1)
    m.fit(docs, x, iters=30, convergence_tol=0.0, gamma_prior="pooled")
    return m


# ---------------------------------------------------------------------------
# 1. Bit-for-bit parity: default == explicit gamma_prior="pooled"
# ---------------------------------------------------------------------------

class TestPooledUnchanged:
    def test_topic_word_identical(self, pooled_model, pooled_model_explicit):
        npt.assert_array_equal(
            pooled_model.topic_word,
            pooled_model_explicit.topic_word,
            err_msg="Default and explicit pooled must produce identical topic_word",
        )

    def test_doc_topic_identical(self, pooled_model, pooled_model_explicit):
        npt.assert_array_equal(
            pooled_model.doc_topic,
            pooled_model_explicit.doc_topic,
            err_msg="Default and explicit pooled must produce identical doc_topic",
        )

    def test_gamma_identical(self, pooled_model, pooled_model_explicit):
        npt.assert_array_equal(
            pooled_model.prevalence_effects,
            pooled_model_explicit.prevalence_effects,
            err_msg="Default and explicit pooled must produce identical gamma",
        )


# ---------------------------------------------------------------------------
# 2. Shapes and invariants for L1 prior
# ---------------------------------------------------------------------------

class TestL1Shapes:
    def test_topic_word_shape(self, l1_model_high_dim):
        k = l1_model_high_dim.num_topics
        v = l1_model_high_dim.topic_word.shape[1]
        assert l1_model_high_dim.topic_word.shape == (k, v)

    def test_doc_topic_rows_sum_to_one(self, l1_model_high_dim):
        row_sums = l1_model_high_dim.doc_topic.sum(axis=1)
        npt.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    def test_prevalence_effects_shape(self, l1_model_high_dim):
        # With 30 continuous covariates + 1 intercept = 31 features; K-1 = 1 topic column.
        gamma = l1_model_high_dim.prevalence_effects
        assert gamma.shape[0] == 31   # 30 covariates + intercept
        assert gamma.shape[1] == 1    # K-1 = 1

    def test_gamma_is_finite(self, l1_model_high_dim):
        assert np.all(np.isfinite(l1_model_high_dim.prevalence_effects))


# ---------------------------------------------------------------------------
# 3. L1 produces sparser gamma than pooled on a high-dimensional design
# ---------------------------------------------------------------------------

class TestL1Sparsity:
    def test_l1_substantially_sparser_than_pooled(
        self, l1_model_high_dim, pooled_model_high_dim
    ):
        """L1 should zero substantially more coefficients than pooled ridge."""
        tol = 1e-6
        # Count zeros in the penalised rows (all rows except intercept row 0).
        g_l1 = l1_model_high_dim.prevalence_effects[1:]
        g_pooled = pooled_model_high_dim.prevalence_effects[1:]

        zeros_l1 = int(np.sum(np.abs(g_l1) < tol))
        zeros_pooled = int(np.sum(np.abs(g_pooled) < tol))

        assert zeros_l1 > zeros_pooled, (
            f"L1 should produce more near-zero coefficients than pooled: "
            f"l1_zeros={zeros_l1}, pooled_zeros={zeros_pooled}"
        )
        # Conservatively: L1 should zero at least 30% of the 30 penalised predictors.
        assert zeros_l1 >= 9, (
            f"L1 should zero at least 9 of 30 penalised coefficients, got {zeros_l1}"
        )


# ---------------------------------------------------------------------------
# 4. Validation errors
# ---------------------------------------------------------------------------

class TestGammaPriorValidation:
    def _base_docs_and_x(self):
        docs, x = _make_binary_corpus(n_per_class=10)
        return docs, x

    def test_invalid_gamma_prior_raises(self):
        docs, x = self._base_docs_and_x()
        m = topica.models.STM(num_topics=2, seed=1)
        with pytest.raises(ValueError, match="gamma_prior"):
            m.fit(docs, x, iters=2, convergence_tol=0.0, gamma_prior="bayes")

    def test_gamma_enet_zero_raises(self):
        docs, x = self._base_docs_and_x()
        m = topica.models.STM(num_topics=2, seed=1)
        with pytest.raises(ValueError, match="gamma_enet"):
            m.fit(docs, x, iters=2, convergence_tol=0.0, gamma_prior="l1", gamma_enet=0.0)

    def test_gamma_enet_negative_raises(self):
        docs, x = self._base_docs_and_x()
        m = topica.models.STM(num_topics=2, seed=1)
        with pytest.raises(ValueError, match="gamma_enet"):
            m.fit(docs, x, iters=2, convergence_tol=0.0, gamma_prior="l1", gamma_enet=-0.5)

    def test_gamma_enet_above_one_raises(self):
        docs, x = self._base_docs_and_x()
        m = topica.models.STM(num_topics=2, seed=1)
        with pytest.raises(ValueError, match="gamma_enet"):
            m.fit(docs, x, iters=2, convergence_tol=0.0, gamma_prior="l1", gamma_enet=1.5)

    def test_gamma_enet_one_accepted(self):
        docs, x = self._base_docs_and_x()
        m = topica.models.STM(num_topics=2, seed=1)
        m.fit(docs, x, iters=2, convergence_tol=0.0, gamma_prior="l1", gamma_enet=1.0)
        # No exception raised; doc_topic is accessible.
        assert m.doc_topic.shape[0] > 0

    def test_gamma_enet_midrange_accepted(self):
        docs, x = self._base_docs_and_x()
        m = topica.models.STM(num_topics=2, seed=1)
        m.fit(docs, x, iters=2, convergence_tol=0.0, gamma_prior="l1", gamma_enet=0.5)
        assert m.doc_topic.shape[0] > 0

    def test_pooled_accepts_any_gamma_enet(self):
        """gamma_enet is ignored for pooled prior — any value is fine."""
        docs, x = self._base_docs_and_x()
        m = topica.models.STM(num_topics=2, seed=1)
        # gamma_enet=0 is only rejected for l1; pooled ignores it entirely.
        m.fit(docs, x, iters=2, convergence_tol=0.0, gamma_prior="pooled", gamma_enet=0.0)
        assert m.doc_topic.shape[0] > 0
