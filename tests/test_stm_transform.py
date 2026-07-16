"""Tests for covariate-aware STM.transform and align_corpus (issue #39).

Covers:
- align_corpus drops OOV tokens and keeps in-vocabulary tokens.
- model.transform(docs) with no covariates still works (baseline prior).
- model.transform(docs, eta_prior_mean=...) uses the per-doc prior.
- topica.stm.transform with a prevalence covariate shifts theta vs the
  covariate-free baseline (when gamma is non-trivial).
- topica.stm.transform with X= alias.
- Pipeline: align_corpus -> covariate transform runs without error.
- shape and row-sum invariants on all transform paths.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import topica
from topica.models import STM
from topica import stm


# ---------------------------------------------------------------------------
# Shared synthetic corpus
# ---------------------------------------------------------------------------
# Two disjoint vocabularies A and B; binary covariate x strongly predicts
# which vocabulary a document uses.  With iters=60 the model consistently
# finds both topics and learns a non-trivial gamma.

_VOCAB_A = ["alpha", "bravo", "charlie", "delta", "echo"]
_VOCAB_B = ["foxtrot", "golf", "hotel", "india", "juliet"]


def _make_corpus(n_per_class=80, seed=7):
    rng = np.random.default_rng(seed)
    docs, labels = [], []
    for _ in range(n_per_class):
        docs.append(
            list(rng.choice(_VOCAB_A, size=8, replace=True))
            + list(rng.choice(_VOCAB_B, size=2, replace=True))
        )
        labels.append(1.0)
    for _ in range(n_per_class):
        docs.append(
            list(rng.choice(_VOCAB_B, size=8, replace=True))
            + list(rng.choice(_VOCAB_A, size=2, replace=True))
        )
        labels.append(0.0)
    x = np.array(labels, dtype=np.float64).reshape(-1, 1)
    return docs, x


@pytest.fixture(scope="module")
def fitted_stm_and_data():
    docs, x = _make_corpus(n_per_class=80, seed=7)
    model = STM(num_topics=2, seed=3)
    model.fit(docs, x, prevalence_names=["x"], iters=60)
    return model, docs, x


@pytest.fixture(scope="module")
def heldout_docs_and_x():
    """A small held-out set: 10 class-A docs + 10 class-B docs."""
    rng = np.random.default_rng(42)
    docs_a = [
        list(rng.choice(_VOCAB_A, size=8, replace=True))
        + list(rng.choice(_VOCAB_B, size=2, replace=True))
        for _ in range(10)
    ]
    docs_b = [
        list(rng.choice(_VOCAB_B, size=8, replace=True))
        + list(rng.choice(_VOCAB_A, size=2, replace=True))
        for _ in range(10)
    ]
    docs = docs_a + docs_b
    x = np.array([1.0] * 10 + [0.0] * 10, dtype=np.float64).reshape(-1, 1)
    return docs, x


# ---------------------------------------------------------------------------
# align_corpus
# ---------------------------------------------------------------------------

class TestAlignCorpus:
    def test_keeps_in_vocab_tokens(self, fitted_stm_and_data):
        model, _, _ = fitted_stm_and_data
        vocab_set = set(model.vocabulary)
        # All _VOCAB_A / _VOCAB_B words are in the fitted vocabulary.
        docs = [["alpha", "bravo", "charlie"]]
        aligned = stm.align_corpus(docs, model)
        assert aligned == [["alpha", "bravo", "charlie"]]

    def test_drops_oov_tokens(self, fitted_stm_and_data):
        model, _, _ = fitted_stm_and_data
        docs = [["alpha", "UNKNOWNWORD", "bravo", "OUTOFDOMAIN"]]
        aligned = stm.align_corpus(docs, model)
        assert aligned == [["alpha", "bravo"]]

    def test_empty_doc_after_oov_filtering(self, fitted_stm_and_data):
        model, _, _ = fitted_stm_and_data
        docs = [["UNKNOWNWORD", "OUTOFDOMAIN"]]
        aligned = stm.align_corpus(docs, model)
        assert aligned == [[]]

    def test_length_preserved(self, fitted_stm_and_data):
        model, _, _ = fitted_stm_and_data
        docs = [["alpha"], ["UNKNOWN", "bravo"], ["charlie", "delta"]]
        aligned = stm.align_corpus(docs, model)
        assert len(aligned) == 3

    def test_topica_top_level_export(self, fitted_stm_and_data):
        """align_corpus is accessible as topica.align_corpus."""
        model, _, _ = fitted_stm_and_data
        docs = [["alpha", "UNKNOWN"]]
        result = topica.align_corpus(docs, model)
        assert result == [["alpha"]]

    def test_all_oov_docs_become_empty(self, fitted_stm_and_data):
        model, _, _ = fitted_stm_and_data
        docs = [["nope1", "nope2"], ["another_unknown"]]
        aligned = stm.align_corpus(docs, model)
        assert aligned == [[], []]


# ---------------------------------------------------------------------------
# model.transform (low-level Rust method)
# ---------------------------------------------------------------------------

class TestSTMTransformBaseline:
    def test_baseline_transform_shape(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, _ = heldout_docs_and_x
        theta = model.transform(held_docs)
        assert theta.shape == (20, 2)

    def test_baseline_transform_rows_sum_to_one(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, _ = heldout_docs_and_x
        theta = model.transform(held_docs)
        npt.assert_allclose(theta.sum(axis=1), np.ones(20), atol=1e-5)

    def test_baseline_transform_nonnegative(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, _ = heldout_docs_and_x
        theta = model.transform(held_docs)
        assert (theta >= 0).all()

    def test_transform_with_eta_prior_mean_shape(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, held_x = heldout_docs_and_x
        n = len(held_docs)
        km1 = model.num_topics - 1
        eta_pm = np.zeros((n, km1), dtype=np.float64)
        theta = model.transform(held_docs, eta_prior_mean=eta_pm)
        assert theta.shape == (n, model.num_topics)

    def test_transform_with_eta_prior_mean_rows_sum_to_one(
        self, fitted_stm_and_data, heldout_docs_and_x
    ):
        model, _, _ = fitted_stm_and_data
        held_docs, held_x = heldout_docs_and_x
        n = len(held_docs)
        km1 = model.num_topics - 1
        eta_pm = np.zeros((n, km1), dtype=np.float64)
        theta = model.transform(held_docs, eta_prior_mean=eta_pm)
        npt.assert_allclose(theta.sum(axis=1), np.ones(n), atol=1e-5)

    def test_transform_bad_eta_shape_raises(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, _ = heldout_docs_and_x
        bad_eta = np.zeros((5, model.num_topics - 1), dtype=np.float64)  # wrong n_docs
        with pytest.raises((ValueError, Exception)):
            model.transform(held_docs, eta_prior_mean=bad_eta)


# ---------------------------------------------------------------------------
# stm.transform (covariate-aware Python wrapper)
# ---------------------------------------------------------------------------

class TestSTMTransformCovariate:
    def test_covariate_transform_shape(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, held_x = heldout_docs_and_x
        theta = stm.transform(model, held_docs, prevalence=held_x)
        assert theta.shape == (20, 2)

    def test_covariate_transform_rows_sum_to_one(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, held_x = heldout_docs_and_x
        theta = stm.transform(model, held_docs, prevalence=held_x)
        npt.assert_allclose(theta.sum(axis=1), np.ones(20), atol=1e-5)

    def test_covariate_transform_nonnegative(self, fitted_stm_and_data, heldout_docs_and_x):
        model, _, _ = fitted_stm_and_data
        held_docs, held_x = heldout_docs_and_x
        theta = stm.transform(model, held_docs, prevalence=held_x)
        assert (theta >= 0).all()

    def test_covariate_x_alias(self, fitted_stm_and_data, heldout_docs_and_x):
        """X= is an alias for prevalence=."""
        model, _, _ = fitted_stm_and_data
        held_docs, held_x = heldout_docs_and_x
        theta_prev = stm.transform(model, held_docs, prevalence=held_x)
        theta_x = stm.transform(model, held_docs, X=held_x)
        npt.assert_array_equal(theta_prev, theta_x)

    def test_no_covariates_matches_model_transform(self, fitted_stm_and_data, heldout_docs_and_x):
        """stm.transform without covariates gives the same result as model.transform."""
        model, _, _ = fitted_stm_and_data
        held_docs, _ = heldout_docs_and_x
        theta_direct = model.transform(held_docs)
        theta_wrapper = stm.transform(model, held_docs)
        npt.assert_array_equal(theta_direct, theta_wrapper)

    def test_covariate_transform_differs_from_baseline(
        self, fitted_stm_and_data, heldout_docs_and_x
    ):
        """When gamma is non-trivial, covariate-aware theta differs from baseline.

        The model was fit with a strong binary covariate (x=1 selects VOCAB_A,
        x=0 selects VOCAB_B).  The held-out set mixes both groups, so the
        per-doc prior should shift theta compared to the shared baseline prior.
        """
        model, _, _ = fitted_stm_and_data
        held_docs, held_x = heldout_docs_and_x

        theta_baseline = stm.transform(model, held_docs)
        theta_covariate = stm.transform(model, held_docs, prevalence=held_x)

        # With a non-trivial gamma the two arrays must differ.
        gamma = model.prevalence_effects
        assert not np.allclose(gamma, 0), "gamma is zero — model did not learn covariates"
        assert not np.allclose(theta_baseline, theta_covariate), (
            "Covariate-aware theta is identical to baseline — gamma may be trivial"
        )

    def test_covariate_pushes_theta_in_right_direction(
        self, fitted_stm_and_data, heldout_docs_and_x
    ):
        """Class-A docs with x=1 prior should show higher A-topic weight than
        those same docs with x=0 prior (and vice versa for class-B docs)."""
        model, _, _ = fitted_stm_and_data
        held_docs, _ = heldout_docs_and_x  # first 10 are A-class, last 10 are B-class

        # Identify A topic (high weight on VOCAB_A words).
        a_set = set(_VOCAB_A)
        a_topic = None
        for t in range(model.num_topics):
            top5 = [w for w, _ in model.top_words(5, topic=t)]
            if sum(1 for w in top5 if w in a_set) >= 3:
                a_topic = t
                break
        if a_topic is None:
            pytest.skip("Could not identify A topic by top words")

        x_ones = np.ones((20, 1), dtype=np.float64)
        x_zeros = np.zeros((20, 1), dtype=np.float64)

        theta_x1 = stm.transform(model, held_docs, prevalence=x_ones)
        theta_x0 = stm.transform(model, held_docs, prevalence=x_zeros)

        # On average, x=1 prior should raise A-topic weight vs x=0 prior.
        mean_a_x1 = theta_x1[:, a_topic].mean()
        mean_a_x0 = theta_x0[:, a_topic].mean()
        assert mean_a_x1 > mean_a_x0, (
            f"Expected x=1 prior to raise A-topic weight; got {mean_a_x1:.3f} vs {mean_a_x0:.3f}"
        )

    def test_wrong_covariate_columns_raises(self, fitted_stm_and_data, heldout_docs_and_x):
        """Wrong number of covariate columns raises ValueError."""
        model, _, _ = fitted_stm_and_data
        held_docs, _ = heldout_docs_and_x
        # The model was fit with 1 covariate; supply 3 columns (wrong).
        bad_x = np.zeros((20, 3), dtype=np.float64)
        with pytest.raises((ValueError, Exception)):
            stm.transform(model, held_docs, prevalence=bad_x)


# ---------------------------------------------------------------------------
# align_corpus -> covariate transform pipeline
# ---------------------------------------------------------------------------

class TestAlignThenTransform:
    def test_pipeline_runs_without_error(self, fitted_stm_and_data):
        model, _, _ = fitted_stm_and_data
        # Mix in-vocabulary words with OOV tokens.
        rng = np.random.default_rng(99)
        raw_docs = [
            list(rng.choice(_VOCAB_A, size=6, replace=True)) + ["UNKNOWN1", "UNKNOWN2"]
            for _ in range(5)
        ] + [
            list(rng.choice(_VOCAB_B, size=6, replace=True)) + ["OUTOFVOCAB"]
            for _ in range(5)
        ]
        x_held = np.array([1.0] * 5 + [0.0] * 5, dtype=np.float64).reshape(-1, 1)

        aligned = stm.align_corpus(raw_docs, model)
        # No OOV token should remain.
        vocab_set = set(model.vocabulary)
        for doc in aligned:
            assert all(t in vocab_set for t in doc)

        theta = stm.transform(model, aligned, prevalence=x_held)
        assert theta.shape == (10, 2)
        npt.assert_allclose(theta.sum(axis=1), np.ones(10), atol=1e-5)

    def test_baseline_pipeline_runs(self, fitted_stm_and_data):
        """align_corpus -> model.transform (no covariates) works."""
        model, _, _ = fitted_stm_and_data
        raw_docs = [["alpha", "UNKNOWN", "bravo"], ["golf", "OOVTOKEN", "hotel"]]
        aligned = stm.align_corpus(raw_docs, model)
        theta = model.transform(aligned)
        assert theta.shape == (2, 2)
        npt.assert_allclose(theta.sum(axis=1), np.ones(2), atol=1e-5)


class TestSplineFormulaGuard:
    def test_spline_formula_rejected(self, fitted_stm_and_data):
        """A spline() prevalence formula is rejected in transform: its knots
        would be recomputed on the new docs rather than reused from fit."""
        import pandas as pd

        model, docs, _ = fitted_stm_and_data
        data = pd.DataFrame({"yr": list(range(len(docs)))})
        with pytest.raises(ValueError, match="spline"):
            stm.transform(model, docs, formula="~ spline(yr, df=3)", data=data)
