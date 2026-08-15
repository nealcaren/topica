"""Tests for the STM (Structural Topic Model) class.

Covers: constructor validation, unfitted guards, output shapes and invariants,
topic recovery, covariate recovery (the headline STM property), determinism,
input type parity (Corpus vs list[list[str]]), multiple covariates,
top_words, and coherence.

NOTE: This file tests the STM *model class* (topica.STM).
      tests/test_stm.py tests the topica.stm *analysis toolkit*.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from topica import STM, Corpus, stm


# ---------------------------------------------------------------------------
# Shared synthetic corpus
# ---------------------------------------------------------------------------
# Two disjoint vocabularies A (5 words) and B (5 words), 250 documents,
# binary covariate x:
#   x=1  -> ~80% vocab-A tokens (topic A)
#   x=0  -> ~80% vocab-B tokens (topic B)
#
# Signal is strong enough that covariate-recovery z > 300 in practice.

_VOCAB_A = ["alpha", "bravo", "charlie", "delta", "echo"]
_VOCAB_B = ["foxtrot", "golf", "hotel", "india", "juliet"]
_N_DOCS = 250          # 125 per class
_DOC_LEN_MAJOR = 8    # tokens drawn from the dominant vocabulary
_DOC_LEN_MINOR = 2    # tokens drawn from the other vocabulary


def _make_stm_corpus(n_per_class=125, seed=1):
    """Return (docs, x_2d) where x_2d is (N,1) float64."""
    rng = np.random.default_rng(seed)
    docs = []
    labels = []
    for _ in range(n_per_class):
        docs.append(
            list(rng.choice(_VOCAB_A, size=_DOC_LEN_MAJOR, replace=True))
            + list(rng.choice(_VOCAB_B, size=_DOC_LEN_MINOR, replace=True))
        )
        labels.append(1.0)
    for _ in range(n_per_class):
        docs.append(
            list(rng.choice(_VOCAB_B, size=_DOC_LEN_MAJOR, replace=True))
            + list(rng.choice(_VOCAB_A, size=_DOC_LEN_MINOR, replace=True))
        )
        labels.append(0.0)
    x_2d = np.array(labels, dtype=np.float64).reshape(-1, 1)
    return docs, x_2d


@pytest.fixture(scope="module")
def stm_corpus_and_x():
    """Shared (docs, x_2d) for the synthetic corpus."""
    return _make_stm_corpus(n_per_class=125, seed=1)


@pytest.fixture(scope="module")
def fitted_stm(stm_corpus_and_x):
    """STM(num_topics=2, seed=1) fitted on the synthetic corpus (iters=60)."""
    docs, x_2d = stm_corpus_and_x
    model = STM(num_topics=2, seed=1)
    model.fit(docs, x_2d, prevalence_names=["x"], iters=60)
    return model


@pytest.fixture(scope="module")
def a_topic_idx(fitted_stm):
    """Index of the 'A' topic (top words dominated by _VOCAB_A)."""
    a_set = set(_VOCAB_A)
    for t in range(fitted_stm.num_topics):
        top5 = [w for w, _ in fitted_stm.top_words(5, topic=t, weights=True)]
        if sum(1 for w in top5 if w in a_set) >= 3:
            return t
    pytest.fail("Could not identify the A topic by top words")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestSTMConstructor:
    def test_num_topics_available_before_fit(self):
        m = STM(5)
        assert m.num_topics == 5

    def test_num_topics_one_raises(self):
        with pytest.raises(ValueError):
            STM(1)

    def test_num_topics_zero_raises(self):
        with pytest.raises(ValueError):
            STM(0)

    def test_sigma_shrink_negative_raises(self):
        with pytest.raises(ValueError):
            STM(2, sigma_shrink=-0.1)

    def test_sigma_shrink_above_one_raises(self):
        with pytest.raises(ValueError):
            STM(2, sigma_shrink=1.5)

    def test_sigma_shrink_zero_accepted(self):
        m = STM(2, sigma_shrink=0.0)
        assert m.num_topics == 2

    def test_sigma_shrink_one_accepted(self):
        m = STM(2, sigma_shrink=1.0)
        assert m.num_topics == 2

    def test_sigma_shrink_midrange_accepted(self):
        m = STM(3, sigma_shrink=0.5)
        assert m.num_topics == 3

    def test_seed_parameter_accepted(self):
        m = STM(2, seed=99)
        assert m.num_topics == 2


# ---------------------------------------------------------------------------
# Unfitted-property guards
# ---------------------------------------------------------------------------

class TestSTMUnfittedGuards:
    UNFITTED_PROPERTIES = [
        "topic_word",
        "doc_topic",
        "topic_correlation",
        "prevalence_effects",
        "feature_names",
        "vocabulary",
        "doc_names",
    ]

    @pytest.mark.parametrize("prop", UNFITTED_PROPERTIES)
    def test_property_raises_before_fit(self, prop):
        m = STM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(m, prop)

    def test_top_words_raises_before_fit(self):
        m = STM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.top_words()

    def test_coherence_raises_before_fit(self):
        m = STM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.coherence()

    def test_num_topics_available_before_fit(self):
        """num_topics must NOT raise before fit — it returns the constructor value."""
        m = STM(4)
        assert m.num_topics == 4  # no RuntimeError


# ---------------------------------------------------------------------------
# Output shapes and basic invariants
# ---------------------------------------------------------------------------

class TestSTMShapes:
    def test_topic_word_shape(self, fitted_stm):
        # 2 topics, 10 unique words (5+5)
        assert fitted_stm.topic_word.shape == (2, 10)

    def test_doc_topic_shape(self, fitted_stm):
        assert fitted_stm.doc_topic.shape == (250, 2)

    def test_doc_topic_rows_sum_to_one(self, fitted_stm):
        npt.assert_allclose(
            fitted_stm.doc_topic.sum(axis=1),
            np.ones(250),
            atol=1e-5,
        )

    def test_topic_correlation_shape(self, fitted_stm):
        assert fitted_stm.topic_correlation.shape == (2, 2)

    def test_topic_correlation_symmetric(self, fitted_stm):
        C = fitted_stm.topic_correlation
        assert np.allclose(C, C.T), "topic_correlation must be symmetric"

    def test_topic_correlation_unit_diagonal(self, fitted_stm):
        npt.assert_allclose(
            np.diag(fitted_stm.topic_correlation), np.ones(2), atol=1e-6
        )

    def test_prevalence_effects_shape(self, fitted_stm):
        # F=1 covariate -> num_features=2 (intercept + x); num_topics-1=1
        assert fitted_stm.prevalence_effects.shape == (2, 1)

    def test_feature_names_content(self, fitted_stm):
        assert fitted_stm.feature_names == ["intercept", "x"]

    def test_feature_names_length(self, fitted_stm):
        # length = F+1 (intercept prepended)
        assert len(fitted_stm.feature_names) == 2

    def test_vocabulary_length_matches_topic_word(self, fitted_stm):
        assert len(fitted_stm.vocabulary) == fitted_stm.topic_word.shape[1]

    def test_doc_names_length_matches_doc_topic(self, fitted_stm):
        assert len(fitted_stm.doc_names) == fitted_stm.doc_topic.shape[0]

    def test_num_topics_after_fit(self, fitted_stm):
        assert fitted_stm.num_topics == 2


# ---------------------------------------------------------------------------
# Topic recovery: each topic's top words come from one disjoint vocabulary
# ---------------------------------------------------------------------------

class TestSTMTopicRecovery:
    def test_a_topic_top_words_dominated_by_vocab_a(self, fitted_stm, a_topic_idx):
        a_set = set(_VOCAB_A)
        top5 = [w for w, _ in fitted_stm.top_words(5, topic=a_topic_idx, weights=True)]
        overlap = sum(1 for w in top5 if w in a_set)
        assert overlap >= 3, (
            f"A topic top-5 words not dominated by _VOCAB_A: {top5}"
        )

    def test_b_topic_top_words_dominated_by_vocab_b(self, fitted_stm, a_topic_idx):
        b_idx = 1 - a_topic_idx
        b_set = set(_VOCAB_B)
        top5 = [w for w, _ in fitted_stm.top_words(5, topic=b_idx, weights=True)]
        overlap = sum(1 for w in top5 if w in b_set)
        assert overlap >= 3, (
            f"B topic top-5 words not dominated by _VOCAB_B: {top5}"
        )

    def test_two_topics_use_distinct_vocabularies(self, fitted_stm):
        """The two topics should not both have the same top word."""
        top0 = {w for w, _ in fitted_stm.top_words(3, topic=0, weights=True)}
        top1 = {w for w, _ in fitted_stm.top_words(3, topic=1, weights=True)}
        assert len(top0 & top1) == 0, (
            f"Topics share top-3 words: {top0 & top1}"
        )


# ---------------------------------------------------------------------------
# Covariate recovery (the headline STM test)
# ---------------------------------------------------------------------------

class TestSTMCovariateRecovery:
    """The STM's distinguishing feature: prevalence covariates shift topic use.

    With a strong binary covariate (x=1 -> A words, x=0 -> B words) and
    separate vocabularies, estimate_effect should find a large positive z for
    x on the A-topic and a large negative z on the B-topic.  In practice z > 300.
    """

    def test_a_topic_z_large_positive(self, fitted_stm, stm_corpus_and_x, a_topic_idx):
        """A-topic's x coefficient has |z| > 3 and is positive."""
        _, x_2d = stm_corpus_and_x
        effects = stm.estimate_effect(
            fitted_stm.doc_topic, x_2d, feature_names=["x"]
        )
        a_eff = effects[a_topic_idx]
        xi = a_eff.feature_names.index("x")
        assert a_eff.z[xi] > 3, (
            f"Expected z > 3 for A topic, got z={a_eff.z[xi]:.2f}"
        )
        assert a_eff.coef[xi] > 0, (
            f"Expected positive coef for A topic (x=1 favors A), got {a_eff.coef[xi]:.4f}"
        )

    def test_b_topic_x_effect_negative(self, fitted_stm, stm_corpus_and_x, a_topic_idx):
        """B-topic's x coefficient is negative (x=1 suppresses B)."""
        _, x_2d = stm_corpus_and_x
        b_idx = 1 - a_topic_idx
        effects = stm.estimate_effect(
            fitted_stm.doc_topic, x_2d, feature_names=["x"]
        )
        b_eff = effects[b_idx]
        xi = b_eff.feature_names.index("x")
        assert b_eff.coef[xi] < 0, (
            f"Expected negative coef for B topic (x=1 suppresses B), got {b_eff.coef[xi]:.4f}"
        )

    def test_a_topic_z_large_negative_on_b(self, fitted_stm, stm_corpus_and_x, a_topic_idx):
        """B-topic's x z-statistic is large and negative."""
        _, x_2d = stm_corpus_and_x
        b_idx = 1 - a_topic_idx
        effects = stm.estimate_effect(
            fitted_stm.doc_topic, x_2d, feature_names=["x"]
        )
        b_eff = effects[b_idx]
        xi = b_eff.feature_names.index("x")
        assert b_eff.z[xi] < -3, (
            f"Expected z < -3 for B topic, got z={b_eff.z[xi]:.2f}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestSTMDeterminism:
    def test_same_seed_identical_topic_word(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m1 = STM(2, seed=42)
        m1.fit(docs, x_2d, prevalence_names=["x"], iters=20)
        m2 = STM(2, seed=42)
        m2.fit(docs, x_2d, prevalence_names=["x"], iters=20)
        assert np.array_equal(m1.topic_word, m2.topic_word), (
            "Same seed must produce identical topic_word"
        )

    def test_same_seed_identical_doc_topic(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m1 = STM(2, seed=42)
        m1.fit(docs, x_2d, prevalence_names=["x"], iters=20)
        m2 = STM(2, seed=42)
        m2.fit(docs, x_2d, prevalence_names=["x"], iters=20)
        assert np.array_equal(m1.doc_topic, m2.doc_topic), (
            "Same seed must produce identical doc_topic"
        )

    def test_same_seed_identical_prevalence_effects(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m1 = STM(2, seed=42)
        m1.fit(docs, x_2d, prevalence_names=["x"], iters=20)
        m2 = STM(2, seed=42)
        m2.fit(docs, x_2d, prevalence_names=["x"], iters=20)
        assert np.array_equal(m1.prevalence_effects, m2.prevalence_effects), (
            "Same seed must produce identical prevalence_effects"
        )


# ---------------------------------------------------------------------------
# Input type parity: numpy 2D array vs list of float lists
# ---------------------------------------------------------------------------

class TestSTMInputTypes:
    def test_numpy_array_input_accepted(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m = STM(2, seed=1)
        m.fit(docs, x_2d, prevalence_names=["x"], iters=10)
        assert m.doc_topic.shape == (250, 2)

    def test_list_of_float_lists_accepted(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        x_list = x_2d.tolist()  # list of [float]
        m = STM(2, seed=1)
        m.fit(docs, x_list, prevalence_names=["x"], iters=10)
        assert m.doc_topic.shape == (250, 2)

    def test_numpy_and_list_give_identical_topic_word(self, stm_corpus_and_x):
        """Same data, same seed: numpy and list input produce the same topic_word."""
        docs, x_2d = stm_corpus_and_x
        x_list = x_2d.tolist()
        m1 = STM(2, seed=5)
        m1.fit(docs, x_2d, prevalence_names=["x"], iters=15)
        m2 = STM(2, seed=5)
        m2.fit(docs, x_list, prevalence_names=["x"], iters=15)
        assert np.array_equal(m1.topic_word, m2.topic_word), (
            "numpy array and list[list[float]] input must produce identical topic_word"
        )

    def test_corpus_input_accepted(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        corpus = Corpus.from_documents(docs)
        m = STM(2, seed=1)
        m.fit(corpus, x_2d, prevalence_names=["x"], iters=10)
        assert m.doc_topic.shape == (250, 2)

    def test_corpus_and_list_give_identical_topic_word(self, stm_corpus_and_x):
        """Corpus and list[list[str]] input (same seed) produce identical topic_word."""
        docs, x_2d = stm_corpus_and_x
        corpus = Corpus.from_documents(docs)
        m1 = STM(2, seed=7)
        m1.fit(corpus, x_2d, prevalence_names=["x"], iters=15)
        m2 = STM(2, seed=7)
        m2.fit(docs, x_2d, prevalence_names=["x"], iters=15)
        assert np.array_equal(m1.topic_word, m2.topic_word)


# ---------------------------------------------------------------------------
# Multiple covariates
# ---------------------------------------------------------------------------

class TestSTMMultipleCovariates:
    @pytest.fixture(scope="class")
    def multi_cov_model(self, stm_corpus_and_x):
        """STM fitted with 2 covariates (x + a random control)."""
        docs, x_2d = stm_corpus_and_x
        rng = np.random.default_rng(77)
        ctrl = rng.standard_normal((250, 1))
        x2 = np.hstack([x_2d, ctrl])
        m = STM(2, seed=1)
        m.fit(docs, x2, iters=30)
        return m

    def test_prevalence_effects_shape_two_covariates(self, multi_cov_model):
        # F=2 -> num_features=3; num_topics-1=1
        assert multi_cov_model.prevalence_effects.shape == (3, 1)

    def test_feature_names_auto_named(self, multi_cov_model):
        assert multi_cov_model.feature_names == ["intercept", "feature_0", "feature_1"]

    def test_feature_names_explicit_names(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        rng = np.random.default_rng(77)
        ctrl = rng.standard_normal((250, 1))
        x2 = np.hstack([x_2d, ctrl])
        m = STM(2, seed=1)
        m.fit(docs, x2, prevalence_names=["treatment", "control"], iters=30)
        assert m.feature_names == ["intercept", "treatment", "control"]

    def test_prevalence_effects_shape_with_names(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        rng = np.random.default_rng(77)
        ctrl = rng.standard_normal((250, 1))
        x2 = np.hstack([x_2d, ctrl])
        m = STM(2, seed=1)
        m.fit(docs, x2, prevalence_names=["x", "ctrl"], iters=30)
        assert m.prevalence_effects.shape == (3, 1)


# ---------------------------------------------------------------------------
# Validation errors in fit()
# ---------------------------------------------------------------------------

class TestSTMFitValidation:
    def test_prevalence_row_count_mismatch_raises(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        bad_x = x_2d[:-5]  # 245 rows, corpus has 250
        m = STM(2, seed=1)
        with pytest.raises(ValueError):
            m.fit(docs, bad_x, iters=5)

    def test_ragged_prevalence_rows_raises(self, stm_corpus_and_x):
        docs, _ = stm_corpus_and_x
        ragged = [[1.0, 2.0]] * 100 + [[1.0]] * 150
        m = STM(2, seed=1)
        with pytest.raises(ValueError):
            m.fit(docs, ragged, iters=5)

    def test_prevalence_names_length_mismatch_raises(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m = STM(2, seed=1)
        with pytest.raises(ValueError):
            m.fit(docs, x_2d, prevalence_names=["x", "wrong_extra"], iters=5)


# ---------------------------------------------------------------------------
# top_words method
# ---------------------------------------------------------------------------

class TestSTMTopWords:
    def test_all_topics_returns_num_topics_lists(self, fitted_stm):
        result = fitted_stm.top_words()
        assert isinstance(result, list)
        assert len(result) == 2
        for topic_list in result:
            assert isinstance(topic_list, list)

    def test_top_words_n_controls_length(self, fitted_stm):
        result = fitted_stm.top_words(3)
        for topic_list in result:
            assert len(topic_list) == 3

    def test_single_topic_returns_list_of_tuples(self, fitted_stm):
        result = fitted_stm.top_words(5, topic=0, weights=True)
        assert isinstance(result, list)
        assert len(result) == 5
        for word, prob in result:
            assert isinstance(word, str)
            assert isinstance(prob, float)

    def test_top_words_probabilities_descending(self, fitted_stm):
        for topic_list in fitted_stm.top_words(5, weights=True):
            probs = [p for _, p in topic_list]
            assert probs == sorted(probs, reverse=True), (
                "top_words probabilities must be sorted descending"
            )

    def test_top_words_out_of_range_raises_value_error(self, fitted_stm):
        with pytest.raises(ValueError):
            fitted_stm.top_words(topic=99)

    def test_top_words_topic_none_vs_explicit_consistent(self, fitted_stm):
        all_results = fitted_stm.top_words(5)
        for t in range(fitted_stm.num_topics):
            single = fitted_stm.top_words(5, topic=t)
            assert single == all_results[t], (
                f"top_words(topic={t}) must match top_words()[{t}]"
            )


# ---------------------------------------------------------------------------
# coherence method
# ---------------------------------------------------------------------------

class TestSTMCoherence:
    def test_coherence_shape(self, fitted_stm):
        c = fitted_stm.coherence(n=5)
        assert c.shape == (2,)

    def test_coherence_default_n(self, fitted_stm):
        c = fitted_stm.coherence()
        assert c.shape == (fitted_stm.num_topics,)

    def test_coherence_values_nonpositive(self, fitted_stm):
        c = fitted_stm.coherence(n=5)
        assert (c <= 0).all(), (
            f"UMass coherence values must be <= 0; got {c}"
        )


# ---------------------------------------------------------------------------
# EM convergence (convergence_tol / bound / converged) — matches R stm's emtol stop
# ---------------------------------------------------------------------------

class TestSTMConvergence:
    def test_bound_history_monotone_increasing(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m = STM(num_topics=2, seed=1)
        m.fit(docs, x_2d, prevalence_names=["x"], iters=200, convergence_tol=1e-6)
        h = m.bound_history
        assert len(h) >= 2
        assert all(h[i + 1] >= h[i] - 1e-6 for i in range(len(h) - 1))
        assert np.isfinite(m.bound)
        assert m.bound == pytest.approx(h[-1])

    def test_converges_before_cap(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m = STM(num_topics=2, seed=1)
        m.fit(docs, x_2d, prevalence_names=["x"], iters=500, convergence_tol=1e-5)
        assert m.converged is True
        assert len(m.bound_history) < 500

    def test_convergence_tol_zero_runs_full_cap(self, stm_corpus_and_x):
        docs, x_2d = stm_corpus_and_x
        m = STM(num_topics=2, seed=1)
        m.fit(docs, x_2d, prevalence_names=["x"], iters=15, convergence_tol=0.0)
        assert m.converged is False
        assert len(m.bound_history) == 15

    def test_convergence_state_survives_save_load(self, stm_corpus_and_x, tmp_path):
        docs, x_2d = stm_corpus_and_x
        m = STM(num_topics=2, seed=1)
        m.fit(docs, x_2d, prevalence_names=["x"], iters=500, convergence_tol=1e-5)
        path = str(tmp_path / "stm.bin")
        m.save(path)
        reloaded = STM.load(path)
        assert reloaded.converged == m.converged
        assert reloaded.bound == pytest.approx(m.bound)
        assert reloaded.bound_history == pytest.approx(m.bound_history)

    def test_accessors_require_fit(self):
        m = STM(num_topics=2)
        with pytest.raises(RuntimeError):
            _ = m.bound


# ---------------------------------------------------------------------------
# spectral_projection_threshold (issue #542)
# ---------------------------------------------------------------------------
# The spectral (anchor-word) init switches from the exact V*V co-occurrence to a
# deterministic random projection once the vocabulary exceeds the threshold. The
# default (10000) matches R stm; lowering it forces the (different) projected path.


class TestSpectralProjectionThreshold:
    def test_default_matches_high_explicit_threshold(self, stm_corpus_and_x):
        # This corpus's vocabulary (10 types) is well below the default 10000, so
        # the default and any high explicit threshold both take the exact path and
        # must be bit-for-bit identical.
        docs, x_2d = stm_corpus_and_x
        m_default = STM(num_topics=2, seed=1)
        m_default.fit(docs, x_2d, prevalence_names=["x"], iters=8)
        m_high = STM(num_topics=2, seed=1)
        m_high.fit(
            docs,
            x_2d,
            prevalence_names=["x"],
            iters=8,
            spectral_projection_threshold=100_000,
        )
        assert np.array_equal(m_default.topic_word, m_high.topic_word)

    def test_low_threshold_changes_init_and_result(self, stm_corpus_and_x):
        # Lowering the threshold below V (10) forces the approximate projected
        # co-occurrence path, which yields a different init and hence a different
        # fit from the exact (default) path.
        docs, x_2d = stm_corpus_and_x
        m_exact = STM(num_topics=2, seed=1)
        m_exact.fit(docs, x_2d, prevalence_names=["x"], iters=8)
        m_proj = STM(num_topics=2, seed=1)
        m_proj.fit(
            docs,
            x_2d,
            prevalence_names=["x"],
            iters=8,
            spectral_projection_threshold=1,
        )
        assert not np.array_equal(m_exact.topic_word, m_proj.topic_word)

    def test_projected_path_is_deterministic(self, stm_corpus_and_x):
        # A fixed (corpus, threshold) gives a reproducible projected init/fit.
        docs, x_2d = stm_corpus_and_x
        a = STM(num_topics=2, seed=1)
        a.fit(docs, x_2d, prevalence_names=["x"], iters=8,
              spectral_projection_threshold=1)
        b = STM(num_topics=2, seed=1)
        b.fit(docs, x_2d, prevalence_names=["x"], iters=8,
              spectral_projection_threshold=1)
        assert np.array_equal(a.topic_word, b.topic_word)
