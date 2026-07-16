"""Tests for LDA goodness-of-fit and diagnostic methods.

Covers: evaluate(), perplexity(), coherence(), diagnostics()
"""

import math

import numpy as np
import pytest
from numpy.random import default_rng

from topica.models import LDA
from topica import Corpus

# ---------------------------------------------------------------------------
# Module-level fixtures / helpers
# ---------------------------------------------------------------------------

# Small toy corpus from conftest (15 animal + 15 space docs, 7 unique words)
# Available as `toy_docs` session fixture.


def _fitted_toy(toy_docs, seed=42, num_topics=2, iters=300, **kwargs):
    """Return a fitted LDA on toy_docs with sensible defaults."""
    model = LDA(num_topics, seed=seed, optimize_interval=0, **kwargs)
    model.fit(toy_docs, iters=iters, num_samples=3, sample_interval=10)
    return model


def _make_cluster_corpus(n_per_cluster=40, doc_len=10, seed=0):
    """Build a well-separated two-cluster corpus and a held-out split.

    Returns (train_docs, held_docs) where train has 2*n_per_cluster docs
    and held has 20 docs.
    """
    animal_pool = [
        "cat", "dog", "fish", "puppy", "kitten",
        "paw", "fur", "tail", "bark", "meow",
    ]
    space_pool = [
        "planet", "star", "moon", "rocket", "orbit",
        "galaxy", "nebula", "comet", "asteroid", "cosmos",
    ]

    rng_train = default_rng(seed)
    rng_held = default_rng(seed + 1)

    def _docs(pool, n, rng):
        return [
            [pool[int(i)] for i in rng.integers(0, len(pool), doc_len)]
            for _ in range(n)
        ]

    train = _docs(animal_pool, n_per_cluster, rng_train) + _docs(space_pool, n_per_cluster, rng_train)
    held = _docs(animal_pool, 10, rng_held) + _docs(space_pool, 10, rng_held)
    return train, held


# ---------------------------------------------------------------------------
# Unfitted-model guards
# ---------------------------------------------------------------------------

class TestUnfittedGuards:
    """All four methods must raise RuntimeError before fit() is called."""

    def _unfitted(self):
        return LDA(2)

    def test_evaluate_raises_before_fit(self, toy_docs):
        with pytest.raises(RuntimeError, match="not fitted"):
            self._unfitted().evaluate(toy_docs[:2])

    def test_perplexity_raises_before_fit(self, toy_docs):
        with pytest.raises(RuntimeError, match="not fitted"):
            self._unfitted().perplexity(toy_docs[:2])

    def test_coherence_raises_before_fit(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            self._unfitted().coherence()

    def test_diagnostics_raises_before_fit(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            self._unfitted().diagnostics()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_num_particles_zero_raises_valueerror(self, toy_docs):
        model = _fitted_toy(toy_docs)
        with pytest.raises(ValueError):
            model.perplexity(toy_docs[:2], num_particles=0)

    def test_num_particles_zero_raises_valueerror_via_evaluate(self, toy_docs):
        model = _fitted_toy(toy_docs)
        with pytest.raises(ValueError):
            model.evaluate(toy_docs[:2], num_particles=0)


# ---------------------------------------------------------------------------
# evaluate() — return type and key structure
# ---------------------------------------------------------------------------

class TestEvaluateStructure:
    def test_returns_dict(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert set(result.keys()) == {"log_likelihood", "perplexity", "num_tokens", "num_oov"}

    def test_log_likelihood_is_float(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert isinstance(result["log_likelihood"], float)

    def test_perplexity_is_float(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert isinstance(result["perplexity"], float)

    def test_num_tokens_is_int(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert isinstance(result["num_tokens"], int)

    def test_num_oov_is_int(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert isinstance(result["num_oov"], int)

    def test_perplexity_positive_and_finite(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert math.isfinite(result["perplexity"])
        assert result["perplexity"] > 0

    def test_num_tokens_positive(self, toy_docs):
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert result["num_tokens"] > 0

    def test_num_oov_zero_for_in_vocab_docs(self, toy_docs):
        """All training tokens are in-vocab, so OOV count should be zero."""
        model = _fitted_toy(toy_docs)
        result = model.evaluate(toy_docs[:5], num_particles=5, seed=42)
        assert result["num_oov"] == 0

    def test_accepts_corpus_object(self, toy_docs, toy_corpus):
        """evaluate() must accept a Corpus as well as list[list[str]]."""
        model = _fitted_toy(toy_corpus)
        result = model.evaluate(toy_corpus, num_particles=5, seed=42)
        assert result["num_tokens"] > 0


# ---------------------------------------------------------------------------
# evaluate() — OOV behavior
# ---------------------------------------------------------------------------

class TestEvaluateOOV:
    def test_all_oov_returns_num_tokens_zero(self, toy_docs):
        model = _fitted_toy(toy_docs)
        oov_docs = [["xyz", "qqqq", "notaword"], ["blah", "blah"]]
        result = model.evaluate(oov_docs, num_particles=5, seed=42)
        assert result["num_tokens"] == 0

    def test_all_oov_num_oov_positive(self, toy_docs):
        model = _fitted_toy(toy_docs)
        oov_docs = [["xyz", "qqqq", "notaword"], ["blah", "blah"]]
        result = model.evaluate(oov_docs, num_particles=5, seed=42)
        assert result["num_oov"] > 0

    def test_all_oov_perplexity_is_nan(self, toy_docs):
        model = _fitted_toy(toy_docs)
        oov_docs = [["xyz", "qqqq", "notaword"], ["blah", "blah"]]
        result = model.evaluate(oov_docs, num_particles=5, seed=42)
        assert math.isnan(result["perplexity"])

    def test_partial_oov_counts_correctly(self, toy_docs):
        """OOV tokens are dropped; only in-vocab tokens are scored."""
        model = _fitted_toy(toy_docs)
        # 2 in-vocab tokens, 2 OOV tokens
        partial_oov = [["cat", "xyz", "dog", "unknown"]]
        result = model.evaluate(partial_oov, num_particles=5, seed=42)
        assert result["num_tokens"] == 2
        assert result["num_oov"] == 2

    def test_all_oov_num_oov_matches_total_token_count(self, toy_docs):
        """When every token is OOV, num_oov should equal total tokens across docs."""
        model = _fitted_toy(toy_docs)
        oov_docs = [["xyz", "zzz"], ["abc"]]
        result = model.evaluate(oov_docs, num_particles=5, seed=42)
        assert result["num_oov"] == 3  # 2 + 1


# ---------------------------------------------------------------------------
# perplexity() — convenience wrapper
# ---------------------------------------------------------------------------

class TestPerplexityWrapper:
    def test_returns_float(self, toy_docs):
        model = _fitted_toy(toy_docs)
        p = model.perplexity(toy_docs[:5], num_particles=5, seed=42)
        assert isinstance(p, float)

    def test_matches_evaluate_perplexity_key(self, toy_docs):
        """perplexity() must return the same value as evaluate()['perplexity']."""
        model = _fitted_toy(toy_docs)
        data = toy_docs[:8]
        result = model.evaluate(data, num_particles=10, seed=99)
        p = model.perplexity(data, num_particles=10, seed=99)
        assert p == pytest.approx(result["perplexity"], rel=1e-9)

    def test_deterministic_for_fixed_seed(self, toy_docs):
        """Two calls with the same seed must return identical values."""
        model = _fitted_toy(toy_docs)
        p1 = model.perplexity(toy_docs[:5], num_particles=10, seed=7)
        p2 = model.perplexity(toy_docs[:5], num_particles=10, seed=7)
        assert p1 == p2

    def test_perplexity_positive_finite(self, toy_docs):
        model = _fitted_toy(toy_docs)
        p = model.perplexity(toy_docs[:5], num_particles=5, seed=42)
        assert math.isfinite(p) and p > 0


# ---------------------------------------------------------------------------
# perplexity() — recovers true K
# ---------------------------------------------------------------------------

class TestPerplexityRecoversK:
    """Held-out perplexity should be lower at the true k=2 than at k=1 or k=5
    on a clearly two-cluster corpus."""

    @pytest.fixture(scope="class")
    def cluster_data(self):
        return _make_cluster_corpus(n_per_cluster=40, doc_len=10, seed=0)

    def _fit(self, k, train_docs):
        model = LDA(k, seed=42, optimize_interval=0)
        model.fit(train_docs, iters=300, num_samples=3, sample_interval=10)
        return model

    def test_k2_lower_than_k1(self, cluster_data):
        train, held = cluster_data
        p1 = self._fit(1, train).perplexity(held, num_particles=10, seed=42)
        p2 = self._fit(2, train).perplexity(held, num_particles=10, seed=42)
        assert p2 < p1, f"Expected k=2 perplexity ({p2:.2f}) < k=1 ({p1:.2f})"

    def test_k2_lower_than_k5(self, cluster_data):
        train, held = cluster_data
        p2 = self._fit(2, train).perplexity(held, num_particles=10, seed=42)
        p5 = self._fit(5, train).perplexity(held, num_particles=10, seed=42)
        assert p2 < p5, f"Expected k=2 perplexity ({p2:.2f}) < k=5 ({p5:.2f})"


# ---------------------------------------------------------------------------
# perplexity() — held-out >= training
# ---------------------------------------------------------------------------

class TestHeldOutVsTrainingPerplexity:
    def test_held_out_ge_training(self, toy_docs):
        """Held-out evaluation should be at least as hard as training."""
        # Use different in-distribution docs for the held-out set
        animal = [["cat", "dog", "fish"]] * 5
        space = [["planet", "star", "moon"]] * 5
        held_docs = animal + space

        model = _fitted_toy(toy_docs, seed=42, iters=300)
        train_perp = model.perplexity(toy_docs, num_particles=10, seed=42)
        held_perp = model.perplexity(held_docs, num_particles=10, seed=42)
        assert held_perp >= train_perp, (
            f"Held-out perplexity ({held_perp:.4f}) should be >= "
            f"training perplexity ({train_perp:.4f})"
        )


# ---------------------------------------------------------------------------
# coherence()
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_shape_equals_num_topics(self, toy_docs):
        model = _fitted_toy(toy_docs)
        c = model.coherence(10)
        assert c.shape == (model.num_topics,)

    def test_returns_numpy_array(self, toy_docs):
        model = _fitted_toy(toy_docs)
        c = model.coherence(10)
        assert isinstance(c, np.ndarray)

    def test_values_are_finite(self, toy_docs):
        model = _fitted_toy(toy_docs)
        c = model.coherence(10)
        assert np.all(np.isfinite(c))

    def test_umass_values_nonpositive(self, toy_docs):
        """UMass coherence is non-positive (log of ratio <= 1)."""
        model = _fitted_toy(toy_docs)
        c = model.coherence(10)
        assert np.all(c <= 0), f"Expected all UMass coherences <= 0, got {c}"

    def test_shape_with_different_n(self, toy_docs):
        model = _fitted_toy(toy_docs)
        c5 = model.coherence(5)
        c3 = model.coherence(3)
        assert c5.shape == (2,)
        assert c3.shape == (2,)

    def test_coherence_higher_for_good_model(self, toy_docs):
        """A well-separated model should have higher (less-negative) coherence
        than a random-looking single-topic model."""
        train, _ = _make_cluster_corpus(n_per_cluster=40, doc_len=10, seed=0)
        good_model = LDA(2, seed=42, optimize_interval=0)
        good_model.fit(train, iters=300, num_samples=3, sample_interval=10)
        bad_model = LDA(1, seed=42, optimize_interval=0)
        bad_model.fit(train, iters=300, num_samples=3, sample_interval=10)
        # k=2 mean coherence should be closer to 0 than k=1
        assert np.mean(good_model.coherence(5)) > np.mean(bad_model.coherence(5))


# ---------------------------------------------------------------------------
# diagnostics()
# ---------------------------------------------------------------------------

DIAG_KEYS = frozenset(
    {"topic", "tokens", "coherence", "exclusivity", "effective_words",
     "document_entropy", "uniform_dist", "corpus_dist",
     "rank1_docs", "alpha", "top_words"}
)


class TestDiagnosticsStructure:
    def test_returns_list(self, toy_docs):
        model = _fitted_toy(toy_docs)
        d = model.diagnostics()
        assert isinstance(d, list)

    def test_length_equals_num_topics(self, toy_docs):
        model = _fitted_toy(toy_docs)
        d = model.diagnostics()
        assert len(d) == model.num_topics

    def test_each_element_is_dict(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics():
            assert isinstance(di, dict)

    def test_each_dict_has_exact_keys(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics():
            assert set(di.keys()) == DIAG_KEYS

    def test_topic_indices_are_ints(self, toy_docs):
        model = _fitted_toy(toy_docs)
        topics = [di["topic"] for di in model.diagnostics()]
        assert all(isinstance(t, int) for t in topics)

    def test_tokens_are_ints(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics():
            assert isinstance(di["tokens"], int)

    def test_top_words_is_list_of_str(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics(n=5):
            assert isinstance(di["top_words"], list)
            assert all(isinstance(w, str) for w in di["top_words"])

    def test_top_words_length_at_most_n(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics(n=4):
            assert len(di["top_words"]) <= 4

    def test_exclusivity_in_unit_interval(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics():
            assert 0.0 <= di["exclusivity"] <= 1.0, (
                f"exclusivity out of [0,1]: {di['exclusivity']}"
            )

    def test_effective_words_ge_one(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics():
            assert di["effective_words"] >= 1.0, (
                f"effective_words < 1: {di['effective_words']}"
            )

    def test_rank1_docs_sum_le_num_docs(self, toy_docs):
        model = _fitted_toy(toy_docs)
        total = sum(di["rank1_docs"] for di in model.diagnostics())
        num_docs = len(toy_docs)
        assert total <= num_docs, (
            f"rank1_docs sum ({total}) > num_docs ({num_docs})"
        )

    def test_rank1_docs_are_ints(self, toy_docs):
        model = _fitted_toy(toy_docs)
        for di in model.diagnostics():
            assert isinstance(di["rank1_docs"], int)


class TestDiagnosticsSemantics:
    """Content checks on the well-separated two-cluster corpus."""

    @pytest.fixture(scope="class")
    def cluster_model(self):
        train, _ = _make_cluster_corpus(n_per_cluster=40, doc_len=10, seed=0)
        m = LDA(2, seed=42, optimize_interval=0)
        m.fit(train, iters=300, num_samples=3, sample_interval=10)
        return m

    def test_top_words_disjoint_between_topics(self, cluster_model):
        """Perfectly separated clusters should have non-overlapping top words."""
        d = cluster_model.diagnostics(n=5)
        words0 = set(d[0]["top_words"])
        words1 = set(d[1]["top_words"])
        assert words0.isdisjoint(words1), (
            f"Top words not disjoint: {words0} ∩ {words1} = {words0 & words1}"
        )

    def test_exclusivity_high_for_clean_separation(self, cluster_model):
        """With perfectly non-overlapping vocabularies, exclusivity should be high."""
        d = cluster_model.diagnostics(n=5)
        for di in d:
            assert di["exclusivity"] >= 0.9, (
                f"Expected high exclusivity for clean cluster, got {di['exclusivity']:.3f}"
            )

    def test_coherence_in_diagnostics_matches_coherence_method(self, cluster_model):
        """coherence field in diagnostics() should match coherence() per topic."""
        d = cluster_model.diagnostics(n=10)
        c_array = cluster_model.coherence(10)
        for di in d:
            topic_idx = di["topic"]
            assert di["coherence"] == pytest.approx(c_array[topic_idx], rel=1e-6)
