"""Tests for LDA determinism and progress callback."""

import numpy as np
import pytest

from topica.models import LDA


# Toy corpus used for determinism tests (same as conftest but self-contained
# so this module can be run in isolation).
_ANIMAL_DOCS = [["cat", "dog", "fish", "cat", "dog"] for _ in range(15)]
_SPACE_DOCS = [["planet", "star", "moon", "rocket", "planet"] for _ in range(15)]
_TOY_DOCS = _ANIMAL_DOCS + _SPACE_DOCS


def _fit(seed: int, num_topics: int = 2, **kwargs) -> np.ndarray:
    """Return topic_word from a quickly fitted model."""
    model = LDA(num_topics, seed=seed, optimize_interval=0, **kwargs)
    model.fit(
        _TOY_DOCS,
        iters=100,
        num_samples=2,
        sample_interval=5,
    )
    return model.topic_word


# ---------------------------------------------------------------------------
# Same seed → identical results
# ---------------------------------------------------------------------------

class TestSameSeed:
    def test_same_seed_identical_topic_word(self):
        tw1 = _fit(seed=42)
        tw2 = _fit(seed=42)
        assert np.array_equal(tw1, tw2), (
            "Two runs with the same seed produced different topic_word matrices"
        )

    def test_same_seed_identical_doc_topic(self):
        model_a = LDA(2, seed=7, optimize_interval=0)
        model_b = LDA(2, seed=7, optimize_interval=0)
        kw = dict(iters=100, num_samples=2, sample_interval=5)
        model_a.fit(_TOY_DOCS, **kw)
        model_b.fit(_TOY_DOCS, **kw)
        assert np.array_equal(model_a.doc_topic, model_b.doc_topic)


# ---------------------------------------------------------------------------
# Different seed → different results
# ---------------------------------------------------------------------------

class TestDifferentSeeds:
    def test_different_seeds_different_topic_word(self):
        # Over-specify K (5 topics for a 2-cluster corpus): the extra topics
        # are split seed-dependently, so the seed genuinely affects the fit. On
        # a fully identified corpus (K = number of clusters) the sampler
        # converges to the unique solution for any seed, which is correct but
        # makes "different seeds differ" untestable.
        tw42 = _fit(seed=42, num_topics=5)
        tw99 = _fit(seed=99, num_topics=5)
        assert not np.array_equal(tw42, tw99), (
            "Two runs with different seeds produced identical topic_word matrices "
            "(extremely unlikely to be correct)"
        )


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_callback_invoked(self):
        """Progress callback must be called at least once."""
        calls = []

        def cb(iteration, ll):
            calls.append((iteration, ll))

        model = LDA(2, seed=42, optimize_interval=0)
        model.fit(
            _TOY_DOCS,
            iters=100,
            num_samples=2,
            sample_interval=5,
            progress=cb,
            progress_interval=50,
        )
        assert len(calls) > 0

    def test_callback_receives_int_iteration(self):
        """First element of each callback tuple must be an int."""
        calls = []

        def cb(iteration, ll):
            calls.append((iteration, ll))

        model = LDA(2, seed=42, optimize_interval=0)
        model.fit(
            _TOY_DOCS,
            iters=100,
            num_samples=2,
            sample_interval=5,
            progress=cb,
            progress_interval=20,
        )
        assert all(isinstance(i, int) for i, _ in calls)

    def test_callback_receives_float_ll(self):
        """Second element of each callback tuple must be a float."""
        calls = []

        def cb(iteration, ll):
            calls.append((iteration, ll))

        model = LDA(2, seed=42, optimize_interval=0)
        model.fit(
            _TOY_DOCS,
            iters=100,
            num_samples=2,
            sample_interval=5,
            progress=cb,
            progress_interval=20,
        )
        assert all(isinstance(ll, float) for _, ll in calls)

    def test_callback_cadence(self):
        """Callback should fire every progress_interval iterations (approx)."""
        calls = []

        def cb(iteration, ll):
            calls.append(iteration)

        model = LDA(2, seed=42, optimize_interval=0)
        model.fit(
            _TOY_DOCS,
            iters=100,
            num_samples=2,
            sample_interval=5,
            progress=cb,
            progress_interval=25,
        )
        # Expect calls at multiples of 25: 25, 50, 75, 100 → 4 calls
        assert len(calls) == 4
        for expected, actual in zip([25, 50, 75, 100], calls):
            assert actual == expected

    def test_no_callback_when_none(self):
        """Passing progress=None should not raise."""
        model = LDA(2, seed=42, optimize_interval=0)
        model.fit(
            _TOY_DOCS,
            iters=50,
            num_samples=2,
            sample_interval=5,
            progress=None,
        )
        # just verify it completes without error
        assert model.topic_word.shape == (2, 7)
