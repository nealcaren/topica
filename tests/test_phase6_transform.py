"""Phase 6: held-out transform for KeyATM, SeededLDA, SAGE, PA, PT.

Tests that each model's transform:
- returns (n, K) array with rows summing to ~1
- K equals num_topics (num_sub for PA)
- drops OOV tokens silently
- a document with no in-vocabulary tokens returns a valid prior row
- is deterministic given the same seed

Payoff: topica.diagnostics.perplexity and topica.diagnostics.eval_heldout now work for KeyATM and
SeededLDA (they previously raised "no transform").
"""

import numpy as np
import pytest

import topica

# ---------------------------------------------------------------------------
# Shared tiny corpus
# ---------------------------------------------------------------------------
DOCS = [
    ["cat", "cat", "dog", "pet", "animal"],
    ["dog", "dog", "cat", "pet", "walk"],
    ["code", "python", "rust", "program", "software"],
    ["python", "code", "rust", "software", "compile"],
    ["cat", "dog", "code", "python", "pet"],
    ["animal", "walk", "rust", "program", "compile"],
    ["cat", "animal", "pet", "dog", "walk"],
    ["code", "software", "program", "rust", "python"],
]

HELD_OUT = [
    ["cat", "pet", "animal"],
    ["code", "python", "software"],
    ["unknown_word_xyz", "another_oov"],  # all OOV
    ["cat", "code"],
]


def _rows_sum_to_one(arr, atol=1e-6):
    return np.allclose(arr.sum(axis=1), 1.0, atol=atol)


# ---------------------------------------------------------------------------
# KeyATM
# ---------------------------------------------------------------------------

class TestKeyATMTransform:
    @pytest.fixture(scope="class")
    def model(self):
        m = topica.models.KeyATM({"animals": ["cat", "dog"], "tech": ["code", "python"]},
                          num_topics=2, seed=7)
        m.fit(DOCS, iters=50)
        return m

    def test_shape(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert out.shape == (len(HELD_OUT), model.num_topics)

    def test_rows_sum_to_one(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert _rows_sum_to_one(out)

    def test_oov_only_doc_is_valid(self, model):
        """A document with no in-vocabulary tokens gets the prior row (sums to 1)."""
        out = model.transform([["unknown_xyz"]], iters=20, burn_in=5,
                              num_samples=5, sample_interval=2, seed=0)
        assert out.shape == (1, model.num_topics)
        assert _rows_sum_to_one(out)

    def test_deterministic(self, model):
        kw = dict(iters=20, burn_in=5, num_samples=5, sample_interval=2, seed=42)
        a = model.transform(HELD_OUT, **kw)
        b = model.transform(HELD_OUT, **kw)
        np.testing.assert_array_equal(a, b)

    def test_perplexity_works(self, model):
        pp = topica.diagnostics.perplexity(model, HELD_OUT[:2], seed=0)
        assert np.isfinite(pp) and pp > 0

    def test_eval_heldout_works(self, model):
        heldout = topica.diagnostics.make_heldout(DOCS, seed=1)
        m2 = topica.models.KeyATM({"animals": ["cat", "dog"], "tech": ["code", "python"]},
                           num_topics=2, seed=7)
        m2.fit(heldout.documents, iters=50)
        result = topica.diagnostics.eval_heldout(m2, heldout, seed=0)
        assert np.isfinite(result.mean_per_doc_loglik)


# ---------------------------------------------------------------------------
# SeededLDA
# ---------------------------------------------------------------------------

class TestSeededLDATransform:
    @pytest.fixture(scope="class")
    def model(self):
        m = topica.models.SeededLDA({"animals": ["cat", "dog"], "tech": ["code", "python"]},
                             seed=7)
        m.fit(DOCS, iters=50)
        return m

    def test_shape(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert out.shape == (len(HELD_OUT), model.num_topics)

    def test_rows_sum_to_one(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert _rows_sum_to_one(out)

    def test_oov_only_doc_is_valid(self, model):
        out = model.transform([["unknown_xyz"]], iters=20, burn_in=5,
                              num_samples=5, sample_interval=2, seed=0)
        assert out.shape == (1, model.num_topics)
        assert _rows_sum_to_one(out)

    def test_deterministic(self, model):
        kw = dict(iters=20, burn_in=5, num_samples=5, sample_interval=2, seed=42)
        a = model.transform(HELD_OUT, **kw)
        b = model.transform(HELD_OUT, **kw)
        np.testing.assert_array_equal(a, b)

    def test_perplexity_works(self, model):
        pp = topica.diagnostics.perplexity(model, HELD_OUT[:2], seed=0)
        assert np.isfinite(pp) and pp > 0

    def test_eval_heldout_works(self, model):
        heldout = topica.diagnostics.make_heldout(DOCS, seed=1)
        m2 = topica.models.SeededLDA({"animals": ["cat", "dog"], "tech": ["code", "python"]},
                               seed=7)
        m2.fit(heldout.documents, iters=50)
        result = topica.diagnostics.eval_heldout(m2, heldout, seed=0)
        assert np.isfinite(result.mean_per_doc_loglik)


# ---------------------------------------------------------------------------
# SAGE
# ---------------------------------------------------------------------------

class TestSAGETransform:
    @pytest.fixture(scope="class")
    def model(self):
        m = topica.models.SAGE(2, seed=7)
        groups = ["a", "b", "a", "b", "a", "b", "a", "b"]
        m.fit(DOCS, groups, iters=50)
        return m

    def test_shape(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert out.shape == (len(HELD_OUT), model.num_topics)

    def test_rows_sum_to_one(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert _rows_sum_to_one(out)

    def test_oov_only_doc_is_valid(self, model):
        out = model.transform([["unknown_xyz"]], iters=20, burn_in=5,
                              num_samples=5, sample_interval=2, seed=0)
        assert out.shape == (1, model.num_topics)
        assert _rows_sum_to_one(out)

    def test_deterministic(self, model):
        kw = dict(iters=20, burn_in=5, num_samples=5, sample_interval=2, seed=42)
        a = model.transform(HELD_OUT, **kw)
        b = model.transform(HELD_OUT, **kw)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# PA
# ---------------------------------------------------------------------------

class TestPATransform:
    @pytest.fixture(scope="class")
    def model(self):
        m = topica.models.PA(num_super=2, num_sub=3, seed=7)
        m.fit(DOCS, iters=50)
        return m

    def test_shape_is_num_sub(self, model):
        """PA transform returns (n, num_sub), not (n, num_super)."""
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert out.shape == (len(HELD_OUT), model.num_sub)

    def test_rows_sum_to_one(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert _rows_sum_to_one(out)

    def test_oov_only_doc_is_valid(self, model):
        out = model.transform([["unknown_xyz"]], iters=20, burn_in=5,
                              num_samples=5, sample_interval=2, seed=0)
        assert out.shape == (1, model.num_sub)
        assert _rows_sum_to_one(out)

    def test_deterministic(self, model):
        kw = dict(iters=20, burn_in=5, num_samples=5, sample_interval=2, seed=42)
        a = model.transform(HELD_OUT, **kw)
        b = model.transform(HELD_OUT, **kw)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# PT
# ---------------------------------------------------------------------------

class TestPTTransform:
    @pytest.fixture(scope="class")
    def model(self):
        m = topica.models.PT(num_topics=2, num_pseudo=4, seed=7)
        m.fit(DOCS, iters=50)
        return m

    def test_shape(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert out.shape == (len(HELD_OUT), model.num_topics)

    def test_rows_sum_to_one(self, model):
        out = model.transform(HELD_OUT, iters=20, burn_in=5, num_samples=5,
                              sample_interval=2, seed=0)
        assert _rows_sum_to_one(out)

    def test_oov_only_doc_is_valid(self, model):
        out = model.transform([["unknown_xyz"]], iters=20, burn_in=5,
                              num_samples=5, sample_interval=2, seed=0)
        assert out.shape == (1, model.num_topics)
        assert _rows_sum_to_one(out)

    def test_deterministic(self, model):
        kw = dict(iters=20, burn_in=5, num_samples=5, sample_interval=2, seed=42)
        a = model.transform(HELD_OUT, **kw)
        b = model.transform(HELD_OUT, **kw)
        np.testing.assert_array_equal(a, b)
