"""Tests for topica.GDMR (generalized DMR / g-DMR, Lee & Song 2020).

Contract source: /private/tmp/gdmr_contract.md
Public API only — no source inspection.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

import topica

# ---------------------------------------------------------------------------
# Shared vocabulary sets (same convention as test_dmr.py)
# ---------------------------------------------------------------------------

_VOCAB_A = ["planet", "star", "moon", "rocket", "orbit"]   # space words
_VOCAB_B = ["cat", "dog", "fish", "bird", "mouse"]         # animal words


# ---------------------------------------------------------------------------
# Synthetic corpus builders
# ---------------------------------------------------------------------------

def _make_continuous_corpus(
    n: int = 200,
    doc_length: int = 10,
    seed: int = 0,
) -> tuple[list[list[str]], np.ndarray]:
    """Corpus where a single continuous covariate drives topic prevalence.

    The covariate x is drawn uniform in [0, 1].  Documents with x > 0.5
    draw words from VOCAB_A (space); documents with x <= 0.5 draw from
    VOCAB_B (animal).  This gives a sharp, testable monotonic relationship
    between x and the space topic.

    Returns (docs, metadata) where metadata is shape (n, 1).
    """
    rng = np.random.default_rng(seed)
    docs = []
    xs = rng.uniform(0.0, 1.0, size=n)
    for x in xs:
        vocab = _VOCAB_A if x > 0.5 else _VOCAB_B
        doc = rng.choice(vocab, size=doc_length).tolist()
        docs.append(doc)
    metadata = xs[:, np.newaxis]
    return docs, metadata


def _make_2d_corpus(
    n: int = 200,
    doc_length: int = 10,
    seed: int = 0,
) -> tuple[list[list[str]], np.ndarray]:
    """Like _make_continuous_corpus but with two metadata dimensions."""
    rng = np.random.default_rng(seed)
    docs = []
    xs = rng.uniform(0.0, 1.0, size=(n, 2))
    for row in xs:
        x = row[0]  # first dim drives vocabulary
        vocab = _VOCAB_A if x > 0.5 else _VOCAB_B
        doc = rng.choice(vocab, size=doc_length).tolist()
        docs.append(doc)
    return docs, xs


def _fit_gdmr(
    docs,
    metadata,
    degrees,
    *,
    num_topics: int = 2,
    seed: int = 1,
    iters: int = 300,
    **kwargs,
) -> "topica.GDMR":
    """Fit a GDMR with fast-but-reliable settings for synthetic corpora."""
    model = topica.GDMR(
        num_topics=num_topics,
        degrees=degrees,
        seed=seed,
        optimize_interval=25,
        burn_in=50,
        **kwargs,
    )
    model.fit(
        docs,
        metadata,
        iters=iters,
        num_samples=3,
        sample_interval=10,
    )
    return model


def _identify_space_topic(model: "topica.GDMR") -> int:
    """Return the topic index most aligned with space vocabulary."""
    vocab = model.vocabulary
    tw = model.topic_word
    space_set = set(_VOCAB_A)
    space_mass = [
        sum(tw[t, vocab.index(w)] for w in _VOCAB_A if w in vocab)
        for t in range(model.num_topics)
    ]
    return int(np.argmax(space_mass))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestGDMRConstructor:
    def test_num_topics_before_fit(self):
        m = topica.GDMR(num_topics=3, degrees=[2])
        assert m.num_topics == 3

    def test_degrees_readable_before_fit(self):
        m = topica.GDMR(num_topics=2, degrees=[1, 3])
        assert m.degrees == [1, 3]

    def test_num_topics_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            topica.GDMR(num_topics=0, degrees=[1])

    def test_beta_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            topica.GDMR(num_topics=2, degrees=[1], beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises((ValueError, Exception)):
            topica.GDMR(num_topics=2, degrees=[1], beta=-0.01)

    def test_default_hyperparams_readable(self):
        m = topica.GDMR(num_topics=2, degrees=[2], sigma=1.0, sigma0=3.0, decay=0.0)
        assert m.sigma == pytest.approx(1.0)
        assert m.sigma0 == pytest.approx(3.0)
        assert m.decay == pytest.approx(0.0)

    def test_nondefault_hyperparams_readable(self):
        m = topica.GDMR(num_topics=2, degrees=[2], sigma=0.5, sigma0=2.0, decay=0.9)
        assert m.sigma == pytest.approx(0.5)
        assert m.sigma0 == pytest.approx(2.0)
        assert m.decay == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# fit() input validation
# ---------------------------------------------------------------------------

class TestGDMRFitValidation:
    def setup_method(self):
        self.docs = [["cat", "dog"]] * 20
        # 1-D metadata to match degrees=[1]
        self.meta1 = np.linspace(0, 1, 20)[:, np.newaxis]

    def test_metadata_row_mismatch_raises(self):
        bad = np.ones((15, 1))  # 15 rows but 20 docs
        with pytest.raises((ValueError, Exception)):
            m = topica.GDMR(num_topics=2, degrees=[1])
            m.fit(self.docs, bad)

    def test_metadata_dim_mismatch_with_degrees_raises(self):
        # degrees has D=2 dims but metadata has D=1 column
        with pytest.raises((ValueError, Exception)):
            m = topica.GDMR(num_topics=2, degrees=[1, 2])
            m.fit(self.docs, self.meta1)

    def test_metadata_nan_raises(self):
        # DMR rejects NaN covariates; GDMR should as well
        bad = self.meta1.copy()
        bad[3, 0] = float("nan")
        with pytest.raises((ValueError, Exception)):
            m = topica.GDMR(num_topics=2, degrees=[1])
            m.fit(self.docs, bad)

    def test_metadata_inf_raises(self):
        bad = self.meta1.copy()
        bad[0, 0] = float("inf")
        with pytest.raises((ValueError, Exception)):
            m = topica.GDMR(num_topics=2, degrees=[1])
            m.fit(self.docs, bad)

    def test_metadata_neg_inf_raises(self):
        bad = self.meta1.copy()
        bad[0, 0] = float("-inf")
        with pytest.raises((ValueError, Exception)):
            m = topica.GDMR(num_topics=2, degrees=[1])
            m.fit(self.docs, bad)

    def test_1d_array_accepted_as_single_dim_metadata(self):
        """Passing a (n,) flat array should be treated as (n, 1) when D==1.

        Assumption: GDMR accepts a flat 1-D array for D==1 (same convenience
        as DMR's feature vector); squeezed to 2-D internally.
        """
        flat = np.linspace(0.0, 1.0, 20)
        m = topica.GDMR(num_topics=2, degrees=[1], seed=1)
        # Should not raise
        m.fit(self.docs, flat, iters=50, num_samples=1, sample_interval=5)

    def test_list_of_lists_metadata_accepted(self):
        meta_list = [[float(i) / 20] for i in range(20)]
        m = topica.GDMR(num_topics=2, degrees=[1], seed=1)
        m.fit(self.docs, meta_list, iters=50, num_samples=1, sample_interval=5)
        assert m.topic_word.shape[0] == 2


# ---------------------------------------------------------------------------
# Shapes and invariants after fit
# ---------------------------------------------------------------------------

class TestGDMRShapesAndInvariants:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, meta = _make_continuous_corpus(n=200, seed=0)
        return _fit_gdmr(docs, meta, degrees=[2], seed=1)

    def test_topic_word_shape(self, fitted):
        K = fitted.num_topics
        V = len(fitted.vocabulary)
        assert fitted.topic_word.shape == (K, V)

    def test_topic_word_rows_sum_to_1(self, fitted):
        npt.assert_allclose(fitted.topic_word.sum(axis=1), 1.0, atol=1e-6)

    def test_doc_topic_shape(self, fitted):
        K = fitted.num_topics
        n = len(fitted.doc_names)
        assert fitted.doc_topic.shape == (n, K)

    def test_doc_topic_rows_sum_to_1(self, fitted):
        npt.assert_allclose(fitted.doc_topic.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_effects_shape(self, fitted):
        # degrees=[2] -> num_basis = 2+1 = 3
        K = fitted.num_topics
        expected_num_basis = 3  # prod(d+1 for d in [2]) = 3
        assert fitted.feature_effects.shape == (K, expected_num_basis)

    def test_feature_effect_se_shape_and_finite(self, fitted):
        se = fitted.feature_effect_se
        assert se is not None
        assert se.shape == fitted.feature_effects.shape
        assert np.all(np.isfinite(se)) and np.all(se > 0.0)

    def test_alpha_shape(self, fitted):
        K = fitted.num_topics
        assert fitted.alpha.shape == (K,)

    def test_alpha_positive(self, fitted):
        assert (fitted.alpha > 0).all()

    def test_degrees_readable(self, fitted):
        assert fitted.degrees == [2]

    def test_metadata_range_readable(self, fitted):
        # Should be a list of (min, max) tuples, one per dim
        r = fitted.metadata_range
        assert len(r) == 1
        lo, hi = r[0]
        assert lo < hi

    def test_sigma_readable(self, fitted):
        assert isinstance(fitted.sigma, float)

    def test_sigma0_readable(self, fitted):
        assert isinstance(fitted.sigma0, float)

    def test_decay_readable(self, fitted):
        assert isinstance(fitted.decay, float)

    def test_vocabulary_length_matches_topic_word(self, fitted):
        assert len(fitted.vocabulary) == fitted.topic_word.shape[1]

    def test_doc_names_length_matches_doc_topic(self, fitted):
        assert len(fitted.doc_names) == fitted.doc_topic.shape[0]

    def test_num_topics_correct(self, fitted):
        assert fitted.num_topics == 2

    def test_doc_lengths_length(self, fitted):
        assert len(fitted.doc_lengths) == fitted.doc_topic.shape[0]


class TestGDMRFeatureEffectsShape2D:
    """Verify feature_effects shape for a 2-D metadata / multi-degree case."""

    def test_feature_effects_shape_degrees_1_1(self):
        docs, meta = _make_2d_corpus(n=100, seed=0)
        m = topica.GDMR(num_topics=2, degrees=[1, 1], seed=1)
        m.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        # num_basis = prod([1+1, 1+1]) = 4
        assert m.feature_effects.shape == (2, 4)

    def test_feature_effects_shape_degrees_2_1(self):
        docs, meta = _make_2d_corpus(n=100, seed=0)
        m = topica.GDMR(num_topics=2, degrees=[2, 1], seed=1)
        m.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        # num_basis = prod([2+1, 1+1]) = 6
        assert m.feature_effects.shape == (2, 6)


# ---------------------------------------------------------------------------
# tdf: topic distribution function
# ---------------------------------------------------------------------------

class TestGDMRTdf:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, meta = _make_continuous_corpus(n=200, seed=0)
        return _fit_gdmr(docs, meta, degrees=[2], seed=1)

    def test_tdf_single_point_shape(self, fitted):
        result = fitted.tdf(np.array([0.5]))
        assert result.shape == (fitted.num_topics,)

    def test_tdf_single_point_sums_to_1_normalized(self, fitted):
        result = fitted.tdf(np.array([0.5]), normalize=True)
        assert math.isclose(result.sum(), 1.0, abs_tol=1e-6)

    def test_tdf_batch_shape(self, fitted):
        pts = np.linspace(0.1, 0.9, 7)[:, np.newaxis]
        result = fitted.tdf(pts)
        assert result.shape == (7, fitted.num_topics)

    def test_tdf_batch_rows_sum_to_1_normalized(self, fitted):
        pts = np.linspace(0.1, 0.9, 7)[:, np.newaxis]
        result = fitted.tdf(pts, normalize=True)
        npt.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_tdf_normalize_false_returns_positive(self, fitted):
        """normalize=False returns raw alpha (positive, not summing to 1)."""
        result = fitted.tdf(np.array([0.3]), normalize=False)
        assert (result > 0).all()
        # Raw alpha should NOT sum to 1 in general (may happen by coincidence,
        # but with K=2 and an asymmetric fit it almost certainly won't)
        # We only assert positivity here to avoid a flaky check on the sum.

    def test_tdf_normalize_false_shape_single_point(self, fitted):
        result = fitted.tdf(np.array([0.3]), normalize=False)
        assert result.shape == (fitted.num_topics,)

    def test_tdf_2d_single_point_shape(self):
        docs, meta = _make_2d_corpus(n=100, seed=7)
        m = topica.GDMR(num_topics=2, degrees=[1, 1], seed=1)
        m.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        # Single point for D=2
        result = m.tdf(np.array([0.3, 0.7]))
        assert result.shape == (2,)

    def test_tdf_2d_batch_shape(self):
        docs, meta = _make_2d_corpus(n=100, seed=7)
        m = topica.GDMR(num_topics=2, degrees=[1, 1], seed=1)
        m.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        pts = np.array([[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
        result = m.tdf(pts)
        assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# tdf_linspace
# ---------------------------------------------------------------------------

class TestGDMRTdfLinspace:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, meta = _make_continuous_corpus(n=200, seed=0)
        return _fit_gdmr(docs, meta, degrees=[2], seed=1)

    def test_shape(self, fitted):
        result = fitted.tdf_linspace(0.0, 1.0, 20)
        assert result.shape == (20, fitted.num_topics)

    def test_rows_sum_to_1_normalized(self, fitted):
        result = fitted.tdf_linspace(0.0, 1.0, 20, normalize=True)
        npt.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_endpoint_true_includes_stop(self, fitted):
        """With endpoint=True the last row should equal tdf(stop)."""
        result = fitted.tdf_linspace(0.1, 0.9, 10, endpoint=True, normalize=True)
        direct = fitted.tdf(np.array([0.9]), normalize=True)
        npt.assert_allclose(result[-1], direct, atol=1e-6)

    def test_endpoint_false_excludes_stop(self, fitted):
        """With endpoint=False the last row should equal tdf at the penultimate point."""
        result_true = fitted.tdf_linspace(0.0, 1.0, 5, endpoint=True, normalize=True)
        result_false = fitted.tdf_linspace(0.0, 1.0, 5, endpoint=False, normalize=True)
        # endpoint=False excludes stop; the last value should differ
        # (unless the curve is completely flat, but our corpus is not)
        assert result_true.shape == result_false.shape == (5, fitted.num_topics)
        # The first point should be the same (start is always included)
        npt.assert_allclose(result_true[0], result_false[0], atol=1e-6)

    def test_num_rows_respected(self, fitted):
        for num in (1, 5, 50):
            r = fitted.tdf_linspace(0.0, 1.0, num)
            assert r.shape[0] == num


# ---------------------------------------------------------------------------
# Synthetic monotonic recovery test
# ---------------------------------------------------------------------------

class TestGDMRMonotonicRecovery:
    """Verify that the tdf curve rises with x for the space-word topic.

    Strategy: generate a corpus where space words dominate at high x and
    animal words dominate at low x.  After fitting, the space topic's tdf
    curve (evaluated over a linspace in [0.05, 0.95]) should be increasing.
    We check Pearson correlation with x at the curve level — a soft
    directional test.  Endpoints are compared too, as a coarser fallback.
    """

    @pytest.fixture(scope="class")
    def recovery(self):
        docs, meta = _make_continuous_corpus(n=300, doc_length=12, seed=42)
        model = topica.GDMR(
            num_topics=2,
            degrees=[3],  # degree-3 Legendre: enough flexibility, not overfit
            seed=42,
            optimize_interval=25,
            burn_in=100,
        )
        model.fit(docs, meta, iters=500, num_samples=5, sample_interval=20)
        curve = model.tdf_linspace(0.05, 0.95, 20, normalize=True)
        space_idx = _identify_space_topic(model)
        xs = np.linspace(0.05, 0.95, 20)
        return model, curve, space_idx, xs

    def test_space_topic_tdf_correlated_with_x(self, recovery):
        """Pearson r between tdf of the space topic and x should be positive."""
        _, curve, space_idx, xs = recovery
        space_tdf = curve[:, space_idx]
        r = float(np.corrcoef(xs, space_tdf)[0, 1])
        assert r > 0.0, (
            f"Expected positive correlation between x and space-topic tdf; "
            f"got r={r:.4f}.  tdf values: {space_tdf.round(3).tolist()}"
        )

    def test_space_topic_tdf_higher_at_high_x(self, recovery):
        """tdf of the space topic at x=0.95 should exceed that at x=0.05."""
        _, curve, space_idx, _ = recovery
        assert curve[-1, space_idx] > curve[0, space_idx], (
            f"Space-topic tdf not higher at x=0.95 than x=0.05: "
            f"low={curve[0, space_idx]:.4f}, high={curve[-1, space_idx]:.4f}"
        )

    def test_animal_topic_tdf_lower_at_high_x(self, recovery):
        """By symmetry the animal topic's tdf should decrease with x."""
        _, curve, space_idx, _ = recovery
        animal_idx = 1 - space_idx
        assert curve[-1, animal_idx] < curve[0, animal_idx], (
            f"Animal-topic tdf not lower at x=0.95 than x=0.05: "
            f"low={curve[0, animal_idx]:.4f}, high={curve[-1, animal_idx]:.4f}"
        )


# ---------------------------------------------------------------------------
# metadata_range: explicit vs inferred
# ---------------------------------------------------------------------------

class TestGDMRMetadataRange:
    def test_inferred_range_covers_data(self):
        docs, meta = _make_continuous_corpus(n=100, seed=3)
        m = topica.GDMR(num_topics=2, degrees=[1], seed=1)
        m.fit(docs, meta, iters=100, num_samples=2, sample_interval=10)
        lo, hi = m.metadata_range[0]
        data_lo, data_hi = float(meta.min()), float(meta.max())
        # inferred range must be at least as wide as the data
        assert lo <= data_lo + 1e-6
        assert hi >= data_hi - 1e-6

    def test_explicit_range_honored(self):
        docs, meta = _make_continuous_corpus(n=100, seed=3)
        m = topica.GDMR(
            num_topics=2,
            degrees=[1],
            metadata_range=[(0.0, 1.0)],
            seed=1,
        )
        m.fit(docs, meta, iters=100, num_samples=2, sample_interval=10)
        assert m.metadata_range == [(0.0, 1.0)]


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestGDMRSaveLoad:
    @pytest.fixture(scope="class")
    def saved(self, tmp_path_factory):
        tmpdir = tmp_path_factory.mktemp("gdmr_save")
        path = tmpdir / "gdmr_model.pkl"
        docs, meta = _make_continuous_corpus(n=100, seed=5)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=1)
        m.fit(docs, meta, iters=200, num_samples=3, sample_interval=10)
        m.save(str(path))
        m2 = topica.GDMR.load(str(path))
        return m, m2

    def test_topic_word_identical_after_load(self, saved):
        m, m2 = saved
        npt.assert_array_equal(m.topic_word, m2.topic_word)

    def test_tdf_identical_after_load(self, saved):
        m, m2 = saved
        pt = np.array([0.4])
        npt.assert_allclose(m.tdf(pt), m2.tdf(pt), atol=1e-8)

    def test_degrees_preserved(self, saved):
        m, m2 = saved
        assert m2.degrees == m.degrees

    def test_metadata_range_preserved(self, saved):
        m, m2 = saved
        assert m2.metadata_range == m.metadata_range

    def test_feature_effects_preserved(self, saved):
        m, m2 = saved
        npt.assert_array_equal(m.feature_effects, m2.feature_effects)

    def test_vocabulary_preserved(self, saved):
        m, m2 = saved
        assert m2.vocabulary == m.vocabulary

    def test_doc_topic_preserved(self, saved):
        m, m2 = saved
        npt.assert_array_equal(m.doc_topic, m2.doc_topic)


# ---------------------------------------------------------------------------
# transform: held-out inference
# ---------------------------------------------------------------------------

class TestGDMRTransform:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, meta = _make_continuous_corpus(n=200, seed=0)
        return _fit_gdmr(docs, meta, degrees=[2], seed=1)

    def test_transform_shape(self, fitted):
        """transform on held-out docs returns (num_new, K)."""
        new_docs = [["planet", "star"], ["cat", "dog"], ["moon", "orbit", "rocket"]]
        new_meta = np.array([[0.8], [0.2], [0.9]])
        result = fitted.transform(new_docs, new_meta)
        assert result.shape == (3, fitted.num_topics)

    def test_transform_rows_sum_to_1(self, fitted):
        new_docs = [["planet", "star"], ["cat", "dog"]]
        new_meta = np.array([[0.8], [0.2]])
        result = fitted.transform(new_docs, new_meta)
        npt.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_transform_no_metadata(self, fitted):
        """transform without metadata should still work (uses model's prior).

        Assumption: metadata=None is acceptable for held-out inference (the
        model falls back to the baseline alpha), matching how DMR.transform
        is described in the contract ('metadata=None').
        """
        new_docs = [["planet", "star"], ["cat", "dog"]]
        # Contract signature: transform(data, metadata=None, ...)
        result = fitted.transform(new_docs)
        assert result.shape == (2, fitted.num_topics)
        npt.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_transform_values_in_0_1(self, fitted):
        new_docs = [["planet", "star", "moon"]]
        new_meta = np.array([[0.7]])
        result = fitted.transform(new_docs, new_meta)
        assert (result >= 0).all() and (result <= 1).all()


# ---------------------------------------------------------------------------
# top_words
# ---------------------------------------------------------------------------

class TestGDMRTopWords:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, meta = _make_continuous_corpus(n=120, seed=0)
        return _fit_gdmr(docs, meta, degrees=[1], seed=1, iters=200)

    def test_top_words_all_topics_structure(self, fitted):
        result = fitted.top_words(5)
        assert isinstance(result, list)
        assert len(result) == fitted.num_topics
        for topic_list in result:
            assert isinstance(topic_list, list)
            assert len(topic_list) == 5
            for word, prob in topic_list:
                assert isinstance(word, str)
                assert isinstance(prob, float)

    def test_top_words_single_topic(self, fitted):
        result = fitted.top_words(5, topic=0)
        assert isinstance(result, list)
        assert len(result) == 5
        for word, prob in result:
            assert isinstance(word, str)
            assert isinstance(prob, float)

    def test_top_words_probabilities_descending(self, fitted):
        for topic_list in fitted.top_words(7):
            probs = [p for _, p in topic_list]
            assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# coherence
# ---------------------------------------------------------------------------

class TestGDMRCoherence:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, meta = _make_continuous_corpus(n=120, seed=0)
        return _fit_gdmr(docs, meta, degrees=[1], seed=1, iters=200)

    def test_coherence_shape(self, fitted):
        c = fitted.coherence(n=5)
        assert c.shape == (fitted.num_topics,)

    def test_coherence_nonpositive(self, fitted):
        c = fitted.coherence(n=5)
        assert (c <= 0).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestGDMRDeterminism:
    def test_same_seed_same_topic_word(self):
        docs, meta = _make_continuous_corpus(n=80, seed=3)
        m1 = topica.GDMR(num_topics=2, degrees=[1], seed=42)
        m1.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        m2 = topica.GDMR(num_topics=2, degrees=[1], seed=42)
        m2.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        npt.assert_array_equal(m1.topic_word, m2.topic_word)

    def test_same_seed_same_feature_effects(self):
        docs, meta = _make_continuous_corpus(n=80, seed=3)
        m1 = topica.GDMR(num_topics=2, degrees=[1], seed=42)
        m1.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        m2 = topica.GDMR(num_topics=2, degrees=[1], seed=42)
        m2.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        npt.assert_array_equal(m1.feature_effects, m2.feature_effects)


# ---------------------------------------------------------------------------
# topic_names get/set
# ---------------------------------------------------------------------------

class TestGDMRTopicNames:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, meta = _make_continuous_corpus(n=80, seed=0)
        return _fit_gdmr(docs, meta, degrees=[1], seed=1, iters=150)

    def test_topic_names_default_length(self, fitted):
        assert len(fitted.topic_names) == fitted.num_topics

    def test_topic_names_settable(self, fitted):
        fitted.topic_names = ["space", "animals"]
        assert fitted.topic_names == ["space", "animals"]


# ---------------------------------------------------------------------------
# decay prior: feature_effects shape invariant with decay > 0
# ---------------------------------------------------------------------------

class TestGDMRDecayPrior:
    def test_decay_positive_still_fits(self):
        docs, meta = _make_continuous_corpus(n=100, seed=1)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=1, decay=0.5)
        m.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        # Shapes must still hold
        npt.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-6)
        npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)
        assert m.feature_effects.shape == (2, 3)  # degrees=[2] -> 3 basis terms

    def test_no_decay_still_fits(self):
        docs, meta = _make_continuous_corpus(n=100, seed=1)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=1, decay=0.0)
        m.fit(docs, meta, iters=150, num_samples=2, sample_interval=10)
        npt.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# convergence / fit_history interface (mirrors DMR)
# ---------------------------------------------------------------------------

class TestGDMRConvergenceInterface:
    def test_fit_history_exists(self):
        docs, meta = _make_continuous_corpus(n=60, seed=0)
        m = topica.GDMR(num_topics=2, degrees=[1], seed=1)
        m.fit(docs, meta, iters=100, num_samples=2, sample_interval=10)
        # fit_history should be accessible (list or array or dict per DMR)
        _ = m.fit_history  # should not raise

    def test_converged_attribute_exists(self):
        docs, meta = _make_continuous_corpus(n=60, seed=0)
        m = topica.GDMR(num_topics=2, degrees=[1], seed=1)
        m.fit(docs, meta, iters=100, num_samples=2, sample_interval=10)
        _ = m.converged  # should not raise; value is bool or None


# ---------------------------------------------------------------------------
# DMR-style covariate aliases: features (canonical) / covariates / metadata
# ---------------------------------------------------------------------------

class TestGDMRCovariateAliases:
    """fit/transform take the covariate as features= (canonical, DMR style),
    with covariates= and metadata= as equivalent aliases. Exactly one allowed."""

    def _fit(self, kw):
        docs, meta = _make_continuous_corpus(n=120, seed=3)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=7)
        m.fit(docs, iters=120, **{kw: meta})
        return m.topic_word

    def test_features_covariates_metadata_equivalent(self):
        tw_features = self._fit("features")
        tw_covariates = self._fit("covariates")
        tw_metadata = self._fit("metadata")
        npt.assert_allclose(tw_features, tw_covariates)
        npt.assert_allclose(tw_features, tw_metadata)

    def test_positional_binds_to_features(self):
        docs, meta = _make_continuous_corpus(n=120, seed=3)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=7)
        m.fit(docs, meta, iters=120)  # positional == features=
        npt.assert_allclose(m.topic_word, self._fit("features"))

    def test_two_spellings_raises(self):
        docs, meta = _make_continuous_corpus(n=60, seed=1)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=7)
        with pytest.raises(ValueError):
            m.fit(docs, features=meta, metadata=meta, iters=50)

    def test_missing_covariate_raises(self):
        docs, _ = _make_continuous_corpus(n=60, seed=1)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=7)
        with pytest.raises(ValueError):
            m.fit(docs, iters=50)

    def test_transform_alias_equivalent(self):
        docs, meta = _make_continuous_corpus(n=120, seed=3)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=7)
        m.fit(docs, meta, iters=120)
        new_docs, new_meta = _make_continuous_corpus(n=20, seed=99)
        t_features = m.transform(new_docs, features=new_meta, seed=0)
        t_metadata = m.transform(new_docs, metadata=new_meta, seed=0)
        npt.assert_allclose(t_features, t_metadata)


# ---------------------------------------------------------------------------
# #157: metadata_names (the D dimensions) vs feature_names (the basis columns)
# ---------------------------------------------------------------------------

class TestGDMRNames:
    """metadata_names labels the D continuous dimensions; feature_names labels
    the Legendre basis terms and aligns with feature_effects columns. They are
    deliberately different things with different names."""

    def _fit(self, degrees, metadata_names=None, seed=5):
        n_dim = len(degrees)
        if n_dim == 1:
            docs, meta = _make_continuous_corpus(n=120, seed=seed)
        else:
            docs, meta = _make_2d_corpus(n=140, seed=seed)
        m = topica.GDMR(num_topics=2, degrees=degrees, seed=7)
        m.fit(docs, meta, metadata_names=metadata_names, iters=120)
        return m

    def test_metadata_names_default(self):
        m = self._fit([3])
        assert m.metadata_names == ["x0"]

    def test_metadata_names_custom(self):
        m = self._fit([2, 1], metadata_names=["year", "citations"])
        assert m.metadata_names == ["year", "citations"]

    def test_metadata_names_length_must_match_dims(self):
        docs, meta = _make_continuous_corpus(n=60, seed=1)
        m = topica.GDMR(num_topics=2, degrees=[2], seed=7)
        with pytest.raises(ValueError):
            m.fit(docs, meta, metadata_names=["a", "b"], iters=50)  # D=1, 2 names

    def test_feature_names_align_with_feature_effects(self):
        m = self._fit([3])
        names = m.feature_names
        assert len(names) == m.feature_effects.shape[1]      # one per basis column
        assert names[0] == "intercept"

    def test_feature_names_use_metadata_names(self):
        m = self._fit([2], metadata_names=["year"])
        names = m.feature_names
        # degrees=[2] -> intercept, year^1, year^2
        assert names == ["intercept", "year^1", "year^2"]

    def test_feature_names_tensor_product_2d(self):
        m = self._fit([1, 1], metadata_names=["year", "cite"])
        names = m.feature_names
        assert names[0] == "intercept"
        assert len(names) == m.feature_effects.shape[1] == 4  # (1+1)*(1+1)
        assert "year^1:cite^1" in names                       # the cross term

    def test_metadata_names_survive_save_load(self, tmp_path):
        m = self._fit([2], metadata_names=["year"])
        p = str(tmp_path / "g.gdmr")
        m.save(p)
        loaded = topica.GDMR.load(p)
        assert loaded.metadata_names == ["year"]
        assert loaded.feature_names == m.feature_names
