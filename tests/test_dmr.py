"""Tests for the DMR class and one_hot helper."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from topica import DMR, Corpus, one_hot


# ---------------------------------------------------------------------------
# Synthetic corpus builder
# ---------------------------------------------------------------------------

# Two disjoint vocabularies: A = space words, B = animal words
_VOCAB_A = ["planet", "star", "moon", "rocket", "orbit"]
_VOCAB_B = ["cat", "dog", "fish", "bird", "mouse"]


def _make_corpus(n=120, doc_length=8, seed=0):
    """Return (docs, covariate, features) with a binary covariate driving topics.

    When is_A=1, the document draws words from VOCAB_A; when is_A=0, from VOCAB_B.
    """
    rng = np.random.default_rng(seed)
    docs = []
    covariate = []
    for _ in range(n):
        is_A = int(rng.integers(0, 2))
        covariate.append(is_A)
        vocab = _VOCAB_A if is_A else _VOCAB_B
        doc = rng.choice(vocab, size=doc_length).tolist()
        docs.append(doc)
    features = np.array([[float(x)] for x in covariate])
    return docs, covariate, features


def _fit_dmr(docs, features, seed=1, iters=300, feature_names=None, **kwargs):
    """Return a fitted DMR with fast but reliable settings for the synthetic corpus."""
    model = DMR(
        num_topics=2,
        seed=seed,
        optimize_interval=25,
        burn_in=50,
        **kwargs,
    )
    model.fit(
        docs,
        features,
        feature_names=feature_names if feature_names is not None else ["is_A"],
        iters=iters,
        num_samples=3,
        sample_interval=10,
    )
    return model


def _identify_A_topic(model):
    """Return the index of the topic whose top words are space words."""
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

class TestDMRConstructor:
    def test_num_topics_available_before_fit(self):
        model = DMR(3)
        assert model.num_topics == 3

    def test_num_topics_zero_raises(self):
        with pytest.raises(ValueError):
            DMR(0)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            DMR(2, beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises(ValueError):
            DMR(2, beta=-0.01)

    def test_prior_variance_zero_raises(self):
        with pytest.raises(ValueError):
            DMR(2, prior_variance=0.0)

    def test_prior_variance_negative_raises(self):
        with pytest.raises(ValueError):
            DMR(2, prior_variance=-1.0)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            DMR(2, alpha=0.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError):
            DMR(2, alpha=-0.1)

    def test_alpha_epsilon_negative_raises(self):
        with pytest.raises(ValueError):
            DMR(2, alpha_epsilon=-1e-9)

    def test_default_alpha_and_epsilon(self):
        d = DMR(2)
        assert d.settings["alpha"] == 0.1
        assert d.settings["alpha_epsilon"] == 1e-10


class TestCovariatePriorFaithfulness:
    """#563: the covariate prior must not over-steer. For a *null* covariate,
    DMR must reduce to LDA at the same baseline concentration (the λ optimizer
    used to diverge to |λ|~100, collapsing θ toward uniform)."""

    def _null_cov_corpus(self, n=300, doc_length=12, seed=0):
        rng = np.random.default_rng(seed)
        docs = []
        for i in range(n):
            vocab = _VOCAB_A if i % 2 == 0 else _VOCAB_B
            docs.append(rng.choice(vocab, size=doc_length).tolist())
        # A covariate that carries NO signal about the text.
        x = rng.normal(size=(n, 1))
        return docs, x

    def test_lambda_stays_bounded_under_optimization(self):
        docs, x = self._null_cov_corpus(seed=1)
        m = DMR(num_topics=2, seed=1)
        m.fit(docs, x, feature_names=["null"], iters=600)
        fe = np.asarray(m.feature_effects)
        # Pre-fix this ran to |lambda| ~ 40-100; a faithful fit keeps it O(1).
        assert np.abs(fe).max() < 10.0, f"lambda diverged: max|λ|={np.abs(fe).max()}"

    def test_dmr_reduces_to_lda_for_null_covariate(self):
        from topica import LDA

        docs, x = self._null_cov_corpus(seed=2)
        d = DMR(num_topics=2, seed=1)
        d.fit(docs, x, feature_names=["null"], iters=600)
        lda = LDA(num_topics=2, seed=1)
        lda.fit(docs, iters=600)

        # Align topic-word rows across the shared vocabulary and require a high
        # aligned cosine (DMR with a null covariate ≈ LDA).
        dv = {w: i for i, w in enumerate(lda.vocabulary)}
        shared = [w for w in d.vocabulary if w in dv]
        Pd = np.asarray(d.topic_word)[:, [i for i, w in enumerate(d.vocabulary) if w in dv]]
        Pl = np.asarray(lda.topic_word)[:, [dv[w] for w in shared]]
        Pd = Pd / np.linalg.norm(Pd, axis=1, keepdims=True)
        Pl = Pl / np.linalg.norm(Pl, axis=1, keepdims=True)
        best = max(
            np.mean([Pd[i] @ Pl[p[i]] for i in range(2)])
            for p in [(0, 1), (1, 0)]
        )
        assert best > 0.7, f"DMR diverges from LDA for a null covariate: cos={best:.3f}"


# ---------------------------------------------------------------------------
# Unfitted access raises RuntimeError
# ---------------------------------------------------------------------------

class TestDMRUnfittedGuards:
    PROPERTIES = [
        "topic_word",
        "doc_topic",
        "feature_effects",
        "feature_names",
        "vocabulary",
        "doc_names",
    ]

    @pytest.mark.parametrize("prop", PROPERTIES)
    def test_property_raises_before_fit(self, prop):
        model = DMR(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(model, prop)

    def test_coherence_raises_before_fit(self):
        model = DMR(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.coherence()

    def test_top_words_raises_before_fit(self):
        model = DMR(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.top_words()


# ---------------------------------------------------------------------------
# Input validation errors during fit()
# ---------------------------------------------------------------------------

class TestDMRFitValidation:
    def setup_method(self):
        self.docs = [["cat", "dog"]] * 10
        self.features = np.ones((10, 1))

    def test_features_row_count_mismatch_raises(self):
        bad_features = np.ones((8, 1))  # 8 rows, 10 docs
        with pytest.raises(ValueError):
            model = DMR(2)
            model.fit(self.docs, bad_features)

    def test_ragged_feature_rows_raises(self):
        ragged = [[1.0, 2.0], [1.0]]  # unequal widths
        with pytest.raises(ValueError):
            model = DMR(2)
            model.fit([["cat", "dog"]] * 2, ragged)

    def test_feature_names_wrong_length_raises(self):
        # F=1 column but feature_names has 2 entries
        with pytest.raises(ValueError):
            model = DMR(2)
            model.fit(self.docs, self.features, feature_names=["a", "b"])

    def test_feature_names_correct_length_accepted(self):
        # Should not raise
        model = DMR(2, seed=1)
        model.fit(
            self.docs,
            self.features,
            feature_names=["my_feature"],
            iters=50,
            num_samples=1,
            sample_interval=5,
        )
        assert model.feature_names == ["intercept", "my_feature"]


# ---------------------------------------------------------------------------
# Shapes and basic properties after fit
# ---------------------------------------------------------------------------

class TestDMRShapesAndProperties:
    @pytest.fixture(scope="class")
    def fitted_model(self):
        docs, covariate, features = _make_corpus(n=120, seed=0)
        return _fit_dmr(docs, features, seed=1)

    def test_topic_word_shape(self, fitted_model):
        # 2 topics, 10 unique words (5 space + 5 animal)
        assert fitted_model.topic_word.shape == (2, 10)

    def test_doc_topic_shape(self, fitted_model):
        assert fitted_model.doc_topic.shape == (120, 2)

    def test_doc_topic_rows_sum_to_one(self, fitted_model):
        npt.assert_allclose(
            fitted_model.doc_topic.sum(axis=1), np.ones(120), atol=1e-6
        )

    def test_feature_effects_shape(self, fitted_model):
        # (num_topics, num_features) = (2, 2) [intercept + is_A]
        assert fitted_model.feature_effects.shape == (2, 2)

    def test_feature_effect_se_shape_and_finite(self, fitted_model):
        se = fitted_model.feature_effect_se
        assert se is not None
        assert se.shape == fitted_model.feature_effects.shape == (2, 2)
        assert np.all(np.isfinite(se)) and np.all(se > 0.0)

    def test_feature_names_intercept_first(self, fitted_model):
        assert fitted_model.feature_names[0] == "intercept"
        assert fitted_model.feature_names == ["intercept", "is_A"]

    def test_vocabulary_length(self, fitted_model):
        assert len(fitted_model.vocabulary) == fitted_model.topic_word.shape[1]

    def test_doc_names_length(self, fitted_model):
        assert len(fitted_model.doc_names) == fitted_model.doc_topic.shape[0]

    def test_num_topics(self, fitted_model):
        assert fitted_model.num_topics == 2


# ---------------------------------------------------------------------------
# Covariate recovery: DMR must recover the known covariate -> topic relationship
# ---------------------------------------------------------------------------

class TestDMRCovariateRecovery:
    @pytest.fixture(scope="class")
    def recovery_model(self):
        docs, covariate, features = _make_corpus(n=120, seed=0)
        return _fit_dmr(docs, features, seed=1)

    def test_topics_separate_into_two_vocabularies(self, recovery_model):
        """Top words of each topic should come from only one vocabulary."""
        top = recovery_model.top_words(5, weights=True)
        vocab_A_set = set(_VOCAB_A)
        vocab_B_set = set(_VOCAB_B)
        for topic_words in top:
            words = {w for w, _ in topic_words}
            in_A = words & vocab_A_set
            in_B = words & vocab_B_set
            # Each topic should be dominated by one vocabulary
            assert len(in_A) == 0 or len(in_B) == 0, (
                f"Topic mixed vocabularies: A-words={in_A}, B-words={in_B}"
            )

    def test_covariate_effect_sign_and_magnitude(self, recovery_model):
        """The is_A coefficient for the A-topic should be much higher than for the B-topic.

        Expected effect size ~ +6 (well above threshold of 0.5).
        """
        A_topic = _identify_A_topic(recovery_model)
        B_topic = 1 - A_topic
        effect_diff = (
            recovery_model.feature_effects[A_topic, 1]
            - recovery_model.feature_effects[B_topic, 1]
        )
        assert effect_diff > 0.5, (
            f"Covariate recovery failed: effect_diff={effect_diff:.4f} "
            f"(A_topic={A_topic}, feature_effects=\n{recovery_model.feature_effects})"
        )

    def test_strong_effect_is_significant(self, recovery_model):
        """The planted is_A effect is large, so it should sit many SEs from zero."""
        A_topic = _identify_A_topic(recovery_model)
        eff = recovery_model.feature_effects[A_topic, 1]
        se = recovery_model.feature_effect_se[A_topic, 1]
        assert abs(eff) / se > 2.0, f"z={eff / se:.2f} for a planted effect"

    def test_a_topic_has_positive_is_A_effect(self, recovery_model):
        """The A-topic (space words) should have a positive is_A feature effect."""
        A_topic = _identify_A_topic(recovery_model)
        assert recovery_model.feature_effects[A_topic, 1] > 0

    def test_b_topic_has_negative_is_A_effect(self, recovery_model):
        """The B-topic (animal words) should have a negative is_A feature effect."""
        B_topic = 1 - _identify_A_topic(recovery_model)
        assert recovery_model.feature_effects[B_topic, 1] < 0


# ---------------------------------------------------------------------------
# Determinism: same seed -> identical outputs
# ---------------------------------------------------------------------------

class TestDMRDeterminism:
    def test_identical_seed_gives_identical_feature_effects(self):
        docs, _, features = _make_corpus(n=40, seed=5)
        m1 = DMR(num_topics=2, seed=42)
        m1.fit(docs, features, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)
        m2 = DMR(num_topics=2, seed=42)
        m2.fit(docs, features, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)
        assert np.array_equal(m1.feature_effects, m2.feature_effects)

    def test_identical_seed_gives_identical_topic_word(self):
        docs, _, features = _make_corpus(n=40, seed=5)
        m1 = DMR(num_topics=2, seed=42)
        m1.fit(docs, features, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)
        m2 = DMR(num_topics=2, seed=42)
        m2.fit(docs, features, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)
        assert np.array_equal(m1.topic_word, m2.topic_word)


# ---------------------------------------------------------------------------
# Input type parity: numpy array vs list-of-lists features
# ---------------------------------------------------------------------------

class TestDMRInputTypeParity:
    def test_numpy_vs_list_features_give_identical_results(self):
        docs, covariate, features_np = _make_corpus(n=30, seed=3)
        features_list = features_np.tolist()

        m1 = DMR(num_topics=2, seed=7)
        m1.fit(docs, features_np, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)

        m2 = DMR(num_topics=2, seed=7)
        m2.fit(docs, features_list, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)

        assert np.array_equal(m1.feature_effects, m2.feature_effects)
        assert np.array_equal(m1.topic_word, m2.topic_word)

    def test_corpus_vs_token_list_give_identical_results(self):
        docs, _, features = _make_corpus(n=30, seed=3)
        corpus = Corpus.from_documents(docs)

        m1 = DMR(num_topics=2, seed=7)
        m1.fit(docs, features, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)

        m2 = DMR(num_topics=2, seed=7)
        m2.fit(corpus, features, feature_names=["is_A"], iters=100, num_samples=2, sample_interval=5)

        assert np.array_equal(m1.feature_effects, m2.feature_effects)
        assert np.array_equal(m1.topic_word, m2.topic_word)


# ---------------------------------------------------------------------------
# top_words and coherence
# ---------------------------------------------------------------------------

class TestDMRTopWordsAndCoherence:
    @pytest.fixture(scope="class")
    def fitted_model(self):
        docs, _, features = _make_corpus(n=60, seed=0)
        model = DMR(num_topics=2, seed=1)
        model.fit(docs, features, feature_names=["is_A"], iters=200, num_samples=2, sample_interval=10)
        return model

    def test_top_words_all_topics_structure(self, fitted_model):
        result = fitted_model.top_words(5, weights=True)
        assert isinstance(result, list)
        assert len(result) == 2
        for topic_list in result:
            assert isinstance(topic_list, list)
            assert len(topic_list) == 5
            for word, prob in topic_list:
                assert isinstance(word, str)
                assert isinstance(prob, float)

    def test_top_words_single_topic_structure(self, fitted_model):
        result = fitted_model.top_words(5, topic=0, weights=True)
        assert isinstance(result, list)
        assert len(result) == 5
        for word, prob in result:
            assert isinstance(word, str)
            assert isinstance(prob, float)

    def test_top_words_probabilities_descending(self, fitted_model):
        for topic_list in fitted_model.top_words(7, weights=True):
            probs = [p for _, p in topic_list]
            assert probs == sorted(probs, reverse=True)

    def test_coherence_shape(self, fitted_model):
        c = fitted_model.coherence(n=5)
        assert c.shape == (2,)

    def test_coherence_values_nonpositive(self, fitted_model):
        c = fitted_model.coherence(n=5)
        assert (c <= 0).all()


# ---------------------------------------------------------------------------
# one_hot helper
# ---------------------------------------------------------------------------

class TestOneHot:
    def test_drop_first_true_shape(self):
        values = ["cat", "dog", "cat", "bird", "dog"]
        matrix, names = one_hot(values, drop_first=True)
        # sorted categories: bird, cat, dog -> drop bird -> 2 columns
        assert matrix.shape == (5, 2)

    def test_reference_chooses_baseline(self):
        # issue #733: pick the reference level explicitly instead of the silent
        # alphabetical default.
        values = ["cat", "dog", "bird"]
        matrix, names = one_hot(values, reference="dog")
        assert "dog" not in names and set(names) == {"cat", "bird"}
        # the dog row is all zeros (it is the baseline)
        assert matrix[1].sum() == 0.0

    def test_reference_unknown_raises(self):
        with pytest.raises(ValueError, match="not one of the categories"):
            one_hot(["a", "b"], reference="z")

    def test_multilevel_default_warns_binary_does_not(self):
        # 3+ levels with a silent alphabetical baseline is a Tier-1 footgun -> warn;
        # a binary covariate has an unambiguous baseline -> stay quiet.
        with pytest.warns(UserWarning, match="alphabetically-first"):
            one_hot(["cat", "dog", "bird"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # binary: must NOT warn
            one_hot(["cat", "dog"])
        # passing reference= silences the multilevel warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            one_hot(["cat", "dog", "bird"], reference="bird")

    def test_drop_first_false_shape(self):
        values = ["cat", "dog", "cat", "bird", "dog"]
        matrix, names = one_hot(values, drop_first=False)
        # sorted categories: bird, cat, dog -> 3 columns
        assert matrix.shape == (5, 3)

    def test_drop_first_drops_first_sorted_category(self):
        values = ["cat", "dog", "cat", "bird", "dog"]
        matrix, names = one_hot(values, drop_first=True)
        # "bird" is alphabetically first; it should be dropped
        assert "bird" not in names
        assert "cat" in names
        assert "dog" in names

    def test_drop_first_false_keeps_all_categories(self):
        values = ["cat", "dog", "cat", "bird", "dog"]
        matrix, names = one_hot(values, drop_first=False)
        assert "bird" in names
        assert "cat" in names
        assert "dog" in names

    def test_names_sorted_drop_first(self):
        values = ["z", "a", "m", "z", "a"]
        _, names = one_hot(values, drop_first=True)
        # drop "a" (first sorted); remaining: m, z
        assert names == ["m", "z"]

    def test_names_sorted_drop_first_false(self):
        values = ["z", "a", "m"]
        _, names = one_hot(values, drop_first=False)
        assert names == ["a", "m", "z"]

    def test_prefix_applied_to_names(self):
        values = ["cat", "dog"]
        _, names = one_hot(values, drop_first=False, prefix="animal_")
        assert all(n.startswith("animal_") for n in names)

    def test_prefix_with_drop_first(self):
        values = ["cat", "dog", "bird"]
        _, names = one_hot(values, drop_first=True, prefix="x_")
        assert all(n.startswith("x_") for n in names)

    def test_drop_first_true_reference_row_is_all_zeros(self):
        """The dropped reference category's rows should be all zeros."""
        values = ["cat", "dog", "cat", "bird", "dog"]
        matrix, names = one_hot(values, drop_first=True)
        # "bird" is the reference (sorted first); index 3 is "bird"
        bird_idx = 3  # 4th element = "bird"
        npt.assert_array_equal(matrix[bird_idx], np.zeros(matrix.shape[1]))

    def test_drop_first_false_rows_sum_to_one(self):
        values = ["cat", "dog", "cat", "bird", "dog"]
        matrix, names = one_hot(values, drop_first=False)
        npt.assert_array_equal(matrix.sum(axis=1), np.ones(5))

    def test_drop_first_true_retained_rows_sum_to_one(self):
        values = ["cat", "dog", "cat", "bird", "dog"]
        matrix, names = one_hot(values, drop_first=True)
        # All non-reference rows should have exactly one 1
        for i, v in enumerate(values):
            if v != "bird":  # "bird" is reference
                assert matrix[i].sum() == 1.0

    def test_matrix_values_are_zero_or_one(self):
        values = ["a", "b", "c", "a", "b"]
        matrix, _ = one_hot(values, drop_first=False)
        assert set(matrix.flatten().tolist()).issubset({0.0, 1.0})

    def test_dmr_accepts_one_hot_features(self):
        """DMR.fit should accept the matrix returned by one_hot without error."""
        rng = np.random.default_rng(42)
        groups = rng.choice(["A", "B"], size=30).tolist()
        docs = []
        for g in groups:
            vocab = _VOCAB_A if g == "A" else _VOCAB_B
            doc = rng.choice(vocab, size=6).tolist()
            docs.append(doc)

        matrix, names = one_hot(groups, drop_first=True)
        model = DMR(num_topics=2, seed=1)
        model.fit(
            docs,
            matrix,
            feature_names=names,
            iters=100,
            num_samples=2,
            sample_interval=5,
        )
        # intercept is prepended, so feature_names should start with "intercept"
        assert model.feature_names[0] == "intercept"
        assert model.feature_effects.shape[1] == len(names) + 1


# ---------------------------------------------------------------------------
# WarpLDA sampler (cache-efficient large-K backend, per-document-α doc phase)
# ---------------------------------------------------------------------------

class TestDmrWarp:
    def test_recovers_covariate_topics(self):
        docs, _, feats = _make_corpus(seed=2)
        m = _fit_dmr(docs, feats, sampler="warp")
        a = _identify_A_topic(m)
        # The A-topic's top words are the space vocab; the other topic is animals.
        top_a = {w for w, _ in m.top_words(5, topic=a, weights=True)}
        assert top_a == set(_VOCAB_A)

    def test_valid_distributions(self):
        docs, _, feats = _make_corpus(seed=2)
        m = _fit_dmr(docs, feats, sampler="warp")
        npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
        npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)

    def test_deterministic(self):
        docs, _, feats = _make_corpus(seed=2)
        a = _fit_dmr(docs, feats, seed=4, sampler="warp")
        b = _fit_dmr(docs, feats, seed=4, sampler="warp")
        npt.assert_array_equal(a.topic_word, b.topic_word)
        npt.assert_array_equal(a.feature_effects, b.feature_effects)

    def test_feature_effect_sign_matches_sparse(self):
        # is_A should raise the A-topic's prevalence under both samplers.
        docs, _, feats = _make_corpus(seed=2)
        sparse = _fit_dmr(docs, feats, sampler="sparse")
        warp = _fit_dmr(docs, feats, sampler="warp")
        a_s = _identify_A_topic(sparse)
        a_w = _identify_A_topic(warp)
        # column 1 is is_A (column 0 is the prepended intercept).
        assert sparse.feature_effects[a_s, 1] > 0
        assert warp.feature_effects[a_w, 1] > 0

    def test_bad_sampler_rejected(self):
        with pytest.raises(ValueError):
            DMR(num_topics=2, sampler="banana")

    def test_sampler_aliases(self):
        docs, _, feats = _make_corpus(seed=2)
        for name in ("warp", "warplda"):
            m = DMR(num_topics=2, seed=1, sampler=name)
            m.fit(docs, feats, iters=50)
            assert m.topic_word.shape[0] == 2


class TestDmrCvb0:
    def test_recovers_and_valid(self):
        docs, _, feats = _make_corpus(seed=2)
        m = _fit_dmr(docs, feats, sampler="cvb0")
        a = _identify_A_topic(m)
        assert {w for w, _ in m.top_words(5, topic=a, weights=True)} == set(_VOCAB_A)
        npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
        npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)

    def test_deterministic_and_no_draws(self):
        docs, _, feats = _make_corpus(seed=2)
        a = _fit_dmr(docs, feats, seed=4, sampler="cvb0")
        b = _fit_dmr(docs, feats, seed=4, sampler="cvb0")
        npt.assert_array_equal(a.topic_word, b.topic_word)
        assert a.theta_draws is None

    def test_feature_effect_sign_matches_sparse(self):
        docs, _, feats = _make_corpus(seed=2)
        sparse = _fit_dmr(docs, feats, sampler="sparse")
        cvb = _fit_dmr(docs, feats, sampler="cvb0")
        assert sparse.feature_effects[_identify_A_topic(sparse), 1] > 0
        assert cvb.feature_effects[_identify_A_topic(cvb), 1] > 0
