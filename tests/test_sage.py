"""Tests for the SAGE class (content-covariate topic model).

SAGE learns shared topics whose word distributions vary by a document-level
group covariate: log β_{k,g,v} = m_v + κT_{k,v} + κC_{g,v} + κI_{k,g,v}.
"""

import numpy as np
import numpy.testing as npt
import pytest

from topica import SAGE, Corpus


# ---------------------------------------------------------------------------
# Synthetic bilingual corpus — the core validation fixture
# ---------------------------------------------------------------------------
# Two topics (weather / food) × two language groups (en / de).
# Each group uses a DIFFERENT vocabulary for the same topics, with no shared
# words, giving fully disjoint top-word sets — the key SAGE property.
#
# Corpus design: each doc has one dominant topic (10 words) and the other
# topic as background (2 words), both drawn from the doc's language group.
# This gives SAGE a clear 2-topic × 2-group structure to recover.

_EN_WEATHER = ["rain", "sun", "cloud", "wind", "storm"]
_DE_WEATHER = ["regen", "sonne", "wolke", "sturm", "nebel"]   # fully disjoint from EN
_EN_FOOD    = ["bread", "cheese", "wine", "apple", "meat"]
_DE_FOOD    = ["brot",  "kaese",  "wein", "apfel", "fleisch"]

_EN_VOCAB = set(_EN_WEATHER) | set(_EN_FOOD)
_DE_VOCAB = set(_DE_WEATHER) | set(_DE_FOOD)


def _make_bilingual_corpus(n_per_cell=50, seed=42):
    """Return (docs, groups) with a known 2-topic × 2-language structure.

    Creates n_per_cell weather-dominant and n_per_cell food-dominant docs
    for each language group, giving 4 * n_per_cell documents total.
    """
    rng = np.random.default_rng(seed)
    docs   = []
    groups = []

    # English weather-heavy docs (10 weather + 2 food)
    for _ in range(n_per_cell):
        docs.append(
            rng.choice(_EN_WEATHER, size=10).tolist()
            + rng.choice(_EN_FOOD, size=2).tolist()
        )
        groups.append("en")

    # English food-heavy docs (10 food + 2 weather)
    for _ in range(n_per_cell):
        docs.append(
            rng.choice(_EN_FOOD, size=10).tolist()
            + rng.choice(_EN_WEATHER, size=2).tolist()
        )
        groups.append("en")

    # German weather-heavy docs (10 weather + 2 food)
    for _ in range(n_per_cell):
        docs.append(
            rng.choice(_DE_WEATHER, size=10).tolist()
            + rng.choice(_DE_FOOD, size=2).tolist()
        )
        groups.append("de")

    # German food-heavy docs (10 food + 2 weather)
    for _ in range(n_per_cell):
        docs.append(
            rng.choice(_DE_FOOD, size=10).tolist()
            + rng.choice(_DE_WEATHER, size=2).tolist()
        )
        groups.append("de")

    return docs, groups


def _fit_bilingual(docs, groups, seed=1):
    """Fit SAGE with the settings used throughout the bilingual tests."""
    model = SAGE(num_topics=2, seed=seed, optimize_interval=25, burn_in=50)
    model.fit(
        docs,
        groups,
        iters=300,
        num_samples=3,
        sample_interval=10,
    )
    return model


@pytest.fixture(scope="module")
def bilingual_model():
    """Module-scoped fixture: one fit of the bilingual corpus, reused by all tests."""
    docs, groups = _make_bilingual_corpus(n_per_cell=50)
    return _fit_bilingual(docs, groups)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestSAGEConstructor:
    def test_num_topics_available_before_fit(self):
        assert SAGE(3).num_topics == 3

    def test_num_topics_zero_raises(self):
        with pytest.raises(ValueError):
            SAGE(0)

    def test_num_topics_negative_raises(self):
        # Negative values can raise either ValueError or OverflowError
        # depending on how the Rust layer handles unsigned conversion
        with pytest.raises((ValueError, OverflowError)):
            SAGE(-1)

    def test_prior_variance_zero_raises(self):
        with pytest.raises(ValueError):
            SAGE(2, prior_variance=0.0)

    def test_prior_variance_negative_raises(self):
        with pytest.raises(ValueError):
            SAGE(2, prior_variance=-1.0)

    def test_valid_defaults_do_not_raise(self):
        # Should construct without error
        SAGE(num_topics=2, alpha=0.1, prior_variance=1.0,
             optimize_interval=50, burn_in=100, seed=42, lbfgs_iters=20)


# ---------------------------------------------------------------------------
# Unfitted guards: all properties/methods must raise RuntimeError before fit()
# ---------------------------------------------------------------------------

class TestSAGEUnfittedGuards:
    PROPERTIES = [
        "topic_word",
        "topic_word_marginal",
        "doc_topic",
        "groups",
        "vocabulary",
        "doc_names",
        "num_groups",
    ]

    @pytest.mark.parametrize("prop", PROPERTIES)
    def test_property_raises_before_fit(self, prop):
        model = SAGE(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(model, prop)

    def test_top_words_raises_before_fit(self):
        model = SAGE(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.top_words(5)

    def test_word_contrast_raises_before_fit(self):
        model = SAGE(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.word_contrast(0, "a", "b")

    def test_coherence_raises_before_fit(self):
        model = SAGE(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.coherence()


# ---------------------------------------------------------------------------
# fit() input validation
# ---------------------------------------------------------------------------

class TestSAGEFitValidation:
    def setup_method(self):
        self.docs   = [["cat", "dog"]] * 10
        self.groups = ["a"] * 10

    def test_groups_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            model = SAGE(2)
            model.fit(self.docs, self.groups[:-1])   # 9 groups, 10 docs

    def test_group_label_not_in_group_names_raises(self):
        with pytest.raises(ValueError):
            model = SAGE(2)
            # group "a" is in data but not in group_names
            model.fit(self.docs, self.groups, group_names=["b"])

    def test_top_words_topic_out_of_range_raises(self):
        model = SAGE(2, seed=1)
        model.fit(self.docs, self.groups, iters=50, num_samples=1, sample_interval=5)
        with pytest.raises(ValueError):
            model.top_words(5, topic=99)

    def test_top_words_unknown_group_name_raises(self):
        model = SAGE(2, seed=1)
        model.fit(self.docs, self.groups, iters=50, num_samples=1, sample_interval=5)
        with pytest.raises(ValueError):
            model.top_words(5, topic=0, group="z")

    def test_top_words_group_index_out_of_range_raises(self):
        model = SAGE(2, seed=1)
        model.fit(self.docs, self.groups, iters=50, num_samples=1, sample_interval=5)
        with pytest.raises(ValueError):
            model.top_words(5, topic=0, group=99)

    def test_word_contrast_topic_out_of_range_raises(self):
        model = SAGE(2, seed=1)
        model.fit(self.docs + [["bird"]], self.groups + ["b"],
                  iters=50, num_samples=1, sample_interval=5)
        with pytest.raises(ValueError):
            model.word_contrast(99, "a", "b")


# ---------------------------------------------------------------------------
# Shapes and basic properties after fit
# ---------------------------------------------------------------------------

class TestSAGEShapesAndProperties:
    def test_topic_word_shape(self, bilingual_model):
        m = bilingual_model
        assert m.topic_word.shape == (2, 2, len(m.vocabulary))

    def test_topic_word_marginal_shape(self, bilingual_model):
        m = bilingual_model
        assert m.topic_word_marginal.shape == (2, len(m.vocabulary))

    def test_doc_topic_shape(self, bilingual_model):
        m = bilingual_model
        assert m.doc_topic.shape == (200, 2)

    def test_doc_topic_rows_sum_to_one(self, bilingual_model):
        npt.assert_allclose(
            bilingual_model.doc_topic.sum(axis=1),
            np.ones(200),
            atol=1e-6,
        )

    def test_num_groups_equals_2(self, bilingual_model):
        assert bilingual_model.num_groups == 2

    def test_groups_are_sorted(self, bilingual_model):
        assert bilingual_model.groups == sorted(["de", "en"])

    def test_vocabulary_length_matches_topic_word(self, bilingual_model):
        m = bilingual_model
        assert len(m.vocabulary) == m.topic_word.shape[2]

    def test_doc_names_length_matches_doc_topic(self, bilingual_model):
        m = bilingual_model
        assert len(m.doc_names) == m.doc_topic.shape[0]

    def test_num_topics(self, bilingual_model):
        assert bilingual_model.num_topics == 2

    def test_topic_word_nonnegative(self, bilingual_model):
        assert (bilingual_model.topic_word >= 0).all()

    def test_topic_word_rows_sum_to_one(self, bilingual_model):
        """Each (topic, group) distribution must sum to 1."""
        tw = bilingual_model.topic_word
        npt.assert_allclose(tw.sum(axis=2), np.ones((2, 2)), atol=1e-5)


# ---------------------------------------------------------------------------
# Group-specific wording: the core SAGE property
# ---------------------------------------------------------------------------

class TestSAGEGroupSpecificWording:
    """Verify that each topic uses different vocabulary per group.

    Topic 0 words in 'en' should be English; in 'de' should be German.
    The sets should be (nearly) disjoint and each should be a subset of
    the corresponding language's full vocabulary.
    """

    def test_en_top_words_are_english(self, bilingual_model):
        """Top words for each topic in the 'en' group are English words."""
        m = bilingual_model
        for t in range(2):
            top = {w for w, _ in m.top_words(7, topic=t, group="en")}
            assert top <= _EN_VOCAB, (
                f"Topic {t} en top-words contain non-English words: "
                f"{top - _EN_VOCAB}"
            )

    def test_de_top_words_are_german(self, bilingual_model):
        """Top words for each topic in the 'de' group are German words."""
        m = bilingual_model
        for t in range(2):
            top = {w for w, _ in m.top_words(7, topic=t, group="de")}
            # German vocab + shared word "wind"
            assert top <= _DE_VOCAB, (
                f"Topic {t} de top-words contain non-German words: "
                f"{top - _DE_VOCAB}"
            )

    def test_en_and_de_top_words_nearly_disjoint(self, bilingual_model):
        """The top-word sets for 'en' and 'de' should share at most 1 word."""
        m = bilingual_model
        for t in range(2):
            en_words = {w for w, _ in m.top_words(7, topic=t, group="en")}
            de_words = {w for w, _ in m.top_words(7, topic=t, group="de")}
            shared = en_words & de_words
            assert len(shared) <= 1, (
                f"Topic {t}: en and de top-word sets share too many words: "
                f"{shared}"
            )

    def test_word_contrast_en_vs_de_surfaces_english_words(self, bilingual_model):
        """word_contrast(t, 'en', 'de') should surface English words (positive log-ratio)."""
        m = bilingual_model
        for t in range(2):
            contrast = m.word_contrast(t, "en", "de", 5)
            top_words = {w for w, _ in contrast}
            # All contrasting words should be in the English vocabulary
            assert top_words <= _EN_VOCAB, (
                f"Topic {t} en-vs-de contrast has non-English words: "
                f"{top_words - _EN_VOCAB}"
            )

    def test_word_contrast_de_vs_en_surfaces_german_words(self, bilingual_model):
        """word_contrast(t, 'de', 'en') should surface German words (positive log-ratio)."""
        m = bilingual_model
        for t in range(2):
            contrast = m.word_contrast(t, "de", "en", 5)
            top_words = {w for w, _ in contrast}
            assert top_words <= _DE_VOCAB, (
                f"Topic {t} de-vs-en contrast has non-German words: "
                f"{top_words - _DE_VOCAB}"
            )

    def test_word_contrast_signs_flip(self, bilingual_model):
        """Reversing group_a / group_b should flip the log-ratio signs."""
        m = bilingual_model
        t = 0
        ab = dict(m.word_contrast(t, "en", "de", 10))
        ba = dict(m.word_contrast(t, "de", "en", 10))
        # Words that appear in both contrasts should have flipped signs
        common = set(ab) & set(ba)
        if common:
            for w in common:
                assert np.sign(ab[w]) != np.sign(ba[w]), (
                    f"Word '{w}' has same sign in en-vs-de and de-vs-en contrast"
                )

    def test_word_contrast_log_ratios_are_finite(self, bilingual_model):
        m = bilingual_model
        for t in range(2):
            for lr in [m.word_contrast(t, "en", "de"), m.word_contrast(t, "de", "en")]:
                assert all(np.isfinite(v) for _, v in lr)


# ---------------------------------------------------------------------------
# top_words: group as name vs index gives identical results
# ---------------------------------------------------------------------------

class TestSAGETopWordsByNameVsIndex:
    def test_group_name_and_index_give_same_result(self, bilingual_model):
        m = bilingual_model
        # groups == ['de', 'en'], so index 0 = 'de', index 1 = 'en'
        de_idx = m.groups.index("de")
        en_idx = m.groups.index("en")

        for t in range(2):
            tw_de_name = m.top_words(5, topic=t, group="de")
            tw_de_idx  = m.top_words(5, topic=t, group=de_idx)
            assert tw_de_name == tw_de_idx, (
                f"top_words(topic={t}, group='de') != top_words(topic={t}, group={de_idx})"
            )

            tw_en_name = m.top_words(5, topic=t, group="en")
            tw_en_idx  = m.top_words(5, topic=t, group=en_idx)
            assert tw_en_name == tw_en_idx

    def test_group_none_returns_marginal(self, bilingual_model):
        """top_words with group=None should return the group-averaged distribution."""
        m = bilingual_model
        for t in range(2):
            tw_none = m.top_words(5, topic=t, group=None)
            assert isinstance(tw_none, list)
            assert len(tw_none) == 5
            for w, p in tw_none:
                assert isinstance(w, str)
                assert isinstance(p, float)
                assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Integer group labels
# ---------------------------------------------------------------------------

class TestSAGEIntegerGroups:
    def test_int_groups_are_stringified(self):
        """When group labels are ints, groups property should be ['0', '1']."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = [0] * 10 + [1] * 10
        model = SAGE(num_topics=2, seed=1)
        model.fit(docs, groups, iters=50, num_samples=1, sample_interval=5)
        assert model.groups == ["0", "1"]

    def test_int_groups_fit_completes(self):
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = [0] * 10 + [1] * 10
        model = SAGE(num_topics=2, seed=1)
        model.fit(docs, groups, iters=50, num_samples=1, sample_interval=5)
        assert model.doc_topic.shape == (20, 2)


# ---------------------------------------------------------------------------
# group_names fixes group order
# ---------------------------------------------------------------------------

class TestSAGEGroupNames:
    def test_group_names_fixes_order_en_de(self):
        """group_names=['en','de'] → groups == ['en','de'] (index 0 = en)."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = ["en"] * 10 + ["de"] * 10
        model = SAGE(num_topics=2, seed=1)
        model.fit(docs, groups, group_names=["en", "de"],
                  iters=50, num_samples=1, sample_interval=5)
        assert model.groups == ["en", "de"]
        # Index 0 should be 'en'
        assert model.groups[0] == "en"

    def test_group_names_fixes_order_reversed(self):
        """group_names=['de','en'] → groups == ['de','en']."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = ["en"] * 10 + ["de"] * 10
        model = SAGE(num_topics=2, seed=1)
        model.fit(docs, groups, group_names=["de", "en"],
                  iters=50, num_samples=1, sample_interval=5)
        assert model.groups == ["de", "en"]
        assert model.groups[0] == "de"

    def test_group_names_index_0_matches_first_name(self):
        """topic_word[:, 0, :] should correspond to the first group in group_names."""
        docs   = [["cat", "dog"]] * 10 + [["bird", "fish"]] * 10
        groups = ["en"] * 10 + ["de"] * 10
        m_en0 = SAGE(num_topics=2, seed=1)
        m_en0.fit(docs, groups, group_names=["en", "de"],
                  iters=50, num_samples=1, sample_interval=5)
        m_de0 = SAGE(num_topics=2, seed=1)
        m_de0.fit(docs, groups, group_names=["de", "en"],
                  iters=50, num_samples=1, sample_interval=5)
        # group_names ordering should change which slice is index 0
        tw_en0_g0 = m_en0.top_words(3, topic=0, group=0)
        tw_en0_gname = m_en0.top_words(3, topic=0, group="en")
        assert tw_en0_g0 == tw_en0_gname


# ---------------------------------------------------------------------------
# Input type parity: Corpus vs list[list[str]]
# ---------------------------------------------------------------------------

class TestSAGEInputTypeParity:
    def test_corpus_and_token_list_give_identical_results(self):
        docs   = [["cat", "dog"]] * 20 + [["bird", "fish"]] * 20
        groups = ["a"] * 20 + ["b"] * 20
        corpus = Corpus.from_documents(docs)

        m1 = SAGE(num_topics=2, seed=5)
        m1.fit(docs, groups, iters=100, num_samples=2, sample_interval=5)

        m2 = SAGE(num_topics=2, seed=5)
        m2.fit(corpus, groups, iters=100, num_samples=2, sample_interval=5)

        npt.assert_array_equal(m1.topic_word, m2.topic_word)
        npt.assert_array_equal(m1.doc_topic, m2.doc_topic)


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

class TestSAGECoherence:
    def test_coherence_shape(self, bilingual_model):
        c = bilingual_model.coherence(n=5)
        assert c.shape == (2,)

    def test_coherence_values_nonpositive(self, bilingual_model):
        c = bilingual_model.coherence(n=5)
        assert (c <= 0).all()


# ---------------------------------------------------------------------------
# top_words structure and probabilities
# ---------------------------------------------------------------------------

class TestSAGETopWordsStructure:
    def test_top_words_returns_word_prob_tuples(self, bilingual_model):
        result = bilingual_model.top_words(5, topic=0, group="en")
        assert isinstance(result, list)
        assert len(result) == 5
        for w, p in result:
            assert isinstance(w, str)
            assert isinstance(p, float)
            assert 0.0 <= p <= 1.0

    def test_top_words_probabilities_descending(self, bilingual_model):
        for t in range(2):
            for g in ["en", "de"]:
                probs = [p for _, p in bilingual_model.top_words(7, topic=t, group=g)]
                assert probs == sorted(probs, reverse=True)

    def test_top_words_marginal_probabilities_descending(self, bilingual_model):
        for t in range(2):
            probs = [p for _, p in bilingual_model.top_words(5, topic=t)]
            assert probs == sorted(probs, reverse=True)

    def test_word_contrast_returns_word_log_ratio_tuples(self, bilingual_model):
        result = bilingual_model.word_contrast(0, "en", "de", 5)
        assert isinstance(result, list)
        assert len(result) == 5
        for w, lr in result:
            assert isinstance(w, str)
            assert isinstance(lr, float)
            assert np.isfinite(lr)

    def test_word_contrast_positive_log_ratios_at_top(self, bilingual_model):
        """The top entries of word_contrast(t, 'en', 'de') should be positive (favour 'en')."""
        for t in range(2):
            top_lr = [lr for _, lr in bilingual_model.word_contrast(t, "en", "de", 3)]
            assert all(lr > 0 for lr in top_lr), (
                f"Topic {t}: expected positive log-ratios, got {top_lr}"
            )


# ---------------------------------------------------------------------------
# #105: canonical top_words(n=10, *, topic=None, group=None) shape contract
# ---------------------------------------------------------------------------

class TestSAGETopWordsCanonicalSignature:
    """Verify that SAGE.top_words now matches the canonical model shape:
    - top_words(5) returns 5 words per topic (all topics, list of lists)
    - top_words(5, topic=0) returns just topic 0's 5 words (flat list)
    - group= still filters by group as before
    """

    def test_top_words_n_only_returns_all_topics(self, bilingual_model):
        """top_words(5) returns a list[list] with one entry per topic."""
        result = bilingual_model.top_words(5)
        assert isinstance(result, list)
        assert len(result) == bilingual_model.num_topics
        for topic_words in result:
            assert isinstance(topic_words, list)
            assert len(topic_words) == 5
            for w, p in topic_words:
                assert isinstance(w, str)
                assert isinstance(p, float)

    def test_top_words_5_returns_5_words_per_topic(self, bilingual_model):
        """Explicit check: top_words(5) gives exactly 5 words for each topic."""
        result = bilingual_model.top_words(5)
        for topic_words in result:
            assert len(topic_words) == 5

    def test_top_words_topic_kwarg_returns_single_list(self, bilingual_model):
        """top_words(5, topic=0) returns a flat list (not a list of lists)."""
        result = bilingual_model.top_words(5, topic=0)
        assert isinstance(result, list)
        assert len(result) == 5
        # Each element is a (word, prob) tuple, not a nested list.
        for item in result:
            assert isinstance(item, tuple)
            w, p = item
            assert isinstance(w, str)
            assert isinstance(p, float)

    def test_top_words_topic_1_returns_single_list(self, bilingual_model):
        result = bilingual_model.top_words(5, topic=1)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_top_words_with_group_all_topics(self, bilingual_model):
        """top_words(5, group='en') returns 5 words per topic using en distribution."""
        result = bilingual_model.top_words(5, group="en")
        assert len(result) == bilingual_model.num_topics
        for topic_words in result:
            assert len(topic_words) == 5

    def test_top_words_with_group_single_topic(self, bilingual_model):
        """top_words(5, topic=0, group='de') returns the de words for topic 0."""
        result = bilingual_model.top_words(5, topic=0, group="de")
        assert isinstance(result, list)
        assert len(result) == 5
        words = {w for w, _ in result}
        # Must be German words (fully disjoint vocabulary)
        assert words <= _DE_VOCAB


# ---------------------------------------------------------------------------
# Issue #422 correctness fixes
# ---------------------------------------------------------------------------
def _group_corpus(seed=0, n=60):
    rng = np.random.default_rng(seed)
    A = "cat dog pet vet paw".split()
    B = "star moon sky sun orbit".split()
    docs, grp = [], []
    for i in range(n):
        base = A if i % 2 == 0 else B
        docs.append(list(rng.choice(base, size=12)) + list(rng.choice(A + B, size=2)))
        grp.append("g0" if i < n // 2 else "g1")
    return docs, grp


class TestSageConvergenceFix:
    """convergence_tol must not trip before any kappa is learned (issue #422):
    with kappa=0 the word LL is a corpus constant, so the old code stopped at
    iter 20, before burn_in."""

    def test_no_false_stop_before_kappa(self):
        docs, grp = _group_corpus()
        # burn_in=200 -> first kappa update at iter 250; must not stop before then
        m = SAGE(2, seed=1).fit(docs, grp, iters=300,
                                convergence_tol=1e-4, check_every=10)
        if m.converged:
            assert m.fit_history[-1][0] > 200  # never the old iter-20 stop

    def test_earliest_stop_is_the_second_post_kappa_trace(self):
        """Misaligned update/trace cadences lock the off-by-one: with the first
        kappa update at iter 49 (optimize_interval=7 past burn_in=45, NOT a trace
        iteration) and traces at 50, 60, ..., the earliest legitimate stop is the
        second post-kappa trace (iter 60), never a pre-kappa point (40/50)."""
        docs, grp = _group_corpus()
        m = SAGE(2, seed=1, burn_in=45, optimize_interval=7).fit(
            docs, grp, iters=400, convergence_tol=0.5, check_every=10)
        assert m.converged
        assert m.fit_history[-1][0] == 60  # exactly the first valid comparison

    def test_early_stop_still_fires_after_kappa(self):
        docs, grp = _group_corpus()
        m = SAGE(2, seed=1, burn_in=40, optimize_interval=10).fit(
            docs, grp, iters=2000, convergence_tol=1e-3, check_every=10)
        assert m.converged
        assert m.fit_history[-1][0] > 50  # after the first kappa update (iter 50)


class TestSageNonFiniteGuard:
    """A non-finite kappa optimization step must not corrupt the topics (#422)."""

    def test_extreme_prior_variance_warns_and_stays_finite(self):
        import warnings
        docs, grp = _group_corpus()
        # 5e-324 (smallest subnormal) -> inv_var overflows -> non-finite L-BFGS result
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = SAGE(2, seed=1, prior_variance=5e-324).fit(
                docs, grp, iters=300, convergence_tol=1e-3, check_every=10)
        msgs = [str(x.message) for x in caught if "non-finite" in str(x.message)]
        assert len(msgs) == 1                       # warned once, aggregated
        assert np.isfinite(m.topic_word).all()      # topics not corrupted
        assert np.isfinite(m.doc_topic).all()
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)
        assert not m.converged                      # early stop impossible (LL constant)


class TestSageValidation:
    """Input validation added in issue #422."""

    @pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": -1.0},
                                        {"alpha": float("nan")}, {"alpha": float("inf")},
                                        {"lbfgs_iters": 0}])
    def test_bad_constructor_args(self, kwargs):
        with pytest.raises(ValueError):
            SAGE(2, **kwargs)

    def test_num_samples_zero_rejected(self):
        docs, grp = _group_corpus()
        with pytest.raises(ValueError):
            SAGE(2).fit(docs, grp, iters=50, num_samples=0)

    @pytest.mark.parametrize("tol", [-1.0, float("nan"), float("inf")])
    def test_bad_convergence_tol_rejected(self, tol):
        docs, grp = _group_corpus()
        with pytest.raises(ValueError):
            SAGE(2).fit(docs, grp, iters=20, convergence_tol=tol)

    def test_duplicate_group_names_rejected(self):
        # Both real groups (g0, g1) are present, so only the duplicate check can
        # raise — the pre-#422 code would train g0 twice without error.
        docs, grp = _group_corpus()
        with pytest.raises(ValueError, match="duplicate"):
            SAGE(2).fit(docs, grp, group_names=["g0", "g1", "g0"], iters=20)


# ---------------------------------------------------------------------------
# Sparse prior (issue #422 part 2)
# ---------------------------------------------------------------------------
def _shared_corpus(seed=0, n=80):
    """Genuinely group-discriminative: group g0 draws from A, g1 from B (the
    vocabulary is tied to the GROUP, not to document parity), plus a few shared
    background words common to both groups — so a sparse prior can drive the
    background deviations to ~0 while keeping the discriminative ones."""
    rng = np.random.default_rng(seed)
    A = "cat dog pet vet paw".split()
    B = "star moon sky sun orbit".split()
    shared = "the a of".split()
    docs, grp = [], []
    for i in range(n):
        g0 = i < n // 2
        base = A if g0 else B
        docs.append(list(rng.choice(base, size=10)) + list(rng.choice(shared, size=3)))
        grp.append("g0" if g0 else "g1")
    return docs, grp


class TestSageSparsePrior:
    def test_default_prior_is_laplace(self):
        assert SAGE(2).settings["prior"] == "laplace"

    def test_bad_prior_rejected(self):
        with pytest.raises(ValueError):
            SAGE(2, prior="banana")

    @pytest.mark.parametrize("prior", ["laplace", "gaussian", "jeffreys"])
    def test_fit_and_content_kappa(self, prior):
        docs, grp = _shared_corpus()
        m = SAGE(2, prior=prior, seed=1).fit(docs, grp, iters=250)
        ck = m.content_kappa
        assert set(ck) == {"topic", "group", "interaction"}
        V = len(m.vocabulary)
        assert ck["topic"].shape == (2, V)
        assert ck["group"].shape == (m.num_groups, V)
        assert ck["interaction"].shape == (2 * m.num_groups, V)
        assert m.prior == prior
        # a valid fit under every prior: doc_topic rows sum to 1
        npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)

    def test_laplace_shrinks_background_harder(self):
        # Sparsity shows on the NON-discriminative (shared-background) words: their
        # group deviation should be ~0. Total L1 is the wrong measure — Laplace
        # correctly lets the true group signals grow (that is what L1 vs L2 does),
        # so it must be measured on the background words specifically (as the Rust
        # test does).
        docs, grp = _shared_corpus()
        lap = SAGE(2, prior="laplace", seed=1).fit(docs, grp, iters=300)
        gau = SAGE(2, prior="gaussian", seed=1).fit(docs, grp, iters=300)
        shared = {"the", "a", "of"}
        bg = [i for i, w in enumerate(lap.vocabulary) if w in shared]
        assert bg, "background words not in vocabulary"
        lap_bg = np.abs(lap.content_kappa["group"][:, bg]).sum()
        gau_bg = np.abs(gau.content_kappa["group"][:, bg]).sum()
        assert lap_bg < gau_bg, f"laplace bg |κ| {lap_bg:.3f} !< gaussian {gau_bg:.3f}"

    def test_save_load_preserves_prior_and_kappa(self, tmp_path):
        docs, grp = _shared_corpus()
        m = SAGE(2, prior="jeffreys", seed=1).fit(docs, grp, iters=200)
        p = str(tmp_path / "m.tt")
        m.save(p)
        n = SAGE.load(p)
        assert n.prior == "jeffreys"
        assert np.array_equal(n.content_kappa["group"], m.content_kappa["group"])
        assert np.array_equal(n.topic_word, m.topic_word)

    def test_content_kappa_requires_fit(self):
        with pytest.raises(RuntimeError):
            _ = SAGE(2).content_kappa

    def test_jeffreys_is_scale_free(self):
        # The Normal-Jeffreys prior is scale-free: prior_variance must not change it.
        docs, grp = _shared_corpus()
        a = SAGE(2, prior="jeffreys", prior_variance=1e-3, seed=1).fit(docs, grp, iters=200)
        b = SAGE(2, prior="jeffreys", prior_variance=1e3, seed=1).fit(docs, grp, iters=200)
        np.testing.assert_allclose(
            a.content_kappa["group"], b.content_kappa["group"], atol=1e-9
        )
