"""Topic coherence (u_mass / c_uci / c_npmi / c_v) and topic diversity.

The measures are validated by their defining properties: a topic of words that
always co-occur should score higher than a topic of words that never do, the
normalized measures must respect their ranges, and the gensim-style API should
accept both explicit word lists and fitted models.
"""

import numpy as np
import pytest

import topica
from topica.models import LDA

ALL_TYPES = ["u_mass", "c_uci", "c_npmi", "c_v"]


@pytest.fixture(scope="module")
def reference():
    """A corpus where {a,b,c} always co-occur and the r* words are scattered."""
    rng = np.random.default_rng(0)
    texts = []
    for _ in range(500):
        d = []
        if rng.random() < 0.5:
            d += ["a", "b", "c"] * 3
        d += [f"r{int(rng.integers(200))}" for _ in range(6)]
        rng.shuffle(d)
        texts.append(d)
    return texts


COHERENT = ["a", "b", "c"]
INCOHERENT = ["r1", "r2", "r3"]


class TestCoherenceRanksTopics:
    @pytest.mark.parametrize("ct", ALL_TYPES)
    def test_coherent_beats_incoherent(self, reference, ct):
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type=ct, topn=3)
        assert s.shape == (2,)
        assert s[0] > s[1], f"{ct}: coherent {s[0]} !> incoherent {s[1]}"

    def test_per_topic_shape(self, reference):
        s = topica.coherence([COHERENT, INCOHERENT, ["a", "b", "c"]], reference, coherence_type="c_v", topn=3)
        assert s.shape == (3,)


class TestCoherenceCI:
    def test_shape_and_estimate_matches_point(self, reference):
        r = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=100, seed=0
        )
        point = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3)
        for arr in (r.estimate, r.se, r.ci_low, r.ci_high):
            assert arr.shape == (2,)
        np.testing.assert_allclose(r.estimate, point)

    def test_interval_brackets_estimate_and_orders(self, reference):
        r = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=150, seed=1
        )
        assert np.all(r.ci_low <= r.estimate + 1e-9)
        assert np.all(r.estimate <= r.ci_high + 1e-9)
        assert np.all(r.ci_low <= r.ci_high)
        assert np.all(np.isfinite(r.se)) and np.all(r.se >= 0.0)

    def test_wider_ci_widens_band(self, reference):
        wide = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=150, ci=0.95, seed=2
        )
        narrow = topica.coherence_ci(
            [COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3, n_boot=150, ci=0.5, seed=2
        )
        assert float(np.sum(wide.ci_high - wide.ci_low)) > float(np.sum(narrow.ci_high - narrow.ci_low))

    def test_accepts_fitted_model(self, reference):
        m = LDA(num_topics=2, seed=1)
        m.fit(reference, iters=200)
        r = topica.coherence_ci(m, reference, coherence_type="c_npmi", topn=3, n_boot=50, seed=0)
        assert r.estimate.shape == (2,)

    def test_empty_texts_raises(self):
        with pytest.raises(ValueError, match="texts is empty"):
            topica.coherence_ci([COHERENT], [], n_boot=10)


class TestRanges:
    def test_npmi_in_unit_range(self, reference):
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3)
        assert np.all(s >= -1.0001) and np.all(s <= 1.0001)

    def test_cv_nonnegative_ish(self, reference):
        # C_v is a cosine of non-negative-ish context vectors; in [0, 1].
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_v", topn=3)
        assert np.all(s >= -0.01) and np.all(s <= 1.0001)

    def test_umass_nonpositive_for_rare(self, reference):
        s = topica.coherence([INCOHERENT], reference, coherence_type="u_mass", topn=3)
        assert s[0] <= 0.0


class TestApi:
    def test_invalid_type_raises(self, reference):
        with pytest.raises(ValueError):
            topica.coherence([COHERENT], reference, coherence_type="c_bogus")

    def test_accepts_fitted_model(self, reference):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        s = topica.coherence(m, docs, coherence_type="c_npmi", topn=3)
        assert s.shape == (2,)
        assert np.all(np.isfinite(s))

    def test_accepts_word_prob_pairs(self, reference):
        topics = [[("a", 0.5), ("b", 0.3), ("c", 0.2)]]
        s = topica.coherence(topics, reference, coherence_type="c_v", topn=3)
        assert s.shape == (1,)

    def test_window_size_override(self, reference):
        s = topica.coherence([COHERENT], reference, coherence_type="c_npmi", topn=3, window_size=5)
        assert s.shape == (1,) and np.isfinite(s[0])

    def test_default_is_cv(self, reference):
        a = topica.coherence([COHERENT], reference, topn=3)
        b = topica.coherence([COHERENT], reference, coherence_type="c_v", topn=3)
        assert np.allclose(a, b)


class TestDiversity:
    def test_disjoint_is_one(self):
        assert topica.topic_diversity([["a", "b", "c"], ["d", "e", "f"]], topn=3) == 1.0

    def test_identical_is_half(self):
        # two identical 3-word topics: 3 unique / 6 total.
        assert topica.topic_diversity([["a", "b", "c"], ["a", "b", "c"]], topn=3) == 0.5

    def test_accepts_model(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        d = topica.topic_diversity(m, topn=3)
        assert 0.0 < d <= 1.0


class TestSemanticDiversity:
    """topic_semantic_diversity (Wu, Nguyen & Luu 2024, Eq. 18): fraction of
    top-word *pairs* that are unique to a single topic."""

    def test_disjoint_is_one(self):
        # No shared words → no shared pairs → every pair unique.
        tsd = topica.topic_semantic_diversity(
            [["a", "b", "c"], ["d", "e", "f"]], topn=3
        )
        assert tsd == 1.0

    def test_identical_is_zero(self):
        # Two identical topics: every pair appears in BOTH topics, so no pair
        # occurrence has global count 1 → 0.0. (Note: unlike topic_diversity,
        # which counts unique *words* and gives 0.5 here.)
        tsd = topica.topic_semantic_diversity(
            [["a", "b", "c"], ["a", "b", "c"]], topn=3
        )
        assert tsd == 0.0

    def test_partial_overlap_exact(self):
        # A = {a,b,c,d}, B = {a,b,e,f}, topn=4 → 6 pairs each, 12 total.
        # The only shared pair is {a,b} (count 2); it occurs once in A and once
        # in B → 2 non-unique occurrences. Unique = 10 → TSD = 10/12.
        tsd = topica.topic_semantic_diversity(
            [["a", "b", "c", "d"], ["a", "b", "e", "f"]], topn=4
        )
        assert tsd == 10 / 12

    def test_topn_two_edge(self):
        # topn=2 → exactly one pair per topic; disjoint pairs → 1.0.
        tsd = topica.topic_semantic_diversity(
            [["a", "b"], ["c", "d"]], topn=2
        )
        assert tsd == 1.0

    def test_accepts_model(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        tsd = topica.topic_semantic_diversity(m, topn=3)
        assert 0.0 <= tsd <= 1.0

    def test_topn_below_two_raises(self):
        with pytest.raises(ValueError):
            topica.topic_semantic_diversity([["a", "b", "c"]], topn=1)


class TestAnalysisContract:
    """Any object exposing the analysis contract -- ``topic_word`` /
    ``doc_topic`` / ``vocabulary`` -- works with the model-agnostic diagnostics,
    even without a ``top_words`` method. This pins the extensibility guarantee:
    a foreign model that presents the two matrices inherits the stack for free.
    """

    def test_duck_typed_model_works_without_top_words(self):
        docs = [["cat", "dog", "pet"]] * 30 + [["star", "moon", "sky"]] * 30
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)

        class Contract:  # the four members, NO top_words
            topic_word = m.topic_word
            doc_topic = m.doc_topic
            vocabulary = m.vocabulary

        c = Contract()
        # coherence / topic_diversity derive top words from topic_word + vocabulary
        np.testing.assert_allclose(
            topica.coherence(c, docs), topica.coherence(m, docs)
        )
        assert topica.topic_diversity(c) == topica.topic_diversity(m)
        np.testing.assert_allclose(topica.exclusivity(c), topica.exclusivity(m))


class TestUMassExternalCorpus:
    """u_mass with an external reference corpus that does not contain some top
    words must not produce a spuriously large positive score (issue #103)."""

    def test_absent_word_does_not_inflate_umass(self):
        # Reference corpus contains only "a" and "b"; "z" is absent.
        reference = [["a", "b"]] * 20
        topic_present = ["a", "b"]   # both words in reference: normal score
        topic_absent = ["a", "z"]    # "z" absent from reference

        s_present = topica.coherence([topic_present], reference,
                                     coherence_type="u_mass", topn=2)[0]
        s_absent = topica.coherence([topic_absent], reference,
                                    coherence_type="u_mass", topn=2)[0]

        # Before the fix, s_absent was a large positive (log(1/eps)).
        # After the fix: the pair (a, z) is skipped, so s_absent equals
        # nan (no pairs counted) or a non-positive finite value.
        assert np.isnan(s_absent) or s_absent <= 0.0, (
            f"u_mass with absent word should be nan or <= 0, got {s_absent}"
        )

    def test_umass_all_words_present_is_small_and_finite(self):
        # Classic case: topic words all appear in the reference. With numerator
        # +1 smoothing and perfect co-occurrence the score can be a hair above
        # zero (log((occ+1)/occ)), but it is finite and near zero, never the
        # large spurious positive the absent-word path used to produce.
        reference = [["a", "b", "c"]] * 50
        score = topica.coherence([["a", "b", "c"]], reference,
                                 coherence_type="u_mass", topn=3)[0]
        assert np.isfinite(score)
        assert score < 0.1
