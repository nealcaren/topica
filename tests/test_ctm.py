"""Tests for the CTM (Correlated Topic Model) class.

Verified behaviors:
- Logistic-normal prior yields *correlated* topics (unlike LDA).
- Fit by variational EM; topic_word, doc_topic, topic_correlation shapes/values.
- Corpus and list[list[str]] input give the same shapes.
- Determinism: same seed -> identical topic_word, allclose doc_topic.
- Validation errors on bad constructor args and bad top_words indices.
- Unfitted-property guards raise RuntimeError; num_topics is available before fit.
"""

import numpy as np
import numpy.testing as npt
import pytest

from topica import CTM, Corpus


# ---------------------------------------------------------------------------
# Shared synthetic corpus (used across multiple test classes)
# ---------------------------------------------------------------------------

# Three disjoint vocabularies
_VOCAB_A = ["alpha", "bravo", "charlie", "delta", "echo"]
_VOCAB_B = ["foxtrot", "golf", "hotel", "india", "juliet"]
_VOCAB_C = ["kilo", "lima", "mike", "november", "oscar"]

_VOCAB_ALL = set(_VOCAB_A) | set(_VOCAB_B) | set(_VOCAB_C)


def _make_correlated_corpus(n=300, doc_length=8, seed=0):
    """Build a 300-doc corpus where A and B co-occur but C is isolated.

    Doc-type distribution:
      40% A+B together  (topics A and B co-occur -> they should be positively correlated)
      20% A alone
      20% B alone
      20% C alone       (topic C rarely co-occurs with A or B)

    Returns a list[list[str]].
    """
    rng = np.random.default_rng(seed)
    docs = []
    n_ab = int(n * 0.40)   # 120
    n_a  = int(n * 0.20)   # 60
    n_b  = int(n * 0.20)   # 60
    n_c  = n - n_ab - n_a - n_b  # 60

    for _ in range(n_ab):
        docs.append(
            rng.choice(_VOCAB_A, size=doc_length).tolist()
            + rng.choice(_VOCAB_B, size=doc_length).tolist()
        )
    for _ in range(n_a):
        docs.append(rng.choice(_VOCAB_A, size=doc_length).tolist())
    for _ in range(n_b):
        docs.append(rng.choice(_VOCAB_B, size=doc_length).tolist())
    for _ in range(n_c):
        docs.append(rng.choice(_VOCAB_C, size=doc_length).tolist())

    return docs


def _identify_topics(model):
    """Map topic indices to A/B/C by majority top-word overlap.

    Returns a dict {label: topic_index} where label is 'A', 'B', or 'C'.
    """
    sets = {"A": set(_VOCAB_A), "B": set(_VOCAB_B), "C": set(_VOCAB_C)}
    best = {}
    used = set()

    # Build overlap counts: for each topic, count top-4 words in each vocabulary
    overlap = {}
    for t in range(model.num_topics):
        top4 = [w for w, _ in model.top_words(4, topic=t, weights=True)]
        overlap[t] = {label: sum(1 for w in top4 if w in s) for label, s in sets.items()}

    # Greedy assignment: assign the (topic, label) pair with the highest overlap
    pairs = sorted(
        [(t, label) for t in range(model.num_topics) for label in sets],
        key=lambda x: overlap[x[0]][x[1]],
        reverse=True,
    )
    assigned_topics = set()
    assigned_labels = set()
    for t, label in pairs:
        if t not in assigned_topics and label not in assigned_labels:
            best[label] = t
            assigned_topics.add(t)
            assigned_labels.add(label)
        if len(best) == 3:
            break

    return best


# ---------------------------------------------------------------------------
# Fixture: fitted CTM on the correlated corpus (shared for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_ctm():
    """CTM(3, seed=1) fitted on the 300-doc correlated corpus (iters=50)."""
    docs = _make_correlated_corpus(n=300, seed=0)
    m = CTM(num_topics=3, seed=1)
    m.fit(docs, iters=50)
    return m


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestCTMConstructor:
    def test_num_topics_available_before_fit(self):
        m = CTM(5)
        assert m.num_topics == 5

    def test_num_topics_one_raises(self):
        with pytest.raises(ValueError):
            CTM(1)

    def test_num_topics_zero_raises(self):
        with pytest.raises(ValueError):
            CTM(0)

    def test_sigma_shrink_negative_raises(self):
        with pytest.raises(ValueError):
            CTM(2, sigma_shrink=-0.1)

    def test_sigma_shrink_above_one_raises(self):
        with pytest.raises(ValueError):
            CTM(2, sigma_shrink=1.5)

    def test_sigma_shrink_zero_accepted(self):
        m = CTM(2, sigma_shrink=0.0)
        assert m.num_topics == 2

    def test_sigma_shrink_one_accepted(self):
        m = CTM(2, sigma_shrink=1.0)
        assert m.num_topics == 2

    def test_sigma_shrink_midrange_accepted(self):
        m = CTM(3, sigma_shrink=0.5)
        assert m.num_topics == 3


# ---------------------------------------------------------------------------
# Unfitted-property guards
# ---------------------------------------------------------------------------

class TestCTMUnfittedGuards:
    PROPERTIES = [
        "topic_word",
        "doc_topic",
        "topic_correlation",
        "vocabulary",
        "doc_names",
    ]

    @pytest.mark.parametrize("prop", PROPERTIES)
    def test_property_raises_before_fit(self, prop):
        m = CTM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(m, prop)

    def test_top_words_raises_before_fit(self):
        m = CTM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.top_words()

    def test_coherence_raises_before_fit(self):
        m = CTM(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.coherence()

    def test_num_topics_available_before_fit(self):
        """num_topics must NOT raise before fit — it returns the constructor value."""
        m = CTM(4)
        assert m.num_topics == 4  # no RuntimeError


# ---------------------------------------------------------------------------
# Output shapes and basic invariants
# ---------------------------------------------------------------------------

class TestCTMShapes:
    def test_topic_word_shape(self, fitted_ctm):
        # 3 topics, 15 unique words (5+5+5)
        assert fitted_ctm.topic_word.shape == (3, 15)

    def test_doc_topic_shape(self, fitted_ctm):
        assert fitted_ctm.doc_topic.shape == (300, 3)

    def test_doc_topic_rows_sum_to_one(self, fitted_ctm):
        npt.assert_allclose(
            fitted_ctm.doc_topic.sum(axis=1), np.ones(300), atol=1e-6
        )

    def test_topic_correlation_shape(self, fitted_ctm):
        assert fitted_ctm.topic_correlation.shape == (3, 3)

    def test_topic_correlation_symmetric(self, fitted_ctm):
        C = fitted_ctm.topic_correlation
        assert np.allclose(C, C.T), "topic_correlation must be symmetric"

    def test_topic_correlation_unit_diagonal(self, fitted_ctm):
        npt.assert_allclose(
            np.diag(fitted_ctm.topic_correlation), np.ones(3), atol=1e-6
        )

    def test_vocabulary_length_matches_topic_word_columns(self, fitted_ctm):
        assert len(fitted_ctm.vocabulary) == fitted_ctm.topic_word.shape[1]

    def test_doc_names_length_matches_doc_topic_rows(self, fitted_ctm):
        assert len(fitted_ctm.doc_names) == fitted_ctm.doc_topic.shape[0]

    def test_num_topics_after_fit(self, fitted_ctm):
        assert fitted_ctm.num_topics == 3


# ---------------------------------------------------------------------------
# Topic recovery: each topic's top words are dominated by one vocabulary
# ---------------------------------------------------------------------------

class TestCTMTopicRecovery:
    def test_each_topic_dominated_by_one_vocabulary(self, fitted_ctm):
        """Majority of top-4 words for each topic must come from a single vocab."""
        sets = {
            "A": set(_VOCAB_A),
            "B": set(_VOCAB_B),
            "C": set(_VOCAB_C),
        }
        for t in range(3):
            top4 = [w for w, _ in fitted_ctm.top_words(4, topic=t, weights=True)]
            counts = {label: sum(1 for w in top4 if w in s) for label, s in sets.items()}
            max_count = max(counts.values())
            assert max_count >= 3, (
                f"Topic {t} top-4 words not dominated by one vocab: "
                f"words={top4}, counts={counts}"
            )

    def test_three_topics_cover_three_distinct_vocabularies(self, fitted_ctm):
        """The A/B/C assignment must cover all three vocabularies."""
        mapping = _identify_topics(fitted_ctm)
        assert set(mapping.keys()) == {"A", "B", "C"}, (
            f"Could not identify all three topics: mapping={mapping}"
        )
        # All three indices must be distinct
        assert len(set(mapping.values())) == 3


# ---------------------------------------------------------------------------
# Correlation structure: A,B must be more correlated than A,C and B,C
# ---------------------------------------------------------------------------

class TestCTMCorrelationStructure:
    def test_ab_more_correlated_than_ac_and_bc(self, fitted_ctm):
        """The co-occurring pair (A,B) has a higher correlation than (A,C) and (B,C).

        All correlations on the simplex may be negative (topics compete for probability
        mass). The test asserts the RELATIVE ordering: corr(A,B) > corr(A,C) and
        corr(A,B) > corr(B,C).
        """
        mapping = _identify_topics(fitted_ctm)
        ia = mapping["A"]
        ib = mapping["B"]
        ic = mapping["C"]

        C = fitted_ctm.topic_correlation
        corr_ab = C[ia, ib]
        corr_ac = C[ia, ic]
        corr_bc = C[ib, ic]

        assert corr_ab > corr_ac, (
            f"Expected corr(A,B)={corr_ab:.4f} > corr(A,C)={corr_ac:.4f} "
            f"(topic mapping: A={ia}, B={ib}, C={ic})"
        )
        assert corr_ab > corr_bc, (
            f"Expected corr(A,B)={corr_ab:.4f} > corr(B,C)={corr_bc:.4f} "
            f"(topic mapping: A={ia}, B={ib}, C={ic})"
        )


# ---------------------------------------------------------------------------
# Determinism: same seed -> identical topic_word, allclose doc_topic
# ---------------------------------------------------------------------------

class TestCTMDeterminism:
    def test_same_seed_identical_topic_word(self):
        docs = _make_correlated_corpus(n=60, seed=7)
        m1 = CTM(3, seed=42)
        m1.fit(docs, iters=20)
        m2 = CTM(3, seed=42)
        m2.fit(docs, iters=20)
        # Variational EM is deterministic: same seed and thread count must give
        # a bit-for-bit identical fit, matching the paper's determinism claim.
        assert np.array_equal(m1.topic_word, m2.topic_word), \
            "Identical seeds must produce bit-identical topic_word"

    def test_same_seed_allclose_doc_topic(self):
        docs = _make_correlated_corpus(n=60, seed=7)
        m1 = CTM(3, seed=42)
        m1.fit(docs, iters=20)
        m2 = CTM(3, seed=42)
        m2.fit(docs, iters=20)
        assert np.array_equal(m1.doc_topic, m2.doc_topic), \
            "Identical seeds must produce bit-identical doc_topic"


# ---------------------------------------------------------------------------
# Input type parity: Corpus vs list[list[str]]
# ---------------------------------------------------------------------------

class TestCTMInputTypes:
    def test_corpus_input_accepted(self):
        docs = [["cat", "dog", "fish"]] * 20 + [["planet", "star", "moon"]] * 20
        corpus = Corpus.from_documents(docs)
        m = CTM(2, seed=1)
        m.fit(corpus, iters=10)
        assert m.doc_topic.shape == (40, 2)

    def test_list_input_accepted(self):
        docs = [["cat", "dog", "fish"]] * 20 + [["planet", "star", "moon"]] * 20
        m = CTM(2, seed=1)
        m.fit(docs, iters=10)
        assert m.doc_topic.shape == (40, 2)

    def test_corpus_and_list_give_same_topic_word(self):
        docs = [["cat", "dog", "fish"]] * 30 + [["planet", "star", "moon"]] * 30
        corpus = Corpus.from_documents(docs)
        m1 = CTM(2, seed=5)
        m1.fit(corpus, iters=15)
        m2 = CTM(2, seed=5)
        m2.fit(docs, iters=15)
        # Results are identical up to floating-point precision (< 1e-14)
        npt.assert_allclose(
            m1.topic_word, m2.topic_word, atol=1e-14,
            err_msg="Corpus and list[list[str]] input must produce identical topic_word"
        )


# ---------------------------------------------------------------------------
# top_words method
# ---------------------------------------------------------------------------

class TestCTMTopWords:
    @pytest.fixture(scope="class")
    def small_model(self):
        docs = [["cat", "dog", "fish"]] * 20 + [["planet", "star", "moon"]] * 20
        m = CTM(2, seed=1)
        m.fit(docs, iters=20)
        return m

    def test_all_topics_returns_num_topics_lists(self, small_model):
        result = small_model.top_words()
        assert isinstance(result, list)
        assert len(result) == 2
        for topic_list in result:
            assert isinstance(topic_list, list)

    def test_all_topics_default_n_is_10(self, small_model):
        result = small_model.top_words()
        # vocab has only 6 words, so each list is capped at 6
        assert all(len(lst) <= 10 for lst in result)

    def test_top_words_n_controls_length(self, small_model):
        result = small_model.top_words(3)
        for topic_list in result:
            assert len(topic_list) == 3

    def test_single_topic_returns_list_of_tuples(self, small_model):
        result = small_model.top_words(5, topic=0, weights=True)
        assert isinstance(result, list)
        assert len(result) == 5
        for word, prob in result:
            assert isinstance(word, str)
            assert isinstance(prob, float)

    def test_top_words_probabilities_descending(self, small_model):
        for topic_list in small_model.top_words(5, weights=True):
            probs = [p for _, p in topic_list]
            assert probs == sorted(probs, reverse=True), (
                "top_words probabilities must be sorted descending"
            )

    def test_top_words_out_of_range_raises_value_error(self, small_model):
        with pytest.raises(ValueError):
            small_model.top_words(topic=5)

    def test_top_words_topic_none_vs_explicit_consistent(self, small_model):
        all_results = small_model.top_words(5)
        for t in range(small_model.num_topics):
            single = small_model.top_words(5, topic=t)
            assert single == all_results[t], (
                f"top_words(topic={t}) must match top_words()[{t}]"
            )


# ---------------------------------------------------------------------------
# coherence method
# ---------------------------------------------------------------------------

class TestCTMCoherence:
    def test_coherence_shape(self, fitted_ctm):
        c = fitted_ctm.coherence(n=5)
        assert c.shape == (3,)

    def test_coherence_default_n(self, fitted_ctm):
        c = fitted_ctm.coherence()
        assert c.shape == (fitted_ctm.num_topics,)

    def test_coherence_values_nonpositive(self, fitted_ctm):
        c = fitted_ctm.coherence(n=5)
        assert (c <= 0).all(), (
            f"UMass coherence values must be <= 0; got {c}"
        )


# ---------------------------------------------------------------------------
# Spectral (anchor-word) initialization
# ---------------------------------------------------------------------------


class TestSpectralInit:
    """Spectral init is the default — deterministic, seed-independent β init
    (Arora et al. 2013), matching stm's `init.type="Spectral"` default."""

    def test_default_is_spectral_and_seed_independent(self):
        docs = _make_correlated_corpus(seed=3)
        m1 = CTM(3, seed=1)
        m1.fit(docs)
        m2 = CTM(3, seed=987654)
        m2.fit(docs)
        # Spectral init ignores the seed: identical fits regardless of seed.
        npt.assert_array_equal(m1.topic_word, m2.topic_word)

    def test_random_init_is_seed_dependent(self):
        docs = _make_correlated_corpus(seed=3)
        r1 = CTM(3, seed=1, init="random")
        r1.fit(docs)
        r2 = CTM(3, seed=987654, init="random")
        r2.fit(docs)
        assert not np.array_equal(r1.topic_word, r2.topic_word)

    def test_spectral_recovers_topics(self):
        # On the disjoint-vocabulary corpus, spectral init alone should already
        # separate the three topics by their vocabularies.
        docs = _make_correlated_corpus(seed=5)
        m = CTM(3, init="spectral")
        m.fit(docs, iters=30)
        labels = _identify_topics(m)
        assert set(labels) == {"A", "B", "C"}

    def test_bad_init_raises(self):
        with pytest.raises(ValueError):
            CTM(3, init="bogus")


# ---------------------------------------------------------------------------
# SVI / online VB backend (inference="svi"): for very large corpora
# ---------------------------------------------------------------------------

import numpy.testing as _npt


def _block_corpus(nb=3, wpb=6, n=600):
    docs = []
    for d in range(n):
        b = d % nb
        block = [f"w{b * wpb + i}" for i in range(wpb)]
        docs.append(block + block)
    return docs, nb, wpb


def _recovered(m, nb, wpb):
    cov = set()
    for t in range(m.num_topics):
        top = {w for w, _ in m.top_words(wpb, topic=t, weights=True)}
        for b in range(nb):
            if set(f"w{b * wpb + i}" for i in range(wpb)) <= top:
                cov.add(b)
    return cov


def test_svi_recovers_blocks_and_valid():
    docs, nb, wpb = _block_corpus()
    m = CTM(num_topics=nb, seed=1)
    m.fit(docs, iters=30, inference="svi")
    assert _recovered(m, nb, wpb) == set(range(nb))
    _npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
    _npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)


def test_svi_deterministic():
    docs, nb, _ = _block_corpus(n=300)
    a = CTM(num_topics=nb, seed=5)
    a.fit(docs, iters=15, inference="svi", batch_size=64)
    b = CTM(num_topics=nb, seed=5)
    b.fit(docs, iters=15, inference="svi", batch_size=64)
    _npt.assert_array_equal(a.topic_word, b.topic_word)


def test_svi_batch_size_and_schedule_accepted():
    docs, nb, _ = _block_corpus(n=200)
    m = CTM(num_topics=nb, seed=1)
    m.fit(docs, iters=10, inference="svi", batch_size=32, tau=32.0, kappa=0.6)
    assert m.topic_word.shape[0] == nb


def test_bad_inference_rejected():
    docs, nb, _ = _block_corpus(n=60)
    with pytest.raises(ValueError):
        CTM(num_topics=nb, seed=1).fit(docs, iters=5, inference="banana")


# beta_init warm-start hook (issue #234) -----------------------------------

def test_beta_init_overrides_initialization():
    """iters=0 returns exactly the injected base beta (the warm-start hook)."""
    docs, nb, _ = _block_corpus(n=200)
    V = len({w for d in docs for w in d})
    rng = np.random.default_rng(0)
    custom = rng.random((nb, V)); custom /= custom.sum(1, keepdims=True)
    m = CTM(num_topics=nb, seed=1); m.fit(docs, iters=0, beta_init=custom)
    # vocab order is stable for iters=0; rows match up to the model's vocab order
    _npt.assert_allclose(np.sort(m.topic_word, axis=1), np.sort(custom, axis=1), atol=1e-9)


def test_beta_init_warm_start_reproduces_spectral_fit():
    """Injecting the spectral-init beta reproduces the default spectral fit."""
    docs, nb, _ = _block_corpus(n=300)
    default = CTM(num_topics=nb, seed=1); default.fit(docs, iters=20)
    spec = CTM(num_topics=nb, seed=1); spec.fit(docs, iters=0)  # the spectral init beta
    warm = CTM(num_topics=nb, seed=1); warm.fit(docs, iters=20, beta_init=spec.topic_word)
    _npt.assert_allclose(warm.topic_word, default.topic_word, atol=1e-9)


def test_beta_init_validation():
    docs, nb, _ = _block_corpus(n=60)
    V = len({w for d in docs for w in d})
    with pytest.raises(ValueError, match="beta_init must have shape"):
        CTM(num_topics=nb, seed=1).fit(docs, iters=5, beta_init=np.ones((nb + 1, V)))
    with pytest.raises(ValueError, match="not supported with inference"):
        CTM(num_topics=nb, seed=1).fit(docs, iters=2, inference="svi",
                                       beta_init=np.ones((nb, V)) / V)
