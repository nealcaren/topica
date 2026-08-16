"""Tests for LDA multi-threaded (parallel) training mode (num_threads parameter).

The feature: LDA(num_topics, ..., num_threads=1) (default = exact/sequential).
num_threads > 1 enables MALLET-style approximate parallel Gibbs sampling.

Key properties under test:
- num_threads=0 is accepted (clamped to 1); behaves identically to num_threads=1.
- Parallel results are deterministic: same (num_threads, seed) -> identical topic_word.
- Parallel model recovers two-cluster structure (animal vs space words).
- doc_topic.shape and row-sum-to-1 invariant holds after parallel training.
- topic_word.shape is correct after parallel training.
- Parallel and sequential results intentionally differ (approximation); this is
  visible on a multi-topic corpus but may coincide on very simple two-topic corpora.
- Held-out perplexity from a parallel-trained model is finite and positive.
- All downstream API methods work on a parallel-trained model.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from topica import LDA


# ---------------------------------------------------------------------------
# Shared two-cluster corpus (60 animal + 60 space, 10 tokens/doc).
# Used for shape/quality/downstream tests where two topics are sufficient.
# ---------------------------------------------------------------------------

ANIMAL_WORDS = [
    "cat", "dog", "fish", "bird", "horse", "rabbit", "hamster", "turtle",
    "lizard", "parrot",
]
SPACE_WORDS = [
    "planet", "star", "moon", "rocket", "galaxy", "asteroid", "comet",
    "nebula", "telescope", "orbit",
]


def _make_two_cluster_corpus(n_each=60, tokens_per_doc=10, seed=0):
    rng = np.random.default_rng(seed)
    animal_docs = [
        [ANIMAL_WORDS[rng.integers(len(ANIMAL_WORDS))] for _ in range(tokens_per_doc)]
        for _ in range(n_each)
    ]
    space_docs = [
        [SPACE_WORDS[rng.integers(len(SPACE_WORDS))] for _ in range(tokens_per_doc)]
        for _ in range(n_each)
    ]
    return animal_docs + space_docs


# ---------------------------------------------------------------------------
# Four-cluster corpus — required to observe that parallel != sequential.
# On very simple two-topic corpora the parallel reconciliation can converge to
# identical values; a larger vocabulary / more topics reveals the divergence.
# ---------------------------------------------------------------------------

_FOUR_CLUSTER_VOCABS = [
    ["cat", "dog", "fish", "bird", "horse", "rabbit", "hamster", "turtle", "lizard", "parrot"],
    ["planet", "star", "moon", "rocket", "galaxy", "asteroid", "comet", "nebula", "telescope", "orbit"],
    ["apple", "banana", "cherry", "grape", "mango", "orange", "pear", "plum", "peach", "kiwi"],
    ["math", "algebra", "calculus", "geometry", "topology", "analysis", "logic", "proof", "theorem", "lemma"],
]


def _make_four_cluster_corpus(n_each=100, tokens_per_doc=15, seed=0):
    rng = np.random.default_rng(seed)
    docs = []
    for grp in _FOUR_CLUSTER_VOCABS:
        for _ in range(n_each):
            docs.append([grp[rng.integers(len(grp))] for _ in range(tokens_per_doc)])
    return docs


# Build corpora once at module level for re-use across tests.
_CORPUS_2 = _make_two_cluster_corpus(n_each=60, tokens_per_doc=10, seed=0)
_N_DOCS_2 = len(_CORPUS_2)   # 120

_CORPUS_4 = _make_four_cluster_corpus(n_each=100, tokens_per_doc=15, seed=0)
_N_DOCS_4 = len(_CORPUS_4)   # 400


def _fit_2topic(num_threads, seed=1, num_topics=2):
    """Fit a model on the two-cluster corpus (2 topics by default)."""
    model = LDA(num_topics, seed=seed, optimize_interval=0, num_threads=num_threads)
    model.fit(_CORPUS_2, iters=300, num_samples=3, sample_interval=10)
    return model


def _fit_4topic(num_threads, seed=1):
    """Fit a 4-topic model on the four-cluster corpus."""
    model = LDA(4, seed=seed, optimize_interval=0, num_threads=num_threads)
    model.fit(_CORPUS_4, iters=300, num_samples=3, sample_interval=10)
    return model


# ---------------------------------------------------------------------------
# Fixtures — cached per module so each model is only fitted once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_t1():
    """Sequential 2-topic model (num_threads=1)."""
    return _fit_2topic(num_threads=1)


@pytest.fixture(scope="module")
def model_t2():
    """Parallel 2-topic model (num_threads=2)."""
    return _fit_2topic(num_threads=2)


@pytest.fixture(scope="module")
def model_t4():
    """Parallel 2-topic model (num_threads=4)."""
    return _fit_2topic(num_threads=4)


@pytest.fixture(scope="module")
def model4_t1():
    """Sequential 4-topic model (for approximation-divergence tests)."""
    return _fit_4topic(num_threads=1)


@pytest.fixture(scope="module")
def model4_t2():
    """Parallel 4-topic model (num_threads=2)."""
    return _fit_4topic(num_threads=2)


# ---------------------------------------------------------------------------
# Constructor: num_threads clamping
# ---------------------------------------------------------------------------

class TestNumThreadsConstructor:
    def test_num_threads_zero_accepted(self):
        """num_threads=0 must be accepted without error (clamped to 1)."""
        model = LDA(2, num_threads=0)
        assert model.num_topics == 2

    def test_num_threads_one_accepted(self):
        """num_threads=1 is the default; must be accepted."""
        model = LDA(2, num_threads=1)
        assert model.num_topics == 2

    def test_num_threads_large_accepted(self):
        """num_threads=16 must be accepted without error."""
        model = LDA(2, num_threads=16)
        assert model.num_topics == 2

    def test_num_threads_zero_clamped_identical_to_one(self):
        """num_threads=0 is clamped to 1: results must equal num_threads=1 for same seed."""
        kw = dict(iters=200, num_samples=3, sample_interval=10)
        m0 = LDA(2, seed=7, optimize_interval=0, num_threads=0)
        m1 = LDA(2, seed=7, optimize_interval=0, num_threads=1)
        m0.fit(_CORPUS_2, **kw)
        m1.fit(_CORPUS_2, **kw)
        assert np.array_equal(m0.topic_word, m1.topic_word), (
            "num_threads=0 should be clamped to 1 and give identical results to num_threads=1"
        )


# ---------------------------------------------------------------------------
# Output shapes after parallel training
# ---------------------------------------------------------------------------

class TestParallelOutputShapes:
    def test_topic_word_shape_t2(self, model_t2):
        # 2 topics, 20 unique words (10 animal + 10 space)
        assert model_t2.topic_word.shape == (2, 20)

    def test_topic_word_shape_t4(self, model_t4):
        assert model_t4.topic_word.shape == (2, 20)

    def test_doc_topic_shape_t2(self, model_t2):
        assert model_t2.doc_topic.shape == (_N_DOCS_2, 2)

    def test_doc_topic_shape_t4(self, model_t4):
        assert model_t4.doc_topic.shape == (_N_DOCS_2, 2)

    def test_doc_topic_rows_sum_to_one_t2(self, model_t2):
        npt.assert_allclose(
            model_t2.doc_topic.sum(axis=1), np.ones(_N_DOCS_2), atol=1e-6
        )

    def test_doc_topic_rows_sum_to_one_t4(self, model_t4):
        npt.assert_allclose(
            model_t4.doc_topic.sum(axis=1), np.ones(_N_DOCS_2), atol=1e-6
        )

    def test_4topic_doc_topic_rows_sum_to_one(self, model4_t2):
        npt.assert_allclose(
            model4_t2.doc_topic.sum(axis=1), np.ones(_N_DOCS_4), atol=1e-6
        )


# ---------------------------------------------------------------------------
# Determinism: same (num_threads, seed) -> identical topic_word
# ---------------------------------------------------------------------------

class TestParallelDeterminism:
    def test_two_runs_t2_identical(self):
        """Two num_threads=2, seed=1 fits must produce identical topic_word."""
        m_a = _fit_2topic(num_threads=2, seed=1)
        m_b = _fit_2topic(num_threads=2, seed=1)
        assert np.array_equal(m_a.topic_word, m_b.topic_word), (
            "num_threads=2: repeated fit with same seed gave different topic_word "
            "(implementation must be deterministic for fixed threads+seed)"
        )

    def test_two_runs_t4_identical(self):
        """Two num_threads=4, seed=1 fits must produce identical topic_word."""
        m_a = _fit_2topic(num_threads=4, seed=1)
        m_b = _fit_2topic(num_threads=4, seed=1)
        assert np.array_equal(m_a.topic_word, m_b.topic_word), (
            "num_threads=4: repeated fit with same seed gave different topic_word"
        )

    def test_two_runs_t1_identical(self):
        """Sequential baseline: same seed -> identical topic_word."""
        m_a = _fit_2topic(num_threads=1, seed=1)
        m_b = _fit_2topic(num_threads=1, seed=1)
        assert np.array_equal(m_a.topic_word, m_b.topic_word)

    def test_different_seeds_t2_differ(self):
        """Different seeds with the same num_threads must give different results.

        Uses an over-specified K (5 topics for a 2-cluster corpus): the extra
        topics split seed-dependently, so the seed affects the fit. At K equal
        to the cluster count the sampler converges to the unique solution for
        any seed (correct, but it makes this assertion untestable).
        """
        m1 = _fit_2topic(num_threads=2, seed=1, num_topics=5)
        m2 = _fit_2topic(num_threads=2, seed=99, num_topics=5)
        assert not np.array_equal(m1.topic_word, m2.topic_word), (
            "Different seeds with num_threads=2 gave identical topic_word — unexpected"
        )

    def test_4topic_determinism_t2(self):
        """Determinism holds on the 4-topic corpus too."""
        m_a = _fit_4topic(num_threads=2, seed=3)
        m_b = _fit_4topic(num_threads=2, seed=3)
        assert np.array_equal(m_a.topic_word, m_b.topic_word)


# ---------------------------------------------------------------------------
# Approximation: parallel != sequential (verified on the 4-topic corpus)
#
# On a very simple two-topic corpus the parallel reconciliation can produce
# bit-identical results to the sequential path; a larger multi-topic corpus
# reliably shows the approximation diverging from the exact sequential path.
# ---------------------------------------------------------------------------

class TestParallelIsApproximation:
    def test_4topic_t2_differs_from_t1(self, model4_t1, model4_t2):
        """Parallel (num_threads=2) results must differ from sequential on a
        multi-topic corpus — the parallel path is an approximation."""
        assert not np.array_equal(model4_t1.topic_word, model4_t2.topic_word), (
            "num_threads=2 produced identical topic_word to num_threads=1 on a "
            "4-topic corpus — the parallel path does not appear to be active"
        )


# ---------------------------------------------------------------------------
# Topic quality: parallel model recovers the two-cluster structure
# ---------------------------------------------------------------------------

class TestParallelTopicQuality:
    def _get_topics(self, model):
        """Return (animal_topic_idx, space_topic_idx)."""
        vocab = model.vocabulary
        cat_idx = vocab.index("cat")
        planet_idx = vocab.index("planet")
        animal_t = int(model.topic_word[:, cat_idx].argmax())
        space_t = int(model.topic_word[:, planet_idx].argmax())
        return animal_t, space_t

    def test_t2_separates_clusters(self, model_t2):
        """num_threads=2 model must separate animal and space clusters."""
        animal_t, space_t = self._get_topics(model_t2)
        assert animal_t != space_t, (
            "num_threads=2 model failed to separate animal vs space topics"
        )

    def test_t4_separates_clusters(self, model_t4):
        """num_threads=4 model must separate animal and space clusters."""
        animal_t, space_t = self._get_topics(model_t4)
        assert animal_t != space_t, (
            "num_threads=4 model failed to separate animal vs space topics"
        )

    def test_t2_top_animal_words_disjoint_from_space_words(self, model_t2):
        """In the parallel-trained model the two dominant topics should have
        disjoint leading vocabularies (animal words vs space words)."""
        animal_t, space_t = self._get_topics(model_t2)
        top_animal = {w for w, _ in model_t2.top_words(5, topic=animal_t, weights=True)}
        top_space = {w for w, _ in model_t2.top_words(5, topic=space_t, weights=True)}
        assert top_animal.issubset(set(ANIMAL_WORDS)), (
            f"Top animal-topic words contain non-animal words: "
            f"{top_animal - set(ANIMAL_WORDS)}"
        )
        assert top_space.issubset(set(SPACE_WORDS)), (
            f"Top space-topic words contain non-space words: "
            f"{top_space - set(SPACE_WORDS)}"
        )


# ---------------------------------------------------------------------------
# Held-out perplexity from a parallel-trained model
# ---------------------------------------------------------------------------

class TestParallelPerplexity:
    """Perplexity from a parallel-trained model should be finite and positive.
    We do NOT assert it equals the sequential value — it is an approximation."""

    def _held_out(self):
        return [
            ["cat", "dog", "fish"],
            ["planet", "star", "moon"],
            ["rabbit", "turtle", "bird"],
            ["comet", "galaxy", "orbit"],
        ]

    def test_t2_perplexity_finite_positive(self, model_t2):
        ppl = model_t2.perplexity(self._held_out(), num_particles=5, seed=0)
        assert math.isfinite(ppl), f"Perplexity is not finite: {ppl}"
        assert ppl > 0, f"Perplexity must be positive, got {ppl}"

    def test_t4_perplexity_finite_positive(self, model_t4):
        ppl = model_t4.perplexity(self._held_out(), num_particles=5, seed=0)
        assert math.isfinite(ppl), f"Perplexity is not finite: {ppl}"
        assert ppl > 0, f"Perplexity must be positive, got {ppl}"

    def test_evaluate_returns_expected_keys(self, model_t2):
        result = model_t2.evaluate(self._held_out(), num_particles=5, seed=0)
        for key in ("log_likelihood", "perplexity", "num_tokens", "num_oov"):
            assert key in result, f"Missing key '{key}' in evaluate() result"


# ---------------------------------------------------------------------------
# Downstream API smoke tests on a parallel-trained model
# ---------------------------------------------------------------------------

class TestParallelDownstreamAPI:
    """All post-fit methods and properties must work on a parallel-trained model."""

    def test_top_words_all_topics(self, model_t2):
        result = model_t2.top_words(5)
        assert isinstance(result, list)
        assert len(result) == 2
        for topic_list in result:
            assert len(topic_list) == 5

    def test_top_words_single_topic(self, model_t2):
        result = model_t2.top_words(5, topic=0)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_diagnostics_returns_one_dict_per_topic(self, model_t2):
        diags = model_t2.diagnostics(n=5)
        assert isinstance(diags, list)
        assert len(diags) == 2
        for d in diags:
            for key in ("topic", "tokens", "coherence", "exclusivity",
                        "effective_words", "rank1_docs", "alpha", "top_words"):
                assert key in d, f"Missing key '{key}' in diagnostics dict"

    def test_coherence_shape(self, model_t2):
        c = model_t2.coherence(n=5)
        assert isinstance(c, np.ndarray)
        assert c.shape == (2,)

    def test_coherence_values_leq_zero(self, model_t2):
        c = model_t2.coherence(n=5)
        assert np.all(c <= 0), f"UMass coherence values should be <= 0, got {c}"

    def test_transform_shape(self, model_t2):
        new_docs = [["cat", "dog"], ["planet", "star"]]
        result = model_t2.transform(new_docs, seed=0)
        assert result.shape == (2, 2)
        npt.assert_allclose(result.sum(axis=1), np.ones(2), atol=1e-6)

    def test_top_documents_returns_list(self, model_t2):
        result = model_t2.top_documents(0, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        for name, weight in result:
            assert isinstance(name, str)
            assert 0 <= weight <= 1

    def test_topic_divergence_shape(self, model_t2):
        D = model_t2.topic_divergence
        assert D.shape == (2, 2)
        npt.assert_allclose(np.diag(D), np.zeros(2), atol=1e-10)

    def test_similar_documents_returns_list(self, model_t2):
        result = model_t2.similar_documents(0, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        for name, div in result:
            assert isinstance(name, str)
            assert 0 <= div <= 1

    def test_log_likelihood_finite(self, model_t2):
        ll = model_t2.log_likelihood()
        assert isinstance(ll, float)
        assert math.isfinite(ll)

    def test_vocabulary_length(self, model_t2):
        assert len(model_t2.vocabulary) == model_t2.topic_word.shape[1]

    def test_doc_names_length(self, model_t2):
        assert len(model_t2.doc_names) == model_t2.doc_topic.shape[0]

    def test_alpha_shape(self, model_t2):
        assert model_t2.alpha.shape == (2,)

    def test_beta_is_float(self, model_t2):
        assert isinstance(model_t2.beta, float)

    def test_save_topic_word(self, model_t2, tmp_path):
        path = tmp_path / "tw_parallel.tsv"
        model_t2.save_topic_word(str(path))
        assert path.exists()
        assert len(path.read_text()) > 0

    def test_save_doc_topic(self, model_t2, tmp_path):
        path = tmp_path / "dt_parallel.tsv"
        model_t2.save_doc_topic(str(path))
        assert path.exists()
        assert len(path.read_text()) > 0
