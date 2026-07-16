"""Tests for the LabeledLDA class (supervised Labeled LDA, Ramage et al. 2009)."""

import numpy as np
import numpy.testing as npt
import pytest

from topica.models import LabeledLDA
from topica import Corpus


# ---------------------------------------------------------------------------
# Synthetic corpus builder
# ---------------------------------------------------------------------------

# Three disjoint label vocabularies (6 words each, no overlap)
_VOCAB = {
    "sports":   ["football", "basketball", "soccer", "baseball", "tennis", "hockey"],
    "politics": ["election", "senate", "congress", "president", "democrat", "republican"],
    "tech":     ["computer", "software", "internet", "algorithm", "database", "network"],
}
_ALL_LABELS = sorted(_VOCAB.keys())   # ['politics', 'sports', 'tech']


def _make_corpus(n=150, words_per_label=8, seed=1):
    """Return (docs, labels) for the 3-label recovery corpus.

    Each document gets 1 or 2 randomly chosen labels; its words are drawn
    exclusively from those labels' vocabularies, so clean recovery is expected.
    """
    rng = np.random.default_rng(seed)
    docs = []
    labels = []
    for _ in range(n):
        n_labels = int(rng.integers(1, 3))   # 1 or 2 labels per doc
        chosen = rng.choice(_ALL_LABELS, size=n_labels, replace=False).tolist()
        labels.append(chosen)
        words = []
        for lbl in chosen:
            words.extend(rng.choice(_VOCAB[lbl], size=words_per_label).tolist())
        docs.append(words)
    return docs, labels


def _fit_recovery(docs, labels, seed=1, iters=300):
    """Fit LabeledLDA with settings known to recover the 3-label corpus cleanly."""
    model = LabeledLDA(alpha=0.1, seed=seed)
    model.fit(
        docs,
        labels,
        iters=iters,
        num_samples=3,
        sample_interval=10,
    )
    return model


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestLabeledLDAConstructor:
    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            LabeledLDA(alpha=0.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError):
            LabeledLDA(alpha=-0.1)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            LabeledLDA(beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises(ValueError):
            LabeledLDA(beta=-0.01)

    def test_valid_constructor_does_not_raise(self):
        # Just construction; no fit needed
        LabeledLDA(alpha=0.1, beta=0.01, seed=42)


# ---------------------------------------------------------------------------
# Unfitted guards — all must raise RuntimeError before fit()
# ---------------------------------------------------------------------------

class TestLabeledLDAUnfittedGuards:
    PROPERTIES = [
        "topic_word",
        "doc_topic",
        "labels",
        "vocabulary",
        "doc_names",
        "num_topics",
    ]

    @pytest.mark.parametrize("prop", PROPERTIES)
    def test_property_raises_before_fit(self, prop):
        model = LabeledLDA()
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(model, prop)

    def test_top_words_raises_before_fit(self):
        model = LabeledLDA()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.top_words()

    def test_coherence_raises_before_fit(self):
        model = LabeledLDA()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.coherence()


# ---------------------------------------------------------------------------
# fit() validation errors
# ---------------------------------------------------------------------------

class TestLabeledLDAFitValidation:
    def setup_method(self):
        self.docs = [["football", "basketball"]] * 10
        self.labels = [["sports"]] * 10

    def test_labels_length_mismatch_raises(self):
        bad_labels = self.labels[:-1]   # 9 labels for 10 docs
        with pytest.raises(ValueError):
            model = LabeledLDA()
            model.fit(self.docs, bad_labels)

    def test_all_empty_labels_without_label_names_raises(self):
        """All-empty label lists with no label_names → no topics → ValueError."""
        empty_labels = [[] for _ in self.docs]
        with pytest.raises(ValueError):
            model = LabeledLDA()
            model.fit(self.docs, empty_labels)

    def test_empty_label_names_list_raises(self):
        """Explicitly passing label_names=[] should also raise (no topics)."""
        with pytest.raises(ValueError):
            model = LabeledLDA()
            model.fit(self.docs, self.labels, label_names=[])


# ---------------------------------------------------------------------------
# Supervised recovery: shapes, constraint, and word recovery
# ---------------------------------------------------------------------------

class TestLabeledLDARecovery:
    """Core validation against the 3-label synthetic corpus."""

    @pytest.fixture(scope="class")
    def fitted(self):
        docs, labels = _make_corpus(n=150, seed=1)
        return _fit_recovery(docs, labels, seed=1), labels

    # --- num_topics and labels property ---

    def test_num_topics_equals_3(self, fitted):
        model, _ = fitted
        assert model.num_topics == 3

    def test_labels_property_is_sorted_union(self, fitted):
        model, _ = fitted
        assert model.labels == _ALL_LABELS   # ['politics', 'sports', 'tech']

    # --- Shapes ---

    def test_topic_word_shape(self, fitted):
        model, _ = fitted
        assert model.topic_word.shape == (3, len(model.vocabulary))

    def test_doc_topic_shape(self, fitted):
        model, _ = fitted
        assert model.doc_topic.shape == (150, 3)

    def test_doc_topic_rows_sum_to_one(self, fitted):
        model, _ = fitted
        npt.assert_allclose(model.doc_topic.sum(axis=1), np.ones(150), atol=1e-6)

    # --- Word recovery: each topic's top-4 dominated by its own vocabulary ---

    def test_top_words_dominated_by_label_vocabulary(self, fitted):
        """For each topic, at least 3 of the top-4 words must be from that label's vocab."""
        model, _ = fitted
        label_to_idx = {lbl: i for i, lbl in enumerate(model.labels)}
        for lbl, vocab_words in _VOCAB.items():
            t = label_to_idx[lbl]
            top4 = [w for w, _ in model.top_words(4, topic=t)]
            in_vocab = sum(1 for w in top4 if w in vocab_words)
            assert in_vocab >= 3, (
                f"Topic '{lbl}' (idx {t}): only {in_vocab}/4 top words are in "
                f"its vocabulary. top4={top4}, expected words from {vocab_words}"
            )

    # --- Supervised constraint: non-label topics are zero ---

    def test_supervised_constraint_nonlabel_topics_zero(self, fitted):
        """For every document, doc_topic must be zero for every topic NOT in its labels."""
        model, labels = fitted
        dt = model.doc_topic
        label_to_idx = {lbl: i for i, lbl in enumerate(model.labels)}
        for d, doc_labels in enumerate(labels):
            allowed = {label_to_idx[l] for l in doc_labels}
            for t in range(model.num_topics):
                if t not in allowed:
                    assert dt[d, t] < 1e-9, (
                        f"Supervised constraint violated: doc {d} has label set "
                        f"{doc_labels!r} but doc_topic[{d},{t}]={dt[d,t]:.2e} "
                        f"(topic '{model.labels[t]}' not in labels)"
                    )


# ---------------------------------------------------------------------------
# label_names: fixes topic order
# ---------------------------------------------------------------------------

class TestLabeledLDALabelNames:
    def test_label_names_fixes_topic_order(self):
        """Passing label_names overrides the sorted-union ordering."""
        docs, labels = _make_corpus(n=60, seed=2)
        model = LabeledLDA(alpha=0.1, seed=1)
        model.fit(
            docs,
            labels,
            label_names=["tech", "sports", "politics"],
            iters=100,
            num_samples=2,
            sample_interval=5,
        )
        assert model.labels == ["tech", "sports", "politics"]

    def test_label_names_topic0_matches_first_name(self):
        """topic 0 should correspond to the first label_names entry."""
        docs, labels = _make_corpus(n=60, seed=2)
        model = LabeledLDA(alpha=0.1, seed=1)
        model.fit(
            docs,
            labels,
            label_names=["tech", "sports", "politics"],
            iters=100,
            num_samples=2,
            sample_interval=5,
        )
        assert model.labels[0] == "tech"


# ---------------------------------------------------------------------------
# Empty label list — unconstrained document
# ---------------------------------------------------------------------------

class TestLabeledLDAEmptyLabels:
    def test_unconstrained_doc_row_sums_to_one(self):
        """A document with an empty label list should still sum to 1."""
        docs, labels = _make_corpus(n=30, seed=3)
        # Append a doc with [] labels (unconstrained)
        extra_doc = ["football", "election", "computer"]
        all_docs = docs + [extra_doc]
        all_labels = labels + [[]]   # empty = unconstrained

        model = LabeledLDA(alpha=0.1, seed=1)
        model.fit(
            all_docs,
            all_labels,
            label_names=_ALL_LABELS,
            iters=100,
            num_samples=2,
            sample_interval=5,
        )
        row_sum = model.doc_topic[-1].sum()
        npt.assert_allclose(row_sum, 1.0, atol=1e-6)

    def test_unconstrained_doc_may_have_any_topic_nonzero(self):
        """Unconstrained doc is not restricted; at least fit completes without error."""
        docs, labels = _make_corpus(n=30, seed=3)
        all_docs = docs + [["football", "election", "computer"]]
        all_labels = labels + [[]]

        model = LabeledLDA(alpha=0.1, seed=1)
        model.fit(
            all_docs,
            all_labels,
            label_names=_ALL_LABELS,
            iters=100,
            num_samples=2,
            sample_interval=5,
        )
        # Model fitted without error; unconstrained row sums to 1
        npt.assert_allclose(model.doc_topic[-1].sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism: same seed → identical arrays
# ---------------------------------------------------------------------------

class TestLabeledLDADeterminism:
    def test_identical_seed_identical_topic_word(self):
        docs, labels = _make_corpus(n=60, seed=4)
        m1 = LabeledLDA(alpha=0.1, seed=7)
        m1.fit(docs, labels, iters=100, num_samples=2, sample_interval=5)
        m2 = LabeledLDA(alpha=0.1, seed=7)
        m2.fit(docs, labels, iters=100, num_samples=2, sample_interval=5)
        assert np.array_equal(m1.topic_word, m2.topic_word)

    def test_identical_seed_identical_doc_topic(self):
        docs, labels = _make_corpus(n=60, seed=4)
        m1 = LabeledLDA(alpha=0.1, seed=7)
        m1.fit(docs, labels, iters=100, num_samples=2, sample_interval=5)
        m2 = LabeledLDA(alpha=0.1, seed=7)
        m2.fit(docs, labels, iters=100, num_samples=2, sample_interval=5)
        assert np.array_equal(m1.doc_topic, m2.doc_topic)


# ---------------------------------------------------------------------------
# Input type parity: list[list[str]] vs Corpus
# ---------------------------------------------------------------------------

class TestLabeledLDAInputTypeParity:
    def test_corpus_vs_token_list_same_results(self):
        """Corpus input and list[list[str]] input should give identical results."""
        docs, labels = _make_corpus(n=40, seed=5)
        corpus = Corpus.from_documents(docs)

        m1 = LabeledLDA(alpha=0.1, seed=99)
        m1.fit(docs, labels, iters=100, num_samples=2, sample_interval=5)

        m2 = LabeledLDA(alpha=0.1, seed=99)
        m2.fit(corpus, labels, iters=100, num_samples=2, sample_interval=5)

        assert np.array_equal(m1.topic_word, m2.topic_word)
        assert np.array_equal(m1.doc_topic, m2.doc_topic)

    def test_labels_as_list_of_lists(self):
        """labels parameter is accepted as a plain list[list[str]]."""
        docs, labels = _make_corpus(n=30, seed=6)
        assert isinstance(labels, list)
        assert all(isinstance(l, list) for l in labels)
        model = LabeledLDA(alpha=0.1, seed=1)
        model.fit(docs, labels, iters=50, num_samples=1, sample_interval=5)
        assert model.num_topics == 3


# ---------------------------------------------------------------------------
# top_words and coherence
# ---------------------------------------------------------------------------

class TestLabeledLDATopWordsAndCoherence:
    @pytest.fixture(scope="class")
    def fitted(self):
        docs, labels = _make_corpus(n=60, seed=0)
        model = LabeledLDA(alpha=0.1, seed=1)
        model.fit(docs, labels, iters=200, num_samples=2, sample_interval=10)
        return model

    def test_top_words_all_topics_structure(self, fitted):
        result = fitted.top_words(5)
        assert isinstance(result, list)
        assert len(result) == 3   # one list per topic
        for topic_list in result:
            assert len(topic_list) == 5
            for word, prob in topic_list:
                assert isinstance(word, str)
                assert isinstance(prob, float)

    def test_top_words_single_topic_structure(self, fitted):
        result = fitted.top_words(5, topic=0)
        assert isinstance(result, list)
        assert len(result) == 5
        for word, prob in result:
            assert isinstance(word, str)
            assert isinstance(prob, float)

    def test_top_words_probabilities_descending(self, fitted):
        for topic_list in fitted.top_words(6):
            probs = [p for _, p in topic_list]
            assert probs == sorted(probs, reverse=True)

    def test_top_words_out_of_range_raises(self, fitted):
        with pytest.raises(ValueError):
            fitted.top_words(5, topic=10)

    def test_coherence_shape(self, fitted):
        c = fitted.coherence(n=5)
        assert c.shape == (3,)

    def test_coherence_values_nonpositive(self, fitted):
        c = fitted.coherence(n=5)
        assert (c <= 0).all()


# ---------------------------------------------------------------------------
# CVB0 backend (deterministic; the masked-γ inference warp could not do well)
# ---------------------------------------------------------------------------

class TestLabeledCvb0:
    def _fit(self, seed=1, iters=300):
        docs, labels = _make_corpus(seed=2)
        m = LabeledLDA(alpha=0.1, seed=seed, sampler="cvb0")
        m.fit(docs, labels, iters=iters)
        return m, labels

    def test_recovers_labels(self):
        m, _ = self._fit()
        label_to_idx = {lbl: i for i, lbl in enumerate(m.labels)}
        for lbl, vocab_words in _VOCAB.items():
            t = label_to_idx[lbl]
            top4 = [w for w, _ in m.top_words(4, topic=t)]
            assert sum(1 for w in top4 if w in vocab_words) >= 3

    def test_mask_respected_exactly(self):
        # The supervised constraint must hold under CVB0: zero θ off the label set.
        m, labels = self._fit()
        dt = m.doc_topic
        label_to_idx = {lbl: i for i, lbl in enumerate(m.labels)}
        for d, doc_labels in enumerate(labels):
            allowed = {label_to_idx[l] for l in doc_labels}
            for t in range(m.num_topics):
                if t not in allowed:
                    assert dt[d, t] < 1e-12

    def test_valid_distributions_and_no_draws(self):
        m, _ = self._fit()
        npt.assert_allclose(m.topic_word.sum(axis=1), 1.0)
        npt.assert_allclose(m.doc_topic.sum(axis=1), 1.0)
        assert m.theta_draws is None

    def test_deterministic(self):
        a, _ = self._fit(seed=4, iters=120)
        b, _ = self._fit(seed=4, iters=120)
        npt.assert_array_equal(a.topic_word, b.topic_word)

    def test_aliases_and_bad_name(self):
        docs, labels = _make_corpus(seed=2)
        for name in ("cvb0", "cvb"):
            m = LabeledLDA(seed=1, sampler=name)
            m.fit(docs, labels, iters=30)
            assert m.num_topics == 3
        with pytest.raises(ValueError):
            LabeledLDA(sampler="banana")
