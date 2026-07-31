"""Inverse lookup: ``topics_for_term`` — which topics rank a term highest (#595).

The forward path (``label_topics`` / ``frex``) goes topic → words; this goes
word → topics over the same φ matrix. Most cases use an explicit φ + vocabulary
so the expected ranking is exact and needs no model fit; one case checks the
fitted-model path and array/model parity.
"""

import numpy as np
import pytest

import topica


VOCAB = ["a", "b", "c", "d"]

# topic 0 loves "a", topic 1 loves "b", topic 2 is mixed. Column "d" ties across
# all three topics, which pins down the tie-break rule.
PHI = np.array(
    [
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.1],
    ]
)


class TestSingleTerm:
    def test_ranks_topics_by_column_weight(self):
        out = topica.topics_for_term(PHI, "a", VOCAB)
        assert [t for t, _ in out] == [0, 2, 1]  # 0.7, 0.2, 0.1
        assert out[0] == (0, pytest.approx(0.7))

    def test_returns_plain_int_and_float(self):
        (topic, weight), *_ = topica.topics_for_term(PHI, "b", VOCAB)
        assert type(topic) is int and type(weight) is float
        assert topic == 1

    def test_top_n_limits_length(self):
        assert len(topica.topics_for_term(PHI, "a", VOCAB, top_n=2)) == 2

    def test_top_n_none_returns_all_topics(self):
        out = topica.topics_for_term(PHI, "a", VOCAB, top_n=None)
        assert len(out) == PHI.shape[0]

    def test_top_n_larger_than_k_is_clamped(self):
        out = topica.topics_for_term(PHI, "a", VOCAB, top_n=99)
        assert len(out) == PHI.shape[0]

    def test_tie_break_is_ascending_topic_id(self):
        # Column "d" is uniform across topics; ties resolve to 0, 1, 2.
        assert [t for t, _ in topica.topics_for_term(PHI, "d", VOCAB)] == [0, 1, 2]


class TestMultipleTerms:
    def test_pooled_ranking_sums_columns(self):
        # a+b weights per topic: [0.8, 0.7, 0.5] -> 0, 1, 2.
        out = topica.topics_for_term(PHI, ["a", "b"], VOCAB)
        assert [t for t, _ in out] == [0, 1, 2]
        assert out[0][1] == pytest.approx(0.8)

    def test_per_term_returns_dict_keyed_by_term(self):
        out = topica.topics_for_term(PHI, ["a", "b"], VOCAB, per_term=True)
        assert set(out) == {"a", "b"}
        assert [t for t, _ in out["a"]] == [0, 2, 1]
        assert [t for t, _ in out["b"]] == [1, 2, 0]

    def test_single_element_list_still_pools_to_a_list(self):
        # A one-element list is not a bare string, so it takes the pooled path
        # and returns a list (not a dict).
        out = topica.topics_for_term(PHI, ["a"], VOCAB)
        assert isinstance(out, list)
        assert [t for t, _ in out] == [0, 2, 1]

    def test_duplicate_terms_are_collapsed_pooled(self):
        # A repeated term must not double-count and flip the pooled ranking; the
        # result matches querying each unique term once.
        dup = topica.topics_for_term(PHI, ["a", "a", "b"], VOCAB)
        uniq = topica.topics_for_term(PHI, ["a", "b"], VOCAB)
        assert dup == uniq

    def test_duplicate_terms_collapsed_per_term(self):
        # per_term must not silently drop a requested term to dict-key uniqueness;
        # ["a", "a"] collapses to a single "a" entry, same as ["a"].
        out = topica.topics_for_term(PHI, ["a", "a"], VOCAB, per_term=True)
        assert set(out) == {"a"}

    def test_numpy_array_terms_give_plain_str_keys(self):
        out = topica.topics_for_term(PHI, np.array(["a", "b"]), VOCAB, per_term=True)
        assert all(type(k) is str for k in out)


class TestNormalize:
    def test_single_term_is_conditional_topic_distribution(self):
        # normalize=True returns P(topic | word) = column / column-sum.
        col = PHI[:, VOCAB.index("a")]
        expected = col / col.sum()
        out = topica.topics_for_term(PHI, "a", VOCAB, top_n=None, normalize=True)
        got = {t: w for t, w in out}
        for t in range(PHI.shape[0]):
            assert got[t] == pytest.approx(expected[t])
        assert sum(w for _, w in out) == pytest.approx(1.0)

    def test_pooling_weights_terms_equally(self):
        # Raw pooling is dominated by the higher-mass term; normalized pooling
        # adds two per-word distributions, so each term contributes total mass 1.
        out = topica.topics_for_term(PHI, ["a", "b"], VOCAB, top_n=None, normalize=True)
        assert sum(w for _, w in out) == pytest.approx(2.0)

    def test_all_zero_column_does_not_divide_by_zero(self):
        phi = np.array([[0.0, 1.0], [0.0, 1.0]])
        out = topica.topics_for_term(phi, "x", ["x", "y"], normalize=True)
        assert [w for _, w in out] == [0.0, 0.0]


class TestWithLabels:
    def test_single_term_returns_triples_with_top_words(self):
        out = topica.topics_for_term(PHI, "a", VOCAB, top_n=2, with_labels=True)
        assert len(out[0]) == 3
        topic, weight, words = out[0]
        assert topic == 0 and weight == pytest.approx(0.7)
        # topic 0's highest-prob words, in order: a (0.7) then b/c/d (all 0.1,
        # tie-break ascending index -> b, c, ...).
        assert words[0] == "a"
        assert all(isinstance(w, str) for w in words)

    def test_label_n_controls_word_count(self):
        out = topica.topics_for_term(PHI, "a", VOCAB, with_labels=True, label_n=2)
        assert all(len(words) == 2 for _, _, words in out)

    def test_labels_match_phi_top_words(self):
        out = topica.topics_for_term(PHI, "a", VOCAB, top_n=None, with_labels=True)
        for topic, _, words in out:
            expected = [VOCAB[i] for i in np.argsort(-PHI[topic], kind="stable")[:5]]
            assert words == expected

    def test_per_term_carries_labels(self):
        out = topica.topics_for_term(PHI, ["a", "b"], VOCAB, per_term=True,
                                     with_labels=True, label_n=1)
        assert all(len(entry) == 3 for lst in out.values() for entry in lst)

    def test_default_still_returns_pairs(self):
        out = topica.topics_for_term(PHI, "a", VOCAB)
        assert all(len(entry) == 2 for entry in out)

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_bad_label_n_raises(self, bad):
        with pytest.raises(ValueError, match="label_n"):
            topica.topics_for_term(PHI, "a", VOCAB, with_labels=True, label_n=bad)


class TestVocabularyGuard:
    def test_vocab_longer_than_phi_raises_clearly(self):
        with pytest.raises(ValueError, match="does not match"):
            topica.topics_for_term(PHI, "a", VOCAB + ["extra"])

    def test_vocab_shorter_than_phi_raises_clearly(self):
        with pytest.raises(ValueError, match="does not match"):
            topica.topics_for_term(PHI, "a", VOCAB[:-1])


class TestMissingTerms:
    def test_single_missing_term_raises(self):
        with pytest.raises(ValueError, match="none of the requested terms"):
            topica.topics_for_term(PHI, "zzz", VOCAB)

    def test_all_missing_raises(self):
        with pytest.raises(ValueError, match="none of the requested terms"):
            topica.topics_for_term(PHI, ["zzz", "qqq"], VOCAB)

    def test_partial_missing_warns_and_drops(self):
        with pytest.warns(UserWarning, match="not in the vocabulary"):
            out = topica.topics_for_term(PHI, ["a", "zzz"], VOCAB, per_term=True)
        assert set(out) == {"a"}


class TestValidation:
    def test_empty_terms_raises(self):
        with pytest.raises(ValueError, match="empty"):
            topica.topics_for_term(PHI, [], VOCAB)

    def test_non_string_term_raises(self):
        with pytest.raises(ValueError, match="string"):
            topica.topics_for_term(PHI, [1, 2], VOCAB)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, False])
    def test_bad_top_n_raises(self, bad):
        # True/False are ints in Python; they must be rejected, not read as 1/0.
        with pytest.raises(ValueError, match="top_n"):
            topica.topics_for_term(PHI, "a", VOCAB, top_n=bad)

    def test_numpy_integer_top_n_is_accepted(self):
        assert len(topica.topics_for_term(PHI, "a", VOCAB, top_n=np.int64(2))) == 2


class TestModelPath:
    def _two_topic_model(self):
        rng = np.random.default_rng(0)
        animals = ["cat", "dog", "pet", "kitten", "puppy", "vet"]
        space = ["star", "moon", "sky", "sun", "comet", "orbit"]
        docs = []
        for _ in range(80):
            v = animals if rng.random() < 0.5 else space
            docs.append([v[int(rng.integers(len(v)))] for _ in range(10)])
        m = topica.LDA(num_topics=2, seed=1)
        m.fit(docs, iters=400)
        return m

    def test_infers_vocabulary_from_model(self):
        m = self._two_topic_model()
        out = topica.topics_for_term(m, "cat")
        assert len(out) == 2 and out[0][0] in (0, 1)

    def test_model_and_array_agree(self):
        m = self._two_topic_model()
        from_model = topica.topics_for_term(m, "cat")
        from_array = topica.topics_for_term(
            np.asarray(m.topic_word), "cat", list(m.vocabulary)
        )
        assert from_model == from_array

    def test_dominant_topic_for_planted_word(self):
        # "cat" belongs to the animals topic; that topic should carry the most
        # weight on it, so it ranks first.
        m = self._two_topic_model()
        phi = np.asarray(m.topic_word)
        j = list(m.vocabulary).index("cat")
        expected = int(np.argmax(phi[:, j]))
        assert topica.topics_for_term(m, "cat")[0][0] == expected
