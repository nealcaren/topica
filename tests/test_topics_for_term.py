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

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_bad_top_n_raises(self, bad):
        with pytest.raises(ValueError, match="top_n"):
            topica.topics_for_term(PHI, "a", VOCAB, top_n=bad)


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
