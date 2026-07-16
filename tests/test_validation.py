"""Social-science validation diagnostics: per-topic exclusivity and the
word/document intrusion tests (Chang et al. 2009, "Reading Tea Leaves").

These pair with per-topic coherence (``model.coherence(n)``) to support the
standard topic-quality workflow: the coherence-vs-exclusivity trade-off plot and
human intrusion validation.
"""

import numpy as np
import pytest

import topica


A = ["cat", "dog", "pet", "kitten", "puppy", "vet"]
B = ["star", "moon", "sky", "sun", "comet", "orbit"]


def _two_topic_model(seed=1, n=80):
    rng = np.random.default_rng(0)
    docs, is_a = [], []
    for _ in range(n):
        a = rng.random() < 0.5
        v = A if a else B
        docs.append([v[int(rng.integers(len(v)))] for _ in range(10)])
        is_a.append(a)
    m = topica.models.LDA(num_topics=2, seed=seed)
    m.fit(docs, iters=400)
    return m, docs, np.array(is_a)


class TestExclusivity:
    def test_shape_and_range(self):
        m, _, _ = _two_topic_model()
        ex = topica.diagnostics.exclusivity(m, n=5)
        assert ex.shape == (2,)
        # stm's exclusivity is a FREX summary over the top-n words (roughly [0, n]),
        # not a [0, 1] mean — assert non-negative and finite.
        assert np.all(ex >= 0.0) and np.all(np.isfinite(ex))

    def test_disjoint_vocab_is_highly_exclusive(self):
        # The two planted vocabularies don't overlap, so each topic's top words
        # are near-perfectly exclusive.
        m, _, _ = _two_topic_model()
        ex = topica.diagnostics.exclusivity(m, n=5)
        assert np.all(ex > 0.9)

    def test_accepts_array_or_model(self):
        m, _, _ = _two_topic_model()
        from_model = topica.diagnostics.exclusivity(m, n=5)
        from_array = topica.diagnostics.exclusivity(m.topic_word, n=5)
        np.testing.assert_allclose(from_model, from_array)

    def test_pairs_with_coherence(self):
        # The canonical quality plot needs one value per topic from each.
        m, _, _ = _two_topic_model()
        assert topica.diagnostics.exclusivity(m, n=10).shape == m.coherence(10).shape


class TestWordIntrusion:
    def test_structure_and_answer_key(self):
        m, _, _ = _two_topic_model()
        tests = topica.diagnostics.word_intrusion(m, n_words=4, seed=0)
        assert len(tests) == 2
        for r in tests:
            assert len(r["words"]) == 5  # n_words + 1 intruder
            # The answer key points at the intruder.
            assert r["words"][r["intruder_index"]] == r["intruder"]

    def test_intruder_comes_from_the_other_topic(self):
        # With two disjoint-vocab topics, the intruder must be the other block's
        # word — exactly what a human should be able to spot.
        m, _, _ = _two_topic_model()
        blocks = [set(A), set(B)]
        for r in topica.diagnostics.word_intrusion(m, n_words=4, seed=0):
            native = [w for w in r["words"] if w != r["intruder"]]
            home = 0 if sum(w in blocks[0] for w in native) >= len(native) / 2 else 1
            assert r["intruder"] in blocks[1 - home]

    def test_deterministic(self):
        m, _, _ = _two_topic_model()
        a = topica.diagnostics.word_intrusion(m, n_words=4, seed=7)
        b = topica.diagnostics.word_intrusion(m, n_words=4, seed=7)
        assert [x["words"] for x in a] == [x["words"] for x in b]

    def test_array_needs_vocabulary(self):
        m, _, _ = _two_topic_model()
        with pytest.raises(ValueError):
            topica.diagnostics.word_intrusion(m.topic_word)  # no vocabulary
        ok = topica.diagnostics.word_intrusion(m.topic_word, vocabulary=list(m.vocabulary))
        assert len(ok) == 2

    def test_needs_two_topics(self):
        phi = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError):
            topica.diagnostics.word_intrusion(phi, vocabulary=["a", "b"])


class TestDocumentIntrusion:
    def test_structure_and_answer_key(self):
        m, docs, _ = _two_topic_model()
        texts = [" ".join(d) for d in docs]
        tests = topica.diagnostics.document_intrusion(m, texts=texts, n_docs=3, seed=0)
        assert len(tests) == 2
        for r in tests:
            assert len(r["doc_indices"]) == 4
            assert len(r["texts"]) == 4
            assert 0 <= r["intruder_index"] < 4

    def test_intruder_has_low_topic_share(self):
        # The intruder document should load LESS on the topic than every genuine
        # member of the set.
        m, docs, _ = _two_topic_model()
        theta = m.doc_topic
        for r in topica.diagnostics.document_intrusion(m, n_docs=3, seed=0):
            t = r["topic"]
            intruder = r["doc_indices"][r["intruder_index"]]
            members = [d for d in r["doc_indices"] if d != intruder]
            assert all(theta[intruder, t] < theta[d, t] for d in members)

    def test_texts_optional(self):
        m, _, _ = _two_topic_model()
        r = topica.diagnostics.document_intrusion(m, n_docs=3, seed=0)
        assert "texts" not in r[0]

    def test_deterministic(self):
        m, _, _ = _two_topic_model()
        a = topica.diagnostics.document_intrusion(m, n_docs=3, seed=3)
        b = topica.diagnostics.document_intrusion(m, n_docs=3, seed=3)
        assert [x["doc_indices"] for x in a] == [x["doc_indices"] for x in b]
