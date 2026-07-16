"""Guided topic models: SeededLDA (seed-word priors) and KeyATM (keyword-assisted).
Both should steer the named topics onto their seed/keyword blocks while still
learning the rest of the vocabulary.
"""

import numpy as np
import pytest

import topica


def _blocky_corpus(seed=3, n=400, n_blocks=5, doc_len=15):
    rng = np.random.default_rng(seed)
    blocks = {f"B{i}": [f"b{i}w{j}" for j in range(8)] for i in range(n_blocks)}
    names = list(blocks)
    docs = []
    for _ in range(n):
        b = names[int(rng.integers(n_blocks))]
        docs.append([blocks[b][int(rng.integers(8))] for _ in range(doc_len)])
    return docs, blocks


# ---------------------------------------------------------------------------
# SeededLDA
# ---------------------------------------------------------------------------

class TestSeededLDA:
    def _seeds(self, blocks):
        return {"B0": blocks["B0"][:3], "B2": blocks["B2"][:3], "B4": blocks["B4"][:3]}

    def test_seeds_steer_topics(self):
        docs, blocks = _blocky_corpus()
        m = topica.models.SeededLDA(self._seeds(blocks), residual=2, seed=1)
        m.fit(docs, iters=500)
        assert m.num_topics == 5                       # 3 seeded + 2 residual
        assert m.topic_names[:3] == ["B0", "B2", "B4"]
        assert m.topic_names[3:] == ["residual_1", "residual_2"]
        for t, name in enumerate(["B0", "B2", "B4"]):
            top = {w for w, _ in m.top_words(5, topic=t)}
            assert len(top & set(blocks[name])) >= 4   # the seed steered the topic

    def test_shapes(self):
        docs, blocks = _blocky_corpus()
        m = topica.models.SeededLDA(self._seeds(blocks), residual=1, seed=1)
        m.fit(docs, iters=200)
        assert m.topic_word.shape == (4, len(m.vocabulary))
        assert m.doc_topic.shape == (len(docs), 4)
        np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)

    def test_deterministic_and_save_load(self, tmp_path):
        docs, blocks = _blocky_corpus()
        a = topica.models.SeededLDA(self._seeds(blocks), seed=1); a.fit(docs, iters=200)
        b = topica.models.SeededLDA(self._seeds(blocks), seed=1); b.fit(docs, iters=200)
        assert np.array_equal(a.topic_word, b.topic_word)
        p = str(tmp_path / "s.tt"); a.save(p)
        loaded = topica.models.SeededLDA.load(p)
        assert np.array_equal(a.topic_word, loaded.topic_word)
        assert loaded.topic_names == a.topic_names

    def test_bad_params(self):
        with pytest.raises(ValueError):
            topica.models.SeededLDA({})                            # no seeded topics
        with pytest.raises(ValueError):
            topica.models.SeededLDA({"a": ["x"]})                  # only 1 topic, no residual


# ---------------------------------------------------------------------------
# KeyATM
# ---------------------------------------------------------------------------

class TestKeyATM:
    def _keywords(self, blocks):
        return {"B0": blocks["B0"][:3], "B2": blocks["B2"][:3], "B4": blocks["B4"][:3]}

    def test_keywords_steer_topics(self):
        docs, blocks = _blocky_corpus()
        m = topica.models.KeyATM(self._keywords(blocks), num_topics=5, seed=1)
        m.fit(docs, iters=500)
        assert m.num_topics == 5
        for t, name in enumerate(["B0", "B2", "B4"]):
            top = {w for w, _ in m.top_words(5, topic=t)}
            assert len(top & set(blocks[name])) >= 4

    def test_keyword_rate(self):
        docs, blocks = _blocky_corpus()
        m = topica.models.KeyATM(self._keywords(blocks), num_topics=5, seed=1)
        m.fit(docs, iters=500)
        rate = m.keyword_rate
        assert rate.shape == (5,)
        assert np.all(rate >= 0) and np.all(rate <= 1)
        assert np.all(rate[:3] > 0.05)                 # keyword topics use keywords
        assert np.all(rate[3:] == 0.0)                 # regular topics do not

    def test_defaults_to_keyword_topic_count(self):
        docs, blocks = _blocky_corpus()
        m = topica.models.KeyATM(self._keywords(blocks), seed=1)   # num_topics omitted
        m.fit(docs, iters=200)
        assert m.num_topics == 3

    def test_shapes_and_save_load(self, tmp_path):
        docs, blocks = _blocky_corpus()
        m = topica.models.KeyATM(self._keywords(blocks), num_topics=4, seed=1)
        m.fit(docs, iters=200)
        assert m.topic_word.shape == (4, len(m.vocabulary))
        np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
        p = str(tmp_path / "k.tt"); m.save(p)
        loaded = topica.models.KeyATM.load(p)
        assert np.array_equal(m.topic_word, loaded.topic_word)
        assert np.array_equal(m.keyword_rate, loaded.keyword_rate)

    def test_bad_params(self):
        docs, blocks = _blocky_corpus()
        with pytest.raises(ValueError):
            topica.models.KeyATM(self._keywords(blocks), num_topics=2)  # fewer than keyword topics
