"""save / load for the embedding-cluster models, and add_ngrams for bigram
topic words — the two pieces the BERTopic best-practices workflow needs."""

import warnings

import numpy as np
import pytest

import topica


def _planted(seed=0, per=60, k=3, dim=12):
    rng = np.random.default_rng(seed)
    themes = {
        "a": "policy reward agent rl env".split(),
        "b": "image conv segmentation pixel vision".split(),
        "c": "language token attention transformer embed".split(),
    }
    centres = rng.normal(0, 4, (k, dim))
    docs, emb = [], []
    for ci, words in enumerate(list(themes.values())[:k]):
        for _ in range(per):
            docs.append(list(rng.choice(words, size=14)))
            emb.append(centres[ci] + rng.normal(0, 0.5, dim))
    vocab = sorted({w for ws in themes.values() for w in ws})
    return docs, np.array(emb), vocab


@pytest.mark.parametrize("model_cls", ["BERTopic", "Top2Vec"])
def test_save_load_round_trip(model_cls, tmp_path):
    docs, emb, vocab = _planted()
    cls = getattr(topica.models, model_cls)
    m = cls(min_cluster_size=20, seed=1)
    if model_cls == "Top2Vec":
        word_emb = np.random.default_rng(1).normal(0, 1, (len(vocab), emb.shape[1]))
        m.fit(docs, emb, word_embeddings=word_emb, vocabulary=vocab)
    else:
        m.fit(docs, emb)

    p = str(tmp_path / "m.tt")
    m.save(p)
    loaded = cls.load(p)

    assert loaded.num_topics == m.num_topics
    assert np.allclose(loaded.topic_word, m.topic_word)
    assert np.allclose(loaded.doc_topic, m.doc_topic)
    assert list(loaded.labels) == list(m.labels)
    assert list(loaded.vocabulary) == list(m.vocabulary)
    # the loaded model still predicts (the BERTopic save -> load -> infer pattern)
    if model_cls == "Top2Vec":
        out = loaded.transform(docs[:5], emb[:5])
    else:
        out = loaded.transform(docs[:5])
    assert out.shape == (5, m.num_topics)


def test_save_requires_fitted(tmp_path):
    with pytest.raises(Exception):
        topica.models.BERTopic(min_cluster_size=5).save(str(tmp_path / "x.tt"))


def test_add_ngrams_basic():
    docs = [["machine", "learning", "model"], ["deep", "learning", "model"]]
    out = topica.add_ngrams(docs, ngram_range=(1, 2))
    assert "machine_learning" in out[0] and "learning_model" in out[0]
    assert "machine" in out[0]  # unigrams kept
    # bigrams only
    only = topica.add_ngrams(docs, ngram_range=(2, 2))
    assert all("_" in t for t in only[0])
    # min_df prunes rare terms, document count preserved
    pruned = topica.add_ngrams(docs, ngram_range=(1, 2), min_df=2)
    assert len(pruned) == len(docs)
    kept = {t for d in pruned for t in d}
    assert "learning" in kept and "machine_learning" not in kept  # df 1 dropped


def test_add_ngrams_rejects_bad_range():
    with pytest.raises(ValueError, match="ngram_range"):
        topica.add_ngrams([["a", "b"]], ngram_range=(2, 1))


def test_bigrams_flow_into_bertopic_topic_words():
    rng = np.random.default_rng(0)
    themes = {"ml": ["machine", "learning", "neural", "network"],
              "bio": ["gene", "protein", "cell", "dna"]}
    base, emb = [], []
    for ci, words in enumerate(themes.values()):
        for _ in range(60):
            base.append(list(rng.choice(words, size=10)))
            emb.append(rng.normal([ci * 6, 0], 0.5, 2))
    ng = topica.add_ngrams(base, ngram_range=(1, 2), min_df=3)
    assert len(ng) == len(base)  # alignment with embeddings preserved
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.models.BERTopic(min_cluster_size=20, seed=1)
        m.fit(ng, np.array(emb))
    words = {w for t in range(m.num_topics) for w, _ in m.top_words(8, topic=t)}
    assert any("_" in w for w in words)
