"""CombinedTM and ZeroShotTM (Bianchi et al. 2021) on topica's hand-coded VAE
core. Both are ProdLDA with a different encoder input: CombinedTM concatenates
the bag of words with a caller-supplied document embedding; ZeroShotTM reads the
embedding alone. The decoder reconstructs the bag of words in both cases.

These tests cover construction, fitting with embeddings, the standard fitted
surface, top_words, save/load round-trip, transform, determinism, and input
validation. Topic-recovery quality is checked rigorously in the Rust unit tests
(planted-block recovery + finite-difference gradient checks)."""

import numpy as np
import pytest

import topica

MODELS = [topica.CombinedTM, topica.ZeroShotTM]
MODEL_IDS = ["CombinedTM", "ZeroShotTM"]


def _planted(k=3, block=8, n=180, length=15, seed=0):
    """K word-blocks; each document draws its tokens from one block, and its
    embedding is one-hot along its block's axis (plus noise) so the embedding
    encodes the planted topic structure. Returns (docs, embeddings, vocab)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    docs, embs = [], []
    for d in range(n):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(length)])
        e = np.zeros(k)
        e[b] = 3.0
        e = e + rng.normal(0.0, 0.1, k)
        embs.append(e)
    return docs, np.asarray(embs, dtype=np.float64), vocab


def _model(cls, num_topics=3, **kw):
    return cls(num_topics=num_topics, batch_size=60, lr=0.01, dropout=0.0, **kw)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_construction(cls):
    m = cls(4, alpha=0.5, hidden_size=64, dropout=0.1, batch_size=128, lr=0.003, seed=7)
    assert m.num_topics == 4
    assert "num_topics=4" in repr(m)
    assert "fitted=false" in repr(m)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_fit_surface(cls):
    docs, embs, vocab = _planted()
    m = _model(cls, seed=1)
    out = m.fit(docs, embs, iters=120)
    assert out is m  # fit() trains in place and returns self (#402)
    assert m.num_topics == 3
    assert m.topic_word.shape == (3, len(vocab))
    assert m.doc_topic.shape == (len(docs), 3)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert (m.topic_word >= 0).all()
    assert (m.doc_topic >= 0).all()
    assert isinstance(m.bound, float)
    assert len(m.bound_history) == m.epochs_run
    assert len(m.fit_history) == m.epochs_run
    assert set(m.vocabulary) == set(vocab)
    assert "fitted=true" in repr(m)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_fit_transform_matches_doc_topic(cls):
    docs, embs, _ = _planted()
    m = _model(cls, seed=1)
    theta = m.fit_transform(docs, embs, iters=80)
    assert theta.shape == (len(docs), 3)
    assert np.array_equal(theta, m.doc_topic)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_top_words(cls):
    docs, embs, _ = _planted()
    m = _model(cls, seed=1)
    m.fit(docs, embs, iters=80)
    per_topic = m.top_words(5)
    assert len(per_topic) == 3
    for words in per_topic:
        assert len(words) == 5
        for term, weight in words:
            assert isinstance(term, str)
            assert weight >= 0
    one = m.top_words(4, topic=0)
    assert len(one) == 4


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_save_load_roundtrip(cls, tmp_path):
    docs, embs, _ = _planted()
    m = _model(cls, seed=1)
    m.fit(docs, embs, iters=60)
    path = str(tmp_path / "model.bin")
    m.save(path)
    loaded = cls.load(path)
    assert loaded.num_topics == m.num_topics
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)
    # The loaded encoder transforms identically.
    a = m.transform(docs[:5], embs[:5])
    b = loaded.transform(docs[:5], embs[:5])
    assert np.array_equal(a, b)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_transform_shape(cls):
    docs, embs, _ = _planted()
    m = _model(cls, seed=1)
    m.fit(docs, embs, iters=60)
    theta = m.transform(docs[:10], embs[:10])
    assert theta.shape == (10, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_determinism(cls):
    docs, embs, _ = _planted()
    a = _model(cls, seed=3)
    a.fit(docs, embs, iters=60)
    b = _model(cls, seed=3)
    b.fit(docs, embs, iters=60)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_rejects_wrong_embedding_rows(cls):
    docs, embs, _ = _planted(n=60)
    m = _model(cls, seed=1)
    with pytest.raises(ValueError, match="doc_embeddings"):
        m.fit(docs, embs[:30], iters=10)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_rejects_k_below_two(cls):
    with pytest.raises(ValueError):
        cls(1)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_rejects_empty_corpus(cls):
    m = _model(cls, seed=1)
    with pytest.raises(ValueError):
        m.fit([], np.zeros((0, 3)), iters=10)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_transform_validates_embedding_rows(cls):
    docs, embs, _ = _planted(n=60)
    m = _model(cls, seed=1)
    m.fit(docs, embs, iters=20)
    with pytest.raises(ValueError, match="doc_embeddings"):
        m.transform(docs[:10], embs[:5])


def test_recovers_planted_blocks():
    """A light end-to-end recovery check (rigorous version lives in Rust). With
    embeddings that cleanly encode the block structure, each topic's top words
    should come predominantly from a single planted block."""
    docs, embs, vocab = _planted(n=240, seed=2)
    block = 8
    for cls in MODELS:
        m = _model(cls, seed=1)
        m.fit(docs, embs, iters=200)
        covered = set()
        for words in m.top_words(4):
            blocks = {term.split("w")[0] for term, _ in words}
            covered.add(tuple(sorted(blocks)))
        # Each topic concentrates on few blocks; together they touch all blocks.
        all_blocks = {term.split("w")[0] for words in m.top_words(4) for term, _ in words}
        assert len(all_blocks) == 3, f"{cls.__name__}: blocks covered {all_blocks}"


# --- VAE objective/prior flags (#174 contrastive, #176 prior) ----------------

@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_flags_accepted_and_exposed(cls):
    m = cls(num_topics=3, prior="dirichlet", contrastive=True,
            contrastive_weight=0.3, contrastive_temp=0.2)
    assert m.prior == "dirichlet"
    assert m.contrastive is True


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_flag_validation(cls):
    with pytest.raises(ValueError):
        cls(num_topics=3, prior="nope")
    with pytest.raises(ValueError):
        cls(num_topics=3, contrastive=True, contrastive_weight=-0.1)
    with pytest.raises(ValueError):
        cls(num_topics=3, contrastive=True, contrastive_temp=-1.0)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_flags_change_results(cls):
    docs, embs, _ = _planted(n=120)
    base = _model(cls, seed=1); base.fit(docs, embs, iters=60)
    dir_m = _model(cls, seed=1, prior="dirichlet"); dir_m.fit(docs, embs, iters=60)
    con_m = _model(cls, seed=1, contrastive=True); con_m.fit(docs, embs, iters=60)
    assert not np.allclose(base.topic_word, dir_m.topic_word)
    assert not np.allclose(base.topic_word, con_m.topic_word)
    for m in (dir_m, con_m):
        assert np.allclose(m.topic_word.sum(axis=1), 1.0)
        assert np.allclose(m.doc_topic.sum(axis=1), 1.0)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_flags_deterministic(cls):
    docs, embs, _ = _planted(n=120)
    a = _model(cls, seed=4, prior="dirichlet", contrastive=True); a.fit(docs, embs, iters=60)
    b = _model(cls, seed=4, prior="dirichlet", contrastive=True); b.fit(docs, embs, iters=60)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_flags_save_load_roundtrip(cls, tmp_path):
    docs, embs, _ = _planted(n=120)
    m = _model(cls, seed=2, prior="dirichlet", contrastive=True, contrastive_weight=0.4)
    m.fit(docs, embs, iters=60)
    path = str(tmp_path / "ctm_flags.bin")
    m.save(path)
    loaded = cls.load(path)
    assert loaded.prior == "dirichlet"
    assert loaded.contrastive is True
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)


@pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
def test_stick_breaking_valid_deterministic_roundtrip(cls, tmp_path):
    docs, embs, _ = _planted(n=120)
    base = _model(cls, seed=1); base.fit(docs, embs, iters=60)
    sb = _model(cls, seed=1, prior="stick_breaking"); sb.fit(docs, embs, iters=60)
    assert sb.prior == "stick_breaking"
    assert not np.allclose(base.topic_word, sb.topic_word)
    assert np.allclose(sb.topic_word.sum(axis=1), 1.0)
    assert np.allclose(sb.doc_topic.sum(axis=1), 1.0)
    # deterministic
    sb2 = _model(cls, seed=1, prior="stick_breaking"); sb2.fit(docs, embs, iters=60)
    assert np.array_equal(sb.topic_word, sb2.topic_word)
    # save/load
    path = str(tmp_path / "ctm_sb.bin")
    sb.save(path)
    loaded = cls.load(path)
    assert loaded.prior == "stick_breaking"
    assert np.array_equal(loaded.doc_topic, sb.doc_topic)
