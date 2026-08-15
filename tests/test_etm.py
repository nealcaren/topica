"""ETM (Embedded Topic Model) on topica's variational-EM core: it recovers
planted word blocks from embeddings, exposes the standard fitted surface, and
validates its inputs."""

import numpy as np
import pytest

import topica


def _planted(k=3, block=8, e=3, seed=0):
    """K word-blocks; each word's embedding points along its block's axis, each
    document draws from one block. Returns (docs, vocab, word_emb, truth)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    word_emb = np.zeros((k * block, e))
    for w in range(k * block):
        word_emb[w, w // block] = 3.0
        word_emb[w] += rng.normal(0, 0.2, e)
    docs, truth = [], []
    for d in range(120):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(10)])
        truth.append(b)
    return docs, vocab, word_emb, np.array(truth)


def test_etm_recovers_planted_blocks():
    docs, vocab, word_emb, truth = _planted()
    m = topica.ETM(num_topics=3, seed=1)
    m.fit(docs, word_emb, vocab, iters=50)

    assert m.topic_word.shape == (3, len(vocab))
    assert m.doc_topic.shape == (len(docs), 3)
    assert m.topic_embeddings.shape == (3, word_emb.shape[1])
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.converged or m.bound < 0  # a real bound was tracked

    # Each topic's top words come from one block, covering all blocks.
    covered = set()
    for t in range(3):
        blocks = {w.split("w")[0] for w, _ in m.top_words(4, topic=t, weights=True)}
        assert len(blocks) == 1, f"topic {t} mixes blocks: {blocks}"
        covered |= blocks
    assert len(covered) == 3


def test_etm_top_words_all_topics_and_names():
    docs, vocab, word_emb, _ = _planted()
    m = topica.ETM(num_topics=3, seed=1)
    m.fit(docs, word_emb, vocab, iters=20)
    allw = m.top_words(5)  # topic=None -> list per topic
    assert len(allw) == 3 and all(len(t) == 5 for t in allw)
    assert m.topic_names == ["topic_0", "topic_1", "topic_2"]
    assert list(m.vocabulary) == vocab


def test_etm_validation():
    docs, vocab, word_emb, _ = _planted()
    with pytest.raises(ValueError):
        topica.ETM(num_topics=1)  # need >= 2 topics
    with pytest.raises(ValueError):
        topica.ETM(num_topics=3, inference="bogus")  # em or vae only
    m = topica.ETM(num_topics=3, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, word_emb[:-1], vocab)  # embeddings rows != vocab length
    with pytest.raises(RuntimeError):
        topica.ETM(num_topics=3).topic_word  # not fitted


# --- VAE inference path -----------------------------------------------------


def _vae(num_topics=3, **kw):
    return topica.ETM(
        num_topics=num_topics, inference="vae", hidden_size=64,
        batch_size=64, lr=0.01, **kw
    )


def test_etm_vae_recovers_planted_blocks():
    docs, vocab, word_emb, _ = _planted()
    m = _vae(seed=1)
    theta = m.fit_transform(docs, word_emb, vocab, iters=200)
    assert m.inference == "vae"
    assert m.topic_word.shape == (3, len(vocab))
    assert theta.shape == (len(docs), 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    covered = set()
    for t in range(3):
        words = [w for w, _ in m.top_words(4, topic=t, weights=True)]
        blocks = {w.split("w")[0] for w in words}
        assert len(blocks) == 1, f"topic {t} mixes blocks: {words}"
        covered |= blocks
    assert len(covered) == 3


def test_etm_vae_transform_is_encoder_pass():
    docs, vocab, word_emb, _ = _planted()
    m = _vae(seed=2)
    m.fit(docs, word_emb, vocab, iters=200)
    # transform takes token docs and returns a distribution per doc.
    new = [["b0w1", "b0w2"], ["b1w0", "b1w3"], ["b2w5", "b2w4"]]
    theta = m.transform(new)
    assert theta.shape == (3, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert len(set(theta.argmax(axis=1))) == 3  # each lands on a distinct topic


def test_etm_vae_determinism():
    docs, vocab, word_emb, _ = _planted()
    a = _vae(seed=7)
    a.fit(docs, word_emb, vocab, iters=200)
    b = _vae(seed=7)
    b.fit(docs, word_emb, vocab, iters=200)
    assert np.allclose(a.topic_word, b.topic_word)
    assert np.allclose(a.doc_topic, b.doc_topic)


# --- VAE objective/prior flags (#174 contrastive, #176 prior) ----------------

def test_etm_vae_flags_accepted():
    m = topica.ETM(num_topics=3, inference="vae", prior="dirichlet",
                   contrastive=True, contrastive_weight=0.3, contrastive_temp=0.2)
    assert m.inference == "vae"


def test_etm_flag_validation():
    with pytest.raises(ValueError):
        topica.ETM(num_topics=3, inference="vae", prior="gaussian")
    with pytest.raises(ValueError):
        topica.ETM(num_topics=3, inference="vae", contrastive=True, contrastive_weight=-1.0)
    with pytest.raises(ValueError):
        topica.ETM(num_topics=3, inference="vae", contrastive=True, contrastive_temp=0.0)


def test_etm_vae_flags_change_results():
    docs, vocab, word_emb, _ = _planted()
    base = _vae(seed=1); base.fit(docs, word_emb, vocab, iters=80)
    dir_m = _vae(seed=1, prior="dirichlet"); dir_m.fit(docs, word_emb, vocab, iters=80)
    con_m = _vae(seed=1, contrastive=True); con_m.fit(docs, word_emb, vocab, iters=80)
    assert not np.allclose(base.topic_word, dir_m.topic_word)
    assert not np.allclose(base.topic_word, con_m.topic_word)
    for m in (dir_m, con_m):
        assert np.allclose(m.topic_word.sum(axis=1), 1.0)
        assert np.allclose(m.doc_topic.sum(axis=1), 1.0)


def test_etm_vae_flags_deterministic():
    docs, vocab, word_emb, _ = _planted()
    a = _vae(seed=3, prior="dirichlet", contrastive=True); a.fit(docs, word_emb, vocab, iters=80)
    b = _vae(seed=3, prior="dirichlet", contrastive=True); b.fit(docs, word_emb, vocab, iters=80)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_etm_vae_flags_save_load_roundtrip(tmp_path):
    docs, vocab, word_emb, _ = _planted()
    m = _vae(seed=2, prior="dirichlet", contrastive=True)
    m.fit(docs, word_emb, vocab, iters=80)
    path = str(tmp_path / "etm_vae_flags.bin")
    m.save(path)
    loaded = topica.ETM.load(path)
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)


def test_etm_stick_breaking_valid_and_deterministic():
    docs, vocab, word_emb, _ = _planted()
    base = _vae(seed=1); base.fit(docs, word_emb, vocab, iters=80)
    sb = _vae(seed=1, prior="stick_breaking"); sb.fit(docs, word_emb, vocab, iters=80)
    # The prior changes the fit but the outputs stay on the simplex.
    assert not np.allclose(base.topic_word, sb.topic_word)
    assert np.allclose(sb.topic_word.sum(axis=1), 1.0)
    assert np.allclose(sb.doc_topic.sum(axis=1), 1.0)
    # determinism
    sb2 = _vae(seed=1, prior="stick_breaking"); sb2.fit(docs, word_emb, vocab, iters=80)
    assert np.array_equal(sb.topic_word, sb2.topic_word)
    assert np.array_equal(sb.doc_topic, sb2.doc_topic)


def test_etm_stick_breaking_save_load_roundtrip(tmp_path):
    docs, vocab, word_emb, _ = _planted()
    m = _vae(seed=2, prior="stick_breaking"); m.fit(docs, word_emb, vocab, iters=80)
    path = str(tmp_path / "etm_sb.bin")
    m.save(path)
    loaded = topica.ETM.load(path)
    # The stick-breaking map survives the round-trip (doc_topic matches exactly).
    assert np.array_equal(loaded.topic_word, m.topic_word)
    assert np.array_equal(loaded.doc_topic, m.doc_topic)
