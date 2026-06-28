"""Smoke + behavior tests for IdealPointTM, the experimental embedded topic model
with a latent ideal-point head.

IdealPointTM is gated and unvalidated, so the bar is: the gate works, the fitted
surface is well-shaped (simplex outputs, conformant attributes, standardized
positions), save/load round-trips bit-for-bit, the fit is reproducible from a
seed, and its distinctive claims hold on a planted corpus -- authors with
systematically different word choice within a topic get separated positions, the
discriminating topic carries the larger ||W_k||, anchors orient the sign, and
position_shift reads the axis.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import topica


@pytest.fixture(autouse=True)
def _experimental():
    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    yield
    topica.enable_experimental(was)


K = 3


def _planted(n_authors=24, docs_per=10, length=40, seed=0):
    """Two author camps. Topic 0 is contested: a camp-dependent latent trait t_a
    pushes word choice toward the 'left' or 'right' half of topic 0's block. Topic
    1 and 2 are neutral. Word embeddings carry a topic-identity signal plus, for
    topic 0, a left/right axis the loading can find."""
    rng = np.random.default_rng(seed)
    block = 8
    V = K * block
    # vocab and embeddings: dims 0..K topic identity, dim K is topic-0 L/R axis.
    E = K + 1
    vocab = [f"t{b}w{i}" for b in range(K) for i in range(block)]
    emb = np.zeros((V, E))
    for v in range(V):
        b = v // block
        emb[v, b] = 3.0
        if b == 0:
            # first half of topic-0 words = "left" (-), second half = "right" (+)
            emb[v, K] = -2.0 if (v % block) < block // 2 else 2.0
    emb += rng.normal(scale=0.1, size=emb.shape)

    trait = rng.uniform(-1.0, 1.0, size=n_authors)
    docs, group = [], []
    for a in range(n_authors):
        p_right = 1.0 / (1.0 + np.exp(-3.0 * trait[a]))
        for _ in range(docs_per):
            doc = []
            for _ in range(length):
                t = rng.integers(0, K)
                if t == 0:
                    half = 1 if rng.random() < p_right else 0
                    lo = half * (block // 2)
                    w = lo + rng.integers(0, block // 2)
                else:
                    w = t * block + rng.integers(0, block)
                doc.append(vocab[w])
            docs.append(doc)
            group.append(f"author_{a}")
    return docs, vocab, emb, group, trait


def test_gate_blocks_construction():
    topica.enable_experimental(False)
    with pytest.raises(Exception):
        topica.IdealPointTM(K)
    topica.enable_experimental(True)  # restore for the rest (fixture also resets)


def test_fitted_surface_is_well_shaped():
    docs, vocab, emb, group, _ = _planted(seed=1)
    m = topica.IdealPointTM(K, num_dims=1, seed=1)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group, iters=25)

    assert m.topic_word.shape == (K, len(vocab))
    assert m.doc_topic.shape == (len(docs), K)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.num_authors == 24
    assert m.author_positions.shape == (24, 1)
    # positions are standardized to mean 0 / unit variance per dimension.
    assert abs(m.author_positions.mean()) < 1e-6
    assert abs(m.author_positions.std() - 1.0) < 1e-6
    assert m.topic_discrimination.shape == (K,)
    assert len(m.author_names) == 24
    assert m.coherence().shape == (K,)
    assert len(m.top_words(5)) == K


@pytest.mark.parametrize("representation", ["word2vec", "counts"])
def test_position_se_is_well_shaped_and_shrinks_with_data(representation):
    # Observed-information SE on the latent positions: aligned to author_positions,
    # finite and positive, capped by the prior SD (sqrt(x_prior_variance)=1), and
    # smaller for authors who contribute more tokens. Holds for both representations.
    docs, vocab, emb, group, _ = _planted(seed=4)
    # Give author_0 many extra documents so it is the data-rich author.
    extra = [docs[i] for i in range(len(group)) if group[i] == "author_0"]
    docs = docs + extra * 6
    group = group + ["author_0"] * (len(extra) * 6)

    m = topica.IdealPointTM(K, num_dims=1, seed=1)
    kw = dict(word_embeddings=emb, vocabulary=vocab) if representation == "word2vec" else {}
    m.fit(docs, group=group, iters=25, **kw)
    assert m.representation == representation

    se = m.position_se
    assert se.shape == m.author_positions.shape == (m.num_authors, 1)
    assert np.all(np.isfinite(se)) and np.all(se > 0.0)
    assert np.all(se <= 1.0 + 1e-9)  # prior SD caps the SE

    names = list(m.author_names)
    rich = names.index("author_0")
    others = [i for i in range(len(names)) if i != rich]
    assert se[rich, 0] < np.median(se[others, 0])


def test_position_se_survives_save_load():
    # The SE is reconstructed from saved state (corpus + doc-topic proportions), so
    # it is identical after a round trip even though it is not itself persisted.
    docs, vocab, emb, group, _ = _planted(seed=5)
    m = topica.IdealPointTM(K, num_dims=1, seed=1)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group, iters=20)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.topica")
        m.save(path)
        m2 = topica.IdealPointTM.load(path)
    assert np.array_equal(m.position_se, m2.position_se)


def test_recovers_positions():
    # The headline ideal-point claim: latent positions recover the planted author
    # trait. (Which topic absorbs the contrast is not asserted here -- on this
    # approximate generator the left/right split can be explained as within-topic
    # content or as topic-splitting; the exact-model localization claim lives in
    # the Rust unit test, which samples from the model with no such confound.)
    docs, vocab, emb, group, trait = _planted(seed=2)
    lo, hi = int(np.argmin(trait)), int(np.argmax(trait))
    anchors = {f"author_{lo}": -1.0, f"author_{hi}": 1.0}
    m = topica.IdealPointTM(K, num_dims=1, seed=3)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group, anchors=anchors, iters=60)

    pos = m.author_positions[:, 0]
    order = [int(name.split("_")[1]) for name in m.author_names]
    r = np.corrcoef(pos, trait[order])[0, 1]
    assert r > 0.85, f"position correlation too low: {r}"
    # the ideal-point head is actually used (some topic carries discrimination).
    assert m.topic_discrimination.max() > 0.1


def test_anchors_orient_sign_deterministically():
    docs, vocab, emb, group, trait = _planted(seed=4)
    lo, hi = int(np.argmin(trait)), int(np.argmax(trait))
    m = topica.IdealPointTM(K, num_dims=1, seed=5)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group,
          anchors={f"author_{lo}": -1.0, f"author_{hi}": 1.0}, iters=40)
    names = m.author_names
    pos = m.author_positions[:, 0]
    p_lo = pos[names.index(f"author_{lo}")]
    p_hi = pos[names.index(f"author_{hi}")]
    assert p_lo < p_hi, "anchors should orient the low-trait author below the high"


def test_position_shift_reads_the_axis():
    docs, vocab, emb, group, trait = _planted(seed=6)
    lo, hi = int(np.argmin(trait)), int(np.argmax(trait))
    m = topica.IdealPointTM(K, num_dims=1, seed=7)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group,
          anchors={f"author_{lo}": -1.0, f"author_{hi}": 1.0}, iters=60)
    # read the most discriminating topic's within-topic contrast.
    kd = int(np.argmax(m.topic_discrimination))
    pos, neg = m.position_shift(kd, n=4)
    assert len(pos) == 4 and len(neg) == 4
    pos_words = {w for w, _ in pos}
    neg_words = {w for w, _ in neg}
    # the two ends of the axis are driven by different words, and the log-ratios
    # point in opposite directions.
    assert pos_words.isdisjoint(neg_words)
    assert all(s >= 0 for _, s in pos) and all(s <= 0 for _, s in neg)


def test_loadings_and_weighting_surface():
    docs, vocab, emb, group, _ = _planted(seed=11)
    m = topica.IdealPointTM(K, num_dims=1, seed=2)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group, iters=25)
    # loadings expose the per-topic discrimination directions.
    L = m.loadings
    assert L.shape == (K, emb.shape[1])  # (num_topics, num_dims*E), d=1
    # both weightings work; prob (default) and logratio rank words.
    kd = int(np.argmax(m.topic_discrimination))
    pp, nn = m.position_shift(kd, n=4, weighting="prob")
    lp, ln = m.position_shift(kd, n=4, weighting="logratio")
    assert len(pp) == 4 and len(lp) == 4
    with pytest.raises(Exception):
        m.position_shift(kd, weighting="nonsense")


def test_reproducible_from_seed():
    docs, vocab, emb, group, _ = _planted(seed=8)
    a = topica.IdealPointTM(K, seed=11)
    a.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group, iters=20)
    b = topica.IdealPointTM(K, seed=11)
    b.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group, iters=20)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.author_positions, b.author_positions)


def test_save_load_roundtrip():
    docs, vocab, emb, group, _ = _planted(seed=9)
    m = topica.IdealPointTM(K, num_dims=1, seed=13)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, group=group, iters=20)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ip.topica")
        m.save(path)
        m2 = topica.IdealPointTM.load(path)
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert np.array_equal(m.author_positions, m2.author_positions)
    assert np.array_equal(m.topic_discrimination, m2.topic_discrimination)
    assert m2.author_names == m.author_names
    assert m2.num_authors == m.num_authors


def test_ungrouped_defaults_to_per_document_authors():
    docs, vocab, emb, _, _ = _planted(n_authors=6, docs_per=4, seed=10)
    m = topica.IdealPointTM(K, seed=1)
    m.fit(docs, word_embeddings=emb, vocabulary=vocab, iters=12)  # no group
    assert m.num_authors == len(docs)
    assert m.author_positions.shape == (len(docs), 1)
