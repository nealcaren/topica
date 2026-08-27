"""Tests for ReplyTM — the reply-threaded topic model (experimental).

The Rust side (src/reply_tm.rs) carries the planted-recovery test through the full model; these
exercise the Python binding contract: the experimental gate, the fit surface (parents +
covariate), array shapes, input validation, and a light end-to-end topic + prevalence recovery.
"""
import subprocess
import sys

import numpy as np
import pytest

import topica

topica.enable_experimental()


def _threaded_corpus(seed=13, n_threads=60, depth=10, doc_len=40):
    """Two disjoint block topics, an OU-ish prevalence walk down chains, two covariate groups
    (group g's roots lean toward topic g)."""
    rng = np.random.default_rng(seed)
    vocab = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]  # blocks 0..4 and 5..9
    docs, parents, cov = [], [], []
    for t in range(n_threads):
        g = t % 2
        base = len(docs)
        eta = 1.2 if g == 0 else -1.2  # group leans toward its block
        for step in range(depth):
            parents.append(-1 if step == 0 else base + step - 1)
            eta += rng.normal(0, 0.4)
            p_a = 1.0 / (1.0 + np.exp(-eta))
            doc = []
            for _ in range(doc_len):
                blk = 0 if rng.random() < p_a else 1
                doc.append(vocab[blk * 5 + rng.integers(5)])
            docs.append(doc)
            cov.append(g)
    return docs, parents, cov, vocab


def test_experimental_gate_blocks_fit():
    """fit() must refuse to run until experimental models are enabled (fresh interpreter)."""
    code = (
        "import topica; m = topica.ReplyTM(3)\n"
        "try:\n"
        "    m.fit([['a','b','c']], parents=[-1]); print('NOTGATED')\n"
        "except RuntimeError as e:\n"
        "    print('GATED' if 'experimental' in str(e) else 'OTHER')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "GATED", out.stdout + out.stderr


def test_fit_shapes_and_readouts():
    docs, parents, cov, vocab = _threaded_corpus()
    m = topica.ReplyTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariate=cov, covariate_labels=["A", "B"])
    D, K, V, G = len(docs), 2, len(vocab), 2
    assert m.topic_word().shape == (K, V)
    assert m.doc_topic().shape == (D, K)
    assert m.group_prevalence().shape == (G, K)
    assert np.allclose(m.topic_word().sum(1), 1.0)
    assert np.allclose(m.doc_topic().sum(1), 1.0)
    assert m.group_labels() == ["A", "B"]
    assert set(m.vocabulary()) == set(vocab)
    assert np.isfinite(m.kappa) and np.isfinite(m.sigma2)
    assert len(m.bound_history) >= 1
    # ELBO should not decrease overall
    assert m.bound_history[-1] >= m.bound_history[0] - 1e-6


def test_topic_and_prevalence_recovery():
    docs, parents, cov, vocab = _threaded_corpus()
    m = topica.ReplyTM(2, em_iters=100, seed=13)
    m.fit(docs, parents=parents, covariate=cov, covariate_labels=["A", "B"])
    beta = m.topic_word()
    vidx = {w: i for i, w in enumerate(m.vocabulary())}
    a_cols = [vidx[f"a{i}"] for i in range(5)]
    # each true block should be captured by a distinct topic (mass concentrated on the block)
    a_mass = [beta[k][a_cols].sum() for k in range(2)]
    # one topic is mostly A-block, the other mostly B-block
    assert max(a_mass) > 0.8 and min(a_mass) < 0.2, a_mass
    # group prevalence differs by group (A-group leans to a different topic than B-group)
    gp = m.group_prevalence()
    assert not np.allclose(gp[0], gp[1], atol=0.1)


def test_parent_validation():
    m = topica.ReplyTM(2, em_iters=5)
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1])  # wrong length
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1, 5])  # out of range
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1, 1])  # self-parent


def test_unfitted_raises():
    m = topica.ReplyTM(3)
    with pytest.raises(RuntimeError):
        m.topic_word()
