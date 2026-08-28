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
    m.fit(docs, parents=parents, covariates=cov, covariate_labels=["A", "B"])
    D, K, V, G = len(docs), 2, len(vocab), 2
    assert m.topic_word.shape == (K, V)
    assert m.doc_topic.shape == (D, K)
    assert m.group_prevalence.shape == (G, K)
    assert np.allclose(m.topic_word.sum(1), 1.0)
    assert np.allclose(m.doc_topic.sum(1), 1.0)
    assert m.group_labels() == ["A", "B"]
    assert set(m.vocabulary) == set(vocab)
    assert m.num_topics == K
    assert np.isfinite(m.kappa) and np.isfinite(m.sigma2)
    assert len(m.bound_history) >= 1
    # ELBO should not decrease overall
    assert m.bound_history[-1] >= m.bound_history[0] - 1e-6
    # uncertainty is reported: prevalence SE (G x K-1) and a kappa CI bracketing kappa
    assert m.prevalence_se.shape == (G, K - 1)
    assert np.all(m.prevalence_se >= 0)
    lo, hi = m.kappa_ci
    assert lo <= m.kappa + 1e-9 and hi >= m.kappa - 1e-9
    # top_words: all-topics mode returns K lists; single-topic mode returns one list
    allw = m.top_words()
    assert len(allw) == K and all(isinstance(t, list) for t in allw)
    assert isinstance(m.top_words(3, topic=0), list) and len(m.top_words(3, topic=0)) == 3


def test_topic_and_prevalence_recovery():
    docs, parents, cov, vocab = _threaded_corpus()
    m = topica.ReplyTM(2, em_iters=100, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_labels=["A", "B"])
    beta = m.topic_word
    vidx = {w: i for i, w in enumerate(m.vocabulary)}
    a_cols = [vidx[f"a{i}"] for i in range(5)]
    # each true block should be captured by a distinct topic (mass concentrated on the block)
    a_mass = [beta[k][a_cols].sum() for k in range(2)]
    # one topic is mostly A-block, the other mostly B-block
    assert max(a_mass) > 0.8 and min(a_mass) < 0.2, a_mass
    # group prevalence differs by group (A-group leans to a different topic than B-group)
    gp = m.group_prevalence
    assert not np.allclose(gp[0], gp[1], atol=0.1)


def test_parent_validation():
    m = topica.ReplyTM(2, em_iters=5)
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1])  # wrong length
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1, 5])  # out of range
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1, 1])  # self-parent
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[1, 0])  # cycle A->B->A


def test_unfitted_raises():
    m = topica.ReplyTM(3)
    with pytest.raises(RuntimeError):
        m.topic_word


def test_reduces_to_flat_when_no_tree():
    """With no reply edges ReplyTM is a plain logistic-normal topic model: topics still recover,
    and the reply parameters are correctly flagged unidentified (NaN)."""
    import math
    import warnings

    docs, parents, cov, vocab = _threaded_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # parents=None warns on purpose
        m = topica.ReplyTM(2, em_iters=80, seed=13)
        m.fit(docs, parents=None)
    beta = m.topic_word
    vidx = {w: i for i, w in enumerate(m.vocabulary)}
    a_cols = [vidx[f"a{i}"] for i in range(5)]
    a_mass = [beta[k][a_cols].sum() for k in range(2)]
    assert max(a_mass) > 0.8 and min(a_mass) < 0.2, a_mass  # topics still recovered
    assert math.isnan(m.kappa) and math.isnan(m.sigma2)  # no edges → unidentified


def test_held_out_beat_tree_vs_no_tree():
    """Acceptance gate: on a persistence-structured corpus, the tree prior (the parent's θ)
    predicts held-out leaf tokens better than the no-tree baseline (group prevalence)."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=80, depth=12)
    d = len(docs)
    has_child = {p for p in parents if p >= 0}
    leaves = [i for i in range(d) if parents[i] >= 0 and i not in has_child]
    rng = np.random.default_rng(0)
    test = set(rng.choice(leaves, size=len(leaves) // 3, replace=False).tolist())
    train = [[] if i in test else docs[i] for i in range(d)]  # hold out leaf tokens
    m = topica.ReplyTM(2, em_iters=100, seed=13)
    m.fit(train, parents=parents, covariates=cov, covariate_labels=["A", "B"])
    beta, theta, anchor = m.topic_word, m.doc_topic, m.group_prevalence
    vidx = {w: i for i, w in enumerate(m.vocabulary)}

    def tokll(i, th):
        ids = [vidx[w] for w in docs[i] if w in vidx]
        if not ids:
            return None
        pw = beta[:, ids].T @ th
        return float(np.mean(np.log(pw + 1e-12)))

    tree, no_tree = [], []
    for dn in test:
        a = tokll(dn, theta[parents[dn]])  # predict from the parent (tree)
        b = tokll(dn, anchor[cov[dn]])  # predict from the group baseline (no tree)
        if a is not None and b is not None:
            tree.append(a)
            no_tree.append(b)
    assert np.mean(tree) > np.mean(no_tree), (np.mean(tree), np.mean(no_tree))


def test_fit_accepts_corpus():
    """fit() takes a topica.Corpus (doc order preserved) as well as raw token lists, so the
    reply tree indexes line up with the corpus documents."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    corpus = topica.Corpus.from_documents(docs)
    m = topica.ReplyTM(2, em_iters=40, seed=13)
    m.fit(corpus, parents=parents, covariates=cov, covariate_labels=["A", "B"])
    assert m.topic_word.shape == (2, len(vocab))
    assert m.doc_topic.shape == (len(docs), 2)
    assert set(m.vocabulary) == set(vocab)


def test_min_count_emptying_warns():
    """A document whose every token is rarer than min_count is emptied but kept as a tree node;
    the user is warned rather than silently losing the document's content."""
    import warnings

    docs = [["a0", "a0", "a1"], ["rareX", "rareY"], ["a1", "a1", "a0"]]
    m = topica.ReplyTM(2, em_iters=10, seed=13)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(docs, parents=[-1, 0, 1], min_count=2)
    assert any("emptied" in str(x.message) for x in w), [str(x.message) for x in w]


def test_inspect_integration():
    """The taught inspect API must work on ReplyTM (regression: it was misdispatched as a
    time-sliced model because topic_word/vocabulary were methods, not properties)."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    m = topica.ReplyTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents)
    table = topica.inspect.topic_table(m)
    assert len(table) == 2
