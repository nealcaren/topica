"""Tests for topica.threads: thread-context shrinkage over a fitted base model (#895).

Validation basis is planted recovery. We simulate reply threads with known topic mixes in
which a reply either inherits its parent's topics or does not, fit an LDA base, and check that
the smoother (a) finds a positive parent pseudo-count and a placebo-netted edge effect whose
interval excludes zero when replies inherit, (b) moves short replies' topic mixes toward the
truth, and (c) finds no edge effect on a no-inheritance null.

Known limit (documented in topica.threads): one pseudo-count per context borrows for every
reply, so when only about half of replies inherit and the rest start sharply different topics,
borrowing hurts the non-inheriting replies as much as it helps the others and the fit returns
zero. A per-reply inherit-or-innovate switch is the follow-up.
"""
from collections import Counter
from itertools import permutations

import numpy as np
import pytest

import topica
from topica import threads

topica.enable_experimental()

K, V = 4, 120


def _simulate(inherit, n_threads=150, seed=0):
    """Threads of one long root and 4-10 short replies. With probability ``inherit`` a reply
    draws its topic mix near its parent's; otherwise it draws a fresh one."""
    rng = np.random.default_rng(seed)
    beta = rng.dirichlet(np.full(V, 0.05), K)
    docs, parents, truth = [], [], []

    def emit(theta, length):
        z = rng.choice(K, size=length, p=theta)
        return [f"w{rng.choice(V, p=beta[k])}" for k in z]

    for _ in range(n_threads):
        members = [len(docs)]
        th = rng.dirichlet(np.full(K, 0.2))
        docs.append(emit(th, 60)), parents.append(-1), truth.append(th)
        for _ in range(rng.integers(4, 11)):
            p = int(rng.choice(members))
            if rng.random() < inherit:
                th = rng.dirichlet(80 * truth[p] + 0.05)
            else:
                th = rng.dirichlet(np.full(K, 0.2))
            members.append(len(docs))
            docs.append(emit(th, int(rng.integers(5, 13)))), parents.append(p)
            truth.append(th)
    return docs, parents, np.array(truth), beta


def _lda(corpus):
    return topica.LDA(K, seed=1).fit(corpus, iters=400)


def _align(est_beta, vocab, true_beta):
    """Best permutation of estimated topics onto true topics (K is small: brute force)."""
    cols = [int(w[1:]) for w in vocab]
    tb = true_beta[:, cols]
    tb = tb / tb.sum(1, keepdims=True)
    best, best_perm = -np.inf, None
    for perm in permutations(range(K)):
        s = sum(np.dot(est_beta[perm[k]], tb[k]) for k in range(K))
        if s > best:
            best, best_perm = s, perm
    return list(best_perm)


@pytest.fixture(scope="module")
def inherited():
    docs, parents, truth, beta = _simulate(inherit=0.9, n_threads=300, seed=0)
    sm = threads.ThreadSmoother().fit(docs, parents, base=_lda, seed=3, n_boot=300)
    return sm, docs, parents, truth, beta


def test_recovers_positive_parent_pseudocount_and_edge_effect(inherited):
    sm = inherited[0]
    assert sm.alpha["parent"] > 0
    assert sm.edge_effect["lo"] > 0, sm.edge_effect
    assert sm.completion["lo"] > 0, sm.completion
    # The gain concentrates in the shortest replies.
    g = sm.completion["by_length"]["gain"]
    assert g[0] > g[2]


def test_smoothing_moves_short_replies_toward_the_truth(inherited):
    sm, docs, parents, truth, beta = inherited
    perm = _align(np.asarray(sm.base_model.topic_word), list(sm.corpus.vocabulary), beta)
    kept = np.asarray(sm.corpus.kept_indices)
    base = np.full((len(docs), K), np.nan)
    base[kept] = np.asarray(sm.base_model.doc_topic)[:, perm]
    smooth = sm.theta_tilde[:, perm]
    short = np.array([p >= 0 and len(d) <= 8 for d, p in zip(docs, parents)])
    ok = short & ~np.isnan(base[:, 0]) & ~np.isnan(smooth[:, 0])
    l1_base = np.abs(base[ok] - truth[ok]).sum(1).mean()
    l1_smooth = np.abs(smooth[ok] - truth[ok]).sum(1).mean()
    assert l1_smooth < 0.9 * l1_base, (l1_smooth, l1_base)


def test_no_edge_effect_without_inheritance():
    docs, parents, _, _ = _simulate(inherit=0.0, seed=1)
    sm = threads.ThreadSmoother().fit(docs, parents, base=_lda, seed=3, n_boot=300,
                                      final=False)
    assert sm.edge_effect["lo"] <= 0 <= sm.edge_effect["hi"], sm.edge_effect
    assert sm.alpha["parent"] == 0  # no borrowing when replies do not inherit


def test_shuffle_preserves_depth_thread_and_child_counts():
    docs, parents, _, _ = _simulate(inherit=0.5, n_threads=40, seed=2)
    sp = threads.shuffle_parents(parents, seed=0)
    d0, r0, _ = threads.thread_structure(parents)
    d1, r1, _ = threads.thread_structure(sp)
    assert (d0 == d1).all() and (r0 == r1).all()
    assert Counter(p for p in parents if p >= 0) == Counter(p for p in sp if p >= 0)
    assert sp != parents  # something actually moved


def test_transform_is_a_convex_mix_and_handles_empty_documents(inherited):
    sm, docs, parents, *_ = inherited
    th = sm.theta_tilde
    rows = ~np.isnan(th[:, 0])
    assert np.allclose(th[rows].sum(1), 1.0)
    assert (th[rows] >= 0).all()
    # A document the vocabulary empties still gets a context mix.
    docs2 = [list(d) for d in docs]
    j = next(i for i, p in enumerate(parents) if p >= 0)
    docs2[j] = ["zzz_unseen"]
    corpus = topica.Corpus.from_documents(docs2)
    model = _lda(corpus)
    out = sm.transform(model, corpus, parents)
    assert not np.isnan(out[j]).any() and np.isclose(out[j].sum(), 1.0)


def test_single_context_and_groups():
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=80, seed=4)
    groups = ["a" if i % 2 else "b" for i in range(len(docs))]
    sm = threads.ThreadSmoother(contexts=("parent",)).fit(
        docs, parents, base=_lda, groups=groups, seed=3, n_boot=100, final=False)
    assert set(sm.alpha) == {"parent"} and sm.parent_share is None
    assert set(sm.alpha_by_group) <= {"a", "b"}


@pytest.mark.parametrize("model", ["STM", "CTM"])
def test_logistic_normal_bases(model):
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=60, seed=5)
    cls = getattr(topica, model)
    X = np.random.default_rng(0).integers(0, 2, (len(docs), 1)).astype(float)

    def base(c):  # STM's prevalence rows must follow the corpus's kept documents
        if model == "STM":
            return cls(K, seed=1).fit(c, prevalence=X[np.asarray(c.kept_indices)])
        return cls(K, seed=1).fit(c)
    sm = threads.ThreadSmoother().fit(docs, parents, base=base, seed=3, n_boot=100)
    assert sm.theta_tilde.shape == (len(docs), K)
    assert np.isfinite(sm.completion["estimate"])


def test_strip_helpers():
    assert threads.strip_quotes("&gt;quoted line<p>my reply") == "my reply"
    assert threads.strip_quotes("> quoted\nmine").strip() == "mine"
    assert threads.strip_quotes("as you said <i>the cat</i> no") == "as you said   no"
    docs = [list("abcdefg"), list("xxcdefgyy")]
    assert threads.strip_copied_runs(docs, [-1, 0]) == [list("abcdefg"), list("xxyy")]


def test_validation_and_gate():
    with pytest.raises(ValueError):
        threads.ThreadSmoother(contexts=("sibling",))
    with pytest.raises(ValueError):
        threads.thread_structure([1, 0])  # cycle
    topica.enable_experimental(False)
    try:
        with pytest.raises(RuntimeError):
            threads.ThreadSmoother()
    finally:
        topica.enable_experimental()
