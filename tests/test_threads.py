"""Tests for topica.threads: thread-context shrinkage over a fitted base model (#895).

Validation basis is planted recovery. We simulate reply threads with known topic mixes in
which a reply either inherits its parent's topics or does not, fit an LDA base, and check that
the smoother (a) finds a positive parent pseudo-count and a placebo-netted edge effect whose
interval excludes zero when replies inherit, (b) moves short replies' topic mixes toward the
truth, and (c) finds no edge effect on a no-inheritance null.

Known limit (documented in topica.threads): one pseudo-count per context borrows for every
reply, so when only about half of replies inherit and the rest start sharply different topics,
borrowing hurts the non-inheriting replies as much as it helps the others and the fit returns
zero. The per-reply inherit-or-innovate switch (`switch=True`, #897) handles that case; its
tests are at the end of this file.
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


def test_parent_share_stays_inside_the_unit_interval(inherited):
    sm = inherited[0]
    lo, hi = sm.parent_share_ci
    assert 0 < lo <= sm.parent_share <= hi < 1
    assert isinstance(sm.parent_share_at_bound, bool)
    assert 0 <= sm.p_no_borrowing <= 1
    assert sm.draws["parent_share"].shape[0] == sm.settings["n_boot"]


def test_refit_bootstrap_pools_calibrations_and_reseeds_the_base():
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=120, seed=6)
    seen = []

    def base(corpus, seed):
        seen.append(seed)
        return topica.LDA(K, seed=seed).fit(corpus, iters=300)

    sm = threads.ThreadSmoother().fit(docs, parents, base=base, seed=3, n_boot=100,
                                      n_refit=2, final=False)
    assert sm.uncertainty == "refit" and len(sm.replicates) == 3
    assert len(set(seen)) == 3                      # a new base seed per calibration
    assert sm.draws["alpha"].shape == (300, 2)      # draws pooled across calibrations
    assert sm.edge_effect["lo"] <= sm.edge_effect["estimate"] <= sm.edge_effect["hi"]


def test_thread_context_excludes_the_parent_by_default():
    # One thread: root 0 -> reply 1 -> reply 2, plus a sibling 3 of reply 1.
    theta = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    lengths = np.array([10.0, 10.0, 10.0, 10.0])
    parents = [-1, 0, 1, 0]
    _, root, _ = threads.thread_structure(parents)
    _, _, ctx = threads._context_mixes(theta, lengths, np.arange(4), parents, root,
                                       ("parent", "thread"), exclude_parent=True)
    assert np.allclose(ctx["parent"][2], [0.0, 1.0])
    assert np.allclose(ctx["thread"][2], [1.0, 0.0])   # docs 0 and 3 only, not parent 1
    _, _, ctx2 = threads._context_mixes(theta, lengths, np.arange(4), parents, root,
                                        ("parent", "thread"), exclude_parent=False)
    assert np.allclose(ctx2["thread"][2], [2 / 3, 1 / 3])   # docs 0, 1 and 3


def test_threadtm_model_surface():
    docs, parents, truth, beta = _simulate(inherit=0.9, n_threads=300, seed=8)
    m = topica.ThreadTM(K, seed=3).fit(docs, parents, iters=400, n_boot=200)
    assert m.doc_topic.shape == (len(m.corpus.kept_indices), K)
    assert m.doc_topic_all.shape == (len(docs), K)
    assert m.topic_word.shape == (K, len(m.vocabulary))
    assert len(m.top_words(5)) == K
    assert m.alpha["parent"] > 0 and m.edge_effect["lo"] > 0
    assert 0 < m.parent_share < 1
    assert "ThreadTM(num_topics=4" in repr(m)


def test_threadtm_stm_base_aligns_prevalence_rows():
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=60, seed=9)
    docs[5] = ["zzz_rare_only"]            # emptied by min_cf, so kept rows != input rows
    X = np.random.default_rng(1).integers(0, 2, (len(docs), 1)).astype(float)
    m = topica.ThreadTM(K, base="stm", seed=3).fit(
        docs, parents, prevalence=X, n_boot=50, corpus_kwargs={"min_cf": 2},
        fit_kwargs={"restarts": 1})
    kept = np.asarray(m.corpus.kept_indices)
    assert 5 not in kept and m.doc_topic.shape == (len(kept), K)   # rows follow kept_indices
    assert m.doc_topic_all.shape == (len(docs), K)
    assert np.allclose(m.doc_topic, m.doc_topic_all[kept])
    with pytest.raises(ValueError):
        topica.ThreadTM(K, base="stm").fit(docs, parents)


def _simulate_thread_level(n_threads=300, seed=0):
    """Every reply draws its topics near its THREAD's mix, not its parent's: the parent is no
    more informative than any other comment in the thread."""
    rng = np.random.default_rng(seed)
    beta = rng.dirichlet(np.full(V, 0.05), K)
    docs, parents = [], []

    def emit(theta, length):
        z = rng.choice(K, size=length, p=theta)
        return [f"w{rng.choice(V, p=beta[k])}" for k in z]

    for _ in range(n_threads):
        th_thread = rng.dirichlet(np.full(K, 0.2))
        members = [len(docs)]
        docs.append(emit(rng.dirichlet(80 * th_thread + 0.05), 60)), parents.append(-1)
        for _ in range(rng.integers(4, 11)):
            p = int(rng.choice(members))
            members.append(len(docs))
            docs.append(emit(rng.dirichlet(80 * th_thread + 0.05), int(rng.integers(5, 13))))
            parents.append(p)
    return docs, parents


def test_placebo_contexts_come_from_the_shuffled_tree():
    # Regression (PR #896 review): the placebo must build BOTH contexts from the shuffled tree,
    # so it differs from the true tree only in which comment is the parent. The pre-fix code
    # kept the true tree's thread mix, which leaves the true parent out of the placebo entirely.
    rng = np.random.default_rng(0)
    parents = [-1] + [0] * 6 + [1 + (i % 6) for i in range(12)]   # root, 6 depth-1, 12 depth-2
    n = len(parents)
    theta = rng.dirichlet(np.ones(K), n)
    lengths = np.full(n, 10.0)
    _, root, _ = threads.thread_structure(parents)
    for draw_i, ctx in enumerate(threads._placebo_contexts(
            theta, lengths, np.arange(n), parents, root, ("parent", "thread"), True, 3, 5)):
        sp = threads.shuffle_parents(parents, rng=np.random.default_rng(5 * 1000 + draw_i))
        for d in range(7, n):
            true_p, stand_in = parents[d], sp[d]
            others = [j for j in range(n) if j not in (d, stand_in)]
            expect = theta[others].mean(0)                     # equal lengths: plain mean
            assert np.allclose(ctx["parent"][d], theta[stand_in])
            assert np.allclose(ctx["thread"][d], expect)       # includes true_p when moved
            if stand_in != true_p:
                assert true_p in others


def test_no_edge_effect_when_the_parent_adds_nothing_beyond_the_thread():
    # Replies track their thread, not their parent: the placebo-netted edge effect must not
    # manufacture a parent effect. (A behavioral check; the construction is pinned above.)
    docs, parents = _simulate_thread_level(seed=11)
    sm = threads.ThreadSmoother().fit(docs, parents, base=_lda, seed=3, n_boot=300, final=False)
    assert sm.completion["lo"] > 0                          # the thread context does help
    assert sm.edge_effect["lo"] <= 0 <= sm.edge_effect["hi"], sm.edge_effect


def test_large_val_frac_still_leaves_test_threads():
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=6, seed=12)
    sm = threads.ThreadSmoother().fit(docs, parents, base=_lda, seed=3, n_boot=20,
                                      val_frac=0.95, final=False)
    assert sm.settings["n_val_threads"] < sm.settings["n_eval_threads"]


def test_seed_is_passed_only_to_a_parameter_named_seed():
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=40, seed=13)
    seen = {}

    def no_seed(corpus, k=K):              # a second parameter that is NOT a seed
        seen["k"] = k
        return topica.LDA(k, seed=1).fit(corpus, iters=100)

    threads.ThreadSmoother().fit(docs, parents, base=no_seed, seed=99, n_boot=10, final=False)
    assert seen["k"] == K

    def with_seed(corpus, *, seed):
        seen["seed"] = seed
        return topica.LDA(K, seed=seed).fit(corpus, iters=100)

    threads.ThreadSmoother().fit(docs, parents, base=with_seed, seed=99, n_boot=10, final=False)
    assert seen["seed"] == 99


def test_groups_are_read_by_position_not_label():
    pd = pytest.importorskip("pandas")
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=80, seed=14)
    labels = ["a" if i % 2 else "b" for i in range(len(docs))]
    series = pd.Series(labels, index=np.arange(len(docs))[::-1] + 1000)   # non-default index
    kw = dict(base=_lda, seed=3, n_boot=20, final=False)
    a = threads.ThreadSmoother(contexts=("parent",)).fit(docs, parents, groups=labels, **kw)
    b = threads.ThreadSmoother(contexts=("parent",)).fit(docs, parents, groups=series, **kw)
    assert a.alpha_by_group == b.alpha_by_group


# ------------------------------------------------ inherit-or-innovate switch (#897)


def test_sequential_log_ml_is_exact_for_two_tokens():
    # For two tokens the soft-count Polya urn is the exact Dirichlet-multinomial marginal:
    # p(w1) = sum_k m_k b_k,w1 and p(w2 | w1) = sum_k E[theta_k | w1] b_k,w2.
    beta = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    m = np.array([[0.3, 0.7]])
    grid = np.array([0.5, 5.0])
    got = threads._sequential_log_ml([np.array([0, 2])], m, grid, beta)[0]
    for a, g in zip(grid, got):
        p1 = m[0] * beta[:, 0]
        r = p1 / p1.sum()
        post = (a * m[0] + r) / (a + 1)
        assert np.isclose(g, np.log(p1.sum()) + np.log(post @ beta[:, 2]))


def test_switch_recovers_the_edge_effect_at_half_inheritance():
    # The pooled pseudo-count fit borrows nothing here (see module docstring); the switch lets
    # each reply's own words decide, so the inheriting half gains and the rest are left alone.
    # 600 threads, the size #897 set: the placebo averages its draws inside the log, which is
    # conservative, and at 300 threads the interval grazes zero.
    docs, parents, _, _ = _simulate(inherit=0.5, n_threads=600, seed=0)
    sm = threads.ThreadSmoother(switch=True).fit(docs, parents, base=_lda, seed=3, n_boot=100)
    assert sm.edge_effect["lo"] > 0, sm.edge_effect
    assert sm.completion["lo"] > 0, sm.completion
    assert min(sm.completion["by_length"]["gain"]) > 0
    replies = np.array(parents) >= 0
    w = sm.inherit_weights[replies]
    assert w.shape == (replies.sum(),) and ((w >= 0) & (w <= 1)).all()
    assert abs(w.mean() - 0.5) < 0.15                    # tracks the planted rate
    assert sm.rho_ci[0] <= sm.rho <= sm.rho_ci[1]
    assert 0 < sm.parent_share < 1
    th = sm.theta_tilde[~np.isnan(sm.theta_tilde[:, 0])]
    assert np.allclose(th.sum(1), 1.0) and (th >= 0).all()


def test_switch_finds_no_edge_effect_without_inheritance():
    docs, parents, _, _ = _simulate(inherit=0.0, seed=1)
    sm = threads.ThreadSmoother(switch=True).fit(docs, parents, base=_lda, seed=3, n_boot=100,
                                                 final=False)
    assert sm.edge_effect["lo"] <= 0 <= sm.edge_effect["hi"], sm.edge_effect
    assert sm.rho < 0.2


def test_switch_weights_rise_with_inheritance(inherited):
    docs, parents, *_ = inherited[1:3]
    sm = threads.ThreadSmoother(switch=True).fit(docs, parents, base=_lda, seed=3, n_boot=50)
    w = sm.inherit_weights[np.array(parents) >= 0]
    assert w.mean() > 0.75                                # planted rate 0.9


def test_switch_rejects_groups():
    docs, parents, _, _ = _simulate(inherit=0.9, n_threads=20, seed=15)
    with pytest.raises(ValueError, match="groups"):
        threads.ThreadSmoother(switch=True).fit(docs, parents, base=_lda,
                                                groups=[0] * len(docs))


# ------------------------------------------------ the original post as a context (#900)


def _simulate_op(n_threads=300, seed=0):
    """Replies at every depth draw their topics near the thread ROOT (the original post), not
    near their parent. Most replies answer another reply, so parent and OP differ."""
    rng = np.random.default_rng(seed)
    beta = rng.dirichlet(np.full(V, 0.05), K)
    docs, parents = [], []

    def emit(theta, length):
        z = rng.choice(K, size=length, p=theta)
        return [f"w{rng.choice(V, p=beta[k])}" for k in z]

    for _ in range(n_threads):
        root = len(docs)
        th_root = rng.dirichlet(np.full(K, 0.2))
        docs.append(emit(th_root, 60)), parents.append(-1)
        members = [root]
        for _ in range(rng.integers(6, 13)):
            p = int(rng.choice(members[1:] if len(members) > 2 and rng.random() < 0.7
                               else members))
            th = (rng.dirichlet(80 * th_root + 0.05) if rng.random() < 0.9
                  else rng.dirichlet(np.full(K, 0.2)))
            members.append(len(docs))
            docs.append(emit(th, int(rng.integers(5, 13)))), parents.append(p)
    return docs, parents


def test_op_context_is_the_root_for_deep_replies_and_leaves_the_thread_mix():
    # root 0 -> 1 -> 2, and 0 -> 3
    theta = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    parents = [-1, 0, 1, 0]
    _, root, _ = threads.thread_structure(parents)
    _, _, ctx = threads._context_mixes(theta, np.full(4, 10.0), np.arange(4), parents, root,
                                       ("parent", "op", "thread"), exclude_parent=True)
    assert np.isnan(ctx["op"][1, 0]) and np.isnan(ctx["op"][3, 0])   # parent is the root
    assert np.allclose(ctx["op"][2], [1.0, 0.0])
    assert np.allclose(ctx["thread"][2], [0.2, 0.8])                  # not 0 (root) or 1


def test_op_placebo_arms_share_one_thread_pool():
    # root 0 -> 1 -> 2, and 0 -> 3, 0 -> 4. Reply 2's donor must be 3 or 4; both arms leave the
    # root and the donor out of the thread mix, and differ only in the original post.
    theta = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])
    parents = [-1, 0, 1, 0, 0]
    _, root, _ = threads.thread_structure(parents)
    for true, placebo in threads._op_placebo_contexts(
            theta, np.full(5, 10.0), np.arange(5), parents, root,
            ("parent", "op", "thread"), True, 4, 5):
        donor = 3 if np.allclose(placebo["op"][2], theta[3]) else 4
        other = 7 - donor
        assert np.allclose(placebo["op"][2], theta[donor])
        assert np.allclose(true["op"][2], theta[0])
        assert np.allclose(true["thread"][2], theta[other])
        assert np.allclose(placebo["thread"][2], theta[other])
        assert np.allclose(true["parent"][2], placebo["parent"][2])


def test_op_placebo_keeps_a_dropped_root_missing():
    # The corpus dropped the root (row 0 not kept): neither arm may have an original post.
    theta = np.array([[0.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])   # rows 1..4
    parents = [-1, 0, 1, 0, 0]
    _, root, _ = threads.thread_structure(parents)
    kept = np.array([1, 2, 3, 4])
    for true, placebo in threads._op_placebo_contexts(
            theta, np.full(4, 10.0), kept, parents, root, ("parent", "op", "thread"),
            True, 3, 5):
        assert np.isnan(true["op"][2, 0]) and np.isnan(placebo["op"][2, 0])


@pytest.mark.parametrize("switch", [False, True])
def test_op_context_is_recovered_when_replies_answer_the_original_post(switch):
    docs, parents = _simulate_op(seed=16)
    sm = threads.ThreadSmoother(contexts=("parent", "op", "thread"), switch=switch).fit(
        docs, parents, base=_lda, seed=3, n_boot=100, final=False)
    assert sm.alpha["op"] > sm.alpha["parent"], sm.alpha      # a_p not inflated
    assert sm.op_effect["lo"] > 0, sm.op_effect
    assert sm.edge_effect["lo"] <= 0 <= sm.edge_effect["hi"], sm.edge_effect


def test_op_context_finds_no_op_effect_when_replies_ignore_the_original_post():
    # Replies share a thread topic that the original post does not carry (the root is drawn
    # fresh), so the OP should add nothing beyond a same-thread comment standing in for it.
    # (When the OP itself carries the thread's topic, replies following the thread and replies
    # answering the OP generate the same data; a positive op_effect is then expected.)
    rng = np.random.default_rng(18)
    beta = rng.dirichlet(np.full(V, 0.05), K)
    docs, parents = [], []

    def emit(theta, length):
        z = rng.choice(K, size=length, p=theta)
        return [f"w{rng.choice(V, p=beta[k])}" for k in z]

    for _ in range(300):
        th_thread = rng.dirichlet(np.full(K, 0.2))
        members = [len(docs)]
        docs.append(emit(rng.dirichlet(np.full(K, 0.2)), 60)), parents.append(-1)
        for _ in range(rng.integers(6, 13)):
            p = int(rng.choice(members))
            members.append(len(docs))
            docs.append(emit(rng.dirichlet(80 * th_thread + 0.05), int(rng.integers(5, 13))))
            parents.append(p)
    sm = threads.ThreadSmoother(contexts=("parent", "op", "thread")).fit(
        docs, parents, base=_lda, seed=3, n_boot=100, final=False)
    assert sm.op_effect["lo"] <= 0 <= sm.op_effect["hi"], sm.op_effect


def _switch_harness(contexts, share, log_a, theta_rows, docs_tokens, parents, kept):
    """A ThreadSmoother with hand-set switch parameters and a fake two-topic model."""
    from types import SimpleNamespace
    sm = threads.ThreadSmoother(contexts=contexts, switch=True)
    sm.alpha = {c: 1.0 for c in contexts}
    sm._switch_x = np.array([log_a, 0.0, 0.0])            # A = exp(log_a), a_new = 1, rho = .5
    sm._switch_share = np.asarray(share, float)
    model = SimpleNamespace(doc_topic=np.asarray(theta_rows, float),
                            topic_word=np.array([[0.9, 0.1], [0.1, 0.9]]))
    corpus = SimpleNamespace(vocabulary=["a", "b"], kept_indices=list(kept),
                             doc_lengths=np.array([len(docs_tokens[k]) for k in kept]),
                             documents=lambda: [docs_tokens[k] for k in kept])
    return sm, model, corpus


def test_switch_drops_a_missing_contexts_pseudocount():
    # Parent share 0.01, OP share 0.99 of A = 100. A top-level reply has no OP, so it must
    # borrow with pseudo-count 1 (the parent's), not the full 100.
    docs = [["a"] * 20, ["b"]]
    sm, model, corpus = _switch_harness(("parent", "op"), [0.01, 0.99], np.log(100.0),
                                        [[1.0, 0.0], [0.0, 1.0]], docs, [-1, 0], [0, 1])
    out = sm.transform(model, corpus, [-1, 0])
    w = sm.inherit_weights[1]
    expect = w * (1 * np.array([0.0, 1.0]) + 1.0 * np.array([1.0, 0.0])) / 2 \
        + (1 - w) * np.array([0.0, 1.0])
    assert np.allclose(out[1], expect)


def test_switch_gives_an_empty_document_its_pure_context_mix():
    # Reply 1 is empty but its row is kept (no pruning); root 2 is empty and has no context.
    docs = [["a"] * 20, [], []]
    sm, model, corpus = _switch_harness(("parent",), [1.0], np.log(5.0),
                                        [[0.9, 0.1], [0.5, 0.5], [0.5, 0.5]], docs,
                                        [-1, 0, -1], [0, 1, 2])
    out = sm.transform(model, corpus, [-1, 0, -1])
    assert np.allclose(out[1], [0.9, 0.1])
    assert np.isnan(out[2]).all()


def test_switch_bootstrap_intervals_are_not_degenerate_with_one_context():
    # One context gives one share candidate; the intervals must still reflect re-optimizing
    # the continuous parameters on each bootstrap sample.
    docs, parents, _, _ = _simulate(inherit=0.5, n_threads=150, seed=19)
    sm = threads.ThreadSmoother(contexts=("parent",), switch=True).fit(
        docs, parents, base=_lda, seed=3, n_boot=50, final=False)
    assert sm.rho_ci[0] < sm.rho_ci[1]
