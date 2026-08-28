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
    # Clear TOPICA_EXPERIMENTAL so an env var set by another test in the suite cannot leak into
    # the subprocess and un-gate the model (the gate must hold from a clean environment).
    import os

    env = {k: v for k, v in os.environ.items() if k != "TOPICA_EXPERIMENTAL"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.stdout.strip() == "GATED", out.stdout + out.stderr


def test_fit_shapes_and_readouts():
    docs, parents, cov, vocab = _threaded_corpus()
    m = topica.ReplyTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
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
    # the variational objective (not a true ELBO) should not decrease overall
    assert m.bound_history[-1] >= m.bound_history[0] - 1e-6
    # uncertainty is reported: prevalence SE (G x K-1) and a kappa CI bracketing kappa
    assert m.prevalence_se.shape == (G, K - 1)
    finite_se = m.prevalence_se[np.isfinite(m.prevalence_se)]
    assert np.all(finite_se >= 0)
    lo, hi = m.kappa_ci
    assert lo <= m.kappa + 1e-9 and hi >= m.kappa - 1e-9
    # top_words: all-topics mode returns K lists; single-topic mode returns one list
    allw = m.top_words()
    assert len(allw) == K and all(isinstance(t, list) for t in allw)
    assert isinstance(m.top_words(3, topic=0), list) and len(m.top_words(3, topic=0)) == 3
    # weights=True returns (word, prob) pairs
    ww = m.top_words(3, topic=0, weights=True)
    assert len(ww) == 3 and all(isinstance(w, str) and isinstance(p, float) for w, p in ww)


def test_topic_and_prevalence_recovery():
    docs, parents, cov, vocab = _threaded_corpus()
    m = topica.ReplyTM(2, em_iters=100, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
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
    m.fit(train, parents=parents, covariates=cov, covariate_names=["A", "B"])
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
    m.fit(corpus, parents=parents, covariates=cov, covariate_names=["A", "B"])
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


def test_kappa_is_fit_not_frozen():
    """Regression: κ/σ²/p0 must be ESTIMATED, never the init constants left frozen when EM
    converges inside the warm-up window (the bug that reported κ=0.3=1-a_init on any
    fast-converging corpus). With too few iterations to reach the field fit, they are NaN
    (unidentified), not the inits."""
    import math

    docs, parents, cov, vocab = _threaded_corpus()
    m = topica.ReplyTM(2, em_iters=100, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    # the field was fit: κ is a real estimate, not the 0.3 initializer, and σ²/p0 are not the 1.0 inits
    assert m.kappa != pytest.approx(0.3), "kappa is frozen at 1 - a_init (field never fit)"
    assert not (m.sigma2 == 1.0 and m.p0 == 1.0), "sigma2/p0 frozen at inits"
    assert np.isfinite(m.kappa) and np.isfinite(m.sigma2)
    # too few iterations to reach the warm-up-gated field fit → unidentified, reported as NaN
    m2 = topica.ReplyTM(2, em_iters=8, seed=13)
    m2.fit(docs, parents=parents)
    assert math.isnan(m2.kappa) and math.isnan(m2.sigma2), (m2.kappa, m2.sigma2)


def test_prevalence_se_cluster_robust():
    """The prevalence SE clusters on the thread root. A group with only one thread has no
    between-thread variation, so its SE is NaN (unidentified), not a spuriously tight 0."""
    import math

    # group 0: many threads; group 1: a SINGLE long thread (one cluster)
    docs, parents, cov = [], [], []
    rng = np.random.default_rng(0)
    for t in range(20):  # group-0 threads
        base = len(docs)
        for step in range(4):
            parents.append(-1 if step == 0 else base + step - 1)
            docs.append([f"a{rng.integers(5)}" for _ in range(20)])
            cov.append(0)
    base = len(docs)  # group-1: one thread only
    for step in range(12):
        parents.append(-1 if step == 0 else base + step - 1)
        docs.append([f"b{rng.integers(5)}" for _ in range(20)])
        cov.append(1)
    m = topica.ReplyTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["multi", "single"])
    se = m.prevalence_se
    assert np.all(np.isfinite(se[0])), "multi-thread group should have a finite SE"
    assert np.all(np.isnan(se[1])), "single-thread group SE must be NaN (unidentified)"


def test_kappa_within_its_ci():
    """The point estimate kappa must lie inside kappa_ci on every fit (regression: the CI used to
    force the fitted a_hat into the interval even when its own profile was below the cutoff, and the
    field mean was fit free while the CI profiled at m=0, so kappa could fall outside its CI)."""
    import math

    for seed in (1, 7, 13, 21):
        docs, parents, cov, vocab = _threaded_corpus(seed=seed, n_threads=30, depth=8)
        m = topica.ReplyTM(2, em_iters=80, seed=seed)
        m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
        lo, hi = m.kappa_ci
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
        assert lo - 1e-9 <= m.kappa <= hi + 1e-9, (seed, m.kappa, (lo, hi))


def test_persistence():
    """persistence() is the identifiable replacement for the boundary-prone kappa: it refits an
    uncoupled (no-tree) pass and regresses child η on parent η. Returns observed persistence
    (always identified), a reliability gate, and a measurement-error-corrected structural kappa."""
    import math

    docs, parents, cov, vocab = _threaded_corpus(n_threads=60, depth=8)
    m = topica.ReplyTM(4, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    r = m.persistence(bootstrap=200)
    for key in (
        "observed_persistence",
        "observed_ci",
        "reliability",
        "structural_kappa",
        "structural_kappa_ci",
    ):
        assert key in r
    # observed persistence is identified: finite, positive (replies track parents), CI brackets it
    a = r["observed_persistence"]
    assert math.isfinite(a) and a > 0
    lo, hi = r["observed_ci"]
    assert lo <= a <= hi
    # deterministic (seeded internal fit + bootstrap over sorted threads); NaN-aware compare
    r2 = m.persistence(bootstrap=200)

    def eq(x, y):
        return x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y))

    assert eq(r["observed_persistence"], r2["observed_persistence"])
    assert eq(r["structural_kappa"], r2["structural_kappa"])
    assert all(eq(x, y) for x, y in zip(r["structural_kappa_ci"], r2["structural_kappa_ci"]))


def test_persistence_requires_tree():
    """persistence() needs reply edges; a no-tree fit raises rather than returning garbage."""
    import warnings

    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.ReplyTM(3, em_iters=30, seed=13)
        m.fit(docs, parents=None)
    with pytest.raises(ValueError):
        m.persistence()


def test_coherence():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    m = topica.ReplyTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents)
    coh = m.coherence(10)
    assert coh.shape == (2,) and np.all(np.isfinite(coh))
    # coherence_type/texts are keyword-only; passing a corpus positionally is a clear TypeError,
    # not an opaque int-coercion error (the TopN gensim-muscle-memory guard).
    with pytest.raises(TypeError):
        m.coherence(docs)


def test_save_load_roundtrip(tmp_path):
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    m = topica.ReplyTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    p = str(tmp_path / "reply.topica")
    m.save(p)
    m2 = topica.ReplyTM.load(p)
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert np.array_equal(m.group_prevalence, m2.group_prevalence)
    assert m.kappa == m2.kappa and m.kappa_ci == m2.kappa_ci
    assert m.group_labels() == m2.group_labels()
    # coherence still works after load (the corpus snapshot round-tripped)
    assert np.allclose(m.coherence(10), m2.coherence(10))
    # the persistence() inputs round-trip too: η, ν, and the method's result
    assert np.array_equal(m.doc_eta, m2.doc_eta)
    assert np.array_equal(m.doc_topic_var, m2.doc_topic_var)
    r1, r2 = m.persistence(bootstrap=100), m2.persistence(bootstrap=100)
    assert r1["observed_persistence"] == r2["observed_persistence"]
    assert r1["observed_ci"] == r2["observed_ci"]


def test_inspect_integration():
    """The taught inspect API must work on ReplyTM (regression: it was misdispatched as a
    time-sliced model because topic_word/vocabulary were methods, not properties)."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    m = topica.ReplyTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents)
    table = topica.inspect.topic_table(m)
    assert len(table) == 2


def _branching_corpus(seed=1, n_threads=45, K=5, V=90, persistence=0.9,
                      max_depth=5, branch=(1, 4)):
    """Branching reply threads with a per-edge topic-persistence walk. Branching (not chains)
    so the depth-stratified parent-permutation placebo has candidates to shuffle among."""
    rng = np.random.default_rng(seed)
    beta = rng.dirichlet(np.ones(V) * 0.15, size=K)
    docs, parents, eta_of = [], [], {}

    def emit(parent_idx, eta):
        eta = persistence * eta + (1 - persistence) * rng.normal(size=K)
        theta = np.exp(eta)
        theta /= theta.sum()
        idx = len(docs)
        ids = rng.choice(K, size=int(rng.integers(6, 16)), p=theta)
        docs.append([f"w{int(rng.choice(V, p=beta[z]))}" for z in ids])
        parents.append(parent_idx)
        eta_of[idx] = eta
        return idx

    for _ in range(n_threads):
        frontier = [emit(-1, rng.normal(size=K))]
        for _d in range(max_depth):
            frontier = [emit(node, eta_of[node])
                        for node in frontier for _ in range(int(rng.integers(*branch)))]
            if not frontier:
                break
    return docs, parents


def test_reply_completion_planted_beats_and_placebo_shrinks():
    """evaluate.reply_completion: on a persistence-structured corpus the true tree beats the
    matched no-tree baseline on held-out leaf tokens (CI excludes zero), and the gain is
    attributed to the observed edge (the parent-permutation placebo advantage is much smaller)."""
    docs, parents = _branching_corpus(seed=1, persistence=0.92)
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=5, em_iters=60, seed=13, n_boot=300)
    assert res.beats_no_tree
    assert res.delta["no_tree"]["ci"][0] > 0.0
    # placebo is non-degenerate on this branching corpus, and shrinks the advantage
    assert res.settings["perm_changed_frac"] > 0.5
    assert res.delta["permuted"]["estimate"] < res.delta["no_tree"]["estimate"]
    assert res.n_eval_leaves > 0 and res.n_threads >= 2


def test_reply_completion_null_does_not_beat():
    """With no real parent-child persistence, the tree must not beat the no-tree baseline
    (fails safe): the model does not manufacture a predictive gain from tree structure alone."""
    docs, parents = _branching_corpus(seed=2, persistence=0.0)
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=5, em_iters=60, seed=13, n_boot=300)
    assert not res.beats_no_tree


def test_reply_completion_requires_branching_for_placebo():
    """The placebo warns on chain-like corpora, where every depth layer has one node so the
    within-thread parent permutation is a no-op."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=8)
    with pytest.warns(UserWarning, match="placebo"):
        topica.evaluate.reply_completion(
            docs, parents, num_topics=2, em_iters=40, seed=13, n_boot=100)


# ---------------------------------------------------------------------------
# transform: inference for new reply forests
# ---------------------------------------------------------------------------

def test_transform_shapes_and_simplex():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=6)
    m = topica.ReplyTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    theta = m.transform(docs, parents=parents, covariates=cov)
    assert theta.shape == (len(docs), 2)
    assert np.allclose(theta.sum(1), 1.0)
    assert (theta >= 0).all()


def test_transform_recovers_training_theta():
    """A single topological pass with the topics/field/anchors frozen is the exact structured
    mean-field fixed point, so transform on the training forest reproduces the fitted doc_topic."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=6)
    m = topica.ReplyTM(2, em_iters=80, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    theta = m.transform(docs, parents=parents, covariates=cov)
    dt = np.asarray(m.doc_topic)
    # Near-exact, not bit-identical: the fit's final E-step reads the one-iteration-lagged parent
    # lambda, while transform reads the converged parent, so transform is the more-converged pass.
    assert np.abs(theta - dt).mean() < 2e-3
    assert np.corrcoef(theta[:, 0], dt[:, 0])[0, 1] > 0.999


def test_transform_tree_couples_reply_to_parent():
    """The reply coupling must act at transform time: a reply's inferred mix tracks its parent's
    more under the tree than under the tree-blind (parents=None) pass."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=40, depth=6)
    m = topica.ReplyTM(2, em_iters=80, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    tree = m.transform(docs, parents=parents, covariates=cov)
    flat = m.transform(docs, covariates=cov)  # every doc a root
    # child-parent agreement in topic-0 proportion, over real edges
    edges = [(d, p) for d, p in enumerate(parents) if p >= 0]
    tree_gap = np.mean([abs(tree[d, 0] - tree[p, 0]) for d, p in edges])
    flat_gap = np.mean([abs(flat[d, 0] - flat[p, 0]) for d, p in edges])
    assert tree_gap < flat_gap


def test_transform_new_thread_and_defaults():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=6)
    m = topica.ReplyTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    # a fresh thread expressed in the training vocabulary
    new = [[vocab[i] for i in (0, 1, 2, 3)], [vocab[i] for i in (0, 1, 5, 6)]]
    th = m.transform(new, parents=[-1, 0], covariates=[0, 0])
    assert th.shape == (2, 2) and np.allclose(th.sum(1), 1.0)
    # covariates omitted -> across-group mean anchor (still a valid simplex)
    th2 = m.transform(new, parents=[-1, 0])
    assert np.allclose(th2.sum(1), 1.0)
    # out-of-vocabulary / empty documents survive as a prior-only row
    th3 = m.transform([["zzz_oov_token"]], parents=[-1])
    assert th3.shape == (1, 2) and np.allclose(th3.sum(1), 1.0)


def test_transform_requires_tree_fit():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=5)
    m = topica.ReplyTM(2, em_iters=30, seed=13)
    with pytest.warns(UserWarning):
        m.fit(docs)  # no tree -> field undefined
    with pytest.raises(ValueError, match="reply tree"):
        m.transform(docs, parents=parents)


def test_transform_validates_covariate_range():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=5)
    m = topica.ReplyTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    with pytest.raises(ValueError, match="group id"):
        m.transform(docs[:2], parents=[-1, 0], covariates=[0, 9])
    with pytest.raises(ValueError, match="out of range"):
        m.transform(docs[:2], parents=[-1, 5])


def test_transform_scores_through_eval_heldout():
    """transform lets ReplyTM ride the generic tree-blind held-out scorer (issue #828 half 2)."""
    docs, parents = _branching_corpus(seed=3, persistence=0.9)
    m = topica.ReplyTM(5, em_iters=60, seed=13)
    m.fit(docs, parents=parents)
    heldout = topica.evaluate.make_heldout(docs, seed=13)
    m2 = topica.ReplyTM(5, em_iters=60, seed=13)
    m2.fit(heldout.documents, parents=parents)
    res = topica.evaluate.eval_heldout(m2, heldout)
    assert np.isfinite(res.mean_per_doc_loglik)


# ---------------------------------------------------------------------------
# reply_completion off-the-shelf baselines (issue #828)
# ---------------------------------------------------------------------------

def test_reply_completion_offshelf_baselines():
    """LDA and STM comparators are fit on the same reduced corpus and scored through the identical
    protocol, so they land in delta with a thread-clustered CI alongside the tree."""
    docs, parents = _branching_corpus(seed=1, persistence=0.92)
    cov = [i % 2 for i in range(len(docs))]
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=5, covariates=cov, covariate_names=["g0", "g1"],
        baselines=("no_tree", "lda", "stm"), em_iters=50, seed=13, n_boot=200)
    for name in ("no_tree", "lda", "stm"):
        assert name in res.per_token_ll
        assert name in res.delta
        lo, hi = res.delta[name]["ci"]
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    assert res.settings["baselines"] == ["no_tree", "lda", "stm"]


def test_reply_completion_stm_needs_covariate():
    docs, parents = _branching_corpus(seed=1, persistence=0.9)
    with pytest.raises(ValueError, match="covariate"):
        topica.evaluate.reply_completion(
            docs, parents, num_topics=5, baselines=("stm",), em_iters=30, seed=13, n_boot=50)


def test_reply_completion_rejects_unknown_baseline():
    docs, parents = _branching_corpus(seed=1, persistence=0.9)
    with pytest.raises(ValueError, match="unknown baseline"):
        topica.evaluate.reply_completion(
            docs, parents, num_topics=5, baselines=("bogus",), em_iters=20, seed=13)


def _drop_inducing_corpus(n_threads=45):
    """Each eval leaf carries two corpus-common words plus one unique-rare word. Under
    min_count=2 the rare word is dropped from the vocabulary, so on the ~1/3 of leaves whose
    single seen token happens to be the rare one, the fixed-vocab Corpus empties and drops the
    leaf (an off-the-shelf baseline cannot score it) while ReplyTM still scores its in-vocab
    held tokens from a prior-only theta. This exercises the per-baseline pairing."""
    docs, parents = [], []
    for t in range(n_threads):
        docs.append(["cw0", "cw1", "cw2"]); parents.append(-1)
        root = len(docs) - 1
        docs.append(["cw0", "cw1", f"rare_{t}"]); parents.append(root)
    return docs, parents


def test_reply_completion_offshelf_does_not_shift_core_deltas():
    """Requesting an off-the-shelf baseline must not change the ReplyTM-vs-ReplyTM contrasts:
    each delta is paired over the leaves that baseline scored, so no_tree (which never drops a
    leaf) is invariant to whether lda is also requested, even when lda drops some leaves."""
    docs, parents = _drop_inducing_corpus()
    kw = dict(num_topics=3, em_iters=40, seed=13, n_boot=200, min_count=2)
    base = topica.evaluate.reply_completion(docs, parents, baselines=("no_tree",), **kw)
    with pytest.warns(UserWarning, match="off-the-shelf"):
        withlda = topica.evaluate.reply_completion(
            docs, parents, baselines=("no_tree", "lda"), **kw)
    assert base.delta["no_tree"]["estimate"] == withlda.delta["no_tree"]["estimate"]
    assert base.delta["no_tree"]["ci"] == withlda.delta["no_tree"]["ci"]
    # lda still produced a finite delta from the leaves it could score
    assert "lda" in withlda.delta and np.isfinite(withlda.delta["lda"]["estimate"])


def test_reply_completion_repr_shows_all_deltas():
    """The repr must surface beats_no_tree and every requested comparison, not only the
    flattering tree-no_tree line (a reader could otherwise misattribute a covariate/tool gain
    to the reply tree)."""
    docs, parents = _branching_corpus(seed=1, persistence=0.92)
    cov = [i % 2 for i in range(len(docs))]
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=5, covariates=cov,
        baselines=("no_tree", "lda", "stm"), em_iters=40, seed=13, n_boot=100)
    r = repr(res)
    assert "beats_no_tree=" in r
    assert "tree-no_tree=" in r and "tree-lda=" in r and "tree-stm=" in r


def test_transform_covariates_none_uses_mean_anchor():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=6)
    m = topica.ReplyTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    th = m.transform(docs, parents=parents)  # covariates omitted
    assert th.shape == (len(docs), 2) and np.allclose(th.sum(1), 1.0)


def test_transform_accepts_corpus_input():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=25, depth=5)
    m = topica.ReplyTM(2, em_iters=50, seed=13)
    m.fit(docs, parents=parents)
    from_lists = m.transform(docs, parents=parents)
    corpus = topica.Corpus.from_documents(docs)
    from_corpus = m.transform(corpus, parents=parents)
    assert np.allclose(from_lists, from_corpus)


def test_transform_rejects_negative_covariate():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=5)
    m = topica.ReplyTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    with pytest.raises(ValueError, match="group id"):
        m.transform(docs[:2], parents=[-1, 0], covariates=[-1, 0])
