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
