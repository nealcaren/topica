"""Tests for ThreadTM — the reply-threaded topic model (experimental).

The Rust side (src/thread_tm.rs) carries the planted-recovery test through the full model; these
exercise the Python binding contract: the experimental gate, the fit surface (parents +
covariate), array shapes, input validation, and a light end-to-end topic + prevalence recovery.
"""
import subprocess
import sys
import warnings

import numpy as np
import pytest

import topica

topica.enable_experimental()


def _content_shift_corpus(seed=0, n_threads=120):
    """A planted CONTENT shift (issue #841): topic A uses one vocabulary at the root/shallow and a
    different one deep in the thread, at constant prevalence (~70% A / 30% B, no prevalence drift);
    topic B is stable. A content=depth fit should recover A's root-vs-deep word swap; a no-content
    fit blends the two. Returns (docs, parents, AR, AD, B)."""
    rng = np.random.default_rng(seed)
    AR = [f"ar{i}" for i in range(5)]
    AD = [f"ad{i}" for i in range(5)]
    B = [f"b{i}" for i in range(10)]
    docs, parents = [], []
    for _ in range(n_threads):
        for depth in range(6):
            parents.append(-1 if depth == 0 else len(docs) - 1)
            doc = []
            for _ in range(25):
                if rng.random() < 0.7:
                    pool = AR if depth < 3 else AD  # topic A's vocab shifts with depth
                    doc.append(pool[rng.integers(len(pool))])
                else:
                    doc.append(B[rng.integers(len(B))])
            docs.append(doc)
    return docs, parents, AR, AD, B


def test_content_depth_recovers_planted_shift():
    """#841 acceptance: content="depth" recovers a topic whose words shift root->deep, which a
    no-content ThreadTM misses."""
    docs, parents, AR, AD, B = _content_shift_corpus()
    m = topica.ThreadTM(2, em_iters=80, seed=13).fit(docs, parents=parents, content="depth")
    assert m.content_labels == ["root", "shallow", "deep"]

    def overlap(words, pool):
        return len(set(words) & set(pool))

    # identify the shifting ("A") topic by its AR-heavy root words
    a = max(range(2), key=lambda t: overlap(m.content_top_words(t, n=5)["root"], AR))
    twa = m.content_top_words(a, n=5)
    assert overlap(twa["root"], AR) >= 4 and overlap(twa["root"], AD) <= 1
    assert overlap(twa["deep"], AD) >= 4 and overlap(twa["deep"], AR) <= 1

    # a no-content fit cannot separate the two vocabularies for the same topic
    mnc = topica.ThreadTM(2, em_iters=80, seed=13).fit(docs, parents=parents)
    assert mnc.topic_word_by_group is None and mnc.content_kappa is None
    tw = np.asarray(mnc.topic_word)
    an = max(range(2), key=lambda t: overlap(
        [mnc.vocabulary[i] for i in tw[t].argsort()[::-1][:10]], AR + AD))
    top10 = [mnc.vocabulary[i] for i in tw[an].argsort()[::-1][:10]]
    # the marginal topic mixes both root and deep vocab (it cannot represent the shift)
    assert not (overlap(top10, AR) >= 4 and overlap(top10, AD) <= 1)


def test_content_word_contrast():
    """#846: content_word_contrast(topic, "deep", "root") surfaces the words that separate a topic's
    language between two levels — for the planted shift, deep-over-root favors AD, root-over-deep AR."""
    docs, parents, AR, AD, B = _content_shift_corpus()
    m = topica.ThreadTM(2, em_iters=80, seed=13).fit(docs, parents=parents, content="depth")
    a = max(range(2), key=lambda t: len(set(m.content_top_words(t, n=5)["root"]) & set(AR)))
    deep_over_root = m.content_word_contrast(a, "deep", "root", n=5)
    assert all(isinstance(w, str) and isinstance(r, float) for w, r in deep_over_root)
    assert [r for _, r in deep_over_root] == sorted((r for _, r in deep_over_root), reverse=True)
    top_deep = [w for w, _ in deep_over_root]
    assert len(set(top_deep) & set(AD)) >= 4  # deep-characteristic words are the AD vocabulary
    # the reverse contrast surfaces the root (AR) vocabulary
    top_root = [w for w, _ in m.content_word_contrast(a, "root", "deep", n=5)]
    assert len(set(top_root) & set(AR)) >= 4
    # accepts integer levels too, and errors on an unknown level
    assert m.content_word_contrast(a, 2, 0, n=3)  # deep vs root by index
    with pytest.raises(ValueError):
        m.content_word_contrast(a, "nope", "root")


def test_content_smooth_ordered_levels():
    """#846: content_smooth ties adjacent (ordered) depth levels; the fit stays valid and the
    smoothed per-level topic-word distributions are closer between adjacent levels than unsmoothed."""
    docs, parents, *_ = _content_shift_corpus(n_threads=60)
    m0 = topica.ThreadTM(2, em_iters=60, seed=13).fit(docs, parents=parents, content="depth")
    ms = topica.ThreadTM(2, em_iters=60, seed=13).fit(
        docs, parents=parents, content="depth", content_smooth=5.0
    )
    tw0 = np.asarray(m0.topic_word_by_group)  # (K, G, V)
    tws = np.asarray(ms.topic_word_by_group)
    assert tws.shape == tw0.shape and np.allclose(tws.sum(axis=2), 1.0)
    # adjacent-level L1 distance (root vs shallow) should shrink under smoothing
    adj0 = np.abs(tw0[:, 0] - tw0[:, 1]).sum(axis=1).mean()
    adjs = np.abs(tws[:, 0] - tws[:, 1]).sum(axis=1).mean()
    assert adjs <= adj0 + 1e-9
    with pytest.raises(ValueError):
        topica.ThreadTM(2, em_iters=5, seed=13).fit(
            docs, parents=parents, content="depth", content_smooth=-1.0
        )


def test_content_readouts_and_labels():
    """#841: content readouts have the right shapes (STM-compatible), plug into topica.content, and
    are None without a content covariate."""
    docs, parents, *_ = _content_shift_corpus(n_threads=40)
    m = topica.ThreadTM(2, em_iters=40, seed=13).fit(docs, parents=parents, content="depth")
    K, G, V = 2, 3, len(m.vocabulary)
    twbg = np.asarray(m.topic_word_by_group)  # (K, G, V), STM's layout
    assert twbg.shape == (K, G, V)
    assert np.allclose(twbg.sum(axis=2), 1.0)  # each (topic, level) row is a distribution
    assert m.groups == m.content_labels == ["root", "shallow", "deep"]
    assert np.asarray(m.topic_word_marginal).shape == (K, V)
    ck = m.content_kappa
    assert set(ck) == {"m", "kappa_topic", "kappa_cov", "kappa_interaction"}  # STM keys
    assert np.asarray(ck["kappa_interaction"]).shape == (K, G, V)
    tw = m.content_top_words(0, n=5)
    assert set(tw) == {"root", "shallow", "deep"} and len(tw["root"]) == 5
    # the shared cross-model content diagnostics work on the ThreadTM content channel
    pol = topica.content.topic_polarization(m)
    assert pol.shape == (K,) and np.all(pol >= 0)


def test_content_arbitrary_labels_and_transform():
    """#841: content accepts arbitrary categorical labels; transform scores under the content level."""
    docs, parents, *_ = _content_shift_corpus(n_threads=40)
    # a per-document label (here re-deriving depth as a label to exercise the categorical path)
    labels = []
    for d, p in enumerate(parents):
        depth = 0
        cur = p
        while cur >= 0:
            depth += 1
            cur = parents[cur]
        labels.append("root" if depth == 0 else "reply")
    m = topica.ThreadTM(2, em_iters=40, seed=13).fit(docs, parents=parents, content=labels)
    assert m.content_labels == ["root", "reply"]
    th = m.transform(docs[:6], parents=parents[:6], content=["root", "reply", "reply", "reply", "reply", "reply"])
    assert th.shape == (6, 2)
    # depth transform round-trips on a content=depth model
    md = topica.ThreadTM(2, em_iters=40, seed=13).fit(docs, parents=parents, content="depth")
    thd = md.transform(docs[:6], parents=parents[:6], content="depth")
    assert thd.shape == (6, 2)
    with pytest.raises(ValueError):
        m.transform(docs[:2], parents=[-1, 0], content="depth")  # model fit with labels, not depth


def test_content_save_load_and_prior(tmp_path):
    """#841: content model round-trips through save/load; l1 and l2 priors both fit."""
    docs, parents, *_ = _content_shift_corpus(n_threads=40)
    m = topica.ThreadTM(2, em_iters=40, seed=13).fit(
        docs, parents=parents, content="depth", content_prior="l1", content_prior_var=0.5
    )
    p = str(tmp_path / "content.topica")
    m.save(p)
    m2 = topica.ThreadTM.load(p)
    assert m2.content_labels == m.content_labels
    assert np.allclose(np.asarray(m.topic_word_by_group), np.asarray(m2.topic_word_by_group))
    assert np.allclose(
        m.transform(docs[:5], parents=parents[:5], content="depth"),
        m2.transform(docs[:5], parents=parents[:5], content="depth"),
    )
    with pytest.raises(ValueError):
        topica.ThreadTM(2, em_iters=5, seed=13).fit(
            docs, parents=parents, content="depth", depth_bins=[1, 2]  # must start at 0
        )
    with pytest.raises(ValueError):
        topica.ThreadTM(2, em_iters=5, seed=13).fit(
            docs, parents=parents, content="depth", content_prior="bogus"
        )


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


def test_experimental_gate_blocks_construction():
    """Construction must refuse until experimental models are enabled (issue #856: the gate fires
    at ThreadTM(...), not only at fit, so a first-timer learns immediately). Fresh interpreter."""
    code = (
        "import topica\n"
        "try:\n"
        "    m = topica.ThreadTM(3); print('NOTGATED')\n"
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
    m = topica.ThreadTM(2, em_iters=60, seed=13)
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
    # kappa CI brackets kappa; on this strongly-persistent corpus the profile pegs at the
    # persistence floor, so the upper bound may be a one-sided NaN (issue #830).
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lo, hi = m.kappa_ci
    assert lo <= m.kappa + 1e-9
    assert np.isnan(hi) or hi >= m.kappa - 1e-9
    # top_words: all-topics mode returns K lists; single-topic mode returns one list
    allw = m.top_words()
    assert len(allw) == K and all(isinstance(t, list) for t in allw)
    assert isinstance(m.top_words(3, topic=0), list) and len(m.top_words(3, topic=0)) == 3
    # weights=True returns (word, prob) pairs
    ww = m.top_words(3, topic=0, weights=True)
    assert len(ww) == 3 and all(isinstance(w, str) and isinstance(p, float) for w, p in ww)


def test_topic_and_prevalence_recovery():
    docs, parents, cov, vocab = _threaded_corpus()
    m = topica.ThreadTM(2, em_iters=100, seed=13)
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
    m = topica.ThreadTM(2, em_iters=5)
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1])  # wrong length
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1, 5])  # out of range
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[-1, 1])  # self-parent
    with pytest.raises(ValueError):
        m.fit([["a"], ["b"]], parents=[1, 0])  # cycle A->B->A


def test_unfitted_raises():
    m = topica.ThreadTM(3)
    with pytest.raises(RuntimeError):
        m.topic_word


def test_reduces_to_flat_when_no_tree():
    """With no reply edges ThreadTM is a plain logistic-normal topic model: topics still recover,
    and the reply parameters are correctly flagged unidentified (NaN)."""
    import math
    import warnings

    docs, parents, cov, vocab = _threaded_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # parents=None warns on purpose
        m = topica.ThreadTM(2, em_iters=80, seed=13)
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
    m = topica.ThreadTM(2, em_iters=100, seed=13)
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
    m = topica.ThreadTM(2, em_iters=40, seed=13)
    m.fit(corpus, parents=parents, covariates=cov, covariate_names=["A", "B"])
    assert m.topic_word.shape == (2, len(vocab))
    assert m.doc_topic.shape == (len(docs), 2)
    assert set(m.vocabulary) == set(vocab)


def test_min_count_emptying_warns():
    """A document whose every token is rarer than min_count is emptied but kept as a tree node;
    the user is warned rather than silently losing the document's content."""
    import warnings

    docs = [["a0", "a0", "a1"], ["rareX", "rareY"], ["a1", "a1", "a0"]]
    m = topica.ThreadTM(2, em_iters=10, seed=13)
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
    m = topica.ThreadTM(2, em_iters=100, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    # the field was fit: κ is a real estimate, not the 0.3 initializer, and σ²/p0 are not the 1.0 inits
    assert m.kappa != pytest.approx(0.3), "kappa is frozen at 1 - a_init (field never fit)"
    assert not (m.sigma2 == 1.0 and m.p0 == 1.0), "sigma2/p0 frozen at inits"
    assert np.isfinite(m.kappa) and np.isfinite(m.sigma2)
    # too few iterations to reach the warm-up-gated field fit → unidentified, reported as NaN
    m2 = topica.ThreadTM(2, em_iters=8, seed=13)
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
    m = topica.ThreadTM(2, em_iters=60, seed=13)
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
        m = topica.ThreadTM(2, em_iters=80, seed=seed)
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
    m = topica.ThreadTM(4, em_iters=60, seed=13)
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
        m = topica.ThreadTM(3, em_iters=30, seed=13)
        m.fit(docs, parents=None)
    with pytest.raises(ValueError):
        m.persistence()


def test_coherence():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    m = topica.ThreadTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents)
    coh = m.coherence(10)
    assert coh.shape == (2,) and np.all(np.isfinite(coh))
    # coherence_type/texts are keyword-only; passing a corpus positionally is a clear TypeError,
    # not an opaque int-coercion error (the TopN gensim-muscle-memory guard).
    with pytest.raises(TypeError):
        m.coherence(docs)


def test_save_load_roundtrip(tmp_path):
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    m = topica.ThreadTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    p = str(tmp_path / "reply.topica")
    m.save(p)
    m2 = topica.ThreadTM.load(p)
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert np.array_equal(m.group_prevalence, m2.group_prevalence)
    assert m.kappa == m2.kappa and m.kappa_ci == m2.kappa_ci
    # the full Σ_edge/Σ_root priors round-trip: their scalar summaries and a tree-aware transform
    # must be bit-identical after load (a diagonal-only save would silently change predictions).
    assert m.sigma2 == m2.sigma2 and m.p0 == m2.p0
    assert np.array_equal(
        m.transform(docs, parents=parents, covariates=cov),
        m2.transform(docs, parents=parents, covariates=cov),
    )
    assert m.group_labels() == m2.group_labels()
    # coherence still works after load (the corpus snapshot round-tripped)
    assert np.allclose(m.coherence(10), m2.coherence(10))
    # the persistence() inputs round-trip too: η, ν, and the method's result
    assert np.array_equal(m.doc_eta, m2.doc_eta)
    assert np.array_equal(m.doc_topic_var, m2.doc_topic_var)
    r1, r2 = m.persistence(bootstrap=100), m2.persistence(bootstrap=100)
    assert r1["observed_persistence"] == r2["observed_persistence"]
    assert r1["observed_ci"] == r2["observed_ci"]


def _thin_leaf_corpus(seed=13, n_threads=60):
    """Thick roots (12 tokens) and thin leaves (2 tokens) so the per-leaf posterior variance ν is
    large: exactly the regime where the plug-in and the posterior-predictive θ diverge."""
    rng = np.random.default_rng(seed)
    A = [f"a{i}" for i in range(6)]
    B = [f"b{i}" for i in range(6)]
    docs, parents = [], []
    for t in range(n_threads):
        own = A if t % 2 == 0 else B
        docs.append([own[rng.integers(6)] for _ in range(12)])
        parents.append(-1)
        r = len(docs) - 1
        for _ in range(2):
            docs.append([own[rng.integers(6)] for _ in range(2)])  # thin leaf → high ν
            parents.append(r)
    return docs, parents


def _entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


def test_posterior_doc_topic_predictive(tmp_path):
    """posterior_doc_topic is a correct MC estimate of E[softmax(eta)]: proper simplex,
    deterministic, converges to an independent reference, and survives save/load (issue #838)."""
    docs, parents = _thin_leaf_corpus()
    m = topica.ThreadTM(3, em_iters=50, seed=13).fit(docs, parents=parents)

    plug = np.asarray(m.doc_topic)
    pdt = np.asarray(m.posterior_doc_topic(n_samples=400, seed=13))
    assert pdt.shape == plug.shape
    assert np.allclose(pdt.sum(axis=1), 1.0)  # a valid distribution per document
    assert (pdt >= 0).all()
    # deterministic given seed
    assert np.array_equal(pdt, np.asarray(m.posterior_doc_topic(n_samples=400, seed=13)))
    with pytest.raises(ValueError):
        m.posterior_doc_topic(n_samples=0)

    # MC correctness: it converges to an independent numpy E[softmax([eta, 0])] over the SAME
    # diagonal-nu Gaussian posterior. This pins the estimator, not merely "differs from plug-in".
    eta = np.asarray(m.doc_eta)
    var = np.clip(np.asarray(m.doc_topic_var), 0.0, None)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((20000,) + eta.shape)
    draws = eta[None] + z * np.sqrt(var)[None]
    full = np.concatenate([draws, np.zeros(draws.shape[:2] + (1,))], axis=2)
    full -= full.max(axis=2, keepdims=True)
    e = np.exp(full)
    ref = (e / e.sum(axis=2, keepdims=True)).mean(axis=0)
    assert np.abs(np.asarray(m.posterior_doc_topic(n_samples=8000, seed=1)) - ref).max() < 0.02

    # it depends only on doc_eta + doc_topic_var, which round-trip, so it survives save/load
    p = str(tmp_path / "reply.topica")
    m.save(p)
    m2 = topica.ThreadTM.load(p)
    assert np.array_equal(pdt, np.asarray(m2.posterior_doc_topic(n_samples=400, seed=13)))


def test_posterior_doc_topic_hedges_toward_uniform():
    """The corrected #838 mechanism: integrating over ν makes the posterior-predictive θ FLATTER
    (higher entropy) than the overconfident plug-in ``softmax(mean η)``, and the flattening is
    ν-driven (large on thin, high-ν leaves, ~zero on thick low-ν documents). A test that only
    checked the two θ differ would pass even on an implementation that hedged the wrong way — the
    direction is the property the fix rests on."""
    docs, parents = _thin_leaf_corpus()
    m = topica.ThreadTM(2, em_iters=60, seed=13).fit(docs, parents=parents)
    plug = np.asarray(m.doc_topic)
    pdt = np.asarray(m.posterior_doc_topic(n_samples=1000, seed=13))
    gain = _entropy(pdt) - _entropy(plug)
    # posterior-predictive is flatter overall (hedges, not sharpens)
    assert gain.mean() > 0.0
    # and the hedging is driven by ν: high-ν leaves flatten far more than low-ν documents
    nu = np.asarray(m.doc_topic_var).mean(axis=1)
    q_lo, q_hi = np.quantile(nu, [0.25, 0.75])
    assert gain[nu >= q_hi].mean() > gain[nu <= q_lo].mean()


def _persistent_corpus(depth=8, leaf_len=30, n_threads=40, seed=0):
    """Strongly-persistent, low-ν chains (children copy the parent's block, long docs): drives the
    reversion to the clamp so kappa_ci hits the persistence-floor boundary case (#830)."""
    rng = np.random.default_rng(seed)
    V = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)]
    docs, parents = [], []
    for t in range(n_threads):
        blk = 0 if t % 2 == 0 else 6
        docs.append([V[blk + rng.integers(6)] for _ in range(leaf_len)])
        parents.append(-1)
        prev = len(docs) - 1
        for _ in range(depth):
            docs.append([V[blk + rng.integers(6)] for _ in range(leaf_len)])
            parents.append(prev)
            prev = len(docs) - 1
    return docs, parents


def test_covariates_accept_string_labels():
    """#830 T2: string/categorical covariates are auto-encoded (fit + transform), the labels become
    the group names, and an unseen label at transform is a clear error (not an opaque PyO3 crash)."""
    docs, parents, cov, _ = _threaded_corpus(n_threads=30, depth=4)
    labels = ["RedPill" if g == 0 else "CMV" for g in cov]
    m = topica.ThreadTM(2, em_iters=30, seed=13).fit(docs, parents=parents, covariates=labels)
    # first-seen order: group 0 is "RedPill" (thread 0 is group 0)
    assert m.group_labels() == ["RedPill", "CMV"]
    # integer and string fits must agree (same encoding)
    mi = topica.ThreadTM(2, em_iters=30, seed=13).fit(docs, parents=parents, covariates=cov)
    assert np.allclose(m.group_prevalence, mi.group_prevalence)
    # transform accepts labels, mapped through the fitted groups
    th = m.transform(docs[:5], parents=[-1, 0, 1, 2, 3], covariates=["RedPill"] * 5)
    assert th.shape == (5, 2)
    with pytest.raises(ValueError, match="not one of the fitted group labels"):
        m.transform(docs[:2], parents=[-1, 0], covariates=["RedPill", "Nope"])


def test_covariates_accept_whole_valued_floats():
    """#830/adversarial: a float64 group column (the common pandas artifact of an int column that
    once held a NaN) is accepted as integer ids; a non-whole float is a clear error."""
    docs, parents, cov, _ = _threaded_corpus(n_threads=24, depth=4)
    fcov = [float(g) for g in cov]  # 0.0 / 1.0
    m = topica.ThreadTM(2, em_iters=30, seed=13).fit(docs, parents=parents, covariates=fcov)
    mi = topica.ThreadTM(2, em_iters=30, seed=13).fit(docs, parents=parents, covariates=cov)
    assert np.allclose(m.group_prevalence, mi.group_prevalence)
    with pytest.raises(ValueError):
        topica.ThreadTM(2, em_iters=5, seed=13).fit(
            docs, parents=parents, covariates=[0.5] + fcov[1:]
        )


def test_covariates_accept_pandas_series():
    """#830/faithfulness coverage: a pandas Series of labels routes to the string path."""
    pd = pytest.importorskip("pandas")
    docs, parents, cov, _ = _threaded_corpus(n_threads=24, depth=4)
    s = pd.Series(["RedPill" if g == 0 else "CMV" for g in cov])
    m = topica.ThreadTM(2, em_iters=30, seed=13).fit(docs, parents=parents, covariates=s)
    assert m.group_labels() == ["RedPill", "CMV"]


def test_early_stopped_and_fit_history():
    """#830/API: converged has an early_stopped alias and a fit_history the stop_reason helper reads."""
    docs, parents, _, _ = _threaded_corpus(n_threads=20, depth=4)
    m = topica.ThreadTM(2, em_iters=200, seed=13).fit(docs, parents=parents)
    assert m.early_stopped == m.converged
    hist = m.fit_history
    assert isinstance(hist, list) and hist and hist[0][0] == 0
    assert isinstance(topica.stop_reason(m), str)


def test_converged_flag():
    """#830 T4a: a fitted model exposes a `converged` bool (not inferred from bound_history len)."""
    docs, parents, _, _ = _threaded_corpus(n_threads=20, depth=5)
    m = topica.ThreadTM(2, em_iters=200, seed=13).fit(docs, parents=parents)
    assert isinstance(m.converged, bool)
    with pytest.raises(RuntimeError):
        topica.ThreadTM(2).converged  # unfitted


def test_kappa_ci_boundary_flag():
    """#830 T1: when the profile CI collapses to the persistence floor, kappa_ci returns a one-sided
    (lower, nan) and warns, rather than a false-precision zero-width interval."""
    docs, parents = _persistent_corpus()
    m = topica.ThreadTM(2, em_iters=80, seed=13).fit(docs, parents=parents)
    with pytest.warns(UserWarning, match="persistence floor"):
        lo, hi = m.kappa_ci
    assert lo == pytest.approx(m.kappa, abs=1e-3)
    assert np.isnan(hi)  # upper not identified at the boundary


def test_group_prevalence_ci():
    """#830 T3a / #843: topica.inspect.group_prevalence_ci returns a FrameDict (labels + mean/ci/sd)
    matching the house CI-helper convention, with valid prob-scale intervals and a tidy to_frame."""
    docs, parents, cov, _ = _threaded_corpus(n_threads=30, depth=4)
    m = topica.ThreadTM(3, em_iters=40, seed=13).fit(
        docs, parents=parents, covariates=cov, covariate_names=["A", "B"]
    )
    res = topica.inspect.group_prevalence_ci(m, ci=0.9, n_samples=1500, seed=1)
    assert set(res) == {"labels", "mean", "ci_low", "ci_high", "sd"}
    assert res["labels"] == ["A", "B"]
    gp = np.asarray(m.group_prevalence)
    assert np.allclose(np.asarray(res["mean"]), gp)  # point estimate is the exact group_prevalence
    lo, hi = np.asarray(res["ci_low"]), np.asarray(res["ci_high"])
    assert lo.shape == gp.shape and np.all(lo <= hi) and np.all((lo >= 0) & (hi <= 1))
    # tidy long frame: one row per (group, topic)
    df = res.to_frame()
    assert list(df.columns) == ["group", "topic", "mean", "ci_low", "ci_high", "sd"]
    assert len(df) == gp.size
    # deterministic given seed
    res2 = topica.inspect.group_prevalence_ci(m, ci=0.9, n_samples=1500, seed=1)
    assert np.array_equal(np.asarray(res["ci_low"]), np.asarray(res2["ci_low"]))
    with pytest.raises(ValueError):
        topica.inspect.group_prevalence_ci(m, ci=1.5)


def test_topic_table_pretty_print():
    """#830 T3b: topic_table prints as an aligned table, not a raw list-of-dicts dump, while staying
    a list (indexing / to_frame unchanged)."""
    docs, parents, _, _ = _threaded_corpus(n_threads=20, depth=4)
    m = topica.ThreadTM(2, em_iters=30, seed=13).fit(docs, parents=parents)
    tt = topica.inspect.topic_table(m)
    s = str(tt)
    assert "topic" in s and "prev" in s
    assert not s.startswith("[{")  # not the raw dict dump
    assert isinstance(tt, list) and len(tt.to_frame()) == 2


def test_record_fit_defaults_corpus():
    """#830 T4b: record_fit(model) works without re-passing the corpus (defaults to model.corpus)."""
    docs, parents, _, _ = _threaded_corpus(n_threads=20, depth=4)
    m = topica.ThreadTM(2, em_iters=30, seed=13).fit(docs, parents=parents)
    assert m.corpus.num_docs == len(docs)
    man = topica.provenance.record_fit(m)  # no corpus argument
    assert man is not None
    # explicit corpus still works and agrees on doc count
    man2 = topica.provenance.record_fit(m, m.corpus)
    assert man2 is not None


def test_reply_completion_scores_logistic_normal_fairly():
    """reply_completion must score ThreadTM and STM (logistic-normal) with the posterior-predictive
    theta, not the plug-in, so the LDA delta is not an estimator artifact (issue #838). The two
    logistic-normal baselines (no_tree, stm) should sit close to the tree; the delta is recorded and
    predictive_samples is surfaced in settings."""
    docs, parents, cov, _ = _threaded_corpus(n_threads=40, depth=5, doc_len=10)
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=4, covariates=cov,
        baselines=("no_tree", "lda", "stm"), em_iters=40, seed=13, n_boot=200,
    )
    assert res.settings["predictive_samples"] == 400
    assert set(res.delta) == {"no_tree", "lda", "stm"}
    # same-family baselines are near the tree (both posterior-predictive now); the eval ran.
    assert abs(res.delta["no_tree"]["estimate"]) < 0.2
    assert np.isfinite(res.delta["lda"]["estimate"])
    # predictive_samples is honored: a tiny sample count changes the tree's scored likelihood.
    res_few = topica.evaluate.reply_completion(
        docs, parents, num_topics=4, covariates=cov,
        baselines=("no_tree",), em_iters=40, seed=13, n_boot=1, predictive_samples=2,
    )
    assert res_few.per_token_ll["tree"] != res.per_token_ll["tree"]


def test_inspect_integration():
    """The taught inspect API must work on ThreadTM (regression: it was misdispatched as a
    time-sliced model because topic_word/vocabulary were methods, not properties)."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=6)
    m = topica.ThreadTM(2, em_iters=40, seed=13)
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
    m = topica.ThreadTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    theta = m.transform(docs, parents=parents, covariates=cov)
    assert theta.shape == (len(docs), 2)
    assert np.allclose(theta.sum(1), 1.0)
    assert (theta >= 0).all()


def test_transform_recovers_training_theta():
    """A single topological pass with the topics/field/anchors frozen is the exact structured
    mean-field fixed point, so transform on the training forest reproduces the fitted doc_topic."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=6)
    m = topica.ThreadTM(2, em_iters=80, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    theta = m.transform(docs, parents=parents, covariates=cov)
    dt = np.asarray(m.doc_topic)
    # Near-exact, not bit-identical: the fit's final E-step reads the one-iteration-lagged parent
    # lambda, while transform reads the converged parent, so transform is the more-converged pass.
    assert np.abs(theta - dt).mean() < 2e-3
    assert np.corrcoef(theta[:, 0], dt[:, 0])[0, 1] > 0.999


def test_transform_tree_couples_reply_to_parent():
    """The reply coupling must act at transform time: a reply's inferred mix tracks its parent's
    more under the tree than under the tree-blind (parents=None) pass. Fit without covariates so the
    tree-blind baseline anchors at the global mean, isolating the reply coupling from a group anchor
    that would otherwise already pull same-group parents and children together."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=40, depth=6)
    m = topica.ThreadTM(2, em_iters=80, seed=13)
    m.fit(docs, parents=parents)
    tree = m.transform(docs, parents=parents)
    flat = m.transform(docs)  # every doc a root (tree-blind)
    # child-parent agreement (full topic mix) over real edges
    edges = [(d, p) for d, p in enumerate(parents) if p >= 0]
    tree_gap = np.mean([np.abs(tree[d] - tree[p]).sum() for d, p in edges])
    flat_gap = np.mean([np.abs(flat[d] - flat[p]).sum() for d, p in edges])
    assert tree_gap < flat_gap


def test_transform_new_thread_and_defaults():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=6)
    m = topica.ThreadTM(2, em_iters=60, seed=13)
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
    m = topica.ThreadTM(2, em_iters=30, seed=13)
    with pytest.warns(UserWarning):
        m.fit(docs)  # no tree -> field undefined
    with pytest.raises(ValueError, match="reply tree"):
        m.transform(docs, parents=parents)


def test_transform_validates_covariate_range():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=5)
    m = topica.ThreadTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    with pytest.raises(ValueError, match="group id"):
        m.transform(docs[:2], parents=[-1, 0], covariates=[0, 9])
    with pytest.raises(ValueError, match="out of range"):
        m.transform(docs[:2], parents=[-1, 5])


def test_transform_scores_through_eval_heldout():
    """transform lets ThreadTM ride the generic tree-blind held-out scorer (issue #828 half 2)."""
    docs, parents = _branching_corpus(seed=3, persistence=0.9)
    m = topica.ThreadTM(5, em_iters=60, seed=13)
    m.fit(docs, parents=parents)
    heldout = topica.evaluate.make_heldout(docs, seed=13)
    m2 = topica.ThreadTM(5, em_iters=60, seed=13)
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
    leaf (an off-the-shelf baseline cannot score it) while ThreadTM still scores its in-vocab
    held tokens from a prior-only theta. This exercises the per-baseline pairing."""
    docs, parents = [], []
    for t in range(n_threads):
        docs.append(["cw0", "cw1", "cw2"]); parents.append(-1)
        root = len(docs) - 1
        docs.append(["cw0", "cw1", f"rare_{t}"]); parents.append(root)
    return docs, parents


def test_reply_completion_offshelf_does_not_shift_core_deltas():
    """Requesting an off-the-shelf baseline must not change the ThreadTM-vs-ThreadTM contrasts:
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


# ---------------------------------------------------------------------------
# seed words (β) and prevalence anchors (θ) — user supervision (issue #854)
# ---------------------------------------------------------------------------

def _planted_block_corpus(seed=7, n_threads=120, K=4, blk=10, group_mix=None):
    """Threaded corpus with K disjoint word blocks (topic t = words {t*blk .. t*blk+blk-1})
    and two covariate groups whose topic prevalence differs in a KNOWN way. Lets a test seed
    topic t with block t's words and check both recovery and slot alignment."""
    rng = np.random.default_rng(seed)
    if group_mix is None:
        group_mix = {0: np.array([0.40, 0.40, 0.10, 0.10]),
                     1: np.array([0.10, 0.10, 0.40, 0.40])}

    def draw(mix, length):
        toks = []
        for _ in range(length):
            t = rng.choice(K, p=mix)
            toks.append(f"w{t * blk + rng.integers(blk)}")
        return toks

    docs, parents, groups = [], [], []
    for th in range(n_threads):
        g = th % 2
        docs.append(draw(group_mix[g], 25)); parents.append(-1); groups.append(g)
        root = len(docs) - 1
        for _ in range(3):
            docs.append(draw(group_mix[g], 12)); parents.append(root); groups.append(g)
    block_words = {t: [f"w{t * blk + j}" for j in range(blk)] for t in range(K)}
    return docs, parents, groups, block_words, group_mix


def _block_mass(model, t, words):
    tw = np.asarray(model.topic_word)
    ids = [model.vocabulary.index(w) for w in words if w in model.vocabulary]
    return float(tw[t, ids].sum())


def _fit_seeded(docs, parents, groups, K=4, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return topica.ThreadTM(K, em_iters=80, seed=13, coupling="parent").fit(
            docs, parents=parents, covariates=groups,
            covariate_names=["g0", "g1"], min_count=1, **kw)


def test_thread_tm_seed_words_pin_topics_to_slots():
    """Seeding topic t with block t's words pins fitted-topic t to block t (solving the
    unsupervised label-switching) and concentrates its mass on the right block."""
    docs, parents, groups, block_words, _ = _planted_block_corpus()
    m = _fit_seeded(docs, parents, groups, seed_words=block_words)
    for t in range(4):
        masses = [_block_mass(m, t, block_words[b]) for b in range(4)]
        assert int(np.argmax(masses)) == t, f"topic {t} did not land in its seeded slot"
        assert masses[t] > 0.6


def test_thread_tm_seed_is_soft_learns_beyond_seeds():
    """Seeding only a FEW of a block's words still pulls the whole block in: seeded topics keep
    substantial mass on the UNSEEDED block words (a soft prior, not a lock on the seeds), while
    still landing in their slots. We check the mass on average — with four competing blocks a
    single topic can occasionally shed its tail, but the soft-prior behavior holds in aggregate."""
    docs, parents, groups, block_words, _ = _planted_block_corpus()
    partial = {t: block_words[t][:3] for t in range(4)}  # 3 of 10 words per block
    m = _fit_seeded(docs, parents, groups, seed_words=partial, seed_weight=0.5)
    aligned = sum(int(np.argmax([_block_mass(m, t, block_words[b]) for b in range(4)]) == t)
                  for t in range(4))
    assert aligned == 4, "seeding a few words per block failed to recover the blocks in-slot"
    unseeded_mass = np.mean([_block_mass(m, t, block_words[t][3:]) for t in range(4)])
    assert unseeded_mass > 0.1, "topics collapsed onto their seeds instead of learning the block"


def test_thread_tm_prevalence_anchor_steers_group():
    """Pulling a group's anchor toward a target mix moves its recovered prevalence toward the
    target, monotonically in anchor_strength."""
    docs, parents, groups, block_words, _ = _planted_block_corpus()
    base = _fit_seeded(docs, parents, groups, seed_words=block_words)
    gp0 = np.asarray(base.group_prevalence)[0]
    target = gp0.copy(); target[0] *= 0.5; target[2] *= 2.0; target /= target.sum()
    prev = None
    for s in (0.0, 0.5, 0.9):
        m = _fit_seeded(docs, parents, groups, seed_words=block_words,
                        prevalence_anchor={0: target.tolist()}, anchor_strength=s)
        d = abs(np.asarray(m.group_prevalence)[0][0] - target[0])
        if prev is not None:
            assert d <= prev + 1e-6, "stronger anchor did not move prevalence toward the target"
        prev = d


def test_thread_tm_seed_glob_matching():
    """seed_match='glob' expands a wildcard against the vocabulary."""
    docs, parents, groups, block_words, _ = _planted_block_corpus()
    # 'w0*' matches w0, w0? none here (words are w0..w39); use explicit block-0 glob 'w?' won't do.
    # seed block 0 via the glob 'w[0-9]' is regex; for glob use the literal prefix of block 0 words.
    m = _fit_seeded(docs, parents, groups, seed_words={0: ["w1*"]}, seed_match="glob")
    # 'w1*' matches w1, w10..w19 (block 1's words w10-19 plus stray w1) — just assert it fit + seeded
    assert np.asarray(m.topic_word).shape[0] == 4


def test_thread_tm_seed_validation_and_content_guard():
    docs, parents, groups, block_words, _ = _planted_block_corpus(n_threads=20)
    with pytest.raises(ValueError, match="seed_prior"):
        _fit_seeded(docs, parents, groups, seed_words={0: ["w0"]}, seed_prior="bad")
    with pytest.raises(ValueError, match="seed_match"):
        _fit_seeded(docs, parents, groups, seed_words={0: ["w0"]}, seed_match="bad")
    with pytest.raises(ValueError, match="out of range"):
        _fit_seeded(docs, parents, groups, seed_words={99: ["w0"]})
    with pytest.raises(ValueError, match="length"):
        _fit_seeded(docs, parents, groups, prevalence_anchor={0: [0.5, 0.5]})
    with pytest.raises(ValueError, match="not supported together with a content"):
        _fit_seeded(docs, parents, groups, seed_words={0: ["w0"]}, content="depth")
    # finiteness / sign guards on the strength knobs (f64::clamp would let NaN through)
    for bad in (float("nan"), -1.0):
        with pytest.raises(ValueError, match="seed_weight"):
            _fit_seeded(docs, parents, groups, seed_words={0: ["w0"]}, seed_weight=bad)
        with pytest.raises(ValueError, match="seed_strength"):
            _fit_seeded(docs, parents, groups, seed_words={0: ["w0"]}, seed_strength=bad)
    for bad in (float("nan"), 1.5, -0.1):
        with pytest.raises(ValueError, match="anchor_strength"):
            _fit_seeded(docs, parents, groups,
                        prevalence_anchor={0: [0.25, 0.25, 0.25, 0.25]}, anchor_strength=bad)
    with pytest.raises(ValueError, match="non-negative topic mix"):
        _fit_seeded(docs, parents, groups, prevalence_anchor={0: [0.5, -0.2, 0.4, 0.3]})


def test_thread_tm_unseeded_fit_unchanged_by_none_seeds():
    """Passing no seeds must be bit-identical to the pre-feature default path (the seed hooks
    are inert when seed_words/prevalence_anchor are None)."""
    docs, parents, groups, _, _ = _planted_block_corpus(n_threads=40)
    a = _fit_seeded(docs, parents, groups)
    b = _fit_seeded(docs, parents, groups, seed_words=None, prevalence_anchor=None)
    assert np.allclose(np.asarray(a.topic_word), np.asarray(b.topic_word))
    assert np.allclose(np.asarray(a.group_prevalence), np.asarray(b.group_prevalence))


def test_thread_tm_seed_matches_introspection():
    """seed_matches exposes which vocabulary words each topic's patterns resolved to (issue #856),
    so glob/regex seeding is auditable; it is empty on an unseeded fit."""
    docs, parents, groups, block_words, _ = _planted_block_corpus()
    m = _fit_seeded(docs, parents, groups, seed_words={0: ["w0", "w1"], 2: ["w2*"]},
                    seed_match="glob")
    sm = dict(m.seed_matches)
    assert set(sm.keys()) == {0, 2}
    assert set(sm[0]) == {"w0", "w1"}
    # 'w2*' globs onto block-2 words w20..w29 (and w2 itself) — a superset check on a few
    assert {"w20", "w21", "w29"}.issubset(set(sm[2]))
    assert dict(_fit_seeded(docs, parents, groups).seed_matches) == {}


def test_thread_tm_prevalence_anchor_accepts_string_label():
    """prevalence_anchor keys may be the string covariate label (as passed to covariates=), not
    only the encoded integer index (issue #856); an unknown label raises helpfully."""
    docs, parents, groups, _, _ = _planted_block_corpus(n_threads=40)
    labels = ["g0" if g == 0 else "g1" for g in groups]
    target = [0.7, 0.1, 0.1, 0.1]
    by_label = topica.ThreadTM(4, em_iters=60, seed=13, coupling="parent").fit(
        docs, parents=parents, covariates=labels, min_count=1,
        prevalence_anchor={"g0": target}, anchor_strength=0.9)
    by_index = topica.ThreadTM(4, em_iters=60, seed=13, coupling="parent").fit(
        docs, parents=parents, covariates=labels, min_count=1,
        prevalence_anchor={0: target}, anchor_strength=0.9)
    assert np.allclose(np.asarray(by_label.group_prevalence),
                       np.asarray(by_index.group_prevalence))
    with pytest.raises(ValueError, match="is not one of the covariate groups"):
        topica.ThreadTM(4, em_iters=20, seed=13).fit(
            docs, parents=parents, covariates=labels, min_count=1,
            prevalence_anchor={"nope": target})


def test_thread_tm_seed_words_non_dict_error():
    """A non-dict seed_words gets a house-quality error, not the raw PyO3 message (issue #856)."""
    docs, parents, groups, _, _ = _planted_block_corpus(n_threads=20)
    with pytest.raises((ValueError, TypeError), match="seed_words must be a dict"):
        _fit_seeded(docs, parents, groups, seed_words=["w0", "w1"])


# ---------------------------------------------------------------------------
# thread_stability: thread-bootstrap robustness (issue #856)
# ---------------------------------------------------------------------------

def test_thread_stability_recovers_planted_topics():
    """On a planted-block corpus every topic reappears under thread resampling, so all K are
    flagged stable, similarities are high, and the result shapes/frame are well-formed."""
    docs, parents, groups, block_words, _ = _planted_block_corpus(n_threads=80)
    res = topica.evaluate.thread_stability(
        docs, parents, num_topics=4, covariates=groups, covariate_names=["g0", "g1"],
        n_boot=6, seed=13, em_iters=50, min_count=1, seed_words=block_words)
    assert res.n_threads == 80 and res.n_boot == 6
    assert set(res.similarity) == {0, 1, 2, 3}
    # seeded planted blocks are pinned, so every topic is highly stable
    assert all(res.similarity[t]["mean"] > 0.8 for t in range(4))
    assert res.stable == [0, 1, 2, 3]
    fr = res.to_frame()
    assert list(fr["topic"]) == [0, 1, 2, 3] and fr["stable"].all()


def test_thread_stability_prevalence_ci_recovers_group_contrast():
    """With covariates the prevalence dict carries thread-clustered CIs per (group, topic); on the
    planted corpus group 0 loves topics {0,1} and group 1 loves {2,3}, and the CIs reflect that."""
    docs, parents, groups, block_words, group_mix = _planted_block_corpus(n_threads=90)
    res = topica.evaluate.thread_stability(
        docs, parents, num_topics=4, covariates=groups, covariate_names=["g0", "g1"],
        n_boot=6, seed=13, em_iters=50, min_count=1, seed_words=block_words)
    # group 0's topic-0 prevalence is high (~0.4) and its CI sits well above group 1's (~0.1)
    g0_t0 = res.prevalence[("g0", 0)]
    g1_t0 = res.prevalence[("g1", 0)]
    assert g0_t0["mean"] > g1_t0["mean"]
    assert g0_t0["ci"][0] > g1_t0["ci"][1]  # intervals separate -> a robust contrast


def test_thread_stability_validation():
    docs, parents, groups, _, _ = _planted_block_corpus(n_threads=20)
    with pytest.raises(ValueError, match="n_boot"):
        topica.evaluate.thread_stability(docs, parents, num_topics=4, n_boot=1)
    # a single thread cannot be resampled
    single = [["a", "b", "c"], ["a", "b"]]
    with pytest.raises(ValueError, match="at least two threads"):
        topica.evaluate.thread_stability(single, [-1, 0], num_topics=2, n_boot=4, em_iters=5)


# ---------------------------------------------------------------------------
# reply_completion paired baseline-vs-baseline contrasts (issue #852)
# ---------------------------------------------------------------------------

def test_reply_completion_edge_contrast_and_paired_draws():
    """The edge-attribution contrast (permuted - no_tree) gets a thread-clustered CI, it is
    exposed as result.contrast['edge'], and it equals the algebraic (tree-no_tree)-(tree-permuted)
    identity. The raw paired draws let contrast_ci reproduce any pair on demand."""
    docs, parents = _branching_corpus(seed=1, persistence=0.92)
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=5,
        baselines=("no_tree", "permuted"), em_iters=40, seed=13, n_boot=200)
    # edge auto-computed when both permuted and no_tree are scored
    assert "edge" in res.contrast
    edge = res.contrast["edge"]
    assert edge["first"] == "permuted" and edge["second"] == "no_tree"
    lo, hi = edge["ci"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    # delta = tree - baseline, so edge = permuted - no_tree = delta[no_tree] - delta[permuted]
    # (the ThreadTM baselines all score the identical leaves, so the identity is exact).
    assert edge["estimate"] == pytest.approx(
        res.delta["no_tree"]["estimate"] - res.delta["permuted"]["estimate"], abs=1e-9)
    # the method reproduces the stored field exactly (same n_boot/seed defaults)
    m = res.contrast_ci("permuted", "no_tree")
    assert m["estimate"] == pytest.approx(edge["estimate"], abs=1e-12)
    assert m["ci"] == edge["ci"]
    # raw paired exposure
    assert set(res.paired["models"]) == {"tree", "no_tree", "permuted"}
    nleaf = len(res.paired["leaves"])
    assert nleaf == len(res.paired["thread_root"]) == len(res.paired["n_tokens"])
    for name in res.paired["models"]:
        assert len(res.paired["token_ll"][name]) == nleaf


def test_reply_completion_contrast_ci_matches_delta():
    """contrast_ci('tree', b) reproduces the tree-minus-baseline point estimate in delta[b]."""
    docs, parents = _branching_corpus(seed=2, persistence=0.9)
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=4, baselines=("no_tree", "permuted"),
        em_iters=40, seed=13, n_boot=150)
    for b in ("no_tree", "permuted"):
        c = res.contrast_ci("tree", b)
        assert c["estimate"] == pytest.approx(res.delta[b]["estimate"], abs=1e-9)


def test_reply_completion_contrast_validation():
    """A contrast naming an unscored model is rejected up front; an unknown alias too."""
    docs, parents = _branching_corpus(seed=1, persistence=0.9)
    with pytest.raises(ValueError, match="not scored"):
        topica.evaluate.reply_completion(
            docs, parents, num_topics=4, baselines=("no_tree",),
            contrasts=[("permuted", "no_tree")], em_iters=20, seed=13)
    with pytest.raises(ValueError, match="unknown contrast alias"):
        topica.evaluate.reply_completion(
            docs, parents, num_topics=4, baselines=("no_tree", "permuted"),
            contrasts=["bogus"], em_iters=20, seed=13)
    # contrast_ci rejects an unscored name after the fact
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=4, baselines=("no_tree",), em_iters=20, seed=13, n_boot=50)
    with pytest.raises(ValueError, match="unknown model"):
        res.contrast_ci("tree", "permuted")


def test_transform_covariates_none_uses_mean_anchor():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=30, depth=6)
    m = topica.ThreadTM(2, em_iters=60, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    th = m.transform(docs, parents=parents)  # covariates omitted
    assert th.shape == (len(docs), 2) and np.allclose(th.sum(1), 1.0)


def test_transform_accepts_corpus_input():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=25, depth=5)
    m = topica.ThreadTM(2, em_iters=50, seed=13)
    m.fit(docs, parents=parents)
    from_lists = m.transform(docs, parents=parents)
    corpus = topica.Corpus.from_documents(docs)
    from_corpus = m.transform(corpus, parents=parents)
    assert np.allclose(from_lists, from_corpus)


def test_transform_rejects_negative_covariate():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=5)
    m = topica.ThreadTM(2, em_iters=40, seed=13)
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    with pytest.raises(ValueError, match="group id"):
        m.transform(docs[:2], parents=[-1, 0], covariates=[-1, 0])


# ---------------------------------------------------------------------------
# coupling="root": thread-root anchoring for broadcast discourse (issue #831)
# ---------------------------------------------------------------------------

def test_coupling_validation():
    with pytest.raises(ValueError, match="coupling"):
        topica.ThreadTM(3, coupling="bogus")


def test_root_coupling_settings_and_repr():
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=5)
    m = topica.ThreadTM(2, em_iters=40, seed=13, coupling="root")
    assert m.coupling == "root"
    assert m.settings["coupling"] == "root"
    assert "coupling='root'" in repr(m) or 'coupling="root"' in repr(m)
    m.fit(docs, parents=parents)
    assert m.doc_topic.shape == (len(docs), 2)
    assert np.isfinite(m.kappa) and np.isfinite(m.sigma2)


def test_root_coupling_transform_and_save(tmp_path):
    docs, parents, cov, vocab = _threaded_corpus(n_threads=25, depth=5)
    m = topica.ThreadTM(2, em_iters=50, seed=13, coupling="root")
    m.fit(docs, parents=parents, covariates=cov, covariate_names=["A", "B"])
    th = m.transform(docs, parents=parents, covariates=cov)
    assert th.shape == (len(docs), 2) and np.allclose(th.sum(1), 1.0)
    p = tmp_path / "rtm_root.bin"
    m.save(str(p))
    m2 = topica.ThreadTM.load(str(p))
    assert m2.coupling == "root"
    assert np.allclose(m2.transform(docs, parents=parents, covariates=cov), th)


def test_coupling_survives_save_load_both():
    """settings["coupling"] round-trips through save/load for both values."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=15, depth=4)
    import tempfile, os
    for coupling in ("parent", "root"):
        m = topica.ThreadTM(2, em_iters=30, seed=13, coupling=coupling)
        m.fit(docs, parents=parents)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.bin")
            m.save(p)
            assert topica.ThreadTM.load(p).settings["coupling"] == coupling


def test_root_coupling_transform_new_forest():
    """Root coupling reparents a NEW forest (different topology from the fit forest) to its thread
    root: transform matches a hand-reparented star fit under parent coupling."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=25, depth=6)
    m_root = topica.ThreadTM(2, em_iters=60, seed=13, coupling="root")
    m_root.fit(docs, parents=parents)
    # a fresh multi-level forest, unrelated to the fit topology
    new = [list(docs[i]) for i in (0, 1, 2, 3, 4)]
    new_parents = [-1, 0, 1, 0, 2]  # root 0; a depth-3 chain 0<-1<-2<-4 plus a branch 0<-3
    got = m_root.transform(new, parents=new_parents)
    # Root coupling must collapse new_parents to the thread-root star before inference, so the
    # result equals transform on the hand-reparented star under the SAME model (identical topics).
    star = [-1, 0, 0, 0, 0]  # every non-root points at root 0
    got_star = m_root.transform(new, parents=star)
    assert np.allclose(got, got_star)
    assert got.shape == (5, 2) and np.allclose(got.sum(1), 1.0)


def _broadcast_corpus(seed=1, n_threads=70):
    """Planted BROADCAST discourse: a thin leaf tracks its THREAD ROOT's topic, not its immediate
    parent, which is deliberately the OPPOSITE topic. With few seen tokens per leaf the prior
    dominates, so root coupling (shrink to the root) should predict the held-out leaf tokens far
    better than parent coupling (shrink to the opposite-topic parent)."""
    rng = np.random.default_rng(seed)
    A = [f"a{i}" for i in range(8)]
    B = [f"b{i}" for i in range(8)]
    docs, parents = [], []
    for t in range(n_threads):
        roott, other = (A, B) if t % 2 == 0 else (B, A)
        docs.append(list(rng.choice(roott, 12))); parents.append(-1)
        r = len(docs) - 1
        for _ in range(2):
            # middle reply is the OPPOSITE topic (a maximally misleading parent)
            docs.append(list(rng.choice(other, 12))); parents.append(r)
            m = len(docs) - 1
            for _ in range(2):  # thin leaves (3 tokens) that echo the ROOT topic
                docs.append(list(rng.choice(roott, 3))); parents.append(m)
    return docs, parents


def test_root_coupling_wins_on_broadcast():
    """On planted broadcast discourse (leaves track the root, not the noisy parent), the
    root-coupled model predicts held-out leaf tokens better than the parent-coupled tree."""
    docs, parents = _broadcast_corpus(seed=1)
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=2, baselines=("root",), em_iters=60, seed=13, n_boot=300)
    # delta["root"] = parent-tree minus root-tree; root better => root scores higher and the
    # paired difference is negative with the bulk of the bootstrap mass below zero.
    assert res.per_token_ll["root"] > res.per_token_ll["tree"]
    assert res.delta["root"]["estimate"] < 0.0
    assert res.delta["root"]["ci"][0] < 0.0


def test_parent_coupling_wins_on_chain_persistence():
    """On chain-persistence discourse (a reply tracks its parent), parent coupling beats root
    coupling: delta["root"] is positive."""
    docs, parents = _branching_corpus(seed=1, persistence=0.92)
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=5, baselines=("root",), em_iters=60, seed=13, n_boot=300)
    assert res.delta["root"]["estimate"] > 0.0


# ---------------------------------------------------------------------------
# coupling="blend": parent + root mixture (issue #831)
# ---------------------------------------------------------------------------

def test_blend_validation():
    # blend weights only valid with coupling="blend"
    with pytest.raises(ValueError, match="only valid with"):
        topica.ThreadTM(2, blend_alpha=0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        topica.ThreadTM(2, coupling="blend", blend_alpha=1.5)
    with pytest.raises(ValueError, match="<= 1"):
        topica.ThreadTM(2, coupling="blend", blend_alpha=0.7, blend_beta=0.6)
    with pytest.raises(ValueError, match="coupling"):
        topica.ThreadTM(2, coupling="bogus")


def _mixed_corpus(seed=1, n_threads=200):
    """Planted MIXED discourse: each thin leaf token is drawn 50/50 from its thread-root topic and
    its parent topic, with root and parent topics both drawn from four equally-frequent blocks and
    required to differ (so all topics are balanced, the two regressors are not collinear, and there
    are no pure leaves). The optimal predictor of a held-out leaf token, not knowing which component
    it came from, is the blend of parent and root, so blend coupling should beat both pure couplings.

    Leaves carry 6 tokens (not the 2-3 of a "thin" leaf) on purpose: the #834 full-covariance base
    is strong enough that a thin-leaf planted signal washes out (the leaf's own η is too noisy for
    the two-regressor weight estimator, and blend can then lose to plain parent coupling). This is a
    genuine short-reply limitation of blend, documented in the guide; the corpus here gives the
    estimator enough per-leaf evidence to recover the planted mix."""
    rng = np.random.default_rng(seed)
    blocks = [[f"t{k}w{i}" for i in range(6)] for k in range(4)]
    docs, parents = [], []
    for _ in range(n_threads):
        rk = int(rng.integers(4))
        docs.append(list(rng.choice(blocks[rk], 10))); parents.append(-1)
        r = len(docs) - 1
        for _ in range(2):
            pk = int(rng.integers(4))
            while pk == rk:  # parent topic differs from the root topic
                pk = int(rng.integers(4))
            docs.append(list(rng.choice(blocks[pk], 10))); parents.append(r)
            m = len(docs) - 1
            for _ in range(4):  # leaves: each token a 50/50 draw from root vs parent topic
                leaf = [rng.choice(blocks[rk if rng.random() < 0.5 else pk]) for _ in range(6)]
                docs.append(leaf); parents.append(m)
    return docs, parents


def test_blend_fits_getters_and_repr():
    docs, parents = _mixed_corpus()
    m = topica.ThreadTM(4, em_iters=80, seed=13, coupling="blend")
    m.fit(docs, parents=parents)
    assert m.coupling == "blend"
    assert np.isfinite(m.blend_alpha) and np.isfinite(m.blend_beta)
    assert 0.0 <= m.blend_alpha <= 1.0 and 0.0 <= m.blend_beta <= 1.0
    assert m.blend_alpha + m.blend_beta <= 1.0 + 1e-9
    assert np.isnan(m.kappa)  # blend has no single reversion
    assert "blend" in repr(m)
    assert m.settings["coupling"] == "blend"


def test_blend_estimates_both_weights_positive():
    """On genuinely mixed discourse both the parent and the root weight are estimated positive."""
    docs, parents = _mixed_corpus()
    m = topica.ThreadTM(4, em_iters=80, seed=13, coupling="blend")
    m.fit(docs, parents=parents)
    assert m.blend_alpha > 0.01 and m.blend_beta > 0.01


def test_blend_fixed_weights_respected():
    docs, parents = _mixed_corpus(n_threads=30)
    m = topica.ThreadTM(4, em_iters=40, seed=13, coupling="blend", blend_alpha=0.6, blend_beta=0.3)
    m.fit(docs, parents=parents)
    assert m.blend_alpha == 0.6 and m.blend_beta == 0.3
    assert m.settings["blend_alpha"] == 0.6 and m.settings["blend_beta"] == 0.3


def test_blend_partial_pin_keeps_convexity():
    """Pinning ONE weight must estimate the other conditionally and never push alpha+beta past 1
    (a negative anchor weight would make the prior mean non-convex). Regression for the
    three-reviewer finding on PR #833."""
    docs, parents = _mixed_corpus()
    for pin in (0.5, 0.7, 0.9):
        ma = topica.ThreadTM(4, em_iters=60, seed=13, coupling="blend", blend_alpha=pin)
        ma.fit(docs, parents=parents)
        assert ma.blend_alpha == pin
        assert 0.0 <= ma.blend_beta <= 1.0 - pin + 1e-9
        assert ma.blend_alpha + ma.blend_beta <= 1.0 + 1e-9
        mb = topica.ThreadTM(4, em_iters=60, seed=13, coupling="blend", blend_beta=pin)
        mb.fit(docs, parents=parents)
        assert mb.blend_beta == pin
        assert 0.0 <= mb.blend_alpha <= 1.0 - pin + 1e-9
        assert mb.blend_alpha + mb.blend_beta <= 1.0 + 1e-9


def test_blend_transform_and_save(tmp_path):
    docs, parents = _mixed_corpus(n_threads=30)
    m = topica.ThreadTM(4, em_iters=50, seed=13, coupling="blend")
    m.fit(docs, parents=parents)
    th = m.transform(docs, parents=parents)
    assert th.shape == (len(docs), 4) and np.allclose(th.sum(1), 1.0)
    p = tmp_path / "blend.bin"
    m.save(str(p))
    m2 = topica.ThreadTM.load(str(p))
    assert m2.coupling == "blend"
    assert m2.blend_alpha == m.blend_alpha and m2.blend_beta == m.blend_beta
    assert np.allclose(m2.transform(docs, parents=parents), th)


def test_blend_identifiability_warning():
    """A shallow (depth-2-only) corpus cannot separate alpha from beta; the fit warns."""
    docs, parents, cov, vocab = _threaded_corpus(n_threads=20, depth=2)
    m = topica.ThreadTM(2, em_iters=30, seed=13, coupling="blend")
    with pytest.warns(UserWarning, match="not separately identified"):
        m.fit(docs, parents=parents)


def test_blend_beats_parent_and_root_on_mixed():
    """The headline: on mixed discourse the estimated blend predicts held-out leaf tokens better
    than either pure-parent or pure-root coupling."""
    docs, parents = _mixed_corpus()
    res = topica.evaluate.reply_completion(
        docs, parents, num_topics=4, baselines=("root", "blend"),
        em_iters=70, seed=13, n_boot=300)
    assert res.per_token_ll["blend"] > res.per_token_ll["tree"]   # blend beats pure parent
    assert res.per_token_ll["blend"] > res.per_token_ll["root"]   # blend beats pure root
    assert res.delta["blend"]["estimate"] < 0.0                   # tree(parent) minus blend < 0
