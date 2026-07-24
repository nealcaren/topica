"""Tests for RTM, the Relational Topic Model (Chang & Blei, AOAS 2010).

Covers the four idioms (shapes/normalization, planted-data recovery,
determinism, save-load + bad-params) plus the analysis surface, the link head
(eta/nu, predict_link, suggest_links), and edge cases. The variational math is
validated against a standalone NumPy reference in parity/rtm_reference.py.
"""
import numpy as np
import pytest

import topica


def _planted(seed=0, D=48, K=3, block=6, doclen=40):
    """K word-blocks; each doc drawn from one block; links dense within a latent
    group, sparse across. Returns (str_docs, edges, groups, vocab_size)."""
    rng = np.random.default_rng(seed)
    V = K * block
    groups = np.tile(np.arange(K), D // K)
    docs = []
    for g in groups:
        w = rng.integers(g * block, (g + 1) * block, size=doclen)
        noise = rng.random(doclen) < 0.12
        w[noise] = rng.integers(0, V, size=noise.sum())
        docs.append([str(x) for x in w])
    edges = []
    for i in range(D):
        for j in range(i + 1, D):
            p = 0.45 if groups[i] == groups[j] else 0.01
            if rng.random() < p:
                edges.append((i, j))
    return docs, edges, groups, V


def _toy():
    docs = [["a", "b", "c"], ["a", "b", "b"], ["c", "c", "d"], ["d", "e", "f"]]
    edges = [(0, 1), (2, 3)]
    return docs, edges


@pytest.mark.parametrize("link", ["logistic", "exponential"])
def test_shapes_and_normalization(link):
    docs, edges, _g, V = _planted()
    m = topica.RTM(3, link=link, seed=0).fit(docs, edges, iters=25)
    assert m.topic_word.shape == (3, len(m.vocabulary))
    assert m.doc_topic.shape == (len(docs), 3)
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)
    # phi_bar is a distinct (D, K) simplex, not doc_topic
    assert m.phi_bar.shape == (len(docs), 3)
    np.testing.assert_allclose(m.phi_bar.sum(axis=1), 1.0, atol=1e-9)
    assert m.eta.shape == (3,)
    assert isinstance(m.nu, float)


def test_fit_returns_self():
    docs, edges = _toy()
    m = topica.RTM(2, seed=0)
    assert m.fit(docs, edges, iters=10) is m


@pytest.mark.parametrize("link", ["logistic", "exponential"])
def test_recovers_planted_topics(link):
    docs, edges, groups, V = _planted()
    m = topica.RTM(3, link=link, alpha=0.5, seed=0).fit(docs, edges, iters=40)
    # each topic owns a distinct word block
    tw = m.topic_word[:, np.argsort([int(w) for w in m.vocabulary])]
    owned = [max(range(3), key=lambda b: tw[t, b * 6:(b + 1) * 6].sum()) for t in range(3)]
    assert set(owned) == {0, 1, 2}
    # objective rises over EM
    hist = m.fit_history
    assert hist[-1][1] >= hist[0][1]
    # link prediction separates in-group from cross-group pairs
    same = [m.predict_link(i, j) for i in range(0, 48, 3) for j in range(i + 1, 48, 5)
            if groups[i] == groups[j]]
    diff = [m.predict_link(i, j) for i in range(0, 48, 3) for j in range(i + 1, 48, 5)
            if groups[i] != groups[j]]
    assert np.mean(same) > np.mean(diff)


def test_suggest_links_from_words():
    docs, edges, groups, V = _planted()
    m = topica.RTM(3, alpha=0.5, seed=0).fit(docs, edges, iters=40)
    # a fresh doc built from group-0's block should rank group-0 docs highest
    new_doc = [str(w) for w in np.random.default_rng(1).integers(0, 6, size=40)]
    ranked = m.suggest_links(new_doc, top_n=10)
    assert len(ranked) == 10
    top_groups = [groups[d] for d, _ in ranked]
    assert top_groups.count(0) >= 6  # majority from the matching group
    # exclude removes a candidate
    excl = ranked[0][0]
    assert excl not in [d for d, _ in m.suggest_links(new_doc, top_n=10, exclude=[excl])]


def test_determinism():
    docs, edges, _g, _V = _planted()
    a = topica.RTM(3, seed=3).fit(docs, edges, iters=20)
    b = topica.RTM(3, seed=3).fit(docs, edges, iters=20)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.eta, b.eta)
    c = topica.RTM(3, seed=99).fit(docs, edges, iters=20)
    assert not np.array_equal(a.topic_word, c.topic_word)


def test_save_load_roundtrip(tmp_path):
    docs, edges, _g, _V = _planted()
    m = topica.RTM(3, link="exponential", seed=0).fit(docs, edges, iters=20)
    p = str(tmp_path / "m.tt")
    m.save(p)
    n = topica.RTM.load(p)
    assert np.array_equal(m.topic_word, n.topic_word)
    assert np.array_equal(m.phi_bar, n.phi_bar)
    assert np.array_equal(m.eta, n.eta)
    assert n.link == "exponential"
    assert n.predict_link(0, 1) == m.predict_link(0, 1)


def test_bad_params():
    with pytest.raises(ValueError):
        topica.RTM(1)
    with pytest.raises(ValueError):
        topica.RTM(3, link="probit")
    with pytest.raises(ValueError):
        topica.RTM(3, alpha=0.0)
    with pytest.raises(ValueError):
        topica.RTM(3, negative_ratio=-1.0)
    # The pseudo-negative count (rho / negative_ratio, the paper's regularization,
    # R lda's `lambda`) prevents the degenerate positive-links-only fit, so zero is
    # rejected -- not merely negatives. `ridge` (the l2 prior) still allows zero.
    with pytest.raises(ValueError, match="finite and > 0"):
        topica.RTM(3, negative_ratio=0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        topica.RTM(3, rho=0.0)
    topica.RTM(3, ridge=0.0)  # zero l2 prior (plain MLE) is allowed


# --- collapsed-Gibbs backend (#424) -----------------------------------------

# R lda's rtm.em restarts a fresh sampler each M-step, so quality comes from a
# converged E-sweep budget, and its estimate.params beta runs away negative after
# too many M-steps -- so the Gibbs tests use a converged, few-M-step config that
# mirrors parity/rtm_gibbs_gold.py (see fit_rtm_gibbs docs).
_GIBBS = dict(iters=5, e_sweeps=60)


def test_gibbs_recovers_topics_and_round_trips_settings():
    docs, edges, groups, V = _planted()
    m = topica.RTM(3, link="exponential", inference="gibbs", alpha=0.5, seed=0)
    m.fit(docs, edges, **_GIBBS)
    assert m.settings["inference"] == "gibbs"
    assert "beta" in m.settings
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)
    # each topic owns a distinct planted word block
    tw = m.topic_word[:, np.argsort([int(w) for w in m.vocabulary])]
    owned = [max(range(3), key=lambda b: tw[t, b * 6:(b + 1) * 6].sum()) for t in range(3)]
    assert set(owned) == {0, 1, 2}


def test_gibbs_link_beta_is_negative_like_r():
    # R lda's estimate.params sets beta_k = log(p_k), p_k in (0,1), so the link
    # coefficient is negative even on strongly-linked data (the documented quirk).
    docs, edges, _g, _V = _planted()
    m = topica.RTM(3, link="exponential", inference="gibbs", seed=0).fit(docs, edges, **_GIBBS)
    assert (np.asarray(m.eta) < 0).all()
    assert m.nu == 0.0  # the reference's exponential link has no intercept


def test_gibbs_determinism_and_save_load(tmp_path):
    docs, edges, _g, _V = _planted()
    a = topica.RTM(3, inference="gibbs", seed=3).fit(docs, edges, **_GIBBS)
    b = topica.RTM(3, inference="gibbs", seed=3).fit(docs, edges, **_GIBBS)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.eta, b.eta)
    c = topica.RTM(3, inference="gibbs", seed=99).fit(docs, edges, **_GIBBS)
    assert not np.array_equal(a.topic_word, c.topic_word)
    # save/load preserves the backend + fit
    p = str(tmp_path / "gibbs.tt")
    a.save(p)
    n = topica.RTM.load(p)
    assert n.settings["inference"] == "gibbs"
    assert np.array_equal(n.topic_word, a.topic_word)
    assert np.array_equal(n.eta, a.eta)


def test_bad_inference_rejected():
    with pytest.raises(ValueError, match="inference"):
        topica.RTM(3, inference="mcmc")


def test_gibbs_link_defaults_to_exponential_and_rejects_explicit_conflict():
    # R lda's collapsed-Gibbs sampler is exponential-only. `link` is a sentinel: an
    # unset link under gibbs resolves to (and stores) exponential; an explicit
    # non-exponential link is rejected, not silently overwritten.
    docs, edges, _g, _V = _planted()
    m = topica.RTM(3, inference="gibbs", seed=0).fit(docs, edges, iters=5, e_sweeps=20)
    assert m.settings["link"] == "exponential"
    # explicit exponential is fine; explicit logistic under gibbs raises
    topica.RTM(3, link="exponential", inference="gibbs")
    with pytest.raises(ValueError, match="exponential"):
        topica.RTM(3, link="logistic", inference="gibbs")
    # variational still defaults to logistic
    assert topica.RTM(3).settings["link"] == "logistic"
    # exponential link scores are exp(sum eta*z*z) <= 1; a logistic read would be 0.5-ish
    pr = m.predict_link(0, 1)
    assert 0.0 < pr <= 1.0


def test_gibbs_handles_no_links_and_empty_docs():
    docs, _edges, _g, _V = _planted()
    # no links: reduces to an LDA-like Gibbs fit, still valid output
    m = topica.RTM(3, inference="gibbs", seed=0).fit(docs, [], iters=5, e_sweeps=20)
    assert m.topic_word.shape == (3, len(m.vocabulary))
    np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
    # an empty document is tolerated (its phi_bar row is all-zero, no div-by-zero)
    docs2 = docs[:10] + [[]] + docs[10:]
    edges2 = [(0, 2), (3, 5)]
    m2 = topica.RTM(3, inference="gibbs", seed=1).fit(docs2, edges2, iters=15)
    assert np.isfinite(m2.doc_topic).all()
    assert np.isfinite(m2.phi_bar).all()


def test_edge_cases():
    docs, edges = _toy()
    # empty corpus
    with pytest.raises((ValueError, RuntimeError)):
        topica.RTM(2).fit([], [])
    # link index out of range
    with pytest.raises(ValueError):
        topica.RTM(2).fit(docs, [(0, 99)])
    # no links reduces to LDA-like fit (should still produce valid output)
    m = topica.RTM(2, seed=0).fit(docs, [], iters=15)
    assert m.topic_word.shape == (2, len(m.vocabulary))
    # unfitted access raises
    with pytest.raises(RuntimeError):
        topica.RTM(2).topic_word


def test_analysis_surface():
    docs, edges, _g, _V = _planted()
    m = topica.RTM(3, seed=0).fit(docs, edges, iters=20)
    assert topica.summary(m) is not None
    tbl = topica.topic_table(m)
    assert tbl is not None
    cov = m.coherence(10)
    assert cov.shape == (3,)
    # settings round-trips every constructor arg
    s = m.settings
    assert s["num_topics"] == 3 and s["link"] == "logistic"
    assert set(s) >= {"num_topics", "link", "alpha", "rho", "negative_ratio", "ridge", "seed"}
