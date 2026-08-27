"""Tests for ReplyTM (reply-conditioned topic model, issue #810).

Follows the four idioms in CONTRIBUTING-MODELS.md (shapes/normalization,
planted-data recovery, determinism, save-load + bad-params) plus an
analysis-surface check and edge cases. The exact tiny-tree enumeration gate that
guards the sampler's correctness lives in `src/reply_tm.rs`
(`reply_tm_enumeration_gate`); here we validate directed-response recovery through
the Python binding end-to-end.
"""
import numpy as np
import pytest

import topica

topica.enable_experimental()

K, V = 6, 60
BLOCK = V // K


def _planted_phi():
    phi = np.full((K, V), 0.02)
    for k in range(K):
        phi[k, k * BLOCK:(k + 1) * BLOCK] += 5.0
    return phi / phi.sum(1, keepdims=True)


def _planted_T():
    T = np.full((K, K), 0.6 / (K - 1))
    for k in range(K):
        T[k, k] = 1.4
    T[0, 3] += 1.1
    T[1, 4] += 0.9
    T[2, 0] += 0.8
    return T / T.sum(1, keepdims=True)


def _generate(n_roots=900, seed=13):
    """A reply forest drawn from the ReplyTM generative model with a known,
    asymmetric response matrix (parent 0->3, 1->4, 2->0)."""
    rng = np.random.default_rng(seed)
    phi, T = _planted_phi(), _planted_T()
    mu = np.log(np.full(K, 0.3))
    rho = 6.0
    docs, parents = [], []

    def draw(a):
        th = rng.dirichlet(np.maximum(a, 1e-6))
        L = max(4, rng.poisson(30))
        z = rng.choice(K, size=L, p=th)
        return z, [f"w{int(rng.choice(V, p=phi[k]))}" for k in z]

    for _ in range(n_roots):
        z, w = draw(np.exp(mu))
        ridx = len(docs)
        docs.append(w)
        parents.append(-1)
        frontier = [(ridx, np.bincount(z, minlength=K) / len(z), 1)]
        while frontier:
            pidx, zbar_p, depth = frontier.pop()
            if depth >= 5:
                continue
            for _ in range(rng.integers(1, 4)):
                if rng.random() < 0.3 + 0.12 * depth:
                    continue
                a = np.exp(mu) + rho * (T.T @ zbar_p)
                z, w = draw(a)
                cidx = len(docs)
                docs.append(w)
                parents.append(pidx)
                frontier.append((cidx, np.bincount(z, minlength=K) / len(z), depth + 1))
    return docs, parents, phi, T


def _toy_docs():
    return [["a", "b", "c"], ["a", "b", "b"], ["c", "c", "d"], ["d", "e", "f"]]


# --- idiom 1: shapes / normalization -----------------------------------------
def test_shapes_and_normalization():
    m = topica.ReplyTM(2, seed=0).fit(_toy_docs(), parents=[-1, 0, -1, 2], iters=40)
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    # response matrix: one (K, K) simplex-row matrix per group
    assert m.num_groups == 1
    T = np.asarray(m.response_matrix[0])
    assert T.shape == (2, 2)
    assert np.allclose(T.sum(axis=1), 1.0)
    # credible intervals bracket the posterior mean
    lo = np.asarray(m.response_matrix_lower[0])
    hi = np.asarray(m.response_matrix_upper[0])
    assert np.all(lo <= T + 1e-9) and np.all(hi >= T - 1e-9)
    assert np.asarray(m.response_strength).shape == (1,)
    assert np.asarray(m.baseline).shape == (1, 2)


# --- idiom 2: planted directed-response recovery -----------------------------
@pytest.mark.slow
def test_recovers_planted_directed_response():
    docs, parents, phi, T = _generate()
    m = topica.ReplyTM(K, seed=13, covariate_response="global").fit(
        docs, parents=parents, iters=500
    )
    # align inferred topics to planted via phi cosine
    from scipy.optimize import linear_sum_assignment

    tw = np.asarray(m.topic_word)
    vocab = list(m.vocabulary)
    col = {int(w[1:]): j for j, w in enumerate(vocab)}
    keep = [w for w in range(V) if w in col]
    twk = tw[:, [col[w] for w in keep]]
    phik = phi[:, keep]
    cos = (twk @ phik.T) / (
        np.linalg.norm(twk, axis=1)[:, None] * np.linalg.norm(phik, axis=1)[None, :]
    )
    ri, ci = linear_sum_assignment(-cos)
    assert cos[ri, ci].mean() > 0.9  # topics recovered
    inv = np.zeros(K, dtype=int)
    for t in range(K):
        inv[ci[t]] = ri[t]
    That = np.asarray(m.response_matrix[0])
    Tal = That[np.ix_(inv, inv)]
    Tal = Tal / Tal.sum(1, keepdims=True)
    off = ~np.eye(K, dtype=bool)
    c_off = np.corrcoef(Tal[off], T[off])[0, 1]
    assert c_off >= 0.6, f"off-diagonal response not recovered: corr={c_off}"
    # the three planted directed responses should top their rows' off-diagonals
    planted = [(0, 3), (1, 4), (2, 0)]
    top = sorted(
        [(i, j) for i in range(K) for j in range(K) if i != j], key=lambda ij: -Tal[ij]
    )[:3]
    assert sum(p in top for p in planted) >= 2, f"top responses {top} miss planted {planted}"


# --- idiom 3: determinism -----------------------------------------------------
def test_determinism():
    d, p = _toy_docs(), [-1, 0, -1, 2]
    a = topica.ReplyTM(2, seed=1).fit(d, parents=p, iters=40)
    b = topica.ReplyTM(2, seed=1).fit(d, parents=p, iters=40)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(np.asarray(a.response_matrix[0]), np.asarray(b.response_matrix[0]))


# --- idiom 4: save/load + bad params -----------------------------------------
def test_save_load_roundtrip(tmp_path):
    m = topica.ReplyTM(2, seed=3).fit(_toy_docs(), parents=[-1, 0, -1, 2], iters=40)
    p = tmp_path / "m.bin"
    m.save(str(p))
    m2 = topica.ReplyTM.load(str(p))
    assert np.allclose(m.topic_word, m2.topic_word)
    assert np.allclose(np.asarray(m.response_matrix[0]), np.asarray(m2.response_matrix[0]))
    assert m2.settings == m.settings


def test_bad_params():
    with pytest.raises(ValueError):
        topica.ReplyTM(0)
    with pytest.raises(ValueError):
        topica.ReplyTM(2, alpha=0.0)
    with pytest.raises(ValueError):
        topica.ReplyTM(2, covariate_response="nonsense")
    with pytest.raises(ValueError):
        topica.ReplyTM(2, response_link="loglinear")  # reserved, not implemented
    with pytest.raises(ValueError):
        topica.ReplyTM(2, t_inference="map")  # reserved, not implemented


# --- uncertainty surface + readers -------------------------------------------
def test_uncertainty_surface_and_readers():
    docs = _toy_docs() * 5
    parents = ([-1, 0, -1, 2]) * 5
    cov = ([0, 0, 1, 1]) * 5
    m = topica.ReplyTM(3, seed=7).fit(
        docs, parents=parents, covariate=cov, covariate_labels=["A", "B"], iters=120
    )
    assert m.group_labels == ["A", "B"]
    rs, rl, ru = (np.asarray(x) for x in (m.response_strength, m.response_strength_lower, m.response_strength_upper))
    assert np.all(rl <= rs + 1e-9) and np.all(ru >= rs - 1e-9)
    bl = np.asarray(m.baseline_lower); bu = np.asarray(m.baseline_upper); b = np.asarray(m.baseline)
    assert bl.shape == b.shape and np.all(bl <= b + 1e-9) and np.all(bu >= b - 1e-9)
    fh = m.fit_history
    assert len(fh) >= 3
    assert isinstance(m.converged, bool)
    rt = topica.inspect.response_table(m, group="A", top=4)
    assert rt and all(not r["is_homophily"] for r in rt)
    assert all({"lift_over_base", "ci_lower", "ci_upper"} <= set(r) for r in rt)
    rc = topica.inspect.response_contrast(m, "A", "B", top=4)
    assert rc and all("separated" in r and r["parent_topic"] != r["child_topic"] for r in rc)


def test_num_threads_deterministic_and_recovers():
    docs = _toy_docs() * 6
    parents = ([-1, 0, -1, 2]) * 6
    a = topica.ReplyTM(2, seed=13).fit(docs, parents=parents, iters=100, num_threads=3)
    b = topica.ReplyTM(2, seed=13).fit(docs, parents=parents, iters=100, num_threads=3)
    # reproducible for a fixed thread count
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(np.asarray(a.response_matrix[0]), np.asarray(b.response_matrix[0]))
    assert np.allclose(a.doc_topic.sum(1), 1.0)


@pytest.mark.slow
def test_ci_coverage_is_near_nominal():
    """Simulation-based coverage check (Gate-B F2): the calibrated 95% credible
    intervals for the off-diagonal response cells must cover the planted truth at
    close to the nominal rate across repeated corpora. Guards the CI_CALIBRATION."""
    K, V = 5, 50
    block = V // K
    phi = np.full((K, V), 0.02)
    for k in range(K):
        phi[k, k*block:(k+1)*block] += 6.0
    phi /= phi.sum(1, keepdims=True)
    T = np.full((K, K), 0.5/(K-1))
    for k in range(K):
        T[k, k] = 1.2
    T[0, 2] += 0.9; T[1, 3] += 0.7; T[2, 4] += 0.6
    T /= T.sum(1, keepdims=True)
    mu = np.log(np.full(K, 0.4))
    from scipy.optimize import linear_sum_assignment

    def simulate(seed):
        rng = np.random.default_rng(seed)
        docs, parents = [], []
        for _ in range(400):
            th = rng.dirichlet(np.exp(mu)); z = rng.choice(K, size=max(4, rng.poisson(30)), p=th)
            docs.append([f"w{int(rng.choice(V, p=phi[k]))}" for k in z]); r = len(docs)-1; parents.append(-1)
            fr = [(r, np.bincount(z, minlength=K)/len(z), 1)]
            while fr:
                pi, zb, dep = fr.pop()
                if dep >= 4: continue
                for _ in range(rng.integers(1, 4)):
                    if rng.random() < 0.35: continue
                    a = np.exp(mu) + 5.0*(T.T @ zb); th = rng.dirichlet(a)
                    z = rng.choice(K, size=max(4, rng.poisson(30)), p=th)
                    docs.append([f"w{int(rng.choice(V, p=phi[k]))}" for k in z]); ci = len(docs)-1
                    parents.append(pi); fr.append((ci, np.bincount(z, minlength=K)/len(z), dep+1))
        return docs, parents

    hits = tot = 0
    for s in range(12):
        docs, parents = simulate(500+s)
        m = topica.ReplyTM(K, seed=13, covariate_response="global").fit(docs, parents=parents, iters=400)
        tw = np.asarray(m.topic_word); col = {int(w[1:]): j for j, w in enumerate(m.vocabulary)}
        keep = [w for w in range(V) if w in col]
        cos = tw[:, [col[w] for w in keep]] @ phi[:, keep].T
        cos /= (np.linalg.norm(tw[:, [col[w] for w in keep]], axis=1)[:, None]
                * np.linalg.norm(phi[:, keep], axis=1)[None, :])
        ri, ci = linear_sum_assignment(-cos); inv = np.zeros(K, dtype=int)
        for t in range(K): inv[ci[t]] = ri[t]
        lo = np.asarray(m.response_matrix_lower[0])[np.ix_(inv, inv)]
        hi = np.asarray(m.response_matrix_upper[0])[np.ix_(inv, inv)]
        Tn = T / T.sum(1, keepdims=True)
        for i in range(K):
            for j in range(K):
                if i == j: continue
                tot += 1
                if lo[i, j] <= Tn[i, j] <= hi[i, j]: hits += 1
    coverage = hits / tot
    assert coverage >= 0.88, f"calibrated CI coverage {coverage:.3f} below acceptable band"


def test_covariate_labels_length_validated():
    with pytest.raises(ValueError):
        topica.ReplyTM(2).fit(_toy_docs(), parents=[-1, 0, -1, 2],
                              covariate=[0, 0, 1, 1], covariate_labels=["only_one"])


# --- covariate-varying response ----------------------------------------------
def test_per_group_response_matrices():
    docs = _toy_docs() * 4
    parents = ([-1, 0, -1, 2]) * 4
    cov = ([0, 0, 1, 1]) * 4
    m = topica.ReplyTM(2, seed=5).fit(docs, parents=parents, covariate=cov, iters=60)
    assert m.num_groups == 2
    assert len(m.response_matrix) == 2
    for g in range(2):
        assert np.allclose(np.asarray(m.response_matrix[g]).sum(1), 1.0)
    assert m.group_labels == ["group_0", "group_1"]


# --- analysis surface ---------------------------------------------------------
def test_analysis_surface():
    m = topica.ReplyTM(3, seed=0).fit(_toy_docs(), parents=[-1, 0, -1, 2], iters=40)
    # method-of-composition SEs need alpha + doc_topic; both present
    assert np.asarray(m.alpha).shape == (3,)
    assert np.all(np.asarray(m.alpha) > 0)
    tt = topica.inspect.topic_table(m)
    assert tt is not None
    tw = m.top_words(3)
    assert len(tw) == 3


# --- edge cases ---------------------------------------------------------------
def test_all_roots_runs_like_lda():
    # No parents => every doc a root; should still fit and normalize.
    m = topica.ReplyTM(2, seed=0).fit(_toy_docs(), iters=40)
    assert np.allclose(m.doc_topic.sum(1), 1.0)
    assert m.num_groups == 1


def test_rejects_empty_corpus():
    with pytest.raises((ValueError, RuntimeError)):
        topica.ReplyTM(2).fit([])


def test_rejects_bad_parents():
    with pytest.raises(ValueError):
        topica.ReplyTM(2).fit(_toy_docs(), parents=[0, 0, 0, 0])  # doc 0 points at itself
    with pytest.raises(ValueError):
        topica.ReplyTM(2).fit(_toy_docs(), parents=[-1, 0])  # wrong length
    with pytest.raises(ValueError):
        topica.ReplyTM(2).fit(_toy_docs(), parents=[1, 0, -1, 2])  # cycle 0<->1


def test_requires_experimental():
    # With experimental disabled, construction raises; re-enable for other tests.
    topica.enable_experimental(False)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            topica.ReplyTM(2)
    finally:
        topica.enable_experimental()
