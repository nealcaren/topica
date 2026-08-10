"""Tests for GaussianLDA (Das, Zaheer & Dyer, ACL 2015; issue #689).

Gaussian LDA models each topic as a Gaussian over the word-embedding space (NIW
prior), fit by collapsed Gibbs with a Student-t posterior predictive. These tests
cover the analysis surface, planted-cluster recovery (the k-means default and the
reference-faithful random-init path), determinism, save/load, transform, and edge
cases.
"""

import numpy as np
import pytest

import topica


def _planted(k=3, e=8, words_per_topic=12, n_docs=60, tokens=20, seed=0):
    """K well-separated Gaussian clusters on orthogonal axes; each doc draws its tokens
    from one dominant cluster's words. Returns (docs, embeddings, vocab, doc_label)."""
    rng = np.random.default_rng(seed)
    v = k * words_per_topic
    centers = np.zeros((k, e))
    for t in range(k):
        centers[t, t] = 10.0
    emb = np.zeros((v, e))
    for w in range(v):
        emb[w] = centers[w // words_per_topic] + rng.normal(0, 0.3, size=e)
    vocab = [f"w{i:03d}" for i in range(v)]
    docs, labels = [], []
    for _ in range(n_docs):
        t = int(rng.integers(0, k))
        labels.append(t)
        docs.append([vocab[t * words_per_topic + int(rng.integers(0, words_per_topic))]
                     for _ in range(tokens)])
    return docs, emb, vocab, np.array(labels)


def test_shapes_and_normalization():
    docs, emb, vocab, _ = _planted()
    m = topica.GaussianLDA(3, seed=0).fit(docs, emb, vocab, iters=30)
    V = len(vocab)
    assert m.topic_word.shape == (3, V)
    assert np.allclose(m.topic_word.sum(axis=1), 1.0)
    assert m.doc_topic.shape == (len(docs), 3)
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.topic_means.shape == (3, emb.shape[1])
    assert m.topic_covariances.shape == (3, emb.shape[1], emb.shape[1])
    assert m.topic_scale_matrices.shape == (3, emb.shape[1], emb.shape[1])
    assert sum(m.topic_counts) == sum(len(d) for d in docs)
    # log-likelihood history: iters + 1 entries (post-init + one per sweep)
    assert len(m.log_likelihood_history) == 31


def test_determinism():
    docs, emb, vocab, _ = _planted(seed=1)
    a = topica.GaussianLDA(3, seed=7).fit(docs, emb, vocab, iters=30)
    b = topica.GaussianLDA(3, seed=7).fit(docs, emb, vocab, iters=30)
    assert np.array_equal(a.topic_word, b.topic_word)
    assert np.array_equal(a.doc_topic, b.doc_topic)
    assert np.array_equal(a.topic_means, b.topic_means)


def _cluster_recovery(m, emb, vocab, word_topic, k):
    """Mean aligned cosine of the fitted topic means to the true cluster centers."""
    from scipy.optimize import linear_sum_assignment

    centers = np.array([emb[word_topic == t].mean(0) for t in range(k)])
    a = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    b = np.asarray(m.topic_means)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = a @ b.T
    r, c = linear_sum_assignment(-sim)
    return float(np.mean([sim[i, c[i]] for i in range(k)]))


def test_recovers_planted_topics_kmeans():
    # k-means init (default) recovers well-separated clusters reliably (no collapse).
    k, wpt = 3, 12
    docs, emb, vocab, _ = _planted(k=k, words_per_topic=wpt, seed=2)
    word_topic = np.array([i // wpt for i in range(len(vocab))])
    m = topica.GaussianLDA(k, seed=0).fit(docs, emb, vocab, iters=60)
    assert _cluster_recovery(m, emb, vocab, word_topic, k) > 0.98
    # every topic is used (k-means init avoids the empty-topic collapse of random init)
    assert all(c > 0 for c in m.topic_counts)


def test_random_init_option():
    # The reference Cholesky sampler's random-init path is available and runs. (Random
    # init is collapse-prone on small data, so a collapse warning here is expected.)
    import warnings

    docs, emb, vocab, _ = _planted(seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.GaussianLDA(3, init="random", seed=0).fit(docs, emb, vocab, iters=30)
    assert m.doc_topic.shape == (len(docs), 3)
    assert m.settings["init"] == "random"


def test_transform_new_docs():
    docs, emb, vocab, _ = _planted(seed=4)
    m = topica.GaussianLDA(3, seed=0).fit(docs, emb, vocab, iters=40)
    theta = m.transform(docs[:5])
    assert theta.shape == (5, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)


def test_save_load_roundtrip(tmp_path):
    docs, emb, vocab, _ = _planted(seed=5)
    m = topica.GaussianLDA(3, seed=0).fit(docs, emb, vocab, iters=30)
    p = str(tmp_path / "glda.bin")
    m.save(p)
    m2 = topica.GaussianLDA.load(p)
    assert np.array_equal(m.topic_word, m2.topic_word)
    assert np.array_equal(m.doc_topic, m2.doc_topic)
    assert np.array_equal(m.topic_means, m2.topic_means)
    assert m2.settings == m.settings
    # transform still works after load (word densities were persisted)
    assert m2.transform(docs[:3]).shape == (3, 3)


def test_covariance_vs_scale_matrix():
    # topic_covariances = Psi_k / (nu_k - E - 1); topic_scale_matrices = Psi_k.
    docs, emb, vocab, _ = _planted(seed=6)
    E = emb.shape[1]
    m = topica.GaussianLDA(3, seed=0).fit(docs, emb, vocab, iters=30)
    nu0 = float(E)  # default nu clamps to E
    for t in range(3):
        nu_k = nu0 + m.topic_counts[t]
        expected = np.asarray(m.topic_scale_matrices)[t] / (nu_k - E - 1)
        assert np.allclose(np.asarray(m.topic_covariances)[t], expected)


def test_settings_keys():
    m = topica.GaussianLDA(4, alpha=0.2, kappa=0.5, nu=25.0, psi_scale=2.0, seed=9)
    s = m.settings
    assert s["num_topics"] == 4
    assert s["alpha"] == 0.2
    assert s["kappa"] == 0.5
    assert s["nu"] == 25.0
    assert s["psi_scale"] == 2.0
    assert s["init"] == "kmeans"
    assert s["seed"] == 9


def test_student_t_density_matches_numpy():
    # topic_word must equal a direct multivariate Student-t computation from the exposed
    # mu_k / Psi_k / N_k. Odd E exercises the real 0.5*E normalizer (Java int-div bug).
    from scipy.special import gammaln

    for E in (7, 8):  # odd and even
        docs, emb, vocab, _ = _planted(k=3, e=E, words_per_topic=10, n_docs=40, seed=E)
        m = topica.GaussianLDA(3, seed=0).fit(docs, emb, vocab, iters=25)
        mu = np.asarray(m.topic_means)
        psi = np.asarray(m.topic_scale_matrices)
        N = np.array(m.topic_counts)
        nu0 = float(E)
        kappa0 = 0.1

        def logt(x, k):
            kn, nun = kappa0 + N[k], nu0 + N[k]
            nu = nu0 + N[k] - E + 1
            s = (kn + 1) / (kn * (nun - E + 1))
            L = np.linalg.cholesky(s * psi[k])
            y = np.linalg.solve(L, x - mu[k])
            val = y @ y
            logdet = 2 * np.log(np.diag(L)).sum()
            return gammaln((nu + E) / 2) - (
                gammaln(nu / 2) + 0.5 * E * (np.log(nu) + np.log(np.pi))
                + 0.5 * logdet + (nu + E) / 2 * np.log(1 + val / nu)
            )

        V = len(vocab)
        tw = np.zeros((3, V))
        for k in range(3):
            lg = np.array([logt(emb[w], k) for w in range(V)])
            tw[k] = np.exp(lg - lg.max())
            tw[k] /= tw[k].sum()
        assert np.abs(tw - np.asarray(m.topic_word)).max() < 1e-10


def test_collapse_warning_and_diagnostics():
    # A degenerate fit (K far exceeding the number of separable clusters on tiny data)
    # collapses; the model must WARN and surface it via n_effective_topics, not silently.
    rng = np.random.default_rng(0)
    # 2 tight clusters but ask for 6 topics -> guaranteed empties
    emb = np.vstack([rng.normal(0, 0.05, (10, 4)) + [9, 0, 0, 0],
                     rng.normal(0, 0.05, (10, 4)) + [0, 9, 0, 0]])
    vocab = [f"w{i}" for i in range(20)]
    docs = [[vocab[int(rng.integers(0, 10)) + (0 if d % 2 else 10)] for _ in range(15)]
            for d in range(40)]
    with pytest.warns(UserWarning, match="collapsed"):
        m = topica.GaussianLDA(6, seed=0).fit(docs, emb, vocab, iters=30)
    assert m.n_effective_topics < 6
    assert m.converged is False
    # empty topics have NaN covariance (no posterior mean), finite scale matrix
    cov = np.asarray(m.topic_covariances)
    scale = np.asarray(m.topic_scale_matrices)
    for t in range(6):
        if m.topic_counts[t] == 0:
            assert np.isnan(cov[t]).all()
            assert np.isfinite(scale[t]).all()


def test_effective_hyperparams():
    docs, emb, vocab, _ = _planted(seed=2)
    m = topica.GaussianLDA(3, seed=0).fit(docs, emb, vocab, iters=10)
    assert m.effective_alpha == pytest.approx(1.0 / 3.0)
    assert m.effective_nu == emb.shape[1]  # nu=None -> E


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        topica.GaussianLDA(0)
    with pytest.raises(ValueError):
        topica.GaussianLDA(2, kappa=0.0)
    with pytest.raises(ValueError):
        topica.GaussianLDA(2, init="nope")


def test_rejects_mismatched_embeddings():
    docs, emb, vocab, _ = _planted(seed=7)
    with pytest.raises(ValueError):
        topica.GaussianLDA(3, seed=0).fit(docs, emb[:-1], vocab, iters=5)


def test_rejects_empty_corpus():
    _, emb, vocab, _ = _planted(seed=8)
    with pytest.raises((ValueError, RuntimeError)):
        topica.GaussianLDA(3, seed=0).fit([], emb, vocab, iters=5)
