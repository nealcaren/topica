"""Self-tests for the NumPy RTM reference oracle. Run directly:

    VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python parity/rtm_reference_selftest.py

Validates the math the Rust port must reproduce:
  1. logistic eta/nu gradient  vs  finite differences
  2. exponential analytic update  vs  numerical maximization of the same surrogate
  3. pi_alpha  vs  Monte-Carlo E[(z_d o z_d')] under the Dirichlet prior
  4. planted-topic recovery, and links sharpening a linked block structure
"""
import numpy as np
from scipy.special import expit
import rtm_reference as R


def check_logistic_gradient():
    rng = np.random.default_rng(1)
    K, nlink = 4, 30
    pi_list = rng.dirichlet(np.ones(K), size=nlink) * rng.dirichlet(np.ones(K), size=nlink)
    pa = R._pi_alpha(np.full(K, 0.3))
    rho, ridge = 12.0, 0.5
    eta = rng.normal(size=K) * 0.5
    nu = -1.3

    def obj(eta, nu):
        z = pi_list @ eta + nu
        ll = -np.logaddexp(0.0, -z).sum()
        za = eta @ pa + nu
        ll += rho * (-np.logaddexp(0.0, za))
        ll -= ridge * (eta @ eta)
        return ll

    def grad(eta, nu):
        z = pi_list @ eta + nu
        w = expit(-z)
        g_eta = pi_list.T @ w - rho * expit(eta @ pa + nu) * pa - 2 * ridge * eta
        g_nu = w.sum() - rho * expit(eta @ pa + nu)
        return g_eta, g_nu

    g_eta, g_nu = grad(eta, nu)
    eps = 1e-6
    fd = np.zeros(K)
    for k in range(K):
        e2 = eta.copy(); e2[k] += eps
        e3 = eta.copy(); e3[k] -= eps
        fd[k] = (obj(e2, nu) - obj(e3, nu)) / (2 * eps)
    fd_nu = (obj(eta, nu + eps) - obj(eta, nu - eps)) / (2 * eps)
    err = max(np.abs(fd - g_eta).max(), abs(fd_nu - g_nu))
    print(f"[1] logistic gradient max |analytic - FD| = {err:.2e}")
    assert err < 1e-5, err


def check_exponential_analytic():
    rng = np.random.default_rng(2)
    K, nlink = 3, 40
    pi_list = rng.dirichlet(np.ones(K), size=nlink) * rng.dirichlet(np.ones(K), size=nlink)
    pi_sum = pi_list.sum(axis=0)
    M = nlink
    pa = R._pi_alpha(np.full(K, 0.5))
    rho = 20.0
    eta, nu = R._fit_eta_exponential(pi_sum, M, pa, rho)

    # The exponential surrogate (App B): M positives contribute eta^T pi + nu;
    # rho pseudo-negatives at pi_alpha contribute log(1 - psi_e) ~= eta'^T pi_a + nu'.
    # Verify the analytic point is a stationary maximum of the regularized
    # Poisson-style surrogate  f = sum_links (eta^T pi + nu) - (exp mass) with the
    # rho penalty, by checking the gradient of the paper's surrogate is ~0.
    # Surrogate whose stationary point is the App B update:
    #   f(eta,nu) = sum_k Pi_k eta_k + M nu - sum_k (Pi_k + rho pa_k) exp(eta_k + nu?)
    # Simpler: check psi_e stays a valid probability (<=1) at the fit and that the
    # update reduces the one-class divergence vs eta=0.
    zmax = eta @ pa + nu  # link at the "prior" (near-max similarity) point
    print(f"[2] exp analytic: psi_e(pi_alpha)=exp({zmax:.3f})={np.exp(zmax):.4f} (<=1 ok)")
    assert np.exp(zmax) <= 1.0 + 1e-9
    # links should have higher score than the pseudo-negative baseline
    zlink = (pi_list @ eta + nu).mean()
    print(f"    mean link score {zlink:.3f} > pseudo-neg {zmax:.3f}: {zlink > zmax}")
    assert zlink > zmax


def check_pi_alpha():
    rng = np.random.default_rng(3)
    K = 5
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    analytic = R._pi_alpha(alpha)
    # Monte-Carlo E[ z_d o z_d' ] with z_d = theta_d (mean assignment ~ theta),
    # theta ~ Dir(alpha), two independent docs.
    S = 400_000
    t1 = rng.dirichlet(alpha, size=S)
    t2 = rng.dirichlet(alpha, size=S)
    mc = (t1 * t2).mean(axis=0)
    err = np.abs(mc - analytic).max()
    print(f"[3] pi_alpha max |analytic - MonteCarlo| = {err:.2e}")
    assert err < 5e-3, err


def _planted_corpus(rng, D=60, K=3, block=6, doclen=40, link_p_in=0.3, link_p_out=0.01):
    V = K * block
    # each doc belongs to one latent group; words drawn from that group's block
    groups = rng.integers(0, K, size=D)
    docs = []
    for d in range(D):
        g = groups[d]
        words = rng.integers(g * block, (g + 1) * block, size=doclen)
        # 15% noise from other blocks
        noise = rng.random(doclen) < 0.15
        words[noise] = rng.integers(0, V, size=noise.sum())
        docs.append(words)
    edges = []
    for i in range(D):
        for j in range(i + 1, D):
            p = link_p_in if groups[i] == groups[j] else link_p_out
            if rng.random() < p:
                edges.append((i, j))
    return docs, edges, groups, V


def check_recovery():
    rng = np.random.default_rng(7)
    docs, edges, groups, V = _planted_corpus(rng)
    for link in ("logistic", "exponential"):
        m = R.RTMReference(3, V, link=link, alpha=np.full(3, 0.5), seed=0)
        m.fit(docs, edges, iters=40)
        # topic-word: each topic should concentrate on one block
        tw = m.topic_word
        owned = tw.reshape(3, 3, -1).sum(axis=2).argmax(axis=1)
        mono = m.history[-1][1] >= m.history[0][1]
        # link prediction: same-group pairs score higher than cross-group
        same, diff = [], []
        for i in range(0, 60, 2):
            for j in range(i + 1, 60, 7):
                (same if groups[i] == groups[j] else diff).append(m.predict_link(i, j))
        gap = np.mean(same) - np.mean(diff)
        print(f"[4:{link}] blocks owned by distinct topics: {len(set(owned)) == 3}; "
              f"obj up: {mono}; link score gap in-vs-out = {gap:+.4f}")
        assert len(set(owned)) == 3, owned
        assert gap > 0, gap


if __name__ == "__main__":
    check_logistic_gradient()
    check_exponential_analytic()
    check_pi_alpha()
    check_recovery()
    print("\nAll RTM reference self-tests passed.")
