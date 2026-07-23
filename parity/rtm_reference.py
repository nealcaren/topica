"""Standalone NumPy reference for the Relational Topic Model (Chang & Blei,
AOAS 2010, "Hierarchical Relational Models for Document Networks").

This is the PRIMARY NUMERICAL ORACLE for topica's Rust RTM port. The R `lda`
package's `rtm.em` is a *collapsed Gibbs* sampler (its body calls
`rtm.collapsed.gibbs.sampler`), not the paper's variational EM, so it can only be
a directional baseline. This file implements the paper's variational EM exactly,
so the Rust core can be finite-difference- and value-checked against it.

Faithful to the 2010 paper:
  - Generative model of §2, undirected links.
  - Variational EM of §3 / Appendix A-B.
  - phi update eq (3.5)/(3.6); the link message is constant across tokens in a
    doc, so we work word-type-major.
  - Link functions: logistic psi_sigma (eq 2.1) and exponential psi_e (eq 2.2).
  - rho regularization for the one-class link problem (App B); pi_alpha is the
    expected Hadamard product under the Dirichlet prior.

Not a topica import target — a self-contained reference used by parity checks and
during development. Pure NumPy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln, expit  # expit = logistic sigmoid


# --------------------------------------------------------------------------- #
# Corpus helpers
# --------------------------------------------------------------------------- #
def _doc_word_types(doc_ids: np.ndarray, vocab_size: int):
    """Collapse a token-id list into (unique_word_ids, counts)."""
    if len(doc_ids) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    uw, cnt = np.unique(doc_ids, return_counts=True)
    return uw.astype(np.int64), cnt.astype(np.float64)


def _symmetrize_edges(edges, num_docs):
    """Undirected adjacency list from an edge iterable; drop self-loops/dupes."""
    adj = [set() for _ in range(num_docs)]
    obs = set()
    for i, j in edges:
        if i == j:
            continue
        if not (0 <= i < num_docs and 0 <= j < num_docs):
            raise ValueError(f"edge ({i},{j}) out of range for {num_docs} docs")
        a, b = (i, j) if i < j else (j, i)
        obs.add((a, b))
        adj[i].add(j)
        adj[j].add(i)
    return [np.array(sorted(s), dtype=np.int64) for s in adj], sorted(obs)


# --------------------------------------------------------------------------- #
# Link functions: expected log-likelihood term and its gradient wrt pi_bar
# --------------------------------------------------------------------------- #
def _link_grad_pi(link, eta, nu, pi_bar):
    """grad_{pi_bar} E_q[log psi(pi_bar)]  (a K-vector), eqs after (3.6).

    logistic:  (1 - sigma(eta^T pi_bar + nu)) * eta
    exponential: eta                       (exact, since E[log psi_e] = eta^T pi_bar + nu)
    """
    if link == "exponential":
        return eta
    s = expit(eta @ pi_bar + nu)
    return (1.0 - s) * eta


def _pi_alpha(alpha):
    """Expected Hadamard product under the Dirichlet prior: (a/1^T a) o (a/1^T a)."""
    p = alpha / alpha.sum()
    return p * p


# --------------------------------------------------------------------------- #
# M-step for the link parameters (eta, nu)
# --------------------------------------------------------------------------- #
def _link_objective_logistic(eta, nu, pi_sum, num_links, pi_alpha, rho, ridge):
    """Regularized logistic log-likelihood: M real positives (aggregated via
    their summed feature pi_sum only enters the gradient; the objective needs the
    per-link dot products, so we pass those separately) — see _fit_eta_logistic.
    """
    raise NotImplementedError  # objective computed inline where per-link feats live


def _fit_eta_logistic(pi_list, pi_alpha, rho, ridge, eta0, nu0, iters=200, lr=None):
    """Maximize the rho-regularized logistic link log-likelihood by gradient
    ascent with backtracking. pi_list: (num_links, K) array of per-link pi_bar.

    Objective  L(eta,nu) = sum_links log sigma(eta^T pi + nu)
                           + rho * log(1 - sigma(eta^T pi_alpha + nu))
                           - ridge * ||eta||^2
    Gradient (eta):  sum_links (1-sigma_l) pi  - rho sigma_alpha pi_alpha - 2 ridge eta
    Gradient (nu):   sum_links (1-sigma_l)      - rho sigma_alpha
    Concave (sum of logistic log-likelihoods) so gradient ascent converges.
    """
    K = pi_alpha.shape[0]
    eta = eta0.copy()
    nu = float(nu0)

    def obj(eta, nu):
        z = pi_list @ eta + nu
        ll = -np.logaddexp(0.0, -z).sum()  # sum log sigma(z)
        za = eta @ pi_alpha + nu
        ll += rho * (-np.logaddexp(0.0, za))  # rho * log(1 - sigma) = rho*log sigma(-za)
        ll -= ridge * (eta @ eta)
        return ll

    def grad(eta, nu):
        z = pi_list @ eta + nu
        w = expit(-z)  # 1 - sigma(z)
        g_eta = pi_list.T @ w
        g_nu = w.sum()
        sa = expit(eta @ pi_alpha + nu)
        g_eta -= rho * sa * pi_alpha
        g_nu -= rho * sa
        g_eta -= 2.0 * ridge * eta
        return g_eta, g_nu

    step = 1.0 / max(1.0, len(pi_list)) if lr is None else lr
    f = obj(eta, nu)
    for _ in range(iters):
        g_eta, g_nu = grad(eta, nu)
        gnorm = np.sqrt(g_eta @ g_eta + g_nu * g_nu)
        if gnorm < 1e-8:
            break
        s = step
        # backtracking line search
        for _ls in range(30):
            e2 = eta + s * g_eta
            n2 = nu + s * g_nu
            f2 = obj(e2, n2)
            if f2 >= f + 1e-4 * s * (gnorm ** 2):
                break
            s *= 0.5
        if f2 <= f:  # no progress
            break
        eta, nu, f = e2, n2, f2
    return eta, nu


def _fit_eta_exponential(pi_sum, num_links, pi_alpha, rho):
    """Analytic exponential-link update (App B, page 147).

    nu <- log(M - 1^T Pi_bar) - log(rho (1 - 1^T pi_alpha) + M - 1^T Pi_bar)
    eta <- log(Pi_bar) - log(Pi_bar + rho pi_alpha) - 1 nu
    where M = num observed links, Pi_bar = sum over links of pi_bar (a K-vector).
    Guards against zero sufficient statistics.
    """
    M = float(num_links)
    Pi = pi_sum.astype(np.float64)
    one_T_Pi = Pi.sum()
    one_T_pa = pi_alpha.sum()
    eps = 1e-12
    resid = max(M - one_T_Pi, eps)
    nu = np.log(resid) - np.log(max(rho * (1.0 - one_T_pa), 0.0) + resid)
    eta = np.log(np.maximum(Pi, eps)) - np.log(Pi + rho * pi_alpha + eps) - nu
    return eta, float(nu)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class RTMReference:
    def __init__(self, num_topics, vocab_size, *, link="logistic", alpha=None,
                 rho=None, negative_ratio=1.0, ridge=1.0, seed=0):
        if link not in ("logistic", "exponential"):
            raise ValueError("link must be 'logistic' or 'exponential'")
        self.K = int(num_topics)
        self.V = int(vocab_size)
        self.link = link
        self.alpha = (np.full(self.K, 1.0 / self.K) if alpha is None
                      else np.asarray(alpha, dtype=np.float64))
        self.negative_ratio = float(negative_ratio)
        self.rho_arg = rho
        self.ridge = float(ridge)
        self.seed = int(seed)
        # fitted state
        self.log_beta = None      # K x V
        self.eta = None           # K
        self.nu = None            # scalar
        self.phi_bar = None       # D x K  (mean topic assignment; the link quantity)
        self.gamma = None         # D x K
        self.history = []

    # ---- inference ---------------------------------------------------------
    def fit(self, docs_ids, edges, *, iters=50, e_sweeps=3, e_inner=5,
            var_tol=1e-4, damping=1.0):
        rng = np.random.default_rng(self.seed)
        D = len(docs_ids)
        K, V = self.K, self.V
        docs = [np.asarray(d, dtype=np.int64) for d in docs_ids]
        types = [_doc_word_types(d, V) for d in docs]
        Nd = np.array([float(len(d)) for d in docs])
        adj, obs_links = _symmetrize_edges(edges, D)
        M = len(obs_links)
        # rho as a count: negative_ratio * (#links) unless an explicit rho given.
        rho = self.rho_arg if self.rho_arg is not None else self.negative_ratio * max(M, 1)
        pa = _pi_alpha(self.alpha)

        # init: beta from smoothed random counts; phi_bar uniform.
        beta = rng.dirichlet(np.full(V, 1.0), size=K) + 1e-8
        beta /= beta.sum(axis=1, keepdims=True)
        log_beta = np.log(beta)
        phi_bar = np.full((D, K), 1.0 / K)
        gamma = self.alpha[None, :] + Nd[:, None] / K
        phi_store = [np.full((len(uw), K), 1.0 / K) for uw, _ in types]
        eta = np.zeros(K)
        nu = 0.0

        for it in range(iters):
            # ---------------- E-step (coupled, serial Gauss-Seidel) ----------
            for _sweep in range(e_sweeps):
                max_delta = 0.0
                for d in range(D):
                    uw, cnt = types[d]
                    if len(uw) == 0:
                        continue
                    elogtheta = digamma(gamma[d]) - digamma(gamma[d].sum())
                    # link message m_d (constant across tokens), eq (3.5)/(3.6)
                    m_d = np.zeros(K)
                    if len(adj[d]) > 0 and Nd[d] > 0:
                        for dp in adj[d]:
                            pib = phi_bar[d] * phi_bar[dp]
                            g = _link_grad_pi(self.link, eta, nu, pib)
                            m_d += g * phi_bar[dp]
                        m_d /= Nd[d]
                    prev = phi_bar[d].copy()
                    for _inner in range(e_inner):
                        base = log_beta[:, uw].T + elogtheta[None, :] + m_d[None, :]
                        base -= base.max(axis=1, keepdims=True)
                        phi = np.exp(base)
                        phi /= phi.sum(axis=1, keepdims=True)
                        phi_store[d] = phi
                        new_bar = (cnt[:, None] * phi).sum(axis=0) / Nd[d]
                        if damping < 1.0:
                            new_bar = damping * new_bar + (1 - damping) * phi_bar[d]
                        phi_bar[d] = new_bar
                        gamma[d] = self.alpha + cnt @ phi
                        elogtheta = digamma(gamma[d]) - digamma(gamma[d].sum())
                        # recompute logistic message with updated phi_bar[d]
                        if self.link == "logistic" and len(adj[d]) > 0:
                            m_d = np.zeros(K)
                            for dp in adj[d]:
                                pib = phi_bar[d] * phi_bar[dp]
                                g = _link_grad_pi(self.link, eta, nu, pib)
                                m_d += g * phi_bar[dp]
                            m_d /= Nd[d]
                    max_delta = max(max_delta, np.abs(phi_bar[d] - prev).max())
                if max_delta < var_tol:
                    break

            # ---------------- M-step: beta ----------------------------------
            ss = np.full((K, V), 1e-6)
            for d in range(D):
                uw, cnt = types[d]
                if len(uw) == 0:
                    continue
                np.add.at(ss.T, uw, cnt[:, None] * phi_store[d])
            beta = ss / ss.sum(axis=1, keepdims=True)
            log_beta = np.log(beta)

            # ---------------- M-step: eta, nu -------------------------------
            if M > 0:
                pi_list = np.array([phi_bar[i] * phi_bar[j] for (i, j) in obs_links])
                pi_sum = pi_list.sum(axis=0)
                if self.link == "exponential":
                    eta, nu = _fit_eta_exponential(pi_sum, M, pa, rho)
                else:
                    eta, nu = _fit_eta_logistic(pi_list, pa, rho, self.ridge,
                                                eta, nu)

            obj = self._objective(types, phi_store, phi_bar, gamma, log_beta,
                                   eta, nu, obs_links, pa, rho)
            self.history.append((it, obj))

        self.log_beta = log_beta
        self.eta = eta
        self.nu = nu
        self.phi_bar = phi_bar
        self.gamma = gamma
        self._obs_links = obs_links
        return self

    # ---- objective (link + word log-likelihood, for monotonicity) ----------
    def _objective(self, types, phi_store, phi_bar, gamma, log_beta, eta, nu,
                   obs_links, pa, rho):
        K = self.K
        word_ll = 0.0
        z_ll = 0.0
        for d in range(len(types)):
            uw, cnt = types[d]
            if len(uw) == 0:
                continue
            elogtheta = digamma(gamma[d]) - digamma(gamma[d].sum())
            phi = phi_store[d]
            word_ll += (cnt[:, None] * phi * log_beta[:, uw].T).sum()
            z_ll += (cnt[:, None] * phi * (elogtheta[None, :] - np.log(phi + 1e-300))).sum()
        link_ll = 0.0
        for (i, j) in obs_links:
            pib = phi_bar[i] * phi_bar[j]
            z = eta @ pib + nu
            if self.link == "exponential":
                link_ll += z
            else:
                link_ll += -np.logaddexp(0.0, -z)
        return float(word_ll + z_ll + link_ll)

    # ---- fitted-model surface ---------------------------------------------
    @property
    def topic_word(self):
        return np.exp(self.log_beta)

    @property
    def doc_topic(self):
        return self.gamma / self.gamma.sum(axis=1, keepdims=True)

    def predict_link(self, i, j):
        """Plug-in link probability psi(phi_bar_i o phi_bar_j)."""
        pib = self.phi_bar[i] * self.phi_bar[j]
        z = self.eta @ pib + self.nu
        return float(np.exp(z)) if self.link == "exponential" else float(expit(z))

    def infer_phi_bar(self, doc_ids, *, iters=50):
        """Cold-start: infer phi_bar for a NEW doc from words only (no links)."""
        uw, cnt = _doc_word_types(np.asarray(doc_ids, dtype=np.int64), self.V)
        if len(uw) == 0:
            return np.full(self.K, 1.0 / self.K)
        N = cnt.sum()
        gamma = self.alpha + N / self.K
        phi = np.full((len(uw), self.K), 1.0 / self.K)
        for _ in range(iters):
            elogtheta = digamma(gamma) - digamma(gamma.sum())
            base = self.log_beta[:, uw].T + elogtheta[None, :]
            base -= base.max(axis=1, keepdims=True)
            phi = np.exp(base)
            phi /= phi.sum(axis=1, keepdims=True)
            gamma = self.alpha + cnt @ phi
        return (cnt[:, None] * phi).sum(axis=0) / N

    def suggest_links(self, doc_ids, top_n=20, exclude=None):
        pb = self.infer_phi_bar(doc_ids)
        exclude = set() if exclude is None else set(exclude)
        scores = []
        for d in range(self.phi_bar.shape[0]):
            if d in exclude:
                continue
            pib = pb * self.phi_bar[d]
            z = self.eta @ pib + self.nu
            p = np.exp(z) if self.link == "exponential" else expit(z)
            scores.append((d, float(p)))
        scores.sort(key=lambda t: -t[1])
        return scores[:top_n]
