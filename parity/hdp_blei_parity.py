"""Parity: topica HDP (resample_conc=True) vs an independent NumPy
implementation of the blei-lab/hdp concentration-update *equations*
(Escobar-West/Teh 2006, `sample_first_level_concentration` /
`sample_second_level_concentration`).

Scope and caveats (deliberately narrow):
  * This validates topica's concentration-resampling equations against an
    independent reimplementation of the same equations — NOT against the blei-lab
    C++ binary. topica's `resample_gamma`/`resample_alpha` reproduce those routines
    up to numeric-safety guards.
  * Both topica and this oracle are *direct-assignment* samplers (tables
    marginalised, counts augmented via Antoniak sampling). blei-lab/hdp and
    tomotopy use an explicit per-word table (CRF) representation, so this check
    does NOT prove topica matches the reference table-based samplers' topic count;
    it confirms topica's own sampler is self-consistent with the equations and,
    crucially, lands far from the degenerate single-background-topic collapse that
    fixed low concentrations produce.

The two RNGs differ, so the match is statistical (a tolerance band on the
posterior), not bit-for-bit. topica + NumPy only; sklearn (the corpus source) is
optional. Run directly to print the comparison; imported by
tests/test_hdp_blei_parity.py for the assertions.
"""

import numpy as np


def synthetic_corpus(n_blocks=8, words_per_block=5, docs_per_block=25, noise=0.25, seed=0):
    """Planted-block corpus with cross-block noise so K is non-trivial but the
    canonical sampler still lands well above 1 topic."""
    rng = np.random.default_rng(seed)
    V = n_blocks * words_per_block
    docs = []
    for b in range(n_blocks):
        block = list(range(b * words_per_block, (b + 1) * words_per_block))
        for _ in range(docs_per_block):
            n = rng.integers(18, 30)
            d = []
            for _ in range(n):
                if rng.random() < noise:
                    d.append(int(rng.integers(0, V)))
                else:
                    d.append(int(rng.choice(block)))
            docs.append(d)
    return docs, V


def oracle_hdp(ids, V, alpha0, gamma0, eta, iters, seed=1):
    """Direct-assignment HDP with Escobar-West/Teh concentration resampling,
    written from the equations (independent of topica). Returns (K, alpha, gamma,
    top_topic_share)."""
    rng = np.random.default_rng(seed)
    D = len(ids)
    nkw, nk, beta = [], [], []
    njk = [np.zeros(0) for _ in range(D)]
    beta_u = 1.0
    z = [np.zeros(len(d), dtype=int) for d in ids]
    alpha, gamma = alpha0, gamma0

    def add_topic():
        nonlocal beta_u
        b = rng.beta(1.0, gamma)
        beta.append(b * beta_u)
        beta_u *= 1 - b
        nkw.append(np.zeros(V))
        nk.append(0.0)
        for jj in range(D):
            njk[jj] = np.append(njk[jj], 0.0)

    def assign(j, i, w):
        K = len(nk)
        p = np.empty(K + 1)
        for t in range(K):
            f = (nkw[t][w] + eta) / (nk[t] + V * eta)
            p[t] = (njk[j][t] + alpha * beta[t]) * f
        p[K] = alpha * beta_u * (1.0 / V)
        p /= p.sum()
        k = int(rng.choice(K + 1, p=p))
        if k == K:
            add_topic()
        nkw[k][w] += 1
        nk[k] += 1
        njk[j][k] += 1
        z[j][i] = k

    for j, d in enumerate(ids):
        for i, w in enumerate(d):
            assign(j, i, w)

    def resample_beta():
        K = len(nk)
        m = np.zeros(K)
        tj = np.zeros(D)
        for j in range(D):
            for t in range(K):
                n = int(njk[j][t])
                if n == 0:
                    continue
                a = alpha * beta[t]
                tbl = 1 + sum(1 for i in range(1, n) if rng.random() < a / (a + i))
                m[t] += tbl
                tj[j] += tbl
        g = np.array([rng.gamma(mk) if mk > 0 else 0.0 for mk in m])
        gu = rng.gamma(gamma)
        tot = g.sum() + gu
        for t in range(K):
            beta[t] = g[t] / tot
        return gu / tot, m.sum(), tj

    for _ in range(iters):
        for j, d in enumerate(ids):
            for i, w in enumerate(d):
                k = z[j][i]
                nkw[k][w] -= 1
                nk[k] -= 1
                njk[j][k] -= 1
                assign(j, i, w)
        keep = [t for t in range(len(nk)) if nk[t] > 0]
        if len(keep) < len(nk):
            remap = {t: i for i, t in enumerate(keep)}
            for j in range(D):
                for i in range(len(z[j])):
                    z[j][i] = remap[z[j][i]]
            nk[:] = [nk[t] for t in keep]
            nkw[:] = [nkw[t] for t in keep]
            beta[:] = [beta[t] for t in keep]
            for j in range(D):
                njk[j] = njk[j][keep]
        beta_u, m_total, tj = resample_beta()
        # Escobar-West first-level (gamma), a=b=1 prior. Matches blei
        # sample_first_level_concentration: pi numerator is a+K-1 = K (not K-1),
        # and the shape draws are a+K / a+K-1 = 1+K / K.
        K = len(nk)
        if m_total >= 1:
            eta_aux = max(rng.beta(gamma + 1.0, m_total), 1e-12)
            a = 1.0
            pi = (a + K - 1) / ((a + K - 1) + m_total * (1 - np.log(eta_aux)))
            shape = (a + K) if rng.random() < pi else (a + K - 1)
            gamma = rng.gamma(shape) / (1 - np.log(eta_aux))
        # Teh second-level (alpha), inner auxiliary loop
        doclen = [int(sum(njk[j])) for j in range(D)]
        T = tj.sum()
        for _ in range(20):
            slw = 0.0
            ss = 0.0
            for nj in doclen:
                if nj == 0:
                    continue
                slw += np.log(max(rng.beta(alpha + 1.0, nj), 1e-12))
                if rng.random() < nj / (nj + alpha):
                    ss += 1
            sh, rt = 1 + T - ss, 1 - slw
            if sh > 0 and rt > 0:
                alpha = rng.gamma(sh) / rt

    ntok = sum(len(d) for d in ids)
    sizes = sorted((int(x) for x in nk), reverse=True)
    return len(nk), alpha, gamma, (sizes[0] / ntok if sizes else 1.0)


def newsgroups_corpus(n_docs=80, min_len=5, seed=0):
    """A 20-newsgroups subset, integer-encoded. Its rich, heavy-tailed vocabulary
    drives the canonical HDP into the multi-topic regime (a clean planted-block
    corpus collapses to K=1 under gamma=0.1, which is uninformative for parity).
    Returns (int_docs, str_docs, V) or raises ImportError if sklearn is absent."""
    import re

    from sklearn.datasets import fetch_20newsgroups  # noqa: PLC0415

    raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data[:n_docs]
    tok = [[w for w in re.findall(r"[a-z]{3,}", t.lower())] for t in raw]
    tok = [d for d in tok if len(d) >= min_len]
    vocab = {}
    for d in tok:
        for w in d:
            vocab.setdefault(w, len(vocab))
    ids = [[vocab[w] for w in d] for d in tok]
    return ids, tok, len(vocab)


def run(iters=100, n_docs=80):
    import topica

    ids, str_docs, V = newsgroups_corpus(n_docs=n_docs)

    ok, oa, og, oshare = oracle_hdp(ids, V, 0.1, 0.1, 0.01, iters=iters, seed=1)

    m = topica.HDP(alpha=0.1, gamma=0.1, beta=0.01, seed=1)  # resample_conc=True default
    m.fit(str_docs, iters=iters)
    dt = np.asarray(m.doc_topic)
    dl = np.array([len(d) for d in str_docs])
    tok = (dt * dl[:, None]).sum(0)
    tk, ta, tg = dt.shape[1], m.alpha, m.gamma
    tshare = float(tok.max() / tok.sum())

    # Degenerate fixed-low-concentration reference (what the old default did).
    mf = topica.HDP(alpha=0.1, gamma=0.1, beta=0.01, resample_conc=False, seed=1)
    mf.fit(str_docs, iters=iters)
    fdt = np.asarray(mf.doc_topic)
    ftok = (fdt * dl[:, None]).sum(0)
    fk = fdt.shape[1]
    fshare = float(ftok.max() / ftok.sum())

    return {
        "oracle": {"K": ok, "alpha": oa, "gamma": og, "top_share": oshare},
        "topica": {"K": tk, "alpha": ta, "gamma": tg, "top_share": tshare},
        "topica_fixed": {"K": fk, "top_share": fshare},
        "V": V,
        "n_docs": len(str_docs),
    }


if __name__ == "__main__":
    r = run()
    print(f"20NG subset          D={r['n_docs']}  V={r['V']}")
    o, t = r["oracle"], r["topica"]
    print(f"oracle (Teh eqns)    K={o['K']:3d}  alpha={o['alpha']:.2f}  gamma={o['gamma']:.1f}  top_share={o['top_share']:.1%}")
    print(f"topica (est-conc)    K={t['K']:3d}  alpha={t['alpha']:.2f}  gamma={t['gamma']:.1f}  top_share={t['top_share']:.1%}")
    f = r["topica_fixed"]
    print(f"topica fixed-conc    K={f['K']:3d}  top_share={f['top_share']:.1%}  (degenerate reference)")
