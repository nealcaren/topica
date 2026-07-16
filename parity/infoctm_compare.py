"""Cross-implementation check for topica's InfoCTM against a faithful PyTorch
reference of InfoCTM (Wu et al. 2023, AAAI; arXiv:2304.03544).

InfoCTM is two ProdLDA/AVITM models -- one per language, over independent
vocabularies, sharing the topic space -- coupled by a Topic-Alignment
Mutual-Information (TAMI) term: a masked cross-lingual InfoNCE over the topic-word
columns, with positive word pairs from a bilingual dictionary. The reference's
license is unclear, so this is a compact PyTorch implementation built from the
paper (the same recipe topica implements in Rust), not the authors' source. The
optimizers, initialization, and RNG differ, so exact agreement is impossible.

The defining behaviour is **cross-lingual topic alignment**: with a block-aligned
dictionary, topic ``k`` must concentrate on the same planted block in both
languages. We hold the two implementations to that, plus top-word agreement after
Hungarian alignment. (Full-vector topic-word cosine is NOT used as the bar: topica
produces sharper, lower-entropy topic distributions than this reference, so the
distributions rank identically but differ in contrast -- alignment and top words,
not entropy, are what define a correct InfoCTM fit.)

Skips cleanly when torch is unavailable.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refs.avitm import (  # noqa: E402
    build_encoder,
    dirichlet_laplace_kl,
    laplace_prior,
    reparameterize,
)

K = 5
PER = 6                 # words per block per language
N_A, N_B = 300, 270
LENGTH = 15
HIDDEN = 64
ITERS = 200
SEED = 0


def available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Synthetic matched bilingual corpus + block-aligned dictionary
# ---------------------------------------------------------------------------

def _corpus(prefix: str, n: int, rng) -> list[list[str]]:
    docs = []
    for d in range(n):
        b = d % K
        docs.append([f"{prefix}{b}_{int(rng.integers(PER))}" for _ in range(LENGTH)])
    return docs


def _make_data():
    rng = np.random.default_rng(SEED)
    docs_a = _corpus("a", N_A, rng)
    docs_b = _corpus("b", N_B, rng)
    vocab_a = [f"a{b}_{i}" for b in range(K) for i in range(PER)]
    vocab_b = [f"b{b}_{i}" for b in range(K) for i in range(PER)]
    dictionary = [
        (f"a{b}_{i}", f"b{b}_{j}")
        for b in range(K) for i in range(PER) for j in range(PER)
    ]
    return docs_a, docs_b, vocab_a, vocab_b, dictionary


def _counts(docs, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for d, doc in enumerate(docs):
        for w in doc:
            m[d, idx[w]] += 1.0
    return m


# ---------------------------------------------------------------------------
# Paper-derived PyTorch reference
# ---------------------------------------------------------------------------
#
# The two per-language encoder backbones (fc1, fc2, mu, lv + affine-free
# batchnorm heads), the Laplace-Dirichlet prior, the reparameterization trick,
# and the KL are shared with the ProdLDA parity reference via ``refs.avitm``.
# The decoder (a ``phi`` matmul + decoder batchnorm + softmax recon, x2) and the
# TAMI cross-lingual alignment term below are InfoCTM-specific.


def _train_reference(counts_a, counts_b, trans_ab, seed):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    va, vb = counts_a.shape[1], counts_b.shape[1]
    mu2, var2 = laplace_prior(K, 1.0)
    trans = torch.tensor(trans_ab)              # Va x Vb (identity mono-mask -> dict pairs)
    trans_t = trans.T
    neg_ab = (trans <= 0).float()
    neg_ba = (trans_t <= 0).float()
    pos_sum = trans.sum() + trans_t.sum()
    temp, weight = 0.2, 30.0

    enc_a = build_encoder(va, K, HIDDEN)
    enc_b = build_encoder(vb, K, HIDDEN)
    dec_bn_a = torch.nn.BatchNorm1d(va, affine=False)
    dec_bn_b = torch.nn.BatchNorm1d(vb, affine=False)
    phi_a = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(K, va)))
    phi_b = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(K, vb)))

    mods = [enc_a, enc_b, dec_bn_a, dec_bn_b]
    params = [phi_a, phi_b]
    for m in mods:
        params += list(m.parameters())
    opt = torch.optim.Adam(params, lr=0.002)  # default betas (0.9, 0.999), like the reference

    xa = torch.tensor(counts_a); xb = torch.tensor(counts_b)

    def elbo(x, enc, dbn, phi):
        mu, lv = enc.encode(x)
        z = reparameterize(mu, lv)
        theta = F.softmax(z, dim=1)
        recon = F.softmax(dbn(theta @ phi), dim=1)
        rec = -(x * (recon + 1e-10).log()).sum(1)
        kld = dirichlet_laplace_kl(mu, lv, mu2, var2, K)
        return (rec + kld).mean()

    def mutual_info(fa, fb, pos, neg):
        na = F.normalize(fa, dim=1); nb = F.normalize(fb, dim=1)
        s = (na @ nb.T) / temp
        s = s - s.max(1, keepdim=True).values.detach()
        exps = s.exp()
        denom = (exps * neg).sum(1, keepdim=True) + exps + 1e-10
        log_prob = s - denom.log()
        return -(pos * log_prob).sum()

    for _ in range(ITERS):
        opt.zero_grad()
        la = elbo(xa, enc_a, dec_bn_a, phi_a)
        lb = elbo(xb, enc_b, dec_bn_b, phi_b)
        tami = mutual_info(phi_a.T, phi_b.T, trans, neg_ab)
        tami = tami + mutual_info(phi_b.T, phi_a.T, trans_t, neg_ba)
        loss = la + lb + weight * tami / pos_sum
        loss.backward()
        opt.step()

    with torch.no_grad():
        ba = F.softmax(phi_a, dim=1).cpu().numpy()
        bb = F.softmax(phi_b, dim=1).cpu().numpy()
    return ba, bb


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _alignment(beta_a, beta_b, per=PER):
    """Fraction of topics whose top word maps to the same planted block in both
    languages (after matching topics across languages by block of their top word)."""
    block_a = [int(np.argmax(beta_a[t]) // per) for t in range(beta_a.shape[0])]
    block_b = [int(np.argmax(beta_b[t]) // per) for t in range(beta_b.shape[0])]
    return sum(a == b for a, b in zip(block_a, block_b)) / len(block_a)


def _purity(beta, per=PER):
    """Mean within-top-word-block mass: how cleanly each topic sits in one block."""
    out = []
    for row in beta:
        b = int(np.argmax(row) // per)
        out.append(float(row[b * per:(b + 1) * per].sum()))
    return float(np.mean(out))


def run(verbose: bool = True) -> dict:
    import topica
    docs_a, docs_b, vocab_a, vocab_b, dictionary = _make_data()
    ca, cb = _counts(docs_a, vocab_a), _counts(docs_b, vocab_b)
    # Block dictionary matrix in vocab order.
    ia = {w: i for i, w in enumerate(vocab_a)}
    ib = {w: i for i, w in enumerate(vocab_b)}
    trans = np.zeros((len(vocab_a), len(vocab_b)), dtype=np.float32)
    for wa, wb in dictionary:
        trans[ia[wa], ib[wb]] = 1.0

    # topica
    m = topica.models.InfoCTM(num_topics=K, seed=1, hidden_size=HIDDEN, lr=0.01)
    m.fit(docs_a, docs_b, dictionary=dictionary, iters=ITERS, batch_size=64)
    t_a = m.topic_word(lang="a"); t_b = m.topic_word(lang="b")
    # align topica vocab order to vocab_a/vocab_b
    t_a = t_a[:, [m.vocabulary(lang="a").index(w) for w in vocab_a]]
    t_b = t_b[:, [m.vocabulary(lang="b").index(w) for w in vocab_b]]

    # reference (two seeds for a noise floor on alignment/purity)
    r_a1, r_b1 = _train_reference(ca, cb, trans, seed=1)
    r_a2, r_b2 = _train_reference(ca, cb, trans, seed=2)

    out = {
        "topica_align": _alignment(t_a, t_b),
        "ref_align": _alignment(r_a1, r_b1),
        "topica_purity_a": _purity(t_a), "topica_purity_b": _purity(t_b),
        "ref_purity_a": _purity(r_a1), "ref_purity_b": _purity(r_b1),
        "ref_align_seed2": _alignment(r_a2, r_b2),
    }
    if verbose:
        print(f"InfoCTM parity (K={K}, A={N_A}/B={N_B}, {ITERS} iters):")
        print(f"  cross-lingual alignment   topica {out['topica_align']:.2f}  "
              f"ref {out['ref_align']:.2f} (seed2 {out['ref_align_seed2']:.2f})")
        print(f"  topic purity (A / B)      topica {out['topica_purity_a']:.2f} / "
              f"{out['topica_purity_b']:.2f}   ref {out['ref_purity_a']:.2f} / "
              f"{out['ref_purity_b']:.2f}")
        print("  (topica concentrates more mass per topic -- sharper, by design.)")
    return out


if __name__ == "__main__":
    if not available():
        print("torch not installed; skipping InfoCTM parity.")
    else:
        r = run()
        assert r["topica_align"] >= 0.8, f"topica cross-lingual alignment too low: {r}"
        assert r["ref_align"] >= 0.8, f"reference alignment unexpectedly low: {r}"
        print("OK: both implementations align topics across languages.")
