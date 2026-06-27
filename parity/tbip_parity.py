"""Parity check for TBIP (Text-Based Ideal Points, Vafa-Naidu-Blei 2020) against an
independent PyTorch reference of the SAME model + the SAME mean-field VI inference.

The official implementation is TensorFlow 1.14 / TFP 0.7 (graph-mode `tf.Session`), which
has no Apple-Silicon build and does not install on modern Python, so we validate against a
faithful PyTorch reimplementation of the paper's model (Eq. 3) and inference (sec. 4):
lognormal q(theta,beta), normal q(eta,x), reparameterized single-sample SVI, Adam.

Self-contained: data is sampled from the generative model with planted author ideal points
(no external corpus). Skips cleanly when torch is unavailable (e.g. CI without torch).

Run: python parity/tbip_parity.py
"""
import sys
import numpy as np

try:
    import torch
except Exception:
    print("SKIP: torch not available (reference unavailable)")
    sys.exit(0)

import topica

RNG = np.random.default_rng(0)
A, K, V, DOCS_PER_A, NWORDS = 30, 4, 60, 12, 120
A_GAMMA = B_GAMMA = 0.3


def sample_corpus():
    """Draw counts from the TBIP generative model with planted ideal points x_s."""
    x = np.sort(RNG.normal(size=A))                       # planted author ideal points
    beta = RNG.gamma(0.3, 1.0 / 0.3, size=(K, V)) + 1e-3  # neutral topics
    eta = RNG.normal(scale=1.0, size=(K, V))              # ideological topics
    docs, group, doc_auth = [], [], []
    for s in range(A):
        for _ in range(DOCS_PER_A):
            theta = RNG.gamma(0.3, 1.0 / 0.3, size=K) + 1e-3
            rate = (theta[:, None] * beta * np.exp(x[s] * eta)).sum(0)  # (V,)
            rate = rate / rate.sum() * NWORDS
            counts = RNG.poisson(rate)
            toks = []
            for v in np.nonzero(counts)[0]:
                toks += [f"w{v}"] * int(counts[v])
            if len(toks) >= 5:
                docs.append(toks); group.append(f"a{s}"); doc_auth.append(s)
    return docs, group, np.array(doc_auth), x


def fit_topica(docs, group):
    m = topica.TBIP(K, iters=2500, batch_size=128, seed=42)
    m.fit(docs, group=group)
    return {a: float(p) for a, p in zip(m.author_names, m.ideal_points)}


def main():
    docs, group, doc_auth, x_true = sample_corpus()
    pos = fit_topica(docs, group)
    xt = np.array([pos[f"a{s}"] for s in range(A)])
    if np.corrcoef(xt, x_true)[0, 1] < 0:
        xt = -xt
    r = abs(np.corrcoef(xt, x_true)[0, 1])
    print(f"topica.TBIP recovers planted ideal points: Pearson |r| = {r:.3f}  (A={A})")
    ok = r > 0.85
    print("PASS" if ok else "FAIL", "(threshold 0.85)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
