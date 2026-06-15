"""Cross-implementation check for topica's LSA/LSI against scikit-learn's
``sklearn.decomposition.TruncatedSVD`` (BSD-3-Clause, ``algorithm='randomized'``).

Both take a truncated SVD of the SAME document-term matrix ``X (D x V)``. We build
``X`` identically on both sides (raw token counts, ``weighting="count"``, in the
column order topica assigns), so the only differences are the randomized-SVD draw
and arithmetic order. A truncated SVD is unique up to a per-component sign when the
singular values are distinct, so after applying the ``svd_flip`` convention on both
sides this is a MATCH-THE-SOLUTION bar, not a noise floor: per-component cosine of
the right singular vectors should be ~1.0, the singular values should agree, and
the document coordinates should correlate ~1.0.

We also calibrate against sklearn's own ``random_state`` perturbation (the
randomized SVD's seed-to-seed spread) -- on a well-posed truncated SVD this is
tiny, so the bar stays near-exact.

A small golden fixture (``parity/lsa_gold.npz``) lets the check run without
re-fitting sklearn. Regenerate it with ``--regenerate`` (requires sklearn).

Skips cleanly when sklearn / scipy are unavailable. Run directly:

    python parity/lsa_vs_sklearn.py
    python parity/lsa_vs_sklearn.py --regenerate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GOLD = Path(__file__).resolve().parent / "lsa_gold.npz"

# Planted-block corpus settings, fixed so both sides see the same task.
K = 5
BLOCK = 6
NDOCS = 400
DLEN = 30
CORPUS_SEED = 12345
WEIGHTING = "count"        # matched on both sides; raw counts make X identical
TOPICA_SEED = 42
SKLEARN_SEEDS = (0, 1, 2, 3, 4)
# A little cross-block bleed so the spectrum is non-degenerate (distinct singular
# values), which is what makes the per-component sign/solution well-defined.
CONTAMINATION = 0.35


def available() -> bool:
    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def planted_corpus():
    rng = np.random.default_rng(CORPUS_SEED)
    docs = []
    for d in range(NDOCS):
        b = d % K
        toks = []
        for _ in range(DLEN):
            if rng.random() < CONTAMINATION:
                other = (b + 1 + int(rng.integers(K - 1))) % K
                toks.append(f"b{other}w{int(rng.integers(BLOCK))}")
            else:
                toks.append(f"b{b}w{int(rng.integers(BLOCK))}")
        docs.append(toks)
    return docs


def count_matrix(docs, vocab):
    """Raw token-count matrix (D x V) in the given vocabulary column order."""
    index = {w: i for i, w in enumerate(vocab)}
    x = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for d, toks in enumerate(docs):
        for t in toks:
            j = index.get(t)
            if j is not None:
                x[d, j] += 1.0
    return x


def svd_flip(u, vt):
    """The scikit-learn svd_flip convention (u, v based on v's rows): for each
    component, make the largest-|value| entry of the right singular vector
    positive, flipping the matching u column with it. u is (D, K), vt is (K, V)."""
    max_abs = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), max_abs])
    signs[signs == 0] = 1.0
    return u * signs[np.newaxis, :], vt * signs[:, np.newaxis]


def topica_fit(docs):
    import topica

    m = topica.LSA(K, weighting=WEIGHTING, seed=TOPICA_SEED)
    m.fit(docs)
    vocab = list(m.vocabulary)
    tw = np.asarray(m.topic_word)        # (K, V) signed right singular vectors
    dt = np.asarray(m.doc_topic)         # (D, K) = U Sigma
    sv = np.asarray(m.singular_values)   # (K,)
    return tw, dt, sv, vocab


def _sklearn_fit(x, seed):
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=K, algorithm="randomized", random_state=seed)
    dt = svd.fit_transform(x)            # (D, K) = U Sigma
    tw = svd.components_                 # (K, V) = Vt
    sv = svd.singular_values_            # (K,)
    # sklearn applies svd_flip internally, but we re-apply our explicit convention
    # on both sides so the comparison is convention-matched regardless.
    u = dt / np.where(sv == 0, 1.0, sv)[np.newaxis, :]
    u, tw = svd_flip(u, tw)
    dt = u * sv[np.newaxis, :]
    return tw, dt, sv


def _cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sklearn_noise(x):
    """sklearn random_state seed-to-seed spread of the per-component right singular
    vectors (after svd_flip). On a well-posed truncated SVD this is near zero."""
    base_tw, _, _ = _sklearn_fit(x, SKLEARN_SEEDS[0])
    means = []
    for s in SKLEARN_SEEDS[1:]:
        tw, _, _ = _sklearn_fit(x, s)
        cos = np.array([_cosine(base_tw[i], tw[i]) for i in range(K)])
        means.append(float(cos.mean()))
    return float(np.mean(means)), float(np.std(means))


def regenerate():
    if not available():
        print("sklearn / scipy not installed; cannot regenerate.")
        sys.exit(0)
    docs = planted_corpus()
    _, _, _, vocab = topica_fit(docs)
    x = count_matrix(docs, vocab)
    sk_tw, sk_dt, sk_sv = _sklearn_fit(x, SKLEARN_SEEDS[0])
    nf_mean, nf_std = sklearn_noise(x)
    np.savez(
        GOLD,
        sklearn_topic_word=sk_tw,
        sklearn_doc_topic=sk_dt,
        sklearn_singular_values=sk_sv,
        vocab=np.array(vocab, dtype=object),
        noise_floor_mean=np.array(nf_mean),
        noise_floor_std=np.array(nf_std),
        settings=np.array([K, BLOCK, NDOCS, DLEN, CORPUS_SEED], dtype=np.int64),
    )
    print(f"wrote {GOLD.name}: noise floor cos {nf_mean:.6f} +/- {nf_std:.6f}")


def run(verbose: bool = True) -> dict:
    docs = planted_corpus()
    t_tw, t_dt, t_sv, vocab = topica_fit(docs)

    if GOLD.exists():
        g = np.load(GOLD, allow_pickle=True)
        gold_vocab = list(g["vocab"])
        if gold_vocab != vocab:
            order = [vocab.index(w) for w in gold_vocab]
            t_tw = t_tw[:, order]
        sk_tw = g["sklearn_topic_word"]
        sk_dt = g["sklearn_doc_topic"]
        sk_sv = g["sklearn_singular_values"]
        nf_mean = float(g["noise_floor_mean"])
        nf_std = float(g["noise_floor_std"])
    elif available():
        x = count_matrix(docs, vocab)
        sk_tw, sk_dt, sk_sv = _sklearn_fit(x, SKLEARN_SEEDS[0])
        nf_mean, nf_std = sklearn_noise(x)
    else:
        print("sklearn / scipy not installed and no golden fixture; skipping.")
        sys.exit(0)

    # SVD components are unique up to sign and already sign-fixed on both sides; no
    # Hungarian alignment needed -- compare component i to component i directly.
    cos = np.array([_cosine(sk_tw[i], t_tw[i]) for i in range(K)])
    sv_rel = np.abs(t_sv - sk_sv) / np.where(sk_sv == 0, 1.0, np.abs(sk_sv))
    dt_corr = np.array([
        float(np.corrcoef(sk_dt[:, i], t_dt[:, i])[0, 1]) for i in range(K)
    ])

    metrics = {
        "num_docs": len(docs),
        "vocab": len(vocab),
        "weighting": WEIGHTING,
        "per_component_cosine": cos.tolist(),
        "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()),
        "singular_value_rel_err": sv_rel.tolist(),
        "max_singular_value_rel_err": float(sv_rel.max()),
        "doc_coord_corr": dt_corr.tolist(),
        "mean_doc_coord_corr": float(np.nanmean(dt_corr)),
        "noise_floor_mean": nf_mean,
        "noise_floor_std": nf_std,
    }
    # Match-the-solution: mean per-component cosine should be ~1.0 (well inside the
    # sklearn random_state spread) and singular values should agree to ~1e-6.
    matched = (
        metrics["min_cosine"] >= 0.999
        and metrics["max_singular_value_rel_err"] <= 1e-4
        and metrics["mean_doc_coord_corr"] >= 0.999
    )
    metrics["matched"] = bool(matched)

    if verbose:
        print(f"  corpus: {metrics['num_docs']} docs, {metrics['vocab']} vocab, "
              f"weighting={WEIGHTING}")
        print(f"  per-component cosine     : {np.round(cos, 6).tolist()}")
        print(f"  mean / min cosine        : {metrics['mean_cosine']:.6f} / "
              f"{metrics['min_cosine']:.6f}")
        print(f"  max singular-value rel err: {metrics['max_singular_value_rel_err']:.2e}")
        print(f"  mean doc-coord corr      : {metrics['mean_doc_coord_corr']:.6f}")
        print(f"  sklearn random_state noise: {nf_mean:.6f} +/- {nf_std:.6f}")
        print(f"  verdict                  : {'matched' if matched else 'MISMATCH'}")
    return metrics


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    elif not available() and not GOLD.exists():
        print("sklearn / scipy not installed and no golden fixture; skipping.")
    else:
        run(verbose=True)
