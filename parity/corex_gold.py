"""Committed `corextopic` gold for topica CorEx (#688).

CorEx (Gallagher, Reing, Kale & Ver Steeg, "Anchored Correlation Explanation," TACL
2017) is an information-theoretic topic model: it learns binary latent topics that
maximize the total correlation they explain about the words, with optional anchor
words for semi-supervision. It is NOT generative and NOT a factorization.

Reference: the `corextopic` package (Apache-2.0), `corextopic.corextopic.Corex`, in
its default tree/binarize mode. Apache-2.0 is permissive, so the update math is
transcribed and reimplemented in Rust.

We freeze, from a fixed-seed planted binary corpus with a known K-block structure:

  1. corextopic's mutual-information matrix (mis), per-topic TC (tcs), total
     correlation (tc), word->topic membership (alpha/clusters), and binary doc
     labels, under its DEFAULT init at two seeds -> a topic-word (MI) cosine
     self-consistency floor + the planted-recovery bar; and
  2. the same under an ANCHORED fit (one anchor word per block) -> the anchoring
     target (anchored words land in their assigned topics).

Parity is topic-aligned (CorEx is non-convex; topica's ChaCha RNG != numpy's, so
there is no bit-exact-init path): aligned MI cosine, total correlation, exact word
clusters (label-invariant Adjusted Rand Index), and anchored-word placement.

Runs in CI WITHOUT corextopic: the reference fit is frozen in the committed
``parity/corex_gold.npz`` + ``.json``.

    python parity/corex_gold.py --regenerate   # needs corextopic
    python parity/corex_gold.py                # offline compare against the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "corex"

N_TOPICS = 4
BLOCK = 8
N_BLOCKS = 4
V = N_BLOCKS * BLOCK
N_DOCS = 200
IN_BLOCK_P = 0.7
NOISE_P = 0.05
CORPUS_SEED = 271
ITERS = 200
EPS = 1e-5
ANCHOR_STRENGTH = 2.0
MARGIN = 0.15

# One anchor word per block (first word of each block), guiding topics 0..3.
ANCHORS: dict[str, list[str]] = {f"g{t}": [f"w{t * BLOCK}"] for t in range(N_BLOCKS)}
VOCAB = [f"w{j}" for j in range(V)]


def _planted_corpus(seed: int):
    """Binary planted corpus: doc d belongs to block d%K; its block words are present
    w.p. IN_BLOCK_P, every word present w.p. NOISE_P (noise). Returns (token_docs,
    X binary DxV)."""
    rng = np.random.RandomState(seed)
    X = np.zeros((N_DOCS, V), dtype=float)
    for d in range(N_DOCS):
        t = d % N_BLOCKS
        X[d, t * BLOCK:(t + 1) * BLOCK] = (rng.rand(BLOCK) < IN_BLOCK_P)
        X[d] = np.maximum(X[d], (rng.rand(V) < NOISE_P))
    docs = [[VOCAB[j] for j in range(V) if X[d, j] > 0] for d in range(N_DOCS)]
    return docs, X


def _fit_corex(X, *, seed, anchors=None, anchor_strength=ANCHOR_STRENGTH):
    import corextopic.corextopic as ct
    import scipy.sparse as ss

    m = ct.Corex(n_hidden=N_TOPICS, seed=seed, max_iter=ITERS, eps=EPS)
    Xs = ss.csr_matrix(X)
    if anchors is not None:
        anchor_cols = [[VOCAB.index(w) for w in ws] for ws in anchors.values()]
        m.fit(Xs, anchors=anchor_cols, anchor_strength=anchor_strength, words=VOCAB)
    else:
        m.fit(Xs, words=VOCAB)
    return m


def _corex_version() -> str:
    try:
        from importlib.metadata import version
        return f"corextopic {version('corextopic')}"
    except Exception:
        return "corextopic (version unknown)"


def regenerate() -> None:
    try:
        import corextopic  # noqa: F401
    except Exception:
        raise SystemExit("regenerate needs corextopic (pip install corextopic)")

    docs, X = _planted_corpus(CORPUS_SEED)

    m1 = _fit_corex(X, seed=13)
    m2 = _fit_corex(X, seed=99)
    self_cos, _ = harness.align_cosine(np.asarray(m1.mis), np.asarray(m2.mis))

    ma = _fit_corex(X, seed=13, anchors=ANCHORS)

    harness.save_gold(
        NAME,
        arrays={
            "mis": np.asarray(m1.mis),                 # K x V mutual information (bits)
            "tcs": np.asarray(m1.tcs),                 # per-topic TC
            "alpha": np.asarray(m1.alpha),             # K x V membership
            "labels": np.asarray(m1.labels, dtype=np.int8),   # D x K binary
            "p_y_given_x": np.asarray(m1.p_y_given_x),
            "clusters": np.asarray(m1.clusters),       # V word->topic argmax
            "mis_anchored": np.asarray(ma.mis),
            "clusters_anchored": np.asarray(ma.clusters),
            "X": X,
            "vocab": np.array(VOCAB, dtype=object),
        },
        meta={
            "reference": _corex_version(),
            "model": "CorEx (gregversteeg/corex_topic, tree/binarize)",
            "corpus": f"planted binary {N_DOCS}-doc / {N_BLOCKS}-block (seed {CORPUS_SEED})",
            "anchors": ANCHORS,
            "num_topics": N_TOPICS,
            "iters": ITERS,
            "eps": EPS,
            "anchor_strength": ANCHOR_STRENGTH,
            "count": "binarize",
            "vocab_size": V,
            "num_docs": N_DOCS,
            "margin": MARGIN,
            "total_correlation": float(m1.tc),
            "corex_self_mis_cosine": self_cos,
            "date": datetime.date.today().isoformat(),
            "pass_bar": "topica mis cosine >= corex_self - margin; anchored words in "
            "their assigned topics; tc within tolerance",
        },
    )
    print(f"regenerated {NAME} gold:")
    print(f"  corex self MI cosine (seed13 vs seed99): {self_cos:.4f}")
    print(f"  total correlation (tc): {float(m1.tc):.4f}  tcs {np.round(m1.tcs, 3)}")
    print(f"  clusters (word->topic): {np.asarray(m1.clusters)}")


def _align_to_vocab(mat_kv, topica_vocab):
    tv = {w: i for i, w in enumerate(topica_vocab)}
    out = np.zeros((mat_kv.shape[0], V))
    for j, w in enumerate(VOCAB):
        if w in tv:
            out[:, j] = mat_kv[:, tv[w]]
    return out


def _fit_topica(docs, *, anchors=None):
    import topica

    corpus = topica.Corpus.from_documents(docs, vocabulary=VOCAB)
    kwargs = {}
    if anchors is not None:
        kwargs = {"anchor_words": anchors, "anchor_strength": ANCHOR_STRENGTH}
    m = topica.CorEx(N_TOPICS, convergence_tol=EPS, seed=13, **kwargs).fit(corpus, iters=ITERS)
    # topic_word is alpha*mis; raw mis exposed separately for the MI-cosine parity.
    mis = _align_to_vocab(np.asarray(getattr(m, "mis", m.topic_word)), list(m.vocabulary))
    clusters = np.asarray(getattr(m, "clusters", np.argmax(mis, axis=0)))
    tcs = np.asarray(getattr(m, "topic_tc", np.full(N_TOPICS, np.nan)))
    tc = float(getattr(m, "total_correlation", np.nan))
    return mis, clusters, tcs, tc


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    self_cos = float(meta["corex_self_mis_cosine"])
    margin = float(meta["margin"])
    bar = self_cos - margin
    docs, _X = _planted_corpus(CORPUS_SEED)

    from collections import Counter

    # (1) unanchored: topic-word (MI) aligned cosine vs gold, clears the self floor.
    mis, clusters, tcs, tc = _fit_topica(docs)
    cos, perm = harness.align_cosine(arrays["mis"], mis)
    cos_ok = cos >= bar

    # (2) planted recovery: each block's words share one cluster (K distinct clusters).
    recovered = sum(
        1
        for b in range(N_BLOCKS)
        if Counter(clusters[b * BLOCK:(b + 1) * BLOCK]).most_common(1)[0][1] >= BLOCK // 2
    )
    recovery_ok = recovered == N_BLOCKS and len(set(clusters)) == N_TOPICS

    # (3) EXACT cluster match vs the gold word->topic partition (label-invariant
    #     Adjusted Rand Index == 1.0), not just "recovers the blocks".
    ari = harness.adjusted_rand_index(arrays["clusters"], clusters)
    clusters_exact_ok = ari > 0.999

    # (4) total correlation: topica tc within tolerance of the gold; per-topic tcs
    #     agree after aligning topics (sorted, since both recover the same blocks).
    gold_tc = float(meta["total_correlation"])
    tc_ok = np.isfinite(tc) and abs(tc - gold_tc) / max(gold_tc, 1e-9) < 0.10
    tcs_ok = np.allclose(np.sort(tcs), np.sort(arrays["tcs"]), rtol=0.15, atol=0.05) if np.all(
        np.isfinite(tcs)) else False

    # (5) anchored: anchor group t (insertion order) -> topic t; and topica's anchored
    #     clusters match the anchored GOLD partition (ARI) with aligned anchored MI.
    misA, clustersA, _tcsA, _tcA = _fit_topica(docs, anchors=ANCHORS)
    anchored_placed = all(
        clustersA[t * BLOCK] == t
        and Counter(clustersA[t * BLOCK:(t + 1) * BLOCK]).most_common(1)[0][0] == t
        for t in range(N_BLOCKS)
    )
    anchored_ari = harness.adjusted_rand_index(arrays["clusters_anchored"], clustersA)
    anchored_cos, _ = harness.align_cosine(arrays["mis_anchored"], misA)
    anchored_ok = anchored_placed and anchored_ari > 0.999 and anchored_cos >= bar

    result = {
        "mis_cosine": cos,
        "corex_self_mis_cosine": self_cos,
        "bar": bar,
        "clusters_ari": ari,
        "anchored_ari": anchored_ari,
        "anchored_mis_cosine": anchored_cos,
        "total_correlation": tc,
        "gold_tc": gold_tc,
        "tc_ok": bool(tc_ok),
        "tcs_ok": bool(tcs_ok),
        "passes": bool(
            cos_ok and recovery_ok and clusters_exact_ok and tc_ok and tcs_ok and anchored_ok
        ),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"(1) topic-word MI cosine = {cos:.4f}  (corex self {self_cos:.4f}, bar {bar:.4f}) "
              f"-> {'OK' if cos_ok else 'LOW'}")
        print(f"(2) planted recovery: {recovered}/{N_BLOCKS} blocks, "
              f"{len(set(clusters))} distinct clusters -> {'OK' if recovery_ok else 'FAIL'}")
        print(f"(3) exact clusters vs gold (ARI) = {ari:.4f} -> {'OK' if clusters_exact_ok else 'FAIL'}")
        print(f"(4) total correlation: topica {tc:.3f} vs gold {gold_tc:.3f} "
              f"-> {'OK' if tc_ok else 'OFF'}; per-topic tcs -> {'OK' if tcs_ok else 'OFF'}")
        print(f"(5) anchored: placed={anchored_placed} ARI={anchored_ari:.3f} "
              f"MIcos={anchored_cos:.4f} -> {'OK' if anchored_ok else 'FAIL'}")
        print(f"verdict: {'PASS' if result['passes'] else 'FAIL'}")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
