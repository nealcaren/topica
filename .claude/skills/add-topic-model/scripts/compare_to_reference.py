#!/usr/bin/env python
"""Score a topica model's topics against a reference's, after topic alignment.

Topic models from different implementations cannot be compared element-wise: the
topic order is arbitrary and the RNG differs. This aligns the two topic-word
matrices by Hungarian assignment (maximizing cosine) and reports the per-topic and
mean aligned cosine, the top-word Jaccard, and -- if doc-topic matrices are given --
the doc-topic correlation after the same permutation.

Inputs are .npz files. Each must contain `topic_word` (K, V). Optionally
`doc_topic` (D, K) and `vocab` (V,) of word strings (for Jaccard). The two files
must share a vocabulary in the same column order.

Usage:
    python compare_to_reference.py --a topica.npz --b reference.npz [--topn 10]

Calibrate the bar with the reference's own seed-to-seed variation: run this with
two reference fits (different seeds) as --a and --b to get the noise floor, then
require the topica-vs-reference score to land at or above it.
"""
from __future__ import annotations

import argparse

import numpy as np


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Ka, Kb) cosine between every row of a and every row of b."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def align(a_tw: np.ndarray, b_tw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (perm, cosines): perm[i] is the b-topic matched to a-topic i, and
    cosines[i] their cosine. Hungarian assignment maximizing total cosine."""
    sim = _cosine_matrix(a_tw, b_tw)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-sim)
    except ImportError:
        # Greedy fallback if scipy is unavailable.
        cols = np.full(a_tw.shape[0], -1)
        used = set()
        for i in np.argsort(-sim.max(axis=1)):
            order = np.argsort(-sim[i])
            for j in order:
                if j not in used:
                    cols[i] = j
                    used.add(j)
                    break
        rows = np.arange(a_tw.shape[0])
    perm = np.asarray(cols)
    cosines = sim[np.asarray(rows), perm]
    return perm, cosines


def top_word_jaccard(a_tw, b_tw, perm, vocab, topn) -> float:
    scores = []
    for i, j in enumerate(perm):
        ta = set(np.argsort(-a_tw[i])[:topn])
        tb = set(np.argsort(-b_tw[j])[:topn])
        scores.append(len(ta & tb) / len(ta | tb))
    return float(np.mean(scores))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help=".npz with topic_word (the topica fit)")
    p.add_argument("--b", required=True, help=".npz with topic_word (the reference)")
    p.add_argument("--topn", type=int, default=10, help="top words per topic for Jaccard")
    args = p.parse_args()

    da, db = np.load(args.a, allow_pickle=True), np.load(args.b, allow_pickle=True)
    a_tw, b_tw = da["topic_word"], db["topic_word"]
    if a_tw.shape != b_tw.shape:
        print(f"warning: topic_word shapes differ {a_tw.shape} vs {b_tw.shape}; "
              "comparison assumes a shared vocabulary in matching column order.")

    perm, cosines = align(a_tw, b_tw)
    print(f"topics: {a_tw.shape[0]}   vocab: {a_tw.shape[1]}")
    print(f"mean aligned cosine: {cosines.mean():.4f}   "
          f"min: {cosines.min():.4f}   max: {cosines.max():.4f}")
    print("per-topic cosine:", np.array2string(np.round(cosines, 3), max_line_width=100))

    if "vocab" in da:
        jac = top_word_jaccard(a_tw, b_tw, perm, da["vocab"], args.topn)
        print(f"mean top-{args.topn} Jaccard: {jac:.4f}")

    if "doc_topic" in da and "doc_topic" in db:
        a_dt, b_dt = da["doc_topic"], db["doc_topic"]
        if a_dt.shape == b_dt.shape:
            b_perm = b_dt[:, perm]
            corr = np.corrcoef(a_dt.ravel(), b_perm.ravel())[0, 1]
            print(f"doc-topic correlation (aligned): {corr:.4f}")
        else:
            print(f"doc_topic shapes differ {a_dt.shape} vs {b_dt.shape}; skipped.")


if __name__ == "__main__":
    main()
