"""Committed-gold parity for topica LDA vs Java MALLET (issue #271, Wave 1).

Java MALLET and topica are independent collapsed-Gibbs LDA implementations with
different RNGs, so they are never byte-identical: agreement is *statistical*. On a
corpus with planted (disjoint-vocabulary) topics both should recover the same
topics, and in practice the alignment is near-exact.

The corpus is the planted five-topic fixture from the live parity script
``parity/mallet_parity.py`` (``planted_corpus``): 250 short documents, each drawn
from one of five disjoint six-word vocabularies. This is a well-identified design
where both engines land on the same five topics, so the absolute agreement is
high — not just above MALLET's own seed-to-seed floor.

The committed gold freezes MALLET's topic-word matrix, the vocab, the exact
tokenized corpus (so the offline refit reproduces the same documents), and
MALLET's own seed-to-seed agreement floor (jaccard + cosine), so the bar is
benchmarked against MALLET's own reproducibility rather than an invented
threshold.

Two phases (mirrors parity/stm_gold.py / gdmr_gold.py exactly):

  * ``--regenerate`` (needs the ``mallet`` CLI): runs MALLET's SparseLDA twice
    (two seeds) to measure its seed-to-seed agreement floor, freezes one run's
    topic-word matrix + vocab + corpus, and writes the committed gold
    (``parity/lda_gold.npz`` + ``.json``).
  * default (no MALLET): loads the committed gold, fits topica LDA on the same
    corpus, aligns to MALLET's topic-word matrix, and checks the bar.

Run directly::

    python parity/lda_gold.py               # offline compare against committed gold
    python parity/lda_gold.py --regenerate  # run MALLET twice, write the gold
"""

from __future__ import annotations

import datetime
import sys

import numpy as np

import harness
import mallet_parity

NAME = "lda"

ITERS = 800
TOP_N = 6
# topica clears MALLET's own seed-to-seed floor by a comfortable margin on this
# well-identified planted corpus; the slack absorbs RNG basin differences.
MARGIN = 0.10


def _corpus():
    """The planted five-topic corpus from the live script (deterministic)."""
    return mallet_parity.planted_corpus(seed=0)


def _topica_phi(docs, k, vocab):
    """Fit topica LDA and return its topic-word matrix aligned to ``vocab``."""
    from topica import LDA

    model = LDA(num_topics=k, seed=1, optimize_interval=0)
    model.fit(docs, iters=ITERS, num_samples=5, sample_interval=25)
    oi = {w: i for i, w in enumerate(model.vocabulary)}
    phi = np.zeros((k, len(vocab)))
    for j, w in enumerate(vocab):
        if w in oi:
            phi[:, j] = np.asarray(model.topic_word)[:, oi[w]]
    return phi


def regenerate() -> None:
    if not mallet_parity.mallet_available():
        print("mallet CLI not available; cannot regenerate.")
        sys.exit(1)

    docs, k = _corpus()

    # Two MALLET seeds: one frozen as gold, the pair for the seed-to-seed floor.
    mal_phi, vocab = mallet_parity._mallet_phi(docs, k, ITERS, seed=1)
    mal_phi2, vocab2 = mallet_parity._mallet_phi(docs, k, ITERS, seed=2)

    # Align the second run onto the gold run over the shared vocab.
    vi = {w: j for j, w in enumerate(vocab)}
    mal_phi2_aligned = np.zeros_like(mal_phi)
    for j, w in enumerate(vocab2):
        if w in vi:
            mal_phi2_aligned[:, vi[w]] = mal_phi2[:, j]
    mallet_self_cos, _ = harness.align_cosine(mal_phi, mal_phi2_aligned)
    mallet_self_jacc = harness.top_word_jaccard(mal_phi, mal_phi2_aligned, n=TOP_N)

    # topica fit summary captured at regenerate time for the provenance log.
    t_phi = _topica_phi(docs, k, vocab)
    topica_cos, _ = harness.align_cosine(mal_phi, t_phi)
    topica_jacc = harness.top_word_jaccard(mal_phi, t_phi, n=TOP_N)

    harness.save_gold(
        NAME,
        arrays={
            "mallet_phi": mal_phi.astype(np.float32),
            "vocab": np.array(vocab, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "k": np.array(k),
        },
        meta={
            "reference": f"Java MALLET {mallet_parity.mallet_home()}",
            "model": "LDA (collapsed Gibbs / SparseLDA)",
            "corpus": ("planted five-topic fixture, 250 docs x 12 tokens, disjoint "
                       "six-word vocabularies (from parity/mallet_parity.py "
                       "planted_corpus)"),
            "num_docs": len(docs),
            "vocab_size": len(vocab),
            "K": int(k),
            "iters": ITERS,
            "top_n": TOP_N,
            "seeds": {"gold": 1, "noise_floor": 2},
            "margin": MARGIN,
            "mallet_self_cosine": mallet_self_cos,
            "mallet_self_jaccard": mallet_self_jacc,
            "topica_cosine": topica_cos,
            "topica_jaccard": topica_jacc,
            "date": datetime.date.today().isoformat(),
            "pass_bar": ("aligned topic-word cosine AND top-word jaccard vs MALLET "
                         ">= MALLET's own seed-to-seed floor - margin"),
            "kind": "cross-implementation (Java MALLET SparseLDA)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  MALLET self cosine  : {mallet_self_cos:.4f}   jaccard: {mallet_self_jacc:.4f}")
    print(f"  topica vs MALLET cos: {topica_cos:.4f}   jaccard: {topica_jacc:.4f}")


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    mal_phi = arrays["mallet_phi"].astype(np.float64)
    vocab = list(arrays["vocab"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    k = int(arrays["k"])
    margin = float(meta.get("margin", MARGIN))
    self_cos = float(meta.get("mallet_self_cosine", 0.0))
    self_jacc = float(meta.get("mallet_self_jaccard", 0.0))

    t_phi = _topica_phi(docs, k, vocab)
    cos, _ = harness.align_cosine(mal_phi, t_phi)
    jacc = harness.top_word_jaccard(mal_phi, t_phi, n=TOP_N)

    cos_bar = self_cos - margin
    jacc_bar = self_jacc - margin
    result = {
        "cosine": cos,
        "jaccard": jacc,
        "mallet_self_cosine": self_cos,
        "mallet_self_jaccard": self_jacc,
        "cosine_bar": cos_bar,
        "jaccard_bar": jacc_bar,
        "cosine_margin": cos - cos_bar,
        "jaccard_margin": jacc - jacc_bar,
        "passes": bool(cos >= cos_bar and jacc >= jacc_bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica cosine vs MALLET  : {cos:.4f} "
              f"(MALLET self {self_cos:.4f}, bar {cos_bar:.4f})")
        print(f"  topica jaccard vs MALLET : {jacc:.4f} "
              f"(MALLET self {self_jacc:.4f}, bar {jacc_bar:.4f})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(cos margin {result['cosine_margin']:+.4f}, "
              f"jacc margin {result['jaccard_margin']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
