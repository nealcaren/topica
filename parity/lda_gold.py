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
# Optimize-on case (#713-#4): hyperparameter optimization on in both engines and a
# terminal topica estimator, so the Wallach alpha/beta optimizer is actually
# exercised rather than the optimize-off, snapshot-averaged path above.
ITERS_OPT = 1000
OPT_INTERVAL = 50
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


def _topica_phi_optimized(docs, k, vocab):
    """Fit topica LDA with hyperparameter optimization ON and a terminal estimator
    (num_samples=1), aligned to ``vocab``; also return (alpha_sum, beta)."""
    from topica import LDA

    model = LDA(num_topics=k, seed=1, optimize_interval=OPT_INTERVAL, burn_in=200)
    model.fit(docs, iters=ITERS_OPT, num_samples=1)
    oi = {w: i for i, w in enumerate(model.vocabulary)}
    phi = np.zeros((k, len(vocab)))
    for j, w in enumerate(vocab):
        if w in oi:
            phi[:, j] = np.asarray(model.topic_word)[:, oi[w]]
    return phi, float(np.sum(model.alpha)), float(model.beta)


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

    # Optimize-on gold (#713-#4): MALLET with --optimize-interval 50, two seeds for
    # its own optimize-on floor, plus topica's terminal optimize-on fit.
    mal_phi_opt, vocab_opt = mallet_parity._mallet_phi(
        docs, k, ITERS_OPT, seed=1, optimize_interval=OPT_INTERVAL)
    mal_phi_opt2, vocab_opt2 = mallet_parity._mallet_phi(
        docs, k, ITERS_OPT, seed=2, optimize_interval=OPT_INTERVAL)
    vi_opt = {w: j for j, w in enumerate(vocab_opt)}
    mal_phi_opt2_aligned = np.zeros_like(mal_phi_opt)
    for j, w in enumerate(vocab_opt2):
        if w in vi_opt:
            mal_phi_opt2_aligned[:, vi_opt[w]] = mal_phi_opt2[:, j]
    mallet_opt_self_cos, _ = harness.align_cosine(mal_phi_opt, mal_phi_opt2_aligned)
    mallet_opt_self_jacc = harness.top_word_jaccard(mal_phi_opt, mal_phi_opt2_aligned, n=TOP_N)
    t_phi_opt, t_alpha_sum, t_beta = _topica_phi_optimized(docs, k, vocab_opt)
    topica_opt_cos, _ = harness.align_cosine(mal_phi_opt, t_phi_opt)
    topica_opt_jacc = harness.top_word_jaccard(mal_phi_opt, t_phi_opt, n=TOP_N)

    harness.save_gold(
        NAME,
        arrays={
            "mallet_phi": mal_phi.astype(np.float32),
            "vocab": np.array(vocab, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "k": np.array(k),
            "mallet_phi_opt": mal_phi_opt.astype(np.float32),
            "vocab_opt": np.array(vocab_opt, dtype=object),
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
            "optimize_interval": OPT_INTERVAL,
            "iters_opt": ITERS_OPT,
            "mallet_opt_self_cosine": mallet_opt_self_cos,
            "mallet_opt_self_jaccard": mallet_opt_self_jacc,
            "topica_opt_cosine": topica_opt_cos,
            "topica_opt_jaccard": topica_opt_jacc,
            "topica_opt_alpha_sum": t_alpha_sum,
            "topica_opt_beta": t_beta,
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
    print(f"  [optimize on] MALLET self cos: {mallet_opt_self_cos:.4f} "
          f"jacc: {mallet_opt_self_jacc:.4f}")
    print(f"  [optimize on] topica vs MALLET cos: {topica_opt_cos:.4f} "
          f"jacc: {topica_opt_jacc:.4f}  (alpha_sum {t_alpha_sum:.4f}, beta {t_beta:.4f})")


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

    # Optimize-on gold (#713-#4), present in golds regenerated after the Wallach
    # optimizer landed. Exercises topica's terminal optimize-on fit against MALLET
    # with --optimize-interval 50.
    if "mallet_phi_opt" in arrays:
        mal_phi_opt = arrays["mallet_phi_opt"].astype(np.float64)
        vocab_opt = list(arrays["vocab_opt"])
        opt_self_cos = float(meta.get("mallet_opt_self_cosine", 0.0))
        opt_self_jacc = float(meta.get("mallet_opt_self_jaccard", 0.0))
        t_phi_opt, t_alpha_sum, t_beta = _topica_phi_optimized(docs, k, vocab_opt)
        opt_cos, _ = harness.align_cosine(mal_phi_opt, t_phi_opt)
        opt_jacc = harness.top_word_jaccard(mal_phi_opt, t_phi_opt, n=TOP_N)
        opt_cos_bar = opt_self_cos - margin
        opt_jacc_bar = opt_self_jacc - margin
        opt_passes = bool(opt_cos >= opt_cos_bar and opt_jacc >= opt_jacc_bar)
        result.update({
            "opt_cosine": opt_cos,
            "opt_jaccard": opt_jacc,
            "opt_cosine_bar": opt_cos_bar,
            "opt_jaccard_bar": opt_jacc_bar,
            "opt_alpha_sum": t_alpha_sum,
            "opt_beta": t_beta,
            "opt_passes": opt_passes,
        })
        result["passes"] = result["passes"] and opt_passes

    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica cosine vs MALLET  : {cos:.4f} "
              f"(MALLET self {self_cos:.4f}, bar {cos_bar:.4f})")
        print(f"  topica jaccard vs MALLET : {jacc:.4f} "
              f"(MALLET self {self_jacc:.4f}, bar {jacc_bar:.4f})")
        if "opt_cosine" in result:
            print(f"  [optimize on] topica cosine : {result['opt_cosine']:.4f} "
                  f"(bar {result['opt_cosine_bar']:.4f}), jaccard "
                  f"{result['opt_jaccard']:.4f} (bar {result['opt_jaccard_bar']:.4f}); "
                  f"alpha_sum {result['opt_alpha_sum']:.4f}, beta {result['opt_beta']:.4f}")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(cos margin {result['cosine_margin']:+.4f}, "
              f"jacc margin {result['jaccard_margin']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
