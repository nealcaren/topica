"""Cross-implementation check: topica's c_v coherence vs gensim's CoherenceModel.

c_v is the Röder et al. (2015) coherence gensim popularized. topica computes it in
its own pipeline (co-occurrence counting in Rust, the NPMI/cosine/c_v scoring in
Python), so this pins how closely the two agree on the SAME top-word lists and the
SAME reference corpus.

What we find, and why this is a *ranking* guarantee rather than a digit-for-digit
one (design-review #04.1):

  - On corpora whose documents are mostly LONGER than the c_v sliding window (110
    tokens) -- e.g. poliblog, median ~160 tokens -- topica and gensim agree almost
    exactly: Spearman rho > 0.99 and a per-topic offset around 0.002.
  - On SHORT-document corpora (most documents shorter than the window) -- e.g.
    gadarian survey responses, median ~14 tokens -- the two diverge modestly:
    topica scores a few hundredths higher (offset ~0.04) and the ranking loosens
    (rho ~0.9). This is the window-construction difference the review flagged:
    topica emits one whole-document boolean window when a document is shorter than
    the window, where gensim slides one token at a time and yields a window at
    every position. Neither is canonical -- c_v is implementation-sensitive even
    across gensim versions -- so we do not bend topica's windows to match.

The guarantee we therefore assert and commit: c_v is used to RANK topics/models
within a corpus, and topica's ranking tracks gensim's (Spearman rho high in both
regimes). The absolute offset is small and documented, and grows for documents
shorter than the window. Absolute c_v is not comparable across implementations;
within-corpus ranking is.

Skips (exit 0) if gensim is unavailable. Run directly:

    python parity/coherence_gensim_compare.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

import topica

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def gensim_available() -> bool:
    try:
        import gensim.corpora  # noqa: F401
        import gensim.models  # noqa: F401
        import scipy.stats  # noqa: F401
        return True
    except Exception:
        return False


def _load_poliblog():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "examples", "poliblog.csv"))))
    return [r["text"].split() for r in rows]


def _load_gadarian():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "examples", "gadarian.csv"))))
    return [topica.tokenize(r["open.ended.response"], min_length=3) for r in rows]


def _compare(docs: list, ks=(10, 20)) -> dict:
    """Pool per-topic c_v across a few K and compare topica to gensim."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from scipy.stats import spearmanr

    dct = Dictionary(docs)
    topica_cv, gensim_cv = [], []
    for k in ks:
        m = topica.LDA(num_topics=k, seed=1)
        m.fit(docs, iters=300)
        top_words = [[w for w in m.top_words(10, topic=t)] for t in range(k)]
        topica_cv.extend(np.asarray(topica.coherence(m, docs, coherence_type="c_v")))
        cm = CoherenceModel(topics=top_words, texts=docs, dictionary=dct,
                            coherence="c_v", processes=1)
        gensim_cv.extend(np.asarray(cm.get_coherence_per_topic()))

    a, b = np.array(topica_cv), np.array(gensim_cv)
    lens = np.array([len(d) for d in docs])
    return {
        "n_topics": len(a),
        "rho": float(spearmanr(a, b).statistic),
        "offset_mean": float((a - b).mean()),
        "offset_sd": float((a - b).std()),
        "topica_mean": float(a.mean()),
        "gensim_mean": float(b.mean()),
        "median_doc_len": float(np.median(lens)),
        "frac_below_window": float((lens < 110).mean()),
    }


def run(verbose: bool = True) -> dict:
    if not gensim_available():
        raise RuntimeError("gensim (and scipy) not available")
    out = {}
    for name, loader in (("poliblog (long docs)", _load_poliblog),
                         ("gadarian (short docs)", _load_gadarian)):
        m = _compare(loader())
        out[name] = m
        if verbose:
            print(f"{name}: median_len={m['median_doc_len']:.0f} "
                  f"({m['frac_below_window']:.0%} < window)  "
                  f"n={m['n_topics']}  Spearman rho={m['rho']:.3f}  "
                  f"offset(topica-gensim)={m['offset_mean']:+.4f} (sd {m['offset_sd']:.4f})  "
                  f"means topica={m['topica_mean']:.3f}/gensim={m['gensim_mean']:.3f}")
    return out


def main() -> int:
    if not gensim_available():
        print("skipping c_v gensim parity: gensim/scipy not available")
        return 0
    out = run()
    poli = out["poliblog (long docs)"]
    gad = out["gadarian (short docs)"]
    # Ranking guarantee in both regimes (the only thing c_v is used for); the
    # absolute offset is small and documented, larger for short documents.
    assert poli["rho"] > 0.95, (
        f"long-doc c_v ranking diverges from gensim: rho={poli['rho']:.3f}"
    )
    assert gad["rho"] > 0.85, (
        f"short-doc c_v ranking diverges from gensim: rho={gad['rho']:.3f}"
    )
    # Long docs should agree closely in absolute terms; short docs may not.
    assert abs(poli["offset_mean"]) < 0.02, (
        f"long-doc c_v offset unexpectedly large: {poli['offset_mean']:+.4f}"
    )
    print("OK: topica c_v ranks topics as gensim does (offset small and documented; "
          "larger for documents shorter than the window).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
