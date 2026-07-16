"""Committed-gold parity for topica LabeledLDA vs Java MALLET (issue #271, Wave 1).

MALLET's LabeledLDA and topica's LabeledLDA both constrain each document's topic
support to its observed label set, so topics correspond to labels and align by
name (no Hungarian needed). The RNGs differ, so agreement is *statistical*: given
the same labeled corpus, the per-label topic-word distributions should be nearly
identical.

The corpus is the synthetic multi-label fixture from the live parity script
``parity/mallet_parity.py`` (``labeled_parity``): 200 documents, each tagged with
one or two of three labels (sports / politics / tech), with words drawn from each
chosen label's six-word vocabulary. This is a well-identified design where both
engines recover the same per-label content.

The committed gold freezes MALLET's per-label topic-word matrix, the label order,
the shared word set, the exact labeled corpus (so the offline refit reproduces the
same documents), and MALLET's own seed-to-seed agreement floor, so the bar is
benchmarked against MALLET's own reproducibility rather than an invented
threshold.

Two phases (mirrors parity/stm_gold.py / dmr_gold.py exactly):

  * ``--regenerate`` (needs the ``mallet`` jars + javac to compile
    ``parity/LabeledLDADriver.java``): runs MALLET's LabeledLDA twice (two seeds)
    to measure its seed-to-seed floor, freezes one run's per-label topic-word
    matrix + corpus, and writes the committed gold
    (``parity/labeledlda_gold.npz`` + ``.json``).
  * default (no MALLET/Java): loads the committed gold, fits topica LabeledLDA on
    the same labeled corpus, aligns by label name, and checks the bar.

Run directly::

    python parity/labeledlda_gold.py               # offline compare vs committed gold
    python parity/labeledlda_gold.py --regenerate  # run MALLET twice, write the gold
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import harness
import mallet_parity

NAME = "labeledlda"

ITERS = 800
ALPHA = 0.1
BETA = 0.01
# topica clears MALLET's own seed-to-seed floor by a comfortable margin on this
# well-identified labeled corpus.
MARGIN = 0.05

_VOCAB = {
    "sports": "game team score player coach win".split(),
    "politics": "election vote senate policy congress law".split(),
    "tech": "computer software code data network chip".split(),
}


def _corpus(seed: int = 0):
    """Synthetic multi-label corpus (deterministic): (docs, labels)."""
    rng = np.random.default_rng(seed)
    docs, labels = [], []
    for _ in range(200):
        chosen = list(rng.choice(list(_VOCAB), size=rng.integers(1, 3), replace=False))
        words = []
        for lab in chosen:
            words += list(rng.choice(_VOCAB[lab], 8))
        docs.append(words)
        labels.append(chosen)
    return docs, labels


def _mallet_labeled(docs, labels, iters, seed):
    """Run the Java LabeledLDADriver; return (phi (K,W), mal_labels, words)."""
    if not mallet_parity._ensure_compiled("LabeledLDADriver"):
        raise RuntimeError("could not compile LabeledLDADriver")
    cp = mallet_parity._classpath()
    here = mallet_parity.HERE

    d = tempfile.mkdtemp()
    try:
        inp, out = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
        with open(inp, "w") as f:
            for toks, labs in zip(docs, labels):
                f.write(f"{','.join(labs)}\t{' '.join(toks)}\n")
        subprocess.run(
            ["java", "-cp", f"{cp}:{here}", "LabeledLDADriver", inp,
             str(iters), str(seed), str(ALPHA), str(BETA), out],
            check=True, capture_output=True, text=True,
        )
        lines = open(out).read().splitlines()
        mal_labels = lines[0].split(",")
        counts = {}
        for ln in lines[1:]:
            p = ln.split()
            if p:
                counts[p[0]] = {int(x.split(":")[0]): int(x.split(":")[1]) for x in p[1:]}
    finally:
        shutil.rmtree(d, ignore_errors=True)

    K = len(mal_labels)
    tpt = np.zeros(K)
    for cc in counts.values():
        for t, c in cc.items():
            tpt[t] += c
    words = sorted(counts)
    W = len(words)
    phi = np.zeros((K, W))
    for j, w in enumerate(words):
        for t, c in counts[w].items():
            phi[t, j] = (c + BETA) / (tpt[t] + BETA * W)
    return phi, mal_labels, words


def _topica_phi(docs, labels, mal_labels, words):
    """Fit topica LabeledLDA; return phi (K,W) row-ordered to ``mal_labels``."""
    from topica.models import LabeledLDA

    model = LabeledLDA(alpha=ALPHA, beta=BETA, seed=1)
    model.fit(docs, labels, iters=ITERS, num_samples=5, sample_interval=25)
    oi = {w: i for i, w in enumerate(model.vocabulary)}
    olabels = list(model.labels)
    phi = np.zeros((len(mal_labels), len(words)))
    for t_mal, lab in enumerate(mal_labels):
        t_our = olabels.index(lab)
        for j, w in enumerate(words):
            if w in oi:
                phi[t_mal, j] = np.asarray(model.topic_word)[t_our, oi[w]]
    return phi


def _label_cosines(a, b):
    """Per-row (per-label) cosine of two already label-aligned matrices."""
    cos = []
    for t in range(a.shape[0]):
        x, y = a[t], b[t]
        cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    return cos


def regenerate() -> None:
    if not mallet_parity.java_drivers_available():
        print("mallet jars / javac not available; cannot regenerate.")
        sys.exit(1)

    docs, labels = _corpus()

    mal_phi, mal_labels, words = _mallet_labeled(docs, labels, ITERS, seed=1)
    mal_phi2, mal_labels2, words2 = _mallet_labeled(docs, labels, ITERS, seed=2)

    # Re-order run 2 to run 1's label order + shared word set.
    wi = {w: j for j, w in enumerate(words)}
    l2 = {lab: t for t, lab in enumerate(mal_labels2)}
    mal_phi2_aligned = np.zeros_like(mal_phi)
    for t, lab in enumerate(mal_labels):
        for j2, w in enumerate(words2):
            if w in wi:
                mal_phi2_aligned[t, wi[w]] = mal_phi2[l2[lab], j2]
    mallet_self_cos = float(np.mean(_label_cosines(mal_phi, mal_phi2_aligned)))

    # topica fit summary captured at regenerate time for the provenance log.
    t_phi = _topica_phi(docs, labels, mal_labels, words)
    topica_cos = float(np.mean(_label_cosines(mal_phi, t_phi)))

    harness.save_gold(
        NAME,
        arrays={
            "mallet_phi": mal_phi.astype(np.float32),
            "labels": np.array(mal_labels, dtype=object),
            "words": np.array(words, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "doc_labels": np.array([",".join(ls) for ls in labels], dtype=object),
        },
        meta={
            "reference": f"Java MALLET {mallet_parity.mallet_home()} (LabeledLDADriver.java)",
            "model": "LabeledLDA (label-constrained LDA)",
            "corpus": ("synthetic multi-label fixture, 200 docs, 1-2 of three labels "
                       "(sports/politics/tech), six-word per-label vocab (from "
                       "parity/mallet_parity.py labeled_parity)"),
            "num_docs": len(docs),
            "vocab_size": len(words),
            "num_labels": len(mal_labels),
            "iters": ITERS,
            "alpha": ALPHA,
            "beta": BETA,
            "seeds": {"gold": 1, "noise_floor": 2},
            "margin": MARGIN,
            "mallet_self_cosine": mallet_self_cos,
            "topica_cosine": topica_cos,
            "date": datetime.date.today().isoformat(),
            "pass_bar": ("mean per-label topic-word cosine vs MALLET >= MALLET's "
                         "own seed-to-seed floor - margin"),
            "kind": "cross-implementation (Java MALLET LabeledLDA)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  MALLET self cosine  : {mallet_self_cos:.4f}")
    print(f"  topica vs MALLET cos: {topica_cos:.4f}")


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    mal_phi = arrays["mallet_phi"].astype(np.float64)
    mal_labels = list(arrays["labels"])
    words = list(arrays["words"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    labels = [s.split(",") for s in arrays["doc_labels"]]
    margin = float(meta.get("margin", MARGIN))
    self_cos = float(meta.get("mallet_self_cosine", 0.0))

    t_phi = _topica_phi(docs, labels, mal_labels, words)
    cos = float(np.mean(_label_cosines(mal_phi, t_phi)))

    cos_bar = self_cos - margin
    result = {
        "cosine": cos,
        "cosine_bar": cos_bar,
        "cosine_margin": cos - cos_bar,
        "mallet_self_cosine": self_cos,
        "passes": bool(cos >= cos_bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica cosine vs MALLET : {cos:.4f} "
              f"(MALLET self {self_cos:.4f}, bar {cos_bar:.4f})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(cos margin {result['cosine_margin']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
