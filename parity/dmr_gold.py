"""Committed-gold parity for topica DMR vs Java MALLET (issue #271, Wave 1).

MALLET's ``DMRTopicModel`` and topica's DMR both put a log-linear (Dirichlet-
Multinomial Regression) prior on the document-topic distribution, fitting the
feature weights by L-BFGS. The optimizers and RNGs differ between
implementations, so agreement is *statistical*: on a corpus whose single binary
covariate drives the topic mixture, both should (a) recover the same two topics
and (b) agree on the SIGN and substantial MAGNITUDE of the covariate's effect.

The corpus is the synthetic two-cluster fixture from the live parity script
``parity/mallet_parity.py`` (``dmr_parity``): 160 short documents whose binary
``is_space`` covariate switches the generating vocabulary (animal vs space
words). This is a well-identified design where the space topic's covariate weight
must be strongly positive, so both engines agree sharply.

The committed gold freezes MALLET's topic-word matrix, its per-topic feature
weights (intercept + is_space), the vocab, the exact tokenized corpus + covariate
(so the offline refit reproduces the same documents), and MALLET's own
seed-to-seed agreement floor, so the bar is benchmarked against MALLET's own
reproducibility rather than an invented threshold.

Two phases (mirrors parity/stm_gold.py / lda_gold.py exactly):

  * ``--regenerate`` (needs the ``mallet`` jars + javac to compile
    ``parity/DMRDriver.java``): runs MALLET's DMR twice (two seeds) to measure its
    seed-to-seed floor, freezes one run's topic-word matrix + feature weights +
    corpus, and writes the committed gold (``parity/dmr_gold.npz`` + ``.json``).
  * default (no MALLET/Java): loads the committed gold, fits topica DMR on the same
    corpus + covariate, aligns to MALLET, and checks the bar.

Run directly::

    python parity/dmr_gold.py               # offline compare against committed gold
    python parity/dmr_gold.py --regenerate  # run MALLET twice, write the gold
"""

from __future__ import annotations

import collections
import datetime
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import harness
import mallet_parity

NAME = "dmr"

K = 2
ITERS = 800
NUM_DOCS = 160
# topica clears MALLET's own seed-to-seed topic floor by a comfortable margin on
# this well-identified corpus.
MARGIN = 0.10

_ANIMAL = "cat dog fish bird horse cow".split()
_SPACE = "planet star moon rocket comet galaxy".split()


def _corpus(seed: int = 0):
    """Synthetic two-cluster corpus + binary covariate (deterministic)."""
    rng = np.random.default_rng(seed)
    docs, cov = [], []
    for _ in range(NUM_DOCS):
        x = int(rng.integers(0, 2))
        docs.append(list(rng.choice(_SPACE if x else _ANIMAL, 8)))
        cov.append(x)
    return docs, cov


def _mallet_dmr(docs, cov, iters, seed):
    """Run the Java DMRDriver; return (phi (K,W), lambda (K,2 icpt+is_space), words)."""
    if not mallet_parity._ensure_compiled("DMRDriver"):
        raise RuntimeError("could not compile DMRDriver")
    cp = mallet_parity._classpath()
    here = mallet_parity.HERE

    d = tempfile.mkdtemp()
    try:
        inp = os.path.join(d, "in.txt")
        oc, op = os.path.join(d, "c.txt"), os.path.join(d, "p.txt")
        with open(inp, "w") as f:
            for toks, x in zip(docs, cov):
                f.write(f"is_space={x}\t{' '.join(toks)}\n")
        subprocess.run(
            ["java", "-cp", f"{cp}:{here}", "DMRDriver", inp, str(K), str(iters),
             str(seed), oc, op],
            check=True, capture_output=True, text=True,
        )
        tw = collections.defaultdict(dict)
        wset = set()
        for ln in open(oc):
            p = ln.split("\t")
            if len(p) >= 3:
                tw[int(p[0])][p[1]] = float(p[2])
                wset.add(p[1])
        plines = open(op).read().splitlines()
        lam = np.array([[float(x) for x in ln.split()] for ln in plines[1:]])
    finally:
        shutil.rmtree(d, ignore_errors=True)

    words = sorted(wset)
    W = len(words)
    phi = np.zeros((K, W))
    for t in range(K):
        for j, w in enumerate(words):
            phi[t, j] = tw[t].get(w, 0.0)
        s = phi[t].sum()
        if s:
            phi[t] /= s
    return phi, lam, words


def _space_topic(phi, words):
    sc = [sum(phi[t, words.index(w)] for w in _SPACE if w in words) for t in range(K)]
    return int(np.argmax(sc))


def _topica_fit(docs, cov, words):
    """Fit topica DMR; return (phi (K,W) aligned to words, feature_effects)."""
    from topica.models import DMR

    model = DMR(num_topics=K, seed=1, optimize_interval=25, burn_in=50)
    model.fit(docs, np.array(cov, float)[:, None], feature_names=["is_space"],
              iters=ITERS, num_samples=5, sample_interval=25)
    oi = {w: i for i, w in enumerate(model.vocabulary)}
    phi = np.zeros((K, len(words)))
    for j, w in enumerate(words):
        if w in oi:
            phi[:, j] = np.asarray(model.topic_word)[:, oi[w]]
    s = phi.sum(axis=1, keepdims=True)
    phi = phi / np.where(s == 0, 1, s)
    return phi, np.asarray(model.feature_effects)


def regenerate() -> None:
    if not mallet_parity.java_drivers_available():
        print("mallet jars / javac not available; cannot regenerate.")
        sys.exit(1)

    docs, cov = _corpus()

    mal_phi, mal_lam, words = _mallet_dmr(docs, cov, ITERS, seed=1)
    mal_phi2, _, words2 = _mallet_dmr(docs, cov, ITERS, seed=2)

    # Align run 2 onto run 1 over the shared word set.
    wi = {w: j for j, w in enumerate(words)}
    mal_phi2_aligned = np.zeros_like(mal_phi)
    for j, w in enumerate(words2):
        if w in wi:
            mal_phi2_aligned[:, wi[w]] = mal_phi2[:, j]
    mallet_self_cos = float(
        (harness._row_normalize(mal_phi) @ harness._row_normalize(mal_phi2_aligned).T)
        .max(axis=1).mean()
    )

    ms = _space_topic(mal_phi, words)
    mallet_effect = float(mal_lam[ms, 1] - mal_lam[1 - ms, 1])

    # topica fit summary captured at regenerate time for the provenance log.
    t_phi, t_fe = _topica_fit(docs, cov, words)
    topica_cos = float(
        (harness._row_normalize(mal_phi) @ harness._row_normalize(t_phi).T)
        .max(axis=1).mean()
    )
    os_ = _space_topic(t_phi, words)
    topica_effect = float(t_fe[os_, 1] - t_fe[1 - os_, 1])

    harness.save_gold(
        NAME,
        arrays={
            "mallet_phi": mal_phi.astype(np.float32),
            "mallet_lambda": mal_lam.astype(np.float64),
            "words": np.array(words, dtype=object),
            "corpus": np.array(harness.docs_to_lines(docs), dtype=object),
            "covariate": np.array(cov, dtype=np.int64),
        },
        meta={
            "reference": f"Java MALLET {mallet_parity.mallet_home()} (DMRDriver.java)",
            "model": "DMR (Dirichlet-Multinomial Regression LDA)",
            "corpus": ("synthetic two-cluster fixture, 160 docs x 8 tokens, binary "
                       "is_space covariate switching animal/space vocab (from "
                       "parity/mallet_parity.py dmr_parity)"),
            "num_docs": len(docs),
            "vocab_size": len(words),
            "K": K,
            "iters": ITERS,
            "seeds": {"gold": 1, "noise_floor": 2},
            "margin": MARGIN,
            "mallet_self_cosine": mallet_self_cos,
            "mallet_effect": mallet_effect,
            "topica_cosine": topica_cos,
            "topica_effect": topica_effect,
            "date": datetime.date.today().isoformat(),
            "pass_bar": ("aligned topic cosine vs MALLET >= MALLET's own "
                         "seed-to-seed floor - margin, AND both effects "
                         "(space-minus-animal is_space weight) > 1.0"),
            "kind": "cross-implementation (Java MALLET DMRTopicModel)",
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  MALLET self cosine  : {mallet_self_cos:.4f}")
    print(f"  topica vs MALLET cos: {topica_cos:.4f}")
    print(f"  effect MALLET={mallet_effect:+.2f}  topica={topica_effect:+.2f}")


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    mal_phi = arrays["mallet_phi"].astype(np.float64)
    words = list(arrays["words"])
    docs = harness.lines_to_docs(str(arrays["corpus"]))
    cov = list(arrays["covariate"])
    margin = float(meta.get("margin", MARGIN))
    self_cos = float(meta.get("mallet_self_cosine", 0.0))
    mallet_effect = float(meta.get("mallet_effect", 0.0))

    t_phi, t_fe = _topica_fit(docs, cov, words)
    cos = float(
        (harness._row_normalize(mal_phi) @ harness._row_normalize(t_phi).T)
        .max(axis=1).mean()
    )
    os_ = _space_topic(t_phi, words)
    topica_effect = float(t_fe[os_, 1] - t_fe[1 - os_, 1])

    cos_bar = self_cos - margin
    result = {
        "cosine": cos,
        "cosine_bar": cos_bar,
        "cosine_margin": cos - cos_bar,
        "mallet_self_cosine": self_cos,
        "mallet_effect": mallet_effect,
        "topica_effect": topica_effect,
        "passes": bool(cos >= cos_bar and mallet_effect > 1.0 and topica_effect > 1.0),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica cosine vs MALLET : {cos:.4f} "
              f"(MALLET self {self_cos:.4f}, bar {cos_bar:.4f})")
        print(f"  effect MALLET={mallet_effect:+.2f}  topica={topica_effect:+.2f} "
              "(both must be > 1.0)")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(cos margin {result['cosine_margin']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
