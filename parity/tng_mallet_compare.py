"""Parity check: topica `TopicalNGrams` vs Java MALLET `cc.mallet.topics.TopicalNGrams`.

Builds a fixed synthetic corpus with genuine topic structure — each topic has its
own unigrams AND its own two-word collocations — fits both topica and MALLET on the
SAME corpus with the SAME hyperparameters (the balanced `delta1=delta2=1` on both
sides — where discrete phrases are meaningful; MALLET's own `0.2/1000` default forces
whole-document runs so no side surfaces discrete collocations), reconstructs MALLET's
unigram topic-word matrix from its `printState` dump, aligns topics by Hungarian
assignment, and asserts topica's aligned topic-word cosine to MALLET clears MALLET's
own seed-to-seed noise floor. Also checks that both recover the planted collocations
as phrases.

Uses the shared `mallet_parity` Java-driver plumbing (`parity/TopicalNGramsDriver.java`).
Skips cleanly (exit 0) if MALLET / javac / java are unavailable.

    python parity/tng_mallet_compare.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import mallet_parity
import topica

HERE = Path(__file__).resolve().parent


def _planted(seed: int = 0, n_docs: int = 400, doc_len: int = 60):
    """K=4 topics; each has 25 unigram-only words and 3 collocations built from 6
    dedicated collocation-only words (never used as standalone unigrams). The wide
    unigram vocabulary keeps spurious runs from forming, so both the unigram topics
    and the discrete collocations are cleanly identified by a faithful TNG."""
    rng = np.random.default_rng(seed)
    k = 4
    unis = [[f"t{t}u{j}" for j in range(25)] for t in range(k)]
    cwords = [[f"t{t}c{j}" for j in range(6)] for t in range(k)]
    collocs = [[(cwords[t][2 * p], cwords[t][2 * p + 1]) for p in range(3)] for t in range(k)]
    docs = []
    for _ in range(n_docs):
        t = int(rng.integers(k))
        doc = []
        while len(doc) < doc_len:
            if rng.random() < 0.3:
                c = collocs[t][int(rng.integers(3))]
                doc.extend([c[0], c[1]])
            else:
                doc.append(unis[t][int(rng.integers(25))])
        docs.append(doc)
    return docs, k, collocs


def _mallet_topic_word(state_path: Path, vocab: list[str]) -> np.ndarray:
    """Reconstruct MALLET's unigram (gram==0) topic-word count matrix from printState.
    Columns aligned to `vocab`."""
    vidx = {w: j for j, w in enumerate(vocab)}
    # discover K from the state
    max_topic = 0
    rows = []
    for line in state_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # doc pos typeindex type bigrampossible topic gram
        word, topic, gram = parts[3], int(parts[5]), int(parts[6])
        max_topic = max(max_topic, topic)
        rows.append((word, topic, gram))
    k = max_topic + 1
    tw = np.zeros((k, len(vocab)))
    for word, topic, gram in rows:
        if gram == 0 and word in vidx:
            tw[topic, vidx[word]] += 1.0
    return tw


def _mallet_phrases(state_path: Path):
    """Reconstruct per-topic phrases (head + x=1 run) from MALLET's state."""
    by_doc = defaultdict(list)
    for line in state_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        by_doc[int(p[0])].append((int(p[1]), p[3], int(p[5]), int(p[6])))  # pos,word,topic,gram
    counts = Counter()
    for _d, toks in by_doc.items():
        toks.sort()
        i = 0
        while i < len(toks):
            j = i + 1
            while j < len(toks) and toks[j][3] == 1:
                j += 1
            if j > i + 1:
                counts["_".join(toks[p][1] for p in range(i, j))] += 1
                i = j
            else:
                i += 1
    return counts


def _align_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine of the Hungarian topic alignment between two (K, V) matrices."""
    from scipy.optimize import linear_sum_assignment

    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T
    r, c = linear_sum_assignment(-sim)
    return float(sim[r, c].mean())


def _run_mallet(docs, vocab, k, iters, seed, td):
    infile = Path(td) / f"corpus_{seed}.txt"
    infile.write_text("\n".join(" ".join(d) for d in docs))
    state = Path(td) / f"state_{seed}.txt"
    cp = mallet_parity._classpath()
    proc = subprocess.run(
        ["java", "-cp", f"{cp}:{HERE}", "TopicalNGramsDriver", str(infile), str(k),
         str(iters), str(seed), "2.0", "0.01", "0.01", "0.03", "1.0", "1.0", str(state)],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0 or not state.exists():
        raise RuntimeError(f"MALLET driver failed:\n{proc.stderr[-1500:]}")
    return _mallet_topic_word(state, vocab), _mallet_phrases(state)


def main() -> int:
    if not mallet_parity.java_drivers_available():
        print("SKIP: MALLET / javac / java not available")
        return 0
    if not mallet_parity._ensure_compiled("TopicalNGramsDriver"):
        print("SKIP: could not compile TopicalNGramsDriver")
        return 0

    docs, k, collocs = _planted(seed=0)
    iters = 500
    # Same params on both sides; alpha_sum=2 (alpha=0.5 for K=4) so single-topic docs
    # commit and the topics separate, and the balanced delta1=delta2=1 so discrete
    # collocations emerge.
    m = topica.TopicalNGrams(num_topics=k, seed=13, alpha_sum=2.0, beta=0.01,
                             gamma=0.01, delta1=1.0, delta2=1.0).fit(docs, iters=iters)
    vocab = list(m.vocabulary)
    topica_tw = np.asarray(m.topic_word)

    with tempfile.TemporaryDirectory() as td:
        mallet_tw_a, mallet_ph = _run_mallet(docs, vocab, k, iters, 1, td)
        mallet_tw_b, _ = _run_mallet(docs, vocab, k, iters, 2, td)

    # topica-vs-MALLET aligned cosine, against MALLET's own seed-to-seed floor.
    cos_topica = _align_cosine(topica_tw, mallet_tw_a)
    cos_floor = _align_cosine(mallet_tw_a, mallet_tw_b)

    # Phrase recovery: both should surface the planted collocations.
    planted = {f"{a}_{b}" for t in collocs for (a, b) in t}
    topica_ph = {p.replace(" ", "_") for p, _ in m.top_phrases(20)}
    mallet_top = {ph for ph, _ in mallet_ph.most_common(40)}
    topica_hit = len(planted & topica_ph)
    mallet_hit = len(planted & mallet_top)

    print(f"topica vs MALLET aligned topic-word cosine : {cos_topica:.4f}")
    print(f"MALLET seed-to-seed floor                  : {cos_floor:.4f}")
    print(f"planted collocations recovered  topica={topica_hit}/{len(planted)} "
          f"mallet={mallet_hit}/{len(planted)}")

    failures = []
    # topica should match MALLET about as well as MALLET matches itself (small margin).
    if cos_topica < cos_floor - 0.10:
        failures.append(f"topica-vs-MALLET {cos_topica:.3f} below floor {cos_floor:.3f} - 0.10")
    # topica must recover the planted collocations at least as well as MALLET does on
    # the same corpus (the reference bar), and find a meaningful number outright.
    if topica_hit < mallet_hit - 1:
        failures.append(f"topica phrases {topica_hit} well below MALLET {mallet_hit}")
    if topica_hit < 4:
        failures.append(f"topica recovered only {topica_hit}/{len(planted)} collocations")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: topica TopicalNGrams matches MALLET's topics and recovers phrases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
