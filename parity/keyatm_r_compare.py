"""Cross-implementation validation: topica KeyATM vs. the R `keyATM` package.

R's `keyATM` and topica are independent implementations of the same model
(keyword-assisted collapsed Gibbs; Eshima, Imai & Sasaki 2024). They share no
code and no RNG, and — unlike STM's spectral init — keyATM initializes at
random, so even two R runs from different seeds land in different basins. As
with the STM parity check, validation here is *statistical*: fit both on the
SAME tokenized corpus + keyword sets and ask whether they recover the same
topics, benchmarked against R's own seed-to-seed reproducibility.

keyATM's signature is keyword anchoring, so the fair, sharp comparison is the
*keyword* topics: their content is pinned by the supplied keywords, so they
should agree across implementations at least as well as R agrees with itself.
The free (no-keyword) topics are reported too, for context.

This shells out to `Rscript` with the `keyATM` package; callers should check
`r_keyatm_available()` first. Run directly:

    python parity/keyatm_r_compare.py

It preprocesses `examples/poliblog.csv` (already stemmed), hands the identical
documents and keyword sets to both engines, and prints the alignment. Skips
(exit 0) if R or the `keyATM` package is unavailable.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLIBLOG = os.path.join(ROOT, "examples", "poliblog.csv")

ITERS = int(os.environ.get("KEYATM_ITERS", "1500"))
MIN_DOC_FREQ = int(os.environ.get("KEYATM_MIN_DF", "3"))
NUM_REGULAR = int(os.environ.get("KEYATM_REGULAR", "6"))

# Keyword sets (stemmed, to match the corpus). One keyword topic each; the
# regular topics are added on top via no_keyword_topics.
KEYWORD_SETS = {
    "econ": ["tax", "economi", "econom", "market", "spend", "budget"],
    "war": ["iraq", "iraqi", "war", "troop", "militari", "surg"],
    "elect": ["obama", "mccain", "vote", "voter", "campaign", "elect"],
    "social": ["abort", "gay", "marriag", "religi", "church", "famili"],
}


def r_keyatm_available() -> bool:
    """True iff `Rscript` is on PATH and the `keyATM` package loads."""
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("keyATM", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().endswith("TRUE")


def load_and_prep():
    """Load the (already preprocessed) poliblog corpus and keep keyword sets to
    the vocabulary. Returns (docs, keywords) where docs is list[list[str]] and
    keywords is dict[name -> list[str]] (empty sets dropped)."""
    with open(POLIBLOG, newline="") as f:
        rows = list(csv.DictReader(f))
    toks = [r["text"].split() for r in rows]
    df = Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= MIN_DOC_FREQ}
    toks = [[w for w in d if w in vocab] for d in toks]
    docs = [d for d in toks if d]
    keywords = {
        name: [w for w in ws if w in vocab]
        for name, ws in KEYWORD_SETS.items()
    }
    keywords = {name: ws for name, ws in keywords.items() if ws}
    return docs, keywords


# R driver. Reads space-joined docs + the keyword sets, fits keyATM Base with
# two seeds, and exports each topic-word matrix (K x V, vocab columns). keyATM
# orders the keyword topics first (in the keyword-list order), then the regular
# topics, which matches topica's ordering.
_R_DRIVER = r"""
suppressMessages(library(keyATM))
suppressMessages(library(quanteda))
lines <- readLines(file.path(dir, "vdocs.txt"))
# Each line is one already-tokenized document; build a dfm so keyATM_read takes
# the text directly (a bare character vector would be read as file paths).
toks  <- quanteda::as.tokens(strsplit(lines, " ", fixed = TRUE))
dfmat <- quanteda::dfm(toks)
docs  <- keyATM_read(texts = dfmat)
keywords <- jsonlite::fromJSON(file.path(dir, "keywords.json"), simplifyVector = FALSE)
keywords <- lapply(keywords, function(x) unlist(x))
phi_of <- function(seed) {
  out <- keyATM(docs = docs, model = "base", no_keyword_topics = NREG,
                keywords = keywords,
                options = list(seed = seed, iterations = ITERS, verbose = FALSE))
  out$phi  # K x V, colnames = vocab; keyword topics first
}
p1 <- phi_of(1)
p2 <- phi_of(2)
write.csv(p1, file.path(dir, "r_phi1.csv"))
write.csv(p2, file.path(dir, "r_phi2.csv"))
cat(ncol(p1), "\n")  # sanity
cat("ok\n")
"""


def _read_r_phi(path, vocab):
    """Read an R keyATM phi CSV (K rows; first column is the row name, remaining
    columns named by vocab) into a K x |vocab| array aligned to `vocab`."""
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        cols = [h.strip('"') for h in header[1:]]  # drop the row-name column
        rows = [[float(x) for x in row[1:]] for row in rdr]
    mat = np.array(rows)
    idx = {w: i for i, w in enumerate(cols)}
    out = np.zeros((mat.shape[0], len(vocab)))
    for j, w in enumerate(vocab):
        if w in idx:
            out[:, j] = mat[:, idx[w]]
    return out


def _best_alignment_cosine(a, b, return_pairs=False):
    """Mean cosine of the best one-to-one topic alignment between two K x V
    topic-word matrices (greedy)."""
    k = a.shape[0]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T
    used = set()
    total = 0.0
    pairs = []
    for i in np.argsort(-sim.max(axis=1)):
        for j in np.argsort(-sim[i]):
            if j not in used:
                used.add(j)
                total += sim[i, j]
                pairs.append((int(i), int(j), float(sim[i, j])))
                break
    return (total / k, pairs) if return_pairs else total / k


def run(verbose: bool = True) -> dict:
    if not r_keyatm_available():
        raise RuntimeError("Rscript with the 'keyATM' package is not available")

    from topica.models import KeyATM

    docs, keywords = load_and_prep()
    num_keyword = len(keywords)
    num_topics = num_keyword + NUM_REGULAR

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "keywords.json"), "w") as f:
            json.dump(keywords, f)

        script = (
            f'dir <- "{d}"\nNREG <- {NUM_REGULAR}\nITERS <- {ITERS}\n'
            'if (!requireNamespace("jsonlite", quietly=TRUE)) stop("need jsonlite")\n'
            + _R_DRIVER
        )
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=3600
        )
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")

        # Recover R's vocab order from the phi header.
        with open(os.path.join(d, "r_phi1.csv"), newline="") as f:
            header = next(csv.reader(f))
        r_vocab = [h.strip('"') for h in header[1:]]
        r_phi1 = _read_r_phi(os.path.join(d, "r_phi1.csv"), r_vocab)
        r_phi2 = _read_r_phi(os.path.join(d, "r_phi2.csv"), r_vocab)

        # topica, same docs + keywords, keyword topics first (its default order).
        model = KeyATM(keywords, num_topics=num_topics, seed=1)
        model.fit(docs, iters=ITERS)
        tt_vocab = list(model.vocabulary)
        tt_phi_raw = np.asarray(model.topic_word)

        tt_idx = {w: i for i, w in enumerate(tt_vocab)}
        tt_phi = np.zeros((tt_phi_raw.shape[0], len(r_vocab)))
        for j, w in enumerate(r_vocab):
            if w in tt_idx:
                tt_phi[:, j] = tt_phi_raw[:, tt_idx[w]]

    # Overall alignment, and the sharper keyword-topic-only view (topics
    # 0..num_keyword are the anchored ones in both engines).
    overall_cosine, pairs = _best_alignment_cosine(r_phi1, tt_phi, return_pairs=True)
    r_self_cosine = _best_alignment_cosine(r_phi1, r_phi2)
    kw_r_tt = _best_alignment_cosine(r_phi1[:num_keyword], tt_phi[:num_keyword])
    kw_r_self = _best_alignment_cosine(r_phi1[:num_keyword], r_phi2[:num_keyword])

    result = {
        "overall_cosine": overall_cosine,
        "r_self_cosine": r_self_cosine,
        "keyword_cosine": kw_r_tt,
        "keyword_r_self_cosine": kw_r_self,
        "num_topics": num_topics,
        "num_keyword": num_keyword,
        "n_docs": len(docs),
        "vocab_size": len(r_vocab),
    }
    if verbose:
        print(f"corpus: {len(docs)} docs, {len(r_vocab)} vocab, "
              f"{num_topics} topics ({num_keyword} keyword + {NUM_REGULAR} regular)")
        print(f"keyword topics  — R vs topica : {kw_r_tt:.3f}   (R vs R: {kw_r_self:.3f})")
        print(f"all topics      — R vs topica : {overall_cosine:.3f}   (R vs R: {r_self_cosine:.3f})")
        # keyATM is random-init Gibbs, so the bar is R's own seed-to-seed spread.
        gap = kw_r_self - kw_r_tt
        if kw_r_tt >= 0.9 or gap <= 0.05:
            verdict = (
                f"PASS — topica's keyword topics match R's as well as R matches "
                f"itself across seeds (cosine {kw_r_tt:.3f})"
            )
        elif gap <= 0.15:
            verdict = f"COMPARABLE — within R's own seed-to-seed spread (gap {gap:.3f})"
        else:
            verdict = f"DIVERGENT — {gap:.3f} beyond R's own seed spread; investigate"
        print(verdict)
    return result


if __name__ == "__main__":
    import sys

    if not r_keyatm_available():
        print("SKIP: Rscript with the 'keyATM' package is not available")
        sys.exit(0)
    run(verbose=True)
