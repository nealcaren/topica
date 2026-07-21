"""Parity check: topica's BTM vs the reference R `BTM` package (Wijffels, Apache-2.0).

Fits both implementations on the same fixed synthetic short-text corpus with
matching hyperparameters, aligns the topic-word matrices (greedy cosine), and
asserts the aligned per-topic cosine is high. The RNG differs (topica uses
ChaCha8, R uses Mersenne-Twister), so this is a distributional parity check, not
bit-exact; on well-separated corpora both converge to the same topics.

Skips cleanly when Rscript or the R BTM package is unavailable.
"""

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np

import topica

K, BLOCK, NDOCS, LENGTH = 4, 10, 600, 6
ALPHA, BETA, WINDOW, ITERS = 50.0 / K, 0.01, 15, 800
COSINE_MIN = 0.95


def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}_w{i}" for b in range(K) for i in range(BLOCK)]
    docs = [
        [f"b{d % K}_w{int(rng.integers(BLOCK))}" for _ in range(LENGTH)]
        for d in range(NDOCS)
    ]
    return docs, vocab


def _reference_phi(docs, tmp):
    """Run the R BTM package and return (tokens, phi[V x K]); None if unavailable."""
    if shutil.which("Rscript") is None:
        return None
    csvp = os.path.join(tmp, "docs.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "token"])
        for i, doc in enumerate(docs):
            for tok in doc:
                w.writerow([f"doc{i}", tok])
    phip = os.path.join(tmp, "phi.csv")
    rscript = f"""
    ok <- suppressWarnings(suppressMessages(require(BTM)))
    if (!ok) quit(status = 42)
    set.seed(123)
    d <- read.csv("{csvp}", stringsAsFactors = FALSE, colClasses = "character")
    m <- BTM(d, k = {K}, alpha = {ALPHA}, beta = {BETA}, iter = {ITERS},
             window = {WINDOW}, background = FALSE, trace = FALSE)
    phi <- m$phi
    write.csv(data.frame(token = rownames(phi), phi), "{phip}", row.names = FALSE)
    """
    res = subprocess.run(["Rscript", "-e", rscript], capture_output=True, text=True)
    if res.returncode == 42 or not os.path.exists(phip):
        return None
    res.check_returncode()
    tokens, rows = [], []
    with open(phip) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            tokens.append(row[0])
            rows.append([float(x) for x in row[1:]])
    return tokens, np.array(rows)


def _aligned_cosine(a, b):
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    cm = an @ bn.T
    used, cs = set(), []
    for i in range(a.shape[0]):
        j = max((x for x in range(b.shape[0]) if x not in used), key=lambda x: cm[i, x])
        used.add(j)
        cs.append(cm[i, j])
    return cs


def run_comparison():
    docs, _ = _corpus()
    tmp = tempfile.mkdtemp()
    ref = _reference_phi(docs, tmp)
    if ref is None:
        print("Reference R BTM not available. Skipping parity check.")
        return
    r_tokens, r_phi = ref  # V x K
    rmap = {t: i for i, t in enumerate(r_tokens)}

    m = topica.BTM(num_topics=K, alpha=ALPHA, beta=BETA, iters=ITERS, window=WINDOW, seed=123)
    m.fit(docs)
    t_phi = m.topic_word  # K x V
    vocab = m.vocabulary
    r_aligned = np.zeros((K, len(vocab)))
    for j, tok in enumerate(vocab):
        if tok in rmap:
            r_aligned[:, j] = r_phi[rmap[tok], :]

    cs = _aligned_cosine(t_phi, r_aligned)
    print("topica-vs-R BTM aligned topic-word cosine:")
    for i, c in enumerate(cs):
        print(f"  topic {i}: cosine={c:.6f}")
        assert c > COSINE_MIN, f"BTM parity mismatch: cosine {c:.6f} < {COSINE_MIN}"
    print(f"mean cosine {np.mean(cs):.6f}")
    print("SUCCESS: BTM parity check passed!")


if __name__ == "__main__":
    run_comparison()
