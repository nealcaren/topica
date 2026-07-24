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


def _corpus_varlen(seed=0):
    """Variable-length corpus: alternating short (3-token) and long (20-token)
    documents. Biterm participation then diverges from raw token frequency, so the
    background distribution `pw_b` (and every topic it perturbs) is only correct if
    `pw_b` is accumulated over biterm words, not tokens (#492).

    Uses K-1 planted blocks so the K-1 *content* topics (topic 0 is the background
    under background=True) cover the blocks one-to-one — otherwise which block the
    scarce content topics merge is RNG-dependent and not cross-impl comparable."""
    nblocks = K - 1
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}_w{i}" for b in range(nblocks) for i in range(BLOCK)]
    docs = []
    for d in range(NDOCS):
        length = 3 if d % 2 == 0 else 20
        docs.append([f"b{d % nblocks}_w{int(rng.integers(BLOCK))}" for _ in range(length)])
    return docs, vocab


def _reference_phi(docs, tmp, background=False):
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
    bg = "TRUE" if background else "FALSE"
    rscript = f"""
    ok <- suppressWarnings(suppressMessages(require(BTM)))
    if (!ok) quit(status = 42)
    set.seed(123)
    d <- read.csv("{csvp}", stringsAsFactors = FALSE, colClasses = "character")
    m <- BTM(d, k = {K}, alpha = {ALPHA}, beta = {BETA}, iter = {ITERS},
             window = {WINDOW}, background = {bg}, trace = FALSE)
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


def _compare(docs, background, label, cosine_min=COSINE_MIN):
    """Fit R + topica BTM on `docs` with the given `background` flag and assert the
    aligned topic-word cosine clears the bar. Returns False if R is unavailable.

    Under background=True the shared background topic (index 0 in both topica and
    R BTM) is EXCLUDED: its phi is emitted from the biterm counts it accumulated,
    not the pw_b it sampled from, so it depends on each engine's RNG and is not
    cross-implementation comparable. The K-1 content topics are what the pw_b fix
    (#492) governs, and they are compared."""
    tmp = tempfile.mkdtemp()
    ref = _reference_phi(docs, tmp, background=background)
    if ref is None:
        print(f"[{label}] Reference R BTM not available. Skipping.")
        return False
    r_tokens, r_phi = ref  # V x K
    rmap = {t: i for i, t in enumerate(r_tokens)}

    m = topica.BTM(
        num_topics=K, alpha=ALPHA, beta=BETA, iters=ITERS, window=WINDOW,
        background=background, seed=123,
    )
    m.fit(docs)
    t_phi = np.asarray(m.topic_word)  # K x V
    vocab = m.vocabulary
    r_aligned = np.zeros((K, len(vocab)))
    for j, tok in enumerate(vocab):
        if tok in rmap:
            r_aligned[:, j] = r_phi[rmap[tok], :]

    # Topic 0 is the background under background=True; compare only content topics.
    t_cmp, r_cmp = (t_phi[1:], r_aligned[1:]) if background else (t_phi, r_aligned)
    cs = _aligned_cosine(t_cmp, r_cmp)
    kind = "content topic" if background else "topic"
    print(f"[{label}] topica-vs-R BTM aligned cosine (background={background}):")
    for i, c in enumerate(cs):
        print(f"  {kind} {i}: cosine={c:.6f}")
        assert c > cosine_min, f"[{label}] BTM parity mismatch: cosine {c:.6f} < {cosine_min}"
    print(f"[{label}] mean cosine {np.mean(cs):.6f}  PASS")
    return True


def run_comparison():
    ran = False
    # Leg 1: the original uniform-length, background=False content parity.
    docs, _ = _corpus()
    ran |= _compare(docs, background=False, label="uniform/bg=False")
    # Leg 2 (#492): a background=True variable-length sanity check — confirms
    # topica's background=True mode still tracks R BTM (K-1 content topics match at
    # cosine ~1.0; the RNG-dependent background topic 0 is excluded). NOTE this is
    # NOT a discriminating test of the pw_b fix: on separable corpora R BTM's
    # background topic stays near-empty (theta_0 ~ 0.005), so token- vs
    # biterm-weighted pw_b barely propagates to the content topics and the leg
    # passes either way. The discriminating guard for the fix is the deterministic
    # Rust unit test `pw_b_is_biterm_weighted_not_token_weighted`, which pins the
    # exact biterm-participation shares.
    vdocs, _ = _corpus_varlen()
    ran |= _compare(vdocs, background=True, label="varlen/bg=True")
    if not ran:
        print("Reference R BTM not available. Skipping parity check.")
        return
    print("SUCCESS: BTM parity check passed!")


if __name__ == "__main__":
    run_comparison()
