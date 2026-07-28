"""Parity check: topica's FactorialLDA vs Michael Paul's reference Java fLDA (GPL v2).

The reference builds a fresh, unseeded ``java.util.Random`` per token, so it is NOT
seed-reproducible and NO bit/seed parity is possible; this is a QUALITATIVE check.
We fit both on the same planted topic x sentiment corpus and confirm both recover
the same six (topic, sentiment) tuple word-profiles (up to factor/component
permutation, since the factors are role-exchangeable). topica's own numerical
correctness is certified independently by the finite-difference gradient and
factor-tying tests in ``src/factorial_lda.rs`` and ``tests/test_factorial_lda.py``.

The reference source is GPL and is NOT vendored into topica (topica is
permissively licensed; the port is implemented from the paper's mathematics). This
script looks for a local checkout of the reference (Michael Paul's ``flda`` release,
e.g. under ``~/Downloads/flda``) via the ``TOPICA_FLDA_REF`` environment variable or
a couple of default locations, and skips cleanly when it — or a Java toolchain — is
absent.
"""

import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import topica

TOPICS = {
    0: ["game", "team", "score", "play", "ball", "coach", "season", "league"],
    1: ["market", "stock", "price", "trade", "bank", "profit", "invest", "shares"],
    2: ["film", "movie", "actor", "scene", "director", "plot", "cast", "screen"],
}
SENTS = {
    0: ["great", "love", "excellent", "wonderful", "best", "brilliant", "superb"],
    1: ["terrible", "hate", "awful", "worst", "poor", "boring", "disappointing"],
}
BG = ["the", "a", "and", "of", "to", "is", "was", "very", "really", "this"]


def _find_reference():
    cand = []
    env = os.environ.get("TOPICA_FLDA_REF")
    if env:
        cand.append(env)
    cand += [
        os.path.expanduser("~/Downloads/flda"),
        os.path.expanduser("~/flda"),
    ]
    for d in cand:
        if d and os.path.isfile(os.path.join(d, "FLDA.java")):
            return d
    return None


def _planted(n_docs=300, doc_len=40, seed=12345):
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        t, s = rng.randrange(3), rng.randrange(2)
        words = []
        for _ in range(doc_len):
            r = rng.random()
            if r < 0.45:
                words.append(rng.choice(TOPICS[t]))
            elif r < 0.80:
                words.append(rng.choice(SENTS[s]))
            else:
                words.append(rng.choice(BG))
        docs.append(words)
    return docs


def _planted_profiles(vocab):
    vi = {w: i for i, w in enumerate(vocab)}

    def prof(words, mass):
        out = np.zeros(len(vocab))
        for w in words:
            if w in vi:
                out[vi[w]] += mass / len(words)
        return out

    profs = []
    for t in range(3):
        for s in range(2):
            p = prof(TOPICS[t], 0.45) + prof(SENTS[s], 0.35) + prof(BG, 0.20)
            profs.append(p / p.sum())
    return np.array(profs)


def _mean_aligned_cosine(phi, planted):
    from scipy.optimize import linear_sum_assignment

    def norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    sim = norm(phi) @ norm(planted).T
    r, c = linear_sum_assignment(-sim)
    return float(np.mean([sim[i, j] for i, j in zip(r, c)]))


def test_topica_recovers_planted_tuples():
    """topica leg: recovers the six planted tuples (runs with or without the ref)."""
    docs = _planted()
    m = topica.FactorialLDA(factor_sizes=[3, 2], seed=1)
    m.fit(docs, iters=800, samples=150)
    profs = _planted_profiles(list(m.vocabulary))
    cos = _mean_aligned_cosine(m.topic_word, profs)
    assert cos > 0.9, f"topica mean aligned cosine {cos:.3f}"


def test_reference_qualitative_agreement(tmp_path):
    """Reference leg (qualitative): the Java fLDA, when available, also recovers the
    six planted tuples. Skips cleanly without the GPL reference or a JDK."""
    ref = _find_reference()
    if ref is None:
        pytest.skip("reference fLDA source not found (set TOPICA_FLDA_REF)")
    if not (shutil.which("javac") and shutil.which("java")):
        pytest.skip("Java toolchain (javac/java) not available")

    work = tmp_path / "flda"
    work.mkdir()
    for fn in ("FLDA.java", "LearnTopicModel.java", "TopicModel.java"):
        src = os.path.join(ref, fn)
        if not os.path.isfile(src):
            pytest.skip(f"reference file {fn} missing")
        shutil.copy(src, work / fn)
    if subprocess.run(["javac", *[str(work / f) for f in os.listdir(work)]]).returncode != 0:
        pytest.skip("reference did not compile")

    docs = _planted()
    inp = work / "planted.txt"
    with open(inp, "w") as f:
        for i, d in enumerate(docs):
            f.write(f"{i} " + " ".join(d) + "\n")
    r = subprocess.run(
        ["java", "-cp", str(work), "LearnTopicModel", "-model", "flda",
         "-input", str(inp), "-K", "2", "-Z", "3", "-Y", "2",
         "-iters", "800", "-samples", "150"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"reference run failed: {r.stderr[-300:]}")

    # Read the reference's per-component omega weights, build tuple profiles from
    # exp(omega) and align to the planted profiles.
    def read_omega(path):
        rows = {}
        for line in open(path):
            p = line.split()
            rows[p[0]] = [float(x) for x in p[1:]]
        return rows

    ozw0 = read_omega(str(inp) + ".omegaZW0")
    ozw1 = read_omega(str(inp) + ".omegaZW1")
    vocab = list(ozw0.keys())
    profs_planted = _planted_profiles(vocab)
    # tuple (z0, z1) profile ~ softmax over words of omega0[z0][w] + omega1[z1][w]
    phi = []
    for z0 in range(3):
        for z1 in range(2):
            w = np.array([ozw0[t][z0] + ozw1[t][z1] for t in vocab])
            e = np.exp(w - w.max())
            phi.append(e / e.sum())
    phi = np.array(phi)
    cos = _mean_aligned_cosine(phi, profs_planted)
    # Qualitative: the reference should also land near the planted structure.
    assert cos > 0.6, f"reference mean aligned cosine {cos:.3f}"

    # Direct topica-vs-reference agreement: fit topica on the same corpus and align
    # its tuple phi to the reference's tuple profiles (up to permutation). This
    # compares the two implementations' outputs to each other, not just each to the
    # planted profiles. Qualitative only (the reference is non-reproducible).
    tm = topica.FactorialLDA(factor_sizes=[3, 2], seed=1)
    tm.fit(docs, iters=800, samples=150)
    tvocab = list(tm.vocabulary)
    # reorder reference phi columns to topica's vocabulary
    ref_by_word = phi  # columns are in `vocab` order
    col = {w: i for i, w in enumerate(vocab)}
    ref_aligned = np.array(
        [[row[col[w]] if w in col else 0.0 for w in tvocab] for row in ref_by_word]
    )
    cos_tv = _mean_aligned_cosine(tm.topic_word, ref_aligned)
    assert cos_tv > 0.6, f"topica-vs-reference mean aligned cosine {cos_tv:.3f}"


if __name__ == "__main__":
    docs = _planted()
    m = topica.FactorialLDA(factor_sizes=[3, 2], seed=1)
    m.fit(docs, iters=800, samples=150)
    profs = _planted_profiles(list(m.vocabulary))
    print("topica mean aligned cosine:", _mean_aligned_cosine(m.topic_word, profs))
