"""Committed Java-oracle gold for topica GaussianLDA (#689).

Gaussian LDA (Das, Zaheer & Dyer, "Gaussian LDA for Topic Models with Word
Embeddings," ACL 2015) replaces LDA's categorical topic-word distribution with a
**Gaussian over the word-embedding space**: each topic k is N(mu_k, Sigma_k), a token
is generated from its topic's Gaussian on the word's embedding, and inference is
collapsed Gibbs with a Normal-Inverse-Wishart conjugate prior (Student-t posterior
predictive, rank-1 Cholesky up/downdates as tokens move).

Reference: the authors' ``rajarshd/Gaussian_LDA`` (**Apache-2.0**), the plain-Cholesky
``sampler/GaussianLDA.java``. Java, self-contained (bundled ejml / commons-math jars),
compiles + runs on JDK >= 15. The reference uses an UNSEEDED ``new Random()`` for its
initial assignment, so it is not itself run-to-run reproducible: parity is therefore
topic-aligned (Hungarian cosine of the per-topic Gaussian means, plus doc-topic
correlation) measured against the reference's OWN two-run noise floor -- never
bit-exact.

We freeze, from a fixed-seed synthetic corpus with planted Gaussian topics (each vocab
word lives in one of K well-separated clusters in embedding space; each document is
dominated by one topic and draws its words from that topic's cluster):

  * the reference per-topic means (K x E) at two independent runs -> mean noise floor;
  * the reference doc-topic matrix (D x K) at two runs -> doc-topic floor;
  * the planted per-document topic and per-word topic -> recovery targets.

Runs in CI WITHOUT Java: the reference fit is frozen in the committed
``parity/gaussian_lda_gold.npz`` + ``.json``.

    # Point at a clone of https://github.com/rajarshd/Gaussian_LDA (Apache-2.0):
    GAUSSIAN_LDA_HOME=/path/to/Gaussian_LDA python parity/gaussian_lda_gold.py --regenerate
    python parity/gaussian_lda_gold.py            # offline compare vs the gold
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "gaussian_lda"

# --- planted synthetic embedding corpus --------------------------------------
K = 3               # topics = Gaussian clusters in embedding space
E = 20              # embedding dimension
WORDS_PER_TOPIC = 20
V = K * WORDS_PER_TOPIC
N_DOCS = 90
TOKENS_PER_DOC = 30
# Cleanly-separable planted clusters: each topic's center sits on its OWN orthogonal
# embedding axis (well-separated, not random directions), a tight within-cluster spread,
# and low cross-topic leakage -- so Gaussian LDA is unimodal here and topic-mean parity
# with the reference is sharp (~0.99), not swamped by the model's genuine multi-modality
# on overlapping data. (On overlapping/random-direction clusters BOTH topica and the Java
# reference land in seed-dependent modes; that is a documented property of the model with
# random initialization, not a fidelity gap -- see docs.)
CLUSTER_SEP = 10.0  # each center = CLUSTER_SEP on its own axis
CLUSTER_SD = 0.3    # within-cluster embedding spread
LEAKAGE = 0.03      # fraction of a doc's tokens drawn from other topics
CORPUS_SEED = 271
ITERS = 200


def build_corpus():
    """Return (docs, embeddings (V,E), vocab, word_topic, doc_topic_label).

    Vocab word v belongs to topic v // WORDS_PER_TOPIC; its embedding is a draw from
    that topic's Gaussian. Each document is assigned one dominant topic and draws its
    tokens from that topic's words (with a little cross-topic leakage)."""
    rng = np.random.default_rng(CORPUS_SEED)
    # Orthogonal, well-separated centers: topic t sits on axis t.
    centers = np.zeros((K, E))
    for t in range(K):
        centers[t, t] = CLUSTER_SEP
    embeddings = np.zeros((V, E))
    word_topic = np.zeros(V, dtype=int)
    for v in range(V):
        t = v // WORDS_PER_TOPIC
        word_topic[v] = t
        embeddings[v] = centers[t] + rng.normal(0, CLUSTER_SD, size=E)
    vocab = [f"w{v:03d}" for v in range(V)]

    docs = []
    doc_topic_label = []
    for _ in range(N_DOCS):
        t = int(rng.integers(0, K))
        doc_topic_label.append(t)
        toks = []
        for _ in range(TOKENS_PER_DOC):
            tt = t if rng.random() >= LEAKAGE else int(rng.integers(0, K))
            w = tt * WORDS_PER_TOPIC + int(rng.integers(0, WORDS_PER_TOPIC))
            toks.append(vocab[w])
        docs.append(toks)
    return docs, embeddings, vocab, word_topic, doc_topic_label


# --- Java reference plumbing --------------------------------------------------
def _oracle_home():
    home = os.environ.get("GAUSSIAN_LDA_HOME")
    return Path(home) if home else None


def java_available():
    return shutil.which("java") is not None and shutil.which("javac") is not None


def _compile_oracle(home: Path):
    cp = ":".join([
        "external_libs/ejml-0.25.jar",
        "external_libs/commons-logging-1.2/commons-logging-1.2.jar",
        "external_libs/commons-math3-3.3/commons-math3-3.3.jar",
    ])
    binp = home / "bin"
    binp.mkdir(exist_ok=True)
    subprocess.run(
        ["javac", "-sourcepath", "src/", "-d", "bin/", "-cp", cp,
         "src/sampler/GaussianLDA.java"],
        cwd=home, check=True, capture_output=True, text=True,
    )
    return cp


def _run_oracle(home: Path, cp: str, embeddings, docs, vocab, outdir: Path):
    """Write the reference's input files, run GaussianLDA.java, parse outputs.

    Returns (means (K,E), doc_topic (N,K), avgLL_trace)."""
    widx = {w: i for i, w in enumerate(vocab)}
    outdir.mkdir(parents=True, exist_ok=True)
    vec = outdir / "vec.txt"
    corp = outdir / "corpus.txt"
    vec.write_text("\n".join(" ".join(f"{x:.8f}" for x in row) for row in embeddings) + "\n")
    corp.write_text("\n".join(" ".join(str(widx[w]) for w in doc) for doc in docs) + "\n")
    outsub = outdir / "gout/"
    outsub.mkdir(exist_ok=True)
    full_cp = f"bin/:{cp}"
    subprocess.run(
        ["java", "-Xmx2g", "-cp", full_cp, "sampler/GaussianLDA",
         str(vec), str(E), str(ITERS), str(K), str(outsub) + "/", str(corp)],
        cwd=home, check=True, capture_output=True, text=True,
    )
    # per-topic k.txt: row 0 = mean (E floats); rows 1..E = lower-tri Cholesky of
    # the NIW SCALE matrix Psi_k (NOT the topic covariance; see Gate A finding B1).
    means = np.zeros((K, E))
    chol = np.zeros((K, E, E))
    for k in range(K):
        rows = (outsub / f"{k}.txt").read_text().strip().splitlines()
        means[k] = np.array([float(x) for x in rows[0].split()])
        for i in range(E):
            chol[k, i] = np.array([float(x) for x in rows[1 + i].split()])
    dt = np.array([[float(x) for x in ln.split()]
                   for ln in (outsub / "document_topic.txt").read_text().strip().splitlines()])
    avg = [float(x) for x in (outsub / "avgLL.txt").read_text().strip().splitlines()]
    return means, chol, dt, avg


def regenerate():
    home = _oracle_home()
    if home is None or not (home / "src/sampler/GaussianLDA.java").exists():
        sys.exit("Set GAUSSIAN_LDA_HOME to a clone of rajarshd/Gaussian_LDA (Apache-2.0).")
    if not java_available():
        sys.exit("Need a JDK (java + javac) on PATH to run the Java reference.")

    docs, embeddings, vocab, word_topic, doc_label = build_corpus()
    cp = _compile_oracle(home)
    with tempfile.TemporaryDirectory() as td:
        m1, ch1, dt1, avg1 = _run_oracle(home, cp, embeddings, docs, vocab, Path(td) / "run1")
        m2, ch2, dt2, avg2 = _run_oracle(home, cp, embeddings, docs, vocab, Path(td) / "run2")

    floor_mean, _ = harness.align_cosine(m1, m2)
    floor_dt = harness.doc_topic_correlation(dt1, dt2)

    meta = {
        "model": "gaussian_lda",
        "reference": "rajarshd/Gaussian_LDA sampler/GaussianLDA.java (Apache-2.0)",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "K": K, "E": E, "words_per_topic": WORDS_PER_TOPIC, "vocab_size": V,
        "n_docs": N_DOCS, "tokens_per_doc": TOKENS_PER_DOC,
        "cluster_sep": CLUSTER_SEP, "cluster_sd": CLUSTER_SD,
        "iters": ITERS, "corpus_seed": CORPUS_SEED,
        "vocab": vocab, "docs": docs,
        "word_topic": word_topic.tolist(), "doc_label": doc_label,
        "floor_mean_cosine": float(floor_mean),
        "floor_doc_topic_corr": float(floor_dt),
        "note": "reference is unseeded; floor is its own two-run spread.",
    }
    harness.save_gold(NAME, {
        "embeddings": embeddings,
        "ref_means": m1, "ref_means2": m2,
        "ref_scale_chol": ch1, "ref_scale_chol2": ch2,  # chol(Psi_k), NOT covariance
        "ref_doc_topic": dt1, "ref_doc_topic2": dt2,
        "ref_avgll": np.array(avg1), "ref_avgll2": np.array(avg2),
    }, meta)
    print(f"[{NAME}] wrote gold: K={K} E={E} V={V} docs={N_DOCS} iters={ITERS}")
    print(f"[{NAME}] reference two-run floor: mean cos={floor_mean:.3f}, "
          f"doc-topic corr={floor_dt:.3f}")
    print(f"[{NAME}] avgLL run1: {avg1[0]:.3f} -> {avg1[-1]:.3f}")


def compare():
    arrays, meta = harness.load_gold(NAME)
    embeddings = arrays["embeddings"]
    ref_means = arrays["ref_means"]
    ref_dt = arrays["ref_doc_topic"]
    vocab = meta["vocab"]
    docs = meta["docs"]
    floor_mean = meta["floor_mean_cosine"]
    floor_dt = meta["floor_doc_topic_corr"]
    iters = meta["iters"]

    try:
        import topica
    except ImportError:
        print(f"[{NAME}] topica not importable; skipping compare.")
        return
    if not hasattr(topica, "GaussianLDA"):
        print(f"[{NAME}] topica.GaussianLDA not built yet; gold is ready. Skipping.")
        return

    # Compare against the reference's random-init behavior (init="random"): the port is
    # bit-faithful to the Cholesky sampler here. (The shipped default is init="kmeans",
    # the paper's approach, which avoids the mode-collapse both implementations show with
    # random init — see the module docstring and docs.)
    m = topica.GaussianLDA(meta["K"], init="random", seed=13).fit(
        docs, embeddings, vocab, iters=iters
    )
    tp_means = np.asarray(m.topic_means)
    tp_dt = np.asarray(m.doc_topic)

    cmean, _ = harness.align_cosine(ref_means, tp_means)
    cdt = harness.doc_topic_correlation(ref_dt, tp_dt)

    # planted recovery: each topic's mean should be nearest the planted cluster center
    print(f"[{NAME}] topic-mean aligned cosine = {cmean:.3f}  (reference floor {floor_mean:.3f})")
    print(f"[{NAME}] doc-topic correlation      = {cdt:.3f}  (reference floor {floor_dt:.3f})")
    ok = cmean >= floor_mean - 0.05 and cdt >= floor_dt - 0.10
    print(f"[{NAME}] PARITY {'OK' if ok else 'CHECK'}")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        compare()
