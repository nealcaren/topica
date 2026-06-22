"""Shared parity harness for the R-`stm` family (and future waves).

This module factors the corpus prep, the R-subprocess plumbing, the topic
alignment, and the gold-fixture I/O that the per-model parity scripts share, so a
parity check can be split into two phases:

  * a ``--regenerate`` phase that runs the reference toolchain ONCE and writes a
    committed ``parity/<name>_gold.npz`` (the arrays) plus ``parity/<name>_gold.json``
    (a human-readable provenance log), and
  * a default phase that loads the committed gold and compares topica against it
    WITHOUT the reference toolchain installed — the whole point being that CI can
    validate topica against R `stm` with no Rscript present.

It generalizes the pattern already used by ``parity/nmf_vs_sklearn.py`` (the
run-once/commit-gold/compare-offline split) to the R-`stm` family, whose corpus
prep and R driver previously lived inline in ``stm_r_compare.py`` /
``ctm_r_compare.py`` and ran R live on every invocation.

Dependency-light by design: numpy, scipy, and the standard library only.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GADARIAN = ROOT / "examples" / "gadarian.csv"
POLIBLOG = ROOT / "examples" / "poliblog.csv"
STOPLIST = ROOT / "examples" / "english-stoplist.txt"


# --------------------------------------------------------------------------- #
# Reference availability
# --------------------------------------------------------------------------- #
def r_available(pkg: str = "stm") -> bool:
    """True iff ``Rscript`` is on PATH and the named R package loads.

    Generalizes the per-script ``r_stm_available`` helpers.
    """
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", f'cat(requireNamespace("{pkg}", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().endswith("TRUE")


# --------------------------------------------------------------------------- #
# Shared corpus prep
# --------------------------------------------------------------------------- #
def gadarian_corpus():
    """Preprocess the gadarian corpus exactly as ``examples/stm_vignette.py``.

    Returns ``(docs, treatment, pid_rep, vocab)`` where ``docs`` is
    ``list[list[str]]``, ``treatment`` / ``pid_rep`` are float arrays aligned to
    the kept documents, and ``vocab`` is the sorted set of retained word types.
    Both the STM and CTM gold use this corpus.
    """
    with open(GADARIAN, newline="") as f:
        rows = list(csv.DictReader(f))
    text = [r["open.ended.response"] for r in rows]
    treatment = np.array([float(r["treatment"]) for r in rows])
    pid = np.array([float(r["pid_rep"]) for r in rows])
    stopwords = set(STOPLIST.read_text().split())

    def tok(s):
        return [
            w
            for w in "".join(c.lower() if c.isalnum() else " " for c in s).split()
            if len(w) >= 3 and w not in stopwords
        ]

    toks = [tok(t) for t in text]
    df = Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= 3}
    toks = [[w for w in d if w in vocab] for d in toks]
    keep = np.array([len(d) > 0 for d in toks])
    docs = [d for d, k in zip(toks, keep) if k]
    return docs, treatment[keep], pid[keep], sorted({w for d in docs for w in d})


# Default poliblog subsample: fixed size + seed so the committed gold stays small
# and the offline refit reproduces the EXACT documents the gold was built from.
POLIBLOG_N_DOCS = 2000
POLIBLOG_SEED = 271
POLIBLOG_MIN_DF = 3


def poliblog_corpus(n_docs: int = POLIBLOG_N_DOCS, seed: int = POLIBLOG_SEED):
    """Subsample + preprocess the poliblog vignette corpus deterministically.

    The poliblog text is already stemmed/stopworded; we take a fixed-seed
    subsample of ``n_docs`` rows (so the committed gold fixture stays small and
    the model is well-identified), apply a light document-frequency prune, and
    return ``(docs, rating_lib, day, vocab)`` where ``docs`` is
    ``list[list[str]]``, ``rating_lib`` is the 0/1 Liberal dummy (Conservative =
    baseline, matching R's alphabetical factor coding), and ``day`` is the raw
    day-of-year covariate. Both engines get the identical surviving vocabulary,
    so this only sets corpus size, not the parity.

    Used by both the STM gold (prevalence ``~ rating + s(day)``) and the CTM gold
    (no covariates). Unlike multimodal gadarian K=3, poliblog K=20 is
    well-identified, so topica and R land on essentially the same Spectral
    solution.
    """
    with open(POLIBLOG, newline="") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rows), size=min(n_docs, len(rows)), replace=False))
    rows = [rows[i] for i in idx]

    toks = [r["text"].split() for r in rows]
    rating_lib = np.array([1.0 if r["rating"] == "Liberal" else 0.0 for r in rows])
    day = np.array([float(r["day"]) for r in rows])

    df = Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= POLIBLOG_MIN_DF}
    toks = [[w for w in d if w in vocab] for d in toks]
    keep = np.array([len(d) > 0 for d in toks])
    docs = [d for d, k in zip(toks, keep) if k]
    return docs, rating_lib[keep], day[keep], sorted({w for d in docs for w in d})


def docs_to_lines(docs: list[list[str]]) -> str:
    """Serialize tokenized docs to one space-joined line per doc (for the npz)."""
    return "\n".join(" ".join(doc) for doc in docs) + "\n"


def lines_to_docs(text: str) -> list[list[str]]:
    """Inverse of :func:`docs_to_lines`: parse space-joined lines into docs."""
    return [line.split() for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# R subprocess plumbing
# --------------------------------------------------------------------------- #
def run_rscript(driver: str, files: dict[str, str], reads: list[str], timeout: int = 1200) -> dict[str, str]:
    """Run an R ``driver`` in a tempdir and return the contents of named outputs.

    ``files`` maps a filename -> text content to write into the tempdir before
    running. The ``driver`` is prefixed with ``dir <- "<tempdir>"`` so it can
    reference inputs/outputs via ``file.path(dir, ...)``. ``reads`` names the
    output files the driver writes; their contents are read back and returned in
    a ``{name: contents}`` dict.

    Raises ``RuntimeError`` if the driver does not print ``ok`` on stdout.
    """
    with tempfile.TemporaryDirectory() as d:
        for name, content in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(content)
        script = f'dir <- "{d}"\n' + driver
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=timeout
        )
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")
        out = {}
        for name in reads:
            with open(os.path.join(d, name)) as f:
                out[name] = f.read()
        out["__stdout__"] = proc.stdout
        return out


def read_r_beta_csv(text: str, vocab: list[str]) -> np.ndarray:
    """Parse an R topic-word CSV (K rows, vocab-named columns) into a
    ``K x len(vocab)`` array aligned to ``vocab`` (the R column order)."""
    rdr = csv.reader(text.splitlines())
    header = next(rdr)
    cols = [h.strip('"') for h in header]
    rows = [[float(x) for x in row] for row in rdr if row]
    mat = np.array(rows)
    idx = {w: i for i, w in enumerate(cols)}
    out = np.zeros((mat.shape[0], len(vocab)))
    for j, w in enumerate(vocab):
        if w in idx:
            out[:, j] = mat[:, idx[w]]
    return out


# --------------------------------------------------------------------------- #
# Topic alignment & agreement metrics
# --------------------------------------------------------------------------- #
def _row_normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def align_cosine(beta_a: np.ndarray, beta_b: np.ndarray):
    """Hungarian one-to-one topic alignment of ``beta_b`` onto ``beta_a`` by
    cosine similarity of the topic-word rows.

    Returns ``(mean_cosine, perm)`` where ``beta_b[perm]`` is row-aligned to
    ``beta_a``. Uses ``scipy.optimize.linear_sum_assignment`` (exact, not greedy).
    """
    from scipy.optimize import linear_sum_assignment

    an = _row_normalize(np.asarray(beta_a, dtype=float))
    bn = _row_normalize(np.asarray(beta_b, dtype=float))
    sim = an @ bn.T  # K x K cosine
    row, col = linear_sum_assignment(-sim)
    perm = np.empty(beta_a.shape[0], dtype=int)
    perm[row] = col
    mean_cos = float(np.mean([sim[i, perm[i]] for i in range(beta_a.shape[0])]))
    return mean_cos, perm


def top_word_jaccard(beta_a: np.ndarray, beta_b: np.ndarray, n: int = 10) -> float:
    """Mean top-``n`` word Jaccard overlap under the cosine alignment of
    ``beta_b`` onto ``beta_a``."""
    _, perm = align_cosine(beta_a, beta_b)
    b_aligned = np.asarray(beta_b)[perm]
    scores = []
    for i in range(beta_a.shape[0]):
        sa = set(np.argsort(np.asarray(beta_a)[i])[::-1][:n])
        sb = set(np.argsort(b_aligned[i])[::-1][:n])
        u = len(sa | sb)
        scores.append(len(sa & sb) / u if u else 0.0)
    return float(np.mean(scores))


def doc_topic_correlation(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """Mean per-topic Pearson correlation of two doc-topic matrices (D x K),
    aligned by the cosine alignment of their topic columns."""
    a = np.asarray(theta_a, dtype=float)
    b = np.asarray(theta_b, dtype=float)
    # Align columns (topics) of b onto a using the column vectors as "rows".
    _, perm = align_cosine(a.T, b.T)
    b = b[:, perm]
    cors = []
    for k in range(a.shape[1]):
        if np.std(a[:, k]) < 1e-12 or np.std(b[:, k]) < 1e-12:
            continue
        cors.append(float(np.corrcoef(a[:, k], b[:, k])[0, 1]))
    return float(np.mean(cors)) if cors else float("nan")


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """Adjusted Rand index between two label assignments, numpy-only.

    A vendored equivalent of ``sklearn.metrics.adjusted_rand_score`` so the
    clustering gold tests (Top2Vec/BERTopic) stay reference-toolchain-free at test
    time — CI installs numpy/scipy but not scikit-learn (which is only a *reference*
    here, used at regenerate time)."""
    a = np.asarray(labels_true)
    b = np.asarray(labels_pred)
    _, a_idx = np.unique(a, return_inverse=True)
    _, b_idx = np.unique(b, return_inverse=True)
    n = a.shape[0]
    cont = np.zeros((a_idx.max() + 1, b_idx.max() + 1), dtype=np.int64)
    np.add.at(cont, (a_idx, b_idx), 1)

    def _comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return (x * (x - 1.0) / 2.0).sum()

    sum_comb = _comb2(cont.ravel())
    sum_comb_a = _comb2(cont.sum(axis=1))
    sum_comb_b = _comb2(cont.sum(axis=0))
    comb_n = n * (n - 1.0) / 2.0
    expected = sum_comb_a * sum_comb_b / comb_n if comb_n else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        return 1.0
    return float((sum_comb - expected) / (max_index - expected))


# --------------------------------------------------------------------------- #
# Gold fixture I/O
# --------------------------------------------------------------------------- #
def gold_paths(name: str):
    """The ``(.npz, .json)`` paths for a gold fixture ``name``."""
    return HERE / f"{name}_gold.npz", HERE / f"{name}_gold.json"


def save_gold(name: str, arrays: dict, meta: dict) -> None:
    """Write ``parity/<name>_gold.npz`` (the arrays) and ``parity/<name>_gold.json``
    (a human-readable provenance log).

    ``arrays`` is saved verbatim with ``np.savez``. ``meta`` is a JSON-serializable
    dict; it is the committed provenance the maintainer asked for — reference name
    and version, seed(s), config/formula, K, the corpus identifier, an ISO date
    string, the reference's own seed-to-seed noise floor, and the metric summary.
    """
    npz_path, json_path = gold_paths(name)
    np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in arrays.items()})
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def load_gold(name: str):
    """Load a gold fixture. Returns ``(arrays_dict, meta_dict)``.

    ``arrays_dict`` is a plain dict of numpy arrays (npz materialized).
    ``meta_dict`` is the parsed JSON log, or ``{}`` if the ``.json`` is absent.
    """
    npz_path, json_path = gold_paths(name)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"gold fixture {npz_path} not found; regenerate it with "
            f"`python parity/{name}_gold.py --regenerate` (needs R + stm)"
        )
    with np.load(npz_path, allow_pickle=True) as g:
        arrays = {k: g[k] for k in g.files}
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    return arrays, meta
