"""Parity: topica.embedding_regression vs the R ``conText`` package.

Runs conText's own bundled example data (``cr_sample_corpus``, ``cr_glove_subset``,
``cr_transform``) through both R ``conText`` and topica and checks that the à la
carte document embeddings (``dem``), the squared coefficient norm, and the
HC1-deflated norm agree. This is the reference validation that makes
``topica.embedding_regression`` a drop-in replacement for conText, not a
reimplementation.

Skips cleanly when Rscript, ``conText`` or ``quanteda`` are unavailable (CI does not
install R), exactly like the other ``parity/*_compare.py`` scripts.

Run directly (``python parity/embedding_regression_context.py``) for a verbose
report, or under pytest.
"""

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import topica

# conText's reported values on this fixture (immigration ~ party), for reference
# when R is unavailable; the live R run below is authoritative when present.
_CONTEXT_SQUARED = 9.858888
_CONTEXT_DEFLATED = 8.398819
_CONTEXT_N_INSTANCES = 924


def _run_reference(tmp):
    """Export conText's bundled fixtures and reference outputs. Returns the tmp dir,
    or None if Rscript/conText/quanteda are unavailable."""
    if shutil.which("Rscript") is None:
        return None
    rscript = f"""
    ok <- suppressWarnings(suppressMessages(require(conText))) &&
          suppressWarnings(suppressMessages(require(quanteda)))
    if (!ok) quit(status = 42)
    outdir <- "{tmp}"
    data("cr_sample_corpus"); data("cr_glove_subset"); data("cr_transform")
    toks <- tokens(cr_sample_corpus)
    writeLines(as.character(cr_sample_corpus), file.path(outdir, "texts.txt"))
    write.csv(docvars(cr_sample_corpus), file.path(outdir, "docvars.csv"), row.names=FALSE)
    write.csv(cr_glove_subset, file.path(outdir, "glove.csv"))
    write.csv(cr_transform, file.path(outdir, "transform.csv"), row.names=FALSE)
    ic <- tokens_context(x = toks, pattern = "immigration", window = 6L, verbose = FALSE)
    idem <- dem(x = dfm(ic), pre_trained = cr_glove_subset, transform = TRUE,
                transform_matrix = cr_transform, verbose = FALSE)
    write.csv(as.matrix(idem), file.path(outdir, "ref_dem.csv"), row.names=FALSE)
    con <- file(file.path(outdir, "ctx_tokens.txt"), "w")
    for (ctx in as.list(ic)) writeLines(paste(ctx, collapse=" "), con)
    close(con)
    set.seed(2021)
    m <- conText(formula = immigration ~ party, data = toks, pre_trained = cr_glove_subset,
                 transform = TRUE, transform_matrix = cr_transform, jackknife = FALSE,
                 permute = TRUE, num_permutations = 10, window = 6L,
                 case_insensitive = TRUE, verbose = FALSE)
    write.csv(m@normed_coefficients, file.path(outdir, "ref_norms.csv"), row.names=FALSE)
    """
    res = subprocess.run(["Rscript", "-e", rscript], capture_output=True, text=True)
    if res.returncode == 42 or not os.path.exists(os.path.join(tmp, "ref_norms.csv")):
        return None
    res.check_returncode()
    return tmp


def _load_glove(tmp):
    words, rows = [], []
    with open(os.path.join(tmp, "glove.csv")) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            words.append(row[0].strip('"'))
            rows.append([float(x) for x in row[1:]])
    return words, np.array(rows)


def _fixture():
    tmp = tempfile.mkdtemp(prefix="ctext_parity_")
    if _run_reference(tmp) is None:
        pytest.skip("Rscript / conText / quanteda not available for embedding-regression parity")
    return tmp


def test_alc_embeddings_match_context_dem():
    """topica's ALC embedding equals conText's dem() row-for-row (bit-exact)."""
    tmp = _fixture()
    words, G = _load_glove(tmp)
    A = np.loadtxt(os.path.join(tmp, "transform.csv"), delimiter=",", skiprows=1)
    ref = np.loadtxt(os.path.join(tmp, "ref_dem.csv"), delimiter=",", skiprows=1)
    ctx = [line.split() for line in open(os.path.join(tmp, "ctx_tokens.txt")).read().splitlines()]
    # conText's transform is the transpose of topica's convention.
    Y, _ = topica.alc_embeddings(ctx, (G, words), transform=A.T, target=None)
    assert Y.shape == ref.shape
    assert np.max(np.abs(Y - ref)) < 1e-9


def test_regression_norms_match_context():
    """topica reproduces conText's squared and HC1-deflated coefficient norms."""
    tmp = _fixture()
    words, G = _load_glove(tmp)
    A = np.loadtxt(os.path.join(tmp, "transform.csv"), delimiter=",", skiprows=1)
    texts = open(os.path.join(tmp, "texts.txt")).read().splitlines()
    docvars = list(csv.DictReader(open(os.path.join(tmp, "docvars.csv"))))
    party = np.array([d["party"] for d in docvars])
    docs = [t.split() for t in texts]

    ref = list(csv.DictReader(open(os.path.join(tmp, "ref_norms.csv"))))[0]
    ref_sq = float(ref["normed.estimate.orig"])
    ref_defl = float(ref["normed.estimate.deflated"])

    for stat, ref_val in [("squared", ref_sq), ("squared_deflated", ref_defl)]:
        r = topica.embedding_regression(
            docs, party, (G, words), names=["party"], transform=A.T,
            target="immigration", window=6, aggregate="instance",
            statistic=stat, permutations=0, bootstrap=0)
        assert r.n_obs == _CONTEXT_N_INSTANCES
        assert abs(r.normed_estimate[0] - ref_val) < 1e-4, (stat, r.normed_estimate[0], ref_val)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="ctext_parity_")
    if _run_reference(tmp) is None:
        print("SKIP: Rscript / conText / quanteda unavailable")
        raise SystemExit(0)
    words, G = _load_glove(tmp)
    A = np.loadtxt(os.path.join(tmp, "transform.csv"), delimiter=",", skiprows=1)
    ref = np.loadtxt(os.path.join(tmp, "ref_dem.csv"), delimiter=",", skiprows=1)
    ctx = [line.split() for line in open(os.path.join(tmp, "ctx_tokens.txt")).read().splitlines()]
    Y, _ = topica.alc_embeddings(ctx, (G, words), transform=A.T, target=None)
    print(f"ALC dem max|diff| = {np.max(np.abs(Y - ref)):.2e}  (n={len(ctx)})")

    texts = open(os.path.join(tmp, "texts.txt")).read().splitlines()
    party = np.array([d["party"] for d in csv.DictReader(open(os.path.join(tmp, "docvars.csv")))])
    docs = [t.split() for t in texts]
    refn = list(csv.DictReader(open(os.path.join(tmp, "ref_norms.csv"))))[0]
    for stat, key in [("squared", "normed.estimate.orig"), ("squared_deflated", "normed.estimate.deflated")]:
        r = topica.embedding_regression(docs, party, (G, words), names=["party"], transform=A.T,
                                        target="immigration", window=6, aggregate="instance",
                                        statistic=stat, permutations=0, bootstrap=0)
        print(f"{stat:16s} topica={r.normed_estimate[0]:.6f}  conText={float(refn[key]):.6f}")
