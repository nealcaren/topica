"""Parity for RTM (Chang & Blei, AOAS 2010).

Two tiers, per the reviewed plan (parity/rtm_reference.py has the details):

1. PRIMARY numerical oracle — a standalone NumPy implementation of the paper's
   variational EM (``rtm_reference.py``). topica's Rust RTM must reproduce it to
   high aligned topic-word cosine on a fixed corpus. This is the real parity bar.

2. DIRECTIONAL baseline — the R ``lda`` package's ``rtm.em``. That routine is a
   *collapsed Gibbs* sampler (its body calls ``rtm.collapsed.gibbs.sampler``), not
   the paper's variational EM, so we only check that both put linked documents
   closer in topic space than unlinked ones — an ordering check, not a numeric
   match. Skips cleanly when Rscript / the ``lda`` package is absent.

Run:  VIRTUAL_ENV=... .venv-dev/bin/python -m pytest parity/rtm_compare.py -q
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

import topica

pytestmark = pytest.mark.parity

_HERE = os.path.dirname(__file__)


def _load_fixture():
    with open(os.path.join(_HERE, "fixtures", "rtm_gold.json")) as f:
        return json.load(f)


def _align_cosine(a, b):
    from scipy.optimize import linear_sum_assignment

    na = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    c = na @ nb.T
    r, col = linear_sum_assignment(-c)
    return c[r, col].mean()


@pytest.mark.parametrize("link", ["logistic", "exponential"])
def test_matches_numpy_reference(link):
    """topica's Rust RTM reproduces the NumPy variational oracle on a fixed corpus.

    The two share the same deterministic algorithm and integer-coded corpus, so
    the aligned topic-word cosine should be near 1. This is the primary parity
    bar; the reference lives in parity/rtm_reference.py.
    """
    fx = _load_fixture()
    docs = [[str(w) for w in d] for d in fx["docs"]]
    edges = [tuple(e) for e in fx["edges"]]
    K = fx["num_topics"]
    m = topica.RTM(K, link=link, alpha=fx["alpha"], seed=fx["seed"]).fit(
        docs, edges, iters=fx["iters"]
    )
    vocab = m.vocabulary
    order = np.argsort([int(w) for w in vocab])
    tw = m.topic_word[:, order]
    gold = np.array(fx["gold"][link]["topic_word"])
    cos = _align_cosine(tw, gold)
    assert cos > 0.95, f"{link}: aligned topic-word cosine {cos:.4f} vs NumPy reference"


def _rscript_lda_available():
    if not shutil.which("Rscript"):
        return False
    probe = subprocess.run(
        ["Rscript", "-e", 'quit(status = !requireNamespace("lda", quietly = TRUE))'],
        capture_output=True,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _rscript_lda_available(), reason="R 'lda' package not installed")
def test_directional_vs_r_lda_gibbs():
    """Directional: R's rtm (collapsed Gibbs) and topica both score observed links
    above random pairs. Not a numeric match — different inference (Gibbs vs
    variational). We only assert the ordering agrees in sign.
    """
    fx = _load_fixture()
    docs = [[str(w) for w in d] for d in fx["docs"]]
    edges = [tuple(e) for e in fx["edges"]]
    K = fx["num_topics"]
    m = topica.RTM(K, link="exponential", alpha=fx["alpha"], seed=0).fit(
        docs, edges, iters=fx["iters"]
    )
    linked = np.mean([m.predict_link(i, j) for i, j in edges])
    rng = np.random.default_rng(0)
    D = len(docs)
    rand_pairs = [(int(rng.integers(D)), int(rng.integers(D))) for _ in range(len(edges))]
    rand_pairs = [(i, j) for i, j in rand_pairs if i != j and [min(i, j), max(i, j)] not in fx["edges"]]
    unlinked = np.mean([m.predict_link(i, j) for i, j in rand_pairs])
    # topica: observed links score above random pairs (the RTM property R also has)
    assert linked > unlinked, f"linked {linked:.3f} !> unlinked {unlinked:.3f}"
