"""Committed same-algorithm gold for topica's collapsed-Gibbs RTM (#424).

Unlike the *variational* RTM (validated against a NumPy reimplementation of the
paper's equations, since R lda's `rtm.em` is not variational), the collapsed-Gibbs
backend can be checked against **the author's own implementation on its own terms**:
R lda's `rtm.em` wraps `rtm.collapsed.gibbs.sampler`, the same algorithm topica's
`fit_rtm_gibbs` implements. So R lda is an authoritative oracle here, not a
directional baseline.

We freeze, from a fixed planted document network, R lda's seeded topic-word phi at
two seeds (its own seed-to-seed self-consistency floor). The offline test refits
topica (`inference="gibbs"`) on the same corpus with matching hyperparameters and
asserts its aligned topic-word cosine clears R's floor (minus a margin). Runs in CI
WITHOUT Rscript (the reference is frozen in the committed npz/json).

    python parity/rtm_gibbs_gold.py --regenerate   # needs Rscript + the R `lda` package
    python parity/rtm_gibbs_gold.py                # offline compare against the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "rtm_gibbs"

K = 3
BLOCK = 8
NUM_DOCS = 90
DOCLEN = 40
CORPUS_SEED = 0
ALPHA = 0.1
BETA = 0.1  # R lda's `eta` (topic-word smoothing)
M_ITERS = 25  # rtm.em num.m.iterations
E_SWEEPS = 8  # num.e.iterations (Gibbs sweeps per E-step)
MARGIN = 0.05


def _planted():
    """A planted document network: K word-blocks, each doc drawn mostly from one
    block, links dense within a latent group and sparse across. Deterministic, so
    the offline refit reconstructs the exact corpus the gold was built from."""
    rng = np.random.default_rng(CORPUS_SEED)
    v = K * BLOCK
    groups = [i % K for i in range(NUM_DOCS)]
    docs = []
    for g in groups:
        docs.append([
            str(g * BLOCK + int(rng.integers(BLOCK))) if rng.random() < 0.85
            else str(int(rng.integers(v)))
            for _ in range(DOCLEN)
        ])
    edges = []
    for i in range(NUM_DOCS):
        for j in range(i + 1, NUM_DOCS):
            if rng.random() < (0.35 if groups[i] == groups[j] else 0.01):
                edges.append((i, j))
    vocab = [str(x) for x in range(v)]
    return docs, edges, vocab, groups


def _driver() -> str:
    return f"""
suppressMessages(library(lda))
lines <- readLines(file.path(dir, "docs.txt"))
vocab <- readLines(file.path(dir, "vocab.txt"))
docs <- lexicalize(lines, vocab = vocab)
edges <- read.csv(file.path(dir, "edges.csv"))
D <- length(docs)
links <- vector("list", D); for (d in 1:D) links[[d]] <- integer(0)
if (nrow(edges) > 0) for (r in 1:nrow(edges)) {{
  i <- edges$i[r] + 1; j <- edges$j[r] + 1
  links[[i]] <- c(links[[i]], edges$j[r]); links[[j]] <- c(links[[j]], edges$i[r])
}}
phi <- function(m) {{ tw <- m$topics + {BETA}; tw / rowSums(tw) }}
fit_one <- function(s) {{
  set.seed(s)
  rtm.em(docs, links, K = {K}, vocab = vocab, num.e.iterations = {E_SWEEPS},
         num.m.iterations = {M_ITERS}, alpha = {ALPHA}, eta = {BETA})
}}
m1 <- fit_one(1); m2 <- fit_one(2)
write.csv(phi(m1), file.path(dir, "phi1.csv"), row.names = FALSE)
write.csv(phi(m2), file.path(dir, "phi2.csv"), row.names = FALSE)
writeLines(as.character(m1$beta), file.path(dir, "beta1.txt"))
cat("ok\\n")
"""


def _edge_csv(edges) -> str:
    return "i,j\n" + "\n".join(f"{i},{j}" for i, j in edges) + "\n"


def _fit_topica(docs, edges, vocab):
    import topica

    m = topica.RTM(K, link="exponential", inference="gibbs", alpha=ALPHA, beta=BETA, seed=1)
    m.fit(docs, edges, iters=M_ITERS, e_sweeps=E_SWEEPS)
    tv = {w: i for i, w in enumerate(m.vocabulary)}
    tw = np.asarray(m.topic_word)
    out = np.zeros((K, len(vocab)))
    for j, w in enumerate(vocab):
        if w in tv:
            out[:, j] = tw[:, tv[w]]
    return out, np.asarray(m.eta)


def _r_version() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["Rscript", "-e", 'cat(as.character(packageVersion("lda")))'],
            capture_output=True, text=True, timeout=60,
        )
        return f"R lda {out.stdout.strip()}"
    except Exception:
        return "R lda (version unknown)"


def regenerate() -> None:
    if not harness.r_available("lda"):
        raise SystemExit("regenerate needs Rscript with the R `lda` package")

    docs, edges, vocab, _groups = _planted()
    out = harness.run_rscript(
        _driver(),
        {
            "docs.txt": "\n".join(" ".join(d) for d in docs) + "\n",
            "vocab.txt": "\n".join(vocab) + "\n",
            "edges.csv": _edge_csv(edges),
        },
        ["phi1.csv", "phi2.csv", "beta1.txt"],
        timeout=1800,
    )
    phi1 = harness.read_r_beta_csv(out["phi1.csv"], vocab)
    phi2 = harness.read_r_beta_csv(out["phi2.csv"], vocab)
    r_self, _ = harness.align_cosine(phi1, phi2)
    r_beta = np.array([float(x) for x in out["beta1.txt"].split()])

    topica_phi, topica_eta = _fit_topica(docs, edges, vocab)
    topica_cos, _ = harness.align_cosine(phi1, topica_phi)

    harness.save_gold(
        NAME,
        arrays={
            "phi1": phi1,
            "vocab": np.array(vocab, dtype=object),
            "r_beta": r_beta,
        },
        meta={
            "reference": _r_version(),
            "model": "RTM collapsed Gibbs (R lda rtm.em / rtm.collapsed.gibbs.sampler)",
            "corpus": f"planted network ({NUM_DOCS} docs, K={K}, {len(edges)} links, seed {CORPUS_SEED})",
            "num_topics": K,
            "alpha": ALPHA,
            "beta": BETA,
            "m_iters": M_ITERS,
            "e_sweeps": E_SWEEPS,
            "vocab_size": len(vocab),
            "num_docs": len(docs),
            "num_links": len(edges),
            "margin": MARGIN,
            "topic_r_self_cosine": r_self,
            "topica_topic_cosine": topica_cos,
            "r_link_beta_mean": float(r_beta.mean()),
            "topica_link_beta_mean": float(topica_eta.mean()),
            "date": datetime.date.today().isoformat(),
            "pass_bar": "topica gibbs topic-word cosine >= topic_r_self_cosine - margin",
        },
    )
    print(f"regenerated {NAME} gold:")
    print(f"  R self topic cosine {r_self:.4f}  topica {topica_cos:.4f}  "
          f"(bar {r_self - MARGIN:.4f})")


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    phi1 = arrays["phi1"]
    r_beta = arrays["r_beta"]
    vocab = [str(w) for w in arrays["vocab"]]
    r_self = float(meta["topic_r_self_cosine"])
    margin = float(meta["margin"])
    bar = r_self - margin

    docs, edges, live_vocab, _groups = _planted()
    assert live_vocab == vocab, "planted corpus vocabulary drifted from the committed gold"
    topica_phi, eta = _fit_topica(docs, edges, vocab)
    cos, _ = harness.align_cosine(phi1, topica_phi)

    # Relational-M-step coverage: the link coefficient β = ln(p_k) is negative in
    # both, and (permutation-invariantly) its mean should match R closely — a
    # planted-topic LDA-like fit would not reproduce R's β regime.
    beta_mean_gap = abs(float(eta.mean()) - float(r_beta.mean()))
    result = {
        "topic_cosine": cos,
        "topic_r_self_cosine": r_self,
        "bar": bar,
        "margin_over_bar": cos - bar,
        "link_beta_all_negative": bool((eta < 0).all()),
        "r_link_beta_mean": float(r_beta.mean()),
        "topica_link_beta_mean": float(eta.mean()),
        "link_beta_mean_gap": beta_mean_gap,
        "passes": bool(cos >= bar and (eta < 0).all() and beta_mean_gap < 0.02),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print("same-algorithm topic-word phi (topica gibbs vs R lda rtm.em):")
        print(f"  topica {cos:.4f}  (R self {r_self:.4f}, bar {bar:.4f})")
        print(f"  link β mean: topica {result['topica_link_beta_mean']:+.4f}  "
              f"R {result['r_link_beta_mean']:+.4f}  (gap {beta_mean_gap:.4f})")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
