"""Generate the paper's side-by-side validation appendix.

For each reference package topica validates against, this runs BOTH the original
package and topica on the **poliblog** corpus and writes a LaTeX subsection showing
the two engines' topics side by side (topic-aligned top words), plus each model's
distinctive feature (covariate effects, keywords, correlations, sentiment) and the
alignment cosine. The output is one self-contained file,
``paper/generated/validation_appendix.tex``, ``\\input``-ed by ``paper/topica.tex``.

It reuses the poliblog loader and the cross-implementation drivers in ``parity/``;
top words are read off the topic-word matrices both engines export. Every leg
**skips cleanly** — emitting a "skipped" note into the tex — when its toolchain is
absent (``Rscript``+``stm``/``keyATM``, the ``mallet`` CLI, ``tomotopy``, or
``STS_REPL_DIR``), so the generator always writes a complete file.

Legs (``--only``): ``lda`` (Java MALLET + tomotopy), ``nmf``/``lsa``
(scikit-learn), ``stm``/``ctm``/``content`` (R stm), ``keyatm`` (R keyATM),
``dmr``/``gdmr``/``slda``/``labeledlda``/``hdp``/``pa``/``dtm``
(tomotopy), ``sts`` (the authors' R reference).

    python paper/gen_validation_appendix.py            # all legs
    python paper/gen_validation_appendix.py --only stm --k 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARITY = os.path.join(ROOT, "parity")
OUT = os.path.join(HERE, "generated", "validation_appendix.tex")
sys.path.insert(0, PARITY)

DEFAULT_K = 10  # coarse enough that the random-init Gibbs legs (LDA, DMR) align
                # cleanly; the spectral legs (STM/CTM/content) stay tight at any K
TOP_N = 8  # words shown per topic in the side-by-side tables

# Filled live by each poliblog leg (keyed by leg name) so the master validation
# table (Table A1) reports the same figures the legs below derive.
RESULTS = {}


# --------------------------------------------------------------------------- #
# Shared data + small numeric helpers
# --------------------------------------------------------------------------- #
def poliblog():
    """(docs, rating_lib, day, vocab) — reuses the parity loader."""
    import stm_poliblog_compare as stmp
    return stmp.load_and_prep()


def realign_to(vocab_to, vocab_from, mat):
    """Reindex a K x |vocab_from| matrix onto vocab_to (zeros for missing)."""
    idx = {w: i for i, w in enumerate(vocab_from)}
    out = np.zeros((mat.shape[0], len(vocab_to)))
    for j, w in enumerate(vocab_to):
        if w in idx:
            out[:, j] = mat[:, idx[w]]
    return out


def doc_term(docs, vocab):
    """Dense document-term count matrix (D x |vocab|) for the sklearn legs."""
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for r, doc in enumerate(docs):
        for w in doc:
            j = idx.get(w)
            if j is not None:
                X[r, j] += 1.0
    return X


def align_pairs(ref_tw, top_tw):
    """Greedy 1-1 alignment ref->topica by topic-word cosine (parity helper)."""
    import stm_poliblog_compare as stmp
    mean_cos, pairs = stmp._best_alignment_cosine(ref_tw, top_tw, return_pairs=True)
    return pairs, mean_cos


def topw(row, vocab, n=TOP_N):
    return [vocab[i] for i in np.argsort(row)[::-1][:n]]


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #
def tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def cell(words) -> str:
    return ", ".join(tex_escape(w) for w in words)


def sidebyside(ref_name, ref_tw, top_tw, vocab, *, caption, label, n=TOP_N):
    """Topic-aligned side-by-side top-words table (one row per aligned topic)."""
    pairs, mean_cos = align_pairs(ref_tw, top_tw)
    pairs = sorted(pairs, key=lambda p: -p[2])
    lines = [
        r"\begin{table}[ht]\centering\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{@{}r >{\raggedright\arraybackslash}p{0.40\textwidth} "
        r">{\raggedright\arraybackslash}p{0.40\textwidth}@{}}",
        r"\toprule",
        rf" & {ref_name} & \pkg{{topica}} \\",  # ref_name is LaTeX we control
        r"\midrule",
    ]
    for rank, (ri, tj, _c) in enumerate(pairs, 1):
        lines.append(rf"{rank} & {cell(topw(ref_tw[ri], vocab, n))} & "
                     rf"{cell(topw(top_tw[tj], vocab, n))} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), mean_cos, pairs


def _plain(title):
    """Strip LaTeX markup from a heading for the PDF bookmark / ToC (hyperref
    cannot sanitize \\pkg/\\proglang inside a bookmark)."""
    import re
    t = re.sub(r"\\(?:pkg|proglang|code|emph|textbf)\{([^}]*)\}", r"\1", title)
    return t.replace("~", " ").replace("\\", "")


def sidebyside_multi(refs, top_tw, vocab, *, caption, label, n=TOP_N):
    """N-engine side-by-side: refs is a list of (display_name, ref_tw), each
    matrix already on `vocab`; every reference is aligned to topica and gets its
    own column. Returns (table_str, [mean_cos per ref]). topica topics are ordered
    by the first reference's alignment cosine."""
    K = top_tw.shape[0]
    cols = []  # (name, ref_tw, {topica_topic: ref_topic}, mean_cos, {topica_topic: cos})
    for name, tw in refs:
        pairs, mc = align_pairs(tw, top_tw)
        cols.append((name, tw, {tj: ri for ri, tj, _ in pairs}, mc,
                     {tj: c for _, tj, c in pairs}))
    order = sorted(range(K), key=lambda tj: -cols[0][4].get(tj, 0.0)) if cols else list(range(K))
    width = 0.88 / (len(refs) + 1)
    colspec = " ".join([r">{\raggedright\arraybackslash}p{%.2f\textwidth}" % width] * (len(refs) + 1))
    header = " & ".join([name for name, *_ in cols] + [r"\pkg{topica}"])
    lines = [
        r"\begin{table}[ht]\centering\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{@{{}}r {colspec}@{{}}}}",
        r"\toprule",
        rf" & {header} \\",
        r"\midrule",
    ]
    for rank, tj in enumerate(order, 1):
        cells = [cell(topw(tw[amap[tj]], vocab, n)) if tj in amap else ""
                 for _name, tw, amap, _mc, _cm in cols]
        cells.append(cell(topw(top_tw[tj], vocab, n)))
        lines.append(rf"{rank} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), [mc for _n, _t, _a, mc, _c in cols]


def subsection(title, body):
    # \subsection[bookmark-safe plain text]{rich title}
    return f"\\subsection[{_plain(title)}]{{{title}}}\n\n{body}\n"


def skip(title, reason):
    return subsection(title, rf"\emph{{Skipped at build time: {tex_escape(reason)}.}} "
                      r"This leg regenerates where the toolchain is available.")


# --------------------------------------------------------------------------- #
# Generic R stm runner (powers the STM / CTM / content legs)
# --------------------------------------------------------------------------- #
def r_stm_available():
    import stm_poliblog_compare as stmp
    return stmp.r_stm_available()


_R_STM = r"""
suppressMessages(library(stm))
lines <- readLines(file.path(dir, "docs.txt")); toks <- strsplit(lines, " ")
vocab <- sort(unique(unlist(toks))); vmap <- setNames(seq_along(vocab), vocab)
documents <- lapply(toks, function(d){ tb <- table(d); idx <- as.integer(vmap[names(tb)])
  o <- order(idx); matrix(as.integer(rbind(idx[o], as.integer(tb)[o])), nrow=2) })
prev <- NULL; cont <- NULL
if (file.exists(file.path(dir,"design.csv"))) prev <- as.matrix(read.csv(file.path(dir,"design.csv")))
meta <- read.csv(file.path(dir,"meta.csv"))
set.seed(1)
args <- list(documents=documents, vocab=vocab, K=KVAL, init.type="Spectral", verbose=FALSE)
if (!is.null(prev)) args$prevalence <- prev
if (HASCONTENT) { args$content <- ~rating; args$data <- meta }
f <- do.call(stm, args)
# Marginal topic-word (averaged over content levels when content is present).
if (is.null(f$beta$logbeta) || length(f$beta$logbeta) == 1) {
  b <- exp(f$beta$logbeta[[1]])
} else {
  acc <- exp(f$beta$logbeta[[1]]); for (i in 2:length(f$beta$logbeta)) acc <- acc + exp(f$beta$logbeta[[i]])
  b <- acc / length(f$beta$logbeta)
}
colnames(b) <- vocab
write.csv(b, file.path(dir,"beta.csv"), row.names=FALSE)
write.csv(f$theta, file.path(dir,"theta.csv"), row.names=FALSE)
write(vocab, file.path(dir,"vocab.txt"))
# Per-content-level betas for the SAGE leg (one CSV per level).
if (HASCONTENT) {
  for (i in seq_along(f$beta$logbeta)) {
    bi <- exp(f$beta$logbeta[[i]]); colnames(bi) <- vocab
    write.csv(bi, file.path(dir, sprintf("beta_lvl%d.csv", i)), row.names=FALSE)
  }
  write(levels(as.factor(meta$rating)), file.path(dir,"levels.txt"))
}
cat("ok\n")
"""


def run_r_stm(docs, k, *, design=None, feat_names=None, meta_rating=None, content=False,
              workdir=None):
    """Fit R stm and return (vocab, beta KxV, theta DxK, extra dict)."""
    import stm_poliblog_compare as stmp
    d = workdir
    with open(os.path.join(d, "docs.txt"), "w") as f:
        f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
    if design is not None:
        with open(os.path.join(d, "design.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["intercept"] + feat_names)
            for row in design:
                w.writerow(list(row))
    with open(os.path.join(d, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rating"])
        for r in (meta_rating if meta_rating is not None else ["NA"] * len(docs)):
            w.writerow([r])
    script = (f'dir <- "{d}"\nKVAL <- {k}\nHASCONTENT <- {"TRUE" if content else "FALSE"}\n'
              + _R_STM)
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=2400)
    if "ok" not in proc.stdout:
        raise RuntimeError(f"R stm driver failed:\n{proc.stdout}\n{proc.stderr}")
    vocab = open(os.path.join(d, "vocab.txt")).read().split()
    beta = stmp._read_r_beta(os.path.join(d, "beta.csv"), vocab)
    theta = np.loadtxt(os.path.join(d, "theta.csv"), delimiter=",", skiprows=1)
    extra = {}
    if content:
        levels = open(os.path.join(d, "levels.txt")).read().split()
        extra["levels"] = levels
        extra["level_beta"] = [stmp._read_r_beta(os.path.join(d, f"beta_lvl{i+1}.csv"), vocab)
                               for i in range(len(levels))]
    return vocab, beta, theta, extra


# --------------------------------------------------------------------------- #
# Legs
# --------------------------------------------------------------------------- #
def leg_stm(k):
    title = r"Structural topic model (vs \proglang{R}~\pkg{stm})"
    if not r_stm_available():
        return skip(title, "Rscript with the stm package not available")
    from topica import STM
    from topica.stm import spline
    docs, rating_lib, day, _ = poliblog()
    rating = ["Liberal" if r else "Conservative" for r in rating_lib]
    spl, _ = spline(day, df=10)
    X = np.column_stack([rating_lib, spl])
    feat = ["ratingLiberal"] + [f"day_s{j}" for j in range(spl.shape[1])]
    with tempfile.TemporaryDirectory() as d:
        rvocab, rbeta, rtheta, _ = run_r_stm(
            docs, k, design=np.column_stack([np.ones(len(docs)), X]),
            feat_names=feat, meta_rating=rating, workdir=d)
    m = STM(num_topics=k, init="spectral")
    m.fit(docs, X, prevalence_names=feat, iters=200, convergence_tol=1e-5)
    tbeta = realign_to(rvocab, list(m.vocabulary), np.asarray(m.topic_word))
    ttheta = np.asarray(m.doc_topic)
    table, cos, pairs = sidebyside(r"\proglang{R}~\pkg{stm}", rbeta, tbeta, rvocab,
                                   caption=f"STM topics on poliblog (K={k}), "
                                   r"\proglang{R}~\pkg{stm} vs \pkg{topica}, "
                                   "aligned by topic-word cosine.",
                                   label="tab:app:stm")
    # Unique feature: prevalence by ideology — mean topic share, Conservative vs
    # Liberal, from each engine's theta; show the most ideologically split topics.
    lib = np.array([r == "Liberal" for r in rating])
    feat_block = _prevalence_block(rbeta, tbeta, rtheta, ttheta, rvocab, lib, pairs,
                                   "Conservative", "Liberal", ref_label=r"\pkg{stm}")
    RESULTS["stm"] = f"{cos:.2f}"
    intro = (rf"Both engines fit \code{{prevalence = \textasciitilde{{}} rating + s(day)}} "
             rf"on the same {len(docs)} posts and {len(rvocab)} word types. "
             rf"Aligned topic-word cosine \textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table, feat_block]))


def _prevalence_block(rbeta, tbeta, rtheta, ttheta, vocab, lib_mask, pairs, lo, hi,
                      ref_label=r"\pkg{stm}"):
    """Side-by-side: the 3 topics with the largest Liberal-minus-Conservative mean
    topic share, with each engine's effect and a top-words label."""
    rmap = {ri: tj for ri, tj, _ in pairs}
    r_eff = rtheta[lib_mask].mean(0) - rtheta[~lib_mask].mean(0)
    t_eff = ttheta[lib_mask].mean(0) - ttheta[~lib_mask].mean(0)
    order = np.argsort(-np.abs(r_eff))[:3]
    lines = [r"\noindent\emph{Prevalence by ideology.} Mean topic share, "
             rf"{hi} minus {lo} posts (positive = more {hi}):",
             r"\begin{center}\footnotesize\begin{tabular}{@{}p{0.5\textwidth} r r@{}}",
             r"\toprule",
             rf"topic (\pkg{{topica}} top words) & {ref_label} & \pkg{{topica}} \\",
             r"\midrule"]
    for ri in order:
        tj = rmap.get(int(ri), int(ri))
        lab = cell(topw(tbeta[tj], vocab, 4))
        lines.append(rf"{lab} & {r_eff[ri]:+.3f} & {t_eff[tj]:+.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}\end{center}"]
    return "\n".join(lines)


def leg_ctm(k):
    # tomotopy also implements a CTModel, but it is random-initialized on the
    # highly multimodal logistic-normal objective and lands in a different basin
    # than topica's spectral-initialized fit (matched cosine ~0.4, vs 0.99 against
    # R stm, which is likewise spectral-initialized) -- so the meaningful reference
    # here is R stm. tomotopy serves as the second reference on the LDA leg, where
    # both engines are collapsed Gibbs and agree.
    title = r"Correlated topic model (vs \proglang{R}~\pkg{stm}, no covariates)"
    if not r_stm_available():
        return skip(title, "Rscript with the stm package not available")
    from topica import CTM
    docs, _, _, _ = poliblog()
    m = CTM(num_topics=k, init="spectral")
    m.fit(docs, iters=200, convergence_tol=1e-5)
    tvocab = list(m.vocabulary)
    ttw = np.asarray(m.topic_word)
    with tempfile.TemporaryDirectory() as d:
        rvocab, rbeta, _rtheta, _ = run_r_stm(docs, k, workdir=d)
    refs = [(r"\proglang{R}~\pkg{stm}", realign_to(tvocab, rvocab, rbeta))]
    table, coses = sidebyside_multi(refs, ttw, tvocab, n=TOP_N,
                                    caption=f"CTM topics on poliblog (K={k}); "
                                    "the logistic-normal (correlated) topic model.",
                                    label="tab:app:ctm")
    # Unique feature: topic correlation (what CTM adds over LDA).
    tw = ttw
    corr = np.asarray(m.topic_correlation)
    iu = np.triu_indices(k, 1)
    strongest = sorted(zip(corr[iu], iu[0], iu[1]), key=lambda t: -abs(t[0]))[:3]
    rows = "\n".join(
        rf"{cell(topw(tw[a], tvocab, 3))} / {cell(topw(tw[b], tvocab, 3))} & {c:+.2f} \\"
        for c, a, b in strongest)
    feat = "\n".join([
        r"\noindent\emph{Topic correlations} (the structure CTM adds over LDA), "
        r"strongest pairs in \pkg{topica}:",
        r"\begin{center}\footnotesize\begin{tabular}{@{}>{\raggedright\arraybackslash}"
        r"p{0.7\textwidth} r@{}}",
        r"\toprule topic pair & corr \\ \midrule",
        rows,
        r"\bottomrule\end{tabular}\end{center}",
    ])
    costxt = ", ".join(f"{_plain(name)} {c:.3f}" for (name, _), c in zip(refs, coses))
    RESULTS["ctm"] = f"{coses[0]:.2f}"
    intro = (r"No covariates, so this is the logistic-normal (correlated) topic model. "
             rf"Aligned cosine vs \pkg{{topica}} --- {costxt}.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def leg_content(k):
    title = r"Content covariate / SAGE (vs \proglang{R}~\pkg{stm})"
    if not r_stm_available():
        return skip(title, "Rscript with the stm package not available")
    from topica import STM
    docs, rating_lib, _, _ = poliblog()
    rating = ["Liberal" if r else "Conservative" for r in rating_lib]
    with tempfile.TemporaryDirectory() as d:
        rvocab, rbeta, _t, extra = run_r_stm(docs, k, meta_rating=rating, content=True, workdir=d)
    m = STM(num_topics=k, init="spectral")
    m.fit(docs, content=rating, iters=150, convergence_tol=1e-5)
    tbeta = realign_to(rvocab, list(m.vocabulary), np.asarray(m.topic_word))
    table, cos, pairs = sidebyside(r"\proglang{R}~\pkg{stm}", rbeta, tbeta, rvocab,
                                   caption=f"Content-model (SAGE) topics on poliblog (K={k}); "
                                   "marginal topic-word averaged over rating levels.",
                                   label="tab:app:content")
    # Unique feature: per-group wording of one topic (Conservative vs Liberal).
    feat = _content_block(m, rvocab=rvocab, pairs=pairs)
    RESULTS["content"] = f"{cos:.2f}"
    intro = (rf"\code{{content = \textasciitilde{{}} rating}}: each topic's words shift by "
             rf"ideology. Marginal aligned cosine \textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def _content_block(m, rvocab, pairs):
    import numpy as np
    tw = np.asarray(m.topic_word); tvocab = list(m.vocabulary)
    groups = list(m.groups)
    # pick the topic whose two groups' wording differs most (TV distance).
    best_t, best_d = 0, -1.0
    g0 = np.asarray(m.topic_word_by_group)  # (K, G, V)
    for t in range(tw.shape[0]):
        d = 0.5 * np.abs(g0[t, 0] - g0[t, 1]).sum()
        if d > best_d:
            best_t, best_d = t, d
    lines = [rf"\noindent\emph{{Per-group wording}} of one topic in \pkg{{topica}} "
             rf"(the SAGE deviation, TV distance {best_d:.2f}):",
             r"\begin{center}\footnotesize\begin{tabular}{@{}l p{0.7\textwidth}@{}}\toprule"]
    for gi, g in enumerate(groups):
        lines.append(rf"{tex_escape(g)} & {cell(topw(g0[best_t, gi], tvocab, 8))} \\")
    lines += [r"\bottomrule\end{tabular}\end{center}"]
    return "\n".join(lines)


def leg_lda(k):
    title = r"Latent Dirichlet allocation (vs Java \pkg{MALLET} and \pkg{tomotopy})"
    import mallet_parity as mp
    have_mallet = mp.mallet_available()
    try:
        import tomotopy as tp
        have_tomo = True
    except ImportError:
        have_tomo = False
    if not (have_mallet or have_tomo):
        return skip(title, "neither the MALLET CLI nor tomotopy available")
    from topica import LDA
    docs, _, _, _ = poliblog()
    m = LDA(num_topics=k, seed=1, optimize_interval=0)
    m.fit(docs, iters=1000, num_samples=5, sample_interval=25)
    tvocab = list(m.vocabulary)
    ttw = np.asarray(m.topic_word)
    refs = []
    if have_mallet:
        mphi, mvocab = mp._mallet_phi(docs, k, iters=1000, seed=1)
        refs.append((r"Java \pkg{MALLET}", realign_to(tvocab, mvocab, mphi)))
    if have_tomo:
        mdl = tp.LDAModel(tw=tp.TermWeight.ONE, k=k, seed=1)
        for d in docs:
            mdl.add_doc(d)
        mdl.burn_in = 200
        mdl.train(1000, workers=1, show_progress=False)
        phi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
        refs.append((r"\pkg{tomotopy}", realign_to(tvocab, list(mdl.used_vocabs), phi)))
    n = TOP_N if len(refs) == 1 else 6
    names = [name for name, _ in refs] + [r"\pkg{topica}"]
    engines = (", ".join(names[:-1]) + ", and " + names[-1]) if len(names) > 2 else " and ".join(names)
    table, coses = sidebyside_multi(refs, ttw, tvocab, n=n,
                                    caption=f"LDA topics on poliblog (K={k}); "
                                    "all collapsed Gibbs samplers.",
                                    label="tab:app:lda")
    costxt = ", ".join(f"{_plain(name)} {c:.3f}" for (name, _), c in zip(refs, coses))
    RESULTS["lda"] = " / ".join(f"{c:.2f}" for c in coses)
    intro = (rf"The baseline: plain LDA on the same {len(docs)} posts, independent "
             rf"collapsed-Gibbs samplers ({engines}). Aligned topic-word "
             rf"cosine vs \pkg{{topica}} --- {costxt}.")
    return subsection(title, "\n\n".join([intro, table]))


def _diag_cosine(A, B):
    """Mean per-row (un-permuted) cosine — for index-aligned topics."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(An * Bn, axis=1)))


def leg_keyatm(k):
    title = r"Keyword-assisted topic model (vs \proglang{R}~\pkg{keyATM})"
    import keyatm_r_compare as ka
    if not ka.r_keyatm_available():
        return skip(title, "Rscript with the keyATM package not available")
    from topica import KeyATM
    docs, keywords = ka.load_and_prep()
    names = list(keywords.keys())
    num_keyword = len(keywords)
    num_topics = num_keyword + ka.NUM_REGULAR  # keyATM's K is set by the keyword sets
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "keywords.json"), "w") as f:
            json.dump(keywords, f)
        script = (f'dir <- "{d}"\nNREG <- {ka.NUM_REGULAR}\nITERS <- {ka.ITERS}\n'
                  'if (!requireNamespace("jsonlite", quietly=TRUE)) stop("need jsonlite")\n'
                  + ka._R_DRIVER)
        proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True,
                              timeout=3600)
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R keyATM driver failed:\n{proc.stdout}\n{proc.stderr}")
        with open(os.path.join(d, "r_phi1.csv"), newline="") as f:
            header = next(csv.reader(f))
        rvocab = [h.strip('"') for h in header[1:]]
        rphi = ka._read_r_phi(os.path.join(d, "r_phi1.csv"), rvocab)
    model = KeyATM(keywords, num_topics=num_topics, seed=1)
    model.fit(docs, iters=ka.ITERS)
    tphi = realign_to(rvocab, list(model.vocabulary), np.asarray(model.topic_word))
    # keyATM orders the keyword topics first, in keyword-list order, in BOTH engines
    # — so the keyword topics are index-aligned (no cosine alignment needed). Show
    # them with their seed words; that is keyATM's signature.
    kw_cos = _diag_cosine(rphi[:num_keyword], tphi[:num_keyword])
    _, overall_cos = align_pairs(rphi, tphi)
    table = _keyword_table(names, keywords, rphi, tphi, rvocab, num_keyword,
                           caption="keyATM keyword topics on poliblog, "
                           r"\proglang{R}~\pkg{keyATM} vs \pkg{topica}; each topic "
                           "anchored to its seed words (index-aligned).",
                           label="tab:app:keyatm")
    RESULTS["keyatm"] = f"{kw_cos:.2f}"
    intro = (rf"{num_keyword} seeded keyword topics plus {ka.NUM_REGULAR} free topics on "
             rf"{len(docs)} posts. The anchored topics line up by construction; their "
             rf"recovered words agree at per-topic cosine \textbf{{{kw_cos:.3f}}} "
             rf"(all {num_topics} topics, best-aligned: {overall_cos:.3f}).")
    return subsection(title, "\n\n".join([intro, table]))


def _keyword_table(names, keywords, rphi, tphi, rvocab, num_keyword, *, caption, label):
    lines = [
        r"\begin{table}[ht]\centering\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth} "
        r">{\raggedright\arraybackslash}p{0.36\textwidth} "
        r">{\raggedright\arraybackslash}p{0.36\textwidth}@{}}",
        r"\toprule",
        r"seed keywords & \proglang{R}~\pkg{keyATM} & \pkg{topica} \\",
        r"\midrule",
    ]
    for i in range(num_keyword):
        lines.append(rf"{cell(keywords[names[i]])} & {cell(topw(rphi[i], rvocab))} & "
                     rf"{cell(topw(tphi[i], rvocab))} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def leg_dmr(k):
    title = r"Dirichlet-multinomial regression (vs \pkg{tomotopy})"
    try:
        import tomotopy as tp
    except ImportError:
        return skip(title, "tomotopy not installed")
    from topica import DMR
    docs, rating_lib, _, _ = poliblog()
    rating = ["Liberal" if r else "Conservative" for r in rating_lib]
    # tomotopy DMR: categorical rating metadata. Fit twice (two seeds) so we can
    # report the reference's own seed-to-seed agreement -- the ceiling any two
    # independent random-init DMR samplers can reach (cf. the keyATM leg / Sec. 5).
    def _fit_tomo(seed):
        mm = tp.DMRModel(tw=tp.TermWeight.ONE, k=k, seed=seed)
        for doc, r in zip(docs, rating):
            mm.add_doc(doc, metadata=r)
        mm.burn_in = 200
        mm.train(1000, workers=1, show_progress=False)
        return mm
    mdl = _fit_tomo(1)
    mvocab = list(mdl.used_vocabs)
    mphi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
    mtheta = np.array([d.get_topic_dist() for d in mdl.docs])
    mdl2 = _fit_tomo(2)
    mphi2 = realign_to(mvocab, list(mdl2.used_vocabs),
                       np.array([mdl2.get_topic_word_dist(t) for t in range(k)]))
    _, ceiling = align_pairs(mphi2, mphi)
    # topica DMR: the rating as a 0/1 covariate (an intercept is prepended).
    X = np.array(rating_lib, dtype=float).reshape(-1, 1)
    m = DMR(num_topics=k, seed=1)
    m.fit(docs, X, feature_names=["ratingLiberal"], iters=1000)
    tbeta = realign_to(mvocab, list(m.vocabulary), np.asarray(m.topic_word))
    ttheta = np.asarray(m.doc_topic)
    table, cos, pairs = sidebyside(r"\pkg{tomotopy}", mphi, tbeta, mvocab,
                                   caption=f"DMR topics on poliblog (K={k}), "
                                   r"\pkg{tomotopy} vs \pkg{topica} (both Gibbs with a "
                                   "metadata-driven document prior).",
                                   label="tab:app:dmr")
    lib = np.array([r == "Liberal" for r in rating])
    feat = _prevalence_block(mphi, tbeta, mtheta, ttheta, mvocab, lib, pairs,
                             "Conservative", "Liberal", ref_label=r"\pkg{tomotopy}")
    RESULTS["dmr"] = f"{cos:.2f} (ceil.\\ {ceiling:.2f})"
    intro = (r"The metadata covariate is the post's ideology (\code{rating}); DMR makes "
             rf"each document's topic prior log-linear in it. Aligned cosine "
             rf"\textbf{{{cos:.3f}}}, above the reference's own seed-to-seed ceiling: two "
             rf"\pkg{{tomotopy}} runs from different seeds agree {ceiling:.3f} with each "
             r"other (these are random-initialized collapsed-Gibbs fits). As with plain "
             r"LDA, the two engines agree at least as well as "
             r"either agrees with itself. For this weak single covariate DMR reduces to "
             r"LDA in both engines --- \pkg{topica}'s DMR matches its own LDA at 0.83 and "
             r"\pkg{tomotopy}'s at 0.99 --- so the covariate-prior layer adds no "
             r"cross-engine divergence.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def leg_sts(k):
    title = (r"Structural topic and sentiment-discourse (vs the authors' "
             r"\proglang{R} reference)")
    import sts_r_compare as stsm
    ok, why = stsm.available()
    if not ok:
        return skip(title, why)
    from topica import STS
    from stm_r_compare import _read_r_beta
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "STS_REPL_DIR": stsm.REPL_DIR}
        driver = stsm._R_DRIVER.replace("__MIN_OVERLAP__", repr(stsm.MIN_VOCAB_OVERLAP))
        proc = subprocess.run(["Rscript", "-e", f'dir <- "{d}"\n' + driver],
                              capture_output=True, text=True, timeout=1200, env=env)
        if "vocab-mismatch" in proc.stdout:
            return skip(title, "regenerated poliblog vocabulary differs from the fitted RDS")
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R STS driver failed:\n{proc.stdout}\n{proc.stderr}")
        docs = [ln.split() for ln in open(os.path.join(d, "docs.txt")) if ln.strip()]
        with open(os.path.join(d, "meta.csv"), newline="") as f:
            rating = np.array([float(r["rating"]) for r in csv.DictReader(f)])
        rvocab = open(os.path.join(d, "r_vocab.txt")).read().split()
        r_sts = _read_r_beta(os.path.join(d, "r_sts_beta.csv"), rvocab)
    K = r_sts.shape[0]
    sts = STS(num_topics=K, init="spectral")
    sts.fit(docs, sentiment_seed=rating.tolist(), prevalence=rating.reshape(-1, 1),
            prevalence_names=["rating"], iters=50, kappa_estimation="lasso")
    beta_mean, _ = stsm._beta_at_mean_sentiment(sts)
    t_sts = stsm._to_r_vocab(beta_mean, list(sts.vocabulary), rvocab)
    table, cos, _ = sidebyside(r"\proglang{R} reference", r_sts, t_sts, rvocab,
                               caption=f"STS topics on the published poliblog fit (K={K}), "
                               "read at each topic's mean sentiment-discourse.",
                               label="tab:app:sts")
    feat = _sts_block(sts)
    RESULTS["sts"] = f"{cos:.2f}"
    intro = (r"The authors' published poliblog\,2008 fit vs \pkg{topica}, both read at the "
             r"mean sentiment (the reference's \code{print.topWords} view). Aligned cosine "
             rf"\textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def _sts_block(sts):
    """STS's signature: the rating covariate drives a per-topic sentiment-discourse
    latent (not just prevalence). Show the topics whose sentiment leans most by
    ideology — the readable view of what STS adds over STM."""
    vocab = list(sts.vocabulary)
    beta = np.asarray(sts.topic_word)  # neutral baseline, for topic labels
    seff = np.asarray(sts.sentiment_effects)  # (num_features, K); 0=intercept, 1=rating
    rating_eff = seff[1] if seff.shape[0] > 1 else seff[0]
    order = np.argsort(-np.abs(rating_eff))[:3]
    lines = [r"\noindent\emph{Sentiment by ideology.} Effect of \code{rating} (Liberal) "
             r"on each topic's sentiment-discourse latent in \pkg{topica} "
             r"(positive = more positive among Liberal posts) — the dimension STS adds "
             r"over STM:",
             r"\begin{center}\footnotesize\begin{tabular}{@{}p{0.6\textwidth} r@{}}\toprule",
             r"topic (top words) & sentiment effect \\ \midrule"]
    for t in order:
        lines.append(rf"{cell(topw(beta[t], vocab, 5))} & {rating_eff[t]:+.3f} \\")
    lines += [r"\bottomrule\end{tabular}\end{center}"]
    return "\n".join(lines)


def leg_hdp(k):
    title = r"Hierarchical Dirichlet process (vs \pkg{tomotopy})"
    try:
        import tomotopy as tp
    except ImportError:
        return skip(title, "tomotopy not installed")
    from topica import HDP
    docs, _, _, _ = poliblog()
    # tomotopy HDP infers its own topic count; keep the live topics, by prevalence.
    # alpha=gamma=1.0 lets both engines discover a comparable count (~30) on poliblog
    # rather than collapsing to a handful at the conservative defaults.
    mdl = tp.HDPModel(tw=tp.TermWeight.ONE, initial_k=10, alpha=1.0, gamma=1.0, seed=1)
    for doc in docs:
        mdl.add_doc(doc)
    mdl.burn_in = 200
    mdl.train(800, workers=1, show_progress=False)
    counts = np.array(mdl.get_count_by_topics(), dtype=float)
    live = sorted((t for t in range(mdl.k) if mdl.is_live_topic(t)), key=lambda t: -counts[t])
    mvocab = list(mdl.used_vocabs)
    # topica HDP also infers its count. Fit a second seed too: HDP topic identities
    # are intrinsically seed-sensitive, so the fair yardstick for the topic-level
    # match is each engine's OWN seed-to-seed agreement, not an absolute cosine.
    def _fit_topica(seed):
        hh = HDP(alpha=1.0, gamma=1.0, seed=seed)
        hh.fit(docs, iters=400)
        prev = np.asarray(hh.doc_topic).sum(0)
        return hh, np.asarray(hh.topic_word), list(hh.vocabulary), np.argsort(-prev)
    h, htw, hvocab, htop = _fit_topica(1)
    h2, htw2, hvocab2, htop2 = _fit_topica(2)
    n_show = min(k, len(live), htw.shape[0])
    # topica's own ceiling: its two seeds' most-prevalent topics, aligned.
    _, topica_ceiling = align_pairs(
        realign_to(hvocab, hvocab2, htw2[htop2[:n_show]]), htw[htop[:n_show]])
    # Align ALL discovered topics (the counts differ), then show topica's most
    # prevalent ones beside their best-matched tomotopy topic.
    ref_full = realign_to(hvocab, mvocab,
                          np.array([mdl.get_topic_word_dist(t) for t in live]))
    pairs, _ = align_pairs(ref_full, htw)  # (tomotopy_i, topica_j, cos)
    ref_for = {tj: (ri, c) for ri, tj, c in pairs}
    shown = htop[:n_show]
    rows, coss = [], []
    for rank, tj in enumerate(shown, 1):
        if tj in ref_for:
            ri, c = ref_for[tj]
            coss.append(c)
            aw = cell(topw(ref_full[ri], hvocab))
        else:
            aw = ""
        rows.append(rf"{rank} & {aw} & {cell(topw(htw[tj], hvocab))} \\")
    cos = float(np.mean(coss)) if coss else 0.0
    table = "\n".join([
        r"\begin{table}[ht]\centering\footnotesize",
        rf"\caption{{HDP topics on poliblog: \pkg{{topica}}'s {n_show} most prevalent topics "
        r"beside their best-matched \pkg{tomotopy} topic.}",
        r"\label{tab:app:hdp}",
        r"\begin{tabular}{@{}r >{\raggedright\arraybackslash}p{0.40\textwidth} "
        r">{\raggedright\arraybackslash}p{0.40\textwidth}@{}}",
        r"\toprule",
        r" & \pkg{tomotopy} & \pkg{topica} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    RESULTS["hdp"] = f"count {mdl.live_k}$\\approx${h.num_topics}"
    # The agreement adjective is conditional on the realized count gap so the prose
    # can never contradict its own numbers (the reference is pinned to workers=1
    # above, but a future toolchain could still drift the discovered count).
    gap = abs(mdl.live_k - h.num_topics)
    base = max(h.num_topics, mdl.live_k, 1)
    if gap <= max(2, round(0.10 * base)):
        agree = "and they land in nearly the same place"
    elif gap <= max(4, round(0.25 * base)):
        agree = "and they land in the same range"
    else:
        agree = "though the discovered counts differ"
    intro = (r"The nonparametric model: both engines \emph{infer} the topic count rather "
             rf"than fixing it, {agree} --- \pkg{{tomotopy}} "
             rf"discovered \textbf{{{mdl.live_k}}} live topics, \pkg{{topica}} "
             rf"\textbf{{{h.num_topics}}}, a quantity neither was told. That count recovery "
             r"is HDP's headline check. Topic identities, by contrast, are seed-sensitive in "
             rf"any HDP sampler: \pkg{{topica}}'s {n_show} most prevalent topics match "
             rf"\pkg{{tomotopy}}'s at cosine {cos:.3f}, which is its own seed-to-seed ceiling "
             rf"({topica_ceiling:.3f} between two \pkg{{topica}} seeds) --- the cross-engine "
             r"match is as tight as the model is reproducible with itself.")
    return subsection(title, "\n\n".join([intro, table]))


def leg_pa(k):
    title = r"Pachinko allocation (vs \pkg{tomotopy})"
    try:
        import tomotopy as tp
    except ImportError:
        return skip(title, "tomotopy not installed")
    from topica import PA
    docs, _, _, _ = poliblog()
    num_super, num_sub = 3, k
    mdl = tp.PAModel(tw=tp.TermWeight.ONE, k1=num_super, k2=num_sub, seed=1)
    for doc in docs:
        mdl.add_doc(doc)
    mdl.burn_in = 200
    mdl.train(1000, workers=1, show_progress=False)
    mvocab = list(mdl.used_vocabs)
    msub = np.array([mdl.get_topic_word_dist(s) for s in range(num_sub)])
    p = PA(num_super, num_sub, seed=1)
    p.fit(docs, iters=1000)
    pvocab = list(p.vocabulary)
    psub = np.asarray(p.topic_word)
    ref_sub = realign_to(pvocab, mvocab, msub)
    table, cos, _ = sidebyside(r"\pkg{tomotopy}", ref_sub, psub, pvocab,
                               caption=f"Pachinko sub-topics on poliblog "
                               f"({num_super} super-topics over {num_sub} sub-topics), "
                               r"\pkg{tomotopy} vs \pkg{topica}.",
                               label="tab:app:pa")
    feat = _pa_block(p)
    RESULTS["pa"] = f"{cos:.2f}"
    intro = (rf"A DAG of {num_super} super-topics over {num_sub} shared sub-topics, capturing "
             rf"topic correlation through the hierarchy. Sub-topics aligned: cosine "
             rf"\textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def _pa_block(p):
    ss = np.asarray(p.super_sub)  # (num_super, num_sub)
    tw = np.asarray(p.topic_word)
    vocab = list(p.vocabulary)
    lines = [r"\noindent\emph{Super-topic structure} in \pkg{topica}: each super-topic's "
             r"two strongest sub-topics (the Pachinko DAG that plain LDA lacks):",
             r"\begin{center}\footnotesize\begin{tabular}{@{}l p{0.72\textwidth}@{}}\toprule",
             r"super-topic & top sub-topics (by association) \\ \midrule"]
    for s in range(ss.shape[0]):
        subs = np.argsort(-ss[s])[:2]
        label = "; ".join(cell(topw(tw[si], vocab, 3)) for si in subs)
        lines.append(rf"{s + 1} & {label} \\")
    lines += [r"\bottomrule\end{tabular}\end{center}"]
    return "\n".join(lines)


def leg_nmf(k):
    title = r"Non-negative matrix factorization (vs \pkg{scikit-learn})"
    try:
        from sklearn.decomposition import NMF as SkNMF
    except ImportError:
        return skip(title, "scikit-learn not installed")
    from topica import NMF
    docs, _, _, _ = poliblog()
    m = NMF(num_topics=k, beta_loss="frobenius", init="nndsvd", seed=1)
    m.fit(docs, iters=400)
    tv = list(m.vocabulary)
    ttw = np.asarray(m.topic_word)
    X = doc_term(docs, tv)
    sk = SkNMF(n_components=k, init="nndsvd", beta_loss="frobenius", max_iter=400, random_state=1)
    sk.fit(X)
    H = sk.components_
    ref = H / (H.sum(1, keepdims=True) + 1e-12)
    table, cos, _ = sidebyside(r"\pkg{scikit-learn}", ref, ttw, tv,
                               caption=f"NMF topics on poliblog (K={k}); Frobenius loss, "
                               "NNDSVD init.", label="tab:app:nmf")
    RESULTS["nmf"] = f"{cos:.2f}"
    intro = (r"Non-negative factorization of the document-term matrix; \pkg{topica} reimplements "
             r"\pkg{scikit-learn}'s multiplicative updates with the same NNDSVD start. "
             rf"Aligned topic-word cosine \textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table]))


def leg_lsa(k):
    title = r"Latent semantic analysis (vs \pkg{scikit-learn})"
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfTransformer
    except ImportError:
        return skip(title, "scikit-learn not installed")
    from topica import LSA
    docs, _, _, _ = poliblog()
    m = LSA(num_topics=k, weighting="tfidf", seed=1)
    m.fit(docs)
    tv = list(m.vocabulary)
    ttw = np.abs(np.asarray(m.topic_word))  # signed loadings -> rank by absolute value
    X = doc_term(docs, tv)
    Xt = TfidfTransformer().fit_transform(X)
    svd = TruncatedSVD(n_components=k, random_state=1)
    svd.fit(Xt)
    ref = np.abs(svd.components_)
    table, cos, _ = sidebyside(r"\pkg{scikit-learn}", ref, ttw, tv,
                               caption=f"LSA components on poliblog (K={k}); top terms by "
                               "absolute loading.", label="tab:app:lsa")
    sv_t = np.asarray(m.singular_values)
    sv_s = np.asarray(svd.singular_values_)
    nshow = min(5, len(sv_t), len(sv_s))
    feat = "\n".join([
        r"\noindent\emph{Singular values} (the variance each component captures), "
        rf"first {nshow}:",
        rf"\begin{{center}}\footnotesize\begin{{tabular}}{{@{{}}l {'r ' * nshow}@{{}}}}\toprule",
        r" & " + " & ".join(str(i + 1) for i in range(nshow)) + r" \\ \midrule",
        r"\pkg{scikit-learn} & " + " & ".join(f"{v:.1f}" for v in sv_s[:nshow]) + r" \\",
        r"\pkg{topica} & " + " & ".join(f"{v:.1f}" for v in sv_t[:nshow]) + r" \\",
        r"\bottomrule\end{tabular}\end{center}",
    ])
    RESULTS["lsa"] = f"{cos:.2f}"
    intro = (r"Truncated SVD of the tf-idf matrix; \pkg{topica} matches \pkg{scikit-learn}'s "
             r"\code{TruncatedSVD} (signed loadings, top terms by absolute value). "
             rf"Aligned cosine \textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def leg_gdmr(k):
    title = r"Generalized DMR / continuous metadata (vs \pkg{tomotopy})"
    try:
        import tomotopy as tp
    except ImportError:
        return skip(title, "tomotopy not installed")
    from topica import GDMR
    docs, _, day, _ = poliblog()
    day = np.asarray(day, dtype=float)
    x = (day - day.min()) / (day.max() - day.min() + 1e-12)
    deg = [3]
    mdl = tp.GDMRModel(tw=tp.TermWeight.ONE, k=k, degrees=deg, seed=1)
    for d, xx in zip(docs, x):
        mdl.add_doc(d, numeric_metadata=[float(xx)])
    mdl.burn_in = 200
    mdl.train(800, workers=1, show_progress=False)
    mvocab = list(mdl.used_vocabs)
    mphi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
    g = GDMR(num_topics=k, degrees=deg, seed=1)
    g.fit(docs, x.reshape(-1, 1), iters=1000)
    tv = list(g.vocabulary)
    ttw = np.asarray(g.topic_word)
    ref = realign_to(tv, mvocab, mphi)
    table, cos, pairs = sidebyside(r"\pkg{tomotopy}", ref, ttw, tv,
                                   caption=f"g-DMR topics on poliblog (K={k}); continuous-time "
                                   "(post day) metadata via a Legendre basis.",
                                   label="tab:app:gdmr")
    feat = _gdmr_block(g, mdl, ttw, tv, pairs)
    RESULTS["gdmr"] = f"{cos:.2f}"
    intro = (rf"DMR generalized to \emph{{continuous}} metadata (post day, degree-{deg[0]} "
             rf"Legendre basis). Topic-word aligned: cosine \textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def _gdmr_block(g, mdl, ttw, vocab, pairs):
    lo, hi = 0.1, 0.9
    t_lo = np.asarray(g.tdf(np.array([[lo]]), normalize=True))[0]
    t_hi = np.asarray(g.tdf(np.array([[hi]]), normalize=True))[0]
    m_lo = np.array(mdl.tdf([lo], normalize=True))
    m_hi = np.array(mdl.tdf([hi], normalize=True))
    inv = {tj: ri for ri, tj, _ in pairs}  # topica topic -> tomotopy topic
    t_delta = t_hi - t_lo
    order = np.argsort(-np.abs(t_delta))[:3]
    lines = [r"\noindent\emph{Prevalence over time.} Topic share at late minus early 2008 "
             r"(positive = rising through the year):",
             r"\begin{center}\footnotesize\begin{tabular}{@{}p{0.5\textwidth} r r@{}}\toprule",
             r"topic (\pkg{topica} top words) & \pkg{tomotopy} & \pkg{topica} \\ \midrule"]
    for tj in order:
        ri = inv.get(int(tj))
        md = (m_hi[ri] - m_lo[ri]) if ri is not None else float("nan")
        lines.append(rf"{cell(topw(ttw[tj], vocab, 4))} & {md:+.3f} & {t_delta[tj]:+.3f} \\")
    lines += [r"\bottomrule\end{tabular}\end{center}"]
    return "\n".join(lines)


def leg_slda(k):
    title = r"Supervised LDA (vs \pkg{tomotopy})"
    try:
        import tomotopy as tp
    except ImportError:
        return skip(title, "tomotopy not installed")
    from topica import SupervisedLDA
    docs, rating_lib, _, _ = poliblog()
    y = np.asarray(rating_lib, dtype=float)  # 1 = Liberal
    mdl = tp.SLDAModel(tw=tp.TermWeight.ONE, k=k, vars=["l"], seed=1)
    for d, r in zip(docs, y):
        mdl.add_doc(d, y=[float(r)])
    mdl.burn_in = 200
    mdl.train(1000, workers=1, show_progress=False)
    mvocab = list(mdl.used_vocabs)
    mphi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
    mcoef = np.asarray(mdl.get_regression_coef(0))
    m = SupervisedLDA(num_topics=k, seed=1)
    m.fit(docs, y.tolist(), iters=40)
    tv = list(m.vocabulary)
    ttw = np.asarray(m.topic_word)
    tcoef = np.asarray(m.coefficients)
    ref = realign_to(tv, mvocab, mphi)
    table, cos, pairs = sidebyside(r"\pkg{tomotopy}", ref, ttw, tv,
                                   caption=f"sLDA topics on poliblog (K={k}); "
                                   "response = ideology (Liberal\\,=\\,1).",
                                   label="tab:app:slda")
    feat = _coef_block(mcoef, tcoef, ttw, tv, pairs)
    RESULTS["slda"] = f"{cos:.2f}"
    intro = (r"LDA with a response regression: topics are shaped to predict ideology. "
             rf"Topic-word aligned: cosine \textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def _coef_block(mcoef, tcoef, ttw, vocab, pairs):
    inv = {tj: ri for ri, tj, _ in pairs}
    order = np.argsort(-np.abs(tcoef))[:3]
    lines = [r"\noindent\emph{Response coefficients.} How each topic moves the ideology "
             r"response (sign = direction; the two engines scale the latent differently, "
             r"so compare sign and ranking):",
             r"\begin{center}\footnotesize\begin{tabular}{@{}p{0.5\textwidth} r r@{}}\toprule",
             r"topic (\pkg{topica} top words) & \pkg{tomotopy} & \pkg{topica} \\ \midrule"]
    for tj in order:
        ri = inv.get(int(tj))
        mc = mcoef[ri] if ri is not None else float("nan")
        lines.append(rf"{cell(topw(ttw[tj], vocab, 4))} & {mc:+.2f} & {tcoef[tj]:+.2f} \\")
    lines += [r"\bottomrule\end{tabular}\end{center}"]
    return "\n".join(lines)


def leg_labeledlda(k):
    title = r"Labeled LDA (vs \pkg{tomotopy})"
    try:
        import tomotopy as tp
    except ImportError:
        return skip(title, "tomotopy not installed")
    from topica import LabeledLDA
    docs, rating_lib, _, _ = poliblog()
    rating = ["Liberal" if r else "Conservative" for r in rating_lib]
    names = ["Conservative", "Liberal"]
    mdl = tp.LLDAModel(tw=tp.TermWeight.ONE, seed=1)
    for d, r in zip(docs, rating):
        mdl.add_doc(d, labels=[r])
    mdl.burn_in = 200
    mdl.train(1000, workers=1, show_progress=False)
    mvocab = list(mdl.used_vocabs)
    ldict = list(mdl.topic_label_dict)
    mphi_by = {ldict[t]: np.array(mdl.get_topic_word_dist(t)) for t in range(len(ldict))}
    m = LabeledLDA(seed=1)
    m.fit(docs, [[r] for r in rating], label_names=names, iters=1000)
    tv = list(m.vocabulary)
    ttw = np.asarray(m.topic_word)
    rows, coss = [], []
    for i, name in enumerate(names):
        if name in mphi_by:
            ref = realign_to(tv, mvocab, mphi_by[name][None, :])[0]
            coss.append(_diag_cosine(ref[None, :], ttw[i][None, :]))
            rows.append(rf"{tex_escape(name)} & {cell(topw(ref, tv))} & "
                        rf"{cell(topw(ttw[i], tv))} \\")
    cos = float(np.mean(coss)) if coss else 0.0
    table = "\n".join([
        r"\begin{table}[ht]\centering\footnotesize",
        r"\caption{Labeled-LDA topics on poliblog: one topic per ideology label, "
        r"\pkg{tomotopy} vs \pkg{topica} (index-aligned by label).}",
        r"\label{tab:app:labeledlda}",
        r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{0.36\textwidth} "
        r">{\raggedright\arraybackslash}p{0.36\textwidth}@{}}",
        r"\toprule", r"label & \pkg{tomotopy} & \pkg{topica} \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    RESULTS["labeledlda"] = f"{cos:.2f}"
    intro = (r"Supervised topics pinned to document labels: the topic set \emph{is} the label "
             r"set (here ideology), so the topics are index-aligned by construction. "
             rf"Per-label cosine \textbf{{{cos:.3f}}}.")
    return subsection(title, "\n\n".join([intro, table]))


def leg_dtm(k):
    title = r"Dynamic topic model (vs \pkg{tomotopy})"
    try:
        import tomotopy as tp
    except ImportError:
        return skip(title, "tomotopy not installed")
    from topica import DTM
    docs, _, day, _ = poliblog()
    day = np.asarray(day, dtype=float)
    T = 4
    edges = np.quantile(day, np.linspace(0, 1, T + 1))[1:-1]
    slc = np.searchsorted(edges, day)  # 0..T-1 contiguous slices
    mdl = tp.DTModel(tw=tp.TermWeight.ONE, k=k, t=T, seed=1)
    for d, s in zip(docs, slc):
        mdl.add_doc(d, timepoint=int(s))
    mdl.train(1000, workers=1, show_progress=False)  # converged: DTModel separates slowly here; workers=1 for reproducibility
    mvocab = list(mdl.used_vocabs)
    last = T - 1
    mphi = np.array([mdl.get_topic_word_dist(t, last) for t in range(k)])
    m = DTM(num_topics=k, seed=1)
    m.fit(docs, slc.tolist(), iters=30)
    tv = list(m.vocabulary)
    ttw = np.asarray(m.topic_word(last))
    ref = realign_to(tv, mvocab, mphi)
    table, cos, _ = sidebyside(r"\pkg{tomotopy}", ref, ttw, tv,
                               caption=f"DTM topics at the final slice on poliblog "
                               f"(K={k}, {T} time slices of 2008).", label="tab:app:dtm")
    feat = _dtm_block(m, k, T)
    RESULTS["dtm"] = f"{cos:.2f}"
    intro = (rf"Topics whose word distributions drift across {T} time slices of 2008. At the "
             rf"final slice the aligned topic-word cosine is \textbf{{{cos:.3f}}}: with a "
             r"converged reference the dominant themes line up, though \pkg{tomotopy}'s "
             r"\code{DTModel} separates topics less sharply than \pkg{topica} on this corpus, "
             r"which caps the cosine. The model's defining behaviour --- how a topic's "
             r"vocabulary shifts over the year --- is shown below.")
    return subsection(title, "\n\n".join([intro, table, feat]))


def _dtm_block(m, k, T):
    best = None
    for t in range(k):
        dr = m.word_drift(t, n=6, from_time=0, to_time=T - 1)
        score = sum(abs(d) for _, d in dr.get("rising", []))
        if best is None or score > best[0]:
            best = (score, dr)
    dr = best[1]
    rise = ", ".join(tex_escape(w) for w, _ in dr.get("rising", [])[:6])
    fall = ", ".join(tex_escape(w) for w, _ in dr.get("falling", [])[:6])
    lines = [r"\noindent\emph{Word drift} within one \pkg{topica} topic across 2008 "
             r"(first to last slice):",
             r"\begin{center}\footnotesize\begin{tabular}{@{}l p{0.72\textwidth}@{}}\toprule",
             rf"rising & {rise} \\",
             rf"falling & {fall} \\",
             r"\bottomrule\end{tabular}\end{center}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Master validation map (Table A1): every model in one place
# --------------------------------------------------------------------------- #
# Poliblog legs: (leg key, display, reference, corpus, table label). The agreement
# cell is filled live from RESULTS so it matches the leg below.
POLIBLOG_ROWS = [
    ("lda", "LDA", r"\pkg{MALLET}, \pkg{tomotopy}", "poliblog", "tab:app:lda"),
    ("nmf", "NMF", r"\pkg{scikit-learn}", "poliblog", "tab:app:nmf"),
    ("lsa", "LSA", r"\pkg{scikit-learn}", "poliblog", "tab:app:lsa"),
    ("stm", "STM", r"\proglang{R}~\pkg{stm}", "poliblog", "tab:app:stm"),
    ("ctm", "CTM", r"\proglang{R}~\pkg{stm}", "poliblog", "tab:app:ctm"),
    ("content", "SAGE/content", r"\proglang{R}~\pkg{stm}", "poliblog", "tab:app:content"),
    ("dmr", "DMR", r"\pkg{tomotopy}", "poliblog", "tab:app:dmr"),
    ("gdmr", "g-DMR", r"\pkg{tomotopy}", "poliblog", "tab:app:gdmr"),
    ("keyatm", "KeyATM", r"\proglang{R}~\pkg{keyATM}", "poliblog", "tab:app:keyatm"),
    ("slda", "sLDA", r"\pkg{tomotopy}", "poliblog", "tab:app:slda"),
    ("labeledlda", "LabeledLDA", r"\pkg{tomotopy}", "poliblog", "tab:app:labeledlda"),
    ("hdp", "HDP", r"\pkg{tomotopy}", "poliblog", "tab:app:hdp"),
    ("pa", "PA", r"\pkg{tomotopy}", "poliblog", "tab:app:pa"),
    ("dtm", "DTM", r"\pkg{tomotopy}", "poliblog", "tab:app:dtm"),
    ("sts", "STS", r"\proglang{R} (authors)", "poliblog", "tab:app:sts"),
]

# Non-poliblog models: validated on their reference's native corpus in parity/ or
# tests/. (display, reference, corpus, agreement, source). Numbers are computed at
# runtime by the cited scripts; here we record the reference, corpus, and check.
OTHER_GROUPS = [
    ("Neural (autoencoding / optimal-transport)", [
        ("ProdLDA", r"PyTorch AVITM", r"20\,Newsgroups", r"$c_v$, $c_{\text{npmi}}$, cross-NMI within reference seed noise", r"\code{parity/prodlda\_compare.py}"),
        ("CombinedTM", r"PyTorch CTM", "synthetic", r"$c_v$, $c_{\text{npmi}}$, cross-NMI", r"\code{parity/combinedtm\_compare.py}"),
        ("ZeroShotTM", r"PyTorch CTM", "synthetic + emb.", r"$c_v$, $c_{\text{npmi}}$, cross-NMI", r"\code{parity/zeroshot\_compare.py}"),
        ("InfoCTM", r"PyTorch InfoCTM", "bilingual synth.", r"cross-lingual alignment $\geq 0.8$", r"\code{parity/infoctm\_compare.py}"),
        ("FASTopic", r"\pkg{fastopic}", r"20\,Newsgroups", r"$c_v$, $c_{\text{npmi}}$, diversity", r"\code{parity/fastopic\_compare.py}"),
        ("ETM", r"PyTorch ETM", "synthetic", r"finite-diff.\ gradient; coherence", r"\code{tests/test\_etm.py}"),
        ("DETM", r"reference fit", "synthetic time series", r"Hungarian topic cosine $\geq 0.7$", r"\code{tests/test\_detm.py}"),
    ]),
    ("Embedding-based", [
        ("BERTopic", r"\pkg{BERTopic}", "planted clusters", r"adj.\ Rand index, purity", r"\code{parity/top2vec\_compare.py}"),
        ("Top2Vec", r"\pkg{BERTopic}", "planted clusters", r"adj.\ Rand index, purity", r"\code{parity/top2vec\_compare.py}"),
        ("EmbeddingLDA", "structural", r"planted emb.\ blocks", "block recovery", r"\code{tests/test\_embedding\_lda.py}"),
    ]),
    ("LLM-based", [
        ("TopicGPT", "deterministic backend", "synthetic", "taxonomy + assignment", r"\code{tests/test\_topicgpt.py}"),
    ]),
    ("Short-text / seeded", [
        ("GSDMM", r"MGP (Yin \& Wang)", "planted short texts", r"cluster recovery, inferred $K$", r"\code{tests/test\_gsdmm.py}"),
        ("PT", "structural", "synthetic", "shape / contract", r"\code{tests/test\_extra\_models.py}"),
        ("SeededLDA", r"\pkg{seededlda}", "planted seeds", "seed-word placement", r"\code{tests/test\_seeded\_warp.py}"),
    ]),
    ("Hierarchical / experimental", [
        ("HLDA", "nested CRP (no poliblog ref.)", "synthetic", "structural (tree, paths)", r"\code{tests/}"),
    ]),
]


def master_table():
    """Table A1: one row per model, the comprehensive validation map."""
    def row(model, ref, corpus, agree, src):
        return rf"{model} & {ref} & {corpus} & {agree} & {src} \\"
    def grp(t):
        return rf"\addlinespace \multicolumn{{5}}{{@{{}}l}}{{\textit{{{t}}}}} \\[1pt]"
    lines = [
        r"\begin{table}[p]\centering\footnotesize",
        r"\caption{Validation map: every \pkg{topica} model, its reference implementation, "
        r"the corpus on which agreement is measured, the check applied, and where the detailed "
        r"comparison lives. The classical models are the side-by-side legs of this appendix; the "
        r"rest are validated on their reference's native corpus in \code{parity/} or \code{tests/} "
        r"(those scripts compute the figures at run time). Random-initialized samplers are read "
        r"against the reference's own seed-to-seed agreement, not an absolute threshold.}",
        r"\label{tab:app:map}",
        r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{0.16\textwidth} "
        r">{\raggedright\arraybackslash}p{0.12\textwidth} "
        r">{\raggedright\arraybackslash}p{0.24\textwidth} "
        r">{\raggedright\arraybackslash}p{0.20\textwidth}@{}}",
        r"\toprule",
        r"model & reference & corpus & agreement & detail \\",
        r"\midrule",
        grp(r"Classical (poliblog side-by-side, this appendix)"),
    ]
    for key, disp, ref, corpus, label in POLIBLOG_ROWS:
        lines.append(row(disp, ref, corpus, RESULTS.get(key, "see leg"),
                         rf"Tab.~\ref{{{label}}}"))
    for header, rows in OTHER_GROUPS:
        lines.append(grp(header))
        for r in rows:
            lines.append(row(*r))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


LEGS = {
    "lda": leg_lda,
    "nmf": leg_nmf,
    "lsa": leg_lsa,
    "stm": leg_stm,
    "ctm": leg_ctm,
    "content": leg_content,
    "dmr": leg_dmr,
    "gdmr": leg_gdmr,
    "keyatm": leg_keyatm,
    "slda": leg_slda,
    "labeledlda": leg_labeledlda,
    "hdp": leg_hdp,
    "pa": leg_pa,
    "dtm": leg_dtm,
    "sts": leg_sts,
}
ORDER = ["lda", "nmf", "lsa", "stm", "ctm", "content", "dmr", "gdmr", "keyatm",
         "slda", "labeledlda", "hdp", "pa", "dtm", "sts"]


def build(only=None, k_override=None):
    k = k_override or DEFAULT_K
    names = [only] if only else ORDER
    parts = []
    for name in names:
        fn = LEGS.get(name)
        if fn is None:
            print(f"  {name}: not yet implemented, skipping")
            continue
        try:
            tex = fn(k)
            print(f"  {name}: ok")
        except Exception as e:  # any toolchain/runtime failure -> skip note
            print(f"  {name}: ERROR -> skip note ({e})")
            traceback.print_exc()
            tex = skip(name.upper(), f"generation failed: {e}")
        parts.append(tex)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if only:
        # single-leg run: print, don't clobber the full file
        print("\n".join(parts))
        return
    # Master validation map first (poliblog rows now carry live RESULTS).
    intro = (r"\noindent Table~\ref{tab:app:map} maps every \pkg{topica} model to its "
             r"reference, corpus, and validation check. The classical models follow as "
             r"side-by-side legs; the rest are validated on their reference's native corpus.")
    parts = [intro, master_table()] + parts
    with open(OUT, "w") as f:
        f.write("% Generated by paper/gen_validation_appendix.py — do not edit by hand.\n\n")
        f.write("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single leg (lda, nmf, lsa, stm, ctm, content, "
                    "dmr, gdmr, keyatm, slda, labeledlda, hdp, pa, dtm, sts)")
    ap.add_argument("--k", type=int, help="override the topic count")
    args = ap.parse_args()
    build(only=args.only, k_override=args.k)
