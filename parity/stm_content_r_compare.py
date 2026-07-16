"""Cross-implementation validation of the STM CONTENT model against R `stm`.

The prevalence path is checked in `stm_r_compare.py`; this covers the content
(SAGE) covariate, which is where the topic-collapse bug lived. Same bilingual
corpus to both engines, fit with `content = ~group`, K = 2. Reports, per content
group, how distinct the two topics are in each engine (topic-sep: ~0 separated,
~1 collapsed) and the best-aligned cosine between R's and topica's per-group
word distributions. Skips cleanly when Rscript / `stm` is unavailable.
"""

import csv
import os
import subprocess
import tempfile

import numpy as np

from stm_r_compare import r_stm_available  # reuse the availability check

_EN_W = ["rain", "sun", "cloud", "wind", "storm"]
_DE_W = ["regen", "sonne", "wolke", "sturm", "nebel"]
_EN_F = ["bread", "cheese", "wine", "apple", "meat"]
_DE_F = ["brot", "kaese", "wein", "apfel", "fleisch"]

_R_DRIVER = r"""
suppressMessages(library(stm))
lines <- readLines(file.path(dir, "vdocs.txt"))
toks  <- strsplit(lines, " ")
vocab <- sort(unique(unlist(toks)))
vmap  <- setNames(seq_along(vocab), vocab)
documents <- lapply(toks, function(d){ tb<-table(d); idx<-as.integer(vmap[names(tb)]); o<-order(idx)
  matrix(as.integer(rbind(idx[o], as.integer(tb)[o])), nrow=2) })
meta <- read.csv(file.path(dir,"vmeta.csv"), stringsAsFactors=FALSE)
set.seed(1)
f <- stm(documents, vocab, K=2, content=~group, data=meta, init.type="Spectral", verbose=FALSE)
levs <- f$settings$covariates$yvarlevels
for (g in seq_along(levs)) {
  b <- exp(f$beta$logbeta[[g]]); colnames(b) <- vocab
  write.csv(b, file.path(dir, paste0("r_beta_", levs[g], ".csv")), row.names=FALSE)
}
write(levs, file.path(dir, "r_levels.txt"))
cat("ok\n")
"""


def _make_corpus(seed=42):
    rng = np.random.default_rng(seed)
    docs, groups = [], []
    for words, g, n in [((_EN_W, _EN_F), "en", 50), ((_EN_F, _EN_W), "en", 50),
                        ((_DE_W, _DE_F), "de", 50), ((_DE_F, _DE_W), "de", 50)]:
        for _ in range(n):
            docs.append(rng.choice(words[0], 10).tolist() + rng.choice(words[1], 2).tolist())
            groups.append(g)
    return docs, groups


def _read_beta(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        cols = [h.strip('"') for h in next(r)]
        mat = np.array([[float(x) for x in row] for row in r])
    return cols, mat


def _aligned_cosine(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = an @ bn.T
    used, total = set(), 0.0
    for i in np.argsort(-sim.max(1)):
        for j in np.argsort(-sim[i]):
            if j not in used:
                used.add(j)
                total += sim[i, j]
                break
    return total / a.shape[0]


def _topic_sep(mat):  # K=2: cosine of the two topics (low = separated)
    a = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return float(a[0] @ a[1])


def run(verbose: bool = True) -> dict:
    """Fit both engines content-only and compare. Returns per-group cosines and
    topic-separation for R and topica. Raises if R/stm is unavailable."""
    if not r_stm_available():
        raise RuntimeError("Rscript with the 'stm' package is not available")
    from topica.models import STM

    docs, groups = _make_corpus()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "vdocs.txt"), "w") as f:
            f.write("\n".join(" ".join(x) for x in docs) + "\n")
        with open(os.path.join(d, "vmeta.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group"])
            for g in groups:
                w.writerow([g])
        out = subprocess.run(
            ["Rscript", "-e", f'dir <- "{d}"\n' + _R_DRIVER],
            capture_output=True, text=True,
        )
        if "ok" not in out.stdout:
            raise RuntimeError("R stm content fit failed:\n" + out.stderr[-2000:])
        levs = open(os.path.join(d, "r_levels.txt")).read().split()

        m = STM(num_topics=2, seed=1)
        m.fit(docs, content=groups, iters=80)
        vidx = {w: i for i, w in enumerate(m.vocabulary)}
        twg = np.asarray(m.topic_word_by_group)

        result = {"cosine": {}, "r_topic_sep": {}, "tt_topic_sep": {}}
        for g in levs:
            cols, rb = _read_beta(os.path.join(d, f"r_beta_{g}.csv"))
            rb_al = np.zeros((rb.shape[0], len(m.vocabulary)))
            for j, w in enumerate(cols):
                if w in vidx:
                    rb_al[:, vidx[w]] = rb[:, j]
            ttb = twg[:, m.groups.index(g), :]
            result["cosine"][g] = _aligned_cosine(rb_al, ttb)
            result["r_topic_sep"][g] = _topic_sep(rb_al)
            result["tt_topic_sep"][g] = _topic_sep(ttb)

        if verbose:
            print(f"R levels {levs} | topica groups {m.groups}")
            for g in levs:
                print(f"  {g}: R sep={result['r_topic_sep'][g]:.3f} "
                      f"tt sep={result['tt_topic_sep'][g]:.3f} "
                      f"cosine={result['cosine'][g]:.3f}")
        return result


if __name__ == "__main__":
    run(verbose=True)
