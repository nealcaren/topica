"""Parity check: topica `Wordshoal` vs an R reference oracle.

Wordshoal (Lauderdale & Herzog 2016) has no installable-against-modern-quanteda
reference: the `kbenoit/wordshoal` package targets an old quanteda API. So this
harness re-implements the reference algorithm as an R oracle from the paper's
published equations — R `quanteda.textmodels::textmodel_wordfish` for the stage-1
per-domain scaling (the exact Wordfish the reference calls) plus the stage-2
conditional-ML linear-factor coordinate ascent — and compares topica against it on
a fixed two-stage synthetic corpus. A faithful port recovers the same cross-domain
actor scale as the oracle (|r| ~ 1), and both recover the planted positions.

Shells out to `Rscript` with `quanteda` + `quanteda.textmodels` + `jsonlite`. Skips
cleanly (exit 0) if any is unavailable, mirroring the other `parity/` scripts.

    python parity/wordshoal_r_compare.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import topica

# R oracle: stage-1 quanteda wordfish per domain + stage-2 coordinate ascent coded
# from the Lauderdale & Herzog (2016) equations. Reads corpus.json, writes out.json.
R_ORACLE = r"""
suppressMessages({library(quanteda); library(quanteda.textmodels); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
corpus <- fromJSON(args[1], simplifyVector = FALSE)
docs <- corpus$docs; speakers <- unlist(corpus$speakers); domains <- unlist(corpus$domains)
S <- length(docs)
toks <- tokens(sapply(docs, function(d) paste(unlist(d), collapse = " ")))
dfmat <- dfm(toks)
groups <- as.factor(domains); authors <- as.factor(speakers)
M <- nlevels(groups); N <- nlevels(authors)
jVec <- as.integer(groups); iVec <- as.integer(authors)
psi <- rep(NA_real_, S)
for (j in 1:M) {
  sel <- which(jVec == j)
  gdfm <- dfm_trim(dfmat[sel, ], min_termfreq = 1, min_docfreq = 1)
  gdfm <- gdfm[, colSums(gdfm) > 0]
  wf <- textmodel_wordfish(gdfm, tol = c(1e-3, 1e-8))
  psi[sel] <- as.numeric(wf$theta)
}
psi[is.na(psi)] <- 0
prioralpha<-0.5; priorbeta<-0.5; priortheta<-1; priortau<-1
alpha<-rep(0,M); beta<-rep(0,M); theta<-seq(-2,2,length.out=N); tau<-rep(1,N)
lp_of<-function(){
  lp<-sum(dnorm(alpha,0,prioralpha,log=TRUE))+sum(dnorm(beta,0,priorbeta,log=TRUE))+
      sum(dnorm(theta,0,priortheta,log=TRUE))+sum(dgamma(tau,priortau,priortau,log=TRUE))
  lps<-alpha[jVec]+beta[jVec]*theta[iVec]; lp+sum(dnorm(psi,lps,tau[iVec]^(-1/2),log=TRUE))
}
Pd<-solve(matrix(c(prioralpha^2,0,0,priorbeta^2),2,2)); tol<-1e-3; lastlp<--Inf; lp<-lp_of()
while((lp-lastlp)>abs(tol)){
  for(j in 1:M){locs<-which(jVec==j); Y<-psi[locs]; X<-cbind(1,theta[iVec[locs]]); W<-diag(tau[iVec[locs]],length(locs))
    co<-solve(t(X)%*%W%*%X+Pd)%*%t(X)%*%W%*%Y; alpha[j]<-co[1]; beta[j]<-co[2]}
  for(i in 1:N){locs<-which(iVec==i); Y<-matrix(psi[locs]-alpha[jVec[locs]],ncol=1); X<-matrix(beta[jVec[locs]],ncol=1)
    A<-solve(t(X)%*%X+priortheta^(-2)); co<-A%*%t(X)%*%Y; theta[i]<-co[1,1]
    mu<-A%*%t(X)%*%X%*%co; tau[i]<-(priortau+0.5*length(Y))/(priortau+0.5*(sum(Y^2)-mu*(priortheta^(-2))*mu))}
  lastlp<-lp; lp<-lp_of()
}
thetaSE<-rep(NA_real_,N)
for(i in 1:N){locs<-which(iVec==i); X<-matrix(beta[jVec[locs]],ncol=1)
  thetaSE[i]<-sqrt(solve(t(X)%*%X+priortheta^(-2))/tau[i])}
write_json(list(theta=theta, se=thetaSE, tau=tau, speakers=levels(authors)),
           args[2], auto_unbox=TRUE, digits=10)
"""


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        ["Rscript", "-e",
         'cat(all(sapply(c("quanteda","quanteda.textmodels","jsonlite"), requireNamespace, quietly=TRUE)))'],
        capture_output=True, text=True,
    )
    return "TRUE" in probe.stdout


def _planted(n_authors=40, n_domains=12, n_words=60, seed=13):
    rng = np.random.default_rng(seed)
    theta = np.linspace(-1.5, 1.5, n_authors)
    signs = rng.choice([-1.0, 1.0], n_domains)
    beta = signs * rng.uniform(0.6, 1.4, n_domains)
    alpha = rng.normal(0.0, 0.3, n_domains)
    docs, speakers, domains = [], [], []
    for j in range(n_domains):
        bword = rng.uniform(-1.2, 1.2, n_words)
        pword = np.log(rng.uniform(2.0, 9.0, n_words))
        for i in range(n_authors):
            z = alpha[j] + beta[j] * theta[i] + rng.normal(0.0, 0.15)
            counts = rng.poisson(np.exp(pword + bword * z))
            doc = []
            for w, c in enumerate(counts):
                doc.extend([f"w{w}"] * int(c))
            if len(doc) < 2:
                doc = ["w0", "w1"]
            rng.shuffle(doc)
            docs.append(doc)
            speakers.append(f"s{i:03d}")
            domains.append(f"d{j:02d}")
    return docs, speakers, domains, theta


def _pearson(x, y):
    return abs(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def main() -> int:
    if not _r_available():
        print("SKIP: Rscript or quanteda/quanteda.textmodels/jsonlite not available")
        return 0

    docs, speakers, domains, theta_true = _planted()
    m = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    se = dict(zip(m.author_names, m.position_se))

    with tempfile.TemporaryDirectory() as td:
        infile = Path(td) / "corpus.json"
        outfile = Path(td) / "out.json"
        json.dump({"docs": docs, "speakers": speakers, "domains": domains},
                  open(infile, "w"))
        proc = subprocess.run(
            ["Rscript", "-e", R_ORACLE, str(infile), str(outfile)],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0 or not outfile.exists():
            print("SKIP: R oracle run failed:\n", proc.stderr[-2000:])
            return 0
        orc = json.load(open(outfile))

    r_theta = dict(zip(orc["speakers"], orc["theta"]))
    r_se = dict(zip(orc["speakers"], orc["se"]))
    who = sorted(set(pos) & set(r_theta))
    t_topica = np.array([pos[s] for s in who])
    t_oracle = np.array([r_theta[s] for s in who])
    planted = np.array([theta_true[int(s[1:])] for s in who])

    r_vs_oracle = _pearson(t_topica, t_oracle)
    r_topica_truth = _pearson(t_topica, planted)
    r_oracle_truth = _pearson(t_oracle, planted)

    # Full-vector agreement, not just correlation: sign-align topica to the oracle
    # (both are identified only up to sign), then compare the actual position VALUES
    # and the standard errors. A faithful two-stage fit reproduces the oracle's whole
    # theta / se vector, not merely its rank order.
    sign = 1.0 if np.corrcoef(t_topica, t_oracle)[0, 1] >= 0 else -1.0
    dtheta = np.abs(sign * t_topica - t_oracle)
    se_topica = np.array([se[s] for s in who])
    se_oracle = np.array([r_se[s] for s in who])
    se_reldiff = np.abs(se_topica - se_oracle) / np.maximum(np.abs(se_oracle), 1e-9)

    print(f"topica Wordshoal vs R oracle       : |r| = {r_vs_oracle:.4f}")
    print(f"topica   vs planted truth          : |r| = {r_topica_truth:.4f}")
    print(f"R oracle vs planted truth          : |r| = {r_oracle_truth:.4f}")
    print(f"theta value agreement (sign-aligned): max|d|={dtheta.max():.2e} "
          f"mean|d|={dtheta.mean():.2e}")
    print(f"position_se agreement              : max rel diff={se_reldiff.max():.2e}")

    failures = []
    if r_vs_oracle < 0.95:
        failures.append(f"topica-vs-oracle |r|={r_vs_oracle:.4f} < 0.95")
    # Full-vector parity: the sign-aligned positions and the SEs should match the
    # oracle's, not merely correlate. Loose absolute bars accommodate residual
    # stage-1 Wordfish tolerance differences between Rust and quanteda.
    if dtheta.max() > 0.05:
        failures.append(f"theta value max|d|={dtheta.max():.3f} > 0.05 (not full parity)")
    if se_reldiff.max() > 0.10:
        failures.append(f"position_se max rel diff={se_reldiff.max():.3f} > 0.10")
    # topica should recover the planted scale about as well as the oracle does.
    if r_topica_truth < r_oracle_truth - 0.05:
        failures.append(
            f"topica truth-recovery {r_topica_truth:.4f} well below oracle {r_oracle_truth:.4f}"
        )

    if failures:
        print("FAIL: topica and the Wordshoal oracle disagree (" + "; ".join(failures) + ")")
        return 1
    print("PASS: topica Wordshoal matches the reference two-stage scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
