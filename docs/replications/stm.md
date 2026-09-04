# Structural Topic Model: the `stm` vignette

**Source.** Roberts, M. E., Stewart, B. M., & Tingley, D. (2019). stm: An R
Package for Structural Topic Models. *Journal of Statistical Software*, 91(2).
The `stm` package is the field standard for prevalence- and content-covariate
topic models in the social sciences.

topica's `STM` reimplements the same model: correlated topics with a prevalence
regression, a content (SAGE) covariate, spectral initialization, and effect
estimation by the method of composition. This page asks whether it produces the
same answers as R's `stm`.

## What "replicate" means for STM

STM is fit by variational EM, which is non-convex: the objective has many local
optima, and the solution depends on where the optimization starts. R's own `stm`
does not return one canonical answer. Fit it twice from different random seeds
and the two topic-word matrices agree only to a cosine of about 0.68. So the bar
is not bit-identical output. It is statistical: under a matched initialization,
topica should land in the same neighborhood of solutions that R lands in, and its
agreement with R should sit inside the spread of R's agreement with itself.

We feed identical integer-coded documents to both engines and align topics
one-to-one before comparing. The harnesses live in `parity/`:
[`stm_poliblog_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_poliblog_compare.py)
and [`stm_poliblog5k_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_poliblog5k_compare.py)
for the prevalence model on Poliblog,
[`stm_content_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_content_r_compare.py)
for the content model, and
[`stm_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_r_compare.py)
for the small Gadarian stress case.

## Content model: close agreement, with a different default prior

The content (SAGE) covariate is the deterministic part of STM: given the topic
assignments, the per-group word distributions follow in closed form. One default
differs, so "exact agreement" would overstate it. topica's content prior defaults
to Gaussian **L2** (`content_prior="l2"`, `content_prior_var=0.5`); R `stm`
defaults to `kappa.prior="L1"` (a glmnet word-wise lasso). The two regularizers
pull the per-group deviations differently, so the fits are close but not
identical. On a bilingual corpus fit with `content = ~group`, K = 2
([`stm_content_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_content_r_compare.py)),
R's L1 fit and topica's L2 fit align to a per-group cosine of ~0.999, but the
topic-separation the two produce differs (R ~0.03 per topic, topica ~0.11) — the
expected signature of L1 (sparser deviations) versus L2. Fit topica with the
**matched** L1 prior (`content_prior="l1"`, R's default) and the separation lines
up too (topica L1 ~0.02, essentially R's), which is how we know the gap is the
default prior, not the inference:

| Content group | topica(L2)–R(L1) cosine | topica L2 sep | topica L1 sep | R (L1) sep |
|---|---:|---:|---:|---:|
| `de` | 0.999 | 0.11 | 0.02 | 0.03 |
| `en` | 0.999 | 0.12 | 0.03 | 0.04 |

This is the path where a symmetric-initialization bug once collapsed all topics
to the background; the high matched cosine against R is how we know it is fixed.
If you need R-comparable content regularization, pass `content_prior="l1"`; topica
keeps L2 as its default because that is the path its committed gold validates. The
matched-prior comparison above is checked in
[`stm_content_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_content_r_compare.py).

## Prevalence model: same neighborhood as R, and the same conclusions

For the prevalence model we compare topica's spectral fit to R's spectral fit on
the `stm` Poliblog vignette, against the floor of R's agreement with itself. On
the 2,000-document corpus at K = 20 the two engines' topic-word matrices align to
a cosine of 0.98; on the full 5,000-document corpus at K = 15 it is 0.92. Both
sit well above how closely R reproduces *itself* across initializations:

| Comparison (Poliblog 5k, K = 15) | aligned cosine |
|---|---:|
| R Spectral vs R Random (R's own basin spread) | 0.62 |
| R Random vs R Random (R's self-consistency) | 0.68 |
| **R Spectral vs topica Spectral** | **0.92** |

topica reproduces R's spectral solution more closely than R reproduces itself
from a different seed. Where the per-topic cosine dips (the 5k median is 0.99,
but a few topics fall lower) it is always a handful of genuinely bistable topics
that the two optimizers split differently, never a systematic offset — the
expected behavior of a non-convex model, where there is no single STM fit to
reproduce.

### Initialization: what is and isn't a reproducible target

The cosines above are EM optima of a non-convex objective, so they differ across
optimizers *and* across initializations. The initialization underneath them is the
Arora anchor-word recovery. topica's recovery runs a scale-adaptive exponentiated
gradient to the **constrained optimum**, and on identical documents it matches R
`stm`'s *reference* recovery (`recoverL2(recoverEG = FALSE)`, the quadratic-program
solve) at a cosine of **1.0** (`parity/spectral_recover_stm.py`, issue #234).

R `stm`'s **default** recovery is a different object: `recoverL2(recoverEG = TRUE)`,
a fixed-step exponentiated gradient that runs a set number of iterations *without
converging*. Its output is sensitive to floating-point order (a 1e-9 perturbation of
the co-occurrence matrix moves it substantially), so it is **not a portably
reproducible target** — R, and any reimplementation, land in different basins from
identical inputs (issue #871). topica's converged recovery and R's default are two
different, equally valid starting points; neither reproduces the other bit-for-bit.

Two practical consequences:

- **Reliability — use restarts.** At large vocabularies the logistic-normal EM has
  catastrophic local optima (much worse held-out completion, and the *worst*
  variational bound) that any single initialization — spectral or random — can fall
  into. `STM.fit(..., restarts=N)` fits `N` independently-seeded starts and returns
  the one with the best bound, a deterministic guard that avoids those basins.

  ```python
  model = topica.STM(20).fit(corpus, prevalence=X, restarts=8)  # keeps best-bound fit
  ```

- **Exact replication — inject the reference β.** To reproduce a *specific* R `stm`
  run, start EM from that run's topic-word matrix via `beta_init=`.
  `topica.stm.beta_from_reference` aligns R's `exp(fit$beta$logbeta[[1]])` to
  topica's vocabulary:

  ```python
  binit = topica.stm.beta_from_reference(r_beta, r_vocab, corpus)
  model = topica.STM(20).fit(corpus, prevalence=X, beta_init=binit)  # lands in R's basin
  ```

What replicates stably across optima is the substantive conclusion. On the
2,000-document Poliblog fixture at K = 20, the committed gold now checks the whole
model against R, not just the topic-word matrix. Aligning topica's topics to R's
by β cosine, the agreement is:

| Quantity | topica vs R |
|---|---:|
| topic-word β (aligned cosine) | 0.975 |
| doc-topic θ (mean per-doc cosine) | 0.967 |
| topic correlation (Σ, off-diagonal cosine) | 0.983 |
| prevalence effect on `rating` (Pearson r across topics) | 0.977 |
| prevalence-effect sign agreement | 17 / 20 topics |

γ and Σ are not compared in the raw (K − 1) reference space — two independent
fits' relabeled topics do not align there — but through their interpretable
K-space forms: the per-topic prevalence effect (γ) and the topic-correlation
matrix (Σ). The three topics whose effect sign differs are the bistable ones the
two optimizers split, not a systematic offset. `estimate_effect` computes its
method-of-composition standard errors with R `estimateEffect`'s default **Global**
uncertainty (one shared topic covariance across documents); pass
`uncertainty="local"` for the per-document variational covariance, or `"none"`
for OLS on the point estimate. The [Poliblog](../examples/poliblog.md) worked
example refits the canonical `stm` vignette end to end.

These numbers are gated offline (no R at test time) in
[`stm_gold.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_gold.py):
the R reference fit, the exact corpus, and the design matrix are frozen in the
committed fixture.

The smaller Gadarian survey corpus (339 documents, K = 3) is a deliberately
harder, more multimodal case: with so few short open-ended responses, R itself
self-agrees only to a cosine of 0.81, and topica lands at 0.51 — still inside the
spread of R's own Spectral-versus-Random runs (0.62). It is the stress test, not
the headline; see [`stm_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_r_compare.py).

## Speed

On matched iterations from a spectral start, topica fits the same model 3–22×
faster than R `stm`, single-threaded, and more with multiple cores, since topica
parallelizes the variational E-step while `stm` is single-threaded. The full
table is on the [benchmarks page](../benchmarks.md).
