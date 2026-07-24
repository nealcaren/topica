# STS: reference-fidelity fitting profiles

`STS` is topica's Structural Topic and Sentiment-Discourse model (Chen & Mankad
2024), which extends STM with a per-document, per-topic continuous
sentiment-discourse latent. topica implements the authors' core likelihood,
gradient, and Hessian. This page documents how topica's fitting defaults relate
to the two reference implementations and how to select a reference-fidelity
configuration.

## Two references, one model

There are two R implementations, and they differ in their fitting defaults:

- The **paper replication code** (Chen & Mankad 2024, the *Management Science*
  supplement) defaults to `estimation = "lasso"` and applies no damping to the
  topic-word coefficients.
- The **CRAN `sts` package** (1.4) defaults to `kappaEstimation = "adjusted"` and
  damps each topic-word update against the previous one.

Both share the same initialization: an STM anchor-word spectral fit
(`max.em.its = 0`) supplies the initial per-document `eta`, the topic-block prior
covariance (`invsigma`), and the initial topic-word matrix, with the sentiment
block given prior variance 20.

## What topica exposes

The topica default is **topica-native, not a reference target**: a fast
ridge-penalized Poisson κ estimator (`kappa_estimation="ridge"`) with topica's
own spectral initialization. It is deterministic and supported, and it is the
right choice when you want a fast, stable fit and are not trying to reproduce a
specific R run.

To reproduce an R run, select a `reference` profile:

```python
import topica

model = topica.STS(num_topics=20, seed=1)

# CRAN sts (kappaEstimation = "adjusted"):
model.fit(docs, sentiment_seed=stars, prevalence=X, reference="cran")

# Chen & Mankad 2024 replication code (estimation = "lasso"):
model.fit(docs, sentiment_seed=stars, prevalence=X, reference="paper")
```

Both profiles use the reference-style initialization (STM-derived spectral `eta`,
sentiment-block prior variance 20). They differ exactly as the two references do:

| `reference` | κ estimator | κ aggregation | κ damping |
|-------------|-------------|---------------|-----------|
| `"none"` (default) | as `kappa_estimation` (ridge) | — | none |
| `"paper"` | `"lasso"` | plain group mean of `α^(s)` | none |
| `"cran"` | `"adjusted"` | φ-mass-weighted group mean of `α^(s)` | half-step `(κ_new + κ_old)/2` |

The κ estimators are also available standalone (`kappa_estimation="lasso"` or
`"adjusted"`) without the profile's initialization and damping.

## The "adjusted" estimator

CRAN's `"adjusted"` differs from `"lasso"` only in how the aggregated sentiment
design column is formed. `"lasso"` uses the plain group mean of `α^(s)`;
`"adjusted"` uses a φ-mass-weighted mean, weighting each document's `α^(s)_{d,t}`
by its expected topic-`t` token mass:

$$\bar{\alpha}^{(s)}_{g,t} = \frac{\sum_{d \in g} \alpha^{(s)}_{d,t}\,\phi_{d,t}}{\sum_{d \in g} \phi_{d,t}}, \qquad \phi_{d,t} = \sum_v \phi_{d,v,t}.$$

This reproduces the `weighted_alpha <- alpha_s * softmax(log phiD)` reweighting in
CRAN `opt.kappa.R` (a softmax over log topic mass is a mass-proportional weight).

!!! note "A note on the CRAN `opt.kappa.R` inner loop"
    CRAN's `"adjusted"` branch wraps the reweighting in a `while (kk <=
    max_inner_iter)` loop with `max_inner_iter <- 10`, but the loop never
    reassigns `coef` inside itself, so `delta` is identically zero and it breaks
    on the first pass. The effective computation is a single reweighting from
    `coef = 0` followed by one L1/AIC Poisson fit. topica reproduces the
    *effective* behavior — a single φ-mass-weighted aggregation — which is what
    every published CRAN `sts` result actually computes.

## Evidence

The κ estimator is an L1-penalized Poisson regression over a λ path with
AIC-selected penalty, the same kernel CRAN `opt.kappa.R` delegates to
`glmnet::glmnet(..., family="poisson", alpha=1)`.
[`parity/sts_kappa_glmnet.py`](https://github.com/nealcaren/topica/blob/main/parity/sts_kappa_glmnet.py)
validates topica's native solver head-to-head with R glmnet on the actual STS κ
design (block-diagonal topic dummies plus sentiment slopes). On that design the
two agree essentially bit-for-bit (aligned cosine `> 0.999`), with the occasional
case where AIC selection lands on an adjacent λ. topica's solver is an
independent coordinate-descent implementation, so this is a "tracks glmnet
closely" result, not a bit-identical one. The script shells out to `Rscript` with
`glmnet` and skips cleanly when they are unavailable.

The φ-mass-weighted aggregation and the half-step damping are exact ports of the
CRAN arithmetic, checked by Rust unit tests. The reference initialization is a
documented statistically-equivalent substitute for STM's spectral `eta` /
`invsigma`, not a byte-identical STM port.

### End-to-end vs the R `sts` package

[`parity/sts_r_package_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/sts_r_package_compare.py)
fits the **live R `sts` package** and topica's STS (`reference="cran"`) on the
same poliblog corpus — the one that ships with `stm` (`data(poliblog5k)`), so the
comparison needs no external data — and reports the aligned topic-word cosine at
mean sentiment. Because the two use independent initializations, this is a
statistical check (same topics up to alignment), like the STM/CTM R parity, not a
bit-identical one. A companion script,
[`parity/sts_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/sts_r_compare.py),
compares instead against the authors' *frozen* published fit and needs the
replication package on disk. Both skip cleanly when Rscript or the R packages are
unavailable.

That live comparison is also **frozen into a committed gold** so it gates CI
without an R toolchain.
[`parity/sts_cran_gold.py`](https://github.com/nealcaren/topica/blob/main/parity/sts_cran_gold.py)
records R `sts`'s topic-word distribution at mean sentiment (the adjusted profile,
two `stmSeed`s for a self-consistency floor) plus the exact tokenized corpus on a
fixed 300-doc poliblog subsample. `tests/test_sts_cran_gold.py` refits topica with
`reference="cran"` on that frozen corpus and asserts its topic-word distribution
clears a bar of R's two-seed cosine floor minus a margin. R's adjusted-profile fit
is near-identical across seeds (self cosine ~0.998), so the bar lands at ~0.80 —
an externally calibrated cross-implementation regression threshold (the same floor
the live script uses), which a shuffled-β negative control falls well below. No
Rscript is touched at test time.

## Scope and honest limits

What is validated and committed is: the κ-solver kernel (against R glmnet), the
exact aggregation and damping arithmetic (Rust unit tests), an end-to-end
topic-recovery comparison against the live R `sts` package on the shipped
poliblog corpus, and a committed offline gold of that comparison that gates CI
without R. The `reference` profiles are R-aligned configurations whose
numerical kernel is validated and whose topics track R `sts`; they are not a
claim of byte-for-byte reproduction of a specific R fit, since the two engines
initialize independently and the reference init is a documented
statistically-equivalent substitute for STM's spectral `eta` / `invsigma`.
