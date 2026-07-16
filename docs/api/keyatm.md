# keyATM toolkit

keyATM-specific workflow helpers live in `topica.keyatm`. The general post-hoc
diagnostics (top words, representative documents, coherence, pyLDAvis, covariate
effects, …) are model-agnostic and work on a fitted `KeyATM` directly: see the
[Diagnostics](diagnostics.md) and [STM toolkit](stm.md) pages.

The fitted model also exposes the convergence trace keyATM's `plot_modelfit`
reports, as `KeyATM.log_likelihood_history` — a list of `(iteration, per-token
log-likelihood)` pairs (perplexity is `exp(-log_likelihood)`). The same trace is
available in the cross-model form as `KeyATM.fit_history`. Passing
`convergence_tol` to `fit` (default `0.0`, disabled) opts into early stopping on
that trace: the Gibbs run halts once the relative change in the recorded
log-likelihood drops below the tolerance, and `KeyATM.converged` reports whether
it did. The base, covariate, and dynamic Gibbs backends support it; the CVB0
backend (`sampler="cvb0"`) keeps no trace and never early-stops.

::: topica.keyatm.top_topics

::: topica.keyatm.by_strata

::: topica.keyatm.visualize_keywords

::: topica.keyatm.refine_keywords

## Dynamic model: time-trend credible intervals

Per-period topic prevalence with credible bands from the dynamic keyATM
posterior's retained MCMC theta draws.

::: topica.effects.time_prevalence_ci
