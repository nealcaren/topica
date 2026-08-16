# STM toolkit

The structural / covariate operations live in `topica.stm`. The general
post-hoc diagnostics (labeling, alignment, pyLDAvis, …) are on the
[Diagnostics](diagnostics.md) page.

::: topica.effects.standard_errors

::: topica.stm.estimate_effect

::: topica.effects_across_k

::: topica.effects_across_seeds

::: topica.RobustnessResult

::: topica.stm.average_marginal_effects

::: topica.stm.MarginalEffect

::: topica.stm.AverageMarginalEffects

::: topica.effects.dirichlet_theta_samples

::: topica.stm.posterior_theta_samples

::: topica.effects.model_family

::: topica.stm.spline

::: topica.stm.interaction

## Predicted prevalence

Compute predicted topic prevalence at chosen covariate values, with
simulation-based credible intervals — the model-agnostic counterpart of
R `stm`'s `plot.estimateEffect`.

::: topica.effects.predicted_prevalence

::: topica.effects.PredictedPrevalence

## Permutation test

Distribution-free test of whether a binary prevalence covariate genuinely
shifts topic prevalence, or whether the association could arise by chance.

::: topica.effects.permutation_test

::: topica.effects.PermutationResult

## Per-group prevalence with credible bands

Model-neutral per-group topic prevalence with posterior credible intervals
drawn from the model's retained MCMC theta draws (or the logistic-normal
posterior for STM/CTM).

::: topica.prevalence_ci

## Covariate-aware held-out inference

Infer topic proportions for new documents using a fitted STM's prevalence
model, setting the per-document prior from `mu_d = X_d gamma`.

::: topica.stm.transform

Map new token lists onto the fitted vocabulary before calling `transform`,
dropping any out-of-vocabulary tokens.

::: topica.design.align_corpus

## Model selection at fixed K

Run multiple initializations at a fixed K and compare candidates on the
coherence-exclusivity frontier — the analogue of R `stm`'s `selectModel`.

::: topica.select.select_model

::: topica.select.SelectModelResult

Visualize the coherence-versus-exclusivity scatter across candidate runs.

::: topica.select.plot_models
