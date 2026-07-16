# Covariates & STM

The Structural Topic Model lets topics depend on document metadata in two ways:
**prevalence** (how much a document discusses each topic) and **content** (how a
topic is worded). For the publication-grade version of this workflow, with proper
uncertainty and clustered errors, see [Measure effects properly](../publishing/effects.md).

!!! tip "Importing the covariate helpers"
    All of the design-matrix and effect helpers are top-level: `topica.design.one_hot`,
    `topica.design.design_matrix`, `topica.design.spline`, `topica.design.interaction`,
    `topica.effects.estimate_effect`, and `topica.effects.posterior_theta_samples`. That is the
    canonical path used throughout these docs. The same names are also reachable
    under `topica.stm.*` (they are the identical objects, kept as a compatibility
    alias), but prefer the top-level form.

## End to end: from a DataFrame to effects

The whole covariate workflow in one block: build an aligned corpus from a
DataFrame, turn the metadata into a design matrix with an R-style formula, fit
the STM, and read the effects as a tidy table. Every step uses the canonical
top-level helpers.

```python
import pandas as pd
import topica

# df has columns: text, party, year
corpus = topica.prep.from_dataframe(df, text_col="text")     # metadata kept aligned

# Design matrix from a formula (needs the optional topica[formula] extra).
# corpus.metadata is the surviving rows, already aligned to the documents.
X, names = topica.design.design_matrix("~ party + spline(year, df=3)", corpus.metadata)

# Pick K at the coherence/exclusivity frontier (a knee, not a coherence max,
# which would just return the smallest K), then fit at that K.
scan = topica.select.search_k(corpus, [10, 20, 30], model="stm", prevalence=X, iters=200)
model = topica.models.STM(num_topics=scan.best_k(), seed=1)
model.fit(corpus, prevalence=X, prevalence_names=names)

# Effects with method-of-composition uncertainty, as a tidy long table.
draws = topica.effects.posterior_theta_samples(model, nsims=50, seed=0)
effects = topica.effects.estimate_effect(draws, X, feature_names=names)
table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
```

If you would rather not add the `formulaic` dependency, replace `design_matrix`
with hand-built blocks: `X, names = topica.design.one_hot(df["party"])` combined with
`topica.design.spline` / `topica.design.interaction` via `numpy.hstack`.

## Prevalence covariates

```python
import topica

X, names = topica.design.one_hot(party)                      # design matrix + column names
model = topica.models.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X, prevalence_names=names)

model.prevalence_effects        # learned γ
topica.interpret.topic_correlation(model.doc_topic)
```

## Content covariates

A content model makes the topic-word distribution vary by group (the SAGE
mechanism), so the same topic is phrased differently across, say, conservative
and liberal sources:

```python
model = topica.models.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X, content=source, content_names=groups)

model.topic_word_by_group        # per-group β
# words that most distinguish how a topic is worded across two groups:
# model.word_contrast(topic, "liberal", "conservative")
```

## Estimating effects

Regress topic proportions on covariates with well-calibrated uncertainty, using the method
of composition, optionally with clustered standard errors, survey weights, and
GLM links:

```python
import pandas as pd
import topica

draws = topica.effects.posterior_theta_samples(model, nsims=50, seed=0)
effects = topica.effects.estimate_effect(
    draws, X, feature_names=names,
    cluster=source_id,     # cluster-robust SEs for nested data
    weights=survey_weight,  # weighted least squares (e.g. survey weights)
    # link="logit",        # keep predictions in [0, 1]
)

# One tidy row per (topic, feature): coef, se, z, ci_low, ci_high, r_squared
table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
```

Build non-linear and interaction terms with `topica.design.spline` and
`topica.design.interaction`. Full detail and the journal-grade treatment are in the
[Publishing](../publishing/effects.md) track.

### Average marginal effects

When the design has splines or interactions, no single coefficient is "the
effect" of a covariate. `topica.effects.average_marginal_effects` (alias `topica.effects.ame`)
reports the average change in a topic's proportion per unit of a covariate —
the average derivative for a continuous covariate, or the average
level-vs-reference contrast for a factor — averaged over the observed documents,
with standard errors that propagate the topic-estimation uncertainty:

```python
# Average marginal effect of `year` on every topic (year enters via a spline).
ame = topica.effects.average_marginal_effects(
    model, "year", formula="~ party + spline(year, df=4)", data=meta, nsims=50,
)
ame.to_frame()   # tidy: topic, term, ame, se, ci_low, ci_high
```

!!! warning "Use the same design for `fit` and `estimate_effect`"
    `estimate_effect` regresses on the covariates you pass it, not on whatever
    went into `STM.fit`. Pass the **same** `X` (or the same `formula` + `data`)
    to both, or the coefficients answer a different question than the model.

    Two equivalent ways to supply the design: a prebuilt matrix
    (`estimate_effect(draws, X, feature_names=names)`) or a formula
    (`estimate_effect(draws, data=meta, formula="~ party + year")`, which builds
    `X` for you and needs the optional `topica[formula]` extra). Use the matrix
    form when you already built `X` for `fit` (the common case); use the formula
    form for quick exploration straight from a DataFrame.

## Predicted prevalence

`topica.effects.predicted_prevalence` computes predicted topic prevalence at chosen
covariate values with simulation-based credible intervals — the direct
counterpart of R `stm`'s `plot.estimateEffect`. Three modes mirror `stm`'s
`method` argument:

```python
import topica

# Point grid: predicted prevalence when party is "D" vs "R"
pp = topica.effects.predicted_prevalence(
    model,
    formula="~ party + year",
    data=meta,
    at={"party": ["D", "R"]},
)
for result in pp:
    print(result.topic_name, result.estimate, result.ci_low, result.ci_high)

# Continuous sweep: prevalence as year varies, other covariates at their means
pp = topica.effects.predicted_prevalence(
    model, formula="~ party + year", data=meta,
    continuous="year",
)

# Contrast: difference in prevalence between two covariate settings
pp = topica.effects.predicted_prevalence(
    model, formula="~ party", data=meta,
    contrast={"party": ["D", "R"]},
)
```

## Permutation test for binary covariates

`topica.effects.permutation_test` assesses whether a binary covariate genuinely shifts
topic prevalence, using permutation resampling rather than parametric
assumptions:

```python
results = topica.effects.permutation_test(
    model, corpus=docs, covariate=treated,   # treated: 0/1 array
    n_perm=100, seed=0,
)
for r in results:
    print(f"Topic {r.topic_name}: observed={r.observed:.3f}  p={r.pvalue:.3f}")
```

Each `PermutationResult` carries the observed covariate effect, the full null
distribution (`r.null`), and a two-sided p-value.

## L1/elastic-net prior for high-dimensional designs

When the prevalence design matrix has many columns (many dummies, interaction
terms, or a wide feature matrix), add `gamma_prior="l1"` to `STM.fit` to
penalize the prevalence coefficients:

```python
model = topica.models.STM(num_topics=20, seed=1)
model.fit(
    docs, prevalence=X, prevalence_names=names,
    gamma_prior="l1",     # elastic-net with full L1 (lasso)
    gamma_enet=1.0,       # alpha=1.0 is pure L1; 0 < alpha < 1 mixes L2
)
```

The default `gamma_prior="pooled"` uses the OLS pooled regression from the
original STM paper. Use `"l1"` when p (number of prevalence covariates)
approaches or exceeds the number of documents.

## Choosing K for STM

Use `search_k`, the coherence×exclusivity frontier, and an `HDP` sanity check.
See [Choose and justify K](../publishing/choosing-k.md).
