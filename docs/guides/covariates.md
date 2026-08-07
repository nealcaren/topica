# Covariates & STM

The Structural Topic Model lets topics depend on document metadata in two ways:
**prevalence** (how much a document discusses each topic) and **content** (how a
topic is worded). For the publication-grade version of this workflow, with proper
uncertainty and clustered errors, see [Measure effects properly](../publishing/effects.md).

!!! tip "Importing the covariate helpers"
    All of the design-matrix and effect helpers are top-level: `topica.one_hot`,
    `topica.design_matrix`, `topica.spline`, `topica.interaction`,
    `topica.estimate_effect`, and `topica.posterior_theta_samples`. That is the
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
corpus = topica.from_dataframe(df, text_col="text")     # metadata kept aligned

# Design matrix from a formula (needs the optional topica[formula] extra).
# corpus.metadata is the surviving rows, already aligned to the documents.
X, names = topica.design_matrix("~ party + spline(year, df=3)", corpus.metadata)

# Pick K at the coherence/exclusivity frontier (a knee, not a coherence max,
# which would just return the smallest K), then fit at that K.
scan = topica.search_k(corpus, [10, 20, 30], model="stm", prevalence=X, iters=200)
model = topica.STM(num_topics=scan.best_k(), seed=1)
model.fit(corpus, prevalence=X, prevalence_names=names)

# Effects with method-of-composition uncertainty, as a tidy long table.
draws = topica.posterior_theta_samples(model, nsims=50, seed=0)
effects = topica.estimate_effect(draws, X, feature_names=names)
table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
```

If you would rather not add the `formulaic` dependency, replace `design_matrix`
with hand-built blocks: `X, names = topica.one_hot(df["party"])` combined with
`topica.spline` / `topica.interaction` via `numpy.hstack`.

## Prevalence covariates

```python
import topica

X, names = topica.one_hot(party)                      # design matrix + column names
model = topica.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X, prevalence_names=names)

model.prevalence_effects        # learned γ
topica.topic_correlation(model.doc_topic)
```

## Content covariates

A content model makes the topic-word distribution vary by group (the SAGE
mechanism), so the same topic is phrased differently across, say, conservative
and liberal sources:

```python
model = topica.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X, content=source, content_names=groups)

model.topic_word_by_group        # per-group β
# words that most distinguish how a topic is worded across two groups:
# model.word_contrast(topic, "liberal", "conservative")
```

`topica.content` reads that per-group tensor for STM and SAGE alike:
`topic_polarization(model)` is the per-topic Jensen-Shannon divergence across
groups (how differently the groups word a topic), `group_exclusivity(model)`
checks a topic stays distinctive in every group's sub-vocabulary, and
`split_topics(model, content)` flags near-duplicate topics pulled apart by group
prevalence — the sign that one discourse has *fragmented* into parallel
group-topics instead of living within a topic. STS fits the same functions:
its continuous sentiment axis is discretized into negative/neutral/positive
groups (pass `levels=` to change the sentiment cut points).

```python
import topica
pol = topica.content.topic_polarization(model)   # (K,) in [0, 1]
```

Whether a group difference lands **within** a topic or splits into separate
topics is something you control. The `content_prior_var` prior on the STM/SAGE
group deviations is the dial: raise it and the content model absorbs more group
vocabulary within a topic; lower it and the difference is suppressed toward the
shared baseline. Fragmentation, by contrast, is mostly a *design* choice — putting
the grouping variable in the **prevalence** design (not just content) rewards the
model for spending whole topics on a group. For wording that evolves over ordered
time, an STM `content_time` covariate crosses the group with the period and
`topica.content.content_trajectory` / `content_divergence` read that surface;
the party-platforms example works this through end to end.

## Estimating effects

Regress topic proportions on covariates with well-calibrated uncertainty, using the method
of composition, optionally with clustered standard errors, survey weights, and
GLM links:

```python
import pandas as pd
import topica

draws = topica.posterior_theta_samples(model, nsims=50, seed=0)
effects = topica.estimate_effect(
    draws, X, feature_names=names,
    cluster=source_id,     # cluster-robust SEs for nested data
    weights=survey_weight,  # weighted least squares (e.g. survey weights)
    # link="logit",        # keep predictions in [0, 1]
)

# One tidy row per (topic, feature): coef, se, z, ci_low, ci_high, r_squared
table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
```

Build non-linear and interaction terms with `topica.spline` and
`topica.interaction`. Full detail and the journal-grade treatment are in the
[Publishing](../publishing/effects.md) track.

### On a plain LDA model (not only STM)

You do **not** have to switch to `STM` to get covariate effects with honest
uncertainty. `estimate_effect` takes any fitted model (or its `doc_topic`) — a plain
`LDA` works — and propagates topic-estimation uncertainty via the method of
composition. STM additionally embeds the prevalence covariates in the prior *during*
estimation (so the topics themselves are informed by the covariates); reach for it
when that is the goal, but for "how does topic prevalence differ by group, with a
CI?" on an already-fitted LDA, `estimate_effect(model, X=..., corpus=corpus)` is the
direct path.

```python
model = topica.LDA(num_topics=20, seed=1)
model.fit(corpus, iters=1000)

X, names = topica.one_hot(corpus.metadata["rating"], prefix="rating_")
effects = topica.estimate_effect(model, X=X, feature_names=names, corpus=corpus, nsims=50)
```

**Read effects by name, not by position.** `add_intercept=True` is the default, so
each result's `feature_names` is `["intercept", "rating_Liberal", ...]` and `coef[0]`
is the baseline **intercept**, not your covariate. Reading `coef[0]` as "the effect"
is a common way to publish the wrong number. Use the name-keyed accessors:

```python
e = effects[topic]
e.effect_of("rating_Liberal")   # {'coef':..., 'se':..., 'z':..., 'pvalue':..., 'ci_low':..., 'ci_high':...}
e.by_feature                    # every covariate keyed by name
e.to_frame()                    # tidy rows with a named `feature` column and a pvalue
```

`e.pvalue` (two-sided, from the Wald `z`) is available for significance reporting.

### Random intercepts for nested data

When documents are nested in units — states, outlets, authors — whose baseline
topic level varies, add an lme4-style random intercept with `random="(1 | group)"`
(where `group` is a column of `data`). `estimate_effect` then fits a mixed model
(the fixed-effect design plus a per-group random intercept) by REML for each
posterior draw and Rubin-pools the fixed effects, so the between-unit variation is
absorbed rather than inflating — or hiding in — the fixed-effect standard errors.
The estimated group and residual standard deviations come back on
`TopicEffect.varcomp`. This matches faSTM's `estimateEffect(1:K ~ x + (1 | group))`
and reproduces `lme4::lmer`'s fixed effects exactly.

```python
effects = topica.estimate_effect(
    draws, formula="~ party", data=meta, random="(1 | state)",
)
effects[0].varcomp   # {"state": sd_between, "residual": sd_within}
```

Only a random *intercept* is supported (not random slopes), with the identity link
and without `cluster`/`weights`.

### Average marginal effects

When the design has splines or interactions, no single coefficient is "the
effect" of a covariate. `topica.average_marginal_effects` (alias `topica.ame`)
reports the average change in a topic's proportion per unit of a covariate —
the average derivative for a continuous covariate, or the average
level-vs-reference contrast for a factor — averaged over the observed documents,
with standard errors that propagate the topic-estimation uncertainty:

```python
# Average marginal effect of `year` on every topic (year enters via a spline).
ame = topica.average_marginal_effects(
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

`topica.predicted_prevalence` computes predicted topic prevalence at chosen
covariate values with simulation-based credible intervals — the direct
counterpart of R `stm`'s `plot.estimateEffect`. Three modes mirror `stm`'s
`method` argument:

```python
import topica

# Point grid: predicted prevalence when party is "D" vs "R"
pp = topica.predicted_prevalence(
    model,
    formula="~ party + year",
    data=meta,
    at={"party": ["D", "R"]},
)
for result in pp:
    print(result.topic_name, result.estimate, result.ci_low, result.ci_high)

# Continuous sweep: prevalence as year varies, other covariates at their means
pp = topica.predicted_prevalence(
    model, formula="~ party + year", data=meta,
    continuous="year",
)

# Contrast: difference in prevalence between two covariate settings
pp = topica.predicted_prevalence(
    model, formula="~ party", data=meta,
    contrast={"party": ["D", "R"]},
)
```

## Permutation test for binary covariates

`topica.permutation_test` assesses whether a binary covariate genuinely shifts
topic prevalence, using permutation resampling rather than parametric
assumptions:

```python
results = topica.permutation_test(
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
model = topica.STM(num_topics=20, seed=1)
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

## Covariate effects on meaning: embedding regression

The covariates above act on topic *prevalence* (how much a group discusses a
theme) and, through content covariates, on the *words* a group uses for a topic.
Neither sees a difference in *meaning* that is not a word swap. When two groups
use the same words but frame a term differently, or use different words for the
same idea, the difference lives in a pretrained embedding space, not in counts.

`embedding_regression` implements the à la carte on text (conText) embedding
regression of Rodriguez, Spirling and Stewart (2023). It asks the native
embedding-and-covariate question: *does a covariate shift what a word or theme
means?* Its canonical use is "do Republicans and Democrats mean different things
by `immigration`?". It is a **text-as-data analysis tool, not a topic model** (it
produces no topics and needs no `enable_experimental`), and it is a drop-in for the
R `conText` package: `parity/embedding_regression_context.py` checks topica against
conText on conText's own bundled data and reproduces its à la carte embeddings,
squared coefficient norm, and HC1-deflated norm to numerical precision.

The method regresses à la carte embeddings on your covariates. An à la carte
embedding of a focal term is the (transformed) count-weighted mean of its context
words' pretrained vectors, so it captures how *that* mention is used. For a focal
term the unit of analysis is one row per mention (`aggregate="instance"`, conText's
default); with `target=None` it embeds whole documents. The effect size for a
covariate is the squared Euclidean norm of its coefficient, deflated for
small-sample bias (`statistic="squared_deflated"`, the default and conText's
headline: a value near zero means no detectable shift), with a permutation p-value
and a bootstrap confidence interval.

```python
# You bring pretrained word embeddings as {word: vector} (GloVe, word2vec, ...).
fit = topica.embedding_regression(
    docs,                       # tokenized documents (word order matters)
    covariates=party,           # numeric (N,)/(N, p), or category labels (dummy-coded)
    pre_trained=glove,          # {word: vector} or (matrix, vocab)
    names=["party"],
    target="immigration",       # focal term; omit to embed whole documents
    window=6,
    transform="estimate",       # learn the ALC matrix, pass one, or "additive" (identity)
    permutations=100,
    bootstrap=100,
)

print(fit.summary())            # covariate, effect size, CI, permutation p
fit.nearest_neighbors({"party_R": 1}, n=10)          # what "immigration" means to R
fit.nns_ratio({"party_R": 1}, {"party_R": 0}, n=10)  # R-vs-D contrast
```

Categorical covariates are dummy-coded (first level dropped as the reference), so a
`party` column of `"D"`/`"R"` becomes a `party_R` coefficient. The
`nearest_neighbors` at a covariate value, and the `nns_ratio` contrast between two
values, are how you read *what* the shift is: they rank pretrained vocabulary words
near the predicted embedding, so a partisan split in the meaning of `immigration`
shows up as different neighbor words for each party.

Two notes for conText users. topica follows the method of Rodriguez, Spirling &
Stewart (2023); its inference differs from the current conText package in form
(covariate permutation and a document bootstrap here, vs. residual permutation and a
jackknife there), though the point estimates and deflated norms match. And conText's
published transform matrices (for example `cr_transform`) are the transpose of what
:func:`compute_transform` returns, so pass an external conText matrix as
`transform=cr_transform.T`.

This also sidesteps the trap of putting a document's own embedding on the covariate
side of a topic model (see [Embedding topics](embedding.md)): here the embedding
*is* the outcome being described, and the covariate is external metadata, exactly as
in a regression.
