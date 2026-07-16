# 4. Measure effects properly

Most social-science topic-model papers don't stop at "here are the themes."
They ask **how topic prevalence relates to something**: time, group, treatment,
ideology. Doing this credibly means modeling the relationship *and* reporting
well-calibrated uncertainty.

## Make prevalence depend on covariates: STM

The Structural Topic Model lets a document's topic proportions depend on its
metadata. Fit prevalence as a regression on your covariates:

```python
import topica

X, names = topica.one_hot(party)                     # or build any design matrix
model = topica.models.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X, prevalence_names=names)
```

## Estimate effects with well-calibrated uncertainty

A naive regression of point topic proportions on covariates treats θ as if it
were observed exactly. It isn't. R's `stm` uses the **method of composition**
(Treier & Jackman 2008): draw θ from the model's posterior, regress each draw,
and pool by Rubin's rules so the standard errors include topic-estimation
uncertainty. topica does the same, drawing θ from each document's variational
posterior. (It propagates that per-document θ uncertainty; it does not also
simulate the global parameters β, Σ, and γ, so its pooled standard errors run a
touch smaller than `stm`'s. The inflation over naive point-θ OLS is the part that
matters, and it is there.)

```python
from topica import stm

draws = stm.posterior_theta_samples(model, nsims=50, seed=0)   # (50, D, K)
effects = stm.estimate_effect(draws, X, feature_names=names)
for e in effects:
    d = e.as_dict()
    print(f"Topic {d['topic']}: {names[0]} coef={d[names[0]]['coef']:+.4f} "
          f"z={d[names[0]]['z']:+.2f}")
```

For non-linear time trends and interactions, build the design matrix with
`stm.spline` and `stm.interaction`, the same `~ s(year)` and `~ a*b` you'd write
in R.

`topica.standard_errors` wraps this in one call: it detects the model family,
draws the right posterior for you, and returns the same effects. It also reaches
quantities the manual path doesn't, group prevalence and bootstrap intervals on
the top words, with an alignment-quality flag for the embedding models. See
[Reporting uncertainty](../guides/uncertainty.md).

```python
eff = topica.standard_errors(model, corpus, of="effect",
                             formula="~ party", data=meta, nsims=50)
```

## It is not just for STM

`topica.estimate_effect` regresses *any* model's θ on covariates, and
`topica.by_strata` (mean prevalence by group, with intervals) and
`topica.top_topics` work on any model's θ too.

STM and CTM have a logistic-normal posterior, so `posterior_theta_samples` draws
from it directly. A Gibbs model (LDA, keyATM, SeededLDA, ...) has no such
posterior, but each document's θ still has a Dirichlet conditional given its
length, so `dirichlet_theta_samples` gives you draws to feed the same
method-of-composition machinery:

```python
import numpy as np, topica

model = topica.models.LDA(num_topics=20, seed=1)
model.fit(docs, iters=1000)

lengths = np.array([len(d) for d in docs])
draws = topica.dirichlet_theta_samples(model.doc_topic, lengths, nsims=50, seed=0)
effects = topica.estimate_effect(draws, X, feature_names=names)
```

Pass the point θ (`model.doc_topic`) instead of draws and you get a plain OLS fit
with no uncertainty propagation, which is the right baseline but understates the
standard errors.

## Cluster your standard errors

Text data is almost always **nested**: multiple speeches by the same legislator,
many tweets per user, several articles per outlet, or, if you
[split long documents](corpus.md), many chunks per source document. Ignoring this
nesting understates uncertainty and is a common reason reviewers reject a result.

Pass a `cluster` variable and the standard errors become cluster-robust (CR1):

```python
effects = stm.estimate_effect(
    draws, X, feature_names=names,
    cluster=speaker_id,        # one label per document
)
```

This composes with the method of composition: each posterior draw is clustered,
then the per-draw covariances are Rubin-pooled.

## Keep predictions in bounds: GLM links

Topic proportions live in `[0, 1]`, but OLS on θ can predict values outside that
range. For a bounded model, use a fractional-logit link (Papke & Wooldridge),
fit by quasi-likelihood with robust standard errors:

```python
effects = stm.estimate_effect(draws, X, feature_names=names, link="logit")
# link="log" gives a quasi-Poisson alternative.
```

## Report effects as a table

```python
import pandas as pd
rows = []
for e in effects:
    d = e.as_dict()
    for feat in e.feature_names:
        rows.append({"topic": d["topic"], "term": feat,
                     "estimate": d[feat]["coef"], "se": d[feat]["se"],
                     "z": d[feat]["z"]})
table = pd.DataFrame(rows)
```

Report point estimates, (clustered) standard errors, and confidence intervals,
and say plainly which effects clear conventional thresholds and which don't.

→ Next: [Report and make reproducible](reporting.md).
