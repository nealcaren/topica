# Sentiment as a covariate: what complaints and praise are about

When the thing you want to explain is *valence* — what people talk about when they
condemn versus praise — a covariate topic model earns its keep. This example fits a
[`DMR`](../api/models.md) (Dirichlet-Multinomial Regression: LDA whose topic prior
is a regression on document covariates) with a review's star rating as a single
ordinal covariate, and reads off which topics rise and fall along the one-to-five
scale, each with a confidence interval. DMR is the lighter-weight sibling of
[`STM`](congress.md) when your covariate acts on prevalence alone.

!!! info "Focus of this example"
    **Covariate LDA** (`DMR`) with an **ordinal** covariate · keeping negation with
    `SENTIMENT_STOPWORDS` · reading complaint vs praise topics with CIs. For a
    two-covariate STM (group × time) see [Congress](congress.md).

    Data: [`load_reviews`](../api/datasets.md#topica.datasets.load_reviews) — 1,500
    Yelp reviews, 300 at each star rating, shipped in the wheel (loads offline).

## 1. Keep the words that carry sentiment

The default stoplist would delete `not`, `no`, and `very` — which is fatal here,
because "not clean" is the opposite of "clean". Pass `SENTIMENT_STOPWORDS` instead,
and enter the rating as a single centered ordinal column so its coefficient reads
as a per-star slope.

```python
import numpy as np
import topica

df = topica.datasets.load_reviews()
corpus = topica.from_dataframe(
    df, text_col="text",
    stopwords=topica.data.SENTIMENT_STOPWORDS,   # keeps not / no / very / too
    min_doc_freq=5, max_doc_fraction=0.5,
)
X = (corpus.metadata["stars"].to_numpy(float) - 3.0).reshape(-1, 1)  # centered 1..5
model = topica.DMR(num_topics=12, seed=13).fit(
    corpus.documents(), X, feature_names=["stars"]
)
```

## 2. Estimate the star effect on every topic

`estimate_effect` propagates topic-estimation uncertainty into the slopes. A
negative slope is a **complaint** topic (more prevalent in low-star reviews); a
positive slope is **praise**.

```python
effects = topica.effects.estimate_effect(model.doc_topic, X, feature_names=["stars"],
                                 nsims=60, seed=0)
for eff in effects:
    star = eff.effect_of("stars")     # dict: coef, se, z, ci_low, ci_high, pvalue
```

| | Topic (FREX) | Slope per star | 95% CI |
|---|---|:---:|:---:|
| **Complaint** | us, took, said, asked, wrong | **−0.041** | [−0.044, −0.038] |
| | better, nothing, quality, bad | −0.019 | [−0.022, −0.016] |
| | work, reviews, car, management | −0.016 | [−0.019, −0.014] |
| **Praise** | friendly, favorite, highly, amazing, excellent | **+0.056** | [+0.053, +0.058] |
| | beer, sushi, atmosphere, music, family | +0.015 | [+0.013, +0.018] |
| | store, shop, home, help | +0.014 | [+0.012, +0.017] |

Every interval excludes zero. The split is substantive, not merely tonal:
complaint is a **process narrative** — a first-person account of being ignored or
misled (`us, took, asked, wrong`) — while the strongest praise topic is an
**evaluative register** (`amazing, excellent, highly`) rather than any particular
experience. The steepest complaint topic loses about four points of prevalence per
star; the praise register gains almost six.

## When DMR instead of STM

Reach for `DMR` when your covariates shift topic *prevalence* and you do not need
STM's content covariates, topic correlations, or its variational content model —
DMR is a faster Gibbs sampler with the same prevalence-regression idea. When the
covariate should also change the *words within* a topic, or you want topic
correlations, use [`STM`](congress.md).

## Reproduce

`seed=13`; raise `iters` and `nsims` for a publication run. The single most
important choice on this corpus is the stoplist: `SENTIMENT_STOPWORDS` over the
default, so negation survives to carry the signal.
