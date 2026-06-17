# 2. Pick the right model

Before `K`, choose the *model*. This is a substantive decision. It encodes what
you think generates the text and what you want to learn, and reviewers will ask
you to justify it. Pick the simplest model that matches your question and your
data structure.

The decision tree below is keyed on your *question*. Two practical axes narrow it
further, both filterable with `topica.list_models(...)` (see the
[roster](../guides/models.md#the-roster)):

- **What you bring.** Beyond raw text, do you have document **metadata**
  (covariates), a few **seed** words per topic, document **labels**, document
  **embeddings**, or **time** stamps? Each points to a group:
  `list_models(brings="metadata")`, `brings="embeddings"`, and so on.
- **Reproducibility.** Models with a deterministic fit (CTM, STM, STS — spectral
  anchor-word init — and NMF, whose multiplicative updates converge away from the
  init) are `bit-exact` (identical regardless of seed or thread count); models
  that draw a seeded random state — the samplers, the randomized-SVD LSA, and the
  embedding/VAE/optimal-transport models — are `seed-reproducible` (identical
  once you fix the seed *and* the thread count); an LLM-backed model is
  `llm-bounded`. If seed-free replication is a requirement, prefer a `bit-exact`
  model: `topica.list_models(determinism="bit-exact")`.

## First: is a topic model even the right tool?

!!! note "Good fit"
    - Exploratory: *what themes exist in this corpus?*
    - Large collections where manual reading is infeasible.
    - You want to **discover** structure, not impose categories.
    - You need to track theme prevalence over time or across groups.

!!! warning "Poor fit"
    - Purely **confirmatory** counting against a fixed code list, where you do not
      need topics at all: a dictionary or supervised classifier is simpler. If you
      want to *measure* pre-theorized concepts as topics, a guided model fits, and
      the framing below explains where.
    - Very short documents (< ~50 words) → use a [short-text model](../guides/short-text.md),
      not standard LDA.
    - Highly technical or formulaic text, or a need for crisp category boundaries.

## Start here: do you already know your topics?

One question settles most of the choice. Do you know, from theory, the topics you
are looking for, or are you trying to find out what is in the corpus?

- **No. You want to discover what is there.** Impose as little structure as
  possible and read what emerges. Use [`LDA`](../guides/models.md#lda), or
  [`HDP`](../guides/models.md#hdp) if you would rather not fix the number of
  topics in advance.
- **Yes. You have concepts and want to measure them.** Name each topic with a few
  seed words and let the model anchor a topic to them. This is the guided path,
  [`KeyATM`](../guides/guided.md) or [`SeededLDA`](../guides/guided.md), and it
  buys measurement validity and reproducibility a post-hoc reading of LDA topics
  cannot. From there you can regress prevalence on covariates or model it over
  time within the same family.
- **Yes, and you also care how the same concept is worded differently across
  groups.** Use [`STM`](../guides/models.md#stm) with a content covariate. Modeling
  the topic-word distribution as a function of a group is the one thing the guided
  models cannot do.

Discovery and measurement are not a ranking. They answer different questions, and
strong papers often run a guided model for the concepts they theorized and an
unsupervised model as a robustness check on what they might have missed.

## Match the model to your question

| Your question / data | Model | Why |
|----------------------|-------|-----|
| "What themes are here?" (baseline) | **`LDA`** | Standard, fast, well understood |
| "Does theme prevalence depend on metadata?" | **`STM`** | Prevalence covariates with valid effect estimation; the social-science default |
| "Is the same theme worded differently by group?" | **`STM`** (content) / **`SAGE`** | Content covariates on the topic-word distribution |
| "Do themes co-occur?" | **`CTM`** / `STM` | Logistic-normal allows topic correlation |
| "How many themes are there?" | **`HDP`** | Infers `K` nonparametrically (a check on your `K`) |
| "How does theme *vocabulary* drift over time?" | **`DTM`** | Topics evolve across ordered time slices |
| "Do themes predict an outcome?" | **`SupervisedLDA`** / `DMR` | Response- or covariate-conditioned topics |
| "I already know the themes I expect" | **`KeyATM`**, **`SeededLDA`** | Seed words steer named topics for better validity and reproducibility ([guided topics](../guides/guided.md)) |
| Tweets / survey answers / headlines | **`GSDMM`**, **`PT`** | Built for short, sparse documents |

!!! tip "How far keyATM reaches, and where it stops"
    Within measurement, [`KeyATM`](../guides/guided.md) is close to a complete
    toolkit on its own: keyword-anchored topics, prevalence regression on
    covariates, and a change-point model for time trends, all in one family
    (validated against the R package in the [replication](../replications/keyatm-dynamic.md)).
    Its boundary is its premise. Because it needs keywords, it does not discover
    unknown topics, so reach for `LDA` or `HDP` there. And it models how *much*
    each group discusses a topic, not *how* they word it, so reach for `STM` or
    `SAGE` when content is the question.

## Why STM is the social-science default

For most published social-science work the answer is the **Structural Topic
Model**. It is LDA's correlated cousin *plus* a regression layer, so a single fit
gives you topics, their correlations, and how covariates move them, with
[proper uncertainty](effects.md). The R `stm` package made it the field standard.
topica gives you the same model (spectral init, prevalence and content
covariates, `estimateEffect`, `searchK`, FREX) in Python, faster, with
[clustered standard errors](effects.md) and GLM links on top.

## Justify it in the paper

State the model and *why it fits your design* in one or two sentences:

> Because our research question concerns how immigration framing differs by party
> and shifts across the campaign, we use the Structural Topic Model (STM), which
> lets topic prevalence depend on covariates (party, week) and provides valid
> uncertainty for those effects via the method of composition.

A nonparametric (`HDP`) or alternative-model robustness check pre-empts the
"why this model?" objection: "the substantive topics were stable when we instead
used LDA / a different `K`."

→ Next: [Choose and justify K](choosing-k.md).
