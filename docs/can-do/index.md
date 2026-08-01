# What you can do

topica is a general topic-modeling toolkit. This section tours what it does.
If your goal is a publishable analysis, pair it with
[Publishing in a journal](../publishing/index.md).

## Choose an approach

Arrive with a research goal, leave with a model. Find your goal in the left
column. Every model is validated against a reference implementation, so these
are the fastest defensible defaults rather than the only valid choices; the
[full roster](../guides/models.md) lists the deep bench for each goal.

<!-- BEGIN CHOOSER (generated from topica.registry CHOOSER; edit registry.py, not this block) -->

| If your goal is… | Start with | Also consider | First calls | Note |
|---|---|---|---|---|
| Explore themes with no prior structure | `LDA` | `NMF` | `search_k()`, `topic_table()` | The default first pass. `NMF` is a fast, deterministic alternative. |
| Relate topics to metadata (author, date, party) | `STM` | `DMR` | `estimate_effect()`, `one_hot()`, `spline()` | `STM` gives covariate effects with uncertainty; `DMR` is a lighter Gibbs prior. |
| Measure concepts you can name in advance | `KeyATM` | `SeededLDA` | `fit(docs, keywords=…)`, `.keywords` | Anchor named topics with a few seed words each. |
| Very short documents: tweets, headlines, survey answers | `GSDMM` | `PT` | `fit()` | One topic per document; standard LDA over-fragments short text. |
| Cluster by meaning using embeddings | `BERTopic` | `ETM` | `fit(docs, embeddings=…)` | Clustering, not a posterior: topic-proportion uncertainty and effect estimation behave differently than the models above. |
| How tone or sentiment varies with metadata | `STS` | — | `estimate_effect()` | A heavier, specialized model; reach for it when sentiment-discourse is the question, not general themes. |
<!-- END CHOOSER -->

In code, `topica.list_models(tier=1)` returns the recommended set and
`list_models(group=…, brings=…)` filters the rest.

## Tour

<div class="grid cards" markdown>

- :material-shape: **[The models](../guides/models.md)**

    More than 40 models, from LDA through STM and its STS
    sentiment-discourse extension, HDP, dynamic and supervised topics, to
    short-text and embedding-based models, all with one consistent API.

- :material-broom: **[Preprocessing](../guides/preprocessing.md)**

    Tokenize, build a `Corpus`, prune the vocabulary, detect phrases, and split
    long documents while preserving metadata.

- :material-chart-bell-curve: **[Covariates & STM](../guides/covariates.md)**

    Relate topics to document metadata: prevalence and content covariates,
    effect estimation, clustered SEs, GLM links.

- :material-check-decagram: **[Diagnostics & validation](../guides/diagnostics.md)**

    Coherence, exclusivity, intrusion tests, stability, alignment, ensemble
    consensus across runs, FREX labels, and pyLDAvis, all model-agnostic.

- :material-compare: **[Distinguishing words](../guides/keywords.md)**

    Fighting Words: which words separate two corpora, with significance.

- :material-message-text: **[Short text](../guides/short-text.md)**

    Models built for tweets, headlines, and survey answers (`PT`, `GSDMM`).

- :material-arrow-right-circle: **[Held-out inference](../guides/transform.md)**

    `transform` new documents onto a fitted model across every model family.

</div>

Everything returns NumPy arrays, fits are deterministic for a fixed `seed`, and
the variational models parallelize across cores automatically.
