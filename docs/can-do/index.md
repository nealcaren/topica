# What you can do

topica is a general topic-modeling toolkit. This section tours what it does.
If your goal is a publishable analysis, pair it with
[Publishing in a journal](../publishing/index.md).

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
