# Models

All models share the same shape of API: construct with hyperparameters and a
`seed`, call `fit(documents, ...)`, then read `topic_word` (φ), `doc_topic` (θ),
`top_words(n)`, `coherence(n)`, and `save` / `load`.

This page covers the count-based models. The embedding-based models
(`BERTopic`, `Top2Vec`, `ETM`, `FASTopic`) are on the
[Embedding models](embedding.md) page.

## Finding a model

Not sure which class you want? Start from the [decision matrix](../can-do/index.md),
or filter the roster in code with `topica.list_models()`. Each filter narrows the
result; combine them freely.

```python
import topica

topica.list_models(tier=1)              # the recommended starting points
topica.list_models(group="short-text")  # a purpose group
topica.list_models(brings="metadata")   # models that accept covariates
topica.list_models(inference="gibbs")   # by inference engine
topica.list_models(experimental=False)  # only the validated roster
```

Each result is a `ModelInfo` with `name`, `group`, `brings`, `inference`,
`determinism`, `tags`, `tier`, and `experimental`. The `name` resolves directly:
`getattr(topica, m.name)`. See the [model catalog](../guides/models.md) for the
full grouped table.

::: topica.list_models

::: topica.LDA

::: topica.DMR

::: topica.GDMR

::: topica.RTM

::: topica.NarrativeTM

::: topica.LabeledLDA

::: topica.SAGE

::: topica.CTM

::: topica.STM

::: topica.STS

::: topica.ProdLDA

::: topica.HDP

::: topica.DTM

::: topica.SupervisedLDA

::: topica.PT

::: topica.GSDMM

::: topica.SeededLDA

::: topica.KeyATM

::: topica.PA

::: topica.HLDA

::: topica.Corpus
