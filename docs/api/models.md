# Models

All models share the same shape of API: construct with hyperparameters and a
`seed`, call `fit(documents, ...)`, then read `topic_word` (φ), `doc_topic` (θ),
`top_words(n)`, `coherence(n)`, and `save` / `load`.

This page covers the count-based models. The embedding-based models
(`BERTopic`, `Top2Vec`, `ETM`, `FASTopic`) are on the
[Embedding models](embedding.md) page.

::: topica.models.LDA

::: topica.models.DMR

::: topica.models.GDMR

::: topica.models.NarrativeTM

::: topica.models.LabeledLDA

::: topica.models.SAGE

::: topica.models.CTM

::: topica.models.STM

::: topica.models.ECTM

::: topica.models.STS

::: topica.models.ProdLDA

::: topica.models.HDP

::: topica.models.DTM

::: topica.models.SupervisedLDA

::: topica.models.PT

::: topica.models.GSDMM

::: topica.models.SeededLDA

::: topica.models.KeyATM

::: topica.models.PA

::: topica.models.HLDA

::: topica.Corpus
