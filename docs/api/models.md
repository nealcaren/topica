# Models

All models share the same shape of API: construct with hyperparameters and a
`seed`, call `fit(documents, ...)`, then read `topic_word` (φ), `doc_topic` (θ),
`top_words(n)`, `coherence(n)`, and `save` / `load`.

This page covers the count-based models. The embedding-based models
(`BERTopic`, `Top2Vec`, `ETM`, `FASTopic`) are on the
[Embedding models](embedding.md) page.

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
