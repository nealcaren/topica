# Embedding-based models

These models start from document (and sometimes word) vectors the user supplies
from any embedder, rather than from word counts. They present the same
`topic_word` (φ) / `doc_topic` (θ) surface as the count-based models, so the
diagnostic, labeling, and reporting tools apply to them unchanged. See the
[embedding topics guide](../guides/embedding.md) for a worked walkthrough.

`BERTopic` and `Top2Vec` cluster the embeddings; `ETM` and `FASTopic` are
generative models that factor topics through the embedding space. None require
PyTorch: `fit` takes the vectors you pass in.

::: topica.models.BERTopic

::: topica.models.Top2Vec

::: topica.models.ETM

::: topica.models.FASTopic
