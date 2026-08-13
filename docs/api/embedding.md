# Embedding-based models

These models start from document (and sometimes word) vectors the user supplies
from any embedder, rather than from word counts. They present the same
`topic_word` (φ) / `doc_topic` (θ) surface as the count-based models, so the
diagnostic, labeling, and reporting tools apply to them unchanged. See the
[embedding topics guide](../guides/embedding.md) for a worked walkthrough.

`BERTopic` and `Top2Vec` cluster the embeddings; `ETM` and `FASTopic` are
generative models that factor topics through the embedding space. None require
PyTorch: `fit` takes the vectors you pass in.

::: topica.BERTopic

::: topica.Top2Vec

::: topica.ETM

::: topica.FASTopic

## Mechanistic Topic Models

Topics over **sparse-autoencoder features** rather than words, so each topic is a
distribution over directions the language model itself represents. You bring the SAE
activations; `topica.mtm` turns them into counts and embeddings. Both models are
experimental and gated — see the [model catalog](../guides/models.md#mechanisticlda)
for the recipe, the reference's defaults, and the known divergences.

::: topica.MechanisticLDA

::: topica.MechanisticBERTopic

::: topica.mtm.featurize

::: topica.mtm.feature_thresholds

::: topica.mtm.document_embeddings

::: topica.mtm.FeatureCounts
