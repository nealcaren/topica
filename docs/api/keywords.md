# Keywords & preprocessing

## Distinguishing words

::: topica.fighting_words

::: topica.top_fighting_words

## Preprocessing

::: topica.tokenize

::: topica.data.split_documents

::: topica.design.one_hot

Prune rare (and optionally common) vocabulary from a corpus, keeping
metadata row-aligned with the documents that survive — the analogue of R
`stm`'s `prepDocuments`.

::: topica.data.prep_documents

Sweep document-frequency thresholds and visualize how many documents and
vocabulary terms are removed at each level, to inform the choice of
`lower_thresh`.

::: topica.data.plot_removed

## DataFrames & metadata

These accept pandas **or** Polars frames (and `align` also takes numpy arrays and
lists), keeping document metadata aligned to the rows that survive pruning.

::: topica.from_dataframe

::: topica.data.align

::: topica.design.design_matrix

## Embeddings

::: topica.embeddings.llm_embed

::: topica.embeddings.save_embeddings

::: topica.embeddings.load_embeddings

## Phrases

::: topica.data.learn_phrases

::: topica.data.apply_phrases

::: topica.data.add_ngrams
