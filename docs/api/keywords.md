# Keywords & preprocessing

## Distinguishing words

::: topica.interpret.fighting_words

::: topica.interpret.top_fighting_words

## Preprocessing

::: topica.tokenize

::: topica.prep.split_documents

::: topica.design.one_hot

Prune rare (and optionally common) vocabulary from a corpus, keeping
metadata row-aligned with the documents that survive — the analogue of R
`stm`'s `prepDocuments`.

::: topica.prep.prep_documents

Sweep document-frequency thresholds and visualize how many documents and
vocabulary terms are removed at each level, to inform the choice of
`lower_thresh`.

::: topica.viz.plot_removed

## DataFrames & metadata

These accept pandas **or** Polars frames (and `align` also takes numpy arrays and
lists), keeping document metadata aligned to the rows that survive pruning.

::: topica.prep.from_dataframe

::: topica.prep.align

::: topica.design.design_matrix

## Embeddings

::: topica.embed.llm_embed

::: topica.embed.save_embeddings

::: topica.embed.load_embeddings

## Phrases

::: topica.prep.learn_phrases

::: topica.prep.apply_phrases

::: topica.prep.add_ngrams
