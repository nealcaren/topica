# Datasets

Bundled example datasets for quickstarts and worked examples. Small datasets ship
inside the wheel and load offline; larger ones are downloaded once from GitHub on
first use and cached locally (under `~/.cache/topica/datasets`, or
`TOPICA_DATA_HOME`). The text loaders return a `pandas` DataFrame ready for
[`from_dataframe`](keywords.md); pass `return_path=True` for the cached CSV path
without pandas.

```python
import topica

df = topica.datasets.load_gadarian()
corpus = topica.from_dataframe(
    df, text_col="open.ended.response", stopwords=topica.data.ENGLISH_STOPWORDS
)
```

The embedding-native models want vectors, not a token corpus.
[`load_ng20_minilm`](#topica.datasets.load_ng20_minilm) covers that case: a
20-Newsgroups subset with MiniLM sentence embeddings precomputed for both
documents and vocabulary, so ProdLDA/FASTopic/BERTopic/Top2Vec run offline with
no `sentence-transformers`/`torch` install. It returns a `Bunch` (attribute
access), not a DataFrame.

```python
b = topica.datasets.load_ng20_minilm()
docs = [t.split() for t in b.texts]
# The default density clusterer (UMAP -> HDBSCAN) finds only ~3 topics here, with
# ~80% of the documents in one topic. That is NOT a topica defect: the reference
# umap-learn + HDBSCAN pipeline finds the same few-topic structure on this corpus at
# this min_cluster_size (see parity/bertopic_umap_default_compare.py). It is genuine,
# coarse density structure. For a finer or fixed number of topics, lower
# min_cluster_size, or use a fixed-K clusterer with reduce_frequent, e.g.:
bt = topica.BERTopic(
    clusterer="kmeans", num_clusters=5, reduce_frequent=True, seed=1,
).fit(docs, b.doc_embeddings)
```

::: topica.datasets.load_gadarian

::: topica.datasets.load_poliblog

::: topica.datasets.load_dubois

::: topica.datasets.load_congress

::: topica.datasets.load_reviews

::: topica.datasets.load_ng20_minilm

::: topica.datasets.get_data_home

::: topica.datasets.clear_cache
