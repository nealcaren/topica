# Quickstart

## If you have a CSV, do this

Most corpora start life as a table: one row per document, a text column, and some
metadata columns. `from_dataframe` turns that into a model-ready
[`Corpus`](../guides/preprocessing.md), keeping the metadata aligned to the
documents that survive pruning. The example below runs as written, on a
[bundled dataset](../api/datasets.md):

```python
import topica

df = topica.datasets.load_gadarian()          # bundled; loads offline
corpus = topica.prep.from_dataframe(
    df,
    text_col="open.ended.response",
    stopwords=topica.prep.ENGLISH_STOPWORDS,        # without this, every topic is "the, and, of"
    min_doc_freq=2,                            # drop words in fewer than 2 documents
)

model = topica.models.LDA(num_topics=5, seed=42)
model.fit(corpus)                             # sensible defaults; no other arguments needed
print(topica.interpret.summary(model))                  # top words per topic
```

For your own data, swap the first line for `df = pandas.read_csv("yours.csv")`
and set `text_col` to your text column.

!!! tip "Just `fit(corpus)` is enough"
    `LDA(num_topics=k).fit(corpus)` uses well-chosen defaults. The Gibbs samplers
    expose tuning knobs (`iters`, `num_samples`, `optimize_interval`, …), but you
    do not need to set them to get good topics. See
    [the models guide](../guides/models.md) for when to reach for them.

## Choosing the number of topics

`num_topics` (K) is a research decision, not a tuning parameter. As a starting
point, K=10 gives broad themes, K=30 finer ones. `search_k` scores a range of K
on coherence, exclusivity, and held-out likelihood:

```python
result = topica.select.search_k(corpus, ks=[5, 10, 20, 30], seed=42)
print(result.best_k())
```

The [choosing K guide](../publishing/choosing-k.md) walks through how to justify
the choice to a reader.

## Reading and labeling topics

```python
print(model.topic_word.shape)   # (K, V) — φ, topics × words
print(model.doc_topic.shape)    # (D, K) — θ, docs × topics (rows sum to 1)

# Publication-ready: prevalence + top words + distinctive (FREX) words per topic
table = topica.diagnostics.topic_table(model)

# The documents most associated with a topic — read these to name it
examples = topica.interpret.find_thoughts(model, topic=0, n=3)
```

!!! note "Stemmed words in the output?"
    topica's `tokenize` lowercases and splits but does **not** stem, so your own
    text keeps its surface forms. Some bundled corpora (such as `load_poliblog`)
    ship already stemmed, which is why their top words read like `militari`,
    `economi`. That is the data, not topica. To merge inflections and still get
    readable labels, [lemmatize instead of stemming](../guides/preprocessing.md#readable-topic-words-lemmatize-dont-stem)
    — pass a lemmatizing `tokenizer` to `from_dataframe`.

## Starting from raw text instead

If your documents are not in a DataFrame, tokenize them yourself into a
`list[list[str]]`:

```python
docs = [topica.tokenize(t, stopwords=topica.prep.ENGLISH_STOPWORDS) for t in texts]
corpus = topica.Corpus.from_documents(docs, min_doc_freq=5, rm_top=15)
```

A tiny self-contained corpus, for experiments:

```python
animals = [["cat", "dog", "fish", "cat", "dog"]] * 15
space   = [["planet", "star", "moon", "rocket", "planet"]] * 15
model = topica.models.LDA(num_topics=2, seed=42)
model.fit(animals + space)
for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", "  ".join(f"{w}({p:.2f})" for w, p in words))
```

Fits are deterministic for a fixed `seed`.

## Score and validate

```python
# Per-topic coherence and exclusivity — the standard quality pair.
coherence   = model.coherence(n=10)                 # UMass, per topic
exclusivity = topica.diagnostics.exclusivity(model, n=10)       # per topic
diversity   = topica.diagnostics.topic_diversity(model, topn=25)

# Windowed, human-aligned coherence (gensim-style); needs the reference texts:
cv = topica.diagnostics.coherence(model, corpus.documents(), coherence_type="c_v", topn=10)
```

## Infer topics for new documents

```python
new_docs = [["cat", "dog"], ["rocket", "moon"]]
theta = model.transform(new_docs, seed=0)       # (n, K), rows sum to 1
print(theta.argmax(axis=1))                     # dominant topic per document
```

## Where to go next

- [The models](../guides/models.md): pick the right one for your question.
- [Covariates & STM](../guides/covariates.md): relate topics to metadata.
- [Diagnostics & validation](../guides/diagnostics.md): choose `K`, prove stability.
- [Worked example: Du Bois in *The Crisis*](../examples/dubois.md): the whole workflow end to end.
