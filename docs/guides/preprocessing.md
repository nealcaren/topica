# Preprocessing

topica takes pre-tokenized documents, a `list[list[str]]`, or a `Corpus`.
You control tokenization and vocabulary, because those choices are part of your
method (see [Build a defensible corpus](../publishing/corpus.md)).

## Tokenize

```python
from topica import tokenize

stop = open("stoplist.txt").read().split()        # a list, not a set
tokens = tokenize(text, stopwords=stop, min_length=3)
```

`tokenize` lowercases, applies a regex, drops stopwords and short tokens. It does
**not** stem (stemming hurts interpretability); lemmatize in your own pipeline if
you need it.

### Readable topic words: lemmatize, don't stem

Stemming truncates words to a root (`military` → `militari`, `economy` →
`economi`), so top-word tables read as broken. If your text is not already
stemmed, topica keeps the surface forms as-is. To merge inflections *and* keep
readable words, lemmatize — and because `from_dataframe` (and `tokenize`) take a
`tokenizer` callable, you can drop a lemmatizer straight in:

```python
import topica
from nltk.stem import WordNetLemmatizer   # pip install nltk; nltk.download("wordnet")

_lemm = WordNetLemmatizer()
def lemmatize(text):
    return [_lemm.lemmatize(w)
            for w in topica.tokenize(text, stopwords=topica.ENGLISH_STOPWORDS, min_length=3)]

corpus = topica.from_dataframe(df, text_col="text", tokenizer=lemmatize)
# top words now read "military", "economy" — not "militari", "economi"
```

If your corpus arrives already stemmed (some bundled datasets and `stm`'s
`poliblog` do), there is no way to recover the original words — that is the data,
not topica. Re-process from the raw text if you want readable labels.

## Build a Corpus and prune the vocabulary

```python
from topica import Corpus

corpus = Corpus.from_documents(
    docs,
    min_doc_freq=10,        # keep words in >= 10 documents
    max_doc_fraction=0.5,   # drop words in > 50% of documents
    min_cf=0,               # collection-frequency cutoff
    rm_top=20,              # drop the N most frequent residual words
)
print(corpus.num_docs, corpus.num_words, corpus.total_tokens)
```

The vocabulary is compiled in Rust, so even multi-gigabyte corpora build quickly.
A `Corpus` can also load from disk (one document per line, or MALLET-style TSV).

## Detect phrases

Fixed expressions carry meaning together. Detect collocations and rewrite the
tokens before modeling:

```python
import topica
phrases = topica.learn_phrases(docs, min_count=8, threshold=12.0)
docs = topica.apply_phrases(docs, phrases)            # "health care" -> "health_care"
```

## Split long documents

Long, heterogeneous documents violate the bag-of-words assumption. Segment them
into comparable chunks, copying each source's metadata onto every chunk:

```python
chunks, chunk_meta = topica.split_documents(
    texts, metadata, max_words=200, min_words=50,
)
# chunk_meta[j] = the source row + {"parent": i, "chunk": j}
```

Chunks from the same source are **nested**, so use
[clustered standard errors](../publishing/effects.md) when you model effects.
