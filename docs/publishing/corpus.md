# 1. Build a defensible corpus

Before any modeling, a reviewer wants to know exactly what you fed the model and
why. Preprocessing choices change topics, so they are part of your method, not
plumbing to hide.

## State the population and the unit

Report, in prose and with counts:

- **What population** the documents are drawn from, and how they were sampled.
- **The unit of analysis.** An article? A paragraph? A speech? A tweet? This is
  a substantive choice, not a technical one.
- **The time span and any covariates** you will use later.

## Choose the unit deliberately: split long documents

LDA and STM assume documents are roughly comparable bags of words. A corpus of
60-word tweets *and* 6,000-word transcripts violates that badly. If your
documents are long and heterogeneous, segment them into comparable chunks,
keeping each chunk tied to its source document's metadata.

```python
import topica

# `texts` are long documents; `meta` is one dict of covariates per document.
chunks, chunk_meta = topica.data.split_documents(
    texts, meta,
    max_words=200,     # target chunk length
    min_words=50,      # merge a short tail back rather than drop it
)
# chunk_meta[j] carries the source covariates plus `parent` and `chunk`.
```

Report the splitting rule and how many chunks resulted. When you analyze effects
later, remember that chunks from the same source document are **nested**, which
is exactly when you'll want [clustered standard errors](effects.md).

## Tokenize and prune the vocabulary, and say how

```python
from topica import Corpus, tokenize

stop = open("stoplist.txt").read().split()
docs = [tokenize(t, stopwords=stop, min_length=3) for t in chunks]

corpus = Corpus.from_documents(
    docs,
    min_doc_freq=10,      # a word must appear in >= 10 documents
    max_doc_fraction=0.5, # drop words in > 50% of documents
    rm_top=20,            # drop the 20 most frequent residual words
)
```

A few defensible defaults, all of which you should report:

- **Lowercase, drop punctuation and very short tokens.** Standard.
- **Do not stem.** Stemming wrecks interpretability (`citizen`, `citizenship`,
  and `city` can collapse together). Prefer lemmatization in your own pipeline if
  you need it; topica deliberately does neither for you.
- **Prune rare and ubiquitous terms** (`min_doc_freq`, `max_doc_fraction`,
  `rm_top`). Rare terms add noise and `junk` topics; ubiquitous terms add nothing.
- **Custom stopwords** for corpus-specific boilerplate (a magazine's own name, a
  transcription artifact). Report the list.

!!! warning "Preprocessing is a researcher degree of freedom"
    Different preprocessing yields different topics. Pick choices *before* you
    look at results, motivate them substantively, and check that your
    conclusions survive reasonable alternatives (see
    [validation](validation.md)).

## Detect phrases before modeling

Fixed expressions (`jim crow`, `health care`, `climate change`) carry more
meaning together than apart. Detect them first so a topic can be about the
phrase, not its scattered parts.

```python
phrase_model = topica.data.learn_phrases(docs, min_count=8, threshold=12.0)
docs = topica.data.apply_phrases(docs, phrase_model)
```

## Inspect what survived

```python
print(corpus.num_docs, corpus.num_words, corpus.total_tokens)
```

Report the document count, vocabulary size, and token count *after* pruning.
Those three numbers belong in your methods section.

## Choosing vocabulary thresholds with prep_documents

`topica.data.prep_documents` prunes a `Corpus` by document frequency — dropping
terms that appear in fewer than `lower_thresh` documents — and keeps the
metadata frame aligned with the surviving documents. This is the analogue of
R `stm`'s `prepDocuments`:

```python
import topica

corpus_pruned, meta_pruned = topica.data.prep_documents(
    corpus, meta=meta_df,
    lower_thresh=5,   # drop terms appearing in fewer than 5 docs
    rm_top=20,        # also drop the 20 most frequent residual terms
)
```

Before committing to a threshold, sweep a range and visualize how many
documents and vocabulary terms each level removes:

```python
topica.data.plot_removed(corpus, thresholds=range(1, 15))
```

The chart shows two lines: documents removed (left axis) and vocabulary terms
removed (right axis). A threshold that removes many documents will corrupt a
downstream covariate analysis; aim for the elbow where vocabulary shrinks
rapidly but document loss stays near zero.

→ Next: [Pick the right model](choosing-model.md).
