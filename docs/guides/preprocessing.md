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

### Stopword lists (58 languages)

`topica.data.ENGLISH_STOPWORDS` is a short, stable English default. For other
languages — or a fuller English list — `topica.data.stopwords(lang)` serves the
[stopwords-iso](https://github.com/stopwords-iso/stopwords-iso) lists (58
languages, MIT licensed, bundled in the wheel). Accepts an ISO 639-1 code or an
English name:

```python
import topica

fr = topica.data.stopwords("fr")            # or "french"; case-insensitive
corpus = topica.from_dataframe(df, text_col="texte", stopwords=fr)

topica.data.stopword_languages()            # ['af', 'ar', 'bg', ..., 'zh']
```

Unknown languages raise with the list of available codes. For the cross-lingual
models ([`InfoCTM`](models.md#infoctm), [`ZeroShotTM`](embedding.md#zeroshottm)),
pass the matching list per language. Anything not covered: supply your own list.

For a sentiment or rating study, use `topica.data.SENTIMENT_STOPWORDS` instead of the
default: `ENGLISH_STOPWORDS` strips `not`/`no`/`very`/`too`, which would collapse
"not clean" into "clean" in exactly the analysis whose outcome is valence.

### Web-scraped text: strip HTML and URLs

Text scraped from the web (news blurbs, press releases, forum posts) often carries
markup — `<a href=...>` tags, `www.example.com/page.aspx` URLs — whose fragments
(`href`, `http`, `aspx`) survive tokenization and form a spurious "boilerplate"
topic. Pass `strip_html=True` to remove tags and `http`/`www` URLs before
tokenizing:

```python
corpus = topica.from_dataframe(df, text_col="text", strip_html=True)
```

`from_dataframe` also warns if such tokens survive into the vocabulary when you
did not strip, so the trap is hard to miss. `strip_html` is a conservative clean
(tags and URLs only); for heavier normalization, pass your own `tokenizer`.

### Email, forum, and Usenet text: strip headers, quotes, and signatures

`strip_html` handles markup, but email and newsgroup corpora (the classic
20 Newsgroups set, mailing-list archives, forum dumps) carry a different kind of
boilerplate: RFC headers (`From:`, `Subject:`, `NNTP-Posting-Host:`), quoted
replies (`> ...`, `On <date> so-and-so wrote:`), and signature blocks after a
`-- ` line. None of it is content, all of it survives `min_length=3` pruning, and
left in it forms "topics" of mail-client vocabulary and the most-quoted posters'
names. There is no built-in stripper for this — the shape is corpus-specific — so
clean the raw text before you tokenize. A pragmatic pass for Usenet-style messages:

```python
import re
import topica

def strip_message_boilerplate(raw: str) -> str:
    # 1. drop the header block: everything up to the first blank line
    body = raw.split("\n\n", 1)[-1]
    lines = []
    for line in body.splitlines():
        # 2. drop quoted reply lines and attribution ("On ... wrote:")
        if line.lstrip().startswith(">"):
            continue
        if re.match(r"\s*On .+wrote:\s*$", line):
            continue
        # 3. stop at the signature delimiter
        if line.rstrip() == "--":
            break
        lines.append(line)
    return "\n".join(lines)

df["clean"] = df["text"].map(strip_message_boilerplate)
corpus = topica.from_dataframe(df, text_col="clean")
```

Tune the rules to your source — headers, quote markers, and signature conventions
vary — but the principle holds: remove structural boilerplate in a preprocessing
pass, then let `from_dataframe`/`tokenize` handle the linguistic cleaning. Inspect
`corpus.vocabulary` afterward; if mail-client tokens or frequent poster surnames
still lead the counts, the strip did not reach them.

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
            for w in topica.tokenize(text, stopwords=topica.data.ENGLISH_STOPWORDS, min_length=3)]

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

### Cap the vocabulary size (`max_features`)

To keep only the most frequent terms, pass `max_features`. It caps the vocabulary
to the N most frequent surviving word types, applied after the other filters, and
matches scikit-learn's `CountVectorizer(max_features=)`:

```python
corpus = Corpus.from_documents(docs, max_features=10_000)   # keep the 10k most frequent terms
```

Ties are broken deterministically (by frequency, then by first appearance). Note
that scikit-learn ranks by collection (total) frequency, while gensim's `keep_n`
ranks by document frequency; topica follows scikit-learn.

### Domain boilerplate and proper nouns survive frequency pruning

Frequency filters (`max_doc_fraction`, `rm_top`) remove words that are common
*across the corpus*, so they miss two kinds of noise that are common *within a
genre* but not corpus-wide. The first is template or interface text that rides
along with the content: on a corpus of congressional press releases, even with
`strip_html=True`, `rm_top=20`, and `max_doc_fraction=0.5`, the share-button
labels `print` and `tweet` survive and cluster into their own topic. The second
is proper nouns, especially names of the actors the corpus is about: in the same
corpus, legislator surnames (`durbin`, `tester`, ...) collect into a topic that
tells you who spoke, not what they said.

Neither is a bug in the pruning; it is doing what you asked. When a run surfaces a
boilerplate-or-names topic, add the offending terms to a custom stopword list and
rebuild:

```python
stop = topica.data.stopwords("en") | {"print", "tweet", "share", "email"}
stop |= {"durbin", "tester", "schumer"}   # actor names, if they are not the object of study
corpus = topica.from_dataframe(df, text_col="text", stopwords=stop)
```

Inspect `corpus.word_counts` (or a first fit's `top_words`) before committing to a
list: the terms worth cutting are usually obvious once ranked, and whether a name
is noise depends on the question. If *who spoke* is part of the analysis, keep the
names and model authorship directly (`AuthorTopic`, or a speaker covariate).

## Apply a fixed vocabulary, or vectorize held-out documents

Two related tasks need the vocabulary held fixed rather than learned from the data.

Pin the vocabulary to a predetermined, ordered term list with `vocabulary=`
(scikit-learn's `vocabulary=`). Out-of-vocabulary tokens are dropped, the column
order follows your list, and the frequency filters are not applied (so
`vocabulary` cannot be combined with `min_doc_freq`, `max_features`, and friends):

```python
corpus = Corpus.from_documents(docs, vocabulary=["climate", "policy", "economy"])
```

To score held-out documents with a model you already fit, vectorize them against
the training corpus with `transform`. The result shares the training vocabulary
exactly (same terms, order, and ids, at full width), so the model's `topic_word`
columns stay aligned:

```python
corpus = Corpus.from_documents(train_docs, min_doc_freq=5)
model = topica.LDA(num_topics=20).fit(corpus)

heldout = corpus.transform(test_docs)          # same vocabulary as `corpus`
theta = model.transform(heldout)               # held-out document-topic mixtures
```

This is scikit-learn's `vectorizer.transform` and gensim's `doc2bow` on new text.
A held-out document with no in-vocabulary tokens is dropped; its surviving index
is recorded in `heldout.kept_indices`, so external labels can be realigned the
same way as after pruning.

## Detect phrases

Fixed expressions carry meaning together. Detect collocations and rewrite the
tokens before modeling:

```python
import topica
phrases = topica.data.learn_phrases(docs, min_count=8, threshold=12.0)
docs = topica.data.apply_phrases(docs, phrases)            # "health care" -> "health_care"
```

## Split long documents

Long, heterogeneous documents violate the bag-of-words assumption. Segment them
into comparable chunks, copying each source's metadata onto every chunk:

```python
chunks, chunk_meta = topica.data.split_documents(
    texts, metadata, max_words=200, min_words=50,
)
# chunk_meta[j] = the source row + {"parent": i, "chunk": j}
```

Chunks from the same source are **nested**, so use
[clustered standard errors](../publishing/effects.md) when you model effects.
