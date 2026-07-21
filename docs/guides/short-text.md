# Short text

Tweets, headlines, search queries, and open-ended survey answers break standard
LDA: documents are too short for a stable mixture of topics to be estimated.
topica has two models built for this regime.

## GSDMM — one topic per document

The Gibbs Sampling Dirichlet Multinomial Mixture, a.k.a. the *Movie Group
Process* (Yin & Wang 2014), assumes each short document belongs to a **single**
topic. You give it an upper bound `K`; empty clusters die out during sampling, so
it effectively **infers** the number of topics.

```python
import topica

model = topica.GSDMM(num_topics=30, seed=1)     # 30 is the MAX number of clusters
model.fit(short_docs, iters=30)

print(model.num_topics, "clusters used")    # usually far fewer than 30
model.top_words(8)
model.doc_cluster                            # one cluster id per document
```

`topic_word` and `doc_topic` cover only the non-empty clusters; `doc_cluster`
gives the hard assignment, since GSDMM places each document in exactly one group.

## PT — pseudo-document aggregation

The Pseudo-document Topic model (Zuo et al. 2016) aggregates short texts into a
smaller set of **pseudo-documents**, recovering the longer-document statistics
LDA needs while still mixing topics within a text.

```python
model = topica.PT(num_topics=20, num_pseudo=100, seed=1)
model.fit(short_docs, iters=1000)
```

## BTM — biterm co-occurrence

The Biterm Topic Model (Yan, Guo, Lan & Cheng 2013) attacks short-text sparsity
from the word side. Instead of estimating a topic mixture per document (too few
words to pin down), it models the corpus as a bag of **biterms** — unordered word
pairs co-occurring within a window — and learns one *global* topic distribution
plus per-topic word distributions from those co-occurrences. Both words of a
biterm are drawn from the same topic, so the topic-word distributions absorb the
co-occurrence signal directly. Document topics are read back out afterward by
summing each document's biterms (`p(z|d) = Σ_b p(z|b) p(b|d)`).

```python
model = topica.BTM(num_topics=20, seed=1)       # alpha defaults to 50/k
model.fit(short_docs, iters=1000)

model.topic_word           # per-topic word distributions
model.theta                # the global topic distribution p(z)
scores = model.transform(new_docs)   # document topics for held-out texts
```

`window` (default 15) sets how far apart two words may be to form a biterm;
`background=True` reserves topic 0 for common words (the empirical word
distribution), which can sharpen the remaining topics. Validated against the
reference R `BTM` package (see the [validation record](../publishing/validation.md)).

## Which to use

- **`GSDMM`** when each short text is plausibly about one thing (most tweets,
  most survey answers) and you want the model to find how many groups there are.
- **`PT`** when texts may still blend a few topics and you want LDA-style mixed
  membership that holds up on short texts.
- **`BTM`** when documents are very short and you want the topics driven by
  corpus-wide word co-occurrence rather than any per-document mixture — the
  standard choice for tweet-length text.

All three feed the same [diagnostics](diagnostics.md) and
[validation](../publishing/validation.md) as every other model.
