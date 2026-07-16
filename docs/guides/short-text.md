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

model = topica.models.GSDMM(num_topics=30, seed=1)     # 30 is the MAX number of clusters
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
model = topica.models.PT(num_topics=20, num_pseudo=100, seed=1)
model.fit(short_docs, iters=1000)
```

## Which to use

- **`GSDMM`** when each short text is plausibly about one thing (most tweets,
  most survey answers) and you want the model to find how many groups there are.
- **`PT`** when texts may still blend a few topics and you want LDA-style mixed
  membership that holds up on short texts.

Both feed the same [diagnostics](diagnostics.md) and
[validation](../publishing/validation.md) as every other model.
