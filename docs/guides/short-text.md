# Short text

Tweets, headlines, search queries, and open-ended survey answers break standard
LDA: documents are too short for a stable mixture of topics to be estimated.
topica has two models built for this regime.

## GSDMM — one topic per document

The Gibbs Sampling Dirichlet Multinomial Mixture, a.k.a. the *Movie Group
Process* (Yin & Wang 2014), assumes each short document belongs to a **single**
topic. You give it an upper bound `K`, and it lets clusters shrink during
sampling, so it effectively **infers** the number of topics.

Two caveats to read the discovered count honestly. First, with the default
`alpha=0.1` an emptied cluster keeps `alpha`-proportional mass and can revive, so
clusters are not permanently pruned (that holds only for the paper's `alpha=0`
variant). Second, the number GSDMM settles on is sensitive to the `K` upper
bound, to `beta`, and to how separable your texts are: on clean, well-separated
short text it collapses cleanly toward the true count, but on messy overlapping
text it can sit near your `K` cap. Treat the discovered count as an
upper-bound-conditioned estimate, not a guaranteed recovery of the true `K`. If
the fit uses close to all `K` clusters, `topica` warns you to raise `K` and
refit until the count stabilises below the cap.

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
Prefer `doc_cluster` for the label: `doc_topic` is an in-sample soft score that
over-peaks, and its `argmax` can disagree with `doc_cluster` on a small fraction
of documents.

Because each document commits to one cluster, GSDMM's top-word lists tend to be
**less diverse** than a mixed-membership model's on the same overlapping corpus
(shared words concentrate in one cluster rather than spreading across several).
That is expected, not a defect, but it is worth knowing when you compare top-word
diversity across models. Watch, too, for very small clusters (one or two
documents): their top words, FREX/exclusivity, and coherence are computed from
almost no data and look deceptively strong, so `topica` warns when a fit produces
them — treat clusters that small as noise or merge them. Documents left empty
after vocabulary pruning are dropped from the fit; `corpus.kept_indices` maps the
rows of `doc_topic`/`doc_cluster` back to your original documents.

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
