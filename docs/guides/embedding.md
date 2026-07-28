# Embedding topics

The models elsewhere in topica learn topics from word counts. The models here
start from *embeddings*, in three flavors. **BERTopic** and **Top2Vec** *cluster*
document embeddings and read one topic off each cluster; **ETM** is *generative*,
LDA with the topic-word distribution factored through embeddings; **FASTopic**
reads topics off two *optimal-transport* plans between embedding sets. topica fits
all four with no PyTorch, no UMAP/numba, and no sentence-transformers in the
shipped wheel.

You bring the embeddings. topica does not call an embedding model; you pass a
document-vector matrix (and, for Top2Vec, a matching word-vector matrix) from
wherever you like, a sentence-transformer, an API, or a local model such as
ollama. Everything downstream is in the wheel.

If you would rather not wire up an embedder yourself, `topica.llm_embed` produces
the matrix through Simon Willison's [`llm`](https://llm.datasette.io/) library
(the optional `topica[llm]` extra), which reaches OpenAI embeddings and local
sentence-transformers via plugins:

```python
doc_emb = topica.llm_embed(texts, model="text-embedding-3-small")          # API
doc_emb = topica.llm_embed(texts, model="sentence-transformers/all-MiniLM-L6-v2")  # local
```

Embeddings are costly, so cache them. Pass `cache=path` to embed a corpus once and
reuse it on later runs (it reloads when the file matches the same `texts`, and
recomputes otherwise), or save and load any embedding matrix yourself:

```python
doc_emb = topica.llm_embed(texts, model="text-embedding-3-small", cache="emb.npz")

topica.save_embeddings("emb.npz", doc_emb, texts=texts, model="all-MiniLM-L6-v2")
doc_emb = topica.load_embeddings("emb.npz")
```

End to end, from raw text to a fitted model, with `llm_embed` doing the
text-to-vectors step offline (no API key, runs in the wheel):

```python
import topica

texts = [
    "The economy added jobs as the unemployment rate fell again.",
    "Inflation cooled and the central bank held interest rates steady.",
    "Markets rallied on the strong payrolls and wage-growth report.",
    "The home team scored late to win the playoff game in extra innings.",
    "He threw a complete-game shutout in the opener of the series.",
    "The rookie hit two home runs and drove in five for the win.",
]

# text -> (num_docs, E) vectors; the topica[llm] extra, sentence-transformers backend
doc_emb = topica.llm_embed(texts, model="sentence-transformers/all-MiniLM-L6-v2")

docs = [topica.tokenize(t, stopwords=topica.ENGLISH_STOPWORDS) for t in texts]
model = topica.BERTopic(min_cluster_size=2, seed=1)
model.fit(docs, doc_emb)
print(topica.report(model))
```

## BERTopic

BERTopic defines a topic by **class-based TF-IDF** over its documents' words, so
it needs only the document embeddings. The topic count is discovered by the
clustering, not set in advance.

```python
model = topica.BERTopic(min_cluster_size=15, seed=1)
model.fit(docs, doc_emb)

model.num_topics                       # discovered
model.top_words(8, topic=0)            # [(word, c-TF-IDF weight), ...]
model.topic_word                       # (num_topics, vocab), row-normalized c-TF-IDF
model.doc_topic                        # (num_docs, num_topics) soft membership
model.labels                           # hard cluster per doc; -1 is noise
```

Two BERTopic features carry over. `nr_topics` reduces the discovered topics down
to a target count:

```python
model = topica.BERTopic(min_cluster_size=15, nr_topics=10, seed=1)
model.fit(docs, doc_emb)
```

and `approximate_distribution` gives a soft topic distribution by sliding a
window over a document's words and comparing each window's c-TF-IDF to every
topic. It is the default `doc_topic`, and you can also run it on new documents:

```python
dist = model.approximate_distribution(new_docs, window=4, stride=1)  # (n, num_topics)
```

`min_similarity` (default `0.0`) drops any window-to-topic cosine below its value
before averaging; a document with no surviving evidence becomes a *uniform* row
so `doc_topic` stays a valid distribution.

### Topic-word weighting: c-TF-IDF vs CETopic's TFIDF×IDF_i

`weighting` chooses how topic words are scored. The default `"c-tf-idf"` is
BERTopic's class-based TF-IDF (tuned by `bm25` / `reduce_frequent`). Pass
`weighting="tfidf-idf"` for **CETopic**'s TFIDF×IDF_i scheme (Zhang et al., "Is
Neural Topic Modelling Better than Clustering?", NAACL 2022):

```python
model = topica.BERTopic(
    reducer="umap", clusterer="kmeans", num_clusters=20,
    weighting="tfidf-idf", seed=1,
)
model.fit(docs, doc_emb)   # exactly CETopic's contextual-embeddings → UMAP → K-Means → TFIDF×IDF_i pipeline
```

CETopic multiplies a **corpus-level TF-IDF** (a term's importance across the whole
corpus, averaged over each cluster's documents) by a **cross-cluster IDF** that
penalizes terms appearing in many clusters. c-TF-IDF asks only "frequent here, rare
across clusters"; TFIDF×IDF_i keeps globally salient words *and* actively demotes
words several clusters share, which lifts topic **diversity** (the paper's Table 3
ablation finds this the decisive win over plain per-cluster TF/TF-IDF). It is
topica's faithful port of the reference (`hyintell/topicx`, MIT), reproducing its
scikit-learn `TfidfTransformer` defaults (smoothed idf, per-document L2 norm, then
L1-normalized scores). Under `"tfidf-idf"` the `bm25` / `reduce_frequent` knobs do
not apply (they belong to class-based TF-IDF), and `doc_topic` still uses the
class-based-TF-IDF window geometry. CETopic's canonical clusterer is K-Means, which
leaves no `-1` noise; with the default HDBSCAN, noise documents are excluded from
both the corpus and the cluster means.

### Where topica differs from the `bertopic` package

topica reproduces the reference `bertopic` package on its default c-TF-IDF path,
but a few knobs diverge on purpose (issue #488), so name them when you report a
BERTopic run:

- **Reducer default matches the package.** topica defaults to `reducer="umap"`
  (`n_components=5`, `n_neighbors=15`, `min_dist=0`, cosine), as the package does;
  topica's UMAP is the in-house, seed-reproducible reducer, so the layout is
  deterministic for a fixed `seed`. Pass `reducer="pca"` for a linear, lighter
  projection (L2-normalized onto the unit sphere before clustering so a Euclidean
  clusterer sees the cosine geometry). topica still keeps `min_cluster_size=15`
  where the package uses `min_topic_size=10`, so name that when you report a run.
- **`nr_topics` reduction.** topica greedily folds the most c-TF-IDF-similar pair
  of topics; the package fits one ward `AgglomerativeClustering` over the topic
  *embeddings*. Different distance space and merge tree, so the two can pick
  different merges. topica also reads `nr_topics` as the number of *real* topics
  (the `-1` noise topic is never counted), whereas the package counts `-1` toward
  the total.
- **`bm25=True`** matches the package's class-based BM25 idf exactly, including the
  unclamped log that goes *negative* for a term common to every class (which ranks
  such terms last). The normalized `topic_word` surface floors those negatives to
  zero so it stays a valid distribution; ranking is unaffected.
- **`approximate_distribution`** weights each window with the same
  `bm25`/`reduce_frequent` idf the topics used. The `min_similarity` gate defaults
  to `0.0`; the package uses `0.1` for its own approximate distribution, so pass
  `min_similarity=0.1` to match it. topica returns a uniform row for a document
  with no surviving evidence where the package returns a zero row.

## Top2Vec

Top2Vec places each topic *in the embedding space*: the topic vector is the mean
of its documents' embeddings, and its words are the vocabulary terms nearest that
vector. Pass `word_embeddings` with the aligned `vocabulary` (same space as the
document embeddings) to get those nearest-word topics.

```python
vocab = sorted({w for d in docs for w in d})
word_emb = embed(vocab)                          # (len(vocab), E)

model = topica.Top2Vec(min_cluster_size=15, seed=1)
model.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)

model.top_words(8, topic=0)            # default: centroid view (nearest word vectors)
model.topic_neighbors(0, n=8)          # same centroid words, as (word, cosine)
model.top_words(8, topic=0, representation="c-tf-idf")  # the shared c-TF-IDF view
model.topic_vectors                    # (num_topics, E) topic positions
model.topic_sizes                      # documents per topic, largest first
```

Top2Vec and BERTopic share the class-based TF-IDF `topic_word` matrix, so given
the same clusters their `topic_word` and `topic_table` match. Top2Vec's distinct
view is the **centroid** representation, the vocabulary nearest the cluster
centroid in embedding space. When you pass `word_embeddings`, `top_words` (and so
`summary`) returns that by default; pass `representation="c-tf-idf"` for the
shared view. Without `word_embeddings` Top2Vec still fits and `top_words` is
c-TF-IDF.

Like the reference package, topica orders topics by size (topic 0 is the
largest, read `topic_sizes` for the counts) and exposes the reference search
surface — `search_documents_by_topic`, `search_documents_by_keywords`,
`search_words_by_vector`, `similar_words`, and `search_topics` — when fit with
`word_embeddings`. To collapse to a target topic count the reference way, call
`hierarchical_topic_reduction(n)`, which repeatedly merges the smallest topic
into its nearest topic by topic-vector cosine until `n` remain:

```python
model.hierarchical_topic_reduction(5)   # reduce to 5 topics, then re-order by size
```

## ETM

ETM (the Embedded Topic Model) is not a clustering pipeline; it is LDA with the
topic-word distribution factored through embeddings,
`β_{k,v} = softmax(ρ_v · α_k)`, and a logistic-normal document prior. Each topic
is a *point* `α_k` in the embedding space, and semantically related words share
topic mass even when a topic never saw them. You bring the word embeddings `ρ`;
topica fits the topic embeddings `α` and the prior by the same variational EM as
[`CTM`](models.md), no PyTorch.

```python
import topica

vocab = sorted({w for d in docs for w in d})
word_emb = embed(vocab)                          # (len(vocab), E)

model = topica.ETM(num_topics=20, seed=1)
model.fit(docs, word_emb, vocab)

model.topic_word                       # (num_topics, vocab) β
model.doc_topic                        # (num_docs, num_topics) θ
model.topic_embeddings                 # (num_topics, E) the α points
model.top_words(8, topic=0)
model.bound, model.converged           # the variational evidence bound
```

Because ETM is generative and mixed-membership, you get a proper `θ` and the full
[effects](../publishing/effects.md) and diagnostics stack, not a hard partition.
It fits in a fraction of a second on a few thousand documents.

### Inference: EM or VAE

ETM has two inference engines, selected with `inference=`. The default `"em"` is
the per-document variational EM above: accurate per document, but it runs an
optimizer for every document, so it does not minibatch. `"vae"` is the reference's
amortized autoencoder, an encoder network that maps a document's word counts
straight to its topic proportions. It trains by minibatch Adam, scales to large
corpora, and maps a new document with a single encoder pass rather than a
per-document optimization.

```python
model = topica.ETM(num_topics=20, inference="vae",
                   hidden_size=800, batch_size=1000, lr=0.005, seed=1)
model.fit(docs, word_emb, vocab, iters=150)
model.transform(new_docs)              # fast: one encoder forward pass
```

The reference fits the VAE with PyTorch autograd; topica hand-codes the encoder's
forward and backward (every gradient checked against finite differences) and steps
with Adam, so the VAE path is the same model with no PyTorch. Both engines return
the same surface (`topic_word`, `doc_topic`, `topic_embeddings`); `bound` is the
variational bound for EM and the ELBO for VAE. The trade is the usual one: EM is
more accurate per document, the VAE scales.

The VAE path also accepts the shared `prior=` and `contrastive=` flags described
under [ProdLDA](models.md#objective-and-prior-options): a Weibull-reparameterized
Dirichlet prior and a CLNTM-style InfoNCE term on the topic vectors. They are
ignored on the EM path and default off.

## FASTopic

FASTopic also drops the encoder, but it is not a clustering pipeline and not a
generative LDA. It places topics, words, and documents in one embedding space and
reads the topic proportions `theta` and topic-word matrix `beta` straight off two
*optimal-transport* plans: documents are transported to topics, topics to words.
You bring the document embeddings; topica learns the topic embeddings, the word
embeddings (in the same space), and the transport marginals, minimizing a
bag-of-words reconstruction plus the two transport costs.

```python
import topica

model = topica.FASTopic(num_topics=20, seed=1)
theta = model.fit_transform(docs, doc_emb)   # (num_docs, num_topics)

model.topic_word                       # (num_topics, vocab) beta
model.doc_topic                        # (num_docs, num_topics) theta
model.topic_embeddings                 # (num_topics, E) topic points
model.word_embeddings                  # (vocab, E) learned word points
model.top_words(8, topic=0)
model.loss_history                     # the objective at each epoch
```

Unlike Top2Vec and BERTopic, FASTopic is mixed-membership: each document gets a
full `theta` over topics, so it carries the [effects](../publishing/effects.md)
and diagnostics stack. New documents are mapped to topics by a distance-softmax
over the fitted topic embeddings, so `transform` needs only their embeddings, no
tokens:

```python
theta_new = model.transform(new_doc_emb)   # (n, num_topics)
```

The reference trains by autodiff through the unrolled Sinkhorn iterations; topica
has no autodiff, so it differentiates the fixed point of a hand-coded reverse-mode
Sinkhorn (every gradient checked against finite differences) and steps with Adam.
`dt_alpha`/`tw_alpha` are the inverse entropic regularizations for the two
transport problems (reference defaults 3.0 and 2.0); larger is sharper.

## CombinedTM

CombinedTM (Bianchi, Terragni & Hovy 2021) is [`ProdLDA`](models.md#prodlda) with
a richer encoder input. ProdLDA's encoder reads a document's bag of words;
CombinedTM concatenates that bag of words with a contextual document embedding (a
sentence-transformer vector, an API embedding, an ollama vector) and feeds the
pair to the same encoder. The product-of-experts decoder still reconstructs the
bag of words, and the prior, KL, reparameterization, batchnorm, and Adam are all
unchanged from ProdLDA. Mixing the contextual signal into the encoder yields more
coherent topics than the bag of words alone. You bring the per-document
embeddings at `fit`, one row per document, in corpus order.

```python
import topica

doc_emb = embed(docs)                    # (num_docs, E), your encoder of choice

model = topica.CombinedTM(num_topics=20, seed=1)
model.fit(docs, doc_emb, iters=150)

model.topic_word                         # (num_topics, vocab) softmax(beta_k)
model.doc_topic                          # (num_docs, num_topics) theta
model.top_words(8, topic=0)
model.bound, model.converged             # the ELBO at the final epoch
```

`transform` maps new documents the same way, so it needs both the tokens and
their embeddings:

```python
theta_new = model.transform(new_docs, embed(new_docs))   # (n, num_topics)
```

The reference fits the encoder with PyTorch autograd. We hand-code the encoder's
forward and backward, including the dense embedding block of the first layer
(every gradient checked against finite differences), and step with Adam, so this
is the same model with no PyTorch. Because the encoder is deterministic given a
seed, fits are bit-identical across reruns. CombinedTM also accepts the shared
`prior=` and `contrastive=` flags described under
[ProdLDA](models.md#objective-and-prior-options). The reference implementation is
[contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models)
(Bianchi et al., MIT).

## ZeroShotTM

ZeroShotTM (Bianchi, Nozza & Hovy 2021) takes the same idea one step further: the
encoder reads *only* the contextual document embedding, with no bag of words at
all. The decoder still reconstructs the bag of words, so topics remain proper
word distributions, but topic proportions are inferred from the embedding alone.
The constructor and surface match CombinedTM.

```python
import topica

model = topica.ZeroShotTM(num_topics=20, seed=1)
model.fit(docs, embed(docs), iters=150)
model.topic_word
model.doc_topic
```

Dropping the bag of words from the encoder is what enables **cross-lingual
transfer**. If you embed documents with a multilingual encoder, you can fit the
model on one language and `transform` documents in another: the held-out
documents map to the trained topics through their embeddings, and no shared
vocabulary is needed.

```python
# Fit on English, then map French documents to the same topics.
model.fit(english_docs, multilingual_embed(english_docs), iters=150)
theta_fr = model.transform(french_docs, multilingual_embed(french_docs))
```

As with CombinedTM, we hand-code the encoder's forward and backward over the
embedding-only first layer (finite-difference checked) and fit with Adam, so the
path has no PyTorch and is bit-identical across reruns. ZeroShotTM accepts the same
shared `prior=` and `contrastive=` flags described under
[ProdLDA](models.md#objective-and-prior-options). The reference implementation is
[contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models)
(Bianchi et al., MIT).

## DETM

DETM (the Dynamic Embedded Topic Model) is ETM for time-stamped corpora: the
topic embeddings drift across ordered time slices, so a topic's *words* evolve
while its identity persists. You supply word embeddings and a per-document time
index; the topic-word distribution at each slice is `softmax(alpha_k^(t) . rho)`,
with `alpha` following a Gaussian random walk over time.

```python
model = topica.DETM(num_topics=20, seed=1)
model.fit(docs, word_embeddings, vocabulary, times=year_index, iters=120)

model.topic_word                 # (K, V): time-averaged topics
model.beta_over_time             # (T, K, V): per-slice topic-word distributions
model.top_words_at(t=0, n=10)    # the top words of each topic in slice t
model.eta                        # (T, K): the time-varying topic prevalence prior
```

Inference is structured amortized variational inference (an LSTM over the per-time
word frequencies for the prevalence prior, an encoder for the document
proportions), hand-coded in the Rust core with finite-difference-checked
gradients; no PyTorch. Fits are deterministic from a fixed seed. On large
vocabularies the variational log-variances are clamped for numerical stability,
and an optional `grad_clip=` mirrors the reference's gradient clipping; neither
changes the default result on well-behaved corpora.

DETM is [validated against the reference](https://github.com/adjidieng/DETM)
(Dieng, Ruiz & Blei 2019, MIT) on the paper's UN-debates and ACL corpora: it
recovers topics at the reference's own seed-to-seed agreement (aligned cosine
0.74 / 0.59 against a 0.74 / 0.58 reference-vs-reference floor). One important
caveat: the per-time *prevalence* trajectory (`eta`) is weakly identified in DETM
— the reference implementation cannot reproduce its own `eta` across random seeds
either — so read the topic-word evolution (`beta_over_time`), which is stable, and
do not over-interpret a single `eta` trajectory.

## Post-fit diagnostics

The reduce→cluster pipeline decides almost everything, and its failure modes are
*silent*: a bad configuration still returns a model. BERTopic and Top2Vec run a
cheap post-fit check and emit a one-time `warnings.warn` when the result looks
degenerate — near-total **collapse** (1–2 topics on a sizeable corpus, usually
unnormalized coordinates or too large a `min_cluster_size`), a very **high noise
fraction** (most documents left unassigned), or gross **over-splitting** (far more
topics than the corpus supports). Each message names a concrete fix. The
thresholds are conservative; silence it with `diagnostics=False`:

```python
model = topica.BERTopic(seed=1)                    # warns if the fit is degenerate
model = topica.BERTopic(diagnostics=False, seed=1) # silent
```

## Avoiding the `-1` noise bucket

HDBSCAN (the default) discovers the topic count but leaves sparse documents
unassigned as `-1`. On real sentence-transformer embeddings that bucket can be
large, and for many social-science questions every document should land
somewhere. Two ways out:

- **Switch to an auto-K graph clusterer.** Pass `clusterer="louvain"` or
  `"leiden"` (no `num_clusters` needed). Both build a k-nearest-neighbor graph over
  the reduced embeddings and optimize modularity, so — like HDBSCAN — they
  *discover* the topic count, but unlike HDBSCAN they assign every document (no
  `-1`). `"leiden"` adds a refinement phase (Traag, Waltman & van Eck 2019) that
  guarantees every topic is internally connected. On fine-grained corpora these
  recover a sensible number of topics where HDBSCAN over- or under-splits, and they
  can beat k-means handed the true count.

  ```python
  model = topica.BERTopic(clusterer="leiden", seed=1)   # count discovered, no noise
  model.fit(docs, doc_emb)
  assert -1 not in model.labels
  ```

  "Auto" does not mean unsteerable. `resolution` (default 1.0) trades off how many
  topics they find — raise it for a fine-grained corpus, lower it for broad themes;
  the secondary `knn_neighbors` (default 15) does the same more weakly (smaller =
  more, tighter topics). Both are ignored by the other clusterers.

  ```python
  fine  = topica.BERTopic(clusterer="leiden", resolution=2.0, seed=1)   # more topics
  broad = topica.BERTopic(clusterer="leiden", resolution=0.5, seed=1)   # fewer topics
  ```

- **Switch to a fixed-K clusterer.** Pass `clusterer="kmeans"`, `"gmm"`, or
  `"agglomerative"` with `num_clusters=K` to BERTopic or Top2Vec. All three assign
  every document to one of `K` clusters, so there is no `-1` label (and the topic
  count is fixed, not discovered). KMeans scales; `"gmm"` is a diagonal-covariance
  Gaussian mixture that, unlike k-means, models each topic's spread — so
  unequal-variance topics separate more cleanly, and it tends to match or beat
  k-means on embedding clusters; agglomerative (average linkage) suits moderate
  corpora.

  ```python
  model = topica.BERTopic(clusterer="kmeans", num_clusters=20, seed=1)
  model.fit(docs, doc_emb)
  assert -1 not in model.labels
  ```

  With `clusterer="gmm"`, BERTopic's `doc_topic` is the GMM's **soft** membership
  (the EM posterior responsibilities, rows summing to one), not the c-TF-IDF
  approximate distribution — a genuine mixture `θ` for documents that span several
  topics, where hard clustering assigns only one. The hard `labels` stay the row
  argmax. (This applies to the base fit; combining `gmm` with `nr_topics` topic
  reduction reverts `doc_topic` to the c-TF-IDF distribution.)

  ```python
  model = topica.BERTopic(clusterer="gmm", num_clusters=20, seed=1)
  model.fit(docs, doc_emb)
  theta  = model.doc_topic          # (D, 20) soft membership from GMM responsibilities
  labels = theta.argmax(1)          # == model.labels
  ```

- **Use a fixed-K, every-document model.** `EmbeddingLDA`, `FASTopic`, and `ETM`
  are embedding-driven but give every document a full topic distribution `θ` with
  no noise bucket. In our testing `EmbeddingLDA` gave the best recovery when the
  `-1` bucket was the problem.

`reduce_outliers()` (below) is the third option: keep HDBSCAN, then reassign the
`-1` documents after the fact.

## Inspecting and adjusting clustering models

Top2Vec and BERTopic produce hard `labels` (`-1` is a noise/outlier document), so
they support two post-hoc edits. `reduce_outliers()` reassigns every `-1`
document to the topic whose words best explain it and rebuilds the topic-word
matrix, returning how many it moved. `merge_topics([[3, 7], [1, 2]])` collapses
groups of topics you decide to combine, rebuilding the representation and
renumbering topics. Both also gain `transform`/`fit_transform` for held-out
documents, and the c-TF-IDF knobs `bm25=` and `reduce_frequent=` on BERTopic.

For a quick read of any fitted model (not just these), `topica.topic_info(model,
texts)` returns per-topic size, prevalence, top words, and representative
documents, with an outlier row when present; `topica.topics_over_time(model,
timestamps)` and `topica.topics_per_class(model, groups)` summarize prevalence by
time or group; and `topica.set_topic_labels(model, {...})` stores your own labels.

## The shared surface

Both models expose topica's standard fitted surface, so they slot in alongside
every other model: `topic_word` (`num_topics × vocab`), `doc_topic`
(`num_docs × num_topics`), `top_words`, `num_topics`, `topic_names`,
`vocabulary`, and `labels`. The embedding-native additions are `topic_vectors`
and `topic_neighbors` (Top2Vec) and `approximate_distribution` (BERTopic).

They also `save`/`load` like every other model, so a fitted model reloads and
`transform`s new documents without re-running the pipeline:

```python
model.save("topics.tt")
model = topica.BERTopic.load("topics.tt")   # reload, then transform() forever
```

## Richer topic words: n-grams

The c-TF-IDF topic words are over the tokens you pass in, so bigrams are a
preprocessing choice. `topica.add_ngrams` adds them (the mechanical analog of
scikit-learn's `CountVectorizer(ngram_range=..., min_df=...)`), keeping every
document so the rows stay aligned with the embeddings:

```python
docs = [topica.tokenize(t, stopwords=topica.ENGLISH_STOPWORDS) for t in texts]
docs = topica.add_ngrams(docs, ngram_range=(1, 2), min_df=5)   # unigrams + bigrams
model.fit(docs, doc_emb)        # topic words can now read "machine_learning"
```

For statistically-selected phrases instead of every bigram, use
[`learn_phrases`](../guides/preprocessing.md).

## Tuning and notes

- `min_cluster_size` is the main dial: larger gives fewer, broader topics; smaller
  gives more, finer ones. `min_samples` (default `min_cluster_size`) sets how
  aggressively sparse documents are called noise (label `-1`). These apply to the
  default `clusterer="hdbscan"`; `clusterer="kmeans"`/`"gmm"`/`"agglomerative"`
  use `num_clusters` instead, and `clusterer="louvain"`/`"leiden"` discover the
  count on their own (see above).
- `n_components` is the dimensionality the embeddings are reduced to before
  clustering. `BERTopic` and `Top2Vec` both default to `reducer="umap"` with
  `n_neighbors=15` (matching the upstream BERTopic package and the original
  Top2Vec's default UMAP config, both of which reduce with UMAP). `reducer="pca"` switches to
  a randomized PCA: fast, deterministic, and dependency-free, but it separates less
  sharply than UMAP and on closely spaced themes can merge clusters a UMAP run would
  split. Under PCA the reduced coordinates are L2-normalized onto the unit sphere
  before clustering, so the Euclidean clusterer measures cosine distance — the
  geometry sentence embeddings are trained for. Without this the few highest-variance
  PCA directions dominate the metric and the clusterer under-splits real embeddings
  into a couple of broad topics.
- `reducer="umap"` switches to topica's in-house UMAP reducer (with `n_neighbors`),
  which separates real document embeddings much better than a linear projection
  and, on closely spaced themes, splits clusters PCA would merge. It is a faithful
  reimplementation of `umap-learn` (fuzzy simplicial set, `a`/`b` membership curve,
  and the reference SGD layout), validated to match `umap-learn`'s cluster quality
  on real sentence embeddings. It ships in the wheel, so it is opt-in at runtime,
  not build time — pure Rust, with no `umap-learn`/`numba` dependency.

  Unlike a typical UMAP, topica's is **fully reproducible**: the negative sampling
  is seeded, so a fixed `seed` pins the layout and the whole `reducer="umap"` fit is
  deterministic. There is no non-determinism caveat and no warning.
- The UMAP layout is tunable. Beyond `n_neighbors`, `reducer="umap"` accepts
  `min_dist` (minimum spacing of points in the embedding; lower packs clusters
  tighter — the default `0.0` matches BERTopic), `spread`, `n_epochs` (`0` = auto:
  500 for ≤10k rows), `negative_sample_rate`, `repulsion_strength`, and `metric`
  (`"cosine"` default, or `"euclidean"`). All default to `umap-learn`'s values, so
  touching nothing reproduces the reference; they are ignored under `reducer="pca"`.
  The same knobs are on `topica.project(method="umap", ...)`.

  ```python
  model = topica.BERTopic(reducer="umap", min_dist=0.1, n_neighbors=30, seed=1)
  ```
- Results are reproducible for a fixed `seed`, under either reducer.

!!! note "Faithful to the references"
    On a shared task with shared document embeddings, topica's `Top2Vec` and
    `BERTopic` recover comparable topic structure to the Python `BERTopic`
    package. topica's in-house UMAP reducer matches `umap-learn`'s cluster quality
    on real sentence embeddings (measured by adjusted Rand index against gold
    labels); because topica uses a different HDBSCAN implementation, exact cluster
    assignments still differ from the `umap-learn` + `hdbscan` reference, but the
    recovered topics agree. The payoff is the dependency footprint: topica runs the
    whole pipeline in Rust with none of `torch`, `umap-learn`, or `hdbscan`
    installed.
