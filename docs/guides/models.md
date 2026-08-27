# The models

Every model shares the same shape: construct with hyperparameters and a `seed`,
`fit(documents, ...)`, then read `topic_word` (φ), `doc_topic` (θ),
`top_words(n)`, `coherence(n)`, and `save` / `load`. Full signatures are in the
[API reference](../api/models.md).

## Choosing a model

| If you want to… | Use |
|-----------------|-----|
| Discover themes, fast and standard | [`LDA`](#lda) |
| Relate topic prevalence to metadata | [`STM`](../guides/covariates.md), [`DMR`](#dmr) |
| Let topics correlate | [`CTM`](#ctm), `STM` |
| Have topics worded differently by group | [`SAGE`](#sage), `STM` (content) |
| Measure topic sentiment/discourse from covariates | [`STS`](#sts) |
| Let the data choose the number of topics | [`HDP`](#hdp) |
| Track topics that drift over time | [`DTM`](#dtm) |
| Tie topics to known labels | [`LabeledLDA`](#labeledlda) |
| Shape topics to predict an outcome | [`SupervisedLDA`](#supervisedlda) |
| Steer topics with known keywords | [`keyATM`, `seededlda`](guided.md) |
| Sharper, more coherent topics at scale | [`ProdLDA`](#prodlda) |
| Model short texts (tweets, answers) | [`PT`, `GSDMM`](short-text.md) |
| Build a topic hierarchy | `PA`, `HLDA` |

## The roster

Every model, grouped by purpose. **Brings** is what you supply beyond raw text;
**Reproducibility** is `bit-exact` (identical regardless of thread count),
`seed-reproducible` (identical from a fixed seed and thread count), or
`llm-bounded`. Filter this roster in code with
`topica.list_models(group=…, brings=…, inference=…, determinism=…)`. The table is
generated from `python/topica/registry.py`.

<!-- BEGIN MODEL TABLE (generated from topica.registry; edit registry.py, not this block) -->

*Every model below is validated before it enters the roster: against a maintained reference implementation where one exists (MALLET, gensim, R `stm`, tomotopy, and the like), otherwise by planted recovery on a synthetic corpus with a known answer.* See [validation](https://nealcaren.github.io/topica/contributing/validation/) for where each model stands. The groupings are about **fit to your research design**, not quality: a specialized model is the right first choice when your data calls for it.

### Common starting points

One per common goal, where most social scientists begin. `BERTopic` works differently from the others: it clusters document embeddings rather than fitting a posterior, so topic-proportion uncertainty and covariate-effect estimation do not carry over directly.

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `LDA` | text | gibbs | seed-reproducible | Classic latent Dirichlet allocation via a fast SparseLDA collapsed-Gibbs sampler. |
| `NMF` | text | matrix-factorization | bit-exact | Non-negative matrix factorization of the document-term matrix via multiplicative updates. |
| `STM` | text, metadata | variational | bit-exact | Structural topic model: relate topic prevalence and content to covariates. |
| `KeyATM` | text, seeds | gibbs | seed-reproducible | Keyword-assisted topics: anchor named topics with a few seed words each. |
| `GSDMM` | text | gibbs | seed-reproducible | Gibbs-sampling Dirichlet mixture: one topic per short document. |
| `BERTopic` | text, embeddings | clustering | seed-reproducible | Cluster document embeddings; label topics by class-based TF-IDF. |

### Specialized approaches

The right first choice when your design calls for one: short text, change over time, document networks, multiple languages, ideological scaling, and more.

#### General-purpose

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `OnlineLDA` | text | variational | seed-reproducible | Online (streaming) variational-Bayes LDA (Hoffman et al. 2010): minibatch stochastic VB with a decaying learning rate and a streaming partial_fit; the gensim LdaModel analogue for very large or streaming corpora. |
| `CTM` | text | variational | bit-exact | Correlated topic model: a logistic-normal prior that lets topics co-occur. |
| `ProdLDA` | text | vae | seed-reproducible | Product-of-experts LDA (AVITM) for sharper, more coherent topics; hand-coded VAE. |
| `HDP` | text | gibbs | seed-reproducible | Hierarchical Dirichlet process: infers the number of topics from the data. |
| `LSA` | text | svd | seed-reproducible | Latent semantic analysis: a truncated SVD of the weighted document-term matrix. |
| `AnchorLDA` | text | matrix-factorization | bit-exact | Anchor-words spectral recovery (Arora et al. 2013): deterministic, Gibbs-free topics from the word co-occurrence matrix. |
| `PolylingualLDA` | text | gibbs | seed-reproducible | Polylingual topic model (Mimno et al. 2009): aligned topics across languages from document tuples that share one topic distribution. |
| `CorEx` | text | information-theoretic | seed-reproducible | Correlation Explanation: information-theoretic topic model that maximizes total correlation; supports anchor words. |
| `MGLDA` | text | gibbs | seed-reproducible | Multi-Grain LDA: global (document-level) + local (sliding-window aspect) topics with a per-token grain switch. For reviews / aspect extraction. |
| `TopicalNGrams` | text | gibbs | seed-reproducible | Topical N-Grams (Wang, McCallum & Wei 2007): an LDA extension that jointly discovers topics and topic-specific multiword phrases. A per-token bigram-status indicator, sampled with the topic, decides whether a token continues a phrase from the previous word given its topic, so phrase structure is learned during fitting rather than fixed beforehand. Exposes top_phrases alongside top_words. |

#### Covariates & structure

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `STS` | text, metadata | variational | bit-exact | Structural topic-and-sentiment model over document metadata. |
| `SAGE` | text, metadata | gibbs | seed-reproducible | Sparse additive generative model: the same topic worded differently across groups. |
| `DMR` | text, metadata | gibbs | seed-reproducible | Dirichlet-multinomial regression: a document-metadata prior on topic proportions. |
| `GDMR` | text, metadata | gibbs | seed-reproducible | Generalized DMR with a smooth (Legendre-basis) prior over continuous covariates. |
| `Scholar` | text, metadata, labels | vae | seed-reproducible | SCHOLAR (Card et al. 2018): a ProdLDA VAE with a covariate-shifted prevalence prior, an optional supervised label head, and optional content (topic-covariate) word deviations — neural STM prevalence + sLDA + SAGE. |
| `RTM` | text, links | variational | seed-reproducible | Relational topic model (Chang & Blei 2010): jointly models document text and a link graph (citations, hyperlinks, adjacency); predicts links from words and words from links. |
| `FactorialLDA` | text | gibbs | seed-reproducible | Factorial LDA (Paul & Dredze 2012): each token is a K-tuple of latent factors (e.g. topic x sentiment); structured word priors tie tuples sharing a component and a sparsity prior deactivates unsupported tuples. |
| `AuthorTopic` | text, metadata | gibbs | seed-reproducible | Author-Topic Model: each author has a topic distribution; documents mix their authors. Answers what an author writes about. |
| `AuthorRecipientTopic` | text, metadata | gibbs | seed-reproducible | Author-Recipient-Topic (McCallum et al. 2007): topics conditioned on the (sender, recipient) pair, for the language of a directed social network (who talks to whom about what). Realized over the AuthorTopic engine. |

#### Guided & supervised

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `SeededLDA` | text, seeds | gibbs | seed-reproducible | Seeded LDA: steer named topics toward supplied seed words. |
| `GuidedNMF` | text, seeds | matrix-factorization | seed-reproducible | Guided NMF: seed-word-guided semi-supervised NMF; the matrix-factorization analogue of SeededLDA. |
| `LabeledLDA` | text, labels | gibbs | seed-reproducible | Labeled LDA: each document label is a topic; tokens are restricted to its labels. |
| `SupervisedLDA` | text, labels | variational | seed-reproducible | Supervised LDA: topics shaped to predict a per-document real-valued response. |
| `DiscLDA` | text, labels | gibbs | seed-reproducible | Discriminative LDA (Lacoste-Julien et al. 2008): topics split into per-class and shared blocks; reads how classes talk differently. |

#### Short text

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `PT` | text | gibbs | seed-reproducible | Pseudo-document topic model: pool short texts into pseudo-documents. |
| `BTM` | text | gibbs | seed-reproducible | Biterm topic model: learns topics from corpus-level word co-occurrence (biterms). |

#### Dynamic & hierarchical

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `DTM` | text, times | variational | seed-reproducible | Dynamic topic model: a fixed topic set whose word distributions drift across time slices. |
| `DETM` | text, embeddings, times | vae | seed-reproducible | Dynamic embedded topic model: embedding-factored topics that drift across time slices, fit as an amortized VAE. |
| `TopicsOverTime` | text, times | gibbs | seed-reproducible | Topics over Time: LDA with a per-topic Beta density over continuous timestamps; each topic has a temporal peak. Descriptive continuous-time prevalence (not vocabulary drift). |
| `HLDA` | text | gibbs | seed-reproducible | Hierarchical LDA (nested CRP): a learned tree of super- and sub-topics. |
| `PA` | text | gibbs | seed-reproducible | Pachinko allocation: a DAG of super- and sub-topics. |

#### Embedding-based

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `KeyNMF` | text, embeddings | matrix-factorization | bit-exact | KeyNMF (Kristensen-McLachlan et al. 2024): NMF over an embedding-derived keyword-importance matrix. For each document it scores its words by the similarity between the document embedding and the word embedding, keeps the top-N positive, and factors that sparse doc-word matrix. The bridge between the count-based NMF family and the embedding backend; sparse, readable topics robust to short/noisy text. |
| `Top2Vec` | text, embeddings | clustering | seed-reproducible | Topics as dense regions in a joint document-word embedding space. |
| `SemanticSignalSeparation` | text, embeddings | ica | seed-reproducible | Topics as independent axes of semantic space (S3, Kardos et al. 2025): FastICA over the document embeddings, with each word's importance read off by projecting the vocabulary embeddings onto each axis. Signed poles. |
| `ETM` | text, embeddings | variational | seed-reproducible | Embedded topic model: topic-word distributions factored through word embeddings. |
| `GaussianLDA` | text, embeddings | gibbs | seed-reproducible | Gaussian LDA (Das, Zaheer & Dyer 2015): each topic is a Gaussian over the word-embedding space (Normal-Inverse-Wishart prior), so topics generalize over semantically similar words. Collapsed Gibbs with a Student-t posterior predictive and rank-1 Cholesky up/downdates. |
| `FASTopic` | text, embeddings | optimal-transport | seed-reproducible | Topics from optimal-transport plans between document, topic, and word embeddings. |
| `CombinedTM` | text, embeddings | vae | seed-reproducible | Contextualized ProdLDA: encoder reads the bag of words plus a document embedding. |
| `ZeroShotTM` | text, embeddings | vae | seed-reproducible | Contextualized ProdLDA: encoder reads the document embedding alone, enabling cross-lingual transfer. |
| `InfoCTM` | text, dictionary | vae | seed-reproducible | Cross-lingual: two ProdLDA models aligned by a bilingual dictionary through a mutual-information term. |

#### Ideal point

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `Wordfish` | text | em | bit-exact | Poisson scaling (Slapin & Proksch 2008): an unsupervised one-dimensional ideal-point estimate from word frequencies alone, no topics. The word-frequency baseline companion to IdealPointTM. |
| `Wordshoal` | text, metadata | em | bit-exact | Multi-domain scaling (Lauderdale & Herzog 2016): scales each debate/domain with Wordfish, then combines the within-domain positions into one cross-domain actor scale via a linear factor model. The multi-domain extension of Wordfish, for speeches carrying trusted debate labels. |
| `TBIP` | text | variational | seed-reproducible | Text-Based Ideal Points (Vafa, Naidu & Blei 2020): a Poisson factorization whose neutral topic-word intensities are rescaled by a per-word ideological factor exp(x_s * eta_kv), with the author position x_s latent. Fit by the paper's mean-field variational inference (reparameterized SVI). Recovers ideological scales from unlabeled text. |
| `PartyEmbeddings` | text, metadata | neural-embedding | seed-reproducible | Party embeddings (Rheault & Cochrane 2020): a PV-DM paragraph-vector model trained by negative sampling with party-period metadata tags; the leading principal components of the learned party vectors give the ideological scale, and words share the space so a party's language can be read off by proximity. The corpus-trained word-embedding member of the ideal-point family. |

#### LLM-based

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `TopicGPT` | text, llm | prompting | llm-bounded | LLM-driven topic discovery: prompt a model to propose, refine, and assign a topic taxonomy with descriptions. |

### Experimental

Not (yet) on the validated roster, on one of two grounds: the model is unpublished (a topica original with no paper), or it is a published method whose benefit has not held up against a simpler baseline. A published method with faithful inference stays validated even when its basis is planted-recovery, so planted validation alone does not land a model here. Gated: call `topica.enable_experimental()` (or set `TOPICA_EXPERIMENTAL=1`) before use. These may change or be removed without a deprecation cycle. For the triple gate a model clears to graduate to the validated roster, and where every model stands on validation evidence, see [Validation & graduation](https://nealcaren.github.io/topica/contributing/validation/).

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `TensorLDA` | text | svd | seed-reproducible | Online Tensor LDA (Kangaslahti et al. 2026): deterministic method-of-moments topic modeling via second and third-order cumulants. |
| `NarrativeTM` | text | gibbs | seed-reproducible | Intra-document narrative trajectory model: captures how topic prevalence shifts across the progress of a text. |
| `CSATM` | text, links | gibbs | seed-reproducible | Conversational Structure Aware TM (Sun et al. 2020): weights each comment's tokens by a reply-tree 'popularity' score and, after Gibbs, smooths each comment's topics toward its ancestors along the reply path ('transitivity'). For threaded forum data (posts + nested comments). Ported from the paper (no reference implementation); validated by planted recovery + LDA reduction. |
| `ReplyTM` | text, links, metadata | gibbs | seed-reproducible | Reply-conditioned TM (topica original, #810): a child comment's topic prior is shifted by a learned directed response matrix T applied to its parent's topic proportions, per covariate group (T[i,j] = response mass a topic-i parent places on child topic j), reported with posterior credible intervals. For threaded discussions where the question is how a community responds to a theme, not just which themes cluster. Collapsed Gibbs with T sampled; validated by planted recovery + an exact tiny-tree enumeration gate + LDA reduction. |
| `IdealPointTM` | text, embeddings | variational | seed-reproducible | Topic model with a latent ideal-point head: each author gets a low-dimensional position that shifts within-topic word choice, with a per-topic discrimination. Consumes word tokens as counts (Wordfish with topics) or, when word embeddings are supplied to fit, factored through them as in ETM. The unsupervised, latent-trait twin of the STM content covariate. |
| `IdealPointSentenceTM` | text, embeddings | em | seed-reproducible | Continuous ideal-point topic model over sentence/document embeddings: topics are Gaussian clusters whose centroids are displaced by a latent author position. The sentence-embedding sibling of IdealPointTM, fit by EM. |
| `EmbeddingLDA` | text, embeddings | gibbs | seed-reproducible | LDA anchored by pre-trained embeddings: k-means clusters the vocabulary embeddings, seeds each topic with the words nearest a cluster centroid, and (optionally) biases each document's mixture toward its own embedding. A topica original; validated by planted-recovery only. |

<!-- END MODEL TABLE -->

## LDA

Classic Latent Dirichlet Allocation via MALLET's fast SparseLDA collapsed-Gibbs
sampler. Fits are bit-for-bit reproducible, with optional approximate
multi-threaded training.

```python
import topica
model = topica.LDA(num_topics=20, seed=13)
model.fit(docs, iters=1000)
model.top_words(10)
```

### Inference choice: SparseLDA, WarpLDA, LightLDA, and CVB0

`LDA` ships four interchangeable inference backends for the *same model*,
selected with `sampler=`:

- **`"sparse"`** (default) — MALLET's SparseLDA collapsed-Gibbs sampler,
  `O(K_d + K_w)` per token. Near-optimal for the topic counts typical of social
  science; the fastest, highest-coherence choice up to roughly `K = 200`.
- **`"warp"`** — the cache-efficient two-pass MH sampler of
  [Chen et al. (2016)](https://www.vldb.org/pvldb/vol9/p744-chen.pdf). It holds
  the count tables fixed while every token samples (a delayed-update MCEM
  scheme), which lets each pass touch a single count matrix, for `O(1)` work per
  token with a per-sweep cost that is **flat in `K`**. This is the sampler for
  fine-grained, large-`K` models: at `K = 1,000` on a 2,000-document corpus it
  fits several times faster than SparseLDA *and* reaches higher coherence
  (SparseLDA is too slow to mix well at that `K`), and it beats LightLDA on both
  speed and coherence.
- **`"lightlda"`** — the alias-table MH sampler of
  [Yuan et al. (2015)](https://arxiv.org/abs/1412.1576), `O(1)` per token via
  word/document proposal alias tables. Superseded by `"warp"`, which is faster
  and mixes better at the same `K`; retained for compatibility and as an
  independent cross-check.
- **`"cvb0"`** — collapsed variational Bayes, zeroth-order
  ([Asuncion et al. 2009](https://arxiv.org/abs/1205.2662)). A *deterministic*,
  non-sampling backend: each (document, word-type) cell keeps a soft topic
  responsibility updated from expected counts. It has no burn-in, is exactly
  reproducible for a seed, and tends to give **higher topic coherence**,
  increasingly so at larger `K` (on a 2,000-document corpus at `K = 100`, mean
  `c_v` −68.5 against −79.1 for `"sparse"`). The catch is `O(K)`-per-token
  compute, so it is **slower, not faster** (≈47s vs ≈10s at `K = 100`), and it
  produces no MCMC `theta_draws`. Reach for it when topic quality matters more
  than fit time.

```python
# Fine-grained, large-K model, fast: WarpLDA.
model = topica.LDA(num_topics=1000, seed=1, sampler="warp")
model.fit(docs, iters=1000)

# Highest-coherence topics, fit time not a constraint: CVB0.
model = topica.LDA(num_topics=100, seed=1, sampler="cvb0")
model.fit(docs, iters=300)
```

All four target the same model. Use the default `"sparse"` up to a couple
hundred topics; `"warp"` for large-`K` (`K ≳ 500`) work where speed matters; and
`"cvb0"` when you want the cleanest topics and can spend the compute.

## TopicalNGrams

`TopicalNGrams` ([Wang, McCallum and Wei 2007](https://doi.org/10.1109/ICDM.2007.86)) extends LDA to discover topics *and* the multiword phrases that belong to them, in one fit. Alongside a topic, every token gets a binary bigram-status: does it continue a phrase from the previous word, given that word's topic? A run of continuations is a phrase — "machine learning", "supreme court" — and because the status is topic-conditioned, the same word can head different phrases in different topics. Phrases are learned *during* fitting, so a word's role as a collocation is discovered, not fixed in advance.

```python
model = topica.TopicalNGrams(num_topics=20, seed=13)
model.fit(docs, iters=1000)          # docs are token lists; ORDER matters
model.top_words(10)                  # per-topic unigrams (the usual topic view)
model.top_phrases(10)                # the top multiword phrases, pooled across topics
model.top_phrases(topic=3, n=20)     # topic 3's phrases, as (phrase, probability)
model.topic_word                     # (K, V) unigram topic-word, and doc_topic, coherence, save/load
```

Inference is a collapsed Gibbs sampler that draws the `(topic, bigram-status)` pair jointly for each token: a token is emitted either from its topic's unigram distribution or, when it continues a phrase, from a topic-and-previous-word bigram distribution, with the status governed by a small Beta prior. The fit is seed-reproducible. Because token order carries the phrases, a word dropped by `min_count` breaks the adjacency of its neighbours — "New \<stopword\> York" never becomes "New_York" — so pass raw token lists (not a pre-pruned corpus) when boundary breaks matter.

**Confirm your corpus preserves word order before trusting `top_phrases`.** TopicalNGrams reads adjacency as co-occurrence, so a bag-of-words corpus (tokens sorted or shuffled per document, as many published/preprocessed datasets are — including topica's own `load_poliblog()`) makes it manufacture phrases that are ordering artifacts, not real collocations. `fit` warns when most documents already look sorted, but the check is heuristic: if in doubt, use a corpus where the tokens are in reading order. `top_phrases(max_len=3)` drops the runaway whole-document "phrases" that a structureless corpus produces, leaving only tight collocations.

The one deliberate departure from the MALLET reference is the bigram-status prior. MALLET defaults to `delta1=0.2, delta2=1000`, which forces almost every token into a phrase, so its "phrases" degrade into whole-document runs; topica defaults to a balanced `delta1=delta2=1.0`, which recovers discrete, interpretable collocations. (Pass `delta1=0.2, delta2=1000.0` to reproduce MALLET's own prior.) We validate against Java MALLET's `TopicalNGrams` on a planted-collocation corpus with the same hyperparameters on both sides: the two recover the same topics (unigram topic-word aligned about as closely as two MALLET runs with different seeds) and the same planted collocations as phrases — an identical 6 of 12 in one run (`parity/tng_mallet_compare.py`).

This is not a replacement for phrase preprocessing. topica's [`learn_phrases()` / `apply_phrases()`](preprocessing.md) build a fixed phrase vocabulary *before* fitting and remain the simpler, often preferable workflow; reach for `TopicalNGrams` when topic-specific collocations are themselves the object of study, not merely a way to tidy topic labels.

## OnlineLDA

Online (streaming) variational-Bayes LDA — the minibatch stochastic-VB algorithm
of [Hoffman, Blei & Bach (2010)](https://papers.nips.cc/paper/2010/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html),
and the direct analogue of gensim's `LdaModel`. Where the batch-Gibbs [`LDA`](#lda)
above touches every document each sweep, `OnlineLDA` fits in **minibatches** with a
decaying learning rate `ρ_t = (tau + t)^(−kappa)` and never holds the whole corpus
in memory, so it reaches a good fit in a fraction of an epoch on very large
corpora. It also supports **streaming updates**: fold new documents into an
already-fitted model with `partial_fit`.

```python
import topica

# Batch online-VB fit (passes over the whole corpus in minibatches).
model = topica.OnlineLDA(num_topics=20, batch_size=256, seed=13)
model.fit(docs, iters=10)          # iters = passes over the corpus
model.top_words(10)

# Streaming: fit an initial batch, then fold in later minibatches.
model = topica.OnlineLDA(num_topics=20, batch_size=1024, total_docs=1_000_000, seed=13)
model.fit(first_chunk, iters=1)
for chunk in later_chunks:         # each chunk a list[list[str]]
    theta_chunk = model.partial_fit(chunk)   # (len(chunk), num_topics)

# Held-out inference without updating the model.
theta_new = model.transform(held_out_docs)
```

The constructor mirrors gensim's `LdaModel`: `batch_size` ↔ `chunksize`, `tau` ↔
`offset`, `kappa` ↔ `decay`, `beta` ↔ `eta`, `inner_iters` ↔ `iterations`, and
`fit(iters=)` ↔ `passes`. The vocabulary is fixed at the first `fit` (as gensim
fixes it via its `id2word` dictionary); tokens outside it are dropped by
`partial_fit` / `transform`.

**When to prefer it.** Reach for `OnlineLDA` when the corpus is too large to hold
in memory, arrives as a stream, or must be updated incrementally. For a corpus
that fits comfortably in memory, the batch samplers (or `cvb0`) usually give
better topics per unit of compute — `OnlineLDA` trades some statistical
efficiency for streaming scalability. Fits are seed-reproducible (identical for a
fixed seed and batch schedule). The E-step, sufficient-statistic accumulation, and
global blend reproduce Blei/Hoffman's reference `onlineldavb.py` and gensim's
`LdaModel`; see `parity/online_lda_gensim_compare.py`.

## STM

The full Structural Topic Model: CTM core plus **prevalence** and **content**
covariates. This is the workhorse for social science; it has its own
[guide](covariates.md).

Like CTM, STM takes `variational="diagonal"` to use the mean-field E-step in
place of the default Laplace one (`variational="laplace"`): faster at high K, but
it drops the off-diagonal posterior covariance, so the precision of
topic-correlation and method-of-composition standard errors is lower.

**Wording over ordered time.** Passing an ordered `content_time=` covariate
alongside `content=` crosses the content group with the period into a
base-by-period content design, tied across adjacent periods by a first-order
random walk (`content_smooth`), with an optional sparse L1 prior on the content
deviations (`content_prior="l1"`, `content_prior_var=`). It reduces to plain STM
when `content_time=None`. The reading layer
`topica.content.content_trajectory` (per-word between-group contrast across
periods) and `content_divergence` (whole-distribution group distance per period)
reads that surface, with a design-preserving document/cluster bootstrap for
confidence bands. `examples/stm_content_time_platforms.py` works this through
end to end on U.S. party platforms (1948-2024): inside a stable "environment"
topic, `climate` and `clean` enter the Democratic vocabulary after 2000 while
Republicans never adopt them, and the partisan divergence widens — evolution a
fixed-vocabulary dynamic model cannot represent.

## TensorLDA

!!! warning "Experimental — validation in progress"
    TensorLDA implements the published Online Tensor LDA method of
    Kangaslahti et al. (2026), but topica's Rust implementation has not yet
    cleared the project's reference-parity and known-truth recovery bar. Enable
    it explicitly with `topica.enable_experimental()`. Treat `weights` as model
    diagnostics rather than calibrated topic prevalence. See the
    [TensorLDA validation record](../replications/tlda.md) for the current
    evidence and limitations.

TensorLDA is a method-of-moments topic model: it whitens second-order count
moments and fits a factorized third-order cumulant. It is most useful when you
want a fast, count-based experimental alternative for large corpora. It is not
the right default for covariate-effect or prevalence-measurement questions;
prefer STM or DMR for those. Beyond the in-memory `fit`, it also supports a
streaming `partial_fit(batch, batch_index)` / `finalize()` path (incremental
whitening + per-batch factor SGD) that builds the model one batch at a time
without holding the whole count matrix; see the
[validation record](../replications/tlda.md#streaming-online-fit).

```python
topica.enable_experimental()
m = topica.TensorLDA(num_topics=20, n_eigenvec=20, seed=13)
m.fit(docs)
print(m.top_words(10))
```

## STS

The Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024) extends
STM with a per-document, per-topic **continuous sentiment-discourse** latent that
shifts the wording within a topic, with both topic prevalence and sentiment driven
by document covariates. Use it when you want to measure not just *which* topics a
covariate predicts, but *how* — the tone and slant with which each topic is
discussed.

```python
m = topica.STS(num_topics=10, seed=1)
m.fit(docs, sentiment_seed=rating, prevalence=X, prevalence_names=names)

m.doc_topic          # topic prevalence θ
m.sentiment          # per-document topic sentiment-discourse α^(s)
m.prevalence_effects # covariate → prevalence
m.sentiment_effects  # covariate → sentiment-discourse
m.topic_word_at(2.0) # how the topic is worded at high sentiment
```

`sentiment_seed` (one value per document — e.g. a star rating) seeds the sentiment
and defines the aggregation groups for the topic-word estimation. `kappa_estimation`
selects the topic-word estimator: `"ridge"` (default, fast) or `"lasso"` (matches
the reference R `sts` exactly, at higher cost); the two agree closely on
well-conditioned corpora. Validated against the authors' R `sts` implementation in
`parity/sts_r_compare.py` — on the political-blog corpus topica's STS aligns with
the published fit in the mid-0.90s (topic-word cosine), the same neighborhood as
topica's STM matches R's STM.

## CTM

The Correlated Topic Model (logistic-normal): topics can co-occur, unlike LDA's
Dirichlet. This is the engine STM builds on; `topic_correlation` reports the
learned structure, and `topica.topic_correlation_ci(model)` puts a credible
interval on each cell by propagating the per-document logistic-normal posterior
(it draws θ from `η_d ~ N(λ_d, ν_d)` and recomputes the correlation on each draw),
so you can tell a reliably signed topic relationship from one whose interval
straddles zero. Fit by parallel variational EM.

For corpora too large to sweep in full each EM step, `fit(..., inference="svi")`
switches to stochastic variational inference (online VB, Hoffman et al. 2013):
`iters` becomes the number of epochs, and the global topics, mean, and
covariance update from minibatches of `batch_size` documents (default 256) with
a Robbins-Monro step `ρ_t = (τ + t)^(-κ)` (`tau` default 64, `kappa` default
0.7). Each minibatch still runs STM's Laplace E-step per document, so the
per-token variational quality matches the default `inference="batch"`; the gain
is that one epoch touches every document while the global state stays
minibatch-sized. It is deterministic for a seed but keeps no per-iteration
`bound` trace.

```python
model = topica.CTM(num_topics=50, seed=1)
model.fit(big_corpus, iters=20, inference="svi", batch_size=512)
```

By default the per-document E-step uses the Laplace approximation
(`variational="laplace"`), forming the full posterior covariance `ν = H⁻¹`.
Passing `variational="diagonal"` switches to a mean-field diagonal covariance,
`ν = diag(1/H_ii)`, which skips the per-document Cholesky and inverse for a large
E-step speedup at high K. The cost is that the off-diagonal posterior covariance
is dropped, so the precision of `topic_correlation` and the method-of-composition
standard errors is lower.

```python
model = topica.CTM(num_topics=200, variational="diagonal", seed=1)
model.fit(corpus)
```

## DMR

Dirichlet-Multinomial Regression: each document's topic prior depends on its
metadata, `α_d = exp(Xγ)`. The learned `feature_effects` show how covariates
shift topic propensity, and `feature_effect_se` reports the standard error of
each, so you can tell a real effect from noise.

```python
import numpy as np
X, names = topica.design.one_hot(party)
model = topica.DMR(num_topics=20, seed=1)
model.fit(docs, X, feature_names=names)
z = model.feature_effects / model.feature_effect_se   # |z| > ~2 ⇒ notable
```

`feature_effect_se` is the standard error of each weight λ from the observed
information of the penalized Dirichlet-multinomial likelihood at the fit — the
curvature of the very objective L-BFGS maximizes to estimate the effects, so it
is exact (no bootstrap) and computed once at fit time. The topics couple through
the Dirichlet normalizer, so the full cross-topic Hessian is inverted rather than
a per-topic approximation. `GDMR` exposes the same getter, rescaled to its
Legendre basis.

Like `LDA`, `DMR` accepts the alternate inference backends via `sampler=`:
`"warp"` (WarpLDA with a per-document-α doc phase) for fine-grained, large-`K`
models — flat per-sweep cost in `K`, several times faster than the default
`"sparse"` sweep at `K ≳ 500` — and `"cvb0"` (deterministic collapsed
variational Bayes; the soft expected counts feed the λ optimizer directly) for
higher-coherence topics when fit time is not the constraint. `SeededLDA` takes
the same two. Use the default `"sparse"` up to a couple hundred topics.

## GDMR

Generalized DMR (g-DMR; Lee & Song 2020): DMR over one or more *continuous*
metadata variables, where the covariates enter through a Legendre-polynomial
basis and a decay prior smooths higher-order terms. The result is a topic
distribution function (TDF) you can read off at any metadata value, so you can
trace how each topic's prevalence varies smoothly along a continuous axis (year,
citation impact, age).

```python
model = topica.GDMR(num_topics=20, degrees=[3], seed=1)
model.fit(docs, year, metadata_names=["year"])   # features=/covariates=/metadata= all accepted
curve = model.tdf_linspace(1990, 2020, num=31)   # (31, num_topics) prevalence surface
```

`GDMR` mirrors `DMR`'s interface; `degrees`, `metadata_range`, and the prior
scales `sigma`/`sigma0`/`decay` configure the basis, and `tdf` / `tdf_linspace`
evaluate the fitted surface. `metadata_names` labels the continuous dimensions;
`feature_names` then labels the derived Legendre basis terms (e.g. `year^2`),
aligned with `feature_effects`. Because a continuous covariate's per-degree
coefficients are rarely interpretable on their own, read the surface with `tdf`
rather than the individual basis coefficients.

## Factorial LDA

Factorial LDA (fLDA; Paul & Dredze 2012) models each token as a **K-tuple** of
latent factors rather than a single topic. The canonical case is two factors —
topic × sentiment, or topic × perspective — but the model takes any number: a
document about (ECONOMICS, CONSERVATIVE) shares economics words with (ECONOMICS,
LIBERAL) and conservative words with (ENVIRONMENT, CONSERVATIVE). That sharing is
what makes it factorial rather than an LDA with `∏ Z_k` unrelated topics: a
structured log-linear word prior `ω` ties every tuple that shares a component, and
a relaxed sparsity prior can switch unsupported tuples off.

```python
model = topica.FactorialLDA(factor_sizes=[3, 2], seed=1)   # 3 topics x 2 sentiments
model.fit(docs, iters=2000, samples=100)
model.topic_word          # (6, V) — one word distribution per (topic, sentiment) tuple
model.tuples              # [[0,0], [0,1], ...] tuple index -> factor components
model.factor_top_words(0, 1)   # top words for component 1 of factor 0 (the "overview")
model.tuple_activity      # b_x in (0,1); a tuple with b_x <= 0.5 is effectively off
```

`factor_sizes` lists the components per factor, so `num_topics` is the product
`∏ Z_k`. Fit is Monte Carlo EM: a collapsed Gibbs sweep over tuples, then one
gradient step on the log-linear `α` (document prior) and `ω` (word prior) weights
and the sparsity logits `β` — the same Dirichlet-multinomial gradient `DMR` uses.

The paper's ablations are flags: `word_priors=False` drops the structured word
priors (untied factors), `sparsity=False` fixes every tuple on, and
`symmetric_word_prior=True` fixes the per-word background to zero. An ablation
removes its term outright, so it overrides any seed you pass for that term. You can
otherwise seed the word priors with domain knowledge (the drug-experiences setting
of Paul & Dredze 2013) by passing `omega_priors=` to `fit`:

```python
model.fit(docs, omega_priors={"components": {
    (1, 0): {"great": 2.0, "love": 2.0},     # steer factor 1, component 0 -> positive
    (1, 1): {"awful": 2.0, "hate": 2.0},     # component 1 -> negative
}})
```

Two caveats worth knowing. First, which factor becomes which axis is **not fixed**:
the factors are role-exchangeable and `ω` is identified only up to a per-word shift
(as in the reference), so on an unsupervised run the "topic" axis may land on
factor 0 or factor 1 — align factors to your labels rather than assuming an order,
or seed with `omega_priors` to pin an axis. Second, the block sampler enumerates all
`∏ Z_k` tuples per token, so cost and memory grow multiplicatively in the number of
factors; for many factors raise `block_freq` to sample each factor independently
(additive cost, at some loss of mixing).

**Supervise a factor (semi-supervised).** Because the unsupervised factors are only
weakly identified (below), the reliable way to *get* a topic × sentiment or topic ×
party decomposition is to tell the model what the second axis is. Pass
`observed_factors={factor: labels}` to `fit`, where `labels` has one entry per
document — the observed component of that factor (`0..Z_factor`) or `None` to leave
it latent:

```python
# sentiment known for some reviews (0/1), unknown (None) for the rest
labels = [0, 1, None, 1, None, ...]          # one per document
m = topica.FactorialLDA(factor_sizes=[8, 2], seed=1)
m.fit(docs, observed_factors={1: labels})    # factor 1 IS sentiment now
```

A labeled document's factor is pinned to its label for every token (its topic
distribution lives on that sub-simplex); an unlabeled document's value for that
factor is inferred, and you can read it back by marginalizing `doc_topic` over the
other factors. On 2000 movie reviews, labeling a random 10% lifts transductive
sentiment recovery on the other 90% from 0.52 (unsupervised, ≈chance) to ~0.71, and
50% labels to ~0.77 — better than a seeded lexicon, and it also cleans up the topic
factor by explaining sentiment away. This is *transductive label completion* (the
unlabeled documents take part in the fit), not held-out prediction, and fLDA is a
generative topic model, not a discriminative classifier — treat it as a
semi-supervised joint topic × axis model. When a factor is fully observed its
per-component word weights still learn a sensible lexicon, but its `α` prevalence
terms become confounded with the document bias and should not be read as a latent
prevalence.

**When to reach for it — and when not.** Be realistic about what the extra factors
buy you *unsupervised*. fLDA reliably learns a good *topic* factor, but the second (and later)
factors are only weakly identified: unsupervised, they tend to absorb *more topic
variation* rather than discovering a latent sentiment or perspective axis, because
subject matter dominates the lexicon. On the STM poliblog posts, an unsupervised
`[8, 2]` fit recovers the Conservative/Liberal axis at only ~0.59 (chance 0.5); on
2000 labeled movie reviews the second factor tracks true sentiment at ~0.52 —
essentially chance — and the *original* Java implementation does no better on the
same data (~0.63, one non-reproducible draw). Seeding the axis with `omega_priors`
(a sentiment or partisan lexicon) helps but only to a moderate ~0.67. This is a
property of the method, not the port, and is much of why fLDA never displaced
simpler models. So:

- If you **already have** the second axis (sentiment labels, ideology, discipline),
  a model built to condition on it — [SAGE](#sage), [STM](#stm)/[STS](#sts), or
  [SeededLDA](#guided-topics) — will use it more robustly than fLDA's unsupervised
  factors.
- If you **don't**, don't expect fLDA to hand you a clean sentiment/perspective axis
  for free; at minimum seed it with `omega_priors`.
- The output that *is* worth having is the **tuple view**: the same topic worded
  differently across a (seeded or inferred) second factor — e.g. a war topic framed
  as "terrorist/attack" versus "Iraq/troops/Bush." Read the tuples, not just the
  per-factor overviews.

The reference is Michael Paul's GPL Java, which draws a fresh unseeded RNG per token
and is not reproducible, so topica claims no bit/seed parity against it. Instead the
port's correctness is certified directly: finite-difference gradient tests on every
weight group and factor-tying invariant tests (`src/factorial_lda.rs`), plus
tuple-level recovery of a planted topic × sentiment corpus
(`tests/test_factorial_lda.py`). `parity/factorial_lda_compare.py` puts a topica run
next to a Java run for a loose eyeball comparison only — it lines up topica's
*posterior* topic-word against a *prior* reconstruction from the Java component
weights (without the word background term), so it is a sanity check, not a fidelity
measurement; the gradient and recovery tests are what substantiate faithfulness. A
fixed seed reproduces bit-for-bit.

## CSATM

CSATM (Conversational Structure Aware and Context Sensitive Topic Model; Sun,
Loparo & Kolacinski 2020) is an LDA variant for **threaded forum discussions** —
posts with nested reply trees, one comment per document. It uses the reply tree in
two ways. First, **popularity**: each comment gets a weight equal to the
level-discounted node count of its reply subtree, so a comment that draws many
replies carries more weight in inference (its tokens count for `lambda_ * p_c`
instead of 1 — the same weighted-count idea as KeyATM, but per comment rather than
per word). Second, **transitivity**: after Gibbs, each comment's topic distribution
is smoothed toward its ancestors along the path to the thread root, weighted so a
nearer ancestor (the comment it directly replies to) pulls harder than the root.

Pass the reply structure to `fit` as a `parents` list — `parents[d]` is document
`d`'s parent **index** (`-1` for a thread root). `fit` validates it: a value outside
`[-1, num_docs)`, a self-parent, or a cycle raises `ValueError` rather than fitting on
a scrambled tree.

**`parents` holds indices, so it is fragile to any step that reorders or drops
documents.** If you build a `Corpus` first (which prunes empty documents and reorders
survivors), the *positions* shift **and** some referenced parents disappear — so
realigning with `parents[corpus.kept_indices]` is wrong (that reorders the values but
leaves them pointing at old positions). Rebuild the indices from a stable id after
preprocessing. A worked example from Reddit-style data (a `fullname` / `parent_fullname`
export):

```python
import topica

# rows: each has fullname (e.g. "t1_abc"), parent_fullname ("t1_xyz" or "t3_post"),
# and text. Keep the comments you will model, in one fixed order.
rows = [r for r in load_threaded_csv() if r["row_type"] == "comment"]
docs = [topica.tokenize(r["text"]) for r in rows]

# Drop docs that tokenize to (near-)empty BEFORE building parents, so indices align.
keep = [i for i, d in enumerate(docs) if len(d) >= 3]
docs = [docs[i] for i in keep]
index_of = {rows[i]["fullname"]: new for new, i in enumerate(keep)}
# A parent that is a post, is missing, or was dropped becomes a thread root (-1).
parents = [index_of.get(rows[i]["parent_fullname"], -1) for i in keep]

m = topica.CSATM(num_topics=15, seed=13).fit(docs, parents, iters=1000)
```

Omit `parents` (or pass `None`) and every document is a root: popularity is 1
everywhere and transitivity is a no-op.

**`lambda_` and the LDA relationship.** `lambda_` scales the per-token popularity
weight `lambda_ * p_c`. The model reduces to ordinary LDA **only at `lambda_=1` with
no reply tree** (then every weight is 1). The **default `lambda_=0.1` is not LDA** — it
shrinks every token's count (a leaf counts 0.1), which leans *away* from the thread
structure; raise `lambda_` toward 1 (or higher) and sharpen `weight_d` to lean *into*
it. If you want an LDA baseline to compare against, use `CSATM(lambda_=1).fit(docs)`
with no `parents` as the honest null — not `topica.LDA`, whose separate sampler and
RNG stream will differ from CSATM's even when the two are mathematically equivalent.
The level-weight sequence (`weight_seq`, default `"arithmetic"`) must be non-increasing
and is shared by popularity and transitivity.

**Which doc-topic to report.** `doc_topic` is the transitivity-**smoothed** θ′ (the
default, and what prevalence tables read) — each comment's topics are partly borrowed
from its ancestors, which is the model's intent. `doc_topic_raw` is the pre-smoothing
Gibbs θ. For per-document prevalence that should reflect a comment on its own, report
`doc_topic_raw`; to reflect a comment *in the context of its thread*, report
`doc_topic`. They can differ substantially. `popularity` exposes the per-comment `p_c`
for auditing.

CSATM is **experimental**, and it is worth being precise about why. The inference is a
faithful port of the paper's equations, validated by planted-topic recovery, a
degenerate-case reduction (all-roots + `lambda_=1` collapses the weighted counts to
ordinary LDA), and determinism (a fixed seed reproduces bit-for-bit). For a published
method with faithful inference, that planted-recovery basis is *not* what makes a model
experimental, and the absence of a public reference implementation is not either —
planted recovery is an accepted accuracy basis when no maintained reference exists. The
flag is about the benefit not holding up: as the coherence results below show, the
conversational structure does not measurably improve topics over plain LDA once the
count-scale confound is controlled, and the paper's reported gains are not reproduced
here.

**On coherence, do not expect a free win over LDA.** On a real threaded Reddit corpus
(r/tradwives) the conversational structure did **not** measurably improve topic
coherence over plain LDA once the confound is controlled: raising `lambda_` inflates
token counts relative to the priors, which lifts within-corpus u_mass on its own, so a
naive "CSATM beats LDA" u_mass margin is mostly that count-scale effect. In a
scale-matched test (structure on vs. off at equal mean token weight) the structural
effect on coherence was ~0 (u_mass −0.2, c_npmi −0.003), and plain LDA was in fact the
most coherent on sliding-window c_npmi. CSATM's contribution is the thread-aware topic
*assignment* and the `popularity` / `doc_topic_raw` diagnostics, not a coherence
upgrade. The paper's reported coherence and assignment-accuracy gains were measured
against external references with tuned hyperparameters and are not reproduced here.

## ReplyTM

ReplyTM (a topica original, issue #810) is for **threaded discussions** where the
substantive question is not only which themes appear but **how a community responds to
them**: when a comment is about theme A, what do its replies tend to be about? CSATM
(above) models the tendency of a reply to resemble its parent (homophily along the
thread). ReplyTM instead learns a directed **response matrix** `T`, where `T[i, j]` is
the share of response mass that a topic-`i` parent places on child topic `j`. The
off-diagonal of `T` is the object of interest: it reads "parents about *i* tend to draw
replies about *j*," which a symmetric smoother cannot express.

A child comment's topic prior is its parent's response contribution added to a
per-group baseline:

```text
child concentration  a = exp(b_g) + rho_g * T_g.T @ zbar_parent
```

`zbar_parent` is the parent's topic proportions, `rho_g` the **response strength**
(how hard the parent pulls the child, separated from the shape `T`), and `exp(b_g)` a
baseline concentration per covariate group. Root posts fall back to the baseline alone.
`T`, `rho`, and the baseline are sampled during fitting, so every cell of `T` is
reported with a posterior credible interval.

Pass the reply structure as `parents` (the same contract as CSATM: `parents[d]` is
document `d`'s parent **index**, `-1` for a thread root; the worked Reddit example under
CSATM applies unchanged). Pass an optional `covariate`, a 0-based integer group label
per document, to estimate a **separate response matrix per group**: the motivating use
is contrasting how two communities respond to the same themes.

```python
import topica
topica.enable_experimental()

m = topica.ReplyTM(num_topics=20, seed=13).fit(docs, parents=parents,
                                                covariate=subreddit_id, iters=1000)

m.response_matrix        # list of (K, K) arrays, one per group; rows sum to 1
m.response_matrix_lower  # 2.5% credible bound per cell
m.response_matrix_upper  # 97.5% credible bound per cell
m.response_strength      # rho per group (with response_strength_lower/upper)
```

Pass `covariate_labels=["RedPillWomen", "TheRedPill"]` to `fit` so `group_labels`
and the readers carry names instead of positional `group_0`.

**Use the readers, not the raw matrix.** `topica.inspect.response_table(m, group=…)`
ranks a group's **directed** responses (the off-diagonal), scored by **lift** over
each child topic's base reply rate, with the diagonal excluded and the credible
interval on every row — this is the safe way to read the model, because the raw
matrix's largest cells are the diagonal (homophily) and its off-diagonal is easy to
over-read on a high-prevalence "attractor" topic. `topica.inspect.response_contrast(
m, group_a, group_b)` differences two groups' responses and flags the cells whose
credible intervals **separate**; treat only those as defensible community
differences, and confirm them across seeds.

**Do not compare `rho` across groups naively.** `response_strength` (with
`response_strength_lower/upper`) is on the model's own scale and is entangled with the
baseline and the shape `T` — because `a = exp(b_g) + rho_g · Tᵀ z̄`, a larger `rho_g`
does not by itself mean "community g responds harder." Report `rho` as a within-group
magnitude, not a cross-group ratio; the defensible cross-group object is the response
*shape* contrast from `response_contrast`, read off its credible intervals.

**Convergence and uncertainty.** `m.fit_history` is a `(sweep, log-likelihood)` trace
— it should rise and plateau; a still-climbing tail means raise `iters` before
trusting the intervals (`m.converged` is a coarse flag). The credible intervals come
from a single chain, so they can be optimistic on sparse cells; **fit at 2–3 seeds
and keep only responses stable across them** before reporting a cell.

**Choosing K.** ReplyTM's word model is LDA, so pick K with the LDA selector —
`topica.select.search_k(docs, [8, 12, 16, 20], model="lda")` (the second argument is
the list of K values to score) — then fit ReplyTM at the chosen K.

**Speed.** The reply-coupling makes each sweep heavier than plain LDA (every parent
token re-scores its children). Pass `num_threads=N` to `fit` for AD-LDA
parallelism — whole conversations are partitioned across workers, so the fit stays
reproducible for a fixed `num_threads` + `seed` (but differs across thread counts;
`num_threads=1` is the exact serial fit), and topic recovery holds at high worker
counts. The parallel speedup is modest, though: only the token sweep is threaded,
while the Metropolis parameter block (sampling `T`, `rho`, and the baseline) stays
serial, so throughput plateaus after a few threads — most of ReplyTM's speed over a
naive implementation is algorithmic (a first-order fast path for the child
log-Dirichlet-multinomial), not from threads. A few hundred iterations at a modest K
is usually enough; watch `fit_history` to decide.

**Reading `T` honestly.** The **diagonal** of `T` restates homophily (topics persist
down a thread), which is also what CSATM's transitivity captures; the novel signal is in
the **off-diagonal**, so inspect it separately and lean on the credible intervals before
claiming a directed effect. To decide whether the response really differs between two
groups, compare `response_matrix[g1]` and `response_matrix[g2]` cell by cell and treat a
difference as real only when the credible intervals separate. If they do not, the
`covariate_response="shared_shape"` setting (one shape, per-group strength) or `"global"`
(one matrix) is the more honest report.

**Short comments.** Forum replies are short, and a per-comment topic estimate from ~30
tokens is noisy. ReplyTM is robust to this because `T` pools evidence across thousands
of parent–child edges, so the per-comment noise averages out; the estimate that suffers
first on very short comments is `rho` (the strength), which a weakly-informative prior
shrinks. `T`'s *shape* holds up well down to realistic comment lengths.

**Why not just cross-tabulate?** The obvious baseline is to fit flat topics, take each
comment's topic, and cross-tab parent topic against child topic. That is a useful first
look and usually agrees with `T` qualitatively. But it treats each comment's topic as
*observed* when it is *estimated* from a few noisy tokens: the resulting response is
attenuated toward the base rate (regression dilution), and its uncertainty is wrong,
because a cross-tab confidence interval cannot know the topics themselves are uncertain.
ReplyTM is the joint (errors-in-variables) version — it samples the topics and `T`
together, so `T`'s credible intervals propagate the topic-estimation uncertainty, and it
separates response *strength* (`rho`) from response *shape*. Reach for it when you want
a directed-response number with a defensible interval, not just an exploratory heatmap.

**`parents` is index-fragile.** As with CSATM, `parents[d]` is a document *index*, so any
step that prunes or reorders documents (building a `Corpus`, dropping empties) invalidates
it — rebuild the indices from a stable id after preprocessing rather than reindexing the
values. See the worked Reddit example in the CSATM section above.

**Validation and status.** ReplyTM is a topica original with no reference
implementation, so it is **experimental** (gate it with `enable_experimental()`) and
validated by planted recovery: on synthetic reply forests with a known asymmetric `T`,
the fit recovers the planted directed responses and the response strength on the model's
own scale. The sampler's correctness is pinned by an exact enumeration test on a tiny
tree (`src/reply_tm.rs`), which checks that the collapsed-Gibbs conditional includes the
term coupling a parent to its children and would fail a sampler that dropped it. With no
reply tree (or `rho=0`) and one group, the fit reduces to LDA.

## Scholar

SCHOLAR ([Card, Tan & Smith 2018](https://aclanthology.org/P18-1189/)) brings
covariates to the *neural* topic models. `STM`, `DMR`, and `SAGE` relate topic
prevalence and content to document metadata in the count world; Scholar does the
prevalence part in a [`ProdLDA`](#prodlda) VAE. A `Linear(n_covariates, K)` layer
turns each document's covariates into a shift of its topic-prior mean,
`μ₀ = W·covariates`, and the KL then pulls the document's posterior toward that
covariate-dependent mean. A covariate that co-occurs with a topic raises that
topic's prevalence, and the fitted weights `W` read directly as a
covariate-by-topic prevalence-effect matrix (`covariate_effects`). This is the
neural analog of STM/DMR prevalence covariates, estimated *inside* the fit rather
than post-hoc on a fixed `θ`.

```python
model = topica.Scholar(num_topics=20, covariate_names=["year", "outlet"], seed=1)
model.fit(docs, covariates=X)          # X is (num_docs, n_covariates), numeric
model.covariate_effects                # (n_covariates, num_topics) prevalence effects
theta = model.transform(new_docs, new_X)   # covariates enter the encoder too
```

The covariates enter in two places, following the reference implementation
([dallascard/scholar](https://github.com/dallascard/scholar), Apache-2.0): they set
the prior mean (above), and they are concatenated to the encoder input so the
posterior can track the shifted prior. `l2_prior_reg` puts an L2 penalty on `W` to
shrink weak effects. Scholar builds on topica's existing ProdLDA backbone — the
encoder, reparameterization, product-of-experts decoder, batch normalization, and
Adam are shared with `ProdLDA` — so it is a mechanism-faithful port on that
backbone rather than a bit-for-bit clone of the reference's single-layer encoder.
The prior covariate path is logistic-normal (`prior="laplace"`), since a
prior-*mean* shift is only defined for the Gaussian latent. The new covariate-weight
gradient is hand-coded and checked against finite differences in the Rust tests.
Because the encoder is topica's two-layer AVITM network, larger topic counts need
somewhat more epochs than the reference's single-layer encoder to fully separate the
topics; if `covariate_effects` looks muddy, raise `iters` before reading it.

### Supervised labels

Pass `labels=` (one class per document, str or int) to add SCHOLAR's supervised head
— a softmax classifier off `theta` whose cross-entropy loss is trained jointly and
pushes a gradient back into the topics, so the topics become predictive of the label.
This is the neural analog of supervised LDA (sLDA). After fitting, `classes` lists
the label space and `predict` / `predict_proba` classify new documents.

```python
m = topica.Scholar(num_topics=20, seed=1)
m.fit(docs, labels=y)                 # covariates optional; labels-only is fine
m.predict(new_docs)                   # or m.predict_proba(new_docs) -> (n_docs, n_classes)
```

Covariates and labels compose: `m.fit(docs, covariates=X, labels=y)` fits the
prevalence prior and the supervised head together. One deliberate deviation from
`dallascard/scholar`: the reference also concatenates the label onto the *encoder*
input (and zeroes it at test), so its inference network is `q(θ | words, label)` at
train time but `q(θ | words, 0)` at prediction. topica supervises the topics only
through the classifier head's gradient — the sLDA mechanism, where the label is not
an inference-time input — so `q(θ | words)` is used at both train and test. This is
both principled (it removes the train/test input-distribution mismatch) and, for this
backbone, empirical: on topica's two-layer AVITM encoder, feeding the label in
degraded held-out label accuracy on a small planted corpus (it fell as training went
on), while supervising only through the classifier head reached full accuracy. The
reference's single-layer encoder at scale does not show this — reconstruction so
dominates the label loss that its encoder does not learn to copy the label — so the
effect is backbone- and regime-dependent, not a flaw in the reference. The label
supervision that shapes the topics, the classifier loss, is unchanged either way.

### Content (topic covariates)

Pass `content=` (a numeric matrix) to add SCHOLAR's third metadata role: content
covariates that change *how* topics are worded across groups, the neural analog of
[`SAGE`](#sage). Each content covariate gets a per-word deviation added to the
decoder logits (`content_effects`, shape `(n_content, vocab)`); with
`interactions=True` the model also learns topic×covariate deviations. `l1_content_reg`
applies a fixed-strength L2 (ridge) penalty that shrinks the deviations. This is a
simplification of the reference, which reweights the penalty per weight each epoch to
approximate an L1 (sparsity-inducing) prior; topica's fixed ridge shrinks but does not
sparsify. Both default to `0.0` (unpenalized additive deviations). The base
per-word background lives in the shared ProdLDA decoder (its batchnorm shift), not a
separate content bias term.

```python
m = topica.Scholar(num_topics=20, content_names=["outlet"], seed=1)
m.fit(docs, content=G)                 # G is (num_docs, n_content), numeric
m.content_effects                      # (n_content, vocab): per-covariate word shifts
```

Unlike labels, a content covariate is observed at prediction, so it enters the
encoder alongside the prevalence covariates (no train/test inconsistency) and also
drives the decoder deviations. All three roles compose:
`m.fit(docs, covariates=X, labels=y, content=G)` fits the prevalence prior, the
supervised head, and the content deviations together.

`topica`'s [`estimate_effect`](covariates.md) still works on any model's `θ`
post-hoc; Scholar's added value is putting the metadata *into* the fit, which better
identifies the topics and exposes `covariate_effects` / `content_effects` as
first-class outputs.

## NarrativeTM

!!! warning "Experimental — unvalidated"
    NarrativeTM ships before a published paper and a reference-implementation
    parity check, topica's bar for a validated model. It is an original
    construction. It is **gated**: call `topica.enable_experimental()` (or set the
    `TOPICA_EXPERIMENTAL=1` environment variable) before constructing or loading
    it, or construction raises. Treat its results as provisional, and expect that
    it may change or be removed without a deprecation cycle.

The Intra-Document Narrative Trajectory Model asks a question the document-level
models cannot: not *which* topics a corpus covers, but *where inside a text* each
topic tends to appear. Introductions, methods, and conclusions draw on different
topics; a news story opens on the event and closes on reaction. NarrativeTM
recovers that average arc from beginning to end.

It works by segmenting each document into ordered pieces, recording each piece's
**relative position** in `[0, 1]` (0 = start, 1 = end), and fitting a
[`GDMR`](#gdmr) with position as its single continuous covariate. The Legendre
basis GDMR already uses to trace prevalence along a continuous axis becomes, here,
the smooth topic-versus-position curve. Because the model is one GDMR underneath,
its topic-word estimates, `top_words`, and `coherence` behave exactly as GDMR's.

```python
topica.enable_experimental()               # NarrativeTM is experimental and gated
m = topica.NarrativeTM(num_topics=10, degree=3, segment_by="sentence", seed=13)
m.fit(docs, iters=1000)

m.top_words(10)                            # topics, read like any GDMR/LDA fit
traj = m.global_trajectory([0.0, 0.5, 1.0])   # (3, K): topic mix at start / middle / end
m.doc_topic                                # (D, K) document-level θ, token-weighted
```

`segment_by="sentence"` splits on sentence punctuation (`. ? ! ;`) and falls back
to fixed chunks when a document carries no such markers; `segment_by="chunk"`
(the default) always cuts fixed windows of `chunk_size` tokens. `degree` sets the
Legendre order of the position curve (3 captures a rise-then-fall arc; raise it
for more inflections, and the `sigma`/`sigma0`/`decay` priors carry over from
GDMR to keep higher orders from overfitting).

The distinctive method is `global_trajectory(t)`: it evaluates the fitted
position curve at any `t` in `[0, 1]` (scalar or array) and returns the topic
proportions the model expects at that point in a text. Sweeping `t` from 0 to 1
traces each topic's narrative arc, the intra-document analogue of GDMR's
`tdf_linspace` over calendar time. The document-level `doc_topic` is reconstructed
as a token-weighted average of the per-segment proportions, so it lines up with
the θ from a plain document-level fit and flows into the usual diagnostics.
`save`/`load` persist the model (the inner GDMR is written alongside);
`scripts/verify_narrative.py` fits it on a synthetic corpus with planted
beginning-middle-end structure and reports the recovered trajectory.

## DTM

The Dynamic Topic Model: a fixed number of topics whose word distributions
**drift** across ordered time slices. `word_evolution(topic, word)` traces one
word's probability through time, and `word_drift(topic)` reports *which* words
rose and fell most within a topic — what makes its vocabulary evolve.

```python
dtm = topica.DTM(num_topics=10, chain_variance=0.05, seed=1)
dtm.fit(docs, times, iters=20)   # `times` = per-doc slice index

drift = dtm.word_drift(topic=3)     # first vs last slice by default
print("rising: ", [w for w, _ in drift["rising"][:5]])
print("falling:", [w for w, _ in drift["falling"][:5]])
```

## HDP

A nonparametric model that **infers** the number of topics rather than taking
`K` as input. Useful as a sanity check on the `K` you chose elsewhere.

```python
hdp = topica.HDP(gamma=0.5, eta=0.3, seed=1)
hdp.fit(docs, iters=300)
print(hdp.num_topics, "topics inferred")
```

By default the concentrations `alpha`/`gamma` are **resampled from the data**
(`resample_conc=True`): topica's concentration-update equations reproduce those of
the reference HDP implementations — blei-lab/hdp's
`sample_first`/`second_level_concentration` and the Escobar-West/Teh (2006) scheme
— with added numeric guards, so the topic count is genuinely inferred. The
starting `gamma`/`alpha` seed the sampler but the data moves them; resampling is
still deterministic under a fixed `seed`.

topica follows **blei-lab/hdp**'s exact new-table marginal, so it discovers a
similar number of topics to blei on the same data. tomotopy's HDP tends to find
*fewer, cleaner* topics — and this is **not** because it keeps explicit per-word
tables (a faithful explicit-table sampler lands in topica's regime; we checked).
It is because tomotopy uses a **simplified new-table weight** that drops the
dominant `Σ_k m_k·f_k` term, under-weighting the creation of new tables and so
regularising toward fewer topics. Neither is a bug — they are different modelling
choices — but if you want tomotopy's coarser result, use tomotopy; topica stays
faithful to blei's marginal.

Set `resample_conc=False` to hold the concentrations fixed — but at the low
defaults that collapses to a few topics dominated by one background topic, so if
you fix them, raise `gamma` (larger discovers more topics). `concentration_max` is
a high divergence backstop on the resampled values, not a prior; on the corpora
tested it does not affect the fit.

Cost note: because the count is inferred, a large heavy-tailed corpus can grow to
many topics, and each Gibbs step is linear in the current topic count — so the
inferring default is more expensive per sweep than a small fixed `K`. If a fit
grows too many topics, lower `concentration_max` to cap the count, or set
`resample_conc=False` with a chosen `gamma`.

## Guided topics

`keyATM` and `seededlda` steer named topics with a few seed words each, for when you know the themes you expect. See the [guided-topics guide](guided.md).

## ProdLDA

ProdLDA ([Srivastava & Sutton 2017](https://arxiv.org/abs/1703.01488)) keeps
LDA's document model but replaces the word-level *mixture* of topics with a
*product of experts*: the word distribution is `softmax(βθ)` with an unnormalized
β, rather than `softmax(β)·θ`. This sharper word model reliably yields more
coherent topics than collapsed-Gibbs LDA. Inference is an amortized variational
autoencoder (the AVITM framework): an encoder network maps a document's bag of
words to a logistic-normal posterior over θ, trained by minibatch Adam on the
ELBO. There is no PyTorch dependency; the network is hand-coded in the Rust core.

```python
model = topica.ProdLDA(num_topics=20, seed=1)
theta = model.fit_transform(docs)      # one encoder pass per document
model.top_words(10)
```

Two details follow the paper's recipe for avoiding *component collapse* (topics
decaying onto the prior early in training): batch normalization on the encoder
heads and decoder, and high-momentum Adam (`β₁ = 0.99`). Because inference is
amortized, `transform` maps new documents with a single forward pass rather than
re-running an optimizer. ProdLDA is bag-of-words (no embeddings); for the
embedding-factored generative model see [`ETM`](embedding.md).

### Objective and prior options

`ProdLDA`, `CombinedTM`, `ZeroShotTM`, and `ETM(inference="vae")` share the same
amortized-VAE core, so two optional flags apply across all four. Both default off,
and the defaults reproduce the standard model exactly.

- `prior=` chooses the document-topic prior. `"laplace"` (the default) is the
  logistic-normal Laplace approximation to a Dirichlet from the AVITM paper.
  `"dirichlet"` puts a true Dirichlet prior on `θ` through the Weibull
  reparameterization (Zhang et al. 2018; Burkhardt & Kramer 2019): the encoder
  parameterizes a Weibull variational posterior on each unnormalized topic weight,
  a Weibull draw is normalized onto the simplex, and the analytic Weibull-to-Gamma
  KL replaces the logistic-normal KL. We reuse the same reparameterization noise the
  laplace path draws, so turning the flag off is bit-for-bit the original model.
  `"stick_breaking"` is the Gaussian stick-breaking construction (Miao,
  Grefenstette & Blunsom 2017; reparameterizable simplex map of Nalisnick & Smyth
  2017): it keeps the same Gaussian latent and Gaussian KL as `"laplace"`, but maps
  it onto the simplex by stick-breaking — `K-1` breaks `ηₜ = sigmoid(zₜ)` give
  `θₜ = ηₜ ∏_{j<t}(1 - η_j)` with the last topic the remainder. The ordered sticks
  let early topics claim most mass and later ones decay, a nonparametric-flavored
  prior that softens the fixed-`K` assumption. Because only the simplex map changes,
  the laplace default stays bit-identical.

- `contrastive=True` adds a CLNTM-style (Nguyen & Luu 2021) InfoNCE term on the
  topic vectors. For each document the *anchor* is its sampled topic vector and the
  *positive view* is the deterministic no-noise topic vector (`softmax(μ)` on the
  laplace path, the median Weibull on the dirichlet path, the no-noise stick-breaking
  of `μ` on the stick-breaking path); the other documents in
  the minibatch are negatives, with cosine similarity at temperature
  `contrastive_temp`. The term is scaled by `contrastive_weight` and added to the
  per-batch loss. We document the positive-view choice because it is what makes the
  term deterministic and finite-difference checkable; the TF-IDF salient-word
  positive construction from CLNTM is a future refinement.

```python
m = topica.ProdLDA(num_topics=20, prior="dirichlet",
                   contrastive=True, contrastive_weight=0.5, contrastive_temp=0.5)
m.fit(docs)
```

The two flags are orthogonal and compose: the contrastive term operates on `θ`
however `θ` was produced. Every new gradient path is hand-coded and checked against
finite differences in the Rust unit tests.

## InfoCTM

`InfoCTM` (Wu et al. 2023) is a **cross-lingual** topic model: it fits two languages
into a shared `K`-topic space so topic `k` denotes the same theme in both. It is two
`ProdLDA` models — one per language, over independent vocabularies — fit jointly and
aligned by a **Topic-Alignment Mutual-Information (TAMI)** term: a masked
cross-lingual InfoNCE over the topic-word columns whose positive pairs come from a
bilingual `dictionary` (optionally densified by per-language word `embeddings`). This
is the dictionary-grounded alternative to the embedding-based `ZeroShotTM` path: it
needs a bilingual lexicon rather than a multilingual embedder.

```python
m = topica.InfoCTM(num_topics=20, mi_weight=30.0, languages=("en", "zh"))
m.fit(corpus_en, corpus_zh, dictionary=en_zh_pairs)   # (word_en, word_zh) pairs
#       optionally: embeddings_en={word: vec}, embeddings_zh={word: vec}
m.topic_word(lang="en"); m.top_words(10, lang="zh")   # aligned across languages
```

Each language keeps the full fitted surface (`topic_word`, `doc_topic`, `top_words`,
`vocabulary`, `transform`) selected by `lang=`. The per-language model is exactly
`ProdLDA`, so its ELBO is the validated AVITM objective; the only added term is TAMI,
whose gradient is hand-coded and finite-difference checked. Determinism is
`seed-reproducible`.

Two training-recipe deviations from the reference, documented for anyone
reproducing the paper: the optimizer follows the InfoCTM reference (Adam,
`beta1=0.9`), not topica's ProdLDA `beta1=0.99`; and topica trains at a **constant**
learning rate, where the reference halves it every 125 epochs (a `StepLR` schedule).
Both leave the model and objective unchanged but can shift the final fit, so an exact
numerical match to a reference run is not expected.

## PolylingualLDA

`PolylingualLDA` (the Polylingual Topic Model, [Mimno, Wallach, Naradowsky, Smith &
McCallum 2009](https://aclanthology.org/D09-1092/)) is the **count-based**
cross-lingual model, LDA extended to aligned document *tuples*. A tuple is a set of
documents that are loosely equivalent — parallel translations, or comparable articles
such as linked Wikipedia pages — written in `L` languages. Every document in a tuple
shares **one** tuple-level topic distribution θ; each topic carries a *per-language*
word distribution φˡ. Because the topic index is shared, topic `k` denotes the same
theme in every language: the topics are aligned by construction, with no post-hoc
matching.

```python
m = topica.PolylingualLDA(num_topics=20)
m.fit({                       # a dict {language: documents}, aligned by index
    "en": docs_en,            # every language has the same number of tuples D;
    "fr": docs_fr,            # tuple d is the same item in each language
    "de": docs_de,
})
m.topic_word(lang="fr")       # (K, V_fr); topic k is the same theme as en's topic k
m.top_words(10, lang="de")    # aligned across languages
m.doc_topic                   # (D, K), shared across languages — one θ per tuple
```

Inference is collapsed Gibbs sampling. The conditional is standard LDA except the
document-topic count is pooled across all languages in the tuple, which is what binds
the languages onto a shared simplex. The asymmetric αm prior is re-estimated by a
Minka fixed-point step every `optimize_interval` iterations after an
`optimize_burn_in` warm-up (`optimize_alpha=True`, burn-in 200 by default, matching
MALLET; optimizing from iteration zero over-sparsifies and can starve a topic into a
merge); pass `optimize_alpha=False` for a fixed symmetric prior. A
tuple absent in some language is an empty document at that index, so the model also
handles *partly comparable* corpora — a small set of aligned "glue" tuples is enough
to align topics across otherwise-separate per-language collections (paper §4.4).
Determinism is `seed-reproducible`.

**PolylingualLDA vs InfoCTM vs ZeroShotTM.** All three align topics across languages,
but they need different inputs and scale differently. `PolylingualLDA` needs
**document-aligned tuples** (which document corresponds to which) and no bilingual
resources at all — the alignment signal is the tuple structure — and it takes any
number of languages at once. `InfoCTM` needs a **bilingual dictionary** (not
document alignment) and handles two languages. `ZeroShotTM` needs a **multilingual
sentence embedder** and transfers zero-shot. Reach for `PolylingualLDA` when the
corpus is naturally paired or comparable across languages (parliamentary proceedings,
linked encyclopedia articles, multi-language editions) and you want plain count-based
topics with per-language word distributions.

Validated against MALLET's `cc.mallet.topics.PolylingualTopicModel` (the reference
from the paper's authors) in `parity/pltm_compare.py`; because MALLET is
weak-copyleft, the port is derived from the paper and MALLET is used only as a
black-box oracle. The RNG differs, so parity is measured by aligned per-language
topic-word cosine, not a bit-exact match.

## NMF

Non-negative matrix factorization ([Lee & Seung 2001](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization)) factors the document-term matrix `X` (D x V, non-negative) as `X ≈ W H` with both factors non-negative, then reads each row of `H` as a topic's word distribution and each row of `W` as a document's topic mixture (both normalized to sum to 1). It is the fast, deterministic baseline familiar from scikit-learn: no sampling and no priors, just multiplicative updates that descend a reconstruction loss.

```python
m = topica.NMF(num_topics=20, seed=1)
m.fit(docs)
theta = m.doc_topic
m.top_words(10)
```

`beta_loss` selects the divergence: `"frobenius"` (default, the squared error `½‖X − WH‖²`) or `"kullback-leibler"` (the generalized-KL loss, equivalent to pLSA on counts). `init` selects the start: `"nndsvd"` (default, a deterministic NNDSVDa initialization seeded by a from-scratch randomized truncated SVD) or `"random"` (seeded). `weighting` builds `X` from topica's TF-IDF (default, classic NMF recipe) or raw counts (`weighting="count"`). Reported `doc_topic` weights each topic by its $H$-row mass before row-normalizing ($W_{d,k} \cdot \text{rowsum}(H_k)$). The Rust core is BLAS-free: the dense products are rayon-parallel and the document-term products exploit `X`'s sparsity, so fits are bit-identical regardless of thread count.

Validated against `sklearn.decomposition.NMF` in `parity/nmf_vs_sklearn.py`. On a planted-block corpus topica matches sklearn to aligned topic-word cosine 1.000 for both divergences. On the political-blog corpus (poliblog5k, 5,000 documents) topica reproduces sklearn's topics at K=10 (aligned cosine 0.999, both divergences); at larger K, where the NMF objective is multimodal, topica reaches an equal-quality alternate optimum (reconstruction loss within about 0.1% of sklearn, sometimes lower) rather than sklearn's exact factorization, as expected for a non-convex problem whose solutions are not unique. On speed, the KL path runs several times faster than sklearn at scale, and the Frobenius path is competitive on the sparse document-term matrices typical of text, with the gap to BLAS-backed sklearn appearing only on near-dense inputs.

## GuidedNMF

Guided NMF ([Vendrow, Haddock, Rebrova & Needell 2021](https://arxiv.org/abs/2010.11365)) is the seed-word-guided cousin of NMF, and the matrix-factorization analogue of [SeededLDA](guided.md). When a corpus is biased toward a dominant theme, plain NMF often spends several components on it and returns redundant or off-theme topics. GuidedNMF lets you name a few seed words for each theme you expect and steers designated topics toward them. We factor the same non-negative document-term matrix `X ≈ A S` (`A` document-topic, `S` topic-word), but add a supervision term that ties the topics to a seed matrix `Y` (one row per seed group, marking that group's words):

$$\min_{A,S,B \ge 0}\; \lVert X - A S\rVert_F^2 \;+\; \lambda\,\lVert Y - B S\rVert_F^2,$$

where `B` (G x K) expresses each seed group as a non-negative combination of the learned topics. The guidance weight `λ` (`guidance=`, alias `lam=`) trades reconstruction against adherence to the seeds.

```python
seeds = {"economy": ["tax", "market", "jobs"], "health": ["disease", "clinic", "patient"]}
m = topica.GuidedNMF(num_topics=10, seed_words=seeds, seed=1)  # guidance defaults to 3.0
m.fit(docs)
m.seed_topic_indices     # which learned topic each seed group steered
dict(zip(m.seed_group_names, m.seed_topic_indices))   # group name -> topic index
m.top_words(10, topic=m.seed_topic_indices[0])   # bare words; add weights=True for (word, weight) pairs
```

A caveat before you report document prevalence: `guidance` (λ) controls how tightly
the guided topics adhere to their seed words. topica defaults to `guidance=3`, lower
than the reference's rarely-used `20`, because at `20` the guided topics are so narrow
they rarely dominate a document — their `doc_topic` share and `argmax` counts can be
near zero even when the theme is clearly present (the topic-*word* content stays
faithful either way). Raise `guidance` toward `20` to reproduce the reference; state
the λ you used when reporting prevalence. `convergence_tol` defaults to `0.0` (run the
full `iters` budget, as the reference does), so a normal fit ends with
`converged=False` — the "ran the whole budget" state, not a failure.

Seed groups are matched to the vocabulary with the same `seed_match` (`"fixed"`/`"glob"`/`"regex"`) and `case_insensitive` machinery as SeededLDA. The number of seed groups may not exceed `num_topics`. `init` defaults to `"random"` (seeded Uniform[0,1], matching the reference, so the guidance term shapes the topics from a blank slate); `"nndsvd"` is a deterministic SVD start, and `"none"` takes caller-supplied factors (`init_a`/`init_s`/`init_b`). `weighting` defaults to `"tfidf"`, the regime the reference `λ` is tuned for; on raw counts scale `λ` down, since the data term grows with the counts while the seed matrix does not. `convergence_tol` defaults to `0.0`, so a fit runs the full `iters` budget (the reference has no early stop); set it above zero to stop on the relative objective decrease. Because NMF is invariant to rescaling a topic's row of `S` against its columns of `A` and `B`, the reported `doc_topic` and `seed_topic_indices` are scale-corrected (weighted by `‖Sₖ‖₁`); the raw factors are available as `factor_a`, `factor_s`, and `factor_b`. The Rust core reuses NMF's BLAS-free rayon-parallel products, so fits are bit-identical regardless of thread count.

Validated against the `ssnmf` package (the engine the reference repository uses, pinned `ssnmf==0.0.2`, supervised Frobenius mode) in `parity/guidednmf_gold.py`. Fed the same document-term matrix and the same initial factors, topica reproduces ssnmf's multiplicative update to floating-point noise: after one update the raw factors agree to a maximum absolute difference of about 1e-15, and after the full 50-iteration run the aligned topic-word cosine is 1.000. On a planted corpus with a known seed structure, each seed group steers the topic that carries its planted block. Guidance is a semi-supervised extension of NMF, so this is faithful reproduction of the reference decomposition, not a probabilistic-recovery claim.

## KeyNMF

`KeyNMF` ([Kristensen-McLachlan et al. 2024](https://aclanthology.org/2024.nlperspectives-1.1/)) is the NMF family's bridge to the embedding backend. Where `NMF` factors a document-term *count* matrix, KeyNMF factors an **embedding-derived keyword-importance** matrix: for each document, every word it contains is scored by the similarity between the document's embedding and the word's embedding, and only the top-N most similar (positive) words are kept. NMF over that sparse doc-word matrix yields sparse, readable topics — the semantics come from the embedding space rather than raw frequency, which is what makes KeyNMF robust on short or noisy text. On real medium-length documents (e.g. 20 Newsgroups) KeyNMF clearly beats a raw-count NMF on coherence and diversity and roughly matches a well-tuned TF-IDF NMF; its advantage widens as the documents get shorter, where term counts alone carry little signal.

```python
# you bring the embeddings (e.g. from sentence-transformers) and the aligned vocabulary
m = topica.KeyNMF(num_topics=10, top_n=25, seed=13)
m.fit(docs, doc_embeddings, word_embeddings=word_embeddings, vocabulary=vocab)
m.top_words(10)          # per-topic top words
m.keywords(doc=0, n=10)  # the (word, importance) keywords extracted for a document
m.topic_word, m.doc_topic  # (D x K) / (K x V) arrays; doc_topic rows are L1-normalized
m.coherence()              # one u_mass score per topic
```

The **vocabulary you supply is the model's vocabulary** (aligned to `word_embeddings`); a document's candidate words are its tokens that appear in that vocabulary, so you control tokenization, stopwords, and `min_count` upstream with topica's `Corpus` preprocessing. Token order does not matter (a document is a set of words here). `num_topics` must be at most `min(num_docs, len(vocabulary))`, an NNDSVD-initialization requirement. The fit is deterministic.

!!! warning "`vocabulary[i]` must be the word embedded in `word_embeddings[i]`"
    The `i`-th row of `word_embeddings` is taken to be the embedding of `vocabulary[i]`.
    topica checks that the two have the same length, but it cannot check the *order*: if
    you build the vocabulary and the embedding matrix in two separate passes (a `set()`
    here, a `model.encode(...)` there), a row-order mismatch fits without error and returns
    clean-looking but meaningless topics. Build them together, and spot-check with
    `m.keywords(doc=0)` — those words should be recognizably about document 0. If a Corpus
    is passed to `fit`, its own vocabulary is not used for alignment; the `vocabulary=`
    argument is authoritative.

Two things differ from the reference `turftopic` implementation, both deliberate and documented. First, topica implements the keyword extraction **correctly**: turftopic's extractor has bugs — it zips the top-N words against a separately positive-filtered similarity list, which scrambles the word-to-importance mapping whenever a selected similarity is non-positive, and an off-by-one that silently drops a keyword (returning nothing for a one-word document). topica keeps each word paired with its own similarity. Second, the factorization uses topica's own NMF backend (NNDSVD init, Frobenius multiplicative updates) rather than turftopic's sklearn coordinate-descent solver; NMF is non-convex, so the two are compared on the reconstruction objective and aligned topics, not the specific decomposition. We validate at two levels (`parity/keynmf_compare.py`): topica's extracted keywords match a correct cosine/top-N/positive computation on every document, and on a MiniLM-embedded corpus topica's topics align with turftopic's at cosine ~0.98. Auto-`K` (BIC), dynamic, online, and cross-lingual matching from turftopic are not ported.

## LSA

Latent semantic analysis ([Deerwester et al. 1990](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9)), also called latent semantic indexing, takes a truncated SVD of the weighted document-term matrix `X` (D x V): `X ≈ U_k Σ_k V_kᵀ`. It is the original distributional-semantics method and the classic baseline behind scikit-learn's `TruncatedSVD`. There is no sampling and no prior, just a direct linear-algebra solve.

```python
m = topica.LSA(num_topics=20, weighting="tfidf", seed=1)
m.fit(docs)
m.singular_values        # the energy of each component
m.top_words(10)          # ranked by absolute loading
```

LSA is not a probabilistic topic model, and its outputs reflect that. `topic_word` (K x V) is the signed right singular vectors `V_k`: term *loadings*, not a word distribution, so the rows are not a simplex and a large negative loading is as defining of a component as a large positive one (`top_words` ranks by absolute value). `doc_topic` (D x K) is `U_k Σ_k`, the documents' coordinates in the reduced space; these are signed and the rows do not sum to 1, because LSA is not mixed-membership. `singular_values` (length K) gives each component's energy. Coherence and any diagnostic that assumes a non-negative φ operate on the absolute loadings and should be read with that caveat.

The SVD is unique only up to a per-component sign, so we fix the sign with the `svd_flip` convention scikit-learn uses: for each component we flip the `(u, v)` pair together so the largest-magnitude entry of the right singular vector is positive. That makes the fit deterministic and directly comparable to the reference. `weighting` builds `X` from topica's own TF-IDF (default, classic LSI) or from raw counts. The Rust core reuses NMF's BLAS-free randomized truncated SVD (rayon-parallel dense products, sparse document-term products), so fits are bit-identical regardless of thread count. The SVD is a direct solve, so there is no `iters` argument, `fit_history` is empty, and `converged` is `None`.

We validate against `sklearn.decomposition.TruncatedSVD` (`algorithm='randomized'`) in `parity/lsa_vs_sklearn.py`. On the same document-term matrix, after applying `svd_flip` on both sides, topica reproduces sklearn's solution exactly: per-component right-singular-vector cosine 1.000000, singular values agreeing to a maximum relative error of 1.5e-9, and document-coordinate correlation 1.000000. Because the truncated SVD is well-posed (a unique solution up to sign when the singular values are distinct), this is a match-the-solution result, not agreement within a noise band.

## CorEx

CorEx ([Correlation Explanation; Gallagher, Reing, Kale & Ver Steeg 2017](https://arxiv.org/abs/1611.10277); Ver Steeg & Galstyan [2014](https://arxiv.org/abs/1406.1222)/[2015](https://arxiv.org/abs/1410.7404)) is the odd one out in topica: it is neither generative nor a factorization. It learns `num_topics` **binary** latent topics that maximize the **total correlation** (the multivariate mutual information) they explain about the words. Each topic is on or off per document, and in tree mode the words are softly partitioned so each word informs one topic. It is fast, deterministic, and — through **anchor words** — semi-supervisable with minimal domain knowledge, which is why it is a staple in computational social science.

```python
m = topica.CorEx(num_topics=10, seed=1).fit(docs)
m.total_correlation        # bits/nats of dependence the topics explain
m.top_words(10, topic=0)   # words ranked by membership-weighted mutual information
# anchored (semi-supervised):
anchors = {"economy": ["tax", "market", "jobs"], "health": ["clinic", "patient"]}
ma = topica.CorEx(10, anchor_words=anchors, anchor_strength=2.0, seed=1).fit(docs)
ma.clusters                # each vocabulary word's topic; anchored words land in theirs
```

Its outputs reflect the different paradigm and should be read with that in mind. `topic_word` (K x V) is `alpha * mis`: the mutual information between each word and topic, weighted by the word's soft membership. It is non-negative but **not a probability distribution** (rows do not sum to 1); `top_words` ranks by it, and the raw `mis` and membership `alpha` are exposed separately. `doc_topic` (D x K) is `p(y_j = 1 | doc)`, the probability each topic is *on* for a document; because topics are independent binary factors rather than a mixture, **its rows do not sum to 1** (a document can turn on several topics or none). `labels` is the hard `p > 0.5` version, `clusters` is each word's argmax topic, and `topic_tc` / `total_correlation` report the correlation explained per topic and overall. `transform` labels held-out documents. Anchor words are matched to the vocabulary with the same `seed_match`/`case_insensitive` machinery as [SeededLDA](guided.md), and the number of anchor groups may not exceed `num_topics`. `count` is `"binarize"` (presence/absence; `"fraction"` is not yet implemented). The Rust core reuses NMF's BLAS-free sparse products, so fits are bit-identical regardless of thread count.

Validated against the `corextopic` package (the reference implementation, Apache-2.0) in `parity/corex_gold.py`. On a planted binary corpus with a known block structure, topica reproduces corextopic to the reference's own seed-to-seed floor: aligned topic-word mutual-information cosine 1.000, total correlation matching to three decimals (5.272 vs 5.272), and identical word clusters; the anchored fit places each anchor group's words in their assigned topic. Because CorEx is non-convex and its topics are not identified up to a fixed labelling, this is topic-aligned agreement plus objective (total-correlation) parity, not a bit-for-bit reproduction of the reference's specific factors.

## AuthorTopic

The Author-Topic Model ([Rosen-Zvi, Griffiths, Steyvers & Smyth 2004](https://arxiv.org/abs/1207.4169)) conditions topics on **authors** instead of documents. Each author has a topic distribution `θ_a`; a document's tokens are generated by drawing one of the document's authors uniformly, then a topic from that author's distribution, then a word. This handles multi-author documents and answers two questions the document-level models cannot: what does a given author write about, and which authors are close in topic space. LDA is the special case where every document has exactly one unique author.

```python
docs = [["deep", "learning", "networks"], ["survey", "methods", "sampling"], ...]
authors = [["Bengio", "LeCun"], ["Gelman"], ...]   # one author list per document
m = topica.AuthorTopic(num_topics=20, seed=13).fit(docs, authors)
m.author_topic          # (num_authors, K) each author's topic distribution
m.authors               # author names, indexing the rows above
m.author_doc_counts     # (num_authors,) documents behind each author's row
m.top_authors(3, n=10)  # the authors most associated with topic 3
m.doc_topic             # (num_docs, K) per-document empirical topic proportions
```

Inference is collapsed Gibbs from the paper: the per-token pair *(author, topic)* is resampled jointly, reusing topica's LDA count machinery for the word-topic side and adding an author-topic count table. `alpha` is the symmetric author-topic Dirichlet (default `50/K`, the paper's value); `beta` is the topic-word Dirichlet (default `0.01`). `author_topic` is the model-defining output — the smoothed per-author topic distribution. `doc_topic` is the document's *empirical* topic mix (the proportions of its sampled token assignments in the terminal draw, content-based exactly as in LDA), not the prior average of its authors' distributions; it reflects the words a document actually contains. This is a single-chain, terminal-state estimator (as with topica's other collapsed-Gibbs models); multi-chain averaging and held-out perplexity are out of scope. Fits are deterministic from a fixed `seed`. ATM is **single-threaded** (no `num_threads`): each token samples over an `|authors| x K` grid, so unlike LDA/LabeledLDA/DMR it does not take the AD-LDA parallel path. It is still fast — a 700-document corpus fits in a few seconds — and about as fast as its Gibbs siblings single-threaded, with the lowest memory of the family.

**Reading author profiles honestly.** An author's `author_topic` row is only as reliable as the number of documents behind it, which is why `author_doc_counts` sits next to it. A *prolific* author's row is pulled toward uniform by the `alpha` prior (lots of tokens, but spread across topics), so it can look uninformative; a *one-document* author's row is near-deterministic in that single article and unstable across seeds — and because those rows are the peakiest, they dominate `top_authors`. Report `author_doc_counts` alongside any author-topic table, and check seed stability (refit at a second `seed` and compare the permutation-invariant author-author similarity structure) before publishing an "authors are similar" or "author X writes about Y" claim. Lowering `alpha` sharpens prolific authors' rows at the cost of the rare ones. If your author field is a delimited string (e.g. `"Smith; Jones"`), split it into a real list — `authors = [s.split("; ") for s in raw]` — so a co-authored document credits both authors; a single composite string becomes a phantom third author and defeats the point of the model. Normalize name/initial variants for the same reason.

**Any grouping variable can play the role of "author."** The `authors` argument is just a per-document set of group labels, so a well-populated categorical — year, decade, outlet, party — often makes a better ATM than a sparse author field, and turns `author_topic` into a per-group topic profile. Using decade as the group on The Crisis corpus (`topica.datasets.load_dubois()`) gives a lightweight over-time view: the 1910s rows load on lynching and the anti-lynching campaign, the 1920s on the Harlem Renaissance, the 1930s on Depression-era labor and economics — each row backed by dozens–hundreds of documents (`author_doc_counts`), so it is stable in a way a one-article author is not.

When every document has exactly **one** author/group, ATM is essentially equivalent to concatenating each group's documents into a single document and running LDA (the per-author counts `C^AT_{a,·}` are exactly that group's pooled token→topic counts) — convenient, but not a distinct model. Its non-reducible advantage is **multi-author documents**: ATM treats the author of each *token* as latent and infers the attribution from the words, so a co-authored document informs every co-author's profile without double-counting and while keeping per-document `doc_topic`. Reach for ATM (over concatenate-then-LDA) when documents genuinely share authors/groups.

```python
df = topica.datasets.load_dubois().drop_duplicates("text")
corpus = topica.from_dataframe(df, text_col="text", stopwords=topica.data.ENGLISH_STOPWORDS)
decade = [[f"{int(y) // 10 * 10}s"] for y in df.year]      # grouping variable as "author"
m = topica.AuthorTopic(12, seed=13).fit(corpus, decade, iters=500)
dict(zip(m.authors, m.author_doc_counts))                 # docs behind each decade
m.top_words(6, topic=int(m.author_topic[i].argmax()))  # a decade's theme
```

Validated against gensim's `AuthorTopicModel` (the reference implementation; gensim is LGPL, so topica implements the paper's collapsed Gibbs and uses gensim only as a black-box oracle) in `parity/author_topic_gold.py`. On a synthetic corpus with a planted author→topic structure and overlapping topics, topica matches gensim about as closely as two gensim runs with different seeds match each other: aligned topic-word cosine 0.99 (one gensim seed pair: 0.999) and aligned author-topic correlation 0.997 (seed pair: 1.000), with each author's dominant topic recovered cleanly and distinctly. Because ATM is stochastic and gensim's inference is variational while topica's is Gibbs, this is topic-aligned agreement near the reference's seed-to-seed noise, not a bit-for-bit match. On fit speed, topica is faster than gensim at matched full-corpus sweeps (roughly 2x on a small corpus, ~7x on a larger one; the exact multiplier depends on how gensim's online passes are configured).

## AuthorRecipientTopic

The Author-Recipient-Topic model ([McCallum, Wang & Corrada-Emmanuel 2007](https://www.jair.org/index.php/jair/article/view/10476)) conditions topics on the **directed (sender, recipient) pair**, not on a single author. For each token a recipient is drawn uniformly from the message's recipient set, a topic from the pair-specific distribution `θ_{sender, recipient}`, and a word from `φ`. It captures the language of a *directed* social network — who talks to whom about what — which the author-only model cannot see. The canonical use is email or reply data; on Reddit-style threads the sender is a comment's author and the single recipient is the parent comment's author.

```python
docs       = [["you", "should", "lift"], ["thanks", "that", "helped"], ...]
authors    = ["userA", "userB", ...]          # one sender per message
recipients = [["userB"], ["userA"], ...]      # recipient list per message (>= 1)
m = topica.AuthorRecipientTopic(num_topics=20, seed=13).fit(
        docs, authors=authors, recipients=recipients)
m.pair_topic    # (num_pairs, K) each (sender, recipient) pair's topic distribution
m.pair_labels   # the (sender, recipient) tuples indexing those rows
m.pair_counts   # messages behind each pair's row (its reliability)
m.topic_word    # (K, V)   m.doc_topic  # (num_docs, K) content-based, as in LDA
```

**Implementation.** ART is mathematically isomorphic to `AuthorTopic` with the "author" taken to be the ordered `(sender, recipient)` pair: because the sender is fixed within a message, uniform sampling over recipients is identical to uniform sampling over that message's pairs. topica therefore realizes ART as a faithful wrapper over the validated AuthorTopic collapsed-Gibbs engine (mapping recipients to pair labels and delegating the fit), which inherits its determinism and — because AuthorTopic's held-in likelihood already carries the `1/|recipients|` factor — the correct recipient-averaged likelihood. `alpha` and `beta` default to the paper's values, `50/K` and `0.1`. Validation is the isomorphism identity (ART's token→pair partition matches an independently-labeled AuthorTopic fit, so `topic_word`/`doc_topic` agree bit-for-bit), planted recovery with unequal recipient-set sizes, the unique-pair-per-document reduction to LDA, and label-collision / set-determinism edge cases, in `tests/test_art.py`.

**Reading pair profiles honestly.** As with AuthorTopic, a pair's `pair_topic` row is only as trustworthy as its `pair_counts`; per-individual pairs are often ultra-sparse (most dyads exchange one message), so report the counts and prefer stable, well-populated pairs. A subtle trap: under the default `50/K` smoothing a *sparse* pair's few token assignments concentrate, so its row looks **spikier** — more confident — than a well-populated pair's flatter row. Peakedness is not confidence; threshold on `pair_counts` before reading any pair's `argmax` as its "dominant topic." Note also that `pair_topic` is the model-defining output: the general `inspect.topic_crosstab` helper reads `doc_topic` (content-based, LDA-style) and so ignores the directed-pair structure — build your reporting table from `pair_topic` / `pair_labels` / `pair_counts` to keep the dyad. **Any dyadic grouping can play sender/recipient.** Passing observed group labels — a poster's home community as sender, the location subreddit as recipient — turns `pair_topic` into a per-group discourse profile and lets ART model how the same population talks differently across spaces. This is a coarsened, observed-group application of ART, not latent-role discovery (the RART role extension is future work). Labels must be serializable (strings or ints), and every message needs at least one recipient (an empty recipient set raises). Fit cost tracks AuthorTopic (roughly on par; the pair-preprocessing pass is negligible), so it inherits the author-family sampler's speed — slower than plain LDA, since each token samples over a `|pairs-in-doc| × K` grid. The `pair_topic` table is `P × K`, where `P` is the number of distinct pairs; with many individual dyads `P` can far exceed the number of senders, so on very large per-individual corpora prefer coarser group labels (which also make each pair's row better populated).

## MGLDA

Multi-Grain LDA ([Titov & McDonald 2008](https://arxiv.org/abs/0801.1063)) is the aspect model for reviews and opinion text. It learns two grains of topic at once: **global** topics that describe a document's overall subject (which product, which brand), and **local** topics that describe the rateable aspects discussed sentence-by-sentence (battery, screen, price). Each token first picks one of the sliding windows covering its sentence, then a global-vs-local grain, then a topic from the appropriate distribution. This separates "what is this document about" from "what aspects does it discuss," which single-grain models (LDA, GSDMM, BTM) conflate.

Input is **sentence-segmented** — `list[list[list[str]]]` (document → sentences → tokens) — because the local grain is defined over sliding *sentence* windows and a bag-of-words `Corpus` cannot carry sentence boundaries.

```python
docs = [
    [["great", "phone", "apple"], ["battery", "dies", "fast"], ["screen", "is", "bright"]],
    [["cheap", "laptop"], ["keyboard", "feels", "mushy"], ["good", "value", "price"]],
]
m = topica.MGLDA(num_global_topics=5, num_local_topics=10, window=3, seed=13).fit(docs)
m.global_topic_word    # (5, V)  document-level themes
m.local_topic_word     # (10, V) rateable aspects
m.global_fraction      # share of tokens the model routed to the global grain
m.top_words(10, topic=0)          # global topic 0 (topics 0..4 global, 5..14 local)
```

**Preparing sentence-segmented input from raw text.** `Corpus` cannot carry sentence boundaries, so build the nested lists yourself: split each document into sentences, then tokenize (lowercase, drop stopwords) each sentence. MG-LDA does its own vocabulary building from the tokens you pass; prune rare words yourself if you want a smaller vocabulary.

```python
import re
raw = topica.datasets.load_dubois()["text"]
stop = topica.data.ENGLISH_STOPWORDS
def to_sentences(text):
    sents = [re.findall(r"[a-z]+", s.lower()) for s in re.split(r"[.!?]", text)]
    sents = [[w for w in s if w not in stop and len(w) > 2] for s in sents]
    return [s for s in sents if s]                       # drop sentences emptied by pruning
docs = [to_sentences(t) for t in raw]
m = topica.MGLDA(8, 12, window=3, seed=13).fit(docs)     # rows align 1:1 with `raw`
```

Inference is collapsed Gibbs over the `(window, grain, topic)` triple, with the sliding-window (`S+T−1` windows for `S` sentences), the per-window grain switch, and the per-sentence window selection all sampled jointly. Hyperparameters default to the reference (tomotopy) values: `window=3`, `alpha_global`/`alpha_local`/`alpha_mix_global`/`alpha_mix_local` = 0.1, `beta_global`/`beta_local` = 0.01, `gamma` = 0.1. `topic_word` stacks the two grains (global rows first, then local); `doc_topic` is the per-document empirical prevalence over the combined `[global | local]` set (rows sum to 1) — a content-based prevalence, not a single generative θ, since global topics are document-level and local topics window-level; `global_doc_topic` is the true doc-level global distribution. Document rows (`doc_topic`, `global_doc_topic`) align 1:1 with your input list, including empty documents. Fits are deterministic from a fixed `seed` (single-threaded); `converged` is always `False` — collapsed Gibbs runs the full `iters` budget with no early-stop test, so watch `fit_history` (an `(iter, held-in log-likelihood)` trace) flatten to judge mixing.

Two honest caveats. First, MG-LDA is often **global-dominant**: on text without strong within-document aspect locality (the same aspect recurring in adjacent sentences), the grain switch routes almost every token global, `global_fraction` approaches 1.0, and `local_topic_word` is then **prior-dominated — its "aspects" are noise, not signal**. `fit` emits a warning when `global_fraction > 0.9`; as a rule of thumb, treat local topics as unidentified above ~0.9 and report only the global grain. The local grain earns its keep on genuine review/opinion text. Second, a **paper-vs-reference discrepancy**: the paper writes the local-topic Gibbs numerator as `(n^loc + α_loc)`, but the canonical implementation (tomotopy) omits the `+α_loc` (raw window-local count), and topica follows the reference — with the rest of the conditional provably identical to tomotopy, the smoothed `+α_loc` variant does not track the reference and stalls the grain switch, so the reference's unsmoothed numerator is what actually separates the two grains. (The `fit_history` log-likelihood diagnostic does use the proper smoothed form so it stays a valid probability.)

Validated against `tomotopy`'s `MGLDAModel` (the reference implementation, MIT; tomotopy 0.13.0, since 0.14.0 has a bug that ignores `k_g`/`k_l`) in `parity/mglda_gold.py`. On a planted corpus with global product themes and local aspects, topica reproduces tomotopy's **global** topics to the reference's own seed-to-seed floor (aligned global topic-word cosine 1.000 on the planted fixture; global themes recovered exactly) **and its grain dynamics** (topica `global_fraction` 0.997 vs tomotopy 0.997 — both route the synthetic corpus almost entirely global). The 1.000 is a planted-fixture figure with disjoint theme vocabularies; on real prose, free-run global topics from two different collapsed-Gibbs RNGs (topica vs tomotopy, and topica vs LDA) align at ~0.6 best-match cosine — related and coherent, not identical, as expected for a stochastic model. The **local** grain is *not* a fidelity target on synthetic data: it collapses to the prior for the reference itself here (tomotopy `global_fraction` ~1.0, its own seed-to-seed local cosine only ~0.89), so there is no local signal to match; local topics become identifiable only on real text with aspect locality. On speed, topica's per-token `T×(K_gl+K_loc)` grid makes it grow with sentences-per-document: roughly **1.5–1.6× slower than tomotopy single-threaded** on realistic long-document corpora (faster than GSDMM, slower than plain LDA by construction).

## TopicsOverTime

Topics over Time ([Wang & McCallum 2006](https://dl.acm.org/doi/10.1145/1150402.1150450)) is LDA with a sense of *when*. Each topic keeps a word distribution, as in LDA, and additionally a **Beta density over continuous time**: the years (or any numeric timestamp) at which the topic is prevalent. The timestamp is not just a covariate read off after fitting — it shapes the fit. A token's topic is drawn jointly from the words around it and the document's date, so a topic is pulled to be coherent in time as well as in vocabulary, and each topic ends up with a peak year and a rise-and-fall profile.

This is **descriptive continuous-time prevalence**, and that is what distinguishes it from topica's other temporal models. DTM/DETM slice time into discrete bins and let each topic's *vocabulary drift* across the bins; STM with a spline on a `year` covariate models the *effect* of time on prevalence with uncertainty. ToT instead asks a simpler, complementary question — "when did this topic peak?" — with a fixed vocabulary and a smooth continuous density, no binning and no covariate design.

```python
d = topica.datasets.load_dubois()
docs = [t.lower().split() for t in d["text"]]     # tokenize however you like
years = d["year"]                                 # one numeric timestamp per document

m = topica.TopicsOverTime(num_topics=10, seed=13).fit(docs, times=years)
m.topic_time_peak       # (K,) peak year per topic, in your input units
m.topic_time_mean       # (K,) mean year per topic
m.topic_time            # (K, 2) raw Beta (psi1, psi2) over normalized [0,1] time
m.time_range            # (min_year, max_year) the peaks/means are reported on
m.top_words(10, topic=int(m.topic_time_peak.argmax()))   # words of the latest-peaking topic
```

`times` is the canonical argument; `timestamps=` is an accepted alias. Any numeric scale works (year, decade, ordinal date, unix time) — timestamps are min-max normalized to `[0,1]` internally, and peaks/means are mapped back to your original units for reporting (`time_range` records the range). Inference is collapsed Gibbs: the LDA conditional multiplied by each topic's Beta time-likelihood of the document's date. The per-topic Beta parameters are re-estimated by method of moments from the timestamps of each topic's assigned tokens, once per sweep. This is the canonical **unit-weight** ToT conditional; the paper also describes an optional exponent that rebalances the word and time modalities, which topica does not implement. Hyperparameters follow the paper: `alpha` defaults to `50/K`, `beta` to `0.1`. Fits are deterministic from a fixed `seed` (single-threaded); `converged` is always `False` (Gibbs runs the full `iters` budget — watch `fit_history`, a list of `(iteration, log-likelihood)` pairs, flatten). ToT shares LDA's word model, so **choose `K` with `search_k(..., model="lda")`** — there is no ToT-specific selector.

Read the temporal outputs with two cautions. **Use `topic_time_peak`, not the mean, for "when did this topic peak"**: the Beta is often skewed, so its mode (peak) and mean differ, and the peak is the interpretable one. The peak is the Beta *mode* when the density is single-humped (both `psi > 1`); for a topic that only rises or only falls across the whole range it is reported at the boundary (earliest/latest date), and for a U-shaped or flat/uniform topic — which has *no* single peak — it is `NaN` by design rather than a misleading midpoint. A topic with no usable temporal signal (its token dates too dispersed to fit a concentrated Beta, or a corpus where every document shares one timestamp, which `fit` warns about) falls back to a uniform Beta (`topic_time` row `[1, 1]`, `NaN` peak, mean at the range midpoint) and the model reduces to LDA on that topic. **When a topic's `topic_time_peak` is `NaN`, its `topic_time_mean` is not a peak** — do not report it as when the topic peaked; inspect `topic_time` and the topic's document dates. Second, ToT weights the time factor once per token, so a very long document contributes its date many times; that is faithful to the paper but means a few long documents can dominate a topic's estimated timeline.

Validated by **planted continuous-time recovery** plus **independent numerical checks** in `parity/tot_gold.py`: there is no maintained reference library (the one public port is GPL and unmaintained), so on a fixed-seed corpus whose topics have disjoint word blocks *and* known early/middle/late Beta time densities, topica recovers both the word blocks (topic-word cosine 0.998) and their temporal ordering exactly (recovered peak years correlate 0.995 with the planted centers); the fitted Beta parameters are recomputed from scratch by a numpy method-of-moments fit (agreement within 5%) and cross-checked against `scipy.stats.beta` (each `psi` is a proper density, the analytic mean matches `topic_time_mean`, and the analytic mode matches `topic_time_peak`). That the Beta time factor *causally* drives assignment — not just the words — is shown by a separate discriminating test (ambiguous vocabulary plus a time-blind control fit): the time factor lifts era recovery far above the time-blind baseline. On speed, ToT is a classic dense collapsed-Gibbs sampler plus a per-document Beta factor, so it runs at roughly **1.3–3× topica's (sparse) LDA**, the ratio growing with `K` because the baseline sampler is sublinear in `K` while ToT scans all topics per token.

## GaussianLDA

Gaussian LDA ([Das, Zaheer & Dyer 2015](https://aclanthology.org/P15-1077/)) moves LDA off the vocabulary and into the embedding space. A topic is no longer a categorical distribution over words but a **Gaussian over the word-embedding space**: each topic has a mean and a covariance, and a token is generated by drawing its word's embedding from its topic's Gaussian. Because topics live in the continuous space, they generalize over semantically similar words: a topic centered near *budget*, *deficit*, and *appropriations* also assigns high density to a related word it never co-occurred with, and to words unseen at training time. You bring the word embeddings, as with ETM; topica fits the per-topic Gaussians.

```python
vocab = [...]                                   # your vocabulary
rho = topica.embeddings.llm_embed(vocab, model=...)        # (len(vocab), E) word embeddings
rho = (rho - rho.mean(0)) / rho.std(0)          # standardize: avoids mode collapse on
                                                # dense contextual embeddings (see below)
m = topica.GaussianLDA(num_topics=10, seed=13).fit(docs, rho, vocab)
assert m.n_effective_topics == m.num_topics     # guard against a collapsed fit
m.topic_means           # (K, E) per-topic Gaussian means
m.topic_covariances     # (K, E, E) posterior covariance Psi_k / (nu_k - E - 1)
m.top_words(10, topic=0)
```

Inference is collapsed Gibbs with a **Normal-Inverse-Wishart** conjugate prior on each topic's `(mean, covariance)`. Integrating the Gaussian parameters out leaves a **multivariate Student-t** posterior predictive: a token's topic is drawn from the LDA document term times each topic's Student-t density at the word's embedding. The cost center is that as a token joins or leaves a topic, the topic's covariance changes by a rank-one update, and topica maintains the Cholesky factor of each topic's scale matrix with rank-1 Cholesky up- and downdates rather than refactoring from scratch. Because topics are densities, `topic_word` (the K x V matrix the rest of topica expects) is **derived**, not sampled: we score every vocabulary word's embedding under each topic's Student-t and softmax over the vocabulary. `topic_means` and `topic_covariances` are the native outputs; `topic_scale_matrices` exposes the raw NIW scale matrix `Psi_k` (what the reference writes), and `topic_covariances` is the posterior-mean covariance `Psi_k / (nu_k - E - 1)`.

The NIW prior follows the reference defaults, which the paper leaves unstated: `kappa` (mean concentration) `0.1`, `nu` (degrees of freedom) the embedding dimension `E` and always clamped to at least `E`, `psi_scale` setting the prior scale matrix to `psi_scale * E * I` with `psi_scale = 3.0`, and `alpha = 1/K` on the document-topic Dirichlet. The prior mean is the mean of your vocabulary embeddings. `log_likelihood_history` reproduces the reference's `avgLL` diagnostic exactly (the mean per-token Gaussian log-density at the current assignment, with covariance `Psi_k / (nu_k - E)`); note that this is a diagnostic, not the model evidence, and it is not guaranteed to increase monotonically. Fits are deterministic from a fixed `seed` (single-threaded). `transform` infers topic proportions for new documents over the fitted vocabulary, holding the topic Gaussians fixed.

**Mind the embedding type — this is the model's sharpest edge.** Gaussian LDA is prone to **mode collapse**: one topic absorbs every token and the rest come back empty. Whether it collapses depends mostly on the embeddings, not the initialization. On **low-dimensional, well-separated** word vectors — the regime the paper targets (word2vec / GloVe, ~50–300d) — it recovers clusters cleanly. On **dense, anisotropic contextual embeddings** (sentence-transformer / BERT / MiniLM vectors) it collapses to a single topic, and so does the original Java reference (both give the same one-topic result on the bundled `load_ng20_minilm` vectors). The cause is anisotropy: a few embedding dimensions carry most of the variance, so one broad Gaussian explains the whole corpus.

There are two practical fixes, and topica helps you catch the failure:

- **Standardize the embeddings per dimension before fitting** — `rho = (rho - rho.mean(0)) / rho.std(0)`. This removes the anisotropy that drives the collapse; on the same MiniLM vectors it turns one degenerate topic into distinct, coherent topics (family/health, computing/graphics, sports). Reducing `E` (e.g. PCA to ~50) also helps speed but does **not** by itself cure the collapse on contextual embeddings — standardize.
- **`fit` warns on collapse** and sets `converged=False`; read `n_effective_topics` (non-empty topics) and `topic_counts` to check. The empty topics' `top_words`/`coherence` are prior-only duplicates, so a degenerate fit is easy to mistake for K real topics if you don't look — the warning is there so you do.

Initialization is the secondary knob. topica **defaults to `init="kmeans"`** (seeded k-means over the vocabulary embeddings), the paper's approach, which prevents collapse on separable data where per-token random init is seed-fragile. Pass `init="random"` to reproduce the reference sampler exactly. On dense contextual embeddings neither init prevents collapse — standardize first.

We validate in `parity/gaussian_lda_gold.py` against the authors' Apache-2.0 Java reference. The reference initializes from an unseeded RNG, so it is not reproducible run to run and the target is topic-aligned agreement, not bit-exactness: on a fixed-seed planted-cluster corpus, topica's per-topic means (fit with `init="random"` for a like-for-like comparison) align to the reference's at cosine 0.94, above the reference's own two-run floor. The core equation is pinned two ways that do not depend on the reference's RNG: a Rust unit test asserts the incrementally maintained per-topic mean and Cholesky factor equal the batch Normal-Inverse-Wishart sufficient statistics after a sequence of adds and removes (including the rank-1 downdate and its positive-definite-failure rebuild path), and the Student-t density matches a direct numpy computation to ~1e-14 (odd and even `E`). On the k-means default, topica recovers planted cluster centers at cosine 1.0. Gaussian LDA is a dense Gibbs sampler with a per-topic `O(K·E^2)` density evaluation, so cost grows with the embedding dimension (a 384-d fit is minutes where a 50-d fit is seconds — prefer `E ≲ 100`). `topic_means` and `topic_covariances` are the native Gaussian outputs (`topic_covariances` is NaN for an empty or singleton topic, where the Inverse-Wishart posterior-mean covariance is undefined; `topic_scale_matrices` gives the always-defined Ψ_k). It is the right tool for topics that reason in a separable, low-dimensional embedding space, and ETM or the count-based models otherwise.

## SemanticSignalSeparation

Semantic Signal Separation ([S³; Kardos, Kostkan, Enevoldsen, Vermillet, Nielbo & Rocca, ACL 2025](https://aclanthology.org/2025.acl-long.32/)) treats topics as *independent axes* of semantic space rather than distributions over a bag of words. It runs FastICA on the document embeddings; each recovered independent component is a topic axis. A word's importance to a topic comes from projecting the vocabulary embeddings onto that axis, so the topics are described in words without any bag-of-words modelling. The reference is the flagship model of [turftopic](https://github.com/x-tabdeveloping/turftopic), and the authors report it is the fastest contextual topic model.

Like topica's other embedding-native models, you bring the embeddings: a `doc_embeddings` matrix (one row per document) and a `vocab_embeddings` matrix (one row per vocabulary term) in the same space. `topica.embeddings.llm_embed(texts, model=...)` can build them.

```python
m = topica.SemanticSignalSeparation(num_topics=10, seed=1)
m.fit(docs, doc_embeddings, vocab_embeddings, vocabulary=vocab)
m.top_words(10, topic=0)                     # positive pole of axis 0
m.top_words(10, topic=0, pole="negative")    # negative pole of axis 0
m.components                                 # signed per-word importance (K x V)
m.source_scores                              # signed document loadings (D x K)
```

ICA axes are **signed**: each topic has a positive and a negative pole, and both are meaningful (an axis might run from, say, sports vocabulary at one pole to finance vocabulary at the other). `top_words` returns the positive pole by default; pass `pole="negative"` for the opposite end. The signed native outputs are `components` (K x V, the per-word importance under `feature_importance`) and `source_scores` (D x K, the raw ICA document loadings); `axial_components` exposes the unweighted projection (turftopic's `axial_components_`). For the shared analysis surface, `topic_word` (φ) and `doc_topic` (θ) are the nonnegative, row-normalized positive poles of those two, so coherence, `topic_table`, and the effect estimators work unchanged.

`feature_importance` chooses how per-word importance is scored, matching turftopic: `"combined"` (default, `axial² × angular`, sharp and signed), `"axial"` (the raw projection), or `"angular"` (cosine of each word's axial vector to the axis). FastICA follows scikit-learn's defaults (logcosh, parallel, unit-variance whitening, `iters=200`, `convergence_tol=1e-4`); the whitening uses a seeded randomized range finder for the top-K components (fast, no external linear-algebra dependency), so a fixed `seed` reproduces the fit bit-for-bit. ICA recovers each axis only up to sign, so we orient each axis to put its heavier word mass on the positive pole; a corpus term you supply no embedding for gets exactly zero importance rather than a spurious score from a placeholder vector.

We validate against the reference algorithm (scikit-learn's `FastICA` with turftopic's projection) in `parity/s3_compare.py`. On planted independent-source embeddings, after Hungarian-aligning the signed component matrices, topica reproduces the reference's axes at mean absolute cosine 1.000, inside the reference's own seed-to-seed spread. Because the sources are well-separated the ICA solution is effectively unique up to sign and permutation, so this is a match-the-solution result.

### Reading S³ results

S³ optimizes a different notion of a topic than a bag-of-words model, so the coherence metric you choose decides which model looks better. On a 20 Newsgroups sample (MiniLM embeddings, top-10 words, K from 10 to 50) we compared S³ against topica's `LDA` and `NMF`:

- On **embedding coherence** (mean pairwise cosine of the top words' embeddings) and **topic diversity** (the share of distinct words across topics), S³ leads at every K. Its diversity holds near 0.97 as K grows, while `LDA` and `NMF` fall toward 0.72 as their high-K topics begin repeating words. This is the strength the authors report.
- On **UMass coherence** (whether the top words co-occur in the corpus), `LDA` and `NMF` lead by a wide margin. S³'s top words are close in semantic space but co-occur in documents less often.

The two results are consistent. S³ groups words by proximity in embedding space, so a topic like "rifle, firearm, weapon, ammunition" reads as tight and distinct even when those words rarely share a document, whereas a bag-of-words model forms a topic only from words that co-occur. Judge S³ by whether its axes are semantically interpretable and distinct, and reach for a count-based model when you want topics whose words actually appear together.

One caveat about stability. FastICA does not reach its convergence tolerance on general contextual embeddings like these within the default 200 iterations (`converged` is `False` on the 20 Newsgroups MiniLM sample), so the axes are only loosely identified. A hard document assignment, the argmax axis per document, is then about 0.33 correlated across seeds. The turftopic reference behaves the same way (about 0.32), so this reflects ICA on contextual embeddings rather than the port. Fix a `seed` for reproducibility, read topics from the top words of each axis rather than from hard document labels, and treat a `converged` of `False` as a signal that the axes are not sharply separated in this embedding space.

## AnchorLDA

The anchor-words algorithm ([Arora et al. 2013](https://proceedings.mlr.press/v28/arora13.html)) recovers topics without Gibbs sampling or EM. It rests on a *separability* assumption: each topic has an anchor word that occurs (almost) only in that topic. Given the anchors, every other word's topic distribution is fixed by how it co-occurs with them, so the whole topic-word matrix follows from one convex solve per word. The result is deterministic, fast, and gives each topic a single human-readable anchor word.

`AnchorLDA` is validated against the [`anchor-topic`](https://pypi.org/project/anchor-topic/) RecoverL2 reference: the co-occurrence matrix agrees to numerical precision, and given the same anchors topica's `recover="l2"` recovery matches the reference to cosine ~1.0 on both a planted corpus and real survey text (`parity/anchor_compare.py`). The two libraries use different (both valid) greedy anchor selectors, so end-to-end agreement is corpus-dependent; the untempered configuration matches the full reference pipeline on a planted separable fixture. The default `frequency_temper=0.5` tempers frequent-word dominance for more distinctive topics — a deliberate, corpus-dependent departure from the reference-exact inversion, not reference parity.

```python
m = topica.embeddings.AnchorLDA(num_topics=20, min_count=5, seed=0)
m.fit(docs)
m.anchors                # the anchor word identifying each topic
m.top_words(10)          # the recovered topic-word distributions
```

The pipeline is the standard one (Arora et al. 2013): form the word-word co-occurrence matrix `Q` from the unbiased per-document estimator `(h hᵀ − diag(h)) / (n(n−1))` and row-normalize it to `p(w₂ | w₁)`; select one anchor per topic by greedy farthest-point search on the rows of `Q` (the near-extreme points of the word simplex), restricted to words above a document-frequency floor (`anchor_min_doc_freq`, default 1% of documents) so the search skips rare, noisy rows; recover `p(topic | word)` for every word against the anchor rows; and Bayes-invert with the word frequencies to the topic-word matrix `p(word | topic)`. `doc_topic` is `p(topic | document)` from the per-word topic responsibilities.

Two recovery solvers are available through `recover`. The default `"kl"` minimizes `KL(Q_i ‖ p(topic | word_i) · Q_anchors)` with a vectorized exponentiated-gradient solver: a handful of matrix multiplies over the whole vocabulary at once, so `iters` is the maximum number of steps (default 200, with an early stop on `tol`) and `fit_history` / `converged` report the objective trace. `recover="l2"` is Arora et al.'s RecoverL2, a simplex-constrained non-negative least squares solved once per word — exact but a serial loop, so it is non-iterative (`fit_history` empty, `converged` `None`). On poliblog (K=15) `"kl"` fits in about a third of `"l2"`'s time at the same coherence and a closer co-occurrence fit; the two give effectively the same topics.

Anchor-words trades a little coherence for determinism and speed. On poliblog (K=15) it reaches about 93% of fully-converged LDA's c_v in roughly a fifth of the time, and it beats short-run LDA on both quality and time; on separable synthetic data it recovers the planted topics almost exactly.

One quirk needs handling: the exact Bayes inversion sets `beta ∝ p(topic | word) · p(word)`, so `beta` is weighted by raw word frequency — more than a Gibbs model's `beta`, where sparse priors push frequent words down. Left alone, pervasive words ("will", "one", "people") dominate many topics, which reads as redundant topics. Two defaults handle this and compose:

- `frequency_temper` (default `0.5`) tempers the inversion to `beta ∝ p(topic | word) · p(word)**γ`, dividing the excess frequency back out of the topic-word matrix itself. On poliblog at K=50 this lifts probability-ranked top-word diversity from ~0.33 (`γ=1`, the exact inversion) to ~0.79, and raises c_v. Set `frequency_temper=1` for the textbook Arora et al. estimate.
- `top_words` then defaults to a **FREX** ranking (`method="frex"`, the frequency/exclusivity balance topica's `frex`/`label_topics` use; `"lift"` and `"prob"` are also available), refining the display further — top-word diversity ~0.88 and c_v ~0.49 at K=50 with both defaults.

The underlying topics were always there; these two knobs keep frequent words from masking them. `AnchorLDA` is a strong fast first pass and a deterministic baseline; for a final model on real text, compare against `LDA` or `STM`.

## IdealPointTM

!!! warning "Experimental"
    `IdealPointTM` is an original construction with no reference implementation to
    validate against. It is gated: call `topica.enable_experimental()` (or set
    `TOPICA_EXPERIMENTAL=1`) before constructing one. Experimental models may
    change or be removed without a deprecation cycle.

A topic model tells you what is talked about. It does not tell you where a speaker stands. The political scientist who wants both runs two models, a topic model for description and a separate ideal-point model for position, and the two never reconcile. `IdealPointTM` estimates both in one fit. It is [`ETM`](embedding.md) with a latent trait per author: as in ETM each topic `k` is a point `alpha_k` in the word-embedding space and `beta_{k,v} = softmax_v(rho_v . alpha_k)`, but each author `a` also has a low-dimensional position `x_a`, and that position displaces the topic embedding before the softmax,

```
beta_{a,k,v} = softmax_v( rho_v . alpha_k + sum_j x_{a,j} (rho_v . W_{k,j}) ).
```

So two authors who discuss the same topic produce systematically different word distributions, shifted along the loading `W_k` by their position. `alpha_k` is the topic at the neutral position `x = 0`; the loading norm `||W_k||` is the topic's **discrimination**, large where word choice within the topic separates positions and near zero where the topic is neutral. The position is latent and estimated, not supplied, which makes this the unsupervised, latent-trait twin of the [`STM`](#stm) content covariate, and the embedding-native generalization of Wordfish ([Slapin and Proksch 2008](https://doi.org/10.1111/j.1540-5907.2008.00338.x)): with one topic and one dimension the log word-rate is `base_v + x_a (rho_v . w)`, Wordfish with a discrimination that is shared across semantically related words.

**Counts or word embeddings.** The representation is a fit-time choice. Pass no `word_embeddings` (the default) and the displacement is parameterized directly over the vocabulary, `beta_{a,k,v} = softmax_v(alpha_{k,v} + sum_j x_{a,j} W_{k,j,v})` — "Wordfish with topics", every word its own dimension, no embeddings needed. Pass `word_embeddings` with the aligned `vocabulary` and the same displacement is factored through them as above (the ETM form). Both are the same model; the embedding is a low-rank factorization of the displaced topic-word matrix. The difference is concentration: the embedding bottleneck localizes the discrimination onto a topic, while the full-vocabulary counts spread it across several, so the recovered author scale is the dependable output in the count form and `topic_discrimination` reads as suggestive. Empirically the two recover author positions about equally well, so the count form is the cheap, robust default and word embeddings are worth it when pretrained vectors carry signal the counts miss or when you want the discrimination to localize. `m.representation` reports which one a fitted model used.

```python
topica.enable_experimental()
m = topica.IdealPointTM(num_topics=30, num_dims=1, seed=0)
m.fit(docs,                                   # counts: no embeddings needed
      group=speaker_id,                       # documents sharing a speaker share a position
      anchors={"Sanders": -1.0, "Cruz": 1.0}) # orient the sign of the axis
m.author_positions          # (num_authors, num_dims): the estimated ideal points
m.position_se               # (num_authors, num_dims): standard error of each position
m.topic_discrimination      # (num_topics,): which topics carry the cleavage
m.position_shift(topic=k)   # the words that move within topic k from one end to the other

# or factor through word embeddings (ETM-style), passing the aligned vocabulary:
m.fit(docs, word_embeddings=rho, vocabulary=vocab, group=speaker_id)
```

**Uncertainty on the positions.** `position_se` is the standard error of each
author's ideal point, from the observed information of the penalized position
objective at the fit — the multinomial-content analog of Wordfish's Hessian-based
`se.theta`. It conditions on the fitted topic content and shrinks with the number of
tokens an author contributes, so a prolific author is placed more precisely than a
quiet one. The same getter is on `IdealPointSentenceTM` (the exact Laplace SE of its
linear-Gaussian position step) and on `TBIP` (the variational posterior SD); `Wordfish`
has had `position_se` all along. To carry that uncertainty into a polarization estimate,
`topica.polarization_ci` propagates the per-author SEs by simulation and returns a
confidence interval on the camp gap — a band that straddles zero means the camps are
not reliably apart. (`PartyEmbeddings`, whose positions are a PCA of doc2vec tag
vectors, has no analytic SE; use `topica.position_intervals` to bootstrap one.)

We fit by variational EM on ETM's core: the E-step is the logistic-normal Laplace step with the author's position-displaced `beta`, and the M-step updates the topic embeddings, the loadings, and the positions in turn. Positions are initialized from the leading principal components of the author-word matrix, as Wordfish does, which keeps the fit off the trivial zero-loading fixed point. Identification is exact and loss-free: each iteration standardizes the positions to mean zero and unit variance and absorbs the rescaling into the embeddings and loadings, then orients the sign to the anchors. On data simulated from the model the positions recover the planted trait at a correlation above 0.98 and `position_shift` reads off the discriminating axis.

We fit by variational EM on ETM's core, with one design choice worth knowing: the
position update reuses a per-topic grid over the latent axis so the cost stays manageable
as the number of authors grows, and the whole fit stays thread-count independent.

### When it works

The single most important thing to know is that **the genre of the text matters more than
any model setting**. We validated `IdealPointTM` against DW-NOMINATE on U.S. congressional
text. On floor speech, recovery is weak (Pearson around 0.4, mostly a party split with
little within-party ordering), because floor speech is dominated by procedure and
boilerplate. On congressional **press releases** for the same chamber, recovery is strong
and replicates across the 115th, 117th, and 118th Houses (Pearson 0.79 to 0.88), because
press releases are crafted ideological messaging. So reach for `IdealPointTM` when:

- the text is **expressive** (messaging, opinion, manifestos, op-eds), not procedural;
- you can **group by author** (`group=`) so each speaker, outlet, or legislator accumulates
  enough text for a stable position;
- the corpus is **clean** (one language, low boilerplate). The model's single position axis
  latches onto the dominant axis of within-topic word choice, so a strong off-topic axis
  (mixed languages, heavy templated text) will capture the scale. Filter those first.

On clean messaging text the model is competitive with Wordfish on the scale itself, and it
adds what Wordfish cannot: coherent topics and a per-topic account of how language differs
by position (`position_shift`).

### Walkthrough

`IdealPointTM` needs word embeddings aligned to your vocabulary, exactly like
[`ETM`](embedding.md). Training them on the corpus itself works well:

```python
import gensim, numpy as np, topica
topica.enable_experimental()

# docs: list[list[str]] (tokenized); author: one author label per document
w2v = gensim.models.Word2Vec(docs, vector_size=100, window=5, min_count=5, sg=1, seed=1)
vocab = list(w2v.wv.index_to_key)
embeddings = np.array([w2v.wv[w] for w in vocab])

m = topica.IdealPointTM(num_topics=20, num_dims=1, seed=1)
m.fit([[w for w in d if w in set(vocab)] for d in docs],
      word_embeddings=embeddings, vocabulary=vocab,
      group=author,                                   # one position per author
      anchors={"known_left": -1.0, "known_right": 1.0})  # orient the sign

# the scale
positions = dict(zip(m.author_names, m.author_positions[:, 0]))

# the topics (the other half of the double job)
m.top_words(8)                       # top words per topic
m.topic_discrimination               # (K,): which topics carry the cleavage

# how language splits by position, within the most discriminating topic
k = int(np.argmax(m.topic_discrimination))
pos, neg = m.position_shift(k, n=10)  # (positive-end words, negative-end words)

m.save("ideal.topica"); m2 = topica.IdealPointTM.load("ideal.topica")
```

`author_positions` are standardized (mean 0, unit variance per dimension). `anchors` only
fixes the otherwise-arbitrary sign and scale, so pass two authors you know sit on opposite
ends; the magnitude of the recovered ordering does not depend on them. `position_shift`
defaults to a probability-weighted score that keeps the contrast inside the topic's own
vocabulary; pass `weighting="logratio"` for the older, rare-word-sensitive ranking. The raw
loadings are exposed as `m.loadings` for inspecting the discrimination directions.

**Word embeddings, not sentence embeddings.** When you do pass embeddings, `IdealPointTM`
factors the topic-word matrix through per-word vectors `rho`, so it takes word (or phrase)
embeddings, like ETM, not document-level sentence embeddings (for those, see
[`IdealPointSentenceTM`](#idealpointsentencetm)). We use
word2vec trained on the corpus, and gensim phrase detection (bigrams/trigrams, so
`estate_tax` becomes one token) helps modestly. We also tested sentence embeddings
(Sentence-Transformers, discretized into sentence "concepts"): they make ideology a more
accessible raw axis, but inside the model they show no consistent advantage over word2vec
for recovering external scores, and on messaging text word2vec with phrases matched or beat
them. Representation is a minor lever next to genre; word2vec with phrases is the default we
recommend.

### Limits

`IdealPointTM` stays experimental for good reason. Its single position axis behaves as a
party detector first and an ideology gradient second, and it is more fragile than plain
Wordfish to a contaminating off-topic axis, so clean inputs matter. Which topic carries the
discrimination is not always stable, since a partisan contrast can show up either as
within-topic content or as topic-splitting. And `num_dims > 1` is implemented but only
robustly identified for the first dimension; a multi-dimensional, issue-specific position
(in the spirit of a hierarchical ideal-point topic model) is the natural next step and the
likely path to both finer interpretation and more robustness.

## Wordfish

`Wordfish` ([Slapin and Proksch 2008](https://doi.org/10.1111/j.1540-5907.2008.00338.x)) is the standard text-scaling model and the word-frequency baseline in the [`IdealPointTM`](#idealpointtm) family: it places authors on a single latent axis from word counts alone, with no topics and no embeddings. It is here so you can measure what topics and embeddings actually add. The count of word `j` by author `i` is Poisson with `log rate = alpha_i + psi_j + beta_j * theta_i`, where `theta_i` is the author position, `beta_j` the word discrimination, `psi_j` its baseline log-rate, and `alpha_i` the author verbosity.

```python
m = topica.Wordfish()
m.fit(docs, group=author,                      # pool documents into one position per author
      anchors={"Sanders": -1.0, "Cruz": 1.0})  # orient the sign of the axis
m.author_positions          # (num_authors, 1): standardized positions
m.word_discrimination       # (vocab,): per-word beta
m.discriminating_words(10)  # the words at the two ends of the axis
```

We fit by the standard Wordfish EM: alternate Newton updates of the per-word `(psi, beta)` and per-author `(alpha, theta)`, with weak Gaussian priors on `beta` and `theta` (`beta_prior_sd`, `theta_prior_sd`; pass `math.inf` for none). Identification is applied every iteration and is lossless: `theta` is standardized to mean 0 / unit variance (the scale absorbed into `beta`, the location into `psi`), `psi` is centered into `alpha`, and the sign is oriented to the anchors — or, with no anchors, to the first two authors (`theta[0] < theta[1]`), mirroring quanteda's default `dir = c(1, 2)`. There is no RNG and the reductions run in a fixed order, so the fit is bit-reproducible. We validate against `quanteda.textmodels::textmodel_wordfish`: on a corpus sampled from the model the two recover the same scale at correlation 1.00 (`parity/wordfish_r_compare.py`).

### Controlling for a confound

Text scaling fails in a specific, well-documented way: when a corpus has a dominant axis of variation that is *not* the one you want (a chamber, a government/opposition split, an era, a language), the single latent position latches onto it and the ideological signal is lost. Wordfish accepts a `control` covariate to absorb exactly that. Pass a categorical label per document (constant within each author); each non-baseline level gets a per-word log-rate offset `delta[level, word]`, so systematic level-specific word usage is explained away instead of contaminating `theta`. The model becomes `log rate = alpha_i + psi_j + beta_j * theta_i + delta[level_i, j]`.

```python
m = topica.Wordfish()
m.fit(docs, group=author, control=chamber,        # absorb the chamber's word usage
      anchors={"Sanders": -1.0, "Cruz": 1.0})
m.control_names           # the level labels (row 0 is the held-out baseline)
m.control_word_offsets    # (num_levels, vocab): the absorbed per-level word effects
```

On a corpus where a control-aligned nuisance axis dominates, plain Wordfish recovers the ideological scale at essentially zero correlation while `control=` restores it (in our planted test, `|r|` with the true scale rises from ~0.03 to ~0.8). The initialization is residualized by level too, so `theta` does not start on the nuisance axis. With no `control` the fit is exactly the historical Wordfish, bit-for-bit.

As a scaling model Wordfish has no topics, so it cannot tell you what is being talked about or how language differs within a topic. When you want the scale and the topics together, reach for [`IdealPointTM`](#idealpointtm) (word embeddings). On clean messaging text the two are comparable on the scale itself; IdealPointTM adds the per-topic framing Wordfish structurally cannot produce.

## Wordshoal

`Wordshoal` ([Lauderdale and Herzog 2016](https://doi.org/10.1093/pan/mpw017)) is the multi-domain extension of [`Wordfish`](#wordfish). A single Wordfish fit over an entire corpus of legislative speech is dominated by *what* each debate is about, not *where* a speaker sits ideologically: the biggest axis of word variation is the topic of the day, so the recovered scale mixes agenda with ideology. Wordshoal removes the agenda by scaling each debate *separately* and then combining. You supply, per document (speech), a `speakers` label and an externally-known `domains` (debate) label — the domains are trusted metadata, not inferred topics.

```python
m = topica.Wordshoal()
m.fit(docs, speakers=speaker, domains=debate,      # one speech per row; both labels required
      anchors={"Sanders": -1.0, "Cruz": 1.0})      # orient the sign of the shared axis
m.author_positions   # (num_authors, 1): the cross-domain actor scale
m.position_se        # (num_authors,): standard error of each position
m.domain_scales      # (num_domains, 2): per-debate [intercept, loading]
m.word_scores("health-2019", n=10)   # the discriminating words within one debate
```

The fit has two stages. **Stage 1** runs Wordfish independently on each debate, giving every speech a within-debate position `psi`. Each debate's axis is standardized and oriented arbitrarily — its sign is not comparable across debates, and that is fine. **Stage 2** is a one-dimensional linear factor model over the stacked within-debate positions, `psi = alpha_j + beta_j * theta_i + noise`, where `theta_i` is the actor's cross-domain position and the debate loading `beta_j` absorbs each debate's arbitrary sign and scale — a debate that runs "backwards" simply gets a negative `beta_j`. It is fit by the reference's conditional-maximum-likelihood coordinate ascent (weakly-informative priors on the loadings, intercepts, positions, and per-actor precisions; deterministic initialization). Identification follows the reference: positions are prior-scaled (not re-standardized to unit variance, unlike Wordfish) and identified up to sign, which `anchors` — or, with no anchors, the first two speakers — orient. The fit has no RNG and is bit-reproducible.

Two edge cases are surfaced rather than hidden. Every domain must have at least two speeches to be scaled (a single-document debate raises an error, so you group and filter to debates with two or more speakers before fitting). And if the speaker-domain graph is *disconnected* — two blocs of speakers that never debate the same topics — the scale is not identified across the blocs; `num_components` reports it, `fit` warns, and `author_components` labels each actor's bloc, because positions in different components are not comparable. Pass `speakers`/`domains` as any labels (strings, ints, dates); they are coerced to strings, and `domain_names` is then sorted lexicographically.

The multi-domain benefit — recovering ideology rather than the agenda of each debate — is best read as *directional*: on corpora with few actors the estimated positions carry real sampling noise, so treat the improvement over plain `Wordfish` as a qualitative gain (better party separation), not a precise number. We validate against an R reference oracle (`quanteda.textmodels` Wordfish per debate plus the paper's stage-2 ascent): on a two-stage synthetic corpus topica and the oracle recover the same actor scale at correlation 1.00, and both recover the planted positions (`parity/wordshoal_r_compare.py`).

Reach for Wordshoal over [`Wordfish`](#wordfish) when your speeches carry trusted debate labels and you want a scale comparable *across* debates; reach for [`TBIP`](#tbip) when the issue domains are unknown and should be learned from the text instead.

## IdealPointSentenceTM

!!! warning "Experimental"
    `IdealPointSentenceTM` is gated: call `topica.enable_experimental()` (or set
    `TOPICA_EXPERIMENTAL=1`) before constructing one.

`IdealPointSentenceTM` is the embedding-native analog of [`IdealPointTM`](#idealpointtm), working on sentence or document *embeddings* instead of words. Topics are Gaussian clusters in embedding space with centroids `mu_k`; an author position `x_a` displaces a topic centroid along a loading `V_k`, so an embedding from author `a` in topic `k` is drawn from `N(mu_k + sum_j x_{a,j} V_{k,j}, sigma^2 I)`. Where IdealPointTM shifts a topic word-softmax by position, IdealPointSentenceTM shifts a topic centroid; `||V_k||` is the discrimination.

```python
topica.enable_experimental()
# embeddings: (N, D) sentence or document embeddings; group: author per row
m = topica.IdealPointSentenceTM(num_topics=20, num_dims=1)
m.fit(embeddings, group=author, anchors={"left_author": -1.0, "right_author": 1.0})
m.author_positions       # (num_authors, num_dims)
m.doc_topic              # (N, num_topics): soft topic assignment per embedding
m.topic_centroids        # (num_topics, D)
m.topic_discrimination   # (num_topics,)
```

Inference is closed-form EM over a Gaussian mixture: the E-step is the soft topic assignment, and the M-step solves weighted least squares for each topics `(mu_k, V_k)`, a small linear system for each authors position, and a residual update for the variance. Positions are standardized each iteration (absorbed losslessly into `mu`/`V`) and oriented to the anchors, exactly as in the other ideal-point models.

This is the only model in the family that takes embeddings *directly* rather than deriving topic-word distributions, so it has no `topic_word` or `coherence` — its topics are clusters, summarized by `topic_centroids` (use a nearest-document or nearest-word lookup to label them). It is the continuous corner of the comparison with [`IdealPointTM`](#idealpointtm) (word tokens, as counts or word embeddings) and [`Wordfish`](#wordfish) (word counts, no topics): pass sentence embeddings grouped by author to ask whether a continuous representation recovers the same latent scale.

## TBIP

`TBIP` is Text-Based Ideal Points (Vafa, Naidu & Blei 2020), a Poisson factorization of word counts. A *neutral* topic-word intensity `beta_kv` is rescaled by a per-word *ideological* factor `exp(x_s * eta_kv)`, where `x_s` is the author's latent ideal point and `eta_kv` is how strongly word `v` in topic `k` separates the two ends of the axis. A document by author `a_d` mixes topics with positive per-doc intensities `theta_dk`:

```text
y_dv ~ Poisson( sum_k theta_dk * beta_kv * exp(x_{a_d} * eta_kv) )
```

A positive `eta_kv` makes word `v` more likely as the author moves to the positive end of the scale; a near-zero `eta_kv` makes the word non-ideological. The position `x_s` is estimated from the text alone — no votes, no labels.

```python
m = topica.TBIP(num_topics=15)
m.fit(docs, group=author)        # group: author label per document
m.ideal_points                   # (num_authors,): author positions (posterior mean)
m.author_names                   # aligned with ideal_points
m.topic_word                     # (num_topics, vocab): neutral topics, exp(mu_beta) normalized
m.ideological_topics             # (num_topics, vocab): eta, the per-word ideological loadings
m.doc_topic                      # (num_docs, num_topics)
```

Inference is the paper's mean-field variational inference (not the MAP shortcut): a fully factored `q` with LogNormal factors for the positive `theta`/`beta` and Normal factors for the real `eta`/`x`, maximized by reparameterized single-sample stochastic gradient ascent (Adam) with document minibatching. KL is analytic for the Gaussian factors; the LogNormal-vs-Gamma terms use the same Monte Carlo sample. The fit is deterministic under a fixed `seed`. TBIP is the word-count member of the ideal-point family that, unlike [`Wordfish`](#wordfish), carries topics, and unlike [`IdealPointTM`](#idealpointtm) in its count form, separates a word's neutral intensity from its ideological loading explicitly.

## PartyEmbeddings

`PartyEmbeddings` ([Rheault and Cochrane 2020](https://doi.org/10.1017/pan.2019.26)) is the corpus-trained word-embedding member of the ideal-point family. Where the others either count words or read pretrained embeddings, this one *learns* its own embeddings from the corpus and places parties by where their learned vectors land. It trains a PV-DM (distributed-memory paragraph-vector) model: a shallow network that predicts each word from the mean of its context-word embeddings plus the document's metadata-tag embeddings, fit by negative sampling. The tags are political metadata, by default a party-period label (so each party gets a vector per parliament, and parties can move over time), with an optional second `control` tag (government status, region) that absorbs a confound without being placed. Because the tag vectors are trained in the same space as the word vectors, you can read a party's language directly off its neighbors.

```python
m = topica.PartyEmbeddings(num_dims=2, vector_size=200, window=20, seed=1)
m.fit(docs, group=party_period,                 # e.g. "D_114", "R_114"
      control=parliament,                        # optional confounder tag
      anchors={"D_114": -1.0, "R_114": 1.0})     # orient the axis
m.author_positions          # (num_parties, num_dims): PCA of the party vectors; col 0 is left-right
m.author_names              # the party-period labels, row order of author_positions
m.nearest_words("R_114")    # the words closest to a party (its "linguistic specificity")
m.guided_positions(left=["public", "workers"], right=["market", "taxpayers"])  # a custom axis
m.distance("D_114", "R_114")  # Euclidean distance between two parties (polarization)
```

The placement is the leading principal components of the learned party vectors: the first is the latent left-right scale, oriented by the `anchors`. The fit is single-threaded stochastic gradient descent, so it is reproducible from a fixed `seed`. The negative-sampling objective is the standard word2vec/doc2vec update (Slapin-style party-period indicators, but estimated at the word level in context). We validate against the gensim `Doc2Vec` reference the original package builds on: on a corpus sampled with a planted party ordering, the two recover the same scale at correlation 1.00 (`parity/party_embeddings_compare.py`).

This is the model to reach for when you want the scale to come from *learned* representations of language in context rather than raw counts ([`Wordfish`](#wordfish), [`IdealPointTM`](#idealpointtm) in its count form) or a topic-structured generative model ([`IdealPointTM`](#idealpointtm) with word embeddings); it is the natural comparison point for asking what topics and a latent-position head add over a plain embedding scaler. It is a pure scaling model: it has no `topic_word`, only party and word vectors and their placement.

Two notes on use. `nearest_words` returns the raw cosine ranking of words to a party; high-frequency function words can crowd the top, so read it relative to a baseline (compare a party's neighbors against another party's, or against the corpus-average party) rather than in isolation. And phrase detection (collocations like "health care", "free enterprise") is preprocessing the caller does upstream: `PartyEmbeddings` consumes token lists, so phrase them before fitting if you want multiword expressions, as the paper does. Implementation notes for fidelity: the hidden layer is the mean of the context-word and tag vectors, the context window shrinks dynamically per token (standard word2vec), and the fit is single-threaded so a fixed `seed` is bit-reproducible (multi-threaded async SGD would not be).

## Validating an ideal-point axis without an external scale

The ideal-point family ([`Wordfish`](#wordfish), [`IdealPointTM`](#idealpointtm), [`IdealPointSentenceTM`](#idealpointsentencetm), [`TBIP`](#tbip), [`PartyEmbeddings`](#partyembeddings)) returns author positions, but how do you know the discovered axis is a real, partisan dimension rather than an artifact, *without* a validated external score like DW-NOMINATE? topica ships intrinsic diagnostics that answer this from the model and the text alone.

`topica.bimodality(positions)` is the bimodality coefficient of the positions: above ~0.555 the authors split into two camps (a polarized, two-pole structure) rather than one blob. It is computed from `author_positions` alone.

`topica.polarization(positions, labels)` measures how far two known camps sit apart on the axis: the distance between the camps' centroids, with `labels` assigning each author to a camp (e.g. their party). It works on any model's `author_positions` (1-D, or Euclidean distance for a multi-dimensional fit), so calling it once per time period traces polarization over time, the way Rheault and Cochrane (2020) use the distance between party embeddings. Pass `normalize=True` for an effect-size form (divided by the pooled within-camp spread) that is comparable across corpora and model scales. Where `bimodality` asks whether *some* two-camp structure exists without labels, `polarization` measures the separation of camps you can name.

`topica.split_half_reliability(fit, group)` refits the scale on two disjoint halves of each author's documents and correlates the two position vectors. A high value means the axis is a stable, reproducible trait of the text, not an artifact of one fit. You supply a one-line `fit` closure, so it is model-agnostic:

```python
import topica
topica.enable_experimental()

def fit(idx):                        # fit on a subset of unit (document) indices
    m = topica.IdealPointTM(20, seed=1)
    m.fit([docs[i] for i in idx], group=[author[i] for i in idx])
    return m.author_names, m.author_positions[:, 0]

m = topica.IdealPointTM(20, seed=1); m.fit(docs, group=author)
topica.bimodality(m.author_positions)        # > 0.555 => two camps (polarized)
topica.polarization(m.author_positions, party_of_author)  # gap between named camps
topica.split_half_reliability(fit, author)   # how much real signal the axis carries
```

We validated these against DW-NOMINATE on U.S. House press releases: split-half reliability tracks the external recovery across congresses (it ranks them correctly and approximates the magnitude), so it stands in for an external scale when none exists. By measurement theory the reliability also bounds how well the axis can correlate with *any* external score, so a low value is an early warning that the axis is too noisy to validate.

For uncertainty on the positions themselves, `topica.position_intervals(fit, group)` returns model-agnostic **bootstrap** standard errors and confidence intervals for any of the four models — it resamples each author's documents and refits, so it reflects the real estimation variability (including the seed-to-seed instability a local analytic SE would miss). `Wordfish` additionally exposes an analytic `position_se` (the Hessian-based standard error, validated equal to R `quanteda`'s `se.theta` at correlation 1.00 in `parity/wordfish_r_compare.py`).

```python
m = topica.Wordfish(); m.fit(docs, group=author, anchors={"left": -1.0, "right": 1.0})
m.position_se                                 # analytic SE per author (quanteda-equivalent)

def fit(idx):                                 # bootstrap intervals for any model
    mm = topica.IdealPointTM(20, seed=1)
    mm.fit([docs[i] for i in idx], group=[author[i] for i in idx])
    return mm.author_names, mm.author_positions[:, 0]
ci = topica.position_intervals(fit, author, n_boot=50)   # author -> (estimate, se, lo, hi)
```

## Short-text models

`PT` and `GSDMM` are built for short documents; see the
[short-text guide](short-text.md).

## SupervisedLDA

Topics shaped to predict a per-document real-valued response (Blei & McAuliffe).
`coefficients` give each topic's pull on the outcome, and `predict` scores new
documents.

Both come with uncertainty, reported as *conditional* variational
approximations. `coefficient_se` is the standard error of each regression
coefficient from the OLS covariance `σ²M⁻¹` of the same normal equations the fit
solves. It conditions on the fitted topics, `β`, and the variational moments, so
it does not propagate topic or `β` uncertainty; read `|coef| > ~2·SE` as an
informal cue for which topics move the outcome, not a calibrated significance
test. `predict(docs, return_std=True)` returns `(mean, std)`, where `std`
propagates the new document's variational topic uncertainty through the
regression plus the residual `σ²`. This is a conditional predictive spread (the
fitted `β`, `η`, `σ²` held fixed), not a full Bayesian posterior-predictive
interval; `mean ± 1.96·std` is a Gaussian approximation under those conditions.

```python
m = topica.SupervisedLDA(num_topics=20, seed=1)
m.fit(docs, y)
z = m.coefficients / m.coefficient_se        # which topics matter
mean, std = m.predict(new_docs, return_std=True)
```

## LabeledLDA

Supervised: each label is a topic, and a document's tokens are restricted to its
labels. Empty labels fall back to unconstrained LDA.

## DiscLDA

`DiscLDA` ([Lacoste-Julien, Sha & Jordan 2008](https://papers.nips.cc/paper/2008/hash/7b13b2203029ed80337f27127a9f1d28-Abstract.html))
is a **discriminative** topic model: given a document-level class label, it learns a
topic space that separates what each class talks about *distinctively* from the
*common ground*. The actual topics partition into `k_class` topics **specific to each
class** (one block per class) and `k_shared` topics **shared by all classes**; a
document of class `c` places topic mass only on its own class block and the shared
block. So for "how do the parties talk differently," DiscLDA hands you the
Republican-specific topics, the Democrat-specific topics, and the shared topics
directly, rather than making you fit unsupervised topics and hunt for the ones that
split.

```python
m = topica.DiscLDA(k_class=8, k_shared=12)     # L = num_classes*8 + 12 topics
m.fit(docs, y=party)                            # one class label per document
m.class_topics("R"); m.class_topics("D")        # each party's distinctive topics
m.shared_topics()                               # the common-ground topics
m.transform(new_docs)                           # class-carrying document features
m.predict(new_docs)                             # DiscLDA as a classifier
```

This is the **fixed block-transform** variant (paper §4.1): with the transform frozen
to the shared/class-specific block structure, DiscLDA is LDA with a per-document topic
restriction (structurally like `LabeledLDA`, but restricting to `class-block ∪
shared-block` rather than a document's own labels), fit by collapsed Gibbs. The
class-marginalized representation `transform` returns — Σ_c p(c|w)·θ_c — is the
supervised, discriminative feature vector the paper uses for classification. Where
`SupervisedLDA` regresses a real-valued response off *all* topics and `LabeledLDA`
restricts tokens to a document's own labels, DiscLDA is the one that builds the
class-specific-vs-shared split and a discriminative representation. Determinism is
`seed-reproducible`.

`predict` / `predict_proba` are a topica-native **plug-in** classifier, not part of
the paper: each class score is the document's posterior-mean-θ likelihood combined
with a class prior and softmaxed, so treat `predict_proba` as a prior-calibrated
score rather than exact evidence. The prior is set by `class_prior`: `"empirical"`
(default) uses the observed class frequencies from fit — so on an imbalanced corpus
the probabilities track prevalence, and an empty/all-out-of-vocabulary document
returns the prior (the majority class) rather than a sorted-order tie. Pass
`class_prior="uniform"` for an equal prior, or a sequence of per-class weights (in
`classes` order) for a custom one; `m.class_prior` and `m.class_counts` report what
was used.

DiscLDA has no canonical reference implementation, so it is validated against the
paper's 20 Newsgroups result (`parity/disclda_20ng.py`): DiscLDA's topic-proportion
features feed a linear classifier better than unsupervised-LDA features of matched
dimension. On the paper's hard `alt.atheism` / `talk.religion.misc` pair, topica
reproduces that ordering (DiscLDA features clearly above LDA features). The learned
transform (paper §4.2, the full discriminative training of `T`) is a planned
follow-up; the fixed-transform model already delivers the shared/class-specific
structure and the discriminative-feature win.

## RTM

`RTM` ([Chang & Blei 2010](https://www.jstor.org/stable/27801582), the *relational
topic model*) is for corpora that come with a **graph**: papers with a citation
network, web pages with hyperlinks, bills with co-sponsorship, states with
geographic adjacency. Every other model in the roster treats documents as
exchangeable and ignores the links; RTM fits the topics and the link structure
*jointly*, so the same topics that explain the words also explain who connects to
whom. That coupling is the point — it lets the model predict a document's links
from its words alone, and it sharpens the topics toward distinctions that the
network cares about.

```python
edges = [(0, 3), (0, 7), (3, 7), ...]          # undirected (i, j) document pairs
m = topica.RTM(num_topics=20, link="logistic") # or link="exponential"
m.fit(docs, edges)
m.predict_link(0, 3)                            # plug-in link probability
m.suggest_links(new_doc, top_n=10)             # rank citations for an unseen doc
m.eta, m.nu                                     # how topic co-occurrence drives links
m.phi_bar                                       # mean topic assignments (the link quantity)
```

RTM is LDA plus a link head: for each observed pair of documents a binary link is
drawn from a function of the two documents' mean topic-assignment vectors
`z̄_d = (1/N_d) Σ_n z_{d,n}`. The link coupling is to `z̄` (not to the Dirichlet mean
`θ`), following supervised LDA — that is what ties links and words to the *same*
topics, and it is why `m.phi_bar` (the quantity the link function reads) is exposed
separately from `m.doc_topic`. We fit by variational EM (paper §3), modelling only
the **observed** links, so cost scales with the number of links, not with `D²`.
Two link functions ship: `logistic` (default, `σ(ηᵀ(z̄_d ∘ z̄_{d'}) + ν)`, a bounded
concave regression) and `exponential` (`exp(ηᵀ(z̄_d ∘ z̄_{d'}) + ν)`, with a
closed-form M-step). Positive-only links make the link estimate a one-class problem,
so both paths use the paper's `ρ` regularization (`negative_ratio` pseudo-negative
links placed at the expected topic co-occurrence under the prior). The logistic
path also applies the paper's ℓ2 ridge on the link coefficients (`ridge=`, default
`1.0`; App B recommends the ℓ2 regularizer "in lieu of or in conjunction with" the
`ρ` term). The ridge is not optional in practice: `ρ`'s pseudo-negatives all sit at
the single point `π̄_α`, so they cannot constrain coefficient directions orthogonal
to it, and with `ridge=0` the logistic coefficients diverge under separable link
structure (topic recovery survives, but `predict_link` degenerates to 0/1). The
exponential link is bounded and needs no ridge. Links are treated
as **undirected** (the paper symmetrizes; directed RTM is a planned follow-up).
Determinism is `seed-reproducible` (a serial, seeded E-step). On large graphs the
`exponential` link is markedly faster — its link M-step is a closed form and the
fit converges in a handful of EM iterations, where `logistic` runs an iterative
gradient M-step each round; the two recover the same structure, so prefer
`exponential` when the network is large.

A second inference backend is available via `inference="gibbs"`: collapsed Gibbs,
matching R `lda`'s `rtm.em` (which wraps `rtm.collapsed.gibbs.sampler`). The default
`inference="variational"` is the shipped EM.

```python
m = topica.RTM(20, inference="gibbs").fit(docs, links=edges)   # R lda's algorithm
```

The variational and Gibbs backends validate against *different* references. The R
`lda` package's `rtm.em` is collapsed Gibbs, not the paper's variational EM, so for
the **variational** path R can only be a directional baseline — it is instead
validated against a standalone NumPy implementation of the paper's variational
equations (`parity/rtm_reference.py`, finite-difference-checked on the link gradients
and the ρ term): topica reproduces it to aligned topic-word cosine ≈ 1
(`parity/rtm_compare.py`). The **Gibbs** path ports R `lda`'s exact mechanics — a
fresh sampler restarted each M-step, per-`(word, count)`-cell block sampling, and
the same link factor and `estimate.params` β update — so R becomes an authoritative
same-algorithm oracle. On a fixed planted network topica's Gibbs fit clears R's own
seed-to-seed self-consistency floor (topic-word cosine **0.999 vs R self 0.999**;
`parity/rtm_gibbs_gold.py`), and its link coefficient matches R's to ~0.001.

Two properties of R's `rtm.em` shape how the parity is measured. First, R restarts
each M-step, so quality comes from a *converged* E-sweep budget, not accumulation;
R's Mersenne-Twister and topica's RNG mix at different rates, so the gold compares
the two at convergence rather than at R's marginal 8-sweep default. Second, R's
`estimate.params` (`β_k = log(p_k/(p_k+reg))`) fits a *negative* coefficient even on
strongly-linked data, and that β can run away over many M-steps (R's own fit
degenerates by ~10 M-steps) — so a modest M-step budget is used. The negative β
means the Gibbs backend's link scores rank dissimilar documents higher: use
`inference="variational"` for `predict_link` / `suggest_links`, and the Gibbs
backend for same-algorithm topic parity.

## SAGE

Content-covariate topics via an additive log-linear model
([Eisenstein, Ahmed & Xing 2011](https://icml.cc/2011/papers/534_icmlpaper.pdf)):
the *same* topic is worded differently across groups. The log topic-word weight is
a background `m` plus **sparse deviations** `κ` (topic, group, and topic×group), so
each group's phrasing is read as a short list of words it up- or down-weights.
`word_contrast(topic, a, b)` shows the words that most distinguish two groups'
phrasing; `content_kappa` exposes the fitted deviations directly.

The sparsity is the point, and it is controlled by `prior=`:

- `prior="laplace"` (**default**) is canonical sparse SAGE — a Laplace prior on `κ`,
  fit by adaptive reweighting, that drives most deviations to ~0.
- `prior="gaussian"` is the dense L2-ridge content model (the STM-style variant).
- `prior="jeffreys"` is a more aggressive sparse prior.

The κ are re-estimated by L-BFGS between Gibbs sweeps. The prior is faithful to the
paper's sparsity mechanism; note that topica infers the topic assignments by
collapsed Gibbs and re-estimates κ periodically by MAP, where the ICML derivation
uses variational expected counts — the model is SAGE, but the inference is not a
literal reproduction.

> **0.5 note:** the default prior changed from the earlier Gaussian ridge to the
> sparse Laplace prior. This is a deliberate correctness fix (#422) — the old
> default was not SAGE's defining sparse prior. Pass `prior="gaussian"` to recover
> the previous behaviour exactly. The sparse deviations change `β`, so a default
> fit's topic-word distributions and held-out `transform`/`doc_topic` differ from
> before (strong group structure is still recovered). A SAGE model saved before
> this change must be re-fit; the older on-disk layout is not migrated.

## Hierarchy models

`PA` (Pachinko Allocation) and `HLDA` (hierarchical, nested-CRP) recover
super-/sub-topic structure.

### HLDA level prior

`HLDA` places each document on a root-to-leaf path and assigns every token a level
along it. The per-document distribution over levels has a selectable prior
(`level_prior=`):

- `"dirichlet"` (default) — a fixed-depth Dirichlet whose concentration `alpha` is
  either a scalar (symmetric, default `0.1`) or a length-`depth` list. A root-heavy
  vector such as `alpha=[5.0, 0.5, 0.1]` biases generic vocabulary toward the
  shallow levels rather than leaving that to the likelihood. This matches
  tomotopy's `HLDAModel`, which likewise takes a scalar or per-level `alpha`.
- `"gem"` — the two-parameter GEM stick-breaking prior of Blei, Griffiths & Jordan
  (2010), with `gem_mean` (`0 < m < 1`, default `0.5`) and `gem_scale` (`> 0`,
  default `100`). It biases mass toward shallower levels by construction; a smaller
  `gem_mean` concentrates documents nearer the root (fewer, broader leaf topics).
  We apply the standard finite (Ishwaran–James) truncation, so the deepest level
  absorbs the truncated tail and the level distribution is proper.

```python
import topica

# asymmetric Dirichlet: keep generic words shallow
m = topica.HLDA(depth=3, alpha=[5.0, 0.5, 0.1]).fit(docs)

# GEM stick-breaking prior (the original paper's prior)
m = topica.HLDA(depth=3, level_prior="gem", gem_mean=0.35).fit(docs)
```

The default (`level_prior="dirichlet"`, scalar `alpha=0.1`) is unchanged from
earlier releases and fits bit-for-bit identically. `beta` (topic-word) stays a
single scalar and the hyperparameters are held fixed, as before.

### `beta` sets the tree granularity

The single most important knob for the *size* of the inferred tree is `beta`, the
topic-word Dirichlet. A small `beta` makes each node's word distribution sharp, so
the collapsed posterior rewards many pure, fine-grained topics; a large `beta`
smooths the nodes and the posterior prefers a few broad topics. On a 500-document
20 Newsgroups sample (`depth=3`, `gamma=0.1`) the number of inferred nodes moves
sharply with `beta`:

| `beta` | nodes |
|--------|-------|
| 0.01 (default) | ~300 |
| 0.1  | ~80 |
| 1.0  | ~10 |

This is not a convergence artifact — it is the collapsed posterior mode at each
`beta` (verified by the joint log-probability), and the fit is much faster with a
larger `beta` because runtime scales with the number of nodes. topica's default
`beta=0.01` is a sharp calibration that produces many specific topics; if you want
a compact, coarse hierarchy (and a faster fit), raise `beta` toward `0.5`–`1.0`.
Note that reference implementations such as tomotopy tend to settle on a small,
coarse tree even at a sharp `beta`; topica instead follows the posterior to the
finer tree, so match `beta` to the granularity you want rather than expecting the
same node count.
