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

### General-purpose

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `LDA` | text | gibbs | seed-reproducible | Classic latent Dirichlet allocation via a fast SparseLDA collapsed-Gibbs sampler. |
| `CTM` | text | variational | bit-exact | Correlated topic model: a logistic-normal prior that lets topics co-occur. |
| `ProdLDA` | text | vae | seed-reproducible | Product-of-experts LDA (AVITM) for sharper, more coherent topics; hand-coded VAE. |
| `HDP` | text | gibbs | seed-reproducible | Hierarchical Dirichlet process: infers the number of topics from the data. |
| `NMF` | text | matrix-factorization | bit-exact | Non-negative matrix factorization of the document-term matrix via multiplicative updates. |
| `LSA` | text | svd | bit-exact | Latent semantic analysis: a truncated SVD of the weighted document-term matrix. |

### Covariates & structure

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `STM` | text, metadata | variational | bit-exact | Structural topic model: relate topic prevalence and content to covariates. |
| `STS` | text, metadata | variational | bit-exact | Structural topic-and-sentiment model over document metadata. |
| `SAGE` | text, metadata | gibbs | seed-reproducible | Sparse additive generative model: the same topic worded differently across groups. |
| `ECTM` | text, metadata, times | variational | seed-reproducible | Evolving content topic model: STM content covariates that vary by group and drift across time periods. |
| `DMR` | text, metadata | gibbs | seed-reproducible | Dirichlet-multinomial regression: a document-metadata prior on topic proportions. |
| `GDMR` | text, metadata | gibbs | seed-reproducible | Generalized DMR with a smooth (Legendre-basis) prior over continuous covariates. |

### Guided & supervised

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `KeyATM` | text, seeds | gibbs | seed-reproducible | Keyword-assisted topics: anchor named topics with a few seed words each. |
| `SeededLDA` | text, seeds | gibbs | seed-reproducible | Seeded LDA: steer named topics toward supplied seed words. |
| `LabeledLDA` | text, labels | gibbs | seed-reproducible | Labeled LDA: each document label is a topic; tokens are restricted to its labels. |
| `SupervisedLDA` | text, labels | gibbs | seed-reproducible | Supervised LDA: topics shaped to predict a per-document real-valued response. |

### Short text

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `GSDMM` | text | gibbs | seed-reproducible | Gibbs-sampling Dirichlet mixture: one topic per short document. |
| `PT` | text | gibbs | seed-reproducible | Pseudo-document topic model: pool short texts into pseudo-documents. |

### Dynamic & hierarchical

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `DTM` | text, times | variational | bit-exact | Dynamic topic model: a fixed topic set whose word distributions drift across time slices. |
| `DETM` | text, embeddings, times | vae | seed-reproducible | Dynamic embedded topic model: embedding-factored topics that drift across time slices, fit as an amortized VAE. |
| `HLDA` | text | gibbs | seed-reproducible | Hierarchical LDA (nested CRP): a learned tree of super- and sub-topics. |
| `PA` | text | gibbs | seed-reproducible | Pachinko allocation: a DAG of super- and sub-topics. |

### Embedding-based

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `BERTopic` | text, embeddings | clustering | seed-reproducible | Cluster document embeddings; label topics by class-based TF-IDF. |
| `Top2Vec` | text, embeddings | clustering | seed-reproducible | Topics as dense regions in a joint document-word embedding space. |
| `ETM` | text, embeddings | variational | bit-exact | Embedded topic model: topic-word distributions factored through word embeddings. |
| `FASTopic` | text, embeddings | optimal-transport | bit-exact | Topics from optimal-transport plans between document, topic, and word embeddings. |
| `EmbeddingLDA` | text, embeddings, seeds | gibbs | seed-reproducible | Seeded LDA whose seed sets are expanded with nearest neighbors in an embedding space. |
| `CombinedTM` | text, embeddings | vae | bit-exact | Contextualized ProdLDA: encoder reads the bag of words plus a document embedding. |
| `ZeroShotTM` | text, embeddings | vae | bit-exact | Contextualized ProdLDA: encoder reads the document embedding alone, enabling cross-lingual transfer. |
| `InfoCTM` | text, dictionary | vae | seed-reproducible | Cross-lingual: two ProdLDA models aligned by a bilingual dictionary through a mutual-information term. |

### LLM-based

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `TopicGPT` | text, llm | prompting | llm-bounded | LLM-driven topic discovery: prompt a model to propose, refine, and assign a topic taxonomy with descriptions. |

<!-- END MODEL TABLE -->

## LDA

Classic Latent Dirichlet Allocation via MALLET's fast SparseLDA collapsed-Gibbs
sampler. Fits are bit-for-bit reproducible, with optional approximate
multi-threaded training.

```python
import topica
model = topica.LDA(num_topics=20, seed=42)
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

## STM

The full Structural Topic Model: CTM core plus **prevalence** and **content**
covariates. This is the workhorse for social science; it has its own
[guide](covariates.md).

Like CTM, STM takes `variational="diagonal"` to use the mean-field E-step in
place of the default Laplace one (`variational="laplace"`): faster at high K, but
it drops the off-diagonal posterior covariance, so the precision of
topic-correlation and method-of-composition standard errors is lower.

## ECTM

The Evolving Content Topic Model extends STM's content covariate with **time**.
STM's content model lets a topic be worded differently across a group (the SAGE
content covariate); ECTM lets that group difference **drift across discrete time
periods**, so you can ask not only "how do these groups word this topic
differently" but "how has that difference *changed*".

Where [`DTM`](#dtm) and [`DETM`](embedding.md) already let a topic's *single*
vocabulary drift over time, ECTM's novelty is the **group-by-time interaction**:
it models how the *difference between groups* in a topic's wording evolves, in the
logistic-normal/variational frame that also carries STM's prevalence covariates.
So the contrast with keyATM below is specifically about keyATM's *dynamic* model,
which holds each topic's word distribution fixed and drifts only prevalence. It fits one topic-word
distribution per (group, period) cell, with a first-order random-walk prior tying
adjacent periods so sparse cells borrow strength from their temporal neighbours
rather than fragmenting the topic.

```python
m = topica.ECTM(num_topics=10, seed=42)
m.fit(docs, times=year, content=party,   # times → periods, content → groups
      period_smooth=5.0, interaction_shrink=2.0)

m.content_word_dist("Republican", 2016)  # topic-word β for one (group, period)
```

The content model is `η_kgtv = m_v + κT_k + κKP_kt + κKG_kg + κKGP_kgt`: a topic
baseline, a shared temporal trajectory, an average group deviation, and the
group-by-time deviation — the changing lexical contrast. `period_smooth` is the
random-walk precision over periods (larger pools adjacent periods more);
`interaction_shrink` pulls the group-by-time term toward zero unless the data earn
it. A prevalence design (`prevalence=`) is optional and behaves as in STM.

ECTM answers two complementary questions, and exposes both. The **content** half
(which words a group uses, and how that drifts) is the topic-word model above; the
**prevalence** half (how *often* each group discusses a topic) is the standard
logistic-normal regression on a `prevalence=` design. Because ECTM is
logistic-normal, the prevalence side comes with method-of-composition standard
errors: pass `prevalence=party*spline(year)` and use
`topica.stm.predicted_prevalence` / `topica.estimate_effect` for attention
trajectories with confidence bands, exactly as for STM. `prevalence_by_group(m,
groups, periods)` gives the quick descriptive version. The two halves can tell
different stories -- on the platforms the parties devote *similar attention* to
the environment while their *language* diverges sharply, a contrast a
prevalence-only model would miss.

The `topica.ectm` helpers read the content result on the word-probability scale:
`content_words(m, topic, group, period)` (top words for a cell),
`content_contrast(m, topic, a, b, period)` (words distinguishing two groups),
`content_trajectory(m, topic, word, contrast=(a, b))` (a word's contrast across
periods), and `content_divergence(m, topic, a, b)` (total-variation distance
between the groups each period).

Four worked examples ship with topica. The two partisan showcases model how
**Democrats and Republicans** word the same topics and how that gap evolves:
`examples/ectm_platforms.py` (national party platforms, 1948-2024, 20 elections)
finds that inside a stable "environment" topic the word `climate` enters the
Democratic vocabulary after 2000 while Republicans never adopt it, `conservation`
fades, and the partisan gap widens — evolution a fixed-vocabulary dynamic model
cannot represent. `examples/ectm_speeches.py` (congressional speeches with
Voteview party, 1948-2008) recovers the documented rise of partisan language: the
mean Democrat-Republican vocabulary distance is flat through the 1950s-80s and
rises in the mid-1990s. The platforms corpus ships in-repo; the speeches corpus is
built by `python examples/prep_speeches.py`.

Two smaller examples round out the set: `examples/ectm_poliblog.py` (Conservative
vs Liberal blogs across the 2008 campaign — a single year, so a stable gap) and
`examples/ectm_congress.py` (the keyATM Congressional-bills corpus, House vs
Senate across 14 Congresses; run `Rscript examples/prep_congress.R` first).

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
learned structure. Fit by parallel variational EM.

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
shift topic propensity.

```python
import numpy as np
X, names = topica.one_hot(party)
model = topica.DMR(num_topics=20, seed=1)
model.fit(docs, X, feature_names=names)
```

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

`gamma` is the main lever on the inferred count: larger values discover more
topics (the conservative default `0.1` lands near a handful, like the reference
implementations). By default the concentrations are held fixed, which gives a
stable, reproducible topic count; `resample_conc=True` lets the model adapt them
to the data instead, useful for exploration but more liberal about adding topics.

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

## NMF

Non-negative matrix factorization ([Lee & Seung 2001](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization)) factors the document-term matrix `X` (D x V, non-negative) as `X ≈ W H` with both factors non-negative, then reads each row of `H` as a topic's word distribution and each row of `W` as a document's topic mixture (both normalized to sum to 1). It is the fast, deterministic baseline familiar from scikit-learn: no sampling and no priors, just multiplicative updates that descend a reconstruction loss.

```python
m = topica.NMF(num_topics=20, seed=1)
theta = m.fit_transform(docs)
m.top_words(10)
```

`beta_loss` selects the divergence: `"frobenius"` (default, the squared error `½‖X − WH‖²`) or `"kullback-leibler"` (the generalized-KL loss, equivalent to pLSA on counts). `init` selects the start: `"nndsvd"` (default, a deterministic NNDSVDa initialization seeded by a from-scratch randomized truncated SVD) or `"random"` (seeded). `weighting` builds `X` from raw counts (default) or topica's own TF-IDF. The Rust core is BLAS-free: the dense products are rayon-parallel and the document-term products exploit `X`'s sparsity, so fits are bit-identical regardless of thread count.

Validated against `sklearn.decomposition.NMF` in `parity/nmf_vs_sklearn.py`. On a planted-block corpus topica matches sklearn to aligned topic-word cosine 1.000 for both divergences. On the political-blog corpus (poliblog5k, 5,000 documents) topica reproduces sklearn's topics at K=10 (aligned cosine 0.999, both divergences); at larger K, where the NMF objective is multimodal, topica reaches an equal-quality alternate optimum (reconstruction loss within about 0.1% of sklearn, sometimes lower) rather than sklearn's exact factorization, as expected for a non-convex problem whose solutions are not unique. On speed, the KL path runs several times faster than sklearn at scale, and the Frobenius path is competitive on the sparse document-term matrices typical of text, with the gap to BLAS-backed sklearn appearing only on near-dense inputs.

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

## Short-text models

`PT` and `GSDMM` are built for short documents; see the
[short-text guide](short-text.md).

## SupervisedLDA

Topics shaped to predict a per-document real-valued response (Blei & McAuliffe).
`coefficients` give each topic's pull on the outcome, and `predict` scores new
documents.

## LabeledLDA

Supervised: each label is a topic, and a document's tokens are restricted to its
labels. Empty labels fall back to unconstrained LDA.

## SAGE

Content-covariate topics via an additive log-linear model: the *same* topic is
worded differently across groups. `word_contrast(topic, a, b)` shows the words
that most distinguish two groups' phrasing.

## Hierarchy models

`PA` (Pachinko Allocation) and `HLDA` (hierarchical, nested-CRP) recover
super-/sub-topic structure.
