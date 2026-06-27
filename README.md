# Topica: fast, all-purpose topic modeling for Python

[![PyPI](https://img.shields.io/pypi/v/topica.svg)](https://pypi.org/project/topica/)
[![CI](https://github.com/nealcaren/topica/actions/workflows/CI.yml/badge.svg)](https://github.com/nealcaren/topica/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://nealcaren.github.io/topica/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`topica` is a fast, all-purpose topic-modeling library for Python, built for computational social scientists who want to go from a column of text to publishable results in one workflow. It brings together models usually split across JVM tools like MALLET and R packages like `stm`, more than two dozen in all (LDA, STM, CTM, plus neural, dynamic, and embedding-based models), each paired with the validation, covariate-effect, and reporting tools reviewers expect. Where general toolkits like Gensim or BERTopic give you topics, topica is built around the question social scientists ask of them next: how topic prevalence and content relate to covariates, with reference-validated models and reproducible fits. It installs as a single wheel that needs only NumPy and pandas: no JVM, no PyTorch.

```bash
pip install topica
```

## Quick start

Point topica at a DataFrame and read the topics. This runs exactly as written, on a bundled example dataset, right after install:

```python
import topica

df = topica.datasets.load_gadarian()          # bundled; loads offline
corpus = topica.from_dataframe(
    df, text_col="open.ended.response", stopwords=topica.ENGLISH_STOPWORDS
)

model = topica.LDA(num_topics=5, seed=42)
model.fit(corpus)                             # sensible defaults; no tuning required
print(topica.summary(model))                  # top words per topic
```

`from_dataframe` keeps your metadata aligned to the documents that survive pruning, so the same corpus feeds a structural topic model that relates topic prevalence to a covariate, with an honest hypothesis test:

```python
prevalence = corpus.metadata[["treatment"]].to_numpy(float)

stm = topica.STM(num_topics=5, seed=42)
stm.fit(corpus, prevalence, prevalence_names=["treatment"])

draws  = topica.posterior_theta_samples(stm, nsims=30, seed=0)
effect = topica.estimate_effect(draws, prevalence, feature_names=["treatment"])
```

Your own data is one line away: pass `pandas.read_csv("yours.csv")` to `from_dataframe`. See the [getting-started guide](https://nealcaren.github.io/topica/getting-started/quickstart/) and the [worked examples](https://nealcaren.github.io/topica/examples/dubois/) for analyses end to end.

Fits are reproducible and validated: the variational models are identical to the bit, the samplers reproduce from a fixed seed and thread count, and every model is checked against its reference implementation (R `stm`, MALLET, keyATM, and more).

The core needs only NumPy and pandas. Optional extras add features without weighing it down: `topica[viz]` (matplotlib plots), `topica[formula]` (R-style formulas), `topica[polars]` (Polars frames), and `topica[llm]` (LLM labels and embeddings, OpenAI or local via ollama).

## Models

Starting out? **`LDA`** for general topics, **`STM`** to relate topics to
covariates, **`HDP`** to let the data choose the number of topics, and
**`BERTopic`** or **`CombinedTM`** for embedding-based topics. The full roster
follows.

<details>
<summary><b>All models</b> (more than two dozen, grouped by what you bring and what you want; click to expand)</summary>

Models are organized by **what you bring and what you want**, not by inference
family. The `from topica import X` namespace is flat; `topica.list_models(group=…,
brings=…, inference=…, determinism=…)` filters this roster in code. **Brings** is
what you supply beyond raw text; **Reproducibility** is `bit-exact` (identical
regardless of thread count), `seed-reproducible` (identical from a fixed seed and
thread count), or `llm-bounded`.

<!-- BEGIN MODEL TABLE (generated from topica.registry; edit registry.py, not this block) -->

### General-purpose

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `LDA` | text | gibbs | seed-reproducible | Classic latent Dirichlet allocation via a fast SparseLDA collapsed-Gibbs sampler. |
| `CTM` | text | variational | bit-exact | Correlated topic model: a logistic-normal prior that lets topics co-occur. |
| `ProdLDA` | text | vae | seed-reproducible | Product-of-experts LDA (AVITM) for sharper, more coherent topics; hand-coded VAE. |
| `HDP` | text | gibbs | seed-reproducible | Hierarchical Dirichlet process: infers the number of topics from the data. |
| `NMF` | text | matrix-factorization | bit-exact | Non-negative matrix factorization of the document-term matrix via multiplicative updates. |
| `LSA` | text | svd | seed-reproducible | Latent semantic analysis: a truncated SVD of the weighted document-term matrix. |

### Covariates & structure

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `STM` | text, metadata | variational | bit-exact | Structural topic model: relate topic prevalence and content to covariates. |
| `STS` | text, metadata | variational | bit-exact | Structural topic-and-sentiment model over document metadata. |
| `SAGE` | text, metadata | gibbs | seed-reproducible | Sparse additive generative model: the same topic worded differently across groups. |
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
| `DTM` | text, times | variational | seed-reproducible | Dynamic topic model: a fixed topic set whose word distributions drift across time slices. |
| `DETM` | text, embeddings, times | vae | seed-reproducible | Dynamic embedded topic model: embedding-factored topics that drift across time slices, fit as an amortized VAE. |
| `HLDA` | text | gibbs | seed-reproducible | Hierarchical LDA (nested CRP): a learned tree of super- and sub-topics. |
| `PA` | text | gibbs | seed-reproducible | Pachinko allocation: a DAG of super- and sub-topics. |

### Embedding-based

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `BERTopic` | text, embeddings | clustering | seed-reproducible | Cluster document embeddings; label topics by class-based TF-IDF. |
| `Top2Vec` | text, embeddings | clustering | seed-reproducible | Topics as dense regions in a joint document-word embedding space. |
| `ETM` | text, embeddings | variational | seed-reproducible | Embedded topic model: topic-word distributions factored through word embeddings. |
| `FASTopic` | text, embeddings | optimal-transport | seed-reproducible | Topics from optimal-transport plans between document, topic, and word embeddings. |
| `EmbeddingLDA` | text, embeddings, seeds | gibbs | seed-reproducible | Seeded LDA whose seed sets are expanded with nearest neighbors in an embedding space. |
| `CombinedTM` | text, embeddings | vae | seed-reproducible | Contextualized ProdLDA: encoder reads the bag of words plus a document embedding. |
| `ZeroShotTM` | text, embeddings | vae | seed-reproducible | Contextualized ProdLDA: encoder reads the document embedding alone, enabling cross-lingual transfer. |
| `InfoCTM` | text, dictionary | vae | seed-reproducible | Cross-lingual: two ProdLDA models aligned by a bilingual dictionary through a mutual-information term. |

### Ideal point

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `Wordfish` | text | em | bit-exact | Poisson scaling (Slapin & Proksch 2008): an unsupervised one-dimensional ideal-point estimate from word frequencies alone, no topics. The word-frequency baseline companion to IdealPointTM. |
| `TBIP` | text | variational | seed-reproducible | Text-Based Ideal Points (Vafa, Naidu & Blei 2020): a Poisson factorization whose neutral topic-word intensities are rescaled by a per-word ideological factor exp(x_s * eta_kv), with the author position x_s latent. Fit by the paper's mean-field variational inference (reparameterized SVI). Recovers ideological scales from unlabeled text. |
| `PartyEmbeddings` | text, metadata | neural-embedding | seed-reproducible | Party embeddings (Rheault & Cochrane 2020): a PV-DM paragraph-vector model trained by negative sampling with party-period metadata tags; the leading principal components of the learned party vectors give the ideological scale, and words share the space so a party's language can be read off by proximity. The corpus-trained word-embedding member of the ideal-point family. |

### LLM-based

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `TopicGPT` | text, llm | prompting | llm-bounded | LLM-driven topic discovery: prompt a model to propose, refine, and assign a topic taxonomy with descriptions. |

### Experimental

Shipped before a published paper and reference-implementation parity (topica's bar for a validated model). Gated: call `topica.enable_experimental()` (or set `TOPICA_EXPERIMENTAL=1`) before use. These may change or be removed without a deprecation cycle.

| Model | Brings | Inference | Reproducibility | Summary |
|---|---|---|---|---|
| `AnchorLDA` | text | matrix-factorization | bit-exact | Anchor-words spectral recovery (Arora et al. 2013): deterministic, Gibbs-free topics from the word co-occurrence matrix. |
| `ECTM` | text, metadata, times | variational | bit-exact | Evolving content topic model: STM content covariates that vary by group and drift across time periods. |
| `IdealPointTM` | text, embeddings | variational | seed-reproducible | Topic model with a latent ideal-point head: each author gets a low-dimensional position that shifts within-topic word choice, with a per-topic discrimination. Consumes word tokens as counts (Wordfish with topics) or, when word embeddings are supplied to fit, factored through them as in ETM. The unsupervised, latent-trait twin of the STM content covariate. |
| `IdealPointSentenceTM` | text, embeddings | em | seed-reproducible | Continuous ideal-point topic model over sentence/document embeddings: topics are Gaussian clusters whose centroids are displaced by a latent author position. The sentence-embedding sibling of IdealPointTM, fit by EM. |

<!-- END MODEL TABLE -->

</details>

Every model exposes the same shape: `fit(docs, …)`, then `topic_word` (φ), `doc_topic` (θ), `top_words(n)`, and `save`/`load`, so one diagnostic, labeling, and effect-estimation stack applies to all of them and a new model inherits it for free. The embedding-based models take document vectors from any embedder (sentence-transformers, an API, or a local model such as ollama; no PyTorch or UMAP/numba in the wheel). Full guides: [the models](https://nealcaren.github.io/topica/guides/models/) and [embedding topics](https://nealcaren.github.io/topica/guides/embedding/).

## Diagnostics & analysis

Model-agnostic: they work on any fitted model's `topic_word`/`doc_topic`:

- **Quality:** `coherence` (`u_mass`, `c_v`, `c_uci`, `c_npmi`; co-occurrence counting in the Rust core), `exclusivity`, `topic_diversity`, `quality_frontier`
- **Labeling:** `label_topics` (prob / FREX / lift / score), `frex`, `relevance`, `find_thoughts`, `topic_table`, `summary`
- **Validation:** `word_intrusion`, `document_intrusion`, `bootstrap_stability`, `search_k`
- **Reliability:** `select_model` (fit many seeds) and `ensemble` (combine runs into a consensus more reliable than any single fit — cluster/align/stable methods, the last a gensim `EnsembleLda` port)
- **Comparison:** `fighting_words` (weighted log-odds) for contrasting corpora
- **Covariate effects:** `estimate_effect` (method of composition, **cluster-robust SEs**, GLM links), `topic_correlation`, and the design helpers `one_hot`, `spline`, and `interaction` (all top level; they build covariate bases for any model's design matrix); `posterior_theta_samples` draws θ for the logistic-normal models (STM/CTM)
- **Preprocessing:** `tokenize`, `learn_phrases` / `apply_phrases`, `split_documents`, the `Corpus` class

See [diagnostics](https://nealcaren.github.io/topica/guides/diagnostics/) and [covariate effects](https://nealcaren.github.io/topica/guides/covariates/).

## Performance

topica runs on a parallel Rust core. It is several times faster than R `stm` — the single-threaded field standard — for the structural and other variational models, and it matches the hand-tuned compiled samplers core for core: parity with Java MALLET on plain LDA and with the C++ `keyATM` on keyword models. Fit to convergence (both at the same `emtol`, spectral start), on real corpora:

| Model | Reference | topica speedup (to convergence) |
|-------|-----------|----------------|
| STM | R `stm` | **1.7–2.7× single-threaded, ~5–7× multicore** |
| LDA | Java MALLET | parity single-threaded; multithread speedup **grows with corpus size** |
| keyATM | R `keyATM` | parity single-threaded, **~2×** multithreaded |

topica also fits in about a quarter of R `stm`'s memory (≈180MB against ≈675MB at 5,000 documents). For the approximate parallel Gibbs samplers the multithreaded speedup **grows with corpus size**: the per-sweep count-table merge is fixed overhead, so larger corpora amortize it over more sampling work. LDA's eight-core speedup over MALLET runs about 3× at 2,000 documents and reaches ~4× at 5,000.

Every fit is reproducible from a fixed seed and validated against its reference. See [Benchmarks](https://nealcaren.github.io/topica/benchmarks/) for the full methodology; reproduce the structural-model table with `python benchmarks/bench_stm_convergence.py` and the size-varying LDA curve with `python benchmarks/speed_vs_size.py`.

## Install from source

```bash
pip install maturin
git clone https://github.com/nealcaren/topica && cd topica
python -m venv .venv && source .venv/bin/activate
maturin develop --release --features python
```

Requires `numpy >= 1.21`. Use `--release` (the debug build is much slower).

## Acknowledgements

Topica stands on a generation of open topic-modeling research and code. Each entry below lists the reference, its authors and year, and the topica class(es) it underlies; the other models are Rust ports or reimplementations, validated against these reference implementations.

- [**MALLET**](https://github.com/mimno/Mallet) (McCallum, 2002) — `LDA`, `DMR`, `LabeledLDA`: the SparseLDA sampler, Dirichlet-multinomial regression, and hyperparameter optimization. `LDA` began as a port of David Mimno's [**RustMallet**](https://github.com/mimno/RustMallet) (Apache-2.0) and follows its SparseLDA sampler and fixed-point optimizer closely, but uses its own RNG (PCG), so it is not byte-identical to RustMallet. Against Java MALLET (also a different RNG) it recovers the same topics on a planted corpus (cosine 1.000)
- [**stm**](https://github.com/bstewart/stm) (Roberts, Stewart & Tingley, 2019) — `STM`, `CTM`, `SAGE`: variational EM, `estimateEffect`, `searchK`, FREX, spectral initialization, and the method of composition
- [**sts**](https://cran.r-project.org/package=sts) (Chen & Mankad, 2024) — `STS`: the Structural Topic and Sentiment-Discourse model — the joint prevalence/sentiment Laplace E-step and the Poisson topic-word M-step, validated against the package
- [**lda-c / ctm-c / dtm**](https://github.com/blei-lab) and [**hdp**](https://github.com/blei-lab/hdp) (Blei lab, 2006–2007) — `CTM`, `DTM`, `HDP`: the CTM, Dynamic Topic Model, and HDP samplers
- [**gensim**](https://github.com/piskvorky/gensim) (Řehůřek & Sojka, 2010) — `DTM`, `ensemble`: the coherence-pipeline conventions (the `coherence_type=` API and default sliding windows; the measures themselves are Röder et al. 2015 and Mimno et al. 2011), the `LdaSeqModel` DTM reference, and the `EnsembleLda` (CBDBSCAN stable-topic) method ported for `ensemble(method="stable")`
- [**tomotopy**](https://github.com/bab2min/tomotopy) (bab2min, 2020) — API conventions (`summary`, the short-text models), and `GDMR` (generalized DMR; Lee & Song, 2020), validated against its `GDMRModel`
- [**scikit-learn**](https://github.com/scikit-learn/scikit-learn) (Pedregosa et al., 2011) — `NMF`: the multiplicative-update solver (Lee & Seung, 2001) and the NNDSVD initialization (Boutsidis & Gallopoulos, 2008), validated against `sklearn.decomposition.NMF` (BSD-3-Clause); and `LSA`: latent semantic analysis / indexing (Deerwester et al., 1990), validated against `sklearn.decomposition.TruncatedSVD` (BSD-3-Clause) including its `svd_flip` sign convention. The numerics are reimplemented in Rust; the randomized truncated SVD shared by both (it seeds NMF's NNDSVD and is the LSA factorization itself) follows Halko et al. (2011).
- [**keyATM**](https://github.com/keyATM/keyATM) (Eshima, Imai & Sasaki, 2024) — `KeyATM`: the base, covariate, and dynamic models, the information-theory token weighting, and the Chib (1998) change-point HMM, validated against the package
- [**seededlda**](https://github.com/koheiw/seededlda) (Watanabe, 2023) — `SeededLDA`: the seeded-prior scheme
- [**LightLDA**](https://github.com/microsoft/LightLDA) (Yuan et al., 2015) — `LDA`: the alias-table Metropolis-Hastings sampler
- **GSDMM** (Yin & Wang, 2014) — `GSDMM`: the movie-group-process mixture for short text
- [**ProdLDA / AVITM**](https://arxiv.org/abs/1703.01488) (Srivastava & Sutton, 2017) — `ProdLDA`: autoencoding variational inference and the product-of-experts word model
- [**BERTopic**](https://github.com/MaartenGr/BERTopic) (Grootendorst, 2022) and [**Top2Vec**](https://github.com/ddangelov/Top2Vec) (Angelov, 2020) — `BERTopic`, `Top2Vec`: the embedding-clustering pipeline, class-based TF-IDF, and the `reduce → cluster → represent` design
- [**ETM**](https://github.com/adjidieng/ETM) (Dieng, Ruiz & Blei, 2020) — `ETM`: the Embedded Topic Model (per-document variational EM and an amortized VAE)
- [**DETM**](https://github.com/adjidieng/DETM) (Dieng, Ruiz & Blei, 2019) — `DETM`: the Dynamic Embedded Topic Model (structured amortized variational inference with a hand-coded LSTM)
- [**FASTopic**](https://github.com/BobXWu/FASTopic) (Wu et al., 2024) — `FASTopic`: the optimal-transport topic model
- [**contextualized-topic-models**](https://github.com/MilaNLProc/contextualized-topic-models) (Bianchi et al., MIT) — `CombinedTM` (Bianchi, Terragni & Hovy, 2021) and `ZeroShotTM` (Bianchi, Nozza & Hovy, 2021): ProdLDA encoders that read a contextual document embedding, alongside or in place of the bag of words
- [**quanteda.textmodels**](https://github.com/quanteda/quanteda.textmodels) (Benoit et al., 2018) — `Wordfish`: the Slapin & Proksch (2008) Poisson scaling model, validated against its `textmodel_wordfish` (the recovered scale and the analytic position standard errors both match at correlation 1.00 on a corpus sampled from the model)
- [**tbip**](https://github.com/keyonvafa/tbip) (Vafa, Naidu & Blei, 2020) — `TBIP`: Text-Based Ideal Points; the official implementation is TensorFlow 1.x, so topica reimplements the published model and its mean-field variational inference, validated against an independent PyTorch reference
- [**partyembed**](https://github.com/lrheault/partyembed) (Rheault & Cochrane, 2020) — `PartyEmbeddings`: party embeddings via a PV-DM paragraph-vector model with party-period metadata tags, placed by PCA of the learned party vectors. The reference builds on gensim's `Doc2Vec`; topica reimplements the PV-DM negative-sampling training in Rust (from Mikolov et al. 2013 and Le & Mikolov 2014) and is validated against that `Doc2Vec` scale (correlation 1.00 on a planted ordering)
- [**CLNTM**](https://arxiv.org/abs/2110.12764) (Nguyen & Luu, 2021) — the InfoNCE contrastive regularization on topic vectors offered by the `contrastive=` flag on the VAE models
- [**WHAI / Weibull-Dirichlet VAE**](https://arxiv.org/abs/1803.01328) (Zhang et al., 2018; Burkhardt & Kramer, 2019) — the Weibull-reparameterized Dirichlet prior offered by `prior="dirichlet"` on the VAE models
- [**Neural variational topic models with alternative priors**](https://arxiv.org/abs/1706.00359) (Miao, Grefenstette & Blunsom, 2017; Nalisnick & Smyth, 2017) — the Gaussian stick-breaking prior offered by `prior="stick_breaking"` on the VAE models
- [**TopicGPT**](https://github.com/chtmp223/topicGPT) (Pham et al., NAACL 2024, MIT) — `TopicGPT`: the generate / refine / assign prompt flow for LLM-driven topic discovery

The embedding-native models build on two pure-Rust crates: [**petal-clustering**](https://github.com/petabi/petal-clustering) for HDBSCAN and [**umap-rs**](https://github.com/wilsonzlin/umap-rs) for the optional UMAP reducer, both BLAS-free.

Full citations for every model and reference implementation, and how to cite topica, are on the [Citing](https://nealcaren.github.io/topica/citing/) page.

## License

Apache-2.0 — see [LICENSE](LICENSE).
