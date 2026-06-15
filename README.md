# Topica: fast, all-purpose topic modeling for Python

[![PyPI](https://img.shields.io/pypi/v/topica.svg)](https://pypi.org/project/topica/)
[![CI](https://github.com/nealcaren/topica/actions/workflows/CI.yml/badge.svg)](https://github.com/nealcaren/topica/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://nealcaren.github.io/topica/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`topica` is a fast topic-modeling library for Python with more than a dozen models, built for social scientists who want to move from text data to publishable results in a single workflow. It brings together models and tools usually split across JVM software like MALLET and R packages like `stm`, and runs them on a parallel Rust core competitive with the standard implementations, with reproducible fits: the variational models are identical to the bit, and the samplers reproduce from a fixed seed and thread count. Each model comes with the validation, covariate-effect, and reporting tools to meet the standards reviewers expect.

```bash
pip install topica
```

The core needs only NumPy. Optional extras add features without weighing the core
down: `topica[viz]` (matplotlib plots), `topica[formula]` (R-style formulas),
`topica[polars]` (Polars frames), and `topica[llm]` (LLM labels and embeddings,
OpenAI or local via ollama). PyTorch is never required.

```python
from topica import LDA

model = LDA(num_topics=2, seed=42)
model.fit([["cat", "dog", "fish"]] * 15 + [["planet", "star", "moon"]] * 15, iters=1000)

for i, words in enumerate(model.top_words(3)):
    print(f"Topic {i}:", " ".join(w for w, _ in words))
```

See the [getting-started guide](https://nealcaren.github.io/topica/getting-started/quickstart/) and the [worked examples](https://nealcaren.github.io/topica/examples/dubois/) for end-to-end analyses.

## Models

**Count-based models** learn topics from word counts (collapsed Gibbs, variational EM, or an amortized VAE):

| Model | What it's for |
|-------|---------------|
| **`LDA`** | Classic topics via fast collapsed-Gibbs (SparseLDA); optional multi-threaded and LightLDA alias samplers |
| **`STM`** | The Structural Topic Model: correlated topics with prevalence **and** content covariates |
| **`STS`** | Structural Topic **and Sentiment-Discourse**: covariate-driven topic sentiment/tone on top of STM |
| **`CTM`** | Correlated topics (logistic-normal) |
| **`DMR`** | Topics conditioned on document metadata (Dirichlet-multinomial regression) |
| **`GDMR`** | Generalized DMR over continuous metadata via a Legendre basis, with topic distribution functions across the metadata range |
| **`DTM`** | Dynamic topics that evolve across time slices |
| **`HDP`** | Nonparametric LDA that learns the number of topics from the data; by default fits with fixed concentrations (topic count steered by `gamma`), with optional concentration resampling |
| **`keyATM` / `seededlda`** | Guided topics steered by seed words |
| **`ProdLDA`** | Sharper, more coherent topics via a product-of-experts word model, fit as an amortized VAE (no PyTorch) |
| **`PT` / `GSDMM`** | Short-text models for tweets, survey answers, headlines |
| **`SupervisedLDA`** | Topics shaped to predict a per-document response |
| **`LabeledLDA`** | Supervised topics tied to document labels |
| **`SAGE`** | Content-covariate topics: the same topic worded differently across groups |
| **`PA` / `HLDA`** | Topic hierarchies (Pachinko, nested-CRP) |

**Embedding-based models** start from document embeddings you supply (no PyTorch, no UMAP/numba in the wheel):

| Model | What it's for |
|-------|---------------|
| **`BERTopic`** | Cluster document embeddings, label topics by class-TF-IDF; topic reduction and a soft per-document distribution |
| **`Top2Vec`** | Topics as points in the embedding space; topic words are the nearest word vectors |
| **`ETM`** | Embedded Topic Model: a logistic-normal topic model with the topic-word distribution factored through embeddings (`β = softmax(ρ·α)`); per-document EM or an amortized VAE (`inference="vae"`) |
| **`FASTopic`** | Topics read off two optimal-transport plans between document, topic, and word embeddings |

Every model exposes the same shape: `fit(docs, …)`, then `topic_word` (φ), `doc_topic` (θ), `top_words(n)`, and `save`/`load`. The count-based variational models (`CTM`/`STM`/`STS`/`SupervisedLDA`/`DTM`) parallelize across cores while staying bit-for-bit deterministic. The embedding models split into two kinds: `BERTopic` and `Top2Vec` run the `reduce → cluster → represent` pipeline, while `ETM` and `FASTopic` are generative and mixed-membership; all of them take vectors from any embedder (sentence-transformers, an API, a local model such as ollama). Full guides: [the models](https://nealcaren.github.io/topica/guides/models/) and [embedding topics](https://nealcaren.github.io/topica/guides/embedding/).

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

topica runs on a parallel Rust core. It is several times faster than R `stm` — the single-threaded field standard — for the structural and other variational models, and it matches the hand-tuned compiled samplers core for core: parity with Java MALLET on plain LDA and with the C++ `keyATM` on keyword models. On the political-blog corpus (2,000 documents, fit time only, same iterations on both sides):

| Model | Reference | topica speedup |
|-------|-----------|----------------|
| STM | R `stm` | **4–10× single-threaded, ~11–32× multithreaded** |
| LDA | Java MALLET | parity single-threaded; multithread speedup **grows with corpus size** |
| keyATM | R `keyATM` | parity single-threaded, **~2×** multithreaded |

For the approximate parallel Gibbs samplers the multithreaded speedup **grows with corpus size**: the per-sweep count-table merge is fixed overhead, so larger corpora amortize it over more sampling work. LDA's eight-core speedup over MALLET runs about 3× at 2,000 documents and reaches ~4× at 5,000, so the small-corpus figures above understate what large-corpus users see.

Every fit is reproducible from a fixed seed and validated against its reference. See [Benchmarks](https://nealcaren.github.io/topica/benchmarks/) for the full methodology; reproduce the 2,000-document table with `python benchmarks/speed_vs_r.py` and the size-varying curve with `python benchmarks/speed_vs_size.py`.

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
- [**keyATM**](https://github.com/keyATM/keyATM) (Eshima, Imai & Sasaki, 2024) — `KeyATM`: the base, covariate, and dynamic models, the information-theory token weighting, and the Chib (1998) change-point HMM, validated against the package
- [**seededlda**](https://github.com/koheiw/seededlda) (Watanabe, 2023) — `SeededLDA`: the seeded-prior scheme
- [**LightLDA**](https://github.com/microsoft/LightLDA) (Yuan et al., 2015) — `LDA`: the alias-table Metropolis-Hastings sampler
- **GSDMM** (Yin & Wang, 2014) — `GSDMM`: the movie-group-process mixture for short text
- [**ProdLDA / AVITM**](https://arxiv.org/abs/1703.01488) (Srivastava & Sutton, 2017) — `ProdLDA`: autoencoding variational inference and the product-of-experts word model
- [**BERTopic**](https://github.com/MaartenGr/BERTopic) (Grootendorst, 2022) and [**Top2Vec**](https://github.com/ddangelov/Top2Vec) (Angelov, 2020) — `BERTopic`, `Top2Vec`: the embedding-clustering pipeline, class-based TF-IDF, and the `reduce → cluster → represent` design
- [**ETM**](https://github.com/adjidieng/ETM) (Dieng, Ruiz & Blei, 2020) — `ETM`: the Embedded Topic Model (per-document variational EM and an amortized VAE)
- [**FASTopic**](https://github.com/BobXWu/FASTopic) (Wu et al., 2024) — `FASTopic`: the optimal-transport topic model

The embedding-native models build on two pure-Rust crates: [**petal-clustering**](https://github.com/petabi/petal-clustering) for HDBSCAN and [**umap-rs**](https://github.com/wilsonzlin/umap-rs) for the optional UMAP reducer, both BLAS-free.

Full citations for every model and reference implementation, and how to cite topica, are on the [Citing](https://nealcaren.github.io/topica/citing/) page.

## License

Apache-2.0 — see [LICENSE](LICENSE).
