# Topica

`topica` is a fast topic-modeling library for Python with more than two dozen
models, built for social scientists who want to move from text data to
publishable results in a single workflow. It brings together models and tools
usually split across JVM software like MALLET and R packages like `stm`, and
runs them on a parallel Rust core competitive with the standard implementations,
with every fit reproducible from a fixed seed. Each model comes with the
validation, covariate-effect, and reporting tools to meet the standards
reviewers expect.

```bash
pip install topica
```

```python
import topica

docs = [["cat", "dog", "fish"]] * 15 + [["planet", "star", "moon"]] * 15
model = topica.models.LDA(num_topics=2, seed=42)
model.fit(docs, iters=1000)

for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", " ".join(w for w, _ in words))
```

## Why topica

- **One package, many models.** LDA, ProdLDA, DMR, Labeled LDA, SAGE, CTM, the full STM
  (prevalence **and** content covariates) and its STS sentiment-discourse extension,
  HDP, dynamic topics, supervised LDA,
  short-text models, and embedding-based models (BERTopic, Top2Vec, ETM,
  FASTopic). See [the models](guides/models.md) and
  [embedding topics](guides/embedding.md).
- **Built for social science.** Covariate effects with the method of
  composition, **clustered standard errors**, GLM links, Fighting Words,
  intrusion tests, bootstrap stability, and `searchK`: the things reviewers ask
  for. See [covariates](guides/covariates.md) and [diagnostics](guides/diagnostics.md).
- **Fast and deterministic.** A Rust core with bit-for-bit reproducible fits. The
  variational models parallelize across cores automatically.
- **No heavy dependencies.** A NumPy-only core. Optional extras add what you need
  — `topica[viz]` for plots, `topica[formula]` for the formula interface,
  `topica[polars]` for Polars, `topica[llm]` for LLM labels and embeddings — and
  PyTorch is never required. See [installation](getting-started/installation.md).

## The model families

| Model | What it's for |
|-------|---------------|
| [`LDA`](api/models.md#topica.models.LDA) | Classic topics via fast collapsed-Gibbs (SparseLDA) |
| [`ProdLDA`](api/models.md#topica.models.ProdLDA) | Sharper, more coherent topics via a product-of-experts word model (amortized VAE) |
| [`DMR`](api/models.md#topica.models.DMR) | Topics conditioned on document metadata |
| [`LabeledLDA`](api/models.md#topica.models.LabeledLDA) | Supervised topics tied to document labels |
| [`CTM`](api/models.md#topica.models.CTM) | Correlated topics (logistic-normal) |
| [`STM`](api/models.md#topica.models.STM) | Structural Topic Model: prevalence **and** content covariates |
| [`STS`](api/models.md#topica.models.STS) | Structural Topic and Sentiment-Discourse: covariate-driven topic sentiment on top of STM |
| [`SAGE`](api/models.md#topica.models.SAGE) | The same topic worded differently across groups |
| [`HDP`](api/models.md#topica.models.HDP) | Nonparametric LDA that *infers* the number of topics |
| [`DTM`](api/models.md#topica.models.DTM) | Dynamic topics that evolve across time slices |
| [`SupervisedLDA`](api/models.md#topica.models.SupervisedLDA) | Topics shaped to predict a per-document response |
| [`PT`](guides/short-text.md) / [`GSDMM`](guides/short-text.md) | Short-text models for tweets, survey answers |
| [`SeededLDA`](guides/guided.md) / [`KeyATM`](guides/guided.md) | Guided topics steered by seed words |
| `PA` / `HLDA` | Topic hierarchies (Pachinko, nested-CRP) |
| [`BERTopic`](guides/embedding.md) / [`Top2Vec`](guides/embedding.md) | Cluster document embeddings you supply into topics |
| [`ETM`](guides/embedding.md) / [`FASTopic`](guides/embedding.md) | Generative topics from embeddings (factored β; optimal transport) |

## Worked examples

End-to-end analyses on real, redistributable corpora:

- [**W.E.B. Du Bois in *The Crisis***](examples/dubois.md): 704 articles,
  1910–1934, the full workflow from preprocessing to dynamic topics.
- [**Gadarian immigration experiment**](examples/gadarian.md): the canonical STM
  vignette, reproduced.
- [**Political blogs**](examples/poliblog.md): STM with ideology and time
  covariates.
- [**Party platforms**](examples/ectm.md): how Democrats and Republicans word the
  same topics, and how that contrast has shifted across 20 elections (ECTM).
- [**LLM embeddings and labels**](examples/llm-topics.md): embedding-based topics
  with contextual document vectors and LLM-generated topic labels.

---

topica is open source on [GitHub](https://github.com/nealcaren/topica).
