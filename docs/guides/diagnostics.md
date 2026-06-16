# Diagnostics & validation

All of these are **model-agnostic**: they take any fitted model's `topic_word` /
`doc_topic`, so they work the same across LDA, STM, HDP, and the rest. They're
exported at the top level (`topica.<name>`) and in `topica.diagnostics`. For how
to *use* them to make an analysis publishable, see
[Validate the topics](../publishing/validation.md).

## Quality metrics

```python
import topica

model.coherence(10)                                   # per-topic UMass (built in)
topica.coherence(model, texts, coherence_type="c_v")      # windowed, human-aligned
topica.exclusivity(model, n=10)                           # per topic
topica.topic_diversity(model, topn=25)                    # fraction of unique top words
topica.topic_semantic_diversity(model, topn=25)           # fraction of unique top-word *pairs*

qf = topica.quality_frontier(model, n=10)                 # coherence, exclusivity, prevalence
# qf["coherence"], qf["exclusivity"] -> the canonical STM quality scatter
```

`topic_diversity` counts unique single words; `topic_semantic_diversity` counts
unique word *pairs* (Wu, Nguyen & Luu 2024). We reach for the pair version when
single-word overlap understates redundancy: two topics can share few exact words
yet co-locate the same word pairs, and a pair pins down word sense without any
embeddings. Both range over `[0, 1]`, and higher means more diverse.

!!! tip "Coherence is fast, even at large K"
    `topica.coherence` runs its co-occurrence counting in the Rust core, scoring only
    the word pairs that actually occur within a topic's top-N rather than a full
    vocabulary×vocabulary matrix. `c_v` on a 500-topic model that took minutes in
    a pure-Python loop now takes a fraction of a second. Two habits still help on
    very large corpora: compute coherence **once** on the final model (never
    inside a fit loop), and pass a **document sample** as `texts` — coherence is
    an estimate, and a few thousand documents give the same ranking. `u_mass`
    (document-level, no sliding window) remains the cheapest option for quick
    `K`-selection sweeps.

## Labeling and interpretation

```python
topica.label_topics(model.topic_word, model.vocabulary, n=10)   # prob / frex / lift / score
topica.frex(model.topic_word, model.vocabulary, n=10)           # frequent + exclusive
topica.relevance(model.topic_word, model.vocabulary, lam=0.6)   # LDAvis relevance
topica.find_thoughts(model.doc_topic, texts, topic=0, n=3)      # representative docs
topica.find_thoughts_html(model, texts, n_docs=3)               # highlighted close-reading
```

For readable labels, `llm_topic_labels` asks an LLM to name each topic from its
top words and representative documents. topica is the plumbing: it assembles the
prompt and you bring the model. Pass any callable (your own client, a local
`ollama` endpoint) as `call`, or name a model through the optional
[`llm`](https://llm.datasette.io/) adapter, which reaches every provider and
local models via plugins.

```python
# Bring your own callable (no extra dependency):
labels = topica.llm_topic_labels(model, texts, call=my_model_fn, set_labels=True)

# Or name a model via the `llm` adapter (pip install "topica[llm]"):
backend = topica.llm_backend("gpt-4o-mini", temperature=0)   # pin for stability
labels = topica.llm_topic_labels(model, texts, call=backend, set_labels=True)

topica.topic_label_prompts(model, texts)[0]   # inspect exactly what the model sees
```

`set_labels=True` flows the labels into `topic_info` and `plot_report`. LLM labels
are a convenience, not a reproducible measurement: pin the model and temperature,
and keep `label_topics` (FREX / probability / lift) as the defensible descriptors.

## Human validation: intrusion tests

```python
topica.word_intrusion(model, n_words=5, seed=0)           # top words + an intruder
topica.document_intrusion(model, texts=texts, n_docs=3)   # top docs + an intruder
```

## LLM-based evaluation

Automated coherence (NPMI, c_v) correlates only weakly with human judgment. Stammbach
et al. (2023) show that an LLM, prompted with the *same* instructions the crowd-workers
received, tracks human ratings more closely — especially the rating task. topica
exposes two such diagnostics; both reuse the provider-agnostic `topica[llm]` backend.

```python
# An open-source model, via OpenRouter or a fully local ollama endpoint:
call = topica.llm_backend("openrouter/meta-llama/llama-3.1-8b-instruct", temperature=0)
# call = topica.llm_backend("ollama/llama3.1")                # fully local, no API key

topica.llm_coherence(model, call=call, n_words=10)         # per-topic 1-3 rating (the headline)
topica.llm_intrusion(model, call=call, n_words=5)          # LLM picks the intruder -> accuracy
topica.llm_select_k(models, docs, call=call, n_docs=10)    # number-of-topics by doc-label purity
```

`llm_coherence` is the one to lead with: in the paper it beats NPMI/c_v at tracking
human topic *rankings* (and on the Hoyle 2021 gold, `parity/llm_coherence_compare.py`
reproduces that here). `llm_intrusion` matches human accuracy on the *task* but is a
weaker ranking signal, so report it alongside, not instead.

`llm_select_k` chooses the number of topics: for each candidate model it labels each
topic's top documents with the LLM and scores by **label purity** (the fraction of a
topic's documents sharing the majority label), returning the model with the highest
mean purity. This is the paper's *working* number-of-topics signal — doc-label purity
tracks ground-truth cluster quality, where rating the top *words* across `k` does not
— and complements `search_k`'s coherence/exclusivity/perplexity criteria.

Small **open** LLMs (Llama-3.1-8B, Mistral-7B, Qwen2.5-14B) are good enough for these
(Tan & D'Souza 2025); a cheap or local open model keeps the whole evaluation
reproducible and key-free.

!!! warning "These are `llm-bounded`, not bit-exact"
    Unlike the rest of topica's diagnostics, these call an external model and are
    **not** reproducible bit-for-bit. Use `temperature=0` (or `n_samples>1`, which
    calls the model repeatedly and aggregates by mean/majority-vote) for stability,
    and read the result as a measurement with model-dependent noise. The paper's
    prompts are kept verbatim in the overridable `topica.LLM_EVAL_PROMPTS` dict. Cost
    is O(K) LLM calls; pass a cheap model.

## Stability and model selection

```python
topica.search_k(docs, ks=[10, 20, 30], held_out=test)     # coherence/exclusivity/perplexity per K
topica.bootstrap_stability(docs, k=20, n_boot=50)         # per-topic stability under resampling
topica.align_topics(model_a, model_b)                     # one-to-one match across fits
topica.topic_stability([model_a, model_b], topn=10)       # cross-fit term overlap
topica.check_residuals(model, docs)                       # Taddy dispersion: is K too small?
```

## Ensemble: combining runs

A single fit is one draw from a noisy procedure. Change the seed and the topics
move, sometimes a lot, and neural models are worse than classical ones (Hoyle et
al. 2022). Rather than fit once and hope, or fit many and pick one with
`select_model`, we can combine independent runs into a consensus that is more
reliable than any single run. In Hoyle et al.'s experiments the ensemble beats the
median run in 97% of settings and never loses to the worst.

`ensemble` takes the runs (a list of fitted models, raw topic-word arrays, or a
`select_model` result) and returns a consensus that behaves like a fitted model:
it carries `topic_word`, `doc_topic`, and `vocabulary`, so it flows straight into
`coherence`, the diagnostics, and the rest. Each consensus topic reports a
`stability` score and a `reliable` flag, so a topic the runs do not actually agree
on is marked rather than trusted.

```python
runs = topica.select_model(docs, K=20, runs=10)   # ten initializations
cons = topica.ensemble(runs)                       # combine them

cons.topic_word.shape       # (20, V)
cons.stability              # per-topic agreement across runs, in [0, 1]
cons.reliable               # per-topic: consistent AND well-supported?
topica.coherence(cons, docs)
```

Three methods are available:

- `method="cluster"` (default) reproduces Hoyle et al. (§6): pool the topics from
  every run, measure a top-weighted rank distance between them that blends the
  topic-word and document-topic views (`lambda_`), cluster the pool into K groups,
  and average within each cluster. Clustering tolerates a topic that splits or
  merges across runs, and flags a cluster that few runs supported.
- `method="align"` is a lighter, fully deterministic alternative: match every
  run's topics one-to-one to a reference run (Hungarian on the topic-word
  distributions) and average the aligned topics.
- `method="stable"` reimplements gensim's `EnsembleLda` (Brigl 2019). It does not
  fix K: it finds dense, reproducible "cores" with Checkback DBSCAN and keeps only
  the clusters with enough cores as stable topics, discarding the rest as noise.
  Use it to let the data decide how many topics are reproducible. It is validated
  against gensim to floating-point precision.

```python
topica.ensemble(runs, method="align")                  # reference matching
topica.ensemble(runs, method="stable", eps=0.1)        # discover stable topics
```

## Convergence

Every iterative model exposes a uniform convergence interface. `model.fit_history`
is a list of `(iteration, objective)` pairs — the ELBO/bound for variational
models (STM, CTM, ProdLDA, ETM, FASTopic) and the per-token log-likelihood for
collapsed-Gibbs models (LDA, keyATM, SeededLDA, …). `model.converged` is `True`
if a tolerance criterion was met during `fit`, `False` if the model ran to the
iteration cap, and `None` for models with no iterative objective (BERTopic,
Top2Vec).

```python
model = topica.LDA(num_topics=20, seed=1)
model.fit(docs, iters=500)

model.converged        # True / False / None
model.fit_history      # [(10, -7.43), (20, -7.31), ...]
```

On collapsed-Gibbs models you can enable early stopping by passing
`convergence_tol` and `check_every` to `fit`:

```python
model.fit(docs, iters=1000, convergence_tol=1e-4, check_every=10)
# stops as soon as the relative change in log-likelihood over one check
# interval drops below 1e-4, rather than running all 1000 sweeps.
```

keyATM takes `convergence_tol` the same way, but its check cadence is the
`report_interval` it already uses for the `model_fit` trace (not a separate
`check_every`).

### Defaults, and why

The defaults follow each family's reference implementation rather than a tuned
guess:

- **Variational EM (STM, CTM, STS)** stop automatically when the relative change
  in the variational bound falls below `em_tol`, default `1e-5` — the same
  criterion and value as R `stm`'s `emtol` ([Roberts, Stewart & Tingley
  2019](https://doi.org/10.18637/jss.v091.i02)).
- **Collapsed-Gibbs samplers (LDA, keyATM, DMR, SeededLDA, …)** default to
  `convergence_tol=0.0` (no early stop): a fixed number of sweeps is the field
  convention, following MALLET and Griffiths & Steyvers (2004), and keeps the
  retained θ-draw thinning (`thin = iters / num_theta_draws`) well defined.
  Setting `convergence_tol > 0` opts into log-likelihood-plateau early stopping
  without changing the default fit.

The cluster models (BERTopic, Top2Vec) and structurally non-iterative models
(DTM, HLDA) return an empty `fit_history` and `converged` of `False` or `None`;
they satisfy the contract without early-stop support. HDP and GSDMM record a
`fit_history` but never early-stop (`converged` stays `False`): they *discover*
their topic and cluster counts, so a log-likelihood plateau is not a convergence
signal.

## Visualization

```python
viz = topica.prepare_pyldavis(model, docs)                # pyLDAvis PreparedData if installed
qf, fig = topica.quality_frontier(model, plot=True)       # matplotlib scatter if installed
```
