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
topica.label_topics(model.topic_word, corpus=corpus, n=10)     # stm-faithful: lift + FREX James-Stein shrinkage from corpus word counts
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
labels = topica.llm_topic_labels(model, texts, backend=my_model_fn, set_labels=True)

# Or name a model via the `llm` adapter (pip install "topica[llm]"):
backend = topica.llm_backend("gpt-4o-mini", temperature=0)   # pin for stability
labels = topica.llm_topic_labels(model, texts, backend=backend, set_labels=True)

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
exposes these diagnostics under the **`topica.llm`** namespace — an `llm-bounded`
family kept distinct from the bit-exact diagnostics above. All reuse the
provider-agnostic `topica[llm]` backend.

```python
# A capable open-source model, via OpenRouter or a local endpoint:
backend = topica.llm.backend("openrouter/meta-llama/llama-3.3-70b-instruct", temperature=0)

topica.llm.coherence(model, backend=backend, n_words=10)        # per-topic 1-3 rating (the headline)
topica.llm.intrusion(model, backend=backend, n_words=5)         # LLM picks the intruder -> accuracy
topica.llm.select_k(models, docs, backend=backend, n_docs=10)   # number-of-topics by doc-label purity
```

`llm.coherence` is the one to lead with: in the paper it beats NPMI/c_v at tracking
human topic *rankings* (and on the Hoyle 2021 gold, `parity/llm_coherence_compare.py`
reproduces that here). `llm.intrusion` matches human accuracy on the *task* but is a
weaker ranking signal, so report it alongside, not instead.

`llm.select_k` chooses the number of topics: for each candidate model it labels each
topic's top documents with the LLM and scores by **label purity** (the fraction of a
topic's documents sharing the majority label), returning the model with the highest
mean purity. This is the paper's *working* number-of-topics signal — doc-label purity
tracks ground-truth cluster quality, where rating the top *words* across `k` does not
— and complements `search_k`'s coherence, exclusivity, held-out, and dispersion criteria.

### A multi-dimensional suite (Tan & D'Souza 2025)

Coherence rating answers one question — *are these words related?* — but a topic can
be coherent and still be redundant, indistinct from its neighbours, or a poor fit for
the documents it claims. Tan & D'Souza (2025) widen the lens to four dimensions, all
exposed under the same namespace and `backend=`:

```python
topica.llm.outlier(model, backend=backend, n_samples=5, threshold=3)  # which words break a topic (unsupervised vote)
topica.llm.repetitiveness(model, backend=backend)                     # is coherence just redundancy? rate + duplicate pairs
topica.llm.diversity(model, backend=backend)                          # pairwise cross-topic distinctiveness (1-3)
topica.llm.alignment(model, docs, backend=backend)                    # per topic: irrelevant words / missing themes vs its top docs
topica.llm.adversarial(model, backend=backend)                        # gold-free capability self-check
```

`llm.outlier` is the unsupervised sibling of `llm.intrusion`: no planted answer, just a
5-runs vote on *which* top words don't belong (kept when flagged in `threshold` of
`n_samples` runs), so it surfaces the specific words making a topic incoherent.
`llm.repetitiveness` checks the failure coherence rating misses — a topic of near-synonyms
scores high on relatedness but is uninformative; it returns a 1-3 rate (3 = distinctive)
plus the duplicate word pairs. `llm.diversity` rates every topic *pair* for thematic
overlap, the LLM analog of `topic_diversity`. `llm.alignment` is the only one that reads
the corpus: per topic it asks, over the topic's top documents, how many topic words are
irrelevant (overrepresentation) and how many document themes are missing
(underrepresentation).

`llm.adversarial` is the one to run first. It plants a known-unrelated word
(`"shakespeare"`) into each topic and measures how often `llm.outlier` catches it — a
**gold-free** check that validates both the metric and your model's capability on *your*
corpus, no human labels required. A detection rate near 1.0 means the model is strong
enough for the rest of the suite; a low rate is the signal to size up before trusting any
of these numbers.

!!! note "Model capability matters — don't use a tiny model"
    These tasks need a **capable** model, and open weights are enough: in our checks a
    70B-class open model (Llama-3.3-70B) handles all three, and `llm.coherence`
    reproduces the paper's human correlation with Qwen3-235B. The tasks differ in
    difficulty — *rating* (`llm.coherence`) is forgiving and an 8B model ranks topics
    sensibly, but *intrusion* (`llm.intrusion`) and *labeling* (`llm.select_k`) are
    harder: an 8B model failed to spot obvious word intruders in our tests. Prefer a
    ~70B+ open model (or a strong hosted one); treat small-model results, especially
    on intrusion/labeling, with suspicion.

!!! warning "These are `llm-bounded`, not bit-exact"
    Unlike the rest of topica's diagnostics, these call an external model and are
    **not** reproducible bit-for-bit. Use `temperature=0` (or `n_samples>1`, which
    calls the model repeatedly and aggregates by mean/majority-vote) for stability,
    and read the result as a measurement with model-dependent noise. The paper's
    prompts are kept verbatim in the overridable `topica.llm.PROMPTS` dict. Cost
    is O(K) LLM calls; pass a cheap model.

## Stability and model selection

```python
topica.search_k(docs, ks=[10, 20, 30], held_out=test)     # coherence, exclusivity, held-out, dispersion per K
topica.bootstrap_stability(docs, k=20, n_boot=50)         # per-topic stability under resampling
topica.align_topics(model_a, model_b)                     # one-to-one match across fits
topica.topic_stability([model_a, model_b], topn=10)       # cross-fit term overlap
topica.check_residuals(model, docs)                       # Taddy dispersion: is K too small?
```

`search_k` returns a `SearchKResult` whose rows carry `coherence`, `exclusivity`,
residual `dispersion`, and (with `held_out=`) a held-out metric per K, plus
optional `criteria=("deveaud", "cao_juan")` columns. Fit several seeds per K with
`num_seeds>1` to get a `<metric>_se` standard error on each (parallelize the fits
with `n_jobs=-1`). Then let `result.best_k(metric=..., rule=...)` name a K:
`rule="best"` (the optimum), `"1se"` (the simplest K within one standard error, so
you don't over-read noise), or `"elbow"` (the diminishing-returns knee of a
held-out curve). The default `best_k()` picks the coherence/exclusivity *frontier*
rather than a single monotone metric. The [Choose and justify K](../publishing/choosing-k.md)
guide works a full example on `poliblog`.

### Topic alignment

To compare topics across different runs, seeds, or even architectures, `topica.align_topics(model_a, model_b)` performs Kuhn-Munkres (Hungarian) matching to align topics one-to-one. It returns a custom `AlignmentResult` object containing matched tuples of `(topic_a, topic_b, distance)`.

It supports several distance metrics:
- `metric="cosine"` (default): Cosine distance.
- `metric="js"`: Jensen-Shannon distance.
- `metric="rbo"`: Rank-Biased Overlap over the top `depth` words, focusing weight on high-probability words.
- `metric="emd"` (or `"ot"`): Earth Mover's Distance / Optimal Transport, which can use a word embeddings dictionary or matrix.

If the models have different vocabularies, `align_topics` automatically intersects them, projects the distributions, and re-normalizes them.

You can inspect relationship classifications (matches, splits, merges, and unaligned topics):
```python
result = topica.align_topics(model_a, model_b, metric="cosine", threshold=0.3)

result.matches     # Hungarian 1-to-1 pairs whose similarity clears `threshold`
result.splits      # topic in A splitting to multiple in B (overlay on the matches)
result.merges      # topic in B merging from multiple in A (overlay on the matches)
result.unaligned_a # topics in A with no match above threshold
result.unaligned_b # topics in B with no match above threshold
```

`threshold` sets the one-to-one match cut. Splits and merges are an *overlay*: an extra
partner is flagged only when it is close to a topic's own best match *relative to this
fit's cross-topic similarity floor*, so a matched topic that also has a close extra
partner shows up in both `matches` and `splits`/`merges`. Because the overlay calibrates
to each fit, correlated-topic families (STM/CTM) — whose off-diagonal cosines are high —
are no longer mislabelled as near-total splits/merges, and `align_topics(tw, tw)` returns
K matches with zero splits/merges for any model (issue #642).

## Topic structure and document outliers

Three post-hoc, no-refit diagnostics that read a fitted model's `topic_word` and
`doc_topic` (so they work on any model — LDA, STM, DMR, CTM, keyATM):

```python
# Is K=20 really a few super-themes, and are any topics near-duplicates?
dnd = topica.topic_dendrogram(model, metric="js")     # needs scipy
dnd.cut(6)                                             # group label per topic at 6 super-topics
dnd.groups(6, n=10)                                    # {group: (member topics, merged top words)}
dnd.merge_candidates()                                 # near-duplicate pairs (relative threshold)
dnd.linkage                                            # SciPy linkage matrix for plotting

# Are these topics real, or did I forget to clean my corpus?
rows = topica.flag_topics(model, docs)                 # per-topic quality + a junk flag
junk = [r for r in rows if r["junk"]]                  # reasons: stopword-soup / dead-tiny / incoherent+flat

# Which documents does the model fail to explain?
res = topica.document_residuals(model, docs)           # per-doc novelty, most anomalous first
res[:10]                                               # off-topic, repetitive, or anomalous docs
```

`topic_dendrogram` is the flat-model counterpart to `HLDA` (which fits a topic
tree directly) and to `ensemble` (which merges across runs): it merges *one*
fitted model's topics by distribution distance. Use a relative `merge_candidates`
threshold — the absolute distance scale shifts with how much common-word mass the
corpus shares.

`flag_topics` scores coherence, exclusivity, topic-word flatness, prevalence, and
top-word stopword fraction, then flags junk *relative to the run*. The cleanest
signal it catches is a forgotten stopword pass, where boilerplate topics light up
as `stopword-soup`.

`document_residuals` reconstructs each document as `theta_d @ beta` and ranks how
poorly that matches the actual words. It complements `check_residuals` (one
corpus-level "is K too small?" number) by pointing at the *specific* documents
the model misses. The headline `novelty` score folds in out-of-vocabulary mass so
off-topic intruders surface; `cross_entropy` is the length-robust in-vocabulary
component (use it, not `kl`, which is length-confounded).

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
cons.agreement              # scalar: mean stability, "how reproducible is this K?"
topica.coherence(cons, docs)
```

`agreement` and `stability` are point estimates. Pass `n_boot>0` to bootstrap
them — the runs are resampled with replacement, the consensus recomputed, and the
result gains `agreement_ci` / `agreement_se` and (for `cluster`/`align`) a
per-topic `stability_ci`. That tells you whether a difference in `agreement`
across K, or across model families, is real or just noise from which runs you
happened to combine:

```python
cons = topica.ensemble(runs, n_boot=300, boot_seed=0)
cons.agreement, cons.agreement_ci   # e.g. 0.79, (0.74, 0.83)
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
- `method="stable"` derives from gensim's `EnsembleLda` (Brigl 2019). It does not
  fix K: it finds dense, reproducible "cores" with Checkback DBSCAN and keeps only
  the clusters with enough cores as stable topics, discarding the rest as noise.
  Use it to let the data decide how many topics are reproducible. On well-separated
  inputs it matches gensim to floating-point precision; it improves on gensim in two
  edge cases where gensim degenerates (small-vocabulary rank masking and
  scan-order-dependent core validation).

```python
topica.ensemble(runs, method="align")                  # reference matching
topica.ensemble(runs, method="stable", eps=0.1)        # discover stable topics
```

### Cross-model consensus ensembling

While `topica.ensemble` is designed to combine independent runs (from different seeds) of the same model class, you can use `topica.cross_ensemble` to combine and align topics across entirely different architectures (e.g. combining LDA, STM, and BERTopic).

This is particularly valuable for proving that your target topics are robust, persisting regardless of whether they are recovered by a Gibbs sampler, variational EM, or neural clustering.

If the models have different vocabularies (due to different preprocessing options), `cross_ensemble` automatically intersects them, projects the models' `topic_word` matrices onto the common vocabulary intersection, and re-normalizes them. If the models have different numbers of topics, it automatically defaults to the median K of the input models.

```python
# Combine different architectures fit on the same corpus
cons = topica.cross_ensemble([lda_model, stm_model, bertopic_model])
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

### Has the chain plateaued, or mixed?

For the collapsed-Gibbs samplers, `convergence_tol` watches the log-likelihood
trace, and a flat trace means the sampler found a mode — not that the chain has
*mixed*. A plateaued log-likelihood and a poorly-mixed chain look identical from
the objective alone. `topica.mcmc` reports the MCMC-native diagnostics a Bayesian
workflow expects, computed from traces the model already keeps: the
log-likelihood history and the thinned `theta_draws`.

```python
model = topica.LDA(num_topics=20, seed=1)
model.fit(docs, iters=2000, num_theta_draws=200)   # more retained draws -> finer ESS

d = topica.mcmc_diagnostics(model)
print(d.summary())
# MCMC diagnostics for LDA (inference=gibbs)
#   retained draws          : 200
#   log-likelihood tau      : 3.10
#   log-likelihood ESS      : 6.5
#   theta ESS (min/median)  : 41.2 / 118.7 (of 200 draws)

d.theta_ess          # (num_docs, num_topics) effective sample size per element
d.loglik_autocorr    # autocorrelation of the log-likelihood trace
```

A low `theta ESS` relative to `retained draws` means the chain is autocorrelated
— the draws carry less information than their count suggests, so run more sweeps
or thin further. The `theta_draws` are already thinned, so raise
`num_theta_draws` on `fit` for a finer estimate.

The underlying estimators are also exposed directly for any trace you hold —
`topica.autocorrelation`, `topica.integrated_autocorr_time` (Geyer's
initial-positive-sequence `tau`), and `topica.effective_sample_size` (`N / tau`,
for one chain or columnwise over a `(draws, params)` matrix).

The variational models (STM, CTM, …) converge a bound and have no MCMC chain —
`mcmc_diagnostics` warns if you point it at one.

### Do independent chains agree? (R-hat)

A single chain can plateau, look well-mixed, and still have settled into a mode
the sampler happened to reach from its seed. The Gelman-Rubin R-hat answers the
question one chain cannot: fit the same model at several seeds and check whether
the chains converged to a *common* distribution. R-hat compares the variance
between chains to the variance within each — near 1 they agree, above ~1.01 they
have not mixed.

```python
chains = []
for seed in (1, 2, 3, 4):
    m = topica.LDA(num_topics=20, seed=seed)
    m.fit(docs, iters=2000, num_theta_draws=200)
    chains.append(m)

d = topica.multichain_diagnostics(chains)
print(d.summary())
# Multi-chain diagnostics for LDA (4 chains, inference=gibbs)
#   log-likelihood R-hat    : 1.008 (ESS 640, n=1000)
#   topic-prevalence R-hat  : max 1.021 / median 1.004 over 20 aligned topics
#   topic alignment (Jaccard): min 0.71 (low -> that topic's R-hat is not comparable)
#   -> chains mixed

d.loglik_rhat        # R-hat of the log-likelihood trace (permutation-invariant)
d.topic_rhat         # (num_topics,) per-topic R-hat of aligned topic prevalence
d.topic_alignment    # (num_topics,) how well each topic matched across chains
d.converged          # every reported R-hat <= 1.01
```

Two views are reported. The **log-likelihood R-hat** is the headline: the
log-likelihood is permutation-invariant, so it compares chains directly with no
alignment. The **per-topic R-hat** is finer but needs care — topic 3 in one chain
need not be topic 3 in another, so `multichain_diagnostics` first aligns the
topics across chains (a Hungarian match on the topic-word matrix, the same
machinery [`align_topics`](../api/diagnostics.md#topica.align_topics) uses) and then compares each
aligned topic's per-draw prevalence. Read `topic_rhat` next to `topic_alignment`:
a topic with a low alignment Jaccard did not line up across chains, so its R-hat
is comparing different topics and means nothing.

The R-hat estimator itself — rank-normalized split-R-hat (Vehtari et al. 2021) —
is exposed directly as `topica.rhat(chains)` for any set of chains you hold, in
the same spirit as the single-chain primitives above.

## Visualization

```python
viz = topica.prepare_pyldavis(model, docs)                # pyLDAvis PreparedData if installed
qf, fig = topica.quality_frontier(model, plot=True)       # matplotlib scatter if installed
```
