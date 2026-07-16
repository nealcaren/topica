# 2. Choose and justify K

!!! quote "The principle"
    **K is a research decision, not a tuning parameter.** Multiple values of `K`
    are often defensible; your job is to pick one for a reason and show your
    conclusions don't hinge on it.

`K` sets the *granularity* of your themes: roughly, `K=10` for broad themes,
`K=30` for specific topics, `K=100` for fine distinctions. There is no single
"correct" `K`, and you should resist any procedure that pretends otherwise.

## Three converging justifications

Good practice combines all three:

**1. Theory-driven.** How many themes would you expect in this corpus? What level
of granularity answers *your* research question? Start from theory and adjust.

**2. Diagnostic-guided.** Scan a range and look at quality metrics:

| Metric | What it measures | Reading |
|--------|------------------|---------|
| Coherence (c_v / UMass) | Do a topic's top words co-occur? | Higher is better |
| Exclusivity | Are words distinctive to a topic? | Higher is better |
| Held-out perplexity | Fit on unseen documents | Lower is better |

!!! warning "Do not just maximize coherence"
    A model with `K=5` may have higher mean coherence yet miss distinctions that
    matter to your argument. Coherence trades off against exclusivity and against
    substantive richness. Use the metrics to *inform* a judgment, not to replace
    it.

**3. Interpretability-focused.** For each candidate `K`: can you label every
topic? Do the topics make substantive sense? How many are "junk" (stopwords,
artifacts)? Do topics split and merge sensibly as `K` grows?

## A concrete procedure

```python
import topica
import numpy as np

# 1) Scan a theoretically plausible range.
held_out = test_docs                     # a held-out split for perplexity
results = topica.select.search_k(
    train_docs, ks=[10, 15, 20, 25, 30],
    held_out=held_out, iters=800,
)
for r in results:
    print(f"K={r['k']:>3}  coherence={r['coherence']:.3f}  "
          f"exclusivity={r['exclusivity']:.3f}  perplexity={r.get('perplexity'):.0f}")
```

Then, for the two or three best candidates, fit the model and **read the
topics**. Count how many you can label, look at the per-topic
coherence×exclusivity spread, and check held-out perplexity directly:

```python
model = topica.models.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X)

table = topica.diagnostics.diagnostics(model, texts)          # one row per topic: coherence,
                                                   # exclusivity, FREX, size, ...
pp = topica.diagnostics.perplexity(model, held_out)            # held-out, lower is better

frontier = topica.diagnostics.quality_frontier(model, n=10)   # per-topic coherence & exclusivity
# scatter frontier["coherence"] vs frontier["exclusivity"];
# weak topics cluster in the lower-left.
```

`topica.diagnostics.perplexity(model, held_out)` works across the generative models (LDA,
DMR, CTM, STM, HDP, …) by inferring each held-out document's topic mixture from
half its tokens and scoring the other half, so it is comparable across `K`.

### Document-completion held-out log-likelihood

`make_heldout` and `eval_heldout` implement R `stm`'s held-out word scoring
rather than the standard perplexity split. We hold out a random fraction of
words from a random fraction of documents, fit the model on the reduced corpus,
then score the withheld words:

```python
import topica

h = topica.diagnostics.make_heldout(corpus, prop_docs=0.5, prop_words=0.5, seed=0)
model = topica.models.STM(num_topics=20, seed=1)
model.fit(h.documents, prevalence=X)

result = topica.diagnostics.eval_heldout(model, h)
print(f"mean per-doc held-out log-likelihood: {result.mean_per_doc_loglik:.3f}")
```

Higher (less negative) values indicate better fit. This metric is comparable
across values of K fit on the same `h.documents` corpus.

### Best-of-N at fixed K

Gibbs models and STM can land on different local optima from different starting
values. `select_model` runs `runs` initializations at a fixed K and returns all
fitted models with their coherence and exclusivity scores:

```python
result = topica.select.select_model(
    docs, K=20,
    runs=20,           # number of random initializations
    model="stm",       # "lda" or "stm"
    prevalence=X,      # required when model="stm"
    fraction=0.5,      # keep only the top 50% after a short burn-in
)
# inspect the coherence-exclusivity frontier across all runs:
topica.viz.plot_models(result)

# pick the run in the upper-right corner and use that model:
best_idx = result.coherence.argmax()   # or use exclusivity, or visual inspection
model = result.models[best_idx]
```

The `fraction` argument mirrors R `stm`'s "run briefly, keep the best ~20%"
heuristic: a short burn-in filters out clearly poor starts before the full
training runs.

A nonparametric model is a useful sanity check on your choice: it *infers* a
topic count rather than taking one.

```python
hdp = topica.models.HDP(eta=0.3, seed=1)
hdp.fit(docs, iters=300)
print("HDP suggests ~", hdp.num_topics, "topics")
```

## Report sensitivity

Pick the `K` that balances metrics, interpretability, and theory, then **show
your finding survives nearby `K`**. Re-run the headline result at `K-5` and
`K+5`; if a covariate effect or a key topic only appears at one exact `K`, say
so. Reviewers read "we used K=20" charitably only when followed by "results were
robust to K ∈ {15, 25}."

→ Next: [Validate the topics](validation.md).
