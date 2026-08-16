# 3. Choose and justify K

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
| Held-out log-likelihood | Fit on withheld words | Higher (less negative) is better |
| Residual dispersion | Is K too small to absorb the variance? | `>> 1` means too small |
| Deveaud / Cao Juan (`criteria=`) | Are the topics distinct / non-redundant? | Deveaud higher, Cao Juan lower |

`search_k(num_seeds>1)` attaches a standard error to each metric, and
`res.best_k(metric, rule=...)` turns a curve into a pick: `"best"` (the optimum),
`"1se"` (the simplest K within one standard error), or `"elbow"` (the
diminishing-returns knee of a monotone metric like held-out likelihood).

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

# 1) Scan a theoretically plausible range. num_seeds>1 refits each K several
#    times so every metric carries a standard error; n_jobs parallelizes the fits.
held = topica.evaluate.make_heldout(topica.Corpus.from_documents(train_docs),
                           prop_docs=0.3, prop_words=0.5, seed=0)
results = topica.select.search_k(
    held.documents, ks=[10, 15, 20, 25, 30],
    held_out=held, num_seeds=4, n_jobs=-1, iters=800,
)
for r in results:
    print(f"K={r['k']:>3}  heldout={r['heldout_loglik']:.1f}±{r['heldout_loglik_se']:.1f}  "
          f"coherence={r['coherence']:.1f}  exclusivity={r['exclusivity']:.2f}  "
          f"dispersion={r['dispersion']:.2f}")

# 2) Let each lens name a K -- they encode different goals and often disagree.
results.best_k()                             # held-out when present, else frontier
results.best_k("heldout_loglik", rule="elbow")   # diminishing-returns knee
results.best_k("frontier")                   # coherence/exclusivity knee
results.best_k("heldout_loglik", rule="1se")     # simplest K within 1 SE (needs num_seeds>1)
```

`best_k` reports the K a criterion prefers; `dispersion` (Taddy residual
dispersion, `>> 1` means K is too small) is a diagnostic column, not a selection
target. See the worked example below for how to read a disagreement.

Then, for the two or three best candidates, fit the model and **read the
topics**. Count how many you can label, look at the per-topic
coherence×exclusivity spread, and check held-out perplexity directly:

```python
model = topica.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X)

table = topica.evaluate.diagnostics(model, texts)          # one row per topic: coherence,
                                                   # exclusivity, FREX, size, ...
pp = topica.evaluate.perplexity(model, held_out)            # held-out, lower is better

frontier = topica.select.quality_frontier(model, n=10)   # per-topic coherence & exclusivity
# scatter frontier["coherence"] vs frontier["exclusivity"];
# weak topics cluster in the lower-left.
```

`topica.evaluate.perplexity(model, held_out)` works across the generative models (LDA,
DMR, CTM, STM, HDP, …) by inferring each held-out document's topic mixture from
half its tokens and scoring the other half, so it is comparable across `K`.

### Document-completion held-out log-likelihood

`make_heldout` and `eval_heldout` implement R `stm`'s held-out word scoring
rather than the standard perplexity split. We hold out a random fraction of
words from a random fraction of documents, fit the model on the reduced corpus,
then score the withheld words:

```python
import topica

h = topica.evaluate.make_heldout(corpus, prop_docs=0.5, prop_words=0.5, seed=0)
model = topica.STM(num_topics=20, seed=1)
model.fit(h.documents, prevalence=X)

result = topica.evaluate.eval_heldout(model, h)
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
topica.select.plot_models(result)

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
hdp = topica.HDP(eta=0.3, seed=1)
hdp.fit(docs, iters=300)
print("HDP suggests ~", hdp.num_topics, "topics")
```

## A worked example: poliblog

The bundled `poliblog` corpus (2,000 political-blog posts from 2008, already
stemmed and stopworded) shows why no single number is "the" K, and how to read
`search_k` honestly. We scan a wide range, fit several seeds per K so every
metric carries a standard error, add a held-out set, and report the extra
`ldatuning`-style criteria alongside the defaults:

```python
import topica

docs = load_poliblog()                                  # list[list[str]], ~2000 docs
held = topica.evaluate.make_heldout(topica.Corpus.from_documents(docs),
                           prop_docs=0.3, prop_words=0.5, seed=0)

res = topica.select.search_k(
    held.documents, ks=[10, 20, 30, 40, 60, 80, 100, 120],
    held_out=held,
    num_seeds=4,                        # each K refit 4 times -> mean +/- SE
    n_jobs=-1,                          # fit the (K, seed) grid in parallel
    criteria=["deveaud", "cao_juan"],   # opt-in ldatuning criteria
    iters=400, seed=11,
)
```

The 32 fits (8 values of K by 4 seeds) finish in about 30 seconds on a laptop
because `n_jobs=-1` runs them across cores. Every metric column now holds the
across-seed mean with a `<metric>_se`:

| K | held-out | coherence | exclusivity | dispersion | deveaud | cao_juan |
|---|---|---|---|---|---|---|
| 10 | −711.8 ± 0.2 | −52.0 | 9.46 | 1.60 | 0.47 | 0.181 |
| 20 | −707.2 ± 0.3 | −60.7 | 9.72 | 1.47 | 0.56 | 0.096 |
| 30 | −704.9 ± 0.2 | −65.8 | 9.81 | 1.39 | 0.60 | 0.063 |
| 40 | −703.6 ± 0.3 | −67.0 | 9.85 | 1.34 | 0.62 | 0.047 |
| 60 | −702.2 ± 0.2 | −70.8 | 9.90 | 1.27 | 0.65 | 0.030 |
| 80 | −701.3 ± 0.3 | −75.4 | 9.92 | 1.22 | 0.66 | 0.020 |
| 100 | −700.9 ± 0.1 | −79.8 | 9.94 | 1.20 | 0.67 | 0.015 |
| 120 | −700.6 ± 0.2 | −83.8 | 9.95 | 1.18 | 0.67 | 0.011 |

Ask each lens for its pick and they disagree, which is the point:

```python
res.best_k("heldout_loglik")                    # 120  -- keeps climbing
res.best_k("heldout_loglik", rule="elbow")      # 40   -- the diminishing-returns knee
res.best_k("frontier")                          # 40   -- coherence/exclusivity knee
res.best_k("deveaud")                           # 120
res.best_k("cao_juan")                          # 120
res.best_k("coherence")                         # 10   (warns: UMass is monotone in K)
res.best_k()                                    # 120  -- defaults to held-out here
```

**Every single-axis metric runs to a grid extreme.** Held-out likelihood,
exclusivity, and both `ldatuning` criteria improve monotonically, so they point
at the largest K scanned; bare coherence falls monotonically, so it points at the
smallest. Selecting on any one of them just returns a grid endpoint, which is why
`best_k("coherence")` warns. The **frontier** is the only criterion that returns
an interior value, because it trades coherence against exclusivity:
`z(coherence) + z(exclusivity)` peaks at K=40.

!!! warning "The frontier is grid-relative"
    The frontier z-scores each metric *across the K values you scanned*, so its
    knee moves with the grid. On this corpus the same data returns K=10 for a
    `[5, 10, 15, 20, 25, 30]` grid and K=40 for `[10, 20, ..., 120]`. Read the
    whole curve; do not treat the single knee as an oracle.

**Read the held-out curve, not just its argmax.** Held-out likelihood is still
rising at K=120, so `best_k("heldout_loglik")` returns the grid ceiling. But the
gains per topic collapse: +7.0 from K=10 to 30, +2.6 from 30 to 60, +1.6 from 60
to 120. The `rule="elbow"` selector finds the diminishing-returns knee of that
curve, K=40, where extra topics stop buying much fit. That it agrees with the
frontier is reassuring: two different constructions land on the same interior K.

**Residual dispersion says you can go higher.** Taddy's residual dispersion falls
steadily (1.60 → 1.18) but stays well above 1 across the whole grid, evidence
that even K=120 leaves structure unmodeled. That is consistent with the political
science literature, which fits poliblog at K≈100 for fine-grained framing topics.
Dispersion is the one diagnostic here pointing toward more topics.

**Uncertainty is tight, so the 1-SE rule does not bite.** With 2,000 documents
the standard errors are small (held-out is −700.9 ± 0.1 at K=100), so every
metric is cleanly separated and `best_k(rule="1se")` returns the same K as
`rule="best"`. The one-standard-error rule earns its keep on smaller or noisier
corpora, where neighbouring K values sit within each other's error bars and the
simplest defensible model is worth preferring.

### What we would report

There is no contradiction to resolve, only a decision to make. For an
interpretable model of the main partisan themes, K≈30–40 is well supported: it is
the held-out elbow and the frontier knee, and the topics are readable. For
fine-grained framing analysis, dispersion and held-out both justify pushing
toward K≈100, at the cost of more topics to label. We would state the goal, name
the K, and (per the section below) show the headline result survives a few nearby
values.

## Embedding + cluster models (BERTopic, Top2Vec)

`search_k` scans LDA/STM/NMF/LSA directly and any other model through `fit=`, but
it **cannot scan embedding + cluster models** — `search_k(..., model="bertopic")`
raises, and passing a BERTopic through `fit=` makes no sense here. Two things
differ for these models:

- **Preprocessing does not change the topics.** Clusters are formed from the
  document *embeddings*, so stopword and frequency choices only affect the
  c-TF-IDF *labels* on already-formed clusters, not which documents cluster
  together. Removing stopwords leaves the topics identical. Preprocess for
  readable labels, but do not report token cleaning as shaping the model.
- **K is a clusterer setting, not a scan.** Use `num_clusters` for a fixed-K
  clusterer (`kmeans`, `gmm`, `agglomerative`) or `min_cluster_size` for HDBSCAN.
  There is no held-out likelihood.

There is no built-in sweep, so score the K knob yourself with the same
coherence-vs-diversity judgment used for the frontier above:

```python
import numpy as np, topica

b = topica.datasets.load_ng20_minilm()
docs = [t.split() for t in b.texts]

for k in [4, 5, 6, 8, 10]:
    m = topica.BERTopic(clusterer="kmeans", num_clusters=k,
                        reduce_frequent=True, seed=1).fit(docs, b.doc_embeddings)
    cv  = float(np.mean(topica.evaluate.coherence(m, docs, coherence_type="c_v")))
    div = topica.evaluate.topic_diversity(m, topn=25)
    print(f"K={k:>2}  c_v={cv:.3f}  diversity={div:.2f}  topics={m.num_topics}")
```

Pick the knee of the coherence/diversity trade-off, not the maximum of either.
With HDBSCAN, sweep `min_cluster_size` instead and watch how much of the corpus
falls into the `-1` noise bucket; a fixed-K clusterer with `reduce_frequent=True`
avoids that bucket and, on this dataset, recovers the newsgroups where the
HDBSCAN default collapses to a degenerate handful (most documents in one topic). The
[embedding-models guide](../guides/embedding.md) covers the clusterer choices and
the noise-bucket problem in depth.

## Fixed-K embedding models (EmbeddingLDA, ETM, FASTopic)

`EmbeddingLDA`, `ETM`, and `FASTopic` are embedding-driven but are **not**
clusterers: K is a model setting you fix in advance and every document gets a full
topic distribution, so unlike BERTopic/Top2Vec you sweep K the ordinary way, by
**refitting per K**. They have no built-in `model=` string, but you can drive them
through `search_k`'s `fit=` hook (closing over the embeddings/vocabulary they
need), or run the coherence-vs-diversity sweep by hand as below:

```python
import numpy as np, topica

topica.enable_experimental()   # EmbeddingLDA is experimental and gated
b = topica.datasets.load_ng20_minilm()
docs = [t.split() for t in b.texts]

for k in [4, 5, 6, 8, 10]:
    m = topica.embeddings.EmbeddingLDA(num_topics=k, embeddings=b.word_embeddings,
                            vocabulary=b.vocab, seed=1).fit(docs, iters=1000)
    coh = float(m.coherence(10).mean())     # per-topic UMass vector -> mean
    div = topica.evaluate.topic_diversity(m, topn=25)
    print(f"K={k:>2}  coherence={coh:.1f}  diversity={div:.2f}")
```

Pick the knee, not the maximum. Note the sign convention differs from the c_v sweep
above: `EmbeddingLDA.coherence` is per-topic UMass (more-negative is worse), so
average it and compare on that scale.

## Matrix-factorization models (NMF, LSA)

`NMF` and `LSA` factor the document-term matrix at a fixed rank K, so K is a model
setting you sweep by **refitting per K**. `search_k` scans them directly with
`model="nmf"` or `model="lsa"`, giving you the same coherence/exclusivity frontier
and `best_k` machinery as LDA. Both add a `reconstruction_error` column (NMF's fit
residual; for LSA the rank-K Frobenius residual `sqrt(‖X‖² − Σσ_k²)`) — read it like
a scree plot: it falls monotonically in K, so take the knee, not the minimum, and
cross it against coherence. Select on it directly with
`rows.best_k("reconstruction_error", rule="elbow")` (bare `rule="best"` returns the
grid edge and warns, because the error keeps shrinking with K). `rows.to_frame()`
gives the whole scan as a tidy DataFrame.

```python
import topica

df = topica.datasets.load_poliblog()      # a DataFrame; the text is already stemmed
corpus = topica.Corpus.from_documents([t.split() for t in df["text"]])

rows = topica.select.search_k(corpus, [10, 15, 20, 30, 40], model="nmf")
for r in rows:
    print(f"K={r['k']:>2}  coherence={r['coherence']:.1f}  "
          f"exclusivity={r['exclusivity']:.2f}  "
          f"reconstruction_error={r['reconstruction_error']:.1f}")
print("frontier K:", rows.best_k())
```

Any other model scans through the `fit=` hook — a callable `(k, seed) -> fitted
model` that closes over the corpus and any covariates or embeddings it needs, so
`search_k` never has to know the model's fit signature:

```python
rows = topica.select.search_k(corpus, [10, 20, 30],
                       fit=lambda k, s: topica.DMR(k, seed=s).fit(corpus, X))
```

Pick the K where coherence and exclusivity plateau and (for NMF) the reconstruction
error has passed its knee. Note the sign convention: coherence is UMass
(more-negative is worse).

For a **stability** check across seeds, refit with `init="random"` — the default
`init="nndsvd"` is deterministic and ignores `seed`, so `topic_stability` over
nndsvd fits would report a meaningless 1.0. Vary the seed under random init, or
resample the documents, to get an honest robustness read.

## Report sensitivity

Pick the `K` that balances metrics, interpretability, and theory, then **show
your finding survives nearby `K`**. Re-run the headline result at `K-5` and
`K+5`; if a covariate effect or a key topic only appears at one exact `K`, say
so. Reviewers read "we used K=20" charitably only when followed by "results were
robust to K ∈ {15, 25}."

→ Next: [Validate the topics](validation.md).
