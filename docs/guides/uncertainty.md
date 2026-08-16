# Reporting uncertainty

Topic estimates are estimates. The proportions `θ`, the top words, the covariate
effects you put in a table all carry sampling error, and a credible result
reports it. `topica.effects.standard_errors` is one entry point for the quantities people
publish, and it propagates the topic-estimation uncertainty rather than treating
the fit as if it were observed exactly.

```python
import topica

se = topica.effects.standard_errors(model, corpus, of="effect", formula="~ year", data=meta)
```

There are two distinct uncertainties, and they call for two methods.

## Method of composition (the default)

Given a fixed fit, `θ` is still uncertain: each document's topic proportions are
estimated from its tokens. The **method of composition** (Treier & Jackman 2008,
the procedure R `stm` uses) draws `θ` from the model's own posterior, computes the
quantity on each draw, and pools by Rubin's rules, so the standard errors inflate
over a naive regression on point `θ`. It is cheap (no refit) and sound for
covariate effects and group prevalence.

`standard_errors` detects the model family and picks the right sampler for you.
STM and CTM draw from their logistic-normal variational posterior. The Gibbs
models (LDA, keyATM, SeededLDA) retain thinned MCMC `θ` draws during the fit by
default (`model.theta_draws`, shape `(num_draws, num_docs, num_topics)`), so the
standard errors propagate real topic-estimation uncertainty: how much each
document's mixture moves across sweeps, which grows when topics overlap and
shrinks when the model is confident. No `Corpus` is needed in this case.

If you fit with `keep_theta_draws=False` (to save the
`num_draws x num_docs x num_topics` of f32 storage), the Gibbs path falls back to
a per-document Dirichlet conditional built from the token counts. A fitted model
retains its own per-document lengths (`model.doc_lengths`), so this still needs no
`Corpus`; pass one only to draw against a different corpus on purpose. That
approximation captures within-document sampling noise (it scales with `1 / N_d`)
but is blind to whether the topics are actually identified, so its intervals can
be wider than the real posterior for short documents and narrower for long ones.

```python
# Covariate effects, uncertainty propagated (one regression per topic):
eff = topica.effects.standard_errors(model, corpus, of="effect",
                             formula="~ party", data=meta, nsims=50)
for e in eff:
    print(e.topic, e.as_dict())

# Each topic's mean prevalence, with an interval:
prev = topica.effects.standard_errors(model, corpus, of="prevalence", nsims=50)
```

The effect result is the same `TopicEffect` list as `estimate_effect` (so
`cluster=`, `link=`, `spline`, and `interaction` all carry over); prevalence
returns a `TopicPrevalence` per topic. `estimate_effect` itself now also accepts
the model directly: `estimate_effect(model, X, corpus=corpus, nsims=50)` draws the
right posterior internally.

Composition covers `of="effect"` and `of="prevalence"` for LDA, keyATM, STM, and
CTM. It cannot give top-word uncertainty (it only varies `θ`), and it does not
apply to the embedding models (they have no posterior over `θ`). For those, use
the bootstrap.

## Bootstrap (top words, embedding models)

Refitting on resampled documents captures the *full* model uncertainty, including
which topics emerge. It is the only route to standard errors for the embedding
models and the only way to get top-word or topic-quality intervals.

```python
# How stable is each topic's top-word list?
tw = topica.effects.standard_errors(model, corpus, of="top_words", method="bootstrap",
                            n_boot=200, topn=10)
for t in tw:
    for word, prob, lo, hi in t.words:
        print(t.topic, word, f"kept in {prob:.0%} of resamples")
```

`of="prevalence"` and `of="effect"` also work with `method="bootstrap"` and report
the across-refit spread.

### The alignment catch

Each refit permutes and drifts the topics, so a quantity must be matched back to
the reference fit before pooling. topica matches by top-word overlap (Hungarian),
but a forced one-to-one match silently corrupts the standard error when a topic
**splits or merges** across resamples, or when the reference topics are not
distinct. This is exactly why R `stm` prefers the method of composition.

So the bootstrap reports two alignment diagnostics per topic and **suppresses the
standard error (sets it to `NaN`, `reliable=False`) when matching is unstable**:

- `alignment_quality` — mean top-word Jaccard with the matched topic (is the match
  *close*?).
- `alignment_margin` — how much better the match is than the next-best topic (is
  the match *unambiguous*?). A high quality with a low margin means the topics are
  interchangeable and the match is arbitrary, which the Jaccard alone would miss.

```python
for t in topica.effects.standard_errors(model, corpus, of="prevalence", method="bootstrap"):
    if not t.reliable:
        print(f"topic {t.topic}: unstable alignment "
              f"(quality={t.alignment_quality:.2f}, margin={t.alignment_margin:.2f}) "
              "— SE suppressed")
```

### Embedding models

The default bootstrap refits a count model on resampled token lists. An embedding
model's `fit` also needs the document embeddings, so pass a `refit` hook that
resamples them alongside the documents and returns a fitted model:

```python
def refit(doc_indices):
    m = topica.BERTopic(min_cluster_size=15, seed=1)
    m.fit([docs[i] for i in doc_indices], doc_emb[doc_indices])
    return m

tw = topica.effects.standard_errors(bertopic, corpus, of="top_words",
                            method="bootstrap", refit=refit, n_boot=200)
```

## Model selection and consensus

The same "report the error bar, not just the estimate" logic applies to picking K
and to combining runs. Both are opt-in.

- **Choosing K.** `search_k(..., num_seeds>1)` refits each K over several seeds, so
  every metric column comes with a `<metric>_se` standard error. `result.best_k(rule="1se")`
  then prefers the simplest K within one standard error of the optimum instead of
  chasing a noisy maximum. See [Choose and justify K](../publishing/choosing-k.md).
- **Combining runs.** `ensemble(runs, n_boot>0)` resamples the runs with
  replacement and recomputes the consensus, reporting `agreement_ci` and per-topic
  `stability_ci` — so you can say whether one K (or model family) is *really* more
  reproducible than another. See [Ensemble](diagnostics.md#ensemble-combining-runs).

## Which to use

| Quantity | Models | Method |
|---|---|---|
| Covariate effects | LDA, keyATM, STM, CTM | `composition` (default) |
| Group / overall prevalence | LDA, keyATM, STM, CTM | `composition` |
| Top words, topic quality | any | `bootstrap` |
| Anything, embedding models | BERTopic, Top2Vec, ETM, FASTopic | `bootstrap` (with `refit=`) |
| Choice of K | any | `search_k(num_seeds>1)` + `best_k(rule="1se")` |
| Consensus reproducibility | any | `ensemble(n_boot>0)` → `agreement_ci` |

Composition is cheaper and avoids the alignment problem, so prefer it where it
applies. Reach for the bootstrap when you need top-word intervals or you are on a
model with no posterior, and read the `reliable` flag before quoting a number.
