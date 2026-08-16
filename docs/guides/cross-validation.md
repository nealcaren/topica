# Cross-validation

Cross-validation gives you shared, reproducible evidence about a model's
*predictive* behavior: refit a fresh model on each training fold, then score the
held-out documents it never saw. `topica.select.cross_validate` runs the whole loop and
records the fold assignments and every seed so a rerun reproduces the result.

It reports evidence; it does not adjudicate. Cross-validation here never picks `K`
for you or declares a topic substantively valid — those stay your decisions.

## The short version

`cross_validate` wants `docs` as a list of token lists (`list[list[str]]`). A bundled
dataset gives you a DataFrame whose `text` column is a space-joined string, so split it
first:

```python
import topica

df = topica.datasets.load_poliblog()      # a real bundled dataset
docs = [t.split() for t in df["text"]]    # -> list[list[str]]

result = topica.select.cross_validate(
    lambda seed: topica.LDA(10, seed=seed),   # a factory: seed -> fresh model
    docs,
    folds=5,
    seed=13,
    fit_kwargs={"iters": 500},
)

print(result.summary())
result.to_frame()   # per-fold metrics as a DataFrame
```

`docs` may also be a `Corpus` (build one with `topica.Corpus.from_documents(docs)`),
though for the leakage-free default you want raw token lists — see
[Vocabulary and comparability](#vocabulary-and-comparability).

The first argument is a **factory**, not a fitted model: `cross_validate` calls it
once per fold with a derived seed so every fold trains a fresh, reproducible model.
Thread the seed into the constructor — `lambda seed: topica.STM(10, seed=seed)`.

## What it reports

For each fold, on the **held-out** test documents:

- **Held-out document-completion perplexity** — half of each test document's tokens
  infer its topic mixture and the other half are scored under it. Lower is better.
  Because the scored tokens are withheld from the estimate, it does not fall
  trivially as `K` grows, so it is a fair quantity to compare across `K`.
- **Fold coherence and exclusivity** — computed on that fold's training text.
- **Cross-fold topic stability** — the mean matched-topic cosine across every pair
  of folds, aligned vocabulary-aware (fold vocabularies differ; see below).

`result.aggregate` gives the macro mean ± std of each metric across folds;
`result.per_fold` is the per-fold detail.

### Covariate-effect stability (keyATM, DMR, GDMR)

For a model that learns covariate coefficients (keyATM covariate, DMR, GDMR),
`cross_validate` also reports whether those effects hold up across folds, in
`result.covariate_stability`. After aligning topics across every fold pair
(vocabulary-aware, since fold vocabularies differ), it compares the fold-aligned
coefficients (λ) two ways:

- **sign-agreement rate** — the fraction of (topic, covariate) cells where two folds
  agree on the sign of the effect, and
- **magnitude correlation** — the Pearson correlation of the aligned coefficients.

Both headline numbers are **macro-averaged over covariates** (the mean of the
per-covariate statistics in `result.covariate_stability["per_feature"]`), not pooled
over raw coefficient cells: one Pearson correlation over pooled cells is dominated by
the highest-variance covariate and would hide instability in a smaller or null one. Get
the per-covariate table with `result.covariate_stability_frame()`.

A covariate model needs enough fit iterations before its coefficients mean anything: an
under-trained keyATM returns effects of ~0 for every topic, which is not a stable effect
but the *absence* of one. When the largest learned effect is ~0, `cross_validate` reports
`covariate_stability["effects_near_zero"] = True` and `summary()` prints
`covariate-effect stability: UNDEFINED` rather than a spurious sign-agreement of 1.0.
keyATM covariate effects typically need several hundred iterations to warm up
(`fit_kwargs={"iters": 800}` or more), well above the ~200 that suffices for the topic
assignments — check `covariate_stability["max_effect_magnitude"]` if in doubt.

A worked keyATM factory (the first argument is a keyword dict, so it does not match the
plain `topica.LDA(K, seed=s)` pattern):

```python
keywords = {"econ": ["market", "tax"], "social": ["family", "church"]}
X, names = topica.design.one_hot(df["rating"])   # (matrix, names) — unpack it
result = topica.select.cross_validate(
    lambda s: topica.KeyATM(keywords, num_topics=10, seed=s),
    docs,
    covariates={"covariates": X},         # D x F numeric matrix
    covariate_names=names,                # label the effect table/plot
    folds=5, fit_kwargs={"iters": 800},
)
print(result.covariate_stability_frame())
```

Pass `covariate_names=` (a flat list, or a dict keyed by the covariate kwarg) so the
per-covariate table and the plot show your column names instead of positional
`covariates[0]` / `covariates[1]` placeholders — keyATM does not preserve covariate
names on its own.

**Reading the numbers.** Sign-agreement near **0.5 is chance** (the folds disagree on
direction as often as not); values toward **1.0** mean the folds consistently recover the
same sign. Magnitude correlation near **0 (or negative)** means the folds do *not* recover
consistent effect *sizes*; toward **1.0** means they do. A genuinely null covariate will
often show ~0.5 sign-agreement and ~0 magnitude correlation — that is the diagnostic
correctly reporting "no stable effect," not a bug. A magnitude correlation of **`NaN`**
(shown as `n/a (no variance)` in `summary()` and the plot) means that covariate's learned
effect had no variance across the compared cells, so there is nothing to correlate — not
an error. The statistics are computed over few
cells (topics × fold pairs), so at small `K` or few folds treat them as directional, not
precise; `result.covariate_stability["topics_compared"]` reports how many topics each
pair actually compared, and `partial_alignment` flags when unaligned topics were dropped
(which can bias stability upward).

This is deliberately **not** a predictive-coverage statistic, and it is never stored in
a `coverage_*` field. λ has no per-document ground truth, and the conditional
`feature_effect_se` is under-dispersed, so a ±2·SE-overlap "stability" would read as
spuriously stable. The effects are learned per fold at *fit* time, so this diagnostic is
unaffected by the marginal held-out fallback that keyATM's perplexity is subject to (a
separate concern — see the conditioning flag above). Do not report this number as if it
were held-out predictive accuracy for the covariate.

## Supervised models (out-of-fold prediction)

For a supervised/measurement model with a numeric response — `SupervisedLDA` — pass
`y=` and `cross_validate` switches to the out-of-fold path: fit on each training fold,
predict the held-out response, assemble the out-of-fold (OOF) prediction vector, and
report regression error, calibration, and interval coverage.

```python
result = topica.select.cross_validate(
    lambda seed: topica.SupervisedLDA(10, seed=seed),
    docs,
    y=response,                 # per-document numeric response, length n_docs
    folds=5, seed=13,
    fit_kwargs={"iters": 25},
)
print(result.summary())
result.oof_predictions          # assembled OOF y-hat (NaN where a doc wasn't scored)
```

What you get:

- **Pooled RMSE / MAE / R²** over every out-of-fold prediction (`aggregate["rmse_pooled"]`,
  etc.), plus per-fold `*_macro` summaries (`{mean, std, n_valid_folds}`).
- **Interval coverage** (`coverage_90`, `coverage_95`): does a nominal 90% interval
  cover ~90% of held-out truths? The interval uses the model's predictive standard
  deviation from `predict(return_std=True)`, which propagates the document's topic
  uncertainty *plus* the residual σ². This is a **conditional Gaussian** interval built
  from the *training* σ², so out-of-fold coverage tends to sit a little **below**
  nominal — that is partly the approximation, not proof the model is miscalibrated.
- **Calibration**: `calibration_intercept` / `calibration_slope` (ideal 0 and 1) from
  regressing observed on predicted, plus a `result.calibration_table` reliability table.

The authoritative evaluated population is `result.scored_mask`. A test document that is
empty after per-fold vocabulary pruning, or that falls in a temporal initial-training
window, is **NaN** in `oof_predictions` and excluded from every metric — never scored as
a fabricated 0. Folds are independent, so `n_jobs>1` runs them on a thread pool
(deterministic; the result is identical to `n_jobs=1`).

Classification metrics are not offered: no model in the roster exposes `predict_proba`.
An unsupervised model passed with `y=` (or a model with no `predict`) raises a clear
error rather than fabricating a score.

## Fold strategies

`make_folds` (called for you, or usable directly) offers three ways to split, and it
is the single place leakage is guarded:

| `strategy` | Split | Requires | Guard |
|---|---|---|---|
| `"kfold"` | shuffle, K test blocks | — | no document in train and test of a fold |
| `"grouped"` | whole groups held out together | `groups=` | no group id straddles a fold (authors, blogs, threads) |
| `"temporal"` | ordered, test always after train | `times=` | `max(train) < min(test)`; tied timestamps stay together |

```python
folds = topica.select.make_folds(len(docs), strategy="grouped", folds=5, groups=author_id)
result = topica.select.cross_validate(factory, docs, folds=folds)
```

For `temporal`, the training window **expands** by default; pass `window=` for a
rolling window. The earliest documents form the initial training block and are never
tested, so they are marked `False` in `result.folds.oof_mask`.

## Covariates

Covariate models take per-document covariates aligned to the documents.
`cross_validate` sub-indexes them to each fold and conditions the held-out inference
on the fold's *test* covariates, so the evaluation is leakage-free. Pass them as a
dict keyed by the model's fit keyword:

```python
# Covariates must be a numeric matrix. Encode a categorical column first.
# one_hot returns (matrix, names) — unpack it; passing the tuple raises.
X, names = topica.design.one_hot(df["rating"])     # or topica.design.design_matrix(...)

result = topica.select.cross_validate(
    lambda seed: topica.STM(10, seed=seed),
    docs,
    covariates={"prevalence": X},           # D x P, aligned to docs
    folds=5,
)
```

Recognized out of the box: STM `prevalence` / `content` / `content_time`, DMR and
GDMR `features`, keyATM `covariates` / `times`. Each per-document array is
length-checked against the corpus up front — a length mismatch is a hard error, never
a silent truncation. Passing a covariate a model does not accept (e.g. `prevalence`
to a plain LDA) is also a hard error, not a silent drop.

**Watch the conditioning flag.** When a family's `transform` cannot take covariates
(keyATM), the held-out score falls back to the *marginal* path — it does **not**
condition on your covariate, so the reported perplexity does not reflect it.
`cross_validate` warns at run time when this happens, `result.summary()` prints a
`covariates:` status line, and every fold carries `covariate_conditioned` in
`result.to_frame()`. STM `prevalence` conditions correctly (`covariate_conditioned=True`);
do not report a keyATM-covariate CV as if it conditioned on the covariate. For keyATM
(and DMR/GDMR) the fold-stability of the learned effect itself is reported separately in
`result.covariate_stability` — see [Covariate-effect stability](#covariate-effect-stability-keyatm-dmr-gdmr).

For anything the named routing does not cover, supply both an
`fit_fn(train_docs, train_idx, seed_fold) -> model` and a
`score_fn(model, test_docs, test_idx, seed_fold) -> dict` and own the sub-indexing on
both sides; `cross_validate` still handles folds, seeds, aggregation, and the record.

## Vocabulary and comparability

By default (`vocab="per_fold"`) the vocabulary and its frequency pruning are learned
on **each training fold only** and the test documents are vectorized into that
vocabulary, with out-of-vocabulary tokens dropped. This is leakage-free: no test-set
word frequency ever informs a fold's feature selection.

The tradeoff is that per-fold vocabularies differ, so the probability space each
perplexity is measured over differs. `cross_validate` reports each fold's perplexity
and their **macro mean** (`perplexity: 1121 +/- 5`), but never a single *token-pooled*
perplexity that concatenates held-out tokens across folds — that would be a false
comparison across different vocabularies. The macro mean is still the right number to
**compare across models or across K at a fixed `seed`**: the same seed gives identical
folds, and within each fold the two models are scored on the identical held-out tokens,
so the comparison is valid. "Not pooled" refers only to the token-concatenation, not to
comparing the reported means.

If you would rather share one vocabulary across folds (comparable, poolable
perplexity) and accept that global pruning lets test frequencies inform feature
selection, pass `vocab="fixed"`. This is standard practice in much of the
topic-model literature; `cross_validate` emits a warning so the tradeoff is explicit.

For a genuinely per-fold vocabulary, pass **raw token lists**, not a pre-built
`Corpus`. A `Corpus` already had its vocabulary and frequency pruning learned on the
whole corpus, and rebuilding per fold cannot undo that; `cross_validate` warns when
you pass a pre-pruned `Corpus` on the `per_fold` path.

An unrecognized covariate key is a hard error, not a silent drop: passing
`covariates={"prevalence": X}` to a model that has no prevalence covariate (an LDA,
say) raises with the list of keys that model does accept.

## Plotting

`topica.viz.plot_cv(result)` turns a run into a figure, switching on the path:

```python
import topica.viz as viz

result = topica.select.cross_validate(lambda s: topica.LDA(10, seed=s), docs, folds=5)
viz.plot_cv(result).to_png("cv.pdf")   # or .to_frame() for the numbers
```

- **Topic path** — the per-fold distribution of each held-out metric (perplexity,
  coherence, exclusivity): each fold is a point, with the macro mean and ±1 std
  overlaid, so the fold-to-fold spread the aggregate hides is visible.
- **Supervised path** — the out-of-fold reliability plot (binned observed vs predicted
  with the 45° line, from `result.calibration_table`) beside the per-fold RMSE / R²
  spread.

When the run carries covariate-effect stability (keyATM / DMR / GDMR), the topic-path
figure adds a second band: per-covariate sign-agreement bars (with the 0.5 chance line)
and magnitude-correlation bars (centered at 0). `import topica.viz` is required before
`topica.viz.plot_cv` (a bare `import topica` does not pull in the submodule).

Like every `topica.viz` panel it also exposes `.to_frame()` (the per-fold table) and
`.to_png(path)` / `.to_pdf(path)`.

## Reproducibility

`seed` fixes both the fold shuffle and every per-fold fit seed, derived with
`numpy.random.SeedSequence` so they are stable across interpreter runs. With
`manifest=True` (the default) the returned `result.manifest` records the document
count and order hash, the exact fold indices, the splitter parameters, and every
derived seed, so a rerun reproduces the folds and fits. Save it alongside your
results:

```python
result.manifest.save("cv_manifest.json")
```

Bit-for-bit reproducibility is claimed only when you do not pass an opaque
`fit_fn`/`score_fn`; with a user callback the manifest records `replayable=False`.
