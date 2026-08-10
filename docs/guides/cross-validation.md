# Cross-validation

Cross-validation gives you shared, reproducible evidence about a model's
*predictive* behavior: refit a fresh model on each training fold, then score the
held-out documents it never saw. `topica.cross_validate` runs the whole loop and
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

result = topica.cross_validate(
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

## Fold strategies

`make_folds` (called for you, or usable directly) offers three ways to split, and it
is the single place leakage is guarded:

| `strategy` | Split | Requires | Guard |
|---|---|---|---|
| `"kfold"` | shuffle, K test blocks | — | no document in train and test of a fold |
| `"grouped"` | whole groups held out together | `groups=` | no group id straddles a fold (authors, blogs, threads) |
| `"temporal"` | ordered, test always after train | `times=` | `max(train) < min(test)`; tied timestamps stay together |

```python
folds = topica.make_folds(len(docs), strategy="grouped", folds=5, groups=author_id)
result = topica.cross_validate(factory, docs, folds=folds)
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
# Covariates must be a numeric matrix. Encode a categorical column first:
X = topica.one_hot(df["rating"])            # or topica.design_matrix(...)

result = topica.cross_validate(
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
do not report a keyATM-covariate CV as if it conditioned on the covariate.

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
