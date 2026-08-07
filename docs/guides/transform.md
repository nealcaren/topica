# Held-out inference

After fitting, `transform` infers topic proportions (θ) for **new, unseen**
documents while holding the fitted topics (φ) fixed: a held-out test set, or
freshly collected texts.

```python
import topica

model = topica.LDA(num_topics=20, seed=13)
model.fit(train_docs, iters=1000)

theta = model.transform(new_docs, seed=0)     # (len(new_docs), num_topics)
theta.argmax(axis=1)                          # dominant topic per new document
```

`transform` accepts a `list[list[str]]` or a `Corpus`. Out-of-vocabulary tokens
are dropped; a document with no in-vocabulary tokens gets the prior. Rows sum to
1, and results are deterministic for a fixed `seed`.

## Available across the model families

Each model uses the same inference it uses at fit time:

| Model | Inference |
|-------|-----------|
| `LDA`, `LabeledLDA`, `SupervisedLDA` | collapsed Gibbs against fixed φ |
| `HDP` | collapsed Gibbs over the discovered topics |
| `DMR` | collapsed Gibbs with `α_d = exp(Xγ)`; pass held-out `features` |
| `CTM`, `STM` | Laplace **variational** E-step against the logistic-normal prior |

For `CTM` / `STM`, the variational `transform` reproduces the model's own
training θ to ~`1e-3`. It is the same inference R's `stm` runs in
`fitNewDocuments`, not an approximation.

```python
# DMR with held-out covariates:
theta = dmr.transform(new_docs, features=new_X)

# STM variational inference (no Gibbs parameters needed):
theta = stm_model.transform(new_docs)
```

## Covariate-aware transform for STM

When new documents carry prevalence covariates, use `topica.stm.transform`
rather than the model method directly. It sets the per-document prior mean to
`mu_d = X_d @ gamma`, which is R `stm`'s `fitNewDocuments` behavior:

```python
import topica

# Via a raw covariate matrix (without the intercept column):
theta = topica.stm.transform(stm_model, new_docs, prevalence=X_new)

# Via a formula and a DataFrame of new-document covariates:
theta = topica.stm.transform(
    stm_model, new_docs,
    formula="~ party + author",
    data=new_meta_df,
)
```

The `formula=` path encodes the design matrix using the same column encoding
as at fit time, then prepends an intercept to match `gamma`. Formulas
containing a `spline()` term are rejected here because the knots would be
recomputed on the new documents rather than reused from the training data.
Build the design matrix manually with `design_matrix_predict` using the
fit-time knot context and pass it as `X=` instead.

## Aligning vocabulary before transform

New documents may contain tokens not seen at fit time. Use `align_corpus` to
drop out-of-vocabulary tokens before passing the documents to `transform`:

```python
aligned = topica.align_corpus(new_docs, stm_model)
theta = topica.stm.transform(stm_model, aligned, prevalence=X_new)
```

`align_corpus` works on any model that exposes a `vocabulary` attribute, so
it also applies before `stm_model.transform(new_docs)` or any other
transform call.
