# Estimator contract

This page specifies the uniform interface every topica estimator must expose.
Follow it when adding a new model or porting an existing one. The conformance
test (`tests/test_conformance.py`) checks every registered estimator against
this contract on every CI run, so gaps in a new model fail the suite rather
than silently degrading the toolkit.

## Constructor

```python
Model(num_topics, *, seed=13, **hyperparameters)
```

Every estimator takes `seed` in its constructor (not in `fit`). If the model
requires non-standard arguments (for example a seed dict for `KeyATM`, a depth
for `HLDA`, or super/sub counts for `PA`), those come first; `seed` is always a
keyword-only argument.

## The three tiers

### Tier 0 — every estimator (the floor)

Every estimator in the registry must pass every Tier 0 check.

**fit signature**

```python
def fit(self, data, ..., *, iters: int = <default>, ...) -> Self:
    ...
```

`iters` is the canonical iteration-count keyword. It controls the number of
Gibbs sweeps, EM steps, or epochs, depending on the model family. Use `iters`
even where the underlying algorithm calls them something else. The `validation`
module's `_accepts_kwarg` helper checks for this at import time; do not hide
`iters` behind `**kwargs` unless you explicitly check `'iters' in kwargs`.

**Properties and methods**

| Name | Shape / return type | Notes |
|------|---------------------|-------|
| `topic_word` | `(K, V)` float array | Each row sums to 1 for generative models; c-TF-IDF weights for cluster models |
| `doc_topic` | `(D, K)` float array | Each row sums to 1 for mixed-membership models |
| `vocabulary` | sequence of str, length V | Vocabulary aligned with `topic_word` columns |
| `num_topics` | int | Number of topics K |
| `topic_names` | list of str, length K | User-set labels; default `["topic_0", "topic_1", ...]` |
| `doc_names` | list of str, length D | Row labels for `doc_topic`; default document indices as strings |
| `top_words(n)` | list of K lists of `word` (or `(word, weight)` with `weights=True`) | Top n words per topic |
| `coherence(n)` | array of K floats | Per-topic UMass coherence over top n words |
| `save(path)` | — | Serialize to disk |
| `load(path)` | — | Class method; restore from disk |
| `fit_history` | list of `(int, float)` | Per-iteration `(iteration, objective)`: the ELBO/bound for variational models, the model log-likelihood for Gibbs models. Empty `[]` for models with no iterative objective |
| `converged` | bool or None | `True` if a tolerance criterion was met during fit, `False` if it ran to the iteration cap, `None` for models with no iterative objective |

The convergence interface is uniform: `model.fit_history` and `model.converged`
answer the same question on every model. The collapsed-Gibbs samplers record the
model log-likelihood every `check_every` sweeps and early-stop when
`convergence_tol > 0` (default `0.0` runs the full `iters` unchanged); the
variational models trace and early-stop on their ELBO. Models with no flat
per-iteration objective return an empty `fit_history` and `converged` is `False`
or `None`: the cluster models (BERTopic, Top2Vec, `converged` is `None`), the
time-sliced `DTM`, and the tree-structured `HLDA`.

### Tier 1 — generative models

Models whose `model_family` is `"dirichlet"` or `"logistic_normal"` must also
expose:

```python
def transform(self, docs) -> np.ndarray:  # shape (n_docs, K)
    ...
```

`transform` infers the topic mixture for new documents (without updating the
model). It is the basis for held-out perplexity, `eval_heldout`, and `search_k`.
Only the models with no flat document-topic representation are structurally
exempt: `HLDA` (a topic tree) and `DTM` (time-sliced). All five keyword,
seeded, and anchored Gibbs models (`KeyATM`, `SeededLDA`, `SAGE`, `PA`, `PT`)
implement `transform` by running collapsed Gibbs with the fitted phi held
fixed.

### Tier 2 — family-specific attributes

**Dirichlet family** (`model_family == "dirichlet"`, i.e. collapsed-Gibbs models):

| Name | Shape | Notes |
|------|-------|-------|
| `alpha` | `(K,)` float array | Per-topic document-topic Dirichlet prior |
| `theta_draws` | `(S, D, K)` float array | Retained MCMC draws of the doc-topic matrix; shape `(0,)` when not collected (pass `keep_theta_draws=True` to `fit`) |
| `doc_lengths` | `(D,)` int array | Tokens per document; used by the Dirichlet theta sampler in `composition_theta` |

**Logistic-normal family** (`model_family == "logistic_normal"`, i.e. STM/CTM):

| Name | Shape | Notes |
|------|-------|-------|
| `eta_mean` | `(D, K-1)` float array | Variational mean of the logistic-normal document representation |
| `eta_cov` | `(D, K-1, K-1)` float array | Variational covariance |

## Principled exemptions

Some requirements will never apply to certain models because their statistics
differ structurally. These exemptions are recorded in
`topica.conformance.EXEMPT`. Do not add a workaround to make a model fake-pass
a requirement it cannot genuinely satisfy; add it to `EXEMPT` instead with a
brief reason.

`topica.conformance.EXEMPT` and `KNOWN_GAPS` are the authoritative lists; the
table below is a snapshot. An exemption is permanent and principled; a gap is a
fixable omission tracked for a later phase.

Current permanent exemptions:

| Model | Requirement | Reason |
|-------|-------------|--------|
| `HLDA` | `doc_topic` | Topic tree; documents have paths, not a (D, K) theta matrix |
| `HLDA` | `num_topics` | Tree model with `num_nodes`; no fixed scalar K |
| `HLDA` | `doc_names` | No static per-document row index |
| `HLDA` | `coherence` | Node-indexed tree structure is incompatible with flat (K, V) coherence |
| `HLDA` | `transform` | No flat K-topic generative distribution |
| `DTM` | `doc_topic` | Time-sliced; no single static theta matrix |
| `DTM` | `doc_names` | No static per-document row index |
| `DTM` | `coherence` | Time-varying phi is incompatible with flat (K, V) coherence |
| `DTM` | `transform` | No time-slice-free held-out inference |
| `BERTopic` | `doc_names` | Exposes cluster `labels`, not a doc_names property |
| `BERTopic` | `iters` | Not an iterative sampler (UMAP + HDBSCAN); no iteration count applies |
| `Top2Vec` | `doc_names` | Exposes cluster `labels`, not a doc_names property |
| `Top2Vec` | `iters` | Not an iterative sampler (UMAP + HDBSCAN); no iteration count applies |

Everything else currently missing (for example `topic_names` on most models,
`transform` on the keyword/seeded/anchored models, `theta_draws`/`doc_lengths`
on the remaining Dirichlet models, `coherence`/`save`/`load` on the neural and
cluster models) is a tracked gap in `KNOWN_GAPS`, not an exemption.

## The Rust core trait

The Python contract above has a mirror one layer down, in the Rust core. Every
fitted model struct (`CtmModel`, `TopicModel`, `StsModel`, `HdpModel`, …)
implements a small trait hierarchy in `topica-core/src/estimator.rs`, so the uniform surface
is testable in `cargo test --lib` with no Python and is the binding point for a
future R frontend (issue #75). The PyO3 getters become thin forwarders to it.

```rust
// topica-core/src/estimator.rs — the Tier-0 floor, on the fitted *struct*, not the pyclass.
pub trait Estimator {
    fn num_topics(&self) -> usize;
    fn topic_word(&self) -> Vec<Vec<f64>>;       // (K, V); topic_word().len() == num_topics()
    fn doc_topic(&self) -> Vec<Vec<f64>>;        // (D, K)
    fn fit_history(&self) -> Vec<(usize, f64)>;  // [] when no per-iteration trace
    fn converged(&self) -> Option<bool>;         // None for non-iterative models
    fn model_family(&self) -> ModelFamily;       // Dirichlet | LogisticNormal | None_
}

// Tier-2 family traits — implementing them is what forces the posterior to exist.
pub trait DirichletModel: Estimator {           // topica-core/src/estimator.rs
    fn alpha(&self) -> Vec<f64>;                 // length K
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>>; // (S, D, K); [] if not retained
    fn doc_lengths(&self) -> Vec<usize>;         // length D
}
pub trait LogisticNormalModel: Estimator {      // topica-core/src/variational/mod.rs
    fn eta_dim(&self) -> usize;                  // K-1 (STM/CTM), 2K-1 (STS)
    fn eta_mean(&self) -> &[Vec<f64>];           // (D, eta_dim)
    fn eta_cov(&self) -> &[Vec<f64>];            // (D, eta_dim²), row-major
}
```

`topic_word` and `alpha` return owned values, not slices: the Gibbs family
computes φ on demand from its count tables and stores a symmetric scalar α, so a
borrowed-slice contract could not be implemented additively. `eta_mean`/`eta_cov`
stay borrowed because every logistic-normal model genuinely stores them — which
is the point of the Tier-2 trait. You cannot implement `LogisticNormalModel`
without producing a variational posterior over η; that requirement is what
surfaced the STS eta gap.

**The Rust conformance check** mirrors the Python one. `src/conformance.rs`
provides `check_conformance(&dyn Estimator)` (Tier-0 shape: `topic_word` row
count, `doc_topic` rows summing to 1 for the generative families) plus
`check_dirichlet(&dyn DirichletModel)` and `check_logistic_normal(&dyn
LogisticNormalModel)` (Tier-2 shapes). Each model's own `#[cfg(test)]` module has
a `*_conforms` test that fits a small instance and asserts these return no
violations — so a contract gap fails `cargo test --lib` at the source, before the
Python layer or a release. `conformance.rs::RUST_ESTIMATORS` is the single-source
registry (family + structural exemptions) that mirrors
`python/topica/conformance.py`; keep the two in lockstep.

**Shared variational kernels.** Logistic-normal models do not re-implement the
E-step. `topica-core/src/variational/` holds the reusable pieces: `laplace_estep` (the
parallel, document-order-preserving Laplace E-step driver — CTM, STM, and STS all
fit through it), `lbfgs_minimize`, `fit_gamma_ridge` (the pooled-ridge Γ M-step),
and `doc_sparse`. A new logistic-normal model should call `laplace_estep` rather
than write its own parallel E-step, both to inherit the bit-for-bit determinism
guarantee (the serial sufficient-statistic reduce stays in document order
regardless of thread count) and to avoid drift.

## Checking your model

Run the conformance test in isolation:

```bash
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python -m pytest tests/test_conformance.py -v
```

Or call the helper directly from Python:

```python
import topica

violations = topica.provenance.check_conformance(MyNewModel(num_topics=5))
for v in violations:
    print(v)
```

An empty list means the model satisfies every applicable requirement. A non-
empty list tells you exactly which attributes or methods are missing before you
open a pull request.

## Adding a new estimator

1. Implement the model in `src/<name>.rs` with PyO3 bindings under `src/python/`
   (in the `src/python/mod.rs` hub, or a per-model `src/python/<name>.rs` module),
   or in pure Python in `python/topica/`.
2. Export it from `python/topica/__init__.py`.
3. Add it to `REGISTRY` in `python/topica/conformance.py` with a zero-arg
   factory lambda and the correct `model_family` string.
4. Add the class to `_topica.pyi` to keep the stub in sync.
5. Run `topica.provenance.check_conformance(MyModel())` to find any missing attributes.
6. For each missing attribute:
   - If it is a structural impossibility for this model, add it to `EXEMPT`
     with a clear reason.
   - Otherwise, implement it so the check passes.
7. Run the full conformance test. All cells for the new model must either pass,
   be exempted, or (temporarily) be listed in `KNOWN_GAPS`.

Do not add entries to `KNOWN_GAPS` for a brand-new model unless you intend to
fix them in an immediately following commit. The purpose of `KNOWN_GAPS` is to
track pre-existing drift, not to grant new models a blanket deferred
implementation.
