# Contributing models to topica: an implementer's playbook

This is the deep guide for two tasks: **(A) adding a feature to an existing
model** and **(B) adding a new model**. It is written to be followed
step by step, including by an LLM coding agent. For general setup and PR
mechanics and build commands see [`CONTRIBUTING.md`](CONTRIBUTING.md). Read it
first; this file assumes it.

> **Adding a whole new model with an LLM agent?** The `add-topic-model` skill
> (`.claude/skills/add-topic-model/`) orchestrates the end-to-end workflow —
> grounding in the literature, building a gold standard from the reference
> implementation, validating with an independent benchmark agent and an
> author-emulation review agent, and opening the PR. It treats *this* file as the
> mechanical reference for the Rust/PyO3 work. Use the skill for the process; use
> this playbook for the how-to it points into.

## Architecture in sixty seconds

topica is a Rust core exposed to Python through PyO3 (0.22, maturin, abi3-py39),
with a thin Python layer on top.

```
src/<model>.rs        pure-Rust algorithm (sampler/EM), no Python types. Unit-testable.
src/python/mod.rs     the PyO3 binding hub: many #[pyclass]es plus free functions, and
                      the single #[pymodule] registration fn (`m.add_class::<...>()`) at
                      the bottom. Large models are extracted into their own module:
src/python/<model>.rs a per-model binding module (btm.rs, disclda.rs, scholar.rs, ...).
                      Defines `pub struct <Model>`; mod.rs pulls it in with
                      `use <model>::<Model>;` and registers it in the #[pymodule] fn.
src/python/{arrays,error,py_corpus,save}.rs   shared binding helpers.
topica-core/          vendorable crate holding the shared CTM/STM variational math.
                      The root crate re-exports it; touch it for CTM/STM-family changes.
src/model.rs          shared TopicModel state used by the MALLET-family samplers.
src/{sampler,optimize,coherence,corpus,linalg,spectral,reduce,represent}.rs
                      shared machinery (Gibbs sweep, hyperparameter optimization,
                      UMass coherence, the Corpus, linear algebra, spectral init,
                      dimensionality reduction, c-TF-IDF / centroid representation).
python/topica/__init__.py   re-exports the compiled surface so `import topica` works.
python/topica/*.py          model-agnostic helpers (analysis, diagnostics, effects,
                            coherence, embedding, frames, formulas) and per-model
                            toolkits (stm.py, keyatm.py).
python/topica/_topica.pyi   type stub for the compiled extension. Must track bindings.
tests/                pytest. parity/ statistical checks vs R/MALLET. docs/ mkdocs.
```

The split that matters: **algorithms live in `src/<model>.rs` and know nothing
about Python; the `src/python/` modules are the only place that touches PyO3.**
Keep that boundary. The algorithm crate is where your unit tests and
finite-difference checks run; the binding is plumbing. Small models add their
`#[pyclass]` directly in `src/python/mod.rs`; larger ones get their own
`src/python/<model>.rs` module (see the extraction pattern above) that mod.rs
declares with `mod <model>;` and registers in the `#[pymodule]` fn.

For an existing model, the generated [model implementation
map](../docs/contributing/model-map.md) lists — per model — its algorithm file,
binding module, shared machinery, required Cargo feature, and parity tests, so
you can find every surface a change touches without searching the tree. It is
generated from `python/topica/registry.py` (edit `IMPL` there, not the table).

## Invariants you must not break

1. **`API_FROZEN.md` is the contract for existing surface.** Never rename or
   change the meaning of a shipped parameter, method, or attribute. Additive
   changes only (new keyword-only args with defaults, new methods, new
   accept-either overloads). `API_FROZEN.md` is intentionally untracked; if it
   shows up staged, `git restore --staged API_FROZEN.md`.
2. **Determinism.** A fixed `seed` (and fixed `num_threads`, where threaded) must
   reproduce bit-identical results. Seed every RNG from the model's `seed` via
   `ChaCha8Rng::seed_from_u64(self.seed)`. No wall-clock, no unseeded `rand`.
3. **Always build `--release`.** Debug Gibbs/EM loops are unusably slow.
4. **The `.pyi` stub tracks every binding signature change.** Out-of-sync stubs
   are a silent correctness bug for users' type checkers.
5. **Validate against a reference.** A new model ships with planted-data recovery
   tests and, where a reference implementation exists (R `stm`/`keyATM`, Java
   MALLET, the paper's released code), a statistical-parity check under
   `parity/`. Describe the validation in the PR.
6. **Prose is in the house register.** README, `docs/`, and docstrings use no em
   dashes, an agent-led "we", concrete claims over hedging, and no LLM filler.
7. **Names follow the shared vocabulary.** The iteration count is `iters`, the
   seed is `seed=42`, counts are `num_*`, a covariate design is reachable as
   `covariates=`, and so on. See
   [`docs/contributing/conventions.md`](../docs/contributing/conventions.md); the
   contract is enforced by `tests/test_naming_conventions.py`, which fails when a
   new model breaks a rule.

## The build and test loop

The dev virtualenv is `.venv-dev`, and maturin needs `VIRTUAL_ENV` set
explicitly:

```bash
# rebuild after any Rust change
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/maturin develop --release --features python

# the three gates (all must pass before a PR)
cargo test --lib                  # add --features embeddings for an embedding model
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python -m pytest tests/ -q
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/mkdocs build --strict
```

The embedding branch (`cluster`, `reduce`, `represent`, `top2vec`, `bertopic`)
is behind the `embeddings` feature, so its `#[cfg(test)]` tests only run under
`cargo test --lib --features embeddings` (which `--features python` implies). A
plain `cargo test --lib` silently skips them.

Pure-Python changes (a helper in `python/topica/*.py`) do not need a rebuild,
only the pytest and mkdocs gates. Any Rust change needs the `maturin develop`
rebuild first.

---

## Testing conventions

topica tests at three levels: Rust unit tests in the algorithm crate, Python
behavior tests in `tests/`, and statistical-parity scripts in `parity/`. Both
Part A and Part B refer back to this section.

### Shared fixtures

`tests/conftest.py` provides a 30-document two-cluster toy corpus (15
"animal" + 15 "space" docs over disjoint vocabularies) as session fixtures
`toy_docs` (`list[list[str]]`) and `toy_corpus` (a `Corpus`). Use them for quick
smoke and recovery tests; only build a bespoke corpus when the model needs a
specific structure (short texts, hierarchy, covariates).

### The four idioms every model test uses

These come straight from `tests/test_extra_models.py` and `test_determinism.py`.

1. **Shapes and normalization.**
   ```python
   m = topica.MyModel(num_topics=2, seed=1); m.fit(docs, iters=300)
   assert m.topic_word.shape == (2, len(m.vocabulary))
   assert m.doc_topic.shape == (len(docs), 2)
   np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
   ```
2. **Planted-data recovery** (disjoint vocabularies, then check each topic owns a
   distinct block, robust to topic permutation):
   ```python
   blocks = [{"cat", "dog", "pet"}, {"star", "moon", "sky"}]
   tops = [{w for w, _ in m.top_words(3, topic=t)} for t in range(2)]
   owned = [max(range(2), key=lambda b: len(tops[t] & blocks[b])) for t in range(2)]
   assert set(owned) == {0, 1}
   ```
3. **Determinism** (same seed identical, on both matrices; for LDA use
   `optimize_interval=0` to remove the one nondeterministic-by-design path):
   ```python
   a = topica.MyModel(num_topics=2, seed=3); a.fit(docs, iters=150)
   b = topica.MyModel(num_topics=2, seed=3); b.fit(docs, iters=150)
   assert np.array_equal(a.topic_word, b.topic_word)
   ```
   Also assert a different seed gives a different result, so the test cannot pass
   trivially.
4. **save / load round-trip** (`tmp_path`) and **bad params**
   (`pytest.raises(ValueError)` for `num_topics` below the model's minimum and for
   any out-of-range hyperparameter):
   ```python
   p = str(tmp_path / "m.tt"); m.save(p)
   assert np.array_equal(m.topic_word, topica.MyModel.load(p).topic_word)
   with pytest.raises(ValueError):
       topica.MyModel(num_topics=1)
   ```

Add one test that runs the **analysis surface** on the model
(`topica.summary(m)`, `topica.topic_table(m)`, `topica.coherence(m, texts)`),
which confirms the four required attributes are wired correctly.

### Rust unit tests

Put `#[cfg(test)] mod tests { ... }` in `src/<model>.rs` and run with
`cargo test --lib`. This is where planted-data recovery, determinism, and (for
variational models) finite-difference gradient checks belong, because they run
without the Python layer and are fast to iterate.

### Parity and reference contracts

Two distinct reference checks, both already in the repo:

- **`parity/<model>_compare.py`** is *statistical* agreement, not byte equality.
  Independent implementations share no RNG, so the bar is "we reproduce the
  reference as faithfully as the reference reproduces itself": fit both on the
  identical integer-coded corpus, align topics, and require the aligned
  topic-word cosine to meet the reference's own seed-to-seed self-consistency
  floor (see the header of `parity/stm_r_compare.py`). These **skip cleanly**
  (`shutil.which("Rscript")` / package probe, then `pytest.skip`) so they are
  no-ops when the tooling is absent.
- **`tests/test_reference_default_contracts.py`** checks that a default topica
  claims to mirror (R `stm`/`keyATM`, MALLET) actually equals the reference
  program's own default. It parses the `.pyi` stub with `ast`, so **a default you
  change in a binding must be changed in the stub too** or this test will catch
  the drift. Where exact parity is not yet true, the mismatch is pinned as a
  visible contract rather than left to drift silently.

Mark slow parity tests with `pytestmark = pytest.mark.parity` (the marker is
registered in `conftest.py`). `tests/test_cli_parity.py` additionally builds the
release binaries and compares binding output against the `train` CLI byte-for-
byte for the count-based models that have a CLI path; if your model is exposed on
the CLI, extend it.

---

## Part A: adding a feature to an existing model

Pick the smallest of the three patterns that fits.

### A1. A new constructor or fit parameter (Rust side)

Example shape: add a `min_cluster_weight` knob, or a new `weights=` mode.

1. **Algorithm first.** Thread the parameter through `src/<model>.rs` so the
   pure-Rust function takes and uses it. Add or extend a `#[cfg(test)]` unit test
   that exercises the new behavior (planted data where the parameter changes the
   recovered result in a predictable direction).
2. **Binding.** In the model's binding (`src/python/mod.rs`, or its
   `src/python/<model>.rs` module if extracted), add the argument to the relevant
   `#[pyo3(signature = (...))]`. It must be **keyword-only** (after the `*`) with
   a default that preserves current behavior. Validate it in the body and raise
   `PyValueError` on bad input, matching the existing messages. Pass it into the
   algorithm call inside `py.allow_threads(...)`.
   ```rust
   #[pyo3(signature = (data, *, iters=30, report_interval=0, new_knob=1.0))]
   fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>,
          iters: usize, report_interval: usize, new_knob: f64) -> PyResult<()> {
       if new_knob <= 0.0 {
           return Err(PyValueError::new_err("new_knob must be > 0"));
       }
       // ... extract corpus, then:
       let (model, corpus) = py.allow_threads(move || {
           let m = mymodel::fit(&corpus.docs, /* ... */, new_knob, &mut rng);
           (m, corpus)
       });
       // ...
   }
   ```
3. **Stub.** Add the parameter (with the same default) to the method in
   `python/topica/_topica.pyi`.
4. **Docstring.** Update the binding's `///` doc and the stub docstring. State
   what the parameter does and its default behavior.
5. **Test.** Add a pytest in `tests/` that sets the parameter and asserts the
   user-visible effect (and that the default path is unchanged).
6. **Changelog.** Add an `### Added` (new capability) or `### Changed` line to
   the `[Unreleased]` section of `CHANGELOG.md`, referencing the issue number.

### A2. A new getter, attribute, or method on a model

Example shape: expose a convergence trace, a new representation, a `transform`.

- Store what you need on the `#[pyclass]` struct (set during `fit`). Guard reads
  with `self.require_fitted()?` so an unfitted model raises the standard
  `RuntimeError("model is not fitted yet; call fit() first")`.
- Return numpy arrays via `to_pyarray_bound(py)` (`Array2<f64>` to
  `PyArray2<f64>`, etc.); return small structured results as `Vec<(...)>`.
- If the new method is a representation of topics, reuse `topic_words_helper`
  and the `#[pyo3(signature = (n=10, *, topic=None))]` shape so it matches
  `top_words` across the codebase.
- Add it to the `.pyi` stub as a `@property` or method, update docstrings, add a
  pytest, add a changelog line.

### A3. A model-agnostic helper (pure Python, no Rust)

Most diagnostics, effect estimators, and reporting helpers are pure Python that
read a fitted model's public attributes. This is the cheapest place to add value
and needs no rebuild.

- Put it in the right module: post-hoc analysis in `diagnostics.py`, the
  model-neutral fitted-model surface in `analysis.py`, covariate/effect work in
  `effects.py`, coherence in `coherence.py`, embeddings in `embedding.py`.
- **Read only the analysis-surface attributes** (next section), never
  model-internal fields, so the helper works across every model.
- **Accept a fitted model or the raw matrix** as the first argument, the pattern
  `frex`/`exclusivity`/`topic_correlation` already use: derive `topic_word` /
  `doc_topic` / `vocabulary` from the model when given one, and raise a clear
  message when a bare matrix is passed without the vocabulary it needs.
- Re-export it in `python/topica/__init__.py` (add to the relevant `from .module
  import (...)` block and to `__all__`), add a pytest, document it on the right
  `docs/api/*.md` page, add a changelog line.

---

## Part B: adding a new model

Use `GSDMM` (`src/gsdmm.rs` + the `GSDMM` block in `src/python/mod.rs`) as the
reference template. It is small, self-contained, and exercises the whole
contract. Work in this order.

### Scaffold the touchpoints first

To avoid missing one of the coordinated edits below, generate the standard files
from a template:

```bash
just new-model MyModel          # or: python scripts/new_model.py --name MyModel
```

This writes `src/<snake>.rs` (algorithm + `Estimator` trait + `#[cfg(test)]`
stubs), `src/python/<snake>.rs` (the binding, following the GSDMM/BTM shape), and
`tests/test_<snake>.py` (a pytest skeleton), each stamped with a
`SCAFFOLD(<Name>)` marker, and prints the wiring checklist filled in for your
model. It deliberately **wires nothing** into the shared files (`lib.rs`,
`src/python/mod.rs`, `__init__.py`, the registry, …), so an un-finished model is
inert — not compiled, not exported, not registered, and so unable to ship or
fake-pass conformance. Work through B0–B8 to implement and wire it in. The
generated files compile as-is (`fit` is a `todo!()`), so build early and often.

`tests/test_scaffold_guard.py` is the backstop: once your model is in the
registry, it fails if any `SCAFFOLD(<Name>)` marker still remains in its files, so
you cannot land a half-finished placeholder. Search for leftovers with
`grep -rn 'SCAFFOLD(MyModel)' src tests`.

### B0. Decide the family and what you can reuse

- **Count-based (Gibbs or variational EM)** over a `Corpus`: lean on
  `src/model.rs` (the MALLET `TopicModel` state), `src/sampler.rs`, and
  `src/optimize.rs`. CTM/STM-style logistic-normal models use `src/linalg.rs`
  and `src/spectral.rs`.
- **Embedding-based** (you bring document or word vectors): lean on
  `src/reduce.rs` (PCA/UMAP-style reduction), `src/cluster.rs` (HDBSCAN, kmeans,
  agglomerative), and `src/represent.rs` (c-TF-IDF, centroid words). See
  `bertopic.rs`/`top2vec.rs`.

Confirm the model is genuinely distinct from what exists. If two models would
share their headline output, differentiate them in `top_words` (see how
`Top2Vec` defaults to the centroid representation while `BERTopic` uses
c-TF-IDF), not just internally.

### B1. The algorithm crate: `src/<model>.rs`

Pure Rust, no PyO3. A `fit_*` function that takes the corpus (or embeddings),
hyperparameters, and a `&mut R: Rng`, and returns a state struct with whatever
the binding will read. Add `#[cfg(test)]` unit tests:

- **Planted-data recovery**: build a corpus with known topics, fit, assert the
  recovered topics match (cosine, or top-word overlap, above a threshold).
- **Determinism**: same seed gives identical output.
- **Gradients (variational models)**: finite-difference check against the
  analytic gradient.

Register the module in `src/lib.rs` (`pub mod <model>;`, matching the others).

**Implement the `Estimator` trait on your fitted struct.** The core trait
hierarchy lives in `src/estimator.rs` (and `src/variational/mod.rs`); it is the
Rust mirror of the [estimator contract](../docs/contributing/estimator-contract.md)
and the binding point for the R frontend. On your `*Model` struct, add:

```rust
use crate::estimator::{Estimator, ModelFamily};

impl Estimator for MyModel {
    fn num_topics(&self) -> usize { self.num_topics }
    fn topic_word(&self) -> Vec<Vec<f64>> { /* stored beta, or compute φ */ }
    fn doc_topic(&self) -> Vec<Vec<f64>> { /* stored/computed θ, rows sum to 1 */ }
    fn fit_history(&self) -> Vec<(usize, f64)> { /* (iter, objective) or Vec::new() */ }
    fn converged(&self) -> Option<bool> { /* Some(flag) or None */ }
    fn model_family(&self) -> ModelFamily { ModelFamily::Dirichlet /* or LogisticNormal / None_ */ }
}
```

Then the family trait, if applicable: `DirichletModel` (collapsed-Gibbs:
`alpha`/`theta_draws`/`doc_lengths`) or `LogisticNormalModel`
(`eta_dim`/`eta_mean`/`eta_cov`, in `src/variational/mod.rs`). A logistic-normal
model should run its E-step through `crate::variational::laplace_estep` rather
than write its own parallel loop — that is what keeps the fit bit-for-bit
deterministic across thread counts. Add a `*_conforms` unit test next to your
recovery tests:

```rust
#[test]
fn mymodel_conforms() {
    let m = /* fit a tiny instance */;
    assert!(crate::conformance::check_conformance(&m).is_empty());
    // and check_dirichlet(&m) / check_logistic_normal(&m) for those families
}
```

Add a row to `RUST_ESTIMATORS` in `src/conformance.rs` (name, family, and any
structural exemptions) so it stays in lockstep with the Python `REGISTRY`. If a
Tier-0 method is structurally undefined (a time-sliced or tree model has no flat
`doc_topic`), return an empty value and record the exemption in that row — do not
fake a value.

### B2. The PyO3 binding in `src/python/`

Add the binding in `src/python/mod.rs` next to the other model blocks, or, for a
larger model, in its own `src/python/<model>.rs` module (declared with
`mod <model>;` in mod.rs and pulled in with `use <model>::<Model>;`).

Add a `#[pyclass(module = "topica")]` struct holding the hyperparameters, a
`fitted: bool`, the fitted state (`phi`/`theta` as `Option<Array2<f64>>`), and
the `Corpus`. Implement, mirroring `GSDMM`:

```rust
#[pyclass(module = "topica")]
pub struct MyModel {
    num_topics: usize,
    // hyperparameters ...
    seed: u64,
    fitted: bool,
    phi: Option<Array2<f64>>,    // K x V  (topic-word)
    theta: Option<Array2<f64>>,  // D x K  (doc-topic)
    corpus: Option<corpus::Corpus>,
}

impl MyModel {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted { Ok(()) }
        else { Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first")) }
    }
}

#[pymethods]
impl MyModel {
    #[new]
    #[pyo3(signature = (num_topics, *, /* hyperparams with defaults */ seed=42))]
    fn new(#[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
           /* ... */ seed: u64) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        Ok(MyModel { num_topics, /* ... */ seed,
                     fitted: false, phi: None, theta: None, corpus: None })
    }

    #[pyo3(signature = (data, *, iters=1000))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        // Accept a Corpus OR a list[list[str]] (no frequency filtering for raw lists).
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None,
                std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (phi, theta, corpus) = py.allow_threads(move || {
            let state = mymodel::fit(&corpus.docs, /* ... */, iters, &mut rng);
            (state.phi, state.theta, corpus)
        });
        self.phi = Some(phi);
        self.theta = Some(theta);
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    // --- The analysis-surface contract (see B3). All four are required. ---
    #[getter] fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?; Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter] fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?; Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter] fn num_topics(&self) -> usize { self.num_topics }
    #[getter] fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?; Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    // --- Conventional extras every model provides ---
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(),
            &self.corpus.as_ref().unwrap().id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    // save/load via a serde *State struct + write_state/read_state (see GsdmmState).
    fn __repr__(&self) -> String {
        format!("MyModel(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}
```

Notes that save time:

- `py_num_topics` is the shared converter that accepts int-like inputs; use it
  on `num_topics`.
- Release the GIL with `py.allow_threads(...)` around the fit. Move the corpus in
  and return it out (as in GSDMM) so you keep ownership for the getters.
- For a cluster model that leaves documents unassigned, also expose a `labels`
  getter (`PyArray1<i64>`, `-1` for noise). The analysis surface reads it.
- For held-out support, add a `transform(data)` method that infers `theta` for
  new documents against the fitted `phi`.

### B3. The model-neutral analysis-surface contract

`analysis.py`, `diagnostics.py`, `effects.py`, and `plot_report` read only these
attributes, so providing them is what makes every report and diagnostic work for
your model for free:

| Attribute     | Type                    | Required | Meaning |
|---------------|-------------------------|----------|---------|
| `topic_word`  | `ndarray (K, V)`        | yes      | phi, topic-word distributions |
| `doc_topic`   | `ndarray (D, K)`        | yes      | theta, rows sum to 1 |
| `num_topics`  | `int`                   | yes      | K (effective K for discovery models) |
| `vocabulary`  | `list[str]` length V    | yes      | column order of `topic_word` |
| `labels`      | `ndarray (D,) int64`    | optional | hard assignment, `-1` = noise (cluster models) |
| `topic_names` | `list[str]`             | optional | defaults derived if absent |
| `doc_names`   | `list[str]`             | optional | for representative-document output |
| `alpha` / `eta_mean`+`eta_cov` | getter | optional | enables composition standard errors (see below) |
| `top_words(n, *, topic=None)` | method  | optional | falls back to `topic_word` if absent |

If you provide the four required ones, `summary`, `report`, `topic_table`,
`topic_info`, `plot_report`, `coherence`, `exclusivity`, `frex`,
`representative_docs`, `estimate_effect`, and `standard_errors` all work without
per-model code.

**Standard errors.** `topica.standard_errors(model, corpus, ...)` propagates
topic-estimation uncertainty. Its bootstrap path (`method="bootstrap"`) needs
only the four required attributes, so it works for any model out of the box (an
embedding model passes a `refit=` hook that resamples its embeddings). Its
cheaper default, method-of-composition, is selected by `model_family`, which
reads the *class*: expose `eta_mean`+`eta_cov` (a logistic-normal posterior, as
STM/CTM do) or `alpha` together with `doc_topic` (a collapsed-Gibbs Dirichlet
model, as LDA/keyATM do) and composition turns on automatically. A Gibbs model
also relies on `Corpus.doc_lengths` for the per-document Dirichlet, which the
`Corpus` already provides. A model with neither (the embedding models) correctly
falls back to the bootstrap.

### B4. Register the class

In the `#[pymodule]` function at the bottom of `src/python/mod.rs` (if you put
the binding in its own module, also add `mod <model>;` and `use <model>::<Model>;`
near the top of mod.rs first):

```rust
m.add_class::<MyModel>()?;
```

### B5. Re-export and stub

In `python/topica/__init__.py`: add `MyModel` to the `from ._topica import (...)`
block and to `__all__`. In `python/topica/_topica.pyi`: add a `class MyModel`
with the constructor, `fit`, and every property/method, docstrings included,
matching the binding exactly.

### B6. Tests

Add `tests/test_<model>.py` following the four idioms in **Testing conventions**
above (shapes/normalization, planted-data recovery, determinism, save-load +
bad-params), plus an analysis-surface test and edge cases (K=1 where meaningful,
empty/single document, a clear error on an empty corpus). Put recovery and
determinism checks for the algorithm itself in `#[cfg(test)]` in
`src/<model>.rs`. If a reference implementation exists, add
`parity/<model>_compare.py` (statistical agreement against the reference's own
self-consistency floor; skips cleanly when the tooling is absent) and, if you
claim a default mirrors a reference, extend
`tests/test_reference_default_contracts.py`.

### B7. Docs

- `docs/guides/models.md`: add the model to the catalog with one paragraph on
  what it is for and when to choose it.
- `docs/api/models.md`: add its API entry.
- `mkdocs.yml`: only if you add a new page; existing pages need no nav change.
- Write real docstrings on the binding and the stub (they are user-facing).
- `CHANGELOG.md`: an `### Added` line under `[Unreleased]` naming the model and
  its reference.

### B8. Validation expectations

A model is not done until it is validated. State in the PR: the reference you
compared against, the metric, and the result (for example, "aligned topic-word
cosine mean 0.93 vs R `stm` on the poliblog corpus, recovered effect within
1 SE"). If no reference exists, show planted-data recovery and a close reading of
the topics on a real corpus.

---

## Definition of done

- [ ] Algorithm in `src/<model>.rs` with `#[cfg(test)]` recovery + determinism tests; `mod` added to `src/lib.rs`.
- [ ] `Estimator` (and the family trait, where it applies) implemented on the fitted struct, a `*_conforms` test added, and a `RUST_ESTIMATORS` row in `src/conformance.rs`.
- [ ] `#[pyclass]` binding in `src/python/` (mod.rs or a per-model `src/python/<model>.rs` module) exposing the four required analysis-surface attributes plus `top_words`, `coherence`, `save`/`load`, `__repr__`.
- [ ] Keyword-only new params with behavior-preserving defaults; clear `PyValueError`/`RuntimeError` messages; GIL released during fit; RNG seeded from `self.seed`.
- [ ] Registered in `#[pymodule]`; re-exported in `__init__.py` (+ `__all__`); `.pyi` stub updated to match.
- [ ] `tests/test_<model>.py` passes; `parity/` check added where a reference exists.
- [ ] Docstrings, `docs/guides/models.md`, `docs/api/models.md`, and `CHANGELOG.md` updated.
- [ ] All three gates pass: `cargo test --lib`, `pytest tests/ -q`, `mkdocs build --strict`.
- [ ] `API_FROZEN.md` not staged; changes to existing surface are additive only.

## Common pitfalls

- Forgetting `maturin develop --release` after a Rust edit, then debugging stale
  behavior. Rebuild first.
- Letting the `.pyi` stub drift from the binding.
- Reading model-internal fields from a Python helper instead of the analysis
  surface, which silently breaks for other models.
- An unseeded or wall-clock RNG path that breaks determinism (and the parity
  tests).
- Renaming a shipped argument to "make it consistent". Add an accept-either
  overload or an alias instead; `API_FROZEN.md` is the contract.
- Em dashes and hedged filler in user-facing prose.
