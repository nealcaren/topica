#!/usr/bin/env python
"""Scaffold the standard touchpoints for a new topic model (issue #386).

Generates three self-contained template files, each stamped with a greppable
``SCAFFOLD(<Name>)`` sentinel:

    src/<snake>.rs           the Rust algorithm crate (fit loop + Estimator trait
                             + #[cfg(test)] recovery/determinism stubs)
    src/python/<snake>.rs    the PyO3 binding (pyclass + the analysis surface)
    tests/test_<snake>.py    the pytest skeleton (skipped until you implement)

It deliberately wires **nothing** into the shared registration files. An
un-wired model is inert: it is not compiled (`src/lib.rs` has no `pub mod`), not
exported, and not in the registry, so it cannot silently ship or fake-pass
conformance. When you are ready, follow the checklist this script prints (it is
the "Definition of done" from CONTRIBUTING-MODELS.md, filled in for your model).

The sentinel is the safety net: ``tests/test_scaffold_guard.py`` fails if any
model that IS registered still carries a ``SCAFFOLD`` marker in its files — so
you cannot ship a half-finished placeholder.

    python scripts/new_model.py --name MyModel
    python scripts/new_model.py --name MyModel --snake my_model --family gibbs

Run ``--help`` for options. Follows the naming and estimator-contract conventions
in .github/CONTRIBUTING-MODELS.md; supply the algorithm, invariants, and
validation yourself.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENTINEL = "SCAFFOLD"


def to_snake(name: str) -> str:
    """PascalCase / CamelCase -> snake_case (MyModel -> my_model, GSDMM -> gsdmm)."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    return s.lower()


RUST_ALGO = r'''//! __NAME__: <one-line description of the model and its reference>.
//!
//! SCAFFOLD(__NAME__): implement the algorithm below, then delete this line and
//! every other `SCAFFOLD(__NAME__)` marker in this file. Pure Rust, no PyO3.
//! See .github/CONTRIBUTING-MODELS.md section B1.
//!
//! Note: the fitted state stores the topic-word and doc-topic matrices as
//! `Vec<Vec<f64>>` (not `ndarray::Array2`) — `ndarray` is behind the `embeddings`
//! feature, so a default-build model file must not depend on it. The binding
//! converts to numpy arrays with the shared `vecs_to_arr2` helper.

use crate::corpus::Corpus;
use crate::estimator::{Estimator, ModelFamily};
use rand::Rng;

/// Fitted state for [`fit`]: whatever the PyO3 binding reads back.
pub struct __NAME__Model {
    pub num_topics: usize,
    pub topic_word: Vec<Vec<f64>>, // K rows of length V
    pub doc_topic: Vec<Vec<f64>>,  // D rows of length K (each sums to 1)
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

/// Fit the model. Takes the corpus, hyperparameters, and a seeded RNG; returns
/// the fitted state. Seed every random draw from `rng` (a `ChaCha8Rng` the
/// binding seeds from `self.seed`) so a fixed seed reproduces bit-for-bit.
pub fn fit<R: Rng>(
    _corpus: &Corpus,
    num_topics: usize,
    _iters: usize,
    _rng: &mut R,
) -> __NAME__Model {
    // SCAFFOLD(__NAME__): implement the fit/inference loop. Fill topic_word
    // (K x V) and doc_topic (D x K, each row summing to 1). Record
    // (iter, objective) in fit_history if the model has a per-iteration
    // objective; leave it empty otherwise.
    let _ = num_topics;
    todo!("implement __NAME__::fit")
}

impl Estimator for __NAME__Model {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word.clone()
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic.clone()
    }
    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.fit_history.clone()
    }
    fn converged(&self) -> Option<bool> {
        Some(self.converged)
    }
    fn model_family(&self) -> ModelFamily {
        // SCAFFOLD(__NAME__): Dirichlet (collapsed-Gibbs), LogisticNormal (STM/CTM),
        // or None_ (embedding/cluster). Also implement DirichletModel /
        // LogisticNormalModel if applicable (see B1).
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // SCAFFOLD(__NAME__): build a planted-topic corpus, fit, and assert the
    // recovered topics match (cosine or top-word overlap above a threshold).
    #[test]
    #[ignore = "SCAFFOLD(__NAME__): write the planted-data recovery test"]
    fn __SNAKE___recovers_planted_topics() {
        todo!("planted-data recovery")
    }

    // Same seed => identical output. Required for every model.
    #[test]
    #[ignore = "SCAFFOLD(__NAME__): write the determinism test"]
    fn __SNAKE___is_deterministic() {
        let mut _rng = ChaCha8Rng::seed_from_u64(42);
        todo!("assert two fits with the same seed are identical")
    }

    // The fitted struct satisfies the estimator contract.
    #[test]
    #[ignore = "SCAFFOLD(__NAME__): fit a tiny instance and check conformance"]
    fn __SNAKE___conforms() {
        // let m = fit(&tiny_corpus, 2, 20, &mut ChaCha8Rng::seed_from_u64(0));
        // assert!(crate::conformance::check_conformance(&m).is_empty());
        todo!("conformance")
    }
}
'''

RUST_BINDING = r'''//! Python bindings for __NAME__.
//!
//! SCAFFOLD(__NAME__): flesh out the binding, then delete every
//! `SCAFFOLD(__NAME__)` marker in this file. Mirrors the GSDMM/BTM pattern; see
//! .github/CONTRIBUTING-MODELS.md section B2.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict; // for the `settings` getter (not re-exported by `use super::*`)
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// __NAME__: <user-facing docstring — what it does and when to choose it>.
#[pyclass(module = "topica")]
pub struct __NAME__ {
    num_topics: usize,
    // SCAFFOLD(__NAME__): add hyperparameter fields here.
    seed: u64,
    fitted: bool,
    model: Option<crate::__SNAKE__::__NAME__Model>,
    corpus: Option<corpus::Corpus>,
}

impl __NAME__ {
    fn fitted_model(&self) -> PyResult<&crate::__SNAKE__::__NAME__Model> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl __NAME__ {
    #[new]
    #[pyo3(signature = (num_topics, *, seed=42))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        // SCAFFOLD(__NAME__): add keyword-only hyperparameters with defaults.
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        Ok(__NAME__ {
            num_topics,
            seed,
            fitted: false,
            model: None,
            corpus: None,
        })
    }

    // `fit` returns `self` so calls chain (estimator contract `-> Self`, #402):
    // take `slf: PyRefMut<'_, Self>` instead of `&mut self`, refer to fields via
    // `slf.`, and return `Ok(slf.into())` on every success path.
    #[pyo3(signature = (data, *, iters=1000))]
    fn fit(mut slf: PyRefMut<'_, Self>, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<Py<Self>> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?.0
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_topics = slf.num_topics;
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let (model, corpus) = py.allow_threads(move || {
            let model = crate::__SNAKE__::fit(&corpus, num_topics, iters, &mut rng);
            (model, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.fitted = true;
        Ok(slf.into())
    }

    // Uniform constructor-config introspection (#400): return `__init__`'s
    // settings as a JSON-serialisable dict, keyword-named to match the
    // constructor, with effective values (not internal flags). `tests/
    // test_model_settings.py` derives the expected keys from the constructor
    // signature and will fail until every hyperparameter appears here (add a
    // factory for __NAME__ there too). Add each hyperparameter field you added
    // to `new` above.
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        // SCAFFOLD(__NAME__): add every constructor hyperparameter (not data args).
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    // --- Required analysis surface (B3) ---
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    // --- Conventional extras ---
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_topics,
            n,
            topic,
        )
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    // SCAFFOLD(__NAME__): add save/load via a serde *State struct (see BtmState).

    fn __repr__(&self) -> String {
        format!("__NAME__(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}
'''

PYTEST = r'''"""Tests for __NAME__ (issue: <link>).

SCAFFOLD(__NAME__): implement the model, then remove the module-level skip below
and write the tests. Follows the four idioms in CONTRIBUTING-MODELS.md
(shapes/normalization, planted-data recovery, determinism, save-load + bad-params)
plus an analysis-surface check and edge cases.
"""
import numpy as np
import pytest

import topica

pytest.skip(
    "SCAFFOLD(__NAME__): not implemented yet; remove this skip once __NAME__ ships",
    allow_module_level=True,
)


def _toy_docs():
    return [["a", "b", "c"], ["a", "b", "b"], ["c", "c", "d"], ["d", "e", "f"]]


def test_shapes_and_normalization():
    m = topica.__NAME__(2, seed=0).fit(_toy_docs(), iters=20)
    assert m.topic_word.shape == (2, len(m.vocabulary))
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)


def test_determinism():
    a = topica.__NAME__(2, seed=1).fit(_toy_docs(), iters=20)
    b = topica.__NAME__(2, seed=1).fit(_toy_docs(), iters=20)
    assert np.array_equal(a.topic_word, b.topic_word)


def test_recovers_planted_topics():
    # SCAFFOLD(__NAME__): build a corpus with known topics and assert recovery.
    raise NotImplementedError


def test_rejects_empty_corpus():
    with pytest.raises((ValueError, RuntimeError)):
        topica.__NAME__(2).fit([])
'''


def render(tpl: str, name: str, snake: str) -> str:
    return (
        tpl.replace("__NAME__", name)
        .replace("__SNAKE__", snake)
    )


def checklist(name: str, snake: str) -> str:
    return f"""
Scaffolded {name}. Nothing is wired in yet, so the model is inert (excluded from
the build and the registry). Implement it, then complete these steps — this is the
"Definition of done" from CONTRIBUTING-MODELS.md, filled in for {name}:

  ALGORITHM
  [ ] Implement src/{snake}.rs: the fit loop, the Estimator trait, and the
      #[cfg(test)] recovery + determinism tests (remove their #[ignore]).
  [ ] Add `pub mod {snake};` to src/lib.rs (alphabetical with the others).
  [ ] Add a row to RUST_ESTIMATORS in src/conformance.rs (name, family, exemptions).

  BINDING
  [ ] Implement src/python/{snake}.rs (hyperparameters, fit, save/load).
  [ ] `fit` returns self (#402): receiver is `mut slf: PyRefMut<'_, Self>`, return
      type `PyResult<Py<Self>>`, and every success path ends `Ok(slf.into())`. The
      scaffold already does this — keep it when you flesh out the body.
  [ ] `settings` getter (#400): return every constructor hyperparameter as a
      JSON-serialisable dict keyed by the `__init__` name (effective values, not
      internal flags). The scaffold stubs it — extend it for each field you add.
  [ ] In src/python/mod.rs: add `mod {snake};`, `use {snake}::{name};`, and
      `m.add_class::<{name}>()?;` in the #[pymodule] fn.

  PYTHON SURFACE
  [ ] python/topica/__init__.py: add {name} to the _topica import block and __all__.
  [ ] python/topica/_topica.pyi: add `class {name}` matching the binding exactly —
      including `settings` as a `@property` and `fit(...) -> "{name}"` (returns self,
      NOT `-> None`). tests/test_stub_sync.py and test_fit_returns_self.py check these.
  [ ] python/topica/registry.py: add a ModelInfo to REGISTRY (its `determinism` tag
      is the per-class default; topica.effective_determinism refines it per-config —
      add a rule branch there only if determinism is config-conditional) AND an
      ImplInfo to IMPL (source/binding/core/feature/validation), then run
      `python scripts/gen_model_tables.py` to regenerate the roster + model map.
  [ ] python/topica/conformance.py: add {name} to REGISTRY there if it needs a
      non-default factory.

  TESTS + DOCS
  [ ] tests/test_{snake}.py: remove the module-level skip and implement the tests.
  [ ] tests/test_model_settings.py: add a `{name}` factory to `_FACTORIES` (and any
      data-only constructor args to `_DATA_ARGS`) — the settings coverage test fails
      until every registered model is covered.
  [ ] parity/{snake}_compare.py if a reference implementation exists.
  [ ] docs/guides/models.md + docs/api/models.md prose; CHANGELOG.md `### Added` line.

  GATES (all green before the PR)
  [ ] just build && just test && just docs   (or the raw commands)
  [ ] No SCAFFOLD({name}) markers remain — `grep -rn 'SCAFFOLD({name})' src tests`
      must be empty. tests/test_scaffold_guard.py enforces this once {name} is
      registered.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="canonical class name, PascalCase (e.g. MyModel)")
    p.add_argument("--snake", help="file stem override (default: snake_case of --name)")
    p.add_argument("--force", action="store_true", help="overwrite existing scaffold files")
    args = p.parse_args()

    name = args.name
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name):
        p.error("--name must be a bare PascalCase identifier (letters/digits, no spaces)")
    snake = args.snake or to_snake(name)

    targets = {
        ROOT / "src" / f"{snake}.rs": render(RUST_ALGO, name, snake),
        ROOT / "src" / "python" / f"{snake}.rs": render(RUST_BINDING, name, snake),
        ROOT / "tests" / f"test_{snake}.py": render(PYTEST, name, snake),
    }

    existing = [t for t in targets if t.exists()]
    if existing and not args.force:
        print("refusing to overwrite (use --force):", file=sys.stderr)
        for t in existing:
            print(f"  {t.relative_to(ROOT)}", file=sys.stderr)
        return 1

    for path, content in targets.items():
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    print(checklist(name, snake))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
