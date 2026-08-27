//! Python binding for ReplyTM — a reply-threaded topic model (CTM/STM logistic-normal topics
//! with a reply-tree structured prior; see `crate::reply_tm`). Experimental tier: topica-original,
//! no published reference yet, so `fit` is gated behind `topica.enable_experimental()`.
//!
//! The class exposes the validated core — fit on token lists + a reply tree + an optional
//! categorical covariate, with topic/proportion/prevalence readouts (prevalence carries a
//! method-of-composition SE), the persistence parameter `kappa` with a profile-likelihood CI,
//! and the ELBO trace. Corpus/formula covariates, save/load, and the namespaced `effects`/
//! `evaluate` surface are follow-ups.

use super::*;
use numpy::PyArray2;
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// ReplyTM: a reply-threaded topic model. A reply's topic prior is coupled to the comment it
/// answers (a persistence-smoothing prior along reply edges), reverting toward its covariate-group
/// baseline; `kappa` measures the reversion (on real corpora it is typically ~0, i.e. persistence-
/// dominated). Reduces to a plain logistic-normal topic model when the reply tree is flat.
/// `num_topics` is K; `em_iters` the variational-EM iteration cap; `seed` makes the fit deterministic.
#[pyclass(module = "topica")]
pub struct ReplyTM {
    num_topics: usize,
    em_iters: usize,
    seed: u64,
    fitted: bool,
    vocab: Vec<String>,
    group_names: Vec<String>,
    beta: Vec<Vec<f64>>,
    doc_topic: Vec<Vec<f64>>,
    group_prevalence: Vec<Vec<f64>>,
    prevalence_se: Vec<Vec<f64>>,
    kappa: f64,
    kappa_ci: (f64, f64),
    sigma2: f64,
    p0: f64,
    bound_history: Vec<f64>,
}

impl ReplyTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "model is not fitted yet; call fit() first",
            ))
        }
    }
}

#[pymethods]
impl ReplyTM {
    #[new]
    #[pyo3(signature = (num_topics, *, em_iters=150, seed=13))]
    fn new(num_topics: usize, em_iters: usize, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if em_iters == 0 {
            return Err(PyValueError::new_err("em_iters must be >= 1"));
        }
        Ok(ReplyTM {
            num_topics,
            em_iters,
            seed,
            fitted: false,
            vocab: Vec::new(),
            group_names: Vec::new(),
            beta: Vec::new(),
            doc_topic: Vec::new(),
            group_prevalence: Vec::new(),
            prevalence_se: Vec::new(),
            kappa: f64::NAN,
            kappa_ci: (f64::NAN, f64::NAN),
            sigma2: f64::NAN,
            p0: f64::NAN,
            bound_history: Vec::new(),
        })
    }

    /// Fit ReplyTM. `docs` is a list of token lists (already tokenized). `parents[d]` is `d`'s
    /// parent **document index** in the reply tree (`-1` for a thread root); build it in the
    /// SAME order as `docs`. `covariates` is an optional per-document categorical group id in a
    /// DENSE range `0..num_groups` (the reversion anchor becomes that group's baseline prevalence);
    /// omit for a single global anchor. `covariate_labels` names the groups for the readouts.
    /// `min_count` drops words rarer than it. Experimental: requires `topica.enable_experimental()`.
    #[pyo3(signature = (docs, parents=None, covariates=None, covariate_labels=None, *, min_count=1))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        docs: Vec<Vec<String>>,
        parents: Option<Vec<i64>>,
        covariates: Option<Vec<usize>>,
        covariate_labels: Option<Vec<String>>,
        min_count: usize,
    ) -> PyResult<()> {
        require_experimental("ReplyTM")?;
        let n = docs.len();
        if n == 0 {
            return Err(PyValueError::new_err("docs is empty"));
        }

        // vocab: words with >= min_count occurrences, in first-seen order
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for doc in &docs {
            for w in doc {
                *counts.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        let mut vocab: Vec<String> = Vec::new();
        let mut wid: HashMap<&str, u32> = HashMap::new();
        for doc in &docs {
            for w in doc {
                if counts[w.as_str()] >= min_count && !wid.contains_key(w.as_str()) {
                    wid.insert(w.as_str(), vocab.len() as u32);
                    vocab.push(w.clone());
                }
            }
        }
        if vocab.is_empty() {
            return Err(PyValueError::new_err(
                "no words survive min_count; lower min_count",
            ));
        }
        let docs_id: Vec<Vec<u32>> = docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|w| wid.get(w.as_str()).copied())
                    .collect()
            })
            .collect();

        // parents: validate length + range + acyclicity (indices into the doc list; -1 = root)
        let par: Vec<i64> = match &parents {
            None => {
                // No reply tree: the model degenerates to a plain logistic-normal topic model
                // and the reply-structure parameters are undefined. Warn rather than silently
                // report a meaningless kappa/sigma2 (they come back NaN with no edges).
                PyErr::warn_bound(
                    py,
                    &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                    "ReplyTM.fit called with parents=None: no reply tree, so the model reduces to \
                     a plain logistic-normal topic model and kappa/sigma2 are undefined (NaN). \
                     Pass parents to use the reply structure.",
                    1,
                )?;
                vec![-1; n]
            }
            Some(p) => {
                if p.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "parents has {} entries but there are {n} documents",
                        p.len()
                    )));
                }
                for (d, &pd) in p.iter().enumerate() {
                    if pd < -1 || pd >= n as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] = {pd} out of range; must be -1 or in [0, {n})"
                        )));
                    }
                    if pd == d as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] points at itself"
                        )));
                    }
                }
                // acyclicity: walking parent links from any node must reach a root within n steps
                for start in 0..n {
                    let (mut cur, mut steps) = (start as i64, 0usize);
                    while cur >= 0 {
                        cur = p[cur as usize];
                        steps += 1;
                        if steps > n {
                            return Err(PyValueError::new_err(format!(
                                "parents contains a cycle reachable from document {start}; the \
                                 reply tree must be acyclic"
                            )));
                        }
                    }
                }
                p.clone()
            }
        };

        // covariate groups
        let (groups, num_groups, group_names) = match &covariates {
            None => (vec![0usize; n], 1usize, vec!["all".to_string()]),
            Some(g) => {
                if g.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "covariates has {} entries but there are {n} documents",
                        g.len()
                    )));
                }
                let ng = g.iter().copied().max().map(|m| m + 1).unwrap_or(1);
                // warn on a non-dense covariate (a gap creates an empty phantom group)
                for gid in 0..ng {
                    if !g.contains(&gid) {
                        PyErr::warn_bound(
                            py,
                            &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                            "covariates has a gap: group ids are not dense 0..num_groups, so an \
                             empty phantom group will appear in group_prevalence. Re-code the \
                             covariate to consecutive ids.",
                            1,
                        )?;
                        break;
                    }
                }
                let names = covariate_labels
                    .clone()
                    .unwrap_or_else(|| (0..ng).map(|i| format!("group{i}")).collect());
                if names.len() != ng {
                    return Err(PyValueError::new_err(format!(
                        "covariate_labels has {} names but the covariate has {ng} groups",
                        names.len()
                    )));
                }
                (g.clone(), ng, names)
            }
        };

        let k = slf.num_topics;
        let v = vocab.len();
        let iters = slf.em_iters;
        let seed = slf.seed;
        let m = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            crate::reply_tm::fit_reply_tm(
                &docs_id,
                &par,
                &groups,
                num_groups,
                k,
                v,
                iters,
                1e-6,
                |_, _, _| true,
                &mut rng,
            )
        });

        let dt = m.doc_topic();
        let gp = m.group_prevalence();
        slf.vocab = vocab;
        slf.group_names = group_names;
        slf.beta = m.beta;
        slf.doc_topic = dt;
        slf.group_prevalence = gp;
        slf.prevalence_se = m.anchor_se;
        slf.kappa = m.kappa;
        slf.kappa_ci = m.kappa_ci;
        slf.sigma2 = m.sigma2;
        slf.p0 = m.p0;
        slf.bound_history = m.bound_history;
        slf.fitted = true;
        Ok(())
    }

    /// K×V topic-word probability matrix.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.beta).to_pyarray_bound(py))
    }

    /// Number of topics K.
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// D×K document-topic proportions θ.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.doc_topic).to_pyarray_bound(py))
    }

    /// G×K per-group baseline topic prevalence (softmax of the covariate anchor): the descriptive
    /// mean topic mix of each covariate group's documents. See `prevalence_se` for uncertainty.
    #[getter]
    fn group_prevalence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.group_prevalence).to_pyarray_bound(py))
    }

    /// G×(K-1) method-of-composition standard error of the group prevalence anchor (in η space):
    /// combines the between-document sampling variance with the mean per-document posterior
    /// variance, so group contrasts can be reported with honest uncertainty.
    #[getter]
    fn prevalence_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.prevalence_se).to_pyarray_bound(py))
    }

    /// The covariate group labels (order matches `group_prevalence` rows).
    fn group_labels(&self) -> Vec<String> {
        self.group_names.clone()
    }

    /// The vocabulary (index order matches the `topic_word` columns).
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.vocab.clone())
    }

    /// Top-`n` words per topic. With `topic=None` (default) returns a list of lists for all K
    /// topics; with an integer `topic`, returns that one topic's words.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words(&self, py: Python<'_>, n: usize, topic: Option<usize>) -> PyResult<PyObject> {
        self.require_fitted()?;
        let one = |t: usize| -> Vec<String> {
            let row = &self.beta[t];
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.into_iter()
                .take(n)
                .map(|i| self.vocab[i].clone())
                .collect()
        };
        match topic {
            Some(t) => {
                if t >= self.num_topics {
                    return Err(PyValueError::new_err(format!(
                        "topic {t} out of range [0, {})",
                        self.num_topics
                    )));
                }
                Ok(one(t).into_py(py))
            }
            None => {
                let all: Vec<Vec<String>> = (0..self.num_topics).map(one).collect();
                Ok(all.into_py(py))
            }
        }
    }

    /// 95% profile-likelihood CI for the reversion `κ` as a `(lower, upper)` tuple; `(nan, nan)`
    /// when there are no reply edges. On real corpora this usually brackets 0 (persistence).
    #[getter]
    fn kappa_ci(&self) -> (f64, f64) {
        self.kappa_ci
    }

    /// Reversion strength `κ = 1 - a` (0 = pure persistence / parent-copy, 1 = no memory).
    #[getter]
    fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Per-edge diffusion variance `σ²`.
    #[getter]
    fn sigma2(&self) -> f64 {
        self.sigma2
    }

    /// Root prior variance.
    #[getter]
    fn p0(&self) -> f64 {
        self.p0
    }

    /// The variational-EM evidence-bound trace (one value per iteration).
    #[getter]
    fn bound_history(&self) -> Vec<f64> {
        self.bound_history.clone()
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named to match
    /// `__init__` (issue #400).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("em_iters", self.em_iters)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "ReplyTM(num_topics={}, fitted, kappa={:.3}, sigma2={:.3})",
                self.num_topics, self.kappa, self.sigma2
            )
        } else {
            format!("ReplyTM(num_topics={}, unfitted)", self.num_topics)
        }
    }
}
