//! Python bindings for ReplyTM (reply-conditioned topic model, #810). Mirrors the
//! CSATM binding shape (shared reply-tree `parents` contract); see
//! .github/CONTRIBUTING-MODELS.md section B2. Experimental (topica-original).

use super::*;
use crate::reply_tm::{CovResponse, ReplyTmParams};
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// ReplyTM: a reply-conditioned topic model for threaded discussions. A child
/// comment's topic prior is shifted by a learned, directed **response matrix**
/// `T` applied to its parent's topic proportions, plus a per-group (covariate)
/// baseline: `a_child = exp(b_g) + rho_g · T_gᵀ z̄_parent`. `T_g[i, j]` reads as the
/// response mass a topic-`i` parent places on child topic `j`, reported per
/// covariate group with posterior credible intervals. Fit by collapsed Gibbs with
/// `T`, `rho`, and the baseline sampled (Metropolis-within-Gibbs). Topica-original,
/// experimental. Reference: issue #810.
#[pyclass(module = "topica")]
pub struct ReplyTM {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    covariate_response: String,
    response_link: String,
    t_inference: String,
    seed: u64,
    fitted: bool,
    covariate_labels: Vec<String>,
    model: Option<crate::reply_tm::ReplyTmModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ReplyTmState {
    num_topics: usize,
    alpha: f64,
    beta: f64,
    covariate_response: String,
    response_link: String,
    t_inference: String,
    seed: u64,
    fitted: bool,
    covariate_labels: Vec<String>,
    corpus: Option<corpus::Corpus>,
    num_groups: usize,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    response_matrix: Option<Vec<Vec<Vec<f64>>>>,
    response_matrix_lo: Option<Vec<Vec<Vec<f64>>>>,
    response_matrix_hi: Option<Vec<Vec<Vec<f64>>>>,
    response_strength: Option<Vec<f64>>,
    response_strength_lo: Option<Vec<f64>>,
    response_strength_hi: Option<Vec<f64>>,
    baseline: Option<Vec<Vec<f64>>>,
    baseline_lo: Option<Vec<Vec<f64>>>,
    baseline_hi: Option<Vec<Vec<f64>>>,
    alpha_mean: Option<Vec<f64>>,
    parent_support: Option<Vec<Vec<f64>>>,
    doc_lengths: Option<Vec<usize>>,
    fit_history: Option<Vec<(usize, f64)>>,
    #[serde(default = "nan")]
    max_rhat: f64,
    converged: bool,
}

fn nan() -> f64 {
    f64::NAN
}

impl ReplyTM {
    fn fitted_model(&self) -> PyResult<&crate::reply_tm::ReplyTmModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    fn cov_response(&self) -> PyResult<CovResponse> {
        match self.covariate_response.as_str() {
            "per_group" => Ok(CovResponse::PerGroup),
            "shared_shape" => Ok(CovResponse::SharedShape),
            "global" => Ok(CovResponse::Global),
            other => Err(PyValueError::new_err(format!(
                "covariate_response must be 'per_group', 'shared_shape', or 'global'; got '{other}'"
            ))),
        }
    }
}

#[pymethods]
impl ReplyTM {
    #[new]
    #[pyo3(signature = (
        num_topics, *, alpha=0.5, beta=0.01,
        covariate_response="per_group".to_string(),
        response_link="simplex".to_string(),
        t_inference="sampled".to_string(),
        seed=13,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: f64,
        beta: f64,
        covariate_response: String,
        response_link: String,
        t_inference: String,
        seed: u64,
    ) -> PyResult<Self> {
        require_experimental("ReplyTM")?;
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(PyValueError::new_err("alpha must be finite and > 0"));
        }
        if !(beta.is_finite() && beta > 0.0) {
            return Err(PyValueError::new_err("beta must be finite and > 0"));
        }
        if !matches!(
            covariate_response.as_str(),
            "per_group" | "shared_shape" | "global"
        ) {
            return Err(PyValueError::new_err(
                "covariate_response must be 'per_group', 'shared_shape', or 'global'",
            ));
        }
        // v1 implements the additive-simplex response and fully-sampled T; the
        // documented alternatives are reserved but not yet implemented.
        if response_link != "simplex" {
            return Err(PyValueError::new_err(
                "response_link='loglinear' (signed/inhibition) is not implemented yet; use 'simplex'",
            ));
        }
        if t_inference != "sampled" {
            return Err(PyValueError::new_err(
                "t_inference='map' is not implemented yet; use 'sampled'",
            ));
        }
        Ok(ReplyTM {
            num_topics,
            alpha,
            beta,
            covariate_response,
            response_link,
            t_inference,
            seed,
            fitted: false,
            covariate_labels: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit ReplyTM. `data` is a `Corpus` or a list of token lists. `parents` is a
    /// per-document list of parent document **indices** (`-1` for a thread root);
    /// when omitted every document is a root and, with one group, the fit reduces
    /// to LDA. `covariate` is a per-document integer group label (0-based) selecting
    /// the response matrix `T_g`; when omitted a single group is used. Both arrays
    /// are indexed in the SAME order as `docs` (they are realigned if empty
    /// documents are pruned).
    #[pyo3(signature = (data, parents=None, covariate=None, *, covariate_labels=None, iters=1000, num_threads=None, num_chains=None, mh_steps=None, mh_step_sd=None, burn=None, rho_prior=None, t_prior_sd=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        parents: Option<Vec<i64>>,
        covariate: Option<Vec<i64>>,
        covariate_labels: Option<Vec<String>>,
        iters: usize,
        num_threads: Option<usize>,
        num_chains: Option<usize>,
        mh_steps: Option<usize>,
        mh_step_sd: Option<f64>,
        burn: Option<usize>,
        rho_prior: Option<(f64, f64)>,
        t_prior_sd: Option<f64>,
    ) -> PyResult<Py<Self>> {
        require_experimental("ReplyTM")?;
        if let Some(nc) = num_chains {
            if nc < 1 {
                return Err(PyValueError::new_err("num_chains must be >= 1"));
            }
        }
        if let Some(sd) = mh_step_sd {
            if !(sd > 0.0 && sd.is_finite()) {
                return Err(PyValueError::new_err(
                    "mh_step_sd must be a positive finite float",
                ));
            }
        }
        if let Some(sd) = t_prior_sd {
            if !(sd > 0.0 && sd.is_finite()) {
                return Err(PyValueError::new_err(
                    "t_prior_sd must be a positive finite float",
                ));
            }
        }
        if let Some((a, b)) = rho_prior {
            if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
                return Err(PyValueError::new_err(
                    "rho_prior must be (mean, sd) with both entries positive and finite",
                ));
            }
        }
        let (corpus, kept, expected_len): (corpus::Corpus, Vec<usize>, usize) =
            if let Ok(c) = data.extract::<Corpus>() {
                let inner = c.inner;
                let n = inner.num_docs();
                (inner, (0..n).collect(), n)
            } else {
                let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                    PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                })?;
                let orig = docs.len();
                let (cp, kept) = build_corpus_from_docs(
                    docs,
                    None,
                    None,
                    std::collections::HashSet::new(),
                    1,
                    1.0,
                    0,
                    0,
                )?;
                (cp, kept, orig)
            };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        // Parents: validate (indices into the ORIGINAL list) and remap to the
        // surviving-document space. Same contract as CSATM.
        let parents_remapped: Vec<i64> = match parents {
            None => Vec::new(),
            Some(p) => {
                if p.len() != expected_len {
                    return Err(PyValueError::new_err(format!(
                        "parents has {} entries but there are {} documents",
                        p.len(),
                        expected_len
                    )));
                }
                for (d, &par) in p.iter().enumerate() {
                    if par < -1 || par >= expected_len as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] = {par} is out of range; must be -1 (a thread root) \
                             or a document index in [0, {expected_len})"
                        )));
                    }
                    if par == d as i64 {
                        return Err(PyValueError::new_err(format!(
                            "parents[{d}] = {d} points at itself; a document cannot be its own parent"
                        )));
                    }
                }
                for start in 0..expected_len {
                    let mut cur = start as i64;
                    let mut steps = 0usize;
                    while cur != -1 {
                        cur = p[cur as usize];
                        steps += 1;
                        if steps > expected_len {
                            return Err(PyValueError::new_err(format!(
                                "parents contains a cycle reachable from document {start}; \
                                 the reply structure must be a forest (acyclic)"
                            )));
                        }
                    }
                }
                let mut old_to_new = vec![-1i64; expected_len];
                for (new_idx, &old) in kept.iter().enumerate() {
                    old_to_new[old] = new_idx as i64;
                }
                kept.iter()
                    .map(|&old| {
                        let par = p[old];
                        if par >= 0 && (par as usize) < expected_len {
                            old_to_new[par as usize]
                        } else {
                            -1
                        }
                    })
                    .collect()
            }
        };

        // Covariate: realign through `kept`; validate non-negative.
        let covariate_remapped: Vec<i64> = match covariate {
            None => Vec::new(),
            Some(c) => {
                if c.len() != expected_len {
                    return Err(PyValueError::new_err(format!(
                        "covariate has {} entries but there are {} documents",
                        c.len(),
                        expected_len
                    )));
                }
                for (d, &g) in c.iter().enumerate() {
                    if g < 0 {
                        return Err(PyValueError::new_err(format!(
                            "covariate[{d}] = {g} is negative; group labels must be 0-based non-negative integers"
                        )));
                    }
                }
                kept.iter().map(|&old| c[old]).collect()
            }
        };

        let defaults = ReplyTmParams::default();
        let params = ReplyTmParams {
            num_topics: slf.num_topics,
            alpha: slf.alpha,
            beta: slf.beta,
            covariate_response: slf.cov_response()?,
            num_threads: num_threads.unwrap_or(1).max(1),
            num_chains: num_chains.unwrap_or(defaults.num_chains).max(1),
            mh_steps: mh_steps.unwrap_or(defaults.mh_steps),
            mh_step_sd: mh_step_sd.unwrap_or(defaults.mh_step_sd),
            burn: burn.unwrap_or(defaults.burn),
            rho_prior: rho_prior.unwrap_or(defaults.rho_prior),
            t_prior_sd: t_prior_sd.unwrap_or(defaults.t_prior_sd),
        };
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let (model, corpus) = py.allow_threads(move || {
            let model = crate::reply_tm::fit(
                &corpus,
                &parents_remapped,
                &covariate_remapped,
                &params,
                iters,
                &mut rng,
            );
            (model, corpus)
        });
        // Group labels: use the caller's names if given (must match the group count),
        // else positional `group_0`, `group_1`, ….
        let ng = model.num_groups;
        slf.covariate_labels = match covariate_labels {
            Some(labels) => {
                if labels.len() != ng {
                    return Err(PyValueError::new_err(format!(
                        "covariate_labels has {} entries but the fit has {} covariate group(s)",
                        labels.len(),
                        ng
                    )));
                }
                labels
            }
            None => (0..ng).map(|g| format!("group_{g}")).collect(),
        };
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("covariate_response", &self.covariate_response)?;
        d.set_item("response_link", &self.response_link)?;
        d.set_item("t_inference", &self.t_inference)?;
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
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.alpha_mean.clone()).to_pyarray_bound(py))
    }

    // --- ReplyTM estimands ---
    #[getter]
    fn num_groups(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_groups)
    }
    /// Response matrices `T_g`, a list of (K, K) arrays (one per covariate group),
    /// each row on the simplex. `T_g[i, j]` = response mass a topic-`i` parent
    /// places on child topic `j`.
    #[getter]
    fn response_matrix<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let m = self.fitted_model()?;
        Ok(m.response_matrix
            .iter()
            .map(|g| vecs_to_arr2(g).to_pyarray_bound(py))
            .collect())
    }
    /// 2.5% credible bound of each `T_g` cell (list of (K, K) arrays).
    #[getter]
    fn response_matrix_lower<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let m = self.fitted_model()?;
        Ok(m.response_matrix_lo
            .iter()
            .map(|g| vecs_to_arr2(g).to_pyarray_bound(py))
            .collect())
    }
    /// 97.5% credible bound of each `T_g` cell (list of (K, K) arrays).
    #[getter]
    fn response_matrix_upper<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let m = self.fitted_model()?;
        Ok(m.response_matrix_hi
            .iter()
            .map(|g| vecs_to_arr2(g).to_pyarray_bound(py))
            .collect())
    }
    /// Response strength `rho_g` per group (posterior mean), with credible bounds.
    #[getter]
    fn response_strength<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.response_strength.clone()).to_pyarray_bound(py))
    }
    #[getter]
    fn response_strength_lower<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.response_strength_lo.clone()).to_pyarray_bound(py))
    }
    #[getter]
    fn response_strength_upper<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.response_strength_hi.clone()).to_pyarray_bound(py))
    }
    /// Baseline concentration `exp(b_g)` per group, a (G, K) array, with bounds.
    #[getter]
    fn baseline<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.baseline).to_pyarray_bound(py))
    }
    #[getter]
    fn baseline_lower<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.baseline_lo).to_pyarray_bound(py))
    }
    #[getter]
    fn baseline_upper<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.baseline_hi).to_pyarray_bound(py))
    }
    /// Group index -> label (the `covariate_labels` passed to `fit`, else positional
    /// `group_0`, `group_1`, …).
    #[getter]
    fn group_labels(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.covariate_labels.clone())
    }
    /// Convergence trace: `(sweep, corpus topic-word log-likelihood)` pairs. The
    /// log-likelihood should rise and plateau; a still-climbing tail means more
    /// `iters` are needed before trusting the credible intervals.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }
    /// `max_rhat < 1.1`: the sampled parameters (T cells, rho, baseline) mixed across
    /// chains. `False` (and `max_rhat` NaN) when fit with `num_chains=1` — a single
    /// chain has no convergence diagnostic.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }
    /// Maximum split-R̂ over all sampled scalars across chains (≈1 at convergence,
    /// `> 1.1` flags a chain that has not mixed). NaN when `num_chains=1`.
    #[getter]
    fn max_rhat(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.max_rhat)
    }
    /// Parent-topic support, a (num_groups, K) array: for each group and topic `i`,
    /// the total parent proportion mass on topic `i` over that group's reply edges.
    /// Row `i` of `T_g` is only identified where this is non-trivial;
    /// `topica.inspect.response_contrast` uses it to suppress cells whose apparent
    /// group difference is really a prevalence (support) difference.
    #[getter]
    fn parent_support<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.parent_support).to_pyarray_bound(py))
    }

    // --- Conventional extras ---
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        topic_words_helper(
            py,
            &phi,
            &self.corpus.as_ref().unwrap().id_to_word,
            self.num_topics,
            n,
            topic,
            weights,
        )
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    // --- save / load ---
    fn save(&self, path: &str) -> PyResult<()> {
        let state = ReplyTmState {
            num_topics: self.num_topics,
            alpha: self.alpha,
            beta: self.beta,
            covariate_response: self.covariate_response.clone(),
            response_link: self.response_link.clone(),
            t_inference: self.t_inference.clone(),
            seed: self.seed,
            fitted: self.fitted,
            covariate_labels: self.covariate_labels.clone(),
            corpus: self.corpus.clone(),
            num_groups: self.model.as_ref().map(|m| m.num_groups).unwrap_or(0),
            topic_word: self.model.as_ref().map(|m| m.topic_word.clone()),
            doc_topic: self.model.as_ref().map(|m| m.doc_topic.clone()),
            response_matrix: self.model.as_ref().map(|m| m.response_matrix.clone()),
            response_matrix_lo: self.model.as_ref().map(|m| m.response_matrix_lo.clone()),
            response_matrix_hi: self.model.as_ref().map(|m| m.response_matrix_hi.clone()),
            response_strength: self.model.as_ref().map(|m| m.response_strength.clone()),
            response_strength_lo: self.model.as_ref().map(|m| m.response_strength_lo.clone()),
            response_strength_hi: self.model.as_ref().map(|m| m.response_strength_hi.clone()),
            baseline: self.model.as_ref().map(|m| m.baseline.clone()),
            baseline_lo: self.model.as_ref().map(|m| m.baseline_lo.clone()),
            baseline_hi: self.model.as_ref().map(|m| m.baseline_hi.clone()),
            alpha_mean: self.model.as_ref().map(|m| m.alpha_mean.clone()),
            parent_support: self.model.as_ref().map(|m| m.parent_support.clone()),
            doc_lengths: self.model.as_ref().map(|m| m.doc_lengths.clone()),
            fit_history: self.model.as_ref().map(|m| m.fit_history.clone()),
            max_rhat: self.model.as_ref().map(|m| m.max_rhat).unwrap_or(f64::NAN),
            converged: self.model.as_ref().map(|m| m.converged).unwrap_or(false),
        };
        let bytes = bincode::serialize(&state)
            .map_err(|e| PyRuntimeError::new_err(format!("serialize failed: {e}")))?;
        std::fs::write(path, bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("write failed: {e}")))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        require_experimental("ReplyTM")?;
        let bytes = std::fs::read(path)
            .map_err(|e| PyRuntimeError::new_err(format!("read failed: {e}")))?;
        let state: ReplyTmState = bincode::deserialize(&bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("deserialize failed: {e}")))?;
        let model = if state.fitted {
            Some(crate::reply_tm::ReplyTmModel {
                num_topics: state.num_topics,
                num_groups: state.num_groups,
                topic_word: state.topic_word.unwrap_or_default(),
                doc_topic: state.doc_topic.unwrap_or_default(),
                response_matrix: state.response_matrix.unwrap_or_default(),
                response_matrix_lo: state.response_matrix_lo.unwrap_or_default(),
                response_matrix_hi: state.response_matrix_hi.unwrap_or_default(),
                response_strength: state.response_strength.unwrap_or_default(),
                response_strength_lo: state.response_strength_lo.unwrap_or_default(),
                response_strength_hi: state.response_strength_hi.unwrap_or_default(),
                baseline: state.baseline.unwrap_or_default(),
                baseline_lo: state.baseline_lo.unwrap_or_default(),
                baseline_hi: state.baseline_hi.unwrap_or_default(),
                alpha_mean: state.alpha_mean.unwrap_or_default(),
                parent_support: state.parent_support.unwrap_or_default(),
                doc_lengths: state.doc_lengths.unwrap_or_default(),
                fit_history: state.fit_history.clone().unwrap_or_default(),
                max_rhat: state.max_rhat,
                converged: state.converged,
            })
        } else {
            None
        };
        Ok(ReplyTM {
            num_topics: state.num_topics,
            alpha: state.alpha,
            beta: state.beta,
            covariate_response: state.covariate_response,
            response_link: state.response_link,
            t_inference: state.t_inference,
            seed: state.seed,
            fitted: state.fitted,
            covariate_labels: state.covariate_labels,
            model,
            corpus: state.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ReplyTM(num_topics={}, covariate_response='{}', fitted={})",
            self.num_topics, self.covariate_response, self.fitted
        )
    }
}
