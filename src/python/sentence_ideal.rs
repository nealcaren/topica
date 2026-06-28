//! IdealPointSentenceTM pyclass: a continuous ideal-point topic model over sentence (or
//! document) embeddings. Topics are Gaussian clusters whose centroids are displaced
//! by the author's latent position. The embedding-native analog of IdealPointTM.
//! Experimental, gated. `use super::*` pulls in the shared bindings.

use super::*;
use crate::sentence_ideal::{self, SentenceIdealModel};
use std::collections::HashMap;

#[pyclass(module = "topica")]
pub struct IdealPointSentenceTM {
    num_topics: usize,
    num_dims: usize,
    convergence_tol: f64,
    x_prior_variance: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    author_names: Vec<String>,
    model: Option<SentenceIdealModel>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SentenceIdealState {
    num_topics: usize,
    num_dims: usize,
    convergence_tol: f64,
    x_prior_variance: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    author_names: Vec<String>,
    dim: Option<usize>,
    num_authors: Option<usize>,
    mu: Option<Vec<Vec<f64>>>,
    v: Option<Vec<Vec<Vec<f64>>>>,
    x: Option<Vec<Vec<f64>>>,
    pi: Option<Vec<f64>>,
    sigma2: Option<f64>,
    resp: Option<Vec<Vec<f64>>>,
    group: Option<Vec<usize>>,
    log_likelihood: Option<f64>,
    ll_history: Option<Vec<f64>>,
    converged: Option<bool>,
    iters_run: Option<usize>,
}

impl IdealPointSentenceTM {
    fn fitted_model(&self) -> PyResult<&SentenceIdealModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl IdealPointSentenceTM {
    /// Create an unfitted model. `num_topics` is K (>= 2); `num_dims` the latent
    /// ideal-point dimensionality (default 1). `x_prior_variance` is the Gaussian
    /// prior on the positions (1.0 matches the unit-variance standardization).
    /// `convergence_tol` stops EM on the relative change in the log-likelihood.
    #[new]
    #[pyo3(signature = (num_topics, *, num_dims=1, convergence_tol=1e-4,
                        x_prior_variance=1.0, seed=42))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        num_dims: usize,
        convergence_tol: f64,
        x_prior_variance: f64,
        seed: u64,
    ) -> PyResult<Self> {
        require_experimental("IdealPointSentenceTM")?;
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if num_dims < 1 {
            return Err(PyValueError::new_err("num_dims must be >= 1"));
        }
        if !finite_pos(x_prior_variance) {
            return Err(PyValueError::new_err("x_prior_variance must be > 0"));
        }
        Ok(IdealPointSentenceTM {
            num_topics,
            num_dims,
            convergence_tol,
            x_prior_variance,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            author_names: Vec::new(),
            model: None,
        })
    }

    /// Fit on `embeddings` (an `(N, D)` array of per-observation sentence or document
    /// embeddings). `group` is an optional list of author labels (length N):
    /// observations sharing a label share one latent position; if omitted, each
    /// observation is its own author. `anchors` is an optional `{author_label: value}`
    /// mapping orienting the sign of the first latent dimension. `iters` sets the EM
    /// iteration cap (default 100).
    #[pyo3(signature = (embeddings, *, group=None, anchors=None, iters=None,
                        convergence_tol=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        embeddings: &Bound<'_, PyAny>,
        group: Option<Vec<String>>,
        anchors: Option<HashMap<String, f64>>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<()> {
        let emb = parse_features(embeddings)?;
        let n = emb.len();
        if n == 0 {
            return Err(PyValueError::new_err("embeddings is empty"));
        }
        check_all_finite_2d("embeddings", &emb)?;
        let dim = emb[0].len();
        if emb.iter().any(|e| e.len() != dim) {
            return Err(PyValueError::new_err(
                "embeddings rows must all have the same length",
            ));
        }

        let (group_idx, author_names): (Vec<usize>, Vec<String>) = match &group {
            Some(labels) => {
                if labels.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "group must have length N ({n}), got {}",
                        labels.len()
                    )));
                }
                let mut names: Vec<String> = labels.clone();
                names.sort();
                names.dedup();
                let index: HashMap<&str, usize> = names
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), i))
                    .collect();
                let idx: Vec<usize> = labels.iter().map(|l| index[l.as_str()]).collect();
                (idx, names)
            }
            None => (
                (0..n).collect(),
                (0..n).map(|i| format!("obs_{i}")).collect(),
            ),
        };
        let num_authors = author_names.len();
        if num_authors < 2 {
            return Err(PyValueError::new_err(
                "IdealPointSentenceTM needs at least 2 authors/observations to scale",
            ));
        }

        let anchor_pairs: Vec<(usize, f64)> = match &anchors {
            None => Vec::new(),
            Some(m) => {
                let mut pairs = Vec::with_capacity(m.len());
                for (label, &target) in m {
                    let i = author_names
                        .iter()
                        .position(|a| a == label)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "anchor label {label:?} is not an author label"
                            ))
                        })?;
                    pairs.push((i, target));
                }
                pairs
            }
        };

        let tol = convergence_tol.unwrap_or(self.convergence_tol);
        let it = iters.unwrap_or(100);
        let (k, dd, xpv, seed) = (
            self.num_topics,
            self.num_dims,
            self.x_prior_variance,
            self.seed,
        );
        let model = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            sentence_ideal::fit_sentence_ideal(
                &emb,
                &group_idx,
                num_authors,
                k,
                dd,
                &anchor_pairs,
                it,
                tol,
                xpv,
                &mut rng,
            )
        });

        self.model = Some(model);
        self.author_names = author_names;
        self.topic_names = (0..self.num_topics).map(|i| format!("topic_{i}")).collect();
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn num_dims(&self) -> usize {
        self.num_dims
    }
    #[getter]
    fn num_authors(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_authors)
    }
    /// Author positions (num_authors, num_dims), standardized to mean 0 / unit var.
    #[getter]
    fn author_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.x).to_pyarray_bound(py))
    }
    /// Standard error of each author position (num_authors, num_dims). The position
    /// is a linear-Gaussian least squares given the topic responsibilities, so this
    /// is the exact Laplace posterior SE, sqrt(diag(H_a^-1)); it shrinks with the
    /// number of the author's observations. Aligned to `author_positions`.
    #[getter]
    fn position_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let se = self.fitted_model()?.position_se(self.x_prior_variance);
        Ok(vecs_to_arr2(&se).to_pyarray_bound(py))
    }
    #[getter]
    fn author_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.author_names.clone())
    }
    /// Soft topic assignments per observation (N, num_topics); each row a simplex.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.resp).to_pyarray_bound(py))
    }
    /// Topic centroids at the neutral position (num_topics, embedding_dim).
    #[getter]
    fn topic_centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.mu).to_pyarray_bound(py))
    }
    /// Per-topic discrimination ||V_k|| (num_topics): how far the topic's centroid
    /// moves along the latent axis. Large = a strongly position-sensitive topic.
    #[getter]
    fn topic_discrimination<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.topic_discrimination()).to_pyarray_bound(py))
    }
    /// The displaced centroid `mu_k + sum_j x_j V_{k,j}` for `topic` at position `x`
    /// (length num_dims). Use to see where a topic's content sits at each end.
    fn position_centroid<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        x: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        if x.len() != self.num_dims {
            return Err(PyValueError::new_err("x must have length num_dims"));
        }
        Ok(Array1::from(m.position_topic_centroid(topic, &x)).to_pyarray_bound(py))
    }
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.log_likelihood)
    }
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .ll_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect())
    }
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(Some(self.fitted_model()?.converged))
    }
    #[getter]
    fn iters_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.iters_run)
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
        write_state(
            path,
            MODEL_TAG_SENTENCE_IDEAL,
            &SentenceIdealState {
                num_topics: self.num_topics,
                num_dims: self.num_dims,
                convergence_tol: self.convergence_tol,
                x_prior_variance: self.x_prior_variance,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                author_names: self.author_names.clone(),
                dim: m.map(|m| m.dim),
                num_authors: m.map(|m| m.num_authors),
                mu: m.map(|m| m.mu.clone()),
                v: m.map(|m| m.v.clone()),
                x: m.map(|m| m.x.clone()),
                pi: m.map(|m| m.pi.clone()),
                sigma2: m.map(|m| m.sigma2),
                resp: m.map(|m| m.resp.clone()),
                group: m.map(|m| m.group.clone()),
                log_likelihood: m.map(|m| m.log_likelihood),
                ll_history: m.map(|m| m.ll_history.clone()),
                converged: m.map(|m| m.converged),
                iters_run: m.map(|m| m.iters_run),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        require_experimental("IdealPointSentenceTM")?;
        let s: SentenceIdealState = read_state(path, MODEL_TAG_SENTENCE_IDEAL)?;
        let model = if s.fitted && s.mu.is_some() {
            Some(SentenceIdealModel {
                num_topics: s.num_topics,
                dim: s.dim.unwrap_or(0),
                num_dims: s.num_dims,
                num_authors: s.num_authors.unwrap_or(0),
                mu: s.mu.unwrap_or_default(),
                v: s.v.unwrap_or_default(),
                x: s.x.unwrap_or_default(),
                pi: s.pi.unwrap_or_default(),
                sigma2: s.sigma2.unwrap_or(1.0),
                resp: s.resp.unwrap_or_default(),
                group: s.group.unwrap_or_default(),
                log_likelihood: s.log_likelihood.unwrap_or(f64::NAN),
                ll_history: s.ll_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                iters_run: s.iters_run.unwrap_or(0),
            })
        } else {
            None
        };
        Ok(IdealPointSentenceTM {
            num_topics: s.num_topics,
            num_dims: s.num_dims,
            convergence_tol: s.convergence_tol,
            x_prior_variance: s.x_prior_variance,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            author_names: s.author_names,
            model,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "IdealPointSentenceTM(num_topics={}, num_dims={}, fitted={})",
            self.num_topics, self.num_dims, self.fitted
        )
    }
}
