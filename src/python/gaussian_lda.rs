//! Python bindings for GaussianLDA (Das, Zaheer & Dyer, ACL 2015). Mirrors the
//! embedding-input surface of ETM: `fit(data, word_embeddings, vocabulary, *, iters=)`.

use super::*;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn default_init() -> String {
    "kmeans".to_string()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GaussianLDAState {
    num_topics: usize,
    alpha: Option<f64>,
    kappa: f64,
    nu: Option<f64>,
    psi_scale: f64,
    #[serde(default = "default_init")]
    init: String,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
    embedding_dim: Option<usize>,
    topic_word: Option<Vec<Vec<f64>>>,
    word_log_density: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    topic_means: Option<Vec<Vec<f64>>>,
    topic_scale_chol: Option<Vec<Vec<f64>>>,
    topic_counts: Option<Vec<usize>>,
    kappa0: Option<f64>,
    nu0: Option<f64>,
    fit_history: Option<Vec<(usize, f64)>>,
    converged: Option<bool>,
}

/// Gaussian LDA (Das, Zaheer & Dyer, ACL 2015): LDA where each topic is a **Gaussian
/// over the word-embedding space** instead of a categorical over the vocabulary. A
/// token is generated from its topic's multivariate Gaussian on the word's embedding,
/// under a Normal-Inverse-Wishart conjugate prior; inference is collapsed Gibbs with a
/// Student-t posterior predictive (rank-1 Cholesky up/downdates as tokens move). You
/// bring the word embeddings; topics generalize over semantically similar words.
/// `topic_word` is derived by scoring each vocabulary word under each topic Gaussian.
#[pyclass(module = "topica")]
pub struct GaussianLDA {
    num_topics: usize,
    alpha: Option<f64>,
    kappa: f64,
    nu: Option<f64>,
    psi_scale: f64,
    init: String,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<crate::gaussian_lda::GaussianLDAModel>,
    corpus: Option<corpus::Corpus>,
}

impl GaussianLDA {
    fn fitted_model(&self) -> PyResult<&crate::gaussian_lda::GaussianLDAModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Effective alpha (1/K when the constructor value is None).
    fn eff_alpha(&self) -> f64 {
        self.alpha.unwrap_or(1.0 / self.num_topics as f64)
    }
}

#[pymethods]
impl GaussianLDA {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Create an unfitted model. ``num_topics`` is the number of topics K. The
    /// Normal-Inverse-Wishart prior follows the reference defaults (Das et al.):
    /// ``alpha`` is the symmetric document-topic Dirichlet concentration (default
    /// ``1/K``); ``kappa`` is the prior mean concentration kappa_0 (default 0.1);
    /// ``nu`` is the prior degrees of freedom nu_0 (default the embedding dimension E,
    /// and always clamped to >= E); ``psi_scale`` sets the prior scale matrix
    /// ``Psi_0 = psi_scale * E * I`` (default 3.0). ``init`` selects topic
    /// initialization: ``"kmeans"`` (default; k-means over the vocabulary embeddings,
    /// the paper's approach, which avoids the mode-collapse of random init) or
    /// ``"random"`` (per-token uniform, the reference Cholesky sampler's behavior).
    /// ``seed`` is the RNG seed.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=None, kappa=0.1, nu=None, psi_scale=3.0, init="kmeans".to_string(), seed=13))]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha: Option<f64>,
        kappa: f64,
        nu: Option<f64>,
        psi_scale: f64,
        init: String,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if init != "kmeans" && init != "random" {
            return Err(PyValueError::new_err(
                "init must be \"kmeans\" or \"random\"",
            ));
        }
        if let Some(a) = alpha {
            if !(a.is_finite() && a > 0.0) {
                return Err(PyValueError::new_err("alpha must be finite and > 0"));
            }
        }
        if !(kappa.is_finite() && kappa > 0.0) {
            return Err(PyValueError::new_err("kappa must be finite and > 0"));
        }
        if let Some(n) = nu {
            if !(n.is_finite() && n > 0.0) {
                return Err(PyValueError::new_err("nu must be finite and > 0"));
            }
        }
        if !(psi_scale.is_finite() && psi_scale > 0.0) {
            return Err(PyValueError::new_err("psi_scale must be finite and > 0"));
        }
        Ok(GaussianLDA {
            num_topics,
            alpha,
            kappa,
            nu,
            psi_scale,
            init,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `word_embeddings`
    /// (`(len(vocabulary), E)`) and the aligned `vocabulary`, which defines the word
    /// ids. Tokens outside the vocabulary are dropped. `iters` sets the number of
    /// Gibbs sweeps (default 100).
    #[pyo3(signature = (data, word_embeddings, vocabulary, *, iters=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        word_embeddings: &Bound<'_, PyAny>,
        vocabulary: Vec<String>,
        iters: Option<usize>,
    ) -> PyResult<Py<Self>> {
        let docs_str: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
                .docs
                .iter()
                .map(|d| {
                    d.iter()
                        .map(|&w| c.inner.id_to_word[w as usize].clone())
                        .collect()
                })
                .collect()
        } else {
            data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?
        };
        let rho = parse_features(word_embeddings)?;
        if rho.len() != vocabulary.len() {
            return Err(PyValueError::new_err(format!(
                "word_embeddings has {} rows but vocabulary has {} words",
                rho.len(),
                vocabulary.len()
            )));
        }
        check_all_finite_2d("word_embeddings", &rho)?;
        if vocabulary.is_empty() {
            return Err(PyValueError::new_err("vocabulary must be non-empty"));
        }
        let map: std::collections::HashMap<&str, u32> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let docs_ids: Vec<Vec<u32>> = docs_str
            .iter()
            .map(|d| {
                d.iter()
                    .filter_map(|w| map.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        if docs_ids.iter().all(|d| d.is_empty()) {
            return Err(PyValueError::new_err(
                "no in-vocabulary tokens in the documents",
            ));
        }

        let (num_topics, alpha, kappa, psi_scale) =
            (slf.num_topics, slf.eff_alpha(), slf.kappa, slf.psi_scale);
        let nu0 = slf.nu.unwrap_or(0.0); // 0.0 -> core clamps to E
        let use_kmeans = slf.init == "kmeans";
        let it = iters.unwrap_or(100);
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let model = py.allow_threads(move || {
            crate::gaussian_lda::fit(
                &docs_ids, &rho, num_topics, alpha, kappa, nu0, psi_scale, it, use_kmeans, &mut rng,
            )
        });

        // Build a corpus aligned to `vocabulary` for coherence / top_words / vocabulary.
        let n = docs_str.len();
        let v = vocabulary.len();
        let mut df = vec![0u32; v];
        let mut tf = vec![0u32; v];
        let mut id_docs: Vec<Vec<u32>> = Vec::with_capacity(n);
        for doc in &docs_str {
            let ids: Vec<u32> = doc
                .iter()
                .filter_map(|w| map.get(w.as_str()).copied())
                .collect();
            let mut seen = std::collections::HashSet::new();
            for &id in &ids {
                tf[id as usize] += 1;
                seen.insert(id as usize);
            }
            for id in seen {
                df[id] += 1;
            }
            id_docs.push(ids);
        }
        slf.corpus = Some(corpus::Corpus {
            id_to_word: vocabulary,
            docs: id_docs,
            doc_names: (0..n).map(|i| format!("doc_{i}")).collect(),
            doc_labels: vec![String::new(); n],
            doc_freqs: df,
            total_freqs: tf,
        });
        slf.topic_names = (0..num_topics).map(|i| format!("topic_{i}")).collect();
        slf.model = Some(model);
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Uniform constructor-config introspection (#400).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("kappa", self.kappa)?;
        d.set_item("nu", self.nu)?;
        d.set_item("psi_scale", self.psi_scale)?;
        d.set_item("init", &self.init)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    // --- Required analysis surface ---
    /// Derived topic-word matrix (num_topics, vocab): each vocabulary word scored under
    /// each topic's Student-t density and softmax-normalized over the vocabulary. A
    /// convenience view (topics are Gaussians, not multinomials), not a sampled parameter.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Document-topic proportions (num_docs, num_topics), each row a distribution.
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
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.num_topics {
            return Err(PyValueError::new_err(format!(
                "expected {} topic names, got {}",
                self.num_topics,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }

    // --- Model-specific outputs ---
    /// Per-topic Gaussian means mu_k (num_topics, E).
    #[getter]
    fn topic_means<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_means).to_pyarray_bound(py))
    }
    /// Per-topic NIW **scale** matrices Psi_k (num_topics, E, E) — exactly what the
    /// reference writes (chol(Psi_k) Lᵀ). This is the scale matrix, NOT the covariance;
    /// see `topic_covariances`.
    #[getter]
    fn topic_scale_matrices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        Ok(self.psi_matrices(py, false))
    }
    /// Per-topic covariances (num_topics, E, E): the Inverse-Wishart posterior-mean
    /// covariance Sigma_k = Psi_k / (nu_k - E - 1) (defined when nu_k > E + 1).
    #[getter]
    fn topic_covariances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        Ok(self.psi_matrices(py, true))
    }
    /// Number of tokens assigned to each topic (num_topics,).
    #[getter]
    fn topic_counts(&self) -> PyResult<Vec<usize>> {
        Ok(self.fitted_model()?.topic_counts.clone())
    }
    /// The reference `avgLL` diagnostic per sweep: mean per-token Gaussian log-density
    /// at the current topic assignment (covariance Psi_k/(nu_k - E)). NOT the model
    /// evidence (drops the Dirichlet term) and not guaranteed monotone. Length iters+1
    /// (entry 0 is the post-initialization value).
    #[getter]
    fn log_likelihood_history(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .fitted_model()?
            .fit_history
            .iter()
            .map(|&(_, v)| v)
            .collect())
    }
    /// Per-sweep `(iteration, avgLL)` trace (see `log_likelihood_history`).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
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
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Infer topic proportions for new documents (closed-vocabulary): scores tokens
    /// under the fitted topic Gaussians via the fitted `vocabulary`'s embeddings and
    /// runs `iters` Gibbs sweeps holding the topics fixed. `word_embeddings` is accepted
    /// for API symmetry but ignored (the fitted vocabulary defines the embeddings);
    /// out-of-vocabulary words are dropped. Returns (num_docs, num_topics).
    #[pyo3(signature = (data, word_embeddings=None, *, iters=50))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        word_embeddings: Option<&Bound<'_, PyAny>>,
        iters: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = word_embeddings; // closed-vocabulary v1
        let model = self.fitted_model()?;
        let corpus = self.corpus.as_ref().unwrap();
        let docs_str: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
                .docs
                .iter()
                .map(|d| {
                    d.iter()
                        .map(|&w| c.inner.id_to_word[w as usize].clone())
                        .collect()
                })
                .collect()
        } else {
            data.extract().map_err(|_| {
                PyValueError::new_err("transform() expects a Corpus or a list of token lists")
            })?
        };
        let map: std::collections::HashMap<&str, u32> = corpus
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let docs_ids: Vec<Vec<u32>> = docs_str
            .iter()
            .map(|d| {
                d.iter()
                    .filter_map(|w| map.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        let alpha = self.eff_alpha();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let dt = py.allow_threads(move || {
            crate::gaussian_lda::transform(model, &docs_ids, alpha, iters, &mut rng)
        });
        Ok(vecs_to_arr2(&dt).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (topica's binary format).
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_GAUSSIAN_LDA,
            &GaussianLDAState {
                num_topics: self.num_topics,
                alpha: self.alpha,
                kappa: self.kappa,
                nu: self.nu,
                psi_scale: self.psi_scale,
                init: self.init.clone(),
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                corpus: self.corpus.clone(),
                embedding_dim: Some(m.embedding_dim),
                topic_word: Some(m.topic_word.clone()),
                word_log_density: Some(m.word_log_density.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                topic_means: Some(m.topic_means.clone()),
                topic_scale_chol: Some(m.topic_scale_chol.clone()),
                topic_counts: Some(m.topic_counts.clone()),
                kappa0: Some(m.kappa0),
                nu0: Some(m.nu0),
                fit_history: Some(m.fit_history.clone()),
                converged: Some(m.converged),
            },
        )
    }

    /// Load a model saved with [`save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: GaussianLDAState = read_state(path, MODEL_TAG_GAUSSIAN_LDA)?;
        let model = if s.fitted {
            Some(crate::gaussian_lda::GaussianLDAModel {
                num_topics: s.num_topics,
                embedding_dim: s.embedding_dim.unwrap_or(0),
                topic_word: s.topic_word.unwrap_or_default(),
                word_log_density: s.word_log_density.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                topic_means: s.topic_means.unwrap_or_default(),
                topic_scale_chol: s.topic_scale_chol.unwrap_or_default(),
                topic_counts: s.topic_counts.unwrap_or_default(),
                kappa0: s.kappa0.unwrap_or(0.1),
                nu0: s.nu0.unwrap_or(0.0),
                fit_history: s.fit_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
            })
        } else {
            None
        };
        Ok(GaussianLDA {
            num_topics: s.num_topics,
            alpha: s.alpha,
            kappa: s.kappa,
            nu: s.nu,
            psi_scale: s.psi_scale,
            init: s.init,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianLDA(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}

impl GaussianLDA {
    /// Build (num_topics, E, E) matrices from the stored Cholesky factors: L Lᵀ = Psi_k,
    /// optionally divided by (nu_k - E - 1) to give the posterior-mean covariance.
    fn psi_matrices<'py>(&self, py: Python<'py>, as_cov: bool) -> Bound<'py, PyArray3<f64>> {
        let m = self.model.as_ref().unwrap();
        let k = m.num_topics;
        let e = m.embedding_dim;
        let mut out = vec![0.0f64; k * e * e];
        for t in 0..k {
            let l = &m.topic_scale_chol[t];
            let denom = if as_cov {
                (m.nu0 + m.topic_counts[t] as f64 - e as f64 - 1.0).max(1e-12)
            } else {
                1.0
            };
            for i in 0..e {
                for j in 0..e {
                    // (L Lᵀ)_{ij} = sum_{c<=min(i,j)} L_ic L_jc
                    let mut s = 0.0;
                    for c in 0..=i.min(j) {
                        s += l[i * e + c] * l[j * e + c];
                    }
                    out[t * e * e + i * e + j] = s / denom;
                }
            }
        }
        let arr = Array3::from_shape_vec((k, e, e), out).unwrap();
        arr.to_pyarray_bound(py)
    }
}
