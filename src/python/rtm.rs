//! Python bindings for RTM (Chang & Blei, AOAS 2010).

use super::*;
use crate::rtm::Link;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;
use pyo3::types::PyType;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// RTM: the Relational Topic Model (Chang & Blei, "Hierarchical Relational Models
/// for Document Networks", AOAS 2010). LDA plus a link model: for each observed
/// pair of documents a binary link is drawn from a function of the two documents'
/// mean topic assignments, so the same topics explain both words and links. Fit
/// with ``fit(docs, links=edges)`` on a document graph (citations, hyperlinks,
/// co-sponsorship, adjacency); predict links from words for unseen documents with
/// ``suggest_links``. Undirected links; ``link="logistic"`` (default) or
/// ``"exponential"``.
#[pyclass(module = "topica")]
pub struct RTM {
    num_topics: usize,
    link: Link,
    alpha: Option<f64>,
    beta: f64,
    rho: Option<f64>,
    negative_ratio: f64,
    ridge: f64,
    // "variational" (default, the shipped EM) or "gibbs" (collapsed Gibbs, R lda
    // rtm.collapsed.gibbs.sampler / rtm.em same-algorithm parity).
    inference: String,
    seed: u64,
    fitted: bool,
    model: Option<crate::rtm::RTMModel>,
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RtmState {
    num_topics: usize,
    link: String,
    alpha: Option<f64>,
    #[serde(default = "rtm_default_beta")]
    beta: f64,
    rho: Option<f64>,
    negative_ratio: f64,
    ridge: f64,
    #[serde(default = "rtm_default_inference")]
    inference: String,
    seed: u64,
    fitted: bool,
    corpus: Option<corpus::Corpus>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    phi_bar: Option<Vec<Vec<f64>>>,
    eta: Option<Vec<f64>>,
    nu: Option<f64>,
    fit_history: Option<Vec<(usize, f64)>>,
    converged: Option<bool>,
}

fn rtm_default_inference() -> String {
    "variational".to_string()
}
fn rtm_default_beta() -> f64 {
    0.1
}

impl RTM {
    fn fitted_model(&self) -> PyResult<&crate::rtm::RTMModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
    fn resolved_alpha(&self) -> f64 {
        self.alpha.unwrap_or(1.0 / self.num_topics as f64)
    }
}

/// Parse an undirected link set: a sequence of `(i, j)` document-index pairs, or a
/// `(E, 2)` integer array. Indices are validated against `num_docs` at fit time.
fn extract_edges(links: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    if let Ok(pairs) = links.extract::<Vec<(i64, i64)>>() {
        return pairs
            .into_iter()
            .map(|(i, j)| {
                if i < 0 || j < 0 {
                    Err(PyValueError::new_err("link indices must be non-negative"))
                } else {
                    Ok((i as usize, j as usize))
                }
            })
            .collect();
    }
    if let Ok(rows) = links.extract::<Vec<Vec<i64>>>() {
        return rows
            .into_iter()
            .map(|r| {
                if r.len() != 2 || r[0] < 0 || r[1] < 0 {
                    Err(PyValueError::new_err(
                        "each link must be a (i, j) pair of non-negative indices",
                    ))
                } else {
                    Ok((r[0] as usize, r[1] as usize))
                }
            })
            .collect();
    }
    Err(PyValueError::new_err(
        "links must be a sequence of (i, j) document-index pairs",
    ))
}

/// Map raw token lists to the fitted vocabulary (dropping OOV), for cold-start
/// link prediction on unseen documents.
fn doc_to_ids(corpus: &corpus::Corpus, doc: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let index: HashMap<&str, u32> = corpus
        .id_to_word
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as u32))
        .collect();
    let tokens: Vec<String> = doc
        .extract()
        .map_err(|_| PyValueError::new_err("expected a document as a list of token strings"))?;
    Ok(tokens
        .iter()
        .filter_map(|w| index.get(w.as_str()).copied())
        .collect())
}

#[pymethods]
impl RTM {
    #[new]
    #[pyo3(signature = (num_topics, *, link="logistic", inference="variational",
                        alpha=None, beta=0.1, rho=None,
                        negative_ratio=1.0, ridge=1.0, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        link: &str,
        inference: &str,
        alpha: Option<f64>,
        beta: f64,
        rho: Option<f64>,
        negative_ratio: f64,
        ridge: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !matches!(inference, "variational" | "gibbs") {
            return Err(PyValueError::new_err(format!(
                "inference must be \"variational\" or \"gibbs\", got {inference:?}"
            )));
        }
        // R lda's collapsed-Gibbs RTM sampler supports only the exponential link,
        // so the Gibbs backend always uses (and stores) it — otherwise it would
        // train exponential but score links with σ on the reference's negative
        // coefficients. `link` defaults to "logistic", so an unspecified link
        // resolves to exponential here; an unknown link string still errors.
        let parsed_link = Link::parse(link).map_err(PyValueError::new_err)?;
        let link = if inference == "gibbs" {
            Link::Exponential
        } else {
            parsed_link
        };
        if let Some(a) = alpha {
            ensure_finite_pos("alpha", a)?;
        }
        // `beta` is the topic-word Dirichlet smoothing (R lda's `eta`) for the Gibbs
        // backend; it must be finite and strictly positive.
        ensure_finite_pos("beta", beta)?;
        // `rho` / `negative_ratio` are the paper's pseudo-negative count (R lda's
        // `lambda`): the regularization that prevents the degenerate positive-links-
        // only fit, so zero is not a valid setting (it removes the negatives and the
        // logistic intercept diverges). `ridge` is the separate l2 Gaussian prior on
        // eta, where zero (plain MLE, no prior) is a legitimate choice.
        if let Some(r) = rho {
            ensure_finite_pos("rho", r)?;
        }
        ensure_finite_pos("negative_ratio", negative_ratio)?;
        ensure_finite_nonneg("ridge", ridge)?;
        Ok(RTM {
            num_topics,
            link,
            alpha,
            beta,
            rho,
            negative_ratio,
            ridge,
            inference: inference.to_string(),
            seed,
            fitted: false,
            model: None,
            corpus: None,
        })
    }

    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named to
    /// match ``__init__`` (issue #400). ``alpha``/``rho`` are ``None`` when left to
    /// resolve at fit (``alpha = 1/num_topics``; ``rho = negative_ratio * #links``).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("link", self.link.as_str())?;
        d.set_item("inference", &self.inference)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("rho", self.rho)?;
        d.set_item("negative_ratio", self.negative_ratio)?;
        d.set_item("ridge", self.ridge)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit RTM on a document graph. ``data`` is a ``Corpus`` or ``list[list[str]]``;
    /// ``links`` is a sequence of undirected ``(i, j)`` document-index pairs.
    #[pyo3(signature = (data, links, *, iters=50, e_sweeps=3, e_inner=5))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        links: &Bound<'_, PyAny>,
        iters: usize,
        e_sweeps: usize,
        e_inner: usize,
    ) -> PyResult<Py<Self>> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(
                docs,
                None,
                None,
                std::collections::HashSet::new(),
                1,
                1.0,
                0,
                0,
            )?
            .0
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let edges = extract_edges(links)?;
        for &(i, j) in &edges {
            if i >= num_docs || j >= num_docs {
                return Err(PyValueError::new_err(format!(
                    "link ({i}, {j}) refers to a document beyond the {num_docs} in the corpus"
                )));
            }
        }
        let params = crate::rtm::RtmParams {
            num_topics: slf.num_topics,
            num_types: corpus.num_types(),
            alpha: slf.resolved_alpha(),
            beta: slf.beta,
            link: slf.link,
            rho: slf.rho,
            negative_ratio: slf.negative_ratio,
            ridge: slf.ridge,
            em_iters: iters,
            e_sweeps,
            e_inner,
            var_tol: 1e-4,
            convergence_tol: 1e-5,
        };
        let gibbs = slf.inference == "gibbs";
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let (model, corpus) = py.allow_threads(move || {
            let model = if gibbs {
                crate::rtm::fit_rtm_gibbs(&corpus.docs, &edges, &params, &mut rng)
            } else {
                crate::rtm::fit_rtm(&corpus.docs, &edges, &params, &mut rng)
            };
            (model, corpus)
        });
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.fitted = true;
        Ok(slf.into())
    }

    // --- Required analysis surface ---
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

    // --- RTM-specific fitted state ---
    /// Mean topic-assignment vectors ``phi_bar`` (D x K) — the quantity the link
    /// function reads. This is NOT ``doc_topic`` (the normalized Dirichlet mean).
    #[getter]
    fn phi_bar<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.phi_bar).to_pyarray_bound(py))
    }
    /// Link-function coefficients ``eta`` (length K): how topic co-occurrence drives
    /// the log-odds (logistic) or log-rate (exponential) of a link.
    #[getter]
    fn eta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.eta.clone()).to_pyarray_bound(py))
    }
    /// Link-function intercept ``nu``.
    #[getter]
    fn nu(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.nu)
    }
    /// The link probability function in use (``"logistic"`` or ``"exponential"``).
    #[getter]
    fn link(&self) -> &str {
        self.link.as_str()
    }
    /// Per-EM-iteration variational objective (word + z + link log-likelihood).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }
    /// Whether the objective met the convergence tolerance before ``iters``.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }

    /// Plug-in link probability between two training documents,
    /// ``psi(phi_bar_i o phi_bar_j)``.
    fn predict_link(&self, i: usize, j: usize) -> PyResult<f64> {
        let m = self.fitted_model()?;
        let d = m.phi_bar.len();
        if i >= d || j >= d {
            return Err(PyValueError::new_err(format!(
                "document index out of range (have {d} documents)"
            )));
        }
        Ok(crate::rtm::link_probability(
            m.link,
            &m.eta,
            m.nu,
            &m.phi_bar[i],
            &m.phi_bar[j],
        ))
    }

    /// Suggest links for a new document from its words alone. Infers ``phi_bar``
    /// from the (in-vocabulary) tokens with the link term removed, then ranks
    /// training documents by plug-in link probability. Returns ``(doc_index,
    /// probability)`` pairs, highest first.
    #[pyo3(signature = (doc, *, top_n=20, exclude=None, infer_iters=50))]
    fn suggest_links(
        &self,
        doc: &Bound<'_, PyAny>,
        top_n: usize,
        exclude: Option<Vec<usize>>,
        infer_iters: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        let m = self.fitted_model()?;
        let corpus = self.corpus.as_ref().unwrap();
        let ids = doc_to_ids(corpus, doc)?;
        let log_beta = m.log_beta();
        let pb = crate::rtm::infer_phi_bar(&log_beta, self.resolved_alpha(), &ids, infer_iters);
        let excl: std::collections::HashSet<usize> =
            exclude.unwrap_or_default().into_iter().collect();
        let mut scored: Vec<(usize, f64)> = (0..m.phi_bar.len())
            .filter(|d| !excl.contains(d))
            .map(|d| {
                (
                    d,
                    crate::rtm::link_probability(m.link, &m.eta, m.nu, &pb, &m.phi_bar[d]),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_n);
        Ok(scored)
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

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
        write_state(
            path,
            MODEL_TAG_RTM,
            &RtmState {
                num_topics: self.num_topics,
                link: self.link.as_str().to_string(),
                alpha: self.alpha,
                beta: self.beta,
                rho: self.rho,
                negative_ratio: self.negative_ratio,
                ridge: self.ridge,
                inference: self.inference.clone(),
                seed: self.seed,
                fitted: self.fitted,
                corpus: self.corpus.clone(),
                topic_word: m.map(|m| m.topic_word.clone()),
                doc_topic: m.map(|m| m.doc_topic.clone()),
                phi_bar: m.map(|m| m.phi_bar.clone()),
                eta: m.map(|m| m.eta.clone()),
                nu: m.map(|m| m.nu),
                fit_history: m.map(|m| m.fit_history.clone()),
                converged: m.map(|m| m.converged),
            },
        )
    }

    /// Load a model from ``path``.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: RtmState = read_state(path, MODEL_TAG_RTM)?;
        let link = Link::parse(&s.link).map_err(PyValueError::new_err)?;
        let model = if s.fitted {
            Some(crate::rtm::RTMModel {
                num_topics: s.num_topics,
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                phi_bar: s.phi_bar.unwrap_or_default(),
                eta: s.eta.unwrap_or_default(),
                nu: s.nu.unwrap_or(0.0),
                link,
                fit_history: s.fit_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
            })
        } else {
            None
        };
        Ok(RTM {
            num_topics: s.num_topics,
            link,
            alpha: s.alpha,
            beta: s.beta,
            rho: s.rho,
            negative_ratio: s.negative_ratio,
            ridge: s.ridge,
            inference: s.inference,
            seed: s.seed,
            fitted: s.fitted,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RTM(num_topics={}, link={:?}, fitted={})",
            self.num_topics,
            self.link.as_str(),
            self.fitted
        )
    }
}
