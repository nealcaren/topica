//! Python bindings for MGLDA (Multi-Grain LDA, Titov & McDonald 2008).
//!
//! `use super::*` pulls in the shared binding helpers (Corpus, build_corpus_from_docs,
//! save/load, array adapters, topic_words_helper, coherence, …).

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyString};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// MGLDA: Multi-Grain LDA (Titov & McDonald, "Modeling Online Reviews with Multi-Grain
/// Topic Models," WWW 2008). Learns GLOBAL topics (the document-level subject) and
/// LOCAL topics (rateable aspects over a sliding sentence window) simultaneously, with
/// a per-token global/local grain switch. Input is sentence-segmented
/// (``list[list[list[str]]]``: doc → sentences → tokens). Reference: tomotopy
/// ``MGLDAModel`` (MIT).
#[pyclass(module = "topica")]
pub struct MGLDA {
    num_global_topics: usize,
    num_local_topics: usize,
    window: usize,
    alpha_global: f64,
    alpha_local: f64,
    alpha_mix_global: f64,
    alpha_mix_local: f64,
    beta_global: f64,
    beta_local: f64,
    gamma: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    doc_names: Vec<String>,
    vocab: Vec<String>,
    model: Option<crate::mg_lda::MgLdaModel>,
    // A flattened bag-of-words corpus (sentences concatenated per doc) kept only for
    // umass coherence; the fit itself runs on the sentence-segmented ids.
    corpus: Option<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct MgLdaState {
    num_global_topics: usize,
    num_local_topics: usize,
    window: usize,
    alpha_global: f64,
    alpha_local: f64,
    alpha_mix_global: f64,
    alpha_mix_local: f64,
    beta_global: f64,
    beta_local: f64,
    gamma: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    doc_names: Vec<String>,
    vocab: Vec<String>,
    corpus: Option<corpus::Corpus>,
    global_topic_word: Option<Vec<Vec<f64>>>,
    local_topic_word: Option<Vec<Vec<f64>>>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    global_doc_topic: Option<Vec<Vec<f64>>>,
    global_fraction: Option<f64>,
    fit_history: Option<Vec<(usize, f64)>>,
    converged: Option<bool>,
}

impl MGLDA {
    fn fitted_model(&self) -> PyResult<&crate::mg_lda::MgLdaModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

/// Parse `data` as sentence-segmented documents (`list[list[list[str]]]`), rejecting a
/// flat `list[list[str]]`: a Python `str` is an iterable of 1-char strings, so a naive
/// `Vec<Vec<Vec<String>>>` extractor would silently treat each word as a sentence and
/// each character as a token (Gemini Gate-A blocker). We explicitly reject a string
/// where a sentence (list of tokens) is expected.
fn parse_sentence_docs(data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Vec<String>>>> {
    let docs = data.iter().map_err(|_| {
        PyValueError::new_err(
            "fit() expects sentence-segmented docs: list[list[list[str]]] \
             (documents -> sentences -> tokens)",
        )
    })?;
    let mut out = Vec::new();
    for doc in docs {
        let doc = doc?;
        if doc.is_instance_of::<PyString>() {
            return Err(PyValueError::new_err(
                "each document must be a list of sentences, not a string",
            ));
        }
        let mut sents = Vec::new();
        for sent in doc.iter().map_err(|_| {
            PyValueError::new_err("each document must be a list of sentences (list of tokens)")
        })? {
            let sent = sent?;
            if sent.is_instance_of::<PyString>() {
                return Err(PyValueError::new_err(
                    "each sentence must be a list of token strings, not a string — MGLDA input \
                     is list[list[list[str]]] (doc -> sentences -> tokens); a flat list[list[str]] \
                     is not accepted",
                ));
            }
            let tokens: Vec<String> = sent.extract().map_err(|_| {
                PyValueError::new_err("each sentence must be a list of token strings")
            })?;
            sents.push(tokens);
        }
        out.push(sents);
    }
    Ok(out)
}

#[pymethods]
impl MGLDA {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Create an unfitted model. ``num_global_topics``/``num_local_topics`` are the
    /// grain sizes K_gl / K_loc. ``window`` is T, the number of sentences per sliding
    /// window (default 3). The Dirichlet hyperparameters default to the reference
    /// (tomotopy) values: ``alpha_global``/``alpha_local`` (doc-global / window-local
    /// topic) 0.1, ``alpha_mix_global``/``alpha_mix_local`` (the grain switch) 0.1,
    /// ``beta_global``/``beta_local`` (topic-word) 0.01, ``gamma`` (sentence-window) 0.1.
    #[new]
    #[pyo3(signature = (num_global_topics, num_local_topics, *, window=3,
                        alpha_global=0.1, alpha_local=0.1, alpha_mix_global=0.1,
                        alpha_mix_local=0.1, beta_global=0.01, beta_local=0.01,
                        gamma=0.1, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_global_topics: usize,
        num_local_topics: usize,
        window: usize,
        alpha_global: f64,
        alpha_local: f64,
        alpha_mix_global: f64,
        alpha_mix_local: f64,
        beta_global: f64,
        beta_local: f64,
        gamma: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_global_topics < 1 || num_local_topics < 1 {
            return Err(PyValueError::new_err(
                "num_global_topics and num_local_topics must both be >= 1",
            ));
        }
        if window < 1 {
            return Err(PyValueError::new_err("window (T) must be >= 1"));
        }
        for (name, val) in [
            ("alpha_global", alpha_global),
            ("alpha_local", alpha_local),
            ("alpha_mix_global", alpha_mix_global),
            ("alpha_mix_local", alpha_mix_local),
            ("beta_global", beta_global),
            ("beta_local", beta_local),
            ("gamma", gamma),
        ] {
            if !(val.is_finite() && val > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "{name} must be finite and > 0"
                )));
            }
        }
        Ok(MGLDA {
            num_global_topics,
            num_local_topics,
            window,
            alpha_global,
            alpha_local,
            alpha_mix_global,
            alpha_mix_local,
            beta_global,
            beta_local,
            gamma,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            doc_names: Vec::new(),
            vocab: Vec::new(),
            model: None,
            corpus: None,
        })
    }

    /// Constructor config as a JSON-serialisable dict (#400).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_global_topics", self.num_global_topics)?;
        d.set_item("num_local_topics", self.num_local_topics)?;
        d.set_item("window", self.window)?;
        d.set_item("alpha_global", self.alpha_global)?;
        d.set_item("alpha_local", self.alpha_local)?;
        d.set_item("alpha_mix_global", self.alpha_mix_global)?;
        d.set_item("alpha_mix_local", self.alpha_mix_local)?;
        d.set_item("beta_global", self.beta_global)?;
        d.set_item("beta_local", self.beta_local)?;
        d.set_item("gamma", self.gamma)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on sentence-segmented `data` (`list[list[list[str]]]`: doc → sentences →
    /// tokens). Out-of-vocabulary tokens are dropped, but sentence boundaries and
    /// document positions are preserved: empty sentences and empty documents are kept,
    /// so the output rows (`doc_topic`, `global_doc_topic`) align 1:1 with the input
    /// documents (an empty document gets a uniform row). `iters` is the number of
    /// collapsed-Gibbs sweeps (default 1000, a topica default).
    #[pyo3(signature = (data, *, iters=1000))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
    ) -> PyResult<Py<Self>> {
        let sent_docs = parse_sentence_docs(data)?;
        if sent_docs.is_empty() {
            return Err(PyValueError::new_err("no documents provided"));
        }

        // Build a flattened bag-of-words corpus for the vocabulary + coherence.
        let flat: Vec<Vec<String>> = sent_docs
            .iter()
            .map(|sents| sents.iter().flatten().cloned().collect())
            .collect();
        let (corpus, _kept) = build_corpus_from_docs(
            flat,
            None,
            None,
            std::collections::HashSet::new(),
            1,
            1.0,
            0,
            0,
        )?;
        let word_to_id: HashMap<&str, u32> = corpus
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let num_types = corpus.num_types();

        // Map each doc's sentences to word ids on the SAME vocabulary, dropping only
        // out-of-vocabulary tokens. Sentence boundaries and document positions are
        // PRESERVED (empty sentences and empty documents are kept), so the returned
        // doc_topic / global_doc_topic rows align 1:1 with the input documents and the
        // window structure (S + T - 1 windows per doc) reflects the caller's sentences.
        let mut id_docs: Vec<Vec<Vec<u32>>> = Vec::with_capacity(sent_docs.len());
        let mut total_tokens = 0usize;
        for sents in &sent_docs {
            let mapped: Vec<Vec<u32>> = sents
                .iter()
                .map(|sent| {
                    let ids: Vec<u32> = sent
                        .iter()
                        .filter_map(|w| word_to_id.get(w.as_str()).copied())
                        .collect();
                    total_tokens += ids.len();
                    ids
                })
                .collect();
            id_docs.push(mapped);
        }
        if total_tokens == 0 {
            return Err(PyValueError::new_err(
                "corpus has no in-vocabulary tokens after tokenization",
            ));
        }

        let (kg, kl, t) = (slf.num_global_topics, slf.num_local_topics, slf.window);
        let (ag, al, amg, aml) = (
            slf.alpha_global,
            slf.alpha_local,
            slf.alpha_mix_global,
            slf.alpha_mix_local,
        );
        let (bg, bl, g) = (slf.beta_global, slf.beta_local, slf.gamma);
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let n_docs = id_docs.len();

        let model = py.allow_threads(move || {
            crate::mg_lda::fit(
                &id_docs, num_types, kg, kl, t, ag, al, amg, aml, bg, bl, g, iters, &mut rng,
            )
        });

        // Warn when the local grain barely fired: the grain switch routed almost every
        // token to the global grain, so `local_topic_word` is dominated by the prior and
        // its topics are NOT identified. This is common on text without within-document
        // aspect locality (MG-LDA's local grain earns its keep on reviews / opinion
        // text). Surfaced at fit time so a noisy local table is not silently published.
        // The 0.9 threshold matches the interpretation rule in the docs (treat local
        // topics as unidentified above ~0.9).
        if kl > 0 && model.global_fraction > 0.9 {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "MGLDA: global_fraction={:.3} — the local grain captured almost no \
                     tokens, so local_topic_word is prior-dominated and its topics are not \
                     identified. Treat local topics as unreliable here; the local grain \
                     needs text with within-document aspect locality (e.g. reviews). The \
                     global topics are unaffected.",
                    model.global_fraction
                ),),
            )?;
        }

        slf.vocab = corpus.id_to_word.clone();
        slf.corpus = Some(corpus);
        slf.doc_names = (0..n_docs).map(|i| format!("doc_{i}")).collect();
        slf.topic_names = (0..kg)
            .map(|i| format!("global_{i}"))
            .chain((0..kl).map(|i| format!("local_{i}")))
            .collect();
        slf.model = Some(model);
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_global_topics + self.num_local_topics
    }
    #[getter]
    fn num_global_topics(&self) -> usize {
        self.num_global_topics
    }
    #[getter]
    fn num_local_topics(&self) -> usize {
        self.num_local_topics
    }

    /// Combined topic-word matrix (num_global+num_local, vocab): global topics first,
    /// then local. Each row sums to 1.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Global topic-word matrix φ^gl (num_global, vocab).
    #[getter]
    fn global_topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.global_topic_word).to_pyarray_bound(py))
    }
    /// Local topic-word matrix φ^loc (num_local, vocab).
    #[getter]
    fn local_topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.local_topic_word).to_pyarray_bound(py))
    }
    /// Per-document empirical topic prevalence over [global | local]
    /// (num_docs, num_global+num_local). Rows sum to 1. This is a content-based
    /// prevalence (proportions of each doc's token assignments), NOT a single
    /// generative Dirichlet θ — global topics are document-level, local topics are
    /// window-level. For the generative doc-level distribution see `global_doc_topic`.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// Document-level global topic distribution θ^gl (num_docs, num_global), smoothed.
    #[getter]
    fn global_doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.global_doc_topic).to_pyarray_bound(py))
    }
    /// Share of tokens assigned to the global grain (vs local). MG-LDA is often
    /// global-dominant; a value near 1.0 means the local grain carried little (common on
    /// text without strong within-document aspect locality).
    #[getter]
    fn global_fraction(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.global_fraction)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.vocab.clone())
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let k = self.num_global_topics + self.num_local_topics;
        if names.len() != k {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {k} (got {})",
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.doc_names.clone())
    }
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }

    /// Top (word, prob) pairs per combined topic (global topics first, then local).
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
            &self.vocab,
            self.num_global_topics + self.num_local_topics,
            n,
            topic,
        )
    }
    /// Per-topic UMass coherence over the top `n` words (negative; closer to 0 is
    /// better), shape (num_topics,). Rows are combined-indexed: global topics first
    /// (0..num_global_topics), then local. Local topics typically score much lower —
    /// on text without aspect locality that reflects a prior-dominated local grain
    /// (see `global_fraction`), not a fixable defect.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let k = self.num_global_topics + self.num_local_topics;
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        let tops = top_word_ids_phi(&phi, k, n);
        Ok(
            Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops))
                .to_pyarray_bound(py),
        )
    }

    /// Save the fitted model to `path`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_MGLDA,
            &MgLdaState {
                num_global_topics: self.num_global_topics,
                num_local_topics: self.num_local_topics,
                window: self.window,
                alpha_global: self.alpha_global,
                alpha_local: self.alpha_local,
                alpha_mix_global: self.alpha_mix_global,
                alpha_mix_local: self.alpha_mix_local,
                beta_global: self.beta_global,
                beta_local: self.beta_local,
                gamma: self.gamma,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                doc_names: self.doc_names.clone(),
                vocab: self.vocab.clone(),
                corpus: self.corpus.clone(),
                global_topic_word: Some(m.global_topic_word.clone()),
                local_topic_word: Some(m.local_topic_word.clone()),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
                global_doc_topic: Some(m.global_doc_topic.clone()),
                global_fraction: Some(m.global_fraction),
                fit_history: Some(m.fit_history.clone()),
                converged: Some(m.converged),
            },
        )
    }

    /// Load a model saved with [`save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: MgLdaState = read_state(path, MODEL_TAG_MGLDA)?;
        let model = if s.fitted {
            Some(crate::mg_lda::MgLdaModel {
                k_gl: s.num_global_topics,
                k_loc: s.num_local_topics,
                window: s.window,
                alpha_gl: s.alpha_global,
                alpha_loc: s.alpha_local,
                alpha_mix_gl: s.alpha_mix_global,
                alpha_mix_loc: s.alpha_mix_local,
                beta_gl: s.beta_global,
                beta_loc: s.beta_local,
                gamma: s.gamma,
                global_topic_word: s.global_topic_word.unwrap_or_default(),
                local_topic_word: s.local_topic_word.unwrap_or_default(),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                global_doc_topic: s.global_doc_topic.unwrap_or_default(),
                global_fraction: s.global_fraction.unwrap_or(0.0),
                fit_history: s.fit_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
            })
        } else {
            None
        };
        Ok(MGLDA {
            num_global_topics: s.num_global_topics,
            num_local_topics: s.num_local_topics,
            window: s.window,
            alpha_global: s.alpha_global,
            alpha_local: s.alpha_local,
            alpha_mix_global: s.alpha_mix_global,
            alpha_mix_local: s.alpha_mix_local,
            beta_global: s.beta_global,
            beta_local: s.beta_local,
            gamma: s.gamma,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            doc_names: s.doc_names,
            vocab: s.vocab,
            model,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MGLDA(num_global_topics={}, num_local_topics={}, window={}, fitted={})",
            self.num_global_topics, self.num_local_topics, self.window, self.fitted
        )
    }
}
