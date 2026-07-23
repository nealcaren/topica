//! Python bindings for the Polylingual Topic Model (PLTM).

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyType};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Polylingual Topic Model (Mimno et al. 2009): LDA for aligned document tuples
/// across `L` languages. Every document in a tuple shares one topic distribution
/// θ; each topic carries a per-language word distribution φˡ. Topic `k` denotes
/// the same theme in every language, so the languages' topics are aligned by
/// construction -- no post-hoc matching. Fit on comparable or parallel corpora
/// (a dict mapping language name to that language's documents).
#[pyclass(module = "topica")]
pub struct PolylingualLDA {
    num_topics: usize,
    alpha: Option<f64>,
    beta: f64,
    iters: usize,
    optimize_alpha: bool,
    optimize_interval: usize,
    optimize_burn_in: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    languages: Vec<String>,
    model: Option<crate::pltm::PltmModel>,
    corpora: Vec<corpus::Corpus>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PltmState {
    num_topics: usize,
    alpha: Option<f64>,
    beta: f64,
    iters: usize,
    optimize_alpha: bool,
    optimize_interval: usize,
    optimize_burn_in: usize,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    languages: Vec<String>,
    corpora: Vec<corpus::Corpus>,
    vocab_sizes: Option<Vec<usize>>,
    resolved_alpha: Option<Vec<f64>>,
    resolved_alpha_sum: Option<f64>,
    resolved_beta: Option<Vec<f64>>,
    topic_word: Option<Vec<Vec<Vec<f64>>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
}

impl PolylingualLDA {
    fn fitted_model(&self) -> PyResult<&crate::pltm::PltmModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Resolve a `lang` selector to a language index. Accepts the language name
    /// or a stringified index ("0", "1", ...); `None` means the first language.
    fn lang_index(&self, lang: Option<&str>) -> PyResult<usize> {
        let lang = match lang {
            None => return Ok(0),
            Some(l) => l,
        };
        if let Some(i) = self.languages.iter().position(|n| n == lang) {
            return Ok(i);
        }
        if let Ok(i) = lang.parse::<usize>() {
            if i < self.languages.len() {
                return Ok(i);
            }
        }
        Err(PyValueError::new_err(format!(
            "lang must be one of {:?} or an index in 0..{}; got {lang:?}",
            self.languages,
            self.languages.len()
        )))
    }
}

/// Extract ordered (language name, string documents) pairs from the `fit`/
/// `transform` argument. Accepts a dict `{lang: docs}` (preferred, keeps names
/// and order) or a bare list `[docs, ...]` (auto-named `lang_0`, `lang_1`, ...).
/// Each `docs` may be a `Corpus` or a `list[list[str]]`.
fn extract_languages(data: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, Vec<Vec<Vec<String>>>)> {
    let to_str_docs = |v: &Bound<'_, PyAny>| -> PyResult<Vec<Vec<String>>> {
        if let Ok(c) = v.extract::<Corpus>() {
            Ok(c.inner
                .docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&wid| c.inner.id_to_word[wid as usize].clone())
                        .collect()
                })
                .collect())
        } else {
            v.extract::<Vec<Vec<String>>>().map_err(|_| {
                PyValueError::new_err(
                    "each language must be a Corpus or a list of token lists (list[list[str]])",
                )
            })
        }
    };

    if let Ok(dict) = data.downcast::<PyDict>() {
        let mut names = Vec::with_capacity(dict.len());
        let mut docs = Vec::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            names.push(
                key.extract::<String>()
                    .map_err(|_| PyValueError::new_err("language keys must be strings"))?,
            );
            docs.push(to_str_docs(&value)?);
        }
        if names.is_empty() {
            return Err(PyValueError::new_err("need at least one language"));
        }
        Ok((names, docs))
    } else if let Ok(seq) = data.extract::<Vec<Bound<'_, PyAny>>>() {
        if seq.is_empty() {
            return Err(PyValueError::new_err("need at least one language"));
        }
        let names = (0..seq.len()).map(|i| format!("lang_{i}")).collect();
        let docs = seq.iter().map(to_str_docs).collect::<PyResult<Vec<_>>>()?;
        Ok((names, docs))
    } else {
        Err(PyValueError::new_err(
            "fit() expects a dict {language: docs} or a list of per-language corpora",
        ))
    }
}

#[pymethods]
impl PolylingualLDA {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). ``alpha`` is ``None`` when left unset
    /// (the core resolves it to the 0.01 default at fit time).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha", self.alpha)?;
        d.set_item("beta", self.beta)?;
        d.set_item("iters", self.iters)?;
        d.set_item("optimize_alpha", self.optimize_alpha)?;
        d.set_item("optimize_interval", self.optimize_interval)?;
        d.set_item("optimize_burn_in", self.optimize_burn_in)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted PLTM. `alpha` is the per-topic document-topic prior
    /// (default `0.01`, the paper's `0.01·T` total with a uniform base measure).
    /// `beta` is the topic-word prior (default `0.01`); it is applied to every
    /// language (the paper's recommended βˡ = 0.01 for all languages). The Rust
    /// core supports a distinct βˡ per language, but the binding exposes a single
    /// shared value. With `optimize_alpha=True` (default) the asymmetric αm prior
    /// is re-estimated every `optimize_interval` Gibbs iterations after an
    /// `optimize_burn_in` warm-up, as in the reference implementation.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=None, beta=0.01, iters=1000,
                        optimize_alpha=true, optimize_interval=10,
                        optimize_burn_in=200, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_topics: usize,
        alpha: Option<f64>,
        beta: f64,
        iters: usize,
        optimize_alpha: bool,
        optimize_interval: usize,
        optimize_burn_in: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if let Some(a) = alpha {
            if a <= 0.0 {
                return Err(PyValueError::new_err("alpha must be > 0.0"));
            }
        }
        if beta <= 0.0 {
            return Err(PyValueError::new_err("beta must be > 0.0"));
        }
        if iters == 0 {
            return Err(PyValueError::new_err("iters must be > 0"));
        }
        Ok(PolylingualLDA {
            num_topics,
            alpha,
            beta,
            iters,
            optimize_alpha,
            optimize_interval,
            optimize_burn_in,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            languages: Vec::new(),
            model: None,
            corpora: Vec::new(),
        })
    }

    /// Fit on aligned document tuples. `data` is a dict `{language: docs}` (or a
    /// list of per-language corpora); every language must have the same number of
    /// tuples `D`, aligned by index. A tuple absent in a language is an empty
    /// document `[]` at that index.
    #[pyo3(signature = (data, *, iters=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: Option<usize>,
    ) -> PyResult<Py<Self>> {
        let (languages, str_docs) = extract_languages(data)?;

        // Build one corpus per language (independent vocabularies), preserving
        // document order and count so tuples stay aligned by index.
        let mut corpora: Vec<corpus::Corpus> = Vec::with_capacity(languages.len());
        for docs in str_docs {
            let (c, _) = build_corpus_from_docs(
                docs,
                None,
                None,
                std::collections::HashSet::new(),
                1,
                1.0,
                0,
                0,
            )?;
            corpora.push(c);
        }

        let num_docs = corpora[0].docs.len();
        for (name, c) in languages.iter().zip(&corpora) {
            if c.docs.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "all languages must have the same number of tuples; \
                     {:?} has {} but the first language has {}",
                    name,
                    c.docs.len(),
                    num_docs
                )));
            }
        }
        for c in &corpora {
            if c.num_types() < slf.num_topics {
                return Err(PyValueError::new_err(
                    "each language's vocabulary must have at least num_topics words",
                ));
            }
        }

        let vocab_sizes: Vec<usize> = corpora.iter().map(|c| c.num_types()).collect();
        let docs_by_lang: Vec<Vec<Vec<u32>>> = corpora.iter().map(|c| c.docs.clone()).collect();
        let iters = iters.unwrap_or(slf.iters);
        let alpha_init = slf.alpha.unwrap_or(0.01);
        let beta = vec![slf.beta; languages.len()];
        let (optimize_alpha, optimize_interval, optimize_burn_in, seed, k) = (
            slf.optimize_alpha,
            slf.optimize_interval,
            slf.optimize_burn_in,
            slf.seed,
            slf.num_topics,
        );

        let model = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            crate::pltm::fit_pltm(
                &docs_by_lang,
                k,
                &vocab_sizes,
                alpha_init,
                &beta,
                iters,
                optimize_alpha,
                optimize_interval,
                optimize_burn_in,
                &mut rng,
            )
        });

        slf.model = Some(model);
        slf.corpora = corpora;
        slf.languages = languages;
        slf.topic_names = (0..slf.num_topics).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Infer tuple-topic distributions θ (num_tuples, num_topics) for new aligned
    /// tuples, holding the fitted per-language φ fixed. `data` has the same shape
    /// as `fit`; out-of-vocabulary tokens are dropped per language.
    #[pyo3(signature = (data, *, sweeps=100))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        sweeps: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let (names, str_docs) = extract_languages(data)?;
        if names.len() != self.languages.len() {
            return Err(PyValueError::new_err(format!(
                "transform got {} languages but the model was fit on {}",
                names.len(),
                self.languages.len()
            )));
        }
        // Map each language's new documents onto its training vocabulary.
        let mut mapped: Vec<Vec<Vec<u32>>> = Vec::with_capacity(self.languages.len());
        for (li, docs) in str_docs.iter().enumerate() {
            let index: HashMap<&str, u32> = self.corpora[li]
                .id_to_word
                .iter()
                .enumerate()
                .map(|(i, w)| (w.as_str(), i as u32))
                .collect();
            mapped.push(
                docs.iter()
                    .map(|doc| {
                        doc.iter()
                            .filter_map(|w| index.get(w.as_str()).copied())
                            .collect()
                    })
                    .collect(),
            );
        }
        let num_docs = mapped[0].len();
        for (name, docs) in names.iter().zip(&mapped) {
            if docs.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "all languages must have the same number of tuples; {name:?} differs"
                )));
            }
        }

        let (alpha, alpha_sum, seed) = (m.alpha.clone(), m.alpha_sum, self.seed);
        let topic_word = m.topic_word.clone();
        let dt: Vec<Vec<f64>> = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            (0..num_docs)
                .map(|d| {
                    let tuple: Vec<Vec<u32>> =
                        (0..mapped.len()).map(|l| mapped[l][d].clone()).collect();
                    crate::pltm::infer_tuple(
                        &tuple,
                        &topic_word,
                        &alpha,
                        alpha_sum,
                        sweeps,
                        &mut rng,
                    )
                })
                .collect()
        });
        Ok(vecs_to_arr2(&dt).to_pyarray_bound(py))
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// The languages, in the order they were supplied to `fit`.
    #[getter]
    fn languages(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.languages.clone())
    }

    /// Per-language topic-word matrix φˡ (num_topics, vocab_l); rows sum to 1.
    /// `lang` selects the language by name or index (default: the first language).
    #[pyo3(signature = (lang=None))]
    fn topic_word<'py>(
        &self,
        py: Python<'py>,
        lang: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let li = self.lang_index(lang)?;
        Ok(vecs_to_arr2(&m.topic_word[li]).to_pyarray_bound(py))
    }

    /// Tuple-topic matrix θ (num_tuples, num_topics); rows sum to 1. Shared
    /// across all languages -- one distribution per tuple.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// The learned asymmetric document-topic prior αm (num_topics).
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.alpha.clone()).to_pyarray_bound(py))
    }

    #[getter]
    fn model_family(&self) -> &'static str {
        "none"
    }

    /// Per-language vocabulary. `lang` selects the language (default: the first).
    #[pyo3(signature = (lang=None))]
    fn vocabulary(&self, lang: Option<&str>) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        let li = self.lang_index(lang)?;
        Ok(self.corpora[li].id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpora[0].doc_names.clone())
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

    /// Top-`n` words per topic in a given language (default: the first).
    #[pyo3(signature = (n=10, *, lang=None, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        lang: Option<&str>,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        let li = self.lang_index(lang)?;
        let phi = vecs_to_arr2(&m.topic_word[li]);
        topic_words_helper(
            py,
            &phi,
            &self.corpora[li].id_to_word,
            self.num_topics,
            n,
            topic,
        )
    }

    /// Topic coherence (UMass) in a given language (default: the first).
    #[pyo3(signature = (n=10, *, lang=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        lang: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let li = self.lang_index(lang)?;
        let phi = vecs_to_arr2(&m.topic_word[li]);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        Ok(Array1::from(umass_coherence(&self.corpora[li], &tops)).to_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "PolylingualLDA(num_topics={}, languages={}, fitted={})",
            self.num_topics,
            self.languages.len(),
            self.fitted
        )
    }

    /// Save the fitted model to `path`. Reload with `PolylingualLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_PLTM,
            &PltmState {
                num_topics: self.num_topics,
                alpha: self.alpha,
                beta: self.beta,
                iters: self.iters,
                optimize_alpha: self.optimize_alpha,
                optimize_interval: self.optimize_interval,
                optimize_burn_in: self.optimize_burn_in,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                languages: self.languages.clone(),
                corpora: self.corpora.clone(),
                vocab_sizes: Some(m.vocab_sizes.clone()),
                resolved_alpha: Some(m.alpha.clone()),
                resolved_alpha_sum: Some(m.alpha_sum),
                resolved_beta: Some(m.beta.clone()),
                topic_word: Some(m.topic_word.clone()),
                doc_topic: Some(m.doc_topic.clone()),
            },
        )
    }

    /// Load a model from `path`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: PltmState = read_state(path, MODEL_TAG_PLTM)?;
        let model = if s.fitted && s.topic_word.is_some() {
            Some(crate::pltm::PltmModel {
                num_topics: s.num_topics,
                vocab_sizes: s.vocab_sizes.unwrap_or_default(),
                alpha: s.resolved_alpha.unwrap_or_default(),
                alpha_sum: s.resolved_alpha_sum.unwrap_or(0.0),
                beta: s.resolved_beta.unwrap_or_default(),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
            })
        } else {
            None
        };
        Ok(PolylingualLDA {
            num_topics: s.num_topics,
            alpha: s.alpha,
            beta: s.beta,
            iters: s.iters,
            optimize_alpha: s.optimize_alpha,
            optimize_interval: s.optimize_interval,
            optimize_burn_in: s.optimize_burn_in,
            seed: s.seed,
            fitted: s.fitted,
            topic_names: s.topic_names,
            languages: s.languages,
            model,
            corpora: s.corpora,
        })
    }
}
