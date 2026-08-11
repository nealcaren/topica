//! TopicalNGrams pyclass: Wang, McCallum & Wei's (2007) joint topic + phrase model.
//! Standard topic-model surface (topic_word/doc_topic/top_words/coherence) plus a
//! phrase API (top_phrases). `use super::*` pulls in the shared bindings.

use super::*;
use pyo3::types::PyDict;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::topical_ngrams::{self, Phrase, Token, TopicalNGramsModel};
use std::collections::HashMap;

#[pyclass(module = "topica")]
pub struct TopicalNGrams {
    num_topics: usize,
    alpha_sum: f64,
    beta: f64,
    gamma: f64,
    delta1: f64,
    delta2: f64,
    min_count: usize,
    seed: u64,
    fitted: bool,
    model: Option<TopicalNGramsModel>,
    corpus: Option<corpus::Corpus>,
    /// Token sequences (word id + bigram eligibility) actually fit, kept for phrase
    /// extraction and save/load.
    token_docs: Vec<Vec<(u32, bool)>>,
    /// Phrases extracted once at fit time, pre-sorted (count desc, words asc).
    phrases: Vec<StoredPhrase>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct StoredPhrase {
    words: Vec<u32>,
    topic: usize,
    count: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TopicalNGramsState {
    num_topics: usize,
    alpha_sum: f64,
    beta: f64,
    gamma: f64,
    delta1: f64,
    delta2: f64,
    min_count: usize,
    seed: u64,
    fitted: bool,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
    doc_names: Vec<String>,
    doc_freqs: Vec<u32>,
    total_freqs: Vec<u32>,
    token_docs: Vec<Vec<(u32, bool)>>,
    topic_word: Option<Vec<Vec<f64>>>,
    doc_topic: Option<Vec<Vec<f64>>>,
    token_topic: Option<Vec<Vec<usize>>>,
    token_gram: Option<Vec<Vec<u8>>>,
    doc_lengths: Option<Vec<usize>>,
    phrases: Vec<StoredPhrase>,
}

impl TopicalNGrams {
    fn fitted_model(&self) -> PyResult<&TopicalNGramsModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl TopicalNGrams {
    /// Create an unfitted Topical N-Grams model. `alpha_sum` is the total doc-topic
    /// Dirichlet mass (per-topic alpha = alpha_sum/num_topics); `beta` the unigram
    /// topic-word prior; `gamma` the bigram topic-word prior; `delta1`/`delta2` the
    /// Beta pseudocounts for a token's bigram status (unigram / bigram). topica
    /// defaults to a **balanced** `delta1 = delta2 = 1.0`, which discovers real
    /// collocations; MALLET's `0.2 / 1000` default forces nearly every token into a
    /// phrase (pass those explicitly to reproduce MALLET). `min_count` drops words
    /// below that corpus frequency; a dropped word breaks a phrase across it.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha_sum=50.0, beta=0.01, gamma=0.01,
                        delta1=1.0, delta2=1.0, min_count=1, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[pyo3(from_py_with = "py_num_topics")] num_topics: usize,
        alpha_sum: f64,
        beta: f64,
        gamma: f64,
        delta1: f64,
        delta2: f64,
        min_count: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 1 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        for (name, val) in [
            ("alpha_sum", alpha_sum),
            ("beta", beta),
            ("gamma", gamma),
            ("delta1", delta1),
            ("delta2", delta2),
        ] {
            if !val.is_finite() || val <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be a finite positive number; got {val}"
                )));
            }
        }
        Ok(TopicalNGrams {
            num_topics,
            alpha_sum,
            beta,
            gamma,
            delta1,
            delta2,
            min_count: min_count.max(1),
            seed,
            fitted: false,
            model: None,
            corpus: None,
            token_docs: Vec::new(),
            phrases: Vec::new(),
        })
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("num_topics", self.num_topics)?;
        d.set_item("alpha_sum", self.alpha_sum)?;
        d.set_item("beta", self.beta)?;
        d.set_item("gamma", self.gamma)?;
        d.set_item("delta1", self.delta1)?;
        d.set_item("delta2", self.delta2)?;
        d.set_item("min_count", self.min_count)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on `data` (a list of token lists, or a Corpus). **Token order matters**
    /// and a phrase never spans a dropped token: when `data` is a list of token
    /// lists, a word pruned by `min_count` breaks the adjacency of its neighbours
    /// (so "New STOPWORD York" cannot form "New_York"). A pre-built `Corpus` has
    /// already discarded such gaps, so all surviving adjacent tokens are treated as
    /// phrase-eligible — pass raw token lists to preserve boundary breaks.
    #[pyo3(signature = (data, *, iters=1000))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iters: usize,
    ) -> PyResult<Py<Self>> {
        // Raw token lists (order preserved), and whether gaps are known.
        let (docs_str, doc_names): (Vec<Vec<String>>, Vec<String>) =
            if let Ok(c) = data.extract::<Corpus>() {
                let strings = c
                    .inner
                    .docs
                    .iter()
                    .map(|d| {
                        d.iter()
                            .map(|&w| c.inner.id_to_word[w as usize].clone())
                            .collect()
                    })
                    .collect();
                (strings, c.inner.doc_names.clone())
            } else {
                let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                    PyValueError::new_err("fit() expects a Corpus or a list of token lists")
                })?;
                let names = (0..docs.len()).map(|i| format!("doc_{i}")).collect();
                (docs, names)
            };
        if docs_str.is_empty() {
            return Err(PyValueError::new_err("data contains no documents"));
        }

        // Guard the core assumption that token order encodes co-occurrence. Some
        // bundled corpora (e.g. a bag-of-words export) arrive alphabetically sorted
        // per document, which turns "adjacent" into "alphabetically consecutive" and
        // makes TNG manufacture phrases that are ordering artifacts, not real
        // collocations. Warn when most multi-token documents are already sorted.
        {
            let multi: Vec<&Vec<String>> = docs_str.iter().filter(|d| d.len() >= 2).collect();
            if !multi.is_empty() {
                let sorted = multi
                    .iter()
                    .filter(|d| d.windows(2).all(|w| w[0] <= w[1]))
                    .count();
                if sorted * 2 > multi.len() {
                    let warnings = py.import_bound("warnings")?;
                    let msg = "most documents are already in sorted token order, so word \
                        order may have been lost (e.g. a bag-of-words corpus). TopicalNGrams \
                        assumes token order encodes co-occurrence; its learned phrases may be \
                        alphabetical-adjacency artifacts rather than real collocations. Pass \
                        order-preserving token lists.";
                    warnings.call_method1("warn", (msg,))?;
                }
            }
        }

        // Vocabulary: corpus frequency >= min_count, ordered by descending frequency
        // then word (deterministic).
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for doc in &docs_str {
            for w in doc {
                *freq.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        let mut pairs: Vec<(&str, usize)> = freq
            .iter()
            .filter(|&(_, &c)| c >= slf.min_count)
            .map(|(&w, &c)| (w, c))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let id_to_word: Vec<String> = pairs.iter().map(|&(w, _)| w.to_string()).collect();
        if id_to_word.len() < 2 {
            return Err(PyValueError::new_err(
                "vocabulary has fewer than 2 words after min_count pruning",
            ));
        }
        let word_id: HashMap<&str, u32> = id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let num_types = id_to_word.len();

        // Build token sequences with bigram eligibility over the RAW sequence: a
        // surviving token is eligible iff its immediate raw predecessor also
        // survived (a pruned/OOV token breaks the chain). Also build in-vocab
        // word-id sequences for the coherence Corpus.
        let mut token_docs: Vec<Vec<(u32, bool)>> = Vec::with_capacity(docs_str.len());
        let mut id_docs: Vec<Vec<u32>> = Vec::with_capacity(docs_str.len());
        for doc in &docs_str {
            let mut toks: Vec<(u32, bool)> = Vec::new();
            let mut ids: Vec<u32> = Vec::new();
            let mut prev_survived = false;
            for w in doc {
                if let Some(&wid) = word_id.get(w.as_str()) {
                    toks.push((wid, prev_survived));
                    ids.push(wid);
                    prev_survived = true;
                } else {
                    prev_survived = false;
                }
            }
            token_docs.push(toks);
            id_docs.push(ids);
        }
        if token_docs.iter().all(|d| d.len() < 2) {
            return Err(PyValueError::new_err(
                "every document has fewer than 2 in-vocabulary tokens after pruning",
            ));
        }

        // Coherence corpus over the same vocabulary.
        let mut doc_freqs = vec![0u32; num_types];
        let mut total_freqs = vec![0u32; num_types];
        for ids in &id_docs {
            let mut seen = std::collections::HashSet::new();
            for &w in ids {
                total_freqs[w as usize] += 1;
                if seen.insert(w) {
                    doc_freqs[w as usize] += 1;
                }
            }
        }
        let corpus = corpus::Corpus {
            id_to_word: id_to_word.clone(),
            docs: id_docs,
            doc_names,
            doc_labels: Vec::new(),
            doc_freqs,
            total_freqs,
        };

        let seqs: Vec<Vec<Token>> = token_docs
            .iter()
            .map(|d| {
                d.iter()
                    .map(|&(w, e)| Token {
                        word: w,
                        eligible: e,
                    })
                    .collect()
            })
            .collect();
        let k = slf.num_topics;
        let alpha = slf.alpha_sum / k as f64;
        let (beta, gamma, d1, d2) = (slf.beta, slf.gamma, slf.delta1, slf.delta2);
        let mut rng = ChaCha8Rng::seed_from_u64(slf.seed);
        let (model, phrases) = py.allow_threads(move || {
            let model = topical_ngrams::fit_tng(
                &seqs, num_types, k, iters, alpha, beta, gamma, d1, d2, &mut rng,
            );
            let phrases = topical_ngrams::extract_phrases(&model, &seqs);
            (model, phrases)
        });

        slf.phrases = phrases
            .into_iter()
            .map(|p: Phrase| StoredPhrase {
                words: p.words,
                topic: p.topic,
                count: p.count,
            })
            .collect();
        slf.model = Some(model);
        slf.corpus = Some(corpus);
        slf.token_docs = token_docs;
        slf.fitted = true;
        Ok(slf.into())
    }

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
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok((0..self.num_topics).map(|k| format!("topic_{k}")).collect())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }
    /// Per-iteration convergence trace as `(iter, objective)` pairs (empty: the
    /// collapsed-Gibbs sampler keeps no running objective).
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.fitted_model()?.fit_history.clone())
    }
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self.fitted_model()?.converged)
    }

    /// Top unigram words per topic (the standard topic-word view). Phrases are
    /// separate — see `top_phrases`.
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

    /// Top phrases (multiword expressions), as `(phrase, probability)` pairs sorted
    /// by descending weight. With `topic=k`, returns topic k's phrases with
    /// probability = count / (topic k's total phrase occurrences). With no topic,
    /// pools phrases across topics ranked by raw count, probability = count / all
    /// phrase occurrences. A phrase is a head word plus its learned continuations,
    /// e.g. "machine learning". Kept separate from `top_words` (never merged).
    /// `max_len` optionally caps a phrase's word count — e.g. `max_len=3` drops the
    /// long whole-document runs that appear on corpora without real phrase structure,
    /// so only the tight collocations remain.
    #[pyo3(signature = (n=10, *, topic=None, max_len=None))]
    fn top_phrases(
        &self,
        n: usize,
        topic: Option<usize>,
        max_len: Option<usize>,
    ) -> PyResult<Vec<(String, f64)>> {
        self.fitted_model()?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let render = |words: &[u32]| -> String {
            words
                .iter()
                .map(|&w| vocab[w as usize].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let keep = |words: &[u32]| max_len.is_none_or(|m| words.len() <= m);
        // Aggregate to (rendered phrase, count) over the requested scope, so the
        // user-facing tie-break can be alphabetical on the phrase STRING (the stored
        // word-id order is frequency-based, not lexical).
        let mut merged: HashMap<String, usize> = HashMap::new();
        for p in &self.phrases {
            if topic.is_some_and(|k| p.topic != k) || !keep(&p.words) {
                continue;
            }
            *merged.entry(render(&p.words)).or_insert(0) += p.count;
        }
        let total: usize = merged.values().sum();
        if total == 0 {
            return Ok(Vec::new());
        }
        let mut pooled: Vec<(String, usize)> = merged.into_iter().collect();
        // Deterministic: count desc, then phrase string asc.
        pooled.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(pooled
            .into_iter()
            .take(n)
            .map(|(phrase, count)| (phrase, count as f64 / total as f64))
            .collect())
    }

    /// The number of distinct phrase types learned (across all topics).
    #[getter]
    fn num_phrases(&self) -> PyResult<usize> {
        self.fitted_model()?;
        Ok(self.phrases.len())
    }

    /// Per-topic coherence of the unigram topics (aligned to topic index). Higher is
    /// more coherent. `texts` supplies the reference corpus for windowed measures
    /// (defaults to the training corpus).
    #[pyo3(signature = (n=10, *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let phi = vecs_to_arr2(&self.fitted_model()?.topic_word);
        let tops = top_word_ids_phi(&phi, self.num_topics, n);
        coherence_dispatch(
            py,
            self.corpus.as_ref().unwrap(),
            &tops,
            n,
            &coherence_type,
            texts,
        )
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
        let c = self.corpus.as_ref();
        write_state(
            path,
            MODEL_TAG_TOPICAL_NGRAMS,
            &TopicalNGramsState {
                num_topics: self.num_topics,
                alpha_sum: self.alpha_sum,
                beta: self.beta,
                gamma: self.gamma,
                delta1: self.delta1,
                delta2: self.delta2,
                min_count: self.min_count,
                seed: self.seed,
                fitted: self.fitted,
                id_to_word: c.map(|c| c.id_to_word.clone()).unwrap_or_default(),
                docs: c.map(|c| c.docs.clone()).unwrap_or_default(),
                doc_names: c.map(|c| c.doc_names.clone()).unwrap_or_default(),
                doc_freqs: c.map(|c| c.doc_freqs.clone()).unwrap_or_default(),
                total_freqs: c.map(|c| c.total_freqs.clone()).unwrap_or_default(),
                token_docs: self.token_docs.clone(),
                topic_word: m.map(|m| m.topic_word.clone()),
                doc_topic: m.map(|m| m.doc_topic.clone()),
                token_topic: m.map(|m| m.token_topic.clone()),
                token_gram: m.map(|m| m.token_gram.clone()),
                doc_lengths: m.map(|m| m.doc_lengths.clone()),
                phrases: self.phrases.clone(),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: TopicalNGramsState = read_state(path, MODEL_TAG_TOPICAL_NGRAMS)?;
        let corpus = if s.fitted {
            Some(corpus::Corpus {
                id_to_word: s.id_to_word.clone(),
                docs: s.docs.clone(),
                doc_names: s.doc_names.clone(),
                doc_labels: Vec::new(),
                doc_freqs: s.doc_freqs.clone(),
                total_freqs: s.total_freqs.clone(),
            })
        } else {
            None
        };
        let model = if s.fitted && s.topic_word.is_some() {
            Some(TopicalNGramsModel {
                num_topics: s.num_topics,
                num_types: s.id_to_word.len(),
                topic_word: s.topic_word.unwrap_or_default(),
                doc_topic: s.doc_topic.unwrap_or_default(),
                token_topic: s.token_topic.unwrap_or_default(),
                token_gram: s.token_gram.unwrap_or_default(),
                doc_lengths: s.doc_lengths.unwrap_or_default(),
                alpha: s.alpha_sum / s.num_topics as f64,
                fit_history: Vec::new(),
                converged: false,
            })
        } else {
            None
        };
        Ok(TopicalNGrams {
            num_topics: s.num_topics,
            alpha_sum: s.alpha_sum,
            beta: s.beta,
            gamma: s.gamma,
            delta1: s.delta1,
            delta2: s.delta2,
            min_count: s.min_count,
            seed: s.seed,
            fitted: s.fitted,
            model,
            corpus,
            token_docs: s.token_docs,
            phrases: s.phrases,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TopicalNGrams(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}
