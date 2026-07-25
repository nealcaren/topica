//! Python bindings for the embedding-clustering models: Top2Vec and BERTopic.
//! Extracted from `mod.rs` (issue #385). Both cluster document embeddings and read
//! topics off the clusters (algorithm in `src/cluster.rs`, `src/reduce.rs`,
//! `src/represent.rs`); the `umap_notice` guard is shared by the two and moves with
//! them. Shared helpers, type aliases, and save-format tags stay in `mod.rs`,
//! reached via `use super::*`. No public API change: both classes are still
//! registered in the `#[pymodule]` fn and imported as `topica.Top2Vec` /
//! `topica.BERTopic`.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::PyDict;

/// Guard `reducer='umap'`: error out on the (non-wheel) build where the `umap`
/// feature was not compiled in. The in-house UMAP reducer is seeded and fully
/// reproducible for a fixed `seed`, so — unlike the old umap-rs path — there is
/// no non-determinism to warn about.
fn umap_notice(_py: Python<'_>, use_umap: bool) -> PyResult<()> {
    if !use_umap {
        return Ok(());
    }
    if !crate::reduce::umap_available() {
        return Err(PyRuntimeError::new_err(
            "reducer='umap' is not available in this build; rebuild with the `umap` \
             feature, or use reducer='pca' (the default)",
        ));
    }
    Ok(())
}

/// Top2Vec: topics by clustering document embeddings. We reduce the document
/// embeddings (randomized PCA), density-cluster them (HDBSCAN), and read each
/// topic off its cluster: the topic vector is the mean of its documents'
/// embeddings, and its words are the vocabulary terms nearest that vector.
///
/// You bring the embeddings. `fit(data, doc_embeddings)` needs one embedding row
/// per document; pass `word_embeddings` with the aligned `vocabulary` (same
/// space) to also get `topic_neighbors`. The topic count is discovered, not set.
///
/// `Top2Vec` and `BERTopic` share the class-based TF-IDF `topic_word` matrix, so
/// their `topic_word` / `topic_table` are the same given the same clusters. What
/// makes Top2Vec distinct is the **centroid** representation — the vocabulary
/// nearest the cluster centroid in embedding space — which `top_words` returns by
/// default when `word_embeddings` are present (pass `representation="c-tf-idf"`
/// for the shared view, or read it from `topic_neighbors`).
///
/// With `word_embeddings` the model also exposes the reference package's search
/// surface: `search_words_by_vector` / `similar_words` (vocabulary nearest a
/// vector or a set of keywords), `search_topics` (topics nearest keywords),
/// `search_documents_by_topic` / `search_documents_by_keywords` (documents
/// nearest a topic or keywords), and `topic_sizes`. topica keeps HDBSCAN's
/// cluster order and does not size-order topics, so read `topic_sizes` for the
/// per-topic counts.
///
/// No embedder of your own? `topica.llm_embed(texts, model=...)` builds the
/// matrix (OpenAI, or offline `sentence-transformers`).
#[pyclass(module = "topica")]
pub struct Top2Vec {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    clusterer: String,
    num_clusters: Option<usize>,
    resolution: f64,
    knn_neighbors: usize,
    umap_params: crate::reduce::UmapParams,
    diagnostics: bool,
    seed: u64,
    fitted: bool,
    has_word_vectors: bool,
    topic_names: Vec<String>,
    model: Option<top2vec::Top2VecModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Top2VecState {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    clusterer: String,
    num_clusters: Option<usize>,
    seed: u64,
    fitted: bool,
    has_word_vectors: bool,
    #[serde(default)]
    topic_names: Vec<String>,
    model: Option<top2vec::Top2VecModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

impl Top2Vec {
    fn fitted_model(&self) -> PyResult<&top2vec::Top2VecModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    /// Map keyword strings to their vocabulary ids, dropping any word the model
    /// was not fit on. Errors if none remain, since there is then no query to
    /// embed. Keyword searches build their query from these words' embeddings.
    fn resolve_keywords(&self, keywords: &[String]) -> PyResult<Vec<usize>> {
        let map: std::collections::HashMap<&str, usize> = self
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();
        let ids: Vec<usize> = keywords
            .iter()
            .filter_map(|k| map.get(k.as_str()).copied())
            .collect();
        if ids.is_empty() {
            return Err(PyValueError::new_err(format!(
                "none of the keywords {keywords:?} are in the fitted vocabulary; \
                 search keywords must be words the model was fit on (they carry \
                 the word embeddings used to build the query)"
            )));
        }
        Ok(ids)
    }

    /// Validate that a user query vector matches the model's embedding dimension.
    fn check_query_dim(&self, got: usize) -> PyResult<()> {
        let want = self.fitted_model()?.embedding_dim();
        if want != 0 && got != want {
            return Err(PyValueError::new_err(format!(
                "query vector has length {got} but the embedding dimension is {want}"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Top2Vec {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). Internal flags are reported under
    /// their public names (``reducer``, ``metric``); values are the effective
    /// ones actually in force (e.g. ``min_samples`` after its
    /// ``min_cluster_size`` default).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n_components", self.n_components)?;
        d.set_item("min_cluster_size", self.min_cluster_size)?;
        d.set_item("min_samples", self.min_samples)?;
        d.set_item("reducer", if self.use_umap { "umap" } else { "pca" })?;
        d.set_item("n_neighbors", self.n_neighbors)?;
        d.set_item("clusterer", self.clusterer.as_str())?;
        d.set_item("num_clusters", self.num_clusters)?;
        d.set_item("resolution", self.resolution)?;
        d.set_item("knn_neighbors", self.knn_neighbors)?;
        d.set_item("diagnostics", self.diagnostics)?;
        d.set_item("min_dist", self.umap_params.min_dist)?;
        d.set_item("spread", self.umap_params.spread)?;
        d.set_item("n_epochs", self.umap_params.n_epochs)?;
        d.set_item(
            "negative_sample_rate",
            self.umap_params.negative_sample_rate,
        )?;
        d.set_item("repulsion_strength", self.umap_params.repulsion_strength)?;
        d.set_item("metric", self.umap_params.metric.as_str())?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model. `n_components` is the reduced dimensionality
    /// before clustering. `clusterer` is `"hdbscan"` (default; discovers the topic
    /// count, leaves a `-1` noise bucket — `min_cluster_size`/`min_samples` are
    /// its knobs), the auto-K graph clusterers `"louvain"` / `"leiden"` (also
    /// discover the count, but assign every document — no noise), or `"kmeans"` /
    /// `"gmm"` / `"agglomerative"`, which assign every document to `num_clusters`
    /// clusters. `"gmm"` is a diagonal-covariance Gaussian mixture: like k-means
    /// but it models each topic's spread, so unequal-variance topics separate more
    /// cleanly. `"leiden"` runs Louvain modularity plus a refinement phase that
    /// guarantees connected topics. `min_samples` defaults to `min_cluster_size`.
    ///
    /// `reducer` is the dimensionality-reduction method, ``"pca"`` (default,
    /// deterministic) or ``"umap"`` (stochastic); `n_neighbors` is the
    /// neighborhood size for the reducer. `resolution` (default 1.0) and
    /// `knn_neighbors` (default 15) steer the ``"louvain"``/``"leiden"`` graph
    /// clusterers — higher `resolution` yields more, smaller topics; they are
    /// ignored by the other clusterers. `diagnostics` (default True) emits a
    /// one-time warning after fitting if the clustering looks degenerate (near-total
    /// collapse, a very high noise fraction, or gross over-splitting); pass False to
    /// silence it. When `reducer="umap"`, `min_dist` / `spread` / `n_epochs`
    /// (0 = auto) / `negative_sample_rate` / `repulsion_strength` / `metric`
    /// (``"cosine"`` or ``"euclidean"``) tune the UMAP layout; the defaults match
    /// `umap-learn`, and they are ignored under `reducer="pca"`. `seed` seeds the
    /// deterministic phases.
    #[new]
    #[pyo3(signature = (*, n_components=5, min_cluster_size=15, min_samples=None,
                        reducer="pca", n_neighbors=15, clusterer="hdbscan",
                        num_clusters=None, resolution=1.0, knn_neighbors=15,
                        diagnostics=true, min_dist=0.0, spread=1.0, n_epochs=0,
                        negative_sample_rate=5, repulsion_strength=1.0, metric="cosine",
                        seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        min_cluster_size: usize,
        min_samples: Option<usize>,
        reducer: &str,
        n_neighbors: usize,
        clusterer: &str,
        num_clusters: Option<i64>,
        resolution: f64,
        knn_neighbors: usize,
        diagnostics: bool,
        min_dist: f64,
        spread: f64,
        n_epochs: usize,
        negative_sample_rate: usize,
        repulsion_strength: f64,
        metric: &str,
        seed: u64,
    ) -> PyResult<Self> {
        if min_cluster_size < 2 {
            return Err(PyValueError::new_err("min_cluster_size must be >= 2"));
        }
        let use_umap = parse_reducer(reducer)?;
        let (clusterer, num_clusters) = parse_clusterer(clusterer, num_clusters)?;
        let (resolution, knn_neighbors) = parse_graph_params(resolution, knn_neighbors)?;
        let umap_params = parse_umap_params(
            min_dist,
            spread,
            n_epochs,
            negative_sample_rate,
            repulsion_strength,
            metric,
        )?;
        Ok(Top2Vec {
            n_components,
            use_umap,
            n_neighbors,
            min_cluster_size,
            min_samples: min_samples.unwrap_or(min_cluster_size),
            clusterer,
            num_clusters,
            resolution,
            knn_neighbors,
            umap_params,
            diagnostics,
            seed,
            fitted: false,
            has_word_vectors: false,
            topic_names: Vec::new(),
            model: None,
            id_to_word: Vec::new(),
            docs: Vec::new(),
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `doc_embeddings`
    /// (`(num_docs, E)`), one row per document. Pass `word_embeddings`
    /// (`(len(vocabulary), E)`) and `vocabulary` together to enable
    /// `topic_neighbors`; the word embeddings are realigned to topica's vocabulary.
    #[pyo3(signature = (data, doc_embeddings, *, word_embeddings=None, vocabulary=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
        word_embeddings: Option<&Bound<'_, PyAny>>,
        vocabulary: Option<Vec<String>>,
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
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let doc_emb = parse_features(doc_embeddings)?;
        if doc_emb.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but corpus has {} documents",
                doc_emb.len(),
                corpus.num_docs()
            )));
        }
        check_all_finite_2d("doc_embeddings", &doc_emb)?;
        let num_types = corpus.num_types();

        // Realign user word embeddings to topica's vocabulary order; words topica
        // kept but the user did not supply get a zero vector (no neighbors there).
        let word_vecs: Vec<Vec<f64>> = match word_embeddings {
            Some(we) => {
                let vocab = vocabulary.ok_or_else(|| {
                    PyValueError::new_err("word_embeddings requires `vocabulary` to align them")
                })?;
                let rows = parse_features(we)?;
                if rows.len() != vocab.len() {
                    return Err(PyValueError::new_err(format!(
                        "word_embeddings has {} rows but vocabulary has {} words",
                        rows.len(),
                        vocab.len()
                    )));
                }
                check_all_finite_2d("word_embeddings", &rows)?;
                let e = rows.first().map(|r| r.len()).unwrap_or(0);
                // Reject ragged word rows and a word/doc embedding-dim mismatch
                // (#489). Topic vectors carry the doc-embedding dim and word
                // vectors the word-embedding dim; `represent::cosine` would
                // otherwise dot them over the shorter length and divide by
                // mismatched norms, returning a silently wrong finite value. In
                // the reference words and documents come from one embedding model
                // so this cannot arise; topica takes the two matrices separately
                // and must check. (The numpy fast path is already rectangular;
                // the list-of-lists path is not.)
                if rows.iter().any(|r| r.len() != e) {
                    return Err(PyValueError::new_err(
                        "word_embeddings has ragged rows; every word vector must \
                         have the same length",
                    ));
                }
                let doc_dim = doc_emb.first().map(|r| r.len()).unwrap_or(0);
                if e != doc_dim {
                    return Err(PyValueError::new_err(format!(
                        "word_embeddings dim ({e}) must equal doc_embeddings dim \
                         ({doc_dim}); word and document vectors must live in the \
                         same embedding space"
                    )));
                }
                let map: std::collections::HashMap<&str, usize> = vocab
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (w.as_str(), i))
                    .collect();
                corpus
                    .id_to_word
                    .iter()
                    .map(|w| match map.get(w.as_str()) {
                        Some(&i) => rows[i].clone(),
                        None => vec![0.0; e],
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        slf.has_word_vectors = !word_vecs.is_empty();
        slf.id_to_word = corpus.id_to_word.clone();
        slf.docs = corpus.docs.clone();

        umap_notice(py, slf.use_umap)?;
        let (nc, uu, nn, mcs, ms, seed) = (
            slf.n_components,
            slf.use_umap,
            slf.n_neighbors,
            slf.min_cluster_size,
            slf.min_samples,
            slf.seed,
        );
        let clusterer = slf.clusterer.clone();
        let num_clusters = slf.num_clusters;
        let (resolution, knn_neighbors) = (slf.resolution, slf.knn_neighbors);
        let umap_params = slf.umap_params.clone();
        let model = py.allow_threads(move || {
            top2vec::fit_top2vec(
                &corpus.docs,
                &doc_emb,
                &word_vecs,
                num_types,
                nc,
                uu,
                nn,
                mcs,
                ms,
                &clusterer,
                num_clusters,
                resolution,
                knn_neighbors,
                &umap_params,
                seed,
            )
        });
        if model.num_topics == 0 {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "Top2Vec: clustering found no clusters (num_topics=0). Lower \
                 min_cluster_size, add data, or check the scale of your embeddings.",
                ),
            )?;
        } else if slf.diagnostics {
            emit_cluster_diagnostics(
                py,
                "Top2Vec",
                &slf.clusterer,
                model.num_topics,
                &model.labels,
            )?;
        }
        let k = model.num_topics;
        slf.model = Some(model);
        slf.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    /// Number of topics discovered (HDBSCAN clusters found).
    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_topics)
    }
    /// Topic-word distribution from class-based TF-IDF, row-normalized
    /// (`(num_topics, vocab)`).
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_word).to_pyarray_bound(py))
    }
    /// Soft document-topic membership (`(num_docs, num_topics)`), rows sum to one.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.doc_topic).to_pyarray_bound(py))
    }
    /// Each topic's vector in the embedding space (`(num_topics, E)`).
    #[getter]
    fn topic_vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.topic_vectors).to_pyarray_bound(py))
    }
    /// Hard cluster assignment per document; `-1` is a noise document with no topic.
    #[getter]
    fn labels(&self) -> PyResult<Vec<i64>> {
        Ok(self.fitted_model()?.labels.clone())
    }
    /// Topic labels (``topic_0`` … by default; settable after fit).
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let k = self.fitted_model()?.num_topics;
        if names.len() != k {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                k,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }

    /// Top `n` words of `topic` (or every topic when `topic` is None) by
    /// class-TF-IDF weight, as `(word, weight)` pairs.
    /// Top words per topic. Top2Vec's distinctive view is the **centroid**
    /// representation: the vocabulary words nearest the cluster centroid in
    /// embedding space. When fit with `word_embeddings`, `top_words` returns that
    /// by default (so `summary`/`top_words` show Top2Vec's identity, not just the
    /// class-based TF-IDF it shares with `BERTopic`). Pass
    /// `representation="c-tf-idf"` for the c-TF-IDF words, or `"centroid"`
    /// explicitly. `topic_word` and `topic_table` always stay c-TF-IDF.
    #[pyo3(signature = (n=10, *, topic=None, representation=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        representation: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
        let rep = match representation {
            Some(r) => r,
            None if self.has_word_vectors => "centroid",
            None => "c-tf-idf",
        };
        match rep {
            "centroid" => {
                if !self.has_word_vectors {
                    return Err(PyValueError::new_err(
                        "representation='centroid' requires fitting with word_embeddings (and vocabulary)",
                    ));
                }
                let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
                    if t >= m.num_topics {
                        return Err(PyValueError::new_err("topic out of range"));
                    }
                    let items: Vec<Bound<'py, PyTuple>> = m
                        .topic_neighbors(n, t)
                        .into_iter()
                        .map(|(w, s)| {
                            PyTuple::new_bound(
                                py,
                                &[self.id_to_word[w].clone().into_py(py), s.into_py(py)],
                            )
                        })
                        .collect();
                    Ok(PyList::new_bound(py, items))
                };
                match topic {
                    Some(t) => Ok(one(t)?.into_any()),
                    None => {
                        let all: Vec<Bound<'py, PyList>> =
                            (0..m.num_topics).map(one).collect::<PyResult<_>>()?;
                        Ok(PyList::new_bound(py, all).into_any())
                    }
                }
            }
            "c-tf-idf" | "ctfidf" | "c_tf_idf" => {
                let phi = vecs_to_arr2(&m.topic_word);
                topic_words_helper(py, &phi, &self.id_to_word, m.num_topics, n, topic)
            }
            other => Err(PyValueError::new_err(format!(
                "representation must be 'centroid' or 'c-tf-idf', got {other:?}"
            ))),
        }
    }

    /// The `n` vocabulary words whose embeddings are nearest `topic`'s vector by
    /// cosine, as `(word, cosine)` pairs. Requires fitting with `word_embeddings`.
    /// `topic` is the first argument, so `topic_neighbors(0, n=8)` reads naturally.
    #[pyo3(signature = (topic, *, n=10))]
    fn topic_neighbors(&self, topic: usize, n: usize) -> PyResult<Vec<(String, f64)>> {
        let m = self.fitted_model()?;
        if !self.has_word_vectors {
            return Err(PyRuntimeError::new_err(
                "fit with word_embeddings (and vocabulary) to use topic_neighbors",
            ));
        }
        if topic >= m.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        Ok(m.topic_neighbors(n, topic)
            .into_iter()
            .map(|(w, s)| (self.id_to_word[w].clone(), s))
            .collect())
    }

    /// Number of documents assigned to each topic (non-noise), indexed by topic
    /// id. topica keeps HDBSCAN's cluster-label order and does **not** reorder
    /// topics by size the way the reference top2vec does (there topic 0 is always
    /// the largest), so read this to recover per-topic sizes.
    #[getter]
    fn topic_sizes(&self) -> PyResult<Vec<usize>> {
        Ok(self.fitted_model()?.topic_sizes())
    }

    /// The `n` vocabulary words whose embeddings are nearest the query `vector`
    /// by cosine, as `(word, cosine)` pairs, best first. `vector` must live in the
    /// fitted embedding space (its length equals the embedding dimension).
    /// Requires fitting with `word_embeddings`. (Reference: `search_words_by_vector`.)
    #[pyo3(signature = (vector, *, n=10))]
    fn search_words_by_vector(&self, vector: Vec<f64>, n: usize) -> PyResult<Vec<(String, f64)>> {
        let m = self.fitted_model()?;
        if !self.has_word_vectors {
            return Err(PyRuntimeError::new_err(
                "fit with word_embeddings (and vocabulary) to search words",
            ));
        }
        self.check_query_dim(vector.len())?;
        Ok(m.search_words_by_vector(n, &vector)
            .into_iter()
            .map(|(w, s)| (self.id_to_word[w].clone(), s))
            .collect())
    }

    /// The `n` vocabulary words nearest the centroid of `keywords` by cosine, as
    /// `(word, cosine)` pairs. Keywords must be words the model was fit on (they
    /// supply the query embedding); any keyword outside the vocabulary is dropped,
    /// and it is an error if none remain. Requires fitting with `word_embeddings`.
    /// (Reference: `similar_words`.)
    #[pyo3(signature = (keywords, *, n=10))]
    fn similar_words(&self, keywords: Vec<String>, n: usize) -> PyResult<Vec<(String, f64)>> {
        let m = self.fitted_model()?;
        if !self.has_word_vectors {
            return Err(PyRuntimeError::new_err(
                "fit with word_embeddings (and vocabulary) to search words",
            ));
        }
        let ids = self.resolve_keywords(&keywords)?;
        let q = m
            .keyword_centroid(&ids)
            .expect("resolve_keywords guarantees a non-empty id list");
        Ok(m.search_words_by_vector(n, &q)
            .into_iter()
            .map(|(w, s)| (self.id_to_word[w].clone(), s))
            .collect())
    }

    /// Topics ranked by cosine of their vector to the centroid of `keywords`, as
    /// `(topic_id, cosine)` pairs, best first. `n` caps the count (default: every
    /// topic). Keywords must be in the fitted vocabulary. Requires fitting with
    /// `word_embeddings`. (Reference: `search_topics`.)
    #[pyo3(signature = (keywords, *, n=None))]
    fn search_topics(
        &self,
        keywords: Vec<String>,
        n: Option<usize>,
    ) -> PyResult<Vec<(usize, f64)>> {
        let m = self.fitted_model()?;
        if !self.has_word_vectors {
            return Err(PyRuntimeError::new_err(
                "fit with word_embeddings (and vocabulary) to search topics",
            ));
        }
        let ids = self.resolve_keywords(&keywords)?;
        let q = m
            .keyword_centroid(&ids)
            .expect("resolve_keywords guarantees a non-empty id list");
        Ok(m.search_topics_by_vector(n.unwrap_or(m.num_topics), &q))
    }

    /// The documents most representative of `topic`, as `(doc_index, cosine)`
    /// pairs: the topic's own member documents ranked by cosine of their embedding
    /// to the topic vector, best first (noise documents are never returned).
    /// `num_docs` caps the count. (Reference: `search_documents_by_topic`.)
    #[pyo3(signature = (topic, *, num_docs=10))]
    fn search_documents_by_topic(
        &self,
        topic: usize,
        num_docs: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        let m = self.fitted_model()?;
        if topic >= m.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        if !m.has_doc_vectors() {
            return Err(PyRuntimeError::new_err(
                "this model has no retained document embeddings, so documents cannot be \
                 searched; refit to enable the document searches",
            ));
        }
        Ok(m.documents_in_topic(num_docs, topic))
    }

    /// Documents ranked by cosine of their embedding to the centroid of
    /// `keywords`, as `(doc_index, cosine)` pairs (every document is eligible, not
    /// just one topic's members). `num_docs` caps the count. Requires fitting with
    /// `word_embeddings` (to embed the keywords).
    /// (Reference: `search_documents_by_keywords`.)
    #[pyo3(signature = (keywords, *, num_docs=10))]
    fn search_documents_by_keywords(
        &self,
        keywords: Vec<String>,
        num_docs: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        let m = self.fitted_model()?;
        if !self.has_word_vectors {
            return Err(PyRuntimeError::new_err(
                "fit with word_embeddings (and vocabulary) to search by keywords",
            ));
        }
        if !m.has_doc_vectors() {
            return Err(PyRuntimeError::new_err(
                "this model has no retained document embeddings, so documents cannot be \
                 searched; refit to enable the document searches",
            ));
        }
        let ids = self.resolve_keywords(&keywords)?;
        let q = m
            .keyword_centroid(&ids)
            .expect("resolve_keywords guarantees a non-empty id list");
        Ok(m.search_documents_by_vector(num_docs, &q))
    }

    /// Soft topic membership for new documents from their embeddings (cosine to
    /// each topic vector, normalized). `data` is accepted for API symmetry but
    /// Top2Vec assigns by embedding only. Returns `(num_docs, num_topics)`.
    /// UMass topic coherence per topic, shape ``(num_topics,)``. `n` is the number
    /// of top words per topic scored.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        let tops = top_word_ids_phi(&phi, m.num_topics, n);
        let corpus = corpus::Corpus {
            id_to_word: self.id_to_word.clone(),
            docs: self.docs.clone(),
            doc_names: (0..self.docs.len()).map(|i| format!("doc_{i}")).collect(),
            doc_labels: vec![String::new(); self.docs.len()],
            doc_freqs: {
                let v = self.id_to_word.len();
                let mut df = vec![0u32; v];
                for doc in &self.docs {
                    let mut seen = std::collections::HashSet::new();
                    for &w in doc {
                        seen.insert(w as usize);
                    }
                    for w in seen {
                        if w < v {
                            df[w] += 1;
                        }
                    }
                }
                df
            },
            total_freqs: {
                let v = self.id_to_word.len();
                let mut tf = vec![0u32; v];
                for doc in &self.docs {
                    for &w in doc {
                        if (w as usize) < v {
                            tf[w as usize] += 1;
                        }
                    }
                }
                tf
            },
        };
        Ok(Array1::from(umass_coherence(&corpus, &tops)).to_pyarray_bound(py))
    }

    /// Assign new documents to the nearest topic by cosine distance.
    /// `doc_embeddings` is the ``(num_docs, embedding_dim)`` embedding matrix,
    /// one row per document.
    #[pyo3(signature = (data, doc_embeddings=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
        let de_obj = doc_embeddings
            .ok_or_else(|| PyValueError::new_err("Top2Vec.transform requires doc_embeddings"))?;
        let _ = data;
        let de = parse_features(de_obj)?;
        Ok(vecs_to_arr2(&m.assign(&de)).to_pyarray_bound(py))
    }

    /// Fit, then return the document-topic proportions (`fit_transform`).
    ///
    /// `doc_embeddings` and `word_embeddings` are the ``(num_docs, E)`` and
    /// ``(len(vocabulary), E)`` dense embedding matrices; `vocabulary` lists the
    /// word strings aligned to the rows of `word_embeddings`.
    #[pyo3(signature = (data, doc_embeddings, *, word_embeddings=None, vocabulary=None))]
    fn fit_transform<'py>(
        slf: PyRefMut<'_, Self>,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: &Bound<'py, PyAny>,
        word_embeddings: Option<&Bound<'py, PyAny>>,
        vocabulary: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = Self::fit(slf, py, data, doc_embeddings, word_embeddings, vocabulary)?;
        Ok(vecs_to_arr2(&fitted.bind(py).borrow().fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Merge groups of topics into single topics, e.g. ``[[3, 7], [1, 2]]``. The
    /// topic vectors, document-topic, and topic-word are rebuilt and topic ids
    /// renumbered to a dense range.
    fn merge_topics(&mut self, groups: Vec<Vec<usize>>) -> PyResult<()> {
        let vocab = self.id_to_word.len();
        let m = self
            .model
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))?;
        m.merge_topics(&self.docs, &groups, vocab);
        Ok(())
    }

    /// Reassign noise documents (label ``-1``) to their nearest topic and rebuild
    /// the topic-word matrix. Returns how many documents were reassigned.
    fn reduce_outliers(&mut self) -> PyResult<usize> {
        let vocab = self.id_to_word.len();
        let m = self
            .model
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))?;
        let before = m.labels.iter().filter(|&&l| l < 0).count();
        m.reduce_outliers(&self.docs, vocab);
        Ok(before - m.labels.iter().filter(|&&l| l < 0).count())
    }

    /// Save the fitted model to `path` (topica's binary format), so a discovered
    /// fit can be reloaded and reused without refitting (useful with the stochastic
    /// `reducer="umap"` discovery).
    fn save(&self, path: &str) -> PyResult<()> {
        self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_TOP2VEC,
            &Top2VecState {
                n_components: self.n_components,
                use_umap: self.use_umap,
                n_neighbors: self.n_neighbors,
                min_cluster_size: self.min_cluster_size,
                min_samples: self.min_samples,
                clusterer: self.clusterer.clone(),
                num_clusters: self.num_clusters,
                seed: self.seed,
                fitted: self.fitted,
                has_word_vectors: self.has_word_vectors,
                topic_names: self.topic_names.clone(),
                model: self.model.clone(),
                id_to_word: self.id_to_word.clone(),
                docs: self.docs.clone(),
            },
        )
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: Top2VecState = read_state(path, MODEL_TAG_TOP2VEC)?;
        let num_topics = s.model.as_ref().map_or(0, |m| m.num_topics);
        let topic_names = if s.topic_names.is_empty() {
            (0..num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(Top2Vec {
            n_components: s.n_components,
            use_umap: s.use_umap,
            n_neighbors: s.n_neighbors,
            min_cluster_size: s.min_cluster_size,
            min_samples: s.min_samples,
            clusterer: s.clusterer,
            num_clusters: s.num_clusters,
            // resolution/knn_neighbors/umap_params are fit-time knobs; the fitted
            // result is fully captured by the stored model, so a loaded model just
            // carries the defaults (it never re-reduces or re-clusters).
            resolution: 1.0,
            knn_neighbors: 15,
            umap_params: crate::reduce::UmapParams::default(),
            // Fit-time flag; a loaded model won't re-fit, so it just carries the
            // default (diagnostics only fire during fit()).
            diagnostics: true,
            seed: s.seed,
            fitted: s.fitted,
            has_word_vectors: s.has_word_vectors,
            topic_names,
            model: s.model,
            id_to_word: s.id_to_word,
            docs: s.docs,
        })
    }

    /// Top2Vec has no iterative objective; fit_history is always ``[]``.
    #[getter]
    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }

    /// Top2Vec is not an iterative sampler (UMAP + clustering); converged is always ``None``.
    #[getter]
    fn converged(&self) -> Option<bool> {
        None
    }

    fn __repr__(&self) -> String {
        let k = self.model.as_ref().map_or(0, |m| m.num_topics);
        format!("Top2Vec(fitted={}, num_topics={})", self.fitted, k)
    }
}

// ---------------------------------------------------------------------------
// BERTopic: embedding-clustering topic model with c-TF-IDF (Grootendorst 2022)
// ---------------------------------------------------------------------------

/// BERTopic: the same reduce/cluster pipeline as `Top2Vec`, but topics are
/// defined by class-based TF-IDF over their documents' words, so no word
/// embeddings are needed. `nr_topics` merges the most similar topics down to a
/// target count; `doc_topic` is the approximate distribution (a sliding window's
/// c-TF-IDF compared to each topic) — except with `clusterer="gmm"` (and no
/// `nr_topics`), where it is the GMM's soft posterior membership. You bring the
/// document embeddings.
///
/// No embedder of your own? `topica.llm_embed(texts, model=...)` builds the
/// matrix (OpenAI, or offline `sentence-transformers`).
#[pyclass(module = "topica")]
pub struct BERTopic {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    nr_topics: Option<usize>,
    window: usize,
    stride: usize,
    bm25: bool,
    reduce_frequent: bool,
    clusterer: String,
    num_clusters: Option<usize>,
    resolution: f64,
    knn_neighbors: usize,
    umap_params: crate::reduce::UmapParams,
    diagnostics: bool,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    model: Option<bertopic::BertopicModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BertopicState {
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    min_cluster_size: usize,
    min_samples: usize,
    nr_topics: Option<usize>,
    window: usize,
    stride: usize,
    bm25: bool,
    reduce_frequent: bool,
    clusterer: String,
    num_clusters: Option<usize>,
    seed: u64,
    fitted: bool,
    #[serde(default)]
    topic_names: Vec<String>,
    model: Option<bertopic::BertopicModel>,
    id_to_word: Vec<String>,
    docs: Vec<Vec<u32>>,
}

impl BERTopic {
    fn fitted_model(&self) -> PyResult<&bertopic::BertopicModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
    /// Map token-list documents to id documents over the fitted vocabulary,
    /// dropping out-of-vocabulary words (used for `approximate_distribution`).
    fn to_ids(&self, docs: &[Vec<String>]) -> Vec<Vec<u32>> {
        let map: std::collections::HashMap<&str, u32> = self
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        docs.iter()
            .map(|d| {
                d.iter()
                    .filter_map(|w| map.get(w.as_str()).copied())
                    .collect()
            })
            .collect()
    }
}

#[pymethods]
impl BERTopic {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// The constructor configuration as a JSON-serialisable dict, keyword-named
    /// to match ``__init__`` (issue #400). Internal flags are reported under
    /// their public names (``reducer``, ``metric``); values are the effective
    /// ones actually in force (e.g. ``window``/``stride`` after their ``.max(1)``
    /// floor, ``min_samples`` after its ``min_cluster_size`` default).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("n_components", self.n_components)?;
        d.set_item("min_cluster_size", self.min_cluster_size)?;
        d.set_item("min_samples", self.min_samples)?;
        d.set_item("nr_topics", self.nr_topics)?;
        d.set_item("window", self.window)?;
        d.set_item("stride", self.stride)?;
        d.set_item("reducer", if self.use_umap { "umap" } else { "pca" })?;
        d.set_item("n_neighbors", self.n_neighbors)?;
        d.set_item("bm25", self.bm25)?;
        d.set_item("reduce_frequent", self.reduce_frequent)?;
        d.set_item("clusterer", self.clusterer.as_str())?;
        d.set_item("num_clusters", self.num_clusters)?;
        d.set_item("resolution", self.resolution)?;
        d.set_item("knn_neighbors", self.knn_neighbors)?;
        d.set_item("diagnostics", self.diagnostics)?;
        d.set_item("min_dist", self.umap_params.min_dist)?;
        d.set_item("spread", self.umap_params.spread)?;
        d.set_item("n_epochs", self.umap_params.n_epochs)?;
        d.set_item(
            "negative_sample_rate",
            self.umap_params.negative_sample_rate,
        )?;
        d.set_item("repulsion_strength", self.umap_params.repulsion_strength)?;
        d.set_item("metric", self.umap_params.metric.as_str())?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Create an unfitted model. `nr_topics` (optional) reduces the discovered
    /// topics to that many by merging the most c-TF-IDF-similar; `window`/`stride`
    /// parameterize the soft `doc_topic` distribution.
    ///
    /// `n_components` is the reduced dimensionality before clustering;
    /// `min_cluster_size` is the smallest HDBSCAN cluster and `min_samples` its
    /// core-point neighborhood (defaults to `min_cluster_size`). `reducer` is
    /// ``"pca"`` (default, deterministic) or ``"umap"`` (stochastic) and
    /// `n_neighbors` its neighborhood size. `bm25` switches the c-TF-IDF term
    /// weighting to class-based BM25 and `reduce_frequent` dampens frequent terms
    /// by a square-root before IDF. `clusterer` is ``"hdbscan"`` (default), the
    /// auto-K graph clusterers ``"louvain"`` / ``"leiden"``, ``"kmeans"``,
    /// ``"gmm"`` (diagonal-covariance Gaussian mixture), or ``"agglomerative"``;
    /// `num_clusters` sets the target count for the last three (ignored by HDBSCAN
    /// and the auto-K clusterers). `resolution` (default 1.0) and `knn_neighbors`
    /// (default 15) steer the ``"louvain"``/``"leiden"`` clusterers — higher
    /// `resolution` yields more, smaller topics; ignored by the others.
    /// `diagnostics` (default True) emits a one-time warning after fitting if the
    /// clustering looks degenerate (near-total collapse, a very high noise
    /// fraction, or gross over-splitting); pass False to silence it. When
    /// `reducer="umap"`, `min_dist` / `spread` / `n_epochs` (0 = auto) /
    /// `negative_sample_rate` / `repulsion_strength` / `metric` (``"cosine"`` or
    /// ``"euclidean"``) tune the UMAP layout (`umap-learn` defaults; ignored under
    /// `reducer="pca"`). `seed` seeds the deterministic phases.
    #[new]
    #[pyo3(signature = (*, n_components=5, min_cluster_size=15, min_samples=None,
                        nr_topics=None, window=4, stride=1, reducer="pca", n_neighbors=15,
                        bm25=false, reduce_frequent=false, clusterer="hdbscan",
                        num_clusters=None, resolution=1.0, knn_neighbors=15,
                        diagnostics=true, min_dist=0.0, spread=1.0, n_epochs=0,
                        negative_sample_rate=5, repulsion_strength=1.0, metric="cosine",
                        seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        min_cluster_size: usize,
        min_samples: Option<usize>,
        nr_topics: Option<usize>,
        window: usize,
        stride: usize,
        reducer: &str,
        n_neighbors: usize,
        bm25: bool,
        reduce_frequent: bool,
        clusterer: &str,
        num_clusters: Option<i64>,
        resolution: f64,
        knn_neighbors: usize,
        diagnostics: bool,
        min_dist: f64,
        spread: f64,
        n_epochs: usize,
        negative_sample_rate: usize,
        repulsion_strength: f64,
        metric: &str,
        seed: u64,
    ) -> PyResult<Self> {
        if min_cluster_size < 2 {
            return Err(PyValueError::new_err("min_cluster_size must be >= 2"));
        }
        let use_umap = parse_reducer(reducer)?;
        let (clusterer, num_clusters) = parse_clusterer(clusterer, num_clusters)?;
        let (resolution, knn_neighbors) = parse_graph_params(resolution, knn_neighbors)?;
        let umap_params = parse_umap_params(
            min_dist,
            spread,
            n_epochs,
            negative_sample_rate,
            repulsion_strength,
            metric,
        )?;
        Ok(BERTopic {
            n_components,
            use_umap,
            n_neighbors,
            min_cluster_size,
            min_samples: min_samples.unwrap_or(min_cluster_size),
            nr_topics,
            window: window.max(1),
            stride: stride.max(1),
            bm25,
            reduce_frequent,
            clusterer,
            num_clusters,
            resolution,
            knn_neighbors,
            umap_params,
            diagnostics,
            seed,
            fitted: false,
            topic_names: Vec::new(),
            model: None,
            id_to_word: Vec::new(),
            docs: Vec::new(),
        })
    }

    /// Fit on `data` (a Corpus or list of token lists) with `doc_embeddings`
    /// (`(num_docs, E)`), one row per document. No word embeddings are needed.
    #[pyo3(signature = (data, doc_embeddings))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        doc_embeddings: &Bound<'_, PyAny>,
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
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let doc_emb = parse_features(doc_embeddings)?;
        if doc_emb.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "doc_embeddings has {} rows but corpus has {} documents",
                doc_emb.len(),
                corpus.num_docs()
            )));
        }
        check_all_finite_2d("doc_embeddings", &doc_emb)?;
        let num_types = corpus.num_types();
        slf.id_to_word = corpus.id_to_word.clone();
        slf.docs = corpus.docs.clone();
        umap_notice(py, slf.use_umap)?;
        let (nc, uu, nn, mcs, ms, nr, win, st, b25, rf, seed) = (
            slf.n_components,
            slf.use_umap,
            slf.n_neighbors,
            slf.min_cluster_size,
            slf.min_samples,
            slf.nr_topics,
            slf.window,
            slf.stride,
            slf.bm25,
            slf.reduce_frequent,
            slf.seed,
        );
        let clusterer = slf.clusterer.clone();
        let num_clusters = slf.num_clusters;
        let (resolution, knn_neighbors) = (slf.resolution, slf.knn_neighbors);
        let umap_params = slf.umap_params.clone();
        let model = py.allow_threads(move || {
            bertopic::fit_bertopic(
                &corpus.docs,
                &doc_emb,
                num_types,
                nc,
                uu,
                nn,
                mcs,
                ms,
                nr,
                win,
                st,
                b25,
                rf,
                &clusterer,
                num_clusters,
                resolution,
                knn_neighbors,
                &umap_params,
                seed,
            )
        });
        if model.num_topics == 0 {
            let warnings = py.import_bound("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "BERTopic: clustering found no clusters (num_topics=0). Lower \
                 min_cluster_size, add data, or check the scale of your embeddings.",
                ),
            )?;
        } else if slf.diagnostics {
            emit_cluster_diagnostics(
                py,
                "BERTopic",
                &slf.clusterer,
                model.num_topics,
                &model.labels,
            )?;
        }
        let k = model.num_topics;
        slf.model = Some(model);
        slf.topic_names = (0..k).map(|i| format!("topic_{i}")).collect();
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_topics)
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
    fn labels(&self) -> PyResult<Vec<i64>> {
        Ok(self.fitted_model()?.labels.clone())
    }
    /// Topic labels (``topic_0`` … by default; settable after fit).
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.topic_names.clone())
    }
    #[setter]
    fn set_topic_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let k = self.fitted_model()?.num_topics;
        if names.len() != k {
            return Err(PyValueError::new_err(format!(
                "topic_names must have length {} (got {})",
                k,
                names.len()
            )));
        }
        self.topic_names = names;
        Ok(())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }

    /// Top `n` words of `topic` (or every topic when None) by c-TF-IDF weight.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
        let phi = vecs_to_arr2(&m.topic_word);
        topic_words_helper(py, &phi, &self.id_to_word, m.num_topics, n, topic)
    }

    /// UMass coherence for each topic's top-`n` words, over the training corpus.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let phi = vecs_to_arr2(&m.topic_word);
        let tops = top_word_ids_phi(&phi, m.num_topics, n);
        let corpus = corpus::Corpus {
            id_to_word: self.id_to_word.clone(),
            docs: self.docs.clone(),
            doc_names: (0..self.docs.len()).map(|i| format!("doc_{i}")).collect(),
            doc_labels: vec![String::new(); self.docs.len()],
            doc_freqs: {
                let v = self.id_to_word.len();
                let mut df = vec![0u32; v];
                for doc in &self.docs {
                    let mut seen = std::collections::HashSet::new();
                    for &w in doc {
                        seen.insert(w as usize);
                    }
                    for w in seen {
                        if w < v {
                            df[w] += 1;
                        }
                    }
                }
                df
            },
            total_freqs: {
                let v = self.id_to_word.len();
                let mut tf = vec![0u32; v];
                for doc in &self.docs {
                    for &w in doc {
                        if (w as usize) < v {
                            tf[w as usize] += 1;
                        }
                    }
                }
                tf
            },
        };
        Ok(Array1::from(umass_coherence(&corpus, &tops)).to_pyarray_bound(py))
    }

    /// The soft topic distribution for `data` (Corpus or token lists), as
    /// `(num_docs, num_topics)`. Words outside the fitted vocabulary are dropped;
    /// `window`/`stride` default to the values set on the model.
    #[pyo3(signature = (data, *, window=None, stride=None))]
    fn approximate_distribution<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        window: Option<usize>,
        stride: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        if m.num_topics == 0 {
            return Err(PyRuntimeError::new_err(
                "model found no topics (num_topics=0); refit with a smaller \
                 min_cluster_size or more data",
            ));
        }
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
                PyValueError::new_err("approximate_distribution expects a Corpus or token lists")
            })?
        };
        let ids = self.to_ids(&docs_str);
        let dist = m.approximate_distribution(
            &ids,
            window.unwrap_or(self.window),
            stride.unwrap_or(self.stride),
        );
        Ok(vecs_to_arr2(&dist).to_pyarray_bound(py))
    }

    /// Soft topic distribution for new documents (the approximate distribution
    /// over their words). BERTopic reads topics from text; `doc_embeddings` is
    /// accepted but not used (for API consistency with the other embedding
    /// models). Returns `(num_docs, num_topics)`.
    #[pyo3(signature = (data, doc_embeddings=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = doc_embeddings;
        self.approximate_distribution(py, data, None, None)
    }

    /// Fit, then return the document-topic distribution (`fit_transform`).
    /// `doc_embeddings` is the ``(num_docs, embedding_dim)`` embedding matrix,
    /// one row per document in corpus order.
    #[pyo3(signature = (data, doc_embeddings))]
    fn fit_transform<'py>(
        slf: PyRefMut<'_, Self>,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        doc_embeddings: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = Self::fit(slf, py, data, doc_embeddings)?;
        Ok(vecs_to_arr2(&fitted.bind(py).borrow().fitted_model()?.doc_topic).to_pyarray_bound(py))
    }

    /// Merge groups of topics into single topics, e.g. ``[[3, 7], [1, 2]]``,
    /// rebuilding the c-TF-IDF representation and the document-topic distribution.
    fn merge_topics(&mut self, groups: Vec<Vec<usize>>) -> PyResult<()> {
        let (vocab, b25, rf, win, st) = (
            self.id_to_word.len(),
            self.bm25,
            self.reduce_frequent,
            self.window,
            self.stride,
        );
        let m = self
            .model
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))?;
        m.merge_topics(&self.docs, &groups, vocab, b25, rf, win, st);
        Ok(())
    }

    /// Reassign noise documents (label ``-1``) to their nearest topic by c-TF-IDF
    /// fit and rebuild. Returns how many documents were reassigned.
    fn reduce_outliers(&mut self) -> PyResult<usize> {
        let (vocab, b25, rf, win, st) = (
            self.id_to_word.len(),
            self.bm25,
            self.reduce_frequent,
            self.window,
            self.stride,
        );
        let m = self
            .model
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))?;
        let before = m.labels.iter().filter(|&&l| l < 0).count();
        m.reduce_outliers(&self.docs, vocab, b25, rf, win, st);
        Ok(before - m.labels.iter().filter(|&&l| l < 0).count())
    }

    /// Save the fitted model to `path` (topica's binary format), so a discovered
    /// fit can be reloaded and reused without refitting (useful with the stochastic
    /// `reducer="umap"` discovery).
    fn save(&self, path: &str) -> PyResult<()> {
        self.fitted_model()?;
        write_state(
            path,
            MODEL_TAG_BERTOPIC,
            &BertopicState {
                n_components: self.n_components,
                use_umap: self.use_umap,
                n_neighbors: self.n_neighbors,
                min_cluster_size: self.min_cluster_size,
                min_samples: self.min_samples,
                nr_topics: self.nr_topics,
                window: self.window,
                stride: self.stride,
                bm25: self.bm25,
                reduce_frequent: self.reduce_frequent,
                clusterer: self.clusterer.clone(),
                num_clusters: self.num_clusters,
                seed: self.seed,
                fitted: self.fitted,
                topic_names: self.topic_names.clone(),
                model: self.model.clone(),
                id_to_word: self.id_to_word.clone(),
                docs: self.docs.clone(),
            },
        )
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: BertopicState = read_state(path, MODEL_TAG_BERTOPIC)?;
        let num_topics = s.model.as_ref().map_or(0, |m| m.num_topics);
        let topic_names = if s.topic_names.is_empty() {
            (0..num_topics).map(|i| format!("topic_{i}")).collect()
        } else {
            s.topic_names
        };
        Ok(BERTopic {
            n_components: s.n_components,
            use_umap: s.use_umap,
            n_neighbors: s.n_neighbors,
            min_cluster_size: s.min_cluster_size,
            min_samples: s.min_samples,
            nr_topics: s.nr_topics,
            window: s.window,
            stride: s.stride,
            bm25: s.bm25,
            reduce_frequent: s.reduce_frequent,
            clusterer: s.clusterer,
            num_clusters: s.num_clusters,
            // Fit-time knobs; a loaded model never re-reduces or re-clusters, so
            // defaults suffice.
            resolution: 1.0,
            knn_neighbors: 15,
            umap_params: crate::reduce::UmapParams::default(),
            // Fit-time flag; diagnostics only fire during fit(), so a loaded model
            // carries the default.
            diagnostics: true,
            seed: s.seed,
            fitted: s.fitted,
            topic_names,
            model: s.model,
            id_to_word: s.id_to_word,
            docs: s.docs,
        })
    }

    /// BERTopic has no iterative objective; fit_history is always ``[]``.
    #[getter]
    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }

    /// BERTopic is not an iterative sampler (UMAP + clustering); converged is always ``None``.
    #[getter]
    fn converged(&self) -> Option<bool> {
        None
    }

    fn __repr__(&self) -> String {
        let k = self.model.as_ref().map_or(0, |m| m.num_topics);
        format!("BERTopic(fitted={}, num_topics={})", self.fitted, k)
    }
}
