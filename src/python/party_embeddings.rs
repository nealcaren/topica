//! PartyEmbeddings pyclass: Rheault & Cochrane (2020) party embeddings. A PV-DM
//! (distributed-memory paragraph-vector) model trained by negative sampling with
//! party-period metadata tags; the ideological placement is the leading principal
//! components of the learned party vectors. A scaling model with no topics, in the
//! ideal-point family alongside `Wordfish`. `use super::*` pulls in the shared
//! bindings (Corpus, arrays, save/load).

use super::*;
use crate::party_embeddings::{self, PartyEmbeddingsModel, PvdmConfig};
use std::collections::HashMap;

#[pyclass(module = "topica")]
pub struct PartyEmbeddings {
    num_dims: usize,
    vector_size: usize,
    window: usize,
    min_count: usize,
    negative: usize,
    sample: f64,
    learning_rate: f64,
    seed: u64,
    fitted: bool,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    // Fitted state.
    positions: Option<Vec<Vec<f64>>>, // num_groups x num_dims (standardized, oriented)
    model: Option<PartyEmbeddingsModel>,
    loss_history: Vec<f64>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PartyEmbeddingsState {
    num_dims: usize,
    vector_size: usize,
    window: usize,
    min_count: usize,
    negative: usize,
    sample: f64,
    learning_rate: f64,
    seed: u64,
    fitted: bool,
    author_names: Vec<String>,
    id_to_word: Vec<String>,
    positions: Option<Vec<Vec<f64>>>,
    num_groups: Option<usize>,
    num_controls: Option<usize>,
    num_words: Option<usize>,
    word_vectors: Option<Vec<f32>>,
    tag_vectors: Option<Vec<f32>>,
    loss_history: Vec<f64>,
}

impl PartyEmbeddings {
    fn fitted_model(&self) -> PyResult<&PartyEmbeddingsModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }

    fn group_index(&self, label: &str) -> PyResult<usize> {
        self.author_names
            .iter()
            .position(|x| x == label)
            .ok_or_else(|| PyValueError::new_err(format!("{label:?} is not a group label")))
    }
}

/// Cosine similarity of two equal-length vectors.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = na.sqrt() * nb.sqrt();
    if d == 0.0 {
        0.0
    } else {
        dot / d
    }
}

/// Standardize each column of `rows` to mean 0 / unit variance in place.
fn standardize_columns(rows: &mut [Vec<f64>]) {
    if rows.is_empty() {
        return;
    }
    let n = rows.len() as f64;
    let d = rows[0].len();
    for k in 0..d {
        let mean = rows.iter().map(|r| r[k]).sum::<f64>() / n;
        let var = rows.iter().map(|r| (r[k] - mean).powi(2)).sum::<f64>() / n;
        let sd = var.sqrt();
        for r in rows.iter_mut() {
            r[k] = if sd > 0.0 { (r[k] - mean) / sd } else { 0.0 };
        }
    }
}

#[pymethods]
impl PartyEmbeddings {
    /// The random seed the model was constructed with.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Create an unfitted PartyEmbeddings model. `num_dims` is the number of
    /// placement dimensions returned in `author_positions` (the leading principal
    /// components of the party vectors; the first is the left-right scale).
    /// `vector_size` is the embedding dimension M (the paper's hidden-layer size);
    /// `window` the context width; `min_count` drops words below that corpus
    /// frequency; `negative` the number of negative samples; `sample` the
    /// frequent-word subsampling threshold; `learning_rate` the initial SGD step.
    /// The fit is single-threaded and reproducible from `seed`.
    #[new]
    #[pyo3(signature = (num_dims=2, *, vector_size=200, window=20, min_count=5,
                        negative=5, sample=1e-4, learning_rate=0.025, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_dims: usize,
        vector_size: usize,
        window: usize,
        min_count: usize,
        negative: usize,
        sample: f64,
        learning_rate: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_dims < 1 {
            return Err(PyValueError::new_err("num_dims must be >= 1"));
        }
        if vector_size < 2 {
            return Err(PyValueError::new_err("vector_size must be >= 2"));
        }
        if window < 1 {
            return Err(PyValueError::new_err("window must be >= 1"));
        }
        if negative < 1 {
            return Err(PyValueError::new_err("negative must be >= 1"));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(PyValueError::new_err("learning_rate must be > 0"));
        }
        Ok(PartyEmbeddings {
            num_dims,
            vector_size,
            window,
            min_count: min_count.max(1),
            negative,
            sample,
            learning_rate,
            seed,
            fitted: false,
            author_names: Vec::new(),
            id_to_word: Vec::new(),
            positions: None,
            model: None,
            loss_history: Vec::new(),
        })
    }

    /// Fit on `data` (a Corpus or list of token lists). `group` (required, length
    /// num_docs) is the party-period label of each document; documents sharing a
    /// label contribute to one party vector. `control`, when given, is a second
    /// per-document metadata tag (e.g. parliament or government status) that is
    /// estimated to absorb its influence but is not placed. `anchors`
    /// (`{group_label: value}`) orients the sign of each placement dimension so it
    /// agrees with the supplied direction. `iters` is the number of training
    /// epochs (default 5).
    #[pyo3(signature = (data, *, group, control=None, anchors=None, iters=5))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        group: Vec<String>,
        control: Option<Vec<String>>,
        anchors: Option<HashMap<String, f64>>,
        iters: usize,
    ) -> PyResult<()> {
        // Extract documents as ordered token lists (PV-DM needs word order).
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
        let num_docs = docs_str.len();
        if num_docs == 0 {
            return Err(PyValueError::new_err("data contains no documents"));
        }
        if group.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "group must have length num_docs ({num_docs}), got {}",
                group.len()
            )));
        }
        if let Some(c) = &control {
            if c.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "control must have length num_docs ({num_docs}), got {}",
                    c.len()
                )));
            }
        }
        if iters < 1 {
            return Err(PyValueError::new_err("iters (epochs) must be >= 1"));
        }

        // Build the vocabulary (corpus frequency >= min_count), ordered by
        // descending frequency then word for determinism.
        let mut freq: HashMap<&str, u32> = HashMap::new();
        for doc in &docs_str {
            for w in doc {
                *freq.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        let mut vocab_pairs: Vec<(&str, u32)> = freq
            .iter()
            .filter(|&(_, &c)| c as usize >= self.min_count)
            .map(|(&w, &c)| (w, c))
            .collect();
        vocab_pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        if vocab_pairs.len() < 2 {
            return Err(PyValueError::new_err(
                "vocabulary has fewer than 2 words after min_count pruning",
            ));
        }
        let id_to_word: Vec<String> = vocab_pairs.iter().map(|&(w, _)| w.to_string()).collect();
        let total_freqs: Vec<u32> = vocab_pairs.iter().map(|&(_, c)| c).collect();
        let word_id: HashMap<&str, u32> = id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i as u32))
            .collect();
        let num_words = id_to_word.len();

        // Map each document to ordered in-vocabulary token ids.
        let docs_ids: Vec<Vec<u32>> = docs_str
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|w| word_id.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        if docs_ids.iter().all(|d| d.is_empty()) {
            return Err(PyValueError::new_err(
                "no in-vocabulary tokens after pruning",
            ));
        }

        // Resolve the party-period grouping (the placed tags).
        let mut group_names: Vec<String> = group.clone();
        group_names.sort();
        group_names.dedup();
        if group_names.len() < 2 {
            return Err(PyValueError::new_err(
                "PartyEmbeddings needs at least 2 groups to scale",
            ));
        }
        let gindex: HashMap<&str, usize> = group_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let group_idx: Vec<usize> = group.iter().map(|l| gindex[l.as_str()]).collect();
        let num_groups = group_names.len();

        // Resolve the optional control grouping (estimated, not placed).
        let (control_idx, num_controls): (Option<Vec<usize>>, usize) = match &control {
            None => (None, 0),
            Some(c) => {
                let mut names: Vec<String> = c.clone();
                names.sort();
                names.dedup();
                let cindex: HashMap<&str, usize> = names
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), i))
                    .collect();
                let idx: Vec<usize> = c.iter().map(|l| cindex[l.as_str()]).collect();
                (Some(idx), names.len())
            }
        };

        // Validate anchors against the group labels before the (slow) fit.
        let anchor_pairs: Vec<(usize, f64)> = match &anchors {
            None => Vec::new(),
            Some(m) => {
                let mut pairs = Vec::with_capacity(m.len());
                for (label, &target) in m {
                    let i = group_names.iter().position(|x| x == label).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "anchor label {label:?} is not a group label"
                        ))
                    })?;
                    pairs.push((i, target));
                }
                pairs
            }
        };

        let cfg = PvdmConfig {
            vector_size: self.vector_size,
            window: self.window,
            negative: self.negative,
            sample: self.sample,
            start_lr: self.learning_rate,
            min_lr: (self.learning_rate * 1e-4).min(1e-4),
            epochs: iters,
        };
        let seed = self.seed;
        let model = py.allow_threads(move || {
            party_embeddings::fit_pvdm(
                &docs_ids,
                &group_idx,
                control_idx.as_deref(),
                num_words,
                num_groups,
                num_controls,
                &total_freqs,
                &cfg,
                seed,
            )
        });

        // Placement: PCA of the party vectors, standardized, sign-oriented to the
        // anchors (each dimension independently).
        let mut positions = crate::reduce::pca(&model.group_matrix(), self.num_dims, self.seed);
        standardize_columns(&mut positions);
        if !anchor_pairs.is_empty() && !positions.is_empty() {
            let ndim = positions[0].len();
            for k in 0..ndim {
                let align: f64 = anchor_pairs.iter().map(|&(i, t)| positions[i][k] * t).sum();
                if align < 0.0 {
                    for r in positions.iter_mut() {
                        r[k] = -r[k];
                    }
                }
            }
        }

        self.loss_history = model.loss_history.clone();
        self.positions = Some(positions);
        self.model = Some(model);
        self.id_to_word = id_to_word;
        self.author_names = group_names;
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn num_authors(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_groups)
    }

    /// Party placements as a (num_authors, num_dims) matrix: the leading principal
    /// components of the learned party vectors, standardized to mean 0 / unit
    /// variance and sign-oriented to the anchors. Column 0 is the left-right scale.
    #[getter]
    fn author_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.fitted_model()?;
        Ok(vecs_to_arr2(self.positions.as_ref().unwrap()).to_pyarray_bound(py))
    }

    /// The group labels, in the row order of `author_positions`.
    #[getter]
    fn author_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.author_names.clone())
    }

    /// The learned party vectors as a (num_authors, vector_size) matrix; the raw
    /// party embeddings before placement, comparable to `word_vectors`.
    #[getter]
    fn author_vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(vecs_to_arr2(&self.fitted_model()?.group_matrix()).to_pyarray_bound(py))
    }

    /// The learned word vectors as a (vocab, vector_size) matrix, in the same space
    /// as `author_vectors`.
    #[getter]
    fn word_vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let rows: Vec<Vec<f64>> = (0..m.num_words).map(|w| m.word_vector(w)).collect();
        Ok(vecs_to_arr2(&rows).to_pyarray_bound(py))
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }

    /// The words whose embeddings are closest (cosine) to a party's, the
    /// "linguistic specificity" of that party. Returns the top-`n` `(word, cosine)`.
    /// This is the raw cosine ranking; high-frequency function words can crowd the
    /// top, so read it relative to a baseline (another party, or the average party)
    /// rather than in isolation.
    #[pyo3(signature = (group, n=10))]
    fn nearest_words(&self, group: &str, n: usize) -> PyResult<Vec<(String, f64)>> {
        let m = self.fitted_model()?;
        let g = self.group_index(group)?;
        let gv = m.group_vector(g);
        let mut scored: Vec<(usize, f64)> = (0..m.num_words)
            .map(|w| (w, cosine(&gv, &m.word_vector(w))))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored
            .into_iter()
            .take(n)
            .map(|(w, s)| (self.id_to_word[w].clone(), s))
            .collect())
    }

    /// Guided placement: project the party vectors onto a custom ideological axis
    /// defined by two word lexicons. The axis is (mean right-word vector - mean
    /// left-word vector); the returned (num_authors,) array is each party's
    /// projection, standardized, with the supplied `right` end positive. Words not
    /// in the vocabulary are ignored.
    #[pyo3(signature = (left, right))]
    fn guided_positions<'py>(
        &self,
        py: Python<'py>,
        left: Vec<String>,
        right: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let m = self.fitted_model()?;
        let word_id: HashMap<&str, usize> = self
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();
        let centroid = |words: &[String]| -> Option<Vec<f64>> {
            let mut acc = vec![0.0f64; m.vector_size];
            let mut n = 0usize;
            for w in words {
                if let Some(&id) = word_id.get(w.as_str()) {
                    let v = m.word_vector(id);
                    for k in 0..m.vector_size {
                        acc[k] += v[k];
                    }
                    n += 1;
                }
            }
            if n == 0 {
                return None;
            }
            for x in acc.iter_mut() {
                *x /= n as f64;
            }
            Some(acc)
        };
        let cl = centroid(&left)
            .ok_or_else(|| PyValueError::new_err("no left-lexicon words are in the vocabulary"))?;
        let cr = centroid(&right)
            .ok_or_else(|| PyValueError::new_err("no right-lexicon words are in the vocabulary"))?;
        let axis: Vec<f64> = (0..m.vector_size).map(|k| cr[k] - cl[k]).collect();
        let mut proj: Vec<f64> = (0..m.num_groups)
            .map(|g| {
                let gv = m.group_vector(g);
                (0..m.vector_size).map(|k| gv[k] * axis[k]).sum::<f64>()
            })
            .collect();
        // standardize
        let n = proj.len() as f64;
        let mean = proj.iter().sum::<f64>() / n;
        let sd = (proj.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
        for x in proj.iter_mut() {
            *x = if sd > 0.0 { (*x - mean) / sd } else { 0.0 };
        }
        Ok(Array1::from(proj).to_pyarray_bound(py))
    }

    /// Euclidean distance between two parties in the embedding space, a measure of
    /// how distinct their language is (the polarization metric of the paper).
    fn distance(&self, group_a: &str, group_b: &str) -> PyResult<f64> {
        let m = self.fitted_model()?;
        let a = m.group_vector(self.group_index(group_a)?);
        let b = m.group_vector(self.group_index(group_b)?);
        Ok((0..m.vector_size)
            .map(|k| (a[k] - b[k]).powi(2))
            .sum::<f64>()
            .sqrt())
    }

    /// Training trace: `(epoch, mean negative-sampling loss)` pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        self.fitted_model()?;
        Ok(self
            .loss_history
            .iter()
            .enumerate()
            .map(|(i, &l)| (i + 1, l))
            .collect())
    }

    /// `None`: the fit runs a fixed number of epochs and does not early-stop.
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        self.fitted_model()?;
        Ok(None)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
        write_state(
            path,
            MODEL_TAG_PARTY_EMBEDDINGS,
            &PartyEmbeddingsState {
                num_dims: self.num_dims,
                vector_size: self.vector_size,
                window: self.window,
                min_count: self.min_count,
                negative: self.negative,
                sample: self.sample,
                learning_rate: self.learning_rate,
                seed: self.seed,
                fitted: self.fitted,
                author_names: self.author_names.clone(),
                id_to_word: self.id_to_word.clone(),
                positions: self.positions.clone(),
                num_groups: m.map(|m| m.num_groups),
                num_controls: m.map(|m| m.num_controls),
                num_words: m.map(|m| m.num_words),
                word_vectors: m.map(|m| m.word_vectors.clone()),
                tag_vectors: m.map(|m| m.tag_vectors.clone()),
                loss_history: self.loss_history.clone(),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: PartyEmbeddingsState = read_state(path, MODEL_TAG_PARTY_EMBEDDINGS)?;
        let model = if s.fitted && s.word_vectors.is_some() {
            Some(PartyEmbeddingsModel {
                vector_size: s.vector_size,
                num_words: s.num_words.unwrap_or(0),
                num_groups: s.num_groups.unwrap_or(0),
                num_controls: s.num_controls.unwrap_or(0),
                word_vectors: s.word_vectors.unwrap_or_default(),
                tag_vectors: s.tag_vectors.unwrap_or_default(),
                loss_history: s.loss_history.clone(),
            })
        } else {
            None
        };
        Ok(PartyEmbeddings {
            num_dims: s.num_dims,
            vector_size: s.vector_size,
            window: s.window,
            min_count: s.min_count,
            negative: s.negative,
            sample: s.sample,
            learning_rate: s.learning_rate,
            seed: s.seed,
            fitted: s.fitted,
            author_names: s.author_names,
            id_to_word: s.id_to_word,
            positions: s.positions,
            model,
            loss_history: s.loss_history,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PartyEmbeddings(num_dims={}, vector_size={}, fitted={})",
            self.num_dims, self.vector_size, self.fitted
        )
    }
}
