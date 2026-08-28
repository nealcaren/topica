//! Python binding for ReplyTM — a reply-threaded topic model (CTM/STM logistic-normal topics
//! with a reply-tree structured prior; see `crate::reply_tm`). Experimental tier: topica-original,
//! no published reference yet, so `fit` is gated behind `topica.enable_experimental()`.
//!
//! The class exposes the validated core — fit on a `Corpus` or token lists + a reply tree + an
//! optional categorical covariate, with topic/proportion/prevalence readouts (prevalence carries a
//! cluster-robust method-of-composition SE), coherence, and save/load. Reply persistence is best
//! read from `persistence()` — an identifiable reduced-form estimate (observed slope + reliability
//! gate + attenuation-corrected structural κ) — rather than the ML `kappa` getter, which collapses
//! to the σ² floor on real corpora. The covariate story lives entirely in
//! `group_prevalence`/`prevalence_se`; ReplyTM is outside the `effects` namespace. Formula
//! covariates and `transform` on new threads are follow-ups.

use super::*;
use numpy::{PyArray1, PyArray2};
use pyo3::types::{PyDict, PyType};
use rand::Rng;
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
    doc_eta: Vec<Vec<f64>>,
    doc_topic_var: Vec<Vec<f64>>,
    group_prevalence: Vec<Vec<f64>>,
    prevalence_se: Vec<Vec<f64>>,
    kappa: f64,
    kappa_ci: (f64, f64),
    sigma2: f64,
    p0: f64,
    bound_history: Vec<f64>,
    // Training reply tree + covariate groups (document-aligned), retained so `persistence()` can
    // re-fit an uncoupled pass and regress child η on parent η.
    fit_parents: Vec<i64>,
    fit_groups: Vec<usize>,
    // Training corpus (all documents kept, including any emptied by min_count, so the reply-tree
    // node indices stay valid). Backs `coherence()` and save/load.
    corpus: Option<corpus::Corpus>,
}

/// Serialisable snapshot of a fitted ReplyTM (see `save`/`load`).
#[derive(serde::Serialize, serde::Deserialize)]
struct ReplyTmState {
    num_topics: usize,
    em_iters: usize,
    seed: u64,
    fitted: bool,
    vocab: Vec<String>,
    group_names: Vec<String>,
    beta: Vec<Vec<f64>>,
    doc_topic: Vec<Vec<f64>>,
    doc_eta: Vec<Vec<f64>>,
    doc_topic_var: Vec<Vec<f64>>,
    group_prevalence: Vec<Vec<f64>>,
    prevalence_se: Vec<Vec<f64>>,
    kappa: f64,
    kappa_ci: (f64, f64),
    sigma2: f64,
    p0: f64,
    bound_history: Vec<f64>,
    fit_parents: Vec<i64>,
    fit_groups: Vec<usize>,
    corpus: Option<corpus::Corpus>,
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
            doc_eta: Vec::new(),
            doc_topic_var: Vec::new(),
            group_prevalence: Vec::new(),
            prevalence_se: Vec::new(),
            kappa: f64::NAN,
            kappa_ci: (f64::NAN, f64::NAN),
            sigma2: f64::NAN,
            p0: f64::NAN,
            bound_history: Vec::new(),
            fit_parents: Vec::new(),
            fit_groups: Vec::new(),
            corpus: None,
        })
    }

    /// Fit ReplyTM. `data` is either a `topica.Corpus` or a list of token lists (already
    /// tokenized). `parents[d]` is `d`'s parent **document index** in the reply tree (`-1` for a
    /// thread root); build it in the SAME order as the documents. `covariates` is an optional
    /// per-document categorical group id in a DENSE range `0..num_groups` (the reversion anchor
    /// becomes that group's baseline prevalence); omit for a single global anchor.
    /// `covariate_names` names the groups for the readouts. `min_count` drops words rarer than it.
    /// Experimental: requires `topica.enable_experimental()`.
    #[pyo3(signature = (data, parents=None, covariates=None, covariate_names=None, *, min_count=1))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        parents: Option<Vec<i64>>,
        covariates: Option<Vec<usize>>,
        covariate_names: Option<Vec<String>>,
        min_count: usize,
    ) -> PyResult<Py<Self>> {
        require_experimental("ReplyTM")?;
        // Accept either a topica.Corpus (materialise its token strings) or raw token lists, so the
        // reply tree can be built in the same document order the corpus was ingested in.
        let docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
                .docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&wid| c.inner.id_to_word[wid as usize].clone())
                        .collect()
                })
                .collect()
        } else {
            data.extract::<Vec<Vec<String>>>().map_err(|_| {
                PyValueError::new_err(
                    "expected a Corpus or a list of token lists (list[list[str]])",
                )
            })?
        };
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
        // Documents emptied by min_count filtering: they stay as tree nodes (their latent η is
        // still coupled to parent/children) but contribute no tokens, so their proportions are
        // prior-only. Warn — a silently emptied node quietly changes the tree the user built.
        let docs_id: Vec<Vec<u32>> = docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|w| wid.get(w.as_str()).copied())
                    .collect()
            })
            .collect();
        let emptied = docs_id
            .iter()
            .zip(&docs)
            .filter(|(ids, orig)| ids.is_empty() && !orig.is_empty())
            .count();
        if emptied > 0 {
            PyErr::warn_bound(
                py,
                &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                &format!(
                    "min_count={min_count} emptied {emptied} document(s) of all tokens; they \
                     remain as reply-tree nodes but contribute no words (their topic proportions \
                     are prior-only). Lower min_count to keep them.",
                ),
                1,
            )?;
        }

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
                let names = covariate_names
                    .clone()
                    .unwrap_or_else(|| (0..ng).map(|i| format!("group{i}")).collect());
                if names.len() != ng {
                    return Err(PyValueError::new_err(format!(
                        "covariate_names has {} names but the covariate has {ng} groups",
                        names.len()
                    )));
                }
                (g.clone(), ng, names)
            }
        };

        // Build a corpus snapshot (all documents kept, in tree-index order) for coherence()
        // and save/load. Uses the SAME vocab ids as `beta`'s columns.
        let corpus_snapshot = {
            let mut doc_freqs = vec![0u32; vocab.len()];
            let mut total_freqs = vec![0u32; vocab.len()];
            for doc in &docs_id {
                let mut seen = vec![false; vocab.len()];
                for &w in doc {
                    total_freqs[w as usize] += 1;
                    if !seen[w as usize] {
                        seen[w as usize] = true;
                        doc_freqs[w as usize] += 1;
                    }
                }
            }
            corpus::Corpus {
                id_to_word: vocab.clone(),
                docs: docs_id.clone(),
                doc_names: (0..n).map(|i| format!("doc{i}")).collect(),
                doc_labels: vec![String::new(); n],
                doc_freqs,
                total_freqs,
            }
        };

        // Retain the tree + groups for persistence() before the move-closure consumes them.
        slf.fit_parents = par.clone();
        slf.fit_groups = groups.clone();

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
        slf.doc_eta = m.lambda.clone();
        slf.doc_topic_var = m.doc_topic_var.clone();
        slf.beta = m.beta;
        slf.doc_topic = dt;
        slf.group_prevalence = gp;
        slf.prevalence_se = m.anchor_se;
        slf.kappa = m.kappa;
        slf.kappa_ci = m.kappa_ci;
        slf.sigma2 = m.sigma2;
        slf.p0 = m.p0;
        slf.bound_history = m.bound_history;
        slf.corpus = Some(corpus_snapshot);
        slf.fitted = true;
        Ok(slf.into())
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

    /// D×(K-1) per-document variational mean η (softmax basis, reference topic K-1 fixed at 0):
    /// the latent topic coordinate. NOTE this is the **tree-coupled** posterior — the reply prior
    /// already pulls a child's η toward its parent's — so regressing these directly to measure
    /// persistence is CIRCULAR. Use `persistence()`, which refits an uncoupled pass for that.
    #[getter]
    fn doc_eta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.doc_eta).to_pyarray_bound(py))
    }

    /// D×(K-1) per-document posterior variance ν of η (the Laplace curvature): the measurement-error
    /// variance of each `doc_eta` row. Exposed for diagnostics; `persistence()` uses it (with an
    /// uncoupled η) to correct its child-on-parent regression for attenuation.
    #[getter]
    fn doc_topic_var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.doc_topic_var).to_pyarray_bound(py))
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

    /// G×(K-1) cluster-robust method-of-composition standard error of the group prevalence anchor
    /// (in η space): combines a between-THREAD sampling variance (clustered on the reply-tree root,
    /// so within-thread correlation does not deflate it) with the mean per-document posterior
    /// variance. `NaN` for a group with fewer than two threads (variance unidentified). To get an
    /// interval on the probability-scale `group_prevalence`, apply the delta method.
    #[getter]
    fn prevalence_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(vecs_to_arr2(&self.prevalence_se).to_pyarray_bound(py))
    }

    /// The covariate group labels (order matches `group_prevalence` rows).
    fn group_labels(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.group_names.clone())
    }

    /// The vocabulary (index order matches the `topic_word` columns).
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.vocab.clone())
    }

    /// Top-`n` words per topic. With `topic=None` (default) returns a list of lists for all K
    /// topics; with an integer `topic`, returns that one topic's words. `weights=True` returns
    /// `(word, probability)` pairs instead of bare words.
    #[pyo3(signature = (n=10, *, topic=None, weights=false))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
        weights: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = vecs_to_arr2(&self.beta);
        topic_words_helper(py, &phi, &self.vocab, self.num_topics, n, topic, weights)
    }

    /// Topic coherence (one score per topic). `coherence_type` is `"u_mass"` (default, uses the
    /// training corpus) or a windowed measure (`"c_v"`, `"c_uci"`, `"c_npmi"`); `texts` supplies
    /// an alternative reference corpus for the windowed measures. Higher is more coherent.
    #[pyo3(signature = (n=TopN(10), *, coherence_type="u_mass".to_string(), texts=None))]
    fn coherence<'py>(
        &self,
        py: Python<'py>,
        n: TopN,
        coherence_type: String,
        texts: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let n = n.0;
        let phi = vecs_to_arr2(&self.beta);
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

    /// Save the fitted model to `path`. Reload with `ReplyTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(
            path,
            MODEL_TAG_REPLYTM,
            &ReplyTmState {
                num_topics: self.num_topics,
                em_iters: self.em_iters,
                seed: self.seed,
                fitted: self.fitted,
                vocab: self.vocab.clone(),
                group_names: self.group_names.clone(),
                beta: self.beta.clone(),
                doc_topic: self.doc_topic.clone(),
                doc_eta: self.doc_eta.clone(),
                doc_topic_var: self.doc_topic_var.clone(),
                group_prevalence: self.group_prevalence.clone(),
                prevalence_se: self.prevalence_se.clone(),
                kappa: self.kappa,
                kappa_ci: self.kappa_ci,
                sigma2: self.sigma2,
                p0: self.p0,
                bound_history: self.bound_history.clone(),
                fit_parents: self.fit_parents.clone(),
                fit_groups: self.fit_groups.clone(),
                corpus: self.corpus.clone(),
            },
        )
    }

    /// Load a model saved with `save`.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let s: ReplyTmState = read_state(path, MODEL_TAG_REPLYTM)?;
        Ok(ReplyTM {
            num_topics: s.num_topics,
            em_iters: s.em_iters,
            seed: s.seed,
            fitted: s.fitted,
            vocab: s.vocab,
            group_names: s.group_names,
            beta: s.beta,
            doc_topic: s.doc_topic,
            doc_eta: s.doc_eta,
            doc_topic_var: s.doc_topic_var,
            group_prevalence: s.group_prevalence,
            prevalence_se: s.prevalence_se,
            kappa: s.kappa,
            kappa_ci: s.kappa_ci,
            sigma2: s.sigma2,
            p0: s.p0,
            bound_history: s.bound_history,
            fit_parents: s.fit_parents,
            fit_groups: s.fit_groups,
            corpus: s.corpus,
        })
    }

    /// Reduced-form reply **persistence** — the honest, identifiable replacement for the
    /// boundary-prone `kappa`. The ML `kappa` collapses to the σ² floor on real corpora; this instead
    /// fits an internal NO-TREE pass (plain logistic-normal η, so a parent and child are estimated
    /// **independently** and the estimate is not circular), then regresses each reply's η on its
    /// parent's η (centered on the covariate-group mean), pooled across topics with a thread-clustered
    /// bootstrap. Returns a dict:
    ///   `observed_persistence` — the raw slope `a` (how much a reply's topic mix tracks its
    ///     parent's); identified whenever parents vary (NaN on a degenerate corpus where every
    ///     eligible parent equals its group anchor). `observed_ci` is its 95% bootstrap interval,
    ///     or `(NaN, NaN)` when too many resamples are unidentifiable for an honest interval.
    ///   `reliability` — `Var(η) / (Var(η) + mean ν)` of the parent, the signal share and the
    ///     identifiability gate: `<= 0` means the per-document η are mostly noise, so the structural
    ///     value cannot be recovered.
    ///   `structural_kappa` — `1 - a/reliability`, the measurement-error-corrected reversion, with
    ///     `structural_kappa_ci`; `NaN` (and NaN bounds) when `reliability <= 0`.
    #[pyo3(signature = (*, bootstrap=400))]
    fn persistence<'py>(&self, py: Python<'py>, bootstrap: usize) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        let corpus = self.corpus.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no training corpus retained; refit the model")
        })?;
        let docs_id = &corpus.docs;
        let n = docs_id.len();
        let n_edges = self.fit_parents.iter().filter(|&&p| p >= 0).count();
        if n_edges == 0 {
            return Err(PyValueError::new_err(
                "persistence needs a reply tree; this model was fit with parents=None",
            ));
        }
        let k = self.num_topics;
        let km1 = k - 1;
        let v = self.vocab.len();
        let ng = self.group_names.len().max(1);
        let groups = self.fit_groups.clone();
        let iters = self.em_iters;
        let seed = self.seed;
        // Uncoupled η: fit the same corpus with every document a root (no parent coupling).
        let all_roots = vec![-1i64; n];
        let docs_owned: Vec<Vec<u32>> = docs_id.clone();
        let m0 = py.allow_threads(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            crate::reply_tm::fit_reply_tm(
                &docs_owned,
                &all_roots,
                &groups,
                ng,
                k,
                v,
                iters,
                1e-6,
                |_, _, _| true,
                &mut rng,
            )
        });
        let eta = &m0.lambda;
        let nu = &m0.doc_topic_var;

        // group anchors = per-group mean of the uncoupled η (documents with tokens)
        let has_tok: Vec<bool> = docs_id.iter().map(|d| !d.is_empty()).collect();
        let mut anchor = vec![vec![0.0f64; km1]; ng];
        let mut gcnt = vec![0usize; ng];
        for i in 0..n {
            if !has_tok[i] {
                continue;
            }
            let g = self.fit_groups[i];
            gcnt[g] += 1;
            for kk in 0..km1 {
                anchor[g][kk] += eta[i][kk];
            }
        }
        for g in 0..ng {
            if gcnt[g] > 0 {
                for kk in 0..km1 {
                    anchor[g][kk] /= gcnt[g] as f64;
                }
            }
        }

        // thread root of each document
        let mut root = vec![0usize; n];
        for start in 0..n {
            let mut cur = start as i64;
            while self.fit_parents[cur as usize] >= 0 {
                cur = self.fit_parents[cur as usize];
            }
            root[start] = cur as usize;
        }
        // per-thread aggregates over reply edges: (Sxy, Sxx, Snu, count)
        let mut agg: HashMap<usize, [f64; 4]> = HashMap::new();
        for ch in 0..n {
            let p = self.fit_parents[ch];
            if p < 0 || !has_tok[ch] || !has_tok[p as usize] {
                continue;
            }
            let g = self.fit_groups[ch];
            let (pe, ce) = (&eta[p as usize], &eta[ch]);
            let pnu = &nu[p as usize];
            let e = agg.entry(root[ch]).or_insert([0.0; 4]);
            for kk in 0..km1 {
                let x = pe[kk] - anchor[g][kk];
                let y = ce[kk] - anchor[g][kk];
                e[0] += x * y;
                e[1] += x * x;
                e[2] += pnu[kk];
                e[3] += 1.0;
            }
        }
        // Sort by thread root so the order (hence float-sum order and bootstrap draws) is
        // deterministic — HashMap iteration order is randomized.
        let mut items: Vec<(usize, [f64; 4])> = agg.into_iter().collect();
        items.sort_by_key(|(root, _)| *root);
        let threads: Vec<[f64; 4]> = items.into_iter().map(|(_, v)| v).collect();
        if threads.len() < 2 {
            return Err(PyValueError::new_err(
                "not enough reply edges across threads to estimate persistence",
            ));
        }
        // point estimate from a set of thread aggregates. Guards the degenerate case where the
        // pooled parent variance is zero (all eligible parents equal their group anchor) — the
        // regression is then unidentified, so the slope/reliability are NaN, not 0/0.
        let est = |sel: &[[f64; 4]]| -> (f64, f64, f64) {
            let mut s = [0.0f64; 4];
            for a in sel {
                for j in 0..4 {
                    s[j] += a[j];
                }
            }
            if !s[1].is_finite() || s[1] <= 0.0 || s[3] <= 0.0 {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            let a = s[0] / s[1]; // slope
            let var_x = s[1] / s[3];
            let mean_nu = s[2] / s[3];
            let rel = 1.0 - mean_nu / var_x;
            let a_corr = if rel > 0.0 { a / rel } else { f64::NAN };
            (a, rel, 1.0 - a_corr) // (observed a, reliability, structural kappa)
        };
        let (a_obs, rel, kappa_s) = est(&threads);

        // thread-clustered bootstrap
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (mut a_bs, mut k_bs): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let t = threads.len();
        let reps = bootstrap.max(1);
        for _ in 0..reps {
            let sel: Vec<[f64; 4]> = (0..t)
                .map(|_| threads[(rng.gen::<f64>() * t as f64) as usize % t])
                .collect();
            let (a, _, kap) = est(&sel);
            a_bs.push(a);
            k_bs.push(kap);
        }
        // Honest percentile CI: report it ONLY if >=90% of resamples are identifiable. Silently
        // dropping non-finite resamples would make the interval conditional on the identifiable
        // ones and look precise even when the parameter is badly non-identified (e.g. many
        // resamples hit reliability<=0), so below that threshold we return NaN bounds.
        let ci = |v: &[f64]| -> (f64, f64) {
            let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
            if f.len() < 2 || (f.len() as f64) < 0.9 * reps as f64 {
                return (f64::NAN, f64::NAN);
            }
            f.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let at = |q: f64| f[((q * (f.len() - 1) as f64).round() as usize).min(f.len() - 1)];
            (at(0.025), at(0.975))
        };
        let obs_ci = ci(&a_bs);
        let kap_ci = ci(&k_bs);

        let d = PyDict::new_bound(py);
        d.set_item("observed_persistence", a_obs)?;
        d.set_item("observed_ci", obs_ci)?;
        d.set_item("reliability", rel)?;
        d.set_item("structural_kappa", kappa_s)?;
        d.set_item("structural_kappa_ci", kap_ci)?;
        Ok(d)
    }

    /// 95% profile-likelihood CI for the reversion `κ` as a `(lower, upper)` tuple, re-optimizing
    /// (σ², p0) at each κ so the a↔σ² ridge is reflected; `(nan, nan)` when there are no reply
    /// edges or the field was not fit. Conditional on the topic fit; the point estimate is biased
    /// toward κ→0 (persistence) by topic-model shrinkage and the interval does not correct that.
    #[getter]
    fn kappa_ci(&self) -> (f64, f64) {
        self.kappa_ci
    }

    /// Reversion strength `κ = 1 - a` (0 = pure persistence / parent-copy, 1 = no memory).
    #[getter]
    fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Per-edge diffusion variance `σ²` (floored at 0.1; a returned 0.1 may be the floor, not an
    /// estimate). `NaN` when the reply field was not identified/fit.
    #[getter]
    fn sigma2(&self) -> f64 {
        self.sigma2
    }

    /// Root prior variance (floored at 0.1). `NaN` when the reply field was not identified/fit.
    #[getter]
    fn p0(&self) -> f64 {
        self.p0
    }

    /// The per-iteration variational-objective trace (sum of the per-document CTM bounds with the
    /// tree coupling plugged in as a fixed mean). This is a monitoring free energy, NOT a true
    /// ELBO for the joint tree model, so it is not guaranteed monotone.
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
