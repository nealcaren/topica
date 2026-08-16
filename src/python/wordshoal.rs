//! Wordshoal pyclass: Lauderdale & Herzog's (2016) two-stage scaling of actor
//! positions from texts grouped into externally-known debate **domains**. The
//! multi-domain sibling of `Wordfish`. `use super::*` pulls in the shared bindings
//! (Corpus, arrays, save/load).

use super::*;
use pyo3::types::PyDict;

use crate::wordshoal::{self, WordshoalModel};
use std::collections::HashMap;

/// Coerce a per-document label column to `Vec<String>`, accepting a list of strings
/// or of any Python objects (ints, floats, ...) by `str()`-ing each — a researcher
/// naturally passes an integer id / day column, and an opaque `TypeError` there is a
/// bad first-run experience (Gate-B sample-user T4-a).
fn coerce_labels(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<String>> {
    if let Ok(v) = obj.extract::<Vec<String>>() {
        return Ok(v);
    }
    let seq = obj
        .iter()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a sequence of labels")))?;
    let mut out = Vec::new();
    for item in seq {
        out.push(item?.str()?.extract::<String>()?);
    }
    Ok(out)
}

#[pyclass(module = "topica")]
pub struct Wordshoal {
    theta_prior_sd: f64,
    loading_prior_sd: f64,
    intercept_prior_sd: f64,
    tau_prior: f64,
    min_count: usize,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    author_names: Vec<String>,
    domain_names: Vec<String>,
    id_to_word: Vec<String>,
    model: Option<WordshoalModel>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WordshoalState {
    theta_prior_sd: f64,
    loading_prior_sd: f64,
    intercept_prior_sd: f64,
    tau_prior: f64,
    min_count: usize,
    convergence_tol: f64,
    seed: u64,
    fitted: bool,
    author_names: Vec<String>,
    domain_names: Vec<String>,
    id_to_word: Vec<String>,
    num_authors: Option<usize>,
    num_domains: Option<usize>,
    theta: Option<Vec<f64>>,
    tau: Option<Vec<f64>>,
    alpha: Option<Vec<f64>>,
    beta: Option<Vec<f64>>,
    position_se: Option<Vec<f64>>,
    log_posterior: Option<f64>,
    lp_history: Option<Vec<f64>>,
    converged: Option<bool>,
    iters_run: Option<usize>,
    domain_word_ids: Option<Vec<Vec<u32>>>,
    domain_word_beta: Option<Vec<Vec<f64>>>,
    psi: Option<Vec<f64>>,
    num_components: Option<usize>,
    #[serde(default)]
    author_components: Option<Vec<usize>>,
}

impl Wordshoal {
    fn fitted_model(&self) -> PyResult<&WordshoalModel> {
        self.model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
    }
}

#[pymethods]
impl Wordshoal {
    /// Create an unfitted Wordshoal model. The prior standard deviations are the
    /// **stage-2** cross-domain factor-model priors: `theta_prior_sd` on the actor
    /// positions (also the actor-update ridge), `loading_prior_sd` on the per-domain
    /// loadings, `intercept_prior_sd` on the per-domain intercepts; `tau_prior` is
    /// the Gamma(shape=rate) prior on the per-actor precisions. `min_count` drops
    /// words occurring fewer than that many times across the whole corpus before
    /// stage 1; `convergence_tol` stops the stage-2 coordinate ascent on the
    /// log-posterior increase and is also passed to the per-domain Wordfish fits.
    /// The stage-1 Wordfish priors are hardwired to the quanteda defaults
    /// (`beta=3`, `theta=1`) and are not exposed here. `seed` is accepted for API
    /// uniformity — the fit is deterministic.
    #[new]
    #[pyo3(signature = (*, theta_prior_sd=1.0, loading_prior_sd=0.5,
                        intercept_prior_sd=0.5, tau_prior=1.0, min_count=1,
                        convergence_tol=1e-3, seed=13))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        theta_prior_sd: f64,
        loading_prior_sd: f64,
        intercept_prior_sd: f64,
        tau_prior: f64,
        min_count: usize,
        convergence_tol: f64,
        seed: u64,
    ) -> PyResult<Self> {
        for (name, v) in [
            ("theta_prior_sd", theta_prior_sd),
            ("loading_prior_sd", loading_prior_sd),
            ("intercept_prior_sd", intercept_prior_sd),
            ("tau_prior", tau_prior),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be a finite positive number; got {v}"
                )));
            }
        }
        ensure_finite_nonneg("convergence_tol", convergence_tol)?;
        Ok(Wordshoal {
            theta_prior_sd,
            loading_prior_sd,
            intercept_prior_sd,
            tau_prior,
            min_count: min_count.max(1),
            convergence_tol,
            seed,
            fitted: false,
            author_names: Vec::new(),
            domain_names: Vec::new(),
            id_to_word: Vec::new(),
            model: None,
        })
    }

    /// The random seed the model was constructed with (the fit is deterministic).
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Constructor configuration as a JSON-serialisable dict keyed by `__init__`
    /// argument name (#400).
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("theta_prior_sd", self.theta_prior_sd)?;
        d.set_item("loading_prior_sd", self.loading_prior_sd)?;
        d.set_item("intercept_prior_sd", self.intercept_prior_sd)?;
        d.set_item("tau_prior", self.tau_prior)?;
        d.set_item("min_count", self.min_count)?;
        d.set_item("convergence_tol", self.convergence_tol)?;
        d.set_item("seed", self.seed)?;
        Ok(d)
    }

    /// Fit on `data` (a Corpus or list of token lists). `speakers` and `domains` are
    /// per-document label lists (length num_docs): each document belongs to one
    /// actor and one externally-observed debate domain. Stage 1 scales each domain's
    /// documents with Wordfish; stage 2 combines the within-domain positions into a
    /// single cross-domain actor scale. `anchors` is an optional
    /// `{speaker_label: value}` map used to orient the sign of the axis. `iters`
    /// caps the stage-2 coordinate ascent (default 100).
    #[pyo3(signature = (data, *, speakers, domains, anchors=None, iters=None,
                        convergence_tol=None))]
    fn fit(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        speakers: &Bound<'_, PyAny>,
        domains: &Bound<'_, PyAny>,
        anchors: Option<HashMap<String, f64>>,
        iters: Option<usize>,
        convergence_tol: Option<f64>,
    ) -> PyResult<Py<Self>> {
        let speakers = coerce_labels(speakers, "speakers")?;
        let domains = coerce_labels(domains, "domains")?;
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
        if speakers.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "speakers must have length num_docs ({num_docs}), got {}",
                speakers.len()
            )));
        }
        if domains.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "domains must have length num_docs ({num_docs}), got {}",
                domains.len()
            )));
        }

        // Sorted-label -> index (matches the reference's alphabetical R factor
        // levels, so the deterministic linspace(-2, 2) init lands in the same basin).
        let index_of = |labels: &[String]| -> (Vec<usize>, Vec<String>) {
            let mut names: Vec<String> = labels.to_vec();
            names.sort();
            names.dedup();
            let idx: HashMap<&str, usize> = names
                .iter()
                .enumerate()
                .map(|(i, s)| (s.as_str(), i))
                .collect();
            (labels.iter().map(|l| idx[l.as_str()]).collect(), names)
        };
        let (speaker_idx, author_names) = index_of(&speakers);
        let (domain_idx, domain_names) = index_of(&domains);
        let num_authors = author_names.len();
        let num_domains = domain_names.len();
        if num_authors < 2 {
            return Err(PyValueError::new_err(
                "Wordshoal needs at least 2 distinct speakers to scale",
            ));
        }
        if num_domains < 1 {
            return Err(PyValueError::new_err("Wordshoal needs at least 1 domain"));
        }

        // Every domain needs >= 2 documents to run Wordfish (matches the reference,
        // which errors on a single-document group).
        let mut domain_doc_count = vec![0usize; num_domains];
        for &j in &domain_idx {
            domain_doc_count[j] += 1;
        }
        if let Some(j) = domain_doc_count.iter().position(|&c| c < 2) {
            return Err(PyValueError::new_err(format!(
                "domain {:?} has fewer than 2 documents; every domain needs >= 2 to scale",
                domain_names[j]
            )));
        }

        // Global vocabulary: corpus frequency >= min_count, ordered by descending
        // frequency then word (deterministic).
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for doc in &docs_str {
            for w in doc {
                *freq.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        let mut vocab_pairs: Vec<(&str, usize)> = freq
            .iter()
            .filter(|&(_, &c)| c >= slf.min_count)
            .map(|(&w, &c)| (w, c))
            .collect();
        vocab_pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let id_to_word: Vec<String> = vocab_pairs.iter().map(|&(w, _)| w.to_string()).collect();
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

        // Per-document sparse global-word-id counts.
        let docs: Vec<Vec<(u32, f64)>> = docs_str
            .iter()
            .map(|doc| {
                let mut m: HashMap<u32, f64> = HashMap::new();
                for w in doc {
                    if let Some(&wid) = word_id.get(w.as_str()) {
                        *m.entry(wid).or_insert(0.0) += 1.0;
                    }
                }
                let mut v: Vec<(u32, f64)> = m.into_iter().collect();
                v.sort_by_key(|&(w, _)| w);
                v
            })
            .collect();

        // Anchors -> (actor_index, target), sorted for a fixed-order sign check.
        let anchor_pairs: Vec<(usize, f64)> = match &anchors {
            None => Vec::new(),
            Some(m) => {
                let mut pairs = Vec::with_capacity(m.len());
                for (label, &target) in m {
                    let i = author_names
                        .iter()
                        .position(|x| x == label)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "anchor label {label:?} is not a speaker label"
                            ))
                        })?;
                    pairs.push((i, target));
                }
                pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
                pairs
            }
        };

        let tol = convergence_tol.unwrap_or(slf.convergence_tol);
        let it = iters.unwrap_or(100);
        let (tsd, lsd, isd, tp) = (
            slf.theta_prior_sd,
            slf.loading_prior_sd,
            slf.intercept_prior_sd,
            slf.tau_prior,
        );
        let model = py.allow_threads(move || {
            wordshoal::fit_wordshoal(
                &docs,
                num_types,
                &speaker_idx,
                num_authors,
                &domain_idx,
                num_domains,
                &anchor_pairs,
                it,
                tol,
                tsd,
                lsd,
                isd,
                tp,
            )
        });

        // Warn (not error — the reference accepts it) when the speaker-domain graph
        // is disconnected, so the cross-component scale is not identified.
        if model.num_components > 1 {
            let warnings = py.import_bound("warnings")?;
            let msg = format!(
                "the speaker-domain graph has {} disconnected components; actor \
                 positions are NOT comparable across components (the scale is only \
                 identified within a connected component)",
                model.num_components
            );
            warnings.call_method1("warn", (msg,))?;
        }

        slf.model = Some(model);
        slf.id_to_word = id_to_word;
        slf.author_names = author_names;
        slf.domain_names = domain_names;
        slf.fitted = true;
        Ok(slf.into())
    }

    #[getter]
    fn num_authors(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_authors)
    }
    #[getter]
    fn num_domains(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_domains)
    }

    /// Actor positions as a (num_authors, 1) matrix — the cross-domain latent scale.
    ///
    /// Identifiability: prior-identified (the `theta ~ N(0, theta_prior_sd^2)` prior
    /// fixes the scale), and identified only up to **sign**. Pass `anchors` to
    /// `fit()` to orient it; with no anchors the axis is default-oriented so the
    /// first two speakers (in sorted label order) satisfy `theta[0] < theta[1]`,
    /// mirroring quanteda `dir = c(1, 2)`. Unlike `Wordfish`, positions are NOT
    /// re-standardized to unit variance — this matches the Wordshoal reference's
    /// identification.
    #[getter]
    fn author_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let pos: Vec<Vec<f64>> = m.theta.iter().map(|&t| vec![t]).collect();
        Ok(vecs_to_arr2(&pos).to_pyarray_bound(py))
    }
    /// Standard error of each actor position (num_authors,), aligned to
    /// `author_names`: `sqrt((b'b + theta_prior_sd^-2)^-1 / tau_i)`, the reference's
    /// `se.theta`.
    #[getter]
    fn position_se<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.position_se.clone()).to_pyarray_bound(py))
    }
    /// The speaker labels, in the row order of `author_positions` (sorted).
    #[getter]
    fn author_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.author_names.clone())
    }
    /// The domain labels, in the row order of `domain_scales`. Sorted
    /// **lexicographically as strings** (e.g. `"10"` before `"9"`), not numerically —
    /// cast a numeric key to a zero-padded string first if you need numeric/time order.
    #[getter]
    fn domain_names(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.domain_names.clone())
    }
    /// Per-domain `[intercept alpha_j, loading beta_j]` as a (num_domains, 2)
    /// matrix, row order matching `domain_names`. The loading `beta_j` is how
    /// strongly domain `j` discriminates on the shared scale (and absorbs the
    /// domain's arbitrary within-domain orientation — its sign is not meaningful in
    /// isolation).
    #[getter]
    fn domain_scales<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.fitted_model()?;
        let rows: Vec<Vec<f64>> = (0..m.num_domains)
            .map(|j| vec![m.alpha[j], m.beta[j]])
            .collect();
        Ok(vecs_to_arr2(&rows).to_pyarray_bound(py))
    }
    /// Per-actor precision `tau_i` (num_authors,), aligned to `author_names`. Higher
    /// precision = a more consistently-placed actor across domains.
    #[getter]
    fn author_precision<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(Array1::from(self.fitted_model()?.tau.clone()).to_pyarray_bound(py))
    }

    /// Stage-1 within-domain word discriminations for one domain, as `(word, beta)`
    /// pairs sorted by descending discrimination. `domain` is a domain label. These
    /// are the per-debate Wordfish betas — the words that move a speaker along that
    /// debate's own axis.
    #[pyo3(signature = (domain, n=None))]
    fn word_scores(&self, domain: &str, n: Option<usize>) -> PyResult<Vec<(String, f64)>> {
        let m = self.fitted_model()?;
        let j = self
            .domain_names
            .iter()
            .position(|d| d == domain)
            .ok_or_else(|| {
                PyValueError::new_err(format!("domain {domain:?} is not a domain label"))
            })?;
        let mut pairs: Vec<(String, f64)> = m.domain_word_ids[j]
            .iter()
            .zip(m.domain_word_beta[j].iter())
            .map(|(&w, &b)| (self.id_to_word[w as usize].clone(), b))
            .collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        if let Some(n) = n {
            pairs.truncate(n);
        }
        Ok(pairs)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.fitted_model()?;
        Ok(self.id_to_word.clone())
    }
    /// The stage-2 log-posterior at convergence (the reference's convergence
    /// quantity — a log-posterior including the four log-priors, not a bare
    /// log-likelihood).
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        Ok(self.fitted_model()?.log_posterior)
    }
    /// Stage-2 convergence trace: `(iter, log_posterior)` pairs.
    #[getter]
    fn fit_history(&self) -> PyResult<Vec<(usize, f64)>> {
        Ok(self
            .fitted_model()?
            .lp_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i, b))
            .collect())
    }
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn converged(&self) -> PyResult<Option<bool>> {
        Ok(Some(self.fitted_model()?.converged))
    }
    /// Alias of :attr:`converged` under the name that says what the flag means:
    /// True only if the fit early-stopped on `convergence_tol`; False when the
    /// full `iters` ran. `converged` is kept as an alias (issue #755).
    /// :func:`topica.stop_reason` turns this flag into a plain-language summary of
    /// why the fit stopped (tolerance met, ``iters`` cap hit, or no early-stop
    /// criterion for this model).
    #[getter]
    fn early_stopped(&self) -> PyResult<Option<bool>> {
        Ok(Some(self.fitted_model()?.converged))
    }
    #[getter]
    fn iters_run(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.iters_run)
    }
    /// The number of connected components of the speaker-domain bipartite graph
    /// (edges only through domains that scaled). `1` means every actor's position is
    /// comparable; `> 1` means the scale is not identified across components (a
    /// warning is emitted at fit time).
    #[getter]
    fn num_components(&self) -> PyResult<usize> {
        Ok(self.fitted_model()?.num_components)
    }
    /// Component label of each actor (num_authors,), aligned to `author_names` /
    /// `author_positions`. Actors with different labels are on **non-comparable**
    /// scales — group by this before comparing positions when `num_components > 1`.
    #[getter]
    fn author_components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let c: Vec<i64> = self
            .fitted_model()?
            .author_components
            .iter()
            .map(|&x| x as i64)
            .collect();
        Ok(Array1::from(c).to_pyarray_bound(py))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let m = self.model.as_ref();
        write_state(
            path,
            MODEL_TAG_WORDSHOAL,
            &WordshoalState {
                theta_prior_sd: self.theta_prior_sd,
                loading_prior_sd: self.loading_prior_sd,
                intercept_prior_sd: self.intercept_prior_sd,
                tau_prior: self.tau_prior,
                min_count: self.min_count,
                convergence_tol: self.convergence_tol,
                seed: self.seed,
                fitted: self.fitted,
                author_names: self.author_names.clone(),
                domain_names: self.domain_names.clone(),
                id_to_word: self.id_to_word.clone(),
                num_authors: m.map(|m| m.num_authors),
                num_domains: m.map(|m| m.num_domains),
                theta: m.map(|m| m.theta.clone()),
                tau: m.map(|m| m.tau.clone()),
                alpha: m.map(|m| m.alpha.clone()),
                beta: m.map(|m| m.beta.clone()),
                position_se: m.map(|m| m.position_se.clone()),
                log_posterior: m.map(|m| m.log_posterior),
                lp_history: m.map(|m| m.lp_history.clone()),
                converged: m.map(|m| m.converged),
                iters_run: m.map(|m| m.iters_run),
                domain_word_ids: m.map(|m| m.domain_word_ids.clone()),
                domain_word_beta: m.map(|m| m.domain_word_beta.clone()),
                psi: m.map(|m| m.psi.clone()),
                num_components: m.map(|m| m.num_components),
                author_components: m.map(|m| m.author_components.clone()),
            },
        )
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: WordshoalState = read_state(path, MODEL_TAG_WORDSHOAL)?;
        let model = if s.fitted && s.theta.is_some() {
            Some(WordshoalModel {
                num_authors: s.num_authors.unwrap_or(0),
                num_domains: s.num_domains.unwrap_or(0),
                theta: s.theta.unwrap_or_default(),
                tau: s.tau.unwrap_or_default(),
                alpha: s.alpha.unwrap_or_default(),
                beta: s.beta.unwrap_or_default(),
                position_se: s.position_se.unwrap_or_default(),
                log_posterior: s.log_posterior.unwrap_or(f64::NAN),
                lp_history: s.lp_history.unwrap_or_default(),
                converged: s.converged.unwrap_or(false),
                iters_run: s.iters_run.unwrap_or(0),
                domain_word_ids: s.domain_word_ids.unwrap_or_default(),
                domain_word_beta: s.domain_word_beta.unwrap_or_default(),
                psi: s.psi.unwrap_or_default(),
                num_components: s.num_components.unwrap_or(1),
                author_components: s
                    .author_components
                    .unwrap_or_else(|| vec![0usize; s.num_authors.unwrap_or(0)]),
            })
        } else {
            None
        };
        Ok(Wordshoal {
            theta_prior_sd: s.theta_prior_sd,
            loading_prior_sd: s.loading_prior_sd,
            intercept_prior_sd: s.intercept_prior_sd,
            tau_prior: s.tau_prior,
            min_count: s.min_count,
            convergence_tol: s.convergence_tol,
            seed: s.seed,
            fitted: s.fitted,
            author_names: s.author_names,
            domain_names: s.domain_names,
            id_to_word: s.id_to_word,
            model,
        })
    }

    fn __repr__(&self) -> String {
        format!("Wordshoal(fitted={})", self.fitted)
    }
}
