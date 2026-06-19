//! The core estimator contract shared by topica's fitted models. Traits operate
//! on the fitted *model structs* (CtmModel, StsModel, ...), not the PyO3
//! pyclasses, so they are testable in `cargo test --lib` with no Python.
//!
//! Mirrors the Python-side contract in `python/topica/conformance.py`: a Tier-0
//! base trait every estimator implements, plus family traits for the Tier-2
//! requirements (Dirichlet: alpha/theta_draws/doc_lengths; logistic-normal:
//! eta_mean/eta_cov — the latter lives in `crate::variational`).

/// Which inference family a fitted model belongs to. The single discriminator a
/// conformance check keys off (mirrors `effects.py::model_family`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Collapsed-Gibbs / Dirichlet-prior models (LDA family).
    Dirichlet,
    /// Logistic-normal variational models (CTM/STM/STS/ETM).
    LogisticNormal,
    /// Neural, nonparametric, or clustering models with no posterior over theta.
    None_,
}

/// Tier-0 numeric surface every fitted topica model exposes. Intentionally
/// minimal — only the methods that genuinely generalize across families.
/// (`vocabulary`, `top_words`, `coherence`, `save`/`load` stay in the binding
/// layer because they need the `Corpus`/serde state that lives on the pyclass.)
pub trait Estimator {
    /// Number of topics K.
    fn num_topics(&self) -> usize;
    /// Topic-word matrix, shape (K, V). Returned by value: most Gibbs models
    /// compute φ on demand from their count tables rather than storing it, so a
    /// borrowed slice cannot be the shared contract. Logistic-normal models that
    /// do store β simply clone it (a cold, inspection-path call).
    fn topic_word(&self) -> Vec<Vec<f64>>;
    /// Document-topic matrix, shape (D, K); rows sum to 1.
    fn doc_topic(&self) -> Vec<Vec<f64>>;
    /// Per-iteration convergence trace as (iteration, objective) pairs; empty if
    /// the model keeps no trace.
    fn fit_history(&self) -> Vec<(usize, f64)>;
    /// Whether fitting stopped on a tolerance criterion; `None` for
    /// non-iterative models.
    fn converged(&self) -> Option<bool>;
    /// The model's inference family.
    fn model_family(&self) -> ModelFamily;
}

/// Tier-2 contract for the Dirichlet (collapsed-Gibbs) family.
pub trait DirichletModel: Estimator {
    /// Document-topic Dirichlet concentration, length K. Returned by value:
    /// most Gibbs models store a symmetric scalar α and broadcast it to length
    /// K here; the asymmetric-α models (LDA/SAGE) clone their stored vector.
    fn alpha(&self) -> Vec<f64>;
    /// Retained MCMC theta draws, shape (S, D, K); empty if not retained.
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>>;
    /// Per-document token counts, length D.
    fn doc_lengths(&self) -> Vec<usize>;
}
