//! Shared kernels for the logistic-normal variational family (CTM/STM/STS/ETM):
//! L-BFGS, the Laplace E-step driver, the Σ/Γ M-step, and sparse-doc prep. In
//! later steps the submodules `lbfgs`, `laplace`, `mstep`, `sparse` are added and
//! the model fits are rewired through them; for now this module only declares the
//! family trait.

pub mod lbfgs;
pub use lbfgs::lbfgs_minimize;

pub mod sparse;
pub use sparse::doc_sparse;

pub mod mstep;
pub use mstep::{fit_gamma_ridge, fit_gamma_ridge_from_ss, fit_gamma_vb, gamma_ss};

pub mod laplace;
pub use laplace::laplace_estep;

pub mod svi;

use crate::estimator::Estimator;

/// Tier-2 contract for the logistic-normal variational family. Implementing this
/// forces a model to expose its variational posterior over the latent η — the
/// gap that let STS ship without an eta posterior.
pub trait LogisticNormalModel: Estimator {
    /// Dimension of the per-document latent η (K-1 for CTM/STM, 2K-1 for STS).
    fn eta_dim(&self) -> usize;
    /// Per-document variational posterior mean λ, shape (D, eta_dim).
    fn eta_mean(&self) -> &[Vec<f64>];
    /// Per-document variational posterior covariance ν, shape (D, eta_dim²)
    /// (each inner vector is the flattened eta_dim×eta_dim matrix, row-major).
    fn eta_cov(&self) -> &[Vec<f64>];
}
