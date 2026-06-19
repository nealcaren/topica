//! `topica-core`: the logistic-normal structural-topic-model core extracted from
//! the `topica` crate. It holds the small, self-contained cluster the STM/CTM
//! family needs — the corpus container, anchor-word spectral initialization, the
//! shared variational kernels (Laplace E-step, M-step, SVI, L-BFGS), the
//! `Estimator` surface, the CTM/STM model and its content-covariate (SAGE)
//! extension, and the CVB0 sampler — with external dependencies collapsed to
//! `rand`, `rand_chacha`, and `rayon` (serialization is opt-in behind the
//! `serde` feature).
//!
//! `topica` re-exports every module here (`pub use topica_core::{...}`), so the
//! public `topica::ctm::*` paths and behavior are unchanged. Downstream Rust
//! consumers (e.g. the faSTM R package, which vendors a minimal crate for CRAN)
//! can depend on `topica-core` alone.

pub mod linalg;
pub mod corpus;
pub mod spectral;
pub mod variational;
pub mod estimator;
pub mod ctm;
pub mod cvb0;
