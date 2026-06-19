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

// Clippy: these lints are allowed deliberately for the numerical core.
// - needless_range_loop: index loops (`for i in 0..n { a[i][j] = ... }`) are the
//   clearer, idiomatic form for the matrix/tensor math here; rewriting them as
//   iterator chains hurts readability of the linear algebra and risks subtle bugs.
// - too_many_arguments / type_complexity: the variational fit entry points
//   (`fit_ctm`, `ctm_hpb`, …) are inherently many-parameter; the option/context
//   struct refactor is tracked separately (and `fit_ctm` is part of the faSTM
//   surface, so its shape is changed only in a coordinated step).
// doc_* lints are rustdoc-rendering cosmetics; the user-facing docs are mkdocs.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod corpus;
pub mod ctm;
pub mod cvb0;
pub mod estimator;
pub mod linalg;
pub mod spectral;
pub mod variational;
