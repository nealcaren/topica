// Clippy allow-list (deliberate, for a numerical + pyo3 crate):
// - needless_range_loop: index loops are the clearer idiom for the matrix math here.
// - too_many_arguments / type_complexity: the model fit/binding entry points are
//   inherently many-parameter; the option-struct refactor is tracked separately.
// - upper_case_acronyms: the public model types (`LDA`, `STM`, `DMR`, …) are named
//   after the methods; renaming them would break the public API.
// - useless_conversion: emitted inside pyo3 0.22's `#[pymethods]`/`#[pyfunction]`
//   error-conversion codegen, not our code — noise we can't fix at the source.
// doc_* lints are rustdoc-rendering cosmetics; the user-facing docs are mkdocs.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::upper_case_acronyms,
    clippy::useless_conversion,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

// The STM/CTM/SAGE + CVB0 core lives in the `topica-core` workspace member. Re-export
// every module here so `topica::ctm::*`, `topica::corpus::*`, etc. resolve exactly as
// before — the extraction is a no-op for everything downstream of `topica`.
pub use topica_core::{corpus, ctm, cvb0, estimator, linalg, spectral, variational};
// Extension trait re-adding `Cvb0::to_topic_model` (it lives here, not in the core,
// because `TopicModel` stays in `topica`).
pub mod cvb0_ext;

pub mod btm;
#[doc(hidden)]
pub mod cli; // argument helpers for the CLI binaries (not part of the public API)
pub mod coherence;
pub mod conformance;
pub mod detm;
pub mod disclda;
pub mod dmr;
pub mod dtm;
pub mod etm;
pub mod etm_vae;
pub mod gemm;
pub mod gsdmm;
pub mod hdp;
pub mod hlda;
pub mod idealpoint;
pub mod idealpoint_lda;
pub mod infoctm;
pub mod keyatm;
pub mod labeled;
pub mod lightlda;
pub mod lsa;
pub mod mathfun;
pub mod mh;
pub mod model;
pub mod nmf;
pub mod online_lda;
pub mod optimize;
pub mod output;
pub mod pa;
pub mod party_embeddings;
pub mod pltm;
pub mod prodlda;
pub mod pt;
pub mod rtm;
pub mod sage;
pub mod sampler;
pub mod saveformat;
pub mod scholar;
pub mod seeded;
pub mod sentence_ideal;
pub mod slda;
pub mod sts;
pub mod tbip;
pub mod tlda;
pub mod warplda;
pub mod wordfish;

// Embedding-native model branch (Top2Vec/BERTopic/...): clustering pipeline over
// user-supplied embeddings. Behind the `embeddings` feature (implied by `python`).
// reduce -> cluster -> represent are the three pipeline stages.
#[cfg(feature = "embeddings")]
pub mod bertopic;
#[cfg(feature = "embeddings")]
pub mod cluster;
pub mod fastopic;
#[cfg(feature = "embeddings")]
pub mod reduce;
#[cfg(feature = "embeddings")]
pub mod represent;
#[cfg(feature = "embeddings")]
pub mod semantic_signal_separation;
#[cfg(feature = "embeddings")]
pub mod top2vec;
// In-house faithful UMAP reducer (replaces the umap-rs crate). Behind the `umap`
// feature, which no longer pulls an external dependency.
#[cfg(feature = "umap")]
pub mod umap;

#[cfg(feature = "python")]
mod python;
