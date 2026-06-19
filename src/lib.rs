// The STM/CTM/SAGE + CVB0 core lives in the `topica-core` workspace member. Re-export
// every module here so `topica::ctm::*`, `topica::corpus::*`, etc. resolve exactly as
// before — the extraction is a no-op for everything downstream of `topica`.
pub use topica_core::{corpus, ctm, cvb0, estimator, linalg, spectral, variational};
// Extension trait re-adding `Cvb0::to_topic_model` (it lives here, not in the core,
// because `TopicModel` stays in `topica`).
pub mod cvb0_ext;

pub mod saveformat;
pub mod coherence;
pub mod conformance;
pub mod dmr;
pub mod ectm;
pub mod sts;
pub mod detm;
pub mod etm;
pub mod etm_vae;
pub mod dtm;
pub mod gsdmm;
pub mod hdp;
pub mod hlda;
pub mod infoctm;
pub mod keyatm;
pub mod labeled;
pub mod lightlda;
pub mod lsa;
pub mod mh;
pub mod model;
pub mod nmf;
pub mod optimize;
pub mod output;
pub mod pa;
pub mod prodlda;
pub mod pt;
pub mod sage;
pub mod sampler;
pub mod seeded;
pub mod slda;
pub mod warplda;

// Embedding-native model branch (Top2Vec/BERTopic/...): clustering pipeline over
// user-supplied embeddings. Behind the `embeddings` feature (implied by `python`).
// reduce -> cluster -> represent are the three pipeline stages.
#[cfg(feature = "embeddings")]
pub mod cluster;
#[cfg(feature = "embeddings")]
pub mod reduce;
#[cfg(feature = "embeddings")]
pub mod represent;
#[cfg(feature = "embeddings")]
pub mod top2vec;
#[cfg(feature = "embeddings")]
pub mod bertopic;
pub mod fastopic;

#[cfg(feature = "python")]
mod python;
