//! Model save/load plumbing: the per-model tag table and the generic
//! `write_state`/`read_state` wrappers over `crate::saveformat`.
//!
//! Save-file header (8 bytes prepended before the bincode payload):
//!   bytes 0..6  : b"TOPICA"   (magic, 6 bytes)
//!   byte  6     : format version u8 = 1
//!   byte  7     : model tag u8 (see MODEL_TAG_* constants below)
//!   bytes 8..   : bincode payload
//!
//! Header encode/decode logic lives in `src/saveformat.rs` (always compiled, so
//! it can have Rust unit tests without the `python` feature gate or libpython).
//! Old (headerless) files produce a clear "not a topica model file" error rather
//! than a bincode panic.

use super::error::io_err;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

// One tag per concrete model type that calls write_state / read_state.
pub(crate) const MODEL_TAG_LDA: u8 = 1;
pub(crate) const MODEL_TAG_DMR: u8 = 2;
pub(crate) const MODEL_TAG_LABELED: u8 = 3;
pub(crate) const MODEL_TAG_SAGE: u8 = 4;
pub(crate) const MODEL_TAG_CTM: u8 = 5;
pub(crate) const MODEL_TAG_STM: u8 = 6;
pub(crate) const MODEL_TAG_STS: u8 = 7;
pub(crate) const MODEL_TAG_HDP: u8 = 8;
pub(crate) const MODEL_TAG_DTM: u8 = 9;
pub(crate) const MODEL_TAG_SLDA: u8 = 10;
pub(crate) const MODEL_TAG_PT: u8 = 11;
pub(crate) const MODEL_TAG_GSDMM: u8 = 12;
pub(crate) const MODEL_TAG_SEEDED: u8 = 13;
pub(crate) const MODEL_TAG_TOP2VEC: u8 = 14;
pub(crate) const MODEL_TAG_BERTOPIC: u8 = 15;
pub(crate) const MODEL_TAG_ETM: u8 = 16;
pub(crate) const MODEL_TAG_PRODLDA: u8 = 17;
pub(crate) const MODEL_TAG_FASTOPIC: u8 = 18;
pub(crate) const MODEL_TAG_KEYATM: u8 = 19;
pub(crate) const MODEL_TAG_PA: u8 = 20;
pub(crate) const MODEL_TAG_HLDA: u8 = 21;
pub(crate) const MODEL_TAG_NMF: u8 = 22;
pub(crate) const MODEL_TAG_LSA: u8 = 23;
pub(crate) const MODEL_TAG_COMBINEDTM: u8 = 24;
pub(crate) const MODEL_TAG_ZEROSHOTTM: u8 = 25;
pub(crate) const MODEL_TAG_DETM: u8 = 26;
// 27 retired: ECTM removed (superseded by STM content_time); do not reuse.
// 28-30 reserved for the parked experimental trio (HyperLDA/TopicRBM/DiffusionTM).
pub(crate) const MODEL_TAG_IDEALPOINT: u8 = 31;
pub(crate) const MODEL_TAG_WORDFISH: u8 = 32;
pub(crate) const MODEL_TAG_IDEALPOINT_LDA: u8 = 33;
pub(crate) const MODEL_TAG_SENTENCE_IDEAL: u8 = 34;
pub(crate) const MODEL_TAG_TBIP: u8 = 35;
pub(crate) const MODEL_TAG_PARTY_EMBEDDINGS: u8 = 36;
pub(crate) const MODEL_TAG_TLDA: u8 = 37;
pub(crate) const MODEL_TAG_BTM: u8 = 38;
pub(crate) const MODEL_TAG_PLTM: u8 = 39;
pub(crate) const MODEL_TAG_DISCLDA: u8 = 40;
pub(crate) const MODEL_TAG_SCHOLAR: u8 = 41;
pub(crate) const MODEL_TAG_RTM: u8 = 42;
pub(crate) const MODEL_TAG_INFOCTM: u8 = 43;
pub(crate) const MODEL_TAG_ONLINE_LDA: u8 = 44;
pub(crate) const MODEL_TAG_S3: u8 = 45;
pub(crate) const MODEL_TAG_FLDA: u8 = 46;
pub(crate) const MODEL_TAG_GUIDED_NMF: u8 = 47;
pub(crate) const MODEL_TAG_COREX: u8 = 48;
pub(crate) const MODEL_TAG_AUTHOR_TOPIC: u8 = 49;
pub(crate) const MODEL_TAG_MGLDA: u8 = 50;
pub(crate) const MODEL_TAG_TOPICS_OVER_TIME: u8 = 51;
pub(crate) const MODEL_TAG_GAUSSIAN_LDA: u8 = 52;
pub(crate) const MODEL_TAG_WORDSHOAL: u8 = 53;
pub(crate) const MODEL_TAG_TOPICAL_NGRAMS: u8 = 54;

pub(crate) fn model_tag_name(tag: u8) -> &'static str {
    match tag {
        MODEL_TAG_LDA => "LDA",
        MODEL_TAG_DMR => "DMR",
        MODEL_TAG_LABELED => "LabeledLDA",
        MODEL_TAG_SAGE => "SAGE",
        MODEL_TAG_CTM => "CTM",
        MODEL_TAG_STM => "STM",
        MODEL_TAG_STS => "STS",
        MODEL_TAG_HDP => "HDP",
        MODEL_TAG_DTM => "DTM",
        MODEL_TAG_SLDA => "SupervisedLDA",
        MODEL_TAG_PT => "PT",
        MODEL_TAG_GSDMM => "GSDMM",
        MODEL_TAG_SEEDED => "SeededLDA",
        MODEL_TAG_TOP2VEC => "Top2Vec",
        MODEL_TAG_BERTOPIC => "BERTopic",
        MODEL_TAG_ETM => "ETM",
        MODEL_TAG_PRODLDA => "ProdLDA",
        MODEL_TAG_FASTOPIC => "FASTopic",
        MODEL_TAG_KEYATM => "KeyATM",
        MODEL_TAG_PA => "PA",
        MODEL_TAG_HLDA => "HLDA",
        MODEL_TAG_NMF => "NMF",
        MODEL_TAG_GUIDED_NMF => "GuidedNMF",
        MODEL_TAG_COREX => "CorEx",
        MODEL_TAG_AUTHOR_TOPIC => "AuthorTopic",
        MODEL_TAG_MGLDA => "MGLDA",
        MODEL_TAG_TOPICS_OVER_TIME => "TopicsOverTime",
        MODEL_TAG_GAUSSIAN_LDA => "GaussianLDA",
        MODEL_TAG_WORDSHOAL => "Wordshoal",
        MODEL_TAG_TOPICAL_NGRAMS => "TopicalNGrams",
        MODEL_TAG_LSA => "LSA",
        MODEL_TAG_COMBINEDTM => "CombinedTM",
        MODEL_TAG_ZEROSHOTTM => "ZeroShotTM",
        MODEL_TAG_DETM => "DETM",
        // Tags 31 and 33 are both IdealPointTM: word-embedding and count
        // representations of the same model, merged into one pyclass.
        MODEL_TAG_IDEALPOINT => "IdealPointTM",
        MODEL_TAG_WORDFISH => "Wordfish",
        MODEL_TAG_IDEALPOINT_LDA => "IdealPointTM",
        MODEL_TAG_SENTENCE_IDEAL => "IdealPointSentenceTM",
        MODEL_TAG_TBIP => "TBIP",
        MODEL_TAG_PARTY_EMBEDDINGS => "PartyEmbeddings",
        MODEL_TAG_TLDA => "TensorLDA",
        MODEL_TAG_BTM => "BTM",
        MODEL_TAG_PLTM => "PolylingualLDA",
        MODEL_TAG_DISCLDA => "DiscLDA",
        MODEL_TAG_SCHOLAR => "Scholar",
        MODEL_TAG_RTM => "RTM",
        MODEL_TAG_INFOCTM => "InfoCTM",
        MODEL_TAG_ONLINE_LDA => "OnlineLDA",
        MODEL_TAG_S3 => "SemanticSignalSeparation",
        _ => "unknown",
    }
}

pub(crate) fn write_state<S: serde::Serialize>(
    path: &str,
    model_tag: u8,
    state: &S,
) -> PyResult<()> {
    let buf = crate::saveformat::encode_state(model_tag, state).map_err(PyValueError::new_err)?;
    std::fs::write(path, buf).map_err(io_err)
}
pub(crate) fn read_state<S: serde::de::DeserializeOwned>(
    path: &str,
    expected_tag: u8,
) -> PyResult<S> {
    let bytes = std::fs::read(path).map_err(io_err)?;
    crate::saveformat::decode_state(&bytes, expected_tag, model_tag_name)
        .map_err(PyValueError::new_err)
}

/// Read just the model tag from a save file, for classes whose `load` dispatches
/// on the tag (e.g. IdealPointTM, which reads both its word-embedding and count
/// save formats).
pub(crate) fn peek_model_tag(path: &str) -> PyResult<u8> {
    let bytes = std::fs::read(path).map_err(io_err)?;
    crate::saveformat::peek_tag(&bytes).map_err(PyValueError::new_err)
}
