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
pub(crate) const MODEL_TAG_ECTM: u8 = 27;
// 28-30 reserved for the parked experimental trio (HyperLDA/TopicRBM/DiffusionTM).
pub(crate) const MODEL_TAG_IDEALPOINT: u8 = 31;
pub(crate) const MODEL_TAG_WORDFISH: u8 = 32;
pub(crate) const MODEL_TAG_IDEALPOINT_LDA: u8 = 33;

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
        MODEL_TAG_LSA => "LSA",
        MODEL_TAG_COMBINEDTM => "CombinedTM",
        MODEL_TAG_ZEROSHOTTM => "ZeroShotTM",
        MODEL_TAG_DETM => "DETM",
        MODEL_TAG_ECTM => "ECTM",
        MODEL_TAG_IDEALPOINT => "IdealPointTM",
        MODEL_TAG_WORDFISH => "Wordfish",
        MODEL_TAG_IDEALPOINT_LDA => "IdealPointLDA",
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
