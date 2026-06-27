//! Save-file header for topica model files.
//!
//! Every model's save() prepends an 8-byte header before the bincode payload:
//!
//!   bytes 0..6 : b"TOPICA"   (magic, 6 bytes)
//!   byte  6    : format version u8 = 1
//!   byte  7    : model tag u8 (see MODEL_TAG_* in python.rs)
//!   bytes 8..  : bincode payload
//!
//! Old (headerless) files produced a bincode panic on mismatched struct layout;
//! they now produce a clear "not a topica model file" error instead.

/// Six-byte file magic.
pub const FILE_MAGIC: &[u8; 6] = b"TOPICA";
/// Current on-disk format version.
pub const FORMAT_VERSION: u8 = 1;

/// Serialize `state` into a headed byte buffer (magic + version + tag + bincode).
pub fn encode_state<S: serde::Serialize>(model_tag: u8, state: &S) -> Result<Vec<u8>, String> {
    let payload = bincode::serialize(state).map_err(|e| format!("serialization failed: {e}"))?;
    let mut buf = Vec::with_capacity(8 + payload.len());
    buf.extend_from_slice(FILE_MAGIC);
    buf.push(FORMAT_VERSION);
    buf.push(model_tag);
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Read just the model tag from a headed byte buffer, validating the magic and
/// version. Used by models whose `load` accepts more than one tag (e.g. a class
/// that absorbed a sibling's save format).
pub fn peek_tag(bytes: &[u8]) -> Result<u8, String> {
    if bytes.len() < 8 || &bytes[..6] != FILE_MAGIC {
        return Err(
            "not a topica model file (unrecognized header; file may be corrupt or saved by an older version)".into()
        );
    }
    let version = bytes[6];
    if version != FORMAT_VERSION {
        return Err(format!(
            "unsupported save-format version {version} (this build supports version {FORMAT_VERSION})"
        ));
    }
    Ok(bytes[7])
}

/// Deserialize a byte buffer that was written by `encode_state`.
/// Returns a clear error if the header is missing, wrong version, or wrong model tag.
pub fn decode_state<S: serde::de::DeserializeOwned>(
    bytes: &[u8],
    expected_tag: u8,
    tag_name: fn(u8) -> &'static str,
) -> Result<S, String> {
    if bytes.len() < 8 || &bytes[..6] != FILE_MAGIC {
        return Err(
            "not a topica model file (unrecognized header; file may be corrupt or saved by an older version)".into()
        );
    }
    let version = bytes[6];
    if version != FORMAT_VERSION {
        return Err(format!(
            "unsupported save-format version {version} (this build supports version {FORMAT_VERSION})"
        ));
    }
    let file_tag = bytes[7];
    if file_tag != expected_tag {
        return Err(format!(
            "file was saved by model {} but you are trying to load it as {}",
            tag_name(file_tag),
            tag_name(expected_tag),
        ));
    }
    bincode::deserialize(&bytes[8..]).map_err(|e| format!("not a valid topica model file: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    // Minimal tagged structs for testing.
    const TAG_FOO: u8 = 1;
    const TAG_BAR: u8 = 2;

    fn tag_name(t: u8) -> &'static str {
        match t {
            TAG_FOO => "Foo",
            TAG_BAR => "Bar",
            _ => "unknown",
        }
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct FooState {
        x: u32,
        y: f64,
        flags: bool,
        draws: Option<Vec<f32>>,
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct BarState {
        name: String,
        seeds: Vec<String>,
        residual: usize,
    }

    // --- Round-trip correctness ---

    #[test]
    fn foo_state_round_trip() {
        let orig = FooState {
            x: 42,
            y: 1.25,
            flags: true,
            draws: Some(vec![0.1, 0.2, 0.3]),
        };
        let buf = encode_state(TAG_FOO, &orig).unwrap();
        let loaded: FooState = decode_state(&buf, TAG_FOO, tag_name).unwrap();
        assert_eq!(loaded, orig);
    }

    #[test]
    fn bar_state_round_trip_preserves_seeds_and_residual() {
        let orig = BarState {
            name: "test_model".into(),
            seeds: vec!["politics".into(), "economy".into()],
            residual: 3,
        };
        let buf = encode_state(TAG_BAR, &orig).unwrap();
        let loaded: BarState = decode_state(&buf, TAG_BAR, tag_name).unwrap();
        assert_eq!(loaded.seeds, orig.seeds);
        assert_eq!(loaded.residual, orig.residual);
        // seeds.len() + residual = 2 + 3 = 5 (mirrors SeededLDA.num_topics_val())
        assert_eq!(loaded.seeds.len() + loaded.residual, 5);
    }

    #[test]
    fn theta_draws_survive_round_trip() {
        let draws: Vec<f32> = (0..30).map(|i| i as f32 / 30.0).collect();
        let orig = FooState {
            x: 1,
            y: 0.0,
            flags: false,
            draws: Some(draws.clone()),
        };
        let buf = encode_state(TAG_FOO, &orig).unwrap();
        let loaded: FooState = decode_state(&buf, TAG_FOO, tag_name).unwrap();
        assert_eq!(loaded.draws.unwrap(), draws);
    }

    // --- Wrong-model-tag gives a clear, human-readable error ---

    #[test]
    fn wrong_model_tag_gives_clear_error() {
        let orig = FooState {
            x: 1,
            y: 2.0,
            flags: false,
            draws: None,
        };
        let buf = encode_state(TAG_FOO, &orig).unwrap();

        let result: Result<BarState, String> = decode_state(&buf, TAG_BAR, tag_name);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("Foo"),
            "error should mention the file's model: {msg}"
        );
        assert!(
            msg.contains("Bar"),
            "error should mention the expected model: {msg}"
        );
    }

    // --- Garbage / truncated buffers give 'not a topica model file' ---

    #[test]
    fn garbage_bytes_give_clear_error() {
        let result: Result<FooState, String> = decode_state(b"not a model", TAG_FOO, tag_name);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("not a topica model file"),
            "error should say 'not a topica model file': {msg}"
        );
    }

    #[test]
    fn empty_buffer_gives_clear_error() {
        let result: Result<FooState, String> = decode_state(b"", TAG_FOO, tag_name);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("not a topica model file"),
            "error for empty buffer should say 'not a topica model file': {msg}"
        );
    }

    #[test]
    fn wrong_version_gives_clear_error() {
        let orig = FooState {
            x: 1,
            y: 2.0,
            flags: false,
            draws: None,
        };
        let mut buf = encode_state(TAG_FOO, &orig).unwrap();
        // Overwrite the version byte with an unsupported value.
        buf[6] = 99;
        let result: Result<FooState, String> = decode_state(&buf, TAG_FOO, tag_name);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("unsupported save-format version"),
            "error should mention unsupported version: {msg}"
        );
    }
}
