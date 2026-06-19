//! Tiny argument helpers shared by the CLI binaries (train/analyze/preprocess/
//! show). They bounds-check option values so a missing or unparseable value
//! prints a usage error and returns `None` (the binaries' "show usage" signal),
//! instead of panicking on an out-of-range `raw[i]` index.
//!
//! Doc-hidden: this is binary-support glue, not part of the public API.

/// Advance `i` to the value following a flag and return it, or print an error
/// and return `None` if the flag was the final token.
pub fn next_val(raw: &[String], i: &mut usize) -> Option<String> {
    *i += 1;
    match raw.get(*i) {
        Some(v) => Some(v.clone()),
        None => {
            eprintln!("error: '{}' requires a value", raw[*i - 1]);
            None
        }
    }
}

/// Like [`next_val`], but parse the value into `T`; prints an error and returns
/// `None` on a missing or unparseable value.
pub fn parse_val<T: std::str::FromStr>(raw: &[String], i: &mut usize) -> Option<T> {
    let flag_pos = *i;
    let v = next_val(raw, i)?;
    match v.parse() {
        Ok(x) => Some(x),
        Err(_) => {
            eprintln!("error: '{}' got an invalid value: {:?}", raw[flag_pos], v);
            None
        }
    }
}
