/// Convert a token sequence to (unique word ids, counts). A BTreeMap keeps the
/// word order deterministic (sorted), so float summation order — and thus the
/// fitted model — is fully reproducible for a given seed.
pub fn doc_sparse(doc: &[u32]) -> (Vec<usize>, Vec<f64>) {
    use std::collections::BTreeMap;
    let mut m: BTreeMap<usize, f64> = BTreeMap::new();
    for &w in doc {
        *m.entry(w as usize).or_insert(0.0) += 1.0;
    }
    let words: Vec<usize> = m.keys().copied().collect();
    let counts: Vec<f64> = words.iter().map(|w| m[w]).collect();
    (words, counts)
}
