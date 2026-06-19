//! The parallel Laplace variational E-step driver shared by the logistic-normal
//! models (CTM/STM, and STS in a later step). It owns only the parallel iteration
//! and ordering: per-document inference is independent, so it runs under rayon,
//! but results are collected in DOCUMENT ORDER and returned that way, so the
//! caller's serial sufficient-statistic reduce sums in the exact same order as a
//! single-threaded loop would — the fit stays bit-for-bit deterministic
//! regardless of thread count. The model-specific objective/gradient/Hessian work
//! lives entirely in the `per_doc` closure.

use rayon::prelude::*;

/// Run `per_doc` over every non-empty document in `sparse` (a slice of
/// `(word_ids, counts)` pairs) in parallel, returning `(doc_index, payload)`
/// pairs in ascending document order. Empty documents (no words) are skipped and
/// produce no entry.
pub fn laplace_estep<O, F>(sparse: &[(Vec<usize>, Vec<f64>)], per_doc: F) -> Vec<(usize, O)>
where
    O: Send,
    F: Fn(usize, &[usize], &[f64]) -> O + Sync,
{
    sparse
        .par_iter()
        .enumerate()
        .filter(|(_, (words, _))| !words.is_empty())
        .map(|(di, (words, counts))| (di, per_doc(di, words, counts)))
        .collect()
}
