//! Fast co-occurrence statistics for topic-coherence scoring.
//!
//! Coherence measures (UMass, UCI, NPMI, C_v) all reduce to two quantities over
//! a reference corpus: how many "windows" each top word appears in, and how many
//! windows each *pair* of top words co-occurs in. The naive approach slides a
//! window one token at a time and, at every position, enumerates all present
//! relevant words and bumps every present pair — `O(n_windows · present²)` work,
//! which is brutal for long documents, wide windows (C_v uses 110), and large K
//! (every topic's top-N words become "relevant").
//!
//! Two observations make this cheap:
//!
//! 1. **Only the pairs that occur within some topic's top-N are ever scored** —
//!    roughly `K · C(topn, 2)` pairs, not the full `R²` matrix.
//! 2. **A word's window membership is an interval union.** A token at position
//!    `p` belongs to every window whose start lies in `[p-w+1, p]`. So for one
//!    word, the set of windows it appears in is the union of those per-position
//!    start-intervals; for a pair, the co-occurring windows are the intersection
//!    of the two unions. Both are computed from the words' sorted position lists
//!    in linear time — no per-window enumeration at all.
//!
//! `window == 0` collapses each document to a single whole-document window,
//! which yields exactly the document-level co-occurrence UMass wants.

use std::collections::HashMap;

/// Marker for a token that is not one of the relevant (top-N) words. Such tokens
/// still occupy a position (so window offsets stay correct) but contribute no
/// word or pair counts.
pub const SENTINEL: u32 = u32::MAX;

/// Merge the window-start intervals for one word's ascending `positions`. A
/// position `p` covers starts `[p-w+1, p]`, clamped to the valid range
/// `[0, s_max]`. Positions arrive sorted, so the merge is a single pass.
fn merged_starts(positions: &[u32], w: u32, s_max: u32) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(positions.len());
    for &p in positions {
        let lo = p.saturating_sub(w - 1);
        let hi = p.min(s_max);
        if lo > hi {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if lo <= last.1 + 1 {
                // Overlapping or adjacent: extend the current run.
                if hi > last.1 {
                    last.1 = hi;
                }
                continue;
            }
        }
        out.push((lo, hi));
    }
    out
}

fn interval_len(merged: &[(u32, u32)]) -> u64 {
    merged.iter().map(|&(lo, hi)| (hi - lo + 1) as u64).sum()
}

/// Total length of the intersection of two merged (sorted, disjoint) interval
/// lists — i.e. the number of windows containing both words.
fn intersect_len(a: &[(u32, u32)], b: &[(u32, u32)]) -> u64 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut total = 0u64;
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo <= hi {
            total += (hi - lo + 1) as u64;
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

/// Compute single-word window counts (`occ`, length `num_relevant`) and pairwise
/// co-occurrence counts (`co`, parallel to `pairs`), plus the total window count.
///
/// `docs` holds relevant-word ids per token, with [`SENTINEL`] for non-relevant
/// tokens. `pairs` are `(a, b)` with `a < b`; each appears once. `window == 0`
/// means a single whole-document window (document-level co-occurrence).
pub fn cooccurrence(
    docs: &[Vec<u32>],
    num_relevant: usize,
    pairs: &[(u32, u32)],
    window: u32,
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut occ = vec![0.0f64; num_relevant];
    let mut co = vec![0.0f64; pairs.len()];
    let mut n_windows = 0.0f64;

    // Adjacency keyed on the smaller endpoint, so each pair is visited once.
    let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_relevant];
    for (pidx, &(a, b)) in pairs.iter().enumerate() {
        adj[a as usize].push((b, pidx as u32));
    }

    let mut posmap: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut merged: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();

    for doc in docs {
        let l = doc.len();
        if l == 0 {
            continue;
        }
        let w = if window > 0 { window as usize } else { l };
        let num_win = if l <= w { 1 } else { l - w + 1 };
        n_windows += num_win as f64;

        // Group the relevant tokens by word; positions stay ascending.
        posmap.clear();
        for (i, &id) in doc.iter().enumerate() {
            if id != SENTINEL {
                posmap.entry(id).or_default().push(i as u32);
            }
        }
        if posmap.is_empty() {
            continue;
        }

        if l <= w {
            // One whole-document window: boolean presence per word / pair.
            for &a in posmap.keys() {
                occ[a as usize] += 1.0;
                for &(b, pidx) in &adj[a as usize] {
                    if posmap.contains_key(&b) {
                        co[pidx as usize] += 1.0;
                    }
                }
            }
        } else {
            let s_max = (l - w) as u32; // valid starts: [0, s_max], count l - w + 1
            merged.clear();
            for (&a, positions) in posmap.iter() {
                let m = merged_starts(positions, w as u32, s_max);
                occ[a as usize] += interval_len(&m) as f64;
                merged.insert(a, m);
            }
            for &a in posmap.keys() {
                let ma = &merged[&a];
                for &(b, pidx) in &adj[a as usize] {
                    if let Some(mb) = merged.get(&b) {
                        co[pidx as usize] += intersect_len(ma, mb) as f64;
                    }
                }
            }
        }
    }
    (occ, co, n_windows)
}
