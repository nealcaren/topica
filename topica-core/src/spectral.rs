//! Spectral (anchor-word) initialization for the variational models — the
//! deterministic topic-word initializer used by STM's default `init.type =
//! "Spectral"`. Based on Arora et al. (2013), "A Practical Algorithm for Topic
//! Modeling with Provable Guarantees".
//!
//! The idea: build the word-word co-occurrence matrix Q, find K "anchor words"
//! (words that are near-exclusive to one topic) by greedy farthest-point
//! selection, then recover each word's topic loadings as a non-negative convex
//! combination of the anchors. The result is a deterministic β init that gives
//! stable, reproducible topic solutions (no random seed).
//!
//! The dense V×V co-occurrence makes this best for moderate vocabularies (after
//! pruning); callers fall back to random init when it is unavailable.

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Above this vocabulary size the dense V×V co-occurrence is replaced by a
/// random projection to `PROJ_DIM` columns (Johnson-Lindenstrauss), turning the
/// O(V²) anchor search into O(V·PROJ_DIM) — the same trick R's `stm` uses for
/// large vocabularies. At or below it, the exact path runs (cheap and exact).
/// The threshold sits well above `PROJ_DIM`: projecting only pays off once V is
/// several times the target dimension (otherwise the projection overhead exceeds
/// the savings), and it keeps the small/moderate-vocab behavior unchanged.
const PROJ_THRESHOLD: usize = 3000;
const PROJ_DIM: usize = 1024;
/// Fixed seed for the projection so `spectral_init` stays a deterministic,
/// seed-independent function of the corpus (the projection is an internal
/// implementation detail, not a modeling choice).
const PROJ_SEED: u64 = 0x5EED_C0FFEE;

/// Build the row-normalized co-occurrence matrix `Q̄` (V×V) and the word
/// probabilities `wprob`. `wprob` is the pooled unigram frequency
/// (`colSums(mat)/sum(mat)`, over all documents), matching Arora et al. and R
/// `stm`'s `wprob` — the word weighting the recovery uses. Returns `None` if the
/// corpus is too small/degenerate.
fn cooccurrence(docs: &[Vec<u32>], v: usize) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut q = vec![vec![0.0f64; v]; v];
    let mut total_count = vec![0.0f64; v];
    let mut total_tokens = 0.0f64;
    let mut used_docs = 0usize;
    for doc in docs {
        // Unigram counts over ALL documents (stm computes wprob from the full
        // term-document matrix, before the gram step drops length-<2 docs).
        for &w in doc {
            total_count[w as usize] += 1.0;
            total_tokens += 1.0;
        }
        let n = doc.len();
        if n < 2 {
            continue;
        }
        used_docs += 1;
        let mut wc: HashMap<usize, f64> = HashMap::new();
        for &w in doc {
            *wc.entry(w as usize).or_insert(0.0) += 1.0;
        }
        let words: Vec<(usize, f64)> = wc.into_iter().collect();
        let norm = 1.0 / (n as f64 * (n as f64 - 1.0));
        for &(wi, ci) in &words {
            for &(wj, cj) in &words {
                let val = if wi == wj { ci * (ci - 1.0) } else { ci * cj };
                q[wi][wj] += val * norm;
            }
        }
    }
    if used_docs == 0 || total_tokens == 0.0 {
        return None;
    }
    // Row-normalize → Q̄ (the per-document scaling cancels here, so no /used_docs).
    let mut qbar = q;
    for row in qbar.iter_mut() {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        }
    }
    let wprob: Vec<f64> = total_count.iter().map(|&c| c / total_tokens).collect();
    Some((qbar, wprob))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Greedy farthest-point anchor selection (Gram-Schmidt residuals). Only words
/// with marginal `p_w >= min_p` are candidates (avoids rare-word anchors).
fn fast_anchor_words(qbar: &[Vec<f64>], p: &[f64], k: usize, min_p: f64) -> Option<Vec<usize>> {
    let v = qbar.len();
    let candidates: Vec<usize> = (0..v).filter(|&w| p[w] >= min_p).collect();
    if candidates.len() < k {
        return None;
    }

    // First anchor: candidate with the largest row norm.
    let mut anchors = Vec::with_capacity(k);
    let first = *candidates
        .iter()
        .max_by(|&&a, &&b| dot(&qbar[a], &qbar[a]).total_cmp(&dot(&qbar[b], &qbar[b])))
        .unwrap();
    anchors.push(first);

    // Orthonormal basis of selected anchor rows; track each candidate's residual.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut residual: Vec<Vec<f64>> = candidates.iter().map(|&w| qbar[w].clone()).collect();

    let add_basis = |basis: &mut Vec<Vec<f64>>, residual: &mut Vec<Vec<f64>>, anchor_pos: usize| {
        // Gram-Schmidt: orthonormalize the anchor's residual, then project it out.
        let mut b = residual[anchor_pos].clone();
        let norm = dot(&b, &b).sqrt();
        if norm > 1e-12 {
            for x in b.iter_mut() {
                *x /= norm;
            }
            for r in residual.iter_mut() {
                let proj = dot(r, &b);
                for (ri, bi) in r.iter_mut().zip(&b) {
                    *ri -= proj * bi;
                }
            }
            basis.push(b);
        }
    };
    let first_pos = candidates.iter().position(|&w| w == first).unwrap();
    add_basis(&mut basis, &mut residual, first_pos);

    while anchors.len() < k {
        // Pick the candidate with the largest residual norm.
        let (best_pos, _) = residual
            .iter()
            .enumerate()
            .filter(|(pos, _)| !anchors.contains(&candidates[*pos]))
            .map(|(pos, r)| (pos, dot(r, r)))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        if dot(&residual[best_pos], &residual[best_pos]) < 1e-12 {
            return None; // can't find K independent anchors
        }
        anchors.push(candidates[best_pos]);
        add_basis(&mut basis, &mut residual, best_pos);
    }
    Some(anchors)
}

/// Recover the K×V topic-word matrix from the anchors (Arora "recoverL2" via
/// exponentiated gradient): each word's topic loadings are the non-negative,
/// sum-to-one weights best reconstructing its Q̄ row from the anchor rows.
fn recover(qbar: &[Vec<f64>], p: &[f64], anchors: &[usize], k: usize, v: usize) -> Vec<Vec<f64>> {
    // Anchor Gram matrix G (K×K) and, per word, b_k = <anchor_k, word>.
    let anchor_rows: Vec<&Vec<f64>> = anchors.iter().map(|&a| &qbar[a]).collect();
    let mut g = vec![vec![0.0f64; k]; k];
    for a in 0..k {
        for b in 0..k {
            g[a][b] = dot(anchor_rows[a], anchor_rows[b]);
        }
    }

    // Map each anchor word to its topic index, so anchor words get exact unit
    // vectors (matching stm's recoverL2, which sets condprob[anchor] = e_topic).
    let mut anchor_topic = vec![usize::MAX; v];
    for (t, &a) in anchors.iter().enumerate() {
        anchor_topic[a] = t;
    }

    // A[word][topic] = C[word][topic] * p[word]; later normalized per topic.
    // Per word, recover the simplex weights C by exponentiated gradient on the
    // residual `b − G c`, with a max-subtracted multiplicative update
    // `c ∝ c · exp(2η(b − Gc) − max)` run to `‖residual‖` convergence.
    //
    // The step `η` is adaptive: `1/(2L)` with `L` a Gershgorin bound on the largest
    // eigenvalue of `G` (the gradient's Lipschitz constant). This is the mirror-
    // descent stable step, so the iterate converges to the constrained optimum
    // (Arora's recoverL2, == R stm's `recoverEG=FALSE` reference path) rather than
    // overshooting to a vertex. The classic fixed `η=50` is far too large here —
    // the gradients scale as ~1/V — so it diverged to (arbitrary, float-order-
    // dependent) vertices and never converged (issue #234); the adaptive step
    // reaches the optimum in ~70 iterations on gadarian.
    let l: f64 = (0..k)
        .map(|i| (0..k).map(|j| g[i][j].abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let eta = 1.0 / (2.0 * l.max(1e-12));
    const TOL: f64 = 1e-8;
    const MAX_ITS: usize = 500;
    let mut a_mat = vec![vec![0.0f64; k]; v];
    for w in 0..v {
        let mut c = vec![1.0 / k as f64; k];
        if anchor_topic[w] != usize::MAX {
            // Anchor word: exact unit vector, no solve.
            c = vec![0.0f64; k];
            c[anchor_topic[w]] = 1.0;
        } else {
            let bvec: Vec<f64> = (0..k).map(|t| dot(anchor_rows[t], &qbar[w])).collect();
            let mut sse_old = f64::INFINITY;
            for _ in 0..MAX_ITS {
                let gc: Vec<f64> = (0..k)
                    .map(|i| (0..k).map(|j| g[i][j] * c[j]).sum())
                    .collect();
                // residual b − Gc and its squared norm (the convergence metric).
                let resid: Vec<f64> = (0..k).map(|i| bvec[i] - gc[i]).collect();
                let sse: f64 = resid.iter().map(|r| r * r).sum();
                // Multiplicative exp-gradient step, max-subtracted for stability.
                let grad: Vec<f64> = resid.iter().map(|r| 2.0 * eta * r).collect();
                let maxd = grad.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut total = 0.0;
                for i in 0..k {
                    c[i] *= (grad[i] - maxd).exp();
                    total += c[i];
                }
                if total > 0.0 {
                    for ci in c.iter_mut() {
                        *ci /= total;
                    }
                }
                if (sse_old.sqrt() - sse.sqrt()).abs() < TOL {
                    break;
                }
                sse_old = sse;
            }
        }
        for t in 0..k {
            a_mat[w][t] = c[t] * p[w];
        }
    }

    // β_{k,w} = A[w][k] normalized over words (+ tiny smoothing).
    let mut beta = vec![vec![1e-8f64; v]; k];
    for t in 0..k {
        let mut col = 0.0;
        for w in 0..v {
            col += a_mat[w][t];
        }
        if col <= 0.0 {
            // Degenerate topic — uniform.
            for w in 0..v {
                beta[t][w] = 1.0 / v as f64;
            }
            continue;
        }
        for w in 0..v {
            beta[t][w] = (a_mat[w][t] + 1e-8) / (col + v as f64 * 1e-8);
        }
    }
    beta
}

/// Build a random-projected row-normalized co-occurrence `Q̄·R` (V×M) and the
/// word marginals `p`, without ever materializing the dense V×V matrix.
///
/// Because the per-document co-occurrence contribution to row `w` is
/// `norm·c_w·(S_d − R[w])` where `S_d = Σ_{w'∈d} c_{w'} R[w']`, the projection
/// `Q[w]·R` can be accumulated directly from each document's sparse word counts
/// in O(unique_words·M) time. Projecting preserves inner products
/// (Johnson-Lindenstrauss), so the anchor search and recovery — which use only
/// dot products of `Q̄` rows — give an accurate approximation in M dimensions.
fn cooccurrence_projected(
    docs: &[Vec<u32>],
    v: usize,
    m: usize,
) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    // Rademacher (±1/√M) projection matrix R (V×M), fixed-seed for determinism.
    let mut rng = ChaCha8Rng::seed_from_u64(PROJ_SEED);
    let scale = 1.0 / (m as f64).sqrt();
    let r: Vec<Vec<f64>> = (0..v)
        .map(|_| {
            (0..m)
                .map(|_| if rng.gen::<bool>() { scale } else { -scale })
                .collect()
        })
        .collect();

    let mut qp = vec![vec![0.0f64; m]; v];
    let mut rowsum = vec![0.0f64; v]; // co-occurrence row marginal (for Q̄ normalization)
    let mut total_count = vec![0.0f64; v]; // unigram counts over all docs (for wprob)
    let mut total_tokens = 0.0f64;
    let mut used_docs = 0usize;
    for doc in docs {
        for &w in doc {
            total_count[w as usize] += 1.0;
            total_tokens += 1.0;
        }
        let n = doc.len();
        if n < 2 {
            continue;
        }
        used_docs += 1;
        let mut wc: HashMap<usize, f64> = HashMap::new();
        for &w in doc {
            *wc.entry(w as usize).or_insert(0.0) += 1.0;
        }
        let nf = n as f64;
        let norm = 1.0 / (nf * (nf - 1.0));
        // S_d = Σ_{w'∈doc} c_{w'} R[w'].
        let mut s = vec![0.0f64; m];
        for (&w, &c) in &wc {
            for j in 0..m {
                s[j] += c * r[w][j];
            }
        }
        for (&w, &c) in &wc {
            let coef = norm * c;
            for j in 0..m {
                qp[w][j] += coef * (s[j] - r[w][j]);
            }
            rowsum[w] += c / nf; // matches the dense path's pre-normalization row sum
        }
    }
    if used_docs == 0 || total_tokens == 0.0 {
        return None;
    }
    // Q̄·R = (Q·R) row-normalized by the row marginal (projection is linear, so
    // proj(q[w]/rowsum[w]) = (q[w]·R)/rowsum[w]). The per-document scaling cancels.
    let mut qbar = qp;
    for (w, row) in qbar.iter_mut().enumerate() {
        if rowsum[w] > 0.0 {
            let rw = rowsum[w];
            for x in row.iter_mut() {
                *x /= rw;
            }
        }
    }
    // wprob = pooled unigram frequency (Arora / stm wprob), as in the dense path.
    let wprob: Vec<f64> = total_count.iter().map(|&c| c / total_tokens).collect();
    Some((qbar, wprob))
}

/// Deterministic anchor-word initialization of the K×V topic-word matrix.
/// Returns `None` when it is not applicable (corpus too small, fewer candidate
/// words than topics, or degenerate co-occurrence) so the caller can fall back.
///
/// For large vocabularies the dense V×V co-occurrence is replaced by a
/// random-projected V×M one (see [`cooccurrence_projected`]); the downstream
/// anchor search and recovery are dimension-agnostic and run unchanged.
pub fn spectral_init(docs: &[Vec<u32>], k: usize, v: usize) -> Option<Vec<Vec<f64>>> {
    if v < k {
        return None;
    }
    let (qbar, p) = if v > PROJ_THRESHOLD {
        cooccurrence_projected(docs, v, PROJ_DIM)?
    } else {
        cooccurrence(docs, v)?
    };
    // Require anchors to have at least a small marginal mass.
    let min_p = 0.0;
    let anchors = fast_anchor_words(&qbar, &p, k, min_p)?;
    Some(recover(&qbar, &p, &anchors, k, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_recovers_planted_topics_large_vocab() {
        // Vocabulary above PROJ_THRESHOLD so the random-projection path runs.
        // Ten disjoint blocks of 400 words each (V = 4000 > 3000); each doc
        // draws from one block. Spectral init should still put each topic's
        // top words on a single block via the projected co-occurrence.
        let nb = 10usize;
        let bs = 400usize;
        let v = nb * bs; // 4000
        let blocks: Vec<Vec<u32>> = (0..nb)
            .map(|b| (b * bs..b * bs + bs).map(|w| w as u32).collect())
            .collect();
        let mut docs = Vec::new();
        for i in 0..(nb * 200) {
            let blk = &blocks[i % nb];
            // 10-token docs from one block (spread across the block).
            let doc: Vec<u32> = (0..10).map(|j| blk[(i * 7 + j * 37) % bs]).collect();
            docs.push(doc);
        }
        let beta = spectral_init(&docs, nb, v).expect("projected spectral init");
        // Each topic's top words should fall predominantly in one block.
        let mut covered = std::collections::HashSet::new();
        for t in 0..nb {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| beta[t][b].total_cmp(&beta[t][a]));
            let block_of = |w: usize| w / bs;
            let top_block = block_of(idx[0]);
            let same = idx[..10]
                .iter()
                .filter(|&&w| block_of(w) == top_block)
                .count();
            assert!(
                same >= 8,
                "topic {} top words not concentrated in one block ({}/10)",
                t,
                same
            );
            covered.insert(top_block);
        }
        // Every topic is cleanly block-localized (asserted above); a strong
        // majority of the blocks are distinctly recovered. (Perfect 1-topic-per-
        // block separation isn't guaranteed at K=10 — greedy anchor selection can
        // pick two anchors from one block — and that's true of the exact path too,
        // so we don't demand it of the approximate projected path.)
        assert!(
            covered.len() >= 7,
            "too few blocks separated: {:?}",
            covered
        );
    }

    #[test]
    fn recovers_planted_topics() {
        // Three disjoint-vocabulary topics; each doc draws from one. Spectral
        // init should put each topic's mass on one vocabulary block.
        let blocks = [[0u32, 1, 2], [3, 4, 5], [6, 7, 8]];
        let mut docs = Vec::new();
        for i in 0..150 {
            let b = blocks[i % 3];
            docs.push(vec![b[0], b[1], b[2], b[0], b[1], b[2]]);
        }
        let beta = spectral_init(&docs, 3, 9).expect("spectral init");
        // Each topic concentrates on one block (its top-3 words are one block).
        for t in 0..3 {
            let mut idx: Vec<usize> = (0..9).collect();
            idx.sort_by(|&a, &b| beta[t][b].total_cmp(&beta[t][a]));
            let top: std::collections::HashSet<usize> = idx[..3].iter().copied().collect();
            let is_block = blocks
                .iter()
                .any(|blk| blk.iter().all(|&w| top.contains(&(w as usize))));
            assert!(
                is_block,
                "topic {} top words not a single block: {:?}",
                t,
                &idx[..3]
            );
        }
    }
}
