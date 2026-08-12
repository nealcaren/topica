//! A faithful, dependency-free UMAP reducer (McInnes, Healy & Melville 2018).
//!
//! topica's embedding-clustering heads (BERTopic, Top2Vec) reduce document
//! embeddings before a density clusterer. PCA (the default) is fast and
//! deterministic; UMAP separates closely-spaced themes a linear projection
//! merges. We used to delegate the UMAP path to the `umap-rs` crate, but its
//! layout gradient diverges from the reference `umap-learn` (an extra
//! `dist_squared` factor in the attractive/repulsive denominators), so it
//! recovered noticeably worse cluster structure. This module reimplements UMAP
//! to match `umap-learn` numerically, staying inside topica's constraints: pure
//! Rust, BLAS-free, `Vec<Vec<f64>>`, and — because the negative sampling is
//! seeded with `ChaCha8Rng` — a **fully reproducible** fit, unlike the crate.
//!
//! The pipeline mirrors `umap-learn`:
//!   1. kNN graph (brute force) under a cosine or Euclidean metric.
//!   2. `smooth_knn_dist`: per-point bandwidth `sigma` and local-connectivity
//!      offset `rho` calibrating a fuzzy membership of cardinality `log2(k)`.
//!   3. Fuzzy simplicial set: membership strengths, symmetrized by the fuzzy
//!      union `P = A + Aᵀ − A ⊙ Aᵀ` (for `set_op_mix_ratio = 1`).
//!   4. Spectral (Laplacian-eigenmap) initialization of the layout — `umap-learn`'s
//!      default `init="spectral"`. Solved by seeded subspace iteration on the
//!      normalized adjacency (scalable where a dense eigensolve is not), with a
//!      seeded-random fallback. Seeding the SGD from the graph's eigenmap starts
//!      clusters in separated basins; the old uniform-random init gave clustered
//!      real embeddings no structure to open (a residual `#555` contributor).
//!   5. SGD layout: sample edges in proportion to their weight, pull endpoints
//!      together, push uniformly-drawn negatives apart, with the `umap-learn`
//!      gradient and the ±4 clip.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Fit the `(a, b)` of the low-dimensional membership curve
/// `1 / (1 + a·x^(2b))` to an offset exponential decay defined by `spread` and
/// `min_dist`. `umap-learn` uses SciPy `curve_fit` (Levenberg-Marquardt); we match
/// it with a damped Gauss-Newton solve on the same 2-parameter least-squares problem.
///
/// The previous plain-gradient-descent fit under-converged from its `a = 1.5` start:
/// at `min_dist = 0.0` (BERTopic's default) the true `a ≈ 1.933`, but GD stalled near
/// `1.577`, giving a membership curve with too-weak short-range attraction. The layout
/// then failed to open the density valleys HDBSCAN needs, collapsing to ~2 topics on
/// real embeddings (issue #555). Gauss-Newton reaches the true optimum (`curve_fit`
/// parity to 4 decimals across `min_dist`), restoring the correct attraction.
pub fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    let n_points = 300usize;
    let xv: Vec<f64> = (0..n_points)
        .map(|i| (spread * 3.0) * (i as f64) / (n_points as f64 - 1.0))
        .collect();
    let yv: Vec<f64> = xv
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    // curve_fit's default initial guess is (1, 1).
    let mut a = 1.0;
    let mut b = 1.0;
    let lambda = 1e-3; // Levenberg-Marquardt diagonal damping.
    for _ in 0..100 {
        // Accumulate the Gauss-Newton normal equations JᵀJ (2×2, symmetric) and Jᵀr.
        let (mut jtj00, mut jtj01, mut jtj11) = (0.0, 0.0, 0.0);
        let (mut jtr0, mut jtr1) = (0.0, 0.0);
        for i in 0..n_points {
            let x = xv[i];
            if x <= 0.0 {
                continue; // x = 0 contributes a zero row (residual 0, zero Jacobian).
            }
            let x_2b = x.powf(2.0 * b);
            let denom = 1.0 + a * x_2b;
            let r = 1.0 / denom - yv[i];
            let d_da = -x_2b / (denom * denom);
            let d_db = -2.0 * a * x_2b * x.ln() / (denom * denom);
            jtj00 += d_da * d_da;
            jtj01 += d_da * d_db;
            jtj11 += d_db * d_db;
            jtr0 += d_da * r;
            jtr1 += d_db * r;
        }
        // Solve (JᵀJ + λ·diag(JᵀJ)) Δ = −Jᵀr for the step Δ.
        let a00 = jtj00 * (1.0 + lambda);
        let a11 = jtj11 * (1.0 + lambda);
        let a01 = jtj01;
        let det = a00 * a11 - a01 * a01;
        if det.abs() < 1e-30 {
            break;
        }
        let step_a = -(a11 * jtr0 - a01 * jtr1) / det;
        let step_b = -(-a01 * jtr0 + a00 * jtr1) / det;
        a = (a + step_a).max(1e-3);
        b = (b + step_b).max(1e-3);
        if step_a * step_a + step_b * step_b < 1e-14 {
            break;
        }
    }
    (a, b)
}

/// Brute-force k-nearest-neighbor graph. Returns, per point, the indices and
/// distances of its `k` neighbors **including itself** at column 0 (distance 0),
/// matching `umap-learn`'s convention. `metric` is `"cosine"` (default; rows are
/// L2-normalized and distance is `1 − cos`) or `"euclidean"`.
fn knn_graph(data: &[Vec<f64>], k: usize, metric: &str) -> (Vec<Vec<u32>>, Vec<Vec<f64>>) {
    let n = data.len();
    let cosine = metric != "euclidean";
    // For cosine, precompute unit rows so the neighbor distance is 1 − dot.
    let unit: Vec<Vec<f64>> = if cosine {
        data.iter()
            .map(|row| {
                let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
                if norm > 0.0 {
                    row.iter().map(|&v| v / norm).collect()
                } else {
                    row.clone()
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    let rows: Vec<(Vec<u32>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut d: Vec<(f64, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = if cosine {
                        let dot: f64 = unit[i].iter().zip(&unit[j]).map(|(&x, &y)| x * y).sum();
                        (1.0 - dot).max(0.0)
                    } else {
                        data[i]
                            .iter()
                            .zip(&data[j])
                            .map(|(&x, &y)| (x - y) * (x - y))
                            .sum::<f64>()
                            .sqrt()
                    };
                    (dist, j as u32)
                })
                .collect();
            // k-1 real neighbors; column 0 is self. Full sort is negligible next
            // to the O(n·dim) distance computation above.
            d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = k.saturating_sub(1).min(d.len());
            d.truncate(take);
            let mut idx = Vec::with_capacity(k);
            let mut dst = Vec::with_capacity(k);
            idx.push(i as u32);
            dst.push(0.0);
            for (dd, jj) in d {
                idx.push(jj);
                dst.push(dd);
            }
            (idx, dst)
        })
        .collect();

    let mut indices = Vec::with_capacity(n);
    let mut dists = Vec::with_capacity(n);
    for (idx, dst) in rows {
        indices.push(idx);
        dists.push(dst);
    }
    (indices, dists)
}

const SMOOTH_K_TOLERANCE: f64 = 1e-5;
const MIN_K_DIST_SCALE: f64 = 1e-3;

/// Per-point bandwidth `sigma` and local-connectivity offset `rho`. For each row
/// we binary-search `sigma` so that `Σ_j exp(-(d_j − rho)/sigma)` over the real
/// neighbors equals `log2(k)`; `rho` is the distance to the
/// `local_connectivity`-th nearest non-zero neighbor. Faithful to `umap-learn`.
fn smooth_knn_dist(
    knn_dists: &[Vec<f64>],
    k: usize,
    local_connectivity: f64,
    bandwidth: f64,
) -> (Vec<f64>, Vec<f64>) {
    let target = (k as f64).log2() * bandwidth;
    let n_all: f64 = knn_dists.iter().flat_map(|r| r.iter()).copied().sum();
    let count: usize = knn_dists.iter().map(|r| r.len()).sum();
    let mean_distances = if count > 0 { n_all / count as f64 } else { 0.0 };

    knn_dists
        .par_iter()
        .map(|ith| {
            let non_zero: Vec<f64> = ith.iter().copied().filter(|&d| d > 0.0).collect();
            let mut rho = 0.0;
            if non_zero.len() >= local_connectivity.floor() as usize && !non_zero.is_empty() {
                let index = local_connectivity.floor() as usize;
                let interpolation = local_connectivity - local_connectivity.floor();
                if index > 0 {
                    rho = non_zero[index - 1];
                    if interpolation > SMOOTH_K_TOLERANCE {
                        rho += interpolation * (non_zero[index] - non_zero[index - 1]);
                    }
                } else {
                    rho = interpolation * non_zero[0];
                }
            } else if !non_zero.is_empty() {
                rho = non_zero.iter().copied().fold(0.0, f64::max);
            }

            // Binary search for sigma.
            let mut lo = 0.0f64;
            let mut hi = f64::INFINITY;
            let mut mid = 1.0f64;
            for _ in 0..64 {
                let mut psum = 0.0;
                for &d in ith.iter().skip(1) {
                    let dd = d - rho;
                    psum += if dd > 0.0 { (-(dd) / mid).exp() } else { 1.0 };
                }
                if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                    break;
                }
                if psum > target {
                    hi = mid;
                    mid = (lo + hi) / 2.0;
                } else {
                    lo = mid;
                    if hi == f64::INFINITY {
                        mid *= 2.0;
                    } else {
                        mid = (lo + hi) / 2.0;
                    }
                }
            }
            let mut sigma = mid;
            // Minimum-distance scale floor.
            if rho > 0.0 {
                let mean_ith: f64 = ith.iter().sum::<f64>() / ith.len() as f64;
                if sigma < MIN_K_DIST_SCALE * mean_ith {
                    sigma = MIN_K_DIST_SCALE * mean_ith;
                }
            } else if sigma < MIN_K_DIST_SCALE * mean_distances {
                sigma = MIN_K_DIST_SCALE * mean_distances;
            }
            (sigma, rho)
        })
        .unzip()
}

/// Membership strength of the directed edge `i → j`: `1` inside the local
/// connectivity radius, else `exp(-(d − rho_i)/sigma_i)`.
fn membership(d: f64, rho: f64, sigma: f64) -> f64 {
    if d - rho <= 0.0 || sigma == 0.0 {
        1.0
    } else {
        (-(d - rho) / sigma).exp()
    }
}

/// Build the symmetric fuzzy simplicial set as a weighted edge list
/// `(head, tail, weight)`. Directed memberships are combined by the probabilistic
/// t-conorm `set_op * (A + Aᵀ) + (1 − 2·set_op)·(A ⊙ Aᵀ)`; for the default
/// `set_op_mix_ratio = 1` this is `A + Aᵀ − A ⊙ Aᵀ`.
fn fuzzy_simplicial_set(
    knn_indices: &[Vec<u32>],
    knn_dists: &[Vec<f64>],
    sigmas: &[f64],
    rhos: &[f64],
    set_op_mix_ratio: f64,
) -> Vec<(u32, u32, f64)> {
    use std::collections::{HashMap, HashSet};
    let n = knn_indices.len();
    // Directed strengths keyed by (i, j), i != j. The map is used only for O(1)
    // lookup of the reverse edge; iteration order below is deterministic.
    let mut a: HashMap<(u32, u32), f64> = HashMap::new();
    for i in 0..n {
        for (col, &j) in knn_indices[i].iter().enumerate() {
            if j as usize == i {
                continue;
            }
            let v = membership(knn_dists[i][col], rhos[i], sigmas[i]);
            if v != 0.0 {
                a.insert((i as u32, j), v);
            }
        }
    }
    // Symmetrize by iterating points and their neighbors in a fixed order (not
    // the HashMap's), so the emitted edge list — and thus the SGD update order —
    // is reproducible.
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut edges = Vec::with_capacity(a.len() * 2);
    for i in 0..n {
        let iu = i as u32;
        for &j in knn_indices[i].iter() {
            if j as usize == i {
                continue;
            }
            let key = if iu < j { (iu, j) } else { (j, iu) };
            if !seen.insert(key) {
                continue;
            }
            let val = a.get(&(iu, j)).copied().unwrap_or(0.0);
            let val_t = a.get(&(j, iu)).copied().unwrap_or(0.0);
            let combined =
                set_op_mix_ratio * (val + val_t) + (1.0 - 2.0 * set_op_mix_ratio) * (val * val_t);
            if combined > 0.0 {
                // Both directions carry the symmetric weight.
                edges.push((iu, j, combined));
                edges.push((j, iu, combined));
            }
        }
    }
    edges
}

/// `epochs_per_sample[e] = max_weight / weight[e]`; an edge of maximal weight is
/// sampled every epoch, lighter edges proportionally less often.
fn make_epochs_per_sample(weights: &[f64], n_epochs: usize) -> Vec<f64> {
    let max_w = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_w = if max_w <= 0.0 { 1.0 } else { max_w };
    weights
        .iter()
        .map(|&w| {
            let n_samples = n_epochs as f64 * (w / max_w);
            if n_samples > 0.0 {
                n_epochs as f64 / n_samples
            } else {
                -1.0
            }
        })
        .collect()
}

#[inline]
fn clip(v: f64) -> f64 {
    v.clamp(-4.0, 4.0)
}

/// Rescale a flat `n × dim` embedding so each axis independently spans `[0, 10]`
/// (umap-learn's `10 * (e − min) / (max − min)` applied per column before the SGD).
/// A degenerate axis (zero range) collapses to the neutral midpoint `5.0`, avoiding
/// the NaN the reference division would produce. Deterministic and in place.
fn rescale_unit_box(embedding: &mut [f64], n: usize, dim: usize) {
    for d in 0..dim {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..n {
            let v = embedding[i * dim + d];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let range = hi - lo;
        for i in 0..n {
            let idx = i * dim + d;
            embedding[idx] = if range > 1e-12 {
                10.0 * (embedding[idx] - lo) / range
            } else {
                5.0
            };
        }
    }
}

/// SGD layout optimization, faithful to `umap-learn`'s
/// `optimize_layout_euclidean` (sequential, seeded negative sampling).
#[allow(clippy::too_many_arguments)]
fn optimize_layout(
    embedding: &mut [f64],
    n: usize,
    dim: usize,
    head: &[u32],
    tail: &[u32],
    epochs_per_sample: &[f64],
    a: f64,
    b: f64,
    gamma: f64,
    n_epochs: usize,
    negative_sample_rate: usize,
    seed: u64,
) {
    // Per-vertex negative-sampling RNG state (splitmix64), mirroring umap-learn's
    // `rng_state_per_sample[j]`. Keying the stream to the head vertex `j` (not one
    // global stream advanced in edge-processing order) makes negative sampling
    // independent of edge order, which — together with the canonical edge sort in
    // `umap()` — is what stops the #555 layout collapse.
    let mut vstate: Vec<u64> = (0..n)
        .map(|j| {
            let s = seed ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        })
        .collect();
    let mut epoch_of_next_sample = epochs_per_sample.to_vec();
    let epochs_per_negative_sample: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate as f64)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    for epoch in 0..n_epochs {
        let alpha = 1.0 * (1.0 - epoch as f64 / n_epochs as f64);
        for e in 0..epochs_per_sample.len() {
            if epochs_per_sample[e] <= 0.0 || epoch_of_next_sample[e] > epoch as f64 {
                continue;
            }
            let j = head[e] as usize;
            let k = tail[e] as usize;
            let (jb, kb) = (j * dim, k * dim);

            let mut dist_sq = 0.0;
            for d in 0..dim {
                let diff = embedding[jb + d] - embedding[kb + d];
                dist_sq += diff * diff;
            }
            let grad_coeff = if dist_sq > 0.0 {
                let dpb = dist_sq.powf(b);
                (-2.0 * a * b * dpb / dist_sq) / (a * dpb + 1.0)
            } else {
                0.0
            };
            for d in 0..dim {
                let diff = embedding[jb + d] - embedding[kb + d];
                let grad_d = clip(grad_coeff * diff) * alpha;
                embedding[jb + d] += grad_d;
                embedding[kb + d] -= grad_d;
            }

            epoch_of_next_sample[e] += epochs_per_sample[e];

            let n_neg = ((epoch as f64 - epoch_of_next_negative_sample[e])
                / epochs_per_negative_sample[e]) as usize;
            for _ in 0..n_neg {
                // splitmix64 draw from vertex j's own stream (order-independent).
                vstate[j] = vstate[j].wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = vstate[j];
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let k = (z % n as u64) as usize;
                if k == j {
                    continue;
                }
                let kb = k * dim;
                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let diff = embedding[jb + d] - embedding[kb + d];
                    dist_sq += diff * diff;
                }
                let grad_coeff = if dist_sq > 0.0 {
                    let dpb = dist_sq.powf(b);
                    2.0 * gamma * b / ((0.001 + dist_sq) * (a * dpb + 1.0))
                } else {
                    0.0
                };
                if grad_coeff > 0.0 {
                    for d in 0..dim {
                        let diff = embedding[jb + d] - embedding[kb + d];
                        embedding[jb + d] += clip(grad_coeff * diff) * alpha;
                    }
                }
            }
            epoch_of_next_negative_sample[e] += n_neg as f64 * epochs_per_negative_sample[e];
        }
    }
}

/// Modified Gram-Schmidt orthonormalization of the `m` length-`n` columns stored
/// column-major in `basis` (`basis[c*n + i]`). A column that collapses to ~0 is
/// left as zeros; the caller's Rayleigh-Ritz step tolerates the rank drop.
fn orthonormalize_cols(basis: &mut [f64], n: usize, m: usize) {
    for c in 0..m {
        // Subtract projections onto the earlier, already-normalized columns.
        for p in 0..c {
            let mut dot = 0.0;
            for i in 0..n {
                dot += basis[p * n + i] * basis[c * n + i];
            }
            for i in 0..n {
                basis[c * n + i] -= dot * basis[p * n + i];
            }
        }
        let mut nrm = 0.0;
        for i in 0..n {
            nrm += basis[c * n + i] * basis[c * n + i];
        }
        let nrm = nrm.sqrt();
        if nrm > 1e-12 {
            for i in 0..n {
                basis[c * n + i] /= nrm;
            }
        }
    }
}

/// Spectral (Laplacian-eigenmap) initial layout, matching `umap-learn`'s default
/// `init="spectral"`. The layout is the top `n_components` non-trivial eigenvectors
/// of the normalized adjacency `P = D^{-1/2} A D^{-1/2}` of the fuzzy simplicial set
/// — equivalently the smallest non-zero eigenvectors of the normalized Laplacian
/// `I − P` — which places clusters in the separated density basins the SGD then
/// refines. topica previously seeded the SGD from a uniform random cloud; on real
/// (clustered) sentence embeddings that cloud has no cluster structure for the
/// short-range attraction to open, so the layout collapsed to ~2 topics (#555, the
/// residual cause after the `a,b` and edge-order fixes). Spectral init supplies that
/// structure.
///
/// Solved by seeded subspace (block power) iteration on `P` — O(n·k·m·iters), scalable
/// where a dense eigensolve is not — plus a Rayleigh-Ritz step (small `m×m` Jacobi) to
/// order the Ritz vectors. Returns `None` (⇒ caller falls back to random init) when the
/// graph has an isolated vertex, matching `umap-learn`'s random fallback on a failed
/// spectral solve. Deterministic for a fixed `seed`.
fn spectral_init(
    edges: &[(u32, u32, f64)],
    n: usize,
    n_components: usize,
    seed: u64,
) -> Option<Vec<f64>> {
    // Weighted degrees of the (already symmetrized) graph.
    let mut deg = vec![0.0f64; n];
    for &(h, _, w) in edges {
        deg[h as usize] += w;
    }
    // Isolated vertices (all their edges pruned) get D^{-1/2}=0 — a zero row/column in
    // P, so they sit at the origin and are jittered by the SGD noise. Only bail to random
    // init if the graph is *mostly* isolated (no usable structure to embed).
    let n_isolated = deg.iter().filter(|&&d| d <= 0.0).count();
    if n_isolated * 2 >= n {
        return None;
    }
    let inv_sqrt_d: Vec<f64> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // y = P x from the edge list: y[h] = D^{-1/2}[h] Σ_edges(h→t) w · D^{-1/2}[t] · x[t].
    let matvec = |x: &[f64], y: &mut [f64]| {
        for v in y.iter_mut() {
            *v = 0.0;
        }
        for &(h, t, w) in edges {
            let (h, t) = (h as usize, t as usize);
            y[h] += w * inv_sqrt_d[h] * inv_sqrt_d[t] * x[t];
        }
    };

    // Top m = n_components+1 eigenvectors (the +1 is the trivial ~constant vector,
    // eigenvalue 1, dropped below).
    let m = (n_components + 1).min(n);
    if m < 2 {
        return None;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5EED_15DE_ADBE_EF01);
    let mut basis = vec![0.0f64; n * m]; // column-major: basis[c*n + i]
    for v in basis.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }
    orthonormalize_cols(&mut basis, n, m);

    let mut tmp = vec![0.0f64; n];
    let mut ritz_vecs = vec![0.0f64; n * m];
    let mut prev_vals = vec![f64::INFINITY; m];
    for iter in 0..300 {
        // Power step: basis <- orthonormal( P · basis ).
        let mut yb = vec![0.0f64; n * m];
        for c in 0..m {
            matvec(&basis[c * n..(c + 1) * n], &mut tmp);
            yb[c * n..(c + 1) * n].copy_from_slice(&tmp);
        }
        orthonormalize_cols(&mut yb, n, m);
        basis.copy_from_slice(&yb);

        // Rayleigh-Ritz occasionally: project P into the basis, eigensolve the small
        // m×m, rotate to ordered Ritz vectors, and check the Ritz values for a plateau.
        if iter % 10 == 9 {
            let mut pb = vec![0.0f64; n * m];
            for c in 0..m {
                matvec(&basis[c * n..(c + 1) * n], &mut tmp);
                pb[c * n..(c + 1) * n].copy_from_slice(&tmp);
            }
            let mut mm = vec![0.0f64; m * m];
            for a in 0..m {
                for b in 0..m {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += basis[a * n + i] * pb[b * n + i];
                    }
                    mm[a * m + b] = s;
                }
            }
            let (vals, vecs) = crate::reduce::jacobi_eigen_symmetric(&mm, m);
            for (j, vecs_j) in vecs.iter().enumerate() {
                for i in 0..n {
                    let mut s = 0.0;
                    for a in 0..m {
                        s += basis[a * n + i] * vecs_j[a];
                    }
                    ritz_vecs[j * n + i] = s;
                }
            }
            let delta: f64 = vals
                .iter()
                .zip(&prev_vals)
                .map(|(a, b)| (a - b).abs())
                .sum();
            prev_vals.copy_from_slice(&vals);
            if delta < 1e-7 {
                break;
            }
        }
    }

    // Drop the trivial top eigenvector (index 0, eigenvalue ≈ 1); keep the next
    // n_components, ordered by descending eigenvalue (jacobi_eigen_symmetric sorts so).
    let mut out = vec![0.0f64; n * n_components];
    for comp in 0..n_components {
        let j = comp + 1;
        for i in 0..n {
            out[i * n_components + comp] = ritz_vecs[j * n + i];
        }
    }
    // Guard against a degenerate all-zero layout (e.g. the solve stalled): fall back.
    if out.iter().all(|&v| v.abs() < 1e-12) {
        return None;
    }
    Some(out)
}

/// Reduce `data` (`n × features`) to `n_components` with UMAP. `n_neighbors` is
/// the graph neighborhood; `min_dist`/`spread` shape the embedding; `n_epochs`
/// the SGD length (0 ⇒ 500 for ≤10k rows, else 200); `negative_sample_rate` and
/// `repulsion_strength` control the repulsive term; `metric` is `"cosine"`
/// (default) or `"euclidean"`. Deterministic for a fixed `seed`.
#[allow(clippy::too_many_arguments)]
pub fn umap(
    data: &[Vec<f64>],
    n_components: usize,
    n_neighbors: usize,
    min_dist: f64,
    spread: f64,
    n_epochs: usize,
    negative_sample_rate: usize,
    repulsion_strength: f64,
    metric: &str,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n = data.len();
    if n == 0 || n_components == 0 {
        return vec![Vec::new(); n];
    }
    if n <= n_components + 1 {
        // Too few points for a meaningful graph; fall back to zeros of the right shape.
        return vec![vec![0.0; n_components]; n];
    }
    let k = n_neighbors.clamp(2, n);
    let n_epochs = if n_epochs > 0 {
        n_epochs
    } else if n <= 10_000 {
        500
    } else {
        200
    };

    let (knn_indices, knn_dists) = knn_graph(data, k, metric);
    let (sigmas, rhos) = smooth_knn_dist(&knn_dists, k, 1.0, 1.0);
    let mut edges = fuzzy_simplicial_set(&knn_indices, &knn_dists, &sigmas, &rhos, 1.0);

    // Prune edges whose weight is below max_weight / n_epochs (they would never
    // be sampled), matching umap-learn's eliminate_zeros step.
    let max_w = edges
        .iter()
        .map(|&(_, _, w)| w)
        .fold(f64::NEG_INFINITY, f64::max);
    let cutoff = if max_w > 0.0 {
        max_w / n_epochs as f64
    } else {
        0.0
    };
    edges.retain(|&(_, _, w)| w >= cutoff);
    if edges.is_empty() {
        return vec![vec![0.0; n_components]; n];
    }

    // Canonical (head, tail) ordering, matching umap-learn's `sum_duplicates()` COO
    // layout. `fuzzy_simplicial_set` emits an undirected edge's two directed halves
    // (i,j) and (j,i) *adjacently*; with the same weight they share a schedule and
    // fire back-to-back, so the sequential SGD applies the i↔j attraction twice in a
    // row (move_other moves both endpoints) and the layout over-collapses. Sorting
    // puts (i,j) in head i's block and (j,i) in head j's block, far apart, exactly as
    // in umap-learn — the primary #555 fix.
    edges.sort_by_key(|&(h, t, _)| (h, t));

    let (a, b) = find_ab_params(spread, min_dist);

    // Spectral init (umap-learn's default): seed the SGD from the graph's Laplacian
    // eigenmap so clusters start in separated basins. Falls back to seeded random init
    // in [-10, 10] (umap-learn's random-init range) when the spectral solve is
    // unreliable (isolated vertex / stalled solve).
    let mut embedding = spectral_init(&edges, n, n_components, seed).unwrap_or_else(|| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut e = vec![0.0f64; n * n_components];
        for v in e.iter_mut() {
            *v = rng.gen_range(-10.0..10.0);
        }
        e
    });

    // Rescale the initial layout so each axis spans [0, 10], matching umap-learn's
    // `10 * (e - min) / (max - min)` step applied right before the SGD (umap_.py). Without
    // it topica started at ~2x the reference linear scale (span 20 vs 10), which halves the
    // effective attractive velocity (grad ~ -2b/d for large d) so clusters never condense
    // into the density valleys HDBSCAN cuts on — the #555 collapse.
    rescale_unit_box(&mut embedding, n, n_components);

    // Small seeded jitter so a symmetric spectral layout is not an exact SGD fixed point
    // (umap-learn adds N(0, 1e-4) noise to the spectral init for the same reason).
    let mut noise_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_5A5A_1234_5678);
    for v in embedding.iter_mut() {
        *v += noise_rng.gen_range(-1e-3..1e-3);
    }

    let head: Vec<u32> = edges.iter().map(|&(h, _, _)| h).collect();
    let tail: Vec<u32> = edges.iter().map(|&(_, t, _)| t).collect();
    let weights: Vec<f64> = edges.iter().map(|&(_, _, w)| w).collect();
    let eps = make_epochs_per_sample(&weights, n_epochs);

    optimize_layout(
        &mut embedding,
        n,
        n_components,
        &head,
        &tail,
        &eps,
        a,
        b,
        repulsion_strength,
        n_epochs,
        negative_sample_rate,
        seed,
    );

    (0..n)
        .map(|i| embedding[i * n_components..(i + 1) * n_components].to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn ab_params_match_reference() {
        // Must match umap-learn's SciPy curve_fit across min_dist, especially the
        // min_dist = 0.0 default path that the old GD fit got wrong (#555): there the
        // reference is (1.933, 0.790), not the ~1.577 GD stalled at.
        for (md, ea, eb) in [
            (0.0, 1.933, 0.790),
            (0.1, 1.577, 0.895),
            (0.5, 0.583, 1.334),
        ] {
            let (a, b) = find_ab_params(1.0, md);
            assert!(
                (a - ea).abs() < 0.02,
                "min_dist={md}: a = {a}, expected {ea}"
            );
            assert!(
                (b - eb).abs() < 0.02,
                "min_dist={md}: b = {b}, expected {eb}"
            );
        }
    }

    #[test]
    fn rescale_spans_unit_box_per_axis() {
        // Two axes on very different raw scales both land in [0, 10] independently,
        // matching umap-learn's per-column min-max init rescale (the #555 fix).
        let mut e = vec![
            -10.0, 100.0, // point 0
            10.0, 200.0, // point 1
            0.0, 150.0, // point 2
        ];
        rescale_unit_box(&mut e, 3, 2);
        for d in 0..2 {
            let col: Vec<f64> = (0..3).map(|i| e[i * 2 + d]).collect();
            let lo = col.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!((lo - 0.0).abs() < 1e-9, "axis {d} min = {lo}");
            assert!((hi - 10.0).abs() < 1e-9, "axis {d} max = {hi}");
        }
        // Midpoint of each axis maps to 5.0 (point 2 is the midpoint of both).
        assert!((e[4] - 5.0).abs() < 1e-9 && (e[5] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn rescale_degenerate_axis_to_midpoint() {
        // A zero-range axis must not divide by zero — it collapses to the midpoint.
        let mut e = vec![7.0, -3.0, 7.0, 4.0];
        rescale_unit_box(&mut e, 2, 2);
        assert_eq!(e[0], 5.0, "degenerate axis 0 should be midpoint");
        assert_eq!(e[2], 5.0, "degenerate axis 0 should be midpoint");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let data: Vec<Vec<f64>> = (0..60)
            .map(|_| (0..8).map(|_| rng.gen::<f64>()).collect())
            .collect();
        let a = umap(&data, 2, 15, 0.1, 1.0, 200, 5, 1.0, "cosine", 7);
        let b = umap(&data, 2, 15, 0.1, 1.0, 200, 5, 1.0, "cosine", 7);
        assert_eq!(a, b, "UMAP must be reproducible for a fixed seed");
    }

    #[test]
    fn separates_three_blobs() {
        // Three well-separated Gaussian blobs in 10-D should keep each point
        // nearest its own blob centroid in the 2-D layout.
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let dim = 10;
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for c in 0..3 {
            let mut center = vec![0.0; dim];
            center[c] = 20.0;
            for _ in 0..40 {
                let row: Vec<f64> = (0..dim)
                    .map(|t| center[t] + (rng.gen::<f64>() - 0.5) * 4.0)
                    .collect();
                data.push(row);
                truth.push(c);
            }
        }
        let emb = umap(&data, 2, 15, 0.1, 1.0, 500, 5, 1.0, "cosine", 1);
        assert!(
            emb.iter().all(|r| r.iter().all(|v| v.is_finite())),
            "NaN in layout"
        );
        // Per-blob centroids.
        let mut cents = [[0.0f64; 2]; 3];
        let mut counts = [0.0f64; 3];
        for (i, &c) in truth.iter().enumerate() {
            cents[c][0] += emb[i][0];
            cents[c][1] += emb[i][1];
            counts[c] += 1.0;
        }
        for c in 0..3 {
            cents[c][0] /= counts[c];
            cents[c][1] /= counts[c];
        }
        let mut correct = 0;
        for (i, &c) in truth.iter().enumerate() {
            let nearest = (0..3)
                .min_by(|&x, &y| {
                    let dx = (emb[i][0] - cents[x][0]).powi(2) + (emb[i][1] - cents[x][1]).powi(2);
                    let dy = (emb[i][0] - cents[y][0]).powi(2) + (emb[i][1] - cents[y][1]).powi(2);
                    dx.partial_cmp(&dy).unwrap()
                })
                .unwrap();
            if nearest == c {
                correct += 1;
            }
        }
        assert!(
            correct as f64 / truth.len() as f64 > 0.9,
            "UMAP did not separate the blobs: {correct}/{}",
            truth.len()
        );
    }

    #[test]
    fn spectral_init_separates_two_components() {
        // Two dense cliques joined by a single weak bridge (a barbell) — one connected
        // component with a bottleneck, like a real fuzzy kNN graph over two clusters.
        // The Laplacian eigenmap's first non-trivial (Fiedler) component must take
        // opposite signs on the two sides, the cluster-separating structure random init
        // lacks (the umap-learn default init topica now matches).
        let mut edges: Vec<(u32, u32, f64)> = Vec::new();
        let block = 15u32;
        for grp in 0..2u32 {
            let base = grp * block;
            for i in 0..block {
                for j in (i + 1)..block {
                    edges.push((base + i, base + j, 1.0));
                    edges.push((base + j, base + i, 1.0));
                }
            }
        }
        // Weak bridge so the graph is connected (the Fiedler vector then separates).
        edges.push((0, block, 0.01));
        edges.push((block, 0, 0.01));
        let n = (block * 2) as usize;
        let layout = spectral_init(&edges, n, 2, 42).expect("spectral init should succeed");
        // First component averaged over each clique — the two means must have opposite
        // sign and be well separated.
        let mean = |g: usize| {
            let mut s = 0.0;
            for i in 0..block as usize {
                s += layout[(g * block as usize + i) * 2];
            }
            s / block as f64
        };
        let (m0, m1) = (mean(0), mean(1));
        assert!(
            m0 * m1 < 0.0 && (m0 - m1).abs() > 0.1,
            "spectral init did not separate the two components: means {m0}, {m1}"
        );
        // Deterministic for a fixed seed.
        let again = spectral_init(&edges, n, 2, 42).unwrap();
        assert_eq!(layout, again, "spectral init must be reproducible");
    }
}
