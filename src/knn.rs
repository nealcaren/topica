//! Exact brute-force k-nearest-neighbor search, shared by the UMAP reducer and the
//! graph clusterers.
//!
//! Distances are screened in row blocks with a cache-blocked GEMM (`crate::gemm`):
//! for unit rows the cosine distance is `1 − a·b`, and the squared Euclidean
//! distance is `|a|² + |b|² − 2a·b`, so one `B×n` matrix product scores a whole
//! block of query rows at once. That screen only *selects candidates*: every point
//! whose screened distance lies within a small tolerance of the k-th smallest is
//! kept, and the candidates' distances are then recomputed with the plain
//! per-pair formula and sorted by `(distance, index)`. The returned neighbors and
//! distances are therefore the same, bit for bit, as a scalar scan with an
//! index tie-break; the GEMM changes only how fast they are found.

use rayon::prelude::*;

/// Distance used for the neighbor search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// `1 − cos(a, b)`, clamped at 0, computed on L2-normalized rows.
    Cosine,
    /// `|a − b|`.
    Euclidean,
    /// `|a − b|²` (same order as `Euclidean`, without the square root).
    SqEuclidean,
}

/// Rows of `data` scaled to unit length (zero rows left as is), matching the
/// cosine convention of the old per-pair scan.
fn unit_rows(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
}

/// The exact per-pair distance, written as the scalar scans always wrote it.
#[inline]
fn exact(metric: Metric, a: &[f64], b: &[f64]) -> f64 {
    match metric {
        Metric::Cosine => {
            let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
            (1.0 - dot).max(0.0)
        }
        Metric::Euclidean => a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt(),
        Metric::SqEuclidean => a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum(),
    }
}

/// For every row of `data`, its `k` nearest *other* rows as `(index, distance)`,
/// sorted by distance with ties broken by ascending index. `k` is clamped to
/// `n − 1`. Parallel over rows; deterministic regardless of thread count.
pub fn exact_knn(data: &[Vec<f64>], k: usize, metric: Metric) -> Vec<Vec<(u32, f64)>> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n - 1);
    if k == 0 {
        return vec![Vec::new(); n];
    }
    let dim = data[0].len();
    let rows: std::borrow::Cow<[Vec<f64>]> = if metric == Metric::Cosine {
        std::borrow::Cow::Owned(unit_rows(data))
    } else {
        std::borrow::Cow::Borrowed(data)
    };
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let sq: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|&v| v * v).sum())
        .collect();
    let max_sq = sq.iter().copied().fold(0.0f64, f64::max);

    // Row block sized so the B×n score matrix stays around 64 MB.
    let block = (8_000_000 / n).clamp(1, 1024).min(n);
    let mut out: Vec<Vec<(u32, f64)>> = Vec::with_capacity(n);
    let mut scores = vec![0.0f64; block * n];
    let mut start = 0;
    while start < n {
        let b = block.min(n - start);
        let s = &mut scores[..b * n];
        crate::gemm::matmul_nt(b, dim, n, &flat[start * dim..(start + b) * dim], &flat, s);
        let block_out: Vec<Vec<(u32, f64)>> = s
            .par_chunks(n)
            .enumerate()
            .map(|(r, dots)| {
                let i = start + r;
                // Screened distance, monotone in the exact one up to rounding.
                let screen = |j: usize| match metric {
                    Metric::Cosine => 1.0 - dots[j],
                    _ => sq[i] + sq[j] - 2.0 * dots[j],
                };
                let mut cand: Vec<(f64, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (screen(j), j as u32))
                    .collect();
                cand.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
                let kth = cand[..k]
                    .iter()
                    .map(|c| c.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                // Keep everything the screen cannot separate from the k-th: the
                // GEMM's rounding (~1e-15 relative) must not decide membership.
                let tol = 1e-9 * (1.0 + kth.abs() + sq[i] + max_sq);
                // Non-finite screen values (overflowing norms above ~1e154, or NaN
                // input) cannot bound anything: score every candidate exactly.
                let screen_ok = tol.is_finite() && cand[..k].iter().all(|c| c.0.is_finite());
                let mut exact_d: Vec<(f64, u32)> = cand
                    .iter()
                    .filter(|c| !screen_ok || c.0 <= kth + tol)
                    .map(|&(_, j)| (exact(metric, &rows[i], &rows[j as usize]), j))
                    .collect();
                exact_d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                exact_d.truncate(k);
                exact_d.into_iter().map(|(d, j)| (j, d)).collect()
            })
            .collect();
        out.extend(block_out);
        start += b;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// The plain O(n²) scan the GEMM path replaces.
    fn naive(data: &[Vec<f64>], k: usize, metric: Metric) -> Vec<Vec<(u32, f64)>> {
        let rows = if metric == Metric::Cosine {
            unit_rows(data)
        } else {
            data.to_vec()
        };
        (0..rows.len())
            .map(|i| {
                let mut d: Vec<(f64, u32)> = (0..rows.len())
                    .filter(|&j| j != i)
                    .map(|j| (exact(metric, &rows[i], &rows[j]), j as u32))
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                d.truncate(k.min(rows.len() - 1));
                d.into_iter().map(|(x, j)| (j, x)).collect()
            })
            .collect()
    }

    fn cloud(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f64>() - 0.5).collect())
            .collect()
    }

    #[test]
    fn matches_naive_scan_bit_for_bit() {
        for metric in [Metric::Cosine, Metric::Euclidean, Metric::SqEuclidean] {
            for &(n, dim, k) in &[(300, 16, 10), (257, 64, 15), (40, 3, 39), (5, 2, 9)] {
                let data = cloud(n, dim, n as u64);
                assert_eq!(exact_knn(&data, k, metric), naive(&data, k, metric));
            }
        }
    }

    // Exact duplicates and ties: the index tie-break must survive the screen.
    #[test]
    fn ties_break_by_index() {
        let mut data = cloud(50, 4, 3);
        for i in 10..30 {
            data[i] = data[5].clone();
        }
        for metric in [Metric::Cosine, Metric::Euclidean, Metric::SqEuclidean] {
            assert_eq!(exact_knn(&data, 12, metric), naive(&data, 12, metric));
        }
    }

    // Far from the origin the Euclidean expansion |a|²+|b|²−2a·b cancels badly;
    // the tolerance must widen the candidate set enough to stay exact.
    #[test]
    fn large_offset_stays_exact() {
        for &(offset, jitter) in &[(1e6, 1e-3), (1e8, 1e-4), (1e4, 1e-6)] {
            let data: Vec<Vec<f64>> = cloud(200, 8, 5)
                .into_iter()
                .map(|r| r.into_iter().map(|v| offset + jitter * v).collect())
                .collect();
            for metric in [Metric::Euclidean, Metric::SqEuclidean, Metric::Cosine] {
                assert_eq!(exact_knn(&data, 10, metric), naive(&data, 10, metric));
            }
        }
    }

    // Overflowing norms make the screen non-finite; fall back to exact scoring.
    #[test]
    fn overflowing_norms_fall_back_to_exact() {
        let data: Vec<Vec<f64>> = cloud(60, 3, 9)
            .into_iter()
            .map(|r| r.into_iter().map(|v| 1e155 * v).collect())
            .collect();
        for metric in [Metric::Euclidean, Metric::SqEuclidean] {
            let got = exact_knn(&data, 7, metric);
            assert!(got.iter().all(|row| row.len() == 7));
            assert_eq!(got, naive(&data, 7, metric));
        }
    }

    // Multiple row blocks (block size shrinks as n grows) agree with the scan.
    #[test]
    fn spans_several_blocks() {
        let data = cloud(9000, 4, 11);
        let got = exact_knn(&data, 5, Metric::Euclidean);
        let want = naive(&data[..], 5, Metric::Euclidean);
        assert_eq!(got, want);
    }

    #[test]
    fn degenerate_inputs() {
        assert!(exact_knn(&[], 5, Metric::Cosine).is_empty());
        assert_eq!(
            exact_knn(&[vec![1.0, 0.0]], 5, Metric::Cosine),
            vec![vec![]]
        );
        // Zero row under cosine: distance 1 to everything, no NaN.
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let nn = exact_knn(&data, 2, Metric::Cosine);
        assert!(nn.iter().flatten().all(|&(_, d)| d.is_finite()));
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    #[test]
    #[ignore]
    fn time_knn_18k_384() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let data: Vec<Vec<f64>> = (0..18000)
            .map(|_| (0..384).map(|_| rng.gen::<f64>() - 0.5).collect())
            .collect();
        let t = std::time::Instant::now();
        let a = exact_knn(&data, 15, Metric::Cosine);
        eprintln!("gemm knn: {:.2}s", t.elapsed().as_secs_f64());
        let unit = unit_rows(&data);
        let t = std::time::Instant::now();
        let b: Vec<Vec<(u32, f64)>> = (0..unit.len())
            .into_par_iter()
            .map(|i| {
                let mut d: Vec<(f64, u32)> = (0..unit.len())
                    .filter(|&j| j != i)
                    .map(|j| (exact(Metric::Cosine, &unit[i], &unit[j]), j as u32))
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                d.truncate(15);
                d.into_iter().map(|(x, j)| (j, x)).collect()
            })
            .collect();
        eprintln!("old parallel scan: {:.2}s", t.elapsed().as_secs_f64());
        assert_eq!(a, b);
    }
}
