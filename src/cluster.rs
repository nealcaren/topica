//! Density clustering for the embedding-model branch (Top2Vec, BERTopic).
//!
//! A thin wrapper over `petal-clustering`'s HDBSCAN so the rest of topica keeps
//! working in plain `Vec<Vec<f64>>` and never sees `ndarray` directly. HDBSCAN
//! is the clustering stage both Top2Vec and BERTopic run after reducing the
//! document embeddings: it finds clusters of varying density and leaves sparse
//! points unassigned (the "outlier" topic, conventionally label `-1`).

use ndarray::Array2;
use petal_clustering::{Fit, HDbscan};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Cluster `points` (row-major; each row is one embedding vector) with HDBSCAN.
///
/// Returns one label per point: cluster ids are a dense `0..n_clusters` range
/// (assigned in a deterministic order), and noise points get `-1`, matching the
/// HDBSCAN / BERTopic outlier convention. `min_cluster_size` is the smallest
/// group that counts as a topic; `min_samples` controls how conservative the
/// density estimate is (larger = more points called noise).
pub fn hdbscan_labels(
    points: &[Vec<f64>],
    min_cluster_size: usize,
    min_samples: usize,
) -> Vec<i64> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = points[0].len();
    assert!(
        points.iter().all(|r| r.len() == dim),
        "all points must share the same dimensionality"
    );
    let mcs = min_cluster_size.max(2);
    let ms = min_samples.max(1);
    // petal-clustering's MST / core-distance step panics when the corpus is too
    // small for the requested density parameters: it tries to read `min_samples`
    // nearest neighbours from fewer points (e.g. min_cluster_size=100 on 2 docs).
    // No cluster can form in that regime, so report all-noise (num_topics=0) and
    // let the caller's "lower min_cluster_size / add data" warning fire, rather
    // than letting a Rust panic escape into Python.
    if n < mcs || n <= ms {
        return vec![-1i64; n];
    }

    let flat: Vec<f64> = points.iter().flat_map(|r| r.iter().copied()).collect();
    let array = Array2::from_shape_vec((n, dim), flat).expect("point matrix shape");

    let mut model: HDbscan<f64, _> = HDbscan::default();
    model.min_cluster_size = mcs;
    model.min_samples = ms;
    let (clusters, _outliers, _scores) = model.fit(&array, None);

    // Map petal's arbitrary cluster keys to a dense, deterministic 0..k.
    let mut ids: Vec<usize> = clusters.keys().copied().collect();
    ids.sort_unstable();
    let mut labels = vec![-1i64; n];
    for (new_id, old_id) in ids.into_iter().enumerate() {
        for &idx in &clusters[&old_id] {
            labels[idx] = new_id as i64;
        }
    }
    labels
}

/// Dispatch to the requested clustering method. `clusterer` is `"hdbscan"`
/// (default; uses `min_cluster_size`/`min_samples`), `"kmeans"`, or
/// `"agglomerative"` (both use `num_clusters`, falling back to `min_cluster_size`
/// if it is `None`). Unknown names fall back to HDBSCAN.
pub fn cluster_points(
    points: &[Vec<f64>],
    clusterer: &str,
    num_clusters: Option<usize>,
    min_cluster_size: usize,
    min_samples: usize,
    seed: u64,
) -> Vec<i64> {
    match clusterer {
        "kmeans" => kmeans_labels(points, num_clusters.unwrap_or(min_cluster_size), seed),
        "agglomerative" => agglomerative_labels(points, num_clusters.unwrap_or(min_cluster_size)),
        _ => hdbscan_labels(points, min_cluster_size, min_samples),
    }
}

fn sqdist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Remap the assigned labels (all `>= 0`) to a dense `0..m` range in ascending
/// order, dropping any cluster id that ended up empty. Keeps downstream
/// `num_topics = max(label) + 1` gap-free.
fn densify(labels: &mut [i64]) {
    let mut seen: Vec<i64> = labels.iter().copied().filter(|&l| l >= 0).collect();
    seen.sort_unstable();
    seen.dedup();
    for l in labels.iter_mut() {
        if *l >= 0 {
            *l = seen.binary_search(l).unwrap() as i64;
        }
    }
}

/// K-means (Lloyd) with k-means++ seeding. Every point is assigned to its nearest
/// centroid, so there is no `-1` noise label — useful when every document must
/// land in a topic. Deterministic for a fixed `seed`; `k` is clamped to `1..=n`,
/// and empty clusters are dropped so the returned ids are a dense `0..m` range.
pub fn kmeans_labels(points: &[Vec<f64>], k: usize, seed: u64) -> Vec<i64> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = points[0].len();
    let k = k.clamp(1, n);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // k-means++ seeding: each new center is sampled with probability proportional
    // to its squared distance from the nearest existing center.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    centroids.push(points[rng.gen_range(0..n)].clone());
    let mut d2 = vec![f64::INFINITY; n];
    while centroids.len() < k {
        let c = centroids.last().unwrap();
        let mut sum = 0.0;
        for i in 0..n {
            let dist = sqdist(&points[i], c);
            if dist < d2[i] {
                d2[i] = dist;
            }
            sum += d2[i];
        }
        if sum <= 0.0 {
            break; // all remaining points coincide with a center
        }
        let mut target = rng.gen::<f64>() * sum;
        let mut chosen = n - 1;
        for (i, &di) in d2.iter().enumerate() {
            target -= di;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(points[chosen].clone());
    }

    // Lloyd iterations.
    let kc = centroids.len();
    let mut labels = vec![0i64; n];
    for _ in 0..100 {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0usize;
            let mut bestd = f64::INFINITY;
            for (c, cen) in centroids.iter().enumerate() {
                let d = sqdist(&points[i], cen);
                if d < bestd {
                    bestd = d;
                    best = c;
                }
            }
            if labels[i] != best as i64 {
                labels[i] = best as i64;
                changed = true;
            }
        }
        let mut sums = vec![vec![0.0; dim]; kc];
        let mut counts = vec![0usize; kc];
        for i in 0..n {
            let c = labels[i] as usize;
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += points[i][d];
            }
        }
        for c in 0..kc {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    densify(&mut labels);
    labels
}

/// Agglomerative clustering (average linkage, Lance-Williams) cut at `k`
/// clusters. Every point is assigned (no `-1`). `k` is clamped to `1..=n`. This
/// is O(n^2) memory and O(n^2 k') time, so it suits moderate corpora; for large
/// ones prefer `kmeans_labels`.
pub fn agglomerative_labels(points: &[Vec<f64>], k: usize) -> Vec<i64> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.clamp(1, n);
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active = vec![true; n];
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = sqdist(&points[i], &points[j]).sqrt();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    let mut num_active = n;
    while num_active > k {
        // Closest active pair (ties broken by the lower index pair).
        let mut bi = 0usize;
        let mut bj = 0usize;
        let mut bd = f64::INFINITY;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if active[j] && dist[i][j] < bd {
                    bd = dist[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }
        // Merge bj into bi with the average-linkage update.
        let ni = members[bi].len() as f64;
        let nj = members[bj].len() as f64;
        for m in 0..n {
            if !active[m] || m == bi || m == bj {
                continue;
            }
            let new_d = (ni * dist[bi][m] + nj * dist[bj][m]) / (ni + nj);
            dist[bi][m] = new_d;
            dist[m][bi] = new_d;
        }
        let moved = std::mem::take(&mut members[bj]);
        members[bi].extend(moved);
        active[bj] = false;
        num_active -= 1;
    }
    let mut labels = vec![-1i64; n];
    let mut cid = 0i64;
    for i in 0..n {
        if active[i] {
            for &m in &members[i] {
                labels[m] = cid;
            }
            cid += 1;
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two tight, well-separated blobs should fall into two distinct clusters.
    // HDBSCAN may legitimately call a few border points noise, so we assert on
    // each blob's majority label rather than demanding every point be assigned.
    #[test]
    fn separates_two_blobs() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut pts = Vec::new();
        for _ in 0..30 {
            pts.push(vec![rng.gen::<f64>() * 0.3, rng.gen::<f64>() * 0.3]); // near origin
        }
        for _ in 0..30 {
            pts.push(vec![
                5.0 + rng.gen::<f64>() * 0.3,
                5.0 + rng.gen::<f64>() * 0.3,
            ]); // near (5,5)
        }
        let labels = hdbscan_labels(&pts, 5, 2);

        let majority = |slice: &[i64]| {
            let mut counts = std::collections::HashMap::new();
            for &l in slice {
                if l >= 0 {
                    *counts.entry(l).or_insert(0) += 1;
                }
            }
            counts
                .into_iter()
                .max_by_key(|&(_, c)| c)
                .map(|(l, c)| (l, c))
        };
        let (a, na) = majority(&labels[..30]).expect("blob 1 has a cluster");
        let (b, nb) = majority(&labels[30..]).expect("blob 2 has a cluster");
        assert!(a != b, "blobs share a label: {labels:?}");
        assert!(na >= 24 && nb >= 24, "blobs too fragmented: {labels:?}");
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(hdbscan_labels(&[], 4, 2).is_empty());
        assert!(kmeans_labels(&[], 4, 0).is_empty());
        assert!(agglomerative_labels(&[], 4).is_empty());
    }

    #[test]
    fn tiny_corpus_with_huge_min_cluster_size_is_all_noise() {
        // Regression for the petal-clustering MST panic (issue #122): a corpus
        // smaller than min_cluster_size / min_samples must report all-noise, not
        // panic inside the crate.
        let pts = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let labels = hdbscan_labels(&pts, 100, 100);
        assert_eq!(labels, vec![-1, -1]);
    }

    fn two_blobs() -> Vec<Vec<f64>> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut pts = Vec::new();
        for _ in 0..30 {
            pts.push(vec![rng.gen::<f64>() * 0.3, rng.gen::<f64>() * 0.3]);
        }
        for _ in 0..30 {
            pts.push(vec![
                5.0 + rng.gen::<f64>() * 0.3,
                5.0 + rng.gen::<f64>() * 0.3,
            ]);
        }
        pts
    }

    // KMeans and agglomerative must assign *every* point (no -1) and split the two
    // well-separated blobs cleanly.
    #[test]
    fn kmeans_assigns_everything_and_splits_blobs() {
        let pts = two_blobs();
        let labels = kmeans_labels(&pts, 2, 0);
        assert_eq!(labels.len(), 60);
        assert!(labels.iter().all(|&l| l >= 0), "no point may be noise");
        assert!(labels[..30].iter().all(|&l| l == labels[0]));
        assert!(labels[30..].iter().all(|&l| l == labels[59]));
        assert!(labels[0] != labels[59]);
    }

    #[test]
    fn agglomerative_assigns_everything_and_splits_blobs() {
        let pts = two_blobs();
        let labels = agglomerative_labels(&pts, 2);
        assert!(labels.iter().all(|&l| l >= 0));
        assert!(labels[..30].iter().all(|&l| l == labels[0]));
        assert!(labels[30..].iter().all(|&l| l == labels[59]));
        assert!(labels[0] != labels[59]);
    }

    // k larger than the obvious structure stays within bounds and dense.
    #[test]
    fn kmeans_clamps_and_densifies() {
        let pts = two_blobs();
        let labels = kmeans_labels(&pts, 5, 1);
        let max = *labels.iter().max().unwrap();
        assert!(max >= 0 && (max as usize) < 5);
        // dense: every id in 0..=max appears.
        for id in 0..=max {
            assert!(labels.contains(&id), "id {id} missing -> not dense");
        }
    }
}
