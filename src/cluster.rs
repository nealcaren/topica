//! Density clustering for the embedding-model branch (Top2Vec, BERTopic).
//!
//! A thin wrapper over `petal-clustering`'s HDBSCAN so the rest of topica keeps
//! working in plain `Vec<Vec<f64>>` and never sees `ndarray` directly. HDBSCAN
//! is the clustering stage both Top2Vec and BERTopic run after reducing the
//! document embeddings: it finds clusters of varying density and leaves sparse
//! points unassigned (the "outlier" topic, conventionally label `-1`).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

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
    // Faithful HDBSCAN* (crate::hdbscan), matching the reference `hdbscan` package.
    // Replaces petal-clustering, whose excess-of-mass selection diverged from the
    // reference on real embedding projections (issue #555). The reimplementation
    // handles the too-few-points / all-noise regimes internally.
    crate::hdbscan::labels(points, min_cluster_size, min_samples)
}

/// Dispatch to the requested clustering method. `clusterer` is `"hdbscan"`
/// (default; uses `min_cluster_size`/`min_samples`), `"kmeans"`, `"gmm"`, or
/// `"agglomerative"` (all three use `num_clusters`, falling back to
/// `min_cluster_size` if it is `None`), or the auto-K graph clusterers
/// `"louvain"` / `"leiden"` (modularity on a k-NN graph; steered by `resolution`
/// and `knn_k`, no `num_clusters`). `resolution`/`knn_k` are ignored by the
/// non-graph clusterers. Unknown names fall back to HDBSCAN.
#[allow(clippy::too_many_arguments)]
pub fn cluster_points(
    points: &[Vec<f64>],
    clusterer: &str,
    num_clusters: Option<usize>,
    min_cluster_size: usize,
    min_samples: usize,
    resolution: f64,
    knn_k: usize,
    seed: u64,
) -> Vec<i64> {
    match clusterer {
        "kmeans" => kmeans_labels(points, num_clusters.unwrap_or(min_cluster_size), seed),
        "gmm" => gmm_labels(points, num_clusters.unwrap_or(min_cluster_size), seed),
        "agglomerative" => agglomerative_labels(points, num_clusters.unwrap_or(min_cluster_size)),
        "louvain" => graph_labels(points, resolution, knn_k, seed, false),
        "leiden" => graph_labels(points, resolution, knn_k, seed, true),
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

/// k-means++ seeding: pick `k` initial centers, each new one sampled with
/// probability proportional to its squared distance from the nearest center
/// already chosen. Deterministic for a fixed `rng`. May return fewer than `k`
/// centers when the remaining points all coincide with a chosen center (the
/// squared-distance mass collapses to zero); callers treat `centroids.len()` as
/// the effective component count.
fn kmeanspp_init(points: &[Vec<f64>], k: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<f64>> {
    let n = points.len();
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
    centroids
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

    let mut centroids = kmeanspp_init(points, k, &mut rng);

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

/// Run the diagonal-covariance GMM EM and return `(raw component label per point,
/// posterior responsibilities)`. Labels are the argmax component in `0..kc` (not
/// yet densified); `resp[i]` is the length-`kc` soft membership of point `i`. The
/// public `gmm_labels` / `gmm_soft_labels` wrap this. See `gmm_labels` for the
/// modeling notes (k-means++ init, log-sum-exp E-step, `reg_covar` floor).
fn gmm_fit(points: &[Vec<f64>], k: usize, seed: u64) -> (Vec<i64>, Vec<Vec<f64>>) {
    let n = points.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let dim = points[0].len();
    let k = k.clamp(1, n);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Means from k-means++; the effective component count is whatever it returned
    // (fewer than k only when points coincide).
    let mut means = kmeanspp_init(points, k, &mut rng);
    let kc = means.len();

    // Global per-dimension variance seeds every component's variance and sets the
    // regularization floor, so the model is scale-aware without a magic constant.
    let mut gmean = vec![0.0f64; dim];
    for p in points {
        for d in 0..dim {
            gmean[d] += p[d];
        }
    }
    for g in gmean.iter_mut() {
        *g /= n as f64;
    }
    let mut gvar = vec![0.0f64; dim];
    for p in points {
        for d in 0..dim {
            let z = p[d] - gmean[d];
            gvar[d] += z * z;
        }
    }
    for g in gvar.iter_mut() {
        *g /= n as f64;
    }
    let mean_var = (gvar.iter().sum::<f64>() / dim.max(1) as f64).max(1e-12);
    let reg_covar = 1e-6 * mean_var;

    let mut var: Vec<Vec<f64>> = (0..kc)
        .map(|_| gvar.iter().map(|&v| v.max(reg_covar)).collect())
        .collect();
    let mut weights = vec![1.0 / kc as f64; kc];

    let mut resp = vec![vec![0.0f64; kc]; n];
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let mut prev_ll = f64::NEG_INFINITY;
    for _ in 0..100 {
        // E-step: posterior responsibilities via log-sum-exp for numerical safety.
        let mut ll = 0.0;
        let mut logp = vec![0.0f64; kc];
        for i in 0..n {
            for c in 0..kc {
                let mut lp = weights[c].max(1e-300).ln();
                for d in 0..dim {
                    let v = var[c][d];
                    let z = points[i][d] - means[c][d];
                    lp += -0.5 * (ln_2pi + v.ln() + z * z / v);
                }
                logp[c] = lp;
            }
            let m = logp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for &lp in &logp {
                sum += (lp - m).exp();
            }
            let lse = m + sum.ln();
            ll += lse;
            for c in 0..kc {
                resp[i][c] = (logp[c] - lse).exp();
            }
        }

        // M-step: closed-form weight/mean/variance updates from the responsibilities.
        for c in 0..kc {
            let nk: f64 = (0..n).map(|i| resp[i][c]).sum();
            if nk <= 1e-12 {
                weights[c] = 0.0; // component died; the ln(1e-300) floor keeps it dead
                continue;
            }
            weights[c] = nk / n as f64;
            for d in 0..dim {
                let mu: f64 = (0..n).map(|i| resp[i][c] * points[i][d]).sum::<f64>() / nk;
                means[c][d] = mu;
            }
            for d in 0..dim {
                let s: f64 = (0..n)
                    .map(|i| {
                        let z = points[i][d] - means[c][d];
                        resp[i][c] * z * z
                    })
                    .sum();
                var[c][d] = (s / nk).max(reg_covar);
            }
        }

        if (ll - prev_ll).abs() < 1e-6 * (1.0 + ll.abs()) {
            break;
        }
        prev_ll = ll;
    }

    // Hard assignment: each point to its most-likely component (raw ids in 0..kc).
    let mut labels = vec![0i64; n];
    for i in 0..n {
        let mut best = 0usize;
        let mut bestr = f64::NEG_INFINITY;
        for c in 0..kc {
            if resp[i][c] > bestr {
                bestr = resp[i][c];
                best = c;
            }
        }
        labels[i] = best as i64;
    }
    (labels, resp)
}

/// Gaussian mixture (diagonal covariance) fit by EM with k-means++ initialization.
///
/// Like `kmeans_labels` this is a fixed-`k`, assign-everything clusterer (no `-1`
/// noise bucket), so it needs `num_clusters`. Unlike k-means it models each
/// component's per-dimension spread, which lets elongated or unequal-variance
/// topics separate where k-means (equal, spherical clusters) would split them
/// wrongly; the EM posterior is also a *soft* membership, exposed by
/// `gmm_soft_labels` and collapsed here to the argmax component for the `Vec<i64>`
/// interface. Deterministic for a fixed `seed`. `k` is clamped to `1..=n`, empty
/// components are dropped so the ids form a dense `0..m` range, and a `reg_covar`
/// floor (a tiny fraction of the mean feature variance) keeps a component's
/// variance from collapsing to zero on coincident points.
pub fn gmm_labels(points: &[Vec<f64>], k: usize, seed: u64) -> Vec<i64> {
    let (mut labels, _resp) = gmm_fit(points, k, seed);
    densify(&mut labels);
    labels
}

/// GMM clustering that also returns each document's *soft* membership — the EM
/// posterior responsibilities as a `(n, num_topics)` matrix whose rows sum to one.
/// This is the mixture representation hard clustering can't give: a document that
/// sits between topics gets a blend instead of a single label. Returns
/// `(labels, doc_topic)`, both over the same dense `0..num_topics` set (empty
/// components dropped in lockstep, and each row renormalized over the surviving
/// components). `doc_topic[i].argmax()` equals `labels[i]` — the surviving
/// components always include every document's argmax.
pub fn gmm_soft_labels(points: &[Vec<f64>], k: usize, seed: u64) -> (Vec<i64>, Vec<Vec<f64>>) {
    let (raw, resp) = gmm_fit(points, k, seed);
    if raw.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Kept components = those that are some document's argmax (matches `densify`).
    let mut kept: Vec<usize> = raw.iter().map(|&l| l as usize).collect();
    kept.sort_unstable();
    kept.dedup();
    let remap: HashMap<usize, usize> = kept
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();
    let k_final = kept.len();

    let labels: Vec<i64> = raw.iter().map(|&l| remap[&(l as usize)] as i64).collect();
    let doc_topic: Vec<Vec<f64>> = resp
        .iter()
        .map(|row| {
            let mut r: Vec<f64> = kept.iter().map(|&c| row[c]).collect();
            let s: f64 = r.iter().sum();
            if s > 0.0 {
                for x in r.iter_mut() {
                    *x /= s;
                }
            } else {
                for x in r.iter_mut() {
                    *x = 1.0 / k_final as f64;
                }
            }
            r
        })
        .collect();
    (labels, doc_topic)
}

// ===================================================================
// Graph clustering: Louvain / Leiden modularity over a k-NN graph.
//
// Auto-K clusterers (like HDBSCAN, they discover the topic count rather than
// taking `num_clusters`), but — unlike HDBSCAN — every document is assigned (no
// `-1` noise bucket). `"louvain"` runs the classic local-moving + aggregation
// modularity optimizer; `"leiden"` adds the refinement phase (Traag, Waltman &
// van Eck 2019) that guarantees every returned community is internally connected,
// the property Louvain can violate. Both are deterministic for a fixed `seed`
// (the only randomness is the seeded node-visit order).
// ===================================================================

/// A weighted undirected graph in adjacency-list form, plus each node's self-loop
/// weight (nonzero only after aggregation). `adj[i]` holds `(neighbor, weight)`
/// pairs, sorted by neighbor for deterministic traversal.
struct Graph {
    adj: Vec<Vec<(usize, f64)>>,
    self_loop: Vec<f64>,
}

impl Graph {
    fn n(&self) -> usize {
        self.adj.len()
    }
    /// Weighted degree k_i = Σ_j A_ij + 2·(self-loop) — self-loops count twice, as
    /// in the standard modularity definition.
    fn degrees(&self) -> Vec<f64> {
        self.adj
            .iter()
            .enumerate()
            .map(|(i, ns)| {
                ns.iter().map(|&(_, w)| w).sum::<f64>() + 2.0 * self_loop_at(&self.self_loop, i)
            })
            .collect()
    }
}

fn self_loop_at(sl: &[f64], i: usize) -> f64 {
    sl.get(i).copied().unwrap_or(0.0)
}

/// Build a symmetric k-NN graph under Euclidean distance — the same metric
/// `kmeans_labels` / `agglomerative_labels` use (the embedding pipeline
/// L2-normalizes the reduced coordinates first, so Euclidean nearest-neighbors
/// track cosine nearest-neighbors). Each node keeps its `knn_k` nearest others; an
/// undirected edge exists if either endpoint kept the other (union symmetrization)
/// and is unweighted (weight 1.0) — the classic, scale-free kNN graph for
/// modularity clustering. O(n²·d), the same complexity class as
/// `agglomerative_labels`. Neighbor lists are sorted for deterministic traversal.
fn knn_graph(points: &[Vec<f64>], knn_k: usize) -> Graph {
    let n = points.len();
    let k = knn_k.min(n.saturating_sub(1)).max(1);

    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    let mut dists: Vec<(usize, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        dists.clear();
        for j in 0..n {
            if i == j {
                continue;
            }
            dists.push((j, sqdist(&points[i], &points[j])));
        }
        // k nearest by distance asc; ties broken by ascending index for determinism.
        dists.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for &(j, _) in dists.iter().take(k) {
            let key = if i < j { (i, j) } else { (j, i) };
            edges.insert(key, 1.0);
        }
    }

    let mut adj = vec![Vec::new(); n];
    for (&(a, b), &w) in &edges {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    for a in adj.iter_mut() {
        a.sort_by_key(|&(x, _)| x);
    }
    Graph {
        adj,
        self_loop: vec![0.0; n],
    }
}

/// A seeded Fisher-Yates permutation of `0..n` — the node-visit order for
/// local-moving. Deterministic for a fixed `rng`.
fn shuffled_order(n: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        order.swap(i, j);
    }
    order
}

/// One Louvain local-moving pass: repeatedly move each node (in `order`) to the
/// neighboring community that most increases modularity (resolution γ), until no
/// node moves. `comm` starts as one community per node and is mutated in place.
/// Returns whether any node changed community. Candidate communities are scanned
/// in ascending id with a strict-improvement rule, so the outcome is deterministic
/// given `order`.
fn louvain_local_move(
    g: &Graph,
    degree: &[f64],
    two_m: f64,
    resolution: f64,
    order: &[usize],
    comm: &mut [usize],
) -> bool {
    let n = g.n();
    let mut comm_tot = vec![0.0f64; n];
    for i in 0..n {
        comm_tot[comm[i]] += degree[i];
    }
    let mut moved_any = false;
    let mut improved = true;
    while improved {
        improved = false;
        for &i in order {
            let ci = comm[i];
            let ki = degree[i];
            // Weight from i to each neighboring community.
            let mut nbr_w: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &g.adj[i] {
                if j == i {
                    continue;
                }
                *nbr_w.entry(comm[j]).or_insert(0.0) += w;
            }
            // Remove i from its community before scoring candidates.
            comm_tot[ci] -= ki;
            let w_to_ci = *nbr_w.get(&ci).unwrap_or(&0.0);
            let mut best_c = ci;
            let mut best_gain = w_to_ci - resolution * comm_tot[ci] * ki / two_m;
            let mut cands: Vec<(usize, f64)> = nbr_w.into_iter().collect();
            cands.sort_by_key(|&(c, _)| c);
            for (c, w) in cands {
                if c == ci {
                    continue;
                }
                let gain = w - resolution * comm_tot[c] * ki / two_m;
                if gain > best_gain + 1e-12 {
                    best_gain = gain;
                    best_c = c;
                }
            }
            comm_tot[best_c] += ki;
            if best_c != ci {
                comm[i] = best_c;
                improved = true;
                moved_any = true;
            }
        }
    }
    moved_any
}

/// Aggregate `g` by the partition in `comm`: every community becomes one node,
/// inter-community edge weights sum into new edges, and intra-community weights
/// (plus old self-loops) become the new node's self-loop. `comm` is relabeled to a
/// dense `0..K` and the contracted graph is returned.
fn aggregate(g: &Graph, comm: &mut [usize]) -> Graph {
    let n = g.n();
    let mut ids: Vec<usize> = comm.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let mut remap = HashMap::new();
    for (new, &old) in ids.iter().enumerate() {
        remap.insert(old, new);
    }
    let k = ids.len();
    for c in comm.iter_mut() {
        *c = remap[c];
    }

    let mut new_self = vec![0.0f64; k];
    for i in 0..n {
        new_self[comm[i]] += self_loop_at(&g.self_loop, i);
    }
    let mut emap: HashMap<(usize, usize), f64> = HashMap::new();
    for i in 0..n {
        let ci = comm[i];
        for &(j, w) in &g.adj[i] {
            if j <= i {
                continue; // count each undirected edge once
            }
            let cj = comm[j];
            if ci == cj {
                new_self[ci] += w;
            } else {
                let key = if ci < cj { (ci, cj) } else { (cj, ci) };
                *emap.entry(key).or_insert(0.0) += w;
            }
        }
    }
    let mut adj = vec![Vec::new(); k];
    for (&(a, b), &w) in &emap {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    for a in adj.iter_mut() {
        a.sort_by_key(|&(x, _)| x);
    }
    Graph {
        adj,
        self_loop: new_self,
    }
}

/// Leiden refinement: split each community of `comm` into internally-connected
/// sub-communities. Starting from singletons, each node greedily joins the
/// best-modularity sub-community *within its own `comm` community* that it shares
/// an edge with. Because a node only ever joins a community it is directly
/// connected to, every returned sub-community is a connected subgraph — the
/// guarantee Leiden adds over Louvain, whose communities can be internally
/// disconnected. This is a deterministic reading (best positive gain, ties by
/// ascending id) of Leiden's randomized merge step. Returns a refined label per
/// node; refined ids are a subdivision of `comm`.
fn refine_partition(
    g: &Graph,
    degree: &[f64],
    two_m: f64,
    resolution: f64,
    comm: &[usize],
    order: &[usize],
) -> Vec<usize> {
    let n = g.n();
    let mut refined: Vec<usize> = (0..n).collect();
    let mut ref_tot: Vec<f64> = degree.to_vec();
    for &v in order {
        let cv = comm[v];
        let kv = degree[v];
        // Edge weight from v to each refined sub-community inside v's own community.
        let mut nbr_w: HashMap<usize, f64> = HashMap::new();
        for &(u, w) in &g.adj[v] {
            if u == v || comm[u] != cv {
                continue;
            }
            *nbr_w.entry(refined[u]).or_insert(0.0) += w;
        }
        if nbr_w.is_empty() {
            continue; // no same-community neighbor: v stays its own sub-community
        }
        ref_tot[refined[v]] -= kv;
        let mut best_c = refined[v];
        let mut best_gain = 0.0; // baseline: v alone in its current sub-community
        let mut cands: Vec<(usize, f64)> = nbr_w.into_iter().collect();
        cands.sort_by_key(|&(c, _)| c);
        for (c, w) in cands {
            let gain = w - resolution * ref_tot[c] * kv / two_m;
            if gain > best_gain + 1e-12 {
                best_gain = gain;
                best_c = c;
            }
        }
        ref_tot[best_c] += kv;
        refined[v] = best_c;
    }
    refined
}

/// Aggregate `g` by the refined partition (Leiden style): every refined
/// sub-community becomes one node, exactly as [`aggregate`], and additionally the
/// new graph's *initial* community assignment is inherited from `comm` — so
/// sub-communities that came from the same Louvain community start grouped, and
/// the next local-moving pass can move whole refined communities between them.
/// Returns `(contracted graph, old-node → new-node map, initial community per new
/// node)`, both maps densified to `0..`.
fn aggregate_refined(
    g: &Graph,
    refined: &[usize],
    comm: &[usize],
) -> (Graph, Vec<usize>, Vec<usize>) {
    let n = g.n();
    let mut ids: Vec<usize> = refined.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let mut remap = HashMap::new();
    for (new, &old) in ids.iter().enumerate() {
        remap.insert(old, new);
    }
    let k = ids.len();
    let refined_dense: Vec<usize> = refined.iter().map(|c| remap[c]).collect();

    let mut new_self = vec![0.0f64; k];
    for i in 0..n {
        new_self[refined_dense[i]] += self_loop_at(&g.self_loop, i);
    }
    let mut emap: HashMap<(usize, usize), f64> = HashMap::new();
    for i in 0..n {
        let ci = refined_dense[i];
        for &(j, w) in &g.adj[i] {
            if j <= i {
                continue;
            }
            let cj = refined_dense[j];
            if ci == cj {
                new_self[ci] += w;
            } else {
                let key = if ci < cj { (ci, cj) } else { (cj, ci) };
                *emap.entry(key).or_insert(0.0) += w;
            }
        }
    }
    let mut adj = vec![Vec::new(); k];
    for (&(a, b), &w) in &emap {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    for a in adj.iter_mut() {
        a.sort_by_key(|&(x, _)| x);
    }

    // Each new node inherits the Louvain community of the nodes it contains.
    let mut louvain_of = vec![0usize; k];
    for i in 0..n {
        louvain_of[refined_dense[i]] = comm[i];
    }
    let mut lids: Vec<usize> = louvain_of.clone();
    lids.sort_unstable();
    lids.dedup();
    let mut lremap = HashMap::new();
    for (new, &old) in lids.iter().enumerate() {
        lremap.insert(old, new);
    }
    let new_init: Vec<usize> = louvain_of.iter().map(|c| lremap[c]).collect();

    (
        Graph {
            adj,
            self_loop: new_self,
        },
        refined_dense,
        new_init,
    )
}

/// Cluster `points` by modularity optimization over a k-NN graph. With
/// `refine=false` this is Louvain (local-moving + aggregation); `refine=true`
/// selects Leiden, which inserts a refinement phase before each aggregation so the
/// contracted graph is built from connected sub-communities and the next level
/// resumes from the Louvain partition (Traag, Waltman & van Eck 2019). Auto-K,
/// assigns every point (no `-1`), deterministic for a fixed `seed`. Returns dense
/// `0..K` labels.
pub fn graph_labels(
    points: &[Vec<f64>],
    resolution: f64,
    knn_k: usize,
    seed: u64,
    refine: bool,
) -> Vec<i64> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut g = knn_graph(points, knn_k);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // For each original node, the current super-node it belongs to.
    let mut node_to_super: Vec<usize> = (0..n).collect();
    // Initial community assignment for local-moving at the current level: singletons
    // at level 0; for Leiden, the inherited Louvain partition at deeper levels.
    let mut init_comm: Vec<usize> = (0..g.n()).collect();
    // Reported labels: the Louvain community of each original node at the current
    // (coarsest reached) level. Defaults to all-singletons for the no-edge case.
    let mut labels: Vec<i64> = (0..n as i64).collect();

    loop {
        let m = g.n();
        let degree = g.degrees();
        let two_m: f64 = degree.iter().sum();
        if two_m <= 0.0 {
            break; // no edges left to merge
        }
        let mut comm = init_comm.clone();
        let order = shuffled_order(m, &mut rng);
        let moved = louvain_local_move(&g, &degree, two_m, resolution, &order, &mut comm);

        // Read the current partition back to the original nodes before mutating.
        for (o, lab) in labels.iter_mut().enumerate() {
            *lab = comm[node_to_super[o]] as i64;
        }
        if !moved {
            break; // converged at this level
        }

        if refine {
            let rorder = shuffled_order(m, &mut rng);
            let refined = refine_partition(&g, &degree, two_m, resolution, &comm, &rorder);
            let (new_g, refined_dense, new_init) = aggregate_refined(&g, &refined, &comm);
            for s in node_to_super.iter_mut() {
                *s = refined_dense[*s];
            }
            if new_g.n() == m {
                break; // no coarsening
            }
            g = new_g;
            init_comm = new_init;
        } else {
            g = aggregate(&g, &mut comm); // densifies `comm` to 0..K
            for s in node_to_super.iter_mut() {
                *s = comm[*s];
            }
            if g.n() == m {
                break; // no coarsening
            }
            init_comm = (0..g.n()).collect(); // Louvain re-singletons each level
        }
    }

    densify(&mut labels);
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
            counts.into_iter().max_by_key(|&(_, c)| c)
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
        assert!(gmm_labels(&[], 4, 0).is_empty());
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

    // GMM (like kmeans/agglomerative) assigns every point and recovers the two
    // well-separated blobs.
    #[test]
    fn gmm_assigns_everything_and_splits_blobs() {
        let pts = two_blobs();
        let labels = gmm_labels(&pts, 2, 0);
        assert_eq!(labels.len(), 60);
        assert!(labels.iter().all(|&l| l >= 0), "no point may be noise");
        assert!(labels[..30].iter().all(|&l| l == labels[0]));
        assert!(labels[30..].iter().all(|&l| l == labels[59]));
        assert!(labels[0] != labels[59]);
    }

    // GMM separates two blobs with very different spreads where equal-variance
    // k-means is prone to slicing the wide blob — the case diagonal covariances
    // are meant to handle. Assert on each blob's majority label.
    #[test]
    fn gmm_handles_unequal_variance() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut pts = Vec::new();
        for _ in 0..60 {
            // wide blob around the origin
            pts.push(vec![
                rng.gen::<f64>() * 4.0 - 2.0,
                rng.gen::<f64>() * 4.0 - 2.0,
            ]);
        }
        for _ in 0..60 {
            // tight blob far away
            pts.push(vec![
                20.0 + rng.gen::<f64>() * 0.2,
                20.0 + rng.gen::<f64>() * 0.2,
            ]);
        }
        let labels = gmm_labels(&pts, 2, 0);
        let majority = |slice: &[i64]| {
            let mut counts = std::collections::HashMap::new();
            for &l in slice {
                *counts.entry(l).or_insert(0) += 1;
            }
            counts.into_iter().max_by_key(|&(_, c)| c).unwrap()
        };
        let (a, na) = majority(&labels[..60]);
        let (b, nb) = majority(&labels[60..]);
        assert!(a != b, "blobs share a label: {labels:?}");
        assert!(na >= 55 && nb >= 55, "blobs too fragmented: {labels:?}");
    }

    // k larger than the structure stays within bounds and dense; determinism holds.
    #[test]
    fn gmm_clamps_densifies_and_is_deterministic() {
        let pts = two_blobs();
        let labels = gmm_labels(&pts, 5, 1);
        let max = *labels.iter().max().unwrap();
        assert!(max >= 0 && (max as usize) < 5);
        for id in 0..=max {
            assert!(labels.contains(&id), "id {id} missing -> not dense");
        }
        assert_eq!(labels, gmm_labels(&pts, 5, 1), "same seed -> same labels");
    }

    // gmm_soft_labels returns a (n, num_topics) soft membership whose rows sum to
    // one, whose argmax equals the hard labels, and whose hard labels match
    // gmm_labels exactly (same EM, same densify).
    #[test]
    fn gmm_soft_labels_are_normalized_and_consistent() {
        let pts = two_blobs();
        let (labels, soft) = gmm_soft_labels(&pts, 3, 1);
        assert_eq!(
            labels,
            gmm_labels(&pts, 3, 1),
            "hard labels must match gmm_labels"
        );
        let k = (*labels.iter().max().unwrap() + 1) as usize;
        assert_eq!(soft.len(), pts.len());
        for (i, row) in soft.iter().enumerate() {
            assert_eq!(row.len(), k, "each row has one column per topic");
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "row {i} sums to {s}, not 1");
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                argmax as i64, labels[i],
                "soft argmax must equal hard label"
            );
        }
    }

    // Three well-separated blobs in embedding space. The graph clusterer discovers
    // the count on its own (auto-K, no num_clusters), assigns every point (no -1),
    // and lands on three communities.
    fn three_blobs() -> Vec<Vec<f64>> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]];
        let mut pts = Vec::new();
        for c in centers {
            for _ in 0..40 {
                pts.push(vec![
                    c[0] + rng.gen::<f64>() * 0.4,
                    c[1] + rng.gen::<f64>() * 0.4,
                ]);
            }
        }
        pts
    }

    #[test]
    fn louvain_separates_blobs_and_assigns_everything() {
        let pts = three_blobs();
        let labels = graph_labels(&pts, 1.0, 15, 0, false);
        assert_eq!(labels.len(), 120);
        // Auto-K, every point assigned (no -1 noise bucket).
        assert!(labels.iter().all(|&l| l >= 0), "no point may be noise");
        let k = (*labels.iter().max().unwrap() + 1) as usize;
        assert!(
            (3..=8).contains(&k),
            "K should be near 3 blobs, got {k}: {labels:?}"
        );
        // Purity: no community may span two different blobs. (Plain Louvain can
        // over-split a uniform ball into sub-communities — the resolution knob and
        // the Leiden refinement phase address that — but it must never *merge*
        // separated topics, which is the property that matters.)
        let blobs = [&labels[..40], &labels[40..80], &labels[80..]];
        for (bi, a) in blobs.iter().enumerate() {
            for (bj, b) in blobs.iter().enumerate() {
                if bi >= bj {
                    continue;
                }
                for &la in a.iter() {
                    assert!(
                        !b.contains(&la),
                        "community {la} spans two blobs: {labels:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn louvain_is_deterministic_and_dense() {
        let pts = three_blobs();
        let a = graph_labels(&pts, 1.0, 15, 7, false);
        let b = graph_labels(&pts, 1.0, 15, 7, false);
        assert_eq!(a, b, "same seed -> same labels");
        // dense 0..=max
        let max = *a.iter().max().unwrap();
        for id in 0..=max {
            assert!(a.contains(&id), "id {id} missing -> not dense");
        }
    }

    #[test]
    fn graph_labels_handles_trivial_inputs() {
        for refine in [false, true] {
            assert!(graph_labels(&[], 1.0, 15, 0, refine).is_empty());
            assert_eq!(graph_labels(&[vec![1.0, 2.0]], 1.0, 15, 0, refine), vec![0]);
        }
    }

    // Every community Leiden returns must be connected in the k-NN graph — the
    // guarantee it adds over Louvain, whose communities can be internally
    // disconnected. Verify by BFS within each community. Also checks purity (no
    // community spans two blobs) and full assignment.
    #[test]
    fn leiden_communities_are_connected_and_pure() {
        use std::collections::HashSet;
        let pts = three_blobs();
        let labels = graph_labels(&pts, 1.0, 15, 0, true);
        assert_eq!(labels.len(), 120);
        assert!(labels.iter().all(|&l| l >= 0), "no point may be noise");

        // Purity: no community spans two different blobs.
        let blobs = [&labels[..40], &labels[40..80], &labels[80..]];
        for (bi, a) in blobs.iter().enumerate() {
            for (bj, b) in blobs.iter().enumerate() {
                if bi >= bj {
                    continue;
                }
                for &la in a.iter() {
                    assert!(
                        !b.contains(&la),
                        "community {la} spans two blobs: {labels:?}"
                    );
                }
            }
        }

        // Connectivity: BFS within each community over the same k-NN graph the
        // clusterer built.
        let g = knn_graph(&pts, 15);
        let k = (*labels.iter().max().unwrap() + 1) as usize;
        for c in 0..k as i64 {
            let members: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            let mut seen: HashSet<usize> = HashSet::new();
            let mut stack = vec![members[0]];
            seen.insert(members[0]);
            while let Some(v) = stack.pop() {
                for &(u, _) in &g.adj[v] {
                    if labels[u] == c && seen.insert(u) {
                        stack.push(u);
                    }
                }
            }
            assert_eq!(
                seen.len(),
                members.len(),
                "community {c} is internally disconnected: {members:?}"
            );
        }
    }

    #[test]
    fn leiden_is_deterministic() {
        let pts = three_blobs();
        let a = graph_labels(&pts, 1.0, 15, 3, true);
        let b = graph_labels(&pts, 1.0, 15, 3, true);
        assert_eq!(a, b, "same seed -> same labels");
    }
}
