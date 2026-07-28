//! Faithful HDBSCAN*, matching the reference `hdbscan` Python package (Campello,
//! Moulavi & Sander 2013) rather than `petal-clustering`, whose cluster selection
//! diverges badly from the reference on real embedding projections (issue #555:
//! ARI 0.12-0.16, 2 topics where the reference finds 13). The pipeline is the
//! textbook one: mutual-reachability MST, single-linkage hierarchy, condensed tree
//! (`min_cluster_size`), excess-of-mass cluster selection (`allow_single_cluster =
//! false`), then leaf labelling with a `-1` noise bucket.
//!
//! Distances/core distances/MST are dense `O(n^2)` (Prim), which is fine for the
//! embedding-clustering corpus sizes topica targets; the reference accelerates this
//! with space trees, but the extracted clustering is identical. `min_samples`
//! controls the core-distance density estimate; `min_cluster_size` the smallest
//! group that counts as a cluster.

/// HDBSCAN* labels for `points` (row-major). Returns one `i64` per point: a dense
/// cluster id `0..k`, or `-1` for noise. Deterministic.
pub fn labels(points: &[Vec<f64>], min_cluster_size: usize, min_samples: usize) -> Vec<i64> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let mcs = min_cluster_size.max(2);
    let ms = min_samples.max(1);
    // Not enough points to form even one cluster: everything is noise.
    if n <= mcs || n <= ms {
        return vec![-1i64; n];
    }
    let dim = points[0].len();

    // 1. Dense pairwise Euclidean distances.
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for d in 0..dim {
                let df = points[i][d] - points[j][d];
                s += df * df;
            }
            let v = s.sqrt();
            dist[i * n + j] = v;
            dist[j * n + i] = v;
        }
    }

    // 2. Core distance = the `min_samples`-th smallest distance in the row (self
    //    counts as index 0, matching `np.partition(D, min_samples)[min_samples]`).
    let core: Vec<f64> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| dist[i * n + j]).collect();
            let k = ms.min(n - 1);
            row.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
            row[k]
        })
        .collect();

    // 3. Mutual-reachability MST via Prim's algorithm (dense). mreach(i,j) =
    //    max(core[i], core[j], dist(i,j)). Edges are (a, b, weight).
    let mreach = |i: usize, j: usize| core[i].max(core[j]).max(dist[i * n + j]);
    let mut in_tree = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut best_from = vec![usize::MAX; n];
    best[0] = 0.0;
    let mut mst: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
    for _ in 0..n {
        // Pick the closest not-yet-in-tree vertex.
        let mut u = usize::MAX;
        let mut bd = f64::INFINITY;
        for v in 0..n {
            if !in_tree[v] && best[v] < bd {
                bd = best[v];
                u = v;
            }
        }
        if u == usize::MAX {
            break;
        }
        in_tree[u] = true;
        if best_from[u] != usize::MAX {
            mst.push((best_from[u], u, best[u]));
        }
        for v in 0..n {
            if !in_tree[v] {
                let w = mreach(u, v);
                if w < best[v] {
                    best[v] = w;
                    best_from[v] = u;
                }
            }
        }
    }

    // 4. Single-linkage hierarchy from the MST: sort edges ascending and union with
    //    the reference's relabelling union-find. hierarchy[i] = (a, b, delta, size),
    //    where a,b are current node labels (points `< n`, internal `>= n`).
    mst.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap());
    let mut uf = LinkageUnionFind::new(n);
    // hierarchy has n-1 rows; row i creates internal node n+i.
    let mut hier: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(n - 1);
    for &(a, b, delta) in &mst {
        let aa = uf.find(a);
        let bb = uf.find(b);
        let sz = uf.size(aa) + uf.size(bb);
        hier.push((aa, bb, delta, sz));
        uf.union(aa, bb);
    }

    // 5. Condense the tree, 6. compute stabilities, 7. EOM-select clusters,
    //    8. label the leaves.
    let condensed = condense_tree(&hier, n, mcs);
    let stability = compute_stability(&condensed, n);
    let selected = select_clusters_eom(&condensed, n, stability);
    do_labelling(&condensed, n, &selected)
}

/// Reference `label()` union-find: points `0..n` start as roots; each `union`
/// mints a fresh internal label `n, n+1, ...` whose size is the sum of its parts.
struct LinkageUnionFind {
    parent: Vec<isize>, // -1 = root
    size: Vec<usize>,
    next_label: usize,
}

impl LinkageUnionFind {
    fn new(n: usize) -> Self {
        let mut size = vec![0usize; 2 * n - 1];
        for s in size.iter_mut().take(n) {
            *s = 1;
        }
        LinkageUnionFind {
            parent: vec![-1; 2 * n - 1],
            size,
            next_label: n,
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        // Walk to the root (parent == -1), then path-compress.
        let start = x;
        while self.parent[x] != -1 {
            x = self.parent[x] as usize;
        }
        let root = x;
        let mut p = start;
        while self.parent[p] != -1 && self.parent[p] as usize != root {
            let nxt = self.parent[p] as usize;
            self.parent[p] = root as isize;
            p = nxt;
        }
        root
    }
    fn size(&self, x: usize) -> usize {
        self.size[x]
    }
    fn union(&mut self, a: usize, b: usize) {
        let lbl = self.next_label;
        self.parent[a] = lbl as isize;
        self.parent[b] = lbl as isize;
        self.size[lbl] = self.size[a] + self.size[b];
        self.next_label += 1;
    }
}

/// One condensed-tree edge: `child` leaves cluster `parent` at `lambda_val`.
/// `child_size` is 1 for a point, or the cluster's point count for a sub-cluster.
struct CondensedEdge {
    parent: usize,
    child: usize,
    lambda_val: f64,
    child_size: usize,
}

/// BFS the single-linkage hierarchy from `root`, returning node ids top-down.
fn bfs_hierarchy(hier: &[(usize, usize, f64, usize)], n: usize, root: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut queue = vec![root];
    while !queue.is_empty() {
        let mut next = Vec::new();
        for &node in &queue {
            out.push(node);
            if node >= n {
                let (a, b, _, _) = hier[node - n];
                next.push(a);
                next.push(b);
            }
        }
        queue = next;
    }
    out
}

/// Condense the hierarchy: split points that shed a below-`min_cluster_size` child
/// are treated as that child "falling out" (its points become noise below this
/// level) rather than a true bifurcation; only splits where both sides stay
/// `>= min_cluster_size` mint new clusters. Mirrors `hdbscan._hdbscan_tree.condense_tree`.
fn condense_tree(hier: &[(usize, usize, f64, usize)], n: usize, mcs: usize) -> Vec<CondensedEdge> {
    let root = 2 * (n - 1); // = 2n-2, the last internal node
    let mut relabel = vec![0usize; 2 * n - 1];
    relabel[root] = n;
    let mut next_label = n + 1;
    let mut ignore = vec![false; 2 * n - 1];
    let mut result: Vec<CondensedEdge> = Vec::new();

    let node_list = bfs_hierarchy(hier, n, root);
    let count_of = |node: usize| -> usize {
        if node >= n {
            hier[node - n].3
        } else {
            1
        }
    };
    for node in node_list {
        if ignore[node] || node < n {
            continue;
        }
        let (left, right, delta, _) = hier[node - n];
        let lambda_value = if delta > 0.0 {
            1.0 / delta
        } else {
            f64::INFINITY
        };
        let lc = count_of(left);
        let rc = count_of(right);

        if lc >= mcs && rc >= mcs {
            relabel[left] = next_label;
            next_label += 1;
            result.push(CondensedEdge {
                parent: relabel[node],
                child: relabel[left],
                lambda_val: lambda_value,
                child_size: lc,
            });
            relabel[right] = next_label;
            next_label += 1;
            result.push(CondensedEdge {
                parent: relabel[node],
                child: relabel[right],
                lambda_val: lambda_value,
                child_size: rc,
            });
        } else if lc < mcs && rc < mcs {
            for sub in bfs_hierarchy(hier, n, left) {
                if sub < n {
                    result.push(CondensedEdge {
                        parent: relabel[node],
                        child: sub,
                        lambda_val: lambda_value,
                        child_size: 1,
                    });
                }
                ignore[sub] = true;
            }
            for sub in bfs_hierarchy(hier, n, right) {
                if sub < n {
                    result.push(CondensedEdge {
                        parent: relabel[node],
                        child: sub,
                        lambda_val: lambda_value,
                        child_size: 1,
                    });
                }
                ignore[sub] = true;
            }
        } else if lc < mcs {
            // left falls out, right persists as the same cluster.
            relabel[right] = relabel[node];
            for sub in bfs_hierarchy(hier, n, left) {
                if sub < n {
                    result.push(CondensedEdge {
                        parent: relabel[node],
                        child: sub,
                        lambda_val: lambda_value,
                        child_size: 1,
                    });
                }
                ignore[sub] = true;
            }
        } else {
            // right falls out, left persists.
            relabel[left] = relabel[node];
            for sub in bfs_hierarchy(hier, n, right) {
                if sub < n {
                    result.push(CondensedEdge {
                        parent: relabel[node],
                        child: sub,
                        lambda_val: lambda_value,
                        child_size: 1,
                    });
                }
                ignore[sub] = true;
            }
        }
    }
    result
}

/// Cluster stability = Σ over a cluster's shed children of `(lambda - birth) * size`,
/// where `birth` is the lambda at which the cluster itself split off. Returns a map
/// keyed by cluster id (`>= n`).
fn compute_stability(
    condensed: &[CondensedEdge],
    n: usize,
) -> std::collections::HashMap<usize, f64> {
    use std::collections::HashMap;
    // birth[cluster] = lambda_val of the row where `cluster` appears as a child.
    let mut births: HashMap<usize, f64> = HashMap::new();
    for e in condensed {
        if e.child >= n {
            births.insert(e.child, e.lambda_val);
        }
    }
    // The root cluster (id n) is born at lambda 0.
    births.insert(n, 0.0);
    let mut stability: HashMap<usize, f64> = HashMap::new();
    for e in condensed {
        let birth = *births.get(&e.parent).unwrap_or(&0.0);
        *stability.entry(e.parent).or_insert(0.0) += (e.lambda_val - birth) * e.child_size as f64;
    }
    stability
}

/// Excess-of-mass selection (`allow_single_cluster = false`). Bottom-up: a cluster
/// is kept only if its own stability beats the summed stability of its selected
/// sub-clusters; otherwise that larger sum propagates upward and the cluster is
/// dropped. Selecting a cluster deselects all its descendants. Mirrors
/// `hdbscan._hdbscan_tree.get_clusters(..., cluster_selection_method="eom")`.
fn select_clusters_eom(
    condensed: &[CondensedEdge],
    n: usize,
    mut stability: std::collections::HashMap<usize, f64>,
) -> std::collections::HashSet<usize> {
    use std::collections::{HashMap, HashSet};
    // child clusters of each cluster (rows whose child is itself a cluster).
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in condensed {
        if e.child >= n {
            children.entry(e.parent).or_default().push(e.child);
        }
    }
    // Process clusters in descending id order (children before parents).
    let mut nodes: Vec<usize> = stability.keys().copied().collect();
    nodes.sort_unstable();
    nodes.reverse();
    // allow_single_cluster = false: the root (smallest id, n) is never a cluster.
    let root = n;
    let mut is_cluster: HashSet<usize> = nodes.iter().copied().filter(|&x| x != root).collect();

    for &node in &nodes {
        if node == root {
            continue;
        }
        let child_sum: f64 = children
            .get(&node)
            .map(|cs| cs.iter().map(|c| stability[c]).sum())
            .unwrap_or(0.0);
        let own = stability[&node];
        if child_sum > own {
            is_cluster.remove(&node);
            stability.insert(node, child_sum);
        } else {
            // Select `node`; drop every descendant cluster.
            let mut stack = vec![node];
            while let Some(cur) = stack.pop() {
                if let Some(cs) = children.get(&cur) {
                    for &c in cs {
                        if c != node {
                            is_cluster.remove(&c);
                            stack.push(c);
                        }
                    }
                }
            }
        }
    }
    is_cluster
}

/// Assign each point to the deepest selected cluster among its ancestors, or `-1`.
/// Selected cluster ids are densified to `0..k` in ascending order.
fn do_labelling(
    condensed: &[CondensedEdge],
    n: usize,
    selected: &std::collections::HashSet<usize>,
) -> Vec<i64> {
    use std::collections::HashMap;
    // parent_of[cluster] = its parent cluster in the condensed tree.
    let mut parent_of: HashMap<usize, usize> = HashMap::new();
    for e in condensed {
        if e.child >= n {
            parent_of.insert(e.child, e.parent);
        }
    }
    // Dense label per selected cluster (ascending id -> 0..k).
    let mut sel: Vec<usize> = selected.iter().copied().collect();
    sel.sort_unstable();
    let label_of: HashMap<usize, i64> = sel
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as i64))
        .collect();

    let mut result = vec![-1i64; n];
    for e in condensed {
        if e.child < n {
            // Walk up from the cluster the point fell out of to the nearest selected ancestor.
            let mut c = e.parent;
            loop {
                if selected.contains(&c) {
                    result[e.child] = label_of[&c];
                    break;
                }
                match parent_of.get(&c) {
                    Some(&p) if p != c => c = p,
                    _ => break,
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(cx: f64, cy: f64, k: usize, seed: &mut u64) -> Vec<Vec<f64>> {
        // deterministic tiny LCG jitter around (cx, cy)
        (0..k)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let jx = ((*seed >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.4;
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let jy = ((*seed >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.4;
                vec![cx + jx, cy + jy]
            })
            .collect()
    }

    #[test]
    fn separates_three_blobs() {
        let mut seed = 1u64;
        let mut pts = Vec::new();
        pts.extend(blob(0.0, 0.0, 30, &mut seed));
        pts.extend(blob(10.0, 0.0, 30, &mut seed));
        pts.extend(blob(0.0, 10.0, 30, &mut seed));
        let lab = labels(&pts, 10, 10);
        let k = lab
            .iter()
            .filter(|&&x| x >= 0)
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(
            k, 3,
            "three well-separated blobs must give three clusters: {lab:?}"
        );
    }

    #[test]
    fn too_few_points_is_all_noise() {
        let pts = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        assert_eq!(labels(&pts, 15, 15), vec![-1, -1]);
    }

    #[test]
    fn empty_is_empty() {
        let pts: Vec<Vec<f64>> = Vec::new();
        assert!(labels(&pts, 5, 5).is_empty());
    }
}
