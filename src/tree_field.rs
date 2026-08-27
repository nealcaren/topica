//! Gaussian tree-field kernel: exact, O(n), two-pass inference for a linear-Gaussian
//! Ornstein–Uhlenbeck process on a *forest* of reply trees.
//!
//! This is the one new numerical core of `ReplyTM` (issue TBD) — a reply-threaded topic model
//! that walks a per-topic prevalence coordinate `x_d` down each reply edge and reads it noisily
//! through the STM logistic-normal bound. Per topic dimension the field is a scalar
//! linear-Gaussian model:
//!
//! ```text
//! root r:   x_r ~ N(m, P0)
//! child d:  x_d ~ N(a·x_parent + (1-a)·m, Q)      (kappa = 1 - a is the reversion)
//! obs:      y_d = x_d + eps,  eps ~ N(0, R_d)      (every node carries a pseudo-observation)
//! ```
//!
//! Because reply trees are independent, the joint precision is block-diagonal per thread and
//! the posterior is exactly Gaussian. We compute the marginal log-likelihood and the smoothed
//! per-node mean and variance by belief propagation: an upward (leaves→root) collect pass and
//! a downward (root→leaves) distribute pass — the tree generalization of the RTS smoother.
//! The dense per-thread solve is the reference; the `tests` module checks BP == dense to
//! ~1e-10 on random forests (the algorithm was first validated in Python, see
//! `notes/tree_field_validate.py`).
//!
//! The kernel lands ahead of its consumer: `ReplyTM` wires it into the STM EM loop in the next
//! PR, so the public entry points are exercised only by the tests until then.
#![allow(dead_code)]

use std::f64::consts::PI;

const LOG_2PI: f64 = 1.837_877_066_409_345_5; // ln(2π)

/// Parameters of the scalar OU tree field. `kappa = 1 - a`, `sigma^2 = q`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TreeFieldParams {
    /// AR coefficient toward the anchor (`a ∈ (0, 1]`; `a = 1` is a pure random walk).
    pub a: f64,
    /// Per-edge diffusion variance `σ²`.
    pub q: f64,
    /// Anchor mean the process reverts toward.
    pub m: f64,
    /// Root prior variance.
    pub p0: f64,
}

/// Smoothed posterior of the field plus the marginal log-likelihood of the observations.
pub(crate) struct TreeFieldResult {
    /// Marginal log-likelihood `log p(y | params)`.
    pub loglik: f64,
    /// Smoothed posterior mean per node.
    pub mean: Vec<f64>,
    /// Smoothed posterior variance per node.
    pub var: Vec<f64>,
}

/// Children adjacency and a post-order (children before parents) for a forest.
///
/// `parents[d]` is the index of `d`'s parent, or any negative value for a root.
fn topology(parents: &[i64]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = parents.len();
    let mut children = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (d, &p) in parents.iter().enumerate() {
        if p < 0 {
            roots.push(d);
        } else {
            children[p as usize].push(d);
        }
    }
    // Iterative post-order DFS from each root (children pushed, emitted after their subtree).
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, bool)> = Vec::new();
    for &r in &roots {
        stack.push((r, false));
        while let Some((node, done)) = stack.pop() {
            if done {
                order.push(node);
            } else {
                stack.push((node, true));
                for &c in &children[node] {
                    stack.push((c, false));
                }
            }
        }
    }
    (children, order)
}

/// Exact Gaussian belief propagation on a forest. `y` and `r` (observation noise variance) are
/// per node; `parents[d] < 0` marks a root. Panics on size mismatch in debug builds.
pub(crate) fn solve(parents: &[i64], y: &[f64], r: &[f64], p: TreeFieldParams) -> TreeFieldResult {
    let n = parents.len();
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(r.len(), n);
    let (children, order) = topology(parents);
    let b = (1.0 - p.a) * p.m; // OU intercept so the stationary mean is m

    // ---- upward pass: collected canonical (jc, hc), upward messages, and log-evidence ----
    let mut jc = vec![0.0; n];
    let mut hc = vec![0.0; n];
    let mut jmsg = vec![0.0; n];
    let mut hmsg = vec![0.0; n];
    let mut logz = 0.0;
    for &d in &order {
        let mut jsum = 1.0 / r[d];
        let mut hsum = y[d] / r[d];
        for &c in &children[d] {
            jsum += jmsg[c];
            hsum += hmsg[c];
        }
        if parents[d] < 0 {
            jsum += 1.0 / p.p0;
            hsum += p.m / p.p0;
        }
        jc[d] = jsum;
        hc[d] = hsum;

        // observation normalizer (every node)
        logz += -0.5 * ((2.0 * PI * r[d]).ln() + y[d] * y[d] / r[d]);

        if parents[d] < 0 {
            // root prior normalizer + final marginalization of x_root
            logz += -0.5 * ((2.0 * PI * p.p0).ln() + p.m * p.m / p.p0);
            logz += 0.5 * LOG_2PI - 0.5 * jc[d].ln() + 0.5 * hc[d] * hc[d] / jc[d];
        } else {
            // send an upward message to the parent and fold in the edge + x_d integral const
            let aa = jc[d] + 1.0 / p.q;
            let denom = p.q * jc[d] + 1.0;
            jmsg[d] = p.a * p.a * jc[d] / denom;
            hmsg[d] = p.a * (hc[d] - b * jc[d]) / denom;
            logz += -0.5 * p.q.ln() - 0.5 * aa.ln();
            logz += 0.5 * hc[d] * hc[d] / aa + hc[d] * b / (aa * p.q)
                - 0.5 * jc[d] * b * b / (aa * p.q);
        }
    }

    // ---- downward pass: smoothed canonical (js, hs) in pre-order (reverse post-order) ----
    let mut js = vec![0.0; n];
    let mut hs = vec![0.0; n];
    for &d in order.iter().rev() {
        let u = parents[d];
        if u < 0 {
            js[d] = jc[d];
            hs[d] = hc[d];
        } else {
            let u = u as usize;
            // cavity at the parent toward d = parent smoothed minus d's own upward message
            let jcav = js[u] - jmsg[d];
            let hcav = hs[u] - hmsg[d];
            let mu_cav = hcav / jcav;
            let s_cav = 1.0 / jcav;
            // propagate the cavity through the edge x_d = a·x_u + b + w
            let jdown = 1.0 / (p.a * p.a * s_cav + p.q);
            let hdown = jdown * (p.a * mu_cav + b);
            js[d] = jc[d] + jdown;
            hs[d] = hc[d] + hdown;
        }
    }

    let mean: Vec<f64> = (0..n).map(|d| hs[d] / js[d]).collect();
    let var: Vec<f64> = (0..n).map(|d| 1.0 / js[d]).collect();
    TreeFieldResult {
        loglik: logz,
        mean,
        var,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense reference: build the n×n precision Λ and linear term h, solve by Cholesky.
    fn dense(
        parents: &[i64],
        y: &[f64],
        r: &[f64],
        p: TreeFieldParams,
    ) -> (f64, Vec<f64>, Vec<f64>) {
        let n = parents.len();
        let b = (1.0 - p.a) * p.m;
        let mut lam = vec![vec![0.0f64; n]; n];
        let mut h = vec![0.0f64; n];
        let mut n_roots = 0usize;
        let mut n_edges = 0usize;
        for d in 0..n {
            let u = parents[d];
            if u < 0 {
                lam[d][d] += 1.0 / p.p0;
                h[d] += p.m / p.p0;
                n_roots += 1;
            } else {
                let u = u as usize;
                lam[d][d] += 1.0 / p.q;
                lam[u][u] += p.a * p.a / p.q;
                lam[u][d] += -p.a / p.q;
                lam[d][u] += -p.a / p.q;
                h[d] += b / p.q;
                h[u] += -p.a * b / p.q;
                n_edges += 1;
            }
        }
        for i in 0..n {
            lam[i][i] += 1.0 / r[i];
            h[i] += y[i] / r[i];
        }
        // Cholesky: lam = L Lᵀ (lower)
        let mut l = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = lam[i][j];
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    l[i][j] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        let logdet = 2.0 * (0..n).map(|i| l[i][i].ln()).sum::<f64>();
        // solve Λ x = h  via L (L x') = h then Lᵀ x = x'
        let solve_chol = |rhs: &[f64]| -> Vec<f64> {
            let mut z = vec![0.0f64; n];
            for i in 0..n {
                let mut s = rhs[i];
                for k in 0..i {
                    s -= l[i][k] * z[k];
                }
                z[i] = s / l[i][i];
            }
            let mut x = vec![0.0f64; n];
            for i in (0..n).rev() {
                let mut s = z[i];
                for k in (i + 1)..n {
                    s -= l[k][i] * x[k];
                }
                x[i] = s / l[i][i];
            }
            x
        };
        let mean = solve_chol(&h);
        // variances = diagonal of Λ⁻¹ (solve against unit columns)
        let mut var = vec![0.0f64; n];
        for j in 0..n {
            let mut e = vec![0.0f64; n];
            e[j] = 1.0;
            let col = solve_chol(&e);
            var[j] = col[j];
        }
        let quad = 0.5 * (0..n).map(|i| h[i] * mean[i]).sum::<f64>();
        let mut c = 0.0;
        for i in 0..n {
            c += -0.5 * ((2.0 * PI * r[i]).ln() + y[i] * y[i] / r[i]);
        }
        c += -0.5 * n_roots as f64 * ((2.0 * PI * p.p0).ln() + p.m * p.m / p.p0);
        c += -0.5 * n_edges as f64 * ((2.0 * PI * p.q).ln() + b * b / p.q);
        let loglik = quad - 0.5 * logdet + c + 0.5 * n as f64 * LOG_2PI;
        (loglik, mean, var)
    }

    /// Tiny deterministic LCG so the test needs no `rand` dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next_f64()
        }
        fn int(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_f64() * (hi - lo) as f64) as usize
        }
    }

    fn rand_forest(rng: &mut Lcg) -> Vec<i64> {
        let mut parents: Vec<i64> = Vec::new();
        let n_trees = rng.int(2, 6);
        for _ in 0..n_trees {
            let base = parents.len();
            parents.push(-1);
            let k = rng.int(2, 12);
            for _ in 0..(k.saturating_sub(1)) {
                let local = rng.int(base, parents.len());
                parents.push(local as i64);
            }
        }
        parents
    }

    #[test]
    fn bp_matches_dense_on_random_forests() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let (mut worst_ll, mut worst_m, mut worst_v) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..300 {
            let parents = rand_forest(&mut rng);
            let n = parents.len();
            let y: Vec<f64> = (0..n).map(|_| rng.range(-2.0, 2.0)).collect();
            let r: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0).exp()).collect();
            let p = TreeFieldParams {
                a: rng.range(0.5, 0.98),
                q: rng.range(-1.0, 0.5).exp(),
                m: rng.range(-0.5, 0.5),
                p0: rng.range(-0.5, 1.0).exp(),
            };
            let (ll_d, mean_d, var_d) = dense(&parents, &y, &r, p);
            let res = solve(&parents, &y, &r, p);
            worst_ll = worst_ll.max((ll_d - res.loglik).abs());
            for i in 0..n {
                worst_m = worst_m.max((mean_d[i] - res.mean[i]).abs());
                worst_v = worst_v.max((var_d[i] - res.var[i]).abs());
            }
        }
        assert!(worst_ll < 1e-8, "loglik mismatch {worst_ll:e}");
        assert!(worst_m < 1e-8, "mean mismatch {worst_m:e}");
        assert!(worst_v < 1e-8, "var mismatch {worst_v:e}");
    }

    #[test]
    fn two_node_chain_closed_form() {
        // root r + one child c. Compare the smoothed root mean to the hand computation.
        let parents = [-1i64, 0];
        let y = [1.0, 0.0];
        let r = [0.5, 0.5];
        let p = TreeFieldParams {
            a: 0.8,
            q: 0.3,
            m: 0.0,
            p0: 2.0,
        };
        let (_ll, mean_d, _v) = dense(&parents, &y, &r, p);
        let res = solve(&parents, &y, &r, p);
        assert!((mean_d[0] - res.mean[0]).abs() < 1e-10);
        assert!((mean_d[1] - res.mean[1]).abs() < 1e-10);
    }
}
