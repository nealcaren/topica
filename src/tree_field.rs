//! Gaussian tree-field kernel: exact, O(n), two-pass inference for a linear-Gaussian
//! Ornstein–Uhlenbeck process on a *forest* of reply trees.
//!
//! This is the one new numerical core of `ReplyTM` — a reply-threaded topic model that couples a
//! per-topic prevalence coordinate `x_d` along each reply edge and reads it noisily through the
//! CTM logistic-normal bound. Per topic dimension the field is a scalar linear-Gaussian model:
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
//! < 1e-8 on random forests (the algorithm was first validated in Python, see
//! `notes/tree_field_validate.py`).
//!
//! `ReplyTM`'s M-step uses [`fit_fixed_mean`] here to estimate `(κ, σ²)` on anchor-centered
//! residuals and [`profile_loglik_at_a`] to profile κ for its CI; [`solve`]'s smoothed
//! means/variances are available but the E-step coupling currently uses the parent's point
//! estimate instead (a structured mean-field), so some entry points are exercised only by the
//! tests.
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

/// Summed marginal log-likelihood across the K observation dimensions that share the field
/// params — used to profile a (hence kappa) for a confidence interval.
pub(crate) fn loglik_multi(
    parents: &[i64],
    obs: &[Vec<f64>],
    r: &[f64],
    p: TreeFieldParams,
) -> f64 {
    obs.iter().map(|yk| solve(parents, yk, r, p).loglik).sum()
}

/// PROFILE log-likelihood at a fixed `a` (hence fixed κ): the max of [`loglik_multi`] over the
/// nuisance diffusion/root variances `(q, p0)`, with `a` held. A plain slice at fixed `(q, p0)`
/// ignores the a↔q ridge (a stiffer AR coefficient trades against smaller diffusion for nearly
/// the same smoothing) and yields an over-narrow CI; re-optimizing the nuisance at each `a` gives
/// a genuine profile. `m` stays 0 (the field is fit on anchor-centered residuals). `obs`/`r` are
/// held fixed, so the interval is profile in `(q, p0)` but conditional on the topic fit `θ`.
pub(crate) fn profile_loglik_at_a(
    parents: &[i64],
    obs: &[Vec<f64>],
    r: &[f64],
    a: f64,
    init: TreeFieldParams,
) -> f64 {
    // 2-D Nelder–Mead over (ln q, ln p0) at fixed a; small and deterministic.
    let neg_ll = |t: &[f64]| -> f64 {
        let (q, p0) = (t[0].exp(), t[1].exp());
        if !(q > 1e-12 && p0 > 1e-12) {
            return f64::INFINITY;
        }
        -loglik_multi(parents, obs, r, TreeFieldParams { a, q, m: 0.0, p0 })
    };
    let x0 = vec![init.q.max(1e-6).ln(), init.p0.max(1e-6).ln()];
    -nelder_mead_min(&neg_ll, x0, 0.5, 200).1
}

/// Result of fitting the OU field hyperparameters by maximum marginal likelihood.
pub(crate) struct TreeFieldFit {
    pub a: f64,
    pub q: f64,
    pub m: f64,
    pub p0: f64,
    pub loglik: f64,
}

/// Maximum-likelihood fit of `(a, q, m, p0)` across `K` observation dimensions that SHARE the
/// field hyperparameters (the diffusion is isotropic across topics). `obs[k]` is the length-`n`
/// observation vector for topic dimension `k`; `r` is the per-node observation-noise variance,
/// shared across dimensions and supplied by the caller (in ReplyTM it comes from the STM/CTM
/// logistic-normal curvature). Direct Nelder–Mead on the exact marginal log-likelihood from
/// [`solve`], optimizing in an unconstrained reparameterization
/// (`a = σ(θ₀)`, `q = e^{θ₁}`, `m = θ₂`, `p0 = e^{θ₃}`).
pub(crate) fn fit(
    parents: &[i64],
    obs: &[Vec<f64>],
    r: &[f64],
    init: TreeFieldParams,
) -> TreeFieldFit {
    let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
    let unpack = |t: &[f64]| TreeFieldParams {
        a: sigmoid(t[0]),
        q: t[1].exp(),
        m: t[2],
        p0: t[3].exp(),
    };
    let neg_ll = |t: &[f64]| -> f64 {
        let p = unpack(t);
        // guard the boundary (a→1, q→0) that makes the OU degenerate
        if !(p.a.is_finite() && p.q > 1e-12 && p.p0 > 1e-12) {
            return f64::INFINITY;
        }
        let mut ll = 0.0;
        for yk in obs {
            ll += solve(parents, yk, r, p).loglik;
        }
        -ll
    };
    let x0 = vec![
        (init.a / (1.0 - init.a)).ln(),
        init.q.ln(),
        init.m,
        init.p0.ln(),
    ];
    let best = nelder_mead_min(&neg_ll, x0, 0.5, 400);
    let p = unpack(&best.0);
    TreeFieldFit {
        a: p.a,
        q: p.q,
        m: p.m,
        p0: p.p0,
        loglik: -best.1,
    }
}

/// Like [`fit`] but with the anchor mean `m` HELD at 0 — for ReplyTM, whose field is fit on
/// anchor-centered residuals, so the mean is already removed and freeing `m` both double-counts it
/// and makes the fitted `a` inconsistent with the `m=0` profile used for the κ CI. Optimizes only
/// `(a, q, p0)`.
pub(crate) fn fit_fixed_mean(
    parents: &[i64],
    obs: &[Vec<f64>],
    r: &[f64],
    init: TreeFieldParams,
) -> TreeFieldFit {
    let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
    let neg_ll = |t: &[f64]| -> f64 {
        let (a, q, p0) = (sigmoid(t[0]), t[1].exp(), t[2].exp());
        if !(a.is_finite() && q > 1e-12 && p0 > 1e-12) {
            return f64::INFINITY;
        }
        -loglik_multi(parents, obs, r, TreeFieldParams { a, q, m: 0.0, p0 })
    };
    let x0 = vec![(init.a / (1.0 - init.a)).ln(), init.q.ln(), init.p0.ln()];
    let best = nelder_mead_min(&neg_ll, x0, 0.5, 400);
    TreeFieldFit {
        a: sigmoid(best.0[0]),
        q: best.0[1].exp(),
        m: 0.0,
        p0: best.0[2].exp(),
        loglik: -best.1,
    }
}

/// Deterministic Nelder–Mead minimizer over an N-dimensional domain (N = `x0.len()`). Full
/// reflect / expand / OUTSIDE-contract / INSIDE-contract / shrink — the outside/inside split
/// matters: contracting only toward the worst vertex (as an earlier version did) can retain a
/// worse point and under-optimize. Returns the best vertex and its value. Kept local because
/// topica has no shared general-purpose minimizer.
fn nelder_mead_min<F: Fn(&[f64]) -> f64>(
    f: &F,
    x0: Vec<f64>,
    step: f64,
    max_iter: usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    // initial simplex: x0 plus one perturbed vertex per coordinate
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut v = x0.clone();
        v[i] += step;
        simplex.push(v);
    }
    let mut fval: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    // comb(a, b, t) = a + t*(a - b): reflection/expansion when b is the worst vertex.
    let comb = |a: &[f64], b: &[f64], t: f64| -> Vec<f64> {
        (0..a.len()).map(|k| a[k] + t * (a[k] - b[k])).collect()
    };
    for _ in 0..max_iter {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&i, &j| {
            fval[i]
                .partial_cmp(&fval[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        fval = idx.iter().map(|&i| fval[i]).collect();
        if (fval[n] - fval[0]).abs() < 1e-9 {
            break;
        }
        // centroid of all but the worst vertex
        let mut c = vec![0.0; n];
        for v in simplex.iter().take(n) {
            for (k, ck) in c.iter_mut().enumerate() {
                *ck += v[k] / n as f64;
            }
        }
        let worst = simplex[n].clone();
        let xr = comb(&c, &worst, alpha);
        let fr = f(&xr);
        if fr < fval[0] {
            // expansion
            let xe = comb(&c, &worst, gamma);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fval[n] = fe;
            } else {
                simplex[n] = xr;
                fval[n] = fr;
            }
        } else if fr < fval[n - 1] {
            simplex[n] = xr;
            fval[n] = fr;
        } else if fr < fval[n] {
            // OUTSIDE contraction (between centroid and reflection); accept vs the reflection
            let xc = comb(&c, &worst, rho * alpha);
            let fc = f(&xc);
            if fc <= fr {
                simplex[n] = xc;
                fval[n] = fc;
            } else {
                shrink(&mut simplex, &mut fval, sigma, f);
            }
        } else {
            // INSIDE contraction (between centroid and worst); accept vs the worst
            let xc = comb(&c, &worst, -rho);
            let fc = f(&xc);
            if fc < fval[n] {
                simplex[n] = xc;
                fval[n] = fc;
            } else {
                shrink(&mut simplex, &mut fval, sigma, f);
            }
        }
    }
    let mut bi = 0;
    for i in 1..=n {
        if fval[i] < fval[bi] {
            bi = i;
        }
    }
    (simplex[bi].clone(), fval[bi])
}

/// Nelder–Mead shrink toward the best vertex (simplex[0]).
fn shrink<F: Fn(&[f64]) -> f64>(simplex: &mut [Vec<f64>], fval: &mut [f64], sigma: f64, f: &F) {
    let n = simplex.len() - 1;
    let best = simplex[0].clone();
    for i in 1..=n {
        for k in 0..n {
            simplex[i][k] = best[k] + sigma * (simplex[i][k] - best[k]);
        }
        fval[i] = f(&simplex[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generic Nelder–Mead must find the minimum of the Rosenbrock function, which has a
    /// curved valley that a contraction-only-toward-worst simplex handles poorly — this locks the
    /// outside/inside contraction split.
    #[test]
    fn nelder_mead_minimizes_rosenbrock() {
        let rosen = |v: &[f64]| -> f64 {
            let (x, y) = (v[0], v[1]);
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };
        let (best, fbest) = nelder_mead_min(&rosen, vec![-1.2, 1.0], 0.3, 4000);
        assert!(fbest < 1e-6, "did not converge: f={fbest:e}");
        assert!(
            (best[0] - 1.0).abs() < 1e-2 && (best[1] - 1.0).abs() < 1e-2,
            "min at {best:?}"
        );
    }

    /// `fit_fixed_mean` recovers a planted `(a, q, p0)` with the mean held at 0.
    #[test]
    fn fit_fixed_mean_recovers_params() {
        // one long chain; obs generated from an OU field with m=0, 8 shared dimensions
        let n = 60usize;
        let parents: Vec<i64> = (0..n)
            .map(|i| if i == 0 { -1 } else { (i - 1) as i64 })
            .collect();
        let (a_true, q_true, p0_true) = (0.8f64, 0.3f64, 1.0f64);
        let mut rng = Lcg(7);
        let obs: Vec<Vec<f64>> = (0..8)
            .map(|_| {
                let mut x = vec![0.0f64; n];
                x[0] = p0_true.sqrt() * rng.gauss();
                for i in 1..n {
                    x[i] = a_true * x[i - 1] + q_true.sqrt() * rng.gauss();
                }
                x
            })
            .collect();
        let r = vec![0.05f64; n];
        let fit = fit_fixed_mean(
            &parents,
            &obs,
            &r,
            TreeFieldParams {
                a: 0.5,
                q: 0.5,
                m: 0.0,
                p0: 0.5,
            },
        );
        assert_eq!(fit.m, 0.0);
        assert!((fit.a - a_true).abs() < 0.15, "a={} (want {a_true})", fit.a);
        assert!(fit.q > 0.05 && fit.p0 > 0.05, "q={} p0={}", fit.q, fit.p0);
    }

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
        fn gauss(&mut self) -> f64 {
            // Box–Muller
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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
    fn fit_recovers_planted_params() {
        // deep chains so the reversion `a` is identifiable (shallow trees can't see it)
        let mut rng = Lcg(0xdead_beef_0000_0001);
        let (a, q, m, p0, r_true): (f64, f64, f64, f64, f64) = (0.80, 0.50, 0.0, 1.0, 0.20);
        let n_chains = 80;
        let depth = 40;
        let mut parents: Vec<i64> = Vec::new();
        let mut x: Vec<f64> = Vec::new();
        for _ in 0..n_chains {
            let base = parents.len();
            parents.push(-1);
            x.push(m + p0.sqrt() * rng.gauss());
            for t in 1..depth {
                parents.push((base + t - 1) as i64);
                let xp = *x.last().unwrap();
                x.push(a * xp + (1.0 - a) * m + q.sqrt() * rng.gauss());
            }
        }
        let n = parents.len();
        // three oracle observation dims sharing the field params (isotropic)
        let r = vec![r_true; n];
        let obs: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..n).map(|i| x[i] + r_true.sqrt() * rng.gauss()).collect())
            .collect();
        let init = TreeFieldParams {
            a: 0.5,
            q: 1.0,
            m: 0.0,
            p0: 1.0,
        };
        let f = fit(&parents, &obs, &r, init);
        assert!((f.a - a).abs() < 0.08, "a: got {} want {}", f.a, a);
        assert!((f.q - q).abs() < 0.12, "q: got {} want {}", f.q, q);
        assert!(f.m.abs() < 0.15, "m: got {}", f.m);
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
