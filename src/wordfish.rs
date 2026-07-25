//! Wordfish: the Slapin & Proksch (2008) Poisson scaling model. A pure
//! word-frequency ideal-point estimator with no topics and no embeddings — the
//! baseline companion to [`crate::idealpoint`] (word embeddings + topics).
//!
//! Model: the count of word `j` by author `i` is
//! `y_ij ~ Poisson(exp(alpha_i + psi_j + beta_j * theta_i))`, where `theta_i` is
//! the author's latent position, `beta_j` the word's discrimination, `psi_j` a word
//! intercept and `alpha_i` an author "verbosity" effect (absorbs document length).
//!
//! Estimation is the standard Wordfish EM: alternate Newton updates of the
//! per-word parameters `(psi_j, beta_j)` (holding authors fixed) and the per-author
//! parameters `(alpha_i, theta_i)` (holding words fixed), with weak Gaussian priors
//! on `beta` and `theta` for regularization. Identification is exact and applied
//! every iteration: `theta` is standardized to mean 0 / unit variance (the scale is
//! absorbed into `beta`, the location into `psi`), `psi` is centered (absorbed into
//! `alpha`), and the sign is oriented to the anchors. The fit is deterministic — no
//! RNG, fixed-order reductions — so it is bit-reproducible.

/// A fitted Wordfish model. `theta` are the author positions (standardized);
/// `beta`/`psi` are the per-word discrimination / intercept; `alpha` the per-author
/// effect.
pub struct WordfishModel {
    pub num_authors: usize,
    pub num_types: usize,
    pub theta: Vec<f64>,
    pub alpha: Vec<f64>,
    pub psi: Vec<f64>,
    pub beta: Vec<f64>,
    pub log_likelihood: f64,
    pub ll_history: Vec<f64>,
    pub converged: bool,
    pub iters_run: usize,
    /// Prior sd on theta, retained so the asymptotic standard error can be computed.
    pub theta_prior_sd: f64,
    /// Per-control-level per-word log-rate offsets `delta[c][j]` (the control
    /// covariate). `delta[0]` is the baseline level and is held at zero. With no
    /// control covariate this is a single all-zero row, so the model reduces exactly
    /// to plain Wordfish.
    pub delta: Vec<Vec<f64>>,
    /// Each author's control level (index into `delta`); all zero with no control.
    pub level: Vec<usize>,
}

impl WordfishModel {
    /// Asymptotic standard error of each author position, from the observed
    /// information of the penalized Poisson log-likelihood. For author `i` the
    /// document-level 2x2 information in `(alpha_i, theta_i)` is
    /// `I_aa = sum_j mu_ij`, `I_at = sum_j mu_ij beta_j`,
    /// `I_tt = sum_j mu_ij beta_j^2 + 1/sd^2` (with `mu_ij = exp(alpha_i + psi_j +
    /// beta_j theta_i)`); the marginal variance of `theta_i`, accounting for the
    /// verbosity nuisance `alpha_i`, is `I_aa / (I_aa I_tt - I_at^2)`. This is a
    /// Hessian-based observed-information SE comparable to quanteda's `se.theta` —
    /// not guaranteed to equal it exactly: it profiles out `alpha_i` and folds in
    /// the theta-prior precision, which quanteda's `se.theta` need not do. The R
    /// parity harness (`parity/wordfish_r_compare.py`) reports the observed ratio.
    pub fn position_se(&self) -> Vec<f64> {
        let inv_var = if self.theta_prior_sd.is_finite() && self.theta_prior_sd > 0.0 {
            1.0 / (self.theta_prior_sd * self.theta_prior_sd)
        } else {
            0.0
        };
        (0..self.num_authors)
            .map(|i| {
                let ai = self.alpha[i];
                let ti = self.theta[i];
                let dl = &self.delta[self.level[i]];
                let (mut i_aa, mut i_at, mut i_tt) = (0.0, 0.0, inv_var);
                for j in 0..self.num_types {
                    let mu = (ai + self.psi[j] + self.beta[j] * ti + dl[j]).exp();
                    i_aa += mu;
                    i_at += mu * self.beta[j];
                    i_tt += mu * self.beta[j] * self.beta[j];
                }
                let det = i_aa * i_tt - i_at * i_at;
                if det > 0.0 && i_aa > 0.0 {
                    (i_aa / det).sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

/// Sparse author-major counts: `counts[i]` is the list of `(word_id, count)` for
/// author `i`. `num_types` is the vocabulary size; `anchors` is a list of
/// `(author_index, target)` used only to orient the sign of the axis.
///
/// Plain Wordfish (no control covariate). Equivalent to
/// [`fit_wordfish_controlled`] with a single control level, so the fit is
/// bit-identical to the historical implementation.
#[allow(clippy::too_many_arguments)]
pub fn fit_wordfish(
    counts: &[Vec<(u32, f64)>],
    num_types: usize,
    anchors: &[(usize, f64)],
    iters: usize,
    tol: f64,
    beta_prior_sd: f64,
    theta_prior_sd: f64,
) -> WordfishModel {
    let level = vec![0usize; counts.len()];
    fit_wordfish_controlled(
        counts,
        num_types,
        &level,
        1,
        anchors,
        iters,
        tol,
        beta_prior_sd,
        theta_prior_sd,
    )
}

/// Wordfish with a categorical **control covariate**. `level[i]` is the control
/// level of author `i` (an index in `0..num_levels`); level 0 is the baseline.
/// Each non-baseline level `c` gets a per-word log-rate offset `delta[c][j]`, so
/// the rate is `exp(alpha_i + psi_j + beta_j theta_i + delta[level_i][j])`. This
/// absorbs systematic, level-specific word usage (a chamber, a government/opposition
/// status, an era, a language) into `delta` instead of letting it contaminate the
/// latent position `theta`. With `num_levels == 1` the offsets are all zero and the
/// fit is exactly plain Wordfish.
#[allow(clippy::too_many_arguments)]
pub fn fit_wordfish_controlled(
    counts: &[Vec<(u32, f64)>],
    num_types: usize,
    level: &[usize],
    num_levels: usize,
    anchors: &[(usize, f64)],
    iters: usize,
    tol: f64,
    beta_prior_sd: f64,
    theta_prior_sd: f64,
) -> WordfishModel {
    let a = counts.len();
    let v = num_types;
    let inv_beta_var = if beta_prior_sd.is_finite() && beta_prior_sd > 0.0 {
        1.0 / (beta_prior_sd * beta_prior_sd)
    } else {
        0.0
    };
    let inv_theta_var = if theta_prior_sd.is_finite() && theta_prior_sd > 0.0 {
        1.0 / (theta_prior_sd * theta_prior_sd)
    } else {
        0.0
    };

    // Author totals and a dense view of the counts for the (dense over j) author
    // step and (dense over i) word step. Wordfish is inherently dense in the
    // expected counts (mu_ij > 0 everywhere), so we materialize a word-major copy.
    let mut author_total = vec![0.0f64; a];
    let mut word_total = vec![0.0f64; v];
    // word-major: by_word[j] = list of (author_i, count)
    let mut by_word: Vec<Vec<(u32, f64)>> = vec![Vec::new(); v];
    for (i, doc) in counts.iter().enumerate() {
        for &(w, c) in doc {
            author_total[i] += c;
            word_total[w as usize] += c;
            by_word[w as usize].push((i as u32, c));
        }
    }

    // Control covariate: authors grouped by level, and the level-major word totals
    // (y_{c,j}) that drive the per-(level, word) offset step. Empty work when
    // num_levels == 1 (the plain-Wordfish path), keeping delta all-zero.
    let mut level_authors: Vec<Vec<usize>> = vec![Vec::new(); num_levels];
    for (i, &c) in level.iter().enumerate() {
        level_authors[c].push(i);
    }
    let mut level_word_total: Vec<Vec<f64>> = vec![vec![0.0; v]; num_levels];
    for (i, doc) in counts.iter().enumerate() {
        let c = level[i];
        for &(w, cnt) in doc {
            level_word_total[c][w as usize] += cnt;
        }
    }
    // delta[c][j]: per-level per-word log-rate offset; level 0 is the held-out
    // baseline and stays zero.
    let mut delta: Vec<Vec<f64>> = vec![vec![0.0; v]; num_levels];

    // --- deterministic initialization ---------------------------------------
    // alpha_i = log author total; psi_j = log mean word rate; theta from the
    // leading principal axis of the row-normalized, column-centered log matrix.
    let eps = 1e-9;
    let mut alpha: Vec<f64> = author_total.iter().map(|&t| (t.max(1.0)).ln()).collect();
    let mut psi: Vec<f64> = word_total
        .iter()
        .map(|&t| (t.max(1.0) / a as f64).ln())
        .collect();
    let mut beta = vec![0.0f64; v];
    let mut theta = init_theta(counts, level, num_levels, a, v, eps);

    standardize_theta(&mut theta, &mut beta, &mut psi);
    center_psi(&mut psi, &mut alpha);

    let mut ll_history = Vec::with_capacity(iters);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iters_run = 0;

    for it in 0..iters {
        iters_run = it + 1;

        // M-step part 1: per-word (psi_j, beta_j) Newton step (dense over authors).
        for j in 0..v {
            let (mut g0, mut g1) = (0.0, 0.0); // grad psi, grad beta
            let (mut h00, mut h01, mut h11) = (0.0, 0.0, 0.0);
            // sufficient counts from the sparse column
            let mut y_sum = 0.0;
            let mut y_theta = 0.0;
            for &(i, c) in &by_word[j] {
                y_sum += c;
                y_theta += c * theta[i as usize];
            }
            for i in 0..a {
                let mu = (alpha[i] + psi[j] + beta[j] * theta[i] + delta[level[i]][j]).exp();
                g0 -= mu;
                g1 -= mu * theta[i];
                h00 -= mu;
                h01 -= mu * theta[i];
                h11 -= mu * theta[i] * theta[i];
            }
            g0 += y_sum;
            g1 += y_theta - inv_beta_var * beta[j];
            h11 -= inv_beta_var;
            let (dp, db) = solve2(h00, h01, h01, h11, g0, g1);
            psi[j] -= dp;
            beta[j] -= db;
        }

        // M-step part 2: per-author (alpha_i, theta_i) Newton step (dense over words).
        for i in 0..a {
            let (mut g0, mut g1) = (0.0, 0.0); // grad alpha, grad theta
            let (mut h00, mut h01, mut h11) = (0.0, 0.0, 0.0);
            let mut y_sum = 0.0;
            let mut y_beta = 0.0;
            for &(w, c) in &counts[i] {
                y_sum += c;
                y_beta += c * beta[w as usize];
            }
            let dl = &delta[level[i]];
            for j in 0..v {
                let mu = (alpha[i] + psi[j] + beta[j] * theta[i] + dl[j]).exp();
                g0 -= mu;
                g1 -= mu * beta[j];
                h00 -= mu;
                h01 -= mu * beta[j];
                h11 -= mu * beta[j] * beta[j];
            }
            g0 += y_sum;
            g1 += y_beta - inv_theta_var * theta[i];
            h11 -= inv_theta_var;
            let (da, dt) = solve2(h00, h01, h01, h11, g0, g1);
            alpha[i] -= da;
            theta[i] -= dt;
        }

        // M-step part 3: per-(level, word) control offset delta[c][j], a 1-D Newton
        // step over the authors in level c. Level 0 is the baseline (delta == 0), so
        // this loop is empty for plain Wordfish (num_levels == 1).
        for c in 1..num_levels {
            for j in 0..v {
                let mut mu_sum = 0.0;
                for &i in &level_authors[c] {
                    mu_sum += (alpha[i] + psi[j] + beta[j] * theta[i] + delta[c][j]).exp();
                }
                let grad = level_word_total[c][j] - mu_sum; // d loglik / d delta
                let hess = -mu_sum - 1e-9; // observed information (negative), ridged
                let step = (grad / hess).clamp(-5.0, 5.0);
                delta[c][j] -= step;
            }
        }

        // Identification: standardize theta (lossless), center psi (lossless).
        standardize_theta(&mut theta, &mut beta, &mut psi);
        center_psi(&mut psi, &mut alpha);

        // Convergence on the Poisson log-likelihood.
        let ll = loglik(counts, &alpha, &psi, &beta, &theta, level, &delta, a, v);
        ll_history.push(ll);
        if prev_ll.is_finite() {
            let denom = prev_ll.abs().max(1.0);
            if (ll - prev_ll).abs() / denom < tol {
                converged = true;
                prev_ll = ll;
                break;
            }
        }
        prev_ll = ll;
    }

    orient(&mut theta, &mut beta, anchors);

    WordfishModel {
        num_authors: a,
        num_types: v,
        theta,
        alpha,
        psi,
        beta,
        log_likelihood: prev_ll,
        ll_history,
        converged,
        iters_run,
        theta_prior_sd,
        delta,
        level: level.to_vec(),
    }
}

/// Leading principal axis of the doubly-centered log-count matrix, via power
/// iteration on the author x author gram (no dense V x V). Deterministic.
///
/// Column centering is done **per control level**: subtracting `col_mean[level][j]`
/// removes the control axis from the starting position, so `theta` is not initialized
/// on a dominant, level-specific nuisance axis. With `num_levels == 1` this is the
/// plain global column centering, bit-identical to before.
fn init_theta(
    counts: &[Vec<(u32, f64)>],
    level: &[usize],
    num_levels: usize,
    a: usize,
    v: usize,
    eps: f64,
) -> Vec<f64> {
    // Doubly-centered residual on present entries: r_ij = log1p(y_ij) minus the row
    // mean, the (level-specific) column mean, plus the grand mean. Its leading
    // author-space axis is a robust, dependency-free starting position.
    let val = |c: f64| (c + 1.0).ln();
    let mut row_mean = vec![0.0f64; a];
    let mut col_mean = vec![vec![0.0f64; v]; num_levels];
    let mut col_n = vec![vec![0.0f64; v]; num_levels];
    let mut grand_mean = 0.0;
    let mut nnz = 0.0;
    for (i, doc) in counts.iter().enumerate() {
        let c = level[i];
        for &(w, cnt) in doc {
            let x = val(cnt);
            row_mean[i] += x;
            col_mean[c][w as usize] += x;
            col_n[c][w as usize] += 1.0;
            grand_mean += x;
            nnz += 1.0;
        }
    }
    for (i, rm) in row_mean.iter_mut().enumerate() {
        *rm /= counts[i].len().max(1) as f64;
    }
    for c in 0..num_levels {
        for j in 0..v {
            if col_n[c][j] > 0.0 {
                col_mean[c][j] /= col_n[c][j];
            }
        }
    }
    if nnz > 0.0 {
        grand_mean /= nnz;
    }
    // Sparse residual rows: r_i = { (j, val - row_mean_i - col_mean[level_i][j] + grand_mean) }.
    let resid: Vec<Vec<(u32, f64)>> = counts
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let c = level[i];
            doc.iter()
                .map(|&(w, cnt)| {
                    (
                        w,
                        val(cnt) - row_mean[i] - col_mean[c][w as usize] + grand_mean,
                    )
                })
                .collect()
        })
        .collect();
    // Power iteration on G = R R^T (A x A) implied implicitly: Gx = R (R^T x).
    let mut theta = vec![0.0f64; a];
    for (i, t) in theta.iter_mut().enumerate() {
        // deterministic, non-degenerate start
        *t = ((i % 7) as f64 - 3.0) + 0.123 * (i as f64 + 1.0).ln();
    }
    normalize(&mut theta, eps);
    for _ in 0..50 {
        // u[j] = sum_i resid_i[j] * theta_i
        let mut u = vec![0.0f64; v];
        for (i, row) in resid.iter().enumerate() {
            let ti = theta[i];
            for &(w, r) in row {
                u[w as usize] += r * ti;
            }
        }
        // new theta_i = sum_j resid_i[j] * u[j]
        let mut nt = vec![0.0f64; a];
        for (i, row) in resid.iter().enumerate() {
            let mut s = 0.0;
            for &(w, r) in row {
                s += r * u[w as usize];
            }
            nt[i] = s;
        }
        normalize(&mut nt, eps);
        theta = nt;
    }
    theta
}

fn normalize(x: &mut [f64], eps: f64) {
    let n: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(eps);
    for v in x.iter_mut() {
        *v /= n;
    }
}

/// Standardize theta to mean 0 / unit variance, absorbing the affine change into
/// beta and psi so `beta_j*theta_i + psi_j` is invariant (lossless).
fn standardize_theta(theta: &mut [f64], beta: &mut [f64], psi: &mut [f64]) {
    let n = theta.len() as f64;
    if n == 0.0 {
        return;
    }
    let mean = theta.iter().sum::<f64>() / n;
    let var = theta.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd <= 1e-12 {
        for t in theta.iter_mut() {
            *t -= mean;
        }
        return;
    }
    for t in theta.iter_mut() {
        *t = (*t - mean) / sd;
    }
    // theta_old = sd*theta_new + mean => beta*theta_old + psi
    //           = (beta*sd)*theta_new + (psi + beta*mean)
    for (b, p) in beta.iter_mut().zip(psi.iter_mut()) {
        *p += *b * mean;
        *b *= sd;
    }
}

/// Center psi to mean 0, absorbing the shift into alpha (lossless: alpha_i + psi_j
/// is invariant). Resolves the additive non-identifiability between alpha and psi.
fn center_psi(psi: &mut [f64], alpha: &mut [f64]) {
    let n = psi.len() as f64;
    if n == 0.0 {
        return;
    }
    let mean = psi.iter().sum::<f64>() / n;
    for p in psi.iter_mut() {
        *p -= mean;
    }
    for x in alpha.iter_mut() {
        *x += mean;
    }
}

/// Orient the sign of the axis so it aligns with the anchors (positive correlation
/// between anchored positions and their targets). No-op without anchors.
fn orient(theta: &mut [f64], beta: &mut [f64], anchors: &[(usize, f64)]) {
    if anchors.len() < 2 {
        // a single anchor: make its sign match its target's sign
        if let Some(&(i, target)) = anchors.first() {
            if theta[i] * target < 0.0 {
                flip(theta, beta);
            }
        }
        return;
    }
    // covariance between theta[anchor] and target across anchors
    let n = anchors.len() as f64;
    let mt = anchors.iter().map(|&(i, _)| theta[i]).sum::<f64>() / n;
    let mv = anchors.iter().map(|&(_, t)| t).sum::<f64>() / n;
    let cov: f64 = anchors
        .iter()
        .map(|&(i, t)| (theta[i] - mt) * (t - mv))
        .sum();
    if cov < 0.0 {
        flip(theta, beta);
    }
}

fn flip(theta: &mut [f64], beta: &mut [f64]) {
    for t in theta.iter_mut() {
        *t = -*t;
    }
    for b in beta.iter_mut() {
        *b = -*b;
    }
}

/// Solve the 2x2 system H d = g (H = [[h00,h01],[h10,h11]]) for d, with a small
/// ridge if near-singular. Returns the Newton step (g is the gradient, H the
/// Hessian; the caller subtracts d).
fn solve2(h00: f64, h01: f64, h10: f64, h11: f64, g0: f64, g1: f64) -> (f64, f64) {
    let mut det = h00 * h11 - h01 * h10;
    let (mut a00, mut a11) = (h00, h11);
    if det.abs() < 1e-12 {
        // ridge toward a gradient step
        a00 -= 1e-6;
        a11 -= 1e-6;
        det = a00 * a11 - h01 * h10;
        if det.abs() < 1e-12 {
            return (0.0, 0.0);
        }
    }
    let d0 = (a11 * g0 - h01 * g1) / det;
    let d1 = (a00 * g1 - h10 * g0) / det;
    // clamp the step for numerical safety on early, ill-conditioned iterations
    let clamp = |x: f64| x.clamp(-5.0, 5.0);
    (clamp(d0), clamp(d1))
}

#[allow(clippy::too_many_arguments)]
fn loglik(
    counts: &[Vec<(u32, f64)>],
    alpha: &[f64],
    psi: &[f64],
    beta: &[f64],
    theta: &[f64],
    level: &[usize],
    delta: &[Vec<f64>],
    a: usize,
    v: usize,
) -> f64 {
    // sum_ij [ y_ij * eta_ij - exp(eta_ij) ], eta = alpha + psi + beta*theta +
    // delta[level_i][j]. The -exp term is dense; the y*eta term is sparse.
    let mut ll = 0.0;
    for i in 0..a {
        let ai = alpha[i];
        let ti = theta[i];
        let dl = &delta[level[i]];
        for j in 0..v {
            ll -= (ai + psi[j] + beta[j] * ti + dl[j]).exp();
        }
        for &(w, c) in &counts[i] {
            let j = w as usize;
            ll += c * (ai + psi[j] + beta[j] * ti + dl[j]);
        }
    }
    ll
}

impl WordfishModel {
    /// Author positions as a (num_authors, 1) matrix, for parity with the other
    /// ideal-point models' `author_positions`.
    pub fn positions(&self) -> Vec<Vec<f64>> {
        self.theta.iter().map(|&t| vec![t]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn poisson(rng: &mut ChaCha8Rng, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        let l = (-lambda).exp();
        let mut k = 0u32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= rng.gen::<f64>();
            if p <= l {
                break;
            }
        }
        (k - 1) as f64
    }

    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for (&a, &b) in x.iter().zip(y) {
            cov += (a - mx) * (b - my);
            vx += (a - mx) * (a - mx);
            vy += (b - my) * (b - my);
        }
        cov / (vx.sqrt() * vy.sqrt())
    }

    #[test]
    fn fit_wordfish_recovers_positions() {
        // Sample counts from the Wordfish model with planted positions and word
        // discriminations, then check the fit recovers both.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (a, v) = (60usize, 80usize);
        let theta_true: Vec<f64> = (0..a)
            .map(|i| (i as f64 / (a as f64 - 1.0)) * 2.0 - 1.0)
            .collect();
        let beta_true: Vec<f64> = (0..v)
            .map(|j| ((j as f64 / (v as f64 - 1.0)) * 2.0 - 1.0) * 1.0)
            .collect();
        // word base rates 2..10
        let psi: Vec<f64> = (0..v)
            .map(|j| (2.0 + 8.0 * (j as f64 / (v as f64 - 1.0))).ln())
            .collect();
        let mut counts: Vec<Vec<(u32, f64)>> = Vec::with_capacity(a);
        for i in 0..a {
            let mut row = Vec::new();
            for j in 0..v {
                let lam = (psi[j] + beta_true[j] * theta_true[i]).exp();
                let c = poisson(&mut rng, lam);
                if c > 0.0 {
                    row.push((j as u32, c));
                }
            }
            counts.push(row);
        }
        let anchors = vec![(0usize, -1.0), (a - 1, 1.0)];
        let m = fit_wordfish(&counts, v, &anchors, 100, 1e-8, 3.0, 1.0);

        let r = pearson(&m.theta, &theta_true).abs();
        assert!(r > 0.9, "theta recovery r={r}");
        let rb = pearson(&m.beta, &beta_true).abs();
        assert!(rb > 0.8, "beta recovery r={rb}");
        // positions are standardized
        let mean = m.theta.iter().sum::<f64>() / a as f64;
        assert!(mean.abs() < 1e-6, "theta not centered: {mean}");
        // anchors orient the sign: low anchor negative, high anchor positive
        assert!(m.theta[0] < m.theta[a - 1], "anchors did not orient");
    }

    #[test]
    fn control_covariate_rescues_axis_under_contamination() {
        // A corpus with a weak ideology signal plus a DOMINANT, control-aligned
        // nuisance axis. Without the control, theta is hijacked by the nuisance;
        // with it, theta recovers ideology.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let a = 50usize;
        // vocab: 0..8 ideology-left, 8..16 ideology-right, 16..24 nuisance-level0,
        // 24..32 nuisance-level1, 32..62 filler
        let v = 62usize;
        let ideo: Vec<f64> = (0..a)
            .map(|i| (i as f64 / (a as f64 - 1.0)) * 2.0 - 1.0)
            .collect();
        let level: Vec<usize> = (0..a).map(|i| i % 2).collect();
        let mut counts: Vec<Vec<(u32, f64)>> = Vec::with_capacity(a);
        for i in 0..a {
            let mut row = vec![0.0f64; v];
            let pr_right = (ideo[i] + 1.0) / 2.0;
            for _ in 0..560 {
                let u: f64 = rng.gen();
                let w = if u < 0.45 {
                    // dominant nuisance, determined by the control level
                    let base = if level[i] == 1 { 24 } else { 16 };
                    base + (rng.gen::<usize>() % 8)
                } else if u < 0.85 {
                    if rng.gen::<f64>() < pr_right {
                        8 + (rng.gen::<usize>() % 8)
                    } else {
                        rng.gen::<usize>() % 8
                    }
                } else {
                    32 + (rng.gen::<usize>() % 30)
                };
                row[w] += 1.0;
            }
            counts.push(
                row.iter()
                    .enumerate()
                    .filter(|&(_, &c)| c > 0.0)
                    .map(|(j, &c)| (j as u32, c))
                    .collect(),
            );
        }
        let no_control = fit_wordfish(&counts, v, &[], 120, 1e-8, 3.0, 1.0);
        let with_control = fit_wordfish_controlled(&counts, v, &level, 2, &[], 120, 1e-8, 3.0, 1.0);
        let r_plain = pearson(&no_control.theta, &ideo).abs();
        let r_ctrl = pearson(&with_control.theta, &ideo).abs();
        assert!(
            r_plain < 0.4,
            "expected the nuisance to hijack plain Wordfish, got ideology r={r_plain}"
        );
        assert!(
            r_ctrl > 0.8,
            "control covariate should recover ideology, got r={r_ctrl}"
        );
    }

    #[test]
    fn single_level_control_is_plain_wordfish() {
        // num_levels == 1 must reproduce plain Wordfish bit-for-bit.
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (a, v) = (30usize, 40usize);
        let mut counts = Vec::new();
        for _ in 0..a {
            let mut row = Vec::new();
            for j in 0..v {
                let c = poisson(&mut rng, 3.0);
                if c > 0.0 {
                    row.push((j as u32, c));
                }
            }
            counts.push(row);
        }
        let plain = fit_wordfish(&counts, v, &[], 50, 1e-8, 3.0, 1.0);
        let level = vec![0usize; a];
        let ctrl = fit_wordfish_controlled(&counts, v, &level, 1, &[], 50, 1e-8, 3.0, 1.0);
        assert_eq!(plain.theta, ctrl.theta);
        assert_eq!(plain.beta, ctrl.beta);
        assert_eq!(plain.psi, ctrl.psi);
    }
}
