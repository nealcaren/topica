//! Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024, *Management
//! Science*). STS extends STM/CTM with a per-document, per-topic *continuous*
//! sentiment-discourse latent `α^(s)` that modulates the topic-word distribution.
//!
//! This module ports the variational E-step from the authors' reference R/`sts`
//! implementation (`opt.alpha.R`, `estimateHessian.R`). The per-document latent is
//! the `2K-1` vector `α = [α^(p)_{1..K-1}, α^(s)_{1..K}]`: prevalence uses `K-1`
//! dimensions (the last topic is the logistic-normal reference category, exactly
//! as in CTM), and sentiment uses the full `K`. The prevalence sub-block of the
//! objective/gradient/Hessian is identical to [`crate::ctm`]; STS adds the
//! sentiment block and the dependence of `β` on `α^(s)`.
//!
//! The E-step maximizes the per-document log-posterior `f(α)` by quasi-Newton, then
//! takes the Laplace covariance as the inverse of `-∇²f` at the optimum.
//!
//! Generative form (Chen & Mankad eq. 2): for topic `k`,
//! `β_{k,v} ∝ exp(m_v + κ^(t)_{k,v} + κ^(s)_{k,v}·α^(s)_k)`.

/// Per-topic topic-word coefficients. `kappa_t[k]` and `kappa_s[k]` are each a
/// length-`V` vector (baseline and sentiment-discourse loadings for topic `k`).
pub struct Kappa {
    pub kappa_t: Vec<Vec<f64>>, // K × V
    pub kappa_s: Vec<Vec<f64>>, // K × V
}

/// `β_{k,·}` for one topic: softmax over the vocabulary of
/// `m_v + κ^(t)_{k,v} + κ^(s)_{k,v}·α^(s)_k`.
fn topic_beta(mv: &[f64], kt: &[f64], ks: &[f64], alpha_s: f64) -> Vec<f64> {
    let v = mv.len();
    let mut lin = vec![0.0f64; v];
    let mut mx = f64::NEG_INFINITY;
    for i in 0..v {
        lin[i] = mv[i] + kt[i] + ks[i] * alpha_s;
        if lin[i] > mx {
            mx = lin[i];
        }
    }
    let mut s = 0.0;
    for x in lin.iter_mut() {
        *x = (*x - mx).exp();
        s += *x;
    }
    for x in lin.iter_mut() {
        *x /= s;
    }
    lin
}

/// `exp([α^(p), 0])` — the K unnormalized topic weights (last topic is the
/// reference, weight `exp(0)=1`).
fn expeta(alpha_p: &[f64]) -> Vec<f64> {
    let mut e = Vec::with_capacity(alpha_p.len() + 1);
    for &x in alpha_p {
        e.push(x.exp());
    }
    e.push(1.0);
    e
}

/// Shared per-document quantities derived from `α`: the K topic betas, the
/// unnormalized weights `expeta`, and `θ = expeta/Σexpeta`.
struct DocBeta {
    beta: Vec<Vec<f64>>, // K × V
    expeta: Vec<f64>,    // K
    theta: Vec<f64>,     // K
}

fn doc_beta(alpha: &[f64], kappa: &Kappa, mv: &[f64], k: usize) -> DocBeta {
    let expeta = expeta(&alpha[..k - 1]);
    let sum_e: f64 = expeta.iter().sum();
    let theta: Vec<f64> = expeta.iter().map(|&e| e / sum_e).collect();
    let mut beta = Vec::with_capacity(k);
    for t in 0..k {
        beta.push(topic_beta(
            mv,
            &kappa.kappa_t[t],
            &kappa.kappa_s[t],
            alpha[k - 1 + t],
        ));
    }
    DocBeta {
        beta,
        expeta,
        theta,
    }
}

/// Per-document log-posterior `f(α)` (to MAXIMIZE), `opt.alpha.R::log_posterior_byDoc`.
///
/// `alpha` is length `2K-1`. `words`/`counts` are the document's word indices and
/// counts. `mu`/`siginv` are the variational prior mean and precision (length
/// `2K-1` and `(2K-1)²`).
pub fn sts_lhood(
    alpha: &[f64],
    kappa: &Kappa,
    mv: &[f64],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
    k: usize,
) -> f64 {
    let n = 2 * k - 1;
    let db = doc_beta(alpha, kappa, mv, k);
    let sum_e: f64 = db.expeta.iter().sum();
    let ndoc: f64 = counts.iter().sum();

    // term2: multinomial log-likelihood of the words under the mixture θ·β.
    let mut term2 = 0.0;
    for (wi, &w) in words.iter().enumerate() {
        let mut s = 0.0;
        for t in 0..k {
            s += db.expeta[t] * db.beta[t][w];
        }
        term2 += counts[wi] * s.ln();
    }
    term2 -= ndoc * sum_e.ln();

    // term1: logistic-normal prior, -0.5 (α-μ)ᵀ Σ⁻¹ (α-μ).
    let mut quad = 0.0;
    for i in 0..n {
        let di = alpha[i] - mu[i];
        for j in 0..n {
            quad += di * siginv[i * n + j] * (alpha[j] - mu[j]);
        }
    }
    -0.5 * quad + term2
}

/// Responsibilities `beta_bar[wi][t] = expeta_t β_{t,w} / Σ_t expeta_t β_{t,w}`
/// for each document word, plus the column sums `Σ_v κ^(s)_{k,v} β_{k,v}` used by
/// the gradient and Hessian.
struct DocResp {
    beta_bar: Vec<Vec<f64>>,  // ntok × K
    kappa_bar: Vec<Vec<f64>>, // K × V  (κ^(s) minus its β-weighted mean)
    ks_beta_mean: Vec<f64>,   // K
}

fn doc_resp(db: &DocBeta, kappa: &Kappa, words: &[usize], k: usize, v: usize) -> DocResp {
    let mut beta_bar = Vec::with_capacity(words.len());
    for &w in words {
        let mut row = vec![0.0f64; k];
        let mut denom = 0.0;
        for t in 0..k {
            row[t] = db.expeta[t] * db.beta[t][w];
            denom += row[t];
        }
        for t in 0..k {
            row[t] /= denom;
        }
        beta_bar.push(row);
    }

    let mut ks_beta_mean = vec![0.0f64; k];
    let mut kappa_bar = Vec::with_capacity(k);
    for t in 0..k {
        let mut m = 0.0;
        for i in 0..v {
            m += kappa.kappa_s[t][i] * db.beta[t][i];
        }
        ks_beta_mean[t] = m;
        kappa_bar.push(kappa.kappa_s[t].iter().map(|&x| x - m).collect());
    }
    DocResp {
        beta_bar,
        kappa_bar,
        ks_beta_mean,
    }
}

/// Gradient of [`sts_lhood`] w.r.t. `α` (length `2K-1`),
/// `opt.alpha.R::lapl_grad_alpha_eta`.
pub fn sts_grad(
    alpha: &[f64],
    kappa: &Kappa,
    mv: &[f64],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
    k: usize,
) -> Vec<f64> {
    let n = 2 * k - 1;
    let v = mv.len();
    let db = doc_beta(alpha, kappa, mv, k);
    let dr = doc_resp(&db, kappa, words, k, v);
    let ndoc: f64 = counts.iter().sum();

    // g1 (prevalence): expected topic counts minus N_d·θ, first K-1 entries.
    let mut g1 = vec![0.0f64; k];
    for (wi, _) in words.iter().enumerate() {
        for t in 0..k {
            g1[t] += counts[wi] * dr.beta_bar[wi][t];
        }
    }
    for t in 0..k {
        g1[t] -= ndoc * db.theta[t];
    }

    // g2 (sentiment): Σ_w c_w β̄_{w,k} (κ^(s)_{k,w} − mean_β κ^(s)_k).
    let mut g2 = vec![0.0f64; k];
    for (wi, &w) in words.iter().enumerate() {
        for t in 0..k {
            g2[t] += counts[wi] * dr.beta_bar[wi][t] * dr.kappa_bar[t][w];
        }
    }

    // grad f = [g1[0..K-1], g2] − Σ⁻¹(α−μ).
    let mut g = vec![0.0f64; n];
    for i in 0..(k - 1) {
        g[i] = g1[i];
    }
    for t in 0..k {
        g[k - 1 + t] = g2[t];
    }
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += siginv[i * n + j] * (alpha[j] - mu[j]);
        }
        g[i] -= s;
    }
    g
}

/// The precision matrix `-∇²f(α) = -H_data + Σ⁻¹` (row-major `(2K-1)²`),
/// `estimateHessian.R`. Positive-definite at the optimum; its inverse is the
/// Laplace (variational) covariance `ν_d`.
pub fn sts_precision(
    alpha: &[f64],
    kappa: &Kappa,
    mv: &[f64],
    words: &[usize],
    counts: &[f64],
    siginv: &[f64],
    k: usize,
) -> Vec<f64> {
    let n = 2 * k - 1;
    let v = mv.len();
    let db = doc_beta(alpha, kappa, mv, k);
    let dr = doc_resp(&db, kappa, words, k, v);
    let ndoc: f64 = counts.iter().sum();

    // g1, g2 again (the diagonal terms of the data Hessian blocks).
    let mut g1 = vec![0.0f64; k];
    let mut g2 = vec![0.0f64; k];
    for (wi, &w) in words.iter().enumerate() {
        for t in 0..k {
            g1[t] += counts[wi] * dr.beta_bar[wi][t];
            g2[t] += counts[wi] * dr.beta_bar[wi][t] * dr.kappa_bar[t][w];
        }
    }
    for t in 0..k {
        g1[t] -= ndoc * db.theta[t];
    }

    // h_pp = diag(g1) − Σ_w c_w β̄_a β̄_b + N_d θθᵀ          (K×K)
    // h_ps = diag(g2) − Σ_w c_w κ̄_a β̄_a β̄_b                (K×K, rows=sentiment)
    // h_ss = diag(S_a) − Σ_w c_w κ̄_a β̄_a κ̄_b β̄_b           (K×K)
    let mut h_pp = vec![0.0f64; k * k];
    let mut h_ps = vec![0.0f64; k * k];
    let mut h_ss = vec![0.0f64; k * k];

    for (wi, &w) in words.iter().enumerate() {
        let c = counts[wi];
        for a in 0..k {
            let ba = dr.beta_bar[wi][a];
            let kba = dr.kappa_bar[a][w];
            for b in 0..k {
                let bb = dr.beta_bar[wi][b];
                h_pp[a * k + b] -= c * ba * bb;
                h_ps[a * k + b] -= c * kba * ba * bb;
                h_ss[a * k + b] -= c * kba * ba * dr.kappa_bar[b][w] * bb;
            }
        }
    }
    for a in 0..k {
        for b in 0..k {
            h_pp[a * k + b] += ndoc * db.theta[a] * db.theta[b];
        }
        h_pp[a * k + a] += g1[a];
        h_ps[a * k + a] += g2[a];
    }
    // h_ss diagonal: S_a = Σ_w c_w β̄_a (κ̄_a² − Σ_v κ̄_{a,v} κ^(s)_{a,v} β_{a,v}).
    for a in 0..k {
        let mut kbks = 0.0; // Σ_v κ̄_{a,v} κ^(s)_{a,v} β_{a,v}
        for i in 0..v {
            kbks += dr.kappa_bar[a][i] * kappa.kappa_s[a][i] * db.beta[a][i];
        }
        let mut s_a = 0.0;
        for (wi, &w) in words.iter().enumerate() {
            s_a +=
                counts[wi] * dr.beta_bar[wi][a] * (dr.kappa_bar[a][w] * dr.kappa_bar[a][w] - kbks);
        }
        h_ss[a * k + a] += s_a;
    }
    let _ = dr.ks_beta_mean; // (already folded into kappa_bar)

    // Assemble the (2K-1) data Hessian H, then return -H + Σ⁻¹.
    // Prevalence indices 0..K-1; sentiment indices (K-1)..(2K-2).
    let mut precision = vec![0.0f64; n * n];
    for i in 0..(k - 1) {
        for j in 0..(k - 1) {
            precision[i * n + j] = -h_pp[i * k + j];
        }
    }
    for a in 0..k {
        for b in 0..(k - 1) {
            // sentiment row a, prevalence col b
            precision[(k - 1 + a) * n + b] = -h_ps[a * k + b];
            // symmetric transpose
            precision[b * n + (k - 1 + a)] = -h_ps[a * k + b];
        }
    }
    for a in 0..k {
        for b in 0..k {
            precision[(k - 1 + a) * n + (k - 1 + b)] = -h_ss[a * k + b];
        }
    }
    for idx in 0..(n * n) {
        precision[idx] += siginv[idx];
    }
    precision
}

// ---------------------------------------------------------------------------
// Fitted model + EM driver (PR1: κ held fixed; the Poisson κ M-step is PR2)
// ---------------------------------------------------------------------------

use crate::estimator::{Estimator, ModelFamily};
use crate::linalg::{
    cholesky, half_logdet, make_diagonally_dominant, spd_inverse, spd_inverse_from_chol,
};
use crate::variational::LogisticNormalModel;
use crate::variational::{doc_sparse, fit_gamma_ridge, lbfgs_minimize};
use rand::Rng;

/// A fitted STS model. With the E-step done and `κ` held fixed, this carries the
/// per-document latent prevalence/sentiment, the prior regression `Γ` / covariance
/// `Σ`, and the (fixed) topic-word coefficients `κ`.
pub struct StsModel {
    pub k: usize,
    pub num_types: usize,
    pub alpha: Vec<Vec<f64>>, // D × (2K-1): [α^(p)_{1..K-1}, α^(s)_{1..K}]
    /// Per-document variational covariance ν = H⁻¹ ((2K-1)² flattened, row-major).
    /// Empty when the model was fit with `keep_nu = false`; use `recompute_nu_sts`
    /// to regenerate on demand.
    pub nu: Vec<Vec<f64>>, // D × (2K-1)²: Laplace covariance per doc
    pub gamma: Option<Vec<Vec<f64>>>, // F × (2K-1): prevalence+sentiment regression
    pub sigma: Vec<f64>,      // (2K-1)²
    /// The prior covariance Σ that was used in the final E-step (sigma before the
    /// final M-step). Used by `recompute_nu_sts` to exactly reproduce the stored ν.
    pub sigma_estep: Vec<f64>,
    pub kappa_t: Vec<Vec<f64>>, // K × V (final, after last M-step)
    pub kappa_s: Vec<Vec<f64>>, // K × V (final, after last M-step)
    /// The topic-word coefficients κ_t, κ_s that were used in the final E-step
    /// (before the κ M-step updated `kappa_t`/`kappa_s`). Used by `recompute_nu_sts`
    /// to reproduce ν exactly.
    pub kappa_t_estep: Vec<Vec<f64>>,
    pub kappa_s_estep: Vec<Vec<f64>>,
    pub mv: Vec<f64>,        // V
    pub beta: Vec<Vec<f64>>, // K × V baseline topic-word at α^(s)=0
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
}

impl StsModel {
    /// Per-document topic prevalence `θ = softmax([α^(p), 0])` (length K).
    pub fn doc_topics(&self) -> Vec<Vec<f64>> {
        let k = self.k;
        self.alpha
            .iter()
            .map(|a| {
                let e = expeta(&a[..k - 1]);
                let s: f64 = e.iter().sum();
                e.iter().map(|x| x / s).collect()
            })
            .collect()
    }

    /// Per-document topic sentiment-discourse `α^(s)` (length K).
    pub fn doc_sentiment(&self) -> Vec<Vec<f64>> {
        let k = self.k;
        self.alpha.iter().map(|a| a[k - 1..].to_vec()).collect()
    }
}

impl Estimator for StsModel {
    fn num_topics(&self) -> usize {
        self.k
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.beta.clone()
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topics()
    }
    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect()
    }
    fn converged(&self) -> Option<bool> {
        Some(self.converged)
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::LogisticNormal
    }
}

impl LogisticNormalModel for StsModel {
    fn eta_dim(&self) -> usize {
        2 * self.k - 1
    }
    fn eta_mean(&self) -> &[Vec<f64>] {
        &self.alpha
    }
    /// Returns the stored per-document covariances. Returns `&[]` when the model
    /// was fit with `keep_nu = false` (use `recompute_nu_sts` to regenerate).
    fn eta_cov(&self) -> &[Vec<f64>] {
        &self.nu
    }
}

/// Infer per-document prevalence θ (D×K) for *new* documents against fixed
/// global parameters (κ_t, κ_s, m, Σ) by the same Laplace E-step used in fitting,
/// with a zero prior mean (held-out documents carry no covariates). For each
/// document the 2K-1 latent α = [α^(p), α^(s)] is found by the L-BFGS optimum of
/// the per-document objective, then θ_d = softmax([α^(p)_d, 0]); the sentiment
/// latent is inferred jointly but not returned. Empty documents (all tokens out
/// of vocabulary) get a uniform θ. Document order is preserved, and the parallel
/// inference is order-deterministic (one independent optimum per document).
pub fn sts_infer(
    docs: &[Vec<u32>],
    kappa_t: &[Vec<f64>],
    kappa_s: &[Vec<f64>],
    mv: &[f64],
    sigma: &[f64],
    k: usize,
) -> Vec<Vec<f64>> {
    let n = 2 * k - 1;
    let kappa = Kappa {
        kappa_t: kappa_t.to_vec(),
        kappa_s: kappa_s.to_vec(),
    };
    let siginv = spd_inverse(sigma, n).unwrap_or_else(|| {
        let mut s = sigma.to_vec();
        make_diagonally_dominant(&mut s, n);
        spd_inverse(&s, n).unwrap()
    });
    let mu = vec![0.0f64; n];
    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|d| doc_sparse(d)).collect();
    let inferred = crate::variational::laplace_estep(&sparse, |_di, words, counts| {
        let a_hat = lbfgs_minimize(
            vec![0.0f64; n],
            |a| {
                (
                    -sts_lhood(a, &kappa, mv, words, counts, &mu, &siginv, k),
                    sts_grad(a, &kappa, mv, words, counts, &mu, &siginv, k)
                        .iter()
                        .map(|g| -g)
                        .collect(),
                )
            },
            100,
            7,
            1e-5,
        );
        let e = expeta(&a_hat[..k - 1]);
        let s: f64 = e.iter().sum();
        e.iter().map(|x| x / s).collect::<Vec<f64>>()
    });
    // Reassemble in document order; empty docs (skipped by the driver) get uniform θ.
    let mut out = vec![vec![1.0 / k as f64; k]; docs.len()];
    for (di, theta) in inferred {
        out[di] = theta;
    }
    out
}

/// A two-parameter Poisson regression `log E[y_g] = offset_g + a + b·z_g` fit by
/// Newton's method with a small ridge. Returns `(a, b)` = `(κ^(t), κ^(s))` for one
/// (word, topic). When the covariate `z` has no spread (or fewer than two groups)
/// the slope is unidentified, so only the intercept is fit and `b = 0`.
fn poisson_2param(y: &[f64], z: &[f64], offset: &[f64], ridge: f64) -> (f64, f64) {
    let g = y.len();
    let zmin = z.iter().cloned().fold(f64::INFINITY, f64::min);
    let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let fit_slope = g >= 2 && (zmax - zmin).abs() > 1e-9;

    let (mut a, mut b) = (0.0f64, 0.0f64);
    for _ in 0..100 {
        let (mut g0, mut g1) = (0.0f64, 0.0f64);
        let (mut h00, mut h01, mut h11) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..g {
            let eta = (offset[i] + a + b * z[i]).clamp(-30.0, 30.0);
            let mu = eta.exp();
            let r = y[i] - mu;
            g0 += r;
            h00 += mu;
            if fit_slope {
                g1 += r * z[i];
                h01 += mu * z[i];
                h11 += mu * z[i] * z[i];
            }
        }
        g0 -= ridge * a;
        h00 += ridge;
        if !fit_slope {
            let step = g0 / h00.max(1e-12);
            a += step;
            if step.abs() < 1e-10 {
                break;
            }
            continue;
        }
        g1 -= ridge * b;
        h11 += ridge;
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-12 {
            break;
        }
        let da = (h11 * g0 - h01 * g1) / det;
        let db = (h00 * g1 - h01 * g0) / det;
        a += da;
        b += db;
        if da.abs() + db.abs() < 1e-10 {
            break;
        }
    }
    (a, b)
}

fn poisson_deviance(y: &[f64], mu: &[f64]) -> f64 {
    let mut d = 0.0;
    for i in 0..y.len() {
        let yi = y[i];
        let term = if yi > 0.0 {
            yi * (yi / mu[i]).ln()
        } else {
            0.0
        };
        d += term - (yi - mu[i]);
    }
    2.0 * d
}

fn soft_threshold(z: f64, g: f64) -> f64 {
    if z > g {
        z - g
    } else if z < -g {
        z + g
    } else {
        0.0
    }
}

/// L1-penalized (lasso) Poisson regression over a λ path with the penalty selected
/// by AIC — the glmnet path used by the reference `opt.kappa.R` (`family="poisson"`,
/// `alpha=1`, `intercept=FALSE`, `standardize=FALSE`). Fit by IRLS with inner
/// coordinate descent (Friedman, Hastie & Tibshirani 2010); warm-started down the
/// path. `x` is `n×p`, with a fixed `offset`. Returns the AIC-selected coefficients.
fn poisson_lasso(
    x: &[Vec<f64>],
    y: &[f64],
    offset: &[f64],
    nlambda: usize,
    lambda_min_ratio: f64,
) -> Vec<f64> {
    let n = x.len();
    let p = if n > 0 { x[0].len() } else { 0 };
    if p == 0 {
        return Vec::new();
    }

    // λ_max: the smallest λ that zeros every coefficient (gradient at β=0).
    let mu0: Vec<f64> = offset.iter().map(|&o| o.clamp(-30.0, 30.0).exp()).collect();
    let mut lam_max: f64 = 0.0;
    for j in 0..p {
        let g: f64 = (0..n).map(|i| x[i][j] * (y[i] - mu0[i])).sum();
        lam_max = lam_max.max(g.abs());
    }
    if lam_max <= 0.0 {
        return vec![0.0; p];
    }

    let mut beta = vec![0.0f64; p];
    let mut xbeta = vec![0.0f64; n];
    let mut best_beta = beta.clone();
    let mut best_aic = f64::INFINITY;

    for li in 0..nlambda {
        let frac = li as f64 / (nlambda.max(2) - 1) as f64;
        let lam = lam_max * lambda_min_ratio.powf(frac);

        // IRLS: weighted-LS coordinate descent around the current fit.
        for _ in 0..25 {
            let mut w = vec![0.0f64; n];
            let mut target = vec![0.0f64; n]; // working response for Xβ (offset removed)
            for i in 0..n {
                let mu = (offset[i] + xbeta[i]).clamp(-30.0, 30.0).exp();
                w[i] = mu.max(1e-6);
                target[i] = xbeta[i] + (y[i] - mu) / w[i];
            }
            let mut max_step = 0.0f64;
            for _ in 0..50 {
                let mut pass_step = 0.0f64;
                for j in 0..p {
                    let mut denom = 0.0;
                    let mut rho = 0.0;
                    for i in 0..n {
                        if x[i][j] == 0.0 {
                            continue;
                        }
                        denom += w[i] * x[i][j] * x[i][j];
                        let resid = target[i] - xbeta[i] + x[i][j] * beta[j];
                        rho += w[i] * x[i][j] * resid;
                    }
                    if denom < 1e-12 {
                        continue;
                    }
                    let new_bj = soft_threshold(rho, lam) / denom;
                    let delta = new_bj - beta[j];
                    if delta != 0.0 {
                        for i in 0..n {
                            xbeta[i] += x[i][j] * delta;
                        }
                        beta[j] = new_bj;
                        pass_step = pass_step.max(delta.abs());
                    }
                }
                if pass_step < 1e-7 {
                    break;
                }
                max_step = max_step.max(pass_step);
            }
            if max_step < 1e-6 {
                break;
            }
        }

        let mu: Vec<f64> = (0..n)
            .map(|i| (offset[i] + xbeta[i]).clamp(-30.0, 30.0).exp())
            .collect();
        let dev = poisson_deviance(y, &mu);
        let df = beta.iter().filter(|&&b| b != 0.0).count();
        let aic = dev + 2.0 * df as f64;
        if aic < best_aic {
            best_aic = aic;
            best_beta = beta.clone();
        }
    }
    best_beta
}

/// How the topic-word coefficients `κ` are estimated in the M-step.
#[derive(Clone, Copy)]
pub enum KappaEst {
    /// Ridge-penalized Poisson (Newton) per (word, topic). Stable and fast.
    Ridge(f64),
    /// L1 (lasso) Poisson over a λ path with AIC-selected penalty — the reference
    /// `opt.kappa.R` default (glmnet). Sparser κ; closer to the R `sts` solution.
    Lasso {
        nlambda: usize,
        lambda_min_ratio: f64,
    },
}

/// M-step for the topic-word coefficients `κ` (Chen & Mankad §4.2; `opt.kappa.R`).
///
/// Estimating `κ` is a multinomial logistic regression, recast via the
/// multinomial–Poisson equivalence (Taddy 2015) into independent Poisson
/// regressions per vocabulary word, with a document-fixed-effect offset. The
/// design is block-diagonal across topics: per word, topic `k` contributes an
/// intercept `κ^(t)_{k,v}` and a slope `κ^(s)_{k,v}` on the group-mean sentiment.
/// [`KappaEst::Ridge`] fits each topic's 2 parameters independently; the reference
/// default [`KappaEst::Lasso`] fits the joint `2K` design with a shared λ path and
/// global AIC, exactly as glmnet does.
///
/// `phi_by_group[g][v][k] = Σ_{d∈g} φ_{d,v,k}` are the aggregated expected
/// word-topic counts; `alpha_agg[g][k]` the group-mean `α^(s)`.
fn opt_kappa(
    phi_by_group: &[Vec<Vec<f64>>],
    alpha_agg: &[Vec<f64>],
    mv: &[f64],
    k: usize,
    v: usize,
    num_groups: usize,
    est: KappaEst,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Group-topic totals φ_{g,k} = Σ_v φ_{g,v,k} → the offset log φ_{g,k} + m_v.
    let mut phi_sum = vec![vec![0.0f64; k]; num_groups];
    for g in 0..num_groups {
        for t in 0..k {
            phi_sum[g][t] = phi_by_group[g].iter().map(|row| row[t]).sum();
        }
    }
    let mut kappa_t = vec![vec![0.0f64; v]; k];
    let mut kappa_s = vec![vec![0.0f64; v]; k];

    match est {
        KappaEst::Ridge(ridge) => {
            let mut z = vec![0.0f64; num_groups];
            let mut off = vec![0.0f64; num_groups];
            let mut y = vec![0.0f64; num_groups];
            for t in 0..k {
                for g in 0..num_groups {
                    z[g] = alpha_agg[g][t];
                }
                for word in 0..v {
                    for g in 0..num_groups {
                        off[g] = (phi_sum[g][t].max(1e-12)).ln() + mv[word];
                        y[g] = phi_by_group[g][word][t];
                    }
                    let (a, b) = poisson_2param(&y, &z, &off, ridge);
                    kappa_t[t][word] = a;
                    kappa_s[t][word] = b;
                }
            }
        }
        KappaEst::Lasso {
            nlambda,
            lambda_min_ratio,
        } => {
            // Joint (G·K)×(2K) design, word-independent: row (g,t) carries a 1 in
            // the topic-t dummy column and α^(s)_agg in the topic-t slope column.
            let n = num_groups * k;
            let p = 2 * k;
            let mut x = vec![vec![0.0f64; p]; n];
            let mut base_off = vec![0.0f64; n];
            for g in 0..num_groups {
                for t in 0..k {
                    let r = g * k + t;
                    x[r][t] = 1.0;
                    x[r][k + t] = alpha_agg[g][t];
                    base_off[r] = (phi_sum[g][t].max(1e-12)).ln();
                }
            }
            let mut y = vec![0.0f64; n];
            let mut off = vec![0.0f64; n];
            for word in 0..v {
                for g in 0..num_groups {
                    for t in 0..k {
                        let r = g * k + t;
                        y[r] = phi_by_group[g][word][t];
                        off[r] = base_off[r] + mv[word];
                    }
                }
                let coef = poisson_lasso(&x, &y, &off, nlambda, lambda_min_ratio);
                for t in 0..k {
                    kappa_t[t][word] = coef[t];
                    kappa_s[t][word] = coef[k + t];
                }
            }
        }
    }
    (kappa_t, kappa_s)
}

/// Per-group means of the sentiment block `α^(s)` (the continuous covariate the
/// `κ` Poisson regression aggregates over). Returns `num_groups × K`.
fn group_means(alpha: &[Vec<f64>], group: &[usize], num_groups: usize, k: usize) -> Vec<Vec<f64>> {
    let mut sums = vec![vec![0.0f64; k]; num_groups];
    let mut counts = vec![0usize; num_groups];
    for (di, a) in alpha.iter().enumerate() {
        let g = group[di];
        counts[g] += 1;
        for t in 0..k {
            sums[g][t] += a[k - 1 + t];
        }
    }
    for g in 0..num_groups {
        if counts[g] > 0 {
            for t in 0..k {
                sums[g][t] /= counts[g] as f64;
            }
        }
    }
    sums
}

/// Fit the STS model (Chen & Mankad 2024). The E-step is the Laplace variational
/// inference of [`sts_lhood`]/[`sts_grad`]/[`sts_precision`]; the M-step updates
/// `Γ` (pooled ridge), `Σ` (as in CTM/STM), and `κ` (the Poisson regression of
/// [`opt_kappa`]). Aggregation groups for the `κ` step are the distinct levels of
/// `sentiment_seed`, which also seeds the initial `α^(s)`.
#[allow(clippy::too_many_arguments)]
pub fn fit_sts<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    em_iters: usize,
    em_tol: f64,
    prevalence: Option<&[Vec<f64>]>,
    sentiment_seed: Option<&[f64]>,
    kappa_est: KappaEst,
    init_spectral: bool,
    keep_nu: bool,
    rng: &mut R,
) -> StsModel {
    let k = num_topics;
    let n = 2 * k - 1;
    let d = docs.len();
    let v = num_types;
    let nf = prevalence.map(|x| x[0].len());

    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // Baseline log word rates m_v.
    let mut freq = vec![1.0f64; v];
    let mut total = v as f64;
    for doc in docs {
        for &w in doc {
            freq[w as usize] += 1.0;
            total += 1.0;
        }
    }
    let mv: Vec<f64> = (0..v).map(|i| (freq[i] / total).ln()).collect();

    // β init (anchor-word spectral, else random), then κ_t = ln β − m so that at
    // α^(s)=0 the topics start at β; κ_s fixed proportional to κ_t.
    let beta = if init_spectral {
        crate::spectral::spectral_init(docs, k, v).unwrap_or_else(|| {
            let mut b = vec![vec![0.0f64; v]; k];
            for row in b.iter_mut() {
                let mut s = 0.0;
                for x in row.iter_mut() {
                    *x = 1.0 + rng.gen::<f64>();
                    s += *x;
                }
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
            b
        })
    } else {
        let mut b = vec![vec![0.0f64; v]; k];
        for row in b.iter_mut() {
            let mut s = 0.0;
            for x in row.iter_mut() {
                *x = 1.0 + rng.gen::<f64>();
                s += *x;
            }
            for x in row.iter_mut() {
                *x /= s;
            }
        }
        b
    };
    // Aggregation groups for the κ M-step: the distinct levels of the sentiment
    // seed. Without a seed, a single group leaves κ_s unidentified and the model
    // reduces toward CTM on the prevalence side.
    let (group, num_groups) = match sentiment_seed {
        Some(seed) => {
            let mut levels: Vec<f64> = seed.to_vec();
            levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
            levels.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            let g: Vec<usize> = seed
                .iter()
                .map(|s| levels.iter().position(|l| (l - s).abs() < 1e-12).unwrap())
                .collect();
            (g, levels.len())
        }
        None => (vec![0usize; d], 1),
    };

    // κ_t starts at ln β − m (topics begin at the spectral β when α^(s)=0); κ_s
    // starts at 0 and is set by the initial κ estimation below.
    let mut kappa_t = vec![vec![0.0f64; v]; k];
    let mut kappa_s = vec![vec![0.0f64; v]; k];
    for t in 0..k {
        for i in 0..v {
            kappa_t[t][i] = beta[t][i].max(1e-12).ln() - mv[i];
        }
    }

    // Latent init: prevalence at 0, sentiment at the centered seed (or 0).
    let mut alpha = vec![vec![0.0f64; n]; d];
    if let Some(seed) = sentiment_seed {
        let mean: f64 = seed.iter().sum::<f64>() / d as f64;
        for di in 0..d {
            for t in 0..k {
                alpha[di][k - 1 + t] = seed[di] - mean;
            }
        }
    }

    // Initial κ (Chen & Mankad §4.3): aggregate the spectral-β responsibilities
    // (uniform θ) by group and run the Poisson M-step against the seeded α^(s), so
    // κ_s enters the first E-step non-zero.
    {
        let mut phi_by_group = vec![vec![vec![0.0f64; k]; v]; num_groups];
        for (di, (words, counts)) in sparse.iter().enumerate() {
            for (wi, &w) in words.iter().enumerate() {
                let denom: f64 = (0..k).map(|t| beta[t][w]).sum::<f64>().max(1e-12);
                for t in 0..k {
                    phi_by_group[group[di]][w][t] += counts[wi] * beta[t][w] / denom;
                }
            }
        }
        let alpha_agg = group_means(&alpha, &group, num_groups, k);
        let (kt, ks) = opt_kappa(&phi_by_group, &alpha_agg, &mv, k, v, num_groups, kappa_est);
        kappa_t = kt;
        kappa_s = ks;
    }

    let mut gamma: Option<Vec<Vec<f64>>> = nf.map(|f| vec![vec![0.0f64; n]; f]);
    let mut mu_shared = vec![0.0f64; n];
    let mut sigma = vec![0.0f64; n * n];
    for i in 0..n {
        sigma[i * n + i] = 1.0;
    }
    // When keep_nu=false, don't allocate storage for per-doc ν (saves O(N·(2K-1)²)
    // memory), but still consume it transiently for the Σ sufficient-stat update.
    let mut nu_store: Vec<Vec<f64>> = if keep_nu {
        vec![vec![0.0f64; n * n]; d]
    } else {
        Vec::new()
    };

    let doc_mu = |di: usize, gamma: &Option<Vec<Vec<f64>>>, mu_shared: &[f64]| -> Vec<f64> {
        match (prevalence, gamma) {
            (Some(x), Some(g)) => (0..n)
                .map(|t| x[di].iter().zip(g).map(|(xi, gr)| xi * gr[t]).sum())
                .collect(),
            _ => mu_shared.to_vec(),
        }
    };

    let mut bound_history = Vec::with_capacity(em_iters);
    let mut converged = false;
    let mut em_iters_run = 0usize;
    // Track sigma and kappa used at the START of each E-step. After the loop,
    // these hold the values that produced the stored α/ν — which may differ from
    // `sigma`/`kappa_t`/`kappa_s` (updated by the M-steps after the last E-step).
    // `recompute_nu_sts` uses these to reproduce ν exactly.
    // Note: μ cancels from the STS Hessian (`sts_precision` takes `siginv` but not
    // `mu`), so ν is independent of the per-document prior mean — no need to track it.
    let mut sigma_estep = sigma.clone();
    let mut kappa_t_estep = kappa_t.clone();
    let mut kappa_s_estep = kappa_s.clone();

    for em in 0..em_iters {
        em_iters_run = em + 1;
        sigma_estep = sigma.clone(); // capture sigma before E-step
        kappa_t_estep = kappa_t.clone(); // capture kappa before E-step
        kappa_s_estep = kappa_s.clone();
        let kappa = Kappa {
            kappa_t: kappa_t.clone(),
            kappa_s: kappa_s.clone(),
        };
        let siginv = spd_inverse(&sigma, n).unwrap_or_else(|| {
            let mut s = sigma.clone();
            make_diagonally_dominant(&mut s, n);
            spd_inverse(&s, n).unwrap()
        });
        let entropy = match cholesky(&sigma, n) {
            Some(l) => half_logdet(&l, n),
            None => 0.0,
        };

        // E-step. Per-document variational inference is independent across
        // documents, so run it in parallel and reduce serially in document order:
        // the sufficient statistics (φ, bound) are summed in the same order as a
        // serial loop, so the fit stays bit-for-bit deterministic regardless of
        // thread count (the guarantee ctm.rs makes for STM/CTM). Each document
        // returns its α̂, ν_d, bound contribution, and sparse responsibilities.
        let doc_results: Vec<(usize, (Vec<f64>, Vec<f64>, f64, Vec<(usize, Vec<f64>)>))> =
            crate::variational::laplace_estep(&sparse, |di, words, counts| {
                let mu_d = doc_mu(di, &gamma, &mu_shared);
                let a_hat = lbfgs_minimize(
                    alpha[di].clone(),
                    |a| {
                        (
                            -sts_lhood(a, &kappa, &mv, words, counts, &mu_d, &siginv, k),
                            sts_grad(a, &kappa, &mv, words, counts, &mu_d, &siginv, k)
                                .iter()
                                .map(|g| -g)
                                .collect(),
                        )
                    },
                    100,
                    7,
                    1e-5,
                );
                let mut prec = sts_precision(&a_hat, &kappa, &mv, words, counts, &siginv, k);
                let (nu_d, half_ld) = match cholesky(&prec, n) {
                    Some(l) => (spd_inverse_from_chol(&l, n), half_logdet(&l, n)),
                    None => {
                        make_diagonally_dominant(&mut prec, n);
                        let l = cholesky(&prec, n).expect("PD after diagonal dominance");
                        (spd_inverse_from_chol(&l, n), half_logdet(&l, n))
                    }
                };
                // bound = f(α̂) − 0.5·log|prec| − 0.5·log|Σ| (standard Laplace ELBO;
                // ll − 0.5 quad collapses to sts_lhood at the optimum).
                let f_at = sts_lhood(&a_hat, &kappa, &mv, words, counts, &mu_d, &siginv, k);
                let bound_contrib = f_at - half_ld - entropy;

                // Responsibilities φ_{d,w,k} = c_{d,w}·θ_k β_{k,w}/Σ θ β (sparse over
                // the document's words), reduced into the aggregation group below.
                let db = doc_beta(&a_hat, &kappa, &mv, k);
                let phi_contrib: Vec<(usize, Vec<f64>)> = words
                    .iter()
                    .enumerate()
                    .map(|(wi, &w)| {
                        let denom: f64 = (0..k).map(|t| db.expeta[t] * db.beta[t][w]).sum();
                        let row = (0..k)
                            .map(|t| counts[wi] * db.expeta[t] * db.beta[t][w] / denom)
                            .collect();
                        (w, row)
                    })
                    .collect();
                (a_hat, nu_d, bound_contrib, phi_contrib)
            });

        // Serial reduction in document order (deterministic).
        // When keep_nu=false we accumulate the ν sufficient-stat for Σ on the fly
        // (so the Σ M-step is bit-identical whether or not we retain per-doc ν).
        let mut phi_by_group = vec![vec![vec![0.0f64; k]; v]; num_groups];
        let mut total_bound = 0.0;
        let mut nu_sum_sstat = if keep_nu {
            Vec::new()
        } else {
            vec![0.0f64; n * n]
        };
        for (di, (a_hat, nu_d, bound_contrib, phi_contrib)) in doc_results {
            let g = group[di];
            for (w, row) in &phi_contrib {
                for t in 0..k {
                    phi_by_group[g][*w][t] += row[t];
                }
            }
            total_bound += bound_contrib;
            alpha[di] = a_hat;
            if keep_nu {
                nu_store[di] = nu_d;
            } else {
                // Accumulate ν into the sufficient stat; discard the per-doc array.
                for idx in 0..(n * n) {
                    nu_sum_sstat[idx] += nu_d[idx];
                }
            }
        }
        bound_history.push(total_bound);

        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (total_bound - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }

        // M-step: Γ (pooled ridge) or shared mean μ.
        if let (Some(x), Some(f)) = (prevalence, nf) {
            gamma = Some(fit_gamma_ridge(x, &alpha, f, n, 1e-6));
        } else {
            for i in 0..n {
                mu_shared[i] = alpha.iter().map(|a| a[i]).sum::<f64>() / d as f64;
            }
        }

        // Σ = (1/D)[ Σ_d ν_d + Σ_d (α_d − μ_d)(α_d − μ_d)ᵀ ].
        let mus: Vec<Vec<f64>> = (0..d).map(|di| doc_mu(di, &gamma, &mu_shared)).collect();
        for i in 0..n {
            for j in 0..n {
                let mut cross = 0.0;
                for di in 0..d {
                    cross += (alpha[di][i] - mus[di][i]) * (alpha[di][j] - mus[di][j]);
                }
                let nu_sum: f64 = if keep_nu {
                    nu_store.iter().map(|nu| nu[i * n + j]).sum()
                } else {
                    nu_sum_sstat[i * n + j]
                };
                sigma[i * n + j] = (nu_sum + cross) / d as f64;
            }
        }

        // κ M-step: Poisson regression of the aggregated counts on the group-mean
        // sentiment (skipped when there is only one group, which leaves κ_s
        // unidentified and κ_t at its β-derived initialization).
        if num_groups >= 2 {
            let alpha_agg = group_means(&alpha, &group, num_groups, k);
            let (kt, ks) = opt_kappa(&phi_by_group, &alpha_agg, &mv, k, v, num_groups, kappa_est);
            kappa_t = kt;
            kappa_s = ks;
        }
    }

    let beta: Vec<Vec<f64>> = (0..k)
        .map(|t| topic_beta(&mv, &kappa_t[t], &kappa_s[t], 0.0))
        .collect();

    StsModel {
        k,
        num_types: v,
        alpha,
        nu: nu_store,
        gamma,
        sigma,
        sigma_estep,
        kappa_t,
        kappa_s,
        kappa_t_estep,
        kappa_s_estep,
        mv,
        beta,
        bound_history,
        converged,
        em_iters_run,
    }
}

/// Recompute the per-document variational covariance ν from the stored
/// variational means α and the fitted global parameters. Used when the model
/// was fit with `keep_nu = false` to reconstruct ν on demand (e.g. for
/// method-of-composition uncertainty). Parallelized with rayon.
///
/// The Laplace covariance ν = H⁻¹ depends only on α, κ, m and Σ⁻¹ — the
/// prior mean μ cancels out of the STS Hessian (`sts_precision` takes `siginv`
/// but not `mu`). So ν is independent of the per-document μ. We evaluate using
/// `sigma_estep` and `kappa_t_estep`/`kappa_s_estep` (the globals that were active
/// during the last E-step) to reproduce the stored ν exactly.
pub fn recompute_nu_sts(model: &StsModel, sparse: &[(Vec<usize>, Vec<f64>)]) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let d = sparse.len();
    let k = model.k;
    let n = 2 * k - 1;
    // Use sigma_estep (the sigma active during the last E-step) rather than
    // model.sigma (updated by the final M-step and may differ).
    let sig = &model.sigma_estep;
    let siginv = spd_inverse(sig, n).unwrap_or_else(|| {
        let mut s = sig.clone();
        make_diagonally_dominant(&mut s, n);
        spd_inverse(&s, n).unwrap()
    });
    // Use kappa from the last E-step (before the final κ M-step updated kappa_t/kappa_s).
    let kappa = Kappa {
        kappa_t: model.kappa_t_estep.clone(),
        kappa_s: model.kappa_s_estep.clone(),
    };
    (0..d)
        .into_par_iter()
        .map(|di| {
            let (words, counts) = &sparse[di];
            let mut prec = sts_precision(
                &model.alpha[di],
                &kappa,
                &model.mv,
                words,
                counts,
                &siginv,
                k,
            );
            match cholesky(&prec, n) {
                Some(l) => spd_inverse_from_chol(&l, n),
                None => {
                    make_diagonally_dominant(&mut prec, n);
                    let l = cholesky(&prec, n).expect("PD after diagonal dominance");
                    spd_inverse_from_chol(&l, n)
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (
        usize,
        usize,
        Kappa,
        Vec<f64>,
        Vec<usize>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        // Small deterministic problem: K=3 topics, V=6 vocabulary.
        let k = 3usize;
        let v = 6usize;
        // Distinct kappa per topic so the sentiment block is non-degenerate.
        let mut kt = vec![vec![0.0; v]; k];
        let mut ks = vec![vec![0.0; v]; k];
        for t in 0..k {
            for i in 0..v {
                kt[t][i] = 0.10 * ((t * 7 + i * 3 % 5) as f64) - 0.5;
                ks[t][i] = 0.20 * (((i + 2 * t) % 4) as f64) - 0.3;
            }
        }
        let kappa = Kappa {
            kappa_t: kt,
            kappa_s: ks,
        };
        let mv: Vec<f64> = (0..v).map(|i| -1.0 - 0.1 * i as f64).collect();
        let words = vec![0usize, 2, 3, 5];
        let counts = vec![3.0, 1.0, 4.0, 2.0];
        let n = 2 * k - 1;
        let mu: Vec<f64> = (0..n).map(|i| 0.05 * i as f64 - 0.1).collect();
        // A symmetric positive-definite precision: 2I + small off-diagonal.
        let mut siginv = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                siginv[i * n + j] = if i == j { 2.0 } else { 0.1 };
            }
        }
        (k, v, kappa, mv, words, counts, mu, siginv)
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let (k, _v, kappa, mv, words, counts, mu, siginv) = setup();
        let n = 2 * k - 1;
        let alpha: Vec<f64> = vec![0.3, -0.2, 0.4, -0.1, 0.25];
        let g = sts_grad(&alpha, &kappa, &mv, &words, &counts, &mu, &siginv, k);
        let eps = 1e-6;
        for i in 0..n {
            let mut ap = alpha.clone();
            let mut am = alpha.clone();
            ap[i] += eps;
            am[i] -= eps;
            let fp = sts_lhood(&ap, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            let fm = sts_lhood(&am, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            let fd = (fp - fm) / (2.0 * eps);
            assert!((g[i] - fd).abs() < 1e-5, "grad[{i}] {} vs fd {}", g[i], fd);
        }
    }

    #[test]
    fn precision_is_negative_hessian_of_f() {
        let (k, _v, kappa, mv, words, counts, mu, siginv) = setup();
        let n = 2 * k - 1;
        let alpha: Vec<f64> = vec![0.3, -0.2, 0.4, -0.1, 0.25];
        let p = sts_precision(&alpha, &kappa, &mv, &words, &counts, &siginv, k);
        // Central-difference the gradient: ∇²f_{ij} = d grad_i / d alpha_j.
        // The returned matrix is -∇²f, so it must equal -FD(grad).
        let eps = 1e-6;
        for j in 0..n {
            let mut ap = alpha.clone();
            let mut am = alpha.clone();
            ap[j] += eps;
            am[j] -= eps;
            let gp = sts_grad(&ap, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            let gm = sts_grad(&am, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            for i in 0..n {
                let hess_ij = (gp[i] - gm[i]) / (2.0 * eps);
                assert!(
                    (p[i * n + j] + hess_ij).abs() < 1e-4,
                    "precision[{i},{j}] {} vs -hess {}",
                    p[i * n + j],
                    -hess_ij
                );
            }
        }
    }

    #[test]
    fn symmetric_precision() {
        let (k, _v, kappa, mv, words, counts, _mu, siginv) = setup();
        let n = 2 * k - 1;
        let alpha: Vec<f64> = vec![0.3, -0.2, 0.4, -0.1, 0.25];
        let p = sts_precision(&alpha, &kappa, &mv, &words, &counts, &siginv, k);
        for i in 0..n {
            for j in 0..n {
                assert!((p[i * n + j] - p[j * n + i]).abs() < 1e-9);
            }
        }
    }

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Two topics on disjoint vocabulary blocks; each document is drawn from one
    /// block, and the prevalence covariate is the block indicator.
    fn planted_corpus() -> (Vec<Vec<u32>>, Vec<Vec<f64>>, Vec<usize>, usize) {
        let v = 8usize;
        let mut rng = StdRng::seed_from_u64(0);
        let block_a: Vec<u32> = (0..4).collect();
        let block_b: Vec<u32> = (4..8).collect();
        let mut docs = Vec::new();
        let mut x = Vec::new();
        let mut truth = Vec::new();
        for d in 0..60 {
            let a = d % 2 == 0;
            let block = if a { &block_a } else { &block_b };
            let doc: Vec<u32> = (0..12)
                .map(|_| block[rng.gen_range(0..block.len())])
                .collect();
            docs.push(doc);
            x.push(vec![1.0, if a { 0.0 } else { 1.0 }]); // intercept + indicator
            truth.push(if a { 0 } else { 1 });
        }
        (docs, x, truth, v)
    }

    #[test]
    fn em_bound_increases_and_recovers_topics() {
        let (docs, x, truth, v) = planted_corpus();
        let mut rng = StdRng::seed_from_u64(1);
        let m = fit_sts(
            &docs,
            2,
            v,
            40,
            1e-6,
            Some(&x),
            None,
            KappaEst::Ridge(1e-3),
            true,
            true,
            &mut rng,
        );

        // The variational bound increases monotonically (allowing tiny slack).
        for w in m.bound_history.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "bound dropped: {} -> {}", w[0], w[1]);
        }

        // The two planted blocks are recovered as the two topics: each topic's
        // top words come from one block. Map topics to blocks by their heaviest
        // word, then check prevalence separates the document groups.
        let tw = m.topic_word();
        let top0 = (0..v)
            .max_by(|&a, &b| tw[0][a].partial_cmp(&tw[0][b]).unwrap())
            .unwrap();
        let topic_for_block_a = if top0 < 4 { 0 } else { 1 };

        let theta = m.doc_topics();
        let mut correct = 0;
        for (d, th) in theta.iter().enumerate() {
            let dominant = if th[0] >= th[1] { 0 } else { 1 };
            let expected = if truth[d] == 0 {
                topic_for_block_a
            } else {
                1 - topic_for_block_a
            };
            if dominant == expected {
                correct += 1;
            }
        }
        assert!(
            correct as f64 / theta.len() as f64 > 0.9,
            "only {correct}/60 docs separated"
        );
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let (docs, x, _truth, v) = planted_corpus();
        let mut r1 = StdRng::seed_from_u64(1);
        let mut r2 = StdRng::seed_from_u64(1);
        let m1 = fit_sts(
            &docs,
            2,
            v,
            15,
            0.0,
            Some(&x),
            None,
            KappaEst::Ridge(1e-3),
            true,
            true,
            &mut r1,
        );
        let m2 = fit_sts(
            &docs,
            2,
            v,
            15,
            0.0,
            Some(&x),
            None,
            KappaEst::Ridge(1e-3),
            true,
            true,
            &mut r2,
        );
        for (a, b) in m1.alpha.iter().flatten().zip(m2.alpha.iter().flatten()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(m1.bound_history.len(), m2.bound_history.len());
    }

    #[test]
    fn poisson_2param_recovers_coefficients() {
        // Counts generated from a known (a, b): y_g = exp(offset_g + a + b·z_g).
        let (a_true, b_true) = (0.7f64, -1.3f64);
        let z: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let offset: Vec<f64> = vec![0.2, -0.1, 0.0, 0.15, -0.2, 0.05];
        let y: Vec<f64> = z
            .iter()
            .zip(&offset)
            .map(|(zi, oi)| (oi + a_true + b_true * zi).exp())
            .collect();
        let (a, b) = poisson_2param(&y, &z, &offset, 0.0);
        assert!((a - a_true).abs() < 1e-4, "a {a} vs {a_true}");
        assert!((b - b_true).abs() < 1e-4, "b {b} vs {b_true}");
    }

    #[test]
    fn poisson_lasso_recovers_sparse_signal() {
        // Two predictors: one with a real effect, one pure noise. The lasso path
        // (AIC-selected) should keep the signal and zero the noise.
        let rng = &mut StdRng::seed_from_u64(0);
        let n = 60;
        let (b0_true, b1_true) = (0.8f64, 0.0f64); // second predictor has no effect
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut off = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = rng.gen_range(-1.5..1.5);
            let x1 = rng.gen_range(-1.5..1.5); // noise predictor
            let o = 0.5;
            let mean = (o + b0_true * x0 + b1_true * x1).exp();
            // Poisson draw via simple inversion.
            let mut k = 0;
            let mut pp = (-mean).exp();
            let mut f = pp;
            let u: f64 = rng.gen();
            while u > f && k < 1000 {
                k += 1;
                pp *= mean / k as f64;
                f += pp;
            }
            x.push(vec![x0, x1]);
            y.push(k as f64);
            off.push(o);
        }
        let coef = poisson_lasso(&x, &y, &off, 100, 0.001);
        assert!(
            (coef[0] - b0_true).abs() < 0.25,
            "signal coef {} vs {}",
            coef[0],
            b0_true
        );
        assert!(
            coef[1].abs() < 0.1,
            "noise coef should be ~0, got {}",
            coef[1]
        );
    }

    #[test]
    fn poisson_2param_intercept_only_when_no_spread() {
        // No covariate spread: slope unidentified, intercept still recovered.
        let z = vec![1.0, 1.0, 1.0];
        let offset = vec![0.0, 0.0, 0.0];
        let y = vec![2.0f64.exp(); 3];
        let (a, b) = poisson_2param(&y, &z, &offset, 0.0);
        assert!((a - 2.0).abs() < 1e-4);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn learns_sentiment_with_groups() {
        // The block indicator doubles as the sentiment seed (two groups), so the
        // κ M-step is exercised: it should learn a non-zero κ_s while still
        // recovering the topics and increasing the bound overall.
        let (docs, x, _truth, v) = planted_corpus();
        let seed: Vec<f64> = x.iter().map(|row| row[1]).collect();
        let mut rng = StdRng::seed_from_u64(2);
        let m = fit_sts(
            &docs,
            2,
            v,
            30,
            1e-6,
            Some(&x),
            Some(&seed),
            KappaEst::Ridge(1e-3),
            true,
            true,
            &mut rng,
        );

        let ks_max = m
            .kappa_s
            .iter()
            .flatten()
            .fold(0.0f64, |acc, &x| acc.max(x.abs()));
        assert!(ks_max > 1e-3, "kappa_s never moved off zero: {ks_max}");
        assert!(
            *m.bound_history.last().unwrap() >= m.bound_history[0] - 1e-6,
            "bound did not improve overall"
        );
    }

    #[test]
    fn lasso_kappa_fits_end_to_end() {
        // The lasso κ M-step path runs through a full fit and recovers topics.
        let (docs, x, truth, v) = planted_corpus();
        let seed: Vec<f64> = x.iter().map(|row| row[1]).collect();
        let mut rng = StdRng::seed_from_u64(3);
        let est = KappaEst::Lasso {
            nlambda: 60,
            lambda_min_ratio: 0.001,
        };
        let m = fit_sts(
            &docs,
            2,
            v,
            20,
            1e-6,
            Some(&x),
            Some(&seed),
            est,
            true,
            true,
            &mut rng,
        );

        let tw = m.topic_word();
        let top0 = (0..v)
            .max_by(|&a, &b| tw[0][a].partial_cmp(&tw[0][b]).unwrap())
            .unwrap();
        let topic_for_a = if top0 < 4 { 0 } else { 1 };
        let theta = m.doc_topics();
        let correct = theta
            .iter()
            .enumerate()
            .filter(|(d, th)| {
                let dominant = if th[0] >= th[1] { 0 } else { 1 };
                let expected = if truth[*d] == 0 {
                    topic_for_a
                } else {
                    1 - topic_for_a
                };
                dominant == expected
            })
            .count();
        assert!(
            correct as f64 / theta.len() as f64 > 0.85,
            "lasso fit only {correct}/60 separated"
        );
    }

    #[test]
    fn sts_conforms() {
        use crate::conformance::{check_conformance, check_logistic_normal};
        let (docs, x, _truth, v) = planted_corpus();
        let seed: Vec<f64> = x.iter().map(|row| row[1]).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let m = fit_sts(
            &docs,
            3,
            v,
            5,
            0.0,
            Some(&x),
            Some(&seed),
            KappaEst::Ridge(1e-3),
            true,
            true,
            &mut rng,
        );
        let base = check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let ln = check_logistic_normal(&m);
        assert!(ln.is_empty(), "check_logistic_normal: {:?}", ln);
        assert_eq!(m.eta_dim(), 2 * m.k - 1);
        assert_eq!(m.eta_mean().len(), m.eta_cov().len());
    }
}
