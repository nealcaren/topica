//! TBIP: Text-Based Ideal Points (Vafa, Naidu & Blei 2020, ACL; arXiv 2005.04232).
//!
//! A Poisson factorization in which an author's latent ideal point `x_s` rescales
//! a *neutral* topic-word intensity `beta_kv` by an *ideological* per-word factor
//! `exp(x_s * eta_kv)`. Documents are mixtures over topics with positive per-doc
//! intensities `theta_dk`:
//!
//! ```text
//!   theta_dk ~ Gamma(a,b)   beta_kv ~ Gamma(a,b)   eta_kv ~ N(0,1)   x_s ~ N(0,1)
//!   y_dv ~ Poisson( sum_k theta_dk * beta_kv * exp(x_{a_d} * eta_kv) )
//! ```
//!
//! Inference is the paper's mean-field VI (NOT the MAP shortcut): a fully factored
//! `q` with LogNormal factors for the positive `theta`/`beta` and Normal factors
//! for the real `eta`/`x`. The ELBO is maximized by reparameterized single-sample
//! stochastic gradient ascent (Adam) with document minibatching (SVI). KL is
//! analytic for the Gaussian factors; the LogNormal-vs-Gamma terms use the same MC
//! sample. The gradients are hand-coded (reverse-mode by hand, like `prodlda.rs`)
//! and FD-checked in the unit tests below.
//!
//! This is a faithful reimplementation of the published model and inference; the
//! official code is TF1.14/TFP0.7 graph-mode (no Apple-Silicon build), so the
//! numerics are validated against a PyTorch reference (scratchpad/tbip_vi.py),
//! synthetic planted-position recovery, and FD gradient checks.

use rand::Rng;
use rayon::prelude::*;

const SQRT_FLOOR: f64 = 1e-4; // sigma = softplus(rs) + this
const RATE_FLOOR: f64 = 1e-7; // Poisson rate floor (matches the reference)

/// Standard normal via Box-Muller (matches the prodlda/etm_vae precedent; one RNG
/// drives both minibatch and reparam noise for deterministic fits).
#[inline]
fn randn<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

#[inline]
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Elementwise Adam matching torch's defaults (beta1=0.9, beta2=0.999, eps=1e-8),
/// with no weight decay. `t` is shared across calls so the bias correction tracks
/// the global step.
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    lr: f64,
}

impl Adam {
    fn new(len: usize, lr: f64) -> Self {
        Adam {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
            lr,
        }
    }
    /// One ascent step: `p += lr * m_hat / (sqrt(v_hat)+eps)` for a gradient of the
    /// objective we *maximize* (so grad is dELBO/dp, and we add).
    fn step(&mut self, p: &mut [f64], grad: &[f64]) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for (pi, (&g, (mi, vi))) in p
            .iter_mut()
            .zip(grad.iter().zip(self.m.iter_mut().zip(self.v.iter_mut())))
        {
            *mi = B1 * *mi + (1.0 - B1) * g;
            *vi = B2 * *vi + (1.0 - B2) * g * g;
            *pi += self.lr * (*mi / bc1) / ((*vi / bc2).sqrt() + EPS);
        }
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// Variational parameters (mean + raw scale per factor). Flat row-major buffers.
pub struct TbipParams {
    pub num_docs: usize,
    pub num_topics: usize,
    pub num_types: usize,
    pub num_authors: usize,
    pub mu_theta: Vec<f64>, // D*K
    pub rs_theta: Vec<f64>, // D*K
    pub mu_beta: Vec<f64>,  // K*V
    pub rs_beta: Vec<f64>,  // K*V
    pub mu_eta: Vec<f64>,   // K*V
    pub rs_eta: Vec<f64>,   // K*V
    pub mu_x: Vec<f64>,     // A
    pub rs_x: Vec<f64>,     // A
}

/// A fitted TBIP. Posterior means of the variational factors are the point
/// estimates; `ideal_points` is `mu_x`.
pub struct TbipModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub num_authors: usize,
    pub params: TbipParams,
    pub group: Vec<usize>,
    pub elbo_history: Vec<f64>,
    pub iters_run: usize,
}

impl TbipModel {
    /// Author ideal points = posterior mean `mu_x` (length A).
    pub fn ideal_points(&self) -> Vec<f64> {
        self.params.mu_x.clone()
    }

    /// Standard error of each author ideal point (length A): the standard deviation
    /// of the Gaussian variational posterior `q(x_s) = N(mu_x, sig_x^2)`,
    /// `sig_x = softplus(rs_x) + SQRT_FLOOR`. This is the model's own (mean-field)
    /// posterior uncertainty on the position, estimated jointly with the mean; like
    /// any mean-field VI it can understate the true posterior spread.
    pub fn position_se(&self) -> Vec<f64> {
        self.params
            .rs_x
            .iter()
            .map(|&rs| softplus(rs) + SQRT_FLOOR)
            .collect()
    }

    /// Neutral topic-word matrix from `exp(mu_beta)`, row-normalized for display
    /// (K x V simplices).
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        let (k, v) = (self.num_topics, self.num_types);
        (0..k)
            .map(|kk| {
                let row: Vec<f64> = (0..v)
                    .map(|vv| self.params.mu_beta[kk * v + vv].exp())
                    .collect();
                let z: f64 = row.iter().sum::<f64>().max(1e-300);
                row.iter().map(|&e| e / z).collect()
            })
            .collect()
    }

    /// Ideological topics `eta` (K x V, real-valued = `mu_eta`).
    pub fn ideological_topics(&self) -> Vec<Vec<f64>> {
        let (k, v) = (self.num_topics, self.num_types);
        (0..k)
            .map(|kk| (0..v).map(|vv| self.params.mu_eta[kk * v + vv]).collect())
            .collect()
    }

    /// Document-topic intensities, row-normalized to proportions (D x K).
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        let (d, k) = (self.params.num_docs, self.num_topics);
        (0..d)
            .map(|dd| {
                let row: Vec<f64> = (0..k)
                    .map(|kk| self.params.mu_theta[dd * k + kk].exp())
                    .collect();
                let z: f64 = row.iter().sum::<f64>().max(1e-300);
                row.iter().map(|&e| e / z).collect()
            })
            .collect()
    }
}

/// Hyperparameters / schedule for a TBIP fit. Defaults mirror the reference.
#[derive(Clone, Copy)]
pub struct TbipConfig {
    pub a_gamma: f64,
    pub b_gamma: f64,
    pub iters: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

impl Default for TbipConfig {
    fn default() -> Self {
        TbipConfig {
            a_gamma: 0.3,
            b_gamma: 0.3,
            iters: 7000,
            batch_size: 512,
            learning_rate: 0.05,
        }
    }
}

/// Initialize the variational parameters from data (per the reference / spec):
/// mu_theta=-2, rs_theta=-4; mu_beta=log(empirical word freq) broadcast over K,
/// rs_beta=-4; mu_eta=0.01*randn, rs_eta=-4; mu_x=0.1*randn, rs_x=-2.
fn init_params<R: Rng>(
    docs: &[Vec<u32>],
    group: &[usize],
    num_authors: usize,
    k: usize,
    v: usize,
    rng: &mut R,
) -> TbipParams {
    let d = docs.len();
    // Empirical word frequency (over the whole corpus).
    let mut wf = vec![0.0f64; v];
    let mut total = 0.0f64;
    for doc in docs {
        for &w in doc {
            wf[w as usize] += 1.0;
            total += 1.0;
        }
    }
    let total = total.max(1.0);
    let mut mu_beta = vec![0.0f64; k * v];
    for kk in 0..k {
        for vv in 0..v {
            mu_beta[kk * v + vv] = (wf[vv] / total + 1e-4).ln();
        }
    }
    let mu_theta = vec![-2.0f64; d * k];
    let rs_theta = vec![-4.0f64; d * k];
    let rs_beta = vec![-4.0f64; k * v];
    let mut mu_eta = vec![0.0f64; k * v];
    for e in mu_eta.iter_mut() {
        let z: f64 = randn(rng);
        *e = 0.01 * z;
    }
    let rs_eta = vec![-4.0f64; k * v];
    let mut mu_x = vec![0.0f64; num_authors];
    for xs in mu_x.iter_mut() {
        let z: f64 = randn(rng);
        *xs = 0.1 * z; // break symmetry
    }
    let rs_x = vec![-2.0f64; num_authors];
    debug_assert_eq!(group.len(), d);
    TbipParams {
        num_docs: d,
        num_topics: k,
        num_types: v,
        num_authors,
        mu_theta,
        rs_theta,
        mu_beta,
        rs_beta,
        mu_eta,
        rs_eta,
        mu_x,
        rs_x,
    }
}

/// Per-step gradients accumulated for the sampled minibatch. The theta gradients
/// are sparse (only the B sampled rows are touched).
struct Grads {
    // dense globals
    g_mu_beta: Vec<f64>,
    g_rs_beta: Vec<f64>,
    g_mu_eta: Vec<f64>,
    g_rs_eta: Vec<f64>,
    g_mu_x: Vec<f64>,
    g_rs_x: Vec<f64>,
    // sparse theta updates: (doc, dmu[K], drs[K])
    theta_rows: Vec<(usize, Vec<f64>, Vec<f64>)>,
    elbo: f64,
}

/// One reparameterized single-sample ELBO + gradient evaluation over a minibatch
/// of documents `bidx`. `eps_*` are the reparam noise draws for this step. Returns
/// the (D/B)-scaled ELBO estimate and gradients (of the ELBO, for ascent).
///
/// This is the analytic adjoint of the forward pass in the spec; the unit tests
/// FD-check every component.
#[allow(clippy::too_many_arguments)]
fn elbo_and_grads(
    p: &TbipParams,
    docs: &[Vec<u32>],
    counts_buf: &mut [f64], // scratch (V) cleared per doc
    group: &[usize],
    bidx: &[usize],
    eps_theta: &[f64], // B*K
    eps_beta: &[f64],  // K*V
    eps_eta: &[f64],   // K*V
    eps_x: &[f64],     // A
    cfg: &TbipConfig,
) -> Grads {
    let k = p.num_topics;
    let v = p.num_types;
    let a_n = p.num_authors;
    let d_total = p.num_docs as f64;
    let b = bidx.len();
    let svi = d_total / b as f64;
    let (a_g, b_g) = (cfg.a_gamma, cfg.b_gamma);

    // ---- sample the global factors (full each step) ----
    // beta_kv = exp(clamp(mu+sig*eps,-12,6)), with the clamped log kept for grads.
    let mut sig_beta = vec![0.0f64; k * v];
    let mut beta = vec![0.0f64; k * v];
    let mut beta_unclamped = vec![false; k * v]; // true if clamp was active (grad=0 path)
    for i in 0..k * v {
        let sig = softplus(p.rs_beta[i]) + SQRT_FLOOR;
        sig_beta[i] = sig;
        let u = p.mu_beta[i] + sig * eps_beta[i];
        let uc = u.clamp(-12.0, 6.0);
        beta[i] = uc.exp();
        beta_unclamped[i] = (-12.0..=6.0).contains(&u);
    }
    // eta_kv = mu+sig*eps (no clamp).
    let mut sig_eta = vec![0.0f64; k * v];
    let mut eta = vec![0.0f64; k * v];
    for i in 0..k * v {
        let sig = softplus(p.rs_eta[i]) + SQRT_FLOOR;
        sig_eta[i] = sig;
        eta[i] = p.mu_eta[i] + sig * eps_eta[i];
    }
    // x_s = mu+sig*eps (no clamp).
    let mut sig_x = vec![0.0f64; a_n];
    let mut x = vec![0.0f64; a_n];
    for s in 0..a_n {
        let sig = softplus(p.rs_x[s]) + SQRT_FLOOR;
        sig_x[s] = sig;
        x[s] = p.mu_x[s] + sig * eps_x[s];
    }

    // ---- sample local theta for the batch ----
    // theta[b,k] = exp(clamp(mu+sig*eps,-10,6)).
    let mut sig_theta = vec![0.0f64; b * k];
    let mut theta = vec![0.0f64; b * k];
    let mut theta_unclamped = vec![false; b * k];
    for bi in 0..b {
        let dd = bidx[bi];
        for kk in 0..k {
            let idx = dd * k + kk;
            let sig = softplus(p.rs_theta[idx]) + SQRT_FLOOR;
            sig_theta[bi * k + kk] = sig;
            let u = p.mu_theta[idx] + sig * eps_theta[bi * k + kk];
            let uc = u.clamp(-10.0, 6.0);
            theta[bi * k + kk] = uc.exp();
            theta_unclamped[bi * k + kk] = (-10.0..=6.0).contains(&u);
        }
    }

    // ---- likelihood + gradient pass: chunk-parallel over the minibatch ----
    // Each document's rate and gradient contributions are independent, so we split
    // the batch into a FIXED number of contiguous chunks, evaluate them in parallel
    // (rayon), and reduce them in chunk order. Fixing both the partition and the
    // reduction order keeps the fit bit-reproducible for a given seed regardless of
    // how many threads run -- the determinism guarantee topica requires. Within a
    // doc we cache `adj = exp(clamp(xb*eta))` once and reuse it in the gradient pass
    // (the exp was the dominant cost, previously computed twice).
    let _ = &counts_buf; // each chunk uses its own scratch
    const NCHUNKS: usize = 16;
    let nchunks = NCHUNKS.min(b).max(1);
    let chunk_size = b.div_ceil(nchunks);

    struct Partial {
        loglik: f64,
        dl_dbeta: Vec<f64>,  // K*V
        dl_deta: Vec<f64>,   // K*V
        dl_dx: Vec<f64>,     // A
        start: usize,        // first bi handled by this chunk
        dl_dtheta: Vec<f64>, // (end-start)*K
    }

    let partials: Vec<Partial> = (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk_size;
            let end = ((ci + 1) * chunk_size).min(b);
            let rows = end.saturating_sub(start);
            let mut part = Partial {
                loglik: 0.0,
                dl_dbeta: vec![0.0f64; k * v],
                dl_deta: vec![0.0f64; k * v],
                dl_dx: vec![0.0f64; a_n],
                start,
                dl_dtheta: vec![0.0f64; rows * k],
            };
            let mut rate = vec![0.0f64; v];
            let mut adj = vec![0.0f64; k * v]; // exp(clamp(xb*eta)), cached per doc
            let mut counts = vec![0.0f64; v];
            for bi in start..end {
                let dd = bidx[bi];
                let s = group[dd];
                let xb = x[s];
                for c in counts.iter_mut() {
                    *c = 0.0;
                }
                for &w in &docs[dd] {
                    counts[w as usize] += 1.0;
                }
                for vv in 0..v {
                    rate[vv] = RATE_FLOOR;
                }
                // rate, caching adj (single exp per (k,v)).
                for kk in 0..k {
                    let th = theta[bi * k + kk];
                    let base = kk * v;
                    for vv in 0..v {
                        let a = (xb * eta[base + vv]).clamp(-8.0, 8.0).exp();
                        adj[base + vv] = a;
                        rate[vv] += th * beta[base + vv] * a;
                    }
                }
                for vv in 0..v {
                    let r = rate[vv];
                    part.loglik += counts[vv] * r.ln() - r;
                }
                // gradient pass (reuse cached adj).
                for kk in 0..k {
                    let th = theta[bi * k + kk];
                    let base = kk * v;
                    let mut acc_theta = 0.0;
                    for vv in 0..v {
                        let bv = beta[base + vv];
                        let et = eta[base + vv];
                        let c = th * bv * adj[base + vv]; // contribution
                        let g = counts[vv] / rate[vv] - 1.0;
                        let gc = g * c;
                        acc_theta += gc / th;
                        part.dl_dbeta[base + vv] += gc / bv;
                        // chain through the clamp on xb*eta (zero gradient when clamped)
                        if (-8.0..=8.0).contains(&(xb * et)) {
                            part.dl_deta[base + vv] += gc * xb;
                            part.dl_dx[s] += gc * et;
                        }
                    }
                    part.dl_dtheta[(bi - start) * k + kk] = acc_theta;
                }
            }
            part
        })
        .collect();

    // ---- deterministic reduction in chunk order ----
    let mut dl_dtheta = vec![0.0f64; b * k]; // (B,K)
    let mut dl_dbeta = vec![0.0f64; k * v]; // (K,V)
    let mut dl_deta = vec![0.0f64; k * v]; // (K,V)
    let mut dl_dx = vec![0.0f64; a_n]; // (A,)
    let mut loglik = 0.0f64;
    for part in &partials {
        loglik += part.loglik;
        for i in 0..k * v {
            dl_dbeta[i] += part.dl_dbeta[i];
            dl_deta[i] += part.dl_deta[i];
        }
        for s in 0..a_n {
            dl_dx[s] += part.dl_dx[s];
        }
        let rows = part.dl_dtheta.len() / k;
        for r in 0..rows {
            let bi = part.start + r;
            dl_dtheta[bi * k..bi * k + k].copy_from_slice(&part.dl_dtheta[r * k..r * k + k]);
        }
    }

    // The whole likelihood (and thus its gradient) is SVI-scaled by D/B.
    loglik *= svi;

    // ---- assemble parameter gradients ----
    // theta/beta reparam: z=exp(u); dz/dmu=z; dz/drs = z*eps*logistic(rs).
    // theta/beta MC prior term: T = a*u - b*z + log(sig) + const, u=log z.
    //   dT/dmu = a - b*z ;  dT/drs = (a - b*z)*eps*logistic(rs) + logistic(rs)/sig.
    // eta/x reparam: z=u; dz/dmu=1; dz/drs=eps*logistic(rs).
    // eta/x -KL: d/dmu=-mu ; d/dsig=-(sig-1/sig) ; d/drs = that*logistic(rs).

    // theta (local; SVI-scaled likelihood AND prior MC term).
    let mut theta_rows: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::with_capacity(b);
    let mut elbo = loglik;
    for bi in 0..b {
        let dd = bidx[bi];
        let mut dmu = vec![0.0f64; k];
        let mut drs = vec![0.0f64; k];
        for kk in 0..k {
            let z = theta[bi * k + kk];
            let eps = eps_theta[bi * k + kk];
            let sig = sig_theta[bi * k + kk];
            let lg = logistic(p.rs_theta[dd * k + kk]);
            // likelihood part (chain dL/dz through reparam), SVI-scaled.
            let mut grad_mu;
            let mut grad_rs;
            if theta_unclamped[bi * k + kk] {
                grad_mu = svi * dl_dtheta[bi * k + kk] * z; // dz/dmu = z
                grad_rs = svi * dl_dtheta[bi * k + kk] * (z * eps * lg); // dz/drs
            } else {
                grad_mu = 0.0;
                grad_rs = 0.0;
            }
            // MC prior/entropy term T (also SVI-scaled, since theta is per-doc/local).
            let dt_dmu = a_g - b_g * z;
            let dt_drs = (a_g - b_g * z) * eps * lg + lg / sig;
            grad_mu += svi * dt_dmu;
            grad_rs += svi * dt_drs;
            dmu[kk] = grad_mu;
            drs[kk] = grad_rs;
            // ELBO MC prior contribution (lognormal-vs-gamma), SVI-scaled.
            let logz = z.max(1e-300).ln();
            let logp = (a_g - 1.0) * logz - b_g * z + a_g * b_g.ln() - ln_gamma(a_g);
            let logq = -logz - sig.ln() - 0.5 * LOG2PI - 0.5 * eps * eps;
            elbo += svi * (logp - logq);
        }
        theta_rows.push((dd, dmu, drs));
    }

    // beta (global): likelihood (already accumulated over batch; SVI-scaled like the
    // whole likelihood) + MC prior at full weight.
    let mut g_mu_beta = vec![0.0f64; k * v];
    let mut g_rs_beta = vec![0.0f64; k * v];
    for i in 0..k * v {
        let z = beta[i];
        let eps = eps_beta[i];
        let sig = sig_beta[i];
        let lg = logistic(p.rs_beta[i]);
        if beta_unclamped[i] {
            // likelihood chain through reparam, SVI-scaled.
            g_mu_beta[i] += svi * dl_dbeta[i] * z;
            g_rs_beta[i] += svi * dl_dbeta[i] * (z * eps * lg);
        }
        // MC prior term at full weight (global factor).
        let dt_dmu = a_g - b_g * z;
        let dt_drs = (a_g - b_g * z) * eps * lg + lg / sig;
        g_mu_beta[i] += dt_dmu;
        g_rs_beta[i] += dt_drs;
        let logz = z.max(1e-300).ln();
        let logp = (a_g - 1.0) * logz - b_g * z + a_g * b_g.ln() - ln_gamma(a_g);
        let logq = -logz - sig.ln() - 0.5 * LOG2PI - 0.5 * eps * eps;
        elbo += logp - logq;
    }

    // eta (global): likelihood (SVI-scaled) + analytic -KL at full weight.
    let mut g_mu_eta = vec![0.0f64; k * v];
    let mut g_rs_eta = vec![0.0f64; k * v];
    for i in 0..k * v {
        let eps = eps_eta[i];
        let sig = sig_eta[i];
        let lg = logistic(p.rs_eta[i]);
        // likelihood: dz/dmu=1, dz/drs=eps*lg.
        g_mu_eta[i] += svi * dl_deta[i];
        g_rs_eta[i] += svi * dl_deta[i] * (eps * lg);
        // -KL(N(mu,sig^2)||N(0,1)).
        let mu = p.mu_eta[i];
        g_mu_eta[i] += -mu;
        g_rs_eta[i] += -(sig - 1.0 / sig) * lg;
        elbo += -(0.5 * (sig * sig + mu * mu - 1.0) - sig.ln());
    }

    // x (global): likelihood (SVI-scaled) + analytic -KL at full weight.
    let mut g_mu_x = vec![0.0f64; a_n];
    let mut g_rs_x = vec![0.0f64; a_n];
    for s in 0..a_n {
        let eps = eps_x[s];
        let sig = sig_x[s];
        let lg = logistic(p.rs_x[s]);
        g_mu_x[s] += svi * dl_dx[s];
        g_rs_x[s] += svi * dl_dx[s] * (eps * lg);
        let mu = p.mu_x[s];
        g_mu_x[s] += -mu;
        g_rs_x[s] += -(sig - 1.0 / sig) * lg;
        elbo += -(0.5 * (sig * sig + mu * mu - 1.0) - sig.ln());
    }

    Grads {
        g_mu_beta,
        g_rs_beta,
        g_mu_eta,
        g_rs_eta,
        g_mu_x,
        g_rs_x,
        theta_rows,
        elbo,
    }
}

const LOG2PI: f64 = 1.837_877_066_409_345_6; // ln(2*pi)

/// Lanczos log-gamma (sufficient accuracy for the constant Gamma-prior term).
fn ln_gamma(x: f64) -> f64 {
    // Use the standard Lanczos approximation (g=7, n=9).
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * LOG2PI + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Fit TBIP by reparameterized SVI (Adam). Deterministic for a fixed `rng` seed:
/// one RNG drives both the minibatch sampling and the reparam noise.
pub fn fit_tbip<R: Rng>(
    docs: &[Vec<u32>],
    group: &[usize],
    num_authors: usize,
    num_topics: usize,
    num_types: usize,
    cfg: &TbipConfig,
    rng: &mut R,
) -> TbipModel {
    let k = num_topics;
    let v = num_types;
    let d = docs.len();
    let b = cfg.batch_size.min(d).max(1);

    let mut p = init_params(docs, group, num_authors, k, v, rng);

    // One Adam per parameter buffer; theta uses a full-D Adam with sparse updates.
    let mut ad_mu_theta = Adam::new(d * k, cfg.learning_rate);
    let mut ad_rs_theta = Adam::new(d * k, cfg.learning_rate);
    let mut ad_mu_beta = Adam::new(k * v, cfg.learning_rate);
    let mut ad_rs_beta = Adam::new(k * v, cfg.learning_rate);
    let mut ad_mu_eta = Adam::new(k * v, cfg.learning_rate);
    let mut ad_rs_eta = Adam::new(k * v, cfg.learning_rate);
    let mut ad_mu_x = Adam::new(num_authors, cfg.learning_rate);
    let mut ad_rs_x = Adam::new(num_authors, cfg.learning_rate);

    let steps = cfg.iters;
    let m1 = steps / 2; // halve at 50%
    let m2 = (steps * 4) / 5; // halve at 80%
    let mut elbo_history: Vec<f64> = Vec::new();
    let mut counts_buf = vec![0.0f64; v];

    for step in 0..steps {
        // LR schedule (halve at 50% and 80%).
        let lr = cfg.learning_rate
            * if step >= m2 {
                0.25
            } else if step >= m1 {
                0.5
            } else {
                1.0
            };
        for ad in [
            &mut ad_mu_theta,
            &mut ad_rs_theta,
            &mut ad_mu_beta,
            &mut ad_rs_beta,
            &mut ad_mu_eta,
            &mut ad_rs_eta,
            &mut ad_mu_x,
            &mut ad_rs_x,
        ] {
            ad.set_lr(lr);
        }

        // Minibatch (with replacement, mirroring the reference's torch.randint).
        let bidx: Vec<usize> = (0..b).map(|_| rng.gen_range(0..d)).collect();
        // Reparam noise.
        let eps_theta: Vec<f64> = (0..b * k).map(|_| randn(rng)).collect();
        let eps_beta: Vec<f64> = (0..k * v).map(|_| randn(rng)).collect();
        let eps_eta: Vec<f64> = (0..k * v).map(|_| randn(rng)).collect();
        let eps_x: Vec<f64> = (0..num_authors).map(|_| randn(rng)).collect();

        let g = elbo_and_grads(
            &p,
            docs,
            &mut counts_buf,
            group,
            &bidx,
            &eps_theta,
            &eps_beta,
            &eps_eta,
            &eps_x,
            cfg,
        );

        // Global updates.
        ad_mu_beta.step(&mut p.mu_beta, &g.g_mu_beta);
        ad_rs_beta.step(&mut p.rs_beta, &g.g_rs_beta);
        ad_mu_eta.step(&mut p.mu_eta, &g.g_mu_eta);
        ad_rs_eta.step(&mut p.rs_eta, &g.g_rs_eta);
        ad_mu_x.step(&mut p.mu_x, &g.g_mu_x);
        ad_rs_x.step(&mut p.rs_x, &g.g_rs_x);

        // Sparse theta updates: build full-D grad buffers with zeros except touched
        // rows. If a doc appears twice in the batch its grads sum (matches autodiff).
        let mut full_mu = vec![0.0f64; d * k];
        let mut full_rs = vec![0.0f64; d * k];
        let mut touched: Vec<usize> = Vec::with_capacity(b);
        for (dd, dmu, drs) in &g.theta_rows {
            for kk in 0..k {
                full_mu[dd * k + kk] += dmu[kk];
                full_rs[dd * k + kk] += drs[kk];
            }
            touched.push(*dd);
        }
        // Full Adam step with zero grad on untouched rows would advance the moment
        // estimates toward 0 on those rows; instead we step only touched rows so the
        // optimizer state for an untouched doc is frozen that step (sparse Adam).
        step_sparse(&mut ad_mu_theta, &mut p.mu_theta, &full_mu, &touched, k);
        step_sparse(&mut ad_rs_theta, &mut p.rs_theta, &full_rs, &touched, k);

        elbo_history.push(g.elbo);
        let _ = step;
    }

    TbipModel {
        num_topics: k,
        num_types: v,
        num_authors,
        params: p,
        group: group.to_vec(),
        elbo_history,
        iters_run: steps,
    }
}

/// Sparse Adam step over only the `touched` doc rows (each K-wide), to avoid moving
/// the moment estimates of untouched documents.
fn step_sparse(ad: &mut Adam, p: &mut [f64], grad: &[f64], touched: &[usize], k: usize) {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;
    ad.t += 1;
    let bc1 = 1.0 - B1.powi(ad.t as i32);
    let bc2 = 1.0 - B2.powi(ad.t as i32);
    // dedup touched rows so a doc appearing twice in the batch advances its moment
    // estimate once (its grad was already summed into `grad`).
    let mut seen = std::collections::HashSet::new();
    for &dd in touched {
        if !seen.insert(dd) {
            continue;
        }
        for kk in 0..k {
            let idx = dd * k + kk;
            let gg = grad[idx];
            ad.m[idx] = B1 * ad.m[idx] + (1.0 - B1) * gg;
            ad.v[idx] = B2 * ad.v[idx] + (1.0 - B2) * gg * gg;
            p[idx] += ad.lr * (ad.m[idx] / bc1) / ((ad.v[idx] / bc2).sqrt() + EPS);
        }
    }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for TbipModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        TbipModel::topic_word(self)
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        TbipModel::doc_topic(self)
    }
    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.elbo_history
            .iter()
            .enumerate()
            .map(|(i, &e)| (i + 1, e))
            .collect()
    }
    fn converged(&self) -> Option<bool> {
        None
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (&ai, &bi) in a.iter().zip(b) {
            cov += (ai - ma) * (bi - mb);
            va += (ai - ma) * (ai - ma);
            vb += (bi - mb) * (bi - mb);
        }
        cov / (va.sqrt() * vb.sqrt() + 1e-12)
    }

    /// Evaluate the ELBO ONLY (no grads) at a perturbed parameter buffer. Mirrors
    /// `elbo_and_grads` exactly so FD checks are consistent (same clamps, same MC
    /// sample given the same eps).
    #[allow(clippy::too_many_arguments)]
    fn elbo_only(
        p: &TbipParams,
        docs: &[Vec<u32>],
        group: &[usize],
        bidx: &[usize],
        eps_theta: &[f64],
        eps_beta: &[f64],
        eps_eta: &[f64],
        eps_x: &[f64],
        cfg: &TbipConfig,
    ) -> f64 {
        let mut buf = vec![0.0f64; p.num_types];
        let g = elbo_and_grads(
            p, docs, &mut buf, group, bidx, eps_theta, eps_beta, eps_eta, eps_x, cfg,
        );
        g.elbo
    }

    /// Build a tiny synthetic dataset for the FD check.
    fn tiny() -> (Vec<Vec<u32>>, Vec<usize>, usize, usize, usize) {
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 2, 0, 3],
            vec![1, 1, 4, 2],
            vec![3, 4, 4, 0, 1],
            vec![2, 2, 0, 4],
            vec![0, 3, 1, 1, 4],
            vec![4, 0, 2, 3],
        ];
        let group = vec![0usize, 0, 1, 1, 2, 2];
        let (a_n, k, v) = (3usize, 2usize, 5usize);
        (docs, group, a_n, k, v)
    }

    /// FD-check every parameter type. For each, perturb a handful of entries by
    /// ±h, recompute the ELBO with the SAME eps, and compare central difference to
    /// the analytic gradient. Max relative error must be < 1e-3.
    #[test]
    fn fd_gradient_check() {
        let (docs, group, a_n, k, v) = tiny();
        let cfg = TbipConfig {
            a_gamma: 0.3,
            b_gamma: 0.3,
            iters: 1,
            batch_size: 6,
            learning_rate: 0.05,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut p = init_params(&docs, &group, a_n, k, v, &mut rng);
        // Push the means off their flat init so theta/beta/eta/x all carry signal
        // (otherwise some gradients are near-degenerate).
        for m in p.mu_theta.iter_mut() {
            *m = -1.0 + 0.3 * (randn(&mut rng));
        }
        for m in p.rs_theta.iter_mut() {
            *m = -1.5 + 0.2 * (randn(&mut rng));
        }
        for m in p.mu_beta.iter_mut() {
            *m += 0.4 * (randn(&mut rng));
        }
        for m in p.rs_beta.iter_mut() {
            *m = -1.0 + 0.2 * (randn(&mut rng));
        }
        for m in p.mu_eta.iter_mut() {
            *m = 0.5 * (randn(&mut rng));
        }
        for m in p.rs_eta.iter_mut() {
            *m = -1.0 + 0.2 * (randn(&mut rng));
        }
        for m in p.mu_x.iter_mut() {
            *m = 0.7 * (randn(&mut rng));
        }
        for m in p.rs_x.iter_mut() {
            *m = -1.0 + 0.2 * (randn(&mut rng));
        }

        let bidx: Vec<usize> = (0..6).collect();
        let eps_theta: Vec<f64> = (0..6 * k).map(|_| randn(&mut rng)).collect();
        let eps_beta: Vec<f64> = (0..k * v).map(|_| randn(&mut rng)).collect();
        let eps_eta: Vec<f64> = (0..k * v).map(|_| randn(&mut rng)).collect();
        let eps_x: Vec<f64> = (0..a_n).map(|_| randn(&mut rng)).collect();

        let mut buf = vec![0.0f64; v];
        let g = elbo_and_grads(
            &p, &docs, &mut buf, &group, &bidx, &eps_theta, &eps_beta, &eps_eta, &eps_x, &cfg,
        );

        let h = 1e-5;
        let fd = |p: &mut TbipParams, get: &dyn Fn(&mut TbipParams) -> *mut f64| -> f64 {
            // central difference at the pointer
            let ptr = get(p);
            let orig = unsafe { *ptr };
            unsafe { *ptr = orig + h };
            let fp = elbo_only(
                p, &docs, &group, &bidx, &eps_theta, &eps_beta, &eps_eta, &eps_x, &cfg,
            );
            unsafe { *ptr = orig - h };
            let fm = elbo_only(
                p, &docs, &group, &bidx, &eps_theta, &eps_beta, &eps_eta, &eps_x, &cfg,
            );
            unsafe { *ptr = orig };
            (fp - fm) / (2.0 * h)
        };

        let rel = |analytic: f64, numeric: f64| -> f64 {
            (analytic - numeric).abs() / (analytic.abs().max(numeric.abs()).max(1e-6))
        };

        // Build analytic grads in dense full-D form for theta.
        let dn = docs.len();
        let mut g_mu_theta = vec![0.0f64; dn * k];
        let mut g_rs_theta = vec![0.0f64; dn * k];
        for (dd, dmu, drs) in &g.theta_rows {
            for kk in 0..k {
                g_mu_theta[dd * k + kk] += dmu[kk];
                g_rs_theta[dd * k + kk] += drs[kk];
            }
        }

        let mut worst: Vec<(&str, f64)> = Vec::new();
        let mut check = |name: &'static str,
                         analytic: &[f64],
                         idxs: &[usize],
                         p: &mut TbipParams,
                         sel: &dyn Fn(&mut TbipParams, usize) -> *mut f64| {
            let mut maxrel = 0.0f64;
            for &i in idxs {
                let num = fd(p, &|pp| sel(pp, i));
                let r = rel(analytic[i], num);
                if r > maxrel {
                    maxrel = r;
                }
            }
            worst.push((name, maxrel));
        };

        // a representative subset of indices per param type
        let some = |n: usize| -> Vec<usize> { (0..n).step_by(1).take(6).collect() };

        check("mu_theta", &g_mu_theta, &some(dn * k), &mut p, &|pp, i| {
            &mut pp.mu_theta[i] as *mut f64
        });
        check("rs_theta", &g_rs_theta, &some(dn * k), &mut p, &|pp, i| {
            &mut pp.rs_theta[i] as *mut f64
        });
        check("mu_beta", &g.g_mu_beta, &some(k * v), &mut p, &|pp, i| {
            &mut pp.mu_beta[i] as *mut f64
        });
        check("rs_beta", &g.g_rs_beta, &some(k * v), &mut p, &|pp, i| {
            &mut pp.rs_beta[i] as *mut f64
        });
        check("mu_eta", &g.g_mu_eta, &some(k * v), &mut p, &|pp, i| {
            &mut pp.mu_eta[i] as *mut f64
        });
        check("rs_eta", &g.g_rs_eta, &some(k * v), &mut p, &|pp, i| {
            &mut pp.rs_eta[i] as *mut f64
        });
        check("mu_x", &g.g_mu_x, &some(a_n), &mut p, &|pp, i| {
            &mut pp.mu_x[i] as *mut f64
        });
        check("rs_x", &g.g_rs_x, &some(a_n), &mut p, &|pp, i| {
            &mut pp.rs_x[i] as *mut f64
        });

        for (name, r) in &worst {
            println!("FD grad check {name}: max rel err = {r:.3e}");
        }
        for (name, r) in &worst {
            assert!(*r < 1e-3, "{name} FD gradient error too large: {r:.3e}");
        }
    }

    /// Sample counts from the generative model with planted ideal points, fit, and
    /// recover the positions at Pearson r > 0.9.
    #[test]
    fn synthetic_recovery() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (k, v, a_n) = (3usize, 30usize, 25usize);
        let docs_per = 8usize;
        let a_g = 0.3;
        let b_g = 0.3;

        // Planted parameters. eta carries a clear ideological signal on a subset of
        // words; beta/theta from the Gamma prior; x planted ~ uniform.
        let gamma_sample = |rng: &mut ChaCha8Rng, a: f64, b: f64| -> f64 {
            // Marsaglia-Tsang for shape a (a<1 via boosting).
            let d;
            let c;
            let boost;
            if a < 1.0 {
                let u: f64 = rng.gen::<f64>().max(1e-12);
                boost = u.powf(1.0 / a);
                d = 1.0 - 1.0 / 3.0 + a;
                c = 1.0 / (9.0 * d).sqrt();
            } else {
                boost = 1.0;
                d = a - 1.0 / 3.0;
                c = 1.0 / (9.0 * d).sqrt();
            }
            loop {
                let z: f64 = randn(rng);
                let x1 = 1.0 + c * z;
                if x1 <= 0.0 {
                    continue;
                }
                let vv = x1 * x1 * x1;
                let u: f64 = rng.gen::<f64>().max(1e-12);
                if u.ln() < 0.5 * z * z + d - d * vv + d * vv.ln() {
                    return boost * d * vv / b;
                }
            }
        };

        let mut beta = vec![vec![0.0f64; v]; k];
        for kk in 0..k {
            for vv in 0..v {
                beta[kk][vv] = gamma_sample(&mut rng, a_g, b_g) + 0.05;
            }
        }
        // eta: topic kk discriminates on words [kk*block .. (kk+1)*block) strongly.
        let mut eta = vec![vec![0.0f64; v]; k];
        let block = v / k;
        for kk in 0..k {
            for vv in 0..v {
                let base: f64 = randn(&mut rng);
                eta[kk][vv] = 0.1 * base;
            }
            for vv in (kk * block)..((kk + 1) * block) {
                eta[kk][vv] = if vv % 2 == 0 { 1.5 } else { -1.5 };
            }
        }
        let x_true: Vec<f64> = (0..a_n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        for s in 0..a_n {
            let xs = x_true[s];
            for _ in 0..docs_per {
                // per-doc theta
                let theta: Vec<f64> = (0..k)
                    .map(|_| gamma_sample(&mut rng, a_g, b_g) + 0.05)
                    .collect();
                // rate_v = sum_k theta_k beta_kv exp(xs*eta_kv)
                let mut doc: Vec<u32> = Vec::new();
                for vv in 0..v {
                    let mut rate = 0.0;
                    for kk in 0..k {
                        rate += theta[kk] * beta[kk][vv] * (xs * eta[kk][vv]).exp();
                    }
                    // Poisson(rate) via Knuth.
                    let mut count = 0u32;
                    let l = (-rate).exp();
                    let mut pp = 1.0;
                    loop {
                        pp *= rng.gen::<f64>();
                        if pp <= l {
                            break;
                        }
                        count += 1;
                        if count > 200 {
                            break;
                        }
                    }
                    for _ in 0..count {
                        doc.push(vv as u32);
                    }
                }
                if doc.is_empty() {
                    doc.push((s % v) as u32);
                }
                docs.push(doc);
                group.push(s);
            }
        }

        let cfg = TbipConfig {
            a_gamma: 0.3,
            b_gamma: 0.3,
            iters: 1500,
            batch_size: docs.len(),
            learning_rate: 0.05,
        };
        let mut fit_rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_tbip(&docs, &group, a_n, k, v, &cfg, &mut fit_rng);
        let xhat = m.ideal_points();
        let r = pearson(&xhat, &x_true).abs();
        println!("synthetic recovery Pearson r = {r:.4}");
        assert!(r > 0.9, "ideal-point recovery too low: r={r:.4}");
    }

    // The position SE is the variational posterior SD: positive, finite, and equal
    // to softplus(rs_x) + SQRT_FLOOR for every author.
    #[test]
    fn position_se_is_variational_posterior_sd() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (k, block) = (2usize, 6usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..30)
            .map(|d| {
                let bk = d % k;
                (0..10)
                    .map(|_| (bk * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let group: Vec<usize> = (0..docs.len()).map(|d| d % 6).collect();
        let cfg = TbipConfig {
            iters: 40,
            batch_size: docs.len(),
            ..Default::default()
        };
        let m = fit_tbip(&docs, &group, 6, k, v, &cfg, &mut rng);
        let se = m.position_se();
        assert_eq!(se.len(), 6);
        for (s, &sd) in se.iter().enumerate() {
            assert!(sd.is_finite() && sd > 0.0, "bad SE: {sd}");
            let expect = softplus(m.params.rs_x[s]) + SQRT_FLOOR;
            assert!((sd - expect).abs() < 1e-12, "{sd} vs {expect}");
        }
    }

    #[test]
    fn tbip_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 6usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..40)
            .map(|d| {
                let bk = d % k;
                (0..10)
                    .map(|_| (bk * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let group: Vec<usize> = (0..docs.len()).map(|d| d % 8).collect();
        let cfg = TbipConfig {
            iters: 50,
            batch_size: docs.len(),
            ..Default::default()
        };
        let m = fit_tbip(&docs, &group, 8, k, v, &cfg, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
