//! ETM, amortized VAE inference path (Dieng, Ruiz & Blei 2020).
//!
//! [`crate::etm`] fits ETM by per-document variational EM, which is accurate but
//! re-runs an optimizer for every document, so it does not scale to very large
//! corpora. This module is the reference's other inference engine: an *amortized*
//! variational autoencoder. An encoder network maps a document's normalized bag of
//! words to the variational mean and log-variance of a Gaussian, a reparameterized
//! sample is pushed through a softmax to get `theta`, and the same embedding-factored
//! decoder ([`crate::etm::softmax_beta`]) reconstructs the counts. Training is
//! minibatch Adam on the ELBO, so inference for a new document is one encoder
//! forward pass rather than an optimization.
//!
//! ```text
//!   h1 = relu(W1 xn + b1),  h2 = relu(W2 h1 + b2)          (encoder, xn = x / sum x)
//!   mu = W_mu h2 + b_mu,    logvar = W_ls h2 + b_ls         (K each)
//!   z  = mu + exp(logvar/2) * eps,  eps ~ N(0, I)           (reparameterize)
//!   theta = softmax(z),     beta = softmax_v(rho . alpha)   (decoder)
//!   loss = -sum_v x_v log( (theta beta)_v + 1e-6 )          (reconstruction)
//!          - 0.5 sum_k (1 + logvar_k - mu_k^2 - exp logvar_k)  (KL to N(0,I))
//! ```
//!
//! The reference uses torch autograd; topica has none, so the encoder forward and
//! backward are hand-coded and every gradient is checked against finite differences
//! in the unit tests. The word embeddings `rho` are fixed (caller-supplied); the
//! topic embeddings `alpha` and the encoder weights are trained jointly. At
//! inference the sample is dropped (`z = mu`), so `theta = softmax(mu)`.

use crate::etm::softmax_beta;
use crate::prodlda::{
    info_nce_backward, info_nce_loss, normal_cdf, stick_break, stick_break_dz, weibull_gamma_kl,
    weibull_gamma_kl_grad, weibull_weight, AvitmOptions, Prior,
};
use rand::Rng;

/// The implicit symmetric Dirichlet concentration for the ETM Weibull prior. ETM
/// exposes no `alpha`, so we use 1.0 (a uniform Dirichlet) on the dirichlet path.
const ETM_DIR_ALPHA: f64 = 1.0;

/// Kaiming-uniform initialization matching PyTorch's `nn.Linear` default:
/// entries are uniform on `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.
fn kaiming<R: Rng>(len: usize, fan_in: usize, rng: &mut R) -> Vec<f64> {
    let bound = 1.0 / (fan_in as f64).sqrt();
    (0..len)
        .map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * bound)
        .collect()
}

/// A standard-normal sample via Box-Muller.
fn randn<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// The inference network: `V -> hidden -> hidden -> (mu, logvar)`, each head `K`.
/// Weights are stored row-major (`w1` is `hidden x V`, `w_mu` is `K x hidden`, ...).
pub struct Encoder {
    pub v: usize,
    pub hidden: usize,
    pub k: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub w_mu: Vec<f64>,
    pub b_mu: Vec<f64>,
    pub w_ls: Vec<f64>,
    pub b_ls: Vec<f64>,
}

/// Per-document forward activations retained for the backward pass.
struct Cache {
    pre1: Vec<f64>,
    h1: Vec<f64>,
    pre2: Vec<f64>,
    h2: Vec<f64>,
    mu: Vec<f64>,
    logvar: Vec<f64>,
}

impl Encoder {
    fn new<R: Rng>(v: usize, hidden: usize, k: usize, rng: &mut R) -> Self {
        Encoder {
            v,
            hidden,
            k,
            w1: kaiming(hidden * v, v, rng),
            b1: kaiming(hidden, v, rng),
            w2: kaiming(hidden * hidden, hidden, rng),
            b2: kaiming(hidden, hidden, rng),
            w_mu: kaiming(k * hidden, hidden, rng),
            b_mu: kaiming(k, hidden, rng),
            w_ls: kaiming(k * hidden, hidden, rng),
            b_ls: kaiming(k, hidden, rng),
        }
    }

    /// Forward pass for one document, whose normalized bag of words is the sparse
    /// list `xn` of `(word_id, value)` pairs (value = count / length).
    fn forward(&self, xn: &[(usize, f64)]) -> Cache {
        let (h, k) = (self.hidden, self.k);
        // Layer 1 is sparse in the vocabulary: only the document's words contribute.
        let mut pre1 = self.b1.clone();
        for i in 0..h {
            let row = i * self.v;
            let mut s = pre1[i];
            for &(w, val) in xn {
                s += self.w1[row + w] * val;
            }
            pre1[i] = s;
        }
        let h1: Vec<f64> = pre1.iter().map(|&p| p.max(0.0)).collect();
        // Layer 2 dense.
        let mut pre2 = self.b2.clone();
        for i in 0..h {
            let row = i * h;
            let mut s = pre2[i];
            for j in 0..h {
                s += self.w2[row + j] * h1[j];
            }
            pre2[i] = s;
        }
        let h2: Vec<f64> = pre2.iter().map(|&p| p.max(0.0)).collect();
        // Heads.
        let mut mu = self.b_mu.clone();
        let mut logvar = self.b_ls.clone();
        for c in 0..k {
            let row = c * h;
            let (mut sm, mut sl) = (mu[c], logvar[c]);
            for i in 0..h {
                sm += self.w_mu[row + i] * h2[i];
                sl += self.w_ls[row + i] * h2[i];
            }
            mu[c] = sm;
            logvar[c] = sl;
        }
        Cache {
            pre1,
            h1,
            pre2,
            h2,
            mu,
            logvar,
        }
    }

    /// Inference mean only: `theta = softmax(mu)` with no sampling.
    fn encode_mean(&self, xn: &[(usize, f64)]) -> Vec<f64> {
        softmax(&self.forward(xn).mu)
    }
}

/// Gradient accumulators mirroring [`Encoder`]'s parameter blocks.
struct EncoderGrad {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w_mu: Vec<f64>,
    b_mu: Vec<f64>,
    w_ls: Vec<f64>,
    b_ls: Vec<f64>,
}

impl EncoderGrad {
    fn zeros(e: &Encoder) -> Self {
        EncoderGrad {
            w1: vec![0.0; e.w1.len()],
            b1: vec![0.0; e.b1.len()],
            w2: vec![0.0; e.w2.len()],
            b2: vec![0.0; e.b2.len()],
            w_mu: vec![0.0; e.w_mu.len()],
            b_mu: vec![0.0; e.b_mu.len()],
            w_ls: vec![0.0; e.w_ls.len()],
            b_ls: vec![0.0; e.b_ls.len()],
        }
    }
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

/// Logistic sigmoid, the derivative of softplus (used on the Weibull-Dirichlet path
/// where the Weibull shape/scale are `softplus` of the encoder heads).
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Per-document reparameterization, depending on the prior. For [`Prior::Laplace`]
/// (ETM's standard `theta = softmax(mu + s eps)`, KL to `N(0, I)`) the Weibull
/// fields are unused. For [`Prior::Dirichlet`] the topic vector is the normalized
/// Weibull `theta = g / S`, with the analytic Weibull-to-Gamma KL.
struct Reparam {
    theta: Vec<f64>,
    // Weibull scratch (dirichlet path only).
    kw: Vec<f64>,
    lam: Vec<f64>,
    ln_l: Vec<f64>,
    g: Vec<f64>,
    s: f64,
    // Stick-breaking scratch (the eta = sigmoid(z) breaks), empty off that path.
    eta: Vec<f64>,
}

/// Compute `theta` for one document under the chosen prior, given the encoder cache
/// and the per-topic reparameterization noise `eps`. On the dirichlet path the noise
/// is turned into uniforms by `Phi(eps)`; passing `noise_free = true` builds the
/// no-noise positive view (laplace: `softmax(mu)`; dirichlet: median `u = 0.5`).
fn reparam_theta(cache: &Cache, eps: &[f64], prior: Prior, noise_free: bool) -> Reparam {
    let k = cache.mu.len();
    match prior {
        Prior::Laplace => {
            let theta = if noise_free {
                softmax(&cache.mu)
            } else {
                let mut z = vec![0.0; k];
                for t in 0..k {
                    z[t] = cache.mu[t] + (0.5 * cache.logvar[t]).exp() * eps[t];
                }
                softmax(&z)
            };
            Reparam {
                theta,
                kw: vec![],
                lam: vec![],
                ln_l: vec![],
                g: vec![],
                s: 0.0,
                eta: vec![],
            }
        }
        Prior::StickBreaking => {
            // Gaussian latent (same as laplace), simplex via stick-breaking.
            let z = if noise_free {
                cache.mu.clone()
            } else {
                (0..k)
                    .map(|t| cache.mu[t] + (0.5 * cache.logvar[t]).exp() * eps[t])
                    .collect()
            };
            let (eta, theta) = stick_break(&z);
            Reparam {
                theta,
                kw: vec![],
                lam: vec![],
                ln_l: vec![],
                g: vec![],
                s: 0.0,
                eta,
            }
        }
        Prior::Dirichlet => {
            let mut kw = vec![0.0; k];
            let mut lam = vec![0.0; k];
            let mut ln_l = vec![0.0; k];
            let mut g = vec![0.0; k];
            let mut s = 0.0;
            for t in 0..k {
                let l = if noise_free {
                    (-(0.5f64).ln()).ln() // u = 0.5
                } else {
                    let u = normal_cdf(eps[t]).clamp(1e-6, 1.0 - 1e-6);
                    (-(1.0 - u).ln()).ln()
                };
                let (kwt, lamt, gt) = weibull_weight(cache.mu[t], cache.logvar[t], l);
                kw[t] = kwt;
                lam[t] = lamt;
                ln_l[t] = l;
                g[t] = gt;
                s += gt;
            }
            let theta: Vec<f64> = (0..k).map(|t| g[t] / s).collect();
            Reparam {
                theta,
                kw,
                lam,
                ln_l,
                g,
                s,
                eta: vec![],
            }
        }
    }
}

/// Backprop one document through the decoder and encoder, accumulating into the
/// encoder gradient, the beta gradient (`g_beta`, K×V), and returning the document's
/// `(recon, kl)`. `eps` is the fixed reparameterization noise (per topic). `rep`
/// holds the precomputed `theta` (and Weibull scratch) for this document under the
/// active prior. `dtheta_extra` is an optional additive gradient on `theta` (the
/// contrastive anchor term), and `dtheta_pos` is an optional gradient routed through
/// the no-noise positive view (the contrastive positive term). `beta` is precomputed.
#[allow(clippy::too_many_arguments)]
fn backward_doc(
    enc: &Encoder,
    xn: &[(usize, f64)],
    bow: &[(usize, f64)],
    cache: &Cache,
    eps: &[f64],
    rep: &Reparam,
    prior: Prior,
    dtheta_extra: Option<&[f64]>,
    dtheta_pos: Option<&[f64]>,
    beta: &[Vec<f64>],
    g: &mut EncoderGrad,
    g_beta: &mut [Vec<f64>],
) -> (f64, f64) {
    let (h, k, v) = (enc.hidden, enc.k, enc.v);
    let theta = &rep.theta;

    // Reconstruction over the document's words: m_v = sum_k theta_k beta_kv.
    let mut recon = 0.0f64;
    let mut g_theta = vec![0.0f64; k];
    for &(w, c) in bow {
        let mut m = 0.0;
        for t in 0..k {
            m += theta[t] * beta[t][w];
        }
        let mf = m + 1e-6;
        recon -= c * mf.ln();
        let g_m = -c / mf;
        for t in 0..k {
            g_theta[t] += g_m * beta[t][w];
            g_beta[t][w] += g_m * theta[t];
        }
    }
    // Contrastive anchor gradient acts on theta directly.
    if let Some(dx) = dtheta_extra {
        for t in 0..k {
            g_theta[t] += dx[t];
        }
    }

    let mut g_mu = vec![0.0f64; k];
    let mut g_logvar = vec![0.0f64; k];
    let mut kl = 0.0f64;

    match prior {
        Prior::Laplace => {
            // theta = softmax(z): g_z = theta .* (g_theta - <g_theta, theta>).
            let dot: f64 = g_theta.iter().zip(theta).map(|(&g, &t)| g * t).sum();
            let g_z: Vec<f64> = (0..k).map(|t| theta[t] * (g_theta[t] - dot)).collect();
            // z = mu + s*eps, s = exp(logvar/2). Plus the KL gradients.
            for t in 0..k {
                let s = (0.5 * cache.logvar[t]).exp();
                g_mu[t] += g_z[t];
                g_logvar[t] += g_z[t] * eps[t] * 0.5 * s;
                // KL = -0.5 (1 + logvar - mu^2 - exp logvar)
                kl += -0.5
                    * (1.0 + cache.logvar[t] - cache.mu[t] * cache.mu[t] - cache.logvar[t].exp());
                g_mu[t] += cache.mu[t];
                g_logvar[t] += 0.5 * (cache.logvar[t].exp() - 1.0);
            }
            // Contrastive positive view = softmax(mu): grad into mu only.
            if let Some(dzp) = dtheta_pos {
                let tp = softmax(&cache.mu);
                let dot: f64 = (0..k).map(|t| dzp[t] * tp[t]).sum();
                for t in 0..k {
                    g_mu[t] += tp[t] * (dzp[t] - dot);
                }
            }
        }
        Prior::StickBreaking => {
            // Gaussian latent (same reparam + N(0, I) KL as laplace); the simplex
            // map is stick-breaking, so g_z routes through its Jacobian.
            let g_z = stick_break_dz(&g_theta, &rep.eta, theta);
            for t in 0..k {
                let s = (0.5 * cache.logvar[t]).exp();
                g_mu[t] += g_z[t];
                g_logvar[t] += g_z[t] * eps[t] * 0.5 * s;
                kl += -0.5
                    * (1.0 + cache.logvar[t] - cache.mu[t] * cache.mu[t] - cache.logvar[t].exp());
                g_mu[t] += cache.mu[t];
                g_logvar[t] += 0.5 * (cache.logvar[t].exp() - 1.0);
            }
            // Contrastive positive view = stick-breaking on mu: grad into mu only.
            if let Some(dzp) = dtheta_pos {
                let (eta_pos, theta_pos) = stick_break(&cache.mu);
                let dz_pos = stick_break_dz(dzp, &eta_pos, &theta_pos);
                for t in 0..k {
                    g_mu[t] += dz_pos[t];
                }
            }
        }
        Prior::Dirichlet => {
            // theta = g / S. dg_m = (g_theta_m - sum_t g_theta_t theta_t) / S.
            let dot: f64 = (0..k).map(|t| g_theta[t] * theta[t]).sum();
            for m in 0..k {
                let dg = (g_theta[m] - dot) / rep.s;
                let dg_dlam = if rep.lam[m] != 0.0 {
                    rep.g[m] / rep.lam[m]
                } else {
                    0.0
                };
                let dg_dkw = rep.g[m] * (-rep.ln_l[m] / (rep.kw[m] * rep.kw[m]));
                g_mu[m] += dg * dg_dkw * sigmoid(cache.mu[m]);
                g_logvar[m] += dg * dg_dlam * sigmoid(cache.logvar[m]);
            }
            // KL( Weibull(kw, lam) || Gamma(alpha, 1) ).
            for t in 0..k {
                kl += weibull_gamma_kl(rep.kw[t], rep.lam[t], ETM_DIR_ALPHA);
                let (dkw, dlam) = weibull_gamma_kl_grad(rep.kw[t], rep.lam[t], ETM_DIR_ALPHA);
                g_mu[t] += dkw * sigmoid(cache.mu[t]);
                g_logvar[t] += dlam * sigmoid(cache.logvar[t]);
            }
            // Contrastive positive view = normalized Weibull at u = 0.5.
            if let Some(dzp) = dtheta_pos {
                let pos = reparam_theta(cache, eps, Prior::Dirichlet, true);
                let dot: f64 = (0..k).map(|t| dzp[t] * pos.theta[t]).sum();
                for m in 0..k {
                    let dg = (dzp[m] - dot) / pos.s;
                    let dg_dlam = if pos.lam[m] != 0.0 {
                        pos.g[m] / pos.lam[m]
                    } else {
                        0.0
                    };
                    let dg_dkw = pos.g[m] * (-pos.ln_l[m] / (pos.kw[m] * pos.kw[m]));
                    g_mu[m] += dg * dg_dkw * sigmoid(cache.mu[m]);
                    g_logvar[m] += dg * dg_dlam * sigmoid(cache.logvar[m]);
                }
            }
        }
    }

    // Heads: mu = W_mu h2 + b_mu, logvar = W_ls h2 + b_ls.
    let mut g_h2 = vec![0.0f64; h];
    for c in 0..k {
        let row = c * h;
        g.b_mu[c] += g_mu[c];
        g.b_ls[c] += g_logvar[c];
        for i in 0..h {
            g.w_mu[row + i] += g_mu[c] * cache.h2[i];
            g.w_ls[row + i] += g_logvar[c] * cache.h2[i];
            g_h2[i] += g_mu[c] * enc.w_mu[row + i] + g_logvar[c] * enc.w_ls[row + i];
        }
    }
    // relu2.
    let mut g_h1 = vec![0.0f64; h];
    for i in 0..h {
        let gp = if cache.pre2[i] > 0.0 { g_h2[i] } else { 0.0 };
        g.b2[i] += gp;
        let row = i * h;
        for j in 0..h {
            g.w2[row + j] += gp * cache.h1[j];
            g_h1[j] += gp * enc.w2[row + j];
        }
    }
    // relu1, sparse in the vocabulary.
    for i in 0..h {
        let gp = if cache.pre1[i] > 0.0 { g_h1[i] } else { 0.0 };
        g.b1[i] += gp;
        let row = i * v;
        for &(w, val) in xn {
            g.w1[row + w] += gp * val;
        }
    }

    (recon, kl)
}

/// A fitted VAE-ETM. The surface matches [`crate::etm::EtmModel`]: `beta` (K×V) is
/// the topic-word matrix, `alpha` (K×E) the topic embeddings, `doc_topic` (N×K) the
/// encoder-mean proportions for the training documents. The encoder is retained so
/// new documents transform with a single forward pass.
pub struct EtmVaeModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub beta: Vec<Vec<f64>>,
    pub alpha: Vec<Vec<f64>>,
    pub doc_topic: Vec<Vec<f64>>,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub epochs_run: usize,
    pub encoder: Encoder,
    /// The document-topic prior the model was fit under, so inference matches the
    /// training reparameterization. `Laplace` (default) keeps `theta = softmax(mu)`.
    pub prior: Prior,
}

impl EtmVaeModel {
    /// Topic proportions for new documents: one encoder forward pass per document,
    /// no sampling. Laplace uses `theta = softmax(mu)`; the Weibull-Dirichlet path
    /// uses the normalized Weibull at the posterior median (`u = 0.5`), the
    /// deterministic counterpart of its training reparameterization.
    pub fn transform(&self, docs: &[Vec<u32>]) -> Vec<Vec<f64>> {
        docs.iter()
            .map(|d| self.infer_theta(&normalized_bow(d)))
            .collect()
    }

    fn infer_theta(&self, xn: &[(usize, f64)]) -> Vec<f64> {
        match self.prior {
            Prior::Laplace => self.encoder.encode_mean(xn),
            Prior::Dirichlet => {
                let cache = self.encoder.forward(xn);
                let zeros = vec![0.0; self.num_topics];
                reparam_theta(&cache, &zeros, Prior::Dirichlet, true).theta
            }
            Prior::StickBreaking => {
                // Noise-free stick-breaking on the encoder mean, matching training.
                let cache = self.encoder.forward(xn);
                let zeros = vec![0.0; self.num_topics];
                reparam_theta(&cache, &zeros, Prior::StickBreaking, true).theta
            }
        }
    }
}

/// Sparse normalized bag of words `(word_id, count / length)`.
fn normalized_bow(doc: &[u32]) -> Vec<(usize, f64)> {
    let mut counts: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &w in doc {
        *counts.entry(w as usize).or_insert(0.0) += 1.0;
    }
    let total: f64 = counts.values().sum::<f64>().max(1.0);
    counts.into_iter().map(|(w, c)| (w, c / total)).collect()
}

/// Sparse raw bag of words `(word_id, count)`.
fn raw_bow(doc: &[u32]) -> Vec<(usize, f64)> {
    let mut counts: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &w in doc {
        *counts.entry(w as usize).or_insert(0.0) += 1.0;
    }
    counts.into_iter().collect()
}

/// Convert a beta gradient (K×V) into the topic-embedding gradient `g_alpha` (K×E),
/// back through `beta = softmax_v(rho . alpha)`. `rho` is fixed.
fn alpha_grad(g_beta: &[Vec<f64>], beta: &[Vec<f64>], rho: &[Vec<f64>], e: usize) -> Vec<Vec<f64>> {
    let k = beta.len();
    let v = rho.len();
    let mut g_alpha = vec![vec![0.0f64; e]; k];
    for t in 0..k {
        // g_eta[v] = beta[t][v] * (g_beta[t][v] - sum_j g_beta[t][j] beta[t][j]).
        let inner: f64 = (0..v).map(|j| g_beta[t][j] * beta[t][j]).sum();
        for w in 0..v {
            let g_eta = beta[t][w] * (g_beta[t][w] - inner);
            if g_eta != 0.0 {
                for d in 0..e {
                    g_alpha[t][d] += g_eta * rho[w][d];
                }
            }
        }
    }
    g_alpha
}

/// Elementwise Adam over one parameter block (coupled L2 weight decay, as in
/// torch's Adam).
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    lr: f64,
    wd: f64,
}

impl Adam {
    fn new(len: usize, lr: f64, wd: f64) -> Self {
        Adam {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
            lr,
            wd,
        }
    }
    fn step(&mut self, p: &mut [f64], grad: &[f64]) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for (pi, (&g0, (mi, vi))) in p
            .iter_mut()
            .zip(grad.iter().zip(self.m.iter_mut().zip(self.v.iter_mut())))
        {
            let g = g0 + self.wd * *pi;
            *mi = B1 * *mi + (1.0 - B1) * g;
            *vi = B2 * *vi + (1.0 - B2) * g * g;
            *pi -= self.lr * (*mi / bc1) / ((*vi / bc2).sqrt() + EPS);
        }
    }
}

/// Fit ETM by amortized VAE inference (minibatch Adam on the ELBO). `rho` is the
/// fixed V×E word embeddings; `hidden` is the encoder width (reference 800);
/// `batch_size`, `epochs`, `lr`, and `wdecay` are the Adam schedule (reference 1000,
/// 20, 0.005, 1.2e-6); `em_tol` stops on the relative change in the epoch ELBO.
#[allow(clippy::too_many_arguments)]
pub fn fit_etm_vae<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    rho: &[Vec<f64>],
    hidden: usize,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    wdecay: f64,
    em_tol: f64,
    opts: AvitmOptions,
    rng: &mut R,
) -> EtmVaeModel {
    let k = num_topics;
    let v = num_types;
    let e = if v > 0 { rho[0].len() } else { 0 };
    let d = docs.len();
    let xn: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| normalized_bow(doc)).collect();
    let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| raw_bow(doc)).collect();

    let mut enc = Encoder::new(v, hidden, k, rng);
    let mut alpha: Vec<Vec<f64>> = (0..k).map(|_| kaiming(e, e, rng)).collect();

    let mut a_w1 = Adam::new(enc.w1.len(), lr, wdecay);
    let mut a_b1 = Adam::new(enc.b1.len(), lr, wdecay);
    let mut a_w2 = Adam::new(enc.w2.len(), lr, wdecay);
    let mut a_b2 = Adam::new(enc.b2.len(), lr, wdecay);
    let mut a_wmu = Adam::new(enc.w_mu.len(), lr, wdecay);
    let mut a_bmu = Adam::new(enc.b_mu.len(), lr, wdecay);
    let mut a_wls = Adam::new(enc.w_ls.len(), lr, wdecay);
    let mut a_bls = Adam::new(enc.b_ls.len(), lr, wdecay);
    let mut a_alpha = Adam::new(k * e, lr, wdecay);

    let mut bound_history: Vec<f64> = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = 0usize;
    let mut order: Vec<usize> = (0..d).collect();

    for epoch in 0..epochs {
        epochs_run = epoch + 1;
        // Deterministic Fisher-Yates shuffle from the seeded rng.
        for i in (1..d).rev() {
            let j = (rng.gen::<f64>() * (i + 1) as f64) as usize;
            order.swap(i, j.min(i));
        }

        let mut epoch_loss = 0.0f64;
        let mut batches = 0usize;
        for chunk in order.chunks(batch_size.max(1)) {
            let beta = softmax_beta(rho, &alpha); // K×V, shared across the batch
            let mut g = EncoderGrad::zeros(&enc);
            let mut g_beta = vec![vec![0.0f64; v]; k];
            let mut batch_loss = 0.0f64;

            // Forward every document first: caches, noise, theta (anchor) and the
            // no-noise positive-view theta. The contrastive InfoNCE term couples the
            // batch, so its gradient must be computed before the per-document backward.
            let bsz = chunk.len();
            let mut caches = Vec::with_capacity(bsz);
            let mut epss = Vec::with_capacity(bsz);
            let mut reps = Vec::with_capacity(bsz);
            let mut thetas = Vec::with_capacity(bsz);
            let mut thetas_pos = Vec::with_capacity(bsz);
            for &di in chunk {
                let cache = enc.forward(&xn[di]);
                let eps: Vec<f64> = (0..k).map(|_| randn(rng)).collect();
                let rep = reparam_theta(&cache, &eps, opts.prior, false);
                thetas.push(rep.theta.clone());
                if opts.contrastive {
                    thetas_pos.push(reparam_theta(&cache, &eps, opts.prior, true).theta);
                }
                reps.push(rep);
                caches.push(cache);
                epss.push(eps);
            }

            // Contrastive InfoNCE gradients (anchor and positive views), scaled by
            // the contrastive weight. Empty when the flag is off or the batch is < 2.
            let (dz_c, dz_pos) = if opts.contrastive && bsz >= 2 {
                // `info_nce_loss`/`info_nce_backward` return a batch MEAN, but the
                // per-document recon+KL below are SUMMED and the whole batch loss +
                // gradient is divided by `bsz` before the Adam step. Scaling the
                // contrastive term only by `contrastive_weight` therefore left the
                // effective weight at `contrastive_weight / bsz` (~1/1000 at the
                // default batch size), so the advertised weight barely acted. Fold
                // in `bsz` here so it survives the shared `1/bsz` scale as the
                // intended `contrastive_weight * mean(InfoNCE)`.
                let cw = opts.contrastive_weight * bsz as f64;
                let (mut dz, dzp) = info_nce_backward(&thetas, &thetas_pos, opts.contrastive_temp);
                for row in &mut dz {
                    for x in row.iter_mut() {
                        *x *= cw;
                    }
                }
                let dzp: Vec<Vec<f64>> = dzp
                    .into_iter()
                    .map(|row| row.into_iter().map(|x| x * cw).collect())
                    .collect();
                batch_loss += cw * info_nce_loss(&thetas, &thetas_pos, opts.contrastive_temp);
                (Some(dz), Some(dzp))
            } else {
                (None, None)
            };

            for (bi, &di) in chunk.iter().enumerate() {
                let dx = dz_c.as_ref().map(|d| d[bi].as_slice());
                let dxp = dz_pos.as_ref().map(|d| d[bi].as_slice());
                let (recon, kl) = backward_doc(
                    &enc,
                    &xn[di],
                    &bows[di],
                    &caches[bi],
                    &epss[bi],
                    &reps[bi],
                    opts.prior,
                    dx,
                    dxp,
                    &beta,
                    &mut g,
                    &mut g_beta,
                );
                batch_loss += recon + kl;
            }

            // Mean over the batch.
            let bn = chunk.len() as f64;
            let scale = 1.0 / bn;
            let g_alpha = alpha_grad(&g_beta, &beta, rho, e);

            a_w1.step(&mut enc.w1, &scaled(&g.w1, scale));
            a_b1.step(&mut enc.b1, &scaled(&g.b1, scale));
            a_w2.step(&mut enc.w2, &scaled(&g.w2, scale));
            a_b2.step(&mut enc.b2, &scaled(&g.b2, scale));
            a_wmu.step(&mut enc.w_mu, &scaled(&g.w_mu, scale));
            a_bmu.step(&mut enc.b_mu, &scaled(&g.b_mu, scale));
            a_wls.step(&mut enc.w_ls, &scaled(&g.w_ls, scale));
            a_bls.step(&mut enc.b_ls, &scaled(&g.b_ls, scale));
            let mut alpha_flat: Vec<f64> = alpha.iter().flatten().copied().collect();
            let g_alpha_flat: Vec<f64> = g_alpha.iter().flatten().map(|x| x * scale).collect();
            a_alpha.step(&mut alpha_flat, &g_alpha_flat);
            for t in 0..k {
                alpha[t].copy_from_slice(&alpha_flat[t * e..(t + 1) * e]);
            }

            epoch_loss += batch_loss * scale;
            batches += 1;
        }

        let avg = epoch_loss / batches.max(1) as f64;
        bound_history.push(-avg); // report the ELBO (negative loss)
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (-avg - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }
    }

    let beta = softmax_beta(rho, &alpha);
    let model_stub = EtmVaeModel {
        num_topics: k,
        num_types: v,
        beta,
        alpha,
        doc_topic: Vec::new(),
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        epochs_run,
        encoder: enc,
        prior: opts.prior,
    };
    let doc_topic: Vec<Vec<f64>> = xn.iter().map(|x| model_stub.infer_theta(x)).collect();
    EtmVaeModel {
        doc_topic,
        ..model_stub
    }
}

fn scaled(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for EtmVaeModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.beta.clone()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic.clone()
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
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn clone_encoder(e: &Encoder) -> Encoder {
        Encoder {
            v: e.v,
            hidden: e.hidden,
            k: e.k,
            w1: e.w1.clone(),
            b1: e.b1.clone(),
            w2: e.w2.clone(),
            b2: e.b2.clone(),
            w_mu: e.w_mu.clone(),
            b_mu: e.b_mu.clone(),
            w_ls: e.w_ls.clone(),
            b_ls: e.b_ls.clone(),
        }
    }

    // Batch FD harness over a small multi-document batch (so the contrastive term is
    // active), parameterized by `AvitmOptions`. Returns the max relative error over
    // all encoder/alpha parameters; asserts each is within the 1e-4 absolute band.
    fn etm_fd_check(opts: AvitmOptions) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (v, hidden, k, e) = (6usize, 4usize, 3usize, 2usize);
        let enc0 = Encoder::new(v, hidden, k, &mut rng);
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let alpha0: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let docs: Vec<Vec<u32>> = vec![vec![0, 0, 2, 3, 3, 5], vec![1, 4, 4, 5], vec![2, 2, 3, 0]];
        let xns: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| normalized_bow(d)).collect();
        let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| raw_bow(d)).collect();
        let n = docs.len();
        let epss: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|t| 0.2 * (i as f64 + 1.0) - 0.13 * t as f64)
                    .collect()
            })
            .collect();

        // Whole-batch loss as a function of the encoder and alpha.
        let loss_only = |enc: &Encoder, alpha: &[Vec<f64>]| -> f64 {
            let beta = softmax_beta(&rho, alpha);
            let mut thetas = Vec::with_capacity(n);
            let mut thetas_pos = Vec::with_capacity(n);
            let mut total = 0.0;
            for i in 0..n {
                let cache = enc.forward(&xns[i]);
                let rep = reparam_theta(&cache, &epss[i], opts.prior, false);
                // recon
                for &(w, c) in &bows[i] {
                    let m: f64 = (0..k).map(|t| rep.theta[t] * beta[t][w]).sum::<f64>() + 1e-6;
                    total -= c * m.ln();
                }
                // kl
                match opts.prior {
                    Prior::Laplace | Prior::StickBreaking => {
                        // Both keep the Gaussian latent, so the KL is N(q || N(0, I)).
                        for t in 0..k {
                            total += -0.5
                                * (1.0 + cache.logvar[t]
                                    - cache.mu[t].powi(2)
                                    - cache.logvar[t].exp());
                        }
                    }
                    Prior::Dirichlet => {
                        for t in 0..k {
                            total += weibull_gamma_kl(rep.kw[t], rep.lam[t], ETM_DIR_ALPHA);
                        }
                    }
                }
                if opts.contrastive {
                    thetas_pos.push(reparam_theta(&cache, &epss[i], opts.prior, true).theta);
                }
                thetas.push(rep.theta);
            }
            if opts.contrastive && n >= 2 {
                total += opts.contrastive_weight
                    * info_nce_loss(&thetas, &thetas_pos, opts.contrastive_temp);
            }
            total
        };

        // Analytic gradients: forward the batch, contrastive backward, per-doc backward.
        let mut g = EncoderGrad::zeros(&enc0);
        let mut g_beta = vec![vec![0.0f64; v]; k];
        let beta = softmax_beta(&rho, &alpha0);
        let mut caches = Vec::new();
        let mut reps = Vec::new();
        let mut thetas = Vec::new();
        let mut thetas_pos = Vec::new();
        for i in 0..n {
            let cache = enc0.forward(&xns[i]);
            let rep = reparam_theta(&cache, &epss[i], opts.prior, false);
            thetas.push(rep.theta.clone());
            if opts.contrastive {
                thetas_pos.push(reparam_theta(&cache, &epss[i], opts.prior, true).theta);
            }
            reps.push(rep);
            caches.push(cache);
        }
        let (dz_c, dz_pos) = if opts.contrastive && n >= 2 {
            let (mut dz, dzp) = info_nce_backward(&thetas, &thetas_pos, opts.contrastive_temp);
            for row in &mut dz {
                for x in row.iter_mut() {
                    *x *= opts.contrastive_weight;
                }
            }
            let dzp: Vec<Vec<f64>> = dzp
                .into_iter()
                .map(|r| r.into_iter().map(|x| x * opts.contrastive_weight).collect())
                .collect();
            (Some(dz), Some(dzp))
        } else {
            (None, None)
        };
        for i in 0..n {
            let dx = dz_c.as_ref().map(|d| d[i].as_slice());
            let dxp = dz_pos.as_ref().map(|d| d[i].as_slice());
            backward_doc(
                &enc0,
                &xns[i],
                &bows[i],
                &caches[i],
                &epss[i],
                &reps[i],
                opts.prior,
                dx,
                dxp,
                &beta,
                &mut g,
                &mut g_beta,
            );
        }
        let g_alpha = alpha_grad(&g_beta, &beta, &rho, e);

        let eps_fd = 1e-6;
        let mut max_rel = 0.0f64;
        let mut chk = |analytic: f64, num: f64| {
            let abs = (analytic - num).abs();
            let denom = analytic.abs().max(num.abs());
            if denom > 1e-3 {
                max_rel = max_rel.max(abs / denom);
            }
            assert!(abs < 1e-4, "analytic {analytic} vs numeric {num}");
        };
        macro_rules! check_block {
            ($field:ident) => {
                for idx in 0..enc0.$field.len() {
                    let mut ep = clone_encoder(&enc0);
                    ep.$field[idx] += eps_fd;
                    let lp = loss_only(&ep, &alpha0);
                    ep.$field[idx] -= 2.0 * eps_fd;
                    let lm = loss_only(&ep, &alpha0);
                    chk(g.$field[idx], (lp - lm) / (2.0 * eps_fd));
                }
            };
        }
        check_block!(w1);
        check_block!(b1);
        check_block!(w2);
        check_block!(b2);
        check_block!(w_mu);
        check_block!(b_mu);
        check_block!(w_ls);
        check_block!(b_ls);
        for t in 0..k {
            for d in 0..e {
                let mut ap = alpha0.clone();
                ap[t][d] += eps_fd;
                let lp = loss_only(&enc0, &ap);
                ap[t][d] -= 2.0 * eps_fd;
                let lm = loss_only(&enc0, &ap);
                chk(g_alpha[t][d], (lp - lm) / (2.0 * eps_fd));
            }
        }
        max_rel
    }

    #[test]
    fn vae_gradients_match_fd() {
        // Default options reproduce the original Gaussian-softmax / N(0,I) path.
        etm_fd_check(AvitmOptions::default());
    }

    /// The contrastive InfoNCE term must actually move the fit at its face-value
    /// weight. It is computed as a batch MEAN, added to the SUMMED per-document
    /// recon+KL, and the whole batch loss/gradient is divided by the batch size
    /// before the Adam step — so scaling it by `contrastive_weight` alone left the
    /// effective weight at `contrastive_weight / batch_size` and it barely acted.
    /// With the batch-size correction, a real weight visibly changes `topic_word`
    /// relative to a non-contrastive fit. Measured L1 on this fixture (batch of 6):
    /// ~0.037 with the fix, ~0.008 without it — so the 0.02 bar fails the pre-fix
    /// code and clears the corrected code with margin.
    #[test]
    fn contrastive_term_influences_the_fit() {
        let (v, hidden, k, e) = (6usize, 4usize, 3usize, 2usize);
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 0, 1, 2],
            vec![3, 3, 4, 5],
            vec![0, 1, 1, 2],
            vec![3, 4, 5, 5],
            vec![0, 0, 2, 2],
            vec![3, 3, 5, 5],
        ];
        let mut r = ChaCha8Rng::seed_from_u64(7);
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|_| (0..e).map(|_| r.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let off = AvitmOptions::default();
        let on = AvitmOptions {
            contrastive: true,
            contrastive_weight: 20.0,
            contrastive_temp: 0.5,
            ..AvitmOptions::default()
        };
        // Identical fit RNG seed for both runs, so any difference is the contrastive term.
        let mut ra = ChaCha8Rng::seed_from_u64(11);
        let m_off = fit_etm_vae(
            &docs, k, v, &rho, hidden, 60, 8, 0.02, 0.0, 0.0, off, &mut ra,
        );
        let mut rb = ChaCha8Rng::seed_from_u64(11);
        let m_on = fit_etm_vae(
            &docs, k, v, &rho, hidden, 60, 8, 0.02, 0.0, 0.0, on, &mut rb,
        );
        let (a, b) = (m_off.topic_word(), m_on.topic_word());
        let l1: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum::<f64>())
            .sum();
        assert!(
            l1 > 0.02,
            "contrastive term barely influenced the fit (topic_word L1 = {l1}); \
             effective weight is likely still diluted by the batch size"
        );
    }

    #[test]
    fn etm_contrastive_gradients_match_fd() {
        let opts = AvitmOptions {
            contrastive: true,
            contrastive_weight: 0.7,
            contrastive_temp: 0.4,
            ..AvitmOptions::default()
        };
        let max_rel = etm_fd_check(opts);
        assert!(
            max_rel < 1e-4,
            "etm contrastive max relative error {max_rel}"
        );
    }

    #[test]
    fn etm_dirichlet_gradients_match_fd() {
        let opts = AvitmOptions {
            prior: Prior::Dirichlet,
            ..AvitmOptions::default()
        };
        let max_rel = etm_fd_check(opts);
        assert!(max_rel < 1e-4, "etm dirichlet max relative error {max_rel}");
    }

    #[test]
    fn etm_contrastive_and_dirichlet_compose_fd() {
        let opts = AvitmOptions {
            prior: Prior::Dirichlet,
            contrastive: true,
            contrastive_weight: 0.6,
            contrastive_temp: 0.5,
        };
        let max_rel = etm_fd_check(opts);
        assert!(
            max_rel < 1e-4,
            "etm contrastive+dirichlet max relative error {max_rel}"
        );
    }

    #[test]
    fn etm_stick_breaking_gradients_match_fd() {
        let opts = AvitmOptions {
            prior: Prior::StickBreaking,
            ..AvitmOptions::default()
        };
        let max_rel = etm_fd_check(opts);
        assert!(
            max_rel < 1e-4,
            "etm stick-breaking max relative error {max_rel}"
        );
    }

    #[test]
    fn etm_contrastive_and_stick_breaking_compose_fd() {
        let opts = AvitmOptions {
            prior: Prior::StickBreaking,
            contrastive: true,
            contrastive_weight: 0.6,
            contrastive_temp: 0.5,
        };
        let max_rel = etm_fd_check(opts);
        assert!(
            max_rel < 1e-4,
            "etm contrastive+stick-breaking max relative error {max_rel}"
        );
    }

    // Planted blocks: K word-blocks, each document drawn from one block. The VAE
    // path should recover topics whose top words come from a single block.
    #[test]
    fn fit_recovers_planted_blocks() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block, e) = (3usize, 8usize, 3usize);
        let v = k * block;
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|w| {
                let b = w / block;
                (0..e)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.2)
                    .collect()
            })
            .collect();
        let docs: Vec<Vec<u32>> = (0..150)
            .map(|d| {
                let b = d % k;
                (0..12)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();

        let m = fit_etm_vae(
            &docs,
            k,
            v,
            &rho,
            32,
            200,
            64,
            0.01,
            0.0,
            0.0,
            AvitmOptions::default(),
            &mut rng,
        );
        assert_eq!(m.beta.len(), k);
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut ord: Vec<usize> = (0..v).collect();
            ord.sort_by(|&a, &b| m.beta[t][b].total_cmp(&m.beta[t][a]));
            let blocks: std::collections::HashSet<usize> =
                ord[..4].iter().map(|&w| w / block).collect();
            assert_eq!(blocks.len(), 1, "topic {t} top words mix blocks");
            covered.insert(*blocks.iter().next().unwrap());
        }
        assert_eq!(covered.len(), k);
    }

    #[test]
    fn etm_vae_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block, e) = (3usize, 8usize, 3usize);
        let v = k * block;
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|w| {
                let b = w / block;
                (0..e)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.2)
                    .collect()
            })
            .collect();
        let docs: Vec<Vec<u32>> = (0..150)
            .map(|d| {
                let b = d % k;
                (0..12)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let m = fit_etm_vae(
            &docs,
            k,
            v,
            &rho,
            32,
            200,
            64,
            0.01,
            0.0,
            0.0,
            AvitmOptions::default(),
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
