//! DETM: the Dynamic Embedded Topic Model (Dieng, Ruiz & Blei 2019,
//! arXiv:1907.05545). DETM extends ETM to time-stamped corpora: the topic
//! embeddings and the per-time topic prior each follow a Gaussian random walk, so
//! a topic's words drift smoothly over time.
//!
//! Generative model, for `K` topics, `T` time slices, vocabulary `V`, embedding
//! dimension `L`, word embeddings `rho` (V x L):
//!
//! ```text
//!   alpha_k^(0) ~ N(0, I);   alpha_k^(t) ~ N(alpha_k^(t-1), delta I)   (K x T x L)
//!   eta_0       ~ N(0, I);   eta_t       ~ N(eta_{t-1},     delta I)   (T x K)
//!   z_d ~ N(eta_{t_d}, I);   theta_d = softmax(z_d)                    (D x K)
//!   beta_k^(t)  = softmax_v( alpha_k^(t) . rho )                       (K x T x V)
//!   w_dn ~ Cat( theta_d . beta^(t_d) )
//! ```
//!
//! Inference is structured amortized variational inference; the ELBO is maximized
//! by minibatch Adam, with every gradient hand-coded (no autodiff crate), in the
//! style of [`crate::prodlda`] and [`crate::fastopic`]. The variational families:
//!
//! - **q(alpha)**: mean-field Gaussian per (k, t), free parameters `mu_q_alpha` and
//!   `logsigma_q_alpha` (K x T x L), exactly as the reference. `logsigma` is a
//!   log-variance; the random-walk KL uses the reference `get_kl` with prior mean
//!   `alpha_{t-1}` and log-variance `log(delta)` for `t >= 1` (unit Gaussian at
//!   `t = 0`).
//! - **q(theta_d)**: amortized encoder. Input `[normalized_bow_d, eta_{t_d}]` ->
//!   two `relu` layers (`t_hidden_size`) -> `mu`/`logsigma` heads -> reparameterized
//!   `z` -> `softmax`. KL against `N(eta_{t_d}, I)`. This matches the reference.
//! - **q(eta)**: a *direct structured Gaussian* variational treatment of the eta
//!   random walk, with free parameters `mu_q_eta`/`logsigma_q_eta` (T x K) and the
//!   same random-walk KL the reference uses (prior mean `eta_{t-1}`, log-variance
//!   `log(delta)`; unit at `t = 0`). **This is a deliberate, documented deviation
//!   from the reference**, which amortizes q(eta) with a multi-layer LSTM over the
//!   per-time normalized bag of words. The generative model and the eta KL are
//!   identical; only the way the variational mean/variance are produced differs.
//!   A direct VI treatment is strictly more flexible than the amortized one (it
//!   removes the amortization gap), it is the standard structured-Gaussian VI for a
//!   latent random walk, and it avoids hand-coding LSTM backpropagation. The cost is
//!   that q(eta) no longer generalizes to unseen corpora as a learned network; for a
//!   single fitted corpus, which is the topica use case, the per-time `eta` posterior
//!   is the same object. The choice is called out here and in the model docstring so
//!   a fidelity review can scrutinize it.
//!
//! Determinism: all randomness (initialization, the per-epoch document shuffle, the
//! reparameterization noise) is drawn from a single seeded RNG in a fixed order, so
//! a fixed `seed` reproduces the fit bit-for-bit. Training is single-threaded; there
//! is no parallel reduction to order.
//!
//! Reference implementation (MIT, Dieng/Ruiz/Blei, github.com/adjidieng/DETM) was
//! read to match algorithmic detail (the `get_kl` log-variance form, the random-walk
//! prior variances, the minibatch `num_docs/batch_size` scaling on the NLL and the
//! theta KL); it was reimplemented idiomatically in Rust, not copied.

use rand::Rng;

/// Kaiming-uniform initialization matching PyTorch's `nn.Linear` default:
/// entries uniform on `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.
fn kaiming<R: Rng>(len: usize, fan_in: usize, rng: &mut R) -> Vec<f64> {
    let bound = 1.0 / (fan_in.max(1) as f64).sqrt();
    (0..len).map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * bound).collect()
}

/// A standard-normal sample via Box-Muller.
fn randn<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

/// KL( N(q_mu, e^q_logvar) || N(p_mu, e^p_logvar) ), summed over the vector, using
/// the reference's log-variance parameterization (`detm.py::get_kl`):
///
/// ```text
///   0.5 * sum[ (sigma_q^2 + (mu_q - mu_p)^2) / (sigma_p^2 + 1e-6)
///              - 1 + log(sigma_p^2) - log(sigma_q^2) ]
/// ```
fn kl_gauss(q_mu: &[f64], q_logvar: &[f64], p_mu: &[f64], p_logvar: &[f64]) -> f64 {
    let mut kl = 0.0;
    for i in 0..q_mu.len() {
        let sq = q_logvar[i].exp();
        let sp = p_logvar[i].exp();
        let dm = q_mu[i] - p_mu[i];
        kl += (sq + dm * dm) / (sp + 1e-6) - 1.0 + p_logvar[i] - q_logvar[i];
    }
    0.5 * kl
}

/// The encoder for q(theta): `[normalized_bow, eta_td] -> h1 (relu) -> h2 (relu)
/// -> mu, logsigma`. Input width is `V + K`.
#[derive(Clone)]
pub struct ThetaEncoder {
    pub v: usize,
    pub k: usize,
    pub hidden: usize,
    pub w1: Vec<f64>, // hidden x (V + K)
    pub b1: Vec<f64>, // hidden
    pub w2: Vec<f64>, // hidden x hidden
    pub b2: Vec<f64>, // hidden
    pub w_mu: Vec<f64>, // K x hidden
    pub b_mu: Vec<f64>, // K
    pub w_ls: Vec<f64>, // K x hidden
    pub b_ls: Vec<f64>, // K
}

impl ThetaEncoder {
    fn new<R: Rng>(v: usize, k: usize, hidden: usize, rng: &mut R) -> Self {
        let inp = v + k;
        ThetaEncoder {
            v,
            k,
            hidden,
            w1: kaiming(hidden * inp, inp, rng),
            b1: vec![0.0; hidden],
            w2: kaiming(hidden * hidden, hidden, rng),
            b2: vec![0.0; hidden],
            w_mu: kaiming(k * hidden, hidden, rng),
            b_mu: vec![0.0; k],
            w_ls: kaiming(k * hidden, hidden, rng),
            b_ls: vec![0.0; k],
        }
    }

    /// Forward for one document. `xn` is the sparse normalized bag of words and
    /// `eta_td` the per-time prior mean (length K) concatenated after the vocabulary
    /// block. Returns the (mu, logsigma) heads and the cached activations needed for
    /// the backward pass.
    fn forward(&self, xn: &[(usize, f64)], eta_td: &[f64]) -> ThetaCache {
        let (h, k, v) = (self.hidden, self.k, self.v);
        // Layer 1: sparse in the vocabulary block, dense in the eta block.
        let mut pre1 = self.b1.clone();
        for i in 0..h {
            let row = i * (v + k);
            let mut s = pre1[i];
            for &(w, val) in xn {
                s += self.w1[row + w] * val;
            }
            for c in 0..k {
                s += self.w1[row + v + c] * eta_td[c];
            }
            pre1[i] = s;
        }
        let h1: Vec<f64> = pre1.iter().map(|&p| relu(p)).collect();
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
        let h2: Vec<f64> = pre2.iter().map(|&p| relu(p)).collect();
        // Heads.
        let mut mu = self.b_mu.clone();
        let mut ls = self.b_ls.clone();
        for c in 0..k {
            let row = c * h;
            let (mut sm, mut sl) = (mu[c], ls[c]);
            for i in 0..h {
                sm += self.w_mu[row + i] * h2[i];
                sl += self.w_ls[row + i] * h2[i];
            }
            mu[c] = sm;
            ls[c] = sl;
        }
        ThetaCache { pre1, h1, pre2, h2, mu, ls }
    }
}

/// Per-document encoder activations retained for the backward pass.
struct ThetaCache {
    pre1: Vec<f64>,
    h1: Vec<f64>,
    pre2: Vec<f64>,
    h2: Vec<f64>,
    mu: Vec<f64>,
    ls: Vec<f64>,
}

/// Gradient accumulators for the theta encoder.
struct EncGrad {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w_mu: Vec<f64>,
    b_mu: Vec<f64>,
    w_ls: Vec<f64>,
    b_ls: Vec<f64>,
}

impl EncGrad {
    fn zeros(e: &ThetaEncoder) -> Self {
        EncGrad {
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

/// Elementwise Adam matching torch's `Adam` (beta1=0.9, beta2=0.999) with coupled
/// L2 weight decay.
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    lr: f64,
    wd: f64,
}

impl Adam {
    fn new(len: usize, lr: f64, wd: f64) -> Self {
        Adam { m: vec![0.0; len], v: vec![0.0; len], t: 0, lr, wd }
    }
    fn step(&mut self, p: &mut [f64], grad: &[f64]) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for (pi, (&g0, (mi, vi))) in
            p.iter_mut().zip(grad.iter().zip(self.m.iter_mut().zip(self.v.iter_mut())))
        {
            let g = g0 + self.wd * *pi;
            *mi = B1 * *mi + (1.0 - B1) * g;
            *vi = B2 * *vi + (1.0 - B2) * g * g;
            *pi -= self.lr * (*mi / bc1) / ((*vi / bc2).sqrt() + EPS);
        }
    }
}

/// A fitted DETM. `beta_over_time` is `T x K x V` (each `[t][k]` a distribution over
/// the vocabulary); `doc_topic` is `D x K`; `alpha` is `T x K x L` (topic-embedding
/// trajectories, the variational means); `eta` is `T x K` (the time-varying topic
/// prior, the variational means).
pub struct DetmModel {
    pub num_topics: usize,
    pub num_times: usize,
    pub num_types: usize,
    pub beta_over_time: Vec<Vec<Vec<f64>>>, // T x K x V
    pub doc_topic: Vec<Vec<f64>>,           // D x K
    pub alpha: Vec<Vec<Vec<f64>>>,          // T x K x L
    pub eta: Vec<Vec<f64>>,                 // T x K
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub epochs_run: usize,
}

impl DetmModel {
    /// Time-collapsed topic-word matrix `K x V`: the mean of `beta` over time. This
    /// is the standard `topic_word` surface; the per-time matrices are in
    /// `beta_over_time`.
    pub fn topic_word_mean(&self) -> Vec<Vec<f64>> {
        let (k, v, t) = (self.num_topics, self.num_types, self.num_times);
        let mut out = vec![vec![0.0; v]; k];
        for bt in &self.beta_over_time {
            for kk in 0..k {
                for vv in 0..v {
                    out[kk][vv] += bt[kk][vv];
                }
            }
        }
        let inv = 1.0 / t.max(1) as f64;
        for row in &mut out {
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
        out
    }
}

/// Sparse bag of words as `(word_id, count)`, sorted by word id (deterministic).
fn raw_bow(tokens: &[u32], counts: &[u32]) -> Vec<(usize, f64)> {
    let mut m: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for (i, &w) in tokens.iter().enumerate() {
        let c = counts.get(i).copied().unwrap_or(1) as f64;
        *m.entry(w as usize).or_insert(0.0) += c;
    }
    m.into_iter().collect()
}

/// `softmax_v(alpha_k . rho)` for one topic embedding `alpha_k` (length L), with a
/// log-sum-exp for stability. Returns a length-V distribution.
fn beta_row(rho: &[Vec<f64>], alpha_k: &[f64]) -> Vec<f64> {
    let v = rho.len();
    let mut logit = vec![0.0; v];
    let mut max = f64::NEG_INFINITY;
    for (w, rv) in rho.iter().enumerate() {
        let dot: f64 = rv.iter().zip(alpha_k).map(|(r, a)| r * a).sum();
        logit[w] = dot;
        if dot > max {
            max = dot;
        }
    }
    let mut z = 0.0;
    for &e in &logit {
        z += (e - max).exp();
    }
    let logz = max + z.ln();
    logit.iter().map(|&e| (e - logz).exp()).collect()
}

/// Fit DETM by structured amortized variational inference (minibatch Adam on the
/// ELBO). See the module docs for the variational families and the documented
/// q(eta) deviation.
///
/// - `tokens[d]`/`counts[d]` are the sparse bag of words of document `d`.
/// - `times[d]` is the time-slice index of document `d` (0-based, contiguous).
/// - `rho` is the fixed word-embedding matrix (V x L).
/// - `delta` is the random-walk variance knob (reference default 0.005); the prior
///   variance for a step is `delta` (matching the reference, which sets the prior
///   log-variance to `log(delta)`).
/// - `hidden` is the theta encoder width; `epochs`/`batch_size`/`lr`/`wdecay` drive
///   Adam; `em_tol` stops on the relative change in the epoch ELBO.
#[allow(clippy::too_many_arguments)]
pub fn fit_detm<R: Rng>(
    tokens: &[Vec<u32>],
    counts: &[Vec<u32>],
    times: &[usize],
    num_topics: usize,
    num_types: usize,
    num_times: usize,
    rho: &[Vec<f64>],
    delta: f64,
    hidden: usize,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    wdecay: f64,
    em_tol: f64,
    rng: &mut R,
) -> DetmModel {
    let (k, v, t, l) = (
        num_topics,
        num_types,
        num_times,
        if num_types > 0 { rho[0].len() } else { 0 },
    );
    let d = tokens.len();
    let log_delta = delta.max(1e-12).ln();

    // Precompute the per-document sparse raw and normalized bags of words.
    let bows: Vec<Vec<(usize, f64)>> =
        (0..d).map(|i| raw_bow(&tokens[i], &counts[i])).collect();
    let totals: Vec<f64> = bows.iter().map(|b| b.iter().map(|&(_, c)| c).sum::<f64>()).collect();
    let nbows: Vec<Vec<(usize, f64)>> = bows
        .iter()
        .zip(&totals)
        .map(|(b, &tot)| {
            let s = if tot > 0.0 { tot } else { 1.0 };
            b.iter().map(|&(w, c)| (w, c / s)).collect()
        })
        .collect();

    // --- Variational parameters ------------------------------------------------
    // q(alpha): mu/logsigma per (t, k, l), stored time-major to match the reference
    // alpha layout (T x K x L). Reference inits mu/logsigma from N(0,1).
    let mut mu_alpha = vec![vec![vec![0.0f64; l]; k]; t];
    let mut ls_alpha = vec![vec![vec![0.0f64; l]; k]; t];
    for tt in 0..t {
        for kk in 0..k {
            for ll in 0..l {
                mu_alpha[tt][kk][ll] = randn(rng);
                ls_alpha[tt][kk][ll] = randn(rng);
            }
        }
    }
    // q(eta): direct structured Gaussian, mu/logsigma per (t, k).
    let mut mu_eta = vec![vec![0.0f64; k]; t];
    let mut ls_eta = vec![vec![0.0f64; k]; t];
    for tt in 0..t {
        for kk in 0..k {
            mu_eta[tt][kk] = randn(rng);
            ls_eta[tt][kk] = randn(rng);
        }
    }
    // q(theta): amortized encoder.
    let mut enc = ThetaEncoder::new(v, k, hidden, rng);

    // --- Optimizers ------------------------------------------------------------
    let mut a_mu_alpha = Adam::new(t * k * l, lr, wdecay);
    let mut a_ls_alpha = Adam::new(t * k * l, lr, wdecay);
    let mut a_mu_eta = Adam::new(t * k, lr, wdecay);
    let mut a_ls_eta = Adam::new(t * k, lr, wdecay);
    let mut a_w1 = Adam::new(enc.w1.len(), lr, wdecay);
    let mut a_b1 = Adam::new(enc.b1.len(), lr, wdecay);
    let mut a_w2 = Adam::new(enc.w2.len(), lr, wdecay);
    let mut a_b2 = Adam::new(enc.b2.len(), lr, wdecay);
    let mut a_w_mu = Adam::new(enc.w_mu.len(), lr, wdecay);
    let mut a_b_mu = Adam::new(enc.b_mu.len(), lr, wdecay);
    let mut a_w_ls = Adam::new(enc.w_ls.len(), lr, wdecay);
    let mut a_b_ls = Adam::new(enc.b_ls.len(), lr, wdecay);

    let mut bound_history: Vec<f64> = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = 0usize;
    let mut order: Vec<usize> = (0..d).collect();

    for epoch in 0..epochs {
        epochs_run = epoch + 1;
        // Deterministic Fisher-Yates shuffle from the seeded rng (matches the
        // reference's per-epoch torch.randperm over the document indices).
        for i in (1..d).rev() {
            let j = (rng.gen::<f64>() * (i + 1) as f64) as usize;
            order.swap(i, j.min(i));
        }

        let mut epoch_loss = 0.0;
        let mut batches = 0usize;

        // We process the corpus in minibatches; the global latents alpha and eta are
        // resampled per batch, as in the reference's per-forward sampling, and the
        // NLL / theta-KL are scaled by num_docs/batch_size so larger corpora behave
        // correctly while KL_alpha / KL_eta stay global.
        for chunk in order.chunks(batch_size.max(1)) {
            let n = chunk.len();
            let coeff = d as f64 / n as f64;

            // --- Sample alpha (T x K x L) along the random walk, retaining the eps
            // used so the backward pass through the reparameterization is exact. ---
            let mut eps_alpha = vec![vec![vec![0.0f64; l]; k]; t];
            let mut alpha = vec![vec![vec![0.0f64; l]; k]; t];
            for tt in 0..t {
                for kk in 0..k {
                    for ll in 0..l {
                        let e = randn(rng);
                        eps_alpha[tt][kk][ll] = e;
                        let std = (0.5 * ls_alpha[tt][kk][ll]).exp();
                        alpha[tt][kk][ll] = mu_alpha[tt][kk][ll] + std * e;
                    }
                }
            }
            // --- Sample eta (T x K). ---
            let mut eps_eta = vec![vec![0.0f64; k]; t];
            let mut eta = vec![vec![0.0f64; k]; t];
            for tt in 0..t {
                for kk in 0..k {
                    let e = randn(rng);
                    eps_eta[tt][kk] = e;
                    let std = (0.5 * ls_eta[tt][kk]).exp();
                    eta[tt][kk] = mu_eta[tt][kk] + std * e;
                }
            }

            // --- beta[t][k] = softmax(alpha[t][k] . rho). ---
            let beta: Vec<Vec<Vec<f64>>> = (0..t)
                .map(|tt| (0..k).map(|kk| beta_row(rho, &alpha[tt][kk])).collect())
                .collect();

            // Gradient accumulators.
            let mut g_mu_alpha = vec![vec![vec![0.0f64; l]; k]; t];
            let mut g_ls_alpha = vec![vec![vec![0.0f64; l]; k]; t];
            let mut g_mu_eta = vec![vec![0.0f64; k]; t];
            let mut g_ls_eta = vec![vec![0.0f64; k]; t];
            let mut g_enc = EncGrad::zeros(&enc);
            // d loss / d beta[t][k][w], accumulated over the batch's documents.
            let mut dbeta = vec![vec![vec![0.0f64; v]; k]; t];

            let mut batch_loss = 0.0;

            // --- Per-document: q(theta) encode, reparameterize, decode, NLL + KL. ---
            for &di in chunk {
                let td = times[di];
                let cache = enc.forward(&nbows[di], &eta[td]);
                // Reparameterize z = mu + exp(0.5 logsigma) * eps -> theta = softmax(z).
                let mut z = vec![0.0; k];
                let mut eps_z = vec![0.0; k];
                for c in 0..k {
                    let e = randn(rng);
                    eps_z[c] = e;
                    z[c] = cache.mu[c] + (0.5 * cache.ls[c]).exp() * e;
                }
                let theta = softmax(&z);

                // recon_v = sum_k theta_k beta[td][k][v]; NLL = -sum_v bow_v log(recon_v + 1e-6).
                // d NLL / d theta_k = -sum_v bow_v beta[td][k][v] / (recon_v + 1e-6)
                // d NLL / d beta[td][k][v] = -theta_k bow_v / (recon_v + 1e-6)
                let mut dtheta = vec![0.0; k];
                let mut nll = 0.0;
                for &(w, c) in &bows[di] {
                    let mut recon = 0.0;
                    for kk in 0..k {
                        recon += theta[kk] * beta[td][kk][w];
                    }
                    let denom = recon + 1e-6;
                    nll -= c * denom.ln();
                    let cf = c / denom;
                    for kk in 0..k {
                        dtheta[kk] -= cf * beta[td][kk][w];
                        dbeta[td][kk][w] -= cf * theta[kk] * coeff;
                    }
                }
                batch_loss += nll * coeff;

                // KL_theta = KL( N(mu, e^ls) || N(eta_td, 0) ); prior log-variance 0.
                let prior_lv = vec![0.0f64; k];
                let kl_theta = kl_gauss(&cache.mu, &cache.ls, &eta[td], &prior_lv);
                batch_loss += kl_theta * coeff;

                // --- Backward of NLL through theta = softmax(z). ---
                for x in dtheta.iter_mut() {
                    *x *= coeff;
                }
                let dot: f64 = (0..k).map(|c| dtheta[c] * theta[c]).sum();
                let dz: Vec<f64> = (0..k).map(|c| theta[c] * (dtheta[c] - dot)).collect();
                // z = mu + exp(0.5 ls) eps.
                let mut dmu = vec![0.0; k];
                let mut dls = vec![0.0; k];
                for c in 0..k {
                    let std = (0.5 * cache.ls[c]).exp();
                    dmu[c] += dz[c];
                    dls[c] += dz[c] * eps_z[c] * 0.5 * std;
                }
                // --- Backward of KL_theta (scaled by coeff) into mu, ls, and eta_td. ---
                let inv_pv = 1.0 / (1.0 + 1e-6);
                for c in 0..k {
                    let dm = cache.mu[c] - eta[td][c];
                    dmu[c] += coeff * inv_pv * dm;
                    dls[c] += coeff * 0.5 * (cache.ls[c].exp() * inv_pv - 1.0);
                    // d KL / d eta_td = -coeff * inv_pv * (mu - eta).
                    g_mu_eta[td][c] += -coeff * inv_pv * dm;
                }

                // --- Backprop dmu/dls through the encoder heads + MLP. ---
                let (h, vv) = (enc.hidden, enc.v);
                let mut dh2 = vec![0.0; h];
                for c in 0..k {
                    let row = c * h;
                    g_enc.b_mu[c] += dmu[c];
                    g_enc.b_ls[c] += dls[c];
                    for i in 0..h {
                        g_enc.w_mu[row + i] += dmu[c] * cache.h2[i];
                        g_enc.w_ls[row + i] += dls[c] * cache.h2[i];
                        dh2[i] += dmu[c] * enc.w_mu[row + i] + dls[c] * enc.w_ls[row + i];
                    }
                }
                // relu on layer 2.
                let mut dpre2 = vec![0.0; h];
                for i in 0..h {
                    dpre2[i] = if cache.pre2[i] > 0.0 { dh2[i] } else { 0.0 };
                }
                // layer 2: pre2 = W2 h1 + b2.
                let mut dh1 = vec![0.0; h];
                for a in 0..h {
                    let row = a * h;
                    g_enc.b2[a] += dpre2[a];
                    for b in 0..h {
                        g_enc.w2[row + b] += dpre2[a] * cache.h1[b];
                        dh1[b] += dpre2[a] * enc.w2[row + b];
                    }
                }
                // relu on layer 1.
                let mut dpre1 = vec![0.0; h];
                for i in 0..h {
                    dpre1[i] = if cache.pre1[i] > 0.0 { dh1[i] } else { 0.0 };
                }
                // layer 1: pre1 = W1 [nbow, eta_td] + b1.
                let inw = vv + k;
                for a in 0..h {
                    g_enc.b1[a] += dpre1[a];
                    let row = a * inw;
                    for &(w, val) in &nbows[di] {
                        g_enc.w1[row + w] += dpre1[a] * val;
                    }
                    // The eta block of the input also receives a gradient into eta_td.
                    for c in 0..k {
                        g_enc.w1[row + vv + c] += dpre1[a] * eta[td][c];
                        g_mu_eta[td][c] += dpre1[a] * enc.w1[row + vv + c];
                    }
                }
            }

            // --- Backprop dbeta into alpha via beta = softmax(alpha . rho). ---
            // For topic (t,k): logit_w = alpha . rho_w; beta = softmax(logit).
            // dlogit_w = beta_w * (dbeta_w - sum_w' dbeta_w' beta_w'); then project
            // onto rho to get d/d alpha. The reparameterization splits into mu/ls.
            for tt in 0..t {
                for kk in 0..k {
                    let bw = &beta[tt][kk];
                    let db = &dbeta[tt][kk];
                    let dot: f64 = (0..v).map(|w| db[w] * bw[w]).sum();
                    let mut dalpha = vec![0.0f64; l];
                    for w in 0..v {
                        let dlogit = bw[w] * (db[w] - dot);
                        if dlogit != 0.0 {
                            let rw = &rho[w];
                            for ll in 0..l {
                                dalpha[ll] += dlogit * rw[ll];
                            }
                        }
                    }
                    for ll in 0..l {
                        let std = (0.5 * ls_alpha[tt][kk][ll]).exp();
                        g_mu_alpha[tt][kk][ll] += dalpha[ll];
                        g_ls_alpha[tt][kk][ll] += dalpha[ll] * eps_alpha[tt][kk][ll] * 0.5 * std;
                    }
                }
            }

            // --- KL_alpha (global, random walk over time). ---
            // t = 0: prior N(0, I). t >= 1: prior N(alpha_{t-1}, delta I) with the
            // prior *mean* being the sampled alpha_{t-1} (reference uses the sample).
            for kk in 0..k {
                let pm = vec![0.0f64; l];
                let plv = vec![0.0f64; l];
                batch_loss += kl_gauss(&mu_alpha[0][kk], &ls_alpha[0][kk], &pm, &plv);
                for ll in 0..l {
                    let dm = mu_alpha[0][kk][ll];
                    g_mu_alpha[0][kk][ll] += dm / (1.0 + 1e-6);
                    g_ls_alpha[0][kk][ll] += 0.5 * (ls_alpha[0][kk][ll].exp() / (1.0 + 1e-6) - 1.0);
                }
                for tt in 1..t {
                    let prior_mean = alpha[tt - 1][kk].clone(); // sampled alpha_{t-1}
                    let plv: Vec<f64> = vec![log_delta; l];
                    batch_loss += kl_gauss(&mu_alpha[tt][kk], &ls_alpha[tt][kk], &prior_mean, &plv);
                    let inv_pv = 1.0 / (delta + 1e-6);
                    for ll in 0..l {
                        let dm = mu_alpha[tt][kk][ll] - prior_mean[ll];
                        g_mu_alpha[tt][kk][ll] += inv_pv * dm;
                        g_ls_alpha[tt][kk][ll] += 0.5 * (ls_alpha[tt][kk][ll].exp() * inv_pv - 1.0);
                        // The prior mean is the sample alpha_{t-1}, so KL pushes a
                        // gradient back onto alpha_{t-1} (its mu/ls via the eps).
                        let dprior = -inv_pv * dm;
                        let std = (0.5 * ls_alpha[tt - 1][kk][ll]).exp();
                        g_mu_alpha[tt - 1][kk][ll] += dprior;
                        g_ls_alpha[tt - 1][kk][ll] += dprior * eps_alpha[tt - 1][kk][ll] * 0.5 * std;
                    }
                }
            }

            // --- KL_eta (global, random walk over time). ---
            for kk in 0..k {
                {
                    let dm = mu_eta[0][kk];
                    batch_loss += 0.5
                        * ((ls_eta[0][kk].exp() + dm * dm) / (1.0 + 1e-6) - 1.0 + 0.0
                            - ls_eta[0][kk]);
                    g_mu_eta[0][kk] += dm / (1.0 + 1e-6);
                    g_ls_eta[0][kk] += 0.5 * (ls_eta[0][kk].exp() / (1.0 + 1e-6) - 1.0);
                }
                for tt in 1..t {
                    let prior_mean = eta[tt - 1][kk]; // sampled eta_{t-1}
                    let dm = mu_eta[tt][kk] - prior_mean;
                    let inv_pv = 1.0 / (delta + 1e-6);
                    batch_loss += 0.5
                        * ((ls_eta[tt][kk].exp() + dm * dm) * inv_pv - 1.0 + log_delta
                            - ls_eta[tt][kk]);
                    g_mu_eta[tt][kk] += inv_pv * dm;
                    g_ls_eta[tt][kk] += 0.5 * (ls_eta[tt][kk].exp() * inv_pv - 1.0);
                    let dprior = -inv_pv * dm;
                    let std = (0.5 * ls_eta[tt - 1][kk]).exp();
                    g_mu_eta[tt - 1][kk] += dprior;
                    g_ls_eta[tt - 1][kk] += dprior * eps_eta[tt - 1][kk] * 0.5 * std;
                }
            }

            // --- Adam updates. ---
            step3d(&mut a_mu_alpha, &mut mu_alpha, &g_mu_alpha, t, k, l);
            step3d(&mut a_ls_alpha, &mut ls_alpha, &g_ls_alpha, t, k, l);
            step2d(&mut a_mu_eta, &mut mu_eta, &g_mu_eta, t, k);
            step2d(&mut a_ls_eta, &mut ls_eta, &g_ls_eta, t, k);
            a_w1.step(&mut enc.w1, &g_enc.w1);
            a_b1.step(&mut enc.b1, &g_enc.b1);
            a_w2.step(&mut enc.w2, &g_enc.w2);
            a_b2.step(&mut enc.b2, &g_enc.b2);
            a_w_mu.step(&mut enc.w_mu, &g_enc.w_mu);
            a_b_mu.step(&mut enc.b_mu, &g_enc.b_mu);
            a_w_ls.step(&mut enc.w_ls, &g_enc.w_ls);
            a_b_ls.step(&mut enc.b_ls, &g_enc.b_ls);

            epoch_loss += batch_loss / coeff; // report per-doc-scaled (== /num_docs) loss
            batches += 1;
        }

        let avg = if batches > 0 { epoch_loss / batches as f64 } else { f64::NAN };
        bound_history.push(-avg); // ELBO == negative loss
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (-avg - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }
    }

    // --- Eval pass: use the variational means (no sampling), as the reference does
    // at eval time (reparameterize returns mu). ---
    let alpha_mean = mu_alpha.clone();
    let eta_mean = mu_eta.clone();
    let beta_over_time: Vec<Vec<Vec<f64>>> = (0..t)
        .map(|tt| (0..k).map(|kk| beta_row(rho, &alpha_mean[tt][kk])).collect())
        .collect();

    // doc_topic from the encoder mean (theta = softmax(mu)), as in main.py's get_theta.
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let td = times[di];
            let cache = enc.forward(&nbows[di], &eta_mean[td]);
            softmax(&cache.mu)
        })
        .collect();

    DetmModel {
        num_topics: k,
        num_times: t,
        num_types: v,
        beta_over_time,
        doc_topic,
        alpha: alpha_mean,
        eta: eta_mean,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        epochs_run,
    }
}

/// Adam step over a flattened (T x K x L) parameter block.
fn step3d(opt: &mut Adam, p: &mut [Vec<Vec<f64>>], g: &[Vec<Vec<f64>>], t: usize, k: usize, l: usize) {
    let mut flat_p = Vec::with_capacity(t * k * l);
    let mut flat_g = Vec::with_capacity(t * k * l);
    for tt in 0..t {
        for kk in 0..k {
            flat_p.extend_from_slice(&p[tt][kk]);
            flat_g.extend_from_slice(&g[tt][kk]);
        }
    }
    opt.step(&mut flat_p, &flat_g);
    let mut idx = 0;
    for tt in 0..t {
        for kk in 0..k {
            for ll in 0..l {
                p[tt][kk][ll] = flat_p[idx];
                idx += 1;
            }
        }
    }
}

/// Adam step over a flattened (T x K) parameter block.
fn step2d(opt: &mut Adam, p: &mut [Vec<f64>], g: &[Vec<f64>], t: usize, k: usize) {
    let mut flat_p = Vec::with_capacity(t * k);
    let mut flat_g = Vec::with_capacity(t * k);
    for tt in 0..t {
        flat_p.extend_from_slice(&p[tt]);
        flat_g.extend_from_slice(&g[tt]);
    }
    opt.step(&mut flat_p, &flat_g);
    let mut idx = 0;
    for tt in 0..t {
        for kk in 0..k {
            p[tt][kk] = flat_p[idx];
            idx += 1;
        }
    }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for DetmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word_mean()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic.clone()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.bound_history.iter().enumerate().map(|(i, &b)| (i + 1, b)).collect()
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

    // A small planted time-stamped corpus: K blocks of words, with topic 0 rising
    // and topic K-1 falling across time. The word embeddings put each block on its
    // own axis so beta = softmax(alpha . rho) can separate them.
    fn planted_corpus<R: Rng>(
        rng: &mut R,
        k: usize,
        block: usize,
        t: usize,
        d_per_t: usize,
    ) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>, Vec<Vec<f64>>, usize) {
        let v = k * block;
        let l = k + 2;
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|w| {
                let b = w / block;
                (0..l)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.1)
                    .collect()
            })
            .collect();
        let mut tokens = Vec::new();
        let mut counts = Vec::new();
        let mut times = Vec::new();
        for tt in 0..t {
            let frac = tt as f64 / (t - 1).max(1) as f64;
            let mut base = vec![0.25; k];
            base[0] = 0.4 * frac + 0.1;
            base[k - 1] = 0.4 * (1.0 - frac) + 0.1;
            let s: f64 = base.iter().sum();
            for b in base.iter_mut() {
                *b /= s;
            }
            for _ in 0..d_per_t {
                let mut wc = vec![0u32; v];
                let length = 30;
                for _ in 0..length {
                    let r: f64 = rng.gen();
                    let mut acc = 0.0;
                    let mut kk = 0;
                    for (ii, &bp) in base.iter().enumerate() {
                        acc += bp;
                        if r <= acc {
                            kk = ii;
                            break;
                        }
                    }
                    let w = kk * block + (rng.gen::<f64>() * block as f64) as usize;
                    wc[w.min(v - 1)] += 1;
                }
                let toks: Vec<u32> = (0..v as u32).filter(|&w| wc[w as usize] > 0).collect();
                let cnts: Vec<u32> = toks.iter().map(|&w| wc[w as usize]).collect();
                tokens.push(toks);
                counts.push(cnts);
                times.push(tt);
            }
        }
        (tokens, counts, times, rho, v)
    }

    #[test]
    fn beta_rows_are_distributions() {
        let rho = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let row = beta_row(&rho, &[2.0, -1.0]);
        assert_eq!(row.len(), 3);
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(row.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn kl_zero_when_equal() {
        let mu = vec![0.3, -0.5, 1.0];
        let lv = vec![0.1, -0.2, 0.0];
        let kl = kl_gauss(&mu, &lv, &mu, &lv);
        // Not exactly zero: the reference's get_kl adds 1e-6 to the prior variance
        // for numerical safety, so KL(q || q) is a tiny negative number. We match
        // that form, so we only require it is close to zero.
        assert!(kl.abs() < 1e-4, "KL of a distribution with itself should be ~0, got {kl}");
    }

    #[test]
    fn fit_detm_shapes_and_distributions() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (k, block, t, d_per_t) = (3usize, 6usize, 4usize, 20usize);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng, k, block, t, d_per_t);
        let m = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 32, 30, 64, 0.02, 1.2e-6, 0.0, &mut rng,
        );
        assert_eq!(m.num_topics, k);
        assert_eq!(m.num_times, t);
        assert_eq!(m.beta_over_time.len(), t);
        assert_eq!(m.beta_over_time[0].len(), k);
        assert_eq!(m.beta_over_time[0][0].len(), v);
        for bt in &m.beta_over_time {
            for row in bt {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
        }
        assert_eq!(m.doc_topic.len(), tokens.len());
        for row in &m.doc_topic {
            assert_eq!(row.len(), k);
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.topic_word_mean() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert_eq!(m.eta.len(), t);
        assert_eq!(m.eta[0].len(), k);
        assert_eq!(m.alpha.len(), t);
        assert_eq!(m.alpha[0].len(), k);
    }

    #[test]
    fn fit_detm_is_deterministic() {
        let (k, block, t, d_per_t) = (3usize, 6usize, 4usize, 15usize);
        let mut rng0 = ChaCha8Rng::seed_from_u64(7);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng0, k, block, t, d_per_t);

        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let a = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 16, 20, 64, 0.02, 1.2e-6, 0.0, &mut rng_a,
        );
        let mut rng_b = ChaCha8Rng::seed_from_u64(123);
        let b = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 16, 20, 64, 0.02, 1.2e-6, 0.0, &mut rng_b,
        );
        for tt in 0..t {
            for kk in 0..k {
                assert_eq!(a.beta_over_time[tt][kk], b.beta_over_time[tt][kk]);
            }
        }
        assert_eq!(a.doc_topic, b.doc_topic);
        assert_eq!(a.eta, b.eta);
    }

    #[test]
    fn fit_detm_recovers_temporal_drift() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (k, block, t, d_per_t) = (3usize, 8usize, 5usize, 40usize);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng, k, block, t, d_per_t);
        let m = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 32, 120, 1000, 0.02, 1.2e-6, 0.0,
            &mut rng,
        );
        // The time-varying topic prior eta (the latent the random walk regularizes)
        // should move across the slices for at least one topic, recovering the
        // planted prevalence drift (topic 0 rises, topic k-1 falls). softmax(eta_t)
        // is the prior over topic proportions at slice t; we check its first-to-last
        // change.
        let prior_at = |tt: usize| softmax(&m.eta[tt]);
        let p0 = prior_at(0);
        let plast = prior_at(t - 1);
        let max_drift = (0..k).map(|kk| (plast[kk] - p0[kk]).abs()).fold(0.0f64, f64::max);
        assert!(max_drift > 0.03, "eta prior did not drift across time (max {max_drift})");
    }

    #[test]
    fn detm_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block, t, d_per_t) = (3usize, 6usize, 4usize, 20usize);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng, k, block, t, d_per_t);
        let m = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 16, 25, 64, 0.02, 1.2e-6, 0.0, &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
