//! FASTopic (Wu et al., NeurIPS 2024).
//!
//! FASTopic drops the encoder. There is no VAE and no neural network: the topic
//! proportions `theta` and the topic-word distribution `beta` are read directly
//! off two entropic optimal-transport plans between embedding sets. Document
//! embeddings `D` (N×H) are frozen pretrained vectors the caller brings; the only
//! learned parameters are the topic embeddings `T` (K×H), word embeddings `W`
//! (V×H), and two marginal-weight vectors `s` (K) and `u` (V).
//!
//! Each step builds two squared-Euclidean cost matrices and runs Sinkhorn on each,
//!
//! ```text
//!   C1[i,k] = ||d_i - t_k||^2,   pi  = sinkhorn(C1, a = 1/N, b = softmax(s))
//!   C2[k,j] = ||t_k - w_j||^2,   phi = sinkhorn(C2, a = 1/K, b = softmax(u))
//!   theta = N * pi    (rows sum to 1)
//!   beta  = K * phi   (rows sum to 1)
//! ```
//!
//! and minimizes the bag-of-words reconstruction plus the two transport costs,
//! weighted equally (Wu et al. Eq. 8):
//!
//! ```text
//!   L = -(1/N) sum_i x_i . log(theta_i beta) + <pi, C1> + <phi, C2>.
//! ```
//!
//! The reference differentiates this through torch's autograd over the unrolled
//! Sinkhorn iterations. topica has no autodiff, so the gradient w.r.t. `T, W, s, u`
//! is a hand-coded reverse-mode pass over a fixed number of Sinkhorn iterations
//! ([`sinkhorn_forward`] stores the dual-variable trajectory, [`Sinkhorn::backward`]
//! replays it). Every gradient here is checked against finite differences in the
//! unit tests, so the fit is the same objective by the same optimizer (Adam,
//! full-batch), not a re-derivation. Held-out documents are mapped to topics by the
//! reference's distance-softmax (Eq. 9), not by a fresh transport plan.

use rand::Rng;

/// `log(sum_i exp(z_i))`, numerically stabilized.
fn logsumexp(z: &[f64]) -> f64 {
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = z.iter().map(|&v| (v - max).exp()).sum();
    max + s.ln()
}

/// `softmax(v)`, a probability vector.
pub fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

/// Pull the gradient on a softmax output `p = softmax(logits)` back to the logits.
/// For `p_k = softmax(z)_k`, `dL/dz_l = p_l (g_p_l - sum_k g_p_k p_k)`.
fn softmax_backward(p: &[f64], g_p: &[f64]) -> Vec<f64> {
    let dot: f64 = p.iter().zip(g_p).map(|(&pk, &gk)| pk * gk).sum();
    p.iter()
        .zip(g_p)
        .map(|(&pl, &gl)| pl * (gl - dot))
        .collect()
}

/// Squared-Euclidean cost between every row of `x` (n rows) and every row of `y`
/// (m rows): `cost[i*m + k] = ||x_i - y_k||^2`, returned row-major (n×m).
pub fn pairwise_sqdist(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<f64> {
    let n = x.len();
    let m = y.len();
    let mut cost = vec![0.0f64; n * m];
    for (i, xi) in x.iter().enumerate() {
        for (k, yk) in y.iter().enumerate() {
            let d: f64 = xi.iter().zip(yk).map(|(&a, &b)| (a - b) * (a - b)).sum();
            cost[i * m + k] = d;
        }
    }
    cost
}

/// A converged (or iteration-capped) Sinkhorn solve, retaining the dual-variable
/// trajectory so [`Sinkhorn::backward`] can replay the iterations exactly.
///
/// The plan satisfies `pi 1 = a` and `pi^T 1 = b` (to the iteration tolerance),
/// with `pi[i,k] = exp(log_u[i] + log_k[i,k] + log_v[k])` and the entropic kernel
/// `log_k = -alpha * cost`.
pub struct Sinkhorn {
    pub plan: Vec<f64>,
    pub n: usize,
    pub m: usize,
    alpha: f64,
    log_k: Vec<f64>,
    // log_u[t] is the row scaling after iteration t (log_u[0] is the zero init);
    // log_v[t] is the column scaling produced in iteration t (t = 1..=iters).
    log_u_traj: Vec<Vec<f64>>,
    log_v_traj: Vec<Vec<f64>>,
}

/// Run log-domain Sinkhorn-Knopp on `cost` (n×m, row-major) toward marginals
/// `exp(log_a)` (rows) and `exp(log_b)` (columns). `alpha` is the inverse entropic
/// regularization (the reference's `DT_alpha` / `TW_alpha`); larger is sharper.
///
/// With `tol <= 0` it always runs `max_iter` iterations, which makes the solve a
/// fixed function of its inputs (used by the finite-difference gradient tests).
pub fn sinkhorn_forward(
    cost: &[f64],
    n: usize,
    m: usize,
    log_a: &[f64],
    log_b: &[f64],
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Sinkhorn {
    let log_k: Vec<f64> = cost.iter().map(|&c| -alpha * c).collect();
    let mut log_u = vec![0.0f64; n];
    let mut log_v = vec![0.0f64; m];
    let mut log_u_traj: Vec<Vec<f64>> = Vec::with_capacity(max_iter + 1);
    let mut log_v_traj: Vec<Vec<f64>> = Vec::with_capacity(max_iter);
    log_u_traj.push(log_u.clone());

    let mut col = vec![0.0f64; n];
    let mut row = vec![0.0f64; m];
    for _ in 0..max_iter {
        // log_v[k] = log_b[k] - logsumexp_i(log_k[i,k] + log_u[i])
        for k in 0..m {
            for i in 0..n {
                col[i] = log_k[i * m + k] + log_u[i];
            }
            log_v[k] = log_b[k] - logsumexp(&col);
        }
        // log_u[i] = log_a[i] - logsumexp_k(log_k[i,k] + log_v[k])
        for i in 0..n {
            for k in 0..m {
                row[k] = log_k[i * m + k] + log_v[k];
            }
            log_u[i] = log_a[i] - logsumexp(&row);
        }
        log_v_traj.push(log_v.clone());
        log_u_traj.push(log_u.clone());

        if tol > 0.0 {
            // L1 column-marginal error ||P^T 1 - b||_1. Rows are exact after the
            // u-update, so this equals the reference's stop test max(||row - a||_1,
            // ||col - b||_1) (BobXWu/FASTopic `_ETP.py`, `stop_thr`). The reference
            // default 5e-3 is reachable; the previous topica default 1e-4 was ~50x
            // tighter, below the residual floor at large V, so the topic-word plan
            // always ran to the iteration cap and stayed under-converged.
            let mut err = 0.0f64;
            for k in 0..m {
                let mut s = 0.0;
                for i in 0..n {
                    s += (log_u[i] + log_k[i * m + k] + log_v[k]).exp();
                }
                err += (s - log_b[k].exp()).abs();
            }
            if err < tol {
                break;
            }
        }
    }

    let mut plan = vec![0.0f64; n * m];
    for i in 0..n {
        for k in 0..m {
            plan[i * m + k] = (log_u[i] + log_k[i * m + k] + log_v[k]).exp();
        }
    }

    Sinkhorn {
        plan,
        n,
        m,
        alpha,
        log_k,
        log_u_traj,
        log_v_traj,
    }
}

impl Sinkhorn {
    /// Reverse-mode through the stored iterations. Given the upstream gradient on
    /// the plan, returns `(g_cost, g_log_b)`: the gradient w.r.t. the cost matrix
    /// (n×m) and the column-marginal log-weights (m). The row marginals `log_a`
    /// are a fixed uniform constant here, so their gradient is dropped.
    pub fn backward(&self, g_plan: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (n, m) = (self.n, self.m);
        let iters = self.log_v_traj.len();
        let mut g_log_k = vec![0.0f64; n * m];
        let mut g_log_b = vec![0.0f64; m];

        // Seed adjoints at the plan: log_pi = log_u_T + log_k + log_v_T.
        let mut g_log_u = vec![0.0f64; n];
        let mut g_log_v = vec![0.0f64; m];
        for i in 0..n {
            for k in 0..m {
                let p = self.plan[i * m + k];
                let g_log_pi = g_plan[i * m + k] * p;
                g_log_u[i] += g_log_pi;
                g_log_v[k] += g_log_pi;
                g_log_k[i * m + k] += g_log_pi;
            }
        }

        // Replay iterations T..1. Each did v_t = f(u_{t-1}); u_t = g(v_t).
        let mut col = vec![0.0f64; n];
        let mut row = vec![0.0f64; m];
        for t in (1..=iters).rev() {
            let log_u_prev = &self.log_u_traj[t - 1];
            let log_v_t = &self.log_v_traj[t - 1];

            // Reverse u_t[i] = log_a[i] - logsumexp_k(log_k[i,k] + log_v_t[k]).
            // P[i,k] = softmax_k(log_k[i,k] + log_v_t[k]).
            for i in 0..n {
                for k in 0..m {
                    row[k] = self.log_k[i * m + k] + log_v_t[k];
                }
                let lse = logsumexp(&row);
                let gu = g_log_u[i];
                for k in 0..m {
                    let p = (row[k] - lse).exp();
                    g_log_v[k] -= gu * p;
                    g_log_k[i * m + k] -= gu * p;
                }
                g_log_u[i] = 0.0;
            }

            // Reverse v_t[k] = log_b[k] - logsumexp_i(log_k[i,k] + log_u_prev[i]).
            // Q[i,k] = softmax_i(log_k[i,k] + log_u_prev[i]).
            for k in 0..m {
                for i in 0..n {
                    col[i] = self.log_k[i * m + k] + log_u_prev[i];
                }
                let lse = logsumexp(&col);
                let gv = g_log_v[k];
                g_log_b[k] += gv;
                for i in 0..n {
                    let q = (col[i] - lse).exp();
                    g_log_u[i] -= gv * q;
                    g_log_k[i * m + k] -= gv * q;
                }
                g_log_v[k] = 0.0;
            }
        }

        // log_k = -alpha * cost.
        let g_cost: Vec<f64> = g_log_k.iter().map(|&g| -self.alpha * g).collect();
        (g_cost, g_log_b)
    }
}

/// A fitted FASTopic model. `topic_word` (K×V) is `beta`, `doc_topic` (N×K) is
/// `theta` from the held-out distance-softmax over the fitted topic embeddings.
pub struct FastopicModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub topic_word: Vec<Vec<f64>>,
    pub doc_topic: Vec<Vec<f64>>,
    pub topic_embeddings: Vec<Vec<f64>>,
    pub word_embeddings: Vec<Vec<f64>>,
    pub train_doc_embeddings: Vec<Vec<f64>>,
    pub theta_temp: f64,
    pub loss_history: Vec<f64>,
    pub converged: bool,
    pub epochs_run: usize,
}

impl FastopicModel {
    /// Map document embeddings to topic proportions by the reference's inference
    /// rule (Eq. 9): `p_k = exp(-||t_k - d'||^2 / tau) / sum_train exp(-||t_k - d_i||^2 / tau)`,
    /// then normalize over topics. The training-document normalizer is retained in
    /// the model, so this is consistent between train and held-out documents.
    pub fn transform(&self, doc_emb: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        let tau = self.theta_temp;
        // Per-topic log-normalizer over the training documents.
        let mut log_z = vec![0.0f64; k];
        for (t, tk) in self.topic_embeddings.iter().enumerate() {
            let mut terms = Vec::with_capacity(self.train_doc_embeddings.len());
            for d in &self.train_doc_embeddings {
                let dist: f64 = tk.iter().zip(d).map(|(&a, &b)| (a - b) * (a - b)).sum();
                terms.push(-dist / tau);
            }
            log_z[t] = logsumexp(&terms);
        }
        doc_emb
            .iter()
            .map(|d| {
                let mut logp = vec![0.0f64; k];
                for (t, tk) in self.topic_embeddings.iter().enumerate() {
                    let dist: f64 = tk.iter().zip(d).map(|(&a, &b)| (a - b) * (a - b)).sum();
                    logp[t] = -dist / tau - log_z[t];
                }
                softmax(&logp)
            })
            .collect()
    }
}

/// Truncated-normal then L2-normalized embedding rows, matching the reference's
/// parameter initialization.
fn init_embeddings<R: Rng>(rows: usize, dim: usize, rng: &mut R) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            let mut v: Vec<f64> = (0..dim)
                .map(|_| {
                    // Box-Muller, truncated to +/- 2 sigma as in trunc_normal_.
                    loop {
                        let u1: f64 = rng.gen::<f64>().max(1e-12);
                        let u2: f64 = rng.gen::<f64>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        if z.abs() <= 2.0 {
                            break z;
                        }
                    }
                })
                .collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

/// Element-wise Adam state for one parameter block.
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
    fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for ((p, &g), (mi, vi)) in params
            .iter_mut()
            .zip(grad)
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            *mi = B1 * *mi + (1.0 - B1) * g;
            *vi = B2 * *vi + (1.0 - B2) * g * g;
            let m_hat = *mi / bc1;
            let v_hat = *vi / bc2;
            *p -= self.lr * m_hat / (v_hat.sqrt() + EPS);
        }
    }
}

/// Per-document sparse bag of words: `(word_id, count)` pairs.
fn doc_bow(doc: &[u32]) -> Vec<(usize, f64)> {
    let mut counts: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &w in doc {
        *counts.entry(w as usize).or_insert(0.0) += 1.0;
    }
    counts.into_iter().collect()
}

/// Evaluate the FASTopic loss and its gradient w.r.t. the four parameter blocks at
/// the current embeddings/weights. Returns `(loss, g_topic_emb, g_word_emb,
/// g_topic_logit, g_word_logit)`.
#[allow(clippy::too_many_arguments)]
fn loss_and_grad(
    bow: &[Vec<(usize, f64)>],
    doc_emb: &[Vec<f64>],
    topic_emb: &[Vec<f64>],
    word_emb: &[Vec<f64>],
    topic_logit: &[f64],
    word_logit: &[f64],
    dt_alpha: f64,
    tw_alpha: f64,
    sinkhorn_iters: usize,
    sinkhorn_tol: f64,
) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = doc_emb.len();
    let k = topic_emb.len();
    let v = word_emb.len();
    let h = if k > 0 { topic_emb[0].len() } else { 0 };

    // Forward: cost matrices, marginals, transport plans, theta and beta.
    let c1 = pairwise_sqdist(doc_emb, topic_emb); // N×K
    let c2 = pairwise_sqdist(topic_emb, word_emb); // K×V
    let b1 = softmax(topic_logit); // K
    let b2 = softmax(word_logit); // V
    let log_a1: Vec<f64> = vec![-(n as f64).ln(); n];
    let log_b1: Vec<f64> = b1.iter().map(|&p| p.max(1e-30).ln()).collect();
    let log_a2: Vec<f64> = vec![-(k as f64).ln(); k];
    let log_b2: Vec<f64> = b2.iter().map(|&p| p.max(1e-30).ln()).collect();

    let dt = sinkhorn_forward(
        &c1,
        n,
        k,
        &log_a1,
        &log_b1,
        dt_alpha,
        sinkhorn_iters,
        sinkhorn_tol,
    );
    let tw = sinkhorn_forward(
        &c2,
        k,
        v,
        &log_a2,
        &log_b2,
        tw_alpha,
        sinkhorn_iters,
        sinkhorn_tol,
    );
    let theta: Vec<f64> = dt.plan.iter().map(|&p| p * n as f64).collect(); // N×K
    let beta: Vec<f64> = tw.plan.iter().map(|&p| p * k as f64).collect(); // K×V

    // Reconstruction loss over observed (doc, word) entries.
    let mut loss = 0.0f64;
    let mut g_theta = vec![0.0f64; n * k]; // dL/dtheta
    let mut g_beta = vec![0.0f64; k * v]; // dL/dbeta
    for (i, doc) in bow.iter().enumerate() {
        for &(j, x) in doc {
            let mut r = 0.0f64;
            for t in 0..k {
                r += theta[i * k + t] * beta[t * v + j];
            }
            let r = r + 1e-12;
            loss -= (x / n as f64) * r.ln();
            let coef = -(x / n as f64) / r;
            for t in 0..k {
                g_theta[i * k + t] += coef * beta[t * v + j];
                g_beta[t * v + j] += coef * theta[i * k + t];
            }
        }
    }

    // Transport-cost terms <pi, C1> + <phi, C2>.
    let mut g_pi = vec![0.0f64; n * k];
    let mut g_phi = vec![0.0f64; k * v];
    let mut g_c1 = vec![0.0f64; n * k];
    let mut g_c2 = vec![0.0f64; k * v];
    for idx in 0..n * k {
        loss += dt.plan[idx] * c1[idx];
        g_pi[idx] += c1[idx]; // d/d pi of <pi, C1>
        g_c1[idx] += dt.plan[idx]; // explicit d/d C1
        g_pi[idx] += theta_scale_grad(g_theta[idx], n); // theta = N pi
    }
    for idx in 0..k * v {
        loss += tw.plan[idx] * c2[idx];
        g_phi[idx] += c2[idx];
        g_c2[idx] += tw.plan[idx];
        g_phi[idx] += theta_scale_grad(g_beta[idx], k); // beta = K phi
    }

    // Back through Sinkhorn to the cost matrices and column log-marginals.
    let (g_c1_sink, g_log_b1) = dt.backward(&g_pi);
    let (g_c2_sink, g_log_b2) = tw.backward(&g_phi);
    for idx in 0..n * k {
        g_c1[idx] += g_c1_sink[idx];
    }
    for idx in 0..k * v {
        g_c2[idx] += g_c2_sink[idx];
    }

    // log_b = log(softmax(logit)); chain log -> softmax -> logit.
    let g_b1: Vec<f64> = g_log_b1
        .iter()
        .zip(&b1)
        .map(|(&g, &p)| g / p.max(1e-30))
        .collect();
    let g_b2: Vec<f64> = g_log_b2
        .iter()
        .zip(&b2)
        .map(|(&g, &p)| g / p.max(1e-30))
        .collect();
    let g_topic_logit = softmax_backward(&b1, &g_b1);
    let g_word_logit = softmax_backward(&b2, &g_b2);

    // Cost matrices back to the embeddings. C1[i,k] = ||d_i - t_k||^2 (d frozen),
    // C2[k,j] = ||t_k - w_j||^2.
    let mut g_topic_emb = vec![vec![0.0f64; h]; k];
    let mut g_word_emb = vec![vec![0.0f64; h]; v];
    for i in 0..n {
        for t in 0..k {
            let g = g_c1[i * k + t];
            if g != 0.0 {
                for hh in 0..h {
                    g_topic_emb[t][hh] += g * 2.0 * (topic_emb[t][hh] - doc_emb[i][hh]);
                }
            }
        }
    }
    for t in 0..k {
        for j in 0..v {
            let g = g_c2[t * v + j];
            if g != 0.0 {
                for hh in 0..h {
                    let diff = topic_emb[t][hh] - word_emb[j][hh];
                    g_topic_emb[t][hh] += g * 2.0 * diff;
                    g_word_emb[j][hh] += g * 2.0 * (-diff);
                }
            }
        }
    }

    (loss, g_topic_emb, g_word_emb, g_topic_logit, g_word_logit)
}

/// `theta = scale * pi`, so `dL/dpi += scale * dL/dtheta`.
#[inline]
fn theta_scale_grad(g_theta: f64, scale: usize) -> f64 {
    g_theta * scale as f64
}

/// Fit FASTopic by full-batch Adam on the joint objective. `doc_emb` are the frozen
/// document embeddings (N×H); `num_types` is the vocabulary size. `dt_alpha` and
/// `tw_alpha` are the inverse entropic regularizations for the doc-topic and
/// topic-word transport (reference defaults 3.0 and 2.0); `theta_temp` is the
/// inference temperature (Eq. 9). `em_tol` stops on the relative loss change.
#[allow(clippy::too_many_arguments)]
pub fn fit_fastopic<R: Rng, F: FnMut(usize, usize, f64)>(
    docs: &[Vec<u32>],
    doc_emb: &[Vec<f64>],
    num_topics: usize,
    num_types: usize,
    epochs: usize,
    lr: f64,
    dt_alpha: f64,
    tw_alpha: f64,
    theta_temp: f64,
    em_tol: f64,
    sinkhorn_iters: usize,
    sinkhorn_tol: f64,
    mut on_progress: F,
    rng: &mut R,
) -> FastopicModel {
    let k = num_topics;
    let v = num_types;
    let h = if !doc_emb.is_empty() {
        doc_emb[0].len()
    } else {
        0
    };
    let bow: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| doc_bow(d)).collect();

    let mut topic_emb = init_embeddings(k, h, rng);
    let mut word_emb = init_embeddings(v, h, rng);
    let mut topic_logit = vec![0.0f64; k]; // uniform softmax = 1/K
    let mut word_logit = vec![0.0f64; v];

    let mut a_te = Adam::new(k * h, lr);
    let mut a_we = Adam::new(v * h, lr);
    let mut a_tl = Adam::new(k, lr);
    let mut a_wl = Adam::new(v, lr);

    let mut loss_history: Vec<f64> = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = 0usize;

    for epoch in 0..epochs {
        epochs_run = epoch + 1;
        let (loss, g_te, g_we, g_tl, g_wl) = loss_and_grad(
            &bow,
            doc_emb,
            &topic_emb,
            &word_emb,
            &topic_logit,
            &word_logit,
            dt_alpha,
            tw_alpha,
            sinkhorn_iters,
            sinkhorn_tol,
        );
        loss_history.push(loss);
        on_progress(epoch + 1, epochs, -loss);

        // Flatten, step, unflatten the two embedding blocks.
        let mut te_flat: Vec<f64> = topic_emb.iter().flatten().copied().collect();
        let g_te_flat: Vec<f64> = g_te.iter().flatten().copied().collect();
        a_te.step(&mut te_flat, &g_te_flat);
        for t in 0..k {
            topic_emb[t].copy_from_slice(&te_flat[t * h..(t + 1) * h]);
        }
        let mut we_flat: Vec<f64> = word_emb.iter().flatten().copied().collect();
        let g_we_flat: Vec<f64> = g_we.iter().flatten().copied().collect();
        a_we.step(&mut we_flat, &g_we_flat);
        for j in 0..v {
            word_emb[j].copy_from_slice(&we_flat[j * h..(j + 1) * h]);
        }
        a_tl.step(&mut topic_logit, &g_tl);
        a_wl.step(&mut word_logit, &g_wl);

        if em_tol > 0.0 && loss_history.len() >= 2 {
            let prev = loss_history[loss_history.len() - 2];
            let rel = (prev - loss).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }
    }

    // Final beta from the topic-word plan; theta from the inference rule.
    let c2 = pairwise_sqdist(&topic_emb, &word_emb);
    let b2 = softmax(&word_logit);
    let log_a2: Vec<f64> = vec![-(k as f64).ln(); k];
    let log_b2: Vec<f64> = b2.iter().map(|&p| p.max(1e-30).ln()).collect();
    let tw = sinkhorn_forward(
        &c2,
        k,
        v,
        &log_a2,
        &log_b2,
        tw_alpha,
        sinkhorn_iters,
        sinkhorn_tol,
    );
    let topic_word: Vec<Vec<f64>> = (0..k)
        .map(|t| (0..v).map(|j| tw.plan[t * v + j] * k as f64).collect())
        .collect();

    let mut model = FastopicModel {
        num_topics: k,
        num_types: v,
        topic_word,
        doc_topic: Vec::new(),
        topic_embeddings: topic_emb,
        word_embeddings: word_emb,
        train_doc_embeddings: doc_emb.to_vec(),
        theta_temp,
        loss_history,
        converged,
        epochs_run,
    };
    model.doc_topic = model.transform(doc_emb);
    model
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for FastopicModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word.clone()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic.clone()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.loss_history
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

    // Central finite difference of a scalar function at x in coordinate i.
    fn fd<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], i: usize, eps: f64) -> f64 {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += eps;
        xm[i] -= eps;
        (f(&xp) - f(&xm)) / (2.0 * eps)
    }

    #[test]
    fn softmax_backward_matches_fd() {
        let z = [0.3, -1.2, 0.8, 0.1];
        let g_p = [0.5, -0.2, 1.0, 0.3];
        // L = sum_k g_p[k] * softmax(z)[k]
        let loss = |z: &[f64]| {
            softmax(z)
                .iter()
                .zip(&g_p)
                .map(|(&p, &g)| p * g)
                .sum::<f64>()
        };
        let analytic = softmax_backward(&softmax(&z), &g_p);
        for i in 0..z.len() {
            let num = fd(loss, &z, i, 1e-6);
            assert!(
                (analytic[i] - num).abs() < 1e-7,
                "i={i}: {} vs {}",
                analytic[i],
                num
            );
        }
    }

    #[test]
    fn sinkhorn_backward_matches_fd() {
        // A scalar readout L = sum_{i,k} g_plan[i,k] * plan[i,k], differentiated
        // w.r.t. the cost matrix and the column log-marginals.
        let (n, m) = (3usize, 4usize);
        let cost = [0.2, 0.9, 0.4, 1.1, 0.7, 0.1, 0.8, 0.3, 0.5, 0.6, 0.2, 1.0];
        let log_a = vec![-(n as f64).ln(); n];
        let b = softmax(&[0.4, -0.3, 0.1, 0.7]);
        let log_b: Vec<f64> = b.iter().map(|&p| p.ln()).collect();
        let g_plan = [
            0.5, -0.2, 0.3, 0.1, 0.4, 0.0, -0.6, 0.2, 0.1, 0.3, -0.1, 0.5,
        ];
        let alpha = 2.5;
        let iters = 40;

        let fwd = sinkhorn_forward(&cost, n, m, &log_a, &log_b, alpha, iters, 0.0);
        let readout = |plan: &[f64]| plan.iter().zip(&g_plan).map(|(&p, &g)| p * g).sum::<f64>();
        let (g_cost, g_log_b) = fwd.backward(&g_plan);

        // Cost gradient.
        for i in 0..n * m {
            let f = |c: &[f64]| {
                let s = sinkhorn_forward(c, n, m, &log_a, &log_b, alpha, iters, 0.0);
                readout(&s.plan)
            };
            let num = fd(f, &cost, i, 1e-6);
            assert!(
                (g_cost[i] - num).abs() < 1e-5,
                "g_cost[{i}]: {} vs {}",
                g_cost[i],
                num
            );
        }
        // Column log-marginal gradient.
        for k in 0..m {
            let f = |lb: &[f64]| {
                let s = sinkhorn_forward(&cost, n, m, &log_a, lb, alpha, iters, 0.0);
                readout(&s.plan)
            };
            let num = fd(f, &log_b, k, 1e-6);
            assert!(
                (g_log_b[k] - num).abs() < 1e-5,
                "g_log_b[{k}]: {} vs {}",
                g_log_b[k],
                num
            );
        }
    }

    #[test]
    fn loss_grad_matches_fd() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (n, k, v, h) = (5usize, 3usize, 6usize, 4usize);
        let doc_emb: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..h).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let topic_emb = init_embeddings(k, h, &mut rng);
        let word_emb = init_embeddings(v, h, &mut rng);
        let topic_logit: Vec<f64> = (0..k).map(|_| rng.gen::<f64>() * 0.4 - 0.2).collect();
        let word_logit: Vec<f64> = (0..v).map(|_| rng.gen::<f64>() * 0.4 - 0.2).collect();
        let docs: Vec<Vec<u32>> = (0..n)
            .map(|i| (0..6).map(|j| ((i + j) % v) as u32).collect())
            .collect();
        let bow: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| doc_bow(d)).collect();
        let (dt_a, tw_a, it) = (3.0, 2.0, 30usize);

        let loss_of = |te: &[Vec<f64>], we: &[Vec<f64>], tl: &[f64], wl: &[f64]| {
            loss_and_grad(&bow, &doc_emb, te, we, tl, wl, dt_a, tw_a, it, 0.0).0
        };
        let (_, g_te, g_we, g_tl, g_wl) = loss_and_grad(
            &bow,
            &doc_emb,
            &topic_emb,
            &word_emb,
            &topic_logit,
            &word_logit,
            dt_a,
            tw_a,
            it,
            0.0,
        );

        // Topic embeddings.
        for t in 0..k {
            for hh in 0..h {
                let f = |val: f64| {
                    let mut te = topic_emb.clone();
                    te[t][hh] = val;
                    loss_of(&te, &word_emb, &topic_logit, &word_logit)
                };
                let num = (f(topic_emb[t][hh] + 1e-6) - f(topic_emb[t][hh] - 1e-6)) / 2e-6;
                assert!(
                    (g_te[t][hh] - num).abs() < 1e-4,
                    "g_te[{t}][{hh}]: {} vs {}",
                    g_te[t][hh],
                    num
                );
            }
        }
        // Word embeddings.
        for j in 0..v {
            for hh in 0..h {
                let f = |val: f64| {
                    let mut we = word_emb.clone();
                    we[j][hh] = val;
                    loss_of(&topic_emb, &we, &topic_logit, &word_logit)
                };
                let num = (f(word_emb[j][hh] + 1e-6) - f(word_emb[j][hh] - 1e-6)) / 2e-6;
                assert!(
                    (g_we[j][hh] - num).abs() < 1e-4,
                    "g_we[{j}][{hh}]: {} vs {}",
                    g_we[j][hh],
                    num
                );
            }
        }
        // Topic and word logits.
        for t in 0..k {
            let f = |val: f64| {
                let mut tl = topic_logit.clone();
                tl[t] = val;
                loss_of(&topic_emb, &word_emb, &tl, &word_logit)
            };
            let num = (f(topic_logit[t] + 1e-6) - f(topic_logit[t] - 1e-6)) / 2e-6;
            assert!(
                (g_tl[t] - num).abs() < 1e-4,
                "g_tl[{t}]: {} vs {}",
                g_tl[t],
                num
            );
        }
        for j in 0..v {
            let f = |val: f64| {
                let mut wl = word_logit.clone();
                wl[j] = val;
                loss_of(&topic_emb, &word_emb, &topic_logit, &wl)
            };
            let num = (f(word_logit[j] + 1e-6) - f(word_logit[j] - 1e-6)) / 2e-6;
            assert!(
                (g_wl[j] - num).abs() < 1e-4,
                "g_wl[{j}]: {} vs {}",
                g_wl[j],
                num
            );
        }
    }

    // Planted blocks: K word-blocks, each document drawn from one block, and the
    // document embedding placed near that block's axis. FASTopic should recover
    // topics whose top words come from a single block, covering every block.
    #[test]
    fn fit_recovers_planted_blocks() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (k, block, h) = (3usize, 6usize, 3usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..120)
            .map(|d| {
                let b = d % k;
                (0..10)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        // Document embedding: one-hot on its block axis plus noise.
        let doc_emb: Vec<Vec<f64>> = docs
            .iter()
            .map(|doc| {
                let b = doc[0] as usize / block;
                (0..h)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.3)
                    .collect()
            })
            .collect();

        let m = fit_fastopic(
            &docs,
            &doc_emb,
            k,
            v,
            300,
            0.05,
            3.0,
            2.0,
            1.0,
            1e-6,
            50,
            1e-4,
            |_, _, _| {},
            &mut rng,
        );
        assert_eq!(m.topic_word.len(), k);
        // Loss decreases overall.
        assert!(m.loss_history.last().unwrap() < &m.loss_history[0]);
        // doc_topic rows are distributions.
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Each topic's top words come from one block, and all blocks are covered.
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut order: Vec<usize> = (0..v).collect();
            order.sort_by(|&a, &b| m.topic_word[t][b].total_cmp(&m.topic_word[t][a]));
            let blocks: std::collections::HashSet<usize> =
                order[..3].iter().map(|&w| w / block).collect();
            assert_eq!(blocks.len(), 1, "topic {t} top words mix blocks");
            covered.insert(*blocks.iter().next().unwrap());
        }
        assert_eq!(covered.len(), k, "topics did not cover all {k} blocks");
    }

    #[test]
    fn fastopic_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (k, block, h) = (3usize, 6usize, 3usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..120)
            .map(|d| {
                let b = d % k;
                (0..10)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let doc_emb: Vec<Vec<f64>> = docs
            .iter()
            .map(|doc| {
                let b = doc[0] as usize / block;
                (0..h)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.3)
                    .collect()
            })
            .collect();
        let m = fit_fastopic(
            &docs,
            &doc_emb,
            k,
            v,
            300,
            0.05,
            3.0,
            2.0,
            1.0,
            1e-6,
            50,
            1e-4,
            |_, _, _| {},
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
