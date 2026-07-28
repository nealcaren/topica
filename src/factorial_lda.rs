//! Factorial LDA (fLDA): sparse multi-dimensional topic model (Paul & Dredze,
//! NIPS 2012). Each token is assigned a K-tuple `(t_1, .., t_K)` — one component
//! from each of K factors (e.g. topic x sentiment) — rather than a single topic.
//!
//! The factorial structure comes from log-linear structured Dirichlet priors that
//! tie tuples sharing a component: a SAGE-like word prior `omega` (background +
//! per-factor-component deviations over the vocabulary) and a DMR-like,
//! document-specific document prior `alpha`, plus a relaxed group-lasso sparsity
//! mask `b = sigma(beta)` over the tuple set. Inference is Monte Carlo EM: a
//! collapsed Gibbs sweep over tuples, then one gradient-ascent step on the
//! log-linear weights (the Dirichlet-multinomial compound derivative, the same
//! digamma form as `crate::dmr`).
//!
//! Implemented from the paper's mathematics (the reference Java is GPL v2; it was
//! read only to disambiguate defaults/update-order, not ported). See
//! `docs/guides/models.md` and issue #606.
//!
//! Determinism: a single seeded RNG is threaded through the sampler in document,
//! then token order, so a fixed seed reproduces bit-for-bit. (The reference builds
//! a fresh unseeded `Random` per token and is not reproducible; matching the
//! *method*, collapsed Gibbs, not that RNG bug.)

use crate::corpus::Corpus;
use crate::estimator::{Estimator, ModelFamily};
use crate::optimize::digamma;
use rand::Rng;

/// Prior pseudocounts are floored to this ridge before entering `digamma`, so the
/// sparsity mask driving a tuple's prior toward 0 cannot produce `digamma(0) = -inf`
/// and NaN gradients (Gate A blocker; mirrors `dmr.rs`'s `1e-10` alpha floor). The
/// same floored value is used as both the digamma argument and the outer multiplier,
/// which also yields the correct `p*(psi(p+n) - psi(p)) -> {0,1}` limit as `p -> 0`.
const PRIOR_RIDGE: f64 = 1e-10;

/// Hyperparameters for [`fit`]. Names match the Python binding / spec; defaults are
/// the reference's `LearnTopicModel.java` code defaults.
#[derive(Clone)]
pub struct FactorialLdaConfig {
    /// Number of components per factor, `Z_k`. `factor_sizes.len()` is K.
    pub factor_sizes: Vec<usize>,
    pub iters: usize,
    /// Tail samples averaged for the posterior-mean topic-word / doc-topic.
    pub samples: usize,
    pub sigma_alpha: f64,
    pub sigma_alpha_bias: f64,
    pub sigma_omega: f64,
    pub sigma_omega_bias: f64,
    pub delta0: f64,
    pub delta1: f64,
    pub alpha_bias_init: f64,
    pub omega_bias_init: f64,
    pub step_alpha_doc: f64,
    pub step_alpha_corpus: f64,
    pub step_alpha_bias: f64,
    pub step_omega: f64,
    pub step_omega_bias: f64,
    pub step_beta: f64,
    /// Block-sample every `block_freq` iterations (1 = always block, the faithful
    /// default); otherwise sample each factor independently (additive cost).
    pub block_freq: usize,
    /// Iterations before omega/beta start updating (the reference's hard-coded 100).
    pub weight_burnin: usize,
    /// Ablations (paper's base/W/S/SW): structured word priors on/off.
    pub word_priors: bool,
    /// Tuple sparsity on/off (off fixes b = 1).
    pub sparsity: bool,
    /// Symmetric word prior: fix the per-word background `omega_w = 0` (paper's
    /// interpretability setting), optimizing only the bias `omega_b`.
    pub symmetric_word_prior: bool,
    /// Compute the marginal log-likelihood every `eval_every` iters (0 = only at the
    /// end) for the fit history. Does not affect the model, only the trace.
    pub eval_every: usize,
}

impl Default for FactorialLdaConfig {
    fn default() -> Self {
        let step_alpha_doc = 1e-2;
        let step_omega = 1e-3;
        FactorialLdaConfig {
            factor_sizes: vec![],
            iters: 2000,
            samples: 100,
            sigma_alpha: 1.0,
            sigma_alpha_bias: 1.0,
            sigma_omega: 0.5,
            sigma_omega_bias: 10.0,
            delta0: 0.1,
            delta1: 0.1,
            alpha_bias_init: -5.0,
            omega_bias_init: -5.0,
            step_alpha_doc,
            step_alpha_corpus: step_alpha_doc / 100.0,
            step_alpha_bias: step_alpha_doc / 100.0,
            step_omega,
            step_omega_bias: step_omega / 100.0,
            step_beta: 1e-3,
            block_freq: 1,
            weight_burnin: 100,
            word_priors: true,
            sparsity: true,
            symmetric_word_prior: false,
            eval_every: 0,
        }
    }
}

/// Informed omega prior means (the NAACL-2013 `priorPrefix` feature). `eta_w` is a
/// length-V background mean; `eta_zw[k][z]` is a length-V mean for component z of
/// factor k. All default to 0. Omega is *initialized* at these means and its L2
/// regularization pulls back toward them.
#[derive(Clone, Default)]
pub struct OmegaPriors {
    pub eta_w: Vec<f64>,
    pub eta_zw: Vec<Vec<Vec<f64>>>,
}

/// Fitted state read back by the PyO3 binding.
pub struct FactorialLDAModel {
    pub num_topics: usize, // Z0 = number of tuples
    pub factor_sizes: Vec<usize>,
    pub tuples: Vec<Vec<usize>>, // tuple index -> per-factor component vector
    pub topic_word: Vec<Vec<f64>>, // Z0 x V (phi)
    pub doc_topic: Vec<Vec<f64>>, // D x Z0 (theta)
    pub omega_b: f64,
    pub omega_w: Vec<f64>,            // V
    pub omega_zw: Vec<Vec<Vec<f64>>>, // K x Z_k x V
    pub tuple_activity: Vec<f64>,     // Z0, b_x = sigma(beta_x)
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

#[inline]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// digamma with the prior ridge floor applied (Gate A NaN guard).
#[inline]
fn dg(x: f64) -> f64 {
    digamma(x.max(PRIOR_RIDGE))
}

fn zeros_kz(factor_sizes: &[usize]) -> Vec<Vec<f64>> {
    factor_sizes.iter().map(|&zk| vec![0.0; zk]).collect()
}

/// Analytic gradient of the penalized log-likelihood w.r.t. the alpha weights.
struct AlphaGrad {
    b: f64,
    z: Vec<Vec<f64>>,       // K x Z_k
    dz: Vec<Vec<Vec<f64>>>, // K x Z_k x D
    beta: Vec<f64>,         // Z0
}

/// Analytic gradient w.r.t. the omega weights.
struct OmegaGrad {
    b: f64,
    w: Vec<f64>,            // V
    zw: Vec<Vec<Vec<f64>>>, // K x Z_k x V
}

/// Mixed-radix helpers between a flat tuple index and its per-factor vector.
struct TupleIndex {
    factor_sizes: Vec<usize>,
    sub: Vec<usize>, // sub[k] = product of factor_sizes[j] for j > k
    z0: usize,
}

impl TupleIndex {
    fn new(factor_sizes: &[usize]) -> Self {
        let k = factor_sizes.len();
        let mut sub = vec![0usize; k];
        let mut z0 = 1usize;
        for i in (0..k).rev() {
            sub[i] = z0;
            z0 *= factor_sizes[i];
        }
        TupleIndex {
            factor_sizes: factor_sizes.to_vec(),
            sub,
            z0,
        }
    }
    fn to_vector(&self, mut x: usize) -> Vec<usize> {
        let k = self.factor_sizes.len();
        let mut z = vec![0usize; k];
        for i in 0..k {
            z[i] = x / self.sub[i];
            x %= self.sub[i];
        }
        z
    }
    fn to_index(&self, z: &[usize]) -> usize {
        z.iter().enumerate().map(|(i, &zi)| zi * self.sub[i]).sum()
    }
}

/// Internal MCEM state.
struct State<'a> {
    cfg: &'a FactorialLdaConfig,
    ti: TupleIndex,
    docs: &'a [Vec<u32>],
    d: usize,
    v: usize,
    k: usize,
    z0: usize,
    tuple_vecs: Vec<Vec<usize>>,

    // counts
    n_dz: Vec<Vec<u32>>,     // D x Z0
    n_zw: Vec<Vec<u32>>,     // Z0 x V
    n_z: Vec<u32>,           // Z0
    n_d: Vec<u32>,           // D (doc lengths)
    docs_z: Vec<Vec<usize>>, // D x N_d current tuple per token

    // parameters
    alpha_b: f64,
    alpha_z: Vec<Vec<f64>>,       // K x Z_k
    alpha_dz: Vec<Vec<Vec<f64>>>, // K x Z_k x D
    beta: Vec<f64>,               // Z0
    omega_b: f64,
    omega_w: Vec<f64>,            // V
    omega_zw: Vec<Vec<Vec<f64>>>, // K x Z_k x V
    eta_w: Vec<f64>,              // V
    eta_zw: Vec<Vec<Vec<f64>>>,   // K x Z_k x V

    // cached priors (floored to PRIOR_RIDGE)
    prior_dz: Vec<Vec<f64>>, // D x Z0
    alpha_norm: Vec<f64>,    // D
    prior_zw: Vec<Vec<f64>>, // Z0 x V
    omega_norm: Vec<f64>,    // Z0

    // tail-sample accumulators: the per-sample predictive phi/theta are summed
    // (each using that sample's own counts AND prior/normalizer) and averaged, so
    // the estimator is a consistent Monte Carlo posterior mean even while omega is
    // still moving during the tail (Gate B: no count/prior mismatch).
    samp_theta: Vec<Vec<f64>>, // D x Z0
    samp_phi: Vec<Vec<f64>>,   // Z0 x V
    n_collected: f64,
}

impl<'a> State<'a> {
    #[inline]
    fn prior_a(&self, d: usize, x: usize) -> f64 {
        let z = &self.tuple_vecs[x];
        let mut weight = self.alpha_b;
        for i in 0..self.k {
            weight += self.alpha_z[i][z[i]] + self.alpha_dz[i][z[i]][d];
        }
        let b = if self.cfg.sparsity {
            logistic(self.beta[x])
        } else {
            1.0
        };
        (b * weight.exp()).max(PRIOR_RIDGE)
    }

    #[inline]
    fn prior_w(&self, w: usize, x: usize) -> f64 {
        let z = &self.tuple_vecs[x];
        let mut weight = self.omega_b + self.omega_w[w];
        for i in 0..self.k {
            weight += self.omega_zw[i][z[i]][w];
        }
        weight.exp().max(PRIOR_RIDGE)
    }

    fn recompute_alpha_cache(&mut self) {
        for d in 0..self.d {
            let mut norm = 0.0;
            for x in 0..self.z0 {
                let p = self.prior_a(d, x);
                self.prior_dz[d][x] = p;
                norm += p;
            }
            self.alpha_norm[d] = norm;
        }
    }

    fn recompute_omega_cache(&mut self) {
        for x in 0..self.z0 {
            let mut norm = 0.0;
            for w in 0..self.v {
                let p = self.prior_w(w, x);
                self.prior_zw[x][w] = p;
                norm += p;
            }
            self.omega_norm[x] = norm;
        }
    }

    /// Block sampler: enumerate all Z0 tuples for the token.
    fn sample_block<R: Rng>(&mut self, d: usize, n: usize, rng: &mut R) {
        let w = self.docs[d][n] as usize;
        let cur = self.docs_z[d][n];
        self.n_zw[cur][w] -= 1;
        self.n_z[cur] -= 1;
        self.n_dz[d][cur] -= 1;

        let mut p = vec![0.0f64; self.z0];
        let mut total = 0.0;
        for x in 0..self.z0 {
            let pr = (self.n_dz[d][x] as f64 + self.prior_dz[d][x])
                * (self.n_zw[x][w] as f64 + self.prior_zw[x][w])
                / (self.n_z[x] as f64 + self.omega_norm[x]);
            p[x] = pr;
            total += pr;
        }
        let tuple = draw(&p, total, rng);
        self.n_zw[tuple][w] += 1;
        self.n_z[tuple] += 1;
        self.n_dz[d][tuple] += 1;
        self.docs_z[d][n] = tuple;
    }

    /// Independent sampler: resample each factor with the others fixed (additive
    /// cost). Counts are decremented once and shared across the K factor draws,
    /// matching the reference.
    fn sample_ind<R: Rng>(&mut self, d: usize, n: usize, rng: &mut R) {
        let w = self.docs[d][n] as usize;
        let cur = self.docs_z[d][n];
        self.n_zw[cur][w] -= 1;
        self.n_z[cur] -= 1;
        self.n_dz[d][cur] -= 1;

        // Each factor is resampled conditioned on the OTHER factors' ORIGINAL
        // values (the reference's Jacobi-style `sampleInd`, which clones the
        // pre-decrement tuple for every factor draw), not on the newly-sampled
        // ones. Assemble the result in `z_new` so the loop never feeds a fresh
        // draw back into the next factor's candidates.
        let z = self.ti.to_vector(cur);
        let mut z_new = z.clone();
        for kf in 0..self.k {
            let zk = self.cfg.factor_sizes[kf];
            let mut p = vec![0.0f64; zk];
            let mut total = 0.0;
            let mut cand = z.clone();
            for j in 0..zk {
                cand[kf] = j;
                let x = self.ti.to_index(&cand);
                let pr = (self.n_dz[d][x] as f64 + self.prior_dz[d][x])
                    * (self.n_zw[x][w] as f64 + self.prior_zw[x][w])
                    / (self.n_z[x] as f64 + self.omega_norm[x]);
                p[j] = pr;
                total += pr;
            }
            z_new[kf] = draw(&p, total, rng);
        }
        let tuple = self.ti.to_index(&z_new);
        self.n_zw[tuple][w] += 1;
        self.n_z[tuple] += 1;
        self.n_dz[d][tuple] += 1;
        self.docs_z[d][n] = tuple;
    }

    /// Analytic gradient of the penalized log-likelihood w.r.t. the alpha weights
    /// (and beta after burn-in). Pure: reads only the pre-step caches, counts, and
    /// params, so it can be finite-difference checked against [`alpha_objective`].
    fn alpha_gradient(&self, iter: usize) -> AlphaGrad {
        let sig2 = self.cfg.sigma_alpha * self.cfg.sigma_alpha;
        let mut g_b = 0.0;
        let mut g_z: Vec<Vec<f64>> = zeros_kz(&self.cfg.factor_sizes);
        let mut g_dz: Vec<Vec<Vec<f64>>> = self
            .cfg
            .factor_sizes
            .iter()
            .map(|&zk| vec![vec![0.0; self.d]; zk])
            .collect();
        let mut g_beta = vec![0.0f64; self.z0];

        for d in 0..self.d {
            let dg_norm = dg(self.alpha_norm[d]);
            let dg_norm_n = dg(self.alpha_norm[d] + self.n_d[d] as f64);
            for x in 0..self.z0 {
                let z = &self.tuple_vecs[x];
                let pdz = self.prior_dz[d][x];
                let g_ll = pdz * (dg_norm - dg_norm_n + dg(pdz + self.n_dz[d][x] as f64) - dg(pdz));
                for i in 0..self.k {
                    g_z[i][z[i]] += g_ll;
                    g_dz[i][z[i]][d] += g_ll;
                }
                g_b += g_ll;
                if self.cfg.sparsity {
                    g_beta[x] += g_ll * (1.0 - logistic(self.beta[x]));
                }
            }
            for i in 0..self.k {
                for zi in 0..self.cfg.factor_sizes[i] {
                    g_dz[i][zi][d] += -self.alpha_dz[i][zi][d] / sig2;
                }
            }
        }
        for i in 0..self.k {
            for zi in 0..self.cfg.factor_sizes[i] {
                g_z[i][zi] += -self.alpha_z[i][zi] / sig2;
            }
        }
        g_b += -self.alpha_b / (self.cfg.sigma_alpha_bias * self.cfg.sigma_alpha_bias);

        if self.cfg.sparsity && iter >= self.cfg.weight_burnin {
            for x in 0..self.z0 {
                let b = logistic(self.beta[x]);
                let db = b * (1.0 - b);
                g_beta[x] += (self.cfg.delta0 - 1.0) * db / b;
                g_beta[x] += (self.cfg.delta1 - 1.0) * (-db) / (1.0 - b);
            }
        } else {
            g_beta.iter_mut().for_each(|g| *g = 0.0);
        }
        AlphaGrad {
            b: g_b,
            z: g_z,
            dz: g_dz,
            beta: g_beta,
        }
    }

    /// One gradient-ascent step on the alpha weights and (after burn-in) beta.
    fn update_alpha(&mut self, iter: usize) {
        let g = self.alpha_gradient(iter);
        for i in 0..self.k {
            for zi in 0..self.cfg.factor_sizes[i] {
                self.alpha_z[i][zi] += self.cfg.step_alpha_corpus * g.z[i][zi];
                for d in 0..self.d {
                    self.alpha_dz[i][zi][d] += self.cfg.step_alpha_doc * g.dz[i][zi][d];
                }
            }
        }
        self.alpha_b += self.cfg.step_alpha_bias * g.b;
        if self.cfg.sparsity && iter >= self.cfg.weight_burnin {
            for x in 0..self.z0 {
                self.beta[x] += self.cfg.step_beta * g.beta[x];
            }
        }
    }

    /// Analytic gradient w.r.t. the omega weights. Pure (see [`alpha_gradient`]).
    fn omega_gradient(&self) -> OmegaGrad {
        let sig2 = self.cfg.sigma_omega * self.cfg.sigma_omega;
        let mut g_b = 0.0;
        let mut g_w = vec![0.0f64; self.v];
        let mut g_zw: Vec<Vec<Vec<f64>>> = self
            .cfg
            .factor_sizes
            .iter()
            .map(|&zk| vec![vec![0.0; self.v]; zk])
            .collect();
        for w in 0..self.v {
            for x in 0..self.z0 {
                let z = &self.tuple_vecs[x];
                let pzw = self.prior_zw[x][w];
                let g_ll = pzw
                    * (dg(self.omega_norm[x]) - dg(self.omega_norm[x] + self.n_z[x] as f64)
                        + dg(pzw + self.n_zw[x][w] as f64)
                        - dg(pzw));
                for i in 0..self.k {
                    g_zw[i][z[i]][w] += g_ll;
                }
                g_w[w] += g_ll;
                g_b += g_ll;
            }
            if self.cfg.word_priors {
                for i in 0..self.k {
                    for zi in 0..self.cfg.factor_sizes[i] {
                        g_zw[i][zi][w] += -(self.omega_zw[i][zi][w] - self.eta_zw[i][zi][w]) / sig2;
                    }
                }
            } else {
                for i in 0..self.k {
                    for zi in 0..self.cfg.factor_sizes[i] {
                        g_zw[i][zi][w] = 0.0;
                    }
                }
            }
            if self.cfg.symmetric_word_prior {
                g_w[w] = 0.0;
            } else {
                g_w[w] += -(self.omega_w[w] - self.eta_w[w]) / sig2;
            }
        }
        g_b += -self.omega_b / (self.cfg.sigma_omega_bias * self.cfg.sigma_omega_bias);
        OmegaGrad {
            b: g_b,
            w: g_w,
            zw: g_zw,
        }
    }

    /// One gradient-ascent step on the omega weights (after burn-in).
    fn update_omega(&mut self, iter: usize) {
        if iter < self.cfg.weight_burnin {
            return;
        }
        let g = self.omega_gradient();
        for w in 0..self.v {
            if self.cfg.word_priors {
                for i in 0..self.k {
                    for zi in 0..self.cfg.factor_sizes[i] {
                        self.omega_zw[i][zi][w] += self.cfg.step_omega * g.zw[i][zi][w];
                    }
                }
            }
            if !self.cfg.symmetric_word_prior {
                self.omega_w[w] += self.cfg.step_omega * g.w[w];
            }
        }
        self.omega_b += self.cfg.step_omega_bias * g.b;
    }

    /// Marginal log-likelihood of the corpus given the current MAP theta/phi
    /// (matches the reference `computeLL`; used only for the fit trace).
    fn log_likelihood(&self) -> f64 {
        let mut ll = 0.0;
        for d in 0..self.d {
            for &wu in &self.docs[d] {
                let w = wu as usize;
                let mut token = 0.0;
                for x in 0..self.z0 {
                    token += (self.n_dz[d][x] as f64 + self.prior_dz[d][x])
                        / (self.n_d[d] as f64 + self.alpha_norm[d])
                        * (self.n_zw[x][w] as f64 + self.prior_zw[x][w])
                        / (self.n_z[x] as f64 + self.omega_norm[x]);
                }
                ll += token.max(1e-300).ln();
            }
        }
        ll
    }

    /// Penalized objective whose gradient is [`alpha_gradient`] — the
    /// Dirichlet-multinomial compound plus the Gaussian (and Beta, after burn-in)
    /// priors. Recomputes the alpha prior from the current params so a
    /// finite-difference perturbation is reflected (does NOT use the cache).
    #[cfg(test)]
    fn alpha_objective(&self, iter: usize) -> f64 {
        use crate::mathfun::log_gamma;
        let mut val = 0.0;
        for d in 0..self.d {
            let mut norm = 0.0;
            let mut priors = vec![0.0; self.z0];
            for x in 0..self.z0 {
                priors[x] = self.prior_a(d, x);
                norm += priors[x];
            }
            val += log_gamma(norm) - log_gamma(norm + self.n_d[d] as f64);
            for x in 0..self.z0 {
                val += log_gamma(priors[x] + self.n_dz[d][x] as f64) - log_gamma(priors[x]);
            }
        }
        let sig2 = self.cfg.sigma_alpha * self.cfg.sigma_alpha;
        for i in 0..self.k {
            for zi in 0..self.cfg.factor_sizes[i] {
                val -= self.alpha_z[i][zi] * self.alpha_z[i][zi] / (2.0 * sig2);
                for d in 0..self.d {
                    val -= self.alpha_dz[i][zi][d] * self.alpha_dz[i][zi][d] / (2.0 * sig2);
                }
            }
        }
        val -= self.alpha_b * self.alpha_b
            / (2.0 * self.cfg.sigma_alpha_bias * self.cfg.sigma_alpha_bias);
        if self.cfg.sparsity && iter >= self.cfg.weight_burnin {
            for x in 0..self.z0 {
                let b = logistic(self.beta[x]);
                val += (self.cfg.delta0 - 1.0) * b.ln() + (self.cfg.delta1 - 1.0) * (1.0 - b).ln();
            }
        }
        val
    }

    /// Penalized objective whose gradient is [`omega_gradient`].
    #[cfg(test)]
    fn omega_objective(&self) -> f64 {
        use crate::mathfun::log_gamma;
        let mut val = 0.0;
        for x in 0..self.z0 {
            let mut norm = 0.0;
            let mut priors = vec![0.0; self.v];
            for w in 0..self.v {
                priors[w] = self.prior_w(w, x);
                norm += priors[w];
            }
            val += log_gamma(norm) - log_gamma(norm + self.n_z[x] as f64);
            for w in 0..self.v {
                val += log_gamma(priors[w] + self.n_zw[x][w] as f64) - log_gamma(priors[w]);
            }
        }
        let sig2 = self.cfg.sigma_omega * self.cfg.sigma_omega;
        for w in 0..self.v {
            if self.cfg.word_priors {
                for i in 0..self.k {
                    for zi in 0..self.cfg.factor_sizes[i] {
                        let d = self.omega_zw[i][zi][w] - self.eta_zw[i][zi][w];
                        val -= d * d / (2.0 * sig2);
                    }
                }
            }
            if !self.cfg.symmetric_word_prior {
                let d = self.omega_w[w] - self.eta_w[w];
                val -= d * d / (2.0 * sig2);
            }
        }
        val -= self.omega_b * self.omega_b
            / (2.0 * self.cfg.sigma_omega_bias * self.cfg.sigma_omega_bias);
        val
    }

    fn collect_sample(&mut self) {
        // theta_d = (nDZ + priorDZ) / (N_d + alphaNorm), using this sample's caches.
        for d in 0..self.d {
            let denom = (self.n_d[d] as f64 + self.alpha_norm[d]).max(1e-300);
            for x in 0..self.z0 {
                self.samp_theta[d][x] += (self.n_dz[d][x] as f64 + self.prior_dz[d][x]) / denom;
            }
        }
        // phi_x = (nZW + priorZW) / (nZ + omegaNorm), using this sample's caches.
        for x in 0..self.z0 {
            let denom = (self.n_z[x] as f64 + self.omega_norm[x]).max(1e-300);
            for w in 0..self.v {
                self.samp_phi[x][w] += (self.n_zw[x][w] as f64 + self.prior_zw[x][w]) / denom;
            }
        }
        self.n_collected += 1.0;
    }
}

#[inline]
fn draw<R: Rng>(p: &[f64], total: f64, rng: &mut R) -> usize {
    if !total.is_finite() || total <= 0.0 {
        return 0;
    }
    let u = rng.gen::<f64>() * total;
    let mut v = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        v += pi;
        if v > u {
            return i;
        }
    }
    p.len() - 1
}

impl<'a> State<'a> {
    fn new(corpus: &'a Corpus, cfg: &'a FactorialLdaConfig, priors: &OmegaPriors) -> State<'a> {
        let ti = TupleIndex::new(&cfg.factor_sizes);
        let z0 = ti.z0;
        let k = cfg.factor_sizes.len();
        let v = corpus.num_types();
        let docs = &corpus.docs;
        let d = docs.len();
        let tuple_vecs: Vec<Vec<usize>> = (0..z0).map(|x| ti.to_vector(x)).collect();

        // omega initialized AT the prior means eta (Gate A blocker: not 0).
        // The ablation flags REMOVE their term entirely (Gate B blocker): they
        // override any informed prior, so `word_priors=false` zeros omega_zw (and
        // its eta target) and `symmetric_word_prior=true` zeros the background
        // omega_w — matching the paper's "fix omega^(k)=0" / "fix omega^(0)=0",
        // rather than merely freezing a seeded value.
        let eta_w = if cfg.symmetric_word_prior {
            vec![0.0; v]
        } else if priors.eta_w.len() == v {
            priors.eta_w.clone()
        } else {
            vec![0.0; v]
        };
        let eta_zw: Vec<Vec<Vec<f64>>> = if cfg.word_priors && priors.eta_zw.len() == k {
            priors.eta_zw.clone()
        } else {
            cfg.factor_sizes
                .iter()
                .map(|&zk| vec![vec![0.0; v]; zk])
                .collect()
        };
        let omega_w = eta_w.clone();
        let omega_zw = eta_zw.clone();
        let n_d: Vec<u32> = docs.iter().map(|doc| doc.len() as u32).collect();

        State {
            cfg,
            ti,
            docs,
            d,
            v,
            k,
            z0,
            tuple_vecs,
            n_dz: vec![vec![0u32; z0]; d],
            n_zw: vec![vec![0u32; v]; z0],
            n_z: vec![0u32; z0],
            n_d,
            docs_z: docs.iter().map(|doc| vec![0usize; doc.len()]).collect(),
            alpha_b: cfg.alpha_bias_init,
            alpha_z: cfg.factor_sizes.iter().map(|&zk| vec![0.0; zk]).collect(),
            alpha_dz: cfg
                .factor_sizes
                .iter()
                .map(|&zk| vec![vec![0.0; d]; zk])
                .collect(),
            beta: vec![0.0; z0],
            omega_b: cfg.omega_bias_init,
            omega_w,
            omega_zw,
            eta_w,
            eta_zw,
            prior_dz: vec![vec![0.0; z0]; d],
            alpha_norm: vec![0.0; d],
            prior_zw: vec![vec![0.0; v]; z0],
            omega_norm: vec![0.0; z0],
            samp_theta: vec![vec![0.0; z0]; d],
            samp_phi: vec![vec![0.0; v]; z0],
            n_collected: 0.0,
        }
    }

    /// Seed each token's tuple from the word-prior weights (caches must be current).
    fn seed_tokens<R: Rng>(&mut self, rng: &mut R) {
        for di in 0..self.d {
            for n in 0..self.docs[di].len() {
                let w = self.docs[di][n] as usize;
                let mut p = vec![0.0f64; self.z0];
                let mut total = 0.0;
                for x in 0..self.z0 {
                    p[x] = self.prior_zw[x][w];
                    total += p[x];
                }
                let x = draw(&p, total, rng);
                self.docs_z[di][n] = x;
                self.n_zw[x][w] += 1;
                self.n_z[x] += 1;
                self.n_dz[di][x] += 1;
            }
        }
    }
}

/// Fit fLDA with the given configuration and omega priors.
pub fn fit_flda<R: Rng>(
    corpus: &Corpus,
    cfg: &FactorialLdaConfig,
    priors: &OmegaPriors,
    rng: &mut R,
) -> FactorialLDAModel {
    let mut st = State::new(corpus, cfg, priors);
    let z0 = st.z0;
    let d = st.d;
    let docs = corpus.docs.as_slice();

    // initial caches, then seed each token from the word-prior weights.
    st.recompute_omega_cache();
    st.recompute_alpha_cache();
    st.seed_tokens(rng);

    let samples = cfg.samples.max(1);
    let mut fit_history = Vec::new();
    for iter in 1..=cfg.iters {
        let burned_in = iter > cfg.iters.saturating_sub(samples);
        // block-sample every block_freq-th iteration, otherwise sample each factor
        // independently. block_freq is validated >= 1 in the binding; clamp here so
        // a direct Rust caller cannot trigger a modulo-by-zero.
        let block = iter % cfg.block_freq.max(1) == 0;
        for di in 0..d {
            for n in 0..docs[di].len() {
                if block {
                    st.sample_block(di, n, rng);
                } else {
                    st.sample_ind(di, n, rng);
                }
            }
        }
        // one MCEM gradient step (omega first, then alpha, as the reference).
        st.update_omega(iter);
        st.update_alpha(iter);
        st.recompute_alpha_cache();
        st.recompute_omega_cache();

        if cfg.eval_every > 0 && iter % cfg.eval_every == 0 {
            fit_history.push((iter, st.log_likelihood()));
        }
        if burned_in {
            st.collect_sample();
        }
    }
    let final_ll = st.log_likelihood();
    fit_history.push((cfg.iters, final_ll));

    // posterior-mean topic-word (phi) and doc-topic (theta): the average of the
    // per-sample predictive distributions accumulated in collect_sample. Each row
    // is a mean of distributions, so it already sums to 1.
    let nc = st.n_collected.max(1.0);
    let topic_word: Vec<Vec<f64>> = st
        .samp_phi
        .iter()
        .map(|row| row.iter().map(|&p| p / nc).collect())
        .collect();
    let doc_topic: Vec<Vec<f64>> = st
        .samp_theta
        .iter()
        .map(|row| row.iter().map(|&p| p / nc).collect())
        .collect();
    let tuple_activity: Vec<f64> = (0..z0)
        .map(|x| {
            if cfg.sparsity {
                logistic(st.beta[x])
            } else {
                1.0
            }
        })
        .collect();

    FactorialLDAModel {
        num_topics: z0,
        factor_sizes: cfg.factor_sizes.clone(),
        tuples: st.tuple_vecs.clone(),
        topic_word,
        doc_topic,
        omega_b: st.omega_b,
        omega_w: st.omega_w.clone(),
        omega_zw: st.omega_zw.clone(),
        tuple_activity,
        fit_history,
        converged: true,
    }
}

impl Estimator for FactorialLDAModel {
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
        self.fit_history.clone()
    }
    fn converged(&self) -> Option<bool> {
        Some(self.converged)
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn tiny_corpus() -> Corpus {
        // 4 word types: 0,1 = "topic" words; 2,3 = "sentiment" words.
        // Docs mix one topic word group with one sentiment marker.
        let docs = vec![
            vec![0u32, 0, 1, 2, 2],
            vec![0, 1, 1, 3, 3],
            vec![0, 0, 1, 1, 2],
            vec![1, 1, 0, 3, 2],
        ];
        Corpus {
            id_to_word: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            docs,
            doc_names: vec![],
            doc_labels: vec![],
            doc_freqs: vec![],
            total_freqs: vec![],
        }
    }

    fn cfg() -> FactorialLdaConfig {
        FactorialLdaConfig {
            factor_sizes: vec![2, 2],
            iters: 60,
            samples: 20,
            weight_burnin: 10,
            ..Default::default()
        }
    }

    #[test]
    fn tuple_index_roundtrip() {
        let ti = TupleIndex::new(&[3, 2, 4]);
        assert_eq!(ti.z0, 24);
        for x in 0..ti.z0 {
            assert_eq!(ti.to_index(&ti.to_vector(x)), x);
        }
    }

    #[test]
    fn factorial_lda_is_deterministic() {
        let c = tiny_corpus();
        let cfg = cfg();
        let priors = OmegaPriors::default();
        let m1 = fit_flda(&c, &cfg, &priors, &mut ChaCha8Rng::seed_from_u64(42));
        let m2 = fit_flda(&c, &cfg, &priors, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(m1.topic_word, m2.topic_word);
        assert_eq!(m1.doc_topic, m2.doc_topic);
    }

    #[test]
    fn factorial_lda_conforms() {
        let c = tiny_corpus();
        let m = fit_flda(
            &c,
            &cfg(),
            &OmegaPriors::default(),
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
        // shapes: topic_word is Z0 x V, doc_topic rows sum to 1.
        assert_eq!(m.num_topics, 4);
        assert_eq!(m.topic_word.len(), 4);
        assert_eq!(m.topic_word[0].len(), 4);
        for row in &m.doc_topic {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "theta row sums to {s}");
        }
    }

    #[test]
    fn phi_rows_are_distributions() {
        let c = tiny_corpus();
        let m = fit_flda(
            &c,
            &cfg(),
            &OmegaPriors::default(),
            &mut ChaCha8Rng::seed_from_u64(7),
        );
        for row in &m.topic_word {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "phi row sums to {s}");
            assert!(row.iter().all(|&p| p.is_finite() && p >= 0.0));
        }
    }

    // Gate A NaN guard: force tuples off (delta small, long burn) and assert finite.
    #[test]
    fn sparsity_off_tuples_stay_finite() {
        let c = tiny_corpus();
        let mut cfg = cfg();
        cfg.iters = 120;
        cfg.delta0 = 0.01;
        cfg.delta1 = 0.01;
        cfg.step_beta = 0.1; // push beta hard so some b_x -> 0
        let m = fit_flda(
            &c,
            &cfg,
            &OmegaPriors::default(),
            &mut ChaCha8Rng::seed_from_u64(3),
        );
        assert!(m.tuple_activity.iter().all(|b| b.is_finite()));
        assert!(m.topic_word.iter().flatten().all(|p| p.is_finite()));
        assert!(m.doc_topic.iter().flatten().all(|p| p.is_finite()));
    }

    // Factor-tying invariant: bumping omega_zw[k][z] shifts the word prior for every
    // tuple whose k-th component is z, and no other tuple.
    #[test]
    fn omega_tying_touches_only_matching_tuples() {
        let ti = TupleIndex::new(&[3, 2]);
        // factor 0, component 1: tuples with z[0]==1 are indices 2,3.
        let matching: Vec<usize> = (0..ti.z0).filter(|&x| ti.to_vector(x)[0] == 1).collect();
        assert_eq!(matching, vec![2, 3]);
        let non: Vec<usize> = (0..ti.z0).filter(|&x| ti.to_vector(x)[0] != 1).collect();
        assert_eq!(non, vec![0, 1, 4, 5]);

        // And on a fitted model: bump omega_zw[0][1] and confirm prior_w changes for
        // exactly the matching tuples.
        let c = tiny_corpus();
        let cfg = cfg();
        let mut st = fitted_state(&c, &cfg, &mut ChaCha8Rng::seed_from_u64(1));
        let w = 0usize;
        let before: Vec<f64> = (0..st.z0).map(|x| st.prior_w(w, x)).collect();
        for x in 0..st.z0 {
            st.omega_zw[0][1][x % st.v] = 0.0; // no-op guard for index safety
        }
        st.omega_zw[0][1][w] += 1.0;
        for x in 0..st.z0 {
            let after = st.prior_w(w, x);
            if st.tuple_vecs[x][0] == 1 {
                assert!(
                    (after - before[x]).abs() > 1e-9,
                    "matching tuple {x} did not move"
                );
            } else {
                assert!(
                    (after - before[x]).abs() < 1e-12,
                    "non-matching tuple {x} moved"
                );
            }
        }
    }

    // ω is initialized AT the prior means η, not 0 (Gate A blocker, Codex #3).
    #[test]
    fn omega_initialized_from_eta() {
        let c = tiny_corpus();
        let cfg = cfg();
        let priors = OmegaPriors {
            eta_w: vec![0.3, -0.2, 0.1, 0.0],
            eta_zw: vec![
                vec![vec![0.5, 0.0, 0.0, 0.0], vec![0.0, 0.4, 0.0, 0.0]],
                vec![vec![0.0, 0.0, 0.2, 0.0], vec![0.0, 0.0, 0.0, 0.6]],
            ],
        };
        let st = State::new(&c, &cfg, &priors);
        assert_eq!(st.omega_w, priors.eta_w);
        assert_eq!(st.omega_zw, priors.eta_zw);
    }

    /// Build a fitted State (post-burn-in) for gradient/invariant checks.
    fn fitted_state<'a>(
        c: &'a Corpus,
        cfg: &'a FactorialLdaConfig,
        rng: &mut ChaCha8Rng,
    ) -> State<'a> {
        fitted_state_with(c, cfg, &OmegaPriors::default(), rng)
    }

    fn fitted_state_with<'a>(
        c: &'a Corpus,
        cfg: &'a FactorialLdaConfig,
        priors: &OmegaPriors,
        rng: &mut ChaCha8Rng,
    ) -> State<'a> {
        let mut st = State::new(c, cfg, priors);
        st.recompute_omega_cache();
        st.recompute_alpha_cache();
        st.seed_tokens(rng);
        for iter in 1..=cfg.iters {
            for di in 0..st.d {
                for n in 0..st.docs[di].len() {
                    st.sample_block(di, n, rng);
                }
            }
            st.update_omega(iter);
            st.update_alpha(iter);
            st.recompute_alpha_cache();
            st.recompute_omega_cache();
        }
        st
    }

    // BACKBONE: analytic alpha/omega/beta gradients match central finite differences
    // of the exact penalized objective. Certifies the M-step regardless of the
    // non-reproducible reference (Gate A blocker: recovery tests can't catch a sign
    // flip; this can).
    #[test]
    fn gradients_match_finite_difference() {
        let c = tiny_corpus();
        let cfg = cfg();
        let mut st = fitted_state(&c, &cfg, &mut ChaCha8Rng::seed_from_u64(9));
        let iter = cfg.iters; // past burn-in, so beta is active
        let eps = 1e-6;
        let close = |fd: f64, an: f64, what: &str| {
            let tol = 1e-4 + 1e-3 * an.abs();
            assert!(
                (fd - an).abs() < tol,
                "{what}: fd={fd} analytic={an} diff={}",
                (fd - an).abs()
            );
        };

        // ---- alpha side ----
        let ag = st.alpha_gradient(iter);
        // alpha_b
        {
            st.alpha_b += eps;
            let vp = st.alpha_objective(iter);
            st.alpha_b -= 2.0 * eps;
            let vm = st.alpha_objective(iter);
            st.alpha_b += eps;
            close((vp - vm) / (2.0 * eps), ag.b, "alpha_b");
        }
        // alpha_z[0][1]
        {
            st.alpha_z[0][1] += eps;
            let vp = st.alpha_objective(iter);
            st.alpha_z[0][1] -= 2.0 * eps;
            let vm = st.alpha_objective(iter);
            st.alpha_z[0][1] += eps;
            close((vp - vm) / (2.0 * eps), ag.z[0][1], "alpha_z[0][1]");
        }
        // alpha_dz[1][0][2]
        {
            st.alpha_dz[1][0][2] += eps;
            let vp = st.alpha_objective(iter);
            st.alpha_dz[1][0][2] -= 2.0 * eps;
            let vm = st.alpha_objective(iter);
            st.alpha_dz[1][0][2] += eps;
            close((vp - vm) / (2.0 * eps), ag.dz[1][0][2], "alpha_dz[1][0][2]");
        }
        // beta[2] (sparsity gradient)
        {
            st.beta[2] += eps;
            let vp = st.alpha_objective(iter);
            st.beta[2] -= 2.0 * eps;
            let vm = st.alpha_objective(iter);
            st.beta[2] += eps;
            close((vp - vm) / (2.0 * eps), ag.beta[2], "beta[2]");
        }

        // ---- omega side ----
        let og = st.omega_gradient();
        // omega_b
        {
            st.omega_b += eps;
            let vp = st.omega_objective();
            st.omega_b -= 2.0 * eps;
            let vm = st.omega_objective();
            st.omega_b += eps;
            close((vp - vm) / (2.0 * eps), og.b, "omega_b");
        }
        // omega_w[1]
        {
            st.omega_w[1] += eps;
            let vp = st.omega_objective();
            st.omega_w[1] -= 2.0 * eps;
            let vm = st.omega_objective();
            st.omega_w[1] += eps;
            close((vp - vm) / (2.0 * eps), og.w[1], "omega_w[1]");
        }
        // omega_zw[0][1][3]
        {
            st.omega_zw[0][1][3] += eps;
            let vp = st.omega_objective();
            st.omega_zw[0][1][3] -= 2.0 * eps;
            let vm = st.omega_objective();
            st.omega_zw[0][1][3] += eps;
            close((vp - vm) / (2.0 * eps), og.zw[0][1][3], "omega_zw[0][1][3]");
        }
    }

    // The omega gradient's `-(omega - eta)/sigma^2` prior term must be FD-correct for
    // NON-ZERO eta too (Gate B: the default test only covered eta = 0). Uses informed
    // priors so omega is pulled toward a non-zero target.
    #[test]
    fn omega_gradient_matches_fd_with_nonzero_eta() {
        let c = tiny_corpus();
        let cfg = cfg();
        let priors = OmegaPriors {
            eta_w: vec![0.3, -0.2, 0.1, 0.05],
            eta_zw: vec![
                vec![vec![0.5, -0.1, 0.0, 0.2], vec![-0.3, 0.4, 0.1, 0.0]],
                vec![vec![0.0, 0.2, -0.2, 0.1], vec![0.1, -0.1, 0.3, 0.6]],
            ],
        };
        let mut st = fitted_state_with(&c, &cfg, &priors, &mut ChaCha8Rng::seed_from_u64(11));
        let og = st.omega_gradient();
        let eps = 1e-6;
        let close = |fd: f64, an: f64, what: &str| {
            assert!(
                (fd - an).abs() < 1e-4 + 1e-3 * an.abs(),
                "{what}: fd={fd} analytic={an}"
            );
        };
        for (i, j, w) in [(0usize, 0usize, 0usize), (0, 1, 2), (1, 0, 3)] {
            st.omega_zw[i][j][w] += eps;
            let vp = st.omega_objective();
            st.omega_zw[i][j][w] -= 2.0 * eps;
            let vm = st.omega_objective();
            st.omega_zw[i][j][w] += eps;
            close(
                (vp - vm) / (2.0 * eps),
                og.zw[i][j][w],
                "omega_zw(nonzero eta)",
            );
        }
        for w in [0usize, 3] {
            st.omega_w[w] += eps;
            let vp = st.omega_objective();
            st.omega_w[w] -= 2.0 * eps;
            let vm = st.omega_objective();
            st.omega_w[w] += eps;
            close((vp - vm) / (2.0 * eps), og.w[w], "omega_w(nonzero eta)");
        }
    }

    // Ablations override informed priors (Gate B): word_priors=false zeros omega_zw
    // even when omega_priors seed it, and symmetric_word_prior=true zeros omega_w.
    #[test]
    fn ablations_override_informed_priors() {
        let c = tiny_corpus();
        let priors = OmegaPriors {
            eta_w: vec![0.3, -0.2, 0.1, 0.05],
            eta_zw: vec![
                vec![vec![0.5, 0.0, 0.0, 0.0], vec![0.0, 0.4, 0.0, 0.0]],
                vec![vec![0.0, 0.0, 0.2, 0.0], vec![0.0, 0.0, 0.0, 0.6]],
            ],
        };
        let mut cfg_no_wp = cfg();
        cfg_no_wp.word_priors = false;
        let st = State::new(&c, &cfg_no_wp, &priors);
        assert!(st.omega_zw.iter().flatten().flatten().all(|&x| x == 0.0));
        assert!(st.eta_zw.iter().flatten().flatten().all(|&x| x == 0.0));

        let mut cfg_sym = cfg();
        cfg_sym.symmetric_word_prior = true;
        let st = State::new(&c, &cfg_sym, &priors);
        assert!(st.omega_w.iter().all(|&x| x == 0.0));
        assert!(st.eta_w.iter().all(|&x| x == 0.0));
    }
}
