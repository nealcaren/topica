//! Dynamic Topic Model (Blei & Lafferty 2006) — topics whose word
//! distributions **evolve over time**. The corpus is split into time slices;
//! each topic's natural parameters β_{t,k,w} follow a Gaussian random walk
//! (a state-space model), so a topic's vocabulary drifts smoothly from one
//! slice to the next instead of being fixed.
//!
//! Inference is variational, following Blei's C `dtm` and its faithful gensim
//! port (`LdaSeqModel`):
//!   - each topic-word chain is a **state-space language model** (`Sslm`) whose
//!     latent means are obtained by a Kalman filter/smoother over the slices;
//!   - per document we run ordinary **LDA variational inference** against that
//!     slice's topic distributions (γ/φ updates);
//!   - the M-step refits each chain's variational observations by maximizing the
//!     state-space bound (Newton/quasi-Newton on the same objective gensim's CG
//!     uses — here our L-BFGS, with the exact `f_obs`/`df_obs` gradient).
//!
//! Constants and recursions mirror gensim's `ldaseqmodel.py` verbatim
//! (`INIT_VARIANCE_CONST = 1000`, `OBS_NORM_CUTOFF = 2`, chain/obs variance
//! defaults 0.005 / 0.5), so results track the reference implementation.

use crate::variational::lbfgs_minimize;
use crate::optimize::digamma;
use rand::Rng;
use rayon::prelude::*;

const INIT_VARIANCE_CONST: f64 = 1000.0;
const OBS_NORM_CUTOFF: f64 = 2.0;

/// Stirling-series log Γ (accurate to ~1e-10 for the small positive arguments
/// the bound needs); shifts the argument up to z ≥ 10 for accuracy.
fn lgamma(mut z: f64) -> f64 {
    const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;
    let mut shift = 0i32;
    while z < 10.0 {
        z += 1.0;
        shift += 1;
    }
    let mut result = HALF_LOG_TWO_PI + (z - 0.5) * z.ln() - z + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z * z * z * z * z);
    while shift > 0 {
        shift -= 1;
        z -= 1.0;
        result -= z.ln();
    }
    result
}

fn log_add(a: f64, b: f64) -> f64 {
    if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

// ---------------------------------------------------------------------------
// State-space language model for one topic chain (vocab V × T time slices)
// ---------------------------------------------------------------------------

struct Sslm {
    v: usize,
    t: usize,
    chain_variance: f64,
    obs_variance: f64,
    obs: Vec<Vec<f64>>,          // V × T
    mean: Vec<Vec<f64>>,         // V × (T+1)
    fwd_mean: Vec<Vec<f64>>,     // V × (T+1)
    variance: Vec<Vec<f64>>,     // V × (T+1)
    fwd_variance: Vec<Vec<f64>>, // V × (T+1)
    zeta: Vec<f64>,              // T
    e_log_prob: Vec<Vec<f64>>,   // V × T
}

impl Sslm {
    fn new(v: usize, t: usize, chain_variance: f64, obs_variance: f64) -> Self {
        Sslm {
            v,
            t,
            chain_variance,
            obs_variance,
            obs: vec![vec![0.0; t]; v],
            mean: vec![vec![0.0; t + 1]; v],
            fwd_mean: vec![vec![0.0; t + 1]; v],
            variance: vec![vec![0.0; t + 1]; v],
            fwd_variance: vec![vec![0.0; t + 1]; v],
            zeta: vec![0.0; t],
            e_log_prob: vec![vec![0.0; t]; v],
        }
    }

    /// Kalman forward/backward pass for the posterior variance of word `w`.
    fn compute_post_variance(&mut self, w: usize) {
        let (cv, ov, t) = (self.chain_variance, self.obs_variance, self.t);
        let fv = &mut self.fwd_variance[w];
        fv[0] = cv * INIT_VARIANCE_CONST;
        for i in 1..=t {
            let c = if ov != 0.0 { ov / (fv[i - 1] + cv + ov) } else { 0.0 };
            fv[i] = c * (fv[i - 1] + cv);
        }
        let var = &mut self.variance[w];
        var[t] = self.fwd_variance[w][t];
        for i in (0..t).rev() {
            let fvt = self.fwd_variance[w][i];
            let c = if fvt > 0.0 { (fvt / (fvt + cv)).powi(2) } else { 0.0 };
            var[i] = c * (var[i + 1] - cv) + (1.0 - c) * fvt;
        }
    }

    /// Kalman forward/backward pass for the posterior mean of word `w`
    /// (depends on the current observations `obs[w]`).
    fn compute_post_mean(&mut self, w: usize) {
        let (cv, ov, t) = (self.chain_variance, self.obs_variance, self.t);
        self.fwd_mean[w][0] = 0.0;
        for i in 1..=t {
            let c = ov / (self.fwd_variance[w][i - 1] + cv + ov);
            self.fwd_mean[w][i] = c * self.fwd_mean[w][i - 1] + (1.0 - c) * self.obs[w][i - 1];
        }
        self.mean[w][t] = self.fwd_mean[w][t];
        for i in (0..t).rev() {
            let c = if cv == 0.0 { 0.0 } else { cv / (self.fwd_variance[w][i] + cv) };
            self.mean[w][i] = c * self.fwd_mean[w][i] + (1.0 - c) * self.mean[w][i + 1];
        }
    }

    fn update_zeta(&mut self) {
        for j in 0..self.t {
            let mut s = 0.0;
            for w in 0..self.v {
                s += (self.mean[w][j + 1] + self.variance[w][j + 1] / 2.0).exp();
            }
            self.zeta[j] = s;
        }
    }

    fn compute_expected_log_prob(&mut self) {
        for w in 0..self.v {
            for t in 0..self.t {
                self.e_log_prob[w][t] = self.mean[w][t + 1] - self.zeta[t].ln();
            }
        }
    }

    /// Derivative of the posterior mean sequence of word `w` w.r.t. obs at
    /// `time` (mirrors gensim's `compute_mean_deriv`, which uses the *smoothed*
    /// variance as its forward variance).
    fn compute_mean_deriv(&self, w: usize, time: usize) -> Vec<f64> {
        let (cv, ov, t) = (self.chain_variance, self.obs_variance, self.t);
        let var = &self.variance[w];
        let mut deriv = vec![0.0; t + 1];
        for i in 1..=t {
            let coef = if ov > 0.0 { ov / (var[i - 1] + cv + ov) } else { 0.0 };
            let mut val = coef * deriv[i - 1];
            if time == i - 1 {
                val += 1.0 - coef;
            }
            deriv[i] = val;
        }
        for i in (0..t).rev() {
            let coef = if cv == 0.0 { 0.0 } else { cv / (var[i] + cv) };
            deriv[i] = coef * deriv[i] + (1.0 - coef) * deriv[i + 1];
        }
        deriv
    }
}

/// Posterior mean sequence (length T+1) for an observation vector `obs`
/// (length T), given the precomputed forward variance. Standalone so the obs
/// optimizer can evaluate it without borrowing the whole model.
fn post_mean_from_obs(obs: &[f64], fwd_variance: &[f64], cv: f64, ov: f64) -> Vec<f64> {
    let t = obs.len();
    let mut fwd_mean = vec![0.0; t + 1];
    for i in 1..=t {
        let c = ov / (fwd_variance[i - 1] + cv + ov);
        fwd_mean[i] = c * fwd_mean[i - 1] + (1.0 - c) * obs[i - 1];
    }
    let mut mean = vec![0.0; t + 1];
    mean[t] = fwd_mean[t];
    for i in (0..t).rev() {
        let c = if cv == 0.0 { 0.0 } else { cv / (fwd_variance[i] + cv) };
        mean[i] = c * fwd_mean[i] + (1.0 - c) * mean[i + 1];
    }
    mean
}

/// `f_obs` from gensim: the (negated) state-space bound as a function of the
/// observations, evaluated through the resulting posterior `mean`.
fn f_obs_val(mean: &[f64], variance: &[f64], wc: &[f64], totals: &[f64], zeta: &[f64], cv: f64) -> f64 {
    let t = wc.len();
    let mut term1 = 0.0;
    let mut term2 = 0.0;
    for i in 1..=t {
        let d = mean[i] - mean[i - 1];
        term1 += d * d;
        term2 += wc[i - 1] * mean[i]
            - totals[i - 1] * (mean[i] + variance[i] / 2.0).exp() / zeta[i - 1];
    }
    if cv > 0.0 {
        term1 = -(term1 / (2.0 * cv)) - mean[0] * mean[0] / (2.0 * INIT_VARIANCE_CONST * cv);
    } else {
        term1 = 0.0;
    }
    -(term1 + term2)
}

/// `compute_obs_deriv` from gensim: gradient of the bound w.r.t. each obs entry.
#[allow(clippy::too_many_arguments)]
fn obs_deriv(
    mean: &[f64],
    variance: &[f64],
    mean_deriv_mtx: &[Vec<f64>],
    zeta: &[f64],
    wc: &[f64],
    totals: &[f64],
    cv: f64,
) -> Vec<f64> {
    let t = wc.len();
    let temp_vect: Vec<f64> = (0..t).map(|u| (mean[u + 1] + variance[u + 1] / 2.0).exp()).collect();
    let mut deriv = vec![0.0; t];
    for (slot, md) in deriv.iter_mut().zip(mean_deriv_mtx) {
        let mut term1 = 0.0;
        let mut term2 = 0.0;
        for u in 1..=t {
            term1 += (mean[u] - mean[u - 1]) * (md[u] - md[u - 1]);
            term2 += (wc[u - 1] - totals[u - 1] * temp_vect[u - 1] / zeta[u - 1]) * md[u];
        }
        if cv != 0.0 {
            term1 = -(term1 / cv) - (mean[0] * md[0]) / (INIT_VARIANCE_CONST * cv);
        } else {
            term1 = 0.0;
        }
        *slot = term1 + term2;
    }
    deriv
}

impl Sslm {
    /// Initialize observations from static topic-word counts (gensim
    /// `sslm_counts_init`): a smoothed log-distribution replicated across slices.
    fn counts_init(&mut self, sstats: &[f64]) {
        let total: f64 = sstats.iter().sum();
        let mut log_norm: Vec<f64> = sstats.iter().map(|&s| s / total).collect();
        for x in log_norm.iter_mut() {
            *x += 1.0 / self.v as f64;
        }
        let s2: f64 = log_norm.iter().sum();
        for x in log_norm.iter_mut() {
            *x = (*x / s2).ln();
        }
        for w in 0..self.v {
            for t in 0..self.t {
                self.obs[w][t] = log_norm[w];
            }
        }
        for w in 0..self.v {
            self.compute_post_variance(w);
            self.compute_post_mean(w);
        }
        self.update_zeta();
        self.compute_expected_log_prob();
    }

    /// The state-space variational bound (gensim `compute_bound`), recomputing
    /// the posterior means for the current observations.
    fn compute_bound(&mut self, sstats: &[Vec<f64>], totals: &[f64]) -> f64 {
        let (cv, t, v) = (self.chain_variance, self.t, self.v);
        for w in 0..v {
            self.compute_post_mean(w);
        }
        self.update_zeta();
        let mut val: f64 = (0..v).map(|w| self.variance[w][0] - self.variance[w][t]).sum();
        // (val / 2) * cv, matching gensim's sslm.compute_bound verbatim:
        //   val = sum(variance[w][0] - variance[w][t]) / 2 * chain_variance
        // (ldaseqmodel.py; `/ 2 * cv` is left-associative, i.e. a *multiply* by cv).
        // This leading term multiplies by cv; the per-time-slice term below divides
        // by (2*cv), exactly as gensim does -- the two are deliberately different.
        // Do not "fix" this to / (2*cv): that would diverge from gensim by ~1/cv^2.
        val = val / 2.0 * cv;
        for ti in 1..=t {
            let mut term1 = 0.0;
            let mut term2 = 0.0;
            let mut ent = 0.0;
            for w in 0..v {
                let m = self.mean[w][ti];
                let prev_m = self.mean[w][ti - 1];
                let vr = self.variance[w][ti];
                term1 += (m - prev_m).powi(2) / (2.0 * cv) - vr / cv - cv.ln();
                term2 += sstats[w][ti - 1] * m;
                ent += vr.ln() / 2.0;
            }
            let term3 = -totals[ti - 1] * self.zeta[ti - 1].ln();
            val += term2 + term3 + ent - term1;
        }
        val
    }

    /// Re-estimate the observations (gensim `update_obs`): for each word run the
    /// quasi-Newton optimizer over the bound, with the same objective/gradient
    /// gensim hands to scipy's CG. `zeta` is held fixed across words within the
    /// call, then refreshed at the end.
    fn update_obs(&mut self, sstats: &[Vec<f64>], totals: &[f64]) {
        let (cv, ov, t) = (self.chain_variance, self.obs_variance, self.t);
        let zeta = self.zeta.clone();
        for w in 0..self.v {
            let mut wc = sstats[w].clone();
            let counts_norm = wc.iter().map(|c| c * c).sum::<f64>().sqrt();
            if counts_norm < OBS_NORM_CUTOFF {
                wc = vec![0.0; t];
            }
            let mean_deriv_mtx: Vec<Vec<f64>> =
                (0..t).map(|ti| self.compute_mean_deriv(w, ti)).collect();
            let variance_w = self.variance[w].clone();
            let fwd_variance_w = self.fwd_variance[w].clone();
            let totals_v = totals.to_vec();

            let obj = |x: &[f64]| -> (f64, Vec<f64>) {
                let mean = post_mean_from_obs(x, &fwd_variance_w, cv, ov);
                let f = f_obs_val(&mean, &variance_w, &wc, &totals_v, &zeta, cv);
                let mut d = obs_deriv(&mean, &variance_w, &mean_deriv_mtx, &zeta, &wc, &totals_v, cv);
                for di in d.iter_mut() {
                    *di = -*di; // gensim's df_obs returns the negated gradient
                }
                (f, d)
            };
            let x0 = self.obs[w].clone();
            let result = lbfgs_minimize(x0, obj, 50, 5, 1e-3);
            self.obs[w] = result;
            self.compute_post_mean(w);
        }
        self.update_zeta();
    }

    /// One topic-chain M-step (gensim `fit_sslm`): refresh variances, then a few
    /// observation-update / bound iterations; finish by recomputing E[log p].
    fn fit(&mut self, sstats: &[Vec<f64>]) -> f64 {
        for w in 0..self.v {
            self.compute_post_variance(w);
        }
        let totals: Vec<f64> = (0..self.t)
            .map(|t| (0..self.v).map(|w| sstats[w][t]).sum())
            .collect();
        let mut bound = self.compute_bound(sstats, &totals);
        for _ in 0..2 {
            let old = bound;
            self.update_obs(sstats, &totals);
            bound = self.compute_bound(sstats, &totals);
            if old.abs() > 0.0 && ((bound - old) / old).abs() < 1e-6 {
                break;
            }
        }
        self.compute_expected_log_prob();
        bound
    }

    /// p(w | k, t) for slice `t`: softmax of the posterior mean over the vocab.
    fn topic_dist(&self, t: usize) -> Vec<f64> {
        let logits: Vec<f64> = (0..self.v).map(|w| self.mean[w][t + 1]).collect();
        let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - mx).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / s).collect()
    }
}

// ---------------------------------------------------------------------------
// Per-document variational LDA inference (against one slice's topics)
// ---------------------------------------------------------------------------

/// Inference for one document at time slice `t`. Returns (gamma, phi, lhood),
/// where `phi[n]` is the per-unique-word topic distribution. `topic_elp[k]` is
/// chain k's `e_log_prob` (V × T); `doc` is `(word_id, count)` pairs.
fn fit_lda_post(
    doc: &[(usize, f64)],
    t: usize,
    topic_elp: &[Vec<Vec<f64>>],
    alpha: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let k = topic_elp.len();
    let nwords = doc.len();
    let total: f64 = doc.iter().map(|&(_, c)| c).sum();

    let mut gamma = vec![alpha + total / k as f64; k];
    let mut phi = vec![vec![1.0 / k as f64; k]; nwords];
    let mut log_phi = vec![vec![0.0; k]; nwords];

    let elp = |kk: usize, word: usize| topic_elp[kk][word][t];

    let lhood_of = |gamma: &[f64], phi: &[Vec<f64>], log_phi: &[Vec<f64>]| -> f64 {
        let gamma_sum: f64 = gamma.iter().sum();
        let digsum = digamma(gamma_sum);
        let mut lhood = lgamma(alpha * k as f64) - lgamma(gamma_sum);
        for kk in 0..k {
            let e_log_theta = digamma(gamma[kk]) - digsum;
            let mut term = (alpha - gamma[kk]) * e_log_theta + lgamma(gamma[kk]) - lgamma(alpha);
            for (n, &(word, count)) in doc.iter().enumerate() {
                if phi[n][kk] > 0.0 {
                    term += count * phi[n][kk] * (e_log_theta + elp(kk, word) - log_phi[n][kk]);
                }
            }
            lhood += term;
        }
        lhood
    };

    let update_gamma = |phi: &[Vec<f64>]| -> Vec<f64> {
        let mut g = vec![alpha; k];
        for (n, &(_, count)) in doc.iter().enumerate() {
            for kk in 0..k {
                g[kk] += phi[n][kk] * count;
            }
        }
        g
    };

    let update_phi = |gamma: &[f64], phi: &mut [Vec<f64>], log_phi: &mut [Vec<f64>]| {
        let dig: Vec<f64> = gamma.iter().map(|&g| digamma(g)).collect();
        for (n, &(word, _)) in doc.iter().enumerate() {
            for kk in 0..k {
                log_phi[n][kk] = dig[kk] + elp(kk, word);
            }
            let mut v = log_phi[n][0];
            for kk in 1..k {
                v = log_add(v, log_phi[n][kk]);
            }
            for kk in 0..k {
                log_phi[n][kk] -= v;
                phi[n][kk] = log_phi[n][kk].exp();
            }
        }
    };

    let mut lhood = lhood_of(&gamma, &phi, &log_phi);
    let mut iter = 0;
    loop {
        iter += 1;
        let lhood_old = lhood;
        gamma = update_gamma(&phi);
        update_phi(&gamma, &mut phi, &mut log_phi);
        lhood = lhood_of(&gamma, &phi, &log_phi);
        let converged = if lhood_old != 0.0 {
            ((lhood_old - lhood) / (lhood_old * total)).abs()
        } else {
            1.0
        };
        if (converged < 1e-8 && iter > 1) || iter >= max_iter {
            break;
        }
    }
    (gamma, phi, lhood)
}

// ---------------------------------------------------------------------------
// Dynamic Topic Model
// ---------------------------------------------------------------------------

pub struct DtmModel {
    pub num_topics: usize,
    pub num_times: usize,
    pub num_types: usize,
    pub alpha: f64,
    chains: Vec<Sslm>,
    pub bound: f64,
}

impl DtmModel {
    /// p(w | topic, time) — the topic's word distribution at one slice, length V.
    pub fn topic_word(&self, topic: usize, time: usize) -> Vec<f64> {
        self.chains[topic].topic_dist(time)
    }

    /// The full (num_topics × num_words) topic-word matrix at one slice.
    pub fn topic_word_matrix(&self, time: usize) -> Vec<Vec<f64>> {
        (0..self.num_topics).map(|k| self.topic_word(k, time)).collect()
    }
}

/// Compact collapsed-Gibbs LDA used only to seed the chains with a static
/// topic-word distribution (gensim seeds with a full LdaModel; the DTM EM then
/// refines, so a short run suffices). Returns topic-word counts, shape V×K.
pub(crate) fn init_suffstats<R: Rng>(
    docs: &[Vec<u32>],
    v: usize,
    k: usize,
    iters: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let (alpha, beta) = (0.1f64, 0.1f64);
    let mut nwk = vec![vec![0.0f64; k]; v];
    let mut ndk = vec![vec![0.0f64; k]; docs.len()];
    let mut nk = vec![0.0f64; k];
    let mut z: Vec<Vec<usize>> = docs.iter().map(|d| vec![0usize; d.len()]).collect();
    for (d, doc) in docs.iter().enumerate() {
        for (i, &w) in doc.iter().enumerate() {
            let t = (rng.gen::<f64>() * k as f64) as usize % k;
            z[d][i] = t;
            nwk[w as usize][t] += 1.0;
            ndk[d][t] += 1.0;
            nk[t] += 1.0;
        }
    }
    for _ in 0..iters {
        for (d, doc) in docs.iter().enumerate() {
            for (i, &w) in doc.iter().enumerate() {
                let w = w as usize;
                let old = z[d][i];
                nwk[w][old] -= 1.0;
                ndk[d][old] -= 1.0;
                nk[old] -= 1.0;
                let mut probs = vec![0.0; k];
                let mut total = 0.0;
                for t in 0..k {
                    let p = (nwk[w][t] + beta) / (nk[t] + v as f64 * beta) * (ndk[d][t] + alpha);
                    probs[t] = p;
                    total += p;
                }
                let mut r = rng.gen::<f64>() * total;
                let mut t = 0;
                while t < k - 1 {
                    r -= probs[t];
                    if r <= 0.0 {
                        break;
                    }
                    t += 1;
                }
                z[d][i] = t;
                nwk[w][t] += 1.0;
                ndk[d][t] += 1.0;
                nk[t] += 1.0;
            }
        }
    }
    nwk
}

/// Fit a Dynamic Topic Model. `times[d]` is the slice index (0..num_times) of
/// document `d`. Returns the fitted model. Deterministic for a fixed `rng`.
#[allow(clippy::too_many_arguments)]
pub fn fit_dtm<R: Rng>(
    docs: &[Vec<u32>],
    times: &[usize],
    num_types: usize,
    num_topics: usize,
    num_times: usize,
    alpha: f64,
    chain_variance: f64,
    obs_variance: f64,
    em_iters: usize,
    init_spectral: bool,
    rng: &mut R,
) -> DtmModel {
    let k = num_topics;
    let v = num_types;
    let t = num_times;

    // Seed each topic chain with a static topic-word distribution. With
    // `init_spectral` (the default) we use the deterministic anchor-word spectral
    // init shared with STM/CTM/STS/ECTM: a random static-LDA seed leaves the DTM
    // bound multimodal (issue #216 family), so different seeds land in different
    // basins. `counts_init` normalizes the seed, so passing the spectral β
    // distributions directly is sufficient. Falls back to the random static-LDA
    // seed when the spectral solve is unavailable (e.g. K > vocab) or when the
    // caller asks for it (init="random", which matches gensim's LdaModel seed).
    let seed: Vec<Vec<f64>> = match init_spectral
        .then(|| crate::spectral::spectral_init(docs, k, v))
        .flatten()
    {
        Some(beta) => {
            // spectral_init returns [k][v] distributions; init_suffstats is [v][k].
            let mut s = vec![vec![0.0f64; k]; v];
            for (kk, brow) in beta.iter().enumerate() {
                for (w, &bw) in brow.iter().enumerate() {
                    s[w][kk] = bw;
                }
            }
            s
        }
        None => init_suffstats(docs, v, k, 50, rng),
    };
    let mut chains: Vec<Sslm> = (0..k)
        .map(|kk| {
            let mut chain = Sslm::new(v, t, chain_variance, obs_variance);
            let counts: Vec<f64> = (0..v).map(|w| seed[w][kk]).collect();
            chain.counts_init(&counts);
            chain
        })
        .collect();

    // Documents as (word_id, count) bags, indexed by time slice.
    let bags: Vec<Vec<(usize, f64)>> = docs
        .iter()
        .map(|doc| {
            let mut counts: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
            for &w in doc {
                *counts.entry(w as usize).or_insert(0.0) += 1.0;
            }
            counts.into_iter().collect()
        })
        .collect();

    let mut bound = 0.0;
    let mut lda_max_iter = 25usize;
    for iter in 0..em_iters {
        let old_bound = bound;
        // E-step: per-document inference, accumulate topic suff-stats per slice.
        let topic_elp: Vec<Vec<Vec<f64>>> = chains.iter().map(|c| c.e_log_prob.clone()).collect();
        let mut sstats: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; t]; v]; k];
        let mut doc_bound = 0.0;
        // Per-document inference is independent; run it in parallel and fold the
        // suff-stats in serially (document order) so the fit is bit-for-bit
        // identical regardless of thread count.
        let doc_results: Vec<(usize, Vec<Vec<f64>>, f64)> = bags
            .par_iter()
            .enumerate()
            .map(|(d, bag)| {
                let ti = times[d];
                let (_, phi, lhood) = fit_lda_post(bag, ti, &topic_elp, alpha, lda_max_iter);
                (d, phi, lhood)
            })
            .collect();
        for (d, phi, lhood) in &doc_results {
            let ti = times[*d];
            doc_bound += *lhood;
            for (n, &(word, count)) in bags[*d].iter().enumerate() {
                for kk in 0..k {
                    sstats[kk][word][ti] += count * phi[n][kk];
                }
            }
        }
        // M-step: refit each topic chain to its suff-stats.
        let mut topic_bound = 0.0;
        for kk in 0..k {
            topic_bound += chains[kk].fit(&sstats[kk]);
        }
        bound = doc_bound + topic_bound;
        if bound < old_bound && lda_max_iter < 10 {
            lda_max_iter *= 2; // gensim: back off when the bound dips
        }
        if iter > 0 && old_bound != 0.0 && ((bound - old_bound) / old_bound).abs() < 1e-4 {
            break;
        }
    }

    DtmModel { num_topics: k, num_times: t, num_types: v, alpha, chains, bound }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for DtmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    // The Estimator surface reports the final time slice; full trajectories are
    // available via topic_word_matrix(t).
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word_matrix(self.num_times - 1)
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        // DTM is time-sliced; no static D×K matrix — EXEMPT.
        Vec::new()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
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
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn recovers_drifting_topic() {
        // Two topics. Topic A's vocabulary drifts across 3 slices: at t=0 it uses
        // words {0,1,2}, at t=1 {2,3,4}, at t=2 {4,5,6}. Topic B is stable on
        // {10,11,12}. The DTM should track topic A's drift over time.
        let v = 20;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let a_words = [[0u32, 1, 2], [2, 3, 4], [4, 5, 6]];
        let b_words = [10u32, 11, 12];
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut times: Vec<usize> = Vec::new();
        for slice in 0..3 {
            for _ in 0..60 {
                // topic-A document
                let aw = a_words[slice];
                docs.push(vec![aw[0], aw[1], aw[2], aw[0], aw[1], aw[2]]);
                times.push(slice);
                // topic-B document
                docs.push(vec![b_words[0], b_words[1], b_words[2], b_words[0], b_words[1], b_words[2]]);
                times.push(slice);
            }
        }

        // Looser chain variance: the planted drift is abrupt (disjoint vocab per
        // slice), so the random walk needs room to move between slices.
        let model = fit_dtm(&docs, &times, v, 2, 3, 0.01, 0.5, 0.5, 20, false, &mut rng);

        // Identify the drifting topic as the one whose top word changes across
        // slices; the other should be the stable {10,11,12} topic.
        let top_at = |k: usize, t: usize| -> usize {
            let d = model.topic_word(k, t);
            (0..v).max_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap()).unwrap()
        };
        let drift_topic = (0..2)
            .find(|&k| top_at(k, 0) != top_at(k, 2))
            .expect("one topic should drift");
        let stable_topic = 1 - drift_topic;

        // The drifting topic's mass should move *from* the early block {0,1,2}
        // *to* the late block {4,5,6} as time advances (DTM tracks smooth drift,
        // so we check the direction of the shift rather than absolute mass on an
        // adversarially-disjoint vocabulary).
        let d0 = model.topic_word(drift_topic, 0);
        let d2 = model.topic_word(drift_topic, 2);
        let early_block = |d: &[f64]| -> f64 { [0usize, 1, 2].iter().map(|&w| d[w]).sum() };
        let late_block = |d: &[f64]| -> f64 { [4usize, 5, 6].iter().map(|&w| d[w]).sum() };
        assert!(top_at(drift_topic, 0) <= 2, "t=0 top word not in early block");
        assert!((4..=6).contains(&top_at(drift_topic, 2)), "t=2 top word not in late block");
        assert!(
            late_block(&d2) > late_block(&d0),
            "late-block mass should grow over time: t0={} t2={}",
            late_block(&d0),
            late_block(&d2)
        );
        assert!(
            early_block(&d0) > early_block(&d2),
            "early-block mass should shrink over time: t0={} t2={}",
            early_block(&d0),
            early_block(&d2)
        );

        // The stable topic stays anchored on {10,11,12} throughout (its top word
        // is always in that block).
        for t in 0..3 {
            assert!(
                (10..=12).contains(&top_at(stable_topic, t)),
                "stable topic top word at t={} not in {{10,11,12}}",
                t
            );
        }
    }

    #[test]
    fn spectral_init_is_seed_independent() {
        // The spectral seed is deterministic, so DTM is bit-exact: different
        // seeds give identical chains (no random static-LDA basin scatter).
        let v = 12;
        let docs: Vec<Vec<u32>> = (0..30)
            .map(|d| (0..6).map(|i| ((i + d) % v) as u32).collect())
            .collect();
        let times: Vec<usize> = (0..30).map(|d| d % 3).collect();
        let mut r1 = ChaCha8Rng::seed_from_u64(5);
        let mut r2 = ChaCha8Rng::seed_from_u64(99);
        let m1 = fit_dtm(&docs, &times, v, 2, 3, 0.01, 0.005, 0.5, 8, true, &mut r1);
        let m2 = fit_dtm(&docs, &times, v, 2, 3, 0.01, 0.005, 0.5, 8, true, &mut r2);
        assert_eq!(m1.topic_word(0, 0), m2.topic_word(0, 0));
        assert_eq!(m1.topic_word(1, 2), m2.topic_word(1, 2));
    }

    #[test]
    fn random_init_reproducible_for_fixed_seed() {
        let v = 12;
        let docs: Vec<Vec<u32>> = (0..30)
            .map(|d| (0..6).map(|i| ((i + d) % v) as u32).collect())
            .collect();
        let times: Vec<usize> = (0..30).map(|d| d % 3).collect();
        let mut r1 = ChaCha8Rng::seed_from_u64(5);
        let mut r2 = ChaCha8Rng::seed_from_u64(5);
        let m1 = fit_dtm(&docs, &times, v, 2, 3, 0.01, 0.005, 0.5, 8, false, &mut r1);
        let m2 = fit_dtm(&docs, &times, v, 2, 3, 0.01, 0.005, 0.5, 8, false, &mut r2);
        assert_eq!(m1.topic_word(0, 0), m2.topic_word(0, 0));
        assert_eq!(m1.topic_word(1, 2), m2.topic_word(1, 2));
    }

    #[test]
    fn dtm_conforms() {
        let v = 12;
        let docs: Vec<Vec<u32>> = (0..30)
            .map(|d| (0..6).map(|i| ((i + d) % v) as u32).collect())
            .collect();
        let times: Vec<usize> = (0..30).map(|d| d % 3).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let m = fit_dtm(&docs, &times, v, 2, 3, 0.01, 0.005, 0.5, 8, true, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
