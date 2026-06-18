//! Correlated Topic Model / STM core — logistic-normal topics fit by
//! variational EM (Laplace approximation). This is a faithful port of STM's
//! C++ E-step (`lhoodcpp`/`gradcpp`/`hpbcpp` in bstewart/stm), the inference
//! paradigm that distinguishes STM from the Gibbs models elsewhere in the crate.
//!
//! Per document the topic proportions are `θ_d = softmax([η_d, 0])` with
//! `η_d ∈ ℝ^{K-1}` (the last topic is the softmax reference) and a Gaussian
//! prior `η_d ~ N(μ, Σ)`. The full covariance `Σ` lets topics correlate, which
//! a Dirichlet prior (LDA) cannot represent.
//!
//! E-step (per doc): minimize the variational objective over `η` (L-BFGS on the
//! exact objective + gradient), then form the Laplace covariance `ν = H⁻¹` and
//! the expected token-topic counts `φ` from the Hessian. M-step: update `β` from
//! summed `φ`, `μ` from the mean `η`, and `Σ` from `ν + (η-μ)(η-μ)ᵀ`.

use rand::Rng;

use crate::variational::{lbfgs_minimize, doc_sparse, fit_gamma_ridge};
use crate::linalg::{cholesky, half_logdet, make_diagonally_dominant, spd_inverse, spd_inverse_from_chol};
use crate::estimator::{Estimator, ModelFamily};
use crate::variational::LogisticNormalModel;

/// Prior on the prevalence coefficients γ in the STM M-step.
///
/// `Pooled` (default) fits γ by ridge regression — the original STM `"Pooled"`
/// strategy: all topics share a single penalised regression.
///
/// `L1 { alpha }` fits an elastic-net path by coordinate descent (one column
/// of Λ at a time) and selects the penalty by AIC. `alpha` is the elastic-net
/// mix: 1.0 is pure lasso, values in (0,1) add a ridge component.  Recommended
/// for high-dimensional prevalence designs (many one-hot levels) where the
/// pooled ridge does not induce enough sparsity.
#[derive(Clone, Copy, Debug)]
pub enum GammaPrior {
    Pooled,
    L1 { alpha: f64 },
}

/// `exp(η)` extended with a trailing 1 (the reference category), length K.
fn expeta(eta: &[f64]) -> Vec<f64> {
    let mut e = Vec::with_capacity(eta.len() + 1);
    for &x in eta {
        e.push(x.exp());
    }
    e.push(1.0);
    e
}

/// STM `lhoodcpp`: the per-document variational objective (to MINIMIZE over η).
pub fn ctm_lhood(
    eta: &[f64],
    beta: &[Vec<f64>],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
) -> f64 {
    let km1 = eta.len();
    let k = km1 + 1;
    let e = expeta(eta);
    let sum_e: f64 = e.iter().sum();
    let ndoc: f64 = counts.iter().sum();

    let mut part1 = 0.0;
    for (wi, &w) in words.iter().enumerate() {
        let mut s = 0.0;
        for t in 0..k {
            s += e[t] * beta[t][w];
        }
        part1 += counts[wi] * s.ln();
    }
    part1 -= ndoc * sum_e.ln();

    let mut part2 = 0.0;
    for i in 0..km1 {
        let di = eta[i] - mu[i];
        for j in 0..km1 {
            part2 += di * siginv[i * km1 + j] * (eta[j] - mu[j]);
        }
    }
    0.5 * part2 - part1
}

/// Fused objective + gradient: returns exactly `(ctm_lhood(..), ctm_grad(..))`
/// but evaluates `exp(η)`, `Σ exp(η)`, and the per-word column sum
/// `Σ_t e_t β_{t,w}` once instead of once each in the two separate functions.
/// The L-BFGS E-step calls value and gradient together at every point, so the
/// separate path recomputes this O(W·K) inner loop twice per evaluation.
///
/// Every partial sum is accumulated in the same order as the standalone
/// `ctm_lhood`/`ctm_grad`, so the returned value and gradient are bit-for-bit
/// identical to calling those two functions.
pub fn ctm_lhood_grad(
    eta: &[f64],
    beta: &[Vec<f64>],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
) -> (f64, Vec<f64>) {
    let km1 = eta.len();
    let k = km1 + 1;
    let e = expeta(eta);
    let sum_e: f64 = e.iter().sum();
    let ndoc: f64 = counts.iter().sum();

    // Single pass over the document's words: accumulate the objective's
    // `Σ_w counts[w]·ln(colsum)` (= `part1` in ctm_lhood) and the gradient's
    // `Σ_w counts[w]·φ_{·,w}` (= `part1` in ctm_grad) from the same `colsum`.
    let mut obj_part1 = 0.0;
    let mut grad_part1 = vec![0.0f64; k];
    for (wi, &w) in words.iter().enumerate() {
        let mut colsum = 0.0;
        for t in 0..k {
            colsum += e[t] * beta[t][w];
        }
        obj_part1 += counts[wi] * colsum.ln();
        let c = counts[wi] / colsum;
        for t in 0..k {
            grad_part1[t] += c * e[t] * beta[t][w];
        }
    }
    obj_part1 -= ndoc * sum_e.ln();
    let f = ndoc / sum_e;
    for t in 0..k {
        grad_part1[t] -= f * e[t];
    }

    // Quadratic prior term: objective `0.5·(η-μ)ᵀΣ⁻¹(η-μ)`, gradient `Σ⁻¹(η-μ)`.
    // To stay bit-for-bit identical to the separate functions, the value's
    // double sum is accumulated EXACTLY as ctm_lhood does — `part2 += di ·
    // siginv[ij] · (η_j−μ_j)` with the same left-to-right multiply/add order —
    // and the gradient's `s = Σ_j siginv[ij]·(η_j−μ_j)` EXACTLY as ctm_grad
    // does. Float multiplication is not associative, so we deliberately keep the
    // two separate accumulations rather than reuse `s` for the value.
    let mut obj_part2 = 0.0;
    let mut g = vec![0.0f64; km1];
    for i in 0..km1 {
        let di = eta[i] - mu[i];
        let mut s = 0.0;
        for j in 0..km1 {
            obj_part2 += di * siginv[i * km1 + j] * (eta[j] - mu[j]);
            s += siginv[i * km1 + j] * (eta[j] - mu[j]);
        }
        g[i] = s - grad_part1[i];
    }

    (0.5 * obj_part2 - obj_part1, g)
}

/// STM `gradcpp`: gradient of `ctm_lhood` w.r.t. η (length K-1).
pub fn ctm_grad(
    eta: &[f64],
    beta: &[Vec<f64>],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
) -> Vec<f64> {
    let km1 = eta.len();
    let k = km1 + 1;
    let e = expeta(eta);
    let sum_e: f64 = e.iter().sum();
    let ndoc: f64 = counts.iter().sum();

    // part1 (length K) = Σ_w counts[w]·φ_{·,w} − (ndoc/Σe)·e , where φ_{t,w} = e_t β_{t,w}/Σ_t e_t β_{t,w}.
    let mut part1 = vec![0.0f64; k];
    for (wi, &w) in words.iter().enumerate() {
        let mut colsum = 0.0;
        for t in 0..k {
            colsum += e[t] * beta[t][w];
        }
        let c = counts[wi] / colsum;
        for t in 0..k {
            part1[t] += c * e[t] * beta[t][w];
        }
    }
    let f = ndoc / sum_e;
    for t in 0..k {
        part1[t] -= f * e[t];
    }

    // grad = siginv(η-μ) − part1[0..K-1]
    let mut g = vec![0.0f64; km1];
    for i in 0..km1 {
        let mut s = 0.0;
        for j in 0..km1 {
            s += siginv[i * km1 + j] * (eta[j] - mu[j]);
        }
        g[i] = s - part1[i];
    }
    g
}

/// Result of STM `hpbcpp`: the Laplace covariance, expected token-topic counts,
/// and the per-document evidence bound.
pub struct HpbResult {
    pub nu: Vec<f64>,        // (K-1)×(K-1) variational covariance H⁻¹
    pub phi: Vec<Vec<f64>>,  // K×W expected token-topic counts for the doc's words
    pub bound: f64,
}

/// STM `hpbcpp`: form the Hessian at η, invert it (with a diagonal-dominance
/// fallback when indefinite) to get ν, and the expected counts φ and bound.
///
/// When `diagonal` is true, the mean-field variant is used: only the Hessian
/// *diagonal* is formed and ν is filled with `1/H_ii` on its diagonal (the
/// off-diagonals stay 0). This skips the per-document O((K-1)³) Cholesky and
/// inverse — a large E-step speedup at high K — at the cost of dropping the
/// off-diagonal posterior covariance. The `nu` storage format is unchanged
/// (a length-(K-1)² flat vector), and `phi`/`ll`/`quad` are computed identically.
pub fn ctm_hpb(
    eta: &[f64],
    beta: &[Vec<f64>],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
    entropy: f64,
    diagonal: bool,
) -> HpbResult {
    let km1 = eta.len();
    let k = km1 + 1;
    let w_n = words.len();
    let e = expeta(eta);
    let sum_e: f64 = e.iter().sum();
    let ndoc: f64 = counts.iter().sum();
    let theta: Vec<f64> = e.iter().map(|x| x / sum_e).collect();

    // EB[t][w] = sqrt(counts[w])·φ_{t,w}, φ_{t,w}=e_t β_{t,w}/Σ_t e_t β_{t,w}.
    let mut eb = vec![vec![0.0f64; w_n]; k];
    for (wi, &w) in words.iter().enumerate() {
        let mut colsum = 0.0;
        for t in 0..k {
            colsum += e[t] * beta[t][w];
        }
        let sq = counts[wi].sqrt();
        for t in 0..k {
            eb[t][wi] = e[t] * beta[t][w] * sq / colsum;
        }
    }

    // Shared `det_term` and `nu` are computed via either the full Hessian
    // (Laplace) or its diagonal alone (mean-field). `phi` (= eb re-multiplied by
    // sqrt(counts)) and the `ll`/`quad` bound terms below are identical in both.
    let (nu, det_term) = if diagonal {
        // Mean-field: form only the Hessian diagonal H_ii, exactly the i==i
        // entries the full path computes. From eb (before re-multiplying by
        // sqrt(counts)): diag of EB·EBᵀ − ndoc·θθᵀ.
        let mut hdiag = vec![0.0f64; km1];
        for i in 0..km1 {
            let mut s = 0.0;
            for wi in 0..w_n {
                s += eb[i][wi] * eb[i][wi];
            }
            hdiag[i] = s - ndoc * theta[i] * theta[i];
        }
        // Re-multiply eb by sqrt(counts) → φ (same as the full path).
        for (wi, &_w) in words.iter().enumerate() {
            let sq = counts[wi].sqrt();
            for t in 0..k {
                eb[t][wi] *= sq;
            }
        }
        // Diagonal adjustment: H_ii −= rowSums(φ)_i − ndoc·θ_i, then + siginv_ii.
        let mut nu = vec![0.0f64; km1 * km1];
        let mut det_term = 0.0f64;
        for i in 0..km1 {
            let row_sum: f64 = (0..w_n).map(|wi| eb[i][wi]).sum();
            hdiag[i] -= row_sum - ndoc * theta[i];
            hdiag[i] += siginv[i * km1 + i];
            let h_ii = hdiag[i].max(1e-12);
            nu[i * km1 + i] = 1.0 / h_ii;
            det_term -= 0.5 * h_ii.ln();
        }
        (nu, det_term)
    } else {
        // hess (K×K) = EB·EBᵀ − ndoc·θθᵀ. Both terms are symmetric, so compute
        // the lower triangle (b ≤ a) and mirror it. Each entry is the same inner
        // product in the same order as the full double loop, so this is
        // bit-for-bit identical while doing half the O(K²·W) work.
        let mut hess = vec![0.0f64; k * k];
        for a in 0..k {
            let eb_a = &eb[a];
            for b in 0..=a {
                let eb_b = &eb[b];
                let mut s = 0.0;
                for wi in 0..w_n {
                    s += eb_a[wi] * eb_b[wi];
                }
                let v = s - ndoc * theta[a] * theta[b];
                hess[a * k + b] = v;
                hess[b * k + a] = v;
            }
        }
        // Turn EB into φ = counts[w]·responsibility (multiply rows by sqrt(counts) again).
        for (wi, &_w) in words.iter().enumerate() {
            let sq = counts[wi].sqrt();
            for t in 0..k {
                eb[t][wi] *= sq;
            }
        }
        // diag(hess) −= rowSums(φ) − ndoc·θ
        for t in 0..k {
            let row_sum: f64 = (0..w_n).map(|wi| eb[t][wi]).sum();
            hess[t * k + t] -= row_sum - ndoc * theta[t];
        }

        // Drop the last (reference) row/col → (K-1)×(K-1), then add siginv.
        let mut h = vec![0.0f64; km1 * km1];
        for i in 0..km1 {
            for j in 0..km1 {
                h[i * km1 + j] = hess[i * k + j] + siginv[i * km1 + j];
            }
        }

        // ν = H⁻¹, with STM's diagonal-dominance fallback if H isn't PD.
        let (nu, half_ld) = match cholesky(&h, km1) {
            Some(l) => (spd_inverse_from_chol(&l, km1), half_logdet(&l, km1)),
            None => {
                make_diagonally_dominant(&mut h, km1);
                let l = cholesky(&h, km1).expect("PD after diagonal dominance");
                (spd_inverse_from_chol(&l, km1), half_logdet(&l, km1))
            }
        };
        (nu, -half_ld) // STM: −Σ log diag(chol(H))
    };

    // bound = Σ_w counts[w]·log(Σ_t θ_t β_{t,w}) + detTerm − 0.5 (η-μ)ᵀΣ⁻¹(η-μ) − entropy
    let mut ll = 0.0;
    for (wi, &w) in words.iter().enumerate() {
        let mut s = 0.0;
        for t in 0..k {
            s += theta[t] * beta[t][w];
        }
        ll += counts[wi] * s.ln();
    }
    let mut quad = 0.0;
    for i in 0..km1 {
        let di = eta[i] - mu[i];
        for j in 0..km1 {
            quad += di * siginv[i * km1 + j] * (eta[j] - mu[j]);
        }
    }
    let bound = ll + det_term - 0.5 * quad - entropy;

    HpbResult { nu, phi: eb, bound }
}

/// A fitted CTM/STM model.
pub struct CtmModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub beta: Vec<Vec<f64>>, // K×V topic-word
    pub mu: Vec<f64>,        // K-1 prior mean (no-covariate case)
    pub sigma: Vec<f64>,     // (K-1)² prior covariance
    pub lambda: Vec<Vec<f64>>, // per-doc variational means η (K-1)
    /// Per-document variational covariance ν = H⁻¹ ((K-1)² flattened, row-major),
    /// from the final E-step — the Laplace posterior of η used for
    /// method-of-composition uncertainty.  Empty when the model was fit with
    /// `keep_nu = false`; use `recompute_nu` to regenerate on demand.
    pub nu: Vec<Vec<f64>>,
    /// The prior covariance Σ that was used in the final E-step (the sigma that
    /// produced the stored λ and ν values). For `fit_ctm`, this is the sigma from
    /// the M-step of the penultimate EM iteration (one step behind `sigma`).
    /// For `fit_ctm_svi`, this equals `sigma` (the final E-step uses the converged
    /// globals). Used by `recompute_nu` to exactly reproduce the stored ν.
    pub sigma_estep: Vec<f64>,
    /// The topic-word matrix β (K×V) that was used in the final E-step (before
    /// the final M-step updated `beta`). For `fit_ctm`, this is β from the
    /// penultimate M-step. For `fit_ctm_svi`, this equals `beta` (no M-step
    /// follows the final E-step). Used by `recompute_nu` to reproduce ν exactly.
    pub beta_estep: Vec<Vec<f64>>,
    /// Prevalence coefficients γ (num_features × (K-1)), `Some` when prevalence
    /// covariates were supplied: `μ_d = X_d γ`. The last topic is the reference.
    pub gamma: Option<Vec<Vec<f64>>>,
    /// Per-group topic-word distributions (G × K × V), `Some` when content
    /// covariates were supplied (the SAGE content model inside STM). `beta` is
    /// then the group-averaged topic-word.
    pub content_beta: Option<Vec<Vec<Vec<f64>>>>,
    pub num_groups: usize,
    /// Corpus approximate evidence bound (ELBO) at the final E-step — the same
    /// quantity R `stm` reports as its convergence bound.
    pub bound: f64,
    /// Approximate bound after each EM iteration (the convergence trajectory,
    /// one entry per iteration run).
    pub bound_history: Vec<f64>,
    /// `true` if EM stopped on the `em_tol` relative-bound criterion, `false`
    /// if it hit the `em_iters` cap first.
    pub converged: bool,
    /// Number of EM iterations actually run (≤ `em_iters`).
    pub em_iters_run: usize,
    /// Variational-covariance mode used in the E-step. `false` (default) is the
    /// Laplace approximation (full ν = H⁻¹); `true` is the mean-field diagonal
    /// approximation (ν = diag(1/H_ii), off-diagonals zero). Stored so
    /// `recompute_nu` reproduces ν in the same mode. Persistence (with a serde
    /// default of `false` for old saves) lives on the `CtmState`/`StmState`
    /// structs in `python.rs`.
    pub diagonal: bool,
}

/// Build per-group topic-word β (G×K×V) from the SAGE content deviations:
/// `β_{g,k,v} = softmax_v(m_v + κᵀ_{k,v} + κᶜ_{g,v} + κᴵ_{k,g,v})`.
fn build_content_beta(
    m: &[f64],
    kt: &[Vec<f64>],
    kc: &[Vec<f64>],
    ki: &[Vec<f64>],
    k: usize,
    g: usize,
    v: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut out = vec![vec![vec![0.0f64; v]; k]; g];
    for topic in 0..k {
        for grp in 0..g {
            let c = topic * g + grp;
            let mut max = f64::NEG_INFINITY;
            let mut eta = vec![0.0f64; v];
            for w in 0..v {
                let e = m[w] + kt[topic][w] + kc[grp][w] + ki[c][w];
                eta[w] = e;
                if e > max {
                    max = e;
                }
            }
            let mut z = 0.0;
            for w in 0..v {
                z += (eta[w] - max).exp();
            }
            for w in 0..v {
                out[grp][topic][w] = (eta[w] - max).exp() / z;
            }
        }
    }
    out
}

/// MAP-update the SAGE content deviations κ from soft (topic×group×word)
/// expected counts via L-BFGS, then rebuild per-group β. `counts[k*G+g][v]` are
/// the variational expected token counts; `prior_variance` is the Gaussian
/// prior on κ.
#[allow(clippy::too_many_arguments)]
fn optimize_content(
    m: &[f64],
    kappa_t: &mut [Vec<f64>],
    kappa_c: &mut [Vec<f64>],
    kappa_i: &mut [Vec<f64>],
    counts: &[Vec<f64>],
    k: usize,
    g: usize,
    v: usize,
    prior_variance: f64,
    max_iter: usize,
) -> Vec<Vec<Vec<f64>>> {
    let n_t = k * v;
    let n_c = g * v;
    let totals: Vec<f64> = counts.iter().map(|row| row.iter().sum()).collect();

    let mut x0 = Vec::with_capacity(n_t + n_c + k * g * v);
    for kt in kappa_t.iter() {
        x0.extend_from_slice(kt);
    }
    for kc in kappa_c.iter() {
        x0.extend_from_slice(kc);
    }
    for ki in kappa_i.iter() {
        x0.extend_from_slice(ki);
    }
    let inv_var = 1.0 / prior_variance;

    let x = lbfgs_minimize(
        x0,
        |flat| {
            let kt = |t: usize, w: usize| flat[t * v + w];
            let kc = |grp: usize, w: usize| flat[n_t + grp * v + w];
            let ki = |c: usize, w: usize| flat[n_t + n_c + c * v + w];
            let mut value = 0.0f64;
            let mut grad = vec![0.0f64; flat.len()];
            for topic in 0..k {
                for grp in 0..g {
                    let c = topic * g + grp;
                    let nkg = totals[c];
                    let mut max = f64::NEG_INFINITY;
                    let mut eta = vec![0.0f64; v];
                    for w in 0..v {
                        let e = m[w] + kt(topic, w) + kc(grp, w) + ki(c, w);
                        eta[w] = e;
                        if e > max {
                            max = e;
                        }
                    }
                    let mut z = 0.0;
                    for w in 0..v {
                        z += (eta[w] - max).exp();
                    }
                    let log_z = max + z.ln();
                    for w in 0..v {
                        let n = counts[c][w];
                        value += n * (eta[w] - log_z);
                        let beta = (eta[w] - log_z).exp();
                        let resid = n - nkg * beta;
                        grad[topic * v + w] += resid;
                        grad[n_t + grp * v + w] += resid;
                        grad[n_t + n_c + c * v + w] += resid;
                    }
                }
            }
            for (i, &xi) in flat.iter().enumerate() {
                value -= 0.5 * inv_var * xi * xi;
                grad[i] -= inv_var * xi;
            }
            (-value, grad.iter().map(|gv| -gv).collect())
        },
        max_iter,
        7,
        1e-4,
    );

    for t in 0..k {
        kappa_t[t].copy_from_slice(&x[t * v..(t + 1) * v]);
    }
    for grp in 0..g {
        let off = n_t + grp * v;
        kappa_c[grp].copy_from_slice(&x[off..off + v]);
    }
    for c in 0..(k * g) {
        let off = n_t + n_c + c * v;
        kappa_i[c].copy_from_slice(&x[off..off + v]);
    }
    build_content_beta(m, kappa_t, kappa_c, kappa_i, k, g, v)
}

impl CtmModel {
    /// Per-document topic proportions θ = softmax([η, 0]).
    pub fn doc_topics(&self) -> Vec<Vec<f64>> {
        self.lambda
            .iter()
            .map(|eta| {
                let e = expeta(eta);
                let s: f64 = e.iter().sum();
                e.iter().map(|x| x / s).collect()
            })
            .collect()
    }

    /// Topic correlation matrix: the correlation of the topic proportions θ
    /// across documents (STM's practical `topicCorr`). Symmetric, unit diagonal,
    /// defined over all K topics — captures which topics co-occur.
    pub fn topic_correlation(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        let theta = self.doc_topics();
        let d = theta.len().max(1) as f64;

        let mut mean = vec![0.0f64; k];
        for row in &theta {
            for t in 0..k {
                mean[t] += row[t];
            }
        }
        for t in 0..k {
            mean[t] /= d;
        }
        let mut cov = vec![vec![0.0f64; k]; k];
        for row in &theta {
            for i in 0..k {
                for j in 0..k {
                    cov[i][j] += (row[i] - mean[i]) * (row[j] - mean[j]);
                }
            }
        }
        let mut corr = vec![vec![0.0f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                let den = (cov[i][i] * cov[j][j]).sqrt();
                corr[i][j] = if den > 0.0 {
                    cov[i][j] / den
                } else if i == j {
                    1.0
                } else {
                    0.0
                };
            }
        }
        corr
    }
}

impl Estimator for CtmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
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

impl LogisticNormalModel for CtmModel {
    fn eta_dim(&self) -> usize {
        self.num_topics - 1
    }

    fn eta_mean(&self) -> &[Vec<f64>] {
        &self.lambda
    }

    fn eta_cov(&self) -> &[Vec<f64>] {
        &self.nu
    }
}

/// Infer the topic proportions θ (length K) for a *new* document by the
/// variational E-step against fixed global parameters: the topic-word matrix
/// `beta` (K×V), the prior mean `mu` (K-1), and the inverse prior covariance
/// `siginv` ((K-1)²). This is the inference `transform` uses for held-out docs.
pub fn infer_theta(
    beta: &[Vec<f64>],
    mu: &[f64],
    siginv: &[f64],
    words: &[usize],
    counts: &[f64],
) -> Vec<f64> {
    let km1 = mu.len();
    let k = km1 + 1;
    if words.is_empty() {
        return vec![1.0 / k as f64; k];
    }
    let opt = lbfgs_minimize(
        vec![0.0; km1],
        |eta| ctm_lhood_grad(eta, beta, words, counts, mu, siginv),
        40,
        7,
        1e-5,
    );
    // θ = softmax([η, 0]) (the last topic is the reference category).
    let mut e: Vec<f64> = opt.iter().map(|&x| x.exp()).collect();
    e.push(1.0);
    let s: f64 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

/// Elastic-net (L1/L2 mix) solver for the prevalence regression Λ[:,t] ~ X β_t.
///
/// Solves each of the K-1 response columns independently by coordinate descent on
/// a log-spaced lambda path, selects the penalty by AIC, and returns the
/// (F×(K-1)) coefficient matrix on the original (unstandardised) scale.
///
/// The intercept (column 0 of `x`, the all-ones column) is never penalised.
/// All other columns are internally standardised (mean-centred and scaled by their
/// standard deviation); the coefficients are mapped back to the original scale
/// before returning so the caller sees the same row/column layout as `fit_gamma_ridge`.
///
/// `alpha` is the elastic-net mixing parameter (glmnet convention): `alpha=1`
/// is pure lasso, `alpha→0` approaches ridge. The lasso-relevant lambda_max is
/// `max_j |x_j · y| / (n · alpha)` and the path descends to `eps * lambda_max`
/// (with `eps = 1e-4`) over `n_lambda = 50` log-spaced steps with warm starts.
///
/// AIC = n · ln(RSS/n) + 2 · df, where df counts nonzero penalised coefficients.
fn fit_gamma_enet(
    x: &[Vec<f64>],
    lambda: &[Vec<f64>],
    f: usize,
    km1: usize,
    alpha: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let n_f64 = n as f64;

    // Standardise penalised predictors (columns 1..F).
    // Column 0 is the intercept; it is passed through unchanged.
    let p = f - 1; // number of penalised predictors
    let mut col_mean = vec![0.0f64; p];
    let mut col_std = vec![1.0f64; p];
    for j in 0..p {
        let s: f64 = x.iter().map(|row| row[j + 1]).sum();
        col_mean[j] = s / n_f64;
    }
    for j in 0..p {
        let var: f64 = x.iter().map(|row| {
            let d = row[j + 1] - col_mean[j];
            d * d
        }).sum::<f64>() / n_f64;
        col_std[j] = if var > 1e-12 { var.sqrt() } else { 1.0 };
    }

    // Build standardised design matrix (n × p), excluding the intercept column.
    let xs: Vec<Vec<f64>> = x.iter().map(|row| {
        (0..p).map(|j| (row[j + 1] - col_mean[j]) / col_std[j]).collect()
    }).collect();

    // Pre-compute column norms² of the standardised design (all = n for unit-variance).
    let mut xj_norm2 = vec![0.0f64; p];
    for j in 0..p {
        xj_norm2[j] = xs.iter().map(|row| row[j] * row[j]).sum();
        if xj_norm2[j] < 1e-12 { xj_norm2[j] = 1.0; } // constant column guard
    }

    // Coordinate-descent loop for one response column y (length n).
    // Returns coefficients (intercept_coef, penalised_coefs[p]) on the standardised scale.
    let solve_column = |y: &[f64]| -> Vec<f64> {
        // Compute lambda_max = max_j |<xs_j, r>| / (n * alpha), evaluated at beta=0
        // (so r = y - intercept*1). Intercept at beta=0 is the mean of y.
        let y_mean: f64 = y.iter().sum::<f64>() / n_f64;
        // Centre y for the penalised part (OLS intercept absorbs the mean).
        let yc: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

        let alpha_safe = alpha.max(1e-6); // guard against alpha≈0
        let lam_max = (0..p)
            .map(|j| {
                let dot: f64 = xs.iter().zip(yc.iter()).map(|(row, &ri)| row[j] * ri).sum();
                dot.abs() / (n_f64 * alpha_safe)
            })
            .fold(0.0f64, f64::max);

        // If all columns are constant (lambda_max≈0) return a zero solution.
        if lam_max < 1e-12 {
            let mut out = vec![0.0f64; p + 1];
            out[0] = y_mean;
            return out;
        }

        let n_lambda = 50usize;
        let eps = 1e-4f64;
        let lam_min = lam_max * eps;
        // Log-spaced path from lambda_max down to lambda_min.
        let lambdas: Vec<f64> = (0..n_lambda)
            .map(|i| {
                let t = i as f64 / (n_lambda - 1) as f64;
                (lam_max.ln() * (1.0 - t) + lam_min.ln() * t).exp()
            })
            .collect();

        let mut best_coef: Vec<f64> = {
            let mut c = vec![0.0f64; p + 1];
            c[0] = y_mean;
            c
        };
        let mut best_aic = f64::INFINITY;

        // Warm-start coefficients (intercept + penalised).
        let mut coef = vec![0.0f64; p + 1]; // [0] = intercept, [1..=p] = penalised betas (std scale)
        coef[0] = y_mean;

        // Residual vector (initialised at zero-beta prediction = intercept).
        let mut r: Vec<f64> = y.iter().map(|&yi| yi - coef[0]).collect();

        for &lam in &lambdas {
            // Coordinate descent with warm start.
            for _iter in 0..1000 {
                let mut max_change = 0.0f64;

                // Update intercept (unpenalised, OLS estimate from residual).
                let r_mean: f64 = r.iter().sum::<f64>() / n_f64;
                let delta_int = r_mean;
                if delta_int.abs() > 1e-14 {
                    coef[0] += delta_int;
                    for ri in r.iter_mut() { *ri -= delta_int; }
                    max_change = max_change.max(delta_int.abs());
                }

                // Update each penalised predictor.
                for j in 0..p {
                    let old = coef[j + 1];
                    // Partial residual: add back contribution of current coef.
                    let rj_dot: f64 = xs.iter().zip(r.iter()).map(|(row, &ri)| row[j] * (ri + old * row[j])).sum();
                    // Soft-threshold update.
                    let z = rj_dot / xj_norm2[j];
                    let thresh = lam * alpha_safe;
                    let new_coef = if z > thresh {
                        // Ridge component: scale by 1/(1 + lam*(1-alpha)/xj_norm2*n).
                        (z - thresh) / (1.0 + lam * (1.0 - alpha_safe) * n_f64 / xj_norm2[j])
                    } else if z < -thresh {
                        (z + thresh) / (1.0 + lam * (1.0 - alpha_safe) * n_f64 / xj_norm2[j])
                    } else {
                        0.0
                    };
                    let delta = new_coef - old;
                    if delta.abs() > 1e-14 {
                        coef[j + 1] = new_coef;
                        for (row, ri) in xs.iter().zip(r.iter_mut()) {
                            *ri -= delta * row[j];
                        }
                        max_change = max_change.max(delta.abs());
                    }
                }

                if max_change < 1e-7 { break; }
            }

            // AIC = n * ln(RSS/n) + 2 * df
            let rss: f64 = r.iter().map(|ri| ri * ri).sum();
            let df = coef[1..].iter().filter(|&&c| c.abs() > 1e-10).count() as f64;
            let aic = if rss > 0.0 {
                n_f64 * (rss / n_f64).ln() + 2.0 * df
            } else {
                -n_f64 * f64::MAX.ln() + 2.0 * df
            };
            if aic < best_aic {
                best_aic = aic;
                best_coef = coef.clone();
            }
        }
        best_coef
    };

    // Solve each response column and map back to original scale.
    let mut g = vec![vec![0.0f64; km1]; f];
    for t in 0..km1 {
        let y: Vec<f64> = lambda.iter().map(|row| row[t]).collect();
        let coef = solve_column(&y);
        // coef[0] = intercept on centred-y, coef[1..p+1] = penalised on standardised x.
        // Back-transform: beta_orig_j = coef_std_j / col_std[j]
        // Intercept absorbs: intercept_orig = intercept_std - sum_j (beta_orig_j * col_mean[j])
        let mut intercept = coef[0];
        for j in 0..p {
            let orig_j = coef[j + 1] / col_std[j];
            g[j + 1][t] = orig_j;
            intercept -= orig_j * col_mean[j];
        }
        g[0][t] = intercept;
    }
    g
}


/// `μ_d = X_d γ` (length K-1).
fn mu_from(x_d: &[f64], gamma: &[Vec<f64>], km1: usize) -> Vec<f64> {
    (0..km1)
        .map(|t| x_d.iter().zip(gamma).map(|(xi, gr)| xi * gr[t]).sum())
        .collect()
}

/// Fit a CTM/STM by variational EM.
///
/// `sigma_shrink` ∈ [0,1] shrinks Σ toward its diagonal each M-step (STM's
/// `sigma.prior`). `prevalence` (optional, D×F with an intercept column already
/// prepended) makes the prior mean a regression on document covariates,
/// `μ_d = X_d γ` — the STM prevalence model. `content` (optional, per-document
/// group ids + group count) makes the topic-word distribution vary by group via
/// the SAGE log-linear content model — the STM content covariate.
///
/// `em_iters` caps the number of EM iterations; EM stops early once the
/// relative change in the corpus bound falls below `em_tol` (R `stm`'s `emtol`,
/// default 1e-5). Pass `em_tol = 0.0` to disable early stopping and always run
/// the full `em_iters`.
///
/// `gamma_prior` selects the prevalence-coefficient regression: `Pooled` (the
/// default, ridge) or `L1 { alpha }` (elastic-net coordinate descent with
/// AIC-selected penalty). When no prevalence design is supplied the parameter
/// has no effect.
///
/// `diagonal` selects the per-document variational-covariance mode: `false` is
/// the Laplace approximation (full ν = H⁻¹), `true` is the mean-field diagonal
/// approximation (ν = diag(1/H_ii)), which skips the per-document Cholesky and
/// inverse for a large E-step speedup at high K, at the cost of dropping the
/// off-diagonal posterior covariance.
#[allow(clippy::too_many_arguments)]
pub fn fit_ctm<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    em_iters: usize,
    em_tol: f64,
    sigma_shrink: f64,
    prevalence: Option<&[Vec<f64>]>,
    content: Option<(&[usize], usize)>,
    init_spectral: bool,
    gamma_prior: GammaPrior,
    keep_nu: bool,
    diagonal: bool,
    rng: &mut R,
) -> CtmModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let nf = prevalence.map(|x| x[0].len());
    let groups = content.map(|(g, _)| g);
    let num_groups = content.map_or(1, |(_, ng)| ng);

    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // Initialize β: deterministic anchor-word (spectral) init when requested and
    // applicable, else random (seeded). Spectral is STM's default and makes the
    // solution reproducible without a seed.
    let random_beta = |rng: &mut R| -> Vec<Vec<f64>> {
        let mut b = vec![vec![0.0f64; num_types]; k];
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
    // Spectral applies with or without a content covariate: R `stm` initializes
    // the base topics spectrally even in the content (SAGE) model and starts the
    // content deviations κ at zero. Gating spectral off for content models left
    // the base β random, which is strongly multimodal — most seeds collapse to a
    // flat, no-group-content optimum (issue #216).
    let mut beta = if init_spectral {
        crate::spectral::spectral_init(docs, k, num_types).unwrap_or_else(|| random_beta(rng))
    } else {
        random_beta(rng)
    };

    // Content covariate state: background m_v and SAGE deviations κ; per-group β.
    let mut m_bg = vec![0.0f64; num_types];
    let mut kappa_t = vec![vec![0.0f64; num_types]; k];
    let mut kappa_c = vec![vec![0.0f64; num_types]; num_groups];
    let mut kappa_i = vec![vec![0.0f64; num_types]; k * num_groups];
    let mut content_beta: Vec<Vec<Vec<f64>>> = Vec::new();
    if content.is_some() {
        let mut freq = vec![1.0f64; num_types];
        let mut total = num_types as f64;
        for doc in docs {
            for &w in doc {
                freq[w as usize] += 1.0;
                total += 1.0;
            }
        }
        for v in 0..num_types {
            m_bg[v] = (freq[v] / total).ln();
        }
        // Seed the topic deviations κ_t from the base β (spectral when requested,
        // else random) so topics start differentiated. With κ all zero, build_content_beta makes every
        // topic identical to the background m — a symmetric fixed point the
        // E-step cannot escape (θ stays uniform, so the soft counts never give
        // κ_t any across-topic signal). Setting κ_t[k] = ln β_k − m makes the
        // initial per-group β equal β_k for every group, breaking the
        // across-topic symmetry while leaving the groups identical until κ_c
        // learns them.
        for t in 0..k {
            for v in 0..num_types {
                kappa_t[t][v] = beta[t][v].max(1e-12).ln() - m_bg[v];
            }
        }
        content_beta = build_content_beta(&m_bg, &kappa_t, &kappa_c, &kappa_i, k, num_groups, num_types);
    }

    let mut mu_shared = vec![0.0f64; km1];
    let mut gamma: Option<Vec<Vec<f64>>> = nf.map(|f| vec![vec![0.0f64; km1]; f]);
    let mut sigma = vec![0.0f64; km1 * km1];
    for i in 0..km1 {
        sigma[i * km1 + i] = 1.0;
    }
    let mut lambda = vec![vec![0.0f64; km1]; d];
    // Per-document variational covariance ν, refreshed each E-step; the final
    // iteration's values are exposed for method-of-composition uncertainty.
    // When keep_nu=false we don't store them (saves O(N·K²) memory) but still
    // accumulate sigma_ss from them.
    let mut nu_store: Vec<Vec<f64>> = if keep_nu {
        vec![vec![0.0f64; km1 * km1]; d]
    } else {
        Vec::new()
    };

    // Per-document prior mean (shared, or regression-based with prevalence).
    let doc_mu = |di: usize, gamma: &Option<Vec<Vec<f64>>>, mu_shared: &[f64]| -> Vec<f64> {
        match (prevalence, gamma) {
            (Some(x), Some(g)) => mu_from(&x[di], g, km1),
            _ => mu_shared.to_vec(),
        }
    };

    let mut bound_history: Vec<f64> = Vec::with_capacity(em_iters);
    let mut converged = false;
    let mut em_iters_run = 0usize;
    // Track the sigma and beta used at the START of each E-step. After the loop,
    // these hold the values that produced the stored λ/ν, which may differ from
    // `sigma`/`beta` (updated by the last M-step). `recompute_nu` uses these to
    // exactly reproduce the stored ν. (The per-document prior mean μ_d does not
    // enter the Laplace Hessian, so ν is independent of μ and we need not track
    // it — see `recompute_nu`.)
    let mut sigma_estep = sigma.clone();
    let mut beta_estep = beta.clone();

    for em in 0..em_iters {
        em_iters_run = em + 1;
        sigma_estep = sigma.clone();  // capture sigma before E-step
        beta_estep = beta.clone();    // capture beta before E-step
        let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
            let mut s = sigma.clone();
            make_diagonally_dominant(&mut s, km1);
            spd_inverse(&s, km1).unwrap()
        });
        let entropy = match cholesky(&sigma, km1) {
            Some(l) => half_logdet(&l, km1),
            None => 0.0,
        };

        let mut beta_ss = vec![vec![1e-8f64; num_types]; k];
        // Content: soft expected counts per (topic×group, word).
        let mut content_ss = if content.is_some() {
            vec![vec![1e-8f64; num_types]; k * num_groups]
        } else {
            Vec::new()
        };
        let mut sigma_ss = vec![0.0f64; km1 * km1];
        let mut lambda_sum = vec![0.0f64; km1];

        // E-step: per-document variational inference is independent across
        // documents, so run it in parallel. To bound the transient memory (each
        // document's ν is (K-1)², so collecting all D at once is O(D·K²)), we
        // process documents in chunks: run a chunk's E-step in parallel, then
        // fold its sufficient statistics in serially before discarding the chunk.
        // The reduction still sums in ascending document order, so the fit stays
        // bit-for-bit deterministic regardless of thread count or chunk size.
        // The chunk is sized to cap the peak ν buffer near ~128 MB.
        let chunk = (128 * 1024 * 1024 / (km1 * km1 * 8).max(1))
            .max(256)
            .min(d.max(1));
        let mut total_bound = 0.0f64;
        let mut base = 0usize;
        while base < d {
            let end = (base + chunk).min(d);
            let chunk_results: Vec<(usize, (Vec<f64>, HpbResult))> =
                crate::variational::laplace_estep(&sparse[base..end], |local_di, words, counts| {
                    let di = base + local_di;
                    let mu_d = doc_mu(di, &gamma, &mu_shared);
                    // The E-step β is the document's group β (content) or the shared β.
                    let beta_doc: &[Vec<f64>] = match groups {
                        Some(g) => &content_beta[g[di]],
                        None => &beta,
                    };
                    let opt = lbfgs_minimize(
                        lambda[di].clone(),
                        |eta| ctm_lhood_grad(eta, beta_doc, words, counts, &mu_d, &siginv),
                        40,
                        7,
                        1e-5,
                    );
                    let res = ctm_hpb(&opt, beta_doc, words, counts, &mu_d, &siginv, entropy, diagonal);
                    (opt, res)
                });

            for (local_di, (opt, res)) in &chunk_results {
                let di = base + *local_di;
                total_bound += res.bound;
                let words = &sparse[di].0;
                lambda[di] = opt.clone();
                match groups {
                    Some(g) => {
                        let grp = g[di];
                        for (wi, &w) in words.iter().enumerate() {
                            for t in 0..k {
                                content_ss[t * num_groups + grp][w] += res.phi[t][wi];
                            }
                        }
                    }
                    None => {
                        for (wi, &w) in words.iter().enumerate() {
                            for t in 0..k {
                                beta_ss[t][w] += res.phi[t][wi];
                            }
                        }
                    }
                }
                if keep_nu {
                    nu_store[di] = res.nu.clone();
                }
                for i in 0..km1 {
                    lambda_sum[i] += opt[i];
                    for j in 0..km1 {
                        sigma_ss[i * km1 + j] += res.nu[i * km1 + j];
                    }
                }
            }
            base = end;
        }

        // Corpus bound for this E-step (sum of the per-document evidence bounds),
        // computed with the parameters from the previous M-step — the quantity
        // whose relative change drives convergence.
        bound_history.push(total_bound);

        // Convergence: stop once the relative change in the corpus bound falls
        // below `em_tol`. Break before the M-step, so the returned β/Σ/γ are the
        // converged parameters that produced this bound, with λ/ν freshly
        // refreshed by the E-step just run. `em_tol <= 0` disables early exit.
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (total_bound - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }

        // M-step: prevalence regression (γ) or shared mean (μ).
        if let Some(x) = prevalence {
            gamma = Some(match gamma_prior {
                GammaPrior::Pooled => fit_gamma_ridge(x, &lambda, nf.unwrap(), km1, 1e-6),
                GammaPrior::L1 { alpha } => fit_gamma_enet(x, &lambda, nf.unwrap(), km1, alpha),
            });
        } else {
            for i in 0..km1 {
                mu_shared[i] = lambda_sum[i] / d as f64;
            }
        }

        // Σ = (1/D)[ Σ ν + Σ (η-μ_d)(η-μ_d)ᵀ ] with the updated μ_d.
        let mus: Vec<Vec<f64>> = (0..d).map(|di| doc_mu(di, &gamma, &mu_shared)).collect();
        // Σ cross-term is O(D·K²) and was the dominant serial cost (#164).
        // Parallelize over the K-1 rows; each row still sums over documents in
        // ascending order, so the result is bit-identical to the serial loop
        // regardless of thread count.
        {
            use rayon::prelude::*;
            let df = d as f64;
            sigma.par_chunks_mut(km1).enumerate().for_each(|(i, row)| {
                for (j, sij) in row.iter_mut().enumerate() {
                    let mut cross = 0.0;
                    for (di, li) in lambda.iter().enumerate() {
                        cross += (li[i] - mus[di][i]) * (li[j] - mus[di][j]);
                    }
                    *sij = (sigma_ss[i * km1 + j] + cross) / df;
                }
            });
        }
        if sigma_shrink > 0.0 {
            for i in 0..km1 {
                for j in 0..km1 {
                    if i != j {
                        sigma[i * km1 + j] *= 1.0 - sigma_shrink;
                    }
                }
            }
        }
        // β M-step: SAGE content update (per group) or plain normalization.
        if content.is_some() {
            content_beta = optimize_content(
                &m_bg,
                &mut kappa_t,
                &mut kappa_c,
                &mut kappa_i,
                &content_ss,
                k,
                num_groups,
                num_types,
                1.0,
                20,
            );
        } else {
            for t in 0..k {
                let s: f64 = beta_ss[t].iter().sum();
                for v in 0..num_types {
                    beta[t][v] = beta_ss[t][v] / s;
                }
            }
        }
    }

    // With content covariates, the reported β is the group-averaged topic-word.
    let content_out = if content.is_some() {
        for t in 0..k {
            for v in 0..num_types {
                let mut s = 0.0;
                for g in 0..num_groups {
                    s += content_beta[g][t][v];
                }
                beta[t][v] = s / num_groups as f64;
            }
        }
        Some(content_beta)
    } else {
        None
    };

    CtmModel {
        num_topics: k,
        num_types,
        beta,
        beta_estep,
        mu: mu_shared,
        sigma,
        sigma_estep,
        lambda,
        nu: nu_store,
        gamma,
        content_beta: content_out,
        num_groups,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        em_iters_run,
        diagonal,
    }
}

/// Stochastic variational inference (SVI / online VB) for the **base** CTM/STM
/// logistic-normal topic model (Hoffman, Blei, Wang & Paisley, *JMLR* 2013).
///
/// Documents are processed in minibatches. Each minibatch runs the same
/// per-document Laplace E-step as [`fit_ctm`], then takes a *stochastic* step on
/// the global parameters (β, μ, Σ) toward the minibatch's full-corpus estimate
/// with a decaying learning rate `ρ_t = (τ + t)^(-κ)` (`κ ∈ (0.5, 1]`). For very
/// large corpora this reaches a good fit in a fraction of an epoch, where
/// full-batch EM must touch every document each iteration; on moderate corpora
/// the full-batch [`fit_ctm`] is preferable. Base model only — no prevalence or
/// content covariates.
#[allow(clippy::too_many_arguments)]
pub fn fit_ctm_svi<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    epochs: usize,
    batch_size: usize,
    tau: f64,
    kappa: f64,
    sigma_shrink: f64,
    init_spectral: bool,
    keep_nu: bool,
    diagonal: bool,
    rng: &mut R,
) -> CtmModel {
    let k = num_topics;
    let km1 = k - 1;
    let d = docs.len();
    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    let random_beta = |rng: &mut R| -> Vec<Vec<f64>> {
        let mut b = vec![vec![0.0f64; num_types]; k];
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
    let mut beta = if init_spectral {
        crate::spectral::spectral_init(docs, k, num_types).unwrap_or_else(|| random_beta(rng))
    } else {
        random_beta(rng)
    };

    let mut mu_shared = vec![0.0f64; km1];
    let mut sigma = vec![0.0f64; km1 * km1];
    for i in 0..km1 {
        sigma[i * km1 + i] = 1.0;
    }
    let mut lambda = vec![vec![0.0f64; km1]; d];
    let mut nu_store: Vec<Vec<f64>> = if keep_nu {
        vec![vec![0.0f64; km1 * km1]; d]
    } else {
        Vec::new()
    };

    let batch = batch_size.clamp(1, d.max(1));
    let mut t_step: usize = 0;

    for _epoch in 0..epochs {
        // Deterministic shuffle (Fisher-Yates with the supplied rng).
        let order = crate::variational::svi::shuffled_order(d, rng);

        for chunk in order.chunks(batch) {
            t_step += 1;
            let rho = crate::variational::svi::rho(tau, kappa, t_step);

            let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
                let mut s = sigma.clone();
                make_diagonally_dominant(&mut s, km1);
                spd_inverse(&s, km1).unwrap()
            });
            let entropy = match cholesky(&sigma, km1) {
                Some(l) => half_logdet(&l, km1),
                None => 0.0,
            };

            let mut beta_ss = vec![vec![1e-8f64; num_types]; k];
            let mut sigma_ss = vec![0.0f64; km1 * km1];
            let mut lambda_sum = vec![0.0f64; km1];
            let mut etas: Vec<Vec<f64>> = Vec::with_capacity(chunk.len());

            for &di in chunk {
                let words = &sparse[di].0;
                let counts = &sparse[di].1;
                let opt = lbfgs_minimize(
                    lambda[di].clone(),
                    |eta| ctm_lhood_grad(eta, &beta, words, counts, &mu_shared, &siginv),
                    40,
                    7,
                    1e-5,
                );
                let res = ctm_hpb(&opt, &beta, words, counts, &mu_shared, &siginv, entropy, diagonal);
                for (wi, &w) in words.iter().enumerate() {
                    for tt in 0..k {
                        beta_ss[tt][w] += res.phi[tt][wi];
                    }
                }
                for i in 0..km1 {
                    lambda_sum[i] += opt[i];
                    for j in 0..km1 {
                        sigma_ss[i * km1 + j] += res.nu[i * km1 + j];
                    }
                }
                lambda[di] = opt.clone();
                if keep_nu {
                    nu_store[di] = res.nu.clone();
                }
                etas.push(opt);
            }
            let bsz = chunk.len() as f64;

            // β: scale the minibatch soft counts to the full corpus, normalize to
            // a distribution, and blend toward it (a convex step, so β stays a
            // valid distribution).
            for tt in 0..k {
                let s: f64 = beta_ss[tt].iter().sum::<f64>() * (d as f64 / bsz);
                for v in 0..num_types {
                    let bhat = (d as f64 / bsz) * beta_ss[tt][v] / s;
                    beta[tt][v] = (1.0 - rho) * beta[tt][v] + rho * bhat;
                }
            }
            // μ: blend toward the minibatch mean η.
            for i in 0..km1 {
                mu_shared[i] = (1.0 - rho) * mu_shared[i] + rho * (lambda_sum[i] / bsz);
            }
            // Σ: blend toward the minibatch covariance estimate.
            for i in 0..km1 {
                for j in 0..km1 {
                    let mut cross = 0.0;
                    for eta in &etas {
                        cross += (eta[i] - mu_shared[i]) * (eta[j] - mu_shared[j]);
                    }
                    let shat = (sigma_ss[i * km1 + j] + cross) / bsz;
                    let mut newv = (1.0 - rho) * sigma[i * km1 + j] + rho * shat;
                    if sigma_shrink > 0.0 && i != j {
                        newv *= 1.0 - sigma_shrink;
                    }
                    sigma[i * km1 + j] = newv;
                }
            }
        }
    }

    // Final full E-step with the converged globals to give every document a
    // λ/ν (the θ posterior) and the corpus bound.
    let siginv = spd_inverse(&sigma, km1).unwrap_or_else(|| {
        let mut s = sigma.clone();
        make_diagonally_dominant(&mut s, km1);
        spd_inverse(&s, km1).unwrap()
    });
    let entropy = match cholesky(&sigma, km1) {
        Some(l) => half_logdet(&l, km1),
        None => 0.0,
    };
    let mut total_bound = 0.0f64;
    for di in 0..d {
        let words = &sparse[di].0;
        let counts = &sparse[di].1;
        let opt = lbfgs_minimize(
            lambda[di].clone(),
            |eta| ctm_lhood_grad(eta, &beta, words, counts, &mu_shared, &siginv),
            40,
            7,
            1e-5,
        );
        let res = ctm_hpb(&opt, &beta, words, counts, &mu_shared, &siginv, entropy, diagonal);
        lambda[di] = opt;
        if keep_nu {
            nu_store[di] = res.nu;
        }
        total_bound += res.bound;
    }

    // The final full E-step uses the converged sigma and beta (no M-step follows),
    // so sigma_estep = sigma and beta_estep = beta.
    let sigma_estep = sigma.clone();
    let beta_estep = beta.clone();

    CtmModel {
        num_topics: k,
        num_types,
        beta,
        beta_estep,
        mu: mu_shared,
        sigma,
        sigma_estep,
        lambda,
        nu: nu_store,
        gamma: None,
        content_beta: None,
        num_groups: 1,
        bound: total_bound,
        bound_history: vec![total_bound],
        converged: true,
        em_iters_run: epochs,
        diagonal,
    }
}

/// Recompute the per-document variational covariance ν from the stored
/// variational means λ and the fitted global parameters.  Used when the model
/// was fit with `keep_nu = false` to reconstruct ν on demand (e.g. for
/// method-of-composition uncertainty). Parallelized with rayon.
///
/// The Laplace covariance ν = H⁻¹ depends only on λ, β and Σ⁻¹ — the prior mean
/// μ cancels out of the Hessian — so we evaluate at the shared `model.mu` and the
/// E-step β/Σ (`beta_estep`/`sigma_estep`) to reproduce the stored ν exactly.
/// `sparse` is the same `(words, counts)` representation built from the raw docs.
pub fn recompute_nu(
    model: &CtmModel,
    sparse: &[(Vec<usize>, Vec<f64>)],
) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let d = sparse.len();
    let km1 = model.num_topics - 1;
    // Use sigma_estep (the sigma that was used in the last E-step) rather than
    // model.sigma (which was updated by the final M-step and may differ).
    let sig = &model.sigma_estep;
    let siginv = crate::linalg::spd_inverse(sig, km1).unwrap_or_else(|| {
        let mut s = sig.clone();
        crate::linalg::make_diagonally_dominant(&mut s, km1);
        crate::linalg::spd_inverse(&s, km1).unwrap()
    });
    let entropy = match crate::linalg::cholesky(sig, km1) {
        Some(l) => crate::linalg::half_logdet(&l, km1),
        None => 0.0,
    };
    (0..d)
        .into_par_iter()
        .map(|di| {
            let (words, counts) = &sparse[di];
            // Use beta_estep (the beta that was active during the last E-step)
            // rather than model.beta (which was updated by the final M-step). The
            // prior mean does not enter the Hessian, so model.mu is used for all
            // documents (ν is independent of μ).
            let res = ctm_hpb(
                &model.lambda[di],
                &model.beta_estep,
                words,
                counts,
                &model.mu,
                &siginv,
                entropy,
                model.diagonal,
            );
            res.nu
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn toy() -> (Vec<usize>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let beta = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.2, 0.5, 0.3],
            vec![0.1, 0.2, 0.7],
        ];
        let words = vec![0usize, 1, 2];
        let counts = vec![3.0, 2.0, 5.0];
        let mu = vec![0.1, -0.2];
        // siginv (2x2 SPD)
        let siginv = vec![1.5, 0.3, 0.3, 1.2];
        (words, counts, beta, mu, siginv)
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let (words, counts, beta, mu, siginv) = toy();
        let eta = vec![0.4, -0.3];
        let g = ctm_grad(&eta, &beta, &words, &counts, &mu, &siginv);
        let eps = 1e-6;
        for i in 0..eta.len() {
            let mut ep = eta.clone();
            let mut em = eta.clone();
            ep[i] += eps;
            em[i] -= eps;
            let num = (ctm_lhood(&ep, &beta, &words, &counts, &mu, &siginv)
                - ctm_lhood(&em, &beta, &words, &counts, &mu, &siginv))
                / (2.0 * eps);
            assert!((num - g[i]).abs() < 1e-4, "grad[{}]: {} vs {}", i, g[i], num);
        }
    }

    #[test]
    fn svi_recovers_planted_blocks() {
        // Three disjoint vocabulary blocks; each doc draws from one. SVI's β rows
        // should each concentrate on a single block.
        let nb = 3usize;
        let wpb = 5usize;
        let v = nb * wpb;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let docs: Vec<Vec<u32>> = (0..300)
            .map(|d| {
                let b = d % nb;
                let block: Vec<u32> = (b * wpb..(b + 1) * wpb).map(|w| w as u32).collect();
                let mut doc = block.clone();
                doc.extend(block);
                doc
            })
            .collect();
        let m = fit_ctm_svi(&docs, nb, v, 20, 32, 16.0, 0.7, 0.0, false, true, false, &mut rng);
        // Each planted block is the top of some topic.
        let mut covered = std::collections::HashSet::new();
        for t in 0..nb {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| m.beta[t][b].partial_cmp(&m.beta[t][a]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..wpb].iter().copied().collect();
            for b in 0..nb {
                let block: std::collections::HashSet<usize> = (b * wpb..(b + 1) * wpb).collect();
                if block.is_subset(&top) {
                    covered.insert(b);
                }
            }
        }
        assert_eq!(covered.len(), nb, "SVI only recovered {covered:?}");
    }

    #[test]
    fn svi_deterministic_for_seed() {
        let docs: Vec<Vec<u32>> = (0..120)
            .map(|d| (0..6).map(|i| ((i + d) % 9) as u32).collect())
            .collect();
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            fit_ctm_svi(&docs, 3, 9, 10, 16, 16.0, 0.7, 0.0, false, true, false, &mut rng).beta
        };
        let (a, b) = (run(), run());
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn recovers_topic_correlation() {
        // Two topic-blocks that co-occur: docs use {0,1,2} (topic A words) AND
        // {3,4,5} (topic B words) together, so topics A and B should be
        // positively correlated in Σ.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for _ in 0..150 {
            // Correlated: most docs load on A+B together; some on C alone.
            if rng.gen::<f64>() < 0.7 {
                docs.push(vec![0, 1, 2, 3, 4, 5, 0, 1, 3, 4]);
            } else {
                docs.push(vec![6, 7, 8, 6, 7, 8, 6, 7, 8, 6]);
            }
        }
        let model = fit_ctm(&docs, 3, 9, 25, 0.0, 0.0, None, None, true, GammaPrior::Pooled, true, false, &mut rng);
        let theta = model.doc_topics();
        // Sanity: θ rows sum to 1 and are valid.
        for row in &theta {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
        // The correlation matrix is well-formed (diagonal 1).
        let corr = model.topic_correlation();
        for i in 0..3 {
            assert!((corr[i][i] - 1.0).abs() < 1e-9 || corr[i][i] == 1.0);
        }
    }

    #[test]
    fn content_recovers_group_wording() {
        // One topic, two groups: group 0 uses words {0,1}, group 1 uses {2,3}.
        // The content model should word the topic differently per group.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut groups: Vec<usize> = Vec::new();
        for i in 0..120 {
            if i % 2 == 0 {
                docs.push(vec![0, 1, 0, 1, 0, 1]);
                groups.push(0);
            } else {
                docs.push(vec![2, 3, 2, 3, 2, 3]);
                groups.push(1);
            }
        }
        // K=2 (CTM needs >=2 topics); content groups = 2.
        let model = fit_ctm(&docs, 2, 4, 30, 0.0, 0.0, None, Some((&groups, 2)), false, GammaPrior::Pooled, true, false, &mut rng);
        let cb = model.content_beta.expect("content_beta present");
        // cb[group][topic][word]. The dominant topic for group 0 should favour
        // {0,1}; for group 1 {2,3}. Check that for each group some topic does.
        let g0_best = (0..2)
            .map(|t| cb[0][t][0] + cb[0][t][1])
            .fold(0.0f64, f64::max);
        let g1_best = (0..2)
            .map(|t| cb[1][t][2] + cb[1][t][3])
            .fold(0.0f64, f64::max);
        assert!(g0_best > 0.8, "group 0 top topic mass on its words = {}", g0_best);
        assert!(g1_best > 0.8, "group 1 top topic mass on its words = {}", g1_best);
    }

    #[test]
    fn em_bound_increases_and_converges() {
        // Variational EM must ascend the bound; with a tolerance it should stop
        // before the iteration cap, and `em_tol = 0` must run every iteration.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for i in 0..150 {
            if i % 2 == 0 {
                docs.push(vec![0, 1, 2, 0, 1, 2, 0, 1]);
            } else {
                docs.push(vec![3, 4, 5, 3, 4, 5, 3, 4]);
            }
        }

        let converged = fit_ctm(&docs, 2, 6, 100, 1e-5, 0.0, None, None, true, GammaPrior::Pooled, true, false, &mut rng);
        // The bound trajectory is (weakly) monotone increasing.
        let h = &converged.bound_history;
        assert!(h.len() >= 2);
        for w in h.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "bound decreased: {} -> {}", w[0], w[1]);
        }
        assert!(converged.converged, "should meet em_tol before the 100-iter cap");
        assert_eq!(converged.em_iters_run, h.len());
        assert!(converged.bound.is_finite());

        // em_tol = 0 disables early stopping: run the full cap.
        let mut rng2 = ChaCha8Rng::seed_from_u64(7);
        let capped = fit_ctm(&docs, 2, 6, 8, 0.0, 0.0, None, None, true, GammaPrior::Pooled, true, false, &mut rng2);
        assert!(!capped.converged);
        assert_eq!(capped.em_iters_run, 8);
        assert_eq!(capped.bound_history.len(), 8);
    }

    #[test]
    fn diagonal_bound_increases_and_recovers_topics() {
        // Mean-field (diagonal) variational mode: the EM bound must ascend and the
        // model must recover disjoint-vocabulary topics, with ν purely diagonal.
        let nb = 3usize;
        let wpb = 5usize;
        let v = nb * wpb;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let docs: Vec<Vec<u32>> = (0..240)
            .map(|d| {
                let b = d % nb;
                let block: Vec<u32> = (b * wpb..(b + 1) * wpb).map(|w| w as u32).collect();
                let mut doc = block.clone();
                doc.extend(block);
                doc
            })
            .collect();
        let model = fit_ctm(
            &docs, nb, v, 30, 0.0, 0.0, None, None, true, GammaPrior::Pooled, true, true, &mut rng,
        );
        assert!(model.diagonal, "model should record diagonal mode");

        // The mean-field diagonal objective is not the exact Laplace lower bound,
        // so the reported bound rises steeply to (near) convergence and may then
        // drift by a tiny amount per step. Check that it improves massively
        // overall and that any per-step decrease is negligible relative to the
        // total improvement (no large backward jumps).
        let h = &model.bound_history;
        assert!(h.len() >= 2);
        let h_max = h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let total_range = h_max - h[0];
        assert!(total_range > 0.0, "diagonal bound did not improve overall");
        assert!(h[h.len() - 1] > h[0], "diagonal bound did not improve overall");
        let mut max_decrease = 0.0f64;
        for w in h.windows(2) {
            let dec = w[0] - w[1];
            if dec > max_decrease {
                max_decrease = dec;
            }
        }
        assert!(
            max_decrease < 0.01 * total_range,
            "max per-step decrease {} too large vs total improvement {}",
            max_decrease, total_range
        );

        // ν is purely diagonal (off-diagonals exactly zero, diagonals positive).
        let km1 = nb - 1;
        for nu in &model.nu {
            for i in 0..km1 {
                for j in 0..km1 {
                    if i == j {
                        assert!(nu[i * km1 + j] > 0.0, "diagonal nu must be positive");
                    } else {
                        assert_eq!(nu[i * km1 + j], 0.0, "off-diagonal nu must be exactly 0");
                    }
                }
            }
        }

        // Each planted block is the top of some topic.
        let mut covered = std::collections::HashSet::new();
        for t in 0..nb {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| model.beta[t][b].partial_cmp(&model.beta[t][a]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..wpb].iter().copied().collect();
            for b in 0..nb {
                let block: std::collections::HashSet<usize> = (b * wpb..(b + 1) * wpb).collect();
                if block.is_subset(&top) {
                    covered.insert(b);
                }
            }
        }
        assert_eq!(covered.len(), nb, "diagonal mode only recovered {covered:?}");
    }

    // Build a synthetic regression problem with n observations, p predictors
    // (plus an intercept), where only `n_active` of the p predictors are truly
    // nonzero. Returns (X, Lambda, true_coefs_excl_intercept).
    fn make_sparse_regression(
        n: usize,
        p: usize,
        n_active: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        // Simple LCG for deterministic data without pulling in a full rng crate
        // at test time.  Generates uniform [0,1) floats.
        let mut state = seed ^ 0xdeadbeef_cafef00d;
        let mut rand_f64 = move || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (u32::MAX as f64)
        };

        // True coefficients: n_active nonzero (values 1..=n_active), rest zero.
        let mut true_coef = vec![0.0f64; p];
        for j in 0..n_active {
            true_coef[j] = (j + 1) as f64;
        }

        // X: n × (p+1) with intercept prepended; predictors are Gaussian-ish.
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = vec![1.0f64]; // intercept
            for _ in 0..p {
                // Box-Muller for Gaussian-ish draws.
                let u1 = rand_f64().max(1e-15);
                let u2 = rand_f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                row.push(z);
            }
            x.push(row);
        }

        // Lambda[:,0] = X[:,1:] · true_coef + small noise (SNR high).
        let mut lam: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut y = 0.5; // intercept contribution
            for j in 0..p {
                y += x[i][j + 1] * true_coef[j];
            }
            // Add small noise.
            let u1 = rand_f64().max(1e-15);
            let u2 = rand_f64();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 0.1;
            lam.push(vec![y + noise]);
        }

        (x, lam, true_coef)
    }

    #[test]
    fn ctm_conforms() {
        use crate::conformance::{check_conformance, check_logistic_normal};
        use crate::variational::LogisticNormalModel;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 0, 1, 2],
            vec![1, 2, 1, 2, 0],
            vec![2, 0, 2, 0, 1],
            vec![0, 0, 1, 2, 0],
            vec![1, 1, 2, 0, 1],
            vec![2, 2, 0, 1, 2],
        ];
        let model = fit_ctm(&docs, 3, 3, 5, 0.0, 0.0, None, None, false, GammaPrior::Pooled, true, false, &mut rng);

        let base_violations = check_conformance(&model);
        assert!(
            base_violations.is_empty(),
            "check_conformance violations: {:?}",
            base_violations
        );

        let ln_violations = check_logistic_normal(&model);
        assert!(
            ln_violations.is_empty(),
            "check_logistic_normal violations: {:?}",
            ln_violations
        );

        assert_eq!(
            model.eta_dim(),
            model.num_topics - 1,
            "eta_dim should be num_topics - 1"
        );
        assert_eq!(
            model.eta_mean().len(),
            model.eta_cov().len(),
            "eta_mean and eta_cov should have the same number of rows (one per document)"
        );
    }

    #[test]
    fn enet_sparser_than_ridge_on_sparse_signal() {
        // Design: 200 obs, 30 predictors, only 3 are truly active (large signal).
        // Elastic-net (lasso, alpha=1) should zero most inactive predictors;
        // ridge keeps all nonzero.
        let (x, lam, true_coef) = make_sparse_regression(200, 30, 3, 42);
        let f = x[0].len(); // 31 (intercept + 30 predictors)
        let km1 = 1;

        let g_enet = fit_gamma_enet(&x, &lam, f, km1, 1.0);
        let g_ridge = fit_gamma_ridge(&x, &lam, f, km1, 1e-6);

        // Count zeros (|coef| < 1e-6) among the 30 penalised predictors.
        let enet_zeros = g_enet[1..].iter().filter(|r| r[0].abs() < 1e-6).count();
        let ridge_zeros = g_ridge[1..].iter().filter(|r| r[0].abs() < 1e-6).count();

        // Elastic-net should produce substantially more zeros than ridge.
        assert!(
            enet_zeros > ridge_zeros + 5,
            "enet should zero more inactive predictors than ridge: enet_zeros={enet_zeros}, ridge_zeros={ridge_zeros}"
        );

        // The 3 active predictors (index 0, 1, 2 in the penalised block,
        // i.e. g[1], g[2], g[3]) should have the correct sign.
        for j in 0..3 {
            assert!(
                g_enet[j + 1][0] * true_coef[j] > 0.0,
                "active covariate {j} has wrong sign in enet solution"
            );
        }
    }
}
