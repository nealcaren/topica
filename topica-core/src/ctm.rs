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

use crate::estimator::{Estimator, ModelFamily};
use crate::linalg::{
    cholesky, cholesky_jitter, half_logdet, make_diagonally_dominant, spd_inverse_from_chol,
};
use crate::variational::LogisticNormalModel;
use crate::variational::{doc_sparse, fit_gamma_vb, lbfgs_minimize};

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
    pub nu: Vec<f64>,       // (K-1)×(K-1) variational covariance H⁻¹
    pub phi: Vec<Vec<f64>>, // K×W expected token-topic counts for the doc's words
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
                let l = cholesky_jitter(&h, km1);
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

/// The SAGE content-model κ decomposition (the additive parts of the per-group
/// topic-word model): `log β_{k,a,v} = m_v + κ_topic_{k,v} + κ_cov_{a,v} +
/// κ_interaction_{k,a,v}`, softmax-normalized over `v`. `content_beta` is derived
/// from these; the κ pieces are the identifying information R `stm`'s
/// `sageLabels()` / `labelTopics()` rank words by (and cannot be recovered from
/// the per-group β alone).
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContentKappa {
    /// Background log word-frequency `m`, length V.
    pub m: Vec<f64>,
    /// Topic deviations κ_topic, K × V.
    pub kappa_topic: Vec<Vec<f64>>,
    /// Covariate (group) deviations κ_cov, num_groups × V.
    pub kappa_cov: Vec<Vec<f64>>,
    /// Topic×group interaction deviations κ_interaction, (K·num_groups) × V,
    /// indexed `topic * num_groups + group` (the layout `build_content_beta` uses).
    pub kappa_interaction: Vec<Vec<f64>>,
}

/// A fitted CTM/STM model.
pub struct CtmModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub beta: Vec<Vec<f64>>,   // K×V topic-word
    pub mu: Vec<f64>,          // K-1 prior mean (no-covariate case)
    pub sigma: Vec<f64>,       // (K-1)² prior covariance
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
    /// SAGE κ decomposition behind `content_beta`, `Some` when content covariates
    /// were supplied. See [`ContentKappa`].
    pub content_kappa: Option<ContentKappa>,
    pub num_groups: usize,
    /// Per-document group index (length D), `Some` alongside `content_beta` when
    /// content covariates were supplied. `recompute_nu` needs it to rebuild each
    /// document's ν with its own group's topic-word distribution rather than the
    /// group-averaged `beta`.
    pub groups: Option<Vec<usize>>,
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
    /// The initialization route the fit actually took (issue #410): one of
    /// `"spectral"` (anchor-word init succeeded), `"random-fallback"` (spectral
    /// was requested but recovery returned `None`, so a seeded random init ran),
    /// `"random"` (`init="random"`), or `"provided"` (a caller-supplied `init_beta`).
    /// Lets the config-aware determinism report distinguish a genuinely bit-exact
    /// spectral fit from a seeded random fallback.
    pub initialization: String,
    /// Per-group topic-word distributions (G × K × V) that were active during the
    /// final E-step — the content analogue of `beta_estep`. `Some` only for a
    /// fresh content fit; `recompute_nu` uses it to reproduce the stored ν exactly
    /// (the final M-step's `optimize_content` updated `content_beta` afterwards, so
    /// `content_beta` differs). Like `beta_estep`/`sigma_estep`, this is NOT
    /// persisted: a loaded content model has `None` here and `recompute_nu` falls
    /// back to `content_beta` (the same post-M-step approximation `beta_estep`
    /// degrades to on load). Appended last for bincode-positional compatibility.
    pub content_beta_estep: Option<Vec<Vec<Vec<f64>>>>,
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

/// Soft-thresholding operator `sign(z)·max(|z|-g, 0)` — the proximal map of the
/// L1 penalty and the source of exact zeros under the sparse content prior.
fn soft_threshold(z: f64, g: f64) -> f64 {
    if z > g {
        z - g
    } else if z < -g {
        z + g
    } else {
        0.0
    }
}

/// Minimize `f(x) + lam·Σ_{i:penalize} |x_i − anchor_i|` by FISTA with
/// backtracking line search (Beck & Teboulle 2009). `f_and_grad` supplies the
/// smooth part's value and gradient (already in minimization form). The L1 term
/// is non-smooth at its anchor, so plain L-BFGS cannot produce exact zeros; the
/// proximal step (soft-thresholding around `anchor`) does. `penalize[i]` gates
/// which coordinates carry the L1 term — for the sparse content prior only the
/// group and topic×group deviation blocks (κ_cov, κ_interaction) are sparsified;
/// the topic baseline κ_topic keeps its L2 so topic vocabularies stay coherent.
fn fista_l1<F>(
    x0: Vec<f64>,
    f_and_grad: F,
    lam: f64,
    anchor: &[f64],
    penalize: &[bool],
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let n = x0.len();
    let prox_step = |y: &[f64], g: &[f64], t: f64| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let grad_step = y[i] - t * g[i];
                if penalize.get(i).copied().unwrap_or(false) {
                    let a = anchor.get(i).copied().unwrap_or(0.0);
                    a + soft_threshold(grad_step - a, lam * t)
                } else {
                    grad_step
                }
            })
            .collect::<Vec<f64>>()
    };
    let mut x = x0.clone();
    let mut y = x0;
    let mut t_step = 1.0f64;
    let mut theta = 1.0f64;
    for _ in 0..max_iter {
        let (fy, gy) = f_and_grad(&y);
        let mut step = t_step;
        let x_new = loop {
            let cand = prox_step(&y, &gy, step);
            let (fc, _) = f_and_grad(&cand);
            let mut dot = 0.0;
            let mut sq = 0.0;
            for i in 0..n {
                let d = cand[i] - y[i];
                dot += gy[i] * d;
                sq += d * d;
            }
            if fc <= fy + dot + sq / (2.0 * step) + 1e-12 || step < 1e-12 {
                break cand;
            }
            step *= 0.5;
        };
        t_step = step;
        let theta_next = (1.0 + (1.0 + 4.0 * theta * theta).sqrt()) / 2.0;
        let beta = (theta - 1.0) / theta_next;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let d = x_new[i] - x[i];
            num += d * d;
            den += x[i] * x[i];
            y[i] = x_new[i] + beta * d;
        }
        x = x_new;
        theta = theta_next;
        if num.sqrt() / (den.sqrt() + 1e-12) < 1e-4 {
            break;
        }
    }
    x
}

/// MAP-update the SAGE content deviations κ from soft (topic×group×word)
/// expected counts, then rebuild per-group β. `counts[k*G+g][v]` are the
/// variational expected token counts; `prior_variance` is the Gaussian (L2)
/// prior on κ. `content_l1 > 0` swaps the L-BFGS L2 solve for a FISTA solve with
/// a sparse (Laplace/L1) prior of rate `content_l1` on the group and topic×group
/// deviation blocks (κ_cov, κ_interaction), keeping the topic baseline κ_topic on
/// L2; `content_l1 = 0` is the original L-BFGS solve, bit-exact.
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
    content_l1: f64,
    rw_time: Option<(usize, usize, f64)>,
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

    // Smooth part of the objective (minimization form): negative log-likelihood
    // + Gaussian L2 prior + the ordered-time random walk. Shared by the L2 solve
    // (L-BFGS) and the L1 solve (FISTA), which adds a sparse κ prior via its prox.
    let smooth = |flat: &[f64]| -> (f64, Vec<f64>) {
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
        // Gaussian (L2) prior. Under the sparse content prior (content_l1 > 0) the
        // deviation blocks κ_cov / κ_interaction (indices >= n_t) are regularized by
        // the L1 prox in FISTA instead, so they carry NO L2 here — only the baseline
        // κ_topic keeps L2. This makes content_prior="l1" a genuine Laplace prior on
        // the deviations rather than a coupled L1+L2 elastic net (#532).
        let l2_end = if content_l1 > 0.0 { n_t } else { flat.len() };
        for (i, &xi) in flat.iter().enumerate().take(l2_end) {
            value -= 0.5 * inv_var * xi * xi;
            grad[i] -= inv_var * xi;
        }

        // First-order random-walk penalty smoothing the content deviations of
        // an ordered (time) covariate axis. With the caller's convention that
        // the saturated group index decomposes as `grp = base*num_times + time`,
        // each cell is tied to its time-predecessor within the same base group,
        // on both the group main effect κ_cov and the topic×group interaction
        // κ_i: `(rw/2) Σ (x_{·,t} - x_{·,t-1})²`. `rw = 1/τ²` is the RW
        // precision. `None` (no ordered axis) leaves the solve bit-exact.
        if let Some((num_base, num_times, rw)) = rw_time {
            if num_times >= 2 && rw > 0.0 {
                for base in 0..num_base {
                    for time in 1..num_times {
                        let g2 = base * num_times + time;
                        let g1 = base * num_times + (time - 1);
                        for w in 0..v {
                            let a = n_t + g2 * v + w;
                            let b = n_t + g1 * v + w;
                            let diff = flat[a] - flat[b];
                            value -= 0.5 * rw * diff * diff;
                            grad[a] -= rw * diff;
                            grad[b] += rw * diff;
                        }
                        for topic in 0..k {
                            let c2 = topic * g + g2;
                            let c1 = topic * g + g1;
                            for w in 0..v {
                                let a = n_t + n_c + c2 * v + w;
                                let b = n_t + n_c + c1 * v + w;
                                let diff = flat[a] - flat[b];
                                value -= 0.5 * rw * diff * diff;
                                grad[a] -= rw * diff;
                                grad[b] += rw * diff;
                            }
                        }
                    }
                }
            }
        }

        (-value, grad.iter().map(|gv| -gv).collect())
    };

    let x = if content_l1 > 0.0 {
        // Sparse (L1) content prior: sparsify the group and topic×group deviation
        // blocks (κ_cov, κ_interaction), keeping the topic baseline κ_topic on L2.
        let total = n_t + n_c + k * g * v;
        let mut penalize = vec![false; total];
        for p in penalize.iter_mut().take(total).skip(n_t) {
            *p = true;
        }
        let anchor = vec![0.0f64; total];
        // FISTA is warm-started from the current κ each EM iteration, so a modest
        // inner budget suffices; it also stops early on its own relative-change
        // criterion. A large cap here made the content M-step dominate runtime.
        fista_l1(x0, smooth, content_l1, &anchor, &penalize, max_iter.max(40))
    } else {
        lbfgs_minimize(x0, smooth, max_iter, 7, 1e-4)
    };

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
    if words.is_empty() {
        // With no observed tokens the variational objective reduces to the prior
        // term, whose mode is η = μ, so θ = softmax([μ, 0]) (the reference topic
        // K-1 has η = 0). Returning the uniform vector would ignore the document's
        // prior mean — wrong for prevalence models where μ = Xγ varies per document.
        let mut logits = mu.to_vec();
        logits.push(0.0);
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut e: Vec<f64> = logits.iter().map(|&x| (x - m).exp()).collect();
        let s: f64 = e.iter().sum();
        for v in &mut e {
            *v /= s;
        }
        return e;
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
        let var: f64 = x
            .iter()
            .map(|row| {
                let d = row[j + 1] - col_mean[j];
                d * d
            })
            .sum::<f64>()
            / n_f64;
        col_std[j] = if var > 1e-12 { var.sqrt() } else { 1.0 };
    }

    // Build standardised design matrix (n × p), excluding the intercept column.
    let xs: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            (0..p)
                .map(|j| (row[j + 1] - col_mean[j]) / col_std[j])
                .collect()
        })
        .collect();

    // Pre-compute column norms² of the standardised design (all = n for unit-variance).
    let mut xj_norm2 = vec![0.0f64; p];
    for j in 0..p {
        xj_norm2[j] = xs.iter().map(|row| row[j] * row[j]).sum();
        if xj_norm2[j] < 1e-12 {
            xj_norm2[j] = 1.0;
        } // constant column guard
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
                    for ri in r.iter_mut() {
                        *ri -= delta_int;
                    }
                    max_change = max_change.max(delta_int.abs());
                }

                // Update each penalised predictor.
                for j in 0..p {
                    let old = coef[j + 1];
                    // Partial residual: add back contribution of current coef.
                    let rj_dot: f64 = xs
                        .iter()
                        .zip(r.iter())
                        .map(|(row, &ri)| row[j] * (ri + old * row[j]))
                        .sum();
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

                if max_change < 1e-7 {
                    break;
                }
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
    content_time_rw: Option<(usize, usize, f64)>,
    content_prior_var: f64,
    content_l1: f64,
    init_spectral: bool,
    init_beta: Option<&[Vec<f64>]>,
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
    // Content-prior scale guards (shared by every front-end: topica's PyO3 layer
    // and faSTM's extendr layer both reach the core here, so guarding at the source
    // protects both). `content_prior_var` feeds `1/content_prior_var` — the L2
    // precision, or the L1 rate under a sparse content prior — so a non-finite,
    // zero, or negative value would silently produce NaN kappa and a NaN bound.
    if content.is_some() {
        assert!(
            content_prior_var.is_finite() && content_prior_var > 0.0,
            "content_prior_var must be finite and > 0; got {content_prior_var}"
        );
    }
    if let Some((nb, nt, smooth)) = content_time_rw {
        assert!(
            content.is_some(),
            "content_time_rw requires a content covariate"
        );
        assert_eq!(
            nb * nt,
            num_groups,
            "content_time_rw: num_base*num_times must equal the saturated num_groups"
        );
        // The random-walk precision (1/tau^2). Non-finite/negative would poison the
        // RW penalty; 0.0 is legal (recovers the fully saturated content factor).
        assert!(
            smooth.is_finite() && smooth >= 0.0,
            "content_smooth must be finite and >= 0; got {smooth}"
        );
    }

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
    // A caller-supplied `init_beta` (K×V) overrides spectral/random init — the
    // warm-start hook that lets an STM-compatible front end inject an externally
    // computed base β (e.g. R `stm`'s exact spectral β) and reproduce that fit.
    let (mut beta, init_route): (Vec<Vec<f64>>, &'static str) = match init_beta {
        Some(b) => (b.iter().map(|row| row.to_vec()).collect(), "provided"),
        None if init_spectral => match crate::spectral::spectral_init(docs, k, num_types) {
            Some(b) => (b, "spectral"),
            None => (random_beta(rng), "random-fallback"),
        },
        None => (random_beta(rng), "random"),
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
        content_beta = build_content_beta(
            &m_bg, &kappa_t, &kappa_c, &kappa_i, k, num_groups, num_types,
        );
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
    // Content analogue of `beta_estep`: the per-group β the E-step ran against,
    // captured before the final M-step's `optimize_content` overwrites it. Only
    // needed to recompute ν on demand, so we take the G×K×V snapshot solely for a
    // content fit that did NOT keep ν (keep_nu=false); otherwise it stays empty and
    // the copy is skipped every EM iteration.
    let capture_content_beta = content.is_some() && !keep_nu;
    let mut content_beta_estep = if capture_content_beta {
        content_beta.clone()
    } else {
        Vec::new()
    };

    for em in 0..em_iters {
        em_iters_run = em + 1;
        sigma_estep = sigma.clone(); // capture sigma before E-step
        beta_estep = beta.clone(); // capture beta before E-step
        if capture_content_beta {
            content_beta_estep = content_beta.clone(); // capture group β before E-step
        }
        // Inverse and log-det from a single factor so the bound's quadratic
        // and entropy terms stay consistent even when Σ needs a PD repair.
        let (siginv, entropy) = crate::linalg::spd_inverse_and_half_logdet(&sigma, km1);

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
                    let res = ctm_hpb(
                        &opt, beta_doc, words, counts, &mu_d, &siginv, entropy, diagonal,
                    );
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
                GammaPrior::Pooled => fit_gamma_vb(x, &lambda, nf.unwrap(), km1, 1000),
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
                content_prior_var,
                content_l1,
                content_time_rw,
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
    let content_kappa_out = if content.is_some() {
        Some(ContentKappa {
            m: m_bg.clone(),
            kappa_topic: kappa_t.clone(),
            kappa_cov: kappa_c.clone(),
            kappa_interaction: kappa_i.clone(),
        })
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
        content_kappa: content_kappa_out,
        num_groups,
        groups: groups.map(|g| g.to_vec()),
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        em_iters_run,
        diagonal,
        initialization: init_route.to_string(),
        // Only captured for a keep_nu=false content fit (see `capture_content_beta`).
        content_beta_estep: if capture_content_beta {
            Some(content_beta_estep)
        } else {
            None
        },
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
    convergence_tol: f64,
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
    let (mut beta, init_route): (Vec<Vec<f64>>, &'static str) = if init_spectral {
        match crate::spectral::spectral_init(docs, k, num_types) {
            Some(b) => (b, "spectral"),
            None => (random_beta(rng), "random-fallback"),
        }
    } else {
        (random_beta(rng), "random")
    };

    let mut mu_shared = vec![0.0f64; km1];
    let mut sigma = vec![0.0f64; km1 * km1];
    // `m2` is the persistent second-moment sufficient statistic E[ν + η ηᵀ]; Σ is
    // derived from it each M-step as M2 − μ μᵀ. With μ = 0 and Σ = I at init, M2 = I.
    let mut m2 = vec![0.0f64; km1 * km1];
    for i in 0..km1 {
        sigma[i * km1 + i] = 1.0;
        m2[i * km1 + i] = 1.0;
    }
    let mut lambda = vec![vec![0.0f64; km1]; d];
    let mut nu_store: Vec<Vec<f64>> = if keep_nu {
        vec![vec![0.0f64; km1 * km1]; d]
    } else {
        Vec::new()
    };

    let batch = batch_size.clamp(1, d.max(1));

    // The persistent SVI global state for the topic-word distribution is the
    // *unnormalized* count statistic `lambda_beta` (the Dirichlet natural
    // parameter), not the normalized `beta`. Blending the normalized per-minibatch
    // ratio (the old code) cancels the D/B corpus scaling and loses the count
    // magnitude, so a topic dominated by a high-count word drifts toward uniform
    // instead of keeping its mass (#421). `beta` is `lambda_beta` normalized and is
    // what the E-step reads. Seed `lambda_beta` on the corpus count scale from the
    // starting `beta` so its normalization is the same initial `beta` and the first
    // minibatch blend is count-scale-consistent.
    let total_tokens: f64 = sparse
        .iter()
        .map(|(_, c)| c.iter().sum::<f64>())
        .sum::<f64>()
        .max(1.0);
    let beta_mass = total_tokens / k as f64;
    let mut lambda_beta: Vec<Vec<f64>> = beta
        .iter()
        .map(|row| row.iter().map(|&b| 1e-8 + b * beta_mass).collect())
        .collect();

    let mut t_step: usize = 0;

    // Per-epoch running-ELBO trace plus the early-stop bookkeeping. Every doc is
    // visited exactly once per epoch (the shuffled order covers all D), and we sum
    // its per-minibatch bound. Each bound is scored against the globals *as they
    // stood at that minibatch*, which move within the epoch — so this is a streaming
    // training score, not a fixed-parameter corpus ELBO. It is the standard, cheap
    // streaming-VB monitoring signal; evaluating a fixed-global corpus bound each
    // epoch would cost a full extra E-step and defeat SVI's purpose. `convergence_tol
    // > 0` early-stops on the relative epoch-to-epoch change in this score (a
    // heuristic); `converged` reports whether it did.
    let mut bound_history: Vec<f64> = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = epochs;

    for epoch in 0..epochs {
        // Deterministic shuffle (Fisher-Yates with the supplied rng).
        let order = crate::variational::svi::shuffled_order(d, rng);
        let mut epoch_bound = 0.0;

        for chunk in order.chunks(batch) {
            t_step += 1;
            let rho = crate::variational::svi::rho(tau, kappa, t_step);

            // Inverse and log-det from a single factor so the bound's quadratic and
            // entropy terms stay consistent even when Σ needs a PD repair.
            let (siginv, entropy) = crate::linalg::spd_inverse_and_half_logdet(&sigma, km1);

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
                let res = ctm_hpb(
                    &opt, &beta, words, counts, &mu_shared, &siginv, entropy, diagonal,
                );
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
                epoch_bound += res.bound;
                etas.push(opt);
            }
            let bsz = chunk.len() as f64;

            // β: online-VB update of the topic-word count statistic. Blend the
            // D/B-scaled minibatch soft counts (η + (D/B)·Σ_batch φ, the
            // natural-gradient target for the Dirichlet natural parameter) into the
            // persistent `lambda_beta`, then renormalize to the distribution `beta`.
            // Blending the counts — not the normalized ratio, whose D/B cancels —
            // preserves the corpus-scale magnitude (#421). All terms are positive,
            // so `lambda_beta` stays positive and `beta` a valid simplex.
            for tt in 0..k {
                let mut s = 0.0;
                for v in 0..num_types {
                    let lam_hat = 1e-8 + (d as f64 / bsz) * (beta_ss[tt][v] - 1e-8);
                    lambda_beta[tt][v] = (1.0 - rho) * lambda_beta[tt][v] + rho * lam_hat;
                    s += lambda_beta[tt][v];
                }
                for v in 0..num_types {
                    beta[tt][v] = lambda_beta[tt][v] / s;
                }
            }
            // (μ, Σ): faithful SVI on the Gaussian's expected sufficient statistics
            // (Hoffman et al. 2013). The globals are the first and second moments
            //   s1 = E[η]              (= μ)
            //   s2 = E[ν + η ηᵀ]       (= M2, the persistent `m2`)
            // Each is a Robbins-Monro blend toward its minibatch estimate, then
            //   Σ = M2 − μ μᵀ.
            // Deriving Σ from the raw second moment — rather than centering a
            // per-minibatch cross-product on some choice of μ — is what makes this
            // correct: the batch M-step can center on the just-updated μ only because
            // there μ IS the exact mean of the same λ set, but the SVI μ is a blended
            // value that is neither the minibatch mean nor a consistent center, so
            // centering on it folds the μ step into Σ (#421). Accumulating ηηᵀ instead
            // has no centering choice and stays well-defined at batch_size = 1, where
            // any batch-mean-centered covariance would collapse to ν alone.
            let mut m2_hat = vec![0.0f64; km1 * km1];
            for eta in &etas {
                for i in 0..km1 {
                    for j in 0..km1 {
                        m2_hat[i * km1 + j] += eta[i] * eta[j];
                    }
                }
            }
            for i in 0..km1 {
                mu_shared[i] = (1.0 - rho) * mu_shared[i] + rho * (lambda_sum[i] / bsz);
            }
            for i in 0..km1 {
                for j in 0..km1 {
                    let s2_hat = (sigma_ss[i * km1 + j] + m2_hat[i * km1 + j]) / bsz;
                    m2[i * km1 + j] = (1.0 - rho) * m2[i * km1 + j] + rho * s2_hat;
                    let mut sij = m2[i * km1 + j] - mu_shared[i] * mu_shared[j];
                    // Off-diagonal shrinkage toward a diagonal Σ, applied once to the
                    // freshly derived covariance (not fed back into the unshrunk `m2`),
                    // so it is a fixed shrinkage prior — it does not compound step over
                    // step into the collapse the per-minibatch multiply once caused.
                    if sigma_shrink > 0.0 && i != j {
                        sij *= 1.0 - sigma_shrink;
                    }
                    sigma[i * km1 + j] = sij;
                }
            }
        }

        // Epoch-to-epoch convergence on the relative change in the running ELBO.
        // Disabled when `convergence_tol <= 0` (run the full epoch budget).
        if let Some(&prev) = bound_history.last() {
            let rel = (epoch_bound - prev).abs() / prev.abs().max(1e-10);
            if convergence_tol > 0.0 && rel < convergence_tol {
                bound_history.push(epoch_bound);
                converged = true;
                epochs_run = epoch + 1;
                break;
            }
        }
        bound_history.push(epoch_bound);
    }

    // Final full E-step with the converged globals to give every document a
    // λ/ν (the θ posterior) and the corpus bound.
    // Inverse and log-det from a single factor so the bound's quadratic and
    // entropy terms stay consistent even when Σ needs a PD repair.
    let (siginv, entropy) = crate::linalg::spd_inverse_and_half_logdet(&sigma, km1);
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
        let res = ctm_hpb(
            &opt, &beta, words, counts, &mu_shared, &siginv, entropy, diagonal,
        );
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
        content_kappa: None,
        num_groups: 1,
        groups: None,
        bound: total_bound,
        bound_history,
        converged,
        em_iters_run: epochs_run,
        diagonal,
        initialization: init_route.to_string(),
        content_beta_estep: None,
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
pub fn recompute_nu(model: &CtmModel, sparse: &[(Vec<usize>, Vec<f64>)]) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let d = sparse.len();
    let km1 = model.num_topics - 1;
    // Use sigma_estep (the sigma that was used in the last E-step) rather than
    // model.sigma (which was updated by the final M-step and may differ).
    let sig = &model.sigma_estep;
    // Inverse and log-det from a single factor (consistent even after a PD
    // repair); ν itself does not depend on the log-det here.
    let (siginv, entropy) = crate::linalg::spd_inverse_and_half_logdet(sig, km1);
    (0..d)
        .into_par_iter()
        .map(|di| {
            let (words, counts) = &sparse[di];
            // For a content model the E-step ran each document against its own
            // group's topic-word distribution, so ν must be rebuilt the same way:
            // pick that group's β, not the group-averaged `beta_estep`. A fresh fit
            // carries `content_beta_estep` (the group β active during the final
            // E-step) and reproduces the stored ν exactly. A loaded content model
            // does not persist it (like `beta_estep`), so we fall back to the
            // post-M-step `content_beta` — the same one-M-step approximation the
            // shared path already accepts when `beta_estep` degrades to `beta` on
            // load. The prior mean does not enter the Hessian, so model.mu is used
            // for all documents.
            let beta_doc: &[Vec<f64>] = match (
                &model.content_beta_estep,
                &model.content_beta,
                &model.groups,
            ) {
                (Some(cbeta_estep), _, Some(groups)) => &cbeta_estep[groups[di]],
                (None, Some(cbeta), Some(groups)) => &cbeta[groups[di]],
                _ => &model.beta_estep,
            };
            let res = ctm_hpb(
                &model.lambda[di],
                beta_doc,
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

    /// Golden parity guard: a frozen `fit_ctm` output on a fixed fixture, so the
    /// extraction into `topica-core` is provably a numerical no-op and `topica`
    /// (which re-exports this `fit_ctm`) stays in lock-step. Two fits from the same
    /// seed must be bit-identical, and the topic-word matrix must match the frozen
    /// reference. If this fails, the CTM numerics changed — intended or not.
    #[test]
    fn fit_ctm_golden() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 0, 1, 2, 2],
            vec![1, 2, 1, 2, 0, 0],
            vec![2, 0, 2, 0, 1, 1],
            vec![3, 4, 3, 4, 5, 5],
            vec![4, 5, 4, 5, 3, 3],
            vec![5, 3, 5, 3, 4, 4],
        ];
        let fit = || {
            let mut rng = ChaCha8Rng::seed_from_u64(123);
            fit_ctm(
                &docs,
                2,
                6,
                30,
                0.0,
                0.0,
                None,
                None,
                None,
                1.0,
                0.0,
                false,
                None,
                GammaPrior::Pooled,
                true,
                false,
                &mut rng,
            )
        };
        let a = fit();
        let b = fit();
        // Reproducibility: same seed -> bit-identical beta.
        assert_eq!(a.beta, b.beta, "fit_ctm is not seed-reproducible");
        // Frozen reference: each topic concentrates on one disjoint 3-word block.
        let on = 0.333_333_332_8_f64;
        let off = 0.000_000_000_6_f64;
        let expected = [[off, off, off, on, on, on], [on, on, on, off, off, off]];
        for k in 0..2 {
            for v in 0..6 {
                assert!(
                    (a.beta[k][v] - expected[k][v]).abs() < 1e-9,
                    "beta[{k}][{v}] = {} drifted from frozen {}",
                    a.beta[k][v],
                    expected[k][v]
                );
            }
        }
    }

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
            assert!(
                (num - g[i]).abs() < 1e-4,
                "grad[{}]: {} vs {}",
                i,
                g[i],
                num
            );
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
        let m = fit_ctm_svi(
            &docs, nb, v, 20, 32, 16.0, 0.7, 0.0, 0.0, false, true, false, &mut rng,
        );
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
            fit_ctm_svi(
                &docs, 3, 9, 10, 16, 16.0, 0.7, 0.0, 0.0, false, true, false, &mut rng,
            )
            .beta
        };
        let (a, b) = (run(), run());
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    /// #421: SVI Σ shrinkage must scale with the Robbins-Monro step, not be applied
    /// on every minibatch. Multiplying the blended value by (1-shrink) each step
    /// collapsed the off-diagonals toward 0 (a spurious diagonal Σ) regardless of ρ.
    #[test]
    fn svi_shrinkage_scales_with_step_not_per_minibatch() {
        // Topics A and B co-occur, so Σ has a large positive off-diagonal.
        let corr_docs = || {
            let mut rng = ChaCha8Rng::seed_from_u64(1);
            let mut docs: Vec<Vec<u32>> = Vec::new();
            for _ in 0..300 {
                if rng.gen::<f64>() < 0.7 {
                    docs.push(vec![0, 1, 2, 3, 4, 5, 0, 1, 3, 4]);
                } else {
                    docs.push(vec![6, 7, 8, 6, 7, 8, 6, 7, 8, 6]);
                }
            }
            docs
        };
        let max_offdiag = |m: &CtmModel| -> f64 {
            let km1 = m.num_topics - 1;
            let mut w = 0.0f64;
            for i in 0..km1 {
                for j in 0..km1 {
                    if i != j {
                        w = w.max(m.sigma[i * km1 + j].abs());
                    }
                }
            }
            w
        };
        let fit = |shrink: f64| {
            let docs = corr_docs();
            let mut rng = ChaCha8Rng::seed_from_u64(2);
            fit_ctm_svi(
                &docs, 3, 9, 40, 16, 64.0, 0.7, shrink, 0.0, false, false, false, &mut rng,
            )
        };
        let off0 = max_offdiag(&fit(0.0));
        assert!(
            off0 > 1.0,
            "correlated topics should give a large off-diagonal, got {off0}"
        );
        let off3 = max_offdiag(&fit(0.3));
        // Shrinkage should reduce the off-diagonal, but must NOT collapse it to ~0.
        assert!(
            off3 < off0,
            "shrinkage should reduce the off-diagonal ({off3} vs {off0})"
        );
        assert!(
            off3 > 0.05,
            "shrinkage collapsed the off-diagonal to ~0 (the per-minibatch bug): {off3}"
        );
    }

    /// #421: the SVI β update must blend the D/B-scaled minibatch *counts*, not the
    /// normalized per-minibatch ratio (whose D/B cancels), so β reflects the
    /// aggregate count magnitude. Vocab {0, 1}; with `batch_size = 1` half the
    /// documents are word-0 dominated (99:1) and half are balanced (1:1), an
    /// aggregate ratio of ~50:1. The fixed update accumulates counts and puts >0.9
    /// mass on word 0; the old normalized-ratio blend averages 0.99 and 0.5 toward
    /// ~0.74 and fails.
    #[test]
    fn svi_beta_preserves_count_magnitude() {
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for i in 0..80 {
            if i % 2 == 0 {
                let mut doc = vec![0u32; 99];
                doc.push(1);
                docs.push(doc);
            } else {
                docs.push(vec![0u32, 1]);
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let model = fit_ctm_svi(
            &docs, 2, 2, 40, 1, 64.0, 0.7, 0.0, 0.0, false, false, false, &mut rng,
        );
        let max_w0 = (0..model.num_topics)
            .map(|t| model.beta[t][0])
            .fold(0.0f64, f64::max);
        assert!(
            max_w0 > 0.9,
            "β should track the ~50:1 aggregate count ratio (max word-0 mass \
             {max_w0}); the old normalized-ratio blend collapses toward ~0.74"
        );
    }

    /// #421: `fit_ctm_svi` must honor `convergence_tol` — early-stop on the relative
    /// epoch-to-epoch ELBO change and report `converged` honestly. The prior code
    /// hardcoded `converged: true` with a length-1 `bound_history`, so the behavioral
    /// test passed trivially without any convergence test ever running.
    #[test]
    fn svi_honors_convergence_tol() {
        let docs: Vec<Vec<u32>> = (0..120)
            .map(|d| (0..6).map(|i| ((i + d) % 9) as u32).collect())
            .collect();
        let epochs = 30usize;

        // tol = 0 → run the full epoch budget; `converged` must be false and the
        // ELBO trace has one entry per epoch.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let full = fit_ctm_svi(
            &docs, 3, 9, epochs, 16, 16.0, 0.7, 0.0, 0.0, false, false, false, &mut rng,
        );
        assert!(!full.converged, "tol = 0 must not report convergence");
        assert_eq!(full.em_iters_run, epochs);
        assert_eq!(full.bound_history.len(), epochs);

        // A loose tol early-stops well before the budget and reports `converged`.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let early = fit_ctm_svi(
            &docs, 3, 9, epochs, 16, 16.0, 0.7, 0.0, 1e-2, false, false, false, &mut rng,
        );
        assert!(early.converged, "a loose tol should early-stop");
        assert!(
            early.em_iters_run < epochs,
            "SVI should stop before the {epochs}-epoch budget, ran {}",
            early.em_iters_run
        );
        assert_eq!(early.bound_history.len(), early.em_iters_run);
    }

    /// #421: the SVI Σ update must derive the covariance from the raw second-moment
    /// statistic M2 (Σ = M2 − μμᵀ), not from a per-minibatch cross-product centered
    /// on the blended μ. The old centering folded the μ step into Σ, which is worst
    /// at `batch_size = 1` (large per-doc μ swings) — and a batch-mean-centered
    /// covariance would collapse to ν there. Here strongly correlated topics must
    /// still yield a finite, PD Σ with a real positive off-diagonal at B = 1.
    #[test]
    fn svi_covariance_well_defined_at_batch_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for _ in 0..200 {
            if rng.gen::<f64>() < 0.7 {
                docs.push(vec![0, 1, 2, 3, 4, 5, 0, 1, 3, 4]);
            } else {
                docs.push(vec![6, 7, 8, 6, 7, 8, 6, 7, 8, 6]);
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let m = fit_ctm_svi(
            &docs, 3, 9, 40, 1, 64.0, 0.7, 0.0, 0.0, false, false, false, &mut rng,
        );
        let km1 = m.num_topics - 1;
        for i in 0..km1 {
            let d = m.sigma[i * km1 + i];
            assert!(
                d.is_finite() && d > 0.0,
                "Σ diagonal must be finite and positive at batch_size = 1, got {d}"
            );
        }
        let max_off = (0..km1)
            .flat_map(|i| (0..km1).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| m.sigma[i * km1 + j].abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_off > 0.05,
            "correlated topics must leave a real off-diagonal, not collapse to ν \
             (a batch-mean-centered covariance would); got {max_off}"
        );
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
        let model = fit_ctm(
            &docs,
            3,
            9,
            25,
            0.0,
            0.0,
            None,
            None,
            None,
            1.0,
            0.0,
            true,
            None,
            GammaPrior::Pooled,
            true,
            false,
            &mut rng,
        );
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
        let model = fit_ctm(
            &docs,
            2,
            4,
            30,
            0.0,
            0.0,
            None,
            Some((&groups, 2)),
            None,
            1.0,
            0.0,
            false,
            None,
            GammaPrior::Pooled,
            true,
            false,
            &mut rng,
        );
        let cb = model.content_beta.expect("content_beta present");
        // cb[group][topic][word]. The dominant topic for group 0 should favour
        // {0,1}; for group 1 {2,3}. Check that for each group some topic does.
        let g0_best = (0..2)
            .map(|t| cb[0][t][0] + cb[0][t][1])
            .fold(0.0f64, f64::max);
        let g1_best = (0..2)
            .map(|t| cb[1][t][2] + cb[1][t][3])
            .fold(0.0f64, f64::max);
        assert!(
            g0_best > 0.8,
            "group 0 top topic mass on its words = {}",
            g0_best
        );
        assert!(
            g1_best > 0.8,
            "group 1 top topic mass on its words = {}",
            g1_best
        );
    }

    #[test]
    fn content_time_rw_smooths_adjacent_periods() {
        // Ordered-time content: 2 base groups x 4 time periods = 8 saturated
        // groups, index `base*T + time`. Each cell has only a few documents (thin
        // cells), and word usage drifts across time. The RW penalty should pull
        // adjacent-time content_beta cells together vs the unsmoothed fit, and the
        // unsmoothed (None) fit must match the plain saturated-content solve.
        let (nb, nt, v) = (2usize, 4usize, 6usize);
        let g = nb * nt;
        let build = || {
            let mut rng = ChaCha8Rng::seed_from_u64(3);
            let mut docs: Vec<Vec<u32>> = Vec::new();
            let mut groups: Vec<usize> = Vec::new();
            for base in 0..nb {
                for time in 0..nt {
                    let grp = base * nt + time;
                    // A marker word whose weight ramps smoothly with time; base 1
                    // ramps the opposite marker. Thin: 3 docs per cell.
                    for _ in 0..3 {
                        let mut doc = vec![0u32, 1u32];
                        let hi = if base == 0 { 2u32 } else { 4u32 };
                        for _ in 0..(1 + time) {
                            doc.push(hi);
                        }
                        // a little noise so cells genuinely differ
                        doc.push(rng.gen_range(0..v as u32));
                        docs.push(doc);
                        groups.push(grp);
                    }
                }
            }
            (docs, groups)
        };
        let (docs, groups) = build();
        let fit = |rw: Option<(usize, usize, f64)>| {
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            fit_ctm(
                &docs,
                2,
                v,
                40,
                0.0,
                0.0,
                None,
                Some((&groups, g)),
                rw,
                1.0,
                0.0,
                false,
                None,
                GammaPrior::Pooled,
                true,
                false,
                &mut rng,
            )
            .content_beta
            .expect("content_beta present")
        };
        // Mean squared adjacent-time gap of content_beta within each base group.
        let adj_gap = |cb: &Vec<Vec<Vec<f64>>>| -> f64 {
            let mut acc = 0.0;
            let mut n = 0.0;
            for base in 0..nb {
                for time in 1..nt {
                    let g2 = base * nt + time;
                    let g1 = base * nt + (time - 1);
                    for topic in 0..2 {
                        for w in 0..v {
                            let d = cb[g2][topic][w] - cb[g1][topic][w];
                            acc += d * d;
                            n += 1.0;
                        }
                    }
                }
            }
            acc / n
        };
        let plain = fit(None);
        let smoothed = fit(Some((nb, nt, 50.0)));
        // Determinism: the None path is identical across calls (no RW side effect).
        let plain2 = fit(None);
        assert_eq!(plain, plain2, "None path must be deterministic/bit-exact");
        let gap_plain = adj_gap(&plain);
        let gap_smooth = adj_gap(&smoothed);
        assert!(
            gap_smooth < 0.5 * gap_plain,
            "RW smoothing should shrink adjacent-time gaps: plain={gap_plain:.3e} smoothed={gap_smooth:.3e}"
        );
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

        let converged = fit_ctm(
            &docs,
            2,
            6,
            100,
            1e-5,
            0.0,
            None,
            None,
            None,
            1.0,
            0.0,
            true,
            None,
            GammaPrior::Pooled,
            true,
            false,
            &mut rng,
        );
        // The bound trajectory is (weakly) monotone increasing.
        let h = &converged.bound_history;
        assert!(h.len() >= 2);
        for w in h.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "bound decreased: {} -> {}", w[0], w[1]);
        }
        assert!(
            converged.converged,
            "should meet em_tol before the 100-iter cap"
        );
        assert_eq!(converged.em_iters_run, h.len());
        assert!(converged.bound.is_finite());

        // em_tol = 0 disables early stopping: run the full cap.
        let mut rng2 = ChaCha8Rng::seed_from_u64(7);
        let capped = fit_ctm(
            &docs,
            2,
            6,
            8,
            0.0,
            0.0,
            None,
            None,
            None,
            1.0,
            0.0,
            true,
            None,
            GammaPrior::Pooled,
            true,
            false,
            &mut rng2,
        );
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
            &docs,
            nb,
            v,
            30,
            0.0,
            0.0,
            None,
            None,
            None,
            1.0,
            0.0,
            true,
            None,
            GammaPrior::Pooled,
            true,
            true,
            &mut rng,
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
        assert!(
            h[h.len() - 1] > h[0],
            "diagonal bound did not improve overall"
        );
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
            max_decrease,
            total_range
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
        assert_eq!(
            covered.len(),
            nb,
            "diagonal mode only recovered {covered:?}"
        );
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
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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

    // (The `ctm_conforms` test lives in `topica`'s conformance.rs — it depends on
    // `crate::conformance`, which stays in the `topica` crate, not `topica-core`.)

    #[test]
    fn enet_sparser_than_ridge_on_sparse_signal() {
        // Design: 200 obs, 30 predictors, only 3 are truly active (large signal).
        // Elastic-net (lasso, alpha=1) should zero most inactive predictors;
        // ridge keeps all nonzero.
        let (x, lam, true_coef) = make_sparse_regression(200, 30, 3, 42);
        let f = x[0].len(); // 31 (intercept + 30 predictors)
        let km1 = 1;

        let g_enet = fit_gamma_enet(&x, &lam, f, km1, 1.0);
        let g_ridge = crate::variational::fit_gamma_ridge(&x, &lam, f, km1, 1e-6);

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

    /// An empty (or all-OOV) document has no likelihood term, so its inferred θ
    /// must be the prior mode `softmax([μ, 0])`, not a uniform vector. This
    /// matters for prevalence models where μ = Xγ varies per document.
    #[test]
    fn infer_theta_empty_doc_returns_prior_mode() {
        // K = 4 topics -> μ has length K-1 = 3. Reference topic (index 3) has η = 0.
        // All four logits (μ plus the appended reference 0) are DISTINCT, so a bug
        // that misplaced the reference topic would change the expected vector and
        // fail this test — [.., 0.0] with a zero in μ would not catch that swap.
        let mu = vec![1.0, -0.5, 0.7];
        let k = mu.len() + 1;
        // beta/siginv are unused on the empty-doc path, but must be shaped validly.
        let beta = vec![vec![0.25; 3]; k]; // K×V, arbitrary
        let siginv = vec![0.0; mu.len() * mu.len()];

        let theta = infer_theta(&beta, &mu, &siginv, &[], &[]);

        // Expected: softmax([1.0, -0.5, 0.7, 0.0]).
        let logits = [1.0, -0.5, 0.7, 0.0];
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        let expected: Vec<f64> = exps.iter().map(|&x| x / s).collect();

        assert_eq!(theta.len(), k);
        let sum: f64 = theta.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "θ must sum to 1, got {sum}");
        for (t, (&got, &exp)) in theta.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "topic {t}: θ = {got}, expected prior-mode {exp}"
            );
        }
        // Regression: it must NOT be the old uniform fallback.
        let uniform = 1.0 / k as f64;
        assert!(
            (theta[0] - uniform).abs() > 1e-6,
            "θ collapsed to the uniform vector — prior mean μ was ignored"
        );
    }

    /// A large prevalence predictor must not overflow the empty-doc softmax: the
    /// max-shift keeps every exponent <= 0, so θ stays finite and normalized and
    /// concentrates on the dominant topic.
    #[test]
    fn infer_theta_empty_doc_large_mu_is_finite() {
        let mu = vec![800.0, -800.0, 0.0]; // exp(800) would overflow without the shift
        let k = mu.len() + 1;
        let beta = vec![vec![0.25; 3]; k];
        let siginv = vec![0.0; mu.len() * mu.len()];

        let theta = infer_theta(&beta, &mu, &siginv, &[], &[]);

        assert!(
            theta.iter().all(|x| x.is_finite()),
            "θ has non-finite entries: {theta:?}"
        );
        let sum: f64 = theta.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "θ must sum to 1, got {sum}");
        assert!(
            theta[0] > 0.999,
            "θ should concentrate on the dominant topic, got {theta:?}"
        );
    }

    /// `recompute_nu` must rebuild each document's ν against its own group's β for
    /// a content model — using the group-averaged β (the pre-fix behavior) gives a
    /// materially different ν when groups word the topics differently.
    #[test]
    fn recompute_nu_uses_per_group_beta_for_content() {
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
        // Ground truth: keep_nu=true stores the per-group E-step ν. (The snapshot is
        // now captured only when keep_nu=false, so a single fit cannot both store ν
        // and carry the snapshot — use two same-seed fits, which are bit-identical.)
        let m_keep = fit_ctm(
            &docs,
            2,
            4,
            30,
            0.0,
            0.0,
            None,
            Some((&groups, 2)),
            None,
            1.0,
            0.0,
            false,
            None,
            GammaPrior::Pooled,
            true,
            false,
            &mut rng,
        );
        let stored = m_keep.nu.clone(); // per-group E-step ν (keep_nu = true)
        assert_eq!(stored.len(), docs.len(), "nu stored per doc");

        // Recompute source: keep_nu=false captures content_beta_estep.
        let mut rng2 = ChaCha8Rng::seed_from_u64(1);
        let mut model = fit_ctm(
            &docs,
            2,
            4,
            30,
            0.0,
            0.0,
            None,
            Some((&groups, 2)),
            None,
            1.0,
            0.0,
            false,
            None,
            GammaPrior::Pooled,
            false,
            false,
            &mut rng2,
        );
        assert!(model.content_beta.is_some(), "content_beta present");
        assert!(
            model.content_beta_estep.is_some(),
            "content_beta_estep captured for keep_nu=false"
        );
        assert_eq!(
            model.groups.as_ref().map(|g| g.len()),
            Some(120),
            "groups stored"
        );

        let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|d| doc_sparse(d)).collect();

        let mean_l1 = |a: &[Vec<f64>], b: &[Vec<f64>]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum::<f64>())
                .sum::<f64>()
                / a.len() as f64
        };

        // Per-group reconstruction reproduces the stored ν essentially exactly:
        // `content_beta_estep` is the group β the E-step actually ran against, so
        // ctm_hpb at the stored λ returns the same ν (to machine precision).
        let recomputed = recompute_nu(&model, &sparse);
        let err_fixed = mean_l1(&stored, &recomputed);

        // The pre-fix path used the group-averaged β for every document.
        model.groups = None;
        let shared_nu = recompute_nu(&model, &sparse);
        let err_shared = mean_l1(&stored, &shared_nu);

        assert!(
            err_fixed < 1e-9,
            "per-group recompute should reproduce stored ν exactly: {err_fixed}"
        );
        assert!(
            err_shared > 20.0 * err_fixed.max(1e-12),
            "group-averaged recompute ({err_shared}) should be far worse than per-group ({err_fixed})"
        );
    }

    /// The E-step-snapshot guard: with a single EM iteration the final content-β
    /// M-step moves `content_beta` well away from the β the (only) E-step ran
    /// against, so reconstructing ν from the post-M-step `content_beta` (the state
    /// before `content_beta_estep` was tracked) is materially wrong. `recompute_nu`
    /// must use the E-step snapshot and reproduce the stored ν exactly.
    #[test]
    fn recompute_nu_uses_estep_content_beta_not_final_mstep() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
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
        // em_iters = 1: exactly one E-step (against the initial content β) then one
        // M-step that overwrites content_beta. The snapshot is captured only when
        // keep_nu=false, so use two same-seed (bit-identical) fits: keep_nu=true for
        // the stored E-step ν, keep_nu=false for the content_beta_estep snapshot.
        let m_keep = fit_ctm(
            &docs,
            2,
            4,
            1,
            0.0,
            0.0,
            None,
            Some((&groups, 2)),
            None,
            1.0,
            0.0,
            false,
            None,
            GammaPrior::Pooled,
            true,
            false,
            &mut rng,
        );
        let stored = m_keep.nu.clone();
        let mut rng2 = ChaCha8Rng::seed_from_u64(7);
        let mut model = fit_ctm(
            &docs,
            2,
            4,
            1,
            0.0,
            0.0,
            None,
            Some((&groups, 2)),
            None,
            1.0,
            0.0,
            false,
            None,
            GammaPrior::Pooled,
            false,
            false,
            &mut rng2,
        );
        assert!(
            model.content_beta_estep.is_some(),
            "content_beta_estep captured for a keep_nu=false content fit"
        );
        // The final M-step must actually have moved content_beta, or the test would
        // pass trivially even against the buggy post-M-step path.
        let moved = {
            let est = model.content_beta_estep.as_ref().unwrap();
            let fin = model.content_beta.as_ref().unwrap();
            est.iter()
                .zip(fin)
                .map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum::<f64>())
                        .sum::<f64>()
                })
                .sum::<f64>()
        };
        assert!(
            moved > 1e-3,
            "final M-step should have moved content_beta (got {moved}); test would be vacuous"
        );

        let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|d| doc_sparse(d)).collect();
        let recomputed = recompute_nu(&model, &sparse);
        let err: f64 = stored
            .iter()
            .zip(&recomputed)
            .map(|(a, b)| a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f64>())
            .sum::<f64>()
            / stored.len() as f64;
        assert!(
            err < 1e-9,
            "recompute_nu must use the E-step content β snapshot (err {err})"
        );

        // Sanity: recomputing against the post-M-step content_beta (the pre-fix
        // behavior) is materially wrong — clearing the snapshot forces that path.
        model.content_beta_estep = None;
        let post_mstep = recompute_nu(&model, &sparse);
        let err_post: f64 = stored
            .iter()
            .zip(&post_mstep)
            .map(|(a, b)| a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f64>())
            .sum::<f64>()
            / stored.len() as f64;
        assert!(
            err_post > 100.0 * err.max(1e-12),
            "post-M-step content_beta should be far worse ({err_post}) than the snapshot ({err})"
        );
    }
}
