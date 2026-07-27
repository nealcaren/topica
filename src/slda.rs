//! Supervised LDA (Blei & McAuliffe 2007) — LDA with a per-document **response
//! variable** `y_d` regressed on the document's topic usage. Each document's
//! response is Gaussian, `y_d ~ N(ηᵀ z̄_d, σ²)`, where `z̄_d` is the empirical
//! topic frequency of its words. Fitting the topics is thus *supervised* by the
//! response: topics are shaped to be predictive of `y`, and the fitted
//! regression coefficients `η` say how each topic moves the response.
//!
//! Two inference backends are provided:
//!   - [`fit_slda`] — the original variational EM of Blei & McAuliffe (2007):
//!     an **E-step** (per document coordinate ascent on the variational `γ`
//!     Dirichlet and `φ` per-word-topic parameters, whose `φ` update carries the
//!     response-coupling term that ties the words together through `η`/`σ²`) and
//!     an **M-step** (`β` from the expected word-topic counts as in LDA; `η` by
//!     the normal equations `η = (Σ_d E[z̄_d z̄_dᵀ])⁻¹ Σ_d y_d E[z̄_d]`; `σ²`
//!     from the residual).
//!   - [`fit_slda_gibbs`] — the collapsed Gibbs sampler used by \pkg{tomotopy},
//!     sampling each token's topic with the same Gaussian response term applied
//!     to a hard assignment, then re-estimating `η` by ridge regression on the
//!     sampled topic frequencies each sweep (`σ²` is a fixed hyperparameter, as
//!     in tomotopy).
//!
//! Prediction for a new document infers `φ`/`γ` with the response term removed
//! (ordinary LDA inference against the fixed `β`) and returns `ŷ = ηᵀ z̄`.

use crate::linalg::spd_inverse;
use crate::optimize::digamma;
use rand::Rng;
use rayon::prelude::*;

/// A fitted supervised-LDA model.
pub struct SldaModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub alpha: f64,
    pub log_beta: Vec<Vec<f64>>, // K × V
    pub eta: Vec<f64>,           // K regression coefficients
    pub sigma2: f64,             // response variance
    pub gamma: Vec<Vec<f64>>,    // D × K variational Dirichlet (training docs)
    /// `M = Σ_d E[z̄_d z̄_dᵀ]` (K×K, row-major, ridge-stabilized) — the
    /// normal-equations matrix the η estimate solves, retained for coefficient SEs.
    pub m_mat: Vec<f64>,
}

impl SldaModel {
    /// Topic-word distributions β = exp(log_beta), shape K×V.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        self.log_beta
            .iter()
            .map(|row| row.iter().map(|&l| l.exp()).collect())
            .collect()
    }

    /// Document-topic mixtures θ_d = γ_d / Σγ_d, shape D×K.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.gamma
            .iter()
            .map(|g| {
                let s: f64 = g.iter().sum();
                g.iter().map(|&x| x / s).collect()
            })
            .collect()
    }
}

/// Bag-of-words for one document: (word_id, count) pairs.
fn to_bag(doc: &[u32]) -> Vec<(usize, f64)> {
    let mut counts: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &w in doc {
        *counts.entry(w as usize).or_insert(0.0) += 1.0;
    }
    counts.into_iter().collect()
}

/// E-step for one document. Coordinate ascent on (γ, φ). When `y` is `Some`, the
/// φ update includes the Blei-McAuliffe response-coupling term; when `None`
/// (prediction) it reduces to ordinary LDA inference. Returns (gamma, phi,
/// sum_phi) where `phi[i]` corresponds to `bag[i]`'s word and `sum_phi = Σ_n φ_n`
/// over all tokens (with repeats).
fn infer_doc(
    bag: &[(usize, f64)],
    log_beta: &[Vec<f64>],
    eta: &[f64],
    sigma2: f64,
    alpha: f64,
    y: Option<f64>,
    var_iters: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let k = log_beta.len();
    let nwords = bag.len();
    let n_tokens: f64 = bag.iter().map(|&(_, c)| c).sum();

    let mut phi = vec![vec![1.0 / k as f64; k]; nwords];
    let mut gamma = vec![alpha + n_tokens / k as f64; k];

    for _ in 0..var_iters {
        // γ = α + Σ_n φ_n
        for kk in 0..k {
            gamma[kk] = alpha;
        }
        for (i, &(_, c)) in bag.iter().enumerate() {
            for kk in 0..k {
                gamma[kk] += c * phi[i][kk];
            }
        }
        let dig: Vec<f64> = gamma.iter().map(|&g| digamma(g)).collect();

        // Running Σ_n φ_n over all tokens.
        let mut sum_phi = vec![0.0; k];
        for (i, &(_, c)) in bag.iter().enumerate() {
            for kk in 0..k {
                sum_phi[kk] += c * phi[i][kk];
            }
        }

        for (i, &(word, c)) in bag.iter().enumerate() {
            // φ_{-n} excludes a single token of this word.
            let phi_minus: Vec<f64> = (0..k).map(|kk| sum_phi[kk] - phi[i][kk]).collect();
            let eta_dot_minus: f64 = if y.is_some() {
                (0..k).map(|kk| eta[kk] * phi_minus[kk]).sum()
            } else {
                0.0
            };

            let mut logp = vec![0.0; k];
            for kk in 0..k {
                let mut lp = dig[kk] + log_beta[kk][word];
                if let Some(yval) = y {
                    let n = n_tokens;
                    lp += (yval * eta[kk]) / (n * sigma2)
                        - (eta[kk] * eta[kk] + 2.0 * eta[kk] * eta_dot_minus)
                            / (2.0 * n * n * sigma2);
                }
                logp[kk] = lp;
            }
            // Normalize via log-sum-exp.
            let mx = logp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0.0;
            for kk in 0..k {
                logp[kk] = (logp[kk] - mx).exp();
                z += logp[kk];
            }
            let old = phi[i].clone();
            for kk in 0..k {
                phi[i][kk] = logp[kk] / z;
                // Keep sum_phi current as φ_i changes (all c tokens share φ_i).
                sum_phi[kk] += c * (phi[i][kk] - old[kk]);
            }
        }
    }

    let mut sum_phi = vec![0.0; k];
    for (i, &(_, c)) in bag.iter().enumerate() {
        for kk in 0..k {
            sum_phi[kk] += c * phi[i][kk];
        }
    }
    (gamma, phi, sum_phi)
}

/// Predict the response for a new document (ŷ = ηᵀ z̄, z̄ = Σφ / N).
pub fn predict_one(model: &SldaModel, doc: &[u32], var_iters: usize) -> f64 {
    predict_one_var(model, doc, var_iters).0
}

/// Predict the response and its posterior-predictive variance for a new document.
///
/// Returns `(ŷ, var)` with `ŷ = ηᵀ z̄` and
/// `var = ηᵀ Cov(z̄) η + σ²`, where the variational posterior over the topic
/// frequency `z̄ = (1/N) Σ_n z_n` (each token `z_n ~ Categorical(φ_n)`) has
/// `Cov(z̄) = (1/N²) Σ_n [diag(φ_n) − φ_n φ_nᵀ]`. The first term is the topic
/// uncertainty for this document, the second the irreducible response noise. An
/// empty document carries no topic information, so only the residual `σ²` remains.
pub fn predict_one_var(model: &SldaModel, doc: &[u32], var_iters: usize) -> (f64, f64) {
    let bag = to_bag(doc);
    if bag.is_empty() {
        return (0.0, model.sigma2);
    }
    let (_, phi, sum_phi) = infer_doc(
        &bag,
        &model.log_beta,
        &model.eta,
        model.sigma2,
        model.alpha,
        None,
        var_iters,
    );
    let k = model.num_topics;
    let n: f64 = bag.iter().map(|&(_, c)| c).sum();
    let mean: f64 = (0..k).map(|kk| model.eta[kk] * sum_phi[kk] / n).sum();
    // ηᵀ Cov(z̄) η = (1/N²)[ Σ_k η_k² sum_phi_k − Σ_w c_w (ηᵀ φ_w)² ].
    let diag_term: f64 = (0..k)
        .map(|kk| model.eta[kk] * model.eta[kk] * sum_phi[kk])
        .sum();
    let mut quad_term = 0.0f64;
    for (i, &(_, c)) in bag.iter().enumerate() {
        let eta_dot_phi: f64 = (0..k).map(|kk| model.eta[kk] * phi[i][kk]).sum();
        quad_term += c * eta_dot_phi * eta_dot_phi;
    }
    let topic_var = ((diag_term - quad_term) / (n * n)).max(0.0);
    (mean, topic_var + model.sigma2)
}

/// Standard errors of the regression coefficients `η`, from the OLS-style
/// covariance `σ² M⁻¹` where `M = Σ_d E[z̄ z̄ᵀ]` is the (ridge-stabilized)
/// normal-equations matrix that produced `η`. Returns length `num_topics`
/// (`NaN` where `M` is not invertible).
pub fn coefficient_se(m_mat: &[f64], sigma2: f64, num_topics: usize) -> Vec<f64> {
    match spd_inverse(m_mat, num_topics) {
        Some(minv) => (0..num_topics)
            .map(|a| {
                let v = sigma2 * minv[a * num_topics + a];
                if v > 0.0 {
                    v.sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect(),
        None => vec![f64::NAN; num_topics],
    }
}

/// Per-EM-iteration objective: the Gaussian log-likelihood of the response
/// under the current variational E[z̄_d] and model parameters (η, σ²).
/// This is a good scalar proxy for EM convergence.
fn response_log_likelihood(
    bags: &[Vec<(usize, f64)>],
    y: &[f64],
    log_beta: &[Vec<f64>],
    eta: &[f64],
    sigma2: f64,
    alpha: f64,
    var_iters: usize,
) -> f64 {
    let d = y.len();
    let mut ll = 0.0f64;
    for di in 0..d {
        let bag = &bags[di];
        if bag.is_empty() {
            continue;
        }
        let (_, _, sum_phi) = infer_doc(bag, log_beta, eta, sigma2, alpha, None, var_iters);
        let n: f64 = bag.iter().map(|&(_, c)| c).sum();
        let ezbar: f64 = (0..eta.len()).map(|kk| eta[kk] * sum_phi[kk] / n).sum();
        let resid = y[di] - ezbar;
        ll -= resid * resid / (2.0 * sigma2);
    }
    ll -= d as f64 * 0.5 * (2.0 * std::f64::consts::PI * sigma2).ln();
    ll
}

/// Fit a supervised-LDA model by variational EM.
///
/// Returns `(model, bound_history, converged)` where `bound_history` is a vector
/// of `(em_iteration, response_log_likelihood)` pairs, one per EM iteration when
/// `check_every > 0`.
#[allow(clippy::too_many_arguments)]
pub fn fit_slda<R: Rng>(
    docs: &[Vec<u32>],
    y: &[f64],
    num_types: usize,
    num_topics: usize,
    alpha: f64,
    em_iters: usize,
    var_iters: usize,
    convergence_tol: f64,
    check_every: usize,
    rng: &mut R,
) -> (SldaModel, Vec<(usize, f64)>, bool) {
    let k = num_topics;
    let v = num_types;
    let d = docs.len();

    // Seed β from a short static LDA, then take logs.
    let seed = crate::dtm::init_suffstats(docs, v, k, 50, rng);
    let mut log_beta = vec![vec![0.0; v]; k];
    for kk in 0..k {
        let total: f64 = (0..v).map(|w| seed[w][kk]).sum::<f64>() + v as f64 * 1e-6;
        for w in 0..v {
            log_beta[kk][w] = ((seed[w][kk] + 1e-6) / total).ln();
        }
    }

    let mut eta = vec![0.0; k];
    let mut sigma2 = 1.0;
    let mut gamma = vec![vec![alpha + 1.0; k]; d];

    let bags: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| to_bag(doc)).collect();
    let yty: f64 = y.iter().map(|v| v * v).sum();

    let mut bound_history: Vec<(usize, f64)> = Vec::new();
    let mut converged = false;
    let mut last_m = vec![0.0f64; k * k]; // final M = Σ_d E[z̄ z̄ᵀ], for coefficient SEs

    for em_iter in 1..=em_iters {
        let mut beta_ss = vec![vec![1e-6; v]; k]; // K × V, with smoothing
        let mut m_mat = vec![0.0f64; k * k]; // Σ_d E[z̄ z̄ᵀ]
        let mut b_vec = vec![0.0f64; k]; // Σ_d y_d E[z̄]

        // E-step: per-document inference is independent, so run it in parallel
        // and accumulate the sufficient statistics serially in document order so
        // the fit stays bit-for-bit identical regardless of thread count.
        let doc_results: Vec<(usize, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> = bags
            .par_iter()
            .enumerate()
            .filter(|(_, bag)| !bag.is_empty())
            .map(|(di, bag)| {
                let (g, phi, sum_phi) =
                    infer_doc(bag, &log_beta, &eta, sigma2, alpha, Some(y[di]), var_iters);
                (di, g, phi, sum_phi)
            })
            .collect();

        for (di, g, phi, sum_phi) in &doc_results {
            let di = *di;
            let bag = &bags[di];
            gamma[di] = g.clone();

            let n: f64 = bag.iter().map(|&(_, c)| c).sum();
            // β sufficient statistics.
            for (i, &(word, c)) in bag.iter().enumerate() {
                for kk in 0..k {
                    beta_ss[kk][word] += c * phi[i][kk];
                }
            }
            // E[z̄] and E[z̄ z̄ᵀ] for the η/σ² normal equations.
            let ezbar: Vec<f64> = sum_phi.iter().map(|&s| s / n).collect();
            for kk in 0..k {
                b_vec[kk] += y[di] * ezbar[kk];
            }
            // A_d = (1/N²)[ sum_phi sum_phiᵀ − Σ_w c_w φ_w φ_wᵀ + diag(sum_phi) ].
            let inv_n2 = 1.0 / (n * n);
            for a in 0..k {
                for b in 0..k {
                    m_mat[a * k + b] += inv_n2 * sum_phi[a] * sum_phi[b];
                }
            }
            for (i, &(_, c)) in bag.iter().enumerate() {
                for a in 0..k {
                    for b in 0..k {
                        m_mat[a * k + b] -= inv_n2 * c * phi[i][a] * phi[i][b];
                    }
                }
            }
            for a in 0..k {
                m_mat[a * k + a] += inv_n2 * sum_phi[a];
            }
        }

        // M-step: β.
        for kk in 0..k {
            let total: f64 = beta_ss[kk].iter().sum();
            for w in 0..v {
                log_beta[kk][w] = (beta_ss[kk][w] / total).ln();
            }
        }
        // M-step: η = M⁻¹ b (ridge-stabilized), σ² from the residual.
        for a in 0..k {
            m_mat[a * k + a] += 1e-6;
        }
        last_m.copy_from_slice(&m_mat); // retain the (ridge-stabilized) M for SEs
        if let Some(minv) = spd_inverse(&m_mat, k) {
            for a in 0..k {
                eta[a] = (0..k).map(|c| minv[a * k + c] * b_vec[c]).sum();
            }
            let eta_dot_b: f64 = (0..k).map(|a| eta[a] * b_vec[a]).sum();
            sigma2 = ((yty - eta_dot_b) / d as f64).max(1e-6);
        }

        // Bound trace and optional convergence check (uses current η/σ²/log_beta).
        if check_every > 0 && em_iter % check_every == 0 {
            let bnd = response_log_likelihood(&bags, y, &log_beta, &eta, sigma2, alpha, var_iters);
            bound_history.push((em_iter, bnd));
            if convergence_tol > 0.0 && bound_history.len() >= 2 {
                let prev = bound_history[bound_history.len() - 2].1;
                let rel = (bnd - prev).abs() / (prev.abs() + 1e-12);
                if rel < convergence_tol {
                    converged = true;
                    break;
                }
            }
        }
    }

    (
        SldaModel {
            num_topics: k,
            num_types: v,
            alpha,
            log_beta,
            eta,
            sigma2,
            gamma,
            m_mat: last_m,
        },
        bound_history,
        converged,
    )
}

/// Fit a supervised-LDA model by **collapsed Gibbs sampling** (the algorithm
/// used by \pkg{tomotopy}'s `SLDAModel`), an alternative to the variational EM of
/// [`fit_slda`]. The topic-word structure is the standard collapsed LDA sampler;
/// the response couples the tokens through the same Gaussian term the variational
/// φ update uses, applied to a *hard* assignment. Each token `n` in document `d`
/// is drawn with probability
///
/// ```text
/// p(z=k) ∝ (n_dk^{-n} + α) · (n_kw^{-n} + β)/(n_k^{-n} + Vβ)
///          · exp( η_k(y_d − η·z̄_d^{-n})/(N_d σ²) − η_k²/(2 N_d² σ²) )
/// ```
///
/// where `z̄_d^{-n}` is the doc's topic frequency excluding token `n`. After each
/// sweep, `η` is re-estimated by MAP ridge regression of `y` on the sampled topic
/// frequencies `z̄`, with the ridge `λ = σ²/nu_sq` from \pkg{tomotopy}'s Gaussian
/// coefficient prior `N(0, nu_sq)` (default `nu_sq = 1`). The response variance
/// `σ²` (tomotopy's `glm_param`) is a fixed hyperparameter (default `1.0`), not
/// re-estimated, matching tomotopy. `β` (the word prior) defaults to `0.01`, also
/// matching \pkg{tomotopy}.
///
/// Returns `(model, response_ll_history, converged=false)`, mirroring
/// [`fit_slda`]'s shape; the history records `(sweep, response_log_likelihood)`
/// every `check_every` sweeps (no early stopping — Gibbs runs the full `iters`).
#[allow(clippy::too_many_arguments)]
pub fn fit_slda_gibbs<R: Rng>(
    docs: &[Vec<u32>],
    y: &[f64],
    num_types: usize,
    num_topics: usize,
    alpha: f64,
    iters: usize,
    check_every: usize,
    rng: &mut R,
) -> (SldaModel, Vec<(usize, f64)>, bool) {
    let k = num_topics;
    let v = num_types;
    let d = docs.len();
    let beta = 0.01_f64; // word prior; tomotopy SLDAModel default `eta`
    let beta_sum = v as f64 * beta;
    let nu_sq = 1.0_f64; // tomotopy coefficient-prior variance N(0, nu_sq)
                         // Response variance σ² (tomotopy's `glm_param` for a linear variable) is a
                         // fixed hyperparameter, default 1.0 — it is NOT re-estimated. Keeping it fixed
                         // matches tomotopy and keeps the supervised-term scaling identical across
                         // engines.
    let sigma2 = 1.0_f64;

    // Token-topic assignments and the count tables behind the collapsed sampler.
    // usize counts so a large corpus cannot overflow a topic/word total.
    let mut z: Vec<Vec<usize>> = docs.iter().map(|doc| vec![0usize; doc.len()]).collect();
    let mut n_dk = vec![vec![0usize; k]; d]; // D × K
    let mut n_kw = vec![vec![0usize; v]; k]; // K × V
    let mut n_k = vec![0usize; k]; // K
    for (di, doc) in docs.iter().enumerate() {
        for (n, &w) in doc.iter().enumerate() {
            let t = rng.gen_range(0..k);
            z[di][n] = t;
            n_dk[di][t] += 1;
            n_kw[t][w as usize] += 1;
            n_k[t] += 1;
        }
    }

    let mut eta = vec![0.0f64; k];
    let mut probs = vec![0.0f64; k];
    let mut logp = vec![0.0f64; k];
    let mut bound_history: Vec<(usize, f64)> = Vec::new();
    let mut last_m = vec![0.0f64; k * k];

    for sweep in 1..=iters {
        for (di, doc) in docs.iter().enumerate() {
            let nd = doc.len() as f64;
            if nd == 0.0 {
                continue;
            }
            let yd = y[di];
            for (n, &w) in doc.iter().enumerate() {
                let w = w as usize;
                let old = z[di][n];
                n_dk[di][old] -= 1;
                n_kw[old][w] -= 1;
                n_k[old] -= 1;
                // η·z̄^{-n}: topic frequencies of the doc with this token removed.
                let eta_dot_minus: f64 = (0..k).map(|t| eta[t] * (n_dk[di][t] as f64) / nd).sum();
                // Accumulate the conditional in log-space, then softmax — the
                // supervised term can be large in magnitude, so exponentiating each
                // factor directly (`lda * sup.exp()`) risks overflow to +inf / NaN.
                let mut mx = f64::NEG_INFINITY;
                for t in 0..k {
                    let lda = (n_dk[di][t] as f64 + alpha) * (n_kw[t][w] as f64 + beta)
                        / (n_k[t] as f64 + beta_sum);
                    // Same Gaussian response coupling as the variational φ update,
                    // applied to a hard assignment (see module docs).
                    let sup = eta[t] * (yd - eta_dot_minus) / (nd * sigma2)
                        - eta[t] * eta[t] / (2.0 * nd * nd * sigma2);
                    logp[t] = lda.ln() + sup; // lda > 0 (α, β > 0) so ln is finite
                    if logp[t] > mx {
                        mx = logp[t];
                    }
                }
                let mut sum = 0.0f64;
                for t in 0..k {
                    let e = (logp[t] - mx).exp();
                    probs[t] = e;
                    sum += e;
                }
                // Sample a new topic from the (finite, positive) weights.
                let mut r = rng.gen::<f64>() * sum;
                let mut newt = k - 1;
                for t in 0..k {
                    r -= probs[t];
                    if r < 0.0 {
                        newt = t;
                        break;
                    }
                }
                z[di][n] = newt;
                n_dk[di][newt] += 1;
                n_kw[newt][w] += 1;
                n_k[newt] += 1;
            }
        }

        // Re-estimate η by MAP ridge regression of y on the sampled topic
        // frequencies z̄_d (σ² is fixed, see above). M = Σ_d z̄ z̄ᵀ, b = Σ_d y_d z̄;
        // the ridge λ = σ²/nu_sq is the coefficient prior N(0, nu_sq).
        let mut m_mat = vec![0.0f64; k * k];
        let mut b_vec = vec![0.0f64; k];
        for (di, doc) in docs.iter().enumerate() {
            let nd = doc.len() as f64;
            if nd == 0.0 {
                continue;
            }
            let zbar: Vec<f64> = (0..k).map(|t| n_dk[di][t] as f64 / nd).collect();
            for a in 0..k {
                b_vec[a] += y[di] * zbar[a];
                for b in 0..k {
                    m_mat[a * k + b] += zbar[a] * zbar[b];
                }
            }
        }
        let ridge = sigma2 / nu_sq;
        for a in 0..k {
            m_mat[a * k + a] += ridge;
        }
        last_m.copy_from_slice(&m_mat);
        if let Some(minv) = spd_inverse(&m_mat, k) {
            for a in 0..k {
                eta[a] = (0..k).map(|c| minv[a * k + c] * b_vec[c]).sum();
            }
        }

        if check_every > 0 && sweep % check_every == 0 {
            // Response log-likelihood under the current hard z̄ and (η, σ²).
            let mut ll = 0.0f64;
            for (di, doc) in docs.iter().enumerate() {
                let nd = doc.len() as f64;
                if nd == 0.0 {
                    continue;
                }
                let ezbar: f64 = (0..k).map(|t| eta[t] * (n_dk[di][t] as f64) / nd).sum();
                let resid = y[di] - ezbar;
                ll -= resid * resid / (2.0 * sigma2);
            }
            ll -= d as f64 * 0.5 * (2.0 * std::f64::consts::PI * sigma2).ln();
            bound_history.push((sweep, ll));
        }
    }

    // Read out β from the final counts and γ = α + topic counts (for doc_topic).
    let mut log_beta = vec![vec![0.0; v]; k];
    for kk in 0..k {
        let denom = n_k[kk] as f64 + beta_sum;
        for w in 0..v {
            log_beta[kk][w] = ((n_kw[kk][w] as f64 + beta) / denom).ln();
        }
    }
    let gamma: Vec<Vec<f64>> = (0..d)
        .map(|di| (0..k).map(|t| alpha + n_dk[di][t] as f64).collect())
        .collect();

    (
        SldaModel {
            num_topics: k,
            num_types: v,
            alpha,
            log_beta,
            eta,
            sigma2,
            gamma,
            m_mat: last_m,
        },
        bound_history,
        false,
    )
}

use crate::estimator::{DirichletModel, Estimator, ModelFamily};

impl Estimator for SldaModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        // The bound history is returned separately by fit_slda, not stored — leave empty.
        Vec::new()
    }

    fn converged(&self) -> Option<bool> {
        None
    }

    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

impl DirichletModel for SldaModel {
    fn alpha(&self) -> Vec<f64> {
        vec![self.alpha; self.num_topics]
    }

    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        Vec::new()
    }

    fn doc_lengths(&self) -> Vec<usize> {
        // doc_lengths reconstructed from γ (sLDA stores no raw token counts):
        // γ[d] = α + token-topic counts, so N_d ≈ Σγ[d] − α·K.
        self.gamma
            .iter()
            .map(|g| {
                let s: f64 = g.iter().sum();
                (s - self.alpha * self.num_topics as f64).round().max(0.0) as usize
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a corpus with two topics (disjoint vocab). A document's response is
    /// driven by its topic mix: topic 0 pushes y up, topic 1 pushes it down.
    fn supervised_corpus(rng: &mut ChaCha8Rng) -> (Vec<Vec<u32>>, Vec<f64>, usize) {
        let v = 12;
        let t0 = [0u32, 1, 2, 3, 4, 5];
        let t1 = [6u32, 7, 8, 9, 10, 11];
        let mut docs = Vec::new();
        let mut y = Vec::new();
        for _ in 0..200 {
            // Mixing proportion p of topic 0.
            let p = rng.gen::<f64>();
            let mut doc = Vec::new();
            for _ in 0..20 {
                if rng.gen::<f64>() < p {
                    doc.push(t0[(rng.gen::<f64>() * 6.0) as usize % 6]);
                } else {
                    doc.push(t1[(rng.gen::<f64>() * 6.0) as usize % 6]);
                }
            }
            docs.push(doc);
            // Response: higher when topic 0 dominates, plus small noise.
            let noise = (rng.gen::<f64>() - 0.5) * 0.2;
            y.push(2.0 * p - 1.0 + noise);
        }
        (docs, y, v)
    }

    #[test]
    fn recovers_predictive_topics() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, y, v) = supervised_corpus(&mut rng);
        let (model, _, _) = fit_slda(&docs, &y, v, 2, 0.1, 25, 15, 0.0, 0, &mut rng);

        // The two topics should separate the two vocabularies.
        let tw = model.topic_word();
        let topic_of_block = |block: &[usize]| -> usize {
            // Which topic puts more mass on this vocabulary block.
            let m0: f64 = block.iter().map(|&w| tw[0][w]).sum();
            let m1: f64 = block.iter().map(|&w| tw[1][w]).sum();
            if m0 > m1 {
                0
            } else {
                1
            }
        };
        let k0 = topic_of_block(&[0, 1, 2, 3, 4, 5]);
        let k1 = topic_of_block(&[6, 7, 8, 9, 10, 11]);
        assert_ne!(k0, k1, "topics did not separate the two vocabularies");

        // The coefficient on the topic-0 vocabulary should exceed that on topic 1
        // (topic 0 drives the response up).
        assert!(
            model.eta[k0] > model.eta[k1],
            "eta should rank topic-0 above topic-1: {:?}",
            model.eta
        );

        // Predictions should correlate strongly with the true responses.
        let preds: Vec<f64> = docs.iter().map(|d| predict_one(&model, d, 20)).collect();
        let corr = pearson(&preds, &y);
        assert!(corr > 0.7, "prediction correlation too low: {}", corr);
    }

    #[test]
    fn coefficient_se_and_predictive_variance() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, y, v) = supervised_corpus(&mut rng);
        let (model, _, _) = fit_slda(&docs, &y, v, 2, 0.1, 25, 15, 0.0, 0, &mut rng);

        // Coefficient SEs: finite, positive, and the strong predictive topic's
        // coefficient is many SEs from zero on this well-separated corpus.
        let se = coefficient_se(&model.m_mat, model.sigma2, 2);
        assert_eq!(se.len(), 2);
        assert!(
            se.iter().all(|s| s.is_finite() && *s > 0.0),
            "bad SE: {se:?}"
        );
        let hi = if model.eta[0].abs() >= model.eta[1].abs() {
            0
        } else {
            1
        };
        assert!(
            model.eta[hi].abs() / se[hi] > 2.0,
            "predictive coefficient should be significant: eta={:?} se={se:?}",
            model.eta
        );

        // Predictive variance is at least the residual σ² and finite; the mean
        // matches predict_one.
        for d in docs.iter().take(20) {
            let (m, var) = predict_one_var(&model, d, 20);
            assert!(var.is_finite() && var >= model.sigma2 - 1e-9, "var={var}");
            assert!((m - predict_one(&model, d, 20)).abs() < 1e-9);
        }
        // An empty document falls back to the residual variance.
        let (_, var0) = predict_one_var(&model, &[], 20);
        assert!((var0 - model.sigma2).abs() < 1e-12);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut r0 = ChaCha8Rng::seed_from_u64(3);
        let (docs, y, v) = supervised_corpus(&mut r0);
        let mut r1 = ChaCha8Rng::seed_from_u64(9);
        let mut r2 = ChaCha8Rng::seed_from_u64(9);
        let (m1, _, _) = fit_slda(&docs, &y, v, 2, 0.1, 10, 10, 0.0, 0, &mut r1);
        let (m2, _, _) = fit_slda(&docs, &y, v, 2, 0.1, 10, 10, 0.0, 0, &mut r2);
        assert_eq!(m1.eta, m2.eta);
        assert_eq!(m1.sigma2, m2.sigma2);
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for i in 0..a.len() {
            cov += (a[i] - ma) * (b[i] - mb);
            va += (a[i] - ma).powi(2);
            vb += (b[i] - mb).powi(2);
        }
        cov / (va.sqrt() * vb.sqrt())
    }

    #[test]
    fn gibbs_recovers_predictive_topics() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, y, v) = supervised_corpus(&mut rng);
        let (model, hist, conv) = fit_slda_gibbs(&docs, &y, v, 2, 0.1, 300, 50, &mut rng);
        assert!(!conv); // Gibbs runs the full sweep budget
        assert!(!hist.is_empty(), "expected a response-ll trace");

        // The two topics should separate the two disjoint vocabularies.
        let tw = model.topic_word();
        let mass = |t: usize, block: &[usize]| -> f64 { block.iter().map(|&w| tw[t][w]).sum() };
        let k0 = if mass(0, &[0, 1, 2, 3, 4, 5]) > mass(1, &[0, 1, 2, 3, 4, 5]) {
            0
        } else {
            1
        };
        let k1 = 1 - k0;
        assert!(
            mass(k1, &[6, 7, 8, 9, 10, 11]) > mass(k0, &[6, 7, 8, 9, 10, 11]),
            "topics did not separate the two vocabularies"
        );
        // Topic 0's vocabulary drives the response up, so its coefficient leads.
        assert!(
            model.eta[k0] > model.eta[k1],
            "eta should rank topic-0 above topic-1: {:?}",
            model.eta
        );
        // Predictions correlate strongly with the true responses.
        let preds: Vec<f64> = docs.iter().map(|d| predict_one(&model, d, 20)).collect();
        let corr = pearson(&preds, &y);
        assert!(corr > 0.7, "prediction correlation too low: {corr}");
    }

    #[test]
    fn gibbs_deterministic_for_fixed_seed() {
        let mut r0 = ChaCha8Rng::seed_from_u64(3);
        let (docs, y, v) = supervised_corpus(&mut r0);
        let mut r1 = ChaCha8Rng::seed_from_u64(9);
        let mut r2 = ChaCha8Rng::seed_from_u64(9);
        let (m1, _, _) = fit_slda_gibbs(&docs, &y, v, 2, 0.1, 50, 0, &mut r1);
        let (m2, _, _) = fit_slda_gibbs(&docs, &y, v, 2, 0.1, 50, 0, &mut r2);
        assert_eq!(m1.eta, m2.eta);
        assert_eq!(m1.sigma2, m2.sigma2);
        assert_eq!(m1.log_beta, m2.log_beta);
    }

    #[test]
    fn gibbs_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, y, v) = supervised_corpus(&mut rng);
        let (m, _, _) = fit_slda_gibbs(&docs, &y, v, 2, 0.1, 100, 0, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }

    #[test]
    fn slda_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, y, v) = supervised_corpus(&mut rng);
        let (m, _, _) = fit_slda(&docs, &y, v, 2, 0.1, 25, 15, 0.0, 0, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }
}
