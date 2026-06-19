//! Supervised LDA (Blei & McAuliffe 2007) — LDA with a per-document **response
//! variable** `y_d` regressed on the document's topic usage. Each document's
//! response is Gaussian, `y_d ~ N(ηᵀ z̄_d, σ²)`, where `z̄_d` is the empirical
//! topic frequency of its words. Fitting the topics is thus *supervised* by the
//! response: topics are shaped to be predictive of `y`, and the fitted
//! regression coefficients `η` say how each topic moves the response.
//!
//! Inference is the variational EM of Blei & McAuliffe (2007):
//!   - **E-step** (per document): coordinate ascent on the variational `γ`
//!     (Dirichlet) and `φ` (per-word topic) parameters. The `φ` update carries
//!     the response-coupling term that ties the words together through `η`/`σ²`.
//!   - **M-step**: `β` from the expected word-topic counts (as in LDA); `η` by
//!     the normal equations `η = (Σ_d E[z̄_d z̄_dᵀ])⁻¹ Σ_d y_d E[z̄_d]`; and
//!     `σ²` from the residual.
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
    let bag = to_bag(doc);
    if bag.is_empty() {
        return 0.0;
    }
    let (_, _, sum_phi) = infer_doc(
        &bag,
        &model.log_beta,
        &model.eta,
        model.sigma2,
        model.alpha,
        None,
        var_iters,
    );
    let n: f64 = bag.iter().map(|&(_, c)| c).sum();
    (0..model.num_topics)
        .map(|k| model.eta[k] * sum_phi[k] / n)
        .sum()
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
        },
        bound_history,
        converged,
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
