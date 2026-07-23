//! RTM: the Relational Topic Model (Chang & Blei, "Hierarchical Relational
//! Models for Document Networks", Annals of Applied Statistics 4(1), 2010).
//!
//! RTM is LDA plus a link model: for each observed pair of documents a binary
//! link is drawn from a function of the two documents' mean topic assignments
//! `z̄_d = (1/N_d) Σ_n z_{d,n}`. Coupling the link to `z` (not `θ`) forces the
//! same topics to explain both words and links, which is what lets the model
//! predict links from words and words from links. We fit by variational EM
//! (paper §3 / Appendices A-B), modelling only the observed (positive) links so
//! cost scales with the number of links, not with `D²`.
//!
//! Two link functions:
//!   - logistic  `ψ_σ(y=1) = σ(ηᵀ(z̄_d ∘ z̄_{d'}) + ν)`      (eq 2.1) — default,
//!   - exponential `ψ_e(y=1) = exp(ηᵀ(z̄_d ∘ z̄_{d'}) + ν)`   (eq 2.2).
//!
//! The one-class link estimate diverges on positive-only data, so both paths use
//! the paper's ρ regularization: `ρ` pseudo-negative links placed at the expected
//! Hadamard product under the Dirichlet prior, `π̄_α = (α/1ᵀα) ∘ (α/1ᵀα)`.
//!
//! Validated against a standalone NumPy implementation of the same equations
//! (`parity/rtm_reference.py`); the R `lda` package's `rtm.em` is a *collapsed
//! Gibbs* sampler, so it is only a directional baseline (`parity/rtm_compare.py`).
//!
//! Pure Rust, no PyO3. Fitted state stores matrices as `Vec<Vec<f64>>`.

use crate::corpus::Corpus;
use crate::estimator::{Estimator, ModelFamily};
use crate::optimize::digamma;
use rand::Rng;

/// Which link probability function couples topics to links.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Link {
    Logistic,
    Exponential,
}

impl Link {
    pub fn parse(s: &str) -> Result<Link, String> {
        match s {
            "logistic" | "sigmoid" => Ok(Link::Logistic),
            "exponential" | "exp" => Ok(Link::Exponential),
            other => Err(format!(
                "link must be 'logistic' or 'exponential', got '{other}'"
            )),
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Link::Logistic => "logistic",
            Link::Exponential => "exponential",
        }
    }
}

/// Fitted state for [`fit_rtm`].
pub struct RTMModel {
    pub num_topics: usize,
    pub topic_word: Vec<Vec<f64>>, // K rows of length V (β, rows sum to 1)
    pub doc_topic: Vec<Vec<f64>>,  // D rows of length K (γ normalized, rows sum to 1)
    pub phi_bar: Vec<Vec<f64>>,    // D rows of length K (mean topic assignment; link quantity)
    pub eta: Vec<f64>,             // K link coefficients
    pub nu: f64,                   // link intercept
    pub link: Link,
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// numerically stable `log(1 + exp(x))`.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// `∇_{π̄} E_q[log ψ(π̄)]` for a single link (a K-vector).
/// logistic: `(1 - σ(ηᵀπ̄ + ν)) η`; exponential: `η` (exact).
fn link_grad_pi(link: Link, eta: &[f64], nu: f64, pi_bar: &[f64], out: &mut [f64]) {
    match link {
        Link::Exponential => out.copy_from_slice(eta),
        Link::Logistic => {
            let mut dot = nu;
            for k in 0..eta.len() {
                dot += eta[k] * pi_bar[k];
            }
            let w = 1.0 - sigmoid(dot);
            for k in 0..eta.len() {
                out[k] = w * eta[k];
            }
        }
    }
}

/// `π̄_α = (α/1ᵀα) ∘ (α/1ᵀα)`, the expected Hadamard product under Dir(α).
fn pi_alpha(alpha: &[f64]) -> Vec<f64> {
    let s: f64 = alpha.iter().sum();
    alpha.iter().map(|&a| (a / s) * (a / s)).collect()
}

/// Analytic exponential-link M-step (paper App B, page 147).
fn fit_eta_exponential(pi_sum: &[f64], num_links: f64, pa: &[f64], rho: f64) -> (Vec<f64>, f64) {
    let eps = 1e-12;
    let one_t_pi: f64 = pi_sum.iter().sum();
    let one_t_pa: f64 = pa.iter().sum();
    let resid = (num_links - one_t_pi).max(eps);
    let nu = resid.ln() - (rho * (1.0 - one_t_pa).max(0.0) + resid).ln();
    let eta = pi_sum
        .iter()
        .zip(pa.iter())
        .map(|(&pi, &pa_k)| pi.max(eps).ln() - (pi + rho * pa_k + eps).ln() - nu)
        .collect();
    (eta, nu)
}

/// Regularized logistic-link M-step by gradient ascent with backtracking.
/// Objective (concave):
///   `Σ_links log σ(ηᵀπ + ν) + ρ log(1 - σ(ηᵀπ_α + ν)) - λ‖η‖²`.
fn fit_eta_logistic(
    pi_list: &[Vec<f64>],
    pa: &[f64],
    rho: f64,
    ridge: f64,
    eta0: &[f64],
    nu0: f64,
    iters: usize,
) -> (Vec<f64>, f64) {
    let k = pa.len();
    let mut eta = eta0.to_vec();
    let mut nu = nu0;

    let obj = |eta: &[f64], nu: f64| -> f64 {
        let mut ll = 0.0;
        for pi in pi_list {
            let mut z = nu;
            for j in 0..k {
                z += eta[j] * pi[j];
            }
            ll -= softplus(-z); // log σ(z)
        }
        let mut za = nu;
        for j in 0..k {
            za += eta[j] * pa[j];
        }
        ll += rho * (-softplus(za)); // ρ log(1 - σ(za))
        for j in 0..k {
            ll -= ridge * eta[j] * eta[j];
        }
        ll
    };

    let grad = |eta: &[f64], nu: f64| -> (Vec<f64>, f64) {
        let mut g_eta = vec![0.0; k];
        let mut g_nu = 0.0;
        for pi in pi_list {
            let mut z = nu;
            for j in 0..k {
                z += eta[j] * pi[j];
            }
            let w = sigmoid(-z); // 1 - σ(z)
            for j in 0..k {
                g_eta[j] += w * pi[j];
            }
            g_nu += w;
        }
        let mut za = nu;
        for j in 0..k {
            za += eta[j] * pa[j];
        }
        let sa = sigmoid(za);
        for j in 0..k {
            g_eta[j] -= rho * sa * pa[j] + 2.0 * ridge * eta[j];
        }
        g_nu -= rho * sa;
        (g_eta, g_nu)
    };

    let step0 = 1.0 / (pi_list.len().max(1) as f64);
    let mut f = obj(&eta, nu);
    for _ in 0..iters {
        let (g_eta, g_nu) = grad(&eta, nu);
        let gnorm2: f64 = g_eta.iter().map(|g| g * g).sum::<f64>() + g_nu * g_nu;
        if gnorm2.sqrt() < 1e-8 {
            break;
        }
        let mut s = step0;
        let mut improved = false;
        for _ in 0..40 {
            let e2: Vec<f64> = (0..k).map(|j| eta[j] + s * g_eta[j]).collect();
            let n2 = nu + s * g_nu;
            let f2 = obj(&e2, n2);
            if f2 >= f + 1e-4 * s * gnorm2 {
                eta = e2;
                nu = n2;
                f = f2;
                improved = true;
                break;
            }
            s *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (eta, nu)
}

/// Per-document unique-word bag: `(word_id, count)` pairs.
fn to_bag(doc: &[u32]) -> Vec<(usize, f64)> {
    let mut counts: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    for &w in doc {
        *counts.entry(w).or_insert(0.0) += 1.0;
    }
    counts.into_iter().map(|(w, c)| (w as usize, c)).collect()
}

/// Undirected adjacency (only observed links) + the deduped, sorted edge list.
fn build_adjacency(
    edges: &[(usize, usize)],
    num_docs: usize,
) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    use std::collections::BTreeSet;
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); num_docs];
    let mut obs: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(i, j) in edges {
        if i == j || i >= num_docs || j >= num_docs {
            continue;
        }
        adj[i].insert(j);
        adj[j].insert(i);
        obs.insert(if i < j { (i, j) } else { (j, i) });
    }
    let adj_vec = adj.into_iter().map(|s| s.into_iter().collect()).collect();
    (adj_vec, obs.into_iter().collect())
}

/// Hyperparameters for [`fit_rtm`].
pub struct RtmParams {
    pub num_topics: usize,
    pub num_types: usize,
    pub alpha: f64,
    pub link: Link,
    /// Explicit pseudo-negative count. When `None`, `negative_ratio * num_links`.
    pub rho: Option<f64>,
    pub negative_ratio: f64,
    pub ridge: f64,
    pub em_iters: usize,
    pub e_sweeps: usize,
    pub e_inner: usize,
    pub var_tol: f64,
    pub convergence_tol: f64,
}

/// Fit RTM by variational EM. `docs` are token-id lists; `edges` are undirected
/// observed links between document indices. Deterministic given `rng` and a
/// serial (Gauss-Seidel) E-step.
pub fn fit_rtm<R: Rng>(
    docs: &[Vec<u32>],
    edges: &[(usize, usize)],
    p: &RtmParams,
    rng: &mut R,
) -> RTMModel {
    let k = p.num_topics;
    let v = p.num_types;
    let d = docs.len();
    let alpha = vec![p.alpha; k];

    let bags: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| to_bag(doc)).collect();
    let nd: Vec<f64> = bags
        .iter()
        .map(|b| b.iter().map(|&(_, c)| c).sum())
        .collect();
    let (adj, obs_links) = build_adjacency(edges, d);
    let m = obs_links.len() as f64;
    let rho = p.rho.unwrap_or(p.negative_ratio * m.max(1.0));
    let pa = pi_alpha(&alpha);

    // init: β from smoothed random counts; φ̄ uniform; γ = α + N/K.
    let mut log_beta = vec![vec![0.0; v]; k];
    for row in log_beta.iter_mut() {
        let mut s = 0.0;
        let draws: Vec<f64> = (0..v).map(|_| rng.gen::<f64>() + 1e-8).collect();
        for &x in &draws {
            s += x;
        }
        for (w, val) in row.iter_mut().enumerate() {
            *val = (draws[w] / s).ln();
        }
    }
    let mut phi_bar = vec![vec![1.0 / k as f64; k]; d];
    let mut gamma: Vec<Vec<f64>> = (0..d)
        .map(|di| (0..k).map(|kk| alpha[kk] + nd[di] / k as f64).collect())
        .collect();
    let mut phi_store: Vec<Vec<Vec<f64>>> = bags
        .iter()
        .map(|b| vec![vec![1.0 / k as f64; k]; b.len()])
        .collect();
    let mut eta = vec![0.0; k];
    let mut nu = 0.0;
    let mut history = Vec::with_capacity(p.em_iters);
    let mut converged = false;
    let mut prev_obj = f64::NEG_INFINITY;

    let mut scratch = vec![0.0; k]; // link-grad output

    for it in 0..p.em_iters {
        // ---------------- E-step (coupled, serial Gauss-Seidel) -------------
        for _sweep in 0..p.e_sweeps {
            let mut max_delta = 0.0_f64;
            for di in 0..d {
                let bag = &bags[di];
                if bag.is_empty() || nd[di] == 0.0 {
                    continue;
                }
                let gsum: f64 = gamma[di].iter().sum();
                let dsum = digamma(gsum);
                let mut elogtheta: Vec<f64> =
                    gamma[di].iter().map(|&g| digamma(g) - dsum).collect();

                // link message m_d (constant across tokens), eqs (3.5)/(3.6)
                let mut m_d = vec![0.0; k];
                let compute_message = |phi_bar: &Vec<Vec<f64>>,
                                       eta: &[f64],
                                       nu: f64,
                                       scratch: &mut [f64],
                                       out: &mut [f64]| {
                    for o in out.iter_mut() {
                        *o = 0.0;
                    }
                    if !adj[di].is_empty() {
                        for &dp in &adj[di] {
                            let pib: Vec<f64> =
                                (0..k).map(|kk| phi_bar[di][kk] * phi_bar[dp][kk]).collect();
                            link_grad_pi(p.link, eta, nu, &pib, scratch);
                            for kk in 0..k {
                                out[kk] += scratch[kk] * phi_bar[dp][kk];
                            }
                        }
                        for o in out.iter_mut() {
                            *o /= nd[di];
                        }
                    }
                };
                compute_message(&phi_bar, &eta, nu, &mut scratch, &mut m_d);

                let prev = phi_bar[di].clone();
                for _inner in 0..p.e_inner {
                    // φ_{d,w,k} ∝ β_{k,w} exp(Elogθ_k + m_{d,k})
                    let phi = &mut phi_store[di];
                    for (i, &(word, _)) in bag.iter().enumerate() {
                        let mut mx = f64::NEG_INFINITY;
                        for kk in 0..k {
                            let lp = log_beta[kk][word] + elogtheta[kk] + m_d[kk];
                            phi[i][kk] = lp;
                            if lp > mx {
                                mx = lp;
                            }
                        }
                        let mut z = 0.0;
                        for kk in 0..k {
                            let e = (phi[i][kk] - mx).exp();
                            phi[i][kk] = e;
                            z += e;
                        }
                        for kk in 0..k {
                            phi[i][kk] /= z;
                        }
                    }
                    // φ̄_d = (1/N_d) Σ_w c_w φ_w ; γ_d = α + Σ_w c_w φ_w
                    for kk in 0..k {
                        let mut acc = 0.0;
                        for (i, &(_, c)) in bag.iter().enumerate() {
                            acc += c * phi[i][kk];
                        }
                        phi_bar[di][kk] = acc / nd[di];
                        gamma[di][kk] = alpha[kk] + acc;
                    }
                    let gsum: f64 = gamma[di].iter().sum();
                    let dsum = digamma(gsum);
                    for kk in 0..k {
                        elogtheta[kk] = digamma(gamma[di][kk]) - dsum;
                    }
                    // logistic message depends on updated φ̄_d — recompute
                    if p.link == Link::Logistic && !adj[di].is_empty() {
                        compute_message(&phi_bar, &eta, nu, &mut scratch, &mut m_d);
                    }
                }
                let delta: f64 = (0..k)
                    .map(|kk| (phi_bar[di][kk] - prev[kk]).abs())
                    .fold(0.0, f64::max);
                max_delta = max_delta.max(delta);
            }
            if max_delta < p.var_tol {
                break;
            }
        }

        // ---------------- M-step: β -----------------------------------------
        let mut ss = vec![vec![1e-6; v]; k];
        for di in 0..d {
            let phi = &phi_store[di];
            for (i, &(word, c)) in bags[di].iter().enumerate() {
                for kk in 0..k {
                    ss[kk][word] += c * phi[i][kk];
                }
            }
        }
        for kk in 0..k {
            let s: f64 = ss[kk].iter().sum();
            for w in 0..v {
                log_beta[kk][w] = (ss[kk][w] / s).ln();
            }
        }

        // ---------------- M-step: η, ν --------------------------------------
        if m > 0.0 {
            let pi_list: Vec<Vec<f64>> = obs_links
                .iter()
                .map(|&(i, j)| (0..k).map(|kk| phi_bar[i][kk] * phi_bar[j][kk]).collect())
                .collect();
            match p.link {
                Link::Exponential => {
                    let mut pi_sum = vec![0.0; k];
                    for pi in &pi_list {
                        for kk in 0..k {
                            pi_sum[kk] += pi[kk];
                        }
                    }
                    let (e, n) = fit_eta_exponential(&pi_sum, m, &pa, rho);
                    eta = e;
                    nu = n;
                }
                Link::Logistic => {
                    let (e, n) = fit_eta_logistic(&pi_list, &pa, rho, p.ridge, &eta, nu, 200);
                    eta = e;
                    nu = n;
                }
            }
        }

        let obj = objective(
            &bags, &phi_store, &phi_bar, &gamma, &log_beta, &alpha, &eta, nu, &obs_links, p.link,
        );
        history.push((it, obj));
        if (obj - prev_obj).abs() < p.convergence_tol * prev_obj.abs().max(1.0) && it > 0 {
            converged = true;
            break;
        }
        prev_obj = obj;
    }

    let topic_word: Vec<Vec<f64>> = log_beta
        .iter()
        .map(|row| row.iter().map(|&x| x.exp()).collect())
        .collect();
    let doc_topic: Vec<Vec<f64>> = gamma
        .iter()
        .map(|g| {
            let s: f64 = g.iter().sum();
            g.iter().map(|&x| x / s).collect()
        })
        .collect();

    RTMModel {
        num_topics: k,
        topic_word,
        doc_topic,
        phi_bar,
        eta,
        nu,
        link: p.link,
        fit_history: history,
        converged,
    }
}

/// `log Γ(z)` (Lanczos-free Stirling with recurrence), for the Dirichlet terms of
/// the variational bound.
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

/// The variational objective (evidence lower bound, paper §3.2): the word,
/// topic-assignment, Dirichlet-`θ`, and link expected log-likelihood terms plus
/// the `q` entropy. Used as the convergence criterion. It generally increases
/// across EM but is not guaranteed strictly monotone, for two reasons: the
/// logistic link contributes the first-order (Braun-McAuliffe) bound the M-step
/// assumes rather than the exact expectation, and the link M-step maximizes a
/// `ρ`-regularized objective, not this bound. The topic and link estimates do not
/// depend on it.
#[allow(clippy::too_many_arguments)]
fn objective(
    bags: &[Vec<(usize, f64)>],
    phi_store: &[Vec<Vec<f64>>],
    phi_bar: &[Vec<f64>],
    gamma: &[Vec<f64>],
    log_beta: &[Vec<f64>],
    alpha: &[f64],
    eta: &[f64],
    nu: f64,
    obs_links: &[(usize, usize)],
    link: Link,
) -> f64 {
    let k = eta.len();
    let alpha_sum: f64 = alpha.iter().sum();
    let lg_alpha_sum = lgamma(alpha_sum);
    let sum_lg_alpha: f64 = alpha.iter().map(|&a| lgamma(a)).sum();
    let mut acc = 0.0;
    for di in 0..bags.len() {
        let bag = &bags[di];
        if bag.is_empty() {
            continue;
        }
        let gsum: f64 = gamma[di].iter().sum();
        let dsum = digamma(gsum);
        let elogtheta: Vec<f64> = gamma[di].iter().map(|&g| digamma(g) - dsum).collect();
        let phi = &phi_store[di];
        for (i, &(word, c)) in bag.iter().enumerate() {
            for kk in 0..k {
                let p = phi[i][kk];
                acc += c * p * log_beta[kk][word]; // E[log p(w|z,β)]
                acc += c * p * (elogtheta[kk] - (p + 1e-300).ln()); // E[log p(z|θ)] + H(q(z))
            }
        }
        // Dirichlet θ terms: E[log p(θ|α)] − E[log q(θ|γ)].
        acc += lg_alpha_sum - sum_lg_alpha - lgamma(gsum);
        for kk in 0..k {
            acc += (alpha[kk] - gamma[di][kk]) * elogtheta[kk] + lgamma(gamma[di][kk]);
        }
    }
    for &(i, j) in obs_links {
        let mut z = nu;
        for kk in 0..k {
            z += eta[kk] * phi_bar[i][kk] * phi_bar[j][kk];
        }
        acc += match link {
            Link::Exponential => z,
            Link::Logistic => -softplus(-z), // log σ(z)
        };
    }
    acc
}

/// Cold-start: infer `φ̄` for a document from its words only (no links). Mirrors
/// the LDA E-step, used for link prediction on unseen documents.
pub fn infer_phi_bar(log_beta: &[Vec<f64>], alpha: f64, doc: &[u32], iters: usize) -> Vec<f64> {
    let k = log_beta.len();
    let bag = to_bag(doc);
    if bag.is_empty() {
        return vec![1.0 / k as f64; k];
    }
    let n: f64 = bag.iter().map(|&(_, c)| c).sum();
    let mut gamma = vec![alpha + n / k as f64; k];
    let mut phi = vec![vec![1.0 / k as f64; k]; bag.len()];
    for _ in 0..iters {
        let gsum: f64 = gamma.iter().sum();
        let dsum = digamma(gsum);
        let elogtheta: Vec<f64> = gamma.iter().map(|&g| digamma(g) - dsum).collect();
        for (i, &(word, _)) in bag.iter().enumerate() {
            let mut mx = f64::NEG_INFINITY;
            for kk in 0..k {
                let lp = log_beta[kk][word] + elogtheta[kk];
                phi[i][kk] = lp;
                if lp > mx {
                    mx = lp;
                }
            }
            let mut z = 0.0;
            for kk in 0..k {
                let e = (phi[i][kk] - mx).exp();
                phi[i][kk] = e;
                z += e;
            }
            for kk in 0..k {
                phi[i][kk] /= z;
            }
        }
        for kk in 0..k {
            let mut acc = alpha;
            for (i, &(_, c)) in bag.iter().enumerate() {
                acc += c * phi[i][kk];
            }
            gamma[kk] = acc;
        }
    }
    let mut out = vec![0.0; k];
    for kk in 0..k {
        let mut acc = 0.0;
        for (i, &(_, c)) in bag.iter().enumerate() {
            acc += c * phi[i][kk];
        }
        out[kk] = acc / n;
    }
    out
}

/// Plug-in link probability `ψ(φ̄_a ∘ φ̄_b)` between two topic-assignment means.
pub fn link_probability(link: Link, eta: &[f64], nu: f64, a: &[f64], b: &[f64]) -> f64 {
    let mut z = nu;
    for kk in 0..eta.len() {
        z += eta[kk] * a[kk] * b[kk];
    }
    match link {
        Link::Exponential => z.exp().min(1.0),
        Link::Logistic => sigmoid(z),
    }
}

impl RTMModel {
    /// `log_beta` reconstructed from the stored topic-word matrix.
    pub fn log_beta(&self) -> Vec<Vec<f64>> {
        self.topic_word
            .iter()
            .map(|r| r.iter().map(|&x| x.max(1e-300).ln()).collect())
            .collect()
    }
}

impl Estimator for RTMModel {
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
        ModelFamily::None_
    }
}

// A thin wrapper matching the scaffold's `fit(corpus, ...)` shape is intentionally
// omitted: RTM needs the link graph, so the binding calls `fit_rtm` directly.
#[allow(dead_code)]
pub fn fit<R: Rng>(_corpus: &Corpus, _num_topics: usize, _iters: usize, _rng: &mut R) -> RTMModel {
    unreachable!("RTM is fit via fit_rtm(docs, edges, params, rng)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Planted corpus: K blocks of `block` words, docs drawn from one block, links
    /// dense within a group and sparse across. Returns (docs, edges, groups, V).
    fn planted(
        rng: &mut ChaCha8Rng,
        d: usize,
        k: usize,
        block: usize,
        doclen: usize,
    ) -> (Vec<Vec<u32>>, Vec<(usize, usize)>, Vec<usize>, usize) {
        let v = k * block;
        let groups: Vec<usize> = (0..d).map(|i| i % k).collect();
        let mut docs = Vec::with_capacity(d);
        for &g in &groups {
            let mut doc = Vec::with_capacity(doclen);
            for _ in 0..doclen {
                if rng.gen::<f64>() < 0.15 {
                    doc.push(rng.gen_range(0..v) as u32);
                } else {
                    doc.push(rng.gen_range(g * block..(g + 1) * block) as u32);
                }
            }
            docs.push(doc);
        }
        let mut edges = Vec::new();
        for i in 0..d {
            for j in (i + 1)..d {
                let p = if groups[i] == groups[j] { 0.4 } else { 0.01 };
                if rng.gen::<f64>() < p {
                    edges.push((i, j));
                }
            }
        }
        (docs, edges, groups, v)
    }

    fn params(k: usize, v: usize, link: Link) -> RtmParams {
        RtmParams {
            num_topics: k,
            num_types: v,
            alpha: 0.5,
            link,
            rho: None,
            negative_ratio: 1.0,
            ridge: 0.0,
            em_iters: 40,
            e_sweeps: 3,
            e_inner: 5,
            var_tol: 1e-4,
            convergence_tol: 1e-5,
        }
    }

    #[test]
    fn rtm_recovers_planted_topics() {
        for link in [Link::Logistic, Link::Exponential] {
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            let (docs, edges, groups, v) = planted(&mut rng, 60, 3, 6, 40);
            let mut frng = ChaCha8Rng::seed_from_u64(0);
            let m = fit_rtm(&docs, &edges, &params(3, v, link), &mut frng);
            // each topic concentrates on a distinct 6-word block
            let owned: Vec<usize> = (0..3)
                .map(|kk| {
                    (0..3)
                        .max_by(|&a, &b| {
                            let sa: f64 = (a * 6..(a + 1) * 6).map(|w| m.topic_word[kk][w]).sum();
                            let sb: f64 = (b * 6..(b + 1) * 6).map(|w| m.topic_word[kk][w]).sum();
                            sa.partial_cmp(&sb).unwrap()
                        })
                        .unwrap()
                })
                .collect();
            let distinct: std::collections::HashSet<_> = owned.iter().collect();
            assert_eq!(
                distinct.len(),
                3,
                "{link:?}: topics not distinct: {owned:?}"
            );
            // objective rises
            assert!(
                m.fit_history.last().unwrap().1 >= m.fit_history.first().unwrap().1,
                "{link:?}: objective did not rise"
            );
            // link prediction separates in-group from cross-group pairs
            let (mut same, mut diff, mut ns, mut nd_) = (0.0, 0.0, 0.0, 0.0);
            for i in (0..60).step_by(2) {
                for j in ((i + 1)..60).step_by(7) {
                    let p = link_probability(link, &m.eta, m.nu, &m.phi_bar[i], &m.phi_bar[j]);
                    if groups[i] == groups[j] {
                        same += p;
                        ns += 1.0;
                    } else {
                        diff += p;
                        nd_ += 1.0;
                    }
                }
            }
            assert!(
                same / ns > diff / nd_,
                "{link:?}: link score gap not positive ({} vs {})",
                same / ns,
                diff / nd_
            );
        }
    }

    #[test]
    fn rtm_is_deterministic() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (docs, edges, _g, v) = planted(&mut rng, 30, 2, 5, 30);
        let fit = || {
            let mut r = ChaCha8Rng::seed_from_u64(3);
            fit_rtm(&docs, &edges, &params(2, v, Link::Logistic), &mut r)
        };
        let a = fit();
        let b = fit();
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.eta, b.eta);
        assert_eq!(a.phi_bar, b.phi_bar);
        // a different seed gives a different fit (so the test can't pass trivially)
        let mut r2 = ChaCha8Rng::seed_from_u64(99);
        let c = fit_rtm(&docs, &edges, &params(2, v, Link::Logistic), &mut r2);
        assert_ne!(a.topic_word, c.topic_word);
    }

    #[test]
    fn rtm_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (docs, edges, _g, v) = planted(&mut rng, 20, 2, 5, 25);
        let mut frng = ChaCha8Rng::seed_from_u64(0);
        let m = fit_rtm(&docs, &edges, &params(2, v, Link::Logistic), &mut frng);
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }
}
