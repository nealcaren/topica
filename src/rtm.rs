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
//! The variational path is validated against a standalone NumPy implementation of
//! the same equations (`parity/rtm_reference.py`); the R `lda` package's `rtm.em`
//! is a *collapsed Gibbs* sampler, so for the variational fit it is only a
//! directional baseline (`parity/rtm_compare.py`).
//!
//! [`fit_rtm_gibbs`] adds that collapsed-Gibbs backend (matching R lda's
//! `rtm.em` / `rtm.collapsed.gibbs.sampler`), so R lda becomes an *authoritative*
//! same-algorithm oracle for it (`parity/rtm_gibbs_gold.py`).
//!
//! Pure Rust, no PyO3. Fitted state stores matrices as `Vec<Vec<f64>>`.

use crate::estimator::{Estimator, ModelFamily};
use crate::mathfun::log_gamma as lgamma;
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
    /// Topic-word Dirichlet smoothing (R lda's `eta`). Used by the collapsed-Gibbs
    /// backend; the variational backend derives its own smoothed β init.
    pub beta: f64,
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
pub fn fit_rtm<R: Rng, F: FnMut(usize, usize) -> bool>(
    docs: &[Vec<u32>],
    edges: &[(usize, usize)],
    p: &RtmParams,
    mut on_progress: F,
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
            let _ = on_progress(it + 1, it + 1); // snap bar to 100% (#786)
            break;
        }
        prev_obj = obj;
        if !on_progress(it + 1, p.em_iters) {
            break;
        }
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

/// The M-step of R lda's `rtm.em` (`estimate.params`): re-estimate the per-topic
/// link coefficient β from the current topic counts. With smoothed proportions
/// `z̄_d = (n_d + α) / Σ(n_d + α)`, `p_k = Σ_{(d,d')∈links} z̄_{d,k} z̄_{d',k}`
/// (over directed link endpoints), regularized by `λ α² · #pairs` pseudo-negatives,
/// and `β_k = ln( p_k / (p_k + λ α² · #pairs) )`. Because that ratio is in (0, 1),
/// β is negative — the reference's known behaviour (match the algorithm, not the
/// sign quirk). `lambda` defaults to R's link density `Σdegree / #pairs`.
fn rtm_gibbs_estimate_beta(
    ndk: &[Vec<i64>],
    adj: &[Vec<usize>],
    alpha: f64,
    lambda: f64,
    k: usize,
) -> Vec<f64> {
    let d = ndk.len();
    // Smoothed per-document topic proportions z̄_d.
    let zbar: Vec<Vec<f64>> = ndk
        .iter()
        .map(|row| {
            let denom: f64 = row.iter().map(|&c| c as f64 + alpha).sum();
            row.iter().map(|&c| (c as f64 + alpha) / denom).collect()
        })
        .collect();
    // p_k over every directed link endpoint (d -> d'); z̄∘z̄ is symmetric, so an
    // undirected edge contributes both directions, matching R's `unlist(links)`.
    let mut p = vec![0.0f64; k];
    for (di, nbrs) in adj.iter().enumerate() {
        for &dj in nbrs {
            for kk in 0..k {
                p[kk] += zbar[di][kk] * zbar[dj][kk];
            }
        }
    }
    let num_pairs = if d >= 2 {
        d as f64 * (d as f64 - 1.0) / 2.0
    } else {
        1.0
    };
    let reg = lambda * alpha * alpha * num_pairs;
    p.iter()
        .map(|&pk| {
            // R computes `log(p_k / (p_k + reg))`. With no links at all (reg = 0 and
            // p_k = 0) that is `log(0/0) = log(NaN) = NaN` in R; report an honest
            // β_k = 0 (no link signal → an inert exp(0) = 1 factor, i.e. plain LDA)
            // rather than R's NaN or a −690 sentinel. A topic with links but zero
            // co-occurrence (p_k = 0, reg > 0) keeps R's `log(0) → −∞`, floored.
            if pk + reg <= 0.0 {
                0.0
            } else {
                (pk / (pk + reg)).max(1e-300).ln()
            }
        })
        .collect()
}

/// Fit RTM by **collapsed Gibbs**, faithfully matching R lda's `rtm.em` (which
/// wraps the dedicated `rtm()` C sampler / `rtm.collapsed.gibbs.sampler`). This is
/// the same-algorithm counterpart to [`fit_rtm`]'s variational EM, so R lda is an
/// authoritative same-algorithm oracle (distributional, not bit-exact — R uses a
/// Mersenne-Twister stream topica cannot reproduce).
///
/// Mechanics ported exactly from R's `src/gibbs.c` `rtm()`:
/// - **Restart per M-step.** `rtm.em` calls the sampler fresh in each of its
///   `p.em_iters` M-steps, carrying only the link coefficient β; every count table
///   is zeroed and re-seeded. Between M-steps β is re-estimated from the final
///   counts by [`rtm_gibbs_estimate_beta`] (`estimate.params`).
/// - **`p.e_sweeps` sweeps per sampler call, sweep 0 is a uniform-random init.** On
///   sweep 0 every cell draws a topic uniformly (all conditional weights = 1, no
///   link/LDA factor); sweeps `1..e_sweeps-1` are the real RTM Gibbs updates.
/// - **Per-`(word, count)`-cell block sampling.** Each document is a set of
///   `(word, count)` cells; the sampler removes the *whole* count, draws one topic,
///   and adds the whole count back — it does not resample tokens individually.
/// - **Per-cell conditional** = `link_probs[k] · (n_{d,k}+α)(n_{k,w}+η)/(n_k+Vη)`
///   with `link_probs[k] = exp(β_k · Σ_{d'∈links(d)} n_{d',k}/(len_d·len_{d'}))`
///   computed once per document per sweep from neighbours' raw topic counts and the
///   fixed document token lengths. The exponential link's coefficient is stored as
///   `eta` (with `nu = 0`, matching R's intercept-free link).
pub fn fit_rtm_gibbs<R: Rng, F: FnMut(usize, usize) -> bool>(
    docs: &[Vec<u32>],
    edges: &[(usize, usize)],
    p: &RtmParams,
    mut on_progress: F,
    rng: &mut R,
) -> RTMModel {
    use std::collections::BTreeMap;
    let k = p.num_topics;
    let v = p.num_types;
    let d = docs.len();
    let alpha = p.alpha;
    let beta_tw = p.beta; // topic-word smoothing (R lda's `eta`)
    let vbeta = v as f64 * beta_tw;
    let (adj, obs) = build_adjacency(edges, d);
    // R's `document_lengths`: fixed total token count per doc (sum of cell counts).
    let len: Vec<f64> = docs.iter().map(|doc| doc.len().max(1) as f64).collect();

    // Each document as `(word, count)` cells in sorted word-id order (R lexicalize's
    // 2×W matrix). Block sampling assigns one topic per cell.
    let cells: Vec<Vec<(usize, i64)>> = docs
        .iter()
        .map(|doc| {
            let mut m: BTreeMap<usize, i64> = BTreeMap::new();
            for &w in doc {
                *m.entry(w as usize).or_insert(0) += 1;
            }
            m.into_iter().collect()
        })
        .collect();

    // R's default λ = Σdegree / #pairs = 2|E| / (D(D-1)/2).
    let sum_deg: f64 = adj.iter().map(|n| n.len() as f64).sum();
    let num_pairs = if d >= 2 {
        d as f64 * (d as f64 - 1.0) / 2.0
    } else {
        1.0
    };
    let lambda = if obs.is_empty() {
        0.0
    } else {
        sum_deg / num_pairs
    };

    // Count tables (re-zeroed every M-step) and per-cell topic assignments.
    let mut ndk = vec![vec![0i64; k]; d]; // document_sums (Kᵀ per doc)
    let mut nkw = vec![vec![0i64; v]; k]; // topics
    let mut nk = vec![0i64; k]; // topic_sums
    let mut z: Vec<Vec<usize>> = cells.iter().map(|c| vec![0usize; c.len()]).collect();

    // Link coefficient β, initialised as R does (`initial.beta = rep(3, K)`); carried
    // across M-steps and re-estimated after each.
    let mut link_beta = vec![3.0f64; k];

    let mut scores = vec![0.0f64; k];
    let mut nb_sum = vec![0.0f64; k]; // Σ_{d'~d} n_{d',k} / len_{d'}
    let mut link_probs = vec![1.0f64; k];
    let mut history: Vec<(usize, f64)> = Vec::new();

    let m_steps = p.em_iters.max(1);
    let n_sweeps = p.e_sweeps.max(1); // R's num.e.iterations (sweep 0 = init)
    for m in 0..m_steps {
        // Fresh restart: zero every count table (R's `rtm()` reallocates them).
        for row in ndk.iter_mut() {
            row.iter_mut().for_each(|c| *c = 0);
        }
        for row in nkw.iter_mut() {
            row.iter_mut().for_each(|c| *c = 0);
        }
        nk.iter_mut().for_each(|c| *c = 0);

        for sweep in 0..n_sweeps {
            let init_sweep = sweep == 0;
            for di in 0..d {
                // Link factor once per doc per sweep (skipped on the init sweep, as R
                // does — `if (ii > 0)`).
                if !init_sweep && !adj[di].is_empty() {
                    for kk in 0..k {
                        nb_sum[kk] = 0.0;
                    }
                    for &dj in &adj[di] {
                        let inv = 1.0 / len[dj];
                        for kk in 0..k {
                            nb_sum[kk] += ndk[dj][kk] as f64 * inv;
                        }
                    }
                    let inv_di = 1.0 / len[di];
                    for kk in 0..k {
                        link_probs[kk] = (link_beta[kk] * nb_sum[kk] * inv_di).exp();
                    }
                } else {
                    for kk in 0..k {
                        link_probs[kk] = 1.0;
                    }
                }
                for (ci, &(w, count)) in cells[di].iter().enumerate() {
                    // Remove the whole cell count from its current topic (skipped on
                    // the init sweep, where nothing was assigned yet).
                    if !init_sweep {
                        let old = z[di][ci];
                        ndk[di][old] -= count;
                        nkw[old][w] -= count;
                        nk[old] -= count;
                    }

                    let mut total = 0.0;
                    for kk in 0..k {
                        // Init sweep: uniform draw (all weights 1). Otherwise the
                        // link-scaled LDA collapsed conditional.
                        let s = if init_sweep {
                            1.0
                        } else {
                            link_probs[kk]
                                * (ndk[di][kk] as f64 + alpha)
                                * (nkw[kk][w] as f64 + beta_tw)
                                / (nk[kk] as f64 + vbeta)
                        };
                        scores[kk] = s;
                        total += s;
                    }
                    let mut r = rng.gen::<f64>() * total;
                    let mut new_t = k - 1;
                    for kk in 0..k {
                        r -= scores[kk];
                        if r <= 0.0 {
                            new_t = kk;
                            break;
                        }
                    }
                    z[di][ci] = new_t;
                    ndk[di][new_t] += count;
                    nkw[new_t][w] += count;
                    nk[new_t] += count;
                }
            }
        }
        // M-step: re-estimate the link coefficient from the final counts.
        link_beta = rtm_gibbs_estimate_beta(&ndk, &adj, alpha, lambda, k);
        history.push((m + 1, link_beta.iter().sum::<f64>()));
        if !on_progress(m + 1, m_steps) {
            break;
        }
    }

    // Fitted surface (same shape as the variational backend).
    let topic_word: Vec<Vec<f64>> = (0..k)
        .map(|kk| {
            let denom = nk[kk] as f64 + vbeta;
            (0..v)
                .map(|w| (nkw[kk][w] as f64 + beta_tw) / denom)
                .collect()
        })
        .collect();
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let denom: f64 = ndk[di].iter().map(|&c| c as f64 + alpha).sum();
            (0..k)
                .map(|kk| (ndk[di][kk] as f64 + alpha) / denom)
                .collect()
        })
        .collect();
    // phi_bar = the *empirical* mean topic assignment z̄_d = n_{d,k} / N_d (the
    // link quantity), unsmoothed — distinct from the α-smoothed `doc_topic` above.
    // An empty document has no assignments, so its row is left at zero.
    let phi_bar: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let nd: f64 = ndk[di].iter().map(|&c| c as f64).sum();
            if nd > 0.0 {
                (0..k).map(|kk| ndk[di][kk] as f64 / nd).collect()
            } else {
                vec![0.0; k]
            }
        })
        .collect();

    RTMModel {
        num_topics: k,
        topic_word,
        doc_topic,
        phi_bar,
        eta: link_beta,
        nu: 0.0,
        // R lda's collapsed-Gibbs sampler is exponential-link only; store that so
        // link prediction uses exp(·), not σ(·), on the (negative) coefficients.
        link: Link::Exponential,
        fit_history: history,
        converged: false,
    }
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

// RTM needs the link graph, so there is no `fit(corpus, ...)` wrapper matching the
// scaffold shape: the binding calls `fit_rtm(docs, edges, params, rng)` directly.

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
            beta: 0.1,
            link,
            rho: None,
            negative_ratio: 1.0,
            ridge: 1.0,
            em_iters: 40,
            e_sweeps: 3,
            e_inner: 5,
            var_tol: 1e-4,
            convergence_tol: 1e-5,
        }
    }

    /// Stable collapsed-Gibbs config (#424): a converged E-sweep budget (R restarts
    /// each M-step, so quality comes from within-sweep mixing, not accumulation) and
    /// only a few M-steps, before R lda's `estimate.params` β estimate runs away
    /// negative and degenerates the fit. Mirrors `parity/rtm_gibbs_gold.py`.
    fn gibbs_params(k: usize, v: usize) -> RtmParams {
        RtmParams {
            em_iters: 5,
            e_sweeps: 60,
            ..params(k, v, Link::Exponential)
        }
    }

    #[test]
    fn rtm_recovers_planted_topics() {
        for link in [Link::Logistic, Link::Exponential] {
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            let (docs, edges, groups, v) = planted(&mut rng, 60, 3, 6, 40);
            let mut frng = ChaCha8Rng::seed_from_u64(0);
            let m = fit_rtm(&docs, &edges, &params(3, v, link), |_, _| true, &mut frng);
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
            fit_rtm(
                &docs,
                &edges,
                &params(2, v, Link::Logistic),
                |_, _| true,
                &mut r,
            )
        };
        let a = fit();
        let b = fit();
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.eta, b.eta);
        assert_eq!(a.phi_bar, b.phi_bar);
        // a different seed gives a different fit (so the test can't pass trivially)
        let mut r2 = ChaCha8Rng::seed_from_u64(99);
        let c = fit_rtm(
            &docs,
            &edges,
            &params(2, v, Link::Logistic),
            |_, _| true,
            &mut r2,
        );
        assert_ne!(a.topic_word, c.topic_word);
    }

    #[test]
    fn rtm_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (docs, edges, _g, v) = planted(&mut rng, 20, 2, 5, 25);
        let mut frng = ChaCha8Rng::seed_from_u64(0);
        let m = fit_rtm(
            &docs,
            &edges,
            &params(2, v, Link::Logistic),
            |_, _| true,
            &mut frng,
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    // --- collapsed-Gibbs backend (#424) ---------------------------------------

    #[test]
    fn rtm_gibbs_recovers_planted_topics() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, edges, _groups, v) = planted(&mut rng, 60, 3, 6, 40);
        let mut frng = ChaCha8Rng::seed_from_u64(0);
        let m = fit_rtm_gibbs(&docs, &edges, &gibbs_params(3, v), |_, _| true, &mut frng);

        // rows are valid simplices
        for row in &m.topic_word {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // each topic owns a distinct planted word block. (Link-prediction *quality*
        // is deliberately not asserted for the Gibbs backend: R lda's estimate.params
        // fits a negative β, so exp(β·z̄∘z̄) ranks dissimilar docs higher — see
        // rtm_gibbs_link_beta_is_negative_like_r. The Gibbs backend's prize is
        // same-algorithm *topic* parity; use inference="variational" for link scores.)
        let owned: Vec<usize> = (0..3)
            .map(|t| {
                (0..3)
                    .max_by(|&a, &b| {
                        let sa: f64 = m.topic_word[t][a * 6..(a + 1) * 6].iter().sum();
                        let sb: f64 = m.topic_word[t][b * 6..(b + 1) * 6].iter().sum();
                        sa.total_cmp(&sb)
                    })
                    .unwrap()
            })
            .collect();
        assert_eq!(
            owned.iter().collect::<std::collections::HashSet<_>>().len(),
            3,
            "topics did not partition the blocks: {owned:?}"
        );
    }

    #[test]
    fn rtm_gibbs_is_deterministic() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (docs, edges, _g, v) = planted(&mut rng, 30, 2, 5, 30);
        let fit = || {
            let mut r = ChaCha8Rng::seed_from_u64(3);
            fit_rtm_gibbs(&docs, &edges, &gibbs_params(2, v), |_, _| true, &mut r)
        };
        let a = fit();
        let b = fit();
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.eta, b.eta);
        assert_eq!(a.doc_topic, b.doc_topic);
        let mut r2 = ChaCha8Rng::seed_from_u64(99);
        let c = fit_rtm_gibbs(&docs, &edges, &gibbs_params(2, v), |_, _| true, &mut r2);
        assert_ne!(a.topic_word, c.topic_word);
    }

    #[test]
    fn rtm_gibbs_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (docs, edges, _g, v) = planted(&mut rng, 20, 2, 5, 25);
        let mut frng = ChaCha8Rng::seed_from_u64(0);
        let m = fit_rtm_gibbs(&docs, &edges, &gibbs_params(2, v), |_, _| true, &mut frng);
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    #[test]
    fn rtm_gibbs_link_beta_is_negative_like_r() {
        // R lda's estimate.params sets β_k = ln(p_k) with p_k ∈ (0,1), so the link
        // coefficient is negative even on strongly-linked data (the documented
        // reference quirk). Match it.
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let (docs, edges, _g, v) = planted(&mut rng, 60, 3, 6, 40);
        let mut frng = ChaCha8Rng::seed_from_u64(0);
        let m = fit_rtm_gibbs(&docs, &edges, &gibbs_params(3, v), |_, _| true, &mut frng);
        assert!(
            m.eta.iter().all(|&b| b < 0.0),
            "expected all-negative link β (R quirk), got {:?}",
            m.eta
        );
    }

    #[test]
    fn rtm_gibbs_estimate_beta_matches_r_hand_computed() {
        // The relational M-step (R lda `estimate.params`) is the RTM-specific link
        // computation; a plain-LDA fit could clear the topic-cosine gate, so this
        // pins the β math to a hand-computed R value. K=2, 3 docs, edges (0,1),(1,2)
        // symmetrized to adj[0]=[1], adj[1]=[0,2], adj[2]=[1]; α=0.1.
        //   z̄_d = (n_d + α)/(N_d + Kα), all N_d = 4 → denom 4.2.
        //   p_k = Σ_{directed (d→d')} z̄_{d,k} z̄_{d',k}  →  p = [0.6485262, 1.1247166].
        //   λ = Σdeg/#pairs = 4/3, reg = λ·α²·#pairs = 0.04.
        //   β_k = ln(p_k/(p_k+reg))  →  [-0.0598510, -0.0349467].
        let ndk = vec![vec![3i64, 1], vec![1, 3], vec![2, 2]];
        let adj = vec![vec![1usize], vec![0usize, 2], vec![1usize]];
        let lambda = 4.0 / 3.0;
        let beta = rtm_gibbs_estimate_beta(&ndk, &adj, 0.1, lambda, 2);
        assert!((beta[0] - (-0.0598510)).abs() < 1e-5, "β0 = {}", beta[0]);
        assert!((beta[1] - (-0.0349467)).abs() < 1e-5, "β1 = {}", beta[1]);
    }

    #[test]
    fn rtm_gibbs_no_links_beta_is_zero_not_sentinel() {
        // With no links, R's estimate.params computes log(0/0) = NaN; topica reports
        // an honest β = 0 (inert link) rather than a −690 floor sentinel.
        let ndk = vec![vec![3i64, 1], vec![1, 3]];
        let adj = vec![vec![], vec![]];
        let beta = rtm_gibbs_estimate_beta(&ndk, &adj, 0.1, 0.0, 2);
        assert_eq!(
            beta,
            vec![0.0, 0.0],
            "no-link β must be an honest 0, got {beta:?}"
        );
    }
}
