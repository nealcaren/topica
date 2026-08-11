//! Wordshoal: Lauderdale & Herzog's (2016) two-stage one-dimensional scaling of
//! actor (speaker) positions from texts partitioned into externally-known
//! **domains** (debates). The multi-domain extension of [`crate::wordfish`], and a
//! sibling ideal-point model — no topics, no embeddings.
//!
//! Stage 1: run Wordfish independently on each domain, treating each document
//! (speech) as its own unit, to get a within-domain position `psi[s]` per document
//! (Wordfish's `theta`, standardized and default-oriented within that domain — the
//! within-domain orientation is arbitrary and absorbed by the stage-2 loading).
//!
//! Stage 2: a one-dimensional linear factor model over the stacked within-domain
//! positions, `psi[s] = alpha[j(s)] + beta[j(s)] * theta[i(s)] + N(0, tau[i(s)]^-1)`,
//! fit by the reference's conditional-ML coordinate ascent. `theta[i]` is the
//! actor's cross-domain position, `beta[j]` the domain loading (which absorbs each
//! domain's arbitrary sign/scale), `alpha[j]` a domain intercept, `tau[i]` an actor
//! precision.
//!
//! Faithful to `kbenoit/wordshoal` (GPL-3, read for algorithmic understanding only;
//! implemented from the paper's equations). The reference's stage-2 updates are a
//! conditional-ML coordinate ascent, NOT a strict EM: the actor update deliberately
//! omits `tau_i` and the precision update uses a partial residual, so the
//! log-posterior can wiggle down a hair at convergence (that terminal wiggle IS the
//! `while (lp - lastlp) > tol` stop signal). Empirically this is immaterial — the
//! reference and a tau-weighted "correction" agree on `theta` to |r| = 0.9995 — so
//! we replicate the published behavior exactly rather than "fix" it. The fit is
//! deterministic (no RNG in either stage): a given input reproduces bit-for-bit.

use crate::mathfun::log_gamma;
use crate::wordfish::fit_wordfish;
use std::f64::consts::PI;

/// A fitted Wordshoal model. `theta` are the actor positions (prior-identified,
/// oriented); `alpha`/`beta` the per-domain intercept/loading; `tau` the per-actor
/// precision; `position_se` the actor-position standard errors.
pub struct WordshoalModel {
    pub num_authors: usize,
    pub num_domains: usize,
    pub theta: Vec<f64>,
    pub tau: Vec<f64>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub position_se: Vec<f64>,
    /// Stage-2 log-posterior at convergence (includes the four log-priors — it is
    /// the reference's convergence quantity, a log-posterior not a bare loglik).
    pub log_posterior: f64,
    pub lp_history: Vec<f64>,
    pub converged: bool,
    pub iters_run: usize,
    /// Stage-1 per-domain word discriminations, for `word_scores(domain)`.
    /// `domain_word_ids[j][k]` is a global word id; `domain_word_beta[j][k]` its
    /// within-domain Wordfish discrimination. A domain that failed to scale has
    /// empty rows.
    pub domain_word_ids: Vec<Vec<u32>>,
    pub domain_word_beta: Vec<Vec<f64>>,
    /// Stage-1 within-domain position of each document (length S), in input order.
    pub psi: Vec<f64>,
    /// Number of connected components of the speaker-domain bipartite graph
    /// (edges only through domains that successfully scaled). > 1 means the scale is
    /// not identified across components (the binding warns).
    pub num_components: usize,
    /// Component label (0-based, in ascending first-appearance order) of each actor,
    /// aligned to `theta`. Actors in different components are on non-comparable
    /// scales; this lets a caller segregate them instead of comparing across.
    pub author_components: Vec<usize>,
}

/// Fit Wordshoal. `docs` are `S` documents as sparse global-word-id counts;
/// `speaker[s]` / `domain[s]` are the actor / domain index of document `s`
/// (`0..num_authors`, `0..num_domains`), assigned by the binding in sorted-label
/// order so the deterministic `linspace(-2, 2)` init matches the reference's
/// alphabetical R factor levels. `anchors` are `(actor_index, target)` pairs used
/// only to orient the sign of the axis.
///
/// Stage-1 Wordfish priors are hardwired to the quanteda defaults
/// (`beta_prior_sd = 3`, `theta_prior_sd = 1`), independent of the stage-2 priors,
/// which the reference's `textmodel_wordfish` call also leaves at its defaults.
#[allow(clippy::too_many_arguments)]
pub fn fit_wordshoal(
    docs: &[Vec<(u32, f64)>],
    num_types: usize,
    speaker: &[usize],
    num_authors: usize,
    domain: &[usize],
    num_domains: usize,
    anchors: &[(usize, f64)],
    stage2_iters_cap: usize,
    convergence_tol: f64,
    theta_prior_sd: f64,
    loading_prior_sd: f64,
    intercept_prior_sd: f64,
    tau_prior: f64,
) -> WordshoalModel {
    let s_total = docs.len();
    let n = num_authors;
    let m = num_domains;

    // ---- Stage 1: per-domain Wordfish ------------------------------------
    // psi[s] is document s's within-domain position; zero-filled for a domain that
    // fails to scale (mirrors the reference's `replace(psi, is.na, 0)`).
    let mut psi = vec![0.0f64; s_total];
    let mut domain_word_ids: Vec<Vec<u32>> = vec![Vec::new(); m];
    let mut domain_word_beta: Vec<Vec<f64>> = vec![Vec::new(); m];

    for (j, (ids, betas)) in domain_word_ids
        .iter_mut()
        .zip(domain_word_beta.iter_mut())
        .enumerate()
    {
        // Documents in this domain, in input order.
        let doc_idx: Vec<usize> = (0..s_total).filter(|&s| domain[s] == j).collect();
        if doc_idx.len() < 2 {
            // Guarded by the binding (<2 docs is an error); defensive zero-fill.
            continue;
        }
        // Local vocabulary: global word ids present in this domain, ascending.
        let mut present: Vec<u32> = Vec::new();
        {
            let mut seen = vec![false; num_types];
            for &s in &doc_idx {
                for &(w, _) in &docs[s] {
                    if !seen[w as usize] {
                        seen[w as usize] = true;
                        present.push(w);
                    }
                }
            }
            present.sort_unstable();
        }
        if present.len() < 2 {
            continue; // cannot scale a <2-word domain; zero-fill psi
        }
        let local_of: std::collections::HashMap<u32, u32> = present
            .iter()
            .enumerate()
            .map(|(k, &w)| (w, k as u32))
            .collect();
        // Local per-document counts (each document is a Wordfish "author").
        let local_counts: Vec<Vec<(u32, f64)>> = doc_idx
            .iter()
            .map(|&s| {
                let mut v: Vec<(u32, f64)> =
                    docs[s].iter().map(|&(w, c)| (local_of[&w], c)).collect();
                v.sort_by_key(|&(w, _)| w);
                v
            })
            .collect();
        // Wordfish with hardwired quanteda-default priors; no anchors (arbitrary
        // within-domain orientation, absorbed by the stage-2 loading). The reference
        // leaves stage-1 Wordfish tolerance-bound (`textmodel_wordfish(tol=c(tol,
        // 1e-8))`), so we pass a high iteration cap and let `convergence_tol` stop
        // it, rather than a fixed count that could truncate a slow domain.
        let wf = fit_wordfish(
            &local_counts,
            present.len(),
            &[],
            1000,
            convergence_tol,
            3.0,
            1.0,
        );
        for (k, &s) in doc_idx.iter().enumerate() {
            let t = wf.theta[k];
            psi[s] = if t.is_finite() { t } else { 0.0 };
        }
        *ids = present;
        *betas = wf.beta;
    }

    // ---- Stage 2: cross-domain linear factor model -----------------------
    let inv_theta_var = 1.0 / (theta_prior_sd * theta_prior_sd); // priortheta^-2 ridge
                                                                 // Domain-parameter ridge P = diag(1/prioralpha^2, 1/priorbeta^2).
    let p_alpha = 1.0 / (intercept_prior_sd * intercept_prior_sd);
    let p_beta = 1.0 / (loading_prior_sd * loading_prior_sd);

    // Deterministic init: theta evenly spaced (index order == sorted-label order).
    let mut alpha = vec![0.0f64; m];
    let mut beta = vec![0.0f64; m];
    let mut theta: Vec<f64> = (0..n)
        .map(|i| {
            if n <= 1 {
                0.0
            } else {
                -2.0 + 4.0 * (i as f64) / ((n - 1) as f64)
            }
        })
        .collect();
    let mut tau = vec![1.0f64; n];

    // Documents grouped by domain and by speaker (fixed order for determinism).
    let mut by_domain: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut by_speaker: Vec<Vec<usize>> = vec![Vec::new(); n];
    for s in 0..s_total {
        by_domain[domain[s]].push(s);
        by_speaker[speaker[s]].push(s);
    }

    let lp_of = |alpha: &[f64], beta: &[f64], theta: &[f64], tau: &[f64]| -> f64 {
        let mut lp = 0.0;
        for &a in alpha {
            lp += dnorm_log(a, 0.0, intercept_prior_sd);
        }
        for &b in beta {
            lp += dnorm_log(b, 0.0, loading_prior_sd);
        }
        for &t in theta {
            lp += dnorm_log(t, 0.0, theta_prior_sd);
        }
        for &tt in tau {
            lp += dgamma_log(tt, tau_prior, tau_prior);
        }
        for s in 0..s_total {
            let mean = alpha[domain[s]] + beta[domain[s]] * theta[speaker[s]];
            let sd = tau[speaker[s]].powf(-0.5);
            lp += dnorm_log(psi[s], mean, sd);
        }
        lp
    };

    let mut lp_history = Vec::with_capacity(stage2_iters_cap + 1);
    let mut lp = lp_of(&alpha, &beta, &theta, &tau);
    lp_history.push(lp);
    let mut last_lp = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iters_run = 0;

    while (lp - last_lp) > convergence_tol.abs() {
        if iters_run >= stage2_iters_cap {
            break;
        }
        iters_run += 1;

        // Domain update: WLS of psi on [1, theta] weighted by tau, ridged by P.
        for j in 0..m {
            let (mut s11, mut s1t, mut stt) = (p_alpha, 0.0, p_beta); // X'WX + P
            let (mut r1, mut rt) = (0.0, 0.0); // X'W Y
            for &s in &by_domain[j] {
                let w = tau[speaker[s]];
                let t = theta[speaker[s]];
                let y = psi[s];
                s11 += w;
                s1t += w * t;
                stt += w * t * t;
                r1 += w * y;
                rt += w * t * y;
            }
            let det = s11 * stt - s1t * s1t;
            if det.abs() > 1e-12 {
                alpha[j] = (stt * r1 - s1t * rt) / det;
                beta[j] = (s11 * rt - s1t * r1) / det;
            }
        }

        // Actor update: regress residual (psi - alpha) on beta (no intercept),
        // ridged by priortheta^-2. Reference deliberately omits tau_i here.
        for i in 0..n {
            let mut bb = inv_theta_var; // b'b + priortheta^-2
            let mut by = 0.0; // b'(psi - alpha)
            let mut yy = 0.0; // sum (psi - alpha)^2
            let cnt = by_speaker[i].len();
            for &s in &by_speaker[i] {
                let b = beta[domain[s]];
                let y = psi[s] - alpha[domain[s]];
                bb += b * b;
                by += b * y;
                yy += y * y;
            }
            let a_inv = 1.0 / bb; // (b'b + priortheta^-2)^-1
            let th = a_inv * by;
            theta[i] = th;
            // mu = A (b'b) theta ; A = a_inv, (b'b) = bb - inv_theta_var
            let mu = a_inv * (bb - inv_theta_var) * th;
            // Reference tau update: partial residual (psi - alpha), NOT the full
            // regression RSS — coded literally to match kbenoit/wordshoal.
            let denom = tau_prior + 0.5 * (yy - mu * inv_theta_var * mu);
            tau[i] = (tau_prior + 0.5 * cnt as f64) / denom;
        }

        last_lp = lp;
        lp = lp_of(&alpha, &beta, &theta, &tau);
        lp_history.push(lp);
        if (lp - last_lp) <= convergence_tol.abs() {
            converged = true;
        }
    }

    // Standard errors: se[i] = sqrt( (b'b + priortheta^-2)^-1 / tau_i ).
    let mut position_se = vec![f64::NAN; n];
    for i in 0..n {
        let mut bb = inv_theta_var;
        for &s in &by_speaker[i] {
            let b = beta[domain[s]];
            bb += b * b;
        }
        let var = (1.0 / bb) / tau[i];
        position_se[i] = if var > 0.0 { var.sqrt() } else { f64::NAN };
    }

    // Orientation: flip theta AND beta together (alpha/tau fixed) so
    // psi = alpha + beta*theta and the SE are invariant.
    orient(&mut theta, &mut beta, anchors);

    // Bridge speakers only through domains that actually scaled: a failed/degenerate
    // domain (empty word set, zero-filled psi, beta_j pinned near 0) gives no
    // cross-domain constraint, so it must not join two speaker groups.
    let scaled: Vec<bool> = domain_word_ids.iter().map(|ids| !ids.is_empty()).collect();
    let author_components = connected_components(speaker, domain, &scaled, n, m);
    let num_components = {
        let mut distinct = author_components.clone();
        distinct.sort_unstable();
        distinct.dedup();
        distinct.len()
    };

    WordshoalModel {
        num_authors: n,
        num_domains: m,
        theta,
        tau,
        alpha,
        beta,
        position_se,
        log_posterior: lp,
        lp_history,
        converged,
        iters_run,
        domain_word_ids,
        domain_word_beta,
        psi,
        num_components,
        author_components,
    }
}

/// Orient the axis: with anchors, make the anchored positions correlate positively
/// with their targets; with no anchors, default to `theta[0] < theta[1]` (the first
/// two actors in sorted order), mirroring Wordfish / quanteda `dir = c(1, 2)`.
/// Flips both `theta` (actors) and `beta` (domain loadings).
fn orient(theta: &mut [f64], beta: &mut [f64], anchors: &[(usize, f64)]) {
    let flip = |theta: &mut [f64], beta: &mut [f64]| {
        for t in theta.iter_mut() {
            *t = -*t;
        }
        for b in beta.iter_mut() {
            *b = -*b;
        }
    };
    if anchors.is_empty() {
        if theta.len() >= 2 && theta[0] > theta[1] {
            flip(theta, beta);
        }
        return;
    }
    if anchors.len() < 2 {
        if let Some(&(i, target)) = anchors.first() {
            if theta[i] * target < 0.0 {
                flip(theta, beta);
            }
        }
        return;
    }
    let k = anchors.len() as f64;
    let mt = anchors.iter().map(|&(i, _)| theta[i]).sum::<f64>() / k;
    let mv = anchors.iter().map(|&(_, t)| t).sum::<f64>() / k;
    let cov: f64 = anchors
        .iter()
        .map(|&(i, t)| (theta[i] - mt) * (t - mv))
        .sum();
    if cov < 0.0 {
        flip(theta, beta);
    }
}

/// Connected components of the bipartite speaker-domain graph (union-find over
/// `N + M` nodes: speakers `0..N`, domains `N..N+M`), bridging speakers **only
/// through domains that successfully scaled** (`scaled[j]`) — a failed domain
/// carries no cross-domain constraint, so it must not join two speaker groups.
/// Returns a component label per actor (0-based, in ascending first-appearance
/// order over actor index); actors sharing a label are on one comparable scale.
fn connected_components(
    speaker: &[usize],
    domain: &[usize],
    scaled: &[bool],
    n: usize,
    m: usize,
) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n + m).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != c {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }
    for (&i, &j) in speaker.iter().zip(domain.iter()) {
        if !scaled[j] {
            continue; // a failed domain is not an edge
        }
        let a = find(&mut parent, i);
        let b = find(&mut parent, n + j);
        if a != b {
            parent[a] = b;
        }
    }
    // Label each actor by its root, over actor nodes only (domain nodes must not
    // count as components — an isolated failed domain is not a scale). Relabel to
    // dense 0-based ids in ascending actor-index order for a stable, comparable id.
    let mut label_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut out = vec![0usize; n];
    for (i, o) in out.iter_mut().enumerate() {
        let r = find(&mut parent, i);
        let next = label_of.len();
        *o = *label_of.entry(r).or_insert(next);
    }
    out
}

/// log N(x | mean, sd).
fn dnorm_log(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * (2.0 * PI).ln() - sd.ln() - 0.5 * z * z
}

/// log Gamma(x | shape, rate) density (rate parameterization, matching R's
/// `dgamma(x, shape, rate)`).
fn dgamma_log(x: f64, shape: f64, rate: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape * rate.ln() - log_gamma(shape) + (shape - 1.0) * x.ln() - rate * x
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
        for (&a, &b) in x.iter().zip(y) {
            cov += (a - mx) * (b - my);
            vx += (a - mx) * (a - mx);
            vy += (b - my) * (b - my);
        }
        cov / (vx.sqrt() * vy.sqrt())
    }

    fn poisson(rng: &mut ChaCha8Rng, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        let l = (-lambda).exp();
        let (mut k, mut p) = (0u32, 1.0f64);
        loop {
            k += 1;
            p *= rng.gen::<f64>();
            if p <= l {
                break;
            }
        }
        (k - 1) as f64
    }

    /// Build a fixed-seed two-stage corpus: actor positions theta_i, domain loadings
    /// beta_j / intercepts alpha_j, per-domain Wordfish words whose per-document
    /// position is z = alpha_j + beta_j theta_i.
    #[allow(clippy::type_complexity)]
    fn planted(
        seed: u64,
    ) -> (
        Vec<Vec<(u32, f64)>>,
        usize,
        Vec<usize>,
        usize,
        Vec<usize>,
        usize,
        Vec<f64>,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (n, m, v) = (30usize, 10usize, 50usize);
        let theta: Vec<f64> = (0..n)
            .map(|i| -1.5 + 3.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let beta: Vec<f64> = (0..m)
            .map(|j| {
                let s = if j % 2 == 0 { 1.0 } else { -1.0 };
                s * (0.7 + 0.6 * (j as f64 / m as f64))
            })
            .collect();
        let alpha: Vec<f64> = (0..m)
            .map(|j| 0.2 * ((j as f64) - m as f64 / 2.0))
            .collect();
        let mut docs = Vec::new();
        let mut speaker = Vec::new();
        let mut domain = Vec::new();
        for j in 0..m {
            let bword: Vec<f64> = (0..v)
                .map(|w| -1.2 + 2.4 * (w as f64 / (v - 1) as f64))
                .collect();
            let pword: Vec<f64> = (0..v)
                .map(|_| (3.0 + 5.0 * rng.gen::<f64>()).ln())
                .collect();
            // every actor speaks in this domain (dense design -> connected)
            for i in 0..n {
                let z = alpha[j] + beta[j] * theta[i];
                let mut row = Vec::new();
                for w in 0..v {
                    let lam = (pword[w] + bword[w] * z).exp();
                    let c = poisson(&mut rng, lam);
                    if c > 0.0 {
                        row.push((w as u32, c));
                    }
                }
                if row.len() < 2 {
                    row = vec![(0u32, 1.0), (1u32, 1.0)];
                }
                docs.push(row);
                speaker.push(i);
                domain.push(j);
            }
        }
        (docs, v, speaker, n, domain, m, theta)
    }

    #[test]
    fn recovers_actor_positions() {
        let (docs, v, speaker, n, domain, m, theta_true) = planted(1);
        let model = fit_wordshoal(
            &docs,
            v,
            &speaker,
            n,
            &domain,
            m,
            &[],
            50,
            1e-3,
            1.0,
            0.5,
            0.5,
            1.0,
        );
        let r = pearson(&model.theta, &theta_true).abs();
        assert!(r > 0.9, "actor-position recovery r={r}");
        // default orientation: first actor below last
        assert!(model.theta[0] < model.theta[n - 1], "orientation");
        assert_eq!(model.num_components, 1, "dense design is connected");
        // SE finite and positive
        assert!(model.position_se.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn deterministic() {
        let (docs, v, speaker, n, domain, m, _) = planted(2);
        let a = fit_wordshoal(
            &docs,
            v,
            &speaker,
            n,
            &domain,
            m,
            &[],
            50,
            1e-3,
            1.0,
            0.5,
            0.5,
            1.0,
        );
        let b = fit_wordshoal(
            &docs,
            v,
            &speaker,
            n,
            &domain,
            m,
            &[],
            50,
            1e-3,
            1.0,
            0.5,
            0.5,
            1.0,
        );
        assert_eq!(a.theta, b.theta);
        assert_eq!(a.beta, b.beta);
        assert_eq!(a.tau, b.tau);
    }

    #[test]
    fn anchors_orient_sign() {
        let (docs, v, speaker, n, domain, m, _) = planted(3);
        // anchor the last actor positive, first negative
        let anchors = vec![(0usize, -1.0), (n - 1, 1.0)];
        let model = fit_wordshoal(
            &docs, v, &speaker, n, &domain, m, &anchors, 50, 1e-3, 1.0, 0.5, 0.5, 1.0,
        );
        assert!(
            model.theta[0] < model.theta[n - 1],
            "anchors did not orient"
        );
    }

    #[test]
    fn detects_disconnected_components() {
        // Two speaker-domain blocks sharing no domain -> 2 components.
        let (mut docs, v, mut speaker, n, mut domain, m, _) = planted(4);
        // Re-route: speakers 0..n/2 only ever in domains 0..m/2, others in m/2..m.
        docs.clear();
        speaker.clear();
        domain.clear();
        for j in 0..m {
            for i in 0..n {
                let same = (i < n / 2) == (j < m / 2);
                if !same {
                    continue;
                }
                docs.push(vec![(0u32, 3.0), (1u32, 2.0), (2u32, 1.0)]);
                speaker.push(i);
                domain.push(j);
            }
        }
        let model = fit_wordshoal(
            &docs,
            v,
            &speaker,
            n,
            &domain,
            m,
            &[],
            50,
            1e-3,
            1.0,
            0.5,
            0.5,
            1.0,
        );
        assert_eq!(model.num_components, 2, "should detect 2 components");
        // author_components: the two blocks carry distinct labels.
        assert_eq!(model.author_components.len(), n);
        assert_ne!(
            model.author_components[0],
            model.author_components[n - 1],
            "actors in different blocks share a component label"
        );
    }

    #[test]
    fn failed_bridge_domain_does_not_connect() {
        // Two speaker blocks bridged ONLY by a domain that fails to scale (a single
        // repeated word -> <2 local words). That domain carries no cross-domain
        // constraint, so the two blocks must remain separate components.
        let n = 8usize;
        let m = 3usize; // domain 0: block A; domain 1: block B; domain 2: the bad bridge
        let v = 5usize;
        let mut docs: Vec<Vec<(u32, f64)>> = Vec::new();
        let mut speaker: Vec<usize> = Vec::new();
        let mut domain: Vec<usize> = Vec::new();
        // Block A (speakers 0..4) in domain 0, block B (4..8) in domain 1.
        for i in 0..4 {
            docs.push(vec![(0, 3.0), (1, 2.0), (2, 1.0)]);
            speaker.push(i);
            domain.push(0);
        }
        for i in 4..8 {
            docs.push(vec![(0, 1.0), (1, 3.0), (2, 2.0)]);
            speaker.push(i);
            domain.push(1);
        }
        // Bridge domain 2: two speakers, one from each block, but only a single
        // repeated word -> fails the <2-local-word guard, so psi is zero-filled.
        for &i in &[0usize, 7usize] {
            docs.push(vec![(0, 4.0)]);
            speaker.push(i);
            domain.push(2);
        }
        let model = fit_wordshoal(
            &docs,
            v,
            &speaker,
            n,
            &domain,
            m,
            &[],
            50,
            1e-3,
            1.0,
            0.5,
            0.5,
            1.0,
        );
        // The bad bridge must not merge the blocks.
        assert_eq!(
            model.num_components, 2,
            "a failed bridge domain must not connect the two blocks"
        );
    }
}
