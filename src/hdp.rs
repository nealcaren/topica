//! Hierarchical Dirichlet Process topic model (Teh, Jordan, Beal & Blei 2006) —
//! LDA that **infers the number of topics** instead of fixing K up front.
//!
//! The model is a two-level Dirichlet process. A top-level DP with concentration
//! `γ` draws a global distribution over an unbounded set of topics; each document
//! is a DP with concentration `α0` that draws its own topic mixture from that
//! shared global menu. Topics are thus shared across documents (a "franchise"),
//! but how many are actually used is learned from the data.
//!
//! We use the **direct-assignment Gibbs sampler** (Teh et al. 2006, §5.3 / the
//! Chinese Restaurant Franchise). State:
//!   - `z[j][i]`  — the topic of token *i* in document *j*
//!   - `nkw`,`nk` — topic-word and topic-total counts (as in LDA), but K grows
//!   - `njk`      — per-document topic counts
//!   - `beta`     — the top-level topic weights β = (β_1..β_K, β_u), where β_u is
//!                  the leftover stick mass reserved for "a brand-new topic"
//!
//! Sampling a token's topic mixes the existing topics with a fresh-topic option
//! whose weight is `α0·β_u` and whose likelihood is the base measure (uniform
//! over the vocabulary). Picking the fresh option instantiates a new topic and
//! breaks a `Beta(1, γ)` piece off β_u. After each sweep we drop emptied topics,
//! resample the table counts `m_k` (Antoniak), redraw β ~ Dirichlet(m, γ), and
//! (optionally) resample the concentrations α0, γ — so K floats up and down.

use rand::Rng;

/// A fitted HDP topic model. `num_topics()` is the inferred K.
pub struct HdpModel {
    pub num_types: usize,
    pub eta: f64,           // topic-word Dirichlet (symmetric base measure)
    pub alpha: f64,         // α0, document-level DP concentration
    pub gamma: f64,         // γ, top-level DP concentration
    pub nkw: Vec<Vec<u32>>, // K × V topic-word counts
    pub nk: Vec<u32>,       // K topic totals
    pub beta: Vec<f64>,     // K top-level topic weights
    pub beta_u: f64,        // leftover stick mass (the "new topic" weight)
    pub z: Vec<Vec<usize>>, // per-document token topic assignments
    pub njk: Vec<Vec<u32>>, // num_docs × K document-topic counts
    /// Divergence backstop for resampled concentrations (see
    /// [`DEFAULT_CONCENTRATION_MAX`]); only consulted when `resample_conc = true`.
    pub concentration_max: f64,
    /// Discovery + convergence trace, one entry per recorded sweep:
    /// `(iteration, num_active_topics, per-token log-likelihood, alpha, gamma)`.
    /// The topic count and the (resampled) concentrations show the
    /// nonparametric model settling on K; the log-likelihood shows convergence.
    pub trace: Vec<(usize, usize, f64, f64, f64)>,
}

impl HdpModel {
    pub fn num_topics(&self) -> usize {
        self.nk.len()
    }

    /// Topic-word distributions β_{k,w} = (n_{kw}+η)/(n_k+Vη), shape K×V.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        let v = self.num_types;
        self.nkw
            .iter()
            .zip(&self.nk)
            .map(|(row, &nk)| {
                let denom = nk as f64 + v as f64 * self.eta;
                row.iter().map(|&c| (c as f64 + self.eta) / denom).collect()
            })
            .collect()
    }

    /// Document-topic mixtures θ_{j,k} ∝ n_{jk} + α0 β_k, shape D×K. Normalized
    /// over the K instantiated topics (the α0·β_u "new-topic" reserve is dropped
    /// and its mass redistributed), so each row is a proper distribution.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics();
        self.njk
            .iter()
            .map(|counts| {
                let mut row: Vec<f64> = (0..k)
                    .map(|t| counts[t] as f64 + self.alpha * self.beta[t])
                    .collect();
                let s: f64 = row.iter().sum();
                if s > 0.0 {
                    for x in row.iter_mut() {
                        *x /= s;
                    }
                }
                row
            })
            .collect()
    }

    /// Per-token predictive log-likelihood `(1/N) Σ_d Σ_{w∈d} log Σ_k θ_{d,k}
    /// β_{k,w}` — a convergence-trackable score over the current estimates.
    /// Returns `NaN` for an empty corpus.
    pub fn corpus_log_likelihood(&self, docs: &[Vec<u32>]) -> f64 {
        let beta = self.topic_word(); // K×V
        let theta = self.doc_topic(); // D×K
        let k = self.num_topics();
        let mut ll = 0.0f64;
        let mut n = 0usize;
        for (d, doc) in docs.iter().enumerate() {
            for &word in doc {
                let w = word as usize;
                let mut p = 0.0f64;
                for t in 0..k {
                    p += theta[d][t] * beta[t][w];
                }
                ll += p.max(1e-300).ln();
                n += 1;
            }
        }
        if n == 0 {
            f64::NAN
        } else {
            ll / n as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Random-variate helpers (only `rand`'s uniform is available; build the rest).
// ---------------------------------------------------------------------------

fn sample_normal<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller.
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape, 1) variate (Marsaglia & Tsang; boosted for shape < 1).
pub fn sample_gamma<R: Rng>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        let g = sample_gamma(shape + 1.0, rng);
        let u: f64 = rng.gen::<f64>().max(1e-300);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.gen::<f64>();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn sample_beta_dist<R: Rng>(a: f64, b: f64, rng: &mut R) -> f64 {
    let x = sample_gamma(a, rng);
    let y = sample_gamma(b, rng);
    if x + y <= 0.0 {
        0.5
    } else {
        x / (x + y)
    }
}

/// Antoniak: number of occupied tables when `n` customers are seated in a CRP
/// with concentration `theta` — m = Σ_{i=0}^{n-1} Bernoulli(theta/(theta+i)).
fn sample_tables<R: Rng>(n: u32, theta: f64, rng: &mut R) -> u32 {
    let mut m = 0u32;
    for i in 0..n {
        if rng.gen::<f64>() < theta / (theta + i as f64) {
            m += 1;
        }
    }
    m
}

fn sample_index<R: Rng>(probs: &[f64], rng: &mut R) -> usize {
    let total: f64 = probs.iter().sum();
    // A nonfinite or nonpositive total means the weights were corrupted
    // upstream; without this guard the loop below falls through to the last
    // index (the fresh-topic option), silently masking the corruption. Checked
    // only in debug builds to keep the per-token hot path free of branches.
    debug_assert!(
        total.is_finite() && total > 0.0,
        "sample_index: weights must sum to a finite positive total, got {total}"
    );
    let mut r = rng.gen::<f64>() * total;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// Default upper bound on a resampled DP concentration. The Escobar-West update
/// for γ draws from `Gamma(a + K, b - log η)`, whose mean grows linearly with the
/// topic count K; once K is large the rate term cannot pull γ back down, so γ
/// and K reinforce each other without bound (issue #68: K ran to 774 with γ at
/// 102). This cap is a *divergence backstop* for the opt-in `resample_conc = true`
/// path, not a statistical prior: any posterior with appreciable mass above it
/// gets a spurious atom at the cap, biasing the concentrations (and hence K)
/// downward. The conservative default matches the value chosen for #68; corpora
/// that legitimately support larger concentrations can raise it via
/// `HdpModel::concentration_max` (exposed as `concentration_max` on the Python
/// constructor). The default fits with fixed concentrations and never reach it.
pub(crate) const DEFAULT_CONCENTRATION_MAX: f64 = 2.0;

impl HdpModel {
    /// Drop any topic with no tokens, returning its stick mass to β_u and
    /// reindexing assignments/counts so topic indices stay contiguous.
    fn compact(&mut self) {
        let k = self.nk.len();
        let keep: Vec<usize> = (0..k).filter(|&t| self.nk[t] > 0).collect();
        if keep.len() == k {
            return;
        }
        // Return removed topics' β to the leftover stick.
        for t in 0..k {
            if self.nk[t] == 0 {
                self.beta_u += self.beta[t];
            }
        }
        let mut remap = vec![usize::MAX; k];
        for (new, &old) in keep.iter().enumerate() {
            remap[old] = new;
        }
        self.nkw = keep
            .iter()
            .map(|&t| std::mem::take(&mut self.nkw[t]))
            .collect();
        self.nk = keep.iter().map(|&t| self.nk[t]).collect();
        self.beta = keep.iter().map(|&t| self.beta[t]).collect();
        for doc in &mut self.njk {
            *doc = keep.iter().map(|&t| doc[t]).collect();
        }
        for zj in &mut self.z {
            for zi in zj.iter_mut() {
                *zi = remap[*zi];
            }
        }
    }

    /// Resample top-level weights β from the table counts, returning the total
    /// number of tables overall and per document (for concentration resampling).
    fn resample_beta<R: Rng>(&mut self, rng: &mut R) -> (u32, Vec<u32>) {
        let k = self.num_topics();
        let d = self.njk.len();
        let mut m_k = vec![0u32; k];
        let mut t_j = vec![0u32; d];
        for (j, counts) in self.njk.iter().enumerate() {
            for (t, &njk) in counts.iter().enumerate() {
                if njk == 0 {
                    continue;
                }
                let m = sample_tables(njk, self.alpha * self.beta[t], rng).max(1);
                m_k[t] += m;
                t_j[j] += m;
            }
        }
        // β ~ Dirichlet(m_1, …, m_K, γ).
        let mut gammas: Vec<f64> = m_k.iter().map(|&m| sample_gamma(m as f64, rng)).collect();
        let g_u = sample_gamma(self.gamma, rng);
        let total: f64 = gammas.iter().sum::<f64>() + g_u;
        let total = if total > 0.0 { total } else { 1.0 };
        for g in gammas.iter_mut() {
            *g /= total;
        }
        self.beta = gammas;
        self.beta_u = g_u / total;
        let m_total: u32 = m_k.iter().sum();
        (m_total, t_j)
    }

    /// Resample the top-level concentration γ given K topics and `m_total`
    /// tables (Escobar & West 1995 augmentation, Gamma(1,1) prior).
    fn resample_gamma<R: Rng>(&mut self, m_total: u32, rng: &mut R) {
        let k = self.num_topics() as f64;
        let m = m_total as f64;
        if m < 1.0 {
            return;
        }
        let (a, b) = (1.0, 1.0); // weak Gamma(shape, rate) prior
        let eta = sample_beta_dist(self.gamma + 1.0, m, rng).max(1e-12);
        let pi = (a + k - 1.0) / (a + k - 1.0 + m * (b - eta.ln()));
        let shape = if rng.gen::<f64>() < pi {
            a + k
        } else {
            a + k - 1.0
        };
        self.gamma =
            (sample_gamma(shape, rng) / (b - eta.ln())).clamp(1e-3, self.concentration_max);
    }

    /// Resample the document-level concentration α0 given per-document word
    /// counts `n_j` and table counts `t_j`. Matches blei-lab/hdp's
    /// `sample_second_level_concentration` (Teh et al. 2006), including the few
    /// inner auxiliary-resampling steps it runs to mix. Gamma(1,1) prior.
    fn resample_alpha<R: Rng>(&mut self, t_j: &[u32], rng: &mut R) {
        let total_tables: u32 = t_j.iter().sum();
        if total_tables < 1 {
            return;
        }
        let (a, b) = (1.0, 1.0);
        let doc_lengths: Vec<u32> = self.njk.iter().map(|c| c.iter().sum()).collect();
        for _ in 0..20 {
            let mut sum_log_w = 0.0;
            let mut sum_s = 0.0;
            for &nj in &doc_lengths {
                if nj == 0 {
                    continue;
                }
                let w = sample_beta_dist(self.alpha + 1.0, nj as f64, rng).max(1e-12);
                sum_log_w += w.ln();
                // s_j ~ Bernoulli(n_j / (n_j + α0)).
                if rng.gen::<f64>() < nj as f64 / (nj as f64 + self.alpha) {
                    sum_s += 1.0;
                }
            }
            let shape = a + total_tables as f64 - sum_s;
            let rate = b - sum_log_w;
            if shape > 0.0 && rate > 0.0 {
                self.alpha = (sample_gamma(shape, rng) / rate).clamp(1e-3, self.concentration_max);
            }
        }
    }

    /// Draw a topic for token (j, i)=w from the current state (counts for this
    /// token must already be removed), instantiating a fresh topic if drawn, and
    /// add the token back under the chosen topic.
    fn assign_token<R: Rng>(
        &mut self,
        j: usize,
        i: usize,
        w: usize,
        probs: &mut Vec<f64>,
        rng: &mut R,
    ) {
        let v = self.num_types;
        let base = 1.0 / v as f64; // base-measure likelihood for a fresh topic
        let k = self.nk.len();
        // Reusable scratch (one allocation per sweep, not per token); resized to
        // k+1 each call because K grows as fresh topics are instantiated.
        probs.clear();
        probs.resize(k + 1, 0.0);
        for t in 0..k {
            let f = (self.nkw[t][w] as f64 + self.eta) / (self.nk[t] as f64 + v as f64 * self.eta);
            probs[t] = (self.njk[j][t] as f64 + self.alpha * self.beta[t]) * f;
        }
        probs[k] = self.alpha * self.beta_u * base;

        let k_new = sample_index(probs, rng);
        if k_new == k {
            // Instantiate a new topic; break a Beta(1, γ) piece off β_u.
            let b = sample_beta_dist(1.0, self.gamma, rng);
            let new_beta = b * self.beta_u;
            // Subtract the mass actually broken off rather than rescaling by
            // `1 - b`; when b ≈ 1 (small γ) the `1 - b` form cancels to exactly 0
            // and stalls further topic births, whereas this keeps β_u consistent
            // with the mass handed out.
            self.beta_u -= new_beta;
            self.beta.push(new_beta);
            self.nkw.push(vec![0u32; v]);
            self.nk.push(0);
            for doc_counts in &mut self.njk {
                doc_counts.push(0);
            }
        }
        self.nkw[k_new][w] += 1;
        self.nk[k_new] += 1;
        self.njk[j][k_new] += 1;
        self.z[j][i] = k_new;
    }

    /// One full Gibbs sweep over every token, instantiating new topics as drawn.
    fn sweep<R: Rng>(&mut self, docs: &[Vec<u32>], rng: &mut R) {
        let mut probs: Vec<f64> = Vec::new();
        for (j, doc) in docs.iter().enumerate() {
            for (i, &w) in doc.iter().enumerate() {
                let w = w as usize;
                let k_old = self.z[j][i];
                self.nkw[k_old][w] -= 1;
                self.nk[k_old] -= 1;
                self.njk[j][k_old] -= 1;
                self.assign_token(j, i, w, &mut probs, rng);
            }
        }
    }
}

/// Fit an HDP topic model by direct-assignment Gibbs sampling.
///
/// Topics are grown by a **sequential CRF initialization** (each token is drawn
/// from the partial state with the usual fresh-topic option), then refined by
/// Gibbs sweeps. This greedy start gives a compact, well-separated topic set and
/// mixes far better than random seeding. `resample_conc` toggles resampling of
/// the concentrations α0, γ (recommended — fixed concentrations make K very
/// sensitive to their values).
pub fn fit_hdp<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    alpha: f64,
    gamma: f64,
    eta: f64,
    iters: usize,
    resample_conc: bool,
    concentration_max: f64,
    report_interval: usize,
    rng: &mut R,
) -> HdpModel {
    // Guard the numeric preconditions the sampler relies on. The Python
    // constructor already validates its hyperparameters, but direct Rust callers
    // can otherwise trigger NaNs, division-by-zero, or out-of-range indexing.
    assert!(num_types > 0, "fit_hdp: vocabulary size must be positive");
    assert!(
        alpha.is_finite() && alpha > 0.0,
        "fit_hdp: alpha must be finite and positive, got {alpha}"
    );
    assert!(
        gamma.is_finite() && gamma > 0.0,
        "fit_hdp: gamma must be finite and positive, got {gamma}"
    );
    assert!(
        eta.is_finite() && eta > 0.0,
        "fit_hdp: eta must be finite and positive, got {eta}"
    );
    assert!(
        concentration_max.is_finite() && concentration_max > 1e-3,
        "fit_hdp: concentration_max must be finite and > 1e-3, got {concentration_max}"
    );
    debug_assert!(
        docs.iter().flatten().all(|&w| (w as usize) < num_types),
        "fit_hdp: a token id is out of range for num_types={num_types}"
    );
    let mut model = HdpModel {
        num_types,
        eta,
        alpha,
        gamma,
        nkw: Vec::new(),
        nk: Vec::new(),
        beta: Vec::new(),
        beta_u: 1.0, // entire stick is initially "unused"
        z: docs.iter().map(|d| vec![0usize; d.len()]).collect(),
        njk: vec![Vec::new(); docs.len()],
        concentration_max,
        trace: Vec::new(),
    };

    // Sequential CRF init: place each token in turn, growing topics as needed.
    let mut probs: Vec<f64> = Vec::new();
    for (j, doc) in docs.iter().enumerate() {
        for (i, &w) in doc.iter().enumerate() {
            model.assign_token(j, i, w as usize, &mut probs, rng);
        }
    }
    // Establish a proper β from the initial table counts.
    model.resample_beta(rng);

    for it in 0..iters {
        model.sweep(docs, rng);
        model.compact();
        let (m_total, t_j) = model.resample_beta(rng);
        if resample_conc {
            model.resample_gamma(m_total, rng);
            model.resample_alpha(&t_j, rng);
        }
        // Record the discovery/convergence trace (always the final sweep too).
        if report_interval > 0 && ((it + 1) % report_interval == 0 || it + 1 == iters) {
            let ll = model.corpus_log_likelihood(docs);
            model
                .trace
                .push((it + 1, model.num_topics(), ll, model.alpha, model.gamma));
        }
    }
    model.compact();
    model
}

use crate::estimator::{DirichletModel, Estimator, ModelFamily};

impl Estimator for HdpModel {
    fn num_topics(&self) -> usize {
        // Disambiguate inherent num_topics() inside the trait impl.
        HdpModel::num_topics(self)
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.trace
            .iter()
            .map(|&(it, _, ll, _, _)| (it, ll))
            .collect()
    }

    fn converged(&self) -> Option<bool> {
        None
    }

    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

impl DirichletModel for HdpModel {
    fn alpha(&self) -> Vec<f64> {
        vec![self.alpha; HdpModel::num_topics(self)]
    }

    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        Vec::new()
    }

    fn doc_lengths(&self) -> Vec<usize> {
        self.njk
            .iter()
            .map(|c| c.iter().map(|&x| x as usize).sum())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn infers_number_of_planted_topics() {
        // Five disjoint-vocabulary topics; HDP should recover ~5 topics and put
        // each one's mass on a single vocabulary block.
        let blocks: Vec<Vec<u32>> = (0..5).map(|b| (b * 6..b * 6 + 6).collect()).collect();
        let v = 30;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut docs = Vec::new();
        for d in 0..250 {
            let blk = &blocks[d % 5];
            // 12-token docs drawn from one block.
            let doc: Vec<u32> = (0..12).map(|i| blk[(i + d) % blk.len()]).collect();
            docs.push(doc);
        }
        let model = fit_hdp(
            &docs,
            v,
            1.0,
            1.0,
            0.01,
            100,
            true,
            DEFAULT_CONCENTRATION_MAX,
            0,
            &mut rng,
        );
        let k = model.num_topics();
        // Auto-K is approximate and HDP tends to slightly over-segment; the firm
        // requirement is that it recovers a sane count, not the exact 5.
        assert!(
            (4..=12).contains(&k),
            "inferred K={} not in a sane band for 5",
            k
        );

        // The substantive check: every planted block is the top of some topic.
        let tw = model.topic_word();
        let mut covered = std::collections::HashSet::new();
        for row in &tw {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..6].iter().copied().collect();
            for (bi, blk) in blocks.iter().enumerate() {
                if blk.iter().all(|&w| top.contains(&(w as usize))) {
                    covered.insert(bi);
                }
            }
        }
        assert_eq!(
            covered.len(),
            5,
            "recovered only {} of 5 planted blocks",
            covered.len()
        );
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let docs: Vec<Vec<u32>> = (0..40)
            .map(|d| (0..8).map(|i| ((i + d) % 10) as u32).collect())
            .collect();
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(7);
        let m1 = fit_hdp(
            &docs,
            10,
            1.0,
            1.0,
            0.1,
            20,
            true,
            DEFAULT_CONCENTRATION_MAX,
            0,
            &mut r1,
        );
        let m2 = fit_hdp(
            &docs,
            10,
            1.0,
            1.0,
            0.1,
            20,
            true,
            DEFAULT_CONCENTRATION_MAX,
            0,
            &mut r2,
        );
        assert_eq!(m1.num_topics(), m2.num_topics());
        assert_eq!(m1.nk, m2.nk);
    }

    #[test]
    fn discovery_trace_records_and_converges() {
        // Five disjoint-vocabulary topics; with a trace cadence the recorded
        // K should land near 5 and the log-likelihood should rise.
        let blocks: Vec<Vec<u32>> = (0..5).map(|b| (b * 6..b * 6 + 6).collect()).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let docs: Vec<Vec<u32>> = (0..250)
            .map(|d| (0..12).map(|i| blocks[d % 5][(i + d) % 6]).collect())
            .collect();
        let model = fit_hdp(
            &docs,
            30,
            1.0,
            1.0,
            0.01,
            120,
            true,
            DEFAULT_CONCENTRATION_MAX,
            10,
            &mut rng,
        );

        let trace = &model.trace;
        // iters=120, interval=10 -> sweeps 10,20,...,120.
        assert_eq!(
            trace.iter().map(|t| t.0).collect::<Vec<_>>(),
            (1..=12).map(|i| i * 10).collect::<Vec<_>>()
        );
        // Topic count and log-likelihood are sane; the final K is recorded.
        assert!(trace
            .iter()
            .all(|t| t.1 >= 1 && t.2.is_finite() && t.2 < 0.0));
        assert_eq!(trace.last().unwrap().1, model.num_topics());
        // The fit improves from the first recorded sweep to the last.
        assert!(trace.last().unwrap().2 > trace.first().unwrap().2);
    }

    // A large-K state whose Escobar-West posterior for γ sits well above 2.0.
    fn large_k_state(concentration_max: f64) -> HdpModel {
        let k = 300usize;
        let v = 50usize;
        HdpModel {
            num_types: v,
            eta: 0.01,
            alpha: 1.0,
            gamma: 1.0,
            nkw: vec![vec![1u32; v]; k],
            nk: vec![v as u32; k],
            beta: vec![1.0 / (k as f64 + 1.0); k],
            beta_u: 1.0 / (k as f64 + 1.0),
            z: Vec::new(),
            njk: vec![(0..k).map(|t| (t % 3 + 1) as u32).collect(); 200],
            concentration_max,
            trace: Vec::new(),
        }
    }

    #[test]
    fn concentration_resampling_is_capped() {
        // Issue #68: with many topics, the Escobar-West gamma update draws from
        // Gamma(a+K, .) whose mean grows with K, so gamma (and K) ran away to the
        // hundreds. Drive the resamplers from a large-K state and confirm both
        // concentrations stay bounded by the configured cap.
        let cap = DEFAULT_CONCENTRATION_MAX;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut model = large_k_state(cap);
        for _ in 0..50 {
            let (m_total, t_j) = model.resample_beta(&mut rng);
            model.resample_gamma(m_total, &mut rng);
            model.resample_alpha(&t_j, &mut rng);
            assert!(model.gamma <= cap + 1e-9, "gamma {} > cap", model.gamma);
            assert!(model.alpha <= cap + 1e-9, "alpha {} > cap", model.alpha);
        }
    }

    #[test]
    fn raising_the_cap_removes_the_spurious_atom_at_two() {
        // #433: at the old hard cap of 2.0 the same large-K posterior pins γ at
        // exactly 2.0 (a spurious atom that biases K downward). With a generous
        // cap the resampled γ is free to sit above 2.0, and the default cap does
        // in fact clamp it — so the two paths are demonstrably different.
        let mut rng_lo = ChaCha8Rng::seed_from_u64(0);
        let mut lo = large_k_state(DEFAULT_CONCENTRATION_MAX); // 2.0
        let mut rng_hi = ChaCha8Rng::seed_from_u64(0);
        let mut hi = large_k_state(1.0e6);
        let mut hi_exceeded_two = false;
        for _ in 0..50 {
            let (m_lo, t_lo) = lo.resample_beta(&mut rng_lo);
            lo.resample_gamma(m_lo, &mut rng_lo);
            lo.resample_alpha(&t_lo, &mut rng_lo);
            assert!(lo.gamma <= DEFAULT_CONCENTRATION_MAX + 1e-9);

            let (m_hi, t_hi) = hi.resample_beta(&mut rng_hi);
            hi.resample_gamma(m_hi, &mut rng_hi);
            hi.resample_alpha(&t_hi, &mut rng_hi);
            if hi.gamma > DEFAULT_CONCENTRATION_MAX + 1e-6 {
                hi_exceeded_two = true;
            }
        }
        assert!(
            hi_exceeded_two,
            "with a raised cap the large-K posterior should let gamma exceed 2.0"
        );
    }

    #[test]
    fn hdp_conforms() {
        let docs: Vec<Vec<u32>> = (0..40)
            .map(|d| (0..8).map(|i| ((i + d) % 10) as u32).collect())
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_hdp(
            &docs,
            10,
            1.0,
            1.0,
            0.1,
            20,
            true,
            DEFAULT_CONCENTRATION_MAX,
            0,
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }
}
