//! Content-covariate topic model (SAGE / the STM content model).
//!
//! Topic-word distributions vary by a document-level *group* covariate. The log
//! topic-word weight is additive in sparse deviations from a background:
//!
//! ```text
//!   η_{k,g,v} = m_v + κᵀ_{k,v} + κᶜ_{g,v} + κᴵ_{k,g,v}
//!   β_{k,g,v} = softmax_v( η_{k,g,v} )
//! ```
//!
//! where `m_v` is the (fixed) background log word-frequency, `κᵀ` is the topic
//! deviation, `κᶜ` the group deviation, and `κᴵ` the topic×group interaction.
//! A token in a group-`g` document samples its topic using that group's `β`.
//! The κ are MAP-estimated from the topic×group×word counts between sampling
//! sweeps, under a [`SagePrior`]: the canonical **sparse** Laplace/Jeffreys prior
//! (fit by adaptive reweighting, driving most deviations to ~0 — this is what
//! makes it SAGE rather than a ridge content model) or a dense `Gaussian` ridge.
//! Both reuse the L-BFGS from DMR.
//!
//! Inference is collapsed Gibbs for the token–topic assignments with periodic MAP
//! re-estimation of κ — this is the SAGE *model*, not a literal reproduction of the
//! ICML paper's variational EM (which uses expected counts). SAGE has no external
//! reference implementation to benchmark against (no widely-used library ships it),
//! so it is validated by planted-topic recovery, refit self-consistency
//! (`parity/sage_gold.py`), and the defining sparsity behaviour — the sparse prior
//! must drive non-discriminative κ to ≈0 while keeping discriminative words nonzero.

use rand::Rng;

use crate::variational::lbfgs_minimize;

/// The prior on the κ content deviations. Canonical SAGE (Eisenstein, Ahmed &
/// Xing, ICML 2011) is *defined* by a sparsity-inducing prior; `Gaussian` is the
/// non-sparse ridge variant (the STM content-model style).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SagePrior {
    /// L2 ridge with a fixed global variance. Dense κ; the pre-#422 behaviour.
    Gaussian,
    /// Laplace (double-exponential) prior — the canonical sparse SAGE default.
    /// Fit by adaptive reweighting: the per-coefficient precision is `1/(b·|κ|)`,
    /// which drives most κ toward zero.
    Laplace,
    /// Normal-Jeffreys prior (improper `p(τ) ∝ 1/τ`). More aggressive than
    /// Laplace: precision `1/κ²`. Guarded by a floor; can over-sparsify.
    Jeffreys,
}

impl SagePrior {
    pub fn parse(s: &str) -> Result<SagePrior, String> {
        match s {
            "laplace" | "sparse" => Ok(SagePrior::Laplace),
            "gaussian" | "ridge" | "normal" => Ok(SagePrior::Gaussian),
            "jeffreys" => Ok(SagePrior::Jeffreys),
            other => Err(format!(
                "prior must be 'laplace', 'gaussian', or 'jeffreys', got '{other}'"
            )),
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            SagePrior::Gaussian => "gaussian",
            SagePrior::Laplace => "laplace",
            SagePrior::Jeffreys => "jeffreys",
        }
    }
    pub fn is_sparse(&self) -> bool {
        !matches!(self, SagePrior::Gaussian)
    }
}

/// SAGE model state. Counts are dense over (topic, group, word).
pub struct SageModel {
    pub num_topics: usize,
    pub num_groups: usize,
    pub num_types: usize,
    pub alpha: Vec<f64>,
    pub prior_variance: f64,
    pub prior: SagePrior,

    pub m: Vec<f64>,            // background log-freq, len V
    pub kappa_t: Vec<Vec<f64>>, // [K][V]
    pub kappa_c: Vec<Vec<f64>>, // [G][V]
    pub kappa_i: Vec<Vec<f64>>, // [K*G][V]  (index k*G+g)

    pub beta: Vec<Vec<f64>>,   // [K*G][V] cached normalized topic-word dists
    pub counts: Vec<Vec<u32>>, // [K*G][V] token counts n_{k,g,v}
    pub totals: Vec<u32>,      // [K*G] = Σ_v counts
    pub doc_topics: Vec<Vec<u32>>,
}

impl SageModel {
    pub fn new(
        num_topics: usize,
        num_groups: usize,
        num_types: usize,
        alpha: f64,
        prior_variance: f64,
        prior: SagePrior,
    ) -> Self {
        let kg = num_topics * num_groups;
        SageModel {
            num_topics,
            num_groups,
            num_types,
            alpha: vec![alpha; num_topics],
            prior_variance,
            prior,
            m: vec![0.0; num_types],
            kappa_t: vec![vec![0.0; num_types]; num_topics],
            kappa_c: vec![vec![0.0; num_types]; num_groups],
            kappa_i: vec![vec![0.0; num_types]; kg],
            beta: vec![vec![0.0; num_types]; kg],
            counts: vec![vec![0u32; num_types]; kg],
            totals: vec![0u32; kg],
            doc_topics: Vec::new(),
        }
    }

    #[inline]
    fn cell(&self, k: usize, g: usize) -> usize {
        k * self.num_groups + g
    }

    /// Set the fixed background `m_v` to the unsmoothed empirical corpus log
    /// word-frequency (Eisenstein, Ahmed & Xing 2011): `m_v = ln(count_v / N)`.
    /// The κ deviations are estimated relative to this fixed reference, so it is
    /// the raw corpus frequency, not a smoothed distribution. The vocabulary is
    /// built from the same corpus, so every type has `count_v ≥ 1`; a type absent
    /// from `docs` is floored to one count purely to keep `m` finite.
    pub fn set_background(&mut self, docs: &[Vec<u32>]) {
        let mut freq = vec![0.0f64; self.num_types];
        let mut total = 0.0f64;
        for doc in docs {
            for &w in doc {
                freq[w as usize] += 1.0;
                total += 1.0;
            }
        }
        let total = total.max(1.0);
        for v in 0..self.num_types {
            let count = if freq[v] > 0.0 { freq[v] } else { 1.0 };
            self.m[v] = (count / total).ln();
        }
    }

    /// Recompute the cached `β_{k,g,·}` from the current κ (call after every κ update).
    pub fn recompute_beta(&mut self) {
        for k in 0..self.num_topics {
            for g in 0..self.num_groups {
                let c = self.cell(k, g);
                let mut max = f64::NEG_INFINITY;
                for v in 0..self.num_types {
                    let eta =
                        self.m[v] + self.kappa_t[k][v] + self.kappa_c[g][v] + self.kappa_i[c][v];
                    self.beta[c][v] = eta;
                    if eta > max {
                        max = eta;
                    }
                }
                let mut z = 0.0;
                for v in 0..self.num_types {
                    let e = (self.beta[c][v] - max).exp();
                    self.beta[c][v] = e;
                    z += e;
                }
                for v in 0..self.num_types {
                    self.beta[c][v] /= z;
                }
            }
        }
    }

    /// Random initial topic assignments; build counts.
    pub fn initialize<R: Rng>(&mut self, docs: &[Vec<u32>], groups: &[usize], rng: &mut R) {
        self.recompute_beta();
        self.doc_topics = docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .map(|_| rng.gen_range(0..self.num_topics) as u32)
                    .collect()
            })
            .collect();
        for (d, doc) in docs.iter().enumerate() {
            let g = groups[d];
            for (pos, &w) in doc.iter().enumerate() {
                let k = self.doc_topics[d][pos] as usize;
                let c = self.cell(k, g);
                self.counts[c][w as usize] += 1;
                self.totals[c] += 1;
            }
        }
    }
}

/// One Gibbs sweep: each token samples a topic using its document's group `β`.
pub fn run_sweep_sage<R: Rng>(
    model: &mut SageModel,
    docs: &[Vec<u32>],
    groups: &[usize],
    rng: &mut R,
) {
    let k_n = model.num_topics;
    let g_n = model.num_groups;
    let mut local = vec![0u32; k_n];
    let mut scores = vec![0.0f64; k_n];

    for d in 0..docs.len() {
        let g = groups[d];
        for t in local.iter_mut() {
            *t = 0;
        }
        for &t in &model.doc_topics[d] {
            local[t as usize] += 1;
        }

        for pos in 0..docs[d].len() {
            let w = docs[d][pos] as usize;
            let old = model.doc_topics[d][pos] as usize;
            let oc = old * g_n + g;
            model.counts[oc][w] -= 1;
            model.totals[oc] -= 1;
            local[old] -= 1;

            let mut total = 0.0;
            for k in 0..k_n {
                let s = (local[k] as f64 + model.alpha[k]) * model.beta[k * g_n + g][w];
                scores[k] = s;
                total += s;
            }
            let mut r = rng.gen::<f64>() * total;
            let mut chosen = k_n - 1;
            for k in 0..k_n {
                r -= scores[k];
                if r <= 0.0 {
                    chosen = k;
                    break;
                }
            }

            let nc = chosen * g_n + g;
            model.counts[nc][w] += 1;
            model.totals[nc] += 1;
            local[chosen] += 1;
            model.doc_topics[d][pos] = chosen as u32;
        }
    }
}

/// Outer adaptive-reweighting iterations for the sparse priors.
const SPARSE_OUTER_ITERS: usize = 4;
/// Fixed neutral precision for the first (κ = 0) sparse solve, independent of
/// `prior_variance` so Jeffreys stays scale-free.
const SPARSE_INIT_PRECISION: f64 = 1.0;
/// Scale-aware floor on |κ| (Laplace) / κ² (Jeffreys) so the reweighted precision
/// cannot blow up to +inf as a coefficient approaches zero. Caps Laplace precision
/// at `1/(b·FLOOR)` and Jeffreys at `1/FLOOR²`.
const KAPPA_FLOOR: f64 = 1e-3;

/// Per-coefficient precision `w_i = 1/τ_i` for the weighted-ridge κ step, from the
/// current κ. Gaussian is the fixed global `1/σ²`; the sparse priors reweight from
/// the coefficient magnitude (the scale-mixture E[1/τ|κ] update), which shrinks
/// small coefficients ever harder — this is what induces sparsity.
fn reweight_precision(x: &[f64], prior: SagePrior, prior_variance: f64, w: &mut [f64]) {
    match prior {
        SagePrior::Gaussian => {
            let iv = 1.0 / prior_variance;
            for wi in w.iter_mut() {
                *wi = iv;
            }
        }
        // Laplace scale-mixture: E[1/τ | κ] = 1/(b·|κ|), floored.
        SagePrior::Laplace => {
            let b = prior_variance;
            for (i, wi) in w.iter_mut().enumerate() {
                *wi = 1.0 / (b * x[i].abs().max(KAPPA_FLOOR));
            }
        }
        // Normal-Jeffreys: E[1/τ | κ] = 1/κ², floored (scale-free).
        SagePrior::Jeffreys => {
            let floor_sq = KAPPA_FLOOR * KAPPA_FLOOR;
            for (i, wi) in w.iter_mut().enumerate() {
                *wi = 1.0 / (x[i] * x[i]).max(floor_sq);
            }
        }
    }
}

/// MAP-estimate the κ deviations from the current counts under the model's prior,
/// then refresh the cached β. For `Gaussian` this is one L-BFGS run with a fixed
/// ridge; for the sparse priors (`Laplace`/`Jeffreys`) it is an adaptive-reweighting
/// loop (paper: the compound-Gamma / scale-mixture EM) — a weighted-ridge L-BFGS
/// solve alternated with a per-coefficient precision update, warm-started from the
/// incoming κ so the first precision is never taken from an all-zero κ. Returns
/// `false` if any solve produced a non-finite result, in which case κ (and β) are
/// left unchanged rather than corrupted (issue #422).
#[must_use]
pub fn optimize_kappa(model: &mut SageModel, max_iter: usize) -> bool {
    let k_n = model.num_topics;
    let g_n = model.num_groups;
    let v_n = model.num_types;

    // Pack κ into a flat vector: [κT (K*V) | κC (G*V) | κI (K*G*V)].
    let n_t = k_n * v_n;
    let n_c = g_n * v_n;
    let dim = n_t + n_c + k_n * g_n * v_n;
    let mut x = Vec::with_capacity(dim);
    for k in 0..k_n {
        x.extend_from_slice(&model.kappa_t[k]);
    }
    for g in 0..g_n {
        x.extend_from_slice(&model.kappa_c[g]);
    }
    for c in 0..(k_n * g_n) {
        x.extend_from_slice(&model.kappa_i[c]);
    }

    let prior = model.prior;
    let prior_variance = model.prior_variance;

    // Compute κ in a scope that borrows the read-only model fields, so the mutable
    // unpack below is free of the LL closure's borrow.
    let (x, ok) = {
        let m = &model.m;
        let counts = &model.counts;
        let totals = &model.totals;

        // The multinomial log-likelihood value/gradient, penalized by a
        // per-coefficient ridge `w` (the only per-prior difference).
        let solve = |x0: Vec<f64>, w: &[f64]| -> Vec<f64> {
            lbfgs_minimize(
                x0,
                |flat| {
                    let kt = |k: usize, v: usize| flat[k * v_n + v];
                    let kc = |g: usize, v: usize| flat[n_t + g * v_n + v];
                    let ki = |c: usize, v: usize| flat[n_t + n_c + c * v_n + v];

                    let mut value = 0.0f64;
                    let mut grad = vec![0.0f64; flat.len()];

                    for k in 0..k_n {
                        for g in 0..g_n {
                            let c = k * g_n + g;
                            let nkg = totals[c] as f64;
                            let mut max = f64::NEG_INFINITY;
                            let mut eta = vec![0.0f64; v_n];
                            for v in 0..v_n {
                                let e = m[v] + kt(k, v) + kc(g, v) + ki(c, v);
                                eta[v] = e;
                                if e > max {
                                    max = e;
                                }
                            }
                            let mut z = 0.0;
                            for v in 0..v_n {
                                z += (eta[v] - max).exp();
                            }
                            let log_z = max + z.ln();
                            for v in 0..v_n {
                                let n = counts[c][v] as f64;
                                value += n * (eta[v] - log_z);
                                let beta = (eta[v] - log_z).exp();
                                let resid = n - nkg * beta; // ∂LL/∂η_{k,g,v}
                                grad[k * v_n + v] += resid; // κT
                                grad[n_t + g * v_n + v] += resid; // κC
                                grad[n_t + n_c + c * v_n + v] += resid; // κI
                            }
                        }
                    }

                    // Per-coefficient ridge penalty: -½ Σ w_i κ_i².
                    for (i, &xi) in flat.iter().enumerate() {
                        value -= 0.5 * w[i] * xi * xi;
                        grad[i] -= w[i] * xi;
                    }

                    (-value, grad.iter().map(|gv| -gv).collect())
                },
                max_iter,
                7,
                1e-4,
            )
        };

        let mut w = vec![0.0; dim];
        if prior.is_sparse() {
            // Warm-start the precision from the incoming κ; if κ is all-zero (the
            // first update of a fresh fit), seed a plain neutral ridge so the first
            // solve yields a non-zero κ to reweight from — never derive 1/|κ| from
            // κ = 0. This seed is deliberately a FIXED precision, independent of
            // `prior_variance`, so Jeffreys stays scale-free (its reweight `1/κ²`
            // never sees `prior_variance`) and Laplace's scale enters only through
            // its own `1/(b|κ|)` reweight, not the throwaway init.
            if x.iter().all(|&v| v == 0.0) {
                w.iter_mut().for_each(|wi| *wi = SPARSE_INIT_PRECISION);
            } else {
                reweight_precision(&x, prior, prior_variance, &mut w);
            }
            // A non-finite precision (e.g. a subnormal `prior_variance` whose inverse
            // overflows) makes the penalized objective ill-posed. Since L-BFGS now
            // keeps the last finite point rather than propagating a NaN, detect the
            // degeneracy here and roll back (#419 / #422); a mid-loop non-finite
            // reweight is still caught by the next solve's finiteness check.
            let mut ok = w.iter().all(|v| v.is_finite());
            for _ in 0..SPARSE_OUTER_ITERS {
                if !ok {
                    break;
                }
                x = solve(x, &w);
                if x.iter().any(|v| !v.is_finite()) {
                    ok = false;
                    break;
                }
                reweight_precision(&x, prior, prior_variance, &mut w);
                // A non-finite reweighted precision means the next solve is degenerate
                // (e.g. a subnormal `prior_variance` whose inverse overflows). L-BFGS
                // now keeps the last finite point instead of returning NaN, so detect
                // it here and roll back rather than accept a spuriously "finite" κ.
                if w.iter().any(|v| !v.is_finite()) {
                    ok = false;
                    break;
                }
            }
            (x, ok)
        } else {
            reweight_precision(&x, prior, prior_variance, &mut w); // uniform 1/σ²
            if w.iter().any(|v| !v.is_finite()) {
                (x, false) // degenerate precision -> leave κ unchanged
            } else {
                x = solve(x, &w);
                let ok = x.iter().all(|v| v.is_finite());
                (x, ok)
            }
        }
    };

    if !ok {
        return false;
    }

    // Unpack.
    for k in 0..k_n {
        model.kappa_t[k].copy_from_slice(&x[k * v_n..(k + 1) * v_n]);
    }
    for g in 0..g_n {
        let off = n_t + g * v_n;
        model.kappa_c[g].copy_from_slice(&x[off..off + v_n]);
    }
    for c in 0..(k_n * g_n) {
        let off = n_t + n_c + c * v_n;
        model.kappa_i[c].copy_from_slice(&x[off..off + v_n]);
    }

    model.recompute_beta();
    true
}

use crate::estimator::{DirichletModel, Estimator, ModelFamily};

impl Estimator for SageModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        // Average the cached beta (shape (K*G)×V, indexed k*num_groups+g) over groups.
        (0..self.num_topics)
            .map(|k| {
                let mut avg = vec![0.0f64; self.num_types];
                for g in 0..self.num_groups {
                    let row = &self.beta[k * self.num_groups + g];
                    for v in 0..self.num_types {
                        avg[v] += row[v];
                    }
                }
                for v in 0..self.num_types {
                    avg[v] /= self.num_groups as f64;
                }
                avg
            })
            .collect()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        // Smoothed proportions from per-token topic ids with per-topic alpha.
        let alpha_sum: f64 = self.alpha.iter().sum();
        self.doc_topics
            .iter()
            .map(|toks| {
                let mut cnt = vec![0.0f64; self.num_topics];
                for &t in toks {
                    cnt[t as usize] += 1.0;
                }
                let denom = toks.len() as f64 + alpha_sum;
                (0..self.num_topics)
                    .map(|t| (cnt[t] + self.alpha[t]) / denom)
                    .collect()
            })
            .collect()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }

    fn converged(&self) -> Option<bool> {
        None
    }

    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

impl DirichletModel for SageModel {
    fn alpha(&self) -> Vec<f64> {
        self.alpha.clone()
    }

    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        Vec::new()
    }

    fn doc_lengths(&self) -> Vec<usize> {
        self.doc_topics.iter().map(|d| d.len()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // One topic, two groups: group 0 uses words {0,1}, group 1 uses words {2,3}.
    // The content covariate should make that single topic worded differently per
    // group — β favours {0,1} for group 0 and {2,3} for group 1.
    #[test]
    fn recovers_group_specific_wording() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut docs = Vec::new();
        let mut groups = Vec::new();
        for i in 0..120 {
            if i % 2 == 0 {
                docs.push(vec![0u32, 1, 0, 1, 0, 1]);
                groups.push(0usize);
            } else {
                docs.push(vec![2u32, 3, 2, 3, 2, 3]);
                groups.push(1usize);
            }
        }

        let mut model = SageModel::new(1, 2, 4, 0.1, 1.0, SagePrior::Gaussian);
        model.set_background(&docs);
        model.initialize(&docs, &groups, &mut rng);
        for iter in 1..=200 {
            run_sweep_sage(&mut model, &docs, &groups, &mut rng);
            if iter > 50 && iter % 25 == 0 {
                assert!(optimize_kappa(&mut model, 20));
            }
        }

        let g0 = model.cell(0, 0);
        let g1 = model.cell(0, 1);
        // Group 0 puts more mass on {0,1}; group 1 on {2,3}.
        let g0_ab = model.beta[g0][0] + model.beta[g0][1];
        let g1_cd = model.beta[g1][2] + model.beta[g1][3];
        assert!(g0_ab > 0.8, "group 0 mass on its words = {}", g0_ab);
        assert!(g1_cd > 0.8, "group 1 mass on its words = {}", g1_cd);
    }

    /// Fit the same corpus under each prior. Words 0-3 are group-discriminative
    /// ({0,1} favour group 0, {2,3} favour group 1); words 4,5 are shared
    /// background (identical rate in both groups), so a faithful SAGE prior should
    /// drive their group deviations κ_c toward zero. The sparse priors must (a)
    /// still recover the discrimination and (b) shrink the non-discriminative κ_c
    /// harder than the Gaussian ridge.
    fn fit_prior(prior: SagePrior) -> SageModel {
        fit_prior_pv(prior, 1.0)
    }

    fn fit_prior_pv(prior: SagePrior, prior_variance: f64) -> SageModel {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut docs = Vec::new();
        let mut groups = Vec::new();
        for i in 0..160 {
            let g = i % 2;
            // 4 discriminative tokens (group-specific) + 2 shared background tokens.
            let mut doc = if g == 0 {
                vec![0u32, 1, 0, 1]
            } else {
                vec![2u32, 3, 2, 3]
            };
            doc.push(4);
            doc.push(5);
            docs.push(doc);
            groups.push(g);
        }
        let mut model = SageModel::new(1, 2, 6, 0.1, prior_variance, prior);
        model.set_background(&docs);
        model.initialize(&docs, &groups, &mut rng);
        for iter in 1..=200 {
            run_sweep_sage(&mut model, &docs, &groups, &mut rng);
            if iter > 50 && iter % 25 == 0 {
                assert!(optimize_kappa(&mut model, 20));
            }
        }
        model
    }

    #[test]
    fn jeffreys_is_scale_free_in_prior_variance() {
        // The Normal-Jeffreys prior is scale-free, so prior_variance must not affect
        // the fit (its reweight is 1/κ² and its warm-start is a fixed precision).
        let a = fit_prior_pv(SagePrior::Jeffreys, 1e-3);
        let b = fit_prior_pv(SagePrior::Jeffreys, 1e3);
        for g in 0..a.num_groups {
            for v in 0..a.num_types {
                assert!(
                    (a.kappa_c[g][v] - b.kappa_c[g][v]).abs() < 1e-9,
                    "Jeffreys κ_c depends on prior_variance at (g={g}, v={v}): {} vs {}",
                    a.kappa_c[g][v],
                    b.kappa_c[g][v]
                );
            }
        }
    }

    #[test]
    fn sparse_prior_shrinks_background_deviations_and_recovers() {
        let gaussian = fit_prior(SagePrior::Gaussian);
        let laplace = fit_prior(SagePrior::Laplace);

        // (a) recovery under the sparse prior: group 0 still favours {0,1}.
        let g0 = laplace.cell(0, 0);
        let g1 = laplace.cell(0, 1);
        assert!(
            laplace.beta[g0][0] + laplace.beta[g0][1] > laplace.beta[g1][0] + laplace.beta[g1][1],
            "Laplace did not recover the group discrimination"
        );

        // (b) the shared-background group deviations (words 4,5) are shrunk harder
        // by Laplace than by the Gaussian ridge.
        let bg_l1 = |m: &SageModel| -> f64 {
            (0..m.num_groups)
                .map(|g| m.kappa_c[g][4].abs() + m.kappa_c[g][5].abs())
                .sum()
        };
        let (l_bg, g_bg) = (bg_l1(&laplace), bg_l1(&gaussian));
        assert!(
            l_bg < g_bg,
            "Laplace background |κ_c| ({l_bg:.4}) not smaller than Gaussian ({g_bg:.4})"
        );
        // and overall the sparse fit has a smaller total κ_c L1 (sparser).
        let total_l1 =
            |m: &SageModel| -> f64 { m.kappa_c.iter().flatten().map(|x| x.abs()).sum::<f64>() };
        assert!(
            total_l1(&laplace) < total_l1(&gaussian),
            "Laplace total |κ_c| not smaller than Gaussian"
        );
    }

    #[test]
    fn optimize_kappa_rolls_back_on_non_finite() {
        // A degenerate (effectively zero) prior variance overflows inv_var to +inf,
        // so the L-BFGS solve returns non-finite. optimize_kappa must return false
        // and leave κ and β byte-for-byte unchanged (issue #422).
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut docs = Vec::new();
        let mut groups = Vec::new();
        for i in 0..40 {
            if i % 2 == 0 {
                docs.push(vec![0u32, 1, 0, 1]);
                groups.push(0usize);
            } else {
                docs.push(vec![2u32, 3, 2, 3]);
                groups.push(1usize);
            }
        }
        let mut model = SageModel::new(1, 2, 4, 0.1, 1.0, SagePrior::Gaussian);
        model.set_background(&docs);
        model.initialize(&docs, &groups, &mut rng);
        for _ in 0..30 {
            run_sweep_sage(&mut model, &docs, &groups, &mut rng);
        }
        assert!(optimize_kappa(&mut model, 20)); // a healthy update first

        let kappa_t_before = model.kappa_t.clone();
        let kappa_i_before = model.kappa_i.clone();
        let beta_before = model.beta.clone();

        model.prior_variance = 5e-324; // smallest subnormal -> 1/var = +inf
        assert!(
            !optimize_kappa(&mut model, 20),
            "expected a non-finite failure"
        );
        assert_eq!(model.kappa_t, kappa_t_before, "κT mutated on failure");
        assert_eq!(model.kappa_i, kappa_i_before, "κI mutated on failure");
        assert_eq!(model.beta, beta_before, "β mutated on failure");
    }

    #[test]
    fn set_background_is_the_unsmoothed_empirical_log_frequency() {
        // #422: the fixed background is the raw corpus log word-frequency
        // (Eisenstein et al.), not an add-one-smoothed distribution. Word 0 occurs
        // 3x and word 1 once out of 4 tokens, so m = [ln(3/4), ln(1/4)] exactly —
        // the previous +1 smoothing would have given ln(4/6), ln(2/6).
        let mut model = SageModel::new(1, 2, 2, 0.1, 1.0, SagePrior::Gaussian);
        let docs = vec![vec![0u32, 0, 0, 1]];
        model.set_background(&docs);
        assert!((model.m[0] - (3.0_f64 / 4.0).ln()).abs() < 1e-12);
        assert!((model.m[1] - (1.0_f64 / 4.0).ln()).abs() < 1e-12);
        // A vocabulary word absent from the corpus is floored to one count so the
        // background stays finite rather than -inf.
        let mut model2 = SageModel::new(1, 2, 3, 0.1, 1.0, SagePrior::Gaussian);
        model2.set_background(&docs); // word 2 never appears
        assert!(model2.m[2].is_finite());
        assert!((model2.m[2] - (1.0_f64 / 4.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn sage_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut docs = Vec::new();
        let mut groups = Vec::new();
        for i in 0..120 {
            if i % 2 == 0 {
                docs.push(vec![0u32, 1, 0, 1, 0, 1]);
                groups.push(0usize);
            } else {
                docs.push(vec![2u32, 3, 2, 3, 2, 3]);
                groups.push(1usize);
            }
        }
        let mut model = SageModel::new(1, 2, 4, 0.1, 1.0, SagePrior::Gaussian);
        model.set_background(&docs);
        model.initialize(&docs, &groups, &mut rng);
        for iter in 1..=200 {
            run_sweep_sage(&mut model, &docs, &groups, &mut rng);
            if iter > 50 && iter % 25 == 0 {
                assert!(optimize_kappa(&mut model, 20));
            }
        }
        let base = crate::conformance::check_conformance(&model);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&model);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }
}
