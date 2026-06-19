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
//! The κ are MAP-estimated (Gaussian/L2 prior) from the topic×group×word counts
//! between sampling sweeps, via the same L-BFGS used for DMR.

use rand::Rng;

use crate::variational::lbfgs_minimize;

/// SAGE model state. Counts are dense over (topic, group, word).
pub struct SageModel {
    pub num_topics: usize,
    pub num_groups: usize,
    pub num_types: usize,
    pub alpha: Vec<f64>,
    pub prior_variance: f64,

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
    ) -> Self {
        let kg = num_topics * num_groups;
        SageModel {
            num_topics,
            num_groups,
            num_types,
            alpha: vec![alpha; num_topics],
            prior_variance,
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

    /// Set the background `m_v` to the corpus log word-frequency.
    pub fn set_background(&mut self, docs: &[Vec<u32>]) {
        let mut freq = vec![1.0f64; self.num_types]; // +1 smoothing
        let mut total = self.num_types as f64;
        for doc in docs {
            for &w in doc {
                freq[w as usize] += 1.0;
                total += 1.0;
            }
        }
        for v in 0..self.num_types {
            self.m[v] = (freq[v] / total).ln();
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

/// MAP-estimate the κ deviations (Gaussian prior) from the current counts, then
/// refresh the cached β. One L-BFGS run.
pub fn optimize_kappa(model: &mut SageModel, max_iter: usize) {
    let k_n = model.num_topics;
    let g_n = model.num_groups;
    let v_n = model.num_types;
    let sigma2 = model.prior_variance;

    // Pack κ into a flat vector: [κT (K*V) | κC (G*V) | κI (K*G*V)].
    let n_t = k_n * v_n;
    let n_c = g_n * v_n;
    let n_i = k_n * g_n * v_n;
    let mut x0 = Vec::with_capacity(n_t + n_c + n_i);
    for k in 0..k_n {
        x0.extend_from_slice(&model.kappa_t[k]);
    }
    for g in 0..g_n {
        x0.extend_from_slice(&model.kappa_c[g]);
    }
    for c in 0..(k_n * g_n) {
        x0.extend_from_slice(&model.kappa_i[c]);
    }

    let m = &model.m;
    let counts = &model.counts;
    let totals = &model.totals;
    let inv_var = 1.0 / sigma2;

    let x = lbfgs_minimize(
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
                    // β_{k,g,·} and log Z.
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

            // Gaussian prior.
            for (i, &xi) in flat.iter().enumerate() {
                value -= 0.5 * inv_var * xi * xi;
                grad[i] -= inv_var * xi;
            }

            // Minimize the negative.
            (-value, grad.iter().map(|gv| -gv).collect())
        },
        max_iter,
        7,
        1e-4,
    );

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

        let mut model = SageModel::new(1, 2, 4, 0.1, 1.0);
        model.set_background(&docs);
        model.initialize(&docs, &groups, &mut rng);
        for iter in 1..=200 {
            run_sweep_sage(&mut model, &docs, &groups, &mut rng);
            if iter > 50 && iter % 25 == 0 {
                optimize_kappa(&mut model, 20);
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
        let mut model = SageModel::new(1, 2, 4, 0.1, 1.0);
        model.set_background(&docs);
        model.initialize(&docs, &groups, &mut rng);
        for iter in 1..=200 {
            run_sweep_sage(&mut model, &docs, &groups, &mut rng);
            if iter > 50 && iter % 25 == 0 {
                optimize_kappa(&mut model, 20);
            }
        }
        let base = crate::conformance::check_conformance(&model);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&model);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }
}
