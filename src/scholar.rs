//! SCHOLAR (Card, Tan & Smith, "Neural Models for Documents with Metadata",
//! ACL 2018; reference `dallascard/scholar`, Apache-2.0) — the prior-covariate
//! ("prevalence") path.
//!
//! SCHOLAR extends a ProdLDA/AVITM VAE with document metadata in three roles. This
//! module implements the first: **prior covariates** `PC`, which shift the
//! document-topic *prior mean* by `mu_0[i] = W . PC[i]`. The KL then pulls each
//! document's posterior toward that covariate-dependent mean, so a covariate that
//! co-occurs with a topic raises that topic's prevalence — the neural analog of
//! STM/DMR prevalence covariates. `W` (the fitted `covariate_effects`) reads as a
//! covariate-by-topic prevalence-effect matrix. The reference flags this "upstream"
//! prior as future work in the paper (footnote 4) but implements it in code.
//!
//! We build directly on topica's existing ProdLDA VAE (`crate::prodlda`) rather than
//! re-deriving a VAE: the encoder, reparameterization, decoder, batchnorm, Adam, and
//! reconstruction loss are reused verbatim through the shared `batch_forward` /
//! `batch_backward`. The prior covariates enter in two faithful places:
//!   1. **Encoder input.** The reference concatenates `PC` to the encoder input; we
//!      route `PC` through the existing dense-embedding channel (`InputMode::BowEmb`
//!      with `emb_dim = n_prior_covars`), which is exactly "extra dense columns on
//!      the first encoder layer" — no encoder change.
//!   2. **Prior mean.** A new weight block `prior_w` (`K x n_prior_covars`, the
//!      reference's `Linear(n_prior_covars, K, bias=False)`) sets the per-document
//!      prior mean, threaded into the shared Gaussian KL via `Batch::prior_mus`. Its
//!      gradient comes back through the `batch_backward` `d_prior_mu` out-param and
//!      maps to `dW = sum_i d_prior_mu[i] (x) PC[i]` (plus the L2 prior penalty).
//!
//! Because topica's ProdLDA is the two-layer AVITM encoder (Srivastava & Sutton
//! 2017) rather than the reference's single embedding layer, this is a
//! mechanism-faithful port on topica's backbone, not a bit-for-bit clone of
//! `dallascard/scholar` — the same design choice topica's `ProdLDA` already makes.
//! The prior covariate path is Gaussian (logistic-normal) only: a prior-*mean* shift
//! is only defined for the `Prior::Laplace` latent, so Scholar fixes that prior.

use rand::Rng;

use crate::prodlda::{
    batch_backward, batch_forward, laplace_prior, normalized_bow, randn, raw_bow, Adam,
    AvitmOptions, Batch, BatchNorm, Grad, InputMode, Optim, Prior, ProdldaModel, Weights,
};

/// A fitted SCHOLAR (prior-covariate) model. `base` is the underlying ProdLDA VAE
/// (topic-word, doc-topic, encoder, mean-head batchnorm); `prior_w` is the covariate
/// weight matrix (`K x n_prior_covars`, row-major) that shifts the document prior
/// mean.
pub struct ScholarModel {
    pub base: ProdldaModel,
    /// Covariate weights `W`, `K x n_prior_covars` row-major (the reference's
    /// `prior_covar_weights.weight`): `prior_mean[i] = W . PC[i]`.
    pub prior_w: Vec<f64>,
    pub n_prior_covars: usize,
    pub l2_prior_reg: f64,
}

impl ScholarModel {
    pub fn num_topics(&self) -> usize {
        self.base.num_topics
    }

    /// Per-topic word distribution (softmax of the decoder rows), same as ProdLDA.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        self.base.topic_word()
    }

    /// The covariate-by-topic prevalence-effect matrix, `(n_prior_covars, K)`. Entry
    /// `[c][t]` is how much prior covariate `c` shifts the log-prior mean of topic
    /// `t`: positive raises topic `t`'s prevalence for documents with a high value on
    /// covariate `c`. This is `prior_w` transposed (the reference exposes the same,
    /// `prior_covar_weights.weight.T`).
    pub fn covariate_effects(&self) -> Vec<Vec<f64>> {
        let (k, pc) = (self.base.num_topics, self.n_prior_covars);
        (0..pc)
            .map(|c| (0..k).map(|t| self.prior_w[t * pc + c]).collect())
            .collect()
    }

    /// Held-out topic proportions for new documents given their prior covariates.
    /// The covariates enter the encoder (the dense channel) exactly as at fit time.
    pub fn transform(&self, docs: &[Vec<u32>], pcs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.base.transform_with_emb(docs, pcs)
    }
}

/// Compute the per-document prior mean `mu_0 = W . pc` (length `K`).
fn prior_mean_for(prior_w: &[f64], pc: &[f64], k: usize, n_pc: usize) -> Vec<f64> {
    (0..k)
        .map(|t| {
            let base = t * n_pc;
            (0..n_pc).map(|c| prior_w[base + c] * pc[c]).sum()
        })
        .collect()
}

/// Fit SCHOLAR with prior covariates. `docs` are token-id documents; `pcs[i]` is the
/// dense prior-covariate row for document `i` (length `n_prior_covars`, the same for
/// every document). `alpha` is the symmetric Dirichlet concentration behind the
/// Laplace prior variance; `l2_prior_reg` is the L2 penalty on `W`. The training
/// loop mirrors `prodlda::fit_avitm` (same shuffle, noise, dropout, Adam, batchnorm)
/// and additionally updates `prior_w` from the per-document prior-mean gradient.
#[allow(clippy::too_many_arguments)]
pub fn fit_scholar<R: Rng>(
    docs: &[Vec<u32>],
    pcs: &[Vec<f64>],
    num_topics: usize,
    num_types: usize,
    n_prior_covars: usize,
    hidden: usize,
    alpha: f64,
    dropout: f64,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    l2_prior_reg: f64,
    em_tol: f64,
    rng: &mut R,
) -> ScholarModel {
    let (k, v, pc) = (num_topics, num_types, n_prior_covars);
    let d = docs.len();
    let xn: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| normalized_bow(doc)).collect();
    let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| raw_bow(doc)).collect();
    let totals: Vec<f64> = bows
        .iter()
        .map(|b| b.iter().map(|&(_, c)| c).sum())
        .collect();

    let alpha_vec = vec![alpha; k];
    // Shared Laplace prior. Only `prior_var` matters here (the mean is per-document
    // via `prior_w`); `prior_mu` stays as the fallback the shared code expects.
    let (prior_mu, prior_var) = laplace_prior(&alpha_vec);
    let keep = (1.0 - dropout).max(1e-6);
    let opts = AvitmOptions {
        prior: Prior::Laplace,
        ..AvitmOptions::default()
    };

    // Prior covariates ride the dense-embedding channel into the encoder.
    let mut w = Weights::new(v, pc, hidden, k, InputMode::BowEmb, rng);
    // Covariate weight block: zero init, so topics start covariate-agnostic and the
    // prevalence effect is learned from data (deterministic, no extra RNG draws).
    let mut prior_w = vec![0.0; k * pc];
    let mut prior_w_opt = Adam::new(prior_w.len(), lr, 0.99, 0.0);

    let mut bn_mu = BatchNorm::new(k);
    let mut bn_lv = BatchNorm::new(k);
    let mut bn_dec = BatchNorm::new(v);
    let mut opt = Optim::new(&w, lr, 0.99, 0.0);

    let mut bound_history: Vec<f64> = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = 0usize;
    let mut order: Vec<usize> = (0..d).collect();

    for epoch in 0..epochs {
        epochs_run = epoch + 1;
        // Deterministic Fisher-Yates shuffle from the seeded rng.
        for i in (1..d).rev() {
            let j = (rng.gen::<f64>() * (i + 1) as f64) as usize;
            order.swap(i, j.min(i));
        }

        let mut epoch_loss = 0.0;
        let mut batches = 0usize;
        for chunk in order.chunks(batch_size.max(2)) {
            let n = chunk.len();
            if n < 2 {
                continue; // batchnorm needs at least two documents
            }
            let eps: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..k).map(|_| randn(rng)).collect())
                .collect();
            let masks2: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..hidden)
                        .map(|_| {
                            if rng.gen::<f64>() < keep {
                                1.0 / keep
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            let masks_t: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..k)
                        .map(|_| {
                            if rng.gen::<f64>() < keep {
                                1.0 / keep
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            // Per-document prior means from the current covariate weights.
            let prior_mus: Vec<Vec<f64>> = chunk
                .iter()
                .map(|&di| prior_mean_for(&prior_w, &pcs[di], k, pc))
                .collect();

            let batch = Batch {
                xns: chunk.iter().map(|&di| xn[di].as_slice()).collect(),
                embs: chunk.iter().map(|&di| pcs[di].as_slice()).collect(),
                counts: chunk.iter().map(|&di| bows[di].as_slice()).collect(),
                totals: chunk.iter().map(|&di| totals[di]).collect(),
                eps: &eps,
                masks2: &masks2,
                masks_t: &masks_t,
                prior_mus: Some(&prior_mus),
            };

            let (loss, cache, stats) = batch_forward(
                &w, &bn_mu, &bn_lv, &bn_dec, &prior_mu, &prior_var, &alpha_vec, &opts, &batch,
            );
            bn_mu.update_running(&stats[0].0, &stats[0].1);
            bn_lv.update_running(&stats[1].0, &stats[1].1);
            bn_dec.update_running(&stats[2].0, &stats[2].1);

            let mut g = Grad::zeros(&w);
            let mut d_prior_mu = vec![vec![0.0; k]; n];
            batch_backward(
                &w,
                &prior_mu,
                &prior_var,
                &alpha_vec,
                &opts,
                &batch,
                &cache,
                &mut g,
                Some(&mut d_prior_mu),
            );
            g.scale(1.0 / n as f64);
            opt.step(&mut w, &g);

            // Covariate-weight gradient: dW[t][c] = (1/n) sum_i d_prior_mu[i][t]*PC[i][c]
            // (data term, averaged to match the batch mean) plus the un-averaged L2
            // prior penalty 2*l2*W (matching the reference, whose reg is added after
            // the per-document mean).
            let mut g_prior_w = vec![0.0; k * pc];
            for (li, &di) in chunk.iter().enumerate() {
                let pc_row = &pcs[di];
                for t in 0..k {
                    let dpm = d_prior_mu[li][t];
                    let base = t * pc;
                    for c in 0..pc {
                        g_prior_w[base + c] += dpm * pc_row[c];
                    }
                }
            }
            let inv_n = 1.0 / n as f64;
            for idx in 0..g_prior_w.len() {
                g_prior_w[idx] = g_prior_w[idx] * inv_n + 2.0 * l2_prior_reg * prior_w[idx];
            }
            prior_w_opt.step(&mut prior_w, &g_prior_w);

            epoch_loss += loss / n as f64;
            batches += 1;
        }

        let avg = epoch_loss / batches.max(1) as f64;
        bound_history.push(-avg); // report the ELBO (negative loss)
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (-avg - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }
    }

    let base = ProdldaModel {
        num_topics: k,
        num_types: v,
        doc_topic: Vec::new(),
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        epochs_run,
        weights: w,
        bn_mu,
        prior: Prior::Laplace,
    };
    let doc_topic = base.transform_with_emb(docs, pcs);
    let base = ProdldaModel { doc_topic, ..base };
    ScholarModel {
        base,
        prior_w,
        n_prior_covars: pc,
        l2_prior_reg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Recompute the summed batch loss (including the L2 prior penalty on W) for a
    // given (weights, prior_w), at fixed noise and all-ones dropout. This is the
    // scalar objective the analytic prior_w gradient is finite-differenced against.
    #[allow(clippy::too_many_arguments)]
    fn scholar_batch_loss(
        w: &Weights,
        prior_w: &[f64],
        prior_var: &[f64],
        alpha: &[f64],
        opts: &AvitmOptions,
        pcs: &[Vec<f64>],
        l2: f64,
        k: usize,
        n_pc: usize,
        base_batch: &BatchBuild,
    ) -> f64 {
        let prior_mus: Vec<Vec<f64>> = pcs
            .iter()
            .map(|pc| prior_mean_for(prior_w, pc, k, n_pc))
            .collect();
        let batch = base_batch.as_batch(Some(&prior_mus));
        let bn_mu = BatchNorm::new(w.k);
        let bn_lv = BatchNorm::new(w.k);
        let bn_dec = BatchNorm::new(w.v);
        // The shared prior_mu slice is unused when prior_mus is Some; pass zeros.
        let prior_mu = vec![0.0; k];
        let (loss, _, _) = batch_forward(
            w, &bn_mu, &bn_lv, &bn_dec, &prior_mu, prior_var, alpha, opts, &batch,
        );
        let reg: f64 = prior_w.iter().map(|&x| x * x).sum();
        loss + l2 * reg
    }

    // Owns the batch inputs so we can rebuild the borrowed `Batch` at each FD probe.
    struct BatchBuild {
        xns: Vec<Vec<(usize, f64)>>,
        pcs: Vec<Vec<f64>>,
        bows: Vec<Vec<(usize, f64)>>,
        totals: Vec<f64>,
        eps: Vec<Vec<f64>>,
        masks2: Vec<Vec<f64>>,
        masks_t: Vec<Vec<f64>>,
    }
    impl BatchBuild {
        fn as_batch<'a>(&'a self, prior_mus: Option<&'a [Vec<f64>]>) -> Batch<'a> {
            Batch {
                xns: self.xns.iter().map(|x| x.as_slice()).collect(),
                embs: self.pcs.iter().map(|x| x.as_slice()).collect(),
                counts: self.bows.iter().map(|b| b.as_slice()).collect(),
                totals: self.totals.clone(),
                eps: &self.eps,
                masks2: &self.masks2,
                masks_t: &self.masks_t,
                prior_mus,
            }
        }
    }

    // Finite-difference check of the covariate-weight (prior_w) gradient, including
    // the L2 prior penalty. The rest of the VAE gradient is already covered by
    // prodlda's FD tests; this pins the SCHOLAR-specific path.
    #[test]
    fn prior_w_gradient_matches_fd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (v, hidden, k, pc) = (7usize, 5usize, 4usize, 3usize);
        let l2 = 0.05;
        let w = Weights::new(v, pc, hidden, k, InputMode::BowEmb, &mut rng);
        let alpha = vec![1.0; k];
        let (_, prior_var) = laplace_prior(&alpha);
        let opts = AvitmOptions {
            prior: Prior::Laplace,
            ..AvitmOptions::default()
        };

        let docs: Vec<Vec<u32>> = vec![
            vec![0, 0, 2, 3, 6],
            vec![1, 4, 4, 5],
            vec![2, 2, 3, 5, 6, 0],
        ];
        let n = docs.len();
        let pcs: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..pc)
                    .map(|c| 0.4 * (i as f64 + 1.0) - 0.23 * c as f64 + 0.1)
                    .collect()
            })
            .collect();
        // A non-trivial covariate weight so both the KL and the L2 term have signal.
        let mut prior_w: Vec<f64> = (0..k * pc)
            .map(|idx| 0.2 * ((idx % 5) as f64) - 0.3)
            .collect();

        let bb = BatchBuild {
            xns: docs.iter().map(|d| normalized_bow(d)).collect(),
            pcs: pcs.clone(),
            bows: docs.iter().map(|d| raw_bow(d)).collect(),
            totals: docs
                .iter()
                .map(|d| raw_bow(d).iter().map(|&(_, c)| c).sum())
                .collect(),
            eps: (0..n)
                .map(|i| {
                    (0..k)
                        .map(|t| 0.1 * (i as f64 + 1.0) - 0.05 * t as f64)
                        .collect()
                })
                .collect(),
            masks2: vec![vec![1.0; hidden]; n],
            masks_t: vec![vec![1.0; k]; n],
        };

        // Analytic prior_w gradient.
        let prior_mus: Vec<Vec<f64>> = pcs
            .iter()
            .map(|p| prior_mean_for(&prior_w, p, k, pc))
            .collect();
        let batch = bb.as_batch(Some(&prior_mus));
        let bn_mu = BatchNorm::new(k);
        let bn_lv = BatchNorm::new(k);
        let bn_dec = BatchNorm::new(v);
        let prior_mu0 = vec![0.0; k];
        let (_, cache, _) = batch_forward(
            &w, &bn_mu, &bn_lv, &bn_dec, &prior_mu0, &prior_var, &alpha, &opts, &batch,
        );
        let mut g = Grad::zeros(&w);
        let mut d_prior_mu = vec![vec![0.0; k]; n];
        batch_backward(
            &w,
            &prior_mu0,
            &prior_var,
            &alpha,
            &opts,
            &batch,
            &cache,
            &mut g,
            Some(&mut d_prior_mu),
        );
        // Map to dW (summed over docs, NOT averaged — the loss here is the summed
        // batch loss) and add the L2 penalty gradient.
        let mut analytic = vec![0.0; k * pc];
        for (li, p) in pcs.iter().enumerate() {
            for t in 0..k {
                let base = t * pc;
                for c in 0..pc {
                    analytic[base + c] += d_prior_mu[li][t] * p[c];
                }
            }
        }
        for idx in 0..analytic.len() {
            analytic[idx] += 2.0 * l2 * prior_w[idx];
        }

        let fd = 1e-6;
        let mut max_abs = 0.0f64;
        for idx in 0..prior_w.len() {
            let orig = prior_w[idx];
            prior_w[idx] = orig + fd;
            let lp = scholar_batch_loss(
                &w, &prior_w, &prior_var, &alpha, &opts, &pcs, l2, k, pc, &bb,
            );
            prior_w[idx] = orig - fd;
            let lm = scholar_batch_loss(
                &w, &prior_w, &prior_var, &alpha, &opts, &pcs, l2, k, pc, &bb,
            );
            prior_w[idx] = orig;
            let num = (lp - lm) / (2.0 * fd);
            let abs_err = (analytic[idx] - num).abs();
            if abs_err > max_abs {
                max_abs = abs_err;
            }
            assert!(
                abs_err < 1e-4,
                "prior_w[{idx}]: analytic {} vs numeric {}",
                analytic[idx],
                num
            );
        }
        assert!(max_abs < 1e-4, "max abs error {max_abs}");
    }

    // A covariate that shifts topic prevalence should be recovered: fit on a planted
    // corpus where documents split into two covariate groups with different topic
    // mixes, and check the covariate effect separates the topics in the right
    // direction, and that topics and proportions are valid.
    #[test]
    fn fit_recovers_covariate_prevalence() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        // Two word-blocks -> two topics. Group A (covar 0) loads block 0; group B
        // (covar 1) loads block 1.
        let (k, v) = (2usize, 8usize);
        let block0: Vec<u32> = vec![0, 1, 2, 3];
        let block1: Vec<u32> = vec![4, 5, 6, 7];
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut pcs: Vec<Vec<f64>> = Vec::new();
        for g in 0..2 {
            let block = if g == 0 { &block0 } else { &block1 };
            for _ in 0..40 {
                let mut doc = Vec::new();
                for _ in 0..30 {
                    // 85% from the group's block, 15% from the other.
                    let from_block = if rng.gen::<f64>() < 0.85 {
                        block
                    } else if g == 0 {
                        &block1
                    } else {
                        &block0
                    };
                    let w = from_block[(rng.gen::<f64>() * from_block.len() as f64) as usize];
                    doc.push(w);
                }
                docs.push(doc);
                pcs.push(if g == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                });
            }
        }

        let m = fit_scholar(
            &docs, &pcs, k, v, 2, 20, 1.0, 0.2, 120, 40, 0.01, 0.0, 0.0, &mut rng,
        );

        // Topic-word rows are valid distributions.
        let tw = m.topic_word();
        assert_eq!(tw.len(), k);
        for row in &tw {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "topic row sums to {s}");
        }
        // covariate_effects is (n_prior_covars, K).
        let eff = m.covariate_effects();
        assert_eq!(eff.len(), 2);
        assert_eq!(eff[0].len(), k);

        // Identify which topic is block-0 dominated (highest mass on words 0..3).
        let block0_topic = if tw[0][0..4].iter().sum::<f64>() > tw[1][0..4].iter().sum::<f64>() {
            0
        } else {
            1
        };
        let block1_topic = 1 - block0_topic;
        // Covariate 0 (group A) should raise the block-0 topic's prevalence relative
        // to the block-1 topic; covariate 1 the reverse. Compare the contrast.
        let c0 = eff[0][block0_topic] - eff[0][block1_topic];
        let c1 = eff[1][block0_topic] - eff[1][block1_topic];
        assert!(
            c0 > c1,
            "covariate prevalence not recovered: covar0 contrast {c0} !> covar1 contrast {c1}"
        );
    }

    #[test]
    fn fit_is_deterministic() {
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 2, 0, 1],
            vec![3, 4, 5, 3],
            vec![0, 2, 4, 1, 5],
            vec![1, 1, 3, 5, 2],
        ];
        let pcs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(3);
            fit_scholar(
                &docs, &pcs, 2, 6, 2, 8, 1.0, 0.2, 15, 4, 0.01, 0.01, 0.0, &mut rng,
            )
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word(), b.topic_word());
        assert_eq!(a.covariate_effects(), b.covariate_effects());
        assert_eq!(a.base.doc_topic, b.base.doc_topic);
    }
}
