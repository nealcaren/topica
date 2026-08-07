//! ContextualSTM (experimental): a contextual sentence-embedding topic model with
//! STM/SCHOLAR-style prevalence covariates.
//!
//! This composes two already-ported halves of topica's ProdLDA/AVITM VAE
//! (`crate::prodlda`):
//!
//!   * the **contextual encoder** of CombinedTM / ZeroShotTM (Bianchi, Terragni &
//!     Hovy 2021) — the document sentence embedding drives inference, via
//!     [`InputMode::BowEmbAdapt`] (CombinedTM: the embedding is projected into
//!     vocabulary space by the learned `adapt_bert` layer, then concatenated with the
//!     raw bag of words) or [`InputMode::EmbOnly`] (ZeroShotTM: the embedding alone);
//!   * SCHOLAR's **prevalence-covariate prior** (Card, Tan & Smith 2018) — a weight
//!     block `prior_w` (`K x n_covariates`, the reference's
//!     `Linear(n_covariates, K, bias=False)`) sets a per-document prior *mean*
//!     `mu_0 = W . covariates`, so the Gaussian KL pulls each document's posterior
//!     toward the covariate-implied mean and a covariate that co-occurs with a topic
//!     raises that topic's prevalence. The fitted `W` is `covariate_effects`.
//!
//! No published implementation combines a contextual encoder with covariates, so this
//! is a topica-original composition; both halves are individually validated
//! (CombinedTM/ZeroShotTM parity, SCHOLAR covariate recovery). Gated experimental.
//!
//! **Covariate flow** is controlled by [`CovariateMode`]:
//!   * [`CovariateMode::PriorOnly`] — covariates enter *only* the prior (`prior_w`);
//!     the encoder reads the sentence embedding alone. This is the STM-purist reading
//!     (STM has no encoder; prevalence is a pure prior regression). The returned
//!     `doc_topic` is `q(theta | embedding)` and is not covariate-adjusted.
//!   * [`CovariateMode::EncoderPrior`] (default) — covariates enter the prior *and*
//!     the encoder, matching SCHOLAR (which anchors the covariate weights to the data
//!     likelihood by also feeding covariates to the encoder, giving cleaner recovery).
//!     Covariates are concatenated into the encoder's dense channel: for the ZeroShot
//!     (`EmbOnly`) encoder they concatenate raw (exactly SCHOLAR-style); for the
//!     Combined (`BowEmbAdapt`) encoder they pass through the `adapt_bert` projection
//!     alongside the embedding (a richer, word-space covariate loading).
//!
//! The prior is Gaussian (logistic-normal) only: a prior-*mean* shift is defined only
//! for [`Prior::Laplace`], mirroring SCHOLAR. `covariate_effects` is a *point*
//! estimate of `W` (no uncertainty), on the standardized-logit latent scale (the mean
//! head is affine-free batchnormed): it is a partial effect on the log-prior mean, not
//! a proportion change, and magnitudes are not directly comparable across topics. For
//! proportion-scale prevalence effects with honest uncertainty, run the shared
//! `estimate_effect` on the fitted `doc_topic`.

use rand::Rng;

use crate::prodlda::{
    batch_backward, batch_forward, laplace_prior, normalized_bow, randn, raw_bow, Adam,
    AvitmOptions, Batch, BatchNorm, Grad, InputMode, Optim, Prior, ProdldaModel, Weights,
};

/// Which contextual encoder drives inference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EncoderKind {
    /// CombinedTM: `adapt_bert`-projected embedding concatenated with the raw BoW
    /// ([`InputMode::BowEmbAdapt`]).
    Combined,
    /// ZeroShotTM: the document embedding alone ([`InputMode::EmbOnly`]).
    ZeroShot,
}

impl EncoderKind {
    pub fn input_mode(self) -> InputMode {
        match self {
            EncoderKind::Combined => InputMode::BowEmbAdapt,
            EncoderKind::ZeroShot => InputMode::EmbOnly,
        }
    }
}

/// How covariates flow into the model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CovariateMode {
    /// Covariates enter the prior *and* the encoder (SCHOLAR-style; default).
    EncoderPrior,
    /// Covariates enter *only* the prior (STM-purist).
    PriorOnly,
}

/// A fitted ContextualSTM model. `base` is the underlying ProdLDA VAE (topic-word,
/// doc-topic, encoder, mean-head batchnorm); `prior_w` is the covariate weight matrix
/// (`K x n_covariates`) that shifts the document prior mean.
pub struct ContextualStmModel {
    pub base: ProdldaModel,
    /// Covariate weights `W`, `K x n_covariates` row-major (SCHOLAR's
    /// `prior_covar_weights.weight`): `prior_mean[i] = W . covariates[i]`.
    pub prior_w: Vec<f64>,
    pub n_covariates: usize,
    pub l2_prior_reg: f64,
    pub encoder: EncoderKind,
    pub covariate_mode: CovariateMode,
    /// The sentence-embedding width `E` (excluding any covariate columns folded into
    /// the encoder channel), needed to rebuild encoder features at `transform` time.
    pub emb_dim: usize,
}

impl ContextualStmModel {
    pub fn num_topics(&self) -> usize {
        self.base.num_topics
    }

    /// Per-topic word distribution (softmax of the decoder rows), same as ProdLDA.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        self.base.topic_word()
    }

    /// The document-topic matrix from the fit (`q(theta | embedding)`).
    pub fn doc_topic(&self) -> &[Vec<f64>] {
        &self.base.doc_topic
    }

    /// The covariate-by-topic prevalence-effect matrix, `(n_covariates, K)`. Entry
    /// `[c][t]` is how much covariate `c` shifts the log-prior mean of topic `t`:
    /// positive raises topic `t`'s prevalence for documents high on covariate `c`.
    /// This is `prior_w` transposed (SCHOLAR exposes the same). A *point* estimate on
    /// the standardized-logit scale — not a proportion change.
    pub fn covariate_effects(&self) -> Vec<Vec<f64>> {
        let (k, c) = (self.base.num_topics, self.n_covariates);
        (0..c)
            .map(|cc| (0..k).map(|t| self.prior_w[t * c + cc]).collect())
            .collect()
    }

    /// Build the encoder's dense channel for each document: the sentence embedding
    /// under [`CovariateMode::PriorOnly`], or the embedding concatenated with the
    /// covariates under [`CovariateMode::EncoderPrior`].
    fn encoder_feats(&self, doc_embeddings: &[Vec<f64>], covariates: &[Vec<f64>]) -> Vec<Vec<f64>> {
        encoder_feats(self.covariate_mode, doc_embeddings, covariates)
    }

    /// Held-out topic proportions for new documents given their embeddings and
    /// covariates. Covariates are always observed (prevalence is defined at test), so
    /// the encoder path matches fit time.
    pub fn transform(
        &self,
        docs: &[Vec<u32>],
        doc_embeddings: &[Vec<f64>],
        covariates: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let feats = self.encoder_feats(doc_embeddings, covariates);
        self.base.transform_with_emb(docs, &feats)
    }
}

/// The encoder dense channel per covariate mode (free function so `fit` can reuse it
/// before the model exists).
fn encoder_feats(
    mode: CovariateMode,
    doc_embeddings: &[Vec<f64>],
    covariates: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    match mode {
        CovariateMode::PriorOnly => doc_embeddings.to_vec(),
        CovariateMode::EncoderPrior => doc_embeddings
            .iter()
            .zip(covariates.iter())
            .map(|(emb, cov)| {
                let mut f = emb.clone();
                f.extend_from_slice(cov);
                f
            })
            .collect(),
    }
}

/// Per-column mean and population standard deviation of a covariate matrix
/// (`D x C`). Returns `(mean, std)`, each length `C`. A column with (near-)zero
/// variance yields `std == 0`, which the caller should treat as a constant column.
pub fn column_stats(cov: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let d = cov.len();
    let c = cov.first().map(|r| r.len()).unwrap_or(0);
    let mut mean = vec![0.0; c];
    let mut std = vec![0.0; c];
    if d == 0 {
        return (mean, std);
    }
    for row in cov {
        for (j, &x) in row.iter().enumerate() {
            mean[j] += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= d as f64;
    }
    for row in cov {
        for (j, &x) in row.iter().enumerate() {
            let dv = x - mean[j];
            std[j] += dv * dv;
        }
    }
    for s in std.iter_mut() {
        *s = (*s / d as f64).sqrt();
    }
    (mean, std)
}

/// Standardize a covariate matrix column-wise with the given `mean`/`std`. A column
/// whose `std` is (near-)zero is only centered (divided by 1), avoiding a divide by
/// zero; constant columns should be rejected by the caller before fitting.
pub fn standardize_with(cov: &[Vec<f64>], mean: &[f64], std: &[f64]) -> Vec<Vec<f64>> {
    cov.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &x)| {
                    let s = if std[j] > 1e-12 { std[j] } else { 1.0 };
                    (x - mean[j]) / s
                })
                .collect()
        })
        .collect()
}

/// Numeric rank of a covariate matrix (`D x C`) via Gaussian elimination with partial
/// pivoting on its columns, used to detect collinearity (e.g. full dummy coding plus
/// an intercept). Rank `< C` means the covariate weights are unidentified without a
/// ridge penalty.
pub fn covariate_rank(cov: &[Vec<f64>]) -> usize {
    let d = cov.len();
    let c = cov.first().map(|r| r.len()).unwrap_or(0);
    if d == 0 || c == 0 {
        return 0;
    }
    // Work on the transpose (C column-vectors of length D) so we can row-reduce the
    // C columns against each other.
    let mut m: Vec<Vec<f64>> = (0..c)
        .map(|j| (0..d).map(|i| cov[i][j]).collect::<Vec<f64>>())
        .collect();
    let mut rank = 0usize;
    let mut used = vec![false; d];
    for col in 0..c {
        // Find a pivot coordinate in the current column vector `m[col]`.
        let mut pivot = None;
        let mut best = 1e-9;
        for i in 0..d {
            if !used[i] && m[col][i].abs() > best {
                best = m[col][i].abs();
                pivot = Some(i);
            }
        }
        let Some(p) = pivot else { continue };
        used[p] = true;
        rank += 1;
        let pv = m[col][p];
        // Eliminate coordinate `p` from the remaining column vectors.
        for other in (col + 1)..c {
            let factor = m[other][p] / pv;
            if factor != 0.0 {
                for i in 0..d {
                    m[other][i] -= factor * m[col][i];
                }
            }
        }
    }
    rank
}

/// Compute the per-document prior mean `mu_0 = W . cov` (length `K`), `W` row-major
/// `K x C`.
fn prior_mean_for(prior_w: &[f64], cov: &[f64], k: usize, c: usize) -> Vec<f64> {
    (0..k)
        .map(|t| {
            let base = t * c;
            (0..c).map(|cc| prior_w[base + cc] * cov[cc]).sum()
        })
        .collect()
}

/// Fit ContextualSTM. `doc_embeddings` is `D x emb_dim` (the sentence embeddings);
/// `covariates` is `D x n_covariates` (numeric, and — for stable, scale-invariant
/// effects — standardized by the caller). The training loop mirrors
/// `prodlda::fit_avitm` (same shuffle, noise, dropout, Adam, batchnorm) and adds
/// SCHOLAR's covariate prior-mean update. Laplace prior only.
#[allow(clippy::too_many_arguments)]
pub fn fit_contextual_stm<R: Rng>(
    docs: &[Vec<u32>],
    doc_embeddings: &[Vec<f64>],
    covariates: &[Vec<f64>],
    num_topics: usize,
    num_types: usize,
    n_covariates: usize,
    emb_dim: usize,
    hidden: usize,
    alpha: f64,
    dropout: f64,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    l2_prior_reg: f64,
    em_tol: f64,
    encoder: EncoderKind,
    covariate_mode: CovariateMode,
    rng: &mut R,
) -> ContextualStmModel {
    let (k, v, c) = (num_topics, num_types, n_covariates);
    let d = docs.len();

    // The encoder's dense channel: embedding alone (prior-only) or embedding ⊕
    // covariates (encoder+prior). Its width is what `Weights::new` sizes `w1`/`w_adapt`
    // against.
    let feats = encoder_feats(covariate_mode, doc_embeddings, covariates);
    let enc_dim = match covariate_mode {
        CovariateMode::PriorOnly => emb_dim,
        CovariateMode::EncoderPrior => emb_dim + c,
    };

    let xn: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| normalized_bow(doc)).collect();
    let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| raw_bow(doc)).collect();
    let totals: Vec<f64> = bows
        .iter()
        .map(|b| b.iter().map(|&(_, cnt)| cnt).sum())
        .collect();

    let alpha_vec = vec![alpha; k];
    // Shared Laplace prior. Only `prior_var` matters (the mean is per-document via
    // `prior_w`); `prior_mu` is the fallback the shared code expects.
    let (prior_mu, prior_var) = laplace_prior(&alpha_vec);
    let keep = (1.0 - dropout).max(1e-6);
    let opts = AvitmOptions {
        prior: Prior::Laplace,
        ..AvitmOptions::default()
    };

    let mut w = Weights::new(v, enc_dim, hidden, k, encoder.input_mode(), rng);
    // Covariate weight block: zero init, so topics start covariate-agnostic and the
    // prevalence effect is learned from data (deterministic, no extra RNG draws).
    let mut prior_w = vec![0.0; k * c];
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
                .map(|&di| prior_mean_for(&prior_w, &covariates[di], k, c))
                .collect();

            let batch = Batch {
                xns: chunk.iter().map(|&di| xn[di].as_slice()).collect(),
                embs: chunk.iter().map(|&di| feats[di].as_slice()).collect(),
                counts: chunk.iter().map(|&di| bows[di].as_slice()).collect(),
                totals: chunk.iter().map(|&di| totals[di]).collect(),
                eps: &eps,
                masks2: &masks2,
                masks_t: &masks_t,
                prior_mus: Some(&prior_mus),
            };

            let (loss, cache, stats) = batch_forward(
                &w, &bn_mu, &bn_lv, &bn_dec, &prior_mu, &prior_var, &alpha_vec, &opts, &batch, None,
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
                None,
                None,
            );
            g.scale(1.0 / n as f64);
            opt.step(&mut w, &g);

            // Covariate-weight gradient: dW[t][c] = (1/n) sum_i d_prior_mu[i][t]*cov[i][c]
            // (data term, batch-mean) plus the un-averaged L2 prior penalty 2*l2*W
            // (matching SCHOLAR, whose reg is added after the per-document mean).
            let mut g_prior_w = vec![0.0; k * c];
            for (li, &di) in chunk.iter().enumerate() {
                let cov_row = &covariates[di];
                for t in 0..k {
                    let dpm = d_prior_mu[li][t];
                    let base = t * c;
                    for cc in 0..c {
                        g_prior_w[base + cc] += dpm * cov_row[cc];
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
        // Laplace transform is softmax(mu) and never reads bn_lv (#428).
        bn_lv: None,
        prior: Prior::Laplace,
    };
    let doc_topic = base.transform_with_emb(docs, &feats);
    let base = ProdldaModel { doc_topic, ..base };
    ContextualStmModel {
        base,
        prior_w,
        n_covariates: c,
        l2_prior_reg,
        encoder,
        covariate_mode,
        emb_dim,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a planted two-block corpus: `group` selects which word block dominates.
    /// Returns (docs, embeddings, covariates) with a one-hot covariate per group.
    fn planted_corpus(
        n_per_group: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // V = 8, two blocks {0..4} and {4..8}. Group 0 loads block 0, group 1 block 1
        // at an 85/15 mix. Embedding = normalized block-count signature (informative),
        // covariate = one-hot group.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut docs = Vec::new();
        let mut embs = Vec::new();
        let mut covs = Vec::new();
        for group in 0..2usize {
            for _ in 0..n_per_group {
                let mut doc = Vec::new();
                let mut blk = [0.0f64; 2];
                for _ in 0..30 {
                    let major = rng.gen::<f64>() < 0.85;
                    let use_block0 = if major { group == 0 } else { group == 1 };
                    let b = if use_block0 { 0 } else { 1 };
                    blk[b] += 1.0;
                    let w = (b * 4) as u32 + (rng.gen::<f64>() * 4.0) as u32;
                    doc.push(w.min(7));
                }
                let s = blk[0] + blk[1];
                embs.push(vec![blk[0] / s, blk[1] / s]);
                covs.push(if group == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                });
                docs.push(doc);
            }
        }
        (docs, embs, covs)
    }

    fn fit_small(mode: CovariateMode, enc: EncoderKind, seed: u64) -> ContextualStmModel {
        let (docs, embs, covs) = planted_corpus(40, 7);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        fit_contextual_stm(
            &docs, &embs, &covs, 2, 8, 2, 2, 20, 1.0, 0.2, 120, 40, 0.01, 0.0, 0.0, enc, mode,
            &mut rng,
        )
    }

    #[test]
    fn recovers_covariate_prevalence_encoder_prior() {
        let m = fit_small(CovariateMode::EncoderPrior, EncoderKind::Combined, 1);
        // Topic rows are valid simplices.
        for row in m.topic_word() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "topic row must sum to 1, got {s}");
            assert!(row.iter().all(|&x| x >= 0.0));
        }
        let eff = m.covariate_effects();
        assert_eq!(eff.len(), 2);
        assert_eq!(eff[0].len(), 2);
        // Which topic does covariate 0 (group 0 -> block 0) favor? Assert the
        // covariate raises its own group's topic more than the other covariate does.
        let block0_topic = if m.topic_word()[0][0] > m.topic_word()[1][0] {
            0
        } else {
            1
        };
        let block1_topic = 1 - block0_topic;
        let c0_contrast = eff[0][block0_topic] - eff[0][block1_topic];
        let c1_contrast = eff[1][block0_topic] - eff[1][block1_topic];
        assert!(
            c0_contrast > c1_contrast,
            "covariate 0 should favor block-0 topic more than covariate 1 does: {c0_contrast} vs {c1_contrast}"
        );
    }

    #[test]
    fn recovers_covariate_prevalence_prior_only() {
        let m = fit_small(CovariateMode::PriorOnly, EncoderKind::ZeroShot, 1);
        let eff = m.covariate_effects();
        let block0_topic = if m.topic_word()[0][0] > m.topic_word()[1][0] {
            0
        } else {
            1
        };
        let block1_topic = 1 - block0_topic;
        let c0_contrast = eff[0][block0_topic] - eff[0][block1_topic];
        let c1_contrast = eff[1][block0_topic] - eff[1][block1_topic];
        assert!(
            c0_contrast > c1_contrast,
            "prior-only should still recover the prevalence direction: {c0_contrast} vs {c1_contrast}"
        );
    }

    #[test]
    fn fit_is_deterministic() {
        let a = fit_small(CovariateMode::EncoderPrior, EncoderKind::Combined, 3);
        let b = fit_small(CovariateMode::EncoderPrior, EncoderKind::Combined, 3);
        assert_eq!(a.topic_word(), b.topic_word());
        assert_eq!(a.doc_topic(), b.doc_topic());
        assert_eq!(a.covariate_effects(), b.covariate_effects());
    }
}
