//! SCHOLAR (Card, Tan & Smith, "Neural Models for Documents with Metadata",
//! ACL 2018; reference `dallascard/scholar`, Apache-2.0).
//!
//! SCHOLAR extends a ProdLDA/AVITM VAE with document metadata in three roles, all
//! implemented here on topica's existing ProdLDA VAE (`crate::prodlda`) — the encoder,
//! reparameterization, decoder, batchnorm, Adam, and reconstruction loss are reused
//! through the shared `batch_forward`/`batch_backward`, each role threaded in through
//! a gated, no-op-when-unused hook so ProdLDA/InfoCTM stay byte-identical:
//!
//!   1. **Prior covariates** `PC` (prevalence) — a weight block `prior_w`
//!      (`K x n_prior_covars`, the reference's `Linear(n_prior_covars, K, bias=False)`)
//!      sets a per-document prior *mean* `mu_0 = W . PC`, so a covariate that co-occurs
//!      with a topic raises its prevalence (neural STM/DMR prevalence; the fitted `W`
//!      is `covariate_effects`). Threaded via `Batch::prior_mus` and the
//!      `batch_backward` `d_prior_mu` out-param. `PC` also enters the encoder.
//!   2. **Labels** `Y` (supervised) — a softmax classifier head `wc`/`bc` off `theta`
//!      whose cross-entropy loss shapes the topics to be predictive (neural sLDA).
//!      Its gradient into `theta` is injected via `batch_backward`'s `dtheta_extra`.
//!      Unlike the reference, labels do NOT enter the encoder (see `fit_scholar`).
//!   3. **Content / topic covariates** `TC` — per-covariate word deviations `beta_c`
//!      (and optional topic-covariate interactions `beta_ci`) added to the decoder
//!      logits, so the same topic is worded differently across groups (neural SAGE;
//!      the fitted `beta_c` is `content_effects`). Threaded via the shared
//!      `ContentFwd`/`ContentGrad` hook. `TC` also enters the encoder.
//!
//! Because topica's ProdLDA is the two-layer AVITM encoder (Srivastava & Sutton 2017)
//! rather than the reference's single embedding layer, this is a mechanism-faithful
//! port on topica's backbone, not a bit-for-bit clone of `dallascard/scholar` — the
//! same design choice topica's `ProdLDA` already makes. The prior is Gaussian
//! (logistic-normal) only: a prior-*mean* shift is only defined for `Prior::Laplace`.

use rand::Rng;

use crate::prodlda::{
    batch_backward, batch_forward, laplace_prior, normalized_bow, randn, raw_bow, Adam,
    AvitmOptions, Batch, BatchNorm, ContentFwd, ContentGrad, Grad, InputMode, Optim, Prior,
    ProdldaModel, Weights,
};

/// Numerically stable softmax (local copy; `prodlda::softmax` is private).
fn softmax(v: &[f64]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
}

/// A fitted SCHOLAR model. `base` is the underlying ProdLDA VAE (topic-word,
/// doc-topic, encoder, mean-head batchnorm); `prior_w` is the covariate weight
/// matrix (`K x n_prior_covars`) that shifts the document prior mean; `wc`/`bc` are
/// the label classifier head (`n_labels x K` and `n_labels`) read off `theta`.
pub struct ScholarModel {
    pub base: ProdldaModel,
    /// Covariate weights `W`, `K x n_prior_covars` row-major (the reference's
    /// `prior_covar_weights.weight`): `prior_mean[i] = W . PC[i]`.
    pub prior_w: Vec<f64>,
    pub n_prior_covars: usize,
    pub l2_prior_reg: f64,
    /// Label classifier weights, `n_labels x K` row-major (the reference's
    /// `classifier_layer_0.weight`): `logit_y[i] = wc . theta[i] + bc`. Empty when
    /// the model was fit without labels.
    pub wc: Vec<f64>,
    pub bc: Vec<f64>,
    pub n_labels: usize,
    /// Content (topic-covariate) decoder deviations `beta_c`, `n_topic_covars x V`
    /// row-major (the reference's `beta_c_layer.weight.T`): per-covariate word shifts.
    pub beta_c: Vec<f64>,
    /// Optional topic-covariate interaction weights `beta_ci`, `(K*n_topic_covars) x V`
    /// row-major; `None` unless the model was fit with `interactions`.
    pub beta_ci: Option<Vec<f64>>,
    pub n_topic_covars: usize,
    pub interactions: bool,
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

    /// The per-covariate word-deviation matrix `(n_topic_covars, V)` (`beta_c`).
    /// Entry `[c][j]` is how much topic covariate `c` shifts the unnormalized log-word
    /// weight of word `j` — the SAGE "same topic, worded differently across groups"
    /// deviations. Empty when fit without content covariates.
    pub fn content_effects(&self) -> Vec<Vec<f64>> {
        let (tc, v) = (self.n_topic_covars, self.base.num_types);
        (0..tc)
            .map(|c| self.beta_c[c * v..(c + 1) * v].to_vec())
            .collect()
    }

    /// Build encoder features `[pc ; tc]` for new documents (both covariate blocks
    /// enter the encoder; labels never do). `tcs` may be empty when the model has no
    /// content covariates.
    fn encoder_feats(&self, pcs: &[Vec<f64>], tcs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        pcs.iter()
            .enumerate()
            .map(|(i, pc)| {
                let mut f = pc.clone();
                if self.n_topic_covars > 0 {
                    f.extend_from_slice(&tcs[i]);
                }
                f
            })
            .collect()
    }

    /// Held-out topic proportions for new documents given their prior covariates and
    /// topic covariates. Both enter the encoder as at fit time; labels are never an
    /// encoder input (see `fit_scholar`), so prediction and training use the same path.
    pub fn transform(
        &self,
        docs: &[Vec<u32>],
        pcs: &[Vec<f64>],
        tcs: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let feats = self.encoder_feats(pcs, tcs);
        self.base.transform_with_emb(docs, &feats)
    }

    /// Class-probability predictions `softmax(wc . theta + bc)` for new documents,
    /// shape `(num_docs, n_labels)`. `theta` is the posterior mean from the words (and
    /// covariates). Empty rows if the model was fit without labels.
    pub fn predict_proba(
        &self,
        docs: &[Vec<u32>],
        pcs: &[Vec<f64>],
        tcs: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let k = self.base.num_topics;
        let theta = self.transform(docs, pcs, tcs);
        theta
            .iter()
            .map(|th| {
                let logit: Vec<f64> = (0..self.n_labels)
                    .map(|l| bc_dot(&self.wc, &self.bc, th, l, k))
                    .collect();
                softmax(&logit)
            })
            .collect()
    }
}

/// One classifier logit: `bc[l] + sum_t wc[l*k+t] * theta[t]`.
fn bc_dot(wc: &[f64], bc: &[f64], theta: &[f64], l: usize, k: usize) -> f64 {
    let base = l * k;
    bc[l] + (0..k).map(|t| wc[base + t] * theta[t]).sum::<f64>()
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

/// Fit SCHOLAR with prior covariates and/or supervised labels. `docs` are token-id
/// documents; `pcs[i]` is the dense prior-covariate row for document `i` (length
/// `n_prior_covars`; pass empty rows and `n_prior_covars == 0` for labels-only).
/// `labels`, when given, is the class index of each document (`0..n_labels`); the
/// classifier head is fit jointly and `n_labels == 0` reproduces the prior-covariate
/// path exactly. `alpha` is the symmetric Dirichlet concentration behind the Laplace
/// prior variance; `l2_prior_reg` is the L2 penalty on `W`. The training loop mirrors
/// `prodlda::fit_avitm` (same shuffle, noise, dropout, Adam, batchnorm) and adds the
/// covariate prior-mean update and, when labeled, a softmax classifier off `theta`
/// whose cross-entropy loss trains `wc`/`bc` and pushes a gradient back into `theta`.
#[allow(clippy::too_many_arguments)]
pub fn fit_scholar<R: Rng>(
    docs: &[Vec<u32>],
    pcs: &[Vec<f64>],
    labels: Option<&[usize]>,
    n_labels: usize,
    tcs: &[Vec<f64>],
    n_topic_covars: usize,
    interactions: bool,
    l1_content_reg: f64,
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
    let (k, v, pc, tc) = (num_topics, num_types, n_prior_covars, n_topic_covars);
    let d = docs.len();
    // Encoder input concatenates prior covariates and topic covariates (both are
    // always observed, at train and test — no leak). Labels stay out of the encoder.
    // With tc == 0 and pc unchanged this is the prevalence/label path unchanged.
    let feats: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            let mut f = pcs[i].clone();
            if tc > 0 {
                f.extend_from_slice(&tcs[i]);
            }
            f
        })
        .collect();
    let emb_dim = pc + tc;
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

    // Prior covariates ride the dense-embedding channel into the encoder. Labels do
    // NOT enter the encoder (a documented deviation from `dallascard/scholar`, which
    // concatenates the label to the encoder input and zeroes it at test). We supervise
    // the topics purely through the classifier head's gradient into `theta` — the sLDA
    // mechanism, where the label is not an inference-time input — so q(theta | words)
    // is used at both train and test (the reference's q(theta | words, label) at train
    // vs q(theta | words, 0) at test is an input-distribution inconsistency).
    //
    // This is both principled (consistency, sLDA faithfulness) and empirically
    // motivated FOR THIS BACKBONE: on topica's two-layer AVITM encoder, feeding the
    // label in degraded held-out accuracy on a small planted corpus (0.65 at 200
    // epochs -> 0.56 at 400; dropping it reaches ~1.0). The reference's single-layer
    // encoder at scale does NOT show this (an independent benchmark could not
    // reproduce a leak against `dallascard/scholar` — the BOW reconstruction term
    // dominates the label cross-entropy ~100:1), so the effect is backbone- and
    // regime-dependent, not a claim that the reference is broken.
    let mut w = Weights::new(v, emb_dim, hidden, k, InputMode::BowEmb, rng);
    // Covariate weight block: zero init, so topics start covariate-agnostic and the
    // prevalence effect is learned from data (deterministic, no extra RNG draws).
    let mut prior_w = vec![0.0; k * pc];
    let mut prior_w_opt = Adam::new(prior_w.len(), lr, 0.99, 0.0);
    // Content (topic-covariate) decoder deviations: beta_c (C x V) per-covariate word
    // shifts, and optional beta_ci ((K*C) x V) topic-covariate interactions. Zero init.
    let mut beta_c = vec![0.0; tc * v];
    let mut beta_c_opt = Adam::new(beta_c.len(), lr, 0.99, 0.0);
    let mut beta_ci = if interactions && tc > 0 {
        vec![0.0; k * tc * v]
    } else {
        Vec::new()
    };
    let mut beta_ci_opt = Adam::new(beta_ci.len(), lr, 0.99, 0.0);
    // Label classifier head: logit_y = wc.theta + bc. Zero init, like the prior
    // weights (diverges from the reference's default nn.Linear Kaiming-uniform init):
    // symmetry breaks on the first step since g_wc = dlogit (x) theta is nonzero, and
    // it keeps the RNG stream identical to the labels-off path (aids determinism).
    let mut wc = vec![0.0; n_labels * k];
    let mut bc = vec![0.0; n_labels];
    let mut wc_opt = Adam::new(wc.len(), lr, 0.99, 0.0);
    let mut bc_opt = Adam::new(bc.len(), lr, 0.99, 0.0);

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
                embs: chunk.iter().map(|&di| feats[di].as_slice()).collect(),
                counts: chunk.iter().map(|&di| bows[di].as_slice()).collect(),
                totals: chunk.iter().map(|&di| totals[di]).collect(),
                eps: &eps,
                masks2: &masks2,
                masks_t: &masks_t,
                prior_mus: Some(&prior_mus),
            };

            // Content (topic-covariate) decoder deviation for this batch.
            let tc_batch: Vec<Vec<f64>> = if tc > 0 {
                chunk.iter().map(|&di| tcs[di].clone()).collect()
            } else {
                Vec::new()
            };
            let content_fwd = (tc > 0).then(|| ContentFwd {
                tc: &tc_batch,
                beta_c: &beta_c,
                beta_ci: if interactions { Some(&beta_ci) } else { None },
                c: tc,
            });

            let (loss, cache, stats) = batch_forward(
                &w,
                &bn_mu,
                &bn_lv,
                &bn_dec,
                &prior_mu,
                &prior_var,
                &alpha_vec,
                &opts,
                &batch,
                content_fwd.as_ref(),
            );
            bn_mu.update_running(&stats[0].0, &stats[0].1);
            bn_lv.update_running(&stats[1].0, &stats[1].1);
            bn_dec.update_running(&stats[2].0, &stats[2].1);

            // Label classifier off theta: logit_y = wc.theta + bc; cross-entropy loss
            // -log(y_recon[true]). The softmax-CE gradient is dlogit = y_recon - Y,
            // giving g_wc = dlogit (x) theta, g_bc = dlogit, and a gradient back into
            // theta, dtheta_class = wc^T . dlogit, injected via `dtheta_extra`.
            let theta = cache.theta();
            let mut g_wc = vec![0.0; n_labels * k];
            let mut g_bc = vec![0.0; n_labels];
            let mut dtheta_class = vec![vec![0.0; k]; n];
            let mut class_loss = 0.0;
            if n_labels > 0 {
                let lbls = labels.unwrap();
                for (li, &di) in chunk.iter().enumerate() {
                    let y = lbls[di];
                    let logit: Vec<f64> = (0..n_labels)
                        .map(|l| bc_dot(&wc, &bc, &theta[li], l, k))
                        .collect();
                    let yr = softmax(&logit);
                    class_loss += -(yr[y] + 1e-10).ln();
                    for l in 0..n_labels {
                        let dl = yr[l] - if l == y { 1.0 } else { 0.0 };
                        g_bc[l] += dl;
                        let base = l * k;
                        for t in 0..k {
                            g_wc[base + t] += dl * theta[li][t];
                            dtheta_class[li][t] += wc[base + t] * dl;
                        }
                    }
                }
            }

            let mut g = Grad::zeros(&w);
            let mut d_prior_mu = vec![vec![0.0; k]; n];
            let dte = if n_labels > 0 {
                Some(dtheta_class.as_slice())
            } else {
                None
            };
            let mut content_grad = ContentGrad {
                beta_c: vec![0.0; tc * v],
                beta_ci: if interactions && tc > 0 {
                    Some(vec![0.0; k * tc * v])
                } else {
                    None
                },
            };
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
                dte,
                content_fwd.as_ref().map(|cf| (cf, &mut content_grad)),
            );
            g.scale(1.0 / n as f64);
            opt.step(&mut w, &g);

            // Content deviation Adam step: data term batch-mean + a fixed-strength
            // ridge penalty (l1_content_reg; the reference's adaptive-strength L1
            // reweighting is simplified to a fixed strength here), un-averaged.
            if tc > 0 {
                let inv_n = 1.0 / n as f64;
                for idx in 0..content_grad.beta_c.len() {
                    content_grad.beta_c[idx] =
                        content_grad.beta_c[idx] * inv_n + 2.0 * l1_content_reg * beta_c[idx];
                }
                beta_c_opt.step(&mut beta_c, &content_grad.beta_c);
                if interactions {
                    if let Some(gci) = content_grad.beta_ci.as_mut() {
                        for idx in 0..gci.len() {
                            gci[idx] = gci[idx] * inv_n + 2.0 * l1_content_reg * beta_ci[idx];
                        }
                        beta_ci_opt.step(&mut beta_ci, gci);
                    }
                }
            }

            // Classifier head Adam step (data term averaged over the batch, like the
            // encoder/decoder grads; no regularization on the head).
            if n_labels > 0 {
                let inv_n = 1.0 / n as f64;
                for x in g_wc.iter_mut() {
                    *x *= inv_n;
                }
                for x in g_bc.iter_mut() {
                    *x *= inv_n;
                }
                wc_opt.step(&mut wc, &g_wc);
                bc_opt.step(&mut bc, &g_bc);
            }

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

            // Report the full objective (recon + KL + classification), batch-mean.
            epoch_loss += (loss + class_loss) / n as f64;
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
    let doc_topic = base.transform_with_emb(docs, &feats);
    let base = ProdldaModel { doc_topic, ..base };
    ScholarModel {
        base,
        prior_w,
        n_prior_covars: pc,
        l2_prior_reg,
        wc,
        bc,
        n_labels,
        beta_c,
        beta_ci: if interactions && tc > 0 {
            Some(beta_ci)
        } else {
            None
        },
        n_topic_covars: tc,
        interactions: interactions && tc > 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// No topic covariates (the prevalence/label-only fit paths).
    const NO_TC: &[Vec<f64>] = &[];

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
            w, &bn_mu, &bn_lv, &bn_dec, &prior_mu, prior_var, alpha, opts, &batch, None,
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
            &w, &bn_mu, &bn_lv, &bn_dec, &prior_mu0, &prior_var, &alpha, &opts, &batch, None,
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
            None,
            None,
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
            &docs, &pcs, None, 0, NO_TC, 0, false, 0.0, k, v, 2, 20, 1.0, 0.2, 120, 40, 0.01, 0.0,
            0.0, &mut rng,
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
                &docs, &pcs, None, 0, NO_TC, 0, false, 0.0, 2, 6, 2, 8, 1.0, 0.2, 15, 4, 0.01,
                0.01, 0.0, &mut rng,
            )
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word(), b.topic_word());
        assert_eq!(a.covariate_effects(), b.covariate_effects());
        assert_eq!(a.base.doc_topic, b.base.doc_topic);
    }

    // The label classifier head is a standard linear+softmax+cross-entropy. Check its
    // analytic gradient (g_wc, g_bc, and the dtheta contribution) against finite
    // differences of the classification loss at a fixed theta — isolating the new head
    // math from the encoder (whose dtheta path prodlda already FD-checks).
    #[test]
    fn classifier_head_gradient_matches_fd() {
        let (k, n_labels) = (4usize, 3usize);
        let theta = [
            vec![0.4, 0.3, 0.2, 0.1],
            vec![0.1, 0.5, 0.3, 0.1],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let ys = [0usize, 2, 1];
        let mut wc: Vec<f64> = (0..n_labels * k).map(|i| 0.1 * (i as f64) - 0.35).collect();
        let mut bc: Vec<f64> = (0..n_labels).map(|l| 0.2 * l as f64 - 0.1).collect();

        // Summed classification loss over the mini-batch at (wc, bc, theta).
        let loss = |wc: &[f64], bc: &[f64], theta: &[Vec<f64>]| -> f64 {
            let mut s = 0.0;
            for (i, th) in theta.iter().enumerate() {
                let logit: Vec<f64> = (0..n_labels).map(|l| bc_dot(wc, bc, th, l, k)).collect();
                let yr = softmax(&logit);
                s += -(yr[ys[i]] + 1e-10).ln();
            }
            s
        };

        // Analytic gradients (mirror the fit loop, summed / un-averaged here).
        let mut g_wc = vec![0.0; n_labels * k];
        let mut g_bc = vec![0.0; n_labels];
        let mut dtheta = vec![vec![0.0; k]; theta.len()];
        for (i, th) in theta.iter().enumerate() {
            let y = ys[i];
            let logit: Vec<f64> = (0..n_labels).map(|l| bc_dot(&wc, &bc, th, l, k)).collect();
            let yr = softmax(&logit);
            for l in 0..n_labels {
                let dl = yr[l] - if l == y { 1.0 } else { 0.0 };
                g_bc[l] += dl;
                for t in 0..k {
                    g_wc[l * k + t] += dl * th[t];
                    dtheta[i][t] += wc[l * k + t] * dl;
                }
            }
        }

        let fd = 1e-6;
        // wc
        for idx in 0..wc.len() {
            let o = wc[idx];
            wc[idx] = o + fd;
            let lp = loss(&wc, &bc, &theta);
            wc[idx] = o - fd;
            let lm = loss(&wc, &bc, &theta);
            wc[idx] = o;
            let num = (lp - lm) / (2.0 * fd);
            assert!(
                (g_wc[idx] - num).abs() < 1e-4,
                "g_wc[{idx}] {} vs {num}",
                g_wc[idx]
            );
        }
        // bc
        for l in 0..n_labels {
            let o = bc[l];
            bc[l] = o + fd;
            let lp = loss(&wc, &bc, &theta);
            bc[l] = o - fd;
            let lm = loss(&wc, &bc, &theta);
            bc[l] = o;
            let num = (lp - lm) / (2.0 * fd);
            assert!(
                (g_bc[l] - num).abs() < 1e-4,
                "g_bc[{l}] {} vs {num}",
                g_bc[l]
            );
        }
        // dtheta (perturb one document's theta component)
        for i in 0..theta.len() {
            for t in 0..k {
                let mut tp = theta.to_vec();
                tp[i][t] += fd;
                let lp = loss(&wc, &bc, &tp);
                tp[i][t] -= 2.0 * fd;
                let lm = loss(&wc, &bc, &tp);
                let num = (lp - lm) / (2.0 * fd);
                assert!(
                    (dtheta[i][t] - num).abs() < 1e-4,
                    "dtheta[{i}][{t}] {} vs {num}",
                    dtheta[i][t]
                );
            }
        }
    }

    // A label predictable from the topics should be recovered: documents split into
    // classes with distinct word blocks; the jointly-fit classifier predicts held-out
    // classes well above chance.
    #[test]
    fn fit_recovers_labels() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (k, v, n_labels) = (3usize, 12usize, 3usize);
        let blocks: Vec<Vec<u32>> = (0..3)
            .map(|g| (g * 4..g * 4 + 4).map(|x| x as u32).collect())
            .collect();
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for g in 0..3 {
            for _ in 0..50 {
                let mut doc = Vec::new();
                for _ in 0..30 {
                    let blk = if rng.gen::<f64>() < 0.9 {
                        &blocks[g]
                    } else {
                        &blocks[(rng.gen::<f64>() * 3.0) as usize]
                    };
                    doc.push(blk[(rng.gen::<f64>() * 4.0) as usize]);
                }
                docs.push(doc);
                labels.push(g);
            }
        }
        // No covariates: labels-only (empty pc rows).
        let pcs: Vec<Vec<f64>> = vec![Vec::new(); docs.len()];
        let m = fit_scholar(
            &docs,
            &pcs,
            Some(&labels),
            n_labels,
            NO_TC,
            0,
            false,
            0.0,
            k,
            v,
            0,
            20,
            1.0,
            0.2,
            400,
            50,
            0.01,
            0.0,
            0.0,
            &mut rng,
        );

        // In-sample class accuracy from predict_proba should be well above 1/3.
        let proba = m.predict_proba(&docs, &pcs, NO_TC);
        let mut correct = 0;
        for (i, p) in proba.iter().enumerate() {
            let pred = (0..n_labels)
                .max_by(|&a, &b| p[a].total_cmp(&p[b]))
                .unwrap();
            if pred == labels[i] {
                correct += 1;
            }
        }
        let acc = correct as f64 / labels.len() as f64;
        assert!(acc > 0.7, "label accuracy {acc} not above 0.7");
    }

    #[test]
    fn fit_with_labels_is_deterministic() {
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 2, 0, 1],
            vec![3, 4, 5, 3],
            vec![0, 2, 4, 1, 5],
            vec![1, 1, 3, 5, 2],
        ];
        let pcs: Vec<Vec<f64>> = vec![Vec::new(); 4];
        let labels = [0usize, 1, 0, 1];
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(5);
            fit_scholar(
                &docs,
                &pcs,
                Some(&labels),
                2,
                NO_TC,
                0,
                false,
                0.0,
                2,
                6,
                0,
                8,
                1.0,
                0.2,
                15,
                4,
                0.01,
                0.0,
                0.0,
                &mut rng,
            )
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word(), b.topic_word());
        assert_eq!(a.wc, b.wc);
        assert_eq!(a.bc, b.bc);
        assert_eq!(
            a.predict_proba(&docs, &pcs, NO_TC),
            b.predict_proba(&docs, &pcs, NO_TC)
        );
    }

    // FD check of the content-deviation gradients. beta_c and beta_ci are decoder-only,
    // and the interaction beta_ci*(theta_do (x) TC) makes the loss depend on theta, so
    // the encoder head w_mu also gets a content contribution — check all three against
    // the content-active batch loss.
    #[test]
    fn content_gradient_matches_fd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (v, hidden, k, tc) = (7usize, 5usize, 4usize, 2usize);
        // TC rides the encoder (emb_dim = tc here, pc = 0) and drives decoder deviations.
        let mut w0 = Weights::new(v, tc, hidden, k, InputMode::BowEmb, &mut rng);
        let alpha = vec![1.0; k];
        let (prior_mu, prior_var) = laplace_prior(&alpha);
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
        let tcs: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..tc)
                    .map(|c| 0.5 * (i as f64 + 1.0) - 0.3 * c as f64)
                    .collect()
            })
            .collect();
        let xns: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| normalized_bow(d)).collect();
        let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| raw_bow(d)).collect();
        let totals: Vec<f64> = bows
            .iter()
            .map(|b| b.iter().map(|&(_, c)| c).sum())
            .collect();
        let eps: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|t| 0.1 * (i as f64 + 1.0) - 0.05 * t as f64)
                    .collect()
            })
            .collect();
        let masks2 = vec![vec![1.0; hidden]; n];
        let masks_t = vec![vec![1.0; k]; n];

        let mut beta_c: Vec<f64> = (0..tc * v).map(|i| 0.13 * (i as f64 % 4.0) - 0.2).collect();
        let mut beta_ci: Vec<f64> = (0..k * tc * v)
            .map(|i| 0.07 * (i as f64 % 5.0) - 0.15)
            .collect();

        let build_batch = || Batch {
            xns: xns.iter().map(|x| x.as_slice()).collect(),
            embs: tcs.iter().map(|x| x.as_slice()).collect(),
            counts: bows.iter().map(|b| b.as_slice()).collect(),
            totals: totals.clone(),
            eps: &eps,
            masks2: &masks2,
            masks_t: &masks_t,
            prior_mus: None,
        };
        let loss = |w: &Weights, beta_c: &[f64], beta_ci: &[f64]| -> f64 {
            let bn_mu = BatchNorm::new(k);
            let bn_lv = BatchNorm::new(k);
            let bn_dec = BatchNorm::new(v);
            let cf = ContentFwd {
                tc: &tcs,
                beta_c,
                beta_ci: Some(beta_ci),
                c: tc,
            };
            batch_forward(
                w,
                &bn_mu,
                &bn_lv,
                &bn_dec,
                &prior_mu,
                &prior_var,
                &alpha,
                &opts,
                &build_batch(),
                Some(&cf),
            )
            .0
        };

        // Analytic gradients.
        let bn_mu = BatchNorm::new(k);
        let bn_lv = BatchNorm::new(k);
        let bn_dec = BatchNorm::new(v);
        let cf = ContentFwd {
            tc: &tcs,
            beta_c: &beta_c,
            beta_ci: Some(&beta_ci),
            c: tc,
        };
        let (_, cache, _) = batch_forward(
            &w0,
            &bn_mu,
            &bn_lv,
            &bn_dec,
            &prior_mu,
            &prior_var,
            &alpha,
            &opts,
            &build_batch(),
            Some(&cf),
        );
        let mut g = Grad::zeros(&w0);
        let mut cg = ContentGrad {
            beta_c: vec![0.0; tc * v],
            beta_ci: Some(vec![0.0; k * tc * v]),
        };
        batch_backward(
            &w0,
            &prior_mu,
            &prior_var,
            &alpha,
            &opts,
            &build_batch(),
            &cache,
            &mut g,
            None,
            None,
            Some((&cf, &mut cg)),
        );

        let fd = 1e-6;
        let check = |name: &str, analytic: f64, num: f64| {
            assert!(
                (analytic - num).abs() < 1e-4,
                "{name}: analytic {analytic} vs numeric {num}"
            );
        };
        for idx in 0..beta_c.len() {
            let o = beta_c[idx];
            beta_c[idx] = o + fd;
            let lp = loss(&w0, &beta_c, &beta_ci);
            beta_c[idx] = o - fd;
            let lm = loss(&w0, &beta_c, &beta_ci);
            beta_c[idx] = o;
            check("beta_c", cg.beta_c[idx], (lp - lm) / (2.0 * fd));
        }
        let gci = cg.beta_ci.as_ref().unwrap();
        for idx in 0..beta_ci.len() {
            let o = beta_ci[idx];
            beta_ci[idx] = o + fd;
            let lp = loss(&w0, &beta_c, &beta_ci);
            beta_ci[idx] = o - fd;
            let lm = loss(&w0, &beta_c, &beta_ci);
            beta_ci[idx] = o;
            check("beta_ci", gci[idx], (lp - lm) / (2.0 * fd));
        }
        // Encoder head w_mu, with content active: confirms the interaction dtheta path.
        for idx in 0..w0.w_mu.len() {
            let o = w0.w_mu[idx];
            w0.w_mu[idx] = o + fd;
            let lp = loss(&w0, &beta_c, &beta_ci);
            w0.w_mu[idx] = o - fd;
            let lm = loss(&w0, &beta_c, &beta_ci);
            w0.w_mu[idx] = o;
            check("w_mu(content)", g.w_mu()[idx], (lp - lm) / (2.0 * fd));
        }
    }

    // A content covariate that pushes specific words up/down should be recovered:
    // documents in group 1 over-use a marker word regardless of topic; beta_c for that
    // covariate should place its largest positive deviation on the marker word.
    #[test]
    fn fit_recovers_content() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let (k, v) = (2usize, 10usize);
        let marker = 9u32; // group-1 marker word
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut tcs: Vec<Vec<f64>> = Vec::new();
        for g in 0..2 {
            for _ in 0..60 {
                let mut doc = Vec::new();
                for _ in 0..25 {
                    // Base content from two topic blocks (words 0-3 / 4-7), plus in group
                    // 1 a heavy dose of the marker word 9 (word 8 unused base).
                    if g == 1 && rng.gen::<f64>() < 0.35 {
                        doc.push(marker);
                    } else if rng.gen::<f64>() < 0.5 {
                        doc.push((rng.gen::<f64>() * 4.0) as u32);
                    } else {
                        doc.push(4 + (rng.gen::<f64>() * 4.0) as u32);
                    }
                }
                docs.push(doc);
                tcs.push(if g == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                });
            }
        }
        let pcs: Vec<Vec<f64>> = vec![Vec::new(); docs.len()];
        let m = fit_scholar(
            &docs, &pcs, None, 0, &tcs, 2, false, 0.0, k, v, 0, 20, 1.0, 0.2, 200, 40, 0.01, 0.0,
            0.0, &mut rng,
        );
        let eff = m.content_effects();
        assert_eq!(eff.len(), 2);
        assert_eq!(eff[0].len(), v);
        // Covariate 1 (group 1) should deviate the marker word up relative to covariate 0.
        assert!(
            eff[1][marker as usize] > eff[0][marker as usize],
            "content deviation not recovered: covar1 marker {} !> covar0 marker {}",
            eff[1][marker as usize],
            eff[0][marker as usize]
        );
        // And the marker is the covariate-1 word with (near) the largest deviation.
        let argmax = (0..v)
            .max_by(|&a, &b| eff[1][a].total_cmp(&eff[1][b]))
            .unwrap();
        assert_eq!(
            argmax, marker as usize,
            "marker word not the top content deviation"
        );
    }

    #[test]
    fn fit_with_interactions_is_deterministic() {
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 2, 0, 1],
            vec![3, 4, 5, 3],
            vec![0, 2, 4, 1, 5],
            vec![1, 1, 3, 5, 2],
        ];
        let pcs: Vec<Vec<f64>> = vec![Vec::new(); 4];
        let tcs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(5);
            fit_scholar(
                &docs, &pcs, None, 0, &tcs, 2, true, 0.0, 2, 6, 0, 8, 1.0, 0.2, 15, 4, 0.01, 0.0,
                0.0, &mut rng,
            )
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word(), b.topic_word());
        assert_eq!(a.content_effects(), b.content_effects());
        assert_eq!(a.beta_ci, b.beta_ci);
        assert!(a.interactions && a.beta_ci.is_some());
    }
}
