//! InfoCTM: cross-lingual neural topic model (Wu, Pan, Nguyen, Feng, Liu, Nguyen &
//! Luu 2023, AAAI; arXiv:2304.03544; reference github.com/bobxwu/InfoCTM, license
//! unclear -> implemented from the paper, reference read only to disambiguate math).
//!
//! InfoCTM fits two languages into a SHARED K-topic space. It is two ProdLDA/AVITM
//! models (one per language, separate vocabulary, same K) coupled by a
//! **Topic-Alignment Mutual-Information (TAMI)** term: a masked cross-lingual
//! InfoNCE over the topic-word columns (each word's distribution over topics), with
//! positive cross-lingual word pairs from a bilingual dictionary, optionally
//! densified by per-language word embeddings (cosine >= `pos_threshold`).
//!
//! The two per-language ELBOs reuse `prodlda`'s encoder / batchnorm / Adam / Laplace
//! prior / `batch_forward` / `batch_backward` verbatim (so each language is exactly a
//! ProdLDA, already validated against the AVITM reference). The only new gradient
//! code is the TAMI backward, which is hand-coded and finite-difference checked.

use rand::Rng;

use crate::prodlda::{
    batch_backward, batch_forward, laplace_prior, normalized_bow, randn, raw_bow, AvitmOptions,
    Batch, BatchNorm, Grad, InputMode, Optim, Prior, ProdldaModel, Weights,
};

/// Cross-lingual alignment masks for one direction (Va x Vb): which contrast words
/// are positive (a translation/aligned pair) and which are negatives.
struct Masks {
    pos: Vec<Vec<f64>>, // Va x Vb, 1.0 where (i, j) is an aligned cross-lingual pair
    neg: Vec<Vec<f64>>, // Va x Vb, 1.0 where (i, j) is a negative (not positive)
    pos_sum: f64,
}

/// Build the positive/negative masks for direction a->b. `trans_ab` is the binary
/// bilingual dictionary matrix (Va x Vb, 1 where word a_i translates word b_j).
/// `emb_a` (optional, Va x d) densifies positives: a row is first expanded by its
/// monolingual neighbours (cosine >= `pos_threshold`) before translating. With no
/// embeddings the monolingual mask is the identity, so positives are exactly the
/// dictionary pairs.
fn build_masks(trans_ab: &[Vec<f64>], emb_a: Option<&[Vec<f64>]>, pos_threshold: f64) -> Masks {
    let va = trans_ab.len();
    let vb = if va > 0 { trans_ab[0].len() } else { 0 };
    // pos_trans[i][j] = sum_i' pos_mono[i][i'] * trans_ab[i'][j], thresholded to {0,1}.
    let mut pos = vec![vec![0.0f64; vb]; va];
    for i in 0..va {
        // Monolingual neighbours of word i (always includes itself).
        let neighbours: Vec<usize> = match emb_a {
            None => vec![i],
            Some(emb) => {
                let ni = l2(&emb[i]);
                (0..va)
                    .filter(|&ip| {
                        if ip == i {
                            return true;
                        }
                        let c = dot(&emb[i], &emb[ip]) / (ni * l2(&emb[ip]) + 1e-12);
                        c >= pos_threshold
                    })
                    .collect()
            }
        };
        for &ip in &neighbours {
            for j in 0..vb {
                if trans_ab[ip][j] > 0.0 {
                    pos[i][j] = 1.0;
                }
            }
        }
    }
    let mut neg = vec![vec![0.0f64; vb]; va];
    let mut pos_sum = 0.0;
    for i in 0..va {
        for j in 0..vb {
            if pos[i][j] > 0.0 {
                pos_sum += 1.0;
            } else {
                neg[i][j] = 1.0;
            }
        }
    }
    Masks { pos, neg, pos_sum }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
fn l2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Topic-word features for one language: row `i` is word `i`'s K-vector,
/// `fea[i][k] = beta[k*V + i]`, L2-normalized rows returned separately.
struct Features {
    fea: Vec<Vec<f64>>,  // V x K (raw)
    norm: Vec<Vec<f64>>, // V x K (L2-normalized rows)
    inv: Vec<f64>,       // V, 1/||fea[i]||
}

fn features(beta: &[f64], k: usize, v: usize) -> Features {
    let mut fea = vec![vec![0.0f64; k]; v];
    for kk in 0..k {
        for i in 0..v {
            fea[i][kk] = beta[kk * v + i];
        }
    }
    let mut norm = vec![vec![0.0f64; k]; v];
    let mut inv = vec![0.0f64; v];
    for i in 0..v {
        let n = l2(&fea[i]).max(1e-12);
        inv[i] = 1.0 / n;
        for kk in 0..k {
            norm[i][kk] = fea[i][kk] * inv[i];
        }
    }
    Features { fea, norm, inv }
}

/// One direction of the masked InfoNCE: anchor features `A` vs contrast features `B`
/// with positive mask `P` and negative mask `Ng`, temperature `t`. Returns the
/// unnormalized loss `-sum P*log_prob` and, when `grads` is Some, accumulates the
/// gradients w.r.t. the raw anchor/contrast features into `(dA, dB)` (V x K each).
///
/// log_prob[i][j] = s[i][j] - log( sum_j' exp(s[i][j'])Ng[i][j'] + exp(s[i][j]) ),
/// s[i][j] = (norm_a[i] . norm_b[j]) / t. The reference subtracts a per-row max for
/// stability; that shift cancels in log_prob, so it does not enter the gradient.
#[allow(clippy::too_many_arguments)]
fn mutual_info(
    fa: &Features,
    fb: &Features,
    p: &[Vec<f64>],
    ng: &[Vec<f64>],
    temp: f64,
    mut grads: Option<(&mut [Vec<f64>], &mut [Vec<f64>])>,
) -> f64 {
    let va = fa.fea.len();
    let vb = fb.fea.len();
    let k = if va > 0 { fa.fea[0].len() } else { 0 };
    let mut loss = 0.0;
    // ds[i][j] accumulators (only needed for the backward).
    let mut ds = if grads.is_some() {
        vec![vec![0.0f64; vb]; va]
    } else {
        Vec::new()
    };

    for i in 0..va {
        // s[i][.] and a per-row max for numerical stability.
        let mut s = vec![0.0f64; vb];
        let mut smax = f64::NEG_INFINITY;
        for j in 0..vb {
            s[j] = dot(&fa.norm[i], &fb.norm[j]) / temp;
            if s[j] > smax {
                smax = s[j];
            }
        }
        // D[i][j] = sum_j' exp(s_j' - smax) Ng_ij' + exp(s_j - smax)
        let exps: Vec<f64> = (0..vb).map(|j| (s[j] - smax).exp()).collect();
        let neg_sum: f64 = (0..vb).map(|j| exps[j] * ng[i][j]).sum();
        // R[i] = sum_j P_ij / D_ij  (for the gradient).
        let mut r_i = 0.0;
        let mut d_row = vec![0.0f64; vb];
        for j in 0..vb {
            let d = neg_sum + exps[j] + 1e-10;
            d_row[j] = d;
            if p[i][j] > 0.0 {
                // log_prob = (s_j - smax) - log D ; loss += -P*log_prob
                let log_prob = (s[j] - smax) - d.ln();
                loss -= p[i][j] * log_prob;
                r_i += p[i][j] / d;
            }
        }
        if grads.is_some() {
            for j in 0..vb {
                // dL/ds_ij = -P_ij + exp(s_ij)( Ng_ij R_i + P_ij / D_ij )
                ds[i][j] = -p[i][j] + exps[j] * (ng[i][j] * r_i + p[i][j] / d_row[j]);
            }
        }
    }

    if let Some((da, db)) = grads.as_mut() {
        // dL/d norm_a[i] = sum_j ds_ij * norm_b[j] / temp ; symmetric for b.
        // Then push through the L2 normalization to the raw features.
        let mut dna = vec![vec![0.0f64; k]; va];
        let mut dnb = vec![vec![0.0f64; k]; vb];
        for i in 0..va {
            for j in 0..vb {
                let g = ds[i][j] / temp;
                if g != 0.0 {
                    for kk in 0..k {
                        dna[i][kk] += g * fb.norm[j][kk];
                        dnb[j][kk] += g * fa.norm[i][kk];
                    }
                }
            }
        }
        accumulate_norm_grad(&dna, fa, da);
        accumulate_norm_grad(&dnb, fb, db);
    }
    loss
}

/// Push `dn` (grad w.r.t. the L2-normalized rows) back to the raw features and add
/// into `out` (V x K). For n = x/||x||: dL/dx = (1/||x||)( dn - n (n.dn) ).
fn accumulate_norm_grad(dn: &[Vec<f64>], f: &Features, out: &mut [Vec<f64>]) {
    let v = dn.len();
    let k = if v > 0 { dn[0].len() } else { 0 };
    for i in 0..v {
        let ndn: f64 = (0..k).map(|kk| f.norm[i][kk] * dn[i][kk]).sum();
        for kk in 0..k {
            out[i][kk] += f.inv[i] * (dn[i][kk] - f.norm[i][kk] * ndn);
        }
    }
}

/// The full TAMI loss for the two raw topic-word matrices, plus (optionally) the
/// gradients added into `g_beta_a` / `g_beta_b` (K*V row-major, matching `Weights.beta`).
/// `weight` is `mi_weight`, normalized by the total positive count across directions.
#[allow(clippy::too_many_arguments)]
fn tami(
    beta_a: &[f64],
    beta_b: &[f64],
    k: usize,
    va: usize,
    vb: usize,
    mab: &Masks,
    mba: &Masks,
    temp: f64,
    weight: f64,
    mut g_beta: Option<(&mut [f64], &mut [f64])>,
) -> f64 {
    let fa = features(beta_a, k, va);
    let fb = features(beta_b, k, vb);
    let denom = (mab.pos_sum + mba.pos_sum).max(1.0);
    let scale = weight / denom;

    let (mut da, mut db) = if g_beta.is_some() {
        (vec![vec![0.0f64; k]; va], vec![vec![0.0f64; k]; vb])
    } else {
        (Vec::new(), Vec::new())
    };

    let want = g_beta.is_some();
    let loss_ab = mutual_info(
        &fa,
        &fb,
        &mab.pos,
        &mab.neg,
        temp,
        if want { Some((&mut da, &mut db)) } else { None },
    );
    let loss_ba = mutual_info(
        &fb,
        &fa,
        &mba.pos,
        &mba.neg,
        temp,
        if want { Some((&mut db, &mut da)) } else { None },
    );

    if let Some((ga, gb)) = g_beta.as_mut() {
        for kk in 0..k {
            for i in 0..va {
                ga[kk * va + i] += scale * da[i][kk];
            }
            for j in 0..vb {
                gb[kk * vb + j] += scale * db[j][kk];
            }
        }
    }
    scale * (loss_ab + loss_ba)
}

// ---------------------------------------------------------------------------
// Fit driver
// ---------------------------------------------------------------------------

/// Per-language training state: the AVITM weights, the three batchnorms, the
/// optimizer, and the prepared sparse documents.
struct LangState {
    w: Weights,
    bn_mu: BatchNorm,
    bn_lv: BatchNorm,
    bn_dec: BatchNorm,
    opt: Optim,
    xn: Vec<Vec<(usize, f64)>>,
    bows: Vec<Vec<(usize, f64)>>,
    totals: Vec<f64>,
    v: usize,
}

impl LangState {
    fn new<R: Rng>(
        docs: &[Vec<u32>],
        v: usize,
        hidden: usize,
        k: usize,
        lr: f64,
        rng: &mut R,
    ) -> Self {
        let xn: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| normalized_bow(d)).collect();
        let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| raw_bow(d)).collect();
        let totals: Vec<f64> = bows
            .iter()
            .map(|b| b.iter().map(|&(_, c)| c).sum())
            .collect();
        let w = Weights::new(v, 0, hidden, k, InputMode::BowOnly, rng);
        // Adam beta1 = 0.9 to match the InfoCTM reference's optimizer (default
        // torch Adam betas), NOT ProdLDA's high-momentum 0.99 anti-collapse value:
        // this is an InfoCTM port, so it follows InfoCTM's training recipe.
        let opt = Optim::new(&w, lr, 0.9, 0.0);
        LangState {
            w,
            bn_mu: BatchNorm::new(k),
            bn_lv: BatchNorm::new(k),
            bn_dec: BatchNorm::new(v),
            opt,
            xn,
            bows,
            totals,
            v,
        }
    }
}

/// A fitted InfoCTM model: two fitted ProdLDA models (one per language) sharing the
/// topic index, so topic `k` means the same theme in both languages.
pub struct InfoctmModel {
    pub num_topics: usize,
    pub model_a: ProdldaModel,
    pub model_b: ProdldaModel,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub epochs_run: usize,
}

/// Fit InfoCTM. `docs_a`/`docs_b` are the two languages' corpora (independent
/// vocabularies `va`/`vb`); `trans_ab` is the Va x Vb binary dictionary matrix;
/// `emb_a`/`emb_b` are optional per-language embeddings densifying the alignment
/// masks. Hyperparameters mirror the reference (lr 0.002, hidden 100, dropout 0,
/// temperature 0.2, pos_threshold 0.4, weight 30). Returns the two fitted models.
#[allow(clippy::too_many_arguments)]
pub fn fit_infoctm<R: Rng>(
    docs_a: &[Vec<u32>],
    docs_b: &[Vec<u32>],
    va: usize,
    vb: usize,
    trans_ab: &[Vec<f64>],
    emb_a: Option<&[Vec<f64>]>,
    emb_b: Option<&[Vec<f64>]>,
    num_topics: usize,
    hidden: usize,
    dropout: f64,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    mi_weight: f64,
    temperature: f64,
    pos_threshold: f64,
    em_tol: f64,
    rng: &mut R,
) -> InfoctmModel {
    let k = num_topics;
    let alpha_vec = vec![1.0f64; k];
    let (prior_mu, prior_var) = laplace_prior(&alpha_vec);
    let opts = AvitmOptions::default(); // laplace prior, no contrastive
    let keep = (1.0 - dropout).max(1e-6);

    // trans_ba is the transpose of trans_ab.
    let mut trans_ba = vec![vec![0.0f64; va]; vb];
    for (i, row) in trans_ab.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            trans_ba[j][i] = x;
        }
    }
    let mab = build_masks(trans_ab, emb_a, pos_threshold);
    let mba = build_masks(&trans_ba, emb_b, pos_threshold);

    let mut a = LangState::new(docs_a, va, hidden, k, lr, rng);
    let mut b = LangState::new(docs_b, vb, hidden, k, lr, rng);

    let mut bound_history = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = 0usize;

    for epoch in 0..epochs {
        epochs_run = epoch + 1;
        let order_a = shuffle(a.xn.len(), rng);
        let order_b = shuffle(b.xn.len(), rng);
        let chunks_a: Vec<&[usize]> = order_a.chunks(batch_size.max(2)).collect();
        let chunks_b: Vec<&[usize]> = order_b.chunks(batch_size.max(2)).collect();
        let nb = chunks_a.len().min(chunks_b.len());

        let mut epoch_loss = 0.0;
        let mut steps = 0usize;
        for bi in 0..nb {
            let ca = chunks_a[bi];
            let cb = chunks_b[bi];
            if ca.len() < 2 || cb.len() < 2 {
                continue;
            }
            // Per-language ELBO forward/backward (reuses the ProdLDA core).
            let (la, mut ga) = elbo_step(
                &mut a, ca, k, keep, &prior_mu, &prior_var, &alpha_vec, &opts, rng,
            );
            let (lb, mut gb) = elbo_step(
                &mut b, cb, k, keep, &prior_mu, &prior_var, &alpha_vec, &opts, rng,
            );

            // TAMI coupling on the two raw topic-word matrices, added to beta grads.
            let l_tami = tami(
                &a.w.beta,
                &b.w.beta,
                k,
                va,
                vb,
                &mab,
                &mba,
                temperature,
                mi_weight,
                Some((&mut ga.beta, &mut gb.beta)),
            );

            a.opt.step(&mut a.w, &ga);
            b.opt.step(&mut b.w, &gb);

            epoch_loss += (la + lb) / 2.0 + l_tami;
            steps += 1;
        }
        let avg = epoch_loss / steps.max(1) as f64;
        bound_history.push(-avg);
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            if (-avg - prev).abs() / (prev.abs() + 1e-12) < em_tol {
                converged = true;
                break;
            }
        }
    }

    let model_a = finish(
        a,
        k,
        bound_history.last().copied().unwrap_or(f64::NAN),
        &bound_history,
        converged,
        epochs_run,
        docs_a,
    );
    let model_b = finish(
        b,
        k,
        bound_history.last().copied().unwrap_or(f64::NAN),
        &bound_history,
        converged,
        epochs_run,
        docs_b,
    );
    InfoctmModel {
        num_topics: k,
        model_a,
        model_b,
        bound_history,
        converged,
        epochs_run,
    }
}

/// One ELBO minibatch step for a language: forward+backward through the ProdLDA
/// core, scaled by 1/n. Returns the (already scaled) loss and the gradient (with the
/// beta block still open so the caller can add the TAMI gradient before stepping).
#[allow(clippy::too_many_arguments)]
fn elbo_step<R: Rng>(
    s: &mut LangState,
    chunk: &[usize],
    k: usize,
    keep: f64,
    prior_mu: &[f64],
    prior_var: &[f64],
    alpha_vec: &[f64],
    opts: &AvitmOptions,
    rng: &mut R,
) -> (f64, Grad) {
    let n = chunk.len();
    let hidden = s.w.hidden;
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
    let empty: Vec<f64> = Vec::new();
    let batch = Batch {
        xns: chunk.iter().map(|&di| s.xn[di].as_slice()).collect(),
        embs: chunk.iter().map(|_| empty.as_slice()).collect(),
        counts: chunk.iter().map(|&di| s.bows[di].as_slice()).collect(),
        totals: chunk.iter().map(|&di| s.totals[di]).collect(),
        eps: &eps,
        masks2: &masks2,
        masks_t: &masks_t,
        prior_mus: None,
    };
    let (loss, cache, stats) = batch_forward(
        &s.w, &s.bn_mu, &s.bn_lv, &s.bn_dec, prior_mu, prior_var, alpha_vec, opts, &batch, None,
    );
    s.bn_mu.update_running(&stats[0].0, &stats[0].1);
    s.bn_lv.update_running(&stats[1].0, &stats[1].1);
    s.bn_dec.update_running(&stats[2].0, &stats[2].1);
    let mut g = Grad::zeros(&s.w);
    batch_backward(
        &s.w, prior_mu, prior_var, alpha_vec, opts, &batch, &cache, &mut g, None, None, None,
    );
    g.scale(1.0 / n as f64);
    (loss / n as f64, g)
}

fn shuffle<R: Rng>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.gen::<f64>() * (i + 1) as f64) as usize;
        order.swap(i, j.min(i));
    }
    order
}

/// Build a fitted ProdldaModel for one language from its trained state.
fn finish(
    s: LangState,
    k: usize,
    bound: f64,
    bound_history: &[f64],
    converged: bool,
    epochs_run: usize,
    docs: &[Vec<u32>],
) -> ProdldaModel {
    let model = ProdldaModel {
        num_topics: k,
        num_types: s.v,
        doc_topic: Vec::new(),
        bound,
        bound_history: bound_history.to_vec(),
        converged,
        epochs_run,
        weights: s.w,
        bn_mu: s.bn_mu,
        // Laplace transform is softmax(mu) and never reads bn_lv (#428).
        bn_lv: None,
        prior: Prior::Laplace,
    };
    let empty: Vec<Vec<f64>> = vec![Vec::new(); docs.len()];
    let doc_topic = model.transform_with_emb(docs, &empty);
    ProdldaModel { doc_topic, ..model }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // A small fixed TAMI setup: random raw betas, a block dictionary, no embeddings.
    fn setup() -> (Vec<f64>, Vec<f64>, usize, usize, usize, Masks, Masks) {
        let (k, va, vb) = (3usize, 6usize, 5usize);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let beta_a: Vec<f64> = (0..k * va).map(|_| rng.gen::<f64>() - 0.5).collect();
        let beta_b: Vec<f64> = (0..k * vb).map(|_| rng.gen::<f64>() - 0.5).collect();
        // Dictionary: link word i to word (i % vb).
        let mut trans = vec![vec![0.0f64; vb]; va];
        for i in 0..va {
            trans[i][i % vb] = 1.0;
        }
        let mut trans_t = vec![vec![0.0f64; va]; vb];
        for i in 0..va {
            for j in 0..vb {
                trans_t[j][i] = trans[i][j];
            }
        }
        let mab = build_masks(&trans, None, 0.4);
        let mba = build_masks(&trans_t, None, 0.4);
        (beta_a, beta_b, k, va, vb, mab, mba)
    }

    #[test]
    fn tami_gradient_matches_fd() {
        let (beta_a, beta_b, k, va, vb, mab, mba) = setup();
        let (temp, weight) = (0.2, 5.0);
        let mut ga = vec![0.0; beta_a.len()];
        let mut gb = vec![0.0; beta_b.len()];
        tami(
            &beta_a,
            &beta_b,
            k,
            va,
            vb,
            &mab,
            &mba,
            temp,
            weight,
            Some((&mut ga, &mut gb)),
        );

        let fd = 1e-6;
        let mut max_rel: f64 = 0.0;
        // check a sample of beta_a and beta_b entries
        for (beta, g, vv) in [(beta_a.clone(), &ga, va), (beta_b.clone(), &gb, vb)] {
            for idx in 0..beta.len() {
                let mut bp = beta.clone();
                let (other, kk, ii) = if vv == va {
                    (&beta_b, idx / va, idx % va)
                } else {
                    (&beta_a, idx / vb, idx % vb)
                };
                let _ = (kk, ii);
                bp[idx] += fd;
                let lp = if vv == va {
                    tami(&bp, other, k, va, vb, &mab, &mba, temp, weight, None)
                } else {
                    tami(other, &bp, k, va, vb, &mab, &mba, temp, weight, None)
                };
                bp[idx] -= 2.0 * fd;
                let lm = if vv == va {
                    tami(&bp, other, k, va, vb, &mab, &mba, temp, weight, None)
                } else {
                    tami(other, &bp, k, va, vb, &mab, &mba, temp, weight, None)
                };
                let num = (lp - lm) / (2.0 * fd);
                let rel = (num - g[idx]).abs() / (num.abs().max(g[idx].abs()) + 1e-6);
                max_rel = max_rel.max(rel);
            }
        }
        assert!(max_rel < 1e-4, "TAMI gradient max relative error {max_rel}");
    }

    // Two-language planted blocks linked by a block-aligned dictionary. After a
    // joint fit the per-language outputs must be valid distributions, and the run
    // must be bit-for-bit reproducible under a seed.
    fn planted(n: usize, v: usize, k: usize, seed: u64) -> Vec<Vec<u32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let per = v / k;
        (0..n)
            .map(|d| {
                let b = d % k;
                (0..15)
                    .map(|_| (b * per + (rng.gen::<f64>() * per as f64) as usize) as u32)
                    .collect()
            })
            .collect()
    }

    fn block_dict(va: usize, vb: usize, k: usize) -> Vec<Vec<f64>> {
        let (pa, pb) = (va / k, vb / k);
        let mut t = vec![vec![0.0f64; vb]; va];
        for b in 0..k {
            for i in b * pa..(b + 1) * pa {
                for j in b * pb..(b + 1) * pb {
                    t[i][j] = 1.0;
                }
            }
        }
        t
    }

    fn fit_small(seed: u64) -> InfoctmModel {
        let (k, va, vb) = (3usize, 9usize, 6usize);
        let docs_a = planted(90, va, k, 1);
        let docs_b = planted(84, vb, k, 2);
        let trans = block_dict(va, vb, k);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        fit_infoctm(
            &docs_a, &docs_b, va, vb, &trans, None, None, k, 32, 0.0, 60, 40, 0.01, 30.0, 0.2, 0.4,
            0.0, &mut rng,
        )
    }

    #[test]
    fn fit_produces_valid_aligned_models() {
        let m = fit_small(11);
        assert_eq!(m.num_topics, 3);
        for model in [&m.model_a, &m.model_b] {
            for row in model.topic_word() {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
            for row in &model.doc_topic {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(row.iter().all(|&x| x >= 0.0));
            }
        }
    }

    #[test]
    fn fit_is_deterministic() {
        let m1 = fit_small(11);
        let m2 = fit_small(11);
        assert_eq!(m1.model_a.topic_word(), m2.model_a.topic_word());
        assert_eq!(m1.model_b.doc_topic, m2.model_b.doc_topic);
    }
}
