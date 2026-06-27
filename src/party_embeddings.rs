//! Party Embeddings (Rheault & Cochrane 2020): a PV-DM (distributed-memory
//! paragraph-vector) model trained by negative sampling, where each document is
//! tagged with political metadata (a party-period tag, plus an optional control
//! tag). The learned tag vectors are the "party embeddings"; their leading
//! principal components give the ideological placement, and because the tag
//! vectors share a space with the word vectors, words can be ranked by proximity
//! to a party.
//!
//! Pure Rust, no autodiff. The training objective is the standard negative
//! sampling of Mikolov, Sutskever, et al. (2013) over the distributed-memory
//! architecture of Le & Mikolov (2014): the model predicts a center word from the
//! mean of its context-word embeddings and the document's tag embeddings. The
//! gradient is the textbook word2vec/doc2vec update. We implement from those
//! papers (the reference package partyembed is unlicensed and was used only as a
//! black-box output oracle).
//!
//! Determinism: single-threaded SGD seeded from `seed`, documents visited in a
//! fixed order each epoch, so a fixed seed reproduces the fit bit-for-bit.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A fitted Party Embeddings model. Vectors are stored row-major as flat `f32`.
pub struct PartyEmbeddingsModel {
    pub vector_size: usize,
    pub num_words: usize,
    /// Number of placed tags (party-period groups); the first `num_groups` rows of
    /// `tag_vectors`.
    pub num_groups: usize,
    /// Number of control tags (estimated but not placed); rows
    /// `num_groups..num_groups + num_controls` of `tag_vectors`.
    pub num_controls: usize,
    /// Input word embeddings, `num_words * vector_size` (the word2vec `syn0`).
    pub word_vectors: Vec<f32>,
    /// Input tag embeddings, `(num_groups + num_controls) * vector_size`; the
    /// party embeddings are the first `num_groups` rows.
    pub tag_vectors: Vec<f32>,
    /// Mean negative-sampling loss per epoch.
    pub loss_history: Vec<f64>,
}

impl PartyEmbeddingsModel {
    /// Row `g` of `tag_vectors` (a party-period group) as `f64`.
    pub fn group_vector(&self, g: usize) -> Vec<f64> {
        let m = self.vector_size;
        self.tag_vectors[g * m..(g + 1) * m]
            .iter()
            .map(|&x| x as f64)
            .collect()
    }

    /// The `num_groups` party-period vectors as a `(num_groups, vector_size)`
    /// matrix of `f64` rows (PCA input for placement).
    pub fn group_matrix(&self) -> Vec<Vec<f64>> {
        (0..self.num_groups).map(|g| self.group_vector(g)).collect()
    }

    /// Row `w` of `word_vectors` as `f64`.
    pub fn word_vector(&self, w: usize) -> Vec<f64> {
        let m = self.vector_size;
        self.word_vectors[w * m..(w + 1) * m]
            .iter()
            .map(|&x| x as f64)
            .collect()
    }
}

/// Training hyperparameters for the PV-DM fit.
pub struct PvdmConfig {
    pub vector_size: usize,
    pub window: usize,
    pub negative: usize,
    /// Frequent-word subsampling threshold (gensim `sample`); 0 disables it.
    pub sample: f64,
    pub start_lr: f64,
    pub min_lr: f64,
    pub epochs: usize,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x > 6.0 {
        1.0
    } else if x < -6.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Build the negative-sampling cumulative distribution over word ids, proportional
/// to `freq^0.75` (Mikolov et al.). Returns the cumulative sums; a draw is a binary
/// search of a uniform in `[0, total)`.
fn neg_cumulative(total_freqs: &[u32]) -> Vec<f64> {
    let mut cum = Vec::with_capacity(total_freqs.len());
    let mut acc = 0.0f64;
    for &f in total_freqs {
        acc += (f as f64).max(1.0).powf(0.75);
        cum.push(acc);
    }
    cum
}

#[inline]
fn draw_negative(cum: &[f64], rng: &mut ChaCha8Rng) -> usize {
    let total = *cum.last().unwrap();
    let r = rng.gen::<f64>() * total;
    // first index whose cumulative value exceeds r
    match cum.binary_search_by(|x| x.partial_cmp(&r).unwrap()) {
        Ok(i) => i.min(cum.len() - 1),
        Err(i) => i.min(cum.len() - 1),
    }
}

/// Per-word probability of *keeping* a token under frequent-word subsampling
/// (gensim/word2vec formula). `sample <= 0` keeps everything.
fn keep_probs(total_freqs: &[u32], sample: f64) -> Vec<f64> {
    let total: f64 = total_freqs.iter().map(|&f| f as f64).sum();
    total_freqs
        .iter()
        .map(|&f| {
            if sample <= 0.0 || total == 0.0 {
                return 1.0;
            }
            let z = (f as f64) / total;
            if z <= 0.0 {
                return 1.0;
            }
            (((z / sample).sqrt() + 1.0) * (sample / z)).min(1.0)
        })
        .collect()
}

/// Fit a PV-DM party-embeddings model.
///
/// `docs` are ordered token-id sequences; `doc_group[d]` is the party-period tag
/// index of document `d` (`0..num_groups`); `doc_control[d]`, when present, is a
/// second tag index (`0..num_controls`) estimated alongside but not placed.
#[allow(clippy::too_many_arguments)]
pub fn fit_pvdm(
    docs: &[Vec<u32>],
    doc_group: &[usize],
    doc_control: Option<&[usize]>,
    num_words: usize,
    num_groups: usize,
    num_controls: usize,
    total_freqs: &[u32],
    cfg: &PvdmConfig,
    seed: u64,
) -> PartyEmbeddingsModel {
    let m = cfg.vector_size;
    let num_tags = num_groups + num_controls;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialize input vectors as word2vec does: U(-0.5, 0.5)/vector_size. Output
    // (syn1neg) vectors start at zero.
    let init = |rng: &mut ChaCha8Rng, n: usize| -> Vec<f32> {
        (0..n * m)
            .map(|_| (rng.gen::<f32>() - 0.5) / m as f32)
            .collect()
    };
    let mut word_vectors = init(&mut rng, num_words);
    let mut tag_vectors = init(&mut rng, num_tags);
    let mut syn1neg = vec![0.0f32; num_words * m];

    let cum = neg_cumulative(total_freqs);
    let keep = keep_probs(total_freqs, cfg.sample);

    // Total trainable tokens across all epochs, for the linear learning-rate decay.
    let tokens_per_epoch: usize = docs.iter().map(|d| d.len()).sum();
    let total_train = (tokens_per_epoch * cfg.epochs).max(1) as f64;
    let mut done: f64 = 0.0;
    let mut loss_history = Vec::with_capacity(cfg.epochs);

    let mut neu1 = vec![0.0f32; m];
    let mut neu1e = vec![0.0f32; m];

    for _epoch in 0..cfg.epochs {
        let mut epoch_loss = 0.0f64;
        let mut epoch_n = 0usize;

        for (d, doc) in docs.iter().enumerate() {
            // Subsample frequent words, keeping the surviving tokens in order.
            let kept: Vec<u32> = doc
                .iter()
                .copied()
                .filter(|&w| {
                    let p = keep[w as usize];
                    p >= 1.0 || rng.gen::<f64>() < p
                })
                .collect();
            done += doc.len() as f64;
            if kept.is_empty() {
                continue;
            }

            // The document's tag rows (group first, then optional control).
            let mut tags: [usize; 2] = [doc_group[d], 0];
            let n_tags = if let Some(ctrl) = doc_control {
                tags[1] = num_groups + ctrl[d];
                2
            } else {
                1
            };

            // Learning-rate linear decay.
            let lr = (cfg.start_lr
                - (cfg.start_lr - cfg.min_lr) * (done / total_train))
                .max(cfg.min_lr) as f32;

            let len = kept.len();
            for i in 0..len {
                let center = kept[i] as usize;
                // Dynamic window shrink, as in word2vec.
                let b = (rng.gen::<u32>() as usize) % cfg.window;
                let start = i.saturating_sub(cfg.window - b);
                let end = (i + cfg.window - b + 1).min(len);

                // Hidden = mean of context-word vectors + tag vectors.
                for x in neu1.iter_mut() {
                    *x = 0.0;
                }
                let mut cw = 0usize;
                for (j, &kw) in kept.iter().enumerate().take(end).skip(start) {
                    if j == i {
                        continue;
                    }
                    let off = kw as usize * m;
                    for k in 0..m {
                        neu1[k] += word_vectors[off + k];
                    }
                    cw += 1;
                }
                for &t in tags.iter().take(n_tags) {
                    let off = t * m;
                    for k in 0..m {
                        neu1[k] += tag_vectors[off + k];
                    }
                    cw += 1;
                }
                if cw == 0 {
                    continue;
                }
                let inv = 1.0 / cw as f32;
                for x in neu1.iter_mut() {
                    *x *= inv;
                }

                // Negative sampling against the center word.
                for x in neu1e.iter_mut() {
                    *x = 0.0;
                }
                for kk in 0..(cfg.negative + 1) {
                    let (target, label) = if kk == 0 {
                        (center, 1.0f32)
                    } else {
                        let t = draw_negative(&cum, &mut rng);
                        if t == center {
                            continue;
                        }
                        (t, 0.0f32)
                    };
                    let off = target * m;
                    let mut dot = 0.0f32;
                    for k in 0..m {
                        dot += neu1[k] * syn1neg[off + k];
                    }
                    let f = sigmoid(dot);
                    let g = (label - f) * lr;
                    for k in 0..m {
                        neu1e[k] += g * syn1neg[off + k];
                        syn1neg[off + k] += g * neu1[k];
                    }
                    // Accumulate the negative-sampling loss for reporting.
                    let p = if label > 0.5 { f } else { 1.0 - f };
                    epoch_loss += -(p.max(1e-7) as f64).ln();
                    epoch_n += 1;
                }

                // Back-propagate the input gradient to each context word and tag.
                for (j, &kw) in kept.iter().enumerate().take(end).skip(start) {
                    if j == i {
                        continue;
                    }
                    let off = kw as usize * m;
                    for k in 0..m {
                        word_vectors[off + k] += neu1e[k];
                    }
                }
                for &t in tags.iter().take(n_tags) {
                    let off = t * m;
                    for k in 0..m {
                        tag_vectors[off + k] += neu1e[k];
                    }
                }
            }
        }
        loss_history.push(if epoch_n > 0 {
            epoch_loss / epoch_n as f64
        } else {
            0.0
        });
    }

    PartyEmbeddingsModel {
        vector_size: m,
        num_words,
        num_groups,
        num_controls,
        word_vectors,
        tag_vectors,
        loss_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a corpus with a planted 1-D ordering: group `g` has position
    /// `p in [-1,1]`, and emits "right" marker words with probability rising in
    /// `p`, "left" markers with the complement, plus neutral filler. PV-DM tag
    /// vectors should recover the ordering.
    fn planted(seed: u64) -> (Vec<Vec<u32>>, Vec<usize>, usize, Vec<u32>, Vec<f64>) {
        let n_groups = 10usize;
        let docs_per = 60usize;
        let doc_len = 18usize;
        // vocab: 0..12 left, 12..24 right, 24..64 filler
        let n_left = 12u32;
        let n_right = 12u32;
        let n_fill = 40u32;
        let num_words = (n_left + n_right + n_fill) as usize;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let positions: Vec<f64> = (0..n_groups)
            .map(|g| -1.0 + 2.0 * g as f64 / (n_groups - 1) as f64)
            .collect();
        let mut docs = Vec::new();
        let mut groups = Vec::new();
        let mut freq = vec![0u32; num_words];
        for g in 0..n_groups {
            let pr_right = (positions[g] + 1.0) / 2.0;
            for _ in 0..docs_per {
                let mut doc = Vec::with_capacity(doc_len);
                for _ in 0..doc_len {
                    let u: f64 = rng.gen();
                    let w = if u < 0.45 {
                        if rng.gen::<f64>() < pr_right {
                            n_left + (rng.gen::<u32>() % n_right)
                        } else {
                            rng.gen::<u32>() % n_left
                        }
                    } else {
                        n_left + n_right + (rng.gen::<u32>() % n_fill)
                    };
                    freq[w as usize] += 1;
                    doc.push(w);
                }
                docs.push(doc);
                groups.push(g);
            }
        }
        (docs, groups, num_words, freq, positions)
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut sab = 0.0;
        let mut sa = 0.0;
        let mut sb = 0.0;
        for i in 0..a.len() {
            let da = a[i] - ma;
            let db = b[i] - mb;
            sab += da * db;
            sa += da * da;
            sb += db * db;
        }
        sab / (sa.sqrt() * sb.sqrt())
    }

    /// First principal component of the group vectors via power iteration on the
    /// covariance, for a dependency-free test (the binding uses `reduce::pca`).
    fn first_pc(rows: &[Vec<f64>]) -> Vec<f64> {
        let n = rows.len();
        let m = rows[0].len();
        let mean: Vec<f64> = (0..m)
            .map(|k| rows.iter().map(|r| r[k]).sum::<f64>() / n as f64)
            .collect();
        let xc: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| (0..m).map(|k| r[k] - mean[k]).collect())
            .collect();
        let mut v: Vec<f64> = (0..m).map(|k| ((k + 1) as f64).sin()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
        for _ in 0..100 {
            // w = X^T X v
            let xv: Vec<f64> = xc.iter().map(|r| dot(r, &v)).collect();
            let mut w = vec![0.0; m];
            for (i, r) in xc.iter().enumerate() {
                for k in 0..m {
                    w[k] += r[k] * xv[i];
                }
            }
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for x in w.iter_mut() {
                *x /= norm;
            }
            v = w;
        }
        // project rows onto the leading component
        xc.iter().map(|r| dot(r, &v)).collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn recovers_planted_party_ordering() {
        let (docs, groups, num_words, freq, positions) = planted(0);
        let cfg = PvdmConfig {
            vector_size: 48,
            window: 5,
            negative: 5,
            sample: 1e-3,
            start_lr: 0.05,
            min_lr: 1e-4,
            epochs: 40,
        };
        let model = fit_pvdm(&docs, &groups, None, num_words, 10, 0, &freq, &cfg, 0);
        let pc1 = first_pc(&model.group_matrix());
        let r = pearson(&pc1, &positions).abs();
        assert!(
            r > 0.85,
            "planted recovery too low: |r(PC1, planted)| = {r:.3}"
        );
    }

    #[test]
    fn deterministic_same_seed() {
        let (docs, groups, num_words, freq, _) = planted(1);
        let cfg = PvdmConfig {
            vector_size: 32,
            window: 5,
            negative: 5,
            sample: 1e-3,
            start_lr: 0.05,
            min_lr: 1e-4,
            epochs: 5,
        };
        let a = fit_pvdm(&docs, &groups, None, num_words, 10, 0, &freq, &cfg, 7);
        let b = fit_pvdm(&docs, &groups, None, num_words, 10, 0, &freq, &cfg, 7);
        assert_eq!(a.tag_vectors, b.tag_vectors);
        assert_eq!(a.word_vectors, b.word_vectors);
        // A different seed gives a different fit (so the test cannot pass trivially).
        let c = fit_pvdm(&docs, &groups, None, num_words, 10, 0, &freq, &cfg, 8);
        assert_ne!(a.tag_vectors, c.tag_vectors);
    }
}
