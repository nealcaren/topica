//! CorEx: information-theoretic topic modeling by Correlation Explanation
//! (Ver Steeg & Galstyan, NIPS 2014 / AISTATS 2015; Gallagher, Reing, Kale &
//! Ver Steeg, "Anchored Correlation Explanation," TACL 2017).
//!
//! Unlike every other topica model, CorEx is NOT generative and NOT a matrix
//! factorization. It learns `k` BINARY latent topics `Y_j` that maximize the total
//! correlation (multivariate mutual information) they explain about the words. Each
//! topic is on/off per document; in tree mode words are softly partitioned across
//! topics. Anchor words pin chosen words to chosen topics for semi-supervision.
//!
//! Reference: the `corextopic` package (Apache-2.0), which we read to match the
//! update rules and reimplement here. We credit it; we do not copy its code.
//!
//! Faithfulness notes (verified against the reference in the Gate-A review):
//! - **Bits vs nats are mixed on purpose.** `binary_entropy`, `h_x`, and the mutual
//!   information `mis` are in log2 (bits); the marginals `theta`, the priors, the
//!   per-sample `log_z`, and the total correlations `tcs`/`tc` are in natural log
//!   (nats). The P-T bias subtracts `1/(2D)` from the bit-valued MI. Do not unify.
//! - **The softness `s` is per-word persistent state** (a length-V vector), not a
//!   scalar: after each structure step `s_i` grows x1.3 wherever the column-sum of
//!   alpha over high-TC topics exceeds 1.1, and `tau_ji = 1 + s_i |tcs_j|`.
//! - Reported `topic_word = alpha .* mis` (so generic top-word/coherence ranking
//!   surfaces the words CorEx actually assigned); raw `mis` and `alpha` are exposed
//!   separately. `doc_topic = p_y_given_x` is a matrix of independent per-topic
//!   Bernoulli probabilities and does NOT sum to 1 across topics.

use crate::estimator::{Estimator, ModelFamily};
use crate::nmf::{sp_x_b, sp_xt_b, Mat, SpMat};
use rand::Rng;

const EPS_CLIP: f64 = 1e-6; // p_y_given_x clip

/// Fitted CorEx state; the PyO3 binding reads these back.
pub struct CorExModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub num_groups: usize,
    /// `topic_word = alpha .* mis` (K x V), the display/ranking matrix.
    pub topic_word: Vec<Vec<f64>>,
    /// `doc_topic = p_y_given_x` (D x K); independent per-topic probabilities.
    pub doc_topic: Vec<Vec<f64>>,
    /// Raw mutual information `mis` (K x V, bits, unjittered).
    pub mis: Vec<Vec<f64>>,
    /// Word->topic soft membership `alpha` (K x V).
    pub alpha: Vec<Vec<f64>>,
    /// Binary labels `p_y_given_x > 0.5` (D x K).
    pub labels: Vec<Vec<u8>>,
    /// Word cluster = argmax_j alpha[j, i] (length V).
    pub clusters: Vec<usize>,
    /// Sign of correlation of each word with each topic (K x V), +1/-1.
    pub sign: Vec<Vec<i8>>,
    /// Per-topic total correlation (nats).
    pub tcs: Vec<f64>,
    /// Total correlation = sum(tcs).
    pub total_correlation: f64,
    /// Sum-TC per iteration (nats).
    pub tc_history: Vec<f64>,
    pub converged: bool,
    pub iters_run: usize,
    // --- parameters retained for held-out transform ---
    /// log p(y_j=1) per topic (K), reordered to output topic order.
    pub log_p_y: Vec<f64>,
    /// theta planes [lp_0g0, lp_0g1, lp_1g0, lp_1g1], each V x K (output order).
    pub theta: [Vec<Vec<f64>>; 4],
    /// log p(x_i=0) per word (V, nats).
    pub lp0: Vec<f64>,
    /// log p(x_i=0) - log p(x_i=1) per word (V, nats).
    pub px_frac: Vec<f64>,
}

/// Label held-out documents: return `p_y_given_x` (n_docs x K) using the fitted
/// alpha, theta, and priors (the reference `transform`).
pub fn corex_transform(model: &CorExModel, docs: &[Vec<u32>]) -> Vec<Vec<f64>> {
    let (k, v) = (model.num_topics, model.num_types);
    let x = binary_presence(docs, v);
    let alpha = Mat::from_rows(&model.alpha);
    let theta = [
        Mat::from_rows(&model.theta[0]),
        Mat::from_rows(&model.theta[1]),
        Mat::from_rows(&model.theta[2]),
        Mat::from_rows(&model.theta[3]),
    ];
    let _ = (k, v);
    let (pygx, _lz) = calculate_latent(
        &x,
        &alpha,
        &theta,
        &model.log_p_y,
        &model.lp0,
        &model.px_frac,
    );
    pygx.rows_vec()
}

/// Binary entropy in BITS (log2), matching the reference `binary_entropy`.
#[inline]
fn binary_entropy(p: f64) -> f64 {
    if p > 0.0 && p < 1.0 {
        -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
    } else {
        0.0
    }
}

/// `log(1 - exp(x))` for x < 0, guarded.
#[inline]
fn log_1mp(x: f64) -> f64 {
    (-x.exp()).ln_1p()
}

/// `logsumexp([a, b])`.
#[inline]
fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Build a BINARY presence CSR (D x V): each doc's distinct in-vocab words, val 1,
/// columns ascending (for deterministic sparse products).
fn binary_presence(docs: &[Vec<u32>], num_types: usize) -> SpMat {
    let mut indptr = Vec::with_capacity(docs.len() + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    indptr.push(0);
    let mut seen = vec![false; num_types];
    let mut touched = Vec::new();
    for doc in docs {
        for &w in doc {
            let w = w as usize;
            if w < num_types && !seen[w] {
                seen[w] = true;
                touched.push(w);
            }
        }
        touched.sort_unstable();
        for &w in &touched {
            col_idx.push(w);
            vals.push(1.0);
            seen[w] = false;
        }
        touched.clear();
        indptr.push(col_idx.len());
    }
    SpMat {
        rows: docs.len(),
        cols: num_types,
        indptr,
        col_idx,
        vals,
    }
}

/// Fit CorEx (tree mode, binarized counts). `anchors[j]` is the list of word ids
/// anchored to topic `j` (empty for unanchored topics); `anchor_strength` is the
/// alpha value written for anchored (word, topic) pairs. `alpha` and
/// `p_y_given_x` are seeded `Uniform[0,1]` (the reference default). Returns the
/// fitted state.
#[allow(clippy::too_many_arguments)]
pub fn fit_corex<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    anchors: &[Vec<usize>],
    anchor_strength: f64,
    iters: usize,
    eps: f64,
    rng: &mut R,
) -> CorExModel {
    let k = num_topics;
    let v = num_types;
    let x = binary_presence(docs, v);
    let d = x.rows;
    let anchored = anchors.iter().any(|a| !a.is_empty());

    // Per-word statistics. word_counts clipped for never/always-appearing words.
    let mut word_counts = vec![0.0f64; v];
    for (r, _) in docs.iter().enumerate() {
        let (cols, _) = x.row(r);
        for &c in cols {
            word_counts[c] += 1.0;
        }
    }
    for wc in word_counts.iter_mut() {
        if *wc == 0.0 || *wc == d as f64 {
            *wc = wc.clamp(0.01, d as f64 - 0.01);
        }
    }
    let word_freq: Vec<f64> = word_counts.iter().map(|&c| c / d as f64).collect();
    let h_x: Vec<f64> = word_freq.iter().map(|&f| binary_entropy(f)).collect(); // bits
    let lp0: Vec<f64> = word_freq.iter().map(|&f| (1.0 - f).ln()).collect(); // nats
    let px_frac: Vec<f64> = word_freq.iter().map(|&f| (1.0 - f).ln() - f.ln()).collect();

    // Initialize alpha (K x V) and p_y_given_x (D x K), seeded Uniform[0,1].
    let (mut alpha, mut pygx) = {
        let mut a = Mat::zeros(k, v);
        if k > 1 {
            for x in a.data.iter_mut() {
                *x = rng.gen::<f64>();
            }
        } else {
            a.data.iter_mut().for_each(|x| *x = 1.0);
        }
        let mut p = Mat::zeros(d, k);
        for x in p.data.iter_mut() {
            *x = rng.gen::<f64>();
        }
        // Anchored topics: bias pygx toward the anchor words' document presence.
        for (j, aj) in anchors.iter().enumerate() {
            if aj.is_empty() {
                continue;
            }
            for r in 0..d {
                let (cols, _) = x.row(r);
                let hit = cols.iter().filter(|&&c| aj.contains(&c)).count();
                let mean = hit as f64 / aj.len() as f64;
                let cur = p.at(r, j);
                p.set(r, j, 0.5 * cur + 0.5 * mean);
            }
        }
        (a, p)
    };

    let mut s = vec![20.0f64; v]; // per-word softness (persistent)
    let mut log_p_y = vec![0.0f64; k];
    // theta planes, each V x K: [lp_0g0, lp_0g1, lp_1g0, lp_1g1].
    let mut theta = [
        Mat::zeros(v, k),
        Mat::zeros(v, k),
        Mat::zeros(v, k),
        Mat::zeros(v, k),
    ];
    let mut mis = Mat::zeros(k, v);
    let mut sign = Mat::zeros(v, k);
    let mut tcs = vec![0.0f64; k];
    let mut tc_history: Vec<f64> = Vec::new();
    let tc_oom = 1.0 / d as f64;
    let mut converged = false;
    let mut iters_run = 0usize;

    for nloop in 0..iters {
        iters_run = nloop + 1;

        // 1. Label orientation (nloop>1): flip a topic if its top-MI word is
        //    anti-correlated with it.
        if nloop > 1 {
            for j in 0..k {
                let mut best_i = 0usize;
                let mut best = f64::NEG_INFINITY;
                for i in 0..v {
                    if mis.at(j, i) > best {
                        best = mis.at(j, i);
                        best_i = i;
                    }
                }
                if sign.at(best_i, j) < 0.0 {
                    for r in 0..d {
                        pygx.set(r, j, 1.0 - pygx.at(r, j));
                    }
                }
            }
        }

        // 2. log p(y_j=1) = log mean_d p_y_given_x[d,j].
        for j in 0..k {
            let mut m = 0.0;
            for r in 0..d {
                m += pygx.at(r, j);
            }
            log_p_y[j] = (m / d as f64).ln();
        }

        // 3. theta from p_dot_y = X^T . pygx (V x K), with topic-dependent clipping.
        let p_dot_y = sp_xt_b(&x, &pygx); // V x K
        for i in 0..v {
            for j in 0..k {
                let py = log_p_y[j].exp();
                let lo = 0.01 * py;
                let hi = (d as f64 - 0.01) * py;
                let pdy = p_dot_y.at(i, j).clamp(lo, hi);
                let lp_1g1 = pdy.ln() - (d as f64).ln() - log_p_y[j];
                let rest = (word_counts[i] - pdy).max(1e-12);
                let lp_1g0 = rest.ln() - (d as f64).ln() - log_1mp(log_p_y[j]);
                theta[2].set(i, j, lp_1g0);
                theta[3].set(i, j, lp_1g1);
                theta[0].set(i, j, log_1mp(lp_1g0)); // lp_0g0
                theta[1].set(i, j, log_1mp(lp_1g1)); // lp_0g1
                sign.set(i, j, (lp_1g1 - lp_1g0).signum());
            }
        }

        // MI (bits), P-T bias corrected, clipped >= 0.
        compute_mis(&mut mis, &theta, &log_p_y, &h_x, d);

        // 4. Structure step (nloop>0): adaptive per-word softness -> alpha.
        if nloop > 0 && k > 1 {
            // sa_i = sum_{j: tcs_j > 1/D} alpha[j,i]; grow s_i where sa_i > 1.1.
            for i in 0..v {
                let mut sa = 0.0;
                for j in 0..k {
                    if tcs[j] > tc_oom {
                        sa += alpha.at(j, i);
                    }
                }
                if sa > 1.1 {
                    s[i] *= 1.3;
                }
            }
            // Seeded jitter on every column (reference adds 1e-10*rand to each
            // column that contains a row-max, which is every column), then argmax.
            let mut maxmis = vec![f64::NEG_INFINITY; v];
            for i in 0..v {
                for j in 0..k {
                    let jit = mis.at(j, i) + 1e-10 * rng.gen::<f64>();
                    mis.set(j, i, jit);
                    if jit > maxmis[i] {
                        maxmis[i] = jit;
                    }
                }
            }
            for j in 0..k {
                for i in 0..v {
                    let tau = 1.0 + s[i] * tcs[j].abs();
                    let a = (tau * (mis.at(j, i) - maxmis[i]) / h_x[i]).exp();
                    alpha.set(j, i, a);
                }
            }
        } else if k == 1 {
            alpha.data.iter_mut().for_each(|x| *x = 1.0);
        }

        // 5. Anchor override.
        if anchored {
            for aj in anchors.iter() {
                for &a in aj {
                    for j in 0..k {
                        alpha.set(j, a, 0.0);
                    }
                }
            }
            for (j, aj) in anchors.iter().enumerate() {
                for &a in aj {
                    alpha.set(j, a, anchor_strength);
                }
            }
        }

        // 6. Latent update -> p_y_given_x, log_z.
        let (new_pygx, log_z) = calculate_latent(&x, &alpha, &theta, &log_p_y, &lp0, &px_frac);
        pygx = new_pygx;

        // 7. tcs = mean_d log_z[d,j]; record sum.
        for j in 0..k {
            let mut m = 0.0;
            for r in 0..d {
                m += log_z.at(r, j);
            }
            tcs[j] = m / d as f64;
        }
        tc_history.push(tcs.iter().sum());

        // 8. Convergence.
        if tc_history.len() > 10 {
            let n = tc_history.len();
            let recent: f64 = tc_history[n - 5..].iter().sum::<f64>() / 5.0;
            let older: f64 = tc_history[n - 10..n - 5].iter().sum::<f64>() / 5.0;
            if (recent - older).abs() < eps {
                converged = true;
                break;
            }
        }
    }

    // Recompute UNjittered mis for output; recompute latent for labels.
    compute_mis(&mut mis, &theta, &log_p_y, &h_x, d);
    let (final_pygx, _lz) = calculate_latent(&x, &alpha, &theta, &log_p_y, &lp0, &px_frac);
    pygx = final_pygx;

    // Sort topics by descending tcs (only when unanchored; anchored keeps order).
    let order: Vec<usize> = if anchored {
        (0..k).collect()
    } else {
        let mut o: Vec<usize> = (0..k).collect();
        o.sort_by(|&a, &b| {
            tcs[b]
                .partial_cmp(&tcs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        o
    };

    // Assemble reordered outputs.
    let mut tw = vec![vec![0.0f64; v]; k]; // alpha * mis
    let mut mis_out = vec![vec![0.0f64; v]; k];
    let mut alpha_out = vec![vec![0.0f64; v]; k];
    let mut sign_out = vec![vec![0i8; v]; k];
    let tcs_out: Vec<f64> = order.iter().map(|&j| tcs[j]).collect();
    for (jo, &j) in order.iter().enumerate() {
        for i in 0..v {
            let a = alpha.at(j, i);
            let m = mis.at(j, i);
            alpha_out[jo][i] = a;
            mis_out[jo][i] = m;
            tw[jo][i] = a * m;
            sign_out[jo][i] = sign.at(i, j) as i8;
        }
    }
    let mut dt = vec![vec![0.0f64; k]; d];
    let mut labels = vec![vec![0u8; k]; d];
    for r in 0..d {
        for (jo, &j) in order.iter().enumerate() {
            let p = pygx.at(r, j);
            dt[r][jo] = p;
            labels[r][jo] = (p > 0.5) as u8;
        }
    }
    let clusters: Vec<usize> = (0..v)
        .map(|i| {
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for j in 0..k {
                if alpha_out[j][i] > bv {
                    bv = alpha_out[j][i];
                    best = j;
                }
            }
            best
        })
        .collect();
    let total = tcs_out.iter().sum();

    // Retain transform parameters, reordered to output topic order.
    let log_p_y_out: Vec<f64> = order.iter().map(|&j| log_p_y[j]).collect();
    let theta_out: [Vec<Vec<f64>>; 4] = std::array::from_fn(|p| {
        (0..v)
            .map(|i| order.iter().map(|&j| theta[p].at(i, j)).collect())
            .collect()
    });

    CorExModel {
        num_topics: k,
        num_types: v,
        num_groups: anchors.iter().filter(|a| !a.is_empty()).count(),
        topic_word: tw,
        doc_topic: dt,
        mis: mis_out,
        alpha: alpha_out,
        labels,
        clusters,
        sign: sign_out,
        tcs: tcs_out,
        total_correlation: total,
        tc_history,
        converged,
        iters_run,
        log_p_y: log_p_y_out,
        theta: theta_out,
        lp0,
        px_frac,
    }
}

/// MI in bits: `h_x - p_y*H(p(x=1|y=1)) - (1-p_y)*H(p(x=1|y=0))`, minus `1/(2D)`,
/// clipped >= 0. `mis` is K x V; theta planes are V x K.
fn compute_mis(mis: &mut Mat, theta: &[Mat; 4], log_p_y: &[f64], h_x: &[f64], d: usize) {
    let (k, v) = (mis.rows, mis.cols);
    let bias = 1.0 / (2.0 * d as f64);
    for j in 0..k {
        let py = log_p_y[j].exp();
        for i in 0..v {
            let p_x1g1 = theta[3].at(i, j).exp();
            let p_x1g0 = theta[2].at(i, j).exp();
            let m = h_x[i] - py * binary_entropy(p_x1g1) - (1.0 - py) * binary_entropy(p_x1g0);
            mis.set(j, i, (m - bias).max(0.0));
        }
    }
}

/// `p_y_given_x` (D x K) and pointwise `log_z` (D x K) from alpha, theta, priors.
fn calculate_latent(
    x: &SpMat,
    alpha: &Mat,
    theta: &[Mat; 4],
    log_p_y: &[f64],
    lp0: &[f64],
    px_frac: &[f64],
) -> (Mat, Mat) {
    let (k, v) = (alpha.rows, alpha.cols);
    let d = x.rows;
    // c0/c1 (length K); info0/info1 (V x K) for the sparse X . info products.
    let mut c0 = vec![0.0f64; k];
    let mut c1 = vec![0.0f64; k];
    let mut info0 = Mat::zeros(v, k);
    let mut info1 = Mat::zeros(v, k);
    for j in 0..k {
        for i in 0..v {
            let a = alpha.at(j, i);
            c0[j] += a * (theta[0].at(i, j) - lp0[i]);
            c1[j] += a * (theta[1].at(i, j) - lp0[i]);
            info0.set(
                i,
                j,
                a * (theta[2].at(i, j) - theta[0].at(i, j) + px_frac[i]),
            );
            info1.set(
                i,
                j,
                a * (theta[3].at(i, j) - theta[1].at(i, j) + px_frac[i]),
            );
        }
    }
    let x_info0 = sp_x_b(x, &info0); // D x K
    let x_info1 = sp_x_b(x, &info1); // D x K
    let mut pygx = Mat::zeros(d, k);
    let mut log_z = Mat::zeros(d, k);
    for r in 0..d {
        for j in 0..k {
            let u1 = log_p_y[j] + c1[j] + x_info1.at(r, j);
            let u0 = log_1mp(log_p_y[j]) + c0[j] + x_info0.at(r, j);
            let lz = logsumexp2(u0, u1);
            log_z.set(r, j, lz);
            let p = (u1 - lz).exp().clamp(EPS_CLIP, 1.0 - EPS_CLIP);
            pygx.set(r, j, p);
        }
    }
    (pygx, log_z)
}

impl Estimator for CorExModel {
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
        self.tc_history
            .iter()
            .enumerate()
            .map(|(i, &e)| (i + 1, e))
            .collect()
    }
    fn converged(&self) -> Option<bool> {
        Some(self.converged)
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Planted 3-block binary corpus: docs cycle through blocks; block words present,
    // light cross-block noise.
    fn planted() -> (Vec<Vec<u32>>, usize) {
        let (k, bl, reps) = (3usize, 4u32, 40usize);
        let v = k * bl as usize;
        let mut docs = Vec::new();
        for r in 0..reps {
            for t in 0..k as u32 {
                let base = t * bl;
                let mut doc: Vec<u32> = (base..base + bl).collect();
                if r % 5 == 0 {
                    doc.push((t * bl + bl) % v as u32); // a little cross-block noise
                }
                docs.push(doc);
            }
        }
        (docs, v)
    }

    #[test]
    fn corex_recovers_planted_blocks() {
        let (docs, v) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let m = fit_corex(
            &docs,
            v,
            3,
            &[vec![], vec![], vec![]],
            1.0,
            200,
            1e-5,
            &mut rng,
        );
        // Each of the 3 blocks (words 0-3, 4-7, 8-11) should map to one cluster, and
        // there should be 3 distinct clusters.
        assert_eq!(m.clusters.len(), v);
        let distinct: std::collections::HashSet<_> = m.clusters.iter().collect();
        assert_eq!(distinct.len(), 3, "clusters: {:?}", m.clusters);
        for b in 0..3 {
            let block: Vec<usize> = (b * 4..b * 4 + 4).map(|i| m.clusters[i]).collect();
            let first = block[0];
            assert!(
                block.iter().filter(|&&c| c == first).count() >= 3,
                "block {b}: {block:?}"
            );
        }
        assert!(m.total_correlation > 0.0);
    }

    #[test]
    fn corex_is_deterministic() {
        let (docs, v) = planted();
        let fit = || {
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            fit_corex(
                &docs,
                v,
                3,
                &[vec![], vec![], vec![]],
                1.0,
                80,
                1e-5,
                &mut rng,
            )
        };
        let a = fit();
        let b = fit();
        assert_eq!(a.mis, b.mis);
        assert_eq!(a.alpha, b.alpha);
        assert_eq!(a.clusters, b.clusters);
    }

    #[test]
    fn corex_conforms() {
        let (docs, v) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let m = fit_corex(
            &docs,
            v,
            3,
            &[vec![], vec![], vec![]],
            1.0,
            40,
            1e-5,
            &mut rng,
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    #[test]
    fn corex_anchoring_places_words() {
        let (docs, v) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        // Anchor word 0 -> topic 0, word 4 -> topic 1, word 8 -> topic 2.
        let anchors = vec![vec![0usize], vec![4usize], vec![8usize]];
        let m = fit_corex(&docs, v, 3, &anchors, 2.0, 200, 1e-5, &mut rng);
        // Each anchored word is assigned to its topic (anchor order preserved).
        assert_eq!(m.clusters[0], 0);
        assert_eq!(m.clusters[4], 1);
        assert_eq!(m.clusters[8], 2);
    }
}
