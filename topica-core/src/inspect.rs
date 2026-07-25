//! Post-fit topic inspection: FREX / lift / score scores, top words, semantic
//! coherence, and exclusivity. Ports of R `stm`'s `calcfrex` / `calclift` /
//! `calcscore` / `semCoh1beta` / `exclusivity` (and `js.estimate`), the same
//! formulas faSTM uses, so a host (e.g. the Stata plugin) gets stm-faithful
//! labels and diagnostics from the engine instead of reimplementing them.
//!
//! All functions take `beta`, the K×V topic-word probability matrix
//! (`CtmModel.beta`, rows sum to ~1). `word_counts` is the corpus
//! `total_freqs` (length V); `docs` are the corpus token-id lists.

use std::collections::HashMap;

const EPS: f64 = f64::EPSILON;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn safelog(x: f64) -> f64 {
    x.max(EPS).ln()
}

fn logbeta_of(beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
    beta.iter()
        .map(|row| row.iter().map(|&x| safelog(x)).collect())
        .collect()
}

/// Average-tie ranks (R's `rank`), 1-based.
fn rank_avg(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0; // average of the 1-based positions
        for t in idx.iter().take(j + 1).skip(i) {
            ranks[*t] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Indices of the top `n` entries of `row`, descending (ties by ascending index).
fn top_indices(row: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(n.min(row.len()));
    idx
}

/// Column log-sum-exp over topics: for each word v, lse over k of `logbeta[k][v]`.
fn lse_cols(logbeta: &[Vec<f64>]) -> Vec<f64> {
    let k = logbeta.len();
    let v = if k > 0 { logbeta[0].len() } else { 0 };
    (0..v)
        .map(|vv| {
            let m = (0..k).fold(f64::NEG_INFINITY, |acc, kk| acc.max(logbeta[kk][vv]));
            let s: f64 = (0..k).map(|kk| (logbeta[kk][vv] - m).exp()).sum();
            m + s.ln()
        })
        .collect()
}

/// stm `js.estimate`: James-Stein shrinkage of a probability vector toward uniform.
pub fn js_estimate(prob: &[f64], ct: f64) -> Vec<f64> {
    let n = prob.len();
    let unif = 1.0 / n as f64;
    if ct <= 1.0 {
        return vec![unif; n];
    }
    let mlvar: f64 = prob.iter().map(|&p| p * (1.0 - p) / (ct - 1.0)).sum();
    let dev: f64 = prob.iter().map(|&p| (p - unif).powi(2)).sum();
    if dev == 0.0 {
        return prob.to_vec();
    }
    let mut lambda = mlvar / dev;
    if lambda.is_nan() {
        return vec![unif; n];
    }
    lambda = lambda.clamp(0.0, 1.0);
    prob.iter()
        .map(|&p| lambda * unif + (1.0 - lambda) * p)
        .collect()
}

/// Indices of the top `n` words per topic by `scoremat` (K rows of n indices).
pub fn top_words(scoremat: &[Vec<f64>], n: usize) -> Vec<Vec<usize>> {
    scoremat.iter().map(|row| top_indices(row, n)).collect()
}

// ---------------------------------------------------------------------------
// Score matrices (K×V)
// ---------------------------------------------------------------------------

/// FREX scores (K×V): the rank-harmonic-mean of word frequency and exclusivity
/// (Bischof & Airoldi). `w` is the frequency/exclusivity weight (0.5 = equal).
/// Pass non-empty `word_counts` to James-Stein-shrink exclusivity (stm default).
pub fn frex_scores(beta: &[Vec<f64>], word_counts: &[u32], w: f64) -> Vec<Vec<f64>> {
    let k = beta.len();
    let v = if k > 0 { beta[0].len() } else { 0 };
    let logbeta = logbeta_of(beta);
    let lse = lse_cols(&logbeta);
    let mut excl: Vec<Vec<f64>> = (0..k)
        .map(|kk| (0..v).map(|vv| logbeta[kk][vv] - lse[vv]).collect())
        .collect();

    if word_counts.len() == v && v > 0 {
        for vv in 0..v {
            let col: Vec<f64> = (0..k).map(|kk| excl[kk][vv].exp()).collect();
            let shr = js_estimate(&col, word_counts[vv] as f64);
            for kk in 0..k {
                excl[kk][vv] = safelog(shr[kk]);
            }
        }
    }

    let mut frex = vec![vec![0.0; v]; k];
    for kk in 0..k {
        let fr = rank_avg(&logbeta[kk]);
        let ex = rank_avg(&excl[kk]);
        for vv in 0..v {
            let f = fr[vv] / v as f64;
            let e = ex[vv] / v as f64;
            frex[kk][vv] = 1.0 / (w / f + (1.0 - w) / e);
        }
    }
    frex
}

/// Lift (K×V): `log(beta) - log(empirical word probability)`.
pub fn lift_scores(beta: &[Vec<f64>], word_counts: &[u32]) -> Vec<Vec<f64>> {
    let sumwc: f64 = word_counts.iter().map(|&c| c as f64).sum::<f64>().max(1.0);
    let lsum = sumwc.ln();
    beta.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(vv, &x)| {
                    let wc = *word_counts.get(vv).unwrap_or(&1) as f64;
                    safelog(x) - (wc.max(1.0).ln() - lsum)
                })
                .collect()
        })
        .collect()
}

/// Score (K×V): `beta * (log beta - mean_topics log beta)` (stm `calcscore`).
pub fn score_scores(beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = beta.len();
    let v = if k > 0 { beta[0].len() } else { 0 };
    let logbeta = logbeta_of(beta);
    let colmean: Vec<f64> = (0..v)
        .map(|vv| (0..k).map(|kk| logbeta[kk][vv]).sum::<f64>() / k as f64)
        .collect();
    (0..k)
        .map(|kk| {
            (0..v)
                .map(|vv| beta[kk][vv] * (logbeta[kk][vv] - colmean[vv]))
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// stm `semCoh1beta`: UMass semantic coherence per topic over each topic's top-M
/// words. Pair direction is by global word-list index (the union of all topics'
/// top words), with stm's 0.01 smoothing. Returns K values (higher = better).
pub fn semantic_coherence(beta: &[Vec<f64>], docs: &[Vec<u32>], m: usize) -> Vec<f64> {
    let k = beta.len();
    let topw = top_words(beta, m); // K x M word ids, by beta (= by logbeta)

    // Build the word list in stm's order: topic 0's top words, then topic 1's, ...
    let mut pos_of_word: HashMap<usize, usize> = HashMap::new();
    let mut wordlist: Vec<usize> = Vec::new();
    for row in &topw {
        for &word in row {
            if let std::collections::hash_map::Entry::Vacant(e) = pos_of_word.entry(word) {
                e.insert(wordlist.len());
                wordlist.push(word);
            }
        }
    }
    let r = wordlist.len();

    // Document co-occurrence over the word list: co[a][b] = #docs containing both;
    // co[a][a] = document frequency of word a.
    let mut co = vec![vec![0.0f64; r]; r];
    for doc in docs {
        let mut present: Vec<usize> = doc
            .iter()
            .filter_map(|&t| pos_of_word.get(&(t as usize)).copied())
            .collect();
        present.sort_unstable();
        present.dedup();
        for &a in &present {
            co[a][a] += 1.0;
        }
        for i in 0..present.len() {
            for j in (i + 1)..present.len() {
                let (a, b) = (present[i], present[j]);
                co[a][b] += 1.0;
                co[b][a] += 1.0;
            }
        }
    }

    let mut coh = vec![0.0f64; k];
    for kk in 0..k {
        let idx: Vec<usize> = topw[kk].iter().map(|&w| pos_of_word[&w]).collect();
        let mut s = 0.0;
        for a in 0..idx.len() {
            for b in 0..idx.len() {
                if idx[a] > idx[b] {
                    s += (0.01 + co[idx[a]][idx[b]]).ln() - (0.01 + co[idx[b]][idx[b]]).ln();
                }
            }
        }
        coh[kk] = s;
    }
    coh
}

/// stm `exclusivity`: the FREX-summary exclusivity per topic over top-M words.
/// `frexw` is the frequency/exclusivity weight (stm default 0.7). Returns K values.
pub fn exclusivity(beta: &[Vec<f64>], m: usize, frexw: f64) -> Vec<f64> {
    let k = beta.len();
    let v = if k > 0 { beta[0].len() } else { 0 };
    // Column sums over topics, per word: sum_k beta[k][v].
    let colsum: Vec<f64> = (0..v)
        .map(|vv| (0..k).map(|kk| beta[kk][vv]).sum::<f64>().max(EPS))
        .collect();

    let mut excl = vec![0.0; k];
    for kk in 0..k {
        let tcol = &beta[kk]; // length V (= tbeta column for this topic)
        let matcol: Vec<f64> = (0..v).map(|vv| tcol[vv] / colsum[vv]).collect();
        let ex = rank_avg(&matcol);
        let fr = rank_avg(tcol);
        let frex: Vec<f64> = (0..v)
            .map(|vv| {
                let exr = ex[vv] / v as f64;
                let frr = fr[vv] / v as f64;
                1.0 / (frexw / exr + (1.0 - frexw) / frr)
            })
            .collect();
        let top = top_indices(tcol, m);
        excl[kk] = top.iter().map(|&vv| frex[vv]).sum();
    }
    excl
}

/// Result of [`residual_dispersion`]: stm's `checkResiduals` return, split so the
/// caller (Python / R / Stata) can form the chi-squared p-value from `statistic`
/// and `df` with its own upper-tail routine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualDispersion {
    /// Sample dispersion sigma^2 = D / df. Equals ~1 under the data-generating
    /// process; > 1 is evidence K is too small (Taddy 2012). NaN when df <= 0.
    pub dispersion: f64,
    /// Residual degrees of freedom: `nhat - V - num_params` (stm's `df`).
    pub df: f64,
    /// Number of estimated parameters d = n*(K-1) + K*(V-1) (stm's `d`).
    pub num_params: f64,
    /// The aggregated squared-residual statistic D (stm's `D`), the chi-squared
    /// test statistic for sigma^2 = 1 vs sigma^2 > 1.
    pub statistic: f64,
    /// Taddy's approximate effective count Nhat: expected counts exceeding `tol`,
    /// summed over documents (stm's `Nhat`).
    pub nhat: f64,
}

/// stm `checkResiduals` / Taddy (2012) multinomial residual dispersion: the
/// dispersion of the fitted model's multinomial residuals, used to judge whether
/// K is too small. Under a correct model the dispersion is ~1; a value well above
/// 1 means the latent topics cannot absorb the overdispersion (too few topics).
///
/// `beta` is the K×V topic-word probability matrix, `theta` the N×K per-document
/// topic proportions, `docs` the corpus token-id lists (one per document, aligned
/// to `theta`'s rows). `tol` is Taddy's tolerance for the effective-count
/// degrees-of-freedom approximation (stm default 1/100).
///
/// This is a faithful port of stm's `checkResiduals`: for each document with
/// length `m`, the expected word probability is `q_w = sum_k theta_dk beta_kw`
/// and the squared standardized (Pearson) multinomial residual summed over the
/// full vocabulary is `sum_w (x_w - m q_w)^2 / (m q_w (1 - q_w))`, formed via
/// stm's algebraic split `sum_w (x_w^2 - 2 x_w q_w m)/(m q_w (1-q_w)) + sum_w m
/// q_w/(1-q_w)`. `D` sums that over documents; `Nhat` counts, per document, the
/// words whose expected count `m q_w` exceeds `tol`; the parameter count is
/// `d = n*(K-1) + K*(V-1)`; and `df = Nhat - V - d`, `dispersion = D / df`.
/// `q_w` is clamped to `[1e-12, 1 - 1e-12]` to keep the `1 - q_w` denominators
/// finite (smoothed beta stays well inside that range, so the clamp does not move
/// stm's number).
pub fn residual_dispersion(
    beta: &[Vec<f64>],
    theta: &[Vec<f64>],
    docs: &[Vec<u32>],
    tol: f64,
) -> ResidualDispersion {
    let k = beta.len();
    let v = if k > 0 { beta[0].len() } else { 0 };
    let n = theta.len();

    let mut statistic = 0.0f64;
    let mut nhat = 0.0f64;
    for (d, doc) in docs.iter().enumerate().take(n) {
        let th = &theta[d];
        // q_w = sum_k theta_dk beta_kw, clamped to keep (1 - q) finite.
        let mut q = vec![0.0f64; v];
        for kk in 0..k {
            let t = th.get(kk).copied().unwrap_or(0.0);
            if t == 0.0 {
                continue;
            }
            let row = &beta[kk];
            for (vv, qv) in q.iter_mut().enumerate() {
                *qv += t * row[vv];
            }
        }
        for qv in q.iter_mut() {
            *qv = qv.clamp(1e-12, 1.0 - 1e-12);
        }

        // Observed counts x_w and document length m from the token-id list.
        let mut x = vec![0.0f64; v];
        let mut m = 0.0f64;
        for &tok in doc {
            let w = tok as usize;
            if w < v {
                x[w] += 1.0;
                m += 1.0;
            }
        }
        if m == 0.0 {
            continue;
        }

        for vv in 0..v {
            let qv = q[vv];
            if qv * m > tol {
                nhat += 1.0;
            }
            let xv = x[vv];
            let denom = m * qv * (1.0 - qv);
            statistic += (xv * xv - 2.0 * xv * qv * m) / denom + m * qv / (1.0 - qv);
        }
    }

    let num_params = (n as f64) * ((k as f64) - 1.0) + (k as f64) * ((v as f64) - 1.0);
    let df = nhat - v as f64 - num_params;
    let dispersion = if df > 0.0 { statistic / df } else { f64::NAN };
    ResidualDispersion {
        dispersion,
        df,
        num_params,
        statistic,
        nhat,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_estimate_shrinks_to_uniform_for_small_counts() {
        let p = vec![0.7, 0.2, 0.1];
        assert_eq!(js_estimate(&p, 1.0), vec![1.0 / 3.0; 3]); // ct<=1 -> uniform
        let s = js_estimate(&p, 1000.0); // large count -> close to p
        assert!((s[0] - 0.7).abs() < 0.05);
    }

    #[test]
    fn rank_avg_handles_ties() {
        // values: 10, 20, 20, 40 -> ranks 1, 2.5, 2.5, 4
        assert_eq!(
            rank_avg(&[10.0, 20.0, 20.0, 40.0]),
            vec![1.0, 2.5, 2.5, 4.0]
        );
    }

    fn toy_beta() -> Vec<Vec<f64>> {
        // 2 topics, 4 words. Topic 0 loves words 0,1; topic 1 loves words 2,3.
        vec![vec![0.45, 0.45, 0.05, 0.05], vec![0.05, 0.05, 0.45, 0.45]]
    }

    #[test]
    fn frex_and_score_pick_each_topics_words() {
        let beta = toy_beta();
        let wc = vec![10u32, 10, 10, 10];
        let frex = frex_scores(&beta, &wc, 0.5);
        let tw = top_words(&frex, 2);
        // topic 0's top-2 FREX words are {0,1}; topic 1's are {2,3}.
        let mut t0 = tw[0].clone();
        t0.sort_unstable();
        let mut t1 = tw[1].clone();
        t1.sort_unstable();
        assert_eq!(t0, vec![0, 1]);
        assert_eq!(t1, vec![2, 3]);

        let sc = score_scores(&beta);
        assert!(sc[0][0] > sc[0][2]); // topic 0 scores word 0 above word 2
    }

    #[test]
    fn coherence_higher_when_top_words_co_occur() {
        let beta = toy_beta();
        // Docs where each topic's words co-occur cleanly.
        let docs = vec![
            vec![0u32, 1, 0, 1],
            vec![0, 1],
            vec![2, 3, 2, 3],
            vec![2, 3],
        ];
        let coh = semantic_coherence(&beta, &docs, 2);
        assert_eq!(coh.len(), 2);
        assert!(coh.iter().all(|c| c.is_finite()));

        let excl = exclusivity(&beta, 2, 0.7);
        assert_eq!(excl.len(), 2);
        assert!(excl.iter().all(|e| e.is_finite() && *e > 0.0));
    }

    // Two-cluster planted corpus: words {0,1,2} vs {3,4,5}. Docs 0-3 draw only
    // from cluster A, docs 4-7 only from cluster B.
    fn planted_corpus() -> Vec<Vec<u32>> {
        let mut docs = Vec::new();
        for _ in 0..4 {
            docs.push(vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
        }
        for _ in 0..4 {
            docs.push(vec![3u32, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]);
        }
        docs
    }

    #[test]
    fn residual_dispersion_drops_as_k_reaches_truth() {
        let docs = planted_corpus();
        let n = docs.len();

        // K = 1: a single averaged topic cannot represent the two clusters.
        let beta1 = vec![vec![1.0 / 6.0; 6]];
        let theta1 = vec![vec![1.0]; n];
        let r1 = residual_dispersion(&beta1, &theta1, &docs, 0.01);

        // K = 2: topics aligned to the true clusters, near-one-hot proportions.
        let beta2 = vec![
            vec![0.32, 0.32, 0.32, 0.013, 0.013, 0.014],
            vec![0.013, 0.013, 0.014, 0.32, 0.32, 0.32],
        ];
        let mut theta2 = Vec::new();
        for _ in 0..4 {
            theta2.push(vec![0.98, 0.02]);
        }
        for _ in 0..4 {
            theta2.push(vec![0.02, 0.98]);
        }
        let r2 = residual_dispersion(&beta2, &theta2, &docs, 0.01);

        assert!(r1.df > 0.0 && r2.df > 0.0);
        assert!(r1.dispersion.is_finite() && r2.dispersion.is_finite());
        // Too-few-topics model is overdispersed relative to the well-specified one.
        assert!(
            r1.dispersion > r2.dispersion,
            "K=1 dispersion {} should exceed K=2 dispersion {}",
            r1.dispersion,
            r2.dispersion
        );
        assert!(r1.dispersion > 1.0, "K=1 should be overdispersed (>1)");
        // Parameter count matches stm's d = n*(K-1) + K*(V-1).
        assert_eq!(r1.num_params, n as f64 * 0.0 + 1.0 * 5.0);
        assert_eq!(r2.num_params, n as f64 * 1.0 + 2.0 * 5.0);
    }

    #[test]
    fn residual_dispersion_is_deterministic() {
        let docs = planted_corpus();
        let beta = vec![
            vec![0.32, 0.32, 0.32, 0.013, 0.013, 0.014],
            vec![0.013, 0.013, 0.014, 0.32, 0.32, 0.32],
        ];
        let mut theta = Vec::new();
        for _ in 0..4 {
            theta.push(vec![0.98, 0.02]);
        }
        for _ in 0..4 {
            theta.push(vec![0.02, 0.98]);
        }
        let a = residual_dispersion(&beta, &theta, &docs, 0.01);
        let b = residual_dispersion(&beta, &theta, &docs, 0.01);
        assert_eq!(a, b);
    }
}
