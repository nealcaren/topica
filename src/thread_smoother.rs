//! Hot loop of the Python `topica.threads.ThreadSmoother` switch calibration (#907).
//!
//! [`sequential_log_ml`] is the Dirichlet-multinomial log marginal likelihood of each
//! document's tokens with the topics held fixed, on a grid of prior concentrations, in the
//! sequential Polya-urn form with soft counts. The Python layer calls it once per candidate
//! context share and tree, so it dominates a switch fit.

use rayon::prelude::*;

/// Log marginal likelihood `log p(w_d | a * m_d)` of every document for every `a` in
/// `conc_grid`, with the topic-word matrix fixed.
///
/// `tokens` holds every document's word ids back to back and `offsets` (length `D + 1`)
/// their boundaries. `prior_mean` is `D x K` row-major; a row whose first entry is NaN has
/// no prior and gets a NaN row. `beta_t` is the topic-word matrix transposed to `V x K`
/// row-major (one contiguous row of topic weights per word). Returns `D x G` row-major.
///
/// Per token, `p(w_i | w_<i) = sum_k (a m_k + c_k) beta_kw / (a + i)`, where `c` accumulates
/// each earlier token's topic responsibilities.
pub fn sequential_log_ml(
    tokens: &[u32],
    offsets: &[usize],
    prior_mean: &[f64],
    k: usize,
    conc_grid: &[f64],
    beta_t: &[f64],
) -> Vec<f64> {
    let n_docs = offsets.len().saturating_sub(1);
    let g = conc_grid.len();
    let mut out = vec![f64::NAN; n_docs * g];
    out.par_chunks_mut(g.max(1)).enumerate().for_each_init(
        || (vec![0.0; k], vec![0.0; k]),
        |(counts, p), (d, row)| {
            let m = &prior_mean[d * k..(d + 1) * k];
            if m[0].is_nan() {
                return;
            }
            let ids = &tokens[offsets[d]..offsets[d + 1]];
            for (gi, &a) in conc_grid.iter().enumerate() {
                counts.iter_mut().for_each(|c| *c = 0.0);
                let mut ll = 0.0;
                for (i, &w) in ids.iter().enumerate() {
                    let bw = &beta_t[w as usize * k..(w as usize + 1) * k];
                    let mut tot = 0.0;
                    for t in 0..k {
                        p[t] = (a * m[t] + counts[t]) * bw[t];
                        tot += p[t];
                    }
                    ll += (tot / (a + i as f64)).ln();
                    for t in 0..k {
                        counts[t] += p[t] / tot;
                    }
                }
                row[gi] = ll;
            }
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force Dirichlet-multinomial marginal: sum over every topic assignment.
    fn brute(ids: &[u32], m: &[f64], a: f64, beta_t: &[f64], k: usize) -> f64 {
        let n = ids.len();
        let mut total = 0.0;
        for code in 0..k.pow(n as u32) {
            let mut z = Vec::with_capacity(n);
            let mut c = code;
            for _ in 0..n {
                z.push(c % k);
                c /= k;
            }
            // Polya urn over assignments: prod_i (a m_z + n_z,<i) / (a + i) * beta_{z,w}
            let mut counts = vec![0.0; k];
            let mut p = 1.0;
            for (i, (&w, &zi)) in ids.iter().zip(&z).enumerate() {
                p *= (a * m[zi] + counts[zi]) / (a + i as f64) * beta_t[w as usize * k + zi];
                counts[zi] += 1.0;
            }
            total += p;
        }
        total.ln()
    }

    #[test]
    fn first_token_is_exact_and_nan_rows_stay_nan() {
        // Two topics, three words. beta rows (topics): [.7,.2,.1], [.1,.3,.6].
        let beta_t = vec![0.7, 0.1, 0.2, 0.3, 0.1, 0.6];
        let m = vec![0.25, 0.75, f64::NAN, f64::NAN];
        let out = sequential_log_ml(&[2, 0], &[0, 1, 2], &m, 2, &[0.5, 4.0], &beta_t);
        let want = (0.25 * 0.1 + 0.75 * 0.6f64).ln();
        assert!((out[0] - want).abs() < 1e-14 && (out[1] - want).abs() < 1e-14);
        assert!(out[2].is_nan() && out[3].is_nan());
    }

    #[test]
    fn single_topic_documents_match_the_exact_marginal() {
        // With K = 1 the soft counts are exact counts, so the sequential form is exact.
        let beta_t = vec![0.5, 0.3, 0.2];
        let ids = [0u32, 2, 2, 1];
        let out = sequential_log_ml(&ids, &[0, 4], &[1.0], 1, &[0.3], &beta_t);
        assert!((out[0] - brute(&ids, &[1.0], 0.3, &beta_t, 1)).abs() < 1e-12);
    }
}
