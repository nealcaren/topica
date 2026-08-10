//! TopicsOverTime (ToT): Wang & McCallum, "Topics over Time: A Non-Markov
//! Continuous-Time Model of Topical Trends," KDD 2006.
//!
//! ToT is LDA plus a per-topic Beta density over each document's (normalized)
//! timestamp. Collapsed Gibbs is LDA's, with the per-token conditional multiplied by
//! the topic's Beta time-likelihood of the document's timestamp:
//!
//!   P(z_{di}=k | ·) ∝ (n_{d,k}+α_k) · (n_{k,w}+β)/(n_k+Vβ) · Beta(t_d | ψ_k)
//!
//! The per-topic Beta parameters ψ_k=(ψ_{k,1},ψ_{k,2}) are NOT integrated out; they are
//! re-estimated by METHOD OF MOMENTS from the timestamps of the tokens currently
//! assigned to topic k, once per sweep (paper §4), then held fixed while the sweep
//! resamples assignments. Because the timestamp is constant within a document, the log
//! Beta factor is precomputed per (document, topic) each sweep.
//!
//! Numerics: the Beta factor is the only term that can underflow (a sharply concentrated
//! topic's pdf far from its mode), so per document we compute `log_beta[k]` and store the
//! max-shifted `beta_factor[k] = exp(log_beta[k] − max_k log_beta[k])` (in (0,1], max
//! exactly 1); the per-token score is then `LDA_conditional[k] · beta_factor[k]` in
//! linear space — mathematically the log-domain conditional up to the per-doc constant
//! `exp(−max)` that cancels in the categorical normalization, but with no ln/exp in the
//! per-token loop (the LDA cost). The max-factor topic keeps its full, strictly-positive
//! LDA weight, so the total is never all-zero. ψ's concentration is capped (a tiny
//! within-topic timestamp variance would otherwise send ψ→∞ and lock the chain), and a
//! zero-token, moment-invalid, or near-degenerate topic falls back to ψ=(1,1) (uniform →
//! Beta factor 1, that topic behaves as LDA).
//!
//! Timestamps are min-max normalized to [0,1] by the binding (which keeps the original
//! range for reporting) and clipped to (ε,1−ε). Determinism: reuse `TopicModel`'s
//! word-topic machinery + the project RNG; documents in fixed order; single-threaded.

use crate::estimator::{Estimator, ModelFamily};
use crate::model::TopicModel;
use rand::Rng;

/// Largest Beta shape parameter the method-of-moments update may return, so a
/// near-degenerate (tiny-variance) topic cannot drive ψ to infinity and freeze the
/// sampler.
const PSI_MAX: f64 = 1000.0;

/// Fitted Topics-over-Time state read back by the PyO3 binding. Times/peaks here are in
/// the NORMALIZED [0,1] scale; the binding maps peaks/means back to original units.
pub struct TopicsOverTimeModel {
    pub num_topics: usize,
    pub alpha: Vec<f64>,
    pub beta: f64,
    /// Topic-word point estimate φ (K×V).
    pub topic_word: Vec<Vec<f64>>,
    /// Document-topic point estimate θ (D×K), each row sums to 1.
    pub doc_topic: Vec<Vec<f64>>,
    /// Per-topic Beta parameters (ψ_{k,1}, ψ_{k,2}) over normalized time.
    pub psi: Vec<[f64; 2]>,
    /// Per-topic peak time in NORMALIZED [0,1] (mode when it is interior, else the
    /// boundary; NaN for a U-shaped/uniform topic with no single mode).
    pub peak_norm: Vec<f64>,
    /// Per-topic Beta mean in NORMALIZED [0,1] = ψ1/(ψ1+ψ2).
    pub mean_norm: Vec<f64>,
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

/// ln B(a,b) = lnΓ(a)+lnΓ(b)−lnΓ(a+b), the log Beta function normalizer.
fn ln_beta_fn(a: f64, b: f64) -> f64 {
    crate::output::log_gamma(a) + crate::output::log_gamma(b) - crate::output::log_gamma(a + b)
}

/// A fitted Beta whose smaller shape falls below this is treated as having no usable
/// temporal signal and collapses to the uniform fallback (1,1). This catches the
/// near-maximal-variance case where method of moments drives the concentration c→0: the
/// shapes would otherwise underflow toward 0 (an uninterpretable, effectively
/// no-localization Beta whose reported "mean" is not a peak), so we report it honestly
/// as uniform instead. Normal directional or moderately U-shaped topics (shapes ≳ 0.01)
/// are preserved.
const MIN_SHAPE: f64 = 1e-2;

/// Method-of-moments Beta fit from a mean/variance, with the Gate-A guards. Returns
/// (1,1) (uniform) for a zero-count topic, a moment-invalid topic (variance at or above
/// the Beta-representable maximum), or a near-degenerate fit whose concentration
/// collapses (see [`MIN_SHAPE`]). The concentration c=a+b is capped (not the shapes
/// independently) so a very peaked topic keeps its mean and mode when it hits PSI_MAX.
fn mom_beta(count: f64, sum_t: f64, sum_t2: f64) -> [f64; 2] {
    if count <= 0.0 {
        return [1.0, 1.0];
    }
    let mean = sum_t / count;
    let var = (sum_t2 / count - mean * mean).max(0.0);
    let span = mean * (1.0 - mean);
    // var must be strictly inside (0, mean(1-mean)) for a valid Beta moment match.
    if var <= 0.0 || var >= span {
        return [1.0, 1.0];
    }
    let c = span / var - 1.0; // total concentration a+b, > 0 here
    let mut a = mean * c;
    let mut b = (1.0 - mean) * c;
    // Cap the concentration, preserving the mean a/(a+b) (and hence the mode): scale
    // both shapes by the same factor rather than clamping each independently (which would
    // pull a very peaked topic's estimate toward the center).
    let m = a.max(b);
    if m > PSI_MAX {
        let f = PSI_MAX / m;
        a *= f;
        b *= f;
    }
    // Near-degenerate (concentration collapsed to ~0): report as uniform, not a
    // meaningless spike at the extremes.
    if a < MIN_SHAPE || b < MIN_SHAPE {
        return [1.0, 1.0];
    }
    [a, b]
}

/// Peak (mode) of Beta(a,b) on [0,1]. Interior mode when a>1 and b>1; the boundary with
/// the higher density for a monotone density (mode 0 when the density decreases, mode 1
/// when it increases); NaN when the density is U-shaped (a<1 and b<1) or uniform
/// (a=b=1), which have no single mode. A unit shape paired with the other <1 is still
/// monotone (e.g. Beta(1, b<1) increases → mode 1), so those map to a boundary, not NaN.
fn beta_peak(a: f64, b: f64) -> f64 {
    if a > 1.0 && b > 1.0 {
        (a - 1.0) / (a + b - 2.0)
    } else if (a < 1.0 && b < 1.0) || (a == 1.0 && b == 1.0) {
        f64::NAN // U-shaped or uniform: no single mode
    } else if a <= 1.0 && b >= 1.0 {
        0.0 // decreasing (or flat-then-decreasing) density → left boundary
    } else {
        1.0 // a >= 1.0 && b <= 1.0: increasing density → right boundary
    }
}

/// Fit ToT by collapsed Gibbs.
///
/// - `docs`: word-id documents. `num_types`: vocabulary size V.
/// - `times`: per-document normalized timestamps in [0,1], parallel to `docs`, already
///   clipped away from the exact boundaries by the binding.
/// - `alpha`: length-K doc-topic Dirichlet. `beta`: symmetric topic-word Dirichlet.
#[allow(clippy::too_many_arguments)]
pub fn fit<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    times: &[f64],
    num_topics: usize,
    alpha: Vec<f64>,
    beta: f64,
    iters: usize,
    rng: &mut R,
) -> TopicsOverTimeModel {
    let k = num_topics;
    let v = num_types;
    let d = docs.len();
    let alpha_sum: f64 = alpha.iter().sum();
    let beta_sum = beta * v as f64;

    // Reuse TopicModel for word-topic counts + per-token topic assignments.
    let mut wt = TopicModel::new(k, alpha_sum, beta, v);
    let mut type_totals = vec![0usize; v];
    for doc in docs {
        for &w in doc {
            type_totals[w as usize] += 1;
        }
    }
    // Fail fast before packing if the most frequent word could exceed the packed
    // (count, topic) ceiling — otherwise increment_type_topic would silently corrupt an
    // entry mid-sampling (and a corrupted topic index could then index nkw_buf out of
    // bounds). This mirrors TopicModel::initialize, whose guard we skip by building the
    // count store by hand.
    let ceiling = crate::model::max_packable_count(wt.topic_bits) as usize;
    if let Some(&worst) = type_totals.iter().max() {
        assert!(
            worst <= ceiling,
            "topica: a word occurs {worst} times, but the packed (count, topic) table \
             for num_topics={k} can represent at most {ceiling} occurrences of a single \
             word in one topic (32-bit packing). Reduce num_topics or split the corpus."
        );
    }
    wt.type_topic_counts = type_totals.iter().map(|&n| vec![0u32; k.min(n)]).collect();
    wt.tokens_per_topic = vec![0u32; k];
    wt.doc_topics = docs.iter().map(|doc| vec![0u32; doc.len()]).collect();

    // Per-document topic counts n_{d,k} (for the doc-topic term and MoM accumulation).
    let mut n_dk = vec![vec![0u32; k]; d];

    // Precompute per-doc log-time terms for the Beta factor.
    let ln_t: Vec<f64> = times.iter().map(|&t| t.ln()).collect();
    let ln_1mt: Vec<f64> = times.iter().map(|&t| (1.0 - t).ln()).collect();

    // --- initialize: random topics ---
    for (di, doc) in docs.iter().enumerate() {
        for (pos, &w) in doc.iter().enumerate() {
            let t = rng.gen_range(0..k);
            wt.doc_topics[di][pos] = t as u32;
            wt.tokens_per_topic[t] += 1;
            wt.increment_type_topic(w as usize, t);
            n_dk[di][t] += 1;
        }
    }

    let mut psi = vec![[1.0f64, 1.0f64]; k];
    let mut log_beta = vec![0.0f64; k]; // per-topic log Beta factor for the current doc
    let mut beta_factor = vec![0.0f64; k]; // exp(log_beta - max), reused per doc
    let mut ln_norm = vec![0.0f64; k]; // -lnB(psi_k) per topic
    let mut scores = vec![0.0f64; k];
    // Dense scratch for the current word's topic counts. `TopicModel`'s packed sparse
    // store is O(distinct-topics) per random lookup, which would make a dense K-topic
    // scan per token quadratic; scatter the word's counts here once (O(distinct+K)).
    let mut nkw_buf = vec![0u32; k];
    let eval_stride = (iters / 25).max(1);
    let mut fit_history = Vec::new();

    for it in 0..iters {
        // --- estimate psi by method of moments from current assignments, hold for sweep
        for kk in 0..k {
            let (mut s1, mut s2, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
            for di in 0..d {
                let c = n_dk[di][kk] as f64;
                if c > 0.0 {
                    s1 += c * times[di];
                    s2 += c * times[di] * times[di];
                    cnt += c;
                }
            }
            psi[kk] = mom_beta(cnt, s1, s2);
            ln_norm[kk] = -ln_beta_fn(psi[kk][0], psi[kk][1]);
        }

        for di in 0..docs.len() {
            // Per-(doc,topic) Beta time factor. The timestamp is constant within a
            // document, so this is computed once per document, not per token. The LDA
            // term is always well-scaled, so only the Beta factor can underflow (a
            // sharply-peaked topic far from its mode); shifting the log factor by its
            // per-doc max (→ factor in (0,1], max exactly 1) makes the linear-space
            // product safe, and the shift cancels in the categorical normalization. This
            // keeps the per-token inner loop free of ln/exp (the LDA cost), while still
            // being exactly the log-domain conditional up to the per-doc constant.
            let mut max_lb = f64::NEG_INFINITY;
            for kk in 0..k {
                let lb =
                    (psi[kk][0] - 1.0) * ln_t[di] + (psi[kk][1] - 1.0) * ln_1mt[di] + ln_norm[kk];
                log_beta[kk] = lb;
                if lb > max_lb {
                    max_lb = lb;
                }
            }
            for kk in 0..k {
                beta_factor[kk] = (log_beta[kk] - max_lb).exp();
            }
            let doc = &docs[di];
            for pos in 0..doc.len() {
                let w = doc[pos] as usize;
                let old = wt.doc_topics[di][pos] as usize;
                wt.decrement_type_topic(w, old);
                wt.tokens_per_topic[old] -= 1;
                n_dk[di][old] -= 1;

                // Scatter this word's (post-decrement) packed counts into the dense
                // buffer so the K-topic scan below is O(1) per topic, not a linear probe.
                for &entry in &wt.type_topic_counts[w] {
                    if entry == 0 {
                        break;
                    }
                    nkw_buf[(entry & wt.topic_mask) as usize] = entry >> wt.topic_bits;
                }

                // score = LDA conditional (linear) × shifted Beta time factor. The
                // max-factor topic keeps its full LDA weight (factor 1), so the total is
                // always strictly positive — no all-zero underflow.
                let mut total = 0.0f64;
                for kk in 0..k {
                    let lda = (n_dk[di][kk] as f64 + alpha[kk]) * (nkw_buf[kk] as f64 + beta)
                        / (wt.tokens_per_topic[kk] as f64 + beta_sum);
                    let s = lda * beta_factor[kk];
                    scores[kk] = s;
                    total += s;
                }
                let mut r = rng.gen::<f64>() * total;
                let mut chosen = k - 1;
                for (kk, &s) in scores.iter().enumerate() {
                    r -= s;
                    if r <= 0.0 {
                        chosen = kk;
                        break;
                    }
                }

                // Reset only the entries we touched (keeps the buffer zeroed cheaply).
                for &entry in &wt.type_topic_counts[w] {
                    if entry == 0 {
                        break;
                    }
                    nkw_buf[(entry & wt.topic_mask) as usize] = 0;
                }

                wt.increment_type_topic(w, chosen);
                wt.tokens_per_topic[chosen] += 1;
                wt.doc_topics[di][pos] = chosen as u32;
                n_dk[di][chosen] += 1;
            }
        }

        if it % eval_stride == 0 || it == iters - 1 {
            let ll = held_in_loglik(
                docs, &wt, &n_dk, &psi, &ln_norm, &ln_t, &ln_1mt, &alpha, beta, beta_sum, k,
            );
            fit_history.push((it + 1, ll));
        }
    }

    // Final psi from the terminal assignments (so returned psi matches the fit).
    for kk in 0..k {
        let (mut s1, mut s2, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
        for di in 0..d {
            let c = n_dk[di][kk] as f64;
            if c > 0.0 {
                s1 += c * times[di];
                s2 += c * times[di] * times[di];
                cnt += c;
            }
        }
        psi[kk] = mom_beta(cnt, s1, s2);
        ln_norm[kk] = -ln_beta_fn(psi[kk][0], psi[kk][1]);
    }

    // Recompute the terminal log-likelihood under the FINAL (re-estimated) psi so the
    // last fit_history entry is consistent with the returned psi/doc_topic/topic_word,
    // not the start-of-last-sweep psi.
    if let Some(last) = fit_history.last_mut() {
        last.1 = held_in_loglik(
            docs, &wt, &n_dk, &psi, &ln_norm, &ln_t, &ln_1mt, &alpha, beta, beta_sum, k,
        );
    }

    let topic_word = wt.topic_word();
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let denom = docs[di].len() as f64 + alpha_sum;
            if docs[di].is_empty() {
                return vec![1.0 / k as f64; k];
            }
            (0..k)
                .map(|t| (n_dk[di][t] as f64 + alpha[t]) / denom)
                .collect()
        })
        .collect();
    let peak_norm: Vec<f64> = psi.iter().map(|p| beta_peak(p[0], p[1])).collect();
    let mean_norm: Vec<f64> = psi.iter().map(|p| p[0] / (p[0] + p[1])).collect();

    TopicsOverTimeModel {
        num_topics: k,
        alpha,
        beta,
        topic_word,
        doc_topic,
        psi,
        peak_norm,
        mean_norm,
        fit_history,
        converged: false,
    }
}

/// Held-in token log-likelihood under current point estimates: a convergence
/// diagnostic. Σ_i log[ Σ_k θ_{d,k} φ_{k,w} Beta(t_d|ψ_k) ] with the current smoothed
/// estimates (log-domain, max-shifted).
#[allow(clippy::too_many_arguments)]
fn held_in_loglik(
    docs: &[Vec<u32>],
    wt: &TopicModel,
    n_dk: &[Vec<u32>],
    psi: &[[f64; 2]],
    ln_norm: &[f64],
    ln_t: &[f64],
    ln_1mt: &[f64],
    alpha: &[f64],
    beta: f64,
    beta_sum: f64,
    k: usize,
) -> f64 {
    let alpha_sum: f64 = alpha.iter().sum();
    let mut ll = 0.0f64;
    let mut logs = vec![0.0f64; k];
    for (di, doc) in docs.iter().enumerate() {
        if doc.is_empty() {
            continue;
        }
        let denom = doc.len() as f64 + alpha_sum;
        for &w in doc {
            let w = w as usize;
            let mut maxs = f64::NEG_INFINITY;
            for kk in 0..k {
                let theta = (n_dk[di][kk] as f64 + alpha[kk]) / denom;
                let phi = (wt.get_type_topic_count(w, kk) as f64 + beta)
                    / (wt.tokens_per_topic[kk] as f64 + beta_sum);
                let lb =
                    (psi[kk][0] - 1.0) * ln_t[di] + (psi[kk][1] - 1.0) * ln_1mt[di] + ln_norm[kk];
                let s = theta.ln() + phi.ln() + lb;
                logs[kk] = s;
                if s > maxs {
                    maxs = s;
                }
            }
            let mut sum = 0.0f64;
            for &s in &logs {
                sum += (s - maxs).exp();
            }
            ll += maxs + sum.ln();
        }
    }
    ll
}

impl Estimator for TopicsOverTimeModel {
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
        self.fit_history.clone()
    }
    fn converged(&self) -> Option<bool> {
        Some(self.converged)
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Planted continuous-time corpus: K topics, each a disjoint word block AND a
    /// distinct time window (topic k's docs cluster near t≈(k+0.5)/K). Recovering the
    /// per-topic peak ordering is the ToT-specific signal.
    fn planted(k: usize) -> (Vec<Vec<u32>>, usize, Vec<f64>) {
        let block = 5usize;
        let v = k * block;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut docs = Vec::new();
        let mut times = Vec::new();
        for topic in 0..k {
            let center = (topic as f64 + 0.5) / k as f64;
            for _ in 0..40 {
                let doc: Vec<u32> = (0..15)
                    .map(|i| (topic * block + (i % block)) as u32)
                    .collect();
                docs.push(doc);
                // timestamp near the topic's center, small jitter, clipped to (0,1)
                let t = (center + (rng.gen::<f64>() - 0.5) * 0.1).clamp(0.01, 0.99);
                times.push(t);
            }
        }
        (docs, v, times)
    }

    #[test]
    fn tot_recovers_topics_and_time_order() {
        let k = 3;
        let (docs, v, times) = planted(k);
        let alpha = vec![50.0 / k as f64; k];
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let m = fit(&docs, v, &times, k, alpha, 0.1, 300, &mut rng);
        // Each topic concentrates on its 5-word block.
        for row in &m.topic_word {
            let top5: f64 = {
                let mut r = row.clone();
                r.sort_by(|a, b| b.partial_cmp(a).unwrap());
                r[..5].iter().sum()
            };
            assert!(top5 > 0.9, "topic not block-concentrated: {top5}");
        }
        // Recovered peak times are distinct and span the range (temporal signal present).
        let mut peaks: Vec<f64> = m.mean_norm.clone();
        peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            peaks[0] < 0.4 && peaks[k - 1] > 0.6,
            "peaks not spread: {peaks:?}"
        );
        // psi is concentrated (not the uniform fallback) for these tight time clusters.
        assert!(
            m.psi.iter().all(|p| p[0] + p[1] > 2.5),
            "psi collapsed to uniform: {:?}",
            m.psi
        );
    }

    #[test]
    fn tot_is_deterministic() {
        let k = 3;
        let (docs, v, times) = planted(k);
        let alpha = vec![50.0 / k as f64; k];
        let a = fit(
            &docs,
            v,
            &times,
            k,
            alpha.clone(),
            0.1,
            80,
            &mut ChaCha8Rng::seed_from_u64(1),
        );
        let b = fit(
            &docs,
            v,
            &times,
            k,
            alpha,
            0.1,
            80,
            &mut ChaCha8Rng::seed_from_u64(1),
        );
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        assert_eq!(a.psi, b.psi);
    }

    #[test]
    fn tot_conforms() {
        let k = 3;
        let (docs, v, times) = planted(k);
        let alpha = vec![50.0 / k as f64; k];
        let m = fit(
            &docs,
            v,
            &times,
            k,
            alpha,
            0.1,
            20,
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn tot_constant_time_falls_back_to_uniform() {
        // All documents share one timestamp → zero variance → psi=(1,1) (LDA behavior).
        let k = 2;
        let (docs, v, _t) = planted(k);
        let times = vec![0.5f64; docs.len()];
        let alpha = vec![25.0; k];
        let m = fit(
            &docs,
            v,
            &times,
            k,
            alpha,
            0.1,
            40,
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        assert!(
            m.psi.iter().all(|p| p[0] == 1.0 && p[1] == 1.0),
            "psi: {:?}",
            m.psi
        );
        assert!(
            m.peak_norm.iter().all(|x| x.is_nan()),
            "uniform topics have no peak"
        );
    }

    #[test]
    fn beta_peak_cases() {
        assert!((beta_peak(3.0, 3.0) - 0.5).abs() < 1e-12); // symmetric interior mode
        assert!((beta_peak(2.0, 6.0) - 1.0 / 6.0).abs() < 1e-12); // skewed: (a-1)/(a+b-2)
        assert_eq!(beta_peak(0.5, 3.0), 0.0); // monotone decreasing → left boundary
        assert_eq!(beta_peak(3.0, 0.5), 1.0); // monotone increasing → right boundary
        assert!(beta_peak(0.5, 0.5).is_nan()); // U-shaped → no single mode
        assert!(beta_peak(1.0, 1.0).is_nan()); // uniform → no mode
                                               // Unit shape paired with the other < 1 is still monotone, NOT NaN (the bug
                                               // Reviewer A caught): Beta(1, b<1) increases → mode 1; Beta(a<1, 1) decreases → 0.
        assert_eq!(beta_peak(1.0, 0.5), 1.0);
        assert_eq!(beta_peak(0.5, 1.0), 0.0);
        assert_eq!(beta_peak(1.0, 5.0), 0.0); // a=1, b>1 decreasing → left boundary
        assert_eq!(beta_peak(5.0, 1.0), 1.0); // a>1, b=1 increasing → right boundary
    }

    #[test]
    fn mom_beta_degenerate_variance_collapses_to_uniform() {
        // Variance just under the max (span) drives concentration c→0, so the shapes
        // would underflow toward 0; that must collapse to the uniform fallback, not a
        // meaningless near-zero spike (the analytical trap Reviewer D hit on real data).
        let mean = 0.5;
        let span = mean * (1.0 - mean); // 0.25
        let var = span * 0.9999; // barely representable → tiny concentration
        let p = mom_beta(1.0, mean, var + mean * mean);
        assert_eq!(
            p,
            [1.0, 1.0],
            "near-max variance should be uniform, got {p:?}"
        );
    }

    #[test]
    fn mom_beta_caps_concentration_preserving_mean() {
        // Tiny variance → huge concentration; capping must preserve the mean (mode),
        // not clamp each shape independently (which would pull the mean toward 0.5).
        let mean = 0.2;
        let var = 1e-9; // enormous concentration, will hit PSI_MAX
        let p = mom_beta(1.0, mean, var + mean * mean);
        let got_mean = p[0] / (p[0] + p[1]);
        assert!(
            (got_mean - mean).abs() < 1e-6,
            "mean not preserved: {got_mean}"
        );
        assert!(
            p[0].max(p[1]) <= PSI_MAX + 1e-9,
            "concentration not capped: {p:?}"
        );
    }

    /// Discriminating test (Reviewer A's blocker): prove the Beta TIME factor actually
    /// drives assignment, not just the words. Two eras share a heavily OVERLAPPING word
    /// distribution (a weak 65/35 lean toward each half of the vocab), so words alone
    /// separate them only weakly; the eras are cleanly separated in time. We fit the SAME
    /// corpus twice — once with the real timestamps, once with a constant timestamp
    /// (time-blind, i.e. plain LDA) — and require the real-time fit to recover the eras
    /// substantially better. If the time factor were ignored the two accuracies would
    /// match; the gap IS the causal effect of the Beta factor.
    fn era_accuracy(m: &TopicsOverTimeModel, era: &[usize]) -> f64 {
        let dom = |row: &[f64]| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        let t0 = dom(&m.doc_topic[0]); // whichever topic doc 0 (era 0) landed in
        let mut correct = 0;
        for (di, &e) in era.iter().enumerate() {
            let want = if e == 0 { t0 } else { 1 - t0 };
            if dom(&m.doc_topic[di]) == want {
                correct += 1;
            }
        }
        let frac = correct as f64 / era.len() as f64;
        frac.max(1.0 - frac) // label-invariant: 0.5 = chance, 1.0 = perfect
    }

    #[test]
    fn tot_time_factor_drives_assignment() {
        let v = 10usize;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut docs = Vec::new();
        let mut times = Vec::new();
        let mut era = Vec::new();
        // Era 0 leans to words 0..5, era 1 to words 5..10 — but only 65/35, so words
        // alone are a weak signal; the eras are far apart in time.
        for _ in 0..120 {
            for e in 0..2 {
                let doc: Vec<u32> = (0..25)
                    .map(|_| {
                        let own_half = rng.gen::<f64>() < 0.55;
                        let lo = if (e == 0) == own_half { 0 } else { 5 };
                        lo + rng.gen_range(0..5)
                    })
                    .collect();
                docs.push(doc);
                let t = if e == 0 {
                    rng.gen::<f64>() * 0.15 + 0.05
                } else {
                    rng.gen::<f64>() * 0.15 + 0.80
                };
                times.push(t.clamp(0.01, 0.99));
                era.push(e);
            }
        }
        let alpha = vec![25.0, 25.0];
        let real = fit(
            &docs,
            v,
            &times,
            2,
            alpha.clone(),
            0.1,
            400,
            &mut ChaCha8Rng::seed_from_u64(13),
        );
        // Time-blind: constant timestamp → Beta collapses to uniform → plain LDA.
        let blind_times = vec![0.5f64; times.len()];
        let blind = fit(
            &docs,
            v,
            &blind_times,
            2,
            alpha,
            0.1,
            400,
            &mut ChaCha8Rng::seed_from_u64(13),
        );

        let acc_real = era_accuracy(&real, &era);
        let acc_blind = era_accuracy(&blind, &era);
        // The time factor must materially improve era recovery over the time-blind fit,
        // and the real-time fit must recover the eras well.
        assert!(
            acc_real > acc_blind + 0.15 && acc_real > 0.7,
            "time factor did not drive assignment: real={acc_real:.3} blind={acc_blind:.3}"
        );
    }

    #[test]
    fn mom_beta_matches_moments() {
        // mean 0.25, var 0.02 → c = 0.25*0.75/0.02 - 1 = 8.375; a=2.09375, b=6.28125
        let mean = 0.25;
        let var = 0.02;
        let p = mom_beta(1.0, mean, var + mean * mean);
        assert!((p[0] - 2.09375).abs() < 1e-9, "a={}", p[0]);
        assert!((p[1] - 6.28125).abs() < 1e-9, "b={}", p[1]);
    }
}
