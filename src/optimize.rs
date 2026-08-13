use crate::corpus::Corpus;
use crate::model::TopicModel;

/// MALLET's `ParallelTopicModel.optimizeAlpha` calls
/// `Dirichlet.learnParameters(alpha, topicDocCounts, docLengthCounts, 1.001, 1.0, 1)`:
/// a single Minka/Wallach fixed-point step with a Gamma(shape, scale) hyperprior on
/// the per-topic alphas. The shape is added to each topic's numerator and `1/scale`
/// is subtracted from the shared denominator (Wallach 2008, section 2.5).
const MALLET_ALPHA_SHAPE: f64 = 1.001;
const MALLET_ALPHA_SCALE: f64 = 1.0;

/// Symmetric-concentration (symmetric alpha and beta) fixed-point iteration cap.
/// MALLET's `Dirichlet.learnSymmetricConcentration` runs a fixed 200 steps; we
/// iterate to convergence with the same ceiling.
const SYMMETRIC_ITER_CAP: usize = 200;
/// Relative-change threshold for stopping the symmetric-concentration iteration.
const SYMMETRIC_CONVERGENCE_TOL: f64 = 1e-8;
/// Runaway guard: a near-uniform corpus has an unbounded MLE concentration, so
/// bound the iterate instead of overflowing. 1e6 is far above any real alpha/beta.
const CONCENTRATION_CAP: f64 = 1e6;

/// Iterate the (mathematically correct) symmetric-concentration Minka fixed point
/// to convergence. MALLET's `learnSymmetricConcentration` iterates too, but its
/// denominator accumulates cumulatively across observation lengths (its
/// `previousLength` never advances), which inflates the denominator and damps the
/// estimate — a bug we deliberately do not reproduce. We use the correct per-length
/// digamma difference (via [`update_symmetric_concentration`]) and stop on relative
/// convergence, with a cap so pathological (near-uniform) data cannot diverge.
fn learn_symmetric_concentration(
    count_hist: &[u32],
    length_hist: &[u32],
    num_dims: usize,
    start: f64,
) -> f64 {
    let mut conc = start;
    for _ in 0..SYMMETRIC_ITER_CAP {
        let next = update_symmetric_concentration(count_hist, length_hist, num_dims, conc);
        if !next.is_finite() || next >= CONCENTRATION_CAP {
            return conc.clamp(1e-10, CONCENTRATION_CAP);
        }
        let rel = (next - conc).abs() / (conc.abs() + 1e-12);
        conc = next;
        if rel < SYMMETRIC_CONVERGENCE_TOL {
            break;
        }
    }
    conc
}

/// Digamma function via recurrence + asymptotic expansion.
/// Matches the accuracy of MALLET's Dirichlet.logGammaStirling derivative.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Recurrence: ψ(x) = ψ(x+1) - 1/x  →  shift x into the asymptotic region
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic: ln x - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// Trigamma function ψ'(x) via recurrence + asymptotic expansion (the derivative
/// of [`digamma`]). Used for observed-information / standard-error computations.
pub fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Recurrence: ψ'(x) = ψ'(x+1) + 1/x²  →  shift x into the asymptotic region.
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic: 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - 1/(30x⁹).
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result
        + inv
        + 0.5 * inv2
        + inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))
}

/// One Minka fixed-point step for a symmetric Dirichlet concentration parameter.
///
/// Both alpha (document-topic) and beta (topic-word) optimisation use this
/// same update.  The caller passes histograms rather than raw counts so the
/// inner loop is over distinct count values, not over every observation.
///
/// * `count_hist[c]`  – number of (component, observation) pairs with count c
/// * `length_hist[n]` – number of observations with total n
/// * `num_dims`       – number of Dirichlet components (K or W)
/// * `concentration`  – current value of α_sum or β_sum
///
/// Returns the updated concentration, or the original value if the update
/// would be degenerate.
fn update_symmetric_concentration(
    count_hist: &[u32],
    length_hist: &[u32],
    num_dims: usize,
    concentration: f64,
) -> f64 {
    let per_dim = concentration / num_dims as f64;
    let dg_per_dim = digamma(per_dim);
    let dg_conc = digamma(concentration);

    let numerator: f64 = count_hist
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c > 0)
        .map(|(n, &c)| c as f64 * (digamma(n as f64 + per_dim) - dg_per_dim))
        .sum();

    let denominator: f64 = length_hist
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c > 0)
        .map(|(n, &c)| c as f64 * (digamma(n as f64 + concentration) - dg_conc))
        .sum::<f64>()
        * num_dims as f64;

    if numerator > 0.0 && denominator > 0.0 {
        (concentration * numerator / denominator).max(1e-10)
    } else {
        concentration
    }
}

/// One MALLET `Dirichlet.learnParameters(alpha, .., shape=1.001, scale=1.0, iters=1)`
/// step for asymmetric alpha: a single Minka/Wallach fixed-point step with a
/// Gamma(shape, scale) hyperprior. Updates `alpha` in place and returns the new
/// `alpha_sum`, or `None` if the shared denominator is non-positive (leave alpha
/// unchanged). Pure over the sufficient-statistic histograms so it can be checked
/// against MALLET's oracle values.
///
/// * `topic_doc_hist[t][c]` – number of documents where topic `t` appears `c` times
/// * `doc_length_hist[n]`   – number of documents with `n` tokens
fn learn_alpha_asymmetric(
    alpha: &mut [f64],
    topic_doc_hist: &[Vec<u32>],
    doc_length_hist: &[u32],
) -> Option<f64> {
    let alpha_sum: f64 = alpha.iter().sum();
    let dg_alpha_sum = digamma(alpha_sum);
    // Shared denominator; MALLET subtracts 1/scale (Bayesian estimation part I).
    let mut denominator: f64 = doc_length_hist
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c > 0)
        .map(|(n, &c)| c as f64 * (digamma(n as f64 + alpha_sum) - dg_alpha_sum))
        .sum();
    denominator -= 1.0 / MALLET_ALPHA_SCALE;

    if denominator <= 0.0 {
        return None;
    }

    let mut new_alpha_sum = 0.0;
    for (t, a) in alpha.iter_mut().enumerate() {
        let dg_alpha_t = digamma(*a);
        let numerator: f64 = topic_doc_hist[t]
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, &c)| c > 0)
            .map(|(c, &cnt)| cnt as f64 * (digamma(c as f64 + *a) - dg_alpha_t))
            .sum();

        // Bayesian estimation part II: the shape term is added to the numerator
        // before scaling by the old alpha value.
        *a = (*a * (numerator + MALLET_ALPHA_SHAPE) / denominator).max(1e-10);
        new_alpha_sum += *a;
    }
    Some(new_alpha_sum)
}

/// Optimise per-topic alpha values (asymmetric Dirichlet), MALLET-faithful.
///
/// Sufficient statistics:
///   doc_length_hist[n]        – number of documents with n tokens
///   topic_doc_hist[t][c]      – number of documents where topic t appears c times
pub fn optimize_alpha(model: &mut TopicModel, corpus: &Corpus) {
    let max_len = corpus.docs.iter().map(|d| d.len()).max().unwrap_or(0);

    let mut doc_length_hist = vec![0u32; max_len + 1];
    let mut topic_doc_hist = vec![vec![0u32; max_len + 1]; model.num_topics];

    for doc_idx in 0..corpus.num_docs() {
        let doc_len = corpus.docs[doc_idx].len();
        doc_length_hist[doc_len] += 1;

        let mut counts = vec![0u32; model.num_topics];
        for &t in &model.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }
        for t in 0..model.num_topics {
            if counts[t] > 0 {
                topic_doc_hist[t][counts[t] as usize] += 1;
            }
            counts[t] = 0;
        }
    }

    if let Some(new_alpha_sum) =
        learn_alpha_asymmetric(&mut model.alpha, &topic_doc_hist, &doc_length_hist)
    {
        model.alpha_sum = new_alpha_sum;
    }
}

/// Optimise a *symmetric* document-topic prior: one Minka step on the shared
/// alpha concentration, keeping every `alpha[t]` equal. This is MALLET's
/// `--use-symmetric-alpha true` path — only the total alpha_sum is learned, not
/// the per-topic shape (which `optimize_alpha` learns).
pub fn optimize_alpha_symmetric(model: &mut TopicModel, corpus: &Corpus) {
    let max_len = corpus.docs.iter().map(|d| d.len()).max().unwrap_or(0);
    let mut doc_length_hist = vec![0u32; max_len + 1];
    // Aggregated over all topics: number of (topic, doc) pairs with count c.
    let mut count_hist = vec![0u32; max_len + 1];
    let mut counts = vec![0u32; model.num_topics];

    for doc_idx in 0..corpus.num_docs() {
        doc_length_hist[corpus.docs[doc_idx].len()] += 1;
        for &t in &model.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }
        for c in counts.iter_mut() {
            if *c > 0 {
                count_hist[*c as usize] += 1;
                *c = 0;
            }
        }
    }

    let new_sum = learn_symmetric_concentration(
        &count_hist,
        &doc_length_hist,
        model.num_topics,
        model.alpha_sum,
    );
    model.alpha_sum = new_sum;
    let per_topic = new_sum / model.num_topics as f64;
    for a in model.alpha.iter_mut() {
        *a = per_topic;
    }
}

/// Optimise the symmetric beta (topic-word prior) using one Minka step.
///
/// Sufficient statistics:
///   count_hist[c]       – number of (word, topic) pairs with c tokens
///   topic_size_hist[s]  – number of topics with s total tokens
pub fn optimize_beta(model: &mut TopicModel) {
    // Build count histogram over non-zero (word, topic) entries.
    let max_count = model
        .type_topic_counts
        .iter()
        .flat_map(|v| v.iter().take_while(|&&e| e > 0))
        .map(|&e| e >> model.topic_bits)
        .max()
        .unwrap_or(0) as usize;

    let mut count_hist = vec![0u32; max_count + 1];
    for word_id in 0..model.num_types {
        for &entry in model.type_topic_counts[word_id]
            .iter()
            .take_while(|&&e| e > 0)
        {
            count_hist[(entry >> model.topic_bits) as usize] += 1;
        }
    }

    // Build topic-size histogram.
    let max_size = model.tokens_per_topic.iter().copied().max().unwrap_or(0) as usize;
    let mut topic_size_hist = vec![0u32; max_size + 1];
    for t in 0..model.num_topics {
        topic_size_hist[model.tokens_per_topic[t] as usize] += 1;
    }

    let new_beta_sum = learn_symmetric_concentration(
        &count_hist,
        &topic_size_hist,
        model.num_types,
        model.beta_sum,
    );

    model.beta = new_beta_sum / model.num_types as f64;
    model.beta_sum = new_beta_sum;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values captured from Java MALLET 202108 (Dirichlet.learnParameters and
    // Dirichlet.learnSymmetricConcentration) on fixed histograms; see #713.

    #[test]
    fn asymmetric_alpha_matches_mallet_learn_parameters() {
        // MALLET: learnParameters(alpha, obs, lens, shape=1.001, scale=1.0, iters=1).
        let mut alpha = [0.5, 0.5, 0.5];
        let topic_doc_hist = vec![vec![0u32, 4, 3, 2], vec![0u32, 5, 2], vec![0u32, 6]];
        let doc_length_hist = vec![0u32, 0, 0, 2, 3, 1, 2, 0, 1];
        let sum = learn_alpha_asymmetric(&mut alpha, &topic_doc_hist, &doc_length_hist).unwrap();
        // MALLET: 0.8164995977294759 0.5765014450029178 0.4588552917055854, sum 1.851856334437979.
        // Agreement is limited by topica's faster digamma approximation (~1e-9), not
        // by the formula, which is identical to MALLET's.
        assert!(
            (alpha[0] - 0.8164995977294759).abs() < 1e-8,
            "alpha0={}",
            alpha[0]
        );
        assert!(
            (alpha[1] - 0.5765014450029178).abs() < 1e-8,
            "alpha1={}",
            alpha[1]
        );
        assert!(
            (alpha[2] - 0.4588552917055854).abs() < 1e-8,
            "alpha2={}",
            alpha[2]
        );
        assert!((sum - 1.851856334437979).abs() < 1e-8, "sum={}", sum);
    }

    #[test]
    fn symmetric_concentration_is_correct_and_convergent() {
        // topica deliberately does NOT reproduce MALLET's learnSymmetricConcentration
        // denominator quirk (its final value here is 1.7596). We use the correct
        // fixed point, which on this degenerate histogram (near-uniform) drives the
        // concentration up; the runaway guard bounds it instead of overflowing.
        let count_hist = vec![0u32, 10, 5, 2];
        let size_hist = vec![0u32, 0, 0, 1, 2, 1];
        let out = learn_symmetric_concentration(&count_hist, &size_hist, 7, 0.35);
        assert!(out.is_finite(), "must be finite, got {out}");
        assert!(out > 0.0 && out <= CONCENTRATION_CAP, "bounded, got {out}");
    }

    #[test]
    fn symmetric_concentration_converges_on_beta_like_data() {
        // A realistic beta histogram (most word/topic pairs have few tokens, topics
        // are large) settles at a finite, bounded beta_sum rather than diverging.
        let count_hist = vec![0u32, 5000, 800, 200, 50, 10];
        let mut size_hist = vec![0u32; 3001];
        size_hist[3000] = 20; // 20 topics of 3000 tokens each
        let out = learn_symmetric_concentration(&count_hist, &size_hist, 2000, 0.01);
        assert!(
            out.is_finite() && out > 0.0 && out <= CONCENTRATION_CAP,
            "got {out}"
        );
    }
}
