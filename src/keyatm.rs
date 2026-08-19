//! keyATM: Keyword-Assisted Topic Model (base variant).
//!
//! Eshima, Imai & Sasaki (2024), "Keyword-Assisted Topic Models",
//! American Journal of Political Science 68(2):730–750.
//!
//! Like LDA, but the first `num_keyword_topics` topics are "keyword topics"
//! each with a researcher-supplied keyword list. A token in a keyword topic
//! is drawn either from that topic's keyword-only distribution (switch s=1,
//! prob π_k) or from the regular full-vocabulary distribution (switch s=0,
//! prob 1−π_k).  Remaining topics are regular LDA topics (no keywords, s=0
//! forced).
//!
//! ## Generative model (collapsed; integrate out θ, φ, φ̃, π)
//!
//! ```text
//! θ_d  ~ Dir(α·1_K)
//! φ_k  ~ Dir(β·1_V)                      [regular word dist, all topics]
//! φ̃_k ~ Dir(β_key·1_{L_k})               [keyword dist, keyword topics only]
//! π_k  ~ Beta(γ₁, γ₂)                    [keyword switch, keyword topics only]
//!
//! z_{d,i} ~ Categorical(θ_d)
//! s_{d,i} ~ Bernoulli(π_{z_{d,i}})       [only if z is a keyword topic AND
//!                                          w is a keyword of z; else s=0 forced]
//! w_{d,i} | z, s=0 ~ φ_z                 [regular draw]
//! w_{d,i} | z, s=1 ~ φ̃_z                 [keyword draw]
//! ```
//!
//! ## Collapsed Gibbs — joint (z, s) sample per token
//!
//! Counts:
//! - `ndk[d][k]` — doc-topic counts (D×K).
//! - `nkw[k][w]` — topic-word counts for s=0 tokens (K×V).
//! - `nk0[k]`    — total s=0 tokens in topic k.
//! - `nkx[k][j]` — keyword-only counts for s=1 tokens; j = position of word
//!                  in keywords[k] (K×L_k; zero-sized for non-keyword topics).
//! - `nk1[k]`    — total s=1 tokens in topic k.
//!
//! Unnormalised sampling weights (after removing token's current assignment):
//!
//! For every topic k, regular state s=0 (always allowed):
//! ```text
//! P(z=k,s=0) = (α + ndk[d][k])
//!            × (β + nkw[k][w]) / (V·β + nk0[k])
//!            × (γ₂ + nk0[k]) / (γ₁+γ₂ + nk0[k]+nk1[k])   if k is keyword topic
//!            × 1.0                                           if k is regular topic
//! ```
//!
//! Additionally, ONLY if k is a keyword topic AND w ∈ keywords[k], state s=1:
//! ```text
//! P(z=k,s=1) = (α + ndk[d][k])
//!            × (β_key + nkx[k][j]) / (L_k·β_key + nk1[k])
//!            × (γ₁ + nk1[k]) / (γ₁+γ₂ + nk0[k]+nk1[k])
//! ```

use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use crate::mathfun::log_gamma as lgamma;
use rand::Rng;

// ---------------------------------------------------------------------------
// Token weighting (keyATM's "weighted LDA")
// ---------------------------------------------------------------------------

/// How each token is weighted in the count tables. keyATM weights tokens by a
/// function of corpus frequency so that frequent words count for less and rare,
/// informative words count for more (Eshima, Imai & Sasaki 2024, Section 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightScheme {
    /// Unweighted: every token contributes 1.0 (ordinary collapsed Gibbs).
    None,
    /// Information theory (keyATM default): `weight(w) = -log2(freq(w)/N)`, the
    /// word's surprisal in bits.
    InfoTheory,
    /// Inverse frequency: `weight(w) = N / freq(w)`.
    InvFreq,
}

/// Per-word weights from corpus frequencies, matching keyATM's `weights_*`.
/// `num_types` is the vocabulary size; words absent from the corpus get weight
/// 1.0 (they never appear in a token, so the value is immaterial).
pub fn compute_vocab_weights(
    docs: &[Vec<u32>],
    num_types: usize,
    scheme: WeightScheme,
) -> Vec<f64> {
    if scheme == WeightScheme::None {
        return vec![1.0; num_types];
    }
    let mut freq = vec![0.0f64; num_types];
    for doc in docs {
        for &w in doc {
            freq[w as usize] += 1.0;
        }
    }
    let total: f64 = freq.iter().sum();
    let ln2 = std::f64::consts::LN_2;
    (0..num_types)
        .map(|w| {
            let f = freq[w];
            if f <= 0.0 {
                return 1.0;
            }
            match scheme {
                WeightScheme::None => 1.0,
                WeightScheme::InfoTheory => -((f / total).ln()) / ln2,
                WeightScheme::InvFreq => total / f,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Fitted keyATM Base model.
///
/// Stores the final Gibbs state; query it with the provided methods.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct KeyAtmModel {
    /// Vocabulary size V.
    pub num_types: usize,
    /// Total number of topics K (keyword + regular).
    pub num_topics: usize,
    /// Number of keyword topics (topics 0..num_keyword_topics).
    pub num_keyword_topics: usize,
    /// Dirichlet prior on doc-topic distributions (α).
    pub alpha: f64,
    /// Dirichlet prior on regular topic-word distributions (β).
    pub beta: f64,
    /// Dirichlet prior on keyword-only distributions (β_key).
    pub beta_key: f64,
    /// Beta(γ₁, γ₂) prior on keyword switch; γ₁ controls how much weight goes
    /// to the keyword distribution, γ₂ to the regular one.
    pub gamma1: f64,
    /// See `gamma1`.
    pub gamma2: f64,
    /// Keyword sets per topic; entries are word-id vecs, empty for non-keyword topics.
    pub keywords: Vec<Vec<usize>>,
    /// `L[k]` = `keywords[k].len()` (cached).
    pub l: Vec<usize>,
    /// `ndk[d][k]` — weighted doc-topic counts (D×K). Each token contributes its
    /// word's weight (see `vocab_weights`), so these are real-valued, not integer.
    pub ndk: Vec<Vec<f64>>,
    /// `nkw[k][w]` — weighted regular (s=0) topic-word counts (K×V).
    pub nkw: Vec<Vec<f64>>,
    /// `nk0[k]` — weighted total s=0 mass in topic k.
    pub nk0: Vec<f64>,
    /// `nkx[k][j]` — weighted keyword (s=1) count for the j-th keyword of topic k.
    /// Length of inner vec = L_k; empty for non-keyword topics.
    pub nkx: Vec<Vec<f64>>,
    /// `nk1[k]` — weighted total s=1 mass in topic k.
    pub nk1: Vec<f64>,
    /// Per-word weight applied to each token (keyATM's "weighted LDA"). All 1.0
    /// under `WeightScheme::None`; otherwise frequency-based (see `WeightScheme`).
    pub vocab_weights: Vec<f64>,
    /// Covariate model only: learned DMR coefficients `λ[k][f]` for the
    /// log-linear document-topic prior `α_{d,k} = exp(x_d · λ_k)`. `None` for the
    /// base (symmetric-α) model.
    pub lambda: Option<Vec<Vec<f64>>>,
    /// Covariate model only: the document feature matrix used at fit time
    /// (`[D][F]`), kept for held-out inference. `None` for the base model.
    pub features: Option<Vec<Vec<f64>>>,
    /// Covariate model only: a fixed per-document, per-topic embedding offset
    /// `s_{d,k}` added inside the DMR exponent, `α_{d,k} = exp(x_d · λ_k +
    /// s_{d,k})`. Anchors each document toward the topics its embedding is near
    /// while `λ` still estimates covariate effects. `None` for no embedding
    /// anchor (the plain covariate prior).
    #[serde(default)]
    pub prior_offset: Option<Vec<Vec<f64>>>,
    /// Dynamic model only: the fitted Chib (1998) change-point HMM over time
    /// segments. `None` for the base and covariate models.
    pub dynamic: Option<DynamicState>,
    /// Base model only: the estimated asymmetric document-topic Dirichlet prior
    /// `α_k` (length K), slice-sampled each sweep as in R keyATM. `None` for the
    /// covariate model (which uses the DMR `λ`) and the dynamic model (per-state
    /// `α`); when `None` the base `doc_topic` falls back to the symmetric
    /// `alpha`.
    #[serde(default)]
    pub alpha_vec: Option<Vec<f64>>,
    /// Convergence trace: `(iteration, log-likelihood, perplexity)` sampled
    /// every `ll_interval` Gibbs sweeps — the three columns of keyATM's
    /// `model_fit` (`plot_modelfit`). Empty when tracing is disabled.
    #[serde(default)]
    pub log_likelihood_history: Vec<(usize, f64, f64)>,
    /// Trace of the estimated document-topic prior `α` (length K) at each
    /// recorded sweep — keyATM's `plot_alpha` / `values_iter$alpha_iter`. Base
    /// model only (the covariate model traces `λ`, the dynamic model per-state
    /// `α`); empty otherwise or when tracing is disabled.
    #[serde(default)]
    pub alpha_history: Vec<(usize, Vec<f64>)>,
    /// Trace of the per-topic keyword switch rate `π` (length K) at each
    /// recorded sweep — keyATM's `plot_pi` / `values_iter$pi_iter`. Empty for a
    /// keyword-free model or when tracing is disabled.
    #[serde(default)]
    pub pi_history: Vec<(usize, Vec<f64>)>,
    /// Whether the Gibbs run stopped early because the relative change in the
    /// recorded `model_fit` log-likelihood fell below `convergence_tol` (opt-in;
    /// `false` when the full `iters` sweeps ran, the default). Mirrors the LDA
    /// family's `converged` flag.
    #[serde(default)]
    pub converged: bool,
    /// Thinned MCMC θ snapshots (issue #31): the last `num_theta_draws` per-doc
    /// topic distributions taken every `thin` sweeps of the fit loop, stored as
    /// f32 to halve the (S × D × K) footprint. Real cross-sweep posterior draws
    /// that `composition_theta` prefers over the within-document Dirichlet
    /// approximation. A fit-time artifact: empty unless `keep_theta_draws` was
    /// on, and not persisted across save/load.
    #[serde(skip)]
    pub theta_draws: Vec<Vec<Vec<f32>>>,
    /// Covariate model only: standard errors of `lambda` on the **original**
    /// covariate scale (`[K][F]`, aligned to `lambda`), from the observed
    /// information of the penalized Dirichlet-multinomial in the standardized fit
    /// space mapped back by the standardization Jacobian (issue #316). Entries
    /// whose standardized `λ` sat at the ±[`LAMBDA_BOUND`] clamp are `NaN` (the
    /// unconstrained asymptotic SE is invalid at the bound). `None` for the base and
    /// dynamic models, and for a covariate fit where `λ` was never optimized to a
    /// stationary point (`opt_interval=0`, `burn_in>=iters`, `lbfgs_iters=0`, or
    /// non-convergence). A fit-time artifact, not persisted across save/load.
    #[serde(default)]
    pub lambda_se: Option<Vec<Vec<f64>>>,
}

/// Fitted state of the keyATM **Dynamic** model (Eshima, Imai & Sasaki 2024,
/// Section 3.3), a Chib (1998) change-point hidden Markov model on topic
/// prevalence over time.
///
/// Documents are grouped into `num_time` ordered segments (one per timestamp).
/// Each segment is assigned to one of `num_states` latent states via a
/// left-to-right HMM (a segment stays in its state or advances to the next, so
/// the state sequence is non-decreasing and every state is visited). Each state
/// `r` owns its own document-topic Dirichlet prior `alphas[r]`, so topic
/// prevalence shifts at the estimated change points.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DynamicState {
    /// Number of latent states S.
    pub num_states: usize,
    /// Number of time segments T (distinct timestamps).
    pub num_time: usize,
    /// `time_index[d]` — 0-based time segment of document d (length D).
    pub time_index: Vec<usize>,
    /// `alphas[r][k]` — per-state document-topic Dirichlet prior (S×K).
    pub alphas: Vec<Vec<f64>>,
    /// `r_est[t]` — latent state assigned to time segment t (length T).
    pub r_est: Vec<usize>,
    /// `p_est[r][r']` — left-to-right transition matrix (S×S).
    pub p_est: Vec<Vec<f64>>,
}

impl KeyAtmModel {
    /// Effective topic-word distribution for topic k, length V, sums to 1.
    ///
    /// For a keyword topic k, the distribution mixes the regular and keyword
    /// distributions by the learned switch rate:
    /// ```text
    /// π_k  = (γ₁ + nk1[k]) / (γ₁+γ₂ + nk0[k]+nk1[k])
    /// reg_k(w)  = (β + nkw[k][w]) / (V·β + nk0[k])
    /// key_k(w)  = (β_key + nkx[k][j]) / (L_k·β_key + nk1[k])   for w = keywords[k][j]
    ///           = 0                                               otherwise
    /// phi_k(w)  = (1−π_k)·reg_k(w) + π_k·key_k(w)
    /// ```
    /// For a regular topic: `phi_k(w) = reg_k(w)`.
    /// The result is renormalised to sum to 1.
    pub fn topic_word(&self, k: usize) -> Vec<f64> {
        let v = self.num_types;
        let beta = self.beta;

        let reg_denom = v as f64 * beta + self.nk0[k];

        let mut phi = (0..v)
            .map(|w| (beta + self.nkw[k][w]) / reg_denom)
            .collect::<Vec<f64>>();

        if k < self.num_keyword_topics && self.l[k] > 0 {
            let pi_num = self.gamma1 + self.nk1[k];
            let pi_den = self.gamma1 + self.gamma2 + self.nk0[k] + self.nk1[k];
            let pi_k = pi_num / pi_den;
            let one_minus_pi = 1.0 - pi_k;

            let key_denom = self.l[k] as f64 * self.beta_key + self.nk1[k];
            let beta_key = self.beta_key;

            // Scale the regular component.
            for p in &mut phi {
                *p *= one_minus_pi;
            }
            // Add the keyword component.
            for (j, &kw) in self.keywords[k].iter().enumerate() {
                phi[kw] += pi_k * (beta_key + self.nkx[k][j]) / key_denom;
            }
        }

        // Renormalise (handles floating-point drift).
        let sum: f64 = phi.iter().sum();
        for p in &mut phi {
            *p /= sum;
        }
        phi
    }

    /// Topic-word distributions for all K topics. Shape: K × V.
    pub fn topic_word_all(&self) -> Vec<Vec<f64>> {
        (0..self.num_topics).map(|k| self.topic_word(k)).collect()
    }

    /// Doc-topic distributions θ, shape D×K, rows sum to 1.
    ///
    /// Base model: `θ_{d,k} = (ndk[d][k] + α) / (N_d + K·α)`. Covariate model:
    /// the symmetric α is replaced by the per-document prior
    /// `α_{d,k} = exp(x_d · λ_k)`.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        // Dynamic model: each document's α is the prior of its time segment's
        // current HMM state, `α_{d} = alphas[r_est[time_index[d]]]`.
        if let Some(dyn_) = &self.dynamic {
            return self
                .ndk
                .iter()
                .enumerate()
                .map(|(d, row)| {
                    let a_row = &dyn_.alphas[dyn_.r_est[dyn_.time_index[d]]];
                    let a_sum: f64 = a_row.iter().sum();
                    let n_d: f64 = row.iter().sum::<f64>() + a_sum;
                    row.iter()
                        .zip(a_row.iter())
                        .map(|(&c, &a)| (c + a) / n_d)
                        .collect()
                })
                .collect();
        }

        // Covariate model: per-document, per-topic α from the regression
        // (plus the fixed embedding offset, when present).
        if let (Some(lambda), Some(features)) = (&self.lambda, &self.features) {
            // keyATM keeps the bare exp() prior (no α floor) and a zero-mean λ prior.
            let doc_alpha =
                crate::dmr::compute_doc_alpha(lambda, features, self.prior_offset.as_deref(), 0.0);
            return self
                .ndk
                .iter()
                .zip(doc_alpha.iter())
                .map(|(row, a_row)| {
                    let a_sum: f64 = a_row.iter().sum();
                    let n_d: f64 = row.iter().sum::<f64>() + a_sum;
                    row.iter()
                        .zip(a_row.iter())
                        .map(|(&c, &a)| (c + a) / n_d)
                        .collect()
                })
                .collect();
        }

        // Base model: estimated asymmetric α (R keyATM default) when present,
        // else the symmetric `alpha`.
        if let Some(av) = &self.alpha_vec {
            let a_sum: f64 = av.iter().sum();
            return self
                .ndk
                .iter()
                .map(|row| {
                    let n_d: f64 = row.iter().sum::<f64>() + a_sum;
                    row.iter().zip(av).map(|(&c, &a)| (c + a) / n_d).collect()
                })
                .collect();
        }

        let k = self.num_topics;
        let alpha = self.alpha;
        let k_alpha = k as f64 * alpha;

        self.ndk
            .iter()
            .map(|row| {
                let n_d: f64 = row.iter().sum::<f64>() + k_alpha;
                row.iter().map(|&c| (c + alpha) / n_d).collect()
            })
            .collect()
    }

    /// The per-document Dirichlet prior α used by `doc_topic` (D×K): the dynamic
    /// per-state prior, the covariate DMR prior `exp(x·λ)`, the estimated
    /// asymmetric base prior, or the symmetric fallback.
    fn doc_alpha_matrix(&self) -> Vec<Vec<f64>> {
        if let Some(dyn_) = &self.dynamic {
            return self
                .ndk
                .iter()
                .enumerate()
                .map(|(d, _)| dyn_.alphas[dyn_.r_est[dyn_.time_index[d]]].clone())
                .collect();
        }
        if let (Some(lambda), Some(features)) = (&self.lambda, &self.features) {
            return crate::dmr::compute_doc_alpha(
                lambda,
                features,
                self.prior_offset.as_deref(),
                0.0,
            );
        }
        let row = self
            .alpha_vec
            .clone()
            .unwrap_or_else(|| vec![self.alpha; self.num_topics]);
        vec![row; self.ndk.len()]
    }

    /// keyATM's collapsed marginal log-likelihood, the quantity R `keyATM`
    /// reports as `model_fit`'s "Log Likelihood". It integrates out θ, the
    /// topic-word distributions, and π, summing the Dirichlet-multinomial
    /// evidence of the (weighted) count tables plus the keyword-switch
    /// Beta-Bernoulli evidence. Computed over the weighted counts, so it is on
    /// the same scale as R's.
    pub fn model_loglik(&self) -> f64 {
        let k = self.num_topics;
        let v = self.num_types;
        let beta = self.beta;
        let beta_key = self.beta_key;
        let lg_beta = lgamma(beta);
        let lg_beta_key = lgamma(beta_key);
        let mut ll = 0.0f64;

        // Regular (s=0) topic-word evidence, over the full vocabulary, for every
        // topic.
        for kk in 0..k {
            for &c in &self.nkw[kk] {
                ll += lgamma(beta + c) - lg_beta;
            }
            ll += lgamma(beta * v as f64) - lgamma(beta * v as f64 + self.nk0[kk]);
        }

        // Keyword (s=1) topic-word evidence + the keyword-switch π evidence, for
        // keyword topics only.
        let lg_g = lgamma(self.gamma1 + self.gamma2) - lgamma(self.gamma1) - lgamma(self.gamma2);
        for kk in 0..self.num_keyword_topics {
            let lk = self.l[kk];
            if lk == 0 {
                continue;
            }
            for &c in &self.nkx[kk] {
                ll += lgamma(beta_key + c) - lg_beta_key;
            }
            ll += lgamma(beta_key * lk as f64) - lgamma(beta_key * lk as f64 + self.nk1[kk]);
            ll += lg_g + lgamma(self.nk1[kk] + self.gamma1) + lgamma(self.nk0[kk] + self.gamma2)
                - lgamma(self.nk0[kk] + self.nk1[kk] + self.gamma1 + self.gamma2);
        }

        // Document-topic evidence under the per-document Dirichlet prior.
        let doc_alpha = self.doc_alpha_matrix();
        for (d, ndk) in self.ndk.iter().enumerate() {
            let a = &doc_alpha[d];
            let a_sum: f64 = a.iter().sum();
            let n_d: f64 = ndk.iter().sum();
            ll += lgamma(a_sum) - lgamma(a_sum + n_d);
            for kk in 0..k {
                ll += lgamma(ndk[kk] + a[kk]) - lgamma(a[kk]);
            }
        }
        ll
    }

    /// Total weighted token count (Σ over documents of the weighted document
    /// length) — the denominator keyATM divides the log-likelihood by to report
    /// perplexity.
    pub fn total_weight(&self) -> f64 {
        self.ndk.iter().map(|r| r.iter().sum::<f64>()).sum()
    }

    /// keyATM `model_fit` perplexity: `exp(-loglik / total_weighted_tokens)`.
    pub fn perplexity(&self) -> f64 {
        let n = self.total_weight();
        if n <= 0.0 {
            f64::NAN
        } else {
            (-self.model_loglik() / n).exp()
        }
    }

    /// Learned per-topic keyword switch rate π_k (length K).
    ///
    /// π_k = (γ₁ + nk1[k]) / (γ₁+γ₂ + nk0[k]+nk1[k]) for keyword topics.
    /// 0.0 for regular topics.
    pub fn keyword_rate(&self) -> Vec<f64> {
        (0..self.num_topics)
            .map(|k| {
                if k < self.num_keyword_topics && self.l[k] > 0 {
                    let num = self.gamma1 + self.nk1[k];
                    let den = self.gamma1 + self.gamma2 + self.nk0[k] + self.nk1[k];
                    num / den
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Dynamic model: smoothed topic prevalence per time segment, shape T×K,
    /// rows sum to 1. For time segment t in state `r = r_est[t]`, the prevalence
    /// is the normalised state prior `alphas[r] / Σ_k alphas[r][k]` — the
    /// posterior mean topic proportion the HMM assigns to that period.
    ///
    /// Returns `None` for non-dynamic models.
    pub fn time_prevalence(&self) -> Option<Vec<Vec<f64>>> {
        let dyn_ = self.dynamic.as_ref()?;
        Some(
            dyn_.r_est
                .iter()
                .map(|&r| {
                    let a_row = &dyn_.alphas[r];
                    let s: f64 = a_row.iter().sum();
                    a_row.iter().map(|&a| a / s).collect()
                })
                .collect(),
        )
    }
}

impl Estimator for KeyAtmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word_all()
    }
    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic()
    }
    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.log_likelihood_history
            .iter()
            .map(|&(i, ll, _)| (i, ll))
            .collect()
    }
    fn converged(&self) -> Option<bool> {
        Some(self.converged)
    }
    fn model_family(&self) -> ModelFamily {
        ModelFamily::Dirichlet
    }
}

impl DirichletModel for KeyAtmModel {
    fn alpha(&self) -> Vec<f64> {
        self.alpha_vec
            .clone()
            .unwrap_or_else(|| vec![self.alpha; self.num_topics])
    }
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        self.theta_draws
            .iter()
            .map(|d| {
                d.iter()
                    .map(|r| r.iter().map(|&x| x as f64).collect())
                    .collect()
            })
            .collect()
    }
    fn doc_lengths(&self) -> Vec<usize> {
        self.ndk
            .iter()
            .map(|r| r.iter().sum::<f64>() as usize)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sampler internals
// ---------------------------------------------------------------------------

/// Weighted categorical sample; `weights` need not be normalised.
/// Mirrors the `sample_index` in `gsdmm.rs`.
fn sample_index<R: Rng>(weights: &[f64], rng: &mut R) -> usize {
    let total: f64 = weights.iter().sum();
    // A non-finite or non-positive total means a degenerate weight vector — e.g. a
    // NaN from an upstream numeric breakdown. Without this guard `r` is NaN, `r <=
    // 0.0` is never true, and the loop silently falls through to the last index —
    // a biased, wrong draw dressed up as a sample. Catch it loudly in debug/tests;
    // in release fall back to a uniform pick rather than always the last state.
    debug_assert!(
        total.is_finite() && total > 0.0,
        "sample_index: non-finite or non-positive weight total ({total})"
    );
    if !(total.is_finite() && total > 0.0) {
        return rng.gen_range(0..weights.len());
    }
    let mut r = rng.gen::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

// ---------------------------------------------------------------------------
// Numeric helpers for the dynamic (HMM) sampler
// ---------------------------------------------------------------------------

/// Log density of Gamma(shape `a`, scale `b`), matching keyATM's `gammapdfln`:
/// `-a·ln(b) - lnΓ(a) + (a-1)·ln(x) - x/b`.
fn gammapdfln(x: f64, a: f64, b: f64) -> f64 {
    -a * b.ln() - lgamma(a) + (a - 1.0) * x.ln() - x / b
}

/// keyATM `shrinkp`: maps a positive α to (0, 1) via `x / (1 + x)`.
#[inline]
fn shrinkp(x: f64) -> f64 {
    x / (1.0 + x)
}

/// Standard-normal variate (Box–Muller).
fn sample_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape, 1) variate (Marsaglia & Tsang; boosted for shape < 1).
fn sample_gamma<R: Rng>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        let g = sample_gamma(shape + 1.0, rng);
        let u: f64 = rng.gen::<f64>().max(1e-300);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.gen::<f64>();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Beta(a, b) variate via two Gamma draws.
fn sample_beta<R: Rng>(a: f64, b: f64, rng: &mut R) -> f64 {
    let x = sample_gamma(a, rng);
    let y = sample_gamma(b, rng);
    if x + y <= 0.0 {
        0.5
    } else {
        x / (x + y)
    }
}

/// Inverted keyword index: for each word, the keyword topics that claim it.
///
/// The hot path needs, per token, the s=1 candidates (the keyword topics for
/// which the token's word is a keyword). Almost every word is a keyword of zero
/// topics, so `by_word[w]` is usually empty and the inner loop does no work,
/// instead of probing a per-topic hash map K times per token.
struct KeywordIndex {
    /// `by_word[w]` = `(topic, keyword_position)` for every keyword topic that
    /// lists word `w`. Empty for non-keyword words.
    by_word: Vec<Vec<(usize, usize)>>,
}

impl KeywordIndex {
    fn build(keywords: &[Vec<usize>], num_types: usize) -> Self {
        let mut by_word = vec![Vec::new(); num_types];
        for (k, kws) in keywords.iter().enumerate() {
            for (j, &w) in kws.iter().enumerate() {
                by_word[w].push((k, j));
            }
        }
        KeywordIndex { by_word }
    }

    /// Returns `Some(j)` if word `w` is the j-th keyword of topic `k`. Used only
    /// on the remove/re-add steps (once per token), so the short linear scan is
    /// cheaper than the hash map it replaces.
    #[inline]
    fn keyword_index(&self, k: usize, w: usize) -> Option<usize> {
        self.by_word[w]
            .iter()
            .find(|&&(t, _)| t == k)
            .map(|&(_, j)| j)
    }
}

/// Read-only scalar parameters the token sampler needs, bundled so the
/// sequential and parallel paths share one signature.
#[derive(Clone, Copy)]
struct SampleParams {
    num_topics: usize,
    num_keyword_topics: usize,
    num_types: usize,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
}

/// Resample the (z, s) assignment of a single token operating directly on the
/// count tables (not the whole model), so it can run against either the model's
/// own tables (sequential) or a worker's local clones (parallel). Removes the
/// token by its weight, samples a new (k, s), and re-adds it.
///
/// `nkw_t_row` is the word-major shadow of the regular topic-word table for the
/// current word: `nkw_t_row[kk] == nkw[kk][w]`. The s=0 candidate loop reads the
/// per-topic count of `w` across all topics; in the `nkw[k][w]` layout that is a
/// strided column walk (one cache miss per topic), whereas `nkw_t_row` is the
/// contiguous K-length row for `w`, streamed in order. The shadow is kept exactly
/// in step with `nkw` by applying the identical `+= wt` / `-= wt` to both cells,
/// so every value read is bit-for-bit the same f64 the `nkw` layout would
/// supply: the optimisation changes memory layout only, never the arithmetic.
#[allow(clippy::too_many_arguments)]
#[inline]
fn resample_token_inner<R: Rng>(
    nkw: &mut [Vec<f64>],
    nkw_t_row: &mut [f64],
    nk0: &mut [f64],
    nkx: &mut [Vec<f64>],
    nk1: &mut [f64],
    ndk_row: &mut [f64],
    l: &[usize],
    ki: &KeywordIndex,
    p: SampleParams,
    alpha_row: &[f64],
    w: usize,
    wt: f64,
    old_z: usize,
    old_s: u8,
    scratch: &mut [f64],
    rng: &mut R,
) -> (usize, u8) {
    let k = p.num_topics;
    let num_kw = p.num_keyword_topics;
    let beta = p.beta;
    let beta_key = p.beta_key;
    let gamma1 = p.gamma1;
    let gamma2 = p.gamma2;

    // --- Remove token from current counts (by its weight). ---
    ndk_row[old_z] -= wt;
    if old_s == 0 {
        nkw[old_z][w] -= wt;
        nkw_t_row[old_z] -= wt;
        nk0[old_z] -= wt;
    } else {
        let j = ki
            .keyword_index(old_z, w)
            .expect("old s=1 but w not a keyword");
        nkx[old_z][j] -= wt;
        nk1[old_z] -= wt;
    }

    // --- Build candidate weights into `scratch` (len 2*K). ---
    let v_beta = p.num_types as f64 * beta;
    let weights = &mut scratch[..2 * k];
    weights.fill(0.0);

    // s=0 candidate for every topic (always allowed). No keyword lookup here.
    // `nkw_t_row[kk]` is the contiguous word-major count, bit-identical to
    // `nkw[kk][w]` but read in cache order.
    for kk in 0..k {
        let ndk_val = ndk_row[kk] + alpha_row[kk];
        let nkw_val = nkw_t_row[kk] + beta;
        let nk0_val = nk0[kk];
        let reg_likelihood = nkw_val / (v_beta + nk0_val);
        let switch_factor_s0 = if kk < num_kw {
            let nk1_val = nk1[kk];
            (gamma2 + nk0_val) / (gamma1 + gamma2 + nk0_val + nk1_val)
        } else {
            1.0
        };
        weights[2 * kk] = ndk_val * reg_likelihood * switch_factor_s0;
    }

    // s=1 candidates only for the keyword topics that actually list `w`
    // (usually none, so this loop is empty for most tokens).
    for &(kk, j) in &ki.by_word[w] {
        let ndk_val = ndk_row[kk] + alpha_row[kk];
        let nk0_val = nk0[kk];
        let nk1_val = nk1[kk];
        let lk = l[kk] as f64;
        let key_likelihood = (nkx[kk][j] + beta_key) / (lk * beta_key + nk1_val);
        let switch_factor_s1 = (gamma1 + nk1_val) / (gamma1 + gamma2 + nk0_val + nk1_val);
        weights[2 * kk + 1] = ndk_val * key_likelihood * switch_factor_s1;
    }

    // --- Categorical sample ---
    let chosen = sample_index(weights, rng);
    let new_k = chosen / 2;
    let new_s = (chosen % 2) as u8;

    // --- Re-increment counts (by weight). ---
    ndk_row[new_k] += wt;
    if new_s == 0 {
        nkw[new_k][w] += wt;
        nkw_t_row[new_k] += wt;
        nk0[new_k] += wt;
    } else {
        let j = ki
            .keyword_index(new_k, w)
            .expect("new s=1 but w not a keyword");
        nkx[new_k][j] += wt;
        nk1[new_k] += wt;
    }

    (new_k, new_s)
}

/// Word-major shadow of `nkw`: `nkw_t[w][k] == nkw[k][w]`. Built once at the
/// start of a fit and maintained by the token sampler so the hot s=0 loop reads
/// each word's per-topic counts contiguously instead of striding down a `nkw`
/// column. A fit-time artifact, never persisted.
fn build_nkw_t(nkw: &[Vec<f64>], num_types: usize, num_topics: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0f64; num_topics]; num_types];
    for (k, row) in nkw.iter().enumerate() {
        for (w, &c) in row.iter().enumerate() {
            t[w][k] = c;
        }
    }
    t
}

/// Scalar params from the model.
fn sample_params(model: &KeyAtmModel) -> SampleParams {
    SampleParams {
        num_topics: model.num_topics,
        num_keyword_topics: model.num_keyword_topics,
        num_types: model.num_types,
        beta: model.beta,
        beta_key: model.beta_key,
        gamma1: model.gamma1,
        gamma2: model.gamma2,
    }
}

/// One full sequential Gibbs sweep over all tokens in all documents. `nkw_t` is
/// the word-major shadow of `model.nkw` (`nkw_t[w][k] == model.nkw[k][w]`), kept
/// in step so the hot s=0 loop reads each word's counts contiguously.
fn sweep<R: Rng>(
    model: &mut KeyAtmModel,
    docs: &[Vec<u32>],
    assignments: &mut [Vec<(usize, u8)>],
    nkw_t: &mut [Vec<f64>],
    ki: &KeywordIndex,
    doc_alpha: &[Vec<f64>],
    rng: &mut R,
) {
    let p = sample_params(model);
    let mut scratch = vec![0.0f64; 2 * p.num_topics];
    // Disjoint &mut borrows of distinct model fields in a single call are allowed.
    for d in 0..docs.len() {
        let alpha_row = &doc_alpha[d];
        for pos in 0..docs[d].len() {
            let w = docs[d][pos] as usize;
            let wt = model.vocab_weights[w];
            let (old_z, old_s) = assignments[d][pos];
            let (new_z, new_s) = resample_token_inner(
                &mut model.nkw,
                &mut nkw_t[w],
                &mut model.nk0,
                &mut model.nkx,
                &mut model.nk1,
                &mut model.ndk[d],
                &model.l,
                ki,
                p,
                alpha_row,
                w,
                wt,
                old_z,
                old_s,
                &mut scratch,
                rng,
            );
            assignments[d][pos] = (new_z, new_s);
        }
    }
}

/// Contiguous, near-even document ranges for `n_threads` workers.
fn partition_ranges(n_docs: usize, n_threads: usize) -> Vec<(usize, usize)> {
    let t = n_threads.max(1).min(n_docs.max(1));
    let base = n_docs / t;
    let rem = n_docs % t;
    let mut ranges = Vec::with_capacity(t);
    let mut start = 0;
    for i in 0..t {
        let len = base + usize::from(i < rem);
        ranges.push((start, start + len));
        start += len;
    }
    ranges
}

/// Approximate parallel Gibbs sweep (AD-LDA; Newman et al. 2009). Documents are
/// partitioned across workers; each worker clones the global topic-word tables
/// (`nkw`, `nk0`, `nkx`, `nk1`), samples its partition independently, then the
/// per-word deltas are reconciled as `final = Σ_w worker_w − (W−1)·original`.
/// Per-document tables (`ndk`, assignments) are partition-local, so they are
/// written straight back. Deterministic for a fixed `sweep_seed` and worker count.
fn parallel_sweep_keyatm(
    model: &mut KeyAtmModel,
    docs: &[Vec<u32>],
    assignments: &mut [Vec<(usize, u8)>],
    nkw_t: &mut Vec<Vec<f64>>,
    ki: &KeywordIndex,
    doc_alpha: &[Vec<f64>],
    num_threads: usize,
    sweep_seed: u64,
) {
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;
    use rayon::prelude::*;

    let p = sample_params(model);
    let ranges = partition_ranges(docs.len(), num_threads);

    // Snapshots of the global tables for reconciliation.
    let orig_nkw = model.nkw.clone();
    let orig_nk0 = model.nk0.clone();
    let orig_nkx = model.nkx.clone();
    let orig_nk1 = model.nk1.clone();
    let l = &model.l;
    let vocab_weights = &model.vocab_weights;

    struct WorkerOut {
        nkw: Vec<Vec<f64>>,
        nk0: Vec<f64>,
        nkx: Vec<Vec<f64>>,
        nk1: Vec<f64>,
        start: usize,
        ndk: Vec<Vec<f64>>,
        asgn: Vec<Vec<(usize, u8)>>,
        // Distinct word ids in this worker's document partition — the only `nkw`
        // columns it can have modified (`resample_token_inner` writes nkw[*][w]
        // for the token's own word w only). The merge folds just these columns.
        touched: Vec<u32>,
    }

    let outs: Vec<WorkerOut> = ranges
        .par_iter()
        .enumerate()
        .map(|(wid, &(start, end))| {
            let mut nkw = orig_nkw.clone();
            let mut nk0 = orig_nk0.clone();
            let mut nkx = orig_nkx.clone();
            let mut nk1 = orig_nk1.clone();
            let mut ndk: Vec<Vec<f64>> = model.ndk[start..end].to_vec();
            let mut asgn: Vec<Vec<(usize, u8)>> = assignments[start..end].to_vec();
            let mut scratch = vec![0.0f64; 2 * p.num_topics];
            // Worker-local word-major shadow of this worker's `nkw` clone.
            let mut local_t = build_nkw_t(&nkw, p.num_types, p.num_topics);
            let mut rng = Pcg64Mcg::seed_from_u64(
                sweep_seed ^ (wid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );

            for (li, d) in (start..end).enumerate() {
                let alpha_row = &doc_alpha[d];
                for pos in 0..docs[d].len() {
                    let w = docs[d][pos] as usize;
                    let wt = vocab_weights[w];
                    let (old_z, old_s) = asgn[li][pos];
                    let (new_z, new_s) = resample_token_inner(
                        &mut nkw,
                        &mut local_t[w],
                        &mut nk0,
                        &mut nkx,
                        &mut nk1,
                        &mut ndk[li],
                        l,
                        ki,
                        p,
                        alpha_row,
                        w,
                        wt,
                        old_z,
                        old_s,
                        &mut scratch,
                        &mut rng,
                    );
                    asgn[li][pos] = (new_z, new_s);
                }
            }
            // Distinct words in this partition: the columns this worker touched.
            let mut touched: Vec<u32> = docs[start..end].iter().flatten().copied().collect();
            touched.sort_unstable();
            touched.dedup();
            WorkerOut {
                nkw,
                nk0,
                nkx,
                nk1,
                start,
                ndk,
                asgn,
                touched,
            }
        })
        .collect();

    // Write back partition-local tables (disjoint document ranges).
    for out in &outs {
        for (li, (row, a)) in out.ndk.iter().zip(out.asgn.iter()).enumerate() {
            model.ndk[out.start + li] = row.clone();
            assignments[out.start + li] = a.clone();
        }
    }

    let wm1 = (outs.len() as f64) - 1.0;
    let k = p.num_topics;

    // Reconcile the topic-word table `nkw` (the K×V cost center) as a sparse
    // delta merge, parallel over topic rows (which are contiguous in the
    // topic-major layout). Each row starts from the original and adds, in fixed
    // worker order, each worker's net change on the columns that worker touched:
    //     nkw[k][w] = orig[k][w] + Σ_w (worker_w[k][w] − orig[k][w]).
    // Untouched columns stay exactly `orig` (the old dense `Σ_w − (W−1)·orig`
    // left ULP residue on every cell; this is algebraically equal and cleaner —
    // closer to the exact sequential path). Fixed worker order keeps the result
    // deterministic for a given (seed, thread count). Touched cells are clamped
    // to ≥0 after summing, matching the AD-LDA non-negativity guarantee.
    model.nkw = (0..k)
        .into_par_iter()
        .map(|kk| {
            let mut row = orig_nkw[kk].clone();
            for out in &outs {
                let wk = &out.nkw[kk];
                let ok = &orig_nkw[kk];
                for &w in &out.touched {
                    let w = w as usize;
                    row[w] += wk[w] - ok[w];
                }
            }
            for out in &outs {
                for &w in &out.touched {
                    let w = w as usize;
                    if row[w] < 0.0 {
                        row[w] = 0.0;
                    }
                }
            }
            row
        })
        .collect();

    // nk0, nk1, nkx are per-topic / per-keyword (length K and K×keywords), not
    // K×V, so the dense reconcile over them is cheap; keep it unchanged.
    for kk in 0..k {
        let sum0: f64 = outs.iter().map(|o| o.nk0[kk]).sum();
        model.nk0[kk] = (sum0 - wm1 * orig_nk0[kk]).max(0.0);
        let sum1: f64 = outs.iter().map(|o| o.nk1[kk]).sum();
        model.nk1[kk] = (sum1 - wm1 * orig_nk1[kk]).max(0.0);
        for j in 0..model.nkx[kk].len() {
            let sumx: f64 = outs.iter().map(|o| o.nkx[kk][j]).sum();
            model.nkx[kk][j] = (sumx - wm1 * orig_nkx[kk][j]).max(0.0);
        }
    }

    // Rebuild the word-major shadow from the reconciled `nkw` so the next sweep
    // reads a consistent table.
    *nkw_t = build_nkw_t(&model.nkw, p.num_types, k);
}

/// Run one sweep, choosing the exact sequential path or the approximate parallel
/// path by `num_threads`. The parallel path draws one seed from `rng` so it stays
/// deterministic for a fixed seed and worker count.
fn run_sweep<R: Rng>(
    model: &mut KeyAtmModel,
    docs: &[Vec<u32>],
    assignments: &mut [Vec<(usize, u8)>],
    nkw_t: &mut Vec<Vec<f64>>,
    ki: &KeywordIndex,
    doc_alpha: &[Vec<f64>],
    num_threads: usize,
    rng: &mut R,
) {
    if num_threads <= 1 {
        sweep(model, docs, assignments, nkw_t, ki, doc_alpha, rng);
    } else {
        let seed: u64 = rng.gen();
        parallel_sweep_keyatm(
            model,
            docs,
            assignments,
            nkw_t,
            ki,
            doc_alpha,
            num_threads,
            seed,
        );
    }
}

// ---------------------------------------------------------------------------
// Public fit function
// ---------------------------------------------------------------------------

/// Fit a keyATM Base model by collapsed Gibbs sampling.
///
/// # Arguments
/// * `docs`               — corpus; each doc is a list of word-ids in `0..num_types`.
/// * `num_types`          — vocabulary size V.
/// * `num_topics`         — total number of topics K.
/// * `keywords`           — keyword sets, one per topic; length must equal K.
///                          The first `keywords.len()` non-empty entries define
///                          keyword topics; the rest are regular topics.
///                          Pass empty slices for regular topics.
/// * `alpha`              — symmetric Dirichlet prior on doc-topic dists.
/// * `beta`               — symmetric Dirichlet prior on regular topic-word dists.
/// * `beta_key`           — symmetric Dirichlet prior on keyword-only dists.
/// * `gamma1`, `gamma2`   — Beta prior on the keyword switch π_k.
///                          Higher γ₁ → more weight on keyword distribution.
/// * `iters`              — number of full Gibbs sweeps.
/// * `rng`                — random-number source (deterministic for fixed seed).
pub fn fit_keyatm<R: Rng, F: FnMut(usize, usize, f64, f64)>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    alpha: f64,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    iters: usize,
    ll_interval: usize,
    estimate_alpha: bool,
    weights: WeightScheme,
    num_threads: usize,
    draws: ThetaDrawOpts,
    convergence_tol: f64,
    alpha_stride: usize,
    mut on_progress: F,
    rng: &mut R,
) -> KeyAtmModel {
    // A fixed symmetric α (estimate_alpha = false) must be a valid Dirichlet
    // parameter; when α is estimated the passed value is only a start and is not
    // constrained here (#418, item 3).
    if !estimate_alpha {
        assert!(
            alpha.is_finite() && alpha > 0.0,
            "alpha must be finite and positive when estimate_alpha is false (got {alpha})"
        );
    }
    let (mut model, mut assignments, ki) = init_state(
        docs, num_types, num_topics, keywords, alpha, beta, beta_key, gamma1, gamma2, weights, rng,
    );

    // R keyATM's base model estimates an asymmetric document-topic Dirichlet
    // prior α_k by slice-sampling it each sweep (Gamma priors: (1,1) on keyword
    // topics, (2,1) on regular ones — keyATM's defaults). The whole corpus is a
    // single "state", so we reuse the dynamic model's per-state α sampler over
    // all documents. `estimate_alpha = false` skips this (the dominant
    // non-sweep cost) and uses a fixed symmetric `alpha` — faster, at the price
    // of the R-matching asymmetric prior.
    let num_keyword_topics = model.num_keyword_topics;
    let doc_len: Vec<f64> = docs
        .iter()
        .map(|d| d.iter().map(|&w| model.vocab_weights[w as usize]).sum())
        .collect();
    let min_v = shrinkp(1e-9);
    let max_v = shrinkp(100.0);
    let mut alpha_vec = if estimate_alpha {
        vec![50.0 / num_topics as f64; num_topics] // keyATM's start; then learned
    } else {
        vec![alpha; num_topics] // fixed symmetric
    };
    let mut doc_alpha: Vec<Vec<f64>> = vec![alpha_vec.clone(); docs.len()];
    let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut nkw_t = build_nkw_t(&model.nkw, num_types, num_topics);

    for it in 0..iters {
        run_sweep(
            &mut model,
            docs,
            &mut assignments,
            &mut nkw_t,
            &ki,
            &doc_alpha,
            num_threads,
            rng,
        );
        if estimate_alpha {
            dyn_sample_alpha_state(
                &mut alpha_vec,
                num_keyword_topics,
                0,
                docs.len() - 1,
                &model.ndk,
                &doc_len,
                1.0,
                1.0,
                2.0,
                1.0,
                min_v,
                max_v,
                alpha_stride,
                rng,
            );
            for row in doc_alpha.iter_mut() {
                row.clone_from(&alpha_vec);
            }
        }
        draws.maybe_collect(&mut theta_draw_buf, it + 1, &model.ndk, &doc_alpha);
        if record_ll(it + 1, ll_interval, iters) {
            model.alpha_vec = Some(alpha_vec.clone());
            let ll = model.model_loglik();
            let ppl = model.perplexity();
            model.log_likelihood_history.push((it + 1, ll, ppl));
            on_progress(it + 1, iters, ll, ppl);
            // The alpha trace (plot_alpha) is only meaningful when alpha is being
            // learned; with a fixed symmetric prior it would be a flat line.
            if estimate_alpha {
                model.alpha_history.push((it + 1, alpha_vec.clone()));
            }
            if num_keyword_topics > 0 {
                model.pi_history.push((it + 1, model.keyword_rate()));
            }
            if ll_converged(&model.log_likelihood_history, convergence_tol) {
                model.converged = true;
                model.alpha_vec = Some(alpha_vec.clone());
                break;
            }
        }
    }
    model.alpha_vec = Some(alpha_vec);
    model.theta_draws = theta_draw_buf;
    model
}

/// Validate the public-API preconditions the Python wrapper already guarantees,
/// so a direct Rust caller gets a clear panic instead of silent count corruption
/// or an out-of-range index (#418, item 3). Shared by every public fit through
/// [`init_state`]; per-fit hyperparameters (`alpha`, the dynamic `eta`s) and the
/// dynamic `time_index` are validated at their own call sites.
#[allow(clippy::too_many_arguments)]
fn validate_keyatm_inputs(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
) {
    assert!(num_topics >= 1, "num_topics must be at least 1");
    // An empty corpus has no counts to sample and, with estimate_alpha, the base
    // fit passes `docs.len() - 1` (usize) to the α sampler, which would wrap to
    // usize::MAX. Reject it up front so every public fit rejects it identically.
    assert!(!docs.is_empty(), "docs must be non-empty");
    assert_eq!(
        keywords.len(),
        num_topics,
        "keywords length must equal num_topics"
    );
    for (name, val) in [
        ("beta", beta),
        ("beta_key", beta_key),
        ("gamma1", gamma1),
        ("gamma2", gamma2),
    ] {
        assert!(
            val.is_finite() && val > 0.0,
            "{name} must be finite and positive (got {val})"
        );
    }
    // Keyword topics must occupy a contiguous prefix `0..m`: the sampler derives
    // `num_keyword_topics` as the count of non-empty lists but then assumes those
    // topics are `0..num_keyword_topics`, so an empty list *before* a non-empty
    // one would route keyword-switch factors to the wrong topic. Also reject
    // duplicate and out-of-range keyword ids (a duplicate lets `KeywordIndex`
    // score one position but restore counts through another matching position).
    let mut seen_empty = false;
    for (k, kws) in keywords.iter().enumerate() {
        if kws.is_empty() {
            seen_empty = true;
            continue;
        }
        assert!(
            !seen_empty,
            "keyword topics must be the first topics: topic {k} has keywords but an \
             earlier topic's keyword list is empty"
        );
        let mut sorted = kws.clone();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            assert!(
                pair[0] != pair[1],
                "topic {k} lists keyword id {} more than once",
                pair[0]
            );
        }
        for &wid in kws {
            assert!(
                wid < num_types,
                "keyword id {wid} in topic {k} is out of range (num_types = {num_types})"
            );
        }
    }
    // Every document token id must be a valid vocabulary index.
    for (d, doc) in docs.iter().enumerate() {
        for &w in doc {
            assert!(
                (w as usize) < num_types,
                "document {d} has word id {w} out of range (num_types = {num_types})"
            );
        }
    }
}

/// Shared initialisation for the base and covariate fits: build the keyword
/// index, seed token assignments (keyword tokens anchored to their keyword topic
/// with the switch on), and the count tables. Returns the model with
/// `lambda`/`features` unset.
#[allow(clippy::too_many_arguments)]
fn init_state<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    alpha: f64,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    weights: WeightScheme,
    rng: &mut R,
) -> (KeyAtmModel, Vec<Vec<(usize, u8)>>, KeywordIndex) {
    validate_keyatm_inputs(
        docs, num_types, num_topics, keywords, beta, beta_key, gamma1, gamma2,
    );
    let d_count = docs.len();
    let v = num_types;
    let k = num_topics;
    let l: Vec<usize> = keywords.iter().map(|kws| kws.len()).collect();
    let num_keyword_topics = keywords.iter().filter(|kws| !kws.is_empty()).count();
    let ki = KeywordIndex::build(keywords, num_types);
    let vocab_weights = compute_vocab_weights(docs, v, weights);

    let mut word_keyword_topics: Vec<Vec<usize>> = vec![Vec::new(); v];
    for (kk, kws) in keywords.iter().enumerate() {
        for &w in kws {
            word_keyword_topics[w].push(kk);
        }
    }

    // A token whose word is a keyword of some topic starts in that keyword topic
    // (random among ties) with the switch on (s=1); other tokens start in a
    // uniformly random topic with s=0. Each token contributes its word weight.
    let mut ndk = vec![vec![0.0f64; k]; d_count];
    let mut nkw = vec![vec![0.0f64; v]; k];
    let mut nk0 = vec![0.0f64; k];
    let mut nkx: Vec<Vec<f64>> = keywords.iter().map(|kws| vec![0.0f64; kws.len()]).collect();
    let mut nk1 = vec![0.0f64; k];

    let assignments: Vec<Vec<(usize, u8)>> = docs
        .iter()
        .enumerate()
        .map(|(d, doc)| {
            doc.iter()
                .map(|&word| {
                    let w = word as usize;
                    let wt = vocab_weights[w];
                    let cands = &word_keyword_topics[w];
                    let (z, s): (usize, u8) = if cands.is_empty() {
                        ((rng.gen::<f64>() * k as f64) as usize % k, 0)
                    } else {
                        let z =
                            cands[(rng.gen::<f64>() * cands.len() as f64) as usize % cands.len()];
                        (z, 1)
                    };
                    ndk[d][z] += wt;
                    if s == 0 {
                        nkw[z][w] += wt;
                        nk0[z] += wt;
                    } else {
                        let j = ki.keyword_index(z, w).unwrap();
                        nkx[z][j] += wt;
                        nk1[z] += wt;
                    }
                    (z, s)
                })
                .collect()
        })
        .collect();

    let model = KeyAtmModel {
        num_types: v,
        num_topics: k,
        num_keyword_topics,
        alpha,
        beta,
        beta_key,
        gamma1,
        gamma2,
        keywords: keywords.to_vec(),
        l,
        ndk,
        nkw,
        nk0,
        nkx,
        nk1,
        vocab_weights,
        lambda: None,
        features: None,
        prior_offset: None,
        dynamic: None,
        alpha_vec: None,
        log_likelihood_history: Vec::new(),
        alpha_history: Vec::new(),
        pi_history: Vec::new(),
        converged: false,
        theta_draws: Vec::new(),
        lambda_se: None,
    };
    (model, assignments, ki)
}

/// CVB0 (collapsed variational Bayes, zeroth-order) inference for the **base**
/// keyATM model — a deterministic alternative to the Gibbs sampler.
///
/// Each (document, word-type) cell keeps a soft responsibility over the
/// (topic, switch) states the Gibbs sampler draws from: `s=0` for every topic,
/// plus `s=1` for each keyword topic that lists the word. The responsibilities
/// are updated from the expected counts using the very same conditional as
/// [`resample_token_inner`] (mass-weighted by the token weights), so it reuses
/// keyATM's count tables and [`KeyAtmModel`] output methods unchanged.
///
/// This is **not** the R-keyATM estimator: it is a deterministic, non-sampling
/// backend (optional `sampler="cvb0"`), so it does not preserve R-parity. It
/// uses a fixed symmetric `alpha` (no slice sampling) and does not cover the
/// covariate or dynamic variants.
#[allow(clippy::too_many_arguments)]
pub fn fit_keyatm_cvb0<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    alpha: f64,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    iters: usize,
    weights: WeightScheme,
    rng: &mut R,
) -> KeyAtmModel {
    assert!(
        alpha.is_finite() && alpha > 0.0,
        "alpha must be finite and positive (got {alpha})"
    );
    let (mut model, _assign, ki) = init_state(
        docs, num_types, num_topics, keywords, alpha, beta, beta_key, gamma1, gamma2, weights, rng,
    );
    let k = num_topics;
    let num_kw = model.num_keyword_topics;
    let v_beta = num_types as f64 * beta;

    // Cells: per doc, unique (word, weighted mass = Σ token weights of that word).
    let cells: Vec<Vec<(u32, f64)>> = docs
        .iter()
        .map(|doc| {
            let mut m: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
            for &w in doc {
                *m.entry(w).or_insert(0.0) += model.vocab_weights[w as usize];
            }
            let mut c: Vec<(u32, f64)> = m.into_iter().collect();
            c.sort_by_key(|&(w, _)| w);
            c
        })
        .collect();

    // γ per cell: g0[k] over s=0 (length K) and g1 over s=1 (parallel to
    // ki.by_word[w]). Random init, normalized, then build the expected counts.
    for row in model.ndk.iter_mut() {
        row.iter_mut().for_each(|x| *x = 0.0);
    }
    model
        .nkw
        .iter_mut()
        .for_each(|r| r.iter_mut().for_each(|x| *x = 0.0));
    model
        .nkx
        .iter_mut()
        .for_each(|r| r.iter_mut().for_each(|x| *x = 0.0));
    model.nk0.iter_mut().for_each(|x| *x = 0.0);
    model.nk1.iter_mut().for_each(|x| *x = 0.0);

    let mut g0: Vec<Vec<Vec<f64>>> = Vec::with_capacity(docs.len());
    let mut g1: Vec<Vec<Vec<f64>>> = Vec::with_capacity(docs.len());
    for (d, dcells) in cells.iter().enumerate() {
        let mut dg0 = Vec::with_capacity(dcells.len());
        let mut dg1 = Vec::with_capacity(dcells.len());
        for &(w, mass) in dcells {
            let kw = &ki.by_word[w as usize];
            let mut a0 = vec![0.0f64; k];
            let mut a1 = vec![0.0f64; kw.len()];
            let mut sum = 0.0;
            for x in a0.iter_mut() {
                *x = rng.gen::<f64>() + 1e-6;
                sum += *x;
            }
            for x in a1.iter_mut() {
                *x = rng.gen::<f64>() + 1e-6;
                sum += *x;
            }
            for (t, x) in a0.iter_mut().enumerate() {
                *x /= sum;
                model.ndk[d][t] += mass * *x;
                model.nkw[t][w as usize] += mass * *x;
                model.nk0[t] += mass * *x;
            }
            for (idx, &(t, j)) in kw.iter().enumerate() {
                a1[idx] /= sum;
                model.ndk[d][t] += mass * a1[idx];
                model.nkx[t][j] += mass * a1[idx];
                model.nk1[t] += mass * a1[idx];
            }
            dg0.push(a0);
            dg1.push(a1);
        }
        g0.push(dg0);
        g1.push(dg1);
    }

    // Sweeps: mirror resample_token_inner softly per cell.
    for _ in 0..iters {
        for (d, dcells) in cells.iter().enumerate() {
            for (ci, &(w, mass)) in dcells.iter().enumerate() {
                let w = w as usize;
                let kw = &ki.by_word[w];

                // Remove this cell's mass (the `-dw` expected counts).
                for t in 0..k {
                    let c = mass * g0[d][ci][t];
                    model.ndk[d][t] -= c;
                    model.nkw[t][w] -= c;
                    model.nk0[t] -= c;
                }
                for (idx, &(t, j)) in kw.iter().enumerate() {
                    let c = mass * g1[d][ci][idx];
                    model.ndk[d][t] -= c;
                    model.nkx[t][j] -= c;
                    model.nk1[t] -= c;
                }

                // Recompute the (topic, switch) responsibilities.
                let mut sum = 0.0f64;
                for t in 0..k {
                    let ndk_val = model.ndk[d][t] + alpha;
                    let reg = (model.nkw[t][w] + beta) / (v_beta + model.nk0[t]);
                    let sw0 = if t < num_kw {
                        (gamma2 + model.nk0[t]) / (gamma1 + gamma2 + model.nk0[t] + model.nk1[t])
                    } else {
                        1.0
                    };
                    let val = ndk_val * reg * sw0;
                    let val = if val > 0.0 { val } else { 0.0 };
                    g0[d][ci][t] = val;
                    sum += val;
                }
                for (idx, &(t, j)) in kw.iter().enumerate() {
                    let ndk_val = model.ndk[d][t] + alpha;
                    let lk = model.l[t] as f64;
                    let key = (model.nkx[t][j] + beta_key) / (lk * beta_key + model.nk1[t]);
                    let sw1 =
                        (gamma1 + model.nk1[t]) / (gamma1 + gamma2 + model.nk0[t] + model.nk1[t]);
                    let val = ndk_val * key * sw1;
                    let val = if val > 0.0 { val } else { 0.0 };
                    g1[d][ci][idx] = val;
                    sum += val;
                }

                // Normalize and add the new mass back.
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for t in 0..k {
                    let g = g0[d][ci][t] * inv;
                    g0[d][ci][t] = g;
                    let c = mass * g;
                    model.ndk[d][t] += c;
                    model.nkw[t][w] += c;
                    model.nk0[t] += c;
                }
                for (idx, &(t, j)) in kw.iter().enumerate() {
                    let g = g1[d][ci][idx] * inv;
                    g1[d][ci][idx] = g;
                    let c = mass * g;
                    model.ndk[d][t] += c;
                    model.nkx[t][j] += c;
                    model.nk1[t] += c;
                }
            }
        }
    }
    model
}

/// Whether to record the predictive log-likelihood at 1-based sweep `iter`.
/// Records every `interval` sweeps (and always the final one) so the trace
/// ends at the model's returned state; `interval == 0` disables tracing.
#[inline]
fn record_ll(iter: usize, interval: usize, iters: usize) -> bool {
    interval > 0 && (iter.is_multiple_of(interval) || iter == iters)
}

/// Opt-in early-stop test on keyATM's recorded `model_fit` trace: `true` once the
/// last two recorded log-likelihoods differ by less than `tol` in relative terms.
/// `tol <= 0` disables it (the default), so the full `iters` sweeps run. Mirrors
/// the relative-bound criterion the LDA family and R `stm` use; the trace cadence
/// (`ll_interval`) sets the comparison window.
#[inline]
fn ll_converged(history: &[(usize, f64, f64)], tol: f64) -> bool {
    if tol <= 0.0 || history.len() < 2 {
        return false;
    }
    let cur = history[history.len() - 1].1;
    let prev = history[history.len() - 2].1;
    (cur - prev).abs() / (prev.abs() + 1e-12) < tol
}

/// Thinned θ-draw retention schedule (issue #31): snapshot every `thin` sweeps
/// and keep the last `cap` on a ring buffer, so the kept draws are the converged
/// tail of the chain. `cap == 0` disables collection.
#[derive(Clone, Copy)]
pub struct ThetaDrawOpts {
    pub cap: usize,
    pub thin: usize,
}

impl ThetaDrawOpts {
    /// Derive the schedule from the fit knobs and run length. Takes ~`2·cap`
    /// snapshots over the run so the ring-buffered `cap` land in the back half.
    pub fn new(keep: bool, num_draws: usize, iters: usize) -> Self {
        let cap = if keep { num_draws } else { 0 };
        let thin = if cap == 0 {
            0
        } else {
            (iters / (2 * cap)).max(1)
        };
        ThetaDrawOpts { cap, thin }
    }

    /// At 1-based sweep `iter`, snapshot the smoothed θ = (n_dk + α_dk)/(N_d +
    /// Σα_d) from the current counts and per-document prior, pushing onto `buf`
    /// (capped at the last `cap`). No-op off-schedule or when disabled.
    fn maybe_collect(
        &self,
        buf: &mut Vec<Vec<Vec<f32>>>,
        iter: usize,
        ndk: &[Vec<f64>],
        doc_alpha: &[Vec<f64>],
    ) {
        if self.thin == 0 || !iter.is_multiple_of(self.thin) {
            return;
        }
        let snap: Vec<Vec<f32>> = ndk
            .iter()
            .zip(doc_alpha.iter())
            .map(|(row, a)| {
                let denom = row.iter().sum::<f64>() + a.iter().sum::<f64>();
                row.iter()
                    .zip(a.iter())
                    .map(|(&c, &av)| ((c + av) / denom) as f32)
                    .collect()
            })
            .collect();
        buf.push(snap);
        if buf.len() > self.cap {
            buf.remove(0);
        }
    }
}

/// Bound on the standardized regression coefficients `λ`, matching R keyATM's
/// `slice_max`/`slice_min` of ±5. Without it (and without standardized
/// covariates) a high-dimensional design lets one topic's `λ` run away over the
/// sweeps, so `α = exp(x · λ)` blows up and θ collapses onto a single topic
/// (issue #270).
const LAMBDA_BOUND: f64 = 5.0;

/// R keyATM's covariate standardization (issue #270): z-score each non-intercept
/// column to mean 0, sd 1. Column 0 (the prepended intercept) is left untouched.
/// Constant columns (sd ≈ 0) are centered to 0 and given sd 1 (a null direction
/// the N(0,1) prior regularizes). Returns the standardized matrix and the per-
/// column means/sds (index 0 = intercept: mean 0, sd 1) so `λ` can be mapped back
/// to the original covariate scale after fitting.
fn standardize_features(
    features: &[Vec<f64>],
    num_features: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = features.len();
    let mut mean = vec![0.0f64; num_features];
    let mut sd = vec![1.0f64; num_features];
    if n == 0 {
        return (features.to_vec(), mean, sd);
    }
    // Column 0 is the intercept (all 1.0); skip it.
    for f in 1..num_features {
        let m: f64 = features.iter().map(|r| r[f]).sum::<f64>() / n as f64;
        let var: f64 = features.iter().map(|r| (r[f] - m).powi(2)).sum::<f64>() / n as f64;
        let s = var.sqrt();
        mean[f] = m;
        sd[f] = if s > 1e-9 { s } else { 1.0 };
    }
    let std: Vec<Vec<f64>> = features
        .iter()
        .map(|r| {
            (0..num_features)
                .map(|f| {
                    if f == 0 {
                        r[0]
                    } else {
                        (r[f] - mean[f]) / sd[f]
                    }
                })
                .collect()
        })
        .collect();
    (std, mean, sd)
}

/// Standard errors of the covariate-keyATM `λ` on the **original** covariate
/// scale (issue #316). `λ` is fit in the standardized space (z-scored covariates,
/// N(0,1) prior, bounded to ±[`LAMBDA_BOUND`]); its asymptotic covariance is the
/// inverse observed information of the penalized Dirichlet-multinomial there
/// ([`crate::dmr::dmr_lambda_cov`]). The mapping back to the raw scale,
/// `λ_orig = M λ_std` with (per topic)
/// ```text
///   λ_orig[0] = λ_std[0] − Σ_{g≥1} (mean_g/sd_g) λ_std[g],   λ_orig[f] = λ_std[f]/sd_f,
/// ```
/// is a non-diagonal linear map (the intercept mixes the slopes), so the raw-scale
/// covariance of topic `k` is `Cov_orig = M Cov_std[k] Mᵀ` and the SE is
/// `sqrt(diag)`. The map is block-diagonal across topics, so only the within-topic
/// `F×F` blocks of the standardized covariance are needed. Entries whose
/// standardized `λ` sits at the ±[`LAMBDA_BOUND`] clamp are returned as `NaN`: the
/// coefficient is constrained, not at an interior optimum, so the unconstrained
/// asymptotic SE does not apply.
fn covariate_lambda_se(
    lambda_std: &[Vec<f64>],
    features_std: &[Vec<f64>],
    doc_topic_counts: &[Vec<f64>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    feat_mean: &[f64],
    feat_sd: &[f64],
    offset: Option<&[Vec<f64>]>,
) -> Vec<Vec<f64>> {
    let t = num_topics;
    let f = num_features;
    let p = t * f;
    let cov_std = crate::dmr::dmr_lambda_cov(
        lambda_std,
        features_std,
        doc_topic_counts,
        t,
        f,
        prior_variance,
        0.0,
        offset,
    );
    // Per-topic Jacobian M (F×F), identical across topics: row 0 maps the
    // intercept and subtracts each standardized slope's mean shift; row g≥1
    // rescales by 1/sd_g.
    let mut se = vec![vec![0.0f64; f]; t];
    // tol: treat |λ_std| within 1e-6 of the bound as clamped.
    let bound_tol = 1e-6;
    for k in 0..t {
        let base = k * f;
        // Extract the within-topic block C (F×F) from the flat covariance.
        let cblk = |i: usize, j: usize| cov_std[(base + i) * p + (base + j)];
        // Build M·C (F×F).
        let mut mc = vec![vec![0.0f64; f]; f];
        for j in 0..f {
            // Row 0 of M: [1, -mean_1/sd_1, ..., -mean_{F-1}/sd_{F-1}].
            let mut acc = cblk(0, j);
            for g in 1..f {
                acc += -(feat_mean[g] / feat_sd[g]) * cblk(g, j);
            }
            mc[0][j] = acc;
            // Rows f≥1 of M: scale row f of C by 1/sd_f.
            for ff in 1..f {
                mc[ff][j] = cblk(ff, j) / feat_sd[ff];
            }
        }
        // diag of (M·C)·Mᵀ, then sqrt; flag clamped entries as NaN.
        for ff in 0..f {
            let var = if ff == 0 {
                let mut acc = mc[0][0];
                for g in 1..f {
                    acc += mc[0][g] * (-(feat_mean[g] / feat_sd[g]));
                }
                acc
            } else {
                mc[ff][ff] / feat_sd[ff]
            };
            let clamped = lambda_std[k][ff].abs() >= LAMBDA_BOUND - bound_tol;
            // A clamped coefficient, or a non-positive/NaN variance, has no valid SE.
            se[k][ff] = if !clamped && var > 0.0 {
                var.sqrt()
            } else {
                f64::NAN
            };
        }
    }
    se
}

/// Fit a keyATM **Covariate** model. The document-topic prior is a
/// Dirichlet-Multinomial regression on document features,
/// `α_{d,k} = exp(x_d · λ_k)` (Mimno & McCallum 2008). It shares keyATM's
/// generative model, N(0,1) prior, and ±5 bound, but **estimates `λ` by L-BFGS
/// MAP (not keyATM's per-sweep slice/MH sampling)** — so `λ` is a MAP point
/// estimate and its SE (issue #316) is asymptotic observed-information, not a
/// posterior SD. Following R keyATM, the covariates are standardized and `λ` is
/// bounded to ±5 (see [`LAMBDA_BOUND`] and [`standardize_features`]) to keep `α`
/// from blowing up; `λ` is re-estimated every `opt_interval` sweeps once past
/// `burn_in` (so it stays at its zero init — a null covariate effect — if
/// `iters < burn_in + opt_interval`). The keyword (z, s) sampler is otherwise
/// identical to the base model.
#[allow(clippy::too_many_arguments)]
pub fn fit_keyatm_cov<R: Rng, F: FnMut(usize, usize, f64, f64)>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    features: &[Vec<f64>],
    num_features: usize,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    iters: usize,
    opt_interval: usize,
    burn_in: usize,
    prior_variance: f64,
    lbfgs_iters: usize,
    ll_interval: usize,
    weights: WeightScheme,
    num_threads: usize,
    offset: Option<&[Vec<f64>]>,
    draws: ThetaDrawOpts,
    convergence_tol: f64,
    mut on_progress: F,
    rng: &mut R,
) -> KeyAtmModel {
    assert_eq!(
        keywords.len(),
        num_topics,
        "keywords length must equal num_topics"
    );
    assert_eq!(
        features.len(),
        docs.len(),
        "features rows must equal number of documents"
    );
    if let Some(off) = offset {
        assert_eq!(
            off.len(),
            docs.len(),
            "offset rows must equal number of documents"
        );
        assert!(
            off.iter().all(|r| r.len() == num_topics),
            "offset columns must equal num_topics"
        );
    }

    // Standardize covariates (issue #270): we fit λ in the standardized space,
    // bounded to ±LAMBDA_BOUND, then map λ back to the original scale for storage.
    let (features_std, feat_mean, feat_sd) = standardize_features(features, num_features);

    // α is replaced by the covariate prior; pass a nominal 1.0 for the struct.
    let (mut model, mut assignments, ki) = init_state(
        docs, num_types, num_topics, keywords, 1.0, beta, beta_key, gamma1, gamma2, weights, rng,
    );

    let mut lambda = vec![vec![0.0f64; num_features]; num_topics];
    // keyATM keeps DMR's original zero-mean λ prior and bare exp() prior (no α
    // floor); the #563 baseline-α centering is DMR-specific.
    let km_prior_mean = vec![0.0f64; num_features];
    let mut doc_alpha = crate::dmr::compute_doc_alpha(&lambda, &features_std, offset, 0.0);
    let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
    // So the log-likelihood trace uses the covariate doc-topic prior: doc_topic()
    // takes the (lambda, features) path only when both are set on the model. While
    // fitting we keep the standardized features and standardized-space λ on the
    // model so the trace is self-consistent; both are converted to the original
    // scale at the end.
    if ll_interval > 0 {
        model.features = Some(features_std.clone());
        model.prior_offset = offset.map(|o| o.to_vec());
    }
    let mut nkw_t = build_nkw_t(&model.nkw, num_types, num_topics);

    // Counts from the last *converged* λ optimization; the SE is computed from
    // exactly these so it is the observed information at the optimum for the
    // returned λ, not at counts that drifted through the later sampling sweeps
    // (#418, the same cross-model issue fixed for DMR in #419).
    let mut se_counts: Option<Vec<Vec<f64>>> = None;

    for it in 0..iters {
        run_sweep(
            &mut model,
            docs,
            &mut assignments,
            &mut nkw_t,
            &ki,
            &doc_alpha,
            num_threads,
            rng,
        );
        if opt_interval > 0 && it + 1 > burn_in && (it + 1 - burn_in).is_multiple_of(opt_interval) {
            let converged = crate::dmr::optimize_lambda(
                &mut lambda,
                &features_std,
                &model.ndk,
                num_topics,
                num_features,
                prior_variance,
                &km_prior_mean,
                0.0,
                lbfgs_iters,
                offset,
                // keyATM keeps the plain first step (bit-identical to pre-#563): it
                // standardizes features and clamps λ, so it never had DMR's runaway.
                false,
            );
            // Snapshot the counts this optimization ran against (before the next
            // sweep mutates them); only a converged run yields a valid SE optimum.
            se_counts = if converged {
                Some(model.ndk.clone())
            } else {
                None
            };
            // Bound λ to ±LAMBDA_BOUND in the standardized space (R keyATM's
            // slice bounds), so a runaway coefficient cannot blow up α (#270).
            for lk in lambda.iter_mut() {
                for v in lk.iter_mut() {
                    *v = v.clamp(-LAMBDA_BOUND, LAMBDA_BOUND);
                }
            }
            doc_alpha = crate::dmr::compute_doc_alpha(&lambda, &features_std, offset, 0.0);
        }
        draws.maybe_collect(&mut theta_draw_buf, it + 1, &model.ndk, &doc_alpha);
        if record_ll(it + 1, ll_interval, iters) {
            // During the trace, λ and features are both in the standardized space.
            model.lambda = Some(lambda.clone());
            model.features = Some(features_std.clone());
            model.prior_offset = offset.map(|o| o.to_vec());
            let ll = model.model_loglik();
            let ppl = model.perplexity();
            model.log_likelihood_history.push((it + 1, ll, ppl));
            on_progress(it + 1, iters, ll, ppl);
            model.pi_history.push((it + 1, model.keyword_rate()));
            if ll_converged(&model.log_likelihood_history, convergence_tol) {
                model.converged = true;
                break;
            }
        }
    }

    // Standard errors of λ (issue #316): computed in the standardized fit space at
    // the counts of the last converged optimization (#418), then mapped to the
    // original scale by the same standardization Jacobian used for λ below. Done
    // before the λ mapback so the bound check sees the standardized coefficients
    // that were actually clamped. `None` when λ was never optimized to a stationary
    // point (opt_interval=0, burn_in>=iters, lbfgs_iters=0, or non-convergence),
    // since the observed information is only a valid covariance at an optimum.
    let lambda_se = se_counts.as_ref().map(|ndk| {
        covariate_lambda_se(
            &lambda,
            &features_std,
            ndk,
            num_topics,
            num_features,
            prior_variance,
            &feat_mean,
            &feat_sd,
            offset,
        )
    });

    // Map λ from the standardized space back to the original covariate scale, so
    // that exp(features · λ_orig) == exp(features_std · λ_std): for f ≥ 1,
    // λ_orig[f] = λ_std[f] / sd_f, with the intercept absorbing the mean shift.
    // `feature_effects` and held-out `transform` then operate on raw covariates.
    let lambda_orig: Vec<Vec<f64>> = lambda
        .iter()
        .map(|lk| {
            let mut row = lk.clone();
            let mut shift = 0.0;
            for f in 1..num_features {
                row[f] = lk[f] / feat_sd[f];
                shift += lk[f] * feat_mean[f] / feat_sd[f];
            }
            row[0] = lk[0] - shift;
            row
        })
        .collect();
    model.lambda = Some(lambda_orig);
    model.features = Some(features.to_vec());
    model.prior_offset = offset.map(|o| o.to_vec());
    model.lambda_se = lambda_se;
    model.theta_draws = theta_draw_buf;
    model
}

// ---------------------------------------------------------------------------
// Dynamic model (Chib 1998 change-point HMM on topic prevalence)
// ---------------------------------------------------------------------------

/// Dirichlet-multinomial marginal log-likelihood of topic `k`'s contribution
/// over documents `[doc_start, doc_end]` under state prior `alpha`, plus the
/// Gamma(shape, scale) prior on `alpha[k]`. Matches keyATM's `alpha_loglik`.
fn dyn_alpha_loglik(
    alpha: &[f64],
    k: usize,
    doc_start: usize,
    doc_end: usize,
    ndk: &[Vec<f64>],
    doc_len: &[f64],
    prior_shape: f64,
    prior_scale: f64,
) -> f64 {
    let alpha_sum: f64 = alpha.iter().sum();
    let fixed = lgamma(alpha_sum) - lgamma(alpha[k]);
    let mut loglik = gammapdfln(alpha[k], prior_shape, prior_scale);
    for d in doc_start..=doc_end {
        loglik += fixed + lgamma(ndk[d][k] + alpha[k]) - lgamma(doc_len[d] + alpha_sum);
    }
    loglik
}

/// Approximate `dyn_alpha_loglik` (turbo): evaluates the per-document Dirichlet-
/// multinomial term over every `stride`-th document and scales that sum up by the
/// inverse sampling fraction, so the slice sampler sees an unbiased estimate of
/// the full data term while touching only `~1/stride` of the documents. The
/// Gamma prior term is left unscaled (it does not depend on the corpus size).
/// `stride == 1` reduces to the exact `dyn_alpha_loglik`. Used only on the opt-in
/// `turbo_alpha_stride` path; it changes the estimated α and therefore the fit.
fn dyn_alpha_loglik_sub(
    alpha: &[f64],
    k: usize,
    doc_start: usize,
    doc_end: usize,
    ndk: &[Vec<f64>],
    doc_len: &[f64],
    prior_shape: f64,
    prior_scale: f64,
    stride: usize,
) -> f64 {
    if stride <= 1 {
        return dyn_alpha_loglik(
            alpha,
            k,
            doc_start,
            doc_end,
            ndk,
            doc_len,
            prior_shape,
            prior_scale,
        );
    }
    let alpha_sum: f64 = alpha.iter().sum();
    let fixed = lgamma(alpha_sum) - lgamma(alpha[k]);
    let mut data = 0.0f64;
    let mut sampled = 0usize;
    let mut d = doc_start;
    while d <= doc_end {
        data += fixed + lgamma(ndk[d][k] + alpha[k]) - lgamma(doc_len[d] + alpha_sum);
        sampled += 1;
        d += stride;
    }
    let total = doc_end - doc_start + 1;
    let scale = total as f64 / sampled.max(1) as f64;
    gammapdfln(alpha[k], prior_shape, prior_scale) + data * scale
}

/// Pólya (Dirichlet-multinomial) log-likelihood of all documents in time
/// segment `[doc_start, doc_end]` under prior `alpha`. keyATM's `polyapdfln`.
fn dyn_polyapdfln(
    alpha: &[f64],
    doc_start: usize,
    doc_end: usize,
    ndk: &[Vec<f64>],
    doc_len: &[f64],
) -> f64 {
    let alpha_sum: f64 = alpha.iter().sum();
    let lg_alpha_sum = lgamma(alpha_sum);
    let lg_alpha: Vec<f64> = alpha.iter().map(|&a| lgamma(a)).collect();
    let mut loglik = 0.0;
    for d in doc_start..=doc_end {
        loglik += lg_alpha_sum - lgamma(doc_len[d] + alpha_sum);
        for (k, &a) in alpha.iter().enumerate() {
            loglik += lgamma(ndk[d][k] + a) - lg_alpha[k];
        }
    }
    loglik
}

/// Slice-sample every entry of one state's `alpha` vector in place, over the
/// documents `[doc_start, doc_end]` that currently belong to the state. Keyword
/// topics use the Gamma(`eta1`, `eta2`) prior, regular topics Gamma(`eta1_reg`,
/// `eta2_reg`). Mirrors keyATM's `sample_alpha_state`.
///
/// `stride` is the opt-in turbo subsampling stride for the per-document data
/// term: `1` is exact (every document contributes); `> 1` evaluates only every
/// `stride`-th document and scales the data term up by the inverse fraction, an
/// approximation that trades a small bias in the estimated α for a large cut in
/// the slice sampler's lgamma cost (the dominant non-sweep cost on large corpora).
#[allow(clippy::too_many_arguments)]
fn dyn_sample_alpha_state<R: Rng>(
    alpha: &mut [f64],
    num_keyword_topics: usize,
    doc_start: usize,
    doc_end: usize,
    ndk: &[Vec<f64>],
    doc_len: &[f64],
    eta1: f64,
    eta2: f64,
    eta1_reg: f64,
    eta2_reg: f64,
    min_v: f64,
    max_v: f64,
    stride: usize,
    rng: &mut R,
) {
    const MAX_SHRINK_TIME: usize = 200;
    let num_topics = alpha.len();
    let order = shuffled_topic_ids(num_topics, rng);

    for &k in &order {
        let (shape, scale) = if k < num_keyword_topics {
            (eta1, eta2)
        } else {
            (eta1_reg, eta2_reg)
        };

        let keep = alpha[k];
        let store_loglik = dyn_alpha_loglik_sub(
            alpha, k, doc_start, doc_end, ndk, doc_len, shape, scale, stride,
        );

        let mut start = min_v;
        let mut end = max_v;
        let previous_p = shrinkp(alpha[k]);
        let slice_ =
            store_loglik - 2.0 * (1.0 - previous_p).ln() + rng.gen::<f64>().max(1e-300).ln();

        let mut accepted = false;
        for _ in 0..MAX_SHRINK_TIME {
            let new_p = start + (end - start) * rng.gen::<f64>();
            alpha[k] = new_p / (1.0 - new_p); // expandp
            let new_loglik = dyn_alpha_loglik_sub(
                alpha, k, doc_start, doc_end, ndk, doc_len, shape, scale, stride,
            );
            let new_likelihood = new_loglik - 2.0 * (1.0 - new_p).ln();

            if slice_ < new_likelihood {
                accepted = true;
                break;
            } else if previous_p < new_p {
                end = new_p;
            } else if new_p < previous_p {
                start = new_p;
            } else {
                // The shrink interval has collapsed onto the current point; stop
                // and restore below (falling through to the `!accepted` guard).
                break;
            }
        }
        // A slice step that never lands inside the slice — every one of the
        // MAX_SHRINK_TIME proposals rejected, or the interval collapsing onto the
        // current point — must return the current value, not the last rejected
        // proposal (#418). Only the accept branch keeps the drawn value; every
        // other exit restores `keep`, preserving the slice-sampler invariant.
        // (The RNG draw count is unchanged, so a fit that never hit fall-through
        // stays bit-identical.)
        if !accepted {
            alpha[k] = keep;
        }
    }
}

/// Fisher–Yates shuffle of `0..n` using `rng` (deterministic for a fixed seed).
fn shuffled_topic_ids<R: Rng>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.gen::<f64>() * (i as f64 + 1.0)) as usize % (i + 1);
        v.swap(i, j);
    }
    v
}

/// Fit a keyATM **Dynamic** model: a Chib (1998) change-point HMM over time
/// segments, each state carrying its own document-topic prior `alphas[r]`
/// (Eshima, Imai & Sasaki 2024, Section 3.3).
///
/// `time_index` gives the 0-based time segment of each document. Documents must
/// be sorted by time so segments are contiguous (validated). `num_states` is the
/// number of latent regimes the prevalence path is allowed to occupy.
///
/// The token (z, s) sampler is identical to the base model; on top of it each
/// sweep slice-samples the per-state `alphas`, then runs forward filtering /
/// backward sampling (FFBS) of the state path and resamples the left-to-right
/// transition matrix.
#[allow(clippy::too_many_arguments)]
pub fn fit_keyatm_dynamic<R: Rng, F: FnMut(usize, usize, f64, f64)>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    time_index: &[usize],
    num_states: usize,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    eta1: f64,
    eta2: f64,
    eta1_reg: f64,
    eta2_reg: f64,
    iters: usize,
    ll_interval: usize,
    weights: WeightScheme,
    num_threads: usize,
    draws: ThetaDrawOpts,
    convergence_tol: f64,
    mut on_progress: F,
    rng: &mut R,
) -> KeyAtmModel {
    assert_eq!(
        keywords.len(),
        num_topics,
        "keywords length must equal num_topics"
    );
    assert_eq!(
        time_index.len(),
        docs.len(),
        "time_index length must equal number of documents"
    );
    assert!(num_states >= 1, "num_states must be at least 1");
    for (name, val) in [
        ("eta1", eta1),
        ("eta2", eta2),
        ("eta1_reg", eta1_reg),
        ("eta2_reg", eta2_reg),
    ] {
        assert!(
            val.is_finite() && val > 0.0,
            "{name} must be finite and positive (got {val})"
        );
    }

    // Time segments must be contiguous and non-decreasing (docs sorted by time).
    for w in time_index.windows(2) {
        assert!(
            w[1] >= w[0],
            "documents must be sorted by time_index (non-decreasing)"
        );
    }
    let num_time = time_index
        .iter()
        .copied()
        .max()
        .map(|m| {
            m.checked_add(1)
                .expect("time_index entry is usize::MAX; segment count overflows")
        })
        .unwrap_or(0);
    assert!(
        num_time >= num_states,
        "num_time ({num_time}) must be >= num_states ({num_states})"
    );
    // Contiguity: every segment in `0..num_time` must contain at least one
    // document. A gap (e.g. time_index skipping a period, or not starting at 0)
    // leaves that segment's `time_doc_start`/`time_doc_end` at 0 and can underflow
    // `time_doc_start[t + 1] - 1` downstream (#418, item 3).
    let mut segment_seen = vec![false; num_time];
    for &t in time_index {
        segment_seen[t] = true;
    }
    assert!(
        segment_seen.iter().all(|&seen| seen),
        "time_index must be gap-free: every segment in 0..{num_time} must contain at \
         least one document (and it must start at 0)"
    );

    // Document index ranges for each time segment.
    let mut time_doc_start = vec![0usize; num_time];
    let mut time_doc_end = vec![0usize; num_time];
    {
        let mut prev: i64 = -1;
        for (d, &t) in time_index.iter().enumerate() {
            if t as i64 != prev {
                time_doc_start[t] = d;
                prev = t as i64;
            }
        }
        for t in 0..num_time - 1 {
            time_doc_end[t] = time_doc_start[t + 1] - 1;
        }
        time_doc_end[num_time - 1] = docs.len() - 1;
    }

    let min_v = shrinkp(1e-9);
    let max_v = shrinkp(100.0);

    // α is replaced by the per-state HMM prior; pass a nominal 1.0 for the struct.
    let (mut model, mut assignments, ki) = init_state(
        docs, num_types, num_topics, keywords, 1.0, beta, beta_key, gamma1, gamma2, weights, rng,
    );
    let num_keyword_topics = model.num_keyword_topics;

    // Weighted document lengths (Σ token weights), matching the weighted counts
    // the Dirichlet-multinomial marginal (alpha_loglik / polyapdfln) consumes.
    let doc_len: Vec<f64> = docs
        .iter()
        .map(|d| d.iter().map(|&w| model.vocab_weights[w as usize]).sum())
        .collect();

    // --- Initialise HMM state ---
    // Per-state α, keyATM's 50/K start.
    let mut alphas = vec![vec![50.0 / num_topics as f64; num_topics]; num_states];

    // State path: contiguous near-even split, so every state holds >= 1 segment.
    let mut r_est = vec![0usize; num_time];
    {
        let base = num_time / num_states;
        let rem = num_time % num_states;
        let mut idx = 0;
        for r in 0..num_states {
            let cnt = base + usize::from(r < rem);
            for _ in 0..cnt {
                r_est[idx] = r;
                idx += 1;
            }
        }
    }

    // Left-to-right transition matrix: diagonal Beta(1,1), super-diagonal the
    // complement, last state absorbing.
    let mut p_est = vec![vec![0.0f64; num_states]; num_states];
    for r in 0..num_states - 1 {
        let pii = sample_beta(1.0, 1.0, rng);
        p_est[r][r] = pii;
        p_est[r][r + 1] = 1.0 - pii;
    }
    p_est[num_states - 1][num_states - 1] = 1.0;

    let mut r_count = vec![0usize; num_states];
    for &r in &r_est {
        r_count[r] += 1;
    }

    let mut prk = vec![vec![0.0f64; num_states]; num_time];
    let mut theta_draw_buf: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut nkw_t = build_nkw_t(&model.nkw, num_types, num_topics);

    for it in 0..iters {
        // 1. Token (z, s) sweep with each doc's α tied to its segment's state.
        let doc_alpha: Vec<Vec<f64>> = time_index
            .iter()
            .map(|&t| alphas[r_est[t]].clone())
            .collect();
        run_sweep(
            &mut model,
            docs,
            &mut assignments,
            &mut nkw_t,
            &ki,
            &doc_alpha,
            num_threads,
            rng,
        );
        // Snapshot under the (ndk, doc_alpha) pair this sweep just used — coherent
        // smoothing even though r_est/alphas are resampled below.
        draws.maybe_collect(&mut theta_draw_buf, it + 1, &model.ndk, &doc_alpha);

        // 2. Slice-sample each state's α over the documents it currently owns.
        // States are contiguous in time; their document ranges are disjoint, so
        // the per-state slice sampling is independent and parallelises cleanly.
        // The α slice sampler dominates the per-sweep cost on large corpora.
        let ranges: Vec<(usize, usize)> = {
            let mut seg = 0usize;
            (0..num_states)
                .map(|r| {
                    let seg_start = seg;
                    let seg_end = seg + r_count[r] - 1;
                    seg = seg_end + 1;
                    (time_doc_start[seg_start], time_doc_end[seg_end])
                })
                .collect()
        };
        if num_threads > 1 {
            use rand::SeedableRng;
            use rand_pcg::Pcg64Mcg;
            use rayon::prelude::*;
            let base: u64 = rng.gen();
            let ndk = &model.ndk;
            alphas
                .par_iter_mut()
                .zip(ranges.par_iter())
                .enumerate()
                .for_each(|(r, (alpha, &(d_start, d_end)))| {
                    let mut srng = Pcg64Mcg::seed_from_u64(
                        base ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    );
                    dyn_sample_alpha_state(
                        alpha,
                        num_keyword_topics,
                        d_start,
                        d_end,
                        ndk,
                        &doc_len,
                        eta1,
                        eta2,
                        eta1_reg,
                        eta2_reg,
                        min_v,
                        max_v,
                        1,
                        &mut srng,
                    );
                });
        } else {
            for (r, &(d_start, d_end)) in ranges.iter().enumerate() {
                dyn_sample_alpha_state(
                    &mut alphas[r],
                    num_keyword_topics,
                    d_start,
                    d_end,
                    &model.ndk,
                    &doc_len,
                    eta1,
                    eta2,
                    eta1_reg,
                    eta2_reg,
                    min_v,
                    max_v,
                    1,
                    rng,
                );
            }
        }

        // 3. Forward filter: Prk[t][r] = p(state_t = r | y_{1..t}). The Pólya
        // likelihoods logfy[t][r] are independent across (t, r); precompute them
        // (in parallel when threaded — this does not change results, only speed).
        let logfy_all: Vec<Vec<f64>> = if num_threads > 1 {
            use rayon::prelude::*;
            let ndk = &model.ndk;
            (1..num_time)
                .into_par_iter()
                .map(|t| {
                    (0..num_states)
                        .map(|r| {
                            dyn_polyapdfln(
                                &alphas[r],
                                time_doc_start[t],
                                time_doc_end[t],
                                ndk,
                                &doc_len,
                            )
                        })
                        .collect()
                })
                .collect()
        } else {
            (1..num_time)
                .map(|t| {
                    (0..num_states)
                        .map(|r| {
                            dyn_polyapdfln(
                                &alphas[r],
                                time_doc_start[t],
                                time_doc_end[t],
                                &model.ndk,
                                &doc_len,
                            )
                        })
                        .collect()
                })
                .collect()
        };

        for v in prk[0].iter_mut() {
            *v = 0.0;
        }
        prk[0][0] = 1.0; // first segment is always state 0
        for t in 1..num_time {
            let logfy = &logfy_all[t - 1];
            // rt_k[r] = Σ_j Prk[t-1][j] · P[j][r]
            let rt_k: Vec<f64> = (0..num_states)
                .map(|r| (0..num_states).map(|j| prk[t - 1][j] * p_est[j][r]).sum())
                .collect();
            // Normalise log(rt_k[r]) + logfy[r] over the non-zero entries.
            let mut log_unnorm = vec![f64::NEG_INFINITY; num_states];
            for r in 0..num_states {
                if rt_k[r] > 0.0 {
                    log_unnorm[r] = rt_k[r].ln() + logfy[r];
                }
            }
            let m = log_unnorm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let logsum = m + log_unnorm.iter().map(|&v| (v - m).exp()).sum::<f64>().ln();
            for r in 0..num_states {
                prk[t][r] = if rt_k[r] > 0.0 {
                    (log_unnorm[r] - logsum).exp()
                } else {
                    0.0
                };
            }
        }

        // 4. Backward sample the state path; last segment is the final state.
        for c in r_count.iter_mut() {
            *c = 0;
        }
        r_est[num_time - 1] = num_states - 1;
        r_count[num_states - 1] += 1;
        for t in (0..num_time - 1).rev() {
            let next = r_est[t + 1];
            let probs: Vec<f64> = (0..num_states)
                .map(|r| prk[t][r] * p_est[r][next])
                .collect();
            let s = sample_index(&probs, rng);
            r_est[t] = s;
            r_count[s] += 1;
        }

        // 5. Resample the transition matrix from the new path.
        for r in 0..num_states - 1 {
            let pii = sample_beta(r_count[r] as f64, 2.0, rng);
            p_est[r][r] = pii;
            p_est[r][r + 1] = 1.0 - pii;
        }

        if record_ll(it + 1, ll_interval, iters) {
            // doc_topic()'s dynamic path needs the current HMM state; install a
            // snapshot so the per-token log-likelihood reflects it.
            model.dynamic = Some(DynamicState {
                num_states,
                num_time,
                time_index: time_index.to_vec(),
                alphas: alphas.clone(),
                r_est: r_est.clone(),
                p_est: p_est.clone(),
            });
            let ll = model.model_loglik();
            let ppl = model.perplexity();
            model.log_likelihood_history.push((it + 1, ll, ppl));
            on_progress(it + 1, iters, ll, ppl);
            model.pi_history.push((it + 1, model.keyword_rate()));
            if ll_converged(&model.log_likelihood_history, convergence_tol) {
                model.converged = true;
                break;
            }
        }
    }

    model.dynamic = Some(DynamicState {
        num_states,
        num_time,
        time_index: time_index.to_vec(),
        alphas,
        r_est,
        p_est,
    });
    model.theta_draws = theta_draw_buf;
    model
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Covariate standardization (#270): non-intercept columns come out mean ~0,
    /// sd ~1, the intercept is untouched, and mapping λ from standardized back to
    /// the original scale reproduces the same α = exp(features · λ) for every doc.
    #[test]
    fn standardize_and_lambda_mapback_roundtrip() {
        // features: [intercept, a continuous col, a 0/1 dummy col]
        let features = vec![
            vec![1.0, 2.0, 0.0],
            vec![1.0, 4.0, 1.0],
            vec![1.0, 6.0, 0.0],
            vec![1.0, 8.0, 1.0],
        ];
        let nf = 3;
        let (fstd, mean, sd) = standardize_features(&features, nf);
        // Intercept untouched.
        assert!(fstd.iter().all(|r| (r[0] - 1.0).abs() < 1e-12));
        assert!((mean[0]).abs() < 1e-12 && (sd[0] - 1.0).abs() < 1e-12);
        // Non-intercept columns standardized to mean ~0, sd ~1.
        for f in 1..nf {
            let m: f64 = fstd.iter().map(|r| r[f]).sum::<f64>() / 4.0;
            let v: f64 = fstd.iter().map(|r| (r[f] - m).powi(2)).sum::<f64>() / 4.0;
            assert!(m.abs() < 1e-9, "col {f} mean {m}");
            assert!((v.sqrt() - 1.0).abs() < 1e-9, "col {f} sd {}", v.sqrt());
        }
        // A λ in standardized space, mapped back to the original scale, must give
        // the same linear predictor (hence the same α) on the raw features.
        let lambda_std = vec![vec![0.5, -1.2, 0.8], vec![-0.3, 2.0, -1.5]];
        let lambda_orig: Vec<Vec<f64>> = lambda_std
            .iter()
            .map(|lk| {
                let mut row = lk.clone();
                let mut shift = 0.0;
                for f in 1..nf {
                    row[f] = lk[f] / sd[f];
                    shift += lk[f] * mean[f] / sd[f];
                }
                row[0] = lk[0] - shift;
                row
            })
            .collect();
        let a_std = crate::dmr::compute_doc_alpha(&lambda_std, &fstd, None, 0.0);
        let a_orig = crate::dmr::compute_doc_alpha(&lambda_orig, &features, None, 0.0);
        for d in 0..features.len() {
            for k in 0..lambda_std.len() {
                assert!(
                    (a_std[d][k] - a_orig[d][k]).abs() < 1e-9,
                    "doc {d} topic {k}: {} vs {}",
                    a_std[d][k],
                    a_orig[d][k]
                );
            }
        }
    }

    // The covariate λ SE (#316) maps the standardized-space covariance to the
    // original scale by the standardization Jacobian. Check the helper against a
    // brute-force full-matrix transform (M ⊗-block applied to the whole p×p
    // covariance), and that a clamped coefficient is flagged NaN.
    #[test]
    fn covariate_lambda_se_matches_full_jacobian_transform() {
        let (t, f) = (3usize, 3usize);
        let features = vec![
            vec![1.0, 2.0, 0.0],
            vec![1.0, 4.0, 1.0],
            vec![1.0, 6.0, 0.0],
            vec![1.0, 8.0, 1.0],
            vec![1.0, 5.0, 1.0],
            vec![1.0, 3.0, 0.0],
        ];
        let nf = f;
        let (fstd, mean, sd) = standardize_features(&features, nf);
        let lambda_std = vec![
            vec![0.5, -1.2, 0.8],
            vec![-0.3, 2.0, -1.5],
            vec![0.1, 0.4, -0.6],
        ];
        let counts = vec![
            vec![3.0f64, 1.0, 0.0],
            vec![0.0f64, 2.0, 2.0],
            vec![1.0f64, 1.0, 5.0],
            vec![2.0f64, 0.0, 1.0],
            vec![1.0f64, 3.0, 2.0],
            vec![4.0f64, 1.0, 1.0],
        ];
        let sigma2 = 1.0;

        // Reference: full p×p covariance in std space, transformed by the full
        // block-diagonal Jacobian J (each topic block = M), SE = sqrt(diag).
        let p = t * f;
        let cov = crate::dmr::dmr_lambda_cov(&lambda_std, &fstd, &counts, t, f, sigma2, 0.0, None);
        // M (F×F): row 0 = [1, -mean_g/sd_g...]; row g≥1 = e_g / sd_g.
        let mut m = vec![vec![0.0f64; f]; f];
        m[0][0] = 1.0;
        for g in 1..f {
            m[0][g] = -mean[g] / sd[g];
            m[g][g] = 1.0 / sd[g];
        }
        // Full J (p×p) block-diagonal with M on each topic block.
        let mut jac = vec![0.0f64; p * p];
        for k in 0..t {
            for i in 0..f {
                for j in 0..f {
                    jac[(k * f + i) * p + (k * f + j)] = m[i][j];
                }
            }
        }
        // cov_orig = J cov Jᵀ; reference SE = sqrt(diag).
        let mut jc = vec![0.0f64; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for l in 0..p {
                    s += jac[i * p + l] * cov[l * p + j];
                }
                jc[i * p + j] = s;
            }
        }
        let mut ref_se = vec![vec![0.0f64; f]; t];
        for k in 0..t {
            for ff in 0..f {
                let i = k * f + ff;
                let mut var = 0.0;
                for l in 0..p {
                    var += jc[i * p + l] * jac[i * p + l];
                }
                ref_se[k][ff] = var.sqrt();
            }
        }

        let se = covariate_lambda_se(&lambda_std, &fstd, &counts, t, f, sigma2, &mean, &sd, None);
        for k in 0..t {
            for ff in 0..f {
                assert!(
                    (se[k][ff] - ref_se[k][ff]).abs() < 1e-9,
                    "se[{k}][{ff}]: helper {} vs full-transform {}",
                    se[k][ff],
                    ref_se[k][ff]
                );
            }
        }

        // A coefficient at the ±LAMBDA_BOUND clamp is flagged NaN.
        let mut clamped = lambda_std.clone();
        clamped[1][2] = LAMBDA_BOUND;
        let se_c = covariate_lambda_se(&clamped, &fstd, &counts, t, f, sigma2, &mean, &sd, None);
        assert!(se_c[1][2].is_nan(), "clamped λ must give NaN SE");
        assert!(se_c[0][0].is_finite(), "unclamped entries stay finite");
    }

    /// Build a small keyword-annotated corpus and verify that keyword topics
    /// are pulled toward their designated keyword blocks.
    ///
    /// Vocabulary layout:
    /// - Block A: words 0..10
    /// - Block B: words 10..20
    /// - Block C: words 20..30
    ///
    /// Topic 0 keywords: [0, 1]  (block A words)
    /// Topic 1 keywords: [10, 11] (block B words)
    /// Topic 2: regular (no keywords)
    ///
    /// Docs: 100 from block A, 100 from block B, 100 from block C, each with
    /// a small amount of noise from a different block.
    #[test]
    fn keywords_steer_topics() {
        let block_size = 10usize;
        let num_blocks = 3usize;
        let v = num_blocks * block_size; // V = 30

        // 100 docs per block, each doc = 5 tokens cycling through block words +
        // 1 noise token from a different block.
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new(); // ground-truth block
        for b in 0..num_blocks {
            let offset = b * block_size;
            for d in 0..100usize {
                let mut doc: Vec<u32> = (0..5)
                    .map(|i| (offset + (i + d) % block_size) as u32)
                    .collect();
                // one noise token from the next block
                let noise_block = (b + 1) % num_blocks;
                doc.push((noise_block * block_size + d % block_size) as u32);
                docs.push(doc);
                labels.push(b);
            }
        }

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1],   // topic 0 → block A keywords
            vec![10, 11], // topic 1 → block B keywords
            vec![],       // topic 2 → regular
        ];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let model = fit_keyatm(
            &docs,
            v,
            3,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            200,
            0,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut rng,
        );

        // Helper: block with max probability mass in topic_word(k).
        let dominant_block = |k: usize| -> usize {
            let phi = model.topic_word(k);
            (0..num_blocks)
                .max_by(|&ba, &bb| {
                    let sa: f64 = (0..block_size).map(|i| phi[ba * block_size + i]).sum();
                    let sb: f64 = (0..block_size).map(|i| phi[bb * block_size + i]).sum();
                    sa.partial_cmp(&sb).unwrap()
                })
                .unwrap()
        };

        // Topic 0 should be dominated by block A (block 0).
        assert_eq!(
            dominant_block(0),
            0,
            "topic 0 (keyword=block A) should be dominated by block A words"
        );
        // Topic 1 should be dominated by block B (block 1).
        assert_eq!(
            dominant_block(1),
            1,
            "topic 1 (keyword=block B) should be dominated by block B words"
        );
    }

    #[test]
    fn cvb0_keywords_steer_topics() {
        // The CVB0 backend must steer keyword topics the same way as Gibbs.
        let block_size = 10usize;
        let num_blocks = 3usize;
        let v = num_blocks * block_size;
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for b in 0..num_blocks {
            let offset = b * block_size;
            for d in 0..100usize {
                let mut doc: Vec<u32> = (0..5)
                    .map(|i| (offset + (i + d) % block_size) as u32)
                    .collect();
                doc.push(((b + 1) % num_blocks * block_size + d % block_size) as u32);
                docs.push(doc);
            }
        }
        let keywords: Vec<Vec<usize>> = vec![vec![0, 1], vec![10, 11], vec![]];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let model = fit_keyatm_cvb0(
            &docs,
            v,
            3,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            100,
            WeightScheme::None,
            &mut rng,
        );
        let dominant_block = |k: usize| -> usize {
            let phi = model.topic_word(k);
            (0..num_blocks)
                .max_by(|&ba, &bb| {
                    let sa: f64 = (0..block_size).map(|i| phi[ba * block_size + i]).sum();
                    let sb: f64 = (0..block_size).map(|i| phi[bb * block_size + i]).sum();
                    sa.partial_cmp(&sb).unwrap()
                })
                .unwrap()
        };
        assert_eq!(dominant_block(0), 0, "cvb0 topic 0 should be block A");
        assert_eq!(dominant_block(1), 1, "cvb0 topic 1 should be block B");
    }

    #[test]
    fn cvb0_deterministic_for_seed() {
        let docs: Vec<Vec<u32>> = (0..60)
            .map(|d| (0..6).map(|i| ((i + d) % 12) as u32).collect())
            .collect();
        let keywords: Vec<Vec<usize>> = vec![vec![0], vec![6], vec![]];
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            fit_keyatm_cvb0(
                &docs,
                12,
                3,
                &keywords,
                0.1,
                0.1,
                0.5,
                1.0,
                1.0,
                40,
                WeightScheme::None,
                &mut rng,
            )
            .doc_topic()
        };
        let (a, b) = (run(), run());
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    /// Every `keyword_rate()` value must lie in [0, 1], and regular topics
    /// must return exactly 0.0.
    #[test]
    fn keyword_rate_in_unit_interval() {
        let v = 30usize;
        let docs: Vec<Vec<u32>> = (0..60usize)
            .map(|d| (0..4).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1, 2], // keyword topic
            vec![10, 11],  // keyword topic
            vec![],        // regular
            vec![],        // regular
        ];

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let model = fit_keyatm(
            &docs,
            v,
            4,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            50,
            0,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut rng,
        );

        let rates = model.keyword_rate();
        assert_eq!(rates.len(), 4);

        // Keyword topics must be in [0, 1].
        for k in 0..2 {
            assert!(
                rates[k] >= 0.0 && rates[k] <= 1.0,
                "keyword_rate[{k}] = {} not in [0,1]",
                rates[k]
            );
        }
        // Regular topics must be exactly 0.
        for k in 2..4 {
            assert_eq!(
                rates[k], 0.0,
                "keyword_rate[{k}] should be 0 for regular topic"
            );
        }
    }

    /// Shape and normalisation invariants.
    #[test]
    fn shapes_and_normalisation() {
        let v = 20usize;
        let docs: Vec<Vec<u32>> = (0..40usize)
            .map(|d| (0..3).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1, 2], // keyword topic
            vec![],        // regular
            vec![],        // regular
        ];

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let model = fit_keyatm(
            &docs,
            v,
            3,
            &keywords,
            0.5,
            0.1,
            0.5,
            1.0,
            1.0,
            30,
            0,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut rng,
        );

        let d = docs.len();
        let k = 3usize;

        // topic_word(k) length = V, sums to 1.
        for kk in 0..k {
            let phi = model.topic_word(kk);
            assert_eq!(phi.len(), v, "topic_word({kk}) length should be num_types");
            let s: f64 = phi.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "topic_word({kk}) sums to {s:.12}, expected 1.0"
            );
        }

        // doc_topic() is D×K, rows sum to 1.
        let theta = model.doc_topic();
        assert_eq!(theta.len(), d, "doc_topic() should have D rows");
        for (dd, row) in theta.iter().enumerate() {
            assert_eq!(row.len(), k, "doc_topic() row {dd} should have K columns");
            let s: f64 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "doc_topic() row {dd} sums to {s:.12}, expected 1.0"
            );
        }
    }

    /// Dynamic model: a corpus whose topic mix changes halfway through time
    /// should produce a monotone state path that flips at the change point, and
    /// the per-state α should reflect the shift in prevalence.
    #[test]
    fn dynamic_recovers_change_point() {
        // Vocab: block A = 0..6 (economic), block B = 6..12 (social).
        // Keyword topic 0 -> A, topic 1 -> B.
        let keywords: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![6, 7, 8]];
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut time_index: Vec<usize> = Vec::new();
        let num_time = 10usize;
        for t in 0..num_time {
            // First half mostly block A; second half mostly block B.
            let b_heavy = t >= 5;
            for d in 0..30usize {
                let (heavy_off, light_off) = if b_heavy { (6, 0) } else { (0, 6) };
                let mut doc: Vec<u32> = (0..6).map(|i| (heavy_off + (i + d) % 6) as u32).collect();
                doc.push((light_off + d % 6) as u32);
                docs.push(doc);
                time_index.push(t);
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let model = fit_keyatm_dynamic(
            &docs,
            12,
            2,
            &keywords,
            &time_index,
            2,
            0.01,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            300,
            0,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            |_, _, _, _| {},
            &mut rng,
        );

        let dyn_ = model.dynamic.as_ref().expect("dynamic state present");
        // State path is non-decreasing and visits both states.
        for w in dyn_.r_est.windows(2) {
            assert!(w[1] >= w[0], "state path must be non-decreasing");
        }
        assert_eq!(dyn_.r_est[0], 0);
        assert_eq!(dyn_.r_est[num_time - 1], 1);

        // Smoothed prevalence: social topic (1) should rise in the later state.
        let tp = model.time_prevalence().unwrap();
        let early = tp[0][1];
        let late = tp[num_time - 1][1];
        assert!(
            late - early > 0.3,
            "social prevalence should rise: {early} -> {late}"
        );
    }

    /// Dynamic fits with the same seed must be identical.
    #[test]
    fn dynamic_deterministic_for_fixed_seed() {
        let keywords: Vec<Vec<usize>> = vec![vec![0, 1], vec![6, 7]];
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut time_index: Vec<usize> = Vec::new();
        for t in 0..6usize {
            for d in 0..15usize {
                let off = if t >= 3 { 6 } else { 0 };
                docs.push((0..5).map(|i| (off + (i + d) % 6) as u32).collect());
                time_index.push(t);
            }
        }
        let mut r1 = ChaCha8Rng::seed_from_u64(9);
        let mut r2 = ChaCha8Rng::seed_from_u64(9);
        let m1 = fit_keyatm_dynamic(
            &docs,
            12,
            2,
            &keywords,
            &time_index,
            2,
            0.01,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            100,
            0,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            |_, _, _, _| {},
            &mut r1,
        );
        let m2 = fit_keyatm_dynamic(
            &docs,
            12,
            2,
            &keywords,
            &time_index,
            2,
            0.01,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            100,
            0,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            |_, _, _, _| {},
            &mut r2,
        );
        assert_eq!(
            m1.dynamic.as_ref().unwrap().r_est,
            m2.dynamic.as_ref().unwrap().r_est
        );
        assert_eq!(m1.doc_topic(), m2.doc_topic());
    }

    /// Two fits with the same seed must be bit-for-bit identical.
    #[test]
    fn deterministic_for_fixed_seed() {
        let v = 15usize;
        let docs: Vec<Vec<u32>> = (0..30usize)
            .map(|d| (0..4).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1],    // keyword topic
            vec![5, 6, 7], // keyword topic
            vec![],        // regular
        ];

        let mut r1 = ChaCha8Rng::seed_from_u64(77);
        let mut r2 = ChaCha8Rng::seed_from_u64(77);

        let m1 = fit_keyatm(
            &docs,
            v,
            3,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            40,
            0,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut r1,
        );
        let m2 = fit_keyatm(
            &docs,
            v,
            3,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            40,
            0,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut r2,
        );

        assert_eq!(
            m1.topic_word_all(),
            m2.topic_word_all(),
            "topic_word_all() differs between two identical-seed runs"
        );
        assert_eq!(
            m1.doc_topic(),
            m2.doc_topic(),
            "doc_topic() differs between two identical-seed runs"
        );
    }

    #[test]
    fn log_likelihood_trace_is_recorded_and_rises() {
        // The per-token log-likelihood trace should be populated at the requested
        // cadence (plus the final sweep) and trend upward as Gibbs converges.
        let v = 30usize;
        let mut docs: Vec<Vec<u32>> = Vec::new();
        for b in 0..3usize {
            let off = (b * 10) as u32;
            for d in 0..80usize {
                docs.push((0..6).map(|i| off + ((i + d) % 10) as u32).collect());
            }
        }
        let keywords = vec![vec![0usize, 1], vec![10, 11], vec![]];

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let model = fit_keyatm(
            &docs,
            v,
            3,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            60,
            10,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut rng,
        );
        let h = &model.log_likelihood_history;
        // iters=60, interval=10 -> sweeps 10,20,30,40,50,60.
        assert_eq!(
            h.iter().map(|&(i, _, _)| i).collect::<Vec<_>>(),
            vec![10, 20, 30, 40, 50, 60]
        );
        // (iter, collapsed log-likelihood < 0, perplexity > 1).
        assert!(h
            .iter()
            .all(|&(_, ll, ppl)| ll.is_finite() && ll < 0.0 && ppl > 1.0));
        // The collapsed log-likelihood rises (toward 0) as Gibbs converges.
        assert!(
            h.last().unwrap().1 >= h.first().unwrap().1 - 1e-6,
            "log-likelihood degraded: {h:?}"
        );

        // interval 0 disables tracing.
        let mut rng2 = ChaCha8Rng::seed_from_u64(3);
        let none = fit_keyatm(
            &docs,
            v,
            3,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            20,
            0,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut rng2,
        );
        assert!(none.log_likelihood_history.is_empty());
    }

    #[test]
    fn keyatm_conforms() {
        let v = 30usize;
        let docs: Vec<Vec<u32>> = (0..60usize)
            .map(|d| (0..5).map(|i| ((i + d * 3) % v) as u32).collect())
            .collect();
        let keywords: Vec<Vec<usize>> = vec![vec![0, 1], vec![10, 11], vec![]];
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let m = fit_keyatm(
            &docs,
            v,
            3,
            &keywords,
            0.1,
            0.1,
            0.5,
            1.0,
            1.0,
            20,
            0,
            true,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
        let dir = crate::conformance::check_dirichlet(&m);
        assert!(dir.is_empty(), "check_dirichlet: {:?}", dir);
    }

    /// Turbo: `alpha_stride = 1` must reproduce the exact alpha-sampler path
    /// bit-for-bit (it is the same code path), and `> 1` must stay deterministic
    /// for a fixed seed.
    #[test]
    fn turbo_alpha_stride_one_is_exact_and_higher_is_deterministic() {
        let v = 40usize;
        let docs: Vec<Vec<u32>> = (0..400usize)
            .map(|d| (0..6).map(|i| ((i + d * 5) % v) as u32).collect())
            .collect();
        let keywords: Vec<Vec<usize>> = vec![vec![0, 1], vec![10, 11], vec![]];
        let f = |stride: usize, seed: u64| {
            let mut r = ChaCha8Rng::seed_from_u64(seed);
            fit_keyatm(
                &docs,
                v,
                3,
                &keywords,
                0.1,
                0.1,
                0.5,
                1.0,
                1.0,
                60,
                0,
                true,
                WeightScheme::None,
                1,
                ThetaDrawOpts::new(false, 0, 0),
                0.0,
                stride,
                |_, _, _, _| {},
                &mut r,
            )
        };
        // stride = 1 is the exact path: identical to a no-stride fit.
        let exact = f(1, 5);
        let exact2 = f(1, 5);
        assert_eq!(exact.topic_word_all(), exact2.topic_word_all());
        assert_eq!(exact.doc_topic(), exact2.doc_topic());
        // stride > 1 is deterministic for a fixed seed.
        let t_a = f(4, 9);
        let t_b = f(4, 9);
        assert_eq!(t_a.topic_word_all(), t_b.topic_word_all());
        assert_eq!(t_a.doc_topic(), t_b.doc_topic());
    }

    /// #418: a slice step that never accepts must return the current value, not
    /// the last rejected proposal. `dyn_sample_alpha_state` backs both the base
    /// (single-state) and dynamic keyATM α samplers, so this invariant protects
    /// the default `estimate_alpha=true` path too. A non-finite likelihood (here a
    /// NaN in `doc_len`) makes `slice_ < new_likelihood` false for every proposal,
    /// forcing the shrink loop to exhaust `MAX_SHRINK_TIME` without accepting —
    /// exactly the fall-through the fix guards. Every α entry must be restored to
    /// its pre-call value; before the fix each was left at a rejected draw.
    #[test]
    fn dyn_alpha_slice_restores_current_value_when_no_proposal_is_accepted() {
        let num_keyword_topics = 1; // topic count is taken from `alpha.len()` (3)
                                    // Two documents; a NaN weighted length forces a NaN log-likelihood for
                                    // every candidate α, so no proposal can land inside the slice.
        let ndk: Vec<Vec<f64>> = vec![vec![2.0, 1.0, 3.0], vec![1.0, 4.0, 1.0]];
        let doc_len = vec![f64::NAN, 6.0];
        let min_v = shrinkp(1e-9);
        let max_v = shrinkp(100.0);
        let initial = vec![0.7f64, 1.3, 0.4];
        let mut alpha = initial.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        dyn_sample_alpha_state(
            &mut alpha,
            num_keyword_topics,
            0,
            1,
            &ndk,
            &doc_len,
            1.0,
            1.0,
            2.0,
            1.0,
            min_v,
            max_v,
            1,
            &mut rng,
        );
        assert_eq!(
            alpha, initial,
            "a fully-rejected slice step must leave α at its current value, not a rejected proposal"
        );
    }

    // ----- #418 item 3: public-Rust-API input validation ---------------------

    /// A minimal base keyATM fit over a fixed two-token corpus, used to drive the
    /// input-validation guards below.
    fn run_base(
        docs: &[Vec<u32>],
        v: usize,
        k: usize,
        keywords: &[Vec<usize>],
        alpha: f64,
        beta: f64,
        estimate_alpha: bool,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        fit_keyatm(
            docs,
            v,
            k,
            keywords,
            alpha,
            beta,
            0.5,
            1.0,
            1.0,
            1,
            0,
            estimate_alpha,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            1,
            |_, _, _, _| {},
            &mut rng,
        );
    }

    fn two_docs() -> Vec<Vec<u32>> {
        vec![vec![0u32, 1], vec![1u32, 0]]
    }

    #[test]
    #[should_panic(expected = "docs must be non-empty")]
    fn rejects_empty_corpus() {
        // With estimate_alpha=true and iters>0 the base fit passes
        // `docs.len() - 1` to the α sampler; an empty corpus would wrap it to
        // usize::MAX. The validator rejects it first.
        run_base(&[], 2, 1, &[vec![0]], 0.1, 0.1, true);
    }

    #[test]
    #[should_panic(expected = "keyword topics must be the first topics")]
    fn rejects_empty_keyword_list_before_a_nonempty_one() {
        // topic 0 empty, topic 1 has keywords: the sampler would misroute switch
        // factors because num_keyword_topics=1 but the keyword topic is index 1.
        run_base(&two_docs(), 2, 2, &[vec![], vec![0]], 0.1, 0.1, true);
    }

    #[test]
    #[should_panic(expected = "more than once")]
    fn rejects_duplicate_keyword_within_a_topic() {
        run_base(&two_docs(), 2, 1, &[vec![0, 0]], 0.1, 0.1, true);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rejects_out_of_range_keyword_id() {
        run_base(&two_docs(), 2, 1, &[vec![5]], 0.1, 0.1, true);
    }

    #[test]
    #[should_panic(expected = "word id 5 out of range")]
    fn rejects_out_of_range_document_word_id() {
        run_base(&[vec![0u32, 5]], 2, 1, &[vec![0]], 0.1, 0.1, true);
    }

    #[test]
    #[should_panic(expected = "beta must be finite and positive")]
    fn rejects_nonpositive_beta() {
        run_base(&two_docs(), 2, 1, &[vec![0]], 0.1, 0.0, true);
    }

    #[test]
    #[should_panic(expected = "alpha must be finite and positive when estimate_alpha is false")]
    fn rejects_nonpositive_fixed_alpha() {
        run_base(&two_docs(), 2, 1, &[vec![0]], 0.0, 0.1, false);
    }

    #[test]
    fn accepts_nonpositive_alpha_when_estimated() {
        // α is only a start when estimated, so a 0 start is not rejected.
        run_base(&two_docs(), 2, 1, &[vec![0]], 0.0, 0.1, true);
    }

    #[test]
    fn sample_index_returns_in_range_for_valid_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..200 {
            let i = sample_index(&[1.0, 2.0, 3.0], &mut rng);
            assert!(i < 3);
        }
    }

    #[test]
    #[should_panic(expected = "non-finite or non-positive weight total")]
    fn sample_index_rejects_nan_weights() {
        // A NaN weight makes `r <= 0.0` never true; without the guard the loop
        // silently falls through to the last index. The debug_assert catches it.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let _ = sample_index(&[1.0, f64::NAN, 3.0], &mut rng);
    }

    fn run_dynamic(time_index: &[usize], num_states: usize, eta1: f64) {
        let docs: Vec<Vec<u32>> = time_index.iter().map(|_| vec![0u32, 1]).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        fit_keyatm_dynamic(
            &docs,
            2,
            1,
            &[vec![0]],
            time_index,
            num_states,
            0.1,
            0.5,
            1.0,
            1.0,
            eta1,
            1.0,
            2.0,
            1.0,
            1,
            0,
            WeightScheme::None,
            1,
            ThetaDrawOpts::new(false, 0, 0),
            0.0,
            |_, _, _, _| {},
            &mut rng,
        );
    }

    #[test]
    #[should_panic(expected = "gap-free")]
    fn rejects_time_index_with_a_gap() {
        // segment 1 is skipped: [0,0,2,2] leaves segment 1 empty.
        run_dynamic(&[0, 0, 2, 2], 1, 1.0);
    }

    #[test]
    #[should_panic(expected = "gap-free")]
    fn rejects_time_index_not_starting_at_zero() {
        run_dynamic(&[1, 1, 2, 2], 1, 1.0);
    }

    #[test]
    #[should_panic(expected = "eta1 must be finite and positive")]
    fn rejects_nonpositive_eta() {
        run_dynamic(&[0, 0, 1, 1], 1, 0.0);
    }
}
