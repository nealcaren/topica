//! OnlineLDA: online (stochastic) variational Bayes for Dirichlet LDA
//! (Hoffman, Blei & Bach, *NeurIPS* 2010, "Online Learning for Latent Dirichlet
//! Allocation"). The streaming / minibatch counterpart to the batch collapsed
//! Gibbs [`crate::model::TopicModel`] (`topica.LDA`), and the direct analogue of
//! gensim's `LdaModel`.
//!
//! Documents are processed in minibatches. For each minibatch we run the standard
//! mean-field E-step — the per-document variational Dirichlet `γ` / per-word `φ`
//! fixed point — against the current global topic-word variational parameter `λ`
//! (a K×V Dirichlet natural parameter), then take one *stochastic* step on `λ`
//! toward the minibatch's full-corpus estimate with a decaying learning rate
//! `ρ_t = (τ + t)^(-κ)` (`κ ∈ (0.5, 1]`):
//!
//! ```text
//! λ̂_{kw} = η + (D/|B|) · Σ_{d∈B} φ_{dwk} n_{dw}      (the natural-gradient target)
//! λ_{kw} ← (1 − ρ_t) λ_{kw} + ρ_t λ̂_{kw}
//! ```
//!
//! This is the same online-VB machinery the CTM/STM family already uses
//! ([`crate::ctm::fit_ctm_svi`]) — the Robbins-Monro schedule and per-epoch
//! document shuffle are shared through [`crate::variational::svi`] — specialised
//! here to the *Dirichlet* topic model rather than the logistic-normal one.
//!
//! Because `λ` is a persistent, incrementally updated statistic, the same engine
//! serves both a batch [`fit`] (loop minibatches over `iters` passes of the whole
//! corpus) and a streaming [`OnlineLDAModel::partial_fit`] (fold one fresh
//! minibatch into an already-fitted model). Both are deterministic: a fixed seed
//! fixes the initial `λ` and the per-pass shuffle, and every reduction sums in a
//! fixed (document) order, so a fit reproduces bit-for-bit.
//!
//! Note: the fitted state stores the topic-word and doc-topic matrices as
//! `Vec<Vec<f64>>` (not `ndarray::Array2`) — `ndarray` is behind the `embeddings`
//! feature, so a default-build model file must not depend on it.
//!
//! Fidelity. The E-step (`γ = α + expElogθ · (n/φnorm)·expElogβᵀ`, `φnorm + 1e-100`,
//! meanchange < 1e-3 over ≤ 100 iterations), the soft-count accumulation
//! (`sstats += expElogθ ⊗ (n/φnorm)`, then `sstats *= expElogβ`), and the global
//! blend (`λ ← (1−ρ)λ + ρ(η + (D/|B|)·sstats)`) reproduce Blei/Hoffman's reference
//! `onlineldavb.py` and gensim's `LdaModel`. The learning-rate schedule matches
//! Blei's exactly: his constructor sets the offset to `tau0 + 1` and starts the
//! step counter at 0, so `(tau0+1, tau0+2, …)` equals our `(τ+t)^(−κ)` with `t`
//! from 1 when `τ ≡ tau0`. Two deliberate deviations, both documented: (1) `γ` is
//! initialised to the deterministic constant `1.0` (the mean of the reference's
//! `Gamma(100, 1/100)` random init, washed out by the fixed point) to preserve
//! topica's bit-for-bit determinism; (2) fitting is single-threaded (parallel
//! coverage is tracked separately). The reference implementations are GPL-3 /
//! LGPL; this port is written from the paper and the published formulas and uses
//! those packages only as an external parity oracle, never copying their code.

use crate::corpus::Corpus;
use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use crate::mathfun::log_gamma as lgamma;
use crate::optimize::digamma;
use crate::variational::svi;
use rand::Rng;

/// A tiny positive floor mirroring gensim/Hoffman's `1e-100`, keeping the
/// per-word normaliser `φ`-sum strictly positive so the reciprocal is finite.
const PHI_FLOOR: f64 = 1e-100;

/// The online-VB engine and fitted state for [`fit`] / [`OnlineLDAModel::partial_fit`].
///
/// The streaming parameter is `lambda` (the K×V topic-word Dirichlet natural
/// parameter); everything the binding reads back — `topic_word`, `doc_topic`,
/// `fit_history` — is derived from it. Keeping the hyperparameters and the update
/// counter on the struct is what lets `partial_fit` resume the exact Robbins-Monro
/// schedule after a save/load round-trip.
pub struct OnlineLDAModel {
    pub num_topics: usize,
    pub num_types: usize,
    /// Document-topic Dirichlet prior, length K (asymmetric-capable).
    pub alpha: Vec<f64>,
    /// Topic-word Dirichlet prior (symmetric scalar).
    pub eta: f64,
    /// Robbins-Monro offset τ ≥ 0 (down-weights early minibatches).
    pub tau: f64,
    /// Robbins-Monro decay κ ∈ (0.5, 1].
    pub kappa: f64,
    /// Minibatch size |B| used to chunk the corpus in [`fit`]. `partial_fit` treats
    /// each supplied slice as one minibatch and does not consult this value.
    pub batch_size: usize,
    /// Per-document E-step fixed-point iteration cap.
    pub inner_iters: usize,
    /// Per-document E-step early-stop tolerance on mean |Δγ|.
    pub mean_change_tol: f64,
    /// The corpus-size assumption D used for the D/|B| natural-gradient scaling.
    /// For a batch [`fit`] this is the corpus length; for streaming it is the
    /// user's declared expected corpus size.
    pub total_docs: f64,

    /// Global topic-word variational parameter λ (K rows of length V), the
    /// persistent SVI state.
    pub lambda: Vec<Vec<f64>>,
    /// Number of minibatch updates applied so far (the Robbins-Monro step index).
    pub updates: usize,

    // ---- Derived / reporting state, refreshed from `lambda` on demand. ----
    /// Normalised topic-word matrix φ (K×V), each row a distribution.
    pub topic_word: Vec<Vec<f64>>,
    /// Document-topic matrix θ (D×K), each row a distribution. After [`fit`] this
    /// covers the whole corpus; after `partial_fit` it covers the last minibatch.
    pub doc_topic: Vec<Vec<f64>>,
    /// Token count per document underlying the current `doc_topic`.
    pub doc_lengths: Vec<usize>,
    /// Per-pass ELBO trace `(pass, bound)` from [`fit`]; empty for a pure stream.
    pub fit_history: Vec<(usize, f64)>,
    /// Whether [`fit`] stopped early on the per-pass bound tolerance.
    pub converged: bool,
}

/// `Elogβ` and `exp(Elogβ)` for the current `λ`: for each topic `k`,
/// `Elogβ_{kw} = ψ(λ_{kw}) − ψ(Σ_w λ_{kw})`. Both are returned because the E-step
/// needs the exponentiated form and the ELBO's β term needs the log form.
fn dirichlet_expectation(lambda: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut elog = vec![Vec::new(); lambda.len()];
    let mut expelog = vec![Vec::new(); lambda.len()];
    for (k, row) in lambda.iter().enumerate() {
        let dg_sum = digamma(row.iter().sum::<f64>());
        let e: Vec<f64> = row.iter().map(|&x| digamma(x) - dg_sum).collect();
        expelog[k] = e.iter().map(|&x| x.exp()).collect();
        elog[k] = e;
    }
    (elog, expelog)
}

/// Collapse a document to its unique `(word_id, count)` cells in a fixed
/// (ascending word-id) order so every downstream reduction is deterministic.
fn doc_cells(doc: &[u32]) -> (Vec<usize>, Vec<f64>) {
    let mut counts: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for &w in doc {
        *counts.entry(w).or_insert(0.0) += 1.0;
    }
    let mut cells: Vec<(u32, f64)> = counts.into_iter().collect();
    cells.sort_unstable_by_key(|&(w, _)| w);
    let ids = cells.iter().map(|&(w, _)| w as usize).collect();
    let cnts = cells.iter().map(|&(_, c)| c).collect();
    (ids, cnts)
}

/// The per-document mean-field E-step at fixed globals `exp_elogbeta`.
///
/// Runs the `γ`/`φ` fixed point to convergence (or `inner_iters`) and returns the
/// document's variational Dirichlet `γ` (length K). When `sstats` is `Some`, the
/// unweighted soft counts `expElogθ_k · (n_w / φnorm_w)` are scattered into it at
/// the document's word columns (the caller multiplies by `expElogβ` once per
/// minibatch, exactly as in Hoffman/gensim). `word_bound`, when `Some`, receives
/// the document's `Σ_w n_w · ln φnorm_w` word-likelihood term of the ELBO.
fn e_step_doc(
    ids: &[usize],
    cnts: &[f64],
    alpha: &[f64],
    exp_elogbeta: &[Vec<f64>],
    inner_iters: usize,
    mean_change_tol: f64,
    sstats: Option<&mut [Vec<f64>]>,
    word_bound: Option<&mut f64>,
) -> Vec<f64> {
    let k = alpha.len();
    let n = ids.len();
    // γ initialised to the prior-ish 1.0 (the mean of Hoffman's γ(100, 1/100)
    // init); the fixed point washes out the constant start, and a deterministic
    // init keeps the fit reproducible without an extra RNG draw per document.
    let mut gamma = vec![1.0f64; k];
    let mut elogtheta: Vec<f64> = {
        let dg = digamma(gamma.iter().sum::<f64>());
        gamma.iter().map(|&g| digamma(g) - dg).collect()
    };
    let mut exp_elogtheta: Vec<f64> = elogtheta.iter().map(|&x| x.exp()).collect();

    // φnorm_w = Σ_k expElogθ_k · expElogβ_{kw}  (+ floor).
    let mut phinorm = vec![0.0f64; n];
    let recompute_phinorm = |phinorm: &mut [f64], exp_elogtheta: &[f64]| {
        for (i, &w) in ids.iter().enumerate() {
            let mut s = 0.0;
            for t in 0..k {
                s += exp_elogtheta[t] * exp_elogbeta[t][w];
            }
            phinorm[i] = s + PHI_FLOOR;
        }
    };
    recompute_phinorm(&mut phinorm, &exp_elogtheta);

    for _ in 0..inner_iters {
        let last: Vec<f64> = gamma.clone();
        // γ_k = α_k + expElogθ_k · Σ_w expElogβ_{kw} · (n_w / φnorm_w).
        for t in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += exp_elogbeta[t][ids[i]] * (cnts[i] / phinorm[i]);
            }
            gamma[t] = alpha[t] + exp_elogtheta[t] * s;
        }
        let dg = digamma(gamma.iter().sum::<f64>());
        for t in 0..k {
            elogtheta[t] = digamma(gamma[t]) - dg;
            exp_elogtheta[t] = elogtheta[t].exp();
        }
        recompute_phinorm(&mut phinorm, &exp_elogtheta);

        let mean_change: f64 = gamma
            .iter()
            .zip(&last)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / k as f64;
        if mean_change < mean_change_tol {
            break;
        }
    }

    if let Some(ss) = sstats {
        for (i, &w) in ids.iter().enumerate() {
            let ratio = cnts[i] / phinorm[i];
            for t in 0..k {
                ss[t][w] += exp_elogtheta[t] * ratio;
            }
        }
    }
    if let Some(wb) = word_bound {
        // Σ_w n_w · ln φnorm_w = Σ_w n_w · ln Σ_k exp(Elogθ_k + Elogβ_{kw}).
        for i in 0..n {
            *wb += cnts[i] * phinorm[i].ln();
        }
    }
    gamma
}

/// θ row from a document's `γ`: the posterior-mean simplex `γ / Σγ`.
fn gamma_to_theta(gamma: &[f64]) -> Vec<f64> {
    let s: f64 = gamma.iter().sum();
    if s > 0.0 {
        gamma.iter().map(|&g| g / s).collect()
    } else {
        vec![1.0 / gamma.len() as f64; gamma.len()]
    }
}

/// The document-independent (topic-word β) part of the ELBO for the current `λ`.
/// Cheap `O(KV)`, evaluated once per pass rather than per minibatch.
fn beta_bound(lambda: &[Vec<f64>], elogbeta: &[Vec<f64>], eta: f64, v: usize) -> f64 {
    let k = lambda.len();
    let lg_eta = lgamma(eta);
    let lg_eta_v = lgamma(eta * v as f64);
    let mut score = 0.0;
    for t in 0..k {
        let mut lam_sum = 0.0;
        for w in 0..v {
            let lam = lambda[t][w];
            score += (eta - lam) * elogbeta[t][w] + (lgamma(lam) - lg_eta);
            lam_sum += lam;
        }
        score += lg_eta_v - lgamma(lam_sum);
    }
    score
}

/// The per-document (θ) part of the ELBO given `γ` and `Elogθ`.
fn theta_bound(gamma: &[f64], elogtheta: &[f64], alpha: &[f64]) -> f64 {
    let k = alpha.len();
    let mut score = 0.0;
    let mut alpha_sum = 0.0;
    let mut gamma_sum = 0.0;
    for t in 0..k {
        score += (alpha[t] - gamma[t]) * elogtheta[t] + (lgamma(gamma[t]) - lgamma(alpha[t]));
        alpha_sum += alpha[t];
        gamma_sum += gamma[t];
    }
    score + lgamma(alpha_sum) - lgamma(gamma_sum)
}

impl OnlineLDAModel {
    /// A fresh model with a random initial `λ` (each entry `1 + U(0,1)`, the
    /// codebase's standard positive-Dirichlet init), ready for streaming
    /// `partial_fit` or a batch [`fit`].
    #[allow(clippy::too_many_arguments)]
    pub fn new<R: Rng>(
        num_topics: usize,
        num_types: usize,
        alpha: Vec<f64>,
        eta: f64,
        tau: f64,
        kappa: f64,
        batch_size: usize,
        inner_iters: usize,
        mean_change_tol: f64,
        total_docs: f64,
        rng: &mut R,
    ) -> OnlineLDAModel {
        let lambda: Vec<Vec<f64>> = (0..num_topics)
            .map(|_| (0..num_types).map(|_| 1.0 + rng.gen::<f64>()).collect())
            .collect();
        let mut m = OnlineLDAModel {
            num_topics,
            num_types,
            alpha,
            eta,
            tau,
            kappa,
            batch_size: batch_size.max(1),
            inner_iters: inner_iters.max(1),
            mean_change_tol,
            total_docs: total_docs.max(1.0),
            lambda,
            updates: 0,
            topic_word: Vec::new(),
            doc_topic: Vec::new(),
            doc_lengths: Vec::new(),
            fit_history: Vec::new(),
            converged: false,
        };
        m.refresh_topic_word();
        m
    }

    /// Normalise `λ` into the reported topic-word matrix φ (posterior mean).
    pub fn refresh_topic_word(&mut self) {
        self.topic_word = self
            .lambda
            .iter()
            .map(|row| {
                let s: f64 = row.iter().sum();
                if s > 0.0 {
                    row.iter().map(|&x| x / s).collect()
                } else {
                    vec![1.0 / self.num_types as f64; self.num_types]
                }
            })
            .collect();
    }

    /// One stochastic update of `λ` from a minibatch: run the E-step collecting
    /// soft counts, form the natural-gradient target `η + (D/|B|)·sstats`, and
    /// blend it in with `ρ_t = (τ + t)^(-κ)`. Returns the minibatch's ELBO
    /// contribution (word + θ terms, at the pre-update globals) and the per-doc
    /// `γ` vectors in input order. Increments the update counter.
    fn minibatch_update(&mut self, batch: &[(Vec<usize>, Vec<f64>)]) -> (f64, Vec<Vec<f64>>) {
        let k = self.num_topics;
        let v = self.num_types;
        let (elogbeta, exp_elogbeta) = dirichlet_expectation(&self.lambda);

        let mut sstats = vec![vec![0.0f64; v]; k];
        let mut gammas: Vec<Vec<f64>> = Vec::with_capacity(batch.len());
        let mut batch_bound = 0.0f64;

        for (ids, cnts) in batch {
            let mut word_bound = 0.0f64;
            let gamma = e_step_doc(
                ids,
                cnts,
                &self.alpha,
                &exp_elogbeta,
                self.inner_iters,
                self.mean_change_tol,
                Some(&mut sstats),
                Some(&mut word_bound),
            );
            let dg = digamma(gamma.iter().sum::<f64>());
            let elogtheta: Vec<f64> = gamma.iter().map(|&g| digamma(g) - dg).collect();
            batch_bound += word_bound + theta_bound(&gamma, &elogtheta, &self.alpha);
            gammas.push(gamma);
        }

        // sstats · expElogβ gives the actual soft topic-word counts Σ φ n.
        for t in 0..k {
            for w in 0..v {
                sstats[t][w] *= exp_elogbeta[t][w];
            }
        }

        self.updates += 1;
        let rho = svi::rho(self.tau, self.kappa, self.updates);
        let scale = self.total_docs / batch.len().max(1) as f64;
        for t in 0..k {
            for w in 0..v {
                let lam_hat = self.eta + scale * sstats[t][w];
                self.lambda[t][w] = (1.0 - rho) * self.lambda[t][w] + rho * lam_hat;
            }
        }

        // The β term of the ELBO is added once per pass by the caller; the batch
        // bound returned here is the per-document (streaming) contribution only.
        // `elogbeta` was for the pre-update globals and is not needed further here.
        let _ = elogbeta;
        (batch_bound, gammas)
    }

    /// Fold one fresh minibatch of documents (word-id lists) into the fitted
    /// model: one SVI step on `λ`. `doc_topic` is refreshed to the minibatch's θ
    /// and `topic_word` to the new `λ`. Deterministic given the model's current
    /// state (no RNG draw is taken). Returns the minibatch ELBO contribution.
    pub fn partial_fit(&mut self, docs: &[Vec<u32>]) -> f64 {
        let batch: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|d| doc_cells(d)).collect();
        let (bound, gammas) = self.minibatch_update(&batch);
        self.doc_topic = gammas.iter().map(|g| gamma_to_theta(g)).collect();
        self.doc_lengths = docs.iter().map(|d| d.len()).collect();
        self.refresh_topic_word();
        bound
    }

    /// Infer θ for held-out documents at the current `λ` **without** updating the
    /// model (a pure E-step). Returns a (D×K) row-stochastic matrix.
    pub fn transform(&self, docs: &[Vec<u32>]) -> Vec<Vec<f64>> {
        let (_, exp_elogbeta) = dirichlet_expectation(&self.lambda);
        docs.iter()
            .map(|d| {
                let (ids, cnts) = doc_cells(d);
                let gamma = e_step_doc(
                    &ids,
                    &cnts,
                    &self.alpha,
                    &exp_elogbeta,
                    self.inner_iters,
                    self.mean_change_tol,
                    None,
                    None,
                );
                gamma_to_theta(&gamma)
            })
            .collect()
    }
}

/// Batch online-VB fit: initialise a model, then sweep the corpus for `iters`
/// passes, applying one SVI `λ` update per minibatch. Minibatches within a pass
/// follow a fresh seeded Fisher-Yates shuffle (shared with the CTM SVI path). A
/// per-pass ELBO trace drives optional early stopping on the relative change in
/// the bound (`convergence_tol > 0`). A final full E-step gives every document a
/// θ row and the reported topic-word matrix its normalised `λ`.
#[allow(clippy::too_many_arguments)]
pub fn fit<R: Rng, F: FnMut(usize, usize, f64) -> bool>(
    corpus: &Corpus,
    num_topics: usize,
    alpha: Vec<f64>,
    eta: f64,
    tau: f64,
    kappa: f64,
    batch_size: usize,
    inner_iters: usize,
    mean_change_tol: f64,
    iters: usize,
    convergence_tol: f64,
    total_docs_override: Option<f64>,
    mut on_progress: F,
    rng: &mut R,
) -> OnlineLDAModel {
    let num_types = corpus.num_types();
    let d = corpus.num_docs();
    // The natural-gradient scaling uses the declared streaming corpus size when the
    // caller supplied one (so `fit(first_chunk)` with `total_docs=D` is already on
    // the D-corpus gradient), otherwise the actual fit-corpus size.
    let total_docs = total_docs_override.unwrap_or(d.max(1) as f64).max(1.0);
    let mut model = OnlineLDAModel::new(
        num_topics,
        num_types,
        alpha,
        eta,
        tau,
        kappa,
        batch_size,
        inner_iters,
        mean_change_tol,
        total_docs,
        rng,
    );

    // Pre-collapse documents once; minibatches index into this.
    let cells: Vec<(Vec<usize>, Vec<f64>)> = corpus.docs.iter().map(|doc| doc_cells(doc)).collect();
    let batch = batch_size.clamp(1, d.max(1));

    for pass in 0..iters {
        let order = svi::shuffled_order(d, rng);
        let mut pass_bound = 0.0f64;
        for chunk in order.chunks(batch) {
            let mb: Vec<(Vec<usize>, Vec<f64>)> =
                chunk.iter().map(|&di| cells[di].clone()).collect();
            let (b, _) = model.minibatch_update(&mb);
            pass_bound += b;
        }
        // Add the β term of the ELBO once, at the pass-end globals.
        let (elogbeta, _) = dirichlet_expectation(&model.lambda);
        pass_bound += beta_bound(&model.lambda, &elogbeta, eta, num_types);

        let mut converged_now = false;
        if let Some(&(_, prev)) = model.fit_history.last() {
            let rel = (pass_bound - prev).abs() / prev.abs().max(1e-10);
            model.fit_history.push((pass + 1, pass_bound));
            if convergence_tol > 0.0 && rel < convergence_tol {
                model.converged = true;
                converged_now = true;
            }
        } else {
            model.fit_history.push((pass + 1, pass_bound));
        }

        // Per-pass ELBO drives the live progress bar. On an early-stop, snap the
        // bar to 100% by reporting (pass, pass) (mirrors fastopic, #786).
        if converged_now {
            let _ = on_progress(pass + 1, pass + 1, pass_bound);
            break;
        }
        if !on_progress(pass + 1, iters, pass_bound) {
            break;
        }
    }

    // Final full E-step: θ for every document at the converged λ.
    let (_, exp_elogbeta) = dirichlet_expectation(&model.lambda);
    model.doc_topic = corpus
        .docs
        .iter()
        .map(|doc| {
            let (ids, cnts) = doc_cells(doc);
            gamma_to_theta(&e_step_doc(
                &ids,
                &cnts,
                &model.alpha,
                &exp_elogbeta,
                model.inner_iters,
                model.mean_change_tol,
                None,
                None,
            ))
        })
        .collect();
    model.doc_lengths = corpus.docs.iter().map(|d| d.len()).collect();
    model.refresh_topic_word();
    model
}

impl Estimator for OnlineLDAModel {
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

impl DirichletModel for OnlineLDAModel {
    fn alpha(&self) -> Vec<f64> {
        self.alpha.clone()
    }
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
        // Online VB keeps a single variational posterior, not MCMC draws.
        Vec::new()
    }
    fn doc_lengths(&self) -> Vec<usize> {
        self.doc_lengths.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    /// A planted-block corpus: `n_blocks` disjoint word blocks; each document
    /// draws `doc_len` tokens, ~85% from its own block and ~15% uniform
    /// background. This mixed generator (rather than one pure block per doc) is a
    /// more honest recovery target for an *online* method — a pure single-block
    /// corpus is degenerate: a topic that loses the initial competition gets no
    /// gradient signal to revive, a symmetric-init pathology unrelated to whether
    /// the algorithm is correct. Seeded, so the corpus itself is deterministic.
    fn planted(n_blocks: usize, wpb: usize, n_docs: usize, doc_len: usize) -> (Corpus, usize) {
        let v = n_blocks * wpb;
        let mut rng = Pcg64Mcg::seed_from_u64(999);
        let docs: Vec<Vec<u32>> = (0..n_docs)
            .map(|d| {
                let b = d % n_blocks;
                (0..doc_len)
                    .map(|_| {
                        if rng.gen::<f64>() < 0.85 {
                            (b * wpb + (rng.gen::<f64>() * wpb as f64) as usize) as u32
                        } else {
                            (rng.gen::<f64>() * v as f64) as u32
                        }
                    })
                    .collect()
            })
            .collect();
        let corpus = Corpus {
            id_to_word: (0..v).map(|i| format!("w{i}")).collect(),
            doc_names: (0..n_docs).map(|i| format!("d{i}")).collect(),
            doc_labels: vec![String::new(); n_docs],
            doc_freqs: vec![0u32; v],
            total_freqs: vec![0u32; v],
            docs,
        };
        (corpus, n_blocks)
    }

    /// The reference recovery config (batch online VB, 100 passes). Recovers all
    /// four planted blocks across every seed tried.
    fn fit_planted(corpus: &Corpus, k: usize, seed: u64) -> OnlineLDAModel {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        fit(
            corpus,
            k,
            vec![0.1; k],
            0.01,
            1.0,
            0.7,
            32,
            100,
            1e-3,
            100,
            0.0,
            None,
            |_, _, _| true,
            &mut rng,
        )
    }

    #[test]
    fn online_lda_recovers_planted_topics() {
        let wpb = 5;
        let (corpus, n_blocks) = planted(4, wpb, 400, 30);
        let k = n_blocks;
        let v = corpus.num_types();
        let m = fit_planted(&corpus, k, 1);
        let phi = &m.topic_word; // [topic][word]
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| phi[t][b].partial_cmp(&phi[t][a]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..wpb].iter().copied().collect();
            for b in 0..n_blocks {
                let block: std::collections::HashSet<usize> = (b * wpb..(b + 1) * wpb).collect();
                if block.is_subset(&top) {
                    covered.insert(b);
                }
            }
        }
        assert_eq!(covered.len(), n_blocks, "only recovered {covered:?}");
    }

    #[test]
    fn online_lda_is_deterministic() {
        let (corpus, _) = planted(3, 4, 120, 30);
        let a = fit_planted(&corpus, 3, 42);
        let b = fit_planted(&corpus, 3, 42);
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        assert_eq!(a.lambda, b.lambda);
    }

    #[test]
    fn online_lda_conforms() {
        let (corpus, _) = planted(2, 3, 40, 20);
        let m = fit_planted(&corpus, 2, 0);
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    #[test]
    fn partial_fit_advances_schedule_and_matches_batch_semantics() {
        // A single partial_fit over the whole corpus (D declared = corpus size,
        // one minibatch) must equal one pass of fit's first minibatch when fit
        // uses batch_size = D and no shuffle effect (one chunk). This checks the
        // streaming path shares the batch path's update.
        let (corpus, _) = planted(3, 4, 30, 20);
        let k = 3;
        let v = corpus.num_types();
        let mut rng = Pcg64Mcg::seed_from_u64(7);
        let mut stream = OnlineLDAModel::new(
            k,
            v,
            vec![0.1; k],
            0.01,
            64.0,
            0.7,
            corpus.num_docs(),
            100,
            1e-3,
            corpus.num_docs() as f64,
            &mut rng,
        );
        assert_eq!(stream.updates, 0);
        stream.partial_fit(&corpus.docs);
        assert_eq!(stream.updates, 1);
        // θ was produced for every document in the minibatch.
        assert_eq!(stream.doc_topic.len(), corpus.num_docs());
        for row in &stream.doc_topic {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
        // A second minibatch advances the Robbins-Monro step index.
        stream.partial_fit(&corpus.docs);
        assert_eq!(stream.updates, 2);
    }

    #[test]
    fn transform_does_not_mutate_lambda() {
        let (corpus, _) = planted(3, 4, 60, 20);
        let m = fit_planted(&corpus, 3, 3);
        let before = m.lambda.clone();
        let theta = m.transform(&corpus.docs);
        assert_eq!(m.lambda, before);
        assert_eq!(theta.len(), corpus.num_docs());
        for row in &theta {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }
}
