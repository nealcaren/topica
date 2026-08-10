//! GaussianLDA: Gaussian LDA (Das, Zaheer & Dyer, "Gaussian LDA for Topic Models
//! with Word Embeddings", ACL 2015). Each topic is a Gaussian over the word-embedding
//! space with a Normal-Inverse-Wishart conjugate prior; a token is generated from its
//! topic's Gaussian on the word's embedding. Inference is collapsed Gibbs with the
//! multivariate Student-t posterior predictive, using rank-1 Cholesky up/downdates of
//! the per-topic NIW scale matrix as tokens move between topics.
//!
//! Ported from the authors' Apache-2.0 `rajarshd/Gaussian_LDA`
//! (`sampler/GaussianLDA.java`, plain-Cholesky sampler). No PyO3 here.

use crate::estimator::{Estimator, ModelFamily};
use rand::Rng;

/// Per-topic Normal-Inverse-Wishart running state.
///
/// `chol` is the lower-triangular Cholesky factor L of the NIW *scale* matrix Psi_k
/// (so L Lᵀ = Psi_k, NOT the topic covariance — see the ACL paper Eq. 2). `mean` is
/// mu_k; `count` is N_k. `half_log_det` caches `sum_l ln L_ll + (D/2) ln s`, the
/// bracketed determinant term of the Student-t log density, where
/// `s = (kappa_k+1)/(kappa_k (nu_k - D + 1))`.
struct Topic {
    count: usize,
    mean: Vec<f64>, // E
    chol: Vec<f64>, // E*E, row-major, lower-triangular (upper entries stay 0)
    half_log_det: f64,
}

/// Fitted state read back by the PyO3 binding.
pub struct GaussianLDAModel {
    pub num_topics: usize,
    pub embedding_dim: usize,
    pub topic_word: Vec<Vec<f64>>, // K x V, derived (softmax of the Student-t density)
    pub word_log_density: Vec<Vec<f64>>, // K x V, raw log Student-t density per vocab word
    pub doc_topic: Vec<Vec<f64>>,  // D x K, each row sums to 1
    pub topic_means: Vec<Vec<f64>>, // K x E (mu_k)
    pub topic_scale_chol: Vec<Vec<f64>>, // K rows of E*E: chol(Psi_k)
    pub topic_counts: Vec<usize>,  // N_k
    pub kappa0: f64,
    pub nu0: f64,
    pub fit_history: Vec<(usize, f64)>, // (iter, avgLL); iter 0 = post-init
    pub converged: bool,
}

#[inline]
fn at(l: &[f64], e: usize, i: usize, j: usize) -> f64 {
    l[i * e + j]
}

/// Rank-1 Cholesky update: L' = chol(L Lᵀ + x xᵀ), in place; `x` is consumed.
/// (Golub & Van Loan; matches Util.cholRank1Update.)
fn chol_rank1_update(l: &mut [f64], x: &mut [f64], e: usize) {
    for k in 0..e {
        let lkk = l[k * e + k];
        let r = (lkk * lkk + x[k] * x[k]).sqrt();
        let c = r / lkk;
        let s = x[k] / lkk;
        l[k * e + k] = r;
        for row in (k + 1)..e {
            let nv = (l[row * e + k] + s * x[row]) / c;
            l[row * e + k] = nv;
            x[row] = c * x[row] - s * nv;
        }
    }
}

/// Rank-1 Cholesky downdate: L' = chol(L Lᵀ - x xᵀ), in place; `x` is consumed.
/// Returns `false` if the result is not positive-definite (a diagonal would go
/// imaginary), so the caller can rebuild from the batch scatter. The inner `x`
/// recurrence keeps the MINUS sign, exactly as the update (matches
/// Util.cholRank1Downdate — a plus here silently corrupts the factor).
#[must_use]
fn chol_rank1_downdate(l: &mut [f64], x: &mut [f64], e: usize) -> bool {
    // Probe positive-definiteness before mutating so a failure leaves `l` untouched.
    {
        let mut probe: Vec<f64> = x.to_vec();
        for k in 0..e {
            let lkk = l[k * e + k];
            let diff = lkk * lkk - probe[k] * probe[k];
            if diff.is_nan() || diff <= 1e-12 {
                return false;
            }
            let r = diff.sqrt();
            let c = r / lkk;
            let s = probe[k] / lkk;
            for row in (k + 1)..e {
                let nv = (l[row * e + k] - s * probe[row]) / c;
                probe[row] = c * probe[row] - s * nv;
            }
        }
    }
    for k in 0..e {
        let lkk = l[k * e + k];
        let r = (lkk * lkk - x[k] * x[k]).sqrt();
        let c = r / lkk;
        let s = x[k] / lkk;
        l[k * e + k] = r;
        for row in (k + 1)..e {
            let nv = (l[row * e + k] - s * x[row]) / c;
            l[row * e + k] = nv;
            x[row] = c * x[row] - s * nv;
        }
    }
    true
}

/// Forward substitution: solve L y = d for lower-triangular L (E x E), returning y.
fn solve_l(l: &[f64], d: &[f64], e: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; e];
    for i in 0..e {
        let mut acc = d[i];
        for j in 0..i {
            acc -= at(l, e, i, j) * y[j];
        }
        y[i] = acc / at(l, e, i, i);
    }
    y
}

struct Prior {
    mu0: Vec<f64>,
    kappa0: f64,
    nu0: f64,
    e: usize,
    chol_psi0: Vec<f64>, // chol(Psi0) = sqrt(psi_scale*D) * I
}

impl Prior {
    /// half_log_det of an empty topic (state == prior).
    fn empty_half_log_det(&self) -> f64 {
        let d = self.e as f64;
        let s = (self.kappa0 + 1.0) / (self.kappa0 * (self.nu0 - d + 1.0));
        let mut ld = 0.0;
        for l in 0..self.e {
            ld += self.chol_psi0[l * self.e + l].ln();
        }
        ld + 0.5 * d * s.ln()
    }

    fn new_topic(&self) -> Topic {
        Topic {
            count: 0,
            mean: self.mu0.clone(),
            chol: self.chol_psi0.clone(),
            half_log_det: self.empty_half_log_det(),
        }
    }
}

/// Recompute a topic's cached half_log_det from its current chol and count.
fn refresh_half_log_det(t: &mut Topic, prior: &Prior) {
    let d = prior.e as f64;
    let k_n = prior.kappa0 + t.count as f64;
    let nu_n = prior.nu0 + t.count as f64;
    let s = (k_n + 1.0) / (k_n * (nu_n - d + 1.0));
    let mut ld = 0.0;
    for l in 0..prior.e {
        ld += t.chol[l * prior.e + l].ln();
    }
    t.half_log_det = ld + 0.5 * d * s.ln();
}

/// Add embedding `e_vec` to topic `t` (mean uses the NEW mean; matches Java add).
fn topic_add(t: &mut Topic, e_vec: &[f64], prior: &Prior) {
    t.count += 1;
    let k_n = prior.kappa0 + t.count as f64;
    let e = prior.e;
    // mu_new = ((k_n - 1) mu_old + e) / k_n
    for i in 0..e {
        t.mean[i] = ((k_n - 1.0) * t.mean[i] + e_vec[i]) / k_n;
    }
    let coeff = (k_n / (k_n - 1.0)).sqrt();
    let mut x: Vec<f64> = (0..e).map(|i| coeff * (e_vec[i] - t.mean[i])).collect();
    chol_rank1_update(&mut t.chol, &mut x, e);
    refresh_half_log_det(t, prior);
}

/// Remove embedding `e_vec` from topic `t` (downdate uses the OLD mean, then updates
/// the mean; matches Java remove). `rebuild` supplies the batch scatter recompute if
/// the downdate loses positive-definiteness.
fn topic_remove(t: &mut Topic, e_vec: &[f64], prior: &Prior, rebuild: impl FnOnce() -> Vec<f64>) {
    debug_assert!(t.count > 0);
    t.count -= 1;
    let k_n = prior.kappa0 + t.count as f64;
    let e = prior.e;
    let coeff = ((k_n + 1.0) / k_n).sqrt();
    let mut x: Vec<f64> = (0..e).map(|i| coeff * (e_vec[i] - t.mean[i])).collect();
    let ok = chol_rank1_downdate(&mut t.chol, &mut x, e);
    // mu_new = ((k_n + 1) mu_old - e) / k_n
    for i in 0..e {
        t.mean[i] = ((k_n + 1.0) * t.mean[i] - e_vec[i]) / k_n;
    }
    if !ok {
        t.chol = rebuild();
    }
    refresh_half_log_det(t, prior);
}

/// Multivariate Student-t log density of embedding `e_vec` under topic `t`
/// (the collapsed-Gibbs likelihood term). Port of `logMultivariateTDensity`, with
/// `0.5 * D` in the normalizer (fixing Java's integer `Data.D/2` for odd D).
fn log_t_density(t: &Topic, e_vec: &[f64], prior: &Prior) -> f64 {
    let d = prior.e as f64;
    let e = prior.e;
    let k_n = prior.kappa0 + t.count as f64;
    let nu_n = prior.nu0 + t.count as f64;
    let nu = prior.nu0 + t.count as f64 - d + 1.0; // t dof
    let sqrt_s = ((k_n + 1.0) / (k_n * (nu_n - d + 1.0))).sqrt();
    // Solve (sqrt_s L) y = (e - mu): y = solve_l(L, e-mu) / sqrt_s; val = ||y||^2.
    let diff: Vec<f64> = (0..e).map(|i| e_vec[i] - t.mean[i]).collect();
    let z = solve_l(&t.chol, &diff, e);
    let val: f64 = z.iter().map(|zi| zi * zi).sum::<f64>() / (sqrt_s * sqrt_s);
    crate::mathfun::log_gamma((nu + d) / 2.0)
        - (crate::mathfun::log_gamma(nu / 2.0)
            + 0.5 * d * (nu.ln() + std::f64::consts::PI.ln())
            + t.half_log_det
            + (nu + d) / 2.0 * (1.0 + val / nu).ln())
}

/// Reference `avgLL` diagnostic (Util.calculateAvgLL): mean over tokens of the
/// point-estimate multivariate-Normal log-density of each token's embedding at its
/// CURRENT topic, covariance Psi_k/(nu_0+N_k-D). NOT the model evidence (drops the
/// Dirichlet term); not guaranteed monotone. Reproduced exactly for parity.
fn avg_ll(
    topics: &[Topic],
    token_emb: &[usize],
    token_topic: &[usize],
    embeddings: &[Vec<f64>],
    prior: &Prior,
) -> f64 {
    let e = prior.e;
    let d = e as f64;
    // Per-topic scaled-cholesky log-determinant: sum_l ln(L_ll / sqrt(scalar)).
    let mut scalar = vec![0.0f64; topics.len()];
    let mut log_det = vec![0.0f64; topics.len()];
    for (k, t) in topics.iter().enumerate() {
        scalar[k] = prior.nu0 + t.count as f64 - d;
        let sq = scalar[k].sqrt();
        let mut ld = 0.0;
        for l in 0..e {
            ld += (t.chol[l * e + l] / sq).ln();
        }
        log_det[k] = ld;
    }
    let mut total = 0.0f64;
    for (&w, &k) in token_emb.iter().zip(token_topic.iter()) {
        let t = &topics[k];
        let diff: Vec<f64> = (0..e).map(|i| embeddings[w][i] - t.mean[i]).collect();
        // scaledChol = L / sqrt(scalar) => solving it scales the solution by sqrt(scalar).
        let z = solve_l(&t.chol, &diff, e);
        let val: f64 = z.iter().map(|zi| zi * zi).sum::<f64>() * scalar[k];
        let log_density = 0.5 * (val + d * (2.0 * std::f64::consts::PI).ln()) + log_det[k];
        total -= log_density;
    }
    total / token_emb.len() as f64
}

/// Rebuild chol(Psi_k) from the batch NIW sufficient statistics of a topic's current
/// members (PD-failure recovery), excluding token `exclude` (the one being removed):
///
///   Psi_k = Psi0 + sum_i (e_i - xbar)(e_i - xbar)ᵀ
///                + (kappa0 N / (kappa0 + N)) (xbar - mu0)(xbar - mu0)ᵀ
///
/// where xbar is the sample mean of the members and N their count. Returns the
/// lower-triangular Cholesky factor. Independent of the running state, so it is exact.
fn rebuild_chol(
    topic: usize,
    exclude: usize,
    token_emb: &[usize],
    token_topic: &[usize],
    embeddings: &[Vec<f64>],
    prior: &Prior,
) -> Vec<f64> {
    let e = prior.e;
    let members: Vec<usize> = (0..token_topic.len())
        .filter(|&t| t != exclude && token_topic[t] == topic)
        .map(|t| token_emb[t])
        .collect();
    let n = members.len();
    let mut psi = vec![0.0f64; e * e];
    for i in 0..e {
        let dii = prior.chol_psi0[i * e + i];
        psi[i * e + i] = dii * dii; // Psi0
    }
    if n == 0 {
        return cholesky_lower(&psi, e);
    }
    let mut xbar = vec![0.0f64; e];
    for &w in &members {
        for i in 0..e {
            xbar[i] += embeddings[w][i];
        }
    }
    for x in xbar.iter_mut() {
        *x /= n as f64;
    }
    for &w in &members {
        for i in 0..e {
            for j in 0..=i {
                psi[i * e + j] += (embeddings[w][i] - xbar[i]) * (embeddings[w][j] - xbar[j]);
            }
        }
    }
    let c = prior.kappa0 * n as f64 / (prior.kappa0 + n as f64);
    for i in 0..e {
        for j in 0..=i {
            psi[i * e + j] += c * (xbar[i] - prior.mu0[i]) * (xbar[j] - prior.mu0[j]);
        }
    }
    cholesky_lower(&psi, e)
}

/// Dense lower-triangular Cholesky of a symmetric PD matrix stored row-major
/// (reads the lower triangle).
fn cholesky_lower(a: &[f64], e: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; e * e];
    for i in 0..e {
        for j in 0..=i {
            let mut sum = a[i * e + j];
            for k in 0..j {
                sum -= l[i * e + k] * l[j * e + k];
            }
            if i == j {
                l[i * e + i] = sum.max(1e-12).sqrt();
            } else {
                l[i * e + j] = sum / l[j * e + j];
            }
        }
    }
    l
}

/// K-means (Lloyd, k-means++ seeding) over the V vocabulary embeddings, returning each
/// word's cluster in `0..k`. Deterministic from `rng`. Empty clusters are re-seeded from
/// the point farthest from its center so all k clusters are non-empty.
fn kmeans_words<R: Rng>(
    embeddings: &[Vec<f64>],
    k: usize,
    e: usize,
    rng: &mut R,
    iters: usize,
) -> Vec<usize> {
    let n = embeddings.len();
    if n <= k {
        return (0..n).map(|i| i % k).collect();
    }
    let sqdist =
        |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
    // k-means++ seeding
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = ((rng.gen::<f64>() * n as f64) as usize).min(n - 1);
    centers.push(embeddings[first].clone());
    while centers.len() < k {
        let d2: Vec<f64> = embeddings
            .iter()
            .map(|p| {
                centers
                    .iter()
                    .map(|c| sqdist(p, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let sum: f64 = d2.iter().sum();
        let mut target = rng.gen::<f64>() * sum;
        let mut chosen = n - 1;
        for (i, &w) in d2.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.push(embeddings[chosen].clone());
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        for (i, p) in embeddings.iter().enumerate() {
            let mut best = 0;
            let mut bd = f64::INFINITY;
            for (c, ctr) in centers.iter().enumerate() {
                let d = sqdist(p, ctr);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        // recompute centers
        let mut sums = vec![vec![0.0f64; e]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in embeddings.iter().enumerate() {
            counts[assign[i]] += 1;
            for d in 0..e {
                sums[assign[i]][d] += p[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..e {
                    centers[c][d] = sums[c][d] / counts[c] as f64;
                }
            }
        }
        // re-seed any empty cluster from the point farthest from its own center
        for c in 0..k {
            if counts[c] == 0 {
                let mut worst = 0;
                let mut wd = -1.0;
                for (i, p) in embeddings.iter().enumerate() {
                    let d = sqdist(p, &centers[assign[i]]);
                    if d > wd {
                        wd = d;
                        worst = i;
                    }
                }
                centers[c] = embeddings[worst].clone();
                assign[worst] = c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    assign
}

/// Fit Gaussian LDA. `docs` are per-document in-vocabulary word ids; `embeddings` is
/// the (V, E) type-level embedding matrix aligned to the vocabulary. `nu0` is clamped
/// to >= E. `use_kmeans` selects k-means initialization (the paper's default, which
/// avoids the mode-collapse of random init) vs. per-token random init (the reference
/// Cholesky sampler's behavior). Seeds every draw from `rng` for bit-for-bit
/// reproducibility.
#[allow(clippy::too_many_arguments)]
pub fn fit<R: Rng>(
    docs: &[Vec<u32>],
    embeddings: &[Vec<f64>],
    num_topics: usize,
    alpha: f64,
    kappa0: f64,
    nu0: f64,
    psi_scale: f64,
    iters: usize,
    use_kmeans: bool,
    rng: &mut R,
) -> GaussianLDAModel {
    let e = embeddings.first().map(|r| r.len()).unwrap_or(0);
    let v = embeddings.len();
    let d = e as f64;
    let nu0 = nu0.max(d); // clamp nu_0 >= D

    // mu0 = uniform mean of the V vocabulary embeddings (type-weighted, per Java).
    let mut mu0 = vec![0.0f64; e];
    for row in embeddings {
        for i in 0..e {
            mu0[i] += row[i];
        }
    }
    for m in mu0.iter_mut() {
        *m /= v.max(1) as f64;
    }
    // chol(Psi0) = sqrt(psi_scale * D) * I
    let mut chol_psi0 = vec![0.0f64; e * e];
    let diag = (psi_scale * d).sqrt();
    for i in 0..e {
        chol_psi0[i * e + i] = diag;
    }
    let prior = Prior {
        mu0,
        kappa0,
        nu0,
        e,
        chol_psi0,
    };

    // Flatten tokens in document-major order (fixed, for determinism).
    let mut token_emb: Vec<usize> = Vec::new(); // word id per token
    let mut token_doc: Vec<usize> = Vec::new();
    for (di, doc) in docs.iter().enumerate() {
        for &w in doc {
            token_emb.push(w as usize);
            token_doc.push(di);
        }
    }
    let n_tokens = token_emb.len();
    let n_docs = docs.len();

    // Doc-topic counts and topic states.
    let mut ndk = vec![0u32; n_docs * num_topics];
    let mut topics: Vec<Topic> = (0..num_topics).map(|_| prior.new_topic()).collect();
    let mut token_topic = vec![0usize; n_tokens];

    // Initialization: k-means over the vocabulary embeddings (paper default) assigns
    // each word a cluster and each token inherits its word's cluster; otherwise each
    // token draws a topic uniformly at random (the reference Cholesky sampler).
    let word_cluster: Option<Vec<usize>> = if use_kmeans {
        Some(kmeans_words(embeddings, num_topics, e, rng, 50))
    } else {
        None
    };
    for i in 0..n_tokens {
        let k = match &word_cluster {
            Some(wc) => wc[token_emb[i]],
            None => ((rng.gen::<f64>() * num_topics as f64) as usize).min(num_topics - 1),
        };
        token_topic[i] = k;
        ndk[token_doc[i] * num_topics + k] += 1;
        topic_add(&mut topics[k], &embeddings[token_emb[i]], &prior);
    }

    let mut fit_history: Vec<(usize, f64)> = Vec::with_capacity(iters + 1);
    fit_history.push((
        0,
        avg_ll(&topics, &token_emb, &token_topic, embeddings, &prior),
    ));

    let mut logpost = vec![0.0f64; num_topics];
    for it in 0..iters {
        for i in 0..n_tokens {
            let doc = token_doc[i];
            let w = token_emb[i];
            let old = token_topic[i];
            // remove from old topic
            ndk[doc * num_topics + old] -= 1;
            let emb_i = embeddings[w].clone();
            {
                // `token_topic[i]` still holds `old` here, so the rebuild scan excludes
                // token `i` explicitly. The closure borrows only the immutable token
                // arrays + embeddings, disjoint from the `&mut topics[old]` borrow.
                let te = &token_emb;
                let tt = &token_topic;
                topic_remove(&mut topics[old], &emb_i, &prior, || {
                    rebuild_chol(old, i, te, tt, embeddings, &prior)
                });
            }
            // score every topic
            let mut max = f64::NEG_INFINITY;
            for k in 0..num_topics {
                let prior_term = ((ndk[doc * num_topics + k] as f64) + alpha).ln();
                let lp = prior_term + log_t_density(&topics[k], &emb_i, &prior);
                logpost[k] = lp;
                if lp > max {
                    max = lp;
                }
            }
            // softmax (max-subtracted) and sample
            let mut sum = 0.0;
            for k in 0..num_topics {
                let p = (logpost[k] - max).exp();
                logpost[k] = p;
                sum += p;
            }
            let mut target = rng.gen::<f64>() * sum;
            let mut newk = num_topics - 1;
            for k in 0..num_topics {
                target -= logpost[k];
                if target <= 0.0 {
                    newk = k;
                    break;
                }
            }
            token_topic[i] = newk;
            ndk[doc * num_topics + newk] += 1;
            topic_add(&mut topics[newk], &emb_i, &prior);
        }
        fit_history.push((
            it + 1,
            avg_ll(&topics, &token_emb, &token_topic, embeddings, &prior),
        ));
    }

    // Derived topic_word: softmax over vocab of the Student-t density under each topic.
    // Also retain the raw per-word log densities (used by `transform`).
    let mut topic_word = vec![vec![0.0f64; v]; num_topics];
    let mut word_log_density = vec![vec![0.0f64; v]; num_topics];
    for k in 0..num_topics {
        let mut max = f64::NEG_INFINITY;
        for w in 0..v {
            let lw = log_t_density(&topics[k], &embeddings[w], &prior);
            word_log_density[k][w] = lw;
            if lw > max {
                max = lw;
            }
        }
        let mut sum = 0.0;
        for w in 0..v {
            sum += (word_log_density[k][w] - max).exp();
        }
        for w in 0..v {
            topic_word[k][w] = (word_log_density[k][w] - max).exp() / sum;
        }
    }

    // doc_topic: (n_dk + alpha) normalized.
    let mut doc_topic = vec![vec![0.0f64; num_topics]; n_docs];
    for (di, dt) in doc_topic.iter_mut().enumerate() {
        let mut s = 0.0;
        for k in 0..num_topics {
            let val = ndk[di * num_topics + k] as f64 + alpha;
            dt[k] = val;
            s += val;
        }
        if s > 0.0 {
            for x in dt.iter_mut() {
                *x /= s;
            }
        }
    }

    let topic_means: Vec<Vec<f64>> = topics.iter().map(|t| t.mean.clone()).collect();
    let topic_scale_chol: Vec<Vec<f64>> = topics.iter().map(|t| t.chol.clone()).collect();
    let topic_counts: Vec<usize> = topics.iter().map(|t| t.count).collect();

    GaussianLDAModel {
        num_topics,
        embedding_dim: e,
        topic_word,
        word_log_density,
        doc_topic,
        topic_means,
        topic_scale_chol,
        topic_counts,
        kappa0,
        nu0,
        fit_history,
        converged: true,
    }
}

/// Closed-vocabulary `transform`: infer topic proportions for new documents holding
/// the fitted topic Gaussians FIXED. Tokens are scored with the fitted per-word
/// Student-t log densities (`model.word_log_density`); only the new docs' topic counts
/// are sampled. `docs` are in-vocabulary word ids of the fitted vocabulary. Returns
/// (D_new, K).
pub fn transform<R: Rng>(
    model: &GaussianLDAModel,
    docs: &[Vec<u32>],
    alpha: f64,
    iters: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let k = model.num_topics;
    let mut token_word: Vec<usize> = Vec::new();
    let mut token_doc: Vec<usize> = Vec::new();
    for (di, doc) in docs.iter().enumerate() {
        for &w in doc {
            token_word.push(w as usize);
            token_doc.push(di);
        }
    }
    let n_docs = docs.len();
    let mut ndk = vec![0u32; n_docs * k];
    let mut token_topic = vec![0usize; token_word.len()];
    // per-token densities from the fixed fitted topics (lookup, no embeddings needed)
    let mut dens = vec![0.0f64; token_word.len() * k];
    for (i, &w) in token_word.iter().enumerate() {
        for t in 0..k {
            dens[i * k + t] = model.word_log_density[t][w];
        }
    }
    for i in 0..token_word.len() {
        let t = ((rng.gen::<f64>() * k as f64) as usize).min(k - 1);
        token_topic[i] = t;
        ndk[token_doc[i] * k + t] += 1;
    }
    let mut logpost = vec![0.0f64; k];
    for _ in 0..iters {
        for i in 0..token_word.len() {
            let doc = token_doc[i];
            let old = token_topic[i];
            ndk[doc * k + old] -= 1;
            let mut max = f64::NEG_INFINITY;
            for (t, lp) in logpost.iter_mut().enumerate() {
                *lp = ((ndk[doc * k + t] as f64) + alpha).ln() + dens[i * k + t];
                if *lp > max {
                    max = *lp;
                }
            }
            let mut sum = 0.0;
            for lp in logpost.iter_mut() {
                *lp = (*lp - max).exp();
                sum += *lp;
            }
            let mut target = rng.gen::<f64>() * sum;
            let mut newk = k - 1;
            for (t, &p) in logpost.iter().enumerate() {
                target -= p;
                if target <= 0.0 {
                    newk = t;
                    break;
                }
            }
            token_topic[i] = newk;
            ndk[doc * k + newk] += 1;
        }
    }
    (0..n_docs)
        .map(|di| {
            let mut row: Vec<f64> = (0..k).map(|t| ndk[di * k + t] as f64 + alpha).collect();
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
            row
        })
        .collect()
}

impl Estimator for GaussianLDAModel {
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
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a planted embedding corpus: `k` well-separated Gaussian clusters in E-dim
    /// space; each doc draws its tokens from one dominant cluster's words.
    fn planted(
        k: usize,
        e: usize,
        words_per_topic: usize,
        n_docs: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let v = k * words_per_topic;
        // cluster centers on scaled axes
        let mut centers = vec![vec![0.0f64; e]; k];
        for (t, c) in centers.iter_mut().enumerate() {
            c[t % e] = 8.0;
            c[(t + 1) % e] = -8.0;
        }
        let mut embeddings = vec![vec![0.0f64; e]; v];
        for w in 0..v {
            let t = w / words_per_topic;
            for i in 0..e {
                embeddings[w][i] = centers[t][i] + 0.4 * (rng.gen::<f64>() - 0.5);
            }
        }
        let mut docs = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..n_docs {
            let t = (rng.gen::<f64>() * k as f64) as usize % k;
            labels.push(t);
            let mut doc = Vec::new();
            for _ in 0..25 {
                let w = t * words_per_topic
                    + (rng.gen::<f64>() * words_per_topic as f64) as usize % words_per_topic;
                doc.push(w as u32);
            }
            docs.push(doc);
        }
        (docs, embeddings, labels)
    }

    #[test]
    fn gaussian_lda_recovers_planted_topics() {
        let (docs, emb, labels) = planted(3, 10, 15, 90, 7);
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let m = fit(
            &docs,
            &emb,
            3,
            1.0 / 3.0,
            0.1,
            10.0,
            3.0,
            60,
            true,
            &mut rng,
        );
        // Each doc's argmax topic should be consistent per planted label: build a
        // label->topic map and check purity is high.
        let dt = &m.doc_topic;
        let mut map = std::collections::HashMap::new();
        let mut correct = 0;
        for (di, row) in dt.iter().enumerate() {
            let arg = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let entry = map.entry(labels[di]).or_insert(arg);
            if *entry == arg {
                correct += 1;
            }
        }
        assert!(
            correct as f64 / dt.len() as f64 > 0.9,
            "planted purity too low: {}/{}",
            correct,
            dt.len()
        );
    }

    #[test]
    fn gaussian_lda_is_deterministic() {
        let (docs, emb, _) = planted(3, 8, 12, 40, 5);
        let mut r1 = ChaCha8Rng::seed_from_u64(13);
        let mut r2 = ChaCha8Rng::seed_from_u64(13);
        let m1 = fit(&docs, &emb, 3, 0.3, 0.1, 8.0, 3.0, 30, true, &mut r1);
        let m2 = fit(&docs, &emb, 3, 0.3, 0.1, 8.0, 3.0, 30, true, &mut r2);
        assert_eq!(m1.topic_word, m2.topic_word);
        assert_eq!(m1.doc_topic, m2.doc_topic);
        assert_eq!(m1.topic_means, m2.topic_means);
    }

    #[test]
    fn gaussian_lda_conforms() {
        let (docs, emb, _) = planted(2, 6, 10, 20, 1);
        let m = fit(
            &docs,
            &emb,
            2,
            0.5,
            0.1,
            6.0,
            3.0,
            15,
            true,
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }

    /// Gate A finding B5: the incremental per-topic state (mean + chol(Psi_k)) must
    /// equal the batch NIW sufficient statistics after a sequence of adds/removes.
    #[test]
    fn incremental_state_matches_batch() {
        let e = 5;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let pts: Vec<Vec<f64>> = (0..12)
            .map(|_| (0..e).map(|_| rng.gen::<f64>() * 4.0 - 2.0).collect())
            .collect();
        let mut mu0 = vec![0.0f64; e];
        for p in &pts {
            for i in 0..e {
                mu0[i] += p[i];
            }
        }
        for m in mu0.iter_mut() {
            *m /= pts.len() as f64;
        }
        let mut chol_psi0 = vec![0.0f64; e * e];
        for i in 0..e {
            chol_psi0[i * e + i] = (3.0 * e as f64).sqrt();
        }
        let prior = Prior {
            mu0: mu0.clone(),
            kappa0: 0.1,
            nu0: e as f64,
            e,
            chol_psi0: chol_psi0.clone(),
        };
        // add first 8 points incrementally
        let mut t = prior.new_topic();
        for p in pts.iter().take(8) {
            topic_add(&mut t, p, &prior);
        }
        // remove points 6,7 -> topic now holds points 0..6
        topic_remove(&mut t, &pts[7], &prior, Vec::new);
        topic_remove(&mut t, &pts[6], &prior, Vec::new);
        let members = &pts[0..6];
        // batch mean
        let mut bmean = vec![0.0f64; e];
        for p in members {
            for i in 0..e {
                bmean[i] += p[i];
            }
        }
        for m in bmean.iter_mut() {
            *m /= members.len() as f64;
        }
        let n = members.len() as f64;
        let k_n = prior.kappa0 + n;
        // mu_k = (kappa0 mu0 + n xbar) / k_n
        let mut mu_k = vec![0.0f64; e];
        for i in 0..e {
            mu_k[i] = (prior.kappa0 * mu0[i] + n * bmean[i]) / k_n;
        }
        for i in 0..e {
            assert!(
                (t.mean[i] - mu_k[i]).abs() < 1e-9,
                "mean mismatch dim {}",
                i
            );
        }
        // batch Psi = Psi0 + scatter about xbar + (kappa0 n / k_n)(xbar-mu0)(xbar-mu0)^T
        let mut psi = vec![0.0f64; e * e];
        for i in 0..e {
            psi[i * e + i] = 3.0 * e as f64;
        }
        for p in members {
            for i in 0..e {
                for j in 0..e {
                    psi[i * e + j] += (p[i] - bmean[i]) * (p[j] - bmean[j]);
                }
            }
        }
        let c = prior.kappa0 * n / k_n;
        for i in 0..e {
            for j in 0..e {
                psi[i * e + j] += c * (bmean[i] - mu0[i]) * (bmean[j] - mu0[j]);
            }
        }
        let batch_l = cholesky_lower(&psi, e);
        for i in 0..e {
            for j in 0..=i {
                assert!(
                    (t.chol[i * e + j] - batch_l[i * e + j]).abs() < 1e-7,
                    "chol mismatch ({},{}): {} vs {}",
                    i,
                    j,
                    t.chol[i * e + j],
                    batch_l[i * e + j]
                );
            }
        }
    }
}
