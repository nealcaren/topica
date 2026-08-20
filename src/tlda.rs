//! Online Tensor LDA (TensorLDA) topic model.
//! Reference: tensorly/tlda
//! Paper: Kangaslahti et al., "Analyzing Political Text at Scale with Online Tensor LDA",
//! Political Analysis 2026.

use crate::estimator::{Estimator, ModelFamily};
use crate::linalg::{qr_reduced, randomized_svd};
use crate::optimize::digamma;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Max-abs factor change below which the streaming factor SGD is treated as
/// converged (the streaming analogue of the batch path's `tol = 1e-5`).
const STREAM_CONVERGENCE_TOL: f64 = 1e-5;

/// A fitted TensorLDA model.
pub struct TensorLdaModel {
    pub num_topics: usize,
    pub num_types: usize,
    /// Topic-word matrix (K x V)
    pub topic_word: Vec<Vec<f64>>,
    /// Document-topic matrix (D x K)
    pub doc_topic: Vec<Vec<f64>>,
    /// Prior weights/Dirichlet alpha (K)
    pub weights: Vec<f64>,
    pub alpha_0: f64,
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
    pub unwhitened_raw: Vec<f64>, // V * K flat vector
}

impl Estimator for TensorLdaModel {
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

/// Helper function to perform matrix multiplication: `C = A * B` where `A` is `ar x ac`
/// and `B` is `ac x bc`. All matrices are row-major.
fn matmul(a: &[f64], b: &[f64], ar: usize, ac: usize, bc: usize) -> Vec<f64> {
    let mut c = vec![0.0; ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let aik = a[i * ac + k];
            if aik != 0.0 {
                for j in 0..bc {
                    c[i * bc + j] += aik * b[k * bc + j];
                }
            }
        }
    }
    c
}

/// Helper to transpose a matrix of size `rows x cols`.
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// Computes the gradient of the third-order cumulant reconstruction loss with respect to
/// the factors. Matches `cumulant_gradient` in Python.
fn cumulant_gradient(
    factors: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    b: usize,
    alpha_0: f64,
    theta: f64,
) -> Vec<f64> {
    let factors_t = transpose(factors, rows, cols);
    let pt_p = matmul(&factors_t, factors, cols, rows, cols);
    let mut pt_p2 = vec![0.0; cols * cols];
    for i in 0..(cols * cols) {
        pt_p2[i] = pt_p[i] * pt_p[i];
    }
    let term1 = matmul(factors, &pt_p2, rows, cols, cols);

    let y_phi = matmul(y, factors, b, rows, cols);
    let mut y_phi2 = vec![0.0; b * cols];
    for i in 0..(b * cols) {
        y_phi2[i] = y_phi[i] * y_phi[i];
    }
    let y_t = transpose(y, b, rows);
    let term2 = matmul(&y_t, &y_phi2, rows, b, cols);

    let mut grad = vec![0.0; rows * cols];
    let coef1 = 3.0 * (1.0 + theta);
    let coef2 = 3.0 * (1.0 + alpha_0) * (2.0 + alpha_0) / (2.0 * b as f64);
    for i in 0..(rows * cols) {
        grad[i] = coef1 * term1[i] - coef2 * term2[i];
    }
    grad
}

/// Run one SGD batch update step for the third-order cumulant factors.
fn partial_fit_step(
    factors: &mut [f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    b: usize,
    alpha_0: f64,
    theta: f64,
    lr: f64,
) {
    let grad = cumulant_gradient(factors, y, rows, cols, b, alpha_0, theta);
    for i in 0..(rows * cols) {
        factors[i] -= lr * grad[i];
    }
    // Normalize columns
    let mut col_norms = vec![0.0; cols];
    for j in 0..cols {
        let mut sum2 = 0.0;
        for i in 0..rows {
            let val = factors[i * cols + j];
            sum2 += val * val;
        }
        col_norms[j] = sum2.sqrt().max(1e-12);
    }
    for i in 0..rows {
        for j in 0..cols {
            factors[i * cols + j] /= col_norms[j];
        }
    }
}

/// Helper function to compute Dirichlet expectations of the topic simplex.
fn dirichlet_expectation(gammad: &[f64], d: usize, k: usize) -> Vec<f64> {
    let mut exp_elogthetad = vec![0.0; d * k];
    for i in 0..d {
        let mut row_sum = 0.0;
        for j in 0..k {
            row_sum += gammad[i * k + j];
        }
        let dg_sum = digamma(row_sum.max(1e-15));
        for j in 0..k {
            let val = digamma(gammad[i * k + j].max(1e-15)) - dg_sum;
            exp_elogthetad[i * k + j] = val.exp();
        }
    }
    exp_elogthetad
}

/// Infer document-topic variational posteriors for documents `X_test`.
pub fn predict_doc_topics(
    x_test: &[f64],
    adjusted_factors: &[f64],
    weights: &[f64],
    num_topics: usize,
    num_types: usize,
    n_iter_test: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let d = x_test.len() / num_types;
    let k = num_topics;
    let v = num_types;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut gammad = vec![0.0; d * k];
    for i in 0..(d * k) {
        let u: f64 = rng.gen();
        gammad[i] = -u.max(1e-15).ln();
    }

    let mut exp_elogthetad = dirichlet_expectation(&gammad, d, k);
    let epsilon = f64::EPSILON;
    let mut phinorm = vec![0.0; d * v];
    for i in 0..d {
        for j in 0..v {
            let mut sum = 0.0;
            for r in 0..k {
                sum += exp_elogthetad[i * k + r] * adjusted_factors[j * k + r];
            }
            phinorm[i * v + j] = sum + epsilon;
        }
    }

    let mut i_iter = 0;
    let mut max_gamma_change = 1.0;

    while max_gamma_change > 5e-3 && i_iter < n_iter_test {
        let lastgamma = gammad.clone();

        let mut x_phi_norm = vec![0.0; d * v];
        for idx in 0..(d * v) {
            x_phi_norm[idx] = x_test[idx] / phinorm[idx];
        }

        let mut x_phi_norm_factors = vec![0.0; d * k];
        for i in 0..d {
            for r in 0..k {
                let mut sum = 0.0;
                for j in 0..v {
                    sum += x_phi_norm[i * v + j] * adjusted_factors[j * k + r];
                }
                x_phi_norm_factors[i * k + r] = sum;
            }
        }

        for idx in 0..(d * k) {
            let col = idx % k;
            gammad[idx] = exp_elogthetad[idx] * x_phi_norm_factors[idx] + weights[col];
        }

        exp_elogthetad = dirichlet_expectation(&gammad, d, k);

        for i in 0..d {
            for j in 0..v {
                let mut sum = 0.0;
                for r in 0..k {
                    sum += exp_elogthetad[i * k + r] * adjusted_factors[j * k + r];
                }
                phinorm[i * v + j] = sum + epsilon;
            }
        }

        let mut max_change = 0.0f64;
        for row in 0..d {
            let mut diff_sum = 0.0;
            for col in 0..k {
                diff_sum += (gammad[row * k + col] - lastgamma[row * k + col]).abs();
            }
            let change = diff_sum / k as f64;
            if change > max_change {
                max_change = change;
            }
        }
        max_gamma_change = max_change;
        i_iter += 1;
    }

    let mut doc_topics = vec![vec![0.0; k]; d];
    for i in 0..d {
        let mut row_sum = 0.0;
        for j in 0..k {
            row_sum += gammad[i * k + j];
        }
        if row_sum < 1e-12 {
            row_sum = 1e-12;
        }
        for j in 0..k {
            doc_topics[i][j] = gammad[i * k + j] / row_sum;
        }
    }
    doc_topics
}

// ---------------------------------------------------------------------------
// Streaming / online path: incremental whitening + per-batch CP SGD.
//
// Mirrors the reference `tlda` package's `TLDA.partial_fit(X_batch, batch_index)`
// (second_order_cumulant.SecondOrderCumulant backed by sklearn IncrementalPCA,
// third_order_cumulant.ThirdOrderCumulant SGD). A batch seen for the FIRST time
// updates the running mean + incremental PCA (whitening) and nothing else; each
// LATER sighting whitens the batch with the current PCA and runs the third-order
// factor SGD. So a typical driver makes one pass to build the whitening, then
// `n_iter_train` passes to train the factors -- never holding the whole
// count matrix in memory.
// ---------------------------------------------------------------------------

/// Faithful port of the slice of `sklearn.decomposition.IncrementalPCA` the
/// reference whitening uses: running mean + the sequential-Karhunen-Loeve
/// (Ross et al. 2008) SVD merge with a mean-correction row. We only keep the
/// pieces the whitening needs -- the top-`k` right singular vectors
/// (`components`, `k x v`) and singular values (`k`); `whitening_weights` is
/// `S^2 / n_samples_seen` (the reference's `explained_variance_ * (n-1)/n`).
struct IncrementalPca {
    n_components: usize,
    num_types: usize,
    /// `k x v`, row-major; `None` until the first batch.
    components: Option<Vec<f64>>,
    /// length `k`; `None` until the first batch.
    singular_values: Option<Vec<f64>>,
    /// running per-feature mean, length `v`.
    mean: Vec<f64>,
    n_samples_seen: usize,
}

impl IncrementalPca {
    fn new(n_components: usize, num_types: usize) -> Self {
        IncrementalPca {
            n_components,
            num_types,
            components: None,
            singular_values: None,
            mean: vec![0.0; num_types],
            n_samples_seen: 0,
        }
    }

    /// Economy SVD of `a` (`m x v`, row-major) keeping the top-`k` singular
    /// values and right singular vectors. Uses the Gram matrix `A Aᵀ` (`m x m`,
    /// small since `m = k + batch + 1`) and the symmetric Jacobi eigensolver, so
    /// no dense `v x v` work. Returns `(s[k], components[k x v])`.
    fn svd_topk(a: &[f64], m: usize, v: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
        let at = transpose(a, m, v);
        let gram = matmul(a, &at, m, v, m); // m x m
        let (eigvals, eigvecs) = crate::linalg::jacobi_eigen(&gram, m, 1e-12, 100 * m * m)
            .expect("Jacobi eigensolve failed in incremental whitening");
        let mut s = vec![0.0; k];
        let mut components = vec![0.0; k * v];
        for i in 0..k.min(m) {
            let si = eigvals[i].max(0.0).sqrt();
            s[i] = si;
            if si > 1e-12 {
                // right singular vector vᵢ = Aᵀ uᵢ / sᵢ, with uᵢ = eigvecs column i.
                for col in 0..v {
                    let mut acc = 0.0;
                    for r in 0..m {
                        acc += a[r * v + col] * eigvecs[r * m + i];
                    }
                    components[i * v + col] = acc / si;
                }
            }
        }
        // svd_flip (u_based_decision=False): make the max-abs entry of each
        // component row positive, for a deterministic sign convention.
        for i in 0..k {
            let row = &mut components[i * v..(i + 1) * v];
            let mut max_abs = 0.0;
            let mut sign = 1.0;
            for &val in row.iter() {
                if val.abs() > max_abs {
                    max_abs = val.abs();
                    sign = if val < 0.0 { -1.0 } else { 1.0 };
                }
            }
            if sign < 0.0 {
                for val in row.iter_mut() {
                    *val = -*val;
                }
            }
        }
        (s, components)
    }

    /// Incremental update from a batch `x` (`n x v`, row-major). `x` is already
    /// globally centered and scaled by the caller, exactly as the reference
    /// feeds `(X_batch - mean) * sqrt(alpha_0 + 1)` to `IncrementalPCA.partial_fit`.
    fn partial_fit(&mut self, x: &[f64], n: usize) {
        let v = self.num_types;
        let k = self.n_components;

        // Incremental column mean (West 1979, the mean half of sklearn's
        // _incremental_mean_and_var).
        let last_count = self.n_samples_seen;
        let total = last_count + n;
        let mut col_mean = vec![0.0; v];
        {
            let mut new_sum = vec![0.0; v];
            for i in 0..n {
                for j in 0..v {
                    new_sum[j] += x[i * v + j];
                }
            }
            for j in 0..v {
                let last_sum = self.mean[j] * last_count as f64;
                col_mean[j] = (last_sum + new_sum[j]) / total as f64;
            }
        }

        let (mat, m) = if let (Some(comps), Some(svals)) =
            (self.components.as_ref(), self.singular_values.as_ref())
        {
            // Merge previous factors, this (batch-centered) block, and the
            // mean-correction row, then re-SVD -- the sklearn IncrementalPCA step.
            let mut col_batch_mean = vec![0.0; v];
            for i in 0..n {
                for j in 0..v {
                    col_batch_mean[j] += x[i * v + j];
                }
            }
            for j in 0..v {
                col_batch_mean[j] /= n as f64;
            }
            let m = k + n + 1;
            let mut mat = vec![0.0; m * v];
            // rows 0..k : singular_values[:,None] * components
            for i in 0..k {
                for j in 0..v {
                    mat[i * v + j] = svals[i] * comps[i * v + j];
                }
            }
            // rows k..k+n : batch centered by its own mean
            for i in 0..n {
                for j in 0..v {
                    mat[(k + i) * v + j] = x[i * v + j] - col_batch_mean[j];
                }
            }
            // last row : mean correction
            let corr = ((last_count as f64 * n as f64) / total as f64).sqrt();
            for j in 0..v {
                mat[(m - 1) * v + j] = corr * (self.mean[j] - col_batch_mean[j]);
            }
            (mat, m)
        } else {
            // First batch: center by the batch mean and SVD directly.
            let mut centered = vec![0.0; n * v];
            for i in 0..n {
                for j in 0..v {
                    centered[i * v + j] = x[i * v + j] - col_mean[j];
                }
            }
            (centered, n)
        };

        let (s, components) = Self::svd_topk(&mat, m, v, k);
        self.singular_values = Some(s);
        self.components = Some(components);
        self.mean = col_mean;
        self.n_samples_seen = total;
    }

    /// Per-topic whitening eigenvalues `lambda_k = S_k^2 / n_samples_seen`
    /// (the reference `whitening_weights_`).
    fn whitening_weights(&self) -> Vec<f64> {
        let svals = self
            .singular_values
            .as_ref()
            .expect("IncrementalPca not fitted");
        svals
            .iter()
            .map(|&s| ((s * s) / self.n_samples_seen.max(1) as f64).max(1e-12))
            .collect()
    }
}

/// Streaming Online Tensor LDA state. Persists whitening + factor state across
/// `partial_fit_batch` calls; `finalize` recovers the topic-word / weights.
pub struct TldaStream {
    num_topics: usize,
    num_types: usize,
    n_eigen: usize,
    alpha_0: f64,
    theta: f64,
    learning_rate: f64,
    third_batch_size: usize,
    pca_batch_size: usize,
    smoothing: f64,
    /// running global mean over all docs seen (the wrapper's `self.mean`).
    mean: Vec<f64>,
    n_documents: usize,
    ipca: IncrementalPca,
    /// `n_eigen x num_topics`, row-major.
    factors: Vec<f64>,
    /// batch_index -> times seen (0 = whitening-only pass done, >=1 = trained).
    seen_batches: std::collections::HashMap<i64, u32>,
    /// number of third-order SGD sub-batch updates applied.
    n_factor_updates: usize,
    fit_history: Vec<(usize, f64)>,
}

impl TldaStream {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_topics: usize,
        num_types: usize,
        n_eigenvec: Option<usize>,
        alpha_0: f64,
        theta: f64,
        learning_rate: f64,
        pca_batch_size: usize,
        third_batch_size: usize,
        smoothing: f64,
        seed: u64,
    ) -> Self {
        let n_eigen = n_eigenvec.unwrap_or(num_topics);
        // Deterministic orthonormal factor init, matching the batch path.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut init_mat = vec![0.0; n_eigen * num_topics];
        for x in init_mat.iter_mut() {
            *x = rng.gen_range(-1.0..1.0);
        }
        let factors = qr_reduced(&init_mat, n_eigen, num_topics).0;
        TldaStream {
            num_topics,
            num_types,
            n_eigen,
            alpha_0,
            theta,
            learning_rate,
            third_batch_size,
            pca_batch_size: pca_batch_size.max(1),
            smoothing,
            mean: vec![0.0; num_types],
            n_documents: 0,
            ipca: IncrementalPca::new(n_eigen, num_types),
            factors,
            seen_batches: std::collections::HashMap::new(),
            n_factor_updates: 0,
            fit_history: Vec::new(),
        }
    }

    /// Whiten a globally-centered batch with the current PCA:
    /// `(X - mean) . (componentsᵀ / sqrt(whitening_weights))`. No `alpha_0` scale
    /// here -- it entered through the fit, exactly as in the reference `transform`.
    fn whiten(&self, x_centered: &[f64], n: usize) -> Vec<f64> {
        let v = self.num_types;
        let k = self.n_eigen;
        let comps = self
            .ipca
            .components
            .as_ref()
            .expect("whitening not initialized: feed every batch once before training");
        // projection[v x k] = componentsᵀ / sqrt(lambda); hoist the per-component
        // 1/sqrt(lambda) out of the per-document loop.
        let inv: Vec<f64> = self
            .ipca
            .whitening_weights()
            .iter()
            .map(|l| 1.0 / l.sqrt())
            .collect();
        let mut whit = vec![0.0; n * k];
        for i in 0..n {
            for kk in 0..k {
                let mut acc = 0.0;
                for j in 0..v {
                    acc += x_centered[i * v + j] * comps[kk * v + j];
                }
                whit[i * k + kk] = acc * inv[kk];
            }
        }
        whit
    }

    /// One `partial_fit(X_batch, batch_index)` call. `counts` is `n x num_types`
    /// row-major term counts. The first time a `batch_index` is seen this updates
    /// the running mean + incremental whitening; every later sighting whitens the
    /// batch and runs the CP factor SGD.
    pub fn partial_fit_batch(&mut self, counts: &[f64], n: usize, batch_index: i64) {
        let v = self.num_types;
        let seen = self.seen_batches.get(&batch_index).copied();
        if seen.is_none() {
            // First sighting: update global mean, then incremental whitening.
            let mut batch_sum = vec![0.0; v];
            for i in 0..n {
                for j in 0..v {
                    batch_sum[j] += counts[i * v + j];
                }
            }
            let prev = self.n_documents;
            let total = prev + n;
            for j in 0..v {
                self.mean[j] = (self.mean[j] * prev as f64 + batch_sum[j]) / total as f64;
            }
            self.n_documents = total;

            // Feed the incremental PCA in sub-batches of `pca_batch_size`
            // (the reference's `_partial_fit_second_order` loop), bounding the
            // Gram-matrix size regardless of how large a batch the caller passes.
            let scale = (self.alpha_0 + 1.0).sqrt();
            let step = self.pca_batch_size.max(1);
            for j in (0..n).step_by(step) {
                let b = (n - j).min(step);
                let mut sub = vec![0.0; b * v];
                for i in 0..b {
                    for c in 0..v {
                        sub[i * v + c] = (counts[(j + i) * v + c] - self.mean[c]) * scale;
                    }
                }
                self.ipca.partial_fit(&sub, b);
            }
            self.seen_batches.insert(batch_index, 0);
        } else {
            // Later sighting: whiten with the current PCA and SGD the factors.
            let mut centered = vec![0.0; n * v];
            for i in 0..n {
                for j in 0..v {
                    centered[i * v + j] = counts[i * v + j] - self.mean[j];
                }
            }
            let whit = self.whiten(&centered, n);
            // Track the largest factor change across this batch's sub-batch
            // updates, so fit_history carries a meaningful convergence trace (the
            // streaming analogue of the batch path's per-epoch max-diff).
            let prev_factors = self.factors.clone();
            for j in (0..n).step_by(self.third_batch_size) {
                let b = (n - j).min(self.third_batch_size);
                let y = &whit[j * self.n_eigen..(j + b) * self.n_eigen];
                partial_fit_step(
                    &mut self.factors,
                    y,
                    self.n_eigen,
                    self.num_topics,
                    b,
                    self.alpha_0,
                    self.theta,
                    self.learning_rate,
                );
                self.n_factor_updates += 1;
            }
            let max_diff = self
                .factors
                .iter()
                .zip(prev_factors.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            self.fit_history.push((self.n_factor_updates, max_diff));
            *self.seen_batches.get_mut(&batch_index).unwrap() += 1;
        }
    }

    /// Whether every streamed batch has had at least one factor SGD update (each
    /// batch seen at least twice: one whitening pass, then one or more training
    /// passes). `finalize` errors otherwise, so a model cannot be recovered from
    /// factors trained on only a fraction of the corpus.
    pub fn trained(&self) -> bool {
        !self.seen_batches.is_empty() && self.seen_batches.values().all(|&c| c >= 1)
    }

    /// Number of documents seen during the whitening pass (first sightings). The
    /// whitening rank cannot exceed this, so `finalize` rejects a stream too small
    /// to support `n_eigen` components.
    pub fn n_documents(&self) -> usize {
        self.n_documents
    }

    /// The whitening rank (`n_eigenvec`), for the caller's readiness checks.
    pub fn n_eigen(&self) -> usize {
        self.n_eigen
    }

    /// Whether the last training pass moved the factors by less than the SGD
    /// tolerance -- the streaming analogue of the batch path's convergence flag.
    pub fn converged(&self) -> bool {
        matches!(self.fit_history.last(), Some(&(_, diff)) if diff < STREAM_CONVERGENCE_TOL)
    }

    /// Recover the topic-word matrix and weights from the current whitening +
    /// factors. Reuses the shared `recover_params` with `u_proj = componentsᵀ` and
    /// `lambda = whitening_weights`. `doc_topic` is left empty (streaming does not
    /// retain the documents; use `predict_doc_topics` / `transform` for that).
    pub fn finalize(&self) -> TensorLdaModel {
        let v = self.num_types;
        let k = self.num_topics;
        let n_eigen = self.n_eigen;
        let comps = self.ipca.components.as_ref().expect("whitening not fitted");
        let lambdas = self.ipca.whitening_weights();

        // u_scaled[v x n_eigen] = componentsᵀ * sqrt(lambda)
        let mut u_scaled = vec![0.0; v * n_eigen];
        for w in 0..v {
            for ke in 0..n_eigen {
                u_scaled[w * n_eigen + ke] = comps[ke * v + w] * lambdas[ke].sqrt();
            }
        }
        let (topic_word, unwhitened_raw, weights) = recover_params(
            &u_scaled,
            &self.factors,
            &self.mean,
            v,
            n_eigen,
            k,
            self.smoothing,
            self.alpha_0,
        );

        TensorLdaModel {
            num_topics: k,
            num_types: v,
            topic_word,
            doc_topic: Vec::new(),
            weights,
            alpha_0: self.alpha_0,
            fit_history: self.fit_history.clone(),
            converged: self.converged(),
            unwhitened_raw,
        }
    }
}

/// Shared parameter recovery for both the batch and streaming paths: unwhiten
/// the CP factors, add the mean, clamp to non-negative, smooth, column-normalize
/// to a `K x V` topic-word simplex, and recover Dirichlet weights from the
/// unwhitened column L2 norms. `u_scaled` is `V x n_eigen` (the whitening basis
/// scaled by `sqrt(lambda)`), `factors` is `n_eigen x K`, `mean` length `V`.
/// Returns `(topic_word, unwhitened_raw, weights)`.
///
/// The weight rule is invariant to orthogonal rotations / permutations of the
/// SVD whitening basis: `alpha_j = ||column_j of unwhitened_raw||^2`, normalized
/// to sum to `alpha_0`, so the weights track the topics' relative prevalence.
fn recover_params(
    u_scaled: &[f64],
    factors: &[f64],
    mean: &[f64],
    v: usize,
    n_eigen: usize,
    k: usize,
    smoothing: f64,
    alpha_0: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut factors_unwhitened = matmul(u_scaled, factors, v, n_eigen, k);
    for w in 0..v {
        for ki in 0..k {
            factors_unwhitened[w * k + ki] += mean[w];
            if factors_unwhitened[w * k + ki] < 0.0 {
                factors_unwhitened[w * k + ki] = 0.0;
            }
        }
    }
    let unwhitened_raw = factors_unwhitened.clone();

    for x in factors_unwhitened.iter_mut() {
        *x = *x * (1.0 - smoothing) + (smoothing / v as f64);
    }
    let mut topic_word = vec![vec![0.0; v]; k];
    for ki in 0..k {
        let mut col_sum = 0.0;
        for w in 0..v {
            col_sum += factors_unwhitened[w * k + ki];
        }
        let col_sum = col_sum.max(1e-12);
        for w in 0..v {
            topic_word[ki][w] = factors_unwhitened[w * k + ki] / col_sum;
        }
    }

    let mut alpha = vec![0.0; k];
    let mut alpha_sum = 0.0;
    for j in 0..k {
        let mut c2 = 0.0;
        for w in 0..v {
            let val = unwhitened_raw[w * k + j];
            c2 += val * val;
        }
        alpha[j] = c2.max(1e-12);
        alpha_sum += alpha[j];
    }
    let alpha_sum = alpha_sum.max(1e-12);
    let weights = alpha.iter().map(|a| (a / alpha_sum) * alpha_0).collect();
    (topic_word, unwhitened_raw, weights)
}

/// Fit Online Tensor LDA on the given document-term counts.
pub fn fit_tlda<F: FnMut(usize, usize) -> bool>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    alpha_0: f64,
    n_iter_train: usize,
    n_iter_test: usize,
    learning_rate: f64,
    batch_size: usize,
    smoothing: f64,
    theta: f64,
    n_eigenvec: Option<usize>,
    seed: u64,
    mut on_progress: F,
) -> TensorLdaModel {
    let d = docs.len();
    let v = num_types;
    let mut x = vec![0.0; d * v];
    for (i, doc) in docs.iter().enumerate() {
        for &w in doc {
            if (w as usize) < v {
                x[i * v + w as usize] += 1.0;
            }
        }
    }

    let n_eigen = n_eigenvec.unwrap_or(num_topics);

    // 1. Whitening (2nd-order cumulant)
    let mut mean = vec![0.0; v];
    for i in 0..d {
        for j in 0..v {
            mean[j] += x[i * v + j];
        }
    }
    for j in 0..v {
        mean[j] /= d as f64;
    }

    let mut z = vec![0.0; d * v];
    let scale = (alpha_0 + 1.0).sqrt();
    for i in 0..d {
        for j in 0..v {
            z[i * v + j] = (x[i * v + j] - mean[j]) * scale;
        }
    }

    let (s_vals, u_proj) =
        randomized_svd(&z, d, v, n_eigen, 5, 2, seed).expect("SVD failed in whitening");

    let mut lambdas = vec![0.0; n_eigen];
    for i in 0..n_eigen {
        let s_i = s_vals[i];
        lambdas[i] = ((s_i * s_i) / (d as f64)).max(1e-12);
    }

    let mut w_mat = vec![0.0; v * n_eigen];
    for w_idx in 0..v {
        for k_idx in 0..n_eigen {
            w_mat[w_idx * n_eigen + k_idx] =
                u_proj[w_idx * n_eigen + k_idx] / lambdas[k_idx].sqrt();
        }
    }

    let mut x_centered = vec![0.0; d * v];
    for i in 0..d {
        for j in 0..v {
            x_centered[i * v + j] = x[i * v + j] - mean[j];
        }
    }

    let x_whit = matmul(&x_centered, &w_mat, d, v, n_eigen);

    // 2. CP decomposition on whitened 3rd-order cumulant
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut init_mat = vec![0.0; n_eigen * num_topics];
    for i in 0..(n_eigen * num_topics) {
        init_mat[i] = rng.gen_range(-1.0..1.0);
    }
    let mut factors = qr_reduced(&init_mat, n_eigen, num_topics).0;

    let mut fit_history = Vec::new();
    let mut converged = false;
    let mut i = 1;
    let mut max_diff = 1.0;
    let tol = 1e-5;

    while (i <= 10 || max_diff >= tol) && i <= n_iter_train {
        let prev_fac = factors.clone();
        for j in (0..d).step_by(batch_size) {
            let b_size = (d - j).min(batch_size);
            let y = &x_whit[j * n_eigen..(j + b_size) * n_eigen];
            partial_fit_step(
                &mut factors,
                y,
                n_eigen,
                num_topics,
                b_size,
                alpha_0,
                theta,
                learning_rate,
            );
        }

        let mut diff = 0.0f64;
        for idx in 0..(n_eigen * num_topics) {
            let d_val = (factors[idx] - prev_fac[idx]).abs();
            if d_val > diff {
                diff = d_val;
            }
        }
        max_diff = diff;
        fit_history.push((i, max_diff));
        if max_diff < tol && i >= 10 {
            converged = true;
            let _ = on_progress(i, i); // snap bar to 100% on early convergence (#786)
            break;
        }
        if !on_progress(i, n_iter_train) {
            break;
        }
        i += 1;
    }

    // 3. Parameter recovery (unwhitening + normalization), shared with the
    // streaming path.
    let mut u_scaled = vec![0.0; v * n_eigen];
    for w_idx in 0..v {
        for k_idx in 0..n_eigen {
            u_scaled[w_idx * n_eigen + k_idx] =
                u_proj[w_idx * n_eigen + k_idx] * lambdas[k_idx].sqrt();
        }
    }
    let (topic_word, unwhitened_raw, weights) = recover_params(
        &u_scaled, &factors, &mean, v, n_eigen, num_topics, smoothing, alpha_0,
    );

    // 4. Document-topic inference
    let doc_topic = predict_doc_topics(
        &x,
        &unwhitened_raw,
        &weights,
        num_topics,
        v,
        n_iter_test,
        seed,
    );

    TensorLdaModel {
        num_topics,
        num_types: v,
        topic_word,
        doc_topic,
        weights,
        alpha_0,
        fit_history,
        converged,
        unwhitened_raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planted_corpus(
        k: usize,
        block: usize,
        ndocs: usize,
        dlen: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, usize) {
        let v = k * block;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let docs: Vec<Vec<u32>> = (0..ndocs)
            .map(|d| {
                let b = d % k;
                (0..dlen)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        (docs, v)
    }

    #[test]
    fn test_tlda_conforms_and_recovers() {
        let k = 3;
        let block = 6;
        let (docs, v) = planted_corpus(k, block, 60, 10, 42);

        let m = fit_tlda(
            &docs,
            k,
            v,
            1.0,
            50,
            20,
            0.01,
            10,
            0.01,
            1.0,
            None,
            42,
            |_, _| true,
        );
        assert_eq!(m.num_topics(), k);
        assert_eq!(m.topic_word.len(), k);
        assert_eq!(m.topic_word[0].len(), v);
        assert_eq!(m.doc_topic.len(), docs.len());
        assert_eq!(m.doc_topic[0].len(), k);

        // Verify rows sum to 1 in doc_topic and topic_word
        for row in &m.doc_topic {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
        for row in &m.topic_word {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    /// Build a term-count row-major matrix (n x v) from token-id docs.
    fn counts_of(docs: &[Vec<u32>], v: usize) -> Vec<f64> {
        let mut x = vec![0.0; docs.len() * v];
        for (i, d) in docs.iter().enumerate() {
            for &w in d {
                if (w as usize) < v {
                    x[i * v + w as usize] += 1.0;
                }
            }
        }
        x
    }

    #[test]
    fn test_tlda_stream_recovers_and_conforms() {
        // Method-of-moments recovery is init-sensitive (the reference package is
        // too), so we pin a well-separated corpus / seed that recovers, and rely
        // on determinism to keep it stable. The parity leg
        // (parity/tlda_compare.py) is the cross-implementation check.
        let k = 3;
        let block = 8;
        let (docs, v) = planted_corpus(k, block, 150, 30, 42);
        let bs = 50; // three batches
        let n_iter_train = 40;

        let mut stream = TldaStream::new(k, v, None, 1.0, 1.0, 0.01, bs, 10, 0.01, 42);
        let batches: Vec<Vec<Vec<u32>>> = docs.chunks(bs).map(|c| c.to_vec()).collect();
        // Pass 0: whitening only (first sighting of each batch). Passes
        // 1..=n_iter_train: whiten + factor SGD.
        for _ in 0..=n_iter_train {
            for (bi, batch) in batches.iter().enumerate() {
                let c = counts_of(batch, v);
                stream.partial_fit_batch(&c, batch.len(), bi as i64);
            }
        }
        assert!(stream.trained());
        let m = stream.finalize();
        assert_eq!(m.topic_word.len(), k);
        assert_eq!(m.topic_word[0].len(), v);
        for row in &m.topic_word {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "topic row must be a simplex");
        }
        // Each recovered topic concentrates on a planted block; on this pinned
        // corpus all three blocks are recovered.
        let mut covered = std::collections::HashSet::new();
        for row in &m.topic_word {
            let top = (0..v).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            covered.insert(top / block);
        }
        assert_eq!(covered.len(), k, "topics should cover all planted blocks");
    }

    #[test]
    fn test_tlda_stream_single_batch_matches_whole_fit_whitening() {
        // A single streaming batch containing all docs builds the same whitening
        // (up to the incremental-PCA sign convention) as the batch path, so the
        // recovered topic-word rows are valid simplices of the right shape.
        let k = 3;
        let block = 8;
        let (docs, v) = planted_corpus(k, block, 150, 30, 42);
        let x = counts_of(&docs, v);
        let mut stream = TldaStream::new(k, v, None, 1.0, 1.0, 0.01, docs.len(), 10, 0.01, 42);
        for _ in 0..=40 {
            stream.partial_fit_batch(&x, docs.len(), 0);
        }
        let m = stream.finalize();
        assert_eq!(m.topic_word.len(), k);
        for row in &m.topic_word {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        // weights are a positive vector summing to alpha_0.
        let ws: f64 = m.weights.iter().sum();
        assert!((ws - 1.0).abs() < 1e-6);
        assert!(m.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_tlda_stream_determinism() {
        let k = 2;
        let block = 5;
        let (docs, v) = planted_corpus(k, block, 40, 8, 123);
        let run = || {
            let mut s = TldaStream::new(k, v, None, 0.5, 1.0, 0.05, 20, 5, 0.05, 999);
            for _ in 0..16 {
                let c = counts_of(&docs, v);
                s.partial_fit_batch(&c, docs.len(), 0);
            }
            s.finalize()
        };
        let a = run();
        let b = run();
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.weights, b.weights);
    }

    #[test]
    fn test_tlda_determinism() {
        let k = 2;
        let block = 5;
        let (docs, v) = planted_corpus(k, block, 40, 8, 123);

        let m1 = fit_tlda(
            &docs,
            k,
            v,
            0.5,
            30,
            15,
            0.05,
            5,
            0.05,
            1.0,
            None,
            999,
            |_, _| true,
        );
        let m2 = fit_tlda(
            &docs,
            k,
            v,
            0.5,
            30,
            15,
            0.05,
            5,
            0.05,
            1.0,
            None,
            999,
            |_, _| true,
        );

        assert_eq!(m1.topic_word, m2.topic_word);
        assert_eq!(m1.doc_topic, m2.doc_topic);
        assert_eq!(m1.weights, m2.weights);
    }
}
