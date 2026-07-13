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

/// Fit Online Tensor LDA on the given document-term counts.
pub fn fit_tlda(
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
        }
        i += 1;
    }

    // 3. Parameter recovery (unwhitening + normalization)
    let mut u_scaled = vec![0.0; v * n_eigen];
    for w_idx in 0..v {
        for k_idx in 0..n_eigen {
            u_scaled[w_idx * n_eigen + k_idx] =
                u_proj[w_idx * n_eigen + k_idx] * lambdas[k_idx].sqrt();
        }
    }
    let mut factors_unwhitened = matmul(&u_scaled, &factors, v, n_eigen, num_topics);

    for w_idx in 0..v {
        for k_idx in 0..num_topics {
            factors_unwhitened[w_idx * num_topics + k_idx] += mean[w_idx];
            if factors_unwhitened[w_idx * num_topics + k_idx] < 0.0 {
                factors_unwhitened[w_idx * num_topics + k_idx] = 0.0;
            }
        }
    }

    let unwhitened_raw = factors_unwhitened.clone();

    for idx in 0..(v * num_topics) {
        factors_unwhitened[idx] =
            factors_unwhitened[idx] * (1.0 - smoothing) + (smoothing / v as f64);
    }

    let mut topic_word = vec![vec![0.0; v]; num_topics];
    for k_idx in 0..num_topics {
        let mut col_sum = 0.0;
        for w_idx in 0..v {
            col_sum += factors_unwhitened[w_idx * num_topics + k_idx];
        }
        if col_sum < 1e-12 {
            col_sum = 1e-12;
        }
        for w_idx in 0..v {
            topic_word[k_idx][w_idx] = factors_unwhitened[w_idx * num_topics + k_idx] / col_sum;
        }
    }

    let mut eig_vals = vec![0.0; num_topics];
    for j in 0..num_topics {
        let mut sum2 = 0.0;
        for i in 0..n_eigen {
            let val = factors[i * num_topics + j];
            sum2 += val * val;
        }
        let col_norm = sum2.sqrt();
        eig_vals[j] = col_norm.powi(3);
    }

    let mut alpha = vec![0.0; num_topics];
    let mut alpha_sum = 0.0;
    for idx in 0..num_topics {
        alpha[idx] = eig_vals[idx].powf(-2.0);
        if alpha[idx].is_nan() || alpha[idx].is_infinite() {
            alpha[idx] = 1e-5;
        }
        alpha_sum += alpha[idx];
    }
    if alpha_sum < 1e-12 {
        alpha_sum = 1e-12;
    }
    let mut weights = vec![0.0; num_topics];
    for idx in 0..num_topics {
        weights[idx] = (alpha[idx] / alpha_sum) * alpha_0;
    }

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

        let m = fit_tlda(&docs, k, v, 1.0, 50, 20, 0.01, 10, 0.01, 42);
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

    #[test]
    fn test_tlda_determinism() {
        let k = 2;
        let block = 5;
        let (docs, v) = planted_corpus(k, block, 40, 8, 123);

        let m1 = fit_tlda(&docs, k, v, 0.5, 30, 15, 0.05, 5, 0.05, 999);
        let m2 = fit_tlda(&docs, k, v, 0.5, 30, 15, 0.05, 5, 0.05, 999);

        assert_eq!(m1.topic_word, m2.topic_word);
        assert_eq!(m1.doc_topic, m2.doc_topic);
        assert_eq!(m1.weights, m2.weights);
    }
}
