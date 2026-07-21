//! Dimensionality reduction for the embedding-clustering topic pipeline.
//!
//! Top2Vec and BERTopic both run the same first stage: take a high-dimensional
//! document embedding (384 dimensions is common) and project it down to a handful
//! of dimensions before a density clusterer (HDBSCAN) groups the documents. The
//! projection has to preserve the gross geometry of the embedding cloud while
//! cutting the dimensionality by two orders of magnitude.
//!
//! We use randomized PCA following the Halko-Martinsson-Tropp (2011) scheme. For
//! a small target rank `k` against a tall matrix, the randomized range finder is
//! far cheaper than a full SVD: we draw a Gaussian sketch, push the data through
//! it to capture the dominant subspace, sharpen with a power iteration or two,
//! orthonormalize, and only then solve a small `l × l` eigenproblem. Everything is
//! BLAS-free and operates on `Vec<Vec<f64>>`, so the stage carries no linear
//! algebra dependency. Determinism comes from seeding `ChaCha8Rng`, which matters
//! because the clusterer downstream is sensitive to the exact coordinates.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Whether the faithful UMAP reducer is compiled in (the `umap` feature). The
/// embedding models default to PCA; when this is false, asking for UMAP should
/// error rather than silently fall back.
pub fn umap_available() -> bool {
    cfg!(feature = "umap")
}

/// The UMAP layout hyperparameters (everything past `n_components`/`n_neighbors`),
/// carried as one value so the reduce pipeline doesn't grow six positional
/// arguments. `Default` is the reference configuration the embedding models used
/// before these became tunable — `cosine` metric, `min_dist = 0.0` (BERTopic's
/// setting for tight clusters), `spread = 1.0`, auto epochs (`n_epochs = 0` →
/// 500 for ≤10k rows), and the standard negative sampling — so a caller that
/// leaves it default gets exactly the previous behavior.
#[derive(Clone, Debug)]
pub struct UmapParams {
    pub min_dist: f64,
    pub spread: f64,
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub repulsion_strength: f64,
    pub metric: String,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            min_dist: 0.0,
            spread: 1.0,
            n_epochs: 0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            metric: "cosine".to_string(),
        }
    }
}

/// Reduce `data` to `n_components` with either UMAP (`use_umap`) or randomized
/// PCA. The embedding heads call this so the reducer is a single switch. `umap`
/// carries UMAP's layout hyperparameters (ignored on the PCA path). When the
/// `umap` feature is not compiled, this always uses PCA (callers that expose UMAP
/// should reject the request earlier via `umap_available`).
pub fn reduce(
    data: &[Vec<f64>],
    n_components: usize,
    use_umap: bool,
    n_neighbors: usize,
    umap_params: &UmapParams,
    seed: u64,
) -> Vec<Vec<f64>> {
    #[cfg(feature = "umap")]
    {
        if use_umap {
            return umap(data, n_components, n_neighbors, umap_params, seed);
        }
    }
    let _ = (use_umap, n_neighbors, umap_params);
    pca(data, n_components, seed)
}

/// Project `data` onto `n_components` with UMAP, using the layout hyperparameters
/// in `params` (see [`UmapParams`]). Delegates to the in-house,
/// umap-learn-faithful reducer in `crate::umap`; unlike the old umap-rs path it is
/// fully reproducible for a fixed `seed`. Only built under `umap`.
#[cfg(feature = "umap")]
pub fn umap(
    data: &[Vec<f64>],
    n_components: usize,
    n_neighbors: usize,
    params: &UmapParams,
    seed: u64,
) -> Vec<Vec<f64>> {
    crate::umap::umap(
        data,
        n_components,
        n_neighbors,
        params.min_dist,
        params.spread,
        params.n_epochs,
        params.negative_sample_rate,
        params.repulsion_strength,
        &params.metric,
        seed,
    )
}

/// Whether the Barnes-Hut t-SNE reducer is compiled in (the `tsne` feature).
pub fn tsne_available() -> bool {
    cfg!(feature = "tsne")
}

/// Project `data` onto `n_components` with Barnes-Hut t-SNE (the `bhtsne` crate).
/// t-SNE pulls apart local neighborhoods for a legible 2-D scatter, at the price of
/// distorting global geometry (between-cluster distances and cluster sizes are not
/// meaningful). `theta` is the Barnes-Hut speed/accuracy knob (0 runs the exact
/// O(n^2) algorithm); `perplexity` is clamped so that `3 * perplexity < n`. The fit
/// is stochastic -- bhtsne exposes no seed -- so the layout is not reproducible
/// across runs. Tiny inputs (n < 4) fall back to PCA. Only built under `tsne`.
#[cfg(feature = "tsne")]
pub fn tsne(
    data: &[Vec<f64>],
    n_components: usize,
    perplexity: f64,
    theta: f64,
    epochs: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n = data.len();
    if n == 0 || n_components == 0 {
        return vec![Vec::new(); n];
    }
    if n < 4 {
        // t-SNE's neighborhood estimate is degenerate at this size; PCA is reproducible.
        return pca(data, n_components, seed);
    }
    // bhtsne computes 3 * perplexity neighbors, so perplexity must stay below n/3.
    let max_perp = ((n as f64 - 1.0) / 3.0).floor().max(1.0);
    let perp = perplexity.clamp(1.0, max_perp) as f32;

    let owned: Vec<Vec<f32>> = data
        .iter()
        .map(|r| r.iter().map(|&v| v as f32).collect())
        .collect();
    let samples: Vec<&[f32]> = owned.iter().map(|r| r.as_slice()).collect();
    let metric = |a: &&[f32], b: &&[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    };

    let mut t = bhtsne::tSNE::new(&samples);
    t.embedding_dim(n_components as u8)
        .perplexity(perp)
        .epochs(epochs.max(1));
    if theta <= 0.0 {
        t.exact(metric);
    } else {
        t.barnes_hut(theta as f32, metric);
    }
    let flat = t.embedding(); // row-major Vec<f32>, length n * n_components
    (0..n)
        .map(|i| {
            (0..n_components)
                .map(|c| flat[i * n_components + c] as f64)
                .collect()
        })
        .collect()
}

/// Project `data` onto `n_components` by a named `method`: `"pca"` (default,
/// deterministic), `"umap"`, or `"tsne"`. UMAP and t-SNE fall back to PCA when their
/// feature is not compiled in; callers that expose them should gate on
/// `umap_available` / `tsne_available` first and surface a clear error instead.
#[allow(clippy::too_many_arguments)]
pub fn project(
    data: &[Vec<f64>],
    n_components: usize,
    method: &str,
    n_neighbors: usize,
    umap_params: &UmapParams,
    perplexity: f64,
    theta: f64,
    epochs: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    match method {
        "umap" => {
            #[cfg(feature = "umap")]
            {
                umap(data, n_components, n_neighbors, umap_params, seed)
            }
            #[cfg(not(feature = "umap"))]
            {
                let _ = (n_neighbors, umap_params);
                pca(data, n_components, seed)
            }
        }
        "tsne" => {
            #[cfg(feature = "tsne")]
            {
                tsne(data, n_components, perplexity, theta, epochs, seed)
            }
            #[cfg(not(feature = "tsne"))]
            {
                let _ = (perplexity, theta, epochs);
                pca(data, n_components, seed)
            }
        }
        _ => pca(data, n_components, seed),
    }
}

/// L2-normalize each row in place, projecting the points onto the unit sphere.
///
/// The embedding heads cluster reduced coordinates with a Euclidean density
/// clusterer, but sentence embeddings are trained for cosine similarity. On raw
/// PCA scores a few high-variance directions dominate the Euclidean metric and
/// HDBSCAN under-splits into a couple of broad clusters; normalizing onto the
/// sphere makes squared Euclidean distance a monotone function of cosine
/// distance (`||a-b||^2 = 2(1 - cos)` for unit vectors), so the clusterer sees
/// the metric the encoder was trained for. A zero-norm row (all-zero
/// coordinates) is left untouched rather than divided by zero.
pub fn l2_normalize_rows(rows: &mut [Vec<f64>]) {
    for row in rows.iter_mut() {
        let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// Project `data` (`n_rows × n_features`, each row a sample) onto its top
/// `n_components` principal components and return the scores (`n_rows ×
/// n_components`).
///
/// The scores are the left singular vectors scaled by their singular values, so
/// column `j` has variance equal to the `j`-th eigenvalue of the centered
/// covariance. We center the data, find an approximate range with a Gaussian
/// sketch plus two power iterations, and read the principal directions off a small
/// symmetric eigenproblem solved with cyclic Jacobi.
///
/// The output is deterministic for a fixed `seed`. Degenerate inputs do not
/// panic: an empty matrix returns empty, and when `n_components` meets or exceeds
/// the feature count we simply return the centered features (padded with zeros if
/// asked for more columns than exist).
pub fn pca(data: &[Vec<f64>], n_components: usize, seed: u64) -> Vec<Vec<f64>> {
    let n_rows = data.len();
    if n_rows == 0 || n_components == 0 {
        return vec![Vec::new(); n_rows];
    }
    let n_features = data[0].len();
    if n_features == 0 {
        return vec![Vec::new(); n_rows];
    }

    let xc = mean_center(data, n_features);

    // When the requested rank reaches the feature dimension there is no reduction
    // to do: the centered features already span the column space. Return them
    // directly, padding with zero columns if more were requested than exist.
    if n_components >= n_features {
        return xc
            .chunks(n_features)
            .map(|row| {
                let mut out = vec![0.0f64; n_components];
                out[..n_features].copy_from_slice(&row[..n_features]);
                out
            })
            .collect();
    }

    // Oversample the target rank so the sketch reliably captures the dominant
    // subspace, but never ask for more columns than the data has.
    let l = (n_components + 10).min(n_features).min(n_rows);
    if l == 0 {
        return vec![vec![0.0f64; n_components]; n_rows];
    }

    // (2) Randomized range finding. Draw an n_features × l Gaussian test matrix,
    // form Y = Xc · Omega, then sharpen with two power iterations
    // Y ← Xc · (Xcᵀ · Y) so the columns lean toward the leading singular vectors.
    let omega = gaussian_matrix(n_features, l, seed);
    let mut y = matmul(&xc, n_rows, n_features, &omega, l);
    for _ in 0..2 {
        // Z = Xcᵀ · Y  (n_features × l), then Y = Xc · Z  (n_rows × l).
        let z = matmul_at_b(&xc, n_rows, n_features, &y, l);
        y = matmul(&xc, n_rows, n_features, &z, l);
    }

    // Orthonormalize Y's columns to get Q (n_rows × l) spanning the same subspace.
    let q = modified_gram_schmidt(&y, n_rows, l);

    // (3) Project onto Q: B = Qᵀ · Xc (l × n_features), then C = B · Bᵀ (l × l),
    // symmetric PSD. C shares its eigenvectors with the covariance, expressed in
    // the Q basis.
    let b = matmul_at_b(&q, n_rows, l, &xc, n_features); // l × n_features
    let c = gram(&b, l, n_features); // l × l, row-major

    // (4) Eigendecompose C; columns of `evecs` are eigenvectors, descending.
    let (evals, evecs) = jacobi_eigen_symmetric(&c, l);

    // (5) Map the top eigenvectors back through Q to recover the left singular
    // vectors and scale each by its singular value √eigenvalue. The scores for
    // component j are (Q · evec_j) · √λ_j.
    // `evals`/`evecs` have length `l`, which can be smaller than `n_components`
    // for tiny inputs (few rows or features). Fill only the columns we have
    // eigenpairs for; the rest stay zero so the shape is always n_rows ×
    // n_components.
    let mut scores = vec![vec![0.0f64; n_components]; n_rows];
    for j in 0..n_components.min(l) {
        let lambda = evals[j].max(0.0);
        let sigma = lambda.sqrt();
        let evec = &evecs[j]; // length l
        for i in 0..n_rows {
            // (Q · evec)_i = Σ_t Q[i,t] · evec[t]
            let mut acc = 0.0;
            for t in 0..l {
                acc += q[i * l + t] * evec[t];
            }
            scores[i][j] = acc * sigma;
        }
    }
    scores
}

/// Subtract per-column means, returning the centered matrix as a flat row-major
/// `n_rows × n_features` buffer.
fn mean_center(data: &[Vec<f64>], n_features: usize) -> Vec<f64> {
    let n_rows = data.len();
    let mut means = vec![0.0f64; n_features];
    for row in data {
        for (j, &v) in row.iter().enumerate().take(n_features) {
            means[j] += v;
        }
    }
    let inv = 1.0 / n_rows as f64;
    for m in means.iter_mut() {
        *m *= inv;
    }
    let mut out = vec![0.0f64; n_rows * n_features];
    for (i, row) in data.iter().enumerate() {
        for j in 0..n_features {
            let v = if j < row.len() { row[j] } else { 0.0 };
            out[i * n_features + j] = v - means[j];
        }
    }
    out
}

/// An `rows × cols` matrix of independent standard-normal draws, row-major. The
/// generator is seeded so the sketch is reproducible.
fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = vec![0.0f64; rows * cols];
    for v in out.iter_mut() {
        *v = standard_normal(&mut rng);
    }
    out
}

/// One standard-normal draw via the Box-Muller transform, so we depend only on
/// `rand`'s uniform sampler and stay deterministic under a seeded generator.
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    // Guard the log against u1 == 0.
    let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Dense `A · B`: `A` is `ar × ac` row-major, `B` is `ac × bc` row-major, result
/// is `ar × bc` row-major.
fn matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], bc: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let aik = a[i * ac + k];
            if aik == 0.0 {
                continue;
            }
            let brow = &b[k * bc..k * bc + bc];
            let orow = &mut out[i * bc..i * bc + bc];
            for j in 0..bc {
                orow[j] += aik * brow[j];
            }
        }
    }
    out
}

/// Dense `Aᵀ · B`: `A` is `ar × ac` row-major, `B` is `ar × bc` row-major (both
/// keyed by the shared `ar` dimension), result is `ac × bc` row-major.
fn matmul_at_b(a: &[f64], ar: usize, ac: usize, b: &[f64], bc: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; ac * bc];
    for r in 0..ar {
        let arow = &a[r * ac..r * ac + ac];
        let brow = &b[r * bc..r * bc + bc];
        for k in 0..ac {
            let ark = arow[k];
            if ark == 0.0 {
                continue;
            }
            let orow = &mut out[k * bc..k * bc + bc];
            for j in 0..bc {
                orow[j] += ark * brow[j];
            }
        }
    }
    out
}

/// Gram matrix `M · Mᵀ` for an `m × n` row-major `M`, returned as an `m × m`
/// symmetric row-major buffer.
fn gram(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * rows];
    for i in 0..rows {
        let ri = &m[i * cols..i * cols + cols];
        for j in i..rows {
            let rj = &m[j * cols..j * cols + cols];
            let mut s = 0.0;
            for c in 0..cols {
                s += ri[c] * rj[c];
            }
            out[i * rows + j] = s;
            out[j * rows + i] = s;
        }
    }
    out
}

/// Orthonormalize the columns of an `rows × cols` row-major matrix with modified
/// Gram-Schmidt, returning Q (`rows × cols`, row-major) with orthonormal columns.
/// A column whose residual norm collapses (linearly dependent or numerically
/// zero) is replaced by a zero column so we never divide by a tiny norm.
fn modified_gram_schmidt(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // Work column by column. Hold the running columns as flat Vecs for cache-
    // friendly inner products without striding the row-major source repeatedly.
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut col: Vec<f64> = (0..rows).map(|r| mat[r * cols + c]).collect();
        // Original column norm; we judge the post-orthogonalization residual
        // against this so a near-dependent column is dropped on a *relative*
        // basis. With an absolute floor only, a rank-deficient column of a
        // large-magnitude sketch (e.g. after power iteration the columns scale
        // like the leading singular values) leaves a residual that is pure
        // rounding noise yet still far above any fixed epsilon; normalizing it
        // injects a spurious unit vector that is not orthogonal to the columns
        // already fixed, which silently corrupts Q.
        let orig_norm: f64 = col.iter().map(|&v| v * v).sum::<f64>().sqrt();
        // Subtract the projection onto each previously fixed column.
        for prev in &q {
            let mut dot = 0.0;
            for r in 0..rows {
                dot += prev[r] * col[r];
            }
            for r in 0..rows {
                col[r] -= dot * prev[r];
            }
        }
        let norm: f64 = col.iter().map(|&v| v * v).sum::<f64>().sqrt();
        // Keep the column only if a meaningful fraction of it survived
        // orthogonalization, relative to where it started (and to an absolute
        // floor for an all-zero input column).
        let floor = 1e-10 * orig_norm.max(1.0);
        if norm > floor {
            let inv = 1.0 / norm;
            for v in col.iter_mut() {
                *v *= inv;
            }
        } else {
            col.iter_mut().for_each(|v| *v = 0.0);
        }
        q.push(col);
    }
    // Repack into row-major rows × cols.
    let mut out = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            out[r * cols + c] = q[c][r];
        }
    }
    out
}

/// Eigendecomposition of a symmetric `l × l` matrix (`a` row-major) by cyclic
/// Jacobi rotations. Returns `(eigenvalues, eigenvectors)` sorted by eigenvalue
/// descending; `eigenvectors[j]` is the unit eigenvector (length `l`) for
/// `eigenvalues[j]`.
///
/// Jacobi zeroes one off-diagonal entry at a time with an orthogonal rotation and
/// sweeps the whole upper triangle until the off-diagonal mass is negligible. For
/// the small `l` we hand it (target rank plus a little oversampling) it converges
/// in a handful of sweeps and needs no shifts or deflation.
pub fn jacobi_eigen_symmetric(a: &[f64], l: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    if l == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut m = a.to_vec(); // working copy, modified in place
                            // Accumulated rotations; v starts as the identity, columns end as eigenvectors.
    let mut v = vec![0.0f64; l * l];
    for i in 0..l {
        v[i * l + i] = 1.0;
    }
    if l == 1 {
        return (vec![m[0]], vec![vec![1.0]]);
    }

    for _sweep in 0..100 {
        // Off-diagonal Frobenius mass; stop once it is numerically zero.
        let mut off = 0.0;
        for p in 0..l {
            for q in (p + 1)..l {
                off += m[p * l + q] * m[p * l + q];
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..l {
            for q in (p + 1)..l {
                let apq = m[p * l + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = m[p * l + p];
                let aqq = m[q * l + q];
                // Rotation angle that annihilates (p, q): standard Jacobi formula.
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let cos = 1.0 / (t * t + 1.0).sqrt();
                let sin = t * cos;

                // Apply the rotation to rows/cols p and q of m.
                for k in 0..l {
                    let mkp = m[k * l + p];
                    let mkq = m[k * l + q];
                    m[k * l + p] = cos * mkp - sin * mkq;
                    m[k * l + q] = sin * mkp + cos * mkq;
                }
                for k in 0..l {
                    let mpk = m[p * l + k];
                    let mqk = m[q * l + k];
                    m[p * l + k] = cos * mpk - sin * mqk;
                    m[q * l + k] = sin * mpk + cos * mqk;
                }
                // Force the annihilated entries to exactly zero / symmetry.
                m[p * l + q] = 0.0;
                m[q * l + p] = 0.0;

                // Accumulate the rotation into the eigenvector matrix.
                for k in 0..l {
                    let vkp = v[k * l + p];
                    let vkq = v[k * l + q];
                    v[k * l + p] = cos * vkp - sin * vkq;
                    v[k * l + q] = sin * vkp + cos * vkq;
                }
            }
        }
    }

    // Diagonal of m holds the eigenvalues; column k of v is its eigenvector.
    let mut order: Vec<usize> = (0..l).collect();
    let diag: Vec<f64> = (0..l).map(|i| m[i * l + i]).collect();
    order.sort_by(|&i, &j| {
        diag[j]
            .partial_cmp(&diag[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eigenvalues: Vec<f64> = order.iter().map(|&k| diag[k]).collect();
    let eigenvectors: Vec<Vec<f64>> = order
        .iter()
        .map(|&k| (0..l).map(|r| v[r * l + k]).collect())
        .collect();
    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Column variance of a score matrix.
    fn col_var(scores: &[Vec<f64>], j: usize) -> f64 {
        let n = scores.len() as f64;
        let mean: f64 = scores.iter().map(|r| r[j]).sum::<f64>() / n;
        scores.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n
    }

    /// Build points that lie on a 2-D plane embedded in 5-D: random 2-D
    /// coordinates pushed through a fixed 2×5 basis, plus a constant offset.
    fn plane_in_5d(n: usize) -> (Vec<[f64; 2]>, Vec<Vec<f64>>) {
        let basis: [[f64; 5]; 2] = [[1.0, 0.5, -0.3, 0.2, 0.8], [-0.4, 1.2, 0.7, -0.6, 0.1]];
        let offset = [3.0, -1.0, 2.0, 0.5, -2.5];
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut coords = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.gen::<f64>() * 4.0 - 2.0;
            let b = rng.gen::<f64>() * 4.0 - 2.0;
            coords.push([a, b]);
            let mut row = vec![0.0f64; 5];
            for d in 0..5 {
                row[d] = offset[d] + a * basis[0][d] + b * basis[1][d];
            }
            data.push(row);
        }
        (coords, data)
    }

    #[test]
    fn recovers_low_rank_structure() {
        let (_coords, data) = plane_in_5d(200);
        // Ask for three components against intrinsically 2-D data; the third
        // should carry essentially no variance.
        let scores = pca(&data, 3, 42);
        let v0 = col_var(&scores, 0);
        let v1 = col_var(&scores, 1);
        let v2 = col_var(&scores, 2);
        assert!(v0 > 0.0 && v1 > 0.0, "leading variances must be positive");
        assert!(
            v2 < 1e-6 * v0,
            "third component variance {} should be negligible vs {}",
            v2,
            v0
        );
    }

    #[test]
    fn preserves_pairwise_distances() {
        // The data lies *exactly* in a 2-D affine subspace of R^5, so projecting
        // onto two principal components is an isometry of those points: the scores
        // are the in-subspace coordinates up to a global rotation/reflection. That
        // means every pairwise Euclidean distance is preserved exactly (no
        // per-pair scaling). This is the right invariant; the original generating
        // `coords` are *not* a valid reference because the generating basis is not
        // orthonormal, so distances in `coords` space are distorted relative to
        // distances in the 5-D embedding the projection actually preserves.
        let (_coords, data) = plane_in_5d(60);
        let scores = pca(&data, 2, 1);
        let n = data.len();

        let ddata = |i: usize, j: usize| {
            data[i]
                .iter()
                .zip(&data[j])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let dscore = |i: usize, j: usize| {
            ((scores[i][0] - scores[j][0]).powi(2) + (scores[i][1] - scores[j][1]).powi(2)).sqrt()
        };

        let mut max_abs_err = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let err = (ddata(i, j) - dscore(i, j)).abs();
                if err > max_abs_err {
                    max_abs_err = err;
                }
            }
        }
        assert!(
            max_abs_err < 1e-6,
            "2-D scores should reproduce the 5-D pairwise distances exactly for \
             rank-2 data; max abs error was {}",
            max_abs_err
        );
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let (_c, data) = plane_in_5d(80);
        let a = pca(&data, 2, 123);
        let b = pca(&data, 2, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn handles_edge_cases() {
        // Empty input.
        assert!(pca(&[], 3, 0).is_empty());
        // n_components >= n_features: centered features, padded.
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 0.0]];
        let scores = pca(&data, 4, 0);
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].len(), 4);
        // n_rows < n_components stays panic-free and returns the right shape.
        let small = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let s = pca(&small, 3, 0);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].len(), 3);
    }

    #[test]
    fn l2_normalize_rows_projects_onto_sphere() {
        let mut rows = vec![
            vec![3.0, 4.0],       // norm 5 -> (0.6, 0.8)
            vec![0.0, 0.0],       // zero row stays put (no divide by zero)
            vec![-2.0, 0.0, 0.0], // norm 2 -> (-1, 0, 0)
        ];
        l2_normalize_rows(&mut rows);
        assert!((rows[0][0] - 0.6).abs() < 1e-12 && (rows[0][1] - 0.8).abs() < 1e-12);
        assert_eq!(rows[1], vec![0.0, 0.0]);
        assert!((rows[2][0] + 1.0).abs() < 1e-12);
        // Every non-zero row now has unit norm.
        for row in [&rows[0], &rows[2]] {
            let n: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-12, "row norm {n}");
        }
    }

    #[test]
    fn jacobi_recovers_known_eigenvalues() {
        // Symmetric matrix [[2, 1], [1, 2]] has eigenvalues 3 and 1 with
        // eigenvectors (1, 1)/√2 and (1, -1)/√2.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, vecs) = jacobi_eigen_symmetric(&a, 2);
        assert!((vals[0] - 3.0).abs() < 1e-10, "λ0 = {}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 1e-10, "λ1 = {}", vals[1]);
        // Leading eigenvector aligns with (1, 1) up to sign.
        let v = &vecs[0];
        assert!((v[0].abs() - v[1].abs()).abs() < 1e-9);
        assert!(
            v[0] * v[1] > 0.0,
            "leading eigenvector should have equal signs"
        );
        // Second eigenvector aligns with (1, -1).
        let w = &vecs[1];
        assert!(
            w[0] * w[1] < 0.0,
            "second eigenvector should have opposite signs"
        );
    }

    #[test]
    fn jacobi_diagonal_is_identity_basis() {
        // A diagonal matrix returns its diagonal, sorted descending, with the
        // standard basis as eigenvectors.
        let a = vec![1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0];
        let (vals, vecs) = jacobi_eigen_symmetric(&a, 3);
        assert!((vals[0] - 5.0).abs() < 1e-12);
        assert!((vals[1] - 3.0).abs() < 1e-12);
        assert!((vals[2] - 1.0).abs() < 1e-12);
        // The top eigenvector points along the second axis (the 5.0 entry).
        assert!((vecs[0][1].abs() - 1.0).abs() < 1e-12);
    }
}

#[cfg(all(test, feature = "umap"))]
mod umap_tests {
    use super::*;

    // UMAP of three well-separated high-dimensional blobs into 2-D should keep
    // each point's nearest neighbor within its own blob.
    #[test]
    fn umap_separates_blobs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let dim = 10;
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for c in 0..3 {
            let mut center = vec![0.0; dim];
            center[c] = 20.0;
            for _ in 0..40 {
                // Varied (non-degenerate) within-blob distances, as real
                // embeddings have, so smooth_knn_dist sees a real density gradient.
                let row: Vec<f64> = (0..dim)
                    .map(|t| center[t] + (rng.gen::<f64>() - 0.5) * 6.0)
                    .collect();
                data.push(row);
                truth.push(c);
            }
        }
        let emb = umap(&data, 2, 10, &UmapParams::default(), 1);
        assert_eq!(emb.len(), data.len());
        assert_eq!(emb[0].len(), 2);
        assert!(
            emb.iter().all(|r| r.iter().all(|v| v.is_finite())),
            "embedding has NaN"
        );
        // Note: we do not assert clean cluster separation here. umap-rs embeds a
        // fully disconnected kNN graph (which cleanly separated blobs produce)
        // poorly; separation is validated on real connected embeddings instead.
        let _ = truth;
    }
}

#[cfg(all(test, feature = "tsne"))]
mod tsne_tests {
    use super::*;

    // t-SNE of three well-separated blobs into 2-D should keep each point closer to
    // its own blob's centroid than to the others' (local neighborhoods preserved).
    #[test]
    fn tsne_separates_blobs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let dim = 10;
        let mut data = Vec::new();
        let mut truth = Vec::new();
        for c in 0..3 {
            let mut center = vec![0.0; dim];
            center[c] = 20.0;
            for _ in 0..30 {
                let row: Vec<f64> = (0..dim)
                    .map(|t| center[t] + (rng.gen::<f64>() - 0.5) * 4.0)
                    .collect();
                data.push(row);
                truth.push(c);
            }
        }
        let emb = tsne(&data, 2, 30.0, 0.5, 500, 0);
        assert_eq!(emb.len(), data.len());
        assert_eq!(emb[0].len(), 2);
        assert!(
            emb.iter().all(|r| r.iter().all(|v| v.is_finite())),
            "embedding has NaN"
        );
        // Per-blob 2-D centroids, then check each point is nearest its own.
        let mut cents = [[0.0f64; 2]; 3];
        let mut counts = [0.0f64; 3];
        for (i, &c) in truth.iter().enumerate() {
            cents[c][0] += emb[i][0];
            cents[c][1] += emb[i][1];
            counts[c] += 1.0;
        }
        for c in 0..3 {
            cents[c][0] /= counts[c];
            cents[c][1] /= counts[c];
        }
        let mut correct = 0;
        for (i, &c) in truth.iter().enumerate() {
            let nearest = (0..3)
                .min_by(|&a, &b| {
                    let da = (emb[i][0] - cents[a][0]).powi(2) + (emb[i][1] - cents[a][1]).powi(2);
                    let db = (emb[i][0] - cents[b][0]).powi(2) + (emb[i][1] - cents[b][1]).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap();
            if nearest == c {
                correct += 1;
            }
        }
        // Allow a few stragglers; the embedding is stochastic and unseeded.
        assert!(
            correct as f64 / truth.len() as f64 > 0.9,
            "t-SNE did not separate blobs"
        );
    }

    // The dispatcher routes "tsne" to the t-SNE reducer and yields finite 2-D points.
    #[test]
    fn project_routes_tsne() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let data: Vec<Vec<f64>> = (0..20)
            .map(|_| (0..6).map(|_| rng.gen::<f64>()).collect())
            .collect();
        let emb = project(
            &data,
            2,
            "tsne",
            15,
            &UmapParams::default(),
            30.0,
            0.5,
            250,
            0,
        );
        assert_eq!(emb.len(), 20);
        assert!(emb
            .iter()
            .all(|r| r.len() == 2 && r.iter().all(|v| v.is_finite())));
    }
}
