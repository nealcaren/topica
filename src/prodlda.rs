//! ProdLDA, the AVITM autoencoding-variational topic model (Srivastava & Sutton,
//! "Autoencoding Variational Inference For Topic Models", ICLR 2017).
//!
//! ProdLDA is LDA with the word-level mixture `softmax(beta) . theta` replaced by a
//! *product of experts* `softmax(beta . theta)` with an unnormalized topic-word
//! matrix `beta`. Inference is amortized: an encoder network maps a document's
//! normalized bag of words to a logistic-normal posterior over `theta`, a
//! reparameterized sample is decoded, and the network is trained by minibatch Adam
//! on the ELBO. A new document gets its topics from one encoder forward pass.
//!
//! Two design choices follow the paper's prescription for avoiding *component
//! collapse* (topics decaying onto the prior early in training):
//!   - **Batch normalization** on the encoder mean/logvar heads and on the decoder
//!     logits. This is the structural difference from [`crate::etm_vae`]: batchnorm
//!     couples the documents in a minibatch, so the forward and backward passes run
//!     over the whole batch at once rather than per document. We use affine-free
//!     batchnorm (no learned scale/shift), matching the Pyro ProdLDA reference.
//!   - **High-momentum Adam** (`beta1 = 0.99`) with a Laplace approximation to the
//!     Dirichlet prior in the softmax basis (eq. 6 of the paper).
//!
//! ```text
//!   h1 = softplus(W1 xn + b1),  h2 = softplus(W2 h1 + b2)        (encoder, xn = x/sum x)
//!   mu = BN(W_mu h2 + b_mu),    logvar = BN(W_ls h2 + b_ls)       (K each, batchnorm)
//!   z  = mu + exp(logvar/2) * eps,  eps ~ N(0, I)                 (reparameterize)
//!   theta = softmax(z),  recon = softmax_v( BN(theta . beta) )    (product-of-experts decoder)
//!   loss = -sum_v x_v log recon_v                                 (reconstruction)
//!          + KL( N(mu, e^logvar) || N(mu_1, Sigma_1) )            (logistic-normal Laplace prior)
//! ```
//!
//! topica has no autodiff, so the batched forward and backward are hand-coded and
//! every gradient is checked against finite differences in the unit tests.

use rand::Rng;
use rayon::prelude::*;

/// Kaiming-uniform initialization matching PyTorch's `nn.Linear` default:
/// entries uniform on `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.
fn kaiming<R: Rng>(len: usize, fan_in: usize, rng: &mut R) -> Vec<f64> {
    let bound = 1.0 / (fan_in.max(1) as f64).sqrt();
    (0..len)
        .map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * bound)
        .collect()
}

/// A standard-normal sample via Box-Muller.
pub(crate) fn randn<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn softplus(x: f64) -> f64 {
    // log(1 + e^x), stable for large |x|.
    x.max(0.0) + (-(x.abs())).exp().ln_1p()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

const BN_EPS: f64 = 1e-5;

/// Trainable-free batch-normalization layer (affine = false): it stores only the
/// running mean/variance used at evaluation time.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchNorm {
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub momentum: f64,
}

/// What the BN backward pass needs: the normalized activations and the per-feature
/// inverse standard deviation computed from the *batch* statistics.
struct BnCache {
    xhat: Vec<Vec<f64>>, // N x F
    inv_std: Vec<f64>,   // F
}

impl BatchNorm {
    pub(crate) fn new(f: usize) -> Self {
        BatchNorm {
            running_mean: vec![0.0; f],
            running_var: vec![1.0; f],
            momentum: 0.1,
        }
    }

    /// Forward over a minibatch `x` (N x F) using batch statistics. Returns the
    /// normalized output, the backward cache, and the batch (mean, var) so the
    /// caller can fold them into the running statistics.
    fn forward_train(&self, x: &[Vec<f64>]) -> (Vec<Vec<f64>>, BnCache, Vec<f64>, Vec<f64>) {
        let n = x.len();
        // Bessel's correction for the running variance is undefined for a single
        // sample, so a batch of one has no well-defined running-stat update. Every
        // caller already skips batches smaller than two (chunks(batch_size.max(2))
        // + an `if n < 2 { continue }` guard); enforce that invariant loudly here
        // rather than silently decaying the running variance toward zero (#436).
        assert!(
            n >= 2,
            "BatchNorm::forward_train requires a batch of >= 2 documents; \
             callers must skip smaller batches"
        );
        let f = if n > 0 { x[0].len() } else { 0 };
        let mut mean = vec![0.0; f];
        for row in x {
            for j in 0..f {
                mean[j] += row[j];
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut var = vec![0.0; f];
        for row in x {
            for j in 0..f {
                let d = row[j] - mean[j];
                var[j] += d * d;
            }
        }
        for v in &mut var {
            *v /= n as f64;
        }
        let inv_std: Vec<f64> = var.iter().map(|&v| 1.0 / (v + BN_EPS).sqrt()).collect();
        let mut xhat = vec![vec![0.0; f]; n];
        let mut out = vec![vec![0.0; f]; n];
        for i in 0..n {
            for j in 0..f {
                let h = (x[i][j] - mean[j]) * inv_std[j];
                xhat[i][j] = h;
                out[i][j] = h;
            }
        }
        // The in-batch normalization above uses the biased variance (÷ n), but the
        // running statistic must track the UNBIASED (Bessel-corrected, ÷ (n-1))
        // variance — that is what PyTorch/Pyro BatchNorm accumulates, and eval-time
        // normalization reads the running variance. Returning the biased var here
        // left the running variance systematically too small (× (n-1)/n), so a
        // topica-trained model normalized held-out documents differently from the
        // reference at inference. n >= 2 is guaranteed by the assert above, so the
        // Bessel factor cannot divide by zero.
        let bessel = n as f64 / (n as f64 - 1.0);
        let running_var: Vec<f64> = var.iter().map(|&v| v * bessel).collect();
        (out, BnCache { xhat, inv_std }, mean, running_var)
    }

    /// Fold a batch's statistics into the running estimates.
    pub(crate) fn update_running(&mut self, mean: &[f64], var: &[f64]) {
        let m = self.momentum;
        for j in 0..self.running_mean.len() {
            self.running_mean[j] = (1.0 - m) * self.running_mean[j] + m * mean[j];
            self.running_var[j] = (1.0 - m) * self.running_var[j] + m * var[j];
        }
    }

    /// Evaluation-time normalization of a single row, using running statistics.
    fn forward_eval_row(&self, x: &[f64]) -> Vec<f64> {
        (0..x.len())
            .map(|j| (x[j] - self.running_mean[j]) / (self.running_var[j] + BN_EPS).sqrt())
            .collect()
    }

    /// Backward through affine-free batchnorm. `dy` is the upstream gradient
    /// (N x F); returns the gradient w.r.t. the layer input (N x F).
    fn backward(dy: &[Vec<f64>], cache: &BnCache) -> Vec<Vec<f64>> {
        let n = dy.len();
        let f = if n > 0 { dy[0].len() } else { 0 };
        let nf = n as f64;
        let mut dx = vec![vec![0.0; f]; n];
        for j in 0..f {
            let mut sum_dy = 0.0;
            let mut sum_dy_xhat = 0.0;
            for i in 0..n {
                sum_dy += dy[i][j];
                sum_dy_xhat += dy[i][j] * cache.xhat[i][j];
            }
            for i in 0..n {
                dx[i][j] = cache.inv_std[j]
                    * (dy[i][j] - sum_dy / nf - cache.xhat[i][j] * sum_dy_xhat / nf);
            }
        }
        dx
    }
}

/// Which inputs feed the encoder's first layer. The decoder, prior, KL,
/// reparameterization, batchnorm, and BoW reconstruction loss are identical in
/// every mode; only layer 1's input changes.
///
/// - [`InputMode::BowOnly`] is plain ProdLDA: input is the normalized bag of
///   words (sparse, length `V`); `w1` is `hidden x V` and `e == 0`.
/// - [`InputMode::BowEmb`] is the simple-concatenation encoder used by SCHOLAR
///   (`src/scholar.rs`): the normalized bag of words is concatenated with a dense
///   auxiliary channel (length `E`) fed directly, no projection; `w1` is
///   `hidden x (V + E)`, with the BoW columns sparse and the auxiliary columns
///   dense. SCHOLAR rides its covariates in on this channel.
/// - [`InputMode::BowEmbAdapt`] is CombinedTM faithful to Bianchi, Terragni & Hovy
///   (2021): the contextual (SBERT) embedding is first projected into vocabulary
///   space by a learned `adapt_bert` layer (`w_adapt`, `V x E`), then concatenated
///   with the *raw* bag of words, so the encoder's first layer is
///   `Linear(2V, hidden)` (`w1` is `hidden x 2V`). This matches upstream
///   `CombinedInferenceNetwork` (`adapt_bert = nn.Linear(bert_size, input_size)`,
///   `input_layer = nn.Linear(input_size + input_size, hidden)`, raw `CountVectorizer`
///   BoW).
/// - [`InputMode::EmbOnly`] is ZeroShotTM: input is the document embedding alone;
///   `w1` is `hidden x E` (its `V`-column block is unused at the encoder, though
///   `beta` still reconstructs the `V`-word BoW).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum InputMode {
    BowOnly,
    BowEmb,
    BowEmbAdapt,
    EmbOnly,
}

/// The trainable parameters: encoder (`input -> hidden -> hidden -> (mu, logvar)`)
/// and the unnormalized decoder `beta` (K x V, row-major). The encoder's first
/// weight matrix `w1` is `hidden x (V + E)`: the first `V` columns multiply the
/// sparse normalized bag of words, the next `E` columns the dense document
/// embedding. With `e == 0` this is exactly the bag-of-words ProdLDA encoder.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Weights {
    pub v: usize,
    pub e: usize,
    pub hidden: usize,
    pub k: usize,
    pub mode: InputMode,
    pub w1: Vec<f64>,   // hidden x (V + E)
    pub b1: Vec<f64>,   // hidden
    pub w2: Vec<f64>,   // hidden x hidden
    pub b2: Vec<f64>,   // hidden
    pub w_mu: Vec<f64>, // K x hidden
    pub b_mu: Vec<f64>, // K
    pub w_ls: Vec<f64>, // K x hidden
    pub b_ls: Vec<f64>, // K
    pub beta: Vec<f64>, // K x V
    /// CombinedTM `adapt_bert`: the learned linear projection of the contextual
    /// embedding into vocabulary space (`V x E`, row-major), applied before
    /// concatenation with the raw bag of words (Bianchi et al. 2021). Non-empty only
    /// for [`InputMode::BowEmbAdapt`]; empty (and unused) for every other mode, so
    /// the bow-only, simple-concat, and embedding-only encoders are unchanged.
    /// `#[serde(default)]` keeps pre-CombinedTM-projection JSON states loadable.
    #[serde(default)]
    pub w_adapt: Vec<f64>, // V x E (BowEmbAdapt only)
    /// Bias for `adapt_bert` (`V`); non-empty only for [`InputMode::BowEmbAdapt`].
    #[serde(default)]
    pub b_adapt: Vec<f64>, // V (BowEmbAdapt only)
}

impl Weights {
    /// Build encoder/decoder weights for the given input mode. `fan_in` for `w1`
    /// is the encoder's input width: `V` (bow-only), `V + E` (bow+emb), or `E`
    /// (emb-only), matching the reference's `Linear(input_size, hidden)`. The RNG
    /// draw order is identical to the bow-only path when `e == 0`.
    pub(crate) fn new<R: Rng>(
        v: usize,
        e: usize,
        hidden: usize,
        k: usize,
        mode: InputMode,
        rng: &mut R,
    ) -> Self {
        // Encoder layer-1 input width and its `kaiming` fan-in. The bow-only,
        // simple-concat (`BowEmb`), and emb-only paths keep `w1` at `hidden x (V + E)`
        // (the emb-only path leaves its BoW block unused); CombinedTM's faithful
        // encoder concatenates the raw BoW (`V`) with the `adapt_bert`-projected
        // embedding (also `V`), so its first layer is `Linear(2V, hidden)`.
        let (cols, fan_in) = match mode {
            InputMode::BowOnly => (v + e, v),
            InputMode::BowEmb => (v + e, v + e),
            InputMode::BowEmbAdapt => (v + v, v + v),
            InputMode::EmbOnly => (v + e, e),
        };
        let w1 = kaiming(hidden * cols, fan_in, rng);
        // `adapt_bert = nn.Linear(E, V)` is drawn only for CombinedTM, so the RNG
        // stream (and therefore the fitted weights) of every other mode is
        // byte-identical to before this layer existed.
        let (w_adapt, b_adapt) = if mode == InputMode::BowEmbAdapt {
            (kaiming(v * e, e, rng), vec![0.0; v])
        } else {
            (Vec::new(), Vec::new())
        };
        Weights {
            v,
            e,
            hidden,
            k,
            mode,
            w1,
            b1: vec![0.0; hidden],
            w2: kaiming(hidden * hidden, hidden, rng),
            b2: vec![0.0; hidden],
            w_mu: kaiming(k * hidden, hidden, rng),
            b_mu: vec![0.0; k],
            w_ls: kaiming(k * hidden, hidden, rng),
            b_ls: vec![0.0; k],
            beta: kaiming(k * v, k, rng),
            w_adapt,
            b_adapt,
        }
    }

    /// Encoder layer-1 input width, matching the reference `Linear(input_size, hidden)`
    /// per mode: `V + E` for bow-only / simple-concat / emb-only (the emb-only BoW
    /// block is unused), and `2V` for CombinedTM's `adapt_bert` encoder.
    fn w1_cols(&self) -> usize {
        match self.mode {
            InputMode::BowEmbAdapt => self.v + self.v,
            _ => self.v + self.e,
        }
    }

    /// CombinedTM `adapt_bert` forward: project the contextual embedding `emb`
    /// (length `E`) into vocabulary space, `adapted[j] = b_adapt[j] + sum_k
    /// w_adapt[j, k] * emb[k]` (a plain linear layer, no activation, as in the
    /// reference). Returns a length-`V` vector.
    fn adapt_bert(&self, emb: &[f64]) -> Vec<f64> {
        let mut adapted = self.b_adapt.clone();
        for (j, a) in adapted.iter_mut().enumerate() {
            let base = j * self.e;
            let mut s = *a;
            for (kk, &ev) in emb.iter().enumerate() {
                s += self.w_adapt[base + kk] * ev;
            }
            *a = s;
        }
        adapted
    }

    /// Encoder forward for one document up to the pre-batchnorm head outputs,
    /// retaining the activations needed for the backward pass. `xn` is the sparse
    /// normalized bag of words; `emb` is the dense document embedding (length `E`,
    /// empty for bow-only). Layer 1's input depends on the mode: bow-only uses the
    /// BoW columns of `w1`, emb-only the embedding columns, bow+emb both.
    /// Encoder layer 1 for one document: the sparse BoW ⊕ dense second-block matvec
    /// through `w1`, plus its softplus. Returns `(pre1, h1, adapted)` where `adapted`
    /// is the CombinedTM `adapt_bert` projection (empty for other modes). This is the
    /// mode-dependent, sparse part of the encoder; layers 2 and the heads are dense
    /// and are batched into GEMMs by `batch_forward` (issue #378). `encode_raw` keeps
    /// them per-document for single-document inference (`transform`).
    fn encode_layer1(&self, xn: &[(usize, f64)], emb: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let h = self.hidden;
        let cols = self.w1_cols();
        // CombinedTM: project the embedding into vocab space before concatenation.
        let adapted = if self.mode == InputMode::BowEmbAdapt {
            self.adapt_bert(emb)
        } else {
            Vec::new()
        };
        // Layer 1: the BoW part is sparse in the vocabulary, the second block dense.
        // For `BowEmbAdapt` the second block is the `V`-wide adapted embedding; for
        // `BowEmb`/`EmbOnly` it is the `E`-wide raw channel.
        let mut pre1 = self.b1.clone();
        for i in 0..h {
            let row = i * cols;
            let mut s = pre1[i];
            if self.mode != InputMode::EmbOnly {
                for &(w, val) in xn {
                    s += self.w1[row + w] * val;
                }
            }
            match self.mode {
                InputMode::BowOnly => {}
                InputMode::BowEmbAdapt => {
                    let base = row + self.v;
                    for (j, &av) in adapted.iter().enumerate() {
                        s += self.w1[base + j] * av;
                    }
                }
                InputMode::BowEmb | InputMode::EmbOnly => {
                    let base = row + self.v;
                    for (j, &ev) in emb.iter().enumerate() {
                        s += self.w1[base + j] * ev;
                    }
                }
            }
            pre1[i] = s;
        }
        let h1: Vec<f64> = pre1.iter().map(|&p| softplus(p)).collect();
        (pre1, h1, adapted)
    }

    /// Full per-document encoder (layer 1 + layer 2 + heads). Kept for single-document
    /// inference (`transform`) and tests; `batch_forward` batches layers 2 and the
    /// heads into GEMMs instead (issue #378), but produces the same `DocCache`.
    fn encode_raw(&self, xn: &[(usize, f64)], emb: &[f64], mask2: &[f64]) -> DocCache {
        let (h, k) = (self.hidden, self.k);
        let (pre1, h1, adapted) = self.encode_layer1(xn, emb);
        // Layer 2 dense.
        let mut pre2 = self.b2.clone();
        for i in 0..h {
            let row = i * h;
            let mut s = pre2[i];
            for j in 0..h {
                s += self.w2[row + j] * h1[j];
            }
            pre2[i] = s;
        }
        // Dropout on softplus(h2) (inverted: mask carries the 1/keep scale).
        let hd: Vec<f64> = (0..h).map(|i| softplus(pre2[i]) * mask2[i]).collect();
        // Heads (pre-batchnorm).
        let mut mu_raw = self.b_mu.clone();
        let mut lv_raw = self.b_ls.clone();
        for c in 0..k {
            let row = c * h;
            let (mut sm, mut sl) = (mu_raw[c], lv_raw[c]);
            for i in 0..h {
                sm += self.w_mu[row + i] * hd[i];
                sl += self.w_ls[row + i] * hd[i];
            }
            mu_raw[c] = sm;
            lv_raw[c] = sl;
        }
        DocCache {
            pre1,
            h1,
            pre2,
            hd,
            mu_raw,
            lv_raw,
            adapted,
        }
    }
}

/// Per-document encoder activations retained for the backward pass.
struct DocCache {
    pre1: Vec<f64>,
    h1: Vec<f64>,
    pre2: Vec<f64>,
    hd: Vec<f64>,
    mu_raw: Vec<f64>,
    lv_raw: Vec<f64>,
    /// CombinedTM `adapt_bert` output (length `V`); empty for every other mode.
    adapted: Vec<f64>,
}

/// Gradient accumulators mirroring [`Weights`].
pub(crate) struct Grad {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w_mu: Vec<f64>,
    b_mu: Vec<f64>,
    w_ls: Vec<f64>,
    b_ls: Vec<f64>,
    /// Topic-word (decoder) gradient, K x V. Exposed so a coupled model
    /// (`infoctm`) can add a cross-lingual alignment gradient before the Adam step.
    pub(crate) beta: Vec<f64>,
    /// CombinedTM `adapt_bert` gradients (`V x E` and `V`); empty for other modes.
    w_adapt: Vec<f64>,
    b_adapt: Vec<f64>,
}

impl Grad {
    /// The `w_mu` encoder-head gradient block. Exposed `pub(crate)` so `scholar`'s
    /// content-interaction FD test can confirm the interaction's gradient into `theta`
    /// reaches the encoder.
    #[cfg(test)]
    pub(crate) fn w_mu(&self) -> &[f64] {
        &self.w_mu
    }

    pub(crate) fn zeros(w: &Weights) -> Self {
        Grad {
            w1: vec![0.0; w.w1.len()],
            b1: vec![0.0; w.b1.len()],
            w2: vec![0.0; w.w2.len()],
            b2: vec![0.0; w.b2.len()],
            w_mu: vec![0.0; w.w_mu.len()],
            b_mu: vec![0.0; w.b_mu.len()],
            w_ls: vec![0.0; w.w_ls.len()],
            b_ls: vec![0.0; w.b_ls.len()],
            beta: vec![0.0; w.beta.len()],
            w_adapt: vec![0.0; w.w_adapt.len()],
            b_adapt: vec![0.0; w.b_adapt.len()],
        }
    }
    pub(crate) fn scale(&mut self, s: f64) {
        for blk in [
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.w_mu,
            &mut self.b_mu,
            &mut self.w_ls,
            &mut self.b_ls,
            &mut self.beta,
            &mut self.w_adapt,
            &mut self.b_adapt,
        ] {
            for x in blk.iter_mut() {
                *x *= s;
            }
        }
    }
}

/// The Laplace approximation to a Dirichlet(`alpha`) prior in the softmax basis
/// (eq. 6): a diagonal logistic-normal with mean `mu_1` and variance `Sigma_1`.
pub(crate) fn laplace_prior(alpha: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = alpha.len();
    let kf = k as f64;
    let mean_log: f64 = alpha.iter().map(|&a| a.ln()).sum::<f64>() / kf;
    let sum_inv: f64 = alpha.iter().map(|&a| 1.0 / a).sum();
    let mu1: Vec<f64> = alpha.iter().map(|&a| a.ln() - mean_log).collect();
    let var1: Vec<f64> = alpha
        .iter()
        .map(|&a| (1.0 / a) * (1.0 - 2.0 / kf) + sum_inv / (kf * kf))
        .collect();
    (mu1, var1)
}

/// Euler-Mascheroni constant, used in the Weibull-to-Gamma KL.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Standard-normal CDF via `erf`, used on the Dirichlet path to turn the per-batch
/// Gaussian reparameterization noise `eps` into uniform draws `u = Phi(eps)`. We
/// reuse the *same* noise the laplace path draws so the RNG stream (and therefore
/// the laplace path) is byte-identical; only the transform of that noise changes.
pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// `erf` via the Abramowitz-Stegun 7.1.26 rational approximation (|err| < 1.5e-7),
/// which is ample next to the 1e-4 FD tolerance. The noise transform it feeds is
/// constant with respect to the trainable parameters, so it never enters a gradient.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// `ln Γ(x)` (Lanczos approximation, x > 0). Used in the Weibull-to-Gamma KL,
/// where the prior's `ln Γ(α)` term is a constant offset but is kept for a true KL.
fn lgamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx).
        let pi = std::f64::consts::PI;
        (pi / (pi * x).sin()).ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Which prior the VAE path places on the document-topic vector.
///
/// - [`Prior::Laplace`] is the unchanged logistic-normal Laplace approximation to
///   `Dirichlet(alpha)` in the softmax basis (Srivastava & Sutton 2017, eq. 6):
///   `theta = softmax(mu + exp(logvar/2) * eps)` with the diagonal logistic-normal
///   KL.
/// - [`Prior::Dirichlet`] is a true `Dirichlet(alpha)` prior via the Weibull
///   reparameterization (Zhang et al. 2018; Burkhardt & Kramer 2019): the two
///   encoder heads parameterize a Weibull variational posterior on each
///   unnormalized topic weight, a Weibull sample is normalized onto the simplex to
///   give `theta`, and the analytic Weibull-to-Gamma KL replaces the
///   logistic-normal KL. The Gaussian reparameterization is replaced too; we reuse
///   the same Gaussian noise turned into uniforms by `Phi(eps)` so the laplace path
///   is unaffected.
/// - [`Prior::StickBreaking`] is the Gaussian stick-breaking construction (Miao,
///   Grefenstette & Blunsom 2017, "GSB"; the reparameterizable simplex map of
///   Nalisnick & Smyth 2017). It keeps the *same* Gaussian latent and Gaussian KL
///   as `Prior::Laplace` — only the map onto the simplex changes: instead of
///   `softmax(z)`, the `K-1` breaks `eta_t = sigmoid(z_t)` are turned into topic
///   proportions by stick-breaking (`theta_t = eta_t * prod_{j<t}(1 - eta_j)`, with
///   the last stick the remainder). The construction is nonparametric-flavored: the
///   ordered sticks let early topics claim most mass and later ones decay, softening
///   the fixed-`K` assumption. Because the latent and KL are unchanged, the laplace
///   path stays byte-identical.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Prior {
    Laplace,
    Dirichlet,
    StickBreaking,
}

/// Gaussian stick-breaking map: turn a latent vector `z` (length `K`) into topic
/// proportions on the simplex. The first `K-1` entries are breaks
/// `eta_t = sigmoid(z_t)`; `theta_t = eta_t * prod_{j<t}(1 - eta_j)` and the last
/// topic takes the remaining stick. Returns `(eta, theta)`; `eta[K-1]` is unused
/// (left 0) and `theta` sums to 1 by construction. Shared by `prodlda` and `etm_vae`.
pub(crate) fn stick_break(z: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = z.len();
    let mut eta = vec![0.0; k];
    let mut theta = vec![0.0; k];
    if k == 0 {
        return (eta, theta);
    }
    let mut r = 1.0; // remaining stick R_t = prod_{j<t}(1 - eta_j)
    for t in 0..k - 1 {
        let e = sigmoid(z[t]);
        eta[t] = e;
        theta[t] = e * r;
        r *= 1.0 - e;
    }
    theta[k - 1] = r;
    (eta, theta)
}

/// Backward of [`stick_break`]: map `dtheta` (grad w.r.t. the proportions) to `dz`
/// (grad w.r.t. the latent), folding in the `sigmoid` derivative. `eta` and `theta`
/// are the forward values. With `R_j = prod_{i<j}(1 - eta_i)` and the suffix sum
/// `S_j = sum_{t>j} dtheta_t * theta_t`, the break gradient is
/// `deta_j = dtheta_j R_j - S_j / (1 - eta_j)`; multiplying by the sigmoid Jacobian
/// `eta_j (1 - eta_j)` cancels the division, giving
/// `dz_j = eta_j (1 - eta_j) dtheta_j R_j - eta_j S_j`. `dz[K-1] = 0` (unused break).
pub(crate) fn stick_break_dz(dtheta: &[f64], eta: &[f64], theta: &[f64]) -> Vec<f64> {
    let k = dtheta.len();
    let mut dz = vec![0.0; k];
    if k < 2 {
        return dz;
    }
    // suffix[j] = sum_{t>=j} dtheta_t * theta_t, so S_j = suffix[j+1].
    let mut suffix = vec![0.0; k + 1];
    for t in (0..k).rev() {
        suffix[t] = suffix[t + 1] + dtheta[t] * theta[t];
    }
    let mut r = 1.0;
    for j in 0..k - 1 {
        let e = eta[j];
        let s_j = suffix[j + 1];
        dz[j] = e * (1.0 - e) * dtheta[j] * r - e * s_j;
        r *= 1.0 - e;
    }
    dz
}

/// The two orthogonal `fit_avitm` options (#174, #176). Both default to the
/// current code path: `Prior::Laplace` and `contrastive == false` reproduce the
/// pre-change forward/backward exactly, so the defaults are bit-identical to the
/// implementation before these flags existed.
#[derive(Clone, Copy)]
pub struct AvitmOptions {
    pub prior: Prior,
    /// InfoNCE contrastive regularization on the topic vectors (CLNTM-style,
    /// Nguyen & Luu 2021). Off by default.
    pub contrastive: bool,
    /// Scale on the contrastive term added to the per-batch loss.
    pub contrastive_weight: f64,
    /// InfoNCE temperature.
    pub contrastive_temp: f64,
}

impl Default for AvitmOptions {
    fn default() -> Self {
        AvitmOptions {
            prior: Prior::Laplace,
            contrastive: false,
            contrastive_weight: 0.5,
            contrastive_temp: 0.5,
        }
    }
}

/// Inputs and noise for one batch, gathered so the forward and the
/// finite-difference loss can be recomputed identically.
pub(crate) struct Batch<'a> {
    pub(crate) xns: Vec<&'a [(usize, f64)]>,
    pub(crate) embs: Vec<&'a [f64]>,
    pub(crate) counts: Vec<&'a [(usize, f64)]>,
    pub(crate) totals: Vec<f64>,
    pub(crate) eps: &'a [Vec<f64>],
    pub(crate) masks2: &'a [Vec<f64>],
    pub(crate) masks_t: &'a [Vec<f64>],
    /// Per-document prior mean `mu_0[i]` (length `K`) for the Gaussian KL. `None`
    /// means every document shares the `prior_mu` slice passed to
    /// `batch_forward`/`batch_backward` — the ProdLDA/InfoCTM path, byte-identical.
    /// `Some` is SCHOLAR's covariate-dependent prior mean `mu_0[i] = W . PC[i]`
    /// (`src/scholar.rs`), which shifts topic prevalence by document metadata. Only
    /// the mean varies; the prior variance stays the shared Laplace `prior_var`.
    pub(crate) prior_mus: Option<&'a [Vec<f64>]>,
}

/// SCHOLAR content (topic-covariate) decoder deviation (`src/scholar.rs`). The
/// unnormalized decoder logit gains `sum_c beta_c[c]*TC[i][c]` (per-covariate word
/// shifts, the SAGE mechanism) and, when `beta_ci` is set, the topic-covariate
/// interaction `sum_{t,c} beta_ci[t,c]*theta_do[i][t]*TC[i][c]`. `None` on the shared
/// `batch_forward`/`batch_backward` leaves the decoder untouched (ProdLDA/InfoCTM and
/// the covariate/label Scholar paths stay byte-identical).
pub(crate) struct ContentFwd<'a> {
    /// Topic-covariate rows, batch order, `n x c`.
    pub(crate) tc: &'a [Vec<f64>],
    /// Per-covariate word deviations `beta_c`, `c x V` row-major.
    pub(crate) beta_c: &'a [f64],
    /// Topic-covariate interaction weights `beta_ci`, `(K*c) x V` row-major, or `None`.
    pub(crate) beta_ci: Option<&'a [f64]>,
    /// Number of topic covariates `c`.
    pub(crate) c: usize,
}

/// Gradient accumulators for the content deviation, mirroring [`ContentFwd`].
pub(crate) struct ContentGrad {
    pub(crate) beta_c: Vec<f64>,
    pub(crate) beta_ci: Option<Vec<f64>>,
}

/// Caches retained from the batch forward for the backward pass.
pub(crate) struct BatchCache {
    doc: Vec<DocCache>,
    bn_mu: BnCache,
    bn_lv: BnCache,
    bn_dec: BnCache,
    mu: Vec<Vec<f64>>,       // N x K (post-BN)
    lv: Vec<Vec<f64>>,       // N x K (post-BN)
    theta: Vec<Vec<f64>>,    // N x K
    theta_do: Vec<Vec<f64>>, // N x K (post-dropout)
    recon: Vec<Vec<f64>>,    // N x V
    // Dirichlet (Weibull) reparameterization scratch, empty on the laplace path.
    // For each (doc, topic): Weibull shape `kw`, scale `lam`, the constant
    // `L = -ln(1-u)` from the noise, and the unnormalized weight `g = lam * L^(1/kw)`.
    dir: Option<DirCache>,
    // Stick-breaking scratch (the `eta = sigmoid(z)` breaks), empty off that path.
    // Needed by the backward to recompute the stick-breaking Jacobian.
    sb: Option<SbCache>,
    // Contrastive positive-view topic vectors (the no-noise / posterior-mean theta),
    // and the per-doc softmax used to build them, retained so the backward can push
    // the InfoNCE gradient through both views. Empty when `contrastive == false`.
    contrast: Option<ContrastCache>,
}

impl BatchCache {
    /// The sampled topic vectors `theta` (N x K), before the decoder dropout mask.
    /// Exposed so a coupled head (SCHOLAR's label classifier) can read the same
    /// `theta` it is defined to classify from and push its gradient back through
    /// `batch_backward`'s `dtheta_extra`.
    pub(crate) fn theta(&self) -> &[Vec<f64>] {
        &self.theta
    }
}

/// Per-(doc, topic) stick-breaking quantities: the `eta = sigmoid(z)` breaks for the
/// Gaussian stick-breaking prior. `theta` lives in `BatchCache.theta`.
struct SbCache {
    eta: Vec<Vec<f64>>, // N x K (entry K-1 unused)
}

/// Per-(doc, topic) Weibull reparameterization quantities for the Dirichlet prior.
struct DirCache {
    kw: Vec<Vec<f64>>,   // N x K Weibull shape  = softplus(mu) + floor
    lam: Vec<Vec<f64>>,  // N x K Weibull scale  = softplus(lv) + floor
    ln_l: Vec<Vec<f64>>, // N x K  ln L, L = -ln(1-u), u = Phi(eps) (constant in params)
    g: Vec<Vec<f64>>,    // N x K unnormalized weight g = lam * L^(1/kw)
    s: Vec<f64>,         // N      normalizer sum_t g
}

/// The contrastive positive view: a second deterministic topic vector per document.
/// On the laplace path it is `softmax(mu)` (the encoder mean, no sampling noise); on
/// the Dirichlet path it is the normalized Weibull at the posterior median (`u=0.5`).
struct ContrastCache {
    theta_pos: Vec<Vec<f64>>, // N x K positive-view topic vectors
    // Dirichlet-only positive-view weights, mirroring DirCache (empty on laplace).
    g_pos: Vec<Vec<f64>>, // N x K
    s_pos: Vec<f64>,      // N
}

/// Forward pass over a whole minibatch (batchnorm uses batch statistics). Returns
/// the summed loss (reconstruction + KL) and the backward cache, plus the BN batch
/// statistics so the running estimates can be updated by the caller.
/// Floor added to the Weibull scale so it stays strictly positive.
pub(crate) const WEIBULL_FLOOR: f64 = 1e-4;
/// Floor on the Weibull *shape*. A Weibull with small shape has a very large
/// coefficient of variation, so the reparameterized topic vector is too noisy to
/// train; flooring the shape at 1 keeps the variational posterior concentrated
/// enough to learn (a standard WHAI-style choice) while staying strictly positive.
/// This is deliberately a family restriction, not just a numerical floor: it caps
/// how sparse the posterior can be, so a `Dirichlet(alpha)` prior with `alpha < 1`
/// is approximated by a shape-1 (exponential) posterior rather than a matching
/// sub-1 shape. Loosen it only alongside re-validating stability (#428, finding 3).
pub(crate) const WEIBULL_SHAPE_FLOOR: f64 = 1.0;

/// Map a post-BN head pair `(a, b)` to a Weibull `(shape, scale)` and, given the
/// constant `ln_l = ln(-ln(1-u))`, the unnormalized weight `g = scale * L^(1/shape)`.
/// Returns `(kw, lam, g)`. Shared by the sampled and positive (median) views so the
/// reparameterization is identical up to the noise.
pub(crate) fn weibull_weight(a: f64, b: f64, ln_l: f64) -> (f64, f64, f64) {
    let kw = softplus(a) + WEIBULL_SHAPE_FLOOR;
    let lam = softplus(b) + WEIBULL_FLOOR;
    let g = lam * (ln_l / kw).exp(); // lam * L^(1/kw) = lam * exp(ln_l / kw)
    (kw, lam, g)
}

/// The normalized Weibull weights at the posterior median (`u = 0.5`, so
/// `L = ln 2`): the deterministic θ point estimate the `Prior::Dirichlet` forward
/// trains and (for the contrastive view) evaluates under. `mu`/`lv` are the
/// post-batchnorm encoder heads. Kept identical to the training median so
/// `transform` reports the same estimate the model was optimized against.
pub(crate) fn weibull_median_theta(mu: &[f64], lv: &[f64]) -> Vec<f64> {
    let k = mu.len();
    let ln_l = (-(0.5f64).ln()).ln(); // u = 0.5  =>  L = ln 2
    let g: Vec<f64> = (0..k)
        .map(|t| weibull_weight(mu[t], lv[t], ln_l).2)
        .collect();
    let s: f64 = g.iter().sum();
    if s > 0.0 {
        g.iter().map(|x| x / s).collect()
    } else {
        vec![1.0 / k as f64; k]
    }
}

/// Analytic KL( Weibull(kw, lam) || Gamma(alpha, rate=1) ), the per-topic term of
/// the Weibull-Dirichlet prior (Zhang et al. 2018, "WHAI", appendix; Burkhardt &
/// Kramer 2019). A symmetric `Dirichlet(alpha)` prior on the simplex factorizes
/// into independent `Gamma(alpha, 1)` priors on the unnormalized weights, so the KL
/// sums these per-topic terms.
pub(crate) fn weibull_gamma_kl(kw: f64, lam: f64, alpha: f64) -> f64 {
    EULER_GAMMA * alpha / kw - alpha * lam.ln() + kw.ln() + lam * lgamma(1.0 + 1.0 / kw).exp()
        - EULER_GAMMA
        - 1.0
        + lgamma(alpha)
}

/// d/d(kw, lam) of [`weibull_gamma_kl`]. `Γ(1 + 1/kw)` enters through both its value
/// and its derivative `-Γ(1+1/kw) ψ(1+1/kw) / kw^2` w.r.t. `kw` (chain rule on the
/// `1/kw` argument), where `ψ` is the digamma function.
pub(crate) fn weibull_gamma_kl_grad(kw: f64, lam: f64, alpha: f64) -> (f64, f64) {
    let arg = 1.0 + 1.0 / kw;
    let gam = lgamma(arg).exp();
    let dgam_dkw = gam * digamma(arg) * (-1.0 / (kw * kw));
    // d/dkw: -γα/kw^2 + 1/kw + lam * dΓ/dkw
    let dkw = -EULER_GAMMA * alpha / (kw * kw) + 1.0 / kw + lam * dgam_dkw;
    // d/dlam: -alpha/lam + Γ(1+1/kw)
    let dlam = -alpha / lam + gam;
    (dkw, dlam)
}

/// Digamma `ψ(x)` for `x > 0` (asymptotic series with recurrence shift). Accurate
/// well within the 1e-4 FD tolerance.
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// InfoNCE contrastive loss (CLNTM, Nguyen & Luu 2021) over a minibatch of topic
/// vectors. The anchor for document `i` is its sampled topic vector `z[i]`; the
/// positive is the second deterministic view `z_pos[i]`; the negatives are the other
/// documents' sampled topic vectors. Similarity is cosine, scaled by `temp`:
///   `L = -(1/N) Σ_i log[ exp(sim(z_i, z_i^+)/τ) / (exp(sim(z_i,z_i^+)/τ) + Σ_{j≠i} exp(sim(z_i,z_j)/τ)) ]`.
pub(crate) fn info_nce_loss(z: &[Vec<f64>], z_pos: &[Vec<f64>], temp: f64) -> f64 {
    let n = z.len();
    let mut total = 0.0;
    for i in 0..n {
        let pos = cosine(&z[i], &z_pos[i]) / temp;
        let mut max_logit = pos;
        let mut negs = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j != i {
                let s = cosine(&z[i], &z[j]) / temp;
                if s > max_logit {
                    max_logit = s;
                }
                negs.push(s);
            }
        }
        // log-sum-exp over {positive} ∪ {negatives}, stabilized.
        let mut denom = (pos - max_logit).exp();
        for &s in &negs {
            denom += (s - max_logit).exp();
        }
        total += -(pos - max_logit - denom.ln());
    }
    total / n as f64
}

/// Cosine similarity of two K-vectors.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for t in 0..a.len() {
        dot += a[t] * b[t];
        na += a[t] * a[t];
        nb += b[t] * b[t];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Gradient of `cosine(a, b)` w.r.t. `a` (returns a K-vector). By symmetry the
/// gradient w.r.t. `b` is `cosine_grad_a(b, a)`.
fn cosine_grad_a(a: &[f64], b: &[f64]) -> Vec<f64> {
    let k = a.len();
    let mut dot = 0.0;
    let mut na2 = 0.0;
    let mut nb2 = 0.0;
    for t in 0..k {
        dot += a[t] * b[t];
        na2 += a[t] * a[t];
        nb2 += b[t] * b[t];
    }
    let na = na2.sqrt();
    let nb = nb2.sqrt();
    let denom = na * nb + 1e-12;
    // d/da [ dot / (|a||b| + e) ] = b/denom - dot * (|b| a/|a|) / denom^2.
    let mut out = vec![0.0; k];
    let coef = dot * nb / (na.max(1e-12)) / (denom * denom);
    for t in 0..k {
        out[t] = b[t] / denom - coef * a[t];
    }
    out
}

/// Backward of [`info_nce_loss`]. Returns `(dz, dz_pos)`, the gradients of the mean
/// InfoNCE loss w.r.t. the anchor vectors `z` and the positive-view vectors `z_pos`.
/// Each anchor `z_i` appears both as its own anchor and as a negative for every
/// other document, so its gradient accumulates both contributions.
pub(crate) fn info_nce_backward(
    z: &[Vec<f64>],
    z_pos: &[Vec<f64>],
    temp: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = z.len();
    let k = if n > 0 { z[0].len() } else { 0 };
    let mut dz = vec![vec![0.0; k]; n];
    let mut dz_pos = vec![vec![0.0; k]; n];
    let scale = 1.0 / n as f64;
    for i in 0..n {
        // Logits sim/τ for the positive and each negative j; softmax probabilities.
        let pos_sim = cosine(&z[i], &z_pos[i]) / temp;
        let mut logits = vec![pos_sim];
        let mut idx = vec![usize::MAX]; // MAX marks the positive
        for j in 0..n {
            if j != i {
                logits.push(cosine(&z[i], &z[j]) / temp);
                idx.push(j);
            }
        }
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sumexp: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|&e| e / sumexp).collect();
        // L_i = -log p_pos. dL/d logit_m = p_m - 1{m == positive}. Mean over batch.
        for (m, &j) in idx.iter().enumerate() {
            let d_logit = (probs[m] - if m == 0 { 1.0 } else { 0.0 }) * scale / temp;
            if j == usize::MAX {
                // positive: sim(z_i, z_pos_i)
                let ga = cosine_grad_a(&z[i], &z_pos[i]);
                let gb = cosine_grad_a(&z_pos[i], &z[i]);
                for t in 0..k {
                    dz[i][t] += d_logit * ga[t];
                    dz_pos[i][t] += d_logit * gb[t];
                }
            } else {
                // negative: sim(z_i, z_j)
                let ga = cosine_grad_a(&z[i], &z[j]);
                let gb = cosine_grad_a(&z[j], &z[i]);
                for t in 0..k {
                    dz[i][t] += d_logit * ga[t];
                    dz[j][t] += d_logit * gb[t];
                }
            }
        }
    }
    (dz, dz_pos)
}

#[allow(clippy::type_complexity)]
pub(crate) fn batch_forward(
    w: &Weights,
    bn_mu: &BatchNorm,
    bn_lv: &BatchNorm,
    bn_dec: &BatchNorm,
    prior_mu: &[f64],
    prior_var: &[f64],
    alpha: &[f64],
    opts: &AvitmOptions,
    batch: &Batch,
    content: Option<&ContentFwd>,
) -> (f64, BatchCache, [(Vec<f64>, Vec<f64>); 3]) {
    let (k, v) = (w.k, w.v);
    let n = batch.xns.len();

    // Encoder up to the pre-BN heads (issue #378 follow-up). Layer 1 is sparse and
    // mode-dependent, so it stays a per-document map — pure (each doc reads shared
    // weights, writes its own row), so rayon-parallelized and order-preserving.
    // CombinedTM (`BowEmbAdapt`) feeds the *raw* BoW counts (`batch.counts`), matching
    // the reference's `CountVectorizer` input; every other mode keeps the
    // length-normalized BoW (`batch.xns`). Layers 2 and the heads are DENSE, so they
    // batch into deterministic GEMMs (fixed reduction order → bit-identical regardless
    // of thread count, like the decoder in #586) instead of per-document scalar loops.
    let h = w.hidden;
    let l1: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let enc_bow = if w.mode == InputMode::BowEmbAdapt {
                batch.counts[i]
            } else {
                batch.xns[i]
            };
            w.encode_layer1(enc_bow, batch.embs[i])
        })
        .collect();
    // Flatten h1 into a contiguous N×H buffer for the GEMMs.
    let mut h1_flat = vec![0.0; n * h];
    for (i, (_, h1, _)) in l1.iter().enumerate() {
        h1_flat[i * h..(i + 1) * h].copy_from_slice(h1);
    }
    // Layer 2: PRE2 (N×H) = H1 (N×H) · W2ᵀ (w2 is H×H row-major), then + b2.
    let mut pre2_flat = vec![0.0; n * h];
    crate::gemm::matmul_nt(n, h, h, &h1_flat, &w.w2, &mut pre2_flat);
    // Dropout on softplus(pre2): HD (N×H). Also fold the b2 bias in here.
    let mut hd_flat = vec![0.0; n * h];
    for i in 0..n {
        let m2 = &batch.masks2[i];
        for a in 0..h {
            let p = pre2_flat[i * h + a] + w.b2[a];
            pre2_flat[i * h + a] = p;
            hd_flat[i * h + a] = softplus(p) * m2[a];
        }
    }
    // Heads: MU_RAW (N×K) = HD · W_muᵀ + b_mu; LV_RAW = HD · W_lsᵀ + b_ls.
    let mut mu_raw_flat = vec![0.0; n * k];
    let mut lv_raw_flat = vec![0.0; n * k];
    crate::gemm::matmul_nt(n, h, k, &hd_flat, &w.w_mu, &mut mu_raw_flat);
    crate::gemm::matmul_nt(n, h, k, &hd_flat, &w.w_ls, &mut lv_raw_flat);
    for i in 0..n {
        for t in 0..k {
            mu_raw_flat[i * k + t] += w.b_mu[t];
            lv_raw_flat[i * k + t] += w.b_ls[t];
        }
    }
    // Assemble per-document DocCache (unchanged downstream/backward contract).
    let doc: Vec<DocCache> = (0..n)
        .map(|i| {
            let (pre1, h1, adapted) = &l1[i];
            DocCache {
                pre1: pre1.clone(),
                h1: h1.clone(),
                pre2: pre2_flat[i * h..(i + 1) * h].to_vec(),
                hd: hd_flat[i * h..(i + 1) * h].to_vec(),
                mu_raw: mu_raw_flat[i * k..(i + 1) * k].to_vec(),
                lv_raw: lv_raw_flat[i * k..(i + 1) * k].to_vec(),
                adapted: adapted.clone(),
            }
        })
        .collect();
    let mu_raw: Vec<Vec<f64>> = doc.iter().map(|d| d.mu_raw.clone()).collect();
    let lv_raw: Vec<Vec<f64>> = doc.iter().map(|d| d.lv_raw.clone()).collect();

    // Batchnorm the heads.
    let (mu, c_mu, mean_mu, var_mu) = bn_mu.forward_train(&mu_raw);
    let (lv, c_lv, mean_lv, var_lv) = bn_lv.forward_train(&lv_raw);

    let dirichlet = opts.prior == Prior::Dirichlet;
    let stick = opts.prior == Prior::StickBreaking;
    let contrastive = opts.contrastive;

    // Reparameterize and decode.
    let mut theta = vec![vec![0.0; k]; n];
    let mut theta_do = vec![vec![0.0; k]; n];
    let mut logit_raw = vec![vec![0.0; v]; n];
    // Dirichlet scratch.
    let mut d_kw = vec![vec![0.0; k]; n];
    let mut d_lam = vec![vec![0.0; k]; n];
    let mut d_ln_l = vec![vec![0.0; k]; n];
    let mut d_g = vec![vec![0.0; k]; n];
    let mut d_s = vec![0.0; n];
    // Stick-breaking scratch (eta breaks).
    let mut sb_eta = vec![vec![0.0; k]; n];
    // Contrastive positive-view scratch.
    let mut theta_pos = vec![vec![0.0; k]; n];
    let mut g_pos = vec![vec![0.0; k]; n];
    let mut s_pos = vec![0.0; n];

    for i in 0..n {
        let th: Vec<f64> = if dirichlet {
            // Weibull reparameterization: u = Phi(eps), L = -ln(1-u), constant in params.
            let mut s = 0.0;
            for t in 0..k {
                let u = normal_cdf(batch.eps[i][t]).clamp(1e-6, 1.0 - 1e-6);
                let ln_l = (-(1.0 - u).ln()).ln();
                let (kw, lam, g) = weibull_weight(mu[i][t], lv[i][t], ln_l);
                d_kw[i][t] = kw;
                d_lam[i][t] = lam;
                d_ln_l[i][t] = ln_l;
                d_g[i][t] = g;
                s += g;
            }
            d_s[i] = s;
            (0..k).map(|t| d_g[i][t] / s).collect()
        } else {
            // Gaussian latent (shared by laplace and stick-breaking).
            let mut z = vec![0.0; k];
            for t in 0..k {
                z[t] = mu[i][t] + (0.5 * lv[i][t]).exp() * batch.eps[i][t];
            }
            if stick {
                let (eta, th) = stick_break(&z);
                sb_eta[i] = eta;
                th
            } else {
                softmax(&z)
            }
        };
        for t in 0..k {
            theta_do[i][t] = th[t] * batch.masks_t[i][t];
        }
        theta[i] = th;

        // Positive view for the contrastive term: same reparameterization without the
        // noise. Laplace -> softmax(mu); Dirichlet -> normalized Weibull at the median
        // (u = 0.5, so L = ln 2). Both are deterministic and smooth in the params.
        if contrastive {
            if dirichlet {
                let ln_l = (-(0.5f64).ln()).ln(); // u = 0.5
                let mut s = 0.0;
                for t in 0..k {
                    let (_, _, g) = weibull_weight(mu[i][t], lv[i][t], ln_l);
                    g_pos[i][t] = g;
                    s += g;
                }
                s_pos[i] = s;
                for t in 0..k {
                    theta_pos[i][t] = g_pos[i][t] / s;
                }
            } else if stick {
                // No-noise stick-breaking on z = mu.
                let (_eta_pos, th_pos) = stick_break(&mu[i]);
                theta_pos[i] = th_pos;
            } else {
                theta_pos[i] = softmax(&mu[i]);
            }
        }
    }

    // Decoder forward: logit_raw (N×V) = theta_do (N×K) · beta (K×V). This dense
    // product dominates the fit, so route it through a deterministic BLAS-free GEMM
    // (issue #378) instead of the per-document scalar triple-loop. `theta_do` is a
    // Vec<Vec> (rows not contiguous), so flatten it and scatter the result back; the
    // O(N·K)+O(N·V) copies are negligible next to the O(N·K·V) product. The GEMM
    // reduction order is fixed, so the result is bit-identical regardless of thread
    // count (topica's determinism guarantee); it differs from the old scalar loop
    // only at float-rounding level (~1e-15). A zero `theta_do[i][t]` (dropout)
    // contributes zero exactly as the old `if w_t != 0.0` skip did.
    {
        let mut theta_flat = vec![0.0; n * k];
        for (i, td) in theta_do.iter().enumerate() {
            theta_flat[i * k..(i + 1) * k].copy_from_slice(td);
        }
        let mut logit_flat = vec![0.0; n * v];
        crate::gemm::matmul_nn(n, k, v, &theta_flat, &w.beta, &mut logit_flat);
        for (i, row) in logit_raw.iter_mut().enumerate() {
            row.copy_from_slice(&logit_flat[i * v..(i + 1) * v]);
        }
    }

    // SCHOLAR content deviation: per-covariate word shifts (+ optional
    // topic-covariate interactions on theta_do), added onto the decoder logits after
    // the GEMM. Fires only when a content model is attached (ProdLDA/InfoCTM pass
    // None), so the common path is a plain GEMM.
    if let Some(ct) = content {
        for (i, row) in logit_raw.iter_mut().enumerate() {
            let tc_i = &ct.tc[i];
            for (cc, &tcv) in tc_i.iter().enumerate().take(ct.c) {
                if tcv != 0.0 {
                    let base = cc * v;
                    for j in 0..v {
                        row[j] += ct.beta_c[base + j] * tcv;
                    }
                }
            }
            if let Some(bci) = ct.beta_ci {
                for t in 0..k {
                    let w_t = theta_do[i][t];
                    if w_t == 0.0 {
                        continue;
                    }
                    for (cc, &tcv) in tc_i.iter().enumerate().take(ct.c) {
                        if tcv == 0.0 {
                            continue;
                        }
                        let coef = w_t * tcv;
                        let base = (t * ct.c + cc) * v;
                        for j in 0..v {
                            row[j] += bci[base + j] * coef;
                        }
                    }
                }
            }
        }
    }
    let (logit, c_dec, mean_dec, var_dec) = bn_dec.forward_train(&logit_raw);

    // Reconstruction (softmax over the vocabulary) and KL.
    let mut recon = vec![vec![0.0; v]; n];
    let mut loss = 0.0;
    for i in 0..n {
        let r = softmax(&logit[i]);
        for &(word, c) in batch.counts[i] {
            loss -= c * (r[word] + 1e-10).ln();
        }
        recon[i] = r;
        if dirichlet {
            // KL( Weibull(kw, lam) || Gamma(alpha, 1) ), summed over topics
            // (Zhang et al. 2018). A Dirichlet(alpha) prior on theta factorizes into
            // independent Gamma(alpha_t, 1) priors on the unnormalized weights.
            for t in 0..k {
                loss += weibull_gamma_kl(d_kw[i][t], d_lam[i][t], alpha[t]);
            }
        } else {
            // KL( N(mu, e^lv) || N(mu0, var1) ), diagonal (eq. 7, first line). The
            // prior mean is per-document when `prior_mus` is set (SCHOLAR), else the
            // shared `prior_mu`; the prior variance is always shared.
            //
            // All K Gaussian coordinates are regularized. For the stick-breaking
            // prior only the first K-1 breaks affect theta (the last stick is the
            // remainder), so the K-th coordinate is a free, regularized-but-unused
            // latent: a valid ELBO for the K-dimensional augmented model, but a
            // deliberate deviation from Miao et al.'s K-1 Gaussian projection.
            // Summing K-1 for stick-breaking would change the objective and needs
            // re-validation (#428, finding 4); left as-is.
            let pm_i = batch.prior_mus.map(|p| p[i].as_slice()).unwrap_or(prior_mu);
            let mut kl = 0.0;
            for t in 0..k {
                let s0 = lv[i][t].exp();
                let dm = pm_i[t] - mu[i][t];
                kl +=
                    s0 / prior_var[t] + dm * dm / prior_var[t] - 1.0 + prior_var[t].ln() - lv[i][t];
            }
            loss += 0.5 * kl;
        }
    }

    // Contrastive InfoNCE term on the sampled topic vectors (anchor) vs the
    // positive view, with the other documents in the batch as negatives.
    if contrastive && n >= 2 {
        loss += opts.contrastive_weight * info_nce_loss(&theta, &theta_pos, opts.contrastive_temp);
    }

    let cache = BatchCache {
        doc,
        bn_mu: c_mu,
        bn_lv: c_lv,
        bn_dec: c_dec,
        mu,
        lv,
        theta,
        theta_do,
        recon,
        dir: if dirichlet {
            Some(DirCache {
                kw: d_kw,
                lam: d_lam,
                ln_l: d_ln_l,
                g: d_g,
                s: d_s,
            })
        } else {
            None
        },
        sb: if stick {
            Some(SbCache { eta: sb_eta })
        } else {
            None
        },
        contrast: if contrastive {
            Some(ContrastCache {
                theta_pos,
                g_pos,
                s_pos,
            })
        } else {
            None
        },
    };
    let stats = [(mean_mu, var_mu), (mean_lv, var_lv), (mean_dec, var_dec)];
    (loss, cache, stats)
}

/// Backward of the Dirichlet (Weibull) reparameterization for one document: given
/// `dtheta` (grad w.r.t. the normalized topic vector), the unnormalized weights `g`,
/// their sum `s`, the Weibull shape/scale `kw`/`lam`, and the noise constant `ln_l`,
/// accumulate into `dmu`/`dlv` (the post-BN head gradients), where
/// `kw = softplus(mu) + floor`, `lam = softplus(lv) + floor`, `g = lam L^(1/kw)`.
#[allow(clippy::too_many_arguments)]
fn weibull_reparam_backward(
    dtheta: &[f64],
    g: &[f64],
    s: f64,
    kw: &[f64],
    lam: &[f64],
    ln_l: &[f64],
    mu_post: &[f64],
    lv_post: &[f64],
    dmu: &mut [f64],
    dlv: &mut [f64],
) {
    let k = dtheta.len();
    // theta = g / s  ->  dg_m = (dtheta_m - sum_t dtheta_t theta_t) / s.
    let dot: f64 = (0..k).map(|t| dtheta[t] * (g[t] / s)).sum();
    for m in 0..k {
        let dg = (dtheta[m] - dot) / s;
        // g = lam * exp(ln_l / kw).
        let dg_dlam = if lam[m] != 0.0 { g[m] / lam[m] } else { 0.0 };
        let dg_dkw = g[m] * (-ln_l[m] / (kw[m] * kw[m]));
        // kw = softplus(mu)+f, lam = softplus(lv)+f.
        dmu[m] += dg * dg_dkw * sigmoid(mu_post[m]);
        dlv[m] += dg * dg_dlam * sigmoid(lv_post[m]);
    }
}

/// Backward pass over the batch, accumulating into `g`. Returns gradients for the
/// summed loss (the caller scales by 1/N).
/// `d_prior_mu`, when `Some`, is filled with the gradient of the batch loss w.r.t.
/// the per-document prior mean, `d_prior_mu[i][t] = (mu0[i][t] - mu_post[i][t]) /
/// prior_var[t]` (the Gaussian-KL derivative). SCHOLAR uses it to map back to the
/// covariate-weight gradient `dW = sum_i d_prior_mu[i] (x) PC[i]`; ProdLDA/InfoCTM
/// pass `None` and the block is skipped. Only meaningful on the Gaussian
/// (laplace/stick-breaking) path — the Dirichlet KL has no prior mean.
/// `dtheta_extra`, when `Some` (N x K), is added into the per-document gradient
/// w.r.t. `theta` before the reparameterization backward. SCHOLAR's label
/// classifier uses it to inject its `dL/dtheta` (the classifier reads `c.theta()`);
/// ProdLDA/InfoCTM pass `None`.
pub(crate) fn batch_backward(
    w: &Weights,
    prior_mu: &[f64],
    prior_var: &[f64],
    alpha: &[f64],
    opts: &AvitmOptions,
    batch: &Batch,
    c: &BatchCache,
    g: &mut Grad,
    mut d_prior_mu: Option<&mut [Vec<f64>]>,
    dtheta_extra: Option<&[Vec<f64>]>,
    content: Option<(&ContentFwd, &mut ContentGrad)>,
) {
    let (h, k, v) = (w.hidden, w.k, w.v);
    let n = batch.xns.len();
    let dirichlet = opts.prior == Prior::Dirichlet;
    let stick = opts.prior == Prior::StickBreaking;

    // --- Decoder: loss -> logit -> BN -> logit_raw -> (theta_do, beta). ---
    // d loss / d logit_iv = total_i * recon_iv - count_iv.
    let mut dlogit = vec![vec![0.0; v]; n];
    for i in 0..n {
        let total = batch.totals[i];
        for j in 0..v {
            dlogit[i][j] = total * c.recon[i][j];
        }
        for &(word, cnt) in batch.counts[i] {
            dlogit[i][word] -= cnt;
        }
    }
    let dlogit_raw = BatchNorm::backward(&dlogit, &c.bn_dec);

    let (content_fwd, content_grad): (Option<&ContentFwd>, Option<&mut ContentGrad>) = match content
    {
        Some((f, g)) => (Some(f), Some(g)),
        None => (None, None),
    };

    // Decoder backward, the two dense O(N·K·V) terms (issue #378):
    //   dtheta_do (N×K) = dlogit_raw (N×V) · betaᵀ        (per-document)
    //   g.beta   (K×V) += theta_doᵀ (K×N) · dlogit_raw   (cross-document reduction)
    // Both go through the deterministic BLAS-free GEMM: a fixed reduction order, so
    // bit-identical regardless of thread count (topica's determinism guarantee), and
    // it resolves the g.beta reduction that a rayon-over-batch approach could not
    // parallelize without reordering the sum. g.beta is accumulated via a scratch
    // buffer to preserve the "add into g" contract (a coupled caller may seed g.beta).
    let mut dtheta_do = vec![vec![0.0; k]; n];
    {
        // Flatten the Vec<Vec> operands (rows are not contiguous); O(N·V)+O(N·K),
        // negligible next to the O(N·K·V) products.
        let mut dlogit_flat = vec![0.0; n * v];
        for (i, dr) in dlogit_raw.iter().enumerate() {
            dlogit_flat[i * v..(i + 1) * v].copy_from_slice(dr);
        }
        let mut theta_flat = vec![0.0; n * k];
        for (i, td) in c.theta_do.iter().enumerate() {
            theta_flat[i * k..(i + 1) * k].copy_from_slice(td);
        }
        // dtheta_do = dlogit_raw · betaᵀ (beta stored K×V).
        let mut dtheta_flat = vec![0.0; n * k];
        crate::gemm::matmul_nt(n, v, k, &dlogit_flat, &w.beta, &mut dtheta_flat);
        for (i, dt) in dtheta_do.iter_mut().enumerate() {
            dt.copy_from_slice(&dtheta_flat[i * k..(i + 1) * k]);
        }
        // g.beta += theta_doᵀ · dlogit_raw.
        let mut gbeta = vec![0.0; k * v];
        crate::gemm::matmul_tn(k, n, v, &theta_flat, &dlogit_flat, &mut gbeta);
        for (gb, add) in g.beta.iter_mut().zip(gbeta.iter()) {
            *gb += *add;
        }
    }

    // SCHOLAR content-deviation gradients (scalar; fire only with a content model,
    // so ProdLDA/InfoCTM skip this entirely). The topic-covariate interaction
    // beta_ci adds into both dtheta_do and its own gradient; the main deviation
    // beta_c accumulates independent of the topic. Accumulation order matches the
    // old fused loop (i, t, cc, j), so gci/beta_c stay bit-identical.
    if let (Some(cf), Some(cg)) = (content_fwd, content_grad) {
        for i in 0..n {
            let tc_i = &cf.tc[i];
            if let (Some(bci), Some(gci)) = (cf.beta_ci, cg.beta_ci.as_deref_mut()) {
                for t in 0..k {
                    let mut acc = 0.0;
                    for (cc, &tcv) in tc_i.iter().enumerate().take(cf.c) {
                        if tcv == 0.0 {
                            continue;
                        }
                        let cbase = (t * cf.c + cc) * v;
                        for j in 0..v {
                            let dl = dlogit_raw[i][j];
                            acc += dl * bci[cbase + j] * tcv;
                            gci[cbase + j] += c.theta_do[i][t] * tcv * dl;
                        }
                    }
                    dtheta_do[i][t] += acc;
                }
            }
            for (cc, &tcv) in tc_i.iter().enumerate().take(cf.c) {
                if tcv == 0.0 {
                    continue;
                }
                let cbase = cc * v;
                for j in 0..v {
                    cg.beta_c[cbase + j] += dlogit_raw[i][j] * tcv;
                }
            }
        }
    }

    // Contrastive InfoNCE: gradients w.r.t. the sampled topic vectors (the anchor)
    // and the positive-view vectors. The anchor grad joins the decoder path into
    // `dtheta`; the positive-view grad routes only through the positive view.
    let (dz_contrast, dz_pos) = if opts.contrastive && n >= 2 {
        let cc = c.contrast.as_ref().unwrap();
        let (mut dz, dzp) = info_nce_backward(&c.theta, &cc.theta_pos, opts.contrastive_temp);
        for row in &mut dz {
            for x in row.iter_mut() {
                *x *= opts.contrastive_weight;
            }
        }
        let dzp: Vec<Vec<f64>> = dzp
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|x| x * opts.contrastive_weight)
                    .collect()
            })
            .collect();
        (Some(dz), Some(dzp))
    } else {
        (None, None)
    };

    // --- Per-document gradients into the BN-head outputs (mu, lv). ---
    let mut dmu = vec![vec![0.0; k]; n];
    let mut dlv = vec![vec![0.0; k]; n];
    for i in 0..n {
        // Decoder path: dropout on theta. Add the contrastive anchor gradient (which
        // acts on theta directly, before dropout) to get the full grad w.r.t. theta.
        let mut dtheta = vec![0.0; k];
        for t in 0..k {
            dtheta[t] = dtheta_do[i][t] * batch.masks_t[i][t];
            if let Some(dz) = &dz_contrast {
                dtheta[t] += dz[i][t];
            }
            // A coupled head's gradient w.r.t. theta (SCHOLAR's label classifier,
            // which reads the un-dropout `theta` in `c.theta()`). None for ProdLDA.
            if let Some(dte) = dtheta_extra {
                dtheta[t] += dte[i][t];
            }
        }

        if dirichlet {
            let dir = c.dir.as_ref().unwrap();
            weibull_reparam_backward(
                &dtheta,
                &dir.g[i],
                dir.s[i],
                &dir.kw[i],
                &dir.lam[i],
                &dir.ln_l[i],
                &c.mu[i],
                &c.lv[i],
                &mut dmu[i],
                &mut dlv[i],
            );
            // KL( Weibull || Gamma ) gradients, through softplus on each head.
            for t in 0..k {
                let (dkw, dlam) = weibull_gamma_kl_grad(dir.kw[i][t], dir.lam[i][t], alpha[t]);
                dmu[i][t] += dkw * sigmoid(c.mu[i][t]);
                dlv[i][t] += dlam * sigmoid(c.lv[i][t]);
            }
        } else {
            // Gaussian latent z = mu + exp(lv/2) * eps. The simplex map differs:
            // laplace -> softmax(z); stick-breaking -> sigmoid + stick-breaking. Both
            // share the same Gaussian reparameterization and Gaussian KL below.
            let dz: Vec<f64> = if stick {
                let sb = c.sb.as_ref().unwrap();
                stick_break_dz(&dtheta, &sb.eta[i], &c.theta[i])
            } else {
                let dot: f64 = (0..k).map(|t| dtheta[t] * c.theta[i][t]).sum();
                (0..k).map(|t| c.theta[i][t] * (dtheta[t] - dot)).collect()
            };
            // z = mu + exp(lv/2) * eps.
            let pm_i = batch.prior_mus.map(|p| p[i].as_slice()).unwrap_or(prior_mu);
            for t in 0..k {
                let s = (0.5 * c.lv[i][t]).exp();
                dmu[i][t] += dz[t];
                dlv[i][t] += dz[t] * batch.eps[i][t] * 0.5 * s;
                // KL gradients (post-BN mu, lv).
                let kl_dmu = (c.mu[i][t] - pm_i[t]) / prior_var[t];
                dmu[i][t] += kl_dmu;
                dlv[i][t] += 0.5 * (c.lv[i][t].exp() / prior_var[t] - 1.0);
                // Gradient w.r.t. the prior mean (opposite sign): d loss / d mu0.
                if let Some(dpm) = d_prior_mu.as_deref_mut() {
                    dpm[i][t] = -kl_dmu;
                }
            }
        }

        // Contrastive positive view routes its gradient through the no-noise path.
        if let Some(dzp) = &dz_pos {
            let cc = c.contrast.as_ref().unwrap();
            if dirichlet {
                // theta_pos via Weibull at u = 0.5 (L = ln 2), depends on mu and lv.
                let ln2 = std::f64::consts::LN_2;
                let ln_l_pos = vec![ln2.ln(); k]; // ln L, L = -ln(1-0.5) = ln 2
                let kw_pos: Vec<f64> = (0..k)
                    .map(|t| softplus(c.mu[i][t]) + WEIBULL_SHAPE_FLOOR)
                    .collect();
                let lam_pos: Vec<f64> = (0..k)
                    .map(|t| softplus(c.lv[i][t]) + WEIBULL_FLOOR)
                    .collect();
                weibull_reparam_backward(
                    &dzp[i],
                    &cc.g_pos[i],
                    cc.s_pos[i],
                    &kw_pos,
                    &lam_pos,
                    &ln_l_pos,
                    &c.mu[i],
                    &c.lv[i],
                    &mut dmu[i],
                    &mut dlv[i],
                );
            } else if stick {
                // theta_pos via no-noise stick-breaking on z = mu; grad into mu only.
                let (eta_pos, _) = stick_break(&c.mu[i]);
                let dz_pos = stick_break_dz(&dzp[i], &eta_pos, &cc.theta_pos[i]);
                for t in 0..k {
                    dmu[i][t] += dz_pos[t];
                }
            } else {
                // theta_pos = softmax(mu): softmax backward into mu only.
                let tp = &cc.theta_pos[i];
                let dot: f64 = (0..k).map(|t| dzp[i][t] * tp[t]).sum();
                for t in 0..k {
                    dmu[i][t] += tp[t] * (dzp[i][t] - dot);
                }
            }
        }
    }

    // Backprop through the head batchnorms.
    let dmu_raw = BatchNorm::backward(&dmu, &c.bn_mu);
    let dlv_raw = BatchNorm::backward(&dlv, &c.bn_lv);

    // --- Encoder backward (issue #378). ---
    // The dense layer-2 + head projections are batched into deterministic GEMMs
    // (fixed reduction order → bit-identical regardless of thread count, like the
    // decoder in #586); the mode-dependent sparse layer 1 stays a per-document
    // accumulation in fixed doc order; the activation derivatives are elementwise.
    //
    // Flatten the batched intermediates the GEMMs consume (rows are not contiguous in
    // the per-doc cache); O(N·H)+O(N·K), negligible next to the O(N·H²) products.
    let mut hd_flat = vec![0.0; n * h];
    let mut h1_flat = vec![0.0; n * h];
    let mut dmu_flat = vec![0.0; n * k];
    let mut dlv_flat = vec![0.0; n * k];
    for i in 0..n {
        hd_flat[i * h..(i + 1) * h].copy_from_slice(&c.doc[i].hd);
        h1_flat[i * h..(i + 1) * h].copy_from_slice(&c.doc[i].h1);
        dmu_flat[i * k..(i + 1) * k].copy_from_slice(&dmu_raw[i]);
        dlv_flat[i * k..(i + 1) * k].copy_from_slice(&dlv_raw[i]);
    }
    // Head weight gradients (K×H reductions): g.w_mu += dmu_rawᵀ·HD, g.w_ls += dlv_rawᵀ·HD.
    {
        let mut gwmu = vec![0.0; k * h];
        let mut gwls = vec![0.0; k * h];
        crate::gemm::matmul_tn(k, n, h, &dmu_flat, &hd_flat, &mut gwmu);
        crate::gemm::matmul_tn(k, n, h, &dlv_flat, &hd_flat, &mut gwls);
        for (g0, a) in g.w_mu.iter_mut().zip(&gwmu) {
            *g0 += *a;
        }
        for (g0, a) in g.w_ls.iter_mut().zip(&gwls) {
            *g0 += *a;
        }
    }
    // Head bias gradients (column sums over documents, fixed order).
    for i in 0..n {
        for t in 0..k {
            g.b_mu[t] += dmu_raw[i][t];
            g.b_ls[t] += dlv_raw[i][t];
        }
    }
    // dHD (N×H) = dmu_raw·W_mu + dlv_raw·W_ls (W_mu, W_ls are K×H row-major).
    let mut dhd_flat = vec![0.0; n * h];
    {
        let mut tmp = vec![0.0; n * h];
        crate::gemm::matmul_nn(n, k, h, &dmu_flat, &w.w_mu, &mut dhd_flat);
        crate::gemm::matmul_nn(n, k, h, &dlv_flat, &w.w_ls, &mut tmp);
        for (d, t) in dhd_flat.iter_mut().zip(&tmp) {
            *d += *t;
        }
    }
    // dpre2 (N×H) = dHD * mask2 * softplus'(pre2), with softplus'(x) = sigmoid(x).
    let mut dpre2_flat = vec![0.0; n * h];
    for i in 0..n {
        let m2 = &batch.masks2[i];
        let pre2 = &c.doc[i].pre2;
        for a in 0..h {
            dpre2_flat[i * h + a] = dhd_flat[i * h + a] * m2[a] * sigmoid(pre2[a]);
        }
    }
    // Layer-2 weight gradient (H×H reduction): g.w2 += dpre2ᵀ·H1. Bias: column sum.
    {
        let mut gw2 = vec![0.0; h * h];
        crate::gemm::matmul_tn(h, n, h, &dpre2_flat, &h1_flat, &mut gw2);
        for (g0, a) in g.w2.iter_mut().zip(&gw2) {
            *g0 += *a;
        }
    }
    for i in 0..n {
        for a in 0..h {
            g.b2[a] += dpre2_flat[i * h + a];
        }
    }
    // dH1 (N×H) = dpre2·W2 (W2 is H×H row-major).
    let mut dh1_flat = vec![0.0; n * h];
    crate::gemm::matmul_nn(n, h, h, &dpre2_flat, &w.w2, &mut dh1_flat);
    // dpre1 (N×H) = dH1 * softplus'(pre1).
    let mut dpre1_flat = vec![0.0; n * h];
    for i in 0..n {
        let pre1 = &c.doc[i].pre1;
        for a in 0..h {
            dpre1_flat[i * h + a] = dh1_flat[i * h + a] * sigmoid(pre1[a]);
        }
    }
    // Layer 1 (sparse BoW ⊕ dense second block, mode-dependent): accumulate into
    // g.w1/g.b1 (+ the CombinedTM adapt_bert weights) per document in fixed doc order
    // — a sparse cross-document reduction kept serial to preserve bit-identity.
    let cols = w.w1_cols();
    for i in 0..n {
        let dpre1 = &dpre1_flat[i * h..(i + 1) * h];
        if w.mode == InputMode::BowEmbAdapt {
            // CombinedTM: raw BoW ⊕ adapt_bert(emb). Accumulate the gradient into the
            // dense adapted block, back-propagate it into the `adapt_bert` weights.
            let mut dadapted = vec![0.0; v];
            for a in 0..h {
                g.b1[a] += dpre1[a];
                let row = a * cols;
                for &(word, val) in batch.counts[i] {
                    g.w1[row + word] += dpre1[a] * val;
                }
                let base = row + v;
                for j in 0..v {
                    g.w1[base + j] += dpre1[a] * c.doc[i].adapted[j];
                    dadapted[j] += dpre1[a] * w.w1[base + j];
                }
            }
            // adapt_bert = Linear(E, V): adapted[j] = b_adapt[j] + sum_k w_adapt[j,k]*emb[k].
            for j in 0..v {
                g.b_adapt[j] += dadapted[j];
                let base = j * w.e;
                for (kk, &ev) in batch.embs[i].iter().enumerate() {
                    g.w_adapt[base + kk] += dadapted[j] * ev;
                }
            }
        } else {
            for a in 0..h {
                g.b1[a] += dpre1[a];
                let row = a * cols;
                if w.mode != InputMode::EmbOnly {
                    for &(word, val) in batch.xns[i] {
                        g.w1[row + word] += dpre1[a] * val;
                    }
                }
                if w.mode != InputMode::BowOnly {
                    let base = row + v;
                    for (j, &ev) in batch.embs[i].iter().enumerate() {
                        g.w1[base + j] += dpre1[a] * ev;
                    }
                }
            }
        }
    }
}

/// Elementwise Adam with configurable `beta1` (ProdLDA uses 0.99) and coupled L2
/// weight decay, matching torch's `Adam`. Exposed `pub(crate)` so a coupled model
/// (`scholar`) can drive its own extra parameter block (the covariate weights) with
/// the same optimizer the shared blocks use.
pub(crate) struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    lr: f64,
    b1: f64,
    wd: f64,
}

impl Adam {
    pub(crate) fn new(len: usize, lr: f64, b1: f64, wd: f64) -> Self {
        Adam {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
            lr,
            b1,
            wd,
        }
    }
    pub(crate) fn step(&mut self, p: &mut [f64], grad: &[f64]) {
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - self.b1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for (pi, (&g0, (mi, vi))) in p
            .iter_mut()
            .zip(grad.iter().zip(self.m.iter_mut().zip(self.v.iter_mut())))
        {
            let g = g0 + self.wd * *pi;
            *mi = self.b1 * *mi + (1.0 - self.b1) * g;
            *vi = B2 * *vi + (1.0 - B2) * g * g;
            *pi -= self.lr * (*mi / bc1) / ((*vi / bc2).sqrt() + EPS);
        }
    }
}

/// A bundle of Adam states, one per parameter block.
pub(crate) struct Optim {
    w1: Adam,
    b1: Adam,
    w2: Adam,
    b2: Adam,
    w_mu: Adam,
    b_mu: Adam,
    w_ls: Adam,
    b_ls: Adam,
    beta: Adam,
    w_adapt: Adam,
    b_adapt: Adam,
}

impl Optim {
    pub(crate) fn new(w: &Weights, lr: f64, beta1: f64, wd: f64) -> Self {
        Optim {
            w1: Adam::new(w.w1.len(), lr, beta1, wd),
            b1: Adam::new(w.b1.len(), lr, beta1, wd),
            w2: Adam::new(w.w2.len(), lr, beta1, wd),
            b2: Adam::new(w.b2.len(), lr, beta1, wd),
            w_mu: Adam::new(w.w_mu.len(), lr, beta1, wd),
            b_mu: Adam::new(w.b_mu.len(), lr, beta1, wd),
            w_ls: Adam::new(w.w_ls.len(), lr, beta1, wd),
            b_ls: Adam::new(w.b_ls.len(), lr, beta1, wd),
            beta: Adam::new(w.beta.len(), lr, beta1, wd),
            w_adapt: Adam::new(w.w_adapt.len(), lr, beta1, wd),
            b_adapt: Adam::new(w.b_adapt.len(), lr, beta1, wd),
        }
    }
    pub(crate) fn step(&mut self, w: &mut Weights, g: &Grad) {
        self.w1.step(&mut w.w1, &g.w1);
        self.b1.step(&mut w.b1, &g.b1);
        self.w2.step(&mut w.w2, &g.w2);
        self.b2.step(&mut w.b2, &g.b2);
        self.w_mu.step(&mut w.w_mu, &g.w_mu);
        self.b_mu.step(&mut w.b_mu, &g.b_mu);
        self.w_ls.step(&mut w.w_ls, &g.w_ls);
        self.b_ls.step(&mut w.b_ls, &g.b_ls);
        self.beta.step(&mut w.beta, &g.beta);
        self.w_adapt.step(&mut w.w_adapt, &g.w_adapt);
        self.b_adapt.step(&mut w.b_adapt, &g.b_adapt);
    }
}

/// A fitted ProdLDA model. `beta` (K x V) is the unnormalized topic-word matrix;
/// `topic_word()` exposes its per-topic softmax. The encoder and the mean-head
/// batchnorm are retained so new documents transform with one forward pass.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ProdldaModel {
    pub num_topics: usize,
    pub num_types: usize,
    pub doc_topic: Vec<Vec<f64>>,
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub epochs_run: usize,
    pub weights: Weights,
    pub bn_mu: BatchNorm,
    /// The log-variance-head batchnorm. Only the `Prior::Dirichlet` transform reads
    /// it, to form the normalized Weibull *median* θ (a deterministic point estimate;
    /// training reconstruction itself uses Weibull *samples*, and the contrastive
    /// positive view uses this same median). `None` for a model constructed without
    /// it (the Laplace-only wrappers) or loaded from a pre-#428 save, in which case
    /// the Dirichlet transform falls back to `softmax(mu)`.
    pub bn_lv: Option<BatchNorm>,
    /// The prior the model was fit under, so `transform` applies the matching
    /// noise-free simplex map (softmax for laplace, the Weibull median for
    /// dirichlet, stick-breaking for `Prior::StickBreaking`).
    pub prior: Prior,
}

impl ProdldaModel {
    /// Per-topic word distribution, `softmax_v(beta_k)` (the product-of-experts
    /// expert for each topic). Batchnorm is omitted here, as is conventional for
    /// topic display.
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        let (k, v) = (self.num_topics, self.num_types);
        (0..k)
            .map(|t| softmax(&self.weights.beta[t * v..(t + 1) * v]))
            .collect()
    }

    /// Topic proportions for new documents: one encoder forward pass each, no
    /// sampling, using the running batchnorm statistics. The simplex map matches the
    /// training prior — `softmax(BN_eval(mu))` for laplace, stick-breaking for
    /// `StickBreaking`, and the normalized Weibull median for `Dirichlet`.
    /// For bow-only ProdLDA the embedding argument is unused; pass an empty slice.
    pub fn transform(&self, docs: &[Vec<u32>]) -> Vec<Vec<f64>> {
        let empty: Vec<Vec<f64>> = vec![Vec::new(); docs.len()];
        self.transform_with_emb(docs, &empty)
    }

    /// Topic proportions for new documents using both the bag of words and the
    /// caller's per-document embeddings. `docs[i]` and `embs[i]` describe the same
    /// document; the mode stored in the encoder selects which inputs are read.
    pub fn transform_with_emb(&self, docs: &[Vec<u32>], embs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        docs.iter()
            .zip(embs.iter())
            .map(|(d, emb)| {
                // CombinedTM reads raw counts at the encoder (see `batch_forward`);
                // every other mode reads the length-normalized BoW.
                let xn = if self.weights.mode == InputMode::BowEmbAdapt {
                    raw_bow(d)
                } else {
                    normalized_bow(d)
                };
                let no_drop = vec![1.0; self.weights.hidden];
                let dc = self.weights.encode_raw(&xn, emb, &no_drop);
                let mu = self.bn_mu.forward_eval_row(&dc.mu_raw);
                // Noise-free point estimate under the model's prior, matching the
                // map the decoder was trained with: softmax(mu) for laplace,
                // stick-breaking for StickBreaking, and the normalized Weibull
                // median for Dirichlet (#428). The Dirichlet median needs the
                // log-variance head; if it was not persisted (an older save) fall
                // back to softmax(mu), the previous shipped behavior.
                match self.prior {
                    Prior::StickBreaking => stick_break(&mu).1,
                    Prior::Dirichlet => match &self.bn_lv {
                        Some(bn_lv) => {
                            let lv = bn_lv.forward_eval_row(&dc.lv_raw);
                            weibull_median_theta(&mu, &lv)
                        }
                        None => softmax(&mu),
                    },
                    _ => softmax(&mu),
                }
            })
            .collect()
    }
}

/// Sparse normalized bag of words `(word_id, count / length)`.
pub(crate) fn normalized_bow(doc: &[u32]) -> Vec<(usize, f64)> {
    let mut counts: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &w in doc {
        *counts.entry(w as usize).or_insert(0.0) += 1.0;
    }
    let total: f64 = counts.values().sum::<f64>().max(1.0);
    counts.into_iter().map(|(w, c)| (w, c / total)).collect()
}

/// Sparse raw bag of words `(word_id, count)`.
pub(crate) fn raw_bow(doc: &[u32]) -> Vec<(usize, f64)> {
    let mut counts: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &w in doc {
        *counts.entry(w as usize).or_insert(0.0) += 1.0;
    }
    counts.into_iter().collect()
}

/// Fit ProdLDA by amortized VAE inference (minibatch Adam on the ELBO). `hidden` is
/// the encoder width (reference 100); `alpha` is the symmetric Dirichlet prior
/// concentration (reference 1.0); `dropout` is the dropout *rate* on `h2` and
/// `theta`; `epochs`/`batch_size`/`lr` drive Adam (reference 200/200/0.002, with
/// `beta1 = 0.99`); `em_tol` stops on the relative change in the epoch ELBO.
///
/// This is the bag-of-words encoder; [`fit_avitm`] is the generalization that adds
/// a dense per-document embedding block (CombinedTM / ZeroShotTM). The two share
/// the entire forward/backward core, and this path is byte-identical to the
/// pre-embedding implementation.
#[allow(clippy::too_many_arguments)]
pub fn fit_prodlda<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    hidden: usize,
    alpha: f64,
    dropout: f64,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    em_tol: f64,
    rng: &mut R,
) -> ProdldaModel {
    let empty: Vec<Vec<f64>> = vec![Vec::new(); docs.len()];
    fit_avitm(
        docs,
        &empty,
        InputMode::BowOnly,
        num_topics,
        num_types,
        0,
        hidden,
        alpha,
        dropout,
        epochs,
        batch_size,
        lr,
        em_tol,
        AvitmOptions::default(),
        rng,
    )
}

/// Fit the AVITM autoencoding-variational topic model with a chosen encoder input
/// (see [`InputMode`]). `embs[i]` is the dense embedding for document `i` (length
/// `emb_dim`; pass empty rows and `emb_dim == 0` for the bow-only path). The
/// decoder, prior, KL, reparameterization, batchnorm, Adam, and BoW reconstruction
/// loss are identical across modes; only the layer-1 input differs. CombinedTM is
/// [`InputMode::BowEmb`], ZeroShotTM is [`InputMode::EmbOnly`].
#[allow(clippy::too_many_arguments)]
pub fn fit_avitm<R: Rng>(
    docs: &[Vec<u32>],
    embs: &[Vec<f64>],
    mode: InputMode,
    num_topics: usize,
    num_types: usize,
    emb_dim: usize,
    hidden: usize,
    alpha: f64,
    dropout: f64,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    em_tol: f64,
    opts: AvitmOptions,
    rng: &mut R,
) -> ProdldaModel {
    let (k, v) = (num_topics, num_types);
    let d = docs.len();
    let xn: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| normalized_bow(doc)).collect();
    let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|doc| raw_bow(doc)).collect();
    let totals: Vec<f64> = bows
        .iter()
        .map(|b| b.iter().map(|&(_, c)| c).sum())
        .collect();

    let alpha_vec = vec![alpha; k];
    let (prior_mu, prior_var) = laplace_prior(&alpha_vec);
    let keep = (1.0 - dropout).max(1e-6);

    let mut w = Weights::new(v, emb_dim, hidden, k, mode, rng);
    let mut bn_mu = BatchNorm::new(k);
    let mut bn_lv = BatchNorm::new(k);
    let mut bn_dec = BatchNorm::new(v);
    let mut opt = Optim::new(&w, lr, 0.99, 0.0);

    let mut bound_history: Vec<f64> = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = 0usize;
    let mut order: Vec<usize> = (0..d).collect();

    for epoch in 0..epochs {
        epochs_run = epoch + 1;
        // Deterministic Fisher-Yates shuffle from the seeded rng.
        for i in (1..d).rev() {
            let j = (rng.gen::<f64>() * (i + 1) as f64) as usize;
            order.swap(i, j.min(i));
        }

        let mut epoch_loss = 0.0;
        let mut batches = 0usize;
        for chunk in order.chunks(batch_size.max(2)) {
            let n = chunk.len();
            if n < 2 {
                continue; // batchnorm needs at least two documents
            }
            // Per-document reparameterization noise and dropout masks.
            let eps: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..k).map(|_| randn(rng)).collect())
                .collect();
            let masks2: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..hidden)
                        .map(|_| {
                            if rng.gen::<f64>() < keep {
                                1.0 / keep
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            let masks_t: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    (0..k)
                        .map(|_| {
                            if rng.gen::<f64>() < keep {
                                1.0 / keep
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            let batch = Batch {
                xns: chunk.iter().map(|&di| xn[di].as_slice()).collect(),
                embs: chunk.iter().map(|&di| embs[di].as_slice()).collect(),
                counts: chunk.iter().map(|&di| bows[di].as_slice()).collect(),
                totals: chunk.iter().map(|&di| totals[di]).collect(),
                eps: &eps,
                masks2: &masks2,
                masks_t: &masks_t,
                prior_mus: None,
            };

            let (loss, cache, stats) = batch_forward(
                &w, &bn_mu, &bn_lv, &bn_dec, &prior_mu, &prior_var, &alpha_vec, &opts, &batch, None,
            );
            bn_mu.update_running(&stats[0].0, &stats[0].1);
            bn_lv.update_running(&stats[1].0, &stats[1].1);
            bn_dec.update_running(&stats[2].0, &stats[2].1);

            let mut g = Grad::zeros(&w);
            batch_backward(
                &w, &prior_mu, &prior_var, &alpha_vec, &opts, &batch, &cache, &mut g, None, None,
                None,
            );
            g.scale(1.0 / n as f64);
            opt.step(&mut w, &g);

            epoch_loss += loss / n as f64;
            batches += 1;
        }

        let avg = epoch_loss / batches.max(1) as f64;
        bound_history.push(-avg); // report the ELBO (negative loss)
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (-avg - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }
    }

    let model = ProdldaModel {
        num_topics: k,
        num_types: v,
        doc_topic: Vec::new(),
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        epochs_run,
        weights: w,
        bn_mu,
        bn_lv: Some(bn_lv),
        prior: opts.prior,
    };
    let doc_topic = model.transform_with_emb(docs, embs);
    ProdldaModel { doc_topic, ..model }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for ProdldaModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic.clone()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.bound_history
            .iter()
            .enumerate()
            .map(|(i, &b)| (i + 1, b))
            .collect()
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Recompute the summed batch loss for a given set of weights, at fixed noise
    // and fixed (all-ones) dropout masks, using batch-statistic batchnorm. This is
    // the function the analytic gradients are checked against.
    fn batch_loss(
        w: &Weights,
        prior_mu: &[f64],
        prior_var: &[f64],
        alpha: &[f64],
        opts: &AvitmOptions,
        batch: &Batch,
    ) -> f64 {
        let bn_mu = BatchNorm::new(w.k);
        let bn_lv = BatchNorm::new(w.k);
        let bn_dec = BatchNorm::new(w.v);
        batch_forward(
            w, &bn_mu, &bn_lv, &bn_dec, prior_mu, prior_var, alpha, opts, batch, None,
        )
        .0
    }

    // FD gradient check for a given encoder input mode and option set. Every weight
    // block is perturbed by central differences against the analytic batch gradient,
    // including the new dense-embedding columns of `w1`. The maximum relative error
    // across all parameters is returned for reporting.
    fn fd_check_mode_opts(mode: InputMode, emb_dim: usize, opts: AvitmOptions) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (v, hidden, k) = (7usize, 5usize, 4usize);
        let w0 = Weights::new(v, emb_dim, hidden, k, mode, &mut rng);
        let alpha = vec![1.0; k];
        let (prior_mu, prior_var) = laplace_prior(&alpha);

        // A small batch (>=2 docs so batchnorm statistics are well-defined).
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 0, 2, 3, 6],
            vec![1, 4, 4, 5],
            vec![2, 2, 3, 5, 6, 0],
        ];
        let xns: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| normalized_bow(d)).collect();
        let bows: Vec<Vec<(usize, f64)>> = docs.iter().map(|d| raw_bow(d)).collect();
        let totals: Vec<f64> = bows
            .iter()
            .map(|b| b.iter().map(|&(_, c)| c).sum())
            .collect();
        let n = docs.len();
        // Distinct, nonzero embeddings so the embedding columns get a real signal.
        let embs: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..emb_dim)
                    .map(|j| 0.3 * (i as f64 + 1.0) - 0.17 * j as f64 + 0.05)
                    .collect()
            })
            .collect();
        let eps: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|t| 0.1 * (i as f64 + 1.0) - 0.05 * t as f64)
                    .collect()
            })
            .collect();
        let masks2 = vec![vec![1.0; hidden]; n]; // dropout disabled for the check
        let masks_t = vec![vec![1.0; k]; n];
        let batch = Batch {
            xns: xns.iter().map(|x| x.as_slice()).collect(),
            embs: embs.iter().map(|x| x.as_slice()).collect(),
            counts: bows.iter().map(|b| b.as_slice()).collect(),
            totals: totals.clone(),
            eps: &eps,
            masks2: &masks2,
            masks_t: &masks_t,
            prior_mus: None,
        };

        // Analytic gradients.
        let bn_mu = BatchNorm::new(k);
        let bn_lv = BatchNorm::new(k);
        let bn_dec = BatchNorm::new(v);
        let (_, cache, _) = batch_forward(
            &w0, &bn_mu, &bn_lv, &bn_dec, &prior_mu, &prior_var, &alpha, &opts, &batch, None,
        );
        let mut g = Grad::zeros(&w0);
        batch_backward(
            &w0, &prior_mu, &prior_var, &alpha, &opts, &batch, &cache, &mut g, None, None, None,
        );

        let fd = 1e-6;
        let mut max_rel = 0.0f64;
        macro_rules! check_block {
            ($field:ident, $label:expr) => {
                for idx in 0..w0.$field.len() {
                    let mut wp = w0.clone();
                    wp.$field[idx] += fd;
                    let lp = batch_loss(&wp, &prior_mu, &prior_var, &alpha, &opts, &batch);
                    wp.$field[idx] -= 2.0 * fd;
                    let lm = batch_loss(&wp, &prior_mu, &prior_var, &alpha, &opts, &batch);
                    let num = (lp - lm) / (2.0 * fd);
                    let analytic = g.$field[idx];
                    let abs_err = (analytic - num).abs();
                    // Relative error is only meaningful where the gradient has
                    // appreciable magnitude; near zero, central-difference noise
                    // (O(fd^2) plus float cancellation) dominates the ratio, so we
                    // fall back to the absolute tolerance there.
                    let denom = analytic.abs().max(num.abs());
                    if denom > 1e-3 {
                        let rel = abs_err / denom;
                        if rel > max_rel {
                            max_rel = rel;
                        }
                    }
                    assert!(
                        abs_err < 1e-4,
                        "{:?} {} [{}]: analytic {} vs numeric {}",
                        mode,
                        $label,
                        idx,
                        analytic,
                        num
                    );
                }
            };
        }
        check_block!(w1, "w1");
        check_block!(b1, "b1");
        check_block!(w2, "w2");
        check_block!(b2, "b2");
        check_block!(w_mu, "w_mu");
        check_block!(b_mu, "b_mu");
        check_block!(w_ls, "w_ls");
        check_block!(b_ls, "b_ls");
        check_block!(beta, "beta");
        // CombinedTM `adapt_bert` blocks (empty, so no-ops, for every other mode).
        check_block!(w_adapt, "w_adapt");
        check_block!(b_adapt, "b_adapt");
        max_rel
    }

    #[test]
    fn batch_gradients_match_fd() {
        // Bow-only path (plain ProdLDA), default options (laplace, no contrastive).
        fd_check_mode_opts(InputMode::BowOnly, 0, AvitmOptions::default());
    }

    #[test]
    fn batch_gradients_match_fd_bow_emb() {
        // CombinedTM: BoW concatenated with a dense embedding block.
        let max_rel = fd_check_mode_opts(InputMode::BowEmb, 6, AvitmOptions::default());
        assert!(max_rel < 1e-4, "bow+emb max relative error {max_rel}");
    }

    #[test]
    fn batch_gradients_match_fd_bow_emb_adapt() {
        // CombinedTM faithful to Bianchi et al.: raw BoW concatenated with the
        // `adapt_bert`-projected embedding. Exercises the new `w_adapt`/`b_adapt`
        // gradients end-to-end against central differences.
        let max_rel = fd_check_mode_opts(InputMode::BowEmbAdapt, 6, AvitmOptions::default());
        assert!(max_rel < 1e-4, "bow+emb-adapt max relative error {max_rel}");
    }

    #[test]
    fn batch_gradients_match_fd_emb_only() {
        // ZeroShotTM: dense embedding block only, no BoW in the encoder.
        let max_rel = fd_check_mode_opts(InputMode::EmbOnly, 6, AvitmOptions::default());
        assert!(max_rel < 1e-4, "emb-only max relative error {max_rel}");
    }

    // --- #174 contrastive term FD checks (BowOnly + an embedding mode) ----------
    #[test]
    fn contrastive_gradients_match_fd_bow_only() {
        let opts = AvitmOptions {
            contrastive: true,
            contrastive_weight: 0.7,
            contrastive_temp: 0.4,
            ..AvitmOptions::default()
        };
        let max_rel = fd_check_mode_opts(InputMode::BowOnly, 0, opts);
        assert!(
            max_rel < 1e-4,
            "contrastive bow-only max relative error {max_rel}"
        );
    }

    #[test]
    fn contrastive_gradients_match_fd_bow_emb() {
        let opts = AvitmOptions {
            contrastive: true,
            contrastive_weight: 0.7,
            contrastive_temp: 0.4,
            ..AvitmOptions::default()
        };
        let max_rel = fd_check_mode_opts(InputMode::BowEmb, 6, opts);
        assert!(
            max_rel < 1e-4,
            "contrastive bow+emb max relative error {max_rel}"
        );
    }

    // --- #176 Dirichlet (Weibull) prior FD checks (BowOnly + an embedding mode) --
    #[test]
    fn dirichlet_gradients_match_fd_bow_only() {
        let opts = AvitmOptions {
            prior: Prior::Dirichlet,
            ..AvitmOptions::default()
        };
        let max_rel = fd_check_mode_opts(InputMode::BowOnly, 0, opts);
        assert!(
            max_rel < 1e-4,
            "dirichlet bow-only max relative error {max_rel}"
        );
    }

    #[test]
    fn dirichlet_gradients_match_fd_emb_only() {
        let opts = AvitmOptions {
            prior: Prior::Dirichlet,
            ..AvitmOptions::default()
        };
        let max_rel = fd_check_mode_opts(InputMode::EmbOnly, 6, opts);
        assert!(
            max_rel < 1e-4,
            "dirichlet emb-only max relative error {max_rel}"
        );
    }

    #[test]
    fn stick_breaking_gradients_match_fd_bow_only() {
        let opts = AvitmOptions {
            prior: Prior::StickBreaking,
            ..AvitmOptions::default()
        };
        let max_rel = fd_check_mode_opts(InputMode::BowOnly, 0, opts);
        assert!(
            max_rel < 1e-4,
            "stick-breaking bow-only max relative error {max_rel}"
        );
    }

    #[test]
    fn stick_breaking_gradients_match_fd_emb_only() {
        let opts = AvitmOptions {
            prior: Prior::StickBreaking,
            ..AvitmOptions::default()
        };
        let max_rel = fd_check_mode_opts(InputMode::EmbOnly, 6, opts);
        assert!(
            max_rel < 1e-4,
            "stick-breaking emb-only max relative error {max_rel}"
        );
    }

    // --- composition: both flags on at once must still FD-check ------------------
    #[test]
    fn contrastive_and_dirichlet_compose_fd() {
        let opts = AvitmOptions {
            prior: Prior::Dirichlet,
            contrastive: true,
            contrastive_weight: 0.6,
            contrastive_temp: 0.5,
        };
        let max_rel = fd_check_mode_opts(InputMode::BowEmb, 6, opts);
        assert!(
            max_rel < 1e-4,
            "contrastive+dirichlet max relative error {max_rel}"
        );
    }

    #[test]
    fn contrastive_and_stick_breaking_compose_fd() {
        let opts = AvitmOptions {
            prior: Prior::StickBreaking,
            contrastive: true,
            contrastive_weight: 0.6,
            contrastive_temp: 0.5,
        };
        let max_rel = fd_check_mode_opts(InputMode::BowEmb, 6, opts);
        assert!(
            max_rel < 1e-4,
            "contrastive+stick-breaking max relative error {max_rel}"
        );
    }

    // The Weibull-to-Gamma KL gradient checked directly against finite differences.
    #[test]
    fn weibull_gamma_kl_grad_matches_fd() {
        let fd = 1e-7;
        for &(kw, lam, a) in &[(0.7, 1.3, 1.0), (2.0, 0.5, 0.8), (1.1, 2.2, 1.5)] {
            let (dkw, dlam) = weibull_gamma_kl_grad(kw, lam, a);
            let num_kw = (weibull_gamma_kl(kw + fd, lam, a) - weibull_gamma_kl(kw - fd, lam, a))
                / (2.0 * fd);
            let num_lam = (weibull_gamma_kl(kw, lam + fd, a) - weibull_gamma_kl(kw, lam - fd, a))
                / (2.0 * fd);
            assert!((dkw - num_kw).abs() < 1e-5, "dkw {dkw} vs {num_kw}");
            assert!((dlam - num_lam).abs() < 1e-5, "dlam {dlam} vs {num_lam}");
        }
    }

    #[test]
    fn laplace_prior_symmetric() {
        // Symmetric alpha: mean is zero, variance is (1 - 1/K)/alpha.
        let k = 5;
        let alpha = 0.02;
        let (mu, var) = laplace_prior(&vec![alpha; k]);
        for &m in &mu {
            assert!(m.abs() < 1e-12);
        }
        let want = (1.0 - 1.0 / k as f64) / alpha;
        for &vv in &var {
            assert!((vv - want).abs() < 1e-9, "{vv} vs {want}");
        }
    }

    #[test]
    fn fit_recovers_planted_blocks() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 8usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..180)
            .map(|d| {
                let b = d % k;
                (0..15)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();

        let m = fit_prodlda(&docs, k, v, 32, 1.0, 0.0, 250, 60, 0.01, 0.0, &mut rng);
        assert_eq!(m.num_topics, k);
        let tw = m.topic_word();
        for row in &tw {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Each topic's top words should concentrate in a single planted block.
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut ord: Vec<usize> = (0..v).collect();
            ord.sort_by(|&a, &b| tw[t][b].total_cmp(&tw[t][a]));
            let blocks: std::collections::HashSet<usize> =
                ord[..4].iter().map(|&w| w / block).collect();
            assert_eq!(blocks.len(), 1, "topic {t} top words mix blocks");
            covered.insert(*blocks.iter().next().unwrap());
        }
        assert_eq!(covered.len(), k, "topics did not cover all blocks");
    }

    #[test]
    fn prodlda_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 8usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..180)
            .map(|d| {
                let b = d % k;
                (0..15)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let m = fit_prodlda(&docs, k, v, 32, 1.0, 0.0, 250, 60, 0.01, 0.0, &mut rng);
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }

    // A synthetic corpus where the document embedding encodes the planted topic
    // structure: K blocks of words, each document drawn from one block, and the
    // document's E-vector one-hot along its block's axis (plus noise). Returns
    // (docs, embeddings, k, block, v).
    fn planted_emb_corpus(
        n_docs: usize,
        seed: u64,
    ) -> (Vec<Vec<u32>>, Vec<Vec<f64>>, usize, usize, usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (k, block) = (3usize, 8usize);
        let v = k * block;
        let mut docs = Vec::with_capacity(n_docs);
        let mut embs = Vec::with_capacity(n_docs);
        for d in 0..n_docs {
            let b = d % k;
            let doc: Vec<u32> = (0..15)
                .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                .collect();
            // Embedding: 3.0 along the block axis, small noise elsewhere.
            let emb: Vec<f64> = (0..k)
                .map(|j| if j == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.2)
                .collect();
            docs.push(doc);
            embs.push(emb);
        }
        (docs, embs, k, block, v)
    }

    fn top_blocks(tw: &[Vec<f64>], k: usize, v: usize, block: usize) -> usize {
        let mut covered = std::collections::HashSet::new();
        for row in tw.iter().take(k) {
            let mut ord: Vec<usize> = (0..v).collect();
            ord.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let blocks: std::collections::HashSet<usize> =
                ord[..4].iter().map(|&w| w / block).collect();
            assert_eq!(blocks.len(), 1, "topic top words mix blocks");
            covered.insert(*blocks.iter().next().unwrap());
        }
        covered.len()
    }

    #[test]
    fn combinedtm_recovers_planted_blocks() {
        let (docs, embs, k, block, v) = planted_emb_corpus(180, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_avitm(
            &docs,
            &embs,
            InputMode::BowEmb,
            k,
            v,
            k,
            32,
            1.0,
            0.0,
            250,
            60,
            0.01,
            0.0,
            AvitmOptions::default(),
            &mut rng,
        );
        let tw = m.topic_word();
        for row in &tw {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert_eq!(
            top_blocks(&tw, k, v, block),
            k,
            "topics did not cover all blocks"
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }

    #[test]
    fn combinedtm_adapt_recovers_planted_blocks() {
        // CombinedTM faithful to Bianchi et al.: raw BoW ⊕ adapt_bert(emb).
        let (docs, embs, k, block, v) = planted_emb_corpus(180, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_avitm(
            &docs,
            &embs,
            InputMode::BowEmbAdapt,
            k,
            v,
            k,
            32,
            1.0,
            0.0,
            250,
            60,
            0.01,
            0.0,
            AvitmOptions::default(),
            &mut rng,
        );
        let tw = m.topic_word();
        for row in &tw {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert_eq!(
            top_blocks(&tw, k, v, block),
            k,
            "topics did not cover all blocks"
        );
        // The projected embedding weights must be live (non-empty and trained).
        assert_eq!(m.weights.w_adapt.len(), v * k);
        assert_eq!(m.weights.b_adapt.len(), v);
        assert!(m.weights.w_adapt.iter().any(|&x| x != 0.0));
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }

    #[test]
    fn zeroshottm_recovers_planted_blocks() {
        let (docs, embs, k, block, v) = planted_emb_corpus(180, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let m = fit_avitm(
            &docs,
            &embs,
            InputMode::EmbOnly,
            k,
            v,
            k,
            32,
            1.0,
            0.0,
            250,
            60,
            0.01,
            0.0,
            AvitmOptions::default(),
            &mut rng,
        );
        let tw = m.topic_word();
        for row in &tw {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // The encoder never saw the BoW; topics are recovered from embeddings alone.
        assert_eq!(
            top_blocks(&tw, k, v, block),
            k,
            "topics did not cover all blocks"
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }

    #[test]
    fn embedding_modes_are_deterministic() {
        let (docs, embs, k, _block, v) = planted_emb_corpus(60, 2);
        for mode in [
            InputMode::BowEmb,
            InputMode::BowEmbAdapt,
            InputMode::EmbOnly,
        ] {
            let mut r1 = ChaCha8Rng::seed_from_u64(11);
            let mut r2 = ChaCha8Rng::seed_from_u64(11);
            let a = fit_avitm(
                &docs,
                &embs,
                mode,
                k,
                v,
                k,
                16,
                1.0,
                0.0,
                30,
                30,
                0.01,
                0.0,
                AvitmOptions::default(),
                &mut r1,
            );
            let b = fit_avitm(
                &docs,
                &embs,
                mode,
                k,
                v,
                k,
                16,
                1.0,
                0.0,
                30,
                30,
                0.01,
                0.0,
                AvitmOptions::default(),
                &mut r2,
            );
            assert_eq!(
                a.topic_word(),
                b.topic_word(),
                "{mode:?} topic_word not bit-identical"
            );
            assert_eq!(
                a.doc_topic, b.doc_topic,
                "{mode:?} doc_topic not bit-identical"
            );
        }
    }

    // Planted recovery with each flag on: topics still concentrate on single blocks.
    #[test]
    fn fit_recovers_planted_blocks_contrastive() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 8usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..180)
            .map(|d| {
                let b = d % k;
                (0..15)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let opts = AvitmOptions {
            contrastive: true,
            ..AvitmOptions::default()
        };
        let m = fit_avitm(
            &docs,
            &vec![Vec::new(); docs.len()],
            InputMode::BowOnly,
            k,
            v,
            0,
            32,
            1.0,
            0.0,
            250,
            60,
            0.01,
            0.0,
            opts,
            &mut rng,
        );
        let tw = m.topic_word();
        assert_eq!(
            top_blocks(&tw, k, v, block),
            k,
            "contrastive: topics did not cover all blocks"
        );
    }

    #[test]
    fn fit_recovers_planted_blocks_dirichlet() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 8usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..180)
            .map(|d| {
                let b = d % k;
                (0..15)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let opts = AvitmOptions {
            prior: Prior::Dirichlet,
            ..AvitmOptions::default()
        };
        let m = fit_avitm(
            &docs,
            &vec![Vec::new(); docs.len()],
            InputMode::BowOnly,
            k,
            v,
            0,
            32,
            0.5,
            0.0,
            400,
            60,
            0.005,
            0.0,
            opts,
            &mut rng,
        );
        let tw = m.topic_word();
        for row in &tw {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Each topic should be dominated by a single planted block (majority of its
        // top-3 words from one block), and every block should be covered. The
        // Weibull-Dirichlet path mixes a little more than the laplace path, so we use
        // a majority rather than the strict "all top-4 in one block" helper.
        let mut covered = std::collections::HashSet::new();
        for row in tw.iter().take(k) {
            let mut ord: Vec<usize> = (0..v).collect();
            ord.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let mut counts = vec![0usize; k];
            for &w in &ord[..3] {
                counts[w / block] += 1;
            }
            let (dom, &cnt) = counts.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
            assert!(
                cnt >= 2,
                "dirichlet topic top words do not concentrate in a block"
            );
            covered.insert(dom);
        }
        assert_eq!(
            covered.len(),
            k,
            "dirichlet: topics did not cover all blocks"
        );
    }

    // #428 finding 2: a Dirichlet-prior model must transform via the normalized
    // Weibull median it trains under, not softmax(mu). Fit a small Dirichlet model,
    // then check transform == the median recomputed from the eval-time heads and
    // that the median genuinely differs from softmax(mu) (the old / laplace path).
    #[test]
    fn dirichlet_transform_uses_the_weibull_median_not_softmax_mu() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 8usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..120)
            .map(|d| {
                let b = d % k;
                (0..15)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let opts = AvitmOptions {
            prior: Prior::Dirichlet,
            ..AvitmOptions::default()
        };
        let m = fit_avitm(
            &docs,
            &vec![Vec::new(); docs.len()],
            InputMode::BowOnly,
            k,
            v,
            0,
            32,
            0.5,
            0.0,
            200,
            40,
            0.005,
            0.0,
            opts,
            &mut rng,
        );
        assert!(
            m.bn_lv.is_some(),
            "fit must retain bn_lv for the Dirichlet transform"
        );

        let test_docs = &docs[..5];
        let empty = vec![Vec::new(); test_docs.len()];
        let got = m.transform_with_emb(test_docs, &empty);

        let no_drop = vec![1.0; m.weights.hidden];
        let bn_lv = m.bn_lv.as_ref().unwrap();
        let mut saw_difference = false;
        for (d, row) in test_docs.iter().zip(&got) {
            let xn = normalized_bow(d);
            let dc = m.weights.encode_raw(&xn, &[], &no_drop);
            let mu = m.bn_mu.forward_eval_row(&dc.mu_raw);
            let lv = bn_lv.forward_eval_row(&dc.lv_raw);
            let median = weibull_median_theta(&mu, &lv);
            let softmax_mu = softmax(&mu);
            // transform reproduces the training median exactly...
            for t in 0..k {
                assert!(
                    (row[t] - median[t]).abs() < 1e-12,
                    "transform did not match the Weibull median"
                );
            }
            // ...and the median is a genuinely different point estimate than
            // softmax(mu), the pre-#428 (and laplace) behavior.
            let l1: f64 = (0..k).map(|t| (median[t] - softmax_mu[t]).abs()).sum();
            if l1 > 1e-6 {
                saw_difference = true;
            }
        }
        assert!(
            saw_difference,
            "median and softmax(mu) coincided on every test doc; the fix is unobservable here"
        );

        // A model without bn_lv (an older save) falls back to softmax(mu).
        let m_old = ProdldaModel { bn_lv: None, ..m };
        let fallback = m_old.transform_with_emb(test_docs, &empty);
        for (d, row) in test_docs.iter().zip(&fallback) {
            let xn = normalized_bow(d);
            let dc = m_old.weights.encode_raw(&xn, &[], &no_drop);
            let sm = softmax(&m_old.bn_mu.forward_eval_row(&dc.mu_raw));
            for t in 0..k {
                assert!(
                    (row[t] - sm[t]).abs() < 1e-12,
                    "bn_lv=None Dirichlet transform did not fall back to softmax(mu)"
                );
            }
        }
    }

    #[test]
    fn fit_recovers_planted_blocks_stick_breaking() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block) = (3usize, 8usize);
        let v = k * block;
        let docs: Vec<Vec<u32>> = (0..180)
            .map(|d| {
                let b = d % k;
                (0..15)
                    .map(|_| (b * block + (rng.gen::<f64>() * block as f64) as usize) as u32)
                    .collect()
            })
            .collect();
        let opts = AvitmOptions {
            prior: Prior::StickBreaking,
            ..AvitmOptions::default()
        };
        let m = fit_avitm(
            &docs,
            &vec![Vec::new(); docs.len()],
            InputMode::BowOnly,
            k,
            v,
            0,
            32,
            0.5,
            0.0,
            400,
            60,
            0.005,
            0.0,
            opts,
            &mut rng,
        );
        let tw = m.topic_word();
        for row in &tw {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Stick-breaking proportions live on the simplex by construction.
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(row.iter().all(|&x| x >= 0.0));
        }
        // Each topic dominated by a single planted block; every block covered. The
        // ordered sticks mix a little more, so use the same majority criterion.
        let mut covered = std::collections::HashSet::new();
        for row in tw.iter().take(k) {
            let mut ord: Vec<usize> = (0..v).collect();
            ord.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let mut counts = vec![0usize; k];
            for &w in &ord[..3] {
                counts[w / block] += 1;
            }
            let (dom, &cnt) = counts.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
            assert!(
                cnt >= 2,
                "stick-breaking topic top words do not concentrate in a block"
            );
            covered.insert(dom);
        }
        assert_eq!(
            covered.len(),
            k,
            "stick-breaking: topics did not cover all blocks"
        );
    }

    /// BatchNorm must normalize the batch with the biased variance but track the
    /// running variance with Bessel's correction (unbiased), matching PyTorch.
    #[test]
    fn batchnorm_running_var_is_unbiased() {
        // x = [[0], [2]]: mean = 1, biased var = 1, unbiased var = 2.
        let mut bn = BatchNorm::new(1);
        assert!((bn.running_var[0] - 1.0).abs() < 1e-12);
        assert!((bn.momentum - 0.1).abs() < 1e-12);

        let x = vec![vec![0.0], vec![2.0]];
        let (out, cache, mean, var) = bn.forward_train(&x);

        assert!((mean[0] - 1.0).abs() < 1e-12, "mean = {}", mean[0]);
        // The stat returned for the running update is the UNBIASED variance (2.0),
        // not the biased 1.0.
        assert!(
            (var[0] - 2.0).abs() < 1e-12,
            "returned running var = {}",
            var[0]
        );
        // In-batch normalization still uses the biased variance (1.0).
        let inv_std_biased = 1.0 / (1.0 + BN_EPS).sqrt();
        assert!((cache.inv_std[0] - inv_std_biased).abs() < 1e-12);
        assert!((out[0][0] + inv_std_biased).abs() < 1e-9); // (0-1)*inv_std
        assert!((out[1][0] - inv_std_biased).abs() < 1e-9); // (2-1)*inv_std

        bn.update_running(&mean, &var);
        // running_var = (1 - 0.1) * 1.0 + 0.1 * 2.0 = 1.1 (would be 1.0 with biased var).
        assert!(
            (bn.running_var[0] - 1.1).abs() < 1e-12,
            "running_var = {}, expected 1.1",
            bn.running_var[0]
        );
    }

    /// A singleton batch has no defined Bessel-corrected variance, so
    /// `forward_train` must reject it loudly rather than silently decay the running
    /// variance toward zero (#436). Every real caller already skips n < 2.
    #[test]
    #[should_panic(expected = "requires a batch of >= 2")]
    fn batchnorm_forward_train_rejects_singleton_batch() {
        let bn = BatchNorm::new(1);
        let _ = bn.forward_train(&[vec![2.0]]);
    }
}
