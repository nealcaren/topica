//! DETM: the Dynamic Embedded Topic Model (Dieng, Ruiz & Blei 2019,
//! arXiv:1907.05545). DETM extends ETM to time-stamped corpora: the topic
//! embeddings and the per-time topic prior each follow a Gaussian random walk, so
//! a topic's words drift smoothly over time.
//!
//! Generative model, for `K` topics, `T` time slices, vocabulary `V`, embedding
//! dimension `L`, word embeddings `rho` (V x L):
//!
//! ```text
//!   alpha_k^(0) ~ N(0, I);   alpha_k^(t) ~ N(alpha_k^(t-1), delta I)   (K x T x L)
//!   eta_0       ~ N(0, I);   eta_t       ~ N(eta_{t-1},     delta I)   (T x K)
//!   z_d ~ N(eta_{t_d}, I);   theta_d = softmax(z_d)                    (D x K)
//!   beta_k^(t)  = softmax_v( alpha_k^(t) . rho )                       (K x T x V)
//!   w_dn ~ Cat( theta_d . beta^(t_d) )
//! ```
//!
//! Inference is structured amortized variational inference; the ELBO is maximized
//! by minibatch Adam, with every gradient hand-coded (no autodiff crate), in the
//! style of [`crate::prodlda`] and [`crate::fastopic`]. The variational families:
//!
//! - **q(alpha)**: mean-field Gaussian per (k, t), free parameters `mu_q_alpha` and
//!   `logsigma_q_alpha` (K x T x L), exactly as the reference. `logsigma` is a
//!   log-variance; the random-walk KL uses the reference `get_kl` with prior mean
//!   `alpha_{t-1}` and log-variance `log(delta)` for `t >= 1` (unit Gaussian at
//!   `t = 0`).
//! - **q(theta_d)**: amortized encoder. Input `[normalized_bow_d, eta_{t_d}]` ->
//!   two `relu` layers (`t_hidden_size`) -> `mu`/`logsigma` heads -> reparameterized
//!   `z` -> `softmax`. KL against `N(eta_{t_d}, I)`. This matches the reference.
//! - **q(eta)**: the reference's LSTM amortization, reproduced exactly. A feature
//!   map `q_eta_map` = Linear(V -> eta_hidden) is applied to the per-time `rnn_input`
//!   (T x V, the mean count bag of words at each slice), giving a length-T sequence;
//!   a multi-layer LSTM (`eta_nlayers`) sweeps it to per-step outputs; then, per
//!   slice `t`, the heads `mu_q_eta`/`logsigma_q_eta` = Linear(eta_hidden + K -> K)
//!   read `[lstm_out_t, eta_{t-1}]` (with `eta_{-1} = 0`) and reparameterize to
//!   `eta_t`. The random-walk KL on eta is the reference's (prior mean = sampled
//!   `eta_{t-1}`, log-variance `log(delta)` for `t >= 1`; unit at `t = 0`). The LSTM
//!   forward and the backprop-through-time over all gates, layers and timesteps are
//!   hand-coded (no autodiff crate), like the rest of the model; a finite-difference
//!   test (`eta_lstm_bptt_matches_finite_difference`) gates the gradient.
//!
//! Determinism: all randomness (initialization, the per-epoch document shuffle, the
//! reparameterization noise) is drawn from a single seeded RNG in a fixed order, so
//! a fixed `seed` reproduces the fit bit-for-bit. Training is single-threaded; there
//! is no parallel reduction to order.
//!
//! Reference implementation (MIT, Dieng/Ruiz/Blei, github.com/adjidieng/DETM) was
//! read to match algorithmic detail (the `get_kl` log-variance form, the random-walk
//! prior variances, the `get_eta` LSTM amortization and its `init_hidden`/head
//! wiring, the minibatch `num_docs/batch_size` scaling on the NLL and the theta KL);
//! it was reimplemented idiomatically in Rust, not copied.

use rand::Rng;

/// Kaiming-uniform initialization matching PyTorch's `nn.Linear` default:
/// entries uniform on `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.
fn kaiming<R: Rng>(len: usize, fan_in: usize, rng: &mut R) -> Vec<f64> {
    let bound = 1.0 / (fan_in.max(1) as f64).sqrt();
    (0..len).map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * bound).collect()
}

/// A standard-normal sample via Box-Muller.
fn randn<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// PyTorch `nn.LSTM`/`nn.Linear` weight initialization detail: an LSTM inits every
/// weight and bias `uniform(-stdv, stdv)` with `stdv = 1/sqrt(hidden_size)`,
/// regardless of the per-tensor fan-in. This helper draws that block.
fn lstm_init<R: Rng>(len: usize, hidden: usize, rng: &mut R) -> Vec<f64> {
    let bound = 1.0 / (hidden.max(1) as f64).sqrt();
    (0..len).map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * bound).collect()
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

/// KL( N(q_mu, e^q_logvar) || N(p_mu, e^p_logvar) ), summed over the vector, using
/// the reference's log-variance parameterization (`detm.py::get_kl`):
///
/// ```text
///   0.5 * sum[ (sigma_q^2 + (mu_q - mu_p)^2) / (sigma_p^2 + 1e-6)
///              - 1 + log(sigma_p^2) - log(sigma_q^2) ]
/// ```
fn kl_gauss(q_mu: &[f64], q_logvar: &[f64], p_mu: &[f64], p_logvar: &[f64]) -> f64 {
    let mut kl = 0.0;
    for i in 0..q_mu.len() {
        let sq = q_logvar[i].exp();
        let sp = p_logvar[i].exp();
        let dm = q_mu[i] - p_mu[i];
        kl += (sq + dm * dm) / (sp + 1e-6) - 1.0 + p_logvar[i] - q_logvar[i];
    }
    0.5 * kl
}

/// The encoder for q(theta): `[normalized_bow, eta_td] -> h1 (relu) -> h2 (relu)
/// -> mu, logsigma`. Input width is `V + K`.
#[derive(Clone)]
pub struct ThetaEncoder {
    pub v: usize,
    pub k: usize,
    pub hidden: usize,
    pub w1: Vec<f64>, // hidden x (V + K)
    pub b1: Vec<f64>, // hidden
    pub w2: Vec<f64>, // hidden x hidden
    pub b2: Vec<f64>, // hidden
    pub w_mu: Vec<f64>, // K x hidden
    pub b_mu: Vec<f64>, // K
    pub w_ls: Vec<f64>, // K x hidden
    pub b_ls: Vec<f64>, // K
}

impl ThetaEncoder {
    fn new<R: Rng>(v: usize, k: usize, hidden: usize, rng: &mut R) -> Self {
        let inp = v + k;
        ThetaEncoder {
            v,
            k,
            hidden,
            w1: kaiming(hidden * inp, inp, rng),
            b1: vec![0.0; hidden],
            w2: kaiming(hidden * hidden, hidden, rng),
            b2: vec![0.0; hidden],
            w_mu: kaiming(k * hidden, hidden, rng),
            b_mu: vec![0.0; k],
            w_ls: kaiming(k * hidden, hidden, rng),
            b_ls: vec![0.0; k],
        }
    }

    /// Forward for one document. `xn` is the sparse normalized bag of words and
    /// `eta_td` the per-time prior mean (length K) concatenated after the vocabulary
    /// block. Returns the (mu, logsigma) heads and the cached activations needed for
    /// the backward pass.
    fn forward(&self, xn: &[(usize, f64)], eta_td: &[f64]) -> ThetaCache {
        let (h, k, v) = (self.hidden, self.k, self.v);
        // Layer 1: sparse in the vocabulary block, dense in the eta block.
        let mut pre1 = self.b1.clone();
        for i in 0..h {
            let row = i * (v + k);
            let mut s = pre1[i];
            for &(w, val) in xn {
                s += self.w1[row + w] * val;
            }
            for c in 0..k {
                s += self.w1[row + v + c] * eta_td[c];
            }
            pre1[i] = s;
        }
        let h1: Vec<f64> = pre1.iter().map(|&p| relu(p)).collect();
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
        let h2: Vec<f64> = pre2.iter().map(|&p| relu(p)).collect();
        // Heads.
        let mut mu = self.b_mu.clone();
        let mut ls = self.b_ls.clone();
        for c in 0..k {
            let row = c * h;
            let (mut sm, mut sl) = (mu[c], ls[c]);
            for i in 0..h {
                sm += self.w_mu[row + i] * h2[i];
                sl += self.w_ls[row + i] * h2[i];
            }
            mu[c] = sm;
            ls[c] = sl;
        }
        ThetaCache { pre1, h1, pre2, h2, mu, ls }
    }
}

/// Per-document encoder activations retained for the backward pass.
struct ThetaCache {
    pre1: Vec<f64>,
    h1: Vec<f64>,
    pre2: Vec<f64>,
    h2: Vec<f64>,
    mu: Vec<f64>,
    ls: Vec<f64>,
}

/// Gradient accumulators for the theta encoder.
struct EncGrad {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w_mu: Vec<f64>,
    b_mu: Vec<f64>,
    w_ls: Vec<f64>,
    b_ls: Vec<f64>,
}

impl EncGrad {
    fn zeros(e: &ThetaEncoder) -> Self {
        EncGrad {
            w1: vec![0.0; e.w1.len()],
            b1: vec![0.0; e.b1.len()],
            w2: vec![0.0; e.w2.len()],
            b2: vec![0.0; e.b2.len()],
            w_mu: vec![0.0; e.w_mu.len()],
            b_mu: vec![0.0; e.b_mu.len()],
            w_ls: vec![0.0; e.w_ls.len()],
            b_ls: vec![0.0; e.b_ls.len()],
        }
    }
}

/// The reference's amortized q(eta): a feature map `q_eta_map` = Linear(V ->
/// eta_hidden) applied to the per-time `rnn_input` (T x V), a multi-layer LSTM
/// `q_eta` over the length-T sequence, then per-timestep heads `mu_q_eta` /
/// `logsigma_q_eta` = Linear(eta_hidden + K -> K) over `[lstm_out_t, eta_{t-1}]`.
/// All gates and layers are hand-coded; backprop-through-time below.
///
/// PyTorch `nn.LSTM` gate convention (per layer, per timestep), gates stacked in
/// the row order `[input, forget, cell, output]`:
///
/// ```text
///   i = sigmoid(W_ii x + b_ii + W_hi h_{t-1} + b_hi)
///   f = sigmoid(W_if x + b_if + W_hf h_{t-1} + b_hf)
///   g =    tanh(W_ig x + b_ig + W_hg h_{t-1} + b_hg)
///   o = sigmoid(W_io x + b_io + W_ho h_{t-1} + b_ho)
///   c_t = f * c_{t-1} + i * g ;   h_t = o * tanh(c_t)
/// ```
#[derive(Clone)]
pub struct EtaNet {
    pub v: usize,
    pub k: usize,
    pub eh: usize,       // eta_hidden_size
    pub nlayers: usize,  // eta_nlayers
    // q_eta_map: Linear(V -> eh).
    pub map_w: Vec<f64>, // eh x V
    pub map_b: Vec<f64>, // eh
    // LSTM weights per layer; gates stacked [i,f,g,o] (rows 0..4*eh).
    // Layer 0 input width is `eh` (the map output); later layers also `eh`.
    pub w_ih: Vec<Vec<f64>>, // nlayers x (4*eh x eh)
    pub w_hh: Vec<Vec<f64>>, // nlayers x (4*eh x eh)
    pub b_ih: Vec<Vec<f64>>, // nlayers x (4*eh)
    pub b_hh: Vec<Vec<f64>>, // nlayers x (4*eh)
    // Heads: Linear(eh + K -> K).
    pub mu_w: Vec<f64>, // K x (eh + K)
    pub mu_b: Vec<f64>, // K
    pub ls_w: Vec<f64>, // K x (eh + K)
    pub ls_b: Vec<f64>, // K
}

impl EtaNet {
    fn new<R: Rng>(v: usize, k: usize, eh: usize, nlayers: usize, rng: &mut R) -> Self {
        // q_eta_map is a plain nn.Linear(V -> eh): kaiming-uniform with fan_in = V.
        let map_w = kaiming(eh * v, v, rng);
        let map_b = kaiming(eh, v, rng);
        // LSTM layers (uniform(-1/sqrt(eh), 1/sqrt(eh)) on every weight/bias).
        let mut w_ih = Vec::with_capacity(nlayers);
        let mut w_hh = Vec::with_capacity(nlayers);
        let mut b_ih = Vec::with_capacity(nlayers);
        let mut b_hh = Vec::with_capacity(nlayers);
        for _ in 0..nlayers {
            w_ih.push(lstm_init(4 * eh * eh, eh, rng));
            w_hh.push(lstm_init(4 * eh * eh, eh, rng));
            b_ih.push(lstm_init(4 * eh, eh, rng));
            b_hh.push(lstm_init(4 * eh, eh, rng));
        }
        // Heads: nn.Linear(eh + K -> K), fan_in = eh + K.
        let inp = eh + k;
        EtaNet {
            v, k, eh, nlayers,
            map_w, map_b,
            w_ih, w_hh, b_ih, b_hh,
            mu_w: kaiming(k * inp, inp, rng),
            mu_b: kaiming(k, inp, rng),
            ls_w: kaiming(k * inp, inp, rng),
            ls_b: kaiming(k, inp, rng),
        }
    }

    /// Forward q(eta) over the whole length-`T` sequence. `rnn_input` is `T x V`
    /// (per-time mean bag of words). `eps[t]` is the length-`K` reparameterization
    /// noise for slice `t`; when `sample == false` (eval) the heads return the mean
    /// `mu` directly (reference `reparameterize` in eval mode). Returns the sampled
    /// (or mean) `eta` (T x K), the per-slice head `mu`/`logsigma` (for the KL), and
    /// a cache holding every activation BPTT needs.
    fn forward(
        &self,
        rnn_input: &[Vec<f64>],
        eps: &[Vec<f64>],
        sample: bool,
    ) -> EtaForward {
        let (k, eh, nl) = (self.k, self.eh, self.nlayers);
        let t = rnn_input.len();

        // --- q_eta_map: x_t = relu? no, plain linear (no activation in reference). ---
        // map_out[t] (length eh) feeds the LSTM layer 0 as its input at step t.
        let mut map_out = vec![vec![0.0f64; eh]; t];
        for tt in 0..t {
            let row_in = &rnn_input[tt];
            for i in 0..eh {
                let base = i * self.v;
                let mut s = self.map_b[i];
                for vv in 0..self.v {
                    s += self.map_w[base + vv] * row_in[vv];
                }
                map_out[tt][i] = s;
            }
        }

        // --- Multi-layer LSTM. For each layer we sweep t = 0..T. ---
        // layer_in[t] is the input to the current layer at step t; for layer 0 it is
        // map_out, for higher layers it is the previous layer's h sequence.
        // Cache gate activations / cell states for BPTT.
        let mut gates_i = vec![vec![vec![0.0f64; eh]; t]; nl];
        let mut gates_f = vec![vec![vec![0.0f64; eh]; t]; nl];
        let mut gates_g = vec![vec![vec![0.0f64; eh]; t]; nl];
        let mut gates_o = vec![vec![vec![0.0f64; eh]; t]; nl];
        let mut cell = vec![vec![vec![0.0f64; eh]; t]; nl];     // c_t per layer/step
        let mut tanh_c = vec![vec![vec![0.0f64; eh]; t]; nl];   // tanh(c_t)
        let mut hidden = vec![vec![vec![0.0f64; eh]; t]; nl];   // h_t per layer/step
        // Inputs each layer saw (needed for the input-weight gradient).
        let mut layer_inputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(nl);

        for layer in 0..nl {
            let inp_seq: Vec<Vec<f64>> = if layer == 0 {
                map_out.clone()
            } else {
                hidden[layer - 1].clone()
            };
            let w_ih = &self.w_ih[layer];
            let w_hh = &self.w_hh[layer];
            let b_ih = &self.b_ih[layer];
            let b_hh = &self.b_hh[layer];
            for tt in 0..t {
                let x = &inp_seq[tt];
                // Pre-activations for the 4 gate blocks: a = W_ih x + b_ih + W_hh h_{t-1} + b_hh.
                // Row block g occupies rows [g*eh, g*eh+eh).
                for j in 0..eh {
                    // gate index g in {0:i,1:f,2:g,3:o}
                    let mut pre = [0.0f64; 4];
                    for (g, p) in pre.iter_mut().enumerate() {
                        let row = (g * eh + j) * eh;
                        let mut s = b_ih[g * eh + j] + b_hh[g * eh + j];
                        for i in 0..eh {
                            s += w_ih[row + i] * x[i];
                        }
                        if tt > 0 {
                            let hprev = &hidden[layer][tt - 1];
                            for i in 0..eh {
                                s += w_hh[row + i] * hprev[i];
                            }
                        }
                        *p = s;
                    }
                    let it = sigmoid(pre[0]);
                    let ft = sigmoid(pre[1]);
                    let gt = pre[2].tanh();
                    let ot = sigmoid(pre[3]);
                    let cprev = if tt > 0 { cell[layer][tt - 1][j] } else { 0.0 };
                    let ct = ft * cprev + it * gt;
                    let tc = ct.tanh();
                    let ht = ot * tc;
                    gates_i[layer][tt][j] = it;
                    gates_f[layer][tt][j] = ft;
                    gates_g[layer][tt][j] = gt;
                    gates_o[layer][tt][j] = ot;
                    cell[layer][tt][j] = ct;
                    tanh_c[layer][tt][j] = tc;
                    hidden[layer][tt][j] = ht;
                }
            }
            layer_inputs.push(inp_seq);
        }

        // LSTM output sequence = top-layer hidden states.
        let output = hidden[nl - 1].clone();

        // --- Heads + sequential reparameterization. ---
        let mut etas = vec![vec![0.0f64; k]; t];
        let mut mu = vec![vec![0.0f64; k]; t];
        let mut ls = vec![vec![0.0f64; k]; t];
        // head input = [output[t], eta_{t-1}] (eta_{-1} = 0 at t=0); length eh + k.
        let mut head_inp = vec![vec![0.0f64; eh + k]; t];
        for tt in 0..t {
            let inw = eh + k;
            // assemble head input
            for i in 0..eh {
                head_inp[tt][i] = output[tt][i];
            }
            if tt == 0 {
                for c in 0..k {
                    head_inp[tt][eh + c] = 0.0;
                }
            } else {
                for c in 0..k {
                    head_inp[tt][eh + c] = etas[tt - 1][c];
                }
            }
            for c in 0..k {
                let row = c * inw;
                let mut sm = self.mu_b[c];
                let mut slg = self.ls_b[c];
                for i in 0..inw {
                    sm += self.mu_w[row + i] * head_inp[tt][i];
                    slg += self.ls_w[row + i] * head_inp[tt][i];
                }
                mu[tt][c] = sm;
                ls[tt][c] = slg;
                etas[tt][c] = if sample {
                    sm + (0.5 * slg).exp() * eps[tt][c]
                } else {
                    sm
                };
            }
        }

        EtaForward {
            map_out, layer_inputs,
            gates_i, gates_f, gates_g, gates_o, cell, tanh_c, hidden,
            output, head_inp, mu, ls, etas,
        }
    }
}

/// Cached q(eta) forward activations retained for BPTT.
struct EtaForward {
    map_out: Vec<Vec<f64>>,                 // T x eh
    layer_inputs: Vec<Vec<Vec<f64>>>,       // nlayers x T x eh (input seen by each layer)
    gates_i: Vec<Vec<Vec<f64>>>,            // nlayers x T x eh
    gates_f: Vec<Vec<Vec<f64>>>,
    gates_g: Vec<Vec<Vec<f64>>>,
    gates_o: Vec<Vec<Vec<f64>>>,
    cell: Vec<Vec<Vec<f64>>>,
    tanh_c: Vec<Vec<Vec<f64>>>,
    hidden: Vec<Vec<Vec<f64>>>,             // nlayers x T x eh
    output: Vec<Vec<f64>>,                  // T x eh (top-layer hidden)
    head_inp: Vec<Vec<f64>>,                // T x (eh + K)
    mu: Vec<Vec<f64>>,                      // T x K
    ls: Vec<Vec<f64>>,                      // T x K
    etas: Vec<Vec<f64>>,                    // T x K
}

/// Gradient accumulators mirroring [`EtaNet`].
struct EtaGrad {
    map_w: Vec<f64>,
    map_b: Vec<f64>,
    w_ih: Vec<Vec<f64>>,
    w_hh: Vec<Vec<f64>>,
    b_ih: Vec<Vec<f64>>,
    b_hh: Vec<Vec<f64>>,
    mu_w: Vec<f64>,
    mu_b: Vec<f64>,
    ls_w: Vec<f64>,
    ls_b: Vec<f64>,
}

impl EtaGrad {
    fn zeros(n: &EtaNet) -> Self {
        EtaGrad {
            map_w: vec![0.0; n.map_w.len()],
            map_b: vec![0.0; n.map_b.len()],
            w_ih: n.w_ih.iter().map(|w| vec![0.0; w.len()]).collect(),
            w_hh: n.w_hh.iter().map(|w| vec![0.0; w.len()]).collect(),
            b_ih: n.b_ih.iter().map(|b| vec![0.0; b.len()]).collect(),
            b_hh: n.b_hh.iter().map(|b| vec![0.0; b.len()]).collect(),
            mu_w: vec![0.0; n.mu_w.len()],
            mu_b: vec![0.0; n.mu_b.len()],
            ls_w: vec![0.0; n.ls_w.len()],
            ls_b: vec![0.0; n.ls_b.len()],
        }
    }
}

impl EtaNet {
    /// Backprop-through-time for q(eta). `fwd` is the cached forward; `d_eta[t]`
    /// (length K) is `d loss / d eta_t` flowing in from the rest of the model (NLL
    /// + theta-KL, which use `eta_{t_d}` as the theta prior, plus the eta random-walk
    /// KL, whose prior mean is the sampled `eta_{t-1}`). `eps` is the same noise the
    /// forward used. Returns the parameter gradients; also threads gradients through
    /// the sequential `eta_{t-1}` coupling in the heads and the LSTM recurrence.
    ///
    /// `sample` mirrors the forward: when true, `eta_t = mu_t + exp(0.5 ls_t) eps_t`
    /// so `d eta_t` splits into both heads; when false, `eta_t = mu_t`.
    ///
    /// `d_mu_in`/`d_ls_in` are gradients that arrive *directly* on the per-slice head
    /// outputs `mu_t` / `logsigma_t` (the eta-KL and theta-KL terms differentiate the
    /// head mean/log-variance, not only the sampled `eta_t`). They are added on top of
    /// the reparameterization path from `d_eta_in`.
    #[allow(clippy::needless_range_loop)]
    fn backward(
        &self,
        fwd: &EtaForward,
        d_eta_in: &[Vec<f64>],
        d_mu_in: &[Vec<f64>],
        d_ls_in: &[Vec<f64>],
        eps: &[Vec<f64>],
        sample: bool,
        rnn_input: &[Vec<f64>],
    ) -> EtaGrad {
        let (k, eh, nl) = (self.k, self.eh, self.nlayers);
        let t = rnn_input.len();
        let inw = eh + k;
        let mut g = EtaGrad::zeros(self);

        // d loss / d output[t] (top-layer hidden) accumulates from the heads.
        let mut d_output = vec![vec![0.0f64; eh]; t];
        // Running gradient on eta_t, including the contribution that arrives via
        // eta_{t-1} entering the heads at step t+1. We sweep t = T-1 .. 0.
        let mut d_eta = d_eta_in.to_vec(); // start from the external gradient
        for tt in (0..t).rev() {
            // eta_t = sample ? mu_t + exp(0.5 ls_t) eps_t : mu_t. Gradient on the head
            // outputs is the direct KL contribution plus the reparameterization path.
            let mut d_mu = d_mu_in[tt].clone();
            let mut d_ls = d_ls_in[tt].clone();
            for c in 0..k {
                d_mu[c] += d_eta[tt][c];
                if sample {
                    let std = (0.5 * fwd.ls[tt][c]).exp();
                    d_ls[c] += d_eta[tt][c] * eps[tt][c] * 0.5 * std;
                }
            }
            // Heads: mu_t = mu_w · head_inp + mu_b ; same for ls. Accumulate weight
            // gradients and push d into head_inp = [output[t], eta_{t-1}].
            let hin = &fwd.head_inp[tt];
            let mut d_hin = vec![0.0f64; inw];
            for c in 0..k {
                let row = c * inw;
                g.mu_b[c] += d_mu[c];
                g.ls_b[c] += d_ls[c];
                for i in 0..inw {
                    g.mu_w[row + i] += d_mu[c] * hin[i];
                    g.ls_w[row + i] += d_ls[c] * hin[i];
                    d_hin[i] += d_mu[c] * self.mu_w[row + i] + d_ls[c] * self.ls_w[row + i];
                }
            }
            // head_inp[0..eh] = output[t]; head_inp[eh..] = eta_{t-1}.
            for i in 0..eh {
                d_output[tt][i] += d_hin[i];
            }
            if tt > 0 {
                for c in 0..k {
                    d_eta[tt - 1][c] += d_hin[eh + c];
                }
            }
        }

        // --- BPTT through the LSTM stack, top layer down to layer 0. ---
        // d_layer_out[t] is d loss / d (this layer's h_t). For the top layer it is
        // d_output; for lower layers it is the d_input accumulated from the layer above.
        let mut d_upper = d_output; // gradient on top-layer hidden outputs
        for layer in (0..nl).rev() {
            let w_ih = &self.w_ih[layer];
            let w_hh = &self.w_hh[layer];
            let inp_seq = &fwd.layer_inputs[layer];
            // Gradient flowing into this layer's inputs (x_t), to pass to layer-1.
            let mut d_layer_in = vec![vec![0.0f64; eh]; t];
            // Recurrent carries.
            let mut d_h_next = vec![0.0f64; eh]; // d from step t+1's recurrence into h_t
            let mut d_c_next = vec![0.0f64; eh]; // d into c_t from step t+1
            for tt in (0..t).rev() {
                // Total gradient on h_t = external (d_upper) + recurrent (d_h_next).
                let mut d_h = vec![0.0f64; eh];
                for j in 0..eh {
                    d_h[j] = d_upper[tt][j] + d_h_next[j];
                }
                // h_t = o_t * tanh(c_t).
                let o = &fwd.gates_o[layer][tt];
                let it = &fwd.gates_i[layer][tt];
                let ft = &fwd.gates_f[layer][tt];
                let gt = &fwd.gates_g[layer][tt];
                let tc = &fwd.tanh_c[layer][tt];
                // d c_t gets a path from h_t (through o*tanh(c)) plus the carry from t+1.
                let mut d_c = vec![0.0f64; eh];
                let mut d_pre = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for blk in d_pre.iter_mut() {
                    *blk = vec![0.0f64; eh];
                }
                for j in 0..eh {
                    let d_o = d_h[j] * tc[j];
                    let d_tc = d_h[j] * o[j];
                    let d_ct = d_tc * (1.0 - tc[j] * tc[j]) + d_c_next[j];
                    d_c[j] = d_ct;
                    let cprev = if tt > 0 { fwd.cell[layer][tt - 1][j] } else { 0.0 };
                    let d_f = d_ct * cprev;
                    let d_i = d_ct * gt[j];
                    let d_g = d_ct * it[j];
                    // Gate pre-activation gradients (sigmoid' = s(1-s); tanh' = 1-g^2).
                    d_pre[0][j] = d_i * it[j] * (1.0 - it[j]);  // input gate
                    d_pre[1][j] = d_f * ft[j] * (1.0 - ft[j]);  // forget gate
                    d_pre[2][j] = d_g * (1.0 - gt[j] * gt[j]);  // cell gate (tanh)
                    d_pre[3][j] = d_o * o[j] * (1.0 - o[j]);     // output gate
                }
                // Accumulate weight/bias grads and propagate to x_t and h_{t-1}.
                let x = &inp_seq[tt];
                let mut d_x = vec![0.0f64; eh];
                let mut d_hprev = vec![0.0f64; eh];
                for blk in 0..4 {
                    let dp = &d_pre[blk];
                    let gw_ih = &mut g.w_ih[layer];
                    let gw_hh = &mut g.w_hh[layer];
                    let gb_ih = &mut g.b_ih[layer];
                    let gb_hh = &mut g.b_hh[layer];
                    for j in 0..eh {
                        let dpj = dp[j];
                        if dpj == 0.0 {
                            // bias still gets 0; skip the inner loops cheaply.
                        }
                        let row = (blk * eh + j) * eh;
                        gb_ih[blk * eh + j] += dpj;
                        gb_hh[blk * eh + j] += dpj;
                        for i in 0..eh {
                            gw_ih[row + i] += dpj * x[i];
                            d_x[i] += dpj * w_ih[row + i];
                        }
                        if tt > 0 {
                            let hprev = &fwd.hidden[layer][tt - 1];
                            for i in 0..eh {
                                gw_hh[row + i] += dpj * hprev[i];
                                d_hprev[i] += dpj * w_hh[row + i];
                            }
                        }
                    }
                }
                // Carry into step t-1.
                d_h_next = d_hprev;
                for j in 0..eh {
                    d_c_next[j] = d_c[j] * ft[j];
                }
                // Save the input gradient for the layer below (or q_eta_map).
                for i in 0..eh {
                    d_layer_in[tt][i] = d_x[i];
                }
            }
            if layer == 0 {
                // d_layer_in is d loss / d map_out; backprop through q_eta_map.
                for tt in 0..t {
                    let row_in = &rnn_input[tt];
                    for i in 0..eh {
                        let d = d_layer_in[tt][i];
                        if d == 0.0 {
                            continue;
                        }
                        g.map_b[i] += d;
                        let base = i * self.v;
                        for vv in 0..self.v {
                            g.map_w[base + vv] += d * row_in[vv];
                        }
                    }
                }
            } else {
                d_upper = d_layer_in; // becomes the upstream gradient for the lower layer
            }
        }

        g
    }
}

/// Elementwise Adam matching torch's `Adam` (beta1=0.9, beta2=0.999) with coupled
/// L2 weight decay.
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    lr: f64,
    wd: f64,
}

impl Adam {
    fn new(len: usize, lr: f64, wd: f64) -> Self {
        Adam { m: vec![0.0; len], v: vec![0.0; len], t: 0, lr, wd }
    }
    fn step(&mut self, p: &mut [f64], grad: &[f64]) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for (pi, (&g0, (mi, vi))) in
            p.iter_mut().zip(grad.iter().zip(self.m.iter_mut().zip(self.v.iter_mut())))
        {
            let g = g0 + self.wd * *pi;
            *mi = B1 * *mi + (1.0 - B1) * g;
            *vi = B2 * *vi + (1.0 - B2) * g * g;
            *pi -= self.lr * (*mi / bc1) / ((*vi / bc2).sqrt() + EPS);
        }
    }
}

/// A fitted DETM. `beta_over_time` is `T x K x V` (each `[t][k]` a distribution over
/// the vocabulary); `doc_topic` is `D x K`; `alpha` is `T x K x L` (topic-embedding
/// trajectories, the variational means); `eta` is `T x K` (the time-varying topic
/// prior, the variational means).
pub struct DetmModel {
    pub num_topics: usize,
    pub num_times: usize,
    pub num_types: usize,
    pub beta_over_time: Vec<Vec<Vec<f64>>>, // T x K x V
    pub doc_topic: Vec<Vec<f64>>,           // D x K
    pub alpha: Vec<Vec<Vec<f64>>>,          // T x K x L
    pub eta: Vec<Vec<f64>>,                 // T x K
    pub bound: f64,
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub epochs_run: usize,
}

impl DetmModel {
    /// Time-collapsed topic-word matrix `K x V`: the mean of `beta` over time. This
    /// is the standard `topic_word` surface; the per-time matrices are in
    /// `beta_over_time`.
    pub fn topic_word_mean(&self) -> Vec<Vec<f64>> {
        let (k, v, t) = (self.num_topics, self.num_types, self.num_times);
        let mut out = vec![vec![0.0; v]; k];
        for bt in &self.beta_over_time {
            for kk in 0..k {
                for vv in 0..v {
                    out[kk][vv] += bt[kk][vv];
                }
            }
        }
        let inv = 1.0 / t.max(1) as f64;
        for row in &mut out {
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
        out
    }
}

/// Sparse bag of words as `(word_id, count)`, sorted by word id (deterministic).
fn raw_bow(tokens: &[u32], counts: &[u32]) -> Vec<(usize, f64)> {
    let mut m: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for (i, &w) in tokens.iter().enumerate() {
        let c = counts.get(i).copied().unwrap_or(1) as f64;
        *m.entry(w as usize).or_insert(0.0) += c;
    }
    m.into_iter().collect()
}

/// `softmax_v(alpha_k . rho)` for one topic embedding `alpha_k` (length L), with a
/// log-sum-exp for stability. Returns a length-V distribution.
fn beta_row(rho: &[Vec<f64>], alpha_k: &[f64]) -> Vec<f64> {
    let v = rho.len();
    let mut logit = vec![0.0; v];
    let mut max = f64::NEG_INFINITY;
    for (w, rv) in rho.iter().enumerate() {
        let dot: f64 = rv.iter().zip(alpha_k).map(|(r, a)| r * a).sum();
        logit[w] = dot;
        if dot > max {
            max = dot;
        }
    }
    let mut z = 0.0;
    for &e in &logit {
        z += (e - max).exp();
    }
    let logz = max + z.ln();
    logit.iter().map(|&e| (e - logz).exp()).collect()
}

/// Fit DETM by structured amortized variational inference (minibatch Adam on the
/// ELBO). See the module docs for the variational families, including the LSTM
/// amortization of q(eta).
///
/// - `tokens[d]`/`counts[d]` are the sparse bag of words of document `d`.
/// - `times[d]` is the time-slice index of document `d` (0-based, contiguous).
/// - `rho` is the fixed word-embedding matrix (V x L).
/// - `delta` is the random-walk variance knob (reference default 0.005); the prior
///   variance for a step is `delta` (matching the reference, which sets the prior
///   log-variance to `log(delta)`).
/// - `hidden` is the theta encoder width; `eta_hidden`/`eta_nlayers` size the LSTM
///   that amortizes q(eta); `epochs`/`batch_size`/`lr`/`wdecay` drive Adam; `em_tol`
///   stops on the relative change in the epoch ELBO.
#[allow(clippy::too_many_arguments)]
pub fn fit_detm<R: Rng>(
    tokens: &[Vec<u32>],
    counts: &[Vec<u32>],
    times: &[usize],
    num_topics: usize,
    num_types: usize,
    num_times: usize,
    rho: &[Vec<f64>],
    delta: f64,
    hidden: usize,
    eta_hidden: usize,
    eta_nlayers: usize,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    wdecay: f64,
    em_tol: f64,
    rng: &mut R,
) -> DetmModel {
    let (k, v, t, l) = (
        num_topics,
        num_types,
        num_times,
        if num_types > 0 { rho[0].len() } else { 0 },
    );
    let d = tokens.len();
    let log_delta = delta.max(1e-12).ln();

    // Precompute the per-document sparse raw and normalized bags of words.
    let bows: Vec<Vec<(usize, f64)>> =
        (0..d).map(|i| raw_bow(&tokens[i], &counts[i])).collect();
    let totals: Vec<f64> = bows.iter().map(|b| b.iter().map(|&(_, c)| c).sum::<f64>()).collect();
    let nbows: Vec<Vec<(usize, f64)>> = bows
        .iter()
        .zip(&totals)
        .map(|(b, &tot)| {
            let s = if tot > 0.0 { tot } else { 1.0 };
            b.iter().map(|&(w, c)| (w, c / s)).collect()
        })
        .collect();

    // --- rnn_input (T x V): per-time mean count bag of words, exactly the
    // reference's `data.get_rnn_input` (sum of raw counts over the slice's docs,
    // divided by the number of docs in the slice). The LSTM q(eta) reads this. ---
    let mut rnn_input = vec![vec![0.0f64; v]; t];
    let mut slice_counts = vec![0usize; t];
    for di in 0..d {
        let td = times[di];
        slice_counts[td] += 1;
        for &(w, c) in &bows[di] {
            rnn_input[td][w] += c;
        }
    }
    for tt in 0..t {
        let denom = slice_counts[tt].max(1) as f64;
        for x in rnn_input[tt].iter_mut() {
            *x /= denom;
        }
    }

    // --- Variational parameters ------------------------------------------------
    // q(alpha): mu/logsigma per (t, k, l), stored time-major to match the reference
    // alpha layout (T x K x L). Reference inits mu/logsigma from N(0,1).
    let mut mu_alpha = vec![vec![vec![0.0f64; l]; k]; t];
    let mut ls_alpha = vec![vec![vec![0.0f64; l]; k]; t];
    for tt in 0..t {
        for kk in 0..k {
            for ll in 0..l {
                mu_alpha[tt][kk][ll] = randn(rng);
                ls_alpha[tt][kk][ll] = randn(rng);
            }
        }
    }
    // q(eta): the reference's LSTM amortization (q_eta_map -> LSTM -> heads). Init
    // order (map, then per-layer LSTM, then heads) matches `EtaNet::new`.
    let mut eta_net = EtaNet::new(v, k, eta_hidden, eta_nlayers, rng);
    // q(theta): amortized encoder.
    let mut enc = ThetaEncoder::new(v, k, hidden, rng);

    // --- Optimizers ------------------------------------------------------------
    let mut a_mu_alpha = Adam::new(t * k * l, lr, wdecay);
    let mut a_ls_alpha = Adam::new(t * k * l, lr, wdecay);
    // q(eta) LSTM-network optimizers (one Adam per parameter block).
    let mut a_eta_map_w = Adam::new(eta_net.map_w.len(), lr, wdecay);
    let mut a_eta_map_b = Adam::new(eta_net.map_b.len(), lr, wdecay);
    let mut a_eta_w_ih: Vec<Adam> = eta_net.w_ih.iter().map(|w| Adam::new(w.len(), lr, wdecay)).collect();
    let mut a_eta_w_hh: Vec<Adam> = eta_net.w_hh.iter().map(|w| Adam::new(w.len(), lr, wdecay)).collect();
    let mut a_eta_b_ih: Vec<Adam> = eta_net.b_ih.iter().map(|b| Adam::new(b.len(), lr, wdecay)).collect();
    let mut a_eta_b_hh: Vec<Adam> = eta_net.b_hh.iter().map(|b| Adam::new(b.len(), lr, wdecay)).collect();
    let mut a_eta_mu_w = Adam::new(eta_net.mu_w.len(), lr, wdecay);
    let mut a_eta_mu_b = Adam::new(eta_net.mu_b.len(), lr, wdecay);
    let mut a_eta_ls_w = Adam::new(eta_net.ls_w.len(), lr, wdecay);
    let mut a_eta_ls_b = Adam::new(eta_net.ls_b.len(), lr, wdecay);
    let mut a_w1 = Adam::new(enc.w1.len(), lr, wdecay);
    let mut a_b1 = Adam::new(enc.b1.len(), lr, wdecay);
    let mut a_w2 = Adam::new(enc.w2.len(), lr, wdecay);
    let mut a_b2 = Adam::new(enc.b2.len(), lr, wdecay);
    let mut a_w_mu = Adam::new(enc.w_mu.len(), lr, wdecay);
    let mut a_b_mu = Adam::new(enc.b_mu.len(), lr, wdecay);
    let mut a_w_ls = Adam::new(enc.w_ls.len(), lr, wdecay);
    let mut a_b_ls = Adam::new(enc.b_ls.len(), lr, wdecay);

    let mut bound_history: Vec<f64> = Vec::with_capacity(epochs);
    let mut converged = false;
    let mut epochs_run = 0usize;
    let mut order: Vec<usize> = (0..d).collect();

    for epoch in 0..epochs {
        epochs_run = epoch + 1;
        // Deterministic Fisher-Yates shuffle from the seeded rng (matches the
        // reference's per-epoch torch.randperm over the document indices).
        for i in (1..d).rev() {
            let j = (rng.gen::<f64>() * (i + 1) as f64) as usize;
            order.swap(i, j.min(i));
        }

        let mut epoch_loss = 0.0;
        let mut batches = 0usize;

        // We process the corpus in minibatches; the global latents alpha and eta are
        // resampled per batch, as in the reference's per-forward sampling, and the
        // NLL / theta-KL are scaled by num_docs/batch_size so larger corpora behave
        // correctly while KL_alpha / KL_eta stay global.
        for chunk in order.chunks(batch_size.max(1)) {
            let n = chunk.len();
            let coeff = d as f64 / n as f64;

            // --- Sample alpha (T x K x L) along the random walk, retaining the eps
            // used so the backward pass through the reparameterization is exact. ---
            let mut eps_alpha = vec![vec![vec![0.0f64; l]; k]; t];
            let mut alpha = vec![vec![vec![0.0f64; l]; k]; t];
            for tt in 0..t {
                for kk in 0..k {
                    for ll in 0..l {
                        let e = randn(rng);
                        eps_alpha[tt][kk][ll] = e;
                        let std = (0.5 * ls_alpha[tt][kk][ll]).exp();
                        alpha[tt][kk][ll] = mu_alpha[tt][kk][ll] + std * e;
                    }
                }
            }
            // --- Sample eta (T x K) via the LSTM amortization. Draw the per-slice
            // reparameterization noise first (fixed order), then run the network
            // forward; the heads produce mu_t/ls_t and the sampled eta_t. ---
            let mut eps_eta = vec![vec![0.0f64; k]; t];
            for tt in 0..t {
                for kk in 0..k {
                    eps_eta[tt][kk] = randn(rng);
                }
            }
            let eta_fwd = eta_net.forward(&rnn_input, &eps_eta, true);
            let eta = eta_fwd.etas.clone();

            // --- beta[t][k] = softmax(alpha[t][k] . rho). ---
            let beta: Vec<Vec<Vec<f64>>> = (0..t)
                .map(|tt| (0..k).map(|kk| beta_row(rho, &alpha[tt][kk])).collect())
                .collect();

            // Gradient accumulators.
            let mut g_mu_alpha = vec![vec![vec![0.0f64; l]; k]; t];
            let mut g_ls_alpha = vec![vec![vec![0.0f64; l]; k]; t];
            // q(eta) backward inputs: gradient on the sampled eta_t (from the theta
            // encoder's eta-input block and the random-walk KL prior-mean coupling)
            // and direct gradient on the head outputs mu_t / ls_t (from the KL terms).
            let mut d_eta = vec![vec![0.0f64; k]; t];
            let mut d_mu_eta = vec![vec![0.0f64; k]; t];
            let mut d_ls_eta = vec![vec![0.0f64; k]; t];
            let mut g_enc = EncGrad::zeros(&enc);
            // d loss / d beta[t][k][w], accumulated over the batch's documents.
            let mut dbeta = vec![vec![vec![0.0f64; v]; k]; t];

            let mut batch_loss = 0.0;

            // --- Per-document: q(theta) encode, reparameterize, decode, NLL + KL. ---
            for &di in chunk {
                let td = times[di];
                let cache = enc.forward(&nbows[di], &eta[td]);
                // Reparameterize z = mu + exp(0.5 logsigma) * eps -> theta = softmax(z).
                let mut z = vec![0.0; k];
                let mut eps_z = vec![0.0; k];
                for c in 0..k {
                    let e = randn(rng);
                    eps_z[c] = e;
                    z[c] = cache.mu[c] + (0.5 * cache.ls[c]).exp() * e;
                }
                let theta = softmax(&z);

                // recon_v = sum_k theta_k beta[td][k][v]; NLL = -sum_v bow_v log(recon_v + 1e-6).
                // d NLL / d theta_k = -sum_v bow_v beta[td][k][v] / (recon_v + 1e-6)
                // d NLL / d beta[td][k][v] = -theta_k bow_v / (recon_v + 1e-6)
                let mut dtheta = vec![0.0; k];
                let mut nll = 0.0;
                for &(w, c) in &bows[di] {
                    let mut recon = 0.0;
                    for kk in 0..k {
                        recon += theta[kk] * beta[td][kk][w];
                    }
                    let denom = recon + 1e-6;
                    nll -= c * denom.ln();
                    let cf = c / denom;
                    for kk in 0..k {
                        dtheta[kk] -= cf * beta[td][kk][w];
                        dbeta[td][kk][w] -= cf * theta[kk] * coeff;
                    }
                }
                batch_loss += nll * coeff;

                // KL_theta = KL( N(mu, e^ls) || N(eta_td, 0) ); prior log-variance 0.
                let prior_lv = vec![0.0f64; k];
                let kl_theta = kl_gauss(&cache.mu, &cache.ls, &eta[td], &prior_lv);
                batch_loss += kl_theta * coeff;

                // --- Backward of NLL through theta = softmax(z). ---
                for x in dtheta.iter_mut() {
                    *x *= coeff;
                }
                let dot: f64 = (0..k).map(|c| dtheta[c] * theta[c]).sum();
                let dz: Vec<f64> = (0..k).map(|c| theta[c] * (dtheta[c] - dot)).collect();
                // z = mu + exp(0.5 ls) eps.
                let mut dmu = vec![0.0; k];
                let mut dls = vec![0.0; k];
                for c in 0..k {
                    let std = (0.5 * cache.ls[c]).exp();
                    dmu[c] += dz[c];
                    dls[c] += dz[c] * eps_z[c] * 0.5 * std;
                }
                // --- Backward of KL_theta (scaled by coeff) into mu, ls, and eta_td. ---
                let inv_pv = 1.0 / (1.0 + 1e-6);
                for c in 0..k {
                    let dm = cache.mu[c] - eta[td][c];
                    dmu[c] += coeff * inv_pv * dm;
                    dls[c] += coeff * 0.5 * (cache.ls[c].exp() * inv_pv - 1.0);
                    // d KL / d eta_td = -coeff * inv_pv * (mu - eta); eta_td is the
                    // sampled eta at slice td, so route it into d_eta.
                    d_eta[td][c] += -coeff * inv_pv * dm;
                }

                // --- Backprop dmu/dls through the encoder heads + MLP. ---
                let (h, vv) = (enc.hidden, enc.v);
                let mut dh2 = vec![0.0; h];
                for c in 0..k {
                    let row = c * h;
                    g_enc.b_mu[c] += dmu[c];
                    g_enc.b_ls[c] += dls[c];
                    for i in 0..h {
                        g_enc.w_mu[row + i] += dmu[c] * cache.h2[i];
                        g_enc.w_ls[row + i] += dls[c] * cache.h2[i];
                        dh2[i] += dmu[c] * enc.w_mu[row + i] + dls[c] * enc.w_ls[row + i];
                    }
                }
                // relu on layer 2.
                let mut dpre2 = vec![0.0; h];
                for i in 0..h {
                    dpre2[i] = if cache.pre2[i] > 0.0 { dh2[i] } else { 0.0 };
                }
                // layer 2: pre2 = W2 h1 + b2.
                let mut dh1 = vec![0.0; h];
                for a in 0..h {
                    let row = a * h;
                    g_enc.b2[a] += dpre2[a];
                    for b in 0..h {
                        g_enc.w2[row + b] += dpre2[a] * cache.h1[b];
                        dh1[b] += dpre2[a] * enc.w2[row + b];
                    }
                }
                // relu on layer 1.
                let mut dpre1 = vec![0.0; h];
                for i in 0..h {
                    dpre1[i] = if cache.pre1[i] > 0.0 { dh1[i] } else { 0.0 };
                }
                // layer 1: pre1 = W1 [nbow, eta_td] + b1.
                let inw = vv + k;
                for a in 0..h {
                    g_enc.b1[a] += dpre1[a];
                    let row = a * inw;
                    for &(w, val) in &nbows[di] {
                        g_enc.w1[row + w] += dpre1[a] * val;
                    }
                    // The eta block of the input also receives a gradient into eta_td.
                    for c in 0..k {
                        g_enc.w1[row + vv + c] += dpre1[a] * eta[td][c];
                        d_eta[td][c] += dpre1[a] * enc.w1[row + vv + c];
                    }
                }
            }

            // --- Backprop dbeta into alpha via beta = softmax(alpha . rho). ---
            // For topic (t,k): logit_w = alpha . rho_w; beta = softmax(logit).
            // dlogit_w = beta_w * (dbeta_w - sum_w' dbeta_w' beta_w'); then project
            // onto rho to get d/d alpha. The reparameterization splits into mu/ls.
            for tt in 0..t {
                for kk in 0..k {
                    let bw = &beta[tt][kk];
                    let db = &dbeta[tt][kk];
                    let dot: f64 = (0..v).map(|w| db[w] * bw[w]).sum();
                    let mut dalpha = vec![0.0f64; l];
                    for w in 0..v {
                        let dlogit = bw[w] * (db[w] - dot);
                        if dlogit != 0.0 {
                            let rw = &rho[w];
                            for ll in 0..l {
                                dalpha[ll] += dlogit * rw[ll];
                            }
                        }
                    }
                    for ll in 0..l {
                        let std = (0.5 * ls_alpha[tt][kk][ll]).exp();
                        g_mu_alpha[tt][kk][ll] += dalpha[ll];
                        g_ls_alpha[tt][kk][ll] += dalpha[ll] * eps_alpha[tt][kk][ll] * 0.5 * std;
                    }
                }
            }

            // --- KL_alpha (global, random walk over time). ---
            // t = 0: prior N(0, I). t >= 1: prior N(alpha_{t-1}, delta I) with the
            // prior *mean* being the sampled alpha_{t-1} (reference uses the sample).
            for kk in 0..k {
                let pm = vec![0.0f64; l];
                let plv = vec![0.0f64; l];
                batch_loss += kl_gauss(&mu_alpha[0][kk], &ls_alpha[0][kk], &pm, &plv);
                for ll in 0..l {
                    let dm = mu_alpha[0][kk][ll];
                    g_mu_alpha[0][kk][ll] += dm / (1.0 + 1e-6);
                    g_ls_alpha[0][kk][ll] += 0.5 * (ls_alpha[0][kk][ll].exp() / (1.0 + 1e-6) - 1.0);
                }
                for tt in 1..t {
                    let prior_mean = alpha[tt - 1][kk].clone(); // sampled alpha_{t-1}
                    let plv: Vec<f64> = vec![log_delta; l];
                    batch_loss += kl_gauss(&mu_alpha[tt][kk], &ls_alpha[tt][kk], &prior_mean, &plv);
                    let inv_pv = 1.0 / (delta + 1e-6);
                    for ll in 0..l {
                        let dm = mu_alpha[tt][kk][ll] - prior_mean[ll];
                        g_mu_alpha[tt][kk][ll] += inv_pv * dm;
                        g_ls_alpha[tt][kk][ll] += 0.5 * (ls_alpha[tt][kk][ll].exp() * inv_pv - 1.0);
                        // The prior mean is the sample alpha_{t-1}, so KL pushes a
                        // gradient back onto alpha_{t-1} (its mu/ls via the eps).
                        let dprior = -inv_pv * dm;
                        let std = (0.5 * ls_alpha[tt - 1][kk][ll]).exp();
                        g_mu_alpha[tt - 1][kk][ll] += dprior;
                        g_ls_alpha[tt - 1][kk][ll] += dprior * eps_alpha[tt - 1][kk][ll] * 0.5 * std;
                    }
                }
            }

            // --- KL_eta (global, random walk over time). mu_t/ls_t are the LSTM
            // head outputs; the prior mean for t >= 1 is the *sampled* eta_{t-1}, so
            // its KL gradient is routed back onto the sampled eta_{t-1}. ---
            let mu_eta_h = &eta_fwd.mu;
            let ls_eta_h = &eta_fwd.ls;
            for kk in 0..k {
                {
                    let dm = mu_eta_h[0][kk];
                    batch_loss += 0.5
                        * ((ls_eta_h[0][kk].exp() + dm * dm) / (1.0 + 1e-6) - 1.0 + 0.0
                            - ls_eta_h[0][kk]);
                    d_mu_eta[0][kk] += dm / (1.0 + 1e-6);
                    d_ls_eta[0][kk] += 0.5 * (ls_eta_h[0][kk].exp() / (1.0 + 1e-6) - 1.0);
                }
                for tt in 1..t {
                    let prior_mean = eta[tt - 1][kk]; // sampled eta_{t-1}
                    let dm = mu_eta_h[tt][kk] - prior_mean;
                    let inv_pv = 1.0 / (delta + 1e-6);
                    batch_loss += 0.5
                        * ((ls_eta_h[tt][kk].exp() + dm * dm) * inv_pv - 1.0 + log_delta
                            - ls_eta_h[tt][kk]);
                    d_mu_eta[tt][kk] += inv_pv * dm;
                    d_ls_eta[tt][kk] += 0.5 * (ls_eta_h[tt][kk].exp() * inv_pv - 1.0);
                    // d KL_t / d (prior mean = sampled eta_{t-1}).
                    d_eta[tt - 1][kk] += -inv_pv * dm;
                }
            }

            // --- q(eta) backprop-through-time, once for the whole sequence. ---
            let g_eta = eta_net.backward(
                &eta_fwd, &d_eta, &d_mu_eta, &d_ls_eta, &eps_eta, true, &rnn_input,
            );

            // --- Adam updates. ---
            step3d(&mut a_mu_alpha, &mut mu_alpha, &g_mu_alpha, t, k, l);
            step3d(&mut a_ls_alpha, &mut ls_alpha, &g_ls_alpha, t, k, l);
            // q(eta) network updates.
            a_eta_map_w.step(&mut eta_net.map_w, &g_eta.map_w);
            a_eta_map_b.step(&mut eta_net.map_b, &g_eta.map_b);
            for layer in 0..eta_nlayers {
                a_eta_w_ih[layer].step(&mut eta_net.w_ih[layer], &g_eta.w_ih[layer]);
                a_eta_w_hh[layer].step(&mut eta_net.w_hh[layer], &g_eta.w_hh[layer]);
                a_eta_b_ih[layer].step(&mut eta_net.b_ih[layer], &g_eta.b_ih[layer]);
                a_eta_b_hh[layer].step(&mut eta_net.b_hh[layer], &g_eta.b_hh[layer]);
            }
            a_eta_mu_w.step(&mut eta_net.mu_w, &g_eta.mu_w);
            a_eta_mu_b.step(&mut eta_net.mu_b, &g_eta.mu_b);
            a_eta_ls_w.step(&mut eta_net.ls_w, &g_eta.ls_w);
            a_eta_ls_b.step(&mut eta_net.ls_b, &g_eta.ls_b);
            a_w1.step(&mut enc.w1, &g_enc.w1);
            a_b1.step(&mut enc.b1, &g_enc.b1);
            a_w2.step(&mut enc.w2, &g_enc.w2);
            a_b2.step(&mut enc.b2, &g_enc.b2);
            a_w_mu.step(&mut enc.w_mu, &g_enc.w_mu);
            a_b_mu.step(&mut enc.b_mu, &g_enc.b_mu);
            a_w_ls.step(&mut enc.w_ls, &g_enc.w_ls);
            a_b_ls.step(&mut enc.b_ls, &g_enc.b_ls);

            epoch_loss += batch_loss / coeff; // report per-doc-scaled (== /num_docs) loss
            batches += 1;
        }

        let avg = if batches > 0 { epoch_loss / batches as f64 } else { f64::NAN };
        bound_history.push(-avg); // ELBO == negative loss
        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (-avg - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }
    }

    // --- Eval pass: use the variational means (no sampling), as the reference does
    // at eval time (reparameterize returns mu). ---
    let alpha_mean = mu_alpha.clone();
    // q(eta) at eval: reparameterize returns the mean, so eta_t = mu_q_eta(
    // [lstm_out_t, eta_{t-1}]) sequentially using the *mean* eta_{t-1}.
    let zero_eps = vec![vec![0.0f64; k]; t];
    let eta_mean = eta_net.forward(&rnn_input, &zero_eps, false).etas;
    let beta_over_time: Vec<Vec<Vec<f64>>> = (0..t)
        .map(|tt| (0..k).map(|kk| beta_row(rho, &alpha_mean[tt][kk])).collect())
        .collect();

    // doc_topic from the encoder mean (theta = softmax(mu)), as in main.py's get_theta.
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let td = times[di];
            let cache = enc.forward(&nbows[di], &eta_mean[td]);
            softmax(&cache.mu)
        })
        .collect();

    DetmModel {
        num_topics: k,
        num_times: t,
        num_types: v,
        beta_over_time,
        doc_topic,
        alpha: alpha_mean,
        eta: eta_mean,
        bound: bound_history.last().copied().unwrap_or(f64::NAN),
        bound_history,
        converged,
        epochs_run,
    }
}

/// Adam step over a flattened (T x K x L) parameter block.
fn step3d(opt: &mut Adam, p: &mut [Vec<Vec<f64>>], g: &[Vec<Vec<f64>>], t: usize, k: usize, l: usize) {
    let mut flat_p = Vec::with_capacity(t * k * l);
    let mut flat_g = Vec::with_capacity(t * k * l);
    for tt in 0..t {
        for kk in 0..k {
            flat_p.extend_from_slice(&p[tt][kk]);
            flat_g.extend_from_slice(&g[tt][kk]);
        }
    }
    opt.step(&mut flat_p, &flat_g);
    let mut idx = 0;
    for tt in 0..t {
        for kk in 0..k {
            for ll in 0..l {
                p[tt][kk][ll] = flat_p[idx];
                idx += 1;
            }
        }
    }
}

/// Adam step over a flattened (T x K) parameter block.
fn step2d(opt: &mut Adam, p: &mut [Vec<f64>], g: &[Vec<f64>], t: usize, k: usize) {
    let mut flat_p = Vec::with_capacity(t * k);
    let mut flat_g = Vec::with_capacity(t * k);
    for tt in 0..t {
        flat_p.extend_from_slice(&p[tt]);
        flat_g.extend_from_slice(&g[tt]);
    }
    opt.step(&mut flat_p, &flat_g);
    let mut idx = 0;
    for tt in 0..t {
        for kk in 0..k {
            p[tt][kk] = flat_p[idx];
            idx += 1;
        }
    }
}

use crate::estimator::{Estimator, ModelFamily};

impl Estimator for DetmModel {
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        self.topic_word_mean()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        self.doc_topic.clone()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        self.bound_history.iter().enumerate().map(|(i, &b)| (i + 1, b)).collect()
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

    // A small planted time-stamped corpus: K blocks of words, with topic 0 rising
    // and topic K-1 falling across time. The word embeddings put each block on its
    // own axis so beta = softmax(alpha . rho) can separate them.
    fn planted_corpus<R: Rng>(
        rng: &mut R,
        k: usize,
        block: usize,
        t: usize,
        d_per_t: usize,
    ) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>, Vec<Vec<f64>>, usize) {
        let v = k * block;
        let l = k + 2;
        let rho: Vec<Vec<f64>> = (0..v)
            .map(|w| {
                let b = w / block;
                (0..l)
                    .map(|dim| if dim == b { 3.0 } else { 0.0 } + (rng.gen::<f64>() - 0.5) * 0.1)
                    .collect()
            })
            .collect();
        let mut tokens = Vec::new();
        let mut counts = Vec::new();
        let mut times = Vec::new();
        for tt in 0..t {
            let frac = tt as f64 / (t - 1).max(1) as f64;
            let mut base = vec![0.25; k];
            base[0] = 0.4 * frac + 0.1;
            base[k - 1] = 0.4 * (1.0 - frac) + 0.1;
            let s: f64 = base.iter().sum();
            for b in base.iter_mut() {
                *b /= s;
            }
            for _ in 0..d_per_t {
                let mut wc = vec![0u32; v];
                let length = 30;
                for _ in 0..length {
                    let r: f64 = rng.gen();
                    let mut acc = 0.0;
                    let mut kk = 0;
                    for (ii, &bp) in base.iter().enumerate() {
                        acc += bp;
                        if r <= acc {
                            kk = ii;
                            break;
                        }
                    }
                    let w = kk * block + (rng.gen::<f64>() * block as f64) as usize;
                    wc[w.min(v - 1)] += 1;
                }
                let toks: Vec<u32> = (0..v as u32).filter(|&w| wc[w as usize] > 0).collect();
                let cnts: Vec<u32> = toks.iter().map(|&w| wc[w as usize]).collect();
                tokens.push(toks);
                counts.push(cnts);
                times.push(tt);
            }
        }
        (tokens, counts, times, rho, v)
    }

    #[test]
    fn beta_rows_are_distributions() {
        let rho = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let row = beta_row(&rho, &[2.0, -1.0]);
        assert_eq!(row.len(), 3);
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(row.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn kl_zero_when_equal() {
        let mu = vec![0.3, -0.5, 1.0];
        let lv = vec![0.1, -0.2, 0.0];
        let kl = kl_gauss(&mu, &lv, &mu, &lv);
        // Not exactly zero: the reference's get_kl adds 1e-6 to the prior variance
        // for numerical safety, so KL(q || q) is a tiny negative number. We match
        // that form, so we only require it is close to zero.
        assert!(kl.abs() < 1e-4, "KL of a distribution with itself should be ~0, got {kl}");
    }

    #[test]
    fn fit_detm_shapes_and_distributions() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (k, block, t, d_per_t) = (3usize, 6usize, 4usize, 20usize);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng, k, block, t, d_per_t);
        let m = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 32, 16, 2, 30, 64, 0.02, 1.2e-6, 0.0, &mut rng,
        );
        assert_eq!(m.num_topics, k);
        assert_eq!(m.num_times, t);
        assert_eq!(m.beta_over_time.len(), t);
        assert_eq!(m.beta_over_time[0].len(), k);
        assert_eq!(m.beta_over_time[0][0].len(), v);
        for bt in &m.beta_over_time {
            for row in bt {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
        }
        assert_eq!(m.doc_topic.len(), tokens.len());
        for row in &m.doc_topic {
            assert_eq!(row.len(), k);
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.topic_word_mean() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert_eq!(m.eta.len(), t);
        assert_eq!(m.eta[0].len(), k);
        assert_eq!(m.alpha.len(), t);
        assert_eq!(m.alpha[0].len(), k);
    }

    #[test]
    fn fit_detm_is_deterministic() {
        let (k, block, t, d_per_t) = (3usize, 6usize, 4usize, 15usize);
        let mut rng0 = ChaCha8Rng::seed_from_u64(7);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng0, k, block, t, d_per_t);

        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let a = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 16, 16, 2, 20, 64, 0.02, 1.2e-6, 0.0, &mut rng_a,
        );
        let mut rng_b = ChaCha8Rng::seed_from_u64(123);
        let b = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 16, 16, 2, 20, 64, 0.02, 1.2e-6, 0.0, &mut rng_b,
        );
        for tt in 0..t {
            for kk in 0..k {
                assert_eq!(a.beta_over_time[tt][kk], b.beta_over_time[tt][kk]);
            }
        }
        assert_eq!(a.doc_topic, b.doc_topic);
        assert_eq!(a.eta, b.eta);
    }

    #[test]
    fn fit_detm_recovers_temporal_drift() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (k, block, t, d_per_t) = (3usize, 8usize, 5usize, 40usize);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng, k, block, t, d_per_t);
        let m = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 32, 32, 2, 300, 1000, 0.02, 1.2e-6, 0.0,
            &mut rng,
        );
        // The time-varying topic prior eta (the latent the random walk regularizes)
        // should move across the slices for at least one topic, recovering the
        // planted prevalence drift (topic 0 rises, topic k-1 falls). softmax(eta_t)
        // is the prior over topic proportions at slice t; we check its first-to-last
        // change. The LSTM amortizer of q(eta) regularizes the trajectory toward a
        // smoother path than the planted one, so the recovered drift on a tiny corpus
        // is real but modest; we require it clearly moves off flat.
        let prior_at = |tt: usize| softmax(&m.eta[tt]);
        let p0 = prior_at(0);
        let plast = prior_at(t - 1);
        let max_drift = (0..k).map(|kk| (plast[kk] - p0[kk]).abs()).fold(0.0f64, f64::max);
        assert!(max_drift > 0.01, "eta prior did not drift across time (max {max_drift})");
    }

    // ---- Finite-difference check on the LSTM q(eta) path --------------------
    //
    // Loss that exercises the full q(eta) machinery: a linear downstream term on
    // the sampled eta (`sum_t w_t . eta_t`, gradient `w` into the reparameterized
    // eta) plus the exact random-walk eta-KL (prior mean = sampled eta_{t-1}, log-
    // variance log(delta) for t>=1, unit at t=0). This is the same gradient assembly
    // the training loop performs for q(eta). Returns (loss, the three backward
    // inputs) so the analytic gradient can be produced by EtaNet::backward.
    #[allow(clippy::type_complexity)]
    fn eta_fd_loss(
        net: &EtaNet,
        rnn_input: &[Vec<f64>],
        eps: &[Vec<f64>],
        w: &[Vec<f64>],
        delta: f64,
    ) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (k, t) = (net.k, rnn_input.len());
        let log_delta = delta.max(1e-12).ln();
        let fwd = net.forward(rnn_input, eps, true);
        let mut loss = 0.0;
        let mut d_eta = vec![vec![0.0f64; k]; t];
        let mut d_mu = vec![vec![0.0f64; k]; t];
        let mut d_ls = vec![vec![0.0f64; k]; t];
        // downstream linear term on sampled eta.
        for tt in 0..t {
            for c in 0..k {
                loss += w[tt][c] * fwd.etas[tt][c];
                d_eta[tt][c] += w[tt][c];
            }
        }
        // eta-KL (heads' mu/ls, sampled-eta prior mean).
        for c in 0..k {
            let dm = fwd.mu[0][c];
            loss += 0.5 * ((fwd.ls[0][c].exp() + dm * dm) / (1.0 + 1e-6) - 1.0 - fwd.ls[0][c]);
            d_mu[0][c] += dm / (1.0 + 1e-6);
            d_ls[0][c] += 0.5 * (fwd.ls[0][c].exp() / (1.0 + 1e-6) - 1.0);
            for tt in 1..t {
                let prior = fwd.etas[tt - 1][c];
                let dmm = fwd.mu[tt][c] - prior;
                let inv_pv = 1.0 / (delta + 1e-6);
                loss += 0.5
                    * ((fwd.ls[tt][c].exp() + dmm * dmm) * inv_pv - 1.0 + log_delta - fwd.ls[tt][c]);
                d_mu[tt][c] += inv_pv * dmm;
                d_ls[tt][c] += 0.5 * (fwd.ls[tt][c].exp() * inv_pv - 1.0);
                d_eta[tt - 1][c] += -inv_pv * dmm;
            }
        }
        (loss, d_eta, d_mu, d_ls)
    }

    #[test]
    fn eta_lstm_bptt_matches_finite_difference() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (v, k, eh, nl, t) = (5usize, 3usize, 4usize, 2usize, 4usize);
        let delta = 0.005;
        let net = EtaNet::new(v, k, eh, nl, &mut rng);
        // Fixed inputs / noise / downstream weights.
        let rnn_input: Vec<Vec<f64>> =
            (0..t).map(|_| (0..v).map(|_| rng.gen::<f64>()).collect()).collect();
        let eps: Vec<Vec<f64>> =
            (0..t).map(|_| (0..k).map(|_| randn(&mut rng)).collect()).collect();
        let w: Vec<Vec<f64>> =
            (0..t).map(|_| (0..k).map(|_| randn(&mut rng) * 0.3).collect()).collect();

        // Analytic gradient via BPTT.
        let (_, d_eta, d_mu, d_ls) = eta_fd_loss(&net, &rnn_input, &eps, &w, delta);
        let g = net.backward(&net.forward(&rnn_input, &eps, true), &d_eta, &d_mu, &d_ls, &eps, true, &rnn_input);

        let h = 1e-5;
        // Closure: perturb element `idx` of a chosen parameter block and recompute loss.
        let mut max_rel: f64 = 0.0;
        let mut check = |get: &dyn Fn(&mut EtaNet) -> *mut Vec<f64>, grad: &[f64], name: &str| {
            // Sample a handful of indices per block to keep the test fast.
            let len = grad.len();
            let stride = (len / 12).max(1);
            for idx in (0..len).step_by(stride) {
                let mut np = net.clone();
                let p = unsafe { &mut *get(&mut np) };
                let orig = p[idx];
                p[idx] = orig + h;
                let (lp, ..) = eta_fd_loss(&np, &rnn_input, &eps, &w, delta);
                let p = unsafe { &mut *get(&mut np) };
                p[idx] = orig - h;
                let (lm, ..) = eta_fd_loss(&np, &rnn_input, &eps, &w, delta);
                let num = (lp - lm) / (2.0 * h);
                let ana = grad[idx];
                let rel = (num - ana).abs() / (num.abs().max(ana.abs()).max(1e-6));
                assert!(
                    rel < 1e-4,
                    "{name}[{idx}]: analytic {ana:.6e} vs numeric {num:.6e} (rel {rel:.2e})"
                );
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        };
        check(&|n| &mut n.map_w as *mut _, &g.map_w, "map_w");
        check(&|n| &mut n.map_b as *mut _, &g.map_b, "map_b");
        check(&|n| &mut n.w_ih[0] as *mut _, &g.w_ih[0], "w_ih0");
        check(&|n| &mut n.w_hh[0] as *mut _, &g.w_hh[0], "w_hh0");
        check(&|n| &mut n.b_ih[0] as *mut _, &g.b_ih[0], "b_ih0");
        check(&|n| &mut n.b_hh[0] as *mut _, &g.b_hh[0], "b_hh0");
        check(&|n| &mut n.w_ih[1] as *mut _, &g.w_ih[1], "w_ih1");
        check(&|n| &mut n.w_hh[1] as *mut _, &g.w_hh[1], "w_hh1");
        check(&|n| &mut n.b_ih[1] as *mut _, &g.b_ih[1], "b_ih1");
        check(&|n| &mut n.b_hh[1] as *mut _, &g.b_hh[1], "b_hh1");
        check(&|n| &mut n.mu_w as *mut _, &g.mu_w, "mu_w");
        check(&|n| &mut n.mu_b as *mut _, &g.mu_b, "mu_b");
        check(&|n| &mut n.ls_w as *mut _, &g.ls_w, "ls_w");
        check(&|n| &mut n.ls_b as *mut _, &g.ls_b, "ls_b");
        // (max_rel is asserted per-element above; this keeps the variable observed.)
        assert!(max_rel < 1e-4, "max relative FD error {max_rel:.2e}");
    }

    #[test]
    fn detm_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (k, block, t, d_per_t) = (3usize, 6usize, 4usize, 20usize);
        let (tokens, counts, times, rho, v) = planted_corpus(&mut rng, k, block, t, d_per_t);
        let m = fit_detm(
            &tokens, &counts, &times, k, v, t, &rho, 0.005, 16, 16, 2, 25, 64, 0.02, 1.2e-6, 0.0, &mut rng,
        );
        let base = crate::conformance::check_conformance(&m);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
