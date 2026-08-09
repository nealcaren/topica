//! MGLDA: Multi-Grain LDA (Titov & McDonald, "Modeling Online Reviews with
//! Multi-Grain Topic Models," WWW 2008).
//!
//! Each document is a sequence of sentences. A sliding window of `T` consecutive
//! sentences defines the local context (windows overlap). There are `K_gl` GLOBAL
//! topics (the document-level subject) and `K_loc` LOCAL topics (rateable aspects
//! over a window). Every token first picks one of the `T` windows covering its
//! sentence, then a grain (global or local), then a topic from the appropriate
//! distribution, then a word.
//!
//! Inference is collapsed Gibbs from the paper: the per-token triple
//! `(window v, grain r, topic z)` is resampled jointly. For a token in sentence `s`
//! (word `w`), over candidate windows `v = s+j` (`j=0..T-1`):
//!
//!   P(v, gl, z) ∝ (n_{d,s,v}+γ) · π_gl(v) · (n^gl_{d,z}+α_gl)/(n^gl_{d,·}+K_gl·α_gl)
//!                 · (n^gl_{z,w}+β_gl)/(n^gl_z+V·β_gl)
//!   P(v, loc, z) ∝ (n_{d,s,v}+γ) · π_loc(v) · (n^loc_{d,v,z}+α_loc)/(n^loc_{d,v,·}+K_loc·α_loc)
//!                 · (n^loc_{z,w}+β_loc)/(n^loc_z+V·β_loc)
//!
//! with π_gl(v)=(g^gl_{d,v}+α_mg)/(g_{d,v}+α_mg+α_ml), π_loc(v) analogous. The
//! per-sentence ψ denominator `(len_s−1+T·γ)` is constant across a token's candidates
//! and is dropped. The current token is removed from every count before scoring.
//!
//! NOTE ON THE PAPER vs THE REFERENCE: the paper's local-topic Gibbs numerator is the
//! smoothed `(n^loc_{d,v,z} + α_loc)`. The canonical implementation (tomotopy) OMITS
//! `+α_loc` from the local numerator (using the raw window-local count) while keeping
//! `K_loc·α_loc` in the denominator; the global term uses the smoothed `+α_gl`. This
//! module follows the REFERENCE (unsmoothed local numerator): with everything else in
//! the conditional provably identical to tomotopy, the `+α_loc` variant does not track
//! the reference and — because it floors every window-local topic — fails to break the
//! global/local grain symmetry (the grain switch stalls at ~50/50 with local topics
//! absorbing document-wide words). The reference's unsmoothed numerator is what makes
//! the two grains separate, so it is used and validated against tomotopy.
//!
//! Determinism: single-threaded; every draw from the seeded `rng`; documents,
//! sentences, and tokens processed in fixed order. Outputs are `Vec<Vec<f64>>` (no
//! `ndarray`) so a default build carries no `embeddings` dependency.

use crate::estimator::{Estimator, ModelFamily};
use rand::Rng;

/// Fitted Multi-Grain LDA state read back by the PyO3 binding.
pub struct MgLdaModel {
    pub k_gl: usize,
    pub k_loc: usize,
    pub window: usize,
    pub alpha_gl: f64,
    pub alpha_loc: f64,
    pub alpha_mix_gl: f64,
    pub alpha_mix_loc: f64,
    pub beta_gl: f64,
    pub beta_loc: f64,
    pub gamma: f64,
    /// Global topic-word φ^gl (K_gl × V), each row a distribution.
    pub global_topic_word: Vec<Vec<f64>>,
    /// Local topic-word φ^loc (K_loc × V), each row a distribution.
    pub local_topic_word: Vec<Vec<f64>>,
    /// vstack(global, local) — (K_gl+K_loc) × V; the standard `topic_word` surface.
    pub topic_word: Vec<Vec<f64>>,
    /// Per-document empirical topic prevalence over [global | local] (D × (K_gl+K_loc)):
    /// the proportions of each document's token assignments across the combined topic
    /// set. Rows sum to 1. NOT a single generative Dirichlet θ (global is doc-level,
    /// local is window-level) — a content-based prevalence surface.
    pub doc_topic: Vec<Vec<f64>>,
    /// Document-level global topic distribution θ^gl (D × K_gl), smoothed. The true
    /// generative doc-level distribution.
    pub global_doc_topic: Vec<Vec<f64>>,
    /// Overall share of tokens assigned to the global grain (vs local).
    pub global_fraction: f64,
    pub fit_history: Vec<(usize, f64)>,
    pub converged: bool,
}

/// Fit MG-LDA by collapsed Gibbs.
///
/// - `docs`: sentence-segmented word-id documents (`doc → sentence → Vec<u32>`).
///   Empty sentences must already be removed by the caller; a document must have
///   >= 1 non-empty sentence.
/// - `num_types`: vocabulary size V.
#[allow(clippy::too_many_arguments)]
pub fn fit<R: Rng>(
    docs: &[Vec<Vec<u32>>],
    num_types: usize,
    k_gl: usize,
    k_loc: usize,
    window: usize,
    alpha_gl: f64,
    alpha_loc: f64,
    alpha_mix_gl: f64,
    alpha_mix_loc: f64,
    beta_gl: f64,
    beta_loc: f64,
    gamma: f64,
    iters: usize,
    rng: &mut R,
) -> MgLdaModel {
    let t = window;
    let v = num_types;
    let kk = k_gl + k_loc;
    let beta_gl_sum = beta_gl * v as f64;
    let beta_loc_sum = beta_loc * v as f64;

    // Global topic-word counts.
    let mut n_gl_zw = vec![vec![0u32; v]; k_gl];
    let mut n_gl_z = vec![0u32; k_gl];
    // Local topic-word counts.
    let mut n_loc_zw = vec![vec![0u32; v]; k_loc];
    let mut n_loc_z = vec![0u32; k_loc];

    // Per-document structures.
    let d = docs.len();
    // number of windows per doc = S_d + T - 1
    let n_win: Vec<usize> = docs.iter().map(|s| s.len() + t - 1).collect();
    // doc → global topic counts, doc global total
    let mut n_gl_dz = vec![vec![0u32; k_gl]; d];
    let mut n_gl_d = vec![0u32; d];
    // window → local topic counts (per doc: W_d × k_loc), window local totals
    let mut n_loc_dvz: Vec<Vec<Vec<u32>>> = (0..d)
        .map(|di| vec![vec![0u32; k_loc]; n_win[di]])
        .collect();
    let mut n_loc_dv: Vec<Vec<u32>> = (0..d).map(|di| vec![0u32; n_win[di]]).collect();
    // grain-global count per window (grain-local per window == n_loc_dv)
    let mut g_gl_dv: Vec<Vec<u32>> = (0..d).map(|di| vec![0u32; n_win[di]]).collect();
    // sentence → window selection counts (per doc: S_d sentences × T candidate windows)
    let mut n_dsv: Vec<Vec<Vec<u32>>> = docs
        .iter()
        .map(|sents| vec![vec![0u32; t]; sents.len()])
        .collect();

    // Per-token assignments: for each (doc, sentence) a Vec of (j, grain, z) parallel
    // to the sentence's tokens. j is the window offset (window = s + j); grain: false=gl.
    let mut asg: Vec<Vec<Vec<(u8, bool, u32)>>> = docs
        .iter()
        .map(|sents| {
            sents
                .iter()
                .map(|snt| vec![(0u8, false, 0u32); snt.len()])
                .collect()
        })
        .collect();

    // --- initialize ---
    for di in 0..d {
        for (s, sent) in docs[di].iter().enumerate() {
            for (pos, &w) in sent.iter().enumerate() {
                let j = rng.gen_range(0..t);
                let win = s + j;
                let is_loc = rng.gen::<bool>();
                if is_loc {
                    let z = rng.gen_range(0..k_loc);
                    asg[di][s][pos] = (j as u8, true, z as u32);
                    n_loc_zw[z][w as usize] += 1;
                    n_loc_z[z] += 1;
                    n_loc_dvz[di][win][z] += 1;
                    n_loc_dv[di][win] += 1;
                } else {
                    let z = rng.gen_range(0..k_gl);
                    asg[di][s][pos] = (j as u8, false, z as u32);
                    n_gl_zw[z][w as usize] += 1;
                    n_gl_z[z] += 1;
                    n_gl_dz[di][z] += 1;
                    n_gl_d[di] += 1;
                    g_gl_dv[di][win] += 1;
                }
                n_dsv[di][s][j] += 1;
            }
        }
    }

    let mut scores = vec![0.0f64; t * kk];
    let mut gl_wt = vec![0.0f64; k_gl]; // θ^gl(z)·φ^gl(z,w) per token
    let mut loc_phi = vec![0.0f64; k_loc]; // φ^loc(z,w) per token
    let eval_stride = (iters / 25).max(1);
    let mut fit_history = Vec::new();
    let mix_sum = alpha_mix_gl + alpha_mix_loc;

    // --- Gibbs sweeps ---
    for it in 0..iters {
        for di in 0..d {
            for s in 0..docs[di].len() {
                for pos in 0..docs[di][s].len() {
                    let w = docs[di][s][pos] as usize;
                    let (j0, loc0, z0) = asg[di][s][pos];
                    let (j0, z0) = (j0 as usize, z0 as usize);
                    let win0 = s + j0;

                    // remove current token
                    n_dsv[di][s][j0] -= 1;
                    if loc0 {
                        n_loc_zw[z0][w] -= 1;
                        n_loc_z[z0] -= 1;
                        n_loc_dvz[di][win0][z0] -= 1;
                        n_loc_dv[di][win0] -= 1;
                    } else {
                        n_gl_zw[z0][w] -= 1;
                        n_gl_z[z0] -= 1;
                        n_gl_dz[di][z0] -= 1;
                        n_gl_d[di] -= 1;
                        g_gl_dv[di][win0] -= 1;
                    }

                    // per-token grain-independent terms
                    let gl_denom = n_gl_d[di] as f64 + k_gl as f64 * alpha_gl;
                    for z in 0..k_gl {
                        let theta = (n_gl_dz[di][z] as f64 + alpha_gl) / gl_denom;
                        let phi =
                            (n_gl_zw[z][w] as f64 + beta_gl) / (n_gl_z[z] as f64 + beta_gl_sum);
                        gl_wt[z] = theta * phi;
                    }
                    for z in 0..k_loc {
                        loc_phi[z] =
                            (n_loc_zw[z][w] as f64 + beta_loc) / (n_loc_z[z] as f64 + beta_loc_sum);
                    }

                    // score every (window j, grain, topic) candidate
                    let mut total = 0.0f64;
                    for j in 0..t {
                        let win = s + j;
                        let psi = n_dsv[di][s][j] as f64 + gamma;
                        let g_gl = g_gl_dv[di][win] as f64;
                        let g_loc = n_loc_dv[di][win] as f64;
                        let gd = g_gl + g_loc + mix_sum;
                        let pi_gl = psi * (g_gl + alpha_mix_gl) / gd;
                        let pi_loc = psi * (g_loc + alpha_mix_loc) / gd;
                        let base = j * kk;
                        for z in 0..k_gl {
                            let sc = pi_gl * gl_wt[z];
                            scores[base + z] = sc;
                            total += sc;
                        }
                        // Local-topic numerator matches the reference (tomotopy): the
                        // raw window-local count, WITHOUT +alpha_loc, while the
                        // denominator keeps K_loc*alpha_loc. See the module note on the
                        // paper-vs-reference discrepancy — the +alpha_loc form does not
                        // track the reference and fails to break the global/local grain
                        // symmetry, so the reference's unsmoothed numerator is used.
                        let loc_denom = g_loc + k_loc as f64 * alpha_loc;
                        for z in 0..k_loc {
                            let theta = (n_loc_dvz[di][win][z] as f64) / loc_denom;
                            let sc = pi_loc * theta * loc_phi[z];
                            scores[base + k_gl + z] = sc;
                            total += sc;
                        }
                    }

                    // sample a flat candidate index
                    let mut r = rng.gen::<f64>() * total;
                    let mut choice = t * kk - 1;
                    for (i, &sc) in scores.iter().enumerate().take(t * kk) {
                        r -= sc;
                        if r <= 0.0 {
                            choice = i;
                            break;
                        }
                    }
                    let j = choice / kk;
                    let rem = choice % kk;
                    let win = s + j;

                    // apply
                    n_dsv[di][s][j] += 1;
                    if rem < k_gl {
                        let z = rem;
                        asg[di][s][pos] = (j as u8, false, z as u32);
                        n_gl_zw[z][w] += 1;
                        n_gl_z[z] += 1;
                        n_gl_dz[di][z] += 1;
                        n_gl_d[di] += 1;
                        g_gl_dv[di][win] += 1;
                    } else {
                        let z = rem - k_gl;
                        asg[di][s][pos] = (j as u8, true, z as u32);
                        n_loc_zw[z][w] += 1;
                        n_loc_z[z] += 1;
                        n_loc_dvz[di][win][z] += 1;
                        n_loc_dv[di][win] += 1;
                    }
                }
            }
        }

        if it % eval_stride == 0 || it == iters - 1 {
            let ll = held_in_loglik(
                docs,
                &n_gl_zw,
                &n_gl_z,
                &n_loc_zw,
                &n_loc_z,
                &n_gl_dz,
                &n_gl_d,
                &n_loc_dvz,
                &n_loc_dv,
                &g_gl_dv,
                &n_dsv,
                k_gl,
                k_loc,
                t,
                alpha_gl,
                alpha_loc,
                alpha_mix_gl,
                alpha_mix_loc,
                beta_gl,
                beta_loc,
                beta_gl_sum,
                beta_loc_sum,
                gamma,
            );
            fit_history.push((it + 1, ll));
        }
    }

    // --- materialize ---
    let global_topic_word: Vec<Vec<f64>> = (0..k_gl)
        .map(|z| {
            let denom = n_gl_z[z] as f64 + beta_gl_sum;
            (0..v)
                .map(|w| (n_gl_zw[z][w] as f64 + beta_gl) / denom)
                .collect()
        })
        .collect();
    let local_topic_word: Vec<Vec<f64>> = (0..k_loc)
        .map(|z| {
            let denom = n_loc_z[z] as f64 + beta_loc_sum;
            (0..v)
                .map(|w| (n_loc_zw[z][w] as f64 + beta_loc) / denom)
                .collect()
        })
        .collect();
    let mut topic_word = global_topic_word.clone();
    topic_word.extend(local_topic_word.clone());

    // empirical per-doc prevalence over [global | local]
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let mut row = vec![0.0f64; kk];
            for z in 0..k_gl {
                row[z] = n_gl_dz[di][z] as f64;
            }
            for win in 0..n_win[di] {
                for z in 0..k_loc {
                    row[k_gl + z] += n_loc_dvz[di][win][z] as f64;
                }
            }
            let tot: f64 = row.iter().sum();
            if tot > 0.0 {
                for x in &mut row {
                    *x /= tot;
                }
            } else {
                for x in &mut row {
                    *x = 1.0 / kk as f64;
                }
            }
            row
        })
        .collect();

    let global_doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|di| {
            let denom = n_gl_d[di] as f64 + k_gl as f64 * alpha_gl;
            (0..k_gl)
                .map(|z| (n_gl_dz[di][z] as f64 + alpha_gl) / denom)
                .collect()
        })
        .collect();

    let total_gl: u64 = n_gl_z.iter().map(|&x| x as u64).sum();
    let total_loc: u64 = n_loc_z.iter().map(|&x| x as u64).sum();
    let global_fraction = if total_gl + total_loc > 0 {
        total_gl as f64 / (total_gl + total_loc) as f64
    } else {
        0.0
    };

    MgLdaModel {
        k_gl,
        k_loc,
        window,
        alpha_gl,
        alpha_loc,
        alpha_mix_gl,
        alpha_mix_loc,
        beta_gl,
        beta_loc,
        gamma,
        global_topic_word,
        local_topic_word,
        topic_word,
        doc_topic,
        global_doc_topic,
        global_fraction,
        fit_history,
        converged: false,
    }
}

/// Held-in token log-likelihood under current point estimates (a convergence
/// diagnostic): Σ_tokens log Σ_v ψ(v)·[π_gl(v)·Σ_z θ^gl θ φ^gl + π_loc(v)·Σ_z θ^loc φ^loc],
/// normalized by the per-sentence ψ denominator so it is a proper mixture.
#[allow(clippy::too_many_arguments)]
fn held_in_loglik(
    docs: &[Vec<Vec<u32>>],
    n_gl_zw: &[Vec<u32>],
    n_gl_z: &[u32],
    n_loc_zw: &[Vec<u32>],
    n_loc_z: &[u32],
    n_gl_dz: &[Vec<u32>],
    n_gl_d: &[u32],
    n_loc_dvz: &[Vec<Vec<u32>>],
    n_loc_dv: &[Vec<u32>],
    g_gl_dv: &[Vec<u32>],
    n_dsv: &[Vec<Vec<u32>>],
    k_gl: usize,
    k_loc: usize,
    t: usize,
    alpha_gl: f64,
    alpha_loc: f64,
    alpha_mix_gl: f64,
    alpha_mix_loc: f64,
    beta_gl: f64,
    beta_loc: f64,
    beta_gl_sum: f64,
    beta_loc_sum: f64,
    gamma: f64,
) -> f64 {
    let mix_sum = alpha_mix_gl + alpha_mix_loc;
    let mut ll = 0.0f64;
    for di in 0..docs.len() {
        let gl_denom = n_gl_d[di] as f64 + k_gl as f64 * alpha_gl;
        for (s, sent) in docs[di].iter().enumerate() {
            let psi_denom: f64 =
                n_dsv[di][s].iter().map(|&c| c as f64).sum::<f64>() + t as f64 * gamma;
            for &w in sent {
                let w = w as usize;
                let mut p = 0.0f64;
                for j in 0..t {
                    let win = s + j;
                    let psi = (n_dsv[di][s][j] as f64 + gamma) / psi_denom;
                    let g_gl = g_gl_dv[di][win] as f64;
                    let g_loc = n_loc_dv[di][win] as f64;
                    let gd = g_gl + g_loc + mix_sum;
                    let pi_gl = (g_gl + alpha_mix_gl) / gd;
                    let pi_loc = (g_loc + alpha_mix_loc) / gd;
                    let mut gl_sum = 0.0f64;
                    for z in 0..k_gl {
                        let theta = (n_gl_dz[di][z] as f64 + alpha_gl) / gl_denom;
                        let phi =
                            (n_gl_zw[z][w] as f64 + beta_gl) / (n_gl_z[z] as f64 + beta_gl_sum);
                        gl_sum += theta * phi;
                    }
                    let loc_denom = g_loc + k_loc as f64 * alpha_loc;
                    let mut loc_sum = 0.0f64;
                    for z in 0..k_loc {
                        let theta = (n_loc_dvz[di][win][z] as f64) / loc_denom;
                        let phi =
                            (n_loc_zw[z][w] as f64 + beta_loc) / (n_loc_z[z] as f64 + beta_loc_sum);
                        loc_sum += theta * phi;
                    }
                    p += psi * (pi_gl * gl_sum + pi_loc * loc_sum);
                }
                if p > 0.0 {
                    ll += p.ln();
                }
            }
        }
    }
    ll
}

impl Estimator for MgLdaModel {
    fn num_topics(&self) -> usize {
        self.k_gl + self.k_loc
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
        // Combined [global|local] doc_topic is an empirical prevalence, not a single
        // Dirichlet posterior (global is doc-level, local is window-level), so it must
        // NOT advertise the Dirichlet contract. None_ skips the simplex/Dirichlet check.
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Planted corpus: 2 global THEMES (doc-level, words 0..5 in two 3-word blocks)
    // × 3 local ASPECTS (per-sentence, words 6..14 in three 3-word blocks). Theme
    // words recur in every sentence of a doc (document-wide → global); aspect words
    // vary per sentence (→ local). V=15.
    const THEME_BLOCK: usize = 3;
    const ASPECT_BLOCK: usize = 3;
    const N_THEMES: usize = 2;
    const N_ASPECTS: usize = 3;

    fn planted() -> (Vec<Vec<Vec<u32>>>, usize) {
        let v = N_THEMES * THEME_BLOCK + N_ASPECTS * ASPECT_BLOCK; // 15
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let mut docs = Vec::new();
        for _ in 0..80 {
            let g = rng.gen_range(0..N_THEMES);
            let mut sents = Vec::new();
            for _ in 0..6 {
                let a = rng.gen_range(0..N_ASPECTS);
                let mut sent = Vec::new();
                for _ in 0..3 {
                    sent.push((g * THEME_BLOCK + rng.gen_range(0..THEME_BLOCK)) as u32);
                }
                for _ in 0..3 {
                    let base = N_THEMES * THEME_BLOCK + a * ASPECT_BLOCK;
                    sent.push((base + rng.gen_range(0..ASPECT_BLOCK)) as u32);
                }
                sents.push(sent);
            }
            docs.push(sents);
        }
        (docs, v)
    }

    fn theme_words() -> std::ops::Range<usize> {
        0..(N_THEMES * THEME_BLOCK)
    }
    fn aspect_words() -> std::ops::Range<usize> {
        (N_THEMES * THEME_BLOCK)..(N_THEMES * THEME_BLOCK + N_ASPECTS * ASPECT_BLOCK)
    }

    #[test]
    fn mglda_recovers_global_themes() {
        // GLOBAL topics recover the document-level themes: on this corpus each global
        // topic is topped by a theme word and the two split into distinct theme blocks.
        // (This is the robust signal. The local grain's survival vs collapse is highly
        // data-dependent for MG-LDA — the reference tomotopy collapses to near-all-global
        // on clean synthetic corpora too — so local behavior is validated against
        // tomotopy in parity/mglda_gold.py rather than asserted on synthetic data.)
        let (docs, v) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let m = fit(
            &docs, v, 2, 3, 3, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 400, &mut rng,
        );
        let argmax = |r: &[f64]| {
            (0..v)
                .max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap())
                .unwrap()
        };
        assert!(m.global_fraction >= 0.0 && m.global_fraction <= 1.0);
        let g_top: Vec<usize> = m.global_topic_word.iter().map(|r| argmax(r)).collect();
        assert!(
            g_top.iter().all(|&w| theme_words().contains(&w)),
            "global tops not themes: {g_top:?}"
        );
        assert_eq!(
            g_top
                .iter()
                .map(|&w| w / THEME_BLOCK)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            N_THEMES,
            "global topics did not split by theme: {g_top:?}"
        );
        // Local topic-word rows are valid distributions (grain machinery intact).
        for row in &m.local_topic_word {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        let _ = aspect_words();
    }

    #[test]
    fn mglda_is_deterministic() {
        let (docs, v) = planted();
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(7);
        let a = fit(
            &docs, v, 2, 3, 3, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 60, &mut r1,
        );
        let b = fit(
            &docs, v, 2, 3, 3, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 60, &mut r2,
        );
        assert_eq!(a.global_topic_word, b.global_topic_word);
        assert_eq!(a.local_topic_word, b.local_topic_word);
        assert_eq!(a.doc_topic, b.doc_topic);
        assert_eq!(a.global_fraction, b.global_fraction);
    }

    #[test]
    fn mglda_conforms() {
        let (docs, v) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let m = fit(
            &docs, v, 2, 3, 3, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 20, &mut rng,
        );
        assert!(crate::conformance::check_conformance(&m).is_empty());
        // topic_word rows sum to 1; doc_topic rows sum to 1
        for row in &m.topic_word {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        for row in &m.doc_topic {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    /// Short documents (S < T) and single-sentence docs must not panic.
    #[test]
    fn mglda_handles_short_docs() {
        let docs: Vec<Vec<Vec<u32>>> = vec![
            vec![vec![0, 1, 2]],          // S=1 < T=3
            vec![vec![0, 1], vec![2, 3]], // S=2 < T=3
            vec![vec![3, 4, 0]],
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let m = fit(
            &docs, 5, 2, 2, 3, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 20, &mut rng,
        );
        assert_eq!(m.topic_word.len(), 4);
        assert_eq!(m.doc_topic.len(), 3);
    }

    /// T=1: each sentence is its own window (no overlap). Must work.
    #[test]
    fn mglda_window_one() {
        let (docs, v) = planted();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let m = fit(
            &docs, v, 2, 3, 1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 40, &mut rng,
        );
        assert_eq!(m.topic_word.len(), 5);
        assert!(m.global_fraction >= 0.0 && m.global_fraction <= 1.0);
    }
}
