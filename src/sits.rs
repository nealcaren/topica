//! SITS: parametric Speaker Identity for Topic Segmentation (Nguyen, Boyd-Graber &
//! Resnik, ACL 2012; Nguyen et al., Machine Learning 2014), as used by Rossiter
//! (2022, AJPS) to measure agenda setting in conversations.
//!
//! A corpus is a set of conversations, each a linear sequence of speaking turns with
//! an observed speaker. Every turn carries a topic-shift indicator `l`. The first
//! turn of a conversation always shifts; a later turn by speaker `m` shifts with
//! probability `π_m ~ Beta(γ, γ)`. A shift opens a new segment with its own topic
//! mixture `θ ~ Dir(α)`; a non-shift keeps the previous turn's mixture. Words are
//! LDA given the segment mixture: `z ~ Mult(θ)`, `w ~ Mult(φ_z)`, `φ_k ~ Dir(β)`.
//! `π_m`, the per-speaker shift propensity, is the agenda-setting measure.
//!
//! Inference is the collapsed Gibbs sampler of the reference Java implementation
//! (`vietansegan/sits`, `AuthorShiftSampler`, Apache-2.0; and Rossiter's fork
//! `erossiter/sits`). θ, φ and π are integrated out. Each sweep visits the turns in
//! corpus order and, for each turn, samples `l_t` (when the turn is eligible) and
//! then every token's `z`:
//!
//!   P(l_t = 0 | ·) ∝ (c_{m,0} + γ)/(c_m + 2γ) · DM(merged segment)
//!   P(l_t = 1 | ·) ∝ (c_{m,1} + γ)/(c_m + 2γ) · DM(pre) · DM(post)
//!   P(z = k | ·)   ∝ (n_{k,w} + β)/(n_k + Vβ) · (n_{seg,k} + α)
//!
//! where `DM(n) = lgamma(Kα) − K lgamma(α) + Σ_k lgamma(n_k + α) − lgamma(N + Kα)` is
//! the Dirichlet-multinomial marginal of a segment's topic counts, "pre" is the
//! segment before `t` (turns `< t`) and "post" the segment from `t` on. Because the
//! sweep is in turn order, "pre" is always the running sum of the already-visited
//! turns of the current segment, so every `l` draw is O(K) (the Java code splits
//! and merges segment objects; the arithmetic is identical).
//!
//! Reference behaviour this port reproduces, and the one place it departs:
//!
//! - **Speaker counts include forced turns.** As in both Java versions, every turn
//!   contributes to its speaker's `(c_{m,0}, c_{m,1})`: a conversation's first turn
//!   adds a permanent shift, and a turn too short to be sampled adds a permanent
//!   non-shift. (The generative story draws neither from `π`; we follow the code.)
//! - **Eligibility.** A non-first turn is sampled only if it has at least
//!   `min_shift_tokens` tokens (Rossiter's `>=` test; the original used `> 5`).
//! - **Initialization.** Every eligible turn starts as a shift with probability
//!   `init_shift_rate` (Rossiter's random start; the original started all at 0).
//! - **`Compat::Rossiter2022`** reproduces a bookkeeping defect in Rossiter's fork.
//!   The fork draws the random initial shift for *every* non-first turn, including
//!   ineligible (short) ones, as `nextInt(I) == 1`. It later overwrites an
//!   ineligible turn's recorded `l` with 0 but never removes its segment boundary or
//!   its shift count. Those "phantom" boundaries persist for the whole chain and
//!   inflate every speaker's shift prior. The default mode keeps the bookkeeping
//!   consistent (ineligible turns start and stay at 0); the compat mode exists so
//!   published results can be replicated.
//!
//! Recording follows Rossiter's `sitsr::readSits`: every post-burn-in sweep's `l`
//! is kept (no thinning) for the per-turn posterior shift probability, and the
//! per-speaker agenda-setting score of draw `s` is
//! `(γ + Σ_{t: a_t = m} l_t^{(s)}) / (2γ + n_m)` over all of the speaker's turns.
//!
//! Determinism: single-threaded; every draw comes from the seeded `rng` in a fixed
//! order, so a fixed seed reproduces bit-for-bit. Outputs are `Vec<Vec<f64>>` (no
//! `ndarray`), converted by the binding.

use crate::estimator::{Estimator, ModelFamily};
use crate::mathfun::log_gamma;
use rand::Rng;

/// How the sampler treats ineligible (short) turns at initialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compat {
    /// Consistent bookkeeping: ineligible turns start and stay at `l = 0`.
    None,
    /// Rossiter's fork: every non-first turn draws an initial shift with
    /// probability `1/I` (`gen_range(0..I) == 1`, so `I = 1` never shifts);
    /// ineligible turns that start as shifts become permanent phantom boundaries.
    Rossiter2022 { init_every: u32 },
}

/// Sampler configuration.
#[derive(Clone, Debug)]
pub struct SitsConfig {
    pub num_topics: usize,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub min_shift_tokens: usize,
    /// Initial shift probability for eligible turns (default mode only).
    pub init_shift_rate: f64,
    pub compat: Compat,
    /// Sweeps at the start during which the shift indicators stay frozen at their
    /// initial values and only topics are sampled. With `init_shift_rate = 1` this is
    /// a per-turn LDA warm start: the topics form before the segmentation is sampled,
    /// which lets the chain reach the posterior far sooner than the reference start
    /// (the stationary distribution is unchanged; keep `burn_in >= warmup`).
    pub warmup: usize,
    pub iters: usize,
    pub burn_in: usize,
    /// Keep every `sample_interval`-th post-burn-in draw of the per-speaker counts
    /// (for intervals). Posterior means always use every draw.
    pub sample_interval: usize,
}

/// The data a fit needs, all indexed by turn in conversation order.
pub struct SitsData<'a> {
    /// Word ids of each turn (may be empty).
    pub turns: &'a [Vec<u32>],
    /// Speaker id of each turn (`< num_speakers`).
    pub speakers: &'a [u32],
    /// `true` where a new conversation begins (must be `true` at turn 0).
    pub conv_start: &'a [bool],
    pub num_speakers: usize,
    pub num_types: usize,
}

/// Fitted SITS state read back by the PyO3 binding.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SitsModel {
    pub num_topics: usize,
    /// φ (K×V) from the terminal state: (n_{k,w}+β)/(n_k+Vβ), as the reference's
    /// `phi.txt`.
    pub topic_word: Vec<Vec<f64>>,
    /// θ (T×K): each turn's *segment* mixture in the terminal state,
    /// (n_{seg,k}+α)/(n_seg+Kα). This is the model's θ_{d,t}.
    pub doc_topic: Vec<Vec<f64>>,
    /// Per-turn smoothed topic proportions (T×K), (n_{t,k}+α)/(n_t+Kα): the
    /// reference's `theta.txt`.
    pub turn_topic: Vec<Vec<f64>>,
    /// Posterior mean of the recorded `l` per turn over post-burn-in sweeps.
    pub shift_prob: Vec<f64>,
    /// Whether each turn is sampled (non-first and long enough).
    pub eligible: Vec<bool>,
    /// Number of turns per speaker, and eligible turns per speaker.
    pub speaker_turns: Vec<u32>,
    pub speaker_eligible: Vec<u32>,
    /// Posterior mean over every post-burn-in draw of Σ_{t: a_t = m} l_t, and of the
    /// same sum over eligible turns only.
    pub speaker_shift_mean: Vec<f64>,
    pub speaker_eligible_shift_mean: Vec<f64>,
    /// Thinned post-burn-in draws (S×M) of the per-speaker shift sums over all turns
    /// and over eligible turns.
    pub speaker_shift_draws: Vec<Vec<u32>>,
    pub speaker_eligible_shift_draws: Vec<Vec<u32>>,
    /// Number of eligible turns recorded as shifts after each sweep (length = sweeps
    /// run): the convergence trace.
    pub shift_trace: Vec<u32>,
    /// Number of post-burn-in sweeps averaged.
    pub num_draws: usize,
    /// Sampler-internal speaker counts (M×2) in the terminal state. Equal to the
    /// recorded-l counts in default mode; carries phantom shifts in compat mode.
    pub sampler_speaker_counts: Vec<[u32; 2]>,
    /// Number of phantom boundaries (compat mode; 0 otherwise).
    pub num_phantom: usize,
    /// Collapsed log joint (speaker Beta-Bernoulli + topic-word DM + segment DMs)
    /// of the state after `sweep` sweeps, as `(sweep, value)`; the reference logs
    /// the same quantity before each sweep.
    pub fit_history: Vec<(usize, f64)>,
    pub sweeps_run: usize,
    pub converged: bool,
    /// Test-only: post-burn-in frequency that token 0 of turn 0 and token 0 of turn
    /// 3 share a topic (used by the exact-enumeration test).
    #[serde(skip)]
    pub debug_same_topic_freq: Option<f64>,
}

/// Lookup table for `lgamma(n + a)` over integer `n`.
struct LgTable {
    a: f64,
    t: Vec<f64>,
}

impl LgTable {
    fn new(a: f64, max_n: usize) -> Self {
        LgTable {
            a,
            t: (0..=max_n).map(|n| log_gamma(n as f64 + a)).collect(),
        }
    }
    #[inline]
    fn get(&self, n: u32) -> f64 {
        match self.t.get(n as usize) {
            Some(&v) => v,
            None => log_gamma(n as f64 + self.a),
        }
    }
}

/// Segment Dirichlet-multinomial marginal without the constant
/// `lgamma(Kα) − K lgamma(α)` (the caller adds it once per segment).
#[inline]
fn dm_var(counts: &[u32], total: u32, lg_a: &LgTable, lg_ka: &LgTable) -> f64 {
    let mut s = 0.0;
    for &c in counts {
        s += lg_a.get(c);
    }
    s - lg_ka.get(total)
}

/// Draw `l = 1` with probability `1 / (1 + exp(lp0 − lp1))`, overflow-safe. The
/// reference draws `u · (e^{lp0−lp1} + 1) < e^{lp0−lp1}` ⇒ 0; this is the same
/// Bernoulli without the `exp` overflow.
#[inline]
fn draw_shift<R: Rng>(lp0: f64, lp1: f64, rng: &mut R) -> u8 {
    let d = lp1 - lp0;
    let p1 = if d >= 0.0 {
        1.0 / (1.0 + (-d).exp())
    } else {
        let e = d.exp();
        e / (1.0 + e)
    };
    u8::from(rng.gen::<f64>() < p1)
}

/// Fit parametric SITS by collapsed Gibbs. `on_progress(sweep, total)` returns
/// `false` to stop early.
pub fn fit<R: Rng, F: FnMut(usize, usize) -> bool>(
    data: &SitsData<'_>,
    cfg: &SitsConfig,
    mut on_progress: F,
    rng: &mut R,
) -> SitsModel {
    let k = cfg.num_topics;
    let v = data.num_types;
    let nt = data.turns.len();
    let nm = data.num_speakers;
    let (alpha, beta, gamma) = (cfg.alpha, cfg.beta, cfg.gamma);
    let beta_sum = beta * v as f64;
    let total_tokens: usize = data.turns.iter().map(|t| t.len()).sum();

    // Lookup tables for the segment DM terms.
    let lg_a = LgTable::new(alpha, total_tokens);
    let lg_ka = LgTable::new(k as f64 * alpha, total_tokens);
    let dm_const = log_gamma(k as f64 * alpha) - k as f64 * log_gamma(alpha);

    // Eligibility.
    let eligible: Vec<bool> = (0..nt)
        .map(|t| !data.conv_start[t] && data.turns[t].len() >= cfg.min_shift_tokens)
        .collect();

    // --- state ---
    let mut z: Vec<Vec<u32>> = data.turns.iter().map(|d| vec![0u32; d.len()]).collect();
    let mut nwk = vec![0u32; v * k]; // word-major topic-word counts
    let mut nk = vec![0u32; k];
    let mut turn_counts = vec![0u32; nt * k];
    // Segment totals are stored at the segment's start turn.
    let mut seg_counts = vec![0u32; nt * k];
    let mut seg_n = vec![0u32; nt];
    let mut l = vec![0u8; nt]; // recorded shift indicator
    let mut boundary = vec![false; nt]; // true segment boundary (l, or a phantom)
    let mut spk = vec![[0u32; 2]; nm]; // sampler speaker counts (c_{m,0}, c_{m,1})

    // --- initialize (reference order: per turn, l then z's) ---
    let mut cur_start = 0usize;
    let mut num_phantom = 0usize;
    for t in 0..nt {
        let m = data.speakers[t] as usize;
        let shift = if data.conv_start[t] {
            true
        } else {
            match cfg.compat {
                Compat::None => eligible[t] && rng.gen::<f64>() < cfg.init_shift_rate,
                Compat::Rossiter2022 { init_every } => rng.gen_range(0..init_every) == 1,
            }
        };
        boundary[t] = shift;
        if shift && !eligible[t] && !data.conv_start[t] {
            num_phantom += 1;
        }
        l[t] = u8::from(shift && (eligible[t] || data.conv_start[t]));
        spk[m][usize::from(shift)] += 1;
        if shift {
            cur_start = t;
        }
        for (i, &w) in data.turns[t].iter().enumerate() {
            let topic = rng.gen_range(0..k);
            z[t][i] = topic as u32;
            nwk[w as usize * k + topic] += 1;
            nk[topic] += 1;
            turn_counts[t * k + topic] += 1;
            seg_counts[cur_start * k + topic] += 1;
            seg_n[cur_start] += 1;
        }
    }

    // --- recording buffers ---
    let mut shift_sum = vec![0u64; nt];
    let mut spk_sum = vec![0u64; nm];
    let mut spk_elig_sum = vec![0u64; nm];
    let mut spk_draws: Vec<Vec<u32>> = Vec::new();
    let mut spk_elig_draws: Vec<Vec<u32>> = Vec::new();
    let mut shift_trace: Vec<u32> = Vec::with_capacity(cfg.iters);
    let mut fit_history = Vec::new();
    let eval_stride = (cfg.iters / 200).max(1);
    let mut num_draws = 0usize;
    #[allow(unused_mut)]
    let mut same_topic = 0u64;
    let sample_interval = cfg.sample_interval.max(1);

    let mut pre = vec![0u32; k];
    let mut merged = vec![0u32; k];
    let mut post = vec![0u32; k];
    let mut p = vec![0.0f64; k];
    // 1/(n_k + Vβ), refreshed for the two topics a token move touches.
    let mut inv_den: Vec<f64> = nk.iter().map(|&c| 1.0 / (c as f64 + beta_sum)).collect();
    let mut sweeps_run = 0usize;

    for it in 0..cfg.iters {
        let frozen = it < cfg.warmup;
        let mut cur_start = 0usize;
        let mut pre_n = 0u32;
        for t in 0..nt {
            let m = data.speakers[t] as usize;
            if data.conv_start[t] {
                cur_start = t;
                pre.iter_mut().for_each(|x| *x = 0);
                pre_n = 0;
            } else if eligible[t] && !frozen {
                // `pre` holds the visited turns [cur_start, t) of the current segment.
                let cur = l[t];
                spk[m][usize::from(cur)] -= 1;
                let (mut mn, mut pn) = (seg_n[cur_start], 0u32);
                merged.copy_from_slice(&seg_counts[cur_start * k..cur_start * k + k]);
                if cur == 1 {
                    for kk in 0..k {
                        merged[kk] += seg_counts[t * k + kk];
                    }
                    mn += seg_n[t];
                }
                for kk in 0..k {
                    post[kk] = merged[kk] - pre[kk];
                }
                pn += mn - pre_n;
                let denom = (spk[m][0] + spk[m][1]) as f64 + 2.0 * gamma;
                let lp0 = dm_const
                    + dm_var(&merged, mn, &lg_a, &lg_ka)
                    + ((spk[m][0] as f64 + gamma) / denom).ln();
                let lp1 = 2.0 * dm_const
                    + dm_var(&pre, pre_n, &lg_a, &lg_ka)
                    + dm_var(&post, pn, &lg_a, &lg_ka)
                    + ((spk[m][1] as f64 + gamma) / denom).ln();
                let new = draw_shift(lp0, lp1, rng);
                spk[m][usize::from(new)] += 1;
                if new == 1 && cur == 0 {
                    // split: [cur_start, t) keeps `pre`, t opens a segment with `post`.
                    seg_counts[cur_start * k..cur_start * k + k].copy_from_slice(&pre);
                    seg_n[cur_start] = pre_n;
                    seg_counts[t * k..t * k + k].copy_from_slice(&post);
                    seg_n[t] = pn;
                } else if new == 0 && cur == 1 {
                    // merge t's segment into the previous one.
                    seg_counts[cur_start * k..cur_start * k + k].copy_from_slice(&merged);
                    seg_n[cur_start] = mn;
                    seg_counts[t * k..t * k + k].iter_mut().for_each(|x| *x = 0);
                    seg_n[t] = 0;
                }
                l[t] = new;
                boundary[t] = new == 1;
                if new == 1 {
                    cur_start = t;
                    pre.iter_mut().for_each(|x| *x = 0);
                    pre_n = 0;
                }
            } else if boundary[t] {
                // A segment start not resampled here: a compat phantom boundary, or
                // an initial boundary frozen during warm-up.
                cur_start = t;
                pre.iter_mut().for_each(|x| *x = 0);
                pre_n = 0;
            }

            // z's of turn t, in segment `cur_start`.
            let seg_off = cur_start * k;
            for (i, &w) in data.turns[t].iter().enumerate() {
                let w = w as usize;
                let old = z[t][i] as usize;
                nwk[w * k + old] -= 1;
                nk[old] -= 1;
                inv_den[old] = 1.0 / (nk[old] as f64 + beta_sum);
                turn_counts[t * k + old] -= 1;
                seg_counts[seg_off + old] -= 1;
                let row = &nwk[w * k..w * k + k];
                let seg = &seg_counts[seg_off..seg_off + k];
                let mut total = 0.0;
                for kk in 0..k {
                    total += (row[kk] as f64 + beta) * inv_den[kk] * (seg[kk] as f64 + alpha);
                    p[kk] = total;
                }
                let u = rng.gen::<f64>() * total;
                let mut new = k - 1;
                for (kk, &c) in p.iter().enumerate() {
                    if u < c {
                        new = kk;
                        break;
                    }
                }
                z[t][i] = new as u32;
                nwk[w * k + new] += 1;
                nk[new] += 1;
                inv_den[new] = 1.0 / (nk[new] as f64 + beta_sum);
                turn_counts[t * k + new] += 1;
                seg_counts[seg_off + new] += 1;
            }
            for kk in 0..k {
                pre[kk] += turn_counts[t * k + kk];
            }
            pre_n += data.turns[t].len() as u32;
        }

        #[cfg(test)]
        validate_state(
            &turn_counts,
            &seg_counts,
            &seg_n,
            &boundary,
            &l,
            &spk,
            &eligible,
            data,
            k,
        );

        let n_elig_shift = (0..nt).filter(|&t| eligible[t] && l[t] == 1).count() as u32;
        shift_trace.push(n_elig_shift);

        if it >= cfg.burn_in {
            let mut s_all = vec![0u32; nm];
            let mut s_el = vec![0u32; nm];
            for t in 0..nt {
                if l[t] == 1 {
                    let m = data.speakers[t] as usize;
                    shift_sum[t] += 1;
                    s_all[m] += 1;
                    if eligible[t] {
                        s_el[m] += 1;
                    }
                }
            }
            for mm in 0..nm {
                spk_sum[mm] += s_all[mm] as u64;
                spk_elig_sum[mm] += s_el[mm] as u64;
            }
            #[cfg(test)]
            if nt > 3 && !z[0].is_empty() && !z[3].is_empty() && z[0][0] == z[3][0] {
                same_topic += 1;
            }
            if num_draws.is_multiple_of(sample_interval) {
                spk_draws.push(s_all);
                spk_elig_draws.push(s_el);
            }
            num_draws += 1;
        }
        sweeps_run = it + 1;
        if sweeps_run.is_multiple_of(eval_stride) && sweeps_run < cfg.iters {
            let ll = log_joint(
                &spk,
                &nwk,
                &nk,
                &seg_counts,
                &seg_n,
                &boundary,
                data,
                cfg,
                dm_const,
                &lg_a,
                &lg_ka,
            );
            fit_history.push((sweeps_run, ll));
        }
        if !on_progress(it + 1, cfg.iters) {
            break;
        }
    }
    let ll = log_joint(
        &spk,
        &nwk,
        &nk,
        &seg_counts,
        &seg_n,
        &boundary,
        data,
        cfg,
        dm_const,
        &lg_a,
        &lg_ka,
    );
    fit_history.push((sweeps_run, ll));

    // --- materialize ---
    let topic_word: Vec<Vec<f64>> = (0..k)
        .map(|kk| {
            let d = nk[kk] as f64 + beta_sum;
            (0..v)
                .map(|w| (nwk[w * k + kk] as f64 + beta) / d)
                .collect()
        })
        .collect();
    let ka = k as f64 * alpha;
    let mut doc_topic = Vec::with_capacity(nt);
    let mut cur_start = 0usize;
    for t in 0..nt {
        if boundary[t] || data.conv_start[t] {
            cur_start = t;
        }
        let d = seg_n[cur_start] as f64 + ka;
        doc_topic.push(
            (0..k)
                .map(|kk| (seg_counts[cur_start * k + kk] as f64 + alpha) / d)
                .collect::<Vec<f64>>(),
        );
    }
    let turn_topic: Vec<Vec<f64>> = (0..nt)
        .map(|t| {
            let d = data.turns[t].len() as f64 + ka;
            (0..k)
                .map(|kk| (turn_counts[t * k + kk] as f64 + alpha) / d)
                .collect()
        })
        .collect();
    let nd = num_draws.max(1) as f64;
    let shift_prob: Vec<f64> = if num_draws == 0 {
        vec![f64::NAN; nt]
    } else {
        shift_sum.iter().map(|&s| s as f64 / nd).collect()
    };
    let mut speaker_turns = vec![0u32; nm];
    let mut speaker_eligible = vec![0u32; nm];
    for t in 0..nt {
        let m = data.speakers[t] as usize;
        speaker_turns[m] += 1;
        if eligible[t] {
            speaker_eligible[m] += 1;
        }
    }
    let mean_or_nan = |s: &Vec<u64>| -> Vec<f64> {
        if num_draws == 0 {
            vec![f64::NAN; nm]
        } else {
            s.iter().map(|&x| x as f64 / nd).collect()
        }
    };

    SitsModel {
        num_topics: k,
        topic_word,
        doc_topic,
        turn_topic,
        shift_prob,
        eligible,
        speaker_turns,
        speaker_eligible,
        speaker_shift_mean: mean_or_nan(&spk_sum),
        speaker_eligible_shift_mean: mean_or_nan(&spk_elig_sum),
        speaker_shift_draws: spk_draws,
        speaker_eligible_shift_draws: spk_elig_draws,
        shift_trace,
        num_draws,
        sampler_speaker_counts: spk,
        num_phantom,
        fit_history,
        sweeps_run,
        converged: false,
        debug_same_topic_freq: if cfg!(test) && num_draws > 0 {
            Some(same_topic as f64 / num_draws as f64)
        } else {
            None
        },
    }
}

/// Test-only: recompute segment totals and speaker counts from scratch and compare
/// with the incremental bookkeeping.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn validate_state(
    turn_counts: &[u32],
    seg_counts: &[u32],
    seg_n: &[u32],
    boundary: &[bool],
    l: &[u8],
    spk: &[[u32; 2]],
    eligible: &[bool],
    data: &SitsData<'_>,
    k: usize,
) {
    let nt = data.turns.len();
    let mut start = 0usize;
    let mut acc = vec![0u32; nt * k];
    let mut accn = vec![0u32; nt];
    let mut c = vec![[0u32; 2]; data.num_speakers];
    for t in 0..nt {
        if data.conv_start[t] || boundary[t] {
            start = t;
        }
        for kk in 0..k {
            acc[start * k + kk] += turn_counts[t * k + kk];
        }
        accn[start] += data.turns[t].len() as u32;
        let m = data.speakers[t] as usize;
        c[m][usize::from(boundary[t] || data.conv_start[t])] += 1;
        if eligible[t] {
            assert_eq!(boundary[t], l[t] == 1, "eligible turn {t}: boundary != l");
        } else if !data.conv_start[t] {
            assert_eq!(l[t], 0, "ineligible turn {t} recorded as shift");
        }
    }
    for t in 0..nt {
        if data.conv_start[t] || boundary[t] {
            assert_eq!(
                &seg_counts[t * k..t * k + k],
                &acc[t * k..t * k + k],
                "segment {t}"
            );
            assert_eq!(seg_n[t], accn[t], "segment size {t}");
        } else {
            assert_eq!(seg_n[t], 0, "stale segment total at non-boundary {t}");
        }
    }
    assert_eq!(spk, &c[..], "speaker counts");
}

/// Collapsed log joint of the current state, as the reference's
/// `getLogLikelihood`: speaker Beta-Bernoulli + topic-word DM + every segment's DM.
#[allow(clippy::too_many_arguments)]
fn log_joint(
    spk: &[[u32; 2]],
    nwk: &[u32],
    nk: &[u32],
    seg_counts: &[u32],
    seg_n: &[u32],
    boundary: &[bool],
    data: &SitsData<'_>,
    cfg: &SitsConfig,
    dm_const: f64,
    lg_a: &LgTable,
    lg_ka: &LgTable,
) -> f64 {
    let k = cfg.num_topics;
    let v = data.num_types;
    let g = cfg.gamma;
    let b = cfg.beta;
    let mut ll = 0.0;
    let g_const = log_gamma(2.0 * g) - 2.0 * log_gamma(g);
    for c in spk {
        ll += g_const + log_gamma(c[0] as f64 + g) + log_gamma(c[1] as f64 + g)
            - log_gamma((c[0] + c[1]) as f64 + 2.0 * g);
    }
    let b_const = log_gamma(v as f64 * b) - v as f64 * log_gamma(b);
    let lgb = log_gamma(b);
    for kk in 0..k {
        let mut s = b_const;
        for w in 0..v {
            let c = nwk[w * k + kk];
            s += if c == 0 { lgb } else { log_gamma(c as f64 + b) };
        }
        s -= log_gamma(nk[kk] as f64 + v as f64 * b);
        ll += s;
    }
    for t in 0..data.turns.len() {
        if boundary[t] || data.conv_start[t] {
            ll += dm_const + dm_var(&seg_counts[t * k..t * k + k], seg_n[t], lg_a, lg_ka);
        }
    }
    ll
}

/// Geweke z-score of a trace: mean of the first `first` fraction vs the last `last`
/// fraction, with batch-means variances (20 batches per window). Returns `None` when
/// a window is too short or constant.
pub fn geweke_z(trace: &[f64], first: f64, last: f64) -> Option<f64> {
    let n = trace.len();
    let na = ((n as f64) * first) as usize;
    let nb = ((n as f64) * last) as usize;
    if na < 20 || nb < 20 {
        return None;
    }
    let a = &trace[..na];
    let b = &trace[n - nb..];
    let bm = |x: &[f64]| -> (f64, f64) {
        let nbatch = 20usize;
        let size = x.len() / nbatch;
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let means: Vec<f64> = (0..nbatch)
            .map(|i| x[i * size..(i + 1) * size].iter().sum::<f64>() / size as f64)
            .collect();
        let bmean = means.iter().sum::<f64>() / nbatch as f64;
        let var = means.iter().map(|m| (m - bmean).powi(2)).sum::<f64>() / (nbatch as f64 - 1.0);
        // variance of the window mean
        (mean, var / nbatch as f64)
    };
    let (ma, va) = bm(a);
    let (mb, vb) = bm(b);
    let s = (va + vb).sqrt();
    if s <= 0.0 || !s.is_finite() {
        return if (ma - mb).abs() < 1e-12 {
            Some(0.0)
        } else {
            None
        };
    }
    Some((ma - mb) / s)
}

impl Estimator for SitsModel {
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
        // θ rows are segment mixtures shared by several turns, not one Dirichlet
        // posterior per document, so the per-document composition contract does not
        // apply (as for MGLDA).
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Two conversations over two disjoint word blocks, with speaker 0 always
    /// switching blocks and speaker 1 always continuing.
    fn planted(seed: u64) -> (Vec<Vec<u32>>, Vec<u32>, Vec<bool>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (mut turns, mut spk, mut start) = (Vec::new(), Vec::new(), Vec::new());
        for _c in 0..8 {
            let mut block = 0u32;
            for t in 0..30 {
                let speaker = if t % 3 == 0 { 0 } else { 1 };
                if t > 0 && speaker == 0 {
                    block = 1 - block;
                }
                let words: Vec<u32> = (0..8).map(|_| block * 10 + rng.gen_range(0..10)).collect();
                turns.push(words);
                spk.push(speaker);
                start.push(t == 0);
            }
        }
        (turns, spk, start)
    }

    fn cfg(iters: usize, compat: Compat) -> SitsConfig {
        SitsConfig {
            num_topics: 2,
            alpha: 0.1,
            beta: 0.1,
            gamma: 1.0,
            min_shift_tokens: 5,
            init_shift_rate: 0.1,
            compat,
            warmup: 0,
            iters,
            burn_in: iters / 2,
            sample_interval: 1,
        }
    }

    fn run(seed: u64, iters: usize, compat: Compat) -> SitsModel {
        let (turns, spk, start) = planted(1);
        let data = SitsData {
            turns: &turns,
            speakers: &spk,
            conv_start: &start,
            num_speakers: 2,
            num_types: 20,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        fit(&data, &cfg(iters, compat), |_, _| true, &mut rng)
    }

    #[test]
    fn recovers_planted_shifters() {
        let m = run(3, 400, Compat::None);
        let r0 = m.speaker_eligible_shift_mean[0] / m.speaker_eligible[0] as f64;
        let r1 = m.speaker_eligible_shift_mean[1] / m.speaker_eligible[1] as f64;
        assert!(r0 > 0.8, "speaker 0 eligible shift rate {r0}");
        assert!(r1 < 0.2, "speaker 1 eligible shift rate {r1}");
        // topics own the two blocks
        let own0: f64 = m.topic_word[0][..10].iter().sum();
        let own1: f64 = m.topic_word[1][..10].iter().sum();
        assert!((own0 > 0.9 && own1 < 0.1) || (own0 < 0.1 && own1 > 0.9));
    }

    #[test]
    fn deterministic() {
        let a = run(7, 60, Compat::None);
        let b = run(7, 60, Compat::None);
        assert_eq!(a.topic_word, b.topic_word);
        assert_eq!(a.shift_prob, b.shift_prob);
        assert_eq!(a.shift_trace, b.shift_trace);
        let c = run(8, 60, Compat::None);
        assert_ne!(a.shift_trace, c.shift_trace);
    }

    /// The incremental segment bookkeeping must equal a from-scratch recount.
    #[test]
    fn segment_counts_consistent() {
        for compat in [Compat::None, Compat::Rossiter2022 { init_every: 3 }] {
            let m = run(11, 30, compat);
            for row in &m.doc_topic {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
            let tot: u32 = m.sampler_speaker_counts.iter().map(|c| c[0] + c[1]).sum();
            assert_eq!(tot as usize, m.shift_prob.len());
        }
    }

    /// Exact check of the stationary distribution: enumerate every (l, z) state of
    /// a small corpus (two conversations, one short ineligible turn), weight it by
    /// the collapsed joint (written independently here, counting forced turns in the
    /// speaker term as the reference does), and compare posterior shift marginals and
    /// a topic co-assignment probability with long-run sampler frequencies.
    #[test]
    fn matches_exact_enumeration() {
        let turns: Vec<Vec<u32>> = vec![vec![0, 0], vec![1, 1], vec![0, 1], vec![1, 0], vec![0]];
        let spk: Vec<u32> = vec![0, 1, 0, 1, 0];
        let start = vec![true, false, false, true, false];
        let min_tokens = 2; // turn 4 (one token) is ineligible
        let (k, v, a, b, g) = (2usize, 2usize, 0.5f64, 0.5f64, 1.0f64);
        let lg = log_gamma;
        let dm = |c: &[f64], conc: f64| -> f64 {
            let n: f64 = c.iter().sum();
            lg(conc * c.len() as f64) - c.len() as f64 * lg(conc)
                + c.iter().map(|&x| lg(x + conc)).sum::<f64>()
                - lg(n + conc * c.len() as f64)
        };
        let free = [1usize, 2]; // eligible non-first turns
        let ntok: usize = turns.iter().map(|t| t.len()).sum();
        let (mut zsum, mut pl1, mut pl2, mut same) = (0.0, 0.0, 0.0, 0.0);
        for lbits in 0..4u32 {
            let mut l = [1u8, 0, 0, 1, 0];
            l[free[0]] = (lbits & 1) as u8;
            l[free[1]] = ((lbits >> 1) & 1) as u8;
            for zbits in 0..(1u32 << ntok) {
                let zs: Vec<usize> = (0..ntok).map(|i| ((zbits >> i) & 1) as usize).collect();
                let mut lj = 0.0;
                let mut c = [[0.0f64; 2]; 2];
                for t in 0..turns.len() {
                    c[spk[t] as usize][l[t] as usize] += 1.0;
                }
                for cm in &c {
                    lj += dm(cm, g);
                }
                let mut nw = vec![vec![0.0f64; v]; k];
                let mut seg = vec![0.0f64; k];
                let mut i = 0;
                for t in 0..turns.len() {
                    if l[t] == 1 && t > 0 {
                        lj += dm(&seg, a);
                        seg = vec![0.0; k];
                    }
                    for &w in &turns[t] {
                        nw[zs[i]][w as usize] += 1.0;
                        seg[zs[i]] += 1.0;
                        i += 1;
                    }
                }
                lj += dm(&seg, a);
                for row in &nw {
                    lj += dm(row, b);
                }
                let wgt = lj.exp();
                zsum += wgt;
                pl1 += wgt * l[1] as f64;
                pl2 += wgt * l[2] as f64;
                // tokens 0 (turn 0) and 6 (turn 3) share a topic
                if zs[0] == zs[6] {
                    same += wgt;
                }
            }
        }
        let (exact_l1, exact_l2) = (pl1 / zsum, pl2 / zsum);
        let exact_same = same / zsum;

        let data = SitsData {
            turns: &turns,
            speakers: &spk,
            conv_start: &start,
            num_speakers: 2,
            num_types: v,
        };
        let c = SitsConfig {
            num_topics: k,
            alpha: a,
            beta: b,
            gamma: g,
            min_shift_tokens: min_tokens,
            init_shift_rate: 0.5,
            compat: Compat::None,
            warmup: 0,
            iters: 200_000,
            burn_in: 1_000,
            sample_interval: 1000,
        };
        // co-assignment frequency needs the z's; re-run the chain and count directly
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let m = fit(&data, &c, |_, _| true, &mut rng);
        for (got, want, name) in [
            (m.shift_prob[1], exact_l1, "P(l1)"),
            (m.shift_prob[2], exact_l2, "P(l2)"),
        ] {
            assert!((got - want).abs() < 0.01, "{name}={got} exact {want}");
        }
        assert_eq!(m.shift_prob[0], 1.0);
        assert_eq!(m.shift_prob[3], 1.0);
        assert_eq!(m.shift_prob[4], 0.0);
        let got_same = m
            .debug_same_topic_freq
            .expect("test build records co-assignment");
        assert!(
            (got_same - exact_same).abs() < 0.01,
            "P(z0 == z6)={got_same} exact {exact_same}"
        );
    }

    #[test]
    fn warmup_freezes_shifts_then_releases_them() {
        let (turns, spk, start) = planted(1);
        let data = SitsData {
            turns: &turns,
            speakers: &spk,
            conv_start: &start,
            num_speakers: 2,
            num_types: 20,
        };
        let mut c = cfg(200, Compat::None);
        c.init_shift_rate = 1.0;
        c.warmup = 50;
        c.burn_in = 100;
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let m = fit(&data, &c, |_, _| true, &mut rng);
        let n_elig = m.eligible.iter().filter(|&&e| e).count() as u32;
        assert!(m.shift_trace[..50].iter().all(|&x| x == n_elig));
        // speaker 1 never shifts in the planted data: released shifts fall away
        let r1 = m.speaker_eligible_shift_mean[1] / m.speaker_eligible[1] as f64;
        assert!(r1 < 0.2, "follower rate after warm-up {r1}");
    }

    #[test]
    fn geweke_flags_a_trend_and_passes_noise() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let noise: Vec<f64> = (0..4000).map(|_| rng.gen::<f64>()).collect();
        let z = geweke_z(&noise, 0.1, 0.5).unwrap();
        assert!(z.abs() < 3.0, "white noise z = {z}");
        let trend: Vec<f64> = (0..4000)
            .map(|i| i as f64 / 400.0 + rng.gen::<f64>())
            .collect();
        let z = geweke_z(&trend, 0.1, 0.5).unwrap();
        assert!(z < -3.0, "rising trend z = {z}");
        assert!(geweke_z(&noise[..100], 0.1, 0.5).is_none());
    }

    #[test]
    fn draw_shift_is_overflow_safe() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert_eq!(draw_shift(0.0, 1000.0, &mut rng), 1);
        assert_eq!(draw_shift(1000.0, 0.0, &mut rng), 0);
    }

    #[test]
    fn sits_conforms() {
        let m = run(1, 20, Compat::None);
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }
}
