//! ReplyTM: a reply-conditioned topic model for threaded discussions (posts +
//! nested reply trees, one comment per document). A child comment's topic prior
//! is shifted by a learned, directed **response matrix** applied to its parent's
//! topic proportions, plus a per-group (covariate) baseline:
//!
//! ```text
//! root  a_d = exp(b_g)
//! child a_d = exp(b_g) + rho_g * T_g^T zbar_parent(d)
//! theta_d ~ Dir(a_d);  z_dn ~ theta_d;  w_dn ~ phi_{z_dn}
//! ```
//!
//! `T_g` (K×K, rows on the simplex) is the reported estimand: `T_g[i, j]` is the
//! response mass a topic-`i` parent places on child topic `j`, per covariate group
//! `g`. `rho_g > 0` is the response strength (separated from the transition shape),
//! and `exp(b_g)` the baseline concentration (an exp link, as in `dmr.rs`, so the
//! concentration is always positive). `phi` is collapsed (Dirichlet-multinomial,
//! prior `beta`), as in `csatm.rs`.
//!
//! Inference is collapsed Gibbs on the token assignments `z`, with `T_g`, `rho_g`,
//! and `b_g` sampled by Metropolis-within-Gibbs each sweep (so the reported `T`
//! carries posterior credible intervals). The token full-conditional includes the
//! **children Dirichlet-multinomial factor** — because a child's prior depends on
//! its parent's `zbar`, resampling a parent token changes every child's likelihood.
//! Dropping that factor targets the wrong posterior.
//!
//! Two regimes of the children factor: for a **short** parent (large per-token step
//! `rho/len_parent`) the exact log-DM is evaluated, and the enumeration test in
//! `reply_tm_enumeration_gate` proves that path targets the exact posterior and
//! fails a sampler that omits the factor. For a **long** parent (`step <= FAST_STEP`)
//! a first-order (digamma-gradient) approximation of the children factor is used,
//! with `O(step^2)` bias bounded by `FAST_STEP`; `reply_tm_fast_path_matches_exact`
//! bounds that bias against the enumerated posterior. Multithreading (`num_threads`)
//! adds the usual AD-LDA topic-word staleness on top.
//!
//! ReplyTM is a topica-original model (no reference implementation); it is gated
//! behind `enable_experimental()` and validated by planted recovery + the tiny-tree
//! enumeration gates. With `parents` all roots (or `rho=0`) and one group it reduces
//! to LDA(alpha, beta). See .github/CONTRIBUTING-MODELS.md section B1.

use crate::corpus::Corpus;
use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// How the response matrix varies with the covariate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovResponse {
    /// One `T_g` (and `rho_g`, `b_g`) per covariate group.
    PerGroup,
    /// Shared `T` shape; per-group strength `rho_g` and baseline `b_g`.
    SharedShape,
    /// Single group: ignore the covariate entirely.
    Global,
}

/// Hyperparameters for [`fit`].
#[derive(Clone, Debug)]
pub struct ReplyTmParams {
    pub num_topics: usize,
    /// Symmetric prior mean for the baseline concentration (`exp(b_g)` inits at
    /// `alpha`) and the Dirichlet floor for root docs.
    pub alpha: f64,
    /// Topic-word Dirichlet prior.
    pub beta: f64,
    pub covariate_response: CovResponse,
    /// Weakly-informative prior on log `rho` (mean, sd) — the shrinkage that keeps
    /// the response strength from inflating on very short comments.
    pub rho_prior: (f64, f64),
    /// Shrinkage sd on the `T` row logits (toward a flat simplex).
    pub t_prior_sd: f64,
    /// Metropolis sub-iterations for the parameter block each sweep.
    pub mh_steps: usize,
    /// RWM proposal sd for the parameter logits.
    pub mh_step_sd: f64,
    /// Sweeps discarded before collecting `T`/`theta` posterior draws.
    pub burn: usize,
    /// AD-LDA worker count. `1` (default) runs the exact serial sweep, bit-identical
    /// to the un-threaded loop. `>1` partitions whole conversation threads across
    /// workers (so parent↔child stays race-free), writes back each worker's own
    /// docs' assignments, and recomputes the global topic-word table exactly from
    /// the merged `z` each sweep; deterministic for a fixed `num_threads` + seed.
    pub num_threads: usize,
    /// Number of independent MCMC chains run from dispersed seeds. Their draws are
    /// topic-aligned (Hungarian on φ) and pooled, so the reported credible intervals
    /// include between-chain variance and `converged`/`max_rhat` come from split-R̂.
    /// A single chain under-represents posterior spread; `>1` (default 4) is required
    /// for honest intervals. `1` is a fast debug mode with no R̂.
    pub num_chains: usize,
}

impl Default for ReplyTmParams {
    fn default() -> Self {
        ReplyTmParams {
            num_topics: 10,
            alpha: 0.5,
            beta: 0.01,
            covariate_response: CovResponse::PerGroup,
            rho_prior: (1.4, 0.6), // log-mean ~ exp(1.4) ≈ 4 response strength; the
            // tighter sd shrinks the short-comment rho inflation (the L≈15 artifact)
            t_prior_sd: 1.5,
            mh_steps: 5,
            mh_step_sd: 0.3,
            burn: 200,
            num_threads: 1,
            num_chains: 4,
        }
    }
}

/// Assign whole conversation threads (connected components of the parent forest) to
/// `nt` workers, balancing by token count. Keeping a conversation intact on one
/// worker means every parent↔child interaction (the children DM factor) reads only
/// that worker's documents, so `ndk` is race-free without locking. Deterministic.
fn partition_conversations(
    parent_of: &[Option<usize>],
    children: &[Vec<usize>],
    lengths: &[usize],
    nt: usize,
) -> Vec<Vec<usize>> {
    let d = parent_of.len();
    // Roots of each conversation (nodes with no modeled parent).
    let mut assigned = vec![false; d];
    // Collect components in a deterministic order (by root index), each via DFS.
    let mut comps: Vec<(usize, Vec<usize>)> = Vec::new(); // (token weight, docs)
    for root in 0..d {
        if parent_of[root].is_some() || assigned[root] {
            continue;
        }
        let mut docs = Vec::new();
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            if assigned[n] {
                continue;
            }
            assigned[n] = true;
            docs.push(n);
            for &c in &children[n] {
                if !assigned[c] {
                    stack.push(c);
                }
            }
        }
        let wt: usize = docs.iter().map(|&x| lengths[x]).sum();
        comps.push((wt, docs));
    }
    // Any nodes not reached from a root (e.g. a stray cycle) form singleton comps.
    for n in 0..d {
        if !assigned[n] {
            assigned[n] = true;
            comps.push((lengths[n], vec![n]));
        }
    }
    // Greedy longest-processing-time: largest conversations to the lightest worker.
    comps.sort_by_key(|c| std::cmp::Reverse(c.0));
    let mut loads = vec![0usize; nt];
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); nt];
    for (wt, docs) in comps {
        let w = (0..nt).min_by_key(|&i| loads[i]).unwrap();
        loads[w] += wt;
        buckets[w].extend(docs);
    }
    buckets
}

/// Fitted state for [`fit`].
pub struct ReplyTmModel {
    pub num_topics: usize,
    pub num_groups: usize,
    /// phi (K×V), rows sum to 1.
    pub topic_word: Vec<Vec<f64>>,
    /// theta (D×K) posterior mean over retained sweeps, rows sum to 1.
    pub doc_topic: Vec<Vec<f64>>,
    /// T_g (G×K×K) posterior mean, each row on the simplex.
    pub response_matrix: Vec<Vec<Vec<f64>>>,
    /// 2.5% / 97.5% per-cell empirical-quantile credible bounds of T_g (each G×K×K),
    /// from pooled topic-aligned draws across all chains (so they carry between-chain
    /// variance).
    pub response_matrix_lo: Vec<Vec<Vec<f64>>>,
    pub response_matrix_hi: Vec<Vec<Vec<f64>>>,
    /// rho_g (G) posterior mean response strength, with 2.5% / 97.5% quantile bounds.
    pub response_strength: Vec<f64>,
    pub response_strength_lo: Vec<f64>,
    pub response_strength_hi: Vec<f64>,
    /// exp(b_g) (G×K) baseline concentration posterior mean, with quantile bounds.
    pub baseline: Vec<Vec<f64>>,
    pub baseline_lo: Vec<Vec<f64>>,
    pub baseline_hi: Vec<Vec<f64>>,
    /// Representative document-topic Dirichlet (mean over docs of a_d) for
    /// method-of-composition standard errors.
    pub alpha_mean: Vec<f64>,
    /// Parent-topic support (G×K): for each group g and topic i, the total parent
    /// z̄ mass on topic i summed over that group's child edges. Row i of T_g is only
    /// identified where this is non-trivial; `response_contrast`/`response_table` use
    /// it to suppress cells estimated on near-empty support (an unequal-prevalence
    /// contrast otherwise reads as a spurious response difference).
    pub parent_support: Vec<Vec<f64>>,
    pub doc_lengths: Vec<usize>,
    pub fit_history: Vec<(usize, f64)>,
    /// Max split-R̂ over all sampled scalars (T cells, rho, baseline) across chains.
    /// `NaN` when `num_chains == 1` (no R̂ available).
    pub max_rhat: f64,
    /// `max_rhat < 1.1` (multi-chain). Convergence of the *parameters*, not just the
    /// log-likelihood. `false` with a single chain (no R̂).
    pub converged: bool,
}

/// log Γ via the Lanczos approximation (same coefficients as `prodlda.rs`).
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
    // For 0 < x < 0.5 use the recurrence lgamma(x) = lgamma(x+1) - ln(x) (one extra
    // ln) rather than the sin-reflection: the hot path here evaluates concentrations
    // near the baseline (~0.3 < 0.5) constantly, and the reflection's sin + recursion
    // is far costlier. `a_c` is always > 0, so a single shift lands in [1, 1.5).
    if x < 0.5 {
        return lgamma(x + 1.0) - x.ln();
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Digamma ψ(x) for x > 0 (recurrence up to the asymptotic regime, then the
/// standard series). Used for the first-order children-factor fast path.
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

/// log Dirichlet-multinomial of topic counts `n` (length K) under concentration
/// `a` (length K). Supports fractional counts.
fn log_dm(n: &[f64], a: &[f64]) -> f64 {
    let asum: f64 = a.iter().sum();
    let nsum: f64 = n.iter().sum();
    let mut s = lgamma(asum) - lgamma(asum + nsum);
    for k in 0..a.len() {
        s += lgamma(n[k] + a[k]) - lgamma(a[k]);
    }
    s
}

/// `T_g^T zbar` — the response contribution to a child's concentration.
#[inline]
fn t_transpose_dot(t: &[Vec<f64>], zbar: &[f64], k: usize) -> Vec<f64> {
    // (T^T zbar)_j = sum_i zbar_i * T[i][j]
    let mut out = vec![0.0f64; k];
    for i in 0..k {
        let zi = zbar[i];
        if zi == 0.0 {
            continue;
        }
        let row = &t[i];
        for j in 0..k {
            out[j] += zi * row[j];
        }
    }
    out
}

/// Child concentration a_c = exp(b_g) + rho_g * T_g^T zbar_parent.
#[inline]
fn child_concentration(
    base: &[f64],
    rho: f64,
    t: &[Vec<f64>],
    zbar_parent: &[f64],
    k: usize,
) -> Vec<f64> {
    let resp = t_transpose_dot(t, zbar_parent, k);
    (0..k).map(|j| base[j] + rho * resp[j]).collect()
}

/// Unnormalized log-weights over the K topics for resampling one token of doc
/// `dd` (word `w`), filling `out`. `ndk` must already have this token removed.
/// The `include_children` branch adds `Σ_{c∈children(dd)} log DM(n_c | a_c(z̄_dd))`
/// — the reply-coupling factor. Omitting it (a naive child-only sweep) targets the
/// wrong posterior; the enumeration-gate test toggles this flag to prove the gate
/// catches that. Shared by `fit`'s inner loop and the test so the gate exercises
/// the shipped conditional.
#[allow(clippy::too_many_arguments)]
fn token_logweights(
    out: &mut [f64],
    w: usize,
    k: usize,
    ndk: &[Vec<f64>],
    nkw: &[Vec<f64>],
    nk: &[f64],
    beta: f64,
    vbeta: f64,
    dd: usize,
    a_d: &[f64],
    len_dd: f64,
    children_dd: &[usize],
    group_of: &[usize],
    base_now: &[Vec<f64>],
    rho_now: &[f64],
    t_now: &[Vec<Vec<f64>>],
    include_children: bool,
) {
    let has_children = include_children && !children_dd.is_empty();
    let inv = 1.0 / len_dd;
    // phi predictive * own-doc factor (cheap, per candidate).
    for t in 0..k {
        let phi_pred = (nkw[t][w] + beta) / (nk[t] + vbeta);
        let own = ndk[dd][t] + a_d[t];
        out[t] = phi_pred.ln() + own.max(1e-300).ln();
    }
    if !has_children {
        return;
    }
    // Children DM factor. For each child, precompute its concentration WITHOUT this
    // token (`a_c_base` from the decremented z̄_dd); assigning the token to candidate
    // `t` shifts z̄_dd by `inv` at coordinate `t`, so a_c(t)[j] = a_c_base[j] +
    // rho·inv·T_gc[t][j]. Each of the K candidates is then O(K) per child, instead of
    // recomputing the O(K^2) matvec per candidate (an O(K) speedup on this term).
    //
    // The first-order error is O(step^2) per child and ACCUMULATES across a parent's
    // children, so a high-fan-out parent (a popular comment with many replies) can
    // drift meaningfully even at a small per-token step. Gate the fast path on
    // `fan_out * step^2` so busy parents fall back to the exact children factor; such
    // parents are rare (fan-out is heavy-tailed), so the speed cost is small.
    let fan_out = children_dd.len() as f64;
    for &c in children_dd {
        let gc = group_of[c];
        let tg = &t_now[gc];
        let rho = rho_now[gc];
        let mut a_c_base = base_now[gc].clone();
        for i in 0..k {
            let zi = ndk[dd][i] / len_dd; // token removed
            if zi == 0.0 {
                continue;
            }
            let row = &tg[i];
            let s = rho * zi;
            for j in 0..k {
                a_c_base[j] += s * row[j];
            }
        }
        // Only the Σ_j [lgamma(n_j + a_j) - lgamma(a_j)] part of each child's DM
        // depends on the candidate `t`. Because T rows sum to 1, a_c(t) has the SAME
        // total Σ_j a_c(t)[j] = Σ a_c_base + step for every t, so the DM's normalizer
        // terms lgamma(Σa) - lgamma(Σa + n) are t-constant and cancel in the token
        // softmax across candidates.
        let step = rho * inv;
        let n_c = &ndk[c];
        if step <= FAST_STEP && fan_out * step * step <= FAST_FANOUT_BUDGET {
            // Small perturbation (long parent, so one token barely moves its topic
            // mix): the child's log-DM as a function of the candidate `t` is well
            // approximated by its first-order term. a_c(t) = a_c_base + step·T[t], so
            //   logDM(t) ≈ const + Σ_j step·T[t][j]·g[j],  g[j] = ψ(n_j+a_base_j) - ψ(a_base_j).
            // g is computed once per child (O(K) digamma); each candidate is then a
            // dot product T[t]·g (no per-candidate transcendentals). The exact path
            // below runs when the perturbation is large (short parents, and the exact
            // enumeration gate). Error is O(step²), bounded by FAST_STEP.
            let mut g = vec![0.0f64; k];
            for j in 0..k {
                g[j] = digamma(n_c[j] + a_c_base[j]) - digamma(a_c_base[j]);
            }
            for t in 0..k {
                let row_t = &tg[t];
                let mut dot = 0.0;
                for j in 0..k {
                    dot += row_t[j] * g[j];
                }
                out[t] += step * dot;
            }
        } else {
            for t in 0..k {
                let row_t = &tg[t];
                let mut s = 0.0;
                for j in 0..k {
                    let a = a_c_base[j] + step * row_t[j];
                    s += lgamma(n_c[j] + a) - lgamma(a);
                }
                out[t] += s;
            }
        }
    }
}

/// Perturbation threshold below which the children factor uses the first-order
/// (digamma) approximation. `step = rho / len_parent`; for realistic comment
/// lengths and shrunk `rho` this is well under the threshold, while the exact
/// enumeration gate (tiny docs, large step) stays on the exact path.
const FAST_STEP: f64 = 0.2;

/// Fan-out budget for the first-order children-factor fast path: it is used only
/// when `children_count * step^2 <= FAST_FANOUT_BUDGET`, so the accumulated O(step^2)
/// bias across a parent's children stays bounded (a high-fan-out parent uses the
/// exact factor). Calibrated by the `reply_tm_fast_path_high_fanout` enumeration test.
const FAST_FANOUT_BUDGET: f64 = 0.1;

/// Resample every token of document `dd` (one collapsed-Gibbs pass over the doc),
/// mutating `z`/`ndk`/`nkw`/`nk` in place. Shared by the serial sweep and the
/// AD-LDA workers so both run the identical conditional; a worker passes private
/// `nkw`/`nk` tables and a full-size `ndk` clone (so parent/child indexing works).
#[allow(clippy::too_many_arguments)]
fn resample_doc<R: Rng>(
    dd: usize,
    k: usize,
    docs: &[Vec<u32>],
    z: &mut [Vec<usize>],
    ndk: &mut [Vec<f64>],
    nkw: &mut [Vec<f64>],
    nk: &mut [f64],
    beta: f64,
    vbeta: f64,
    parent_of: &[Option<usize>],
    len_d: &[f64],
    group_of: &[usize],
    base_now: &[Vec<f64>],
    rho_now: &[f64],
    t_now: &[Vec<Vec<f64>>],
    children: &[Vec<usize>],
    cond_log: &mut [f64],
    rng: &mut R,
) {
    let g = group_of[dd];
    // d's OWN prior (from its parent) is fixed across this doc's tokens.
    let a_d: Vec<f64> = match parent_of[dd] {
        None => base_now[g].clone(),
        Some(p) => {
            let zbar_p: Vec<f64> = (0..k).map(|t| ndk[p][t] / len_d[p]).collect();
            child_concentration(&base_now[g], rho_now[g], &t_now[g], &zbar_p, k)
        }
    };
    let has_children = !children[dd].is_empty();
    for pos in 0..docs[dd].len() {
        let w = docs[dd][pos] as usize;
        let old = z[dd][pos];
        ndk[dd][old] -= 1.0;
        nkw[old][w] -= 1.0;
        nk[old] -= 1.0;
        token_logweights(
            cond_log,
            w,
            k,
            ndk,
            nkw,
            nk,
            beta,
            vbeta,
            dd,
            &a_d,
            len_d[dd],
            &children[dd],
            group_of,
            base_now,
            rho_now,
            t_now,
            has_children,
        );
        let maxlw = cond_log.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0f64;
        for t in 0..k {
            let e = (cond_log[t] - maxlw).exp();
            cond_log[t] = e;
            total += e;
        }
        let new = if total.is_finite() && total > 0.0 {
            let mut r = rng.gen::<f64>() * total;
            let mut pick = k - 1;
            for t in 0..k {
                r -= cond_log[t];
                if r <= 0.0 {
                    pick = t;
                    break;
                }
            }
            pick
        } else {
            (rng.gen::<f64>() * k as f64) as usize % k
        };
        z[dd][pos] = new;
        ndk[dd][new] += 1.0;
        nkw[new][w] += 1.0;
        nk[new] += 1.0;
    }
}

/// Row-softmax: logits (K×K) -> row-simplex T (K×K).
fn softmax_rows(logits: &[Vec<f64>], _k: usize) -> Vec<Vec<f64>> {
    logits
        .iter()
        .map(|row| {
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = row.iter().map(|&x| (x - m).exp()).collect();
            let s: f64 = ex.iter().sum();
            ex.iter().map(|&e| e / s).collect()
        })
        .collect()
}

/// Per-group parameter state carried through sampling.
struct GroupParams {
    b: Vec<f64>,             // baseline log-concentration (length K)
    log_rho: f64,            // response strength (log scale)
    t_logits: Vec<Vec<f64>>, // K×K response-matrix logits (row-softmax -> T)
}

/// Output of a single MCMC chain: the raw per-sweep draws plus the chain's own
/// posterior-mean topic-word and doc-topic tables (used to topic-align chains before
/// pooling). Means / intervals / R̂ are assembled by [`fit`] across chains.
struct ChainOut {
    num_groups: usize,
    topic_word: Vec<Vec<f64>>,
    doc_topic: Vec<Vec<f64>>,
    t_draws: Vec<Vec<Vec<Vec<f64>>>>, // [g][draw][i][j]
    rho_draws: Vec<Vec<f64>>,         // [g][draw]
    base_draws: Vec<Vec<Vec<f64>>>,   // [g][draw][j]
    fit_history: Vec<(usize, f64)>,
}

/// Run one MCMC chain. `parents[d]` is document `d`'s parent index (negative = thread
/// root); `covariate[d]` is `d`'s group index (0-based, contiguous). Empty `parents`
/// => all roots; empty `covariate` => a single group.
fn fit_one_chain<R: Rng>(
    corpus: &Corpus,
    parents: &[i64],
    covariate: &[i64],
    params: &ReplyTmParams,
    iters: usize,
    rng: &mut R,
) -> ChainOut {
    let k = params.num_topics.max(1);
    let d = corpus.num_docs();
    let v = corpus.num_types();
    let docs = &corpus.docs;
    let beta = params.beta;
    let vbeta = v as f64 * beta;
    // Never let burn-in swallow the whole run: always retain the second half of
    // the sweeps as posterior draws (so short fits still populate T/rho/theta).
    let burn = params.burn.min(iters / 2);

    // Parent forest (fall back to all-roots).
    let all_roots: Vec<i64>;
    let parents: &[i64] = if parents.is_empty() {
        all_roots = vec![-1; d];
        &all_roots
    } else {
        parents
    };
    let parent_of = |dd: usize| -> Option<usize> {
        let p = parents[dd];
        if p >= 0 && (p as usize) < d && (p as usize) != dd {
            Some(p as usize)
        } else {
            None
        }
    };

    // Groups.
    let zero_cov: Vec<i64>;
    let covariate: &[i64] =
        if covariate.is_empty() || params.covariate_response == CovResponse::Global {
            zero_cov = vec![0; d];
            &zero_cov
        } else {
            covariate
        };
    let num_groups = (covariate.iter().cloned().max().unwrap_or(0).max(0) as usize) + 1;
    let group_of: Vec<usize> = (0..d).map(|dd| covariate[dd].max(0) as usize).collect();

    // Children adjacency.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); d];
    for dd in 0..d {
        if let Some(p) = parent_of(dd) {
            children[p].push(dd);
        }
    }
    let parent_slice: Vec<Option<usize>> = (0..d).map(parent_of).collect();

    // AD-LDA worker partition (whole conversations per worker). Empty / nt<=1 => serial.
    let nt = params.num_threads.max(1).min(d.max(1));
    let worker_docs: Vec<Vec<usize>> = if nt > 1 {
        let dl: Vec<usize> = docs.iter().map(|dc| dc.len()).collect();
        partition_conversations(&parent_slice, &children, &dl, nt)
    } else {
        Vec::new()
    };

    // Count tables.
    let mut ndk = vec![vec![0.0f64; k]; d]; // doc-topic
    let mut nkw = vec![vec![0.0f64; v]; k]; // topic-word
    let mut nk = vec![0.0f64; k]; // topic totals
    let mut z: Vec<Vec<usize>> = docs.iter().map(|doc| vec![0usize; doc.len()]).collect();
    let len_d: Vec<f64> = docs.iter().map(|doc| doc.len().max(1) as f64).collect();

    for dd in 0..d {
        for pos in 0..docs[dd].len() {
            let w = docs[dd][pos] as usize;
            let topic = (rng.gen::<f64>() * k as f64) as usize % k;
            z[dd][pos] = topic;
            ndk[dd][topic] += 1.0;
            nkw[topic][w] += 1.0;
            nk[topic] += 1.0;
        }
    }

    // Parameter state per group. Baseline inits at log(alpha), rho at the prior
    // mean, T logits at 0 (flat simplex).
    let mut gp: Vec<GroupParams> = (0..num_groups)
        .map(|_| GroupParams {
            b: vec![params.alpha.ln(); k],
            log_rho: params.rho_prior.0,
            t_logits: vec![vec![0.0f64; k]; k],
        })
        .collect();

    // theta accumulates its posterior mean in-chain; T / rho / baseline retain raw
    // draws (aligned + pooled across chains by the orchestrator).
    let mut theta_acc = vec![vec![0.0f64; k]; d];
    let mut t_draws: Vec<Vec<Vec<Vec<f64>>>> = vec![Vec::new(); num_groups];
    let mut rho_draws: Vec<Vec<f64>> = vec![Vec::new(); num_groups];
    let mut base_draws: Vec<Vec<Vec<f64>>> = vec![Vec::new(); num_groups];
    // Held-in log-likelihood trace (a convergence signal): recorded every
    // `ll_every` sweeps as the collapsed topic-word predictive log-likelihood.
    let ll_every = (iters / 40).max(1);
    let mut fit_history: Vec<(usize, f64)> = Vec::new();

    let mut cond_log = vec![0.0f64; k];
    let mut mh_step_sd = params.mh_step_sd; // adapted during burn-in toward ~0.25 accept

    for sweep in 0..iters {
        // Precompute current T (row-softmax) and baseline (exp) per group.
        let t_now: Vec<Vec<Vec<f64>>> = gp.iter().map(|g| softmax_rows(&g.t_logits, k)).collect();
        let base_now: Vec<Vec<f64>> = gp
            .iter()
            .map(|g| g.b.iter().map(|&x| x.exp()).collect())
            .collect();
        let rho_now: Vec<f64> = gp.iter().map(|g| g.log_rho.exp()).collect();

        // ---- (1) token sweep: collapsed Gibbs with the children DM factor ----
        if nt <= 1 {
            // Exact serial path (bit-identical to the un-threaded loop).
            for dd in 0..d {
                resample_doc(
                    dd,
                    k,
                    docs,
                    &mut z,
                    &mut ndk,
                    &mut nkw,
                    &mut nk,
                    beta,
                    vbeta,
                    &parent_slice,
                    &len_d,
                    &group_of,
                    &base_now,
                    &rho_now,
                    &t_now,
                    &children,
                    &mut cond_log,
                    rng,
                );
            }
        } else {
            // AD-LDA: each worker sweeps its whole-conversation doc set against a
            // full ndk clone (so parent/child indexing is exact and race-free) and a
            // private nkw/nk copy; then each worker's own docs' z/ndk are written back
            // and the global topic-word table is recomputed EXACTLY from the merged z
            // (no delta merge — that degraded recovery at high worker counts).
            // Deterministic for a fixed num_threads + seed via per-(worker,sweep) RNG.
            let nkw_orig = nkw.clone();
            let ndk_snap = &ndk;
            let z_snap = &z;
            let sweep_seed: u64 = rng.gen();
            let results: Vec<(usize, Vec<Vec<usize>>, Vec<Vec<f64>>)> = worker_docs
                .par_iter()
                .enumerate()
                .map(|(wid, wdocs)| {
                    let mut ndk_w = ndk_snap.clone();
                    let mut z_w = z_snap.clone();
                    // Private nkw/nk copies so resample_doc's decrement/increment sees a
                    // consistent local count; discarded after the sweep (the global nkw is
                    // recomputed exactly from the merged z below).
                    let mut nkw_w = nkw_orig.clone();
                    let mut nk_w = nk.clone();
                    let mut cond = vec![0.0f64; k];
                    let mut rng_w =
                        ChaCha8Rng::seed_from_u64(sweep_seed.wrapping_add(wid as u64 + 1));
                    for &dd in wdocs {
                        resample_doc(
                            dd,
                            k,
                            docs,
                            &mut z_w,
                            &mut ndk_w,
                            &mut nkw_w,
                            &mut nk_w,
                            beta,
                            vbeta,
                            &parent_slice,
                            &len_d,
                            &group_of,
                            &base_now,
                            &rho_now,
                            &t_now,
                            &children,
                            &mut cond,
                            &mut rng_w,
                        );
                    }
                    // Return only this worker's own (disjoint) docs' updated rows; the
                    // global nkw is recomputed exactly from the merged z after write-back.
                    let own_ndk: Vec<Vec<f64>> =
                        wdocs.iter().map(|&dd| ndk_w[dd].clone()).collect();
                    let own_z: Vec<Vec<usize>> = wdocs.iter().map(|&dd| z_w[dd].clone()).collect();
                    (wid, own_z, own_ndk)
                })
                .collect();
            // Write back each worker's own docs' z / ndk (disjoint doc sets).
            for (wid, own_z, own_ndk) in &results {
                for (slot, &dd) in worker_docs[*wid].iter().enumerate() {
                    z[dd].clone_from(&own_z[slot]);
                    ndk[dd].clone_from(&own_ndk[slot]);
                }
            }
            // Recompute the topic-word table EXACTLY from the merged assignments,
            // rather than a MALLET-style delta merge. Each worker's z is exact for its
            // own (disjoint) docs, so a fresh count over all tokens gives the exact
            // global nkw/nk with no accumulation of merge/clamp error — this is what
            // keeps recovery stable at high worker counts (the delta merge degraded
            // there). The only remaining AD-LDA approximation is the within-sweep phi
            // staleness workers see, which is bounded and resets every sweep.
            for row in nkw.iter_mut() {
                row.iter_mut().for_each(|x| *x = 0.0);
            }
            for dd in 0..d {
                for pos in 0..docs[dd].len() {
                    nkw[z[dd][pos]][docs[dd][pos] as usize] += 1.0;
                }
            }
            for row in 0..k {
                nk[row] = nkw[row].iter().sum();
            }
        }

        // ---- (2) parameter block: Metropolis-within-Gibbs on (b, rho, T) ----
        // Precompute each doc's zbar (fixed given z) and split docs by group.
        let zbar: Vec<Vec<f64>> = (0..d)
            .map(|dd| (0..k).map(|t| ndk[dd][t] / len_d[dd]).collect())
            .collect();
        let (acc, prop) = update_params(
            &mut gp,
            &group_of,
            &parent_of_vec(parents, d),
            &ndk,
            &zbar,
            k,
            num_groups,
            params,
            mh_step_sd,
            rng,
        );
        // Adaptive Metropolis: during burn-in, nudge the proposal sd toward a ~0.25
        // acceptance rate (log-scale multiplicative update). Frozen after burn-in so
        // the collected draws come from a fixed transition kernel (valid MCMC). This
        // is what keeps the credible intervals honest — a fixed too-small step gives
        // an under-mixed chain and intervals that under-cover.
        if sweep < burn && prop > 0 {
            let rate = acc as f64 / prop as f64;
            mh_step_sd *= ((rate - 0.25) * 1.0).exp();
            mh_step_sd = mh_step_sd.clamp(0.01, 3.0);
        }

        // ---- (3) collect draws after burn-in ----
        // Raw per-sweep draws only; means/intervals/R̂ are computed by the multi-chain
        // orchestrator after topic-aligning and pooling these across chains.
        if sweep >= burn {
            let t_draw: Vec<Vec<Vec<f64>>> =
                gp.iter().map(|g| softmax_rows(&g.t_logits, k)).collect();
            for g in 0..num_groups {
                rho_draws[g].push(gp[g].log_rho.exp());
                base_draws[g].push(gp[g].b.iter().map(|&x| x.exp()).collect());
                t_draws[g].push(t_draw[g].clone());
            }
            // theta from current counts.
            for dd in 0..d {
                let denom: f64 = ndk[dd].iter().sum::<f64>();
                for t in 0..k {
                    theta_acc[dd][t] += if denom > 0.0 {
                        ndk[dd][t] / denom
                    } else {
                        1.0 / k as f64
                    };
                }
            }
        }

        // Convergence signal: corpus topic-word predictive log-likelihood under the
        // current assignment, recorded on a coarse grid. Rises and plateaus.
        if sweep % ll_every == 0 || sweep == iters - 1 {
            let mut ll = 0.0f64;
            for dd in 0..d {
                for pos in 0..docs[dd].len() {
                    let t = z[dd][pos];
                    let w = docs[dd][pos] as usize;
                    ll += ((nkw[t][w] + beta) / (nk[t] + vbeta)).ln();
                }
            }
            fit_history.push((sweep, ll));
        }
    }
    // phi posterior mean from this chain's final counts.
    let topic_word: Vec<Vec<f64>> = (0..k)
        .map(|t| {
            let denom = nk[t] + vbeta;
            (0..v).map(|w| (nkw[t][w] + beta) / denom).collect()
        })
        .collect();
    // theta posterior mean, renormalized (this chain).
    let doc_topic: Vec<Vec<f64>> = (0..d)
        .map(|dd| {
            let s: f64 = theta_acc[dd].iter().sum();
            if s > 0.0 {
                theta_acc[dd].iter().map(|&x| x / s).collect()
            } else {
                vec![1.0 / k as f64; k]
            }
        })
        .collect();

    ChainOut {
        num_groups,
        topic_word,
        doc_topic,
        t_draws,
        rho_draws,
        base_draws,
        fit_history,
    }
}

/// Fit ReplyTM with `params.num_chains` independent MCMC chains from dispersed seeds,
/// topic-align them (Hungarian on φ), and pool their draws. Credible intervals are
/// empirical 2.5/97.5% quantiles of the pooled draws (so they include between-chain
/// variance — a single chain badly under-covers), and convergence is split-R̂ on the
/// pooled scalars, not the log-likelihood trace. `parents[d]`/`covariate[d]` as in
/// [`fit_one_chain`]. Deterministic for a fixed seed + `num_chains` + `num_threads`
/// (chains are seeded from the master rng, run in parallel, and pooled in chain order).
pub fn fit<R: Rng>(
    corpus: &Corpus,
    parents: &[i64],
    covariate: &[i64],
    params: &ReplyTmParams,
    iters: usize,
    rng: &mut R,
) -> ReplyTmModel {
    use rayon::prelude::*;
    let k = params.num_topics.max(1);
    let d = corpus.num_docs();
    let num_chains = params.num_chains.max(1);
    // Draw one seed per chain deterministically, then run the chains in parallel; each
    // chain's output depends only on its seed, so collection order (chain index) makes
    // the whole fit reproducible regardless of scheduling.
    let seeds: Vec<u64> = (0..num_chains).map(|_| rng.gen()).collect();
    let chains: Vec<ChainOut> = seeds
        .par_iter()
        .map(|&s| {
            let mut crng = ChaCha8Rng::seed_from_u64(s);
            fit_one_chain(corpus, parents, covariate, params, iters, &mut crng)
        })
        .collect();

    let num_groups = chains[0].num_groups;
    let v = corpus.num_types();

    // Align every chain's topics to chain 0 (the reference) by maximum φ-cosine, via
    // Hungarian assignment. `perm[c][t]` = the reference topic that chain c's topic t
    // maps to; `inv[c][I]` = chain c's topic that carries reference topic I.
    let ref_phi = &chains[0].topic_word;
    let mut inv: Vec<Vec<usize>> = Vec::with_capacity(num_chains);
    for (c, ch) in chains.iter().enumerate() {
        if c == 0 {
            inv.push((0..k).collect());
            continue;
        }
        // cost[t][t'] = -cosine(phi_c[t], phi_ref[t'])
        let cost: Vec<Vec<f64>> = (0..k)
            .map(|t| {
                (0..k)
                    .map(|tp| -cosine(&ch.topic_word[t], &ref_phi[tp], v))
                    .collect()
            })
            .collect();
        let perm = hungarian(&cost); // perm[t] = ref topic for chain topic t
        let mut invc = vec![0usize; k];
        for (t, &rt) in perm.iter().enumerate() {
            invc[rt] = t;
        }
        inv.push(invc);
    }

    // Aligned point estimates for phi/theta: average across chains.
    let mut topic_word = vec![vec![0.0f64; v]; k];
    for (c, ch) in chains.iter().enumerate() {
        for i in 0..k {
            let src = inv[c][i];
            for w in 0..v {
                topic_word[i][w] += ch.topic_word[src][w];
            }
        }
    }
    for row in topic_word.iter_mut() {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            row.iter_mut().for_each(|x| *x /= s);
        }
    }
    let mut doc_topic = vec![vec![0.0f64; k]; d];
    for (c, ch) in chains.iter().enumerate() {
        for dd in 0..d {
            for i in 0..k {
                doc_topic[dd][i] += ch.doc_topic[dd][inv[c][i]];
            }
        }
    }
    for row in doc_topic.iter_mut() {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            row.iter_mut().for_each(|x| *x /= s);
        }
    }

    // Pool aligned draws per group; report mean + 2.5/97.5% quantiles, and track the
    // max split-R̂ across all sampled scalars (T cells, rho, baseline).
    let mut response_matrix = vec![vec![vec![0.0f64; k]; k]; num_groups];
    let mut response_matrix_lo = vec![vec![vec![0.0f64; k]; k]; num_groups];
    let mut response_matrix_hi = vec![vec![vec![0.0f64; k]; k]; num_groups];
    let mut response_strength = vec![0.0f64; num_groups];
    let mut response_strength_lo = vec![0.0f64; num_groups];
    let mut response_strength_hi = vec![0.0f64; num_groups];
    let mut baseline = vec![vec![0.0f64; k]; num_groups];
    let mut baseline_lo = vec![vec![0.0f64; k]; num_groups];
    let mut baseline_hi = vec![vec![0.0f64; k]; num_groups];
    let mut max_rhat = if num_chains > 1 { 0.0 } else { f64::NAN };

    let mean = |xs: &[f64]| -> f64 {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<f64>() / xs.len() as f64
        }
    };

    for g in 0..num_groups {
        // rho: perm-invariant scalar.
        let rho_seqs: Vec<Vec<f64>> = chains.iter().map(|ch| ch.rho_draws[g].clone()).collect();
        let rho_pool: Vec<f64> = rho_seqs.iter().flatten().copied().collect();
        response_strength[g] = mean(&rho_pool);
        response_strength_lo[g] = quantile(&rho_pool, 0.025).max(0.0);
        response_strength_hi[g] = quantile(&rho_pool, 0.975).max(0.0);
        if num_chains > 1 {
            max_rhat = max_rhat.max(split_rhat(&rho_seqs));
        }
        // baseline[g][J]: aligned per topic.
        for j in 0..k {
            let seqs: Vec<Vec<f64>> = chains
                .iter()
                .enumerate()
                .map(|(c, ch)| ch.base_draws[g].iter().map(|dr| dr[inv[c][j]]).collect())
                .collect();
            let pool: Vec<f64> = seqs.iter().flatten().copied().collect();
            baseline[g][j] = mean(&pool);
            baseline_lo[g][j] = quantile(&pool, 0.025).max(0.0);
            baseline_hi[g][j] = quantile(&pool, 0.975).max(0.0);
            if num_chains > 1 {
                max_rhat = max_rhat.max(split_rhat(&seqs));
            }
        }
        // T[g][I][J]: aligned on both topic axes.
        for i in 0..k {
            for j in 0..k {
                let seqs: Vec<Vec<f64>> = chains
                    .iter()
                    .enumerate()
                    .map(|(c, ch)| {
                        let (ci, cj) = (inv[c][i], inv[c][j]);
                        ch.t_draws[g].iter().map(|dr| dr[ci][cj]).collect()
                    })
                    .collect();
                let pool: Vec<f64> = seqs.iter().flatten().copied().collect();
                response_matrix[g][i][j] = mean(&pool);
                response_matrix_lo[g][i][j] = quantile(&pool, 0.025).clamp(0.0, 1.0);
                response_matrix_hi[g][i][j] = quantile(&pool, 0.975).clamp(0.0, 1.0);
                if num_chains > 1 {
                    max_rhat = max_rhat.max(split_rhat(&seqs));
                }
            }
        }
    }
    let converged = num_chains > 1 && max_rhat.is_finite() && max_rhat < 1.1;

    // Parent-topic support (G×K): total parent z̄ mass on each topic over the group's
    // child edges (identifies which rows of T_g are estimable). Uses the pooled
    // doc_topic as z̄_parent.
    let group_of: Vec<usize> =
        if covariate.is_empty() || params.covariate_response == CovResponse::Global {
            vec![0usize; d]
        } else {
            (0..d).map(|dd| covariate[dd].max(0) as usize).collect()
        };
    let mut parent_support = vec![vec![0.0f64; k]; num_groups];
    if !parents.is_empty() {
        for dd in 0..d {
            let p = parents[dd];
            if p >= 0 && (p as usize) < d && (p as usize) != dd {
                let g = group_of[dd];
                for i in 0..k {
                    parent_support[g][i] += doc_topic[p as usize][i];
                }
            }
        }
    }

    // Representative alpha for composition SEs: mean over docs of a_d.
    let parent_of = |dd: usize| -> Option<usize> {
        if parents.is_empty() {
            return None;
        }
        let p = parents[dd];
        if p >= 0 && (p as usize) < d && (p as usize) != dd {
            Some(p as usize)
        } else {
            None
        }
    };
    let mut alpha_mean = vec![0.0f64; k];
    for dd in 0..d {
        let g = group_of[dd];
        let a_d = match parent_of(dd) {
            None => baseline[g].clone(),
            Some(p) => {
                let zbar_p: Vec<f64> = (0..k).map(|t| doc_topic[p][t]).collect();
                child_concentration(
                    &baseline[g],
                    response_strength[g],
                    &response_matrix[g],
                    &zbar_p,
                    k,
                )
            }
        };
        for t in 0..k {
            alpha_mean[t] += a_d[t];
        }
    }
    if d > 0 {
        for t in 0..k {
            alpha_mean[t] /= d as f64;
        }
    }

    ReplyTmModel {
        num_topics: k,
        num_groups,
        topic_word,
        doc_topic,
        response_matrix,
        response_matrix_lo,
        response_matrix_hi,
        response_strength,
        response_strength_lo,
        response_strength_hi,
        baseline,
        baseline_lo,
        baseline_hi,
        alpha_mean,
        parent_support,
        doc_lengths: corpus.docs.iter().map(|doc| doc.len()).collect(),
        fit_history: chains[0].fit_history.clone(),
        max_rhat,
        converged,
    }
}

/// Cosine similarity of two equal-length dense vectors (length `n`).
fn cosine(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let den = (na.sqrt() * nb.sqrt()).max(1e-300);
    dot / den
}

fn parent_of_vec(parents: &[i64], d: usize) -> Vec<Option<usize>> {
    (0..d)
        .map(|dd| {
            let p = parents[dd];
            if p >= 0 && (p as usize) < d && (p as usize) != dd {
                Some(p as usize)
            } else {
                None
            }
        })
        .collect()
}

/// Metropolis-within-Gibbs update of every group's (b, log_rho, T logits) given
/// the current token assignments (summarized by `ndk` and `zbar`). Componentwise
/// RWM with Gaussian shrinkage priors; each proposal is scored on the group's
/// child + root Dirichlet-multinomial log-likelihood.
#[allow(clippy::too_many_arguments)]
fn update_params<R: Rng>(
    gp: &mut [GroupParams],
    group_of: &[usize],
    parent_of: &[Option<usize>],
    ndk: &[Vec<f64>],
    zbar: &[Vec<f64>],
    k: usize,
    num_groups: usize,
    params: &ReplyTmParams,
    step: f64,
    rng: &mut R,
) -> (u64, u64) {
    // (accepts, proposals) across every Metropolis move this call, for step adaptation.
    let mut accepts = 0u64;
    let mut proposals = 0u64;
    // Docs per group, split into roots and children (with parent zbar).
    let mut roots: Vec<Vec<usize>> = vec![Vec::new(); num_groups];
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); num_groups];
    for dd in 0..ndk.len() {
        let g = group_of[dd];
        if parent_of[dd].is_some() {
            kids[g].push(dd);
        } else {
            roots[g].push(dd);
        }
    }

    let (rho_m, rho_sd) = params.rho_prior;
    let t_sd = params.t_prior_sd;

    // For SharedShape, group 0's T logits are the shared shape; others mirror it.
    let shared = params.covariate_response == CovResponse::SharedShape;

    let group_loglik = |g: usize, b: &[f64], log_rho: f64, t: &[Vec<f64>]| -> f64 {
        let base: Vec<f64> = b.iter().map(|&x| x.exp()).collect();
        let rho = log_rho.exp();
        let mut ll = 0.0f64;
        for &dd in &roots[g] {
            ll += log_dm(&ndk[dd], &base);
        }
        for &dd in &kids[g] {
            let p = parent_of[dd].unwrap();
            let a_c = child_concentration(&base, rho, t, &zbar[p], k);
            ll += log_dm(&ndk[dd], &a_c);
        }
        // priors
        for &bi in b {
            ll += -0.5 * (bi / 3.0).powi(2); // weak prior on baseline logs
        }
        ll += -0.5 * ((log_rho - rho_m) / rho_sd).powi(2);
        for row in t.iter() {
            for &l in row {
                ll += -0.5 * (l / t_sd).powi(2);
            }
        }
        ll
    };

    for _ in 0..params.mh_steps {
        for g in 0..num_groups {
            // current T (softmax of logits, honoring SharedShape).
            let cur_logits = if shared {
                gp[0].t_logits.clone()
            } else {
                gp[g].t_logits.clone()
            };
            let cur_t = softmax_rows(&cur_logits, k);
            let mut cur_ll = group_loglik(g, &gp[g].b, gp[g].log_rho, &cur_t);

            // (a) baseline b
            for j in 0..k {
                let prop = gp[g].b[j] + step * gaussian(rng);
                let mut b2 = gp[g].b.clone();
                b2[j] = prop;
                let ll2 = group_loglik(g, &b2, gp[g].log_rho, &cur_t);
                proposals += 1;
                if (ll2 - cur_ll).exp() > rng.gen::<f64>() {
                    gp[g].b[j] = prop;
                    cur_ll = ll2;
                    accepts += 1;
                }
            }
            // (b) log_rho
            {
                let prop = gp[g].log_rho + step * gaussian(rng);
                let ll2 = group_loglik(g, &gp[g].b, prop, &cur_t);
                proposals += 1;
                if (ll2 - cur_ll).exp() > rng.gen::<f64>() {
                    gp[g].log_rho = prop;
                    cur_ll = ll2;
                    accepts += 1;
                }
            }
            // (c) T logits, one whole ROW per proposal (K rescores/group instead of
            // K^2 — a row is the natural block since row i is a single parent-topic's
            // response simplex). Skip for SharedShape non-lead groups.
            if !(shared && g != 0) {
                for i in 0..k {
                    let mut lg = if shared {
                        gp[0].t_logits.clone()
                    } else {
                        gp[g].t_logits.clone()
                    };
                    for j in 0..k {
                        lg[i][j] += step * gaussian(rng);
                    }
                    let t2 = softmax_rows(&lg, k);
                    let ll2 = group_loglik(g, &gp[g].b, gp[g].log_rho, &t2);
                    proposals += 1;
                    if (ll2 - cur_ll).exp() > rng.gen::<f64>() {
                        if shared {
                            gp[0].t_logits[i].clone_from(&lg[i]);
                        } else {
                            gp[g].t_logits[i].clone_from(&lg[i]);
                        }
                        cur_ll = ll2;
                        accepts += 1;
                    }
                }
            }
        }
        // SharedShape: mirror lead logits into the others so downstream reads are consistent.
        if shared {
            for g in 1..num_groups {
                gp[g].t_logits = gp[0].t_logits.clone();
            }
        }
    }
    (accepts, proposals)
}

/// Box-Muller standard normal from the seeded RNG (fixed order => deterministic).
fn gaussian<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Per-cell 2.5% / 97.5% quantiles of retained T draws.
/// Empirical `q`-quantile (0..=1) of a sample by linear interpolation between order
/// statistics (the "type-7"/numpy default). Clones + sorts, so it is O(n log n) per
/// call — fine at our draw counts. Pooled cross-chain draws make these quantiles
/// include between-chain variance, which is the whole point of multi-chain intervals.
fn quantile(xs: &[f64], q: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return xs[0];
    }
    let mut s: Vec<f64> = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q.clamp(0.0, 1.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        s[lo]
    } else {
        let frac = pos - lo as f64;
        s[lo] * (1.0 - frac) + s[hi] * frac
    }
}

/// Split-`R̂` (potential scale reduction factor) for one scalar parameter across
/// `chains`. Each chain is split in half (so a single non-stationary chain is caught
/// as between-half disagreement), giving `m = 2·C` sequences of length `n`. Returns
/// `sqrt(((n-1)/n·W + B/n) / W)`; ≈1 at convergence, >1.1 flags a chain that has not
/// mixed. Returns 1.0 when there is no variance to compare (identical draws, or too
/// few draws/chains) so a degenerate-but-consistent parameter never reports spurious
/// non-convergence.
fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    // Split each chain in half; keep only the usable equal-length prefix.
    let mut seqs: Vec<&[f64]> = Vec::new();
    let mut n = usize::MAX;
    for c in chains {
        let h = c.len() / 2;
        if h == 0 {
            continue;
        }
        seqs.push(&c[..h]);
        seqs.push(&c[h..2 * h]);
        n = n.min(h);
    }
    let m = seqs.len();
    if m < 2 || n < 2 || n == usize::MAX {
        return 1.0;
    }
    let means: Vec<f64> = seqs
        .iter()
        .map(|s| s[..n].iter().sum::<f64>() / n as f64)
        .collect();
    let grand = means.iter().sum::<f64>() / m as f64;
    let b = n as f64 / (m as f64 - 1.0) * means.iter().map(|&mj| (mj - grand).powi(2)).sum::<f64>();
    let w = seqs
        .iter()
        .zip(&means)
        .map(|(s, &mj)| s[..n].iter().map(|&x| (x - mj).powi(2)).sum::<f64>() / (n as f64 - 1.0))
        .sum::<f64>()
        / m as f64;
    if w <= 1e-300 {
        return 1.0;
    }
    let var_plus = (n as f64 - 1.0) / n as f64 * w + b / n as f64;
    (var_plus / w).sqrt()
}

/// Minimum-cost perfect assignment on a square matrix (Kuhn–Munkres / Hungarian,
/// O(n³) shortest-augmenting-path form). Returns `perm` with `perm[row] = col`. Used
/// to align a chain's topics to a reference chain's before pooling draws: with
/// `cost[t][t'] = −cosine(φ_chain[t], φ_ref[t'])`, `perm[t]` is the reference topic
/// that chain-topic `t` maps to.
fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row (1-indexed) assigned to col j
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut perm = vec![0usize; n];
    for j in 1..=n {
        perm[p[j] - 1] = j - 1;
    }
    perm
}

impl Estimator for ReplyTmModel {
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

impl DirichletModel for ReplyTmModel {
    fn alpha(&self) -> Vec<f64> {
        self.alpha_mean.clone()
    }
    fn theta_draws(&self) -> Vec<Vec<Vec<f64>>> {
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
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn quantile_matches_numpy_type7() {
        let xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        // numpy.quantile(xs, q) with default linear interpolation.
        assert!((quantile(&xs, 0.0) - 1.0).abs() < 1e-9);
        assert!((quantile(&xs, 1.0) - 9.0).abs() < 1e-9);
        assert!((quantile(&xs, 0.5) - 3.5).abs() < 1e-9);
        assert!((quantile(&xs, 0.25) - 1.75).abs() < 1e-9);
    }

    #[test]
    fn split_rhat_near_one_for_iid_and_high_for_offset_chains() {
        // Two well-mixed chains sampling the same distribution -> R̂ ≈ 1.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let draw = |rng: &mut ChaCha8Rng| {
            (0..400)
                .map(|_| {
                    // Box–Muller standard normal.
                    let u1: f64 = (rng.gen::<u64>() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
                    let u2: f64 = (rng.gen::<u64>() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                })
                .collect::<Vec<f64>>()
        };
        let c0 = draw(&mut rng);
        let c1 = draw(&mut rng);
        let rhat_good = split_rhat(&[c0.clone(), c1.clone()]);
        assert!(rhat_good < 1.1, "iid chains R̂ = {rhat_good}");
        // Same chains but one shifted by a large constant -> big between-chain var.
        let c1_off: Vec<f64> = c1.iter().map(|&x| x + 10.0).collect();
        let rhat_bad = split_rhat(&[c0, c1_off]);
        assert!(rhat_bad > 1.5, "offset chains R̂ = {rhat_bad}");
        // Degenerate (identical constant) -> defined, returns 1.0.
        assert_eq!(split_rhat(&[vec![2.0; 20], vec![2.0; 20]]), 1.0);
    }

    #[test]
    fn hungarian_recovers_a_known_permutation() {
        // cost minimized when row t -> col perm_true[t]; give those cells cost 0 and
        // everything else cost 1, so the optimal assignment IS perm_true.
        let perm_true = [2usize, 0, 3, 1];
        let n = perm_true.len();
        let mut cost = vec![vec![1.0f64; n]; n];
        for (t, &c) in perm_true.iter().enumerate() {
            cost[t][c] = 0.0;
        }
        assert_eq!(hungarian(&cost), perm_true.to_vec());
        // Identity when the diagonal is cheapest.
        let eye: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 0.0 } else { 1.0 }).collect())
            .collect();
        assert_eq!(hungarian(&eye), vec![0, 1, 2, 3]);
    }

    // Build a corpus from token-id documents (mirrors csatm.rs test helper).
    fn corpus_from(docs: Vec<Vec<u32>>, v: usize) -> Corpus {
        Corpus {
            id_to_word: (0..v).map(|i| format!("w{i}")).collect(),
            docs,
            doc_names: Vec::new(),
            doc_labels: Vec::new(),
            doc_freqs: vec![0; v],
            total_freqs: vec![0; v],
        }
    }

    #[test]
    fn reply_tm_reduces_to_lda_with_no_replies() {
        // All roots, one group, rho prior centered low: topics should still be
        // recovered as ordinary LDA. Two clearly separated word blocks.
        let mut docs = Vec::new();
        for _ in 0..60 {
            docs.push(vec![0u32, 1, 0, 1, 0]);
            docs.push(vec![2u32, 3, 2, 3, 2]);
        }
        let c = corpus_from(docs, 4);
        let mut params = ReplyTmParams {
            num_topics: 2,
            burn: 30,
            ..Default::default()
        };
        params.mh_steps = 2;
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let m = fit(&c, &[], &[], &params, 120, &mut rng);
        assert_eq!(m.topic_word.len(), 2);
        // each recovered topic concentrates on one of the two blocks
        let concentrated = m.topic_word.iter().all(|row| {
            let block_a = row[0] + row[1];
            let block_b = row[2] + row[3];
            (block_a - block_b).abs() > 0.5
        });
        assert!(concentrated, "topics did not separate: {:?}", m.topic_word);
    }

    #[test]
    fn reply_tm_is_deterministic() {
        let docs: Vec<Vec<u32>> = (0..40)
            .map(|i| {
                if i % 2 == 0 {
                    vec![0u32, 1, 0]
                } else {
                    vec![2u32, 3, 2]
                }
            })
            .collect();
        let c = corpus_from(docs, 4);
        let params = ReplyTmParams {
            num_topics: 2,
            burn: 10,
            mh_steps: 2,
            ..Default::default()
        };
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(7);
        let m1 = fit(&c, &[], &[], &params, 50, &mut r1);
        let m2 = fit(&c, &[], &[], &params, 50, &mut r2);
        assert_eq!(m1.topic_word, m2.topic_word);
        assert_eq!(m1.response_matrix, m2.response_matrix);
    }

    // Exact tiny-tree ENUMERATION GATE. On a 4-doc reply tree (K=2, V=3, 2
    // tokens/doc => 2^8 = 256 z-states) with FIXED (b, rho, T), the collapsed-Gibbs
    // token conditional (`token_logweights`, the shipped code) must converge to the
    // exact enumerated posterior; and the same sampler with the children factor
    // OMITTED must diverge from it. This is the one bug (dropping the child
    // Dirichlet-multinomial term) that silently targets the wrong posterior.
    // Returns (TV of the correct sampler vs the exact enumerated posterior, TV of the
    // child-omitted "buggy" sampler). `rho` sets the per-token step `rho/len_parent`:
    // rho=8, len=2 => step=4 (exact children path); rho=0.4 => step=0.2 (fast path).
    fn enum_gate_tvs(rho: f64) -> (f64, f64) {
        let k = 2usize;
        let v = 3usize;
        let parents: [i64; 4] = [-1, 0, 0, 1];
        let words: [[usize; 2]; 4] = [[0, 1], [1, 2], [0, 2], [2, 0]];
        let d = 4usize;
        let len_d = [2.0f64; 4];
        let children: Vec<Vec<usize>> = (0..d)
            .map(|dd| (0..d).filter(|&c| parents[c] == dd as i64).collect())
            .collect();
        let group_of = vec![0usize; d];
        // Fixed parameters (one group).
        let beta = 0.5f64;
        let vbeta = v as f64 * beta;
        let base_now = vec![vec![0.3f64, 0.3]]; // exp(b) baseline concentration
        let rho_now = vec![rho];
        let t_now = vec![vec![vec![0.85f64, 0.15], vec![0.25, 0.75]]]; // T row-simplex

        // a_d for a full assignment (root: base; child: base + rho T^T zbar_parent).
        let a_of = |dd: usize, z: &[Vec<usize>]| -> Vec<f64> {
            let p = parents[dd];
            if p < 0 {
                base_now[0].clone()
            } else {
                let pu = p as usize;
                let mut zbar = vec![0.0f64; k];
                for &t in &z[pu] {
                    zbar[t] += 1.0 / len_d[pu];
                }
                child_concentration(&base_now[0], rho_now[0], &t_now[0], &zbar, k)
            }
        };
        // Exact joint log P(z, w) = Σ_doc logDM_theta + Σ_topic logDM_phi.
        let joint_log = |z: &[Vec<usize>]| -> f64 {
            let mut lp = 0.0f64;
            for dd in 0..d {
                let mut c = vec![0.0f64; k];
                for &t in &z[dd] {
                    c[t] += 1.0;
                }
                lp += log_dm(&c, &a_of(dd, z));
            }
            let mut m = vec![vec![0.0f64; v]; k];
            for dd in 0..d {
                for pos in 0..2 {
                    m[z[dd][pos]][words[dd][pos]] += 1.0;
                }
            }
            let bvec = vec![beta; v];
            for kk in 0..k {
                lp += log_dm(&m[kk], &bvec);
            }
            lp
        };
        // Enumerate 256 states.
        let state_of = |code: u32| -> Vec<Vec<usize>> {
            (0..d)
                .map(|dd| {
                    (0..2)
                        .map(|pos| ((code >> (dd * 2 + pos)) & 1) as usize)
                        .collect()
                })
                .collect()
        };
        let mut exact = vec![0.0f64; 256];
        let mut maxlp = f64::NEG_INFINITY;
        for code in 0..256u32 {
            let lp = joint_log(&state_of(code));
            exact[code as usize] = lp;
            if lp > maxlp {
                maxlp = lp;
            }
        }
        let mut zsum = 0.0f64;
        for p in exact.iter_mut() {
            *p = (*p - maxlp).exp();
            zsum += *p;
        }
        for p in exact.iter_mut() {
            *p /= zsum;
        }

        // z-only Gibbs using the shipped `token_logweights`, params fixed.
        let run = |include_children: bool, iters: usize, burn: usize| -> Vec<f64> {
            let mut rng = ChaCha8Rng::seed_from_u64(20260826);
            let mut z: Vec<Vec<usize>> = (0..d).map(|_| vec![0usize; 2]).collect();
            let mut ndk = vec![vec![0.0f64; k]; d];
            let mut nkw = vec![vec![0.0f64; v]; k];
            let mut nk = vec![0.0f64; k];
            for dd in 0..d {
                for pos in 0..2 {
                    let t = (rng.gen::<f64>() * k as f64) as usize % k;
                    z[dd][pos] = t;
                    ndk[dd][t] += 1.0;
                    nkw[t][words[dd][pos]] += 1.0;
                    nk[t] += 1.0;
                }
            }
            let mut freq = vec![0.0f64; 256];
            let mut collected = 0.0f64;
            let mut cond = vec![0.0f64; k];
            for it in 0..iters {
                for dd in 0..d {
                    let a_d = a_of(dd, &z);
                    for pos in 0..2 {
                        let w = words[dd][pos];
                        let old = z[dd][pos];
                        ndk[dd][old] -= 1.0;
                        nkw[old][w] -= 1.0;
                        nk[old] -= 1.0;
                        token_logweights(
                            &mut cond,
                            w,
                            k,
                            &ndk,
                            &nkw,
                            &nk,
                            beta,
                            vbeta,
                            dd,
                            &a_d,
                            len_d[dd],
                            &children[dd],
                            &group_of,
                            &base_now,
                            &rho_now,
                            &t_now,
                            include_children,
                        );
                        let mx = cond.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let mut tot = 0.0;
                        for t in 0..k {
                            cond[t] = (cond[t] - mx).exp();
                            tot += cond[t];
                        }
                        let mut r = rng.gen::<f64>() * tot;
                        let mut pick = k - 1;
                        for t in 0..k {
                            r -= cond[t];
                            if r <= 0.0 {
                                pick = t;
                                break;
                            }
                        }
                        z[dd][pos] = pick;
                        ndk[dd][pick] += 1.0;
                        nkw[pick][w] += 1.0;
                        nk[pick] += 1.0;
                    }
                }
                if it >= burn {
                    let mut code = 0u32;
                    for dd in 0..d {
                        for pos in 0..2 {
                            code |= (z[dd][pos] as u32) << (dd * 2 + pos);
                        }
                    }
                    freq[code as usize] += 1.0;
                    collected += 1.0;
                }
            }
            for f in freq.iter_mut() {
                *f /= collected;
            }
            freq
        };
        let tv = |emp: &[f64]| -> f64 {
            0.5 * exact
                .iter()
                .zip(emp)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
        };
        (
            tv(&run(true, 120_000, 12_000)),
            tv(&run(false, 120_000, 12_000)),
        )
    }

    // Exact-path gate: with rho=8 (step=4) the shipped conditional uses the exact
    // children DM; it must match the enumerated posterior, and omitting the children
    // factor must diverge (proving the factor is load-bearing).
    #[test]
    fn reply_tm_enumeration_gate() {
        let (tv_correct, tv_buggy) = enum_gate_tvs(8.0);
        assert!(
            tv_correct < 0.04,
            "correct sampler should match exact posterior; TV={tv_correct}"
        );
        assert!(
            tv_buggy > 2.5 * tv_correct,
            "gate should FAIL the child-omitted sampler; TV_correct={tv_correct} TV_buggy={tv_buggy}"
        );
    }

    // Fast-path fidelity: with rho=0.4 (step=0.2, the FAST_STEP boundary — the
    // worst case within the fast regime) the shipped conditional uses the
    // first-order digamma approximation of the children factor. Its stationary
    // distribution must still match the exact enumerated posterior within the
    // O(step^2) bias bound. Closes the coverage gap that the exact-path gate leaves.
    #[test]
    fn reply_tm_fast_path_matches_exact() {
        let (tv_fast, _) = enum_gate_tvs(0.4);
        assert!(
            tv_fast < 0.05,
            "fast-path (digamma) sampler should match the exact posterior at the \
             FAST_STEP boundary; TV={tv_fast}"
        );
    }

    // High-fan-out fast-path stress (the adversarial worst case): a star tree — one
    // root with SIX children — so the first-order children-factor errors accumulate
    // across many children in the root's token conditional, at the FAST_STEP boundary
    // with small (high-curvature) baseline concentration and a peaked response row.
    // 1 root + 6 children x 2 tokens = 14 tokens => 2^14 exact states. The shipped
    // fast-path sampler must still match the exact enumerated posterior.
    #[test]
    fn reply_tm_fast_path_high_fanout() {
        let k = 2usize;
        let v = 3usize;
        let d = 7usize; // doc 0 = root, docs 1..7 = its children
        let parents: Vec<i64> = std::iter::once(-1)
            .chain(std::iter::repeat_n(0, 6))
            .collect();
        let words: [[usize; 2]; 7] = [[0, 1], [1, 2], [0, 2], [2, 0], [0, 0], [1, 1], [2, 2]];
        let len_d = [2.0f64; 7];
        let children: Vec<Vec<usize>> = (0..d)
            .map(|dd| (0..d).filter(|&c| parents[c] == dd as i64).collect())
            .collect();
        let group_of = vec![0usize; d];
        let beta = 0.5f64;
        let vbeta = v as f64 * beta;
        let base_now = vec![vec![0.3f64, 0.3]]; // small baseline => high psi' curvature
        let rho_now = vec![0.4f64]; // step = 0.4 * 0.5 = 0.2 (FAST_STEP boundary)
        let t_now = vec![vec![vec![0.9f64, 0.1], vec![0.1, 0.9]]]; // peaked response

        let a_of = |dd: usize, z: &[Vec<usize>]| -> Vec<f64> {
            if parents[dd] < 0 {
                base_now[0].clone()
            } else {
                let pu = parents[dd] as usize;
                let mut zbar = vec![0.0f64; k];
                for &t in &z[pu] {
                    zbar[t] += 1.0 / len_d[pu];
                }
                child_concentration(&base_now[0], rho_now[0], &t_now[0], &zbar, k)
            }
        };
        let joint_log = |z: &[Vec<usize>]| -> f64 {
            let mut lp = 0.0f64;
            for dd in 0..d {
                let mut c = vec![0.0f64; k];
                for &t in &z[dd] {
                    c[t] += 1.0;
                }
                lp += log_dm(&c, &a_of(dd, z));
            }
            let mut m = vec![vec![0.0f64; v]; k];
            for dd in 0..d {
                for pos in 0..2 {
                    m[z[dd][pos]][words[dd][pos]] += 1.0;
                }
            }
            let bvec = vec![beta; v];
            for kk in 0..k {
                lp += log_dm(&m[kk], &bvec);
            }
            lp
        };
        let nstates = 1u32 << (2 * d as u32);
        let state_of = |code: u32| -> Vec<Vec<usize>> {
            (0..d)
                .map(|dd| {
                    (0..2)
                        .map(|pos| ((code >> (dd * 2 + pos)) & 1) as usize)
                        .collect()
                })
                .collect()
        };
        let mut exact = vec![0.0f64; nstates as usize];
        let mut maxlp = f64::NEG_INFINITY;
        for code in 0..nstates {
            let lp = joint_log(&state_of(code));
            exact[code as usize] = lp;
            if lp > maxlp {
                maxlp = lp;
            }
        }
        let mut zsum = 0.0f64;
        for p in exact.iter_mut() {
            *p = (*p - maxlp).exp();
            zsum += *p;
        }
        for p in exact.iter_mut() {
            *p /= zsum;
        }
        // Fast-path Gibbs (params fixed), collect state frequencies.
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let mut z: Vec<Vec<usize>> = (0..d).map(|_| vec![0usize; 2]).collect();
        let mut ndk = vec![vec![0.0f64; k]; d];
        let mut nkw = vec![vec![0.0f64; v]; k];
        let mut nk = vec![0.0f64; k];
        for dd in 0..d {
            for pos in 0..2 {
                let t = (rng.gen::<f64>() * k as f64) as usize % k;
                z[dd][pos] = t;
                ndk[dd][t] += 1.0;
                nkw[t][words[dd][pos]] += 1.0;
                nk[t] += 1.0;
            }
        }
        let mut freq = vec![0.0f64; nstates as usize];
        let mut collected = 0.0f64;
        let mut cond = vec![0.0f64; k];
        let parent_slice: Vec<Option<usize>> = parents
            .iter()
            .map(|&p| if p >= 0 { Some(p as usize) } else { None })
            .collect();
        let docs_u32: Vec<Vec<u32>> = words
            .iter()
            .map(|w| vec![w[0] as u32, w[1] as u32])
            .collect();
        for it in 0..150_000 {
            for dd in 0..d {
                resample_doc(
                    dd,
                    k,
                    &docs_u32,
                    &mut z,
                    &mut ndk,
                    &mut nkw,
                    &mut nk,
                    beta,
                    vbeta,
                    &parent_slice,
                    &len_d,
                    &group_of,
                    &base_now,
                    &rho_now,
                    &t_now,
                    &children,
                    &mut cond,
                    &mut rng,
                );
            }
            if it >= 15_000 {
                let mut code = 0u32;
                for dd in 0..d {
                    for pos in 0..2 {
                        code |= (z[dd][pos] as u32) << (dd * 2 + pos);
                    }
                }
                freq[code as usize] += 1.0;
                collected += 1.0;
            }
        }
        for f in freq.iter_mut() {
            *f /= collected;
        }
        // Full-state TV over 2^14 states is MC-noise-dominated at this sample count,
        // so compare a LOW-dimensional statistic instead: each doc's marginal
        // P(#topic-0 tokens in {0,1,2}). The fast-path accumulation bias, if present,
        // shows up on the high-fan-out ROOT's marginal; these marginals have tiny MC
        // noise. Require the worst per-doc marginal L1 error to be small.
        let doc_count0_marginal = |dist: &[f64]| -> Vec<[f64; 3]> {
            let mut marg = vec![[0.0f64; 3]; d];
            for (code, &p) in dist.iter().enumerate() {
                let z = state_of(code as u32);
                for dd in 0..d {
                    let c0 = z[dd].iter().filter(|&&t| t == 0).count();
                    marg[dd][c0] += p;
                }
            }
            marg
        };
        let me = doc_count0_marginal(&exact);
        let ms = doc_count0_marginal(&freq);
        let worst = (0..d)
            .map(|dd| (0..3).map(|c| (me[dd][c] - ms[dd][c]).abs()).sum::<f64>())
            .fold(0.0, f64::max);
        assert!(
            worst < 0.03,
            "high-fan-out fast-path biases a doc's topic marginal; worst L1={worst}"
        );
    }

    #[test]
    fn reply_tm_threaded_is_deterministic_and_recovers() {
        // A reply chain over two planted word blocks; AD-LDA with 3 workers must be
        // reproducible for a fixed thread count and still separate the blocks.
        let mut docs = Vec::new();
        for _ in 0..80 {
            docs.push(vec![0u32, 1, 2, 0, 1]);
            docs.push(vec![3u32, 4, 5, 3, 4]);
        }
        let c = corpus_from(docs, 6);
        let n = c.num_docs();
        let parents: Vec<i64> = (0..n as i64).map(|i| i - 1).collect(); // one long chain
        let params = ReplyTmParams {
            num_topics: 2,
            burn: 20,
            mh_steps: 1,
            num_threads: 3,
            ..Default::default()
        };
        let mut r1 = ChaCha8Rng::seed_from_u64(5);
        let mut r2 = ChaCha8Rng::seed_from_u64(5);
        let m1 = fit(&c, &parents, &[], &params, 80, &mut r1);
        let m2 = fit(&c, &parents, &[], &params, 80, &mut r2);
        assert_eq!(
            m1.topic_word, m2.topic_word,
            "num_threads=3 not deterministic"
        );
        let concentrated = m1
            .topic_word
            .iter()
            .all(|row| ((row[0] + row[1] + row[2]) - (row[3] + row[4] + row[5])).abs() > 0.5);
        assert!(
            concentrated,
            "threaded fit did not separate blocks: {:?}",
            m1.topic_word
        );
    }

    #[test]
    fn reply_tm_conforms() {
        let docs: Vec<Vec<u32>> = (0..20).map(|_| vec![0u32, 1, 2, 3]).collect();
        let c = corpus_from(docs, 4);
        let params = ReplyTmParams {
            num_topics: 2,
            burn: 5,
            mh_steps: 1,
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let m = fit(&c, &[], &[], &params, 20, &mut rng);
        assert!(crate::conformance::check_conformance(&m).is_empty());
    }
}
