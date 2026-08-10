"""Committed gold for topica TopicsOverTime (#694).

Topics over Time (Wang & McCallum, "Topics over Time: A Non-Markov Continuous-Time
Model of Topical Trends," KDD 2006) is LDA with a per-topic Beta density over each
document's normalized timestamp. Collapsed Gibbs is the LDA conditional times the
topic's Beta time-likelihood; the per-topic Beta parameters are estimated by method of
moments from the timestamps of each topic's assigned tokens.

There is NO single maintained reference implementation (the one public port is
GPL + unmaintained), so this gold validates two ways, neither of which needs a
reference topic-modeling library:

  1. PLANTED CONTINUOUS-TIME RECOVERY. A fixed-seed synthetic corpus has K disjoint
     word blocks, and each block's documents are drawn from a distinct, known Beta
     density over time (early / middle / late). A faithful ToT must recover both the
     word blocks (topic-word cosine) AND their temporal ordering (the recovered peaks,
     aligned to the planted topics, are monotone in the planted time centers).

  2. INDEPENDENT NUMERICAL CHECKS. (a) topica's fitted psi are recomputed from scratch
     by a numpy method-of-moments fit on each recovered topic's assigned-document times
     and must agree — this checks topica's Rust MoM against an independent implementation,
     not against itself. (b) With scipy present, each psi is validated as a proper Beta
     density (grid pdf integrates to 1), scipy's analytic mean equals topica's reported
     mean, and topica's reported peak equals the analytic mode mapped back to original
     units. scipy is optional; when absent that sub-check is reported as SKIPPED and does
     NOT count toward PARITY OK (so a scipy-less run cannot silently "pass" it).

  Note: this planted corpus uses (near-)disjoint word blocks, so plain LDA alone would
  also recover the blocks and a post-hoc MoM would then recover the temporal order — i.e.
  the gold measures RECOVERY and NUMERICS, not whether the Beta factor causally drives
  sampling. That causal isolation is the job of the Rust unit test
  `tot_time_factor_drives_assignment` (ambiguous words + a time-blind control fit), which
  shows the time factor lifts era recovery far above the time-blind baseline.

The gold (planted corpus + planted Betas) is frozen dependency-free (numpy only):

    python parity/tot_gold.py --regenerate   # numpy only
    python parity/tot_gold.py                # offline compare vs topica
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "tot"

# --- planted synthetic corpus -------------------------------------------------
K = 3                    # topics
BLOCK = 6                # words per topic block
V = K * BLOCK            # vocabulary size
DOCS_PER_TOPIC = 60
DOC_LEN = 20
CORPUS_SEED = 694
TOPIC_SPREAD = 0.12      # P(a topic-t token comes from another block) -> phi overlap
# Planted per-topic Beta over normalized [0,1] time: early, middle, late.
PLANTED_PSI = np.array([[2.0, 5.0], [5.0, 5.0], [5.0, 2.0]])
ALPHA = 50.0 / K         # paper default
BETA = 0.1               # paper default


def _words_for_topic(t: int) -> list[str]:
    return [f"w{t * BLOCK + i}" for i in range(BLOCK)]


def build_corpus():
    """Return (docs, times, vocab, planted_block, planted_psi) deterministically."""
    rng = np.random.default_rng(CORPUS_SEED)
    docs: list[list[str]] = []
    times: list[float] = []
    planted_block: list[int] = []
    for t in range(K):
        a, b = PLANTED_PSI[t]
        for _ in range(DOCS_PER_TOPIC):
            toks = []
            for _ in range(DOC_LEN):
                wt = t if rng.random() >= TOPIC_SPREAD else int(rng.integers(0, K))
                toks.append(_words_for_topic(wt)[int(rng.integers(0, BLOCK))])
            docs.append(toks)
            # timestamp drawn from the topic's planted Beta, then mapped to a wide
            # arbitrary numeric range (years) to exercise min-max normalization.
            u = float(rng.beta(a, b))
            times.append(1900.0 + 100.0 * u)
            planted_block.append(t)
    vocab = sorted({w for d in docs for w in d}, key=lambda w: int(w[1:]))
    return docs, times, vocab, np.array(planted_block), PLANTED_PSI.copy()


def regenerate():
    docs, times, vocab, planted_block, planted_psi = build_corpus()
    # planted phi as one-hot blocks (uniform within a block), for a cosine target.
    phi = np.zeros((K, V))
    for t in range(K):
        for i in range(BLOCK):
            phi[t, t * BLOCK + i] = 1.0 / BLOCK
    meta = {
        "model": "topics_over_time",
        "reference": "planted continuous-time + scipy.stats.beta numerical check",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "K": K, "V": V, "n_docs": len(docs),
        "alpha": ALPHA, "beta": BETA,
        "corpus_seed": CORPUS_SEED, "topic_spread": TOPIC_SPREAD,
        "vocab": vocab,
        "docs": docs,
        "times": times,
    }
    harness.save_gold(
        NAME,
        {"phi": phi, "planted_block": planted_block, "planted_psi": planted_psi},
        meta,
    )
    print(f"[{NAME}] wrote gold: planted K={K} V={V} docs={len(docs)}; "
          f"planted psi (early/mid/late)=\n{planted_psi}")


def _beta_mode(a: float, b: float) -> float:
    """Mode of Beta(a,b) on [0,1] (mirrors src/topics_over_time.rs::beta_peak): interior
    when a>1,b>1; boundary for a monotone density; NaN for U-shaped (a<1,b<1) or uniform.
    A unit shape with the other <1 is still monotone (Beta(1,b<1) increases -> mode 1)."""
    if a > 1 and b > 1:
        return (a - 1.0) / (a + b - 2.0)
    if (a < 1 and b < 1) or (a == 1 and b == 1):
        return float("nan")
    if a <= 1 and b >= 1:
        return 0.0
    return 1.0


def compare():
    """Offline: fit topica ToT on the SAME corpus and check recovery + Beta numerics."""
    arrays, meta = harness.load_gold(NAME)
    phi_g = arrays["phi"]
    planted_block = arrays["planted_block"]
    planted_psi = arrays["planted_psi"]
    docs = meta["docs"]
    times = meta["times"]
    vocab = meta["vocab"]

    try:
        import topica
    except ImportError:
        print(f"[{NAME}] topica not importable; skipping compare.")
        return
    if not hasattr(topica, "TopicsOverTime"):
        print(f"[{NAME}] topica.TopicsOverTime not built yet; gold is ready. Skipping.")
        return

    m = topica.TopicsOverTime(K, alpha=meta["alpha"], beta=meta["beta"], seed=13).fit(
        docs, times=times, iters=1000
    )

    # topica phi mapped into gold `vocab` column order.
    tv_idx = {w: i for i, w in enumerate(m.vocabulary)}
    tw = np.asarray(m.topic_word)
    phi_t = np.zeros((K, len(vocab)))
    for j, w in enumerate(vocab):
        if w in tv_idx:
            phi_t[:, j] = tw[:, tv_idx[w]]
    phi_t = phi_t / phi_t.sum(axis=1, keepdims=True)

    # Align recovered topics to the planted blocks by phi, reuse the permutation for time.
    cos, perm = harness.align_cosine(phi_t, phi_g)

    psi = np.asarray(m.topic_time)          # (K, 2) in normalized time
    mean_norm = psi[:, 0] / psi.sum(axis=1)
    # planted time centers (Beta mean) in normalized units, ordered early<mid<late.
    planted_mean = planted_psi[:, 0] / planted_psi.sum(axis=1)

    # aligned recovered means vs planted centers. align_cosine returns perm with
    # perm[topica_topic] = planted_topic, so scatter (not gather) into planted order.
    aligned_mean = np.empty(K)
    aligned_mean[perm] = mean_norm
    center_corr = float(np.corrcoef(aligned_mean, planted_mean)[0, 1])
    order_ok = bool(np.array_equal(np.argsort(aligned_mean), np.argsort(planted_mean)))

    print(f"[{NAME}] phi aligned cosine     = {cos:.3f}")
    print(f"[{NAME}] time center corr        = {center_corr:.3f}")
    print(f"[{NAME}] temporal order recovered = {order_ok}  "
          f"(planted mean {np.round(planted_mean,3)} -> recovered {np.round(aligned_mean,3)})")

    # Independent recompute of psi by numpy method-of-moments. topica estimates psi from
    # per-token assignment counts n_dk; those are exactly recoverable from the reported
    # (smoothed) doc_topic via n_dk = doc_topic*(len_d + alpha_sum) - alpha, so numpy can
    # reconstruct the token counts and redo the MoM from scratch. This checks topica's
    # Rust MoM arithmetic against an independent implementation on the same weighting,
    # rather than against topica itself. (A hard argmax grouping would NOT match, because
    # each topic's token-time distribution is softly contaminated by the corpus's
    # cross-block spread — the reconstruction captures that; argmax does not.)
    tn = (np.asarray(times) - min(times)) / (max(times) - min(times))
    dt = np.asarray(m.doc_topic)
    lens = np.array([len(d) for d in docs], dtype=float)
    alpha_k = meta["alpha"]
    alpha_sum = alpha_k * K
    n_dk = np.clip(dt * (lens[:, None] + alpha_sum) - alpha_k, 0.0, None)  # ~integer counts
    mom_ok = True
    for kk in range(K):
        w = n_dk[:, kk]
        tot = w.sum()
        a_t, b_t = psi[kk]
        if tot < 5:
            continue
        mu = float((w * tn).sum() / tot)
        var = float((w * tn * tn).sum() / tot - mu * mu)
        span = mu * (1.0 - mu)
        if var <= 0 or var >= span:
            continue  # both would fall back to uniform; not a numeric comparison
        c = span / var - 1.0
        a_np, b_np = mu * c, (1.0 - mu) * c
        if abs(a_np - a_t) / max(a_t, 1.0) > 0.05 or abs(b_np - b_t) / max(b_t, 1.0) > 0.05:
            mom_ok = False
    print(f"[{NAME}] independent numpy-MoM psi match = {mom_ok}")

    # Independent numerical Beta reference (scipy): (1) psi are proper Beta densities
    # (grid pdf integrates to 1); (2) scipy's analytic mean equals topica's mean_norm;
    # (3) topica's reported peak equals the analytic mode mapped to original units. If
    # scipy is absent the check is SKIPPED (it does NOT count toward PARITY OK).
    beta_ok = None
    try:
        from scipy.stats import beta as _beta

        beta_ok = True
        tmin, tmax = m.time_range
        peak = np.asarray(m.topic_time_peak)
        grid = np.linspace(1e-6, 1 - 1e-6, 20001)
        for kk in range(K):
            a, b = psi[kk]
            # (1) proper density
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            integral = _trapz(_beta.pdf(grid, a, b), grid)
            if abs(integral - 1.0) > 1e-3:
                beta_ok = False
            # (2) independent mean
            if abs(_beta.mean(a, b) - mean_norm[kk]) > 1e-9:
                beta_ok = False
            # (3) peak == analytic mode mapped to original units (NaN <-> NaN)
            mode_norm = _beta_mode(a, b)
            if np.isnan(mode_norm):
                if not np.isnan(peak[kk]):
                    beta_ok = False
            else:
                expected = tmin + mode_norm * (tmax - tmin)
                if abs(expected - peak[kk]) > 1e-6:
                    beta_ok = False
        print(f"[{NAME}] scipy Beta pdf/mean/peak match = {beta_ok}")
    except ImportError:
        print(f"[{NAME}] scipy absent; numerical Beta cross-check SKIPPED (not counted).")

    # Bars: block recovery cosine >= 0.95, temporal ordering exact, center corr >= 0.95,
    # independent numpy-MoM agreement, and (when scipy is present) the scipy Beta numerics.
    ok = cos >= 0.95 and order_ok and center_corr >= 0.95 and mom_ok and (beta_ok is not False)
    print(f"[{NAME}] PARITY {'OK' if ok else 'CHECK'}")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        compare()
