"""Committed-gold parity for topica SITS vs the reference Java sampler (issue #906).

The reference is the parametric SITS Gibbs sampler (``AuthorShiftSampler``) of
Rossiter's fork of Nguyen's code, ``github.com/erossiter/sits`` (Apache-2.0), the
implementation behind Rossiter (2022, AJPS). topica has two modes, and each has its
own Java reference:

  * ``compat="rossiter2022"`` against the fork **as published**. The fork starts
    every non-first turn as a shift with probability 1/I, including turns too short
    to be sampled; those short turns are later recorded as non-shifts but stay
    segment boundaries and stay counted as shifts for the whole chain.
  * the default mode against the fork with a **one-line fix**: short turns start
    (and so stay) as non-shifts. Everything else is the fork's code.

Chains of both implementations use different RNGs, so agreement is statistical.
The metrics are the ones that tell a working sampler from a broken one. Rossiter's
per-speaker score is dominated by conversation openers (always shifts) and short
turns (never shifts), so it can agree with the reference largely because the forced
turns agree. We therefore compare only *eligible* turns:

  1. each speaker's eligible-turn shift rate (posterior mean per chain), as a
     two-sample z against the pooled chain-to-chain spread of both engines;
  2. the overall eligible shift rate, same test;
  3. per-turn posterior shift probability on eligible turns: correlation of a
     three-chain topica average with a three-chain Java average, against the Java
     half-vs-half floor (chains 0-2 vs 3-5, also three vs three) minus a margin;
  4. topic-word agreement after Hungarian alignment, against the Java seed-to-seed
     floor minus a margin.

The corpus is simulated from the SITS generative process (known per-speaker shift
probabilities, ~30% short turns), so the planted truth is also checked.

Two phases (the pattern of parity/dmr_gold.py):

  * ``--regenerate`` (needs ``git``, ``javac``/``java`` and network access): clones
    the fork, compiles it twice (as published and with the fix), runs NUM_CHAINS
    reseeded chains of each through ``parity/SITSDriver.java``, and writes
    ``parity/sits_gold.npz`` + ``.json``.
  * default (no Java): loads the gold, fits NUM_CHAINS topica chains in each mode on
    the same corpus, and checks the bars.

Run directly::

    python parity/sits_compare.py               # offline compare against committed gold
    python parity/sits_compare.py --regenerate  # rebuild the Java reference gold
"""

from __future__ import annotations

import datetime
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

import harness

NAME = "sits"
FORK_URL = "https://github.com/erossiter/sits.git"
FORK_COMMIT = "067e9e6"
SAMPLER = "src/segmentation/parametric/sampler/AuthorShiftSampler.java"
# The one-line fix: a short turn never draws an initial shift.
FIX_OLD = "int rand_l = rand.nextInt(I); //Returns number between [0,I), so P(l=1) = 1/I"
FIX_NEW = "int rand_l = (words[t].length >= 5) ? rand.nextInt(I) : 0;"

# Simulation and sampler settings.
K = 6
V = 120
NUM_SPEAKERS = 6
PI = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 0.70])
NUM_CONVS = 40
TURNS_PER_CONV = 30
SHORT_SHARE = 0.3
ALPHA = 0.3
BETA = 0.1
GAMMA = 1.0
I_INIT = 3
ITERS = 6000
BURN_IN = 3000
NUM_CHAINS = 6

Z_BAR = 3.0          # per-speaker / overall eligible-rate two-sample z
CORR_MARGIN = 0.05   # per-turn shift-probability correlation vs Java half-vs-half
COS_MARGIN = 0.05    # aligned topic-word cosine vs Java seed-to-seed


def simulate(seed: int = 0):
    """Draw a corpus from the SITS generative process (short turns never shift)."""
    rng = np.random.default_rng(seed)
    phi = rng.dirichlet(np.full(V, 0.05), size=K)
    turns, speakers, convs, true_l = [], [], [], []
    for c in range(NUM_CONVS):
        prev = -1
        theta = None
        for t in range(TURNS_PER_CONV):
            m = int(rng.integers(NUM_SPEAKERS))
            while m == prev:
                m = int(rng.integers(NUM_SPEAKERS))
            prev = m
            short = rng.random() < SHORT_SHARE
            n = int(rng.integers(1, 5)) if short else int(rng.integers(6, 16))
            shift = t == 0 or (not short and rng.random() < PI[m])
            if shift:
                theta = rng.dirichlet(np.full(K, ALPHA))
            z = rng.choice(K, size=n, p=theta)
            words = [int(rng.choice(V, p=phi[k])) for k in z]
            turns.append(words)
            speakers.append(m)
            convs.append(c)
            true_l.append(int(shift))
    return turns, np.array(speakers), np.array(convs), np.array(true_l), phi


def _masks(turns, convs):
    first = np.r_[True, convs[1:] != convs[:-1]]
    eligible = ~first & (np.array([len(t) for t in turns]) >= 5)
    return first, eligible


def _write_sits_format(d, turns, speakers, convs):
    words = [f"{len(set(convs))}", f"{len(turns) + len(set(convs))}"]
    authors = []
    for t, (w, m) in enumerate(zip(turns, speakers)):
        if t > 0 and convs[t] != convs[t - 1]:
            words.append("")
            authors.append("-1")
        words.append(f"{len(w)}\t{' '.join(map(str, w))}")
        authors.append(str(m))
    words.append("")
    authors.append("-1")
    open(os.path.join(d, "c.words"), "w").write("\n".join(words) + "\n")
    open(os.path.join(d, "c.authors"), "w").write("\n".join(authors) + "\n")


def _build_reference(work):
    """Clone the fork at FORK_COMMIT and compile it as published and fixed."""
    src = os.path.join(work, "sits")
    subprocess.run(["git", "clone", "-q", FORK_URL, src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "-q", FORK_COMMIT], check=True)
    lib = os.path.join(src, "lib", "*")
    javas = glob.glob(os.path.join(src, "src", "**", "*.java"), recursive=True)
    published = os.path.join(work, "classes_published")
    os.makedirs(published)
    subprocess.run(["javac", "-nowarn", "-cp", lib, "-d", published, *javas],
                   check=True, capture_output=True)
    fixed_src = os.path.join(work, "fixed", "AuthorShiftSampler.java")
    os.makedirs(os.path.dirname(fixed_src))
    code = open(os.path.join(src, SAMPLER)).read()
    assert code.count(FIX_OLD) == 1, "fork source changed; update FIX_OLD"
    open(fixed_src, "w").write(code.replace(FIX_OLD, FIX_NEW))
    fixed = os.path.join(work, "classes_fixed")
    os.makedirs(fixed)
    subprocess.run(["javac", "-nowarn", "-cp", f"{published}:{lib}", "-d", fixed, fixed_src],
                   check=True, capture_output=True)
    driver = os.path.join(work, "driver")
    os.makedirs(driver)
    subprocess.run(["javac", "-nowarn", "-cp", f"{published}:{lib}", "-d", driver,
                    str(harness.HERE / "SITSDriver.java")], check=True, capture_output=True)
    return {"published": f"{driver}:{published}:{lib}",
            "fixed": f"{driver}:{fixed}:{published}:{lib}"}


def _run_java(cp, work, seed, tag):
    out = os.path.join(work, f"run_{tag}_{seed}")
    args = ["java", "-cp", cp, "SITSDriver", os.path.join(work, "c.words"),
            os.path.join(work, "c.authors"), str(V), str(NUM_SPEAKERS), str(K),
            str(ALPHA), str(BETA), str(GAMMA), str(BURN_IN), str(ITERS), str(I_INIT),
            str(seed), out]
    subprocess.run(args, check=True, capture_output=True)
    folder, secs = open(os.path.join(out, "run.txt")).read().split()
    L = np.loadtxt(os.path.join(folder, "all_sampled_shift_asgn.txt"), dtype=np.int8)
    L = L[:, L[0] != -1]  # drop the conversation separators
    phi = np.loadtxt(os.path.join(folder, "phi.txt"))
    return L.mean(axis=0), phi, float(secs)


def _topica_chain(turns, speakers, convs, compat, seed):
    import topica

    toks = [[f"w{w}" for w in t] for t in turns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        m = topica.SITS(K, alpha=ALPHA, beta=BETA, gamma=GAMMA, min_shift_tokens=5,
                        init="random", init_shift_rate=1 / I_INIT, compat=compat,
                        seed=seed).fit(
            toks, speakers.tolist(), conversations=convs.tolist(),
            iters=ITERS, burn_in=BURN_IN)
        secs = time.perf_counter() - t0
    # topic_word columns in the reference's word-id order
    col = {w: i for i, w in enumerate(m.vocabulary)}
    phi = np.zeros((K, V))
    for w in range(V):
        if f"w{w}" in col:
            phi[:, w] = m.topic_word[:, col[f"w{w}"]]
    return np.asarray(m.shift_prob), phi, secs


def _speaker_rates(shift_prob, speakers, eligible):
    return np.array([shift_prob[eligible & (speakers == j)].mean() for j in range(NUM_SPEAKERS)])


def _aligned_cosine(a, b):
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = an @ bn.T
    r, c = linear_sum_assignment(-sim)
    return float(sim[r, c].mean())


def regenerate() -> None:
    for tool in ("git", "javac", "java"):
        if shutil.which(tool) is None:
            print(f"{tool} not available; cannot regenerate.")
            sys.exit(1)
    turns, speakers, convs, true_l, true_phi = simulate()
    work = tempfile.mkdtemp()
    try:
        cps = _build_reference(work)
        _write_sits_format(work, turns, speakers, convs)
        arrays, meta_secs = {}, {}
        for tag in ("published", "fixed"):
            probs, phis, secs = [], [], []
            for s in range(NUM_CHAINS):
                p, f, sec = _run_java(cps[tag], work, 101 + s, tag)
                probs.append(p)
                phis.append(f)
                secs.append(sec)
            arrays[f"{tag}_shift_prob"] = np.array(probs)
            arrays[f"{tag}_phi"] = np.array(phis)
            meta_secs[tag] = float(np.mean(secs))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    arrays["turns"] = np.array(harness.docs_to_lines([[str(w) for w in t] for t in turns]),
                               dtype=object)
    arrays["speakers"] = speakers
    arrays["convs"] = convs
    arrays["true_l"] = true_l
    arrays["true_phi"] = true_phi
    harness.save_gold(NAME, arrays=arrays, meta={
        "reference": f"erossiter/sits @ {FORK_COMMIT} (AuthorShiftSampler), via parity/SITSDriver.java",
        "reference_fixed": f"same, with the one-line init fix: {FIX_NEW}",
        "model": "parametric SITS",
        "corpus": (f"simulated from the SITS generative process: {NUM_CONVS} conversations x "
                   f"{TURNS_PER_CONV} turns, {NUM_SPEAKERS} speakers with pi={PI.tolist()}, "
                   f"{SHORT_SHARE:.0%} short turns (never shift), K={K}, V={V}"),
        "settings": {"K": K, "alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "I": I_INIT,
                     "iters": ITERS, "burn_in": BURN_IN, "min_shift_tokens": 5},
        "chains": NUM_CHAINS,
        "java_seconds_per_chain": meta_secs,
        "date": datetime.date.today().isoformat(),
        "pass_bar": (f"per-speaker and overall eligible shift rate: |z| <= {Z_BAR} against the "
                     "pooled chain spread; per-turn eligible shift-prob correlation >= Java "
                     f"half-vs-half - {CORR_MARGIN}; aligned phi cosine >= Java seed-to-seed "
                     f"- {COS_MARGIN}. topica default vs the fixed fork, compat vs the "
                     "published fork."),
        "kind": "cross-implementation (Java, Rossiter's fork of Nguyen's SITS)",
    })
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name}")
    run(verbose=True)


def _z(a, b):
    """Two-sample z per column. With zero spread on both sides the means must agree
    exactly: identical means give 0, any difference gives inf (a failing gate)."""
    se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    diff = a.mean(axis=0) - b.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, diff / se, np.where(diff == 0, 0.0, np.inf))


def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    turns = [[int(w) for w in t] for t in harness.lines_to_docs(str(arrays["turns"]))]
    # lines_to_docs drops empty lines; the simulator never makes empty turns
    speakers = arrays["speakers"]
    convs = arrays["convs"]
    true_l = arrays["true_l"]
    _, eligible = _masks(turns, convs)
    result = {"passes": True}
    for mode, tag, compat in [("default", "fixed", None),
                              ("compat", "published", "rossiter2022")]:
        java_p = arrays[f"{tag}_shift_prob"]
        java_phi = arrays[f"{tag}_phi"]
        chains = [_topica_chain(turns, speakers, convs, compat, 1 + s)
                  for s in range(java_p.shape[0])]
        top_p = np.array([c[0] for c in chains])
        top_phi = [c[1] for c in chains]
        secs = float(np.mean([c[2] for c in chains]))

        jr = np.array([_speaker_rates(p, speakers, eligible) for p in java_p])
        tr = np.array([_speaker_rates(p, speakers, eligible) for p in top_p])
        z_spk = _z(tr, jr)
        jo = java_p[:, eligible].mean(axis=1)[:, None]
        to = top_p[:, eligible].mean(axis=1)[:, None]
        z_all = float(_z(to, jo)[0])
        half = java_p.shape[0] // 2
        floor_corr = float(np.corrcoef(java_p[:half, eligible].mean(0),
                                       java_p[half:, eligible].mean(0))[0, 1])
        # like-for-like with the floor: three topica chains vs three Java chains
        corr = float(np.mean([
            np.corrcoef(top_p[a, eligible].mean(0), java_p[b, eligible].mean(0))[0, 1]
            for a in (slice(None, half), slice(half, None))
            for b in (slice(None, half), slice(half, None))]))
        floor_cos = float(np.mean([_aligned_cosine(java_phi[i], java_phi[j])
                                   for i in range(len(java_phi))
                                   for j in range(i + 1, len(java_phi))]))
        cos = float(np.mean([_aligned_cosine(f, g) for f in top_phi for g in java_phi]))
        passes = bool(np.all(np.abs(z_spk) <= Z_BAR) and abs(z_all) <= Z_BAR
                      and corr >= floor_corr - CORR_MARGIN and cos >= floor_cos - COS_MARGIN)
        truth = _speaker_rates(true_l.astype(float), speakers, eligible)
        result[mode] = {
            "speaker_rate_topica": tr.mean(0).round(4).tolist(),
            "speaker_rate_java": jr.mean(0).round(4).tolist(),
            "speaker_rate_truth": truth.round(4).tolist(),
            "speaker_z_max": float(np.abs(z_spk).max()),
            "overall_rate_topica": float(to.mean()), "overall_rate_java": float(jo.mean()),
            "overall_z": z_all,
            "turn_corr": corr, "turn_corr_floor": floor_corr,
            "phi_cosine": cos, "phi_cosine_floor": floor_cos,
            "topica_seconds_per_chain": secs,
            "passes": passes,
        }
        result["passes"] &= passes
        if verbose:
            r = result[mode]
            print(f"[{mode}] vs Java fork ({tag}), {java_p.shape[0]} chains each")
            print(f"  eligible shift rate by speaker (truth {r['speaker_rate_truth']})")
            print(f"    topica {r['speaker_rate_topica']}")
            print(f"    java   {r['speaker_rate_java']}   max|z| {r['speaker_z_max']:.2f}")
            print(f"  overall eligible rate: topica {r['overall_rate_topica']:.4f}  "
                  f"java {r['overall_rate_java']:.4f}  z {z_all:+.2f}")
            print(f"  per-turn shift prob corr {corr:.4f} (java half-vs-half {floor_corr:.4f})")
            print(f"  aligned phi cosine {cos:.4f} (java seed-to-seed {floor_cos:.4f})")
            print(f"  seconds/chain: topica {secs:.2f}  java "
                  f"{meta.get('java_seconds_per_chain', {}).get(tag, float('nan')):.2f}")
            print(f"  verdict: {'PASS' if passes else 'FAIL'}")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        out = run(verbose=True)
        sys.exit(0 if out["passes"] else 1)
